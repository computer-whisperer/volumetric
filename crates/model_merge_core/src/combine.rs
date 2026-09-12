//! Combining models: the union, intersection and subtraction glue over
//! merged modules, shared by `boolean_operator` and the assembly
//! operators (which union parts posed by `model_wrap_core`).
//!
//! Generated Model ABI (N-dimensional), for models m_0 .. m_{k-1}:
//! - `get_dimensions() -> u32`: m_0's constant when it has one (so the
//!   wrappers that read dimensionality statically can follow a boolean),
//!   else passed through from m_0
//! - `get_io_ptr() -> i32`: passed through from m_0 (whose memory is the
//!   exported one); every other model's buffer is obtained by calling its
//!   own `get_io_ptr`
//! - `get_bounds(out_ptr: i32)`: m_0's bounds folded with every other
//!   model's per the operation, `2 * n` values at run time (no fixed
//!   dimension count in the glue)
//! - `sample(pos_ptr: i32) -> f32`: the combined occupancy, evaluating the
//!   models in order with an early exit
//! - `sample_channels` / `get_sample_format`: m_0's, with channel 0
//!   replaced by the combined occupancy, when m_0 declares a format
//! - `memory`: m_0's

use wasm_encoder::{BlockType, ExportKind, ExportSection, Function, Instruction, MemArg, ValType};

use crate::{
    MergeSections, ModelExports, OffsetReencoder, SectionCounts, const_i32_export, count_sections,
    parse_model_exports,
};

/// How the models fold together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combine {
    /// Inside any model.
    Union,
    /// Inside the first model and none of the rest.
    Subtract,
    /// Inside every model.
    Intersect,
}

/// One appended model's export indices, rebased into the merged module.
#[derive(Clone, Copy)]
struct Part {
    sample: u32,
    get_bounds: u32,
    get_io_ptr: u32,
    memory: u32,
}

impl Part {
    fn rebased(exports: &ModelExports, offsets: &SectionCounts) -> Self {
        Self {
            sample: exports.sample + offsets.funcs,
            get_bounds: exports.get_bounds + offsets.funcs,
            get_io_ptr: exports.get_io_ptr + offsets.funcs,
            memory: exports.memory + offsets.memories,
        }
    }
}

fn f64_at(memory: u32) -> MemArg {
    MemArg {
        offset: 0,
        align: 3,
        memory_index: memory,
    }
}

fn f32_at(memory: u32) -> MemArg {
    MemArg {
        offset: 0,
        align: 2,
        memory_index: memory,
    }
}

/// The glue a wrapper needs: the first model's exports (its memory is the
/// merged module's), and every other model to fold in.
struct Glue<'a> {
    first: &'a ModelExports,
    others: &'a [Part],
    op: Combine,
    /// The first model's dimensionality when its `get_dimensions` is a
    /// constant, so the merged module's is one too and the wrappers that
    /// read it statically (pattern, pose, extrude, ...) can follow a
    /// boolean.
    dims: Option<i32>,
}

impl Glue<'_> {
    fn first_memory(&self) -> u32 {
        self.first.memory
    }

    /// Emits `get_dimensions`: the first model's constant when it has one
    /// (so the result stays statically readable), else a call through.
    fn add_get_dimensions(&self, sections: &mut MergeSections, exports: &mut ExportSection) {
        let ty = sections.types.len();
        sections.types.ty().function([], [ValType::I32]);
        sections.funcs.function(ty);
        let mut f = Function::new([]);
        match self.dims {
            Some(dims) => f.instruction(&Instruction::I32Const(dims)),
            None => f.instruction(&Instruction::Call(self.first.get_dimensions)),
        };
        f.instruction(&Instruction::End);
        sections.code.function(&f);
        exports.export("get_dimensions", ExportKind::Func, sections.funcs.len() - 1);
    }

    /// Emits `get_bounds(out_ptr)`: the first model writes into `out_ptr`
    /// directly (its memory is the exported one). Subtraction keeps that
    /// box; otherwise every other model writes its own IO buffer and each
    /// of the `2 * n` values folds into `out_ptr` — minimum slots take the
    /// min (union) or max (intersect) of the pair, maximum slots the
    /// reverse. The count comes from `get_dimensions` at run time.
    fn add_get_bounds(&self, sections: &mut MergeSections, exports: &mut ExportSection) {
        let ty = sections.types.len();
        sections.types.ty().function([ValType::I32], []);
        sections.funcs.function(ty);

        // Locals: out_ptr (param 0), n_bytes (1), io (2), j (3), a (4), b (5)
        const OUT: u32 = 0;
        const N_BYTES: u32 = 1;
        const IO: u32 = 2;
        const J: u32 = 3;
        const A: u32 = 4;
        const B: u32 = 5;
        let mut f = Function::new([(3, ValType::I32), (2, ValType::F64)]);

        f.instruction(&Instruction::LocalGet(OUT));
        f.instruction(&Instruction::Call(self.first.get_bounds));

        if self.op != Combine::Subtract {
            // n_bytes = 2 * dims * 8
            f.instruction(&Instruction::Call(self.first.get_dimensions));
            f.instruction(&Instruction::I32Const(4));
            f.instruction(&Instruction::I32Shl);
            f.instruction(&Instruction::LocalSet(N_BYTES));

            for part in self.others {
                f.instruction(&Instruction::Call(part.get_io_ptr));
                f.instruction(&Instruction::LocalSet(IO));
                f.instruction(&Instruction::LocalGet(IO));
                f.instruction(&Instruction::Call(part.get_bounds));

                // for (j = 0; j < n_bytes; j += 8) out[j] = fold(out[j], io[j])
                f.instruction(&Instruction::I32Const(0));
                f.instruction(&Instruction::LocalSet(J));
                f.instruction(&Instruction::Block(BlockType::Empty));
                f.instruction(&Instruction::Loop(BlockType::Empty));
                f.instruction(&Instruction::LocalGet(J));
                f.instruction(&Instruction::LocalGet(N_BYTES));
                f.instruction(&Instruction::I32GeS);
                f.instruction(&Instruction::BrIf(1));

                // store address, then the two operands into locals
                f.instruction(&Instruction::LocalGet(OUT));
                f.instruction(&Instruction::LocalGet(J));
                f.instruction(&Instruction::I32Add);
                f.instruction(&Instruction::LocalGet(OUT));
                f.instruction(&Instruction::LocalGet(J));
                f.instruction(&Instruction::I32Add);
                f.instruction(&Instruction::F64Load(f64_at(self.first_memory())));
                f.instruction(&Instruction::LocalSet(A));
                f.instruction(&Instruction::LocalGet(IO));
                f.instruction(&Instruction::LocalGet(J));
                f.instruction(&Instruction::I32Add);
                f.instruction(&Instruction::F64Load(f64_at(part.memory)));
                f.instruction(&Instruction::LocalSet(B));

                // select(min-slot ? low : high): slot j/8 is a minimum when
                // even, i.e. when bit 3 of the byte offset is clear.
                let (low, high) = match self.op {
                    Combine::Union => (Instruction::F64Min, Instruction::F64Max),
                    Combine::Intersect => (Instruction::F64Max, Instruction::F64Min),
                    Combine::Subtract => unreachable!("subtract keeps the first box"),
                };
                f.instruction(&Instruction::LocalGet(A));
                f.instruction(&Instruction::LocalGet(B));
                f.instruction(&low);
                f.instruction(&Instruction::LocalGet(A));
                f.instruction(&Instruction::LocalGet(B));
                f.instruction(&high);
                f.instruction(&Instruction::LocalGet(J));
                f.instruction(&Instruction::I32Const(8));
                f.instruction(&Instruction::I32And);
                f.instruction(&Instruction::I32Eqz);
                f.instruction(&Instruction::Select);
                f.instruction(&Instruction::F64Store(f64_at(self.first_memory())));

                f.instruction(&Instruction::LocalGet(J));
                f.instruction(&Instruction::I32Const(8));
                f.instruction(&Instruction::I32Add);
                f.instruction(&Instruction::LocalSet(J));
                f.instruction(&Instruction::Br(0));
                f.instruction(&Instruction::End);
                f.instruction(&Instruction::End);
            }
        }

        f.instruction(&Instruction::End);
        sections.code.function(&f);
        exports.export("get_bounds", ExportKind::Func, sections.funcs.len() - 1);
    }

    /// Copies the position at `pos` (first model's memory, `dims_bytes`
    /// long) into every other model's IO buffer, recording each buffer in
    /// `io_locals`. Each model's code reads its own memory, and this runs
    /// before the first model samples because the ABI lets a model clobber
    /// its position buffer.
    fn emit_copy_positions(
        &self,
        f: &mut Function,
        pos: u32,
        dims_bytes: u32,
        i: u32,
        io_locals: &[u32],
    ) {
        for (part, &io) in self.others.iter().zip(io_locals) {
            f.instruction(&Instruction::Call(part.get_io_ptr));
            f.instruction(&Instruction::LocalSet(io));

            // for (i = 0; i < dims_bytes; i += 8) part_mem[io + i] = mem[pos + i]
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::LocalSet(i));
            f.instruction(&Instruction::Block(BlockType::Empty));
            f.instruction(&Instruction::Loop(BlockType::Empty));
            f.instruction(&Instruction::LocalGet(i));
            f.instruction(&Instruction::LocalGet(dims_bytes));
            f.instruction(&Instruction::I32GeS);
            f.instruction(&Instruction::BrIf(1));
            f.instruction(&Instruction::LocalGet(io));
            f.instruction(&Instruction::LocalGet(i));
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::LocalGet(pos));
            f.instruction(&Instruction::LocalGet(i));
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::F64Load(f64_at(self.first_memory())));
            f.instruction(&Instruction::F64Store(f64_at(part.memory)));
            f.instruction(&Instruction::LocalGet(i));
            f.instruction(&Instruction::I32Const(8));
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::LocalSet(i));
            f.instruction(&Instruction::Br(0));
            f.instruction(&Instruction::End);
            f.instruction(&Instruction::End);
        }
    }

    /// Folds every other model's occupancy into `acc` (an i32 boolean local
    /// already holding the first model's), stopping as soon as the answer
    /// is decided: a union at the first occupied model, an intersection at
    /// the first empty one, a subtraction at the first model that carves
    /// the point away. Classification uses the shared occupancy contract
    /// (`volumetric_abi`: OCCUPANCY_THRESHOLD).
    fn emit_combine_others(&self, f: &mut Function, acc: u32, io_locals: &[u32]) {
        f.instruction(&Instruction::Block(BlockType::Empty));
        match self.op {
            Combine::Union => {
                f.instruction(&Instruction::LocalGet(acc));
                f.instruction(&Instruction::BrIf(0));
                for (part, &io) in self.others.iter().zip(io_locals) {
                    f.instruction(&Instruction::LocalGet(io));
                    f.instruction(&Instruction::Call(part.sample));
                    f.instruction(&Instruction::F32Const(0.5.into()));
                    f.instruction(&Instruction::F32Gt);
                    f.instruction(&Instruction::LocalTee(acc));
                    f.instruction(&Instruction::BrIf(0));
                }
            }
            Combine::Intersect => {
                f.instruction(&Instruction::LocalGet(acc));
                f.instruction(&Instruction::I32Eqz);
                f.instruction(&Instruction::BrIf(0));
                for (part, &io) in self.others.iter().zip(io_locals) {
                    f.instruction(&Instruction::LocalGet(io));
                    f.instruction(&Instruction::Call(part.sample));
                    f.instruction(&Instruction::F32Const(0.5.into()));
                    f.instruction(&Instruction::F32Gt);
                    f.instruction(&Instruction::LocalTee(acc));
                    f.instruction(&Instruction::I32Eqz);
                    f.instruction(&Instruction::BrIf(0));
                }
            }
            Combine::Subtract => {
                f.instruction(&Instruction::LocalGet(acc));
                f.instruction(&Instruction::I32Eqz);
                f.instruction(&Instruction::BrIf(0));
                for (part, &io) in self.others.iter().zip(io_locals) {
                    f.instruction(&Instruction::LocalGet(io));
                    f.instruction(&Instruction::Call(part.sample));
                    f.instruction(&Instruction::F32Const(0.5.into()));
                    f.instruction(&Instruction::F32Gt);
                    f.instruction(&Instruction::If(BlockType::Empty));
                    f.instruction(&Instruction::I32Const(0));
                    f.instruction(&Instruction::LocalSet(acc));
                    f.instruction(&Instruction::Br(1));
                    f.instruction(&Instruction::End);
                }
            }
        }
        f.instruction(&Instruction::End);
    }

    /// Emits `sample(pos_ptr) -> f32`: positions are copied to every other
    /// model first, then the first model samples `pos_ptr` in place and
    /// the rest fold in with an early exit.
    fn add_sample(&self, sections: &mut MergeSections, exports: &mut ExportSection) {
        let ty = sections.types.len();
        sections.types.ty().function([ValType::I32], [ValType::F32]);
        sections.funcs.function(ty);

        // Locals: pos_ptr (param 0), dims_bytes (1), i (2), acc (3), io… (4..)
        const POS: u32 = 0;
        const DIMS_BYTES: u32 = 1;
        const I: u32 = 2;
        const ACC: u32 = 3;
        let io_locals: Vec<u32> = (0..self.others.len() as u32).map(|k| 4 + k).collect();
        let mut f = Function::new([(3 + self.others.len() as u32, ValType::I32)]);

        f.instruction(&Instruction::Call(self.first.get_dimensions));
        f.instruction(&Instruction::I32Const(3));
        f.instruction(&Instruction::I32Shl);
        f.instruction(&Instruction::LocalSet(DIMS_BYTES));
        self.emit_copy_positions(&mut f, POS, DIMS_BYTES, I, &io_locals);

        f.instruction(&Instruction::LocalGet(POS));
        f.instruction(&Instruction::Call(self.first.sample));
        f.instruction(&Instruction::F32Const(0.5.into()));
        f.instruction(&Instruction::F32Gt);
        f.instruction(&Instruction::LocalSet(ACC));
        self.emit_combine_others(&mut f, ACC, &io_locals);

        f.instruction(&Instruction::LocalGet(ACC));
        f.instruction(&Instruction::F32ConvertI32S);
        f.instruction(&Instruction::End);
        sections.code.function(&f);
        exports.export("sample", ExportKind::Func, sections.funcs.len() - 1);
    }

    /// Emits `sample_channels(pos_ptr, out_ptr)`: the first model's full
    /// channel row with channel 0 replaced by the combined occupancy. As in
    /// `sample`, positions are copied out before the first model runs.
    fn add_sample_channels(
        &self,
        sections: &mut MergeSections,
        exports: &mut ExportSection,
        first_channels: u32,
    ) {
        let ty = sections.types.len();
        sections
            .types
            .ty()
            .function([ValType::I32, ValType::I32], []);
        sections.funcs.function(ty);

        // Locals: pos_ptr (0), out_ptr (1), dims_bytes (2), i (3), acc (4), io… (5..)
        const POS: u32 = 0;
        const OUT: u32 = 1;
        const DIMS_BYTES: u32 = 2;
        const I: u32 = 3;
        const ACC: u32 = 4;
        let io_locals: Vec<u32> = (0..self.others.len() as u32).map(|k| 5 + k).collect();
        let mut f = Function::new([(3 + self.others.len() as u32, ValType::I32)]);

        f.instruction(&Instruction::Call(self.first.get_dimensions));
        f.instruction(&Instruction::I32Const(3));
        f.instruction(&Instruction::I32Shl);
        f.instruction(&Instruction::LocalSet(DIMS_BYTES));
        self.emit_copy_positions(&mut f, POS, DIMS_BYTES, I, &io_locals);

        f.instruction(&Instruction::LocalGet(POS));
        f.instruction(&Instruction::LocalGet(OUT));
        f.instruction(&Instruction::Call(first_channels));

        let out_mem = f32_at(self.first_memory());
        f.instruction(&Instruction::LocalGet(OUT));
        f.instruction(&Instruction::F32Load(out_mem));
        f.instruction(&Instruction::F32Const(0.5.into()));
        f.instruction(&Instruction::F32Gt);
        f.instruction(&Instruction::LocalSet(ACC));
        self.emit_combine_others(&mut f, ACC, &io_locals);

        f.instruction(&Instruction::LocalGet(OUT));
        f.instruction(&Instruction::LocalGet(ACC));
        f.instruction(&Instruction::F32ConvertI32S);
        f.instruction(&Instruction::F32Store(out_mem));
        f.instruction(&Instruction::End);
        sections.code.function(&f);
        exports.export(
            "sample_channels",
            ExportKind::Func,
            sections.funcs.len() - 1,
        );
    }
}

/// Merges `models` (at least one) into one module whose exports implement
/// `op` over all of them.
pub fn combine_models(models: &[Vec<u8>], op: Combine) -> Result<Vec<u8>, String> {
    let mut sections = MergeSections::default();
    let mut offsets = SectionCounts::default();
    let mut first = None;
    let mut others = Vec::new();
    for (index, wasm) in models.iter().enumerate() {
        let counts = count_sections(wasm).map_err(|e| format!("model {index}: {e}"))?;
        let exports = parse_model_exports(wasm).map_err(|e| format!("model {index}: {e}"))?;
        sections
            .append_module(wasm, &mut OffsetReencoder::after(&offsets))
            .map_err(|e| format!("model {index}: {e}"))?;
        if index == 0 {
            first = Some(exports);
        } else {
            others.push(Part::rebased(&exports, &offsets));
        }
        offsets.extend(&counts);
    }
    let first = first.ok_or_else(|| "no models to merge".to_string())?;
    let dims =
        const_i32_export(&models[0], "get_dimensions").map_err(|e| format!("model 0: {e}"))?;
    let glue = Glue {
        first: &first,
        others: &others,
        op,
        dims,
    };

    let mut exports = ExportSection::new();
    // The merged model's memory and IO buffer are the first model's: its
    // get_io_ptr already points where callers write positions.
    exports.export("memory", ExportKind::Memory, first.memory);
    exports.export("get_io_ptr", ExportKind::Func, first.get_io_ptr);
    glue.add_get_dimensions(&mut sections, &mut exports);
    glue.add_get_bounds(&mut sections, &mut exports);
    glue.add_sample(&mut sections, &mut exports);
    if let (Some(get_sample_format), Some(sample_channels)) =
        (first.get_sample_format, first.sample_channels)
    {
        exports.export("get_sample_format", ExportKind::Func, get_sample_format);
        glue.add_sample_channels(&mut sections, &mut exports, sample_channels);
    }

    let data_count = offsets.has_data_count.then_some(offsets.data);
    Ok(sections.finish(&exports, data_count))
}
