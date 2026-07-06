//! Boolean operator.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Passed through from model A
//! - `get_io_ptr() -> i32`: Passed through from model A (whose memory is the
//!   exported one); B's buffer is obtained by calling B's own `get_io_ptr`
//! - `get_bounds(out_ptr: i32)`: Combines bounds from both models based on operation
//! - `sample(pos_ptr: i32) -> f32`: Combines densities from both models
//! - `memory`: First memory from merged modules
//!
//! Behavior:
//! - Reads WASM model A bytes from input 0
//! - Reads WASM model B bytes from input 1
//! - Reads CBOR configuration from input 2 (schema declared in metadata)
//! - Produces a merged WASM model with sample/bounds implementing union/subtract/intersect

use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, TypeSection,
    ValType,
};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BooleanOp {
    Union,
    Subtract,
    Intersect,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum BooleanOpConfig {
    Union,
    Subtract,
    Intersect,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct BooleanConfig {
    op: Option<BooleanOpConfig>,
}

impl Default for BooleanConfig {
    fn default() -> Self {
        Self {
            op: Some(BooleanOpConfig::Union),
        }
    }
}

impl From<BooleanOpConfig> for BooleanOp {
    fn from(value: BooleanOpConfig) -> Self {
        match value {
            BooleanOpConfig::Union => BooleanOp::Union,
            BooleanOpConfig::Subtract => BooleanOp::Subtract,
            BooleanOpConfig::Intersect => BooleanOp::Intersect,
        }
    }
}

use model_merge_core::{MergeSections, OffsetReencoder, count_sections, parse_model_exports};
use volumetric_abi::host::{post_output, read_input, report_error};

/// Add get_dimensions wrapper that passes through from model A
fn add_get_dimensions_wrapper(
    types: &mut TypeSection,
    funcs: &mut FunctionSection,
    code: &mut CodeSection,
    exports: &mut ExportSection,
    a_idx: u32,
) {
    let ty = types.len();
    types.ty().function([], [ValType::I32]);
    funcs.function(ty);

    let mut f = Function::new([]);
    f.instruction(&Instruction::Call(a_idx));
    f.instruction(&Instruction::End);
    code.function(&f);

    let func_index = (funcs.len() - 1) as u32;
    exports.export("get_dimensions", ExportKind::Func, func_index);
}

/// Add get_bounds wrapper that combines bounds from both models.
///
/// The merged module keeps both models' memories, so each model's get_bounds
/// writes into its *own* memory. A shares the module's exported memory, so it
/// writes straight to `out_ptr`; B writes into its own IO buffer (obtained by
/// calling B's `get_io_ptr`, index `b_io_idx`) in `b_mem_idx`. The combined
/// result is stored to `out_ptr` in A's memory.
#[allow(clippy::too_many_arguments)]
fn add_get_bounds_wrapper(
    types: &mut TypeSection,
    funcs: &mut FunctionSection,
    code: &mut CodeSection,
    exports: &mut ExportSection,
    a_idx: u32,
    b_idx: u32,
    b_io_idx: u32,
    op: BooleanOp,
    a_mem_idx: u32,
    b_mem_idx: u32,
) {
    let ty = types.len();
    types.ty().function([ValType::I32], []);
    funcs.function(ty);

    // Locals: out_ptr (param 0), A's bounds (1-6), B's bounds (7-12), b_io (13)
    let locals = vec![
        (6, ValType::F64), // a_min_x, a_max_x, a_min_y, a_max_y, a_min_z, a_max_z
        (6, ValType::F64), // b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z
        (1, ValType::I32), // B's IO buffer pointer
    ];
    let mut f = Function::new(locals);

    // Call A's get_bounds directly into out_ptr (A's memory is exported)
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::Call(a_idx));

    if op != BooleanOp::Subtract {
        // For subtract, A's bounds at out_ptr are already the answer.
        // Otherwise load A's bounds into locals before overwriting out_ptr.
        for i in 0..6 {
            f.instruction(&Instruction::LocalGet(0));
            f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
                offset: (i * 8) as u64,
                align: 3,
                memory_index: a_mem_idx,
            }));
            f.instruction(&Instruction::LocalSet(1 + i)); // locals 1-6 are A's bounds
        }

        // b_io = B's get_io_ptr()
        f.instruction(&Instruction::Call(b_io_idx));
        f.instruction(&Instruction::LocalSet(13));

        // Call B's get_bounds into B's own IO buffer
        f.instruction(&Instruction::LocalGet(13));
        f.instruction(&Instruction::Call(b_idx));

        // Load B's bounds (B wrote into its own memory)
        for i in 0..6 {
            f.instruction(&Instruction::LocalGet(13));
            f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
                offset: (i * 8) as u64,
                align: 3,
                memory_index: b_mem_idx,
            }));
            f.instruction(&Instruction::LocalSet(7 + i)); // locals 7-12 are B's bounds
        }

        // Combine bounds based on operation
        // Format: min_x(0), max_x(1), min_y(2), max_y(3), min_z(4), max_z(5)
        for i in 0..6 {
            let is_min = i % 2 == 0;
            f.instruction(&Instruction::LocalGet(0)); // out_ptr
            f.instruction(&Instruction::LocalGet(1 + i)); // A's bound
            f.instruction(&Instruction::LocalGet(7 + i)); // B's bound

            match (op, is_min) {
                (BooleanOp::Union, true) => f.instruction(&Instruction::F64Min), // min of mins
                (BooleanOp::Union, false) => f.instruction(&Instruction::F64Max), // max of maxs
                (BooleanOp::Intersect, true) => f.instruction(&Instruction::F64Max), // max of mins
                (BooleanOp::Intersect, false) => f.instruction(&Instruction::F64Min), // min of maxs
                (BooleanOp::Subtract, _) => unreachable!(),
            };

            f.instruction(&Instruction::F64Store(wasm_encoder::MemArg {
                offset: (i * 8) as u64,
                align: 3,
                memory_index: a_mem_idx,
            }));
        }
    }

    f.instruction(&Instruction::End);
    code.function(&f);

    let func_index = (funcs.len() - 1) as u32;
    exports.export("get_bounds", ExportKind::Func, func_index);
}

/// Add sample wrapper that combines densities from both models.
///
/// The caller writes the position into the module's exported memory (A's), so
/// A's sample can read `pos_ptr` directly. B's code reads its *own* memory,
/// so the wrapper copies `dims * 8` bytes from A's memory at `pos_ptr` into
/// B's IO buffer (obtained by calling B's `get_io_ptr`, index `b_io_idx`),
/// then calls B with that pointer. The copy happens before calling A because
/// the ABI allows a model's sample to clobber its position buffer.
#[allow(clippy::too_many_arguments)]
fn add_sample_wrapper(
    types: &mut TypeSection,
    funcs: &mut FunctionSection,
    code: &mut CodeSection,
    exports: &mut ExportSection,
    a_idx: u32,
    b_idx: u32,
    b_io_idx: u32,
    a_dims_idx: u32,
    op: BooleanOp,
    a_mem_idx: u32,
    b_mem_idx: u32,
) {
    let ty = types.len();
    types.ty().function([ValType::I32], [ValType::F32]);
    funcs.function(ty);

    // Locals: pos_ptr (param 0), dims_bytes (1), i (2), b_io (3)
    let mut f = Function::new([(3, ValType::I32)]);

    // dims_bytes = get_dimensions() * 8
    f.instruction(&Instruction::Call(a_dims_idx));
    f.instruction(&Instruction::I32Const(3));
    f.instruction(&Instruction::I32Shl);
    f.instruction(&Instruction::LocalSet(1));

    // b_io = B's get_io_ptr()
    f.instruction(&Instruction::Call(b_io_idx));
    f.instruction(&Instruction::LocalSet(3));

    // for (i = 0; i < dims_bytes; i += 8)
    //     B_mem[b_io + i] = A_mem[pos_ptr + i]
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::LocalSet(2));
    f.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::I32GeS);
    f.instruction(&Instruction::BrIf(1));
    f.instruction(&Instruction::LocalGet(3));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::I32Add); // dest address in B's memory
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::I32Add); // src address in A's memory
    f.instruction(&Instruction::F64Load(wasm_encoder::MemArg {
        offset: 0,
        align: 3,
        memory_index: a_mem_idx,
    }));
    f.instruction(&Instruction::F64Store(wasm_encoder::MemArg {
        offset: 0,
        align: 3,
        memory_index: b_mem_idx,
    }));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::I32Const(8));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalSet(2));
    f.instruction(&Instruction::Br(0));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);

    // Call A's sample with pos_ptr
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::Call(a_idx));
    // Classify with the shared occupancy contract (volumetric_abi:
    // OCCUPANCY_THRESHOLD): a_bool = (a_occupancy > 0.5). The generated model
    // emits canonical 1.0/0.0 and is occupancy-only — child sample formats
    // are dropped by construction (the export section is built explicitly).
    f.instruction(&Instruction::F32Const(0.5.into()));
    f.instruction(&Instruction::F32Gt);

    if op == BooleanOp::Subtract {
        // Call B's sample on the copied position
        f.instruction(&Instruction::LocalGet(3));
        f.instruction(&Instruction::Call(b_idx));
        f.instruction(&Instruction::F32Const(0.5.into()));
        f.instruction(&Instruction::F32Gt);
        // a && !b
        f.instruction(&Instruction::I32Eqz);
        f.instruction(&Instruction::I32And);
    } else if op == BooleanOp::Union {
        // Call B's sample on the copied position
        f.instruction(&Instruction::LocalGet(3));
        f.instruction(&Instruction::Call(b_idx));
        f.instruction(&Instruction::F32Const(0.5.into()));
        f.instruction(&Instruction::F32Gt);
        // a || b
        f.instruction(&Instruction::I32Or);
    } else {
        // Intersect: a && b
        f.instruction(&Instruction::LocalGet(3));
        f.instruction(&Instruction::Call(b_idx));
        f.instruction(&Instruction::F32Const(0.5.into()));
        f.instruction(&Instruction::F32Gt);
        f.instruction(&Instruction::I32And);
    }

    // Convert boolean i32 to f32 0.0/1.0
    f.instruction(&Instruction::F32ConvertI32S);
    f.instruction(&Instruction::End);
    code.function(&f);

    let func_index = (funcs.len() - 1) as u32;
    exports.export("sample", ExportKind::Func, func_index);
}

fn merge_models(a_wasm: &[u8], b_wasm: &[u8], op: BooleanOp) -> Result<Vec<u8>, String> {
    let a_counts = count_sections(a_wasm)?;
    let b_counts = count_sections(b_wasm)?;

    let a_exports = parse_model_exports(a_wasm)?;
    let b_exports = parse_model_exports(b_wasm)?;

    let mut sections = MergeSections::default();
    sections.append_module(a_wasm, &mut OffsetReencoder::identity())?;
    sections.append_module(b_wasm, &mut OffsetReencoder::after(&a_counts))?;

    // Build exports with wrappers.
    let mut exports = ExportSection::new();

    // Export memory from model A
    exports.export("memory", ExportKind::Memory, a_exports.memory);

    add_get_dimensions_wrapper(
        &mut sections.types,
        &mut sections.funcs,
        &mut sections.code,
        &mut exports,
        a_exports.get_dimensions,
    );

    // The merged model's IO buffer is A's: A's memory is the exported one, so
    // A's get_io_ptr already points where callers need to write positions.
    exports.export("get_io_ptr", ExportKind::Func, a_exports.get_io_ptr);

    add_get_bounds_wrapper(
        &mut sections.types,
        &mut sections.funcs,
        &mut sections.code,
        &mut exports,
        a_exports.get_bounds,
        b_exports.get_bounds + a_counts.funcs,
        b_exports.get_io_ptr + a_counts.funcs,
        op,
        a_exports.memory,
        b_exports.memory + a_counts.memories,
    );

    add_sample_wrapper(
        &mut sections.types,
        &mut sections.funcs,
        &mut sections.code,
        &mut exports,
        a_exports.sample,
        b_exports.sample + a_counts.funcs,
        b_exports.get_io_ptr + a_counts.funcs,
        a_exports.get_dimensions,
        op,
        a_exports.memory,
        b_exports.memory + a_counts.memories,
    );

    let data_count =
        (a_counts.has_data_count || b_counts.has_data_count).then(|| a_counts.data + b_counts.data);
    Ok(sections.finish(&exports, data_count))
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let a_buf = read_input(0);

    let b_buf = read_input(1);

    let cfg = {
        let cfg_buf = read_input(2);
        if cfg_buf.is_empty() {
            BooleanConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            ciborium::de::from_reader::<BooleanConfig, _>(&mut cursor).unwrap_or_default()
        }
    };
    let op = cfg.op.unwrap_or(BooleanOpConfig::Union).into();

    let output = match merge_models(&a_buf, &b_buf, op) {
        Ok(out) => out,
        Err(e) => {
            report_error(&format!("model merge failed: {e}"));
            return;
        }
    };
    post_output(0, &output);
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            "{ op: \"union\" / \"subtract\" / \"intersect\" .default \"union\" }".to_string();
        OperatorMetadata {
            name: "boolean_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec![
                "Model A".to_string(),
                "Model B".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
