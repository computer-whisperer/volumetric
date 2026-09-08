//! Wrapper glue for model-to-model operators that keep the input model's
//! memory, dimensionality and sample format and only rewrite the query
//! point and the bounds: translate, rotation, scale, pattern.
//!
//! [`Wrapper::parse`] takes the input model bytes, finds its memory export
//! and its `get_dimensions` constant, and un-exports the three ABI functions
//! a wrapper replaces (`sample`, `sample_channels`, `get_bounds`), keeping
//! them as ordinary functions for the replacements to call. [`Wrapper::wrap`]
//! emits a replacement with the ABI signature and exports it;
//! [`Wrapper::wrap_positions`] and [`Wrapper::wrap_bounds`] cover the common
//! rewrite-then-delegate shape and [`Wrapper::apply_affine`] the whole
//! single-transform operator. `get_dimensions`, `get_io_ptr`,
//! `get_sample_format` and `memory` pass through untouched.
//!
//! Codegen helpers live in [`emit`]; the transforms they apply are
//! [`Affine`] maps over the spatial prefix min(dims, 3) — a 2D sketch
//! transforms in-plane, higher dimensions pass through. Whether a given map
//! is valid for the input's dimensionality (a rotation out of a sketch's
//! plane is not) is the operator's check, made before emitting; see
//! [`Affine::preserves_prefix`].

pub mod affine;
pub mod emit;

pub use affine::Affine;

use std::collections::HashMap;

use walrus::{
    FunctionBuilder, FunctionId, InstrSeqBuilder, LocalId, MemoryId, Module, ModuleConfig,
    ModuleLocals, ValType,
};

/// The ABI functions a wrapper replaces, with their signatures.
const WRAPPED: &[(&str, &[ValType], &[ValType])] = &[
    ("sample", &[ValType::I32], &[ValType::F32]),
    ("sample_channels", &[ValType::I32, ValType::I32], &[]),
    ("get_bounds", &[ValType::I32], &[]),
];

/// Read the constant a trivial `() -> i32` function returns, if its body is
/// a single `i32.const`. Every model generator emits `get_dimensions` this
/// way, so this is how an operator learns the input's dimensionality without
/// being able to instantiate it.
pub fn const_i32_return(module: &Module, func_id: FunctionId) -> Option<i32> {
    let walrus::FunctionKind::Local(local) = &module.funcs.get(func_id).kind else {
        return None;
    };
    let block = local.block(local.entry_block());
    match block.instrs.as_slice() {
        [(walrus::ir::Instr::Const(c), _)] => match c.value {
            walrus::ir::Value::I32(v) => Some(v),
            _ => None,
        },
        _ => None,
    }
}

fn exported_function(module: &Module, name: &str) -> Option<FunctionId> {
    module.exports.iter().find_map(|e| match e.item {
        walrus::ExportItem::Function(f) if e.name == name => Some(f),
        _ => None,
    })
}

/// An input model being wrapped.
pub struct Wrapper {
    pub module: Module,
    /// The model's exported linear memory: positions and bounds live here.
    pub memory: MemoryId,
    /// The model's dimensionality (its `get_dimensions` constant).
    pub dims: usize,
    originals: HashMap<&'static str, FunctionId>,
}

/// What a wrapper body has to work with: the builder of the replacement,
/// the module's locals, its memory, the replacement's parameters in ABI
/// order, and the original function to delegate to.
pub struct WrapBody<'a> {
    pub builder: &'a mut FunctionBuilder,
    pub locals: &'a mut ModuleLocals,
    pub memory: MemoryId,
    pub args: &'a [LocalId],
    pub original: FunctionId,
}

impl Wrapper {
    /// Parse a model and take over its wrapped exports. Errors name what the
    /// input lacks; `sample_channels` is optional (present iff the model
    /// declares typed channels), the rest are required.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut module = Module::from_buffer_with_config(bytes, &ModuleConfig::new())
            .map_err(|e| format!("failed to parse model wasm: {e}"))?;
        let memory = module
            .exports
            .iter()
            .find_map(|e| match e.item {
                walrus::ExportItem::Memory(m) if e.name == "memory" => Some(m),
                _ => None,
            })
            .ok_or("input model missing `memory` export")?;
        if exported_function(&module, "get_io_ptr").is_none() {
            return Err(
                "input model missing `get_io_ptr` export; rebuild it against the \
                        current N-dimensional ABI"
                    .to_string(),
            );
        }
        let dims_fn = exported_function(&module, "get_dimensions")
            .ok_or("input model missing `get_dimensions` export")?;
        let dims = const_i32_return(&module, dims_fn).ok_or(
            "cannot determine input model dimensionality (get_dimensions is not a constant \
             function)",
        )?;
        if dims < 1 {
            return Err(format!("input model reports invalid dimensionality {dims}"));
        }

        let mut originals = HashMap::new();
        for (name, _, _) in WRAPPED {
            let Some(export_id) = module
                .exports
                .iter()
                .find(|e| e.name == *name)
                .map(|e| e.id())
            else {
                continue;
            };
            let walrus::ExportItem::Function(func) = module.exports.get(export_id).item else {
                return Err(format!("input model exports `{name}` as a non-function"));
            };
            module.exports.delete(export_id);
            module.funcs.get_mut(func).name = Some(format!("{name}_inner"));
            originals.insert(*name, func);
        }
        for required in ["sample", "get_bounds"] {
            if !originals.contains_key(required) {
                return Err(format!("input model missing `{required}` export"));
            }
        }
        Ok(Wrapper {
            module,
            memory,
            dims: dims as usize,
            originals,
        })
    }

    /// The spatial prefix a transform touches: min(dims, 3).
    pub fn spatial(&self) -> usize {
        self.dims.min(3)
    }

    /// The input's own implementation of a wrapped ABI function.
    pub fn original(&self, name: &str) -> Option<FunctionId> {
        self.originals.get(name).copied()
    }

    /// Emit and export a replacement for `name` (`sample`, `sample_channels`
    /// or `get_bounds`), built by `body`. Returns false, emitting nothing,
    /// when the input has no such function.
    pub fn wrap(&mut self, name: &str, body: impl FnOnce(&mut WrapBody<'_>)) -> bool {
        let Some(&original) = self.originals.get(name) else {
            return false;
        };
        let (_, params, results) = WRAPPED
            .iter()
            .find(|(n, _, _)| *n == name)
            .unwrap_or_else(|| panic!("`{name}` is not a wrapped ABI function"));
        let mut builder = FunctionBuilder::new(&mut self.module.types, params, results);
        let args: Vec<LocalId> = params
            .iter()
            .map(|ty| self.module.locals.add(*ty))
            .collect();
        body(&mut WrapBody {
            builder: &mut builder,
            locals: &mut self.module.locals,
            memory: self.memory,
            args: &args,
            original,
        });
        let id = builder.finish(args, &mut self.module.funcs);
        self.module.exports.add(name, id);
        true
    }

    /// Replace `sample` and `sample_channels` (when present) with wrappers
    /// that run `rewrite` on the position buffer in place, then delegate
    /// with the same pointers.
    pub fn wrap_positions(
        &mut self,
        rewrite: impl Fn(&mut InstrSeqBuilder, &mut ModuleLocals, MemoryId, LocalId),
    ) {
        self.wrap("sample", |body| {
            let pos = body.args[0];
            let (memory, original) = (body.memory, body.original);
            let mut seq = body.builder.func_body();
            rewrite(&mut seq, body.locals, memory, pos);
            seq.local_get(pos).call(original);
        });
        self.wrap("sample_channels", |body| {
            let (pos, out) = (body.args[0], body.args[1]);
            let (memory, original) = (body.memory, body.original);
            let mut seq = body.builder.func_body();
            rewrite(&mut seq, body.locals, memory, pos);
            seq.local_get(pos).local_get(out).call(original);
        });
    }

    /// Replace `get_bounds` with a wrapper that calls the original into the
    /// output buffer, then runs `rewrite` on that buffer in place.
    pub fn wrap_bounds(
        &mut self,
        rewrite: impl FnOnce(&mut InstrSeqBuilder, &mut ModuleLocals, MemoryId, LocalId),
    ) {
        self.wrap("get_bounds", |body| {
            let out = body.args[0];
            let (memory, original) = (body.memory, body.original);
            let mut seq = body.builder.func_body();
            seq.local_get(out).call(original);
            rewrite(&mut seq, body.locals, memory, out);
        });
    }

    /// Apply one affine map to the model over its spatial prefix: query
    /// points get the inverse, bounds the map itself. Errors when the map
    /// is singular on that prefix.
    pub fn apply_affine(&mut self, map: &Affine) -> Result<(), String> {
        let n = self.spatial();
        let map = map.restricted(n);
        let inverse = map
            .inverse()
            .ok_or("transform is singular (a zero scale factor?)")?;
        self.wrap_positions(|seq, locals, memory, pos| {
            emit::map_point_in_place(seq, locals, memory, pos, &inverse, n);
        });
        self.wrap_bounds(|seq, locals, memory, out| {
            let bounds = emit::Bounds::new(locals, n);
            emit::load_bounds(seq, memory, out, &bounds);
            emit::BoundsMapper::new(locals, n).map(seq, &bounds, &map, &bounds);
            emit::store_bounds(seq, memory, out, &bounds);
        });
        Ok(())
    }

    /// Serialize the wrapped model.
    pub fn finish(mut self) -> Vec<u8> {
        self.module.emit_wasm()
    }
}
