//! Revolve operator: turns a 2D sketch model into a 3D solid of revolution.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! The input must be a 2D model (`get_dimensions() -> 2`), e.g. a Lua
//! sketch. The sketch lives in the r/z plane — its x-axis is the radial
//! distance from the axis of revolution, its y-axis is world z — and the
//! output revolves it a full turn around the world z-axis:
//!
//! ```text
//! sample(x, y, z) = sketch_sample(sqrt(x^2 + y^2), z)
//! ```
//!
//! The radius is always >= 0, so only the x >= 0 half of the sketch
//! contributes (the standard profile-beside-the-axis convention).
//!
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Returns 3
//! - `get_io_ptr() -> i32`: A fresh buffer in a newly reserved memory page —
//!   the sketch's own buffer only holds 2*2 f64s, a 3D model needs 2*3
//! - `get_bounds(out_ptr: i32)`: x/y in [-R, R] with R = max(sketch max_x, 0);
//!   z takes the sketch's y bounds
//! - `sample(pos_ptr: i32) -> f32`: rewrites the position to (r, z) in place,
//!   then calls the sketch's sample
//! - `memory`: Passed through from input model
//!
//! Typed sample channels are dropped: `get_sample_format` and
//! `sample_channels` exports are removed, so the output has the implicit
//! occupancy-only format.
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Rejects non-2D input models
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

use volumetric_abi::host::{post_output, read_input, report_error};

/// ABI exports the wrapper replaces. `sample_channels` and
/// `get_sample_format` are also removed (channels drop); `memory` passes
/// through.
const WRAPPED_FUNCTIONS: &[&str] = &["get_dimensions", "get_io_ptr", "get_bounds", "sample"];

/// Channel exports removed outright: the revolved model is occupancy-only.
const DROPPED_FUNCTIONS: &[&str] = &["get_sample_format", "sample_channels"];

/// Read the constant a trivial `() -> i32` function returns, if its body is
/// a single `i32.const`. Every model generator emits `get_dimensions` this
/// way, so this is how the operator checks the input is 2D without being
/// able to instantiate it.
fn const_i32_return(module: &Module, func_id: FunctionId) -> Option<i32> {
    let local = match &module.funcs.get(func_id).kind {
        walrus::FunctionKind::Local(local) => local,
        _ => return None,
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

/// Transform the input 2D sketch WASM into a full revolution about world z.
fn transform_wasm(input_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

    // Find memory export
    let memory_id: Option<MemoryId> =
        module
            .exports
            .iter()
            .find(|e| e.name == "memory")
            .and_then(|e| {
                if let walrus::ExportItem::Memory(m) = e.item {
                    Some(m)
                } else {
                    None
                }
            });
    let memory_id = memory_id.ok_or("Input model missing memory export")?;

    // Collect the ABI function exports we wrap or drop
    let mut originals: std::collections::HashMap<String, FunctionId> =
        std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();
    for export in module.exports.iter() {
        let name = export.name.as_str();
        if WRAPPED_FUNCTIONS.contains(&name) || DROPPED_FUNCTIONS.contains(&name) {
            if let walrus::ExportItem::Function(func_id) = export.item {
                originals.insert(export.name.clone(), func_id);
                exports_to_remove.push(export.id());
            }
        }
    }

    for required in WRAPPED_FUNCTIONS {
        if !originals.contains_key(*required) {
            return Err(format!("Input model missing `{required}` export"));
        }
    }

    // The input must be a 2D sketch: revolving a 3D model has no meaning
    // under this operator.
    match const_i32_return(&module, originals["get_dimensions"]) {
        Some(2) => {}
        Some(n) => {
            return Err(format!(
                "revolve input must be a 2D sketch, got a {n}-dimensional model"
            ));
        }
        None => {
            return Err(
                "cannot determine input model dimensionality (get_dimensions is not a \
                 constant function)"
                    .to_string(),
            );
        }
    }

    for export_id in exports_to_remove {
        module.exports.delete(export_id);
    }

    // The sketch's IO buffer holds 2*2 f64s; a 3D model must offer 2*3.
    // Reserve a fresh page and put the new buffer at its start.
    let new_io_ptr = {
        let memory = module.memories.get_mut(memory_id);
        let io_ptr = (memory.initial * 65536) as i32;
        memory.initial += 1;
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
        io_ptr
    };

    // get_dimensions() -> 3
    {
        let mut b = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        b.func_body().i32_const(3);
        let fid = b.finish(vec![], &mut module.funcs);
        module.exports.add("get_dimensions", fid);
    }

    // get_io_ptr() -> the new 2*3-f64 buffer
    {
        let mut b = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        b.func_body().i32_const(new_io_ptr);
        let fid = b.finish(vec![], &mut module.funcs);
        module.exports.add("get_io_ptr", fid);
    }

    // sample(pos_ptr): rewrite (x, y, z) to (r, z) in place — r at offset 0,
    // the sketch's second coordinate (world z) at offset 8 — then call the
    // sketch's sample with the same pointer.
    {
        let orig_sample = originals["sample"];
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let r = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::{F64Add, F64Mul};
        use walrus::ir::UnaryOp::F64Sqrt;
        let x_arg = walrus::ir::MemArg {
            align: 3,
            offset: 0,
        };
        let y_arg = walrus::ir::MemArg {
            align: 3,
            offset: 8,
        };
        let z_arg = walrus::ir::MemArg {
            align: 3,
            offset: 16,
        };

        // r = sqrt(x*x + y*y)
        b.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, x_arg)
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, x_arg)
            .binop(F64Mul)
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, y_arg)
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, y_arg)
            .binop(F64Mul)
            .binop(F64Add)
            .unop(F64Sqrt)
            .local_set(r);

        // pos[1] = pos[2] (world z becomes the sketch's second coordinate),
        // then pos[0] = r. Ordering matters: read z before overwriting.
        b.func_body()
            .local_get(pos_ptr)
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, z_arg)
            .store(memory_id, walrus::ir::StoreKind::F64, y_arg)
            .local_get(pos_ptr)
            .local_get(r)
            .store(memory_id, walrus::ir::StoreKind::F64, x_arg);

        b.func_body().local_get(pos_ptr).call(orig_sample);

        let fid = b.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", fid);
    }

    // get_bounds(out_ptr): the sketch writes [min_x, max_x, min_y, max_y] at
    // offsets 0..32. Rearranged: R = max(max_x, 0); x and y span [-R, R] and
    // z takes the sketch's y bounds. Read the y bounds before overwriting
    // their slots.
    {
        let orig_bounds = originals["get_bounds"];
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        let radius = module.locals.add(ValType::F64);
        let min_z = module.locals.add(ValType::F64);
        let max_z = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::F64Max;
        use walrus::ir::UnaryOp::F64Neg;
        let arg = |offset: u64| walrus::ir::MemArg { align: 3, offset };

        b.func_body()
            .local_get(out_ptr)
            .call(orig_bounds)
            // radius = max(sketch max_x, 0)
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, arg(8))
            .f64_const(0.0)
            .binop(F64Max)
            .local_set(radius)
            // stash the sketch's y bounds before their slots are overwritten
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, arg(16))
            .local_set(min_z)
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, arg(24))
            .local_set(max_z)
            // x in [-R, R]
            .local_get(out_ptr)
            .local_get(radius)
            .unop(F64Neg)
            .store(memory_id, walrus::ir::StoreKind::F64, arg(0))
            .local_get(out_ptr)
            .local_get(radius)
            .store(memory_id, walrus::ir::StoreKind::F64, arg(8))
            // y in [-R, R]
            .local_get(out_ptr)
            .local_get(radius)
            .unop(F64Neg)
            .store(memory_id, walrus::ir::StoreKind::F64, arg(16))
            .local_get(out_ptr)
            .local_get(radius)
            .store(memory_id, walrus::ir::StoreKind::F64, arg(24))
            // z takes the sketch's y bounds
            .local_get(out_ptr)
            .local_get(min_z)
            .store(memory_id, walrus::ir::StoreKind::F64, arg(32))
            .local_get(out_ptr)
            .local_get(max_z)
            .store(memory_id, walrus::ir::StoreKind::F64, arg(40));

        let fid = b.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", fid);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);

    let output = match transform_wasm(&buf) {
        Ok(transformed) => transformed,
        Err(e) => {
            report_error(&format!("revolve failed: {e}"));
            return;
        }
    };

    post_output(0, &output);
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "revolve_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        inputs: vec![OperatorMetadataInput::ModelWASM],
        input_names: vec!["Profile (2D)".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}
