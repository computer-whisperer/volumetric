//! Linear extrude operator: turns a 2D sketch model into a 3D model.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! The input must be a 2D model (`get_dimensions() -> 2`), e.g. a Lua
//! sketch. The output is a 3D model that is the sketch swept along +z from
//! `z = 0` to `z = height`:
//!
//! ```text
//! sample(x, y, z) = (0 <= z <= height) ? sketch_sample(x, y) : 0.0
//! ```
//!
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Returns 3
//! - `get_io_ptr() -> i32`: A fresh buffer in a newly reserved memory page —
//!   the sketch's own buffer only holds 2*2 f64s, a 3D model needs 2*3
//! - `get_bounds(out_ptr: i32)`: Sketch x/y bounds, then z in [0, height]
//! - `sample(pos_ptr: i32) -> f32`: z-gated call of the sketch's sample;
//!   x and y are read from pos_ptr as-is (the first two f64s)
//! - `memory`: Passed through from input model
//!
//! Typed sample channels are dropped: `get_sample_format` and
//! `sample_channels` exports are removed, so the output has the implicit
//! occupancy-only format.
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1 (height)
//! - Rejects non-2D input models
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct ExtrudeConfig {
    height: f64,
}

impl Default for ExtrudeConfig {
    fn default() -> Self {
        Self { height: 1.0 }
    }
}

use volumetric_abi::host::{post_output, read_input, report_error};

/// ABI exports the wrapper replaces. `sample_channels` and
/// `get_sample_format` are also removed (channels drop); `memory` passes
/// through.
const WRAPPED_FUNCTIONS: &[&str] = &["get_dimensions", "get_io_ptr", "get_bounds", "sample"];

/// Channel exports removed outright: the extruded model is occupancy-only.
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

/// Transform the input 2D sketch WASM into a 3D extrusion of height `h`.
fn transform_wasm(input_bytes: &[u8], cfg: ExtrudeConfig) -> Result<Vec<u8>, String> {
    if cfg.height <= 0.0 || cfg.height.is_nan() {
        return Err(format!("extrude height must be > 0, got {}", cfg.height));
    }

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

    // The input must be a 2D sketch: extruding a 3D model would silently
    // intersect it with a slab instead.
    match const_i32_return(&module, originals["get_dimensions"]) {
        Some(2) => {}
        Some(n) => {
            return Err(format!(
                "extrude input must be a 2D sketch, got a {n}-dimensional model"
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

    // sample(pos_ptr): gate on 0 <= z <= height, then call the sketch's
    // sample with the same pointer — x and y are already the first two f64s.
    {
        let orig_sample = originals["sample"];
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let z = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::{F64Gt, F64Lt, I32Or};
        let z_arg = walrus::ir::MemArg {
            align: 3,
            offset: 16,
        };

        b.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, z_arg)
            .local_set(z)
            // z < 0.0 || z > height  →  outside
            .local_get(z)
            .f64_const(0.0)
            .binop(F64Lt)
            .local_get(z)
            .f64_const(cfg.height)
            .binop(F64Gt)
            .binop(I32Or)
            .if_else(
                ValType::F32,
                |then| {
                    then.f32_const(0.0);
                },
                |else_| {
                    else_.local_get(pos_ptr).call(orig_sample);
                },
            );

        let fid = b.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", fid);
    }

    // get_bounds(out_ptr): sketch bounds fill offsets 0..32 (x/y), then
    // write z bounds [0, height] at offsets 32/40.
    {
        let orig_bounds = originals["get_bounds"];
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);

        let min_z_arg = walrus::ir::MemArg {
            align: 3,
            offset: 32,
        };
        let max_z_arg = walrus::ir::MemArg {
            align: 3,
            offset: 40,
        };

        b.func_body()
            .local_get(out_ptr)
            .call(orig_bounds)
            .local_get(out_ptr)
            .f64_const(0.0)
            .store(memory_id, walrus::ir::StoreKind::F64, min_z_arg)
            .local_get(out_ptr)
            .f64_const(cfg.height)
            .store(memory_id, walrus::ir::StoreKind::F64, max_z_arg);

        let fid = b.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", fid);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);

    let cfg = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            ExtrudeConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<ExtrudeConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    let output = match transform_wasm(&buf, cfg) {
        Ok(transformed) => transformed,
        Err(e) => {
            report_error(&format!("extrude failed: {e}"));
            return;
        }
    };

    post_output(0, &output);
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ height: float .default 1.0 }".to_string();
        OperatorMetadata {
            name: "extrude_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
