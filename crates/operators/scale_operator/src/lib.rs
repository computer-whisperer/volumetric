//! Scale operator.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Passed through from input model
//! - `get_io_ptr() -> i32`: Passed through from input model
//! - `get_bounds(out_ptr: i32)`: Wrapper that multiplies bounds by scale factors
//! - `sample(pos_ptr: i32) -> f32`: Wrapper that divides the position by the
//!   scale factors in place (the ABI allows clobbering the position buffer)
//! - `sample_channels(pos_ptr: i32, out_ptr: i32)`: Same wrapper as `sample`,
//!   present iff the input model has it
//! - `get_sample_format() -> i64`: Passed through from input model (a
//!   transform doesn't change what the samples mean)
//! - `memory`: Passed through from input model
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply non-uniform scaling (sx, sy, sz)
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
struct ScaleConfig {
    sx: f32,
    sy: f32,
    sz: f32,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            sx: 1.0,
            sy: 1.0,
            sz: 1.0,
        }
    }
}

use volumetric_abi::host::{post_output, read_input, report_error};

/// N-dimensional ABI function names. `get_dimensions`, `get_io_ptr` and
/// `get_sample_format` are passed through unchanged; `get_bounds`, `sample`
/// and `sample_channels` get wrappers.
const ABI_FUNCTIONS_ND: &[&str] = &[
    "get_dimensions",
    "get_io_ptr",
    "get_bounds",
    "sample",
    "sample_channels",
];

/// ABI functions the wrappers replace; the rest keep their exports.
const WRAPPED_FUNCTIONS: &[&str] = &["get_bounds", "sample", "sample_channels"];

fn generate_hex_suffix() -> String {
    let mut state: u32 = 0xC0FFEE77;
    let mut result = String::with_capacity(8);
    for _ in 0..8 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let nibble = (state >> 16) & 0xF;
        result.push(char::from_digit(nibble, 16).unwrap());
    }
    result
}

/// Emit code that divides the first 3 dims by the scale factors in place at
/// `pos_ptr` (shared by the `sample` and `sample_channels` wrappers).
fn emit_position_rewrite(
    b: &mut FunctionBuilder,
    pos_ptr: walrus::LocalId,
    memory_id: MemoryId,
    cfg: &ScaleConfig,
) {
    use walrus::ir::BinaryOp::F64Div;

    // pos[i] /= s for each axis
    for (offset, s) in [(0, cfg.sx), (8, cfg.sy), (16, cfg.sz)] {
        let mem_arg = walrus::ir::MemArg { align: 3, offset };
        b.func_body()
            .local_get(pos_ptr)
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
            .f64_const(s as f64)
            .binop(F64Div)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg);
    }
}

fn transform_wasm(input_bytes: &[u8], cfg: ScaleConfig) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

    let suffix = generate_hex_suffix();

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

    let mut renamed: std::collections::HashMap<String, FunctionId> =
        std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();
    for export in module.exports.iter() {
        if ABI_FUNCTIONS_ND.contains(&export.name.as_str()) {
            if let walrus::ExportItem::Function(func_id) = export.item {
                renamed.insert(export.name.clone(), func_id);
                exports_to_remove.push(export.id());
            }
        }
    }

    if !renamed.contains_key("get_io_ptr") {
        return Err(
            "Input model missing `get_io_ptr` export; rebuild it against the \
                    current N-dimensional ABI"
                .to_string(),
        );
    }

    // Remove the exports we wrap; memory, get_dimensions and get_io_ptr pass through
    for id in exports_to_remove {
        let export_name = module.exports.get(id).name.clone();
        if WRAPPED_FUNCTIONS.contains(&export_name.as_str()) {
            module.exports.delete(id);
        }
    }

    // Rename the wrapped functions by adding suffix
    for (name, func_id) in &renamed {
        if WRAPPED_FUNCTIONS.contains(&name.as_str()) {
            let new_name = format!("{}_{}", name, suffix);
            module.funcs.get_mut(*func_id).name = Some(new_name);
        }
    }

    // sample(pos_ptr): divide the position by the scale factors in place,
    // then call the original with the same pointer
    if let Some(&orig) = renamed.get("sample") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);

        emit_position_rewrite(&mut b, pos_ptr, memory_id, &cfg);
        b.func_body().local_get(pos_ptr).call(orig);

        let fid = b.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", fid);
    }

    // sample_channels(pos_ptr, out_ptr): identical position rewrite, then
    // call the original with both pointers. Only present when the input model
    // declares typed channels; get_sample_format passes through untouched.
    if let Some(&orig) = renamed.get("sample_channels") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32, ValType::I32], &[]);
        let pos_ptr = module.locals.add(ValType::I32);
        let out_ptr = module.locals.add(ValType::I32);

        emit_position_rewrite(&mut b, pos_ptr, memory_id, &cfg);
        b.func_body()
            .local_get(pos_ptr)
            .local_get(out_ptr)
            .call(orig);

        let fid = b.finish(vec![pos_ptr, out_ptr], &mut module.funcs);
        module.exports.add("sample_channels", fid);
    }

    // get_bounds(out_ptr): call original, then multiply bounds by scale factors
    // Also swap min/max if scale is negative
    if let Some(&orig) = renamed.get("get_bounds") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        let min_val = module.locals.add(ValType::F64);
        let max_val = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::{F64Max, F64Min, F64Mul};

        // Call original get_bounds
        b.func_body().local_get(out_ptr).call(orig);

        // Process each axis: scale and handle sign flips
        let scales = [
            (0, 8, cfg.sx as f64),
            (16, 24, cfg.sy as f64),
            (32, 40, cfg.sz as f64),
        ];

        for (min_offset, max_offset, scale) in scales {
            let mem_arg_min = walrus::ir::MemArg {
                align: 3,
                offset: min_offset,
            };
            let mem_arg_max = walrus::ir::MemArg {
                align: 3,
                offset: max_offset,
            };

            // Load min and max, multiply by scale
            b.func_body()
                .local_get(out_ptr)
                .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_min)
                .f64_const(scale)
                .binop(F64Mul)
                .local_set(min_val);

            b.func_body()
                .local_get(out_ptr)
                .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_max)
                .f64_const(scale)
                .binop(F64Mul)
                .local_set(max_val);

            // Store min(scaled_min, scaled_max) and max(scaled_min, scaled_max)
            // to handle negative scale factors
            b.func_body()
                .local_get(out_ptr)
                .local_get(min_val)
                .local_get(max_val)
                .binop(F64Min)
                .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_min);

            b.func_body()
                .local_get(out_ptr)
                .local_get(min_val)
                .local_get(max_val)
                .binop(F64Max)
                .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_max);
        }

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
            ScaleConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            ciborium::de::from_reader::<ScaleConfig, _>(&mut cursor).unwrap_or_default()
        }
    };

    let output = match transform_wasm(&buf, cfg) {
        Ok(t) => t,
        Err(e) => {
            report_error(&format!("transform failed: {e}"));
            return;
        }
    };
    post_output(0, &output);
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ sx: float .default 1.0, sy: float .default 1.0, sz: float .default 1.0 }"
            .to_string();
        OperatorMetadata {
            name: "scale_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
