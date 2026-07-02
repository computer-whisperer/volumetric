//! Translate operator.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Passed through from input model
//! - `get_io_ptr() -> i32`: Passed through from input model
//! - `get_bounds(out_ptr: i32)`: Wrapper that adds translation to bounds
//! - `sample(pos_ptr: i32) -> f32`: Wrapper that subtracts translation from
//!   the position in place (the ABI allows clobbering the position buffer)
//! - `memory`: Passed through from input model
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1 (schema declared in metadata)
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply configurable translation (dx/dy/dz)
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
struct TranslateConfig {
    dx: f32,
    dy: f32,
    dz: f32,
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }
}

use volumetric_abi::host::{post_output, read_input, report_error};

/// N-dimensional ABI function names. `get_dimensions` and `get_io_ptr` are
/// passed through unchanged; `get_bounds` and `sample` get wrappers.
const ABI_FUNCTIONS_ND: &[&str] = &["get_dimensions", "get_io_ptr", "get_bounds", "sample"];

/// ABI functions the wrappers replace; the rest keep their exports.
const WRAPPED_FUNCTIONS: &[&str] = &["get_bounds", "sample"];

/// Generate a random hex string for suffixing renamed functions
fn generate_hex_suffix() -> String {
    let mut state: u32 = 0xDEADBEEF;
    let mut result = String::with_capacity(8);
    for _ in 0..8 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let nibble = (state >> 16) & 0xF;
        result.push(char::from_digit(nibble, 16).unwrap());
    }
    result
}

/// Transform the input WASM module to apply translation by (dx, dy, dz).
fn transform_wasm(input_bytes: &[u8], cfg: TranslateConfig) -> Result<Vec<u8>, String> {
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

    // Find existing ABI functions via exports and collect their function IDs
    let mut renamed_functions: std::collections::HashMap<String, FunctionId> =
        std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();

    for export in module.exports.iter() {
        if ABI_FUNCTIONS_ND.contains(&export.name.as_str()) {
            if let walrus::ExportItem::Function(func_id) = export.item {
                renamed_functions.insert(export.name.clone(), func_id);
                exports_to_remove.push(export.id());
            }
        }
    }

    if !renamed_functions.contains_key("get_io_ptr") {
        return Err(
            "Input model missing `get_io_ptr` export; rebuild it against the \
                    current N-dimensional ABI"
                .to_string(),
        );
    }

    // Remove the exports we wrap; memory, get_dimensions and get_io_ptr pass through
    for export_id in exports_to_remove {
        let export_name = module.exports.get(export_id).name.clone();
        if WRAPPED_FUNCTIONS.contains(&export_name.as_str()) {
            module.exports.delete(export_id);
        }
    }

    // Rename the wrapped functions by adding suffix
    for (original_name, func_id) in &renamed_functions {
        if WRAPPED_FUNCTIONS.contains(&original_name.as_str()) {
            let new_name = format!("{}_{}", original_name, suffix);
            let func = module.funcs.get_mut(*func_id);
            func.name = Some(new_name);
        }
    }

    // Create wrapper for sample
    if let Some(&original_sample_id) = renamed_functions.get("sample") {
        // sample(pos_ptr: i32) -> f32
        // Wrapper subtracts the translation from the first 3 dims in place at
        // pos_ptr, then calls the original sample with the same pointer.
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);

        let pos_ptr = module.locals.add(ValType::I32);

        // pos[i] -= d for each axis
        for (offset, d) in [(0, cfg.dx), (8, cfg.dy), (16, cfg.dz)] {
            let mem_arg = walrus::ir::MemArg { align: 3, offset };
            builder
                .func_body()
                .local_get(pos_ptr)
                .local_get(pos_ptr)
                .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
                .f64_const(-(d as f64))
                .binop(walrus::ir::BinaryOp::F64Add)
                .store(memory_id, walrus::ir::StoreKind::F64, mem_arg);
        }

        // Call original sample with the transformed position
        builder
            .func_body()
            .local_get(pos_ptr)
            .call(original_sample_id);

        let wrapper_id = builder.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", wrapper_id);
    }

    // Create wrapper for get_bounds
    if let Some(&original_bounds_id) = renamed_functions.get("get_bounds") {
        // get_bounds(out_ptr: i32)
        // Wrapper calls original, then adds translation to the returned bounds
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);

        let out_ptr = module.locals.add(ValType::I32);
        let tmp = module.locals.add(ValType::F64);

        // Call original get_bounds
        builder
            .func_body()
            .local_get(out_ptr)
            .call(original_bounds_id);

        let mem_arg = walrus::ir::MemArg {
            align: 3,
            offset: 0,
        };

        // Add dx to min_x (offset 0)
        builder
            .func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
            .f64_const(cfg.dx as f64)
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body().local_get(out_ptr).local_get(tmp).store(
            memory_id,
            walrus::ir::StoreKind::F64,
            mem_arg,
        );

        // Add dx to max_x (offset 8)
        let mem_arg_8 = walrus::ir::MemArg {
            align: 3,
            offset: 8,
        };
        builder
            .func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8)
            .f64_const(cfg.dx as f64)
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body().local_get(out_ptr).local_get(tmp).store(
            memory_id,
            walrus::ir::StoreKind::F64,
            mem_arg_8,
        );

        // Add dy to min_y (offset 16)
        let mem_arg_16 = walrus::ir::MemArg {
            align: 3,
            offset: 16,
        };
        builder
            .func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16)
            .f64_const(cfg.dy as f64)
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body().local_get(out_ptr).local_get(tmp).store(
            memory_id,
            walrus::ir::StoreKind::F64,
            mem_arg_16,
        );

        // Add dy to max_y (offset 24)
        let mem_arg_24 = walrus::ir::MemArg {
            align: 3,
            offset: 24,
        };
        builder
            .func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_24)
            .f64_const(cfg.dy as f64)
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body().local_get(out_ptr).local_get(tmp).store(
            memory_id,
            walrus::ir::StoreKind::F64,
            mem_arg_24,
        );

        // Add dz to min_z (offset 32)
        let mem_arg_32 = walrus::ir::MemArg {
            align: 3,
            offset: 32,
        };
        builder
            .func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_32)
            .f64_const(cfg.dz as f64)
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body().local_get(out_ptr).local_get(tmp).store(
            memory_id,
            walrus::ir::StoreKind::F64,
            mem_arg_32,
        );

        // Add dz to max_z (offset 40)
        let mem_arg_40 = walrus::ir::MemArg {
            align: 3,
            offset: 40,
        };
        builder
            .func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_40)
            .f64_const(cfg.dz as f64)
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body().local_get(out_ptr).local_get(tmp).store(
            memory_id,
            walrus::ir::StoreKind::F64,
            mem_arg_40,
        );

        let wrapper_id = builder.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", wrapper_id);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);

    let cfg = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            TranslateConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            ciborium::de::from_reader::<TranslateConfig, _>(&mut cursor).unwrap_or_default()
        }
    };

    // Transform the WASM
    let output = match transform_wasm(&buf, cfg) {
        Ok(transformed) => transformed,
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
        let schema = "{ dx: float .default 0.0, dy: float .default 0.0, dz: float .default 0.0 }"
            .to_string();
        OperatorMetadata {
            name: "translate_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
