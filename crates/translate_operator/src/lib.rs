//! Translate operator.
//!
//! Host ABI:
//! - `host.get_input_len(i32) -> u32`
//! - `host.get_input_data(i32, ptr, len)`
//! - `host.post_output(i32, ptr, len)`
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply +1 x translation
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, Module, ModuleConfig, ValType};

#[link(wasm_import_module = "host")]
extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

/// The high-level ABI function names that models export
const ABI_FUNCTIONS: &[&str] = &[
    "is_inside",
    "get_bounds_min_x",
    "get_bounds_min_y",
    "get_bounds_min_z",
    "get_bounds_max_x",
    "get_bounds_max_y",
    "get_bounds_max_z",
];

/// Generate a random hex string for suffixing renamed functions
fn generate_hex_suffix() -> String {
    // Simple pseudo-random based on a fixed seed for reproducibility in WASM
    // In a real scenario, we might want to use a proper RNG or hash
    let mut state: u32 = 0xDEADBEEF;
    let mut result = String::with_capacity(8);
    for _ in 0..8 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let nibble = (state >> 16) & 0xF;
        result.push(char::from_digit(nibble, 16).unwrap());
    }
    result
}

/// Transform the input WASM module to apply +1 x translation
fn transform_wasm(input_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

    let suffix = generate_hex_suffix();

    // Find existing ABI functions via exports and collect their function IDs
    let mut renamed_functions: std::collections::HashMap<String, FunctionId> =
        std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();

    for export in module.exports.iter() {
        if ABI_FUNCTIONS.contains(&export.name.as_str()) {
            if let walrus::ExportItem::Function(func_id) = export.item {
                renamed_functions.insert(export.name.clone(), func_id);
                exports_to_remove.push(export.id());
            }
        }
    }

    // Remove old exports
    for export_id in exports_to_remove {
        module.exports.delete(export_id);
    }

    // Rename the original functions by adding suffix
    for (original_name, func_id) in &renamed_functions {
        let new_name = format!("{}_{}", original_name, suffix);
        let func = module.funcs.get_mut(*func_id);
        func.name = Some(new_name);
    }

    // Create wrapper functions for each ABI function
    for original_name in ABI_FUNCTIONS {
        if let Some(&original_func_id) = renamed_functions.get(*original_name) {
            let original_func = module.funcs.get(original_func_id);
            let _original_ty = module.types.get(original_func.ty());

            match *original_name {
                "is_inside" => {
                    // is_inside(x, y, z) -> i32
                    // Wrapper: is_inside(x, y, z) calls original_is_inside(x - 1.0, y, z)
                    let mut builder = FunctionBuilder::new(
                        &mut module.types,
                        &[ValType::F32, ValType::F32, ValType::F32],
                        &[ValType::I32],
                    );

                    let x = module.locals.add(ValType::F32);
                    let y = module.locals.add(ValType::F32);
                    let z = module.locals.add(ValType::F32);

                    builder
                        .func_body()
                        .local_get(x)
                        .f32_const(-1.0) // Subtract 1 from x (equivalent to translating +1)
                        .binop(walrus::ir::BinaryOp::F32Add)
                        .local_get(y)
                        .local_get(z)
                        .call(original_func_id);

                    let wrapper_id = builder.finish(vec![x, y, z], &mut module.funcs);
                    module.exports.add("is_inside", wrapper_id);
                }
                "get_bounds_min_x" => {
                    // get_bounds_min_x() -> f32
                    // Wrapper: returns original + 1.0
                    let mut builder = FunctionBuilder::new(
                        &mut module.types,
                        &[],
                        &[ValType::F32],
                    );

                    builder
                        .func_body()
                        .call(original_func_id)
                        .f32_const(1.0)
                        .binop(walrus::ir::BinaryOp::F32Add);

                    let wrapper_id = builder.finish(vec![], &mut module.funcs);
                    module.exports.add("get_bounds_min_x", wrapper_id);
                }
                "get_bounds_max_x" => {
                    // get_bounds_max_x() -> f32
                    // Wrapper: returns original + 1.0
                    let mut builder = FunctionBuilder::new(
                        &mut module.types,
                        &[],
                        &[ValType::F32],
                    );

                    builder
                        .func_body()
                        .call(original_func_id)
                        .f32_const(1.0)
                        .binop(walrus::ir::BinaryOp::F32Add);

                    let wrapper_id = builder.finish(vec![], &mut module.funcs);
                    module.exports.add("get_bounds_max_x", wrapper_id);
                }
                "get_bounds_min_y" | "get_bounds_max_y" | "get_bounds_min_z" | "get_bounds_max_z" => {
                    // These just pass through unchanged
                    let mut builder = FunctionBuilder::new(
                        &mut module.types,
                        &[],
                        &[ValType::F32],
                    );

                    builder
                        .func_body()
                        .call(original_func_id);

                    let wrapper_id = builder.finish(vec![], &mut module.funcs);
                    module.exports.add(*original_name, wrapper_id);
                }
                _ => {}
            }
        }
    }

    Ok(module.emit_wasm())
}

#[no_mangle]
pub extern "C" fn run() {
    let len = unsafe { get_input_len(0) } as usize;
    let mut buf = vec![0u8; len];

    if len > 0 {
        unsafe {
            get_input_data(0, buf.as_mut_ptr() as i32, len as i32);
        }
    }

    // Transform the WASM
    let output = match transform_wasm(&buf) {
        Ok(transformed) => transformed,
        Err(_) => {
            // On error, pass through unchanged (like identity)
            buf
        }
    };

    unsafe {
        post_output(0, output.as_ptr() as i32, output.len() as i32);
    }
}
