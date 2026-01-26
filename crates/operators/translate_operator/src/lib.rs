//! Translate operator.
//!
//! Host ABI:
//! - `host.get_input_len(i32) -> u32`
//! - `host.get_input_data(i32, ptr, len)`
//! - `host.post_output(i32, ptr, len)`
//!
//! Operator ABI:
//! - `get_metadata() -> i64` returning `(ptr: u32, len: u32)` packed as `ptr | (len << 32)`
//!
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Passed through from input model
//! - `get_bounds(out_ptr: i32)`: Wrapper that adds translation to bounds
//! - `sample(pos_ptr: i32) -> f32`: Wrapper that subtracts translation from position
//! - `memory`: Passed through from input model
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1 (schema declared in metadata)
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply configurable translation (dx/dy/dz)
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataInput {
    ModelWASM,
    VecF64(usize),
}

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataOutput {
    ModelWASM,
}

#[derive(Clone, Debug, serde::Serialize)]
struct OperatorMetadata {
    name: String,
    version: String,
    inputs: Vec<OperatorMetadataInput>,
    outputs: Vec<OperatorMetadataOutput>,
}

/// Read input data from the host
fn read_input(idx: i32) -> Vec<u8> {
    let len = unsafe { get_input_len(idx) } as usize;
    if len == 0 {
        return Vec::new();
    }
    let mut buf = vec![0u8; len];
    unsafe { get_input_data(idx, buf.as_mut_ptr() as i32, len as i32) };
    buf
}

/// Decode a VecF64 from raw bytes (8 bytes per f64, little-endian), with a default fallback
fn decode_vec3(data: &[u8], default: [f64; 3]) -> [f64; 3] {
    if data.len() < 24 {
        return default;
    }
    let x = f64::from_le_bytes(data[0..8].try_into().unwrap());
    let y = f64::from_le_bytes(data[8..16].try_into().unwrap());
    let z = f64::from_le_bytes(data[16..24].try_into().unwrap());
    [x, y, z]
}

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

/// New N-dimensional ABI function names
const ABI_FUNCTIONS_ND: &[&str] = &[
    "get_dimensions",
    "get_bounds",
    "sample",
];

/// Scratch buffer offset for transformed position (after bounds buffer at 256)
const SCRATCH_POS_OFFSET: i32 = 512;

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
fn transform_wasm(input_bytes: &[u8], translation: [f64; 3]) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

    let suffix = generate_hex_suffix();

    // Find memory export
    let memory_id: Option<MemoryId> = module.exports.iter()
        .find(|e| e.name == "memory")
        .and_then(|e| if let walrus::ExportItem::Memory(m) = e.item { Some(m) } else { None });

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

    // Remove old exports (except memory and get_dimensions which we pass through)
    for export_id in exports_to_remove {
        let export_name = module.exports.get(export_id).name.clone();
        if export_name != "get_dimensions" {
            module.exports.delete(export_id);
        }
    }

    // Rename the original functions by adding suffix (except get_dimensions)
    for (original_name, func_id) in &renamed_functions {
        if original_name != "get_dimensions" {
            let new_name = format!("{}_{}", original_name, suffix);
            let func = module.funcs.get_mut(*func_id);
            func.name = Some(new_name);
        }
    }

    // Create wrapper for sample
    if let Some(&original_sample_id) = renamed_functions.get("sample") {
        // sample(pos_ptr: i32) -> f32
        // Wrapper reads position from pos_ptr, subtracts translation for first 3 dims,
        // writes to scratch buffer, then calls original sample with scratch buffer
        let mut builder = FunctionBuilder::new(
            &mut module.types,
            &[ValType::I32],
            &[ValType::F32],
        );

        let pos_ptr = module.locals.add(ValType::I32);
        let x = module.locals.add(ValType::F64);
        let y = module.locals.add(ValType::F64);
        let z = module.locals.add(ValType::F64);

        let mem_arg = walrus::ir::MemArg { align: 3, offset: 0,  };
        let mem_arg_8 = walrus::ir::MemArg { align: 3, offset: 8,  };
        let mem_arg_16 = walrus::ir::MemArg { align: 3, offset: 16,  };

        // Load x, y, z from input position
        builder.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
            .local_set(x);

        builder.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8)
            .local_set(y);

        builder.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16)
            .local_set(z);

        // Subtract translation and write to scratch buffer
        let scratch_arg = walrus::ir::MemArg { align: 3, offset: 0,  };
        let scratch_arg_8 = walrus::ir::MemArg { align: 3, offset: 8,  };
        let scratch_arg_16 = walrus::ir::MemArg { align: 3, offset: 16,  };

        // x - dx -> scratch[0]
        builder.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(x)
            .f64_const(-(translation[0]))
            .binop(walrus::ir::BinaryOp::F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg);

        // y - dy -> scratch[8]
        builder.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(y)
            .f64_const(-(translation[1]))
            .binop(walrus::ir::BinaryOp::F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg_8);

        // z - dz -> scratch[16]
        builder.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(z)
            .f64_const(-(translation[2]))
            .binop(walrus::ir::BinaryOp::F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg_16);

        // Call original sample with scratch buffer pointer
        builder.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .call(original_sample_id);

        let wrapper_id = builder.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", wrapper_id);
    }

    // Create wrapper for get_bounds
    if let Some(&original_bounds_id) = renamed_functions.get("get_bounds") {
        // get_bounds(out_ptr: i32)
        // Wrapper calls original, then adds translation to the returned bounds
        let mut builder = FunctionBuilder::new(
            &mut module.types,
            &[ValType::I32],
            &[],
        );

        let out_ptr = module.locals.add(ValType::I32);
        let tmp = module.locals.add(ValType::F64);

        // Call original get_bounds
        builder.func_body()
            .local_get(out_ptr)
            .call(original_bounds_id);

        let mem_arg = walrus::ir::MemArg { align: 3, offset: 0,  };

        // Add dx to min_x (offset 0)
        builder.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
            .f64_const(translation[0])
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body()
            .local_get(out_ptr)
            .local_get(tmp)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg);

        // Add dx to max_x (offset 8)
        let mem_arg_8 = walrus::ir::MemArg { align: 3, offset: 8,  };
        builder.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8)
            .f64_const(translation[0])
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body()
            .local_get(out_ptr)
            .local_get(tmp)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_8);

        // Add dy to min_y (offset 16)
        let mem_arg_16 = walrus::ir::MemArg { align: 3, offset: 16,  };
        builder.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16)
            .f64_const(translation[1])
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body()
            .local_get(out_ptr)
            .local_get(tmp)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_16);

        // Add dy to max_y (offset 24)
        let mem_arg_24 = walrus::ir::MemArg { align: 3, offset: 24,  };
        builder.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_24)
            .f64_const(translation[1])
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body()
            .local_get(out_ptr)
            .local_get(tmp)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_24);

        // Add dz to min_z (offset 32)
        let mem_arg_32 = walrus::ir::MemArg { align: 3, offset: 32,  };
        builder.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_32)
            .f64_const(translation[2])
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body()
            .local_get(out_ptr)
            .local_get(tmp)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_32);

        // Add dz to max_z (offset 40)
        let mem_arg_40 = walrus::ir::MemArg { align: 3, offset: 40,  };
        builder.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_40)
            .f64_const(translation[2])
            .binop(walrus::ir::BinaryOp::F64Add)
            .local_set(tmp);
        builder.func_body()
            .local_get(out_ptr)
            .local_get(tmp)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_40);

        let wrapper_id = builder.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", wrapper_id);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    // Read input 0: Model WASM bytes
    let wasm_data = read_input(0);

    // Read input 1: translation offset as VecF64(3) - [dx, dy, dz]
    // Default to [0, 0, 0] (no translation)
    let translation_data = read_input(1);
    let translation = decode_vec3(&translation_data, [0.0, 0.0, 0.0]);

    // Transform the WASM
    let output = match transform_wasm(&wasm_data, translation) {
        Ok(transformed) => transformed,
        Err(_) => {
            // On error, pass through unchanged (like identity)
            wasm_data
        }
    };

    unsafe {
        post_output(0, output.as_ptr() as i32, output.len() as i32);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let metadata = OperatorMetadata {
            name: "translate_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::VecF64(3), // translation offset [dx, dy, dz]
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };

        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out)
            .expect("translate_operator metadata CBOR serialization should not fail");
        out
    });

    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}
