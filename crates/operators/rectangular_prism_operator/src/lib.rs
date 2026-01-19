//! Rectangular Prism Operator.
//!
//! Emitter operator that generates a volumetric box model using the N-dimensional ABI.
//!
//! Host ABI:
//! - `host.get_input_len(i32) -> u32`
//! - `host.get_input_data(i32, ptr, len)`
//! - `host.post_output(i32, ptr, len)`
//!
//! Operator ABI:
//! - `get_metadata() -> i64` returning `(ptr: u32, len: u32)` packed as `ptr | (len << 32)`
//!
//! Generated Model ABI:
//! - `get_dimensions() -> u32`: Returns 3
//! - `get_bounds(out_ptr: i32)`: Writes interleaved min/max bounds
//! - `sample(pos_ptr: i32) -> f32`: Reads position from memory, returns density
//! - `memory`: Linear memory export
//!
//! Inputs:
//! - Input 0: CBOR configuration with mode selector
//! - Input 1: VecF64(3) - vector_a (corner or center position)
//! - Input 2: VecF64(3) - vector_b (corner or dimensions)
//!
//! Mode Behavior:
//! - opposite_corners: vector_a = corner min, vector_b = corner max
//! - position_size: vector_a = center position, vector_b = dimensions (width, height, depth)

use walrus::{FunctionBuilder, MemoryId, Module, ValType};

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataInput {
    CBORConfiguration(String),
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

#[derive(Clone, Debug, serde::Deserialize)]
struct ModeConfig {
    mode: String,
}

impl Default for ModeConfig {
    fn default() -> Self {
        Self {
            mode: "opposite_corners".to_string(),
        }
    }
}

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
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
    // Expect exactly 24 bytes (3 * 8 bytes per f64)
    if data.len() < 24 {
        return default;
    }

    let x = f64::from_le_bytes(data[0..8].try_into().unwrap());
    let y = f64::from_le_bytes(data[8..16].try_into().unwrap());
    let z = f64::from_le_bytes(data[16..24].try_into().unwrap());

    [x, y, z]
}

fn generate_wasm(mode: &str, vector_a: [f64; 3], vector_b: [f64; 3]) -> Result<Vec<u8>, String> {
    // Calculate min/max bounds based on mode
    let (min_x, min_y, min_z, max_x, max_y, max_z) = match mode {
        "position_size" => {
            // vector_a = center position, vector_b = dimensions (width, height, depth)
            let cx = vector_a[0];
            let cy = vector_a[1];
            let cz = vector_a[2];
            let half_w = vector_b[0] / 2.0;
            let half_h = vector_b[1] / 2.0;
            let half_d = vector_b[2] / 2.0;
            (
                cx - half_w,
                cy - half_h,
                cz - half_d,
                cx + half_w,
                cy + half_h,
                cz + half_d,
            )
        }
        _ => {
            // opposite_corners (default): vector_a = corner min, vector_b = corner max
            // Ensure min is actually min and max is actually max
            (
                vector_a[0].min(vector_b[0]),
                vector_a[1].min(vector_b[1]),
                vector_a[2].min(vector_b[2]),
                vector_a[0].max(vector_b[0]),
                vector_a[1].max(vector_b[1]),
                vector_a[2].max(vector_b[2]),
            )
        }
    };

    let mut module = Module::default();

    // Add memory (1 page = 64KB, more than enough for I/O buffers)
    // add_local(shared, memory64, initial_pages, maximum_pages, page_size_log2)
    let memory_id: MemoryId = module.memories.add_local(false, false, 1, None, None);
    module.exports.add("memory", memory_id);

    // Create get_dimensions() -> u32
    {
        let mut builder = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        builder.func_body().i32_const(3);
        let func_id = builder.finish(vec![], &mut module.funcs);
        module.exports.add("get_dimensions", func_id);
    }

    // Create get_bounds(out_ptr: i32)
    // Writes interleaved min/max: [min_x, max_x, min_y, max_y, min_z, max_z] as f64
    {
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);

        let mem_arg = walrus::ir::MemArg {
            align: 3, // 2^3 = 8 byte alignment
            offset: 0,
        };

        // Store min_x at out_ptr + 0
        builder.func_body()
            .local_get(out_ptr)
            .f64_const(min_x)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg);

        // Store max_x at out_ptr + 8
        let mem_arg_8 = walrus::ir::MemArg { align: 3, offset: 8 };
        builder.func_body()
            .local_get(out_ptr)
            .f64_const(max_x)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_8);

        // Store min_y at out_ptr + 16
        let mem_arg_16 = walrus::ir::MemArg { align: 3, offset: 16 };
        builder.func_body()
            .local_get(out_ptr)
            .f64_const(min_y)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_16);

        // Store max_y at out_ptr + 24
        let mem_arg_24 = walrus::ir::MemArg { align: 3, offset: 24 };
        builder.func_body()
            .local_get(out_ptr)
            .f64_const(max_y)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_24);

        // Store min_z at out_ptr + 32
        let mem_arg_32 = walrus::ir::MemArg { align: 3, offset: 32 };
        builder.func_body()
            .local_get(out_ptr)
            .f64_const(min_z)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_32);

        // Store max_z at out_ptr + 40
        let mem_arg_40 = walrus::ir::MemArg { align: 3, offset: 40 };
        builder.func_body()
            .local_get(out_ptr)
            .f64_const(max_z)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_40);

        let func_id = builder.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", func_id);
    }

    // Create sample(pos_ptr: i32) -> f32
    // Reads position from memory: [x, y, z] as f64 at pos_ptr
    // Returns 1.0 if inside the box, 0.0 otherwise
    {
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let x = module.locals.add(ValType::F64);
        let y = module.locals.add(ValType::F64);
        let z = module.locals.add(ValType::F64);

        let mem_arg = walrus::ir::MemArg { align: 3, offset: 0 };
        let mem_arg_8 = walrus::ir::MemArg { align: 3, offset: 8 };
        let mem_arg_16 = walrus::ir::MemArg { align: 3, offset: 16 };

        // Load x from pos_ptr + 0
        builder.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
            .local_set(x);

        // Load y from pos_ptr + 8
        builder.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8)
            .local_set(y);

        // Load z from pos_ptr + 16
        builder.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16)
            .local_set(z);

        use walrus::ir::BinaryOp::{F64Ge, F64Le, I32And};
        use walrus::ir::UnaryOp::F32ConvertSI32;

        // Check: (x >= min_x) && (x <= max_x) && (y >= min_y) && (y <= max_y) && (z >= min_z) && (z <= max_z)
        builder.func_body()
            // x >= min_x
            .local_get(x)
            .f64_const(min_x)
            .binop(F64Ge)
            // x <= max_x
            .local_get(x)
            .f64_const(max_x)
            .binop(F64Le)
            // AND
            .binop(I32And)
            // y >= min_y
            .local_get(y)
            .f64_const(min_y)
            .binop(F64Ge)
            // AND
            .binop(I32And)
            // y <= max_y
            .local_get(y)
            .f64_const(max_y)
            .binop(F64Le)
            // AND
            .binop(I32And)
            // z >= min_z
            .local_get(z)
            .f64_const(min_z)
            .binop(F64Ge)
            // AND
            .binop(I32And)
            // z <= max_z
            .local_get(z)
            .f64_const(max_z)
            .binop(F64Le)
            // AND
            .binop(I32And)
            // Convert i32 (0 or 1) to f32
            .unop(F32ConvertSI32);

        let func_id = builder.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", func_id);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    // Read input 0: mode configuration (CBOR)
    let mode_data = read_input(0);
    let mode_config: ModeConfig = if mode_data.is_empty() {
        ModeConfig::default()
    } else {
        let mut cursor = std::io::Cursor::new(&mode_data);
        ciborium::de::from_reader::<ModeConfig, _>(&mut cursor).unwrap_or_default()
    };

    // Read input 1: vector_a (VecF64(3) - raw bytes, 8 bytes per f64 little-endian)
    // Default to [-1, -1, -1] for a unit cube centered at origin
    let vector_a_data = read_input(1);
    let mut vector_a = decode_vec3(&vector_a_data, [-1.0, -1.0, -1.0]);

    // Read input 2: vector_b (VecF64(3) - raw bytes, 8 bytes per f64 little-endian)
    // Default to [1, 1, 1] for a unit cube centered at origin
    let vector_b_data = read_input(2);
    let mut vector_b = decode_vec3(&vector_b_data, [1.0, 1.0, 1.0]);

    // If both vectors are zero (or identical), use sensible defaults
    // This handles the case where the UI provides default [0,0,0] values
    let is_zero_vec = |v: &[f64; 3]| v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0;
    if is_zero_vec(&vector_a) && is_zero_vec(&vector_b) {
        vector_a = [-1.0, -1.0, -1.0];
        vector_b = [1.0, 1.0, 1.0];
    }

    let output = match generate_wasm(&mode_config.mode, vector_a, vector_b) {
        Ok(wasm) => wasm,
        Err(_) => {
            // Return an empty WASM module on error
            Module::default().emit_wasm()
        }
    };
    unsafe { post_output(0, output.as_ptr() as i32, output.len() as i32) };
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let schema = r#"{ mode: "opposite_corners" / "position_size" .default "opposite_corners" }"#.to_string();
        let metadata = OperatorMetadata {
            name: "rectangular_prism_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::CBORConfiguration(schema),
                OperatorMetadataInput::VecF64(3), // vector_a
                OperatorMetadataInput::VecF64(3), // vector_b
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };
        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out)
            .expect("rectangular_prism_operator metadata CBOR serialization should not fail");
        out
    });
    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}
