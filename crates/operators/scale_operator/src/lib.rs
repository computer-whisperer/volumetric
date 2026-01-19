//! Scale operator.
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
//! - `get_bounds(out_ptr: i32)`: Wrapper that multiplies bounds by scale factors
//! - `sample(pos_ptr: i32) -> f32`: Wrapper that divides position by scale factors
//! - `memory`: Passed through from input model
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply non-uniform scaling (sx, sy, sz)
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataInput {
    ModelWASM,
    CBORConfiguration(String),
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
struct ScaleConfig {
    sx: f32,
    sy: f32,
    sz: f32,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self { sx: 1.0, sy: 1.0, sz: 1.0 }
    }
}

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

const ABI_FUNCTIONS_ND: &[&str] = &["get_dimensions", "get_bounds", "sample"];
const SCRATCH_POS_OFFSET: i32 = 512;

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

fn transform_wasm(input_bytes: &[u8], cfg: ScaleConfig) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

    let suffix = generate_hex_suffix();

    // Find memory export
    let memory_id: Option<MemoryId> = module.exports.iter()
        .find(|e| e.name == "memory")
        .and_then(|e| if let walrus::ExportItem::Memory(m) = e.item { Some(m) } else { None });

    let memory_id = memory_id.ok_or("Input model missing memory export")?;

    let mut renamed: std::collections::HashMap<String, FunctionId> = std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();
    for export in module.exports.iter() {
        if ABI_FUNCTIONS_ND.contains(&export.name.as_str()) {
            if let walrus::ExportItem::Function(func_id) = export.item {
                renamed.insert(export.name.clone(), func_id);
                exports_to_remove.push(export.id());
            }
        }
    }

    // Remove old exports except get_dimensions
    for id in exports_to_remove {
        let export_name = module.exports.get(id).name.clone();
        if export_name != "get_dimensions" {
            module.exports.delete(id);
        }
    }

    // Rename original functions except get_dimensions
    for (name, func_id) in &renamed {
        if name != "get_dimensions" {
            let new_name = format!("{}_{}", name, suffix);
            module.funcs.get_mut(*func_id).name = Some(new_name);
        }
    }

    // sample(pos_ptr): call original with (x/sx, y/sy, z/sz)
    if let Some(&orig) = renamed.get("sample") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let x = module.locals.add(ValType::F64);
        let y = module.locals.add(ValType::F64);
        let z = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::F64Div;
        let mem_arg = walrus::ir::MemArg { align: 3, offset: 0 };
        let mem_arg_8 = walrus::ir::MemArg { align: 3, offset: 8 };
        let mem_arg_16 = walrus::ir::MemArg { align: 3, offset: 16 };

        // Load position from input
        b.func_body().local_get(pos_ptr).load(memory_id, walrus::ir::LoadKind::F64, mem_arg).local_set(x);
        b.func_body().local_get(pos_ptr).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8).local_set(y);
        b.func_body().local_get(pos_ptr).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16).local_set(z);

        // Write scaled position to scratch buffer
        let scratch_arg = walrus::ir::MemArg { align: 3, offset: 0 };
        let scratch_arg_8 = walrus::ir::MemArg { align: 3, offset: 8 };
        let scratch_arg_16 = walrus::ir::MemArg { align: 3, offset: 16 };

        b.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(x)
            .f64_const(cfg.sx as f64)
            .binop(F64Div)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg);

        b.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(y)
            .f64_const(cfg.sy as f64)
            .binop(F64Div)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg_8);

        b.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(z)
            .f64_const(cfg.sz as f64)
            .binop(F64Div)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg_16);

        // Call original with scratch buffer
        b.func_body().i32_const(SCRATCH_POS_OFFSET).call(orig);

        let fid = b.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", fid);
    }

    // get_bounds(out_ptr): call original, then multiply bounds by scale factors
    // Also swap min/max if scale is negative
    if let Some(&orig) = renamed.get("get_bounds") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        let min_val = module.locals.add(ValType::F64);
        let max_val = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::{F64Mul, F64Min, F64Max};

        // Call original get_bounds
        b.func_body().local_get(out_ptr).call(orig);

        // Process each axis: scale and handle sign flips
        let scales = [(0, 8, cfg.sx as f64), (16, 24, cfg.sy as f64), (32, 40, cfg.sz as f64)];

        for (min_offset, max_offset, scale) in scales {
            let mem_arg_min = walrus::ir::MemArg { align: 3, offset: min_offset };
            let mem_arg_max = walrus::ir::MemArg { align: 3, offset: max_offset };

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
    let len = unsafe { get_input_len(0) } as usize;
    let mut buf = vec![0u8; len];
    if len > 0 { unsafe { get_input_data(0, buf.as_mut_ptr() as i32, len as i32); } }

    let cfg = {
        let cfg_len = unsafe { get_input_len(1) } as usize;
        if cfg_len == 0 { ScaleConfig::default() } else {
            let mut cfg_buf = vec![0u8; cfg_len];
            unsafe { get_input_data(1, cfg_buf.as_mut_ptr() as i32, cfg_len as i32); }
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            ciborium::de::from_reader::<ScaleConfig, _>(&mut cursor).unwrap_or_default()
        }
    };

    let output = match transform_wasm(&buf, cfg) { Ok(t) => t, Err(_) => buf };
    unsafe { post_output(0, output.as_ptr() as i32, output.len() as i32); }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let schema = "{ sx: float .default 1.0, sy: float .default 1.0, sz: float .default 1.0 }".to_string();
        let metadata = OperatorMetadata {
            name: "scale_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };
        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out).expect("scale_operator metadata CBOR serialization should not fail");
        out
    });
    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}
