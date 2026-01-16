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
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply non-uniform scaling (sx, sy, sz)
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, Module, ModuleConfig, ValType};

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
extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

const ABI_FUNCTIONS: &[&str] = &[
    "is_inside",
    "get_bounds_min_x",
    "get_bounds_min_y",
    "get_bounds_min_z",
    "get_bounds_max_x",
    "get_bounds_max_y",
    "get_bounds_max_z",
];

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

    let mut renamed: std::collections::HashMap<String, FunctionId> = std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();
    for export in module.exports.iter() {
        if ABI_FUNCTIONS.contains(&export.name.as_str()) {
            if let walrus::ExportItem::Function(func_id) = export.item {
                renamed.insert(export.name.clone(), func_id);
                exports_to_remove.push(export.id());
            }
        }
    }
    for id in exports_to_remove { module.exports.delete(id); }
    for (name, func_id) in &renamed {
        let new_name = format!("{}_{}", name, suffix);
        module.funcs.get_mut(*func_id).name = Some(new_name);
    }

    // is_inside(x,y,z): call original with (x/sx, y/sy, z/sz)
    if let Some(&orig) = renamed.get("is_inside") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::F64, ValType::F64, ValType::F64], &[ValType::F32]);
        let x = module.locals.add(ValType::F64);
        let y = module.locals.add(ValType::F64);
        let z = module.locals.add(ValType::F64);
        use walrus::ir::BinaryOp::F64Div;
        b.func_body().local_get(x).f64_const(cfg.sx as f64).binop(F64Div)
            .local_get(y).f64_const(cfg.sy as f64).binop(F64Div)
            .local_get(z).f64_const(cfg.sz as f64).binop(F64Div)
            .call(orig);
        let fid = b.finish(vec![x, y, z], &mut module.funcs);
        module.exports.add("is_inside", fid);
    }

    // Bounds: new min_x = min(sx*min_x, sx*max_x), new max_x = max(sx*min_x, sx*max_x), similarly for y,z
    use walrus::ir::BinaryOp::{F32Mul, F32Min, F32Max};
    let axes = [("x", cfg.sx), ("y", cfg.sy), ("z", cfg.sz)];
    for (i, (axis, s)) in axes.iter().enumerate() {
        let (min_name, max_name) = match i { 0 => ("get_bounds_min_x", "get_bounds_max_x"), 1 => ("get_bounds_min_y", "get_bounds_max_y"), _ => ("get_bounds_min_z", "get_bounds_max_z") };
        if let (Some(&min_orig), Some(&max_orig)) = (renamed.get(min_name), renamed.get(max_name)) {
            // min
            let mut bmin = FunctionBuilder::new(&mut module.types, &[], &[ValType::F32]);
            bmin.func_body().call(min_orig).f32_const(*s).binop(F32Mul);
            bmin.func_body().call(max_orig).f32_const(*s).binop(F32Mul);
            bmin.func_body().binop(F32Min);
            let fid_min = bmin.finish(vec![], &mut module.funcs);
            module.exports.add(min_name, fid_min);

            // max
            let mut bmax = FunctionBuilder::new(&mut module.types, &[], &[ValType::F32]);
            bmax.func_body().call(min_orig).f32_const(*s).binop(F32Mul);
            bmax.func_body().call(max_orig).f32_const(*s).binop(F32Mul);
            bmax.func_body().binop(F32Max);
            let fid_max = bmax.finish(vec![], &mut module.funcs);
            module.exports.add(max_name, fid_max);
        }
    }

    Ok(module.emit_wasm())
}

#[no_mangle]
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

#[no_mangle]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let schema = "{ sx: float, sy: float, sz: float }".to_string();
        let metadata = OperatorMetadata {
            name: "scale_operator".to_string(),
            version: "0.1.0".to_string(),
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
