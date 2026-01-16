//! Rotation operator (Euler angles, degrees).
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
//! - Emits new wrapper functions that apply rotation by rx/ry/rz degrees (X then Y then Z)
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
struct RotationConfig {
    /// Degrees about X, Y, Z applied in order Rx -> Ry -> Rz
    rx_deg: f32,
    ry_deg: f32,
    rz_deg: f32,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self { rx_deg: 0.0, ry_deg: 0.0, rz_deg: 0.0 }
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
    let mut state: u32 = 0xA5A5_1337;
    let mut result = String::with_capacity(8);
    for _ in 0..8 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let nibble = (state >> 16) & 0xF;
        result.push(char::from_digit(nibble, 16).unwrap());
    }
    result
}

/// Build rotation matrices constants from Euler angles (degrees). Returns (R, R_inv, abs(R))
fn rotation_constants(cfg: &RotationConfig) -> ([[f64; 3]; 3], [[f64; 3]; 3], [[f64; 3]; 3]) {
    let (sx, cx) = (cfg.rx_deg as f64).to_radians().sin_cos();
    let (sy, cy) = (cfg.ry_deg as f64).to_radians().sin_cos();
    let (sz, cz) = (cfg.rz_deg as f64).to_radians().sin_cos();

    // R = Rz * Ry * Rx (apply X then Y then Z to a column vector)
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];

    fn mm(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        r
    }

    let r = mm(mm(rz, ry), rx);
    // For pure rotation, inverse is transpose
    let r_inv = [[r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][1], r[2][1]], [r[0][2], r[1][2], r[2][2]]];
    let mut r_abs = r;
    for i in 0..3 {
        for j in 0..3 {
            r_abs[i][j] = r_abs[i][j].abs();
        }
    }
    (r, r_inv, r_abs)
}

fn transform_wasm(input_bytes: &[u8], cfg: RotationConfig) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

    let suffix = generate_hex_suffix();
    let (r, r_inv, r_abs) = rotation_constants(&cfg);

    // Find existing ABI functions
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

    // Helper to add constant 3x3 multiply for a column vector (x,y,z) using matrix m
    fn emit_mat3_mul(builder: &mut FunctionBuilder, x: walrus::LocalId, y: walrus::LocalId, z: walrus::LocalId, m: [[f64;3];3]) {
        use walrus::ir::BinaryOp::{F64Add, F64Mul};
        // returns on stack: (mx, my, mz)
        // mx
        builder.func_body()
            .local_get(x).f64_const(m[0][0]).binop(F64Mul)
            .local_get(y).f64_const(m[0][1]).binop(F64Mul).binop(F64Add)
            .local_get(z).f64_const(m[0][2]).binop(F64Mul).binop(F64Add);
        // my
        builder.func_body()
            .local_get(x).f64_const(m[1][0]).binop(F64Mul)
            .local_get(y).f64_const(m[1][1]).binop(F64Mul).binop(F64Add)
            .local_get(z).f64_const(m[1][2]).binop(F64Mul).binop(F64Add);
        // mz
        builder.func_body()
            .local_get(x).f64_const(m[2][0]).binop(F64Mul)
            .local_get(y).f64_const(m[2][1]).binop(F64Mul).binop(F64Add)
            .local_get(z).f64_const(m[2][2]).binop(F64Mul).binop(F64Add);
    }

    // is_inside wrapper: apply inverse rotation to query point
    if let Some(&orig) = renamed.get("is_inside") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::F64, ValType::F64, ValType::F64], &[ValType::F32]);
        let x = module.locals.add(ValType::F64);
        let y = module.locals.add(ValType::F64);
        let z = module.locals.add(ValType::F64);
        emit_mat3_mul(&mut b, x, y, z, r_inv);
        // Stack: rx ry rz -> call orig
        b.func_body().call(orig);
        let fid = b.finish(vec![x, y, z], &mut module.funcs);
        module.exports.add("is_inside", fid);
    }

    // Bounds wrappers using center/half-extents transform: h' = |R| h, c' = R c
    // We'll build helpers to compute min/max via calls to original getters.
    fn build_bounds_component(module: &mut Module, renamed: &std::collections::HashMap<String, FunctionId>, r: [[f64;3];3], r_abs: [[f64;3];3], axis: usize, name: &str) {
        use walrus::ir::BinaryOp::*;
        let mut b = FunctionBuilder::new(&mut module.types, &[], &[ValType::F64]);

        // Compute c'_axis = row(axis) · c, where c = (min+max)/2 per axis, all in f64
        // term_x = r[axis][0] * (min_x + max_x) * 0.5
        b.func_body()
            .call(*renamed.get("get_bounds_min_x").unwrap())
            .call(*renamed.get("get_bounds_max_x").unwrap())
            .binop(F64Add)
            .f64_const(0.5)
            .binop(F64Mul)
            .f64_const(r[axis][0])
            .binop(F64Mul);
        // + term_y
        b.func_body()
            .call(*renamed.get("get_bounds_min_y").unwrap())
            .call(*renamed.get("get_bounds_max_y").unwrap())
            .binop(F64Add)
            .f64_const(0.5)
            .binop(F64Mul)
            .f64_const(r[axis][1])
            .binop(F64Mul)
            .binop(F64Add);
        // + term_z
        b.func_body()
            .call(*renamed.get("get_bounds_min_z").unwrap())
            .call(*renamed.get("get_bounds_max_z").unwrap())
            .binop(F64Add)
            .f64_const(0.5)
            .binop(F64Mul)
            .f64_const(r[axis][2])
            .binop(F64Mul)
            .binop(F64Add);

        // Compute h'_axis = |row(axis)| · h, where h = (max-min)/2 per axis, all in f64
        // term_x
        b.func_body()
            .call(*renamed.get("get_bounds_max_x").unwrap())
            .call(*renamed.get("get_bounds_min_x").unwrap())
            .binop(F64Sub)
            .f64_const(0.5)
            .binop(F64Mul)
            .f64_const(r_abs[axis][0])
            .binop(F64Mul);
        // + term_y
        b.func_body()
            .call(*renamed.get("get_bounds_max_y").unwrap())
            .call(*renamed.get("get_bounds_min_y").unwrap())
            .binop(F64Sub)
            .f64_const(0.5)
            .binop(F64Mul)
            .f64_const(r_abs[axis][1])
            .binop(F64Mul)
            .binop(F64Add);
        // + term_z
        b.func_body()
            .call(*renamed.get("get_bounds_max_z").unwrap())
            .call(*renamed.get("get_bounds_min_z").unwrap())
            .binop(F64Sub)
            .f64_const(0.5)
            .binop(F64Mul)
            .f64_const(r_abs[axis][2])
            .binop(F64Mul)
            .binop(F64Add);

        // Now stack: c'_axis (f64), h'_axis (f64). For min or max, compute c' -/+ h'
        match name {
            "min" => { b.func_body().binop(F64Sub); },
            _ => { b.func_body().binop(F64Add); },
        }

        let fid = b.finish(vec![], &mut module.funcs);
        let export_name = match (axis, name) { (0, "min") => "get_bounds_min_x", (0, _) => "get_bounds_max_x", (1, "min") => "get_bounds_min_y", (1, _) => "get_bounds_max_y", (2, "min") => "get_bounds_min_z", (2, _) => "get_bounds_max_z", _ => unreachable!() };
        module.exports.add(export_name, fid);
    }

    for axis in 0..3 {
        build_bounds_component(&mut module, &renamed, r, r_abs, axis, "min");
        build_bounds_component(&mut module, &renamed, r, r_abs, axis, "max");
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
        if cfg_len == 0 { RotationConfig::default() } else {
            let mut cfg_buf = vec![0u8; cfg_len];
            unsafe { get_input_data(1, cfg_buf.as_mut_ptr() as i32, cfg_len as i32); }
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            ciborium::de::from_reader::<RotationConfig, _>(&mut cursor).unwrap_or_default()
        }
    };

    let output = match transform_wasm(&buf, cfg) { Ok(t) => t, Err(_) => buf };
    unsafe { post_output(0, output.as_ptr() as i32, output.len() as i32); }
}

#[no_mangle]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let schema = "{ rx_deg: float, ry_deg: float, rz_deg: float }".to_string();
        let metadata = OperatorMetadata {
            name: "rotation_operator".to_string(),
            version: "0.1.0".to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };
        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out).expect("rotation_operator metadata CBOR serialization should not fail");
        out
    });
    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}
