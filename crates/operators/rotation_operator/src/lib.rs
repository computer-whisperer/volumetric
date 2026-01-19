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
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Passed through from input model
//! - `get_bounds(out_ptr: i32)`: Wrapper that applies rotation to bounds AABB
//! - `sample(pos_ptr: i32) -> f32`: Wrapper that applies inverse rotation to position
//! - `memory`: Passed through from input model
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply rotation by rx/ry/rz degrees (X then Y then Z)
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
unsafe extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

const ABI_FUNCTIONS_ND: &[&str] = &["get_dimensions", "get_bounds", "sample"];
const SCRATCH_POS_OFFSET: i32 = 512;

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

    // Find memory export
    let memory_id: Option<MemoryId> = module.exports.iter()
        .find(|e| e.name == "memory")
        .and_then(|e| if let walrus::ExportItem::Memory(m) = e.item { Some(m) } else { None });

    let memory_id = memory_id.ok_or("Input model missing memory export")?;

    // Find existing ABI functions
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

    // sample wrapper: apply inverse rotation to query point before sampling
    if let Some(&orig) = renamed.get("sample") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let x = module.locals.add(ValType::F64);
        let y = module.locals.add(ValType::F64);
        let z = module.locals.add(ValType::F64);
        let rx = module.locals.add(ValType::F64);
        let ry = module.locals.add(ValType::F64);
        let rz = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::{F64Add, F64Mul};
        let mem_arg = walrus::ir::MemArg { align: 3, offset: 0 };
        let mem_arg_8 = walrus::ir::MemArg { align: 3, offset: 8 };
        let mem_arg_16 = walrus::ir::MemArg { align: 3, offset: 16 };

        // Load position from input
        b.func_body().local_get(pos_ptr).load(memory_id, walrus::ir::LoadKind::F64, mem_arg).local_set(x);
        b.func_body().local_get(pos_ptr).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8).local_set(y);
        b.func_body().local_get(pos_ptr).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16).local_set(z);

        // Apply inverse rotation: (rx, ry, rz) = R_inv * (x, y, z)
        // rx = r_inv[0][0]*x + r_inv[0][1]*y + r_inv[0][2]*z
        b.func_body()
            .local_get(x).f64_const(r_inv[0][0]).binop(F64Mul)
            .local_get(y).f64_const(r_inv[0][1]).binop(F64Mul).binop(F64Add)
            .local_get(z).f64_const(r_inv[0][2]).binop(F64Mul).binop(F64Add)
            .local_set(rx);

        // ry = r_inv[1][0]*x + r_inv[1][1]*y + r_inv[1][2]*z
        b.func_body()
            .local_get(x).f64_const(r_inv[1][0]).binop(F64Mul)
            .local_get(y).f64_const(r_inv[1][1]).binop(F64Mul).binop(F64Add)
            .local_get(z).f64_const(r_inv[1][2]).binop(F64Mul).binop(F64Add)
            .local_set(ry);

        // rz = r_inv[2][0]*x + r_inv[2][1]*y + r_inv[2][2]*z
        b.func_body()
            .local_get(x).f64_const(r_inv[2][0]).binop(F64Mul)
            .local_get(y).f64_const(r_inv[2][1]).binop(F64Mul).binop(F64Add)
            .local_get(z).f64_const(r_inv[2][2]).binop(F64Mul).binop(F64Add)
            .local_set(rz);

        // Write rotated position to scratch buffer
        let scratch_arg = walrus::ir::MemArg { align: 3, offset: 0 };
        let scratch_arg_8 = walrus::ir::MemArg { align: 3, offset: 8 };
        let scratch_arg_16 = walrus::ir::MemArg { align: 3, offset: 16 };

        b.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(rx)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg);

        b.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(ry)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg_8);

        b.func_body()
            .i32_const(SCRATCH_POS_OFFSET)
            .local_get(rz)
            .store(memory_id, walrus::ir::StoreKind::F64, scratch_arg_16);

        // Call original sample with scratch buffer
        b.func_body().i32_const(SCRATCH_POS_OFFSET).call(orig);

        let fid = b.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", fid);
    }

    // get_bounds wrapper using center/half-extents transform: h' = |R| h, c' = R c
    if let Some(&orig) = renamed.get("get_bounds") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);

        // Locals for original bounds
        let min_x = module.locals.add(ValType::F64);
        let max_x = module.locals.add(ValType::F64);
        let min_y = module.locals.add(ValType::F64);
        let max_y = module.locals.add(ValType::F64);
        let min_z = module.locals.add(ValType::F64);
        let max_z = module.locals.add(ValType::F64);

        // Locals for center and half-extents
        let cx = module.locals.add(ValType::F64);
        let cy = module.locals.add(ValType::F64);
        let cz = module.locals.add(ValType::F64);
        let hx = module.locals.add(ValType::F64);
        let hy = module.locals.add(ValType::F64);
        let hz = module.locals.add(ValType::F64);

        // Locals for rotated center and half-extents
        let rcx = module.locals.add(ValType::F64);
        let rcy = module.locals.add(ValType::F64);
        let rcz = module.locals.add(ValType::F64);
        let rhx = module.locals.add(ValType::F64);
        let rhy = module.locals.add(ValType::F64);
        let rhz = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::{F64Add, F64Mul, F64Sub};

        // Call original get_bounds to scratch area
        b.func_body().i32_const(SCRATCH_POS_OFFSET).call(orig);

        // Load original bounds from scratch area
        let mem_arg_0 = walrus::ir::MemArg { align: 3, offset: 0 };
        let mem_arg_8 = walrus::ir::MemArg { align: 3, offset: 8 };
        let mem_arg_16 = walrus::ir::MemArg { align: 3, offset: 16 };
        let mem_arg_24 = walrus::ir::MemArg { align: 3, offset: 24 };
        let mem_arg_32 = walrus::ir::MemArg { align: 3, offset: 32 };
        let mem_arg_40 = walrus::ir::MemArg { align: 3, offset: 40 };

        b.func_body().i32_const(SCRATCH_POS_OFFSET).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_0).local_set(min_x);
        b.func_body().i32_const(SCRATCH_POS_OFFSET).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8).local_set(max_x);
        b.func_body().i32_const(SCRATCH_POS_OFFSET).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16).local_set(min_y);
        b.func_body().i32_const(SCRATCH_POS_OFFSET).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_24).local_set(max_y);
        b.func_body().i32_const(SCRATCH_POS_OFFSET).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_32).local_set(min_z);
        b.func_body().i32_const(SCRATCH_POS_OFFSET).load(memory_id, walrus::ir::LoadKind::F64, mem_arg_40).local_set(max_z);

        // Compute center: c = (min + max) / 2
        b.func_body().local_get(min_x).local_get(max_x).binop(F64Add).f64_const(0.5).binop(F64Mul).local_set(cx);
        b.func_body().local_get(min_y).local_get(max_y).binop(F64Add).f64_const(0.5).binop(F64Mul).local_set(cy);
        b.func_body().local_get(min_z).local_get(max_z).binop(F64Add).f64_const(0.5).binop(F64Mul).local_set(cz);

        // Compute half-extents: h = (max - min) / 2
        b.func_body().local_get(max_x).local_get(min_x).binop(F64Sub).f64_const(0.5).binop(F64Mul).local_set(hx);
        b.func_body().local_get(max_y).local_get(min_y).binop(F64Sub).f64_const(0.5).binop(F64Mul).local_set(hy);
        b.func_body().local_get(max_z).local_get(min_z).binop(F64Sub).f64_const(0.5).binop(F64Mul).local_set(hz);

        // Rotate center: c' = R * c
        b.func_body()
            .local_get(cx).f64_const(r[0][0]).binop(F64Mul)
            .local_get(cy).f64_const(r[0][1]).binop(F64Mul).binop(F64Add)
            .local_get(cz).f64_const(r[0][2]).binop(F64Mul).binop(F64Add)
            .local_set(rcx);
        b.func_body()
            .local_get(cx).f64_const(r[1][0]).binop(F64Mul)
            .local_get(cy).f64_const(r[1][1]).binop(F64Mul).binop(F64Add)
            .local_get(cz).f64_const(r[1][2]).binop(F64Mul).binop(F64Add)
            .local_set(rcy);
        b.func_body()
            .local_get(cx).f64_const(r[2][0]).binop(F64Mul)
            .local_get(cy).f64_const(r[2][1]).binop(F64Mul).binop(F64Add)
            .local_get(cz).f64_const(r[2][2]).binop(F64Mul).binop(F64Add)
            .local_set(rcz);

        // Rotate half-extents using |R|: h' = |R| * h
        b.func_body()
            .local_get(hx).f64_const(r_abs[0][0]).binop(F64Mul)
            .local_get(hy).f64_const(r_abs[0][1]).binop(F64Mul).binop(F64Add)
            .local_get(hz).f64_const(r_abs[0][2]).binop(F64Mul).binop(F64Add)
            .local_set(rhx);
        b.func_body()
            .local_get(hx).f64_const(r_abs[1][0]).binop(F64Mul)
            .local_get(hy).f64_const(r_abs[1][1]).binop(F64Mul).binop(F64Add)
            .local_get(hz).f64_const(r_abs[1][2]).binop(F64Mul).binop(F64Add)
            .local_set(rhy);
        b.func_body()
            .local_get(hx).f64_const(r_abs[2][0]).binop(F64Mul)
            .local_get(hy).f64_const(r_abs[2][1]).binop(F64Mul).binop(F64Add)
            .local_get(hz).f64_const(r_abs[2][2]).binop(F64Mul).binop(F64Add)
            .local_set(rhz);

        // Write rotated bounds to out_ptr: min = c' - h', max = c' + h'
        // min_x = rcx - rhx
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcx).local_get(rhx).binop(F64Sub)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_0);
        // max_x = rcx + rhx
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcx).local_get(rhx).binop(F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_8);
        // min_y = rcy - rhy
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcy).local_get(rhy).binop(F64Sub)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_16);
        // max_y = rcy + rhy
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcy).local_get(rhy).binop(F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_24);
        // min_z = rcz - rhz
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcz).local_get(rhz).binop(F64Sub)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_32);
        // max_z = rcz + rhz
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcz).local_get(rhz).binop(F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_40);

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

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let schema = "{ rx_deg: float .default 0.0, ry_deg: float .default 0.0, rz_deg: float .default 0.0 }".to_string();
        let metadata = OperatorMetadata {
            name: "rotation_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
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
