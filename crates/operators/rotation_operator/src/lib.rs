//! Rotation operator (Euler angles, degrees).
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Passed through from input model
//! - `get_io_ptr() -> i32`: Passed through from input model
//! - `get_bounds(out_ptr: i32)`: Wrapper that applies rotation to bounds AABB
//! - `sample(pos_ptr: i32) -> f32`: Wrapper that applies the inverse rotation
//!   to the position in place (the ABI allows clobbering the position buffer)
//! - `sample_channels(pos_ptr: i32, out_ptr: i32)`: Same wrapper as `sample`,
//!   present iff the input model has it
//! - `get_sample_format() -> i64`: Passed through from input model (a
//!   transform doesn't change what the samples mean)
//! - `memory`: Passed through from input model
//!
//! Dimension-adaptive: the wrapper reads the input's `get_dimensions`
//! constant. A 2D sketch rotates in-plane about z using rz only — nonzero
//! rx/ry on a 2D input is an error, not silently dropped. 3D+ inputs use
//! the full Euler rotation on the first three dimensions.
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply rotation by rx/ry/rz degrees (X then Y then Z)
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct RotationConfig {
    /// Degrees about X, Y, Z applied in order Rx -> Ry -> Rz
    rx_deg: f32,
    ry_deg: f32,
    rz_deg: f32,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            rx_deg: 0.0,
            ry_deg: 0.0,
            rz_deg: 0.0,
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
    let r_inv = [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ];
    let mut r_abs = r;
    for i in 0..3 {
        for j in 0..3 {
            r_abs[i][j] = r_abs[i][j].abs();
        }
    }
    (r, r_inv, r_abs)
}

/// Read the constant a trivial `() -> i32` function returns, if its body is
/// a single `i32.const`. Every model generator emits `get_dimensions` this
/// way, so this is how the operator adapts to the input's dimensionality
/// without being able to instantiate it.
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

/// Emit code that applies the inverse in-plane rotation (about z; the config
/// validated rx = ry = 0, so the matrix's top-left 2x2 block is exactly Rz)
/// to the first 2 dims in place at `pos_ptr`.
fn emit_position_rewrite_2d(
    b: &mut FunctionBuilder,
    locals: &mut walrus::ModuleLocals,
    pos_ptr: walrus::LocalId,
    memory_id: MemoryId,
    r_inv: &[[f64; 3]; 3],
) {
    let x = locals.add(ValType::F64);
    let y = locals.add(ValType::F64);
    let rx = locals.add(ValType::F64);
    let ry = locals.add(ValType::F64);

    use walrus::ir::BinaryOp::{F64Add, F64Mul};
    let mem_arg = walrus::ir::MemArg {
        align: 3,
        offset: 0,
    };
    let mem_arg_8 = walrus::ir::MemArg {
        align: 3,
        offset: 8,
    };

    b.func_body()
        .local_get(pos_ptr)
        .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
        .local_set(x);
    b.func_body()
        .local_get(pos_ptr)
        .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8)
        .local_set(y);

    // (rx, ry) = R2_inv * (x, y)
    for (out, row) in [(rx, &r_inv[0]), (ry, &r_inv[1])] {
        b.func_body()
            .local_get(x)
            .f64_const(row[0])
            .binop(F64Mul)
            .local_get(y)
            .f64_const(row[1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(out);
    }

    b.func_body().local_get(pos_ptr).local_get(rx).store(
        memory_id,
        walrus::ir::StoreKind::F64,
        mem_arg,
    );
    b.func_body().local_get(pos_ptr).local_get(ry).store(
        memory_id,
        walrus::ir::StoreKind::F64,
        mem_arg_8,
    );
}

/// Build the 2D `get_bounds` wrapper: call the original, then transform the
/// two-axis AABB by center/half-extents — c' = R2 c, h' = |R2| h — writing
/// only the input's own 2*2 bounds slots.
fn build_bounds_wrapper_2d(
    module: &mut Module,
    memory_id: MemoryId,
    orig: FunctionId,
    r: &[[f64; 3]; 3],
    r_abs: &[[f64; 3]; 3],
) -> FunctionId {
    let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
    let out_ptr = module.locals.add(ValType::I32);
    let c = [
        module.locals.add(ValType::F64),
        module.locals.add(ValType::F64),
    ];
    let h = [
        module.locals.add(ValType::F64),
        module.locals.add(ValType::F64),
    ];
    let rc = [
        module.locals.add(ValType::F64),
        module.locals.add(ValType::F64),
    ];
    let rh = [
        module.locals.add(ValType::F64),
        module.locals.add(ValType::F64),
    ];

    use walrus::ir::BinaryOp::{F64Add, F64Mul, F64Sub};
    let arg = |offset: u64| walrus::ir::MemArg { align: 3, offset };

    b.func_body().local_get(out_ptr).call(orig);

    // Per axis: c = (min + max) / 2, h = (max - min) / 2
    for axis in 0..2 {
        let (min_off, max_off) = ((axis * 16) as u64, (axis * 16 + 8) as u64);
        b.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, arg(min_off))
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, arg(max_off))
            .binop(F64Add)
            .f64_const(0.5)
            .binop(F64Mul)
            .local_set(c[axis]);
        b.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, arg(max_off))
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, arg(min_off))
            .binop(F64Sub)
            .f64_const(0.5)
            .binop(F64Mul)
            .local_set(h[axis]);
    }

    // rc = R2 * c, rh = |R2| * h
    for axis in 0..2 {
        b.func_body()
            .local_get(c[0])
            .f64_const(r[axis][0])
            .binop(F64Mul)
            .local_get(c[1])
            .f64_const(r[axis][1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(rc[axis]);
        b.func_body()
            .local_get(h[0])
            .f64_const(r_abs[axis][0])
            .binop(F64Mul)
            .local_get(h[1])
            .f64_const(r_abs[axis][1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(rh[axis]);
    }

    // min = rc - rh, max = rc + rh
    for axis in 0..2 {
        let (min_off, max_off) = ((axis * 16) as u64, (axis * 16 + 8) as u64);
        b.func_body()
            .local_get(out_ptr)
            .local_get(rc[axis])
            .local_get(rh[axis])
            .binop(F64Sub)
            .store(memory_id, walrus::ir::StoreKind::F64, arg(min_off));
        b.func_body()
            .local_get(out_ptr)
            .local_get(rc[axis])
            .local_get(rh[axis])
            .binop(F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, arg(max_off));
    }

    b.finish(vec![out_ptr], &mut module.funcs)
}

/// Emit code that applies the inverse rotation to the first 3 dims in place
/// at `pos_ptr` (shared by the `sample` and `sample_channels` wrappers).
fn emit_position_rewrite(
    b: &mut FunctionBuilder,
    locals: &mut walrus::ModuleLocals,
    pos_ptr: walrus::LocalId,
    memory_id: MemoryId,
    r_inv: &[[f64; 3]; 3],
) {
    let x = locals.add(ValType::F64);
    let y = locals.add(ValType::F64);
    let z = locals.add(ValType::F64);
    let rx = locals.add(ValType::F64);
    let ry = locals.add(ValType::F64);
    let rz = locals.add(ValType::F64);

    use walrus::ir::BinaryOp::{F64Add, F64Mul};
    let mem_arg = walrus::ir::MemArg {
        align: 3,
        offset: 0,
    };
    let mem_arg_8 = walrus::ir::MemArg {
        align: 3,
        offset: 8,
    };
    let mem_arg_16 = walrus::ir::MemArg {
        align: 3,
        offset: 16,
    };

    // Load position from input
    b.func_body()
        .local_get(pos_ptr)
        .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
        .local_set(x);
    b.func_body()
        .local_get(pos_ptr)
        .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8)
        .local_set(y);
    b.func_body()
        .local_get(pos_ptr)
        .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16)
        .local_set(z);

    // Apply inverse rotation: (rx, ry, rz) = R_inv * (x, y, z)
    for (out, row) in [(rx, &r_inv[0]), (ry, &r_inv[1]), (rz, &r_inv[2])] {
        b.func_body()
            .local_get(x)
            .f64_const(row[0])
            .binop(F64Mul)
            .local_get(y)
            .f64_const(row[1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_get(z)
            .f64_const(row[2])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(out);
    }

    // Write the rotated position back in place at pos_ptr
    b.func_body().local_get(pos_ptr).local_get(rx).store(
        memory_id,
        walrus::ir::StoreKind::F64,
        mem_arg,
    );
    b.func_body().local_get(pos_ptr).local_get(ry).store(
        memory_id,
        walrus::ir::StoreKind::F64,
        mem_arg_8,
    );
    b.func_body().local_get(pos_ptr).local_get(rz).store(
        memory_id,
        walrus::ir::StoreKind::F64,
        mem_arg_16,
    );
}

fn transform_wasm(input_bytes: &[u8], cfg: RotationConfig) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

    let suffix = generate_hex_suffix();
    let (r, r_inv, r_abs) = rotation_constants(&cfg);

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

    // Find existing ABI functions
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

    // Adapt to the input's dimensionality. A 2D sketch rotates in-plane
    // (about z, using rz only); writing 3 axes into its 2*2-f64 IO buffer
    // would corrupt memory past it.
    let dims_func = renamed
        .get("get_dimensions")
        .ok_or("Input model missing `get_dimensions` export")?;
    let dims = const_i32_return(&module, *dims_func).ok_or(
        "cannot determine input model dimensionality (get_dimensions is not a constant function)",
    )?;
    if dims < 2 {
        return Err(format!(
            "rotation needs at least 2 dimensions, input model has {dims}"
        ));
    }
    let spatial = (dims as usize).min(3);
    if spatial == 2 && (cfg.rx_deg != 0.0 || cfg.ry_deg != 0.0) {
        return Err(format!(
            "2D models rotate in-plane only: rx and ry must be 0 (got rx={}, ry={})",
            cfg.rx_deg, cfg.ry_deg
        ));
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

    // sample wrapper: apply inverse rotation to query point before sampling
    if let Some(&orig) = renamed.get("sample") {
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);

        if spatial == 2 {
            emit_position_rewrite_2d(&mut b, &mut module.locals, pos_ptr, memory_id, &r_inv);
        } else {
            emit_position_rewrite(&mut b, &mut module.locals, pos_ptr, memory_id, &r_inv);
        }
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

        if spatial == 2 {
            emit_position_rewrite_2d(&mut b, &mut module.locals, pos_ptr, memory_id, &r_inv);
        } else {
            emit_position_rewrite(&mut b, &mut module.locals, pos_ptr, memory_id, &r_inv);
        }
        b.func_body()
            .local_get(pos_ptr)
            .local_get(out_ptr)
            .call(orig);

        let fid = b.finish(vec![pos_ptr, out_ptr], &mut module.funcs);
        module.exports.add("sample_channels", fid);
    }

    // get_bounds wrapper (2D): in-plane center/half-extents transform over
    // the input's two bounds slots only.
    if spatial == 2 {
        if let Some(&orig) = renamed.get("get_bounds") {
            let fid = build_bounds_wrapper_2d(&mut module, memory_id, orig, &r, &r_abs);
            module.exports.add("get_bounds", fid);
        }
    }

    // get_bounds wrapper using center/half-extents transform: h' = |R| h, c' = R c
    if spatial >= 3
        && let Some(&orig) = renamed.get("get_bounds")
    {
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

        // Call original get_bounds directly into out_ptr; the original values
        // are loaded into locals below before the rotated AABB overwrites them
        b.func_body().local_get(out_ptr).call(orig);

        // Load original bounds from out_ptr
        let mem_arg_0 = walrus::ir::MemArg {
            align: 3,
            offset: 0,
        };
        let mem_arg_8 = walrus::ir::MemArg {
            align: 3,
            offset: 8,
        };
        let mem_arg_16 = walrus::ir::MemArg {
            align: 3,
            offset: 16,
        };
        let mem_arg_24 = walrus::ir::MemArg {
            align: 3,
            offset: 24,
        };
        let mem_arg_32 = walrus::ir::MemArg {
            align: 3,
            offset: 32,
        };
        let mem_arg_40 = walrus::ir::MemArg {
            align: 3,
            offset: 40,
        };

        b.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_0)
            .local_set(min_x);
        b.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_8)
            .local_set(max_x);
        b.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_16)
            .local_set(min_y);
        b.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_24)
            .local_set(max_y);
        b.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_32)
            .local_set(min_z);
        b.func_body()
            .local_get(out_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg_40)
            .local_set(max_z);

        // Compute center: c = (min + max) / 2
        b.func_body()
            .local_get(min_x)
            .local_get(max_x)
            .binop(F64Add)
            .f64_const(0.5)
            .binop(F64Mul)
            .local_set(cx);
        b.func_body()
            .local_get(min_y)
            .local_get(max_y)
            .binop(F64Add)
            .f64_const(0.5)
            .binop(F64Mul)
            .local_set(cy);
        b.func_body()
            .local_get(min_z)
            .local_get(max_z)
            .binop(F64Add)
            .f64_const(0.5)
            .binop(F64Mul)
            .local_set(cz);

        // Compute half-extents: h = (max - min) / 2
        b.func_body()
            .local_get(max_x)
            .local_get(min_x)
            .binop(F64Sub)
            .f64_const(0.5)
            .binop(F64Mul)
            .local_set(hx);
        b.func_body()
            .local_get(max_y)
            .local_get(min_y)
            .binop(F64Sub)
            .f64_const(0.5)
            .binop(F64Mul)
            .local_set(hy);
        b.func_body()
            .local_get(max_z)
            .local_get(min_z)
            .binop(F64Sub)
            .f64_const(0.5)
            .binop(F64Mul)
            .local_set(hz);

        // Rotate center: c' = R * c
        b.func_body()
            .local_get(cx)
            .f64_const(r[0][0])
            .binop(F64Mul)
            .local_get(cy)
            .f64_const(r[0][1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_get(cz)
            .f64_const(r[0][2])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(rcx);
        b.func_body()
            .local_get(cx)
            .f64_const(r[1][0])
            .binop(F64Mul)
            .local_get(cy)
            .f64_const(r[1][1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_get(cz)
            .f64_const(r[1][2])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(rcy);
        b.func_body()
            .local_get(cx)
            .f64_const(r[2][0])
            .binop(F64Mul)
            .local_get(cy)
            .f64_const(r[2][1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_get(cz)
            .f64_const(r[2][2])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(rcz);

        // Rotate half-extents using |R|: h' = |R| * h
        b.func_body()
            .local_get(hx)
            .f64_const(r_abs[0][0])
            .binop(F64Mul)
            .local_get(hy)
            .f64_const(r_abs[0][1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_get(hz)
            .f64_const(r_abs[0][2])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(rhx);
        b.func_body()
            .local_get(hx)
            .f64_const(r_abs[1][0])
            .binop(F64Mul)
            .local_get(hy)
            .f64_const(r_abs[1][1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_get(hz)
            .f64_const(r_abs[1][2])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(rhy);
        b.func_body()
            .local_get(hx)
            .f64_const(r_abs[2][0])
            .binop(F64Mul)
            .local_get(hy)
            .f64_const(r_abs[2][1])
            .binop(F64Mul)
            .binop(F64Add)
            .local_get(hz)
            .f64_const(r_abs[2][2])
            .binop(F64Mul)
            .binop(F64Add)
            .local_set(rhz);

        // Write rotated bounds to out_ptr: min = c' - h', max = c' + h'
        // min_x = rcx - rhx
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcx)
            .local_get(rhx)
            .binop(F64Sub)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_0);
        // max_x = rcx + rhx
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcx)
            .local_get(rhx)
            .binop(F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_8);
        // min_y = rcy - rhy
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcy)
            .local_get(rhy)
            .binop(F64Sub)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_16);
        // max_y = rcy + rhy
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcy)
            .local_get(rhy)
            .binop(F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_24);
        // min_z = rcz - rhz
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcz)
            .local_get(rhz)
            .binop(F64Sub)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_32);
        // max_z = rcz + rhz
        b.func_body()
            .local_get(out_ptr)
            .local_get(rcz)
            .local_get(rhz)
            .binop(F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg_40);

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
            RotationConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<RotationConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
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
        let schema = "{ rx_deg: float .default 0.0, ry_deg: float .default 0.0, rz_deg: float .default 0.0 }".to_string();
        OperatorMetadata {
            name: "rotation_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Rotation".to_string(),
            description: "Rotate a model by Euler angles in degrees.".to_string(),
            category: "Transforms".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>"##,
                r##"<path d="M21 3v5h-5"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
