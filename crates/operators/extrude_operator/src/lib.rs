//! Linear extrude operator: sweeps a profile model along a plane's normal,
//! producing a model one dimension higher.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! The input profile is a k-dimensional model (`get_dimensions() -> k`),
//! e.g. a 2D Lua sketch. The optional Subspace input (input 2) places the
//! profile in space: it must be a hyperplane whose chart matches the
//! profile — rank k in (k+1)-space — and the profile is swept along the
//! plane's oriented normal from `w = 0` to `w = height`:
//!
//! ```text
//! sample(p) = (0 <= w <= height) ? profile_sample(u_1..u_k) : 0.0
//!   where u_j = dot(p - origin, basis_j), w = dot(p - origin, normal)
//! ```
//!
//! To extrude the other way, flip the plane's orientation (swap or negate
//! basis vectors); `height` is always positive. When the Subspace input is
//! unwired the profile must be 2D and the plane defaults to the world xy
//! plane at z = 0 with a +z normal — the classic sketch extrude.
//!
//! Like the transform operators, this is pure module surgery (no host
//! sampling imports). Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Returns k + 1
//! - `get_io_ptr() -> i32`: A fresh buffer in a newly reserved memory page —
//!   the profile's own buffer only holds 2k f64s, the output needs 2(k+1)
//! - `get_bounds(out_ptr: i32)`: the AABB of the profile's chart box swept
//!   through [0, height], by sign-picked interval arithmetic (exact for
//!   axis-aligned planes, a tight enclosure for tilted ones)
//! - `sample(pos_ptr: i32) -> f32`: gates on the normal coordinate, then
//!   evaluates the profile at the chart coordinates (written to scratch —
//!   a k-dim callee may clobber its position buffer)
//! - `sample_channels(pos_ptr, out_ptr)`: same chart mapping; outside the
//!   slab the row is the in-plane sample with occupancy forced to 0
//! - `get_sample_format` and `memory`: passed through — extruding changes
//!   where samples are taken, not what they mean
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1 (height)
//! - Reads an optional Subspace from input 2 (the plane)
//! - Without a plane, rejects non-2D profiles
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::subspace::{Subspace, decode_subspace};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct ExtrudeConfig {
    height: f64,
}

impl Default for ExtrudeConfig {
    fn default() -> Self {
        Self { height: 1.0 }
    }
}

/// N-dimensional ABI function names the wrappers replace. Everything else
/// (`get_sample_format`, `memory`) passes through.
const WRAPPED_FUNCTIONS: &[&str] = &[
    "get_dimensions",
    "get_io_ptr",
    "get_bounds",
    "sample",
    "sample_channels",
];

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

fn f64_mem(offset: usize) -> walrus::ir::MemArg {
    walrus::ir::MemArg {
        align: 3,
        offset: offset as u64,
    }
}

/// Emit the world-to-chart projection shared by the `sample` and
/// `sample_channels` wrappers: read the (k+1) world coordinates at
/// `pos_ptr`, write the k chart coordinates `dot(p - origin, basis_j)`
/// at `scratch`, and leave the normal coordinate `w` in the `w` local.
fn emit_world_to_chart(
    builder: &mut FunctionBuilder,
    pos_ptr: walrus::LocalId,
    w: walrus::LocalId,
    memory_id: MemoryId,
    plane: &Subspace,
    normal: &[f64],
    scratch: i32,
) {
    let k = plane.rank();
    let dot_axis =
        |axis: &[f64]| -> f64 { axis.iter().zip(&plane.origin).map(|(a, o)| a * o).sum() };
    for j in 0..k {
        let basis = plane.basis_vector(j);
        let mut body = builder.func_body();
        body.i32_const(scratch);
        body.f64_const(-dot_axis(basis));
        for (i, &b) in basis.iter().enumerate() {
            if b == 0.0 {
                continue;
            }
            body.local_get(pos_ptr)
                .load(memory_id, walrus::ir::LoadKind::F64, f64_mem(i * 8))
                .f64_const(b)
                .binop(walrus::ir::BinaryOp::F64Mul)
                .binop(walrus::ir::BinaryOp::F64Add);
        }
        body.store(memory_id, walrus::ir::StoreKind::F64, f64_mem(j * 8));
    }
    let mut body = builder.func_body();
    body.f64_const(-dot_axis(normal));
    for (i, &nc) in normal.iter().enumerate() {
        if nc == 0.0 {
            continue;
        }
        body.local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, f64_mem(i * 8))
            .f64_const(nc)
            .binop(walrus::ir::BinaryOp::F64Mul)
            .binop(walrus::ir::BinaryOp::F64Add);
    }
    body.local_set(w);
}

/// Wrap the profile model in extrusion glue for `plane`.
fn transform_wasm(
    input_bytes: &[u8],
    cfg: &ExtrudeConfig,
    plane: Option<&Subspace>,
) -> Result<Vec<u8>, String> {
    if cfg.height <= 0.0 || cfg.height.is_nan() {
        return Err(format!("extrude height must be > 0, got {}", cfg.height));
    }

    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("failed to parse the model WASM: {e}"))?;

    let memory_id: MemoryId = module
        .exports
        .iter()
        .find(|e| e.name == "memory")
        .and_then(|e| match e.item {
            walrus::ExportItem::Memory(m) => Some(m),
            _ => None,
        })
        .ok_or("input model missing memory export")?;

    let mut renamed: std::collections::HashMap<String, FunctionId> =
        std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();
    for export in module.exports.iter() {
        if WRAPPED_FUNCTIONS.contains(&export.name.as_str())
            && let walrus::ExportItem::Function(func_id) = export.item
        {
            renamed.insert(export.name.clone(), func_id);
            exports_to_remove.push(export.id());
        }
    }
    for required in &["get_dimensions", "get_io_ptr", "get_bounds", "sample"] {
        if !renamed.contains_key(*required) {
            return Err(format!("input model missing `{required}` export"));
        }
    }

    let k = const_i32_return(&module, renamed["get_dimensions"]).ok_or(
        "cannot determine input model dimensionality (get_dimensions is not a constant function)",
    )?;
    if k < 1 {
        return Err(format!("input model reports invalid dimensionality {k}"));
    }
    let k = k as usize;

    let default_plane;
    let plane = match plane {
        Some(plane) => {
            if plane.rank() != k || plane.ambient() != k + 1 {
                return Err(format!(
                    "the plane must be a rank-{k} hyperplane in {}-space to carry a \
                     {k}-dimensional profile; got rank {} in {}-space",
                    k + 1,
                    plane.rank(),
                    plane.ambient()
                ));
            }
            plane
        }
        None => {
            // Without a plane the input must be a 2D sketch: extruding a 3D
            // model to 4D is almost certainly a wiring mistake unless a
            // subspace makes the intent explicit.
            if k != 2 {
                return Err(format!(
                    "extrude input must be a 2D sketch, got a {k}-dimensional model \
                     (wire a plane Subspace to extrude other dimensionalities)"
                ));
            }
            default_plane = Subspace::axis_aligned(vec![0.0; 3], &[0, 1]).expect("static plane");
            &default_plane
        }
    };
    let n = k + 1;
    let normal = plane.normal().expect("a rank-k plane in (k+1)-space");

    // A fresh page past everything the model owns: the output's 2n-f64 IO
    // buffer at its base, then scratch for the k-f64 chart position and the
    // profile's 2k-f64 bounds.
    let (new_io_ptr, scratch_pos, scratch_bounds) = {
        let memory = module.memories.get_mut(memory_id);
        let base = (memory.initial * 65536) as i32;
        memory.initial += 1;
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
        (base, base + 512, base + 1024)
    };

    for export_id in exports_to_remove {
        module.exports.delete(export_id);
    }

    // get_dimensions: one above the profile.
    {
        let mut builder = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        builder.func_body().i32_const(n as i32);
        let wrapper_id = builder.finish(vec![], &mut module.funcs);
        module.exports.add("get_dimensions", wrapper_id);
    }

    // get_io_ptr: the new 2n-f64 buffer.
    {
        let mut builder = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        builder.func_body().i32_const(new_io_ptr);
        let wrapper_id = builder.finish(vec![], &mut module.funcs);
        module.exports.add("get_io_ptr", wrapper_id);
    }

    use walrus::ir::BinaryOp::{F64Gt, F64Lt, I32Or};

    // sample: project to the chart, gate on the normal coordinate.
    {
        let original_sample_id = renamed["sample"];
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let w = module.locals.add(ValType::F64);
        emit_world_to_chart(
            &mut builder,
            pos_ptr,
            w,
            memory_id,
            plane,
            &normal,
            scratch_pos,
        );
        builder
            .func_body()
            .local_get(w)
            .f64_const(0.0)
            .binop(F64Lt)
            .local_get(w)
            .f64_const(cfg.height)
            .binop(F64Gt)
            .binop(I32Or)
            .if_else(
                ValType::F32,
                |then| {
                    then.f32_const(0.0);
                },
                |else_| {
                    else_.i32_const(scratch_pos).call(original_sample_id);
                },
            );
        let wrapper_id = builder.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", wrapper_id);
    }

    // sample_channels: same projection; the profile fills the row, then the
    // occupancy slot is forced to 0 outside the slab (the other channels
    // keep their in-plane values — consumers gate on occupancy).
    if let Some(&original_channels_id) = renamed.get("sample_channels") {
        let mut builder =
            FunctionBuilder::new(&mut module.types, &[ValType::I32, ValType::I32], &[]);
        let pos_ptr = module.locals.add(ValType::I32);
        let out_ptr = module.locals.add(ValType::I32);
        let w = module.locals.add(ValType::F64);
        emit_world_to_chart(
            &mut builder,
            pos_ptr,
            w,
            memory_id,
            plane,
            &normal,
            scratch_pos,
        );
        builder
            .func_body()
            .i32_const(scratch_pos)
            .local_get(out_ptr)
            .call(original_channels_id)
            .local_get(w)
            .f64_const(0.0)
            .binop(F64Lt)
            .local_get(w)
            .f64_const(cfg.height)
            .binop(F64Gt)
            .binop(I32Or)
            .if_else(
                walrus::ir::InstrSeqType::Simple(None),
                |then| {
                    then.local_get(out_ptr).f32_const(0.0).store(
                        memory_id,
                        walrus::ir::StoreKind::F32,
                        walrus::ir::MemArg {
                            align: 2,
                            offset: 0,
                        },
                    );
                },
                |_| {},
            );
        let wrapper_id = builder.finish(vec![pos_ptr, out_ptr], &mut module.funcs);
        module.exports.add("sample_channels", wrapper_id);
    }

    // get_bounds: the profile's chart box swept through w in [0, height],
    // embedded per world axis by interval arithmetic — each extreme picks
    // the chart's min or max slot by the sign of the basis component,
    // resolved here at generation time.
    {
        let original_bounds_id = renamed["get_bounds"];
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        builder
            .func_body()
            .i32_const(scratch_bounds)
            .call(original_bounds_id);
        for (i, &nc) in normal.iter().enumerate() {
            // slot 0: the world minimum; slot 1: the maximum.
            for slot in 0..2 {
                let sweep = nc * cfg.height;
                let constant = plane.origin[i]
                    + if slot == 0 {
                        sweep.min(0.0)
                    } else {
                        sweep.max(0.0)
                    };
                let mut body = builder.func_body();
                body.local_get(out_ptr);
                body.f64_const(constant);
                for j in 0..k {
                    let b = plane.basis_vector(j)[i];
                    if b == 0.0 {
                        continue;
                    }
                    let take_min = (b >= 0.0) == (slot == 0);
                    let offset = scratch_bounds as usize + j * 16 + if take_min { 0 } else { 8 };
                    body.i32_const(0)
                        .load(memory_id, walrus::ir::LoadKind::F64, f64_mem(offset))
                        .f64_const(b)
                        .binop(walrus::ir::BinaryOp::F64Mul)
                        .binop(walrus::ir::BinaryOp::F64Add);
                }
                body.store(
                    memory_id,
                    walrus::ir::StoreKind::F64,
                    f64_mem(i * 16 + slot * 8),
                );
            }
        }
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
            ExtrudeConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<ExtrudeConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    let plane = {
        let plane_buf = read_input(2);
        if plane_buf.is_empty() {
            None
        } else {
            match decode_subspace(&plane_buf) {
                Ok(subspace) => Some(subspace),
                Err(e) => {
                    report_error(&format!("input 2 is not a usable subspace: {e}"));
                    return;
                }
            }
        }
    };

    match transform_wasm(&buf, &cfg, plane.as_ref()) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("extrude failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ height: float .default 1.0 }".to_string();
        OperatorMetadata {
            name: "extrude_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
                OperatorMetadataInput::Subspace,
            ],
            input_names: vec![
                "Profile".to_string(),
                "Config".to_string(),
                "Plane".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
