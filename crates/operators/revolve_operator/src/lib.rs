//! Revolve operator: spins a profile model around an axis subspace,
//! producing a solid of revolution one dimension higher.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! The input profile is a k-dimensional model (`get_dimensions() -> k`),
//! e.g. a 2D Lua sketch. Its first coordinate is the radial distance from
//! the axis; the remaining k-1 coordinates run along the axis. The
//! optional Subspace input (input 1) is that axis: it must have rank k-1
//! in (k+1)-space — for the classic 2D profile, a line in 3-space:
//!
//! ```text
//! sample(p) = profile_sample(r, a_1..a_{k-1})
//!   where a_j = dot(p - origin, basis_j)
//!         r   = |p - origin - sum(a_j * basis_j)|
//! ```
//!
//! The radius is always >= 0, so only the first-coordinate >= 0 half of
//! the profile contributes (the standard profile-beside-the-axis
//! convention). When the Subspace input is unwired the profile must be 2D
//! and the axis defaults to the world z axis through the origin — the
//! profile's x is radial, its y is world z.
//!
//! Like the transform operators, this is pure module surgery (no host
//! sampling imports). Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Returns k + 1
//! - `get_io_ptr() -> i32`: A fresh buffer in a newly reserved memory page —
//!   the profile's own buffer only holds 2k f64s, the output needs 2(k+1)
//! - `get_bounds(out_ptr: i32)`: per world axis, the axis-chart interval
//!   embedded by sign-picked interval arithmetic, widened by R * alpha
//!   where R = max(profile max radius, 0) and alpha is that world axis's
//!   component orthogonal to the axis subspace (a generation-time constant)
//! - `sample(pos_ptr: i32) -> f32`: rewrites the position to
//!   (r, a_1..a_{k-1}) in scratch, then calls the profile's sample
//! - `sample_channels(pos_ptr, out_ptr)`: same rewrite; the channel row
//!   passes through
//! - `get_sample_format` and `memory`: passed through — revolving changes
//!   where samples are taken, not what they mean
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads an optional Subspace from input 1 (the axis)
//! - Without an axis, rejects non-2D profiles
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::subspace::{Subspace, decode_subspace};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

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

/// Emit the world-to-profile rewrite shared by the `sample` and
/// `sample_channels` wrappers: read the (k+1) world coordinates at
/// `pos_ptr` and write (r, a_1..a_{k-1}) at `scratch`. The basis is
/// orthonormal, so r^2 = |p - origin|^2 - sum(a_j^2).
fn emit_world_to_profile(
    builder: &mut FunctionBuilder,
    module_locals: &mut walrus::ModuleLocals,
    pos_ptr: walrus::LocalId,
    memory_id: MemoryId,
    axis: &Subspace,
    scratch: i32,
) {
    use walrus::ir::BinaryOp::{F64Add, F64Max, F64Mul, F64Sub};
    use walrus::ir::UnaryOp::F64Sqrt;
    let (n, m) = (axis.ambient(), axis.rank());

    // a_j = dot(p, basis_j) - dot(basis_j, origin), stored past the radius.
    for j in 0..m {
        let basis = axis.basis_vector(j);
        let center: f64 = basis.iter().zip(&axis.origin).map(|(b, o)| b * o).sum();
        let mut body = builder.func_body();
        body.i32_const(scratch);
        body.f64_const(-center);
        for (i, &b) in basis.iter().enumerate() {
            if b == 0.0 {
                continue;
            }
            body.local_get(pos_ptr)
                .load(memory_id, walrus::ir::LoadKind::F64, f64_mem(i * 8))
                .f64_const(b)
                .binop(F64Mul)
                .binop(F64Add);
        }
        body.store(memory_id, walrus::ir::StoreKind::F64, f64_mem((j + 1) * 8));
    }

    // r = sqrt(max(0, |p - origin|^2 - sum(a_j^2))) — the max guards the
    // tiny negatives floating-point cancellation can leave behind.
    let t = module_locals.add(ValType::F64);
    let mut body = builder.func_body();
    body.i32_const(scratch);
    for i in 0..n {
        body.local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, f64_mem(i * 8));
        if axis.origin[i] != 0.0 {
            body.f64_const(axis.origin[i]).binop(F64Sub);
        }
        body.local_tee(t).local_get(t).binop(F64Mul);
        if i > 0 {
            body.binop(F64Add);
        }
    }
    for j in 0..m {
        body.i32_const(0)
            .load(
                memory_id,
                walrus::ir::LoadKind::F64,
                f64_mem(scratch as usize + (j + 1) * 8),
            )
            .i32_const(0)
            .load(
                memory_id,
                walrus::ir::LoadKind::F64,
                f64_mem(scratch as usize + (j + 1) * 8),
            )
            .binop(F64Mul)
            .binop(F64Sub);
    }
    body.f64_const(0.0).binop(F64Max).unop(F64Sqrt).store(
        memory_id,
        walrus::ir::StoreKind::F64,
        f64_mem(0),
    );
}

/// Wrap the profile model in revolution glue for `axis`.
fn transform_wasm(input_bytes: &[u8], axis: Option<&Subspace>) -> Result<Vec<u8>, String> {
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

    let default_axis;
    let axis = match axis {
        Some(axis) => {
            if axis.rank() + 1 != k || axis.ambient() != k + 1 {
                return Err(format!(
                    "the axis must have rank {} in {}-space to revolve a {k}-dimensional \
                     profile (radius + {} along-axis coordinates); got rank {} in {}-space",
                    k - 1,
                    k + 1,
                    k - 1,
                    axis.rank(),
                    axis.ambient()
                ));
            }
            axis
        }
        None => {
            // Without an axis the input must be a 2D sketch: the implicit
            // world z axis has no analogue in other dimensionalities.
            if k != 2 {
                return Err(format!(
                    "revolve input must be a 2D sketch, got a {k}-dimensional model \
                     (wire an axis Subspace to revolve other dimensionalities)"
                ));
            }
            default_axis = Subspace::axis_aligned(vec![0.0; 3], &[2]).expect("static axis");
            &default_axis
        }
    };
    let n = k + 1;
    let m = axis.rank();

    // A fresh page past everything the model owns: the output's 2n-f64 IO
    // buffer at its base, then scratch for the k-f64 profile position and
    // the profile's 2k-f64 bounds.
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

    // sample: rewrite to (r, a_1..) in scratch, evaluate there.
    {
        let original_sample_id = renamed["sample"];
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        emit_world_to_profile(
            &mut builder,
            &mut module.locals,
            pos_ptr,
            memory_id,
            axis,
            scratch_pos,
        );
        builder
            .func_body()
            .i32_const(scratch_pos)
            .call(original_sample_id);
        let wrapper_id = builder.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", wrapper_id);
    }

    // sample_channels: identical rewrite; the channel row passes through.
    if let Some(&original_channels_id) = renamed.get("sample_channels") {
        let mut builder =
            FunctionBuilder::new(&mut module.types, &[ValType::I32, ValType::I32], &[]);
        let pos_ptr = module.locals.add(ValType::I32);
        let out_ptr = module.locals.add(ValType::I32);
        emit_world_to_profile(
            &mut builder,
            &mut module.locals,
            pos_ptr,
            memory_id,
            axis,
            scratch_pos,
        );
        builder
            .func_body()
            .i32_const(scratch_pos)
            .local_get(out_ptr)
            .call(original_channels_id);
        let wrapper_id = builder.finish(vec![pos_ptr, out_ptr], &mut module.funcs);
        module.exports.add("sample_channels", wrapper_id);
    }

    // get_bounds: per world axis i, the along-axis chart interval embeds by
    // sign-picked interval arithmetic (as in the slice operator), then
    // widens by R * alpha_i, where R = max(profile max radius, 0) and
    // alpha_i = |e_i orthogonal to the axis subspace| — the largest radial
    // reach along that world axis, a generation-time constant.
    {
        use walrus::ir::BinaryOp::{F64Add, F64Max, F64Mul, F64Sub};
        let original_bounds_id = renamed["get_bounds"];
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        let radius = module.locals.add(ValType::F64);
        builder
            .func_body()
            .i32_const(scratch_bounds)
            .call(original_bounds_id)
            // R = max(profile max radius, 0)
            .i32_const(0)
            .load(
                memory_id,
                walrus::ir::LoadKind::F64,
                f64_mem(scratch_bounds as usize + 8),
            )
            .f64_const(0.0)
            .binop(F64Max)
            .local_set(radius);
        for i in 0..n {
            let along: f64 = (0..m).map(|j| axis.basis_vector(j)[i].powi(2)).sum();
            let alpha = (1.0 - along).max(0.0).sqrt();
            // slot 0: the world minimum; slot 1: the maximum.
            for slot in 0..2 {
                let mut body = builder.func_body();
                body.local_get(out_ptr);
                body.f64_const(axis.origin[i]);
                for j in 0..m {
                    let b = axis.basis_vector(j)[i];
                    if b == 0.0 {
                        continue;
                    }
                    let take_min = (b >= 0.0) == (slot == 0);
                    let offset =
                        scratch_bounds as usize + (j + 1) * 16 + if take_min { 0 } else { 8 };
                    body.i32_const(0)
                        .load(memory_id, walrus::ir::LoadKind::F64, f64_mem(offset))
                        .f64_const(b)
                        .binop(F64Mul)
                        .binop(F64Add);
                }
                if alpha != 0.0 {
                    body.local_get(radius).f64_const(alpha).binop(F64Mul);
                    body.binop(if slot == 0 { F64Sub } else { F64Add });
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

    let axis = {
        let axis_buf = read_input(1);
        if axis_buf.is_empty() {
            None
        } else {
            match decode_subspace(&axis_buf) {
                Ok(subspace) => Some(subspace),
                Err(e) => {
                    report_error(&format!("input 1 is not a usable subspace: {e}"));
                    return;
                }
            }
        }
    };

    match transform_wasm(&buf, axis.as_ref()) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("revolve failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "revolve_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        inputs: vec![
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::Subspace,
        ],
        input_names: vec!["Profile".to_string(), "Axis".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}
