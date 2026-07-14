//! Slice operator: re-expresses a model in a subspace's chart, producing
//! a k-dimensional model from an n-dimensional one.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! The output model's coordinates are the subspace's chart coordinates:
//! `sample(u_1..u_k)` evaluates the input model at
//! `origin + sum(u_i * basis_i)`. The chart is orthonormal, so chart
//! distances are world distances. By rank:
//! - rank 2 in 3-space: the classic planar cross-section, a 2D model.
//! - rank 1: a 1D profile along a line.
//! - rank n (a full frame): a rigid re-expression of the whole model in
//!   the frame's coordinates (the inverse pose transform).
//! - rank 0 is rejected — the Model ABI has no 0-dimensional models.
//!
//! Like the transform operators, this is pure module surgery (no host
//! sampling imports): the wrapped ABI exports are removed and replaced
//! with glue that maps positions through the chart. The input model's
//! IO buffer holds `2 * n >= 2 * k` f64s, so `get_io_ptr` passes through;
//! the chart-to-world expansion writes into a freshly reserved memory
//! page (a k-dim caller only guarantees k f64s at `pos_ptr`, so the glue
//! must not write n f64s in place).
//!
//! `get_bounds` maps the input's box through the inverse chart with
//! interval arithmetic: per output axis the extreme picks the input's
//! min or max slot by the sign of each basis component — exact for the
//! axis-aligned case and a tight enclosure for tilted charts.
//!
//! `get_sample_format` and the channel row pass through untouched: a
//! slice changes where samples are taken, not what they mean.

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::subspace::{Subspace, decode_subspace};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// N-dimensional ABI function names we need to find. `get_io_ptr` and
/// `get_sample_format` pass through; the rest get wrappers.
const ABI_FUNCTIONS_ND: &[&str] = &[
    "get_dimensions",
    "get_io_ptr",
    "get_bounds",
    "sample",
    "sample_channels",
];

/// ABI functions the wrappers replace; the rest keep their exports.
const WRAPPED_FUNCTIONS: &[&str] = &["get_dimensions", "get_bounds", "sample", "sample_channels"];

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

/// Emit the chart-to-world expansion shared by the `sample` and
/// `sample_channels` wrappers: read k chart coordinates at `pos_ptr`,
/// write the n world coordinates `origin + sum(u_i * basis_i)` at
/// `scratch` (a reserved region — never in place; see the module docs).
fn emit_chart_to_world(
    builder: &mut FunctionBuilder,
    pos_ptr: walrus::LocalId,
    memory_id: MemoryId,
    subspace: &Subspace,
    scratch: i32,
) {
    let (n, k) = (subspace.ambient(), subspace.rank());
    for j in 0..n {
        let mut body = builder.func_body();
        // Address for the store, then the value expression on the stack.
        body.i32_const(scratch);
        body.f64_const(subspace.origin[j]);
        for i in 0..k {
            let b = subspace.basis_vector(i)[j];
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
}

/// Wrap the model in chart-mapping glue for `subspace`.
fn transform_wasm(input_bytes: &[u8], subspace: &Subspace) -> Result<Vec<u8>, String> {
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
        if ABI_FUNCTIONS_ND.contains(&export.name.as_str())
            && let walrus::ExportItem::Function(func_id) = export.item
        {
            renamed.insert(export.name.clone(), func_id);
            exports_to_remove.push(export.id());
        }
    }
    if !renamed.contains_key("get_io_ptr") {
        return Err(
            "input model missing `get_io_ptr` export; rebuild it against the \
             current N-dimensional ABI"
                .to_string(),
        );
    }

    let dims_func = renamed
        .get("get_dimensions")
        .ok_or("input model missing `get_dimensions` export")?;
    let n = const_i32_return(&module, *dims_func).ok_or(
        "cannot determine input model dimensionality (get_dimensions is not a constant function)",
    )?;
    if n < 1 {
        return Err(format!("input model reports invalid dimensionality {n}"));
    }
    if subspace.ambient() != n as usize {
        return Err(format!(
            "subspace lives in {}-space but the model has {n} dimensions",
            subspace.ambient()
        ));
    }
    let k = subspace.rank();

    // Scratch for the n-f64 world position and the input's 2n-f64 bounds:
    // a fresh page past everything the model owns.
    let scratch = {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65536;
        memory.initial += 1;
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
        base as i32
    };

    for export_id in exports_to_remove {
        let export_name = module.exports.get(export_id).name.clone();
        if WRAPPED_FUNCTIONS.contains(&export_name.as_str()) {
            module.exports.delete(export_id);
        }
    }

    // get_dimensions: the chart's rank.
    {
        let mut builder = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        builder.func_body().i32_const(k as i32);
        let wrapper_id = builder.finish(vec![], &mut module.funcs);
        module.exports.add("get_dimensions", wrapper_id);
    }

    // sample: expand the chart position into scratch, evaluate there.
    if let Some(&original_sample_id) = renamed.get("sample") {
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        emit_chart_to_world(&mut builder, pos_ptr, memory_id, subspace, scratch);
        builder
            .func_body()
            .i32_const(scratch)
            .call(original_sample_id);
        let wrapper_id = builder.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", wrapper_id);
    }

    // sample_channels: identical expansion; the channel row passes through.
    if let Some(&original_channels_id) = renamed.get("sample_channels") {
        let mut builder =
            FunctionBuilder::new(&mut module.types, &[ValType::I32, ValType::I32], &[]);
        let pos_ptr = module.locals.add(ValType::I32);
        let out_ptr = module.locals.add(ValType::I32);
        emit_chart_to_world(&mut builder, pos_ptr, memory_id, subspace, scratch);
        builder
            .func_body()
            .i32_const(scratch)
            .local_get(out_ptr)
            .call(original_channels_id);
        let wrapper_id = builder.finish(vec![pos_ptr, out_ptr], &mut module.funcs);
        module.exports.add("sample_channels", wrapper_id);
    }

    // get_bounds: the input's box through the inverse chart, by interval
    // arithmetic. chart_i(world) = sum_j(b_ij * world_j) - dot(b_i, origin);
    // its extreme over the box picks world_j's min or max slot by the sign
    // of b_ij — resolved here at generation time, so the glue is straight
    // loads and multiply-adds.
    if let Some(&original_bounds_id) = renamed.get("get_bounds") {
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        builder
            .func_body()
            .i32_const(scratch)
            .call(original_bounds_id);
        for i in 0..k {
            let basis = subspace.basis_vector(i);
            let center: f64 = basis.iter().zip(&subspace.origin).map(|(b, o)| b * o).sum();
            // slot 0: the chart minimum; slot 1: the maximum.
            for slot in 0..2 {
                let mut body = builder.func_body();
                body.local_get(out_ptr);
                body.f64_const(-center);
                for (j, &b) in basis.iter().enumerate() {
                    if b == 0.0 {
                        continue;
                    }
                    let take_min = (b >= 0.0) == (slot == 0);
                    let offset = scratch as usize + j * 16 + if take_min { 0 } else { 8 };
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
    let subspace = match decode_subspace(&read_input(1)) {
        Ok(subspace) => subspace,
        Err(e) => {
            report_error(&format!("input 1 is not a usable subspace: {e}"));
            return;
        }
    };
    if subspace.rank() == 0 {
        report_error(
            "cannot slice on a point: the Model ABI has no 0-dimensional models. \
             Use a line, plane, or frame subspace.",
        );
        return;
    }

    let input = read_input(0);
    match transform_wasm(&input, &subspace) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("slice failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "slice_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Slice".to_string(),
        description: "Re-express a model in a subspace's chart as a lower-dimensional model."
            .to_string(),
        category: "Construction".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<path d="M3 13h18"/>"##,
            r##"<path d="M21 13a9 9 0 0 1-18 0"/>"##,
            r##"<path d="M6 6a8.5 8.5 0 0 1 12 0"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::Subspace,
        ],
        input_names: vec!["Model".to_string(), "Subspace".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}
