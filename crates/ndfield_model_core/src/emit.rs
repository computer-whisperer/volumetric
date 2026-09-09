//! Turn the `sdf_model_template` evaluator into a complete standalone
//! model (feature `emit`; operators only — the template itself must not
//! pull walrus in). Shared by `offset_operator` and `sweep_operator`, which
//! bake an ndfield payload whose zero crossing IS the model surface and
//! differ only in how they compute the values, and by
//! `volume_import_operator`, whose payload is a signed distance read from
//! a file. `sdf_operator` takes only the patching step ([`patch_payload`])
//! and merges the result with its source model itself.
//!
//! The template binary is checked in beside this crate, the one copy
//! every operator embeds; regenerate it after changing
//! `sdf_model_template` or the read side of this crate:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p sdf_model_template
//! cp target/wasm32-unknown-unknown/release/sdf_model_template.wasm \
//!    crates/ndfield_model_core/template/
//! ```
//!
//! The emitted model: the payload is patched into the template's
//! `sdf_payload_slot`, `sdf_sample` is wrapped by a generated `sample`
//! that classifies the field's zero crossing to canonical 1.0/0.0
//! occupancy, and constant `get_dimensions`/`get_bounds` are generated.
//! What the field's sign means, and whether the field is also exposed as
//! the template's `signed_distance` channel, is the caller's
//! [`FieldSign`].

use model_wrap_core::const_i32_return;

static TEMPLATE: &[u8] = include_bytes!("../template/sdf_model_template.wasm");

/// What the payload's values mean, which fixes how the emitted model
/// classifies occupancy and which channels it declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldSign {
    /// Positive values are inside (offset and sweep bake `distance -
    /// tsdf`). The model is occupancy-only: the template's channel-format
    /// export is dropped, since it describes a signed distance.
    PositiveInside,
    /// A truncated signed distance: negative inside, positive outside,
    /// magnitudes clamped to the band. The template's occupancy +
    /// `signed_distance` format is kept and a `sample_channels` export is
    /// generated, so the model looks exactly like `sdf_operator` output.
    SignedDistance,
}

fn take_slot_export(module: &mut walrus::Module, name: &str) -> Result<i32, String> {
    let export = module
        .exports
        .iter()
        .find(|export| export.name == name)
        .map(|export| (export.id(), export.item))
        .ok_or_else(|| format!("field template missing {name} export"))?;
    let address = match export.1 {
        walrus::ExportItem::Function(function) => const_i32_return(module, function)
            .ok_or_else(|| format!("field template {name} is not a constant function"))?,
        _ => return Err(format!("field template {name} is not a function")),
    };
    module.exports.delete(export.0);
    Ok(address)
}

/// The template with `payload` placed in freshly reserved memory pages
/// and its base address written into the (removed) `sdf_payload_slot`.
fn patched_template(payload: &[u8]) -> Result<(walrus::Module, walrus::MemoryId), String> {
    let mut module =
        walrus::Module::from_buffer_with_config(TEMPLATE, &walrus::ModuleConfig::new())
            .map_err(|error| format!("failed to parse embedded field template: {error}"))?;
    let memory_id = module
        .exports
        .iter()
        .find(|export| export.name == "memory")
        .and_then(|export| match export.item {
            walrus::ExportItem::Memory(memory) => Some(memory),
            _ => None,
        })
        .ok_or("field template missing memory export")?;
    let payload_slot = take_slot_export(&mut module, "sdf_payload_slot")?;
    let base = {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65_536;
        memory.initial += (payload.len() as u64).div_ceil(65_536);
        if let Some(maximum) = memory.maximum {
            memory.maximum = Some(maximum.max(memory.initial));
        }
        base
    };
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(base as i32)),
        },
        payload.to_vec(),
    );
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(payload_slot)),
        },
        (base as u32).to_le_bytes().to_vec(),
    );
    Ok((module, memory_id))
}

/// The template with `payload` patched in, still exporting the template's
/// raw surface (`sdf_sample`, `get_sample_format`, `get_io_ptr`, `memory`)
/// for a caller that merges it into a larger model.
pub fn patch_payload(payload: &[u8]) -> Result<Vec<u8>, String> {
    let (mut module, _) = patched_template(payload)?;
    Ok(module.emit_wasm())
}

/// Patch `payload` into the template and rewire its exports into a
/// complete standalone model advertising `out_bounds`. See the module
/// docs for the emitted surface.
pub fn emit_field_model(
    payload: &[u8],
    dimensions: usize,
    out_bounds: &[f64],
    sign: FieldSign,
) -> Result<Vec<u8>, String> {
    if out_bounds.len() != 2 * dimensions {
        return Err(format!(
            "expected {} bounds values, got {}",
            2 * dimensions,
            out_bounds.len()
        ));
    }
    let (mut module, memory_id) = patched_template(payload)?;

    // sdf_sample has the model `sample` signature and reads its position
    // through the pointer argument, but it returns the raw interpolated
    // field; the ABI classifies occupancy against OCCUPANCY_THRESHOLD
    // (0.5), so `sample` wraps it to the canonical 1.0/0.0 at the
    // field's zero crossing. The template's occupancy+tsdf channel
    // declaration describes a signed distance only, so it goes away for a
    // positive-inside field (exportless models default to occupancy-only).
    let mut field_function = None;
    let mut exports_to_delete = Vec::new();
    for export in module.exports.iter() {
        match export.name.as_str() {
            "sdf_sample" => {
                if let walrus::ExportItem::Function(function) = export.item {
                    field_function = Some(function);
                }
                exports_to_delete.push(export.id());
            }
            "get_sample_format" if sign == FieldSign::PositiveInside => {
                exports_to_delete.push(export.id())
            }
            _ => {}
        }
    }
    for id in exports_to_delete {
        module.exports.delete(id);
    }
    let field_function = field_function.ok_or("field template missing sdf_sample export")?;
    let inside_op = match sign {
        FieldSign::PositiveInside => walrus::ir::BinaryOp::F32Gt,
        FieldSign::SignedDistance => walrus::ir::BinaryOp::F32Lt,
    };

    let mut sample_builder = walrus::FunctionBuilder::new(
        &mut module.types,
        &[walrus::ValType::I32],
        &[walrus::ValType::F32],
    );
    let pos_ptr = module.locals.add(walrus::ValType::I32);
    sample_builder
        .func_body()
        .local_get(pos_ptr)
        .call(field_function)
        .f32_const(0.0)
        .binop(inside_op)
        .if_else(
            walrus::ValType::F32,
            |then| {
                then.f32_const(1.0);
            },
            |otherwise| {
                otherwise.f32_const(0.0);
            },
        );
    let sample_function = sample_builder.finish(vec![pos_ptr], &mut module.funcs);
    module.exports.add("sample", sample_function);

    if sign == FieldSign::SignedDistance {
        // sample_channels(pos_ptr, out_ptr): channel 0 occupancy, channel 1
        // the field itself, both f32 at out_ptr.
        let mut channels_builder = walrus::FunctionBuilder::new(
            &mut module.types,
            &[walrus::ValType::I32, walrus::ValType::I32],
            &[],
        );
        let pos_ptr = module.locals.add(walrus::ValType::I32);
        let out_ptr = module.locals.add(walrus::ValType::I32);
        let value = module.locals.add(walrus::ValType::F32);
        let f32_at = |offset: u64| walrus::ir::MemArg { align: 2, offset };
        channels_builder
            .func_body()
            .local_get(pos_ptr)
            .call(field_function)
            .local_set(value)
            .local_get(out_ptr)
            .local_get(value)
            .f32_const(0.0)
            .binop(inside_op)
            .if_else(
                walrus::ValType::F32,
                |then| {
                    then.f32_const(1.0);
                },
                |otherwise| {
                    otherwise.f32_const(0.0);
                },
            )
            .store(memory_id, walrus::ir::StoreKind::F32, f32_at(0))
            .local_get(out_ptr)
            .local_get(value)
            .store(memory_id, walrus::ir::StoreKind::F32, f32_at(4));
        let channels_function = channels_builder.finish(vec![pos_ptr, out_ptr], &mut module.funcs);
        module.exports.add("sample_channels", channels_function);
    }

    let mut dims_builder =
        walrus::FunctionBuilder::new(&mut module.types, &[], &[walrus::ValType::I32]);
    dims_builder.func_body().i32_const(dimensions as i32);
    let dims_function = dims_builder.finish(Vec::new(), &mut module.funcs);
    module.exports.add("get_dimensions", dims_function);

    let mut bounds_builder =
        walrus::FunctionBuilder::new(&mut module.types, &[walrus::ValType::I32], &[]);
    let out_ptr = module.locals.add(walrus::ValType::I32);
    for (index, &bound) in out_bounds.iter().enumerate() {
        bounds_builder
            .func_body()
            .local_get(out_ptr)
            .f64_const(bound)
            .store(
                memory_id,
                walrus::ir::StoreKind::F64,
                walrus::ir::MemArg {
                    align: 3,
                    offset: (index * 8) as u64,
                },
            );
    }
    let bounds_function = bounds_builder.finish(vec![out_ptr], &mut module.funcs);
    module.exports.add("get_bounds", bounds_function);

    Ok(module.emit_wasm())
}
