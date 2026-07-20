//! Turn a patched `sdf_model_template` into a complete standalone model
//! (feature `emit`; operators only — the template itself must not pull
//! walrus in). Shared by `offset_operator` and `sweep_operator`, which
//! both bake an ndfield payload whose zero crossing IS the model surface
//! and differ only in how they compute the values.
//!
//! The emitted model: the payload is patched into the template's
//! `sdf_payload_slot`, `sdf_sample` is wrapped by a generated `sample`
//! that classifies the field's zero crossing to canonical 1.0/0.0
//! occupancy, constant `get_dimensions`/`get_bounds` are generated, and
//! the template's channel-format export is dropped (the model is
//! occupancy-only).

fn const_i32_return(module: &walrus::Module, function: walrus::FunctionId) -> Option<i32> {
    let local = match &module.funcs.get(function).kind {
        walrus::FunctionKind::Local(local) => local,
        _ => return None,
    };
    let block = local.block(local.entry_block());
    match block.instrs.as_slice() {
        [(walrus::ir::Instr::Const(constant), _)] => match constant.value {
            walrus::ir::Value::I32(value) => Some(value),
            _ => None,
        },
        _ => None,
    }
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

/// Patch `payload` into `template` (an `sdf_model_template` binary) and
/// rewire its exports into a complete standalone model advertising
/// `out_bounds`. See the module docs for the emitted surface.
pub fn emit_field_model(
    template: &[u8],
    payload: &[u8],
    dimensions: usize,
    out_bounds: &[f64],
) -> Result<Vec<u8>, String> {
    if out_bounds.len() != 2 * dimensions {
        return Err(format!(
            "expected {} bounds values, got {}",
            2 * dimensions,
            out_bounds.len()
        ));
    }
    let mut module =
        walrus::Module::from_buffer_with_config(template, &walrus::ModuleConfig::new())
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

    // Payload into freshly reserved pages; base address into the slot.
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

    // sdf_sample has the model `sample` signature and reads its position
    // through the pointer argument, but it returns the raw interpolated
    // field; the ABI classifies occupancy against OCCUPANCY_THRESHOLD
    // (0.5), so `sample` wraps it to the canonical 1.0/0.0 at the
    // field's zero crossing. The template's occupancy+tsdf channel
    // declaration does not describe this model, so the format export
    // goes away entirely (exportless models default to occupancy-only).
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
            "get_sample_format" => exports_to_delete.push(export.id()),
            _ => {}
        }
    }
    for id in exports_to_delete {
        module.exports.delete(id);
    }
    let field_function = field_function.ok_or("field template missing sdf_sample export")?;

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
        .binop(walrus::ir::BinaryOp::F32Gt)
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
