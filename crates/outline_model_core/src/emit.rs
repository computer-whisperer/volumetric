//! Turn an outline payload into a standalone model by patching it into a
//! prebuilt `outline_model_template` (feature `emit`; operators only — the
//! template itself must not pull walrus in). Shared by every operator that
//! bakes contours: `text_model_operator`, `html_card_operator`,
//! `path_sketch_operator`.

use model_wrap_core::const_i32_return;
/// Patch `payload` (from [`crate::build_payload`]) into `template`, the
/// `outline_model_template` binary the calling operator embeds. The payload
/// lands in freshly reserved memory pages, its base address is written into
/// the template's `outline_payload_slot`, and that helper export is dropped
/// (it is not part of the Model ABI). Returns the finished model module.
pub fn patch_template(template: &[u8], payload: &[u8]) -> Result<Vec<u8>, String> {
    let config = walrus::ModuleConfig::new();
    let mut module = walrus::Module::from_buffer_with_config(template, &config)
        .map_err(|e| format!("failed to parse the embedded template: {e}"))?;

    let memory_id = module
        .exports
        .iter()
        .find(|e| e.name == "memory")
        .and_then(|e| match e.item {
            walrus::ExportItem::Memory(m) => Some(m),
            _ => None,
        })
        .ok_or("template missing memory export")?;

    let slot_export = module
        .exports
        .iter()
        .find(|e| e.name == "outline_payload_slot")
        .map(|e| (e.id(), e.item))
        .ok_or("template missing outline_payload_slot export")?;
    let slot_addr = match slot_export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(&module, f)
            .ok_or("template outline_payload_slot is not a constant function")?,
        _ => return Err("template outline_payload_slot is not a function".to_string()),
    };
    module.exports.delete(slot_export.0);

    // Payload in freshly reserved pages; base address into the slot.
    let base = {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65536;
        memory.initial += (payload.len() as u64).div_ceil(65536);
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
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
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(slot_addr)),
        },
        (base as u32).to_le_bytes().to_vec(),
    );

    Ok(module.emit_wasm())
}
