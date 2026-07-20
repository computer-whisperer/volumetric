//! STEP Import Operator.
//!
//! Loads an ISO 10303-21 (STEP AP203/AP214) CAD file as a solid model.
//! No tessellation is involved: faces stay exact analytic surfaces
//! (plane/cylinder/cone/sphere/torus/extrusion/NURBS) in a `brep_core`
//! payload, and the emitted model classifies points by ray parity
//! against those surfaces — so downstream meshing resolution is chosen
//! at build time, not import time. The only toleranced data are the
//! trim polylines (and lowered extrusion profiles), controlled by
//! `chord_tol`.
//!
//! The payload is patched into the prebuilt, stateless
//! `brep_model_template` module as data segments — the same pattern as
//! `mesh_to_model_operator` and `image_model_operator`.
//!
//! Inputs:
//! - Input 0: Blob — the STEP file bytes
//! - Input 1: CBOR config:
//!   - `chord_tol` (metres, default 5e-6): trim flattening tolerance
//!   - `scale` (default 1.0): uniform scale applied on top of the
//!     file's declared unit (which is converted to metres — the
//!     engine's canonical unit — automatically)
//!   - `include_labels` / `exclude_labels`: instance filters matched
//!     against reference designators / product names ("J7", "U2")
//!
//! Output 0: ModelWASM (3D).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p brep_model_template
//! cp target/wasm32-unknown-unknown/release/brep_model_template.wasm \
//!    crates/operators/step_import_operator/template/
//! ```

pub mod convert;
pub mod entities;
pub mod p21;

use brep_core::ir::BRepModel;
use walrus::{FunctionId, Module, ModuleConfig};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/brep_model_template.wasm");

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct StepConfig {
    /// Trim-polyline chordal tolerance in metres.
    pub chord_tol: f64,
    /// Extra uniform scale on top of the file's unit.
    pub scale: f64,
    /// Keep only instances with these labels (empty = all).
    pub include_labels: Vec<String>,
    /// Drop instances with these labels.
    pub exclude_labels: Vec<String>,
}

impl Default for StepConfig {
    fn default() -> Self {
        StepConfig {
            chord_tol: 5e-6,
            scale: 1.0,
            include_labels: Vec::new(),
            exclude_labels: Vec::new(),
        }
    }
}

/// STEP bytes -> BRep IR (parse, convert, filter, scale). Native-callable
/// for tests and the oracle harness.
pub fn import(step_text: &str, cfg: &StepConfig) -> Result<BRepModel, String> {
    if !(cfg.chord_tol > 0.0 && cfg.chord_tol.is_finite()) {
        return Err(format!("chord_tol must be > 0, got {}", cfg.chord_tol));
    }
    if !(cfg.scale > 0.0 && cfg.scale.is_finite()) {
        return Err(format!("scale must be > 0, got {}", cfg.scale));
    }
    let data = p21::parse(step_text)?;
    let opts = convert::Options {
        chord_tol: cfg.chord_tol,
    };
    let mut model = convert::build_model(&data, &opts)?;
    convert::filter_instances(&mut model, &cfg.include_labels, &cfg.exclude_labels);
    if model.instances.is_empty() {
        return Err("label filter removed every instance".into());
    }
    if cfg.scale != 1.0 {
        for inst in &mut model.instances {
            let m = &mut inst.local_to_world.0;
            for v in m.iter_mut() {
                *v *= cfg.scale;
            }
        }
        // Scaling the rotation part scales the solid; translations got
        // scaled with it, which is what "scale the whole model about the
        // origin" means.
    }
    Ok(model)
}

/// Read the constant a trivial `() -> i32` function returns.
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

/// Patch a `brep_core` payload into the embedded template.
pub fn patch_template(payload: &[u8]) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(TEMPLATE, &config)
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

    // The patch slot's address, then drop the helper export — it is not
    // part of the Model ABI.
    let slot_export = module
        .exports
        .iter()
        .find(|e| e.name == "brep_payload_slot")
        .map(|e| (e.id(), e.item))
        .ok_or("template missing brep_payload_slot export")?;
    let slot_addr = match slot_export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(&module, f)
            .ok_or("template brep_payload_slot is not a constant function")?,
        _ => return Err("template brep_payload_slot is not a function".to_string()),
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

#[cfg(target_arch = "wasm32")]
mod operator {
    use super::*;
    use volumetric_abi::host::{post_output, read_input, report_error};

    #[unsafe(no_mangle)]
    pub extern "C" fn run() {
        let cfg = {
            let cfg_buf = read_input(1);
            if cfg_buf.is_empty() {
                StepConfig::default()
            } else {
                let mut cursor = std::io::Cursor::new(&cfg_buf);
                match ciborium::de::from_reader::<StepConfig, _>(&mut cursor) {
                    Ok(cfg) => cfg,
                    Err(e) => {
                        report_error(&format!("invalid configuration: {e}"));
                        return;
                    }
                }
            }
        };

        let blob = read_input(0);
        let text = match std::str::from_utf8(&blob) {
            Ok(t) => t,
            Err(_) => {
                report_error("STEP file is not valid UTF-8/ASCII text");
                return;
            }
        };
        let result = import(text, &cfg)
            .and_then(|model| brep_core::payload::build_payload(&model))
            .and_then(|payload| patch_template(&payload));
        match result {
            Ok(wasm) => post_output(0, &wasm),
            Err(e) => report_error(&format!("STEP import failed: {e}")),
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn get_metadata() -> i64 {
        use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
        static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        volumetric_abi::metadata_reply(&METADATA, || {
            let schema = "{ chord_tol: float .default 5e-6, scale: float .default 1.0, \
                          include_labels: [* tstr], exclude_labels: [* tstr] }"
                .to_string();
            OperatorMetadata {
                name: "step_import_operator".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                display_name: "STEP".to_string(),
                description: "Load a STEP CAD file as an exact solid model (no tessellation)."
                    .to_string(),
                category: "Import".to_string(),
                icon_svg: volumetric_abi::icon_svg!(
                    r##"<path d="M12 2 3 7v10l9 5 9-5V7z"/>"##,
                    r##"<path d="M3 7l9 5 9-5"/>"##,
                    r##"<path d="M12 12v10"/>"##,
                )
                .to_string(),
                inputs: vec![
                    OperatorMetadataInput::Blob,
                    OperatorMetadataInput::CBORConfiguration(schema),
                ],
                input_names: vec!["STEP file".to_string(), "Config".to_string()],
                outputs: vec![OperatorMetadataOutput::ModelWASM],
            }
        })
    }
}
