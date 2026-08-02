//! HTML Card Operator.
//!
//! Renders an HTML fragment styled with Tailwind utility classes as two
//! filled 2D outline models: the **ink** (output 0 — text, borders, dark
//! panels, with light-on-dark knockout) and the **plate** (output 1 — the
//! card's rounded outline). Extrude the plate, extrude the ink a little
//! higher and union for an embossed label — or subtract the ink for
//! engraving. Both outputs share one coordinate frame (card centered on
//! the origin), so they stack in registration.
//!
//! All layout work happens in `html_card_core` at conversion time —
//! flexbox/grid/block via taffy, real Inter text metrics with wrapping and
//! kerning, Bézier-exact glyph flattening — and the contours bake into
//! `outline_model_template` copies exactly like `text_model_operator`.
//! Colors reduce to monochrome ink: white/transparent is paper, light
//! shades are paper for backgrounds but print for text and borders, dark
//! shades are ink; light-on-dark knocks out. Unsupported tags and classes
//! are hard errors listing every offender (a documented handful of purely
//! decorative classes — shadows, transitions — are accepted no-ops).
//!
//! Inputs:
//! - Input 0: Blob — the HTML fragment (UTF-8).
//! - Input 1: CBOR config `{ width, width_px }` — the card's model-space
//!   width and its CSS-pixel layout width (default 384, Tailwind's
//!   `w-96`). Height follows from the content in both spaces.
//!
//! Outputs: 0 = Ink (ModelWASM, 2D), 1 = Plate (ModelWASM, 2D).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p outline_model_template
//! cp target/wasm32-unknown-unknown/release/outline_model_template.wasm \
//!    crates/operators/html_card_operator/template/
//! ```

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
use walrus::{FunctionId, Module, ModuleConfig};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/outline_model_template.wasm");

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct CardConfig {
    /// Model-space card width; height follows the content aspect.
    width: f64,
    /// CSS-pixel layout width the HTML is laid out at.
    width_px: f64,
}

impl Default for CardConfig {
    fn default() -> Self {
        Self {
            width: 1.0,
            width_px: 384.0,
        }
    }
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

fn patch_template(payload: &[u8]) -> Result<Vec<u8>, String> {
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

fn generate(html: &str, cfg: &CardConfig) -> Result<(Vec<u8>, Vec<u8>), String> {
    if !(cfg.width > 0.0 && cfg.width.is_finite()) {
        return Err(format!("width must be > 0, got {}", cfg.width));
    }
    if !(cfg.width_px >= 16.0 && cfg.width_px.is_finite()) {
        return Err(format!("width_px must be >= 16, got {}", cfg.width_px));
    }

    let geometry = html_card_core::render_card(html, cfg.width_px)?;
    if geometry.ink.is_empty() {
        return Err(
            "the card renders no ink — nothing dark to print (add text, borders, or dark panels)"
                .to_string(),
        );
    }
    let ink = html_card_core::to_model_space(&geometry.ink, &geometry, cfg.width);
    let plate = html_card_core::to_model_space(&geometry.plate, &geometry, cfg.width);
    let ink_wasm = patch_template(&outline_model_core::build_payload(&ink)?)?;
    let plate_wasm = patch_template(&outline_model_core::build_payload(&plate)?)?;
    Ok((ink_wasm, plate_wasm))
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let html_bytes = read_input(0);
    if html_bytes.is_empty() {
        report_error("no HTML connected — input 0 takes the card's HTML fragment as a Blob");
        return;
    }
    let html = match String::from_utf8(html_bytes) {
        Ok(html) => html,
        Err(e) => {
            report_error(&format!("the HTML input is not valid UTF-8: {e}"));
            return;
        }
    };
    let cfg = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            CardConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<CardConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    match generate(&html, &cfg) {
        Ok((ink, plate)) => {
            post_output(0, &ink);
            post_output(1, &plate);
        }
        Err(e) => report_error(&format!("html card generation failed:\n{e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ width: float .default 1.0, width_px: float .default 384.0 }".to_string();
        OperatorMetadata {
            name: "html_card_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: String::new(),
            display_name: "HTML Card".to_string(),
            description:
                "Render an HTML+Tailwind fragment as ink and plate 2D models for label cards."
                    .to_string(),
            category: "Primitives".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<rect x="3" y="4" width="18" height="16" rx="2"/>"##,
                r##"<path d="M7 9h6"/>"##,
                r##"<path d="M7 13h10"/>"##,
                r##"<path d="M7 17h4"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["HTML".to_string(), "Config".to_string()],
            outputs: vec![
                OperatorMetadataOutput::ModelWASM,
                OperatorMetadataOutput::ModelWASM,
            ],
            output_names: vec!["Ink".to_string(), "Plate".to_string()],
        }
    })
}
