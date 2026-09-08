//! Translate operator.
//!
//! Host/operator ABI: see the `volumetric_abi` crate; wrapper mechanics:
//! `model_wrap_core`.
//!
//! Wraps the input model so every query point is shifted back by
//! (dx, dy, dz) before sampling and the bounds are shifted forward. The
//! input's memory, dimensionality and sample format pass through.
//!
//! Dimension-adaptive: only the spatial prefix min(dims, 3) moves — a 2D
//! sketch uses dx/dy (dz is ignored), higher dimensions pass through.

use model_wrap_core::{Affine, Wrapper};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
struct TranslateConfig {
    dx: f64,
    dy: f64,
    dz: f64,
}

fn transform_wasm(input_bytes: &[u8], cfg: &TranslateConfig) -> Result<Vec<u8>, String> {
    let mut wrapper = Wrapper::parse(input_bytes)?;
    wrapper.apply_affine(&Affine::translation([cfg.dx, cfg.dy, cfg.dz]))?;
    Ok(wrapper.finish())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);
    let cfg_buf = read_input(1);
    let cfg = if cfg_buf.is_empty() {
        TranslateConfig::default()
    } else {
        match ciborium::de::from_reader(std::io::Cursor::new(&cfg_buf)) {
            Ok(cfg) => cfg,
            Err(e) => {
                report_error(&format!("invalid configuration: {e}"));
                return;
            }
        }
    };
    match transform_wasm(&buf, &cfg) {
        Ok(output) => post_output(0, &output),
        Err(e) => report_error(&format!("transform failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ dx: float .default 0.0, dy: float .default 0.0, dz: float .default 0.0 }"
            .to_string();
        OperatorMetadata {
            name: "translate_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: String::new(),
            display_name: "Translate".to_string(),
            description: "Move a model by a configurable (dx, dy, dz) offset.".to_string(),
            category: "Transforms".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M12 2v20"/>"##,
                r##"<path d="M2 12h20"/>"##,
                r##"<path d="m9 5 3-3 3 3"/>"##,
                r##"<path d="m9 19 3 3 3-3"/>"##,
                r##"<path d="m5 9-3 3 3 3"/>"##,
                r##"<path d="m19 9 3 3-3 3"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
            output_names: vec![],
        }
    })
}
