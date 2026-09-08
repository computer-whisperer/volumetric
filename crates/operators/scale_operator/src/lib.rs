//! Scale operator.
//!
//! Host/operator ABI: see the `volumetric_abi` crate; wrapper mechanics:
//! `model_wrap_core`.
//!
//! Wraps the input model so every query point is divided by the per-axis
//! factors before sampling and the bounds are multiplied by them (negative
//! factors reflect; min/max swap accordingly). The input's memory,
//! dimensionality and sample format pass through.
//!
//! Dimension-adaptive: only the spatial prefix min(dims, 3) scales — a 2D
//! sketch uses sx/sy (sz is ignored), higher dimensions pass through. A
//! zero factor on a scaled axis is an error.

use model_wrap_core::{Affine, Wrapper};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct ScaleConfig {
    sx: f64,
    sy: f64,
    sz: f64,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            sx: 1.0,
            sy: 1.0,
            sz: 1.0,
        }
    }
}

fn transform_wasm(input_bytes: &[u8], cfg: &ScaleConfig) -> Result<Vec<u8>, String> {
    let mut wrapper = Wrapper::parse(input_bytes)?;
    wrapper.apply_affine(&Affine::scaling([cfg.sx, cfg.sy, cfg.sz]))?;
    Ok(wrapper.finish())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);
    let cfg_buf = read_input(1);
    let cfg = if cfg_buf.is_empty() {
        ScaleConfig::default()
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
        let schema = "{ sx: float .default 1.0, sy: float .default 1.0, sz: float .default 1.0 }"
            .to_string();
        OperatorMetadata {
            name: "scale_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: String::new(),
            display_name: "Scale".to_string(),
            description: "Scale a model per axis about the origin.".to_string(),
            category: "Transforms".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M21 3 9 15"/>"##,
                r##"<path d="M16 3h5v5"/>"##,
                r##"<path d="M14 15H9v-5"/>"##,
                r##"<path d="M21 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h6"/>"##,
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
