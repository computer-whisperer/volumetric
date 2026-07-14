//! Model bound operator: selects a feature of a model's axis-aligned
//! bounding box — a face plane, an edge line, a corner point, or the
//! centered full frame — as a [`Subspace`] value.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! One [`BoundSelector`] per axis: `span` axes become the subspace's
//! basis directions (origin centered along them), the others pin the
//! origin at that axis's `min` / `max` / `center` bound. The defaults —
//! x/y span, z min — select the bottom face's plane, e.g. the print-bed
//! plane of a model for the brim operator.
//!
//! The model may have 1 to 3 dimensions; a 2D model uses only the `x`
//! and `y` selectors.

use volumetric_abi::host::{
    input_model_bounds, input_model_dimensions, post_output, read_input, report_error,
};
use volumetric_abi::subspace::{BoundSelector, Subspace, encode_subspace};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct BoundConfig {
    x: BoundSelector,
    y: BoundSelector,
    z: BoundSelector,
}

impl Default for BoundConfig {
    fn default() -> Self {
        Self {
            x: BoundSelector::Span,
            y: BoundSelector::Span,
            z: BoundSelector::Min,
        }
    }
}

fn build_bound_subspace(cfg: &BoundConfig) -> Result<Subspace, String> {
    let dims =
        input_model_dimensions(0).ok_or_else(|| "input 0 is not a usable model".to_string())?;
    let dims = dims as usize;
    if dims > 3 {
        return Err(format!(
            "model has {dims} dimensions; selectors exist for at most 3"
        ));
    }
    let bounds =
        input_model_bounds(0, dims).ok_or_else(|| "failed to read model bounds".to_string())?;
    if bounds.iter().any(|b| !b.is_finite()) {
        return Err(format!("model bounds are not finite: {bounds:?}"));
    }
    let selectors = [cfg.x, cfg.y, cfg.z];
    Subspace::from_bounds(&bounds, &selectors[..dims])
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg = {
        let buf = read_input(1);
        if buf.is_empty() {
            BoundConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    match build_bound_subspace(&cfg) {
        Ok(subspace) => post_output(0, &encode_subspace(&subspace)),
        Err(e) => report_error(&format!("bound selection failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let axis = r#""min" / "max" / "center" / "span""#;
        let schema = format!(
            r#"{{ x: {axis} .default "span", y: {axis} .default "span", z: {axis} .default "min" }}"#
        );
        OperatorMetadata {
            name: "model_bound_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Model Bound".to_string(),
            description: "Select a bounding-box feature of a model, face, edge, corner, or frame, as a subspace.".to_string(),
            category: "Construction".to_string(),
            icon_svg: String::new(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::Subspace],
        }
    })
}
