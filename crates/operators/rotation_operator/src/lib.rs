//! Rotation operator (Euler angles, degrees).
//!
//! Host/operator ABI: see the `volumetric_abi` crate; wrapper mechanics:
//! `model_wrap_core`.
//!
//! Wraps the input model so every query point is rotated back by the
//! inverse of `Rz * Ry * Rx` (X applied first, then Y, then Z, about the
//! origin) before sampling, and the bounds box is rotated forward and
//! re-enclosed. The input's memory, dimensionality and sample format pass
//! through.
//!
//! Dimension-adaptive: a 2D sketch rotates in-plane about z using rz only —
//! nonzero rx/ry on a 2D input is an error, not silently dropped. 3D+
//! inputs rotate their first three dimensions.

use model_wrap_core::{Affine, Wrapper};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
struct RotationConfig {
    /// Degrees about X, Y, Z applied in order Rx -> Ry -> Rz.
    rx_deg: f64,
    ry_deg: f64,
    rz_deg: f64,
}

fn transform_wasm(input_bytes: &[u8], cfg: &RotationConfig) -> Result<Vec<u8>, String> {
    let mut wrapper = Wrapper::parse(input_bytes)?;
    if wrapper.dims < 2 {
        return Err(format!(
            "rotation needs at least 2 dimensions, input model has {}",
            wrapper.dims
        ));
    }
    let map = Affine::euler_deg(cfg.rx_deg, cfg.ry_deg, cfg.rz_deg);
    if wrapper.spatial() == 2 && !map.preserves_prefix(2) {
        return Err(format!(
            "2D models rotate in-plane only: rx and ry must be 0 (got rx={}, ry={})",
            cfg.rx_deg, cfg.ry_deg
        ));
    }
    wrapper.apply_affine(&map)?;
    Ok(wrapper.finish())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);
    let cfg_buf = read_input(1);
    let cfg = if cfg_buf.is_empty() {
        RotationConfig::default()
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
        let schema = "{ rx_deg: float .default 0.0, ry_deg: float .default 0.0, rz_deg: float .default 0.0 }".to_string();
        OperatorMetadata {
            name: "rotation_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: String::new(),
            display_name: "Rotation".to_string(),
            description: "Rotate a model by Euler angles in degrees.".to_string(),
            category: "Transforms".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>"##,
                r##"<path d="M21 3v5h-5"/>"##,
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
