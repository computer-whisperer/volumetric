//! Pose operator: scale and rotate about a pivot, then translate.
//!
//! Host/operator ABI: see the `volumetric_abi` crate; wrapper mechanics:
//! `model_wrap_core`; semantics: README.md (surfaced as the operator's docs).
//!
//! The config's optional blocks compose into one affine map
//! `p' = R S (p - c) + c + t`, applied through `Wrapper::apply_affine`. The
//! `center` pivot reads the input model's bounds through the host at
//! generation time and bakes their centre as a constant.

use model_wrap_core::{Affine, Wrapper};
use volumetric_abi::host::{
    input_model_bounds, input_model_dimensions, post_output, read_input, report_error,
};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum PivotAt {
    Origin,
    Center,
    Point,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct Pivot {
    at: PivotAt,
    px: f64,
    py: f64,
    pz: f64,
}

impl Default for Pivot {
    fn default() -> Self {
        Pivot {
            at: PivotAt::Origin,
            px: 0.0,
            py: 0.0,
            pz: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
struct Rotate {
    rx_deg: f64,
    ry_deg: f64,
    rz_deg: f64,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct Scale {
    sx: f64,
    sy: f64,
    sz: f64,
}

impl Default for Scale {
    fn default() -> Self {
        Scale {
            sx: 1.0,
            sy: 1.0,
            sz: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
struct Translate {
    dx: f64,
    dy: f64,
    dz: f64,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
struct PoseConfig {
    pivot: Option<Pivot>,
    rotate: Option<Rotate>,
    scale: Option<Scale>,
    translate: Option<Translate>,
}

/// The pivot point for a model with `spatial` transformable axes; the
/// `center` choice reads input 0's bounds through the host.
fn pivot_point(pivot: Option<&Pivot>, spatial: usize) -> Result<[f64; 3], String> {
    let Some(pivot) = pivot else {
        return Ok([0.0; 3]);
    };
    match pivot.at {
        PivotAt::Origin => Ok([0.0; 3]),
        PivotAt::Point => Ok([pivot.px, pivot.py, pivot.pz]),
        PivotAt::Center => {
            let dims = input_model_dimensions(0)
                .ok_or("cannot read the input model's dimensions for the center pivot")?;
            let bounds = input_model_bounds(0, dims as usize)
                .ok_or("cannot read the input model's bounds for the center pivot")?;
            let mut center = [0.0; 3];
            for (axis, value) in center.iter_mut().enumerate().take(spatial) {
                *value = 0.5 * (bounds[2 * axis] + bounds[2 * axis + 1]);
            }
            Ok(center)
        }
    }
}

fn transform_wasm(input_bytes: &[u8], cfg: &PoseConfig) -> Result<Vec<u8>, String> {
    let mut wrapper = Wrapper::parse(input_bytes)?;
    let spatial = wrapper.spatial();

    let mut center = pivot_point(cfg.pivot.as_ref(), spatial)?;
    let rotate = cfg.rotate.clone().unwrap_or_default();
    let mut scale = cfg.scale.clone().unwrap_or_default();
    let mut offset = cfg
        .translate
        .as_ref()
        .map_or([0.0; 3], |t| [t.dx, t.dy, t.dz]);
    if spatial == 2 {
        // A sketch poses in-plane: the z components have nothing to act on.
        center[2] = 0.0;
        offset[2] = 0.0;
        scale.sz = 1.0;
    }

    let map = Affine::translation([-center[0], -center[1], -center[2]])
        .then(&Affine::scaling([scale.sx, scale.sy, scale.sz]))
        .then(&Affine::euler_deg(rotate.rx_deg, rotate.ry_deg, rotate.rz_deg))
        .then(&Affine::translation(center))
        .then(&Affine::translation(offset));
    if spatial == 2 && !map.preserves_prefix(2) {
        return Err(format!(
            "2D models pose in-plane only: rx and ry must be 0 (got rx={}, ry={})",
            rotate.rx_deg, rotate.ry_deg
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
        PoseConfig::default()
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
        Err(e) => report_error(&format!("pose failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ ? pivot: { at: "origin" / "center" / "point" .default "origin", px: float .default 0.0, py: float .default 0.0, pz: float .default 0.0 }, ? scale: { sx: float .default 1.0, sy: float .default 1.0, sz: float .default 1.0 }, ? rotate: { rx_deg: float .default 0.0, ry_deg: float .default 0.0, rz_deg: float .default 0.0 }, ? translate: { dx: float .default 0.0, dy: float .default 0.0, dz: float .default 0.0 } }"#.to_string();
        OperatorMetadata {
            name: "pose_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Pose".to_string(),
            description: "Scale and rotate a model about a pivot, then move it, in one step."
                .to_string(),
            category: "Transforms".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M5 3v16h16"/>"##,
                r##"<path d="m5 19 6-6"/>"##,
                r##"<path d="m2 6 3-3 3 3"/>"##,
                r##"<path d="m18 16 3 3-3 3"/>"##,
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
