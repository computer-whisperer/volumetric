//! Mechanism operator: builds a [`Mechanism`] value from a joint list.
//!
//! Host/operator ABI: see the `volumetric_abi` crate; the value and its
//! kinematics: `volumetric_abi::mechanism`; design: `ASSEMBLY_PLAN.md`.
//!
//! Inputs:
//! - 0: CBOR config: the part names and the joints (schema in the
//!   metadata; semantics in README.md, surfaced as the operator's docs).
//! - 1..: optional Subspace axes (the variadic block): a joint whose
//!   `axis_input` is `k` takes its axis from the k-th of them, a line
//!   (rank 1) giving origin and direction, or a frame's first basis
//!   vector. Joints may instead carry their axis inline.
//!
//! Output 0: the validated Mechanism.

use volumetric_abi::host::{input_count, post_output, read_input, report_error};
use volumetric_abi::mechanism::{Axis, Drive, Joint, JointKind, Mechanism, encode_mechanism};
use volumetric_abi::subspace::decode_subspace;
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
struct AxisConfig {
    origin: [f64; 3],
    direction: [f64; 3],
}

#[derive(Clone, Debug, serde::Deserialize)]
struct JointConfig {
    name: String,
    #[serde(default = "fixed")]
    kind: JointKind,
    #[serde(default = "world")]
    parent: String,
    child: String,
    #[serde(default)]
    axis: Option<AxisConfig>,
    /// Index into the Subspace inputs (the variadic block), when the axis
    /// is routed rather than written inline.
    #[serde(default)]
    axis_input: Option<usize>,
    #[serde(default)]
    min: f64,
    #[serde(default)]
    max: f64,
    #[serde(default)]
    default: f64,
    #[serde(default)]
    drive: Option<Drive>,
    #[serde(default)]
    continuous: bool,
}

fn fixed() -> JointKind {
    JointKind::Fixed
}

fn world() -> String {
    volumetric_abi::mechanism::WORLD.to_string()
}

#[derive(Clone, Debug, serde::Deserialize)]
struct MechanismConfig {
    parts: Vec<String>,
    joints: Vec<JointConfig>,
}

/// The axis a routed Subspace supplies: a line's origin and direction, or
/// a frame's origin and first basis vector.
fn axis_of_subspace(bytes: &[u8], slot: usize) -> Result<Axis, String> {
    if bytes.is_empty() {
        return Err(format!("axis input {slot} is not wired"));
    }
    let subspace = decode_subspace(bytes).map_err(|e| format!("axis input {slot}: {e}"))?;
    if subspace.ambient() != 3 || subspace.rank() < 1 {
        return Err(format!(
            "axis input {slot}: need a line or frame in 3-space, got rank {} in {}-space",
            subspace.rank(),
            subspace.ambient()
        ));
    }
    let d = subspace.basis_vector(0);
    Ok(Axis {
        origin: [subspace.origin[0], subspace.origin[1], subspace.origin[2]],
        direction: [d[0], d[1], d[2]],
    })
}

fn build(cfg: MechanismConfig, axes: &[Vec<u8>]) -> Result<Mechanism, String> {
    let joints = cfg
        .joints
        .into_iter()
        .map(|j| {
            let axis = match (j.axis, j.axis_input) {
                (Some(_), Some(_)) => {
                    return Err(format!(
                        "joint `{}`: give the axis inline or by input, not both",
                        j.name
                    ));
                }
                (Some(a), None) => Some(Axis {
                    origin: a.origin,
                    direction: a.direction,
                }),
                (None, Some(k)) => {
                    let bytes = axes.get(k).ok_or_else(|| {
                        format!(
                            "joint `{}`: axis input {k} but only {} axis inputs are wired",
                            j.name,
                            axes.len()
                        )
                    })?;
                    Some(
                        axis_of_subspace(bytes, k)
                            .map_err(|e| format!("joint `{}`: {e}", j.name))?,
                    )
                }
                (None, None) => None,
            };
            Ok(Joint {
                name: j.name,
                kind: j.kind,
                parent: j.parent,
                child: j.child,
                axis,
                min: j.min,
                max: j.max,
                default: j.default,
                drive: j.drive,
                continuous: j.continuous,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mechanism = Mechanism::new(cfg.parts, joints);
    mechanism.validate()?;
    Ok(mechanism)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let count = input_count().max(1);
    let cfg_buf = read_input(0);
    if cfg_buf.is_empty() {
        report_error("mechanism needs a configuration naming its parts and joints");
        return;
    }
    let cfg: MechanismConfig = match ciborium::de::from_reader(std::io::Cursor::new(&cfg_buf)) {
        Ok(cfg) => cfg,
        Err(e) => {
            report_error(&format!("invalid configuration: {e}"));
            return;
        }
    };
    let axes: Vec<Vec<u8>> = (1..count).map(|idx| read_input(idx as i32)).collect();
    match build(cfg, &axes) {
        Ok(mechanism) => post_output(0, &encode_mechanism(&mechanism)),
        Err(e) => report_error(&format!("mechanism failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ parts: [* tstr], joints: [* { name: tstr, kind: "fixed" / "revolute" / "prismatic" .default "fixed", parent: tstr .default "world", child: tstr, ? axis: { origin: [float, float, float], direction: [float, float, float] }, ? axis_input: uint, min: float .default 0.0, max: float .default 0.0, default: float .default 0.0, continuous: bool .default false, ? drive: { joint: tstr, ratio: float, offset: float .default 0.0 } }] }"#.to_string();
        OperatorMetadata {
            name: "mechanism_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Mechanism".to_string(),
            description: "Parts joined by fixed, revolute and prismatic joints: the kinematics of an assembly.".to_string(),
            category: "Assembly".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<circle cx="6" cy="18" r="2"/>"##,
                r##"<circle cx="12" cy="8" r="2"/>"##,
                r##"<circle cx="19" cy="14" r="2"/>"##,
                r##"<path d="m7.5 16.5 3-7"/>"##,
                r##"<path d="m13.8 9 3.6 3.6"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::CBORConfiguration(schema),
                OperatorMetadataInput::Subspace,
            ],
            variadic_input: Some(1),
            input_names: vec!["Config".to_string(), "Axes".to_string()],
            outputs: vec![OperatorMetadataOutput::Mechanism],
            output_names: vec!["mechanism".to_string()],
        }
    })
}
