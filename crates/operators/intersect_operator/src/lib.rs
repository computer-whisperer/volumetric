//! Intersect operator: the affine *meet* of several subspaces — the
//! largest subspace contained in all of them. Two planes meet in a line, a
//! plane and a line in a point. Dual to the Span operator.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Inputs:
//! - 0: CBOR config `{ expect: "any" / "point" / "line" / "plane" /
//!   "frame", tolerance: float }`. `expect` (default `any`) optionally
//!   asserts the meet's rank. `tolerance` (default `1e-9`) is the
//!   world-distance slack for deciding whether the inputs actually meet:
//!   parallel or skew inputs that disagree by more than this fail the
//!   operator instead of returning a bogus subspace.
//! - 1..=5: `Subspace` slots. Unwired slots are absent; meet is
//!   associative, so chain past five. At least one must be wired.
//!
//! The result's chart origin is the meet point nearest the inputs'
//! centroid. All inputs must live in the same ambient space.

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::subspace::{Subspace, decode_subspace, encode_subspace};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Number of `Subspace` input slots. Unwired ones are ignored; meet is
/// associative, so more inputs than this can be chained across nodes.
const SUBSPACE_SLOTS: usize = 5;

/// Optional rank the caller expects the meet to have.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum ExpectRank {
    Any,
    Point,
    Line,
    Plane,
    Frame,
}

impl ExpectRank {
    /// The rank this expectation demands, or `None` for `Any`.
    fn required(self) -> Option<usize> {
        match self {
            ExpectRank::Any => None,
            ExpectRank::Point => Some(0),
            ExpectRank::Line => Some(1),
            ExpectRank::Plane => Some(2),
            ExpectRank::Frame => Some(3),
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct IntersectConfig {
    expect: ExpectRank,
    tolerance: f64,
}

impl Default for IntersectConfig {
    fn default() -> Self {
        Self {
            expect: ExpectRank::Any,
            tolerance: 1e-9,
        }
    }
}

/// Collect the wired subspaces (unwired slots read back empty), decode and
/// validate each.
fn read_parts() -> Result<Vec<Subspace>, String> {
    let mut parts = Vec::new();
    for idx in 1..=SUBSPACE_SLOTS {
        let bytes = read_input(idx as i32);
        if bytes.is_empty() {
            continue;
        }
        let subspace = decode_subspace(&bytes)
            .map_err(|e| format!("input {idx} is not a usable subspace: {e}"))?;
        parts.push(subspace);
    }
    Ok(parts)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg: IntersectConfig = {
        let buf = read_input(0);
        if buf.is_empty() {
            IntersectConfig::default()
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

    let result = read_parts().and_then(|parts| {
        let subspace = Subspace::intersect(&parts, cfg.tolerance)?;
        if let Some(want) = cfg.expect.required()
            && subspace.rank() != want
        {
            return Err(format!(
                "meet has rank {} but was asserted to be {:?} (rank {want})",
                subspace.rank(),
                cfg.expect
            ));
        }
        Ok(subspace)
    });

    match result {
        Ok(subspace) => post_output(0, &encode_subspace(&subspace)),
        Err(e) => report_error(&format!("intersect failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ expect: "any" / "point" / "line" / "plane" / "frame" .default "any", tolerance: float .default 1e-9 }"#.to_string();
        let mut inputs = vec![OperatorMetadataInput::CBORConfiguration(schema)];
        let mut input_names = vec!["Config".to_string()];
        for i in 1..=SUBSPACE_SLOTS {
            inputs.push(OperatorMetadataInput::Subspace);
            input_names.push(format!("Subspace {i}"));
        }
        OperatorMetadata {
            name: "intersect_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Intersect".to_string(),
            description:
                "Affine meet of subspaces: the largest subspace inside them all (plane ∩ plane → line)."
                    .to_string(),
            category: "Construction".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M4 4 20 20"/>"##,
                r##"<path d="M20 4 4 20"/>"##,
                r##"<circle cx="12" cy="12" r="2"/>"##,
            )
            .to_string(),
            inputs,
            input_names,
            outputs: vec![OperatorMetadataOutput::Subspace],
        }
    })
}
