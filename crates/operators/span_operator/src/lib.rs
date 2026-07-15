//! Span operator: the affine *join* of several subspaces — the smallest
//! subspace containing them all. Two points span a line, three
//! non-collinear points a plane, a line and an off-line point a plane, a
//! plane and an off-plane point the full 3-space.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Inputs:
//! - 0: CBOR config `{ expect: "any" / "point" / "line" / "plane" /
//!   "frame" }` — an optional rank assertion. `any` (the default) returns
//!   the honest span, so three collinear points come back as a line; the
//!   others fail the operator when the span is not that rank, catching
//!   "these points don't actually define a plane" at the source.
//! - 1..=5: `Subspace` slots. Unwired slots are absent; wire as many as
//!   the construction needs (chain Span nodes past five — join is
//!   associative). At least one must be wired.
//!
//! The result's chart origin is the centroid of the wired inputs' origins
//! and its rank is the true dimension of the span. All inputs must live in
//! the same ambient space.

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::subspace::{Subspace, decode_subspace, encode_subspace};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Number of `Subspace` input slots. Unwired ones are ignored; join is
/// associative, so more inputs than this can be chained across nodes.
const SUBSPACE_SLOTS: usize = 5;

/// Optional rank the caller expects the span to reach.
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
struct SpanConfig {
    expect: ExpectRank,
}

impl Default for SpanConfig {
    fn default() -> Self {
        Self {
            expect: ExpectRank::Any,
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
        let subspace =
            decode_subspace(&bytes).map_err(|e| format!("input {idx} is not a usable subspace: {e}"))?;
        parts.push(subspace);
    }
    Ok(parts)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg: SpanConfig = {
        let buf = read_input(0);
        if buf.is_empty() {
            SpanConfig::default()
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
        let subspace = Subspace::span(&parts)?;
        if let Some(want) = cfg.expect.required()
            && subspace.rank() != want
        {
            return Err(format!(
                "span has rank {} but was asserted to be {:?} (rank {want}); \
                 the inputs are degenerate for that shape",
                subspace.rank(),
                cfg.expect
            ));
        }
        Ok(subspace)
    });

    match result {
        Ok(subspace) => post_output(0, &encode_subspace(&subspace)),
        Err(e) => report_error(&format!("span failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            r#"{ expect: "any" / "point" / "line" / "plane" / "frame" .default "any" }"#.to_string();
        let mut inputs = vec![OperatorMetadataInput::CBORConfiguration(schema)];
        let mut input_names = vec!["Config".to_string()];
        for i in 1..=SUBSPACE_SLOTS {
            inputs.push(OperatorMetadataInput::Subspace);
            input_names.push(format!("Subspace {i}"));
        }
        OperatorMetadata {
            name: "span_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Span".to_string(),
            description:
                "Affine join of subspaces: the smallest subspace containing them (points/line → plane)."
                    .to_string(),
            category: "Construction".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M12 3 3 20h18Z"/>"##,
                r##"<circle cx="12" cy="3" r="1.6"/>"##,
                r##"<circle cx="3" cy="20" r="1.6"/>"##,
                r##"<circle cx="21" cy="20" r="1.6"/>"##,
            )
            .to_string(),
            inputs,
            input_names,
            outputs: vec![OperatorMetadataOutput::Subspace],
        }
    })
}
