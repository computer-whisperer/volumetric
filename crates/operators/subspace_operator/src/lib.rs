//! Subspace operator: builds a [`Subspace`] value in 3-space from numeric
//! inputs — a point, line, plane, or full frame.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Inputs:
//! - 0: CBOR config `{ kind: "point" / "line" / "plane" / "frame" }`.
//! - 1: `origin` (VecF64(3)) — the chart origin.
//! - 2: `primary` (VecF64(3)) — first direction; any non-zero length
//!   (normalized here). Unused for `point`.
//! - 3: `secondary` (VecF64(3)) — second direction; orthonormalized
//!   against `primary` (Gram-Schmidt), so it only needs to be
//!   non-parallel. Used for `plane` and `frame`.
//!
//! A `frame` completes the pair with `primary x secondary`, so the frame
//! is always right-handed. Degenerate directions (zero, or parallel where
//! two are needed) are errors, not silently patched axes.

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::subspace::{Subspace, encode_subspace};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum SubspaceKind {
    Point,
    Line,
    Plane,
    Frame,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct SubspaceConfig {
    kind: SubspaceKind,
}

impl Default for SubspaceConfig {
    fn default() -> Self {
        Self {
            kind: SubspaceKind::Plane,
        }
    }
}

/// Decode a VecF64(3) input (8 bytes per f64, little-endian), with a
/// default for an empty/unwired slot.
fn decode_vec3(data: &[u8], default: [f64; 3]) -> Result<[f64; 3], String> {
    if data.is_empty() {
        return Ok(default);
    }
    if data.len() < 24 {
        return Err(format!(
            "expected 24 bytes of VecF64(3) data, got {}",
            data.len()
        ));
    }
    Ok(std::array::from_fn(|i| {
        f64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap())
    }))
}

fn normalize(v: [f64; 3], what: &str) -> Result<[f64; 3], String> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if !(len.is_finite() && len > 1e-12) {
        return Err(format!("{what} direction {v:?} is degenerate"));
    }
    Ok(v.map(|c| c / len))
}

/// Gram-Schmidt: `v` minus its projection onto unit vector `u`,
/// normalized.
fn orthonormalize(v: [f64; 3], u: [f64; 3], what: &str) -> Result<[f64; 3], String> {
    let dot = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    let rejected = std::array::from_fn(|i| v[i] - dot * u[i]);
    normalize(rejected, what)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn build_subspace(
    kind: SubspaceKind,
    origin: [f64; 3],
    primary: [f64; 3],
    secondary: [f64; 3],
) -> Result<Subspace, String> {
    if let Some(bad) = origin.iter().find(|v| !v.is_finite()) {
        return Err(format!("origin coordinate {bad} is not finite"));
    }
    let mut basis: Vec<[f64; 3]> = Vec::new();
    if kind != SubspaceKind::Point {
        basis.push(normalize(primary, "primary")?);
    }
    if matches!(kind, SubspaceKind::Plane | SubspaceKind::Frame) {
        basis.push(orthonormalize(
            secondary,
            basis[0],
            "secondary (after removing the primary component)",
        )?);
    }
    if kind == SubspaceKind::Frame {
        basis.push(cross(basis[0], basis[1]));
    }
    let subspace = Subspace {
        dimensions: 3,
        origin: origin.to_vec(),
        basis: basis.into_iter().flatten().collect(),
    };
    subspace.validate()?;
    Ok(subspace)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg = {
        let buf = read_input(0);
        if buf.is_empty() {
            SubspaceConfig::default()
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

    let vec_input = |idx: i32, default: [f64; 3], name: &str| {
        decode_vec3(&read_input(idx), default).map_err(|e| format!("{name}: {e}"))
    };
    let result = vec_input(1, [0.0, 0.0, 0.0], "origin").and_then(|origin| {
        let primary = vec_input(2, [1.0, 0.0, 0.0], "primary")?;
        let secondary = vec_input(3, [0.0, 1.0, 0.0], "secondary")?;
        build_subspace(cfg.kind, origin, primary, secondary)
    });
    match result {
        Ok(subspace) => post_output(0, &encode_subspace(&subspace)),
        Err(e) => report_error(&format!("subspace construction failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            r#"{ kind: "point" / "line" / "plane" / "frame" .default "plane" }"#.to_string();
        OperatorMetadata {
            name: "subspace_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Subspace".to_string(),
            description: "Build a point, line, plane, or frame subspace from numeric inputs."
                .to_string(),
            category: "Construction".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="m2 17 6-5h14l-6 5Z"/>"##,
                r##"<path d="M12 13V4"/>"##,
                r##"<path d="m9 7 3-3 3 3"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::CBORConfiguration(schema),
                OperatorMetadataInput::VecF64(3),
                OperatorMetadataInput::VecF64(3),
                OperatorMetadataInput::VecF64(3),
            ],
            input_names: vec![
                "Config".to_string(),
                "Origin".to_string(),
                "Primary".to_string(),
                "Secondary".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::Subspace],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_produce_the_matching_rank() {
        let origin = [1.0, 2.0, 3.0];
        let point = build_subspace(SubspaceKind::Point, origin, [0.0; 3], [0.0; 3]).unwrap();
        assert_eq!(point.rank(), 0);
        assert_eq!(point.origin, origin.to_vec());

        let line = build_subspace(SubspaceKind::Line, origin, [0.0, 0.0, 2.0], [0.0; 3]).unwrap();
        assert_eq!(line.rank(), 1);
        assert_eq!(line.basis, vec![0.0, 0.0, 1.0]);

        let plane = build_subspace(
            SubspaceKind::Plane,
            origin,
            [2.0, 0.0, 0.0],
            // Not orthogonal to primary: Gram-Schmidt straightens it out.
            [1.0, 1.0, 0.0],
        )
        .unwrap();
        assert_eq!(plane.rank(), 2);
        assert_eq!(plane.basis, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(plane.normal().unwrap(), vec![0.0, 0.0, 1.0]);

        let frame = build_subspace(
            SubspaceKind::Frame,
            origin,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
        .unwrap();
        assert_eq!(frame.rank(), 3);
        assert_eq!(&frame.basis[6..], [0.0, 0.0, 1.0], "right-handed");
    }

    #[test]
    fn degenerate_directions_are_rejected() {
        let origin = [0.0; 3];
        assert!(build_subspace(SubspaceKind::Line, origin, [0.0; 3], [0.0; 3]).is_err());
        // Parallel primary/secondary leave nothing for Gram-Schmidt.
        assert!(
            build_subspace(
                SubspaceKind::Plane,
                origin,
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0]
            )
            .is_err()
        );
        assert!(
            build_subspace(
                SubspaceKind::Point,
                [f64::NAN, 0.0, 0.0],
                [0.0; 3],
                [0.0; 3]
            )
            .is_err()
        );
    }

    #[test]
    fn vec3_decoding_defaults_and_validates() {
        assert_eq!(decode_vec3(&[], [1.0, 2.0, 3.0]).unwrap(), [1.0, 2.0, 3.0]);
        let mut bytes = Vec::new();
        for v in [4.0f64, 5.0, 6.0] {
            bytes.extend(v.to_le_bytes());
        }
        assert_eq!(decode_vec3(&bytes, [0.0; 3]).unwrap(), [4.0, 5.0, 6.0]);
        assert!(decode_vec3(&bytes[..16], [0.0; 3]).is_err());
    }
}
