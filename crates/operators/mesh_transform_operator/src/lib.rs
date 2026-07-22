//! Mesh Transform Operator.
//!
//! Rigid/affine transform of an explicit mesh's node positions: scale,
//! then rotate (Euler angles in degrees, X then Y then Z — the same
//! convention as the model `rotation_operator`), then translate, all
//! about the origin (compose with translations for other pivots). Works
//! on any [`FeaMesh`] kind — point clouds, strut networks, volume
//! meshes.
//!
//! Only geometry moves: node and element fields pass through unchanged.
//! In particular 3-component node fields are NOT rotated (a component
//! count doesn't say whether values are spatial vectors or e.g. colors),
//! and a scale does not adjust a `radius` element field — set strut
//! radii where they are produced.
//!
//! Mirroring (negative scale) is allowed for Point1 and Bar2; for Hex8
//! an orientation-reversing scale (odd number of negative axes) is an
//! error, since it inverts element windings and downstream solvers
//! assume positive Jacobians.
//!
//! Inputs:
//! - Input 0: FeaMesh — the mesh to transform
//! - Input 1: CBOR configuration:
//!   `{ tx: float .default 0.0, ty: float .default 0.0, tz: float
//!   .default 0.0, rx_deg: float .default 0.0, ry_deg: float .default
//!   0.0, rz_deg: float .default 0.0, sx: float .default 1.0, sy: float
//!   .default 1.0, sz: float .default 1.0 }`
//!
//! Output 0: the transformed CBOR-encoded `FeaMesh`.

use volumetric_abi::fea::{FeaElementKind, FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct TransformConfig {
    tx: f64,
    ty: f64,
    tz: f64,
    rx_deg: f64,
    ry_deg: f64,
    rz_deg: f64,
    sx: f64,
    sy: f64,
    sz: f64,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            tz: 0.0,
            rx_deg: 0.0,
            ry_deg: 0.0,
            rz_deg: 0.0,
            sx: 1.0,
            sy: 1.0,
            sz: 1.0,
        }
    }
}

/// Apply scale -> rotate (X, Y, Z) -> translate to every node position.
fn transform_mesh(mesh: &FeaMesh, config: &TransformConfig) -> Result<FeaMesh, String> {
    let values = [
        config.tx, config.ty, config.tz, config.rx_deg, config.ry_deg, config.rz_deg, config.sx,
        config.sy, config.sz,
    ];
    if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
        return Err(format!("non-finite transform value {bad}"));
    }
    let scale = [config.sx, config.sy, config.sz];
    if scale.iter().any(|&s| s == 0.0) {
        return Err(format!(
            "zero scale factor ({scale:?}) would flatten the mesh"
        ));
    }
    if mesh.element_kind == FeaElementKind::Hex8 && scale.iter().product::<f64>() < 0.0 {
        return Err(
            "orientation-reversing scale on a Hex8 mesh (odd number of negative \
             factors) would invert element windings"
                .to_string(),
        );
    }

    // Row-major rotation matrix R = Rz * Ry * Rx (X applied first).
    let (sx, cx) = config.rx_deg.to_radians().sin_cos();
    let (sy, cy) = config.ry_deg.to_radians().sin_cos();
    let (sz, cz) = config.rz_deg.to_radians().sin_cos();
    let r = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ];
    let t = [config.tx, config.ty, config.tz];

    let node_positions = mesh
        .node_positions
        .chunks_exact(3)
        .flat_map(|p| {
            let s = [p[0] * scale[0], p[1] * scale[1], p[2] * scale[2]];
            (0..3).map(move |i| r[i][0] * s[0] + r[i][1] * s[1] + r[i][2] * s[2] + t[i])
        })
        .collect();

    let mesh = FeaMesh {
        element_kind: mesh.element_kind,
        node_positions,
        connectivity: mesh.connectivity.clone(),
        node_fields: mesh.node_fields.clone(),
        element_fields: mesh.element_fields.clone(),
    };
    mesh.validate()?;
    Ok(mesh)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(1);
        if buf.is_empty() {
            TransformConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    let result = decode_fea_mesh(&read_input(0)).and_then(|mesh| transform_mesh(&mesh, &config));
    match result {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("mesh transform failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ tx: float .default 0.0, ty: float .default 0.0, tz: float .default 0.0, rx_deg: float .default 0.0, ry_deg: float .default 0.0, rz_deg: float .default 0.0, sx: float .default 1.0, sy: float .default 1.0, sz: float .default 1.0 }"#
            .to_string();
        OperatorMetadata {
            name: "mesh_transform_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Mesh Transform".to_string(),
            description: "Scale, rotate, and translate a mesh's nodes (points, struts, \
                          volumes)."
                .to_string(),
            category: "Mesh".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M5 3v16h16"/>"##,
                r##"<path d="m5 19 6-6"/>"##,
                r##"<path d="m2 6 3-3 3 3"/>"##,
                r##"<path d="m18 16 3 3-3 3"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Mesh".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::fea::FeaField;

    fn cloud(points: &[[f64; 3]]) -> FeaMesh {
        FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: points.iter().flatten().copied().collect(),
            connectivity: (0..points.len() as u32).collect(),
            node_fields: vec![FeaField {
                name: "weight".to_string(),
                components: 1,
                data: (0..points.len()).map(|i| i as f64).collect(),
            }],
            element_fields: vec![],
        }
    }

    #[test]
    fn applies_scale_rotate_translate_in_order() {
        let mesh = cloud(&[[1.0, 0.0, 0.0]]);
        // Scale x2, rotate 90 deg about z, translate +10 y:
        // (1,0,0) -> (2,0,0) -> (0,2,0) -> (0,12,0).
        let config = TransformConfig {
            sx: 2.0,
            rz_deg: 90.0,
            ty: 10.0,
            ..TransformConfig::default()
        };
        let out = transform_mesh(&mesh, &config).unwrap();
        let p = out.node_position(0);
        assert!(p[0].abs() < 1e-12, "{p:?}");
        assert!((p[1] - 12.0).abs() < 1e-12, "{p:?}");
        assert!(p[2].abs() < 1e-12, "{p:?}");
        // Fields ride along unchanged.
        assert_eq!(out.node_fields, mesh.node_fields);
    }

    #[test]
    fn euler_order_is_x_then_y_then_z() {
        // (0,0,1): Rx(90) -> (0,-1,0); Ry(90) leaves y alone -> (0,-1,0);
        // Rz(90) -> (1,0,0).
        let mesh = cloud(&[[0.0, 0.0, 1.0]]);
        let config = TransformConfig {
            rx_deg: 90.0,
            ry_deg: 90.0,
            rz_deg: 90.0,
            ..TransformConfig::default()
        };
        let p = transform_mesh(&mesh, &config).unwrap().node_position(0);
        assert!((p[0] - 1.0).abs() < 1e-12 && p[1].abs() < 1e-12 && p[2].abs() < 1e-12, "{p:?}");
    }

    #[test]
    fn rejects_degenerate_and_winding_inverting_scales() {
        let mesh = cloud(&[[1.0, 1.0, 1.0]]);
        let flat = TransformConfig {
            sy: 0.0,
            ..TransformConfig::default()
        };
        assert!(transform_mesh(&mesh, &flat).is_err());

        // Mirroring a point cloud is fine...
        let mirror = TransformConfig {
            sx: -1.0,
            ..TransformConfig::default()
        };
        assert_eq!(
            transform_mesh(&mesh, &mirror).unwrap().node_position(0),
            [-1.0, 1.0, 1.0]
        );

        // ...but not a Hex8 volume mesh.
        let mut hex = cloud(&[[0.0; 3]; 8]);
        hex.element_kind = FeaElementKind::Hex8;
        hex.connectivity = (0..8).collect();
        hex.node_fields.clear();
        let err = transform_mesh(&hex, &mirror).unwrap_err();
        assert!(err.contains("windings"), "{err}");
    }
}
