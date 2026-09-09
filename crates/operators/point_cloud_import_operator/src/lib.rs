//! Point Cloud Import Operator.
//!
//! Reads a point-cloud file (PLY today) and emits a Point1 [`FeaMesh`] —
//! the engine's point-cloud value, consumed by the point operators (clip,
//! transform, Voronoi skeleton, remaster) and rendered as a coloured cloud.
//! See README.md (the operator's docs) for what the reader honours. A PLY
//! with faces imports its vertices alone; `ply_import_operator` reads the
//! mesh.
//!
//! Inputs:
//! - Input 0: Blob — point-cloud file bytes
//! - Input 1: CBOR configuration: `scale` (float, default 1.0), `center`
//!   (bool, default false), `stride` (int ≥ 1, default 1: keep every nth
//!   point), `fields` (vertex property names carried through as
//!   one-component node fields, default none). Placement is center → scale.
//!
//! Output 0: FeaMesh (Point1).

use ply_core::convert::{place, vertices};
use ply_core::{is_ply, read_ply};
use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh, encode_fea_mesh};
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct PointCloudImportConfig {
    pub scale: f64,
    pub center: bool,
    pub stride: u32,
    pub fields: Vec<String>,
}

impl Default for PointCloudImportConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            center: false,
            stride: 1,
            fields: Vec::new(),
        }
    }
}

/// Reads the file as a cloud, returning it with the number of faces the
/// file also carried (ignored here, reported as a warning).
pub fn import(bytes: &[u8], config: &PointCloudImportConfig) -> Result<(FeaMesh, usize), String> {
    if !is_ply(bytes) {
        return Err(
            "unrecognised point-cloud format: PLY is the supported format, and this file \
             does not start with its `ply` magic line"
                .to_string(),
        );
    }
    if config.stride == 0 {
        return Err("stride must be at least 1".to_string());
    }
    let file = read_ply(bytes)?;
    let faces = file.element("face").map_or(0, |f| f.count);
    let mut vertex = vertices(&file, &config.fields)?;
    if config.stride > 1 {
        let stride = config.stride as usize;
        vertex.positions = subsample(&vertex.positions, 3, stride);
        for field in &mut vertex.fields {
            field.data = subsample(&field.data, field.components, stride);
        }
    }
    if vertex.positions.is_empty() {
        return Err("the file has no vertices".to_string());
    }
    place(&mut vertex.positions, config.center, config.scale)?;
    let count = vertex.positions.len() / 3;
    let mesh = FeaMesh {
        element_kind: FeaElementKind::Point1,
        node_positions: vertex.positions,
        connectivity: (0..count as u32).collect(),
        node_fields: vertex.fields.into_iter().map(|f: FeaField| f).collect(),
        element_fields: vec![],
    };
    mesh.validate()?;
    Ok((mesh, faces))
}

/// Every `stride`th entry of an entry-major array with `components` values
/// per entry.
fn subsample(data: &[f64], components: usize, stride: usize) -> Vec<f64> {
    data.chunks_exact(components)
        .step_by(stride)
        .flatten()
        .copied()
        .collect()
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let bytes = read_input(0);
    let config = {
        let cfg = read_input(1);
        if cfg.is_empty() {
            PointCloudImportConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&cfg)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    match import(&bytes, &config) {
        Ok((mesh, faces)) => {
            if faces > 0 {
                post_warning(&format!(
                    "the file also carries {faces} face(s), ignored here; PLY Import reads the mesh"
                ));
            }
            post_output(0, &encode_fea_mesh(&mesh));
        }
        Err(e) => report_error(&format!("point cloud import failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ scale: float .default 1.0, center: bool .default false, stride: int .default 1 .ge 1, fields: [* tstr] }"
            .to_string();
        OperatorMetadata {
            name: "point_cloud_import_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Point Cloud Import".to_string(),
            description:
                "Read a point-cloud file (PLY) as a Point1 cloud with colours and normals."
                    .to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<circle cx="6" cy="7" r="1.5"/>"##,
                r##"<circle cx="12" cy="5" r="1.5"/>"##,
                r##"<circle cx="18" cy="8" r="1.5"/>"##,
                r##"<circle cx="8" cy="14" r="1.5"/>"##,
                r##"<circle cx="15" cy="13" r="1.5"/>"##,
                r##"<circle cx="11" cy="19" r="1.5"/>"##,
                r##"<circle cx="18" cy="18" r="1.5"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["Point cloud file".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ply_core::{Element, Format, PlyFile, Property, PropertyData, PropertyKind, ScalarType};
    use volumetric_abi::fea::{COLOR_FIELD_NAME, NORMAL_FIELD_NAME};

    fn scalar(name: &str, ty: ScalarType, values: Vec<f64>) -> Property {
        Property {
            name: name.to_string(),
            kind: PropertyKind::Scalar(ty),
            data: PropertyData::Scalar(values),
        }
    }

    /// Five points along x with normals, colours and a confidence.
    fn cloud() -> Vec<u8> {
        let n = 5;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        ply_core::write_ply(&PlyFile {
            format: Format::BinaryLittleEndian,
            comments: vec![],
            obj_info: vec![],
            elements: vec![Element {
                name: "vertex".to_string(),
                count: n,
                properties: vec![
                    scalar("x", ScalarType::F32, xs.clone()),
                    scalar("y", ScalarType::F32, vec![0.0; n]),
                    scalar("z", ScalarType::F32, vec![2.0; n]),
                    scalar("nx", ScalarType::F32, vec![0.0; n]),
                    scalar("ny", ScalarType::F32, vec![0.0; n]),
                    scalar("nz", ScalarType::F32, vec![1.0; n]),
                    scalar("red", ScalarType::U8, vec![0.0, 51.0, 102.0, 153.0, 204.0]),
                    scalar("green", ScalarType::U8, vec![255.0; n]),
                    scalar("blue", ScalarType::U8, vec![0.0; n]),
                    scalar("confidence", ScalarType::F32, xs.clone()),
                ],
            }],
        })
        .unwrap()
    }

    #[test]
    fn imports_a_cloud_with_fields_and_stride() {
        let config = PointCloudImportConfig {
            scale: 0.5,
            center: false,
            stride: 2,
            fields: vec!["confidence".to_string()],
        };
        let (mesh, faces) = import(&cloud(), &config).unwrap();
        assert_eq!(faces, 0);
        assert_eq!(mesh.element_kind, FeaElementKind::Point1);
        assert_eq!(mesh.element_count(), 3, "stride 2 keeps points 0, 2, 4");
        assert_eq!(
            mesh.node_positions,
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0]
        );
        assert_eq!(mesh.connectivity, vec![0, 1, 2]);
        let names: Vec<&str> = mesh.node_fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            vec![NORMAL_FIELD_NAME, COLOR_FIELD_NAME, "confidence"]
        );
        assert_eq!(
            mesh.node_fields[0].data,
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        );
        assert!(
            (mesh.node_fields[1].data[3] - 0.4).abs() < 1e-12,
            "red of point 2"
        );
        assert_eq!(mesh.node_fields[2].data, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn rejects_non_ply_and_bad_stride() {
        let err = import(b"x y z\n", &PointCloudImportConfig::default()).unwrap_err();
        assert!(err.contains("PLY is the supported format"), "{err}");
        let config = PointCloudImportConfig {
            stride: 0,
            ..Default::default()
        };
        let err = import(&cloud(), &config).unwrap_err();
        assert!(err.contains("stride"), "{err}");
    }
}
