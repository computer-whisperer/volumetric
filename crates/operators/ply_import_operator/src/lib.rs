//! PLY Import Operator.
//!
//! Reads a PLY file and emits a [`TriMesh`] value — an explicit triangle
//! mesh, *not* a sampleable model; converting a (watertight) mesh into an
//! implicit solid is `mesh_to_model_operator`'s job. See README.md (the
//! operator's docs) for what the reader honours. A faceless PLY is a point
//! cloud, which `point_cloud_import_operator` reads instead.
//!
//! Inputs:
//! - Input 0: Blob — PLY file bytes
//! - Input 1: CBOR configuration: `scale` (float, default 1.0), `center`
//!   (bool, default false), `fields` (vertex property names carried through
//!   as one-component vertex fields, default none). Placement is
//!   center → scale.
//!
//! Output 0: TriMesh.

use ply_core::convert::{faces, place, vertices};
use ply_core::read_ply;
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
use volumetric_abi::trimesh::{TriMesh, encode_tri_mesh};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct PlyImportConfig {
    pub scale: f64,
    pub center: bool,
    pub fields: Vec<String>,
}

impl Default for PlyImportConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            center: false,
            fields: Vec::new(),
        }
    }
}

/// Reads the file as a mesh, returning it with the number of degenerate
/// faces (fewer than three vertices) that were dropped.
pub fn import(bytes: &[u8], config: &PlyImportConfig) -> Result<(TriMesh, usize), String> {
    let file = read_ply(bytes)?;
    let Some(face_data) = faces(&file)? else {
        return Err(
            "the file has no faces, so it is a point cloud: import it with Point Cloud Import"
                .to_string(),
        );
    };
    if face_data.indices.is_empty() {
        return Err("the file declares faces but none has three or more vertices".to_string());
    }
    let mut vertex = vertices(&file, &config.fields)?;
    place(&mut vertex.positions, config.center, config.scale)?;
    let mesh = TriMesh {
        positions: vertex.positions,
        indices: face_data.indices,
        vertex_fields: vertex.fields,
        face_fields: vec![],
    };
    mesh.validate()?;
    Ok((mesh, face_data.skipped))
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let bytes = read_input(0);
    let config = {
        let cfg = read_input(1);
        if cfg.is_empty() {
            PlyImportConfig::default()
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
        Ok((mesh, skipped)) => {
            if skipped > 0 {
                post_warning(&format!(
                    "dropped {skipped} face(s) with fewer than three vertices"
                ));
            }
            post_output(0, &encode_tri_mesh(&mesh));
        }
        Err(e) => report_error(&format!("PLY import failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ scale: float .default 1.0, center: bool .default false, fields: [* tstr] }"
            .to_string();
        OperatorMetadata {
            name: "ply_import_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "PLY Import".to_string(),
            description:
                "Read a PLY mesh file as an explicit triangle mesh, colours and normals included."
                    .to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>"##,
                r##"<path d="M14 2v4a2 2 0 0 0 2 2h4"/>"##,
                r##"<path d="m9 18 3-6 3 6Z"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["PLY file".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::TriMesh],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ply_core::{Element, Format, PlyFile, Property, PropertyData, PropertyKind, ScalarType};
    use volumetric_abi::trimesh::COLOR_FIELD_NAME;

    fn scalar(name: &str, ty: ScalarType, values: Vec<f64>) -> Property {
        Property {
            name: name.to_string(),
            kind: PropertyKind::Scalar(ty),
            data: PropertyData::Scalar(values),
        }
    }

    /// A unit square as two triangles (one quad face), coloured, in millimetres.
    fn square(with_faces: bool) -> Vec<u8> {
        let mut elements = vec![Element {
            name: "vertex".to_string(),
            count: 4,
            properties: vec![
                scalar("x", ScalarType::F32, vec![0.0, 10.0, 10.0, 0.0]),
                scalar("y", ScalarType::F32, vec![0.0, 0.0, 10.0, 10.0]),
                scalar("z", ScalarType::F32, vec![0.0; 4]),
                scalar("red", ScalarType::U8, vec![255.0, 0.0, 0.0, 255.0]),
                scalar("green", ScalarType::U8, vec![0.0, 255.0, 0.0, 255.0]),
                scalar("blue", ScalarType::U8, vec![0.0, 0.0, 255.0, 255.0]),
                scalar("quality", ScalarType::F32, vec![0.1, 0.2, 0.3, 0.4]),
            ],
        }];
        if with_faces {
            elements.push(Element {
                name: "face".to_string(),
                count: 2,
                properties: vec![Property {
                    name: "vertex_indices".to_string(),
                    kind: PropertyKind::List {
                        count: ScalarType::U8,
                        item: ScalarType::I32,
                    },
                    data: PropertyData::List {
                        offsets: vec![0, 4, 6],
                        items: vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                    },
                }],
            });
        }
        ply_core::write_ply(&PlyFile {
            format: Format::BinaryLittleEndian,
            comments: vec![],
            obj_info: vec![],
            elements,
        })
        .unwrap()
    }

    #[test]
    fn imports_a_quad_as_two_triangles_with_colour_and_fields() {
        let config = PlyImportConfig {
            scale: 1e-3,
            center: true,
            fields: vec!["quality".to_string()],
        };
        let (mesh, skipped) = import(&square(true), &config).unwrap();
        assert_eq!(skipped, 1, "the two-vertex face is dropped");
        assert_eq!(mesh.triangle_count(), 2);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(mesh.bounds(), Some([-5e-3, 5e-3, -5e-3, 5e-3, 0.0, 0.0]));
        assert_eq!(mesh.vertex_fields.len(), 2);
        assert_eq!(mesh.vertex_fields[0].name, COLOR_FIELD_NAME);
        assert_eq!(mesh.vertex_fields[0].data[3..6], [0.0, 1.0, 0.0]);
        assert_eq!(mesh.vertex_fields[1].name, "quality");
        // Stored as float32 in the file, so read back at f32 precision.
        let expected: Vec<f64> = [0.1f32, 0.2, 0.3, 0.4]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_eq!(mesh.vertex_fields[1].data, expected);
    }

    #[test]
    fn faceless_files_point_at_the_cloud_importer() {
        let err = import(&square(false), &PlyImportConfig::default()).unwrap_err();
        assert!(err.contains("Point Cloud Import"), "{err}");
        let err = import(b"not a ply", &PlyImportConfig::default()).unwrap_err();
        assert!(err.contains("magic"), "{err}");
    }
}
