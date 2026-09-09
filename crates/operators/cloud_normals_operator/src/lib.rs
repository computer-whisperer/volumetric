//! Cloud Normals Operator.
//!
//! Estimates a unit normal at every node of a point cloud from the plane
//! through its nearest neighbours (`cloud_core::normals`) and writes it
//! as the `normal` node field — what Cloud Fit needs for cylinders and
//! what the file writers carry along. See README.md (the operator's
//! docs) for the orientation rule.
//!
//! Inputs:
//! - Input 0: FeaMesh — the cloud (any element kind; the nodes are used).
//! - Input 1: CBOR configuration, see [`CloudNormalsConfig`].
//!
//! Output 0: FeaMesh — the same mesh with the `normal` node field.

pub use cloud_core::normals::{CloudNormalsConfig, Orient};

use cloud_core::Vec3;
use cloud_core::normals::estimate;
use volumetric_abi::fea::{FeaField, FeaMesh, NORMAL_FIELD_NAME};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::fea::{decode_fea_mesh, encode_fea_mesh};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Estimate normals for the mesh's nodes and store them as its `normal`
/// field, replacing an existing one. Returns the degenerate count.
pub fn apply(mesh: &mut FeaMesh, config: &CloudNormalsConfig) -> Result<usize, String> {
    let points: Vec<Vec3> = (0..mesh.node_count())
        .map(|i| mesh.node_position(i))
        .collect();
    let (normals, degenerate) = estimate(&points, config)?;
    let field = FeaField {
        name: NORMAL_FIELD_NAME.to_string(),
        components: 3,
        data: normals.iter().flat_map(|n| n.iter().copied()).collect(),
    };
    mesh.node_fields.retain(|f| f.name != NORMAL_FIELD_NAME);
    mesh.node_fields.push(field);
    Ok(degenerate)
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let mut mesh = match decode_fea_mesh(&read_input(0)) {
        Ok(mesh) => mesh,
        Err(e) => {
            report_error(&format!("input 0 is not a usable mesh: {e}"));
            return;
        }
    };
    let config = {
        let cfg = read_input(1);
        if cfg.is_empty() {
            CloudNormalsConfig::default()
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
    match apply(&mut mesh, &config) {
        Ok(degenerate) => {
            if degenerate > 0 {
                post_warning(&format!(
                    "{degenerate} node(s) have no well-defined neighbourhood plane; their normal \
                     is zero"
                ));
            }
            post_output(0, &encode_fea_mesh(&mesh));
        }
        Err(e) => report_error(&format!("cloud normals failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ neighbours: int .ge 3 .le 256 .default 16, orient: "outward" / "up" / "none" .default "outward" }"#
            .to_string();
        OperatorMetadata {
            name: "cloud_normals_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Cloud Normals".to_string(),
            description: "Estimate a unit normal at every node of a point cloud from its nearest neighbours, as the `normal` field."
                .to_string(),
            category: "Mesh".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 16c4-4 8-6 18-6"/>"##,
                r##"<circle cx="7" cy="13" r="1"/>"##,
                r##"<circle cx="13" cy="11" r="1"/>"##,
                r##"<path d="M7 13V7M13 11V5"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["Cloud".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_core::{add, mul, norm};
    use volumetric_abi::fea::FeaElementKind;

    #[test]
    fn apply_replaces_the_normal_field() {
        // A Fibonacci sphere of radius 0.5.
        let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        let points: Vec<Vec3> = (0..500)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f64 + 0.5) / 500.0;
                let r = (1.0 - y * y).sqrt();
                let theta = golden * i as f64;
                add([0.0; 3], mul([r * theta.cos(), y, r * theta.sin()], 0.5))
            })
            .collect();
        let mut mesh = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: points.iter().flat_map(|p| p.iter().copied()).collect(),
            connectivity: (0..points.len() as u32).collect(),
            node_fields: vec![
                FeaField {
                    name: "color".to_string(),
                    components: 3,
                    data: vec![0.5; points.len() * 3],
                },
                FeaField {
                    name: NORMAL_FIELD_NAME.to_string(),
                    components: 3,
                    data: vec![0.0; points.len() * 3],
                },
            ],
            element_fields: vec![],
        };
        assert_eq!(apply(&mut mesh, &CloudNormalsConfig::default()).unwrap(), 0);
        assert_eq!(mesh.node_fields.len(), 2);
        assert_eq!(mesh.node_fields[0].name, "color");
        let normal = &mesh.node_fields[1];
        assert_eq!(normal.name, NORMAL_FIELD_NAME);
        assert_eq!(normal.data.len(), points.len() * 3);
        assert!((norm([normal.data[0], normal.data[1], normal.data[2]]) - 1.0).abs() < 1e-9);
        let n = [normal.data[0], normal.data[1], normal.data[2]];
        assert!(cloud_core::dot(n, points[0]) > 0.0, "outward");
    }
}
