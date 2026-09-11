//! End-to-end test of splat_import_operator and splat_points_operator: a
//! synthetic 3DGS PLY becomes a Splat asset through the real wasm
//! pipeline, and the Splat becomes a Point1 cloud near a solved marker.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!       -p splat_import_operator -p splat_points_operator
#![cfg(feature = "native")]

use ply_core::{Element, Format, PlyFile, Property, PropertyData, PropertyKind, ScalarType};
use volumetric::fea::{COLOR_FIELD_NAME, NORMAL_FIELD_NAME, decode_fea_mesh};
use volumetric::splat::{SplatKind, decode_splat};
use volumetric::viewset::{
    Marker, Provenance, VIEWSET_SCHEMA, ViewSet, WorldFrame, encode_viewset,
};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset,
    OperatorMetadataInput, OperatorMetadataOutput, Project,
};

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing wasm artifact {} ({e}); build it with \
             `cargo build --target wasm32-unknown-unknown --release -p {name}`",
            path.display()
        )
    })
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn config(entries: Vec<(&str, ciborium::Value)>) -> Vec<u8> {
    let value = ciborium::Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (ciborium::Value::Text(k.to_string()), v))
            .collect(),
    );
    let mut out = Vec::new();
    ciborium::into_writer(&value, &mut out).unwrap();
    out
}

fn scalar(name: &str, values: Vec<f64>) -> Property {
    Property {
        name: name.to_string(),
        kind: PropertyKind::Scalar(ScalarType::F32),
        data: PropertyData::Scalar(values),
    }
}

/// Twenty Gaussians on a 5 x 4 grid over the unit square at z = 0, thin
/// along z, SH degree 2, opaque but for the first row.
fn splat_ply() -> Vec<u8> {
    let n = 20;
    let xs: Vec<f64> = (0..n).map(|i| (i % 5) as f64 * 0.25).collect();
    let ys: Vec<f64> = (0..n).map(|i| (i / 5) as f64 / 3.0).collect();
    let mut properties = vec![
        scalar("x", xs),
        scalar("y", ys),
        scalar("z", vec![0.0; n]),
        scalar("nx", vec![0.0; n]),
        scalar("ny", vec![0.0; n]),
        scalar("nz", vec![0.0; n]),
        scalar("f_dc_0", vec![1.0; n]),
        scalar("f_dc_1", vec![0.0; n]),
        scalar("f_dc_2", vec![-1.0; n]),
    ];
    for k in 0..24 {
        properties.push(scalar(&format!("f_rest_{k}"), vec![0.01 * k as f64; n]));
    }
    properties.push(scalar(
        "opacity",
        (0..n).map(|i| if i < 5 { -4.0 } else { 4.0 }).collect(),
    ));
    properties.push(scalar("scale_0", vec![(0.02f64).ln(); n]));
    properties.push(scalar("scale_1", vec![(0.02f64).ln(); n]));
    properties.push(scalar("scale_2", vec![(0.001f64).ln(); n]));
    properties.push(scalar("rot_0", vec![1.0; n]));
    properties.push(scalar("rot_1", vec![0.0; n]));
    properties.push(scalar("rot_2", vec![0.0; n]));
    properties.push(scalar("rot_3", vec![0.0; n]));
    ply_core::write_ply(&PlyFile {
        format: Format::BinaryLittleEndian,
        comments: vec![],
        obj_info: vec![],
        elements: vec![Element {
            name: "vertex".to_string(),
            count: n,
            properties,
        }],
    })
    .unwrap()
}

/// A view set whose only content is a solved swatch centred on (1, 1, 0).
fn views_with_marker() -> Vec<u8> {
    let set = ViewSet {
        schema: VIEWSET_SCHEMA,
        world: WorldFrame::default(),
        provenance: Provenance::default(),
        cameras: vec![],
        views: vec![],
        markers: vec![Marker {
            id: 3,
            size_m: 0.06,
            corners: [
                [0.97, 0.97, 0.0],
                [1.03, 0.97, 0.0],
                [1.03, 1.03, 0.0],
                [0.97, 1.03, 0.0],
            ],
        }],
        board: None,
    };
    encode_viewset(&set)
}

fn run(
    import_config: Vec<u8>,
    points_config: Vec<u8>,
    views: Option<Vec<u8>>,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "splat_import".to_string(),
        wasm_artifact("splat_import_operator"),
    ));
    project.imports.push(ImportedAsset::operator(
        "splat_points".to_string(),
        wasm_artifact("splat_points_operator"),
    ));
    project.imports.push(ImportedAsset::new(
        "splat_ply".to_string(),
        splat_ply(),
        Some(AssetTypeHint::Binary),
    ));
    let views_input = match views {
        Some(bytes) => {
            project.imports.push(ImportedAsset::new(
                "views".to_string(),
                bytes,
                Some(AssetTypeHint::ViewSet),
            ));
            ExecutionInput::AssetRef("views".to_string())
        }
        None => ExecutionInput::Inline(Vec::new()),
    };
    project.timeline.push(ExecutionStep {
        operator_id: "splat_import".to_string(),
        inputs: vec![
            ExecutionInput::AssetRef("splat_ply".to_string()),
            views_input.clone(),
            ExecutionInput::Inline(import_config),
        ],
        outputs: vec!["splat".to_string()],
    });
    project.timeline.push(ExecutionStep {
        operator_id: "splat_points".to_string(),
        inputs: vec![
            ExecutionInput::AssetRef("splat".to_string()),
            views_input,
            ExecutionInput::Inline(points_config),
        ],
        outputs: vec!["points".to_string()],
    });
    project.exports = vec!["splat".to_string(), "points".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).map_err(|e| e.to_string())?;
    assert_eq!(exports[0].type_hint(), Some(AssetTypeHint::Splat));
    assert_eq!(exports[1].type_hint(), Some(AssetTypeHint::FeaMesh));
    Ok((exports[0].data().to_vec(), exports[1].data().to_vec()))
}

#[test]
fn metadata_declares_the_splat_slots() {
    let import =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("splat_import_operator"))
            .expect("metadata");
    assert_eq!(import.name, "splat_import_operator");
    assert_eq!(import.category, "Import");
    assert!(matches!(import.inputs[0], OperatorMetadataInput::Blob));
    assert!(matches!(import.inputs[1], OperatorMetadataInput::ViewSet));
    assert_eq!(import.outputs, vec![OperatorMetadataOutput::Splat]);
    assert!(import.docs.contains("f_rest"));

    let points =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("splat_points_operator"))
            .expect("metadata");
    assert_eq!(points.name, "splat_points_operator");
    assert!(matches!(points.inputs[0], OperatorMetadataInput::Splat));
    assert!(matches!(points.inputs[1], OperatorMetadataInput::ViewSet));
    assert!(matches!(
        points.inputs[2],
        OperatorMetadataInput::CBORConfiguration(_)
    ));
    assert_eq!(points.outputs, vec![OperatorMetadataOutput::FeaMesh]);
    // The config schema parses with its optional groups.
    if let OperatorMetadataInput::CBORConfiguration(cddl) = &points.inputs[2] {
        let fields = volumetric::operator_config::parse_schema(cddl).expect("schema parses");
        assert!(fields.iter().any(|f| f.name == "near"), "{fields:?}");
    }
}

#[test]
fn a_ply_becomes_a_splat_and_a_cloud_through_wasm() {
    let (splat, points) = run(
        config(vec![
            ("session", ciborium::Value::Text("synthetic".to_string())),
            ("views_hash", ciborium::Value::Text("abc123".to_string())),
        ]),
        Vec::new(),
        None,
    )
    .unwrap();
    let splat = decode_splat(&splat).unwrap();
    assert_eq!(splat.count, 20);
    assert_eq!(
        splat.kind,
        SplatKind::Gaussian3d,
        "1 mm of 20 mm is a fat surfel"
    );
    assert_eq!(splat.sh_degree, 2);
    assert!(splat.normals.is_empty());
    assert_eq!(splat.provenance.session, "synthetic");
    assert_eq!(splat.views_hash, "abc123");
    assert_eq!(splat.mean(7), [0.5, 1.0 / 3.0, 0.0]);
    assert!((splat.sh_rest[24 * 7 + 5] - 0.05).abs() < 1e-6);

    // Default filters keep the fifteen opaque ones, with normals along z.
    let cloud = decode_fea_mesh(&points).unwrap();
    assert_eq!(cloud.node_positions.len(), 15 * 3);
    let normal = cloud
        .node_fields
        .iter()
        .find(|f| f.name == NORMAL_FIELD_NAME)
        .unwrap();
    assert!(normal.data.chunks(3).all(|n| n[2].abs() > 0.999));
    let color = cloud
        .node_fields
        .iter()
        .find(|f| f.name == COLOR_FIELD_NAME)
        .unwrap();
    let expected = [0.5 + 0.28209479, 0.5, 0.5 - 0.28209479];
    for k in 0..3 {
        assert!(
            (color.data[k] - expected[k]).abs() < 1e-6,
            "{:?}",
            &color.data[0..3]
        );
    }
}

#[test]
fn points_near_a_solved_marker_through_wasm() {
    let points_config = config(vec![(
        "near",
        ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("marker".to_string()),
                ciborium::Value::Text("3".to_string()),
            ),
            (
                ciborium::Value::Text("radius".to_string()),
                ciborium::Value::Float(0.2),
            ),
        ]),
    )]);
    let (splat, points) =
        run(Vec::new(), points_config.clone(), Some(views_with_marker())).unwrap();
    // The wired set's content hash lands on the splat.
    let splat = decode_splat(&splat).unwrap();
    let expected = volumetric::content_fingerprint(&views_with_marker());
    assert_eq!(splat.views_hash, hex(&expected));
    let cloud = decode_fea_mesh(&points).unwrap();
    // Only the grid's corner (1, 1, 0) is within 0.2 m of the marker; its
    // neighbour at (0.75, 1, 0) is 0.25 m away.
    assert_eq!(cloud.node_positions, vec![1.0, 1.0, 0.0]);

    // Without the set the marker cannot be resolved.
    let err = run(Vec::new(), points_config, None).unwrap_err();
    assert!(err.contains("needs the view set"), "{err}");
}
