//! End-to-end tests of the PLY importers: ply_import_operator (PLY → TriMesh,
//! chained into mesh_to_model_operator for a solid) and
//! point_cloud_import_operator (PLY → Point1 FeaMesh), executed through the
//! real wasm pipeline.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p ply_import_operator -p point_cloud_import_operator -p mesh_to_model_operator

#![cfg(feature = "native")]

use volumetric::fea::{COLOR_FIELD_NAME, FeaElementKind, NORMAL_FIELD_NAME, decode_fea_mesh};
use volumetric::trimesh::decode_tri_mesh;
use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::is_occupied;

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

/// An ASCII PLY of the axis-aligned cube [0, 10]^3 as six quads, each
/// vertex coloured by its corner, with a per-vertex `quality` scalar.
fn cube_ply() -> Vec<u8> {
    let mut text = String::from(
        "ply\nformat ascii 1.0\ncomment cube for the importer tests\n\
         element vertex 8\nproperty float x\nproperty float y\nproperty float z\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float quality\n\
         element face 6\nproperty list uchar int vertex_indices\nend_header\n",
    );
    for i in 0..8 {
        let (x, y, z) = (
            if i & 1 == 0 { 0 } else { 10 },
            if i & 2 == 0 { 0 } else { 10 },
            if i & 4 == 0 { 0 } else { 10 },
        );
        text.push_str(&format!(
            "{x} {y} {z} {} {} {} {}\n",
            x * 25,
            y * 25,
            z * 25,
            i as f64 / 10.0
        ));
    }
    // Outward-wound quads (matching the STL/3MF test cubes).
    for q in [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ] {
        text.push_str(&format!("4 {} {} {} {}\n", q[0], q[1], q[2], q[3]));
    }
    text.into_bytes()
}

/// A binary little-endian PLY of `n` points along +x at z = 2 with normals
/// and colours: what a scan pipeline writes.
fn cloud_ply(n: usize) -> Vec<u8> {
    let mut out = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {n}\n\
         property float x\nproperty float y\nproperty float z\n\
         property float nx\nproperty float ny\nproperty float nz\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    .into_bytes();
    for i in 0..n {
        for v in [i as f32, 0.0, 2.0, 0.0, 0.0, 1.0] {
            out.extend(v.to_le_bytes());
        }
        out.extend([(i * 255 / n.max(1)) as u8, 255, 0]);
    }
    out
}

/// CBOR-encodes a config map from `(key, value)` pairs.
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

#[test]
fn ply_mesh_imports_with_colours_and_converts_to_a_solid() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("ply_import_operator"),
    ));
    project.imports.push(ImportedAsset::operator(
        "converter".to_string(),
        wasm_artifact("mesh_to_model_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(cube_ply()),
            ExecutionInput::Inline(config(vec![
                ("scale", ciborium::Value::Float(0.1)),
                ("center", ciborium::Value::Bool(true)),
                (
                    "fields",
                    ciborium::Value::Array(vec![ciborium::Value::Text("quality".to_string())]),
                ),
            ])),
        ],
        outputs: vec!["mesh".to_string()],
    });
    project.timeline.push(ExecutionStep {
        operator_id: "converter".to_string(),
        inputs: vec![ExecutionInput::AssetRef("mesh".to_string())],
        outputs: vec!["solid".to_string()],
    });
    project.exports = vec!["mesh".to_string(), "solid".to_string()];

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");
    assert_eq!(exports.len(), 2);

    let mesh_asset = &exports[0];
    assert_eq!(mesh_asset.type_hint(), Some(AssetTypeHint::TriMesh));
    let mesh = decode_tri_mesh(mesh_asset.data()).unwrap();
    assert_eq!(mesh.vertex_count(), 8);
    assert_eq!(
        mesh.triangle_count(),
        12,
        "six quads fan into twelve triangles"
    );
    // [0, 10]^3 centred then scaled by 0.1: [-0.5, 0.5]^3.
    assert_eq!(mesh.bounds(), Some([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]));
    let names: Vec<&str> = mesh.vertex_fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec![COLOR_FIELD_NAME, "quality"]);
    let color = &mesh.vertex_fields[0];
    assert_eq!(color.components, 3);
    // Vertex 7 sits at (10, 10, 10): colour 250/255 on every channel.
    assert!((color.data[21] - 250.0 / 255.0).abs() < 1e-12);
    assert_eq!(mesh.vertex_fields[1].data[7], 0.7);

    let solid = &exports[1];
    assert_eq!(solid.type_hint(), Some(AssetTypeHint::Model));
    let mut executor = create_model_executor(solid.data()).expect("model loads");
    assert_eq!(executor.dimensions().unwrap(), 3);
    assert!(is_occupied(executor.sample_nd(&[0.0, 0.0, 0.0]).unwrap()));
    assert!(is_occupied(
        executor.sample_nd(&[0.45, -0.45, 0.45]).unwrap()
    ));
    assert!(!is_occupied(executor.sample_nd(&[0.6, 0.0, 0.0]).unwrap()));
    assert!(!is_occupied(executor.sample_nd(&[0.0, -0.7, 0.0]).unwrap()));
}

#[test]
fn point_cloud_imports_as_a_point1_mesh() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("point_cloud_import_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(cloud_ply(10)),
            ExecutionInput::Inline(config(vec![
                ("scale", ciborium::Value::Float(1.0)),
                ("center", ciborium::Value::Bool(false)),
                ("stride", ciborium::Value::Integer(3.into())),
            ])),
        ],
        outputs: vec!["cloud".to_string()],
    });
    project.exports = vec!["cloud".to_string()];

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");
    let cloud = &exports[0];
    assert_eq!(cloud.type_hint(), Some(AssetTypeHint::FeaMesh));
    let mesh = decode_fea_mesh(cloud.data()).unwrap();
    assert_eq!(mesh.element_kind, FeaElementKind::Point1);
    assert_eq!(
        mesh.element_count(),
        4,
        "stride 3 over 10 points keeps 0, 3, 6, 9"
    );
    assert_eq!(mesh.node_position(3), [9.0, 0.0, 2.0]);
    let names: Vec<&str> = mesh.node_fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec![NORMAL_FIELD_NAME, COLOR_FIELD_NAME]);
    assert_eq!(mesh.node_fields[0].data[9..12], [0.0, 0.0, 1.0]);
    let color = &mesh.node_fields[1];
    assert_eq!(color.data[10], 1.0, "green channel is full on every point");
    assert!((color.data[9] - (9 * 255 / 10) as f64 / 255.0).abs() < 1e-12);
}

#[test]
fn a_faceless_ply_fails_the_mesh_importer_with_a_pointer() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("ply_import_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(cloud_ply(3)),
            ExecutionInput::Inline(vec![]),
        ],
        outputs: vec!["mesh".to_string()],
    });
    project.exports = vec!["mesh".to_string()];

    let mut env = Environment::new();
    let err = project.run(&mut env).unwrap_err().to_string();
    assert!(err.contains("Point Cloud Import"), "{err}");
}
