//! End-to-end test of obj_import_operator (OBJ + texture → TriMesh), chained
//! into mesh_to_model_operator for a solid, executed through the real wasm
//! pipeline; plus a run over a real scanner export when one is on disk.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p obj_import_operator -p mesh_to_model_operator

#![cfg(feature = "native")]

use volumetric::trimesh::{UV_FIELD_NAME, decode_tri_mesh};
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

/// The axis-aligned cube [0, 10]^3 as six textured quads: each face maps
/// the whole unit UV square, written as a scanner would (v/vt corners,
/// an mtllib and a usemtl).
fn cube_obj() -> Vec<u8> {
    let mut text = String::from("# cube\nmtllib cube.mtl\no cube\n");
    for i in 0..8 {
        let (x, y, z) = (
            if i & 1 == 0 { 0 } else { 10 },
            if i & 2 == 0 { 0 } else { 10 },
            if i & 4 == 0 { 0 } else { 10 },
        );
        text.push_str(&format!("v {x} {y} {z}\n"));
    }
    text.push_str("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl scan\n");
    // Outward-wound quads (matching the STL/3MF/PLY test cubes), 1-based.
    for q in [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ] {
        text.push_str(&format!(
            "f {}/1 {}/2 {}/3 {}/4\n",
            q[0] + 1,
            q[1] + 1,
            q[2] + 1,
            q[3] + 1
        ));
    }
    text.into_bytes()
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

/// A 2x2 RGB PNG: red, green on the top row; blue, white on the bottom.
const PNG_2X2: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0xFD, 0xD4, 0x9A,
    0x73, 0x00, 0x00, 0x00, 0x12, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0xF8, 0xCF, 0xC0, 0xC0,
    0x00, 0xC2, 0x0C, 0xFF, 0x81, 0x00, 0x00, 0x1F, 0xEE, 0x05, 0xFB, 0xF1, 0xAB, 0xBA, 0x77, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

#[test]
fn obj_mesh_imports_with_uvs_and_texture_and_converts_to_a_solid() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("obj_import_operator"),
    ));
    project.imports.push(ImportedAsset::operator(
        "converter".to_string(),
        wasm_artifact("mesh_to_model_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(cube_obj()),
            ExecutionInput::Inline(PNG_2X2.to_vec()),
            ExecutionInput::Inline(config(vec![
                ("scale", ciborium::Value::Float(0.1)),
                ("center", ciborium::Value::Bool(true)),
                (
                    "texture_name",
                    ciborium::Value::Text("cube.png".to_string()),
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
    // A vertex is a distinct (v, vt) pair: each cube corner is used by
    // three faces at the face-local texture corners 1..4, which coincide
    // for some of its uses, so the 24 corners weld into 20 vertices (more
    // than the 8 positions, fewer than the 24 corners).
    assert_eq!(mesh.vertex_count(), 20, "corners split along UV seams");
    assert_eq!(mesh.triangle_count(), 12);
    // [0, 10]^3 centred then scaled by 0.1: [-0.5, 0.5]^3.
    assert_eq!(mesh.bounds(), Some([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]));
    let uv = mesh.uv_field().expect("uv field");
    assert_eq!(uv.name, UV_FIELD_NAME);
    assert_eq!(uv.data.len(), 24 * 2);
    assert!(uv.data.iter().all(|v| (0.0..=1.0).contains(v)));
    let texture = mesh.base_color_texture().expect("texture attached");
    assert_eq!(texture.name, "cube.png");
    assert_eq!(texture.media_type, "image/png");
    assert_eq!(texture.bytes, PNG_2X2);

    let solid = &exports[1];
    assert_eq!(solid.type_hint(), Some(AssetTypeHint::Model));
    let mut executor = create_model_executor(solid.data()).expect("model loads");
    assert_eq!(executor.dimensions().unwrap(), 3);
    assert!(is_occupied(executor.sample_nd(&[0.0, 0.0, 0.0]).unwrap()));
    assert!(is_occupied(
        executor.sample_nd(&[0.45, -0.45, 0.45]).unwrap()
    ));
    assert!(!is_occupied(executor.sample_nd(&[0.6, 0.0, 0.0]).unwrap()));
}

/// Operator-level warnings reach the host: a texture for a file without
/// texture coordinates is dropped and said so.
#[test]
fn texture_without_uvs_is_dropped_and_the_mesh_still_imports() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("obj_import_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".to_vec()),
            ExecutionInput::Inline(PNG_2X2.to_vec()),
            ExecutionInput::Inline(Vec::new()),
        ],
        outputs: vec!["mesh".to_string()],
    });
    project.exports = vec!["mesh".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");
    let mesh = decode_tri_mesh(exports[0].data()).unwrap();
    assert_eq!(mesh.triangle_count(), 1);
    assert!(mesh.textures.is_empty());
    assert!(mesh.uv_field().is_none());
}

/// A real phone-scanner export, when the fixture directory is on this
/// machine (skips otherwise): the whole atlas-textured mesh goes through
/// the importer and the timing is printed.
#[test]
fn real_scan_imports_when_present() {
    let dir =
        std::path::PathBuf::from("/home/ben/Documents/kalogon/git/scanned-cushion/cases/7319/Scan");
    let obj_path = dir.join("7319_scan.obj");
    if !obj_path.is_file() {
        eprintln!("skipping: {} not present", obj_path.display());
        return;
    }
    let obj = std::fs::read(&obj_path).unwrap();
    let jpg = std::fs::read(dir.join("7319_scan.jpg")).unwrap();

    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("obj_import_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(obj),
            ExecutionInput::Inline(jpg.clone()),
            ExecutionInput::Inline(config(vec![(
                "texture_name",
                ciborium::Value::Text("7319_scan.jpg".to_string()),
            )])),
        ],
        outputs: vec!["scan".to_string()],
    });
    project.exports = vec!["scan".to_string()];
    let start = std::time::Instant::now();
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");
    let elapsed = start.elapsed();
    let mesh = decode_tri_mesh(exports[0].data()).unwrap();
    eprintln!(
        "scan: {} vertices, {} triangles, texture {} bytes, imported in {elapsed:?}",
        mesh.vertex_count(),
        mesh.triangle_count(),
        mesh.textures[0].bytes.len()
    );
    assert_eq!(mesh.triangle_count(), 170_783);
    assert!(mesh.vertex_count() >= 93_480, "at least one vertex per vt");
    assert_eq!(mesh.textures[0].bytes, jpg);
    assert_eq!(mesh.textures[0].media_type, "image/jpeg");
    let b = mesh.bounds().unwrap();
    assert!(
        (b[1] - b[0] - 0.588).abs() < 0.01,
        "extent in metres: {b:?}"
    );
}
