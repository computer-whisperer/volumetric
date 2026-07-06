//! Integration tests for mesh_height_operator: a triangle mesh wrapped as a
//! 2D height-query model, and its composition with heightmap_extrude to
//! rebuild the solid under the surface.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::wasm::{
    NativeModelExecutor, OperatorExecutor, OperatorIo, create_operator_executor,
};
use volumetric_abi::trimesh::{TriMesh, encode_tri_mesh};

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing wasm artifact {} ({e}); build it with `cargo build-wasm`",
            path.display()
        )
    })
}

fn run_operator(operator: &str, inputs: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    let operator_wasm = wasm_artifact(operator);
    let mut executor = create_operator_executor(&operator_wasm).expect("create operator executor");
    let result = executor
        .run(OperatorIo::new(inputs))
        .map_err(|e| e.to_string())?;
    result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "operator posted no output".to_string())
}

fn cbor_config(entries: &[(&str, ciborium::value::Value)]) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(
            entries
                .iter()
                .map(|(k, v)| (ciborium::value::Value::Text((*k).into()), v.clone()))
                .collect(),
        ),
        &mut out,
    )
    .unwrap();
    out
}

/// The axis-aligned box [0,1] x [0,2] x [0,3] as an encoded TriMesh —
/// distinct extents per axis so axis mix-ups can't cancel out.
fn box_mesh() -> Vec<u8> {
    let max = [1.0, 2.0, 3.0];
    let corners: Vec<[f64; 3]> = (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { 0.0 } else { max[0] },
                if i & 2 == 0 { 0.0 } else { max[1] },
                if i & 4 == 0 { 0.0 } else { max[2] },
            ]
        })
        .collect();
    let quads = [
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
    ];
    let mut indices = Vec::new();
    for q in quads {
        indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]]);
    }
    encode_tri_mesh(&TriMesh {
        positions: corners.into_iter().flatten().collect(),
        indices,
        vertex_fields: vec![],
        face_fields: vec![],
    })
}

#[test]
fn box_height_query_defaults_to_top_along_z() {
    let model =
        run_operator("mesh_height_operator", vec![box_mesh(), Vec::new()]).expect("mesh converts");
    let mut executor = NativeModelExecutor::new(&model).unwrap();
    assert_eq!(executor.dimensions(), 2);

    // Bounds are the mesh footprint in (x, y).
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (0.0, 1.0));
    assert_eq!((bounds.min(1), bounds.max(1)), (0.0, 2.0));

    // Over the footprint the top face is at z = 3; off it the line misses
    // and the sample is the miss value (default 0).
    assert_eq!(executor.sample_nd(&[0.5, 1.0]).unwrap(), 3.0);
    assert_eq!(executor.sample_nd(&[0.9, 1.9]).unwrap(), 3.0);
    assert_eq!(executor.sample_nd(&[1.5, 1.0]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.5, -0.5]).unwrap(), 0.0);
}

#[test]
fn surface_axis_and_miss_config() {
    use ciborium::value::Value;

    // Bottom surface along z is the z = 0 face.
    let bottom = run_operator(
        "mesh_height_operator",
        vec![
            box_mesh(),
            cbor_config(&[("surface", Value::Text("bottom".into()))]),
        ],
    )
    .expect("mesh converts");
    let mut executor = NativeModelExecutor::new(&bottom).unwrap();
    assert_eq!(executor.sample_nd(&[0.5, 1.0]).unwrap(), 0.0);

    // Height along y: lateral axes are (x, z) ascending, top face at y = 2,
    // and a custom miss value where the line misses.
    let along_y = run_operator(
        "mesh_height_operator",
        vec![
            box_mesh(),
            cbor_config(&[
                ("axis", Value::Text("y".into())),
                ("miss", Value::Float(-1.0)),
            ]),
        ],
    )
    .expect("mesh converts");
    let mut executor = NativeModelExecutor::new(&along_y).unwrap();
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (0.0, 1.0));
    assert_eq!((bounds.min(1), bounds.max(1)), (0.0, 3.0));
    assert_eq!(executor.sample_nd(&[0.5, 1.5]).unwrap(), 2.0);
    assert_eq!(executor.sample_nd(&[0.5, 3.5]).unwrap(), -1.0);

    // Invalid enum values are rejected loudly.
    let err = run_operator(
        "mesh_height_operator",
        vec![
            box_mesh(),
            cbor_config(&[("axis", Value::Text("w".into()))]),
        ],
    )
    .expect_err("bad axis must fail");
    assert!(err.contains("axis"), "{err}");
}

#[test]
fn height_field_extrudes_back_into_the_solid() {
    // mesh → height field → heightmap extrude: recovers the box (its top
    // surface is flat at 3, so scale 1 with max_height 3 rebuilds it).
    let field =
        run_operator("mesh_height_operator", vec![box_mesh(), Vec::new()]).expect("mesh converts");
    let solid = run_operator(
        "heightmap_extrude_operator",
        vec![
            field,
            cbor_config(&[
                ("scale", ciborium::value::Value::Float(1.0)),
                ("max_height", ciborium::value::Value::Float(3.0)),
            ]),
        ],
    )
    .expect("extrude height field");

    let mut executor = NativeModelExecutor::new(&solid).unwrap();
    assert_eq!(executor.dimensions(), 3);
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(2), bounds.max(2)), (0.0, 3.0));

    assert_eq!(executor.sample_nd(&[0.5, 1.0, 0.1]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.5, 1.0, 2.9]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.5, 1.0, 3.1]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[1.5, 1.0, 0.1]).unwrap(), 0.0);
}

#[test]
fn empty_mesh_is_rejected() {
    let empty = encode_tri_mesh(&TriMesh {
        positions: vec![],
        indices: vec![],
        vertex_fields: vec![],
        face_fields: vec![],
    });
    let err = run_operator("mesh_height_operator", vec![empty, Vec::new()])
        .expect_err("empty mesh must fail");
    assert!(err.contains("triangle"), "{err}");
}
