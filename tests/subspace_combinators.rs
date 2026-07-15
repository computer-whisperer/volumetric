//! Integration tests for the subspace-combining operators: `span_operator`
//! (affine join) and `intersect_operator` (affine meet). These drive the
//! real wasm modules through the operator host, so they exercise the whole
//! path — config decode, reading several `Subspace` slots (with unwired
//! ones read back empty), computing, and CBOR-encoding the result.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use std::collections::BTreeMap;

use volumetric::subspace::{Subspace, decode_subspace, encode_subspace};
use volumetric::wasm::{OperatorExecutor, OperatorIo, create_operator_executor};

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

/// Run an operator with the given raw input slots and return output slot 0.
fn run_op(name: &str, inputs: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    let operator_wasm = wasm_artifact(name);
    let mut executor = create_operator_executor(&operator_wasm).map_err(|e| e.to_string())?;
    let result = executor
        .run(OperatorIo::new(inputs))
        .map_err(|e| e.to_string())?;
    result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "operator posted no output".to_string())
}

fn point(x: f64, y: f64, z: f64) -> Vec<u8> {
    encode_subspace(&Subspace::point(vec![x, y, z]))
}

fn plane(origin: Vec<f64>, axes: &[usize]) -> Vec<u8> {
    encode_subspace(&Subspace::axis_aligned(origin, axes).unwrap())
}

/// CBOR config map `{ field: value }` as the operators decode it.
fn config(field: &str, value: &str) -> Vec<u8> {
    let mut map = BTreeMap::new();
    map.insert(field.to_string(), value.to_string());
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(&map, &mut bytes).unwrap();
    bytes
}

fn parallel_to(v: &[f64], axis: [f64; 3]) -> bool {
    let cross = [
        v[1] * axis[2] - v[2] * axis[1],
        v[2] * axis[0] - v[0] * axis[2],
        v[0] * axis[1] - v[1] * axis[0],
    ];
    cross.iter().all(|c| c.abs() < 1e-9)
}

#[test]
fn span_of_three_points_is_a_plane() {
    let out = run_op(
        "span_operator",
        vec![
            vec![], // default config
            point(0.0, 0.0, 0.0),
            point(1.0, 0.0, 0.0),
            point(0.0, 1.0, 0.0),
        ],
    )
    .expect("span should succeed");
    let plane = decode_subspace(&out).unwrap();
    assert_eq!(plane.rank(), 2);
    assert!(parallel_to(&plane.normal().unwrap(), [0.0, 0.0, 1.0]));
}

#[test]
fn span_of_two_points_is_a_line() {
    let out = run_op(
        "span_operator",
        vec![vec![], point(0.0, 0.0, 0.0), point(0.0, 0.0, 3.0)],
    )
    .expect("span should succeed");
    let line = decode_subspace(&out).unwrap();
    assert_eq!(line.rank(), 1);
    assert!(parallel_to(line.basis_vector(0), [0.0, 0.0, 1.0]));
}

#[test]
fn span_expect_plane_rejects_collinear_points() {
    // Three collinear points only span a line; asserting "plane" fails.
    let err = run_op(
        "span_operator",
        vec![
            config("expect", "plane"),
            point(0.0, 0.0, 0.0),
            point(1.0, 0.0, 0.0),
            point(2.0, 0.0, 0.0),
        ],
    )
    .expect_err("collinear points are not a plane");
    assert!(err.contains("rank"), "unexpected error: {err}");
}

#[test]
fn intersect_of_two_planes_is_their_shared_axis() {
    // xy-plane (z=0) meets xz-plane (y=0) in the x-axis.
    let out = run_op(
        "intersect_operator",
        vec![
            vec![],
            plane(vec![0.0, 0.0, 0.0], &[0, 1]),
            plane(vec![0.0, 0.0, 0.0], &[0, 2]),
        ],
    )
    .expect("intersect should succeed");
    let line = decode_subspace(&out).unwrap();
    assert_eq!(line.rank(), 1);
    assert!(parallel_to(line.basis_vector(0), [1.0, 0.0, 0.0]));
}

#[test]
fn intersect_of_parallel_planes_errors() {
    // z=0 and z=1 never meet.
    let err = run_op(
        "intersect_operator",
        vec![
            vec![],
            plane(vec![0.0, 0.0, 0.0], &[0, 1]),
            plane(vec![0.0, 0.0, 1.0], &[0, 1]),
        ],
    )
    .expect_err("parallel planes do not intersect");
    assert!(err.contains("intersect"), "unexpected error: {err}");
}
