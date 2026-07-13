//! Integration tests for subspace_operator and model_bound_operator: the
//! Subspace value type produced from numeric inputs and from a model's
//! bounding box.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::subspace::{Subspace, decode_subspace};
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

fn cbor_map(config: &[(&str, ciborium::value::Value)]) -> Vec<u8> {
    let mut cfg = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(
            config
                .iter()
                .map(|(k, v)| (ciborium::value::Value::Text((*k).into()), v.clone()))
                .collect(),
        ),
        &mut cfg,
    )
    .unwrap();
    cfg
}

fn vec3_bytes(v: [f64; 3]) -> Vec<u8> {
    v.iter().flat_map(|c| c.to_le_bytes()).collect()
}

fn run_operator(name: &str, inputs: Vec<Vec<u8>>) -> Result<Subspace, String> {
    let operator_wasm = wasm_artifact(name);
    let mut executor = create_operator_executor(&operator_wasm).expect("operator executor");
    let result = executor
        .run(OperatorIo::new(inputs))
        .map_err(|e| e.to_string())?;
    let bytes = result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "operator posted no output".to_string())?;
    decode_subspace(&bytes)
}

/// The numeric-input operator builds each kind, orthonormalizing raw
/// direction vectors.
#[test]
fn subspace_operator_builds_a_plane() {
    let subspace = run_operator(
        "subspace_operator",
        vec![
            cbor_map(&[("kind", ciborium::value::Value::Text("plane".into()))]),
            vec3_bytes([1.0, 2.0, 3.0]),
            // Unnormalized and non-orthogonal on purpose.
            vec3_bytes([0.0, 0.0, 5.0]),
            vec3_bytes([2.0, 0.0, 1.0]),
        ],
    )
    .expect("subspace run failed");

    assert_eq!(subspace.dimensions, 3);
    assert_eq!(subspace.rank(), 2);
    assert_eq!(subspace.origin, vec![1.0, 2.0, 3.0]);
    assert_eq!(subspace.basis, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    // z cross x = y.
    assert_eq!(subspace.normal().unwrap(), vec![0.0, 1.0, 0.0]);
}

/// Empty vector inputs fall back to defaults (origin at zero, x/y plane).
#[test]
fn subspace_operator_defaults_to_the_xy_plane() {
    let subspace = run_operator(
        "subspace_operator",
        vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()],
    )
    .expect("subspace run failed");
    assert_eq!(subspace.rank(), 2);
    assert_eq!(subspace.origin, vec![0.0; 3]);
    assert_eq!(subspace.normal().unwrap(), vec![0.0, 0.0, 1.0]);
}

/// Parallel plane directions are an error, not a silently patched axis.
#[test]
fn subspace_operator_rejects_degenerate_directions() {
    let err = run_operator(
        "subspace_operator",
        vec![
            cbor_map(&[("kind", ciborium::value::Value::Text("plane".into()))]),
            Vec::new(),
            vec3_bytes([1.0, 0.0, 0.0]),
            vec3_bytes([-3.0, 0.0, 0.0]),
        ],
    )
    .expect_err("parallel directions must fail");
    assert!(err.contains("degenerate"), "unexpected error: {err}");
}

/// The default model-bound config selects the bottom face: the unit
/// sphere's print-bed plane at z = -1, centered in x/y.
#[test]
fn model_bound_defaults_to_the_bottom_face() {
    let subspace = run_operator(
        "model_bound_operator",
        vec![wasm_artifact("simple_sphere_model"), Vec::new()],
    )
    .expect("model bound run failed");

    assert_eq!(subspace.rank(), 2);
    assert_eq!(subspace.origin, vec![0.0, 0.0, -1.0]);
    assert_eq!(subspace.basis, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
}

/// Explicit selectors pick edges and corners.
#[test]
fn model_bound_selects_edges_and_corners() {
    let text = |s: &str| ciborium::value::Value::Text(s.into());
    let edge = run_operator(
        "model_bound_operator",
        vec![
            wasm_artifact("simple_sphere_model"),
            cbor_map(&[("x", text("min")), ("y", text("max")), ("z", text("span"))]),
        ],
    )
    .expect("model bound run failed");
    assert_eq!(edge.rank(), 1);
    assert_eq!(edge.origin, vec![-1.0, 1.0, 0.0]);
    assert_eq!(edge.basis, vec![0.0, 0.0, 1.0]);

    let corner = run_operator(
        "model_bound_operator",
        vec![
            wasm_artifact("simple_sphere_model"),
            cbor_map(&[("x", text("max")), ("y", text("max")), ("z", text("max"))]),
        ],
    )
    .expect("model bound run failed");
    assert_eq!(corner.rank(), 0);
    assert_eq!(corner.origin, vec![1.0, 1.0, 1.0]);
}
