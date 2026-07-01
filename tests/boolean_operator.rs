//! Integration tests for the boolean operator's merged-model output.
//!
//! These run the real boolean_operator.wasm against the bundled sphere and
//! torus models and sample the merged result through the engine's executor,
//! so they exercise the full merge path including cross-module memory access
//! (regression test for github issue #1: model B read the sample position
//! from its own memory, which the host never wrote).
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p simple_sphere_model -p simple_torus_model -p boolean_operator

#![cfg(feature = "native")]

use volumetric::wasm::{
    OperatorExecutor, OperatorIo, ParallelModelSampler, create_operator_executor,
    create_parallel_sampler,
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

fn cbor_op(op: &str) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text("op".into()),
            ciborium::value::Value::Text(op.into()),
        )]),
        &mut out,
    )
    .unwrap();
    out
}

/// Run the boolean operator on models A and B and return the merged wasm.
fn merge(a: Vec<u8>, b: Vec<u8>, op: &str) -> Vec<u8> {
    let boolean = wasm_artifact("boolean_operator");
    let mut executor = create_operator_executor(&boolean).expect("create boolean executor");
    let io = OperatorIo::new(vec![a, b, cbor_op(op)]);
    let result = executor.run(io).expect("run boolean operator");
    result
        .outputs
        .get(&0)
        .expect("boolean operator posted no output")
        .clone()
}

/// Run the boolean operator on the sphere (A) and torus (B) and return the
/// merged model wasm.
fn merge_sphere_torus(op: &str) -> Vec<u8> {
    merge(
        wasm_artifact("simple_sphere_model"),
        wasm_artifact("simple_torus_model"),
        op,
    )
}

fn sample(merged: &[u8], x: f64, y: f64, z: f64) -> f32 {
    let sampler = create_parallel_sampler(merged).expect("instantiate merged model");
    sampler.sample(x, y, z)
}

// Sphere: radius 1.0 at origin. Torus: R=1.0, r=0.35 in the XZ plane, so it
// reaches x/z = ±1.35 and y = ±0.35.

#[test]
fn union_includes_regions_exclusive_to_each_input() {
    let merged = merge_sphere_torus("union");

    // Inside torus only (outside the unit sphere).
    assert!(
        sample(&merged, 1.2, 0.0, 0.0) > 0.5,
        "torus-only point lost"
    );
    // Inside sphere only (torus spans just y ±0.35).
    assert!(
        sample(&merged, 0.0, 0.9, 0.0) > 0.5,
        "sphere-only point lost"
    );
    // Outside both.
    assert!(
        sample(&merged, 2.0, 2.0, 2.0) < 0.5,
        "exterior point inside"
    );
}

#[test]
fn union_bounds_cover_both_inputs() {
    let merged = merge_sphere_torus("union");
    let sampler = create_parallel_sampler(&merged).expect("instantiate merged model");
    let bounds = sampler.get_bounds().expect("get merged bounds");

    // x/z from the torus (±1.35), y from the sphere (±1.0).
    assert!((bounds.min.0 - -1.35).abs() < 1e-9, "min x: {:?}", bounds);
    assert!((bounds.max.0 - 1.35).abs() < 1e-9, "max x: {:?}", bounds);
    assert!((bounds.min.1 - -1.0).abs() < 1e-9, "min y: {:?}", bounds);
    assert!((bounds.max.1 - 1.0).abs() < 1e-9, "max y: {:?}", bounds);
    assert!((bounds.min.2 - -1.35).abs() < 1e-9, "min z: {:?}", bounds);
    assert!((bounds.max.2 - 1.35).abs() < 1e-9, "max z: {:?}", bounds);
}

#[test]
fn intersect_keeps_only_overlap() {
    let merged = merge_sphere_torus("intersect");

    // Inside both: on the torus ring, still inside the sphere.
    assert!(sample(&merged, 0.9, 0.0, 0.0) > 0.5, "overlap point lost");
    // Torus only.
    assert!(
        sample(&merged, 1.2, 0.0, 0.0) < 0.5,
        "torus-only point kept"
    );
    // Sphere only.
    assert!(
        sample(&merged, 0.0, 0.9, 0.0) < 0.5,
        "sphere-only point kept"
    );
}

#[test]
fn merged_model_composes_as_input_a() {
    // (sphere ∪ torus) − sphere = the torus region outside the sphere.
    let union = merge_sphere_torus("union");
    let carved = merge(union, wasm_artifact("simple_sphere_model"), "subtract");

    assert!(
        sample(&carved, 1.2, 0.0, 0.0) > 0.5,
        "torus-only point lost"
    );
    assert!(
        sample(&carved, 0.9, 0.0, 0.0) < 0.5,
        "point inside sphere kept"
    );
    assert!(
        sample(&carved, 0.0, 0.9, 0.0) < 0.5,
        "sphere-only point kept"
    );
}

#[test]
fn merged_model_composes_as_input_b() {
    // sphere ∩ (sphere ∪ torus) = sphere.
    let union = merge_sphere_torus("union");
    let clipped = merge(wasm_artifact("simple_sphere_model"), union, "intersect");

    assert!(sample(&clipped, 0.0, 0.9, 0.0) > 0.5, "sphere point lost");
    assert!(
        sample(&clipped, 1.2, 0.0, 0.0) < 0.5,
        "torus-only point kept"
    );
}

#[test]
fn subtract_removes_torus_from_sphere() {
    let merged = merge_sphere_torus("subtract");

    // Sphere point carved out by the torus.
    assert!(
        sample(&merged, 0.9, 0.0, 0.0) < 0.5,
        "carved point still inside"
    );
    // Sphere point clear of the torus.
    assert!(sample(&merged, 0.0, 0.9, 0.0) > 0.5, "sphere point lost");
    // Torus-only point stays outside.
    assert!(sample(&merged, 1.2, 0.0, 0.0) < 0.5, "torus point inside");
}
