//! Integration tests for the boolean operator's merged-model output.
//!
//! These run the real boolean_operator.wasm against the bundled sphere,
//! torus and rounded-box models and sample the merged result through the
//! engine's executor, so they exercise the full merge path including
//! cross-module memory access (regression test for github issue #1: model B
//! read the sample position from its own memory, which the host never
//! wrote) and the N-ary glue (any number of models before the config).
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::wasm::{
    NativeModelExecutor, OperatorExecutor, OperatorIo, ParallelModelSampler,
    create_operator_executor, create_parallel_sampler,
};

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

fn cbor_text(field: &str, value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text(field.into()),
            ciborium::value::Value::Text(value.into()),
        )]),
        &mut out,
    )
    .unwrap();
    out
}

fn cbor_op(op: &str) -> Vec<u8> {
    cbor_text("op", op)
}

/// Run the boolean operator on `models` (in slot order, followed by the
/// config) and return the merged wasm.
fn try_merge(models: Vec<Vec<u8>>, op: &str) -> Result<Vec<u8>, String> {
    let boolean = wasm_artifact("boolean_operator");
    let mut executor = create_operator_executor(&boolean).expect("create boolean executor");
    let mut inputs = models;
    inputs.push(cbor_op(op));
    let result = executor
        .run(OperatorIo::new(inputs))
        .map_err(|e| e.to_string())?;
    result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "boolean operator posted no output".to_string())
}

fn merge(models: Vec<Vec<u8>>, op: &str) -> Vec<u8> {
    try_merge(models, op).expect("run boolean operator")
}

fn sphere() -> Vec<u8> {
    wasm_artifact("simple_sphere_model")
}

fn torus() -> Vec<u8> {
    wasm_artifact("simple_torus_model")
}

fn rounded_box() -> Vec<u8> {
    wasm_artifact("rounded_box_model")
}

/// Run the boolean operator on the sphere (A) and torus (B) and return the
/// merged model wasm.
fn merge_sphere_torus(op: &str) -> Vec<u8> {
    merge(vec![sphere(), torus()], op)
}

fn sample(merged: &[u8], x: f64, y: f64, z: f64) -> f32 {
    let sampler = create_parallel_sampler(merged).expect("instantiate merged model");
    sampler.sample(x, y, z)
}

fn inside(merged: &[u8], p: (f64, f64, f64)) -> bool {
    sample(merged, p.0, p.1, p.2) > 0.5
}

// Sphere: radius 1.0 at origin. Torus: R=1.0, r=0.35 in the XZ plane, so it
// reaches x/z = ±1.35 and y = ±0.35. Rounded box: core half-extents
// (0.9, 0.6, 0.4) rounded by 0.2, so it reaches (±1.1, ±0.8, ±0.6).
const SPHERE_ONLY: (f64, f64, f64) = (0.0, 0.9, 0.0);
const TORUS_ONLY: (f64, f64, f64) = (1.2, 0.0, 0.0);
/// Inside the box's rounded corner region, outside the unit sphere
/// (|p| ≈ 1.14) and clear of the torus (y > 0.35).
const BOX_ONLY: (f64, f64, f64) = (0.9, 0.7, 0.0);
const IN_ALL_THREE: (f64, f64, f64) = (0.9, 0.0, 0.0);
const OUTSIDE_ALL: (f64, f64, f64) = (2.0, 2.0, 2.0);

#[test]
fn union_includes_regions_exclusive_to_each_input() {
    let merged = merge_sphere_torus("union");
    assert!(inside(&merged, TORUS_ONLY), "torus-only point lost");
    assert!(inside(&merged, SPHERE_ONLY), "sphere-only point lost");
    assert!(!inside(&merged, OUTSIDE_ALL), "exterior point inside");
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
    assert!(inside(&merged, IN_ALL_THREE), "overlap point lost");
    assert!(!inside(&merged, TORUS_ONLY), "torus-only point kept");
    assert!(!inside(&merged, SPHERE_ONLY), "sphere-only point kept");
}

#[test]
fn merged_model_composes_as_input_a() {
    // (sphere ∪ torus) − sphere = the torus region outside the sphere.
    let union = merge_sphere_torus("union");
    let carved = merge(vec![union, sphere()], "subtract");
    assert!(inside(&carved, TORUS_ONLY), "torus-only point lost");
    assert!(!inside(&carved, IN_ALL_THREE), "point inside sphere kept");
    assert!(!inside(&carved, SPHERE_ONLY), "sphere-only point kept");
}

#[test]
fn merged_model_composes_as_input_b() {
    // sphere ∩ (sphere ∪ torus) = sphere.
    let union = merge_sphere_torus("union");
    let clipped = merge(vec![sphere(), union], "intersect");
    assert!(inside(&clipped, SPHERE_ONLY), "sphere point lost");
    assert!(!inside(&clipped, TORUS_ONLY), "torus-only point kept");
}

#[test]
fn merge_failure_is_reported_not_swallowed() {
    let err = try_merge(vec![sphere(), b"not a wasm module".to_vec()], "union")
        .expect_err("merging garbage should fail");
    assert!(
        err.contains("model merge failed"),
        "unexpected error: {err}"
    );
}

#[test]
fn subtract_removes_torus_from_sphere() {
    let merged = merge_sphere_torus("subtract");
    assert!(!inside(&merged, IN_ALL_THREE), "carved point still inside");
    assert!(inside(&merged, SPHERE_ONLY), "sphere point lost");
    assert!(!inside(&merged, TORUS_ONLY), "torus point inside");
}

#[test]
fn union_of_three_models_covers_each() {
    let merged = merge(vec![sphere(), torus(), rounded_box()], "union");
    assert!(inside(&merged, SPHERE_ONLY), "sphere-only point lost");
    assert!(inside(&merged, TORUS_ONLY), "torus-only point lost");
    assert!(inside(&merged, BOX_ONLY), "box-only point lost");
    assert!(!inside(&merged, OUTSIDE_ALL), "exterior point inside");

    let sampler = create_parallel_sampler(&merged).expect("instantiate merged model");
    let bounds = sampler.get_bounds().expect("get merged bounds");
    // x/z from the torus (±1.35), y from the sphere (±1.0); the box
    // (±1.1, ±0.8, ±0.6) lies within those on every axis, so nothing widens.
    assert!((bounds.max.0 - 1.35).abs() < 1e-9, "max x: {:?}", bounds);
    assert!((bounds.max.1 - 1.0).abs() < 1e-9, "max y: {:?}", bounds);
    assert!((bounds.min.2 - -1.35).abs() < 1e-9, "min z: {:?}", bounds);
}

#[test]
fn intersect_of_three_models_needs_all() {
    let merged = merge(vec![sphere(), torus(), rounded_box()], "intersect");
    assert!(inside(&merged, IN_ALL_THREE), "common point lost");
    for point in [SPHERE_ONLY, TORUS_ONLY, BOX_ONLY] {
        assert!(!inside(&merged, point), "{point:?} is not in every model");
    }

    let sampler = create_parallel_sampler(&merged).expect("instantiate merged model");
    let bounds = sampler.get_bounds().expect("get merged bounds");
    // Tightest on each axis: sphere x (±1), torus y (±0.35), box z (±0.6).
    assert!((bounds.max.0 - 1.0).abs() < 1e-9, "max x: {:?}", bounds);
    assert!((bounds.min.1 - -0.35).abs() < 1e-9, "min y: {:?}", bounds);
    assert!((bounds.max.2 - 0.6).abs() < 1e-9, "max z: {:?}", bounds);
}

#[test]
fn subtract_removes_every_later_model_from_the_first() {
    // sphere − torus − box: only sphere points clear of both remain.
    let merged = merge(vec![sphere(), torus(), rounded_box()], "subtract");
    assert!(
        !inside(&merged, IN_ALL_THREE),
        "point in torus and box kept"
    );
    // (0, 0.5, 0) is inside the box core and the sphere, and 1.1 from the
    // torus ring: carved by the box alone.
    assert!(
        !inside(&merged, (0.0, 0.5, 0.0)),
        "point carved by the box kept"
    );
    // (0, 0.9, 0) is past the box's y reach (0.6 core + 0.2 round) and off
    // the torus: survives.
    assert!(inside(&merged, SPHERE_ONLY), "sphere-only point lost");
    // Deep in the sphere but past the box's z reach (0.6) and off the
    // torus ring: survives.
    assert!(inside(&merged, (0.0, 0.3, 0.7)), "sphere point lost");
    assert!(!inside(&merged, BOX_ONLY), "box-only point inside");

    let sampler = create_parallel_sampler(&merged).expect("instantiate merged model");
    let bounds = sampler.get_bounds().expect("get merged bounds");
    for (min, max) in [
        (bounds.min.0, bounds.max.0),
        (bounds.min.1, bounds.max.1),
        (bounds.min.2, bounds.max.2),
    ] {
        assert!(
            (min + 1.0).abs() < 1e-9 && (max - 1.0).abs() < 1e-9,
            "{bounds:?}"
        );
    }
}

#[test]
fn single_model_passes_through_under_every_operation() {
    for op in ["union", "intersect", "subtract"] {
        let merged = merge(vec![torus()], op);
        assert!(inside(&merged, TORUS_ONLY), "{op}: torus point lost");
        assert!(!inside(&merged, SPHERE_ONLY), "{op}: exterior point inside");
    }
}

#[test]
fn unwired_model_entries_are_skipped() {
    let merged = merge(vec![sphere(), Vec::new(), torus()], "union");
    assert!(inside(&merged, SPHERE_ONLY));
    assert!(inside(&merged, TORUS_ONLY));

    let err = try_merge(vec![Vec::new(), Vec::new()], "union").expect_err("nothing wired");
    assert!(err.contains("no models wired"), "unexpected error: {err}");
}

/// The glue reads the dimension count at run time, so 2D sketches merge
/// with 2D bounds (the old two-model glue hardcoded six bound values).
#[test]
fn two_d_models_merge_with_two_d_bounds() {
    let sketch = |path: &str| -> Vec<u8> {
        let operator = wasm_artifact("path_sketch_operator");
        let mut executor = create_operator_executor(&operator).expect("create sketch executor");
        let result = executor
            .run(OperatorIo::new(vec![cbor_text("path", path)]))
            .expect("run path sketch");
        result.outputs.get(&0).cloned().expect("sketch output")
    };
    let left = sketch("M 0 0 H 1 V 1 H 0 Z");
    let right = sketch("M 2 0 H 3 V 2 H 2 Z");
    let merged = merge(vec![left, right], "union");

    let mut executor = NativeModelExecutor::new(&merged).expect("instantiate merged sketch");
    assert_eq!(executor.dimensions(), 2);
    let bounds = executor.get_bounds_nd().expect("2D bounds");
    assert_eq!(bounds.dimensions(), 2);
    assert!((bounds.min(0) - 0.0).abs() < 1e-9, "{bounds:?}");
    assert!((bounds.max(0) - 3.0).abs() < 1e-9, "{bounds:?}");
    assert!((bounds.min(1) - 0.0).abs() < 1e-9, "{bounds:?}");
    assert!((bounds.max(1) - 2.0).abs() < 1e-9, "{bounds:?}");
    assert_eq!(executor.sample_nd(&[0.5, 0.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[2.5, 1.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.5, 0.5]).unwrap(), 0.0);
}

#[test]
fn metadata_declares_a_variadic_model_block() {
    let metadata =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("boolean_operator"))
            .expect("boolean metadata");
    assert_eq!(metadata.variadic_slot(), Some(0));
    assert!(metadata.accepts_input_count(2));
    assert!(metadata.accepts_input_count(9));
    assert!(!metadata.accepts_input_count(1));
    assert_eq!(metadata.input_label(3, 5).as_deref(), Some("Model 4"));
    assert_eq!(metadata.input_label(4, 5).as_deref(), Some("Config"));
}
