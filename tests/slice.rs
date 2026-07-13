//! Integration tests for slice_operator: re-expressing a model in a
//! subspace's chart — planar cross-sections, line profiles, and full-frame
//! rigid re-expressions.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::subspace::{Subspace, encode_subspace};
use volumetric::wasm::{
    ModelExecutor, NativeModelExecutor, OperatorExecutor, OperatorIo, create_model_executor,
    create_operator_executor,
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

fn run_slice(model: &str, subspace: &Subspace) -> Result<Vec<u8>, String> {
    let operator_wasm = wasm_artifact("slice_operator");
    let mut executor = create_operator_executor(&operator_wasm).expect("operator executor");
    let result = executor
        .run(OperatorIo::new(vec![
            wasm_artifact(model),
            encode_subspace(subspace),
        ]))
        .map_err(|e| e.to_string())?;
    result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "operator posted no output".to_string())
}

fn occupied_at(executor: &mut impl ModelExecutor, p: &[f64]) -> bool {
    volumetric::is_occupied(executor.sample_nd(p).expect("sample"))
}

/// The classic cross-section: the unit sphere sliced on the z = 0 plane
/// is a 2D disc of radius 1.
#[test]
fn plane_slice_of_a_sphere_is_a_disc() {
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1]).unwrap();
    let wasm = run_slice("simple_sphere_model", &plane).expect("slice failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    assert_eq!(model.dimensions().expect("dims"), 2);
    let bounds = model.get_bounds_nd().expect("bounds");
    assert_eq!(bounds.min(0), -1.0);
    assert_eq!(bounds.max(0), 1.0);
    assert_eq!(bounds.min(1), -1.0);
    assert_eq!(bounds.max(1), 1.0);

    assert!(occupied_at(&mut model, &[0.0, 0.0]));
    assert!(occupied_at(&mut model, &[0.9, 0.0]));
    assert!(occupied_at(&mut model, &[0.0, -0.9]));
    assert!(!occupied_at(&mut model, &[1.05, 0.0]));
    assert!(!occupied_at(&mut model, &[0.8, 0.8]));
}

/// An offset plane cuts a smaller cross-section: at z = 0.8 the sphere's
/// disc has radius 0.6.
#[test]
fn offset_plane_slices_the_smaller_cap() {
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.8], &[0, 1]).unwrap();
    let wasm = run_slice("simple_sphere_model", &plane).expect("slice failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    assert!(occupied_at(&mut model, &[0.5, 0.0]));
    assert!(!occupied_at(&mut model, &[0.7, 0.0]));
}

/// A line subspace gives a 1D profile: the sphere along the z axis
/// through the origin is occupied for |t| < 1.
#[test]
fn line_slice_is_a_profile() {
    let line = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[2]).unwrap();
    let wasm = run_slice("simple_sphere_model", &line).expect("slice failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    assert_eq!(model.dimensions().expect("dims"), 1);
    let bounds = model.get_bounds_nd().expect("bounds");
    assert_eq!(bounds.min(0), -1.0);
    assert_eq!(bounds.max(0), 1.0);
    assert!(occupied_at(&mut model, &[0.0]));
    assert!(occupied_at(&mut model, &[0.9]));
    assert!(!occupied_at(&mut model, &[1.1]));
}

/// A full frame re-expresses the model rigidly. The density-gradient cube
/// ([-1,1]^3) in a frame rotated 45 degrees about z: chart u runs along
/// the cube's xy diagonal, so the corner sits at u = sqrt(2).
#[test]
fn frame_slice_is_a_rigid_reexpression() {
    let s = 0.5f64.sqrt();
    let frame = Subspace {
        dimensions: 3,
        origin: vec![0.0, 0.0, 0.0],
        basis: vec![s, s, 0.0, -s, s, 0.0, 0.0, 0.0, 1.0],
    };
    frame.validate().unwrap();
    let wasm = run_slice("density_gradient_model", &frame).expect("slice failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    assert_eq!(model.dimensions().expect("dims"), 3);
    // Interval-arithmetic bounds: the rotated box's chart enclosure is
    // +/- sqrt(2) in u and v, +/- 1 in w.
    let bounds = model.get_bounds_nd().expect("bounds");
    let diag = 2.0 * s;
    assert!((bounds.min(0) - -diag).abs() < 1e-12);
    assert!((bounds.max(0) - diag).abs() < 1e-12);
    assert_eq!(bounds.min(2), -1.0);

    // Chart (0.9 * sqrt(2), 0, 0) is world (0.9, 0.9, 0): inside the cube.
    assert!(occupied_at(&mut model, &[0.9 * diag, 0.0, 0.0]));
    // Chart (1.05 * sqrt(2), 0, 0) is world (1.05, 1.05, 0): outside.
    assert!(!occupied_at(&mut model, &[1.05 * diag, 0.0, 0.0]));
    // Straight up is unrotated.
    assert!(occupied_at(&mut model, &[0.0, 0.0, 0.9]));
    assert!(!occupied_at(&mut model, &[0.0, 0.0, 1.1]));
}

/// Typed channels pass through: a planar slice of the density-gradient
/// cube keeps its density channel, sampled at the embedded position.
#[test]
fn channels_pass_through_the_slice() {
    // Chart u = x, v = y at z = 0; density = 0.5 + 0.5x.
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1]).unwrap();
    let wasm = run_slice("density_gradient_model", &plane).expect("slice failed");
    let mut model = NativeModelExecutor::new(&wasm).expect("model executor");

    assert_eq!(model.sample_format().channels.len(), 2);
    let row = model.sample_channels_nd(&[0.5, 0.0]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.75).abs() < 1e-6, "density was {}", row[1]);
}

/// Point subspaces and ambient-dimension mismatches are errors.
#[test]
fn degenerate_slices_are_rejected() {
    let point = Subspace::point(vec![0.0, 0.0, 0.0]);
    let err = run_slice("simple_sphere_model", &point).expect_err("rank 0 must fail");
    assert!(err.contains("point"), "unexpected error: {err}");

    let flat = Subspace::axis_aligned(vec![0.0, 0.0], &[0]).unwrap();
    let err = run_slice("simple_sphere_model", &flat).expect_err("2-space vs 3D must fail");
    assert!(err.contains("dimensions"), "unexpected error: {err}");
}

/// The 2D preview raster of a channeled slice: colormapping by the
/// density-kind channel (named "infill" in the gradient model) yields
/// the input's gradient masked to the occupied region — this is what
/// the UI's "Color by" renders for 2D outputs.
#[test]
fn channel_raster_of_a_slice_shows_the_gradient() {
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1]).unwrap();
    let wasm = run_slice("density_gradient_model", &plane).expect("slice failed");

    let raster = volumetric::rasterize_sketch_channel_from_bytes(&wasm, 64, Some("infill"))
        .expect("channel raster failed");
    // density = 0.5 + 0.5x over x in [-1, 1], sampled at cell centers.
    assert!(raster.value_min < 0.05, "min was {}", raster.value_min);
    assert!(raster.value_max > 0.95, "max was {}", raster.value_max);
    let mid = raster.value(raster.width / 2, raster.height / 2);
    assert!((mid - 0.5).abs() < 0.05, "center density was {mid}");

    // Unknown channels are an error, not a silent occupancy raster.
    assert!(volumetric::rasterize_sketch_channel_from_bytes(&wasm, 64, Some("nope")).is_err());
}
