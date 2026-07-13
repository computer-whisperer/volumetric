//! Integration tests for island_removal_operator: ablating geometry not
//! sufficiently supported along the build direction.
//!
//! Island scenarios are composed from the bundled models with the
//! translate/scale/boolean/slice operators (a single connected model
//! always seeds its own bottom layer as the bed, so islands need two
//! pieces at different heights).
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

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

fn cbor(config: &[(&str, ciborium::value::Value)]) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(
            config
                .iter()
                .map(|(k, v)| (ciborium::value::Value::Text((*k).into()), v.clone()))
                .collect(),
        ),
        &mut out,
    )
    .unwrap();
    out
}

/// Run `operator` on the given inputs and return output 0.
fn run_operator(operator: &str, inputs: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    let operator_wasm = wasm_artifact(operator);
    let mut executor = create_operator_executor(&operator_wasm).expect("operator executor");
    let result = executor
        .run(OperatorIo::new(inputs))
        .map_err(|e| e.to_string())?;
    result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "operator posted no output".to_string())
}

fn float(v: f64) -> ciborium::value::Value {
    ciborium::value::Value::Float(v)
}

fn int(v: i64) -> ciborium::value::Value {
    ciborium::value::Value::Integer(v.into())
}

fn text(v: &str) -> ciborium::value::Value {
    ciborium::value::Value::Text(v.into())
}

fn translate(model: Vec<u8>, d: [f64; 3]) -> Vec<u8> {
    let cfg = cbor(&[("dx", float(d[0])), ("dy", float(d[1])), ("dz", float(d[2]))]);
    run_operator("translate_operator", vec![model, cfg]).expect("translate")
}

fn scale(model: Vec<u8>, s: [f64; 3]) -> Vec<u8> {
    let cfg = cbor(&[("sx", float(s[0])), ("sy", float(s[1])), ("sz", float(s[2]))]);
    run_operator("scale_operator", vec![model, cfg]).expect("scale")
}

fn union(a: Vec<u8>, b: Vec<u8>) -> Vec<u8> {
    let cfg = cbor(&[("op", text("union"))]);
    run_operator("boolean_operator", vec![a, b, cfg]).expect("union")
}

fn run_island_removal(
    model: Vec<u8>,
    config: &[(&str, ciborium::value::Value)],
) -> Result<Vec<u8>, String> {
    run_operator("island_removal_operator", vec![model, cbor(config)])
}

fn base_config() -> Vec<(&'static str, ciborium::value::Value)> {
    vec![("resolution", int(64))]
}

/// The unit cube on the bed plus a sphere floating well above it.
fn cube_with_floating_sphere() -> Vec<u8> {
    let sphere = translate(wasm_artifact("simple_sphere_model"), [0.0, 0.0, 3.0]);
    union(wasm_artifact("density_gradient_model"), sphere)
}

fn occupied_at(executor: &mut impl ModelExecutor, p: &[f64]) -> bool {
    volumetric::is_occupied(executor.sample_nd(p).expect("sample"))
}

/// A floating sphere is an island: the default pass ablates it and keeps
/// the bed-supported cube intact. The ablated output keeps the input's
/// bounds (ablation only shrinks geometry).
#[test]
fn floating_island_is_ablated() {
    let input = cube_with_floating_sphere();
    let wasm = run_island_removal(input, &base_config()).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    assert!(occupied_at(&mut model, &[0.0, 0.0, 0.0]), "the cube");
    assert!(occupied_at(&mut model, &[0.9, 0.9, 0.9]), "cube corner");
    assert!(
        !occupied_at(&mut model, &[0.0, 0.0, 3.0]),
        "floating sphere center must be ablated"
    );
    assert!(!occupied_at(&mut model, &[0.5, 0.0, 3.0]), "sphere flank");

    let bounds = model.get_bounds_nd().expect("bounds");
    assert_eq!(bounds.min(2), -1.0);
    assert_eq!(bounds.max(2), 4.0, "ablated output keeps input bounds");
}

/// A fully supported model passes through byte-identical — no mask, no
/// merge, no resampling error.
#[test]
fn nothing_removed_passes_through() {
    let input = wasm_artifact("density_gradient_model");
    let wasm = run_island_removal(input.clone(), &base_config()).expect("island removal failed");
    assert_eq!(wasm, input, "fully supported input must pass through");
}

/// `output: "islands"` isolates exactly the removed geometry, bounded by
/// the mask's crop box intersected with the input bounds.
#[test]
fn islands_output_isolates_the_removed_geometry() {
    let input = cube_with_floating_sphere();
    let mut config = base_config();
    config.push(("output", text("islands")));
    let wasm = run_island_removal(input, &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    assert!(
        occupied_at(&mut model, &[0.0, 0.0, 3.0]),
        "the island itself"
    );
    assert!(
        !occupied_at(&mut model, &[0.0, 0.0, 0.0]),
        "kept geometry is not in the islands output"
    );

    // Crop box: the sphere (z in [2, 4]) plus a few cells, intersected
    // with the input bounds.
    let bounds = model.get_bounds_nd().expect("bounds");
    assert!(
        bounds.min(2) > 1.5 && bounds.min(2) <= 2.0,
        "z min {} should hug the island",
        bounds.min(2)
    );
    assert!((bounds.max(2) - 4.0).abs() < 1e-9);
    assert_eq!(bounds.min(0), -1.0, "crop clamps to the input bounds");
}

/// A mushroom (wide slab on a narrow post) gets its cantilever shaved to
/// the overhang limit at 45 degrees, and survives whole at 80.
#[test]
fn overhang_shaving_respects_the_angle() {
    // Post: [-0.25, 0.25]^2 x [-1, 1]. Slab: [-1, 1]^2 x [1.0, 1.3].
    let cube = || wasm_artifact("density_gradient_model");
    let post = scale(cube(), [0.25, 0.25, 1.0]);
    let slab = translate(scale(cube(), [1.0, 1.0, 0.15]), [0.0, 0.0, 1.15]);
    let mushroom = union(post, slab);

    let wasm =
        run_island_removal(mushroom.clone(), &base_config()).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(occupied_at(&mut model, &[0.0, 0.0, 0.0]), "the post");
    assert!(
        occupied_at(&mut model, &[0.0, 0.0, 1.15]),
        "slab above the post"
    );
    assert!(
        occupied_at(&mut model, &[0.35, 0.0, 1.15]),
        "cantilever within the 45-degree reach"
    );
    assert!(
        !occupied_at(&mut model, &[0.9, 0.0, 1.15]),
        "cantilever past the 45-degree reach must be shaved"
    );

    let mut config = base_config();
    config.push(("overhang_angle", float(80.0)));
    let wasm = run_island_removal(mushroom, &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(
        occupied_at(&mut model, &[0.9, 0.0, 1.15]),
        "an 80-degree allowance keeps the whole slab"
    );
}

/// `extreme: "max"` grows support from the top: now the sphere hangs
/// from the ceiling and the cube below the gap is the island.
#[test]
fn extreme_max_supports_from_the_top() {
    let input = cube_with_floating_sphere();
    let mut config = base_config();
    config.push(("extreme", text("max")));
    let wasm = run_island_removal(input, &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    assert!(
        occupied_at(&mut model, &[0.0, 0.0, 3.0]),
        "the ceiling-supported sphere"
    );
    assert!(
        !occupied_at(&mut model, &[0.0, 0.0, 0.0]),
        "the cube below the gap must be ablated"
    );
}

/// The user-reported island_fail_demo_0 scenario: two spheres stacked
/// in y with a gap, build axis y — the upper sphere is the island. Sent
/// with the exact full config map the UI's schema editor produces
/// (regression: a `? axis` schema key once made the axis control a
/// silently ignored unknown field).
#[test]
fn y_axis_ablates_the_upper_of_two_stacked_spheres() {
    let lifted = translate(wasm_artifact("simple_sphere_model"), [0.0, 3.0, 0.0]);
    let spheres = union(wasm_artifact("simple_sphere_model"), lifted);
    let config = vec![
        ("overhang_angle", float(45.0)),
        ("axis", text("y")),
        ("extreme", text("min")),
        ("resolution", int(64)),
        ("output", text("ablated")),
    ];
    let wasm = run_island_removal(spheres.clone(), &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(
        occupied_at(&mut model, &[0.0, 0.0, 0.0]),
        "the bed sphere's center column"
    );
    assert!(
        !occupied_at(&mut model, &[0.0, 3.0, 0.0]),
        "the sphere floating in +y must be ablated"
    );

    // The numeric axis form selects the same axis.
    let config = vec![("axis", int(1)), ("resolution", int(64))];
    let wasm = run_island_removal(spheres, &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(!occupied_at(&mut model, &[0.0, 3.0, 0.0]));
}

/// `mode: "island"` is the resin semantics: arbitrary horizontal
/// overhangs survive whole (in-plane cohesion), but genuinely detached
/// slice regions still die — even ones a near-90 overhang_angle would
/// leak past by reading distant geometry as lateral support.
#[test]
fn island_mode_keeps_overhangs_and_removes_detached_regions() {
    // The mushroom's full slab survives island mode at any angle.
    let cube = || wasm_artifact("density_gradient_model");
    let post = scale(cube(), [0.25, 0.25, 1.0]);
    let slab = translate(scale(cube(), [1.0, 1.0, 0.15]), [0.0, 0.0, 1.15]);
    let mushroom = union(post, slab);
    let mut config = base_config();
    config.push(("mode", text("island")));
    let wasm = run_island_removal(mushroom, &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(
        occupied_at(&mut model, &[0.9, 0.0, 1.15]),
        "a fully horizontal cantilever is fine on resin"
    );

    // The leak case a huge overhang_angle cannot handle: a detached blob
    // whose layers also contain a tall column elsewhere. Island mode
    // removes the blob; near-90 overhang mode keeps it (below).
    let tall = translate(scale(cube(), [0.1, 0.1, 1.0]), [-0.7, 0.0, 0.0]);
    let blob = translate(scale(cube(), [0.15, 0.15, 0.15]), [0.7, 0.0, 0.3]);
    let both = union(tall, blob);
    let mut config = base_config();
    config.push(("mode", text("island")));
    let wasm = run_island_removal(both.clone(), &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(occupied_at(&mut model, &[-0.7, 0.0, 0.9]), "the tall column");
    assert!(
        !occupied_at(&mut model, &[0.7, 0.0, 0.3]),
        "the detached blob must be removed despite the column in its layers"
    );

    // The same scene at overhang_angle 89.9 documents the leak island
    // mode exists to fix: lateral reach reads the column as support.
    let mut config = base_config();
    config.push(("overhang_angle", float(89.9)));
    let wasm = run_island_removal(both, &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(
        occupied_at(&mut model, &[0.7, 0.0, 0.3]),
        "overhang mode at near-90 degrees keeps the blob"
    );
}

/// The pass is dimension-generic: a 2D slice with two disks loses the
/// one floating above the seed disk (build axis defaults to the last
/// axis — y for a 2D model).
#[test]
fn two_dimensional_models_are_supported() {
    use volumetric::subspace::{Subspace, encode_subspace};
    // Slice each sphere to a disk first, then union the 2D models (the
    // slice operator needs a constant get_dimensions, which a merged
    // module doesn't provide).
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1]).unwrap();
    let disk = |model: Vec<u8>| {
        run_operator("slice_operator", vec![model, encode_subspace(&plane)]).expect("slice")
    };
    let lifted = translate(wasm_artifact("simple_sphere_model"), [0.0, 3.0, 0.0]);
    let disks = union(disk(wasm_artifact("simple_sphere_model")), disk(lifted));

    let wasm = run_island_removal(disks, &base_config()).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(
        occupied_at(&mut model, &[0.0, 0.0]),
        "the seed disk's center column"
    );
    assert!(
        !occupied_at(&mut model, &[0.0, 3.0]),
        "the floating disk must be ablated"
    );
}

/// A channeled input keeps its sample format through both outputs:
/// channel 0 is the gated occupancy, the density channel passes through
/// untouched.
#[test]
fn channels_pass_through() {
    // density_gradient_model: [-1,1]^3 cube, density = 0.5 + 0.5x,
    // channels [Occupancy, Density]; the boolean union keeps A's format.
    let input = cube_with_floating_sphere();

    let wasm =
        run_island_removal(input.clone(), &base_config()).expect("island removal failed");
    let mut model = NativeModelExecutor::new(&wasm).expect("model executor");
    assert_eq!(model.sample_format().channels.len(), 2);

    // Inside the kept cube: occupancy plus the input's density gradient.
    let row = model.sample_channels_nd(&[0.5, 0.0, -0.5]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.75).abs() < 1e-6, "density was {}", row[1]);

    // In the ablated island: channel 0 reports the cut.
    let row = model.sample_channels_nd(&[0.0, 0.0, 3.0]).unwrap();
    assert_eq!(row[0], 0.0, "ablated island must be cut in channel 0");
    assert_eq!(row[0], model.sample_nd(&[0.0, 0.0, 3.0]).unwrap());

    // The islands output gates channel 0 the other way around.
    let input = cube_with_floating_sphere();
    let mut config = base_config();
    config.push(("output", text("islands")));
    let wasm = run_island_removal(input, &config).expect("island removal failed");
    let mut model = NativeModelExecutor::new(&wasm).expect("model executor");
    assert_eq!(model.sample_format().channels.len(), 2);
    let row = model.sample_channels_nd(&[0.0, 0.0, 3.0]).unwrap();
    assert_eq!(row[0], 1.0, "the island is the geometry here");
    let row = model.sample_channels_nd(&[0.5, 0.0, -0.5]).unwrap();
    assert_eq!(row[0], 0.0, "kept geometry is cut in the islands output");
    assert!((row[1] - 0.75).abs() < 1e-6, "density still passes through");
}

/// High-resolution smoke run — far past the old dense-storage cap.
/// Discovery samples the geometry's surface, not the volume, and the
/// walk streams layers (run explicitly via
/// `cargo test --test island_removal -- --ignored`).
#[test]
#[ignore = "long-running; run explicitly"]
fn high_resolution_streaming_smoke() {
    let input = cube_with_floating_sphere();
    let config = vec![("resolution", int(1024))];
    let wasm = run_island_removal(input, &config).expect("island removal failed");
    let mut model = create_model_executor(&wasm).expect("model executor");
    assert!(occupied_at(&mut model, &[0.0, 0.0, 0.0]), "the cube");
    assert!(
        !occupied_at(&mut model, &[0.0, 0.0, 3.0]),
        "floating sphere ablated at high resolution"
    );
}

/// Config validation surfaces as an operator error.
#[test]
fn invalid_angle_reports_an_error() {
    let mut config = base_config();
    config.push(("overhang_angle", float(90.0)));
    let err = run_island_removal(wasm_artifact("density_gradient_model"), &config)
        .expect_err("a 90-degree overhang allowance must fail");
    assert!(err.contains("overhang_angle"), "unexpected error: {err}");
}
