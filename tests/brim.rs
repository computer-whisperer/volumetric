//! Integration tests for brim_operator: a print-bed adhesion brim grown
//! from the model's first-layer footprint.
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

fn run_brim_on(model: &str, config: &[(&str, ciborium::value::Value)]) -> Result<Vec<u8>, String> {
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

    let operator_wasm = wasm_artifact("brim_operator");
    let mut executor = create_operator_executor(&operator_wasm).expect("operator executor");
    let result = executor
        .run(OperatorIo::new(vec![wasm_artifact(model), cfg]))
        .map_err(|e| e.to_string())?;
    result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "operator posted no output".to_string())
}

fn run_brim(config: &[(&str, ciborium::value::Value)]) -> Result<Vec<u8>, String> {
    run_brim_on("simple_sphere_model", config)
}

fn base_config() -> Vec<(&'static str, ciborium::value::Value)> {
    vec![
        ("brim_width", ciborium::value::Value::Float(0.5)),
        ("brim_height", ciborium::value::Value::Float(0.2)),
        ("resolution", ciborium::value::Value::Integer(128.into())),
    ]
}

fn occupied_at(executor: &mut impl ModelExecutor, p: [f64; 3]) -> bool {
    volumetric::is_occupied(executor.sample_nd(&p).expect("sample"))
}

/// The unit sphere on the auto-detected bed (z = -1): the brim is an
/// annulus around the first-layer footprint, the part itself is intact,
/// and bounds grow by the brim margin in x/y only.
#[test]
fn sphere_grows_a_brim_ring() {
    // Footprint scan happens at z = bed + brim_height/2 = -0.9, where the
    // sphere's cross-section radius is sqrt(1 - 0.9^2) ~ 0.436; the brim
    // reaches 0.5 beyond that.
    let wasm = run_brim(&base_config()).expect("brim run failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    // The brim (reach ~0.936 plus a cell of crop border) stays inside the
    // sphere's x/y bounds, so the union is exactly the sphere's box.
    let bounds = model.get_bounds_nd().expect("bounds");
    assert_eq!(bounds.min(0), -1.0);
    assert_eq!(bounds.max(0), 1.0);
    assert_eq!(bounds.min(2), -1.0, "z min stays the sphere's");
    assert_eq!(bounds.max(2), 1.0, "z max stays the sphere's");

    // The part itself.
    assert!(occupied_at(&mut model, [0.0, 0.0, 0.0]));
    // Brim ring inside the slab, outside the sphere.
    assert!(occupied_at(&mut model, [0.8, 0.0, -0.9]));
    assert!(occupied_at(&mut model, [-0.8, 0.0, -0.9]));
    assert!(occupied_at(&mut model, [0.0, 0.8, -0.9]));
    // Beyond the brim reach (~0.936).
    assert!(!occupied_at(&mut model, [1.05, 0.0, -0.9]));
    // Same x/y as the ring but above the brim slab (and outside the sphere).
    assert!(!occupied_at(&mut model, [0.8, 0.0, -0.7]));
}

/// `output: "brim"` emits the bare brim solid: slab-bounded in z, present
/// under the footprint (gap 0), absent outside the slab.
#[test]
fn brim_only_output_is_the_bare_slab() {
    let mut config = base_config();
    config.push(("output", ciborium::value::Value::Text("brim".into())));
    let wasm = run_brim(&config).expect("brim run failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    let bounds = model.get_bounds_nd().expect("bounds");
    assert_eq!(bounds.min(2), -1.0);
    assert!((bounds.max(2) - -0.8).abs() < 1e-12);
    // Cropped to the brim reach (~0.936 plus a cell or two), not the
    // scan grid's full margin rectangle.
    assert!(
        bounds.min(0) > -1.0 && bounds.min(0) < -0.9,
        "x min {} should hug the brim contour",
        bounds.min(0)
    );

    assert!(occupied_at(&mut model, [0.0, 0.0, -0.9]), "under the part");
    assert!(occupied_at(&mut model, [0.8, 0.0, -0.9]), "the ring");
    assert!(!occupied_at(&mut model, [0.0, 0.0, 0.0]), "above the slab");
    assert!(
        !occupied_at(&mut model, [1.05, 0.0, -0.9]),
        "past the reach"
    );
}

/// A positive gap detaches the brim from the footprint, skirt-style.
#[test]
fn gap_leaves_a_detached_ring() {
    let mut config = base_config();
    config.push(("output", ciborium::value::Value::Text("brim".into())));
    config.push(("gap", ciborium::value::Value::Float(0.2)));
    let wasm = run_brim(&config).expect("brim run failed");
    let mut model = create_model_executor(&wasm).expect("model executor");

    // Footprint radius ~0.436: distances 0, ~0.064, ~0.264 from it.
    assert!(
        !occupied_at(&mut model, [0.0, 0.0, -0.9]),
        "gap under the part"
    );
    assert!(!occupied_at(&mut model, [0.5, 0.0, -0.9]), "inside the gap");
    assert!(
        occupied_at(&mut model, [0.7, 0.0, -0.9]),
        "the detached ring"
    );
}

/// A channeled input keeps its sample format through the combined
/// output: channel 0 is the union occupancy (1.0 in the brim ring), the
/// density channel passes through untouched. The bare brim solid stays
/// occupancy-only.
#[test]
fn channels_pass_through_the_combined_output() {
    // density_gradient_model: [-1,1]^3 cube, density = 0.5 + 0.5x,
    // channels [Occupancy, Density]. Its footprint at the scan height is
    // the full square, so the brim ring sits beyond |x| or |y| = 1.
    let wasm = run_brim_on("density_gradient_model", &base_config()).expect("brim run failed");
    let mut model = NativeModelExecutor::new(&wasm).expect("model executor");

    let format = model.sample_format().clone();
    assert_eq!(format.channels.len(), 2, "input format passes through");

    // Inside the part: occupancy plus the input's density gradient.
    let row = model.sample_channels_nd(&[0.5, 0.0, -0.9]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.75).abs() < 1e-6, "density was {}", row[1]);

    // In the brim ring: channel 0 reports the union occupancy.
    let row = model.sample_channels_nd(&[1.2, 0.0, -0.9]).unwrap();
    assert_eq!(row[0], 1.0, "brim ring must show in channel 0");
    assert_eq!(row[0], model.sample_nd(&[1.2, 0.0, -0.9]).unwrap());

    // Above the slab, outside the part: unoccupied.
    let row = model.sample_channels_nd(&[1.2, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 0.0);

    // The bare brim solid has no channels to preserve.
    let mut config = base_config();
    config.push(("output", ciborium::value::Value::Text("brim".into())));
    let bare = run_brim_on("density_gradient_model", &config).expect("brim run failed");
    let bare_model = NativeModelExecutor::new(&bare).expect("model executor");
    assert_eq!(bare_model.sample_format().channels.len(), 1);
}

/// A bed plane that misses the part is an error, not an empty brim.
#[test]
fn missing_footprint_reports_an_error() {
    let mut config = base_config();
    config.push(("bed_z", ciborium::value::Value::Float(-3.0)));
    let err = run_brim(&config).expect_err("scan below the part must fail");
    assert!(err.contains("no part geometry"), "unexpected error: {err}");
}
