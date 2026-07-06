//! Integration tests for lattice_operator: a density model filled with a
//! density-modulated implicit lattice.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::wasm::{
    NativeModelExecutor, OperatorExecutor, OperatorIo, create_operator_executor,
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

fn lattice_config(lattice: &str, cell_size: f64) -> Vec<u8> {
    cbor_config(&[
        ("lattice", ciborium::value::Value::Text(lattice.into())),
        ("cell_size", ciborium::value::Value::Float(cell_size)),
    ])
}

/// Fraction of a y-z probe grid at fixed x that the model reports occupied.
fn occupied_fraction(executor: &mut NativeModelExecutor, x: f64) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for j in 0..12 {
        for k in 0..12 {
            let y = -0.92 + 1.84 * (j as f64 / 11.0);
            let z = -0.92 + 1.84 * (k as f64 / 11.0);
            total += 1;
            if volumetric::is_occupied(executor.sample_nd(&[x, y, z]).unwrap()) {
                hits += 1;
            }
        }
    }
    hits as f64 / total as f64
}

/// The density_gradient_model cube ([-1,1]^3, density = 0.5 + 0.5x) filled
/// with a gyroid: occupancy follows the density gradient, bounds and the
/// channel layout pass through.
#[test]
fn gyroid_fill_follows_the_density_gradient() {
    let output = run_operator(
        "lattice_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            lattice_config("gyroid", 0.4),
        ],
    )
    .expect("lattice operator runs");

    let mut executor = NativeModelExecutor::new(&output).expect("output executes");
    assert_eq!(executor.dimensions(), 3);

    // Bounds pass through from the input cube.
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!(bounds.min(0), -1.0);
    assert_eq!(bounds.max(2), 1.0);

    // Channel layout passes through.
    let channels: Vec<String> = executor
        .sample_format()
        .channels
        .iter()
        .map(|c| c.name.clone())
        .collect();
    assert_eq!(channels.len(), 2, "occupancy + density: {channels:?}");

    // Occupancy tracks the density gradient: nearly solid at x -> 1
    // (density -> 1), sparse at x -> -1 (density -> 0), monotone-ish in
    // between; nothing outside the cube.
    let mut executor = NativeModelExecutor::new(&output).unwrap();
    let low = occupied_fraction(&mut executor, -0.7); // density 0.15
    let mid = occupied_fraction(&mut executor, 0.0); // density 0.5
    let high = occupied_fraction(&mut executor, 0.96); // density 0.98
    assert!(low < mid && mid < high, "gradient: {low} < {mid} < {high}");
    assert!(low < 0.6, "low-density end too full: {low}");
    assert!(high > 0.9, "high-density end too empty: {high}");
    // density(x = 1) = 1.0 exactly -> solid.
    assert!(volumetric::is_occupied(
        executor.sample_nd(&[0.999999, 0.31, -0.47]).unwrap()
    ));
    // Outside the input cube stays empty regardless of the lattice.
    assert!(!volumetric::is_occupied(
        executor.sample_nd(&[1.5, 0.0, 0.0]).unwrap()
    ));

    // sample_channels: channel 0 is the lattice occupancy, channel 1 the
    // input's density, at the sampled position.
    let row = executor.sample_channels_nd(&[0.5, 0.1, 0.1]).unwrap();
    assert_eq!(row.len(), 2);
    assert!((row[1] - 0.75).abs() < 1e-6, "density passthrough: {row:?}");
    let occ = executor.sample_nd(&[0.5, 0.1, 0.1]).unwrap();
    assert_eq!(
        volumetric::is_occupied(row[0]),
        volumetric::is_occupied(occ),
        "sample and sample_channels agree"
    );
}

/// All three lattice families produce valid, distinct structures.
#[test]
fn lattice_families_run_and_differ() {
    let input = wasm_artifact("density_gradient_model");
    let mut signatures = Vec::new();
    for lattice in ["gyroid", "schwarz", "struts"] {
        let output = run_operator(
            "lattice_operator",
            vec![input.clone(), lattice_config(lattice, 0.5)],
        )
        .unwrap_or_else(|e| panic!("{lattice} failed: {e}"));
        let mut executor = NativeModelExecutor::new(&output).expect("output executes");
        // Occupancy signature over a probe grid near mid density. x = 0.1
        // keeps the plane off the strut lattice's cell planes (x = 0 lies
        // exactly on one, where struts legitimately cover everything).
        let mut signature = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                let y = -0.9 + 0.35 * i as f64;
                let z = -0.9 + 0.35 * j as f64;
                signature.push(volumetric::is_occupied(
                    executor.sample_nd(&[0.1, y, z]).unwrap(),
                ));
            }
        }
        assert!(
            signature.iter().any(|&b| b) && !signature.iter().all(|&b| b),
            "{lattice} at mid density should be partially occupied"
        );
        signatures.push(signature);
    }
    assert_ne!(signatures[0], signatures[1], "gyroid vs schwarz");
    assert_ne!(signatures[0], signatures[2], "gyroid vs struts");
}

/// Occupancy-only inputs (no channels) lattice-fill with `uniform_density`.
#[test]
fn occupancy_only_input_uses_uniform_density() {
    let sphere = wasm_artifact("simple_sphere_model");
    let output = run_operator(
        "lattice_operator",
        vec![
            sphere,
            cbor_config(&[
                ("lattice", ciborium::value::Value::Text("struts".into())),
                ("cell_size", ciborium::value::Value::Float(0.4)),
                ("uniform_density", ciborium::value::Value::Float(0.3)),
            ]),
        ],
    )
    .expect("lattice operator runs on a plain model");

    let mut executor = NativeModelExecutor::new(&output).expect("output executes");
    // The output declares no channels (input had none).
    assert_eq!(executor.sample_format().channels.len(), 1);

    // Inside the sphere: partially filled (strut pattern at density 0.3).
    let mut inside_hits = 0usize;
    let mut inside_total = 0usize;
    for i in 0..10 {
        for j in 0..10 {
            let x = -0.5 + i as f64 / 9.0;
            let y = -0.5 + j as f64 / 9.0;
            inside_total += 1;
            if volumetric::is_occupied(executor.sample_nd(&[x, y, 0.05]).unwrap()) {
                inside_hits += 1;
            }
        }
    }
    assert!(
        inside_hits > 0 && inside_hits < inside_total,
        "expected a partial fill, got {inside_hits}/{inside_total}"
    );
    // Outside the sphere stays empty.
    assert!(!volumetric::is_occupied(
        executor.sample_nd(&[2.0, 0.0, 0.0]).unwrap()
    ));
}

/// Bad inputs fail with readable errors instead of emitting broken models.
#[test]
fn rejects_non_3d_inputs_and_bad_config() {
    // A minimal 2D model (walrus needs binary wasm, so assemble it).
    let flat = wat::parse_str(
        r#"(module
            (memory (export "memory") 1)
            (func (export "get_dimensions") (result i32) (i32.const 2))
            (func (export "get_io_ptr") (result i32) (i32.const 1024))
            (func (export "get_bounds") (param i32))
            (func (export "sample") (param i32) (result f32) (f32.const 1)))"#,
    )
    .unwrap();
    let err = run_operator(
        "lattice_operator",
        vec![flat, lattice_config("gyroid", 0.25)],
    )
    .expect_err("2D input should be rejected");
    assert!(err.contains("3D"), "{err}");

    let err = run_operator(
        "lattice_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            lattice_config("gyroid", 0.0),
        ],
    )
    .expect_err("zero cell size should be rejected");
    assert!(err.contains("cell_size"), "{err}");
}
