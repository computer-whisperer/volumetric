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

/// All lattice families produce valid, pairwise-distinct structures.
#[test]
fn lattice_families_run_and_differ() {
    let input = wasm_artifact("density_gradient_model");
    const FAMILIES: [&str; 6] = ["gyroid", "schwarz", "struts", "honeycomb", "tetra", "foam"];
    let mut signatures = Vec::new();
    for lattice in FAMILIES {
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
    for a in 0..signatures.len() {
        for b in (a + 1)..signatures.len() {
            assert_ne!(
                signatures[a], signatures[b],
                "{} vs {}",
                FAMILIES[a], FAMILIES[b]
            );
        }
    }
}

/// The density calibration knobs reshape how the density map turns into
/// structure: a floor thickens the sparse end, a cap hollows the dense end,
/// and gamma bends the mid-range.
#[test]
fn density_calibration_reshapes_the_fill() {
    let input = wasm_artifact("density_gradient_model");
    let fraction_at = |extra: &[(&str, ciborium::value::Value)], x: f64| -> f64 {
        let mut entries = vec![
            ("lattice", ciborium::value::Value::Text("gyroid".into())),
            ("cell_size", ciborium::value::Value::Float(0.4)),
        ];
        entries.extend_from_slice(extra);
        let output = run_operator(
            "lattice_operator",
            vec![input.clone(), cbor_config(&entries)],
        )
        .expect("lattice operator runs");
        let mut executor = NativeModelExecutor::new(&output).expect("output executes");
        occupied_fraction(&mut executor, x)
    };

    // density_min floors the sparse end: x = -0.7 samples density 0.15,
    // which a 0.85 floor lifts to ~0.87.
    let default_low = fraction_at(&[], -0.7);
    let floored_low = fraction_at(
        &[("density_min", ciborium::value::Value::Float(0.85))],
        -0.7,
    );
    assert!(
        floored_low > default_low + 0.2,
        "floor should thicken the sparse end: {default_low} -> {floored_low}"
    );

    // density_max caps the dense end: x = 0.96 samples density 0.98.
    let default_high = fraction_at(&[], 0.96);
    let capped_high = fraction_at(&[("density_max", ciborium::value::Value::Float(0.3))], 0.96);
    assert!(
        capped_high < default_high - 0.2,
        "cap should hollow the dense end: {default_high} -> {capped_high}"
    );

    // gamma > 1 thins the mid-range without touching the endpoints.
    let default_mid = fraction_at(&[], 0.0);
    let curved_mid = fraction_at(
        &[("density_gamma", ciborium::value::Value::Float(3.0))],
        0.0,
    );
    assert!(
        curved_mid < default_mid - 0.1,
        "gamma 3 should thin the mid-range: {default_mid} -> {curved_mid}"
    );
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

    let err = run_operator(
        "lattice_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            cbor_config(&[("density_gamma", ciborium::value::Value::Float(0.0))]),
        ],
    )
    .expect_err("non-positive gamma should be rejected");
    assert!(err.contains("density_gamma"), "{err}");

    let err = run_operator(
        "lattice_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            cbor_config(&[
                ("density_min", ciborium::value::Value::Float(0.8)),
                ("density_max", ciborium::value::Value::Float(0.2)),
            ]),
        ],
    )
    .expect_err("inverted density range should be rejected");
    assert!(err.contains("density_min"), "{err}");

    let err = run_operator(
        "lattice_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            cbor_config(&[("irregularity", ciborium::value::Value::Float(1.5))]),
        ],
    )
    .expect_err("out-of-range irregularity should be rejected");
    assert!(err.contains("irregularity"), "{err}");
}

/// The foam's irregularity knob reshapes the cells through the operator.
#[test]
fn foam_irregularity_flows_through() {
    let input = wasm_artifact("density_gradient_model");
    let signature = |irregularity: f64| -> Vec<bool> {
        let output = run_operator(
            "lattice_operator",
            vec![
                input.clone(),
                cbor_config(&[
                    ("lattice", ciborium::value::Value::Text("foam".into())),
                    ("cell_size", ciborium::value::Value::Float(0.5)),
                    ("irregularity", ciborium::value::Value::Float(irregularity)),
                ]),
            ],
        )
        .expect("foam runs");
        let mut executor = NativeModelExecutor::new(&output).expect("output executes");
        // Incommensurate probe steps: a grid aligned to the cell period
        // can alias entirely into the pores of the regular Kelvin foam.
        let mut signature = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let x = -0.81 + 0.487 * i as f64;
                    let y = -0.79 + 0.463 * j as f64;
                    let z = -0.83 + 0.521 * k as f64;
                    signature.push(volumetric::is_occupied(
                        executor.sample_nd(&[x, y, z]).unwrap(),
                    ));
                }
            }
        }
        signature
    };
    let regular = signature(0.0);
    let organic = signature(0.8);
    assert!(regular.iter().any(|&b| b) && !regular.iter().all(|&b| b));
    assert_ne!(regular, organic, "irregularity should reshape the foam");
}
