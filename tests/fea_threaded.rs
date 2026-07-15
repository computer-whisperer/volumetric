//! End-to-end tests of the threaded fea operator variants: the packed
//! fea_solve blob (baseline + embedded wasm32-wasip1-threads build) runs
//! its threaded half on the native executor, so these exercise the guest
//! thread pool, the Schwarz preconditioner behind the operator config, and
//! cooperative cancellation.
//!
//! Requires packed wasm artifacts: `cargo wasm-dist`.

#![cfg(feature = "native")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use volumetric::{
    AssetTypeHint, BuildCache, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::fea::decode_fea_mesh;

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing wasm artifact {} ({e}); build it with `cargo wasm-dist`",
            path.display()
        )
    })
}

fn cbor_map(entries: &[(&str, ciborium::value::Value)]) -> Vec<u8> {
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

fn vec3(v: [f64; 3]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A unit box filled with a cubic strut lattice, compressed by a sphere
/// dipping 0.1 into its top, solved with the given solver config.
fn lattice_press_project(cell_size: f64, solver_config: Vec<u8>) -> Project {
    use ciborium::value::Value;
    Project {
        version: 2,
        imports: vec![
            ImportedAsset::model("sphere".to_string(), wasm_artifact("simple_sphere_model")),
            ImportedAsset::operator(
                "prism".to_string(),
                wasm_artifact("rectangular_prism_operator"),
            ),
            ImportedAsset::operator("translate".to_string(), wasm_artifact("translate_operator")),
            ImportedAsset::operator(
                "pattern".to_string(),
                wasm_artifact("strut_pattern_operator"),
            ),
            ImportedAsset::operator("solver".to_string(), wasm_artifact("fea_solve_operator")),
        ],
        timeline: vec![
            ExecutionStep {
                operator_id: "prism".to_string(),
                inputs: vec![
                    ExecutionInput::Inline(cbor_map(&[(
                        "mode",
                        Value::Text("opposite_corners".into()),
                    )])),
                    ExecutionInput::Inline(vec3([0.0, 0.0, 0.0])),
                    ExecutionInput::Inline(vec3([1.0, 1.0, 1.0])),
                ],
                outputs: vec!["box".to_string()],
            },
            ExecutionStep {
                operator_id: "translate".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("sphere".to_string()),
                    ExecutionInput::Inline(cbor_map(&[
                        ("dx", Value::Float(0.5)),
                        ("dy", Value::Float(0.5)),
                        ("dz", Value::Float(1.9)),
                    ])),
                ],
                outputs: vec!["butt".to_string()],
            },
            ExecutionStep {
                operator_id: "pattern".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("box".to_string()),
                    ExecutionInput::Inline(cbor_map(&[
                        ("family", Value::Text("cubic".into())),
                        ("cell_size", Value::Float(cell_size)),
                        ("radius", Value::Float(0.015)),
                    ])),
                ],
                outputs: vec!["lattice".to_string()],
            },
            ExecutionStep {
                operator_id: "solver".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("lattice".to_string()),
                    ExecutionInput::AssetRef("butt".to_string()),
                    ExecutionInput::Inline(solver_config),
                ],
                outputs: vec!["solved".to_string()],
            },
        ],
        exports: vec!["solved".to_string()],
        baked: None,
    }
}

fn solve_displacement(preconditioner: &str) -> Vec<f64> {
    use ciborium::value::Value;
    let project = lattice_press_project(
        0.125,
        cbor_map(&[("preconditioner", Value::Text(preconditioner.into()))]),
    );
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let asset = &exports[0];
    assert_eq!(asset.type_hint(), Some(AssetTypeHint::FeaMesh));
    let mesh = decode_fea_mesh(asset.data()).expect("solved output decodes");
    assert_eq!(mesh.element_kind, volumetric_abi::fea::FeaElementKind::Bar2);
    mesh.node_fields
        .iter()
        .find(|f| f.name == "displacement")
        .expect("displacement field")
        .data
        .clone()
}

/// The Schwarz preconditioner, reached purely through operator config on
/// the packed threaded blob, must converge to the same solution as the
/// default. This is the arc's payoff path: config → PrecondChoice::parse →
/// fea_core `parallel` → rayon pool on wasi threads.
#[test]
fn schwarz_config_matches_auto_solution_end_to_end() {
    let auto = solve_displacement("auto");
    let schwarz = solve_displacement("schwarz");
    assert_eq!(auto.len(), schwarz.len());

    let scale = auto.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(scale > 1e-3, "degenerate solve: max displacement {scale}");
    let max_diff = auto
        .iter()
        .zip(&schwarz)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_diff < 1e-5 * scale.max(1.0),
        "preconditioners disagree: max diff {max_diff}, scale {scale}"
    );
}

/// Manual fixture writer for remote-daemon verification: writes the fine
/// lattice-press project (packed operators) as a .vproj.
///
///   VPROJ_OUT=/tmp/lattice_schwarz.vproj [VPROJ_PRECOND=auto] cargo test \
///     --features native --test fea_threaded write_lattice_press_vproj -- --ignored
#[test]
#[ignore]
fn write_lattice_press_vproj() {
    use ciborium::value::Value;
    let out = std::env::var("VPROJ_OUT").expect("set VPROJ_OUT=<path>");
    let precond = std::env::var("VPROJ_PRECOND").unwrap_or_else(|_| "schwarz".into());
    let project = lattice_press_project(
        0.04,
        cbor_map(&[("preconditioner", Value::Text(precond))]),
    );
    project
        .save_to_file(std::path::Path::new(&out))
        .expect("write vproj");
    println!("wrote {out}");
}

/// Manual timing probe (`cargo test --features native --test fea_threaded
/// -- --ignored --nocapture`): the same fine lattice solved with each
/// preconditioner. Set VOLUMETRIC_DISABLE_THREADED_OPERATORS=1 /
/// VOLUMETRIC_THREADS to compare configurations.
#[test]
#[ignore]
fn bench_preconditioners_on_fine_lattice() {
    use ciborium::value::Value;
    for preconditioner in ["auto", "schwarz"] {
        let project = lattice_press_project(
            0.04,
            cbor_map(&[("preconditioner", Value::Text(preconditioner.into()))]),
        );
        let started = std::time::Instant::now();
        let result = project.run(&mut Environment::new());
        println!(
            "{preconditioner}: {:?} ({})",
            started.elapsed(),
            if result.is_ok() { "ok" } else { "failed" },
        );
    }
}

/// Cooperative cancellation reaches a threaded solve: the guest polls
/// `host.cancelled` between CG iterations, unwinds, drops its pool, and
/// the run reports promptly instead of finishing the solve.
#[test]
fn threaded_solve_cancels_mid_run() {
    use ciborium::value::Value;
    // Fine lattice + high contact-iteration cap: a solve that takes long
    // enough that a 500ms cancel is unambiguously mid-run.
    let project = lattice_press_project(
        0.04,
        cbor_map(&[("max_contact_iterations", Value::Integer(256.into()))]),
    );

    // Armed from the progress callback so the cancel lands mid-*solve*
    // (the earlier pipeline steps are plain operators; cancelling those
    // wouldn't exercise the guest's cooperative poll).
    let cancel = Arc::new(AtomicBool::new(false));
    let solver_started = Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));

    let cache = BuildCache::new(0); // disabled: this run must execute
    let progress_cancel = Arc::clone(&cancel);
    let progress_started = Arc::clone(&solver_started);
    let result = project.run_monitored_with_cache(
        &mut Environment::new(),
        &cache,
        &cancel,
        &move |progress| {
            if progress.phase.starts_with("solver") {
                progress_started
                    .lock()
                    .unwrap()
                    .get_or_insert_with(std::time::Instant::now);
                let flag = Arc::clone(&progress_cancel);
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(300));
                    flag.store(true, Ordering::Relaxed);
                });
            }
        },
    );
    assert!(result.is_err(), "cancelled run must not succeed");
    let started = solver_started
        .lock()
        .unwrap()
        .expect("the solver step must have started");
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(15),
        "cancellation took {elapsed:?} after solver start — cooperative \
         poll not reached?"
    );
}
