//! End-to-end test of step_import_operator running as WASM under the
//! project engine: STEP blob in, sampleable exact-BRep model out.
//!
//! Requires the wasm32 artifact:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p step_import_operator

#![cfg(feature = "native")]

use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::is_occupied;

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

const FIXTURE: &[u8] =
    include_bytes!("../crates/operators/step_import_operator/tests/fixtures/box_cylinder.step");

#[test]
fn step_fixture_imports_to_a_sampleable_solid() {
    let project = Project {
        version: 2,
        imports: vec![ImportedAsset::operator(
            "importer".to_string(),
            wasm_artifact("step_import_operator"),
        )],
        timeline: vec![ExecutionStep {
            operator_id: "importer".to_string(),
            inputs: vec![
                ExecutionInput::Inline(FIXTURE.to_vec()),
                ExecutionInput::Inline(Vec::new()),
            ],
            outputs: vec!["solid".to_string()],
        }],
        exports: vec!["solid".to_string()],
        baked: None,
    };

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let solid = exports.iter().find(|e| e.id() == "solid").unwrap();
    assert_eq!(solid.type_hint(), Some(AssetTypeHint::Model));

    let mut executor = create_model_executor(solid.data()).expect("model instantiates");
    assert_eq!(executor.dimensions().unwrap(), 3);
    let bounds = executor.get_bounds_nd().unwrap();
    let bounds = bounds.as_slice();
    // Box [-5,5]x[-4,4]x[-3,3] plus cylinder r=3 at x=20, z [-3,9], in the
    // fixture's millimetres; the imported model is in metres. Bounds are
    // conservative but close.
    const M_PER_MM: f64 = 1e-3;
    assert!(
        bounds[0] < -4.99 * M_PER_MM && bounds[0] > -5.1 * M_PER_MM,
        "min_x {}",
        bounds[0]
    );
    assert!(
        bounds[1] > 22.99 * M_PER_MM && bounds[1] < 23.2 * M_PER_MM,
        "max_x {}",
        bounds[1]
    );
    assert!(
        bounds[5] > 8.99 * M_PER_MM && bounds[5] < 9.2 * M_PER_MM,
        "max_z {}",
        bounds[5]
    );

    // The fixture geometry: box at origin, cylinder wall at x = 20.
    for (p, inside) in [
        ([0.0, 0.0, 0.0], true),
        ([4.9, -3.9, 2.9], true),
        ([20.0, 0.0, 5.0], true),
        ([22.5, 0.0, 8.5], true),
        ([5.2, 0.0, 0.0], false),
        ([20.0, 0.0, 9.5], false),
        ([12.0, 0.0, 0.0], false),
        ([20.0, 3.2, 0.0], false),
    ] {
        let p_m = p.map(|v| v * M_PER_MM);
        let sample = executor.sample_nd(&p_m).unwrap();
        assert_eq!(
            is_occupied(sample),
            inside,
            "sample at {p:?} = {sample}, expected inside={inside}"
        );
    }
}

#[test]
fn bad_step_reports_a_clear_error() {
    let project = Project {
        version: 2,
        imports: vec![ImportedAsset::operator(
            "importer".to_string(),
            wasm_artifact("step_import_operator"),
        )],
        timeline: vec![ExecutionStep {
            operator_id: "importer".to_string(),
            inputs: vec![
                ExecutionInput::Inline(b"not a step file".to_vec()),
                ExecutionInput::Inline(Vec::new()),
            ],
            outputs: vec!["solid".to_string()],
        }],
        exports: vec!["solid".to_string()],
        baked: None,
    };
    let mut env = Environment::new();
    let err = project.run(&mut env).expect_err("must fail");
    assert!(
        err.to_string().contains("ISO-10303-21"),
        "unexpected error: {err}"
    );
}
