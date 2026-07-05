//! End-to-end test of the FEA grid-mesh operator: runs the real
//! fea_grid_mesh_operator.wasm against the bundled sphere model through a
//! full project execution, exercising the host's `input_model_*` sampling
//! imports and the FeaMesh output type stamping.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p simple_sphere_model -p fea_grid_mesh_operator

#![cfg(feature = "native")]

use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::fea::{FeaMesh, decode_fea_mesh};

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

fn config(resolution: i64) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text("resolution".into()),
            ciborium::value::Value::Integer(resolution.into()),
        )]),
        &mut out,
    )
    .unwrap();
    out
}

/// Run sphere -> grid mesher and return the decoded mesh (decode validates).
fn mesh_sphere(config_bytes: Vec<u8>) -> (FeaMesh, AssetTypeHint) {
    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::model("sphere".to_string(), wasm_artifact("simple_sphere_model")),
            ImportedAsset::operator(
                "mesher".to_string(),
                wasm_artifact("fea_grid_mesh_operator"),
            ),
        ],
        timeline: vec![ExecutionStep {
            operator_id: "mesher".to_string(),
            inputs: vec![
                ExecutionInput::AssetRef("sphere".to_string()),
                ExecutionInput::Inline(config_bytes),
            ],
            outputs: vec!["mesh".to_string()],
        }],
        exports: vec!["mesh".to_string()],
    };

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let asset = &exports[0];
    let mesh = decode_fea_mesh(asset.data()).expect("output is not a valid FeaMesh");
    (mesh, asset.type_hint().expect("output has no type hint"))
}

#[test]
fn sphere_grid_mesh_is_valid_and_typed() {
    let resolution = 12usize;
    let (mesh, hint) = mesh_sphere(config(resolution as i64));

    assert_eq!(hint, AssetTypeHint::FeaMesh);
    assert!(
        mesh.element_count() > 100,
        "suspiciously few elements: {}",
        mesh.element_count()
    );

    // A sphere fills ~pi/6 of its bounding box; cell-center sampling on a
    // coarse grid stays well inside these brackets.
    let total_cells = resolution.pow(3);
    let fill = mesh.element_count() as f64 / total_cells as f64;
    assert!(
        (0.25..0.85).contains(&fill),
        "unexpected fill fraction {fill:.3} ({} of {total_cells} cells)",
        mesh.element_count()
    );

    // Every element has 8 distinct nodes.
    for e in 0..mesh.element_count() {
        let element = mesh.element(e);
        let mut sorted: Vec<u32> = element.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 8, "element {e} repeats nodes: {element:?}");
    }

    // The mesh must stay inside the model's declared bounds (nodes sit on
    // the cell lattice, so at most one cell of slack from ceil-ing the
    // per-axis cell counts) and be roughly symmetric like the sphere.
    let sphere_bytes = wasm_artifact("simple_sphere_model");
    let mut executor = create_model_executor(&sphere_bytes).expect("model executor");
    let bounds = executor.get_bounds_nd().expect("model bounds");
    let longest = (0..3)
        .map(|a| bounds.max(a) - bounds.min(a))
        .fold(0.0f64, f64::max);
    let cell = longest / resolution as f64;

    for axis in 0..3 {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for n in 0..mesh.node_count() {
            let p = mesh.node_position(n)[axis];
            lo = lo.min(p);
            hi = hi.max(p);
        }
        assert!(
            lo >= bounds.min(axis) - 1e-9 && hi <= bounds.max(axis) + cell + 1e-9,
            "axis {axis}: mesh [{lo}, {hi}] outside model [{}, {}]",
            bounds.min(axis),
            bounds.max(axis)
        );
        let center = bounds.min(axis) + (bounds.max(axis) - bounds.min(axis)) / 2.0;
        assert!(
            ((lo + hi) / 2.0 - center).abs() <= cell,
            "axis {axis}: mesh center {} far from model center {center}",
            (lo + hi) / 2.0
        );
    }

    // Boundary extraction: a solid blob has a closed boundary, well below
    // the all-faces count.
    let quads = mesh.boundary_quads();
    assert!(!quads.is_empty());
    assert!(quads.len() < 6 * mesh.element_count());
}

#[test]
fn empty_config_defaults() {
    let (mesh, hint) = mesh_sphere(Vec::new());
    assert_eq!(hint, AssetTypeHint::FeaMesh);
    assert!(mesh.element_count() > 100);
}

#[test]
fn non_model_input_reports_an_error() {
    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::new(
                "junk".to_string(),
                b"not a wasm module".to_vec(),
                Some(AssetTypeHint::Binary),
            ),
            ImportedAsset::operator(
                "mesher".to_string(),
                wasm_artifact("fea_grid_mesh_operator"),
            ),
        ],
        timeline: vec![ExecutionStep {
            operator_id: "mesher".to_string(),
            inputs: vec![
                ExecutionInput::AssetRef("junk".to_string()),
                ExecutionInput::Inline(Vec::new()),
            ],
            outputs: vec!["mesh".to_string()],
        }],
        exports: vec!["mesh".to_string()],
    };

    let mut env = Environment::new();
    let err = project.run(&mut env).expect_err("junk input must fail");
    let message = err.to_string();
    assert!(
        message.contains("not a usable model"),
        "unexpected error: {message}"
    );
}
