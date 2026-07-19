//! End-to-end coverage for local subspace fitting. A box model is generated
//! in one project step and routed into the fit operator in the next, along
//! with an independently routed seed value.
//!
//! Requires:
//!   cargo build --release --target wasm32-unknown-unknown \
//!     -p rectangular_prism_operator -p subspace_fit_operator

#![cfg(feature = "native")]

use volumetric::subspace::decode_subspace;
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset,
    OperatorMetadataInput, OperatorMetadataOutput, Project,
};

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing wasm artifact {} ({error}); build it with the command in this test's module docs",
            path.display()
        )
    })
}

fn vec3_bytes(values: [f64; 3]) -> Vec<u8> {
    values.into_iter().flat_map(f64::to_le_bytes).collect()
}

fn fit_config(kind: &str, snap_divisions: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![
            (
                ciborium::value::Value::Text("kind".to_string()),
                ciborium::value::Value::Text(kind.to_string()),
            ),
            (
                ciborium::value::Value::Text("tolerance".to_string()),
                ciborium::value::Value::Float(1e-5),
            ),
            (
                ciborium::value::Value::Text("search_radius".to_string()),
                ciborium::value::Value::Float(0.2),
            ),
            (
                ciborium::value::Value::Text("max_iterations".to_string()),
                ciborium::value::Value::Integer(12.into()),
            ),
            (
                ciborium::value::Value::Text("snap_divisions".to_string()),
                ciborium::value::Value::Integer(snap_divisions.into()),
            ),
        ]),
        &mut bytes,
    )
    .unwrap();
    bytes
}

#[test]
fn metadata_declares_model_seed_config_and_subspace_output() {
    let metadata =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("subspace_fit_operator"))
            .expect("fit metadata");
    assert_eq!(metadata.name, "subspace_fit_operator");
    assert_eq!(metadata.inputs.len(), 3);
    assert!(matches!(
        metadata.inputs[0],
        OperatorMetadataInput::ModelWASM
    ));
    assert_eq!(metadata.inputs[1], OperatorMetadataInput::VecF64(3));
    assert!(matches!(
        metadata.inputs[2],
        OperatorMetadataInput::CBORConfiguration(_)
    ));
    assert_eq!(metadata.input_name(0), Some("Model"));
    assert_eq!(metadata.input_name(1), Some("Seed"));
    assert_eq!(metadata.input_name(2), Some("Config"));
    assert_eq!(metadata.outputs, vec![OperatorMetadataOutput::Subspace]);
}

#[test]
fn generated_model_and_seed_route_through_project_dag() {
    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::operator(
                "box_generator".to_string(),
                wasm_artifact("rectangular_prism_operator"),
            ),
            ImportedAsset::operator("fit".to_string(), wasm_artifact("subspace_fit_operator")),
            ImportedAsset::new(
                "minimum".to_string(),
                vec3_bytes([-1.0; 3]),
                Some(AssetTypeHint::VecF64(3)),
            ),
            ImportedAsset::new(
                "maximum".to_string(),
                vec3_bytes([1.0; 3]),
                Some(AssetTypeHint::VecF64(3)),
            ),
            ImportedAsset::new(
                "corner_seed".to_string(),
                vec3_bytes([1.002, 1.001, 1.003]),
                Some(AssetTypeHint::VecF64(3)),
            ),
        ],
        timeline: vec![
            ExecutionStep {
                operator_id: "box_generator".to_string(),
                inputs: vec![
                    ExecutionInput::Inline(Vec::new()),
                    ExecutionInput::AssetRef("minimum".to_string()),
                    ExecutionInput::AssetRef("maximum".to_string()),
                ],
                outputs: vec!["box_model".to_string()],
            },
            ExecutionStep {
                operator_id: "fit".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("box_model".to_string()),
                    ExecutionInput::AssetRef("corner_seed".to_string()),
                    ExecutionInput::Inline(fit_config("point", 2)),
                ],
                outputs: vec!["fitted_corner".to_string()],
            },
        ],
        exports: vec!["fitted_corner".to_string()],
        baked: None,
    };
    assert!(project.validate().is_empty());

    let mut environment = Environment::new();
    let exports = project.run(&mut environment).expect("project run");
    let feature = exports
        .iter()
        .find(|asset| asset.id() == "fitted_corner")
        .expect("fitted subspace export");
    assert_eq!(feature.type_hint(), Some(AssetTypeHint::Subspace));
    assert_eq!(feature.precursor_ids(), &["box_model", "corner_seed"]);

    let subspace = decode_subspace(feature.data()).expect("subspace output");
    assert_eq!(subspace.rank(), 0);
    assert_eq!(subspace.origin, vec![1.0, 1.0, 1.0]);
}
