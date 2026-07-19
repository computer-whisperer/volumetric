//! End-to-end tests for `sdf_operator`: the source occupancy and geometry
//! bounds remain exact while a globally defined truncated signed-distance
//! custom channel is attached.
//!
//! Requires:
//!   cargo build --release --target wasm32-unknown-unknown \
//!     -p simple_sphere_model -p sdf_operator

#![cfg(feature = "native")]

use volumetric::wasm::{
    ModelExecutor, OperatorExecutor, OperatorIo, create_model_executor, create_operator_executor,
};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset,
    OperatorMetadataInput, OperatorMetadataOutput, Project,
};
use volumetric_abi::{ChannelKind, SIGNED_DISTANCE_CHANNEL_NAME, TSDF_CHANNEL_KIND, is_occupied};

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

fn config(resolution: u64, band_width: f64) -> Vec<u8> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![
            (
                ciborium::value::Value::Text("resolution".to_string()),
                ciborium::value::Value::Integer(resolution.into()),
            ),
            (
                ciborium::value::Value::Text("band_width".to_string()),
                ciborium::value::Value::Float(band_width),
            ),
        ]),
        &mut bytes,
    )
    .unwrap();
    bytes
}

fn generate_sphere_sdf() -> Vec<u8> {
    let operator_wasm = wasm_artifact("sdf_operator");
    let mut operator = create_operator_executor(&operator_wasm).unwrap();
    let result = operator
        .run(OperatorIo::new(vec![
            wasm_artifact("simple_sphere_model"),
            config(32, 0.5),
        ]))
        .expect("SDF operator run");
    result.outputs.get(&0).cloned().expect("SDF model output")
}

#[test]
fn metadata_declares_model_config_and_model_output() {
    let metadata = volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("sdf_operator"))
        .expect("SDF metadata");
    assert_eq!(metadata.name, "sdf_operator");
    assert_eq!(metadata.inputs.len(), 2);
    assert!(matches!(
        metadata.inputs[0],
        OperatorMetadataInput::ModelWASM
    ));
    assert!(matches!(
        metadata.inputs[1],
        OperatorMetadataInput::CBORConfiguration(_)
    ));
    assert_eq!(metadata.input_name(0), Some("Model"));
    assert_eq!(metadata.input_name(1), Some("Config"));
    assert_eq!(metadata.outputs, vec![OperatorMetadataOutput::ModelWASM]);
}

#[test]
fn signed_distance_is_available_outside_geometry_bounds_and_field_bounds() {
    let output = generate_sphere_sdf();
    let mut model = create_model_executor(&output).expect("generated SDF model");

    let bounds = model.get_bounds_nd().unwrap();
    assert_eq!(bounds.as_slice(), &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    let format = model.sample_format().unwrap();
    assert_eq!(format.channels.len(), 2);
    assert_eq!(format.channels[0].kind, ChannelKind::Occupancy);
    assert_eq!(format.channels[1].name, SIGNED_DISTANCE_CHANNEL_NAME);
    assert_eq!(
        format.channels[1].kind,
        ChannelKind::Custom(TSDF_CHANNEL_KIND.to_string())
    );

    let center = model.sample_channels_nd(&[0.0, 0.0, 0.0]).unwrap();
    assert_eq!(center[0], 1.0);
    assert!((center[1] + 0.5).abs() < 1e-5, "center TSDF {}", center[1]);

    let inside = model.sample_channels_nd(&[0.75, 0.0, 0.0]).unwrap();
    assert_eq!(inside[0], 1.0);
    assert!((inside[1] + 0.25).abs() < 0.08, "inside TSDF {}", inside[1]);

    let outside_part = model.sample_channels_nd(&[1.2, 0.0, 0.0]).unwrap();
    assert_eq!(outside_part[0], 0.0);
    assert!(
        (outside_part[1] - 0.2).abs() < 0.08,
        "outside TSDF {}",
        outside_part[1]
    );

    // x=2 lies beyond the baked [-1.5, 1.5] field, and x=100 much farther;
    // both return the globally valid positive truncation value.
    for x in [2.0, 100.0] {
        let row = model.sample_channels_nd(&[x, 0.0, 0.0]).unwrap();
        assert_eq!(row, vec![0.0, 0.5], "at x={x}");
        assert!(!is_occupied(model.sample_nd(&[x, 0.0, 0.0]).unwrap()));
    }
}

#[test]
fn sdf_model_routes_through_the_project_dag() {
    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::model("sphere".to_string(), wasm_artifact("simple_sphere_model")),
            ImportedAsset::operator("sdf".to_string(), wasm_artifact("sdf_operator")),
        ],
        timeline: vec![ExecutionStep {
            operator_id: "sdf".to_string(),
            inputs: vec![
                ExecutionInput::AssetRef("sphere".to_string()),
                ExecutionInput::Inline(config(24, 0.25)),
            ],
            outputs: vec!["sphere_with_sdf".to_string()],
        }],
        exports: vec!["sphere_with_sdf".to_string()],
        baked: None,
    };
    assert!(project.validate().is_empty());

    let mut environment = Environment::new();
    let exports = project.run(&mut environment).expect("project run");
    let output = &exports[0];
    assert_eq!(output.id(), "sphere_with_sdf");
    assert_eq!(output.type_hint(), Some(AssetTypeHint::Model));
    assert_eq!(output.precursor_ids(), &["sphere"]);

    let mut model = create_model_executor(output.data()).unwrap();
    let row = model.sample_channels_nd(&[1.1, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 0.0);
    assert!(row[1] > 0.0);
}
