//! Integration tests for the typed sample-channel model ABI (github issue #3
//! follow-through): `get_sample_format` / `sample_channels`, the occupancy
//! default for plain models, format pass-through with position rewriting in
//! transform wrappers, and channel pass-through in the boolean operator
//! (model A's layout with channel 0 replaced by the combined occupancy).
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::wasm::{
    NativeModelExecutor, OperatorExecutor, OperatorIo, create_operator_executor,
};
use volumetric_abi::ChannelKind;

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

/// Run a one-model-input operator and return its output model wasm.
fn run_operator(operator: &str, inputs: Vec<Vec<u8>>) -> Vec<u8> {
    let operator_wasm = wasm_artifact(operator);
    let mut executor = create_operator_executor(&operator_wasm).expect("create operator executor");
    let result = executor.run(OperatorIo::new(inputs)).expect("run operator");
    result
        .outputs
        .get(&0)
        .expect("operator posted no output")
        .clone()
}

fn cbor_map(fields: &[(&str, f32)]) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(
            fields
                .iter()
                .map(|(k, v)| {
                    (
                        ciborium::value::Value::Text((*k).into()),
                        ciborium::value::Value::Float(*v as f64),
                    )
                })
                .collect(),
        ),
        &mut out,
    )
    .unwrap();
    out
}

#[test]
fn density_model_declares_occupancy_plus_density() {
    let executor = NativeModelExecutor::new(&wasm_artifact("density_gradient_model")).unwrap();
    let format = executor.sample_format().clone();
    assert_eq!(format.channels.len(), 2);
    assert_eq!(format.channels[0].kind, ChannelKind::Occupancy);
    assert_eq!(format.channels[1].kind, ChannelKind::Density);
    assert_eq!(format.channels[1].name, "infill");
}

#[test]
fn sample_channels_agrees_with_plain_sample() {
    let mut executor = NativeModelExecutor::new(&wasm_artifact("density_gradient_model")).unwrap();

    // Inside: occupancy 1.0, density follows the x-gradient 0.5 + 0.5x.
    let row = executor.sample_channels_nd(&[0.5, 0.0, 0.0]).unwrap();
    assert_eq!(row.len(), 2);
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.75).abs() < 1e-6, "density was {}", row[1]);
    assert_eq!(row[0], executor.sample_nd(&[0.5, 0.0, 0.0]).unwrap());

    // Outside: occupancy 0.0.
    let row = executor.sample_channels_nd(&[2.0, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 0.0);
}

#[test]
fn plain_models_default_to_occupancy_only() {
    let mut executor = NativeModelExecutor::new(&wasm_artifact("simple_sphere_model")).unwrap();
    let format = executor.sample_format().clone();
    assert_eq!(format.channels.len(), 1);
    assert_eq!(format.channels[0].kind, ChannelKind::Occupancy);

    // sample_channels_nd still works: the row is just the occupancy sample.
    let row = executor.sample_channels_nd(&[0.0, 0.0, 0.0]).unwrap();
    assert_eq!(row, vec![1.0]);
}

#[test]
fn translate_forwards_format_and_rewrites_channel_positions() {
    let translated = run_operator(
        "translate_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            cbor_map(&[("dx", 0.5), ("dy", 0.0), ("dz", 0.0)]),
        ],
    );
    let mut executor = NativeModelExecutor::new(&translated).unwrap();

    // The format passes through untouched.
    assert_eq!(executor.sample_format().channels.len(), 2);

    // Sampling at x = 0.75 in translated space must hit the original model at
    // x = 0.25: density = 0.5 + 0.5 * 0.25 = 0.625. A wrapper that forgot to
    // rewrite the sample_channels position would return 0.875 instead.
    let row = executor.sample_channels_nd(&[0.75, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.625).abs() < 1e-6, "density was {}", row[1]);
}

fn boolean_config(op: &str) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text("op".into()),
            ciborium::value::Value::Text(op.into()),
        )]),
        &mut out,
    )
    .unwrap();
    out
}

#[test]
fn boolean_passes_model_a_channels_through() {
    // Density cube minus the sphere: A's format survives, channel 0 is
    // the boolean result, and the density channel keeps A's values even
    // where the occupancy changed.
    let merged = run_operator(
        "boolean_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            wasm_artifact("simple_sphere_model"),
            boolean_config("subtract"),
        ],
    );
    let mut executor = NativeModelExecutor::new(&merged).unwrap();

    let format = executor.sample_format().clone();
    assert_eq!(format.channels.len(), 2);
    assert_eq!(format.channels[0].kind, ChannelKind::Occupancy);
    assert_eq!(format.channels[1].kind, ChannelKind::Density);

    // Cube corner outside the sphere: solid, density 0.5 + 0.5 * 0.9.
    let row = executor.sample_channels_nd(&[0.9, 0.9, 0.9]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.95).abs() < 1e-6, "density was {}", row[1]);
    assert_eq!(row[0], executor.sample_nd(&[0.9, 0.9, 0.9]).unwrap());

    // Center: subtracted away, but the density channel still reports A's
    // field there.
    let row = executor.sample_channels_nd(&[0.0, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 0.0);
    assert!((row[1] - 0.5).abs() < 1e-6, "density was {}", row[1]);
    assert_eq!(row[0], executor.sample_nd(&[0.0, 0.0, 0.0]).unwrap());
}

#[test]
fn boolean_with_formatless_a_stays_occupancy_only() {
    // A has no declared format, so the output is occupancy-only even
    // though B declares channels: the layout follows model A.
    let merged = run_operator(
        "boolean_operator",
        vec![
            wasm_artifact("simple_sphere_model"),
            wasm_artifact("density_gradient_model"),
            boolean_config("union"),
        ],
    );
    let mut executor = NativeModelExecutor::new(&merged).unwrap();

    assert_eq!(executor.sample_format().channels.len(), 1);
    assert_eq!(
        executor.sample_format().channels[0].kind,
        ChannelKind::Occupancy
    );
    assert!(volumetric_abi::is_occupied(
        executor.sample_nd(&[0.9, 0.9, 0.9]).unwrap()
    ));
}
