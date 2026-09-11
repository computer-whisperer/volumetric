//! End-to-end test of view_detect_operator: a rendered still of the
//! survey card and two swatches, embedded in a view set, gets its
//! observations through the real wasm pipeline.
//!
//! Requires the wasm32 artifact:
//!   cargo build --target wasm32-unknown-unknown --release -p view_detect_operator
#![cfg(feature = "native")]

use cv_core::board::{PlacedBoard, Render, render_scene, square_marker};
use cv_core::dict::Dictionary;
use cv_core::gray::Gray;
use volumetric::viewset::{BoardSpec, CameraModel, View, ViewSet, decode_viewset, encode_viewset};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset,
    OperatorMetadataInput, OperatorMetadataOutput, Project,
};

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

fn config(entries: Vec<(&str, ciborium::Value)>) -> Vec<u8> {
    let value = ciborium::Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (ciborium::Value::Text(k.to_string()), v))
            .collect(),
    );
    let mut out = Vec::new();
    ciborium::into_writer(&value, &mut out).unwrap();
    out
}

/// A binary PGM of the picture.
fn pgm(gray: &Gray) -> Vec<u8> {
    let mut out = format!("P5\n{} {}\n255\n", gray.width, gray.height).into_bytes();
    out.extend_from_slice(&gray.pixels);
    out
}

/// Straight down from 1 m: the survey card and two swatches on the
/// floor, in a set with one view carrying the picture and one without.
fn scene() -> ViewSet {
    let camera = CameraModel::pinhole(1600, 1200, 2500.0, 2500.0, 800.0, 600.0);
    let view = View::posed(
        "top",
        0,
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, 0.0, //
            0.0, 0.0, -1.0, 1.0, //
        ],
    );
    let board = PlacedBoard::new(
        BoardSpec::survey_card(),
        [-0.28, 0.1, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    );
    let swatches = vec![
        square_marker(3, [0.0, 0.2, 0.0], 0.06, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        square_marker(
            42,
            [0.1, 0.05, 0.0],
            0.06,
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ),
    ];
    let picture = pgm(&render_scene(
        &camera,
        &view,
        &swatches,
        &Dictionary::aruco_5x5_100(),
        std::slice::from_ref(&board),
        &Render {
            blur_sigma: 0.8,
            ..Render::default()
        },
    ));
    let mut with_picture = view.clone();
    with_picture.image = Some(picture);
    let blind = View::posed("blind", 0, view.camera_to_world.unwrap());
    ViewSet {
        cameras: vec![camera],
        views: vec![with_picture, blind],
        ..ViewSet::default()
    }
}

fn run(set: Vec<u8>, config: Vec<u8>) -> Result<Vec<u8>, String> {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "detector".to_string(),
        wasm_artifact("view_detect_operator"),
    ));
    project.imports.push(ImportedAsset::new(
        "views".to_string(),
        set,
        Some(AssetTypeHint::ViewSet),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "detector".to_string(),
        inputs: vec![
            ExecutionInput::AssetRef("views".to_string()),
            ExecutionInput::Inline(config),
        ],
        outputs: vec!["views_detected".to_string()],
    });
    project.exports = vec!["views_detected".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).map_err(|e| e.to_string())?;
    assert_eq!(exports[0].type_hint(), Some(AssetTypeHint::ViewSet));
    Ok(exports[0].data().to_vec())
}

#[test]
fn metadata_declares_views_and_config_and_a_views_output() {
    let metadata =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("view_detect_operator"))
            .expect("metadata");
    assert_eq!(metadata.name, "view_detect_operator");
    assert_eq!(metadata.category, "Import");
    assert!(matches!(metadata.inputs[0], OperatorMetadataInput::ViewSet));
    assert!(matches!(
        metadata.inputs[1],
        OperatorMetadataInput::CBORConfiguration(_)
    ));
    assert_eq!(metadata.outputs, vec![OperatorMetadataOutput::ViewSet]);
    assert!(metadata.docs.contains("card"));
}

#[test]
fn observations_are_stored_on_the_views_through_wasm() {
    let set = scene();
    let out = run(encode_viewset(&set), Vec::new()).unwrap();
    let detected = decode_viewset(&out).unwrap();
    assert_eq!(detected.views.len(), 2);
    let obs = detected.views[0].observations.as_ref().unwrap();
    assert_eq!(
        obs.markers.iter().filter(|m| m.family == "5x5_100").count(),
        2
    );
    assert_eq!(
        obs.markers.iter().filter(|m| m.family == "36h11").count(),
        66
    );
    assert_eq!(obs.board.len(), 110);
    assert!(obs.blur_px.is_some_and(|b| b > 0.5 && b < 1.3));
    assert!(detected.views[1].observations.is_none());
    assert_eq!(
        detected.board.as_ref().unwrap().spec,
        BoardSpec::survey_card()
    );

    // Swatches only, at full resolution.
    let out = run(
        encode_viewset(&set),
        config(vec![
            ("card", ciborium::Value::Bool(false)),
            ("search_px", ciborium::Value::Integer(0.into())),
        ]),
    )
    .unwrap();
    let detected = decode_viewset(&out).unwrap();
    let obs = detected.views[0].observations.as_ref().unwrap();
    assert_eq!(obs.markers.len(), 2);
    assert!(obs.board.is_empty());
    assert!(detected.board.is_none());

    // Nothing to look for fails the step with a reason.
    let err = run(
        encode_viewset(&set),
        config(vec![
            ("card", ciborium::Value::Bool(false)),
            ("dictionary", ciborium::Value::Text("none".to_string())),
        ]),
    )
    .unwrap_err();
    assert!(err.contains("nothing to look for"), "{err}");
}
