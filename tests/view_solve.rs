//! End-to-end test of view_solve_operator: a rendered still of marker
//! cards joins a view set as a posed view through the real wasm pipeline.
//!
//! Requires the wasm32 artifact:
//!   cargo build --target wasm32-unknown-unknown --release -p view_solve_operator
#![cfg(feature = "native")]

use cv_core::board::{Render, render, square_marker};
use cv_core::dict::Dictionary;
use cv_core::gray::Gray;
use cv_core::pnp::pose_difference;
use volumetric::viewset::{CameraModel, View, ViewSet, decode_viewset, encode_viewset};
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

/// A tilted camera over five cards on the floor, the picture it takes,
/// and the set carrying the cards' map.
fn scene() -> (View, Vec<u8>, ViewSet) {
    let truth = CameraModel::pinhole(1280, 960, 1000.0, 1000.0, 640.0, 480.0);
    let pitch: f64 = 50f64.to_radians();
    let (sn, cs) = pitch.sin_cos();
    let (forward, down) = ([0.0, cs, -sn], [0.0, -sn, -cs]);
    let view = View::posed(
        "truth",
        0,
        [
            1.0, down[0], forward[0], 0.1, //
            0.0, down[1], forward[1], -0.4, //
            0.0, down[2], forward[2], 1.2, //
        ],
    );
    let right = [1.0, 0.0, 0.0];
    let floor_down = [0.0, -1.0, 0.0];
    let markers = vec![
        square_marker(0, [-0.45, 0.85, 0.0], 0.12, right, floor_down),
        square_marker(1, [0.3, 0.9, 0.0], 0.12, right, floor_down),
        square_marker(2, [-0.35, 0.45, 0.0], 0.12, right, floor_down),
        square_marker(5, [0.25, 0.4, 0.0], 0.12, right, floor_down),
        square_marker(49, [-0.05, 0.65, 0.0], 0.16, right, floor_down),
    ];
    let picture = pgm(&render(
        &truth,
        &view,
        &markers,
        &Dictionary::aruco_5x5_100(),
        &Render::default(),
    ));
    let set = ViewSet {
        board: None,
        schema: 2,
        world: Default::default(),
        provenance: Default::default(),
        cameras: vec![truth],
        views: Vec::new(),
        markers,
    };
    (view, picture, set)
}

fn run(set: Vec<u8>, picture: Vec<u8>, config: Vec<u8>) -> Result<Vec<u8>, String> {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "solver".to_string(),
        wasm_artifact("view_solve_operator"),
    ));
    project.imports.push(ImportedAsset::new(
        "views".to_string(),
        set,
        Some(AssetTypeHint::ViewSet),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "solver".to_string(),
        inputs: vec![
            ExecutionInput::AssetRef("views".to_string()),
            ExecutionInput::Inline(picture),
            ExecutionInput::Inline(config),
        ],
        outputs: vec!["views_with_still".to_string()],
    });
    project.exports = vec!["views_with_still".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).map_err(|e| e.to_string())?;
    assert_eq!(exports[0].type_hint(), Some(AssetTypeHint::ViewSet));
    Ok(exports[0].data().to_vec())
}

#[test]
fn metadata_declares_views_picture_config_and_a_views_output() {
    let metadata =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("view_solve_operator"))
            .expect("metadata");
    assert_eq!(metadata.name, "view_solve_operator");
    assert_eq!(metadata.category, "Import");
    assert!(matches!(metadata.inputs[0], OperatorMetadataInput::ViewSet));
    assert!(matches!(metadata.inputs[1], OperatorMetadataInput::Blob));
    assert!(matches!(
        metadata.inputs[2],
        OperatorMetadataInput::CBORConfiguration(_)
    ));
    assert_eq!(metadata.outputs, vec![OperatorMetadataOutput::ViewSet]);
    assert!(metadata.docs.contains("marker"));
}

#[test]
fn a_still_is_posed_and_appended_through_wasm() {
    let (truth, picture, set) = scene();
    let out = run(
        encode_viewset(&set),
        picture,
        config(vec![
            ("id", ciborium::Value::Text("phone".to_string())),
            ("focal_px", ciborium::Value::Float(1000.0)),
        ]),
    )
    .unwrap();
    let solved = decode_viewset(&out).unwrap();
    assert_eq!(solved.views.len(), 1);
    assert_eq!(solved.markers.len(), 5, "the map survives");
    let (added, camera) = solved.view("phone").unwrap();
    let (angle, dist) = pose_difference(added.pose().unwrap(), truth.pose().unwrap());
    assert!(angle < 0.002 && dist < 0.003, "{angle} rad, {dist} m");
    assert_eq!(camera.fx, 1000.0);
    assert!(added.image.is_some());
    assert!(added.tags.iter().any(|t| t == "solved:markers"));
    assert!(added.tags.iter().any(|t| t.starts_with("rms:")));

    // A picture without cards fails the step with a reason.
    let blank = pgm(&Gray::new(64, 48));
    let err = run(encode_viewset(&set), blank, Vec::new()).unwrap_err();
    assert!(err.contains("cards found"), "{err}");
}
