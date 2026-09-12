//! End-to-end test of survey_operator: a synthetic field (the survey card
//! and four swatches on a plane) observed from a ring of views, with the
//! projections as observations, gets its cameras, poses and map through
//! the real wasm pipeline.
//!
//! Requires the wasm32 artifact:
//!   cargo build --target wasm32-unknown-unknown --release -p survey_operator
#![cfg(feature = "native")]

use cv_core::survey::square_corners;
use volumetric::viewset::{
    Board, BoardSpec, CameraModel, CornerObs, MarkerObs, Observations, View, ViewSet,
    decode_viewset, encode_viewset,
};
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

/// A camera-to-world pose looking from `eye` at `target` with the
/// picture's up along `up` (OpenCV axes: x right, y down, z forward).
fn look_at(eye: [f64; 3], target: [f64; 3], up: [f64; 3]) -> [f64; 12] {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let unit = |a: [f64; 3]| {
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let z = unit(sub(target, eye));
    let x = unit(cross(z, up));
    let y = cross(z, x);
    [
        x[0], y[0], z[0], eye[0], //
        x[1], y[1], z[1], eye[1], //
        x[2], y[2], z[2], eye[2], //
    ]
}

/// The card at the origin of z = 0 and four swatches around it, seen
/// from sixteen views on two rings, projected without noise.
fn field() -> (ViewSet, Vec<(u32, [[f64; 3]; 4])>, Vec<[f64; 12]>) {
    let spec = BoardSpec::survey_card();
    let card: Vec<[f64; 3]> = (0..spec.n_corners())
        .map(|id| {
            let [x, y] = spec.corner(id).unwrap();
            [x, y, 0.0]
        })
        .collect();
    let swatches: Vec<(u32, [[f64; 3]; 4])> = vec![
        (
            3,
            square_corners([-0.3, 0.1, 0.0], 0.06, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ),
        (
            7,
            square_corners([0.5, -0.2, 0.0], 0.06, [0.6, 0.8, 0.0], [-0.8, 0.6, 0.0]),
        ),
        (
            11,
            square_corners([0.4, 0.5, 0.0], 0.09, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ),
        (
            20,
            square_corners([-0.2, 0.6, 0.0], 0.06, [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]),
        ),
    ];
    let truth = CameraModel::pinhole(3000, 2000, 5000.0, 5000.0, 1490.0, 1010.0);
    let seed = CameraModel::pinhole(3000, 2000, 4800.0, 4800.0, 1500.0, 1000.0);
    let mut set = ViewSet {
        cameras: vec![seed],
        board: Some(Board {
            spec: spec.clone(),
            corners: Vec::new(),
        }),
        ..ViewSet::default()
    };
    let mut poses = Vec::new();
    for k in 0..16 {
        let a = k as f64 / 16.0 * std::f64::consts::TAU;
        let (radius, height) = if k % 2 == 0 { (0.9, 1.3) } else { (0.5, 1.6) };
        let target = [
            0.1 + 0.15 * (a * 3.0).cos(),
            0.15 + 0.1 * (a * 2.0).sin(),
            0.0,
        ];
        let eye = [
            target[0] + radius * a.cos(),
            target[1] + radius * a.sin(),
            height,
        ];
        let pose = look_at(eye, target, [0.1 * a.sin(), 1.0, 0.0]);
        let view = View::posed(format!("v{k:02}"), 0, pose);
        let inside =
            |uv: [f64; 2]| uv[0] > 10.0 && uv[0] < 2990.0 && uv[1] > 10.0 && uv[1] < 1990.0;
        let mut obs = Observations {
            blur_px: Some(0.9),
            ..Default::default()
        };
        for (id, x) in card.iter().enumerate() {
            if let Some(uv) = view.project(&truth, *x)
                && inside(uv)
            {
                obs.board.push(CornerObs {
                    id: id as u32,
                    pixel: uv,
                    fit_px: 0.1,
                });
            }
        }
        for (id, corners) in &swatches {
            let uvs: Vec<[f64; 2]> = corners
                .iter()
                .filter_map(|c| view.project(&truth, *c))
                .collect();
            if uvs.len() == 4 && uvs.iter().all(|uv| inside(*uv)) {
                obs.markers.push(MarkerObs {
                    id: *id,
                    family: "5x5_100".to_string(),
                    corners: [uvs[0], uvs[1], uvs[2], uvs[3]],
                    fit_px: 0.1,
                });
            }
        }
        let mut raw = View::unposed(view.id.clone(), 0);
        raw.observations = Some(obs);
        set.views.push(raw);
        poses.push(pose);
    }
    (set, swatches, poses)
}

fn run(set: Vec<u8>, config: Vec<u8>) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "surveyor".to_string(),
        wasm_artifact("survey_operator"),
    ));
    project.imports.push(ImportedAsset::new(
        "views".to_string(),
        set,
        Some(AssetTypeHint::ViewSet),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "surveyor".to_string(),
        inputs: vec![
            ExecutionInput::AssetRef("views".to_string()),
            ExecutionInput::Inline(config),
        ],
        outputs: vec!["views_posed".to_string(), "report".to_string()],
    });
    project.exports = vec!["views_posed".to_string(), "report".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).map_err(|e| e.to_string())?;
    assert_eq!(exports[0].type_hint(), Some(AssetTypeHint::ViewSet));
    assert_eq!(exports[1].type_hint(), Some(AssetTypeHint::F64Map));
    Ok((exports[0].data().to_vec(), exports[1].data().to_vec()))
}

#[test]
fn metadata_declares_views_and_config_and_two_outputs() {
    let metadata = volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("survey_operator"))
        .expect("metadata");
    assert_eq!(metadata.name, "survey_operator");
    assert_eq!(metadata.category, "Import");
    assert!(matches!(metadata.inputs[0], OperatorMetadataInput::ViewSet));
    assert!(matches!(
        metadata.inputs[1],
        OperatorMetadataInput::CBORConfiguration(_)
    ));
    assert_eq!(
        metadata.outputs,
        vec![
            OperatorMetadataOutput::ViewSet,
            OperatorMetadataOutput::F64Map
        ]
    );
    assert_eq!(metadata.output_names, vec!["Views", "Report"]);
    assert!(metadata.docs.contains("calipers"));
}

#[test]
fn a_synthetic_field_is_surveyed_through_wasm() {
    let (set, swatches, poses) = field();
    let (posed, report) = run(encode_viewset(&set), Vec::new()).unwrap();
    let posed = decode_viewset(&posed).unwrap();
    let report = volumetric::f64_map::decode(&report).unwrap();
    assert_eq!(report["frames_posed"], 16.0, "{report:?}");
    assert_eq!(report["frames_rejected"], 0.0);
    assert!(report["rms_px"] < 0.05, "rms {}", report["rms_px"]);
    // The camera is recovered from the 4 % off seed.
    let cam = &posed.cameras[0];
    assert!(
        (cam.fx - 5000.0).abs() < 2.0
            && (cam.cx - 1490.0).abs() < 2.0
            && (cam.cy - 1010.0).abs() < 2.0,
        "{cam:?}"
    );
    // The world frame is the card's, y and z flipped; the swatches land
    // on their places, the views on theirs.
    // A swatch seen from fewer than two posed views is known only along
    // a ray and stays out; the rest are solved.
    for (id, _) in &swatches {
        let seen = set
            .views
            .iter()
            .filter(|v| {
                v.observations
                    .as_ref()
                    .unwrap()
                    .markers
                    .iter()
                    .any(|m| m.id == *id)
            })
            .count();
        assert!(
            posed.markers.iter().any(|m| m.id == *id) || seen < 2,
            "swatch {id} seen by {seen} views is unsolved"
        );
    }
    assert!(posed.markers.len() >= 3, "{} swatches", posed.markers.len());
    for m in &posed.markers {
        let (_, truth) = swatches.iter().find(|s| s.0 == m.id).unwrap();
        for j in 0..4 {
            let t = [truth[j][0], -truth[j][1], -truth[j][2]];
            let d: f64 = (0..3)
                .map(|i| (m.corners[j][i] - t[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(d < 1e-4, "swatch {} corner {j}: {d} m off", m.id);
        }
    }
    for (view, truth) in posed.views.iter().zip(&poses) {
        let pose = view.pose().expect("posed");
        let eye = [pose[3], pose[7], pose[11]];
        let t = [truth[3], -truth[7], -truth[11]];
        let d: f64 = (0..3).map(|i| (eye[i] - t[i]).powi(2)).sum::<f64>().sqrt();
        assert!(d < 2e-4, "view {}: eye {d} m off", view.id);
        assert!(view.tags.iter().any(|t| t == "solved:survey"));
        assert!(view.tags.iter().any(|t| t.starts_with("rms:")));
    }
    let board = posed.board.as_ref().unwrap();
    assert_eq!(board.corners.len(), 110);
    assert!(report["card_planarity_mm"] < 0.01);
    if let Some(side) = report.get("swatch.11.side_mm") {
        assert!((side - 90.0).abs() < 0.05, "{side}");
    }

    // A set without observations fails the step with a reason.
    let mut bare = set.clone();
    for v in &mut bare.views {
        v.observations = None;
    }
    let err = run(
        encode_viewset(&bare),
        config(vec![("rounds", ciborium::Value::Integer(2.into()))]),
    )
    .unwrap_err();
    assert!(err.contains("view-detect"), "{err}");
}
