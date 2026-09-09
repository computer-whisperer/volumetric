//! End-to-end tests of the cloud operators: a Point1 cloud goes through
//! cloud_normals_operator and cloud_fit_operator in the real wasm
//! pipeline, and the fitted feature comes back as a Subspace with an
//! F64Map of the fit.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p cloud_fit_operator -p cloud_normals_operator

#![cfg(feature = "native")]

use volumetric::f64_map::decode as decode_f64_map;
use volumetric::fea::{
    FeaElementKind, FeaMesh, NORMAL_FIELD_NAME, decode_fea_mesh, encode_fea_mesh,
};
use volumetric::subspace::{Subspace, decode_subspace, encode_subspace};
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

fn cloud(points: &[[f64; 3]]) -> Vec<u8> {
    encode_fea_mesh(&FeaMesh {
        element_kind: FeaElementKind::Point1,
        node_positions: points.iter().flat_map(|p| p.iter().copied()).collect(),
        connectivity: (0..points.len() as u32).collect(),
        node_fields: vec![],
        element_fields: vec![],
    })
}

/// A deterministic pseudo-random sequence in [0, 1).
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Points on the plane z = 0.3 over [0, 1]^2 (in metres) plus scattered
/// outliers, slightly noisy.
fn plane_cloud() -> Vec<[f64; 3]> {
    let mut rng = Rng(1);
    let mut points: Vec<[f64; 3]> = (0..3000)
        .map(|_| [rng.unit(), rng.unit(), 0.3 + (rng.unit() - 0.5) * 2e-3])
        .collect();
    points.extend((0..500).map(|_| [rng.unit(), rng.unit(), rng.unit()]));
    points
}

/// Points on a cylinder of radius 0.05 about the y axis through
/// (0.2, *, 0.1), y in [0, 0.4], dense enough for normals.
fn cylinder_cloud() -> Vec<[f64; 3]> {
    let mut rng = Rng(2);
    (0..6000)
        .map(|_| {
            let a = rng.unit() * std::f64::consts::TAU;
            let r = 0.05 + (rng.unit() - 0.5) * 1e-3;
            [0.2 + r * a.cos(), rng.unit() * 0.4, 0.1 + r * a.sin()]
        })
        .collect()
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

fn text(s: &str) -> ciborium::Value {
    ciborium::Value::Text(s.to_string())
}

#[test]
fn metadata_declares_cloud_seed_config_and_two_outputs() {
    let metadata =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("cloud_fit_operator"))
            .expect("metadata");
    assert_eq!(metadata.name, "cloud_fit_operator");
    assert!(matches!(metadata.inputs[0], OperatorMetadataInput::FeaMesh));
    assert!(matches!(
        metadata.inputs[1],
        OperatorMetadataInput::Subspace
    ));
    assert!(matches!(
        metadata.inputs[2],
        OperatorMetadataInput::CBORConfiguration(_)
    ));
    assert_eq!(
        metadata.outputs,
        vec![
            OperatorMetadataOutput::Subspace,
            OperatorMetadataOutput::F64Map
        ]
    );
    assert_eq!(metadata.output_names, vec!["Feature", "Fit"]);
    let normals =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("cloud_normals_operator"))
            .expect("metadata");
    assert_eq!(normals.outputs, vec![OperatorMetadataOutput::FeaMesh]);
}

#[test]
fn a_plane_fit_yields_a_subspace_and_its_statistics() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "fit".to_string(),
        wasm_artifact("cloud_fit_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "fit".to_string(),
        inputs: vec![
            ExecutionInput::Inline(cloud(&plane_cloud())),
            ExecutionInput::Inline(vec![]),
            ExecutionInput::Inline(config(vec![
                ("kind", text("plane")),
                ("tolerance", ciborium::Value::Float(3e-3)),
            ])),
        ],
        outputs: vec!["plane".to_string(), "fit".to_string()],
    });
    project.exports = vec!["plane".to_string(), "fit".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");
    assert_eq!(exports[0].type_hint(), Some(AssetTypeHint::Subspace));
    assert_eq!(exports[1].type_hint(), Some(AssetTypeHint::F64Map));

    let plane = decode_subspace(exports[0].data()).unwrap();
    assert_eq!(plane.rank(), 2);
    assert!((plane.origin[2] - 0.3).abs() < 1e-3, "{:?}", plane.origin);
    let normal = plane.normal().unwrap();
    assert!(normal[2].abs() > 0.9999, "{normal:?}");
    // The cloud's centroid (outliers fill the unit cube) lies above z=0.3
    // on average? No: at z ~0.31 for the mix; the plane faces away from
    // it, so its normal points to -z.
    assert!(normal[2] < 0.0, "{normal:?}");

    let fit = decode_f64_map(exports[1].data()).unwrap();
    assert_eq!(fit["points"], 3500.0);
    assert!(
        fit["inliers"] >= 3000.0 && fit["inliers"] < 3100.0,
        "{fit:?}"
    );
    assert!(fit["rms"] < 1e-3, "{fit:?}");
    assert_eq!(fit["tolerance"], 3e-3);
    assert!((fit["extent_0"] - 1.0).abs() < 0.1 && (fit["extent_1"] - 1.0).abs() < 0.1);
    assert!(!fit.contains_key("radius"));
}

#[test]
fn normals_then_a_cylinder_fit_chain_through_the_dag() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "normals".to_string(),
        wasm_artifact("cloud_normals_operator"),
    ));
    project.imports.push(ImportedAsset::operator(
        "fit".to_string(),
        wasm_artifact("cloud_fit_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "normals".to_string(),
        inputs: vec![
            ExecutionInput::Inline(cloud(&cylinder_cloud())),
            ExecutionInput::Inline(vec![]),
        ],
        outputs: vec!["with_normals".to_string()],
    });
    project.timeline.push(ExecutionStep {
        operator_id: "fit".to_string(),
        inputs: vec![
            ExecutionInput::AssetRef("with_normals".to_string()),
            ExecutionInput::Inline(vec![]),
            ExecutionInput::Inline(config(vec![
                ("kind", text("cylinder")),
                ("tolerance", ciborium::Value::Float(2e-3)),
            ])),
        ],
        outputs: vec!["axis".to_string(), "fit".to_string()],
    });
    project.exports = vec![
        "with_normals".to_string(),
        "axis".to_string(),
        "fit".to_string(),
    ];
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");

    let mesh = decode_fea_mesh(exports[0].data()).unwrap();
    let normal = mesh
        .node_fields
        .iter()
        .find(|f| f.name == NORMAL_FIELD_NAME)
        .expect("normal field");
    assert_eq!(normal.components, 3);
    assert_eq!(normal.data.len(), mesh.node_count() * 3);

    let axis = decode_subspace(exports[1].data()).unwrap();
    assert_eq!(axis.rank(), 1);
    let d = axis.basis_vector(0);
    assert!(d[1].abs() > 0.9999, "axis {d:?}");
    assert!((axis.origin[0] - 0.2).abs() < 1e-3 && (axis.origin[2] - 0.1).abs() < 1e-3);
    let fit = decode_f64_map(exports[2].data()).unwrap();
    assert!((fit["radius"] - 0.05).abs() < 5e-4, "{fit:?}");
    assert!((fit["extent_0"] - 0.4).abs() < 0.02, "{fit:?}");
    assert!(fit["inliers"] > 5900.0, "{fit:?}");
}

#[test]
fn a_seed_line_fits_a_cylinder_without_normals_and_bad_seeds_are_named() {
    let seed = Subspace::axis_aligned(vec![0.0; 3], &[1]).unwrap();
    let run = |seed: &Subspace| {
        let mut project = Project::new();
        project.imports.push(ImportedAsset::operator(
            "fit".to_string(),
            wasm_artifact("cloud_fit_operator"),
        ));
        project.timeline.push(ExecutionStep {
            operator_id: "fit".to_string(),
            inputs: vec![
                ExecutionInput::Inline(cloud(&cylinder_cloud())),
                ExecutionInput::Inline(encode_subspace(seed)),
                ExecutionInput::Inline(config(vec![
                    ("kind", text("cylinder")),
                    ("tolerance", ciborium::Value::Float(2e-3)),
                ])),
            ],
            outputs: vec!["axis".to_string(), "fit".to_string()],
        });
        project.exports = vec!["axis".to_string(), "fit".to_string()];
        let mut env = Environment::new();
        project.run(&mut env).map_err(|e| e.to_string())
    };
    let exports = run(&seed).unwrap();
    let fit = decode_f64_map(exports[1].data()).unwrap();
    assert!((fit["radius"] - 0.05).abs() < 5e-4, "{fit:?}");

    let plane = Subspace::axis_aligned(vec![0.0; 3], &[0, 1]).unwrap();
    let err = run(&plane).unwrap_err();
    assert!(err.contains("rank-2 seed"), "{err}");
}
