//! End-to-end test of density extraction: the full pipeline (box → grid
//! mesh → solve under a sphere → density model) produces a ModelWASM whose
//! geometry matches the box and whose Density channel reflects the strain
//! energy distribution.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p simple_sphere_model -p rectangular_prism_operator \
//!     -p translate_operator -p fea_grid_mesh_operator \
//!     -p fea_solve_operator -p fea_density_operator

#![cfg(feature = "native")]

use volumetric::wasm::NativeModelExecutor;
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::{ChannelKind, is_occupied};

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

#[test]
fn density_model_end_to_end() {
    use ciborium::value::Value;

    // The fea_solve e2e scenario plus the density step: unit box, meshed at
    // resolution 8, pressed 0.1 by a unit sphere from +z, then the solved
    // energies mapped onto densities [0.2, 1.0] over the box's geometry.
    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::model("sphere".to_string(), wasm_artifact("simple_sphere_model")),
            ImportedAsset::operator(
                "prism".to_string(),
                wasm_artifact("rectangular_prism_operator"),
            ),
            ImportedAsset::operator("translate".to_string(), wasm_artifact("translate_operator")),
            ImportedAsset::operator(
                "mesher".to_string(),
                wasm_artifact("fea_grid_mesh_operator"),
            ),
            ImportedAsset::operator("solver".to_string(), wasm_artifact("fea_solve_operator")),
            ImportedAsset::operator("densify".to_string(), wasm_artifact("fea_density_operator")),
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
                operator_id: "mesher".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("box".to_string()),
                    ExecutionInput::Inline(cbor_map(&[("resolution", Value::Integer(8.into()))])),
                ],
                outputs: vec!["mesh".to_string()],
            },
            ExecutionStep {
                operator_id: "solver".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("mesh".to_string()),
                    ExecutionInput::AssetRef("butt".to_string()),
                    ExecutionInput::Inline(Vec::new()),
                ],
                outputs: vec!["solved".to_string()],
            },
            ExecutionStep {
                operator_id: "densify".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("solved".to_string()),
                    ExecutionInput::AssetRef("box".to_string()),
                    ExecutionInput::Inline(cbor_map(&[
                        ("min_density", Value::Float(0.2)),
                        ("max_density", Value::Float(1.0)),
                    ])),
                ],
                outputs: vec!["density_model".to_string()],
            },
        ],
        exports: vec!["density_model".to_string()],
        baked: None,
    };

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let asset = &exports[0];
    assert_eq!(asset.type_hint(), Some(AssetTypeHint::Model));

    let mut executor = NativeModelExecutor::new(asset.data()).expect("density model instantiates");

    // Declared format: [Occupancy, Density].
    let format = executor.sample_format().clone();
    assert_eq!(format.channels.len(), 2);
    assert_eq!(format.channels[0].kind, ChannelKind::Occupancy);
    assert_eq!(format.channels[1].kind, ChannelKind::Density);

    // Geometry passes through: the box is intact.
    assert!(is_occupied(executor.sample_nd(&[0.5, 0.5, 0.5]).unwrap()));
    assert!(!is_occupied(executor.sample_nd(&[1.5, 0.5, 0.5]).unwrap()));

    // Channel 0 always agrees with plain sample; densities stay in range
    // and normalize the peak to max_density. Probe every cell center.
    let mut max_density = 0.0f32;
    let mut min_density = f32::INFINITY;
    for k in 0..8 {
        for j in 0..8 {
            for i in 0..8 {
                let p = [
                    (i as f64 + 0.5) / 8.0,
                    (j as f64 + 0.5) / 8.0,
                    (k as f64 + 0.5) / 8.0,
                ];
                let row = executor.sample_channels_nd(&p).unwrap();
                assert_eq!(row.len(), 2);
                let occ = executor.sample_nd(&p).unwrap();
                assert_eq!(is_occupied(row[0]), is_occupied(occ), "at {p:?}");
                assert!(
                    (0.2..=1.0 + 1e-6).contains(&(row[1] as f64)),
                    "density {} out of range at {p:?}",
                    row[1]
                );
                max_density = max_density.max(row[1]);
                min_density = min_density.min(row[1]);
            }
        }
    }
    assert!(
        (max_density - 1.0).abs() < 1e-6,
        "peak density {max_density}, expected 1.0"
    );
    assert!(min_density >= 0.2 - 1e-6);
    assert!(max_density > min_density, "density field is flat");

    // Energy concentrates under the indenter: the cell below the dip beats
    // a bottom corner cell.
    let under = executor.sample_channels_nd(&[0.5, 0.5, 0.9375]).unwrap()[1];
    let corner = executor
        .sample_channels_nd(&[0.0625, 0.0625, 0.0625])
        .unwrap()[1];
    assert!(
        under > corner + 0.1,
        "no concentration: under={under} corner={corner}"
    );

    // Outside the solid the lookup clamps to the nearest cell (density is
    // only meaningful where occupancy says inside, per the ABI).
    let above = executor.sample_channels_nd(&[0.5, 0.5, 1.4]).unwrap();
    assert!(!is_occupied(above[0]));
    assert!(
        (above[1] - under).abs() < 1e-6,
        "clamped lookup should hit the top cell"
    );
}

#[test]
fn unsolved_mesh_is_rejected() {
    // Skipping the solver: the density operator must ask for a solve.
    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::model("sphere".to_string(), wasm_artifact("simple_sphere_model")),
            ImportedAsset::operator(
                "mesher".to_string(),
                wasm_artifact("fea_grid_mesh_operator"),
            ),
            ImportedAsset::operator("densify".to_string(), wasm_artifact("fea_density_operator")),
        ],
        timeline: vec![
            ExecutionStep {
                operator_id: "mesher".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("sphere".to_string()),
                    ExecutionInput::Inline(Vec::new()),
                ],
                outputs: vec!["mesh".to_string()],
            },
            ExecutionStep {
                operator_id: "densify".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("mesh".to_string()),
                    ExecutionInput::AssetRef("sphere".to_string()),
                    ExecutionInput::Inline(Vec::new()),
                ],
                outputs: vec!["density_model".to_string()],
            },
        ],
        exports: vec!["density_model".to_string()],
        baked: None,
    };

    let mut env = Environment::new();
    let err = project.run(&mut env).expect_err("must require a solve");
    assert!(
        err.to_string().contains("strain_energy_density"),
        "unexpected error: {err}"
    );
}
