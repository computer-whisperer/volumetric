//! End-to-end test of the strut-pattern pipeline: a full project builds a
//! unit box, fills it with an explicit tetra strut lattice (Bar2 FeaMesh),
//! and compresses that lattice under a rigid sphere — the a→b→c legs of
//! the strut workflow (pattern → FEA lattice → frame solve).
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p simple_sphere_model -p rectangular_prism_operator \
//!     -p translate_operator -p strut_pattern_operator -p fea_solve_operator

#![cfg(feature = "native")]

use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::fea::{FeaElementKind, FeaMesh, decode_fea_mesh};

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

/// Unit box → strut pattern; optionally chain a frame solve against the
/// sphere pressed into the top. Returns the exported meshes by name.
fn run_project(
    pattern_config: Vec<u8>,
    with_solve: bool,
) -> Vec<(String, FeaMesh, AssetTypeHint)> {
    use ciborium::value::Value;

    let dip = 0.05;
    let mut imports = vec![
        ImportedAsset::operator(
            "prism".to_string(),
            wasm_artifact("rectangular_prism_operator"),
        ),
        ImportedAsset::operator(
            "pattern".to_string(),
            wasm_artifact("strut_pattern_operator"),
        ),
    ];
    let mut timeline = vec![
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
            operator_id: "pattern".to_string(),
            inputs: vec![
                ExecutionInput::AssetRef("box".to_string()),
                ExecutionInput::Inline(pattern_config),
            ],
            outputs: vec!["lattice".to_string()],
        },
    ];
    let mut exports = vec!["lattice".to_string()];

    if with_solve {
        imports.push(ImportedAsset::model(
            "sphere".to_string(),
            wasm_artifact("simple_sphere_model"),
        ));
        imports.push(ImportedAsset::operator(
            "translate".to_string(),
            wasm_artifact("translate_operator"),
        ));
        imports.push(ImportedAsset::operator(
            "solver".to_string(),
            wasm_artifact("fea_solve_operator"),
        ));
        timeline.push(ExecutionStep {
            operator_id: "translate".to_string(),
            inputs: vec![
                ExecutionInput::AssetRef("sphere".to_string()),
                ExecutionInput::Inline(cbor_map(&[
                    ("dx", Value::Float(0.5)),
                    ("dy", Value::Float(0.5)),
                    ("dz", Value::Float(2.0 - dip)),
                ])),
            ],
            outputs: vec!["butt".to_string()],
        });
        timeline.push(ExecutionStep {
            operator_id: "solver".to_string(),
            inputs: vec![
                ExecutionInput::AssetRef("lattice".to_string()),
                ExecutionInput::AssetRef("butt".to_string()),
                ExecutionInput::Inline(cbor_map(&[("fixed_boundary", Value::Text("zmin".into()))])),
            ],
            outputs: vec!["solved".to_string()],
        });
        exports.push("solved".to_string());
    }

    let project = Project {
        version: 2,
        imports,
        timeline,
        exports,
    };
    let mut env = Environment::new();
    let outputs = project.run(&mut env).expect("project run failed");
    outputs
        .iter()
        .map(|asset| {
            (
                asset.id().to_string(),
                decode_fea_mesh(asset.data()).expect("output is not a valid FeaMesh"),
                asset.type_hint().expect("output has no type hint"),
            )
        })
        .collect()
}

#[test]
fn box_fills_with_a_valid_tetra_lattice() {
    use ciborium::value::Value;
    let config = cbor_map(&[
        ("family", Value::Text("tetra".into())),
        ("cell_size", Value::Float(0.25)),
    ]);
    let outputs = run_project(config, false);
    let (_, mesh, hint) = &outputs[0];

    assert_eq!(*hint, AssetTypeHint::FeaMesh);
    assert_eq!(mesh.element_kind, FeaElementKind::Bar2);
    assert!(
        mesh.element_count() > 200,
        "suspiciously few struts: {}",
        mesh.element_count()
    );

    // Nodes stay inside the box (clipped nodes exactly on its faces).
    for n in 0..mesh.node_count() {
        let p = mesh.node_position(n);
        for axis in 0..3 {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&p[axis]),
                "node {n} outside the box: {p:?}"
            );
        }
    }

    // The default radius is cell_size / 10.
    let radius = mesh
        .element_fields
        .iter()
        .find(|f| f.name == "radius")
        .expect("radius field");
    assert!(radius.data.iter().all(|&r| (r - 0.025).abs() < 1e-12));

    // Struts never exceed the tetra bond length.
    let bond = 3.0f64.sqrt() / 4.0 * 0.25;
    for e in 0..mesh.element_count() {
        let pair = mesh.element(e);
        let a = mesh.node_position(pair[0] as usize);
        let b = mesh.node_position(pair[1] as usize);
        let len = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
        assert!(len <= bond + 1e-9, "strut {e} too long: {len}");
        assert!(len > 1e-6, "strut {e} degenerate: {len}");
    }
}

#[test]
fn box_fills_with_a_valid_foam_lattice() {
    use ciborium::value::Value;
    let config = cbor_map(&[
        ("family", Value::Text("foam".into())),
        ("cell_size", Value::Float(0.25)),
        ("irregularity", Value::Float(0.4)),
    ]);
    let outputs = run_project(config, false);
    let (_, mesh, hint) = &outputs[0];

    assert_eq!(*hint, AssetTypeHint::FeaMesh);
    assert_eq!(mesh.element_kind, FeaElementKind::Bar2);
    // ~24 struts per cell over 4^3 cells, minus boundary losses.
    assert!(
        mesh.element_count() > 400,
        "suspiciously few foam struts: {}",
        mesh.element_count()
    );

    // Nodes stay inside the box; interior nodes have the Plateau degree 4
    // (clipped skin nodes are degree 1 by construction).
    let mut degree = vec![0usize; mesh.node_count()];
    for e in 0..mesh.element_count() {
        degree[mesh.element(e)[0] as usize] += 1;
        degree[mesh.element(e)[1] as usize] += 1;
    }
    let mut interior_checked = 0usize;
    for n in 0..mesh.node_count() {
        let p = mesh.node_position(n);
        for axis in 0..3 {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&p[axis]),
                "node {n} outside the box: {p:?}"
            );
        }
        if p.iter().all(|&v| v > 0.3 && v < 0.7) {
            assert_eq!(degree[n], 4, "interior foam node {n} joins {} struts", degree[n]);
            interior_checked += 1;
        }
    }
    assert!(interior_checked > 5, "no interior nodes checked");
}

#[test]
fn generated_lattice_survives_a_frame_solve() {
    use ciborium::value::Value;
    let config = cbor_map(&[
        ("family", Value::Text("cubic".into())),
        ("cell_size", Value::Float(0.25)),
        ("radius", Value::Float(0.02)),
    ]);
    let outputs = run_project(config, true);
    let solved = outputs
        .iter()
        .find(|(name, _, _)| name == "solved")
        .map(|(_, mesh, _)| mesh)
        .expect("solved output present");

    // The solve produced frame result fields.
    for field in ["displacement", "rotation", "contact_force"] {
        assert!(
            solved
                .node_fields
                .iter()
                .any(|f| f.name == field && f.components == 3),
            "missing node field {field}"
        );
    }
    assert!(
        solved
            .element_fields
            .iter()
            .any(|f| f.name == "strain_energy_density" && f.components == 1),
        "missing strain_energy_density"
    );

    // The sphere pressed the lattice: net downward contact force.
    let contact = solved
        .node_fields
        .iter()
        .find(|f| f.name == "contact_force")
        .unwrap();
    let total_fz: f64 = contact.data.chunks_exact(3).map(|f| f[2]).sum();
    assert!(total_fz < 0.0, "expected downward net force, got {total_fz}");

    // Top-center node (0.5, 0.5, 1.0) is a cubic lattice point under the
    // sphere's deepest dip: pressed down by ~the dip depth.
    let displacement = solved
        .node_fields
        .iter()
        .find(|f| f.name == "displacement")
        .unwrap();
    let top_center = (0..solved.node_count())
        .find(|&n| {
            let p = solved.node_position(n);
            (p[0] - 0.5).abs() < 1e-9 && (p[1] - 0.5).abs() < 1e-9 && (p[2] - 1.0).abs() < 1e-9
        })
        .expect("top-center lattice node exists");
    let uz = displacement.data[top_center * 3 + 2];
    assert!(
        (uz + 0.05).abs() < 1e-3,
        "top-center u_z {uz}, expected ~-0.05"
    );
}
