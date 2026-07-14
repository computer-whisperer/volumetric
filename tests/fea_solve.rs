//! End-to-end test of the FEA solve pipeline: a full project builds a unit
//! box, grid-meshes it, and compresses it under a rigid sphere positioned to
//! dip into its top face — exercising the mesher, the host sampling imports
//! from two operators, and the solver's contact/force/energy fields.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p simple_sphere_model -p rectangular_prism_operator \
//!     -p translate_operator -p fea_grid_mesh_operator -p fea_solve_operator

#![cfg(feature = "native")]

use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::fea::decode_fea_mesh;

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
fn box_compressed_by_sphere_end_to_end() {
    use ciborium::value::Value;

    // Unit box [0,1]^3, meshed at resolution 8 (h = 0.125), compressed by
    // the unit sphere translated to (0.5, 0.5, 1.9): its lower surface dips
    // 0.1 into the box top at the center.
    let dip = 0.1;
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
                        ("dz", Value::Float(2.0 - dip)),
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
                    ExecutionInput::Inline(Vec::new()), // defaults: E=1, nu=0.3, zmin
                ],
                outputs: vec!["solved".to_string()],
            },
        ],
        exports: vec!["solved".to_string()],
        baked: None,
    };

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let asset = &exports[0];
    assert_eq!(asset.type_hint(), Some(AssetTypeHint::FeaMesh));
    let mesh = decode_fea_mesh(asset.data()).expect("solved output decodes");

    // 8^3 full box.
    assert_eq!(mesh.element_count(), 512);
    let nodes = mesh.node_count();

    let field = |fields: &[volumetric::fea::FeaField], name: &str, components: usize| {
        let field = fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("missing field {name}"));
        assert_eq!(field.components, components, "{name} components");
        field.data.clone()
    };
    let displacement = field(&mesh.node_fields, "displacement", 3);
    let contact_force = field(&mesh.node_fields, "contact_force", 3);
    let energy = field(&mesh.element_fields, "strain_energy_density", 1);
    assert_eq!(displacement.len(), nodes * 3);
    assert_eq!(contact_force.len(), nodes * 3);
    assert_eq!(energy.len(), 512);

    // The node under the sphere's deepest point (top center) is pressed
    // down by the dip depth; the bisected surface is exact to ~1e-9.
    let top_center = (0..nodes)
        .find(|n| {
            let p = mesh.node_position(*n);
            (p[0] - 0.5).abs() < 1e-9 && (p[1] - 0.5).abs() < 1e-9 && (p[2] - 1.0).abs() < 1e-9
        })
        .expect("top-center node exists");
    let uz = displacement[top_center * 3 + 2];
    assert!(
        (uz + dip).abs() < 1e-4,
        "top-center u_z = {uz}, expected ~{}",
        -dip
    );

    // The base stays glued, every contact force presses down, and the sum
    // is a sensible downward total.
    for n in 0..nodes {
        if mesh.node_position(n)[2].abs() < 1e-9 {
            for c in 0..3 {
                assert!(displacement[n * 3 + c].abs() < 1e-12, "base node {n} moved");
            }
        }
        assert!(
            contact_force[n * 3 + 2] <= 1e-9,
            "contact force at node {n} points up: {}",
            contact_force[n * 3 + 2]
        );
    }
    let total_fz: f64 = (0..nodes).map(|n| contact_force[n * 3 + 2]).sum();
    assert!(total_fz < -1e-4, "no net downward force: {total_fz}");

    // Strain energy concentrates under the indenter: the top-center column
    // carries more than a bottom corner, and everything is finite and
    // non-negative.
    assert!(energy.iter().all(|e| e.is_finite() && *e >= -1e-12));
    let element_center = |e: usize| {
        let mut c = [0.0f64; 3];
        for node in mesh.element(e) {
            let p = mesh.node_position(*node as usize);
            for axis in 0..3 {
                c[axis] += p[axis] / 8.0;
            }
        }
        c
    };
    let near = |target: [f64; 3]| {
        (0..mesh.element_count())
            .min_by(|a, b| {
                let d = |e: &usize| {
                    let c = element_center(*e);
                    (0..3).map(|i| (c[i] - target[i]).powi(2)).sum::<f64>()
                };
                d(a).partial_cmp(&d(b)).unwrap()
            })
            .unwrap()
    };
    let under_indenter = near([0.5, 0.5, 0.95]);
    let far_corner = near([0.05, 0.05, 0.05]);
    assert!(
        energy[under_indenter] > 10.0 * energy[far_corner].max(1e-12),
        "energy not concentrated: under={} far={}",
        energy[under_indenter],
        energy[far_corner]
    );
}
