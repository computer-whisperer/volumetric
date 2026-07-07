//! End-to-end test of the inverse FEA pipeline: a full project builds a unit
//! box, grid-meshes it, and backs out per-element stiffness so the contact
//! force distribution under a rigid sphere matches a 2D step target map —
//! exercising the mesher, both host-sampled model inputs (3D rigid body, 2D
//! target map), and the inverse loop's stiffness/target output fields.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p simple_sphere_model -p rectangular_prism_operator \
//!     -p translate_operator -p fea_grid_mesh_operator -p fea_inverse_operator

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

/// A hand-assembled 2D model over [0,1]^2: pressure 1 where x < 0.5, else 3.
/// The step sits exactly on the meshed footprint's midline.
fn step_target_map() -> Vec<u8> {
    wat::parse_str(
        r#"(module
            (memory (export "memory") 1)
            (func (export "get_dimensions") (result i32) (i32.const 2))
            (func (export "get_io_ptr") (result i32) (i32.const 1024))
            (func (export "get_bounds") (param $out i32)
                (f64.store (local.get $out) (f64.const 0))
                (f64.store offset=8 (local.get $out) (f64.const 1))
                (f64.store offset=16 (local.get $out) (f64.const 0))
                (f64.store offset=24 (local.get $out) (f64.const 1)))
            (func (export "sample") (param $pos i32) (result f32)
                (select (f32.const 1) (f32.const 3)
                    (f64.lt (f64.load (local.get $pos)) (f64.const 0.5))))
        )"#,
    )
    .expect("target map WAT assembles")
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
fn sphere_press_matches_a_step_target_end_to_end() {
    use ciborium::value::Value;

    // Unit box [0,1]^3 meshed at resolution 8, pressed by the unit sphere
    // dipping 0.1 into the top center — the forward-solve e2e scenario, now
    // asked to make the right half of the contact patch carry three times
    // the pressure of the left half.
    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::model("sphere".to_string(), wasm_artifact("simple_sphere_model")),
            ImportedAsset::model("target".to_string(), step_target_map()),
            ImportedAsset::operator(
                "prism".to_string(),
                wasm_artifact("rectangular_prism_operator"),
            ),
            ImportedAsset::operator("translate".to_string(), wasm_artifact("translate_operator")),
            ImportedAsset::operator(
                "mesher".to_string(),
                wasm_artifact("fea_grid_mesh_operator"),
            ),
            ImportedAsset::operator("inverse".to_string(), wasm_artifact("fea_inverse_operator")),
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
                operator_id: "inverse".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("mesh".to_string()),
                    ExecutionInput::AssetRef("butt".to_string()),
                    ExecutionInput::AssetRef("target".to_string()),
                    ExecutionInput::Inline(Vec::new()), // all defaults
                ],
                outputs: vec!["designed".to_string()],
            },
        ],
        exports: vec!["designed".to_string()],
    };

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let asset = &exports[0];
    assert_eq!(asset.type_hint(), Some(AssetTypeHint::FeaMesh));
    let mesh = decode_fea_mesh(asset.data()).expect("designed output decodes");

    let nodes = mesh.node_count();
    let field = |fields: &[volumetric::fea::FeaField], name: &str, components: usize| {
        let field = fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("missing field {name}"));
        assert_eq!(field.components, components, "{name} components");
        field.data.clone()
    };
    let stiffness = field(&mesh.element_fields, "stiffness_scale", 1);
    let target_force = field(&mesh.node_fields, "target_force", 1);
    let contact_force = field(&mesh.node_fields, "contact_force", 3);
    field(&mesh.node_fields, "displacement", 3);
    field(&mesh.element_fields, "strain_energy_density", 1);
    assert_eq!(stiffness.len(), 512);
    assert_eq!(target_force.len(), nodes);

    // The operator only outputs on convergence, so the achieved forces obey
    // the demanded 1:3 split across x = 0.5 (up to the 0.02 distribution
    // tolerance).
    let mut achieved = [0.0f64; 2]; // [x < 0.5, x >= 0.5]
    let mut demanded = [0.0f64; 2];
    for n in 0..nodes {
        let side = (mesh.node_position(n)[0] >= 0.5) as usize;
        achieved[side] += -contact_force[n * 3 + 2];
        demanded[side] += target_force[n];
    }
    let total = achieved[0] + achieved[1];
    assert!(total > 0.0, "no contact force at all");
    let achieved_high = achieved[1] / total;
    let demanded_high = demanded[1] / (demanded[0] + demanded[1]);
    assert!(
        (achieved_high - demanded_high).abs() < 0.05,
        "high-side share: achieved {achieved_high:.3}, demanded {demanded_high:.3}"
    );
    assert!(
        achieved_high > 0.65,
        "expected ~3/4 of the force on the high side, got {achieved_high:.3}"
    );
    // target_force is scaled to the achieved total.
    let demanded_total = demanded[0] + demanded[1];
    assert!(
        (demanded_total - total).abs() < 1e-6 * total,
        "target_force total {demanded_total} vs achieved {total}"
    );

    // The stiffness follows the demand: under the contact patch, columns on
    // the low-pressure side end up softer. All scales stay in (0, 1].
    assert!(stiffness.iter().all(|s| *s > 0.0 && *s <= 1.0));
    let h = 0.125;
    let mut side_scale = [0.0f64; 2];
    let mut side_count = [0usize; 2];
    for e in 0..mesh.element_count() {
        let base = mesh.node_position(mesh.element(e)[0] as usize);
        let (cx, cy) = (base[0] + h / 2.0, base[1] + h / 2.0);
        // Columns within the contact patch footprint (radius ~0.44), clear
        // of the step line.
        let r = ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt();
        if r > 0.35 || (cx - 0.5).abs() < h {
            continue;
        }
        let side = (cx > 0.5) as usize;
        side_scale[side] += stiffness[e];
        side_count[side] += 1;
    }
    let low = side_scale[0] / side_count[0] as f64;
    let high = side_scale[1] / side_count[1] as f64;
    assert!(
        high > 1.5 * low,
        "stiffness contrast missing under the patch: low side {low:.3}, high side {high:.3}"
    );
}

#[test]
fn strut_lattice_inverse_matches_a_step_target_end_to_end() {
    use ciborium::value::Value;

    // The same scenario on an explicit cubic strut lattice: unit box filled
    // by strut_pattern_operator, sphere dipping 0.1 into the top, right
    // half demanded three times the pressure — the inverse loop driving
    // per-strut mechanical properties.
    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::model("sphere".to_string(), wasm_artifact("simple_sphere_model")),
            ImportedAsset::model("target".to_string(), step_target_map()),
            ImportedAsset::operator(
                "prism".to_string(),
                wasm_artifact("rectangular_prism_operator"),
            ),
            ImportedAsset::operator("translate".to_string(), wasm_artifact("translate_operator")),
            ImportedAsset::operator(
                "pattern".to_string(),
                wasm_artifact("strut_pattern_operator"),
            ),
            ImportedAsset::operator("inverse".to_string(), wasm_artifact("fea_inverse_operator")),
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
                operator_id: "pattern".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("box".to_string()),
                    ExecutionInput::Inline(cbor_map(&[
                        ("family", Value::Text("cubic".into())),
                        ("cell_size", Value::Float(0.125)),
                        ("radius", Value::Float(0.015)),
                    ])),
                ],
                outputs: vec!["lattice".to_string()],
            },
            ExecutionStep {
                operator_id: "inverse".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("lattice".to_string()),
                    ExecutionInput::AssetRef("butt".to_string()),
                    ExecutionInput::AssetRef("target".to_string()),
                    ExecutionInput::Inline(Vec::new()), // all defaults
                ],
                outputs: vec!["designed".to_string()],
            },
        ],
        exports: vec!["designed".to_string()],
    };

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let asset = &exports[0];
    assert_eq!(asset.type_hint(), Some(AssetTypeHint::FeaMesh));
    let mesh = decode_fea_mesh(asset.data()).expect("designed output decodes");
    assert_eq!(
        mesh.element_kind,
        volumetric_abi::fea::FeaElementKind::Bar2
    );

    let field = |fields: &[volumetric::fea::FeaField], name: &str, components: usize| {
        let field = fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("missing field {name}"));
        assert_eq!(field.components, components, "{name} components");
        field.data.clone()
    };
    let stiffness = field(&mesh.element_fields, "stiffness_scale", 1);
    let target_force = field(&mesh.node_fields, "target_force", 1);
    let contact_force = field(&mesh.node_fields, "contact_force", 3);
    field(&mesh.node_fields, "displacement", 3);
    field(&mesh.node_fields, "rotation", 3);
    assert_eq!(stiffness.len(), mesh.element_count());
    assert!(stiffness.iter().all(|s| *s > 0.0 && *s <= 1.0));

    // Convergence means the achieved forces follow the demanded 1:3 step.
    let mut achieved = [0.0f64; 2];
    let mut demanded = [0.0f64; 2];
    for n in 0..mesh.node_count() {
        let side = (mesh.node_position(n)[0] >= 0.5) as usize;
        achieved[side] += -contact_force[n * 3 + 2];
        demanded[side] += target_force[n];
    }
    let total = achieved[0] + achieved[1];
    assert!(total > 0.0, "no contact force at all");
    let achieved_high = achieved[1] / total;
    let demanded_high = demanded[1] / (demanded[0] + demanded[1]);
    assert!(
        (achieved_high - demanded_high).abs() < 0.05,
        "high-side share: achieved {achieved_high:.3}, demanded {demanded_high:.3}"
    );
    assert!(
        achieved_high > 0.6,
        "expected most force on the high side, got {achieved_high:.3}"
    );

    // Struts under the low-pressure half of the patch end up softer.
    let mut side_scale = [0.0f64; 2];
    let mut side_count = [0usize; 2];
    for e in 0..mesh.element_count() {
        let pair = mesh.element(e);
        let a = mesh.node_position(pair[0] as usize);
        let b = mesh.node_position(pair[1] as usize);
        let (cx, cy) = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]));
        let r = ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt();
        if r > 0.35 || (cx - 0.5).abs() < 0.125 {
            continue;
        }
        let side = (cx > 0.5) as usize;
        side_scale[side] += stiffness[e];
        side_count[side] += 1;
    }
    let low = side_scale[0] / side_count[0] as f64;
    let high = side_scale[1] / side_count[1] as f64;
    assert!(
        high > 1.5 * low,
        "stiffness contrast missing under the patch: low side {low:.3}, high side {high:.3}"
    );
}
