//! End-to-end test of the complete strut workflow (a→d): a unit box fills
//! with an explicit strut lattice, the inverse loop designs per-strut
//! stiffness against a step pressure target, and the strut model operator
//! realizes the designed lattice as sampleable capsule geometry — thick
//! struts under the high-pressure half, thin under the low.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p simple_sphere_model -p rectangular_prism_operator \
//!     -p translate_operator -p strut_pattern_operator \
//!     -p fea_inverse_operator -p strut_model_operator

#![cfg(feature = "native")]

use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project};

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
fn designed_lattice_realizes_as_geometry_end_to_end() {
    use ciborium::value::Value;

    let cell = 0.125;
    let base_radius = 0.015;
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
            ImportedAsset::operator("realize".to_string(), wasm_artifact("strut_model_operator")),
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
                        ("cell_size", Value::Float(cell)),
                        ("radius", Value::Float(base_radius)),
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
                    ExecutionInput::Inline(Vec::new()),
                ],
                outputs: vec!["designed".to_string()],
            },
            ExecutionStep {
                operator_id: "realize".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("designed".to_string()),
                    ExecutionInput::Inline(Vec::new()), // rest shape, n = 4
                ],
                outputs: vec!["model".to_string()],
            },
        ],
        exports: vec!["designed".to_string(), "model".to_string()],
        baked: None,
    };

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let designed = volumetric_abi::fea::decode_fea_mesh(
        exports
            .iter()
            .find(|a| a.id() == "designed")
            .expect("designed export")
            .data(),
    )
    .expect("designed mesh decodes");
    let model = exports
        .iter()
        .find(|a| a.id() == "model")
        .expect("model export");

    let mut executor = create_model_executor(model.data()).expect("realized model executes");
    assert_eq!(executor.dimensions().expect("dimensions"), 3);

    // Bounds: the box inflated by at most the base radius.
    let bounds = executor.get_bounds_nd().expect("bounds");
    for axis in 0..3 {
        assert!(
            bounds.min(axis) >= -(base_radius + 1e-6) && bounds.min(axis) <= 1e-6,
            "axis {axis} min {}",
            bounds.min(axis)
        );
        assert!(
            bounds.max(axis) <= 1.0 + base_radius + 1e-6 && bounds.max(axis) >= 1.0 - 1e-6,
            "axis {axis} max {}",
            bounds.max(axis)
        );
    }

    let mut inside = |p: [f64; 3]| executor.sample_nd(&p).expect("sample") > 0.5;

    // Strut axes are the grid lines: points on them are inside, cell body
    // centers are far from every strut (half a cell diagonal ~ 0.088 >>
    // any realized radius).
    assert!(inside([0.5, 0.5, 0.03]), "on a vertical strut axis");
    assert!(inside([0.5 + cell / 2.0, 0.5, 0.25]), "on a lateral strut");
    assert!(
        !inside([0.5 + cell / 2.0, 0.5 + cell / 2.0, 0.25 + cell / 2.0]),
        "cell body center must be empty"
    );
    assert!(!inside([1.3, 0.5, 0.5]), "outside the box");

    // The scale -> radius mapping shows in the geometry exactly: for a
    // vertical strut on each side of the step, measure the realized radius
    // by bisecting the occupancy boundary at the strut's midpoint and
    // compare against base_radius * scale^(1/4) from the designed mesh.
    let scale_field = designed
        .element_fields
        .iter()
        .find(|f| f.name == "stiffness_scale")
        .expect("designed mesh carries stiffness_scale");
    // The vertical strut on the lattice line (x0, 0.5) whose midpoint is
    // nearest z = 0.5.
    let vertical_strut_at = |x0: f64| -> usize {
        (0..designed.element_count())
            .filter(|&e| {
                let pair = designed.element(e);
                let a = designed.node_position(pair[0] as usize);
                let b = designed.node_position(pair[1] as usize);
                (a[0] - x0).abs() < 1e-9
                    && (b[0] - x0).abs() < 1e-9
                    && (a[1] - 0.5).abs() < 1e-9
                    && (b[1] - 0.5).abs() < 1e-9
                    && (a[2] - b[2]).abs() > 1e-9
            })
            .min_by(|&p, &q| {
                let mid = |e: usize| {
                    let pair = designed.element(e);
                    let a = designed.node_position(pair[0] as usize);
                    let b = designed.node_position(pair[1] as usize);
                    (0.5 * (a[2] + b[2]) - 0.5).abs()
                };
                mid(p).total_cmp(&mid(q))
            })
            .expect("vertical strut exists on the lattice line")
    };
    let mut measured_radius = |e: usize| -> f64 {
        let pair = designed.element(e);
        let a = designed.node_position(pair[0] as usize);
        let b = designed.node_position(pair[1] as usize);
        let mid_z = 0.5 * (a[2] + b[2]);
        let (mut lo, mut hi) = (0.0f64, 3.0 * base_radius);
        for _ in 0..40 {
            let r = 0.5 * (lo + hi);
            if inside([a[0] + r, 0.5, mid_z]) {
                lo = r;
            } else {
                hi = r;
            }
        }
        0.5 * (lo + hi)
    };
    for x0 in [0.375, 0.625] {
        let e = vertical_strut_at(x0);
        let expected = base_radius * scale_field.data[e].powf(0.25);
        let measured = measured_radius(e);
        assert!(
            (measured - expected).abs() < 5e-4,
            "strut at x = {x0}: measured radius {measured:.5}, expected \
             {expected:.5} (scale {})",
            scale_field.data[e]
        );
    }

    // And the design itself has sided contrast: the soft side's scale is
    // meaningfully below the stiff side's.
    let (soft, stiff) = (
        scale_field.data[vertical_strut_at(0.375)],
        scale_field.data[vertical_strut_at(0.625)],
    );
    assert!(
        stiff > 1.2 * soft,
        "expected stiffness contrast across the step: soft {soft}, stiff {stiff}"
    );
}
