//! CI-safe end-to-end test on a small committed fixture: a 10x8x6 box
//! at the origin and a radius-3, height-12 cylinder at x = 20, written
//! by OCCT 7.9 (`stepwrite`, AP214).

use brep_core::payload::{PayloadView, build_payload};
use step_import_operator::{StepConfig, import};

const FIXTURE: &str = include_str!("fixtures/box_cylinder.step");

#[test]
fn classifies_box_and_cylinder_exactly() {
    let model = import(FIXTURE, &StepConfig::default()).expect("import fixture");
    assert_eq!(model.solids.len(), 2);
    assert_eq!(model.instances.len(), 2);

    let payload = build_payload(&model).expect("payload");
    let view = PayloadView::new(&payload).unwrap();

    // Analytic ground truth in the fixture's millimetres: box
    // [-5,5]x[-4,4]x[-3,3], cylinder (x-20)^2 + y^2 <= 9, z in [-3, 9].
    // The imported model is in metres, so sample points convert at the
    // classify call.
    let truth = |p: [f64; 3]| -> Option<bool> {
        let in_box = p[0].abs() - 5.0 < 0.0 && p[1].abs() - 4.0 < 0.0 && p[2].abs() - 3.0 < 0.0;
        let box_d = (p[0].abs() - 5.0)
            .max(p[1].abs() - 4.0)
            .max(p[2].abs() - 3.0);
        let rho = ((p[0] - 20.0) * (p[0] - 20.0) + p[1] * p[1]).sqrt();
        let cyl_d = (rho - 3.0).max((p[2] - 9.0).max(-3.0 - p[2]));
        let d = box_d.min(cyl_d);
        if d.abs() < 1e-6 {
            return None;
        }
        let _ = in_box;
        Some(d < 0.0)
    };

    let n = 40;
    let mut tested = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let p = [
                    -8.0 + 34.0 * i as f64 / (n - 1) as f64,
                    -6.0 + 12.0 * j as f64 / (n - 1) as f64,
                    -5.0 + 16.0 * k as f64 / (n - 1) as f64,
                ];
                let Some(expect) = truth(p) else { continue };
                tested += 1;
                let p_m = p.map(|v| v * 1e-3);
                assert_eq!(view.is_inside(p_m), expect, "misclassified {p:?}");
            }
        }
    }
    assert!(tested > 50_000, "only {tested} points tested");
}

#[test]
fn wasm_template_patches_and_validates() {
    let model = import(FIXTURE, &StepConfig::default()).unwrap();
    let payload = build_payload(&model).unwrap();
    let wasm = step_import_operator::patch_template(&payload, false).expect("patch template");
    // The patched module keeps the Model ABI exports and drops the
    // patch-slot helper; an unstyled import also drops the typed-channel
    // exports.
    let module = walrus::Module::from_buffer(&wasm).expect("emitted wasm parses");
    let names: Vec<&str> = module.exports.iter().map(|e| e.name.as_str()).collect();
    for required in ["sample", "get_bounds", "get_dimensions", "memory"] {
        assert!(names.contains(&required), "missing export {required}");
    }
    for dropped in ["brep_payload_slot", "get_sample_format", "sample_channels"] {
        assert!(!names.contains(&dropped), "export {dropped} must be dropped");
    }

    // A colored import keeps the channel exports.
    let wasm = step_import_operator::patch_template(&payload, true).unwrap();
    let module = walrus::Module::from_buffer(&wasm).unwrap();
    let names: Vec<&str> = module.exports.iter().map(|e| e.name.as_str()).collect();
    for required in ["get_sample_format", "sample_channels"] {
        assert!(names.contains(&required), "missing export {required}");
    }
}

#[test]
fn shell_based_surface_model_reached_over_srr() {
    // Rewrite the box into the Onshape hybrid-export topology: the
    // MANIFOLD_SOLID_BREP becomes a SHELL_BASED_SURFACE_MODEL around the
    // same closed shell, parked in a separate representation that hangs
    // off the product's rep via SHAPE_REPRESENTATION_RELATIONSHIP.
    let from_rep = "#36 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#11,#37),#367);";
    let from_body = "#37 = MANIFOLD_SOLID_BREP('',#38);";
    assert!(FIXTURE.contains(from_rep) && FIXTURE.contains(from_body));
    let text = FIXTURE
        .replace(from_rep, "#36 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#11),#367);")
        .replace(
            from_body,
            "#37 = SHELL_BASED_SURFACE_MODEL('',(#38));\n\
             #501 = MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',(#37),#367);\n\
             #502 = SHAPE_REPRESENTATION_RELATIONSHIP('','',#36,#501);",
        );

    let model = import(&text, &StepConfig::default()).expect("import SBSM variant");
    assert_eq!(model.solids.len(), 2);
    assert_eq!(model.instances.len(), 2);

    // The composite box classifies identically to the manifold solid.
    let payload = build_payload(&model).expect("payload");
    let view = PayloadView::new(&payload).unwrap();
    for (p_mm, expect) in [
        ([0.0, 0.0, 0.0], true),
        ([4.9, 3.9, 2.9], true),
        ([5.1, 0.0, 0.0], false),
        ([0.0, 0.0, 3.1], false),
        ([20.0, 0.0, 3.0], true), // cylinder untouched
        ([20.0, 3.1, 3.0], false),
    ] {
        let p = p_mm.map(|v| v * 1e-3);
        assert_eq!(view.is_inside(p), expect, "at {p_mm:?} mm");
    }
}

const COLORED_FIXTURE: &str = include_str!("fixtures/colored_box_cylinder.step");

#[test]
fn styled_items_color_bodies_and_faces() {
    // The fixture styles the box body red, overrides its x = -5 face
    // green, and styles the cylinder body blue (via the pre-defined
    // colour name).
    let model = import(COLORED_FIXTURE, &StepConfig::default()).expect("import");

    let mut seen = std::collections::HashSet::new();
    for solid in &model.solids {
        for face in &solid.faces {
            if let Some(c) = face.color {
                seen.insert(c.map(|v| (v * 255.0) as u8));
            }
        }
    }
    assert_eq!(
        seen,
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]].into_iter().collect(),
        "expected exactly red/green/blue faces, got {seen:?}"
    );

    // Through the payload: nearest-face color at points just off each
    // surface (fixture is in millimetres, the model in metres).
    let payload = build_payload(&model).unwrap();
    let view = brep_core::payload::PayloadView::new(&payload).unwrap();
    const MM: f64 = 1e-3;
    for (p_mm, expect) in [
        ([-5.1, 0.0, 0.0], [0.0, 1.0, 0.0]), // green -x face override
        ([5.1, 0.0, 0.0], [1.0, 0.0, 0.0]),  // red body elsewhere
        ([0.0, 4.1, 0.0], [1.0, 0.0, 0.0]),
        ([23.1, 0.0, 3.0], [0.0, 0.0, 1.0]), // blue cylinder wall
        ([20.0, 0.0, 9.1], [0.0, 0.0, 1.0]), // blue cylinder cap
    ] {
        let p = p_mm.map(|v| v * MM);
        assert_eq!(view.nearest_color(p), Some(expect), "at {p_mm:?} mm");
    }
}

#[test]
fn unstyled_import_has_no_colors() {
    let model = import(FIXTURE, &StepConfig::default()).unwrap();
    assert!(
        model
            .solids
            .iter()
            .all(|s| s.faces.iter().all(|f| f.color.is_none()))
    );
}
