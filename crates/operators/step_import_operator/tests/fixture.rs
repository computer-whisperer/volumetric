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

    // Analytic ground truth: box [-5,5]x[-4,4]x[-3,3], cylinder
    // (x-20)^2 + y^2 <= 9, z in [-3, 9].
    let truth = |p: [f64; 3]| -> Option<bool> {
        let in_box =
            p[0].abs() - 5.0 < 0.0 && p[1].abs() - 4.0 < 0.0 && p[2].abs() - 3.0 < 0.0;
        let box_d = (p[0].abs() - 5.0).max(p[1].abs() - 4.0).max(p[2].abs() - 3.0);
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
                assert_eq!(view.is_inside(p), expect, "misclassified {p:?}");
            }
        }
    }
    assert!(tested > 50_000, "only {tested} points tested");
}

#[test]
fn wasm_template_patches_and_validates() {
    let model = import(FIXTURE, &StepConfig::default()).unwrap();
    let payload = build_payload(&model).unwrap();
    let wasm = step_import_operator::patch_template(&payload).expect("patch template");
    // The patched module keeps the Model ABI exports and drops the
    // patch-slot helper.
    let module = walrus::Module::from_buffer(&wasm).expect("emitted wasm parses");
    let names: Vec<&str> = module.exports.iter().map(|e| e.name.as_str()).collect();
    for required in ["sample", "get_bounds", "get_dimensions", "memory"] {
        assert!(names.contains(&required), "missing export {required}");
    }
    assert!(
        !names.contains(&"brep_payload_slot"),
        "patch slot export must be dropped"
    );
}
