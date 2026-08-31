//! CI-safe end-to-end test for ELLIPSE edge curves: a radius-3 cylinder
//! at the origin truncated by the oblique plane z = 8 - 0.4x (full
//! ellipse edge around the lateral face), and the x <= 0 half of the
//! same body translated to x = 20 (vertex-trimmed half-ellipse arcs).
//! Written by OCCT 7.9 (`stepwrite`, AP214), millimetres.

use brep_core::payload::{PayloadView, build_payload};
use step_import_operator::{StepConfig, import};

const FIXTURE: &str = include_str!("fixtures/oblique_cylinder.step");

#[test]
fn fixture_actually_uses_ellipses() {
    // Guard against a regenerated fixture silently dropping the case.
    let n = FIXTURE.matches("ELLIPSE").count();
    assert!(n >= 2, "fixture has {n} ELLIPSE records");
}

#[test]
fn classifies_oblique_cylinders_exactly() {
    let model = import(FIXTURE, &StepConfig::default()).expect("import fixture");
    assert_eq!(model.solids.len(), 2);

    let payload = build_payload(&model).expect("payload");
    let view = PayloadView::new(&payload).unwrap();

    // Signed distances in the fixture's millimetres; None inside the
    // near-surface band where classification is legitimately ambiguous.
    let plane_n = (1.0f64 + 0.4 * 0.4).sqrt();
    let truth = |p: [f64; 3]| -> Option<bool> {
        let body = |cx: f64| -> f64 {
            let rho = ((p[0] - cx) * (p[0] - cx) + p[1] * p[1]).sqrt();
            (rho - 3.0)
                .max(-p[2])
                .max((0.4 * (p[0] - cx) + p[2] - 8.0) / plane_n)
        };
        let a = body(0.0);
        let b = body(20.0).max(p[0] - 20.0);
        let d = a.min(b);
        if d.abs() < 1e-6 {
            return None;
        }
        Some(d < 0.0)
    };

    let n = 40;
    let mut tested = 0;
    let mut inside = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let p = [
                    -5.0 + 30.0 * i as f64 / (n - 1) as f64,
                    -5.0 + 10.0 * j as f64 / (n - 1) as f64,
                    -2.0 + 16.0 * k as f64 / (n - 1) as f64,
                ];
                let Some(expect) = truth(p) else { continue };
                tested += 1;
                inside += usize::from(expect);
                let p_m = p.map(|v| v * 1e-3);
                assert_eq!(view.is_inside(p_m), expect, "misclassified {p:?}");
            }
        }
    }
    assert!(tested > 50_000, "only {tested} points tested");
    assert!(inside > 2_000, "only {inside} interior points tested");
}
