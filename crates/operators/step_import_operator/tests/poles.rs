//! CI-safe tests of trim loops through the poles of analytic surfaces:
//! sphere poles and the cone apex, where the azimuth is free and which
//! way the loop turns around the pole is settled by the STEP sense
//! flags. Eight small OCCT 7.9 (`stepwrite`, AP214) shapes in
//! `fixtures/poles/`, in millimetres, each classified on a dense grid
//! against its analytic definition:
//!
//! - `sphere`: `psphere 10` — a full sphere, bounded by nothing but a
//!   VERTEX_LOOP at a pole.
//! - `hemi`: the sphere minus the half-space `x > 0` — one loop of two
//!   pole-to-pole meridians; the two halves have the same loop and only
//!   the sense flags tell them apart.
//! - `sector`: the sphere minus the quadrant `x > 0, y > 0` — the loop
//!   turns 270° around each pole, more than a half turn.
//! - `cap`: the sphere above `z = 5` — the seam meridian runs up to the
//!   pole and back down, a full turn around the pole.
//! - `band`: the sphere between `z = ±5` — no pole on any loop.
//! - `cone`: `pcone 10 0 20` — a pointed cone, full turn at the apex.
//! - `halfcone`: the cone minus `x > 0` — a half turn at the apex.
//! - `ring`: the cone below `z = 10` — no pole on any loop.

use brep_core::payload::{PayloadView, build_payload};
use step_import_operator::{StepConfig, import};

const FIXTURES: [(&str, &str); 8] = [
    ("sphere", include_str!("fixtures/poles/sphere.step")),
    ("hemi", include_str!("fixtures/poles/hemi.step")),
    ("sector", include_str!("fixtures/poles/sector.step")),
    ("cap", include_str!("fixtures/poles/cap.step")),
    ("band", include_str!("fixtures/poles/band.step")),
    ("cone", include_str!("fixtures/poles/cone.step")),
    ("halfcone", include_str!("fixtures/poles/halfcone.step")),
    ("ring", include_str!("fixtures/poles/ring.step")),
];

/// Analytic ground truth in millimetres; `None` within 0.1 mm of a
/// surface, where the exact verdict is not the point.
fn truth(name: &str, p: [f64; 3]) -> Option<bool> {
    let [x, y, z] = p;
    let r = (x * x + y * y + z * z).sqrt();
    let rho = (x * x + y * y).sqrt();
    let near = |d: f64| d.abs() < 0.1;
    let sphere = r < 10.0;
    match name {
        "sphere" => (!near(r - 10.0)).then_some(sphere),
        "hemi" => (!near(r - 10.0) && !near(x)).then_some(sphere && x < 0.0),
        "sector" => {
            (!near(r - 10.0) && !near(x) && !near(y)).then_some(sphere && !(x > 0.0 && y > 0.0))
        }
        "cap" => (!near(r - 10.0) && !near(z - 5.0)).then_some(sphere && z > 5.0),
        "band" => {
            (!near(r - 10.0) && !near(z - 5.0) && !near(z + 5.0)).then_some(sphere && z.abs() < 5.0)
        }
        "cone" | "halfcone" | "ring" => {
            let rc = 10.0 * (1.0 - z / 20.0);
            let top = if name == "ring" { 10.0 } else { 20.0 };
            if near(z) || near(z - top) || near(rho - rc) || (name == "halfcone" && near(x)) {
                return None;
            }
            Some(z > 0.0 && z < top && rho < rc && (name != "halfcone" || x < 0.0))
        }
        _ => unreachable!(),
    }
}

#[test]
fn loops_through_poles_classify_exactly() {
    for (name, text) in FIXTURES {
        let model = import(text, &StepConfig::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let payload = build_payload(&model).unwrap_or_else(|e| panic!("{name}: {e}"));
        let view = PayloadView::new(&payload).unwrap();
        // The cones stand on z = 0 with the apex at z = 20; the spheres
        // sit at the origin.
        let z_off = if name.contains("cone") || name == "ring" {
            10.0
        } else {
            0.0
        };
        let n = 40;
        let (mut tested, mut wrong) = (0, 0);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let f = |t: usize| -12.0 + 24.0 * t as f64 / (n - 1) as f64;
                    let p = [f(i), f(j), f(k) + z_off];
                    let Some(expect) = truth(name, p) else {
                        continue;
                    };
                    tested += 1;
                    if view.is_inside(p.map(|v| v * 1e-3)) != expect {
                        wrong += 1;
                    }
                }
            }
        }
        assert!(tested > 50_000, "{name}: only {tested} points tested");
        assert_eq!(wrong, 0, "{name}: {wrong} of {tested} points misclassified");
    }
}
