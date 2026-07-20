//! End-to-end classification: build IR solids, serialize, classify a
//! dense grid through `PayloadView`, and compare with analytic ground
//! truth. Grid points are deliberately axis-aligned with the geometry —
//! the adversarial case the skewed parity ray exists for. Points within
//! a thin analytic shell of the boundary are skipped (either answer is
//! legitimate there); everywhere else classification must be exact.

use brep_core::ir::{BRepModel, Face, Instance, NurbsSurface, Solid, Surface};
use brep_core::math::{Affine, Frame};
use brep_core::payload::{PayloadView, build_payload};
use core::f64::consts::{FRAC_PI_2, PI, TAU};

/// Classify an n³ grid over `bbox` (inflated slightly so outside points
/// are covered); `truth` returns None inside the boundary-ambiguity
/// shell. Panics listing the first mismatches.
fn check_grid(
    model: &BRepModel,
    n: usize,
    bbox: [f64; 6],
    truth: impl Fn([f64; 3]) -> Option<bool>,
) {
    let payload = build_payload(model).expect("build_payload");
    let view = PayloadView::new(&payload).expect("PayloadView");
    let mut mismatches = Vec::new();
    let mut tested = 0usize;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let f = |idx: usize, a: usize| {
                    let lo = bbox[a * 2];
                    let hi = bbox[a * 2 + 1];
                    let pad = (hi - lo) * 0.1;
                    lo - pad + (hi - lo + 2.0 * pad) * idx as f64 / (n - 1) as f64
                };
                let p = [f(i, 0), f(j, 1), f(k, 2)];
                let Some(expect) = truth(p) else { continue };
                tested += 1;
                let got = view.is_inside(p);
                if got != expect && mismatches.len() < 10 {
                    mismatches.push((p, expect, got));
                }
            }
        }
    }
    assert!(tested > n * n * n / 4, "shell excluded too many points");
    assert!(
        mismatches.is_empty(),
        "{} of {tested} misclassified, first: {:?}",
        mismatches.len(),
        mismatches
    );
}

fn shell(d: f64, margin: f64) -> Option<bool> {
    if d.abs() < margin {
        None
    } else {
        Some(d < 0.0)
    }
}

/// Square trim loop of half-extent `h` in UV.
fn square_trim(h: f64) -> Vec<Vec<[f64; 2]>> {
    vec![vec![[-h, -h], [h, -h], [h, h], [-h, h]]]
}

/// A ±1 cube from six plane faces with mixed frame orientations.
fn cube_solid() -> Solid {
    let mut faces = Vec::new();
    for axis in 0..3 {
        for side in [-1.0f64, 1.0] {
            let mut origin = [0.0; 3];
            origin[axis] = side;
            let mut z = [0.0; 3];
            z[axis] = side;
            let x_ref = if axis == 0 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            faces.push(Face {
                surface: Surface::Plane {
                    frame: Frame::from_axis_ref(origin, z, x_ref),
                },
                trims: square_trim(1.0),
            });
        }
    }
    Solid { faces }
}

fn single(solid: Solid) -> BRepModel {
    BRepModel {
        solids: vec![solid],
        instances: vec![Instance {
            solid: 0,
            local_to_world: Affine::IDENTITY,
            label: String::new(),
        }],
    }
}

#[test]
fn cube_of_planes() {
    let model = single(cube_solid());
    check_grid(&model, 24, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], |p| {
        let d = p.iter().fold(f64::NEG_INFINITY, |m, &c| m.max(c.abs())) - 1.0;
        shell(d, 1e-6)
    });
}

#[test]
fn cylinder_solid_with_caps() {
    // Radius 1, z in [-1, 1]. Side face trims the full period with an
    // unwrapped seam-spanning rectangle.
    let side = Face {
        surface: Surface::Cylinder {
            frame: Frame::IDENTITY,
            radius: 1.0,
        },
        trims: vec![vec![[0.0, -1.0], [TAU, -1.0], [TAU, 1.0], [0.0, 1.0]]],
    };
    let circle: Vec<[f64; 2]> = (0..64)
        .map(|i| {
            let a = TAU * i as f64 / 64.0;
            [a.cos(), a.sin()]
        })
        .collect();
    let caps = [1.0f64, -1.0].map(|side_z| Face {
        surface: Surface::Plane {
            frame: Frame::from_axis_ref([0.0, 0.0, side_z], [0.0, 0.0, side_z], [1.0, 0.0, 0.0]),
        },
        trims: vec![circle.clone()],
    });
    let [cap_a, cap_b] = caps;
    let model = single(Solid {
        faces: vec![side, cap_a, cap_b],
    });
    check_grid(&model, 24, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], |p| {
        let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
        // The polygonal cap trim underestimates the circle by up to the
        // sagitta; use a shell wide enough to cover it.
        let d = (rho - 1.0).max(p[2].abs() - 1.0);
        shell(d, 0.006)
    });
}

#[test]
fn sphere_full_face() {
    let face = Face {
        surface: Surface::Sphere {
            frame: Frame::IDENTITY,
            radius: 1.0,
        },
        trims: vec![vec![
            [-PI, -FRAC_PI_2],
            [PI, -FRAC_PI_2],
            [PI, FRAC_PI_2],
            [-PI, FRAC_PI_2],
        ]],
    };
    let model = single(Solid { faces: vec![face] });
    check_grid(&model, 24, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], |p| {
        let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 1.0;
        shell(d, 1e-6)
    });
}

#[test]
fn torus_full_face() {
    let face = Face {
        surface: Surface::Torus {
            frame: Frame::IDENTITY,
            major: 2.0,
            minor: 0.5,
        },
        trims: vec![vec![[0.0, 0.0], [TAU, 0.0], [TAU, TAU], [0.0, TAU]]],
    };
    let model = single(Solid { faces: vec![face] });
    check_grid(&model, 24, [-2.5, 2.5, -2.5, 2.5, -0.5, 0.5], |p| {
        let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let d = ((rho - 2.0) * (rho - 2.0) + p[2] * p[2]).sqrt() - 0.5;
        shell(d, 1e-6)
    });
}

#[test]
fn cube_with_drilled_hole() {
    // ±1 cube with a radius-0.5 z-through-hole: cap faces carry a hole
    // loop, and the hole wall is a cylinder face whose material side is
    // the outside of the cylinder.
    let circle: Vec<[f64; 2]> = (0..64)
        .map(|i| {
            let a = TAU * i as f64 / 64.0;
            [0.5 * a.cos(), 0.5 * a.sin()]
        })
        .collect();
    let mut faces = Vec::new();
    for axis in 0..3 {
        for side in [-1.0f64, 1.0] {
            let mut origin = [0.0; 3];
            origin[axis] = side;
            let mut z = [0.0; 3];
            z[axis] = side;
            let x_ref = if axis == 0 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let mut trims = square_trim(1.0);
            if axis == 2 {
                trims.push(circle.clone());
            }
            faces.push(Face {
                surface: Surface::Plane {
                    frame: Frame::from_axis_ref(origin, z, x_ref),
                },
                trims,
            });
        }
    }
    faces.push(Face {
        surface: Surface::Cylinder {
            frame: Frame::IDENTITY,
            radius: 0.5,
        },
        trims: vec![vec![[0.0, -1.0], [TAU, -1.0], [TAU, 1.0], [0.0, 1.0]]],
    });
    let model = single(Solid { faces });
    check_grid(&model, 24, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], |p| {
        let cube = p.iter().fold(f64::NEG_INFINITY, |m, &c| m.max(c.abs())) - 1.0;
        let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let hole = 0.5 - rho;
        let d = cube.max(hole);
        shell(d, 0.004)
    });
}

#[test]
fn instanced_cubes_union() {
    // One cube solid, two instances: identity, and rotated 90° about z
    // then shifted +3x. Union semantics with correct inverse transforms.
    let rot_shift = Affine([0.0, -1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let model = BRepModel {
        solids: vec![cube_solid()],
        instances: vec![
            Instance {
                solid: 0,
                local_to_world: Affine::IDENTITY,
                label: "a".into(),
            },
            Instance {
                solid: 0,
                local_to_world: rot_shift,
                label: "b".into(),
            },
        ],
    };
    check_grid(&model, 24, [-1.0, 4.0, -1.0, 1.0, -1.0, 1.0], |p| {
        let da = p.iter().fold(f64::NEG_INFINITY, |m, &c| m.max(c.abs())) - 1.0;
        let q = [p[0] - 3.0, p[1], p[2]];
        let db = q.iter().fold(f64::NEG_INFINITY, |m, &c| m.max(c.abs())) - 1.0;
        shell(da.min(db), 1e-6)
    });
}

#[test]
fn cube_with_nurbs_top() {
    // The +z face replaced by an equivalent bilinear NURBS patch:
    // exercises the NURBS payload path (seed boxes, Newton) end to end.
    let mut solid = cube_solid();
    solid.faces.retain(|f| match &f.surface {
        Surface::Plane { frame } => frame.z[2] < 0.5,
        _ => true,
    });
    solid.faces.push(Face {
        surface: Surface::Nurbs(NurbsSurface {
            degree_u: 1,
            degree_v: 1,
            nctrl_u: 2,
            nctrl_v: 2,
            knots_u: vec![0.0, 0.0, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            ctrl: vec![
                [-1.0, -1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
        }),
        trims: vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
    });
    let model = single(solid);
    check_grid(&model, 24, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], |p| {
        let d = p.iter().fold(f64::NEG_INFINITY, |m, &c| m.max(c.abs())) - 1.0;
        shell(d, 1e-6)
    });
}

#[test]
fn bounds_and_validation() {
    let model = single(cube_solid());
    let payload = build_payload(&model).unwrap();
    let view = PayloadView::new(&payload).unwrap();
    let b = view.bounds();
    for (got, expect) in b.iter().zip([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]) {
        assert!((got - expect).abs() < 1e-3, "bounds {b:?}");
    }
    // Corrupt magic rejected.
    let mut bad = payload.clone();
    bad[0] ^= 0xff;
    assert!(PayloadView::new(&bad).is_err());
    // Instance referencing a missing solid rejected.
    let mut broken = single(cube_solid());
    broken.instances[0].solid = 7;
    assert!(build_payload(&broken).is_err());
    // Non-rigid transform rejected.
    let mut sheared = single(cube_solid());
    sheared.instances[0].local_to_world =
        Affine([1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    assert!(build_payload(&sheared).is_err());
}

#[test]
fn closed_profile_extrusion_prism() {
    // A closed pentagon profile swept along z: exercises the u-periodic
    // extrusion path (seam tie-breaking during projection was the bug;
    // here we build the unwrapped trims directly the way the importer
    // now does, spanning the seam).
    let corners: Vec<[f64; 2]> = (0..5)
        .map(|i| {
            let a = TAU * i as f64 / 5.0;
            [2.0 * a.cos(), 2.0 * a.sin()]
        })
        .collect();
    let mut profile = corners.clone();
    profile.push(corners[0]); // closed
    let side = Face {
        surface: Surface::ExtrusionPolyline {
            frame: Frame::IDENTITY,
            profile: profile.clone(),
        },
        // Full period in u (5 segments), v in [-1, 1].
        trims: vec![vec![[0.0, -1.0], [5.0, -1.0], [5.0, 1.0], [0.0, 1.0]]],
    };
    let cap = |z: f64| Face {
        surface: Surface::Plane {
            frame: Frame::from_axis_ref([0.0, 0.0, z], [0.0, 0.0, z], [1.0, 0.0, 0.0]),
        },
        trims: vec![corners.clone().into_iter().collect()],
    };
    let model = single(Solid {
        faces: vec![side, cap(1.0), cap(-1.0)],
    });
    check_grid(&model, 24, [-2.0, 2.0, -2.0, 2.0, -1.0, 1.0], |p| {
        // Point-in-pentagon (prism): even-odd in 2D.
        let mut inside = false;
        let n = corners.len();
        let mut prev = corners[n - 1];
        let mut min_edge = f64::INFINITY;
        for &cur in &corners {
            if (prev[1] <= p[1]) != (cur[1] <= p[1]) {
                let t = (p[1] - prev[1]) / (cur[1] - prev[1]);
                if prev[0] + t * (cur[0] - prev[0]) > p[0] {
                    inside = !inside;
                }
            }
            // Distance to edge for the shell.
            let e = [cur[0] - prev[0], cur[1] - prev[1]];
            let l2 = e[0] * e[0] + e[1] * e[1];
            let t = (((p[0] - prev[0]) * e[0] + (p[1] - prev[1]) * e[1]) / l2).clamp(0.0, 1.0);
            let dx = p[0] - prev[0] - t * e[0];
            let dy = p[1] - prev[1] - t * e[1];
            min_edge = min_edge.min((dx * dx + dy * dy).sqrt());
            prev = cur;
        }
        let d = if inside {
            (-min_edge).max(p[2].abs() - 1.0)
        } else {
            min_edge.max(p[2].abs() - 1.0)
        };
        shell(d, 1e-6)
    });
}

#[test]
fn closed_nurbs_cylinder_seam() {
    // An exact rational-NURBS full cylinder (classic 9-point circle x
    // linear), seam at u = 0 facing +x — straight into the primary
    // parity ray. A seam crossing found by two seeds (u near 0 and u
    // near 1) must dedupe to one crossing.
    let w = core::f64::consts::FRAC_1_SQRT_2;
    let circle: [([f64; 2], f64); 9] = [
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], w),
        ([0.0, 1.0], 1.0),
        ([-1.0, 1.0], w),
        ([-1.0, 0.0], 1.0),
        ([-1.0, -1.0], w),
        ([0.0, -1.0], 1.0),
        ([1.0, -1.0], w),
        ([1.0, 0.0], 1.0),
    ];
    let mut ctrl = Vec::new();
    for (xy, weight) in circle {
        for z in [-1.0, 1.0] {
            ctrl.push([xy[0], xy[1], z, weight]);
        }
    }
    let side = Face {
        surface: Surface::Nurbs(NurbsSurface {
            degree_u: 2,
            degree_v: 1,
            nctrl_u: 9,
            nctrl_v: 2,
            knots_u: vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            ctrl,
        }),
        trims: vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
    };
    let circle64: Vec<[f64; 2]> = (0..64)
        .map(|i| {
            let a = TAU * i as f64 / 64.0;
            [a.cos(), a.sin()]
        })
        .collect();
    let cap = |z: f64| Face {
        surface: Surface::Plane {
            frame: Frame::from_axis_ref([0.0, 0.0, z], [0.0, 0.0, z], [1.0, 0.0, 0.0]),
        },
        trims: vec![circle64.clone()],
    };
    let model = single(Solid {
        faces: vec![side, cap(1.0), cap(-1.0)],
    });
    check_grid(&model, 24, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], |p| {
        let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let d = (rho - 1.0).max(p[2].abs() - 1.0);
        shell(d, 0.006)
    });
}
