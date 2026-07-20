//! Nearest-face color lookup through `PayloadView::nearest_color`: build
//! IR solids with styled faces, serialize, and check the color returned
//! at boundary-adjacent (and interior/exterior) query points.

use brep_core::ir::{BRepModel, Face, Instance, Solid, Surface};
use brep_core::math::{Affine, Frame};
use brep_core::payload::{PayloadView, build_payload};
use core::f64::consts::TAU;

const RED: [f32; 3] = [1.0, 0.0, 0.0];
const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

fn square_trim(h: f64) -> Vec<Vec<[f64; 2]>> {
    vec![vec![[-h, -h], [h, -h], [h, h], [-h, h]]]
}

/// A ±1 cube; `color_of(axis, side)` styles each face.
fn cube_solid(color_of: impl Fn(usize, f64) -> Option<[f32; 3]>) -> Solid {
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
                color: color_of(axis, side),
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

fn view_color(model: &BRepModel, p: [f64; 3]) -> [f32; 3] {
    let payload = build_payload(model).expect("build_payload");
    let view = PayloadView::new(&payload).expect("PayloadView");
    view.nearest_color(p).expect("model has faces")
}

#[test]
fn cube_faces_report_their_own_color() {
    // +x red, -x green, +z blue, everything else unstyled.
    let color_of = |axis: usize, side: f64| match (axis, side as i32) {
        (0, 1) => Some(RED),
        (0, -1) => Some(GREEN),
        (2, 1) => Some(BLUE),
        _ => None,
    };
    let model = single(cube_solid(color_of));
    let payload = build_payload(&model).unwrap();
    let view = PayloadView::new(&payload).unwrap();

    // Points a hair off each face center pick that face, inside and out.
    for (p, expect) in [
        ([1.001, 0.0, 0.0], RED),
        ([0.98, 0.1, -0.1], RED),
        ([-1.001, 0.0, 0.0], GREEN),
        ([-0.98, 0.0, 0.0], GREEN),
        ([0.0, 0.0, 1.001], BLUE),
        ([0.0, 0.0, 0.97], BLUE),
        // Unstyled faces read as white.
        ([0.0, 1.001, 0.0], WHITE),
        ([0.0, -0.99, 0.2], WHITE),
    ] {
        assert_eq!(view.nearest_color(p), Some(expect), "at {p:?}");
    }
}

#[test]
fn trim_clamp_beats_untrimmed_projection() {
    // Two coplanar-ish faces: a red x-y unit square at z = 0 and a green
    // square at z = 0.1 shifted to x in [3, 5]. A query above x ≈ 2.2
    // projects onto the *untrimmed* red plane at distance ~z, but the
    // red face's trim ends at x = 1; the clamped distance must lose to
    // the green face directly below the query.
    let red = Face {
        surface: Surface::Plane {
            frame: Frame::IDENTITY,
        },
        trims: square_trim(1.0),
        color: Some(RED),
    };
    let green = Face {
        surface: Surface::Plane {
            frame: Frame {
                origin: [4.0, 0.0, 0.1],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
                z: [0.0, 0.0, 1.0],
            },
        },
        trims: square_trim(1.0),
        color: Some(GREEN),
    };
    let model = single(Solid {
        faces: vec![red, green],
    });

    assert_eq!(view_color(&model, [0.0, 0.0, 0.05]), RED);
    assert_eq!(view_color(&model, [4.0, 0.0, 0.15]), GREEN);
    // Above the gap but closer to the green square's edge than to the
    // red trim boundary.
    assert_eq!(view_color(&model, [2.8, 0.0, 0.2]), GREEN);
    // And closer to red's trim edge on the other side of the gap.
    assert_eq!(view_color(&model, [1.2, 0.0, 0.2]), RED);
}

#[test]
fn cylinder_and_instances_with_scale() {
    // A colored cylinder wall (radius 1, |z| <= 1), instanced twice:
    // identity and a uniformly scaled + translated copy.
    let wall = Face {
        surface: Surface::Cylinder {
            frame: Frame::IDENTITY,
            radius: 1.0,
        },
        trims: vec![vec![[0.0, -1.0], [TAU, -1.0], [TAU, 1.0], [0.0, 1.0]]],
        color: Some(BLUE),
    };
    let caps = [1.0f64, -1.0].map(|z| Face {
        surface: Surface::Plane {
            frame: Frame {
                origin: [0.0, 0.0, z],
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
                z: [0.0, 0.0, z],
            },
        },
        trims: vec![
            (0..64)
                .map(|i| {
                    let a = TAU * i as f64 / 64.0;
                    [a.cos(), a.sin()]
                })
                .collect(),
        ],
        color: Some(RED),
    });
    let solid = Solid {
        faces: vec![wall, caps[0].clone(), caps[1].clone()],
    };
    // Second instance: scale 2, moved to x = 10.
    let scaled = Affine([
        2.0, 0.0, 0.0, 10.0, //
        0.0, 2.0, 0.0, 0.0, //
        0.0, 0.0, 2.0, 0.0,
    ]);
    let model = BRepModel {
        solids: vec![solid],
        instances: vec![
            Instance {
                solid: 0,
                local_to_world: Affine::IDENTITY,
                label: String::new(),
            },
            Instance {
                solid: 0,
                local_to_world: scaled,
                label: String::new(),
            },
        ],
    };

    // Wall vs cap on the unit instance.
    assert_eq!(view_color(&model, [1.01, 0.0, 0.0]), BLUE);
    assert_eq!(view_color(&model, [0.0, 0.0, 1.05]), RED);
    // The scaled instance's wall is at radius 2 around x = 10; a point
    // near it must resolve against that instance despite the identity
    // instance also being in play.
    assert_eq!(view_color(&model, [12.05, 0.0, 0.0]), BLUE);
    assert_eq!(view_color(&model, [10.0, 0.0, 2.1]), RED);
}

#[test]
fn quantization_round_trips_8_bit() {
    let tint = [0.25f32, 0.5, 0.75];
    let model = single(cube_solid(|_, _| Some(tint)));
    let got = view_color(&model, [1.1, 0.0, 0.0]);
    for (g, t) in got.iter().zip(tint) {
        assert!((g - t).abs() < 1.0 / 255.0, "{got:?} vs {tint:?}");
    }
}
