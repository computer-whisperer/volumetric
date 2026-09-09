//! The Subspace gizmo: an infinite plane, line, point or frame drawn as a
//! window sized by the scene it sits in.

use glam::{Mat4, Vec3};
use volumetric_renderer as renderer;

use crate::PreviewBounds;

/// Subspace gizmo palette: the subspace itself in cyan-blue, the oriented
/// hyperplane normal in amber, frame triads in the usual RGB = xyz.
const SUBSPACE_COLOR: [f32; 4] = [0.30, 0.75, 1.0, 0.9];
const SUBSPACE_GRID_COLOR: [f32; 4] = [0.30, 0.75, 1.0, 0.28];
const SUBSPACE_NORMAL_COLOR: [f32; 4] = [1.0, 0.72, 0.2, 0.9];
const SUBSPACE_FRAME_COLORS: [[f32; 4]; 3] = [
    [0.94, 0.35, 0.35, 0.95],
    [0.42, 0.85, 0.35, 0.95],
    [0.35, 0.55, 1.0, 0.95],
];

/// A subspace's world vector padded into viewport 3-space (2-space values
/// draw in the z = 0 plane).
pub(crate) fn pad3(v: &[f64]) -> Vec3 {
    Vec3::new(
        v.first().copied().unwrap_or(0.0) as f32,
        v.get(1).copied().unwrap_or(0.0) as f32,
        v.get(2).copied().unwrap_or(0.0) as f32,
    )
}

fn gizmo_segment(start: Vec3, end: Vec3, color: [f32; 4]) -> renderer::LineSegment {
    renderer::LineSegment {
        start: start.to_array(),
        end: end.to_array(),
        color,
    }
}

/// Immediate-mode gizmo for a subspace entity. The subspace is infinite,
/// so its display window is sized by the scene bounds and centered on the
/// scene center's projection onto the subspace — a bed plane under a part
/// spans the part no matter where its chart origin sits. Markers: a dot at
/// the chart origin, and for hyperplanes an amber arrow along the oriented
/// normal.
///
/// Called per frame by the viewport and once by the CLI's `render`.
pub fn submit_subspace_gizmo(
    renderer: &mut renderer::Renderer,
    subspace: &volumetric::subspace::Subspace,
    scene: PreviewBounds,
) {
    let scene_center = (scene.min_vec3() + scene.max_vec3()) * 0.5;
    let half = ((scene.max_vec3() - scene.min_vec3()).length() * 0.55).max(1.0);
    let center = {
        let world: Vec<f64> = scene_center.to_array()[..subspace.ambient().min(3)]
            .iter()
            .map(|c| *c as f64)
            .collect();
        let (chart, _) = subspace.project(&world);
        pad3(&subspace.embed(&chart))
    };

    let origin = pad3(&subspace.origin);
    let basis: Vec<Vec3> = (0..subspace.rank())
        .map(|i| pad3(subspace.basis_vector(i)))
        .collect();

    // Depth-tested faint grid vs always-visible structural lines.
    let mut main = Vec::new();
    let mut grid = Vec::new();
    let mut points = vec![renderer::PointInstance {
        position: origin.to_array(),
        color: SUBSPACE_COLOR,
    }];

    match basis.as_slice() {
        [] => {
            let r = half * 0.12;
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                main.push(gizmo_segment(
                    origin - axis * r,
                    origin + axis * r,
                    SUBSPACE_COLOR,
                ));
            }
        }
        [dir] => {
            main.push(gizmo_segment(
                center - *dir * half,
                center + *dir * half,
                SUBSPACE_COLOR,
            ));
        }
        [u, v] => {
            let at = |su: f32, sv: f32| center + *u * (su * half) + *v * (sv * half);
            let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            for (a, b) in corners.iter().zip(corners.iter().cycle().skip(1)) {
                main.push(gizmo_segment(at(a.0, a.1), at(b.0, b.1), SUBSPACE_COLOR));
            }
            for t in [-0.5, 0.0, 0.5] {
                grid.push(gizmo_segment(at(t, -1.0), at(t, 1.0), SUBSPACE_GRID_COLOR));
                grid.push(gizmo_segment(at(-1.0, t), at(1.0, t), SUBSPACE_GRID_COLOR));
            }
        }
        // Full frame: an axis triad at the chart origin.
        triad => {
            for (b, color) in triad.iter().zip(SUBSPACE_FRAME_COLORS) {
                main.push(gizmo_segment(origin, origin + *b * (half * 0.5), color));
            }
        }
    }

    if let Some(normal) = subspace.normal() {
        let tip = center + pad3(&normal) * (half * 0.35);
        main.push(gizmo_segment(center, tip, SUBSPACE_NORMAL_COLOR));
        points.push(renderer::PointInstance {
            position: tip.to_array(),
            color: SUBSPACE_NORMAL_COLOR,
        });
    }

    if !grid.is_empty() {
        renderer.submit_lines(
            &renderer::LineData { segments: grid },
            Mat4::IDENTITY,
            renderer::LineStyle {
                width: 1.0,
                width_mode: renderer::WidthMode::ScreenSpace,
                pattern: renderer::LinePattern::Solid,
                depth_mode: renderer::DepthMode::Normal,
            },
        );
    }
    renderer.submit_lines(
        &renderer::LineData { segments: main },
        Mat4::IDENTITY,
        renderer::LineStyle {
            width: 2.0,
            width_mode: renderer::WidthMode::ScreenSpace,
            pattern: renderer::LinePattern::Solid,
            depth_mode: renderer::DepthMode::Overlay,
        },
    );
    renderer.submit_points(
        &renderer::PointData { points },
        Mat4::IDENTITY,
        renderer::PointStyle {
            size: 8.0,
            size_mode: renderer::WidthMode::ScreenSpace,
            shape: renderer::PointShape::Circle,
            depth_mode: renderer::DepthMode::Overlay,
        },
    );
}
