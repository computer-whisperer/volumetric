//! Import-time geometry helpers: projecting 3D edge-curve points into a
//! surface's UV space (STEP AP214 files from OCCT carry 3D edge curves,
//! not pcurves, so importers must build trim loops by projection) and
//! flattening curves into polylines at a chordal tolerance.

use crate::ir::Surface;
use crate::math::{Vec3, norm, sub};
use crate::nurbs::{self, CurveData};
use crate::surface::SurfaceView;

/// Project a 3D point (in solid-local coordinates) onto the surface's UV
/// parameterization. `hint` is the UV of the previous point along the
/// same edge, used to select the right branch on periodic surfaces (the
/// returned u continues the hint's winding rather than jumping across
/// the seam) and to seed Newton on NURBS surfaces. Pass `None` for the
/// first point of a loop.
///
/// Returns an error when the point is not near the surface (importer
/// bug or unsupported geometry) — callers surface this as a face-level
/// import failure rather than emitting a wrong trim.
pub fn project_point(
    surface: &Surface,
    p: Vec3,
    hint: Option<[f64; 2]>,
    tol: f64,
) -> Result<[f64; 2], String> {
    use core::f64::consts::TAU;
    match surface {
        Surface::Mesh(_) => Err("mesh faces have no UV space to project into".into()),
        Surface::Plane { frame } => {
            let l = frame.to_local(p);
            check_residual(l[2].abs(), tol, "plane")?;
            Ok([l[0], l[1]])
        }
        Surface::Cylinder { frame, radius } => {
            let l = frame.to_local(p);
            let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
            check_residual((rho - radius).abs(), tol, "cylinder")?;
            let u = unwrap(l[1].atan2(l[0]), hint.map(|h| h[0]), TAU);
            Ok([u, l[2]])
        }
        Surface::Cone {
            frame,
            radius,
            half_angle,
        } => {
            let l = frame.to_local(p);
            let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
            let expected = radius + l[2] * half_angle.tan();
            check_residual((rho - expected).abs() * half_angle.cos(), tol, "cone")?;
            let u = unwrap(l[1].atan2(l[0]), hint.map(|h| h[0]), TAU);
            Ok([u, l[2]])
        }
        Surface::Sphere { frame, radius } => {
            let l = frame.to_local(p);
            check_residual((norm(l) - radius).abs(), tol, "sphere")?;
            let u = unwrap(l[1].atan2(l[0]), hint.map(|h| h[0]), TAU);
            let v = (l[2] / radius).clamp(-1.0, 1.0).asin();
            Ok([u, v])
        }
        Surface::Torus {
            frame,
            major,
            minor,
        } => {
            let l = frame.to_local(p);
            let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
            let tube = ((rho - major) * (rho - major) + l[2] * l[2]).sqrt();
            check_residual((tube - minor).abs(), tol, "torus")?;
            let u = unwrap(l[1].atan2(l[0]), hint.map(|h| h[0]), TAU);
            let v = unwrap(l[2].atan2(rho - major), hint.map(|h| h[1]), TAU);
            Ok([u, v])
        }
        Surface::ExtrusionPolyline { frame, profile } => {
            let l = frame.to_local(p);
            // Nearest point on the profile polyline in the XY plane,
            // collecting near-ties: on a closed profile the seam point
            // is equally close to the first and last segment, and the
            // tie must be broken toward the hint (continuity), never by
            // floating-point noise — a wrong pick tears the trim loop.
            let mut cands: Vec<(f64, f64)> = Vec::new(); // (dist, u)
            let mut best_d = f64::INFINITY;
            for i in 0..profile.len() - 1 {
                let a = profile[i];
                let b = profile[i + 1];
                let e = [b[0] - a[0], b[1] - a[1]];
                let len2 = e[0] * e[0] + e[1] * e[1];
                let t = if len2 > 0.0 {
                    (((l[0] - a[0]) * e[0] + (l[1] - a[1]) * e[1]) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let dx = l[0] - a[0] - t * e[0];
                let dy = l[1] - a[1] - t * e[1];
                let d = (dx * dx + dy * dy).sqrt();
                best_d = best_d.min(d);
                cands.push((d, i as f64 + t));
            }
            check_residual(best_d, tol, "extrusion")?;
            let period = if crate::ir::profile_is_closed(profile) {
                (profile.len() - 1) as f64
            } else {
                0.0
            };
            let slack = best_d + (tol * 1e-2).max(1e-12);
            let mut u = f64::NAN;
            let mut u_score = f64::INFINITY;
            for &(d, cu) in &cands {
                if d > slack {
                    continue;
                }
                let (aliased, score) = match hint {
                    Some(h) if period > 0.0 => {
                        let a = cu + ((h[0] - cu) / period).round() * period;
                        (a, (a - h[0]).abs())
                    }
                    Some(h) => (cu, (cu - h[0]).abs()),
                    None => (cu, cu), // no hint: lowest u wins, canonically
                };
                if score < u_score {
                    u_score = score;
                    u = aliased;
                }
            }
            Ok([u, l[2]])
        }
        Surface::Nurbs(n) => {
            // Gauss-Newton is only as good as its seed: polish from the
            // hint AND from the best cells of a dense grid scan (a
            // sparse scan can hand every seed to the same wrong local
            // minimum on folded patches).
            let dom = n.domain();
            let mut seeds: [[f64; 2]; 4] = [[0.0; 2]; 4];
            let mut seed_count = 0;
            if let Some(h) = hint {
                seeds[seed_count] = h;
                seed_count += 1;
            }
            const N: usize = 9;
            let mut ranked: [([f64; 2], f64); 3] = [([0.0; 2], f64::INFINITY); 3];
            for i in 0..=N {
                for j in 0..=N {
                    let u = dom[0] + (dom[1] - dom[0]) * i as f64 / N as f64;
                    let v = dom[2] + (dom[3] - dom[2]) * j as f64 / N as f64;
                    let (s, _, _) = nurbs::surface_eval(n, u, v);
                    let d = norm(sub(s, p));
                    // Insert into the 3-best list.
                    for slot in 0..ranked.len() {
                        if d < ranked[slot].1 {
                            ranked[slot..].rotate_right(1);
                            ranked[slot] = ([u, v], d);
                            break;
                        }
                    }
                }
            }
            for (uv, _) in ranked {
                seeds[seed_count] = uv;
                seed_count += 1;
            }
            let mut best: Option<([f64; 2], f64)> = None;
            for seed in &seeds[..seed_count] {
                if let Some((uv, d)) = nurbs_closest(n, p, *seed)
                    && best.is_none_or(|(_, bd)| d < bd)
                {
                    best = Some((uv, d));
                }
            }
            match best {
                Some((uv, d)) if d <= tol => Ok(uv),
                Some((_, d)) => Err(format!(
                    "point is {d:.3e} from the nurbs surface (tol {tol:.3e})"
                )),
                None => Err("nurbs closest-point projection failed to converge".into()),
            }
        }
    }
}

fn check_residual(residual: f64, tol: f64, what: &str) -> Result<(), String> {
    if residual <= tol {
        Ok(())
    } else {
        Err(format!(
            "point is {residual:.3e} from the {what} surface (tol {tol:.3e})"
        ))
    }
}

/// Choose the branch of a periodic parameter closest to the hint.
fn unwrap(angle: f64, hint: Option<f64>, period: f64) -> f64 {
    match hint {
        None => angle,
        Some(h) => angle + ((h - angle) / period).round() * period,
    }
}

/// Gauss-Newton closest point on a NURBS surface, clamped to the domain.
/// Returns (uv, distance).
fn nurbs_closest(n: &crate::ir::NurbsSurface, p: Vec3, seed: [f64; 2]) -> Option<([f64; 2], f64)> {
    Some(nurbs::surface_closest(n, p, seed))
}

/// Flatten a circle (full or arc, in its own frame at `radius`, from
/// angle `a0` to `a1`, a1 > a0) into 3D points at chordal tolerance
/// `tol`. Endpoints are included.
pub fn flatten_circle(
    frame: &crate::math::Frame,
    radius: f64,
    a0: f64,
    a1: f64,
    tol: f64,
) -> Vec<Vec3> {
    let span = (a1 - a0).abs();
    // Sagitta s = r (1 - cos(h/2)) <= tol per step h.
    let max_step = if tol >= radius {
        span
    } else {
        2.0 * (1.0 - tol / radius).acos()
    };
    let steps = ((span / max_step).ceil() as usize).clamp(1, 4096);
    (0..=steps)
        .map(|i| {
            let a = a0 + span * i as f64 / steps as f64;
            frame.to_world([radius * a.cos(), radius * a.sin(), 0.0])
        })
        .collect()
}

/// Flatten a B-spline curve segment (parameter `t0..t1`) into 3D points
/// at approximately chordal tolerance `tol`, by adaptive bisection on
/// midpoint deviation. Endpoints are included.
pub fn flatten_bspline<C: CurveData>(curve: &C, t0: f64, t1: f64, tol: f64) -> Vec<Vec3> {
    let mut out = Vec::new();
    let (p0, _) = nurbs::curve_eval(curve, t0);
    out.push(p0);
    subdivide(curve, t0, t1, tol, 0, &mut out);
    out
}

fn subdivide<C: CurveData>(
    curve: &C,
    t0: f64,
    t1: f64,
    tol: f64,
    depth: usize,
    out: &mut Vec<Vec3>,
) {
    let tm = (t0 + t1) * 0.5;
    let (p0, _) = nurbs::curve_eval(curve, t0);
    let (pm, _) = nurbs::curve_eval(curve, tm);
    let (p1, _) = nurbs::curve_eval(curve, t1);
    // Midpoint deviation from the chord.
    let chord = sub(p1, p0);
    let chord_len2 = crate::math::dot(chord, chord);
    let dev = if chord_len2 > 0.0 {
        let t = crate::math::dot(sub(pm, p0), chord) / chord_len2;
        norm(sub(pm, crate::math::add(p0, crate::math::scale(chord, t))))
    } else {
        norm(sub(pm, p0))
    };
    if depth >= 24 || dev <= tol {
        out.push(p1);
    } else {
        subdivide(curve, t0, tm, tol, depth + 1, out);
        subdivide(curve, tm, t1, tol, depth + 1, out);
    }
}

/// Sanity check that a projected UV loop actually lies on the surface:
/// re-evaluate each UV point and compare with the source 3D point.
/// Cheap insurance against branch/seam mistakes in importers.
pub fn verify_loop(
    surface: &Surface,
    points_3d: &[Vec3],
    uv: &[[f64; 2]],
    tol: f64,
) -> Result<(), String> {
    let view = SurfaceView::from_ir(surface);
    for (p, q) in points_3d.iter().zip(uv) {
        let s = view.eval(q[0], q[1]);
        let d = norm(sub(s, *p));
        if d > tol {
            return Err(format!(
                "uv ({}, {}) re-evaluates {d:.3e} from its source point",
                q[0], q[1]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NurbsSurface;
    use crate::math::Frame;
    use core::f64::consts::{PI, TAU};

    #[test]
    fn cylinder_projection_unwraps_across_seam() {
        let s = Surface::Cylinder {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        // Walk a circle across the -x seam: u must keep increasing.
        let mut hint = None;
        let mut last = f64::NEG_INFINITY;
        for i in 0..12 {
            let a = 2.5 + i as f64 * 0.2; // passes through π
            let p = [a.cos(), a.sin(), 0.3];
            let uv = project_point(&s, p, hint, 1e-9).unwrap();
            assert!(uv[0] > last, "u went backwards at step {i}");
            assert!((uv[1] - 0.3).abs() < 1e-12);
            last = uv[0];
            hint = Some(uv);
        }
        assert!(last > PI, "walk crossed the seam");
    }

    #[test]
    fn projection_rejects_off_surface_points() {
        let s = Surface::Sphere {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        assert!(project_point(&s, [1.5, 0.0, 0.0], None, 1e-6).is_err());
        assert!(project_point(&s, [1.0, 0.0, 0.0], None, 1e-6).is_ok());
    }

    #[test]
    fn torus_projection_roundtrip() {
        let s = Surface::Torus {
            frame: Frame::IDENTITY,
            major: 2.0,
            minor: 0.5,
        };
        let view = SurfaceView::from_ir(&s);
        for &(u, v) in &[(0.3, 1.2), (3.0, -2.0), (6.0, 3.0)] {
            let p = view.eval(u, v);
            let uv = project_point(&s, p, Some([u, v]), 1e-9).unwrap();
            assert!(
                (uv[0] - u).abs() < 1e-9 && (uv[1] - v).abs() < 1e-9,
                "({u}, {v}) -> {uv:?}"
            );
        }
    }

    #[test]
    fn circle_flattening_respects_tolerance() {
        let pts = flatten_circle(&Frame::IDENTITY, 10.0, 0.0, TAU, 0.01);
        assert!(pts.len() > 60, "10mm circle at 10µm: {} points", pts.len());
        // Every midpoint of every chord must be within tol of the circle.
        for w in pts.windows(2) {
            let m = [(w[0][0] + w[1][0]) / 2.0, (w[0][1] + w[1][1]) / 2.0, 0.0];
            let r = (m[0] * m[0] + m[1] * m[1]).sqrt();
            assert!(10.0 - r <= 0.0100001, "sagitta {}", 10.0 - r);
        }
    }

    #[test]
    fn closed_extrusion_projection_continues_across_seam() {
        // A closed square profile: walking points around the corner
        // where the profile closes (u = 4 back to u = 0) must continue
        // u monotonically via period aliasing, never tear back to 0.
        let profile = vec![
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
        ];
        let s = Surface::ExtrusionPolyline {
            frame: Frame::IDENTITY,
            profile,
        };
        // Walk along the bottom edge (segment 3->0 seam at x = 1),
        // starting just behind the first sample point (u = 3.1).
        let mut hint = Some([3.05, 0.0]);
        let mut last = 3.05;
        for i in 0..8 {
            let x = -0.8 + 1.7 * i as f64 / 7.0; // ends just short of the seam corner
            let p = [x, -1.0, 0.0];
            let uv = project_point(&s, p, hint, 1e-9).unwrap();
            assert!(
                uv[0] >= last - 1e-9,
                "u tore back at step {i}: {} -> {}",
                last,
                uv[0]
            );
            last = uv[0];
            hint = Some(uv);
        }
        // Continue past the seam onto segment 0 (x = 1 wall).
        let uv = project_point(&s, [1.0, -0.5, 0.0], hint, 1e-9).unwrap();
        assert!(
            uv[0] > 3.9 && uv[0] < 4.3,
            "seam crossing must alias forward, got u = {}",
            uv[0]
        );
    }

    #[test]
    fn nurbs_projection_bilinear() {
        let s = Surface::Nurbs(NurbsSurface {
            degree_u: 1,
            degree_v: 1,
            nctrl_u: 2,
            nctrl_v: 2,
            knots_u: vec![0.0, 0.0, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            ctrl: vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ],
        });
        let uv = project_point(&s, [0.25, 0.75, 0.0], None, 1e-9).unwrap();
        assert!((uv[0] - 0.25).abs() < 1e-9 && (uv[1] - 0.75).abs() < 1e-9);
    }
}
