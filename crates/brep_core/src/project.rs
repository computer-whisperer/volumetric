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
            let mut seeds: [[f64; 2]; 4] = [[0.0; 2]; 4];
            let mut seed_count = 0;
            if let Some(h) = hint {
                seeds[seed_count] = h;
                seed_count += 1;
            }
            seed_count += nurbs::scan_seeds(n, p, 9, &mut seeds[seed_count..]);
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

/// A point of a surface where one parameter is free: every value of the
/// free parameter at `value` on the fixed `axis` evaluates to `point`.
/// Sphere poles, the cone apex, and the collapsed edge rows of a NURBS
/// control net (see [`nurbs::pole_edges`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pole {
    pub point: Vec3,
    /// 0 = u is fixed (v free), 1 = v is fixed (u free).
    pub axis: usize,
    pub value: f64,
    /// The free parameter is periodic (period 2π): the pole is the
    /// whole top or bottom edge of the surface's periodic strip.
    pub periodic: bool,
}

/// The poles of a surface, if any.
pub fn poles(surface: &Surface) -> Vec<Pole> {
    use core::f64::consts::FRAC_PI_2;
    match surface {
        Surface::Sphere { frame, radius } => [-FRAC_PI_2, FRAC_PI_2]
            .into_iter()
            .map(|v| Pole {
                point: frame.to_world([0.0, 0.0, radius * v.signum()]),
                axis: 1,
                value: v,
                periodic: true,
            })
            .collect(),
        Surface::Cone {
            frame,
            radius,
            half_angle,
        } => {
            let t = half_angle.tan();
            if t == 0.0 || !t.is_finite() {
                return Vec::new();
            }
            let v = -radius / t;
            vec![Pole {
                point: frame.to_world([0.0, 0.0, v]),
                axis: 1,
                value: v,
                periodic: true,
            }]
        }
        Surface::Nurbs(n) => {
            let dom = n.domain();
            let edges = nurbs::pole_edges(n);
            (0..4)
                .filter(|&k| edges[k])
                .map(|k| {
                    let (u, v) = match k {
                        0 | 1 => (dom[k], 0.5 * (dom[2] + dom[3])),
                        _ => (0.5 * (dom[0] + dom[1]), dom[k]),
                    };
                    Pole {
                        point: nurbs::surface_eval(n, u, v).0,
                        axis: k / 2,
                        value: dom[k],
                        periodic: false,
                    }
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

/// The trim loop of a face that has no edge loops at all: the whole
/// surface. Only a surface whose parameter domain is closed in both
/// directions can be a face without bounds (a full sphere, written with
/// nothing but a vertex loop at a pole; a full torus).
pub fn full_domain_loop(surface: &Surface) -> Result<Vec<[f64; 2]>, String> {
    use core::f64::consts::{FRAC_PI_2, TAU};
    let [u0, u1, v0, v1] = match surface {
        Surface::Sphere { .. } => [0.0, TAU, -FRAC_PI_2, FRAC_PI_2],
        Surface::Torus { .. } => [0.0, TAU, 0.0, TAU],
        _ => return Err("face without edge loops on a surface that is not closed".into()),
    };
    Ok(vec![[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
}

/// Project a closed loop of 3D points (in solid-local coordinates) into
/// the surface's UV space as one trim polyline: each point is projected
/// with its predecessor's UV as the hint, and the result is checked to
/// re-evaluate onto its source point within `2 * tol`.
///
/// A point within `tol` of a pole (see [`poles`]) has a free parameter,
/// so it expands into two UV points — the pole's fixed coordinate paired
/// with each neighbour's free coordinate — and the polyline runs *along*
/// the pole edge from the one neighbour to the other. One arbitrary UV
/// for the pole would cut the corner diagonally and leave a sliver of the
/// face untrimmed. The returned polyline can therefore be longer than
/// `points`.
///
/// On a periodic pole (sphere, cone) the free coordinate of the far
/// neighbour has a branch to choose: the loop turns around the pole by
/// some angle in `(0, 2π]`, and which way it turns is the one thing the
/// geometry alone cannot say — the two meridians bounding a hemisphere
/// bound the other hemisphere just as well. `material_left` (the face
/// material lies on the left of the traversal, seen with the surface's
/// parameterization normal up: STEP `FACE_BOUND.orientation ==
/// ADVANCED_FACE.same_sense`) settles it: the polygon runs
/// counter-clockwise in UV, so along the top edge of the strip it runs
/// toward -u and along the bottom edge toward +u, from the near
/// neighbour until it reaches the far neighbour's branch. A loop that
/// *is* a hole and passes through a periodic pole would need the
/// opposite turn; the format does not reliably mark holes, so that case
/// is not supported (its complement is trimmed).
pub fn project_loop(
    surface: &Surface,
    points: &[Vec3],
    material_left: bool,
    tol: f64,
) -> Result<Vec<[f64; 2]>, String> {
    use core::f64::consts::TAU;
    let n = points.len();
    let poles = poles(surface);
    // A pole is a vertex of the loop, written exactly; the flattened
    // points beside it are a flattening step away, which on a feature
    // smaller than `tol` (a 20 µm fillet sphere) is less than `tol`. So
    // the pole test is relative to the loop's own extent as well.
    let extent = {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in points {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        norm(sub(hi, lo))
    };
    let pole_tol = tol.min(extent * 1e-3);
    let pole_of = |p: Vec3| {
        poles
            .iter()
            .find(|pole| norm(sub(p, pole.point)) <= pole_tol)
    };

    // Start the traversal off a pole, so every pole has a projected
    // predecessor; a closed loop reads the same from any start.
    let start = (0..n)
        .find(|&i| pole_of(points[i]).is_none())
        .ok_or("loop made of nothing but pole points")?;
    let at = |k: usize| points[(start + k) % n];

    let mut uv: Vec<[f64; 2]> = Vec::with_capacity(n + 2);
    let mut sources: Vec<Vec3> = Vec::with_capacity(n + 2);
    let mut hint: Option<[f64; 2]> = None;
    let mut k = 0;
    while k < n {
        let p = at(k);
        let Some(pole) = pole_of(p) else {
            let q = project_point(surface, p, hint, tol)?;
            uv.push(q);
            sources.push(p);
            hint = Some(q);
            k += 1;
            continue;
        };
        let prev = *uv.last().expect("the loop starts off a pole");
        let fixed = pole.axis;
        let free = 1 - fixed;
        // The far neighbour: the next point, or the loop's first point
        // when the pole closes the loop.
        let wraps = k + 1 == n;
        let far_p = at(k + 1);
        if pole_of(far_p).is_some() {
            return Err("two consecutive pole points on one loop".into());
        }
        let mut far = if wraps {
            uv[0]
        } else {
            project_point(surface, far_p, Some(prev), tol)?
        };
        if pole.periodic && fixed == 1 {
            // Counter-clockwise: -u along the top edge, +u along the
            // bottom; mirrored when the material is on the right.
            let top = prev[1] < pole.value;
            let dir = if top == material_left { -1.0 } else { 1.0 };
            let mut span = (dir * (far[0] - prev[0])).rem_euclid(TAU);
            if span < tol {
                span = TAU;
            }
            far[0] = prev[0] + dir * span;
        }
        for neighbour in [prev, far] {
            let mut q = [0.0; 2];
            q[fixed] = pole.value;
            q[free] = neighbour[free];
            uv.push(q);
            sources.push(p);
        }
        if !wraps {
            uv.push(far);
            sources.push(far_p);
        }
        hint = Some(far);
        k += 2;
    }

    verify_loop(surface, &sources, &uv, tol * 2.0)?;
    Ok(uv)
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
    flatten_ellipse(frame, radius, radius, a0, a1, tol)
}

/// Flatten an ellipse arc `(a cos t, b sin t)` in its own frame from
/// parameter `t0` to `t1` (t1 > t0) into 3D points at chordal tolerance
/// `tol`. Endpoints are included. `a`/`b` are the semi-axes along the
/// frame's x/y; a circle is the `a == b` case.
pub fn flatten_ellipse(
    frame: &crate::math::Frame,
    a: f64,
    b: f64,
    t0: f64,
    t1: f64,
    tol: f64,
) -> Vec<Vec3> {
    let span = (t1 - t0).abs();
    // Uniform parameter steps sized for the worst point of the curve:
    // curvature peaks at major/minor^2 (major-axis ends) and speed at
    // |r'| <= major, so a step h covers arc length <= major*h with
    // sagitta <= kappa L^2/8 = major^3 h^2 / (8 minor^2). Solving for
    // tol gives h = (minor/major) * sqrt(8 tol/major), i.e. the exact
    // circle step at the major radius scaled by minor/major (equal to
    // it for a circle).
    let major = a.max(b);
    let minor = a.min(b);
    let max_step = if tol >= major || minor <= 0.0 {
        span
    } else {
        2.0 * (1.0 - tol / major).acos() * (minor / major)
    };
    let steps = ((span / max_step).ceil() as usize).clamp(1, 4096);
    (0..=steps)
        .map(|i| {
            let t = t0 + span * i as f64 / steps as f64;
            frame.to_world([a * t.cos(), b * t.sin(), 0.0])
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
    fn ellipse_flattening_respects_tolerance() {
        // Eccentric ellipse (a/b = 5): the curvature at the major-axis
        // ends is 25x the circle-of-radius-a curvature, which is where
        // a naive circle step count would overshoot the tolerance.
        let (a, b, tol) = (10.0, 2.0, 0.01);
        let pts = flatten_ellipse(&Frame::IDENTITY, a, b, 0.0, TAU, tol);
        assert!(pts.len() < 4096, "hit the step clamp: {}", pts.len());
        let steps = pts.len() - 1;
        for (i, w) in pts.windows(2).enumerate() {
            let t0 = TAU * i as f64 / steps as f64;
            let t1 = TAU * (i + 1) as f64 / steps as f64;
            let chord = sub(w[1], w[0]);
            let len2 = crate::math::dot(chord, chord);
            for k in 1..16 {
                let t = t0 + (t1 - t0) * k as f64 / 16.0;
                let p = [a * t.cos(), b * t.sin(), 0.0];
                let s = crate::math::dot(sub(p, w[0]), chord) / len2;
                let foot = crate::math::add(w[0], crate::math::scale(chord, s));
                let dev = norm(sub(p, foot));
                assert!(dev <= tol * 1.0001, "chord {i}: deviation {dev}");
            }
        }
        // And the degenerate case is exactly the circle.
        let c = flatten_circle(&Frame::IDENTITY, 10.0, 0.0, TAU, 0.01);
        let e = flatten_ellipse(&Frame::IDENTITY, 10.0, 10.0, 0.0, TAU, 0.01);
        assert_eq!(c, e);
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

    /// A triangular bilinear patch: the u = 1 control row collapses to
    /// the apex (1, 0.5, 0), so (1, v) is that one point for every v.
    fn triangle_patch() -> NurbsSurface {
        NurbsSurface {
            degree_u: 1,
            degree_v: 1,
            nctrl_u: 2,
            nctrl_v: 2,
            knots_u: vec![0.0, 0.0, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            ctrl: vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.5, 0.0, 1.0],
                [1.0, 0.5, 0.0, 1.0],
            ],
        }
    }

    #[test]
    fn nurbs_projection_near_a_pole_leaves_the_pole_seed() {
        // Just below the apex on the v = 0 edge (y = 0.5 - 0.5 (1 - u)),
        // seeded from the apex itself where the v partial vanishes.
        let s = Surface::Nurbs(triangle_patch());
        let p = [0.98, 0.49, 0.0];
        let uv = project_point(&s, p, Some([1.0, 1.0]), 1e-9).unwrap();
        assert!(
            (uv[0] - 0.98).abs() < 1e-7 && uv[1].abs() < 1e-7,
            "projected to {uv:?}"
        );
    }

    /// A sphere's two meridians at u = 0 and u = π/2 bound a quarter
    /// and a three-quarter face alike; the sense flag picks the turn.
    #[test]
    fn sphere_loop_turns_around_the_pole_by_the_sense_flag() {
        let s = Surface::Sphere {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        // Up the u = 0 meridian to the north pole, down the u = π/2
        // meridian to the south pole, and back up.
        let mut pts: Vec<Vec3> = Vec::new();
        for i in 1..8 {
            let v = -PI / 2.0 + PI * i as f64 / 8.0;
            pts.push([v.cos(), 0.0, v.sin()]);
        }
        pts.push([0.0, 0.0, 1.0]);
        for i in 1..8 {
            let v = PI / 2.0 - PI * i as f64 / 8.0;
            pts.push([0.0, v.cos(), v.sin()]);
        }
        pts.push([0.0, 0.0, -1.0]);

        let span = |uv: &[[f64; 2]]| {
            let (lo, hi) = uv
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| {
                    (lo.min(q[0]), hi.max(q[0]))
                });
            hi - lo
        };
        // Heading north along x > 0 with the outward normal up, left is
        // y < 0: the material on the left is the three quarters outside
        // the x > 0, y > 0 corner, a three-quarter turn at each pole.
        let uv = project_loop(&s, &pts, true, 1e-9).unwrap();
        assert_eq!(uv.len(), pts.len() + 2);
        assert!(
            (span(&uv) - 3.0 * PI / 2.0).abs() < 1e-9,
            "span {}",
            span(&uv)
        );
        // Material on the right is the quarter.
        let uv = project_loop(&s, &pts, false, 1e-9).unwrap();
        assert!((span(&uv) - PI / 2.0).abs() < 1e-9, "span {}", span(&uv));
    }

    #[test]
    fn nurbs_loop_runs_along_a_collapsed_edge() {
        let s = Surface::Nurbs(triangle_patch());
        assert_eq!(
            nurbs::pole_edges(&triangle_patch()),
            [false, true, false, false]
        );
        // The patch's own boundary: up the u = 0 edge, along v = 1 to the
        // apex, back along v = 0.
        let pts: Vec<Vec3> = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.75, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 0.25, 0.0],
        ];
        let uv = project_loop(&s, &pts, true, 1e-9).unwrap();
        // The apex became two corner points joined along u = 1.
        assert_eq!(uv.len(), pts.len() + 1);
        let close =
            |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-7 && (a[1] - b[1]).abs() < 1e-7;
        assert!(close(uv[3], [0.5, 1.0]), "{:?}", uv[3]);
        assert!(close(uv[4], [1.0, 1.0]), "{:?}", uv[4]);
        assert!(close(uv[5], [1.0, 0.0]), "{:?}", uv[5]);
        assert!(close(uv[6], [0.5, 0.0]), "{:?}", uv[6]);
    }
}
