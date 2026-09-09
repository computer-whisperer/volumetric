//! Curve flattening into polyline points for [`crate::build_payload`],
//! shared by everything that feeds it (glyph outlines in
//! `text_render_core`, path sketches). Every function appends the points
//! *after* the start point and ends exactly on the curve's end point, so
//! consecutive pieces chain without duplicated joints. `tol` is the
//! maximum chord deviation in the caller's units; Bézier subdivision is
//! capped at depth 16 so a degenerate tolerance cannot recurse without
//! bound.

use std::f64::consts::PI;

pub type Point = [f64; 2];

fn mid(a: Point, b: Point) -> Point {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// Quadratic Bézier from `p0` to `p2` with control point `p1`.
pub fn quad(p0: Point, p1: Point, p2: Point, tol: f64, out: &mut Vec<Point>) {
    quad_rec(p0, p1, p2, tol * tol, 0, out);
}

fn quad_rec(p0: Point, p1: Point, p2: Point, tol2: f64, depth: u32, out: &mut Vec<Point>) {
    // Max deviation of a quadratic from its chord is |p1 - mid(p0,p2)|/2.
    let dx = p1[0] - (p0[0] + p2[0]) * 0.5;
    let dy = p1[1] - (p0[1] + p2[1]) * 0.5;
    if depth >= 16 || (dx * dx + dy * dy) * 0.25 <= tol2 {
        out.push(p2);
        return;
    }
    let (a, b) = (mid(p0, p1), mid(p1, p2));
    let m = mid(a, b);
    quad_rec(p0, a, m, tol2, depth + 1, out);
    quad_rec(m, b, p2, tol2, depth + 1, out);
}

/// Cubic Bézier from `p0` to `p3` with control points `p1`, `p2`.
pub fn cubic(p0: Point, p1: Point, p2: Point, p3: Point, tol: f64, out: &mut Vec<Point>) {
    cubic_rec(p0, p1, p2, p3, tol * tol, 0, out);
}

fn cubic_rec(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    tol2: f64,
    depth: u32,
    out: &mut Vec<Point>,
) {
    // Standard cubic flatness bound: deviation² <= (max(d1²)+max(d2²))/16
    // with d1 = 3p1 - 2p0 - p3, d2 = 3p2 - p0 - 2p3 (per component).
    let d1x = 3.0 * p1[0] - 2.0 * p0[0] - p3[0];
    let d1y = 3.0 * p1[1] - 2.0 * p0[1] - p3[1];
    let d2x = 3.0 * p2[0] - p0[0] - 2.0 * p3[0];
    let d2y = 3.0 * p2[1] - p0[1] - 2.0 * p3[1];
    let dev2 = (d1x * d1x).max(d2x * d2x) + (d1y * d1y).max(d2y * d2y);
    if depth >= 16 || dev2 <= 16.0 * tol2 {
        out.push(p3);
        return;
    }
    let (a, b, c) = (mid(p0, p1), mid(p1, p2), mid(p2, p3));
    let (ab, bc) = (mid(a, b), mid(b, c));
    let m = mid(ab, bc);
    cubic_rec(p0, a, ab, m, tol2, depth + 1, out);
    cubic_rec(m, bc, c, p3, tol2, depth + 1, out);
}

/// Elliptical arc about `center` with radii `rx`, `ry` and the x radius
/// rotated by `rotation` radians, from parametric angle `start` sweeping
/// `sweep` radians (positive = counterclockwise in a y-up frame). A circle
/// is `rx == ry`. The segment count comes from the sagitta bound on the
/// larger radius: at least four segments per full turn, at most ~6300.
#[allow(clippy::too_many_arguments)]
pub fn arc(
    center: Point,
    rx: f64,
    ry: f64,
    rotation: f64,
    start: f64,
    sweep: f64,
    tol: f64,
    out: &mut Vec<Point>,
) {
    let r = rx.abs().max(ry.abs());
    let tolerance_bites = tol.is_finite() && tol > 0.0 && tol < r;
    let max_step = if tolerance_bites {
        (2.0 * (1.0 - tol / r).acos()).clamp(1e-3, PI / 2.0)
    } else {
        PI / 2.0
    };
    let n = ((sweep.abs() / max_step).ceil() as usize).max(1);
    let (cr, sr) = (rotation.cos(), rotation.sin());
    for i in 1..=n {
        let a = start + sweep * (i as f64 / n as f64);
        let (x, y) = (rx * a.cos(), ry * a.sin());
        out.push([center[0] + cr * x - sr * y, center[1] + sr * x + cr * y]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Largest distance from any of `samples` to the polyline `p0..points`.
    fn max_deviation(p0: Point, points: &[Point], samples: &[Point]) -> f64 {
        let mut poly = vec![p0];
        poly.extend_from_slice(points);
        samples
            .iter()
            .map(|s| {
                poly.windows(2)
                    .map(|w| {
                        let (a, b) = (w[0], w[1]);
                        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                        let len2 = dx * dx + dy * dy;
                        let t = if len2 > 0.0 {
                            (((s[0] - a[0]) * dx + (s[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let (px, py) = (a[0] + t * dx, a[1] + t * dy);
                        ((s[0] - px).powi(2) + (s[1] - py).powi(2)).sqrt()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn quad_stays_within_tolerance_and_ends_on_the_curve() {
        let (p0, p1, p2) = ([0.0, 0.0], [5.0, 10.0], [10.0, 0.0]);
        let tol = 0.05;
        let mut out = Vec::new();
        quad(p0, p1, p2, tol, &mut out);
        assert_eq!(*out.last().unwrap(), p2);
        let samples: Vec<Point> = (0..=400)
            .map(|i| {
                let t = i as f64 / 400.0;
                let u = 1.0 - t;
                [
                    u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
                    u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
                ]
            })
            .collect();
        assert!(max_deviation(p0, &out, &samples) <= tol * 1.01);
        assert!(
            out.len() > 4,
            "a 10-unit-tall parabola needs several chords"
        );
    }

    #[test]
    fn cubic_stays_within_tolerance_and_ends_on_the_curve() {
        let (p0, p1, p2, p3) = ([0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]);
        let tol = 0.02;
        let mut out = Vec::new();
        cubic(p0, p1, p2, p3, tol, &mut out);
        assert_eq!(*out.last().unwrap(), p3);
        let samples: Vec<Point> = (0..=400)
            .map(|i| {
                let t = i as f64 / 400.0;
                let u = 1.0 - t;
                let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                [
                    w[0] * p0[0] + w[1] * p1[0] + w[2] * p2[0] + w[3] * p3[0],
                    w[0] * p0[1] + w[1] * p1[1] + w[2] * p2[1] + w[3] * p3[1],
                ]
            })
            .collect();
        assert!(max_deviation(p0, &out, &samples) <= tol * 1.01);
    }

    #[test]
    fn arc_points_lie_on_the_circle_with_bounded_sagitta() {
        let (center, r, tol) = ([1.0, -2.0], 3.0, 0.01);
        let mut out = Vec::new();
        arc(center, r, r, 0.0, 0.3, 2.0, tol, &mut out);
        for p in &out {
            let d = ((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2)).sqrt();
            assert!((d - r).abs() < 1e-12);
        }
        let last = *out.last().unwrap();
        assert!((last[0] - (center[0] + r * 2.3f64.cos())).abs() < 1e-12);
        assert!((last[1] - (center[1] + r * 2.3f64.sin())).abs() < 1e-12);
        let step = 2.0 / out.len() as f64;
        assert!(r * (1.0 - (step / 2.0).cos()) <= tol);
    }

    #[test]
    fn arc_handles_negative_sweep_rotation_and_coarse_tolerance() {
        let mut out = Vec::new();
        arc([0.0, 0.0], 2.0, 1.0, PI / 2.0, 0.0, -PI, 100.0, &mut out);
        // Coarse tolerance still yields at least a quarter-turn step: two
        // segments for a half turn.
        assert_eq!(out.len(), 2);
        // Rotating the x radius by 90 degrees puts the start point (rx, 0)
        // at (0, 2); a -pi sweep ends at (0, -2).
        let end = out[1];
        assert!(end[0].abs() < 1e-12 && (end[1] + 2.0).abs() < 1e-12);
    }
}
