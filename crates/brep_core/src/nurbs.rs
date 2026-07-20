//! B-spline / NURBS evaluation (curves and surfaces, values and first
//! derivatives), implemented once against small accessor traits so the
//! owned IR types and the zero-copy payload views share the kernels.
//!
//! Algorithms follow The NURBS Book (A2.1 span search, A2.2 basis
//! functions, A2.3 basis derivatives) with fixed-size scratch arrays —
//! no allocation at sample time. Degrees above [`MAX_DEGREE`] are
//! rejected at import.

use crate::math::Vec3;

pub const MAX_DEGREE: usize = 7;
const MAX_ORDER: usize = MAX_DEGREE + 1;

/// Accessor for B-spline surface data.
pub trait SurfaceData {
    fn degree_u(&self) -> usize;
    fn degree_v(&self) -> usize;
    fn nctrl_u(&self) -> usize;
    fn nctrl_v(&self) -> usize;
    fn knot_u(&self, i: usize) -> f64;
    fn knot_v(&self, i: usize) -> f64;
    /// xyzw control point at net position (i, j).
    fn ctrl(&self, i: usize, j: usize) -> [f64; 4];

    /// Valid parameter domain `[u_min, u_max, v_min, v_max]`.
    fn domain(&self) -> [f64; 4] {
        [
            self.knot_u(self.degree_u()),
            self.knot_u(self.nctrl_u()),
            self.knot_v(self.degree_v()),
            self.knot_v(self.nctrl_v()),
        ]
    }
}

/// Accessor for B-spline curve data (xyzw; z = 0 for 2D profile curves).
pub trait CurveData {
    fn degree(&self) -> usize;
    fn nctrl(&self) -> usize;
    fn knot(&self, i: usize) -> f64;
    fn ctrl(&self, i: usize) -> [f64; 4];

    fn domain(&self) -> [f64; 2] {
        [self.knot(self.degree()), self.knot(self.nctrl())]
    }
}

/// Knot span index for parameter `u` (A2.1), clamped to the valid domain.
fn find_span(n_ctrl: usize, degree: usize, u: f64, knot: impl Fn(usize) -> f64) -> usize {
    let lo_dom = knot(degree);
    let hi_dom = knot(n_ctrl);
    let u = u.clamp(lo_dom, hi_dom);
    if u >= hi_dom {
        // Right end lands in the last non-empty span.
        let mut s = n_ctrl - 1;
        while s > degree && knot(s) >= knot(s + 1) {
            s -= 1;
        }
        return s;
    }
    let (mut lo, mut hi) = (degree, n_ctrl);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if u < knot(mid) { hi = mid } else { lo = mid }
    }
    lo
}

/// Non-zero basis functions and their first derivatives (A2.3, k = 1).
fn basis_funs_ders(
    span: usize,
    u: f64,
    degree: usize,
    knot: impl Fn(usize) -> f64,
) -> ([f64; MAX_ORDER], [f64; MAX_ORDER]) {
    // ndu[j][r]: triangle of basis values and knot differences.
    let mut ndu = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut left = [0.0; MAX_ORDER];
    let mut right = [0.0; MAX_ORDER];
    ndu[0][0] = 1.0;
    for j in 1..=degree {
        left[j] = u - knot(span + 1 - j);
        right[j] = knot(span + j) - u;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = if ndu[j][r] != 0.0 {
                ndu[r][j - 1] / ndu[j][r]
            } else {
                0.0
            };
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    let mut vals = [0.0; MAX_ORDER];
    let mut ders = [0.0; MAX_ORDER];
    for r in 0..=degree {
        vals[r] = ndu[r][degree];
    }
    if degree > 0 {
        let p = degree as f64;
        for r in 0..=degree {
            // First derivative: p * (a1 - a2) with the two adjacent
            // lower-degree bases over their knot spans.
            let a1 = if r >= 1 && ndu[degree][r - 1] != 0.0 {
                ndu[r - 1][degree - 1] / ndu[degree][r - 1]
            } else {
                0.0
            };
            let a2 = if r < degree && ndu[degree][r] != 0.0 {
                ndu[r][degree - 1] / ndu[degree][r]
            } else {
                0.0
            };
            ders[r] = p * (a1 - a2);
        }
    }
    (vals, ders)
}

/// Surface point and first partials at (u, v), rational-aware.
pub fn surface_eval<S: SurfaceData>(s: &S, u: f64, v: f64) -> (Vec3, Vec3, Vec3) {
    let (pu, pv) = (s.degree_u(), s.degree_v());
    let span_u = find_span(s.nctrl_u(), pu, u, |i| s.knot_u(i));
    let span_v = find_span(s.nctrl_v(), pv, v, |i| s.knot_v(i));
    let (nu, du) = basis_funs_ders(span_u, u, pu, |i| s.knot_u(i));
    let (nv, dv) = basis_funs_ders(span_v, v, pv, |i| s.knot_v(i));

    // Homogeneous accumulation: A = Σ N P_w, and its u/v partials.
    let mut a = [0.0f64; 4];
    let mut a_u = [0.0f64; 4];
    let mut a_v = [0.0f64; 4];
    for i in 0..=pu {
        for j in 0..=pv {
            let c = s.ctrl(span_u - pu + i, span_v - pv + j);
            let w = c[3];
            let pw = [c[0] * w, c[1] * w, c[2] * w, w];
            for k in 0..4 {
                a[k] += nu[i] * nv[j] * pw[k];
                a_u[k] += du[i] * nv[j] * pw[k];
                a_v[k] += nu[i] * dv[j] * pw[k];
            }
        }
    }
    let w = a[3];
    let inv_w = if w != 0.0 { 1.0 / w } else { 0.0 };
    let p = [a[0] * inv_w, a[1] * inv_w, a[2] * inv_w];
    // Quotient rule: S' = (A' - w' S) / w.
    let s_u = [
        (a_u[0] - a_u[3] * p[0]) * inv_w,
        (a_u[1] - a_u[3] * p[1]) * inv_w,
        (a_u[2] - a_u[3] * p[2]) * inv_w,
    ];
    let s_v = [
        (a_v[0] - a_v[3] * p[0]) * inv_w,
        (a_v[1] - a_v[3] * p[1]) * inv_w,
        (a_v[2] - a_v[3] * p[2]) * inv_w,
    ];
    (p, s_u, s_v)
}

/// Curve point and first derivative at u, rational-aware.
pub fn curve_eval<C: CurveData>(c: &C, u: f64) -> (Vec3, Vec3) {
    let p = c.degree();
    let span = find_span(c.nctrl(), p, u, |i| c.knot(i));
    let (n, d) = basis_funs_ders(span, u, p, |i| c.knot(i));
    let mut a = [0.0f64; 4];
    let mut a_d = [0.0f64; 4];
    for i in 0..=p {
        let cp = c.ctrl(span - p + i);
        let w = cp[3];
        let pw = [cp[0] * w, cp[1] * w, cp[2] * w, w];
        for k in 0..4 {
            a[k] += n[i] * pw[k];
            a_d[k] += d[i] * pw[k];
        }
    }
    let w = a[3];
    let inv_w = if w != 0.0 { 1.0 / w } else { 0.0 };
    let pt = [a[0] * inv_w, a[1] * inv_w, a[2] * inv_w];
    let der = [
        (a_d[0] - a_d[3] * pt[0]) * inv_w,
        (a_d[1] - a_d[3] * pt[1]) * inv_w,
        (a_d[2] - a_d[3] * pt[2]) * inv_w,
    ];
    (pt, der)
}

impl SurfaceData for crate::ir::NurbsSurface {
    fn degree_u(&self) -> usize {
        self.degree_u
    }
    fn degree_v(&self) -> usize {
        self.degree_v
    }
    fn nctrl_u(&self) -> usize {
        self.nctrl_u
    }
    fn nctrl_v(&self) -> usize {
        self.nctrl_v
    }
    fn knot_u(&self, i: usize) -> f64 {
        self.knots_u[i]
    }
    fn knot_v(&self, i: usize) -> f64 {
        self.knots_v[i]
    }
    fn ctrl(&self, i: usize, j: usize) -> [f64; 4] {
        self.ctrl[i * self.nctrl_v + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NurbsSurface;

    /// A bilinear patch: degree 1x1 over [0,1]², the unit square tilted
    /// into a known plane.
    fn bilinear() -> NurbsSurface {
        NurbsSurface {
            degree_u: 1,
            degree_v: 1,
            nctrl_u: 2,
            nctrl_v: 2,
            knots_u: vec![0.0, 0.0, 1.0, 1.0],
            knots_v: vec![0.0, 0.0, 1.0, 1.0],
            ctrl: vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 1.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 1.0],
            ],
        }
    }

    #[test]
    fn bilinear_eval_matches_lerp() {
        let s = bilinear();
        s.validate().unwrap();
        let (p, su, sv) = surface_eval(&s, 0.25, 0.5);
        assert!((p[0] - 0.75).abs() < 1e-12);
        assert!((p[1] - 1.0).abs() < 1e-12);
        assert!((p[2] - 0.5).abs() < 1e-12);
        assert!((su[0] - 3.0).abs() < 1e-12 && su[1].abs() < 1e-12);
        assert!((sv[1] - 2.0).abs() < 1e-12 && (sv[2] - 1.0).abs() < 1e-12);
    }

    /// A rational quadratic quarter circle: the classic 3-point arc with
    /// the middle weight 1/√2 traces an exact unit circle.
    #[test]
    fn rational_quarter_circle() {
        struct Arc;
        impl CurveData for Arc {
            fn degree(&self) -> usize {
                2
            }
            fn nctrl(&self) -> usize {
                3
            }
            fn knot(&self, i: usize) -> f64 {
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0][i]
            }
            fn ctrl(&self, i: usize) -> [f64; 4] {
                let w = core::f64::consts::FRAC_1_SQRT_2;
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, w],
                    [0.0, 1.0, 0.0, 1.0],
                ][i]
            }
        }
        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let (p, _) = curve_eval(&Arc, u);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r - 1.0).abs() < 1e-12, "u={u}: radius {r}");
        }
    }

    #[test]
    fn span_search_ends() {
        // Clamped cubic with an interior knot; both domain ends must land
        // in valid spans.
        let knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let k = |i: usize| knots[i];
        assert_eq!(find_span(5, 3, 0.0, k), 3);
        assert_eq!(find_span(5, 3, 0.4999, k), 3);
        assert_eq!(find_span(5, 3, 0.5, k), 4);
        assert_eq!(find_span(5, 3, 1.0, k), 4);
        // Out-of-domain clamps.
        assert_eq!(find_span(5, 3, -1.0, k), 3);
        assert_eq!(find_span(5, 3, 2.0, k), 4);
    }
}
