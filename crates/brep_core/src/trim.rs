//! Point-in-trim-region testing in a surface's UV space.
//!
//! The trim region is the even-odd fill over closed UV polylines, which
//! makes the test orientation-blind: outer bounds, holes, STEP sense
//! flags and loop nesting all reduce to crossing parity. Periodic
//! parameters are handled by testing the query point shifted by whole
//! periods against the *unwrapped* loops (see the layout notes in
//! `lib.rs`).

/// Accessor over one trim loop's UV points (owned IR or payload bytes).
pub trait TrimLoop {
    fn len(&self) -> usize;
    fn point(&self, i: usize) -> [f64; 2];
}

impl TrimLoop for &[[f64; 2]] {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn point(&self, i: usize) -> [f64; 2] {
        self[i]
    }
}

/// Random access to a face's trim loops. A PCB outer face can carry
/// hundreds of hole loops, so implementations index rather than
/// materialize.
pub trait LoopSet {
    type Loop<'a>: TrimLoop
    where
        Self: 'a;
    fn len(&self) -> usize;
    fn at(&self, i: usize) -> Self::Loop<'_>;
}

impl LoopSet for &[Vec<[f64; 2]>] {
    type Loop<'a>
        = &'a [[f64; 2]]
    where
        Self: 'a;
    fn len(&self) -> usize {
        (**self).len()
    }
    fn at(&self, i: usize) -> &[[f64; 2]] {
        &self[i]
    }
}

/// Even-odd parity of `(u, v)` against one closed loop, plus the squared
/// distance from the point to the loop in eps-normalized UV. The
/// half-open crossing rule (`v0 <= v < v1`) keeps shared polyline
/// vertices from double-counting.
fn loop_parity_and_dist2<L: TrimLoop>(lp: &L, u: f64, v: f64, inv_eps: [f64; 2]) -> (bool, f64) {
    let n = lp.len();
    if n == 0 {
        return (false, f64::INFINITY);
    }
    let mut inside = false;
    let mut dist2 = f64::INFINITY;
    let mut prev = lp.point(n - 1);
    for i in 0..n {
        let cur = lp.point(i);
        if (prev[1] <= v) != (cur[1] <= v) {
            let t = (v - prev[1]) / (cur[1] - prev[1]);
            let cross_u = prev[0] + t * (cur[0] - prev[0]);
            if cross_u > u {
                inside = !inside;
            }
        }
        // Distance to the segment, normalized per-axis by the suspicion
        // eps so "near the boundary" means near in 3D terms.
        let du = [
            (cur[0] - prev[0]) * inv_eps[0],
            (cur[1] - prev[1]) * inv_eps[1],
        ];
        let dp = [(u - prev[0]) * inv_eps[0], (v - prev[1]) * inv_eps[1]];
        let seg_len2 = du[0] * du[0] + du[1] * du[1];
        let t = if seg_len2 > 0.0 {
            ((dp[0] * du[0] + dp[1] * du[1]) / seg_len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dx = dp[0] - t * du[0];
        let dy = dp[1] - t * du[1];
        dist2 = dist2.min(dx * dx + dy * dy);
        prev = cur;
    }
    (inside, dist2)
}

/// The result of a trim-region query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimHit {
    pub inside: bool,
    /// The query point lies within the suspicion eps of a trim boundary —
    /// the classification is not trustworthy on its own.
    pub near_boundary: bool,
}

/// A face's trim region: loops plus the periodicity of the surface's
/// parameterization.
pub struct Region<S: LoopSet> {
    pub loops: S,
    /// `[min_u, max_u, min_v, max_v]` over all loop points (unwrapped).
    pub uv_aabb: [f64; 4],
    /// 0.0 = aperiodic.
    pub u_period: f64,
    pub v_period: f64,
    /// UV distance per axis equivalent to the 3D suspicion tolerance.
    pub uv_eps: [f64; 2],
}

impl<S: LoopSet> Region<S> {
    /// Test `(u, v)` against the region. On periodic axes the point is
    /// shifted by the whole periods that could land it inside the loops'
    /// unwrapped extent (±1 period around the AABB-centered shift), and
    /// the point counts as inside when any shifted copy has odd parity.
    pub fn contains(&self, u: f64, v: f64) -> TrimHit {
        let inv_eps = [
            if self.uv_eps[0] > 0.0 {
                1.0 / self.uv_eps[0]
            } else {
                0.0
            },
            if self.uv_eps[1] > 0.0 {
                1.0 / self.uv_eps[1]
            } else {
                0.0
            },
        ];
        let (u_shifts, nu) = period_shifts(u, self.u_period, self.uv_aabb[0], self.uv_aabb[1]);
        let (v_shifts, nv) = period_shifts(v, self.v_period, self.uv_aabb[2], self.uv_aabb[3]);

        let mut inside = false;
        let mut near = false;
        for &us in &u_shifts[..nu] {
            for &vs in &v_shifts[..nv] {
                let mut parity = false;
                for i in 0..self.loops.len() {
                    let lp = self.loops.at(i);
                    let (odd, d2) = loop_parity_and_dist2(&lp, us, vs, inv_eps);
                    parity ^= odd;
                    near |= d2 < 1.0;
                }
                inside |= parity;
            }
        }
        TrimHit {
            inside,
            near_boundary: near,
        }
    }
}

/// Candidate parameter values for a periodic axis: the input shifted so
/// it lands nearest the loop extent's center, plus one period either
/// side. Aperiodic axes test the value as-is.
fn period_shifts(x: f64, period: f64, lo: f64, hi: f64) -> ([f64; 3], usize) {
    if period <= 0.0 {
        ([x, 0.0, 0.0], 1)
    } else {
        let center = (lo + hi) * 0.5;
        let base = x + ((center - x) / period).round() * period;
        ([base, base - period, base + period], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(loops: &[Vec<[f64; 2]>], u_period: f64, v_period: f64) -> Region<&[Vec<[f64; 2]>]> {
        let mut aabb = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for lp in loops {
            for p in lp.iter() {
                aabb[0] = aabb[0].min(p[0]);
                aabb[1] = aabb[1].max(p[0]);
                aabb[2] = aabb[2].min(p[1]);
                aabb[3] = aabb[3].max(p[1]);
            }
        }
        Region {
            loops,
            uv_aabb: aabb,
            u_period,
            v_period,
            uv_eps: [1e-9, 1e-9],
        }
    }

    #[test]
    fn square_with_hole() {
        let loops = vec![
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
        ];
        let r = region(&loops, 0.0, 0.0);
        assert!(r.contains(0.5, 0.5).inside);
        assert!(!r.contains(2.0, 2.0).inside, "inside the hole");
        assert!(r.contains(3.5, 2.0).inside, "between hole and outer");
        assert!(!r.contains(5.0, 2.0).inside);
        assert!(!r.contains(-1.0, -1.0).inside);
    }

    #[test]
    fn hole_orientation_is_irrelevant() {
        // Hole wound the same way as the outer loop — even-odd doesn't
        // care.
        let loops = vec![
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            vec![[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]],
        ];
        let r = region(&loops, 0.0, 0.0);
        assert!(!r.contains(2.0, 2.0).inside);
        assert!(r.contains(0.5, 2.0).inside);
    }

    #[test]
    fn near_boundary_flag() {
        let loops = vec![vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]];
        let mut r = region(&loops, 0.0, 0.0);
        r.uv_eps = [0.1, 0.1];
        assert!(r.contains(2.0, 0.05).near_boundary);
        assert!(!r.contains(2.0, 2.0).near_boundary);
        assert!(r.contains(2.0, -0.05).near_boundary);
    }

    #[test]
    fn periodic_seam_crossing_loop() {
        use core::f64::consts::TAU;
        // A cylinder-side trim spanning u in [-0.5, 0.5] around the seam
        // (unwrapped past 0), v in [0, 1].
        let loops = vec![vec![[-0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [-0.5, 1.0]]];
        let r = region(&loops, TAU, 0.0);
        // Query arrives in canonical (-π, π] or [0, 2π) — both aliases
        // must resolve.
        assert!(r.contains(TAU - 0.25, 0.5).inside);
        assert!(r.contains(-0.25, 0.5).inside);
        assert!(r.contains(0.25, 0.5).inside);
        assert!(!r.contains(1.0, 0.5).inside);
        assert!(!r.contains(TAU - 1.0, 0.5).inside);
    }

    #[test]
    fn doubly_periodic_full_domain() {
        use core::f64::consts::TAU;
        // A full-torus face: the trim covers one whole period square.
        let loops = vec![vec![[0.0, 0.0], [TAU, 0.0], [TAU, TAU], [0.0, TAU]]];
        let r = region(&loops, TAU, TAU);
        for &(u, v) in &[(0.1, 0.1), (3.0, 6.0), (6.2, 0.0), (1.0, -1.0)] {
            assert!(r.contains(u, v).inside, "({u}, {v})");
        }
    }
}
