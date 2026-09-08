//! Affine maps of 3-space: the transforms wrapper operators apply.
//!
//! A map is `p' = linear * p + offset`. Constructors cover the operator
//! vocabulary (translation, per-axis scaling, Euler rotation, reflection
//! across an axis-aligned plane, rotation about an axis-parallel line);
//! [`Affine::then`] composes them and [`Affine::inverse`] gives the map a
//! sample wrapper applies to the query point.
//!
//! Wrappers apply a map to the spatial prefix of a model — min(dims, 3)
//! coordinates — so a 2D sketch uses the top-left 2x2 block and the first
//! two offsets. [`Affine::preserves_prefix`] tells whether that truncation
//! is faithful (a rotation about x is not, for a 2D input).

/// A 3-space affine map `p' = linear * p + offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    /// Row-major 3x3 linear part.
    pub linear: [[f64; 3]; 3],
    /// Translation applied after the linear part.
    pub offset: [f64; 3],
}

/// Tolerance below which a matrix entry counts as zero (or a diagonal entry
/// as one) when classifying a map's shape.
const SHAPE_EPSILON: f64 = 1e-12;

impl Affine {
    pub const IDENTITY: Affine = Affine {
        linear: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        offset: [0.0; 3],
    };

    pub fn translation(offset: [f64; 3]) -> Self {
        Affine {
            linear: Self::IDENTITY.linear,
            offset,
        }
    }

    /// Per-axis scaling about the origin (negative factors reflect).
    pub fn scaling(factors: [f64; 3]) -> Self {
        let mut linear = [[0.0; 3]; 3];
        for (i, factor) in factors.iter().enumerate() {
            linear[i][i] = *factor;
        }
        Affine {
            linear,
            offset: [0.0; 3],
        }
    }

    /// Rotation by Euler angles in degrees, applied X, then Y, then Z
    /// (`R = Rz * Ry * Rx`) about the origin.
    pub fn euler_deg(rx_deg: f64, ry_deg: f64, rz_deg: f64) -> Self {
        Self::axis_rotation(0, rx_deg)
            .then(&Self::axis_rotation(1, ry_deg))
            .then(&Self::axis_rotation(2, rz_deg))
    }

    /// Rotation by `deg` about the world axis `axis` (0 = x, 1 = y, 2 = z)
    /// through the origin, right-handed.
    pub fn axis_rotation(axis: usize, deg: f64) -> Self {
        let (s, c) = deg.to_radians().sin_cos();
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut linear = [[0.0; 3]; 3];
        linear[axis][axis] = 1.0;
        linear[u][u] = c;
        linear[u][v] = -s;
        linear[v][u] = s;
        linear[v][v] = c;
        Affine {
            linear,
            offset: [0.0; 3],
        }
    }

    /// Rotation by `deg` about the line through `center` parallel to the
    /// world axis `axis`.
    pub fn rotation_about(axis: usize, center: [f64; 3], deg: f64) -> Self {
        let neg = [-center[0], -center[1], -center[2]];
        Self::translation(neg)
            .then(&Self::axis_rotation(axis, deg))
            .then(&Self::translation(center))
    }

    /// Reflection across the plane `coordinate[axis] = offset` (for a 2D
    /// model, the line).
    pub fn mirror(axis: usize, offset: f64) -> Self {
        let mut map = Self::IDENTITY;
        map.linear[axis][axis] = -1.0;
        map.offset[axis] = 2.0 * offset;
        map
    }

    /// The map applying `self` first, then `next`.
    pub fn then(&self, next: &Affine) -> Affine {
        let mut linear = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                linear[i][j] = (0..3).map(|k| next.linear[i][k] * self.linear[k][j]).sum();
            }
        }
        // L2 (L1 p + t1) + t2: the new offset is `next` applied to t1.
        let offset = next.apply(self.offset);
        Affine { linear, offset }
    }

    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let mut out = self.offset;
        for i in 0..3 {
            out[i] += (0..3).map(|j| self.linear[i][j] * p[j]).sum::<f64>();
        }
        out
    }

    /// The map with every coordinate at index `n` or above left untouched:
    /// what a wrapper effectively applies to an `n`-dimensional input.
    pub fn restricted(&self, n: usize) -> Affine {
        let mut map = Self::IDENTITY;
        for i in 0..n.min(3) {
            map.linear[i][..n.min(3)].copy_from_slice(&self.linear[i][..n.min(3)]);
            map.offset[i] = self.offset[i];
        }
        map
    }

    /// Whether the map sends the first `n` coordinates to themselves and
    /// never mixes them with the rest: true iff applying [`restricted`]
    /// loses nothing for an `n`-dimensional input.
    ///
    /// [`restricted`]: Affine::restricted
    pub fn preserves_prefix(&self, n: usize) -> bool {
        let n = n.min(3);
        (0..3).all(|i| {
            (0..3).all(|j| {
                let mixes = (i < n) != (j < n);
                !mixes || self.linear[i][j].abs() <= SHAPE_EPSILON
            })
        }) && (n..3).all(|i| {
            (self.linear[i][i] - 1.0).abs() <= SHAPE_EPSILON
                && self.offset[i].abs() <= SHAPE_EPSILON
        })
    }

    /// Whether the linear part is the identity (a pure translation).
    pub fn is_translation(&self) -> bool {
        (0..3).all(|i| {
            (0..3).all(|j| {
                let target = if i == j { 1.0 } else { 0.0 };
                (self.linear[i][j] - target).abs() <= SHAPE_EPSILON
            })
        })
    }

    /// Whether the linear part is diagonal (per-axis scaling, reflections
    /// included).
    pub fn is_diagonal(&self) -> bool {
        (0..3).all(|i| (0..3).all(|j| i == j || self.linear[i][j].abs() <= SHAPE_EPSILON))
    }

    /// The inverse map, or `None` when the linear part is singular.
    pub fn inverse(&self) -> Option<Affine> {
        let m = &self.linear;
        let cofactor = |r: usize, c: usize| {
            let (r0, r1) = ((r + 1) % 3, (r + 2) % 3);
            let (c0, c1) = ((c + 1) % 3, (c + 2) % 3);
            m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
        };
        let det: f64 = (0..3).map(|j| m[0][j] * cofactor(0, j)).sum();
        let scale = m.iter().flatten().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if !det.is_finite() || det.abs() <= SHAPE_EPSILON * scale * scale * scale {
            return None;
        }
        let mut linear = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                // Inverse = adjugate / det; adjugate is the transposed cofactor matrix.
                linear[i][j] = cofactor(j, i) / det;
            }
        }
        let inv_linear = Affine {
            linear,
            offset: [0.0; 3],
        };
        let neg = inv_linear.apply(self.offset);
        Some(Affine {
            linear,
            offset: [-neg[0], -neg[1], -neg[2]],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn composition_applies_left_to_right() {
        let map = Affine::scaling([2.0, 1.0, 1.0]).then(&Affine::translation([1.0, 0.0, 0.0]));
        assert!(close(map.apply([1.0, 0.0, 0.0]), [3.0, 0.0, 0.0]));
        let other = Affine::translation([1.0, 0.0, 0.0]).then(&Affine::scaling([2.0, 1.0, 1.0]));
        assert!(close(other.apply([1.0, 0.0, 0.0]), [4.0, 0.0, 0.0]));
    }

    #[test]
    fn inverse_round_trips() {
        let map = Affine::euler_deg(20.0, -35.0, 70.0)
            .then(&Affine::scaling([2.0, -1.0, 0.5]))
            .then(&Affine::translation([1.0, 2.0, 3.0]));
        let inv = map.inverse().unwrap();
        let p = [0.3, -0.7, 1.9];
        assert!(close(inv.apply(map.apply(p)), p));
        assert!(Affine::scaling([1.0, 0.0, 1.0]).inverse().is_none());
    }

    #[test]
    fn rotation_about_a_line_keeps_the_line_fixed() {
        let map = Affine::rotation_about(2, [3.0, 0.0, 0.0], 90.0);
        assert!(close(map.apply([3.0, 0.0, 5.0]), [3.0, 0.0, 5.0]));
        assert!(close(map.apply([4.0, 0.0, 0.0]), [3.0, 1.0, 0.0]));
    }

    #[test]
    fn mirror_reflects_across_the_offset_plane() {
        let map = Affine::mirror(2, 1.0);
        assert!(close(map.apply([0.0, 0.0, 3.0]), [0.0, 0.0, -1.0]));
        assert!(map.is_diagonal() && !map.is_translation());
        assert!(map.preserves_prefix(3) && !map.preserves_prefix(2));
    }

    #[test]
    fn euler_matches_the_x_then_y_then_z_convention() {
        // Rz(90) after Rx(90): +y -> +z (Rx), then +z stays +z (Rz).
        let map = Affine::euler_deg(90.0, 0.0, 90.0);
        assert!(close(map.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]));
        // +x -> +x (Rx) -> +y (Rz).
        assert!(close(map.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        assert!(Affine::euler_deg(0.0, 0.0, 30.0).preserves_prefix(2));
        assert!(!Affine::euler_deg(10.0, 0.0, 0.0).preserves_prefix(2));
        assert!(Affine::euler_deg(360.0, 0.0, 0.0).preserves_prefix(2));
    }

    #[test]
    fn restricted_drops_out_of_prefix_effects() {
        let map = Affine::translation([1.0, 2.0, 3.0]).restricted(2);
        assert_eq!(map.offset, [1.0, 2.0, 0.0]);
        assert!(map.is_translation());
    }
}
