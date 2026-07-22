//! Small fixed-size vector/transform helpers shared by the builder and
//! the sample-time kernels. Everything is plain `[f64; 3]` / row-major
//! arrays — no external math crate, matching the other `*_model_core`
//! crates.

pub type Vec3 = [f64; 3];

#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    if n == 0.0 { a } else { scale(a, 1.0 / n) }
}

/// An orthonormal local frame: origin plus basis rows x, y, z.
/// `to_world(p) = origin + p0*x + p1*y + p2*z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub origin: Vec3,
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
}

impl Frame {
    pub const IDENTITY: Frame = Frame {
        origin: [0.0; 3],
        x: [1.0, 0.0, 0.0],
        y: [0.0, 1.0, 0.0],
        z: [0.0, 0.0, 1.0],
    };

    /// Build a right-handed orthonormal frame from an axis direction and
    /// an approximate reference x direction (Gram-Schmidt; falls back to
    /// an arbitrary perpendicular when the reference is parallel to the
    /// axis).
    pub fn from_axis_ref(origin: Vec3, axis: Vec3, x_ref: Vec3) -> Frame {
        let z = normalize(axis);
        let mut x = sub(x_ref, scale(z, dot(x_ref, z)));
        if dot(x, x) < 1e-24 {
            // Reference parallel to the axis: pick the world axis least
            // aligned with z.
            let pick = if z[0].abs() < z[1].abs() && z[0].abs() < z[2].abs() {
                [1.0, 0.0, 0.0]
            } else if z[1].abs() < z[2].abs() {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            x = sub(pick, scale(z, dot(pick, z)));
        }
        let x = normalize(x);
        let y = cross(z, x);
        Frame { origin, x, y, z }
    }

    #[inline]
    pub fn to_world(&self, p: Vec3) -> Vec3 {
        [
            self.origin[0] + p[0] * self.x[0] + p[1] * self.y[0] + p[2] * self.z[0],
            self.origin[1] + p[0] * self.x[1] + p[1] * self.y[1] + p[2] * self.z[1],
            self.origin[2] + p[0] * self.x[2] + p[1] * self.y[2] + p[2] * self.z[2],
        ]
    }

    /// World point into frame coordinates.
    #[inline]
    pub fn to_local(&self, p: Vec3) -> Vec3 {
        let d = sub(p, self.origin);
        [dot(d, self.x), dot(d, self.y), dot(d, self.z)]
    }

    /// World direction into frame coordinates (no translation).
    #[inline]
    pub fn dir_to_local(&self, d: Vec3) -> Vec3 {
        [dot(d, self.x), dot(d, self.y), dot(d, self.z)]
    }

    /// Frame direction into world coordinates.
    #[inline]
    pub fn dir_to_world(&self, d: Vec3) -> Vec3 {
        [
            d[0] * self.x[0] + d[1] * self.y[0] + d[2] * self.z[0],
            d[0] * self.x[1] + d[1] * self.y[1] + d[2] * self.z[1],
            d[0] * self.x[2] + d[1] * self.y[2] + d[2] * self.z[2],
        ]
    }

    pub fn flat(&self) -> [f64; 12] {
        let mut out = [0.0; 12];
        out[0..3].copy_from_slice(&self.origin);
        out[3..6].copy_from_slice(&self.x);
        out[6..9].copy_from_slice(&self.y);
        out[9..12].copy_from_slice(&self.z);
        out
    }

    pub fn from_flat(f: &[f64; 12]) -> Frame {
        Frame {
            origin: [f[0], f[1], f[2]],
            x: [f[3], f[4], f[5]],
            y: [f[6], f[7], f[8]],
            z: [f[9], f[10], f[11]],
        }
    }
}

/// A row-major 3x4 affine transform: `out = M[..,0..3] * p + M[..,3]`.
/// Instance transforms are rigid plus optional uniform scale, which keeps
/// the inverse exact and cheap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine(pub [f64; 12]);

impl Affine {
    pub const IDENTITY: Affine =
        Affine([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    #[inline]
    pub fn apply(&self, p: Vec3) -> Vec3 {
        let m = &self.0;
        [
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
        ]
    }

    /// Compose `self` after `other`: `(self * other).apply(p) =
    /// self.apply(other.apply(p))`.
    pub fn compose(&self, other: &Affine) -> Affine {
        let a = &self.0;
        let b = &other.0;
        let mut out = [0.0; 12];
        for row in 0..3 {
            for col in 0..3 {
                out[row * 4 + col] =
                    a[row * 4] * b[col] + a[row * 4 + 1] * b[4 + col] + a[row * 4 + 2] * b[8 + col];
            }
            out[row * 4 + 3] =
                a[row * 4] * b[3] + a[row * 4 + 1] * b[7] + a[row * 4 + 2] * b[11] + a[row * 4 + 3];
        }
        Affine(out)
    }

    /// Invert, requiring the linear part to be orthogonal times uniform
    /// scale (validated by the caller at build time): inverse rotation is
    /// the transpose over the squared scale.
    pub fn rigid_inverse(&self) -> Result<Affine, String> {
        let m = &self.0;
        let r0 = [m[0], m[1], m[2]];
        let r1 = [m[4], m[5], m[6]];
        let r2 = [m[8], m[9], m[10]];
        let s0 = dot(r0, r0);
        let s1 = dot(r1, r1);
        let s2 = dot(r2, r2);
        let s = (s0 + s1 + s2) / 3.0;
        if s <= 0.0 || !s.is_finite() {
            return Err("degenerate instance transform".to_string());
        }
        let tol = s * 1e-9;
        if (s0 - s).abs() > tol
            || (s1 - s).abs() > tol
            || (s2 - s).abs() > tol
            || dot(r0, r1).abs() > tol
            || dot(r0, r2).abs() > tol
            || dot(r1, r2).abs() > tol
        {
            return Err("instance transform is not rigid + uniform scale".to_string());
        }
        let inv_s = 1.0 / s;
        let t = [m[3], m[7], m[11]];
        let mut out = [0.0; 12];
        for row in 0..3 {
            for col in 0..3 {
                out[row * 4 + col] = m[col * 4 + row] * inv_s;
            }
            out[row * 4 + 3] = -(m[row] * t[0] + m[4 + row] * t[1] + m[8 + row] * t[2]) * inv_s;
        }
        Ok(Affine(out))
    }

    /// Transform an AABB `[min_x, max_x, ...]`, returning the AABB of the
    /// transformed box (standard per-component interval arithmetic).
    pub fn apply_aabb(&self, aabb: [f64; 6]) -> [f64; 6] {
        let m = &self.0;
        let mut out = [0.0; 6];
        for row in 0..3 {
            let mut lo = m[row * 4 + 3];
            let mut hi = lo;
            for col in 0..3 {
                let c = m[row * 4 + col];
                let (a, b) = (c * aabb[col * 2], c * aabb[col * 2 + 1]);
                lo += a.min(b);
                hi += a.max(b);
            }
            out[row * 2] = lo;
            out[row * 2 + 1] = hi;
        }
        out
    }
}

/// Distance from `p` to the segment `[a, b]`.
#[inline]
pub fn point_segment_dist(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let e = sub(b, a);
    let len2 = dot(e, e);
    let t = if len2 > 0.0 {
        (dot(sub(p, a), e) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    norm(sub(p, add(a, scale(e, t))))
}

/// Distance from `p` to the (filled) triangle `(v0, v1, v2)`.
pub fn point_triangle_dist(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> f64 {
    let n = cross(sub(v1, v0), sub(v2, v0));
    let n_len2 = dot(n, n);
    if n_len2 > 0.0 {
        // Projection onto the plane; inside the triangle when it sits on
        // the inner side of all three edges.
        let d_plane = dot(sub(p, v0), n) / n_len2;
        let q = sub(p, scale(n, d_plane));
        let inside = dot(cross(sub(v1, v0), sub(q, v0)), n) >= 0.0
            && dot(cross(sub(v2, v1), sub(q, v1)), n) >= 0.0
            && dot(cross(sub(v0, v2), sub(q, v2)), n) >= 0.0;
        if inside {
            return d_plane.abs() * n_len2.sqrt();
        }
    }
    point_segment_dist(p, v0, v1)
        .min(point_segment_dist(p, v1, v2))
        .min(point_segment_dist(p, v2, v0))
}

/// Merge `b` into AABB `a` (both `[min_x, max_x, min_y, ...]`).
pub fn aabb_union(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        a[0].min(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].max(b[3]),
        a[4].min(b[4]),
        a[5].max(b[5]),
    ]
}

pub const EMPTY_AABB: [f64; 6] = [
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
];
