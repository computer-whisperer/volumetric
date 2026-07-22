//! Surface views and ray-intersection kernels.
//!
//! [`SurfaceView`] is the borrowed form of [`crate::ir::Surface`] that
//! both the builder (borrowing from IR vectors) and the sample-time
//! payload reader (borrowing raw payload bytes) construct, so every
//! kernel exists exactly once.
//!
//! # Parity safety
//!
//! [`ray_hits`] reports each transversal ray–surface crossing through a
//! callback. Parity classification tolerates missing an *even* number of
//! crossings (a tangency that never pierces), but never an odd number;
//! kernels are written so the failure modes are even-count or flagged:
//! tangent-grade hits, near-apex/pole hits, and unconverged Newton all
//! set `suspect`, which makes the caller re-cast along another ray.

use crate::math::{Frame, Vec3, cross, dot, norm, normalize, sub};
use crate::nurbs::{self, SurfaceData};

/// One ray–surface intersection, in surface UV coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f64,
    pub u: f64,
    pub v: f64,
    /// This is a transversal crossing that participates in parity.
    /// Markers with `counts: false` exist only to force a re-cast:
    /// grazes and skims that never pierce the surface.
    pub counts: bool,
    /// The hit is numerically untrustworthy (tangent-grade incidence,
    /// parameterization singularity, marginal convergence).
    pub suspect: bool,
}

/// A 2D profile polyline, borrowed from IR or payload bytes.
#[derive(Clone, Copy)]
pub enum Profile2<'a> {
    Slice(&'a [[f64; 2]]),
    /// `count` uv pairs of little-endian f64 starting at `offset`.
    Raw {
        bytes: &'a [u8],
        offset: usize,
        count: usize,
    },
}

impl Profile2<'_> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Profile2::Slice(s) => s.len(),
            Profile2::Raw { count, .. } => *count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn point(&self, i: usize) -> [f64; 2] {
        match self {
            Profile2::Slice(s) => s[i],
            Profile2::Raw { bytes, offset, .. } => {
                let base = offset + i * 16;
                [f64_at(bytes, base), f64_at(bytes, base + 8)]
            }
        }
    }
}

#[inline]
pub(crate) fn f64_at(bytes: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[inline]
pub(crate) fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

#[inline]
pub(crate) fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// B-spline surface data read directly from payload bytes (layout in
/// `lib.rs`). Offsets are absolute byte positions within `bytes`.
#[derive(Clone, Copy)]
pub struct NurbsRaw<'a> {
    pub bytes: &'a [u8],
    pub degree_u: usize,
    pub degree_v: usize,
    pub nctrl_u: usize,
    pub nctrl_v: usize,
    pub knots_u_off: usize,
    pub knots_v_off: usize,
    pub ctrl_off: usize,
    /// Newton seed boxes: `seed_nu * seed_nv` records of 32 bytes
    /// (aabb 6xf32, uv center 2xf32).
    pub seeds_off: usize,
    pub seed_nu: usize,
    pub seed_nv: usize,
}

impl SurfaceData for NurbsRaw<'_> {
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
        f64_at(self.bytes, self.knots_u_off + i * 8)
    }
    fn knot_v(&self, i: usize) -> f64 {
        f64_at(self.bytes, self.knots_v_off + i * 8)
    }
    fn ctrl(&self, i: usize, j: usize) -> [f64; 4] {
        let base = self.ctrl_off + (i * self.nctrl_v + j) * 32;
        [
            f64_at(self.bytes, base),
            f64_at(self.bytes, base + 8),
            f64_at(self.bytes, base + 16),
            f64_at(self.bytes, base + 24),
        ]
    }
}

/// A B-spline surface with optional Newton seed boxes. The builder
/// borrows the owned IR surface (no seeds — build-time queries derive
/// their own seeds); the payload view carries precomputed boxes.
#[derive(Clone, Copy)]
pub enum NurbsView<'a> {
    Owned(&'a crate::ir::NurbsSurface),
    Raw(NurbsRaw<'a>),
}

impl SurfaceData for NurbsView<'_> {
    fn degree_u(&self) -> usize {
        match self {
            NurbsView::Owned(s) => s.degree_u,
            NurbsView::Raw(r) => r.degree_u,
        }
    }
    fn degree_v(&self) -> usize {
        match self {
            NurbsView::Owned(s) => s.degree_v,
            NurbsView::Raw(r) => r.degree_v,
        }
    }
    fn nctrl_u(&self) -> usize {
        match self {
            NurbsView::Owned(s) => s.nctrl_u,
            NurbsView::Raw(r) => r.nctrl_u,
        }
    }
    fn nctrl_v(&self) -> usize {
        match self {
            NurbsView::Owned(s) => s.nctrl_v,
            NurbsView::Raw(r) => r.nctrl_v,
        }
    }
    fn knot_u(&self, i: usize) -> f64 {
        match self {
            NurbsView::Owned(s) => s.knots_u[i],
            NurbsView::Raw(r) => r.knot_u(i),
        }
    }
    fn knot_v(&self, i: usize) -> f64 {
        match self {
            NurbsView::Owned(s) => s.knots_v[i],
            NurbsView::Raw(r) => r.knot_v(i),
        }
    }
    fn ctrl(&self, i: usize, j: usize) -> [f64; 4] {
        match self {
            NurbsView::Owned(s) => s.ctrl[i * s.nctrl_v + j],
            NurbsView::Raw(r) => SurfaceData::ctrl(r, i, j),
        }
    }
}

/// Borrowed surface, ready for kernels. See [`crate::ir::Surface`] for
/// the parameterizations.
#[derive(Clone, Copy)]
pub enum SurfaceView<'a> {
    Plane {
        frame: Frame,
    },
    Cylinder {
        frame: Frame,
        radius: f64,
    },
    Cone {
        frame: Frame,
        radius: f64,
        tan_half: f64,
    },
    Sphere {
        frame: Frame,
        radius: f64,
    },
    Torus {
        frame: Frame,
        major: f64,
        minor: f64,
    },
    Extrusion {
        frame: Frame,
        profile: Profile2<'a>,
    },
    Nurbs(NurbsView<'a>),
}

impl<'a> SurfaceView<'a> {
    pub fn from_ir(surface: &'a crate::ir::Surface) -> SurfaceView<'a> {
        use crate::ir::Surface as S;
        match surface {
            S::Plane { frame } => SurfaceView::Plane { frame: *frame },
            S::Cylinder { frame, radius } => SurfaceView::Cylinder {
                frame: *frame,
                radius: *radius,
            },
            S::Cone {
                frame,
                radius,
                half_angle,
            } => SurfaceView::Cone {
                frame: *frame,
                radius: *radius,
                tan_half: half_angle.tan(),
            },
            S::Sphere { frame, radius } => SurfaceView::Sphere {
                frame: *frame,
                radius: *radius,
            },
            S::Torus {
                frame,
                major,
                minor,
            } => SurfaceView::Torus {
                frame: *frame,
                major: *major,
                minor: *minor,
            },
            S::ExtrusionPolyline { frame, profile } => SurfaceView::Extrusion {
                frame: *frame,
                profile: Profile2::Slice(profile),
            },
            S::Nurbs(n) => SurfaceView::Nurbs(NurbsView::Owned(n)),
            S::Mesh(_) => {
                // Mesh faces have no UV surface; the payload dispatches
                // them to their own kernels before building a view.
                debug_assert!(false, "SurfaceView::from_ir on a mesh face");
                SurfaceView::Plane {
                    frame: Frame::IDENTITY,
                }
            }
        }
    }

    /// Evaluate the surface point at (u, v).
    pub fn eval(&self, u: f64, v: f64) -> Vec3 {
        match self {
            SurfaceView::Plane { frame } => frame.to_world([u, v, 0.0]),
            SurfaceView::Cylinder { frame, radius } => {
                frame.to_world([radius * u.cos(), radius * u.sin(), v])
            }
            SurfaceView::Cone {
                frame,
                radius,
                tan_half,
            } => {
                let r = radius + v * tan_half;
                frame.to_world([r * u.cos(), r * u.sin(), v])
            }
            SurfaceView::Sphere { frame, radius } => frame.to_world([
                radius * v.cos() * u.cos(),
                radius * v.cos() * u.sin(),
                radius * v.sin(),
            ]),
            SurfaceView::Torus {
                frame,
                major,
                minor,
            } => {
                let r = major + minor * v.cos();
                frame.to_world([r * u.cos(), r * u.sin(), minor * v.sin()])
            }
            SurfaceView::Extrusion { frame, profile } => {
                let n = profile.len();
                // Closed profiles are u-periodic; unwrapped trim UVs may
                // arrive outside [0, n-1].
                let u = if n >= 4 && profile.point(0) == profile.point(n - 1) {
                    u.rem_euclid((n - 1) as f64)
                } else {
                    u
                };
                let seg = (u.floor().max(0.0) as usize).min(n.saturating_sub(2));
                let s = (u - seg as f64).clamp(0.0, 1.0);
                let a = profile.point(seg);
                let b = profile.point(seg + 1);
                frame.to_world([a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), v])
            }
            SurfaceView::Nurbs(view) => nurbs::surface_eval(view, u, v).0,
        }
    }

    /// Closest point on the *untrimmed* surface: `(uv, 3D distance)`.
    /// Exact for the analytic types. NURBS polishes Gauss-Newton from the
    /// nearest seed boxes (payload views) or a coarse domain scan (owned
    /// views), so the result is a well-polished local minimum rather than
    /// a guarantee — meant for ranking faces by proximity (surface-color
    /// lookup), never for occupancy classification.
    pub fn closest(&self, p: Vec3) -> ([f64; 2], f64) {
        match self {
            SurfaceView::Plane { frame } => {
                let l = frame.to_local(p);
                ([l[0], l[1]], l[2].abs())
            }
            SurfaceView::Cylinder { frame, radius } => {
                let l = frame.to_local(p);
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                ([l[1].atan2(l[0]), l[2]], (rho - radius).abs())
            }
            SurfaceView::Cone {
                frame,
                radius,
                tan_half,
            } => {
                let l = frame.to_local(p);
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                let k = *tan_half;
                // Foot of the perpendicular onto the ruling line
                // rho(v) = radius + k v in the (rho, z) half-plane,
                // clamped to the apex where the real cone ends.
                let mut v = (k * (rho - radius) + l[2]) / (1.0 + k * k);
                if radius + k * v < 0.0 {
                    v = -radius / k;
                }
                let dr = rho - (radius + k * v);
                let dz = l[2] - v;
                ([l[1].atan2(l[0]), v], (dr * dr + dz * dz).sqrt())
            }
            SurfaceView::Sphere { frame, radius } => {
                let l = frame.to_local(p);
                let r = norm(l);
                let v = if r > 0.0 {
                    (l[2] / r).clamp(-1.0, 1.0).asin()
                } else {
                    0.0
                };
                ([l[1].atan2(l[0]), v], (r - radius).abs())
            }
            SurfaceView::Torus {
                frame,
                major,
                minor,
            } => {
                let l = frame.to_local(p);
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                let tube = ((rho - major) * (rho - major) + l[2] * l[2]).sqrt();
                (
                    [l[1].atan2(l[0]), l[2].atan2(rho - major)],
                    (tube - minor).abs(),
                )
            }
            SurfaceView::Extrusion { frame, profile } => {
                let l = frame.to_local(p);
                let mut best = (f64::INFINITY, 0.0f64);
                for i in 0..profile.len().saturating_sub(1) {
                    let a = profile.point(i);
                    let b = profile.point(i + 1);
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
                    if d < best.0 {
                        best = (d, i as f64 + t);
                    }
                }
                ([best.1, l[2]], best.0)
            }
            SurfaceView::Nurbs(view) => {
                // Seeds: the nearest few precomputed boxes (Raw), or a
                // coarse scan of the domain (Owned, build-side only).
                let mut seeds: [[f64; 2]; 4] = [[0.0; 2]; 4];
                let mut seed_count = 0usize;
                match view {
                    NurbsView::Raw(raw) => {
                        let mut ranked: [(f64, [f64; 2]); 4] = [(f64::INFINITY, [0.0; 2]); 4];
                        for s in 0..raw.seed_nu * raw.seed_nv {
                            let base = raw.seeds_off + s * 32;
                            let mut d2 = 0.0f64;
                            for (axis, &c) in p.iter().enumerate() {
                                let lo = f32_at(raw.bytes, base + axis * 4) as f64;
                                let hi = f32_at(raw.bytes, base + 12 + axis * 4) as f64;
                                let gap = (lo - c).max(c - hi).max(0.0);
                                d2 += gap * gap;
                            }
                            for slot in 0..ranked.len() {
                                if d2 < ranked[slot].0 {
                                    ranked[slot..].rotate_right(1);
                                    ranked[slot] = (
                                        d2,
                                        [
                                            f32_at(raw.bytes, base + 24) as f64,
                                            f32_at(raw.bytes, base + 28) as f64,
                                        ],
                                    );
                                    break;
                                }
                            }
                        }
                        for (d2, uv) in ranked {
                            if d2.is_finite() {
                                seeds[seed_count] = uv;
                                seed_count += 1;
                            }
                        }
                    }
                    NurbsView::Owned(_) => {
                        let dom = view.domain();
                        const N: usize = 9;
                        let mut ranked: [(f64, [f64; 2]); 4] = [(f64::INFINITY, [0.0; 2]); 4];
                        for i in 0..=N {
                            for j in 0..=N {
                                let u = dom[0] + (dom[1] - dom[0]) * i as f64 / N as f64;
                                let v = dom[2] + (dom[3] - dom[2]) * j as f64 / N as f64;
                                let (q, _, _) = nurbs::surface_eval(view, u, v);
                                let d2 = {
                                    let r = sub(q, p);
                                    dot(r, r)
                                };
                                for slot in 0..ranked.len() {
                                    if d2 < ranked[slot].0 {
                                        ranked[slot..].rotate_right(1);
                                        ranked[slot] = (d2, [u, v]);
                                        break;
                                    }
                                }
                            }
                        }
                        for (d2, uv) in ranked {
                            if d2.is_finite() {
                                seeds[seed_count] = uv;
                                seed_count += 1;
                            }
                        }
                    }
                }
                let mut best = ([0.0; 2], f64::INFINITY);
                for seed in &seeds[..seed_count] {
                    let (uv, d) = nurbs::surface_closest(view, p, *seed);
                    if d < best.1 {
                        best = (uv, d);
                    }
                }
                best
            }
        }
    }

    /// Report every ray–surface crossing along `origin + t * dir` to
    /// `visit(hit)`. `dir` need not be unit length. `eps` is the solid's
    /// 3D suspicion tolerance; hits of any `t` (either sign) are
    /// reported — the caller applies its own `t > t_eps` counting rule
    /// so that on-surface queries can be recognized.
    pub fn ray_hits(&self, origin: Vec3, dir: Vec3, eps: f64, visit: &mut dyn FnMut(Hit)) {
        match self {
            SurfaceView::Plane { frame } => {
                let o = frame.to_local(origin);
                let d = frame.dir_to_local(dir);
                let denom = d[2];
                let dir_scale = norm(d);
                if denom.abs() < 1e-9 * dir_scale {
                    // Ray in-plane: no transversal crossing; suspect only
                    // when the ray actually skims the surface.
                    if o[2].abs() < eps {
                        visit(Hit {
                            t: 0.0,
                            u: o[0],
                            v: o[1],
                            counts: false,
                            suspect: true,
                        });
                    }
                    return;
                }
                let t = -o[2] / denom;
                visit(Hit {
                    t,
                    u: o[0] + t * d[0],
                    v: o[1] + t * d[1],
                    counts: true,
                    suspect: denom.abs() < 1e-4 * dir_scale,
                });
            }
            SurfaceView::Cylinder { frame, radius } => {
                let o = frame.to_local(origin);
                let d = frame.dir_to_local(dir);
                let a = d[0] * d[0] + d[1] * d[1];
                let dir_scale2 = dot(d, d);
                if a < 1e-18 * dir_scale2 {
                    // Parallel to the axis: never transversal.
                    let rho = (o[0] * o[0] + o[1] * o[1]).sqrt();
                    if (rho - radius).abs() < eps {
                        visit(Hit {
                            t: 0.0,
                            u: o[1].atan2(o[0]),
                            v: o[2],
                            counts: false,
                            suspect: true,
                        });
                    }
                    return;
                }
                let b = o[0] * d[0] + o[1] * d[1];
                let c = o[0] * o[0] + o[1] * o[1] - radius * radius;
                let disc = b * b - a * c;
                if disc <= 0.0 {
                    // Tangent-grade approach still deserves a re-cast.
                    if disc > -eps * eps * a {
                        let t = -b / a;
                        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                        visit(Hit {
                            t,
                            u: p[1].atan2(p[0]),
                            v: p[2],
                            counts: false,
                            suspect: true,
                        });
                    }
                    return;
                }
                let sq = disc.sqrt();
                // Tangency scale: how close the two roots are relative to
                // the suspicion tolerance along the ray.
                let tangent = sq * 2.0 < eps * a.sqrt();
                for t in [(-b - sq) / a, (-b + sq) / a] {
                    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                    visit(Hit {
                        t,
                        u: p[1].atan2(p[0]),
                        v: p[2],
                        counts: true,
                        suspect: tangent,
                    });
                }
            }
            SurfaceView::Cone {
                frame,
                radius,
                tan_half,
            } => {
                let o = frame.to_local(origin);
                let d = frame.dir_to_local(dir);
                // sqrt(x^2 + y^2) = radius + z * tan_half, squared form;
                // reject mirror-cone roots (negative radius) after solving.
                let k = *tan_half;
                let a = d[0] * d[0] + d[1] * d[1] - k * k * d[2] * d[2];
                let rb = radius + o[2] * k;
                let b = o[0] * d[0] + o[1] * d[1] - k * d[2] * rb;
                let c = o[0] * o[0] + o[1] * o[1] - rb * rb;
                let dir_scale2 = dot(d, d);
                let mut roots = [0.0f64; 2];
                let mut count = 0;
                let mut tangent = false;
                let mut marker_only = false;
                if a.abs() < 1e-14 * dir_scale2 {
                    // Ray parallel to one ruling: linear equation. When
                    // it also lies on the surface (b, c both ~0) emit a
                    // skim marker like the cylinder's axis-parallel case.
                    if b.abs() > 1e-14 * dir_scale2 {
                        roots[0] = -c / (2.0 * b);
                        count = 1;
                    } else if c.abs() < eps * eps {
                        visit(Hit {
                            t: 0.0,
                            u: o[1].atan2(o[0]),
                            v: o[2],
                            counts: false,
                            suspect: true,
                        });
                        return;
                    }
                } else {
                    let disc = b * b - a * c;
                    if disc <= 0.0 {
                        if disc > -eps * eps * a.abs() {
                            roots[0] = -b / a;
                            count = 1;
                            tangent = true;
                            marker_only = true;
                        }
                    } else {
                        let sq = disc.sqrt();
                        roots = [(-b - sq) / a, (-b + sq) / a];
                        count = 2;
                        tangent = sq * 2.0 < eps * a.abs().sqrt();
                    }
                }
                for &t in &roots[..count] {
                    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                    let rv = radius + p[2] * k;
                    if rv < 0.0 {
                        continue; // mirror cone
                    }
                    visit(Hit {
                        t,
                        u: p[1].atan2(p[0]),
                        v: p[2],
                        counts: !marker_only,
                        // The apex is a parameterization singularity.
                        suspect: tangent || rv < eps,
                    });
                }
            }
            SurfaceView::Sphere { frame, radius } => {
                let o = frame.to_local(origin);
                let d = frame.dir_to_local(dir);
                let a = dot(d, d);
                let b = dot(o, d);
                let c = dot(o, o) - radius * radius;
                let disc = b * b - a * c;
                if disc <= 0.0 {
                    if disc > -eps * eps * a {
                        let t = -b / a;
                        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                        visit(sphere_hit(p, *radius, t, false, true));
                    }
                    return;
                }
                let sq = disc.sqrt();
                let tangent = sq * 2.0 < eps * a.sqrt();
                for t in [(-b - sq) / a, (-b + sq) / a] {
                    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                    visit(sphere_hit(p, *radius, t, true, tangent));
                }
            }
            SurfaceView::Torus {
                frame,
                major,
                minor,
            } => {
                torus_hits(frame, *major, *minor, origin, dir, eps, visit);
            }
            SurfaceView::Extrusion { frame, profile } => {
                let o = frame.to_local(origin);
                let d = frame.dir_to_local(dir);
                let dir_scale = norm(d);
                let n = profile.len();
                if n < 2 {
                    return;
                }
                let mut prev = profile.point(0);
                for i in 1..n {
                    let cur = profile.point(i);
                    // Strip plane normal (in frame xy): perpendicular to
                    // the segment, z-free.
                    let e = [cur[0] - prev[0], cur[1] - prev[1]];
                    let nrm = [e[1], -e[0]];
                    let seg_len = (e[0] * e[0] + e[1] * e[1]).sqrt();
                    if seg_len == 0.0 {
                        prev = cur;
                        continue;
                    }
                    let denom = nrm[0] * d[0] + nrm[1] * d[1];
                    let dist = nrm[0] * (o[0] - prev[0]) + nrm[1] * (o[1] - prev[1]);
                    if denom.abs() < 1e-9 * dir_scale * seg_len {
                        if dist.abs() < eps * seg_len {
                            visit(Hit {
                                t: 0.0,
                                u: (i - 1) as f64 + 0.5,
                                v: o[2],
                                counts: false,
                                suspect: true,
                            });
                        }
                        prev = cur;
                        continue;
                    }
                    let t = -dist / denom;
                    let px = o[0] + t * d[0];
                    let py = o[1] + t * d[1];
                    let s = ((px - prev[0]) * e[0] + (py - prev[1]) * e[1]) / (seg_len * seg_len);
                    // Half-open [0, 1) so a hit on a shared segment
                    // endpoint counts exactly once. Grazing incidence
                    // makes s unreliable, so it earns a re-cast.
                    let end_eps = eps / seg_len;
                    let grazing = denom.abs() < 1e-4 * dir_scale * seg_len;
                    if (0.0..1.0).contains(&s) {
                        visit(Hit {
                            t,
                            u: (i - 1) as f64 + s,
                            v: o[2] + t * d[2],
                            counts: true,
                            suspect: grazing || s < end_eps || s > 1.0 - end_eps,
                        });
                    } else if s >= -end_eps && s < 1.0 + end_eps {
                        // Missed just outside the half-open window: FP
                        // rounding at a shared vertex can push the hit
                        // off both adjacent segments — flag it so the
                        // re-cast resolves the parity.
                        visit(Hit {
                            t,
                            u: (i - 1) as f64 + s.clamp(0.0, 1.0),
                            v: o[2] + t * d[2],
                            counts: false,
                            suspect: true,
                        });
                    }
                    prev = cur;
                }
            }
            SurfaceView::Nurbs(view) => match view {
                NurbsView::Raw(raw) => nurbs_ray_hits(raw, origin, dir, eps, visit),
                NurbsView::Owned(_) => {
                    // Build-side classification goes through the payload
                    // (the real path); an owned view has no seed boxes.
                    debug_assert!(false, "ray_hits on an owned NURBS view");
                }
            },
        }
    }
}

fn sphere_hit(p: Vec3, radius: f64, t: f64, counts: bool, tangent: bool) -> Hit {
    let v = (p[2] / radius).clamp(-1.0, 1.0).asin();
    Hit {
        t,
        u: p[1].atan2(p[0]),
        v,
        counts,
        // Near the poles u degenerates; classify but ask for a re-cast.
        suspect: tangent || p[2].abs() > radius * 0.999_999,
    }
}

/// Torus crossings by parity-safe root isolation: the quartic's sign
/// changes on a uniform t-grid are bisected to convergence. Grid-width
/// root pairs can be missed *together* (even count — parity-safe); a
/// near-tangent double root close to a grid point sets `suspect` via the
/// small-|f| check.
fn torus_hits(
    frame: &Frame,
    major: f64,
    minor: f64,
    origin: Vec3,
    dir: Vec3,
    eps: f64,
    visit: &mut dyn FnMut(Hit),
) {
    let o = frame.to_local(origin);
    let d0 = frame.dir_to_local(dir);
    let dir_scale = norm(d0);
    if dir_scale == 0.0 {
        return;
    }
    let d = normalize(d0);

    // Quartic f(t) = (|p|^2 + R^2 - r^2)^2 - 4 R^2 (x^2 + y^2), p = o + t d.
    let rr = major * major;
    let sr = minor * minor;
    let f = |t: f64| {
        let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        let q = dot(p, p) + rr - sr;
        q * q - 4.0 * rr * (p[0] * p[0] + p[1] * p[1])
    };

    // The torus fits in a sphere of radius R + r around its center; only
    // t within that window can contain roots.
    let reach = major + minor;
    let b = dot(o, d);
    let c = dot(o, o) - reach * reach;
    let disc = b * b - c;
    if disc <= 0.0 {
        return;
    }
    let sq = disc.sqrt();
    // Expand the window so roots exactly on the enclosing sphere (rays
    // near the equator plane hit it) land strictly inside the scan.
    let margin = sq * 0.02 + eps;
    let (t_lo, t_hi) = (-b - sq - margin, -b + sq + margin);

    const STEPS: usize = 96;
    let dt = (t_hi - t_lo) / STEPS as f64;
    // |f| scale for the tangent-grade test: f is quartic in distance, so
    // an eps-deep graze changes it by ~ eps * |gradient| ~ eps * scale^3.
    let f_scale = {
        let s = reach.max(norm(sub(o, [0.0; 3])));
        s * s * s * eps * 8.0
    };
    let mut prev_t = t_lo;
    let mut prev_f = f(prev_t);
    for i in 1..=STEPS {
        let cur_t = t_lo + dt * i as f64;
        let cur_f = f(cur_t);
        if (prev_f <= 0.0) != (cur_f <= 0.0) {
            // Bisect to a root.
            let (mut a_t, mut a_f, mut b_t) = (prev_t, prev_f, cur_t);
            for _ in 0..60 {
                let m = (a_t + b_t) * 0.5;
                let mf = f(m);
                if (a_f <= 0.0) != (mf <= 0.0) {
                    b_t = m;
                } else {
                    a_t = m;
                    a_f = mf;
                }
            }
            let t = (a_t + b_t) * 0.5;
            let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let u = p[1].atan2(p[0]);
            let v = (p[2]).atan2(rho - major);
            // Report t in the caller's (unnormalized) direction scale.
            visit(Hit {
                t: t / dir_scale,
                u,
                v,
                counts: true,
                // Distance-to-axis degeneracy: near the torus center line
                // u is unstable.
                suspect: rho < minor.max(eps),
            });
        } else if prev_f.abs() < f_scale && cur_f.abs() < f_scale {
            // Grazing the surface without a detected sign change: the
            // parity is fine either way, but ask for a re-cast.
            let t = prev_t;
            let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            visit(Hit {
                t: t / dir_scale,
                u: p[1].atan2(p[0]),
                v: p[2].atan2(rho - major),
                counts: false,
                suspect: true,
            });
        }
        prev_t = cur_t;
        prev_f = cur_f;
    }
}

const MAX_NURBS_ROOTS: usize = 32;

/// Ray–NURBS intersection: slab-test the precomputed seed boxes, run a
/// (u, v, t) Newton from each hit box's UV center, dedupe converged
/// roots. Unconverged seeds and singular Jacobians report a suspect
/// zero-count hit so the caller re-casts.
fn nurbs_ray_hits(
    raw: &NurbsRaw<'_>,
    origin: Vec3,
    dir: Vec3,
    eps: f64,
    visit: &mut dyn FnMut(Hit),
) {
    let seed_count = raw.seed_nu * raw.seed_nv;
    let domain = raw.domain();
    let du_margin = (domain[1] - domain[0]) * 1e-6;
    let dv_margin = (domain[3] - domain[2]) * 1e-6;

    // Roots carry their 3D position: dedupe compares positions, not UV,
    // so a seam alias on a closed surface (u=0 vs u=u_max, same point)
    // merges while two walls of a thin fold (distinct points) don't.
    let mut roots: [(f64, [f64; 3]); MAX_NURBS_ROOTS] = [(0.0, [0.0; 3]); MAX_NURBS_ROOTS];
    let mut root_count = 0usize;
    let inv_dir: Vec3 = core::array::from_fn(|i| 1.0 / dir[i]);

    for s in 0..seed_count {
        let base = raw.seeds_off + s * 32;
        // Slab test against the seed box (handles negative components).
        let mut t_min = f64::NEG_INFINITY;
        let mut t_max = f64::INFINITY;
        for axis in 0..3 {
            let lo = f32_at(raw.bytes, base + axis * 4) as f64;
            let hi = f32_at(raw.bytes, base + 12 + axis * 4) as f64;
            let (mut a, mut b) = (
                (lo - origin[axis]) * inv_dir[axis],
                (hi - origin[axis]) * inv_dir[axis],
            );
            if a > b {
                core::mem::swap(&mut a, &mut b);
            }
            t_min = t_min.max(a);
            t_max = t_max.min(b);
        }
        if t_min > t_max {
            continue;
        }

        let mut u = f32_at(raw.bytes, base + 24) as f64;
        let mut v = f32_at(raw.bytes, base + 28) as f64;
        let mut t = (t_min + t_max) * 0.5;
        if !t.is_finite() {
            t = 0.0;
        }

        let mut converged = false;
        let mut singular = false;
        let mut best_res = f64::INFINITY;
        for _ in 0..24 {
            let (p, su, sv) = nurbs::surface_eval(raw, u, v);
            let r = [
                p[0] - origin[0] - t * dir[0],
                p[1] - origin[1] - t * dir[1],
                p[2] - origin[2] - t * dir[2],
            ];
            let res = norm(r);
            best_res = best_res.min(res);
            // Solve J * [du, dv, dt] = -r with J = [su, sv, -dir].
            let det = dot(su, cross(sv, [-dir[0], -dir[1], -dir[2]]));
            if det.abs() < 1e-14 {
                singular = res < eps * 4.0;
                break;
            }
            let inv = 1.0 / det;
            let neg_d = [-dir[0], -dir[1], -dir[2]];
            let du = dot([-r[0], -r[1], -r[2]], cross(sv, neg_d)) * inv;
            let dv = dot(su, cross([-r[0], -r[1], -r[2]], neg_d)) * inv;
            let dt = dot(su, cross(sv, [-r[0], -r[1], -r[2]])) * inv;
            u += du;
            v += dv;
            t += dt;
            if u < domain[0] - du_margin * 1e5
                || u > domain[1] + du_margin * 1e5
                || v < domain[2] - dv_margin * 1e5
                || v > domain[3] + dv_margin * 1e5
            {
                break; // wandered far out of the patch
            }
            if res < eps * 1e-3 && du.abs() < du_margin && dv.abs() < dv_margin {
                converged = true;
                break;
            }
        }
        if singular {
            visit(Hit {
                t: 0.0,
                u,
                v,
                counts: false,
                suspect: true,
            });
            continue;
        }
        if !converged {
            // Came near the surface but never converged: a crossing may
            // be hiding here. Flag for a re-cast instead of silently
            // dropping it (parity contract: missed crossings are either
            // even-count or flagged). Seeds whose boxes the ray merely
            // clips stay far from the surface and don't fire this.
            if best_res < eps * 8.0 {
                visit(Hit {
                    t: 0.0,
                    u,
                    v,
                    counts: false,
                    suspect: true,
                });
            }
            continue;
        }
        if u < domain[0] - du_margin
            || u > domain[1] + du_margin
            || v < domain[2] - dv_margin
            || v > domain[3] + dv_margin
        {
            continue;
        }
        let (pos, su, sv) = nurbs::surface_eval(raw, u, v);
        // Dedupe by (t, 3D position): different seeds converging to the
        // same physical crossing, including seam aliases.
        let dup = roots[..root_count]
            .iter()
            .any(|&(rt, rp)| (rt - t).abs() < eps && norm(sub(rp, pos)) < eps * 4.0);
        if dup {
            continue;
        }
        if root_count == MAX_NURBS_ROOTS {
            // Can't track more distinct roots — classification unsafe.
            visit(Hit {
                t,
                u,
                v,
                counts: false,
                suspect: true,
            });
            continue;
        }
        roots[root_count] = (t, pos);
        root_count += 1;

        // Tangency check: surface normal vs ray direction.
        let n = cross(su, sv);
        let n_len = norm(n);
        let grazing = n_len == 0.0 || dot(n, dir).abs() < 1e-6 * n_len * norm(dir);
        visit(Hit {
            t,
            u,
            v,
            counts: true,
            suspect: grazing,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Frame;

    fn collect(surface: &SurfaceView, origin: Vec3, dir: Vec3) -> Vec<Hit> {
        let mut hits = Vec::new();
        surface.ray_hits(origin, dir, 1e-9, &mut |h| hits.push(h));
        hits
    }

    #[test]
    fn plane_hit_uv() {
        let s = SurfaceView::Plane {
            frame: Frame::IDENTITY,
        };
        let hits = collect(&s, [0.5, -0.25, 2.0], [0.0, 0.0, -1.0]);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].t - 2.0).abs() < 1e-12);
        assert!((hits[0].u - 0.5).abs() < 1e-12);
        assert!((hits[0].v + 0.25).abs() < 1e-12);
        assert!(!hits[0].suspect);
    }

    #[test]
    fn cylinder_two_crossings() {
        let s = SurfaceView::Cylinder {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        let hits = collect(&s, [-3.0, 0.0, 0.5], [1.0, 0.0, 0.0]);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].t - 2.0).abs() < 1e-9);
        assert!((hits[1].t - 4.0).abs() < 1e-9);
        assert!((hits[0].v - 0.5).abs() < 1e-12);
        // First hit at (-1, 0): u = π.
        assert!((hits[0].u.abs() - core::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn cylinder_axis_parallel_ray_no_hits() {
        let s = SurfaceView::Cylinder {
            frame: Frame::IDENTITY,
            radius: 1.0,
        };
        assert!(collect(&s, [0.2, 0.0, -5.0], [0.0, 0.0, 1.0]).is_empty());
        // Skimming the wall: one suspect marker.
        let hits = collect(&s, [1.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].suspect);
    }

    #[test]
    fn sphere_hits_and_uv() {
        let s = SurfaceView::Sphere {
            frame: Frame::IDENTITY,
            radius: 2.0,
        };
        let hits = collect(&s, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(hits.len(), 2);
        let outward: Vec<_> = hits.iter().filter(|h| h.t > 0.0).collect();
        assert_eq!(outward.len(), 1);
        assert!((outward[0].t - 2.0).abs() < 1e-9);
        assert!(outward[0].u.abs() < 1e-9 && outward[0].v.abs() < 1e-9);
    }

    #[test]
    fn cone_mirror_rejected() {
        // Apex at z = -1 (radius 1 at z = 0, half angle 45°).
        let s = SurfaceView::Cone {
            frame: Frame::IDENTITY,
            radius: 1.0,
            tan_half: 1.0,
        };
        // Ray through both nappes at z = -3: the mirror cone (r_v < 0)
        // must not report crossings.
        let hits = collect(&s, [-10.0, 0.0, -3.0], [1.0, 0.0, 0.0]);
        assert!(hits.is_empty(), "{hits:?}");
        // And through the real cone at z = 1: two crossings at x = ±2.
        let hits = collect(&s, [-10.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].t - 8.0).abs() < 1e-9);
        assert!((hits[1].t - 12.0).abs() < 1e-9);
    }

    #[test]
    fn torus_four_crossings() {
        let s = SurfaceView::Torus {
            frame: Frame::IDENTITY,
            major: 2.0,
            minor: 0.5,
        };
        let hits = collect(&s, [-4.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(hits.len(), 4);
        let mut ts: Vec<f64> = hits.iter().map(|h| h.t).collect();
        ts.sort_by(f64::total_cmp);
        for (t, expect) in ts.iter().zip([1.5, 2.5, 5.5, 6.5]) {
            assert!((t - expect).abs() < 1e-7, "t={t} expect={expect}");
        }
        // Off-plane ray missing the tube.
        assert!(collect(&s, [-4.0, 0.0, 0.75], [1.0, 0.0, 0.0]).is_empty());
    }

    #[test]
    fn extrusion_square_profile() {
        // A unit-square profile (open at the top) swept along z.
        let profile = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let s = SurfaceView::Extrusion {
            frame: Frame::IDENTITY,
            profile: Profile2::Slice(&profile),
        };
        // Ray crossing the y=0 wall (segment 0) interior and the x=1
        // wall (segment 1) at y=0.3, both away from segment endpoints.
        let hits = collect(&s, [0.7, -1.0, 3.0], [0.3, 1.3, 0.0]);
        assert_eq!(hits.len(), 2);
        assert!(
            (hits[0].u - (0.7 + 0.3 / 1.3)).abs() < 1e-9,
            "u={}",
            hits[0].u
        );
        assert!((hits[0].v - 3.0).abs() < 1e-12);
        assert!((hits[1].u - 1.3).abs() < 1e-9, "x=1 wall at y=0.3");
    }

    #[test]
    fn eval_matches_parameterization() {
        let torus = SurfaceView::Torus {
            frame: Frame::IDENTITY,
            major: 2.0,
            minor: 0.5,
        };
        let p = torus.eval(0.0, core::f64::consts::PI);
        assert!((p[0] - 1.5).abs() < 1e-12 && p[1].abs() < 1e-12 && p[2].abs() < 1e-12);
        let sphere = SurfaceView::Sphere {
            frame: Frame::IDENTITY,
            radius: 2.0,
        };
        let p = sphere.eval(0.0, core::f64::consts::FRAC_PI_2);
        assert!(p[2] > 2.0 - 1e-12);
    }
}
