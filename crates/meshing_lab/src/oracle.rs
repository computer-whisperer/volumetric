//! Analytical ground-truth shapes for meshing research.
//!
//! Each shape provides two strictly separated views:
//!
//! - [`OracleShape::is_inside`]: the binary occupancy sampler. This is the
//!   *only* thing meshing algorithms under test may touch.
//! - [`OracleShape::truth`]: exact closed-form surface geometry (nearest smooth
//!   face, its outward normal, distance to the nearest sharp feature). This is
//!   benchmark-side only and must never leak into an algorithm under test.
//!
//! Truth is computed analytically, never by sampling, so it stays correct at
//! any resolution.

use glam::{DQuat, DVec3};

/// Exact surface information at a query point near the surface.
#[derive(Clone, Debug)]
pub struct SurfaceTruth {
    /// Outward unit normal of the smooth face nearest to the query point.
    pub normal: DVec3,
    /// Closest point on the surface.
    pub closest: DVec3,
    /// Identifier of the smooth face the closest point lies on. Faces are
    /// maximal smooth regions: a box has 6, a capped cylinder 3, a sphere 1.
    pub face_id: u32,
    /// Distance to the nearest sharp feature curve (box edge, cylinder rim).
    /// `f64::INFINITY` for shapes with no sharp features.
    pub dist_to_sharp: f64,
}

pub trait OracleShape: Sync {
    fn name(&self) -> String;

    /// Binary occupancy sample. The only shape access an algorithm under test
    /// is allowed.
    fn is_inside(&self, p: DVec3) -> bool;

    /// Exact geometric truth at `p`. Benchmark-side only.
    fn truth(&self, p: DVec3) -> SurfaceTruth;

    /// World-space AABB enclosing the shape (without margin).
    fn world_bounds(&self) -> (DVec3, DVec3);
}

// =============================================================================
// Box
// =============================================================================

/// Axis-aligned box centered at the origin. Combine with [`Rotated`] to avoid
/// grid-aligned features.
pub struct BoxShape {
    pub half: DVec3,
}

impl OracleShape for BoxShape {
    fn name(&self) -> String {
        format!(
            "box {}x{}x{}",
            2.0 * self.half.x,
            2.0 * self.half.y,
            2.0 * self.half.z
        )
    }

    fn is_inside(&self, p: DVec3) -> bool {
        p.x.abs() <= self.half.x && p.y.abs() <= self.half.y && p.z.abs() <= self.half.z
    }

    fn truth(&self, p: DVec3) -> SurfaceTruth {
        let h = self.half;
        // Per-axis signed distance to the slab boundary (negative inside).
        let d = p.abs() - h;
        // Nearest face: the axis whose boundary plane is closest for inside
        // points, or most violated for outside points. For points beyond an
        // edge/corner the nearest face is ambiguous; those queries land in the
        // near-sharp bucket where face normals are not meaningful anyway.
        let axis = if d.x >= d.y && d.x >= d.z {
            0
        } else if d.y >= d.z {
            1
        } else {
            2
        };
        let sign = if p[axis] >= 0.0 { 1.0 } else { -1.0 };
        let mut normal = DVec3::ZERO;
        normal[axis] = sign;
        let mut closest = p.clamp(-h, h);
        closest[axis] = sign * h[axis];
        let face_id = (axis as u32) * 2 + if sign > 0.0 { 0 } else { 1 };

        // Distance to the nearest of the 12 edge segments.
        let mut dist_to_sharp = f64::INFINITY;
        for edge_axis in 0..3usize {
            let a = (edge_axis + 1) % 3;
            let b = (edge_axis + 2) % 3;
            for sa in [-1.0, 1.0] {
                for sb in [-1.0, 1.0] {
                    let mut start = DVec3::ZERO;
                    start[a] = sa * h[a];
                    start[b] = sb * h[b];
                    let mut end = start;
                    start[edge_axis] = -h[edge_axis];
                    end[edge_axis] = h[edge_axis];
                    dist_to_sharp = dist_to_sharp.min(point_segment_distance(p, start, end));
                }
            }
        }

        SurfaceTruth {
            normal,
            closest,
            face_id,
            dist_to_sharp,
        }
    }

    fn world_bounds(&self) -> (DVec3, DVec3) {
        (-self.half, self.half)
    }
}

// =============================================================================
// Sphere
// =============================================================================

/// Sphere centered at the origin: the smooth control shape. No sharp features,
/// so any feature discriminator must stay silent on it.
pub struct SphereShape {
    pub radius: f64,
}

impl OracleShape for SphereShape {
    fn name(&self) -> String {
        format!("sphere r={}", self.radius)
    }

    fn is_inside(&self, p: DVec3) -> bool {
        p.length_squared() <= self.radius * self.radius
    }

    fn truth(&self, p: DVec3) -> SurfaceTruth {
        let normal = p.try_normalize().unwrap_or(DVec3::Z);
        SurfaceTruth {
            normal,
            closest: normal * self.radius,
            face_id: 0,
            dist_to_sharp: f64::INFINITY,
        }
    }

    fn world_bounds(&self) -> (DVec3, DVec3) {
        (DVec3::splat(-self.radius), DVec3::splat(self.radius))
    }
}

// =============================================================================
// Cylinder
// =============================================================================

/// Capped cylinder along the Z axis, centered at the origin. Mixes smooth
/// curvature (the side) with sharp circular rims where the caps meet the side.
pub struct CylinderShape {
    pub radius: f64,
    pub half_height: f64,
}

impl OracleShape for CylinderShape {
    fn name(&self) -> String {
        format!("cylinder r={} h={}", self.radius, 2.0 * self.half_height)
    }

    fn is_inside(&self, p: DVec3) -> bool {
        p.x * p.x + p.y * p.y <= self.radius * self.radius && p.z.abs() <= self.half_height
    }

    fn truth(&self, p: DVec3) -> SurfaceTruth {
        let r = self.radius;
        let h = self.half_height;
        let rho = (p.x * p.x + p.y * p.y).sqrt();
        let radial = if rho > 1e-12 {
            DVec3::new(p.x / rho, p.y / rho, 0.0)
        } else {
            DVec3::X
        };
        // Signed distances to the side surface and the cap planes (negative
        // inside). The larger one is the nearer surface for inside points and
        // the more violated one for outside points; ties beyond the rim are
        // ambiguous and land in the near-sharp bucket.
        let d_side = rho - r;
        let d_cap = p.z.abs() - h;

        let (normal, closest, face_id) = if d_side >= d_cap {
            let closest = radial * r + DVec3::new(0.0, 0.0, p.z.clamp(-h, h));
            (radial, closest, 0)
        } else {
            let sign = if p.z >= 0.0 { 1.0 } else { -1.0 };
            let rho_clamped = rho.min(r);
            let closest = radial * rho_clamped + DVec3::new(0.0, 0.0, sign * h);
            let face_id = if sign > 0.0 { 1 } else { 2 };
            (DVec3::new(0.0, 0.0, sign), closest, face_id)
        };

        // Distance to the nearest rim circle (radius r at z = +/-h).
        let dr = rho - r;
        let dist_top = (dr * dr + (p.z - h) * (p.z - h)).sqrt();
        let dist_bottom = (dr * dr + (p.z + h) * (p.z + h)).sqrt();

        SurfaceTruth {
            normal,
            closest,
            face_id,
            dist_to_sharp: dist_top.min(dist_bottom),
        }
    }

    fn world_bounds(&self) -> (DVec3, DVec3) {
        (
            DVec3::new(-self.radius, -self.radius, -self.half_height),
            DVec3::new(self.radius, self.radius, self.half_height),
        )
    }
}

// =============================================================================
// Rotation wrapper
// =============================================================================

/// Rotates any shape so its features are not grid-aligned. Queries are mapped
/// into the inner shape's local frame; truth vectors are rotated back out.
pub struct Rotated<S: OracleShape> {
    pub shape: S,
    pub rot: DQuat,
}

impl<S: OracleShape> Rotated<S> {
    pub fn new(shape: S, rot: DQuat) -> Self {
        Self { shape, rot }
    }
}

impl<S: OracleShape> OracleShape for Rotated<S> {
    fn name(&self) -> String {
        format!("rotated {}", self.shape.name())
    }

    fn is_inside(&self, p: DVec3) -> bool {
        self.shape.is_inside(self.rot.inverse() * p)
    }

    fn truth(&self, p: DVec3) -> SurfaceTruth {
        let local = self.shape.truth(self.rot.inverse() * p);
        SurfaceTruth {
            normal: self.rot * local.normal,
            closest: self.rot * local.closest,
            face_id: local.face_id,
            dist_to_sharp: local.dist_to_sharp,
        }
    }

    fn world_bounds(&self) -> (DVec3, DVec3) {
        let (lo, hi) = self.shape.world_bounds();
        let mut out_lo = DVec3::splat(f64::INFINITY);
        let mut out_hi = DVec3::splat(f64::NEG_INFINITY);
        for i in 0..8 {
            let corner = DVec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            let rotated = self.rot * corner;
            out_lo = out_lo.min(rotated);
            out_hi = out_hi.max(rotated);
        }
        (out_lo, out_hi)
    }
}

fn point_segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// The standard non-grid-aligned test orientation (35.264° X, 45° Y), the
/// "isometric" rotation used throughout the earlier edge research.
pub fn standard_rotation() -> DQuat {
    DQuat::from_euler(
        glam::EulerRot::XYZ,
        35.264_f64.to_radians(),
        45.0_f64.to_radians(),
        0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random points near the surface; for each, the truth
    /// must be self-consistent with the sampler: stepping out along the normal
    /// from the closest point must leave the shape, stepping in must enter it.
    fn check_sampler_truth_consistency(shape: &dyn OracleShape, eps: f64) {
        let (lo, hi) = shape.world_bounds();
        let span = hi - lo;
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut rand01 = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut checked = 0;
        for _ in 0..4000 {
            let p = lo + DVec3::new(rand01() * span.x, rand01() * span.y, rand01() * span.z);
            let t = shape.truth(p);
            // Only points whose closest surface point is well away from a sharp
            // feature have a locally planar neighborhood to test against. The
            // probe happens at `closest`, so measure feature distance there:
            // a query beyond an edge can be far from the edge while its closest
            // point lies exactly on it.
            if shape.truth(t.closest).dist_to_sharp < 10.0 * eps {
                continue;
            }
            assert!(
                (t.normal.length() - 1.0).abs() < 1e-9,
                "{}: normal not unit at {p:?}",
                shape.name()
            );
            let outside = t.closest + t.normal * eps;
            let inside = t.closest - t.normal * eps;
            assert!(
                !shape.is_inside(outside),
                "{}: point {outside:?} outside-of-normal should be outside (query {p:?})",
                shape.name()
            );
            assert!(
                shape.is_inside(inside),
                "{}: point {inside:?} inside-of-normal should be inside (query {p:?})",
                shape.name()
            );
            checked += 1;
        }
        assert!(
            checked > 500,
            "{}: too few consistency checks ran",
            shape.name()
        );
    }

    #[test]
    fn box_truth_is_consistent_with_sampler() {
        check_sampler_truth_consistency(
            &BoxShape {
                half: DVec3::splat(0.5),
            },
            1e-4,
        );
    }

    #[test]
    fn sphere_truth_is_consistent_with_sampler() {
        check_sampler_truth_consistency(&SphereShape { radius: 0.5 }, 1e-4);
    }

    #[test]
    fn cylinder_truth_is_consistent_with_sampler() {
        check_sampler_truth_consistency(
            &CylinderShape {
                radius: 0.4,
                half_height: 0.5,
            },
            1e-4,
        );
    }

    #[test]
    fn rotated_box_truth_is_consistent_with_sampler() {
        check_sampler_truth_consistency(
            &Rotated::new(
                BoxShape {
                    half: DVec3::splat(0.5),
                },
                standard_rotation(),
            ),
            1e-4,
        );
    }

    #[test]
    fn rotated_cylinder_truth_is_consistent_with_sampler() {
        check_sampler_truth_consistency(
            &Rotated::new(
                CylinderShape {
                    radius: 0.4,
                    half_height: 0.5,
                },
                standard_rotation(),
            ),
            1e-4,
        );
    }

    #[test]
    fn box_face_center_truth() {
        let cube = BoxShape {
            half: DVec3::splat(0.5),
        };
        // Just inside the +X face center.
        let t = cube.truth(DVec3::new(0.45, 0.0, 0.0));
        assert_eq!(t.face_id, 0);
        assert!((t.normal - DVec3::X).length() < 1e-12);
        assert!((t.closest - DVec3::new(0.5, 0.0, 0.0)).length() < 1e-12);
        // The query sits 0.05 inside the face center; nearest edges are at
        // lateral distance 0.5 and depth 0.05.
        let expected = (0.05f64 * 0.05 + 0.5 * 0.5).sqrt();
        assert!((t.dist_to_sharp - expected).abs() < 1e-9);
    }

    #[test]
    fn box_edge_distance() {
        let cube = BoxShape {
            half: DVec3::splat(0.5),
        };
        // On the +X/+Y edge exactly.
        let t = cube.truth(DVec3::new(0.5, 0.5, 0.1));
        assert!(t.dist_to_sharp < 1e-12);
    }

    #[test]
    fn cylinder_rim_distance() {
        let cyl = CylinderShape {
            radius: 0.4,
            half_height: 0.5,
        };
        // On the top rim exactly.
        let t = cyl.truth(DVec3::new(0.4, 0.0, 0.5));
        assert!(t.dist_to_sharp < 1e-12);
        // Side midline: half_height from both rims.
        let t = cyl.truth(DVec3::new(0.39, 0.0, 0.0));
        assert!((t.dist_to_sharp - (0.01f64 * 0.01 + 0.25).sqrt()).abs() < 1e-9);
        assert_eq!(t.face_id, 0);
    }

    #[test]
    fn rotated_normals_rotate() {
        let rot = standard_rotation();
        let cube = Rotated::new(
            BoxShape {
                half: DVec3::splat(0.5),
            },
            rot,
        );
        // Query near the rotated +X face center.
        let local = DVec3::new(0.45, 0.0, 0.0);
        let t = cube.truth(rot * local);
        assert!((t.normal - rot * DVec3::X).length() < 1e-9);
    }
}
