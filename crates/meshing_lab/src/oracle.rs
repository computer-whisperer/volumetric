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
// Polygon prism (concave features)
// =============================================================================

/// Prism along Z over an arbitrary simple polygon cross-section (CCW winding).
/// Concave polygon vertices produce concave (reentrant) prism edges — the
/// feature class the box and cylinder cannot exercise.
///
/// Face ids: side face `i` spans polygon edge `points[i] -> points[i+1]`;
/// the top cap is `n`, the bottom cap `n + 1`.
pub struct PolygonPrism {
    pub points: Vec<glam::DVec2>,
    pub half_height: f64,
}

impl PolygonPrism {
    /// An L cross-section (one reentrant corner) spanning roughly [-s, s].
    pub fn l_shape(s: f64, half_height: f64) -> Self {
        let p = |x: f64, y: f64| glam::DVec2::new(x * s, y * s);
        Self {
            points: vec![
                p(-1.0, -1.0),
                p(1.0, -1.0),
                p(1.0, 0.0),
                p(0.0, 0.0), // reentrant corner
                p(0.0, 1.0),
                p(-1.0, 1.0),
            ],
            half_height,
        }
    }

    fn inside_2d(&self, q: glam::DVec2) -> bool {
        // Even-odd ray casting.
        let n = self.points.len();
        let mut inside = false;
        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];
            if (a.y > q.y) != (b.y > q.y) {
                let x_cross = a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x);
                if q.x < x_cross {
                    inside = !inside;
                }
            }
        }
        inside
    }

    /// Nearest polygon edge to `q`: (edge index, closest point on it).
    fn nearest_edge_2d(&self, q: glam::DVec2) -> (usize, glam::DVec2) {
        let n = self.points.len();
        let mut best = (0usize, self.points[0], f64::INFINITY);
        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];
            let ab = b - a;
            let t = ((q - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            let closest = a + ab * t;
            let d = (q - closest).length();
            if d < best.2 {
                best = (i, closest, d);
            }
        }
        (best.0, best.1)
    }
}

impl OracleShape for PolygonPrism {
    fn name(&self) -> String {
        format!(
            "polygon prism ({} verts) h={}",
            self.points.len(),
            2.0 * self.half_height
        )
    }

    fn is_inside(&self, p: DVec3) -> bool {
        p.z.abs() <= self.half_height && self.inside_2d(glam::DVec2::new(p.x, p.y))
    }

    fn truth(&self, p: DVec3) -> SurfaceTruth {
        let n = self.points.len();
        let h = self.half_height;
        let q = glam::DVec2::new(p.x, p.y);
        let inside_2d = self.inside_2d(q);
        let (edge_idx, boundary_2d) = self.nearest_edge_2d(q);
        let d_boundary_2d = (q - boundary_2d).length();

        // Outward normal of the nearest side face (CCW polygon: outward is
        // the right-hand perpendicular of the edge direction).
        let a = self.points[edge_idx];
        let b = self.points[(edge_idx + 1) % n];
        let dir = (b - a).normalize();
        let side_normal = DVec3::new(dir.y, -dir.x, 0.0);

        // Signed distances to the side wall and cap planes (negative inside).
        let d_side = if inside_2d {
            -d_boundary_2d
        } else {
            d_boundary_2d
        };
        let d_cap = p.z.abs() - h;

        let (normal, closest, face_id) = if d_side >= d_cap {
            let closest = DVec3::new(boundary_2d.x, boundary_2d.y, p.z.clamp(-h, h));
            (side_normal, closest, edge_idx as u32)
        } else {
            let sign = if p.z >= 0.0 { 1.0 } else { -1.0 };
            let q_on_cap = if inside_2d { q } else { boundary_2d };
            let closest = DVec3::new(q_on_cap.x, q_on_cap.y, sign * h);
            let face_id = if sign > 0.0 { n as u32 } else { n as u32 + 1 };
            (DVec3::new(0.0, 0.0, sign), closest, face_id)
        };

        // Sharp features: vertical edges at every polygon vertex (convex and
        // reentrant alike) and the cap rim above every polygon edge.
        let mut dist_to_sharp = f64::INFINITY;
        for i in 0..n {
            let v = self.points[i];
            let d2 = (q - v).length();
            let dz = (p.z.abs() - h).max(0.0);
            dist_to_sharp = dist_to_sharp.min((d2 * d2 + dz * dz).sqrt());
        }
        let rim = (d_boundary_2d * d_boundary_2d + (p.z.abs() - h) * (p.z.abs() - h)).sqrt();
        dist_to_sharp = dist_to_sharp.min(rim);

        SurfaceTruth {
            normal,
            closest,
            face_id,
            dist_to_sharp,
        }
    }

    fn world_bounds(&self) -> (DVec3, DVec3) {
        let mut lo = glam::DVec2::splat(f64::INFINITY);
        let mut hi = glam::DVec2::splat(f64::NEG_INFINITY);
        for &p in &self.points {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        (
            DVec3::new(lo.x, lo.y, -self.half_height),
            DVec3::new(hi.x, hi.y, self.half_height),
        )
    }
}

// =============================================================================
// Mandelbulb (pathological control -- NO analytic truth)
// =============================================================================

/// Power-8 mandelbulb by escape-time iteration: the pathological control
/// shape. Fractal boundary, sub-cell detail everywhere, no meaningful smooth
/// faces.
///
/// There is no closed-form surface truth for this shape: [`OracleShape::truth`]
/// returns placeholders (radial normal, infinite feature distance) so it can
/// flow through the harness, but only *validity* metrics (finiteness, movement
/// bounds, triangle degeneracy) are meaningful — never accuracy metrics.
pub struct MandelbulbShape {
    pub iterations: usize,
}

impl OracleShape for MandelbulbShape {
    fn name(&self) -> String {
        format!("mandelbulb p8 i{} (validity only)", self.iterations)
    }

    fn is_inside(&self, p: DVec3) -> bool {
        let c = p;
        let mut z = p;
        for _ in 0..self.iterations {
            let r = z.length();
            if r > 2.0 {
                return false;
            }
            if r < 1e-12 {
                z = c;
                continue;
            }
            // z -> z^8 + c in spherical coordinates.
            let theta = (z.z / r).clamp(-1.0, 1.0).acos() * 8.0;
            let phi = z.y.atan2(z.x) * 8.0;
            let r8 = r.powi(8);
            z = DVec3::new(
                r8 * theta.sin() * phi.cos(),
                r8 * theta.sin() * phi.sin(),
                r8 * theta.cos(),
            ) + c;
        }
        true
    }

    fn truth(&self, p: DVec3) -> SurfaceTruth {
        SurfaceTruth {
            normal: p.try_normalize().unwrap_or(DVec3::Z),
            closest: p,
            face_id: 0,
            dist_to_sharp: f64::INFINITY,
        }
    }

    fn world_bounds(&self) -> (DVec3, DVec3) {
        (DVec3::splat(-1.25), DVec3::splat(1.25))
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
    fn l_prism_truth_is_consistent_with_sampler() {
        check_sampler_truth_consistency(&PolygonPrism::l_shape(0.5, 0.4), 1e-4);
    }

    #[test]
    fn rotated_l_prism_truth_is_consistent_with_sampler() {
        check_sampler_truth_consistency(
            &Rotated::new(PolygonPrism::l_shape(0.5, 0.4), standard_rotation()),
            1e-4,
        );
    }

    #[test]
    fn l_prism_reentrant_corner_is_sharp() {
        let prism = PolygonPrism::l_shape(0.5, 0.4);
        // The reentrant corner sits at the origin of the cross-section.
        let t = prism.truth(DVec3::new(0.0, 0.0, 0.1));
        assert!(t.dist_to_sharp < 1e-12, "reentrant corner not detected");
        // Just inside the notch, next to the vertical face at x = 0
        // (polygon edge (0,0) -> (0,0.5), outward normal +X).
        let t = prism.truth(DVec3::new(-0.02, 0.2, 0.0));
        assert_eq!(t.face_id, 3);
        assert!((t.normal - DVec3::X).length() < 1e-12);
        assert!(prism.is_inside(DVec3::new(-0.02, 0.2, 0.0)));
        assert!(!prism.is_inside(DVec3::new(0.02, 0.2, 0.0)));
    }

    #[test]
    fn l_shape_polygon_is_ccw() {
        let prism = PolygonPrism::l_shape(1.0, 0.5);
        let n = prism.points.len();
        let shoelace: f64 = (0..n)
            .map(|i| {
                let a = prism.points[i];
                let b = prism.points[(i + 1) % n];
                a.x * b.y - b.x * a.y
            })
            .sum();
        assert!(shoelace > 0.0, "polygon must wind CCW");
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
