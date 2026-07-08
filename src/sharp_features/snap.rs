//! Feature snapping: move feature-zone vertices onto the sharp edge or corner
//! implied by the smooth regions around them.
//!
//! For each candidate vertex (unclaimed, or claimed with a neighbor claimed
//! by a different region), claimed vertices of each adjacent region are
//! gathered within a small radius. These are face-pure *by construction* (the
//! region label came from connectivity, not from geometric separation of a
//! mixed sample cloud — the failure mode of every per-vertex probing attempt).
//! One plane is fitted per side; two sides intersect in the local edge line,
//! three in a corner point, and the vertex is projected onto it.
//!
//! Robustness contract: snapping is opt-in per vertex behind a chain of gates
//! (side support, side fit residual, intersection conditioning, movement
//! clamp, sampler verification). Any gate failing leaves the vertex exactly
//! where the mesher put it, so pathological geometry (fractals, sub-cell
//! features) degrades to the current mesh, never to an invalid one.

use glam::DVec3;

use crate::sharp_features::adjacency::MeshAdjacency;
use crate::sharp_features::fit::fit_plane;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SnapConfig {
    /// Radius (cell units) around the vertex for gathering region points.
    pub gather_radius_cells: f64,
    /// Minimum claimed vertices per region side for that side to qualify.
    pub min_side_points: usize,
    /// Maximum RMS plane-fit residual (cell units) for a side to qualify.
    pub max_side_residual_cells: f64,
    /// Sides closer to parallel than this angle (degrees) define no reliable
    /// edge line.
    pub min_dihedral_deg: f64,
    /// Minimum |det| of the three unit side normals for a corner solve
    /// (the volume they span; 1 for orthogonal faces, 0 for coplanar).
    pub min_corner_det: f64,
    /// Snaps moving the vertex further than this (cell units) are rejected;
    /// real feature-zone vertices sit within about a cell of the feature.
    pub max_move_cells: f64,
    /// Sampler verification probe distance (cell units): the snapped position
    /// must have material just inside and none just outside along the mean
    /// side normal. Set to 0 to disable (e.g. when no sampler is available).
    pub verify_delta_cells: f64,
    /// Bisection iterations for sampler refinement of the snap target. The
    /// fitted side planes are secants on curved faces (a plane through a
    /// cylinder-rim arc sits inside the true tangent), which biases the
    /// intersection target inward and modulates with grid alignment — visible
    /// as rim wobble. Refinement bisects the occupancy boundary along each
    /// side's outward normal so the target lands on the model's actual
    /// surfaces instead of on the fitted planes. 0 disables (also disabled
    /// when no sampler is available); exact for planar faces either way.
    pub refine_iterations: usize,
    /// Bisection bracket half-width (cell units) around the plane-fit target.
    pub refine_bracket_cells: f64,
    /// While bisecting one side, the probe line is shifted this far (cell
    /// units) to the material side of the *other* sides, so it crosses only
    /// the surface being refined.
    pub refine_offset_cells: f64,
}

impl Default for SnapConfig {
    fn default() -> Self {
        Self {
            gather_radius_cells: 3.0,
            min_side_points: 6,
            max_side_residual_cells: 0.10,
            min_dihedral_deg: 10.0,
            min_corner_det: 0.05,
            max_move_cells: 1.5,
            verify_delta_cells: 0.6,
            refine_iterations: 12,
            refine_bracket_cells: 0.75,
            refine_offset_cells: 0.25,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapKind {
    Edge,
    Corner,
}

#[derive(Clone, Debug, Default)]
pub struct SnapStats {
    /// Unclaimed vertices considered.
    pub candidates: usize,
    pub snapped_edges: usize,
    pub snapped_corners: usize,
    /// Fewer than two qualifying region sides.
    pub rejected_sides: usize,
    /// Two sides too close to parallel for a stable edge line.
    pub rejected_parallel: usize,
    /// Corner attempts whose three normals were too close to coplanar; these
    /// fall back to an edge attempt rather than being rejected outright.
    pub corner_fallbacks: usize,
    /// Snap target further than the movement clamp.
    pub rejected_move: usize,
    /// Snap target failed the sampler surface check.
    pub rejected_verify: usize,
    /// Snap target had non-finite coordinates.
    pub rejected_nonfinite: usize,
}

pub struct SnapResult {
    /// Vertex positions with snapped updates applied.
    pub positions: Vec<DVec3>,
    /// What happened to each vertex (`None` for untouched, including all
    /// claimed vertices).
    pub snapped: Vec<Option<SnapKind>>,
    pub stats: SnapStats,
}

struct SidePlane {
    normal: DVec3,
    centroid: DVec3,
    support: usize,
}

/// Snap unclaimed vertices onto locally fitted feature lines/points.
///
/// `sampler` is the model's binary occupancy function; when provided (and
/// `verify_delta_cells > 0`), every snap target is verified against it.
pub fn snap_feature_vertices(
    positions: &[DVec3],
    adjacency: &MeshAdjacency,
    labels: &[Option<u32>],
    cell: f64,
    config: &SnapConfig,
    sampler: Option<&dyn Fn(DVec3) -> bool>,
) -> SnapResult {
    let mut out_positions = positions.to_vec();
    let mut snapped: Vec<Option<SnapKind>> = vec![None; positions.len()];
    let mut stats = SnapStats::default();

    // Around corners the unclaimed pool is wider than along edges, pushing
    // each region's claimed vertices further away; one retry with a larger
    // gather radius recovers those without loosening the common case.
    const RETRY_GATHER_SCALE: f64 = 1.75;
    let gather_radii = [
        config.gather_radius_cells,
        config.gather_radius_cells * RETRY_GATHER_SCALE,
    ];

    // Candidates are the feature-zone (unclaimed) vertices, plus
    // region-boundary vertices: when the sampling grid aligns with a feature,
    // the sawtooth amplitude collapses below the segmentation residual gates
    // and two regions grow into direct contact with no unclaimed band between
    // them. A claimed vertex with a differently-claimed neighbor sits on a
    // feature all the same. (That regime is also exactly when near-feature
    // vertex positions are accurate, so gathering from them is sound.)
    let is_candidate = |v: usize| -> bool {
        match labels[v] {
            None => true,
            Some(a) => adjacency
                .neighbors(v as u32)
                .iter()
                .any(|&u| matches!(labels[u as usize], Some(b) if b != a)),
        }
    };

    for v in 0..positions.len() {
        if !is_candidate(v) {
            continue;
        }
        stats.candidates += 1;
        let origin = positions[v];

        let mut planes: Vec<SidePlane> = Vec::new();
        for &radius_cells in &gather_radii {
            planes = gather_side_planes(
                positions,
                adjacency,
                labels,
                v,
                origin,
                radius_cells,
                cell,
                config,
            );
            if planes.len() >= 2 {
                break;
            }
        }
        if planes.len() < 2 {
            stats.rejected_sides += 1;
            continue;
        }
        planes.sort_by_key(|p| std::cmp::Reverse(p.support));

        // Try a corner when three sides qualify, falling back to the
        // best-supported edge pair when the corner solve is ill-conditioned
        // or its target is out of movement range (vertices along an edge near
        // a corner see three regions but belong on the edge line).
        let max_move = config.max_move_cells * cell;
        let mut target: Option<(DVec3, SnapKind)> = None;
        if planes.len() >= 3 {
            match intersect_three_planes(&planes[0], &planes[1], &planes[2], config.min_corner_det)
            {
                Some(p) if p.is_finite() && (p - origin).length() <= max_move => {
                    target = Some((p, SnapKind::Corner));
                }
                _ => stats.corner_fallbacks += 1,
            }
        }
        if target.is_none() {
            let max_dot = (config.min_dihedral_deg.to_radians()).cos();
            match intersect_two_planes(origin, &planes[0], &planes[1], max_dot) {
                Some(p) => target = Some((p, SnapKind::Edge)),
                None => {
                    stats.rejected_parallel += 1;
                    continue;
                }
            }
        }
        let (mut p, kind) = target.unwrap();

        if !p.is_finite() {
            stats.rejected_nonfinite += 1;
            continue;
        }
        if (p - origin).length() > max_move {
            stats.rejected_move += 1;
            continue;
        }

        if let Some(is_inside) = sampler {
            // PCA normals have arbitrary sign; orient each participating side
            // outward with one probe at its own centroid (far from the
            // feature, so the probe is unambiguous).
            let delta = config.verify_delta_cells.max(0.5) * cell;
            let outward: Vec<DVec3> = planes
                .iter()
                .take(if kind == SnapKind::Corner { 3 } else { 2 })
                .map(|s| orient_outward(s, is_inside, delta))
                .collect();

            // Refine the target onto the model's actual occupancy boundary.
            // The clamp is re-checked because refinement moves the target;
            // exceeding it falls back to the already-clamped plane target.
            if config.refine_iterations > 0 {
                let refined = refine_target(p, &outward, cell, config, is_inside);
                if refined.is_finite() && (refined - origin).length() <= max_move {
                    p = refined;
                }
            }

            // Verify: material just inside, none just outside along the mean
            // outward side normal.
            if config.verify_delta_cells > 0.0 {
                let delta = config.verify_delta_cells * cell;
                let Some(b) = outward.iter().sum::<DVec3>().try_normalize() else {
                    stats.rejected_verify += 1;
                    continue;
                };
                let inside_ok = is_inside(p - b * delta);
                let outside_ok = !is_inside(p + b * delta);
                if !(inside_ok && outside_ok) {
                    stats.rejected_verify += 1;
                    continue;
                }
            }
        }

        out_positions[v] = p;
        snapped[v] = Some(kind);
        match kind {
            SnapKind::Edge => stats.snapped_edges += 1,
            SnapKind::Corner => stats.snapped_corners += 1,
        }
    }

    SnapResult {
        positions: out_positions,
        snapped,
        stats,
    }
}

/// Gather claimed vertices per region within `radius_cells` of `origin` and
/// fit one qualifying plane per side (enough support, tight fit).
#[allow(clippy::too_many_arguments)]
fn gather_side_planes(
    positions: &[DVec3],
    adjacency: &MeshAdjacency,
    labels: &[Option<u32>],
    v: usize,
    origin: DVec3,
    radius_cells: f64,
    cell: f64,
    config: &SnapConfig,
) -> Vec<SidePlane> {
    let ring_depth = radius_cells.ceil() as usize + 1;
    let radius = radius_cells * cell;

    let mut sides: Vec<(u32, Vec<DVec3>)> = Vec::new();
    for u in adjacency.k_ring(v as u32, ring_depth) {
        let Some(label) = labels[u as usize] else {
            continue;
        };
        let p = positions[u as usize];
        if (p - origin).length() > radius {
            continue;
        }
        match sides.iter_mut().find(|(l, _)| *l == label) {
            Some((_, pts)) => pts.push(p),
            None => sides.push((label, vec![p])),
        }
    }

    let mut planes: Vec<SidePlane> = Vec::new();
    for (_, pts) in &sides {
        if pts.len() < config.min_side_points {
            continue;
        }
        let Some(fit) = fit_plane(pts) else {
            continue;
        };
        if fit.rms_residual / cell > config.max_side_residual_cells {
            continue;
        }
        // PCA normal sign is arbitrary; the intersection solves are
        // sign-agnostic and verification orients per-side later.
        planes.push(SidePlane {
            normal: fit.normal,
            centroid: fit.centroid,
            support: pts.len(),
        });
    }
    planes
}

/// Refine a plane-intersection target onto the model's actual occupancy
/// boundary. For each side, bisect along its outward normal, with the probe
/// line shifted slightly to the material side of the other sides so it
/// crosses only the surface being refined. Planar faces refine to themselves
/// (up to bisection resolution); on curved faces this replaces the fitted
/// plane's secant with the true surface. Sides whose bracket doesn't straddle
/// the boundary contribute no correction, so a bad bracket can never make the
/// target worse than the plane intersection it started from.
fn refine_target(
    target: DVec3,
    outward: &[DVec3],
    cell: f64,
    config: &SnapConfig,
    is_inside: &dyn Fn(DVec3) -> bool,
) -> DVec3 {
    let bracket = config.refine_bracket_cells * cell;
    let offset_len = config.refine_offset_cells * cell;
    let mut p = target;
    // Two rounds: the solve is exact for planar faces, the second round
    // cleans up what curvature shifted under the first round's probes.
    for _ in 0..2 {
        // Measure each side's signed offset: how far p must move along the
        // side's outward normal to sit on that side's surface.
        let mut measured: Vec<(DVec3, f64)> = Vec::new();
        for (i, &n) in outward.iter().enumerate() {
            let inward_rest: DVec3 = outward
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &m)| -m)
                .sum();
            let Some(offset_dir) = inward_rest.try_normalize() else {
                continue;
            };
            let base = p + offset_dir * offset_len;
            let (mut lo, mut hi) = (-bracket, bracket);
            if !is_inside(base + n * lo) || is_inside(base + n * hi) {
                continue;
            }
            for _ in 0..config.refine_iterations {
                let mid = 0.5 * (lo + hi);
                if is_inside(base + n * mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            // The crossing was measured on the offset probe line; adding the
            // offset's normal component back expresses it as a constraint at
            // p itself (offset_dir isn't perpendicular to n unless the sides
            // are orthogonal).
            measured.push((n, 0.5 * (lo + hi) + n.dot(offset_dir) * offset_len));
        }
        // Solve the joint constraints n_i . (p' - p) = d_i with minimal
        // movement — the same solves as the plane intersections, but against
        // measured surface positions instead of fitted planes. Conditioning
        // was already gated when the target was accepted; the guards here
        // only protect against division blow-ups.
        match measured.as_slice() {
            [(n, d)] => p += *n * *d,
            [(n1, d1), (n2, d2)] => {
                let dot = n1.dot(*n2);
                let det = 1.0 - dot * dot;
                if det > 1e-4 {
                    let alpha = (d1 - dot * d2) / det;
                    let beta = (d2 - dot * d1) / det;
                    p += *n1 * alpha + *n2 * beta;
                }
            }
            [(n1, d1), (n2, d2), (n3, d3)] => {
                let det = n1.dot(n2.cross(*n3));
                if det.abs() > 1e-3 {
                    p += (n2.cross(*n3) * *d1 + n3.cross(*n1) * *d2 + n1.cross(*n2) * *d3) / det;
                }
            }
            _ => {}
        }
    }
    p
}

/// Orient a side plane normal to point out of the material, determined by one
/// sampler probe from the side centroid.
fn orient_outward(side: &SidePlane, is_inside: &dyn Fn(DVec3) -> bool, delta: f64) -> DVec3 {
    if is_inside(side.centroid + side.normal * delta) {
        -side.normal
    } else {
        side.normal
    }
}

/// Project `origin` onto the intersection line of two planes (each given by a
/// point and unit normal). Returns `None` when the planes are closer to
/// parallel than `max_abs_dot` allows.
fn intersect_two_planes(
    origin: DVec3,
    a: &SidePlane,
    b: &SidePlane,
    max_abs_dot: f64,
) -> Option<DVec3> {
    let dot = a.normal.dot(b.normal);
    if dot.abs() > max_abs_dot {
        return None;
    }
    // Minimize |p - origin|^2 subject to both plane constraints:
    // p = origin + alpha * n_a + beta * n_b.
    let ra = a.normal.dot(a.centroid - origin);
    let rb = b.normal.dot(b.centroid - origin);
    let det = 1.0 - dot * dot;
    let alpha = (ra - dot * rb) / det;
    let beta = (rb - dot * ra) / det;
    Some(origin + a.normal * alpha + b.normal * beta)
}

/// Intersection point of three planes. Returns `None` when the normals span
/// less volume than `min_det` (near-coplanar configuration).
fn intersect_three_planes(
    a: &SidePlane,
    b: &SidePlane,
    c: &SidePlane,
    min_det: f64,
) -> Option<DVec3> {
    let det = a.normal.dot(b.normal.cross(c.normal));
    if det.abs() < min_det {
        return None;
    }
    let da = a.normal.dot(a.centroid);
    let db = b.normal.dot(b.centroid);
    let dc = c.normal.dot(c.centroid);
    Some(
        (b.normal.cross(c.normal) * da
            + c.normal.cross(a.normal) * db
            + a.normal.cross(b.normal) * dc)
            / det,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharp_features::fit::ring_fits;
    use crate::sharp_features::segmentation::{SegmentationConfig, segment_regions};

    fn plane(normal: DVec3, centroid: DVec3) -> SidePlane {
        SidePlane {
            normal: normal.normalize(),
            centroid,
            support: 10,
        }
    }

    #[test]
    fn two_plane_intersection_projects_onto_line() {
        // Planes x = 1 and y = 2 meet in the line (1, 2, t).
        let a = plane(DVec3::X, DVec3::new(1.0, 0.0, 0.0));
        let b = plane(DVec3::Y, DVec3::new(0.0, 2.0, 0.0));
        let p = intersect_two_planes(DVec3::new(0.0, 0.0, 5.0), &a, &b, 0.9).unwrap();
        assert!((p - DVec3::new(1.0, 2.0, 5.0)).length() < 1e-12);
    }

    #[test]
    fn near_parallel_planes_are_rejected() {
        let a = plane(DVec3::X, DVec3::ZERO);
        let tilted = DVec3::new(1.0, 0.05, 0.0);
        let b = plane(tilted, DVec3::ZERO);
        // cos(10 deg) ~= 0.985; these normals are ~2.9 deg apart.
        assert!(intersect_two_planes(DVec3::ZERO, &a, &b, 0.985).is_none());
    }

    #[test]
    fn three_plane_intersection_finds_corner() {
        let a = plane(DVec3::X, DVec3::new(1.0, 9.0, 9.0));
        let b = plane(DVec3::Y, DVec3::new(9.0, 2.0, 9.0));
        let c = plane(DVec3::Z, DVec3::new(9.0, 9.0, 3.0));
        let p = intersect_three_planes(&a, &b, &c, 0.05).unwrap();
        assert!((p - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-12);
    }

    #[test]
    fn near_coplanar_corner_is_rejected() {
        let a = plane(DVec3::X, DVec3::ZERO);
        let b = plane(DVec3::new(1.0, 0.02, 0.0), DVec3::ZERO);
        let c = plane(DVec3::new(1.0, 0.0, 0.03), DVec3::ZERO);
        assert!(intersect_three_planes(&a, &b, &c, 0.05).is_none());
    }

    /// Refinement must pull a plane-fit target off its secant onto the true
    /// curved surface: cylinder rim, radius 20 cells, cap at z = 10.
    #[test]
    fn refine_lands_on_curved_rim() {
        let config = SnapConfig::default();
        let is_inside = |p: DVec3| p.x * p.x + p.y * p.y <= 400.0 && p.z <= 10.0;
        // Plane-fit target with the observed failure mode: pulled inward
        // radially (secant bias), slightly off the cap too.
        let target = DVec3::new(19.8, 0.0, 9.95);
        let outward = [DVec3::X, DVec3::Z]; // barrel side, cap side
        let p = refine_target(target, &outward, 1.0, &config, &is_inside);
        assert!(
            (p - DVec3::new(20.0, 0.0, 10.0)).length() < 5e-3,
            "refined point should land on the rim, got {p:?}"
        );
    }

    /// Non-orthogonal sides: the probe-line offset has a component along the
    /// refined side's normal, which the update must compensate for.
    #[test]
    fn refine_is_exact_for_planar_non_orthogonal_wedge() {
        let config = SnapConfig::default();
        // Wedge z <= 0 AND x + z <= 0; edge along the y axis through origin.
        let is_inside = |p: DVec3| p.z <= 0.0 && p.x + p.z <= 0.0;
        let outward = [DVec3::Z, DVec3::new(1.0, 0.0, 1.0).normalize()];
        let target = DVec3::new(0.12, 0.3, -0.07);
        let p = refine_target(target, &outward, 1.0, &config, &is_inside);
        assert!(
            (p - DVec3::new(0.0, 0.3, 0.0)).length() < 5e-3,
            "refined point should land on the wedge edge, got {p:?}"
        );
    }

    /// A bracket that doesn't straddle the boundary must contribute no
    /// correction: the target comes back unchanged, never worse.
    #[test]
    fn refine_without_boundary_in_bracket_is_identity() {
        let config = SnapConfig::default();
        let is_inside = |_: DVec3| true; // deep inside material
        let outward = [DVec3::X, DVec3::Z];
        let target = DVec3::new(1.0, 2.0, 3.0);
        let p = refine_target(target, &outward, 1.0, &config, &is_inside);
        assert_eq!(p, target);
    }

    /// End-to-end on the synthetic tent: crease vertices must land exactly on
    /// the analytic crease line.
    #[test]
    fn tent_crease_vertices_snap_onto_the_crease_line() {
        let crease = 7usize;
        let n = 15usize;
        let mut positions = Vec::new();
        for j in 0..n {
            for i in 0..n {
                positions.push(if i <= crease {
                    DVec3::new(i as f64, j as f64, 0.0)
                } else {
                    DVec3::new(crease as f64, j as f64, (i - crease) as f64)
                });
            }
        }
        let mut indices = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let a = (j * n + i) as u32;
                let (b, c, d) = (a + 1, a + n as u32, a + n as u32 + 1);
                indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        let adjacency = MeshAdjacency::build(positions.len(), &indices);
        let fits = ring_fits(&positions, &adjacency, &[], 1.0, 1);
        let seg = segment_regions(&adjacency, &fits, &SegmentationConfig::default());

        let config = SnapConfig {
            verify_delta_cells: 0.0, // no sampler for a synthetic mesh
            ..SnapConfig::default()
        };
        let result = snap_feature_vertices(&positions, &adjacency, &seg.labels, 1.0, &config, None);

        // The crease is the line x = 7, z = 0. Crease-column vertices away
        // from the open boundary must snap onto it exactly (planar sides).
        let mut checked = 0;
        for j in 3..n - 3 {
            let v = j * n + crease;
            if seg.labels[v].is_some() {
                continue;
            }
            assert!(
                result.snapped[v].is_some(),
                "crease vertex ({crease},{j}) was not snapped: {:?}",
                result.stats
            );
            let p = result.positions[v];
            assert!(
                (p.x - 7.0).abs() < 1e-9 && p.z.abs() < 1e-9,
                "crease vertex ({crease},{j}) landed off the crease: {p:?}"
            );
            checked += 1;
        }
        assert!(checked >= 5, "too few crease vertices exercised");
        assert_eq!(result.stats.rejected_nonfinite, 0);
    }
}
