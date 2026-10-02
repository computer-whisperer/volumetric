//! Feature snapping: move feature-zone vertices onto the sharp edge or corner
//! implied by the smooth faces around them.
//!
//! For each candidate vertex (unclaimed, or claimed with a claimed neighbor
//! on another face), the claimed vertices within a small radius are gathered
//! and sorted into sides, one per face. These are face-pure *by construction*
//! (a side comes from connectivity, not from geometric separation of a mixed
//! sample cloud — the failure mode of every per-vertex probing attempt). One
//! plane is fitted per side; two sides intersect in the local edge line,
//! three in a corner point, and the vertex is projected onto it.
//!
//! A side is normally a region's vertices. Region labels are global, though,
//! and two faces that meet at a crease here carry one label when they are
//! joined smoothly somewhere else (see [`SmoothFaces`]). So a region that no
//! single plane fits is split into the pieces connected inside the radius.
//!
//! Robustness contract: snapping is opt-in per vertex behind a chain of gates
//! (side support, side fit residual, intersection conditioning, movement
//! clamp, sampler verification). Any gate failing leaves the vertex exactly
//! where the mesher put it, so pathological geometry (fractals, sub-cell
//! features) degrades to the current mesh, never to an invalid one.

use std::sync::atomic::{AtomicBool, Ordering};

use glam::DVec3;

use crate::sharp_features::OccupancyFn;
use crate::sharp_features::adjacency::MeshAdjacency;
use crate::sharp_features::fit::fit_plane;
use crate::sharp_features::segmentation::SmoothFaces;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SnapConfig {
    /// Radius (cell units) around the vertex for gathering face points.
    pub gather_radius_cells: f64,
    /// Minimum claimed vertices on a side for that side to qualify.
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
    /// Fewer than two qualifying sides.
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
    /// The side is one piece of a region that no single plane fitted.
    split: bool,
}

/// What one candidate evaluation produced, folded into [`SnapStats`] and the
/// output arrays by the (serial) driver. `corner_fallback` travels alongside
/// because a failed corner solve falls back to an edge attempt whose own
/// outcome is independent.
enum SnapAttempt {
    Snapped(DVec3, SnapKind),
    RejectedSides,
    RejectedParallel,
    RejectedMove,
    RejectedVerify,
    RejectedNonfinite,
    /// The cancel flag was observed before this candidate ran; counts toward
    /// no statistic (the caller discards the whole result anyway).
    Cancelled,
}

/// Snap unclaimed vertices onto locally fitted feature lines/points.
///
/// `sampler` is the model's binary occupancy function; when provided (and
/// `verify_delta_cells > 0`), every snap target is verified against it.
pub fn snap_feature_vertices(
    positions: &[DVec3],
    adjacency: &MeshAdjacency,
    faces: SmoothFaces,
    cell: f64,
    config: &SnapConfig,
    sampler: Option<&dyn OccupancyFn>,
) -> SnapResult {
    static NEVER: AtomicBool = AtomicBool::new(false);
    snap_feature_vertices_cancellable(positions, adjacency, faces, cell, config, sampler, &NEVER)
}

/// [`snap_feature_vertices`], checking `cancel` before each candidate.
/// Candidates are evaluated in parallel on native builds — each reads only
/// the shared inputs and returns its outcome, which a serial fold applies in
/// candidate order, so the result is identical to the sequential path. On
/// cancellation the remaining candidates are skipped and the (about to be
/// discarded) result returns with whatever snaps completed.
pub fn snap_feature_vertices_cancellable(
    positions: &[DVec3],
    adjacency: &MeshAdjacency,
    faces: SmoothFaces,
    cell: f64,
    config: &SnapConfig,
    sampler: Option<&dyn OccupancyFn>,
    cancel: &AtomicBool,
) -> SnapResult {
    // Candidates are the feature-zone (unclaimed) vertices, plus claimed
    // vertices with a claimed neighbor on another face: when the sampling
    // grid aligns with a feature, the sawtooth amplitude collapses below the
    // segmentation residual gates and two faces grow into direct contact with
    // no unclaimed band between them. Such a vertex sits on a feature all the
    // same. (That regime is also exactly when near-feature vertex positions
    // are accurate, so gathering from them is sound.)
    let is_candidate = |v: usize| -> bool {
        let v = v as u32;
        !faces.claimed(v)
            || adjacency
                .neighbors(v)
                .iter()
                .any(|&u| faces.claimed(u) && !faces.joined(v, u))
    };
    let candidate_mask = crate::parallel_iter::map_range(0..positions.len(), is_candidate);
    let candidates: Vec<u32> = candidate_mask
        .iter()
        .enumerate()
        .filter_map(|(v, &c)| c.then_some(v as u32))
        .collect();

    let outcomes = crate::parallel_iter::map_vec(candidates, |v| {
        if cancel.load(Ordering::Relaxed) {
            return (v, false, SnapAttempt::Cancelled);
        }
        let (corner_fallback, attempt) = snap_one(
            v as usize, positions, adjacency, faces, cell, config, sampler,
        );
        (v, corner_fallback, attempt)
    });

    let mut out_positions = positions.to_vec();
    let mut snapped: Vec<Option<SnapKind>> = vec![None; positions.len()];
    let mut stats = SnapStats::default();
    for (v, corner_fallback, attempt) in outcomes {
        stats.candidates += 1;
        if corner_fallback {
            stats.corner_fallbacks += 1;
        }
        match attempt {
            SnapAttempt::Snapped(p, kind) => {
                out_positions[v as usize] = p;
                snapped[v as usize] = Some(kind);
                match kind {
                    SnapKind::Edge => stats.snapped_edges += 1,
                    SnapKind::Corner => stats.snapped_corners += 1,
                }
            }
            SnapAttempt::RejectedSides => stats.rejected_sides += 1,
            SnapAttempt::RejectedParallel => stats.rejected_parallel += 1,
            SnapAttempt::RejectedMove => stats.rejected_move += 1,
            SnapAttempt::RejectedVerify => stats.rejected_verify += 1,
            SnapAttempt::RejectedNonfinite => stats.rejected_nonfinite += 1,
            SnapAttempt::Cancelled => {}
        }
    }

    SnapResult {
        positions: out_positions,
        snapped,
        stats,
    }
}

/// Evaluate one candidate vertex: gather side planes, solve the feature
/// intersection, refine and verify against the sampler. Reads only shared
/// immutable inputs, so candidates evaluate in parallel.
fn snap_one(
    v: usize,
    positions: &[DVec3],
    adjacency: &MeshAdjacency,
    faces: SmoothFaces,
    cell: f64,
    config: &SnapConfig,
    sampler: Option<&dyn OccupancyFn>,
) -> (bool, SnapAttempt) {
    // Around corners the unclaimed pool is wider than along edges, pushing
    // each face's claimed vertices further away; one retry with a larger
    // gather radius recovers those without loosening the common case.
    const RETRY_GATHER_SCALE: f64 = 1.75;
    let gather_radii = [
        config.gather_radius_cells,
        config.gather_radius_cells * RETRY_GATHER_SCALE,
    ];
    let origin = positions[v];
    let mut corner_fallback = false;

    let mut planes: Vec<SidePlane> = Vec::new();
    for &radius_cells in &gather_radii {
        planes = gather_side_planes(
            positions,
            adjacency,
            faces,
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
        return (corner_fallback, SnapAttempt::RejectedSides);
    }
    planes.sort_by_key(|p| std::cmp::Reverse(p.support));

    // Try a corner when three sides qualify, falling back to the
    // best-supported edge pair when the corner solve is ill-conditioned
    // or its target is out of movement range (vertices along an edge near
    // a corner see three faces but belong on the edge line).
    let max_move = config.max_move_cells * cell;
    let mut target: Option<(DVec3, SnapKind)> = None;
    if planes.len() >= 3 {
        match intersect_three_planes(&planes[0], &planes[1], &planes[2], config.min_corner_det) {
            Some(p) if p.is_finite() && (p - origin).length() <= max_move => {
                target = Some((p, SnapKind::Corner));
            }
            _ => corner_fallback = true,
        }
    }
    if target.is_none() {
        let max_dot = (config.min_dihedral_deg.to_radians()).cos();
        match intersect_two_planes(origin, &planes[0], &planes[1], max_dot) {
            Some(p) => target = Some((p, SnapKind::Edge)),
            None => return (corner_fallback, SnapAttempt::RejectedParallel),
        }
    }
    let (mut p, kind) = target.unwrap();

    if !p.is_finite() {
        return (corner_fallback, SnapAttempt::RejectedNonfinite);
    }
    if (p - origin).length() > max_move {
        return (corner_fallback, SnapAttempt::RejectedMove);
    }

    if let Some(is_inside) = sampler {
        // The gather/refine helpers predate the parallel driver and take the
        // plain closure trait; upcast once here.
        let is_inside: &dyn Fn(DVec3) -> bool = is_inside;
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
                return (corner_fallback, SnapAttempt::RejectedVerify);
            };
            let inside_ok = is_inside(p - b * delta);
            let outside_ok = !is_inside(p + b * delta);
            if !(inside_ok && outside_ok) {
                return (corner_fallback, SnapAttempt::RejectedVerify);
            }

            // Sides that are pieces of one region were told apart only by
            // what lies inside the gather radius, and at that scale a tight
            // fillet looks like a crease: its two flanks come out as two
            // sides, and their planes meet beyond the fillet, at a corner the
            // model does not have. The check above cannot see that when a
            // third face passes through the same point. So such a snap has to
            // show that every side's surface is really there.
            let from_split = planes.iter().take(outward.len()).any(|side| side.split);
            if from_split && !sides_meet_at(p, &outward, cell, config, is_inside) {
                return (corner_fallback, SnapAttempt::RejectedVerify);
            }
        }
    }

    (corner_fallback, SnapAttempt::Snapped(p, kind))
}

/// Gather the claimed vertices within `radius_cells` of `origin`, sort them
/// into sides, and fit one qualifying plane per side (enough support, tight
/// fit).
///
/// A side is a region's gathered vertices when one plane fits them all. When
/// it does not, the region covers more than one face here, and each piece of
/// it that is connected inside the radius ([`connected_pieces`]) is a side of
/// its own.
#[allow(clippy::too_many_arguments)]
fn gather_side_planes(
    positions: &[DVec3],
    adjacency: &MeshAdjacency,
    faces: SmoothFaces,
    v: usize,
    origin: DVec3,
    radius_cells: f64,
    cell: f64,
    config: &SnapConfig,
) -> Vec<SidePlane> {
    let ring_depth = radius_cells.ceil() as usize + 1;
    let radius = radius_cells * cell;

    let mut regions: Vec<(u32, Vec<u32>)> = Vec::new();
    for u in adjacency.k_ring(v as u32, ring_depth) {
        let Some(label) = faces.labels[u as usize] else {
            continue;
        };
        if (positions[u as usize] - origin).length() > radius {
            continue;
        }
        match regions.iter_mut().find(|(l, _)| *l == label) {
            Some((_, members)) => members.push(u),
            None => regions.push((label, vec![u])),
        }
    }

    // A snap target lies on its sides' planes and within the movement clamp
    // of the vertex, so a plane passing further away than the clamp cannot
    // take part in an accepted snap.
    let reach = config.max_move_cells * cell;
    let in_reach = |plane: &SidePlane| plane.normal.dot(plane.centroid - origin).abs() <= reach;

    let mut planes: Vec<SidePlane> = Vec::new();
    for (_, members) in &regions {
        if let Some(plane) = side_plane(positions, members, cell, config) {
            planes.push(plane);
            continue;
        }
        let pieces = connected_pieces(adjacency, faces, members);
        if pieces.len() < 2 {
            continue;
        }
        // Only the pieces in reach become sides. A region that had to be
        // split is bigger than one face here, and its other pieces can be
        // faces at the far edge of the radius that have nothing to do with
        // this vertex; with more support than a near side they would be
        // picked ahead of it.
        planes.extend(
            pieces
                .iter()
                .filter_map(|piece| side_plane(positions, piece, cell, config))
                .filter(in_reach)
                .map(|plane| SidePlane {
                    split: true,
                    ..plane
                }),
        );
    }
    planes
}

/// The plane through one side's vertices, if the side qualifies: enough
/// support and a tight fit.
fn side_plane(
    positions: &[DVec3],
    members: &[u32],
    cell: f64,
    config: &SnapConfig,
) -> Option<SidePlane> {
    if members.len() < config.min_side_points {
        return None;
    }
    let pts: Vec<DVec3> = members.iter().map(|&u| positions[u as usize]).collect();
    let fit = fit_plane(&pts)?;
    if fit.rms_residual / cell > config.max_side_residual_cells {
        return None;
    }
    // PCA normal sign is arbitrary; the intersection solves are
    // sign-agnostic and verification orients per-side later.
    Some(SidePlane {
        normal: fit.normal,
        centroid: fit.centroid,
        support: pts.len(),
        split: false,
    })
}

/// Split `members` into the pieces connected through mesh edges that stay on
/// one face ([`SmoothFaces::joined`]), using `members` only. Connections that
/// exist only outside the set do not count: that is what separates the two
/// faces of a crease when they are joined smoothly somewhere else. Pieces
/// come out in the order of their first member.
fn connected_pieces(
    adjacency: &MeshAdjacency,
    faces: SmoothFaces,
    members: &[u32],
) -> Vec<Vec<u32>> {
    let mut by_vertex: Vec<(u32, usize)> = members.iter().copied().zip(0..).collect();
    by_vertex.sort_unstable();
    let mut parent: Vec<usize> = (0..members.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (i, &u) in members.iter().enumerate() {
        for &w in adjacency.neighbors(u) {
            if w <= u || !faces.joined(u, w) {
                continue;
            }
            if let Ok(at) = by_vertex.binary_search_by_key(&w, |&(vertex, _)| vertex) {
                let (a, b) = (root(&mut parent, i), root(&mut parent, by_vertex[at].1));
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let mut pieces: Vec<(usize, Vec<u32>)> = Vec::new();
    for (i, &u) in members.iter().enumerate() {
        let piece = root(&mut parent, i);
        match pieces.iter_mut().find(|(p, _)| *p == piece) {
            Some((_, vertices)) => vertices.push(u),
            None => pieces.push((piece, vec![u])),
        }
    }
    pieces.into_iter().map(|(_, vertices)| vertices).collect()
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

/// Whether every side's actual surface passes by `target`: the check that a
/// snap target sits where its sides really meet, not where their fitted
/// planes would meet if extended.
///
/// For each side, the occupancy boundary must cross a short probe along the
/// side's normal. The probe is shifted off the target, away from the other
/// sides' surfaces, so that it crosses this side's surface only. Which way
/// that is depends on the edge: behind the other side at a convex edge, in
/// front of it at a concave one. Both are tried.
fn sides_meet_at(
    target: DVec3,
    outward: &[DVec3],
    cell: f64,
    config: &SnapConfig,
    is_inside: &dyn Fn(DVec3) -> bool,
) -> bool {
    let bracket = config.refine_bracket_cells * cell;
    let offset_len = config.refine_offset_cells * cell;
    outward.iter().enumerate().all(|(i, &n)| {
        let others: Vec<DVec3> = outward
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &m)| m)
            .collect();
        (0..1usize << others.len()).any(|signs| {
            let shift: DVec3 = others
                .iter()
                .enumerate()
                .map(|(k, &m)| if signs >> k & 1 == 0 { -m } else { m })
                .sum();
            let Some(shift) = shift.try_normalize() else {
                return false;
            };
            let base = target + shift * offset_len;
            is_inside(base - n * bracket) && !is_inside(base + n * bracket)
        })
    })
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
    use crate::sharp_features::adjacency::grid_mesh;
    use crate::sharp_features::fit::{VertexFit, ring_fits};
    use crate::sharp_features::segmentation::{SegmentationConfig, segment_regions};

    fn plane(normal: DVec3, centroid: DVec3) -> SidePlane {
        SidePlane {
            normal: normal.normalize(),
            centroid,
            support: 10,
            split: false,
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

    /// No sampler exists for a synthetic mesh, so nothing can be verified.
    fn unverified() -> SnapConfig {
        SnapConfig {
            verify_delta_cells: 0.0,
            ..SnapConfig::default()
        }
    }

    /// A pocket cut into a slab (top face z = 0), seen at its inside corner:
    /// walls x = 0 and y = 0, joined by a fillet of radius `fillet`.
    fn pocket_corner(fillet: f64) -> impl Fn(DVec3) -> bool + Send + Sync {
        move |p: DVec3| {
            let in_fillet = p.x < fillet
                && p.y < fillet
                && (p.x - fillet).powi(2) + (p.y - fillet).powi(2) > fillet * fillet;
            let in_pocket = p.x >= 0.0 && p.y >= 0.0 && !in_fillet;
            p.z <= 0.0 && !in_pocket
        }
    }

    /// Convex and concave edges alike: the top face meets each wall at a
    /// convex edge, the walls meet each other at a concave one.
    #[test]
    fn sides_meet_at_a_real_corner() {
        let config = SnapConfig::default();
        let model = pocket_corner(0.0);
        let sides = [DVec3::X, DVec3::Y, DVec3::Z];
        assert!(sides_meet_at(DVec3::ZERO, &sides, 1.0, &config, &model));
        // Along the concave edge between the two walls, below the top face.
        let on_edge = DVec3::new(0.0, 0.0, -5.0);
        assert!(sides_meet_at(on_edge, &sides[..2], 1.0, &config, &model));
        // Along a convex edge between the top face and one wall.
        let on_rim = DVec3::new(0.0, 5.0, 0.0);
        assert!(sides_meet_at(
            on_rim,
            &[DVec3::X, DVec3::Z],
            1.0,
            &config,
            &model
        ));
    }

    /// The walls of a filleted pocket never meet: their planes cross at a
    /// point inside the material, on the top face, where neither wall is.
    /// (The inside/outside check along the mean normal passes there, because
    /// the top face does run through the point.)
    #[test]
    fn sides_do_not_meet_beyond_a_fillet() {
        let config = SnapConfig::default();
        let model = pocket_corner(4.0);
        let sides = [DVec3::X, DVec3::Y, DVec3::Z];
        let mean = DVec3::ONE.normalize() * config.verify_delta_cells;
        assert!(
            model(-mean) && !model(mean),
            "the mean-normal check should pass"
        );
        assert!(!sides_meet_at(DVec3::ZERO, &sides, 1.0, &config, &model));
        // Clear of the fillet the same walls are found again.
        let on_rim = DVec3::new(0.0, 8.0, 0.0);
        assert!(sides_meet_at(
            on_rim,
            &[DVec3::X, DVec3::Z],
            1.0,
            &config,
            &model
        ));
    }

    /// A target off the feature fails too: one side's surface is elsewhere.
    #[test]
    fn sides_do_not_meet_off_the_edge() {
        let config = SnapConfig::default();
        let model = pocket_corner(0.0);
        let off_rim = DVec3::new(-1.5, 5.0, 0.0);
        assert!(!sides_meet_at(
            off_rim,
            &[DVec3::X, DVec3::Z],
            1.0,
            &config,
            &model
        ));
    }

    /// The mesh of a sharp pocket corner, all under one label, so its faces
    /// are told apart as pieces. Against a model with that sharp corner the
    /// corner vertex snaps. Against a model whose pocket walls are joined by a
    /// fillet it must not: the wall the mesh suggests is not there.
    #[test]
    fn sides_split_from_one_region_must_meet_in_the_model() {
        // Four patches: the top face round the pocket (two), and its walls.
        let patches = [
            (
                grid_mesh(7, 13, |i, j| {
                    DVec3::new(i as f64 - 6.0, j as f64 - 6.0, 0.0)
                }),
                DVec3::Z,
            ),
            (
                grid_mesh(7, 7, |i, j| DVec3::new(i as f64, j as f64 - 6.0, 0.0)),
                DVec3::Z,
            ),
            (
                grid_mesh(7, 7, |i, j| DVec3::new(0.0, i as f64, -(j as f64))),
                DVec3::X,
            ),
            (
                grid_mesh(7, 7, |i, j| DVec3::new(i as f64, 0.0, -(j as f64))),
                DVec3::Y,
            ),
        ];
        // Weld the patches along their shared edges; a shared vertex keeps
        // the normal of the first patch that has it.
        let mut index_of = std::collections::HashMap::new();
        let (mut positions, mut indices, mut normals) = (Vec::new(), Vec::new(), Vec::new());
        for ((patch_positions, patch_indices), normal) in &patches {
            let remap: Vec<u32> = patch_positions
                .iter()
                .map(|p| {
                    let key = (p.x as i64, p.y as i64, p.z as i64);
                    *index_of.entry(key).or_insert_with(|| {
                        positions.push(*p);
                        normals.push(*normal);
                        positions.len() as u32 - 1
                    })
                })
                .collect();
            indices.extend(patch_indices.iter().map(|&i| remap[i as usize]));
        }
        let adjacency = MeshAdjacency::build(positions.len(), &indices);
        let labels = vec![Some(0); positions.len()];
        let fits: Vec<Option<VertexFit>> = normals
            .iter()
            .map(|&normal| {
                Some(VertexFit {
                    normal,
                    residual_cells: 0.0,
                })
            })
            .collect();
        let seg_config = SegmentationConfig::default();
        let faces = SmoothFaces {
            labels: &labels,
            fits: &fits,
            config: &seg_config,
        };
        let corner = index_of[&(0, 0, 0)] as usize;

        for (fillet, snaps) in [(0.0, true), (4.0, false)] {
            let model = pocket_corner(fillet);
            let result = snap_feature_vertices(
                &positions,
                &adjacency,
                faces,
                1.0,
                &SnapConfig::default(),
                Some(&model),
            );
            assert_eq!(
                result.snapped[corner].is_some(),
                snaps,
                "fillet {fillet}: {:?}",
                result.stats
            );
        }
    }

    /// End-to-end on the synthetic tent: crease vertices must land exactly on
    /// the analytic crease line.
    #[test]
    fn tent_crease_vertices_snap_onto_the_crease_line() {
        let crease = 7usize;
        let n = 15usize;
        let (positions, indices) = grid_mesh(n, n, |i, j| {
            if i <= crease {
                DVec3::new(i as f64, j as f64, 0.0)
            } else {
                DVec3::new(crease as f64, j as f64, (i - crease) as f64)
            }
        });
        let adjacency = MeshAdjacency::build(positions.len(), &indices);
        let fits = ring_fits(&positions, &adjacency, &[], 1.0, 1);
        let seg_config = SegmentationConfig::default();
        let seg = segment_regions(&adjacency, &fits, &seg_config);
        let faces = SmoothFaces {
            labels: &seg.labels,
            fits: &fits,
            config: &seg_config,
        };
        let result = snap_feature_vertices(&positions, &adjacency, faces, 1.0, &unverified(), None);

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

    /// An edge that is sharp along part of its length and filleted along the
    /// rest: region growth walks round the fillet, so the two faces carry one
    /// label, and the sharp part has that label on both sides. It must snap
    /// all the same.
    #[test]
    fn crease_between_two_faces_of_one_region_snaps() {
        // Profile across the edge, by arc length `s` from the crease: flat
        // (z = 0) for s < 0, vertical (x = 20) for s > 0, joined by a fillet
        // of radius `r`. Sampled half a step off the crease, so no vertex
        // sits on it and the mesh cuts the corner as a mesher's would.
        let profile = |s: f64, r: f64| -> (f64, f64) {
            let half_arc = 0.25 * std::f64::consts::PI * r;
            if s < -half_arc {
                (20.0 - r + s + half_arc, 0.0)
            } else if s > half_arc {
                (20.0, r + s - half_arc)
            } else {
                let theta = (s + half_arc) / r;
                (20.0 - r + r * theta.sin(), r - r * theta.cos())
            }
        };
        // Sharp for rows 0..15, then the fillet grows to a radius of ten
        // cells (a turn of under 6 degrees per step) and stays there.
        let radius = |j: usize| (j as f64 - 14.0).clamp(0.0, 10.0);
        let (nx, ny) = (40usize, 37usize);
        let (positions, indices) = grid_mesh(nx, ny, |i, j| {
            let s = i as f64 - 19.5;
            let r = radius(j);
            let (x, z) = if r == 0.0 {
                if s < 0.0 { (20.0 + s, 0.0) } else { (20.0, s) }
            } else {
                profile(s, r)
            };
            DVec3::new(x, j as f64, z)
        });
        let adjacency = MeshAdjacency::build(positions.len(), &indices);
        let fits = ring_fits(&positions, &adjacency, &[], 1.0, 1);
        let seg_config = SegmentationConfig::default();
        let seg = segment_regions(&adjacency, &fits, &seg_config);

        // The premise: one region covers both faces beside the sharp part.
        let flat = seg.labels[5 * nx + 10].expect("flat face claimed");
        let vertical = seg.labels[5 * nx + 30].expect("vertical face claimed");
        assert_eq!(
            flat, vertical,
            "the fillet should join the two faces into one region"
        );

        let faces = SmoothFaces {
            labels: &seg.labels,
            fits: &fits,
            config: &seg_config,
        };
        let result = snap_feature_vertices(&positions, &adjacency, faces, 1.0, &unverified(), None);

        // Both columns beside the sharp crease land on it: x = 20, z = 0.
        for j in 3..=10 {
            for i in [19, 20] {
                let v = j * nx + i;
                assert_eq!(
                    seg.labels[v], None,
                    "({i},{j}) should be in the feature zone"
                );
                assert_eq!(
                    result.snapped[v],
                    Some(SnapKind::Edge),
                    "({i},{j}) was not snapped: {:?}",
                    result.stats
                );
                let p = result.positions[v];
                assert!(
                    (p.x - 20.0).abs() < 1e-9 && p.z.abs() < 1e-9,
                    "({i},{j}) landed off the crease: {p:?}"
                );
            }
        }
    }

    /// Hand-made fits and labels for a grid: `face(i, j)` gives a vertex's
    /// label and fitted normal, or `None` for an unclaimed vertex.
    fn hand_segmented(
        nx: usize,
        ny: usize,
        face: impl Fn(usize, usize) -> Option<(u32, DVec3)>,
    ) -> (Vec<Option<u32>>, Vec<Option<VertexFit>>) {
        let mut labels = Vec::new();
        let mut fits = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let claimed = face(i, j);
                labels.push(claimed.map(|(label, _)| label));
                fits.push(claimed.map(|(_, normal)| VertexFit {
                    normal,
                    residual_cells: 0.0,
                }));
            }
        }
        (labels, fits)
    }

    /// A grid-aligned crease leaves no unclaimed band: the two faces are
    /// claimed right up to each other. When they also share a label, the only
    /// sign of the crease is the growth gate failing between neighbors. Those
    /// neighbors must be candidates, and must snap.
    #[test]
    fn single_label_crease_without_an_unclaimed_band_snaps() {
        let crease = 7usize;
        let n = 15usize;
        let (positions, indices) = grid_mesh(n, n, |i, j| {
            if i <= crease {
                DVec3::new(i as f64, j as f64, 0.0)
            } else {
                DVec3::new(crease as f64, j as f64, (i - crease) as f64)
            }
        });
        let adjacency = MeshAdjacency::build(positions.len(), &indices);
        let (labels, fits) = hand_segmented(n, n, |i, _| {
            Some((0, if i <= crease { DVec3::Z } else { DVec3::X }))
        });
        let seg_config = SegmentationConfig::default();
        let faces = SmoothFaces {
            labels: &labels,
            fits: &fits,
            config: &seg_config,
        };
        let result = snap_feature_vertices(&positions, &adjacency, faces, 1.0, &unverified(), None);

        // Candidates are the two columns the crease runs between, no more.
        assert_eq!(result.stats.candidates, 2 * n);
        for j in 3..n - 3 {
            for i in [crease, crease + 1] {
                let v = j * n + i;
                assert_eq!(
                    result.snapped[v],
                    Some(SnapKind::Edge),
                    "({i},{j}) was not snapped: {:?}",
                    result.stats
                );
                let p = result.positions[v];
                assert!(
                    (p.x - 7.0).abs() < 1e-9 && p.z.abs() < 1e-9,
                    "({i},{j}) landed off the crease: {p:?}"
                );
            }
            assert_eq!(
                result.snapped[j * n + 3],
                None,
                "a face vertex must not move"
            );
        }
    }

    /// A step two cells high: lower tread, riser, upper tread, with both
    /// treads under one label (they are parallel, and joined somewhere out of
    /// view). Seen from the lower crease, the upper tread has more gathered
    /// vertices than the thinly claimed lower one, but its plane passes two
    /// cells away, beyond the movement clamp. Splitting the label must not
    /// put it in the lower tread's place as a side.
    #[test]
    fn a_side_out_of_reach_does_not_outrank_a_near_one() {
        let (nx, ny) = (18usize, 15usize);
        let (positions, indices) = grid_mesh(nx, ny, |i, j| {
            let y = j as f64;
            match i {
                0..=7 => DVec3::new(i as f64, y, 0.0),
                _ => DVec3::new((i - 1) as f64, y, 2.0),
            }
        });
        let adjacency = MeshAdjacency::build(positions.len(), &indices);
        let row = 7usize;
        let (labels, fits) = hand_segmented(nx, ny, |i, j| match i {
            // Lower tread: claimed only in a small patch beside the crease.
            5..=6 if j.abs_diff(row) <= 1 => Some((0, DVec3::Z)),
            0..=6 => None,
            // Riser: the two crease columns, which both lie in its plane.
            7..=8 => Some((1, DVec3::X)),
            _ => Some((0, DVec3::Z)),
        });
        let seg_config = SegmentationConfig::default();
        let faces = SmoothFaces {
            labels: &labels,
            fits: &fits,
            config: &seg_config,
        };

        // The premise: from the lower crease the upper tread outnumbers the
        // lower one, and each would qualify as a side on support.
        let v = row * nx + 7;
        let config = unverified();
        let count = |upper: bool| {
            adjacency
                .k_ring(v as u32, 4)
                .into_iter()
                .filter(|&u| {
                    let p = positions[u as usize];
                    labels[u as usize] == Some(0)
                        && (p.z > 1.0) == upper
                        && (p - positions[v]).length() <= 3.0
                })
                .count()
        };
        assert!(
            count(false) >= config.min_side_points,
            "lower: {}",
            count(false)
        );
        assert!(
            count(true) > count(false),
            "upper {} vs lower {}",
            count(true),
            count(false)
        );

        let result = snap_feature_vertices(&positions, &adjacency, faces, 1.0, &config, None);
        assert_eq!(
            result.snapped[v],
            Some(SnapKind::Edge),
            "{:?}",
            result.stats
        );
        assert!((result.positions[v] - DVec3::new(7.0, row as f64, 0.0)).length() < 1e-9);
    }
}
