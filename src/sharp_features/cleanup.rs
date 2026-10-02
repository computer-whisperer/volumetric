//! Post-snap band cleanup: weld snapped vertices that landed on (nearly) the
//! same feature point and drop the triangles that collapse as a result.
//!
//! Snapping projects both rows of the feature band onto the same edge line,
//! leaving pairs of vertices a fraction of a cell apart and folded sliver
//! triangles between them. Welding those pairs removes the slivers without
//! touching any unsnapped vertex, so the blast radius stays confined to the
//! feature zone. Only triangles that lose a vertex to welding (two corners
//! mapping to the same output vertex) are dropped; that cannot open a hole,
//! because every neighbor sharing a welded edge sees the same collapse.
//!
//! The weld can leave a "cap": a triangle whose three vertices lie on one
//! line (two welded vertices along a feature and a third between them). It
//! has no area and no normal, and it hides the two-face edge the feature
//! actually runs along. [`flip_zero_area_caps`] removes it by flipping its
//! long edge, which changes no geometry.

use crate::sharp_features::snap::SnapKind;
use glam::DVec3;
use std::collections::HashMap;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CleanupConfig {
    /// Snapped vertices closer than this (cell units) are welded together.
    /// Must stay well below the inter-vertex spacing (~1 cell) so only
    /// cross-band pairs merge, never along-feature neighbors.
    pub weld_radius_cells: f64,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            weld_radius_cells: 0.25,
        }
    }
}

pub struct CleanupResult {
    pub positions: Vec<DVec3>,
    pub indices: Vec<u32>,
    /// Old vertex index -> new vertex index.
    pub remap: Vec<u32>,
    pub welded_vertices: usize,
    pub dropped_triangles: usize,
    /// Zero-area caps removed by an edge flip.
    pub flipped_caps: usize,
}

/// Weld snapped vertices within the configured radius and drop collapsed
/// triangles. Unsnapped vertices are never moved or merged.
pub fn weld_snapped_vertices(
    positions: &[DVec3],
    indices: &[u32],
    snapped: &[Option<SnapKind>],
    cell: f64,
    config: &CleanupConfig,
) -> CleanupResult {
    // Only snapped vertices ever weld; with none, the pass is an identity.
    // Worth an early exit: smooth models (no features to snap) otherwise pay
    // the full union-find + remap walk over millions of vertices for nothing.
    if !snapped.iter().any(|s| s.is_some()) {
        return CleanupResult {
            positions: positions.to_vec(),
            indices: indices.to_vec(),
            remap: (0..positions.len() as u32).collect(),
            welded_vertices: 0,
            dropped_triangles: 0,
            flipped_caps: 0,
        };
    }

    let radius = config.weld_radius_cells * cell;
    let mut parent: Vec<u32> = (0..positions.len() as u32).collect();

    fn find(parent: &mut [u32], v: u32) -> u32 {
        let mut root = v;
        while parent[root as usize] != root {
            root = parent[root as usize];
        }
        let mut cur = v;
        while parent[cur as usize] != root {
            let next = parent[cur as usize];
            parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    // Spatial hash over snapped vertices; bucket size = weld radius, so all
    // partners of a vertex live in its own or one of the 26 adjacent buckets.
    let bucket_of = |p: DVec3| -> (i64, i64, i64) {
        (
            (p.x / radius).floor() as i64,
            (p.y / radius).floor() as i64,
            (p.z / radius).floor() as i64,
        )
    };
    let mut buckets: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for v in 0..positions.len() as u32 {
        if snapped[v as usize].is_none() {
            continue;
        }
        let (bx, by, bz) = bucket_of(positions[v as usize]);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(candidates) = buckets.get(&(bx + dx, by + dy, bz + dz)) {
                        for &u in candidates {
                            if (positions[u as usize] - positions[v as usize]).length() <= radius {
                                let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
                                if ru != rv {
                                    parent[rv as usize] = ru;
                                }
                            }
                        }
                    }
                }
            }
        }
        buckets.entry((bx, by, bz)).or_default().push(v);
    }

    // Cluster positions: mean of members, except that corner snaps win over
    // edge snaps -- corners are exact feature points and must not be dragged
    // along the edge by their welded neighbors. Only snapped vertices are
    // ever unioned, so the map covers just those; every other vertex is a
    // singleton keeping its own position.
    let mut cluster_sum: HashMap<u32, (DVec3, usize, DVec3, usize)> = HashMap::new();
    for v in 0..positions.len() as u32 {
        if snapped[v as usize].is_none() {
            continue;
        }
        let root = find(&mut parent, v);
        let entry = cluster_sum
            .entry(root)
            .or_insert((DVec3::ZERO, 0, DVec3::ZERO, 0));
        entry.0 += positions[v as usize];
        entry.1 += 1;
        if snapped[v as usize] == Some(SnapKind::Corner) {
            entry.2 += positions[v as usize];
            entry.3 += 1;
        }
    }

    // Assign new indices to cluster roots in original order.
    let mut remap = vec![u32::MAX; positions.len()];
    let mut new_positions: Vec<DVec3> = Vec::with_capacity(positions.len());
    for v in 0..positions.len() as u32 {
        if find(&mut parent, v) == v {
            remap[v as usize] = new_positions.len() as u32;
            new_positions.push(match cluster_sum.get(&v) {
                Some(&(_, _, corner_sum, corner_count)) if corner_count > 0 => {
                    corner_sum / corner_count as f64
                }
                Some(&(sum, count, _, _)) => sum / count as f64,
                None => positions[v as usize],
            });
        }
    }
    for v in 0..positions.len() as u32 {
        let root = find(&mut parent, v);
        remap[v as usize] = remap[root as usize];
    }
    let welded_vertices = positions.len() - new_positions.len();

    // Rewrite triangles; drop those that collapsed onto a welded vertex.
    let mut new_indices = Vec::with_capacity(indices.len());
    let mut dropped_triangles = 0usize;
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (
            remap[tri[0] as usize],
            remap[tri[1] as usize],
            remap[tri[2] as usize],
        );
        if a == b || b == c || c == a {
            dropped_triangles += 1;
            continue;
        }
        new_indices.extend_from_slice(&[a, b, c]);
    }

    let flipped_caps = flip_zero_area_caps(&new_positions, &mut new_indices);

    CleanupResult {
        positions: new_positions,
        indices: new_indices,
        remap,
        welded_vertices,
        dropped_triangles,
        flipped_caps,
    }
}

/// A triangle's unit normal, or `None` when it has no usable area: its
/// height over its longest edge is under a millionth of that edge.
pub fn face_unit_normal(a: DVec3, b: DVec3, c: DVec3) -> Option<DVec3> {
    let cross = (b - a).cross(c - a);
    let longest_sq = (b - a)
        .length_squared()
        .max((c - b).length_squared())
        .max((a - c).length_squared());
    // |cross| = longest edge x height.
    (cross.length_squared() > 1e-12 * longest_sq * longest_sq).then(|| cross.normalize())
}

/// Remove zero-area caps: triangles `(a, b, c)` with `b` on the segment
/// `a`-`c`. The cap and the face on the other side of `a`-`c` (third vertex
/// `z`) are replaced by `(a, b, z)` and `(b, c, z)`, which cover exactly the
/// same surface. Returns how many were removed.
///
/// A cap is left alone when the flip is not clean: its long edge is not
/// shared with exactly one other face, or `b`-`z` is an edge already.
pub fn flip_zero_area_caps(positions: &[DVec3], indices: &mut [u32]) -> usize {
    let corners =
        |indices: &[u32], t: usize| [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
    let is_cap = |indices: &[u32], t: usize| {
        let [a, b, c] = corners(indices, t).map(|v| positions[v as usize]);
        face_unit_normal(a, b, c).is_none()
    };
    let tri_count = indices.len() / 3;
    let mut flipped = 0usize;
    // A flip can expose another cap behind it (a run of collinear vertices),
    // so repeat; each round removes at least one or stops.
    for _ in 0..8 {
        let caps: Vec<usize> = (0..tri_count).filter(|&t| is_cap(indices, t)).collect();
        if caps.is_empty() {
            break;
        }
        // Faces around the caps' vertices, found in one scan.
        let mut around: HashMap<u32, Vec<u32>> = HashMap::new();
        for &t in &caps {
            for v in corners(indices, t) {
                around.entry(v).or_default();
            }
        }
        for t in 0..tri_count {
            for v in corners(indices, t) {
                if let Some(faces) = around.get_mut(&v) {
                    faces.push(t as u32);
                }
            }
        }

        let mut touched: Vec<bool> = vec![false; tri_count];
        let mut flipped_this_round = 0usize;
        for &t in &caps {
            if touched[t] {
                continue;
            }
            // Rotate so `b` is the vertex opposite the longest edge.
            let tri = corners(indices, t);
            let longest = (0..3)
                .max_by(|&i, &j| {
                    let len = |k: usize| {
                        (positions[tri[(k + 1) % 3] as usize] - positions[tri[k] as usize])
                            .length_squared()
                    };
                    len(i).total_cmp(&len(j))
                })
                .expect("three edges");
            let (a, c, b) = (tri[longest], tri[(longest + 1) % 3], tri[(longest + 2) % 3]);
            let (pa, pb, pc) = (
                positions[a as usize],
                positions[b as usize],
                positions[c as usize],
            );
            let along = (pb - pa).dot(pc - pa);
            if !(along > 0.0 && along < (pc - pa).length_squared()) {
                continue; // `b` coincides with an end: not a cap
            }
            // The one other face on edge a-c.
            let mut across = around[&a].iter().copied().filter(|&f| {
                f as usize != t && !touched[f as usize] && corners(indices, f as usize).contains(&c)
            });
            let (Some(f), None) = (across.next(), across.next()) else {
                continue;
            };
            let f = f as usize;
            let face = corners(indices, f);
            let Some(&z) = face.iter().find(|&&v| v != a && v != c) else {
                continue;
            };
            let b_meets_z = around[&b]
                .iter()
                .any(|&g| !touched[g as usize] && corners(indices, g as usize).contains(&z));
            if z == b || b_meets_z {
                continue;
            }
            // `face` runs p -> q -> z with {p, q} = {a, c}; b goes between.
            let k = face.iter().position(|&v| v == z).expect("z is a corner");
            let (p, q) = (face[(k + 1) % 3], face[(k + 2) % 3]);
            indices[t * 3..t * 3 + 3].copy_from_slice(&[p, b, z]);
            indices[f * 3..f * 3 + 3].copy_from_slice(&[b, q, z]);
            // Adjacency around these vertices is stale until the next round.
            for v in [a, b, c, z] {
                if let Some(faces) = around.get(&v) {
                    for &g in faces {
                        touched[g as usize] = true;
                    }
                }
            }
            touched[t] = true;
            touched[f] = true;
            flipped_this_round += 1;
        }
        flipped += flipped_this_round;
        if flipped_this_round == 0 {
            break;
        }
    }
    flipped
}

/// Number of boundary edges: undirected edges used by exactly one triangle.
/// Zero for a watertight mesh (non-manifold edges with 3+ triangles are
/// allowed by the meshing contract and do not count).
pub fn boundary_edge_count(indices: &[u32]) -> usize {
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            *counts.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    counts.values().filter(|&&c| c == 1).count()
}

/// Triangles whose face normal points against the outward reference direction
/// (sum of their vertices' reference normals). Reference normals come from the
/// mesher's accumulated normals, which are outward by construction; a
/// significant count here means winding damage.
///
/// Triangles with area below `min_area` are skipped: a near-degenerate
/// sliver's normal is numerical noise, so its sign carries no winding
/// information (and it covers no pixels).
pub fn inward_facing_count(
    positions: &[DVec3],
    indices: &[u32],
    reference_normals: &[DVec3],
    min_area: f64,
) -> usize {
    let mut count = 0;
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        let reference = reference_normals[a] + reference_normals[b] + reference_normals[c];
        if face.length() / 2.0 >= min_area
            && reference.length_squared() > 1e-12
            && face.dot(reference) < 0.0
        {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square (0,1,2,3) whose bottom edge 0-1 carries a midpoint vertex 4
    /// used only by the triangle below it: the cap (0, 4, 1) sits between
    /// the square's lower triangle and the triangles under the edge.
    #[test]
    fn cap_is_flipped_away_without_changing_the_surface() {
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(2.0, 2.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),  // on edge 0-1
            DVec3::new(1.0, -1.0, 0.0), // below
        ];
        let mut indices = vec![
            0, 1, 2, // above the edge, on the long side
            0, 2, 3, //
            0, 4, 1, // the cap
            0, 5, 4, // below, meeting the midpoint
            4, 5, 1, //
        ];
        let area = |indices: &[u32]| -> f64 {
            indices
                .chunks_exact(3)
                .map(|t| {
                    let p = |k: usize| positions[t[k] as usize];
                    (p(1) - p(0)).cross(p(2) - p(0)).z / 2.0
                })
                .sum()
        };
        let before = area(&indices);
        let outline = boundary_edge_count(&indices);
        assert_eq!(flip_zero_area_caps(&positions, &mut indices), 1);
        assert_eq!(indices.len(), 15, "two triangles replaced by two");
        assert!((area(&indices) - before).abs() < 1e-12);
        for t in indices.chunks_exact(3) {
            let p = |k: usize| positions[t[k] as usize];
            assert!(
                (p(1) - p(0)).cross(p(2) - p(0)).z > 0.1,
                "every triangle has area and faces up: {t:?}"
            );
        }
        assert_eq!(boundary_edge_count(&indices), outline, "same outline");
    }

    #[test]
    fn mesh_without_caps_is_untouched() {
        let (positions, mut indices) = tetrahedron();
        let before = indices.clone();
        assert_eq!(flip_zero_area_caps(&positions, &mut indices), 0);
        assert_eq!(indices, before);
    }

    // A closed tetrahedron: 4 vertices, 4 triangles, watertight.
    fn tetrahedron() -> (Vec<DVec3>, Vec<u32>) {
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        ];
        let indices = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
        (positions, indices)
    }

    #[test]
    fn watertight_mesh_has_no_boundary_edges() {
        let (_, indices) = tetrahedron();
        assert_eq!(boundary_edge_count(&indices), 0);
        // Removing one face exposes exactly its three edges.
        assert_eq!(boundary_edge_count(&indices[3..]), 3);
    }

    #[test]
    fn welding_close_snapped_pair_preserves_watertightness() {
        // Two tetrahedra sharing... simpler: a tetrahedron with one vertex
        // split into two coincident snapped copies. Vertex 1 is duplicated as
        // vertex 4 offset by a hair; triangles reference both copies, which
        // models the post-snap band exactly.
        let (mut positions, _) = tetrahedron();
        positions.push(positions[1] + DVec3::new(1e-3, 0.0, 0.0));
        // Fan re-wired so both copies appear; the split leaves the mesh with
        // boundary edges (a crack), which welding must close.
        let indices = vec![0, 2, 1, 0, 4, 3, 0, 3, 2, 1, 2, 3, 4, 2, 3];
        assert_ne!(boundary_edge_count(&indices), 0, "precondition: cracked");

        let snapped = vec![None, Some(SnapKind::Edge), None, None, Some(SnapKind::Edge)];
        let result = weld_snapped_vertices(
            &positions,
            &indices,
            &snapped,
            1.0,
            &CleanupConfig::default(),
        );
        assert_eq!(result.welded_vertices, 1);
        assert_eq!(boundary_edge_count(&result.indices), 0, "crack closed");
    }

    #[test]
    fn sliver_between_welded_pair_collapses() {
        // Vertices 1 and 2 are the cross-band pair; triangle (0,1,2) is the
        // sliver between them and (1,3,2) hangs off the pair.
        let positions = vec![
            DVec3::ZERO,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.001, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let snapped = vec![None, Some(SnapKind::Edge), Some(SnapKind::Edge), None];
        let result = weld_snapped_vertices(
            &positions,
            &indices,
            &snapped,
            1.0,
            &CleanupConfig::default(),
        );
        assert_eq!(result.welded_vertices, 1);
        assert_eq!(result.dropped_triangles, 2, "both collapsed tris dropped");
        assert_eq!(result.positions.len(), 3);
        assert!(result.indices.is_empty());
    }

    #[test]
    fn unsnapped_vertices_are_never_welded() {
        let positions = vec![DVec3::ZERO, DVec3::new(1e-6, 0.0, 0.0), DVec3::Y, DVec3::Z];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let snapped = vec![None, None, None, None];
        let result = weld_snapped_vertices(
            &positions,
            &indices,
            &snapped,
            1.0,
            &CleanupConfig::default(),
        );
        assert_eq!(result.welded_vertices, 0);
        assert_eq!(result.dropped_triangles, 0);
        assert_eq!(result.positions.len(), 4);
    }

    #[test]
    fn corner_position_wins_in_mixed_clusters() {
        let positions = vec![
            DVec3::new(0.1, 0.0, 0.0),  // edge-snapped
            DVec3::new(0.0, 0.0, 0.0),  // corner-snapped: the exact feature
            DVec3::new(0.05, 0.1, 0.0), // edge-snapped
            DVec3::Y,
            DVec3::Z,
        ];
        let indices = vec![0, 3, 4, 1, 4, 3, 2, 3, 4];
        let snapped = vec![
            Some(SnapKind::Edge),
            Some(SnapKind::Corner),
            Some(SnapKind::Edge),
            None,
            None,
        ];
        let result = weld_snapped_vertices(
            &positions,
            &indices,
            &snapped,
            1.0,
            &CleanupConfig::default(),
        );
        assert_eq!(result.welded_vertices, 2);
        let merged = result.positions[result.remap[1] as usize];
        assert!(
            (merged - DVec3::ZERO).length() < 1e-12,
            "cluster should sit at the corner snap, got {merged:?}"
        );
    }

    #[test]
    fn inward_facing_detects_inverted_winding() {
        let positions = vec![DVec3::ZERO, DVec3::X, DVec3::Y];
        let up = vec![DVec3::Z; 3];
        // CCW seen from +Z: outward. Reversed: inward.
        assert_eq!(inward_facing_count(&positions, &[0, 1, 2], &up, 0.0), 0);
        assert_eq!(inward_facing_count(&positions, &[0, 2, 1], &up, 0.0), 1);
        // A triangle below the area floor is skipped regardless of winding.
        assert_eq!(inward_facing_count(&positions, &[0, 2, 1], &up, 10.0), 0);
    }
}
