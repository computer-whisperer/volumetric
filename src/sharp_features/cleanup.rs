//! Post-snap cleanup: turn the snap stage's per-vertex result into a valid
//! mesh again.
//!
//! Snapping moves each feature-zone vertex on its own. Both rows of the band
//! between two faces end up on the same crease line, so the triangles between
//! them are flattened, and a vertex can be carried past one that moved less.
//! Four steps deal with that, in order:
//!
//! 1. **Retract.** A snap that turns one of its triangles over (and that the
//!    steps below will not remove) is undone: the vertex goes back to where
//!    the mesher had it. This is the last gate of the snap's contract, so a
//!    snap that cannot be fitted into the mesh is not made.
//! 2. **Weld.** Snapped vertices within the weld radius of each other are the
//!    same feature point and become one vertex. Triangles that lose a corner
//!    to that are dropped; every neighbor sharing a welded edge sees the same
//!    collapse, so no hole opens.
//! 3. **Cancel.** Welding two vertices that were not joined by an edge folds
//!    the triangles between them onto each other. A pair on the same three
//!    vertices with opposite winding encloses nothing and is removed.
//! 4. **Flip caps.** A snapped vertex within the weld radius of the edge
//!    opposite it in a triangle is on that edge, and the triangle (a "cap")
//!    is what is left of a face the snap flattened. Flipping the edge
//!    replaces the cap and the face across the edge by two triangles that
//!    cover that face and meet at the vertex.
//!
//! Unsnapped vertices are never moved or merged, so the blast radius stays
//! confined to the feature zone.

use crate::sharp_features::snap::SnapKind;
use glam::DVec3;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CleanupConfig {
    /// Snapped vertices closer than this (cell units) are welded together,
    /// and a snapped vertex closer than this to an edge counts as on it.
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
    /// Vertices (old indices, ascending) whose snap was undone because it
    /// turned a triangle over. They are back at their `meshed` positions.
    pub retracted: Vec<u32>,
    pub welded_vertices: usize,
    /// Triangles removed: collapsed by the weld, or half of a cancelling pair.
    pub dropped_triangles: usize,
    /// Caps removed by an edge flip.
    pub flipped_caps: usize,
}

/// Make a valid mesh of the snap stage's result (see the module docs).
///
/// `meshed` are the vertex positions before the snap, `snapped_positions`
/// after it, and `snapped` says which vertices it moved.
pub fn clean_up_snaps(
    meshed: &[DVec3],
    snapped_positions: &[DVec3],
    indices: &[u32],
    snapped: &[Option<SnapKind>],
    cell: f64,
    config: &CleanupConfig,
) -> CleanupResult {
    // Only snapped vertices are ever touched; with none, the pass is an
    // identity. Worth an early exit: smooth models (no features to snap)
    // otherwise pay the remap walk over millions of vertices for nothing.
    if !snapped.iter().any(|s| s.is_some()) {
        return CleanupResult {
            positions: snapped_positions.to_vec(),
            indices: indices.to_vec(),
            remap: (0..snapped_positions.len() as u32).collect(),
            retracted: Vec::new(),
            welded_vertices: 0,
            dropped_triangles: 0,
            flipped_caps: 0,
        };
    }
    let radius = config.weld_radius_cells * cell;

    let mut positions = snapped_positions.to_vec();
    let mut snapped = snapped.to_vec();
    let (clusters, retracted) =
        retract_tangled_snaps(meshed, &mut positions, &mut snapped, indices, radius);

    // New indices go to cluster representatives in original order.
    let mut remap = vec![u32::MAX; positions.len()];
    let mut new_positions: Vec<DVec3> = Vec::with_capacity(positions.len());
    let mut on_feature: Vec<bool> = Vec::with_capacity(positions.len());
    for v in 0..positions.len() as u32 {
        if clusters.root[v as usize] == v {
            remap[v as usize] = new_positions.len() as u32;
            new_positions.push(clusters.position(v, &positions));
            on_feature.push(snapped[v as usize].is_some());
        }
    }
    for v in 0..positions.len() {
        remap[v] = remap[clusters.root[v] as usize];
    }
    let welded_vertices = positions.len() - new_positions.len();

    // Rewrite triangles; drop those that collapsed onto a welded vertex.
    let mut new_indices = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (
            remap[tri[0] as usize],
            remap[tri[1] as usize],
            remap[tri[2] as usize],
        );
        if a != b && b != c && c != a {
            new_indices.extend_from_slice(&[a, b, c]);
        }
    }
    drop_cancelling_pairs(&mut new_indices, &on_feature);
    let dropped_triangles = (indices.len() - new_indices.len()) / 3;

    let flipped_caps = flip_caps(&new_positions, &mut new_indices, &on_feature, radius);

    CleanupResult {
        positions: new_positions,
        indices: new_indices,
        remap,
        retracted,
        welded_vertices,
        dropped_triangles,
        flipped_caps,
    }
}

/// Snapped vertices within the weld radius of each other, joined into
/// clusters (transitively). Every unsnapped vertex is a cluster of its own.
struct Clusters {
    /// Each vertex's cluster representative.
    root: Vec<u32>,
    /// Where each cluster of snapped vertices sits, by representative.
    welded: HashMap<u32, DVec3>,
}

impl Clusters {
    fn build(positions: &[DVec3], snapped: &[Option<SnapKind>], radius: f64) -> Self {
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

        // Spatial hash over snapped vertices; bucket size = weld radius, so
        // all partners of a vertex live in its own or one of the 26 adjacent
        // buckets.
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
                        let Some(partners) = buckets.get(&(bx + dx, by + dy, bz + dz)) else {
                            continue;
                        };
                        for &u in partners {
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
            buckets.entry((bx, by, bz)).or_default().push(v);
        }

        // Cluster positions: mean of members, except that corner snaps win
        // over edge snaps. Corners are exact feature points and must not be
        // dragged along the edge by their welded neighbors.
        let mut sums: HashMap<u32, (DVec3, usize, DVec3, usize)> = HashMap::new();
        for v in 0..positions.len() as u32 {
            let Some(kind) = snapped[v as usize] else {
                continue;
            };
            let root = find(&mut parent, v);
            parent[v as usize] = root;
            let entry = sums.entry(root).or_insert((DVec3::ZERO, 0, DVec3::ZERO, 0));
            entry.0 += positions[v as usize];
            entry.1 += 1;
            if kind == SnapKind::Corner {
                entry.2 += positions[v as usize];
                entry.3 += 1;
            }
        }
        let welded = sums
            .into_iter()
            .map(|(root, (sum, count, corner_sum, corner_count))| {
                let position = if corner_count > 0 {
                    corner_sum / corner_count as f64
                } else {
                    sum / count as f64
                };
                (root, position)
            })
            .collect();
        Self {
            root: parent,
            welded,
        }
    }

    /// Where vertex `v` is once its cluster is welded.
    fn position(&self, v: u32, positions: &[DVec3]) -> DVec3 {
        self.welded
            .get(&self.root[v as usize])
            .copied()
            .unwrap_or(positions[v as usize])
    }
}

/// Undo the snaps that turn a triangle over, and return the clusters of the
/// snaps that remain with the vertices whose snaps were undone.
///
/// A triangle is turned over when its normal with the snapped vertices welded
/// is more than 90 degrees from its normal before the snap. Triangles the
/// later steps remove do not count: those with two corners in one cluster,
/// and caps. Of a turned triangle's snapped vertices, the one that moved
/// furthest is put back at its `meshed` position. That can only be judged
/// with the neighbors' snaps known, and undoing one changes the triangles
/// around it, so the check repeats until nothing is turned.
fn retract_tangled_snaps(
    meshed: &[DVec3],
    positions: &mut [DVec3],
    snapped: &mut [Option<SnapKind>],
    indices: &[u32],
    radius: f64,
) -> (Clusters, Vec<u32>) {
    // In practice two or three rounds; the bound only guards against a
    // retraction front that keeps finding new triangles.
    const MAX_ROUNDS: usize = 32;

    let corners = |t: u32| {
        let t = t as usize * 3;
        [indices[t], indices[t + 1], indices[t + 2]]
    };
    let near_snaps: Vec<u32> = (0..(indices.len() / 3) as u32)
        .filter(|&t| corners(t).iter().any(|&v| snapped[v as usize].is_some()))
        .collect();

    let mut retracted: Vec<u32> = Vec::new();
    let mut clusters = Clusters::build(positions, snapped, radius);
    for _ in 0..MAX_ROUNDS {
        let mut undo: Vec<u32> = Vec::new();
        for &t in &near_snaps {
            let tri = corners(t);
            let on_feature = tri.map(|v| snapped[v as usize].is_some());
            if !on_feature.contains(&true) {
                continue;
            }
            let root = tri.map(|v| clusters.root[v as usize]);
            if root[0] == root[1] || root[1] == root[2] || root[2] == root[0] {
                continue;
            }
            let now = tri.map(|v| clusters.position(v, positions));
            if cap_long_edge(now, on_feature, radius).is_some() {
                continue;
            }
            let before = tri.map(|v| meshed[v as usize]);
            let normal = |p: [DVec3; 3]| (p[1] - p[0]).cross(p[2] - p[0]);
            if normal(now).dot(normal(before)) > 0.0 {
                continue;
            }
            let moved = |v: u32| (positions[v as usize] - meshed[v as usize]).length_squared();
            let furthest = tri
                .into_iter()
                .filter(|&v| snapped[v as usize].is_some())
                .reduce(|u, w| if moved(w) > moved(u) { w } else { u })
                .expect("the triangle has a snapped corner");
            undo.push(furthest);
        }
        if undo.is_empty() {
            break;
        }
        undo.sort_unstable();
        undo.dedup();
        for &v in &undo {
            snapped[v as usize] = None;
            positions[v as usize] = meshed[v as usize];
        }
        retracted.extend(undo);
        clusters = Clusters::build(positions, snapped, radius);
    }
    retracted.sort_unstable();
    (clusters, retracted)
}

/// Remove pairs of triangles that use the same three vertices with opposite
/// winding. Every edge of such a pair loses two faces at once, so no edge is
/// opened. Only triangles with an `on_feature` vertex are looked at.
fn drop_cancelling_pairs(indices: &mut Vec<u32>, on_feature: &[bool]) {
    // (sorted corners, whether the winding follows the sorted order, triangle)
    let mut keyed: Vec<([u32; 3], bool, usize)> = indices
        .chunks_exact(3)
        .enumerate()
        .filter(|(_, tri)| tri.iter().any(|&v| on_feature[v as usize]))
        .map(|(t, tri)| {
            let mut key = [tri[0], tri[1], tri[2]];
            key.sort_unstable();
            let first = tri.iter().position(|&v| v == key[0]).expect("own corner");
            (key, tri[(first + 1) % 3] == key[1], t)
        })
        .collect();
    keyed.sort_unstable();

    let mut dead = vec![false; indices.len() / 3];
    let mut any = false;
    let mut start = 0;
    while start < keyed.len() {
        let mut end = start + 1;
        while end < keyed.len() && keyed[end].0 == keyed[start].0 {
            end += 1;
        }
        // Sorted, so one winding comes first: pair them off from both ends.
        let (mut lo, mut hi) = (start, end - 1);
        while lo < hi && keyed[lo].1 != keyed[hi].1 {
            dead[keyed[lo].2] = true;
            dead[keyed[hi].2] = true;
            any = true;
            lo += 1;
            hi -= 1;
        }
        start = end;
    }
    if any {
        let mut kept = Vec::with_capacity(indices.len());
        for (t, tri) in indices.chunks_exact(3).enumerate() {
            if !dead[t] {
                kept.extend_from_slice(tri);
            }
        }
        *indices = kept;
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

/// Whether a triangle is a cap, and if so where its long edge starts: the
/// returned `k` means the edge from corner `k` to corner `k + 1`, with the
/// vertex that lies on it at corner `k + 2`.
///
/// A cap has a snapped vertex within `radius` of the opposite edge and
/// further than `radius` from both ends of it (near an end it is a needle,
/// which an edge flip would only pass on to the next triangle). A triangle
/// with no area at all is a cap whichever vertex lies between the other two.
fn cap_long_edge(p: [DVec3; 3], on_feature: [bool; 3], radius: f64) -> Option<usize> {
    let length_sq = |k: usize| (p[(k + 1) % 3] - p[k]).length_squared();
    let k = (0..3)
        .reduce(|best, k| {
            if length_sq(k) > length_sq(best) {
                k
            } else {
                best
            }
        })
        .expect("three edges");
    let (a, c, b) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
    let edge = c - a;
    let length = edge.length();
    if length == 0.0 {
        return None;
    }
    // How far along the edge `b` sits, and how far off it.
    let along = (b - a).dot(edge) / length;
    let is_cap = if face_unit_normal(a, c, b).is_none() {
        along > 0.0 && along < length
    } else {
        on_feature[(k + 2) % 3]
            && (b - a).cross(edge).length() / length <= radius
            && along > radius
            && length - along > radius
    };
    is_cap.then_some(k)
}

/// Remove caps (see [`cap_long_edge`]) by flipping their long edge. A cap
/// `(a, c, b)` and the face `(c, a, z)` across its long edge become
/// `(c, b, z)` and `(b, a, z)`: the same surface as that face, now meeting at
/// `b`. Returns the number of flips.
///
/// A cap is left alone when the flip is not clean: its long edge is not
/// shared with exactly one other face, `b`-`z` is an edge already, or a new
/// triangle would face the other way from the face it replaces.
///
/// The triangles a flip makes can be caps in turn: a vertex close to several
/// edges of a fan is passed across them one flip at a time, and vertices in a
/// row on one line take a flip each. So that this ends, an edge that a flip
/// has removed is never made again. (Two caps back to back, four vertices on
/// a line, would otherwise trade places for ever; they wait for a flip on
/// another of their edges.)
fn flip_caps(positions: &[DVec3], indices: &mut [u32], on_feature: &[bool], radius: f64) -> usize {
    /// Marks a directed edge that more than one triangle has.
    const SHARED: u32 = u32::MAX;

    // Caps have a snapped vertex (or no area, which only snapping produces),
    // so everything a flip looks at lies within a triangle of one. The edge
    // map covers the triangles two steps out, which leaves room for a flip
    // to bring in a vertex from further away and still find its faces.
    let mut near = on_feature.to_vec();
    for _ in 0..2 {
        let reached: Vec<u32> = indices
            .chunks_exact(3)
            .filter(|tri| tri.iter().any(|&v| near[v as usize]))
            .flatten()
            .copied()
            .collect();
        for v in reached {
            near[v as usize] = true;
        }
    }
    let mut owner: HashMap<(u32, u32), u32> = HashMap::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        if !tri.iter().any(|&v| near[v as usize]) {
            continue;
        }
        for k in 0..3 {
            owner
                .entry((tri[k], tri[(k + 1) % 3]))
                .and_modify(|face| *face = SHARED)
                .or_insert(t as u32);
        }
        queue.push_back(t as u32);
    }

    let corners = |indices: &[u32], t: u32| {
        let t = t as usize * 3;
        [indices[t], indices[t + 1], indices[t + 2]]
    };
    let long_edge = |tri: [u32; 3]| {
        cap_long_edge(
            tri.map(|v| positions[v as usize]),
            tri.map(|v| on_feature[v as usize]),
            radius,
        )
    };
    queue.retain(|&t| long_edge(corners(indices, t)).is_some());

    let undirected = |u: u32, v: u32| (u.min(v), u.max(v));
    let mut removed: HashSet<(u32, u32)> = HashSet::new();
    let mut flipped = 0usize;
    let mut blocked: Vec<u32> = Vec::new();
    loop {
        let mut progressed = false;
        while let Some(t) = queue.pop_front() {
            let tri = corners(indices, t);
            let Some(k) = long_edge(tri) else {
                continue;
            };
            let (a, c, b) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
            let flip = (|| {
                let f = *owner.get(&(c, a))?;
                if f == SHARED || f == t || owner.get(&(a, c)) != Some(&t) {
                    return None;
                }
                let z = corners(indices, f)
                    .into_iter()
                    .find(|&v| v != a && v != c)?;
                if z == b
                    || owner.contains_key(&(b, z))
                    || owner.contains_key(&(z, b))
                    || removed.contains(&undirected(b, z))
                {
                    return None;
                }
                let [pa, pb, pc, pz] = [a, b, c, z].map(|v| positions[v as usize]);
                let replaced = (pa - pc).cross(pz - pc);
                let facing = |n: DVec3| n.dot(replaced) >= 0.0;
                (facing((pb - pc).cross(pz - pc)) && facing((pa - pb).cross(pz - pb)))
                    .then_some((f, z))
            })();
            let Some((f, z)) = flip else {
                blocked.push(t);
                continue;
            };
            indices[t as usize * 3..t as usize * 3 + 3].copy_from_slice(&[c, b, z]);
            indices[f as usize * 3..f as usize * 3 + 3].copy_from_slice(&[b, a, z]);
            owner.remove(&(a, c));
            owner.remove(&(c, a));
            removed.insert(undirected(a, c));
            owner.insert((b, z), t);
            owner.insert((z, b), f);
            // (c, b) stays with `t` and (a, z) with `f`; the other two change
            // hands, unless they were shared to begin with.
            for (edge, from, to) in [((z, c), f, t), ((b, a), t, f)] {
                if let Some(face) = owner.get_mut(&edge)
                    && *face == from
                {
                    *face = to;
                }
            }
            flipped += 1;
            progressed = true;
            queue.extend([t, f]);
        }
        // A flip can clear what blocked another cap.
        if !progressed || blocked.is_empty() {
            return flipped;
        }
        queue.extend(blocked.drain(..));
    }
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
    use crate::sharp_features::adjacency::grid_mesh;

    const EDGE: Option<SnapKind> = Some(SnapKind::Edge);

    /// Cleanup of a mesh whose snapped vertices have not moved.
    fn weld_in_place(
        positions: &[DVec3],
        indices: &[u32],
        snapped: &[Option<SnapKind>],
    ) -> CleanupResult {
        clean_up_snaps(
            positions,
            positions,
            indices,
            snapped,
            1.0,
            &CleanupConfig::default(),
        )
    }

    fn normals(positions: &[DVec3], indices: &[u32]) -> Vec<DVec3> {
        indices
            .chunks_exact(3)
            .map(|t| {
                let p = |k: usize| positions[t[k] as usize];
                (p(1) - p(0)).cross(p(2) - p(0))
            })
            .collect()
    }

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
        let on_feature = [false, false, false, false, true, false];
        let area = |indices: &[u32]| -> f64 {
            normals(&positions, indices).iter().map(|n| n.z / 2.0).sum()
        };
        let before = area(&indices);
        let outline = boundary_edge_count(&indices);
        assert_eq!(flip_caps(&positions, &mut indices, &on_feature, 0.25), 1);
        assert_eq!(indices.len(), 15, "two triangles replaced by two");
        assert!((area(&indices) - before).abs() < 1e-12);
        for (t, n) in normals(&positions, &indices).iter().enumerate() {
            assert!(n.z > 0.1, "triangle {t} has area and faces up");
        }
        assert_eq!(boundary_edge_count(&indices), outline, "same outline");
    }

    #[test]
    fn mesh_without_caps_is_untouched() {
        let (positions, mut indices) = tetrahedron();
        let before = indices.clone();
        assert_eq!(flip_caps(&positions, &mut indices, &[true; 4], 0.25), 0);
        assert_eq!(indices, before);
    }

    /// A rim: a flat annulus meets a cylinder wall along a circle, and both
    /// rows of the band between them have been snapped onto the circle, at
    /// alternating angles. Every band triangle then has its three corners on
    /// the circle: a sliver in the plane of the annulus, facing whichever
    /// way. All of them have to go, however long the chain.
    #[test]
    fn sliver_chain_along_a_curved_crease_is_flipped_away() {
        let n = 240u32;
        let radius = 80.0;
        let ring = |r: f64, half_steps: u32, z: f64| {
            let angle = std::f64::consts::PI * half_steps as f64 / n as f64;
            DVec3::new(r * angle.cos(), r * angle.sin(), z)
        };
        // Per step i: inner (annulus), a and b (on the circle), low (wall).
        let (inner, a, b, low) = (
            |i: u32| i % n * 4,
            |i: u32| i % n * 4 + 1,
            |i: u32| i % n * 4 + 2,
            |i: u32| i % n * 4 + 3,
        );
        let mut positions = Vec::new();
        let mut snapped = Vec::new();
        let mut indices = Vec::new();
        for i in 0..n {
            positions.extend([
                ring(radius - 1.5, 2 * i, 0.0),
                ring(radius, 2 * i, 0.0),
                ring(radius, 2 * i + 1, 0.0),
                ring(radius, 2 * i + 1, -1.5),
            ]);
            snapped.extend([None, EDGE, EDGE, None]);
            indices.extend([
                // Annulus, facing +z.
                [inner(i), a(i), a(i + 1)],
                [inner(i), a(i + 1), inner(i + 1)],
                // The band, flattened onto the circle.
                [a(i), b(i), a(i + 1)],
                [a(i + 1), b(i), b(i + 1)],
                // Wall, facing outward.
                [b(i + 1), b(i), low(i)],
                [b(i + 1), low(i), low(i + 1)],
            ]);
        }
        let indices: Vec<u32> = indices.into_iter().flatten().collect();
        let outline = boundary_edge_count(&indices);

        let result = weld_in_place(&positions, &indices, &snapped);
        assert_eq!(result.welded_vertices, 0);
        assert!(result.retracted.is_empty());
        assert_eq!(result.flipped_caps, 2 * n as usize, "one flip per sliver");
        assert_eq!(result.indices.len(), indices.len());
        assert_eq!(boundary_edge_count(&result.indices), outline);
        for (t, tri) in result.indices.chunks_exact(3).enumerate() {
            let on_circle = tri.iter().filter(|&&v| snapped[v as usize].is_some());
            assert!(on_circle.count() < 3, "triangle {t} is still a sliver");
            let p = |k: usize| positions[tri[k] as usize];
            let normal = (p(1) - p(0)).cross(p(2) - p(0)).normalize();
            let outward = (p(0) + p(1) + p(2)).with_z(0.0).normalize();
            assert!(
                normal.z > 0.99 || normal.dot(outward) > 0.99,
                "triangle {t} faces neither up nor outward: {normal:?}"
            );
        }
    }

    /// Four snapped vertices in a row on one line, A B C Z, with the two
    /// flattened triangles (A, C, B) and (C, A, Z) back to back between the
    /// real faces above and below. Flipping the edge they share only gives
    /// (C, B, Z) and (B, A, Z), flattened as before, and flipping that back
    /// would go on for ever. Each has to be flipped against a real face.
    #[test]
    fn flattened_triangles_back_to_back_are_both_removed() {
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),  // A
            DVec3::new(1.0, 0.0, 0.0),  // B
            DVec3::new(2.0, 0.0, 0.0),  // C
            DVec3::new(3.0, 0.0, 0.0),  // Z
            DVec3::new(1.5, 1.0, 0.0),  // above the line
            DVec3::new(1.5, -1.0, 0.0), // below it
        ];
        let (a, b, c, z, up, down) = (0, 1, 2, 3, 4, 5);
        let mut indices = vec![
            a, b, up, b, c, up, c, z, up, // above: three faces
            z, a, down, // below: one
            a, c, b, c, a, z, // the flattened pair
        ];
        let on_feature = [true, true, true, true, false, false];
        let outline = boundary_edge_count(&indices);
        assert_eq!(flip_caps(&positions, &mut indices, &on_feature, 0.25), 3);
        assert_eq!(boundary_edge_count(&indices), outline);
        for (t, n) in normals(&positions, &indices).iter().enumerate() {
            assert!(n.z > 0.1, "triangle {t} has no area or faces down");
        }
    }

    /// The same pair with nothing round it: no real face to flip against.
    /// The one flip that is possible must not be undone again, and again.
    #[test]
    fn flattened_triangles_with_no_way_out_are_left() {
        let positions: Vec<DVec3> = (0..4).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
        let mut indices = vec![0, 2, 1, 2, 0, 3];
        assert_eq!(flip_caps(&positions, &mut indices, &[true; 4], 0.25), 1);
    }

    /// A cap whose vertex sits further off the long edge than the far corner
    /// of the face across it: flipping would put that face's two halves on
    /// the wrong side of the vertex, turned over. The cap stays.
    #[test]
    fn a_flip_that_would_turn_a_triangle_over_is_not_made() {
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(1.0, 0.2, 0.0), // on the edge 0-1, by the cap rule
            DVec3::new(1.2, 0.1, 0.0), // the face across the edge is thinner
        ];
        let mut indices = vec![0, 2, 1, 0, 1, 3];
        let before = indices.clone();
        let on_feature = [false, false, true, false];
        assert_eq!(flip_caps(&positions, &mut indices, &on_feature, 0.25), 0);
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
        // A tetrahedron with one vertex split into two coincident snapped
        // copies. Vertex 1 is duplicated as vertex 4 offset by a hair;
        // triangles reference both copies, which models the post-snap band
        // exactly.
        let (mut positions, _) = tetrahedron();
        positions.push(positions[1] + DVec3::new(1e-3, 0.0, 0.0));
        // Fan re-wired so both copies appear; the split leaves the mesh with
        // boundary edges (a crack), which welding must close.
        let indices = vec![0, 2, 1, 0, 4, 3, 0, 3, 2, 1, 2, 3, 4, 2, 3];
        assert_ne!(boundary_edge_count(&indices), 0, "precondition: cracked");

        let snapped = vec![None, EDGE, None, None, EDGE];
        let result = weld_in_place(&positions, &indices, &snapped);
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
        let snapped = vec![None, EDGE, EDGE, None];
        let result = weld_in_place(&positions, &indices, &snapped);
        assert_eq!(result.welded_vertices, 1);
        assert_eq!(result.dropped_triangles, 2, "both collapsed tris dropped");
        assert_eq!(result.positions.len(), 3);
        assert!(result.indices.is_empty());
    }

    /// Two snapped vertices on either side of an edge, not joined to each
    /// other, land on the same point: the two triangles between them end up
    /// back to back on the same three vertices. Here they hang off one edge
    /// of a closed tetrahedron, which then has four faces on that edge.
    #[test]
    fn triangles_folded_onto_each_other_cancel() {
        let (mut positions, mut indices) = tetrahedron();
        let fin = DVec3::new(0.5, -0.5, -0.5);
        positions.extend([fin, fin]);
        indices.extend([0, 1, 4, 1, 0, 5]);
        let snapped = vec![None, None, None, None, EDGE, EDGE];

        let result = weld_in_place(&positions, &indices, &snapped);
        assert_eq!(result.welded_vertices, 1);
        assert_eq!(result.dropped_triangles, 2);
        let (_, tetrahedron_only) = tetrahedron();
        assert_eq!(result.indices, tetrahedron_only);
        assert_eq!(boundary_edge_count(&result.indices), 0);
    }

    /// The same two triangles facing the same way are a doubled face, not a
    /// fold, and removing them would open the surface around them.
    #[test]
    fn triangles_doubled_the_same_way_are_kept() {
        let mut indices = vec![0, 1, 2, 0, 1, 2, 1, 2, 0];
        drop_cancelling_pairs(&mut indices, &[true; 3]);
        assert_eq!(indices.len(), 9);
        let mut indices = vec![0, 1, 2, 0, 1, 2, 2, 1, 0];
        drop_cancelling_pairs(&mut indices, &[true; 3]);
        assert_eq!(indices, vec![0, 1, 2], "one pair cancels, one face stays");
    }

    #[test]
    fn unsnapped_vertices_are_never_welded() {
        let positions = vec![DVec3::ZERO, DVec3::new(1e-6, 0.0, 0.0), DVec3::Y, DVec3::Z];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let snapped = vec![None, None, None, None];
        let result = weld_in_place(&positions, &indices, &snapped);
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
        let snapped = vec![EDGE, Some(SnapKind::Corner), EDGE, None, None];
        let result = weld_in_place(&positions, &indices, &snapped);
        assert_eq!(result.welded_vertices, 2);
        let merged = result.positions[result.remap[1] as usize];
        assert!(
            (merged - DVec3::ZERO).length() < 1e-12,
            "cluster should sit at the corner snap, got {merged:?}"
        );
    }

    /// A flat grid whose row 5 has been snapped onto a crease line a little
    /// further out, with one vertex of row 4 moved by `to` as well.
    fn grid_with_a_stray_snap(
        to: DVec3,
    ) -> (Vec<DVec3>, Vec<DVec3>, Vec<u32>, Vec<Option<SnapKind>>) {
        let n = 10usize;
        let (meshed, indices) = grid_mesh(n, n, |i, j| DVec3::new(i as f64, j as f64, 0.0));
        let mut after = meshed.clone();
        let mut snapped = vec![None; meshed.len()];
        for i in 0..n {
            after[5 * n + i].y = 5.4;
            snapped[5 * n + i] = EDGE;
        }
        after[4 * n + 4] = to;
        snapped[4 * n + 4] = EDGE;
        (meshed, after, indices, snapped)
    }

    /// A vertex one row back from the crease is snapped onto it past its
    /// neighbor, which stays: the triangle between them is turned over. That
    /// snap is undone, and only that one.
    #[test]
    fn a_snap_that_overtakes_a_neighbor_is_retracted() {
        let stray = 4 * 10 + 4;
        let (meshed, after, indices, snapped) = grid_with_a_stray_snap(DVec3::new(5.6, 5.4, 0.0));
        // The premise: as snapped, a triangle faces down.
        assert!(normals(&after, &indices).iter().any(|n| n.z < -0.5));

        let result = clean_up_snaps(
            &meshed,
            &after,
            &indices,
            &snapped,
            1.0,
            &CleanupConfig::default(),
        );
        assert_eq!(result.retracted, vec![stray as u32]);
        assert_eq!(
            result.positions[result.remap[stray] as usize],
            meshed[stray]
        );
        for i in 0..10 {
            let on_crease = result.positions[result.remap[5 * 10 + i] as usize];
            assert_eq!(on_crease.y, 5.4, "row 5 stays snapped");
        }
        for (t, n) in normals(&result.positions, &result.indices)
            .iter()
            .enumerate()
        {
            assert!(n.z > 0.1, "triangle {t} faces down or has no area");
        }
    }

    /// A snapped vertex that ends up just across the far edge of one of its
    /// triangles is on that edge, for the purposes of the mesh. The thin
    /// turned-over triangle is a cap: the edge is flipped and the snap stays.
    #[test]
    fn a_snap_just_across_an_edge_is_kept_and_the_edge_flipped() {
        let stray = 4 * 10 + 4;
        // With row 5 snapped, the far edge of triangle ((4,4), (5,4), (4,5))
        // runs from (5, 4) to (4, 5.4). This is 0.07 beyond its middle.
        let to = DVec3::new(4.557, 4.741, 0.0);
        let (meshed, after, indices, snapped) = grid_with_a_stray_snap(to);
        assert!(normals(&after, &indices).iter().any(|n| n.z < 0.0));

        let result = clean_up_snaps(
            &meshed,
            &after,
            &indices,
            &snapped,
            1.0,
            &CleanupConfig::default(),
        );
        assert!(result.retracted.is_empty());
        assert_eq!(result.flipped_caps, 1);
        assert_eq!(result.positions[result.remap[stray] as usize], to);
        assert_eq!(result.indices.len(), indices.len());
        assert_eq!(
            boundary_edge_count(&result.indices),
            boundary_edge_count(&indices)
        );
        for (t, n) in normals(&result.positions, &result.indices)
            .iter()
            .enumerate()
        {
            assert!(n.z > 0.1, "triangle {t} faces down or has no area");
        }
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
