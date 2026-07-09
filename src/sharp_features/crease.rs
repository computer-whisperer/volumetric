//! Crease vertex duplication: one vertex copy per adjacent smooth region, so
//! each side of a sharp feature shades with its own normal.
//!
//! Smooth (per-vertex-normal) rendering interpolates normals across
//! triangles; a single crease vertex whose normal is the accumulated bisector
//! smears the highlight across the feature. Splitting the vertex per region
//! gives each side a face-true normal while positions stay identical, so the
//! surface remains geometrically sealed — the split is topological only,
//! which is the standard representation for crisp creases in per-vertex
//! normal mesh formats.
//!
//! Only vertices the snap stage actually moved are split. Gate-rejected
//! feature-zone vertices (pathological geometry) keep their blended normal
//! and connectivity: no evidence of a real feature means no topology surgery.

use glam::DVec3;

pub struct CreaseSplitResult {
    pub positions: Vec<DVec3>,
    /// Vertex normals: re-accumulated from the final triangles for crease
    /// copies and for every vertex sharing a triangle with one (their fans
    /// changed when the snap stage moved the feature vertices); carried
    /// through unchanged elsewhere. Unnormalized.
    pub normals: Vec<DVec3>,
    pub indices: Vec<u32>,
    /// Extra vertex copies created (the first region reuses the base slot).
    pub split_vertices: usize,
}

/// Split crease vertices so each adjacent region gets its own copy.
///
/// `labels` are post-weld region labels (`None` for feature-zone vertices);
/// `is_crease` marks vertices the snap stage placed on a feature. `normals`
/// are the carried per-vertex normals; crease copies (and their unsplit
/// neighbors, whose fans the snap changed) get theirs re-derived from the
/// triangles that reference them. `cell` is the finest cell size, used to
/// tell real triangles from folded slivers during that re-derivation.
pub fn split_crease_vertices(
    positions: &[DVec3],
    normals: &[DVec3],
    indices: &[u32],
    labels: &[Option<u32>],
    is_crease: &[bool],
    cell: f64,
) -> CreaseSplitResult {
    // Without crease vertices nothing splits, no triangle is rewritten and no
    // normal is re-derived: the pass is an identity. Worth an early exit:
    // smooth models otherwise pay the full triangle-region assignment over
    // millions of triangles for nothing.
    if !is_crease.iter().any(|&c| c) {
        return CreaseSplitResult {
            positions: positions.to_vec(),
            normals: normals.to_vec(),
            indices: indices.to_vec(),
            split_vertices: 0,
        };
    }

    // Assign each triangle to a region by its claimed vertices. When labels
    // conflict within one triangle (regions in direct contact, which happens
    // when grid alignment leaves no unclaimed band), the tie is broken
    // geometrically: the region whose corners' carried normals best agree
    // with the triangle's face normal wins. Label majority alone mis-assigns
    // half the cross-feature triangles in that regime. Triangles are
    // independent here, so this runs in parallel on native builds; the three
    // corners bound the candidate set, so a fixed-size buffer avoids one
    // allocation per triangle.
    let tri_count = indices.len() / 3;
    let mut tri_region: Vec<Option<u32>> = crate::parallel_iter::map_range(0..tri_count, |t| {
        let tri = &indices[t * 3..t * 3 + 3];
        let mut candidates = [(0u32, DVec3::ZERO); 3];
        let mut candidate_count = 0usize;
        for &v in tri {
            if let Some(label) = labels[v as usize] {
                match candidates[..candidate_count]
                    .iter_mut()
                    .find(|(l, _)| *l == label)
                {
                    Some((_, n)) => *n += normals[v as usize],
                    None => {
                        candidates[candidate_count] = (label, normals[v as usize]);
                        candidate_count += 1;
                    }
                }
            }
        }
        match candidate_count {
            0 => None,
            1 => Some(candidates[0].0),
            _ => {
                let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                let face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
                candidates[..candidate_count]
                    .iter()
                    .max_by(|(_, na), (_, nb)| {
                        let da = face.dot(na.normalize_or_zero());
                        let db = face.dot(nb.normalize_or_zero());
                        da.total_cmp(&db)
                    })
                    .map(|&(l, _)| l)
            }
        }
    });

    // Region-less triangles that touch a crease vertex would otherwise keep
    // referencing base slots — aliasing their face into whichever region
    // reuses each slot, and starving the vertex of a copy for the face the
    // triangle actually lies on. The canonical case is a corner cap whose
    // three corners are all unclaimed feature-band vertices (a cube corner
    // welds to corner + two edge vertices; one face's cap triangle survives
    // with no claimed corner). Resolve each against the assigned triangles
    // around its corners: the region whose nearby face normals best agree
    // with its own wins. Sweeping to fixpoint lets multi-triangle pockets
    // resolve from their rim inward.
    //
    // The agreement gate is strict: a triangle genuinely lying on a region's
    // face scores near 1 against that region's neighboring face normals,
    // while the crumpled fold triangles that corner pockets also contain
    // (see the folded-sliver debt) score well below it in every direction —
    // assigning those would mint copies whose re-derived normals are pure
    // fold noise, worse than the base-slot blend they have today. Folds,
    // zero-area slivers, and triangles with no assigned neighbor anywhere
    // stay region-less and keep the base-slot behavior.
    const MIN_RESOLVE_DOT: f64 = 0.9;
    // Vertex -> incident triangles in compressed (CSR) form: two counting
    // passes instead of millions of per-vertex Vec allocations.
    let mut vt_starts = vec![0u32; positions.len() + 1];
    for &v in indices {
        vt_starts[v as usize + 1] += 1;
    }
    for i in 1..vt_starts.len() {
        vt_starts[i] += vt_starts[i - 1];
    }
    let mut vt_cursor = vt_starts.clone();
    let mut vt_tris = vec![0u32; indices.len()];
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        for &v in tri {
            vt_tris[vt_cursor[v as usize] as usize] = t as u32;
            vt_cursor[v as usize] += 1;
        }
    }
    let vertex_tris = |v: usize| -> &[u32] {
        &vt_tris[vt_starts[v] as usize..vt_starts[v + 1] as usize]
    };
    loop {
        let mut changed = false;
        for (t, tri) in indices.chunks_exact(3).enumerate() {
            if tri_region[t].is_some() || !tri.iter().any(|&v| is_crease[v as usize]) {
                continue;
            }
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let Some(face) = (positions[b] - positions[a])
                .cross(positions[c] - positions[a])
                .try_normalize()
            else {
                continue;
            };
            let mut neighbors: Vec<u32> = tri
                .iter()
                .flat_map(|&v| vertex_tris(v as usize).iter().copied())
                .collect();
            neighbors.sort_unstable();
            neighbors.dedup();
            let mut candidates: Vec<(u32, DVec3)> = Vec::new();
            for &u in &neighbors {
                let Some(label) = tri_region[u as usize] else {
                    continue;
                };
                let utri = &indices[u as usize * 3..u as usize * 3 + 3];
                let (ua, ub, uc) = (utri[0] as usize, utri[1] as usize, utri[2] as usize);
                let un = (positions[ub] - positions[ua]).cross(positions[uc] - positions[ua]);
                match candidates.iter_mut().find(|(l, _)| *l == label) {
                    Some((_, n)) => *n += un,
                    None => candidates.push((label, un)),
                }
            }
            let best = candidates
                .iter()
                .map(|&(l, n)| (l, face.dot(n.normalize_or_zero())))
                .max_by(|(_, da), (_, db)| da.total_cmp(db));
            if let Some((label, dot)) = best {
                if dot > MIN_RESOLVE_DOT {
                    tri_region[t] = Some(label);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // First pass: which regions touch each crease vertex (sorted for
    // deterministic copy order; the first region reuses the base slot).
    let mut vertex_regions: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        if let Some(region) = tri_region[t] {
            for &v in tri {
                if is_crease[v as usize] && !vertex_regions[v as usize].contains(&region) {
                    vertex_regions[v as usize].push(region);
                }
            }
        }
    }
    for regions in &mut vertex_regions {
        regions.sort_unstable();
    }

    // Allocate copies: (vertex, region) -> index. The first region keeps the
    // base slot so no vertex is orphaned; the base slot also keeps serving
    // any unassigned triangles.
    let mut positions_out = positions.to_vec();
    let mut normals_out = normals.to_vec();
    let mut copy_index: Vec<Vec<(u32, u32)>> = vec![Vec::new(); positions.len()];
    let mut split_vertices = 0usize;
    for v in 0..positions.len() {
        for (i, &region) in vertex_regions[v].iter().enumerate() {
            let index = if i == 0 {
                v as u32
            } else {
                positions_out.push(positions[v]);
                // Seeded with the base carried normal so a copy that ends up
                // referenced only by slivers has a sane fallback below.
                normals_out.push(normals[v]);
                split_vertices += 1;
                (positions_out.len() - 1) as u32
            };
            copy_index[v].push((region, index));
        }
    }

    // Second pass: rewrite triangle corners onto their region's copy.
    let mut indices_out = Vec::with_capacity(indices.len());
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        for &v in tri {
            let index = match tri_region[t] {
                Some(region) if is_crease[v as usize] => copy_index[v as usize]
                    .iter()
                    .find(|(r, _)| *r == region)
                    .map(|&(_, idx)| idx)
                    .unwrap_or(v),
                _ => v,
            };
            indices_out.push(index);
        }
    }

    // Re-derive normals from the final triangles for every vertex whose fan
    // the snap stage changed: the crease copies themselves (including reused
    // base slots and snapped-but-unsplit vertices), plus their unsplit
    // neighbors. The neighbors' carried normals were accumulated over the
    // *pre-snap* fan — moving the rim vertices tilted those triangles, so a
    // shoulder vertex half a cell from a feature can carry a normal 15-20
    // degrees off the face it provably sits on. Vertices with no crease
    // contact keep their carried normals (which may be probe-refined).
    let mut rederive = vec![false; positions_out.len()];
    let mut is_crease_slot = vec![false; positions_out.len()];
    for v in 0..positions.len() {
        if is_crease[v] {
            is_crease_slot[v] = true;
        }
        for &(_, idx) in &copy_index[v] {
            is_crease_slot[idx as usize] = true;
        }
    }
    for tri in indices_out.chunks_exact(3) {
        if tri.iter().any(|&v| is_crease_slot[v as usize]) {
            for &v in tri {
                rederive[v as usize] = true;
            }
        }
    }
    let carried = normals_out.clone();
    for (v, flag) in rederive.iter().enumerate() {
        if *flag {
            normals_out[v] = DVec3::ZERO;
        }
    }
    for tri in indices_out.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if !(rederive[a] || rederive[b] || rederive[c]) {
            continue;
        }
        let face = (positions_out[b] - positions_out[a]).cross(positions_out[c] - positions_out[a]);
        for &v in &[a, b, c] {
            if rederive[v] {
                normals_out[v] += face;
            }
        }
    }
    // A slot referenced only by folded slivers accumulates a near-zero
    // vector whose direction is fold noise (up to 90 degrees off any true
    // face). Area weighting already drowns slivers wherever a real triangle
    // contributes, so only these all-sliver slots need repair: shade them
    // like the well-accumulated vertices they share those slivers with
    // (same-region crease neighbors — the slot itself is invisible, but its
    // normal shouldn't be noise). Carried normals are the last resort.
    // (|cross| = 2x area; the threshold is a tiny fraction of one
    // finest-cell triangle.)
    let min_accum = 0.02 * cell * cell;
    let accumulated = normals_out.clone();
    let starved = |v: usize| -> bool {
        rederive[v] && accumulated[v].length_squared() < min_accum * min_accum
    };
    for tri in indices_out.chunks_exact(3) {
        if !tri.iter().any(|&v| starved(v as usize)) {
            continue;
        }
        for &v in tri {
            if !starved(v as usize) {
                continue;
            }
            for &u in tri {
                if u != v && !starved(u as usize) {
                    let n = accumulated[u as usize];
                    normals_out[v as usize] += n.normalize_or_zero() * (cell * cell);
                }
            }
        }
    }
    for (v, flag) in rederive.iter().enumerate() {
        if *flag && normals_out[v].length_squared() < min_accum * min_accum {
            normals_out[v] = carried[v];
        }
    }

    CreaseSplitResult {
        positions: positions_out,
        normals: normals_out,
        indices: indices_out,
        split_vertices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharp_features::fit::unsigned_angle_degrees;

    /// Two quads meeting at a 90-degree crease along the Y axis:
    /// left face in the z=0 plane, right face in the x=1 plane.
    /// Vertices: a0,a1 (left), c0,c1 (crease), b0,b1 (right).
    fn crease_mesh() -> (Vec<DVec3>, Vec<u32>, Vec<Option<u32>>, Vec<bool>) {
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0), // a0
            DVec3::new(0.0, 1.0, 0.0), // a1
            DVec3::new(1.0, 0.0, 0.0), // c0
            DVec3::new(1.0, 1.0, 0.0), // c1
            DVec3::new(1.0, 0.0, 1.0), // b0
            DVec3::new(1.0, 1.0, 1.0), // b1
        ];
        let indices = vec![
            0, 2, 1, // left
            1, 2, 3, // left
            2, 4, 3, // right
            3, 4, 5, // right
        ];
        let labels = vec![Some(0), Some(0), None, None, Some(1), Some(1)];
        let is_crease = vec![false, false, true, true, false, false];
        (positions, indices, labels, is_crease)
    }

    #[test]
    fn crease_vertices_split_with_per_region_normals() {
        let (positions, indices, labels, is_crease) = crease_mesh();
        let carried = vec![DVec3::ONE; positions.len()];
        let result =
            split_crease_vertices(&positions, &carried, &indices, &labels, &is_crease, 1.0);

        // Two crease vertices, two regions each: one extra copy per vertex.
        assert_eq!(result.split_vertices, 2);
        assert_eq!(result.positions.len(), 8);

        // Left triangles' crease corners shade with the left face normal
        // (-Z for this winding... verify against the actual face plane), and
        // right triangles' with the right face normal; the two are 90 apart.
        let left_tri = &result.indices[0..3];
        let right_tri = &result.indices[6..9];
        let left_crease_normal = result.normals[left_tri[1] as usize];
        let right_crease_normal = result.normals[right_tri[0] as usize];
        assert!(
            unsigned_angle_degrees(left_crease_normal.normalize(), DVec3::Z) < 1e-9,
            "left crease copy should carry the z-plane normal, got {left_crease_normal:?}"
        );
        assert!(
            unsigned_angle_degrees(right_crease_normal.normalize(), DVec3::X) < 1e-9,
            "right crease copy should carry the x-plane normal, got {right_crease_normal:?}"
        );

        // Copies coincide in position with their base vertex: the split is
        // topological, not geometric.
        assert_eq!(result.positions[left_tri[1] as usize], positions[2]);
        assert_eq!(result.positions[right_tri[0] as usize], positions[2]);
        assert_ne!(left_tri[1], right_tri[0], "sides use distinct copies");

        // Unsplit vertices sharing a triangle with a crease vertex are
        // re-accumulated from the final fan: face-true for planar sides.
        assert!(
            unsigned_angle_degrees(result.normals[0].normalize(), DVec3::Z) < 1e-9,
            "left-face neighbor should re-accumulate the z-plane normal"
        );
        assert!(
            unsigned_angle_degrees(result.normals[4].normalize(), DVec3::X) < 1e-9,
            "right-face neighbor should re-accumulate the x-plane normal"
        );
    }

    /// The post-weld fan at a cube corner: three mutually perpendicular
    /// faces (+X, +Y, +Z) meeting at corner K, with the unclaimed feature
    /// band welded down to K plus one vertex per edge (Ex, Ey, Ez — all
    /// crease, all label-less). Each face keeps one claimed interior vertex
    /// (Dx, Dy, Dz). The quad diagonalization leaves the +X face with a cap
    /// triangle (K, Ey, Ex) whose corners are all unclaimed — the triangle
    /// that used to stay region-less, alias into the base slot, and shade
    /// the corner with a cross-face blended normal (the "dog ears" bug).
    #[test]
    fn corner_cap_triangle_resolves_to_its_face() {
        let k = DVec3::new(1.0, 1.0, 1.0);
        let positions = vec![
            k,                         // 0: K corner (crease)
            DVec3::new(1.0, 1.0, 0.9), // 1: Ex, edge x=1,y=1 (crease)
            DVec3::new(1.0, 0.9, 1.0), // 2: Ey, edge x=1,z=1 (crease)
            DVec3::new(0.9, 1.0, 1.0), // 3: Ez, edge y=1,z=1 (crease)
            DVec3::new(1.0, 0.9, 0.9), // 4: Dx, +X face interior
            DVec3::new(0.9, 1.0, 0.9), // 5: Dy, +Y face interior
            DVec3::new(0.9, 0.9, 1.0), // 6: Dz, +Z face interior
        ];
        let indices = vec![
            4, 1, 2, // +X face, claimed corner Dx
            0, 2, 1, // +X cap: all corners unclaimed crease vertices
            5, 0, 1, // +Y face
            5, 3, 0, // +Y face
            6, 0, 3, // +Z face
            6, 2, 0, // +Z face
        ];
        let labels = vec![None, None, None, None, Some(0), Some(1), Some(2)];
        let is_crease = vec![true, true, true, true, false, false, false];
        // Carried normals: blended for the feature band, face-true inside.
        let carried = vec![
            DVec3::ONE.normalize(),
            DVec3::new(1.0, 1.0, 0.0).normalize(),
            DVec3::new(1.0, 0.0, 1.0).normalize(),
            DVec3::new(0.0, 1.0, 1.0).normalize(),
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
        ];
        let result =
            split_crease_vertices(&positions, &carried, &indices, &labels, &is_crease, 0.1);

        // The corner must shade all three faces: three K copies, one per
        // region, each carrying that face's axis normal. Without the
        // region-less resolution K gets two copies, one of them a blend.
        let k_slots: Vec<usize> = (0..result.positions.len())
            .filter(|&v| result.positions[v] == k)
            .collect();
        assert_eq!(k_slots.len(), 3, "corner needs one copy per face");
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            assert!(
                k_slots.iter().any(|&v| {
                    unsigned_angle_degrees(result.normals[v].normalize(), axis) < 1e-6
                }),
                "no corner copy shades the {axis:?} face: {:?}",
                k_slots
                    .iter()
                    .map(|&v| result.normals[v].normalize())
                    .collect::<Vec<_>>()
            );
        }

        // The cap's corners reference the same copies as the face triangle's
        // crease corners: the +X face shades seamlessly.
        let face_tri = &result.indices[0..3];
        let cap_tri = &result.indices[3..6];
        assert_eq!(face_tri[1], cap_tri[2], "Ex shared between cap and face");
        assert_eq!(face_tri[2], cap_tri[1], "Ey shared between cap and face");
    }

    /// A crumpled fold triangle across the crease (all corners unclaimed,
    /// face normal ~45 degrees off both adjacent faces — the folded-sliver
    /// family that survives with real area at corner pockets) must NOT be
    /// resolved to either region: assigning it would mint a copy whose
    /// re-derived normal is fold noise. It stays on base slots.
    #[test]
    fn fold_triangle_stays_region_less() {
        let (mut positions, mut indices, mut labels, mut is_crease) = crease_mesh();
        positions.push(DVec3::new(1.05, 0.5, 0.05)); // 6: fold apex off both planes
        labels.push(None);
        is_crease.push(true);
        indices.extend_from_slice(&[2, 6, 3]); // fold: c0, apex, c1 — all unclaimed
        let carried = vec![DVec3::ONE; positions.len()];
        let result =
            split_crease_vertices(&positions, &carried, &indices, &labels, &is_crease, 1.0);

        // Same copies as without the fold: no region resolved for it.
        assert_eq!(result.split_vertices, 2);
        // The fold triangle still references the base slots.
        let fold = &result.indices[12..15];
        assert_eq!(fold, &[2, 6, 3], "fold must keep base slots, got {fold:?}");
    }

    #[test]
    fn mesh_without_creases_is_unchanged() {
        let (positions, indices, labels, _) = crease_mesh();
        let carried = vec![DVec3::ONE; positions.len()];
        let no_crease = vec![false; positions.len()];
        let result =
            split_crease_vertices(&positions, &carried, &indices, &labels, &no_crease, 1.0);
        assert_eq!(result.split_vertices, 0);
        assert_eq!(result.indices, indices);
        assert_eq!(result.positions.len(), positions.len());
        assert!(result.normals.iter().all(|&n| n == DVec3::ONE));
    }
}
