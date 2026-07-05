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
    /// Vertex normals: carried through for unsplit vertices, re-accumulated
    /// from referencing triangles for crease copies. Unnormalized.
    pub normals: Vec<DVec3>,
    pub indices: Vec<u32>,
    /// Extra vertex copies created (the first region reuses the base slot).
    pub split_vertices: usize,
}

/// Split crease vertices so each adjacent region gets its own copy.
///
/// `labels` are post-weld region labels (`None` for feature-zone vertices);
/// `is_crease` marks vertices the snap stage placed on a feature. `normals`
/// are the carried per-vertex normals; crease copies get theirs re-derived
/// from the triangles that reference them.
pub fn split_crease_vertices(
    positions: &[DVec3],
    normals: &[DVec3],
    indices: &[u32],
    labels: &[Option<u32>],
    is_crease: &[bool],
) -> CreaseSplitResult {
    // Assign each triangle to a region by its claimed vertices. Regions never
    // touch directly (an unclaimed band separates them), so claimed labels
    // within one triangle agree in practice; majority is a defensive choice.
    let tri_count = indices.len() / 3;
    let mut tri_region: Vec<Option<u32>> = Vec::with_capacity(tri_count);
    for tri in indices.chunks_exact(3) {
        let mut counts: Vec<(u32, usize)> = Vec::new();
        for &v in tri {
            if let Some(label) = labels[v as usize] {
                match counts.iter_mut().find(|(l, _)| *l == label) {
                    Some((_, c)) => *c += 1,
                    None => counts.push((label, 1)),
                }
            }
        }
        // Majority, ties broken by smaller region id for determinism.
        counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        tri_region.push(counts.first().map(|&(l, _)| l));
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
                normals_out.push(DVec3::ZERO);
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

    // Re-derive crease copy normals from exactly the triangles that reference
    // them (including reused base slots), leaving carried normals elsewhere.
    let mut is_copy_slot = vec![false; positions_out.len()];
    for v in 0..positions.len() {
        for &(_, idx) in &copy_index[v] {
            is_copy_slot[idx as usize] = true;
            normals_out[idx as usize] = DVec3::ZERO;
        }
    }
    for tri in indices_out.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if !(is_copy_slot[a] || is_copy_slot[b] || is_copy_slot[c]) {
            continue;
        }
        let face = (positions_out[b] - positions_out[a]).cross(positions_out[c] - positions_out[a]);
        for &v in &[a, b, c] {
            if is_copy_slot[v] {
                normals_out[v] += face;
            }
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
        let result = split_crease_vertices(&positions, &carried, &indices, &labels, &is_crease);

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

        // Unsplit vertices keep their carried normals.
        assert_eq!(result.normals[0], DVec3::ONE);
        assert_eq!(result.normals[4], DVec3::ONE);
    }

    #[test]
    fn mesh_without_creases_is_unchanged() {
        let (positions, indices, labels, _) = crease_mesh();
        let carried = vec![DVec3::ONE; positions.len()];
        let no_crease = vec![false; positions.len()];
        let result = split_crease_vertices(&positions, &carried, &indices, &labels, &no_crease);
        assert_eq!(result.split_vertices, 0);
        assert_eq!(result.indices, indices);
        assert_eq!(result.positions.len(), positions.len());
        assert!(result.normals.iter().all(|&n| n == DVec3::ONE));
    }
}
