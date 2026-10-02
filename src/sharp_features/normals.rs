//! Shading normals split at feature edges: the last step of the pipeline.
//!
//! Smooth (per-vertex-normal) rendering interpolates normals across
//! triangles, so a crease vertex carrying one blended normal smears the
//! highlight across the feature. Here the faces around each feature vertex
//! are divided into fans by the feature edges meeting there, and each fan
//! gets its own vertex copy with a normal taken from its own faces only.
//! Positions stay identical, so the surface remains geometrically sealed:
//! the split is topological, the standard representation of a crisp crease
//! in per-vertex-normal mesh formats.
//!
//! This runs after decimation, on the triangles that are actually emitted.
//! Splitting earlier (as the pipeline once did) hands the decimator a mesh
//! whose crease sides are separate borders; it collapses them independently
//! and the surface tears.

use glam::DVec3;

use crate::adaptive_surface_nets_2::IndexedMesh2;
use crate::sharp_features::feature_edges::FeatureEdges;

/// Split the vertices on feature edges into one copy per fan and give every
/// copy its fan's normal. Returns the number of copies added.
///
/// Normals are also re-derived for the unsplit vertices sharing a triangle
/// with a feature vertex: theirs were accumulated before the snap stage
/// moved their neighbours onto the feature. All other normals are kept.
///
/// A fan's normal weights each face by its area, as every other normal in
/// the pipeline is. Where the snap stage could not resolve a feature, a
/// vertex on the flat side of it shares its fan with sub-cell tilted facets
/// of the sawtooth; after decimation it is also a corner of triangles a
/// hundred cells long. Weighting by corner angle lets a facet tilt that
/// vertex's normal and the tilt shows as a streak along each long triangle
/// (measured on the keychain sleeve's keyring hole). By area the facet is
/// outvoted and takes the flat side's normal instead, which nobody can see.
pub fn split_normals_at_features(mesh: &mut IndexedMesh2, features: &FeatureEdges) -> usize {
    if features.is_empty() {
        return 0;
    }
    let vertex_count = mesh.vertices.len();
    let degrees = features.degrees(vertex_count);
    let position = |vertices: &[(f32, f32, f32)], v: u32| -> DVec3 {
        let (x, y, z) = vertices[v as usize];
        DVec3::new(x as f64, y as f64, z as f64)
    };

    // Vertex -> incident triangles (CSR).
    let mut starts = vec![0u32; vertex_count + 1];
    for &v in &mesh.indices {
        starts[v as usize + 1] += 1;
    }
    for i in 1..starts.len() {
        starts[i] += starts[i - 1];
    }
    let mut cursor = starts.clone();
    let mut incident = vec![0u32; mesh.indices.len()];
    for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
        for &v in tri {
            incident[cursor[v as usize] as usize] = t as u32;
            cursor[v as usize] += 1;
        }
    }

    // A face's contribution to a normal: its area-weighted normal.
    let face_normal = |vertices: &[(f32, f32, f32)], tri: &[u32]| -> DVec3 {
        let p = position(vertices, tri[0]);
        (position(vertices, tri[1]) - p).cross(position(vertices, tri[2]) - p)
    };

    let mut rederive = vec![false; vertex_count];
    for tri in mesh.indices.chunks_exact(3) {
        if tri.iter().any(|&v| degrees[v as usize] > 0) {
            for &v in tri {
                rederive[v as usize] = true;
            }
        }
    }

    let original_indices = mesh.indices.clone();
    let mut split_vertices = 0usize;
    // Scratch, reused per vertex: fan label per incident face (union-find).
    let mut parent: Vec<usize> = Vec::new();
    for v in 0..vertex_count as u32 {
        if !rederive[v as usize] {
            continue;
        }
        let faces = &incident[starts[v as usize] as usize..starts[v as usize + 1] as usize];
        let set_normal = |mesh: &mut IndexedMesh2, slot: u32, sum: DVec3| {
            // A fan of area-less faces has no direction of its own; it keeps
            // the normal the vertex came in with.
            if let Some(n) = sum.try_normalize() {
                mesh.normals[slot as usize] = (n.x as f32, n.y as f32, n.z as f32);
            }
        };

        if degrees[v as usize] == 0 {
            let sum: DVec3 = faces
                .iter()
                .map(|&t| {
                    face_normal(
                        &mesh.vertices,
                        &original_indices[t as usize * 3..t as usize * 3 + 3],
                    )
                })
                .sum();
            set_normal(mesh, v, sum);
            continue;
        }

        // Join the faces that share a plain edge at this vertex: one that is
        // not a feature edge and has exactly two faces on it.
        parent.clear();
        parent.extend(0..faces.len());
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let others = |t: u32| -> [u32; 2] {
            let tri = &original_indices[t as usize * 3..t as usize * 3 + 3];
            let k = tri
                .iter()
                .position(|&u| u == v)
                .expect("corner of its face");
            [tri[(k + 1) % 3], tri[(k + 2) % 3]]
        };
        for i in 0..faces.len() {
            for x in others(faces[i]) {
                if features.contains(v, x) {
                    continue;
                }
                let mut partner = None;
                let mut sharing = 0;
                for (j, &u) in faces.iter().enumerate() {
                    if others(u).contains(&x) {
                        sharing += 1;
                        if j != i {
                            partner = Some(j);
                        }
                    }
                }
                if let (2, Some(j)) = (sharing, partner) {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    // Smaller root wins, so fans are numbered by their first
                    // face and the first fan keeps the original vertex.
                    parent[a.max(b)] = a.min(b);
                }
            }
        }

        let mut fan_slot: Vec<(usize, u32, DVec3)> = Vec::new(); // (root, vertex slot, normal sum)
        for i in 0..faces.len() {
            let root = find(&mut parent, i);
            let t = faces[i] as usize;
            let contribution = face_normal(&mesh.vertices, &original_indices[t * 3..t * 3 + 3]);
            let fan = match fan_slot.iter().position(|&(r, _, _)| r == root) {
                Some(fan) => fan,
                None => {
                    let slot = if fan_slot.is_empty() {
                        v
                    } else {
                        mesh.vertices.push(mesh.vertices[v as usize]);
                        mesh.normals.push(mesh.normals[v as usize]);
                        split_vertices += 1;
                        mesh.vertices.len() as u32 - 1
                    };
                    fan_slot.push((root, slot, DVec3::ZERO));
                    fan_slot.len() - 1
                }
            };
            fan_slot[fan].2 += contribution;
            let slot = fan_slot[fan].1;
            for corner in &mut mesh.indices[t * 3..t * 3 + 3] {
                if *corner == v {
                    *corner = slot;
                }
            }
        }
        for (_, slot, sum) in fan_slot {
            set_normal(mesh, slot, sum);
        }
    }
    split_vertices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharp_features::feature_edges::{FeatureEdgeConfig, edge_key};

    fn mesh_of(positions: &[DVec3], indices: &[u32]) -> IndexedMesh2 {
        IndexedMesh2 {
            vertices: positions
                .iter()
                .map(|p| (p.x as f32, p.y as f32, p.z as f32))
                .collect(),
            // A blended normal every vertex should lose where it matters.
            normals: vec![(0.577, 0.577, 0.577); positions.len()],
            indices: indices.to_vec(),
        }
    }

    fn normal(mesh: &IndexedMesh2, v: u32) -> DVec3 {
        let (x, y, z) = mesh.normals[v as usize];
        DVec3::new(x as f64, y as f64, z as f64)
    }

    fn face_normal(mesh: &IndexedMesh2, t: usize) -> DVec3 {
        let p = |v: u32| {
            let (x, y, z) = mesh.vertices[v as usize];
            DVec3::new(x as f64, y as f64, z as f64)
        };
        let tri = &mesh.indices[t * 3..t * 3 + 3];
        (p(tri[1]) - p(tri[0]))
            .cross(p(tri[2]) - p(tri[0]))
            .normalize()
    }

    /// Every corner of a cube gets three copies, and every triangle corner
    /// shades with exactly its own face's normal.
    #[test]
    fn cube_corners_split_three_ways_and_shade_flat() {
        let (positions, indices) = crate::sharp_features::feature_edges::unit_cube();
        let features = FeatureEdges::classify(&positions, &indices, &FeatureEdgeConfig::default());
        let mut mesh = mesh_of(&positions, &indices);
        let copies = split_normals_at_features(&mut mesh, &features);
        assert_eq!(copies, 16, "8 corners, 2 extra copies each");
        assert_eq!(mesh.vertices.len(), 24);
        for t in 0..mesh.indices.len() / 3 {
            let face = face_normal(&mesh, t);
            for &v in &mesh.indices[t * 3..t * 3 + 3] {
                assert!(
                    normal(&mesh, v).dot(face) > 1.0 - 1e-6,
                    "triangle {t} corner {v}: {:?} vs face {face:?}",
                    normal(&mesh, v)
                );
            }
        }
        // Positions are copies: merged by position the cube is still closed.
        let mut unique: Vec<(u32, u32, u32)> = mesh
            .vertices
            .iter()
            .map(|&(x, y, z)| (x.to_bits(), y.to_bits(), z.to_bits()))
            .collect();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 8);
    }

    /// A crease along one edge of a flat fan, ending at the vertex (one
    /// feature edge): there is only one fan, so nothing is split.
    #[test]
    fn a_single_feature_edge_at_a_vertex_does_not_split_it() {
        // A flat hexagon fan around vertex 0.
        let mut positions = vec![DVec3::ZERO];
        for i in 0..6 {
            let a = std::f64::consts::TAU * i as f64 / 6.0;
            positions.push(DVec3::new(a.cos(), a.sin(), 0.0));
        }
        let indices: Vec<u32> = (0..6u32)
            .flat_map(|i| [0, 1 + i, 1 + (i + 1) % 6])
            .collect();
        let features = FeatureEdges::from_keys([edge_key(0, 1)]);
        let mut mesh = mesh_of(&positions, &indices);
        // Only the rim vertex at the edge's other end splits: there the
        // feature edge and the open border leave its two faces unconnected.
        assert_eq!(split_normals_at_features(&mut mesh, &features), 1);
        assert!(
            mesh.indices.chunks_exact(3).all(|tri| tri[0] == 0),
            "the centre keeps one slot"
        );
        assert!(normal(&mesh, 0).dot(DVec3::Z) > 1.0 - 1e-6);
        // Its neighbours' normals were re-derived too.
        assert!(normal(&mesh, 3).dot(DVec3::Z) > 1.0 - 1e-6);
    }

    /// A vertex on a crease (two feature edges) splits into one copy per
    /// side, each shading with its own side's normal.
    #[test]
    fn crease_vertex_splits_into_two_fans() {
        // Vertex 0 at the origin on a 90-degree crease along x: two faces in
        // the z = 0 plane, two in the y = 0 plane.
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 100.0, 0.0),
            DVec3::new(0.0, 0.0, -1.0),
        ];
        let indices = vec![0, 1, 3, 0, 3, 2, 0, 4, 1, 0, 2, 4];
        let features = FeatureEdges::from_keys([edge_key(0, 1), edge_key(0, 2)]);
        let mut mesh = mesh_of(&positions, &indices);
        // One copy for vertex 0, and one for each end of the crease.
        assert_eq!(split_normals_at_features(&mut mesh, &features), 3);
        let top = mesh.indices[0];
        let side = mesh.indices[6];
        assert_ne!(top, side);
        assert!(normal(&mesh, top).dot(DVec3::Z) > 1.0 - 1e-6);
        assert!(normal(&mesh, side).dot(DVec3::NEG_Y) > 1.0 - 1e-6);
    }

    /// A sub-cell tilted facet sharing a fan with a large flat triangle
    /// barely moves the normal, however wide its corner: the streak guard.
    #[test]
    fn fan_normal_weights_faces_by_area() {
        let positions = vec![
            DVec3::ZERO,
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(100.0, 10.0, 0.0),
            DVec3::new(0.0, 0.1, 0.0),
            DVec3::new(0.0, 0.0, 0.1),
        ];
        // (0, 1, 2): a long flat needle facing +z, 10 degrees at vertex 0.
        // (0, 3, 4): a tiny facet facing +x, 90 degrees at vertex 0.
        let indices = vec![0, 1, 2, 0, 3, 4];
        // A feature edge away from vertex 0, so its normal is re-derived
        // without being split.
        let features = FeatureEdges::from_keys([edge_key(1, 2)]);
        let mut mesh = mesh_of(&positions, &indices);
        assert_eq!(split_normals_at_features(&mut mesh, &features), 0);
        assert!(
            normal(&mesh, 0).dot(DVec3::Z) > 1.0 - 1e-6,
            "{:?}",
            normal(&mesh, 0)
        );
    }

    #[test]
    fn no_features_means_no_change() {
        let (positions, indices) = crate::sharp_features::feature_edges::unit_cube();
        let mut mesh = mesh_of(&positions, &indices);
        let before = (
            mesh.vertices.clone(),
            mesh.normals.clone(),
            mesh.indices.clone(),
        );
        assert_eq!(
            split_normals_at_features(&mut mesh, &FeatureEdges::default()),
            0
        );
        assert_eq!(before, (mesh.vertices, mesh.normals, mesh.indices));
    }
}
