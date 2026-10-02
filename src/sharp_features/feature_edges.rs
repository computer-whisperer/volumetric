//! Feature edges: the mesh edges a crease runs along.
//!
//! An edge is a feature edge when the angle between its two faces exceeds
//! the crease angle. That is a property of the geometry alone: no region
//! labels are involved, so a crease whose two sides belong to one smooth
//! region (a hole tangent to a wall it shares a face with) is found like any
//! other.
//!
//! The classification is made once, on the fine welded mesh, where a curved
//! surface turns a few degrees per edge and a crease turns tens. Decimation
//! then carries the set along ([`crate::mesh_decimation`]): after it a small
//! hole's facets can stand 40 degrees apart, and classifying there would
//! read them as creases. The final normal split
//! ([`crate::sharp_features::normals`]) cuts along whatever set reaches it.
//!
//! Where snapping failed, the sawtooth it left is itself full of feature
//! edges, and nearly all its vertices are corners. They stay at cell pitch
//! through decimation and every tooth shades flat: the honest picture of
//! geometry the snap stage could not resolve.

use std::collections::HashSet;

use glam::DVec3;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FeatureEdgeConfig {
    /// Edges whose faces meet at more than this angle (degrees between the
    /// face normals) are feature edges. A strut two cells in radius turns
    /// about 28 degrees per edge, so anything thinner reads as creased.
    pub crease_angle_deg: f64,
}

impl Default for FeatureEdgeConfig {
    fn default() -> Self {
        Self {
            crease_angle_deg: 30.0,
        }
    }
}

/// Order-independent key of the edge between two vertices.
#[inline]
pub fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((hi as u64) << 32) | lo as u64
}

/// The two vertices of an [`edge_key`], lower index first.
#[inline]
pub fn edge_vertices(key: u64) -> (u32, u32) {
    ((key & 0xffff_ffff) as u32, (key >> 32) as u32)
}

/// A set of feature edges over one mesh's vertex indices.
///
/// A vertex's class follows from how many feature edges meet at it: none
/// for a smooth vertex, two for a vertex on a crease, anything else for a
/// corner (a crease end, or three or more creases meeting).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FeatureEdges {
    edges: HashSet<u64>,
}

impl FeatureEdges {
    /// Classify the edges of a mesh.
    ///
    /// Beyond the crease angle, an edge also counts when one of its faces
    /// has no area (the angle is undefined, and such debris must not join
    /// the faces either side of it into one smooth fan) and when more than
    /// two faces share it. An open border edge (one face) does not: the fan
    /// ends there anyway, and decimation has its own border rule.
    pub fn classify(positions: &[DVec3], indices: &[u32], config: &FeatureEdgeConfig) -> Self {
        let tri_count = indices.len() / 3;
        let face_normals: Vec<Option<DVec3>> = crate::parallel_iter::map_range(0..tri_count, |t| {
            let (a, b, c) = (
                positions[indices[t * 3] as usize],
                positions[indices[t * 3 + 1] as usize],
                positions[indices[t * 3 + 2] as usize],
            );
            crate::sharp_features::cleanup::face_unit_normal(a, b, c)
        });

        // Every face's three edges, sorted so the faces sharing an edge sit
        // next to each other.
        let mut edge_faces: Vec<(u64, u32)> = Vec::with_capacity(indices.len());
        for (t, tri) in indices.chunks_exact(3).enumerate() {
            for k in 0..3 {
                edge_faces.push((edge_key(tri[k], tri[(k + 1) % 3]), t as u32));
            }
        }
        crate::parallel_iter::sort_unstable(&mut edge_faces);

        let min_dot = config.crease_angle_deg.to_radians().cos();
        let mut edges = HashSet::new();
        let mut start = 0;
        while start < edge_faces.len() {
            let key = edge_faces[start].0;
            let mut end = start + 1;
            while end < edge_faces.len() && edge_faces[end].0 == key {
                end += 1;
            }
            let is_feature = match end - start {
                1 => false,
                2 => match (
                    face_normals[edge_faces[start].1 as usize],
                    face_normals[edge_faces[start + 1].1 as usize],
                ) {
                    (Some(n0), Some(n1)) => n0.dot(n1) < min_dot,
                    _ => true,
                },
                _ => true,
            };
            if is_feature {
                edges.insert(key);
            }
            start = end;
        }
        Self { edges }
    }

    pub fn from_keys(keys: impl IntoIterator<Item = u64>) -> Self {
        Self {
            edges: keys.into_iter().collect(),
        }
    }

    pub fn into_keys(self) -> HashSet<u64> {
        self.edges
    }

    #[inline]
    pub fn contains(&self, a: u32, b: u32) -> bool {
        self.edges.contains(&edge_key(a, b))
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// How many feature edges meet at each vertex (saturating at 255).
    pub fn degrees(&self, vertex_count: usize) -> Vec<u8> {
        feature_degrees(&self.edges, vertex_count)
    }

    /// The same edges under a vertex renumbering; `u32::MAX` marks a vertex
    /// that no longer exists, and its edges are dropped.
    pub fn remapped(&self, remap: &[u32]) -> Self {
        let edges = self
            .edges
            .iter()
            .filter_map(|&key| {
                let (a, b) = edge_vertices(key);
                let (a, b) = (remap[a as usize], remap[b as usize]);
                (a != u32::MAX && b != u32::MAX && a != b).then(|| edge_key(a, b))
            })
            .collect();
        Self { edges }
    }
}

/// [`FeatureEdges::degrees`] over a bare key set.
pub(crate) fn feature_degrees(edges: &HashSet<u64>, vertex_count: usize) -> Vec<u8> {
    let mut degrees = vec![0u8; vertex_count];
    for &key in edges {
        let (a, b) = edge_vertices(key);
        degrees[a as usize] = degrees[a as usize].saturating_add(1);
        degrees[b as usize] = degrees[b as usize].saturating_add(1);
    }
    degrees
}

/// A unit cube, two triangles per face, eight shared vertices.
#[cfg(test)]
pub(crate) fn unit_cube() -> (Vec<DVec3>, Vec<u32>) {
    let positions: Vec<DVec3> = (0..8)
        .map(|i| DVec3::new((i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64))
        .collect();
    // Quads (a, b, c, d) wound counter-clockwise seen from outside.
    let quads: [[u32; 4]; 6] = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    let mut indices = Vec::new();
    for [a, b, c, d] in quads {
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    }
    (positions, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_has_twelve_feature_edges_and_eight_corners() {
        let (positions, indices) = unit_cube();
        let features = FeatureEdges::classify(&positions, &indices, &FeatureEdgeConfig::default());
        // The 12 cube edges, and none of the 6 face diagonals.
        assert_eq!(features.len(), 12);
        assert!(features.contains(0, 1) && features.contains(6, 7));
        assert!(!features.contains(0, 3), "a face diagonal is flat");
        assert!(features.degrees(8).iter().all(|&d| d == 3));
    }

    /// A closed prism over a regular n-gon: the rims are creases at any n,
    /// the side edges only while the polygon turns more than the crease
    /// angle per side.
    fn prism(sides: u32) -> (Vec<DVec3>, Vec<u32>) {
        let mut positions = Vec::new();
        for z in [0.0, 1.0] {
            for i in 0..sides {
                let a = std::f64::consts::TAU * i as f64 / sides as f64;
                positions.push(DVec3::new(a.cos(), a.sin(), z));
            }
        }
        positions.push(DVec3::new(0.0, 0.0, 0.0)); // bottom centre
        positions.push(DVec3::new(0.0, 0.0, 1.0)); // top centre
        let (bottom, top) = (2 * sides, 2 * sides + 1);
        let mut indices = Vec::new();
        for i in 0..sides {
            let j = (i + 1) % sides;
            indices.extend_from_slice(&[i, j, sides + j, i, sides + j, sides + i]);
            indices.extend_from_slice(&[bottom, j, i]);
            indices.extend_from_slice(&[top, sides + i, sides + j]);
        }
        (positions, indices)
    }

    #[test]
    fn cylinder_has_two_rims_and_no_corners() {
        let sides = 64; // 5.6 degrees per side
        let (positions, indices) = prism(sides);
        let features = FeatureEdges::classify(&positions, &indices, &FeatureEdgeConfig::default());
        assert_eq!(features.len(), 2 * sides as usize, "the two rims");
        let degrees = features.degrees(positions.len());
        assert!(degrees[..2 * sides as usize].iter().all(|&d| d == 2));
        assert_eq!(
            degrees[2 * sides as usize..],
            [0, 0],
            "cap centres are smooth"
        );
    }

    #[test]
    fn a_coarse_prism_is_creased_along_its_sides_too() {
        let sides = 8; // 45 degrees per side
        let (positions, indices) = prism(sides);
        let features = FeatureEdges::classify(&positions, &indices, &FeatureEdgeConfig::default());
        assert_eq!(features.len(), 3 * sides as usize);
        assert!(
            features.degrees(positions.len())[..16]
                .iter()
                .all(|&d| d == 3)
        );
    }

    #[test]
    fn sphere_has_no_feature_edges() {
        // An octahedron subdivided three times and pushed onto the sphere.
        let mut positions = vec![
            DVec3::X,
            DVec3::NEG_X,
            DVec3::Y,
            DVec3::NEG_Y,
            DVec3::Z,
            DVec3::NEG_Z,
        ];
        let mut indices: Vec<u32> = vec![
            0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
        ];
        for _ in 0..3 {
            let mut midpoints = std::collections::HashMap::new();
            let mut next = Vec::new();
            for tri in indices.chunks_exact(3) {
                let mut mid = |a: u32, b: u32| -> u32 {
                    *midpoints.entry(edge_key(a, b)).or_insert_with(|| {
                        let p = (positions[a as usize] + positions[b as usize]).normalize();
                        positions.push(p);
                        positions.len() as u32 - 1
                    })
                };
                let (a, b, c) = (tri[0], tri[1], tri[2]);
                let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
                next.extend_from_slice(&[a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca]);
            }
            indices = next;
        }
        let features = FeatureEdges::classify(&positions, &indices, &FeatureEdgeConfig::default());
        assert!(features.is_empty());
    }

    #[test]
    fn zero_area_face_and_shared_edge_are_features_but_a_border_is_not() {
        // Two coplanar triangles sharing edge (0, 1); a third, collinear
        // (zero-area) triangle hanging off edge (1, 2).
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(1.0, -1.0, 0.0),
            DVec3::new(0.5, 0.5, 0.0),
        ];
        let indices = vec![0, 1, 2, 1, 0, 3, 1, 4, 2];
        let features = FeatureEdges::classify(&positions, &indices, &FeatureEdgeConfig::default());
        assert!(!features.contains(0, 1), "flat interior edge");
        assert!(features.contains(1, 2), "edge of a zero-area face");
        assert!(!features.contains(0, 3), "open border");
    }

    #[test]
    fn remap_renumbers_and_drops_dead_vertices() {
        let features = FeatureEdges::from_keys([edge_key(0, 1), edge_key(1, 2), edge_key(2, 3)]);
        let remapped = features.remapped(&[5, 4, u32::MAX, 0]);
        assert_eq!(remapped, FeatureEdges::from_keys([edge_key(5, 4)]));
    }
}
