//! Analytical Ground Truth for Rotated Cube
//!
//! Extends the basic AnalyticalRotatedCube with complete geometric queries
//! for finding closest points on surfaces, edges, and corners.
//!
//! The rotated cube test case uses Euler angles (XYZ order):
//! - rx = 35.264° (arctan(1/√2) - ensures equal angles from all three axes)
//! - ry = 45°
//! - rz = 0°
//!
//! This rotation ensures no face normal is aligned with any coordinate axis,
//! making it an excellent test case for geometry probing algorithms.

use super::oracle::{OracleClassification, OracleHit, OracleShape};
use super::validation::generate_validation_points;


/// Analytical ground truth for a rotated unit cube centered at origin.
///
/// The cube has half-side length 0.5 (total side length 1.0), centered at (0,0,0).
/// Face normals and vertex positions are computed analytically after rotation.
#[derive(Clone, Debug)]
pub struct AnalyticalRotatedCube {
    /// The 6 face normals after rotation: +X, -X, +Y, -Y, +Z, -Z faces
    pub face_normals: [(f64, f64, f64); 6],
    /// Face plane offsets (distance from origin along normal)
    /// For unit cube: always 0.5
    pub face_offsets: [f64; 6],
    /// The 8 corner positions after rotation
    pub corners: [(f64, f64, f64); 8],
    /// The 12 edges, each defined by two corner indices
    pub edge_corner_pairs: [(usize, usize); 12],
    /// The rotation matrix (for transforming points to/from cube-local space)
    pub rotation_matrix: [[f64; 3]; 3],
    /// Inverse rotation matrix (transpose, since rotation matrices are orthogonal)
    pub inverse_rotation: [[f64; 3]; 3],
}

/// Result of finding the closest surface point
#[derive(Clone, Debug)]
pub struct ClosestPointResult {
    /// The closest point on the cube surface
    pub point: (f64, f64, f64),
    /// Distance from query point to closest surface point
    pub distance: f64,
    /// Classification of the closest point
    pub classification: SurfaceClassification,
}

/// An analytical edge of the cube
#[derive(Clone, Debug)]
pub struct AnalyticalEdge {
    /// Direction vector of the edge (unit length)
    pub direction: (f64, f64, f64),
    /// A point on the edge (typically the midpoint)
    pub point_on_edge: (f64, f64, f64),
    /// Normal of face A
    pub face_a_normal: (f64, f64, f64),
    /// Normal of face B
    pub face_b_normal: (f64, f64, f64),
    /// The two corner positions that form this edge
    pub endpoints: ((f64, f64, f64), (f64, f64, f64)),
    /// Indices of the two faces that share this edge
    pub face_indices: (usize, usize),
}

/// An analytical corner of the cube
#[derive(Clone, Debug)]
pub struct AnalyticalCorner {
    /// Position of the corner
    pub position: (f64, f64, f64),
    /// Normals of the 3 faces meeting at this corner
    pub face_normals: [(f64, f64, f64); 3],
    /// Indices of the 3 faces meeting at this corner
    pub face_indices: [usize; 3],
    /// Index of this corner in the corners array
    pub corner_index: usize,
}

/// Classification of a point on the cube surface
#[derive(Clone, Debug)]
pub enum SurfaceClassification {
    /// Point is on a face (away from edges)
    OnFace {
        normal: (f64, f64, f64),
        face_index: usize,
    },
    /// Point is on an edge (between two faces)
    OnEdge { edge: AnalyticalEdge },
    /// Point is at a corner (three faces meet)
    OnCorner { corner: AnalyticalCorner },
}

impl AnalyticalRotatedCube {
    /// Create analytical ground truth for rotated cube with given Euler angles (degrees).
    /// Uses XYZ rotation order: R = Rz * Ry * Rx
    ///
    /// The cube is a unit cube centered at origin (corners at ±0.5 on each axis before rotation).
    pub fn new(rx_deg: f64, ry_deg: f64, rz_deg: f64) -> Self {
        let rx = rx_deg.to_radians();
        let ry = ry_deg.to_radians();
        let rz = rz_deg.to_radians();

        let (sx, cx) = (rx.sin(), rx.cos());
        let (sy, cy) = (ry.sin(), ry.cos());
        let (sz, cz) = (rz.sin(), rz.cos());

        // Combined rotation matrix R = Rz * Ry * Rx (row-major)
        let rotation_matrix = [
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy],
        ];

        // Inverse is transpose for rotation matrices
        let inverse_rotation = [
            [
                rotation_matrix[0][0],
                rotation_matrix[1][0],
                rotation_matrix[2][0],
            ],
            [
                rotation_matrix[0][1],
                rotation_matrix[1][1],
                rotation_matrix[2][1],
            ],
            [
                rotation_matrix[0][2],
                rotation_matrix[1][2],
                rotation_matrix[2][2],
            ],
        ];

        // Original face normals (unit cube faces)
        let original_normals = [
            (1.0, 0.0, 0.0),  // +X face
            (-1.0, 0.0, 0.0), // -X face
            (0.0, 1.0, 0.0),  // +Y face
            (0.0, -1.0, 0.0), // -Y face
            (0.0, 0.0, 1.0),  // +Z face
            (0.0, 0.0, -1.0), // -Z face
        ];

        // Apply rotation to each normal
        let mut face_normals = [(0.0, 0.0, 0.0); 6];
        for (i, &n) in original_normals.iter().enumerate() {
            face_normals[i] = Self::apply_matrix(&rotation_matrix, n);
        }

        // Face offsets are all 0.5 for unit cube (half-side length)
        let face_offsets = [0.5; 6];

        // Original corner positions (±0.5 on each axis)
        // Using standard corner ordering: (0,0,0), (1,0,0), (0,1,0), (1,1,0), ...
        let h = 0.5;
        let original_corners = [
            (-h, -h, -h), // 0: ---
            (h, -h, -h),  // 1: +--
            (-h, h, -h),  // 2: -+-
            (h, h, -h),   // 3: ++-
            (-h, -h, h),  // 4: --+
            (h, -h, h),   // 5: +-+
            (-h, h, h),   // 6: -++
            (h, h, h),    // 7: +++
        ];

        let mut corners = [(0.0, 0.0, 0.0); 8];
        for (i, &c) in original_corners.iter().enumerate() {
            corners[i] = Self::apply_matrix(&rotation_matrix, c);
        }

        // Edge definitions: pairs of corner indices that share an edge
        // Edges along X: 0-1, 2-3, 4-5, 6-7
        // Edges along Y: 0-2, 1-3, 4-6, 5-7
        // Edges along Z: 0-4, 1-5, 2-6, 3-7
        let edge_corner_pairs = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7), // X edges
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7), // Y edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // Z edges
        ];

        Self {
            face_normals,
            face_offsets,
            corners,
            edge_corner_pairs,
            rotation_matrix,
            inverse_rotation,
        }
    }

    /// Create with the standard test cube rotation (rx=35.264°, ry=45°, rz=0°).
    ///
    /// This rotation ensures no face is axis-aligned:
    /// - rx = arctan(1/√2) ≈ 35.264° rotates so the body diagonal aligns with Z
    /// - ry = 45° rotates around Y, ensuring all faces are at compound angles
    pub fn standard_test_cube() -> Self {
        // arctan(1/√2) in degrees
        let rx_deg = (1.0_f64 / 2.0_f64.sqrt()).atan().to_degrees();
        Self::new(rx_deg, 45.0, 0.0)
    }

    /// Create from an arbitrary 3x3 rotation matrix.
    ///
    /// The matrix should be orthogonal (R^T R = I) with determinant +1.
    /// This is useful for random rotation augmentation during training.
    pub fn from_rotation_matrix(rotation_matrix: [[f64; 3]; 3]) -> Self {
        // Inverse is transpose for rotation matrices
        let inverse_rotation = [
            [
                rotation_matrix[0][0],
                rotation_matrix[1][0],
                rotation_matrix[2][0],
            ],
            [
                rotation_matrix[0][1],
                rotation_matrix[1][1],
                rotation_matrix[2][1],
            ],
            [
                rotation_matrix[0][2],
                rotation_matrix[1][2],
                rotation_matrix[2][2],
            ],
        ];

        // Original face normals (unit cube faces)
        let original_normals = [
            (1.0, 0.0, 0.0),  // +X face
            (-1.0, 0.0, 0.0), // -X face
            (0.0, 1.0, 0.0),  // +Y face
            (0.0, -1.0, 0.0), // -Y face
            (0.0, 0.0, 1.0),  // +Z face
            (0.0, 0.0, -1.0), // -Z face
        ];

        // Apply rotation to each normal
        let mut face_normals = [(0.0, 0.0, 0.0); 6];
        for (i, &n) in original_normals.iter().enumerate() {
            face_normals[i] = Self::apply_matrix(&rotation_matrix, n);
        }

        // Face offsets are all 0.5 for unit cube
        let face_offsets = [0.5; 6];

        // Original corner positions
        let h = 0.5;
        let original_corners = [
            (-h, -h, -h),
            (h, -h, -h),
            (-h, h, -h),
            (h, h, -h),
            (-h, -h, h),
            (h, -h, h),
            (-h, h, h),
            (h, h, h),
        ];

        let mut corners = [(0.0, 0.0, 0.0); 8];
        for (i, &c) in original_corners.iter().enumerate() {
            corners[i] = Self::apply_matrix(&rotation_matrix, c);
        }

        let edge_corner_pairs = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];

        Self {
            face_normals,
            face_offsets,
            corners,
            edge_corner_pairs,
            rotation_matrix,
            inverse_rotation,
        }
    }

    /// Get the rotation matrix for this cube.
    pub fn get_rotation_matrix(&self) -> [[f64; 3]; 3] {
        self.rotation_matrix
    }

    /// Apply a 3x3 matrix to a vector
    fn apply_matrix(m: &[[f64; 3]; 3], v: (f64, f64, f64)) -> (f64, f64, f64) {
        (
            m[0][0] * v.0 + m[0][1] * v.1 + m[0][2] * v.2,
            m[1][0] * v.0 + m[1][1] * v.1 + m[1][2] * v.2,
            m[2][0] * v.0 + m[2][1] * v.1 + m[2][2] * v.2,
        )
    }

    /// Transform a world point to cube-local space (unrotated)
    pub fn world_to_local(&self, p: (f64, f64, f64)) -> (f64, f64, f64) {
        Self::apply_matrix(&self.inverse_rotation, p)
    }

    /// Transform a local point to world space (rotated)
    pub fn local_to_world(&self, p: (f64, f64, f64)) -> (f64, f64, f64) {
        Self::apply_matrix(&self.rotation_matrix, p)
    }

    /// Given a point, return the closest point on the cube surface.
    ///
    /// This works by transforming to local (unrotated) space, finding the
    /// closest point on an axis-aligned unit cube, then transforming back.
    pub fn closest_surface_point(&self, p: (f64, f64, f64)) -> ClosestPointResult {
        // Transform to local space
        let local_p = self.world_to_local(p);

        // Find closest point on axis-aligned cube [-0.5, 0.5]³
        let (local_closest, local_classification) = self.closest_point_on_aabb(local_p);

        // Transform back to world space
        let world_closest = self.local_to_world(local_closest);

        // Transform classification
        let classification = self.transform_classification(local_classification);

        let distance = distance_3d(p, world_closest);

        ClosestPointResult {
            point: world_closest,
            distance,
            classification,
        }
    }

    /// Find closest point on axis-aligned bounding box [-0.5, 0.5]³
    fn closest_point_on_aabb(
        &self,
        p: (f64, f64, f64),
    ) -> ((f64, f64, f64), LocalSurfaceClassification) {
        let h = 0.5;
        let edge_threshold = 1e-9; // Threshold for considering point on edge/corner

        // Clamp point to box
        let clamped = (p.0.clamp(-h, h), p.1.clamp(-h, h), p.2.clamp(-h, h));

        // Check if point is inside the box
        let inside = p.0.abs() < h && p.1.abs() < h && p.2.abs() < h;

        if inside {
            // Point is inside: find closest face
            let distances = [
                h - p.0,  // +X face
                h + p.0,  // -X face
                h - p.1,  // +Y face
                h + p.1,  // -Y face
                h - p.2,  // +Z face
                h + p.2,  // -Z face
            ];

            let (min_idx, _) = distances
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            // Project onto that face
            let closest = match min_idx {
                0 => (h, clamped.1, clamped.2),
                1 => (-h, clamped.1, clamped.2),
                2 => (clamped.0, h, clamped.2),
                3 => (clamped.0, -h, clamped.2),
                4 => (clamped.0, clamped.1, h),
                5 => (clamped.0, clamped.1, -h),
                _ => unreachable!(),
            };

            return self.classify_local_point(closest, edge_threshold);
        }

        // Point is outside: use clamped point and classify it
        self.classify_local_point(clamped, edge_threshold)
    }

    /// Classify a local point that's on the AABB surface
    fn classify_local_point(
        &self,
        p: (f64, f64, f64),
        threshold: f64,
    ) -> ((f64, f64, f64), LocalSurfaceClassification) {
        let h = 0.5;

        // Count how many axes are at the boundary
        let on_x_boundary = (p.0.abs() - h).abs() < threshold;
        let on_y_boundary = (p.1.abs() - h).abs() < threshold;
        let on_z_boundary = (p.2.abs() - h).abs() < threshold;

        let boundary_count = on_x_boundary as u8 + on_y_boundary as u8 + on_z_boundary as u8;

        match boundary_count {
            3 => {
                // Corner
                let corner_idx = Self::corner_index_from_signs(p.0 > 0.0, p.1 > 0.0, p.2 > 0.0);
                (p, LocalSurfaceClassification::Corner(corner_idx))
            }
            2 => {
                // Edge - find which two faces share this edge
                let (face_a, face_b, edge_axis) = if !on_x_boundary {
                    // Y-Z edge
                    (
                        if p.1 > 0.0 { 2 } else { 3 },
                        if p.2 > 0.0 { 4 } else { 5 },
                        0,
                    )
                } else if !on_y_boundary {
                    // X-Z edge
                    (
                        if p.0 > 0.0 { 0 } else { 1 },
                        if p.2 > 0.0 { 4 } else { 5 },
                        1,
                    )
                } else {
                    // X-Y edge
                    (
                        if p.0 > 0.0 { 0 } else { 1 },
                        if p.1 > 0.0 { 2 } else { 3 },
                        2,
                    )
                };
                (p, LocalSurfaceClassification::Edge(face_a, face_b, edge_axis))
            }
            1 => {
                // Face
                let face_idx = if on_x_boundary {
                    if p.0 > 0.0 { 0 } else { 1 }
                } else if on_y_boundary {
                    if p.1 > 0.0 { 2 } else { 3 }
                } else {
                    if p.2 > 0.0 { 4 } else { 5 }
                };
                (p, LocalSurfaceClassification::Face(face_idx))
            }
            _ => {
                // Should not happen for a point on the surface
                // Default to nearest face
                (p, LocalSurfaceClassification::Face(0))
            }
        }
    }

    /// Get corner index from coordinate signs
    fn corner_index_from_signs(x_pos: bool, y_pos: bool, z_pos: bool) -> usize {
        (x_pos as usize) + (y_pos as usize) * 2 + (z_pos as usize) * 4
    }

    /// Transform local classification to world classification
    fn transform_classification(
        &self,
        local: LocalSurfaceClassification,
    ) -> SurfaceClassification {
        match local {
            LocalSurfaceClassification::Face(face_idx) => SurfaceClassification::OnFace {
                normal: self.face_normals[face_idx],
                face_index: face_idx,
            },
            LocalSurfaceClassification::Edge(face_a, face_b, edge_axis) => {
                let edge = self.build_analytical_edge(face_a, face_b, edge_axis);
                SurfaceClassification::OnEdge { edge }
            }
            LocalSurfaceClassification::Corner(corner_idx) => {
                let corner = self.build_analytical_corner(corner_idx);
                SurfaceClassification::OnCorner { corner }
            }
        }
    }

    /// Build an AnalyticalEdge from face indices
    fn build_analytical_edge(
        &self,
        face_a: usize,
        face_b: usize,
        _edge_axis: usize,
    ) -> AnalyticalEdge {
        // Edge direction is cross product of face normals (normalized)
        let na = self.face_normals[face_a];
        let nb = self.face_normals[face_b];
        let dir = cross(na, nb);
        let dir = normalize(dir);

        // Find the corners that share both faces
        let corners_on_edge = self.find_corners_on_edge(face_a, face_b);
        let endpoint_a = self.corners[corners_on_edge.0];
        let endpoint_b = self.corners[corners_on_edge.1];

        // Midpoint
        let midpoint = (
            (endpoint_a.0 + endpoint_b.0) / 2.0,
            (endpoint_a.1 + endpoint_b.1) / 2.0,
            (endpoint_a.2 + endpoint_b.2) / 2.0,
        );

        AnalyticalEdge {
            direction: dir,
            point_on_edge: midpoint,
            face_a_normal: na,
            face_b_normal: nb,
            endpoints: (endpoint_a, endpoint_b),
            face_indices: (face_a, face_b),
        }
    }

    /// Find the two corners that lie on the edge between two faces
    fn find_corners_on_edge(&self, face_a: usize, face_b: usize) -> (usize, usize) {
        let corners_a = Self::corners_of_face(face_a);
        let corners_b = Self::corners_of_face(face_b);

        let mut shared = Vec::new();
        for &ca in &corners_a {
            if corners_b.contains(&ca) {
                shared.push(ca);
            }
        }

        if shared.len() == 2 {
            (shared[0], shared[1])
        } else {
            // Fallback - should not happen for adjacent faces
            (0, 1)
        }
    }

    /// Get the 4 corner indices that belong to a face
    fn corners_of_face(face_idx: usize) -> [usize; 4] {
        match face_idx {
            0 => [1, 3, 5, 7], // +X face
            1 => [0, 2, 4, 6], // -X face
            2 => [2, 3, 6, 7], // +Y face
            3 => [0, 1, 4, 5], // -Y face
            4 => [4, 5, 6, 7], // +Z face
            5 => [0, 1, 2, 3], // -Z face
            _ => [0, 0, 0, 0],
        }
    }

    /// Build an AnalyticalCorner from corner index
    fn build_analytical_corner(&self, corner_idx: usize) -> AnalyticalCorner {
        // Find the 3 faces that share this corner
        let faces = Self::faces_of_corner(corner_idx);

        AnalyticalCorner {
            position: self.corners[corner_idx],
            face_normals: [
                self.face_normals[faces[0]],
                self.face_normals[faces[1]],
                self.face_normals[faces[2]],
            ],
            face_indices: faces,
            corner_index: corner_idx,
        }
    }

    /// Get the 3 face indices that share a corner
    fn faces_of_corner(corner_idx: usize) -> [usize; 3] {
        // Corner index encodes signs: bit0=X, bit1=Y, bit2=Z
        let x_pos = (corner_idx & 1) != 0;
        let y_pos = (corner_idx & 2) != 0;
        let z_pos = (corner_idx & 4) != 0;

        [
            if x_pos { 0 } else { 1 }, // +X or -X face
            if y_pos { 2 } else { 3 }, // +Y or -Y face
            if z_pos { 4 } else { 5 }, // +Z or -Z face
        ]
    }

    /// Given a point, return the closest edge if within threshold distance.
    ///
    /// # Arguments
    /// * `p` - Query point in world space
    /// * `threshold` - Maximum distance to consider (world units)
    pub fn closest_edge(&self, p: (f64, f64, f64), threshold: f64) -> Option<AnalyticalEdge> {
        let mut best_edge: Option<AnalyticalEdge> = None;
        let mut best_distance = threshold;

        for i in 0..12 {
            let edge = self.get_edge(i);
            let dist = self.distance_to_edge(p, &edge);

            if dist < best_distance {
                best_distance = dist;
                best_edge = Some(edge);
            }
        }

        best_edge
    }

    /// Get the i-th corner (0-7) as an AnalyticalCorner
    pub fn get_corner(&self, corner_idx: usize) -> AnalyticalCorner {
        self.build_analytical_corner(corner_idx)
    }

    /// Get the i-th edge (0-11) as an AnalyticalEdge
    pub fn get_edge(&self, edge_idx: usize) -> AnalyticalEdge {
        let (c0, c1) = self.edge_corner_pairs[edge_idx];
        let p0 = self.corners[c0];
        let p1 = self.corners[c1];

        // Find the two faces that share this edge
        let (face_a, face_b) = self.faces_sharing_edge(c0, c1);

        let dir = normalize(sub(p1, p0));
        let midpoint = (
            (p0.0 + p1.0) / 2.0,
            (p0.1 + p1.1) / 2.0,
            (p0.2 + p1.2) / 2.0,
        );

        AnalyticalEdge {
            direction: dir,
            point_on_edge: midpoint,
            face_a_normal: self.face_normals[face_a],
            face_b_normal: self.face_normals[face_b],
            endpoints: (p0, p1),
            face_indices: (face_a, face_b),
        }
    }

    /// Find the two faces that share an edge defined by two corners
    fn faces_sharing_edge(&self, c0: usize, c1: usize) -> (usize, usize) {
        let mut shared_faces = Vec::new();

        for face_idx in 0..6 {
            let corners = Self::corners_of_face(face_idx);
            if corners.contains(&c0) && corners.contains(&c1) {
                shared_faces.push(face_idx);
            }
        }

        if shared_faces.len() == 2 {
            (shared_faces[0], shared_faces[1])
        } else {
            (0, 1)
        }
    }

    /// Distance from point to edge (line segment)
    fn distance_to_edge(&self, p: (f64, f64, f64), edge: &AnalyticalEdge) -> f64 {
        let (a, b) = edge.endpoints;
        let ab = sub(b, a);
        let ap = sub(p, a);

        let t = dot(ap, ab) / dot(ab, ab);
        let t_clamped = t.clamp(0.0, 1.0);

        let closest = (
            a.0 + t_clamped * ab.0,
            a.1 + t_clamped * ab.1,
            a.2 + t_clamped * ab.2,
        );

        distance_3d(p, closest)
    }

    /// Given a point, return the closest corner if within threshold distance.
    pub fn closest_corner(&self, p: (f64, f64, f64), threshold: f64) -> Option<AnalyticalCorner> {
        let mut best_corner: Option<AnalyticalCorner> = None;
        let mut best_distance = threshold;

        for i in 0..8 {
            let dist = distance_3d(p, self.corners[i]);

            if dist < best_distance {
                best_distance = dist;
                best_corner = Some(self.build_analytical_corner(i));
            }
        }

        best_corner
    }

    /// Classify a surface point as on a face, edge, or corner.
    ///
    /// The point should be on or very close to the surface.
    /// Uses threshold to determine if point is "on" an edge or corner.
    pub fn classify_surface_point(
        &self,
        p: (f64, f64, f64),
        edge_threshold: f64,
    ) -> SurfaceClassification {
        let result = self.closest_surface_point(p);
        let surface_point = result.point;

        // Check for corner first (most specific)
        if let Some(corner) = self.closest_corner(surface_point, edge_threshold) {
            return SurfaceClassification::OnCorner { corner };
        }

        // Check for edge
        if let Some(edge) = self.closest_edge(surface_point, edge_threshold) {
            return SurfaceClassification::OnEdge { edge };
        }

        // Must be on a face
        result.classification
    }

    /// Find the best-matching analytical normal for a given normal.
    /// Returns (best_normal, error_degrees).
    pub fn find_best_match(&self, normal: (f64, f64, f64)) -> ((f64, f64, f64), f64) {
        let mut best_normal = self.face_normals[0];
        let mut best_error = 180.0f64;

        for &face_normal in &self.face_normals {
            let error = angle_between(normal, face_normal).to_degrees();
            if error < best_error {
                best_error = error;
                best_normal = face_normal;
            }
        }

        (best_normal, best_error)
    }

    /// Compute pair errors for edge validation
    pub fn compute_pair_errors(
        &self,
        normal_a: (f64, f64, f64),
        normal_b: Option<(f64, f64, f64)>,
    ) -> (f64, Option<f64>) {
        let (_, error_a) = self.find_best_match(normal_a);

        let error_b = normal_b.map(|nb| {
            let (_, err) = self.find_best_match(nb);
            err
        });

        (error_a, error_b)
    }

    /// Get the center of a face (world coordinates)
    pub fn face_center(&self, face_idx: usize) -> (f64, f64, f64) {
        // Face center in local coords is 0.5 * face_normal (unrotated)
        let h = 0.5;
        let local_center = match face_idx {
            0 => (h, 0.0, 0.0),
            1 => (-h, 0.0, 0.0),
            2 => (0.0, h, 0.0),
            3 => (0.0, -h, 0.0),
            4 => (0.0, 0.0, h),
            5 => (0.0, 0.0, -h),
            _ => (0.0, 0.0, 0.0),
        };
        self.local_to_world(local_center)
    }

    /// Get the midpoint of an edge (world coordinates)
    pub fn edge_midpoint(&self, edge_idx: usize) -> (f64, f64, f64) {
        let (c0, c1) = self.edge_corner_pairs[edge_idx];
        let p0 = self.corners[c0];
        let p1 = self.corners[c1];
        (
            (p0.0 + p1.0) / 2.0,
            (p0.1 + p1.1) / 2.0,
            (p0.2 + p1.2) / 2.0,
        )
    }
}

impl OracleShape for AnalyticalRotatedCube {
    fn name(&self) -> &str {
        "AnalyticalRotatedCube"
    }

    fn classify(&self, point: (f64, f64, f64)) -> OracleHit {
        let result = self.closest_surface_point(point);
        match result.classification {
            SurfaceClassification::OnFace { normal, .. } => OracleHit {
                classification: OracleClassification::Face,
                surface_position: result.point,
                normals: vec![normal],
                edge_direction: None,
                corner_position: None,
            },
            SurfaceClassification::OnEdge { edge } => OracleHit {
                classification: OracleClassification::Edge,
                surface_position: result.point,
                normals: vec![edge.face_a_normal, edge.face_b_normal],
                edge_direction: Some(edge.direction),
                corner_position: None,
            },
            SurfaceClassification::OnCorner { corner } => OracleHit {
                classification: OracleClassification::Corner,
                surface_position: result.point,
                normals: vec![
                    corner.face_normals[0],
                    corner.face_normals[1],
                    corner.face_normals[2],
                ],
                edge_direction: None,
                corner_position: Some(corner.position),
            },
        }
    }

    fn validation_points(&self, _seed: u64) -> Vec<(f64, f64, f64)> {
        generate_validation_points(self)
            .into_iter()
            .map(|p| p.position)
            .collect()
    }
}

/// Local (unrotated) classification
#[derive(Clone, Debug)]
enum LocalSurfaceClassification {
    Face(usize),
    Edge(usize, usize, usize), // face_a, face_b, edge_axis
    Corner(usize),
}

// Vector math helpers

fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn length(v: (f64, f64, f64)) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = length(v);
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 1.0)
    }
}

fn distance_3d(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    length(sub(a, b))
}

/// Angle between two vectors in radians
pub fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let a_norm = normalize(a);
    let b_norm = normalize(b);
    let d = dot(a_norm, b_norm).clamp(-1.0, 1.0);
    d.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_cube_face_normals() {
        let cube = AnalyticalRotatedCube::standard_test_cube();

        // All face normals should be unit length
        for normal in &cube.face_normals {
            let len = length(*normal);
            assert!(
                (len - 1.0).abs() < 1e-10,
                "Face normal not unit length: {}",
                len
            );
        }

        // Opposite faces should have opposite normals
        for i in 0..3 {
            let n1 = cube.face_normals[i * 2];
            let n2 = cube.face_normals[i * 2 + 1];
            let sum = (n1.0 + n2.0, n1.1 + n2.1, n1.2 + n2.2);
            assert!(
                length(sum) < 1e-10,
                "Opposite faces not opposite: {:?} vs {:?}",
                n1,
                n2
            );
        }
    }

    #[test]
    fn test_closest_surface_point_at_face_center() {
        let cube = AnalyticalRotatedCube::standard_test_cube();

        for face_idx in 0..6 {
            let center = cube.face_center(face_idx);
            let result = cube.closest_surface_point(center);

            // Should be on the face
            match &result.classification {
                SurfaceClassification::OnFace {
                    face_index,
                    normal,
                } => {
                    assert_eq!(
                        *face_index, face_idx,
                        "Wrong face index at center of face {}",
                        face_idx
                    );
                    let expected_normal = cube.face_normals[face_idx];
                    let error = angle_between(*normal, expected_normal);
                    assert!(error < 1e-10, "Wrong normal at face center");
                }
                other => panic!(
                    "Face center {} classified as {:?}, expected OnFace",
                    face_idx, other
                ),
            }

            // Distance should be nearly zero
            assert!(
                result.distance < 1e-10,
                "Face center not on surface: d={}",
                result.distance
            );
        }
    }

    #[test]
    fn test_closest_surface_point_at_edge() {
        let cube = AnalyticalRotatedCube::standard_test_cube();

        for edge_idx in 0..12 {
            let midpoint = cube.edge_midpoint(edge_idx);
            let result = cube.closest_surface_point(midpoint);

            // Should be on an edge
            match &result.classification {
                SurfaceClassification::OnEdge { edge } => {
                    // Verify edge direction is unit length
                    let len = length(edge.direction);
                    assert!((len - 1.0).abs() < 1e-10);

                    // Verify two face normals are different
                    let angle = angle_between(edge.face_a_normal, edge.face_b_normal);
                    assert!(
                        angle > 0.1,
                        "Edge faces should have different normals, angle={}",
                        angle.to_degrees()
                    );
                }
                other => panic!(
                    "Edge midpoint {} classified as {:?}, expected OnEdge",
                    edge_idx, other
                ),
            }
        }
    }

    #[test]
    fn test_closest_surface_point_at_corner() {
        let cube = AnalyticalRotatedCube::standard_test_cube();

        for corner_idx in 0..8 {
            let corner_pos = cube.corners[corner_idx];
            let result = cube.closest_surface_point(corner_pos);

            // Should be on a corner
            match &result.classification {
                SurfaceClassification::OnCorner { corner } => {
                    assert_eq!(corner.corner_index, corner_idx);
                    // Should have 3 distinct face normals
                    assert_eq!(corner.face_normals.len(), 3);
                }
                other => panic!(
                    "Corner {} classified as {:?}, expected OnCorner",
                    corner_idx, other
                ),
            }
        }
    }

    #[test]
    fn test_closest_edge_from_nearby_point() {
        let cube = AnalyticalRotatedCube::standard_test_cube();

        // Point slightly outside an edge midpoint
        let edge = cube.get_edge(0);
        let offset = normalize(edge.face_a_normal);
        let nearby = (
            edge.point_on_edge.0 + offset.0 * 0.1,
            edge.point_on_edge.1 + offset.1 * 0.1,
            edge.point_on_edge.2 + offset.2 * 0.1,
        );

        let found = cube.closest_edge(nearby, 0.2);
        assert!(found.is_some(), "Should find edge within threshold");

        let found_edge = found.unwrap();
        let dir_angle = angle_between(found_edge.direction, edge.direction);
        // Direction might be opposite, so check both
        let dir_angle_min = dir_angle.min(std::f64::consts::PI - dir_angle);
        assert!(
            dir_angle_min < 1e-6,
            "Found edge direction differs: {}°",
            dir_angle.to_degrees()
        );
    }

    #[test]
    fn test_no_axis_aligned_normals() {
        let cube = AnalyticalRotatedCube::standard_test_cube();

        // For standard test cube, no normal should be axis-aligned
        // A normal is "axis-aligned" if it's nearly parallel to an axis (angle close to 0 or 180)
        let axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];

        for normal in &cube.face_normals {
            for axis in &axes {
                let angle = angle_between(*normal, *axis).to_degrees();
                // Not nearly parallel (close to 0°) or anti-parallel (close to 180°)
                let nearly_parallel = angle < 10.0 || angle > 170.0;
                assert!(
                    !nearly_parallel,
                    "Normal {:?} is nearly parallel to axis {:?} (angle = {:.1}°)",
                    normal,
                    axis,
                    angle
                );
            }
        }
    }
}
