//! Conversion functions for renderer types.

use super::{MeshData, MeshVertex, PointData, PointInstance};

/// Package mesh vertices into MeshData.
pub fn convert_mesh_data(vertices: &[MeshVertex], indices: Option<&[u32]>) -> MeshData {
    MeshData {
        vertices: vertices.to_vec(),
        indices: indices.map(|i| i.to_vec()),
    }
}

/// Convert point positions to PointData with gradient coloring.
pub fn convert_points_to_point_data(points: &[(f32, f32, f32)]) -> PointData {
    PointData {
        points: points
            .iter()
            .map(|&(x, y, z)| PointInstance {
                position: [x, y, z],
                color: [
                    (x + 1.0) * 0.5, // R gradient
                    (y + 1.0) * 0.5, // G gradient
                    (z + 1.0) * 0.5, // B gradient
                    1.0,
                ],
            })
            .collect(),
    }
}
