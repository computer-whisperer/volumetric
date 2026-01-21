//! Stage 3: Topology Finalization
//!
//! Converts sparse EdgeId-based triangles to an indexed mesh with proper
//! vertex indices and accumulated face normals.

use std::collections::HashMap;

use crate::adaptive_surface_nets_2::types::{EdgeId, IndexedMesh2, SparseTriangle, Stage3Result};

/// Compute the face normal for a triangle (not normalized).
/// Returns the cross product of two edges, with magnitude proportional to area.
fn compute_face_normal(v0: (f64, f64, f64), v1: (f64, f64, f64), v2: (f64, f64, f64)) -> (f64, f64, f64) {
    // Edge vectors
    let e1 = (v1.0 - v0.0, v1.1 - v0.1, v1.2 - v0.2);
    let e2 = (v2.0 - v0.0, v2.1 - v0.1, v2.2 - v0.2);

    // Cross product
    (
        e1.1 * e2.2 - e1.2 * e2.1,
        e1.2 * e2.0 - e1.0 * e2.2,
        e1.0 * e2.1 - e1.1 * e2.0,
    )
}

/// Normalize a vector, returning (0,1,0) if the vector is too small.
pub fn normalize_or_default(v: (f64, f64, f64)) -> (f32, f32, f32) {
    let len_sq = v.0 * v.0 + v.1 * v.1 + v.2 * v.2;
    if len_sq > 1e-12 {
        let inv_len = 1.0 / len_sq.sqrt();
        (
            (v.0 * inv_len) as f32,
            (v.1 * inv_len) as f32,
            (v.2 * inv_len) as f32,
        )
    } else {
        (0.0, 1.0, 0.0) // Default up vector
    }
}

/// Stage 3: Topology Finalization
///
/// Converts sparse EdgeId-based triangles to an indexed mesh with proper
/// vertex indices and accumulated face normals.
///
/// # Algorithm
/// 1. Collect all unique EdgeIds and assign monotonic vertex indices
/// 2. Compute initial vertex positions from edge midpoints
/// 3. Rewrite triangle indices from EdgeIds to vertex indices
/// 4. Compute and accumulate face normals per vertex
pub fn stage3_topology_finalization(
    sparse_triangles: Vec<SparseTriangle>,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
) -> Stage3Result {
    // Step 1: Collect unique EdgeIds and assign monotonic indices
    let mut edge_to_vertex: HashMap<EdgeId, u32> = HashMap::new();
    let mut next_vertex_idx = 0u32;

    for tri in &sparse_triangles {
        for edge_id in &tri.vertices {
            edge_to_vertex.entry(*edge_id).or_insert_with(|| {
                let idx = next_vertex_idx;
                next_vertex_idx += 1;
                idx
            });
        }
    }

    let vertex_count = next_vertex_idx as usize;

    // Step 2: Compute initial vertex positions from edge midpoints
    let mut vertices: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); vertex_count];

    for (edge_id, &vertex_idx) in &edge_to_vertex {
        let pos = edge_id.midpoint_world_pos(bounds_min, cell_size);
        vertices[vertex_idx as usize] = pos;
    }

    // Step 3: Rewrite triangle indices
    let mut indices: Vec<u32> = Vec::with_capacity(sparse_triangles.len() * 3);

    for tri in &sparse_triangles {
        indices.push(edge_to_vertex[&tri.vertices[0]]);
        indices.push(edge_to_vertex[&tri.vertices[1]]);
        indices.push(edge_to_vertex[&tri.vertices[2]]);
    }

    // Step 4: Compute and accumulate face normals per vertex
    let mut accumulated_normals: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); vertex_count];

    for tri_idx in 0..(indices.len() / 3) {
        let i0 = indices[tri_idx * 3] as usize;
        let i1 = indices[tri_idx * 3 + 1] as usize;
        let i2 = indices[tri_idx * 3 + 2] as usize;

        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        let face_normal = compute_face_normal(v0, v1, v2);

        // Accumulate to each vertex (area-weighted by virtue of unnormalized cross product)
        accumulated_normals[i0].0 += face_normal.0;
        accumulated_normals[i0].1 += face_normal.1;
        accumulated_normals[i0].2 += face_normal.2;

        accumulated_normals[i1].0 += face_normal.0;
        accumulated_normals[i1].1 += face_normal.1;
        accumulated_normals[i1].2 += face_normal.2;

        accumulated_normals[i2].0 += face_normal.0;
        accumulated_normals[i2].1 += face_normal.1;
        accumulated_normals[i2].2 += face_normal.2;
    }

    Stage3Result {
        vertices,
        accumulated_normals,
        indices,
        edge_to_vertex,
    }
}

/// Convert Stage3Result to final IndexedMesh2 (without refinement).
/// Normalizes the accumulated normals and converts to f32.
#[allow(dead_code)]
pub fn stage3_to_indexed_mesh(result: Stage3Result) -> IndexedMesh2 {
    let vertices: Vec<(f32, f32, f32)> = result
        .vertices
        .iter()
        .map(|v| (v.0 as f32, v.1 as f32, v.2 as f32))
        .collect();

    let normals: Vec<(f32, f32, f32)> = result
        .accumulated_normals
        .iter()
        .map(|n| normalize_or_default(*n))
        .collect();

    IndexedMesh2 {
        vertices,
        normals,
        indices: result.indices,
    }
}
