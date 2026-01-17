//! # Adaptive Surface Nets v2 - Architecture & Implementation
//!
//! A complete rewrite of the adaptive surface net sampler with proper organization,
//! documentation, and a well-defined algorithm.
//!
//! ## Problem Statement
//!
//! Given a volumetric sampling function `is_inside(x: f64, y: f64, z: f64) -> f32`:
//! - Returns `> 0.0` for points inside the model
//! - Returns `0.0` for points in empty space
//! - **NOT an SDF** - only the sign/threshold matters, not the magnitude
//!
//! We must produce a triangle mesh that:
//! 1. Is **airtight** (no holes or gaps) - unless the surface intersects the bounding box
//! 2. Is **non-self-intersecting**
//! 3. Has **consistent winding order** (all normals point outward)
//! 4. May have non-manifold edges (4+ triangles sharing an edge is acceptable)
//!
//! **Boundary Handling**: If the surface intersects the bounding box boundary, we return
//! an open surface. The frontier expansion simply stops at the edge of the box rather
//! than attempting to close the mesh artificially.
//!
//! ## Key Constraints
//!
//! - **Minimize sampler calls**: `is_inside` is expensive (microseconds per call, increasing
//!   with geometry complexity). This is the primary optimization target.
//! - **Accuracy over vertex count**: Spend sample budget on vertex position refinement
//!   and normal estimation rather than raw triangle count.
//! - **Multithreading**: Leverage parallel sampling. Accept occasional redundant samples
//!   rather than complex synchronization. `is_inside` is deterministic.
//! - **Integer-based topology**: ALL position-derived IDs (CuboidId, EdgeId) MUST be
//!   computed from integer cell coordinates, NEVER from floating-point positions.
//!   This ensures deterministic vertex welding across adjacent cells.
//!
//! ## Canonical Corner Indexing System
//!
//! We define a canonical mapping from corner index (0-7) to position offset within a cell.
//! This system is used consistently throughout the entire pipeline.
//!
//! ```text
//! Corner Index → (dx, dy, dz) offset from cell minimum corner:
//!   0 → (0, 0, 0)  "---"
//!   1 → (1, 0, 0)  "+--"
//!   2 → (0, 1, 0)  "-+-"
//!   3 → (1, 1, 0)  "++-"
//!   4 → (0, 0, 1)  "--+"
//!   5 → (1, 0, 1)  "+-+"
//!   6 → (0, 1, 1)  "-++"
//!   7 → (1, 1, 1)  "+++"
//!
//! Bit encoding: index = (z << 2) | (y << 1) | x
//!   - Bit 0 (LSB): X offset (0 or 1)
//!   - Bit 1: Y offset (0 or 1)
//!   - Bit 2: Z offset (0 or 1)
//! ```
//!
//! The CornerMask uses this same bit ordering: bit N is set if corner N is inside.
//!
//! ## Canonical Edge Indexing System
//!
//! Each cell has 12 edges. Edges are identified by:
//! - The minimum corner position (in finest-level integer units)
//! - The axis direction (0=X, 1=Y, 2=Z)
//!
//! ```text
//! Edge Index → (corner_a, corner_b, axis):
//!   Edges along X-axis (axis=0):
//!     0: corners 0-1, min=(x, y, z)
//!     1: corners 2-3, min=(x, y+1, z)
//!     2: corners 4-5, min=(x, y, z+1)
//!     3: corners 6-7, min=(x, y+1, z+1)
//!   Edges along Y-axis (axis=1):
//!     4: corners 0-2, min=(x, y, z)
//!     5: corners 1-3, min=(x+1, y, z)
//!     6: corners 4-6, min=(x, y, z+1)
//!     7: corners 5-7, min=(x+1, y, z+1)
//!   Edges along Z-axis (axis=2):
//!     8: corners 0-4, min=(x, y, z)
//!     9: corners 1-5, min=(x+1, y, z)
//!    10: corners 2-6, min=(x, y+1, z)
//!    11: corners 3-7, min=(x+1, y+1, z)
//! ```
//!
//! ## Algorithm Overview (4 Stages)
//!
//! ### Stage 1: Coarse Grid Discovery
//! Sample the volume at low resolution to find regions containing the surface.
//! - Grid of `base_resolution³` samples
//! - Identify "mixed" edges (inside→outside transitions)
//! - Output: Initial work queue of coarse mixed cells WITH pre-sampled corners
//!
//! ### Stage 2: Parallel Subdivision & Triangle Emission
//! Process cuboids in parallel, subdividing and expanding the frontier:
//! ```text
//! while work_queue not empty:
//!     work_item = work_queue.pop()  // includes CuboidId + [Option<bool>; 8] known corners
//!     corners = complete_corner_samples(work_item)  // only sample unknown corners
//!     if is_mixed(corners):
//!         if depth < max_depth:
//!             subdivide into 8 children
//!             for each child: propagate known corners, add to queue
//!         else:  // at max depth, emit geometry
//!             for each transitioning_edge:
//!                 edge_id = compute_edge_id(cell_coords, edge_index)  // INTEGER ONLY
//!                 emit_triangle(edge_ids, corner_states)
//!             for each neighbor (within bounds):
//!                 if neighbor_is_mixed(corners):
//!                     work_queue.add_deduplicated(neighbor, propagated_corners)
//! ```
//!
//! **Key Concepts**:
//! - **Pre-packed corner samples**: Each work queue entry carries `[Option<bool>; 8]`
//!   with already-known corner samples. This replaces the sample cache entirely and
//!   is simpler, more effective, and avoids cache synchronization issues.
//! - **Deterministic Edge IDs**: Computed ONLY from integer cell coordinates.
//!   EdgeId = (cell_x + dx, cell_y + dy, cell_z + dz, axis) where dx/dy/dz are
//!   integer offsets from the canonical edge table. Adjacent cells sharing an edge
//!   will compute identical EdgeIds, causing automatic vertex welding.
//! - Triangles are stored with their sparse edge IDs for later index rewriting.
//! - Frontier expansion stops at bounding box edges (open surfaces allowed).
//! - Output: Collection of triangles with sparse edge-based vertex IDs
//!
//! ### Stage 3: Topology Finalization
//! Convert sparse edge IDs to monotonic vertex indices:
//! - Iterate over all unique edge IDs encountered, assign each a monotonic index (0, 1, 2, ...)
//! - Rewrite triangle indices from sparse edge IDs to monotonic vertex indices
//! - Accumulate face normals per vertex (sum of adjacent triangle normals)
//! - Output: IndexedMesh with proper indices and accumulated normals per vertex
//!
//! Note: This is NOT deduplication - each edge ID is already unique. We are simply
//! converting from a sparse ID space (edge positions) to a dense monotonic space
//! (vertex buffer indices).
//!
//! ### Stage 4: Vertex Refinement & Normal Estimation
//! For each vertex:
//! 1. Use accumulated face normals as initial direction estimate
//! 2. Perform bounded binary search ± a small distance along the normal for N iterations
//!    to refine the intersection point
//! 3. For confusing/ambiguous results, keep the vertex at its original position
//! 4. If good refinement result, proceed to refine normal with additional bisections
//! - Output: Final vertex positions and smoothed normals
//!
//! ## Winding Consistency & Ambiguous Cases
//!
//! We use a lookup table approach similar to Marching Cubes, indexed by the 8-bit CornerMask.
//! This gives us 256 possible configurations.
//!
//! **Winding Rule**: Triangles are wound so that the normal points from inside→outside
//! (i.e., from the "inside" corners toward the "outside" corners). We use the right-hand
//! rule: vertices ordered counter-clockwise when viewed from outside.
//!
//! **Ambiguous Cases**: Some CornerMask values (e.g., 0x3C, 0x69, 0x96, 0xC3) have
//! topologically ambiguous configurations where the surface could be connected in
//! multiple valid ways. Our resolution strategy:
//! 1. Use a fixed, deterministic choice for each ambiguous case in the lookup table
//! 2. Prioritize configurations that avoid creating holes or self-intersections
//! 3. The same choice is made consistently across all cells with the same CornerMask
//!
//! This ensures global consistency: adjacent cells with compatible corner states will
//! always produce compatible triangle configurations.
//!
//! ## Integer Size Analysis
//!
//! **CuboidId coordinates (i32)**:
//! - At max_depth, resolution = base_resolution * 2^max_depth
//! - With base_resolution=8, max_depth=20: 8 * 2^20 = 8,388,608 cells per axis
//! - i32 range: ±2,147,483,647 → sufficient for ~256x this resolution
//! - **Verdict**: i32 is adequate for any practical use case
//!
//! **EdgeId coordinates (i32)**:
//! - Same analysis as CuboidId - edges are at finest-level resolution
//! - **Verdict**: i32 is adequate
//!
//! **Monotonic vertex index (u32)**:
//! - Maximum vertices ≈ number of transitioning edges
//! - Worst case: every cell at max depth has ~12 edges, but most are shared
//! - Realistic estimate: ~1-2 vertices per surface cell
//! - With 8M³ cells, surface area scales as N², so ~64 trillion surface cells max
//! - This exceeds u32! However, practical meshes are much smaller.
//! - **Verdict**: u32 sufficient for meshes up to ~4 billion vertices.
//!   For extreme cases, could upgrade to u64 indices.
//!
//! **Triangle indices in output (u32)**:
//! - Standard GPU vertex buffer index type
//! - **Verdict**: u32 is the practical choice; matches GPU expectations
//!
//! ## Data Structures

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Configuration for the adaptive surface nets algorithm.
#[derive(Clone, Debug)]
pub struct AdaptiveMeshConfig2 {
    /// Base grid resolution for initial discovery (e.g., 8 means 8³ coarse cells)
    pub base_resolution: usize,
    
    /// Maximum refinement depth (total resolution = base_resolution * 2^max_depth)
    pub max_depth: usize,
    
    /// Number of binary search iterations for vertex position refinement
    pub vertex_refinement_iterations: usize,
    
    /// Number of samples for normal estimation per vertex
    pub normal_sample_iterations: usize,
    
    /// Epsilon for normal sampling (fraction of cell size)
    pub normal_epsilon_frac: f32,
    
    /// Number of worker threads (0 = use available parallelism)
    pub num_threads: usize,
}

impl Default for AdaptiveMeshConfig2 {
    fn default() -> Self {
        Self {
            base_resolution: 8,
            max_depth: 4,
            vertex_refinement_iterations: 8,
            normal_sample_iterations: 6,
            normal_epsilon_frac: 0.1,
            num_threads: 0,
        }
    }
}

/// A unique identifier for a cuboid at a specific position and depth.
/// 
/// Uses explicit (x, y, z, depth) representation for clarity and debuggability.
/// Morton encoding could be added later if cache performance becomes an issue.
/// 
/// **CRITICAL**: All coordinates are INTEGER cell indices at the given depth level.
/// Never compute CuboidId from floating-point positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CuboidId {
    /// X coordinate (in units at this depth level)
    pub x: i32,
    /// Y coordinate (in units at this depth level)
    pub y: i32,
    /// Z coordinate (in units at this depth level)
    pub z: i32,
    /// Depth level (0 = coarsest, max_depth = finest)
    pub depth: u8,
}

impl CuboidId {
    /// Create a new CuboidId
    pub fn new(x: i32, y: i32, z: i32, depth: u8) -> Self {
        Self { x, y, z, depth }
    }
    
    /// Convert this cell's coordinates to finest-level units.
    /// At depth d, each cell spans 2^(max_depth - d) finest-level units.
    #[inline]
    pub fn to_finest_level(&self, max_depth: u8) -> (i32, i32, i32) {
        let scale = 1i32 << (max_depth - self.depth);
        (self.x * scale, self.y * scale, self.z * scale)
    }
    
    /// Get the 8 child CuboidIds when subdividing this cell.
    /// Returns children in canonical corner order (see CORNER_OFFSETS).
    pub fn subdivide(&self) -> [CuboidId; 8] {
        let child_depth = self.depth + 1;
        let bx = self.x * 2;
        let by = self.y * 2;
        let bz = self.z * 2;
        [
            CuboidId::new(bx,     by,     bz,     child_depth), // corner 0
            CuboidId::new(bx + 1, by,     bz,     child_depth), // corner 1
            CuboidId::new(bx,     by + 1, bz,     child_depth), // corner 2
            CuboidId::new(bx + 1, by + 1, bz,     child_depth), // corner 3
            CuboidId::new(bx,     by,     bz + 1, child_depth), // corner 4
            CuboidId::new(bx + 1, by,     bz + 1, child_depth), // corner 5
            CuboidId::new(bx,     by + 1, bz + 1, child_depth), // corner 6
            CuboidId::new(bx + 1, by + 1, bz + 1, child_depth), // corner 7
        ]
    }
    
    /// Get a neighbor cell at the same depth level.
    /// Returns None if the neighbor would be outside the valid range.
    pub fn neighbor(&self, dx: i32, dy: i32, dz: i32, max_cells: i32) -> Option<CuboidId> {
        let nx = self.x + dx;
        let ny = self.y + dy;
        let nz = self.z + dz;
        if nx >= 0 && nx < max_cells && ny >= 0 && ny < max_cells && nz >= 0 && nz < max_cells {
            Some(CuboidId::new(nx, ny, nz, self.depth))
        } else {
            None
        }
    }
    
    /// Compute the EdgeId for a given edge index (0-11) of this cell.
    /// The EdgeId is computed in finest-level integer coordinates.
    /// 
    /// **CRITICAL**: This is the ONLY correct way to compute EdgeIds.
    /// Never compute EdgeIds from floating-point positions.
    pub fn edge_id(&self, edge_index: usize, max_depth: u8) -> EdgeId {
        let (_, _, axis, dx, dy, dz) = EDGE_TABLE[edge_index];
        let (fx, fy, fz) = self.to_finest_level(max_depth);
        EdgeId {
            x: fx + dx,
            y: fy + dy,
            z: fz + dz,
            axis,
        }
    }
}

/// A unique identifier for an edge in the voxel grid.
/// Edges are identified by their minimum corner and axis direction.
/// 
/// **CRITICAL**: All coordinates are INTEGER positions in finest-level units.
/// EdgeIds must be computed via CuboidId::edge_id(), never from floats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId {
    /// X coordinate of minimum corner (in finest-level units)
    pub x: i32,
    /// Y coordinate of minimum corner
    pub y: i32,
    /// Z coordinate of minimum corner
    pub z: i32,
    /// Axis: 0=X, 1=Y, 2=Z
    pub axis: u8,
}

impl EdgeId {
    /// Compute the world-space position of this edge's midpoint.
    /// Used for initial vertex placement before refinement.
    pub fn midpoint_world_pos(
        &self,
        bounds_min: (f64, f64, f64),
        cell_size: f64,
    ) -> (f64, f64, f64) {
        // Edge midpoint is at (x + 0.5, y, z) for X-axis edge, etc.
        let (mx, my, mz) = match self.axis {
            0 => (self.x as f64 + 0.5, self.y as f64, self.z as f64),
            1 => (self.x as f64, self.y as f64 + 0.5, self.z as f64),
            2 => (self.x as f64, self.y as f64, self.z as f64 + 0.5),
            _ => unreachable!(),
        };
        (
            bounds_min.0 + mx * cell_size,
            bounds_min.1 + my * cell_size,
            bounds_min.2 + mz * cell_size,
        )
    }
}

/// Corner sample states for a cuboid (8 corners, packed into u8)
#[derive(Clone, Copy, Debug)]
pub struct CornerMask(pub u8);

impl CornerMask {
    /// Check if corner at index (0-7) is inside
    #[inline]
    pub fn is_inside(&self, index: usize) -> bool {
        (self.0 >> index) & 1 != 0
    }
    
    /// Check if this cuboid has mixed corners (some inside, some outside)
    #[inline]
    pub fn is_mixed(&self) -> bool {
        self.0 != 0 && self.0 != 0xFF
    }
    
    /// Create from 8 boolean values
    pub fn from_bools(corners: [bool; 8]) -> Self {
        let mut mask = 0u8;
        for (i, &inside) in corners.iter().enumerate() {
            if inside {
                mask |= 1 << i;
            }
        }
        Self(mask)
    }
}

// =============================================================================
// CANONICAL LOOKUP TABLES
// =============================================================================

/// Corner offset table: CORNER_OFFSETS[i] = (dx, dy, dz) for corner index i.
/// Uses the canonical bit encoding: index = (z << 2) | (y << 1) | x
pub const CORNER_OFFSETS: [(i32, i32, i32); 8] = [
    (0, 0, 0), // 0: ---
    (1, 0, 0), // 1: +--
    (0, 1, 0), // 2: -+-
    (1, 1, 0), // 3: ++-
    (0, 0, 1), // 4: --+
    (1, 0, 1), // 5: +-+
    (0, 1, 1), // 6: -++
    (1, 1, 1), // 7: +++
];

/// Edge definition table: EDGE_TABLE[i] = (corner_a, corner_b, axis, dx, dy, dz)
/// - corner_a, corner_b: the two corner indices this edge connects
/// - axis: 0=X, 1=Y, 2=Z
/// - (dx, dy, dz): offset from cell origin to edge's minimum corner (in cell units)
pub const EDGE_TABLE: [(usize, usize, u8, i32, i32, i32); 12] = [
    // X-axis edges (axis=0)
    (0, 1, 0, 0, 0, 0), // edge 0: corners 0-1
    (2, 3, 0, 0, 1, 0), // edge 1: corners 2-3
    (4, 5, 0, 0, 0, 1), // edge 2: corners 4-5
    (6, 7, 0, 0, 1, 1), // edge 3: corners 6-7
    // Y-axis edges (axis=1)
    (0, 2, 1, 0, 0, 0), // edge 4: corners 0-2
    (1, 3, 1, 1, 0, 0), // edge 5: corners 1-3
    (4, 6, 1, 0, 0, 1), // edge 6: corners 4-6
    (5, 7, 1, 1, 0, 1), // edge 7: corners 5-7
    // Z-axis edges (axis=2)
    (0, 4, 2, 0, 0, 0), // edge 8: corners 0-4
    (1, 5, 2, 1, 0, 0), // edge 9: corners 1-5
    (2, 6, 2, 0, 1, 0), // edge 10: corners 2-6
    (3, 7, 2, 1, 1, 0), // edge 11: corners 3-7
];

// =============================================================================
// WORK QUEUE STRUCTURES
// =============================================================================

/// A work queue entry containing a cuboid ID and pre-sampled corner states.
/// 
/// This replaces the sample cache approach. Each work queue entry carries
/// the corner samples that are already known, avoiding redundant sampling
/// and eliminating cache synchronization complexity.
#[derive(Clone, Copy, Debug)]
pub struct WorkQueueEntry {
    /// The cuboid to process
    pub cuboid: CuboidId,
    /// Pre-sampled corner states: Some(true) = inside, Some(false) = outside, None = unknown
    /// Uses canonical corner indexing (see CORNER_OFFSETS)
    pub known_corners: [Option<bool>; 8],
}

impl WorkQueueEntry {
    /// Create a new work queue entry with no known corners
    pub fn new(cuboid: CuboidId) -> Self {
        Self {
            cuboid,
            known_corners: [None; 8],
        }
    }
    
    /// Create a new work queue entry with some known corners
    pub fn with_corners(cuboid: CuboidId, known_corners: [Option<bool>; 8]) -> Self {
        Self { cuboid, known_corners }
    }
    
    /// Check if all corners are known
    pub fn all_corners_known(&self) -> bool {
        self.known_corners.iter().all(|c| c.is_some())
    }
    
    /// Convert known corners to a CornerMask (panics if not all corners known)
    pub fn to_corner_mask(&self) -> CornerMask {
        let mut mask = 0u8;
        for (i, corner) in self.known_corners.iter().enumerate() {
            if corner.expect("all corners must be known") {
                mask |= 1 << i;
            }
        }
        CornerMask(mask)
    }
}

/// A triangle stored with sparse EdgeId-based vertex references.
/// Used during Stage 2 before index rewriting in Stage 3.
#[derive(Clone, Copy, Debug)]
pub struct SparseTriangle {
    /// The three vertex references as EdgeIds
    /// Winding order is counter-clockwise when viewed from outside (normal pointing out)
    pub vertices: [EdgeId; 3],
}

impl SparseTriangle {
    pub fn new(v0: EdgeId, v1: EdgeId, v2: EdgeId) -> Self {
        Self { vertices: [v0, v1, v2] }
    }
}

/// Statistics for monitoring algorithm performance
#[derive(Default)]
pub struct SamplingStats {
    pub total_samples: AtomicU64,
    pub cuboids_processed: AtomicU64,
    pub triangles_emitted: AtomicU64,
}

/// The indexed mesh output format
#[derive(Clone, Debug)]
pub struct IndexedMesh2 {
    /// Vertex positions
    pub vertices: Vec<(f32, f32, f32)>,
    /// Per-vertex normals
    pub normals: Vec<(f32, f32, f32)>,
    /// Triangle indices (groups of 3)
    pub indices: Vec<u32>,
}

// =============================================================================
// DESIGN NOTES
// =============================================================================
//
// ## Key Design Decisions (Resolved)
//
// 1. **CuboidId representation**: Use (x, y, z, depth) tuples for clarity and
//    debuggability. Morton codes could be added later if cache performance
//    becomes an issue.
//
// 2. **Sample propagation via WorkQueueEntry**: Instead of a sample cache,
//    each work queue entry carries [Option<bool>; 8] with known corner samples.
//    This is simpler, avoids cache synchronization, and naturally propagates
//    samples during subdivision and neighbor expansion.
//
// 3. **Triangle emission timing**: Triangles are generated during the parallel
//    traversal step but stored with sparse edge IDs. Index rewriting happens
//    in Stage 3 after all triangles are collected.
//
// 4. **Boundary handling**: Open surfaces are allowed. Frontier expansion
//    simply stops at bounding box edges.
//
// 5. **Integer-only topology**: ALL position-derived IDs (CuboidId, EdgeId)
//    are computed from integer cell coordinates, NEVER from floating-point.
//    This ensures deterministic vertex welding across adjacent cells.
//
// 6. **Canonical indexing**: Corner and edge indices follow a fixed canonical
//    system (see CORNER_OFFSETS and EDGE_TABLE) used consistently everywhere.
//
// ## Implementation Considerations
//
// - **Winding consistency**: Use lookup tables indexed by CornerMask to
//   determine correct triangle configurations (similar to Marching Cubes).
//   Ambiguous cases use a fixed deterministic choice per CornerMask value.
//
// - **Work queue deduplication**: Use a concurrent set (dashmap) keyed by
//   CuboidId to prevent processing the same cell multiple times.
//
// - **Corner propagation during subdivision**: When subdividing a cell into
//   8 children, the parent's 8 corners become corners of the children.
//   New samples are only needed for the 19 new points (face centers, edge
//   midpoints, and cell center).
//
// - **Corner propagation to neighbors**: When adding a neighbor to the queue,
//   propagate the 4 shared corner samples from the current cell.
//
// - **Thin feature detection**: Consider optional feature detection pass
//   that samples along edges of coarse cells even if corners are all same state.

// =============================================================================
// PLACEHOLDER IMPLEMENTATIONS (to be filled in)
// =============================================================================

/// Main entry point for adaptive surface nets meshing.
/// 
/// # Arguments
/// * `sampler` - Function that returns true if point is inside the model
/// * `bounds_min` - Minimum corner of bounding box
/// * `bounds_max` - Maximum corner of bounding box  
/// * `config` - Algorithm configuration
///
/// # Returns
/// An indexed mesh with vertices, normals, and triangle indices
pub fn adaptive_surface_nets_2<F>(
    _sampler: F,
    _bounds_min: (f32, f32, f32),
    _bounds_max: (f32, f32, f32),
    _config: &AdaptiveMeshConfig2,
) -> IndexedMesh2
where
    F: Fn(f64, f64, f64) -> f32 + Send + Sync,
{
    // TODO: Implement the algorithm
    IndexedMesh2 {
        vertices: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_corner_mask() {
        let mask = CornerMask::from_bools([true, false, true, false, false, false, false, false]);
        assert!(mask.is_inside(0));
        assert!(!mask.is_inside(1));
        assert!(mask.is_inside(2));
        assert!(mask.is_mixed());
        
        let all_inside = CornerMask(0xFF);
        assert!(!all_inside.is_mixed());
        
        let all_outside = CornerMask(0x00);
        assert!(!all_outside.is_mixed());
    }
    
    #[test]
    fn test_config_default() {
        let config = AdaptiveMeshConfig2::default();
        assert_eq!(config.base_resolution, 8);
        assert_eq!(config.max_depth, 4);
    }
}
