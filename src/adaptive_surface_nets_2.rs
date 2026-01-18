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
//! **Winding Rule**: Triangles use CCW winding when viewed from outside the surface,
//! matching the project-wide convention documented in README.md. The face normal
//! (computed as cross(AB, AC)) points outward from the solid. This is achieved by
//! reversing the vertex order from the standard MC tables.
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

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use dashmap::DashSet;
use rayon::prelude::*;

/// Configuration for the adaptive surface nets algorithm.
#[derive(Clone, Debug)]
pub struct AdaptiveMeshConfig2 {
    /// Base grid resolution for initial discovery (e.g., 8 means 8³ coarse cells)
    pub base_resolution: usize,

    /// Maximum refinement depth (total resolution = base_resolution * 2^max_depth)
    pub max_depth: usize,

    /// Number of binary search iterations for vertex position refinement
    pub vertex_refinement_iterations: usize,

    /// Number of binary search iterations for normal refinement probing.
    ///
    /// When set to 0, accumulated face normals are used directly (fast, good for
    /// most cases).
    ///
    /// When set to > 0, normals are refined by probing the surface in tangent
    /// directions and fitting a plane to the discovered surface points. This
    /// provides smoother normals, especially on curved surfaces. Each probe uses
    /// this many binary search iterations to find the surface crossing.
    ///
    /// Typical values: 0 (disabled), 4-8 (good quality).
    /// Higher values give more precise surface point locations but cost more samples.
    pub normal_sample_iterations: usize,

    /// Probe distance for normal refinement (fraction of cell size).
    ///
    /// Controls how far apart the tangent probes are spaced relative to the
    /// finest-level cell size. Larger values capture more surface curvature.
    ///
    /// Typical values: 0.1 - 0.5
    pub normal_epsilon_frac: f32,

    /// Number of worker threads (0 = use available parallelism)
    pub num_threads: usize,
}

impl Default for AdaptiveMeshConfig2 {
    fn default() -> Self {
        Self {
            base_resolution: 8,
            max_depth: 4,
            vertex_refinement_iterations: 12,
            // Enable normal refinement by default - probing works well with binary samplers
            normal_sample_iterations: 12,
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
    ///
    /// # Arguments
    /// * `dx, dy, dz` - Direction offset (-1, 0, or 1)
    /// * `max_cells` - Number of cells at this depth level (original grid)
    /// * `boundary_expansion` - Number of extra cells to allow beyond the grid on each side
    ///   (e.g., 1 means allow cells from -1 to max_cells inclusive)
    pub fn neighbor(&self, dx: i32, dy: i32, dz: i32, max_cells: i32, boundary_expansion: i32) -> Option<CuboidId> {
        let nx = self.x + dx;
        let ny = self.y + dy;
        let nz = self.z + dz;
        let min_valid = -boundary_expansion;
        let max_valid = max_cells + boundary_expansion;
        if nx >= min_valid && nx < max_valid && ny >= min_valid && ny < max_valid && nz >= min_valid && nz < max_valid {
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
        cell_size: (f64, f64, f64),
    ) -> (f64, f64, f64) {
        // Edge midpoint is at (x + 0.5, y, z) for X-axis edge, etc.
        let (mx, my, mz) = match self.axis {
            0 => (self.x as f64 + 0.5, self.y as f64, self.z as f64),
            1 => (self.x as f64, self.y as f64 + 0.5, self.z as f64),
            2 => (self.x as f64, self.y as f64, self.z as f64 + 0.5),
            _ => unreachable!(),
        };
        (
            bounds_min.0 + mx * cell_size.0,
            bounds_min.1 + my * cell_size.1,
            bounds_min.2 + mz * cell_size.2,
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
// MARCHING CUBES CONVENTION TRANSLATION
// =============================================================================
//
// Our corner numbering uses: index = (z << 2) | (y << 1) | x
//   0:(0,0,0) 1:(1,0,0) 2:(0,1,0) 3:(1,1,0) 4:(0,0,1) 5:(1,0,1) 6:(0,1,1) 7:(1,1,1)
//
// Standard MC corner numbering (Paul Bourke):
//   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
//
// Mapping: MC corner N corresponds to our corner:
//   MC 0 -> Our 0, MC 1 -> Our 1, MC 2 -> Our 3, MC 3 -> Our 2
//   MC 4 -> Our 4, MC 5 -> Our 5, MC 6 -> Our 7, MC 7 -> Our 6
//
// Similarly for edges (MC edge -> Our edge):
//   0->0, 1->5, 2->1, 3->4, 4->2, 5->7, 6->3, 7->6, 8->8, 9->9, 10->11, 11->10

/// Convert our corner mask to standard MC corner mask for table lookup.
/// Our bits: 7 6 5 4 3 2 1 0 (our corners 7,6,5,4,3,2,1,0)
/// MC bits:  7 6 5 4 3 2 1 0 (MC corners 7,6,5,4,3,2,1,0)
///
/// Mapping: MC bit N should contain our corner that maps to MC corner N
///   MC bit 0 <- our bit 0 (our corner 0 -> MC corner 0)
///   MC bit 1 <- our bit 1 (our corner 1 -> MC corner 1)
///   MC bit 2 <- our bit 3 (our corner 3 -> MC corner 2)
///   MC bit 3 <- our bit 2 (our corner 2 -> MC corner 3)
///   MC bit 4 <- our bit 4 (our corner 4 -> MC corner 4)
///   MC bit 5 <- our bit 5 (our corner 5 -> MC corner 5)
///   MC bit 6 <- our bit 7 (our corner 7 -> MC corner 6)
///   MC bit 7 <- our bit 6 (our corner 6 -> MC corner 7)
#[inline]
fn our_mask_to_mc_mask(our_mask: u8) -> u8 {
    (our_mask & 0b00000011) |           // bits 0,1 stay
    ((our_mask & 0b00000100) << 1) |    // our bit 2 -> MC bit 3
    ((our_mask & 0b00001000) >> 1) |    // our bit 3 -> MC bit 2
    (our_mask & 0b00110000) |           // bits 4,5 stay
    ((our_mask & 0b01000000) << 1) |    // our bit 6 -> MC bit 7
    ((our_mask & 0b10000000) >> 1)      // our bit 7 -> MC bit 6
}

/// Convert MC edge index to our edge index.
/// Standard MC edges connect MC corners, we need to map to our edge numbering.
const MC_EDGE_TO_OUR_EDGE: [usize; 12] = [
    0,  // MC edge 0 (MC corners 0-1) -> our edge 0 (our corners 0-1)
    5,  // MC edge 1 (MC corners 1-2) -> our edge 5 (our corners 1-3)
    1,  // MC edge 2 (MC corners 2-3) -> our edge 1 (our corners 2-3, but MC 2=our 3, MC 3=our 2)
    4,  // MC edge 3 (MC corners 3-0) -> our edge 4 (our corners 0-2)
    2,  // MC edge 4 (MC corners 4-5) -> our edge 2 (our corners 4-5)
    7,  // MC edge 5 (MC corners 5-6) -> our edge 7 (our corners 5-7)
    3,  // MC edge 6 (MC corners 6-7) -> our edge 3 (our corners 6-7, but MC 6=our 7, MC 7=our 6)
    6,  // MC edge 7 (MC corners 7-4) -> our edge 6 (our corners 4-6)
    8,  // MC edge 8 (MC corners 0-4) -> our edge 8 (our corners 0-4)
    9,  // MC edge 9 (MC corners 1-5) -> our edge 9 (our corners 1-5)
    11, // MC edge 10 (MC corners 2-6) -> our edge 11 (MC 2=our 3, MC 6=our 7 -> corners 3-7)
    10, // MC edge 11 (MC corners 3-7) -> our edge 10 (MC 3=our 2, MC 7=our 6 -> corners 2-6)
];

/// Marching Cubes edge flags: MC_EDGE_FLAGS[corner_mask] = bitmask of active edges
/// Bit N is set if edge N has a sign change (crosses the surface).
/// This uses the standard Marching Cubes table adapted to our corner indexing.
pub const MC_EDGE_FLAGS: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// Marching Cubes triangle table: MC_TRI_TABLE[corner_mask] = list of edge indices forming triangles
/// Each entry is a slice of edge indices, grouped in threes (each triple forms one triangle).
/// -1 marks the end of the list. Maximum 5 triangles (15 edges) per configuration.
/// Winding is set so normals point from inside (1) to outside (0).
pub const MC_TRI_TABLE: [[i8; 16]; 256] = include!("mc_tri_table.inc");

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

/// Statistics for monitoring algorithm performance (atomic counters for concurrent use)
#[derive(Default)]
pub struct SamplingStats {
    pub total_samples: AtomicU64,
    pub cuboids_processed: AtomicU64,
    pub triangles_emitted: AtomicU64,

    // Vertex refinement diagnostics
    /// Vertices refined using primary direction (accumulated normal)
    pub refine_primary_hit: AtomicU64,
    /// Vertices refined using fallback X direction
    pub refine_fallback_x_hit: AtomicU64,
    /// Vertices refined using fallback Y direction
    pub refine_fallback_y_hit: AtomicU64,
    /// Vertices refined using fallback Z direction
    pub refine_fallback_z_hit: AtomicU64,
    /// Vertices where no direction found a crossing (kept original position)
    pub refine_miss: AtomicU64,
}

/// Per-stage timing and statistics for profiling
#[derive(Clone, Debug, Default)]
pub struct MeshingStats2 {
    /// Total time for the entire meshing operation (seconds)
    pub total_time_secs: f64,

    /// Stage 1: Coarse grid discovery
    pub stage1_time_secs: f64,
    pub stage1_samples: u64,
    pub stage1_mixed_cells: usize,

    /// Stage 2: Subdivision & triangle emission
    pub stage2_time_secs: f64,
    pub stage2_samples: u64,
    pub stage2_cuboids_processed: u64,
    pub stage2_triangles_emitted: u64,

    /// Stage 3: Topology finalization
    pub stage3_time_secs: f64,
    pub stage3_unique_vertices: usize,

    /// Stage 4: Vertex refinement & normal estimation
    pub stage4_time_secs: f64,
    pub stage4_samples: u64,

    /// Vertex refinement diagnostics
    pub stage4_refine_primary_hit: u64,
    pub stage4_refine_fallback_x_hit: u64,
    pub stage4_refine_fallback_y_hit: u64,
    pub stage4_refine_fallback_z_hit: u64,
    pub stage4_refine_miss: u64,

    /// Summary statistics
    pub total_samples: u64,
    pub total_vertices: usize,
    pub total_triangles: usize,

    /// Configuration used
    pub effective_resolution: usize,
}

// =============================================================================
// NORMAL DIAGNOSTIC TYPES (only available with "normal-diagnostic" feature)
// =============================================================================

/// A single entry in the normal diagnostic results.
/// Compares normals computed with a specific iteration count against high-precision reference.
#[cfg(feature = "normal-diagnostic")]
#[derive(Clone, Debug, Default)]
pub struct NormalDiagnosticEntry {
    /// Number of binary search iterations used (0 = topology-only normals)
    pub iterations: usize,
    /// Mean angular error in degrees
    pub mean_error_degrees: f64,
    /// Median (P50) angular error in degrees
    pub p50_error_degrees: f64,
    /// 95th percentile angular error in degrees
    pub p95_error_degrees: f64,
    /// Maximum angular error in degrees
    pub max_error_degrees: f64,
    /// Number of extra samples used (beyond topology-only)
    pub extra_samples: u64,
}

impl MeshingStats2 {
    /// Print a human-readable profiling report to stdout
    pub fn print_report(&self) {
        println!("=== Adaptive Surface Nets v2 Profiling Report ===");
        println!("Total time: {:.2}ms", self.total_time_secs * 1000.0);
        println!();
        println!("Stage 1 (Coarse Discovery): {:.2}ms ({:.1}%)",
            self.stage1_time_secs * 1000.0,
            self.stage1_time_secs / self.total_time_secs * 100.0);
        println!("  Samples: {}", self.stage1_samples);
        println!("  Mixed cells found: {}", self.stage1_mixed_cells);
        println!();
        println!("Stage 2 (Subdivision & Emission): {:.2}ms ({:.1}%)",
            self.stage2_time_secs * 1000.0,
            self.stage2_time_secs / self.total_time_secs * 100.0);
        println!("  Samples: {}", self.stage2_samples);
        println!("  Cuboids processed: {}", self.stage2_cuboids_processed);
        println!("  Triangles emitted: {}", self.stage2_triangles_emitted);
        println!();
        println!("Stage 3 (Topology Finalization): {:.2}ms ({:.1}%)",
            self.stage3_time_secs * 1000.0,
            self.stage3_time_secs / self.total_time_secs * 100.0);
        println!("  Unique vertices: {}", self.stage3_unique_vertices);
        println!();
        println!("Stage 4 (Vertex Refinement): {:.2}ms ({:.1}%)",
            self.stage4_time_secs * 1000.0,
            self.stage4_time_secs / self.total_time_secs * 100.0);
        println!("  Samples: {}", self.stage4_samples);
        let total_refined = self.stage4_refine_primary_hit
            + self.stage4_refine_fallback_x_hit
            + self.stage4_refine_fallback_y_hit
            + self.stage4_refine_fallback_z_hit
            + self.stage4_refine_miss;
        if total_refined > 0 {
            println!("  Refinement outcomes:");
            println!("    Primary hit:    {:>8} ({:>5.1}%)",
                self.stage4_refine_primary_hit,
                self.stage4_refine_primary_hit as f64 / total_refined as f64 * 100.0);
            println!("    Fallback X hit: {:>8} ({:>5.1}%)",
                self.stage4_refine_fallback_x_hit,
                self.stage4_refine_fallback_x_hit as f64 / total_refined as f64 * 100.0);
            println!("    Fallback Y hit: {:>8} ({:>5.1}%)",
                self.stage4_refine_fallback_y_hit,
                self.stage4_refine_fallback_y_hit as f64 / total_refined as f64 * 100.0);
            println!("    Fallback Z hit: {:>8} ({:>5.1}%)",
                self.stage4_refine_fallback_z_hit,
                self.stage4_refine_fallback_z_hit as f64 / total_refined as f64 * 100.0);
            println!("    MISS (no crossing): {:>8} ({:>5.1}%)",
                self.stage4_refine_miss,
                self.stage4_refine_miss as f64 / total_refined as f64 * 100.0);
        }
        println!();
        println!("Summary:");
        println!("  Total samples: {}", self.total_samples);
        println!("  Avg sample time: {:.2}µs",
            self.total_time_secs * 1_000_000.0 / self.total_samples as f64);
        println!("  Total vertices: {}", self.total_vertices);
        println!("  Total triangles: {}", self.total_triangles);
        println!("  Effective resolution: {}³", self.effective_resolution);
        println!("================================================");
    }
}

/// Result of the adaptive surface nets meshing operation
#[derive(Clone, Debug)]
pub struct MeshingResult2 {
    /// The generated indexed mesh
    pub mesh: IndexedMesh2,
    /// Profiling statistics
    pub stats: MeshingStats2,
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
// STAGE 1: COARSE GRID DISCOVERY
// =============================================================================

/// Sample a single point and return whether it's inside (density > 0).
#[inline]
fn sample_is_inside<F>(sampler: &F, x: f64, y: f64, z: f64, stats: &SamplingStats) -> bool
where
    F: Fn(f64, f64, f64) -> f32,
{
    stats.total_samples.fetch_add(1, Ordering::Relaxed);
    sampler(x, y, z) > 0.0
}

/// Stage 1: Coarse Grid Discovery
///
/// Samples the volume at low resolution (`base_resolution³` cells) to find
/// regions containing the surface. Returns a work queue of mixed cells with
/// their corner samples pre-filled.
///
/// # Algorithm
/// 1. Create a grid of (base_resolution + 1)³ sample points (corners)
/// 2. Sample all corner points to determine inside/outside state
/// 3. For each cell, check if corners have mixed states (surface crosses cell)
/// 4. Return WorkQueueEntry for each mixed cell with all 8 corners known
fn stage1_coarse_discovery<F>(
    sampler: &F,
    bounds_min: (f64, f64, f64),
    bounds_max: (f64, f64, f64),
    config: &AdaptiveMeshConfig2,
    stats: &SamplingStats,
) -> Vec<WorkQueueEntry>
where
    F: Fn(f64, f64, f64) -> f32 + Send + Sync,
{
    let res = config.base_resolution;
    let num_corners = res + 1;

    // Cell size at depth 0 (coarsest level)
    let cell_size = (
        (bounds_max.0 - bounds_min.0) / res as f64,
        (bounds_max.1 - bounds_min.1) / res as f64,
        (bounds_max.2 - bounds_min.2) / res as f64,
    );

    // Sample all corner points into a 3D array
    // Layout: corners[z][y][x] for cache-friendly Z-slice iteration
    let mut corners = vec![vec![vec![false; num_corners]; num_corners]; num_corners];

    for iz in 0..num_corners {
        for iy in 0..num_corners {
            for ix in 0..num_corners {
                let x = bounds_min.0 + ix as f64 * cell_size.0;
                let y = bounds_min.1 + iy as f64 * cell_size.1;
                let z = bounds_min.2 + iz as f64 * cell_size.2;
                corners[iz][iy][ix] = sample_is_inside(sampler, x, y, z, stats);
            }
        }
    }

    // Find mixed cells and create work queue entries
    let mut work_queue = Vec::new();

    for iz in 0..res {
        for iy in 0..res {
            for ix in 0..res {
                // Gather the 8 corner samples for this cell using canonical corner ordering
                let cell_corners: [bool; 8] = [
                    corners[iz][iy][ix],         // corner 0: (0,0,0)
                    corners[iz][iy][ix + 1],     // corner 1: (1,0,0)
                    corners[iz][iy + 1][ix],     // corner 2: (0,1,0)
                    corners[iz][iy + 1][ix + 1], // corner 3: (1,1,0)
                    corners[iz + 1][iy][ix],     // corner 4: (0,0,1)
                    corners[iz + 1][iy][ix + 1], // corner 5: (1,0,1)
                    corners[iz + 1][iy + 1][ix], // corner 6: (0,1,1)
                    corners[iz + 1][iy + 1][ix + 1], // corner 7: (1,1,1)
                ];

                let mask = CornerMask::from_bools(cell_corners);

                if mask.is_mixed() {
                    let cuboid = CuboidId::new(ix as i32, iy as i32, iz as i32, 0);
                    let known_corners = cell_corners.map(Some);
                    work_queue.push(WorkQueueEntry::with_corners(cuboid, known_corners));
                }
            }
        }
    }

    work_queue
}

// =============================================================================
// STAGE 2: PARALLEL SUBDIVISION & TRIANGLE EMISSION
// =============================================================================

/// Compute the world-space position of a cell corner.
///
/// # Arguments
/// * `cell` - The cell ID
/// * `corner_index` - Corner index (0-7)
/// * `bounds_min` - World-space minimum bounds
/// * `cell_size` - Size of a cell at the finest level (per-axis)
/// * `max_depth` - Maximum refinement depth
#[inline]
fn corner_world_position(
    cell: &CuboidId,
    corner_index: usize,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
    max_depth: u8,
) -> (f64, f64, f64) {
    let (fx, fy, fz) = cell.to_finest_level(max_depth);
    let (dx, dy, dz) = CORNER_OFFSETS[corner_index];

    // Scale factor: how many finest-level cells does this cell span?
    let scale = 1i32 << (max_depth - cell.depth);

    (
        bounds_min.0 + (fx + dx * scale) as f64 * cell_size.0,
        bounds_min.1 + (fy + dy * scale) as f64 * cell_size.1,
        bounds_min.2 + (fz + dz * scale) as f64 * cell_size.2,
    )
}

/// Complete corner samples for a work queue entry.
/// Only samples corners that are not already known.
///
/// Returns the completed corner states as a CornerMask.
fn complete_corner_samples<F>(
    entry: &mut WorkQueueEntry,
    sampler: &F,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
    max_depth: u8,
    stats: &SamplingStats,
) -> CornerMask
where
    F: Fn(f64, f64, f64) -> f32,
{
    for corner_idx in 0..8 {
        if entry.known_corners[corner_idx].is_none() {
            let (x, y, z) = corner_world_position(
                &entry.cuboid,
                corner_idx,
                bounds_min,
                cell_size,
                max_depth,
            );
            entry.known_corners[corner_idx] = Some(sample_is_inside(sampler, x, y, z, stats));
        }
    }
    entry.to_corner_mask()
}

/// Subdivide a parent cell, sampling all child corners at once.
///
/// This function samples all 27 positions in the 3x3x3 grid of child corners,
/// reusing the 8 parent corners that are already known. Then it determines
/// which of the 8 children are mixed, and returns only those with all corners
/// fully populated.
///
/// This is more efficient than sampling each child independently because:
/// - Each position is sampled exactly once (siblings share 19 interior points)
/// - Non-mixed children are filtered out before entering the work queue
fn subdivide_and_filter_mixed<F>(
    parent: &WorkQueueEntry,
    sampler: &F,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
    max_depth: u8,
    stats: &SamplingStats,
) -> Vec<WorkQueueEntry>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let children = parent.cuboid.subdivide();

    // The children form a 2x2x2 grid. Their corners form a 3x3x3 grid.
    // samples[iz][iy][ix] where ix, iy, iz ∈ {0, 1, 2}
    //
    // The parent's 8 corners map to the 8 "even" positions:
    // - Parent corner 0 (0,0,0) -> samples[0][0][0]
    // - Parent corner 1 (1,0,0) -> samples[0][0][2]
    // - Parent corner 2 (0,1,0) -> samples[0][2][0]
    // - Parent corner 3 (1,1,0) -> samples[0][2][2]
    // - Parent corner 4 (0,0,1) -> samples[2][0][0]
    // - Parent corner 5 (1,0,1) -> samples[2][0][2]
    // - Parent corner 6 (0,1,1) -> samples[2][2][0]
    // - Parent corner 7 (1,1,1) -> samples[2][2][2]

    let mut samples = [[[false; 3]; 3]; 3];

    // Copy parent corners to the appropriate positions
    // Parent corner index = (z << 2) | (y << 1) | x maps to samples[z*2][y*2][x*2]
    for parent_corner in 0..8 {
        let px = parent_corner & 1;
        let py = (parent_corner >> 1) & 1;
        let pz = (parent_corner >> 2) & 1;
        samples[pz * 2][py * 2][px * 2] = parent.known_corners[parent_corner]
            .expect("Parent corners should all be known at subdivision time");
    }

    // Sample the 19 new positions (the "odd" positions in the 3x3x3 grid)
    // These are: 12 edge midpoints + 6 face centers + 1 cell center
    let (fx, fy, fz) = parent.cuboid.to_finest_level(max_depth);
    let parent_scale = 1i32 << (max_depth - parent.cuboid.depth);
    let child_scale = parent_scale / 2; // Scale for child cells

    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                // Skip even positions (already have parent corners)
                if ix % 2 == 0 && iy % 2 == 0 && iz % 2 == 0 {
                    continue;
                }

                // Position in finest-level units
                let px = fx + ix as i32 * child_scale;
                let py = fy + iy as i32 * child_scale;
                let pz = fz + iz as i32 * child_scale;

                let world_x = bounds_min.0 + px as f64 * cell_size.0;
                let world_y = bounds_min.1 + py as f64 * cell_size.1;
                let world_z = bounds_min.2 + pz as f64 * cell_size.2;

                samples[iz][iy][ix] = sample_is_inside(sampler, world_x, world_y, world_z, stats);
            }
        }
    }

    // Now extract corners for each child and filter for mixed ones
    let mut result = Vec::with_capacity(8);

    for (child_idx, child_cuboid) in children.iter().enumerate() {
        // Child index = (cz << 2) | (cy << 1) | cx where cx, cy, cz ∈ {0, 1}
        let cx = child_idx & 1;
        let cy = (child_idx >> 1) & 1;
        let cz = (child_idx >> 2) & 1;

        // Extract the 8 corners for this child
        let mut child_corners = [Some(false); 8];
        for corner_idx in 0..8 {
            // Corner offset within child
            let dx = corner_idx & 1;
            let dy = (corner_idx >> 1) & 1;
            let dz = (corner_idx >> 2) & 1;

            // Position in the 3x3x3 samples grid
            let sx = cx + dx;
            let sy = cy + dy;
            let sz = cz + dz;

            child_corners[corner_idx] = Some(samples[sz][sy][sx]);
        }

        // Check if this child is mixed
        let mask = CornerMask::from_bools([
            child_corners[0].unwrap(),
            child_corners[1].unwrap(),
            child_corners[2].unwrap(),
            child_corners[3].unwrap(),
            child_corners[4].unwrap(),
            child_corners[5].unwrap(),
            child_corners[6].unwrap(),
            child_corners[7].unwrap(),
        ]);

        if mask.is_mixed() {
            result.push(WorkQueueEntry::with_corners(*child_cuboid, child_corners));
        }
    }

    result
}

/// Neighbor direction to shared corner mapping.
/// When we have a cell and want to add its neighbor in direction (dx, dy, dz),
/// we can propagate 4 corner samples that are shared between them.
///
/// Returns: [(our_corner, neighbor_corner); 4] pairs
fn shared_corners_with_neighbor(dx: i32, dy: i32, dz: i32) -> [(usize, usize); 4] {
    match (dx, dy, dz) {
        // -X neighbor: our corners 0,2,4,6 are their corners 1,3,5,7
        (-1, 0, 0) => [(0, 1), (2, 3), (4, 5), (6, 7)],
        // +X neighbor: our corners 1,3,5,7 are their corners 0,2,4,6
        (1, 0, 0) => [(1, 0), (3, 2), (5, 4), (7, 6)],
        // -Y neighbor: our corners 0,1,4,5 are their corners 2,3,6,7
        (0, -1, 0) => [(0, 2), (1, 3), (4, 6), (5, 7)],
        // +Y neighbor: our corners 2,3,6,7 are their corners 0,1,4,5
        (0, 1, 0) => [(2, 0), (3, 1), (6, 4), (7, 5)],
        // -Z neighbor: our corners 0,1,2,3 are their corners 4,5,6,7
        (0, 0, -1) => [(0, 4), (1, 5), (2, 6), (3, 7)],
        // +Z neighbor: our corners 4,5,6,7 are their corners 0,1,2,3
        (0, 0, 1) => [(4, 0), (5, 1), (6, 2), (7, 3)],
        _ => panic!("Invalid neighbor direction: ({}, {}, {})", dx, dy, dz),
    }
}

/// Create a work queue entry for a neighbor cell, propagating shared corners.
///
/// # Arguments
/// * `current` - The current work queue entry
/// * `dx, dy, dz` - Direction to the neighbor
/// * `max_cells` - Number of cells at this depth level (original grid)
/// * `boundary_expansion` - Number of extra cells to allow beyond the grid on each side
fn create_neighbor_entry(
    current: &WorkQueueEntry,
    dx: i32,
    dy: i32,
    dz: i32,
    max_cells: i32,
    boundary_expansion: i32,
) -> Option<WorkQueueEntry> {
    let neighbor_cuboid = current.cuboid.neighbor(dx, dy, dz, max_cells, boundary_expansion)?;
    let mut neighbor = WorkQueueEntry::new(neighbor_cuboid);

    // Propagate shared corners
    for (our_corner, their_corner) in shared_corners_with_neighbor(dx, dy, dz) {
        neighbor.known_corners[their_corner] = current.known_corners[our_corner];
    }

    Some(neighbor)
}

/// Check if the shared face with a neighbor is mixed (has surface crossing).
///
/// If the 4 shared corners have mixed states (some inside, some outside),
/// the face definitely crosses the surface, and the neighbor is guaranteed
/// to be mixed. If all 4 shared corners are the same state, the face doesn't
/// cross the surface, and we don't need to explore this neighbor from this
/// direction (if it's mixed, another neighbor with a mixed shared face will
/// discover it).
///
/// Returns true if the shared face is mixed (should explore this neighbor).
fn shared_face_is_mixed(current_corners: &[Option<bool>; 8], dx: i32, dy: i32, dz: i32) -> bool {
    let shared = shared_corners_with_neighbor(dx, dy, dz);

    // Get the states of our shared corners
    let mut has_inside = false;
    let mut has_outside = false;

    for (our_corner, _) in shared {
        match current_corners[our_corner] {
            Some(true) => has_inside = true,
            Some(false) => has_outside = true,
            None => {
                // Unknown corner - be conservative and explore
                return true;
            }
        }
        // Early exit if we've found both states
        if has_inside && has_outside {
            return true;
        }
    }

    // Face is mixed only if we have both inside and outside corners
    has_inside && has_outside
}

/// Emit triangles for a cell at maximum depth using the Marching Cubes lookup table.
fn emit_triangles_for_cell(
    cell: &CuboidId,
    corner_mask: CornerMask,
    max_depth: u8,
    triangles: &mut Vec<SparseTriangle>,
    stats: &SamplingStats,
) {
    // Convert our corner mask to MC corner mask for table lookup
    let mc_mask = our_mask_to_mc_mask(corner_mask.0);
    let tri_config = &MC_TRI_TABLE[mc_mask as usize];

    let mut i = 0;
    while i < 16 && tri_config[i] >= 0 {
        // MC edge indices from the table
        let mc_e0 = tri_config[i] as usize;
        let mc_e1 = tri_config[i + 1] as usize;
        let mc_e2 = tri_config[i + 2] as usize;

        // Convert MC edge indices to our edge indices
        let e0 = MC_EDGE_TO_OUR_EDGE[mc_e0];
        let e1 = MC_EDGE_TO_OUR_EDGE[mc_e1];
        let e2 = MC_EDGE_TO_OUR_EDGE[mc_e2];

        let edge_id_0 = cell.edge_id(e0, max_depth);
        let edge_id_1 = cell.edge_id(e1, max_depth);
        let edge_id_2 = cell.edge_id(e2, max_depth);

        // Reverse winding order (e0, e2, e1) instead of (e0, e1, e2) to produce
        // CCW winding when viewed from outside, matching project conventions.
        // The standard MC tables produce CW winding for our "inside = positive" convention.
        triangles.push(SparseTriangle::new(edge_id_0, edge_id_2, edge_id_1));
        stats.triangles_emitted.fetch_add(1, Ordering::Relaxed);

        i += 3;
    }
}

/// Stage 2: Parallel Subdivision & Triangle Emission
///
/// Processes the work queue from Stage 1, recursively subdividing mixed cells
/// until reaching max_depth, then emitting triangles using the MC lookup table.
///
/// # Algorithm
/// 1. Process each work queue entry
/// 2. Complete corner samples (only sample unknown corners)
/// 3. If mixed and depth < max_depth: subdivide into 8 children
/// 4. If mixed and depth == max_depth: emit triangles, expand to neighbors
/// 5. Use deduplication set to avoid processing same cell twice
///
/// # Returns
/// Vector of SparseTriangles with EdgeId-based vertex references
/// Result of processing a single work queue entry
struct ProcessedEntry {
    /// New work items to add to the queue
    new_work: Vec<WorkQueueEntry>,
    /// Triangles emitted (only at max depth)
    triangles: Vec<SparseTriangle>,
}

/// Process a single work queue entry, returning new work items and triangles
fn process_work_entry<F>(
    mut entry: WorkQueueEntry,
    sampler: &F,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
    max_depth: u8,
    base_res: i32,
    visited: &DashSet<CuboidId>,
    stats: &SamplingStats,
) -> ProcessedEntry
where
    F: Fn(f64, f64, f64) -> f32 + Send + Sync,
{
    stats.cuboids_processed.fetch_add(1, Ordering::Relaxed);

    // Complete corner samples
    let corner_mask = complete_corner_samples(
        &mut entry,
        sampler,
        bounds_min,
        cell_size,
        max_depth,
        stats,
    );

    // Skip if not mixed (all inside or all outside)
    if !corner_mask.is_mixed() {
        return ProcessedEntry {
            new_work: Vec::new(),
            triangles: Vec::new(),
        };
    }

    let current_depth = entry.cuboid.depth;

    if current_depth < max_depth {
        // Subdivide and get only mixed children (with all corners sampled)
        let mixed_children = subdivide_and_filter_mixed(
            &entry,
            sampler,
            bounds_min,
            cell_size,
            max_depth,
            stats,
        );

        // Filter to only new (unvisited) children
        let new_work: Vec<WorkQueueEntry> = mixed_children
            .into_iter()
            .filter(|child| visited.insert(child.cuboid))
            .collect();

        ProcessedEntry {
            new_work,
            triangles: Vec::new(),
        }
    } else {
        // At max depth: emit triangles and expand frontier
        let mut triangles = Vec::new();
        emit_triangles_for_cell(
            &entry.cuboid,
            corner_mask,
            max_depth,
            &mut triangles,
            stats,
        );

        // Expand to neighbors (frontier expansion)
        let cells_at_depth = base_res * (1i32 << current_depth);

        const NEIGHBOR_DIRS: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        let mut new_work = Vec::new();
        for (dx, dy, dz) in NEIGHBOR_DIRS {
            if !shared_face_is_mixed(&entry.known_corners, dx, dy, dz) {
                continue;
            }

            if let Some(neighbor) = create_neighbor_entry(&entry, dx, dy, dz, cells_at_depth, 1) {
                if visited.insert(neighbor.cuboid) {
                    new_work.push(neighbor);
                }
            }
        }

        ProcessedEntry { new_work, triangles }
    }
}

fn stage2_subdivision_and_emission<F>(
    initial_queue: Vec<WorkQueueEntry>,
    sampler: &F,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
    config: &AdaptiveMeshConfig2,
    stats: &SamplingStats,
) -> Vec<SparseTriangle>
where
    F: Fn(f64, f64, f64) -> f32 + Send + Sync,
{
    let max_depth = config.max_depth as u8;
    let base_res = config.base_resolution as i32;

    // Deduplication set: tracks CuboidIds we've already processed or queued
    let visited: DashSet<CuboidId> = DashSet::new();

    // Output triangles (thread-safe collection)
    let all_triangles: std::sync::Mutex<Vec<SparseTriangle>> = std::sync::Mutex::new(Vec::new());

    // Current batch of work items
    let mut current_batch: Vec<WorkQueueEntry> = initial_queue;

    // Mark initial entries as visited
    for entry in &current_batch {
        visited.insert(entry.cuboid);
    }

    // Process in parallel batches until no more work
    while !current_batch.is_empty() {
        // Process current batch in parallel
        let results: Vec<ProcessedEntry> = current_batch
            .into_par_iter()
            .map(|entry| {
                process_work_entry(
                    entry,
                    sampler,
                    bounds_min,
                    cell_size,
                    max_depth,
                    base_res,
                    &visited,
                    stats,
                )
            })
            .collect();

        // Collect new work and triangles from results
        let mut next_batch = Vec::new();
        {
            let mut triangles_guard = all_triangles.lock().unwrap();
            for result in results {
                next_batch.extend(result.new_work);
                triangles_guard.extend(result.triangles);
            }
        }

        current_batch = next_batch;
    }

    all_triangles.into_inner().unwrap()
}

// =============================================================================
// STAGE 3: TOPOLOGY FINALIZATION
// =============================================================================

/// Intermediate result from Stage 3 before vertex refinement.
/// Contains indexed mesh with accumulated (non-normalized) face normals.
pub struct Stage3Result {
    /// Vertex positions (edge midpoints, not yet refined)
    pub vertices: Vec<(f64, f64, f64)>,
    /// Accumulated face normals per vertex (not yet normalized)
    pub accumulated_normals: Vec<(f64, f64, f64)>,
    /// Triangle indices (groups of 3)
    pub indices: Vec<u32>,
    /// Mapping from EdgeId to vertex index (for debugging/verification)
    pub edge_to_vertex: std::collections::HashMap<EdgeId, u32>,
}

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
fn stage3_topology_finalization(
    sparse_triangles: Vec<SparseTriangle>,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
) -> Stage3Result {
    use std::collections::HashMap;

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

/// Normalize a vector, returning (0,1,0) if the vector is too small.
fn normalize_or_default(v: (f64, f64, f64)) -> (f32, f32, f32) {
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

/// Convert Stage3Result to final IndexedMesh2 (without refinement).
/// Normalizes the accumulated normals and converts to f32.
#[allow(dead_code)]
fn stage3_to_indexed_mesh(result: Stage3Result) -> IndexedMesh2 {
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

// =============================================================================
// STAGE 4: VERTEX REFINEMENT & NORMAL ESTIMATION
// =============================================================================

/// Try to find a surface crossing along a given direction from initial_pos.
/// Returns Some((a, b, inside_a)) if a crossing is found, where [a, b] brackets the surface.
/// Returns None if no crossing is found along this direction.
fn find_crossing_along_direction<F>(
    initial_pos: (f64, f64, f64),
    dir: (f64, f64, f64),
    search_distance: f64,
    sampler: &F,
    stats: &SamplingStats,
    initial_inside: bool,
) -> Option<((f64, f64, f64), (f64, f64, f64), bool)>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let pos_plus = (
        initial_pos.0 + dir.0 * search_distance,
        initial_pos.1 + dir.1 * search_distance,
        initial_pos.2 + dir.2 * search_distance,
    );
    let pos_minus = (
        initial_pos.0 - dir.0 * search_distance,
        initial_pos.1 - dir.1 * search_distance,
        initial_pos.2 - dir.2 * search_distance,
    );

    let inside_plus = sample_is_inside(sampler, pos_plus.0, pos_plus.1, pos_plus.2, stats);
    let inside_minus = sample_is_inside(sampler, pos_minus.0, pos_minus.1, pos_minus.2, stats);

    if initial_inside != inside_plus {
        // Crossing between initial and +direction
        Some((initial_pos, pos_plus, initial_inside))
    } else if initial_inside != inside_minus {
        // Crossing between initial and -direction
        Some((initial_pos, pos_minus, initial_inside))
    } else if inside_plus != inside_minus {
        // Crossing between +direction and -direction (surface passed through)
        Some((pos_minus, pos_plus, inside_minus))
    } else {
        None
    }
}

/// Which direction succeeded in vertex refinement
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefineOutcome {
    /// Primary direction (accumulated normal) found a crossing
    Primary,
    /// Fallback X axis found a crossing
    FallbackX,
    /// Fallback Y axis found a crossing
    FallbackY,
    /// Fallback Z axis found a crossing
    FallbackZ,
    /// No direction found a crossing
    Miss,
}

/// Perform binary search to refine a crossing and return the refined position.
fn binary_search_crossing<F>(
    mut a: (f64, f64, f64),
    mut b: (f64, f64, f64),
    mut inside_a: bool,
    iterations: usize,
    sampler: &F,
    stats: &SamplingStats,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    for _ in 0..iterations {
        let mid = (
            (a.0 + b.0) * 0.5,
            (a.1 + b.1) * 0.5,
            (a.2 + b.2) * 0.5,
        );

        let inside_mid = sample_is_inside(sampler, mid.0, mid.1, mid.2, stats);

        if inside_mid == inside_a {
            a = mid;
            inside_a = inside_mid;
        } else {
            b = mid;
        }
    }

    // Return midpoint of final interval
    (
        (a.0 + b.0) * 0.5,
        (a.1 + b.1) * 0.5,
        (a.2 + b.2) * 0.5,
    )
}

/// Compute squared distance between two points.
#[inline]
fn dist_sq(p1: (f64, f64, f64), p2: (f64, f64, f64)) -> f64 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    let dz = p1.2 - p2.2;
    dx * dx + dy * dy + dz * dz
}

/// Refine a single vertex position using binary search along multiple directions.
///
/// # Algorithm
/// 1. Start at edge midpoint position
/// 2. Try accumulated normal AND all three cardinal axes
/// 3. For each direction that finds a crossing, refine it with binary search
/// 4. Pick the crossing that is NEAREST to the initial position
///
/// # Returns
/// A tuple of (refined_position, outcome) where outcome indicates which direction was used.
fn refine_vertex_position<F>(
    initial_pos: (f64, f64, f64),
    search_direction: (f64, f64, f64),
    search_distance: f64,
    iterations: usize,
    sampler: &F,
    stats: &SamplingStats,
) -> ((f64, f64, f64), RefineOutcome)
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Sample at the initial position to determine which side we're on
    let initial_inside = sample_is_inside(sampler, initial_pos.0, initial_pos.1, initial_pos.2, stats);

    // Build list of candidate directions with their outcome labels
    let mut candidates: Vec<((f64, f64, f64), RefineOutcome)> = Vec::with_capacity(4);

    // Primary direction: accumulated normal (if valid)
    let dir_len_sq = search_direction.0 * search_direction.0
        + search_direction.1 * search_direction.1
        + search_direction.2 * search_direction.2;

    if dir_len_sq >= 1e-12 {
        let inv_len = 1.0 / dir_len_sq.sqrt();
        candidates.push((
            (
                search_direction.0 * inv_len,
                search_direction.1 * inv_len,
                search_direction.2 * inv_len,
            ),
            RefineOutcome::Primary,
        ));
    }

    // Cardinal axis fallbacks
    candidates.push(((1.0, 0.0, 0.0), RefineOutcome::FallbackX));
    candidates.push(((0.0, 1.0, 0.0), RefineOutcome::FallbackY));
    candidates.push(((0.0, 0.0, 1.0), RefineOutcome::FallbackZ));

    // Try all directions and collect crossings that succeed
    let mut best_result: Option<((f64, f64, f64), RefineOutcome, f64)> = None; // (pos, outcome, dist_sq)

    for (dir, outcome) in &candidates {
        if let Some((a, b, inside_a)) = find_crossing_along_direction(
            initial_pos, *dir, search_distance, sampler, stats, initial_inside,
        ) {
            // Refine this crossing with binary search
            let refined = binary_search_crossing(a, b, inside_a, iterations, sampler, stats);
            let d_sq = dist_sq(initial_pos, refined);

            // Keep the nearest crossing
            match &best_result {
                None => {
                    best_result = Some((refined, *outcome, d_sq));
                }
                Some((_, _, best_dist_sq)) if d_sq < *best_dist_sq => {
                    best_result = Some((refined, *outcome, d_sq));
                }
                _ => {}
            }
        }
    }

    let (refined, outcome) = match best_result {
        Some((pos, outcome, _)) => (pos, outcome),
        None => {
            // No crossing found in any direction - keep original position
            return (initial_pos, RefineOutcome::Miss);
        }
    };

    // No displacement clamping needed - we already picked the nearest crossing
    (refined, outcome)
}

// =============================================================================
// NORMAL REFINEMENT VIA TANGENT PLANE PROBING
// =============================================================================

/// Compute an orthonormal basis (T1, T2) perpendicular to the given normal vector.
///
/// Uses the "Hughes-Möller" method to avoid numerical instability when the normal
/// is aligned with a coordinate axis.
fn orthonormal_basis_perpendicular_to(n: (f64, f64, f64)) -> ((f64, f64, f64), (f64, f64, f64)) {
    // Normalize input
    let len_sq = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
    if len_sq < 1e-12 {
        // Degenerate - return arbitrary basis
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
    }
    let inv_len = 1.0 / len_sq.sqrt();
    let n = (n.0 * inv_len, n.1 * inv_len, n.2 * inv_len);

    // Choose the axis that n is least aligned with
    let abs_x = n.0.abs();
    let abs_y = n.1.abs();
    let abs_z = n.2.abs();

    let reference = if abs_x <= abs_y && abs_x <= abs_z {
        (1.0, 0.0, 0.0) // n is least aligned with X
    } else if abs_y <= abs_z {
        (0.0, 1.0, 0.0) // n is least aligned with Y
    } else {
        (0.0, 0.0, 1.0) // n is least aligned with Z
    };

    // T1 = normalize(reference × n)
    let t1_raw = (
        reference.1 * n.2 - reference.2 * n.1,
        reference.2 * n.0 - reference.0 * n.2,
        reference.0 * n.1 - reference.1 * n.0,
    );
    let t1_len = (t1_raw.0 * t1_raw.0 + t1_raw.1 * t1_raw.1 + t1_raw.2 * t1_raw.2).sqrt();
    let t1 = (t1_raw.0 / t1_len, t1_raw.1 / t1_len, t1_raw.2 / t1_len);

    // T2 = n × T1 (already normalized since n and T1 are unit vectors and perpendicular)
    let t2 = (
        n.1 * t1.2 - n.2 * t1.1,
        n.2 * t1.0 - n.0 * t1.2,
        n.0 * t1.1 - n.1 * t1.0,
    );

    (t1, t2)
}

/// Find a surface crossing point by binary search starting from a given position,
/// searching along the specified direction.
///
/// Returns Some(surface_point) if a crossing is found, None otherwise.
/// Find a surface point by searching along a specific direction from start_pos.
///
/// This function intentionally searches ONLY along the given direction, not fallbacks.
/// For normal refinement, we need consistent search directions across all tangent probes
/// to get a coherent plane fit. Using different directions for different probes would
/// create inconsistent surface point sets.
fn find_surface_point_along_direction<F>(
    start_pos: (f64, f64, f64),
    search_dir: (f64, f64, f64),
    search_distance: f64,
    iterations: usize,
    sampler: &F,
    stats: &SamplingStats,
) -> Option<(f64, f64, f64)>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Normalize direction
    let dir_len_sq = search_dir.0 * search_dir.0
        + search_dir.1 * search_dir.1
        + search_dir.2 * search_dir.2;

    if dir_len_sq < 1e-12 {
        return None;
    }

    let inv_len = 1.0 / dir_len_sq.sqrt();
    let dir = (
        search_dir.0 * inv_len,
        search_dir.1 * inv_len,
        search_dir.2 * inv_len,
    );

    // Sample at start position
    let start_inside = sample_is_inside(sampler, start_pos.0, start_pos.1, start_pos.2, stats);

    // Try to find crossing along this direction
    if let Some((a, b, inside_a)) = find_crossing_along_direction(
        start_pos, dir, search_distance, sampler, stats, start_inside,
    ) {
        // Refine with binary search
        let refined = binary_search_crossing(a, b, inside_a, iterations, sampler, stats);
        Some(refined)
    } else {
        None
    }
}

/// Fit a plane to a set of 3D points using PCA (Principal Component Analysis).
///
/// Returns (normal, residual) where:
/// - normal: the direction perpendicular to the best-fit plane
/// - residual: sum of squared distances from points to the plane (indicates fit quality)
///
/// For edge/corner detection: low residual = smooth surface, high residual = edge/corner
fn fit_plane_to_points(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), f64) {
    let n = points.len();
    if n < 3 {
        // Not enough points - return arbitrary normal
        return ((0.0, 1.0, 0.0), f64::MAX);
    }

    // Compute centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for p in points {
        cx += p.0;
        cy += p.1;
        cz += p.2;
    }
    let inv_n = 1.0 / n as f64;
    cx *= inv_n;
    cy *= inv_n;
    cz *= inv_n;

    // For small point sets (up to 5 points), use cross-product averaging approach.
    // This is more numerically stable than matrix inversion when points lie on a plane.
    //
    // Approach: form triangles from centroid to pairs of points, compute each triangle's
    // normal via cross product, and average them.
    if n <= 5 {
        let mut sum_normal = (0.0, 0.0, 0.0);

        for i in 0..n {
            let j = (i + 1) % n;
            let v1 = (
                points[i].0 - cx,
                points[i].1 - cy,
                points[i].2 - cz,
            );
            let v2 = (
                points[j].0 - cx,
                points[j].1 - cy,
                points[j].2 - cz,
            );
            // Cross product v1 × v2
            let normal = (
                v1.1 * v2.2 - v1.2 * v2.1,
                v1.2 * v2.0 - v1.0 * v2.2,
                v1.0 * v2.1 - v1.1 * v2.0,
            );
            sum_normal.0 += normal.0;
            sum_normal.1 += normal.1;
            sum_normal.2 += normal.2;
        }

        let len = (sum_normal.0 * sum_normal.0 + sum_normal.1 * sum_normal.1 + sum_normal.2 * sum_normal.2).sqrt();
        if len < 1e-12 {
            // Degenerate - all points collinear
            return ((0.0, 1.0, 0.0), f64::MAX);
        }

        let fitted_normal = (sum_normal.0 / len, sum_normal.1 / len, sum_normal.2 / len);

        // Compute residual: sum of squared distances from each point to the fitted plane
        let mut residual = 0.0;
        for p in points {
            let d = (p.0 - cx) * fitted_normal.0 + (p.1 - cy) * fitted_normal.1 + (p.2 - cz) * fitted_normal.2;
            residual += d * d;
        }

        return (fitted_normal, residual);
    }

    // For larger point sets, use covariance matrix approach with deflation-based PCA
    // Build covariance matrix (symmetric 3x3)
    // M = Σ (p - c)(p - c)^T
    let mut m00 = 0.0;
    let mut m01 = 0.0;
    let mut m02 = 0.0;
    let mut m11 = 0.0;
    let mut m12 = 0.0;
    let mut m22 = 0.0;

    for p in points {
        let dx = p.0 - cx;
        let dy = p.1 - cy;
        let dz = p.2 - cz;
        m00 += dx * dx;
        m01 += dx * dy;
        m02 += dx * dz;
        m11 += dy * dy;
        m12 += dy * dz;
        m22 += dz * dz;
    }

    // Find eigenvector with smallest eigenvalue using deflation:
    // 1. Find largest eigenvector via power iteration on M
    // 2. Deflate: M' = M - λ1 * v1 * v1^T
    // 3. Find largest eigenvector of M' (= second eigenvector of M)
    // 4. Normal = v1 × v2 (perpendicular to both in-plane directions)

    // Power iteration to find largest eigenvector
    let mut v1: (f64, f64, f64) = (1.0, 0.0, 0.0);
    for _ in 0..10 {
        let new_v = (
            m00 * v1.0 + m01 * v1.1 + m02 * v1.2,
            m01 * v1.0 + m11 * v1.1 + m12 * v1.2,
            m02 * v1.0 + m12 * v1.1 + m22 * v1.2,
        );
        let len = (new_v.0 * new_v.0 + new_v.1 * new_v.1 + new_v.2 * new_v.2).sqrt();
        if len < 1e-12 {
            break;
        }
        v1 = (new_v.0 / len, new_v.1 / len, new_v.2 / len);
    }

    // Deflate: M2 = M - λ1 * v1 * v1^T where λ1 = v1^T * M * v1
    let lambda1 = m00 * v1.0 * v1.0 + m11 * v1.1 * v1.1 + m22 * v1.2 * v1.2
        + 2.0 * m01 * v1.0 * v1.1 + 2.0 * m02 * v1.0 * v1.2 + 2.0 * m12 * v1.1 * v1.2;

    let d00 = m00 - lambda1 * v1.0 * v1.0;
    let d01 = m01 - lambda1 * v1.0 * v1.1;
    let d02 = m02 - lambda1 * v1.0 * v1.2;
    let d11 = m11 - lambda1 * v1.1 * v1.1;
    let d12 = m12 - lambda1 * v1.1 * v1.2;
    let d22 = m22 - lambda1 * v1.2 * v1.2;

    // Find largest eigenvector of deflated matrix (= second eigenvector of M)
    // Start with vector orthogonal to v1
    let mut v2 = if v1.0.abs() < 0.9 {
        (1.0 - v1.0 * v1.0, -v1.0 * v1.1, -v1.0 * v1.2)
    } else {
        (-v1.1 * v1.0, 1.0 - v1.1 * v1.1, -v1.1 * v1.2)
    };
    let len2 = (v2.0 * v2.0 + v2.1 * v2.1 + v2.2 * v2.2).sqrt();
    if len2 > 1e-12 {
        v2 = (v2.0 / len2, v2.1 / len2, v2.2 / len2);
    }

    for _ in 0..10 {
        let new_v = (
            d00 * v2.0 + d01 * v2.1 + d02 * v2.2,
            d01 * v2.0 + d11 * v2.1 + d12 * v2.2,
            d02 * v2.0 + d12 * v2.1 + d22 * v2.2,
        );
        let len = (new_v.0 * new_v.0 + new_v.1 * new_v.1 + new_v.2 * new_v.2).sqrt();
        if len < 1e-12 {
            break;
        }
        v2 = (new_v.0 / len, new_v.1 / len, new_v.2 / len);
        // Orthogonalize against v1
        let dot = v2.0 * v1.0 + v2.1 * v1.1 + v2.2 * v1.2;
        v2 = (v2.0 - dot * v1.0, v2.1 - dot * v1.1, v2.2 - dot * v1.2);
        let len = (v2.0 * v2.0 + v2.1 * v2.1 + v2.2 * v2.2).sqrt();
        if len < 1e-12 {
            break;
        }
        v2 = (v2.0 / len, v2.1 / len, v2.2 / len);
    }

    // The normal is perpendicular to both v1 and v2
    let normal = (
        v1.1 * v2.2 - v1.2 * v2.1,
        v1.2 * v2.0 - v1.0 * v2.2,
        v1.0 * v2.1 - v1.1 * v2.0,
    );
    let len = (normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2).sqrt();
    if len < 1e-12 {
        return ((0.0, 1.0, 0.0), f64::MAX);
    }
    let v = (normal.0 / len, normal.1 / len, normal.2 / len);

    // Compute residual: sum of squared distances to the fitted plane
    let mut residual = 0.0;
    for p in points {
        let d = (p.0 - cx) * v.0 + (p.1 - cy) * v.1 + (p.2 - cz) * v.2;
        residual += d * d;
    }

    (v, residual)
}

/// Refine a surface normal using Bayesian inference with plane-fitting.
///
/// # Algorithm
/// This treats normal refinement as a Bayesian update problem:
/// 1. **Prior**: The topology-derived normal with estimated uncertainty (~1°)
/// 2. **Measurement**: Plane-fitting to probed surface points gives a normal estimate
/// 3. **Update**: Blend prior and measurement using inverse-variance weighting
///
/// The measurement uncertainty depends on binary search precision:
/// - Position error ≈ search_distance / 2^iterations
/// - This translates to angular uncertainty in the fitted normal
///
/// This means:
/// - Low iterations → high uncertainty → measurements barely affect the prior
/// - High iterations → low uncertainty → measurements dominate
///
/// # Arguments
/// * `surface_pos` - The refined vertex position (known to be on surface)
/// * `initial_normal` - Initial normal estimate (accumulated face normal)
/// * `probe_epsilon` - Distance to step in tangent directions
/// * `search_distance` - Distance to search along normal for surface crossings
/// * `binary_search_iterations` - Iterations for each binary search (affects measurement precision)
/// * `sampler` - The density sampling function
/// * `stats` - Sampling statistics
///
/// # Returns
/// The refined normal vector (unnormalized), pointing outward from the surface.
fn refine_normal_via_probing<F>(
    surface_pos: (f64, f64, f64),
    initial_normal: (f64, f64, f64),
    probe_epsilon: f64,
    search_distance: f64,
    binary_search_iterations: usize,
    sampler: &F,
    stats: &SamplingStats,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Normalize initial normal (our prior)
    let n_len_sq = initial_normal.0 * initial_normal.0
        + initial_normal.1 * initial_normal.1
        + initial_normal.2 * initial_normal.2;
    if n_len_sq < 1e-12 {
        return (0.0, 1.0, 0.0); // Degenerate - return up
    }
    let n_inv_len = 1.0 / n_len_sq.sqrt();
    let n = (
        initial_normal.0 * n_inv_len,
        initial_normal.1 * n_inv_len,
        initial_normal.2 * n_inv_len,
    );

    // Compute tangent basis
    let (t1, t2) = orthonormal_basis_perpendicular_to(n);

    // Prior uncertainty: topology normals have roughly 0.5-1° error empirically.
    // We use 1° as a conservative estimate.
    let prior_sigma: f64 = 1.0_f64.to_radians(); // 1 degree in radians

    // Measurement uncertainty: position error from binary search divided by probe distance
    // gives angular uncertainty in the fitted plane normal.
    // Position error ≈ search_distance / 2^iterations
    // Angular error ≈ position_error / probe_epsilon (for small angles)
    let position_error = search_distance / (1u64 << binary_search_iterations) as f64;
    let measurement_sigma = position_error / probe_epsilon;

    // Collect surface points for plane fitting
    let mut surface_points: Vec<(f64, f64, f64)> = vec![surface_pos];

    // Probe directions: ±T1, ±T2
    let probe_dirs = [
        (t1.0, t1.1, t1.2),
        (-t1.0, -t1.1, -t1.2),
        (t2.0, t2.1, t2.2),
        (-t2.0, -t2.1, -t2.2),
    ];

    for dir in probe_dirs.iter() {
        let probe_pos = (
            surface_pos.0 + dir.0 * probe_epsilon,
            surface_pos.1 + dir.1 * probe_epsilon,
            surface_pos.2 + dir.2 * probe_epsilon,
        );

        if let Some(found) = find_surface_point_along_direction(
            probe_pos,
            n,
            search_distance,
            binary_search_iterations,
            sampler,
            stats,
        ) {
            surface_points.push(found);
        }
    }

    // Need at least 3 points to fit a plane
    if surface_points.len() < 3 {
        return n;
    }

    // Fit plane to get measurement normal
    let (fitted_normal, residual) = fit_plane_to_points(&surface_points);

    // Orient fitted normal to match prior direction
    let dot = fitted_normal.0 * n.0 + fitted_normal.1 * n.1 + fitted_normal.2 * n.2;
    let measured = if dot < 0.0 {
        (-fitted_normal.0, -fitted_normal.1, -fitted_normal.2)
    } else {
        fitted_normal
    };

    // Check residual - high residual indicates edge/corner, trust prior more
    let num_points = surface_points.len() as f64;
    let residual_threshold = probe_epsilon * probe_epsilon * num_points * 4.0;
    let residual_factor = if residual > residual_threshold {
        // High residual: increase measurement uncertainty significantly
        // This makes us fall back toward the prior
        10.0
    } else {
        1.0
    };

    let adjusted_measurement_sigma = measurement_sigma * residual_factor;

    // Bayesian blending using inverse-variance weighting
    // weight = 1 / sigma²
    let prior_weight = 1.0 / (prior_sigma * prior_sigma);
    let measurement_weight = 1.0 / (adjusted_measurement_sigma * adjusted_measurement_sigma);
    let total_weight = prior_weight + measurement_weight;

    // Blend in tangent space to avoid issues with vector averaging
    // Project measured normal deviation onto tangent plane
    let measured_dev_t1 = measured.0 * t1.0 + measured.1 * t1.1 + measured.2 * t1.2;
    let measured_dev_t2 = measured.0 * t2.0 + measured.1 * t2.1 + measured.2 * t2.2;

    // Prior deviation is 0 by definition (n is our prior)
    // Weighted average of deviations
    let blended_dev_t1 = (measurement_weight * measured_dev_t1) / total_weight;
    let blended_dev_t2 = (measurement_weight * measured_dev_t2) / total_weight;

    // Reconstruct normal from blended deviation
    // For the prior: n = 1*n + 0*t1 + 0*t2 (no deviation)
    // For measurement: measured ≈ dot*n + measured_dev_t1*t1 + measured_dev_t2*t2
    // Blended: we interpolate the t1/t2 components
    let blended = (
        n.0 + blended_dev_t1 * t1.0 + blended_dev_t2 * t2.0,
        n.1 + blended_dev_t1 * t1.1 + blended_dev_t2 * t2.1,
        n.2 + blended_dev_t1 * t1.2 + blended_dev_t2 * t2.2,
    );

    blended
}

// =============================================================================
// NORMAL DIAGNOSTIC FUNCTIONS (only available with "normal-diagnostic" feature)
// =============================================================================

/// Compute a high-precision reference normal using plane fitting.
///
/// This uses 32 binary search iterations for maximum accuracy.
/// Used as ground truth for diagnostic comparison only.
#[cfg(feature = "normal-diagnostic")]
fn compute_reference_normal<F>(
    surface_pos: (f64, f64, f64),
    initial_normal: (f64, f64, f64),
    probe_epsilon: f64,
    search_distance: f64,
    sampler: &F,
    stats: &SamplingStats,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Normalize initial normal
    let n_len_sq = initial_normal.0 * initial_normal.0
        + initial_normal.1 * initial_normal.1
        + initial_normal.2 * initial_normal.2;
    if n_len_sq < 1e-12 {
        return (0.0, 1.0, 0.0);
    }
    let n_inv_len = 1.0 / n_len_sq.sqrt();
    let n = (
        initial_normal.0 * n_inv_len,
        initial_normal.1 * n_inv_len,
        initial_normal.2 * n_inv_len,
    );

    // Compute tangent basis
    let (t1, t2) = orthonormal_basis_perpendicular_to(n);

    // Collect surface points
    let mut surface_points: Vec<(f64, f64, f64)> = vec![surface_pos];

    // Probe all 4 directions with high precision (32 iterations)
    let probe_dirs = [
        (t1.0, t1.1, t1.2),
        (-t1.0, -t1.1, -t1.2),
        (t2.0, t2.1, t2.2),
        (-t2.0, -t2.1, -t2.2),
    ];

    for dir in probe_dirs.iter() {
        let probe_pos = (
            surface_pos.0 + dir.0 * probe_epsilon,
            surface_pos.1 + dir.1 * probe_epsilon,
            surface_pos.2 + dir.2 * probe_epsilon,
        );

        if let Some(found_pos) = find_surface_point_along_direction(
            probe_pos,
            n,
            search_distance,
            32, // High precision
            sampler,
            stats,
        ) {
            surface_points.push(found_pos);
        }
    }

    if surface_points.len() < 3 {
        return n;
    }

    let (fitted_normal, _residual) = fit_plane_to_points(&surface_points);

    // Orient to match initial
    let dot = fitted_normal.0 * n.0 + fitted_normal.1 * n.1 + fitted_normal.2 * n.2;
    if dot < 0.0 {
        (-fitted_normal.0, -fitted_normal.1, -fitted_normal.2)
    } else {
        fitted_normal
    }
}

/// Compute angular error in degrees between two unit vectors.
#[cfg(feature = "normal-diagnostic")]
fn angular_error_degrees(a: (f32, f32, f32), b: (f64, f64, f64)) -> f64 {
    let dot = (a.0 as f64) * b.0 + (a.1 as f64) * b.1 + (a.2 as f64) * b.2;
    // Clamp to [-1, 1] to handle numerical errors
    let clamped = dot.clamp(-1.0, 1.0);
    clamped.acos().to_degrees()
}

/// Compute error statistics for a set of angular errors.
#[cfg(feature = "normal-diagnostic")]
fn compute_error_stats(errors: &mut [f64]) -> (f64, f64, f64, f64) {
    if errors.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let max = errors.last().copied().unwrap_or(0.0);
    let p50 = errors[errors.len() / 2];
    let p95 = errors[errors.len() * 95 / 100];

    (mean, p50, p95, max)
}

// =============================================================================

/// Recompute accumulated face normals from vertex positions.
///
/// Given refined vertex positions, recomputes face normals for each triangle
/// and accumulates them per vertex. This gives better normal estimates than
/// using the original normals computed from unrefined edge midpoints.
fn recompute_accumulated_normals(
    vertices: &[(f64, f64, f64)],
    indices: &[u32],
) -> Vec<(f64, f64, f64)> {
    let vertex_count = vertices.len();
    let mut accumulated_normals: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, 0.0); vertex_count];

    // Iterate over triangles and accumulate face normals
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

    accumulated_normals
}

/// Stage 4: Vertex Refinement & Normal Estimation
///
/// Refines vertex positions using binary search along the accumulated normals,
/// then recomputes normals from the refined mesh topology, and optionally
/// further refines normals by probing the surface.
///
/// # Algorithm
/// 1. **Phase 4a - Vertex Refinement**: For each vertex, binary search along
///    multiple candidate directions to find the nearest surface crossing
/// 2. **Phase 4b - Normal Recomputation**: Recompute face normals from the
///    refined vertex positions and accumulate per vertex. This gives much
///    better initial normal estimates than the original unrefined positions.
/// 3. **Phase 4c - Normal Refinement** (optional): If normal_sample_iterations > 0,
///    further refine normals by probing the surface in tangent directions
fn stage4_vertex_refinement<F>(
    stage3: Stage3Result,
    sampler: &F,
    cell_size: (f64, f64, f64),
    config: &AdaptiveMeshConfig2,
    stats: &SamplingStats,
) -> IndexedMesh2
where
    F: Fn(f64, f64, f64) -> f32 + Send + Sync,
{
    let vertex_count = stage3.vertices.len();

    // Use maximum cell size for search distance to handle non-cubic bounds.
    let max_cell_size = cell_size.0.max(cell_size.1).max(cell_size.2);
    let min_cell_size = cell_size.0.min(cell_size.1).min(cell_size.2);

    // Search distance for vertex refinement
    let search_distance = max_cell_size;

    // Probe epsilon for normal refinement
    let probe_epsilon = min_cell_size * config.normal_epsilon_frac as f64;

    // =========================================================================
    // Phase 4a: Vertex Position Refinement
    // =========================================================================
    let refined_positions: Vec<(f64, f64, f64)> = if config.vertex_refinement_iterations > 0 {
        (0..vertex_count)
            .into_par_iter()
            .map(|i| {
                let initial_pos = stage3.vertices[i];
                let accumulated_normal = stage3.accumulated_normals[i];

                let (pos, outcome) = refine_vertex_position(
                    initial_pos,
                    accumulated_normal,
                    search_distance,
                    config.vertex_refinement_iterations,
                    sampler,
                    stats,
                );

                // Track refinement outcome
                match outcome {
                    RefineOutcome::Primary => {
                        stats.refine_primary_hit.fetch_add(1, Ordering::Relaxed);
                    }
                    RefineOutcome::FallbackX => {
                        stats.refine_fallback_x_hit.fetch_add(1, Ordering::Relaxed);
                    }
                    RefineOutcome::FallbackY => {
                        stats.refine_fallback_y_hit.fetch_add(1, Ordering::Relaxed);
                    }
                    RefineOutcome::FallbackZ => {
                        stats.refine_fallback_z_hit.fetch_add(1, Ordering::Relaxed);
                    }
                    RefineOutcome::Miss => {
                        stats.refine_miss.fetch_add(1, Ordering::Relaxed);
                    }
                }

                pos
            })
            .collect()
    } else {
        stage3.vertices.clone()
    };

    // =========================================================================
    // Phase 4b: Recompute Accumulated Normals from Refined Positions
    // =========================================================================
    // This gives much better normal estimates than the original normals
    // computed from unrefined edge midpoints.
    let recomputed_normals = recompute_accumulated_normals(&refined_positions, &stage3.indices);

    // =========================================================================
    // Phase 4c: Normal Refinement (optional) or Direct Use
    // =========================================================================
    let final_normals: Vec<(f32, f32, f32)> = if config.normal_sample_iterations > 0 {
        // Further refine normals via tangent plane probing
        let normal_search_distance = search_distance * 2.0;

        (0..vertex_count)
            .into_par_iter()
            .map(|i| {
                let refined_pos = refined_positions[i];
                let recomputed_normal = recomputed_normals[i];

                let refined = refine_normal_via_probing(
                    refined_pos,
                    recomputed_normal,
                    probe_epsilon,
                    normal_search_distance,
                    config.normal_sample_iterations,
                    sampler,
                    stats,
                );
                normalize_or_default(refined)
            })
            .collect()
    } else {
        // Just normalize the recomputed normals
        recomputed_normals
            .iter()
            .map(|n| normalize_or_default(*n))
            .collect()
    };

    // Convert positions to f32
    let final_vertices: Vec<(f32, f32, f32)> = refined_positions
        .iter()
        .map(|p| (p.0 as f32, p.1 as f32, p.2 as f32))
        .collect();

    IndexedMesh2 {
        vertices: final_vertices,
        normals: final_normals,
        indices: stage3.indices,
    }
}

/// Run normal diagnostics: compute reference normals and compare various iteration levels.
///
/// Tests iteration counts: 0 (topology only), 4, 8, 12, 16, 24
/// Returns error statistics for each level compared to high-precision (32-iter) reference.
#[cfg(feature = "normal-diagnostic")]
pub fn run_normal_diagnostics<F>(
    refined_positions: &[(f64, f64, f64)],
    recomputed_normals: &[(f64, f64, f64)],
    probe_epsilon: f64,
    search_distance: f64,
    sampler: &F,
    stats: &SamplingStats,
) -> Vec<NormalDiagnosticEntry>
where
    F: Fn(f64, f64, f64) -> f32 + Send + Sync,
{
    let vertex_count = refined_positions.len();
    eprintln!("Running normal diagnostics on {} vertices...", vertex_count);

    // First, compute high-precision reference normals (32 iterations)
    eprintln!("  Computing reference normals (32 iterations)...");
    let samples_before_ref = stats.total_samples.load(Ordering::Relaxed);

    let reference_normals: Vec<(f64, f64, f64)> = (0..vertex_count)
        .into_par_iter()
        .map(|i| {
            compute_reference_normal(
                refined_positions[i],
                recomputed_normals[i],
                probe_epsilon,
                search_distance,
                sampler,
                stats,
            )
        })
        .collect();

    let samples_for_ref = stats.total_samples.load(Ordering::Relaxed) - samples_before_ref;
    eprintln!("  Reference normals computed ({} samples)", samples_for_ref);

    // Test iteration levels: 0 (topology-only), 4, 8, 12, 16, 24
    let test_iterations = [0usize, 4, 8, 12, 16, 24];
    let mut results = Vec::new();

    for &iterations in &test_iterations {
        let samples_before = stats.total_samples.load(Ordering::Relaxed);

        // Compute normals at this iteration level
        let test_normals: Vec<(f32, f32, f32)> = if iterations == 0 {
            // Topology-only: just normalize recomputed normals
            recomputed_normals
                .iter()
                .map(|n| normalize_or_default(*n))
                .collect()
        } else {
            // Probe-based refinement at this iteration count
            (0..vertex_count)
                .into_par_iter()
                .map(|i| {
                    let refined = refine_normal_via_probing(
                        refined_positions[i],
                        recomputed_normals[i],
                        probe_epsilon,
                        search_distance,
                        iterations,
                        sampler,
                        stats,
                    );
                    normalize_or_default(refined)
                })
                .collect()
        };

        let samples_used = stats.total_samples.load(Ordering::Relaxed) - samples_before;

        // Compute errors vs reference
        let mut errors: Vec<f64> = test_normals
            .iter()
            .zip(reference_normals.iter())
            .map(|(test, reference)| angular_error_degrees(*test, *reference))
            .collect();

        let (mean, p50, p95, max) = compute_error_stats(&mut errors);

        eprintln!(
            "  iterations={:>2}: mean={:>6.2}°  p50={:>6.2}°  p95={:>6.2}°  max={:>6.2}°  samples={}",
            iterations, mean, p50, p95, max, samples_used
        );

        results.push(NormalDiagnosticEntry {
            iterations,
            mean_error_degrees: mean,
            p50_error_degrees: p50,
            p95_error_degrees: p95,
            max_error_degrees: max,
            extra_samples: samples_used,
        });
    }

    results
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

/// Main entry point for adaptive surface nets meshing.
///
/// # Arguments
/// * `sampler` - Function that returns > 0.0 if point is inside the model
/// * `bounds_min` - Minimum corner of bounding box
/// * `bounds_max` - Maximum corner of bounding box
/// * `config` - Algorithm configuration
///
/// # Returns
/// A MeshingResult2 containing the indexed mesh and detailed profiling statistics
pub fn adaptive_surface_nets_2<F>(
    sampler: F,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    config: &AdaptiveMeshConfig2,
) -> MeshingResult2
where
    F: Fn(f64, f64, f64) -> f32 + Send + Sync,
{
    let total_start = Instant::now();
    let stats = SamplingStats::default();

    // Convert bounds to f64 for internal calculations
    let bounds_min_f64 = (
        bounds_min.0 as f64,
        bounds_min.1 as f64,
        bounds_min.2 as f64,
    );
    let bounds_max_f64 = (
        bounds_max.0 as f64,
        bounds_max.1 as f64,
        bounds_max.2 as f64,
    );

    // Calculate cell size at finest level (per-axis for non-cubic bounds)
    // Total cells at finest level = base_resolution * 2^max_depth
    let finest_cells_per_axis = config.base_resolution * (1 << config.max_depth);
    let cell_size = (
        (bounds_max_f64.0 - bounds_min_f64.0) / finest_cells_per_axis as f64,
        (bounds_max_f64.1 - bounds_min_f64.1) / finest_cells_per_axis as f64,
        (bounds_max_f64.2 - bounds_min_f64.2) / finest_cells_per_axis as f64,
    );

    // Stage 1: Coarse grid discovery
    let stage1_start = Instant::now();
    let samples_before_stage1 = stats.total_samples.load(Ordering::Relaxed);
    let initial_work_queue = stage1_coarse_discovery(
        &sampler,
        bounds_min_f64,
        bounds_max_f64,
        config,
        &stats,
    );
    let stage1_time = stage1_start.elapsed().as_secs_f64();
    let stage1_samples = stats.total_samples.load(Ordering::Relaxed) - samples_before_stage1;
    let stage1_mixed_cells = initial_work_queue.len();

    // Stage 2: Subdivision and triangle emission
    let stage2_start = Instant::now();
    let samples_before_stage2 = stats.total_samples.load(Ordering::Relaxed);
    let cuboids_before_stage2 = stats.cuboids_processed.load(Ordering::Relaxed);
    let triangles_before_stage2 = stats.triangles_emitted.load(Ordering::Relaxed);
    let sparse_triangles = stage2_subdivision_and_emission(
        initial_work_queue,
        &sampler,
        bounds_min_f64,
        cell_size,
        config,
        &stats,
    );
    let stage2_time = stage2_start.elapsed().as_secs_f64();
    let stage2_samples = stats.total_samples.load(Ordering::Relaxed) - samples_before_stage2;
    let stage2_cuboids = stats.cuboids_processed.load(Ordering::Relaxed) - cuboids_before_stage2;
    let stage2_triangles = stats.triangles_emitted.load(Ordering::Relaxed) - triangles_before_stage2;

    // Stage 3: Topology finalization
    let stage3_start = Instant::now();
    let stage3_result = stage3_topology_finalization(
        sparse_triangles,
        bounds_min_f64,
        cell_size,
    );
    let stage3_time = stage3_start.elapsed().as_secs_f64();
    let stage3_unique_vertices = stage3_result.vertices.len();

    // Stage 4: Vertex refinement & normal estimation
    let stage4_start = Instant::now();
    let samples_before_stage4 = stats.total_samples.load(Ordering::Relaxed);
    let mesh = stage4_vertex_refinement(
        stage3_result,
        &sampler,
        cell_size,
        config,
        &stats,
    );
    let stage4_time = stage4_start.elapsed().as_secs_f64();
    let stage4_samples = stats.total_samples.load(Ordering::Relaxed) - samples_before_stage4;

    let total_time = total_start.elapsed().as_secs_f64();
    let total_samples = stats.total_samples.load(Ordering::Relaxed);

    let meshing_stats = MeshingStats2 {
        total_time_secs: total_time,
        stage1_time_secs: stage1_time,
        stage1_samples,
        stage1_mixed_cells,
        stage2_time_secs: stage2_time,
        stage2_samples,
        stage2_cuboids_processed: stage2_cuboids,
        stage2_triangles_emitted: stage2_triangles,
        stage3_time_secs: stage3_time,
        stage3_unique_vertices,
        stage4_time_secs: stage4_time,
        stage4_samples,
        stage4_refine_primary_hit: stats.refine_primary_hit.load(Ordering::Relaxed),
        stage4_refine_fallback_x_hit: stats.refine_fallback_x_hit.load(Ordering::Relaxed),
        stage4_refine_fallback_y_hit: stats.refine_fallback_y_hit.load(Ordering::Relaxed),
        stage4_refine_fallback_z_hit: stats.refine_fallback_z_hit.load(Ordering::Relaxed),
        stage4_refine_miss: stats.refine_miss.load(Ordering::Relaxed),
        total_samples,
        total_vertices: mesh.vertices.len(),
        total_triangles: mesh.indices.len() / 3,
        effective_resolution: finest_cells_per_axis,
    };

    MeshingResult2 {
        mesh,
        stats: meshing_stats,
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

    // =========================================================================
    // Stage 1 Tests
    // =========================================================================

    /// Helper: Unit sphere centered at origin (radius 1)
    fn sphere_sampler(x: f64, y: f64, z: f64) -> f32 {
        let r2 = x * x + y * y + z * z;
        if r2 < 1.0 { 1.0 } else { 0.0 }
    }

    #[test]
    fn test_stage1_sample_count() {
        // With base_resolution=4, we should sample 5³ = 125 corner points
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            ..Default::default()
        };
        let stats = SamplingStats::default();

        let _ = stage1_coarse_discovery(
            &sphere_sampler,
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
            &config,
            &stats,
        );

        let expected_samples = 5 * 5 * 5; // (base_resolution + 1)³
        assert_eq!(
            stats.total_samples.load(Ordering::Relaxed),
            expected_samples,
            "Should sample exactly (base_resolution + 1)³ corner points"
        );
    }

    #[test]
    fn test_stage1_finds_mixed_cells() {
        // Sphere of radius 1 in a [-2, 2]³ box with resolution 4
        // Cell size = 4/4 = 1, so cells at the surface should be detected
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            ..Default::default()
        };
        let stats = SamplingStats::default();

        let work_queue = stage1_coarse_discovery(
            &sphere_sampler,
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
            &config,
            &stats,
        );

        // Should find some mixed cells (sphere surface crosses the grid)
        assert!(
            !work_queue.is_empty(),
            "Should find mixed cells where sphere surface crosses grid"
        );

        // All returned entries should have all corners known
        for entry in &work_queue {
            assert!(
                entry.all_corners_known(),
                "All work queue entries should have all corners pre-sampled"
            );

            // The entry should be mixed
            let mask = entry.to_corner_mask();
            assert!(mask.is_mixed(), "All entries should be mixed cells");
        }
    }

    #[test]
    fn test_stage1_empty_for_fully_inside() {
        // Tiny sphere that fits entirely within a single cell
        // Box is [-1, 1]³, resolution 2, so cell size = 1
        // Sphere radius 0.1 centered at (0.5, 0.5, 0.5) fits in one cell
        let tiny_sphere = |x: f64, y: f64, z: f64| -> f32 {
            let dx = x - 0.5;
            let dy = y - 0.5;
            let dz = z - 0.5;
            if dx * dx + dy * dy + dz * dz < 0.01 { 1.0 } else { 0.0 }
        };

        let config = AdaptiveMeshConfig2 {
            base_resolution: 2,
            ..Default::default()
        };
        let stats = SamplingStats::default();

        let work_queue = stage1_coarse_discovery(
            &tiny_sphere,
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
            &config,
            &stats,
        );

        // The sphere is so small that no coarse grid corner will hit it
        // All cells should be fully outside, hence no mixed cells
        assert!(
            work_queue.is_empty(),
            "Tiny sphere should not create mixed cells at coarse resolution"
        );
    }

    #[test]
    fn test_stage1_corner_ordering() {
        // Verify that corner ordering matches CORNER_OFFSETS
        // Create a sampler that returns different values based on position
        // to verify the corners are correctly indexed
        let config = AdaptiveMeshConfig2 {
            base_resolution: 1, // Single cell
            ..Default::default()
        };
        let stats = SamplingStats::default();

        // Sampler: inside only if x > 0.5 (so corners 1, 3, 5, 7 are inside)
        let half_space = |x: f64, _y: f64, _z: f64| -> f32 {
            if x > 0.5 { 1.0 } else { 0.0 }
        };

        let work_queue = stage1_coarse_discovery(
            &half_space,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            &config,
            &stats,
        );

        assert_eq!(work_queue.len(), 1, "Should find exactly one mixed cell");

        let entry = &work_queue[0];
        let corners = entry.known_corners;

        // Corners with x=1 should be inside (corners 1, 3, 5, 7)
        assert_eq!(corners[0], Some(false), "Corner 0 (0,0,0) should be outside");
        assert_eq!(corners[1], Some(true), "Corner 1 (1,0,0) should be inside");
        assert_eq!(corners[2], Some(false), "Corner 2 (0,1,0) should be outside");
        assert_eq!(corners[3], Some(true), "Corner 3 (1,1,0) should be inside");
        assert_eq!(corners[4], Some(false), "Corner 4 (0,0,1) should be outside");
        assert_eq!(corners[5], Some(true), "Corner 5 (1,0,1) should be inside");
        assert_eq!(corners[6], Some(false), "Corner 6 (0,1,1) should be outside");
        assert_eq!(corners[7], Some(true), "Corner 7 (1,1,1) should be inside");
    }

    #[test]
    fn test_cuboid_id_to_finest_level() {
        // At depth 0 with max_depth 3, scale = 2^3 = 8
        let cuboid = CuboidId::new(1, 2, 3, 0);
        let (fx, fy, fz) = cuboid.to_finest_level(3);
        assert_eq!((fx, fy, fz), (8, 16, 24));

        // At depth 2 with max_depth 3, scale = 2^1 = 2
        let cuboid = CuboidId::new(5, 6, 7, 2);
        let (fx, fy, fz) = cuboid.to_finest_level(3);
        assert_eq!((fx, fy, fz), (10, 12, 14));

        // At max depth, scale = 1
        let cuboid = CuboidId::new(5, 6, 7, 3);
        let (fx, fy, fz) = cuboid.to_finest_level(3);
        assert_eq!((fx, fy, fz), (5, 6, 7));
    }

    #[test]
    fn test_cuboid_subdivide() {
        let parent = CuboidId::new(1, 2, 3, 0);
        let children = parent.subdivide();

        // Check that children are at depth 1
        for child in &children {
            assert_eq!(child.depth, 1);
        }

        // Check child positions follow canonical corner ordering
        assert_eq!((children[0].x, children[0].y, children[0].z), (2, 4, 6)); // (0,0,0)
        assert_eq!((children[1].x, children[1].y, children[1].z), (3, 4, 6)); // (1,0,0)
        assert_eq!((children[2].x, children[2].y, children[2].z), (2, 5, 6)); // (0,1,0)
        assert_eq!((children[3].x, children[3].y, children[3].z), (3, 5, 6)); // (1,1,0)
        assert_eq!((children[4].x, children[4].y, children[4].z), (2, 4, 7)); // (0,0,1)
        assert_eq!((children[5].x, children[5].y, children[5].z), (3, 4, 7)); // (1,0,1)
        assert_eq!((children[6].x, children[6].y, children[6].z), (2, 5, 7)); // (0,1,1)
        assert_eq!((children[7].x, children[7].y, children[7].z), (3, 5, 7)); // (1,1,1)
    }

    #[test]
    fn test_edge_id_computation() {
        // Test that edge IDs are computed correctly from cell coordinates
        let cell = CuboidId::new(1, 2, 3, 2); // At depth 2
        let max_depth = 4;

        // At depth 2 with max_depth 4, scale = 2^2 = 4
        // So finest-level coords are (4, 8, 12)

        // Edge 0: X-axis at (0,0,0) offset -> (4, 8, 12, axis=0)
        let e0 = cell.edge_id(0, max_depth);
        assert_eq!((e0.x, e0.y, e0.z, e0.axis), (4, 8, 12, 0));

        // Edge 5: Y-axis at (1,0,0) offset -> (5, 8, 12, axis=1)
        let e5 = cell.edge_id(5, max_depth);
        assert_eq!((e5.x, e5.y, e5.z, e5.axis), (5, 8, 12, 1));

        // Edge 11: Z-axis at (1,1,0) offset -> (5, 9, 12, axis=2)
        let e11 = cell.edge_id(11, max_depth);
        assert_eq!((e11.x, e11.y, e11.z, e11.axis), (5, 9, 12, 2));
    }

    // =========================================================================
    // Stage 2 Tests
    // =========================================================================

    #[test]
    fn test_stage2_emits_triangles_for_sphere() {
        // Unit sphere in [-2, 2]³ box
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            ..Default::default()
        };
        let stats = SamplingStats::default();
        let bounds_min = (-2.0, -2.0, -2.0);
        let bounds_max = (2.0, 2.0, 2.0);

        // Calculate cell size at finest level (per-axis)
        let finest_cells = config.base_resolution * (1 << config.max_depth);
        let cell_size = (
            (bounds_max.0 - bounds_min.0) / finest_cells as f64,
            (bounds_max.1 - bounds_min.1) / finest_cells as f64,
            (bounds_max.2 - bounds_min.2) / finest_cells as f64,
        );

        let initial_queue = stage1_coarse_discovery(
            &sphere_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats,
        );

        let triangles = stage2_subdivision_and_emission(
            initial_queue,
            &sphere_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats,
        );

        // Should produce triangles for a sphere
        assert!(
            !triangles.is_empty(),
            "Stage 2 should emit triangles for sphere surface"
        );

        // A sphere should produce a reasonable number of triangles
        // At resolution 4 * 2^2 = 16 cells per axis, we expect ~hundreds of triangles
        assert!(
            triangles.len() > 10,
            "Should have more than 10 triangles, got {}",
            triangles.len()
        );
    }

    #[test]
    fn test_stage2_shared_corners_neighbor_propagation() {
        // Test that shared_corners_with_neighbor returns correct mappings
        // +X neighbor: our corners 1,3,5,7 -> their corners 0,2,4,6
        let mapping = shared_corners_with_neighbor(1, 0, 0);
        assert_eq!(mapping, [(1, 0), (3, 2), (5, 4), (7, 6)]);

        // -X neighbor: our corners 0,2,4,6 -> their corners 1,3,5,7
        let mapping = shared_corners_with_neighbor(-1, 0, 0);
        assert_eq!(mapping, [(0, 1), (2, 3), (4, 5), (6, 7)]);

        // +Y neighbor: our corners 2,3,6,7 -> their corners 0,1,4,5
        let mapping = shared_corners_with_neighbor(0, 1, 0);
        assert_eq!(mapping, [(2, 0), (3, 1), (6, 4), (7, 5)]);

        // +Z neighbor: our corners 4,5,6,7 -> their corners 0,1,2,3
        let mapping = shared_corners_with_neighbor(0, 0, 1);
        assert_eq!(mapping, [(4, 0), (5, 1), (6, 2), (7, 3)]);
    }

    #[test]
    fn test_shared_face_is_mixed() {
        // All corners known, +X face is mixed (corners 1,3,5,7 have different states)
        let corners_mixed_x = [
            Some(true),  // 0
            Some(true),  // 1 - +X face
            Some(true),  // 2
            Some(false), // 3 - +X face (different!)
            Some(true),  // 4
            Some(true),  // 5 - +X face
            Some(true),  // 6
            Some(true),  // 7 - +X face
        ];
        assert!(
            shared_face_is_mixed(&corners_mixed_x, 1, 0, 0),
            "+X face should be mixed (corner 3 differs)"
        );

        // All corners same state on +X face
        let corners_uniform_x = [
            Some(false), // 0
            Some(true),  // 1 - +X face
            Some(false), // 2
            Some(true),  // 3 - +X face
            Some(false), // 4
            Some(true),  // 5 - +X face
            Some(false), // 6
            Some(true),  // 7 - +X face
        ];
        assert!(
            !shared_face_is_mixed(&corners_uniform_x, 1, 0, 0),
            "+X face should NOT be mixed (all +X corners are true)"
        );

        // -Z face: corners 0,1,2,3
        let corners_mixed_neg_z = [
            Some(true),  // 0 - -Z face
            Some(false), // 1 - -Z face (different!)
            Some(true),  // 2 - -Z face
            Some(true),  // 3 - -Z face
            Some(true),  // 4
            Some(true),  // 5
            Some(true),  // 6
            Some(true),  // 7
        ];
        assert!(
            shared_face_is_mixed(&corners_mixed_neg_z, 0, 0, -1),
            "-Z face should be mixed"
        );

        // Unknown corner should be conservative (return true)
        let corners_with_unknown = [
            Some(true),
            None, // unknown
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
        ];
        assert!(
            shared_face_is_mixed(&corners_with_unknown, 1, 0, 0),
            "Unknown corner should trigger conservative exploration"
        );
    }

    #[test]
    fn test_subdivide_and_filter_mixed() {
        // Test that subdivide_and_filter_mixed correctly:
        // 1. Samples all 27 child corner positions
        // 2. Returns only mixed children
        // 3. All returned children have all corners filled in

        // Create a parent with mixed corners (half inside, half outside based on X)
        let parent_cuboid = CuboidId::new(0, 0, 0, 0);
        let parent_corners = [
            Some(false), // corner 0 (0,0,0) - x=0, outside
            Some(true),  // corner 1 (1,0,0) - x=1, inside
            Some(false), // corner 2 (0,1,0) - x=0, outside
            Some(true),  // corner 3 (1,1,0) - x=1, inside
            Some(false), // corner 4 (0,0,1) - x=0, outside
            Some(true),  // corner 5 (1,0,1) - x=1, inside
            Some(false), // corner 6 (0,1,1) - x=0, outside
            Some(true),  // corner 7 (1,1,1) - x=1, inside
        ];
        let parent = WorkQueueEntry::with_corners(parent_cuboid, parent_corners);

        // Sampler: inside if x > 0.25 (in [0,1] bounds)
        // This means the -X children (cx=0) will have surface crossing them
        // and the +X children (cx=1) will be fully inside
        let half_space = |x: f64, _y: f64, _z: f64| -> f32 {
            if x > 0.25 { 1.0 } else { 0.0 }
        };

        let stats = SamplingStats::default();
        let bounds_min = (0.0, 0.0, 0.0);
        let cell_size = (0.5, 0.5, 0.5); // Finest cell size (at max_depth=1, parent spans 2 finest cells)
        let max_depth = 1;

        let mixed_children = subdivide_and_filter_mixed(
            &parent,
            &half_space,
            bounds_min,
            cell_size,
            max_depth,
            &stats,
        );

        // All returned children should have all corners known
        for child in &mixed_children {
            assert!(
                child.all_corners_known(),
                "All returned children should have all corners filled in"
            );

            // And they should be mixed
            let mask = child.to_corner_mask();
            assert!(mask.is_mixed(), "All returned children should be mixed");
        }

        // We should have sampled exactly 19 new positions (27 - 8 parent corners)
        assert_eq!(
            stats.total_samples.load(Ordering::Relaxed),
            19,
            "Should sample exactly 19 new positions (27 total - 8 reused parent corners)"
        );
    }

    #[test]
    fn test_stage2_edge_welding() {
        // Two adjacent cells should produce the same EdgeId for their shared edge
        let cell_a = CuboidId::new(0, 0, 0, 2);
        let cell_b = CuboidId::new(1, 0, 0, 2); // +X neighbor
        let max_depth = 2;

        // Cell A's edge 5 (Y-axis at +X face) should equal cell B's edge 4 (Y-axis at origin)
        // Edge 5: corners 1-3, offset (1, 0, 0)
        // Edge 4: corners 0-2, offset (0, 0, 0)
        // For cell A at (0,0,0): edge 5 is at (0+1, 0+0, 0+0) = (1, 0, 0)
        // For cell B at (1,0,0): edge 4 is at (1+0, 0+0, 0+0) = (1, 0, 0)

        let edge_a = cell_a.edge_id(5, max_depth);
        let edge_b = cell_b.edge_id(4, max_depth);

        assert_eq!(
            edge_a, edge_b,
            "Adjacent cells should produce identical EdgeIds for shared edges"
        );
    }

    #[test]
    fn test_stage2_triangle_count_consistency() {
        // Run twice with same input, should get same triangle count
        let config = AdaptiveMeshConfig2 {
            base_resolution: 2,
            max_depth: 2,
            ..Default::default()
        };

        let bounds_min = (-1.5, -1.5, -1.5);
        let bounds_max = (1.5, 1.5, 1.5);
        let finest_cells = config.base_resolution * (1 << config.max_depth);
        let cell_size = (
            (bounds_max.0 - bounds_min.0) / finest_cells as f64,
            (bounds_max.1 - bounds_min.1) / finest_cells as f64,
            (bounds_max.2 - bounds_min.2) / finest_cells as f64,
        );

        let stats1 = SamplingStats::default();
        let initial_queue1 = stage1_coarse_discovery(
            &sphere_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats1,
        );
        let triangles1 = stage2_subdivision_and_emission(
            initial_queue1,
            &sphere_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats1,
        );

        let stats2 = SamplingStats::default();
        let initial_queue2 = stage1_coarse_discovery(
            &sphere_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats2,
        );
        let triangles2 = stage2_subdivision_and_emission(
            initial_queue2,
            &sphere_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats2,
        );

        assert_eq!(
            triangles1.len(),
            triangles2.len(),
            "Repeated runs should produce same triangle count"
        );
    }

    #[test]
    fn test_stage2_no_triangles_for_empty_space() {
        // Sampler that returns nothing inside
        let empty_sampler = |_x: f64, _y: f64, _z: f64| -> f32 { 0.0 };

        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            ..Default::default()
        };
        let stats = SamplingStats::default();
        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);
        let finest_cells = config.base_resolution * (1 << config.max_depth);
        let cell_size = (
            (bounds_max.0 - bounds_min.0) / finest_cells as f64,
            (bounds_max.1 - bounds_min.1) / finest_cells as f64,
            (bounds_max.2 - bounds_min.2) / finest_cells as f64,
        );

        let initial_queue = stage1_coarse_discovery(
            &empty_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats,
        );

        // Empty space should produce no mixed cells
        assert!(initial_queue.is_empty(), "Empty space should have no mixed cells");

        let triangles = stage2_subdivision_and_emission(
            initial_queue,
            &empty_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats,
        );

        assert!(triangles.is_empty(), "Empty space should produce no triangles");
    }

    #[test]
    fn test_corner_world_position() {
        let cell = CuboidId::new(0, 0, 0, 0);
        let bounds_min = (0.0, 0.0, 0.0);
        let cell_size = (1.0, 1.0, 1.0);
        let max_depth = 2;

        // At depth 0, max_depth 2: cell spans 4 finest-level cells
        // Corner 0 (0,0,0) should be at (0, 0, 0)
        let c0 = corner_world_position(&cell, 0, bounds_min, cell_size, max_depth);
        assert_eq!(c0, (0.0, 0.0, 0.0));

        // Corner 7 (1,1,1) should be at (4, 4, 4) since scale = 2^2 = 4
        let c7 = corner_world_position(&cell, 7, bounds_min, cell_size, max_depth);
        assert_eq!(c7, (4.0, 4.0, 4.0));

        // A cell at depth 2 (finest level) should have corners 1 unit apart
        let fine_cell = CuboidId::new(2, 3, 4, 2);
        let c0_fine = corner_world_position(&fine_cell, 0, bounds_min, cell_size, max_depth);
        let c7_fine = corner_world_position(&fine_cell, 7, bounds_min, cell_size, max_depth);
        assert_eq!(c0_fine, (2.0, 3.0, 4.0));
        assert_eq!(c7_fine, (3.0, 4.0, 5.0));
    }

    // =========================================================================
    // Stage 4 Tests
    // =========================================================================

    #[test]
    fn test_refine_vertex_position_finds_surface() {
        // Simple half-space: inside if x < 0.3
        let half_space = |x: f64, _y: f64, _z: f64| -> f32 {
            if x < 0.3 { 1.0 } else { 0.0 }
        };
        let stats = SamplingStats::default();

        // Start at x=0.5 (outside), search toward -x (normal points outward from inside)
        // Surface is at x=0.3
        let initial_pos = (0.5, 0.0, 0.0);
        let search_dir = (-1.0, 0.0, 0.0); // Points toward inside
        let search_distance = 0.5;
        let iterations = 10;

        let refined = refine_vertex_position(
            initial_pos,
            search_dir,
            search_distance,
            iterations,
            &half_space,
            &stats,
        );

        // Should be close to x=0.3
        let (pos, _outcome) = refined;
        assert!(
            (pos.0 - 0.3).abs() < 0.01,
            "Refined x should be close to 0.3, got {}",
            pos.0
        );
        assert_eq!(pos.1, 0.0, "Y should be unchanged");
        assert_eq!(pos.2, 0.0, "Z should be unchanged");
    }

    #[test]
    fn test_refine_vertex_position_no_crossing() {
        // All outside
        let all_outside = |_x: f64, _y: f64, _z: f64| -> f32 { 0.0 };
        let stats = SamplingStats::default();

        let initial_pos = (0.5, 0.5, 0.5);
        let search_dir = (1.0, 0.0, 0.0);
        let search_distance = 1.0;
        let iterations = 5;

        let refined = refine_vertex_position(
            initial_pos,
            search_dir,
            search_distance,
            iterations,
            &all_outside,
            &stats,
        );

        // Should return original position since no crossing found
        let (pos, _outcome) = refined;
        assert_eq!(pos, initial_pos, "Should return original when no crossing");
    }

    #[test]
    fn test_refine_vertex_position_zero_direction() {
        // When primary direction is zero, the function uses fallback directions (+X, +Y, +Z)
        let sphere = |x: f64, y: f64, z: f64| -> f32 {
            if x * x + y * y + z * z < 1.0 { 1.0 } else { 0.0 }
        };
        let stats = SamplingStats::default();

        let initial_pos = (0.5, 0.0, 0.0);
        let search_dir = (0.0, 0.0, 0.0); // Zero direction
        let search_distance = 0.5;
        let iterations = 5;

        let refined = refine_vertex_position(
            initial_pos,
            search_dir,
            search_distance,
            iterations,
            &sphere,
            &stats,
        );

        // With zero primary direction, fallback +X finds sphere surface near x=1.0
        // The point (0.5, 0, 0) is inside, so +X searches toward surface at x=1.0
        let (pos, outcome) = refined;
        assert!(
            (pos.0 - 1.0).abs() < 0.02,
            "X should be near 1.0 (sphere surface), got {}",
            pos.0
        );
        assert_eq!(pos.1, 0.0, "Y should be unchanged");
        assert_eq!(pos.2, 0.0, "Z should be unchanged");
        assert_eq!(outcome, RefineOutcome::FallbackX, "Should use X fallback");
    }

    #[test]
    fn test_refine_normal_points_outward() {
        // Sphere: normal should point radially outward
        let sphere = |x: f64, y: f64, z: f64| -> f32 {
            let r2 = x * x + y * y + z * z;
            if r2 < 1.0 { 1.0 } else { 0.0 }
        };
        let stats = SamplingStats::default();

        // Point on +X axis of sphere surface
        let surface_pos = (1.0, 0.0, 0.0);
        // Initial normal estimate (pointing outward in +X)
        let initial_normal = (1.0, 0.0, 0.0);
        let probe_epsilon = 0.1;
        let search_distance = 0.2;
        let iterations = 4;

        let normal = refine_normal_via_probing(
            surface_pos,
            initial_normal,
            probe_epsilon,
            search_distance,
            iterations,
            &sphere,
            &stats,
        );

        // Normal should point in +X direction (outward)
        assert!(
            normal.0 > 0.9,
            "Normal X component should be close to 1.0 (pointing outward), got {}",
            normal.0
        );
        // Y and Z should be near zero for a point on the X axis
        assert!(
            normal.1.abs() < 0.1,
            "Normal Y component should be near zero, got {}",
            normal.1
        );
        assert!(
            normal.2.abs() < 0.1,
            "Normal Z component should be near zero, got {}",
            normal.2
        );
    }

    #[test]
    fn test_stage4_full_pipeline() {
        // Run the full pipeline with refinement enabled
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            vertex_refinement_iterations: 4,
            normal_sample_iterations: 1,
            normal_epsilon_frac: 0.1,
            ..Default::default()
        };

        let result = adaptive_surface_nets_2(
            sphere_sampler,
            (-1.5, -1.5, -1.5),
            (1.5, 1.5, 1.5),
            &config,
        );
        let mesh = &result.mesh;

        // Should produce a mesh
        assert!(!mesh.vertices.is_empty(), "Should have vertices");
        assert!(!mesh.normals.is_empty(), "Should have normals");
        assert!(!mesh.indices.is_empty(), "Should have indices");

        // Vertices and normals should have same count
        assert_eq!(
            mesh.vertices.len(),
            mesh.normals.len(),
            "Vertex and normal counts should match"
        );

        // All normals should be unit length (approximately)
        for (i, n) in mesh.normals.iter().enumerate() {
            let len = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
            assert!(
                (len - 1.0).abs() < 0.01,
                "Normal {} should be unit length, got {}",
                i,
                len
            );
        }

        // Vertices should be roughly on the unit sphere surface
        // (with some tolerance for mesh approximation)
        for (i, v) in mesh.vertices.iter().enumerate() {
            let r = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
            assert!(
                (r - 1.0).abs() < 0.3, // Allow 30% tolerance for low-res mesh
                "Vertex {} should be near sphere surface, got r={}",
                i,
                r
            );
        }
    }

    #[test]
    fn test_stage4_without_refinement() {
        // Run with refinement disabled
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            vertex_refinement_iterations: 0, // Disabled
            normal_sample_iterations: 0,     // Disabled
            ..Default::default()
        };

        let result = adaptive_surface_nets_2(
            sphere_sampler,
            (-1.5, -1.5, -1.5),
            (1.5, 1.5, 1.5),
            &config,
        );
        let mesh = &result.mesh;

        // Should still produce a valid mesh
        assert!(!mesh.vertices.is_empty(), "Should have vertices");
        assert_eq!(
            mesh.vertices.len(),
            mesh.normals.len(),
            "Vertex and normal counts should match"
        );
    }
}
