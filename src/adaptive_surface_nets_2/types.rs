//! Type definitions for the adaptive surface nets algorithm.
//!
//! Contains all core types including configuration, IDs, work queue structures,
//! statistics, and output types.

use std::sync::atomic::AtomicU64;
use crate::adaptive_surface_nets_2::lookup_tables::EDGE_TABLE;

// =============================================================================
// SAMPLER TRAIT
// =============================================================================

/// Marker trait for sampler functions used in adaptive surface nets.
///
/// On native builds, samplers must be Send + Sync for parallel iteration.
/// On web builds, only Fn is required since iteration is sequential.
#[cfg(feature = "native")]
pub trait SamplerFn: Fn(f64, f64, f64) -> f32 + Send + Sync {}
#[cfg(feature = "native")]
impl<F> SamplerFn for F where F: Fn(f64, f64, f64) -> f32 + Send + Sync {}

#[cfg(not(feature = "native"))]
pub trait SamplerFn: Fn(f64, f64, f64) -> f32 {}
#[cfg(not(feature = "native"))]
impl<F> SamplerFn for F where F: Fn(f64, f64, f64) -> f32 {}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the adaptive surface nets algorithm.
#[derive(Clone, Debug)]
pub struct AdaptiveMeshConfig2 {
    /// Base cell size for the coarse grid (world units, cubic cells).
    pub base_cell_size: f64,

    /// Maximum refinement depth (cell size = base_cell_size / 2^max_depth)
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

    /// Configuration for sharp edge detection.
    /// When Some, enables sharp edge detection and vertex duplication.
    /// When None (default), sharp edge detection is disabled.
    ///
    /// NOTE: Sharp edge detection is currently stubbed out (Phase 4 removed).
    /// This field is kept for API compatibility but has no effect.
    pub sharp_edge_config: Option<SharpEdgeConfig>,
}

impl Default for AdaptiveMeshConfig2 {
    fn default() -> Self {
        Self {
            base_cell_size: 0.25,
            max_depth: 4,
            vertex_refinement_iterations: 12,
            // Enable normal refinement by default - probing works well with binary samplers
            normal_sample_iterations: 12,
            normal_epsilon_frac: 0.1,
            num_threads: 0,
            sharp_edge_config: None, // Disabled by default
        }
    }
}

/// Configuration for sharp edge detection and handling.
///
/// Sharp edges occur where two smooth surfaces meet at an angle (e.g., cube edges).
/// This configuration controls how these edges are detected and handled.
///
/// NOTE: Sharp edge detection is currently stubbed out (Phase 4 removed).
/// This struct is kept for API compatibility but has no effect.
#[derive(Clone, Debug)]
pub struct SharpEdgeConfig {
    /// Angle threshold for Case 2 detection (radians).
    /// When two adjacent vertices have normals differing by more than this angle,
    /// the mesh edge between them is considered to cross a geometric sharp edge.
    /// Default: 30 degrees (π/6 ≈ 0.524 radians)
    pub angle_threshold: f64,

    /// Residual multiplier for Case 1 detection.
    /// When plane fitting residual exceeds (base_threshold * residual_multiplier),
    /// the vertex is considered to straddle a sharp edge.
    /// Default: 4.0
    pub residual_multiplier: f64,
}

impl Default for SharpEdgeConfig {
    fn default() -> Self {
        Self {
            angle_threshold: std::f64::consts::PI / 6.0, // 30 degrees
            residual_multiplier: 4.0,
        }
    }
}

// =============================================================================
// CELL AND EDGE IDENTIFIERS
// =============================================================================

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
    /// * `max_cells` - Number of cells at this depth level (per axis)
    /// * `boundary_expansion` - Number of extra cells to allow beyond the grid on each side
    ///   (e.g., 1 means allow cells from -1 to max_cells inclusive)
    pub fn neighbor(
        &self,
        dx: i32,
        dy: i32,
        dz: i32,
        max_cells: (i32, i32, i32),
        boundary_expansion: i32,
    ) -> Option<CuboidId> {
        let nx = self.x + dx;
        let ny = self.y + dy;
        let nz = self.z + dz;
        let min_valid = -boundary_expansion;
        let max_valid_x = max_cells.0 + boundary_expansion;
        let max_valid_y = max_cells.1 + boundary_expansion;
        let max_valid_z = max_cells.2 + boundary_expansion;
        if nx >= min_valid
            && nx < max_valid_x
            && ny >= min_valid
            && ny < max_valid_y
            && nz >= min_valid
            && nz < max_valid_z
        {
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

// =============================================================================
// STATISTICS
// =============================================================================

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

    /// Stage 4: Vertex refinement & normal estimation (now stubbed)
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
    pub effective_resolution: (usize, usize, usize),

    /// Stage 4.5: Sharp edge detection (when enabled) - currently stubbed
    pub stage4_5_time_secs: f64,
    /// Number of Case 1 sharp vertices (straddling edge)
    pub sharp_vertices_case1: usize,
    /// Number of Case 2 edge crossings detected
    pub sharp_edge_crossings: usize,
    /// Number of vertices added for edge crossings
    pub sharp_vertices_inserted: usize,
    /// Number of vertices duplicated for sharp normals
    pub sharp_vertices_duplicated: usize,
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
        println!("Stage 4 (Passthrough - stubbed): {:.2}ms ({:.1}%)",
            self.stage4_time_secs * 1000.0,
            self.stage4_time_secs / self.total_time_secs * 100.0);
        println!("  Samples: {}", self.stage4_samples);
        println!();
        println!("Summary:");
        println!("  Total samples: {}", self.total_samples);
        if self.total_samples > 0 {
            println!("  Avg sample time: {:.2}µs",
                self.total_time_secs * 1_000_000.0 / self.total_samples as f64);
        }
        println!("  Total vertices: {}", self.total_vertices);
        println!("  Total triangles: {}", self.total_triangles);
        println!(
            "  Effective resolution: {} x {} x {}",
            self.effective_resolution.0,
            self.effective_resolution.1,
            self.effective_resolution.2
        );
        println!("================================================");
    }
}

// =============================================================================
// OUTPUT TYPES
// =============================================================================

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

// =============================================================================
// DIAGNOSTIC TYPES (feature-gated)
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

/// High-precision reference edge information computed via extensive probing.
///
/// This is the "ground truth" computed with many probe directions and high
/// binary search iterations. Used for comparing fast detection methods.
#[cfg(feature = "edge-diagnostic")]
#[derive(Clone, Debug)]
pub struct ReferenceEdgeInfo {
    /// Whether this vertex is on a sharp edge (high confidence)
    pub is_sharp: bool,
    /// Primary surface normal (always present)
    pub normal_a: (f64, f64, f64),
    /// Secondary surface normal (present if sharp)
    pub normal_b: Option<(f64, f64, f64)>,
    /// Direction of the sharp edge (cross product of normals)
    pub edge_direction: Option<(f64, f64, f64)>,
    /// Point on the intersection line closest to the vertex
    pub intersection_point: Option<(f64, f64, f64)>,
    /// Confidence score (0.0 to 1.0, based on residuals and cluster separation)
    pub confidence: f64,
}

/// A single entry in the edge diagnostic results.
///
/// Compares a fast edge detection method against high-precision reference.
#[cfg(feature = "edge-diagnostic")]
#[derive(Clone, Debug, Default)]
pub struct EdgeDiagnosticEntry {
    /// Method being tested (e.g., "4-probe", "8-probe", "residual-only")
    pub method: String,
    /// True positive rate (detected sharp when actually sharp)
    pub true_positive_rate: f64,
    /// False positive rate (detected sharp when actually smooth)
    pub false_positive_rate: f64,
    /// Mean angular error for normal_a (degrees) - vs reference
    pub normal_a_error_mean: f64,
    /// Mean angular error for normal_b (degrees) - vs reference
    pub normal_b_error_mean: f64,
    /// Mean position error for intersection point (as fraction of cell size)
    pub position_error_mean: f64,
    /// P95 position error
    pub position_error_p95: f64,
    /// Number of samples used by this method (per vertex)
    pub samples_per_vertex: u64,
    /// Mean angular error for normal_a (degrees) - vs analytical ground truth (if available)
    pub analytical_normal_a_error: Option<f64>,
    /// Mean angular error for normal_b (degrees) - vs analytical ground truth (if available)
    pub analytical_normal_b_error: Option<f64>,
}
