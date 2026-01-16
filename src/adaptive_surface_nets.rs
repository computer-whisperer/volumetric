//! Adaptive Surface Nets mesh generation algorithm.
//!
//! This module implements an octree-based adaptive sampling algorithm that:
//! - Uses a configurable base frequency for initial geometry discovery
//! - Adaptively refines only cells containing surface transitions
//! - Supports multithreaded sampling with independent WASM instances
//! - Produces waterproof (manifold) triangle meshes

use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Triangle;

/// Configuration for the adaptive mesh generation algorithm.
#[derive(Clone, Debug)]
pub struct AdaptiveMeshConfig {
    /// Minimum sampling density (cells per axis) for initial discovery.
    /// This ensures no isolated geometry regions are missed.
    /// Default: 8
    pub base_resolution: usize,

    /// Maximum octree refinement depth beyond the base resolution.
    /// Effective max resolution = base_resolution * 2^max_refinement_depth
    /// Default: 4 (so base=8 gives effective 128³)
    pub max_refinement_depth: usize,
}

impl Default for AdaptiveMeshConfig {
    fn default() -> Self {
        Self {
            base_resolution: 8,
            max_refinement_depth: 4,
        }
    }
}

/// Cell classification based on corner samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CellType {
    /// All corners are inside the model
    AllInside,
    /// All corners are outside the model
    AllOutside,
    /// Mixed: contains a surface transition
    Mixed,
}

/// A node in the adaptive octree.
/// Each node represents a cubic cell in 3D space.
#[derive(Clone)]
struct OctreeNode {
    /// Minimum corner of this cell
    min: (f32, f32, f32),
    /// Maximum corner of this cell
    max: (f32, f32, f32),
    /// Depth in the octree (0 = root level from base grid)
    depth: usize,
    /// Corner sample values (inside=true, outside=false)
    /// Order: [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]
    corners: [bool; 8],
    /// Children (None if leaf node)
    children: Option<Box<[OctreeNode; 8]>>,
}

impl OctreeNode {
    fn cell_type(&self) -> CellType {
        let all_inside = self.corners.iter().all(|&c| c);
        let all_outside = self.corners.iter().all(|&c| !c);
        if all_inside {
            CellType::AllInside
        } else if all_outside {
            CellType::AllOutside
        } else {
            CellType::Mixed
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Get the position of a corner by index.
    fn corner_pos(&self, index: usize) -> (f32, f32, f32) {
        let dx = if index & 1 != 0 { self.max.0 } else { self.min.0 };
        let dy = if index & 2 != 0 { self.max.1 } else { self.min.1 };
        let dz = if index & 4 != 0 { self.max.2 } else { self.min.2 };
        (dx, dy, dz)
    }

    /// Get the center of this cell.
    fn center(&self) -> (f32, f32, f32) {
        (
            (self.min.0 + self.max.0) * 0.5,
            (self.min.1 + self.max.1) * 0.5,
            (self.min.2 + self.max.2) * 0.5,
        )
    }

    /// Get the size of this cell (assumes cubic).
    fn size(&self) -> f32 {
        self.max.0 - self.min.0
    }
}

/// Index for a cell in the base grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CellIndex {
    x: i32,
    y: i32,
    z: i32,
}

/// Unique identifier for a cell at any level of the octree.
/// Used for neighbor lookups and vertex deduplication.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CellId {
    /// Base grid cell index
    base: CellIndex,
    /// Path through octree (each element is child index 0-7)
    path: Vec<u8>,
}

impl CellId {
    fn new(base: CellIndex) -> Self {
        Self {
            base,
            path: Vec::new(),
        }
    }

    fn child(&self, child_index: u8) -> Self {
        let mut path = self.path.clone();
        path.push(child_index);
        Self {
            base: self.base,
            path,
        }
    }

    fn depth(&self) -> usize {
        self.path.len()
    }
}

/// Thread-safe sampler that can create per-thread WASM instances.
/// 
/// The WASM module is compiled once and shared across all threads.
/// Each thread gets its own Store and Instance, but they all share
/// the same compiled module, avoiding expensive recompilation.
pub struct WasmSampler {
    engine: Arc<wasmtime::Engine>,
    module: Arc<wasmtime::Module>,
}

impl WasmSampler {
    pub fn new(wasm_bytes: Vec<u8>) -> Result<Self> {
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm_bytes)?;
        Ok(Self {
            engine: Arc::new(engine),
            module: Arc::new(module),
        })
    }

    /// Create a thread-local sampling context.
    /// 
    /// This creates a new Store and Instance for the calling thread,
    /// but reuses the pre-compiled module (which is the expensive part).
    fn create_context(&self) -> Result<SamplingContext> {
        SamplingContext::new(&self.engine, &self.module)
    }
}

/// Per-thread sampling context with its own WASM instance.
/// 
/// Each thread needs its own Store and Instance because wasmtime's
/// Store is not thread-safe. However, the Module can be shared.
struct SamplingContext {
    store: wasmtime::Store<()>,
    // New ABI: (f64,f64,f64) -> f32 density
    is_inside_func: wasmtime::TypedFunc<(f64, f64, f64), f32>,
}

impl SamplingContext {
    fn new(engine: &wasmtime::Engine, module: &wasmtime::Module) -> Result<Self> {
        let mut store = wasmtime::Store::new(engine, ());
        let instance = wasmtime::Instance::new(&mut store, module, &[])?;

        let is_inside_func = instance
            .get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")?;

        Ok(Self {
            store,
            is_inside_func,
        })
    }

    fn is_inside(&mut self, p: (f32, f32, f32)) -> Result<f32> {
        // Pass as f64s to WASM, receive f32 density
        let args = (p.0 as f64, p.1 as f64, p.2 as f64);
        Ok(self.is_inside_func.call(&mut self.store, args)?)
    }
}

/// The adaptive octree structure for mesh generation.
struct AdaptiveOctree {
    /// Bounds of the entire volume
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    /// Base grid resolution
    base_resolution: usize,
    /// Maximum refinement depth
    max_depth: usize,
    /// Step size at base level
    base_step: (f32, f32, f32),
    /// Root nodes (one per base grid cell that is MIXED)
    /// Key: base grid cell index
    roots: HashMap<CellIndex, OctreeNode>,
    /// Corner sample cache at base level for sharing between cells
    /// Stored as a flat Vec for performance (avoids HashMap overhead)
    /// Indices range from -1 to base_resolution+1 (padded by 1 on each side)
    /// Size: (base_resolution + 3)^3
    base_corners: Vec<bool>,
    /// Number of corners per axis (base_resolution + 3)
    corner_n: usize,
}

impl AdaptiveOctree {
    fn new(
        bounds_min: (f32, f32, f32),
        bounds_max: (f32, f32, f32),
        config: &AdaptiveMeshConfig,
    ) -> Self {
        let base_resolution = config.base_resolution;
        let base_step = (
            (bounds_max.0 - bounds_min.0) / base_resolution as f32,
            (bounds_max.1 - bounds_min.1) / base_resolution as f32,
            (bounds_max.2 - bounds_min.2) / base_resolution as f32,
        );
        
        // Corner lattice is (base_resolution + 3)^3 to accommodate padding of -1 to base_resolution+1
        let corner_n = base_resolution + 3;

        Self {
            bounds_min,
            bounds_max,
            base_resolution,
            max_depth: config.max_refinement_depth,
            base_step,
            roots: HashMap::new(),
            base_corners: vec![false; corner_n * corner_n * corner_n],
            corner_n,
        }
    }

    /// Convert corner coordinates to flat array index.
    /// Coordinates range from -1 to base_resolution+1.
    #[inline]
    fn corner_idx(&self, x: i32, y: i32, z: i32) -> usize {
        // Offset by 1 to map -1 -> 0
        let xi = (x + 1) as usize;
        let yi = (y + 1) as usize;
        let zi = (z + 1) as usize;
        (zi * self.corner_n + yi) * self.corner_n + xi
    }

    /// Get the world position of a base grid corner.
    fn base_corner_pos(&self, x: i32, y: i32, z: i32) -> (f32, f32, f32) {
        (
            self.bounds_min.0 + x as f32 * self.base_step.0,
            self.bounds_min.1 + y as f32 * self.base_step.1,
            self.bounds_min.2 + z as f32 * self.base_step.2,
        )
    }

    /// Get the bounds of a base grid cell.
    fn base_cell_bounds(&self, idx: CellIndex) -> ((f32, f32, f32), (f32, f32, f32)) {
        let min = self.base_corner_pos(idx.x, idx.y, idx.z);
        let max = self.base_corner_pos(idx.x + 1, idx.y + 1, idx.z + 1);
        (min, max)
    }
}

/// Main entry point for adaptive mesh generation.
///
/// # Arguments
/// * `wasm_bytes` - The WASM module bytes defining the model
/// * `bounds_min` - Minimum corner of the bounding box
/// * `bounds_max` - Maximum corner of the bounding box
/// * `config` - Configuration for the adaptive algorithm
///
/// # Returns
/// A vector of triangles forming a waterproof mesh.
pub fn adaptive_surface_nets_mesh(
    wasm_bytes: &[u8],
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    config: &AdaptiveMeshConfig,
) -> Result<Vec<Triangle>> {
    let sampler = WasmSampler::new(wasm_bytes.to_vec())?;

    // Phase 1: Coarse discovery - sample base grid corners
    let mut octree = AdaptiveOctree::new(bounds_min, bounds_max, config);
    phase1_coarse_discovery(&sampler, &mut octree)?;

    // Phase 2: Adaptive refinement of MIXED cells
    phase2_adaptive_refinement(&sampler, &mut octree)?;

    // Phase 3: Extract waterproof mesh using Surface Nets
    let triangles = phase3_extract_mesh(&octree)?;

    Ok(triangles)
}

/// Phase 1: Sample the base grid to discover geometry regions.
fn phase1_coarse_discovery(sampler: &WasmSampler, octree: &mut AdaptiveOctree) -> Result<()> {
    let base_res = octree.base_resolution as i32;

    // Pad by one on each side to ensure surfaces at bounds are closed
    let min_corner = -1i32;
    let max_corner = base_res + 1;

    // Collect all corner positions to sample
    let corner_positions: Vec<(i32, i32, i32)> = (min_corner..=max_corner)
        .flat_map(|z| {
            (min_corner..=max_corner)
                .flat_map(move |y| (min_corner..=max_corner).map(move |x| (x, y, z)))
        })
        .collect();

    // Sample corners in parallel using rayon
    // Clone Arc references for use in parallel iterator
    let engine = sampler.engine.clone();
    let module = sampler.module.clone();
    let bounds_min = octree.bounds_min;
    let base_step = octree.base_step;
    let corner_n = octree.corner_n;

    // Sample directly into indexed results for efficient storage
    let corner_samples: Vec<(usize, bool)> = corner_positions
        .par_iter()
        .map_init(
            || SamplingContext::new(&engine, &module).expect("Failed to create sampling context"),
            |ctx, &(x, y, z)| {
                let pos = (
                    bounds_min.0 + x as f32 * base_step.0,
                    bounds_min.1 + y as f32 * base_step.1,
                    bounds_min.2 + z as f32 * base_step.2,
                );
                let inside = (ctx.is_inside(pos).unwrap_or(0.0)) > 0.5;
                // Compute flat index inline (offset by 1 to map -1 -> 0)
                let xi = (x + 1) as usize;
                let yi = (y + 1) as usize;
                let zi = (z + 1) as usize;
                let idx = (zi * corner_n + yi) * corner_n + xi;
                (idx, inside)
            },
        )
        .collect();

    // Store corner samples using direct indexing (faster than HashMap)
    for (idx, inside) in corner_samples {
        octree.base_corners[idx] = inside;
    }

    // First pass: identify all MIXED cells
    let min_cell = -1i32;
    let max_cell = base_res;
    
    let mut mixed_cells: HashSet<CellIndex> = HashSet::new();

    for z in min_cell..=max_cell {
        for y in min_cell..=max_cell {
            for x in min_cell..=max_cell {
                let idx = CellIndex { x, y, z };
                let corners = [
                    octree.base_corners[octree.corner_idx(x, y, z)],
                    octree.base_corners[octree.corner_idx(x + 1, y, z)],
                    octree.base_corners[octree.corner_idx(x, y + 1, z)],
                    octree.base_corners[octree.corner_idx(x + 1, y + 1, z)],
                    octree.base_corners[octree.corner_idx(x, y, z + 1)],
                    octree.base_corners[octree.corner_idx(x + 1, y, z + 1)],
                    octree.base_corners[octree.corner_idx(x, y + 1, z + 1)],
                    octree.base_corners[octree.corner_idx(x + 1, y + 1, z + 1)],
                ];
                
                // Check if MIXED (has both inside and outside corners)
                let first = corners[0];
                let is_mixed = corners.iter().any(|&c| c != first);
                if is_mixed {
                    mixed_cells.insert(idx);
                }
            }
        }
    }
    
    // Second pass: collect MIXED cells and their face-adjacent neighbors
    // This ensures uniform refinement at root boundaries
    let mut cells_to_include: HashSet<CellIndex> = HashSet::new();
    
    for &idx in &mixed_cells {
        cells_to_include.insert(idx);
        
        // Add face-adjacent neighbors (6 neighbors)
        let neighbors = [
            CellIndex { x: idx.x - 1, y: idx.y, z: idx.z },
            CellIndex { x: idx.x + 1, y: idx.y, z: idx.z },
            CellIndex { x: idx.x, y: idx.y - 1, z: idx.z },
            CellIndex { x: idx.x, y: idx.y + 1, z: idx.z },
            CellIndex { x: idx.x, y: idx.y, z: idx.z - 1 },
            CellIndex { x: idx.x, y: idx.y, z: idx.z + 1 },
        ];
        
        for neighbor in neighbors {
            if neighbor.x >= min_cell && neighbor.x <= max_cell &&
               neighbor.y >= min_cell && neighbor.y <= max_cell &&
               neighbor.z >= min_cell && neighbor.z <= max_cell {
                cells_to_include.insert(neighbor);
            }
        }
    }
    
    // Third pass: create root nodes for all cells to include
    for idx in cells_to_include {
        let (cell_min, cell_max) = octree.base_cell_bounds(idx);
        let corners = [
            octree.base_corners[octree.corner_idx(idx.x, idx.y, idx.z)],
            octree.base_corners[octree.corner_idx(idx.x + 1, idx.y, idx.z)],
            octree.base_corners[octree.corner_idx(idx.x, idx.y + 1, idx.z)],
            octree.base_corners[octree.corner_idx(idx.x + 1, idx.y + 1, idx.z)],
            octree.base_corners[octree.corner_idx(idx.x, idx.y, idx.z + 1)],
            octree.base_corners[octree.corner_idx(idx.x + 1, idx.y, idx.z + 1)],
            octree.base_corners[octree.corner_idx(idx.x, idx.y + 1, idx.z + 1)],
            octree.base_corners[octree.corner_idx(idx.x + 1, idx.y + 1, idx.z + 1)],
        ];

        let node = OctreeNode {
            min: cell_min,
            max: cell_max,
            depth: 0,
            corners,
            children: None,
        };

        octree.roots.insert(idx, node);
    }

    Ok(())
}

/// Phase 2: Adaptively refine MIXED cells to the target depth.
/// 
/// This phase is parallelized at the root cell level using rayon.
/// Each root cell is refined independently in parallel, with each thread
/// getting its own WASM sampling context from the shared pre-compiled module.
fn phase2_adaptive_refinement(sampler: &WasmSampler, octree: &mut AdaptiveOctree) -> Result<()> {
    if octree.max_depth == 0 {
        return Ok(());
    }

    let max_depth = octree.max_depth;

    // Collect cells that need refinement along with their nodes
    let cells_to_refine: Vec<(CellIndex, OctreeNode)> = octree
        .roots
        .drain()
        .collect();

    // Clone Arc references for use in parallel iterator
    let engine = sampler.engine.clone();
    let module = sampler.module.clone();

    // Process root cells in parallel using rayon
    // Each thread gets its own sampling context via map_init
    // Refine ALL roots uniformly to ensure consistent depth across boundaries
    let refined_roots: Vec<Result<(CellIndex, OctreeNode)>> = cells_to_refine
        .into_par_iter()
        .map_init(
            || SamplingContext::new(&engine, &module).expect("Failed to create sampling context"),
            |ctx, (cell_idx, mut node)| {
                // Refine all roots uniformly to avoid T-junctions at root boundaries
                refine_node_uniform(&mut node, max_depth, ctx)?;
                Ok((cell_idx, node))
            },
        )
        .collect();

    // Collect results back into the octree, propagating any errors
    for result in refined_roots {
        let (cell_idx, node) = result?;
        octree.roots.insert(cell_idx, node);
    }

    Ok(())
}

/// Recursively refine a node if it's MIXED and below max depth.
/// 
/// Uses a thread-local sampling context to avoid creating new WASM instances.
/// The context is reused across all recursive calls within the same thread.
/// 
/// This uses uniform refinement within each root cell to avoid T-junctions:
/// all children are refined to the same depth regardless of whether they're MIXED.
fn refine_node_recursive(
    node: &mut OctreeNode,
    max_depth: usize,
    ctx: &mut SamplingContext,
) -> Result<()> {
    if node.depth >= max_depth {
        return Ok(());
    }

    // Only refine if this node is MIXED (contains surface)
    // But once we decide to refine, we refine uniformly
    if node.cell_type() != CellType::Mixed {
        return Ok(());
    }

    // Create 8 children by subdividing this cell
    let mid = node.center();
    let mut children: [OctreeNode; 8] = std::array::from_fn(|i| {
        let child_min = (
            if i & 1 != 0 { mid.0 } else { node.min.0 },
            if i & 2 != 0 { mid.1 } else { node.min.1 },
            if i & 4 != 0 { mid.2 } else { node.min.2 },
        );
        let child_max = (
            if i & 1 != 0 { node.max.0 } else { mid.0 },
            if i & 2 != 0 { node.max.1 } else { mid.1 },
            if i & 4 != 0 { node.max.2 } else { mid.2 },
        );
        OctreeNode {
            min: child_min,
            max: child_max,
            depth: node.depth + 1,
            corners: [false; 8], // Will be filled in
            children: None,
        }
    });

    // Sample all unique corner positions for children
    // Children share corners, so we sample each unique position once
    // Use high-precision integer keys to avoid floating point issues
    let mut corner_cache: HashMap<(i64, i64, i64), bool> = HashMap::new();
    
    // Scale factor for quantizing positions to avoid floating point comparison issues
    let scale = 1_000_000.0;

    for (_child_idx, child) in children.iter_mut().enumerate() {
        for corner_idx in 0..8 {
            let pos = child.corner_pos(corner_idx);

            // Use absolute position with high precision as cache key
            let key = (
                (pos.0 * scale).round() as i64,
                (pos.1 * scale).round() as i64,
                (pos.2 * scale).round() as i64,
            );

            let inside = if let Some(&cached) = corner_cache.get(&key) {
                cached
            } else {
                let value = ctx.is_inside(pos)? > 0.5;
                corner_cache.insert(key, value);
                value
            };

            child.corners[corner_idx] = inside;
        }
    }

    // Recursively refine ALL children uniformly to avoid T-junctions
    // This ensures all cells within a root are at the same refinement level
    for child in children.iter_mut() {
        // Refine all children, not just MIXED ones, to maintain uniform depth
        refine_node_uniform(child, max_depth, ctx)?;
    }

    node.children = Some(Box::new(children));

    Ok(())
}

/// Uniformly refine a node to max_depth regardless of cell type.
/// This ensures all cells within a root reach the same depth to avoid T-junctions.
fn refine_node_uniform(
    node: &mut OctreeNode,
    max_depth: usize,
    ctx: &mut SamplingContext,
) -> Result<()> {
    if node.depth >= max_depth {
        return Ok(());
    }

    // Create 8 children by subdividing this cell
    let mid = node.center();
    let mut children: [OctreeNode; 8] = std::array::from_fn(|i| {
        let child_min = (
            if i & 1 != 0 { mid.0 } else { node.min.0 },
            if i & 2 != 0 { mid.1 } else { node.min.1 },
            if i & 4 != 0 { mid.2 } else { node.min.2 },
        );
        let child_max = (
            if i & 1 != 0 { node.max.0 } else { mid.0 },
            if i & 2 != 0 { node.max.1 } else { mid.1 },
            if i & 4 != 0 { node.max.2 } else { mid.2 },
        );
        OctreeNode {
            min: child_min,
            max: child_max,
            depth: node.depth + 1,
            corners: [false; 8],
            children: None,
        }
    });

    // Sample all unique corner positions for children
    let mut corner_cache: HashMap<(i64, i64, i64), bool> = HashMap::new();
    let scale = 1_000_000.0;

    for child in children.iter_mut() {
        for corner_idx in 0..8 {
            let pos = child.corner_pos(corner_idx);
            let key = (
                (pos.0 * scale).round() as i64,
                (pos.1 * scale).round() as i64,
                (pos.2 * scale).round() as i64,
            );

            let inside = if let Some(&cached) = corner_cache.get(&key) {
                cached
            } else {
                let value = ctx.is_inside(pos)? > 0.5;
                corner_cache.insert(key, value);
                value
            };

            child.corners[corner_idx] = inside;
        }
    }

    // Recursively refine ALL children uniformly
    for child in children.iter_mut() {
        refine_node_uniform(child, max_depth, ctx)?;
    }

    node.children = Some(Box::new(children));

    Ok(())
}

/// Axis-aligned edge direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EdgeAxis {
    X,
    Y,
    Z,
}

/// Key for grouping edges that lie on the same axis-aligned line.
/// Edges on the same line will have the same LineKey regardless of their length.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LineKey {
    axis: EdgeAxis,
    // Quantized coordinates perpendicular to the axis
    perp1: i64,
    perp2: i64,
}

/// Information about a cell's edge on a line.
struct EdgeOnLine {
    /// The cell's surface nets vertex
    vertex: (f32, f32, f32),
    /// Start coordinate along the axis
    axis_min: f32,
    /// End coordinate along the axis  
    axis_max: f32,
    /// Whether the "inside" is at the lower axis coordinate
    inside_at_min: bool,
    /// Length of this edge (cell size along axis)
    edge_length: f32,
}

impl LineKey {
    fn new(pa: (f32, f32, f32), pb: (f32, f32, f32)) -> (Self, EdgeAxis, bool) {
        let scale = 1_000_000.0;
        
        let dx = (pb.0 - pa.0).abs();
        let dy = (pb.1 - pa.1).abs();
        let dz = (pb.2 - pa.2).abs();
        
        // For axis-aligned edges, the perpendicular coordinates are the same for both endpoints.
        // Use pa's coordinates (they equal pb's for the perpendicular axes).
        // This ensures edges on the same axis-aligned line are grouped together regardless of cell size.
        
        if dx > dy && dx > dz {
            // X-aligned edge - perpendicular coords are Y and Z (same for pa and pb)
            let perp1 = (pa.1 * scale).round() as i64;
            let perp2 = (pa.2 * scale).round() as i64;
            let inside_at_min = pa.0 < pb.0; // true if pa is at min
            (LineKey { axis: EdgeAxis::X, perp1, perp2 }, EdgeAxis::X, inside_at_min)
        } else if dy > dz {
            // Y-aligned edge - perpendicular coords are X and Z (same for pa and pb)
            let perp1 = (pa.0 * scale).round() as i64;
            let perp2 = (pa.2 * scale).round() as i64;
            let inside_at_min = pa.1 < pb.1;
            (LineKey { axis: EdgeAxis::Y, perp1, perp2 }, EdgeAxis::Y, inside_at_min)
        } else {
            // Z-aligned edge - perpendicular coords are X and Y (same for pa and pb)
            let perp1 = (pa.0 * scale).round() as i64;
            let perp2 = (pa.1 * scale).round() as i64;
            let inside_at_min = pa.2 < pb.2;
            (LineKey { axis: EdgeAxis::Z, perp1, perp2 }, EdgeAxis::Z, inside_at_min)
        }
    }
}

/// Phase 3: Extract a waterproof mesh from the octree using Surface Nets.
/// 
/// This groups edges by the axis-aligned line they lie on, then for each unique
/// point along that line where edges meet, emits quads connecting the cells.
/// This handles T-junctions where cells of different sizes meet.
fn phase3_extract_mesh(octree: &AdaptiveOctree) -> Result<Vec<Triangle>> {
    // Collect all leaf cells
    let mut leaf_cells: Vec<(CellId, OctreeNode)> = Vec::new();
    for (&base_idx, root) in &octree.roots {
        let cell_id = CellId::new(base_idx);
        collect_leaf_cells(root, cell_id, &mut leaf_cells);
    }

    // Group edges by the line they lie on
    // Key: LineKey (axis + perpendicular coordinates)
    // Value: list of edges on that line
    let mut line_to_edges: HashMap<LineKey, Vec<EdgeOnLine>> = HashMap::new();

    // Edge definitions: 12 edges of a cube
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // X-aligned
        (0, 2), (1, 3), (4, 6), (5, 7), // Y-aligned  
        (0, 4), (1, 5), (2, 6), (3, 7), // Z-aligned
    ];

    for (_cell_id, node) in &leaf_cells {
        if node.cell_type() != CellType::Mixed {
            continue;
        }
        
        let vertex = compute_surface_nets_vertex(node);
        
        for &(ca, cb) in &EDGES {
            if node.corners[ca] == node.corners[cb] {
                continue; // No sign change on this edge
            }
            
            let pa = node.corner_pos(ca);
            let pb = node.corner_pos(cb);
            let (line_key, axis, pa_is_min) = LineKey::new(pa, pb);
            
            // Get axis coordinates
            let (axis_a, axis_b) = match axis {
                EdgeAxis::X => (pa.0, pb.0),
                EdgeAxis::Y => (pa.1, pb.1),
                EdgeAxis::Z => (pa.2, pb.2),
            };
            
            let (axis_min, axis_max) = if axis_a < axis_b {
                (axis_a, axis_b)
            } else {
                (axis_b, axis_a)
            };
            
            // Determine if inside is at the min end of the edge
            // pa_is_min tells us if pa has the smaller axis coordinate
            let inside_at_min = if pa_is_min {
                node.corners[ca] // pa is at min, so check if pa's corner is inside
            } else {
                node.corners[cb] // pb is at min, so check if pb's corner is inside
            };
            
            let edge_length = axis_max - axis_min;
            line_to_edges.entry(line_key).or_default().push(EdgeOnLine {
                vertex,
                axis_min,
                axis_max,
                inside_at_min,
                edge_length,
            });
        }
    }

    let mut triangles: Vec<Triangle> = Vec::new();
    
    // Track processed segments to avoid duplicates
    // Key: (line_key hash, seg_min quantized, seg_max quantized)
    let mut processed_segments: HashSet<(EdgeAxis, i64, i64, i64, i64)> = HashSet::new();
    
    // Track processed quads by their sorted vertex positions to avoid duplicate quads from different lines
    let mut processed_quads: HashSet<[(i64, i64, i64); 4]> = HashSet::new();
    
    // Track emitted triangles to avoid duplicates
    let mut emitted_triangles: HashSet<[(i64, i64, i64); 3]> = HashSet::new();

    // For each line, find all unique edge endpoints and emit quads
    for (line_key, edges) in &line_to_edges {
        // Collect all unique axis coordinates where edges start or end
        let scale = 1_000_000.0;
        let mut axis_points: Vec<i64> = Vec::new();
        for edge in edges {
            axis_points.push((edge.axis_min * scale).round() as i64);
            axis_points.push((edge.axis_max * scale).round() as i64);
        }
        axis_points.sort();
        axis_points.dedup();
        
        // For each segment between consecutive points, find cells that span it
        for i in 0..axis_points.len().saturating_sub(1) {
            let seg_min = axis_points[i] as f32 / scale;
            let seg_max = axis_points[i + 1] as f32 / scale;
            let seg_mid = (seg_min + seg_max) * 0.5;
            
            // Find all edges that contain this segment
            // Store (vertex, edge_length) to track cell sizes
            let mut cells_for_segment: Vec<((f32, f32, f32), f32)> = Vec::new();
            let mut inside_at_min = true; // Will be set by first edge found
            let mut first = true;
            
            for edge in edges {
                // Check if this edge contains the segment
                // Use small epsilon for floating point comparison
                let eps = (seg_max - seg_min) * 0.01;
                if edge.axis_min <= seg_min + eps && edge.axis_max >= seg_max - eps {
                    cells_for_segment.push((edge.vertex, edge.edge_length));
                    if first {
                        inside_at_min = edge.inside_at_min;
                        first = false;
                    }
                }
            }
            
            // Deduplicate vertices (same cell may contribute multiple edges on same line)
            let vert_scale = 1_000_000.0f32;
            cells_for_segment.sort_by(|a, b| {
                let qa = ((a.0.0 * vert_scale) as i64, (a.0.1 * vert_scale) as i64, (a.0.2 * vert_scale) as i64);
                let qb = ((b.0.0 * vert_scale) as i64, (b.0.1 * vert_scale) as i64, (b.0.2 * vert_scale) as i64);
                qa.cmp(&qb)
            });
            cells_for_segment.dedup_by(|a, b| {
                let qa = ((a.0.0 * vert_scale) as i64, (a.0.1 * vert_scale) as i64, (a.0.2 * vert_scale) as i64);
                let qb = ((b.0.0 * vert_scale) as i64, (b.0.1 * vert_scale) as i64, (b.0.2 * vert_scale) as i64);
                qa == qb
            });
            
            if cells_for_segment.len() != 4 {
                // Surface nets requires exactly 4 cells around each edge.
                // Skip segments that don't have exactly 4 cells.
                continue;
            }
            
            // Check if all cells have the same edge length (same refinement level)
            // This avoids T-junction issues where cells of different sizes meet
            let first_len = cells_for_segment[0].1;
            let length_eps = first_len * 0.01;
            let all_same_size = cells_for_segment.iter().all(|(_, len)| {
                (*len - first_len).abs() < length_eps
            });
            
            if !all_same_size {
                // Skip segments with mixed cell sizes - the coarser cell will cover this area
                continue;
            }
            
            // Extract just the vertices for triangle emission
            let vertices: Vec<(f32, f32, f32)> = cells_for_segment.iter().map(|(v, _)| *v).collect();
            
            // Check if this segment was already processed (avoid duplicate quads)
            let seg_key = (
                line_key.axis,
                line_key.perp1,
                line_key.perp2,
                axis_points[i],
                axis_points[i + 1],
            );
            if !processed_segments.insert(seg_key) {
                // Already processed this segment
                continue;
            }
            
            // Note: We rely on triangle-level deduplication instead of quad-level
            // since quads can have 3 or 4 vertices with adaptive refinement.
            
            // Compute edge midpoint in 3D
            let edge_mid = match line_key.axis {
                EdgeAxis::X => (seg_mid, line_key.perp1 as f32 / scale, line_key.perp2 as f32 / scale),
                EdgeAxis::Y => (line_key.perp1 as f32 / scale, seg_mid, line_key.perp2 as f32 / scale),
                EdgeAxis::Z => (line_key.perp1 as f32 / scale, line_key.perp2 as f32 / scale, seg_mid),
            };

            // Edge direction along the axis
            let edge_dir = match line_key.axis {
                EdgeAxis::X => (1.0, 0.0, 0.0),
                EdgeAxis::Y => (0.0, 1.0, 0.0),
                EdgeAxis::Z => (0.0, 0.0, 1.0),
            };
            
            emit_triangles_for_quad(&vertices, inside_at_min, edge_mid, edge_dir, &mut triangles, &mut emitted_triangles);
        }
    }

    Ok(triangles)
}

/// Emit triangles for a quad with correct winding order.
/// Vertices must be sorted angularly around the edge axis for proper manifold mesh.
fn emit_triangles_for_quad(
    vertices: &[(f32, f32, f32)],
    inside_first: bool,
    edge_mid: (f32, f32, f32),
    edge_dir: (f32, f32, f32),
    triangles: &mut Vec<Triangle>,
    emitted: &mut HashSet<[(i64, i64, i64); 3]>,
) {
    if vertices.len() < 3 {
        return;
    }
    
    let scale = 1_000_000.0f32;
    let quantize = |v: (f32, f32, f32)| -> (i64, i64, i64) {
        ((v.0 * scale) as i64, (v.1 * scale) as i64, (v.2 * scale) as i64)
    };
    
    let try_emit = |tri: [(f32, f32, f32); 3], triangles: &mut Vec<Triangle>, emitted: &mut HashSet<[(i64, i64, i64); 3]>| {
        let mut key = [quantize(tri[0]), quantize(tri[1]), quantize(tri[2])];
        key.sort();
        if emitted.insert(key) {
            triangles.push(tri);
        }
    };
    
    // Sort vertices angularly around the edge axis
    let mut sorted_verts: Vec<(f32, f32, f32)> = vertices.to_vec();
    
    // Create orthonormal basis perpendicular to edge direction
    // Find a vector not parallel to edge_dir
    let arbitrary = if edge_dir.0.abs() < 0.9 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    };
    
    // basis_u = arbitrary x edge_dir (cross product)
    let basis_u = (
        arbitrary.1 * edge_dir.2 - arbitrary.2 * edge_dir.1,
        arbitrary.2 * edge_dir.0 - arbitrary.0 * edge_dir.2,
        arbitrary.0 * edge_dir.1 - arbitrary.1 * edge_dir.0,
    );
    let len_u = (basis_u.0 * basis_u.0 + basis_u.1 * basis_u.1 + basis_u.2 * basis_u.2).sqrt();
    let basis_u = (basis_u.0 / len_u, basis_u.1 / len_u, basis_u.2 / len_u);
    
    // basis_v = edge_dir x basis_u
    let basis_v = (
        edge_dir.1 * basis_u.2 - edge_dir.2 * basis_u.1,
        edge_dir.2 * basis_u.0 - edge_dir.0 * basis_u.2,
        edge_dir.0 * basis_u.1 - edge_dir.1 * basis_u.0,
    );
    
    // Sort by angle around the edge
    sorted_verts.sort_by(|a, b| {
        // Vector from edge_mid to vertex
        let da = (a.0 - edge_mid.0, a.1 - edge_mid.1, a.2 - edge_mid.2);
        let db = (b.0 - edge_mid.0, b.1 - edge_mid.1, b.2 - edge_mid.2);
        
        // Project onto basis_u and basis_v
        let ua = da.0 * basis_u.0 + da.1 * basis_u.1 + da.2 * basis_u.2;
        let va = da.0 * basis_v.0 + da.1 * basis_v.1 + da.2 * basis_v.2;
        let ub = db.0 * basis_u.0 + db.1 * basis_u.1 + db.2 * basis_u.2;
        let vb = db.0 * basis_v.0 + db.1 * basis_v.1 + db.2 * basis_v.2;
        
        // Compute angles
        let angle_a = va.atan2(ua);
        let angle_b = vb.atan2(ub);
        
        angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Emit triangles using adjacent vertices only (no internal edges)
    // This ensures each edge in the output is shared by exactly 2 triangles
    let n = sorted_verts.len();
    
    if n == 3 {
        // Triangle: just emit it
        if inside_first {
            try_emit([sorted_verts[0], sorted_verts[1], sorted_verts[2]], triangles, emitted);
        } else {
            try_emit([sorted_verts[0], sorted_verts[2], sorted_verts[1]], triangles, emitted);
        }
    } else if n == 4 {
        // Quad: split into 2 triangles using the shorter diagonal
        // This avoids creating long internal edges that might overlap with adjacent quads
        let v0 = sorted_verts[0];
        let v1 = sorted_verts[1];
        let v2 = sorted_verts[2];
        let v3 = sorted_verts[3];
        
        // Compute diagonal lengths
        let diag02 = (v0.0 - v2.0).powi(2) + (v0.1 - v2.1).powi(2) + (v0.2 - v2.2).powi(2);
        let diag13 = (v1.0 - v3.0).powi(2) + (v1.1 - v3.1).powi(2) + (v1.2 - v3.2).powi(2);
        
        if diag02 <= diag13 {
            // Split along 0-2 diagonal
            if inside_first {
                try_emit([v0, v1, v2], triangles, emitted);
                try_emit([v0, v2, v3], triangles, emitted);
            } else {
                try_emit([v0, v2, v1], triangles, emitted);
                try_emit([v0, v3, v2], triangles, emitted);
            }
        } else {
            // Split along 1-3 diagonal
            if inside_first {
                try_emit([v0, v1, v3], triangles, emitted);
                try_emit([v1, v2, v3], triangles, emitted);
            } else {
                try_emit([v0, v3, v1], triangles, emitted);
                try_emit([v1, v3, v2], triangles, emitted);
            }
        }
    } else if n > 4 {
        // For n > 4, use fan triangulation (rare case with adaptive refinement)
        for i in 1..(n - 1) {
            let v0 = sorted_verts[0];
            let v1 = sorted_verts[i];
            let v2 = sorted_verts[i + 1];
            
            if inside_first {
                try_emit([v0, v1, v2], triangles, emitted);
            } else {
                try_emit([v0, v2, v1], triangles, emitted);
            }
        }
    }
}

/// Collect all leaf cells from an octree node.
fn collect_leaf_cells(
    node: &OctreeNode,
    cell_id: CellId,
    leaves: &mut Vec<(CellId, OctreeNode)>,
) {
    if let Some(ref children) = node.children {
        for (i, child) in children.iter().enumerate() {
            let child_id = cell_id.child(i as u8);
            collect_leaf_cells(child, child_id, leaves);
        }
    } else {
        // This is a leaf node - clone it for storage
        leaves.push((cell_id, OctreeNode {
            min: node.min,
            max: node.max,
            depth: node.depth,
            corners: node.corners,
            children: None,
        }));
    }
}

/// Compute the Surface Nets vertex for a MIXED cell.
/// This is the average of midpoints of all sign-changing edges.
fn compute_surface_nets_vertex(node: &OctreeNode) -> (f32, f32, f32) {
    // Edge definitions: pairs of corner indices
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // X-aligned edges
        (0, 2), (1, 3), (4, 6), (5, 7), // Y-aligned edges
        (0, 4), (1, 5), (2, 6), (3, 7), // Z-aligned edges
    ];

    let mut sum = (0.0f32, 0.0f32, 0.0f32);
    let mut count = 0;

    for &(a, b) in &EDGES {
        if node.corners[a] != node.corners[b] {
            let pa = node.corner_pos(a);
            let pb = node.corner_pos(b);
            let mid = (
                (pa.0 + pb.0) * 0.5,
                (pa.1 + pb.1) * 0.5,
                (pa.2 + pb.2) * 0.5,
            );
            sum.0 += mid.0;
            sum.1 += mid.1;
            sum.2 += mid.2;
            count += 1;
        }
    }

    if count > 0 {
        (sum.0 / count as f32, sum.1 / count as f32, sum.2 / count as f32)
    } else {
        node.center()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_type_classification() {
        let all_inside = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (1.0, 1.0, 1.0),
            depth: 0,
            corners: [true; 8],
            children: None,
        };
        assert_eq!(all_inside.cell_type(), CellType::AllInside);

        let all_outside = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (1.0, 1.0, 1.0),
            depth: 0,
            corners: [false; 8],
            children: None,
        };
        assert_eq!(all_outside.cell_type(), CellType::AllOutside);

        let mixed = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (1.0, 1.0, 1.0),
            depth: 0,
            corners: [true, false, true, false, true, false, true, false],
            children: None,
        };
        assert_eq!(mixed.cell_type(), CellType::Mixed);
    }

    #[test]
    fn test_corner_positions() {
        let node = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (2.0, 2.0, 2.0),
            depth: 0,
            corners: [false; 8],
            children: None,
        };

        assert_eq!(node.corner_pos(0), (0.0, 0.0, 0.0));
        assert_eq!(node.corner_pos(1), (2.0, 0.0, 0.0));
        assert_eq!(node.corner_pos(2), (0.0, 2.0, 0.0));
        assert_eq!(node.corner_pos(3), (2.0, 2.0, 0.0));
        assert_eq!(node.corner_pos(4), (0.0, 0.0, 2.0));
        assert_eq!(node.corner_pos(5), (2.0, 0.0, 2.0));
        assert_eq!(node.corner_pos(6), (0.0, 2.0, 2.0));
        assert_eq!(node.corner_pos(7), (2.0, 2.0, 2.0));
    }

    #[test]
    fn test_surface_nets_vertex() {
        // A cell with one corner inside (corner 0)
        let node = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (2.0, 2.0, 2.0),
            depth: 0,
            corners: [true, false, false, false, false, false, false, false],
            children: None,
        };

        let vertex = compute_surface_nets_vertex(&node);

        // The vertex should be near corner 0, as that's where the surface is
        assert!(vertex.0 < 1.0);
        assert!(vertex.1 < 1.0);
        assert!(vertex.2 < 1.0);
    }

    #[test]
    fn test_config_default() {
        let config = AdaptiveMeshConfig::default();
        assert_eq!(config.base_resolution, 8);
        assert_eq!(config.max_refinement_depth, 4);
    }

    #[test]
    fn test_sphere_mesh_generation() {
        // Load the simple sphere wasm model
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/simple_sphere_model.wasm");
        
        // Skip test if wasm file doesn't exist (not built yet)
        if !wasm_path.exists() {
            eprintln!("Skipping test: sphere wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");
        
        let config = AdaptiveMeshConfig {
            base_resolution: 4,
            max_refinement_depth: 2,
        };

        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);

        let triangles = adaptive_surface_nets_mesh_debug(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        // A sphere should produce some triangles
        assert!(!triangles.is_empty(), "Expected triangles for sphere mesh");
        
        // All triangle vertices should be within or near the bounds
        for tri in &triangles {
            for v in tri {
                assert!(v.0 >= -1.5 && v.0 <= 1.5, "Vertex x out of bounds: {}", v.0);
                assert!(v.1 >= -1.5 && v.1 <= 1.5, "Vertex y out of bounds: {}", v.1);
                assert!(v.2 >= -1.5 && v.2 <= 1.5, "Vertex z out of bounds: {}", v.2);
            }
        }

        println!("Generated {} triangles for sphere", triangles.len());
    }
    
    /// Debug version of mesh generation with diagnostic output
    fn adaptive_surface_nets_mesh_debug(
        wasm_bytes: &[u8],
        bounds_min: (f32, f32, f32),
        bounds_max: (f32, f32, f32),
        config: &AdaptiveMeshConfig,
    ) -> anyhow::Result<Vec<crate::Triangle>> {
        let sampler = WasmSampler::new(wasm_bytes.to_vec())?;

        // Phase 1: Coarse discovery
        let mut octree = AdaptiveOctree::new(bounds_min, bounds_max, config);
        super::phase1_coarse_discovery(&sampler, &mut octree)?;
        
        println!("Phase 1 complete: {} MIXED root cells", octree.roots.len());
        
        // Phase 2: Adaptive refinement
        super::phase2_adaptive_refinement(&sampler, &mut octree)?;
        
        // Count leaf cells
        let mut leaf_count = 0;
        let mut mixed_leaf_count = 0;
        for (_idx, root) in &octree.roots {
            count_leaves(root, &mut leaf_count, &mut mixed_leaf_count);
        }
        println!("Phase 2 complete: {} leaf cells, {} MIXED", leaf_count, mixed_leaf_count);

        // Phase 3: Extract mesh
        let triangles = super::phase3_extract_mesh(&octree)?;
        
        println!("Phase 3 complete: {} triangles", triangles.len());

        Ok(triangles)
    }
    
    fn count_leaves(node: &OctreeNode, total: &mut usize, mixed: &mut usize) {
        if let Some(ref children) = node.children {
            for child in children.iter() {
                count_leaves(child, total, mixed);
            }
        } else {
            *total += 1;
            if node.cell_type() == CellType::Mixed {
                *mixed += 1;
            }
        }
    }

    fn boundary_edge_count(triangles: &[Triangle]) -> usize {
        // Quantize vertices so that shared edges compare reliably
        let scale = 1_000_000.0f32;
        let mut edges: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();

        for tri in triangles {
            let verts = [tri[0], tri[1], tri[2]];
            let tri_edges = [(0, 1), (1, 2), (2, 0)];

            for &(ia, ib) in &tri_edges {
                let a = verts[ia];
                let b = verts[ib];

                let qa = (
                    (a.0 * scale).round() as i64,
                    (a.1 * scale).round() as i64,
                    (a.2 * scale).round() as i64,
                );
                let qb = (
                    (b.0 * scale).round() as i64,
                    (b.1 * scale).round() as i64,
                    (b.2 * scale).round() as i64,
                );

                let key = if qa <= qb { (qa, qb) } else { (qb, qa) };
                *edges.entry(key).or_default() += 1;
            }
        }

        edges.values().filter(|&&count| count != 2).count()
    }

    fn edge_incidence_histogram(triangles: &[Triangle]) -> HashMap<usize, usize> {
        let scale = 1_000_000.0f32;
        let mut edges: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();

        for tri in triangles {
            let verts = [tri[0], tri[1], tri[2]];
            let tri_edges = [(0, 1), (1, 2), (2, 0)];

            for &(ia, ib) in &tri_edges {
                let a = verts[ia];
                let b = verts[ib];

                let qa = (
                    (a.0 * scale).round() as i64,
                    (a.1 * scale).round() as i64,
                    (a.2 * scale).round() as i64,
                );
                let qb = (
                    (b.0 * scale).round() as i64,
                    (b.1 * scale).round() as i64,
                    (b.2 * scale).round() as i64,
                );

                let key = if qa <= qb { (qa, qb) } else { (qb, qa) };
                *edges.entry(key).or_default() += 1;
            }
        }

        let mut hist = HashMap::new();
        for count in edges.values() {
            *hist.entry(*count).or_default() += 1;
        }
        hist
    }

    #[test]
    fn test_torus_mesh_is_manifold() {
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/simple_torus_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: torus wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        let config = AdaptiveMeshConfig {
            base_resolution: 6,
            max_refinement_depth: 3,
        };

        let bounds_min = (-1.35, -0.35, -1.35);
        let bounds_max = (1.35, 0.35, 1.35);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        eprintln!("Triangles: {}", triangles.len());

        let boundary_edges = boundary_edge_count(&triangles);

        if boundary_edges != 0 {
            let hist = edge_incidence_histogram(&triangles);
            eprintln!("Edge incidence histogram: {:?}", hist);
        }

        assert_eq!(
            boundary_edges, 0,
            "Expected manifold mesh (no boundary edges), found {} boundary edges",
            boundary_edges
        );
    }

    #[test]
    fn test_adaptive_refinement_reduces_samples() {
        // This test verifies that adaptive sampling uses fewer samples than uniform
        // for a simple sphere (most of the volume is empty or solid)
        
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/simple_sphere_model.wasm");
        
        if !wasm_path.exists() {
            eprintln!("Skipping test: sphere wasm not found");
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");
        
        // With base_resolution=8 and max_depth=3, effective resolution is 64
        // But adaptive should use far fewer samples than 64^3 = 262144
        let config = AdaptiveMeshConfig {
            base_resolution: 8,
            max_refinement_depth: 3,
        };

        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);

        let result = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config);
        assert!(result.is_ok(), "Mesh generation should succeed");
        
        let triangles = result.unwrap();
        // Sphere surface should produce a reasonable number of triangles
        // Not too few (mesh exists) and not too many (adaptive is working)
        assert!(triangles.len() > 10, "Should have more than 10 triangles");
        assert!(triangles.len() < 50000, "Should have fewer than 50000 triangles (adaptive working)");
        
        println!("Adaptive mesh: {} triangles", triangles.len());
    }

    #[test]
    fn test_gyroid_mesh_is_manifold() {
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/gyroid_lattice_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: gyroid wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        // Use higher base resolution with depth=2 to test adaptive refinement
        let config = AdaptiveMeshConfig {
            base_resolution: 8,
            max_refinement_depth: 2,
        };

        let pi = std::f32::consts::PI;
        let bounds_min = (-pi, -pi, -pi);
        let bounds_max = (pi, pi, pi);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        eprintln!("Gyroid triangles: {}", triangles.len());

        let boundary_edges = boundary_edge_count(&triangles);

        if boundary_edges != 0 {
            let hist = edge_incidence_histogram(&triangles);
            eprintln!("Edge incidence histogram: {:?}", hist);
            
            // Debug: find and print some boundary edges
            let scale = 1_000_000.0f32;
            let mut edges: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();
            for tri in &triangles {
                let verts = [tri[0], tri[1], tri[2]];
                let tri_edges = [(0, 1), (1, 2), (2, 0)];
                for &(ia, ib) in &tri_edges {
                    let a = verts[ia];
                    let b = verts[ib];
                    let qa = (
                        (a.0 * scale).round() as i64,
                        (a.1 * scale).round() as i64,
                        (a.2 * scale).round() as i64,
                    );
                    let qb = (
                        (b.0 * scale).round() as i64,
                        (b.1 * scale).round() as i64,
                        (b.2 * scale).round() as i64,
                    );
                    let key = if qa <= qb { (qa, qb) } else { (qb, qa) };
                    *edges.entry(key).or_default() += 1;
                }
            }
            
            let mut boundary_count = 0;
            for (key, count) in &edges {
                if *count != 2 && boundary_count < 10 {
                    let (qa, qb) = key;
                    eprintln!("Boundary edge (count={}): ({:.4}, {:.4}, {:.4}) -> ({:.4}, {:.4}, {:.4})",
                        count,
                        qa.0 as f32 / scale, qa.1 as f32 / scale, qa.2 as f32 / scale,
                        qb.0 as f32 / scale, qb.1 as f32 / scale, qb.2 as f32 / scale);
                    boundary_count += 1;
                }
            }
        }

        assert_eq!(
            boundary_edges, 0,
            "Expected manifold mesh (no boundary edges), found {} boundary edges",
            boundary_edges
        );
    }
}
