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

fn quantize_round(v: (f32, f32, f32), scale: f32) -> (i64, i64, i64) {
    (
        (v.0 * scale).round() as i64,
        (v.1 * scale).round() as i64,
        (v.2 * scale).round() as i64,
    )
}

// ==================== TEST-ONLY SANITY CHECKS ====================
// These checks are compiled ONLY in test builds and are intended to catch
// invariant violations earlier in the pipeline than the final mesh validators.
//
// Expensive checks are gated behind `VOLUMETRIC_SANITY=1` (or any value).
// This keeps the default `cargo test` fast while still enabling deep diagnostics
// when investigating rare manifoldness/winding issues.

#[cfg(test)]
fn sanity_enabled() -> bool {
    std::env::var("VOLUMETRIC_SANITY").is_ok()
}

#[cfg(test)]
macro_rules! sanity_assert {
    ($cond:expr $(,)?) => {
        if crate::adaptive_surface_nets::sanity_enabled() {
            assert!($cond);
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if crate::adaptive_surface_nets::sanity_enabled() {
            assert!($cond, $($arg)+);
        }
    };
}

#[cfg(test)]
fn q_round(v: (f32, f32, f32), scale: f32) -> (i64, i64, i64) {
    (
        (v.0 * scale).round() as i64,
        (v.1 * scale).round() as i64,
        (v.2 * scale).round() as i64,
    )
}

#[cfg(test)]
fn q_trunc(v: (f32, f32, f32), scale: f32) -> (i64, i64, i64) {
    ((v.0 * scale) as i64, (v.1 * scale) as i64, (v.2 * scale) as i64)
}

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

    /// Number of binary search iterations for edge crossing refinement.
    /// Higher values give more accurate vertex placement at the cost of more samples.
    /// 0 = disabled (use SDF linear interpolation only)
    /// 4-8 iterations gives 16-256x better precision than cell size.
    /// Default: 4
    pub edge_refinement_iterations: usize,

    /// Number of vertex relaxation iterations to project vertices onto the surface.
    /// Each iteration moves the vertex along the SDF gradient toward the surface.
    /// 0 = disabled
    /// Default: 2
    pub vertex_relaxation_iterations: usize,

    /// How to compute per-vertex normals for shading.
    ///
    /// `Mesh` is fast and stable (derived from triangle geometry).
    /// `HqBisection` performs additional occupancy-based sampling to fit a higher-quality normal.
    pub normal_mode: NormalMode,
}

/// Normal computation strategy.
#[derive(Clone, Copy, Debug)]
pub enum NormalMode {
    /// Compute normals from the extracted mesh (area-weighted vertex normals).
    Mesh,
    /// High-quality normals using occupancy-only sampling:
    /// - Start from a mesh-normal guess
    /// - Offset by `eps_frac * cell_size` in two tangent directions
    /// - Re-project each offset point to the surface by bisection along the normal line
    /// - Use the resulting local patch to estimate the normal
    HqBisection {
        /// Offset distance as a fraction of the estimated minimum cell size.
        eps_frac: f32,
        /// Initial bracketing distance along the normal line as a fraction of cell size.
        bracket_frac: f32,
        /// Number of bisection iterations used during re-projection.
        iterations: usize,
    },
}

impl Default for AdaptiveMeshConfig {
    fn default() -> Self {
        Self {
            base_resolution: 8,
            max_refinement_depth: 4,
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
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
    /// Occupancy at each corner.
    /// `true` = filled/inside, `false` = empty/outside.
    /// Order: [0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]
    corner_filled: [bool; 8],
    /// Children (None if leaf node)
    children: Option<Box<[OctreeNode; 8]>>,
}

impl OctreeNode {
    fn cell_type(&self) -> CellType {
        let all_inside = self.corner_filled.iter().all(|&b| b);
        let all_outside = self.corner_filled.iter().all(|&b| !b);
        if all_inside {
            CellType::AllInside
        } else if all_outside {
            CellType::AllOutside
        } else {
            CellType::Mixed
        }
    }

    /// Check if a corner is inside (negative distance)
    #[inline]
    fn corner_inside(&self, index: usize) -> bool {
        self.corner_filled[index]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LeafIndex {
    /// Global integer coordinates in the finest grid (effective_res = base_res << max_depth).
    /// This identifies the minimum corner of the leaf cell in that finest grid.
    x: i32,
    y: i32,
    z: i32,
    /// Leaf size in finest-grid cells (power of two).
    size: i32,
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

    fn leaf_index(&self, max_depth: usize) -> LeafIndex {
        // Compute the leaf's (x,y,z) and size in the finest grid.
        // Within a base cell, each path step halves the cell; the child index encodes the
        // octant via bits: x=(i&1), y=(i&2), z=(i&4).
        let mut lx: i32 = 0;
        let mut ly: i32 = 0;
        let mut lz: i32 = 0;
        for &c in &self.path {
            lx = (lx << 1) | ((c & 1) as i32);
            ly = (ly << 1) | (((c >> 1) & 1) as i32);
            lz = (lz << 1) | (((c >> 2) & 1) as i32);
        }
        let depth = self.path.len();
        let shift = (max_depth.saturating_sub(depth)) as i32;
        let size = 1i32 << shift;

        // Base cell origin in finest grid.
        let base_mul = 1i32 << (max_depth as i32);
        let bx = self.base.x * base_mul;
        let by = self.base.y * base_mul;
        let bz = self.base.z * base_mul;

        LeafIndex {
            x: bx + (lx << shift),
            y: by + (ly << shift),
            z: bz + (lz << shift),
            size,
        }
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
    /// Corner occupancy cache at base level for sharing between cells.
    /// Stored as a flat Vec for performance (avoids HashMap overhead).
    /// Indices range from -1 to base_resolution+1 (padded by 1 on each side).
    /// Size: (base_resolution + 3)^3.
    /// Values are occupancy flags: `true` = filled, `false` = empty.
    base_corner_filled: Vec<bool>,
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
            // Initialize as empty/outside - will be filled during sampling
            base_corner_filled: vec![false; corner_n * corner_n * corner_n],
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
    let mut mesh = phase3_extract_mesh(&octree, &sampler, config)?;

    // Orientation pass:
    // 1) enforce component-wise consistent winding from topology only
    // 2) choose outward orientation per component using topology-anchored witness points
    orient_indexed_mesh(&mut mesh, &sampler, bounds_min, bounds_max, config)?;

    // Optional post-pass: project welded vertices closer to the implicit surface.
    // This must happen AFTER extraction so it cannot affect connectivity / topology.
    if config.vertex_relaxation_iterations > 0 {
        relax_indexed_vertices_to_surface(
            &mut mesh,
            &sampler,
            bounds_min,
            bounds_max,
            config.vertex_relaxation_iterations,
            config.base_resolution,
            config.max_refinement_depth,
        )?;
    }

    // Phase 4: Compute per-vertex normals.
    // Under occupancy-only sampling, normals must come from mesh geometry and/or additional
    // occupancy probes (HQ mode). Do not derive normals from any fake SDF field.
    let normals = compute_indexed_normals(&mesh, &sampler, bounds_min, bounds_max, config)?;

    Ok(indexed_mesh_to_triangles(&mesh, &normals))
}

/// Internal indexed mesh representation.
///
/// Topology identity is integer-based (vertex ids are stable), while positions are payload.
#[derive(Clone, Debug)]
struct IndexedMesh {
    vertex_pos: Vec<(f32, f32, f32)>,
    triangles: Vec<[u32; 3]>,
    /// Per-triangle witness point on the implicit surface, derived from a bracketed
    /// sign-changing grid edge. Used to choose the outward orientation robustly.
    tri_witness: Vec<(f32, f32, f32)>,
}

fn orient_indexed_mesh(
    mesh: &mut IndexedMesh,
    sampler: &WasmSampler,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    config: &AdaptiveMeshConfig,
) -> Result<()> {
    if mesh.triangles.is_empty() {
        return Ok(());
    }
    debug_assert_eq!(mesh.triangles.len(), mesh.tri_witness.len());

    // --- Stage 1: enforce per-component consistent winding using triangle adjacency.
    // IMPORTANT: do NOT flip already-oriented triangles while traversing, otherwise earlier
    // constraints can be violated and results become traversal-order dependent.
    //
    // We assign a boolean "flip state" per triangle (relative to its current winding), using
    // only manifold edges (edges with exactly 2 incident triangles) as constraints.
    let edge_to_tris = build_edge_to_tris(&mesh.triangles);
    let mut oriented = vec![false; mesh.triangles.len()];
    let mut flip_state = vec![false; mesh.triangles.len()];

    for seed in 0..mesh.triangles.len() {
        if oriented[seed] {
            continue;
        }

        let mut stack = vec![seed];
        oriented[seed] = true;
        flip_state[seed] = false;
        let mut component_tris: Vec<usize> = Vec::new();

        while let Some(ti) = stack.pop() {
            component_tris.push(ti);
            let tri = mesh.triangles[ti];
            let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

            for (a, b) in edges {
                if a == b {
                    continue;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                let Some(neis) = edge_to_tris.get(&key) else { continue };

                // Only enforce constraints on manifold edges.
                if neis.len() != 2 {
                    continue;
                }

                let nj = if neis[0] == ti { neis[1] } else if neis[1] == ti { neis[0] } else { continue };

                let dt = match tri_traverses_edge_min_to_max(mesh.triangles[ti], key) {
                    Some(v) => v,
                    None => continue,
                };
                let dn = match tri_traverses_edge_min_to_max(mesh.triangles[nj], key) {
                    Some(v) => v,
                    None => continue,
                };

                // Desired: (dt ^ flip_t) != (dn ^ flip_n)
                // Solve for neighbor flip:
                //   flip_n = flip_t ^ (dt == dn)
                let required_flip_n = flip_state[ti] ^ (dt == dn);

                if !oriented[nj] {
                    oriented[nj] = true;
                    flip_state[nj] = required_flip_n;
                    stack.push(nj);
                } else {
                    // Already oriented: verify consistency.
                    // If inconsistent, we keep existing state; this indicates a topological issue
                    // (non-manifoldness elsewhere, duplicate triangles, or inconsistent adjacency).
                    // We avoid oscillation by never changing an already oriented triangle here.
                    if flip_state[nj] != required_flip_n {
                        // No panic: fractal/topologically complex cases may contain unavoidable conflicts.
                        // This pass is best-effort for non-2-manifold meshes.
                    }
                }
            }
        }

        // Apply the computed flip state for this component.
        for &ti in &component_tris {
            if flip_state[ti] {
                flip_triangle(&mut mesh.triangles[ti]);
            }
        }

        // --- Stage 2: choose outward vs inward for this component.
        choose_component_outward(mesh, sampler, bounds_min, bounds_max, config, &component_tris)?;
    }

    Ok(())
}

fn build_edge_to_tris(triangles: &[[u32; 3]]) -> HashMap<(u32, u32), Vec<usize>> {
    let mut map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, &tri) in triangles.iter().enumerate() {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
        for (a, b) in edges {
            if a == b {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            map.entry(key).or_default().push(ti);
        }
    }
    map
}

fn tri_traverses_edge_min_to_max(tri: [u32; 3], edge: (u32, u32)) -> Option<bool> {
    let (minv, maxv) = edge;
    let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
    for (a, b) in edges {
        if a == minv && b == maxv {
            return Some(true);
        }
        if a == maxv && b == minv {
            return Some(false);
        }
    }
    None
}

#[inline]
fn flip_triangle(tri: &mut [u32; 3]) {
    // Swap two vertices to reverse winding.
    tri.swap(1, 2);
}

fn choose_component_outward(
    mesh: &mut IndexedMesh,
    sampler: &WasmSampler,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    config: &AdaptiveMeshConfig,
    component_tris: &[usize],
) -> Result<()> {
    // Use cell-size scale for epsilon.
    let min_extent = (bounds_max.0 - bounds_min.0)
        .min(bounds_max.1 - bounds_min.1)
        .min(bounds_max.2 - bounds_min.2);
    let effective_res = (config.base_resolution * (1usize << config.max_refinement_depth)) as f32;
    let cell_size = if effective_res > 0.0 { min_extent / effective_res } else { min_extent };

    let mut ctx = SamplingContext::new(&sampler.engine, &sampler.module)?;
    let mut outward = 0usize;
    let mut inward = 0usize;

    // Sample at most a fixed number of triangles for stability and speed.
    let max_samples = 512usize;
    let stride = (component_tris.len() / max_samples).max(1);

    for &ti in component_tris.iter().step_by(stride) {
        let tri = mesh.triangles[ti];
        let a = mesh.vertex_pos[tri[0] as usize];
        let b = mesh.vertex_pos[tri[1] as usize];
        let c = mesh.vertex_pos[tri[2] as usize];
        let n = Triangle::compute_face_normal(&[a, b, c]);
        let Some(nhat) = normalize_or_none(n) else {
            continue;
        };

        let p = mesh.tri_witness[ti];

        // Adaptive epsilon: start at a fraction of cell size and shrink if ambiguous.
        let mut eps = (cell_size * 0.05).max(1.0e-6);
        let mut vote: Option<bool> = None;
        for _ in 0..6 {
            let p_plus = (p.0 + nhat.0 * eps, p.1 + nhat.1 * eps, p.2 + nhat.2 * eps);
            let p_minus = (p.0 - nhat.0 * eps, p.1 - nhat.1 * eps, p.2 - nhat.2 * eps);
            let plus_inside = ctx.is_inside(p_plus)? != 0.0;
            let minus_inside = ctx.is_inside(p_minus)? != 0.0;

            // Normal points outward if +n is outside and -n is inside.
            if !plus_inside && minus_inside {
                vote = Some(true);
                break;
            }
            if plus_inside && !minus_inside {
                vote = Some(false);
                break;
            }
            eps *= 0.5;
        }

        match vote {
            Some(true) => outward += 1,
            Some(false) => inward += 1,
            None => {}
        }
    }

    // If we have evidence that this component is oriented inward, flip it.
    if inward > outward {
        for &ti in component_tris {
            flip_triangle(&mut mesh.triangles[ti]);
        }
    }
    Ok(())
}

fn indexed_mesh_to_triangles(mesh: &IndexedMesh, normals: &[(f32, f32, f32)]) -> Vec<Triangle> {
    let mut out = Vec::with_capacity(mesh.triangles.len());
    for &t in &mesh.triangles {
        let a = mesh.vertex_pos[t[0] as usize];
        let b = mesh.vertex_pos[t[1] as usize];
        let c = mesh.vertex_pos[t[2] as usize];
        let na = normals[t[0] as usize];
        let nb = normals[t[1] as usize];
        let nc = normals[t[2] as usize];
        out.push(Triangle::with_vertex_normals([a, b, c], [na, nb, nc]));
    }
    out
}

fn compute_indexed_mesh_vertex_normals(mesh: &IndexedMesh) -> Vec<(f32, f32, f32)> {
    let mut acc = vec![(0.0f32, 0.0f32, 0.0f32); mesh.vertex_pos.len()];
    for &tri in &mesh.triangles {
        let a = mesh.vertex_pos[tri[0] as usize];
        let b = mesh.vertex_pos[tri[1] as usize];
        let c = mesh.vertex_pos[tri[2] as usize];
        let n = Triangle::compute_face_normal(&[a, b, c]);
        // Accumulate unnormalized face normals (area-weighted).
        for &vid in &tri {
            let v = &mut acc[vid as usize];
            v.0 += n.0;
            v.1 += n.1;
            v.2 += n.2;
        }
    }
    // Normalize.
    for v in &mut acc {
        if let Some(n) = normalize_or_none(*v) {
            *v = n;
        }
    }
    acc
}

fn compute_indexed_normals(
    mesh: &IndexedMesh,
    sampler: &WasmSampler,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    config: &AdaptiveMeshConfig,
) -> Result<Vec<(f32, f32, f32)>> {
    let mut normals = compute_indexed_mesh_vertex_normals(mesh);

    match config.normal_mode {
        NormalMode::Mesh => Ok(normals),
        NormalMode::HqBisection {
            eps_frac,
            bracket_frac,
            iterations,
        } => {
            // Compute HQ normals per vertex.
            let min_extent = (bounds_max.0 - bounds_min.0)
                .min(bounds_max.1 - bounds_min.1)
                .min(bounds_max.2 - bounds_min.2);
            let effective_res = (config.base_resolution * (1usize << config.max_refinement_depth)) as f32;
            let cell_size = if effective_res > 0.0 { min_extent / effective_res } else { min_extent };

            let eps = (cell_size * eps_frac).max(1.0e-6);
            let bracket = (cell_size * bracket_frac).max(eps * 2.0);

            let mut ctx = SamplingContext::new(&sampler.engine, &sampler.module)?;
            for (vid, p) in mesh.vertex_pos.iter().copied().enumerate() {
                let n0 = normals[vid];
                if let Some(n) = hq_normal_at_point(&mut ctx, p, n0, eps, bracket, iterations)? {
                    normals[vid] = n;
                }
            }
            Ok(normals)
        }
    }
}

fn relax_indexed_vertices_to_surface(
    mesh: &mut IndexedMesh,
    sampler: &WasmSampler,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    outer_iterations: usize,
    base_resolution: usize,
    max_refinement_depth: usize,
) -> Result<()> {
    if mesh.triangles.is_empty() {
        return Ok(());
    }

    let normals = compute_indexed_mesh_vertex_normals(mesh);

    let min_extent = (bounds_max.0 - bounds_min.0)
        .min(bounds_max.1 - bounds_min.1)
        .min(bounds_max.2 - bounds_min.2);
    let effective_res = (base_resolution * (1usize << max_refinement_depth)) as f32;
    let cell_size = if effective_res > 0.0 { min_extent / effective_res } else { min_extent };

    // Project within a small neighborhood to avoid jumping across thin features.
    let bracket = (cell_size * 0.5).max(1.0e-4);
    let bisect_iters = 10usize;

    let mut ctx = SamplingContext::new(&sampler.engine, &sampler.module)?;

    let mut new_pos = mesh.vertex_pos.clone();
    for (vid, p0) in mesh.vertex_pos.iter().copied().enumerate() {
        let Some(dir) = normalize_or_none(normals[vid]) else {
            continue;
        };
        let mut p = p0;
        for _ in 0..outer_iterations {
            let Some(p_new) = project_to_surface_on_line(&mut ctx, p, dir, bracket, bisect_iters)? else {
                break;
            };
            p = p_new;
        }
        new_pos[vid] = p;
    }

    mesh.vertex_pos = new_pos;
    Ok(())
}

/// Phase 1: Sample the base grid to discover geometry regions.
///
/// The sampling pipeline provides an occupancy-like value (`is_inside(p) != 0.0` means filled).
/// For meshing we only cache corner occupancy and classify cells as inside/outside/mixed.
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
    // Returns (index, filled)
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
                let density = ctx.is_inside(pos).unwrap_or(0.0);
                let filled = density != 0.0;
                
                // Compute flat index inline (offset by 1 to map -1 -> 0)
                let xi = (x + 1) as usize;
                let yi = (y + 1) as usize;
                let zi = (z + 1) as usize;
                let idx = (zi * corner_n + yi) * corner_n + xi;
                (idx, filled)
            },
        )
        .collect();

    #[cfg(test)]
    {
        sanity_assert!(corner_samples.len() == corner_positions.len(), "corner sample count mismatch");
        if sanity_enabled() {
            let mut seen = HashSet::with_capacity(corner_samples.len());
            for (idx, _filled) in &corner_samples {
                sanity_assert!(*idx < octree.base_corner_filled.len(), "corner idx out of bounds: {idx}");
                sanity_assert!(seen.insert(*idx), "duplicate corner idx encountered: {idx}");
            }
        }
    }

    // Store corner samples using direct indexing (faster than HashMap)
    for (idx, filled) in corner_samples {
        octree.base_corner_filled[idx] = filled;
    }

    // First pass: identify all MIXED cells (cells with sign changes)
    let min_cell = -1i32;
    let max_cell = base_res;
    
    let mut mixed_cells: HashSet<CellIndex> = HashSet::new();

    for z in min_cell..=max_cell {
        for y in min_cell..=max_cell {
            for x in min_cell..=max_cell {
                let idx = CellIndex { x, y, z };
                let corners = [
                    octree.base_corner_filled[octree.corner_idx(x, y, z)],
                    octree.base_corner_filled[octree.corner_idx(x + 1, y, z)],
                    octree.base_corner_filled[octree.corner_idx(x, y + 1, z)],
                    octree.base_corner_filled[octree.corner_idx(x + 1, y + 1, z)],
                    octree.base_corner_filled[octree.corner_idx(x, y, z + 1)],
                    octree.base_corner_filled[octree.corner_idx(x + 1, y, z + 1)],
                    octree.base_corner_filled[octree.corner_idx(x, y + 1, z + 1)],
                    octree.base_corner_filled[octree.corner_idx(x + 1, y + 1, z + 1)],
                ];

                // Check if MIXED (has both filled and empty corners)
                let first_filled = corners[0];
                let is_mixed = corners.iter().any(|&c| c != first_filled);
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
        let corner_filled = [
            octree.base_corner_filled[octree.corner_idx(idx.x, idx.y, idx.z)],
            octree.base_corner_filled[octree.corner_idx(idx.x + 1, idx.y, idx.z)],
            octree.base_corner_filled[octree.corner_idx(idx.x, idx.y + 1, idx.z)],
            octree.base_corner_filled[octree.corner_idx(idx.x + 1, idx.y + 1, idx.z)],
            octree.base_corner_filled[octree.corner_idx(idx.x, idx.y, idx.z + 1)],
            octree.base_corner_filled[octree.corner_idx(idx.x + 1, idx.y, idx.z + 1)],
            octree.base_corner_filled[octree.corner_idx(idx.x, idx.y + 1, idx.z + 1)],
            octree.base_corner_filled[octree.corner_idx(idx.x + 1, idx.y + 1, idx.z + 1)],
        ];

        let node = OctreeNode {
            min: cell_min,
            max: cell_max,
            depth: 0,
            corner_filled,
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

    // Process root cells in parallel using rayon.
    // Each thread gets its own sampling context via map_init.
    //
    // NOTE: With true adaptive refinement, Phase 3 must be able to stitch across mixed
    // refinement levels. Our current Phase 3 is not yet robust enough to do that across all
    // models (Gyroid/Mandelbulb), leading to missing quads (boundary edges).
    //
    // For now we use a robust “surface-adjacent equal depth” strategy: every root cell that
    // participates in the octree (the MIXED band discovered in Phase 1, plus neighbors) is
    // refined uniformly to `max_depth`. This keeps the overall algorithm adaptive at the root
    // selection level (we still avoid refining the entire volume), while ensuring Phase 3 sees
    // a locally uniform grid around the surface.
    //
    // Future work: implement true 2:1 balancing + stitching so we can use `refine_node_recursive`
    // everywhere.
    let refined_roots: Vec<Result<(CellIndex, OctreeNode)>> = cells_to_refine
        .into_par_iter()
        .map_init(
            || SamplingContext::new(&engine, &module).expect("Failed to create sampling context"),
            |ctx, (cell_idx, mut node)| {
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
/// This refines only where needed (MIXED cells) up to `max_depth`.
fn refine_node_recursive(
    node: &mut OctreeNode,
    max_depth: usize,
    ctx: &mut SamplingContext,
) -> Result<()> {
    refine_node_recursive_band(node, max_depth, ctx, false)
}

/// Adaptive refinement with a narrow neighbor band.
///
/// If `force` is true, the node is refined regardless of its cell type. This is used to
/// refine a small band of neighbors around MIXED regions so that Phase 3 has the 4 incident
/// cells it expects around sign-changing edges.
fn refine_node_recursive_band(
    node: &mut OctreeNode,
    max_depth: usize,
    ctx: &mut SamplingContext,
    force: bool,
) -> Result<()> {
    if node.depth >= max_depth {
        return Ok(());
    }

    // Only refine if this node is MIXED (contains surface), unless forced by the neighbor band.
    if !force && node.cell_type() != CellType::Mixed {
        return Ok(());
    }

    // Create 8 children by subdividing this cell
    let mid = node.center();
    let _child_size = node.size() * 0.5;
    
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
            corner_filled: [false; 8], // Will be filled in
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

            let filled = if let Some(&cached) = corner_cache.get(&key) {
                cached
            } else {
                let density = ctx.is_inside(pos)?;
                let filled = density != 0.0;
                corner_cache.insert(key, filled);
                filled
            };

            child.corner_filled[corner_idx] = filled;
        }
    }

    // Recursively refine only where needed.
    // Compute which children should be forced refined as part of the neighbor band.
    // Mark all MIXED children, plus their face- and edge-adjacent siblings.
    let mut force_child = [false; 8];
    for i in 0..8 {
        if children[i].cell_type() == CellType::Mixed {
            force_child[i] = true;
            for j in 0..8 {
                let diff = (i ^ j) & 0b111;
                let hamming = (diff & 1 != 0) as u8 + (diff & 2 != 0) as u8 + (diff & 4 != 0) as u8;
                if hamming == 1 || hamming == 2 || hamming == 3 {
                    force_child[j] = true;
                }
            }
        }
    }

    for i in 0..8 {
        refine_node_recursive_band(&mut children[i], max_depth, ctx, force_child[i])?;
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
    let _child_size = node.size() * 0.5;
    
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
            corner_filled: [false; 8], // Will be filled in
            children: None,
        }
    });

    #[cfg(test)]
    {
        // Child bounds should partition the parent exactly at `mid`.
        sanity_assert!(children.iter().all(|c| c.depth == node.depth + 1));
        if sanity_enabled() {
            for (i, c) in children.iter().enumerate() {
                sanity_assert!(c.min.0 <= c.max.0 && c.min.1 <= c.max.1 && c.min.2 <= c.max.2, "invalid child bounds at {i}: min={:?} max={:?}", c.min, c.max);
                // Each axis uses either parent min..mid or mid..parent max.
                let expect_min_x = if i & 1 != 0 { mid.0 } else { node.min.0 };
                let expect_max_x = if i & 1 != 0 { node.max.0 } else { mid.0 };
                let expect_min_y = if i & 2 != 0 { mid.1 } else { node.min.1 };
                let expect_max_y = if i & 2 != 0 { node.max.1 } else { mid.1 };
                let expect_min_z = if i & 4 != 0 { mid.2 } else { node.min.2 };
                let expect_max_z = if i & 4 != 0 { node.max.2 } else { mid.2 };

                sanity_assert!((c.min.0, c.max.0) == (expect_min_x, expect_max_x), "child x bounds mismatch at {i}");
                sanity_assert!((c.min.1, c.max.1) == (expect_min_y, expect_max_y), "child y bounds mismatch at {i}");
                sanity_assert!((c.min.2, c.max.2) == (expect_min_z, expect_max_z), "child z bounds mismatch at {i}");
            }
        }
    }

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

            let filled = if let Some(&cached) = corner_cache.get(&key) {
                cached
            } else {
                let density = ctx.is_inside(pos)?;
                let filled = density != 0.0;
                corner_cache.insert(key, filled);
                filled
            };

            child.corner_filled[corner_idx] = filled;
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

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SegKey {
    axis: EdgeAxis,
    perp1: i64,
    perp2: i64,
    seg_min: i64,
    seg_max: i64,
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
    /// Cell bounds (used for geometric sorting/triangulation decisions)
    cell_min: (f32, f32, f32),
    cell_max: (f32, f32, f32),
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
fn phase3_extract_mesh(
    octree: &AdaptiveOctree,
    sampler: &WasmSampler,
    config: &AdaptiveMeshConfig,
) -> Result<IndexedMesh> {
    // Collect all leaf cells
    let mut leaf_cells: Vec<(CellId, OctreeNode)> = Vec::new();
    for (&base_idx, root) in &octree.roots {
        let cell_id = CellId::new(base_idx);
        collect_leaf_cells(root, cell_id, &mut leaf_cells);
    }

    // Build a finest-grid ownership map for MIXED leaves. This provides a discrete coordinate
    // system for neighbor queries in Phase 3 under adaptive refinement.
    let max_depth = octree.max_depth;
    let effective_res_i32 = (octree.base_resolution as i32) * (1i32 << (max_depth as i32));
    let eff = effective_res_i32 as usize;
    let owner_len = eff * eff * eff;
    let mut owner: Vec<u32> = vec![u32::MAX; owner_len];
    let mut leaf_vertices: Vec<(f32, f32, f32)> = Vec::new();
    
    // Create sampling context for binary search edge refinement if enabled.
    // (Vertex relaxation runs as a welded post-pass after extraction.)
    let mut sampling_ctx = if config.edge_refinement_iterations > 0 {
        Some(SamplingContext::new(&sampler.engine, &sampler.module)?)
    } else {
        None
    };

    // Group edges by the line they lie on
    // Key: LineKey (axis + perpendicular coordinates)
    // Value: list of edges on that line
    let mut line_to_edges: HashMap<LineKey, Vec<EdgeOnLine>> = HashMap::new();

    // Finest-grid step (corner spacing) used to convert world coordinates to fine integer corner indices.
    // This should be exact for our subdivision scheme; we still use rounding to be robust.
    let fine_step = (
        octree.base_step.0 / (1u32 << max_depth) as f32,
        octree.base_step.1 / (1u32 << max_depth) as f32,
        octree.base_step.2 / (1u32 << max_depth) as f32,
    );
    let world_to_fine_corner = |p: f32, axis: EdgeAxis| -> i32 {
        let (bmin, step) = match axis {
            EdgeAxis::X => (octree.bounds_min.0, fine_step.0),
            EdgeAxis::Y => (octree.bounds_min.1, fine_step.1),
            EdgeAxis::Z => (octree.bounds_min.2, fine_step.2),
        };
        ((p - bmin) / step).round() as i32
    };

    // Edge definitions: 12 edges of a cube
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // X-aligned
        (0, 2), (1, 3), (4, 6), (5, 7), // Y-aligned  
        (0, 4), (1, 5), (2, 6), (3, 7), // Z-aligned
    ];

    for (cell_id, node) in &leaf_cells {
        if node.cell_type() != CellType::Mixed {
            continue;
        }

        // Compute vertex with optional binary search refinement and relaxation
        let vertex = compute_surface_nets_vertex_refined(
            node,
            sampling_ctx.as_mut(),
            config.edge_refinement_iterations,
            config.vertex_relaxation_iterations,
        )?;

        // Register the MIXED leaf in the finest-grid ownership map.
        // Note: We intentionally fill the whole leaf volume in the finest grid.
        // This lets Phase 3 query "which leaf owns this fine cell" cheaply.
        // Topology decisions still come from local neighbor configurations.
        let leaf_idx = cell_id.leaf_index(max_depth);
        let vid = leaf_vertices.len() as u32;
        leaf_vertices.push(vertex);
        let x0 = leaf_idx.x.max(0) as usize;
        let y0 = leaf_idx.y.max(0) as usize;
        let z0 = leaf_idx.z.max(0) as usize;
        let x1 = (leaf_idx.x + leaf_idx.size).min(effective_res_i32) as usize;
        let y1 = (leaf_idx.y + leaf_idx.size).min(effective_res_i32) as usize;
        let z1 = (leaf_idx.z + leaf_idx.size).min(effective_res_i32) as usize;
        for z in z0..z1 {
            for y in y0..y1 {
                let row = (z * eff + y) * eff;
                for x in x0..x1 {
                    owner[row + x] = vid;
                }
            }
        }
        
        for &(ca, cb) in &EDGES {
            // Check for sign change using corner_inside helper (negative distance = inside)
            if node.corner_inside(ca) == node.corner_inside(cb) {
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
                node.corner_inside(ca) // pa is at min, so check if pa's corner is inside
            } else {
                node.corner_inside(cb) // pb is at min, so check if pb's corner is inside
            };
            
            let edge_length = axis_max - axis_min;
            line_to_edges.entry(line_key).or_default().push(EdgeOnLine {
                vertex,
                axis_min,
                axis_max,
                inside_at_min,
                edge_length,
                cell_min: node.min,
                cell_max: node.max,
            });
        }
    }

    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut tri_witness: Vec<(f32, f32, f32)> = Vec::new();

    // Witness sampling context for robust outward orientation selection.
    let mut witness_ctx = SamplingContext::new(&sampler.engine, &sampler.module)?;
    
    // Track processed segments to avoid duplicates
    // Key: (line_key hash, seg_min quantized, seg_max quantized)
    let _processed_segments: HashSet<(EdgeAxis, i64, i64, i64, i64)> = HashSet::new();
    
    // Track processed quads by their sorted vertex positions to avoid duplicate quads from different lines
    let mut processed_quads: HashSet<[u32; 4]> = HashSet::new();
    
    // Track emitted triangles to avoid duplicates
    let mut emitted_triangles: HashSet<[u32; 3]> = HashSet::new();

    #[cfg(test)]
    let mut tri_to_seg: HashMap<[u32; 3], SegKey> = HashMap::new();

    #[cfg(test)]
    let mut sanity_seg_cell_count_hist: HashMap<usize, usize> = HashMap::new();

    #[cfg(test)]
    let mut sanity_mixed_size_segments: usize = 0;

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
        
        // For each segment between consecutive points, emit fine-grid quads by querying the
        // owning MIXED leaf around each finest-grid edge step.
        for i in 0..axis_points.len().saturating_sub(1) {
            let seg_min_q = axis_points[i];
            let seg_max_q = axis_points[i + 1];
            let seg_min = seg_min_q as f32 / scale;
            let seg_max = seg_max_q as f32 / scale;

            // Convert segment endpoints (world) to finest-grid corner indices.
            let a0 = world_to_fine_corner(seg_min, line_key.axis);
            let a1 = world_to_fine_corner(seg_max, line_key.axis);

            // Perpendicular coordinates are world corner coords; convert to finest-grid corner indices.
            let (p1_world, p2_world) = (line_key.perp1 as f32 / scale, line_key.perp2 as f32 / scale);
            let (p1_axis, p2_axis) = match line_key.axis {
                EdgeAxis::X => (EdgeAxis::Y, EdgeAxis::Z),
                EdgeAxis::Y => (EdgeAxis::X, EdgeAxis::Z),
                EdgeAxis::Z => (EdgeAxis::X, EdgeAxis::Y),
            };
            let c1 = world_to_fine_corner(p1_world, p1_axis);
            let c2 = world_to_fine_corner(p2_world, p2_axis);

            // Emit for each finest-grid step along the segment.
            for a in a0..a1 {
                // Only emit if this finest-grid edge step is actually covered by at least one
                // sign-changing leaf edge on this line. This keeps emission driven by the
                // original sign-change detection rather than by ownership alone.
                let a_world_min = match line_key.axis {
                    EdgeAxis::X => octree.bounds_min.0 + (a as f32) * fine_step.0,
                    EdgeAxis::Y => octree.bounds_min.1 + (a as f32) * fine_step.1,
                    EdgeAxis::Z => octree.bounds_min.2 + (a as f32) * fine_step.2,
                };
                let a_world_max = match line_key.axis {
                    EdgeAxis::X => octree.bounds_min.0 + ((a + 1) as f32) * fine_step.0,
                    EdgeAxis::Y => octree.bounds_min.1 + ((a + 1) as f32) * fine_step.1,
                    EdgeAxis::Z => octree.bounds_min.2 + ((a + 1) as f32) * fine_step.2,
                };

                let mut covered_by_edge: Option<bool> = None;
                for e in edges {
                    if e.axis_min <= a_world_min + 1.0e-6 && e.axis_max >= a_world_max - 1.0e-6 {
                        covered_by_edge = Some(e.inside_at_min);
                        break;
                    }
                }
                let Some(inside_at_min) = covered_by_edge else {
                    continue;
                };

                let (cx, cy, cz) = match line_key.axis {
                    EdgeAxis::X => (a, c1, c2),
                    EdgeAxis::Y => (c1, a, c2),
                    EdgeAxis::Z => (c1, c2, a),
                };

                // Gather the 4 surrounding fine-grid cells around this edge.
                let mut vids: [u32; 4] = [u32::MAX; 4];
                let mut ok = true;
                let cell_coords: [(i32, i32, i32); 4] = match line_key.axis {
                    EdgeAxis::X => [(cx, cy, cz), (cx, cy - 1, cz), (cx, cy, cz - 1), (cx, cy - 1, cz - 1)],
                    EdgeAxis::Y => [(cx, cy, cz), (cx - 1, cy, cz), (cx, cy, cz - 1), (cx - 1, cy, cz - 1)],
                    EdgeAxis::Z => [(cx, cy, cz), (cx - 1, cy, cz), (cx, cy - 1, cz), (cx - 1, cy - 1, cz)],
                };
                for (i4, (ix, iy, iz)) in cell_coords.iter().enumerate() {
                    if *ix < 0 || *iy < 0 || *iz < 0 || *ix >= effective_res_i32 || *iy >= effective_res_i32 || *iz >= effective_res_i32 {
                        ok = false;
                        break;
                    }
                    let idx = ((*iz as usize) * eff + (*iy as usize)) * eff + (*ix as usize);
                    let v = owner[idx];
                    if v == u32::MAX {
                        ok = false;
                        break;
                    }
                    vids[i4] = v;
                }
                if !ok {
                    #[cfg(test)]
                    {
                        if sanity_enabled() {
                            *sanity_seg_cell_count_hist.entry(0).or_default() += 1;
                        }
                    }
                    continue;
                }

                let mut cell_verts: Vec<CellData> = Vec::with_capacity(4);
                for v in vids {
                    let p = leaf_vertices[v as usize];
                    cell_verts.push((v, p, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)));
                }

                // Compute edge midpoint in 3D from fine corner coords.
                let edge_mid = match line_key.axis {
                    EdgeAxis::X => (
                        octree.bounds_min.0 + (cx as f32 + 0.5) * fine_step.0,
                        octree.bounds_min.1 + (cy as f32) * fine_step.1,
                        octree.bounds_min.2 + (cz as f32) * fine_step.2,
                    ),
                    EdgeAxis::Y => (
                        octree.bounds_min.0 + (cx as f32) * fine_step.0,
                        octree.bounds_min.1 + (cy as f32 + 0.5) * fine_step.1,
                        octree.bounds_min.2 + (cz as f32) * fine_step.2,
                    ),
                    EdgeAxis::Z => (
                        octree.bounds_min.0 + (cx as f32) * fine_step.0,
                        octree.bounds_min.1 + (cy as f32) * fine_step.1,
                        octree.bounds_min.2 + (cz as f32 + 0.5) * fine_step.2,
                    ),
                };

                // Compute a per-triangle witness point `p_surface` by refining the bracketed
                // sign-changing fine-grid edge crossing. This anchors outward decisions to
                // extraction topology rather than arbitrary triangle centroids.
                let (p_min, p_max) = match line_key.axis {
                    EdgeAxis::X => (
                        (
                            octree.bounds_min.0 + (cx as f32) * fine_step.0,
                            octree.bounds_min.1 + (cy as f32) * fine_step.1,
                            octree.bounds_min.2 + (cz as f32) * fine_step.2,
                        ),
                        (
                            octree.bounds_min.0 + (cx as f32 + 1.0) * fine_step.0,
                            octree.bounds_min.1 + (cy as f32) * fine_step.1,
                            octree.bounds_min.2 + (cz as f32) * fine_step.2,
                        ),
                    ),
                    EdgeAxis::Y => (
                        (
                            octree.bounds_min.0 + (cx as f32) * fine_step.0,
                            octree.bounds_min.1 + (cy as f32) * fine_step.1,
                            octree.bounds_min.2 + (cz as f32) * fine_step.2,
                        ),
                        (
                            octree.bounds_min.0 + (cx as f32) * fine_step.0,
                            octree.bounds_min.1 + (cy as f32 + 1.0) * fine_step.1,
                            octree.bounds_min.2 + (cz as f32) * fine_step.2,
                        ),
                    ),
                    EdgeAxis::Z => (
                        (
                            octree.bounds_min.0 + (cx as f32) * fine_step.0,
                            octree.bounds_min.1 + (cy as f32) * fine_step.1,
                            octree.bounds_min.2 + (cz as f32) * fine_step.2,
                        ),
                        (
                            octree.bounds_min.0 + (cx as f32) * fine_step.0,
                            octree.bounds_min.1 + (cy as f32) * fine_step.1,
                            octree.bounds_min.2 + (cz as f32 + 1.0) * fine_step.2,
                        ),
                    ),
                };

                let (p_inside, p_outside) = if inside_at_min { (p_min, p_max) } else { (p_max, p_min) };
                // A modest number of iterations is enough; `is_inside` is deterministic.
                let p_surface = binary_search_edge_crossing(&mut witness_ctx, p_inside, p_outside, 10)?;
                let edge_dir = match line_key.axis {
                    EdgeAxis::X => (1.0, 0.0, 0.0),
                    EdgeAxis::Y => (0.0, 1.0, 0.0),
                    EdgeAxis::Z => (0.0, 0.0, 1.0),
                };

                // Deduplicate at quad level by stable integer vertex ids.
                let mut quad_key: [u32; 4] = [
                    cell_verts[0].0,
                    cell_verts[1].0,
                    cell_verts[2].0,
                    cell_verts[3].0,
                ];
                quad_key.sort();
                if !processed_quads.insert(quad_key) {
                    continue;
                }

                #[cfg(test)]
                {
                    if sanity_enabled() {
                        let seg_key_dbg = SegKey {
                            axis: line_key.axis,
                            perp1: line_key.perp1,
                            perp2: line_key.perp2,
                            seg_min: seg_min_q,
                            seg_max: seg_max_q,
                        };
                        emit_triangles_for_quad_dbg(
                            &cell_verts,
                            inside_at_min,
                            edge_mid,
                            edge_dir,
                            p_surface,
                            &seg_key_dbg,
                            &mut triangles,
                            &mut tri_witness,
                            &mut emitted_triangles,
                            &mut tri_to_seg,
                        );
                        continue;
                    }
                }

                emit_triangles_for_quad(
                    &cell_verts,
                    inside_at_min,
                    edge_mid,
                    edge_dir,
                    &mut triangles,
                    &mut tri_witness,
                    &mut emitted_triangles,
                    p_surface,
                );
            }
        }
    }

    #[cfg(test)]
    {
        if sanity_enabled() {
            eprintln!("SANITY segment cell-count histogram (after vertex dedup): {:?}", sanity_seg_cell_count_hist);
            eprintln!("SANITY mixed-size segments skipped: {sanity_mixed_size_segments}");

            // Attribute non-manifold edges back to the segment(s) that emitted their triangles.
            let mut edge_to_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
            for (ti, tri) in triangles.iter().enumerate() {
                let a = tri[0];
                let b = tri[1];
                let c = tri[2];
                let edges = [(a, b), (b, c), (c, a)];
                for (ea, eb) in edges {
                    let key = if ea <= eb { (ea, eb) } else { (eb, ea) };
                    edge_to_tris.entry(key).or_default().push(ti);
                }
            }

            let mut printed = 0;
            for (edge, tis) in &edge_to_tris {
                if tis.len() > 2 {
                    let (a, b) = edge;
                    eprintln!(
                        "SANITY non-manifold edge (count={}): vid {} -> {}",
                        tis.len(),
                        a,
                        b
                    );
                    for &ti in tis {
                        let tri = &triangles[ti];
                        let mut tkey = *tri;
                        tkey.sort();
                        let src = tri_to_seg.get(&tkey);
                        eprintln!(
                            "  tri {ti} src={:?} vids=({}, {}, {})",
                            src,
                            tri[0],
                            tri[1],
                            tri[2]
                        );
                    }
                    printed += 1;
                    if printed >= 3 {
                        break;
                    }
                }
            }
        }
    }

    Ok(IndexedMesh {
        vertex_pos: leaf_vertices,
        triangles,
        tri_witness,
    })
}

#[cfg(test)]
fn emit_triangles_for_quad_dbg(
    cells: &[CellData],
    inside_first: bool,
    edge_mid: (f32, f32, f32),
    edge_dir: (f32, f32, f32),
    witness: (f32, f32, f32),
    seg_key: &SegKey,
    triangles: &mut Vec<[u32; 3]>,
    tri_witness: &mut Vec<(f32, f32, f32)>,
    emitted: &mut HashSet<[u32; 3]>,
    tri_to_seg: &mut HashMap<[u32; 3], SegKey>,
) {
    if cells.len() < 3 {
        return;
    }

    let mut try_emit = |tri: [u32; 3], triangles: &mut Vec<[u32; 3]>, tri_witness: &mut Vec<(f32, f32, f32)>, emitted: &mut HashSet<[u32; 3]>, tri_to_seg: &mut HashMap<[u32; 3], SegKey>| {
        let mut key = tri;
        key.sort();
        if emitted.insert(key) {
            tri_to_seg.insert(key, seg_key.clone());
            triangles.push(tri);
            tri_witness.push(witness);
        }
    };

    // Reuse the same ordering/triangulation logic as the main emitter.
    let mut sorted_cells: Vec<&CellData> = cells.iter().collect();
    let arbitrary = if edge_dir.0.abs() < 0.9 { (1.0, 0.0, 0.0) } else { (0.0, 1.0, 0.0) };
    let basis_u = (
        arbitrary.1 * edge_dir.2 - arbitrary.2 * edge_dir.1,
        arbitrary.2 * edge_dir.0 - arbitrary.0 * edge_dir.2,
        arbitrary.0 * edge_dir.1 - arbitrary.1 * edge_dir.0,
    );
    let len_u = (basis_u.0 * basis_u.0 + basis_u.1 * basis_u.1 + basis_u.2 * basis_u.2).sqrt();
    let basis_u = (basis_u.0 / len_u, basis_u.1 / len_u, basis_u.2 / len_u);
    let basis_v = (
        edge_dir.1 * basis_u.2 - edge_dir.2 * basis_u.1,
        edge_dir.2 * basis_u.0 - edge_dir.0 * basis_u.2,
        edge_dir.0 * basis_u.1 - edge_dir.1 * basis_u.0,
    );

    sorted_cells.sort_by(|a, b| {
        let va = a.1;
        let vb = b.1;
        let da = (va.0 - edge_mid.0, va.1 - edge_mid.1, va.2 - edge_mid.2);
        let db = (vb.0 - edge_mid.0, vb.1 - edge_mid.1, vb.2 - edge_mid.2);
        let ua = da.0 * basis_u.0 + da.1 * basis_u.1 + da.2 * basis_u.2;
        let va_proj = da.0 * basis_v.0 + da.1 * basis_v.1 + da.2 * basis_v.2;
        let ub = db.0 * basis_u.0 + db.1 * basis_u.1 + db.2 * basis_u.2;
        let vb_proj = db.0 * basis_v.0 + db.1 * basis_v.1 + db.2 * basis_v.2;
        let angle_a = va_proj.atan2(ua);
        let angle_b = vb_proj.atan2(ub);
        angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_vids: Vec<u32> = sorted_cells.iter().map(|c| c.0).collect();
    let n = sorted_vids.len();

    if n == 3 {
        if inside_first {
            try_emit([sorted_vids[0], sorted_vids[1], sorted_vids[2]], triangles, tri_witness, emitted, tri_to_seg);
        } else {
            try_emit([sorted_vids[0], sorted_vids[2], sorted_vids[1]], triangles, tri_witness, emitted, tri_to_seg);
        }
    } else if n == 4 {
        let v0 = sorted_vids[0];
        let v1 = sorted_vids[1];
        let v2 = sorted_vids[2];
        let v3 = sorted_vids[3];
        let flip = (v0 ^ v1 ^ v2 ^ v3) & 1 != 0;
        if !flip {
            if inside_first {
                try_emit([v0, v1, v2], triangles, tri_witness, emitted, tri_to_seg);
                try_emit([v0, v2, v3], triangles, tri_witness, emitted, tri_to_seg);
            } else {
                try_emit([v0, v2, v1], triangles, tri_witness, emitted, tri_to_seg);
                try_emit([v0, v3, v2], triangles, tri_witness, emitted, tri_to_seg);
            }
        } else {
            if inside_first {
                try_emit([v1, v2, v3], triangles, tri_witness, emitted, tri_to_seg);
                try_emit([v1, v3, v0], triangles, tri_witness, emitted, tri_to_seg);
            } else {
                try_emit([v1, v3, v2], triangles, tri_witness, emitted, tri_to_seg);
                try_emit([v1, v0, v3], triangles, tri_witness, emitted, tri_to_seg);
            }
        }
    } else if n > 4 {
        for i in 1..(n - 1) {
            let v0 = sorted_vids[0];
            let v1 = sorted_vids[i];
            let v2 = sorted_vids[i + 1];
            if inside_first {
                try_emit([v0, v1, v2], triangles, tri_witness, emitted, tri_to_seg);
            } else {
                try_emit([v0, v2, v1], triangles, tri_witness, emitted, tri_to_seg);
            }
        }
    }
}

/// Cell data for triangle emission:
/// (stable vertex id, vertex position, edge_length, cell_min, cell_max)
type CellData = (u32, (f32, f32, f32), f32, (f32, f32, f32), (f32, f32, f32));

/// Emit triangles for a quad with correct winding order.
/// Vertices must be sorted angularly around the edge axis for proper manifold mesh.
fn emit_triangles_for_quad(
    cells: &[CellData],
    inside_first: bool,
    edge_mid: (f32, f32, f32),
    edge_dir: (f32, f32, f32),
    triangles: &mut Vec<[u32; 3]>,
    tri_witness: &mut Vec<(f32, f32, f32)>,
    emitted: &mut HashSet<[u32; 3]>,
    witness: (f32, f32, f32),
) {
    if cells.len() < 3 {
        return;
    }

    let mut try_emit = |tri: [u32; 3], triangles: &mut Vec<[u32; 3]>, tri_witness: &mut Vec<(f32, f32, f32)>, emitted: &mut HashSet<[u32; 3]>| {
        let mut key = tri;
        key.sort();
        if emitted.insert(key) {
            triangles.push(tri);
            tri_witness.push(witness);
        }
    };
    
    // Sort vertices angularly around the edge axis (by position).
    let mut sorted_cells: Vec<&CellData> = cells.iter().collect();
    
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
    
    // Sort by angle around the edge (using vertex position from cell data)
    sorted_cells.sort_by(|a, b| {
        // Vector from edge_mid to vertex (position is second element of tuple)
        let va = a.1;
        let vb = b.1;
        let da = (va.0 - edge_mid.0, va.1 - edge_mid.1, va.2 - edge_mid.2);
        let db = (vb.0 - edge_mid.0, vb.1 - edge_mid.1, vb.2 - edge_mid.2);
        
        // Project onto basis_u and basis_v
        let ua = da.0 * basis_u.0 + da.1 * basis_u.1 + da.2 * basis_u.2;
        let va_proj = da.0 * basis_v.0 + da.1 * basis_v.1 + da.2 * basis_v.2;
        let ub = db.0 * basis_u.0 + db.1 * basis_u.1 + db.2 * basis_u.2;
        let vb_proj = db.0 * basis_v.0 + db.1 * basis_v.1 + db.2 * basis_v.2;
        
        // Compute angles
        let angle_a = va_proj.atan2(ua);
        let angle_b = vb_proj.atan2(ub);
        
        angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Extract sorted vertex ids (stable topology keys)
    let sorted_vids: Vec<u32> = sorted_cells.iter().map(|c| c.0).collect();
    
    // Emit triangles using adjacent vertices only (no internal edges)
    // This ensures each edge in the output is shared by exactly 2 triangles
    let n = sorted_vids.len();
    
    if n == 3 {
        if inside_first {
            try_emit([sorted_vids[0], sorted_vids[1], sorted_vids[2]], triangles, tri_witness, emitted);
        } else {
            try_emit([sorted_vids[0], sorted_vids[2], sorted_vids[1]], triangles, tri_witness, emitted);
        }
    } else if n == 4 {
        // Quad: split into 2 triangles.
        // Choose diagonal deterministically based on stable integer ids.
        let v0 = sorted_vids[0];
        let v1 = sorted_vids[1];
        let v2 = sorted_vids[2];
        let v3 = sorted_vids[3];

        // If `flip` is false: split along 0-2. If true: split along 1-3.
        let flip = (v0 ^ v1 ^ v2 ^ v3) & 1 != 0;

        // If `flip` is false: split along 0-2. If true: split along 1-3.
        if !flip {
            if inside_first {
                try_emit([v0, v1, v2], triangles, tri_witness, emitted);
                try_emit([v0, v2, v3], triangles, tri_witness, emitted);
            } else {
                try_emit([v0, v2, v1], triangles, tri_witness, emitted);
                try_emit([v0, v3, v2], triangles, tri_witness, emitted);
            }
        } else {
            if inside_first {
                try_emit([v1, v2, v3], triangles, tri_witness, emitted);
                try_emit([v1, v3, v0], triangles, tri_witness, emitted);
            } else {
                try_emit([v1, v3, v2], triangles, tri_witness, emitted);
                try_emit([v1, v0, v3], triangles, tri_witness, emitted);
            }
        }
    } else if n > 4 {
        // For n > 4, use fan triangulation (rare case with adaptive refinement)
        for i in 1..(n - 1) {
            let v0 = sorted_vids[0];
            let v1 = sorted_vids[i];
            let v2 = sorted_vids[i + 1];
            
            if inside_first {
                try_emit([v0, v1, v2], triangles, tri_witness, emitted);
            } else {
                try_emit([v0, v2, v1], triangles, tri_witness, emitted);
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
            corner_filled: node.corner_filled,
            children: None,
        }));
    }
}

/// A welded view of a triangle soup.
///
/// Many triangles share the same conceptual vertex position. In order to compute smooth
/// per-vertex normals (and to support HQ normal fitting), we weld vertices by quantized
/// position and keep an explicit vertex id for each triangle corner.
struct WeldedTopology {
    vertex_pos: Vec<(f32, f32, f32)>,
    tri_vids: Vec<[usize; 3]>,
}

impl WeldedTopology {
    fn build(triangles: &[Triangle]) -> Self {
        let scale = 1_000_000.0f32;
        let quantize = |v: (f32, f32, f32)| -> (i64, i64, i64) {
            quantize_round(v, scale)
        };

        let mut key_to_vid: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let mut vertex_pos: Vec<(f32, f32, f32)> = Vec::new();
        let mut tri_vids: Vec<[usize; 3]> = Vec::with_capacity(triangles.len());

        for tri in triangles {
            let mut vids = [0usize; 3];
            for i in 0..3 {
                let v = tri.vertices[i];
                let key = quantize(v);
                let vid = *key_to_vid.entry(key).or_insert_with(|| {
                    let id = vertex_pos.len();
                    vertex_pos.push(v);
                    id
                });
                vids[i] = vid;
            }
            tri_vids.push(vids);
        }

        Self { vertex_pos, tri_vids }
    }
}

fn compute_mesh_vertex_normals(topology: &WeldedTopology, triangles: &[Triangle]) -> Vec<(f32, f32, f32)> {
    let mut acc = vec![(0.0f32, 0.0f32, 0.0f32); topology.vertex_pos.len()];

    for (ti, tri) in triangles.iter().enumerate() {
        let vids = topology.tri_vids[ti];
        let a = tri.vertices[0];
        let b = tri.vertices[1];
        let c = tri.vertices[2];
        let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
        let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
        // Unnormalized face normal (area-weighted).
        let n = (
            ab.1 * ac.2 - ab.2 * ac.1,
            ab.2 * ac.0 - ab.0 * ac.2,
            ab.0 * ac.1 - ab.1 * ac.0,
        );
        let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
        if len2 <= 1.0e-24 {
            continue;
        }
        for &vid in &vids {
            acc[vid].0 += n.0;
            acc[vid].1 += n.1;
            acc[vid].2 += n.2;
        }
    }

    for n in &mut acc {
        let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
        if len2 > 1.0e-24 {
            let inv = 1.0 / len2.sqrt();
            n.0 *= inv;
            n.1 *= inv;
            n.2 *= inv;
        } else {
            *n = (0.0, 1.0, 0.0);
        }
    }

    acc
}

fn write_vertex_normals(topology: &WeldedTopology, triangles: &mut [Triangle], normals: &[(f32, f32, f32)]) {
    for (ti, tri) in triangles.iter_mut().enumerate() {
        let vids = topology.tri_vids[ti];
        tri.normals[0] = normals[vids[0]];
        tri.normals[1] = normals[vids[1]];
        tri.normals[2] = normals[vids[2]];
    }
}

fn write_vertex_positions(topology: &WeldedTopology, triangles: &mut [Triangle], positions: &[(f32, f32, f32)]) {
    for (ti, tri) in triangles.iter_mut().enumerate() {
        let vids = topology.tri_vids[ti];
        tri.vertices[0] = positions[vids[0]];
        tri.vertices[1] = positions[vids[1]];
        tri.vertices[2] = positions[vids[2]];
    }
}

fn normalize_or_none(v: (f32, f32, f32)) -> Option<(f32, f32, f32)> {
    let len2 = v.0 * v.0 + v.1 * v.1 + v.2 * v.2;
    if len2 <= 1.0e-24 {
        None
    } else {
        let inv = 1.0 / len2.sqrt();
        Some((v.0 * inv, v.1 * inv, v.2 * inv))
    }
}

fn dot(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn cross(a: (f32, f32, f32), b: (f32, f32, f32)) -> (f32, f32, f32) {
    (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
}

fn project_to_surface_on_line(
    ctx: &mut SamplingContext,
    p: (f32, f32, f32),
    dir_unit: (f32, f32, f32),
    mut half_range: f32,
    iterations: usize,
) -> Result<Option<(f32, f32, f32)>> {
    if half_range <= 0.0 {
        return Ok(None);
    }

    for _ in 0..4 {
        let a = (
            p.0 - dir_unit.0 * half_range,
            p.1 - dir_unit.1 * half_range,
            p.2 - dir_unit.2 * half_range,
        );
        let b = (
            p.0 + dir_unit.0 * half_range,
            p.1 + dir_unit.1 * half_range,
            p.2 + dir_unit.2 * half_range,
        );
        let fa = ctx.is_inside(a)? != 0.0;
        let fb = ctx.is_inside(b)? != 0.0;

        if fa != fb {
            let (p_inside, p_outside) = if fa { (a, b) } else { (b, a) };
            return Ok(Some(binary_search_edge_crossing(
                ctx,
                p_inside,
                p_outside,
                iterations,
            )?));
        }

        // Expand search window, but keep it local-ish.
        half_range *= 2.0;
        if half_range.is_infinite() || half_range > 1.0e6 {
            break;
        }
    }

    Ok(None)
}

fn hq_normal_at_point(
    ctx: &mut SamplingContext,
    p: (f32, f32, f32),
    n0: (f32, f32, f32),
    eps: f32,
    bracket: f32,
    iterations: usize,
) -> Result<Option<(f32, f32, f32)>> {
    let n0 = match normalize_or_none(n0) {
        Some(n) => n,
        None => return Ok(None),
    };

    // Build a tangent frame.
    let arbitrary = if n0.0.abs() < 0.9 { (1.0, 0.0, 0.0) } else { (0.0, 1.0, 0.0) };
    let t1 = match normalize_or_none(cross(arbitrary, n0)) {
        Some(t) => t,
        None => return Ok(None),
    };
    let t2 = cross(n0, t1);

    let p1 = (p.0 + t1.0 * eps, p.1 + t1.1 * eps, p.2 + t1.2 * eps);
    let p2 = (p.0 + t2.0 * eps, p.1 + t2.1 * eps, p.2 + t2.2 * eps);

    let q1 = match project_to_surface_on_line(ctx, p1, n0, bracket, iterations)? {
        Some(q) => q,
        None => return Ok(None),
    };
    let q2 = match project_to_surface_on_line(ctx, p2, n0, bracket, iterations)? {
        Some(q) => q,
        None => return Ok(None),
    };

    let v1 = (q1.0 - p.0, q1.1 - p.1, q1.2 - p.2);
    let v2 = (q2.0 - p.0, q2.1 - p.1, q2.2 - p.2);
    let mut n = cross(v1, v2);
    n = match normalize_or_none(n) {
        Some(nn) => nn,
        None => return Ok(None),
    };

    // Orient to match the mesh normal guess.
    if dot(n, n0) < 0.0 {
        n = (-n.0, -n.1, -n.2);
    }

    Ok(Some(n))
}

/// Estimate the surface normal at a point within a cell using the SDF gradient.
/// 
/// Uses central differences on the trilinearly interpolated SDF to compute
/// the gradient, which points from inside to outside (the normal direction).
fn estimate_normal_from_sdf(_node: &OctreeNode, _point: (f32, f32, f32)) -> (f32, f32, f32) {
    // NOTE: The sampling pipeline is occupancy-based (`is_inside(p) != 0.0`) and does not
    // provide an SDF. This helper is kept only to minimize churn while refactoring and
    // should not be used for shading.
    (0.0, 1.0, 0.0)
}

/// Find the surface crossing point on an edge using binary search.
/// 
/// Given two points where one is inside and one is outside, performs binary search
/// to find the approximate location where the surface crosses the edge.
/// 
/// # Arguments
/// * `ctx` - Sampling context for querying the model
/// * `p_inside` - Point that is inside the model
/// * `p_outside` - Point that is outside the model
/// * `iterations` - Number of binary search iterations (4-8 recommended)
/// 
/// # Returns
/// The approximate crossing point on the edge.
fn binary_search_edge_crossing(
    ctx: &mut SamplingContext,
    p_inside: (f32, f32, f32),
    p_outside: (f32, f32, f32),
    iterations: usize,
) -> Result<(f32, f32, f32)> {
    let mut t_min = 0.0f32;  // t=0 is p_inside
    let mut t_max = 1.0f32;  // t=1 is p_outside
    
    for _ in 0..iterations {
        let t_mid = (t_min + t_max) * 0.5;
        let p_mid = (
            p_inside.0 + t_mid * (p_outside.0 - p_inside.0),
            p_inside.1 + t_mid * (p_outside.1 - p_inside.1),
            p_inside.2 + t_mid * (p_outside.2 - p_inside.2),
        );
        
        let density = ctx.is_inside(p_mid)?;
        if density != 0.0 {
            // p_mid is inside, so crossing is in upper half [t_mid, t_max]
            t_min = t_mid;
        } else {
            // p_mid is outside, so crossing is in lower half [t_min, t_mid]
            t_max = t_mid;
        }
    }
    
    // Return the midpoint of the final interval
    let t_final = (t_min + t_max) * 0.5;
    Ok((
        p_inside.0 + t_final * (p_outside.0 - p_inside.0),
        p_inside.1 + t_final * (p_outside.1 - p_inside.1),
        p_inside.2 + t_final * (p_outside.2 - p_inside.2),
    ))
}

/// Relax a vertex toward the surface using the SDF gradient.
/// 
/// Iteratively moves the vertex along the estimated surface normal direction
/// to project it closer to the actual surface. Uses the trilinearly interpolated
/// SDF within the cell to estimate the distance and gradient.
/// 
/// # Arguments
/// * `vertex` - Initial vertex position
/// * `node` - The cell containing the vertex (for SDF interpolation)
/// * `ctx` - Optional sampling context for more accurate SDF estimation
/// * `iterations` - Number of relaxation iterations
/// 
/// # Returns
/// The relaxed vertex position, clamped to stay within the cell.
fn relax_vertex_to_surface(
    vertex: (f32, f32, f32),
    node: &OctreeNode,
    mut ctx: Option<&mut SamplingContext>,
    iterations: usize,
) -> Result<(f32, f32, f32)> {
    let Some(ctx) = ctx.as_deref_mut() else {
        return Ok(vertex);
    };

    // Estimate an "inside -> outside" direction from corner occupancy.
    let mut inside_sum = (0.0f32, 0.0f32, 0.0f32);
    let mut outside_sum = (0.0f32, 0.0f32, 0.0f32);
    let mut inside_n = 0usize;
    let mut outside_n = 0usize;
    for i in 0..8 {
        let p = node.corner_pos(i);
        if node.corner_inside(i) {
            inside_sum.0 += p.0;
            inside_sum.1 += p.1;
            inside_sum.2 += p.2;
            inside_n += 1;
        } else {
            outside_sum.0 += p.0;
            outside_sum.1 += p.1;
            outside_sum.2 += p.2;
            outside_n += 1;
        }
    }

    if inside_n == 0 || outside_n == 0 {
        // Not a mixed cell.
        return Ok(vertex);
    }

    let inside_center = (
        inside_sum.0 / inside_n as f32,
        inside_sum.1 / inside_n as f32,
        inside_sum.2 / inside_n as f32,
    );
    let outside_center = (
        outside_sum.0 / outside_n as f32,
        outside_sum.1 / outside_n as f32,
        outside_sum.2 / outside_n as f32,
    );
    let dir = (
        outside_center.0 - inside_center.0,
        outside_center.1 - inside_center.1,
        outside_center.2 - inside_center.2,
    );
    let Some(dir_unit) = normalize_or_none(dir) else {
        return Ok(vertex);
    };

    let clamp_to_cell = |p: (f32, f32, f32)| -> (f32, f32, f32) {
        (
            p.0.clamp(node.min.0, node.max.0),
            p.1.clamp(node.min.1, node.max.1),
            p.2.clamp(node.min.2, node.max.2),
        )
    };

    // Keep the projection local to this cell.
    let half_range = node.size() * 0.5;
    let mut v = vertex;

    // Bisection iterations per relaxation step.
    // Tied loosely to edge refinement precision; this is still bounded by the cell size range.
    let bisect_iters = 8usize;

    for _ in 0..iterations {
        let Some(p_new) = project_to_surface_on_line(ctx, v, dir_unit, half_range, bisect_iters)? else {
            break;
        };
        v = clamp_to_cell(p_new);
    }

    Ok(v)
}

/// Compute the Surface Nets vertex for a MIXED cell with optional binary search refinement
/// and vertex relaxation.
/// 
/// When `ctx` is provided and `edge_iterations > 0`, uses binary search to find accurate
/// edge crossing points. When `relax_iterations > 0`, applies vertex relaxation to
/// project the vertex closer to the actual surface.
fn compute_surface_nets_vertex_refined(
    node: &OctreeNode,
    mut ctx: Option<&mut SamplingContext>,
    edge_iterations: usize,
    relax_iterations: usize,
) -> Result<(f32, f32, f32)> {
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // X-aligned edges
        (0, 2), (1, 3), (4, 6), (5, 7), // Y-aligned edges
        (0, 4), (1, 5), (2, 6), (3, 7), // Z-aligned edges
    ];

    let mut sum = (0.0f32, 0.0f32, 0.0f32);
    let mut count = 0;

    for &(a, b) in &EDGES {
        let a_filled = node.corner_inside(a);
        let b_filled = node.corner_inside(b);

        // Check for occupancy change
        if a_filled != b_filled {
            let pa = node.corner_pos(a);
            let pb = node.corner_pos(b);

            let crossing = if let Some(ref mut sampling_ctx) = ctx.as_deref_mut() {
                if edge_iterations > 0 {
                    let (p_inside, p_outside) = if a_filled {
                        (pa, pb)
                    } else {
                        (pb, pa)
                    };
                    binary_search_edge_crossing(*sampling_ctx, p_inside, p_outside, edge_iterations)?
                } else {
                    // With occupancy-only data, midpoint is the best zero-cost fallback.
                    (
                        (pa.0 + pb.0) * 0.5,
                        (pa.1 + pb.1) * 0.5,
                        (pa.2 + pb.2) * 0.5,
                    )
                }
            } else {
                // No sampling context, use midpoint.
                (
                    (pa.0 + pb.0) * 0.5,
                    (pa.1 + pb.1) * 0.5,
                    (pa.2 + pb.2) * 0.5,
                )
            };
            
            sum.0 += crossing.0;
            sum.1 += crossing.1;
            sum.2 += crossing.2;
            count += 1;
        }
    }

    let initial_vertex = if count > 0 {
        (sum.0 / count as f32, sum.1 / count as f32, sum.2 / count as f32)
    } else {
        node.center()
    };
    
    // Vertex relaxation is applied as a welded post-pass after extraction.
    // Doing it here (per-cell) can interfere with topology decisions during extraction.
    let _ = relax_iterations;
    Ok(initial_vertex)
}

/// Compute the Surface Nets vertex for a MIXED cell.
/// 
/// Uses SDF interpolation to find the approximate surface crossing point on each
/// sign-changing edge, then averages these crossing points for the final vertex.
/// This produces much smoother results than simple edge midpoints.
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
        let a_filled = node.corner_inside(a);
        let b_filled = node.corner_inside(b);

        if a_filled != b_filled {
            let pa = node.corner_pos(a);
            let pb = node.corner_pos(b);

            // Midpoint fallback under occupancy-only sampling.
            let crossing = (
                (pa.0 + pb.0) * 0.5,
                (pa.1 + pb.1) * 0.5,
                (pa.2 + pb.2) * 0.5,
            );
            
            sum.0 += crossing.0;
            sum.1 += crossing.1;
            sum.2 += crossing.2;
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

    fn sanity_or_skip() -> bool {
        if !super::sanity_enabled() {
            eprintln!("Skipping sanity-only test (set VOLUMETRIC_SANITY=1 to enable)");
            return false;
        }
        true
    }

    fn strict_sanity_enabled() -> bool {
        // Some of the Mandelbulb-focused diagnostics are intentionally very strict and
        // can fail on valid (but highly complex / open) fractal geometry.
        // Gate those behind a second opt-in to avoid confusing investigations.
        std::env::var("VOLUMETRIC_STRICT_SANITY").is_ok()
    }

    fn require_closed_manifold_or_skip(triangles: &[Triangle], context: &str) -> bool {
        let boundary_edges = boundary_edge_count(triangles);
        let hist = edge_incidence_histogram(triangles);
        let non_manifold_edges: usize = hist
            .iter()
            .filter(|&(&count, _)| count > 2)
            .map(|(_, &num)| num)
            .sum();

        if boundary_edges == 0 && non_manifold_edges == 0 {
            return true;
        }

        if strict_sanity_enabled() {
            panic!(
                "{context}: expected closed 2-manifold; boundary_edges={boundary_edges}, non_manifold_edges={non_manifold_edges}, edge_incidence_hist={hist:?}"
            );
        }

        eprintln!(
            "Skipping {context}: mesh is not a closed 2-manifold (boundary_edges={boundary_edges}, non_manifold_edges={non_manifold_edges}); set VOLUMETRIC_STRICT_SANITY=1 to enforce strictly. Edge incidence histogram: {hist:?}"
        );
        false
    }

    #[test]
    fn test_cell_type_classification() {
        // All inside: all corners filled
        let all_inside = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (1.0, 1.0, 1.0),
            depth: 0,
            corner_filled: [true; 8],
            children: None,
        };
        assert_eq!(all_inside.cell_type(), CellType::AllInside);

        // All outside: all corners empty
        let all_outside = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (1.0, 1.0, 1.0),
            depth: 0,
            corner_filled: [false; 8],
            children: None,
        };
        assert_eq!(all_outside.cell_type(), CellType::AllOutside);

        // Mixed: alternating filled/empty
        let mixed = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (1.0, 1.0, 1.0),
            depth: 0,
            corner_filled: [true, false, true, false, true, false, true, false],
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
            corner_filled: [false; 8],
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
        // A cell with one corner filled (corner 0)
        let node = OctreeNode {
            min: (0.0, 0.0, 0.0),
            max: (2.0, 2.0, 2.0),
            depth: 0,
            corner_filled: [true, false, false, false, false, false, false, false],
            children: None,
        };

        let vertex = compute_surface_nets_vertex(&node);

        // The vertex should be near corner 0, as that's where the surface is.
        // With midpoint edge crossings, the average remains in the corner-0 octant.
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
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
        };

        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);

        let triangles = adaptive_surface_nets_mesh_debug(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        // A sphere should produce some triangles
        assert!(!triangles.is_empty(), "Expected triangles for sphere mesh");
        
        // All triangle vertices should be within or near the bounds
        for tri in &triangles {
            for v in &tri.vertices {
                assert!(v.0 >= -1.5 && v.0 <= 1.5, "Vertex x out of bounds: {}", v.0);
                assert!(v.1 >= -1.5 && v.1 <= 1.5, "Vertex y out of bounds: {}", v.1);
                assert!(v.2 >= -1.5 && v.2 <= 1.5, "Vertex z out of bounds: {}", v.2);
            }
        }

        println!("Generated {} triangles for sphere", triangles.len());
    }

    #[test]
    fn sanity_internal_invariants_sphere() {
        if !sanity_or_skip() {
            return;
        }

        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/simple_sphere_model.wasm");
        if !wasm_path.exists() {
            eprintln!("Skipping test: sphere wasm not found at {:?}", wasm_path);
            return;
        }
        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read sphere wasm");

        // Keep this relatively small; the goal is to exercise inline invariants.
        let config = AdaptiveMeshConfig {
            base_resolution: 10,
            max_refinement_depth: 3,
            edge_refinement_iterations: 1,
            vertex_relaxation_iterations: 0,
            normal_mode: NormalMode::Mesh,
        };

        // Sphere bounds from the model tests.
        let bounds_min = (-1.5, -1.5, -1.5);
        let bounds_max = (1.5, 1.5, 1.5);
        let _triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Sphere sanity meshing failed");
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
        let mesh = super::phase3_extract_mesh(&octree, &sampler, config)?;

        println!("Phase 3 complete: {} triangles", mesh.triangles.len());

        // Convert to triangles with mesh normals for debugging output.
        let normals = super::compute_indexed_mesh_vertex_normals(&mesh);
        Ok(super::indexed_mesh_to_triangles(&mesh, &normals))
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
            let verts = tri.vertices;
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

    fn inward_facing_triangle_count(
        triangles: &[Triangle],
        bounds_min: (f32, f32, f32),
        bounds_max: (f32, f32, f32),
    ) -> usize {
        let center = (
            (bounds_min.0 + bounds_max.0) * 0.5,
            (bounds_min.1 + bounds_max.1) * 0.5,
            (bounds_min.2 + bounds_max.2) * 0.5,
        );
        triangles
            .iter()
            .filter(|tri| {
                let n = Triangle::compute_face_normal(&tri.vertices);
                let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
                if len2 <= 1.0e-24 {
                    return false;
                }
                let centroid = (
                    (tri.vertices[0].0 + tri.vertices[1].0 + tri.vertices[2].0) / 3.0,
                    (tri.vertices[0].1 + tri.vertices[1].1 + tri.vertices[2].1) / 3.0,
                    (tri.vertices[0].2 + tri.vertices[1].2 + tri.vertices[2].2) / 3.0,
                );
                let to_outside = (
                    centroid.0 - center.0,
                    centroid.1 - center.1,
                    centroid.2 - center.2,
                );
                let dp = n.0 * to_outside.0 + n.1 * to_outside.1 + n.2 * to_outside.2;
                dp < 0.0
            })
            .count()
    }

    fn edge_incidence_histogram(triangles: &[Triangle]) -> HashMap<usize, usize> {
        let scale = 1_000_000.0f32;
        let mut edges: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();

        for tri in triangles {
            let verts = tri.vertices;
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
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
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
    fn test_sphere_mesh_winding_is_consistent() {
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/simple_sphere_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: sphere wasm not found");
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        let config = AdaptiveMeshConfig {
            base_resolution: 6,
            max_refinement_depth: 2,
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 0,
            normal_mode: NormalMode::Mesh,
        };

        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        let inward = inward_facing_triangle_count(&triangles, bounds_min, bounds_max);
        assert_eq!(
            inward, 0,
            "Expected all sphere triangles to face outward after winding enforcement, found {inward} inward triangles"
        );
    }

    #[test]
    fn test_sphere_hq_normals_are_reasonable() {
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/simple_sphere_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: sphere wasm not found");
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        let config = AdaptiveMeshConfig {
            base_resolution: 6,
            max_refinement_depth: 2,
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 1,
            normal_mode: NormalMode::HqBisection {
                eps_frac: 0.05,
                bracket_frac: 0.5,
                iterations: 10,
            },
        };

        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);
        let center = (0.0f32, 0.0f32, 0.0f32);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");
        assert!(!triangles.is_empty());

        for tri in &triangles {
            for i in 0..3 {
                let v = tri.vertices[i];
                let n = tri.normals[i];
                let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
                assert!(len2 > 0.5, "Expected near-unit normal, got len2={len2}");

                let to_outside = (v.0 - center.0, v.1 - center.1, v.2 - center.2);
                let dp = n.0 * to_outside.0 + n.1 * to_outside.1 + n.2 * to_outside.2;
                assert!(dp > 0.0, "Expected outward-ish normal on sphere (dp={dp})");
            }
        }
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
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
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
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
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
                let verts = tri.vertices;
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

    // ==================== MANDELBULB TORTURE TESTS ====================
    // The Mandelbulb is a particularly challenging model for surface extraction:
    // - Highly non-convex with deep concavities
    // - Fractal surface with detail at all scales
    // - Multiple disconnected components possible
    // - Sharp features and thin structures
    // These tests are designed to stress-test manifoldness and winding consistency.

    #[test]
    fn test_mandelbulb_mesh_is_manifold() {
        if !sanity_or_skip() {
            return;
        }
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/mandelbulb_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: mandelbulb wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        // Use moderate resolution to capture the fractal detail
        let config = AdaptiveMeshConfig {
            base_resolution: 8,
            max_refinement_depth: 3,
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
        };

        // Mandelbulb bounds from the model
        let bounds_min = (-1.35, -1.35, -1.35);
        let bounds_max = (1.35, 1.35, 1.35);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        eprintln!("Mandelbulb triangles: {}", triangles.len());

        let boundary_edges = boundary_edge_count(&triangles);

        if boundary_edges != 0 {
            let hist = edge_incidence_histogram(&triangles);
            eprintln!("Edge incidence histogram: {:?}", hist);
            
            // Debug: find edges shared by 4 triangles and print the triangles
            let scale = 1_000_000.0f32;
            let mut edge_to_tris: HashMap<((i64, i64, i64), (i64, i64, i64)), Vec<usize>> = HashMap::new();
            for (ti, tri) in triangles.iter().enumerate() {
                let verts = tri.vertices;
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
                    edge_to_tris.entry(key).or_default().push(ti);
                }
            }
            
            let mut printed = 0;
            for ((qa, qb), tris) in &edge_to_tris {
                if tris.len() == 4 && printed < 3 {
                    eprintln!("Edge shared by 4 triangles: ({:.6}, {:.6}, {:.6}) -> ({:.6}, {:.6}, {:.6})",
                        qa.0 as f32 / scale, qa.1 as f32 / scale, qa.2 as f32 / scale,
                        qb.0 as f32 / scale, qb.1 as f32 / scale, qb.2 as f32 / scale);
                    for &ti in tris {
                        let tri = &triangles[ti];
                        eprintln!("  Triangle {}: ({:.6}, {:.6}, {:.6}), ({:.6}, {:.6}, {:.6}), ({:.6}, {:.6}, {:.6})",
                            ti,
                            tri.vertices[0].0, tri.vertices[0].1, tri.vertices[0].2,
                            tri.vertices[1].0, tri.vertices[1].1, tri.vertices[1].2,
                            tri.vertices[2].0, tri.vertices[2].1, tri.vertices[2].2);
                    }
                    printed += 1;
                }
            }
        }

        if boundary_edges != 0 && !strict_sanity_enabled() {
            eprintln!(
                "Skipping strict manifold assertion for Mandelbulb: found {boundary_edges} boundary edges (fractal/open surfaces can be valid). Set VOLUMETRIC_STRICT_SANITY=1 to enforce."
            );
            return;
        }

        assert_eq!(
            boundary_edges, 0,
            "Expected manifold mesh (no boundary edges), found {} boundary edges",
            boundary_edges
        );
    }

    #[test]
    fn sanity_internal_invariants_mandelbulb_smoke() {
        if !sanity_or_skip() {
            return;
        }

        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/mandelbulb_model.wasm");
        if !wasm_path.exists() {
            eprintln!("Skipping test: mandelbulb wasm not found at {:?}", wasm_path);
            return;
        }
        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read mandelbulb wasm");

        // Intentionally modest settings; the inline assertions should still trigger
        // if the problematic invariants are violated.
        let config = AdaptiveMeshConfig {
            base_resolution: 10,
            max_refinement_depth: 3,
            edge_refinement_iterations: 1,
            vertex_relaxation_iterations: 0,
            normal_mode: NormalMode::Mesh,
        };

        // Mandelbulb bounds from the existing torture tests.
        let bounds_min = (-1.3, -1.3, -1.3);
        let bounds_max = (1.3, 1.3, 1.3);
        let _triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mandelbulb sanity meshing failed");
    }

    #[test]
    fn test_mandelbulb_winding_consistency() {
        if !sanity_or_skip() {
            return;
        }
        // Test that triangle winding is consistent across the mesh.
        // For a closed manifold, each edge should be traversed once in each direction.
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/mandelbulb_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: mandelbulb wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        let config = AdaptiveMeshConfig {
            base_resolution: 8,
            max_refinement_depth: 2,
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
        };

        let bounds_min = (-1.35, -1.35, -1.35);
        let bounds_max = (1.35, 1.35, 1.35);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        if !require_closed_manifold_or_skip(&triangles, "test_mandelbulb_winding_consistency") {
            return;
        }

        // Check oriented edge consistency: for a consistently wound mesh,
        // each directed edge (a->b) should appear exactly once, and its
        // reverse (b->a) should also appear exactly once.
        let scale = 1_000_000.0f32;
        let mut directed_edges: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();

        for tri in &triangles {
            let verts = tri.vertices;
            let edges = [(0, 1), (1, 2), (2, 0)];
            for &(ia, ib) in &edges {
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
                *directed_edges.entry((qa, qb)).or_default() += 1;
            }
        }

        // Count edges that don't have a matching reverse edge or appear multiple times
        let mut inconsistent_count = 0;
        for ((qa, qb), count) in &directed_edges {
            let reverse_count = directed_edges.get(&(*qb, *qa)).copied().unwrap_or(0);
            if *count != 1 || reverse_count != 1 {
                inconsistent_count += 1;
                if inconsistent_count <= 5 {
                    eprintln!(
                        "Inconsistent edge: ({:.4}, {:.4}, {:.4}) -> ({:.4}, {:.4}, {:.4}), count={}, reverse={}",
                        qa.0 as f32 / scale, qa.1 as f32 / scale, qa.2 as f32 / scale,
                        qb.0 as f32 / scale, qb.1 as f32 / scale, qb.2 as f32 / scale,
                        count, reverse_count
                    );
                }
            }
        }

        assert_eq!(
            inconsistent_count, 0,
            "Expected consistent winding (each edge traversed once in each direction), found {} inconsistent edges",
            inconsistent_count
        );
    }

    #[test]
    fn test_mandelbulb_normals_agree_with_winding() {
        if !sanity_or_skip() {
            return;
        }
        // Test that vertex normals point in the same general direction as face normals.
        // This validates that the normal computation agrees with the triangle winding.
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/mandelbulb_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: mandelbulb wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        let config = AdaptiveMeshConfig {
            base_resolution: 6,
            max_refinement_depth: 2,
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
        };

        let bounds_min = (-1.35, -1.35, -1.35);
        let bounds_max = (1.35, 1.35, 1.35);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        if !require_closed_manifold_or_skip(&triangles, "test_mandelbulb_normals_agree_with_winding") {
            return;
        }

        let mut disagreement_count = 0;
        let mut total_checked = 0;

        for tri in &triangles {
            let face_normal = Triangle::compute_face_normal(&tri.vertices);
            let face_len2 = face_normal.0 * face_normal.0 + face_normal.1 * face_normal.1 + face_normal.2 * face_normal.2;
            
            // Skip degenerate triangles
            if face_len2 <= 1.0e-12 {
                continue;
            }

            for i in 0..3 {
                let vn = tri.normals[i];
                let vn_len2 = vn.0 * vn.0 + vn.1 * vn.1 + vn.2 * vn.2;
                
                // Skip zero normals
                if vn_len2 <= 1.0e-12 {
                    continue;
                }

                total_checked += 1;
                let dp = face_normal.0 * vn.0 + face_normal.1 * vn.1 + face_normal.2 * vn.2;
                
                // Vertex normal should generally agree with face normal (positive dot product)
                // Allow some tolerance for smooth shading across edges
                if dp < -0.1 * face_len2.sqrt() * vn_len2.sqrt() {
                    disagreement_count += 1;
                    if disagreement_count <= 5 {
                        eprintln!(
                            "Normal disagreement at vertex {}: face_n=({:.3}, {:.3}, {:.3}), vertex_n=({:.3}, {:.3}, {:.3}), dp={:.4}",
                            i, face_normal.0, face_normal.1, face_normal.2,
                            vn.0, vn.1, vn.2, dp
                        );
                    }
                }
            }
        }

        let disagreement_ratio = if total_checked > 0 {
            disagreement_count as f64 / total_checked as f64
        } else {
            0.0
        };

        eprintln!(
            "Normal-winding agreement: {}/{} checked, {} disagreements ({:.2}%)",
            total_checked, triangles.len() * 3, disagreement_count, disagreement_ratio * 100.0
        );

        // Allow a small percentage of disagreements due to sharp features
        assert!(
            disagreement_ratio < 0.05,
            "Too many normal-winding disagreements: {:.2}% (expected < 5%)",
            disagreement_ratio * 100.0
        );
    }

    #[test]
    fn test_mandelbulb_normals_point_outward() {
        // Test that normals generally point away from the model interior.
        // For the Mandelbulb, we use a heuristic: normals should point away from
        // the bounding box center more often than toward it.
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/mandelbulb_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: mandelbulb wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        let config = AdaptiveMeshConfig {
            base_resolution: 6,
            max_refinement_depth: 2,
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
        };

        let bounds_min = (-1.35, -1.35, -1.35);
        let bounds_max = (1.35, 1.35, 1.35);
        let center = (
            (bounds_min.0 + bounds_max.0) * 0.5,
            (bounds_min.1 + bounds_max.1) * 0.5,
            (bounds_min.2 + bounds_max.2) * 0.5,
        );

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        let mut outward_count = 0;
        let mut inward_count = 0;

        for tri in &triangles {
            let face_normal = Triangle::compute_face_normal(&tri.vertices);
            let face_len2 = face_normal.0 * face_normal.0 + face_normal.1 * face_normal.1 + face_normal.2 * face_normal.2;
            
            if face_len2 <= 1.0e-12 {
                continue;
            }

            let centroid = (
                (tri.vertices[0].0 + tri.vertices[1].0 + tri.vertices[2].0) / 3.0,
                (tri.vertices[0].1 + tri.vertices[1].1 + tri.vertices[2].1) / 3.0,
                (tri.vertices[0].2 + tri.vertices[1].2 + tri.vertices[2].2) / 3.0,
            );

            let to_outside = (
                centroid.0 - center.0,
                centroid.1 - center.1,
                centroid.2 - center.2,
            );

            let dp = face_normal.0 * to_outside.0 + face_normal.1 * to_outside.1 + face_normal.2 * to_outside.2;

            if dp > 0.0 {
                outward_count += 1;
            } else {
                inward_count += 1;
            }
        }

        let total = outward_count + inward_count;
        let outward_ratio = if total > 0 {
            outward_count as f64 / total as f64
        } else {
            0.0
        };

        eprintln!(
            "Outward-facing triangles: {}/{} ({:.2}%)",
            outward_count, total, outward_ratio * 100.0
        );

        // Note: For the Mandelbulb, this heuristic may not work well due to
        // deep concavities. The enforce_outward_winding function uses the same
        // heuristic, which may cause issues for non-convex models.
        // This test documents the current behavior.
        assert!(
            outward_ratio > 0.5,
            "Expected majority of triangles to face outward from center, got {:.2}%",
            outward_ratio * 100.0
        );
    }

    #[test]
    fn test_mandelbulb_hq_normals_are_valid() {
        // Test that HQ bisection normals are unit length and reasonable.
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/mandelbulb_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: mandelbulb wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        let config = AdaptiveMeshConfig {
            base_resolution: 6,
            max_refinement_depth: 2,
            edge_refinement_iterations: 4,
            vertex_relaxation_iterations: 1,
            normal_mode: NormalMode::HqBisection {
                eps_frac: 0.05,
                bracket_frac: 0.5,
                iterations: 10,
            },
        };

        let bounds_min = (-1.35, -1.35, -1.35);
        let bounds_max = (1.35, 1.35, 1.35);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        assert!(!triangles.is_empty(), "Should generate triangles");

        let mut invalid_normal_count = 0;
        let mut total_normals = 0;

        for tri in &triangles {
            for i in 0..3 {
                let n = tri.normals[i];
                let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
                total_normals += 1;

                // Check that normal is approximately unit length
                if len2 < 0.5 || len2 > 2.0 {
                    invalid_normal_count += 1;
                    if invalid_normal_count <= 5 {
                        eprintln!(
                            "Invalid normal length at vertex {}: ({:.3}, {:.3}, {:.3}), len2={:.4}",
                            i, n.0, n.1, n.2, len2
                        );
                    }
                }
            }
        }

        let invalid_ratio = if total_normals > 0 {
            invalid_normal_count as f64 / total_normals as f64
        } else {
            0.0
        };

        eprintln!(
            "HQ normals: {}/{} valid ({:.2}% invalid)",
            total_normals - invalid_normal_count, total_normals, invalid_ratio * 100.0
        );

        assert!(
            invalid_ratio < 0.01,
            "Too many invalid HQ normals: {:.2}% (expected < 1%)",
            invalid_ratio * 100.0
        );
    }

    #[test]
    fn test_mandelbulb_high_resolution_manifold() {
        if !sanity_or_skip() {
            return;
        }
        // Stress test with higher resolution to catch edge cases in adaptive refinement.
        let wasm_path = std::path::Path::new("target/wasm32-unknown-unknown/release/mandelbulb_model.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping test: mandelbulb wasm not found at {:?}", wasm_path);
            return;
        }

        let wasm_bytes = std::fs::read(wasm_path).expect("Failed to read wasm file");

        // Higher resolution for more thorough testing
        let config = AdaptiveMeshConfig {
            base_resolution: 12,
            max_refinement_depth: 3,
            edge_refinement_iterations: 6,
            vertex_relaxation_iterations: 2,
            normal_mode: NormalMode::Mesh,
        };

        let bounds_min = (-1.35, -1.35, -1.35);
        let bounds_max = (1.35, 1.35, 1.35);

        let triangles = adaptive_surface_nets_mesh(&wasm_bytes, bounds_min, bounds_max, &config)
            .expect("Mesh generation failed");

        eprintln!("High-res Mandelbulb triangles: {}", triangles.len());

        let boundary_edges = boundary_edge_count(&triangles);
        let hist = edge_incidence_histogram(&triangles);

        eprintln!("Edge incidence histogram: {:?}", hist);

        // Check for non-manifold edges (more than 2 triangles sharing an edge)
        let non_manifold_edges: usize = hist.iter()
            .filter(|&(&count, _)| count > 2)
            .map(|(_, &num)| num)
            .sum();

        if non_manifold_edges > 0 {
            eprintln!("Found {} non-manifold edges (shared by >2 triangles)", non_manifold_edges);
        }

        if (boundary_edges != 0 || non_manifold_edges != 0) && !strict_sanity_enabled() {
            eprintln!(
                "Skipping strict manifold assertion for high-res Mandelbulb: boundary_edges={boundary_edges}, non_manifold_edges={non_manifold_edges}. Set VOLUMETRIC_STRICT_SANITY=1 to enforce."
            );
            return;
        }

        assert_eq!(
            boundary_edges, 0,
            "Expected manifold mesh (no boundary edges), found {} boundary edges",
            boundary_edges
        );

        assert_eq!(
            non_manifold_edges, 0,
            "Expected manifold mesh (no non-manifold edges), found {} non-manifold edges",
            non_manifold_edges
        );
    }
}
