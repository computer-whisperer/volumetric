//! Adaptive Surface Nets v2 - Modular implementation
//!
//! This module implements a high-quality mesh extraction algorithm for binary
//! (inside/outside) volumetric data. The algorithm operates in four stages:
//!
//! 1. **Stage 1: Coarse Grid Discovery** - Sample the volume at low resolution
//!    to identify regions containing the surface.
//!
//! 2. **Stage 2: Subdivision & Triangle Emission** - Recursively subdivide mixed
//!    cells to the target resolution and emit triangles using marching cubes.
//!
//! 3. **Stage 3: Topology Finalization** - Convert sparse triangle data to an
//!    indexed mesh with accumulated face normals.
//!
//! 4. **Stage 4: Vertex Refinement** (STUBBED) - The original implementation
//!    included vertex position refinement, normal recomputation, and sharp edge
//!    detection. This is currently stubbed out. See PHASE4_ARCHIVE.md for details.
//!
//! # Algorithm Overview
//!
//! For detailed algorithm documentation, see `adaptive_surface_nets_2.md`.

use std::sync::atomic::Ordering;
use web_time::Instant;

// Submodules
pub mod diagnostics;
pub mod lookup_tables;
pub mod parallel_iter;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage4_stub;
pub mod types;

#[cfg(test)]
mod tests;

// Re-exports for public API
pub use types::{
    AdaptiveMeshConfig2, IndexedMesh2, MeshingResult2, MeshingStats2, SamplerFn, SharpEdgeConfig,
};

#[cfg(feature = "normal-diagnostic")]
pub use types::NormalDiagnosticEntry;

#[cfg(feature = "edge-diagnostic")]
pub use types::{EdgeDiagnosticEntry, ReferenceEdgeInfo};

#[cfg(all(feature = "normal-diagnostic", feature = "native"))]
pub use diagnostics::run_normal_diagnostics;

#[cfg(all(feature = "edge-diagnostic", feature = "native"))]
pub use diagnostics::{run_crossing_diagnostics, run_edge_diagnostics};

use stage1::stage1_coarse_discovery;
use stage2::stage2_subdivision_and_emission;
use stage3::stage3_topology_finalization;
use stage4_stub::{stage4_result_to_mesh, stage4_vertex_refinement_stub};
use types::SamplingStats;

fn compute_base_cells(
    bounds_min: (f64, f64, f64),
    bounds_max: (f64, f64, f64),
    base_cell_size: f64,
) -> (i32, i32, i32) {
    let size_x = (bounds_max.0 - bounds_min.0).max(0.0);
    let size_y = (bounds_max.1 - bounds_min.1).max(0.0);
    let size_z = (bounds_max.2 - bounds_min.2).max(0.0);
    let cell = base_cell_size.max(1e-9);
    let cells_x = (size_x / cell).ceil().max(1.0) as i32;
    let cells_y = (size_y / cell).ceil().max(1.0) as i32;
    let cells_z = (size_z / cell).ceil().max(1.0) as i32;
    (cells_x, cells_y, cells_z)
}

/// Main entry point for adaptive surface nets meshing (native version).
///
/// # Arguments
/// * `sampler` - Function that returns > 0.0 if point is inside the model
/// * `bounds_min` - Minimum corner of bounding box
/// * `bounds_max` - Maximum corner of bounding box
/// * `config` - Algorithm configuration
///
/// # Returns
/// A MeshingResult2 containing the indexed mesh and detailed profiling statistics
#[cfg(feature = "native")]
pub fn adaptive_surface_nets_2<F>(
    sampler: F,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    config: &AdaptiveMeshConfig2,
) -> MeshingResult2
where
    F: SamplerFn,
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

    let base_cells = compute_base_cells(bounds_min_f64, bounds_max_f64, config.base_cell_size);
    let finest_cell_size = config.base_cell_size / (1 << config.max_depth) as f64;
    let cell_size = (finest_cell_size, finest_cell_size, finest_cell_size);

    // Stage 1: Coarse grid discovery
    let stage1_start = Instant::now();
    let samples_before_stage1 = stats.total_samples.load(Ordering::Relaxed);
    let initial_work_queue = stage1_coarse_discovery(
        &sampler,
        bounds_min_f64,
        base_cells,
        config.base_cell_size,
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
        base_cells,
        config,
        &stats,
    );
    let stage2_time = stage2_start.elapsed().as_secs_f64();
    let stage2_samples = stats.total_samples.load(Ordering::Relaxed) - samples_before_stage2;
    let stage2_cuboids = stats.cuboids_processed.load(Ordering::Relaxed) - cuboids_before_stage2;
    let stage2_triangles = stats.triangles_emitted.load(Ordering::Relaxed) - triangles_before_stage2;

    // Stage 3: Topology finalization
    let stage3_start = Instant::now();
    let stage3_result = stage3_topology_finalization(sparse_triangles, bounds_min_f64, cell_size);
    let stage3_time = stage3_start.elapsed().as_secs_f64();
    let stage3_unique_vertices = stage3_result.vertices.len();

    // Stage 4: Vertex refinement (STUBBED - passes through unchanged)
    let stage4_start = Instant::now();
    let samples_before_stage4 = stats.total_samples.load(Ordering::Relaxed);
    let stage4_result = stage4_vertex_refinement_stub(
        stage3_result,
        &sampler,
        cell_size,
        config,
        &stats,
    );
    let stage4_time = stage4_start.elapsed().as_secs_f64();
    let stage4_samples = stats.total_samples.load(Ordering::Relaxed) - samples_before_stage4;

    // Convert to final mesh
    let mesh = stage4_result_to_mesh(stage4_result);

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
        effective_resolution: (
            (base_cells.0 as usize) * (1 << config.max_depth),
            (base_cells.1 as usize) * (1 << config.max_depth),
            (base_cells.2 as usize) * (1 << config.max_depth),
        ),
        stage4_5_time_secs: 0.0,
        sharp_vertices_case1: 0,
        sharp_edge_crossings: 0,
        sharp_vertices_inserted: 0,
        sharp_vertices_duplicated: 0,
    };

    MeshingResult2 {
        mesh,
        stats: meshing_stats,
    }
}

/// Main entry point for adaptive surface nets meshing (web/sequential version).
///
/// This version does not require Send + Sync bounds since web is single-threaded.
/// Uses sequential iteration instead of parallel iteration.
#[cfg(not(feature = "native"))]
pub fn adaptive_surface_nets_2<F>(
    sampler: F,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    config: &AdaptiveMeshConfig2,
) -> MeshingResult2
where
    F: SamplerFn,
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

    let base_cells = compute_base_cells(bounds_min_f64, bounds_max_f64, config.base_cell_size);
    let finest_cell_size = config.base_cell_size / (1 << config.max_depth) as f64;
    let cell_size = (finest_cell_size, finest_cell_size, finest_cell_size);

    // Stage 1: Coarse grid discovery
    let stage1_start = Instant::now();
    let samples_before_stage1 = stats.total_samples.load(Ordering::Relaxed);
    let initial_work_queue = stage1_coarse_discovery(
        &sampler,
        bounds_min_f64,
        base_cells,
        config.base_cell_size,
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
        base_cells,
        config,
        &stats,
    );
    let stage2_time = stage2_start.elapsed().as_secs_f64();
    let stage2_samples = stats.total_samples.load(Ordering::Relaxed) - samples_before_stage2;
    let stage2_cuboids = stats.cuboids_processed.load(Ordering::Relaxed) - cuboids_before_stage2;
    let stage2_triangles = stats.triangles_emitted.load(Ordering::Relaxed) - triangles_before_stage2;

    // Stage 3: Topology finalization
    let stage3_start = Instant::now();
    let stage3_result = stage3_topology_finalization(sparse_triangles, bounds_min_f64, cell_size);
    let stage3_time = stage3_start.elapsed().as_secs_f64();
    let stage3_unique_vertices = stage3_result.vertices.len();

    // Stage 4: Vertex refinement (STUBBED - passes through unchanged)
    let stage4_start = Instant::now();
    let samples_before_stage4 = stats.total_samples.load(Ordering::Relaxed);
    let stage4_result = stage4_vertex_refinement_stub(
        stage3_result,
        &sampler,
        cell_size,
        config,
        &stats,
    );
    let stage4_time = stage4_start.elapsed().as_secs_f64();
    let stage4_samples = stats.total_samples.load(Ordering::Relaxed) - samples_before_stage4;

    // Convert to final mesh
    let mesh = stage4_result_to_mesh(stage4_result);

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
        effective_resolution: (
            (base_cells.0 as usize) * (1 << config.max_depth),
            (base_cells.1 as usize) * (1 << config.max_depth),
            (base_cells.2 as usize) * (1 << config.max_depth),
        ),
        stage4_5_time_secs: 0.0,
        sharp_vertices_case1: 0,
        sharp_edge_crossings: 0,
        sharp_vertices_inserted: 0,
        sharp_vertices_duplicated: 0,
    };

    MeshingResult2 {
        mesh,
        stats: meshing_stats,
    }
}
