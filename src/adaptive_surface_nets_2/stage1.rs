//! Stage 1: Coarse Grid Discovery
//!
//! Samples the volume at low resolution to find regions containing the surface.
//! Returns a work queue of mixed cells with their corner samples pre-filled.

use std::sync::atomic::Ordering;

use crate::adaptive_surface_nets_2::types::{
    AdaptiveMeshConfig2, CornerMask, CuboidId, SamplerFn, SamplingStats, WorkQueueEntry,
};

/// Sample a single point and return whether it's inside (density > 0).
#[inline]
pub fn sample_is_inside<F>(sampler: &F, x: f64, y: f64, z: f64, stats: &SamplingStats) -> bool
where
    F: SamplerFn,
{
    stats.total_samples.fetch_add(1, Ordering::Relaxed);
    sampler(x, y, z) > 0.0
}

/// Stage 1: Coarse Grid Discovery
///
/// Samples the volume at low resolution to find regions containing the surface.
/// Returns a work queue of mixed cells with their corner samples pre-filled.
///
/// The grid is expanded by 1 cell on each side beyond the model bounds to
/// correctly handle models that perfectly fill their advertised bounding box.
/// This ensures surface detection at boundaries where the model meets empty space.
///
/// # Algorithm
/// 1. Create an expanded grid of (base_resolution + 3)³ sample points
///    - Covers cells from -1 to base_resolution (inclusive)
/// 2. Sample all corner points to determine inside/outside state
/// 3. For each cell, check if corners have mixed states (surface crosses cell)
/// 4. Return WorkQueueEntry for each mixed cell with all 8 corners known
pub fn stage1_coarse_discovery<F>(
    sampler: &F,
    bounds_min: (f64, f64, f64),
    bounds_max: (f64, f64, f64),
    config: &AdaptiveMeshConfig2,
    stats: &SamplingStats,
) -> Vec<WorkQueueEntry>
where
    F: SamplerFn,
{
    let res = config.base_resolution;

    // Expand the grid by 1 cell on each side to detect surfaces at the boundary.
    // This handles models that perfectly fill their advertised bounds.
    // Grid now covers cells from -1 to res (inclusive), so res+2 cells per axis.
    let expanded_cells = res + 2;
    let num_corners = expanded_cells + 1; // res + 3 corners per axis

    // Cell size at depth 0 (coarsest level) - based on original bounds
    let cell_size = (
        (bounds_max.0 - bounds_min.0) / res as f64,
        (bounds_max.1 - bounds_min.1) / res as f64,
        (bounds_max.2 - bounds_min.2) / res as f64,
    );

    // Expanded bounds: start one cell before bounds_min
    let expanded_min = (
        bounds_min.0 - cell_size.0,
        bounds_min.1 - cell_size.1,
        bounds_min.2 - cell_size.2,
    );

    // Sample all corner points into a 3D array
    // Layout: corners[z][y][x] for cache-friendly Z-slice iteration
    let mut corners = vec![vec![vec![false; num_corners]; num_corners]; num_corners];

    for iz in 0..num_corners {
        for iy in 0..num_corners {
            for ix in 0..num_corners {
                let x = expanded_min.0 + ix as f64 * cell_size.0;
                let y = expanded_min.1 + iy as f64 * cell_size.1;
                let z = expanded_min.2 + iz as f64 * cell_size.2;
                corners[iz][iy][ix] = sample_is_inside(sampler, x, y, z, stats);
            }
        }
    }

    // Find mixed cells and create work queue entries
    // Cell indices now go from -1 to res (i.e., 0..expanded_cells in array coords)
    let mut work_queue = Vec::new();

    for iz in 0..expanded_cells {
        for iy in 0..expanded_cells {
            for ix in 0..expanded_cells {
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
                    // CuboidId uses actual grid coordinates: -1 to res
                    let cuboid = CuboidId::new(ix as i32 - 1, iy as i32 - 1, iz as i32 - 1, 0);
                    let known_corners = cell_corners.map(Some);
                    work_queue.push(WorkQueueEntry::with_corners(cuboid, known_corners));
                }
            }
        }
    }

    work_queue
}
