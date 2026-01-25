//! Stage 1: Coarse Grid Discovery
//!
//! Samples the volume at low resolution to find regions containing the surface.
//! Returns a work queue of mixed cells with their corner samples pre-filled.

use std::sync::atomic::Ordering;

use crate::adaptive_surface_nets_2::types::{CornerMask, CuboidId, SamplerFn, SamplingStats, WorkQueueEntry};

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
/// 1. Create an expanded grid of (base_cells + 3) sample points per axis
///    - Covers cells from -1 to base_cells (inclusive)
/// 2. Sample all corner points to determine inside/outside state
/// 3. For each cell, check if corners have mixed states (surface crosses cell)
/// 4. Return WorkQueueEntry for each mixed cell with all 8 corners known
pub fn stage1_coarse_discovery<F>(
    sampler: &F,
    bounds_min: (f64, f64, f64),
    base_cells: (i32, i32, i32),
    base_cell_size: f64,
    stats: &SamplingStats,
) -> Vec<WorkQueueEntry>
where
    F: SamplerFn,
{
    // Expand the grid by 1 cell on each side to detect surfaces at the boundary.
    // This handles models that perfectly fill their advertised bounds.
    // Grid now covers cells from -1 to base_cells (inclusive).
    let expanded_cells = (
        (base_cells.0 + 2) as usize,
        (base_cells.1 + 2) as usize,
        (base_cells.2 + 2) as usize,
    );
    let num_corners = (
        expanded_cells.0 + 1,
        expanded_cells.1 + 1,
        expanded_cells.2 + 1,
    );

    // Expanded bounds: start one cell before bounds_min
    let expanded_min = (
        bounds_min.0 - base_cell_size,
        bounds_min.1 - base_cell_size,
        bounds_min.2 - base_cell_size,
    );

    // Sample all corner points into a 3D array
    // Layout: corners[z][y][x] for cache-friendly Z-slice iteration
    let mut corners = vec![vec![vec![false; num_corners.0]; num_corners.1]; num_corners.2];

    for iz in 0..num_corners.2 {
        for iy in 0..num_corners.1 {
            for ix in 0..num_corners.0 {
                let x = expanded_min.0 + ix as f64 * base_cell_size;
                let y = expanded_min.1 + iy as f64 * base_cell_size;
                let z = expanded_min.2 + iz as f64 * base_cell_size;
                corners[iz][iy][ix] = sample_is_inside(sampler, x, y, z, stats);
            }
        }
    }

    // Find mixed cells and create work queue entries
    // Cell indices now go from -1 to res (i.e., 0..expanded_cells in array coords)
    let mut work_queue = Vec::new();

    for iz in 0..expanded_cells.2 {
        for iy in 0..expanded_cells.1 {
            for ix in 0..expanded_cells.0 {
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
