//! Stage 2: Parallel Subdivision & Triangle Emission
//!
//! Processes the work queue from Stage 1, recursively subdividing mixed cells
//! until reaching max_depth, then emitting triangles using the MC lookup table.

use std::sync::atomic::Ordering;

use dashmap::DashSet;

use crate::adaptive_surface_nets_2::lookup_tables::{CORNER_OFFSETS, MC_EDGE_TO_OUR_EDGE, MC_TRI_TABLE, our_mask_to_mc_mask};
use crate::adaptive_surface_nets_2::parallel_iter;
use crate::adaptive_surface_nets_2::stage1::sample_is_inside;
use crate::adaptive_surface_nets_2::types::{
    AdaptiveMeshConfig2, CornerMask, CuboidId, SamplerFn, SamplingStats, SparseTriangle, WorkQueueEntry,
};

/// Compute the world-space position of a cell corner.
///
/// # Arguments
/// * `cell` - The cell ID
/// * `corner_index` - Corner index (0-7)
/// * `bounds_min` - World-space minimum bounds
/// * `cell_size` - Size of a cell at the finest level (per-axis)
/// * `max_depth` - Maximum refinement depth
#[inline]
pub fn corner_world_position(
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
pub fn complete_corner_samples<F>(
    entry: &mut WorkQueueEntry,
    sampler: &F,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
    max_depth: u8,
    stats: &SamplingStats,
) -> CornerMask
where
    F: SamplerFn,
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
pub fn subdivide_and_filter_mixed<F>(
    parent: &WorkQueueEntry,
    sampler: &F,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
    max_depth: u8,
    stats: &SamplingStats,
) -> Vec<WorkQueueEntry>
where
    F: SamplerFn,
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
pub fn shared_corners_with_neighbor(dx: i32, dy: i32, dz: i32) -> [(usize, usize); 4] {
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
    max_cells: (i32, i32, i32),
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
pub fn shared_face_is_mixed(current_corners: &[Option<bool>; 8], dx: i32, dy: i32, dz: i32) -> bool {
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
    base_cells: (i32, i32, i32),
    visited: &DashSet<CuboidId>,
    stats: &SamplingStats,
) -> ProcessedEntry
where
    F: SamplerFn,
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
        let scale = 1i32 << current_depth;
        let cells_at_depth = (
            base_cells.0 * scale,
            base_cells.1 * scale,
            base_cells.2 * scale,
        );

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
pub fn stage2_subdivision_and_emission<F>(
    initial_queue: Vec<WorkQueueEntry>,
    sampler: &F,
    bounds_min: (f64, f64, f64),
    cell_size: (f64, f64, f64),
    base_cells: (i32, i32, i32),
    config: &AdaptiveMeshConfig2,
    stats: &SamplingStats,
) -> Vec<SparseTriangle>
where
    F: SamplerFn,
{
    let max_depth = config.max_depth as u8;

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
        // Process current batch in parallel (or sequentially on web)
        let results: Vec<ProcessedEntry> = parallel_iter::map_vec(current_batch, |entry| {
            process_work_entry(
                entry,
                sampler,
                bounds_min,
                cell_size,
                max_depth,
                base_cells,
                &visited,
                stats,
            )
        });

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
