//! Smooth-region segmentation: the core of the region-based edge pipeline.
//!
//! Vertices are grouped into maximal smooth regions by seeded growth over mesh
//! adjacency. Sharp features are *not* detected directly; they emerge as the
//! boundaries between regions, where they can later be fitted from
//! region-labeled (face-pure by construction) samples.
//!
//! Growth is gated by two scale-invariant signals measured in
//! `patch_fit_bench`:
//! - the pairwise fitted-normal jump between adjacent vertices, which stays
//!   O(dihedral angle) across sharp features at any resolution but shrinks
//!   linearly with cell size on smooth surfaces, and
//! - the fit residual in cell units, which is an order of magnitude larger for
//!   feature-straddling vertices than for clean ones.
//!
//! High-residual vertices are never claimed; they form the feature zone
//! between regions and are resolved by later pipeline stages.

use crate::sharp_features::adjacency::MeshAdjacency;
use crate::sharp_features::fit::VertexFit;
use crate::sharp_features::fit::unsigned_angle_degrees;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct SegmentationConfig {
    /// Maximum fitted-normal angle (degrees) between adjacent vertices of the
    /// same region. Smooth-surface jumps at practical resolutions are a few
    /// degrees; sharp features measure ~40. Growth compares neighbors
    /// pairwise, so gradual curvature (e.g. around a cylinder) accumulates
    /// freely without tripping the gate.
    pub max_normal_jump_deg: f64,
    /// Only vertices with fit residual at or below this (cell units) may seed
    /// a new region. Seeds should be unambiguously clean.
    pub max_seed_residual_cells: f64,
    /// Vertices with residual above this are never claimed by any region;
    /// they form the unclaimed feature zone.
    pub max_grow_residual_cells: f64,
    /// Regions smaller than this are dissolved back to unclaimed; they are
    /// fit noise, not smooth faces.
    pub min_region_size: usize,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            max_normal_jump_deg: 15.0,
            max_seed_residual_cells: 0.02,
            max_grow_residual_cells: 0.06,
            min_region_size: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Segmentation {
    /// Region id per vertex; `None` for unclaimed (feature-zone) vertices.
    pub labels: Vec<Option<u32>>,
    pub region_count: usize,
    pub region_sizes: Vec<usize>,
}

/// Segment a mesh into smooth regions by seeded breadth-first growth.
///
/// Seeds are taken in ascending residual order, so regions start from the most
/// unambiguously smooth vertices and spread outward until they hit the gates.
pub fn segment_regions(
    adjacency: &MeshAdjacency,
    fits: &[Option<VertexFit>],
    config: &SegmentationConfig,
) -> Segmentation {
    let n = fits.len();
    let mut labels: Vec<Option<u32>> = vec![None; n];

    let mut seed_order: Vec<u32> = (0..n as u32)
        .filter(|&v| {
            fits[v as usize]
                .as_ref()
                .is_some_and(|f| f.residual_cells <= config.max_seed_residual_cells)
        })
        .collect();
    seed_order.sort_by(|&a, &b| {
        let ra = fits[a as usize].as_ref().unwrap().residual_cells;
        let rb = fits[b as usize].as_ref().unwrap().residual_cells;
        ra.total_cmp(&rb)
    });

    let mut regions: Vec<usize> = Vec::new();
    for &seed in &seed_order {
        if labels[seed as usize].is_some() {
            continue;
        }
        let region = regions.len() as u32;
        let mut size = 0usize;
        let mut queue = VecDeque::from([seed]);
        labels[seed as usize] = Some(region);
        while let Some(u) = queue.pop_front() {
            size += 1;
            let fit_u = fits[u as usize].as_ref().unwrap();
            for &w in adjacency.neighbors(u) {
                if labels[w as usize].is_some() {
                    continue;
                }
                let Some(fit_w) = fits[w as usize].as_ref() else {
                    continue;
                };
                if fit_w.residual_cells > config.max_grow_residual_cells {
                    continue;
                }
                if unsigned_angle_degrees(fit_u.normal, fit_w.normal) > config.max_normal_jump_deg {
                    continue;
                }
                labels[w as usize] = Some(region);
                queue.push_back(w);
            }
        }
        regions.push(size);
    }

    // Dissolve undersized regions and compact the surviving ids.
    let mut remap: Vec<Option<u32>> = Vec::with_capacity(regions.len());
    let mut region_sizes = Vec::new();
    for &size in &regions {
        if size >= config.min_region_size {
            remap.push(Some(region_sizes.len() as u32));
            region_sizes.push(size);
        } else {
            remap.push(None);
        }
    }
    for label in &mut labels {
        *label = label.and_then(|r| remap[r as usize]);
    }

    Segmentation {
        labels,
        region_count: region_sizes.len(),
        region_sizes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharp_features::fit::ring_fits;
    use glam::DVec3;

    /// Triangulated (nx x ny) vertex grid; `pos` maps grid coordinates to 3D.
    fn grid_mesh(
        nx: usize,
        ny: usize,
        pos: impl Fn(usize, usize) -> DVec3,
    ) -> (Vec<DVec3>, Vec<u32>) {
        let mut positions = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                positions.push(pos(i, j));
            }
        }
        let mut indices = Vec::new();
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let a = (j * nx + i) as u32;
                let b = a + 1;
                let c = a + nx as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        (positions, indices)
    }

    fn segment_grid(positions: &[DVec3], indices: &[u32]) -> Segmentation {
        let adjacency = MeshAdjacency::build(positions.len(), indices);
        let fits = ring_fits(positions, &adjacency, &[], 1.0, 1);
        segment_regions(&adjacency, &fits, &SegmentationConfig::default())
    }

    #[test]
    fn flat_grid_is_one_region() {
        let (positions, indices) = grid_mesh(15, 15, |i, j| DVec3::new(i as f64, j as f64, 0.0));
        let adjacency = MeshAdjacency::build(positions.len(), &indices);
        let fits = ring_fits(&positions, &adjacency, &[], 1.0, 1);
        let seg = segment_regions(&adjacency, &fits, &SegmentationConfig::default());
        assert_eq!(seg.region_count, 1);
        // Every vertex with a valid fit is claimed on a flat grid. (Two grid
        // corners have <4 ring members and no fit -- an open-boundary
        // artifact that closed surface-nets meshes don't have.)
        for v in 0..positions.len() {
            assert_eq!(
                seg.labels[v].is_none(),
                fits[v].is_none(),
                "vertex {v}: only fitless vertices may be unclaimed on a flat grid"
            );
        }
    }

    #[test]
    fn tent_splits_into_two_regions_with_unclaimed_crease() {
        // Two 90-degree planes meeting at the grid row i == 7: a "tent".
        // Left of the crease the surface is horizontal, right of it vertical.
        let crease = 7usize;
        let (positions, indices) = grid_mesh(15, 15, |i, j| {
            if i <= crease {
                DVec3::new(i as f64, j as f64, 0.0)
            } else {
                DVec3::new(crease as f64, j as f64, (i - crease) as f64)
            }
        });
        let seg = segment_grid(&positions, &indices);
        assert_eq!(seg.region_count, 2, "tent should split at the crease");

        // Crease-row vertices straddle both planes: high residual, unclaimed.
        for j in 2..13 {
            let v = j * 15 + crease;
            assert!(
                seg.labels[v].is_none(),
                "crease vertex ({crease},{j}) should be unclaimed"
            );
        }
        // Vertices well inside each half get distinct labels.
        let left = seg.labels[7 * 15 + 2].expect("left interior claimed");
        let right = seg.labels[7 * 15 + 12].expect("right interior claimed");
        assert_ne!(left, right);
    }
}
