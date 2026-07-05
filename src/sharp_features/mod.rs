//! Region-based sharp feature reconstruction for binary-sampled meshes.
//!
//! Reconstructs sharp edges and corners on adaptive-surface-nets output by
//! inverting the failed per-vertex probing approach: smooth faces are
//! identified first, and features emerge as the boundaries between them.
//!
//! Pipeline over the refined stage-4 mesh:
//! 1. **Patch fits** ([`fit::ring_fits`]): a plane per vertex over its 1-ring
//!    of refined (on-surface) neighbors. Face-interior fits are sub-degree
//!    accurate; feature-straddling fits have order-of-magnitude larger
//!    residuals — the classification signal.
//! 2. **Segmentation** ([`segmentation::segment_regions`]): seeded growth
//!    into maximal smooth regions, gated by pairwise fitted-normal jumps
//!    (scale-invariant: an O(1) dihedral across a feature vs curvature that
//!    shrinks with cell size) and by fit residual. Feature-zone vertices stay
//!    unclaimed.
//! 3. **Snap** ([`snap::snap_feature_vertices`]): each unclaimed vertex
//!    gathers nearby claimed vertices per adjacent region — face-pure *by
//!    construction* — fits one plane per side, and projects onto the
//!    intersection (edge line or corner point).
//! 4. **Weld** ([`cleanup::weld_snapped_vertices`]): cross-band vertex pairs
//!    that landed on the same feature point merge, collapsing the folded
//!    slivers between them.
//!
//! Robustness contract: every snap sits behind a chain of gates (side
//! support, side residual, intersection conditioning, movement clamp, sampler
//! verification). Any gate failing leaves the vertex where the mesher put it,
//! so pathological geometry (fractals, sub-cell features) degrades to the
//! unsnapped mesh, never to an invalid one. Watertightness is preserved:
//! welding only merges vertices, and only triangles that lose an edge to a
//! merge are dropped.
//!
//! Development history, oracle benchmarks, and research notes live in
//! `crates/meshing_lab`, whose benchmarks run against these exact modules.

pub mod adjacency;
pub mod cleanup;
pub mod fit;
pub mod segmentation;
pub mod snap;

use glam::DVec3;

/// Configuration for sharp feature reconstruction. The defaults are the
/// oracle-benchmarked values from `meshing_lab`; they are expressed relative
/// to the finest cell size and need no per-model tuning.
#[derive(Clone, Debug, Default)]
pub struct SharpFeatureConfig {
    pub segmentation: segmentation::SegmentationConfig,
    pub snap: snap::SnapConfig,
    pub cleanup: cleanup::CleanupConfig,
}

#[derive(Clone, Debug, Default)]
pub struct SharpFeatureStats {
    /// Smooth regions found by segmentation.
    pub regions: usize,
    /// Feature-zone (unclaimed) vertices considered for snapping.
    pub candidates: usize,
    pub snapped_edges: usize,
    pub snapped_corners: usize,
    pub welded_vertices: usize,
    pub dropped_triangles: usize,
}

pub struct SharpFeatureOutput {
    pub positions: Vec<(f64, f64, f64)>,
    /// Accumulated (unnormalized) vertex normals carried through the weld.
    pub normals: Vec<(f64, f64, f64)>,
    pub indices: Vec<u32>,
    pub stats: SharpFeatureStats,
}

/// Run the full pipeline on a refined mesh.
///
/// `positions`/`normals` are the stage-4 refined vertex positions and their
/// accumulated outward normals; `cell` is the finest cell size; `is_inside`
/// is the model's binary occupancy function (used only for snap
/// verification).
pub fn apply_sharp_features(
    positions: &[(f64, f64, f64)],
    normals: &[(f64, f64, f64)],
    indices: &[u32],
    cell: f64,
    config: &SharpFeatureConfig,
    is_inside: &dyn Fn(f64, f64, f64) -> bool,
) -> SharpFeatureOutput {
    let positions_v: Vec<DVec3> = positions
        .iter()
        .map(|&(x, y, z)| DVec3::new(x, y, z))
        .collect();
    let normals_v: Vec<DVec3> = normals
        .iter()
        .map(|&(x, y, z)| DVec3::new(x, y, z))
        .collect();

    let adjacency = adjacency::MeshAdjacency::build(positions.len(), indices);
    let fits = fit::ring_fits(&positions_v, &adjacency, &normals_v, cell, 1);
    let seg = segmentation::segment_regions(&adjacency, &fits, &config.segmentation);

    let sampler = |p: DVec3| is_inside(p.x, p.y, p.z);
    let snapped = snap::snap_feature_vertices(
        &positions_v,
        &adjacency,
        &seg.labels,
        cell,
        &config.snap,
        Some(&sampler),
    );

    let cleaned = cleanup::weld_snapped_vertices(
        &snapped.positions,
        indices,
        &snapped.snapped,
        cell,
        &config.cleanup,
    );

    // Carry accumulated normals through the weld remap; cluster members agree
    // in orientation, so summing preserves the outward direction.
    let mut out_normals = vec![DVec3::ZERO; cleaned.positions.len()];
    for v in 0..positions.len() {
        out_normals[cleaned.remap[v] as usize] += normals_v[v];
    }

    SharpFeatureOutput {
        positions: cleaned.positions.iter().map(|p| (p.x, p.y, p.z)).collect(),
        normals: out_normals.iter().map(|n| (n.x, n.y, n.z)).collect(),
        indices: cleaned.indices,
        stats: SharpFeatureStats {
            regions: seg.region_count,
            candidates: snapped.stats.candidates,
            snapped_edges: snapped.stats.snapped_edges,
            snapped_corners: snapped.stats.snapped_corners,
            welded_vertices: cleaned.welded_vertices,
            dropped_triangles: cleaned.dropped_triangles,
        },
    }
}
