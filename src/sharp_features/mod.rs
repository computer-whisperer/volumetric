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
//! 3. **Snap** ([`snap::snap_feature_vertices`]): each feature candidate
//!    (unclaimed vertex, or region-boundary vertex where grid alignment left
//!    no unclaimed band) gathers nearby claimed vertices per adjacent region
//!    — face-pure *by construction* — fits one plane per side, and projects
//!    onto the intersection (edge line or corner point).
//! 4. **Weld** ([`cleanup::weld_snapped_vertices`]): cross-band vertex pairs
//!    that landed on the same feature point merge, collapsing the folded
//!    slivers between them.
//! 5. **Crease split** ([`crease::split_crease_vertices`]): snapped vertices
//!    are duplicated per adjacent region so each side of a feature shades
//!    with its own normal. Positions coincide — the surface stays
//!    geometrically sealed; only shading topology splits.
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
pub mod crease;
pub mod fit;
pub mod segmentation;
pub mod snap;

use std::sync::atomic::{AtomicBool, Ordering};

use glam::DVec3;

/// Marker trait for the binary occupancy sampler used by snap verification.
///
/// On native builds the sampler must be `Send + Sync`: snap evaluates
/// candidates in parallel, each probing the sampler independently (the
/// wasmtime-backed [`crate::wasm::ParallelModelSampler`] is built for exactly
/// this). On web builds iteration is sequential and only `Fn` is required —
/// the same split as [`crate::adaptive_surface_nets_2::SamplerFn`].
#[cfg(feature = "native")]
pub trait OccupancyFn: Fn(DVec3) -> bool + Send + Sync {}
#[cfg(feature = "native")]
impl<F> OccupancyFn for F where F: Fn(DVec3) -> bool + Send + Sync {}

#[cfg(not(feature = "native"))]
pub trait OccupancyFn: Fn(DVec3) -> bool {}
#[cfg(not(feature = "native"))]
impl<F> OccupancyFn for F where F: Fn(DVec3) -> bool {}

/// Configuration for sharp feature reconstruction. The defaults are the
/// oracle-benchmarked values from `meshing_lab`; they are expressed relative
/// to the finest cell size and need no per-model tuning.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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
    /// Extra vertex copies created for per-region crease shading.
    pub crease_splits: usize,
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
    is_inside: &dyn OccupancyFn,
) -> SharpFeatureOutput {
    static NEVER: AtomicBool = AtomicBool::new(false);
    apply_sharp_features_cancellable(positions, normals, indices, cell, config, is_inside, &NEVER)
        .expect("sharp features with a never-set cancel flag cannot be cancelled")
}

/// [`apply_sharp_features`] with cooperative cancellation: the flag is checked
/// at every pipeline-stage boundary and per candidate inside the (parallel)
/// snap stage. Returns `None` once the flag is observed set — the mesh under
/// construction is discarded, there is no partial result.
pub fn apply_sharp_features_cancellable(
    positions: &[(f64, f64, f64)],
    normals: &[(f64, f64, f64)],
    indices: &[u32],
    cell: f64,
    config: &SharpFeatureConfig,
    is_inside: &dyn OccupancyFn,
    cancel: &AtomicBool,
) -> Option<SharpFeatureOutput> {
    let positions_v: Vec<DVec3> = crate::parallel_iter::map_range(0..positions.len(), |i| {
        let (x, y, z) = positions[i];
        DVec3::new(x, y, z)
    });
    let normals_v: Vec<DVec3> = crate::parallel_iter::map_range(0..normals.len(), |i| {
        let (x, y, z) = normals[i];
        DVec3::new(x, y, z)
    });

    let adjacency = adjacency::MeshAdjacency::build(positions.len(), indices);
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let fits = fit::ring_fits(&positions_v, &adjacency, &normals_v, cell, 1);
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let seg = segmentation::segment_regions(&adjacency, &fits, &config.segmentation);
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let snapped = snap::snap_feature_vertices_cancellable(
        &positions_v,
        &adjacency,
        &seg.labels,
        cell,
        &config.snap,
        Some(is_inside),
        cancel,
    );
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let cleaned = cleanup::weld_snapped_vertices(
        &snapped.positions,
        indices,
        &snapped.snapped,
        cell,
        &config.cleanup,
    );
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    // Carry accumulated normals through the weld remap; cluster members agree
    // in orientation, so summing preserves the outward direction.
    let mut welded_normals = vec![DVec3::ZERO; cleaned.positions.len()];
    let mut welded_labels: Vec<Option<u32>> = vec![None; cleaned.positions.len()];
    let mut is_crease = vec![false; cleaned.positions.len()];
    for v in 0..positions.len() {
        let out = cleaned.remap[v] as usize;
        welded_normals[out] += normals_v[v];
        if let Some(label) = seg.labels[v] {
            welded_labels[out] = Some(label);
        }
        if snapped.snapped[v].is_some() {
            is_crease[out] = true;
        }
    }

    let split = crease::split_crease_vertices(
        &cleaned.positions,
        &welded_normals,
        &cleaned.indices,
        &welded_labels,
        &is_crease,
        cell,
    );
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    Some(SharpFeatureOutput {
        positions: crate::parallel_iter::map_range(0..split.positions.len(), |i| {
            let p = split.positions[i];
            (p.x, p.y, p.z)
        }),
        normals: crate::parallel_iter::map_range(0..split.normals.len(), |i| {
            let n = split.normals[i];
            (n.x, n.y, n.z)
        }),
        indices: split.indices,
        stats: SharpFeatureStats {
            regions: seg.region_count,
            candidates: snapped.stats.candidates,
            snapped_edges: snapped.stats.snapped_edges,
            snapped_corners: snapped.stats.snapped_corners,
            welded_vertices: cleaned.welded_vertices,
            dropped_triangles: cleaned.dropped_triangles,
            crease_splits: split.split_vertices,
        },
    })
}
