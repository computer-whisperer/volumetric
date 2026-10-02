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
//!    (unclaimed vertex, or face-boundary vertex where grid alignment left
//!    no unclaimed band) gathers nearby claimed vertices, splits them into
//!    the sides that are connected without crossing a crease — face-pure *by
//!    construction* — fits one plane per side, and projects onto the
//!    intersection (edge line or corner point). The sampler then locates
//!    that point on the model's own surfaces, or rejects it.
//! 4. **Cleanup** ([`cleanup::clean_up_snaps`]): snaps that turn a
//!    triangle over are undone; cross-band vertex pairs that landed on the
//!    same feature point merge; and the triangles the snap flattened onto a
//!    feature are removed.
//! 5. **Feature edges** ([`feature_edges::FeatureEdges::classify`]): the
//!    edges of the welded mesh whose faces meet at more than the crease
//!    angle. Decimation keeps crease vertices on their crease with it.
//!
//! The mesh this returns is still one connected surface with one vertex per
//! position. The mesher decimates that, and only then
//! ([`normals::split_normals_at_features`]) duplicates the vertices on
//! feature edges so each side of a crease shades with its own normal.
//! Positions coincide, so the surface stays geometrically sealed.
//!
//! Robustness contract: every snap sits behind a chain of gates (side
//! support, side residual, intersection conditioning, movement clamp, every
//! side found in the model, no triangle turned over). Any gate failing leaves
//! the vertex where the mesher put it, so pathological geometry (fractals,
//! sub-cell features) degrades to the unsnapped mesh, never to an invalid
//! one. Watertightness is preserved: welding only merges vertices, triangles
//! are dropped only when they lose an edge to a merge or cancel in pairs,
//! and an edge flip replaces two triangles by two.
//!
//! Development history, oracle benchmarks, and research notes live in
//! `crates/meshing_lab`, whose benchmarks run against these exact modules.

pub mod adjacency;
pub mod cleanup;
pub mod feature_edges;
pub mod fit;
pub mod normals;
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
    pub feature_edges: feature_edges::FeatureEdgeConfig,
}

#[derive(Clone, Debug, Default)]
pub struct SharpFeatureStats {
    /// Smooth regions found by segmentation.
    pub regions: usize,
    /// Vertices considered for snapping: the feature zone (unclaimed), and
    /// claimed vertices beside another face.
    pub candidates: usize,
    /// Vertices on an edge line in the output (snapped and not retracted).
    pub snapped_edges: usize,
    /// Vertices on a corner point in the output (snapped and not retracted).
    pub snapped_corners: usize,
    /// Snaps undone in cleanup because they turned a triangle over.
    pub retracted_snaps: usize,
    pub welded_vertices: usize,
    /// Triangles the weld collapsed, plus cancelling pairs.
    pub dropped_triangles: usize,
    /// Feature edges found on the welded mesh.
    pub feature_edges: usize,
}

pub struct SharpFeatureOutput {
    pub positions: Vec<(f64, f64, f64)>,
    /// Accumulated (unnormalized) vertex normals carried through the weld.
    pub normals: Vec<(f64, f64, f64)>,
    pub indices: Vec<u32>,
    /// The edges creases run along, over `indices`' vertex numbering.
    pub feature_edges: feature_edges::FeatureEdges,
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
        segmentation::SmoothFaces {
            labels: &seg.labels,
            fits: &fits,
            config: &config.segmentation,
        },
        cell,
        &config.snap,
        Some(is_inside),
        cancel,
    );
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let cleaned = cleanup::clean_up_snaps(
        &positions_v,
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
    for v in 0..positions.len() {
        welded_normals[cleaned.remap[v] as usize] += normals_v[v];
    }

    let feature_edges = feature_edges::FeatureEdges::classify(
        &cleaned.positions,
        &cleaned.indices,
        &config.feature_edges,
    );
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    let retracted = |kind: snap::SnapKind| {
        cleaned
            .retracted
            .iter()
            .filter(|&&v| snapped.snapped[v as usize] == Some(kind))
            .count()
    };
    Some(SharpFeatureOutput {
        positions: crate::parallel_iter::map_range(0..cleaned.positions.len(), |i| {
            let p = cleaned.positions[i];
            (p.x, p.y, p.z)
        }),
        normals: crate::parallel_iter::map_range(0..welded_normals.len(), |i| {
            let n = welded_normals[i];
            (n.x, n.y, n.z)
        }),
        indices: cleaned.indices,
        stats: SharpFeatureStats {
            regions: seg.region_count,
            candidates: snapped.stats.candidates,
            snapped_edges: snapped.stats.snapped_edges - retracted(snap::SnapKind::Edge),
            snapped_corners: snapped.stats.snapped_corners - retracted(snap::SnapKind::Corner),
            retracted_snaps: cleaned.retracted.len(),
            welded_vertices: cleaned.welded_vertices,
            dropped_triangles: cleaned.dropped_triangles,
            feature_edges: feature_edges.len(),
        },
        feature_edges,
    })
}
