//! Preview construction shared by the GUI viewport and the CLI's `render`.
//!
//! A [`PreviewRequest`] names an output asset and the per-kind recipe
//! ([`PreviewPlan`]) to draw it with; [`build_preview_scene`] turns it into
//! a [`PreviewEntity`]: renderer geometry, world bounds and the statistics
//! the UI shows. 3D models are meshed (point cloud, marching cubes, or the
//! adaptive surface nets mesher through an [`ExecutionBackend`]) and
//! colormapped by a sample channel; 2D models raster to flat quads; FEA
//! meshes, point clouds and triangle meshes draw their explicit data;
//! Subspace values draw as gizmos sized by the whole scene
//! ([`submit_subspace_gizmo`]); ViewSet values draw their cameras as
//! frustums, and one view can be looked through ([`ViewFrame`]); Splat
//! values draw their Gaussians ([`splat_data`]).
//!
//! Everything here is window-free and GPU-free until submission, so the
//! same code builds the viewport's scene, the CLI's PNG, and the tests.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use glam::Vec3;
use volumetric::{AssetTypeHint, adaptive_surface_nets_2};
use volumetric_renderer as renderer;

mod gizmo;
mod scene;
mod splats;
mod views;

pub use gizmo::submit_subspace_gizmo;
pub use scene::{
    PendingMesh, PreviewStage, build_preview_scene, build_preview_scene_cancellable,
    build_preview_scene_monitored, build_preview_scene_with, field_channel_to_linear,
    format_error_chain, part_tint, preview_postlude, preview_prelude, srgb_to_linear,
    wireframe_style,
};
pub use splats::{splat_data, splat_detail};
pub use views::{
    FRUSTUM_DEPTH_M, Framed, LookThrough, MarkKind, MarkLabel, ViewFrame, clip_planes_for,
    frustum_segments, highlight_lines, mark_labels, observation_lines, pick_cross_px, pose_matrix,
    submit_view_highlight, viewset_detail,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PreviewRenderMode {
    Points,
    MarchingCubes,
    AdaptiveSurfaceNets2,
}

impl PreviewRenderMode {
    pub const ALL: [Self; 3] = [
        Self::Points,
        Self::MarchingCubes,
        Self::AdaptiveSurfaceNets2,
    ];

    pub fn route_name(self) -> &'static str {
        match self {
            Self::Points => "points",
            Self::MarchingCubes => "marching-cubes",
            Self::AdaptiveSurfaceNets2 => "asn2",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Points => "Points",
            Self::MarchingCubes => "MC",
            Self::AdaptiveSurfaceNets2 => "ASN2",
        }
    }

    pub fn full_label(self) -> &'static str {
        match self {
            Self::Points => "Point Cloud",
            Self::MarchingCubes => "Marching Cubes",
            Self::AdaptiveSurfaceNets2 => "Adaptive Surface Nets v2",
        }
    }

    pub fn from_route_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.route_name() == name)
    }
}

/// ASN2 meshing quality settings, adjustable per output. Angles and the
/// residual multiplier are stored as scaled integers so the settings stay
/// `Eq`/`Hash`-able for the preview cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Asn2Settings {
    /// Binary-search iterations for vertex position refinement (0-16).
    pub vertex_refinement_iterations: usize,
    /// Binary-search iterations for normal refinement probing (0 = face
    /// normals, 4-8 = smooth probed normals).
    pub normal_sample_iterations: usize,
    /// Reconstruct sharp edges and corners (region-based snapping).
    pub sharp_edges: bool,
    /// Sharp features: max same-region normal jump between adjacent
    /// vertices, in whole degrees (10-90).
    pub sharp_angle_degrees: u16,
    /// Constrain vertex refinement to each vertex's own grid edge. Prevents
    /// refinement from capturing a neighboring parallel surface — the fix
    /// for thin lattice sheets visually bonding together — at the cost of
    /// slightly more quantized vertex placement.
    pub edge_constrained_refinement: bool,
    /// Stage-5 quadric decimation: collapse edges whose removal stays
    /// within the tolerance budget, cutting the grid-pitch triangle counts
    /// of flat and gently curved regions while preserving topology.
    pub simplify: bool,
    /// Decimation error budget, in tenths of the finest cell size (1-30).
    pub simplify_tolerance_tenths: u16,
    /// Aperiodic interior probes per corner-uniform discovery cell (0
    /// disables). Catches geometry thinner than the coarse discovery grid's
    /// pitch — lattice struts sitting between corner samples — that the
    /// regular scan would silently lose.
    pub discovery_probes: usize,
    /// Stage-1 discovery grid override (0 = automatic 6/8 split). A denser
    /// base spends more up-front samples for more reliable discovery of
    /// busy geometry at the same finest resolution.
    pub base_resolution: usize,
}

impl Default for Asn2Settings {
    fn default() -> Self {
        Self {
            vertex_refinement_iterations: 8,
            normal_sample_iterations: 0,
            sharp_edges: true,
            sharp_angle_degrees: 15,
            edge_constrained_refinement: false,
            simplify: true,
            simplify_tolerance_tenths: 10,
            discovery_probes: 8,
            base_resolution: 0,
        }
    }
}

/// World-space `(min, max)` corners of an output's axis-aligned bounds.
pub type BoundsCorners = ((f32, f32, f32), (f32, f32, f32));

/// Meshing statistics for one built preview, reported by the host's worker.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OutputStats {
    pub mesh_ms: f64,
    pub triangles: usize,
    pub points: usize,
    /// Total sampler invocations, when the mesher reports them (ASN2).
    pub samples: u64,
    /// Preformatted per-stage profiling lines (ASN2 only).
    pub detail: Vec<String>,
    /// Colormappable fields of an FEA mesh output, qualified as
    /// `node:{name}` / `element:{name}` (node fields with 1 or 3 components,
    /// element fields with 1). Feeds the per-output field picker.
    pub fea_fields: Vec<String>,
    /// Declared sample channels of a 3D model output, in channel order
    /// (channel 0 is occupancy). Feeds the "Color by" picker and the slice
    /// lightbox's channel row; empty when the model declares no format.
    pub model_channels: Vec<String>,
    /// World-space `(min, max)` bounds of the built output. For model
    /// outputs this is the wasm-reported `get_bounds` domain, not the
    /// meshed geometry's tight box. Feeds the bounds overlay's dimension
    /// readouts.
    pub bounds: Option<BoundsCorners>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewMeshPlan {
    PointCloud {
        resolution: usize,
    },
    MarchingCubes {
        resolution: usize,
    },
    AdaptiveSurfaceNets2 {
        target_resolution: usize,
        base_resolution: usize,
        max_depth: usize,
        settings: Asn2Settings,
    },
}

impl PreviewMeshPlan {
    pub fn for_mode(mode: PreviewRenderMode, resolution: usize, asn2: Asn2Settings) -> Self {
        match mode {
            PreviewRenderMode::Points => Self::PointCloud { resolution },
            PreviewRenderMode::MarchingCubes => Self::MarchingCubes { resolution },
            PreviewRenderMode::AdaptiveSurfaceNets2 => {
                let (base_resolution, max_depth) =
                    asn2_resolution_split(resolution, asn2.base_resolution);
                Self::AdaptiveSurfaceNets2 {
                    target_resolution: resolution,
                    base_resolution,
                    max_depth,
                    settings: asn2,
                }
            }
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::PointCloud { resolution } => format!("Point cloud {resolution}^3"),
            Self::MarchingCubes { resolution } => format!("Marching cubes {resolution}^3"),
            Self::AdaptiveSurfaceNets2 {
                target_resolution,
                base_resolution,
                max_depth,
                ..
            } => format!("ASN2 {target_resolution}^3 ({base_resolution} x 2^{max_depth})"),
        }
    }

    pub fn adaptive_surface_nets_config(
        &self,
    ) -> Option<adaptive_surface_nets_2::AdaptiveMeshConfig2> {
        let Self::AdaptiveSurfaceNets2 {
            base_resolution,
            max_depth,
            settings,
            ..
        } = self
        else {
            return None;
        };

        Some(adaptive_surface_nets_2::AdaptiveMeshConfig2 {
            base_resolution: *base_resolution,
            max_depth: *max_depth,
            discovery_probes: settings.discovery_probes,
            vertex_refinement_iterations: settings.vertex_refinement_iterations,
            normal_sample_iterations: settings.normal_sample_iterations,
            normal_epsilon_frac: 0.1,
            num_threads: 0,
            edge_constrained_refinement: settings.edge_constrained_refinement,
            sharp_features: settings.sharp_edges.then(|| {
                let mut sharp = volumetric::sharp_features::SharpFeatureConfig::default();
                sharp.segmentation.max_normal_jump_deg = f64::from(settings.sharp_angle_degrees);
                sharp
            }),
            decimation: settings
                .simplify
                .then(|| volumetric::mesh_decimation::DecimationConfig {
                    error_tolerance_cells: f64::from(settings.simplify_tolerance_tenths) / 10.0,
                    ..Default::default()
                }),
        })
    }
}

/// Splits a target resolution into (base_resolution, max_depth). A non-zero
/// `base_override` pins the stage-1 discovery grid (clamped to the target);
/// otherwise the automatic 6/8 split applies.
pub fn asn2_resolution_split(target_resolution: usize, base_override: usize) -> (usize, usize) {
    let base_resolution = if base_override > 0 {
        base_override.min(target_resolution)
    } else if !target_resolution.is_power_of_two() && target_resolution.is_multiple_of(6) {
        6
    } else {
        8
    };
    let mut max_depth = 0;
    let mut effective_resolution = base_resolution;
    while effective_resolution < target_resolution {
        effective_resolution *= 2;
        max_depth += 1;
    }
    (base_resolution, max_depth)
}

#[derive(Clone, Debug)]
pub struct PreviewRequest {
    pub asset_id: String,
    /// Stable identity of `data`, independent of which process allocated it.
    /// Pointer identity is not sufficient for remote results or long-lived
    /// generated-mesh artifacts.
    pub source_hash: [u8; 32],
    /// The asset's raw bytes: a model WASM module, or CBOR mesh data for
    /// `AssetTypeHint::FeaMesh` outputs.
    pub data: Arc<Vec<u8>>,
    pub type_hint: Option<AssetTypeHint>,
    pub precursor_ids: Vec<String>,
    /// The per-kind build recipe (part of the preview cache key).
    pub plan: PreviewPlan,
    /// Overlay the mesh edges as lines (display-only; not part of the mesh
    /// cache key, so toggling never re-meshes).
    pub wireframe: bool,
    pub show_grid: bool,
    /// Overlay each output's wasm-reported bounding box (display-only,
    /// like `wireframe`).
    pub show_bounds: bool,
    pub ssao: bool,
    pub ssao_radius: f32,
    pub ssao_bias: f32,
    pub ssao_strength: f32,
    pub stale: bool,
}

/// The per-kind build recipe a [`PreviewRequest`] carries; part of the
/// preview cache key, so any change here rebuilds the preview.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewPlan {
    Model3d {
        mesh: PreviewMeshPlan,
        /// Sample channel to colormap points/vertices by, when declared.
        color_channel: Option<String>,
        /// Give the model a muted distinct tint (keyed by output id) when
        /// it carries no surface colors of its own — distinguishes
        /// flush-fitting parts pinned into one viewport. Part of the
        /// cache key, like `color_channel`.
        tint_uncolored: bool,
    },
    Sketch {
        resolution: usize,
        color_channel: Option<String>,
    },
    FeaMesh {
        deformed: bool,
        exaggeration_tenths: u16,
        color_field: Option<String>,
        /// The values the colormap spans; `None` spans the field's own
        /// range. Values beyond it take the end colours.
        color_range: Option<ColorRange>,
    },
    TriMesh,
    Subspace,
    /// Every view's frustum and every marker's square.
    ViewSet,
    /// The Gaussians themselves, blended back to front.
    Splat,
}

/// A fixed colormap span in millionths of the field's unit (micrometres
/// for a distance in metres), integers so a plan stays `Eq` for the
/// preview cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorRange {
    pub lo_micro: i64,
    pub hi_micro: i64,
}

impl ColorRange {
    /// From the field's units; `lo` must be below `hi`.
    pub fn new(lo: f64, hi: f64) -> Option<Self> {
        (lo.is_finite() && hi.is_finite() && lo < hi).then(|| Self {
            lo_micro: (lo * 1e6).round() as i64,
            hi_micro: (hi * 1e6).round() as i64,
        })
    }

    pub fn lo(self) -> f64 {
        self.lo_micro as f64 * 1e-6
    }

    pub fn hi(self) -> f64 {
        self.hi_micro as f64 * 1e-6
    }
}

impl PreviewPlan {
    pub fn label(&self) -> String {
        match self {
            Self::Model3d {
                mesh,
                color_channel,
                ..
            } => match color_channel {
                Some(channel) => format!("{} · {channel}", mesh.label()),
                None => mesh.label(),
            },
            Self::Sketch {
                resolution,
                color_channel,
            } => match color_channel {
                Some(channel) => format!("2D raster {resolution} · {channel}"),
                None => format!("2D raster {resolution}"),
            },
            Self::FeaMesh { color_field, .. } => match color_field {
                Some(field) => format!("FEA mesh · {field}"),
                None => "FEA mesh".to_string(),
            },
            Self::TriMesh => "Triangle mesh".to_string(),
            Self::Subspace => "Subspace".to_string(),
            Self::ViewSet => "Views".to_string(),
            Self::Splat => "Splat".to_string(),
        }
    }
}

/// Compact count formatting: 950, 12.4k, 3.1M.
pub fn format_count(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// A single built output: its meshed geometry, world-space bounds, and the
/// meshing statistics surfaced in the UI.
///
/// Built by [`build_preview_scene`] for the viewport and for the CLI's
/// `render`, which draw it through the same renderer.
#[derive(Clone)]
pub struct PreviewEntity {
    pub scene: renderer::SceneData,
    pub bounds: PreviewBounds,
    pub stats: OutputStats,
    /// Unique mesh edges, prebuilt so the wireframe toggle is display-only
    /// (composited in per frame when requested; `None` for point clouds and
    /// sketch previews).
    pub wireframe_lines: Option<renderer::LineData>,
    /// The decoded subspace value, for entities that are subspace gizmos.
    /// Gizmo geometry is generated at submit time so its extent can follow
    /// the whole scene's bounds, not this entity's own.
    pub subspace: Option<volumetric::subspace::Subspace>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreviewBounds {
    pub min: (f32, f32, f32),
    pub max: (f32, f32, f32),
}

impl PreviewBounds {
    pub fn min_vec3(self) -> Vec3 {
        Vec3::new(self.min.0, self.min.1, self.min.2)
    }

    pub fn max_vec3(self) -> Vec3 {
        Vec3::new(self.max.0, self.max.1, self.max.2)
    }

    /// The bounding box enclosing both `self` and `other`.
    pub fn union(self, other: PreviewBounds) -> PreviewBounds {
        PreviewBounds {
            min: (
                self.min.0.min(other.min.0),
                self.min.1.min(other.min.1),
                self.min.2.min(other.min.2),
            ),
            max: (
                self.max.0.max(other.max.0),
                self.max.1.max(other.max.1),
                self.max.2.max(other.max.2),
            ),
        }
    }
}

/// Where the heavy compute inside background jobs executes: in this process
/// ([`LocalBackend`]) or forwarded to a build daemon
/// (the UI's remote backend). Only the work that scales with model
/// size goes through here — project runs and ASN2 meshing. Lightbox
/// sampling, marching-cubes/point-cloud previews, scene assembly, and
/// bundled-operator metadata always run locally regardless of backend.
pub trait ExecutionBackend: Send + Sync {
    /// `progress` receives [`volumetric::BuildProgress`] snapshots while the
    /// run executes (per timeline step locally; at the polling cadence
    /// remotely). Called from the executing thread — keep it cheap.
    fn run_project(
        &self,
        project: &volumetric::Project,
        cancel: &AtomicBool,
        progress: &dyn Fn(volumetric::BuildProgress),
        artifact_ready: &dyn Fn(&volumetric::LoadedAsset),
    ) -> Result<Vec<volumetric::LoadedAsset>, String>;

    /// `Ok(None)` means the cancel flag was observed and the mesh abandoned.
    fn mesh_model(
        &self,
        model_wasm: &[u8],
        config: &volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2,
        cancel: &AtomicBool,
        progress: &dyn Fn(volumetric::BuildProgress),
    ) -> Result<Option<Arc<volumetric::AdaptiveMeshV2Result>>, String>;
}

/// Executes in this process, on whatever thread calls it.
pub struct LocalBackend;

impl ExecutionBackend for LocalBackend {
    fn run_project(
        &self,
        project: &volumetric::Project,
        cancel: &AtomicBool,
        progress: &dyn Fn(volumetric::BuildProgress),
        artifact_ready: &dyn Fn(&volumetric::LoadedAsset),
    ) -> Result<Vec<volumetric::LoadedAsset>, String> {
        project
            .run_monitored_with_artifacts(
                &mut volumetric::Environment::new(),
                cancel,
                progress,
                artifact_ready,
            )
            .map_err(|err| err.to_string())
    }

    fn mesh_model(
        &self,
        model_wasm: &[u8],
        config: &volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2,
        cancel: &AtomicBool,
        progress: &dyn Fn(volumetric::BuildProgress),
    ) -> Result<Option<Arc<volumetric::AdaptiveMeshV2Result>>, String> {
        let key = volumetric::MeshCacheKey::new(model_wasm, config);
        let artifact = volumetric::mesh_cache::global()
            .get_or_build(key, cancel, || {
                volumetric::generate_adaptive_mesh_v2_from_bytes_monitored(
                    model_wasm, config, cancel, progress,
                )
            })
            .map_err(format_error_chain)?;
        if let Some(artifact) = &artifact {
            let phase = match artifact.source {
                volumetric::mesh_cache::MeshCacheSource::Hit => Some("mesh artifact cache hit"),
                volumetric::mesh_cache::MeshCacheSource::Shared => {
                    Some("shared identical in-flight mesh artifact")
                }
                volumetric::mesh_cache::MeshCacheSource::Built => None,
            };
            if let Some(phase) = phase {
                progress(volumetric::BuildProgress {
                    phase: phase.to_string(),
                    fraction: Some(1.0),
                });
            }
        }
        Ok(artifact.map(|artifact| artifact.mesh))
    }
}
