//! Damascene-based v2 UI shell for Volumetric.
//!
//! This crate intentionally starts as a separate app path so the current egui UI
//! remains the usable baseline while the Damascene port grows toward parity.

use std::sync::{Arc, LazyLock};

use volumetric::operator_config::{ConfigField, ConfigFieldType, ConfigValue};
use volumetric::{
    AssetTypeHint, ExecutionInput, LoadedAsset, OperatorMetadata, OperatorMetadataInput, Project,
    adaptive_surface_nets_2, operator_config,
};
use volumetric_renderer::CameraControlScheme;

use damascene_core::SvgIcon;
use damascene_core::prelude::*;
// The scene mesh lives in damascene's pinned glam (a different major than
// this crate's own `glam` dep) — build its types through `scene::glam`.
use damascene_core::scene::{
    GridPlanes, Material, MeshData as SceneMeshData, MeshHandle as SceneMeshHandle,
    MeshVertex as SceneMeshVertex, SceneSpec, glam::Vec3 as SceneVec3,
};

/// App-supplied glyphs for names missing from damascene's built-in icon
/// vocabulary (`damascene_core::all_icon_names()`). Lucide path data in the
/// same 24×24 stroke style as the built-ins; `parse_current_color` makes
/// them tint like any other icon.
static EYE_ICON: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/></svg>"##,
    )
    .expect("eye icon SVG parses")
});
static PIN_ICON: LazyLock<SvgIcon> = LazyLock::new(|| {
    SvgIcon::parse_current_color(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1z"/></svg>"##,
    )
    .expect("pin icon SVG parses")
});

/// Category glyphs for Add-catalog entries — lucide path data like the
/// icons above. A module's own declared `icon_svg` (parsed once, cached
/// on its [`catalog::CatalogEntry`]) takes precedence; these cover
/// entries that declare none and unscanned entries (kind fallbacks:
/// `activity` / `settings`).
macro_rules! category_icon {
    ($name:ident, $body:literal) => {
        static $name: LazyLock<SvgIcon> = LazyLock::new(|| {
            SvgIcon::parse_current_color(concat!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">"##,
                $body,
                "</svg>"
            ))
            .expect("category icon SVG parses")
        });
    };
}
category_icon!(
    PRIMITIVES_ICON,
    r##"<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>"##
);
category_icon!(
    COMBINE_ICON,
    r##"<circle cx="9" cy="12" r="6"/><circle cx="15" cy="12" r="6"/>"##
);
category_icon!(
    TRANSFORMS_ICON,
    r##"<path d="M5 3v16h16"/><path d="m5 19 6-6"/><path d="m2 6 3-3 3 3"/><path d="m18 16 3 3-3 3"/>"##
);
category_icon!(
    CONSTRUCTION_ICON,
    r##"<circle cx="19" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><path d="M5 17A12 12 0 0 1 17 5"/>"##
);
category_icon!(
    LATTICE_ICON,
    r##"<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/>"##
);
category_icon!(
    FEA_ICON,
    r##"<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>"##
);
category_icon!(
    MESH_ICON,
    r##"<path d="M13.73 4a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"/>"##
);
category_icon!(
    FABRICATION_ICON,
    r##"<path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><path d="M6 9V3a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v6"/><rect x="6" y="14" width="12" height="8" rx="1"/>"##
);
category_icon!(
    SCRIPTING_ICON,
    r##"<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>"##
);

/// The icon slot source for a catalog entry: the module's own declared
/// icon when it parsed, else its category's glyph, with per-kind
/// fallbacks while the entry is unscanned or uncategorized.
fn catalog_icon_source(entry: &catalog::CatalogEntry) -> IconSource {
    if let Some(icon) = &entry.icon {
        return IconSource::Custom(icon.clone());
    }
    let category = entry
        .ready()
        .map(|metadata| metadata.category.as_str())
        .unwrap_or("");
    let custom = |icon: &LazyLock<SvgIcon>| IconSource::Custom((**icon).clone());
    match category {
        "Primitives" => custom(&PRIMITIVES_ICON),
        "Combine" => custom(&COMBINE_ICON),
        "Transforms" => custom(&TRANSFORMS_ICON),
        "Construction" => custom(&CONSTRUCTION_ICON),
        "Lattice" => custom(&LATTICE_ICON),
        "FEA" => custom(&FEA_ICON),
        "Mesh" => custom(&MESH_ICON),
        "Fabrication" => custom(&FABRICATION_ICON),
        "Scripting" => custom(&SCRIPTING_ICON),
        "Import" => "file-text".into_icon_source(),
        _ => match entry.kind {
            volumetric_assets::AssetCategory::Model => "activity".into_icon_source(),
            volumetric_assets::AssetCategory::Operator => "settings".into_icon_source(),
        },
    }
}

pub mod catalog;
#[cfg(not(target_arch = "wasm32"))]
pub mod host;
// The daemon-outcome mapping is shared with the web shell; the blocking
// RemoteBackend inside is native-only.
pub mod remote;
pub mod session;
#[cfg(not(target_arch = "wasm32"))]
pub mod settings;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod web_host;

pub const VIEWPORT_KEY: &str = "viewport";
pub const NEW_PROJECT_KEY: &str = "action:new-project";
pub const OPEN_PROJECT_KEY: &str = "action:open-project";
pub const SAVE_PROJECT_KEY: &str = "action:save-project";
pub const SAVE_PROJECT_AS_KEY: &str = "action:save-project-as";
pub const SAVE_BUILT_COPY_KEY: &str = "action:save-built-copy";
pub const IMPORT_WASM_KEY: &str = "action:import-wasm";
pub const IMPORT_STL_KEY: &str = "action:import-stl";
pub const IMPORT_IMAGE_KEY: &str = "action:import-image";
pub const RUN_PROJECT_KEY: &str = "action:run-project";
pub const CANCEL_RUN_KEY: &str = "action:cancel-run";
pub const TOGGLE_AUTO_REBUILD_KEY: &str = "action:toggle-auto-rebuild";
pub const TOGGLE_REMOTE_BUILD_KEY: &str = "action:toggle-remote-build";
pub const CANCEL_MESH_KEY: &str = "action:cancel-mesh";
pub const REMESH_KEY: &str = "action:remesh";
pub const TOGGLE_AUTO_REMESH_KEY: &str = "action:toggle-auto-remesh";
pub const TOGGLE_GRID_KEY: &str = "viewport:toggle-grid";
pub const TOGGLE_BOUNDS_KEY: &str = "viewport:toggle-bounds";
pub const TOGGLE_SSAO_KEY: &str = "viewport:toggle-ssao";
pub const FRAME_PREVIEW_KEY: &str = "viewport:frame-preview";
pub const RESET_CAMERA_KEY: &str = "viewport:reset-camera";

/// Top application menubar; the only menu value is `file`.
const MENUBAR_KEY: &str = "main-menu";
/// One-click add of a cataloged module from the Add modal.
const ADD_MODEL_PREFIX: &str = "add:model:";
const ADD_OPERATOR_PREFIX: &str = "add:operator:";
/// The Add-catalog modal: the rail tile that opens it, its dismiss scrim,
/// and its controlled search input.
const ADD_OPEN_KEY: &str = "add:open";
const ADD_DISMISS_KEY: &str = "add:dismiss";
const ADD_SEARCH_KEY: &str = "add-search-input";
/// Recents-rail tile for a cataloged module; the catalog kind decides
/// whether it adds as a model import or an operator step.
const RAIL_ADD_PREFIX: &str = "rail:add:";
/// Pipeline accordion in the project panel; values `imports|steps|exports`.
const PIPELINE_KEY: &str = "pipeline";
/// Viewport overlay value pickers (controlled select widgets).
const MODE_SELECT_KEY: &str = "view:mode";
const RESOLUTION_SELECT_KEY: &str = "view:res";
const CAMERA_SELECT_KEY: &str = "view:camera";
/// SSAO parameter popover trigger; steppers use `view:ssao-adj:{field}:{dir}`.
const SSAO_SETTINGS_KEY: &str = "view:ssao";
const SSAO_ADJUST_PREFIX: &str = "view:ssao-adj:";
/// Remote-build settings popover trigger and the daemon address input in it.
const REMOTE_SETTINGS_KEY: &str = "view:remote-settings";

/// Build-cache stats popover trigger; budget steppers use
/// `view:cache-adj:{up|down}`.
const CACHE_SETTINGS_KEY: &str = "view:cache-settings";
const CACHE_ADJUST_PREFIX: &str = "view:cache-adj:";
const REMOTE_ADDRESS_KEY: &str = "remote-address-input";
/// Per-output render settings: `output:settings:{id}` opens the popover
/// (plus `:dismiss`); mode/res routes end `:{value}` after the asset id.
const OUTPUT_SETTINGS_PREFIX: &str = "output:settings:";
const OUTPUT_MODE_PREFIX: &str = "output:mode:";
const OUTPUT_RESOLUTION_PREFIX: &str = "output:res:";
const OUTPUT_DEFAULTS_PREFIX: &str = "output:defaults:";
/// ASN2 setting stepper: `output:asn2:{id}:{field}:{up|down}`.
const OUTPUT_ASN2_PREFIX: &str = "output:asn2:";
const OUTPUT_WIREFRAME_PREFIX: &str = "output:wire:";
/// FEA view controls: deformed toggle (`output:fea-deformed:{id}`),
/// exaggeration stepper (`output:fea-exag:{id}:{up|down}`), colormap field
/// (`output:fea-field:{id}:{node|element}:{name}` or `output:fea-field:{id}:none`).
const OUTPUT_FEA_DEFORMED_PREFIX: &str = "output:fea-deformed:";
const OUTPUT_FEA_EXAG_PREFIX: &str = "output:fea-exag:";
const OUTPUT_FEA_FIELD_PREFIX: &str = "output:fea-field:";
/// Colormap channel picker for 3D model outputs; value `{id}:none` or
/// `{id}:{channel}`.
const OUTPUT_CHANNEL_PREFIX: &str = "output:channel:";
/// Slice controls inside a 3D model's lightbox; values `axis:{0|1|2}`,
/// `pos:{up|down}`, `channel:{name}`.
const LIGHTBOX_SLICE_PREFIX: &str = "lightbox:slice:";
/// 2D field inspection lightbox: `output:inspect:{id}` opens it for an
/// output; the modal scrim/close emit `lightbox:dismiss`.
const OUTPUT_INSPECT_PREFIX: &str = "output:inspect:";
const LIGHTBOX_KEY: &str = "lightbox";
/// Mesh export modal: `output:export-mesh:{id}` opens it for an output.
/// Inside, the scrim and Cancel emit `export:dismiss`, the primary button
/// `export:confirm`, unit presets `export:unit:{mm|cm|m|in}`, and the scale
/// factor is a controlled text input at `export:scale`.
const EXPORT_MESH_PREFIX: &str = "output:export-mesh:";
const EXPORT_DISMISS_KEY: &str = "export:dismiss";
const EXPORT_CONFIRM_KEY: &str = "export:confirm";
const EXPORT_SCALE_KEY: &str = "export:scale";
const EXPORT_UNIT_PREFIX: &str = "export:unit:";
const EXPORT_SCENE_KEY: &str = "export:scene";
const EXPORT_WASM_PREFIX: &str = "output:wasm:";
/// Draggable divider between the viewport and the project panel.
const PANEL_RESIZE_KEY: &str = "panel:resize";
const PANEL_WIDTH_DEFAULT: f32 = 320.0;
const PANEL_WIDTH_MIN: f32 = 240.0;
const PANEL_WIDTH_MAX: f32 = 560.0;
const SELECT_IMPORT_PREFIX: &str = "project:select-import:";
/// Replace an imported operator's bytes with the matching bundled build.
const UPGRADE_IMPORT_PREFIX: &str = "project:upgrade-import:";
const DELETE_IMPORT_PREFIX: &str = "project:delete-import:";
const SELECT_STEP_PREFIX: &str = "project:select-step:";
const DELETE_STEP_PREFIX: &str = "project:delete-step:";
const MOVE_STEP_UP_PREFIX: &str = "project:move-step-up:";
const MOVE_STEP_DOWN_PREFIX: &str = "project:move-step-down:";
const SET_STEP_MODEL_PREFIX: &str = "project:set-step-model:";
const SELECT_EXPORT_PREFIX: &str = "project:select-export:";
const DELETE_EXPORT_PREFIX: &str = "project:delete-export:";
const ADD_EXPORT_PREFIX: &str = "project:add-export:";
const SELECT_RUNTIME_ASSET_PREFIX: &str = "runtime:select-asset:";
const TOGGLE_PIN_PREFIX: &str = "runtime:toggle-pin:";
/// Text buffer for the selected step's output name; committed by the Rename
/// button (`project:rename-output:{step_idx}`), not per keystroke, so
/// half-typed names never leak into exports/references.
const OUTPUT_NAME_KEY: &str = "step-output-name";
const RENAME_OUTPUT_PREFIX: &str = "project:rename-output:";
const RESET_LUA_PREFIX: &str = "project:reset-lua:";
const CONFIG_FIELD_PREFIX: &str = "cfg:";
/// Text inputs for a `VecF64` operator input; value `{input_idx}:{component}`.
const VEC_INPUT_PREFIX: &str = "vec:";
const CONFIG_BOOL_PREFIX: &str = "cfg-bool:";
const CONFIG_ENUM_PREFIX: &str = "cfg-enum:";
const LUA_SOURCE_KEY: &str = "lua-source";
const PREVIEW_RESOLUTIONS: [usize; 13] =
    [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
/// Raster resolutions offered for 2D field outputs (cells along the longer
/// bounds axis).
const SKETCH_RESOLUTIONS: [usize; 5] = [64, 128, 256, 512, 1024];
const SKETCH_RESOLUTION_DEFAULT: usize = 256;
/// FEA deformation exaggeration ladder, in tenths (0.1x .. 100x).
const FEA_EXAGGERATION_TENTHS: [u16; 10] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProjectSelection {
    Import(usize),
    Step(usize),
    Export(usize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PreviewRenderMode {
    Points,
    MarchingCubes,
    AdaptiveSurfaceNets2,
}

impl PreviewRenderMode {
    const ALL: [Self; 3] = [
        Self::Points,
        Self::MarchingCubes,
        Self::AdaptiveSurfaceNets2,
    ];

    fn route_name(self) -> &'static str {
        match self {
            Self::Points => "points",
            Self::MarchingCubes => "marching-cubes",
            Self::AdaptiveSurfaceNets2 => "asn2",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Points => "Points",
            Self::MarchingCubes => "MC",
            Self::AdaptiveSurfaceNets2 => "ASN2",
        }
    }

    fn full_label(self) -> &'static str {
        match self {
            Self::Points => "Point Cloud",
            Self::MarchingCubes => "Marching Cubes",
            Self::AdaptiveSurfaceNets2 => "Adaptive Surface Nets v2",
        }
    }

    fn from_route_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.route_name() == name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeAssetSummary {
    pub id: String,
    pub bytes: usize,
    pub type_hint: Option<AssetTypeHint>,
    pub precursor_count: usize,
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

/// Everything the 2D inspection lightbox displays for one output, computed
/// by a background job (sampling the model is too slow for the UI thread).
/// `analytics` is an open-ended list of label/value rows — the place new
/// engineering statistics get added over time.
#[derive(Clone, Debug, PartialEq)]
pub struct LightboxData {
    /// RGBA8 pixels of the colormapped raster, row 0 at the top (image
    /// convention, ready for texture upload).
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bounds_min: (f32, f32),
    pub bounds_max: (f32, f32),
    /// Occupancy mask (no meaningful value range or colorbar).
    pub binary: bool,
    pub value_min: f32,
    pub value_max: f32,
    /// Label/value analytics rows, in display order.
    pub analytics: Vec<(String, String)>,
}

/// What the lightbox samples: a 2D model's field directly, or an axis-
/// aligned slice through a 3D model's declared sample channel. Part of the
/// background job's identity — results for a stale mode are dropped.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LightboxMode {
    /// 2D model: rasterize the field over its bounds.
    #[default]
    Sketch,
    /// 3D model: sample `channel` on the plane `position[axis] = min +
    /// (frac_percent/100)·extent`, occupancy-masked.
    Slice {
        axis: usize,
        frac_percent: u16,
        channel: String,
    },
}

/// The open lightbox: which output, and the pipeline of things that arrive
/// asynchronously after it opens (sampled data from the background job,
/// then the GPU textures uploaded by the session).
#[derive(Debug, Default)]
pub struct LightboxState {
    pub asset_id: String,
    pub mode: LightboxMode,
    pub data: Option<LightboxData>,
    pub texture: Option<AppTexture>,
    pub colorbar: Option<AppTexture>,
}

/// The open Add-catalog modal. An empty query browses the catalog grouped
/// by declared category; typing switches to ranked search rows.
#[derive(Debug, Default)]
pub struct AddModalState {
    /// Controlled search buffer.
    pub query: String,
}

/// How many recently added modules the rail remembers.
const RECENT_ADDS_CAP: usize = 8;

/// The recents rail's fresh-install seed: a starter kit spanning the
/// common flows (a solid, the boolean, a transform, a generator), so the
/// rail is useful before the user has any history.
fn default_recent_adds() -> Vec<String> {
    [
        "simple_sphere_model",
        "rectangular_prism_operator",
        "boolean_operator",
        "translate_operator",
        "scale_operator",
    ]
    .map(str::to_string)
    .to_vec()
}

/// The open mesh-export modal: which output, the preview geometry as a
/// damascene scene handle (delivered by the session from the preview cache
/// the frame after opening), and the export configuration being edited.
#[derive(Debug)]
pub struct ExportDialogState {
    pub asset_id: String,
    pub mesh: ExportPreviewMesh,
    /// Controlled scale-factor text buffer; parsed (finite, positive) at
    /// build time for the size readout and the Export button's enabled
    /// state, and again at confirm.
    pub scale_text: String,
}

/// The modal's scene-ready copy of the output's cached preview mesh.
#[derive(Debug, Default)]
pub enum ExportPreviewMesh {
    /// Waiting for the session to copy the cached preview (one frame).
    #[default]
    Pending,
    /// No cached mesh preview — the output was meshed as points, or the
    /// project hasn't been run.
    Missing,
    Ready {
        /// Geometry for the modal's `chart3d` preview.
        handle: SceneMeshHandle,
        triangles: usize,
        /// Tight `(min, max)` bounds of the unscaled triangles.
        bounds: ((f32, f32, f32), (f32, f32, f32)),
    },
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
    fn for_mode(mode: PreviewRenderMode, resolution: usize, asn2: Asn2Settings) -> Self {
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

/// Where the shell's background worker executes heavy jobs. Produced by the
/// remote-build toggle, consumed by the host, which rebuilds its worker
/// around the matching [`session::ExecutionBackend`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorChoice {
    Local,
    Remote(String),
}

#[derive(Clone, Debug)]
pub struct PreviewRequest {
    pub asset_id: String,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PreviewBuildStatus {
    #[default]
    Idle,
    Building {
        label: String,
    },
    Ready {
        label: String,
    },
    /// Requested previews are out of date but were not dispatched: either
    /// auto-remesh is off, or their builds were explicitly cancelled. An
    /// explicit Remesh dispatches them.
    Stale {
        label: String,
    },
    Failed {
        label: String,
        error: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewportCameraCommand {
    FramePreview,
    /// Return the camera to its default pose.
    Reset,
}

/// Lifecycle of the (host-driven) asynchronous project run.
///
/// The app never executes the project itself; it requests a run and the host's
/// background worker reports back through [`VolumetricUiV2::apply_run_result`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RunState {
    #[default]
    Idle,
    Running,
}

impl PreviewBuildStatus {
    fn label(&self) -> String {
        match self {
            Self::Idle => "preview idle".to_string(),
            Self::Building { label } => format!("building {label}"),
            Self::Ready { label } => format!("ready {label}"),
            Self::Stale { label } => format!("stale {label}"),
            Self::Failed { label, .. } => format!("failed {label}"),
        }
    }

    fn tooltip(&self) -> Option<&str> {
        match self {
            Self::Failed { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectSummary {
    pub imports: usize,
    pub timeline_steps: usize,
    pub exports: usize,
    pub selected_export: Option<String>,
    pub selected_project_item: Option<ProjectSelection>,
    pub pinned_outputs: Vec<String>,
    pub render_mode: PreviewRenderMode,
    pub preview_resolution: usize,
    pub camera_control_scheme: CameraControlScheme,
    pub show_grid: bool,
    pub show_bounds: bool,
    pub ssao: bool,
    pub runtime_assets: Vec<RuntimeAssetSummary>,
    pub last_run_elapsed_ms: Option<u128>,
    pub last_run_error: Option<String>,
    pub last_run_stale: bool,
    pub run_state: RunState,
    pub auto_rebuild: bool,
}

/// Editing state for the selected operator step's config input: the parsed
/// schema plus a raw text buffer per field. The buffer is the source of truth
/// while typing; parseable values are committed into the step's CBOR blob.
/// What kind of asset an operator input slot accepts (drives which assets
/// its picker offers).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssetSlotKind {
    Model,
    FeaMesh,
    TriMesh,
    Subspace,
}

#[derive(Clone, Debug)]
struct AssetSlot {
    input_idx: usize,
    kind: AssetSlotKind,
    /// The operator-declared label for this slot, when its metadata names it.
    name: Option<String>,
}

/// Edit buffers for one `VecF64` input: one text field per component,
/// committed component-wise into the step's inline little-endian f64 bytes.
#[derive(Debug)]
struct VecForm {
    input_idx: usize,
    /// The operator-declared label for this input, when metadata names it.
    name: Option<String>,
    buffers: Vec<String>,
}

#[derive(Debug)]
struct StepEditState {
    step_idx: usize,
    /// Input indices that reference assets (`ModelWASM` / `FeaMesh` slots),
    /// in declaration order.
    asset_slots: Vec<AssetSlot>,
    /// One editor per declared `VecF64` input, in declaration order.
    vecs: Vec<VecForm>,
    /// The config form, present only when the operator declares a config input.
    config: Option<ConfigForm>,
    /// The Lua source editor, present only for a `LuaSource` input.
    lua: Option<LuaForm>,
    /// Edit buffer for the step's (first) output name; committed via Rename.
    output_name: String,
}

#[derive(Debug)]
struct ConfigForm {
    input_idx: usize,
    fields: Vec<ConfigField>,
    buffers: std::collections::BTreeMap<String, String>,
}

/// Editor state for a `LuaSource` input. The `source` buffer is the edit
/// source of truth; every change is written straight back to the step (no
/// parse gate — any text is a valid script).
#[derive(Debug)]
struct LuaForm {
    input_idx: usize,
    source: String,
}

/// A file operation the app has requested. The host owns the native file
/// dialogs (and, for STL, the cached preview meshes), so the app queues the
/// intent and the host drains it via `take_file_action`.
#[derive(Clone, Debug, PartialEq)]
pub enum FileAction {
    OpenProject,
    /// Save with a path dialog (first save / Save As).
    SaveProject,
    /// Re-save in place to the known project path (no dialog).
    SaveProjectTo(std::path::PathBuf),
    /// Save a copy with the built step results embedded (path dialog; see
    /// `volumetric::baked`).
    SaveBuiltCopy,
    /// Export the cached preview mesh of the named output as binary STL,
    /// with vertices multiplied by `scale` (the export modal's setting).
    ExportMesh {
        id: String,
        scale: f32,
    },
    /// Export the named output's model WASM bytes verbatim.
    ExportWasm(String),
    /// Import a model WASM file into the project.
    ImportWasm,
    /// Import an STL mesh via the bundled `stl_import_operator`.
    ImportStl,
    /// Import an image as a 2D field model via the bundled
    /// `image_model_operator`.
    ImportImage,
}

/// What an output *is*, for view purposes: each kind gets its own render
/// settings and its own controls in the per-output settings popover.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputKind {
    /// A 3D model WASM: meshed by one of the volume mesh plans.
    Model3d,
    /// A 2D model WASM (sketch or scalar field): flat raster preview.
    Model2d,
    /// An FEA mesh value: boundary faces with FEA-specific view modes.
    FeaMesh,
    /// A triangle mesh value: drawn as-is.
    TriMesh,
    /// An affine subspace value: point/line/plane gizmo.
    Subspace,
}

/// FEA-specific view settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeaRender {
    /// Apply the `displacement` node field to positions (when present).
    pub deformed: bool,
    /// Displacement exaggeration in tenths (10 = true scale); integer so
    /// the settings stay `Eq` for the preview cache key.
    pub exaggeration_tenths: u16,
    /// Colormapped field, qualified as `node:{name}` / `element:{name}`;
    /// `None` draws plain shaded faces.
    pub color_field: Option<String>,
    /// Element-edge overlay.
    pub wireframe: bool,
}

impl Default for FeaRender {
    fn default() -> Self {
        Self {
            deformed: true,
            exaggeration_tenths: 10,
            color_field: None,
            wireframe: false,
        }
    }
}

/// Render settings for one output, per [`OutputKind`]. Stored per asset id
/// when the user overrides the defaults for that output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputRender {
    Model3d {
        mode: PreviewRenderMode,
        resolution: usize,
        asn2: Asn2Settings,
        /// Draw the mesh's edges as an overlay (display-only; never
        /// re-meshes).
        wireframe: bool,
        /// Colormap points/vertices by this declared sample channel
        /// (e.g. `density`); `None` renders plain.
        color_channel: Option<String>,
    },
    Model2d {
        /// Raster cells along the longer bounds axis.
        resolution: usize,
        /// Colormap the raster by this declared sample channel (masked
        /// by occupancy); `None` rasters plain sample values.
        color_channel: Option<String>,
    },
    FeaMesh(FeaRender),
    TriMesh {
        wireframe: bool,
    },
    /// Subspace gizmos have no settings yet; the extent adapts to the
    /// scene.
    Subspace,
}

impl OutputRender {
    fn kind(&self) -> OutputKind {
        match self {
            Self::Model3d { .. } => OutputKind::Model3d,
            Self::Model2d { .. } => OutputKind::Model2d,
            Self::FeaMesh(_) => OutputKind::FeaMesh,
            Self::TriMesh { .. } => OutputKind::TriMesh,
            Self::Subspace => OutputKind::Subspace,
        }
    }

    /// One-line summary for the outputs table ("ASN2 · 64^3", "2D raster
    /// · 256", …).
    fn summary(&self) -> String {
        match self {
            Self::Model3d {
                mode,
                resolution,
                color_channel,
                ..
            } => match color_channel {
                Some(channel) => format!("{} · {}^3 · {channel}", mode.label(), resolution),
                None => format!("{} · {}^3", mode.label(), resolution),
            },
            Self::Model2d {
                resolution,
                color_channel,
            } => match color_channel {
                Some(channel) => format!("2D raster · {resolution} · {channel}"),
                None => format!("2D raster · {resolution}"),
            },
            Self::FeaMesh(fea) => match &fea.color_field {
                Some(field) => format!(
                    "FEA · {}",
                    field.split_once(':').map(|(_, name)| name).unwrap_or(field)
                ),
                None => "FEA".to_string(),
            },
            Self::TriMesh { .. } => "triangle mesh".to_string(),
            Self::Subspace => "subspace".to_string(),
        }
    }

    /// The display-only wireframe overlay flag, where the kind has one.
    fn wireframe(&self) -> bool {
        match self {
            Self::Model3d { wireframe, .. } | Self::TriMesh { wireframe } => *wireframe,
            Self::FeaMesh(fea) => fea.wireframe,
            Self::Model2d { .. } | Self::Subspace => false,
        }
    }
}

/// The per-kind build recipe a [`PreviewRequest`] carries; part of the
/// preview cache key, so any change here rebuilds the preview.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewPlan {
    Model3d {
        mesh: PreviewMeshPlan,
        /// Sample channel to colormap points/vertices by, when declared.
        color_channel: Option<String>,
    },
    Sketch {
        resolution: usize,
        color_channel: Option<String>,
    },
    FeaMesh {
        deformed: bool,
        exaggeration_tenths: u16,
        color_field: Option<String>,
    },
    TriMesh,
    Subspace,
}

impl PreviewPlan {
    pub fn label(&self) -> String {
        match self {
            Self::Model3d {
                mesh,
                color_channel,
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
        }
    }
}

#[derive(Debug)]
pub struct VolumetricUiV2 {
    project: Project,
    selected_export: Option<String>,
    selected_project_item: Option<ProjectSelection>,
    /// Outputs pinned to stay in the viewport regardless of the current
    /// selection. The rendered scene is the selected node's output plus these.
    pinned_outputs: std::collections::BTreeSet<String>,
    /// Per-output render settings; outputs without an entry follow the
    /// kind's defaults (3D models: the global `render_mode`/
    /// `preview_resolution`).
    output_overrides: std::collections::BTreeMap<String, OutputRender>,
    /// Statically probed model dimensionality per output id, keyed by the
    /// asset bytes' (ptr, len) so re-runs revalidate. Interior mutability:
    /// the view path (settings popover) queries kinds through `&self`.
    output_dims:
        std::cell::RefCell<std::collections::HashMap<String, ((usize, usize), Option<u32>)>>,
    render_mode: PreviewRenderMode,
    preview_resolution: usize,
    camera_control_scheme: CameraControlScheme,
    show_grid: bool,
    show_bounds: bool,
    ssao: bool,
    ssao_radius: f32,
    ssao_bias: f32,
    ssao_strength: f32,
    runtime_assets: Vec<LoadedAsset>,
    last_run_elapsed_ms: Option<u128>,
    last_run_error: Option<String>,
    last_run_stale: bool,
    run_state: RunState,
    /// Latest progress snapshot from the in-flight run (shown in the run
    /// chip); `None` when idle or before the first report arrives.
    run_progress: Option<volumetric::BuildProgress>,
    /// Latest progress snapshot from an in-flight preview mesh build.
    preview_progress: Option<volumetric::BuildProgress>,
    auto_rebuild: bool,
    pending_run: bool,
    cancel_requested: bool,
    /// Rebuild viewport meshes automatically when their settings change.
    /// Off, changes accumulate as "stale" until an explicit Remesh — the
    /// sane mode for lattice-scale models where every build is expensive.
    auto_remesh: bool,
    /// One-shot: an explicit Remesh was clicked; the next preview sync
    /// dispatches stale builds even with auto-remesh off.
    remesh_requested: bool,
    /// One-shot: cancel all in-flight preview mesh builds.
    mesh_cancel_requested: bool,
    /// Run project builds and ASN2 meshing on a remote daemon. The address
    /// buffer below applies when this is toggled on.
    remote_build: bool,
    /// Daemon base URL (controlled text input in the remote settings
    /// popover). Edits take effect the next time remote build is toggled on.
    remote_address: String,
    /// One-shot: the shell should swap its execution backend.
    executor_request: Option<ExecutorChoice>,
    /// An Add-menu operator click waiting for dispatch to the background
    /// worker (reading operator metadata compiles its wasm — too slow for
    /// the UI thread). The step is inserted when the result arrives. Only
    /// cold-catalog clicks land here; a warm catalog inserts synchronously.
    operator_add_request: Option<String>,
    /// The operator add whose metadata read is on the worker right now.
    operator_add_inflight: Option<String>,
    /// The Add catalog: every bundled module's declared metadata, warmed
    /// from the persisted cache and background scans (see [`catalog`]).
    pub(crate) catalog: catalog::Catalog,
    /// The Add-catalog modal, when open (holds the search buffer).
    add_modal: Option<AddModalState>,
    /// Catalog names of recently added modules, most recent first, capped
    /// at [`RECENT_ADDS_CAP`] — the recents rail's tiles. Persisted in
    /// settings; fresh installs start from a curated seed.
    recent_adds: Vec<String>,
    preview_build_status: PreviewBuildStatus,
    pending_camera_command: Option<ViewportCameraCommand>,
    viewport_texture: Option<AppTexture>,
    /// Global text selection/focus for controlled text inputs (config fields).
    selection: Selection,
    /// Editor state (model slots + config form) for the selected step, if any.
    step_edit: Option<StepEditState>,
    status: String,
    /// Open menubar menu (`file` | `add`), if any.
    open_menu: Option<String>,
    /// Open viewport picker (one of the `view:*` select keys), if any.
    open_select: Option<String>,
    /// Expanded pipeline accordion sections (`imports` | `steps` | `exports`).
    pipeline_open: std::collections::BTreeSet<String>,
    /// Meshing stats per built output, mirrored from the host's preview cache.
    output_stats: std::collections::BTreeMap<String, OutputStats>,
    /// Geometry the viewport had to drop at the GPU buffer size limit,
    /// mirrored from the renderer each frame; shown as a HUD warning.
    viewport_overflow: Option<String>,
    /// The open 2D inspection lightbox, if any.
    lightbox: Option<LightboxState>,
    /// The open mesh-export modal, if any.
    export_dialog: Option<ExportDialogState>,
    /// A queued file operation for the host to run (dialogs are host-side).
    pending_file_action: Option<FileAction>,
    /// Where the project was last opened from / saved to; Save re-saves here,
    /// Save As always asks. Cleared by New Project.
    project_path: Option<std::path::PathBuf>,
    /// Whether the file at `project_path` carries baked results. Kept true
    /// across re-saves (each one embeds a fresh bake) so a built copy stays
    /// built; set on open/save, cleared by New Project.
    baked_on_disk: bool,
    /// The user's build-cache budget preference (persisted in settings).
    /// The live cache budget can sit above it after a large bake was seeded
    /// ([`volumetric::BuildCache::reserve`]).
    cache_budget_bytes: usize,
    /// Project panel width, adjusted by the divider's resize handle.
    panel_width: f32,
    panel_drag: ResizeDrag,
    /// Decoded operator metadata keyed by operator asset id, so per-frame
    /// consumers (asset pickers) don't instantiate operator WASM on every
    /// call. Entries are revalidated by the import's byte length — an
    /// operator asset replaced by a same-length module with different
    /// metadata would serve one stale answer, which the next run corrects.
    operator_metadata_cache:
        std::cell::RefCell<std::collections::HashMap<String, (usize, Option<OperatorMetadata>)>>,
}

impl Default for VolumetricUiV2 {
    fn default() -> Self {
        let mut app = Self::empty();
        if let Some(model) = volumetric_assets::models().first() {
            app.add_model(model.name);
        }
        app
    }
}

impl VolumetricUiV2 {
    /// Builds the app with no project contents. `default()` additionally seeds
    /// the first bundled model.
    pub fn empty() -> Self {
        Self {
            project: Project::new(),
            selected_export: None,
            selected_project_item: None,
            pinned_outputs: std::collections::BTreeSet::new(),
            output_overrides: std::collections::BTreeMap::new(),
            output_dims: std::cell::RefCell::new(std::collections::HashMap::new()),
            render_mode: PreviewRenderMode::AdaptiveSurfaceNets2,
            preview_resolution: 64,
            camera_control_scheme: CameraControlScheme::default(),
            show_grid: true,
            show_bounds: false,
            ssao: true,
            // Renderer defaults (renderer::RenderSettings::default()).
            ssao_radius: 0.5,
            ssao_bias: 0.025,
            ssao_strength: 1.0,
            runtime_assets: Vec::new(),
            last_run_elapsed_ms: None,
            last_run_error: None,
            last_run_stale: false,
            run_state: RunState::Idle,
            run_progress: None,
            preview_progress: None,
            auto_rebuild: false,
            pending_run: false,
            cancel_requested: false,
            auto_remesh: true,
            remesh_requested: false,
            mesh_cancel_requested: false,
            remote_build: false,
            remote_address: format!("http://127.0.0.1:{}", volumetric_protocol::DEFAULT_PORT),
            executor_request: None,
            operator_add_request: None,
            operator_add_inflight: None,
            catalog: catalog::Catalog::default(),
            add_modal: None,
            recent_adds: default_recent_adds(),
            preview_build_status: PreviewBuildStatus::Idle,
            pending_camera_command: None,
            viewport_texture: None,
            selection: Selection::default(),
            step_edit: None,
            status: "idle".to_string(),
            open_menu: None,
            open_select: None,
            // Steps carry the editing workflow; imports/exports start folded.
            pipeline_open: ["steps"].into_iter().map(str::to_string).collect(),
            output_stats: std::collections::BTreeMap::new(),
            viewport_overflow: None,
            lightbox: None,
            export_dialog: None,
            pending_file_action: None,
            project_path: None,
            baked_on_disk: false,
            cache_budget_bytes: volumetric::build_cache::DEFAULT_BUDGET_BYTES,
            panel_width: PANEL_WIDTH_DEFAULT,
            panel_drag: ResizeDrag::default(),
            operator_metadata_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn runtime_assets(&self) -> &[LoadedAsset] {
        &self.runtime_assets
    }

    pub fn selected_runtime_asset(&self) -> Option<&LoadedAsset> {
        let selected_export = self.selected_export.as_deref()?;
        self.runtime_assets
            .iter()
            .find(|asset| asset.id() == selected_export)
    }

    /// The set of runtime outputs to render this frame: the selected pipeline
    /// node's output plus every pinned output. The host composites them into one
    /// multi-entity scene. Ids that aren't materialized in the current run are
    /// silently skipped (e.g. a selected step whose output isn't exported yet).
    pub fn preview_requests(&self) -> Vec<PreviewRequest> {
        let mut ids: Vec<&str> = self.pinned_outputs.iter().map(String::as_str).collect();
        if let Some(selected) = self.selected_render_id()
            && !ids.contains(&selected)
        {
            ids.push(selected);
        }

        ids.into_iter()
            .filter_map(|id| self.runtime_assets.iter().find(|asset| asset.id() == id))
            .filter_map(|asset| self.render_request_for_asset(asset))
            .collect()
    }

    /// The runtime asset id the current pipeline selection points at, if any.
    /// Exports and imports map to their own id; a step maps to its primary
    /// output. The id may or may not be materialized — callers resolve it
    /// against `runtime_assets`.
    fn selected_render_id(&self) -> Option<&str> {
        match self.selected_project_item.as_ref()? {
            ProjectSelection::Export(idx) => self.project.exports().get(*idx).map(String::as_str),
            ProjectSelection::Import(idx) => self
                .project
                .imports()
                .get(*idx)
                .map(|import| import.id.as_str()),
            ProjectSelection::Step(idx) => self
                .project
                .timeline()
                .get(*idx)
                .and_then(|step| step.outputs.first())
                .map(String::as_str),
        }
    }

    /// Whether the current selection resolves to a materialized runtime output.
    /// When false, the selected node contributes nothing to the viewport (only
    /// pinned outputs render) and the inspector shows a "run to preview" hint.
    fn selection_is_renderable(&self) -> bool {
        self.selected_render_id()
            .is_some_and(|id| self.runtime_assets.iter().any(|asset| asset.id() == id))
    }

    /// Toggles whether an output stays pinned in the viewport across selection
    /// changes.
    fn toggle_pin(&mut self, id: &str) {
        if self.pinned_outputs.take(id).is_none() {
            self.pinned_outputs.insert(id.to_string());
            self.status = format!("pinned {id}");
        } else {
            self.status = format!("unpinned {id}");
        }
    }

    /// Whether an output is currently drawn in the viewport (pinned, or the
    /// resolvable selection).
    fn output_is_visible(&self, id: &str) -> bool {
        self.pinned_outputs.contains(id)
            || (self.selection_is_renderable() && self.selected_render_id() == Some(id))
    }

    /// What kind of output `id` is, for view purposes. FEA and triangle
    /// meshes classify by type hint; models split 2D/3D by a cheap static
    /// scan of the wasm (cached per asset id + byte length). Unknown ids
    /// and unparseable models default to 3D.
    fn output_kind(&self, id: &str) -> OutputKind {
        let Some(asset) = self.runtime_assets.iter().find(|a| a.id() == id) else {
            return OutputKind::Model3d;
        };
        match asset.type_hint() {
            Some(AssetTypeHint::FeaMesh) => OutputKind::FeaMesh,
            Some(AssetTypeHint::TriMesh) => OutputKind::TriMesh,
            Some(AssetTypeHint::Subspace) => OutputKind::Subspace,
            _ => {
                let data = asset.data_arc();
                let key = (Arc::as_ptr(&data) as usize, data.len());
                let mut cache = self.output_dims.borrow_mut();
                let dims = match cache.get(id) {
                    Some((cached_key, dims)) if *cached_key == key => *dims,
                    _ => {
                        let dims = volumetric::model_dimensions_static(&data);
                        cache.insert(id.to_string(), (key, dims));
                        dims
                    }
                };
                if dims == Some(2) {
                    OutputKind::Model2d
                } else {
                    OutputKind::Model3d
                }
            }
        }
    }

    /// Default render settings for an output of the given kind. 3D models
    /// follow the viewport-global mode/resolution defaults.
    fn default_output_render(&self, kind: OutputKind) -> OutputRender {
        match kind {
            OutputKind::Model3d => OutputRender::Model3d {
                mode: self.render_mode,
                resolution: self.preview_resolution,
                asn2: Asn2Settings::default(),
                wireframe: false,
                color_channel: None,
            },
            OutputKind::Model2d => OutputRender::Model2d {
                resolution: SKETCH_RESOLUTION_DEFAULT,
                color_channel: None,
            },
            OutputKind::FeaMesh => OutputRender::FeaMesh(FeaRender::default()),
            OutputKind::TriMesh => OutputRender::TriMesh { wireframe: false },
            OutputKind::Subspace => OutputRender::Subspace,
        }
    }

    /// Effective render settings for an output: its override if present
    /// (and still matching the output's kind — a re-run can change what an
    /// id produces), otherwise the kind's defaults.
    fn output_render(&self, id: &str) -> OutputRender {
        let kind = self.output_kind(id);
        match self.output_overrides.get(id) {
            Some(render) if render.kind() == kind => render.clone(),
            _ => self.default_output_render(kind),
        }
    }

    /// Adjusts one ASN2 setting for an output; `up` steps the value up. The
    /// `sharp` field toggles regardless of direction. 3D models only.
    fn adjust_output_asn2(&mut self, id: &str, field: &str, up: bool) {
        let mut render = self.output_render(id);
        let OutputRender::Model3d { asn2, .. } = &mut render else {
            return;
        };
        let step_usize = |v: usize| {
            if up {
                (v + 1).min(16)
            } else {
                v.saturating_sub(1)
            }
        };
        match field {
            "vr" => {
                asn2.vertex_refinement_iterations = step_usize(asn2.vertex_refinement_iterations)
            }
            "nr" => asn2.normal_sample_iterations = step_usize(asn2.normal_sample_iterations),
            "sharp" => asn2.sharp_edges = !asn2.sharp_edges,
            "edgec" => asn2.edge_constrained_refinement = !asn2.edge_constrained_refinement,
            "simp" => asn2.simplify = !asn2.simplify,
            "simptol" => {
                let next = i32::from(asn2.simplify_tolerance_tenths) + if up { 1 } else { -1 };
                asn2.simplify_tolerance_tenths = next.clamp(1, 30) as u16;
            }
            "angle" => {
                let next = i32::from(asn2.sharp_angle_degrees) + if up { 5 } else { -5 };
                asn2.sharp_angle_degrees = next.clamp(10, 90) as u16;
            }
            "probes" => asn2.discovery_probes = step_usize(asn2.discovery_probes),
            "base" => {
                // Discovery grid presets; 0 is the automatic 6/8 split.
                const BASES: [usize; 8] = [0, 8, 12, 16, 24, 32, 48, 64];
                let index = BASES
                    .iter()
                    .position(|b| *b == asn2.base_resolution)
                    .unwrap_or(0);
                let next = if up {
                    (index + 1).min(BASES.len() - 1)
                } else {
                    index.saturating_sub(1)
                };
                asn2.base_resolution = BASES[next];
            }
            _ => return,
        }
        self.output_overrides.insert(id.to_string(), render);
    }

    fn set_output_mode(&mut self, id: &str, mode: PreviewRenderMode) {
        let mut render = self.output_render(id);
        let OutputRender::Model3d { mode: slot, .. } = &mut render else {
            return;
        };
        *slot = mode;
        self.output_overrides.insert(id.to_string(), render);
        self.status = format!("{id}: {}", mode.full_label());
    }

    fn set_output_resolution(&mut self, id: &str, resolution: usize) {
        let mut render = self.output_render(id);
        match &mut render {
            OutputRender::Model3d {
                resolution: slot, ..
            } => {
                *slot = resolution;
                self.status = format!("{id}: {resolution}^3");
            }
            OutputRender::Model2d {
                resolution: slot, ..
            } => {
                *slot = resolution;
                self.status = format!("{id}: {resolution} px raster");
            }
            _ => return,
        }
        self.output_overrides.insert(id.to_string(), render);
    }

    fn toggle_output_wireframe(&mut self, id: &str) {
        let mut render = self.output_render(id);
        let slot = match &mut render {
            OutputRender::Model3d { wireframe, .. } | OutputRender::TriMesh { wireframe } => {
                wireframe
            }
            OutputRender::FeaMesh(fea) => &mut fea.wireframe,
            OutputRender::Model2d { .. } | OutputRender::Subspace => return,
        };
        *slot = !*slot;
        let state = if *slot { "on" } else { "off" };
        self.output_overrides.insert(id.to_string(), render);
        self.status = format!("{id}: wireframe {state}");
    }

    fn toggle_output_fea_deformed(&mut self, id: &str) {
        let mut render = self.output_render(id);
        let OutputRender::FeaMesh(fea) = &mut render else {
            return;
        };
        fea.deformed = !fea.deformed;
        let state = if fea.deformed {
            "deformed"
        } else {
            "undeformed"
        };
        self.output_overrides.insert(id.to_string(), render);
        self.status = format!("{id}: {state}");
    }

    /// Steps the FEA deformation exaggeration through its preset ladder.
    fn adjust_output_fea_exaggeration(&mut self, id: &str, up: bool) {
        let mut render = self.output_render(id);
        let OutputRender::FeaMesh(fea) = &mut render else {
            return;
        };
        let pos = FEA_EXAGGERATION_TENTHS
            .iter()
            .position(|&t| t >= fea.exaggeration_tenths)
            .unwrap_or(FEA_EXAGGERATION_TENTHS.len() - 1);
        let next = if up {
            (pos + 1).min(FEA_EXAGGERATION_TENTHS.len() - 1)
        } else {
            pos.saturating_sub(1)
        };
        fea.exaggeration_tenths = FEA_EXAGGERATION_TENTHS[next];
        self.status = format!(
            "{id}: deformation x{}",
            fea.exaggeration_tenths as f32 / 10.0
        );
        self.output_overrides.insert(id.to_string(), render);
    }

    /// Sets the sample channel a 3D model output colormaps by (`None` for
    /// plain shading).
    fn set_output_channel(&mut self, id: &str, channel: Option<String>) {
        let mut render = self.output_render(id);
        let slot = match &mut render {
            OutputRender::Model3d { color_channel, .. }
            | OutputRender::Model2d { color_channel, .. } => color_channel,
            _ => return,
        };
        *slot = channel;
        self.status = match slot {
            Some(name) => format!("{id}: color by {name}"),
            None => format!("{id}: plain shading"),
        };
        self.output_overrides.insert(id.to_string(), render);
    }

    /// Sets the FEA colormap field (a `node:{name}` / `element:{name}`
    /// qualified name, or `None` for plain shading).
    fn set_output_fea_field(&mut self, id: &str, field: Option<String>) {
        let mut render = self.output_render(id);
        let OutputRender::FeaMesh(fea) = &mut render else {
            return;
        };
        self.status = match &field {
            Some(name) => format!("{id}: colormap {name}"),
            None => format!("{id}: plain shading"),
        };
        fea.color_field = field;
        self.output_overrides.insert(id.to_string(), render);
    }

    fn clear_output_override(&mut self, id: &str) {
        self.output_overrides.remove(id);
        self.status = format!("{id}: viewport defaults");
    }

    fn open_lightbox(&mut self, id: &str) {
        let mode = match self.output_kind(id) {
            // 3D models inspect an interior slice; default to a z midplane
            // of the first channel beyond occupancy (the interesting one —
            // e.g. density), falling back to the occupancy cross-section.
            OutputKind::Model3d => {
                let channels = self.output_channels(id);
                LightboxMode::Slice {
                    axis: 2,
                    frac_percent: 50,
                    channel: channels
                        .get(1)
                        .or_else(|| channels.first())
                        .cloned()
                        .unwrap_or_else(|| "occupancy".to_string()),
                }
            }
            _ => LightboxMode::Sketch,
        };
        self.lightbox = Some(LightboxState {
            asset_id: id.to_string(),
            mode,
            ..Default::default()
        });
        self.open_select = None;
        self.status = format!("inspecting {id}");
    }

    /// The declared sample channels of a 3D model output, mirrored from its
    /// built preview's stats (empty until a preview builds).
    fn output_channels(&self, id: &str) -> Vec<String> {
        self.output_stats
            .get(id)
            .map(|stats| stats.model_channels.clone())
            .unwrap_or_default()
    }

    /// Opens the mesh-export modal for an output. The session delivers the
    /// cached preview mesh on its next sync (`export_dialog_wants_mesh` →
    /// `set_export_mesh`).
    fn open_export_dialog(&mut self, id: &str) {
        self.export_dialog = Some(ExportDialogState {
            asset_id: id.to_string(),
            mesh: ExportPreviewMesh::Pending,
            scale_text: "1".to_string(),
        });
        self.open_select = None;
    }

    /// The output the open export modal still needs preview geometry for.
    pub fn export_dialog_wants_mesh(&self) -> Option<&str> {
        let export = self.export_dialog.as_ref()?;
        matches!(export.mesh, ExportPreviewMesh::Pending).then_some(export.asset_id.as_str())
    }

    /// Delivers the export modal's preview triangles (ignored if the modal
    /// closed or moved to a different output meanwhile). Empty means no
    /// cached mesh preview; the modal explains instead of exporting.
    pub fn set_export_mesh(&mut self, asset_id: &str, triangles: &[volumetric::Triangle]) {
        let Some(export) = self.export_dialog.as_mut() else {
            return;
        };
        if export.asset_id != asset_id {
            return;
        }
        export.mesh = if triangles.is_empty() {
            ExportPreviewMesh::Missing
        } else {
            let (data, bounds) = export_scene_mesh(triangles);
            ExportPreviewMesh::Ready {
                handle: SceneMeshHandle::new(data),
                triangles: triangles.len(),
                bounds,
            }
        };
    }

    /// Queues the export file action with the modal's settings and closes
    /// the modal. Refuses (status hint) when the mesh never arrived or the
    /// scale doesn't parse — the Export button is disabled in both states,
    /// but synthetic clicks and races land here.
    fn confirm_export(&mut self) {
        let Some(export) = self.export_dialog.as_ref() else {
            return;
        };
        if !matches!(export.mesh, ExportPreviewMesh::Ready { .. }) {
            self.status = "nothing to export — no cached mesh preview".to_string();
            return;
        }
        let Some(scale) = parse_export_scale(&export.scale_text) else {
            self.status = "export scale must be a positive number".to_string();
            return;
        };
        self.pending_file_action = Some(FileAction::ExportMesh {
            id: export.asset_id.clone(),
            scale,
        });
        self.export_dialog = None;
    }

    /// Routes a `lightbox:slice:` control: axis pick, position step, or
    /// channel pick. Any change drops the sampled data so the session
    /// re-enqueues the slice job.
    fn adjust_lightbox_slice(&mut self, rest: &str) {
        let Some(lightbox) = self.lightbox.as_mut() else {
            return;
        };
        let LightboxMode::Slice {
            axis,
            frac_percent,
            channel,
        } = &mut lightbox.mode
        else {
            return;
        };
        if let Some(picked) = rest.strip_prefix("axis:") {
            let Ok(picked) = picked.parse::<usize>() else {
                return;
            };
            if picked > 2 || picked == *axis {
                return;
            }
            *axis = picked;
        } else if let Some(direction) = rest.strip_prefix("pos:") {
            let step: i32 = if direction == "up" { 5 } else { -5 };
            let next = (i32::from(*frac_percent) + step).clamp(0, 100) as u16;
            if next == *frac_percent {
                return;
            }
            *frac_percent = next;
        } else if let Some(picked) = rest.strip_prefix("channel:") {
            if picked == channel.as_str() {
                return;
            }
            *channel = picked.to_string();
        } else {
            return;
        }
        lightbox.data = None;
        lightbox.texture = None;
        lightbox.colorbar = None;
    }

    /// The output (and sampling mode) the open lightbox still needs data
    /// for, if any. The session turns this into a background job.
    pub fn lightbox_wants_data(&self) -> Option<(String, LightboxMode, Arc<Vec<u8>>)> {
        let lightbox = self.lightbox.as_ref()?;
        if lightbox.data.is_some() {
            return None;
        }
        let asset = self
            .runtime_assets
            .iter()
            .find(|a| a.id() == lightbox.asset_id)?;
        Some((
            lightbox.asset_id.clone(),
            lightbox.mode.clone(),
            asset.data_arc(),
        ))
    }

    /// The lightbox data awaiting texture upload, if any. The session
    /// uploads it and hands back the textures.
    pub fn lightbox_wants_texture(&self) -> Option<&LightboxData> {
        let lightbox = self.lightbox.as_ref()?;
        if lightbox.texture.is_some() {
            return None;
        }
        lightbox.data.as_ref()
    }

    /// Delivers the background job's sampled data (ignored if the lightbox
    /// moved to another output, changed slice parameters, or closed
    /// meanwhile — leaving `data` empty makes the session re-enqueue with
    /// the current mode).
    pub fn set_lightbox_data(
        &mut self,
        asset_id: &str,
        mode: &LightboxMode,
        result: Result<LightboxData, String>,
    ) {
        let Some(lightbox) = self.lightbox.as_mut() else {
            return;
        };
        if lightbox.asset_id != asset_id || lightbox.mode != *mode {
            return;
        }
        match result {
            Ok(data) => lightbox.data = Some(data),
            Err(err) => {
                self.lightbox = None;
                self.status = format!("inspect failed: {err}");
            }
        }
    }

    /// Delivers the uploaded raster + colorbar textures for the open
    /// lightbox (called in the same frame as `lightbox_wants_texture`, so
    /// there's no id to re-check).
    pub fn set_lightbox_textures(&mut self, raster: AppTexture, colorbar: AppTexture) {
        if let Some(lightbox) = self.lightbox.as_mut() {
            lightbox.texture = Some(raster);
            lightbox.colorbar = Some(colorbar);
        }
    }

    /// Builds a render request for a single runtime asset, or `None` when it is
    /// not renderable (models and FEA meshes are).
    fn render_request_for_asset(&self, asset: &LoadedAsset) -> Option<PreviewRequest> {
        if !matches!(
            asset.type_hint(),
            Some(
                AssetTypeHint::Model
                    | AssetTypeHint::FeaMesh
                    | AssetTypeHint::TriMesh
                    | AssetTypeHint::Subspace
            ) | None
        ) {
            return None;
        }

        let render = self.output_render(asset.id());
        let plan = match &render {
            OutputRender::Model3d {
                mode,
                resolution,
                asn2,
                color_channel,
                ..
            } => PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::for_mode(*mode, *resolution, *asn2),
                color_channel: color_channel.clone(),
            },
            OutputRender::Model2d {
                resolution,
                color_channel,
            } => PreviewPlan::Sketch {
                resolution: *resolution,
                color_channel: color_channel.clone(),
            },
            OutputRender::FeaMesh(fea) => PreviewPlan::FeaMesh {
                deformed: fea.deformed,
                exaggeration_tenths: fea.exaggeration_tenths,
                color_field: fea.color_field.clone(),
            },
            OutputRender::TriMesh { .. } => PreviewPlan::TriMesh,
            OutputRender::Subspace => PreviewPlan::Subspace,
        };
        Some(PreviewRequest {
            asset_id: asset.id().to_string(),
            data: asset.data_arc(),
            type_hint: asset.type_hint(),
            precursor_ids: asset.precursor_ids().to_vec(),
            plan,
            wireframe: render.wireframe(),
            show_grid: self.show_grid,
            show_bounds: self.show_bounds,
            ssao: self.ssao,
            ssao_radius: self.ssao_radius,
            ssao_bias: self.ssao_bias,
            ssao_strength: self.ssao_strength,
            stale: self.last_run_stale,
        })
    }

    /// Adjusts one SSAO parameter; `up` steps the value up. Radius and bias
    /// step geometrically (their useful ranges span orders of magnitude).
    fn adjust_ssao(&mut self, field: &str, up: bool) {
        let scale = if up { 1.5 } else { 1.0 / 1.5 };
        match field {
            "radius" => self.ssao_radius = (self.ssao_radius * scale).clamp(0.005, 0.5),
            "bias" => self.ssao_bias = (self.ssao_bias * scale).clamp(0.0001, 0.02),
            "strength" => {
                let delta = if up { 0.25 } else { -0.25 };
                self.ssao_strength = (self.ssao_strength + delta).clamp(0.5, 4.0);
            }
            _ => {}
        }
    }

    pub fn camera_control_scheme(&self) -> CameraControlScheme {
        self.camera_control_scheme
    }

    pub(crate) fn set_preview_build_status(&mut self, status: PreviewBuildStatus) {
        self.preview_build_status = status;
    }

    pub(crate) fn set_viewport_overflow(&mut self, message: Option<String>) {
        self.viewport_overflow = message;
    }

    pub(crate) fn take_camera_command(&mut self) -> Option<ViewportCameraCommand> {
        self.pending_camera_command.take()
    }

    pub(crate) fn set_viewport_texture(&mut self, texture: AppTexture) {
        self.viewport_texture = Some(texture);
    }

    pub fn run_state(&self) -> RunState {
        self.run_state
    }

    pub fn auto_rebuild(&self) -> bool {
        self.auto_rebuild
    }

    /// Host hook: consumes a pending run request queued by the UI (a Run click
    /// or, when enabled, an auto-rebuild after a project edit).
    ///
    /// A structurally invalid project never dispatches: the request is dropped
    /// and the validation issues land where a mid-run failure would, so the
    /// user sees "step 2 references 'foo'..." instead of a doomed background
    /// run's NoSuchAssetId.
    pub(crate) fn take_pending_run(&mut self) -> bool {
        if !std::mem::take(&mut self.pending_run) {
            return false;
        }
        let issues = self.project.validate();
        if issues.is_empty() {
            return true;
        }
        let summary = issues
            .iter()
            .map(|issue| issue.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        self.last_run_error = Some(format!("project is invalid: {summary}"));
        self.status = "run blocked: project is invalid".to_string();
        false
    }

    /// Host hook: consumes a pending cancel request for the in-flight run.
    pub(crate) fn take_cancel_request(&mut self) -> bool {
        std::mem::take(&mut self.cancel_requested)
    }

    /// Whether viewport meshes rebuild automatically when their settings
    /// change (the preview sync consults this every frame).
    pub fn auto_remesh(&self) -> bool {
        self.auto_remesh
    }

    /// Host hook: consumes a pending explicit-remesh request.
    pub(crate) fn take_remesh_request(&mut self) -> bool {
        std::mem::take(&mut self.remesh_requested)
    }

    /// Host hook: consumes a pending cancel request for in-flight preview
    /// mesh builds.
    pub(crate) fn take_mesh_cancel_request(&mut self) -> bool {
        std::mem::take(&mut self.mesh_cancel_requested)
    }

    /// One-shot: the executor swap requested by the remote-build toggle.
    pub(crate) fn take_executor_request(&mut self) -> Option<ExecutorChoice> {
        self.executor_request.take()
    }

    pub fn remote_build(&self) -> bool {
        self.remote_build
    }

    /// Host hook: consumes a queued operator-add metadata request for
    /// dispatch to the background worker.
    pub(crate) fn take_operator_metadata_request(&mut self) -> Option<String> {
        let name = self.operator_add_request.take()?;
        self.operator_add_inflight = Some(name.clone());
        Some(name)
    }

    /// Host hook: a module metadata read finished (add-click read or
    /// catalog warm scan — any read warms the catalog). A read the Add
    /// flow is waiting on also inserts its step (or surfaces the error).
    pub(crate) fn on_module_metadata(
        &mut self,
        name: &str,
        result: Result<OperatorMetadata, String>,
    ) {
        self.catalog.on_metadata(name, &result);
        if self.operator_add_inflight.as_deref() != Some(name) {
            return; // a warm scan, or superseded; nothing was waiting on it
        }
        self.operator_add_inflight = None;
        match result {
            Ok(metadata) => self.insert_operator_step(name, &metadata),
            Err(err) => self.status = format!("couldn't read {name} metadata: {err}"),
        }
    }

    /// Host hook: `count` in-flight preview builds were signalled to cancel.
    pub(crate) fn on_mesh_builds_cancelled(&mut self, count: usize) {
        self.status = match count {
            0 => "no mesh builds in flight".to_string(),
            1 => "cancelled 1 mesh build".to_string(),
            n => format!("cancelled {n} mesh builds"),
        };
    }

    pub(crate) fn set_run_state(&mut self, state: RunState) {
        self.run_state = state;
        self.run_progress = None;
        if state == RunState::Running {
            self.status = "running project".to_string();
        }
    }

    /// Host hook: a progress snapshot arrived from the in-flight run.
    pub(crate) fn on_run_progress(&mut self, progress: volumetric::BuildProgress) {
        if self.run_state == RunState::Running {
            self.run_progress = Some(progress);
        }
    }

    /// Host hook: a progress snapshot arrived from an in-flight preview build.
    pub(crate) fn on_preview_progress(&mut self, progress: volumetric::BuildProgress) {
        self.preview_progress = Some(progress);
    }

    /// Host hook: a preview build finished (however it ended); drop its
    /// progress so the chip doesn't show a stale phase.
    pub(crate) fn clear_preview_progress(&mut self) {
        self.preview_progress = None;
    }

    /// Host hook: the in-flight run was cancelled. Its (abandoned) result, if it
    /// still arrives, is discarded by generation on the host side.
    pub(crate) fn on_run_cancelled(&mut self) {
        self.run_state = RunState::Idle;
        self.run_progress = None;
        self.last_run_stale = true;
        self.status = "run cancelled".to_string();
    }

    /// Host hook: apply the result of a completed background project run.
    pub(crate) fn apply_run_result(
        &mut self,
        result: Result<Vec<LoadedAsset>, String>,
        elapsed_ms: u128,
    ) {
        self.run_state = RunState::Idle;
        self.run_progress = None;
        self.last_run_elapsed_ms = Some(elapsed_ms);
        match result {
            Ok(assets) => {
                self.last_run_error = None;
                self.last_run_stale = false;
                self.runtime_assets = assets;

                // Drop pins and render overrides for outputs this run no
                // longer produces.
                let live: std::collections::BTreeSet<String> = self
                    .runtime_assets
                    .iter()
                    .map(|asset| asset.id().to_string())
                    .collect();
                self.pinned_outputs.retain(|id| live.contains(id));
                self.output_overrides.retain(|id, _| live.contains(id));

                // Make sure the viewport shows something: if the selection points
                // at nothing materialized, follow the primary export.
                if !self.selection_is_renderable()
                    && let Some(id) = self
                        .runtime_assets
                        .first()
                        .map(|asset| asset.id().to_string())
                {
                    let export_idx = self.project.exports().iter().position(|e| *e == id);
                    self.selected_export = Some(id);
                    self.selected_project_item = export_idx.map(ProjectSelection::Export);
                }

                self.status = format!(
                    "ran project: {} exports in {elapsed_ms}ms",
                    self.runtime_assets.len()
                );
            }
            Err(err) => {
                // Keep the last good runtime assets on screen; surface the error
                // and leave the output marked stale.
                self.last_run_error = Some(err.clone());
                self.last_run_stale = true;
                self.status = format!("project run failed: {err}");
            }
        }
    }

    pub fn summary(&self) -> ProjectSummary {
        ProjectSummary {
            imports: self.project.imports().len(),
            timeline_steps: self.project.timeline().len(),
            exports: self.project.exports().len(),
            selected_export: self.selected_export.clone(),
            selected_project_item: self.selected_project_item.clone(),
            pinned_outputs: self.pinned_outputs.iter().cloned().collect(),
            render_mode: self.render_mode,
            preview_resolution: self.preview_resolution,
            camera_control_scheme: self.camera_control_scheme,
            show_grid: self.show_grid,
            show_bounds: self.show_bounds,
            ssao: self.ssao,
            runtime_assets: self
                .runtime_assets
                .iter()
                .map(|asset| RuntimeAssetSummary {
                    id: asset.id().to_string(),
                    bytes: asset.data().len(),
                    type_hint: asset.type_hint(),
                    precursor_count: asset.precursor_ids().len(),
                })
                .collect(),
            last_run_elapsed_ms: self.last_run_elapsed_ms,
            last_run_error: self.last_run_error.clone(),
            last_run_stale: self.last_run_stale,
            run_state: self.run_state,
            auto_rebuild: self.auto_rebuild,
        }
    }

    /// Drops materialized runtime assets and run bookkeeping. Used when the
    /// whole project is replaced (new/open) — there is no stale output worth
    /// keeping on screen.
    fn clear_runtime_assets(&mut self) {
        self.runtime_assets.clear();
        self.pinned_outputs.clear();
        self.output_overrides.clear();
        self.last_run_elapsed_ms = None;
        self.last_run_error = None;
        self.last_run_stale = false;
        self.pending_run = false;
    }

    /// Marks the last run's output stale after an incremental project edit. The
    /// previous runtime assets stay in place so the viewport keeps showing the
    /// last good preview until a fresh run replaces it; the run is queued now
    /// when auto-rebuild is enabled.
    fn mark_project_dirty(&mut self) {
        self.last_run_stale = true;
        if self.auto_rebuild {
            self.pending_run = true;
        }
    }

    /// Queues an asynchronous project run for the host to pick up.
    fn request_run(&mut self) {
        self.pending_run = true;
        self.status = "run queued".to_string();
    }

    /// One-shot: the file operation queued by a File-menu or output action.
    pub(crate) fn take_file_action(&mut self) -> Option<FileAction> {
        self.pending_file_action.take()
    }

    /// Applies a project the host's file worker loaded off-thread: swaps it
    /// in and marks it stale. A run is queued only when auto-rebuild is on —
    /// opening a file must not commit the user to a build that can take
    /// many minutes; the Run button materializes outputs on demand.
    pub(crate) fn apply_opened_project(
        &mut self,
        path: std::path::PathBuf,
        result: Result<Project, String>,
    ) {
        match result {
            Ok(mut project) => {
                // A built copy's bake moves into the process cache here, so
                // the next run serves those steps without executing them.
                let had_bake = project.baked.is_some();
                let seed = project.seed_build_cache(volumetric::build_cache::global());
                self.project = project;
                self.selected_export = None;
                self.selected_project_item = None;
                self.clear_runtime_assets();
                self.mark_project_dirty();
                self.status = match (had_bake, seed.corrupt_blobs) {
                    (false, _) => format!("opened {}", path.display()),
                    (true, 0) => format!(
                        "opened built copy {} ({} steps ready)",
                        path.display(),
                        seed.seeded_steps
                    ),
                    (true, corrupt) => format!(
                        "opened built copy {} ({} steps ready; {corrupt} damaged blobs dropped)",
                        path.display(),
                        seed.seeded_steps
                    ),
                };
                self.baked_on_disk = had_bake;
                self.project_path = Some(path);
            }
            Err(err) => self.status = format!("failed to open project: {err}"),
        }
    }

    /// Applies the result of a save the host's file worker performed
    /// off-thread; a successful save pins the path for one-click re-saves.
    /// `bake` reports what a built-copy save embedded (`None` for a lean
    /// save) and pins whether the file on disk is now a built copy.
    pub(crate) fn apply_saved_project(
        &mut self,
        path: std::path::PathBuf,
        result: Result<(), String>,
        bake: Option<volumetric::BakeCoverage>,
    ) {
        match result {
            Ok(()) => {
                self.status = match bake {
                    None => format!("saved {}", path.display()),
                    Some(c) if c.is_complete() => format!(
                        "saved built copy {} ({} steps embedded)",
                        path.display(),
                        c.baked_steps
                    ),
                    Some(c) => format!(
                        "saved built copy {} ({}/{} steps embedded — run the build for a complete copy)",
                        path.display(),
                        c.baked_steps,
                        c.total_steps
                    ),
                };
                self.baked_on_disk = bake.is_some_and(|c| c.baked_steps > 0);
                self.project_path = Some(path);
            }
            Err(err) => self.status = format!("failed to save project: {err}"),
        }
    }

    /// The project as it should be written by a save: a built copy gets a
    /// fresh bake collected from the process cache (reported alongside); a
    /// lean save is a plain clone. The in-memory project never carries a
    /// bake (opening consumes it), so this attaches to the copy only.
    pub(crate) fn project_for_save(
        &self,
        built: bool,
    ) -> (Project, Option<volumetric::BakeCoverage>) {
        let mut project = self.project.clone();
        project.baked = None;
        if !built {
            return (project, None);
        }
        let (baked, coverage) = project.collect_baked(volumetric::build_cache::global());
        project.baked = (!baked.is_empty()).then_some(baked);
        (project, Some(coverage))
    }

    pub(crate) fn baked_on_disk(&self) -> bool {
        self.baked_on_disk
    }

    pub(crate) fn project_path(&self) -> Option<&std::path::Path> {
        self.project_path.as_deref()
    }

    /// Doubles or halves the build-cache budget preference and applies it
    /// to the process cache immediately.
    fn adjust_cache_budget(&mut self, up: bool) {
        const MIN: usize = 64 << 20;
        const MAX: usize = 64 << 30;
        self.cache_budget_bytes = if up {
            self.cache_budget_bytes.saturating_mul(2).min(MAX)
        } else {
            (self.cache_budget_bytes / 2).max(MIN)
        };
        self.set_cache_budget(self.cache_budget_bytes);
        self.status = format!(
            "build cache budget: {}",
            format_mb(self.cache_budget_bytes)
        );
    }

    /// Sets the build-cache budget preference (settings restore path).
    pub(crate) fn set_cache_budget(&mut self, bytes: usize) {
        self.cache_budget_bytes = bytes;
        volumetric::build_cache::global().set_budget(bytes);
    }

    pub(crate) fn cache_budget_bytes(&self) -> usize {
        self.cache_budget_bytes
    }

    /// Loads a project from disk inline and applies it. Test convenience —
    /// the host shell routes disk I/O through its file worker instead.
    #[cfg(test)]
    pub(crate) fn open_project_file(&mut self, path: &std::path::Path) {
        let result = Project::load_from_file(path).map_err(|err| err.to_string());
        self.apply_opened_project(path.to_path_buf(), result);
    }

    /// Saves the project to disk inline and applies the outcome. Test
    /// convenience — the host shell routes disk I/O through its file worker
    /// instead.
    #[cfg(test)]
    pub(crate) fn save_project_file(&mut self, path: &std::path::Path) {
        let result = self
            .project
            .save_to_file(path)
            .map_err(|err| err.to_string());
        self.apply_saved_project(path.to_path_buf(), result, None);
    }

    /// Status line for host-side operations (e.g. STL export results).
    pub(crate) fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    /// Meshing stats mirrored from the host's preview cache each frame.
    pub(crate) fn set_output_stats(
        &mut self,
        stats: std::collections::BTreeMap<String, OutputStats>,
    ) {
        self.output_stats = stats;
    }

    /// Rebuilds the step editor when the selected step changes. While the same
    /// step stays selected the existing config buffers (in-progress edits) are
    /// kept.
    fn sync_step_edit(&mut self) {
        let step_idx = match self.selected_project_item {
            Some(ProjectSelection::Step(idx)) => Some(idx),
            _ => None,
        };
        let Some(step_idx) = step_idx else {
            self.step_edit = None;
            return;
        };
        if self.step_edit.as_ref().map(|edit| edit.step_idx) == Some(step_idx) {
            return;
        }
        self.step_edit = self.build_step_edit(step_idx);
    }

    fn build_step_edit(&self, step_idx: usize) -> Option<StepEditState> {
        let step = self.project.timeline().get(step_idx)?;
        let op_bytes = self.operator_bytes(&step.operator_id)?;
        let metadata = volumetric::operator_metadata_from_wasm_bytes(&op_bytes).ok()?;

        let asset_slots: Vec<AssetSlot> = metadata
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(idx, input)| {
                let kind = match input {
                    OperatorMetadataInput::ModelWASM => AssetSlotKind::Model,
                    OperatorMetadataInput::FeaMesh => AssetSlotKind::FeaMesh,
                    OperatorMetadataInput::TriMesh => AssetSlotKind::TriMesh,
                    OperatorMetadataInput::Subspace => AssetSlotKind::Subspace,
                    _ => return None,
                };
                Some(AssetSlot {
                    input_idx: idx,
                    kind,
                    name: metadata.input_name(idx).map(str::to_string),
                })
            })
            .collect();

        let vecs: Vec<VecForm> = metadata
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(idx, input)| {
                let OperatorMetadataInput::VecF64(dim) = input else {
                    return None;
                };
                let values = match step.inputs.get(idx) {
                    Some(ExecutionInput::Inline(bytes)) => decode_vec_f64(bytes, *dim),
                    _ => vec![0.0; *dim],
                };
                Some(VecForm {
                    input_idx: idx,
                    name: metadata.input_name(idx).map(str::to_string),
                    buffers: values.iter().map(|v| format!("{v}")).collect(),
                })
            })
            .collect();

        let config = metadata
            .inputs
            .iter()
            .enumerate()
            .find_map(|(idx, input)| match input {
                OperatorMetadataInput::CBORConfiguration(cddl) => Some((idx, cddl.clone())),
                _ => None,
            })
            .and_then(|(input_idx, cddl)| self.build_config_form(step, input_idx, &cddl));

        let lua = metadata
            .inputs
            .iter()
            .position(|input| matches!(input, OperatorMetadataInput::LuaSource(_)))
            .map(|input_idx| {
                let source = match step.inputs.get(input_idx) {
                    Some(ExecutionInput::Inline(bytes)) => {
                        String::from_utf8_lossy(bytes).into_owned()
                    }
                    _ => String::new(),
                };
                LuaForm { input_idx, source }
            });

        // Nothing editable → no step editor.
        if asset_slots.is_empty() && vecs.is_empty() && config.is_none() && lua.is_none() {
            return None;
        }

        Some(StepEditState {
            step_idx,
            asset_slots,
            vecs,
            config,
            lua,
            output_name: step.outputs.first().cloned().unwrap_or_default(),
        })
    }

    fn build_config_form(
        &self,
        step: &volumetric::ExecutionStep,
        input_idx: usize,
        cddl: &str,
    ) -> Option<ConfigForm> {
        let fields = operator_config::parse_schema(cddl).ok()?;
        if fields.is_empty() {
            return None;
        }
        let current = match step.inputs.get(input_idx) {
            Some(ExecutionInput::Inline(bytes)) => operator_config::decode(bytes),
            _ => std::collections::BTreeMap::new(),
        };
        let buffers = fields
            .iter()
            .map(|field| {
                // An unset optional field shows an empty buffer (clearing the
                // field is how it gets unset again).
                let text = match current.get(&field.name) {
                    Some(value) => value.to_display_string(),
                    None if field.optional => String::new(),
                    None => field.seed_value().to_display_string(),
                };
                (field.name.clone(), text)
            })
            .collect();
        Some(ConfigForm {
            input_idx,
            fields,
            buffers,
        })
    }

    fn operator_bytes(&self, operator_id: &str) -> Option<Vec<u8>> {
        self.project
            .imports()
            .iter()
            .find(|import| import.id == operator_id)
            .map(|import| import.data.clone())
    }

    /// The decoded metadata of an operator import, cached (see the
    /// `operator_metadata_cache` field for the invalidation rule).
    fn operator_metadata_cached(&self, operator_id: &str) -> Option<OperatorMetadata> {
        let import_len = self
            .project
            .imports()
            .iter()
            .find(|import| import.id == operator_id)?
            .data
            .len();

        let mut cache = self.operator_metadata_cache.borrow_mut();
        if let Some((len, metadata)) = cache.get(operator_id)
            && *len == import_len
        {
            return metadata.clone();
        }
        let metadata = self
            .operator_bytes(operator_id)
            .and_then(|bytes| volumetric::operator_metadata_from_wasm_bytes(&bytes).ok());
        cache.insert(operator_id.to_string(), (import_len, metadata.clone()));
        metadata
    }

    /// Declared assets with step-output hints refined by operator metadata.
    /// `Project::declared_assets` alone is a static inspection that assumes
    /// every step output is a model; here outputs of operators that declare
    /// other output types (e.g. FeaMesh) get their true hint.
    fn declared_assets_typed(&self) -> Vec<(String, Option<AssetTypeHint>)> {
        let mut assets = self.project.declared_assets();
        let mut output_hints: std::collections::HashMap<String, AssetTypeHint> =
            std::collections::HashMap::new();
        for step in self.project.timeline() {
            let Some(metadata) = self.operator_metadata_cached(&step.operator_id) else {
                continue;
            };
            for (idx, output_id) in step.outputs.iter().enumerate() {
                if let Some(output) = metadata.outputs.get(idx) {
                    output_hints.insert(output_id.clone(), AssetTypeHint::from(output));
                }
            }
        }
        for (id, hint) in assets.iter_mut() {
            if let Some(refined) = output_hints.get(id) {
                *hint = Some(*refined);
            }
        }
        assets
    }

    /// Retargets one model input slot of a step. Retargeting the primary (first)
    /// model slot also renames the step's first output after the new input and
    /// rewires exports, matching how a freshly added operator is named.
    fn set_step_model_input(&mut self, step_idx: usize, input_idx: usize, asset_id: &str) {
        let primary_slot = self
            .step_edit
            .as_ref()
            .filter(|edit| edit.step_idx == step_idx)
            .and_then(|edit| edit.asset_slots.first().map(|slot| slot.input_idx))
            .unwrap_or(0);
        let rename_output = input_idx == primary_slot;

        let Some(step) = self.project.timeline().get(step_idx) else {
            return;
        };
        if input_idx >= step.inputs.len() {
            return;
        }
        let operator_id = step.operator_id.clone();
        let old_outputs = step.outputs.clone();
        let new_output = rename_output.then(|| {
            self.project
                .default_output_name(&operator_id, Some(asset_id))
        });

        if let Some(step) = self.project.timeline_mut().get_mut(step_idx) {
            step.inputs[input_idx] = ExecutionInput::AssetRef(asset_id.to_string());
            if let Some(new_output) = &new_output {
                if step.outputs.is_empty() {
                    step.outputs.push(new_output.clone());
                } else {
                    step.outputs[0] = new_output.clone();
                }
            }
        }

        if let Some(new_output) = new_output {
            for old_output in old_outputs {
                self.project.exports_mut().retain(|id| id != &old_output);
            }
            if !self.project.exports().iter().any(|id| id == &new_output) {
                self.project.exports_mut().push(new_output.clone());
            }
            self.selected_export = Some(new_output);
        }

        self.mark_project_dirty();
        self.selected_project_item = Some(ProjectSelection::Step(step_idx));
        self.status = format!(
            "step {} input {} -> {asset_id}",
            step_idx + 1,
            input_idx + 1
        );
    }

    /// Restores the selected step's Lua source to the operator's template.
    fn reset_lua_source(&mut self, step_idx: usize) {
        let Some(step) = self.project.timeline().get(step_idx) else {
            return;
        };
        let Some(op_bytes) = self.operator_bytes(&step.operator_id) else {
            return;
        };
        let Ok(metadata) = volumetric::operator_metadata_from_wasm_bytes(&op_bytes) else {
            return;
        };
        let Some(template) = metadata.inputs.iter().find_map(|input| match input {
            OperatorMetadataInput::LuaSource(template) => Some(template.clone()),
            _ => None,
        }) else {
            return;
        };
        let Some(mut edit) = self.step_edit.take() else {
            return;
        };
        if edit.step_idx == step_idx
            && let Some(lua) = edit.lua.as_mut()
        {
            lua.source = template;
            self.write_lua_source(step_idx, lua);
            self.status = format!("step {} script reset to template", step_idx + 1);
        }
        self.step_edit = Some(edit);
    }

    /// Commits the step editor's output-name buffer: renames the step's
    /// outputs and rewrites every reference (exports, downstream `AssetRef`
    /// inputs, pins, render overrides, selection). v1 left downstream
    /// references dangling on rename; this doesn't.
    fn rename_step_output(&mut self, step_idx: usize) {
        let Some(edit) = &self.step_edit else {
            return;
        };
        if edit.step_idx != step_idx {
            return;
        }
        let new_name = edit.output_name.trim().to_string();
        if new_name.is_empty() {
            self.status = "output name can't be empty".to_string();
            return;
        }
        let Some(step) = self.project.timeline().get(step_idx) else {
            return;
        };
        let old_outputs = step.outputs.clone();
        let Some(old_name) = old_outputs.first().cloned() else {
            return;
        };
        if new_name == old_name {
            return;
        }
        if self
            .project
            .declared_assets()
            .iter()
            .any(|(id, _)| *id == new_name)
        {
            self.status = format!("{new_name} is already in use");
            return;
        }

        // First output takes the new name; extra outputs get `_i` suffixes
        // (same convention as v1).
        let new_outputs: Vec<String> = old_outputs
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i == 0 {
                    new_name.clone()
                } else {
                    format!("{new_name}_{i}")
                }
            })
            .collect();
        if let Some(step) = self.project.timeline_mut().get_mut(step_idx) {
            step.outputs = new_outputs.clone();
        }

        for (old, new) in old_outputs.iter().zip(&new_outputs) {
            for export in self.project.exports_mut().iter_mut() {
                if export == old {
                    *export = new.clone();
                }
            }
            for step in self.project.timeline_mut().iter_mut() {
                for input in step.inputs.iter_mut() {
                    if let ExecutionInput::AssetRef(id) = input
                        && id == old
                    {
                        *id = new.clone();
                    }
                }
            }
            if self.pinned_outputs.remove(old) {
                self.pinned_outputs.insert(new.clone());
            }
            if let Some(render) = self.output_overrides.remove(old) {
                self.output_overrides.insert(new.clone(), render);
            }
            if self.selected_export.as_deref() == Some(old.as_str()) {
                self.selected_export = Some(new.clone());
            }
        }

        self.mark_project_dirty();
        self.status = format!("renamed {old_name} -> {new_name}");
    }

    /// Commits the field's current buffer into the step's CBOR blob, but only if
    /// it parses; an invalid intermediate (e.g. an empty number field) is left
    /// in the buffer untouched.
    /// Parses one `VecF64` component buffer and patches its 8 bytes in the
    /// step's inline little-endian f64 payload. Unparseable text is left in
    /// the buffer (like config fields) and simply not committed.
    fn commit_vec_component(
        &mut self,
        step_idx: usize,
        input_idx: usize,
        component: usize,
        buffer: &str,
    ) {
        let Ok(value) = buffer.trim().parse::<f64>() else {
            return;
        };
        let Some(step) = self.project.timeline_mut().get_mut(step_idx) else {
            return;
        };
        let Some(ExecutionInput::Inline(bytes)) = step.inputs.get_mut(input_idx) else {
            return;
        };
        let end = (component + 1) * 8;
        if bytes.len() < end {
            bytes.resize(end, 0);
        }
        bytes[component * 8..end].copy_from_slice(&value.to_le_bytes());
        self.mark_project_dirty();
    }

    fn commit_config_buffer(&mut self, step_idx: usize, config: &ConfigForm, field_name: &str) {
        let Some(field) = config.fields.iter().find(|f| f.name == field_name) else {
            return;
        };
        let Some(buffer) = config.buffers.get(field_name) else {
            return;
        };
        // Clearing an optional field unsets it: the value drops out of the
        // encoded map and the operator's absent-field behavior applies.
        let value = if field.optional && buffer.trim().is_empty() {
            None
        } else {
            match ConfigValue::parse(&field.ty, buffer) {
                Some(value) => Some(value),
                None => return,
            }
        };
        self.write_config_value(step_idx, config, field_name, value);
        self.mark_project_dirty();
    }

    fn write_config_value(
        &mut self,
        step_idx: usize,
        config: &ConfigForm,
        field_name: &str,
        value: Option<ConfigValue>,
    ) {
        let Some(step) = self.project.timeline_mut().get_mut(step_idx) else {
            return;
        };
        let mut values = match step.inputs.get(config.input_idx) {
            Some(ExecutionInput::Inline(bytes)) => operator_config::decode(bytes),
            _ => return,
        };
        match value {
            Some(value) => values.insert(field_name.to_string(), value),
            None => values.remove(field_name),
        };
        let encoded = operator_config::encode(&config.fields, &values);
        if let Some(ExecutionInput::Inline(slot)) = step.inputs.get_mut(config.input_idx) {
            *slot = encoded;
        }
    }

    /// Sets a config field's buffer to `text` (bool/enum controls) and commits.
    fn set_config_buffer(&mut self, field_name: &str, text: String) {
        let Some(mut edit) = self.step_edit.take() else {
            return;
        };
        if let Some(config) = edit.config.as_mut() {
            if let Some(buffer) = config.buffers.get_mut(field_name) {
                *buffer = text;
            }
        }
        if let Some(config) = edit.config.as_ref() {
            self.commit_config_buffer(edit.step_idx, config, field_name);
        }
        self.step_edit = Some(edit);
    }

    fn toggle_config_bool(&mut self, field_name: &str) {
        let current = self
            .step_edit
            .as_ref()
            .and_then(|edit| edit.config.as_ref())
            .and_then(|config| config.buffers.get(field_name))
            .map(|buffer| buffer == "true")
            .unwrap_or(false);
        self.set_config_buffer(field_name, (!current).to_string());
    }

    /// Writes the Lua editor's current source into the step's input bytes.
    fn write_lua_source(&mut self, step_idx: usize, lua: &LuaForm) {
        if let Some(step) = self.project.timeline_mut().get_mut(step_idx) {
            if let Some(ExecutionInput::Inline(slot)) = step.inputs.get_mut(lua.input_idx) {
                *slot = lua.source.clone().into_bytes();
            }
        }
        self.mark_project_dirty();
    }

    /// Imports external model WASM bytes as a project import. Called by the
    /// host after its Import dialog.
    pub(crate) fn import_model_wasm(&mut self, name: &str, bytes: Vec<u8>) {
        let id = self.project.insert_model(name, bytes);
        self.mark_project_dirty();
        self.selected_export = Some(id.clone());
        self.selected_project_item = self
            .project
            .imports()
            .len()
            .checked_sub(1)
            .map(ProjectSelection::Import);
        self.status = format!("imported {id}");
    }

    /// Imports a data file (STL mesh, image, …) by staging the named
    /// bundled operator with the file bytes wired into its Blob input. Called
    /// by the host after its Import dialog.
    pub(crate) fn import_blob_asset(
        &mut self,
        operator_name: &str,
        output_base: &str,
        blob: Vec<u8>,
    ) {
        let Some(asset) = volumetric_assets::get_operator(operator_name) else {
            self.status = format!("missing bundled operator {operator_name}");
            return;
        };
        let metadata = match volumetric::operator_metadata_from_wasm_bytes(asset.bytes) {
            Ok(metadata) => metadata,
            Err(err) => {
                self.status = format!("couldn't read {operator_name} metadata: {err}");
                return;
            }
        };
        // Import operators take no model inputs, so the primary-model id is
        // never referenced; the schema-derived config defaults are what matter.
        let mut inputs = operator_step_inputs(&metadata, &SlotPrimaries::default());
        let Some(blob_slot) = metadata
            .inputs
            .iter()
            .position(|input| matches!(input, OperatorMetadataInput::Blob))
        else {
            self.status = format!("{operator_name} declares no Blob input");
            return;
        };
        inputs[blob_slot] = ExecutionInput::Inline(blob);

        let output_id = self.project.default_output_name(output_base, None);
        self.project.insert_operation(
            asset.name,
            asset.bytes.to_vec(),
            inputs,
            vec![output_id.clone()],
            output_id.clone(),
        );

        self.mark_project_dirty();
        self.selected_export = Some(output_id.clone());
        self.selected_project_item = self
            .project
            .timeline()
            .len()
            .checked_sub(1)
            .map(ProjectSelection::Step);
        self.status = format!("imported {output_id}");
    }

    fn add_model(&mut self, name: &str) {
        let Some(asset) = volumetric_assets::get_model(name) else {
            self.status = format!("missing bundled model {name}");
            return;
        };
        let display = self.catalog.display_name(name).to_string();

        let id = self.project.insert_model(asset.name, asset.bytes.to_vec());
        self.mark_project_dirty();
        self.selected_export = Some(id.clone());
        self.selected_project_item = self
            .project
            .imports()
            .len()
            .checked_sub(1)
            .map(ProjectSelection::Import);
        self.status = format!("imported {display} as {id}");
        self.record_recent_add(name);
    }

    /// Move `name` to the front of the recents rail (added or re-added).
    fn record_recent_add(&mut self, name: &str) {
        self.recent_adds.retain(|recent| recent != name);
        self.recent_adds.insert(0, name.to_string());
        self.recent_adds.truncate(RECENT_ADDS_CAP);
    }

    /// Inserts a timeline step for the named bundled operator. With a warm
    /// catalog the declared metadata is already known and the step inserts
    /// synchronously; otherwise the metadata read runs on the background
    /// worker (it compiles the operator's wasm — seconds on a cold debug
    /// cache) and [`Self::on_module_metadata`] inserts the step when it
    /// lands.
    fn add_operator(&mut self, name: &str) {
        if volumetric_assets::get_operator(name).is_none() {
            self.status = format!("missing bundled operator {name}");
            return;
        }
        if let Some(metadata) = self
            .catalog
            .get(name)
            .and_then(catalog::CatalogEntry::ready)
        {
            let metadata = metadata.clone();
            self.insert_operator_step(name, &metadata);
            return;
        }
        if self.operator_add_request.is_some() || self.operator_add_inflight.is_some() {
            self.status = "still adding the previous operator".to_string();
            return;
        }
        self.operator_add_request = Some(name.to_string());
        self.status = format!("adding {}…", self.catalog.display_name(name));
    }

    /// Appends a timeline step for a bundled operator whose metadata just
    /// arrived from the worker: typed slots wire to the current selection
    /// (or the first asset of the right kind), config/Lua slots get their
    /// schema defaults. Operators with a model input need a model to exist;
    /// generators (no model slots) insert into an empty project.
    fn insert_operator_step(&mut self, name: &str, metadata: &OperatorMetadata) {
        let Some(asset) = volumetric_assets::get_operator(name) else {
            self.status = format!("missing bundled operator {name}");
            return;
        };
        let display = self.catalog.display_name(name).to_string();
        let selection = self.selected_export.clone();

        let typed = self.declared_assets_typed();
        let hint_of = |id: &str| {
            typed
                .iter()
                .find(|(asset_id, _)| asset_id == id)
                .and_then(|(_, hint)| *hint)
        };
        let first_of = |kind: AssetTypeHint| {
            typed
                .iter()
                .find(|(_, hint)| *hint == Some(kind))
                .map(|(id, _)| id.clone())
        };
        let selected_hint = selection.as_deref().map(hint_of);
        let for_kind = |kind: AssetTypeHint| {
            if selected_hint == Some(Some(kind)) {
                selection.clone()
            } else {
                first_of(kind)
            }
        };
        // The selection counts as the primary model when it is one (or its
        // type is unknown); otherwise the first model in the project serves.
        let primary_model = if matches!(selected_hint, Some(Some(AssetTypeHint::Model) | None)) {
            selection.clone()
        } else {
            first_of(AssetTypeHint::Model)
        };
        let needs_model = metadata
            .inputs
            .iter()
            .any(|input| matches!(input, OperatorMetadataInput::ModelWASM));
        if needs_model && primary_model.is_none() {
            self.status = format!("{display} needs a model input — add or select a model first");
            return;
        }

        let output_id = self
            .project
            .default_output_name(asset.name, primary_model.as_deref());
        let inputs = operator_step_inputs(
            metadata,
            &SlotPrimaries {
                model: primary_model.as_deref().unwrap_or_default(),
                fea: for_kind(AssetTypeHint::FeaMesh).as_deref(),
                trimesh: for_kind(AssetTypeHint::TriMesh).as_deref(),
                subspace: for_kind(AssetTypeHint::Subspace).as_deref(),
            },
        );

        self.project.insert_operation(
            asset.name,
            asset.bytes.to_vec(),
            inputs,
            vec![output_id.clone()],
            output_id.clone(),
        );
        self.mark_project_dirty();

        self.selected_export = Some(output_id.clone());
        self.selected_project_item = self
            .project
            .timeline()
            .len()
            .checked_sub(1)
            .map(ProjectSelection::Step);
        self.status = format!("added {display} -> {output_id}");
        self.record_recent_add(name);
    }

    /// Replaces an imported operator's bytes with the matching bundled
    /// build (the offer surfaced by [`operator_upgrade_offer`]). Steps keep
    /// referencing the import by id; if the new metadata declares more
    /// inputs than a step carries, the missing slots are appended with
    /// defaults (fewer: extras truncated) so upgraded steps stay runnable.
    fn upgrade_import(&mut self, idx: usize) {
        let Some(import) = self.project.imports().get(idx) else {
            return;
        };
        let import_id = import.id.clone();
        let Some(metadata) = self.operator_metadata_cached(&import_id) else {
            return;
        };
        let Some(bundled) = volumetric_assets::get_operator(&metadata.name) else {
            return;
        };
        let Ok(new_metadata) = volumetric::operator_metadata_from_wasm_bytes(bundled.bytes) else {
            return;
        };

        self.project.imports_mut()[idx].data = bundled.bytes.to_vec();
        // The metadata cache revalidates by byte length only; same-length
        // upgrades would otherwise serve the old metadata.
        self.operator_metadata_cache.borrow_mut().remove(&import_id);

        let fresh_inputs = operator_step_inputs(&new_metadata, &SlotPrimaries::default());
        for step in self.project.timeline_mut() {
            if step.operator_id != import_id {
                continue;
            }
            step.inputs.truncate(fresh_inputs.len());
            for missing in step.inputs.len()..fresh_inputs.len() {
                step.inputs.push(fresh_inputs[missing].clone());
            }
        }

        self.step_edit = None;
        self.mark_project_dirty();
        self.status = format!(
            "updated {import_id} to {} {}",
            new_metadata.name, new_metadata.version
        );
    }

    fn delete_import(&mut self, idx: usize) {
        if idx >= self.project.imports().len() {
            return;
        }

        let import_id = self.project.imports()[idx].id.clone();
        self.project.imports_mut().remove(idx);
        self.mark_project_dirty();

        let mut removed_outputs = Vec::new();
        for step_idx in (0..self.project.timeline().len()).rev() {
            let remove_step = self
                .project
                .timeline()
                .get(step_idx)
                .is_some_and(|step| step_depends_on_asset(step, &import_id));
            if remove_step {
                if let Some(step) = self.project.timeline().get(step_idx) {
                    removed_outputs.extend(step.outputs.clone());
                }
                self.project.timeline_mut().remove(step_idx);
            }
        }

        self.project
            .exports_mut()
            .retain(|id| id != &import_id && !removed_outputs.iter().any(|removed| removed == id));

        if self.selected_export.as_deref() == Some(import_id.as_str())
            || self
                .selected_export
                .as_ref()
                .is_some_and(|id| removed_outputs.iter().any(|removed| removed == id))
        {
            self.selected_export = self.project.exports().last().cloned();
        }

        self.selected_project_item = None;
        self.status = format!("removed import {import_id}");
    }

    fn delete_step(&mut self, idx: usize) {
        if idx >= self.project.timeline().len() {
            return;
        }

        let step = self.project.timeline_mut().remove(idx);
        self.mark_project_dirty();
        for output_id in &step.outputs {
            self.project.exports_mut().retain(|id| id != output_id);
        }

        if self
            .selected_export
            .as_ref()
            .is_some_and(|id| step.outputs.iter().any(|output| output == id))
        {
            self.selected_export = self.project.exports().last().cloned();
        }

        self.selected_project_item = None;
        self.status = format!("removed step {}", idx + 1);
    }

    fn move_step(&mut self, idx: usize, offset: isize) {
        let len = self.project.timeline().len();
        let Some(target_idx) = idx.checked_add_signed(offset) else {
            return;
        };
        if idx >= len || target_idx >= len {
            return;
        }

        self.project.timeline_mut().swap(idx, target_idx);
        self.mark_project_dirty();
        self.selected_project_item = Some(ProjectSelection::Step(target_idx));
        self.status = format!("moved step {} to {}", idx + 1, target_idx + 1);
    }

    fn delete_export(&mut self, idx: usize) {
        if idx >= self.project.exports().len() {
            return;
        }

        let export_id = self.project.exports_mut().remove(idx);
        self.mark_project_dirty();
        if self.selected_export.as_deref() == Some(export_id.as_str()) {
            self.selected_export = self.project.exports().last().cloned();
        }
        self.selected_project_item = None;
        self.status = format!("removed export {export_id}");
    }

    fn add_export(&mut self, asset_id: &str) {
        if self.project.exports().iter().any(|id| id == asset_id) {
            return;
        }

        self.project.exports_mut().push(asset_id.to_string());
        self.mark_project_dirty();
        self.selected_export = Some(asset_id.to_string());
        self.selected_project_item = self
            .project
            .exports()
            .len()
            .checked_sub(1)
            .map(ProjectSelection::Export);
        self.status = format!("exporting {asset_id}");
    }

    /// Executes the project synchronously and applies the result. Production
    /// runs go through the host's background worker; this blocking path exists
    /// so tests can exercise execution without a host.
    #[cfg(test)]
    fn run_project(&mut self) {
        self.pending_run = false;
        let start = std::time::Instant::now();
        let result = self
            .project
            .run(&mut volumetric::Environment::new())
            .map_err(|err| err.to_string());
        self.apply_run_result(result, start.elapsed().as_millis());
    }

    fn set_render_mode(&mut self, mode: PreviewRenderMode) {
        self.render_mode = mode;
        self.status = format!("preview mode: {}", mode.full_label());
    }

    fn set_preview_resolution(&mut self, resolution: usize) {
        self.preview_resolution = resolution;
        self.status = format!("preview resolution: {resolution}^3");
    }

    /// Folds trigger/dismiss/pick events for the three viewport pickers
    /// (render mode, resolution, camera scheme) into app state. Returns true
    /// when the event belonged to one of them.
    fn handle_view_select(&mut self, event: &UiEvent) -> bool {
        for key in [
            MODE_SELECT_KEY,
            RESOLUTION_SELECT_KEY,
            CAMERA_SELECT_KEY,
            // Not value pickers (controls live inside), but the trigger and
            // dismiss-scrim routes follow the same shape; Pick never fires.
            SSAO_SETTINGS_KEY,
            REMOTE_SETTINGS_KEY,
            CACHE_SETTINGS_KEY,
        ] {
            let Some(action) = select::classify_event(event, key) else {
                continue;
            };
            match action {
                SelectAction::Toggle => {
                    self.open_select = if self.open_select.as_deref() == Some(key) {
                        None
                    } else {
                        Some(key.to_string())
                    };
                    // At most one popover layer at a time (see menubar above).
                    self.open_menu = None;
                }
                SelectAction::Dismiss => self.open_select = None,
                SelectAction::Pick(value) => {
                    self.open_select = None;
                    match key {
                        MODE_SELECT_KEY => {
                            if let Some(mode) = PreviewRenderMode::from_route_name(&value) {
                                self.set_render_mode(mode);
                            }
                        }
                        RESOLUTION_SELECT_KEY => {
                            if let Ok(resolution) = value.parse() {
                                self.set_preview_resolution(resolution);
                            }
                        }
                        CAMERA_SELECT_KEY => {
                            if let Some(scheme) = camera_control_scheme_from_route(&value) {
                                self.set_camera_control_scheme(scheme);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                // Future SelectAction variants (non-exhaustive enum): ignore.
                _ => {}
            }
            return true;
        }
        false
    }

    fn set_camera_control_scheme(&mut self, scheme: CameraControlScheme) {
        self.camera_control_scheme = scheme;
        self.status = format!("camera controls: {}", scheme.name());
    }
}

impl App for VolumetricUiV2 {
    fn build(&self, _cx: &BuildCx) -> El {
        shell(self)
    }

    fn before_build(&mut self) {
        self.sync_step_edit();
    }

    fn selection(&self) -> Selection {
        self.selection.clone()
    }

    fn on_event(&mut self, event: UiEvent, _cx: &EventCx) {
        // Escape closes the topmost layer: the Add or export modal first,
        // then any open menu/picker popover (the popover contract: the
        // scrim handles outside clicks, the app handles Escape).
        if matches!(event.kind, UiEventKind::Escape) {
            if self.add_modal.take().is_some() {
                return;
            }
            if self.export_dialog.take().is_some() {
                return;
            }
            if self.open_menu.take().is_some() | self.open_select.take().is_some() {
                return;
            }
        }

        // Panel divider drags (pointer down/drag/up + arrow keys on the
        // focused handle) — not clicks, so they run before the gate below.
        if event.route() == Some(PANEL_RESIZE_KEY) {
            resize_handle::apply_event_fixed(
                &mut self.panel_width,
                &mut self.panel_drag,
                &event,
                PANEL_RESIZE_KEY,
                Axis::Row,
                resize_handle::Side::End,
                PANEL_WIDTH_MIN,
                PANEL_WIDTH_MAX,
            );
            return;
        }

        // Controlled text editing for config fields runs first: text, key, and
        // selection events aren't clicks and would be dropped by the gate below.
        if let Some(field_name) = event
            .target_key()
            .and_then(|key| key.strip_prefix(CONFIG_FIELD_PREFIX))
            .map(str::to_string)
        {
            if let Some(mut edit) = self.step_edit.take() {
                let key = format!("{CONFIG_FIELD_PREFIX}{field_name}");
                let mut changed = false;
                if let Some(config) = edit.config.as_mut() {
                    if let Some(buffer) = config.buffers.get_mut(&field_name) {
                        changed =
                            text_input::apply_event(buffer, &mut self.selection, &event, &key);
                    }
                }
                if changed {
                    if let Some(config) = edit.config.as_ref() {
                        self.commit_config_buffer(edit.step_idx, config, &field_name);
                    }
                }
                self.step_edit = Some(edit);
                return;
            }
        }
        // Controlled text editing for VecF64 component fields.
        if let Some(rest) = event
            .target_key()
            .and_then(|key| key.strip_prefix(VEC_INPUT_PREFIX))
            .map(str::to_string)
        {
            if let Some(mut edit) = self.step_edit.take() {
                let key = format!("{VEC_INPUT_PREFIX}{rest}");
                let mut commit = None;
                if let Some((idx_str, component_str)) = rest.split_once(':')
                    && let (Ok(input_idx), Ok(component)) =
                        (idx_str.parse::<usize>(), component_str.parse::<usize>())
                    && let Some(vec_form) = edit.vecs.iter_mut().find(|v| v.input_idx == input_idx)
                    && let Some(buffer) = vec_form.buffers.get_mut(component)
                    && text_input::apply_event(buffer, &mut self.selection, &event, &key)
                {
                    commit = Some((input_idx, component, buffer.clone()));
                }
                if let Some((input_idx, component, buffer)) = commit {
                    self.commit_vec_component(edit.step_idx, input_idx, component, &buffer);
                }
                self.step_edit = Some(edit);
                return;
            }
        }
        // Controlled editing for the output-name buffer (committed by Rename).
        if event.target_key() == Some(OUTPUT_NAME_KEY) {
            if let Some(mut edit) = self.step_edit.take() {
                text_input::apply_event(
                    &mut edit.output_name,
                    &mut self.selection,
                    &event,
                    OUTPUT_NAME_KEY,
                );
                self.step_edit = Some(edit);
                return;
            }
        }
        // Controlled editing for the Lua source area.
        if event.target_key() == Some(LUA_SOURCE_KEY) {
            if let Some(mut edit) = self.step_edit.take() {
                let mut changed = false;
                if let Some(lua) = edit.lua.as_mut() {
                    changed = text_area::apply_event(
                        &mut lua.source,
                        &mut self.selection,
                        &event,
                        LUA_SOURCE_KEY,
                    );
                }
                if changed {
                    if let Some(lua) = edit.lua.as_ref() {
                        self.write_lua_source(edit.step_idx, lua);
                    }
                }
                self.step_edit = Some(edit);
                return;
            }
        }
        // Controlled editing for the Add modal's search buffer.
        if event.target_key() == Some(ADD_SEARCH_KEY) {
            if let Some(modal) = self.add_modal.as_mut() {
                text_input::apply_event(
                    &mut modal.query,
                    &mut self.selection,
                    &event,
                    ADD_SEARCH_KEY,
                );
            }
            return;
        }
        // Controlled editing for the remote daemon address buffer (takes
        // effect the next time the Remote toggle is switched on).
        if event.target_key() == Some(REMOTE_ADDRESS_KEY) {
            text_input::apply_event(
                &mut self.remote_address,
                &mut self.selection,
                &event,
                REMOTE_ADDRESS_KEY,
            );
            return;
        }
        // Controlled editing for the export modal's scale factor buffer
        // (applied at confirm; a half-typed value just disables Export).
        if event.target_key() == Some(EXPORT_SCALE_KEY) {
            if let Some(export) = self.export_dialog.as_mut() {
                text_input::apply_event(
                    &mut export.scale_text,
                    &mut self.selection,
                    &event,
                    EXPORT_SCALE_KEY,
                );
            }
            return;
        }

        // Track focus/selection moving to another widget (e.g. clicking away).
        if let Some(selection) = event.selection.clone() {
            self.selection = selection;
        }

        // Menubar trigger toggles and dismiss-scrim clicks. At most one
        // popover layer at a time: opening a menu closes any picker.
        if menubar::apply_event(&mut self.open_menu, &event, MENUBAR_KEY) {
            self.open_select = None;
            return;
        }

        // Viewport value pickers share one open slot; picks route to state.
        if self.handle_view_select(&event) {
            return;
        }

        if event.is_click_or_activate(NEW_PROJECT_KEY) {
            self.project = Project::new();
            self.selected_export = None;
            self.selected_project_item = None;
            self.clear_runtime_assets();
            self.open_menu = None;
            self.project_path = None;
            self.baked_on_disk = false;
            self.status = "new project".to_string();
            return;
        }

        if event.is_click_or_activate(OPEN_PROJECT_KEY) {
            self.pending_file_action = Some(FileAction::OpenProject);
            self.open_menu = None;
            return;
        }

        if event.is_click_or_activate(SAVE_PROJECT_KEY) {
            // Re-save in place when the path is known; first save asks. Both
            // go through the host's file worker — disk writes don't belong
            // on the UI thread any more than dialogs do.
            self.pending_file_action = Some(match self.project_path.clone() {
                Some(path) => FileAction::SaveProjectTo(path),
                None => FileAction::SaveProject,
            });
            self.open_menu = None;
            return;
        }

        if event.is_click_or_activate(SAVE_PROJECT_AS_KEY) {
            self.pending_file_action = Some(FileAction::SaveProject);
            self.open_menu = None;
            return;
        }

        if event.is_click_or_activate(SAVE_BUILT_COPY_KEY) {
            self.pending_file_action = Some(FileAction::SaveBuiltCopy);
            self.open_menu = None;
            return;
        }

        if event.is_click_or_activate(IMPORT_WASM_KEY) {
            self.pending_file_action = Some(FileAction::ImportWasm);
            self.open_menu = None;
            self.add_modal = None;
            return;
        }

        if event.is_click_or_activate(IMPORT_STL_KEY) {
            self.pending_file_action = Some(FileAction::ImportStl);
            self.open_menu = None;
            self.add_modal = None;
            return;
        }

        if event.is_click_or_activate(IMPORT_IMAGE_KEY) {
            self.pending_file_action = Some(FileAction::ImportImage);
            self.open_menu = None;
            self.add_modal = None;
            return;
        }

        if event.is_click_or_activate(ADD_OPEN_KEY) {
            self.add_modal = Some(AddModalState::default());
            // Focus lands in the search field so typing filters immediately.
            self.selection = Selection::caret(ADD_SEARCH_KEY, 0);
            self.open_menu = None;
            self.open_select = None;
            return;
        }

        if event.is_click_or_activate(ADD_DISMISS_KEY) {
            self.add_modal = None;
            return;
        }

        if event.is_click_or_activate(RUN_PROJECT_KEY) {
            self.request_run();
            return;
        }

        if event.is_click_or_activate(CANCEL_RUN_KEY) {
            self.cancel_requested = true;
            self.status = "cancelling run".to_string();
            return;
        }

        if event.is_click_or_activate(CANCEL_MESH_KEY) {
            self.mesh_cancel_requested = true;
            self.status = "cancelling mesh builds".to_string();
            return;
        }

        if event.is_click_or_activate(REMESH_KEY) {
            self.remesh_requested = true;
            self.status = "remesh queued".to_string();
            return;
        }

        if event.is_click_or_activate(TOGGLE_AUTO_REMESH_KEY) {
            self.auto_remesh = !self.auto_remesh;
            self.status = if self.auto_remesh {
                "auto-remesh on".to_string()
            } else {
                "auto-remesh off (changes wait for Remesh)".to_string()
            };
            return;
        }

        if event.is_click_or_activate(TOGGLE_AUTO_REBUILD_KEY) {
            self.auto_rebuild = !self.auto_rebuild;
            self.status = if self.auto_rebuild {
                "auto-rebuild on".to_string()
            } else {
                "auto-rebuild off".to_string()
            };
            return;
        }

        if event.is_click_or_activate(TOGGLE_REMOTE_BUILD_KEY) {
            self.remote_build = !self.remote_build;
            let address = self.remote_address.trim().to_string();
            self.executor_request = Some(if self.remote_build {
                ExecutorChoice::Remote(address.clone())
            } else {
                ExecutorChoice::Local
            });
            // Work in flight sits on the worker being replaced and will
            // never report back: write it off now, and queue a fresh run +
            // remesh so the new executor rebuilds everything.
            self.cancel_requested = true;
            self.mesh_cancel_requested = true;
            self.remesh_requested = true;
            self.pending_run = true;
            self.status = if self.remote_build {
                format!("remote build on {address}")
            } else {
                "remote build off (building locally)".to_string()
            };
            return;
        }

        if event.is_click_or_activate(TOGGLE_GRID_KEY) {
            self.show_grid = !self.show_grid;
            self.status = if self.show_grid {
                "grid enabled".to_string()
            } else {
                "grid hidden".to_string()
            };
            return;
        }

        if event.is_click_or_activate(TOGGLE_BOUNDS_KEY) {
            self.show_bounds = !self.show_bounds;
            self.status = if self.show_bounds {
                "bounding boxes shown".to_string()
            } else {
                "bounding boxes hidden".to_string()
            };
            return;
        }

        if event.is_click_or_activate(TOGGLE_SSAO_KEY) {
            self.ssao = !self.ssao;
            self.status = if self.ssao {
                "SSAO enabled".to_string()
            } else {
                "SSAO disabled".to_string()
            };
            return;
        }

        if event.is_click_or_activate(FRAME_PREVIEW_KEY) {
            self.pending_camera_command = Some(ViewportCameraCommand::FramePreview);
            self.status = "framing preview".to_string();
            return;
        }

        if event.is_click_or_activate(RESET_CAMERA_KEY) {
            self.pending_camera_command = Some(ViewportCameraCommand::Reset);
            self.status = "camera reset".to_string();
            return;
        }

        if !matches!(event.kind, UiEventKind::Click | UiEventKind::Activate) {
            return;
        }

        let Some(route) = event.route() else {
            return;
        };

        if let Some(accordion::AccordionAction::Toggle(section)) =
            accordion::classify_event(&event, PIPELINE_KEY)
        {
            if !self.pipeline_open.remove(section) {
                self.pipeline_open.insert(section.to_string());
            }
        } else if let Some(name) = route.strip_prefix(ADD_MODEL_PREFIX) {
            self.add_model(name);
            self.add_modal = None;
        } else if let Some(name) = route.strip_prefix(ADD_OPERATOR_PREFIX) {
            self.add_operator(name);
            self.add_modal = None;
        } else if let Some(name) = route.strip_prefix(RAIL_ADD_PREFIX) {
            // Rail tiles carry the catalog kind: models import, operators
            // insert a step.
            match self.catalog.get(name).map(|entry| entry.kind) {
                Some(volumetric_assets::AssetCategory::Model) => self.add_model(name),
                Some(volumetric_assets::AssetCategory::Operator) => self.add_operator(name),
                None => self.status = format!("missing bundled module {name}"),
            }
        } else if let Some(idx) = parse_index_route(route, SELECT_IMPORT_PREFIX) {
            self.selected_project_item = Some(ProjectSelection::Import(idx));
            self.status = format!("selected import {}", idx + 1);
        } else if let Some(idx) = parse_index_route(route, UPGRADE_IMPORT_PREFIX) {
            self.upgrade_import(idx);
        } else if let Some(idx) = parse_index_route(route, DELETE_IMPORT_PREFIX) {
            self.delete_import(idx);
        } else if let Some(idx) = parse_index_route(route, SELECT_STEP_PREFIX) {
            self.selected_project_item = Some(ProjectSelection::Step(idx));
            self.status = format!("selected step {}", idx + 1);
        } else if let Some(idx) = parse_index_route(route, DELETE_STEP_PREFIX) {
            self.delete_step(idx);
        } else if let Some(idx) = parse_index_route(route, RENAME_OUTPUT_PREFIX) {
            self.rename_step_output(idx);
        } else if let Some(idx) = parse_index_route(route, RESET_LUA_PREFIX) {
            self.reset_lua_source(idx);
        } else if let Some(idx) = parse_index_route(route, MOVE_STEP_UP_PREFIX) {
            self.move_step(idx, -1);
        } else if let Some(idx) = parse_index_route(route, MOVE_STEP_DOWN_PREFIX) {
            self.move_step(idx, 1);
        } else if let Some((step_idx, input_idx, asset_id)) =
            parse_step_model_route(route, SET_STEP_MODEL_PREFIX)
        {
            self.set_step_model_input(step_idx, input_idx, asset_id);
        } else if let Some(idx) = parse_index_route(route, SELECT_EXPORT_PREFIX) {
            if let Some(export_id) = self.project.exports().get(idx) {
                self.selected_export = Some(export_id.clone());
            }
            self.selected_project_item = Some(ProjectSelection::Export(idx));
            self.status = format!("selected export {}", idx + 1);
        } else if let Some(idx) = parse_index_route(route, DELETE_EXPORT_PREFIX) {
            self.delete_export(idx);
        } else if let Some(asset_id) = route.strip_prefix(ADD_EXPORT_PREFIX) {
            self.add_export(asset_id);
        } else if let Some(asset_id) = route.strip_prefix(SELECT_RUNTIME_ASSET_PREFIX) {
            self.selected_export = Some(asset_id.to_string());
            // Fold runtime selection into the pipeline selection so the viewport
            // follows it: a runtime asset is the output of its matching export.
            self.selected_project_item = self
                .project
                .exports()
                .iter()
                .position(|export| export == asset_id)
                .map(ProjectSelection::Export);
            self.status = format!("selected runtime asset {asset_id}");
        } else if let Some(asset_id) = route.strip_prefix(TOGGLE_PIN_PREFIX) {
            self.toggle_pin(asset_id);
        } else if let Some(rest) = route.strip_prefix(OUTPUT_SETTINGS_PREFIX) {
            // Per-output settings popover: trigger toggles, scrim dismisses.
            if rest.ends_with(":dismiss") {
                self.open_select = None;
            } else {
                self.open_select = if self.open_select.as_deref() == Some(route) {
                    None
                } else {
                    Some(route.to_string())
                };
                self.open_menu = None;
            }
        } else if let Some(rest) = route.strip_prefix(OUTPUT_MODE_PREFIX) {
            if let Some((id, mode_name)) = rest.rsplit_once(':')
                && let Some(mode) = PreviewRenderMode::from_route_name(mode_name)
            {
                self.set_output_mode(id, mode);
            }
        } else if let Some(rest) = route.strip_prefix(OUTPUT_RESOLUTION_PREFIX) {
            if let Some((id, res)) = rest.rsplit_once(':')
                && let Ok(resolution) = res.parse()
            {
                self.set_output_resolution(id, resolution);
            }
        } else if let Some(id) = route.strip_prefix(OUTPUT_DEFAULTS_PREFIX) {
            self.clear_output_override(id);
        } else if let Some(id) = route.strip_prefix(OUTPUT_WIREFRAME_PREFIX) {
            self.toggle_output_wireframe(id);
        } else if route == format!("{LIGHTBOX_KEY}:dismiss") {
            self.lightbox = None;
        } else if let Some(id) = route.strip_prefix(OUTPUT_INSPECT_PREFIX) {
            self.open_lightbox(id);
        } else if let Some(id) = route.strip_prefix(OUTPUT_FEA_DEFORMED_PREFIX) {
            self.toggle_output_fea_deformed(id);
        } else if let Some(rest) = route.strip_prefix(OUTPUT_FEA_EXAG_PREFIX) {
            if let Some((id, direction)) = rest.rsplit_once(':') {
                self.adjust_output_fea_exaggeration(id, direction == "up");
            }
        } else if let Some(rest) = route.strip_prefix(OUTPUT_CHANNEL_PREFIX) {
            // `{id}:none` or `{id}:{channel}` — value split off the right so
            // asset ids with colons still resolve.
            if let Some((id, channel)) = rest.rsplit_once(':') {
                let channel = (channel != "none").then(|| channel.to_string());
                self.set_output_channel(id, channel);
            }
        } else if let Some(rest) = route.strip_prefix(LIGHTBOX_SLICE_PREFIX) {
            self.adjust_lightbox_slice(rest);
        } else if let Some(rest) = route.strip_prefix(OUTPUT_FEA_FIELD_PREFIX) {
            // `{id}:none` or `{id}:{node|element}:{name}` — split the value
            // off the right so asset ids with colons still resolve.
            if let Some((id, "none")) = rest
                .rsplit_once(':')
                .and_then(|(head, tail)| (tail == "none").then_some((head, tail)))
            {
                self.set_output_fea_field(id, None);
            } else if let Some((head, name)) = rest.rsplit_once(':')
                && let Some((id, container)) = head.rsplit_once(':')
                && matches!(container, "node" | "element")
            {
                self.set_output_fea_field(id, Some(format!("{container}:{name}")));
            }
        } else if let Some(rest) = route.strip_prefix(SSAO_ADJUST_PREFIX) {
            if let Some((field, direction)) = rest.split_once(':') {
                self.adjust_ssao(field, direction == "up");
            }
        } else if let Some(direction) = route.strip_prefix(CACHE_ADJUST_PREFIX) {
            self.adjust_cache_budget(direction == "up");
        } else if let Some(rest) = route.strip_prefix(OUTPUT_ASN2_PREFIX) {
            if let Some((rest, direction)) = rest.rsplit_once(':')
                && let Some((id, field)) = rest.rsplit_once(':')
            {
                self.adjust_output_asn2(id, field, direction == "up");
            }
        } else if let Some(id) = route.strip_prefix(EXPORT_MESH_PREFIX) {
            self.open_export_dialog(id);
        } else if route == EXPORT_DISMISS_KEY {
            self.export_dialog = None;
        } else if route == EXPORT_CONFIRM_KEY {
            self.confirm_export();
        } else if let Some(unit) = route.strip_prefix(EXPORT_UNIT_PREFIX) {
            // Presets set the model-unit → millimetre factor (STL consumers
            // read the file as mm).
            let factor = match unit {
                "mm" => Some("1"),
                "cm" => Some("10"),
                "m" => Some("1000"),
                "in" => Some("25.4"),
                _ => None,
            };
            if let (Some(factor), Some(export)) = (factor, self.export_dialog.as_mut()) {
                export.scale_text = factor.to_string();
            }
        } else if let Some(id) = route.strip_prefix(EXPORT_WASM_PREFIX) {
            self.pending_file_action = Some(FileAction::ExportWasm(id.to_string()));
            self.open_select = None;
        } else if let Some(field_name) = route.strip_prefix(CONFIG_BOOL_PREFIX) {
            self.toggle_config_bool(field_name);
        } else if let Some(rest) = route.strip_prefix(CONFIG_ENUM_PREFIX) {
            if let Some((field_name, value)) = rest.split_once(':') {
                self.set_config_buffer(field_name, value.to_string());
            }
        }
    }
}

pub fn shell(app: &VolumetricUiV2) -> El {
    let main = column([
        top_bar(app),
        row([
            add_rail(app),
            viewport_pane(app),
            resize_handle(PANEL_RESIZE_KEY, Axis::Row),
            project_panel(app),
        ])
        // Gap keeps the handle's expanded grab band off the viewport's
        // hit target (HitOverflowCollision lint).
        .gap(tokens::SPACE_1)
        .width(Size::Fill(1.0))
        .height(Size::Fill(1.0)),
    ])
    .fill_size();

    overlays(
        main,
        [
            menu_layer(app),
            select_layer(app),
            lightbox_layer(app),
            export_layer(app),
            add_layer(app),
        ],
    )
}

/// The single strip of application chrome above the viewport: menubar
/// (File / Add), then run controls and status on the right.
fn top_bar(app: &VolumetricUiV2) -> El {
    let open = app.open_menu.as_deref();
    let mut items = vec![
        icon("layout-dashboard").icon_size(tokens::ICON_SM).muted(),
        text("Volumetric").label().semibold().key("brand-title"),
        menubar([menubar_trigger(
            MENUBAR_KEY,
            "file",
            "File",
            open == Some("file"),
        )]),
        spacer(),
        run_status_chip(app),
        preview_status_chip(app),
    ];
    items.extend(mesh_control(app));
    items.extend([
        toggle_chip("Auto mesh", app.auto_remesh, TOGGLE_AUTO_REMESH_KEY),
        toggle_chip("Auto run", app.auto_rebuild, TOGGLE_AUTO_REBUILD_KEY),
    ]);
    items.extend([
        toggle_chip("Remote", app.remote_build, TOGGLE_REMOTE_BUILD_KEY),
        icon_button("chevron-down")
            .ghost()
            .xsmall()
            .tooltip("Remote build settings")
            .key(REMOTE_SETTINGS_KEY),
        icon_button("download")
            .ghost()
            .xsmall()
            .tooltip("Build cache")
            .key(CACHE_SETTINGS_KEY),
    ]);
    items.push(run_control(app));
    toolbar(items)
        .gap(tokens::SPACE_2)
        .padding(Sides::xy(tokens::SPACE_3, tokens::SPACE_2))
}

/// The open menubar menu, rendered as a root overlay layer.
fn menu_layer(app: &VolumetricUiV2) -> Option<El> {
    match app.open_menu.as_deref()? {
        "file" => Some(menubar_menu(
            MENUBAR_KEY,
            "file",
            [
                menubar_item_with_icon("plus", "New Project").key(NEW_PROJECT_KEY),
                menubar_separator(),
                menubar_item_with_icon("folder", "Open Project…").key(OPEN_PROJECT_KEY),
                menubar_item_with_icon("download", "Save Project").key(SAVE_PROJECT_KEY),
                menubar_item_with_icon("download", "Save Project As…").key(SAVE_PROJECT_AS_KEY),
                menubar_item_with_icon("download", "Save Built Copy…").key(SAVE_BUILT_COPY_KEY),
            ],
        )),
        _ => None,
    }
}

/// The left rail: the Add-catalog affordance on top, then one-click tiles
/// for recently added modules (most recent first, persisted in settings).
fn add_rail(app: &VolumetricUiV2) -> El {
    let mut items = vec![
        icon_button("plus")
            .tooltip("Add — browse the module catalog")
            .key(ADD_OPEN_KEY),
        divider(),
    ];
    for name in &app.recent_adds {
        // A recent that no longer exists in the catalog (renamed module,
        // stale settings) just doesn't render.
        let Some(entry) = app.catalog.get(name) else {
            continue;
        };
        items.push(
            icon_button(catalog_icon_source(entry))
                .ghost()
                .tooltip(entry.display_name())
                .key(format!("{RAIL_ADD_PREFIX}{name}")),
        );
    }
    column(items)
        .gap(tokens::SPACE_2)
        .padding(Sides::xy(tokens::SPACE_2, tokens::SPACE_2))
        .align(Align::Center)
        .width(Size::Hug)
        .height(Size::Fill(1.0))
}

/// Fixed presentation order for the known categories in the browse view;
/// categories declared by modules but missing here append after, and
/// entries without metadata yet group under a trailing "Scanning" header.
const CATEGORY_ORDER: [&str; 10] = [
    "Primitives",
    "Combine",
    "Transforms",
    "Construction",
    "Lattice",
    "Mesh",
    "FEA",
    "Fabrication",
    "Scripting",
    "Import",
];

/// The declared category of an entry, if scanned and non-empty.
fn entry_category(entry: &catalog::CatalogEntry) -> Option<&str> {
    entry
        .ready()
        .map(|metadata| metadata.category.as_str())
        .filter(|category| !category.is_empty())
}

/// The click route for a catalog row. Import-shaped operators open their
/// file dialog (staging them bare would only fail the next run); everything
/// else adds directly by kind.
fn catalog_row_key(entry: &catalog::CatalogEntry) -> String {
    match entry.name.as_str() {
        "stl_import_operator" => IMPORT_STL_KEY.to_string(),
        "image_model_operator" => IMPORT_IMAGE_KEY.to_string(),
        _ => match entry.kind {
            volumetric_assets::AssetCategory::Model => {
                format!("{ADD_MODEL_PREFIX}{}", entry.name)
            }
            volumetric_assets::AssetCategory::Operator => {
                format!("{ADD_OPERATOR_PREFIX}{}", entry.name)
            }
        },
    }
}

/// The catalog entry's one-line status: its declared description once
/// known, a scan/failure note until then.
fn catalog_description(entry: &catalog::CatalogEntry) -> String {
    match &entry.metadata {
        catalog::CatalogMetadata::Ready(metadata) => metadata.description.clone(),
        catalog::CatalogMetadata::Pending => "reading module metadata…".to_string(),
        catalog::CatalogMetadata::Failed(_) => "module metadata unavailable".to_string(),
    }
}

/// One Add-modal search row: icon, display name, and the declared
/// description (search results stay rows — the description column is what
/// disambiguates near-matches).
fn catalog_row(entry: &catalog::CatalogEntry) -> El {
    command_item([
        command_icon(catalog_icon_source(entry)),
        command_label(entry.display_name()).width(Size::Fixed(170.0)),
        text(catalog_description(entry))
            .caption()
            .muted()
            .ellipsis()
            .width(Size::Fill(1.0)),
    ])
    .key(catalog_row_key(entry))
}

/// Browse-grid columns; 640px modal ÷ 4 leaves room for the longest
/// display names ("Heightmap Extrude") before the ellipsis bites.
const ADD_GRID_COLS: usize = 4;
const ADD_CARD_HEIGHT: f32 = 72.0;

/// The browse-card recipe shared by catalog entries and the Model-WASM
/// action: a column-axis [`command_item`] (keeping its focus, hover, and
/// the grid's 2D arrow-nav) with the glyph over a centered label; the
/// description rides the tooltip — cards have no room for it inline.
fn add_card(icon_source: IconSource, label: &str, tooltip: &str, key: String) -> El {
    command_item([
        icon(icon_source)
            .icon_size(tokens::ICON_LG)
            .color(tokens::FOREGROUND),
        text(label)
            .caption()
            .center_text()
            .ellipsis()
            .width(Size::Fill(1.0)),
    ])
    .axis(Axis::Column)
    .align(Align::Center)
    .justify(Justify::Center)
    .gap(tokens::SPACE_2)
    .height(Size::Fixed(ADD_CARD_HEIGHT))
    .padding(Sides::all(tokens::SPACE_2))
    .tooltip(tooltip)
    .key(key)
}

/// One Add-modal browse card for a catalog entry.
fn catalog_card(entry: &catalog::CatalogEntry) -> El {
    add_card(
        catalog_icon_source(entry),
        entry.display_name(),
        &catalog_description(entry),
        catalog_row_key(entry),
    )
}

const IMPORT_WASM_LABEL: &str = "Model WASM…";
const IMPORT_WASM_BLURB: &str = "Import a compiled model module from disk.";

/// The import action for external module files (not a catalog entry: it
/// imports an arbitrary compiled model from disk) — search-row shape.
fn import_wasm_row() -> El {
    command_item([
        command_icon("file-text"),
        command_label(IMPORT_WASM_LABEL).width(Size::Fixed(170.0)),
        text(IMPORT_WASM_BLURB)
            .caption()
            .muted()
            .ellipsis()
            .width(Size::Fill(1.0)),
    ])
    .key(IMPORT_WASM_KEY)
}

/// [`import_wasm_row`]'s browse-grid shape.
fn import_wasm_card() -> El {
    add_card(
        "file-text".into_icon_source(),
        IMPORT_WASM_LABEL,
        IMPORT_WASM_BLURB,
        IMPORT_WASM_KEY.to_string(),
    )
}

/// Browse view: a card grid per declared category in [`CATEGORY_ORDER`],
/// unknown declared categories after, unscanned entries last.
fn add_browse_rows(app: &VolumetricUiV2) -> Vec<El> {
    let entries = app.catalog.entries();
    let mut categories: Vec<&str> = CATEGORY_ORDER.to_vec();
    for entry in entries {
        if let Some(category) = entry_category(entry)
            && !categories.contains(&category)
        {
            categories.push(category);
        }
    }

    let mut rows = Vec::new();
    for category in categories {
        let group: Vec<&catalog::CatalogEntry> = entries
            .iter()
            .filter(|entry| entry_category(entry) == Some(category))
            .collect();
        // The Import group always exists — the Model WASM action lives
        // there even before its sibling entries are scanned.
        if group.is_empty() && category != "Import" {
            continue;
        }
        rows.push(menubar_label(category));
        let mut cards: Vec<El> = group.into_iter().map(catalog_card).collect();
        if category == "Import" {
            cards.push(import_wasm_card());
        }
        rows.push(grid(ADD_GRID_COLS, tokens::SPACE_2, cards));
    }

    let unscanned: Vec<&catalog::CatalogEntry> = entries
        .iter()
        .filter(|entry| entry_category(entry).is_none())
        .collect();
    if !unscanned.is_empty() {
        rows.push(menubar_label("Scanning"));
        rows.push(grid(
            ADD_GRID_COLS,
            tokens::SPACE_2,
            unscanned.into_iter().map(catalog_card),
        ));
    }
    rows
}

/// Rank an entry against the search query: lower is better, `None` filters
/// it out. Display-name prefix beats display-name contains beats module
/// name, category, then description hits.
fn add_search_score(entry: &catalog::CatalogEntry, query: &str) -> Option<u32> {
    let display = entry.display_name().to_lowercase();
    if display.starts_with(query) {
        return Some(0);
    }
    if display.contains(query) {
        return Some(1);
    }
    if entry.name.contains(query) {
        return Some(2);
    }
    let metadata = entry.ready()?;
    if metadata.category.to_lowercase().contains(query) {
        return Some(3);
    }
    if metadata.description.to_lowercase().contains(query) {
        return Some(4);
    }
    None
}

/// Search view: matching entries ranked by [`add_search_score`], stable by
/// catalog order within a rank.
fn add_search_rows(app: &VolumetricUiV2, query: &str) -> Vec<El> {
    let query = query.to_lowercase();
    let mut scored: Vec<(u32, usize, &catalog::CatalogEntry)> = app
        .catalog
        .entries()
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| add_search_score(entry, &query).map(|s| (s, idx, entry)))
        .collect();
    scored.sort_by_key(|(score, idx, _)| (*score, *idx));
    let mut rows: Vec<El> = scored
        .into_iter()
        .map(|(_, _, entry)| catalog_row(entry))
        .collect();
    if "model wasm import module".contains(&query) {
        rows.push(import_wasm_row());
    }
    if rows.is_empty() {
        rows.push(text("no matches").label().muted());
    }
    rows
}

/// The Add-catalog modal, rendered as a root overlay layer: a search box
/// over the cataloged modules — an empty query browses by category, typing
/// ranks matches. Rows one-click add (or open the matching import dialog)
/// and close the modal.
fn add_layer(app: &VolumetricUiV2) -> Option<El> {
    let modal = app.add_modal.as_ref()?;
    let query = modal.query.trim();
    let rows = if query.is_empty() {
        add_browse_rows(app)
    } else {
        add_search_rows(app, query)
    };
    let body = [
        text_input_with(
            ADD_SEARCH_KEY,
            &modal.query,
            &app.selection,
            TextInputOpts::default().placeholder("Search models and operators…"),
        )
        .width(Size::Fill(1.0)),
        column(rows)
            .gap(tokens::SPACE_1)
            .width(Size::Fill(1.0))
            .height(Size::Fixed(440.0))
            .scrollable()
            .scrollbar()
            .clip(),
    ];
    Some(overlay([
        scrim(ADD_DISMISS_KEY),
        modal_panel("Add", body)
            .width(Size::Fixed(640.0))
            .block_pointer(),
    ]))
}

/// The open viewport picker menu (or per-output settings popover), rendered
/// as a root overlay layer.
fn select_layer(app: &VolumetricUiV2) -> Option<El> {
    if let Some(id) = app
        .open_select
        .as_deref()?
        .strip_prefix(OUTPUT_SETTINGS_PREFIX)
    {
        return Some(output_settings_popover(app, id));
    }
    match app.open_select.as_deref()? {
        MODE_SELECT_KEY => Some(select_menu(
            MODE_SELECT_KEY,
            PreviewRenderMode::ALL
                .into_iter()
                .map(|mode| (mode.route_name(), mode.full_label())),
        )),
        RESOLUTION_SELECT_KEY => Some(select_menu(
            RESOLUTION_SELECT_KEY,
            PREVIEW_RESOLUTIONS
                .into_iter()
                .map(|resolution| (resolution.to_string(), format!("{resolution}^3 voxels"))),
        )),
        CAMERA_SELECT_KEY => Some(select_menu(
            CAMERA_SELECT_KEY,
            CameraControlScheme::ALL.iter().copied().map(|scheme| {
                (
                    camera_scheme_route_name(scheme),
                    camera_scheme_tooltip(scheme),
                )
            }),
        )),
        SSAO_SETTINGS_KEY => Some(ssao_settings_popover(app)),
        REMOTE_SETTINGS_KEY => Some(remote_settings_popover(app)),
        CACHE_SETTINGS_KEY => Some(cache_settings_popover(app)),
        _ => None,
    }
}

/// The 2D field inspection lightbox: a wide modal with the colormapped
/// raster (pixel-exact — no viewport lighting), a colorbar over the sampled
/// value range, and the analytics rows. More engineering statistics land in
/// `LightboxData::analytics` over time; this just displays them.
fn lightbox_layer(app: &VolumetricUiV2) -> Option<El> {
    let lightbox = app.lightbox.as_ref()?;
    let mut body: Vec<El> = Vec::new();
    // Slice controls render above the image and stay live while a new
    // sample is in flight (each change re-samples in the background).
    if let LightboxMode::Slice {
        axis,
        frac_percent,
        channel,
    } = &lightbox.mode
    {
        let axis_button = |a: usize, label: &str| {
            let button = button(label)
                .xsmall()
                .key(format!("{LIGHTBOX_SLICE_PREFIX}axis:{a}"));
            if *axis == a {
                button.primary()
            } else {
                button.secondary()
            }
        };
        let mut controls = vec![
            axis_button(0, "X"),
            axis_button(1, "Y"),
            axis_button(2, "Z"),
            icon_button("chevron-left")
                .ghost()
                .xsmall()
                .key(format!("{LIGHTBOX_SLICE_PREFIX}pos:down")),
            text(format!("{frac_percent}%"))
                .label()
                .text_align(TextAlign::Center)
                .width(Size::Fixed(48.0)),
            icon_button("chevron-right")
                .ghost()
                .xsmall()
                .key(format!("{LIGHTBOX_SLICE_PREFIX}pos:up")),
        ];
        let channels = app.output_channels(&lightbox.asset_id);
        for name in &channels {
            let button = button(name.clone())
                .xsmall()
                .key(format!("{LIGHTBOX_SLICE_PREFIX}channel:{name}"));
            controls.push(if name == channel {
                button.primary()
            } else {
                button.secondary()
            });
        }
        body.push(row(controls).gap(tokens::SPACE_1).align(Align::Center));
        body.push(divider());
    }
    match (&lightbox.data, &lightbox.texture) {
        (Some(data), Some(texture)) => {
            body.push(
                surface(texture.clone())
                    .surface_alpha(SurfaceAlpha::Opaque)
                    .surface_fit(ImageFit::Contain)
                    .width(Size::Fill(1.0))
                    .height(Size::Fixed(460.0))
                    .clip(),
            );
            if data.binary {
                body.push(text("occupancy mask").caption().muted());
            } else if let Some(colorbar) = &lightbox.colorbar {
                body.push(
                    row([
                        text(format!("{:.4}", data.value_min)).caption().muted(),
                        surface(colorbar.clone())
                            .surface_alpha(SurfaceAlpha::Opaque)
                            .surface_fit(ImageFit::Fill)
                            .height(Size::Fixed(12.0))
                            .width(Size::Fill(1.0))
                            .clip(),
                        text(format!("{:.4}", data.value_max)).caption().muted(),
                    ])
                    .gap(tokens::SPACE_2)
                    .align(Align::Center),
                );
            }
            body.push(divider());
            for (label, value) in &data.analytics {
                body.push(
                    row([
                        text(label).caption().muted().width(Size::Fixed(140.0)),
                        text(value).caption().width(Size::Fill(1.0)),
                    ])
                    .gap(tokens::SPACE_2),
                );
            }
        }
        _ => body.push(text("sampling…").label().muted()),
    }
    Some(overlay([
        scrim(format!("{LIGHTBOX_KEY}:dismiss")),
        modal_panel(lightbox.asset_id.clone(), body)
            .width(Size::Fixed(720.0))
            .block_pointer(),
    ]))
}

/// The mesh-export modal: an orbitable `chart3d` preview of the export
/// geometry over the export configuration (scale factor with unit presets)
/// and the Cancel/Export actions. 3MF joins the format row when it lands.
fn export_layer(app: &VolumetricUiV2) -> Option<El> {
    let export = app.export_dialog.as_ref()?;
    let scale = parse_export_scale(&export.scale_text);
    let mut body: Vec<El> = Vec::new();
    match &export.mesh {
        ExportPreviewMesh::Ready {
            handle,
            triangles,
            bounds,
        } => {
            let scene = SceneSpec::new()
                .mesh_with(
                    handle.clone(),
                    Material::matte(Color::srgb_u8(178, 186, 200)),
                )
                .grid(GridPlanes::XZ);
            body.push(
                chart3d(scene)
                    .key(EXPORT_SCENE_KEY)
                    .height(Size::Fixed(320.0)),
            );
            body.push(
                text("drag to orbit · shift-drag to pan · wheel to zoom")
                    .caption()
                    .muted(),
            );
            // Size readout follows the scale factor live, so the preset
            // buttons double as a sanity check on the exported dimensions.
            let factor = scale.unwrap_or(1.0);
            let (min, max) = *bounds;
            let scaled_min = (min.0 * factor, min.1 * factor, min.2 * factor);
            let scaled_max = (max.0 * factor, max.1 * factor, max.2 * factor);
            body.push(
                text(format!(
                    "{} triangles · {} mm",
                    format_count(*triangles),
                    format_dims(scaled_min, scaled_max),
                ))
                .caption()
                .muted(),
            );
        }
        ExportPreviewMesh::Pending => {
            body.push(
                text("copying preview mesh…")
                    .label()
                    .muted()
                    .height(Size::Fixed(320.0)),
            );
        }
        ExportPreviewMesh::Missing => {
            body.push(
                text("no cached mesh preview — view this output in a mesh render mode first")
                    .label()
                    .muted(),
            );
        }
    }
    body.push(divider());
    body.push(field_row("Format", text("STL · binary").label()));
    let unit_button = |unit: &str, label: &str| {
        button(label)
            .xsmall()
            .secondary()
            .key(format!("{EXPORT_UNIT_PREFIX}{unit}"))
    };
    body.push(field_row(
        "Scale",
        row([
            text_input(EXPORT_SCALE_KEY, &export.scale_text, &app.selection)
                .width(Size::Fixed(96.0)),
            unit_button("mm", "mm"),
            unit_button("cm", "cm"),
            unit_button("m", "m"),
            unit_button("in", "in"),
        ])
        .gap(tokens::SPACE_1)
        .align(Align::Center),
    ));
    body.push(
        text("Unit presets set the factor from model units to millimetres — STL is read as mm.")
            .caption()
            .muted(),
    );
    if scale.is_none() {
        body.push(form_message("Scale must be a positive number."));
    }
    body.push(divider());
    let can_export = scale.is_some() && matches!(export.mesh, ExportPreviewMesh::Ready { .. });
    let confirm = button("Export STL…").primary().key(EXPORT_CONFIRM_KEY);
    let confirm = if can_export {
        confirm
    } else {
        confirm.disabled()
    };
    body.push(
        row([
            spacer(),
            button("Cancel").secondary().key(EXPORT_DISMISS_KEY),
            confirm,
        ])
        .gap(tokens::SPACE_2)
        .align(Align::Center),
    );
    Some(overlay([
        scrim(EXPORT_DISMISS_KEY),
        modal_panel(format!("Export {}", export.asset_id), body)
            .width(Size::Fixed(560.0))
            .block_pointer(),
    ]))
}

/// Anchored popover with SSAO parameter steppers. Steppers keep it open;
/// outside click or Escape dismisses.
fn ssao_settings_popover(app: &VolumetricUiV2) -> El {
    let stepper = |field: &str, label: &str, value: String| {
        field_row(
            label,
            row([
                icon_button("chevron-left")
                    .ghost()
                    .xsmall()
                    .key(format!("{SSAO_ADJUST_PREFIX}{field}:down")),
                text(value)
                    .label()
                    .text_align(TextAlign::Center)
                    .width(Size::Fixed(56.0)),
                icon_button("chevron-right")
                    .ghost()
                    .xsmall()
                    .key(format!("{SSAO_ADJUST_PREFIX}{field}:up")),
            ])
            .gap(tokens::SPACE_1)
            .align(Align::Center),
        )
        .gap(tokens::SPACE_2)
    };
    popover(
        SSAO_SETTINGS_KEY,
        Anchor::below_key(SSAO_SETTINGS_KEY),
        popover_panel([column([
            text("SSAO").label().semibold(),
            stepper("radius", "Radius", format!("{:.3}", app.ssao_radius)),
            stepper("bias", "Bias", format!("{:.4}", app.ssao_bias)),
            stepper("strength", "Strength", format!("{:.2}", app.ssao_strength)),
        ])
        .gap(tokens::SPACE_2)
        .padding(tokens::SPACE_2)
        .width(Size::Fixed(240.0))]),
    )
}

/// Human-readable size for cache figures: MB below 1 GB, GB above.
fn format_mb(bytes: usize) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / f64::from(1 << 30))
    } else {
        format!("{:.0} MB", bytes as f64 / f64::from(1 << 20))
    }
}

/// Anchored popover with live build-cache stats and the budget stepper.
/// Stats come straight off the process-global cache each frame; the shown
/// budget is the live one, which can sit above the preference after a
/// large built copy was seeded.
fn cache_settings_popover(app: &VolumetricUiV2) -> El {
    let stats = volumetric::build_cache::global().stats();
    popover(
        CACHE_SETTINGS_KEY,
        Anchor::below_key(CACHE_SETTINGS_KEY),
        popover_panel([column([
            text("Build cache").label().semibold(),
            field_row(
                "Resident",
                text(format!("{} in {} steps", format_mb(stats.bytes), stats.entries)).label(),
            )
            .gap(tokens::SPACE_2),
            field_row(
                "Budget",
                row([
                    icon_button("chevron-left")
                        .ghost()
                        .xsmall()
                        .key(format!("{CACHE_ADJUST_PREFIX}down")),
                    text(format!(
                        "{}{}",
                        format_mb(app.cache_budget_bytes()),
                        if stats.budget > app.cache_budget_bytes() {
                            format!(" (now {})", format_mb(stats.budget))
                        } else {
                            String::new()
                        }
                    ))
                    .label()
                    .text_align(TextAlign::Center)
                    .width(Size::Fixed(120.0)),
                    icon_button("chevron-right")
                        .ghost()
                        .xsmall()
                        .key(format!("{CACHE_ADJUST_PREFIX}up")),
                ])
                .gap(tokens::SPACE_1)
                .align(Align::Center),
            )
            .gap(tokens::SPACE_2),
            field_row(
                "Hits / misses",
                text(format!("{} / {}", stats.hits, stats.misses)).label(),
            )
            .gap(tokens::SPACE_2),
            text("Opening a built copy raises the budget to fit its results.")
                .caption()
                .muted(),
        ])
        .gap(tokens::SPACE_2)
        .padding(tokens::SPACE_2)
        .width(Size::Fixed(300.0))]),
    )
}

/// Anchored popover with the remote build daemon address. The address is a
/// controlled text input into a plain buffer; it takes effect the next time
/// the Remote toggle is switched on, so mid-edit keystrokes never tear down
/// a live executor.
fn remote_settings_popover(app: &VolumetricUiV2) -> El {
    let state = if app.remote_build {
        format!("building on {}", app.remote_address.trim())
    } else {
        "building locally".to_string()
    };
    popover(
        REMOTE_SETTINGS_KEY,
        Anchor::below_key(REMOTE_SETTINGS_KEY),
        popover_panel([column([
            text("Remote build").label().semibold(),
            field_row(
                "Daemon",
                text_input(REMOTE_ADDRESS_KEY, &app.remote_address, &app.selection)
                    .width(Size::Fill(1.0)),
            )
            .gap(tokens::SPACE_2),
            text(state).caption().muted(),
            text("Address changes apply when Remote is toggled on.")
                .caption()
                .muted(),
        ])
        .gap(tokens::SPACE_2)
        .padding(tokens::SPACE_2)
        .width(Size::Fixed(320.0))]),
    )
}

/// Anchored settings popover for one output, with controls for the output's
/// kind (3D mesh plans, 2D raster resolution, FEA view modes, …) and a
/// reset to the defaults when overridden. Picks keep the popover open (it's
/// a settings panel, not a value picker); outside click dismisses.
fn output_settings_popover(app: &VolumetricUiV2, id: &str) -> El {
    let render = app.output_render(id);
    let trigger_key = format!("{OUTPUT_SETTINGS_PREFIX}{id}");

    let mut body = vec![text(id).label().semibold().ellipsis()];
    match &render {
        OutputRender::Model3d {
            mode,
            resolution,
            asn2,
            wireframe,
            color_channel,
        } => {
            body.extend(model3d_settings(
                app,
                id,
                *mode,
                *resolution,
                asn2,
                *wireframe,
                color_channel.as_deref(),
            ));
        }
        OutputRender::Model2d {
            resolution,
            color_channel,
        } => {
            body.push(
                button_with_icon("search", "Inspect field…")
                    .xsmall()
                    .secondary()
                    .width(Size::Fill(1.0))
                    .key(format!("{OUTPUT_INSPECT_PREFIX}{id}")),
            );
            body.push(text("Raster Resolution").caption().muted());
            body.push(
                row(SKETCH_RESOLUTIONS.iter().map(|&preset| {
                    let button = button(preset.to_string())
                        .xsmall()
                        .key(format!("{OUTPUT_RESOLUTION_PREFIX}{id}:{preset}"));
                    if *resolution == preset {
                        button.primary()
                    } else {
                        button.secondary()
                    }
                }))
                .gap(tokens::SPACE_1),
            );
            body.extend(color_by_buttons(app, id, color_channel.as_deref()));
        }
        OutputRender::FeaMesh(fea) => {
            body.extend(fea_settings(app, id, fea));
        }
        OutputRender::TriMesh { wireframe } => {
            body.push(
                field_row(
                    "Wireframe",
                    switch(format!("{OUTPUT_WIREFRAME_PREFIX}{id}"), *wireframe),
                )
                .gap(tokens::SPACE_2),
            );
        }
        OutputRender::Subspace => {
            body.push(text("Subspace gizmo · no settings yet").caption().muted());
        }
    }
    if let Some(stats) = app.output_stats.get(id) {
        body.push(divider());
        let mut summary = format!("meshed in {:.1} ms", stats.mesh_ms);
        if stats.triangles > 0 {
            summary.push_str(&format!(" · {} tris", format_count(stats.triangles)));
        }
        if stats.points > 0 {
            summary.push_str(&format!(" · {} pts", format_count(stats.points)));
        }
        if stats.samples > 0 {
            summary.push_str(&format!(
                " · {} samples",
                format_count(stats.samples as usize)
            ));
        }
        body.push(text(summary).caption().muted());
        if let Some((min, max)) = stats.bounds {
            body.push(
                text(format!("size {}", format_dims(min, max)))
                    .caption()
                    .muted(),
            );
            body.push(
                text(format!(
                    "bounds ({}, {}, {}) to ({}, {}, {})",
                    format_dim(min.0),
                    format_dim(min.1),
                    format_dim(min.2),
                    format_dim(max.0),
                    format_dim(max.1),
                    format_dim(max.2),
                ))
                .caption()
                .muted(),
            );
        }
        for line in &stats.detail {
            body.push(text(line).caption().muted());
        }
    }
    if app.output_overrides.contains_key(id) {
        body.push(
            button("Use viewport defaults")
                .xsmall()
                .secondary()
                .width(Size::Fill(1.0))
                .key(format!("{OUTPUT_DEFAULTS_PREFIX}{id}")),
        );
    }
    body.push(divider());
    body.push(
        button_with_icon("download", "Export mesh…")
            .xsmall()
            .secondary()
            .width(Size::Fill(1.0))
            .key(format!("{EXPORT_MESH_PREFIX}{id}")),
    );
    body.push(
        button_with_icon("upload", "Export WASM…")
            .xsmall()
            .secondary()
            .width(Size::Fill(1.0))
            .key(format!("{EXPORT_WASM_PREFIX}{id}")),
    );

    popover(
        trigger_key.clone(),
        Anchor::below_key(trigger_key),
        // popover_panel paints the floating card surface (fill + border +
        // shadow); a bare column would float transparently over the panel.
        popover_panel([column(body).gap(tokens::SPACE_2).padding(tokens::SPACE_2)]),
    )
}

/// 3D model settings: render mode, voxel resolution, wireframe, ASN2 knobs.
fn model3d_settings(
    app: &VolumetricUiV2,
    id: &str,
    mode: PreviewRenderMode,
    resolution: usize,
    asn2: &Asn2Settings,
    wireframe: bool,
    color_channel: Option<&str>,
) -> Vec<El> {
    let mode_buttons = row(PreviewRenderMode::ALL.into_iter().map(|preset| {
        let button = button(preset.label())
            .xsmall()
            .tooltip(preset.full_label())
            .key(format!("{OUTPUT_MODE_PREFIX}{id}:{}", preset.route_name()));
        if mode == preset {
            button.primary()
        } else {
            button.secondary()
        }
    }))
    .gap(tokens::SPACE_1);
    // Two rows: the preset ladder is too wide for one popover line.
    let resolution_buttons = column(PREVIEW_RESOLUTIONS.chunks(5).map(|chunk| {
        row(chunk.iter().map(|&preset| {
            let button = button(format!("{preset}^3"))
                .xsmall()
                .key(format!("{OUTPUT_RESOLUTION_PREFIX}{id}:{preset}"));
            if resolution == preset {
                button.primary()
            } else {
                button.secondary()
            }
        }))
        .gap(tokens::SPACE_1)
    }))
    .gap(tokens::SPACE_1);

    let mut body = vec![
        text("Render Mode").caption().muted(),
        mode_buttons,
        text("Resolution").caption().muted(),
        resolution_buttons,
    ];
    if mode != PreviewRenderMode::Points {
        body.push(
            field_row(
                "Wireframe",
                switch(format!("{OUTPUT_WIREFRAME_PREFIX}{id}"), wireframe),
            )
            .gap(tokens::SPACE_2),
        );
    }
    if mode == PreviewRenderMode::AdaptiveSurfaceNets2 {
        body.push(text("ASN2 Quality").caption().muted());
        body.push(asn2_stepper_row(
            id,
            "vr",
            "Vertex refine",
            &asn2.vertex_refinement_iterations.to_string(),
        ));
        body.push(asn2_stepper_row(
            id,
            "nr",
            "Normal refine",
            &asn2.normal_sample_iterations.to_string(),
        ));
        body.push(
            field_row(
                "Sharp edges",
                switch(
                    format!("{OUTPUT_ASN2_PREFIX}{id}:sharp:up"),
                    asn2.sharp_edges,
                ),
            )
            .gap(tokens::SPACE_2),
        );
        // Refine vertices only along their own grid edge: prevents thin
        // lattice sheets from visually bonding when refinement captures a
        // neighboring parallel surface.
        body.push(
            field_row(
                "Edge-constrained",
                switch(
                    format!("{OUTPUT_ASN2_PREFIX}{id}:edgec:up"),
                    asn2.edge_constrained_refinement,
                ),
            )
            .gap(tokens::SPACE_2),
        );
        // Stage-5 quadric decimation; tolerance is the allowed surface
        // deviation in finest-cell units.
        body.push(
            field_row(
                "Simplify",
                switch(format!("{OUTPUT_ASN2_PREFIX}{id}:simp:up"), asn2.simplify),
            )
            .gap(tokens::SPACE_2),
        );
        if asn2.simplify {
            body.push(asn2_stepper_row(
                id,
                "simptol",
                "Tolerance (cells)",
                &format!("{:.1}", f64::from(asn2.simplify_tolerance_tenths) / 10.0),
            ));
        }
        if asn2.sharp_edges {
            body.push(asn2_stepper_row(
                id,
                "angle",
                "Angle (deg)",
                &asn2.sharp_angle_degrees.to_string(),
            ));
        }
        // Stage-1 discovery: interior probes catch lattice geometry thinner
        // than the coarse grid pitch; the base grid can be pinned denser
        // than the automatic split.
        body.push(asn2_stepper_row(
            id,
            "probes",
            "Discovery probes",
            &asn2.discovery_probes.to_string(),
        ));
        let base_label = if asn2.base_resolution == 0 {
            "auto".to_string()
        } else {
            asn2.base_resolution.to_string()
        };
        body.push(asn2_stepper_row(id, "base", "Discovery grid", &base_label));
    }

    // Channel colormap + slice inspection, for models that declare sample
    // channels beyond occupancy (e.g. fea_density's density channel).
    if app.output_channels(id).len() > 1 {
        body.extend(color_by_buttons(app, id, color_channel));
        body.push(
            button_with_icon("search", "Inspect slice…")
                .secondary()
                .xsmall()
                .width(Size::Fill(1.0))
                .key(format!("{OUTPUT_INSPECT_PREFIX}{id}")),
        );
    }
    body
}

/// The "Color by" channel picker rows, for outputs whose model declares
/// sample channels beyond occupancy. Empty when there is nothing beyond
/// channel 0 to color by.
fn color_by_buttons(app: &VolumetricUiV2, id: &str, color_channel: Option<&str>) -> Vec<El> {
    let channels = app.output_channels(id);
    if channels.len() <= 1 {
        return Vec::new();
    }
    let mut rows = vec![text("Color by").caption().muted()];
    let none_button = button("None")
        .xsmall()
        .width(Size::Fill(1.0))
        .key(format!("{OUTPUT_CHANNEL_PREFIX}{id}:none"));
    rows.push(if color_channel.is_none() {
        none_button.primary()
    } else {
        none_button.secondary()
    });
    for channel in channels.iter().skip(1) {
        let button = button(channel.clone())
            .xsmall()
            .width(Size::Fill(1.0))
            .key(format!("{OUTPUT_CHANNEL_PREFIX}{id}:{channel}"));
        rows.push(if color_channel == Some(channel.as_str()) {
            button.primary()
        } else {
            button.secondary()
        });
    }
    rows
}

/// FEA mesh settings: deformed view + exaggeration, wireframe, and the
/// colormap field picker (fields mirrored from the built preview's stats).
fn fea_settings(app: &VolumetricUiV2, id: &str, fea: &FeaRender) -> Vec<El> {
    let mut body = vec![
        field_row(
            "Deformed",
            switch(format!("{OUTPUT_FEA_DEFORMED_PREFIX}{id}"), fea.deformed),
        )
        .gap(tokens::SPACE_2),
    ];
    if fea.deformed {
        body.push(
            field_row(
                "Exaggeration",
                row([
                    icon_button("chevron-left")
                        .ghost()
                        .xsmall()
                        .key(format!("{OUTPUT_FEA_EXAG_PREFIX}{id}:down")),
                    text(format!("x{}", fea.exaggeration_tenths as f32 / 10.0))
                        .label()
                        .text_align(TextAlign::Center)
                        .width(Size::Fixed(56.0)),
                    icon_button("chevron-right")
                        .ghost()
                        .xsmall()
                        .key(format!("{OUTPUT_FEA_EXAG_PREFIX}{id}:up")),
                ])
                .gap(tokens::SPACE_1)
                .align(Align::Center),
            )
            .gap(tokens::SPACE_2),
        );
    }
    body.push(
        field_row(
            "Wireframe",
            switch(format!("{OUTPUT_WIREFRAME_PREFIX}{id}"), fea.wireframe),
        )
        .gap(tokens::SPACE_2),
    );

    body.push(text("Colormap").caption().muted());
    let none_button = button("None")
        .xsmall()
        .width(Size::Fill(1.0))
        .key(format!("{OUTPUT_FEA_FIELD_PREFIX}{id}:none"));
    body.push(if fea.color_field.is_none() {
        none_button.primary()
    } else {
        none_button.secondary()
    });
    let fields = app
        .output_stats
        .get(id)
        .map(|stats| stats.fea_fields.clone())
        .unwrap_or_default();
    if fields.is_empty() {
        body.push(
            text("fields appear after the first render")
                .caption()
                .muted(),
        );
    }
    for field in fields {
        let label = field
            .split_once(':')
            .map(|(_, name)| name)
            .unwrap_or(&field);
        let button = button(label)
            .xsmall()
            .tooltip(field.clone())
            .width(Size::Fill(1.0))
            .key(format!("{OUTPUT_FEA_FIELD_PREFIX}{id}:{field}"));
        body.push(if fea.color_field.as_deref() == Some(field.as_str()) {
            button.primary()
        } else {
            button.secondary()
        });
    }
    body
}

/// A label + `- value +` stepper row driving one ASN2 setting.
fn asn2_stepper_row(id: &str, field: &str, label: &str, value: &str) -> El {
    field_row(
        label,
        row([
            icon_button("chevron-left")
                .ghost()
                .xsmall()
                .key(format!("{OUTPUT_ASN2_PREFIX}{id}:{field}:down")),
            text(value)
                .label()
                .text_align(TextAlign::Center)
                .width(Size::Fixed(40.0)),
            icon_button("chevron-right")
                .ghost()
                .xsmall()
                .key(format!("{OUTPUT_ASN2_PREFIX}{id}:{field}:up")),
        ])
        .gap(tokens::SPACE_1)
        .align(Align::Center),
    )
    .gap(tokens::SPACE_2)
}

/// The viewport plus its floating chrome: view controls in the top-right
/// corner, a status HUD along the bottom. Only keyed controls hit-test, so
/// camera input passes through everywhere else.
fn viewport_pane(app: &VolumetricUiV2) -> El {
    stack([viewport_placeholder(app), viewport_overlay(app)])
        .width(Size::Fill(1.0))
        .height(Size::Fill(1.0))
}

fn viewport_overlay(app: &VolumetricUiV2) -> El {
    column([
        row([spacer(), view_controls_cluster(app)]).width(Size::Fill(1.0)),
        spacer().height(Size::Fill(1.0)),
        viewport_hud(app),
    ])
    .fill_size()
    .padding(tokens::SPACE_2)
}

/// Compact floating cluster of view toggles + pickers.
fn view_controls_cluster(app: &VolumetricUiV2) -> El {
    let toggle = |label: &str, on: bool, key: &str| {
        let button = button(label).xsmall().key(key);
        if on {
            button.primary()
        } else {
            button.secondary()
        }
    };
    card([row([
        toggle("Grid", app.show_grid, TOGGLE_GRID_KEY),
        toggle("Bounds", app.show_bounds, TOGGLE_BOUNDS_KEY),
        toggle("SSAO", app.ssao, TOGGLE_SSAO_KEY),
        icon_button("chevron-down")
            .ghost()
            .xsmall()
            .tooltip("SSAO settings")
            .key(SSAO_SETTINGS_KEY),
        button("Frame")
            .xsmall()
            .secondary()
            .tooltip("Frame the preview in view")
            .key(FRAME_PREVIEW_KEY),
        button("Reset")
            .xsmall()
            .secondary()
            .tooltip("Reset the camera to its default pose")
            .key(RESET_CAMERA_KEY),
        vertical_separator().height(Size::Fixed(20.0)),
        select_trigger(MODE_SELECT_KEY, app.render_mode.label()).width(Size::Fixed(90.0)),
        select_trigger(
            RESOLUTION_SELECT_KEY,
            format!("{}^3", app.preview_resolution),
        )
        .width(Size::Fixed(84.0)),
        select_trigger(
            CAMERA_SELECT_KEY,
            camera_scheme_short_label(app.camera_control_scheme),
        )
        .width(Size::Fixed(104.0)),
    ])
    .gap(tokens::SPACE_1)
    .align(Align::Center)])
    .padding(tokens::SPACE_1)
    // Hug, not the card's default Fill: anything wider is a click-eating
    // band over the viewport (keyed controls win the stack hit-test).
    .width(Size::Hug)
}

/// One-line readout at the bottom of the viewport. Unkeyed, so it never
/// intercepts camera input.
fn viewport_hud(app: &VolumetricUiV2) -> El {
    let requests = app.preview_requests();
    let triangles: usize = app.output_stats.values().map(|s| s.triangles).sum();
    let points: usize = app.output_stats.values().map(|s| s.points).sum();
    let mut badges = vec![
        badge(format!("{} in viewport", requests.len()))
            .muted()
            .xsmall(),
        badge(format!("{} outputs", app.runtime_assets.len()))
            .muted()
            .xsmall(),
    ];
    if triangles > 0 {
        badges.push(
            badge(format!("{} tris", format_count(triangles)))
                .muted()
                .xsmall(),
        );
    }
    if points > 0 {
        badges.push(
            badge(format!("{} pts", format_count(points)))
                .muted()
                .xsmall(),
        );
    }
    if app.show_bounds {
        for request in &requests {
            let Some((min, max)) = app
                .output_stats
                .get(&request.asset_id)
                .and_then(|s| s.bounds)
            else {
                continue;
            };
            badges.push(
                badge(format!("{}: {}", request.asset_id, format_dims(min, max)))
                    .secondary()
                    .xsmall(),
            );
        }
    }
    if let Some(warning) = &app.viewport_overflow {
        badges.push(badge(warning).destructive().xsmall());
    }
    badges.push(badge(&app.status).secondary().xsmall());
    badges.push(spacer());
    row(badges).gap(tokens::SPACE_1).align(Align::Center)
}

/// Convert preview triangles to a damascene scene mesh (unindexed,
/// per-vertex normals carried over) plus their tight `(min, max)` bounds.
/// Both vocabularies are Y-up, so positions carry over unmapped.
fn export_scene_mesh(
    triangles: &[volumetric::Triangle],
) -> (SceneMeshData, ((f32, f32, f32), (f32, f32, f32))) {
    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    let mut min = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for triangle in triangles {
        for (position, normal) in triangle.vertices.iter().zip(&triangle.normals) {
            min = (
                min.0.min(position.0),
                min.1.min(position.1),
                min.2.min(position.2),
            );
            max = (
                max.0.max(position.0),
                max.1.max(position.1),
                max.2.max(position.2),
            );
            vertices.push(SceneMeshVertex {
                position: SceneVec3::new(position.0, position.1, position.2),
                normal: SceneVec3::new(normal.0, normal.1, normal.2),
            });
        }
    }
    (
        SceneMeshData {
            vertices,
            indices: None,
        },
        (min, max),
    )
}

/// Parse the export modal's scale buffer: a finite, strictly positive
/// factor. `None` disables Export and flags the field.
fn parse_export_scale(text: &str) -> Option<f32> {
    let value: f32 = text.trim().parse().ok()?;
    (value.is_finite() && value > 0.0).then_some(value)
}

/// Scale export triangles about the origin. Uniform, so the (unit) normals
/// are unchanged. Runs on the file worker, not the UI thread.
pub(crate) fn scale_triangles(triangles: &mut [volumetric::Triangle], scale: f32) {
    if scale == 1.0 {
        return;
    }
    for triangle in triangles {
        for vertex in &mut triangle.vertices {
            vertex.0 *= scale;
            vertex.1 *= scale;
            vertex.2 *= scale;
        }
    }
}

/// `W × D × H` size of a bounding box, compactly formatted.
fn format_dims(min: (f32, f32, f32), max: (f32, f32, f32)) -> String {
    format!(
        "{} × {} × {}",
        format_dim(max.0 - min.0),
        format_dim(max.1 - min.1),
        format_dim(max.2 - min.2),
    )
}

/// Compact dimension formatting: up to 3 decimals, trailing zeros trimmed.
fn format_dim(value: f32) -> String {
    let text = format!("{value:.3}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    text.to_string()
}

/// Compact count formatting: 950, 12.4k, 3.1M.
fn format_count(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

fn toggle_chip(label: &str, value: bool, key: &str) -> El {
    row([text(label).caption().muted().ellipsis(), switch(key, value)])
        .gap(tokens::SPACE_1)
        .align(Align::Center)
        .height(Size::Fixed(24.0))
}

/// Run/Cancel button that reflects the async run lifecycle. While a run is in
/// flight it becomes a Cancel action; otherwise it triggers a run request.
fn run_control(app: &VolumetricUiV2) -> El {
    match app.run_state() {
        RunState::Running => button_with_icon("x", "Cancel")
            .destructive()
            .xsmall()
            .key(CANCEL_RUN_KEY),
        RunState::Idle => button_with_icon("refresh-cw", "Run")
            .primary()
            .xsmall()
            .key(RUN_PROJECT_KEY),
    }
}

fn run_status_chip(app: &VolumetricUiV2) -> El {
    match app.run_state() {
        RunState::Running => {
            let label = match &app.run_progress {
                Some(progress) => match progress.fraction {
                    Some(fraction) => {
                        format!("{} · {:.0}%", progress.phase, fraction * 100.0)
                    }
                    None => progress.phase.clone(),
                },
                None => "running".to_string(),
            };
            row([
                spinner().width(Size::Fixed(14.0)).height(Size::Fixed(14.0)),
                badge(label).info().xsmall(),
            ])
            .gap(tokens::SPACE_1)
            .align(Align::Center)
        }
        RunState::Idle => {
            if app.last_run_stale && !app.runtime_assets.is_empty() {
                badge("stale").secondary().xsmall()
            } else if let Some(ms) = app.last_run_elapsed_ms {
                badge(format!("ran {ms}ms")).success().xsmall()
            } else {
                badge("not run").muted().xsmall()
            }
        }
    }
}

fn preview_status_chip(app: &VolumetricUiV2) -> El {
    let mut label = app.preview_build_status.label();
    // While a build is in flight, append its latest meshing phase (e.g.
    // "subdividing (1.2M cells)") so long builds show what they're doing.
    if matches!(
        app.preview_build_status,
        PreviewBuildStatus::Building { .. }
    ) && let Some(progress) = &app.preview_progress
    {
        label = format!("{label} · {}", progress.phase);
    }
    let badge = match &app.preview_build_status {
        PreviewBuildStatus::Idle => badge(label).muted().xsmall(),
        PreviewBuildStatus::Building { .. } => badge(label).info().xsmall(),
        PreviewBuildStatus::Ready { .. } => badge(label).success().xsmall(),
        PreviewBuildStatus::Stale { .. } => badge(label).secondary().xsmall(),
        PreviewBuildStatus::Failed { .. } => badge(label).destructive().xsmall(),
    };
    let badge = if let Some(tooltip) = app.preview_build_status.tooltip() {
        badge.tooltip(tooltip)
    } else {
        badge
    };

    if matches!(
        app.preview_build_status,
        PreviewBuildStatus::Building { .. }
    ) {
        row([
            spinner().width(Size::Fixed(14.0)).height(Size::Fixed(14.0)),
            badge,
        ])
        .gap(tokens::SPACE_1)
        .align(Align::Center)
    } else {
        badge
    }
}

/// Contextual mesh-build action: Cancel while previews build, Remesh when
/// they are stale (auto-remesh off, or the builds were cancelled). `None`
/// when there is nothing actionable.
fn mesh_control(app: &VolumetricUiV2) -> Option<El> {
    match &app.preview_build_status {
        PreviewBuildStatus::Building { .. } => Some(
            button_with_icon("x", "Cancel")
                .destructive()
                .xsmall()
                .key(CANCEL_MESH_KEY),
        ),
        PreviewBuildStatus::Stale { .. } => Some(
            button_with_icon("refresh-cw", "Remesh")
                .primary()
                .xsmall()
                .key(REMESH_KEY),
        ),
        _ => None,
    }
}

fn viewport_placeholder(app: &VolumetricUiV2) -> El {
    if let Some(texture) = &app.viewport_texture {
        surface(texture.clone())
            .surface_alpha(SurfaceAlpha::Opaque)
            .surface_fit(ImageFit::Fill)
            .key(VIEWPORT_KEY)
            .fill_size()
            .clip()
    } else {
        spacer().key(VIEWPORT_KEY).fill_size().clip()
    }
}

/// The single project panel: pipeline spine (imports → steps → exports),
/// materialized outputs, and the inspector for the current selection.
fn project_panel(app: &VolumetricUiV2) -> El {
    column([scroll([
        pipeline_accordion(app),
        divider(),
        panel_section("Outputs", outputs_rows(app)),
        divider(),
        panel_section("Inspector", inspector_rows(app)),
    ])
    .key("project-panel-scroll")
    .gap(tokens::SPACE_3)
    // Gutter so keyboard focus rings on full-width rows aren't clipped
    // by the scroll's horizontal scissor.
    .px(tokens::RING_WIDTH)])
    .width(Size::Fixed(app.panel_width))
    .height(Size::Fill(1.0))
    .padding(tokens::SPACE_3)
    .gap(tokens::SPACE_2)
}

fn panel_section(title: &str, body: Vec<El>) -> El {
    let mut children = vec![text(title).muted().caption().semibold()];
    children.extend(body);
    column(children).gap(tokens::SPACE_2).width(Size::Fill(1.0))
}

fn pipeline_accordion(app: &VolumetricUiV2) -> El {
    let open = |section: &str| app.pipeline_open.contains(section);
    let section = |value: &str, label: String, rows: Vec<El>| {
        accordion_item(
            PIPELINE_KEY,
            value,
            label,
            open(value),
            // Gutter so the trigger's focus ring isn't occluded by the first
            // row painted flush below it.
            [table([table_body(rows).gap(tokens::SPACE_1)])
                .width(Size::Fill(1.0))
                .py(tokens::RING_WIDTH)],
        )
    };
    accordion([
        section(
            "imports",
            format!("Imports ({})", app.project.imports().len()),
            import_rows(app),
        ),
        section(
            "steps",
            format!("Steps ({})", app.project.timeline().len()),
            step_rows(app),
        ),
        section(
            "exports",
            format!("Exports ({})", app.project.exports().len()),
            export_rows(app),
        ),
    ])
    // The stock accordion is gapless; with adjacent *collapsed* items the
    // triggers' expanded hit targets overlap and focus rings occlude
    // (damascene's own bundle lint flags it). A ring-width gap keeps every
    // trigger's hit band and focus ring its own.
    .gap(tokens::RING_WIDTH)
}

fn inspector_rows(app: &VolumetricUiV2) -> Vec<El> {
    let mut rows = match &app.selected_project_item {
        Some(ProjectSelection::Import(idx)) => import_detail_rows(app, *idx),
        Some(ProjectSelection::Step(idx)) => step_detail_rows(app, *idx),
        Some(ProjectSelection::Export(idx)) => export_detail_rows(app, *idx),
        None => vec![
            text("Select an import, operation, or export.")
                .muted()
                .small(),
        ],
    };

    // The viewport follows the selected node; flag when it has no materialized
    // output to show (only pinned outputs render until the project runs).
    if app.selected_render_id().is_some() && !app.selection_is_renderable() {
        rows.push(
            alert([alert_description(
                "No materialized output — run the project to preview this node.",
            )])
            .info()
            .padding(tokens::SPACE_2),
        );
    }

    rows
}

fn outputs_rows(app: &VolumetricUiV2) -> Vec<El> {
    let mut rows = Vec::new();

    if let Some(error) = &app.last_run_error {
        rows.push(
            alert([alert_description(error)])
                .destructive()
                .padding(tokens::SPACE_2),
        );
    } else if app.runtime_assets.is_empty() {
        rows.push(
            text("Run the project to materialize exports.")
                .muted()
                .small(),
        );
    } else {
        rows.push(detail_row(
            "Last run",
            &format!(
                "{}ms{}",
                app.last_run_elapsed_ms.unwrap_or_default(),
                if app.last_run_stale { " stale" } else { "" }
            ),
        ));
        rows.push(
            table([table_body(
                app.runtime_assets
                    .iter()
                    .map(|asset| runtime_asset_row(app, asset))
                    .collect::<Vec<_>>(),
            )
            .gap(tokens::SPACE_1)])
            .width(Size::Fill(1.0)),
        );
    }

    rows
}

fn runtime_asset_row(app: &VolumetricUiV2, asset: &LoadedAsset) -> El {
    let id = asset.id();
    let pinned = app.pinned_outputs.contains(id);
    let visible = app.output_is_visible(id);
    let render = app.output_render(id);
    let overridden = app.output_overrides.contains_key(id);

    let pin = icon_button(&*PIN_ICON)
        .xsmall()
        .tooltip(if pinned { "Unpin" } else { "Pin to viewport" })
        .key(format!("{TOGGLE_PIN_PREFIX}{id}"));
    let pin = if pinned { pin.primary() } else { pin.ghost() };

    let settings = icon_button("settings")
        .xsmall()
        .tooltip(if overridden {
            "Render settings (overriding defaults)"
        } else {
            "Render settings"
        })
        .key(format!("{OUTPUT_SETTINGS_PREFIX}{id}"));
    let settings = if overridden {
        settings.primary()
    } else {
        settings.ghost()
    };

    table_row([
        // Dot marks outputs currently drawn in the viewport.
        text(if visible { "●" } else { "○" })
            .caption()
            .muted()
            .width(Size::Fixed(12.0)),
        column([
            text(format!(
                "{} · {}{}",
                asset_type_label(asset.type_hint()),
                render.summary(),
                if overridden { " *" } else { "" },
            ))
            .caption()
            .muted()
            .ellipsis()
            .width(Size::Fill(1.0)),
            text(id).label().ellipsis().width(Size::Fill(1.0)),
        ])
        .gap(1.0)
        .width(Size::Fill(1.0)),
        text(format_bytes(asset.data().len()))
            .caption()
            .muted()
            .text_align(TextAlign::End)
            .width(Size::Fixed(56.0)),
        pin,
        icon_button(&*EYE_ICON)
            .ghost()
            .xsmall()
            .tooltip("View")
            .key(format!("{SELECT_RUNTIME_ASSET_PREFIX}{id}")),
        settings,
    ])
    .height(Size::Fixed(36.0))
    .padding(Sides::xy(tokens::SPACE_2, 0.0))
    .gap(tokens::SPACE_1)
}

fn import_detail_rows(app: &VolumetricUiV2, idx: usize) -> Vec<El> {
    let Some(import) = app.project.imports().get(idx) else {
        return vec![text("Selected import no longer exists.").muted().small()];
    };

    let mut rows = vec![
        detail_row("Kind", asset_type_label(import.type_hint)),
        detail_row("Asset", &import.id),
        detail_row("Bytes", &import.data.len().to_string()),
    ];
    if import.type_hint == Some(AssetTypeHint::Operator)
        && let Some(metadata) = app.operator_metadata_cached(&import.id)
    {
        rows.push(detail_row("Version", &metadata.version));
        if let Some((_, bundled_version)) = operator_upgrade_offer(app, import) {
            rows.push(
                button_with_icon("refresh-cw", format!("Update to {bundled_version}"))
                    .secondary()
                    .xsmall()
                    .width(Size::Fill(1.0))
                    .key(format!("{UPGRADE_IMPORT_PREFIX}{idx}")),
            );
        }
    }
    rows.push(
        button_with_icon("x", "Delete")
            .destructive()
            .xsmall()
            .width(Size::Fill(1.0))
            .key(format!("{DELETE_IMPORT_PREFIX}{idx}")),
    );
    rows
}

fn step_detail_rows(app: &VolumetricUiV2, idx: usize) -> Vec<El> {
    let Some(step) = app.project.timeline().get(idx) else {
        return vec![text("Selected step no longer exists.").muted().small()];
    };

    let mut rows = vec![
        detail_row("Step", &(idx + 1).to_string()),
        detail_row("Operator", &step.operator_id),
        detail_row("Outputs", &step.outputs.join(", ")),
    ];

    rows.extend(step_edit_rows(app, idx));

    rows.push(
        toolbar([
            button_with_icon("chevron-up", "Up")
                .secondary()
                .xsmall()
                .key(format!("{MOVE_STEP_UP_PREFIX}{idx}")),
            button_with_icon("chevron-down", "Down")
                .secondary()
                .xsmall()
                .key(format!("{MOVE_STEP_DOWN_PREFIX}{idx}")),
        ])
        .gap(tokens::SPACE_1)
        .width(Size::Fill(1.0)),
    );
    rows.push(
        button_with_icon("x", "Delete")
            .destructive()
            .xsmall()
            .width(Size::Fill(1.0))
            .key(format!("{DELETE_STEP_PREFIX}{idx}")),
    );

    rows
}

/// Renders the editor for the selected step: a per-slot model selector for each
/// `ModelWASM` input, then the config form (if the operator declares config).
fn step_edit_rows(app: &VolumetricUiV2, step_idx: usize) -> Vec<El> {
    let Some(edit) = &app.step_edit else {
        return Vec::new();
    };
    if edit.step_idx != step_idx {
        return Vec::new();
    }
    let Some(step) = app.project.timeline().get(step_idx) else {
        return Vec::new();
    };

    let mut rows = Vec::new();

    if !edit.asset_slots.is_empty() {
        let models = step_input_options(app, step_idx, AssetSlotKind::Model);
        let meshes = step_input_options(app, step_idx, AssetSlotKind::FeaMesh);
        let trimeshes = step_input_options(app, step_idx, AssetSlotKind::TriMesh);
        let subspaces = step_input_options(app, step_idx, AssetSlotKind::Subspace);
        rows.push(text("Inputs").muted().caption().semibold());
        for (n, slot) in edit.asset_slots.iter().enumerate() {
            let current = match step.inputs.get(slot.input_idx) {
                Some(ExecutionInput::AssetRef(id)) => id.as_str(),
                _ => "",
            };
            let options = match slot.kind {
                AssetSlotKind::Model => &models,
                AssetSlotKind::FeaMesh => &meshes,
                AssetSlotKind::TriMesh => &trimeshes,
                AssetSlotKind::Subspace => &subspaces,
            };
            rows.push(asset_slot_selector(step_idx, slot, n, current, options));
        }
    }

    for vec_form in &edit.vecs {
        let label = vec_form
            .name
            .clone()
            .unwrap_or_else(|| format!("Vector (input {})", vec_form.input_idx + 1));
        rows.push(text(label).muted().caption().semibold());
        rows.push(
            row(vec_form
                .buffers
                .iter()
                .enumerate()
                .map(|(component, buffer)| {
                    text_input(
                        &format!("{VEC_INPUT_PREFIX}{}:{component}", vec_form.input_idx),
                        buffer,
                        &app.selection,
                    )
                    .width(Size::Fill(1.0))
                })
                .collect::<Vec<_>>())
            .gap(tokens::SPACE_1)
            .width(Size::Fill(1.0)),
        );
    }

    if let Some(config) = &edit.config {
        rows.push(text("Config").muted().caption().semibold());
        for field in &config.fields {
            let buffer = config
                .buffers
                .get(&field.name)
                .map(String::as_str)
                .unwrap_or("");
            rows.push(config_field_row(field, buffer, &app.selection));
        }
    }

    if let Some(lua) = &edit.lua {
        rows.push(text("Script").muted().caption().semibold());
        rows.push(
            text_area(LUA_SOURCE_KEY, &lua.source, &app.selection)
                .width(Size::Fill(1.0))
                .height(Size::Fixed(180.0)),
        );
        rows.push(
            button("Reset to template")
                .xsmall()
                .secondary()
                .width(Size::Fill(1.0))
                .key(format!("{RESET_LUA_PREFIX}{step_idx}")),
        );
    }

    rows.push(text("Output").muted().caption().semibold());
    rows.push(
        row([
            text_input(OUTPUT_NAME_KEY, &edit.output_name, &app.selection).width(Size::Fill(1.0)),
            button("Rename")
                .xsmall()
                .secondary()
                .key(format!("{RENAME_OUTPUT_PREFIX}{step_idx}")),
        ])
        .gap(tokens::SPACE_1)
        .align(Align::Center),
    );

    rows
}

/// The assets step `step_idx` may take as an input of the given kind: only
/// what exists before the step runs — imports plus outputs of earlier steps.
/// The step's own outputs and anything produced downstream would be a
/// self-reference or forward reference (the run loop executes the timeline
/// in order).
fn step_input_options(app: &VolumetricUiV2, step_idx: usize, kind: AssetSlotKind) -> Vec<String> {
    let not_yet_produced: std::collections::HashSet<&str> = app
        .project
        .timeline()
        .iter()
        .skip(step_idx)
        .flat_map(|step| step.outputs.iter().map(String::as_str))
        .collect();
    let ids = match kind {
        AssetSlotKind::Model => editable_model_asset_ids(app),
        AssetSlotKind::FeaMesh => editable_fea_asset_ids(app),
        AssetSlotKind::TriMesh => editable_trimesh_asset_ids(app),
        AssetSlotKind::Subspace => editable_subspace_asset_ids(app),
    };
    ids.into_iter()
        .filter(|id| !not_yet_produced.contains(id.as_str()))
        .collect()
}

/// An asset input slot: a labelled column of full-width buttons, one per
/// available asset of the slot's kind, with the current target highlighted.
fn asset_slot_selector(
    step_idx: usize,
    slot: &AssetSlot,
    ordinal: usize,
    current: &str,
    options: &[String],
) -> El {
    let label = slot
        .name
        .clone()
        .unwrap_or_else(|| format!("Input {}", ordinal + 1));
    let mut items = vec![text(label).muted().caption().width(Size::Fill(1.0))];
    if options.is_empty() {
        items.push(
            text(format!(
                "No {} produced before this step",
                asset_slot_kind_label(slot.kind),
            ))
            .muted()
            .caption(),
        );
    }
    for id in options {
        let button = button_with_icon("git-branch", id.clone())
            .xsmall()
            .width(Size::Fill(1.0))
            .key(format!(
                "{SET_STEP_MODEL_PREFIX}{step_idx}:{}:{id}",
                slot.input_idx
            ));
        items.push(if id == current {
            button.primary()
        } else {
            button.secondary()
        });
    }
    column(items).gap(tokens::SPACE_1).width(Size::Fill(1.0))
}

fn asset_slot_kind_label(kind: AssetSlotKind) -> &'static str {
    match kind {
        AssetSlotKind::Model => "model",
        AssetSlotKind::FeaMesh => "FEA mesh",
        AssetSlotKind::TriMesh => "triangle mesh",
        AssetSlotKind::Subspace => "subspace",
    }
}

fn config_field_row(field: &ConfigField, buffer: &str, selection: &Selection) -> El {
    let control = match &field.ty {
        ConfigFieldType::Bool => switch(
            format!("{CONFIG_BOOL_PREFIX}{}", field.name),
            buffer == "true",
        ),
        ConfigFieldType::Enum(options) => config_enum_control(&field.name, options, buffer),
        _ => text_input(
            &format!("{CONFIG_FIELD_PREFIX}{}", field.name),
            buffer,
            selection,
        )
        .width(Size::Fixed(132.0)),
    };
    // Optional fields (CDDL `?`) are marked; clearing one unsets it and the
    // operator's absent-field behavior applies.
    let label = if field.optional {
        format!("{} (optional)", field.name)
    } else {
        field.name.clone()
    };
    field_row(&label, control).gap(tokens::SPACE_2)
}

fn config_enum_control(field_name: &str, options: &[String], buffer: &str) -> El {
    row(options
        .iter()
        .map(|option| {
            let button = button(option.clone())
                .xsmall()
                .key(format!("{CONFIG_ENUM_PREFIX}{field_name}:{option}"));
            if option == buffer {
                button.primary()
            } else {
                button.secondary()
            }
        })
        .collect::<Vec<_>>())
    .gap(tokens::SPACE_1)
}

fn export_detail_rows(app: &VolumetricUiV2, idx: usize) -> Vec<El> {
    let Some(export_id) = app.project.exports().get(idx) else {
        return vec![text("Selected export no longer exists.").muted().small()];
    };

    let mut rows = vec![detail_row("Export", export_id)];
    // Lineage of the materialized output, when the last run produced one.
    if let Some(asset) = app
        .runtime_assets
        .iter()
        .find(|asset| asset.id() == export_id)
        && !asset.precursor_ids().is_empty()
    {
        rows.push(detail_row("Precursors", &asset.precursor_ids().join(", ")));
    }
    rows.push(
        button_with_icon("x", "Delete")
            .destructive()
            .xsmall()
            .width(Size::Fill(1.0))
            .key(format!("{DELETE_EXPORT_PREFIX}{idx}")),
    );
    rows
}

fn import_rows(app: &VolumetricUiV2) -> Vec<El> {
    if app.project.imports().is_empty() {
        return vec![empty_project_row("No imports")];
    }

    app.project
        .imports()
        .iter()
        .enumerate()
        .map(|(idx, import)| {
            let kind = if operator_upgrade_offer(app, import).is_some() {
                "Operator · update available".to_string()
            } else {
                asset_type_label(import.type_hint).to_string()
            };
            project_row(
                &kind,
                &import.id,
                format!("{SELECT_IMPORT_PREFIX}{idx}"),
                format!("{DELETE_IMPORT_PREFIX}{idx}"),
            )
        })
        .collect()
}

fn step_rows(app: &VolumetricUiV2) -> Vec<El> {
    if app.project.timeline().is_empty() {
        return vec![empty_project_row("No operations")];
    }

    app.project
        .timeline()
        .iter()
        .enumerate()
        .flat_map(|(idx, step)| {
            let mut rows = vec![project_row(
                &format!("Step {}", idx + 1),
                &step.operator_id,
                format!("{SELECT_STEP_PREFIX}{idx}"),
                format!("{DELETE_STEP_PREFIX}{idx}"),
            )];
            let metadata = app.operator_metadata_cached(&step.operator_id);
            for (input_idx, input) in step.inputs.iter().enumerate() {
                let label = metadata
                    .as_ref()
                    .and_then(|m| m.input_name(input_idx))
                    .unwrap_or("input");
                let value = match input {
                    ExecutionInput::AssetRef(id) => id.clone(),
                    inline => inline.display(),
                };
                rows.push(project_note_row(label, &value));
            }
            for output in &step.outputs {
                rows.push(project_note_row("output", output));
            }
            rows
        })
        .collect()
}

fn export_rows(app: &VolumetricUiV2) -> Vec<El> {
    let mut rows: Vec<El> = app
        .project
        .exports()
        .iter()
        .enumerate()
        .map(|(idx, export_id)| {
            project_row(
                "Export",
                export_id,
                format!("{SELECT_EXPORT_PREFIX}{idx}"),
                format!("{DELETE_EXPORT_PREFIX}{idx}"),
            )
        })
        .collect();

    if rows.is_empty() {
        rows.push(empty_project_row("No exports"));
    }

    let exported: std::collections::HashSet<&str> =
        app.project.exports().iter().map(String::as_str).collect();
    for (asset_id, _) in app.project.declared_assets() {
        if !exported.contains(asset_id.as_str()) {
            rows.push(
                button_with_icon("upload", format!("Export {asset_id}"))
                    .secondary()
                    .xsmall()
                    .width(Size::Fill(1.0))
                    .key(format!("{ADD_EXPORT_PREFIX}{asset_id}")),
            );
        }
    }

    rows
}

fn project_row(kind: &str, label: &str, select_route: String, delete_route: String) -> El {
    table_row([
        column([
            text(kind)
                .caption()
                .muted()
                .ellipsis()
                .width(Size::Fill(1.0)),
            text(label).label().ellipsis().width(Size::Fill(1.0)),
        ])
        .gap(1.0)
        .width(Size::Fill(1.0)),
        icon_button("settings")
            .ghost()
            .xsmall()
            .tooltip("Edit")
            .key(select_route),
        icon_button("x")
            .ghost()
            .destructive()
            .xsmall()
            .tooltip("Delete")
            .key(delete_route),
    ])
    .height(Size::Fixed(40.0))
    .padding(Sides::xy(tokens::SPACE_2, 0.0))
    .gap(tokens::SPACE_1)
    .align(Align::Center)
}

fn project_note_row(kind: &str, label: &str) -> El {
    table_row([
        // Wide enough for operator-declared input labels ("Rigid body",
        // "Height field (2D)"), not just the generic "input"/"output".
        text(kind)
            .caption()
            .muted()
            .ellipsis()
            .width(Size::Fixed(96.0)),
        text(label)
            .muted()
            .caption()
            .ellipsis()
            .width(Size::Fill(1.0)),
    ])
    .height(Size::Fixed(22.0))
    .padding(Sides::xy(tokens::SPACE_2, 0.0))
    .gap(tokens::SPACE_1)
}

fn empty_project_row(label: &str) -> El {
    table_row([text(label).muted().caption().width(Size::Fill(1.0))])
        .height(Size::Fixed(28.0))
        .padding(Sides::xy(tokens::SPACE_2, 0.0))
}

fn detail_row(label: &str, value: &str) -> El {
    field_row(
        label,
        text(value)
            .muted()
            .caption()
            .ellipsis()
            .text_align(TextAlign::End)
            .width(Size::Fixed(132.0)),
    )
    .gap(tokens::SPACE_2)
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Declared metadata name → version of every bundled operator, from the
/// build-time asset registry. Operator crates declare `name: <crate name>`
/// and `version: env!("CARGO_PKG_VERSION")` in their metadata (a convention
/// the `bundled_asset_registry_matches_declared_metadata` test enforces), so
/// the registry answers this without compiling any wasm — the previous
/// implementation compiled all ~28 MB of bundled operators on first use,
/// a ~40-second UI stall in a debug build.
fn bundled_operator_versions() -> &'static std::collections::HashMap<String, String> {
    static VERSIONS: LazyLock<std::collections::HashMap<String, String>> = LazyLock::new(|| {
        volumetric_assets::operators()
            .iter()
            .map(|asset| (asset.name.to_string(), asset.version.to_string()))
            .collect()
    });
    &VERSIONS
}

/// `(imported_version, bundled_version)` when an imported operator's
/// declared name matches a bundled operator whose version differs —
/// the signal that a one-click upgrade is worth offering. Projects embed
/// operator bytes, so fixes only reach them through this path (or re-adding
/// the operator by hand).
fn operator_upgrade_offer(
    app: &VolumetricUiV2,
    import: &volumetric::ImportedAsset,
) -> Option<(String, String)> {
    if import.type_hint != Some(AssetTypeHint::Operator) {
        return None;
    }
    let metadata = app.operator_metadata_cached(&import.id)?;
    let bundled_version = bundled_operator_versions().get(&metadata.name)?;
    (metadata.version != *bundled_version)
        .then(|| (metadata.version.clone(), bundled_version.clone()))
}

fn asset_type_label(type_hint: Option<AssetTypeHint>) -> &'static str {
    match type_hint {
        Some(AssetTypeHint::Model) => "Model",
        Some(AssetTypeHint::Operator) => "Operator",
        Some(AssetTypeHint::Config) => "Config",
        Some(AssetTypeHint::LuaSource) => "Lua",
        Some(AssetTypeHint::Binary) => "Binary",
        Some(AssetTypeHint::VecF64(_)) => "Vec",
        Some(AssetTypeHint::FeaMesh) => "FEA Mesh",
        Some(AssetTypeHint::TriMesh) => "Tri Mesh",
        Some(AssetTypeHint::Subspace) => "Subspace",
        None => "Asset",
    }
}

/// The assets a fresh step's typed slots initially point at (retargeted
/// later via the step editor's pickers).
#[derive(Default)]
struct SlotPrimaries<'a> {
    model: &'a str,
    fea: Option<&'a str>,
    trimesh: Option<&'a str>,
    subspace: Option<&'a str>,
}

/// Builds a fresh step's inputs, one per declared operator metadata input, in
/// order. Typed asset slots point at the matching primary; config slots get
/// a default-seeded CBOR map; Lua slots get the template; other slots get
/// empty/zeroed placeholders.
fn operator_step_inputs(
    metadata: &OperatorMetadata,
    primaries: &SlotPrimaries<'_>,
) -> Vec<ExecutionInput> {
    let primary_model = primaries.model;
    let primary_fea = primaries.fea;
    let primary_trimesh = primaries.trimesh;
    metadata
        .inputs
        .iter()
        .map(|input| match input {
            OperatorMetadataInput::ModelWASM => ExecutionInput::AssetRef(primary_model.to_string()),
            OperatorMetadataInput::CBORConfiguration(cddl) => {
                let fields = operator_config::parse_schema(cddl).unwrap_or_default();
                ExecutionInput::Inline(operator_config::encode(
                    &fields,
                    &operator_config::default_values(&fields),
                ))
            }
            OperatorMetadataInput::LuaSource(template) => {
                ExecutionInput::Inline(template.clone().into_bytes())
            }
            OperatorMetadataInput::Blob => ExecutionInput::Inline(Vec::new()),
            OperatorMetadataInput::VecF64(dim) => ExecutionInput::Inline(vec![0u8; dim * 8]),
            // With no mesh of the right kind in the project yet, the empty
            // placeholder makes the operator report a bad input at run time;
            // the step editor's picker fills it in once a producer exists.
            OperatorMetadataInput::FeaMesh => match primary_fea {
                Some(id) => ExecutionInput::AssetRef(id.to_string()),
                None => ExecutionInput::Inline(Vec::new()),
            },
            OperatorMetadataInput::TriMesh => match primary_trimesh {
                Some(id) => ExecutionInput::AssetRef(id.to_string()),
                None => ExecutionInput::Inline(Vec::new()),
            },
            OperatorMetadataInput::Subspace => match primaries.subspace {
                Some(id) => ExecutionInput::AssetRef(id.to_string()),
                None => ExecutionInput::Inline(Vec::new()),
            },
        })
        .collect()
}

/// Decodes a `VecF64` input payload (little-endian f64s); missing or short
/// bytes read as zeros so a malformed slot still yields an editable form.
fn decode_vec_f64(bytes: &[u8], dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|i| {
            bytes
                .get(i * 8..(i + 1) * 8)
                .and_then(|chunk| chunk.try_into().ok())
                .map(f64::from_le_bytes)
                .unwrap_or(0.0)
        })
        .collect()
}

fn step_depends_on_asset(step: &volumetric::ExecutionStep, asset_id: &str) -> bool {
    step.operator_id == asset_id
        || step.inputs.iter().any(|input| match input {
            ExecutionInput::AssetRef(id) => id == asset_id,
            ExecutionInput::Inline(_) => false,
        })
}

fn parse_index_route(route: &str, prefix: &str) -> Option<usize> {
    route.strip_prefix(prefix)?.parse().ok()
}

/// Parses a `{prefix}{step}:{input}:{asset_id}` route.
fn parse_step_model_route<'a>(route: &'a str, prefix: &str) -> Option<(usize, usize, &'a str)> {
    let rest = route.strip_prefix(prefix)?;
    let (step_idx, rest) = rest.split_once(':')?;
    let (input_idx, asset_id) = rest.split_once(':')?;
    Some((step_idx.parse().ok()?, input_idx.parse().ok()?, asset_id))
}

/// Splits a target resolution into (base_resolution, max_depth). A non-zero
/// `base_override` pins the stage-1 discovery grid (clamped to the target);
/// otherwise the automatic 6/8 split applies.
fn asn2_resolution_split(target_resolution: usize, base_override: usize) -> (usize, usize) {
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

fn camera_scheme_route_name(scheme: CameraControlScheme) -> &'static str {
    match scheme {
        CameraControlScheme::Blender => "blender",
        CameraControlScheme::OnShape => "onshape",
        CameraControlScheme::Fusion360 => "fusion360",
        CameraControlScheme::SolidWorks => "solidworks",
        CameraControlScheme::Maya => "maya",
    }
}

fn camera_control_scheme_from_route(name: &str) -> Option<CameraControlScheme> {
    CameraControlScheme::ALL
        .iter()
        .copied()
        .find(|scheme| camera_scheme_route_name(*scheme) == name)
}

fn camera_scheme_short_label(scheme: CameraControlScheme) -> &'static str {
    match scheme {
        CameraControlScheme::Blender => "Blender",
        CameraControlScheme::OnShape => "OnShape",
        CameraControlScheme::Fusion360 => "Fusion",
        CameraControlScheme::SolidWorks => "Solid",
        CameraControlScheme::Maya => "Maya",
    }
}

fn camera_scheme_tooltip(scheme: CameraControlScheme) -> &'static str {
    match scheme {
        CameraControlScheme::Blender => "Blender: middle orbit, Shift+middle pan, wheel zoom",
        CameraControlScheme::OnShape => "OnShape: right orbit, middle pan, wheel zoom",
        CameraControlScheme::Fusion360 => "Fusion 360: middle orbit, Shift+middle pan, wheel zoom",
        CameraControlScheme::SolidWorks => "SolidWorks: middle orbit, Ctrl+middle pan, wheel zoom",
        CameraControlScheme::Maya => "Maya: Alt+left orbit, Alt+middle pan, Alt+right/wheel zoom",
    }
}

fn editable_model_asset_ids(app: &VolumetricUiV2) -> Vec<String> {
    app.declared_assets_typed()
        .into_iter()
        .filter_map(|(id, type_hint)| match type_hint {
            Some(AssetTypeHint::Model) | None => Some(id),
            _ => None,
        })
        .collect()
}

fn editable_fea_asset_ids(app: &VolumetricUiV2) -> Vec<String> {
    app.declared_assets_typed()
        .into_iter()
        .filter_map(|(id, type_hint)| (type_hint == Some(AssetTypeHint::FeaMesh)).then_some(id))
        .collect()
}

fn editable_trimesh_asset_ids(app: &VolumetricUiV2) -> Vec<String> {
    app.declared_assets_typed()
        .into_iter()
        .filter_map(|(id, type_hint)| (type_hint == Some(AssetTypeHint::TriMesh)).then_some(id))
        .collect()
}

fn editable_subspace_asset_ids(app: &VolumetricUiV2) -> Vec<String> {
    app.declared_assets_typed()
        .into_iter()
        .filter_map(|(id, type_hint)| (type_hint == Some(AssetTypeHint::Subspace)).then_some(id))
        .collect()
}

pub fn shell_bundle(viewport: Rect) -> damascene_core::bundle::artifact::Bundle {
    let app = VolumetricUiV2::default();
    let mut tree = shell(&app);
    damascene_core::bundle::artifact::render_bundle(&mut tree, viewport)
}

/// The Add modal's browse view (category card grids) over a catalog
/// warmed synchronously from the bundled module bytes — the headless
/// review sheet for each module's declared display metadata and
/// hand-authored icon. Unlike the real Add modal, nothing is Pending and
/// no scroll area clips the grid.
pub fn catalog_sheet_bundle(viewport: Rect) -> damascene_core::bundle::artifact::Bundle {
    let mut app = VolumetricUiV2::default();
    let names: Vec<String> = app
        .catalog
        .entries()
        .iter()
        .map(|entry| entry.name.clone())
        .collect();
    for name in &names {
        let asset = volumetric_assets::get_asset(name).expect("bundled asset");
        let result = volumetric::operator_metadata_from_wasm_bytes(asset.bytes)
            .map_err(|err| err.to_string());
        app.catalog.on_metadata(name, &result);
    }
    // Overlay root so the cards' tooltips have a layer to mount on (the
    // real shell provides this; the lint flags a bare column root).
    let mut tree = overlays(column(add_browse_rows(&app)).gap(tokens::SPACE_1), []);
    damascene_core::bundle::artifact::render_bundle(&mut tree, viewport)
}

/// Like [`shell_bundle`], with the mesh-export modal open over a synthetic
/// tetrahedron — the headless-artifact view of the export dialog.
pub fn export_modal_bundle(viewport: Rect) -> damascene_core::bundle::artifact::Bundle {
    let mut app = VolumetricUiV2::default();
    let id = app
        .project
        .exports()
        .first()
        .cloned()
        .unwrap_or_else(|| "demo".to_string());
    app.open_export_dialog(&id);
    let tetrahedron = [
        [(1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0)],
        [(1.0, 1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0)],
        [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, 1.0)],
        [(-1.0, -1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0)],
    ]
    .map(volumetric::Triangle::new);
    app.set_export_mesh(&id, &tetrahedron);
    let mut tree = shell(&app);
    damascene_core::bundle::artifact::render_bundle(&mut tree, viewport)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dispatch a synthetic event with an empty `EventCx` — the app's
    /// handlers don't read event-time geometry, so tests don't need one.
    fn dispatch(app: &mut VolumetricUiV2, event: UiEvent) {
        app.on_event(event, &EventCx::new());
    }

    /// Click the Add-menu entry for the named bundled model.
    fn add_model_click(app: &mut VolumetricUiV2, name: &str) {
        dispatch(
            app,
            UiEvent::synthetic_click(format!("{ADD_MODEL_PREFIX}{name}")),
        );
    }

    /// Click the Add-menu entry for the named bundled operator, then pump
    /// the metadata round-trip the background worker performs in production
    /// (the click only queues a request; the step inserts on the result).
    fn add_operator_click(app: &mut VolumetricUiV2, name: &str) {
        dispatch(
            app,
            UiEvent::synthetic_click(format!("{ADD_OPERATOR_PREFIX}{name}")),
        );
        pump_operator_add(app);
    }

    /// Executes a queued operator-add metadata request inline and feeds the
    /// result back, standing in for the session/worker round-trip.
    fn pump_operator_add(app: &mut VolumetricUiV2) {
        if let Some(name) = app.take_operator_metadata_request() {
            let result = match volumetric_assets::get_operator(&name) {
                Some(asset) => volumetric::operator_metadata_from_wasm_bytes(asset.bytes)
                    .map_err(|err| err.to_string()),
                None => Err(format!("missing bundled operator {name}")),
            };
            app.on_module_metadata(&name, result);
        }
    }

    fn first_operator_name() -> &'static str {
        volumetric_assets::operators()
            .first()
            .expect("bundled operators")
            .name
    }

    /// The asset registry's build-time name/version must match what each
    /// operator actually declares at runtime: `bundled_operator_versions`
    /// (and with it the upgrade offer) reads the registry precisely so the
    /// UI never compiles all bundled operators just to learn their versions.
    /// Compiles every bundled module, so this is the slowest test here.
    /// Models go through the same executor read as operators — this is
    /// also the proof that the catalog scan can read every bundled module.
    #[test]
    fn bundled_asset_registry_matches_declared_metadata() {
        for asset in volumetric_assets::models()
            .iter()
            .chain(volumetric_assets::operators())
        {
            let metadata = volumetric::operator_metadata_from_wasm_bytes(asset.bytes)
                .unwrap_or_else(|e| panic!("{} metadata: {e}", asset.name));
            assert_eq!(
                metadata.name, asset.name,
                "declared metadata name must equal the crate name"
            );
            assert_eq!(
                metadata.version, asset.version,
                "{}: declared version must equal the crate version bundled at build time",
                asset.name
            );
            assert!(
                !metadata.display_name.is_empty() && !metadata.category.is_empty(),
                "{}: bundled modules must declare catalog display metadata",
                asset.name
            );
        }
    }

    /// The Add flow end-to-end through the catalog: a scanned entry shows
    /// its declared display name, and a warm-catalog operator click inserts
    /// its step synchronously (no background metadata request).
    #[test]
    fn warm_catalog_names_entries_and_inserts_synchronously() {
        let mut app = VolumetricUiV2::default();
        assert_eq!(
            app.catalog.display_name("simple_sphere_model"),
            "simple_sphere_model",
            "unscanned entries fall back to the module name"
        );

        // Pump the first scan the way the session worker would. Entry 0 is
        // a model, so this also exercises the model get_metadata path.
        let name = app.catalog.take_scan_request().expect("pending entries");
        let asset = volumetric_assets::get_asset(&name).expect("bundled");
        let result =
            volumetric::operator_metadata_from_wasm_bytes(asset.bytes).map_err(|e| e.to_string());
        app.on_module_metadata(&name, result);
        assert_eq!(
            app.catalog.display_name("simple_sphere_model"),
            "Simple Sphere"
        );

        // Warm the catalog entry for a generator operator by hand, then
        // click it: the step must insert without queueing a worker read.
        let prism = "rectangular_prism_operator";
        let metadata = volumetric::operator_metadata_from_wasm_bytes(
            volumetric_assets::get_operator(prism)
                .expect("bundled")
                .bytes,
        )
        .expect("prism metadata");
        app.on_module_metadata(prism, Ok(metadata));
        let mut app = {
            let mut fresh = VolumetricUiV2::empty();
            std::mem::swap(&mut fresh.catalog, &mut app.catalog);
            fresh
        };
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{ADD_OPERATOR_PREFIX}{prism}")),
        );
        assert_eq!(app.summary().timeline_steps, 1, "{}", app.status);
        assert!(
            app.take_operator_metadata_request().is_none(),
            "warm click must not queue a background read"
        );
    }

    /// A generator operator (no model inputs) inserts into an empty project
    /// — the guard only demands a model when the operator declares a model
    /// slot. This is the fresh-session repro: new session, Add menu,
    /// Rectangular Prism.
    #[test]
    fn generator_operator_adds_without_a_selection() {
        let mut app = VolumetricUiV2::empty();
        assert!(app.selected_export.is_none());
        add_operator_click(&mut app, "rectangular_prism_operator");
        assert_eq!(app.summary().timeline_steps, 1, "{}", app.status);
        assert!(app.status.starts_with("added"), "{}", app.status);
    }

    /// An operator with a model slot still refuses to insert into a project
    /// with no model to wire it to.
    #[test]
    fn model_operator_refused_without_any_model() {
        let mut app = VolumetricUiV2::empty();
        add_operator_click(&mut app, "translate_operator");
        assert_eq!(app.summary().timeline_steps, 0);
        assert!(app.status.contains("needs a model input"), "{}", app.status);
    }

    /// A second Add click while the first's metadata read is still on the
    /// worker is refused instead of queueing a competing request.
    #[test]
    fn concurrent_operator_adds_are_serialized() {
        let mut app = VolumetricUiV2::default();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{ADD_OPERATOR_PREFIX}rectangular_prism_operator")),
        );
        let name = app.take_operator_metadata_request().expect("queued");
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{ADD_OPERATOR_PREFIX}translate_operator")),
        );
        assert!(app.status.contains("still adding"), "{}", app.status);
        assert!(app.take_operator_metadata_request().is_none());

        let asset = volumetric_assets::get_operator(&name).unwrap();
        let result =
            volumetric::operator_metadata_from_wasm_bytes(asset.bytes).map_err(|e| e.to_string());
        app.on_module_metadata(&name, result);
        assert_eq!(app.summary().timeline_steps, 1);
    }

    #[test]
    fn shell_reserves_a_viewport_region() {
        let bundle = shell_bundle(Rect::new(0.0, 0.0, 1280.0, 800.0));
        assert!(bundle.tree_dump.contains(VIEWPORT_KEY));
        assert!(bundle.lint.findings.is_empty(), "{}", bundle.lint.text());
    }

    #[test]
    fn shell_root_is_overlay_for_runtime_tooltips() {
        let shell = shell(&VolumetricUiV2::default());
        assert_eq!(shell.axis, Axis::Overlay);
    }

    #[test]
    fn default_app_starts_with_a_model_project() {
        let app = VolumetricUiV2::default();
        let summary = app.summary();
        assert_eq!(summary.imports, 1);
        assert_eq!(summary.exports, 1);
        assert!(summary.selected_export.is_some());
    }

    #[test]
    fn invalid_project_blocks_run_dispatch_with_diagnostics() {
        let mut app = VolumetricUiV2::default();
        // Break the project: export an id nothing defines.
        app.project.exports_mut().push("ghost".to_string());

        app.request_run();
        assert!(!app.take_pending_run(), "invalid project must not dispatch");
        let err = app.last_run_error.as_deref().expect("run error set");
        assert!(err.contains("ghost"), "diagnostic names the bad id: {err}");

        // A sound project still dispatches.
        app.project.exports_mut().retain(|id| id != "ghost");
        app.request_run();
        assert!(app.take_pending_run());
    }

    #[test]
    fn add_menu_click_appends_unique_import() {
        let mut app = VolumetricUiV2::default();
        add_model_click(&mut app, "simple_sphere_model");

        let summary = app.summary();
        assert_eq!(summary.imports, 2);
        assert_eq!(summary.exports, 2);
        assert_eq!(app.project().exports()[0], "simple_sphere_model");
        assert_eq!(app.project().exports()[1], "simple_sphere_model_2");
    }

    #[test]
    fn menubar_opens_and_menu_item_click_closes_it() {
        let mut app = VolumetricUiV2::default();
        dispatch(&mut app, UiEvent::synthetic_click("main-menu:menu:file"));
        assert_eq!(app.open_menu.as_deref(), Some("file"));

        // A menu item click acts and closes the menu.
        dispatch(&mut app, UiEvent::synthetic_click(NEW_PROJECT_KEY));
        assert_eq!(app.open_menu, None);

        // Clicking the trigger again toggles closed too.
        dispatch(&mut app, UiEvent::synthetic_click("main-menu:menu:file"));
        dispatch(&mut app, UiEvent::synthetic_click("main-menu:menu:file"));
        assert_eq!(app.open_menu, None);

        // The Add affordance opens the modal, not a menu.
        dispatch(&mut app, UiEvent::synthetic_click(ADD_OPEN_KEY));
        assert!(app.add_modal.is_some());
        assert_eq!(app.open_menu, None);
    }

    #[test]
    fn add_operator_click_appends_timeline_step() {
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, first_operator_name());

        let summary = app.summary();
        assert_eq!(summary.timeline_steps, 1);
        assert_eq!(summary.exports, 2);
        assert_eq!(
            summary.selected_project_item,
            Some(ProjectSelection::Step(0))
        );
    }

    #[test]
    fn added_operator_wires_all_declared_inputs() {
        let mut app = VolumetricUiV2::default();
        let op_name = first_operator_name();
        add_operator_click(&mut app, op_name);

        let asset = volumetric_assets::get_operator(op_name).expect("bundled operator");
        let metadata =
            volumetric::operator_metadata_from_wasm_bytes(asset.bytes).expect("operator metadata");

        let step = &app.project().timeline()[0];
        assert_eq!(
            step.inputs.len(),
            metadata.inputs.len(),
            "one step input per declared metadata input"
        );
        for (input, meta) in step.inputs.iter().zip(&metadata.inputs) {
            match meta {
                OperatorMetadataInput::ModelWASM => {
                    assert!(matches!(input, ExecutionInput::AssetRef(_)));
                }
                OperatorMetadataInput::CBORConfiguration(cddl) => {
                    let ExecutionInput::Inline(bytes) = input else {
                        panic!("config input should be inline CBOR");
                    };
                    // The default config decodes and covers every required
                    // schema field; optional fields start unset so the
                    // operator's absent-field behavior applies.
                    let fields = operator_config::parse_schema(cddl).unwrap_or_default();
                    let decoded = operator_config::decode(bytes);
                    for field in &fields {
                        if field.optional {
                            assert!(
                                !decoded.contains_key(&field.name),
                                "optional {} must start unset",
                                field.name
                            );
                        } else {
                            assert!(decoded.contains_key(&field.name), "missing {}", field.name);
                        }
                    }
                }
                _ => assert!(matches!(input, ExecutionInput::Inline(_))),
            }
        }
    }

    /// The app-supplied SVG glyphs (names damascene's built-in vocabulary
    /// lacks) must parse — a bad path would otherwise panic at first render.
    #[test]
    fn custom_icons_parse() {
        let _ = &*EYE_ICON;
        let _ = &*PIN_ICON;
    }

    #[test]
    fn step_editor_labels_slots_with_declared_names() {
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, "boolean_operator");
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SELECT_STEP_PREFIX}0")),
        );
        app.before_build();

        let edit = app.step_edit.as_ref().expect("step editor state");
        let names: Vec<Option<&str>> = edit
            .asset_slots
            .iter()
            .map(|slot| slot.name.as_deref())
            .collect();
        assert_eq!(names, vec![Some("Model A"), Some("Model B")]);
    }

    /// A minimal operator (WAT text — the executor compiles it directly)
    /// whose metadata mimics a bundled operator at a different version.
    fn stale_operator_wasm(name: &str, version: &str) -> Vec<u8> {
        let metadata = volumetric::encode_metadata(&OperatorMetadata {
            name: name.to_string(),
            version: version.to_string(),
            display_name: String::new(),
            description: String::new(),
            category: String::new(),
            icon_svg: String::new(),
            inputs: vec![OperatorMetadataInput::ModelWASM],
            input_names: vec!["Model".to_string()],
            outputs: vec![volumetric::OperatorMetadataOutput::ModelWASM],
        });
        let data: String = metadata.iter().map(|b| format!("\\{b:02x}")).collect();
        let packed = 1024_i64 | ((metadata.len() as i64) << 32);
        format!(
            r#"(module
                (memory (export "memory") 1)
                (data (i32.const 1024) "{data}")
                (func (export "get_metadata") (result i64) (i64.const {packed}))
                (func (export "run")))"#
        )
        .into_bytes()
    }

    #[test]
    fn stale_operator_import_offers_and_applies_upgrade() {
        let mut app = VolumetricUiV2::empty();
        app.project
            .imports_mut()
            .push(volumetric::ImportedAsset::operator(
                "translate_operator".to_string(),
                stale_operator_wasm("translate_operator", "0.0.1"),
            ));
        app.project.timeline_mut().push(volumetric::ExecutionStep {
            operator_id: "translate_operator".to_string(),
            // One input where the bundled build declares two (model+config):
            // the upgrade must append the missing slot.
            inputs: vec![ExecutionInput::AssetRef("part".to_string())],
            outputs: vec!["moved".to_string()],
        });

        let bundled_version = bundled_operator_versions()
            .get("translate_operator")
            .expect("bundled translate_operator decodes")
            .clone();
        let offer = operator_upgrade_offer(&app, &app.project.imports()[0]);
        assert_eq!(offer, Some(("0.0.1".to_string(), bundled_version)));

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{UPGRADE_IMPORT_PREFIX}0")),
        );

        let bundled = volumetric_assets::get_operator("translate_operator").unwrap();
        let import = &app.project().imports()[0];
        assert_eq!(import.data, bundled.bytes);
        assert!(
            operator_upgrade_offer(&app, import).is_none(),
            "up-to-date import offers nothing"
        );
        let step = &app.project().timeline()[0];
        assert_eq!(step.inputs.len(), 2, "inputs realigned to new arity");
        assert!(matches!(
            &step.inputs[0],
            ExecutionInput::AssetRef(id) if id == "part"
        ));
    }

    #[test]
    fn vec_inputs_round_trip_through_the_step_editor() {
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, "rectangular_prism_operator");
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SELECT_STEP_PREFIX}0")),
        );
        app.before_build();

        let edit = app.step_edit.as_ref().expect("step editor state");
        assert_eq!(edit.vecs.len(), 2, "prism declares two VecF64 inputs");
        assert_eq!(edit.vecs[0].buffers, vec!["0", "0", "0"]);
        let input_idx = edit.vecs[0].input_idx;

        app.commit_vec_component(0, input_idx, 1, "2.5");
        let step = &app.project().timeline()[0];
        let ExecutionInput::Inline(bytes) = &step.inputs[input_idx] else {
            panic!("VecF64 slot should stay inline");
        };
        assert_eq!(f64::from_le_bytes(bytes[8..16].try_into().unwrap()), 2.5);
        assert_eq!(f64::from_le_bytes(bytes[0..8].try_into().unwrap()), 0.0);

        // A rebuilt editor decodes the committed bytes back into buffers.
        app.step_edit = app.build_step_edit(0);
        let edit = app.step_edit.as_ref().expect("step editor state");
        assert_eq!(edit.vecs[0].buffers[1], "2.5");

        // Unparseable text leaves the slot untouched.
        app.commit_vec_component(0, input_idx, 1, "not a number");
        let step = &app.project().timeline()[0];
        let ExecutionInput::Inline(bytes) = &step.inputs[input_idx] else {
            panic!("VecF64 slot should stay inline");
        };
        assert_eq!(f64::from_le_bytes(bytes[8..16].try_into().unwrap()), 2.5);
    }

    /// Input pickers must only offer assets that exist before the step
    /// runs — not the step's own outputs, not downstream outputs.
    #[test]
    fn step_input_picker_excludes_own_and_downstream_outputs() {
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, "translate_operator");
        add_operator_click(&mut app, "translate_operator");

        let step0_out = app.project().timeline()[0].outputs[0].clone();
        let step1_out = app.project().timeline()[1].outputs[0].clone();
        assert_ne!(step0_out, step1_out);

        let options0 = step_input_options(&app, 0, AssetSlotKind::Model);
        assert!(!options0.is_empty(), "imports remain available");
        assert!(!options0.contains(&step0_out), "own output excluded");
        assert!(!options0.contains(&step1_out), "downstream output excluded");

        let options1 = step_input_options(&app, 1, AssetSlotKind::Model);
        assert!(options1.contains(&step0_out), "upstream output available");
        assert!(!options1.contains(&step1_out), "own output excluded");
    }

    #[test]
    fn delete_import_removes_dependent_step_and_exports() {
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, first_operator_name());
        dispatch(
            &mut app,
            UiEvent::synthetic_click("project:delete-import:0"),
        );

        let summary = app.summary();
        assert_eq!(summary.imports, 1);
        assert_eq!(summary.timeline_steps, 0);
        assert_eq!(summary.exports, 0);
    }

    #[test]
    fn delete_export_keeps_declared_asset_available() {
        let mut app = VolumetricUiV2::default();
        dispatch(
            &mut app,
            UiEvent::synthetic_click("project:delete-export:0"),
        );

        let summary = app.summary();
        assert_eq!(summary.imports, 1);
        assert_eq!(summary.exports, 0);
        assert!(
            app.project()
                .declared_assets()
                .iter()
                .any(|(id, _)| id == "simple_sphere_model")
        );
    }

    #[test]
    fn viewport_pickers_update_preview_settings() {
        let mut app = VolumetricUiV2::default();
        // Trigger click opens the picker; option click applies and closes it.
        dispatch(&mut app, UiEvent::synthetic_click(MODE_SELECT_KEY));
        assert_eq!(app.open_select.as_deref(), Some(MODE_SELECT_KEY));
        dispatch(
            &mut app,
            UiEvent::synthetic_click("view:mode:option:points"),
        );
        assert_eq!(app.open_select, None);
        dispatch(&mut app, UiEvent::synthetic_click("view:res:option:96"));
        dispatch(
            &mut app,
            UiEvent::synthetic_click("view:camera:option:onshape"),
        );

        let summary = app.summary();
        assert_eq!(summary.render_mode, PreviewRenderMode::Points);
        assert_eq!(summary.preview_resolution, 96);
        assert_eq!(summary.camera_control_scheme, CameraControlScheme::OnShape);
    }

    #[test]
    fn viewport_toggles_are_controlled_by_app_state() {
        let mut app = VolumetricUiV2::default();
        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_GRID_KEY));
        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_SSAO_KEY));
        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_BOUNDS_KEY));

        let summary = app.summary();
        assert!(!summary.show_grid);
        assert!(!summary.ssao);
        assert!(summary.show_bounds, "bounds overlay defaults off");
    }

    #[test]
    fn dimension_formatting_is_compact() {
        assert_eq!(
            format_dims((-1.0, 0.0, 0.5), (1.0, 0.25, 10.625)),
            "2 × 0.25 × 10.125"
        );
        assert_eq!(format_dim(0.1234567), "0.123");
        assert_eq!(format_dim(100.0), "100");
    }

    #[test]
    fn frame_preview_action_queues_camera_command() {
        let mut app = VolumetricUiV2::default();

        dispatch(&mut app, UiEvent::synthetic_click(FRAME_PREVIEW_KEY));

        assert_eq!(
            app.take_camera_command(),
            Some(ViewportCameraCommand::FramePreview)
        );
        assert_eq!(app.status, "framing preview");
        assert_eq!(app.take_camera_command(), None);
    }

    #[test]
    fn run_project_action_materializes_runtime_exports() {
        let mut app = VolumetricUiV2::default();
        app.run_project();

        let summary = app.summary();
        assert_eq!(summary.runtime_assets.len(), 1);
        assert_eq!(summary.runtime_assets[0].id, "simple_sphere_model");
        assert!(summary.last_run_error.is_none());
        assert!(!summary.last_run_stale);
        assert_eq!(
            app.selected_runtime_asset().map(LoadedAsset::id),
            Some("simple_sphere_model")
        );
    }

    #[test]
    fn preview_requests_carry_render_mode_and_controls() {
        let mut app = VolumetricUiV2::default();
        app.run_project();
        dispatch(&mut app, UiEvent::synthetic_click("view:mode:option:asn2"));
        dispatch(&mut app, UiEvent::synthetic_click("view:res:option:96"));

        let requests = app.preview_requests();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.asset_id, "simple_sphere_model");
        let PreviewPlan::Model3d {
            mesh: mesh_plan, ..
        } = &request.plan
        else {
            panic!(
                "sphere output should carry a 3D plan, got {:?}",
                request.plan
            );
        };
        assert_eq!(
            *mesh_plan,
            PreviewMeshPlan::AdaptiveSurfaceNets2 {
                target_resolution: 96,
                base_resolution: 6,
                max_depth: 4,
                settings: Asn2Settings::default(),
            }
        );
        let config = mesh_plan
            .adaptive_surface_nets_config()
            .expect("asn2 config");
        assert_eq!(config.base_resolution, 6);
        assert_eq!(config.max_depth, 4);
        assert!(!request.show_bounds, "bounds overlay defaults off");

        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_BOUNDS_KEY));
        assert!(app.preview_requests()[0].show_bounds);
    }

    /// Builds a two-export project and returns (app, export ids) after a run.
    fn two_export_app() -> (VolumetricUiV2, Vec<String>) {
        let mut app = VolumetricUiV2::default();
        app.add_model("simple_sphere_model"); // second copy; unique id via `insert_model`
        app.run_project();
        let exports = app.project().exports().to_vec();
        assert_eq!(exports.len(), 2);
        (app, exports)
    }

    #[test]
    fn selecting_an_export_renders_only_that_output() {
        let (mut app, exports) = two_export_app();

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SELECT_EXPORT_PREFIX}0")),
        );
        let ids: Vec<_> = app
            .preview_requests()
            .into_iter()
            .map(|r| r.asset_id)
            .collect();
        assert_eq!(ids, vec![exports[0].clone()]);

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SELECT_EXPORT_PREFIX}1")),
        );
        let ids: Vec<_> = app
            .preview_requests()
            .into_iter()
            .map(|r| r.asset_id)
            .collect();
        assert_eq!(ids, vec![exports[1].clone()]);
    }

    #[test]
    fn pinning_keeps_an_output_visible_across_selection() {
        let (mut app, exports) = two_export_app();

        // Pin the first export, then view the second: both should render.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{}", exports[0])),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SELECT_EXPORT_PREFIX}1")),
        );

        let ids: std::collections::BTreeSet<_> = app
            .preview_requests()
            .into_iter()
            .map(|r| r.asset_id)
            .collect();
        assert!(ids.contains(&exports[0]), "pinned output stays visible");
        assert!(ids.contains(&exports[1]), "selected output visible");
        assert_eq!(ids.len(), 2);

        // Unpinning drops it back to just the selection.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{}", exports[0])),
        );
        let ids: Vec<_> = app
            .preview_requests()
            .into_iter()
            .map(|r| r.asset_id)
            .collect();
        assert_eq!(ids, vec![exports[1].clone()]);
    }

    #[test]
    fn run_click_queues_a_run_request_without_executing() {
        let mut app = VolumetricUiV2::default();
        dispatch(&mut app, UiEvent::synthetic_click(RUN_PROJECT_KEY));

        // The click only requests a run; execution happens on the host worker.
        assert!(app.summary().runtime_assets.is_empty());
        assert!(app.take_pending_run());
        // The request is one-shot.
        assert!(!app.take_pending_run());
    }

    #[test]
    fn apply_run_result_materializes_exports_off_thread() {
        let mut app = VolumetricUiV2::default();
        dispatch(&mut app, UiEvent::synthetic_click(RUN_PROJECT_KEY));
        assert!(app.take_pending_run());
        app.set_run_state(RunState::Running);
        assert_eq!(app.run_state(), RunState::Running);

        // Simulate the worker executing and reporting back.
        let assets = app
            .project()
            .run(&mut volumetric::Environment::new())
            .expect("project run");
        app.apply_run_result(Ok(assets), 42);

        let summary = app.summary();
        assert_eq!(summary.run_state, RunState::Idle);
        assert_eq!(summary.runtime_assets.len(), 1);
        assert_eq!(summary.last_run_elapsed_ms, Some(42));
        assert!(!summary.last_run_stale);
    }

    #[test]
    fn cancel_click_requests_cancellation() {
        let mut app = VolumetricUiV2::default();
        app.set_run_state(RunState::Running);
        dispatch(&mut app, UiEvent::synthetic_click(CANCEL_RUN_KEY));

        assert!(app.take_cancel_request());
        assert!(!app.take_cancel_request());

        app.on_run_cancelled();
        assert_eq!(app.run_state(), RunState::Idle);
        assert!(app.summary().last_run_stale);
    }

    /// The mesh-build controls mirror the run controls: a cancel click and a
    /// remesh click each arm a one-shot host hook, and the auto-remesh chip
    /// toggles the policy the preview sync consults.
    #[test]
    fn mesh_control_clicks_arm_cancel_remesh_and_auto_toggle() {
        let mut app = VolumetricUiV2::default();
        assert!(app.auto_remesh(), "auto-remesh defaults on");

        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_AUTO_REMESH_KEY));
        assert!(!app.auto_remesh());
        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_AUTO_REMESH_KEY));
        assert!(app.auto_remesh());

        dispatch(&mut app, UiEvent::synthetic_click(CANCEL_MESH_KEY));
        assert!(app.take_mesh_cancel_request());
        assert!(!app.take_mesh_cancel_request(), "one-shot");

        dispatch(&mut app, UiEvent::synthetic_click(REMESH_KEY));
        assert!(app.take_remesh_request());
        assert!(!app.take_remesh_request(), "one-shot");
    }

    #[test]
    fn auto_rebuild_queues_a_run_on_project_edit() {
        let mut app = VolumetricUiV2::default();
        assert!(!app.auto_rebuild());
        // Editing without auto-rebuild does not queue a run.
        add_model_click(&mut app, "simple_sphere_model");
        assert!(!app.take_pending_run());

        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_AUTO_REBUILD_KEY));
        assert!(app.auto_rebuild());
        add_model_click(&mut app, "simple_sphere_model");
        assert!(app.take_pending_run());
    }

    #[test]
    fn remote_toggle_requests_an_executor_swap_and_rebuild() {
        let mut app = VolumetricUiV2::default();
        assert!(app.take_executor_request().is_none());

        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_REMOTE_BUILD_KEY));
        assert!(app.remote_build());
        let choice = app.take_executor_request().expect("swap requested");
        assert!(matches!(choice, ExecutorChoice::Remote(addr) if addr.starts_with("http://")));
        // In-flight work is written off and a fresh run + remesh queued for
        // the new executor.
        assert!(app.take_cancel_request());
        assert!(app.take_mesh_cancel_request());
        assert!(app.take_remesh_request());
        assert!(app.take_pending_run());

        dispatch(&mut app, UiEvent::synthetic_click(TOGGLE_REMOTE_BUILD_KEY));
        assert!(!app.remote_build());
        assert_eq!(app.take_executor_request(), Some(ExecutorChoice::Local));
    }

    #[test]
    fn project_edit_keeps_stale_runtime_until_rerun() {
        let mut app = VolumetricUiV2::default();
        app.run_project();
        assert_eq!(app.summary().runtime_assets.len(), 1);

        add_model_click(&mut app, "simple_sphere_model");

        let summary = app.summary();
        // The previous run's output stays materialized (marked stale) so the
        // viewport keeps its last good preview until a new run replaces it.
        assert_eq!(summary.runtime_assets.len(), 1);
        assert!(summary.last_run_stale);
        assert!(summary.last_run_error.is_none());
        // Auto-rebuild is off by default, so no run was queued.
        assert!(!app.take_pending_run());
    }

    #[test]
    fn pinned_output_survives_a_stale_edit() {
        let (mut app, exports) = two_export_app();

        // Pin the first export so it stays in the viewport independent of
        // selection, then make an edit that doesn't rerun.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{}", exports[0])),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click("project:delete-export:1"),
        );

        let summary = app.summary();
        assert!(summary.last_run_stale);
        // The pinned output's runtime asset is untouched until a rerun, so it
        // keeps rendering (flagged stale) even though the edit cleared selection.
        let requests = app.preview_requests();
        assert!(
            requests.iter().any(|r| r.asset_id == exports[0] && r.stale),
            "pinned output stays on screen, stale"
        );
    }

    #[test]
    fn per_output_override_changes_only_that_output() {
        let (mut app, exports) = two_export_app();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{}", exports[0])),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{}", exports[1])),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_MODE_PREFIX}{}:points", exports[0])),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_RESOLUTION_PREFIX}{}:24", exports[0])),
        );

        let requests = app.preview_requests();
        let overridden = requests
            .iter()
            .find(|r| r.asset_id == exports[0])
            .expect("overridden output renders");
        let default = requests
            .iter()
            .find(|r| r.asset_id == exports[1])
            .expect("default output renders");
        assert_eq!(
            overridden.plan,
            PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::for_mode(
                    PreviewRenderMode::Points,
                    24,
                    Asn2Settings::default()
                ),
                color_channel: None,
            }
        );
        assert!(
            matches!(
                &default.plan,
                PreviewPlan::Model3d {
                    mesh: PreviewMeshPlan::AdaptiveSurfaceNets2 { .. },
                    ..
                }
            ),
            "un-overridden output keeps the viewport default plan"
        );
    }

    #[test]
    fn ssao_steppers_and_camera_reset() {
        let mut app = VolumetricUiV2::default();
        dispatch(&mut app, UiEvent::synthetic_click(SSAO_SETTINGS_KEY));
        assert_eq!(app.open_select.as_deref(), Some(SSAO_SETTINGS_KEY));
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SSAO_ADJUST_PREFIX}radius:down")),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SSAO_ADJUST_PREFIX}strength:up")),
        );
        // Steppers keep the popover open.
        assert_eq!(app.open_select.as_deref(), Some(SSAO_SETTINGS_KEY));

        app.run_project();
        let requests = app.preview_requests();
        let request = &requests[0];
        assert!((request.ssao_radius - 0.5 / 1.5).abs() < 1e-6);
        assert!((request.ssao_strength - 1.25).abs() < 1e-6);

        dispatch(&mut app, UiEvent::synthetic_click(RESET_CAMERA_KEY));
        assert_eq!(
            app.take_camera_command(),
            Some(ViewportCameraCommand::Reset)
        );
    }

    #[test]
    fn asn2_steppers_adjust_output_settings() {
        let (mut app, exports) = two_export_app();
        let id = exports[0].clone();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{id}")),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:vr:down")),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:angle:up")),
        );

        let OutputRender::Model3d { asn2, .. } = app.output_render(&id) else {
            panic!("model output should carry 3D settings");
        };
        // Sharp edges default on for 3D model outputs; the toggle inverts.
        assert!(asn2.sharp_edges);
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:sharp:up")),
        );
        let OutputRender::Model3d { asn2: toggled, .. } = app.output_render(&id) else {
            panic!("model output should carry 3D settings");
        };
        assert!(!toggled.sharp_edges);
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:sharp:up")),
        );
        // Edge-constrained refinement defaults off; the toggle inverts.
        assert!(!asn2.edge_constrained_refinement);
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:edgec:up")),
        );
        // Simplify defaults on; toggling twice round-trips, and the
        // tolerance stepper moves in tenths of a cell.
        assert!(asn2.simplify);
        assert_eq!(asn2.simplify_tolerance_tenths, 10);
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:simp:up")),
        );
        let OutputRender::Model3d { asn2: toggled, .. } = app.output_render(&id) else {
            panic!("model output should carry 3D settings");
        };
        assert!(!toggled.simplify);
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:simp:up")),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:simptol:up")),
        );
        // Discovery probes default to 8; the base grid steps auto -> 8 -> 12.
        assert_eq!(asn2.discovery_probes, 8);
        assert_eq!(asn2.base_resolution, 0);
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:probes:up")),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:base:up")),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_ASN2_PREFIX}{id}:base:up")),
        );
        assert_eq!(asn2.vertex_refinement_iterations, 7);
        assert_eq!(asn2.sharp_angle_degrees, 20);

        // The settings flow through to the meshing config.
        let requests = app.preview_requests();
        let request = requests.iter().find(|r| r.asset_id == id).unwrap();
        let PreviewPlan::Model3d {
            mesh: mesh_plan, ..
        } = &request.plan
        else {
            panic!("model output should carry a 3D plan");
        };
        let config = mesh_plan
            .adaptive_surface_nets_config()
            .expect("asn2 config");
        assert_eq!(config.vertex_refinement_iterations, 7);
        assert!(config.edge_constrained_refinement);
        assert_eq!(config.discovery_probes, 9);
        assert_eq!(
            config.base_resolution, 12,
            "base override pins the discovery grid"
        );
        let sharp = config.sharp_features.expect("sharp features enabled");
        assert!((sharp.segmentation.max_normal_jump_deg - 20.0).abs() < 1e-9);
        let decimation = config.decimation.expect("simplify enabled");
        assert!((decimation.error_tolerance_cells - 1.1).abs() < 1e-9);
    }

    #[test]
    fn wireframe_toggle_is_display_only() {
        let (mut app, exports) = two_export_app();
        let id = exports[0].clone();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{id}")),
        );
        let plan_before = app
            .preview_requests()
            .iter()
            .find(|r| r.asset_id == id)
            .unwrap()
            .plan
            .clone();

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_WIREFRAME_PREFIX}{id}")),
        );
        assert!(app.output_render(&id).wireframe());
        let requests = app.preview_requests();
        let request = requests.iter().find(|r| r.asset_id == id).unwrap();
        assert!(request.wireframe);
        // The plan (the rebuild cache key) is untouched by the toggle.
        assert_eq!(request.plan, plan_before);

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_WIREFRAME_PREFIX}{id}")),
        );
        assert!(!app.output_render(&id).wireframe());
    }

    #[test]
    fn channel_route_overrides_model3d_colormap() {
        let (mut app, exports) = two_export_app();
        let id = exports[0].clone();

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_CHANNEL_PREFIX}{id}:density")),
        );
        let OutputRender::Model3d { color_channel, .. } = app.output_render(&id) else {
            panic!("model output should carry 3D settings");
        };
        assert_eq!(color_channel.as_deref(), Some("density"));
        // The channel is part of the rebuild plan (cache key).
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{id}")),
        );
        let requests = app.preview_requests();
        let request = requests.iter().find(|r| r.asset_id == id).unwrap();
        assert!(matches!(
            &request.plan,
            PreviewPlan::Model3d { color_channel: Some(name), .. } if name == "density"
        ));

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_CHANNEL_PREFIX}{id}:none")),
        );
        let OutputRender::Model3d { color_channel, .. } = app.output_render(&id) else {
            panic!("model output should carry 3D settings");
        };
        assert_eq!(color_channel, None);
    }

    #[test]
    fn lightbox_slice_controls_update_mode_and_resample() {
        let (mut app, exports) = two_export_app();
        let id = exports[0].clone();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_INSPECT_PREFIX}{id}")),
        );
        // A 3D model output opens in slice mode: z midplane, occupancy
        // fallback channel (no preview stats have arrived in this test).
        let lightbox = app.lightbox.as_ref().expect("lightbox opens");
        assert_eq!(
            lightbox.mode,
            LightboxMode::Slice {
                axis: 2,
                frac_percent: 50,
                channel: "occupancy".to_string(),
            }
        );

        // Pretend data arrived, then change the axis: the data drops so the
        // session re-samples with the new mode.
        app.lightbox.as_mut().unwrap().data = Some(LightboxData {
            rgba: vec![0; 4],
            width: 1,
            height: 1,
            bounds_min: (0.0, 0.0),
            bounds_max: (1.0, 1.0),
            binary: true,
            value_min: 0.0,
            value_max: 1.0,
            analytics: vec![],
        });
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{LIGHTBOX_SLICE_PREFIX}axis:0")),
        );
        let lightbox = app.lightbox.as_ref().unwrap();
        assert!(lightbox.data.is_none(), "axis change drops stale data");
        assert!(matches!(
            &lightbox.mode,
            LightboxMode::Slice { axis: 0, .. }
        ));

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{LIGHTBOX_SLICE_PREFIX}pos:up")),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{LIGHTBOX_SLICE_PREFIX}channel:density")),
        );
        assert_eq!(
            app.lightbox.as_ref().unwrap().mode,
            LightboxMode::Slice {
                axis: 0,
                frac_percent: 55,
                channel: "density".to_string(),
            }
        );

        // Stale-mode results are dropped: data delivered for the old mode
        // leaves the lightbox waiting.
        app.set_lightbox_data(
            &id,
            &LightboxMode::Slice {
                axis: 2,
                frac_percent: 50,
                channel: "occupancy".to_string(),
            },
            Ok(LightboxData {
                rgba: vec![0; 4],
                width: 1,
                height: 1,
                bounds_min: (0.0, 0.0),
                bounds_max: (1.0, 1.0),
                binary: true,
                value_min: 0.0,
                value_max: 1.0,
                analytics: vec![],
            }),
        );
        assert!(app.lightbox.as_ref().unwrap().data.is_none());
    }

    #[test]
    fn inspect_route_opens_and_dismisses_the_lightbox() {
        let (mut app, exports) = two_export_app();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_INSPECT_PREFIX}{}", exports[0])),
        );
        let lightbox = app.lightbox.as_ref().expect("lightbox opens");
        assert_eq!(lightbox.asset_id, exports[0]);
        // The freshly opened lightbox asks for sampled data for its output.
        let (id, _, _) = app.lightbox_wants_data().expect("wants data");
        assert_eq!(id, exports[0]);
        assert!(app.lightbox_wants_texture().is_none(), "no data yet");

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{LIGHTBOX_KEY}:dismiss")),
        );
        assert!(app.lightbox.is_none());
    }

    #[test]
    fn output_defaults_click_clears_override() {
        let (mut app, exports) = two_export_app();
        // Pin it so it participates in preview_requests.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{}", exports[0])),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_MODE_PREFIX}{}:points", exports[0])),
        );
        assert!(app.output_overrides.contains_key(&exports[0]));

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_DEFAULTS_PREFIX}{}", exports[0])),
        );
        assert!(app.output_overrides.is_empty());
        let requests = app.preview_requests();
        let request = requests
            .iter()
            .find(|r| r.asset_id == exports[0])
            .expect("output renders");
        assert!(
            matches!(
                &request.plan,
                PreviewPlan::Model3d {
                    mesh: PreviewMeshPlan::AdaptiveSurfaceNets2 { .. },
                    ..
                }
            ),
            "back on viewport defaults"
        );
    }

    #[test]
    fn output_override_pruned_when_output_disappears() {
        let (mut app, exports) = two_export_app();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_MODE_PREFIX}{}:points", exports[1])),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click("project:delete-export:1"),
        );
        app.run_project();
        assert!(!app.output_overrides.contains_key(&exports[1]));
    }

    #[test]
    fn output_settings_popover_opens_and_survives_picks() {
        let (mut app, exports) = two_export_app();
        let key = format!("{OUTPUT_SETTINGS_PREFIX}{}", exports[0]);
        dispatch(&mut app, UiEvent::synthetic_click(key.clone()));
        assert_eq!(app.open_select.as_deref(), Some(key.as_str()));

        // A pick adjusts the override but keeps the settings panel open.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_MODE_PREFIX}{}:points", exports[0])),
        );
        assert_eq!(app.open_select.as_deref(), Some(key.as_str()));

        // Outside click (popover scrim) dismisses.
        dispatch(&mut app, UiEvent::synthetic_click(format!("{key}:dismiss")));
        assert_eq!(app.open_select, None);
    }

    #[test]
    fn file_menu_clicks_queue_file_actions_for_the_host() {
        let mut app = VolumetricUiV2::default();
        dispatch(&mut app, UiEvent::synthetic_click("main-menu:menu:file"));
        dispatch(&mut app, UiEvent::synthetic_click(OPEN_PROJECT_KEY));
        assert_eq!(app.take_file_action(), Some(FileAction::OpenProject));
        assert_eq!(app.open_menu, None, "menu closes on item click");
        assert_eq!(app.take_file_action(), None, "one-shot");

        dispatch(&mut app, UiEvent::synthetic_click(SAVE_PROJECT_KEY));
        assert_eq!(app.take_file_action(), Some(FileAction::SaveProject));
    }

    #[test]
    fn export_mesh_click_opens_modal_and_confirm_queues_action() {
        let (mut app, exports) = two_export_app();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_SETTINGS_PREFIX}{}", exports[0])),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{EXPORT_MESH_PREFIX}{}", exports[0])),
        );
        assert_eq!(app.open_select, None, "settings popover closes");
        assert_eq!(
            app.export_dialog_wants_mesh(),
            Some(exports[0].as_str()),
            "modal opens pending the session's mesh delivery"
        );
        assert_eq!(app.take_file_action(), None, "opening exports nothing");

        // Confirm refuses until the mesh arrives (and while it's missing).
        dispatch(&mut app, UiEvent::synthetic_click(EXPORT_CONFIRM_KEY));
        assert_eq!(app.take_file_action(), None);

        app.set_export_mesh(
            &exports[0],
            &[volumetric::Triangle::new([
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ])],
        );
        assert_eq!(app.export_dialog_wants_mesh(), None, "delivered once");

        // The inch preset rewrites the scale buffer; confirm carries it.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{EXPORT_UNIT_PREFIX}in")),
        );
        dispatch(&mut app, UiEvent::synthetic_click(EXPORT_CONFIRM_KEY));
        assert_eq!(
            app.take_file_action(),
            Some(FileAction::ExportMesh {
                id: exports[0].clone(),
                scale: 25.4,
            })
        );
        assert!(app.export_dialog.is_none(), "confirm closes the modal");

        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{EXPORT_WASM_PREFIX}{}", exports[1])),
        );
        assert_eq!(
            app.take_file_action(),
            Some(FileAction::ExportWasm(exports[1].clone()))
        );
    }

    #[test]
    fn export_modal_missing_mesh_and_bad_scale_refuse_confirm() {
        let (mut app, exports) = two_export_app();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{EXPORT_MESH_PREFIX}{}", exports[0])),
        );
        // Empty delivery: output has no cached mesh preview.
        app.set_export_mesh(&exports[0], &[]);
        assert_eq!(app.export_dialog_wants_mesh(), None, "asked only once");
        dispatch(&mut app, UiEvent::synthetic_click(EXPORT_CONFIRM_KEY));
        assert_eq!(app.take_file_action(), None, "nothing to export");
        assert!(app.export_dialog.is_some(), "modal stays open to explain");

        // A mesh arrives after a rerun-and-reopen, but the scale is garbage.
        app.set_export_mesh(
            &exports[0],
            &[volumetric::Triangle::new([
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ])],
        );
        app.export_dialog.as_mut().unwrap().scale_text = "-2".to_string();
        dispatch(&mut app, UiEvent::synthetic_click(EXPORT_CONFIRM_KEY));
        assert_eq!(app.take_file_action(), None, "bad scale refuses");

        // Dismiss closes without queueing anything.
        dispatch(&mut app, UiEvent::synthetic_click(EXPORT_DISMISS_KEY));
        assert!(app.export_dialog.is_none());
        assert_eq!(app.take_file_action(), None);
    }

    #[test]
    fn import_menu_clicks_queue_file_actions() {
        let mut app = VolumetricUiV2::default();
        dispatch(&mut app, UiEvent::synthetic_click(IMPORT_WASM_KEY));
        assert_eq!(app.take_file_action(), Some(FileAction::ImportWasm));
        dispatch(&mut app, UiEvent::synthetic_click(IMPORT_STL_KEY));
        assert_eq!(app.take_file_action(), Some(FileAction::ImportStl));
        dispatch(&mut app, UiEvent::synthetic_click(IMPORT_IMAGE_KEY));
        assert_eq!(app.take_file_action(), Some(FileAction::ImportImage));
    }

    #[test]
    fn import_model_wasm_adds_a_selected_import() {
        let mut app = VolumetricUiV2::empty();
        let bytes = volumetric_assets::models()[0].bytes.to_vec();
        app.import_model_wasm("custom_part", bytes);

        let summary = app.summary();
        assert_eq!(summary.imports, 1);
        assert_eq!(summary.exports, 1);
        assert!(summary.last_run_stale);
        assert_eq!(
            summary.selected_project_item,
            Some(ProjectSelection::Import(0))
        );
        assert_eq!(app.project().exports()[0], "custom_part");
    }

    #[test]
    fn import_blob_asset_wires_bytes_into_the_blob_slot() {
        let mut app = VolumetricUiV2::empty();
        let stl = vec![0xAB; 96];
        app.import_blob_asset("stl_import_operator", "stl_import", stl.clone());

        // An STL import stages exactly one step (STL -> TriMesh); users add
        // mesh_to_model themselves when they want a sampleable solid.
        let summary = app.summary();
        assert_eq!(summary.timeline_steps, 1);
        assert_eq!(summary.exports, 1);
        assert_eq!(
            summary.selected_project_item,
            Some(ProjectSelection::Step(0))
        );

        // The operator's Blob slot holds the file bytes; the config slot got
        // its schema defaults.
        let asset = volumetric_assets::get_operator("stl_import_operator").unwrap();
        let metadata = volumetric::operator_metadata_from_wasm_bytes(asset.bytes).unwrap();
        let step = &app.project().timeline()[0];
        let blob_slot = metadata
            .inputs
            .iter()
            .position(|input| matches!(input, OperatorMetadataInput::Blob))
            .expect("stl operator has a blob input");
        assert!(matches!(
            &step.inputs[blob_slot],
            ExecutionInput::Inline(bytes) if *bytes == stl
        ));
    }

    #[test]
    fn fea_mesh_outputs_stay_out_of_model_pickers() {
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, "fea_grid_mesh_operator");

        let step = &app.project().timeline()[0];
        let mesh_output = step.outputs[0].clone();

        // The static inspection alone would call the output a model...
        assert!(
            app.project()
                .declared_assets()
                .iter()
                .any(|(id, hint)| *id == mesh_output && *hint == Some(AssetTypeHint::Model))
        );
        // ...but the metadata-refined view types it correctly, and the model
        // pickers exclude it.
        assert!(
            app.declared_assets_typed()
                .iter()
                .any(|(id, hint)| *id == mesh_output && *hint == Some(AssetTypeHint::FeaMesh))
        );
        assert!(!editable_model_asset_ids(&app).contains(&mesh_output));
    }

    #[test]
    fn solve_operator_wires_mesh_model_and_config_slots() {
        let mut app = VolumetricUiV2::default();
        let model_id = app.project().imports()[0].id.clone();
        add_operator_click(&mut app, "fea_grid_mesh_operator");
        let mesh_id = app.project().timeline()[0].outputs[0].clone();

        // The mesher's output is now selected; adding the solver must wire
        // its FeaMesh slot to that output and fall back to the project's
        // model for the rigid-body slot.
        add_operator_click(&mut app, "fea_solve_operator");
        let step = &app.project().timeline()[1];
        assert!(
            matches!(&step.inputs[0], ExecutionInput::AssetRef(id) if *id == mesh_id),
            "mesh slot: {:?}",
            step.inputs[0]
        );
        assert!(
            matches!(&step.inputs[1], ExecutionInput::AssetRef(id) if *id == model_id),
            "rigid slot: {:?}",
            step.inputs[1]
        );

        // The seeded config decodes with the schema defaults (in particular
        // the fixed_boundary enum default must arrive unquoted).
        let ExecutionInput::Inline(config) = &step.inputs[2] else {
            panic!("config slot should be inline CBOR");
        };
        let decoded = operator_config::decode(config);
        assert_eq!(
            decoded.get("fixed_boundary"),
            Some(&operator_config::ConfigValue::Text("zmin".to_string()))
        );
        assert_eq!(
            decoded.get("youngs_modulus"),
            Some(&operator_config::ConfigValue::Float(1.0))
        );

        // And the step editor's FeaMesh picker offers the mesh outputs (the
        // mesher's, and the solver's own downstream one) but no models.
        let fea_ids = editable_fea_asset_ids(&app);
        assert!(fea_ids.contains(&mesh_id), "picker: {fea_ids:?}");
        assert!(!fea_ids.contains(&model_id));
        assert!(!editable_model_asset_ids(&app).contains(&mesh_id));
    }

    fn collect_keys(el: &El, keys: &mut Vec<String>) {
        if let Some(key) = &el.key {
            keys.push(key.clone());
        }
        for child in &el.children {
            collect_keys(child, keys);
        }
    }

    /// Built-copy state machine: the menu action queues SaveBuiltCopy,
    /// opening a baked file consumes the bake and flags the mode, and save
    /// outcomes pin or clear it (a built copy re-saves built; a lean save
    /// reverts the file to lean).
    #[test]
    fn built_copy_flags_follow_open_and_save() {
        let mut app = VolumetricUiV2::default();
        dispatch(&mut app, UiEvent::synthetic_click(SAVE_BUILT_COPY_KEY));
        assert_eq!(app.take_file_action(), Some(FileAction::SaveBuiltCopy));

        // Opening a project that carries a bake: the bake moves into the
        // process cache (never kept resident twice) and the mode is set.
        let mut baked_project = Project::new();
        baked_project.baked = Some(volumetric::BakedResults::default());
        let path = std::path::PathBuf::from("bracket.vproj");
        app.apply_opened_project(path.clone(), Ok(baked_project));
        assert!(app.baked_on_disk());
        assert!(app.project().baked.is_none(), "seeding consumes the bake");
        assert!(app.status.contains("built copy"), "{}", app.status);

        // Ordinary saves keep the mode via gather (tested by flag), and the
        // outcome pins what actually got written.
        let full = volumetric::BakeCoverage {
            baked_steps: 2,
            total_steps: 2,
        };
        app.apply_saved_project(path.clone(), Ok(()), Some(full));
        assert!(app.baked_on_disk());
        assert!(app.status.contains("built copy"), "{}", app.status);

        let partial = volumetric::BakeCoverage {
            baked_steps: 1,
            total_steps: 2,
        };
        app.apply_saved_project(path.clone(), Ok(()), Some(partial));
        assert!(app.baked_on_disk());
        assert!(app.status.contains("1/2"), "{}", app.status);

        app.apply_saved_project(path, Ok(()), None);
        assert!(!app.baked_on_disk(), "a lean save reverts the mode");

        // The built-copy project snapshot never mutates the live project.
        let (copy, coverage) = app.project_for_save(true);
        assert!(copy.baked.is_none(), "nothing cached for an empty project");
        assert_eq!(coverage.map(|c| c.total_steps), Some(0));
        assert!(app.project().baked.is_none());
    }

    /// Import-shaped operators (bare Blob inputs) route to their file
    /// dialogs from the Add modal, never to a bare operator insert.
    #[test]
    fn import_operators_route_to_file_dialogs() {
        let mut app = VolumetricUiV2::default();
        app.add_modal = Some(AddModalState::default());
        let mut keys = Vec::new();
        collect_keys(&add_layer(&app).expect("modal open"), &mut keys);
        assert!(
            !keys.contains(&format!("{ADD_OPERATOR_PREFIX}stl_import_operator")),
            "import operators only reachable via their file dialogs"
        );
        assert!(keys.contains(&IMPORT_WASM_KEY.to_string()));

        // Searching finds them under the same file-dialog routes.
        app.add_modal = Some(AddModalState {
            query: "stl".to_string(),
        });
        let mut keys = Vec::new();
        collect_keys(&add_layer(&app).expect("modal open"), &mut keys);
        assert!(
            !keys.contains(&format!("{ADD_OPERATOR_PREFIX}stl_import_operator")),
            "{keys:?}"
        );
    }

    /// The rail opens the modal, search ranking prefers display-name
    /// prefixes, and a row click adds and closes the modal.
    #[test]
    fn add_modal_searches_and_adds() {
        let mut app = VolumetricUiV2::empty();
        dispatch(&mut app, UiEvent::synthetic_click(ADD_OPEN_KEY));
        assert!(app.add_modal.is_some());

        // Warm two entries so search sees display metadata.
        for name in ["simple_sphere_model", "simple_torus_model"] {
            let asset = volumetric_assets::get_asset(name).expect("bundled");
            let result = volumetric::operator_metadata_from_wasm_bytes(asset.bytes)
                .map_err(|e| e.to_string());
            app.on_module_metadata(name, result);
        }
        app.add_modal = Some(AddModalState {
            query: "torus".to_string(),
        });
        let mut keys = Vec::new();
        collect_keys(&add_layer(&app).expect("modal open"), &mut keys);
        let torus_key = format!("{ADD_MODEL_PREFIX}simple_torus_model");
        assert!(keys.contains(&torus_key), "{keys:?}");
        assert!(!keys.contains(&format!("{ADD_MODEL_PREFIX}simple_sphere_model")));

        // Clicking the row adds the model, closes the modal, records recents.
        dispatch(&mut app, UiEvent::synthetic_click(torus_key));
        assert_eq!(app.summary().imports, 1);
        assert!(app.add_modal.is_none());
        assert_eq!(
            app.recent_adds.first().map(String::as_str),
            Some("simple_torus_model")
        );
    }

    /// Every category glyph parses. The statics are lazy, so without this
    /// a malformed SVG would panic at first render after a scan lands.
    #[test]
    fn category_icons_parse_for_every_known_category() {
        for category in CATEGORY_ORDER {
            let entry = catalog::CatalogEntry {
                name: "probe".to_string(),
                kind: volumetric_assets::AssetCategory::Operator,
                hash: String::new(),
                metadata: catalog::CatalogMetadata::Ready(OperatorMetadata {
                    name: "probe".to_string(),
                    version: "0.0.0".to_string(),
                    display_name: String::new(),
                    description: String::new(),
                    category: category.to_string(),
                    icon_svg: String::new(),
                    inputs: vec![],
                    input_names: vec![],
                    outputs: vec![],
                }),
                icon: None,
            };
            // Forces the LazyLock (and with it the parse) for each glyph.
            let _ = catalog_icon_source(&entry);
        }
    }

    /// Rail tiles re-add by catalog kind: a recent model tile imports it
    /// again without the modal.
    #[test]
    fn rail_tile_adds_by_kind() {
        let mut app = VolumetricUiV2::empty();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{RAIL_ADD_PREFIX}simple_sphere_model")),
        );
        assert_eq!(app.summary().imports, 1, "{}", app.status);

        // The default rail renders the curated seed as tiles.
        let mut keys = Vec::new();
        collect_keys(&add_rail(&app), &mut keys);
        assert!(keys.contains(&ADD_OPEN_KEY.to_string()));
        assert!(
            keys.contains(&format!("{RAIL_ADD_PREFIX}boolean_operator")),
            "{keys:?}"
        );
    }

    #[test]
    fn save_remembers_the_path_for_one_click_resave() {
        let path = std::env::temp_dir().join(format!(
            "volumetric_ui_v2_resave_{}.vproj",
            std::process::id()
        ));
        let mut app = VolumetricUiV2::default();
        app.save_project_file(&path);

        // Save re-saves in place: the known path is queued for the host's
        // file worker, no dialog variant.
        dispatch(&mut app, UiEvent::synthetic_click(SAVE_PROJECT_KEY));
        assert_eq!(
            app.take_file_action(),
            Some(FileAction::SaveProjectTo(path.clone()))
        );

        // Save As always asks.
        dispatch(&mut app, UiEvent::synthetic_click(SAVE_PROJECT_AS_KEY));
        assert_eq!(app.take_file_action(), Some(FileAction::SaveProject));

        // New Project forgets the path.
        dispatch(&mut app, UiEvent::synthetic_click(NEW_PROJECT_KEY));
        dispatch(&mut app, UiEvent::synthetic_click(SAVE_PROJECT_KEY));
        assert_eq!(app.take_file_action(), Some(FileAction::SaveProject));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn project_save_open_roundtrip() {
        let path = std::env::temp_dir().join(format!(
            "volumetric_ui_v2_roundtrip_{}.vproj",
            std::process::id()
        ));
        let mut source = VolumetricUiV2::default();
        add_operator_click(&mut source, first_operator_name());
        source.save_project_file(&path);
        assert!(source.status.starts_with("saved"), "{}", source.status);

        let mut opened = VolumetricUiV2::empty();
        opened.open_project_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(opened.status.starts_with("opened"), "{}", opened.status);
        let summary = opened.summary();
        let expected = source.summary();
        // Imports include the operator wasm alongside the model.
        assert_eq!(summary.imports, expected.imports);
        assert_eq!(summary.timeline_steps, expected.timeline_steps);
        assert_eq!(summary.exports, expected.exports);
        // Opening must not commit the user to a run (a project's first
        // build can take many minutes) — it lands stale for a manual Run.
        assert!(!opened.take_pending_run(), "open must not queue a run");
        assert!(opened.summary().last_run_stale);
    }

    #[test]
    fn opening_a_project_queues_a_run_only_under_auto_rebuild() {
        let path = std::env::temp_dir().join(format!(
            "volumetric_ui_v2_open_autorun_{}.vproj",
            std::process::id()
        ));
        let mut source = VolumetricUiV2::default();
        add_operator_click(&mut source, first_operator_name());
        source.save_project_file(&path);

        let mut opened = VolumetricUiV2::empty();
        dispatch(
            &mut opened,
            UiEvent::synthetic_click(TOGGLE_AUTO_REBUILD_KEY),
        );
        assert!(opened.auto_rebuild());
        opened.open_project_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(opened.take_pending_run(), "auto-rebuild opens should run");
    }

    #[test]
    fn panel_divider_is_in_the_shell() {
        let bundle = shell_bundle(Rect::new(0.0, 0.0, 1280.0, 800.0));
        assert!(bundle.tree_dump.contains(PANEL_RESIZE_KEY));
    }

    #[test]
    fn new_project_clears_runtime_and_pending_run() {
        let mut app = VolumetricUiV2::default();
        app.run_project();
        assert_eq!(app.summary().runtime_assets.len(), 1);

        dispatch(&mut app, UiEvent::synthetic_click(NEW_PROJECT_KEY));

        let summary = app.summary();
        assert!(summary.runtime_assets.is_empty());
        assert!(!summary.last_run_stale);
        assert!(!app.take_pending_run());
        assert!(app.preview_requests().is_empty());
    }

    /// Adds each bundled operator and, for the first one that declares a config
    /// schema, edits its first field and asserts the change lands in the step's
    /// CBOR blob.
    #[test]
    fn editing_config_updates_the_step_cbor() {
        let mut exercised = false;
        for op in volumetric_assets::operators() {
            let mut app = VolumetricUiV2::default();
            add_operator_click(&mut app, op.name);
            app.before_build();

            let Some(edit) = app.step_edit.as_ref() else {
                continue;
            };
            let Some(config) = edit.config.as_ref() else {
                continue;
            };
            let step_idx = edit.step_idx;
            let config_input_idx = config.input_idx;
            let Some(field) = config.fields.first().cloned() else {
                continue;
            };
            let field_name = field.name.clone();

            let expected = match &field.ty {
                ConfigFieldType::Bool => {
                    let before = config.buffers[&field_name] == "true";
                    app.toggle_config_bool(&field_name);
                    ConfigValue::Bool(!before)
                }
                ConfigFieldType::Enum(options) => {
                    let target = options.last().cloned().unwrap();
                    app.set_config_buffer(&field_name, target.clone());
                    ConfigValue::Text(target)
                }
                ConfigFieldType::Int => {
                    app.set_config_buffer(&field_name, "5".to_string());
                    ConfigValue::Int(5)
                }
                ConfigFieldType::Float => {
                    app.set_config_buffer(&field_name, "2.5".to_string());
                    ConfigValue::Float(2.5)
                }
                ConfigFieldType::Text => {
                    app.set_config_buffer(&field_name, "hello".to_string());
                    ConfigValue::Text("hello".to_string())
                }
            };

            let step = &app.project().timeline()[step_idx];
            let ExecutionInput::Inline(bytes) = &step.inputs[config_input_idx] else {
                panic!("config input should be inline CBOR");
            };
            let decoded = operator_config::decode(bytes);
            assert_eq!(
                decoded.get(&field_name),
                Some(&expected),
                "operator {}",
                op.name
            );
            exercised = true;
            break;
        }
        assert!(
            exercised,
            "expected at least one bundled operator with a config schema"
        );
    }

    /// Optional (CDDL `?`) config fields start unset, appear in the CBOR only
    /// while a value is committed, and clearing the buffer unsets them again
    /// (regression: brim's `? bed_position` was once always sent — and under
    /// a mis-parsed `? `-prefixed key at that).
    #[test]
    fn optional_config_fields_set_and_clear() {
        let mut exercised = false;
        for op in volumetric_assets::operators() {
            let mut app = VolumetricUiV2::default();
            add_operator_click(&mut app, op.name);
            app.before_build();

            let Some(config) = app.step_edit.as_ref().and_then(|edit| edit.config.as_ref()) else {
                continue;
            };
            let Some(field) = config
                .fields
                .iter()
                .find(|f| f.optional && f.ty == ConfigFieldType::Float)
                .cloned()
            else {
                continue;
            };
            let step_idx = app.step_edit.as_ref().unwrap().step_idx;
            let config_input_idx = config.input_idx;
            let decoded_config = |app: &VolumetricUiV2| {
                let step = &app.project().timeline()[step_idx];
                let ExecutionInput::Inline(bytes) = &step.inputs[config_input_idx] else {
                    panic!("config input should be inline CBOR");
                };
                operator_config::decode(bytes)
            };

            assert!(
                !decoded_config(&app).contains_key(&field.name),
                "{}: optional {} must start unset",
                op.name,
                field.name
            );
            assert_eq!(
                app.step_edit
                    .as_ref()
                    .unwrap()
                    .config
                    .as_ref()
                    .unwrap()
                    .buffers[&field.name],
                "",
                "unset optional field shows an empty buffer"
            );

            app.set_config_buffer(&field.name, "1.5".to_string());
            assert_eq!(
                decoded_config(&app).get(&field.name),
                Some(&ConfigValue::Float(1.5))
            );

            app.set_config_buffer(&field.name, String::new());
            assert!(
                !decoded_config(&app).contains_key(&field.name),
                "clearing the buffer unsets the field"
            );
            exercised = true;
            break;
        }
        assert!(
            exercised,
            "expected a bundled operator with an optional float config field \
             (brim's bed_position)"
        );
    }

    #[test]
    fn renaming_an_output_rewrites_every_reference() {
        // sphere -> op A (output O1) -> op B (input O1, output O2)
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, first_operator_name());
        let o1 = app.project().timeline()[0].outputs[0].clone();
        add_operator_click(&mut app, first_operator_name());
        assert!(matches!(
            &app.project().timeline()[1].inputs[0],
            ExecutionInput::AssetRef(id) if *id == o1
        ));

        // Pin + override the old name so the rekeying paths are exercised.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{TOGGLE_PIN_PREFIX}{o1}")),
        );
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{OUTPUT_MODE_PREFIX}{o1}:points")),
        );

        // Edit step 0 and rename its output.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SELECT_STEP_PREFIX}0")),
        );
        app.before_build();
        app.step_edit.as_mut().unwrap().output_name = "renamed_part".to_string();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{RENAME_OUTPUT_PREFIX}0")),
        );

        assert_eq!(app.project().timeline()[0].outputs[0], "renamed_part");
        assert!(matches!(
            &app.project().timeline()[1].inputs[0],
            ExecutionInput::AssetRef(id) if id == "renamed_part"
        ));
        assert!(app.project().exports().iter().any(|e| e == "renamed_part"));
        assert!(!app.project().exports().contains(&o1));
        assert!(app.pinned_outputs.contains("renamed_part"));
        assert!(app.output_overrides.contains_key("renamed_part"));
        assert!(app.summary().last_run_stale);
    }

    #[test]
    fn rename_rejects_names_already_in_use() {
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, first_operator_name());
        let original = app.project().timeline()[0].outputs[0].clone();

        app.before_build();
        app.step_edit.as_mut().unwrap().output_name = "simple_sphere_model".to_string();
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{RENAME_OUTPUT_PREFIX}0")),
        );

        assert_eq!(app.project().timeline()[0].outputs[0], original);
        assert!(app.status.contains("already in use"), "{}", app.status);
    }

    #[test]
    fn selecting_a_non_step_clears_the_step_editor() {
        let mut app = VolumetricUiV2::default();
        add_operator_click(&mut app, first_operator_name());
        app.before_build();

        // Select an import (not a step); the step editor should clear.
        dispatch(
            &mut app,
            UiEvent::synthetic_click(format!("{SELECT_IMPORT_PREFIX}0")),
        );
        app.before_build();
        assert!(app.step_edit.is_none());
    }

    #[test]
    fn model_slot_can_be_retargeted_per_slot() {
        // Add a second model, then an operator, then retarget the operator's
        // first model slot (input 0) to the sphere via the per-slot route.
        let mut app = VolumetricUiV2::default();
        add_model_click(&mut app, "simple_torus_model");
        add_operator_click(&mut app, first_operator_name());
        app.before_build();
        dispatch(
            &mut app,
            UiEvent::synthetic_click("project:set-step-model:0:0:simple_sphere_model"),
        );

        let step = &app.project().timeline()[0];
        assert!(matches!(
            &step.inputs[0],
            ExecutionInput::AssetRef(id) if id == "simple_sphere_model"
        ));
        // Retargeting the primary slot renames the first output after the input.
        assert!(
            step.outputs[0].starts_with("simple_sphere_model_"),
            "{:?}",
            step.outputs
        );
        assert!(app.project().exports().contains(&step.outputs[0]));
    }

    /// Adds each bundled operator and, for the first one with a Lua input, edits
    /// the source and asserts it lands in the step's input bytes.
    #[test]
    fn editing_lua_source_updates_step_bytes() {
        let mut exercised = false;
        for op in volumetric_assets::operators() {
            let mut app = VolumetricUiV2::default();
            add_operator_click(&mut app, op.name);
            app.before_build();

            let Some(edit) = app.step_edit.as_ref() else {
                continue;
            };
            if edit.lua.is_none() {
                continue;
            }

            // Simulate a text-area edit: set the buffer and commit it.
            let new_source = "-- edited\nreturn 1\n".to_string();
            let mut edit = app.step_edit.take().unwrap();
            let input_idx = {
                let lua = edit.lua.as_mut().unwrap();
                lua.source = new_source.clone();
                lua.input_idx
            };
            let lua_source = edit.lua.as_ref().unwrap();
            app.write_lua_source(edit.step_idx, lua_source);
            let step_idx = edit.step_idx;
            app.step_edit = Some(edit);

            let step = &app.project().timeline()[step_idx];
            let ExecutionInput::Inline(bytes) = &step.inputs[input_idx] else {
                panic!("lua input should be inline bytes");
            };
            assert_eq!(
                bytes.as_slice(),
                new_source.as_bytes(),
                "operator {}",
                op.name
            );
            exercised = true;
            break;
        }
        assert!(
            exercised,
            "expected at least one bundled operator with a Lua input"
        );
    }
}
