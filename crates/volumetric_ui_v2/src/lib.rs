//! Damascene-based v2 UI shell for Volumetric.
//!
//! This crate intentionally starts as a separate app path so the current egui UI
//! remains the usable baseline while the Damascene port grows toward parity.

use std::sync::Arc;

use volumetric::operator_config::{ConfigField, ConfigFieldType, ConfigValue};
use volumetric::{
    AssetTypeHint, ExecutionInput, LoadedAsset, OperatorMetadata, OperatorMetadataInput, Project,
    adaptive_surface_nets_2, operator_config,
};
use volumetric_renderer::CameraControlScheme;

use damascene_core::prelude::*;

pub mod host;

pub const VIEWPORT_KEY: &str = "viewport";
pub const NEW_PROJECT_KEY: &str = "action:new-project";
pub const RUN_PROJECT_KEY: &str = "action:run-project";
pub const CANCEL_RUN_KEY: &str = "action:cancel-run";
pub const TOGGLE_AUTO_REBUILD_KEY: &str = "action:toggle-auto-rebuild";
pub const TOGGLE_GRID_KEY: &str = "viewport:toggle-grid";
pub const TOGGLE_SSAO_KEY: &str = "viewport:toggle-ssao";
pub const FRAME_PREVIEW_KEY: &str = "viewport:frame-preview";

/// Top application menubar; menu values are `file` and `add`.
const MENUBAR_KEY: &str = "main-menu";
/// One-click add of a bundled asset from the Add menu.
const ADD_MODEL_PREFIX: &str = "add:model:";
const ADD_OPERATOR_PREFIX: &str = "add:operator:";
/// Pipeline accordion in the project panel; values `imports|steps|exports`.
const PIPELINE_KEY: &str = "pipeline";
/// Viewport overlay value pickers (controlled select widgets).
const MODE_SELECT_KEY: &str = "view:mode";
const RESOLUTION_SELECT_KEY: &str = "view:res";
const CAMERA_SELECT_KEY: &str = "view:camera";
/// Per-output render settings: `output:settings:{id}` opens the popover
/// (plus `:dismiss`); mode/res routes end `:{value}` after the asset id.
const OUTPUT_SETTINGS_PREFIX: &str = "output:settings:";
const OUTPUT_MODE_PREFIX: &str = "output:mode:";
const OUTPUT_RESOLUTION_PREFIX: &str = "output:res:";
const OUTPUT_DEFAULTS_PREFIX: &str = "output:defaults:";
const SELECT_IMPORT_PREFIX: &str = "project:select-import:";
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
const CONFIG_FIELD_PREFIX: &str = "cfg:";
const CONFIG_BOOL_PREFIX: &str = "cfg-bool:";
const CONFIG_ENUM_PREFIX: &str = "cfg-enum:";
const LUA_SOURCE_KEY: &str = "lua-source";
const PREVIEW_RESOLUTIONS: [usize; 4] = [24, 48, 64, 96];

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
        vertex_refinement_iterations: usize,
        normal_sample_iterations: usize,
    },
}

impl PreviewMeshPlan {
    fn for_mode(mode: PreviewRenderMode, resolution: usize) -> Self {
        match mode {
            PreviewRenderMode::Points => Self::PointCloud { resolution },
            PreviewRenderMode::MarchingCubes => Self::MarchingCubes { resolution },
            PreviewRenderMode::AdaptiveSurfaceNets2 => {
                let (base_resolution, max_depth) = asn2_resolution_split(resolution);
                Self::AdaptiveSurfaceNets2 {
                    target_resolution: resolution,
                    base_resolution,
                    max_depth,
                    vertex_refinement_iterations: 8,
                    normal_sample_iterations: 0,
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
            vertex_refinement_iterations,
            normal_sample_iterations,
            ..
        } = self
        else {
            return None;
        };

        Some(adaptive_surface_nets_2::AdaptiveMeshConfig2 {
            base_resolution: *base_resolution,
            max_depth: *max_depth,
            vertex_refinement_iterations: *vertex_refinement_iterations,
            normal_sample_iterations: *normal_sample_iterations,
            normal_epsilon_frac: 0.1,
            num_threads: 0,
            sharp_edge_config: None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PreviewRequest {
    pub asset_id: String,
    pub wasm_bytes: Arc<Vec<u8>>,
    pub type_hint: Option<AssetTypeHint>,
    pub precursor_ids: Vec<String>,
    pub render_mode: PreviewRenderMode,
    pub mesh_plan: PreviewMeshPlan,
    pub show_grid: bool,
    pub ssao: bool,
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
    Failed {
        label: String,
        error: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewportCameraCommand {
    FramePreview,
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
#[derive(Debug)]
struct StepEditState {
    step_idx: usize,
    /// Input indices that are `ModelWASM` slots, in declaration order.
    model_slots: Vec<usize>,
    /// The config form, present only when the operator declares a config input.
    config: Option<ConfigForm>,
    /// The Lua source editor, present only for a `LuaSource` input.
    lua: Option<LuaForm>,
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

/// Render settings for one output. Stored per asset id when the user
/// overrides the viewport defaults for that output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRender {
    pub mode: PreviewRenderMode,
    pub resolution: usize,
}

#[derive(Debug)]
pub struct VolumetricUiV2 {
    project: Project,
    selected_export: Option<String>,
    selected_project_item: Option<ProjectSelection>,
    /// Outputs pinned to stay in the viewport regardless of the current
    /// selection. The rendered scene is the selected node's output plus these.
    pinned_outputs: std::collections::BTreeSet<String>,
    /// Per-output render settings; outputs without an entry follow the global
    /// `render_mode`/`preview_resolution` defaults.
    output_overrides: std::collections::BTreeMap<String, OutputRender>,
    render_mode: PreviewRenderMode,
    preview_resolution: usize,
    camera_control_scheme: CameraControlScheme,
    show_grid: bool,
    ssao: bool,
    runtime_assets: Vec<LoadedAsset>,
    last_run_elapsed_ms: Option<u128>,
    last_run_error: Option<String>,
    last_run_stale: bool,
    run_state: RunState,
    auto_rebuild: bool,
    pending_run: bool,
    cancel_requested: bool,
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
            render_mode: PreviewRenderMode::AdaptiveSurfaceNets2,
            preview_resolution: 64,
            camera_control_scheme: CameraControlScheme::default(),
            show_grid: true,
            ssao: true,
            runtime_assets: Vec::new(),
            last_run_elapsed_ms: None,
            last_run_error: None,
            last_run_stale: false,
            run_state: RunState::Idle,
            auto_rebuild: false,
            pending_run: false,
            cancel_requested: false,
            preview_build_status: PreviewBuildStatus::Idle,
            pending_camera_command: None,
            viewport_texture: None,
            selection: Selection::default(),
            step_edit: None,
            status: "idle".to_string(),
            open_menu: None,
            open_select: None,
            pipeline_open: ["imports", "steps", "exports"]
                .into_iter()
                .map(str::to_string)
                .collect(),
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

    /// Effective render settings for an output: its override if present,
    /// otherwise the viewport defaults.
    fn output_render(&self, id: &str) -> OutputRender {
        self.output_overrides
            .get(id)
            .copied()
            .unwrap_or(OutputRender {
                mode: self.render_mode,
                resolution: self.preview_resolution,
            })
    }

    fn set_output_mode(&mut self, id: &str, mode: PreviewRenderMode) {
        let mut render = self.output_render(id);
        render.mode = mode;
        self.output_overrides.insert(id.to_string(), render);
        self.status = format!("{id}: {}", mode.full_label());
    }

    fn set_output_resolution(&mut self, id: &str, resolution: usize) {
        let mut render = self.output_render(id);
        render.resolution = resolution;
        self.output_overrides.insert(id.to_string(), render);
        self.status = format!("{id}: {resolution}^3");
    }

    fn clear_output_override(&mut self, id: &str) {
        self.output_overrides.remove(id);
        self.status = format!("{id}: viewport defaults");
    }

    /// Builds a render request for a single runtime asset, or `None` when it is
    /// not a renderable model.
    fn render_request_for_asset(&self, asset: &LoadedAsset) -> Option<PreviewRequest> {
        if !matches!(asset.type_hint(), Some(AssetTypeHint::Model) | None) {
            return None;
        }

        let render = self.output_render(asset.id());
        Some(PreviewRequest {
            asset_id: asset.id().to_string(),
            wasm_bytes: asset.data_arc(),
            type_hint: asset.type_hint(),
            precursor_ids: asset.precursor_ids().to_vec(),
            render_mode: render.mode,
            mesh_plan: PreviewMeshPlan::for_mode(render.mode, render.resolution),
            show_grid: self.show_grid,
            ssao: self.ssao,
            stale: self.last_run_stale,
        })
    }

    pub fn camera_control_scheme(&self) -> CameraControlScheme {
        self.camera_control_scheme
    }

    pub(crate) fn set_preview_build_status(&mut self, status: PreviewBuildStatus) {
        self.preview_build_status = status;
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
    pub(crate) fn take_pending_run(&mut self) -> bool {
        std::mem::take(&mut self.pending_run)
    }

    /// Host hook: consumes a pending cancel request for the in-flight run.
    pub(crate) fn take_cancel_request(&mut self) -> bool {
        std::mem::take(&mut self.cancel_requested)
    }

    pub(crate) fn set_run_state(&mut self, state: RunState) {
        self.run_state = state;
        if state == RunState::Running {
            self.status = "running project".to_string();
        }
    }

    /// Host hook: the in-flight run was cancelled. Its (abandoned) result, if it
    /// still arrives, is discarded by generation on the host side.
    pub(crate) fn on_run_cancelled(&mut self) {
        self.run_state = RunState::Idle;
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

        let model_slots: Vec<usize> = metadata
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(idx, input)| {
                matches!(input, OperatorMetadataInput::ModelWASM).then_some(idx)
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
        if model_slots.is_empty() && config.is_none() && lua.is_none() {
            return None;
        }

        Some(StepEditState {
            step_idx,
            model_slots,
            config,
            lua,
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
                let value = current
                    .get(&field.name)
                    .cloned()
                    .unwrap_or_else(|| field.seed_value());
                (field.name.clone(), value.to_display_string())
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

    /// Retargets one model input slot of a step. Retargeting the primary (first)
    /// model slot also renames the step's first output after the new input and
    /// rewires exports, matching how a freshly added operator is named.
    fn set_step_model_input(&mut self, step_idx: usize, input_idx: usize, asset_id: &str) {
        let primary_slot = self
            .step_edit
            .as_ref()
            .filter(|edit| edit.step_idx == step_idx)
            .and_then(|edit| edit.model_slots.first().copied())
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

    /// Commits the field's current buffer into the step's CBOR blob, but only if
    /// it parses; an invalid intermediate (e.g. an empty number field) is left
    /// in the buffer untouched.
    fn commit_config_buffer(&mut self, step_idx: usize, config: &ConfigForm, field_name: &str) {
        let Some(field) = config.fields.iter().find(|f| f.name == field_name) else {
            return;
        };
        let Some(buffer) = config.buffers.get(field_name) else {
            return;
        };
        let Some(value) = ConfigValue::parse(&field.ty, buffer) else {
            return;
        };
        self.write_config_value(step_idx, config, field_name, value);
        self.mark_project_dirty();
    }

    fn write_config_value(
        &mut self,
        step_idx: usize,
        config: &ConfigForm,
        field_name: &str,
        value: ConfigValue,
    ) {
        let Some(step) = self.project.timeline_mut().get_mut(step_idx) else {
            return;
        };
        let mut values = match step.inputs.get(config.input_idx) {
            Some(ExecutionInput::Inline(bytes)) => operator_config::decode(bytes),
            _ => return,
        };
        values.insert(field_name.to_string(), value);
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

    fn add_model(&mut self, name: &str) {
        let Some(asset) = volumetric_assets::get_model(name) else {
            self.status = format!("missing bundled model {name}");
            return;
        };

        let id = self.project.insert_model(asset.name, asset.bytes.to_vec());
        self.mark_project_dirty();
        self.selected_export = Some(id.clone());
        self.selected_project_item = self
            .project
            .imports()
            .len()
            .checked_sub(1)
            .map(ProjectSelection::Import);
        self.status = format!("imported {} as {id}", asset.display_name);
    }

    /// Appends a timeline step for the named bundled operator, wired to the
    /// currently selected export as its primary model input.
    fn add_operator(&mut self, name: &str) {
        let Some(asset) = volumetric_assets::get_operator(name) else {
            self.status = format!("missing bundled operator {name}");
            return;
        };
        let Some(input_id) = self.selected_export.clone() else {
            self.status = "add or select a model before adding an operator".to_string();
            return;
        };

        let output_id = self
            .project
            .default_output_name(asset.name, Some(&input_id));

        // Build the step's inputs from the operator's declared metadata so
        // config blobs, Lua templates, and extra model slots are all wired up
        // (not just the first model input).
        let inputs = match volumetric::operator_metadata_from_wasm_bytes(asset.bytes) {
            Ok(metadata) => operator_step_inputs(&metadata, &input_id),
            Err(err) => {
                self.status = format!("couldn't read {} metadata: {err}", asset.display_name);
                vec![ExecutionInput::AssetRef(input_id.clone())]
            }
        };

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
        self.status = format!("added {} -> {output_id}", asset.display_name);
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
        for key in [MODE_SELECT_KEY, RESOLUTION_SELECT_KEY, CAMERA_SELECT_KEY] {
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
        // Escape closes any open menu/picker popover (the popover contract:
        // the scrim handles outside clicks, the app handles Escape).
        if matches!(event.kind, UiEventKind::Escape)
            && (self.open_menu.take().is_some() | self.open_select.take().is_some())
        {
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
            self.status = "new project".to_string();
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

        if event.is_click_or_activate(TOGGLE_AUTO_REBUILD_KEY) {
            self.auto_rebuild = !self.auto_rebuild;
            self.status = if self.auto_rebuild {
                "auto-rebuild on".to_string()
            } else {
                "auto-rebuild off".to_string()
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
            self.open_menu = None;
        } else if let Some(name) = route.strip_prefix(ADD_OPERATOR_PREFIX) {
            self.add_operator(name);
            self.open_menu = None;
        } else if let Some(idx) = parse_index_route(route, SELECT_IMPORT_PREFIX) {
            self.selected_project_item = Some(ProjectSelection::Import(idx));
            self.status = format!("selected import {}", idx + 1);
        } else if let Some(idx) = parse_index_route(route, DELETE_IMPORT_PREFIX) {
            self.delete_import(idx);
        } else if let Some(idx) = parse_index_route(route, SELECT_STEP_PREFIX) {
            self.selected_project_item = Some(ProjectSelection::Step(idx));
            self.status = format!("selected step {}", idx + 1);
        } else if let Some(idx) = parse_index_route(route, DELETE_STEP_PREFIX) {
            self.delete_step(idx);
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
        row([viewport_pane(app), project_panel(app)])
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0)),
    ])
    .fill_size();

    overlays(main, [menu_layer(app), select_layer(app)])
}

/// The single strip of application chrome above the viewport: menubar
/// (File / Add), then run controls and status on the right.
fn top_bar(app: &VolumetricUiV2) -> El {
    let open = app.open_menu.as_deref();
    toolbar([
        icon("layout-dashboard").icon_size(tokens::ICON_SM).muted(),
        text("Volumetric").label().semibold().key("brand-title"),
        menubar([
            menubar_trigger(MENUBAR_KEY, "file", "File", open == Some("file")),
            menubar_trigger(MENUBAR_KEY, "add", "Add", open == Some("add")),
        ]),
        spacer(),
        run_status_chip(app),
        preview_status_chip(app),
        toggle_chip("Auto", app.auto_rebuild, TOGGLE_AUTO_REBUILD_KEY),
        run_control(app),
    ])
    .gap(tokens::SPACE_2)
    .padding(Sides::xy(tokens::SPACE_3, tokens::SPACE_2))
}

/// The open menubar menu, rendered as a root overlay layer.
fn menu_layer(app: &VolumetricUiV2) -> Option<El> {
    match app.open_menu.as_deref()? {
        "file" => Some(menubar_menu(
            MENUBAR_KEY,
            "file",
            [menubar_item_with_icon("plus", "New Project").key(NEW_PROJECT_KEY)],
        )),
        "add" => Some(menubar_menu(MENUBAR_KEY, "add", add_menu_items())),
        _ => None,
    }
}

/// The Add menu body: every bundled model and operator, one click to add.
fn add_menu_items() -> Vec<El> {
    let mut items = vec![menubar_label("Models")];
    for asset in volumetric_assets::models() {
        items.push(
            menubar_item_with_icon("activity", asset.display_name)
                .key(format!("{ADD_MODEL_PREFIX}{}", asset.name)),
        );
    }
    items.push(menubar_separator());
    items.push(menubar_label("Operators"));
    for asset in volumetric_assets::operators() {
        items.push(
            menubar_item_with_icon("settings", asset.display_name)
                .key(format!("{ADD_OPERATOR_PREFIX}{}", asset.name)),
        );
    }
    items
}

/// The open viewport picker menu (or per-output settings popover), rendered
/// as a root overlay layer.
fn select_layer(app: &VolumetricUiV2) -> Option<El> {
    if let Some(id) = app.open_select.as_deref()?.strip_prefix(OUTPUT_SETTINGS_PREFIX) {
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
            CameraControlScheme::ALL
                .iter()
                .copied()
                .map(|scheme| (camera_scheme_route_name(scheme), camera_scheme_tooltip(scheme))),
        )),
        _ => None,
    }
}

/// Anchored settings popover for one output: render mode + resolution, and a
/// reset to the viewport defaults when overridden. Picks keep the popover
/// open (it's a settings panel, not a value picker); outside click dismisses.
fn output_settings_popover(app: &VolumetricUiV2, id: &str) -> El {
    let render = app.output_render(id);
    let trigger_key = format!("{OUTPUT_SETTINGS_PREFIX}{id}");

    let mode_buttons = row(PreviewRenderMode::ALL.into_iter().map(|mode| {
        let button = button(mode.label())
            .xsmall()
            .tooltip(mode.full_label())
            .key(format!("{OUTPUT_MODE_PREFIX}{id}:{}", mode.route_name()));
        if render.mode == mode {
            button.primary()
        } else {
            button.secondary()
        }
    }))
    .gap(tokens::SPACE_1);
    let resolution_buttons = row(PREVIEW_RESOLUTIONS.into_iter().map(|resolution| {
        let button = button(format!("{resolution}^3"))
            .xsmall()
            .key(format!("{OUTPUT_RESOLUTION_PREFIX}{id}:{resolution}"));
        if render.resolution == resolution {
            button.primary()
        } else {
            button.secondary()
        }
    }))
    .gap(tokens::SPACE_1);

    let mut body = vec![
        text(id).label().semibold().ellipsis(),
        text("Render Mode").caption().muted(),
        mode_buttons,
        text("Resolution").caption().muted(),
        resolution_buttons,
    ];
    if app.output_overrides.contains_key(id) {
        body.push(divider());
        body.push(
            button("Use viewport defaults")
                .xsmall()
                .secondary()
                .width(Size::Fill(1.0))
                .key(format!("{OUTPUT_DEFAULTS_PREFIX}{id}")),
        );
    }

    popover(
        trigger_key.clone(),
        Anchor::below_key(trigger_key),
        // popover_panel paints the floating card surface (fill + border +
        // shadow); a bare column would float transparently over the panel.
        popover_panel([column(body).gap(tokens::SPACE_2).padding(tokens::SPACE_2)]),
    )
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
        if on { button.primary() } else { button.secondary() }
    };
    card([row([
        toggle("Grid", app.show_grid, TOGGLE_GRID_KEY),
        toggle("SSAO", app.ssao, TOGGLE_SSAO_KEY),
        button("Frame")
            .xsmall()
            .secondary()
            .tooltip("Frame the preview in view")
            .key(FRAME_PREVIEW_KEY),
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
    let visible = app.preview_requests().len();
    row([
        badge(format!("{visible} in viewport")).muted().xsmall(),
        badge(format!("{} outputs", app.runtime_assets.len()))
            .muted()
            .xsmall(),
        badge(&app.status).secondary().xsmall(),
        spacer(),
    ])
    .gap(tokens::SPACE_1)
    .align(Align::Center)
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
        RunState::Running => row([
            spinner().width(Size::Fixed(14.0)).height(Size::Fixed(14.0)),
            badge("running").info().xsmall(),
        ])
        .gap(tokens::SPACE_1)
        .align(Align::Center),
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
    let label = app.preview_build_status.label();
    let badge = match &app.preview_build_status {
        PreviewBuildStatus::Idle => badge(label).muted().xsmall(),
        PreviewBuildStatus::Building { .. } => badge(label).info().xsmall(),
        PreviewBuildStatus::Ready { .. } => badge(label).success().xsmall(),
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
    column([
        scroll([
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
        .px(tokens::RING_WIDTH),
    ])
    .width(Size::Fixed(320.0))
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

    let pin = icon_button("pin")
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
                "{} · {} · {}^3{}",
                asset_type_label(asset.type_hint()),
                render.mode.label(),
                render.resolution,
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
        icon_button("eye")
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

    vec![
        detail_row("Kind", asset_type_label(import.type_hint)),
        detail_row("Asset", &import.id),
        detail_row("Bytes", &import.data.len().to_string()),
        button_with_icon("x", "Delete")
            .destructive()
            .xsmall()
            .width(Size::Fill(1.0))
            .key(format!("{DELETE_IMPORT_PREFIX}{idx}")),
    ]
}

fn step_detail_rows(app: &VolumetricUiV2, idx: usize) -> Vec<El> {
    let Some(step) = app.project.timeline().get(idx) else {
        return vec![text("Selected step no longer exists.").muted().small()];
    };

    let mut rows = vec![
        detail_row("Step", &(idx + 1).to_string()),
        detail_row("Operator", &step.operator_id),
        detail_row("Outputs", &step.outputs.len().to_string()),
    ];

    rows.extend(step_edit_rows(app, idx));

    rows.push(
        toolbar([
            button_with_icon("chevron-left", "Up")
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

    if !edit.model_slots.is_empty() {
        let models = editable_model_asset_ids(app);
        rows.push(text("Inputs").muted().caption().semibold());
        for (n, &slot) in edit.model_slots.iter().enumerate() {
            let current = match step.inputs.get(slot) {
                Some(ExecutionInput::AssetRef(id)) => id.as_str(),
                _ => "",
            };
            rows.push(model_slot_selector(step_idx, slot, n, current, &models));
        }
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
    }

    rows
}

/// A model input slot: a labelled column of full-width buttons, one per
/// available model, with the current target highlighted.
fn model_slot_selector(
    step_idx: usize,
    slot: usize,
    ordinal: usize,
    current: &str,
    models: &[String],
) -> El {
    let mut items = vec![
        text(format!("Input {}", ordinal + 1))
            .muted()
            .caption()
            .width(Size::Fill(1.0)),
    ];
    for id in models {
        let button = button_with_icon("git-branch", id.clone())
            .xsmall()
            .width(Size::Fill(1.0))
            .key(format!("{SET_STEP_MODEL_PREFIX}{step_idx}:{slot}:{id}"));
        items.push(if id == current {
            button.primary()
        } else {
            button.secondary()
        });
    }
    column(items).gap(tokens::SPACE_1).width(Size::Fill(1.0))
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
    field_row(&field.name, control).gap(tokens::SPACE_2)
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

    vec![
        detail_row("Export", export_id),
        button_with_icon("x", "Delete")
            .destructive()
            .xsmall()
            .width(Size::Fill(1.0))
            .key(format!("{DELETE_EXPORT_PREFIX}{idx}")),
    ]
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
            project_row(
                asset_type_label(import.type_hint),
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
            for input in &step.inputs {
                rows.push(project_note_row("input", &input.display()));
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
        text(kind)
            .caption()
            .muted()
            .ellipsis()
            .width(Size::Fixed(42.0)),
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

fn asset_type_label(type_hint: Option<AssetTypeHint>) -> &'static str {
    match type_hint {
        Some(AssetTypeHint::Model) => "Model",
        Some(AssetTypeHint::Operator) => "Operator",
        Some(AssetTypeHint::Config) => "Config",
        Some(AssetTypeHint::LuaSource) => "Lua",
        Some(AssetTypeHint::Binary) => "Binary",
        Some(AssetTypeHint::VecF64(_)) => "Vec",
        None => "Asset",
    }
}

/// Builds a fresh step's inputs, one per declared operator metadata input, in
/// order. Model slots point at `primary_model` (retarget later); config slots
/// get a default-seeded CBOR map; Lua slots get the template; other slots get
/// empty/zeroed placeholders.
fn operator_step_inputs(metadata: &OperatorMetadata, primary_model: &str) -> Vec<ExecutionInput> {
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

fn asn2_resolution_split(target_resolution: usize) -> (usize, usize) {
    let base_resolution =
        if !target_resolution.is_power_of_two() && target_resolution.is_multiple_of(6) {
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
    app.project
        .declared_assets()
        .into_iter()
        .filter_map(|(id, type_hint)| match type_hint {
            Some(AssetTypeHint::Model) | None => Some(id),
            _ => None,
        })
        .collect()
}

pub fn shell_bundle(viewport: Rect) -> damascene_core::bundle::artifact::Bundle {
    let app = VolumetricUiV2::default();
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
        dispatch(app, UiEvent::synthetic_click(format!("{ADD_MODEL_PREFIX}{name}")));
    }

    /// Click the Add-menu entry for the named bundled operator.
    fn add_operator_click(app: &mut VolumetricUiV2, name: &str) {
        dispatch(
            app,
            UiEvent::synthetic_click(format!("{ADD_OPERATOR_PREFIX}{name}")),
        );
    }

    fn first_operator_name() -> &'static str {
        volumetric_assets::operators()
            .first()
            .expect("bundled operators")
            .name
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
        dispatch(&mut app, UiEvent::synthetic_click("main-menu:menu:add"));
        assert_eq!(app.open_menu.as_deref(), Some("add"));

        add_model_click(&mut app, "simple_sphere_model");
        assert_eq!(app.open_menu, None);
        assert_eq!(app.summary().imports, 2);

        // Clicking the trigger again toggles closed too.
        dispatch(&mut app, UiEvent::synthetic_click("main-menu:menu:file"));
        dispatch(&mut app, UiEvent::synthetic_click("main-menu:menu:file"));
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
                    // The default config decodes and covers every schema field.
                    let fields = operator_config::parse_schema(cddl).unwrap_or_default();
                    let decoded = operator_config::decode(bytes);
                    for field in &fields {
                        assert!(decoded.contains_key(&field.name), "missing {}", field.name);
                    }
                }
                _ => assert!(matches!(input, ExecutionInput::Inline(_))),
            }
        }
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
        dispatch(&mut app, UiEvent::synthetic_click("view:mode:option:points"));
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

        let summary = app.summary();
        assert!(!summary.show_grid);
        assert!(!summary.ssao);
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
        assert_eq!(request.render_mode, PreviewRenderMode::AdaptiveSurfaceNets2);
        assert_eq!(
            request.mesh_plan,
            PreviewMeshPlan::AdaptiveSurfaceNets2 {
                target_resolution: 96,
                base_resolution: 6,
                max_depth: 4,
                vertex_refinement_iterations: 8,
                normal_sample_iterations: 0,
            }
        );
        let config = request
            .mesh_plan
            .adaptive_surface_nets_config()
            .expect("asn2 config");
        assert_eq!(config.base_resolution, 6);
        assert_eq!(config.max_depth, 4);
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
        assert_eq!(overridden.render_mode, PreviewRenderMode::Points);
        assert_eq!(
            overridden.mesh_plan,
            PreviewMeshPlan::for_mode(PreviewRenderMode::Points, 24)
        );
        assert_eq!(
            default.render_mode,
            PreviewRenderMode::AdaptiveSurfaceNets2
        );
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
        assert_eq!(
            request.render_mode,
            PreviewRenderMode::AdaptiveSurfaceNets2,
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
