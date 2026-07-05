//! Platform-neutral session core for the v2 host.
//!
//! Everything a host shell needs that is not tied to a window system lives
//! here: the viewport renderer and its per-output preview cache, project-run
//! bookkeeping (generations, cooperative cancel), camera input expressed over
//! Damascene's event vocabulary, and the background-job types plus their
//! coalescing queue and executor. The winit shell in `host.rs` is the only
//! native-specific layer; a future web shell reuses this module unchanged and
//! substitutes its own event source, dialogs, and job execution strategy.

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use damascene_core::prelude::*;
use glam::{Mat4, Vec2, Vec3};
use volumetric_renderer as renderer;

use crate::{
    OutputStats, PreviewBuildStatus, PreviewMeshPlan, PreviewRenderMode, PreviewRequest, RunState,
    VolumetricUiV2,
};

/// Per-frame and per-event driver for one running app instance.
///
/// The shell owns the window, the Damascene runner, and a job executor; the
/// session owns everything in between. The frame protocol is:
///
/// 1. [`Session::pre_frame`] with the executor's completed results (routes
///    them, honors a cancel request).
/// 2. The shell fulfills any queued [`crate::FileAction`] (dialogs are
///    platform code).
/// 3. [`Session::sync`] — dispatches a queued project run, reconciles preview
///    requests, applies camera commands, sizes the render target — returning
///    jobs for the shell to hand to its executor.
/// 4. The shell builds and prepares the Damascene tree, then calls
///    [`Session::render`] with the laid-out viewport rect.
///
/// Pointer/wheel input for the camera goes through the `pointer_*`/`wheel`
/// methods, typed entirely in Damascene's event vocabulary.
pub struct Session {
    viewport: ViewportRenderer,
    /// Monotonic id for the most recently dispatched project run. Results whose
    /// generation is older than the active run are discarded (superseded or
    /// cancelled).
    run_generation: u64,
    /// The in-flight run's generation and its cooperative cancel flag.
    active_run: Option<(u64, Arc<AtomicBool>)>,
    /// Pointer position of an in-progress camera drag (a press that landed in
    /// the viewport rect). `None` when the pointer isn't driving the camera.
    camera_pointer: Option<(f32, f32)>,
    camera_buttons: CameraButtons,
    /// The viewport widget's rect from the last completed layout, used to
    /// hit-test camera input and to size the render target between frames.
    viewport_rect: Option<Rect>,
}

impl Session {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self {
            viewport: ViewportRenderer::new(device, queue, format),
            run_generation: 0,
            active_run: None,
            camera_pointer: None,
            camera_buttons: CameraButtons::default(),
            viewport_rect: None,
        }
    }

    /// Routes completed background work into the app and viewport, then honors
    /// a cancel request for the in-flight run: signal the executor and bump the
    /// generation so the (abandoned) result is ignored on arrival.
    pub fn pre_frame(
        &mut self,
        app: &mut VolumetricUiV2,
        results: impl IntoIterator<Item = BackgroundResult>,
    ) {
        for result in results {
            match result {
                BackgroundResult::ProjectComplete {
                    generation,
                    result,
                    elapsed_ms,
                } => {
                    if self.active_run.as_ref().map(|(g, _)| *g) == Some(generation) {
                        self.active_run = None;
                        app.apply_run_result(result, elapsed_ms);
                    }
                    // Otherwise the run was superseded or cancelled; discard it.
                }
                BackgroundResult::PreviewComplete(preview) => {
                    self.viewport.accept_preview_result(preview);
                }
            }
        }

        if app.take_cancel_request() {
            if let Some((_, cancel)) = self.active_run.take() {
                cancel.store(true, Ordering::Relaxed);
            }
            self.run_generation += 1;
            app.on_run_cancelled();
        }
    }

    /// Frame-state sync between app and viewport. Dispatches a queued project
    /// run, reconciles the requested preview set against the mesh cache,
    /// applies a queued camera command, and (re)sizes the render target for
    /// the last known viewport rect. Returns the background jobs the shell
    /// should enqueue on its executor.
    pub fn sync(
        &mut self,
        app: &mut VolumetricUiV2,
        device: &wgpu::Device,
        requests: &[PreviewRequest],
        scale_factor: f32,
    ) -> Vec<BackgroundJob> {
        let mut jobs = Vec::new();

        if app.take_pending_run() {
            self.run_generation += 1;
            let cancel = Arc::new(AtomicBool::new(false));
            self.active_run = Some((self.run_generation, cancel.clone()));
            app.set_run_state(RunState::Running);
            jobs.push(BackgroundJob::RunProject {
                generation: self.run_generation,
                project: app.project().clone(),
                cancel,
            });
        }

        let (status, preview_jobs) = self.viewport.preview_cache.sync(requests);
        app.set_preview_build_status(status);
        app.set_output_stats(self.viewport.preview_cache.output_stats());
        jobs.extend(preview_jobs.into_iter().map(BackgroundJob::BuildPreview));

        if let Some(command) = app.take_camera_command() {
            self.viewport.apply_camera_command(command);
        }
        if let Some(rect) = self.viewport_rect {
            self.viewport
                .ensure_target_for_rect(device, rect, scale_factor);
        }
        app.set_viewport_texture(self.viewport.app_texture());
        jobs
    }

    /// Renders the composited preview scene into the viewport texture at the
    /// laid-out rect, remembering the rect for next frame's target sizing and
    /// camera hit tests. Returns whether the render target was reallocated
    /// (the shell should schedule another frame).
    pub fn render(&mut self, params: ViewportRenderParams<'_>) -> bool {
        self.viewport_rect = params.logical_rect;
        self.viewport.render(params)
    }

    pub fn has_pending_preview(&self) -> bool {
        self.viewport.preview_cache.has_pending()
    }

    pub fn run_in_flight(&self) -> bool {
        self.active_run.is_some()
    }

    /// World-space triangles of an output's cached preview mesh, for STL
    /// export. Empty when the output isn't cached or was meshed as points.
    pub fn preview_triangles(&self, id: &str) -> Vec<volumetric::Triangle> {
        let Some(scene) = self.viewport.preview_cache.entity_scene(id) else {
            return Vec::new();
        };
        let mut triangles = Vec::new();
        for (mesh, transform, _) in &scene.meshes {
            mesh_triangles(mesh, *transform, &mut triangles);
        }
        triangles
    }

    /// A pointer press: starts a camera drag when it lands in the viewport.
    pub fn pointer_down(&mut self, pos: (f32, f32), button: PointerButton) {
        self.camera_buttons.set(button, true);
        if point_in_rect(self.viewport_rect, pos) {
            self.camera_pointer = Some(pos);
        }
    }

    /// A pointer release: ends the camera drag once no buttons remain down.
    pub fn pointer_up(&mut self, button: PointerButton) {
        self.camera_buttons.set(button, false);
        if !self.camera_buttons.any() {
            self.camera_pointer = None;
        }
    }

    /// A pointer move: advances an in-progress camera drag. Returns whether
    /// the camera changed (the shell should schedule a frame).
    pub fn pointer_moved(
        &mut self,
        pos: (f32, f32),
        modifiers: KeyModifiers,
        scheme: renderer::CameraControlScheme,
    ) -> bool {
        let Some((last_x, last_y)) = self.camera_pointer else {
            return false;
        };
        if !self.camera_buttons.any() {
            return false;
        }
        let input = renderer::CameraInputState {
            left_down: self.camera_buttons.primary,
            middle_down: self.camera_buttons.middle,
            right_down: self.camera_buttons.secondary,
            shift_down: modifiers.shift,
            ctrl_down: modifiers.ctrl,
            alt_down: modifiers.alt,
            mouse_delta: Vec2::new(pos.0 - last_x, pos.1 - last_y),
            scroll_delta: 0.0,
        };
        let changed = self.viewport.apply_camera_input(&input, scheme);
        self.camera_pointer = Some(pos);
        changed
    }

    /// The pointer left the window: abandon any camera drag.
    pub fn pointer_left(&mut self) {
        self.camera_pointer = None;
        self.camera_buttons = CameraButtons::default();
    }

    /// A wheel scroll: zooms the camera when the pointer is over the viewport.
    /// Returns whether the camera consumed the scroll.
    pub fn wheel(
        &mut self,
        pos: (f32, f32),
        scroll_delta: f32,
        scheme: renderer::CameraControlScheme,
    ) -> bool {
        if !point_in_rect(self.viewport_rect, pos) {
            return false;
        }
        self.viewport.zoom_camera(scroll_delta, scheme);
        true
    }
}

/// Which camera-relevant pointer buttons are currently held, in Damascene's
/// button vocabulary (primary = left, secondary = right).
#[derive(Clone, Copy, Default)]
struct CameraButtons {
    primary: bool,
    secondary: bool,
    middle: bool,
}

impl CameraButtons {
    fn set(&mut self, button: PointerButton, down: bool) {
        match button {
            PointerButton::Primary => self.primary = down,
            PointerButton::Secondary => self.secondary = down,
            PointerButton::Middle => self.middle = down,
        }
    }

    fn any(self) -> bool {
        self.primary || self.secondary || self.middle
    }
}

struct ViewportRenderer {
    renderer: renderer::Renderer,
    target: ViewportTarget,
    surface_format: wgpu::TextureFormat,
    /// Per-output mesh cache + build bookkeeping (no GPU state; unit-tested).
    preview_cache: PreviewCache,
    /// Union bounds of the currently composited scene, for the Frame command.
    scene_bounds: Option<PreviewBounds>,
    /// The set of output ids the camera was last framed against. The camera
    /// re-frames when this set changes (an output added/removed) but not on a
    /// mere resolution/mode tweak of the same set.
    framed_ids: Option<Vec<String>>,
    pending_frame_preview: bool,
    camera: Option<renderer::Camera>,
}

pub struct ViewportRenderParams<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub logical_rect: Option<Rect>,
    pub scale_factor: f32,
    pub clear_color: wgpu::Color,
    pub preview_requests: Vec<PreviewRequest>,
}

impl ViewportRenderer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let mut renderer = renderer::Renderer::new(format);
        renderer.initialize(device, queue, None);

        let target = ViewportTarget::new(device, (1, 1), format);

        Self {
            renderer,
            target,
            surface_format: format,
            preview_cache: PreviewCache::default(),
            scene_bounds: None,
            framed_ids: None,
            pending_frame_preview: false,
            camera: None,
        }
    }

    fn app_texture(&self) -> AppTexture {
        self.target.app_texture.clone()
    }

    /// (Re)allocates the render target to match the keyed viewport rect — the
    /// renderer's real size is that rect, not the full window, so this runs
    /// whenever Damascene layout resolves a new rect rather than on window
    /// resize events.
    fn ensure_target_for_rect(
        &mut self,
        device: &wgpu::Device,
        rect: Rect,
        scale_factor: f32,
    ) -> bool {
        let extent = viewport_extent(rect, scale_factor);
        if self.target.extent == extent {
            return false;
        }

        self.target = ViewportTarget::new(device, extent, self.surface_format);
        true
    }

    fn render(&mut self, params: ViewportRenderParams<'_>) -> bool {
        let ViewportRenderParams {
            device,
            queue,
            encoder,
            logical_rect,
            scale_factor,
            clear_color,
            preview_requests,
        } = params;

        let Some(rect) = logical_rect else {
            return false;
        };

        let target_resized = self.ensure_target_for_rect(device, rect, scale_factor);
        let (w, h) = self.target.extent;

        self.renderer.set_viewport_size(device, w, h);
        let scene = self.resolve_scene(&preview_requests);
        for (mesh, transform, material) in &scene.meshes {
            self.renderer.submit_mesh(mesh, *transform, *material);
        }
        for (lines, transform, style) in &scene.lines {
            self.renderer.submit_lines(lines, *transform, style.clone());
        }
        for (points, transform, style) in &scene.points {
            self.renderer
                .submit_points(points, *transform, style.clone());
        }
        let camera = self
            .camera
            .get_or_insert_with(renderer::test_scenes::create_test_camera)
            .clone();

        let settings = render_settings(preview_requests.first(), clear_color);
        self.renderer.render(
            device,
            queue,
            encoder,
            &camera,
            &settings,
            &self.target.view,
        );
        self.renderer.end_frame();
        target_resized
    }

    fn apply_camera_input(
        &mut self,
        input: &renderer::CameraInputState,
        scheme: renderer::CameraControlScheme,
    ) -> bool {
        let Some(camera) = &mut self.camera else {
            return false;
        };

        match scheme.determine_action(input) {
            renderer::CameraAction::Orbit => {
                camera.orbit(-input.mouse_delta.x * 0.01, -input.mouse_delta.y * 0.01);
                true
            }
            renderer::CameraAction::Pan => {
                camera.pan(input.mouse_delta, Vec2::ZERO);
                true
            }
            renderer::CameraAction::Zoom => {
                let zoom_delta = if input.scroll_delta != 0.0 {
                    input.scroll_delta * 0.5
                } else {
                    input.mouse_delta.x * 0.02
                };
                camera.zoom_clamped(zoom_delta, 0.1, 1000.0);
                true
            }
            renderer::CameraAction::None => false,
        }
    }

    fn zoom_camera(&mut self, scroll_delta: f32, scheme: renderer::CameraControlScheme) {
        let input = renderer::CameraInputState {
            scroll_delta,
            ..Default::default()
        };
        let _ = self.apply_camera_input(&input, scheme);
    }

    fn apply_camera_command(&mut self, command: crate::ViewportCameraCommand) {
        match command {
            crate::ViewportCameraCommand::FramePreview => self.frame_preview(),
            crate::ViewportCameraCommand::Reset => {
                self.camera = Some(renderer::test_scenes::create_test_camera());
                self.pending_frame_preview = false;
            }
        }
    }

    /// The Frame command re-centers the camera on the current scene. The actual
    /// framing happens in `resolve_scene`, which owns the union bounds; here we
    /// only raise the request so it fires even if the scene isn't cached yet.
    fn frame_preview(&mut self) {
        self.pending_frame_preview = true;
    }

    fn accept_preview_result(&mut self, result: PreviewBuildResult) {
        self.preview_cache.accept(result);
    }

    /// Composites the cached meshes for the current output set into one scene,
    /// updating the camera: framing the union bounds when the output set changes
    /// or a Frame command is pending, and otherwise leaving the user's view. When
    /// nothing is cached yet it falls back to the renderer test scene.
    fn resolve_scene(&mut self, requests: &[PreviewRequest]) -> renderer::SceneData {
        let Some((scene, bounds, ids)) = self.preview_cache.composite(requests) else {
            // Nothing materialized yet: keep the placeholder scene, don't disturb
            // the camera the user may already have moved.
            if self.scene_bounds.is_none() {
                self.camera
                    .get_or_insert_with(renderer::test_scenes::create_test_camera);
            }
            return renderer::test_scenes::create_test_scene();
        };

        self.scene_bounds = Some(bounds);
        let reframe = self.pending_frame_preview || self.framed_ids.as_ref() != Some(&ids);
        if reframe {
            let mut camera = self.camera.take().unwrap_or_default();
            camera.focus_on(bounds.min_vec3(), bounds.max_vec3());
            self.camera = Some(camera);
            self.framed_ids = Some(ids);
            self.pending_frame_preview = false;
        }
        scene
    }
}

struct ViewportTarget {
    _texture: Arc<wgpu::Texture>,
    view: wgpu::TextureView,
    app_texture: AppTexture,
    extent: (u32, u32),
}

impl ViewportTarget {
    fn new(device: &wgpu::Device, extent: (u32, u32), format: wgpu::TextureFormat) -> Self {
        let extent = (extent.0.max(1), extent.1.max(1));
        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volumetric_ui_v2::viewport_render_target"),
            size: wgpu::Extent3d {
                width: extent.0,
                height: extent.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let app_texture = damascene_wgpu::app_texture(texture.clone());
        Self {
            _texture: texture,
            view,
            app_texture,
            extent,
        }
    }
}

/// Expands a (possibly indexed) render mesh into world-space STL triangles.
fn mesh_triangles(mesh: &renderer::MeshData, transform: Mat4, out: &mut Vec<volumetric::Triangle>) {
    let corner = |idx: usize| -> Option<((f32, f32, f32), (f32, f32, f32))> {
        let vertex = mesh.vertices.get(idx)?;
        let position = transform.transform_point3(Vec3::from(vertex.position));
        let normal = transform
            .transform_vector3(Vec3::from(vertex.normal))
            .normalize_or_zero();
        Some((position.into(), normal.into()))
    };
    let mut push = |a: usize, b: usize, c: usize| {
        if let (Some(a), Some(b), Some(c)) = (corner(a), corner(b), corner(c)) {
            out.push(volumetric::Triangle {
                vertices: [a.0, b.0, c.0],
                normals: [a.1, b.1, c.1],
            });
        }
    };
    match &mesh.indices {
        Some(indices) => {
            for tri in indices.chunks_exact(3) {
                push(tri[0] as usize, tri[1] as usize, tri[2] as usize);
            }
        }
        None => {
            for base in (0..mesh.vertices.len()).step_by(3) {
                push(base, base + 1, base + 2);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreviewSceneKey {
    asset_id: String,
    wasm_ptr: usize,
    wasm_len: usize,
    render_mode: PreviewRenderMode,
    mesh_plan: PreviewMeshPlan,
}

impl From<&PreviewRequest> for PreviewSceneKey {
    fn from(request: &PreviewRequest) -> Self {
        Self {
            asset_id: request.asset_id.clone(),
            wasm_ptr: Arc::as_ptr(&request.wasm_bytes) as usize,
            wasm_len: request.wasm_bytes.len(),
            render_mode: request.render_mode,
            mesh_plan: request.mesh_plan.clone(),
        }
    }
}

impl PreviewSceneKey {
    fn label(&self) -> String {
        self.mesh_plan.label()
    }
}

/// A single built output: its meshed geometry, world-space bounds, and the
/// meshing statistics surfaced in the UI.
#[derive(Clone)]
struct PreviewEntity {
    scene: renderer::SceneData,
    bounds: PreviewBounds,
    stats: OutputStats,
    /// Unique mesh edges, prebuilt so the wireframe toggle is display-only
    /// (composited in per frame when requested; `None` for point clouds and
    /// sketch previews).
    wireframe_lines: Option<renderer::LineData>,
}

#[derive(Clone, Copy)]
struct PreviewBounds {
    min: (f32, f32, f32),
    max: (f32, f32, f32),
}

impl PreviewBounds {
    fn min_vec3(self) -> Vec3 {
        Vec3::new(self.min.0, self.min.1, self.min.2)
    }

    fn max_vec3(self) -> Vec3 {
        Vec3::new(self.max.0, self.max.1, self.max.2)
    }

    /// The bounding box enclosing both `self` and `other`.
    fn union(self, other: PreviewBounds) -> PreviewBounds {
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

/// Per-output mesh cache and build bookkeeping. Holds no GPU state, so its
/// reconcile/accept/composite logic is unit-tested directly.
///
/// Keyed by asset id: each output has at most one cached entity, which is kept
/// (stale) until a fresh build for the same asset replaces it, so a resolution
/// or mode change never blanks an output mid-rebuild.
#[derive(Default)]
struct PreviewCache {
    entities: HashMap<String, (PreviewSceneKey, PreviewEntity)>,
    pending: HashMap<String, PreviewSceneKey>,
    failed: HashMap<String, (PreviewSceneKey, String)>,
    /// Memoized composite, rebuilt only when the contributing keys (or their
    /// wireframe flags) change.
    composite: Option<(
        Vec<(PreviewSceneKey, bool)>,
        renderer::SceneData,
        PreviewBounds,
    )>,
}

impl PreviewCache {
    /// The cached scene for one output, if a build has completed for it.
    fn entity_scene(&self, id: &str) -> Option<&renderer::SceneData> {
        self.entities.get(id).map(|(_, entity)| &entity.scene)
    }

    /// Meshing stats for every cached output, keyed by asset id.
    fn output_stats(&self) -> std::collections::BTreeMap<String, OutputStats> {
        self.entities
            .iter()
            .map(|(id, (_, entity))| (id.clone(), entity.stats.clone()))
            .collect()
    }

    /// Drops any outputs no longer requested, then returns the aggregate build
    /// status and the jobs needed to bring the requested set up to date.
    fn sync(&mut self, requests: &[PreviewRequest]) -> (PreviewBuildStatus, Vec<PreviewBuildJob>) {
        let desired: std::collections::HashSet<&str> =
            requests.iter().map(|r| r.asset_id.as_str()).collect();
        self.entities.retain(|id, _| desired.contains(id.as_str()));
        self.pending.retain(|id, _| desired.contains(id.as_str()));
        self.failed.retain(|id, _| desired.contains(id.as_str()));

        let mut jobs = Vec::new();
        for request in requests {
            let key = PreviewSceneKey::from(request);
            let id = &request.asset_id;
            let cached = self.entities.get(id).map(|(k, _)| k) == Some(&key);
            let building = self.pending.get(id) == Some(&key);
            let failed = self.failed.get(id).map(|(k, _)| k) == Some(&key);
            if cached || building || failed {
                continue;
            }
            self.failed.remove(id);
            self.pending.insert(id.clone(), key.clone());
            jobs.push(PreviewBuildJob {
                key,
                request: request.clone(),
            });
        }

        let status = if !self.pending.is_empty() {
            PreviewBuildStatus::Building {
                label: format!("{} building", self.pending.len()),
            }
        } else if let Some((_, (key, error))) = self.failed.iter().next() {
            PreviewBuildStatus::Failed {
                label: key.label(),
                error: error.clone(),
            }
        } else if self.entities.is_empty() {
            PreviewBuildStatus::Idle
        } else {
            PreviewBuildStatus::Ready {
                label: format!("{} outputs", self.entities.len()),
            }
        };
        (status, jobs)
    }

    /// Records a completed build, ignoring results superseded by a newer request.
    fn accept(&mut self, result: PreviewBuildResult) {
        let PreviewBuildResult { key, result } = result;
        let id = key.asset_id.clone();
        if self.pending.get(&id) != Some(&key) {
            return; // a newer build for this output was requested; drop this one.
        }
        self.pending.remove(&id);
        match result {
            Ok(entity) => {
                self.failed.remove(&id);
                self.entities.insert(id, (key, entity));
                self.composite = None;
            }
            Err(error) => {
                self.failed.insert(id, (key, error));
            }
        }
    }

    fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Composites the cached geometry for the requested outputs into one scene.
    /// Returns the merged scene, its union bounds, and the sorted ids of the
    /// contributing outputs (used to decide when to re-frame the camera), or
    /// `None` when no requested output has materialized yet. The result is
    /// memoized against the contributing keys so a static scene isn't re-merged
    /// every frame.
    fn composite(
        &mut self,
        requests: &[PreviewRequest],
    ) -> Option<(renderer::SceneData, PreviewBounds, Vec<String>)> {
        let mut keys = Vec::new();
        let mut ids = Vec::new();
        for request in requests {
            if let Some((key, _)) = self.entities.get(&request.asset_id) {
                keys.push((key.clone(), request.wireframe));
                ids.push(request.asset_id.clone());
            }
        }
        if keys.is_empty() {
            return None;
        }
        ids.sort();

        if self
            .composite
            .as_ref()
            .is_none_or(|(cached_keys, _, _)| cached_keys != &keys)
        {
            let mut scene = renderer::SceneData::new();
            let mut bounds: Option<PreviewBounds> = None;
            for request in requests {
                if let Some((_, entity)) = self.entities.get(&request.asset_id) {
                    scene.meshes.extend(entity.scene.meshes.iter().cloned());
                    scene.lines.extend(entity.scene.lines.iter().cloned());
                    scene.points.extend(entity.scene.points.iter().cloned());
                    if request.wireframe
                        && let Some(lines) = &entity.wireframe_lines
                    {
                        scene
                            .lines
                            .push((lines.clone(), glam::Mat4::IDENTITY, wireframe_style()));
                    }
                    bounds = Some(match bounds {
                        Some(acc) => acc.union(entity.bounds),
                        None => entity.bounds,
                    });
                }
            }
            self.composite = Some((keys, scene, bounds.expect("non-empty keys imply bounds")));
        }

        let (_, scene, bounds) = self.composite.as_ref().unwrap();
        Some((scene.clone(), *bounds, ids))
    }
}

pub struct PreviewBuildJob {
    key: PreviewSceneKey,
    request: PreviewRequest,
}

pub struct PreviewBuildResult {
    key: PreviewSceneKey,
    result: Result<PreviewEntity, String>,
}

/// A unit of background work, produced by [`Session::sync`] and consumed by
/// [`execute_job`] on whatever executor the shell provides (a thread natively;
/// inline or a web worker on wasm).
pub enum BackgroundJob {
    RunProject {
        generation: u64,
        project: volumetric::Project,
        cancel: Arc<AtomicBool>,
    },
    BuildPreview(PreviewBuildJob),
}

/// A completed unit of background work, routed by [`Session::pre_frame`].
pub enum BackgroundResult {
    ProjectComplete {
        generation: u64,
        result: Result<Vec<volumetric::LoadedAsset>, String>,
        elapsed_ms: u128,
    },
    PreviewComplete(PreviewBuildResult),
}

/// Coalescing job queue: the newest queued project run wins (a burst of edits
/// collapses to one rebuild), preview jobs coalesce per output id, and `pop`
/// always yields a pending run before any preview so a following preview sees
/// fresh assets. Executors should re-`push` everything newly queued before
/// each `pop`, so a fresh run preempts remaining previews.
#[derive(Default)]
pub struct JobQueue {
    run: Option<BackgroundJob>,
    previews: HashMap<String, PreviewBuildJob>,
}

impl JobQueue {
    pub fn push(&mut self, job: BackgroundJob) {
        match job {
            run @ BackgroundJob::RunProject { .. } => self.run = Some(run),
            BackgroundJob::BuildPreview(preview) => {
                // Newest job per output wins.
                self.previews.insert(preview.key.asset_id.clone(), preview);
            }
        }
    }

    /// The next job to execute: the pending run if any, else one preview.
    pub fn pop(&mut self) -> Option<BackgroundJob> {
        if let Some(run) = self.run.take() {
            return Some(run);
        }
        let id = self.previews.keys().next()?.clone();
        let preview = self.previews.remove(&id).expect("key just observed");
        Some(BackgroundJob::BuildPreview(preview))
    }

    pub fn is_empty(&self) -> bool {
        self.run.is_none() && self.previews.is_empty()
    }
}

/// Executes one background job to completion. Blocking; run it off the UI
/// thread (or accept the stall, as a single-threaded web shell might).
pub fn execute_job(job: BackgroundJob) -> BackgroundResult {
    match job {
        BackgroundJob::RunProject {
            generation,
            project,
            cancel,
        } => {
            // std::time::Instant panics on wasm32-unknown-unknown; swap for
            // web-time when the web shell lands.
            let start = std::time::Instant::now();
            let result = project
                .run_cancellable(&mut volumetric::Environment::new(), &cancel)
                .map_err(|err| err.to_string());
            BackgroundResult::ProjectComplete {
                generation,
                result,
                elapsed_ms: start.elapsed().as_millis(),
            }
        }
        BackgroundJob::BuildPreview(job) => {
            let result = build_preview_scene(&job.request);
            BackgroundResult::PreviewComplete(PreviewBuildResult {
                key: job.key,
                result,
            })
        }
    }
}

fn build_preview_scene(request: &PreviewRequest) -> Result<PreviewEntity, String> {
    let build_start = std::time::Instant::now();
    let mut stats = OutputStats::default();

    // 2D sketches get a flat raster preview; the 3D mesh plans don't apply.
    let dims = volumetric::model_dimensions_from_bytes(request.wasm_bytes.as_slice())
        .map_err(format_error_chain)?;
    if dims == 2 {
        return build_sketch_preview(request, build_start);
    }

    let (scene, wireframe_lines, bounds_min, bounds_max) = match &request.mesh_plan {
        PreviewMeshPlan::PointCloud { resolution } => {
            let (points, bounds_min, bounds_max) =
                volumetric::sample_model_from_bytes(request.wasm_bytes.as_slice(), *resolution)
                    .map_err(format_error_chain)?;
            stats.points = points.len();
            let mut scene = renderer::SceneData::new();
            scene.add_points(
                renderer::convert_points_to_point_data(&points),
                glam::Mat4::IDENTITY,
                renderer::PointStyle {
                    size: 4.0,
                    size_mode: renderer::WidthMode::ScreenSpace,
                    shape: renderer::PointShape::Circle,
                    depth_mode: renderer::DepthMode::Normal,
                },
            );
            (scene, None, bounds_min, bounds_max)
        }
        PreviewMeshPlan::MarchingCubes { resolution } => {
            let (triangles, bounds_min, bounds_max) =
                volumetric::generate_marching_cubes_mesh_from_bytes(
                    request.wasm_bytes.as_slice(),
                    *resolution,
                )
                .map_err(format_error_chain)?;
            stats.triangles = triangles.len();
            let vertices = triangles_to_mesh_vertices(&triangles);
            let wireframe = mesh_edge_lines(&vertices, None);
            let mut scene = renderer::SceneData::new();
            scene.add_mesh(
                renderer::MeshData {
                    vertices,
                    indices: None,
                },
                glam::Mat4::IDENTITY,
                renderer::MaterialId(0),
            );
            (scene, Some(wireframe), bounds_min, bounds_max)
        }
        PreviewMeshPlan::AdaptiveSurfaceNets2 { .. } => {
            let config = request
                .mesh_plan
                .adaptive_surface_nets_config()
                .ok_or_else(|| "missing adaptive surface nets config".to_string())?;
            let mesh = volumetric::generate_adaptive_mesh_v2_from_bytes(
                request.wasm_bytes.as_slice(),
                &config,
            )
            .map_err(format_error_chain)?;
            stats.triangles = mesh.indices.len() / 3;
            stats.samples = mesh.stats.total_samples;
            stats.detail = asn2_stage_lines(&mesh.stats);
            let vertices: Vec<renderer::MeshVertex> = mesh
                .vertices
                .iter()
                .zip(mesh.normals.iter())
                .map(|(position, normal)| renderer::MeshVertex {
                    position: (*position).into(),
                    _pad0: 0.0,
                    normal: (*normal).into(),
                    _pad1: 0.0,
                })
                .collect();
            let wireframe = mesh_edge_lines(&vertices, Some(&mesh.indices));
            let mut scene = renderer::SceneData::new();
            scene.add_mesh(
                renderer::MeshData {
                    vertices,
                    indices: Some(mesh.indices),
                },
                glam::Mat4::IDENTITY,
                renderer::MaterialId(0),
            );
            (scene, Some(wireframe), mesh.bounds_min, mesh.bounds_max)
        }
    };

    stats.mesh_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    Ok(PreviewEntity {
        scene,
        bounds: PreviewBounds {
            min: bounds_min,
            max: bounds_max,
        },
        stats,
        wireframe_lines,
    })
}

/// Style for the wireframe overlay: thin dark depth-tested lines. The line
/// pipeline uses `LessEqual` depth compare, so lines coincident with mesh
/// edges win over the faces they border.
fn wireframe_style() -> renderer::LineStyle {
    renderer::LineStyle {
        width: 1.0,
        width_mode: renderer::WidthMode::ScreenSpace,
        pattern: renderer::LinePattern::Solid,
        depth_mode: renderer::DepthMode::Normal,
    }
}

/// Unique edges of a mesh as line segments. With an index buffer, edges are
/// deduplicated by index pair; for triangle soup, by quantized endpoint
/// positions.
fn mesh_edge_lines(
    vertices: &[renderer::MeshVertex],
    indices: Option<&[u32]>,
) -> renderer::LineData {
    const COLOR: [f32; 4] = [0.05, 0.06, 0.08, 0.9];
    let mut segments = Vec::new();
    match indices {
        Some(indices) => {
            let mut seen = std::collections::HashSet::new();
            for tri in indices.chunks_exact(3) {
                for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    if seen.insert((a.min(b), a.max(b))) {
                        segments.push(renderer::LineSegment {
                            start: vertices[a as usize].position,
                            end: vertices[b as usize].position,
                            color: COLOR,
                        });
                    }
                }
            }
        }
        None => {
            let key = |p: [f32; 3]| (p[0].to_bits(), p[1].to_bits(), p[2].to_bits());
            let mut seen = std::collections::HashSet::new();
            for tri in vertices.chunks_exact(3) {
                for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
                    let (pa, pb) = (tri[a].position, tri[b].position);
                    let (ka, kb) = (key(pa), key(pb));
                    let edge_key = if ka <= kb { (ka, kb) } else { (kb, ka) };
                    if seen.insert(edge_key) {
                        segments.push(renderer::LineSegment {
                            start: pa,
                            end: pb,
                            color: COLOR,
                        });
                    }
                }
            }
        }
    }
    renderer::LineData { segments }
}

/// Per-stage profiling lines for the ASN2 mesher, shown in the output's
/// settings popover (v1 had these in a collapsible "Profiling Details").
fn asn2_stage_lines(stats: &volumetric::adaptive_surface_nets_2::MeshingStats2) -> Vec<String> {
    let ms = |secs: f64| secs * 1000.0;
    let mut lines = vec![
        format!(
            "S1 discovery {:.1} ms · {} samples · {} cells",
            ms(stats.stage1_time_secs),
            stats.stage1_samples,
            stats.stage1_mixed_cells
        ),
        format!(
            "S2 subdivide {:.1} ms · {} tris",
            ms(stats.stage2_time_secs),
            stats.stage2_triangles_emitted
        ),
        format!(
            "S3 topology {:.1} ms · {} verts",
            ms(stats.stage3_time_secs),
            stats.stage3_unique_vertices
        ),
        format!(
            "S4 refine {:.1} ms · {} samples",
            ms(stats.stage4_time_secs),
            stats.stage4_samples
        ),
    ];
    if stats.sharp_regions > 0 || stats.sharp_candidates > 0 {
        lines.push(format!(
            "S4.5 sharp {:.1} ms · {} regions · {} snapped · {} welded",
            ms(stats.stage4_5_time_secs),
            stats.sharp_regions,
            stats.sharp_snapped_edges + stats.sharp_snapped_corners,
            stats.sharp_welded_vertices
        ));
    }
    lines
}

fn render_settings(
    request: Option<&PreviewRequest>,
    clear_color: wgpu::Color,
) -> renderer::RenderSettings {
    let mut settings = renderer::RenderSettings {
        background_color: [
            clear_color.r as f32,
            clear_color.g as f32,
            clear_color.b as f32,
            clear_color.a as f32,
        ],
        ..Default::default()
    };

    if let Some(request) = request {
        settings.ssao_enabled = request.ssao;
        settings.ssao_radius = request.ssao_radius;
        settings.ssao_bias = request.ssao_bias;
        settings.ssao_strength = request.ssao_strength;
        if !request.show_grid {
            settings.grid.planes = renderer::GridPlanes::NONE;
        }
    }

    settings
}

/// Flat z=0 preview of a 2D sketch: run-length spans of occupied raster
/// cells become double-sided quads (one +z face, one -z face).
fn build_sketch_preview(
    request: &PreviewRequest,
    build_start: std::time::Instant,
) -> Result<PreviewEntity, String> {
    let resolution = sketch_raster_resolution(&request.mesh_plan);
    let raster = volumetric::rasterize_sketch_from_bytes(request.wasm_bytes.as_slice(), resolution)
        .map_err(format_error_chain)?;

    let cell_w = (raster.bounds_max.0 - raster.bounds_min.0) / raster.width as f32;
    let cell_h = (raster.bounds_max.1 - raster.bounds_min.1) / raster.height as f32;

    let mut vertices: Vec<renderer::MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut emit_quad = |x0: f32, x1: f32, y0: f32, y1: f32| {
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
        for (normal, winding) in [
            ([0.0f32, 0.0, 1.0], [0u32, 1, 2, 0, 2, 3]),
            ([0.0f32, 0.0, -1.0], [0u32, 2, 1, 0, 3, 2]),
        ] {
            let base = vertices.len() as u32;
            for (x, y) in corners {
                vertices.push(renderer::MeshVertex {
                    position: [x, y, 0.0],
                    _pad0: 0.0,
                    normal,
                    _pad1: 0.0,
                });
            }
            indices.extend(winding.iter().map(|i| base + i));
        }
    };

    for yi in 0..raster.height {
        let y0 = raster.bounds_min.1 + cell_h * yi as f32;
        let y1 = y0 + cell_h;
        let mut run_start: Option<usize> = None;
        for xi in 0..=raster.width {
            let occupied = xi < raster.width && raster.cell(xi, yi);
            match (occupied, run_start) {
                (true, None) => run_start = Some(xi),
                (false, Some(start)) => {
                    let x0 = raster.bounds_min.0 + cell_w * start as f32;
                    let x1 = raster.bounds_min.0 + cell_w * xi as f32;
                    emit_quad(x0, x1, y0, y1);
                    run_start = None;
                }
                _ => {}
            }
        }
    }

    let triangles = indices.len() / 3;
    let mut scene = renderer::SceneData::new();
    scene.add_mesh(
        renderer::MeshData {
            vertices,
            indices: Some(indices),
        },
        glam::Mat4::IDENTITY,
        renderer::MaterialId(0),
    );

    let stats = OutputStats {
        triangles,
        samples: (raster.width * raster.height) as u64,
        detail: vec![format!(
            "2D sketch raster {}x{}",
            raster.width, raster.height
        )],
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };
    Ok(PreviewEntity {
        scene,
        bounds: PreviewBounds {
            min: (raster.bounds_min.0, raster.bounds_min.1, 0.0),
            max: (raster.bounds_max.0, raster.bounds_max.1, 0.0),
        },
        stats,
        wireframe_lines: None,
    })
}

/// Raster resolution for sketch previews, reusing the output's configured
/// mesh resolution.
fn sketch_raster_resolution(plan: &PreviewMeshPlan) -> usize {
    let resolution = match plan {
        PreviewMeshPlan::PointCloud { resolution } => *resolution,
        PreviewMeshPlan::MarchingCubes { resolution } => *resolution,
        PreviewMeshPlan::AdaptiveSurfaceNets2 {
            target_resolution, ..
        } => *target_resolution,
    };
    resolution.clamp(16, 1024)
}

fn triangles_to_mesh_vertices(triangles: &[volumetric::Triangle]) -> Vec<renderer::MeshVertex> {
    let mut out = Vec::with_capacity(triangles.len() * 3);

    for tri in triangles {
        let a = tri.vertices[0];
        let b = tri.vertices[1];
        let c = tri.vertices[2];
        let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
        let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
        let face_n = (
            ab.1 * ac.2 - ab.2 * ac.1,
            ab.2 * ac.0 - ab.0 * ac.2,
            ab.0 * ac.1 - ab.1 * ac.0,
        );
        let avg_n = (
            tri.normals[0].0 + tri.normals[1].0 + tri.normals[2].0,
            tri.normals[0].1 + tri.normals[1].1 + tri.normals[2].1,
            tri.normals[0].2 + tri.normals[1].2 + tri.normals[2].2,
        );
        let dot = face_n.0 * avg_n.0 + face_n.1 * avg_n.1 + face_n.2 * avg_n.2;
        let idxs: [usize; 3] = if dot < 0.0 { [0, 2, 1] } else { [0, 1, 2] };

        for i in idxs {
            let v = tri.vertices[i];
            let n = tri.normals[i];
            let normal = if n.0 == 0.0 && n.1 == 0.0 && n.2 == 0.0 {
                [0.0, 1.0, 0.0]
            } else {
                [n.0, n.1, n.2]
            };
            out.push(renderer::MeshVertex {
                position: [v.0, v.1, v.2],
                _pad0: 0.0,
                normal,
                _pad1: 0.0,
            });
        }
    }

    out
}

fn format_error_chain(error: anyhow::Error) -> String {
    error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

fn viewport_extent(rect: Rect, scale_factor: f32) -> (u32, u32) {
    let w = (rect.w * scale_factor).ceil().max(1.0) as u32;
    let h = (rect.h * scale_factor).ceil().max(1.0) as u32;
    (w, h)
}

fn point_in_rect(rect: Option<Rect>, pos: (f32, f32)) -> bool {
    rect.is_some_and(|rect| {
        pos.0 >= rect.x && pos.0 <= rect.x + rect.w && pos.1 >= rect.y && pos.1 <= rect.y + rect.h
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: &str, resolution: usize) -> PreviewRequest {
        PreviewRequest {
            asset_id: id.to_string(),
            wasm_bytes: Arc::new(vec![1, 2, 3]),
            type_hint: None,
            precursor_ids: Vec::new(),
            render_mode: PreviewRenderMode::Points,
            mesh_plan: PreviewMeshPlan::PointCloud { resolution },
            wireframe: false,
            show_grid: true,
            ssao: true,
            ssao_radius: 0.5,
            ssao_bias: 0.025,
            ssao_strength: 1.0,
            stale: false,
        }
    }

    /// A one-point entity, so a composite's point count reveals how many outputs
    /// contributed.
    fn entity() -> PreviewEntity {
        let mut scene = renderer::SceneData::new();
        scene.add_points(
            renderer::convert_points_to_point_data(&[(0.0, 0.0, 0.0)]),
            glam::Mat4::IDENTITY,
            renderer::PointStyle {
                size: 1.0,
                size_mode: renderer::WidthMode::ScreenSpace,
                shape: renderer::PointShape::Circle,
                depth_mode: renderer::DepthMode::Normal,
            },
        );
        PreviewEntity {
            scene,
            bounds: PreviewBounds {
                min: (0.0, 0.0, 0.0),
                max: (1.0, 1.0, 1.0),
            },
            stats: OutputStats::default(),
            wireframe_lines: None,
        }
    }

    fn accept_ok(cache: &mut PreviewCache, job: PreviewBuildJob) {
        cache.accept(PreviewBuildResult {
            key: job.key,
            result: Ok(entity()),
        });
    }

    /// A 2D model output routes to the flat sketch preview instead of the
    /// 3D mesh plan: compile a circle sketch with the bundled Lua operator
    /// and build its preview scene.
    #[test]
    fn two_dimensional_outputs_get_a_sketch_preview() {
        use volumetric::wasm::OperatorExecutor;

        let lua =
            volumetric_assets::get_operator("lua_script_operator").expect("bundled lua operator");
        let script = br#"
function is_inside(x, y)
    if x*x + y*y <= 1.0 then
        return 1.0
    else
        return 0.0
    end
end
function get_bounds_min_x() return -1.5 end
function get_bounds_max_x() return 1.5 end
function get_bounds_min_y() return -1.5 end
function get_bounds_max_y() return 1.5 end
"#;
        let mut executor =
            volumetric::wasm::create_operator_executor(lua.bytes).expect("create lua executor");
        let result = executor
            .run(volumetric::wasm::OperatorIo::new(vec![script.to_vec()]))
            .expect("compile sketch");
        let sketch = result.outputs.get(&0).expect("sketch wasm").clone();

        let mut req = request("sketch", 64);
        req.wasm_bytes = Arc::new(sketch);
        let entity = build_preview_scene(&req).expect("sketch preview");

        assert!(
            entity.stats.triangles > 0,
            "sketch preview emitted no quads"
        );
        assert!(
            entity.stats.detail.iter().any(|l| l.contains("2D sketch")),
            "stats should mention the sketch raster: {:?}",
            entity.stats.detail
        );
        // Flat at z = 0, covering the sketch bounds
        assert_eq!(entity.bounds.min.2, 0.0);
        assert_eq!(entity.bounds.max.2, 0.0);
        assert_eq!(entity.bounds.min.0, -1.5);
        assert_eq!(entity.bounds.max.1, 1.5);
    }

    /// The milestone: two requested outputs each get a build job, and once both
    /// land they composite into a single multi-entity scene.
    /// Wireframe is display-only: toggling it injects the prebuilt edge
    /// lines into the composite without scheduling any rebuild.
    #[test]
    fn wireframe_toggle_composites_lines_without_rebuilding() {
        let mut cache = PreviewCache::default();
        let requests = vec![request("a", 32)];
        let (_, jobs) = cache.sync(&requests);
        assert_eq!(jobs.len(), 1);
        for job in jobs {
            let mut built = entity();
            built.wireframe_lines = Some(renderer::LineData {
                segments: vec![renderer::LineSegment {
                    start: [0.0; 3],
                    end: [1.0, 0.0, 0.0],
                    color: [0.0, 0.0, 0.0, 1.0],
                }],
            });
            cache.accept(PreviewBuildResult {
                key: job.key,
                result: Ok(built),
            });
        }
        let (scene, _, _) = cache.composite(&requests).expect("scene");
        assert!(scene.lines.is_empty(), "wireframe off: no line batches");

        let mut wire_requests = requests.clone();
        wire_requests[0].wireframe = true;
        let (_, jobs) = cache.sync(&wire_requests);
        assert!(jobs.is_empty(), "toggling wireframe must not rebuild");
        let (scene, _, _) = cache.composite(&wire_requests).expect("scene");
        assert_eq!(scene.lines.len(), 1, "wireframe on: edge lines composited");
    }

    #[test]
    fn sync_meshes_each_output_then_composites_all() {
        let mut cache = PreviewCache::default();
        let requests = vec![request("a", 8), request("b", 8)];

        let (status, jobs) = cache.sync(&requests);
        assert_eq!(jobs.len(), 2, "one build job per output");
        assert!(matches!(status, PreviewBuildStatus::Building { .. }));
        assert!(cache.composite(&requests).is_none(), "nothing built yet");

        for job in jobs {
            accept_ok(&mut cache, job);
        }

        let (status, jobs) = cache.sync(&requests);
        assert!(jobs.is_empty(), "everything cached, no rebuild");
        assert!(matches!(status, PreviewBuildStatus::Ready { .. }));

        let (scene, _bounds, ids) = cache.composite(&requests).expect("composited scene");
        assert_eq!(scene.points.len(), 2, "both outputs render together");
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    /// Changing an output's resolution requeues its build but keeps the last good
    /// mesh on screen until the new one arrives.
    #[test]
    fn resolution_change_keeps_stale_output_until_rebuilt() {
        let mut cache = PreviewCache::default();
        let coarse = vec![request("a", 8)];
        let (_, jobs) = cache.sync(&coarse);
        accept_ok(&mut cache, jobs.into_iter().next().unwrap());

        let fine = vec![request("a", 32)];
        let (status, jobs) = cache.sync(&fine);
        assert_eq!(jobs.len(), 1, "the finer resolution requeues a build");
        assert!(matches!(status, PreviewBuildStatus::Building { .. }));

        let (scene, _, _) = cache.composite(&fine).expect("stale mesh still shown");
        assert_eq!(scene.points.len(), 1);
    }

    /// An output that is no longer requested is evicted from the cache.
    #[test]
    fn dropping_an_output_evicts_it() {
        let mut cache = PreviewCache::default();
        // Clone shared requests so each output keeps one stable Arc (and thus a
        // stable cache key) across syncs, mirroring the app's `data_arc()`.
        let (a, b) = (request("a", 8), request("b", 8));
        let both = vec![a.clone(), b.clone()];
        let (_, jobs) = cache.sync(&both);
        for job in jobs {
            accept_ok(&mut cache, job);
        }

        let only_a = vec![a.clone()];
        let (_, jobs) = cache.sync(&only_a);
        assert!(jobs.is_empty());
        let (scene, _, ids) = cache.composite(&only_a).unwrap();
        assert_eq!(scene.points.len(), 1);
        assert_eq!(ids, vec!["a".to_string()]);

        // "b" was evicted, so requesting it again requires a fresh build.
        let (_, jobs) = cache.sync(&both);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].key.asset_id, "b");
    }

    /// A build result for a superseded request (its output was re-requested with
    /// different settings) is discarded rather than shown.
    #[test]
    fn superseded_build_result_is_ignored() {
        let mut cache = PreviewCache::default();
        let (_, jobs) = cache.sync(&[request("a", 8)]);
        let stale_job = jobs.into_iter().next().unwrap();

        // Re-request "a" at a new resolution before the first build returns.
        let fine = vec![request("a", 32)];
        let (_, jobs) = cache.sync(&fine);
        let fresh_job = jobs.into_iter().next().unwrap();

        // The stale build lands first and must be dropped.
        accept_ok(&mut cache, stale_job);
        assert!(cache.composite(&fine).is_none(), "stale result discarded");

        accept_ok(&mut cache, fresh_job);
        assert!(cache.composite(&fine).is_some(), "fresh result accepted");
    }

    fn preview_job(id: &str, resolution: usize) -> BackgroundJob {
        let request = request(id, resolution);
        BackgroundJob::BuildPreview(PreviewBuildJob {
            key: PreviewSceneKey::from(&request),
            request,
        })
    }

    fn run_job(generation: u64) -> BackgroundJob {
        BackgroundJob::RunProject {
            generation,
            project: VolumetricUiV2::default().project().clone(),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The queue's scheduling contract: the newest run wins, the newest preview
    /// per output wins, and a pending run always pops before any preview.
    #[test]
    fn job_queue_coalesces_and_prioritizes_runs() {
        let mut queue = JobQueue::default();
        assert!(queue.is_empty());

        queue.push(preview_job("a", 8));
        queue.push(preview_job("a", 32)); // supersedes the coarse "a" build
        queue.push(preview_job("b", 8));
        queue.push(run_job(1));
        queue.push(run_job(2)); // supersedes generation 1

        let Some(BackgroundJob::RunProject { generation, .. }) = queue.pop() else {
            panic!("run pops before previews");
        };
        assert_eq!(generation, 2, "newest run wins");

        let mut previews = Vec::new();
        while let Some(job) = queue.pop() {
            let BackgroundJob::BuildPreview(preview) = job else {
                panic!("only previews remain");
            };
            previews.push((preview.key.asset_id.clone(), preview.request.mesh_plan));
        }
        previews.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(previews.len(), 2, "one coalesced job per output");
        assert_eq!(
            previews[0],
            (
                "a".to_string(),
                PreviewMeshPlan::PointCloud { resolution: 32 }
            ),
            "newest job per output wins"
        );
        assert!(queue.is_empty());
    }
}
