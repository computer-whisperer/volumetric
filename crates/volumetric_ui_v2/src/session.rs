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
    LightboxData, LightboxMode, OutputStats, PreviewArtifactStatus, PreviewBuildStatus,
    PreviewMeshPlan, PreviewPlan, PreviewRequest, RunState, VolumetricUiV2,
};
use volumetric::AssetTypeHint;
// Model executors come from the engine's backend factory so both wasm
// backends (wasmtime natively, the JS bridge in the browser) satisfy the
// same call sites.
use volumetric::wasm::ModelExecutor as _;

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
    /// Imports and step outputs reported by the active run. They are displayed
    /// provisionally and eligible for thumbnails immediately, but only a
    /// successful terminal result promotes them into viewport/runtime state.
    staged_artifacts: Vec<volumetric::LoadedAsset>,
    /// Content-addressed thumbnail jobs currently executing on the shell's
    /// dedicated thumbnail lane.
    thumbnail_inflight: HashMap<[u8; 32], Arc<AtomicBool>>,
    /// Deterministic thumbnail failures are suppressed by content hash so a
    /// malformed/unsupported artifact cannot create a per-frame retry loop.
    thumbnail_failed: std::collections::HashSet<[u8; 32]>,
    /// Completed CPU rasters waiting for `sync` to upload them with the GPU
    /// device/queue (which `pre_frame` intentionally does not borrow).
    thumbnail_uploads: Vec<([u8; 32], volumetric::direct_preview::DirectPreviewRaster)>,
    /// Pointer position of an in-progress camera drag (a press that landed in
    /// the viewport rect). `None` when the pointer isn't driving the camera.
    camera_pointer: Option<(f32, f32)>,
    camera_buttons: CameraButtons,
    /// The viewport widget's rect from the last completed layout, used to
    /// hit-test camera input and to size the render target between frames.
    viewport_rect: Option<Rect>,
    /// The output + sampling mode a lightbox job is in flight for, so a
    /// slow job isn't re-queued every frame (a mode change re-queues).
    lightbox_inflight: Option<(String, LightboxMode)>,
    /// The viridis colorbar gradient, uploaded once and reused across
    /// lightboxes (its content never changes).
    colorbar_texture: Option<AppTexture>,
}

impl Session {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self {
            viewport: ViewportRenderer::new(device, queue, format),
            run_generation: 0,
            active_run: None,
            staged_artifacts: Vec::new(),
            thumbnail_inflight: HashMap::new(),
            thumbnail_failed: std::collections::HashSet::new(),
            thumbnail_uploads: Vec::new(),
            camera_pointer: None,
            camera_buttons: CameraButtons::default(),
            lightbox_inflight: None,
            colorbar_texture: None,
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
                        self.staged_artifacts.clear();
                        app.apply_run_result(result, elapsed_ms);
                    }
                    // Otherwise the run was superseded or cancelled; discard it.
                }
                BackgroundResult::RunProgress {
                    generation,
                    progress,
                } => {
                    if self.active_run.as_ref().map(|(g, _)| *g) == Some(generation) {
                        app.on_run_progress(progress);
                    }
                }
                BackgroundResult::ArtifactReady {
                    generation,
                    artifact,
                } => {
                    if self.active_run.as_ref().map(|(g, _)| *g) == Some(generation) {
                        upsert_artifact(&mut self.staged_artifacts, artifact.clone());
                        app.stage_run_artifact(artifact);
                    }
                }
                BackgroundResult::ThumbnailComplete {
                    source_hash,
                    result,
                } => {
                    self.thumbnail_inflight.remove(&source_hash);
                    match result {
                        Ok(Some(raster)) => {
                            self.thumbnail_failed.remove(&source_hash);
                            self.thumbnail_uploads.push((source_hash, raster));
                        }
                        Ok(None) => {}
                        Err(_) => {
                            self.thumbnail_failed.insert(source_hash);
                        }
                    }
                }
                BackgroundResult::PreviewProgress { progress, .. } => {
                    app.on_preview_progress(progress);
                }
                BackgroundResult::PreviewComplete(preview) => {
                    app.clear_preview_progress();
                    self.viewport.accept_preview_result(*preview);
                }
                BackgroundResult::LightboxComplete {
                    asset_id,
                    mode,
                    result,
                } => {
                    if self
                        .lightbox_inflight
                        .as_ref()
                        .is_some_and(|(id, inflight_mode)| {
                            *id == asset_id && *inflight_mode == mode
                        })
                    {
                        self.lightbox_inflight = None;
                    }
                    app.set_lightbox_data(&asset_id, &mode, result);
                }
                BackgroundResult::ModuleMetadataReady { name, result } => {
                    app.on_module_metadata(&name, result);
                }
                BackgroundResult::ImportMetadataReady { id, result } => {
                    app.on_import_metadata(&id, result);
                }
            }
        }

        if app.take_cancel_request() {
            if let Some((_, cancel)) = self.active_run.take() {
                cancel.store(true, Ordering::Relaxed);
            }
            self.run_generation += 1;
            self.staged_artifacts.clear();
            app.on_run_cancelled();
        }

        if app.take_mesh_cancel_request() {
            let cancelled = self.viewport.preview_cache.cancel_pending();
            app.on_mesh_builds_cancelled(cancelled);
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
        queue: &wgpu::Queue,
        requests: &[PreviewRequest],
        scale_factor: f32,
    ) -> Vec<BackgroundJob> {
        let mut jobs = Vec::new();

        if app.take_pending_run() {
            self.run_generation += 1;
            self.staged_artifacts.clear();
            let cancel = Arc::new(AtomicBool::new(false));
            self.active_run = Some((self.run_generation, cancel.clone()));
            app.set_run_state(RunState::Running);
            jobs.push(BackgroundJob::RunProject {
                generation: self.run_generation,
                project: app.project().clone(),
                cancel,
            });
        }

        for (source_hash, raster) in self.thumbnail_uploads.drain(..) {
            let texture = upload_rgba_texture(
                device,
                queue,
                "volumetric_ui_v2::artifact_thumbnail",
                raster.width,
                raster.height,
                &raster.rgba,
            );
            app.set_artifact_thumbnail(source_hash, texture);
        }

        let mut thumbnail_candidates = std::collections::HashSet::new();
        for artifact in self
            .staged_artifacts
            .iter()
            .chain(app.runtime_assets().iter())
        {
            if !matches!(artifact.type_hint(), Some(AssetTypeHint::Model) | None) {
                continue;
            }
            let source_hash = artifact.content_hash();
            if !thumbnail_candidates.insert(source_hash)
                || app.has_artifact_thumbnail(&source_hash)
                || self.thumbnail_inflight.contains_key(&source_hash)
                || self.thumbnail_failed.contains(&source_hash)
            {
                continue;
            }
            let cancel = Arc::new(AtomicBool::new(false));
            self.thumbnail_inflight.insert(source_hash, cancel.clone());
            jobs.push(BackgroundJob::BuildThumbnail(ThumbnailBuildJob {
                source_hash,
                data: artifact.data_arc(),
                cancel,
            }));
        }
        self.thumbnail_failed
            .retain(|source_hash| thumbnail_candidates.contains(source_hash));

        if let Some(name) = app.take_operator_metadata_request() {
            jobs.push(BackgroundJob::ReadOperatorMetadata { name });
        }

        // One deferred import metadata read per frame; each completion pings a
        // repaint (the shell keeps painting while `has_pending_metadata`), so
        // the queue drains steadily without a burst of byte-carrying jobs.
        if let Some((id, bytes)) = app.take_import_metadata_request() {
            jobs.push(BackgroundJob::ReadImportMetadata { id, bytes });
        }

        // Catalog warm scan: one idle-priority metadata read at a time
        // until every Add-catalog entry knows its declared metadata.
        if let Some(name) = app.catalog.take_scan_request() {
            jobs.push(BackgroundJob::ScanModuleMetadata { name });
        }

        // Artifact lifetime follows the materialized project outputs, not the
        // selected/pinned viewport set. Merely looking at another output must
        // neither discard a completed mesh nor cancel a costly in-flight one.
        let live_assets: std::collections::HashMap<&str, [u8; 32]> = app
            .runtime_assets()
            .iter()
            .map(|asset| (asset.id(), asset.content_hash()))
            .collect();
        self.viewport.preview_cache.retain_assets(&live_assets);
        let (status, preview_jobs) = self.viewport.preview_cache.sync(
            requests,
            app.auto_remesh(),
            app.take_remesh_request(),
        );
        app.set_preview_build_status(status);
        app.set_output_stats(self.viewport.preview_cache.output_stats());
        let artifact_requests = app.preview_artifact_requests();
        app.set_preview_artifacts(
            self.viewport
                .preview_cache
                .artifact_statuses(&artifact_requests),
        );
        app.set_viewport_overflow(self.viewport.frame_overflow_message());
        jobs.extend(preview_jobs.into_iter().map(BackgroundJob::BuildPreview));

        // Export modal: deliver the cached preview mesh the sync after the
        // modal opens. A synchronous copy, not a re-mesh — empty means no
        // cached mesh and the modal explains instead.
        if let Some(asset_id) = app.export_dialog_wants_mesh().map(str::to_string) {
            let triangles = self.preview_triangles(&asset_id);
            app.set_export_mesh(&asset_id, &triangles);
        }

        // Lightbox: dispatch sampling for a freshly opened inspection (or a
        // slice parameter change), and upload arrived data as textures.
        if let Some((asset_id, mode, data)) = app.lightbox_wants_data() {
            let spec = (asset_id, mode);
            if self.lightbox_inflight.as_ref() != Some(&spec) {
                self.lightbox_inflight = Some(spec.clone());
                let (asset_id, mode) = spec;
                jobs.push(BackgroundJob::BuildLightbox {
                    asset_id,
                    mode,
                    data,
                });
            }
        } else if app.lightbox_wants_texture().is_none() {
            self.lightbox_inflight = None;
        }
        if let Some(data) = app.lightbox_wants_texture() {
            let raster = upload_rgba_texture(
                device,
                queue,
                "volumetric_ui_v2::lightbox_raster",
                data.width,
                data.height,
                &data.rgba,
            );
            let colorbar = self
                .colorbar_texture
                .get_or_insert_with(|| {
                    let mut rgba = Vec::with_capacity(256 * 4);
                    for i in 0..256 {
                        let c = volumetric::viridis(i as f32 / 255.0);
                        rgba.extend(c.map(|ch| (ch * 255.0).round() as u8));
                        rgba.push(255);
                    }
                    upload_rgba_texture(
                        device,
                        queue,
                        "volumetric_ui_v2::lightbox_colorbar",
                        256,
                        1,
                        &rgba,
                    )
                })
                .clone();
            app.set_lightbox_textures(raster, colorbar);
        }

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

    pub fn has_pending_thumbnail(&self) -> bool {
        !self.thumbnail_inflight.is_empty() || !self.thumbnail_uploads.is_empty()
    }

    /// Executor replacement drops the old worker's result receiver. Cancel
    /// and forget its thumbnail jobs so `sync` can enqueue them on the new
    /// worker instead of leaving their hashes permanently marked in flight.
    pub fn abandon_thumbnail_jobs(&mut self) {
        for cancel in self.thumbnail_inflight.values() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.thumbnail_inflight.clear();
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
    /// GPU residency per output: retained buffers, uploaded once per build
    /// revision and dropped when the output leaves the viewport. Per frame
    /// only the handles are submitted — the dense geometry never travels to
    /// the device again until a rebuild lands.
    resident: HashMap<String, ResidentEntity>,
    /// Union bounds of the currently composited scene, for the Frame command.
    scene_bounds: Option<PreviewBounds>,
    /// The set of output ids the camera was last framed against. The camera
    /// re-frames when this set changes (an output added/removed) but not on a
    /// mere resolution/mode tweak of the same set.
    framed_ids: Option<Vec<String>>,
    pending_frame_preview: bool,
    camera: Option<renderer::Camera>,
    /// The viewport rect's logical size as of the last render, so camera
    /// gestures (whose deltas arrive in logical pixels) can map drag
    /// distance to world units 1:1.
    viewport_logical_size: Vec2,
}

/// One output's GPU-resident preview geometry.
struct ResidentEntity {
    /// The cache revision these buffers were uploaded from.
    revision: u64,
    scene: renderer::RetainedScene,
    /// The prebuilt edge wireframe, uploaded lazily the first frame the
    /// (display-only) toggle asks for it, then kept for the entity's
    /// lifetime so toggling stays free.
    wireframe: Option<Arc<renderer::GpuLines>>,
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
            resident: HashMap::new(),
            scene_bounds: None,
            framed_ids: None,
            pending_frame_preview: false,
            camera: None,
            viewport_logical_size: Vec2::ZERO,
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
        self.viewport_logical_size = Vec2::new(rect.w, rect.h);

        let target_resized = self.ensure_target_for_rect(device, rect, scale_factor);
        let (w, h) = self.target.extent;

        self.renderer.set_viewport_size(device, w, h);
        self.submit_scene(device, &preview_requests);
        let camera = self
            .camera
            .get_or_insert_with(renderer::test_scenes::create_test_camera);
        camera.fit_clip_planes();
        let camera = camera.clone();

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
                camera.pan(input.mouse_delta, self.viewport_logical_size);
                true
            }
            renderer::CameraAction::Zoom => {
                let zoom_delta = if input.scroll_delta != 0.0 {
                    input.scroll_delta * 0.5
                } else {
                    input.mouse_delta.x * 0.02
                };
                let (min_radius, max_radius) = zoom_limits(self.scene_bounds);
                camera.zoom_clamped(zoom_delta, min_radius, max_radius);
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

    /// A user-facing description of geometry the last rendered frame had to
    /// drop at the device's buffer size limit; `None` when everything fit.
    fn frame_overflow_message(&self) -> Option<String> {
        self.renderer.frame_overflow().map(overflow_message)
    }

    /// Submits the current output set for rendering, reconciling GPU
    /// residency with the preview cache: an output's retained buffers are
    /// created when its build revision changes and merely re-submitted (by
    /// handle) every other frame. Also updates camera framing — the union
    /// bounds are framed when the output set changes or a Frame command is
    /// pending, otherwise the user's view is left alone. When nothing is
    /// cached yet the immediate-mode placeholder test scene is drawn.
    fn submit_scene(&mut self, device: &wgpu::Device, requests: &[PreviewRequest]) {
        let visible = self.preview_cache.visible(requests);
        if visible.is_empty() {
            self.resident.clear();
            // Nothing materialized yet: keep the placeholder scene, don't
            // disturb the camera the user may already have moved.
            if self.scene_bounds.is_none() {
                self.camera
                    .get_or_insert_with(renderer::test_scenes::create_test_camera);
            }
            let scene = renderer::test_scenes::create_test_scene();
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
            return;
        }

        let mut bounds = visible[0].3.bounds;
        for (_, _, _, entity) in &visible[1..] {
            bounds = bounds.union(entity.bounds);
        }
        let mut ids: Vec<String> = visible.iter().map(|(id, ..)| (*id).to_string()).collect();
        ids.sort();
        self.scene_bounds = Some(bounds);
        let reframe = self.pending_frame_preview || self.framed_ids.as_ref() != Some(&ids);
        if reframe {
            let mut camera = self.camera.take().unwrap_or_default();
            camera.focus_on(bounds.min_vec3(), bounds.max_vec3());
            self.camera = Some(camera);
            self.framed_ids = Some(ids);
            self.pending_frame_preview = false;
        }

        let show_bounds = requests.first().is_some_and(|r| r.show_bounds);
        let visible_ids: std::collections::HashSet<&str> =
            visible.iter().map(|(id, ..)| *id).collect();
        self.resident
            .retain(|id, _| visible_ids.contains(id.as_str()));
        for (id, revision, wireframe, entity) in visible {
            if self
                .resident
                .get(id)
                .is_none_or(|resident| resident.revision != revision)
            {
                self.resident.insert(
                    id.to_string(),
                    ResidentEntity {
                        revision,
                        scene: self.renderer.create_retained_scene(device, &entity.scene),
                        wireframe: None,
                    },
                );
            }
            let resident = self
                .resident
                .get_mut(id)
                .expect("resident was just ensured");
            if wireframe
                && resident.wireframe.is_none()
                && let Some(lines) = &entity.wireframe_lines
            {
                resident.wireframe = self.renderer.create_retained_lines(
                    device,
                    lines,
                    glam::Mat4::IDENTITY,
                    &wireframe_style(),
                );
            }
            for mesh in &resident.scene.meshes {
                self.renderer.submit_retained_mesh(mesh);
            }
            for lines in &resident.scene.lines {
                self.renderer.submit_retained_lines(lines);
            }
            for points in &resident.scene.points {
                self.renderer.submit_retained_points(points);
            }
            if wireframe && let Some(lines) = &resident.wireframe {
                self.renderer.submit_retained_lines(lines);
            }
            // Subspace gizmos are generated per frame (not retained) so
            // their display extent can track the whole scene's bounds.
            if let Some(subspace) = &entity.subspace {
                submit_subspace_gizmo(&mut self.renderer, subspace, bounds);
            }
            // The bounds box is 12 segments; immediate-mode submission each
            // frame is cheaper than retaining it.
            if show_bounds {
                self.renderer.submit_lines(
                    &bounds_box_lines(entity.bounds),
                    Mat4::IDENTITY,
                    bounds_box_style(entity.bounds),
                );
            }
        }
    }
}

/// Scene-relative zoom range: a fixed minimum orbit distance would stop far
/// short of a tiny part's surface (and the clip planes follow the radius, so
/// the range must scale with the scene). Falls back to the legacy fixed
/// range until something has been framed.
fn zoom_limits(bounds: Option<PreviewBounds>) -> (f32, f32) {
    if let Some(bounds) = bounds {
        let diagonal = (bounds.max_vec3() - bounds.min_vec3()).length();
        if diagonal.is_finite() && diagonal > 0.0 {
            return (diagonal * 0.02, diagonal * 50.0);
        }
    }
    (0.1, 1000.0)
}

/// The 12 edges of an output's world-space bounding box.
fn bounds_box_lines(bounds: PreviewBounds) -> renderer::LineData {
    const COLOR: [f32; 4] = [1.0, 0.72, 0.2, 0.9];
    let xs = [bounds.min.0, bounds.max.0];
    let ys = [bounds.min.1, bounds.max.1];
    let zs = [bounds.min.2, bounds.max.2];
    let mut segments = Vec::with_capacity(12);
    for &y in &ys {
        for &z in &zs {
            segments.push(renderer::LineSegment {
                start: [xs[0], y, z],
                end: [xs[1], y, z],
                color: COLOR,
            });
        }
    }
    for &x in &xs {
        for &z in &zs {
            segments.push(renderer::LineSegment {
                start: [x, ys[0], z],
                end: [x, ys[1], z],
                color: COLOR,
            });
        }
    }
    for &x in &xs {
        for &y in &ys {
            segments.push(renderer::LineSegment {
                start: [x, y, zs[0]],
                end: [x, y, zs[1]],
                color: COLOR,
            });
        }
    }
    renderer::LineData { segments }
}

/// Style for the bounds overlay: dashed lines drawn over the scene so the
/// far edges stay visible behind the part. Dash lengths are world-space in
/// the line shader, so they scale with the box.
fn bounds_box_style(bounds: PreviewBounds) -> renderer::LineStyle {
    let extent = (bounds.max.0 - bounds.min.0)
        .max(bounds.max.1 - bounds.min.1)
        .max(bounds.max.2 - bounds.min.2)
        .max(f32::EPSILON);
    renderer::LineStyle {
        width: 1.5,
        width_mode: renderer::WidthMode::ScreenSpace,
        pattern: renderer::LinePattern::Dashed {
            dash_length: extent * 0.03,
            gap_length: extent * 0.02,
        },
        depth_mode: renderer::DepthMode::Overlay,
    }
}

/// Subspace gizmo palette: the subspace itself in cyan-blue, the oriented
/// hyperplane normal in amber, frame triads in the usual RGB = xyz.
const SUBSPACE_COLOR: [f32; 4] = [0.30, 0.75, 1.0, 0.9];
const SUBSPACE_GRID_COLOR: [f32; 4] = [0.30, 0.75, 1.0, 0.28];
const SUBSPACE_NORMAL_COLOR: [f32; 4] = [1.0, 0.72, 0.2, 0.9];
const SUBSPACE_FRAME_COLORS: [[f32; 4]; 3] = [
    [0.94, 0.35, 0.35, 0.95],
    [0.42, 0.85, 0.35, 0.95],
    [0.35, 0.55, 1.0, 0.95],
];

/// A subspace's world vector padded into viewport 3-space (2-space values
/// draw in the z = 0 plane).
fn pad3(v: &[f64]) -> Vec3 {
    Vec3::new(
        v.first().copied().unwrap_or(0.0) as f32,
        v.get(1).copied().unwrap_or(0.0) as f32,
        v.get(2).copied().unwrap_or(0.0) as f32,
    )
}

fn gizmo_segment(start: Vec3, end: Vec3, color: [f32; 4]) -> renderer::LineSegment {
    renderer::LineSegment {
        start: start.to_array(),
        end: end.to_array(),
        color,
    }
}

/// Immediate-mode gizmo for a subspace entity. The subspace is infinite,
/// so its display window is sized by the scene bounds and centered on the
/// scene center's projection onto the subspace — a bed plane under a part
/// spans the part no matter where its chart origin sits. Markers: a dot at
/// the chart origin, and for hyperplanes an amber arrow along the oriented
/// normal.
///
/// Public (with `build_preview_scene`) for the headless debug examples.
pub fn submit_subspace_gizmo(
    renderer: &mut renderer::Renderer,
    subspace: &volumetric::subspace::Subspace,
    scene: PreviewBounds,
) {
    let scene_center = (scene.min_vec3() + scene.max_vec3()) * 0.5;
    let half = ((scene.max_vec3() - scene.min_vec3()).length() * 0.55).max(1.0);
    let center = {
        let world: Vec<f64> = scene_center.to_array()[..subspace.ambient().min(3)]
            .iter()
            .map(|c| *c as f64)
            .collect();
        let (chart, _) = subspace.project(&world);
        pad3(&subspace.embed(&chart))
    };

    let origin = pad3(&subspace.origin);
    let basis: Vec<Vec3> = (0..subspace.rank())
        .map(|i| pad3(subspace.basis_vector(i)))
        .collect();

    // Depth-tested faint grid vs always-visible structural lines.
    let mut main = Vec::new();
    let mut grid = Vec::new();
    let mut points = vec![renderer::PointInstance {
        position: origin.to_array(),
        color: SUBSPACE_COLOR,
    }];

    match basis.as_slice() {
        [] => {
            let r = half * 0.12;
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                main.push(gizmo_segment(
                    origin - axis * r,
                    origin + axis * r,
                    SUBSPACE_COLOR,
                ));
            }
        }
        [dir] => {
            main.push(gizmo_segment(
                center - *dir * half,
                center + *dir * half,
                SUBSPACE_COLOR,
            ));
        }
        [u, v] => {
            let at = |su: f32, sv: f32| center + *u * (su * half) + *v * (sv * half);
            let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            for (a, b) in corners.iter().zip(corners.iter().cycle().skip(1)) {
                main.push(gizmo_segment(at(a.0, a.1), at(b.0, b.1), SUBSPACE_COLOR));
            }
            for t in [-0.5, 0.0, 0.5] {
                grid.push(gizmo_segment(at(t, -1.0), at(t, 1.0), SUBSPACE_GRID_COLOR));
                grid.push(gizmo_segment(at(-1.0, t), at(1.0, t), SUBSPACE_GRID_COLOR));
            }
        }
        // Full frame: an axis triad at the chart origin.
        triad => {
            for (b, color) in triad.iter().zip(SUBSPACE_FRAME_COLORS) {
                main.push(gizmo_segment(origin, origin + *b * (half * 0.5), color));
            }
        }
    }

    if let Some(normal) = subspace.normal() {
        let tip = center + pad3(&normal) * (half * 0.35);
        main.push(gizmo_segment(center, tip, SUBSPACE_NORMAL_COLOR));
        points.push(renderer::PointInstance {
            position: tip.to_array(),
            color: SUBSPACE_NORMAL_COLOR,
        });
    }

    if !grid.is_empty() {
        renderer.submit_lines(
            &renderer::LineData { segments: grid },
            Mat4::IDENTITY,
            renderer::LineStyle {
                width: 1.0,
                width_mode: renderer::WidthMode::ScreenSpace,
                pattern: renderer::LinePattern::Solid,
                depth_mode: renderer::DepthMode::Normal,
            },
        );
    }
    renderer.submit_lines(
        &renderer::LineData { segments: main },
        Mat4::IDENTITY,
        renderer::LineStyle {
            width: 2.0,
            width_mode: renderer::WidthMode::ScreenSpace,
            pattern: renderer::LinePattern::Solid,
            depth_mode: renderer::DepthMode::Overlay,
        },
    );
    renderer.submit_points(
        &renderer::PointData { points },
        Mat4::IDENTITY,
        renderer::PointStyle {
            size: 8.0,
            size_mode: renderer::WidthMode::ScreenSpace,
            shape: renderer::PointShape::Circle,
            depth_mode: renderer::DepthMode::Overlay,
        },
    );
}

/// The HUD warning for a frame that dropped geometry at the GPU buffer
/// size limit.
fn overflow_message(overflow: &renderer::GeometryOverflow) -> String {
    let mut dropped = Vec::new();
    if overflow.dropped_triangles > 0 {
        dropped.push(format!(
            "{} of {} triangles",
            crate::format_count(overflow.dropped_triangles),
            crate::format_count(overflow.total_triangles),
        ));
    }
    if overflow.dropped_lines > 0 {
        dropped.push(format!(
            "{} lines",
            crate::format_count(overflow.dropped_lines)
        ));
    }
    if overflow.dropped_points > 0 {
        dropped.push(format!(
            "{} points",
            crate::format_count(overflow.dropped_points)
        ));
    }
    let limit_mib = overflow.max_buffer_bytes / (1024 * 1024);
    format!(
        "over the {limit_mib} MiB GPU buffer limit — dropped {}; reduce preview resolution",
        dropped.join(", ")
    )
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

/// Uploads CPU RGBA8 pixels (sRGB, row 0 at the top) as a sampleable
/// texture for damascene `surface()` elements.
fn upload_rgba_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> AppTexture {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    queue.write_texture(
        texture.as_image_copy(),
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    damascene_wgpu::app_texture(texture)
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
pub(crate) struct PreviewSceneKey {
    pub(crate) asset_id: String,
    source_hash: [u8; 32],
    plan: PreviewPlan,
}

impl From<&PreviewRequest> for PreviewSceneKey {
    fn from(request: &PreviewRequest) -> Self {
        Self {
            asset_id: request.asset_id.clone(),
            source_hash: request.source_hash,
            plan: request.plan.clone(),
        }
    }
}

impl PreviewSceneKey {
    fn label(&self) -> String {
        self.plan.label()
    }
}

/// A single built output: its meshed geometry, world-space bounds, and the
/// meshing statistics surfaced in the UI.
///
/// Public (with `build_preview_scene`) for the headless debug examples,
/// which reproduce viewport rendering without a window.
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

#[derive(Clone, Copy)]
pub struct PreviewBounds {
    pub min: (f32, f32, f32),
    pub max: (f32, f32, f32),
}

impl PreviewBounds {
    fn min_vec3(self) -> Vec3 {
        Vec3::new(self.min.0, self.min.1, self.min.2)
    }

    fn max_vec3(self) -> Vec3 {
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

/// Per-output mesh cache and build bookkeeping. Holds no GPU state, so its
/// reconcile/accept logic is unit-tested directly; each accepted build gets
/// a monotonic revision the viewport uses to keep GPU residency in step.
///
/// Keyed by asset id: each materialized output has at most one cached entity,
/// which survives viewport selection changes and is kept stale until a fresh
/// build for the same asset replaces it. Only removal from the materialized
/// project output set evicts it.
#[derive(Default)]
struct PreviewCache {
    entities: HashMap<String, CachedBuild>,
    pending: HashMap<String, PendingBuild>,
    failed: HashMap<String, (PreviewSceneKey, String)>,
    /// Builds the user explicitly cancelled, by the key they were cancelled
    /// at: not re-dispatched until the key changes (a settings edit) or an
    /// explicit remesh clears the suppression. Without this, cancelling a
    /// still-requested build would just re-queue it on the next frame.
    declined: HashMap<String, PreviewSceneKey>,
    /// Stamps accepted builds, so GPU residency can tell a rebuild from the
    /// same still-cached entity.
    next_revision: u64,
}

/// One accepted build in the cache.
struct CachedBuild {
    key: PreviewSceneKey,
    entity: PreviewEntity,
    revision: u64,
}

/// One in-flight build: its cache key plus the cooperative cancel flag the
/// executing job polls. Superseding, dropping, or explicitly cancelling the
/// build sets the flag so the serial background worker frees up early.
struct PendingBuild {
    key: PreviewSceneKey,
    cancel: Arc<AtomicBool>,
}

/// "1 output" / "N outputs", the count phrasing shared by the aggregate
/// build-status labels (rendered behind a "building"/"stale"/"ready" prefix).
fn count_outputs(n: usize) -> String {
    if n == 1 {
        "1 output".to_string()
    } else {
        format!("{n} outputs")
    }
}

impl PreviewCache {
    /// Reconciles artifact ownership against the materialized project outputs.
    /// Visibility is intentionally absent from this operation: selected and
    /// pinned requests decide what is submitted, never what remains cached.
    fn retain_assets(&mut self, live: &std::collections::HashMap<&str, [u8; 32]>) {
        self.entities
            .retain(|id, build| live.get(id.as_str()) == Some(&build.key.source_hash));
        self.pending.retain(|id, build| {
            let keep = live.get(id.as_str()) == Some(&build.key.source_hash);
            if !keep {
                build.cancel.store(true, Ordering::Relaxed);
            }
            keep
        });
        self.failed
            .retain(|id, (key, _)| live.get(id.as_str()) == Some(&key.source_hash));
        self.declined
            .retain(|id, key| live.get(id.as_str()) == Some(&key.source_hash));
    }

    /// The cached scene for one output, if a build has completed for it.
    fn entity_scene(&self, id: &str) -> Option<&renderer::SceneData> {
        self.entities.get(id).map(|build| &build.entity.scene)
    }

    /// Meshing stats for every cached output, keyed by asset id. The
    /// entity's bounds ride along so the UI can show dimensions without a
    /// second source of truth.
    fn output_stats(&self) -> std::collections::BTreeMap<String, OutputStats> {
        self.entities
            .iter()
            .map(|(id, build)| {
                let mut stats = build.entity.stats.clone();
                stats.bounds = Some((build.entity.bounds.min, build.entity.bounds.max));
                (id.clone(), stats)
            })
            .collect()
    }

    /// Per-output lifecycle for each current render recipe. Querying every
    /// materialized output is read-only: only [`Self::sync`] dispatches work,
    /// and it receives the selected/pinned request set separately.
    fn artifact_statuses(
        &self,
        requests: &[PreviewRequest],
    ) -> std::collections::BTreeMap<String, PreviewArtifactStatus> {
        requests
            .iter()
            .map(|request| {
                let id = &request.asset_id;
                let key = PreviewSceneKey::from(request);
                let cached = self.entities.get(id);
                let pending = self.pending.get(id).map(|build| &build.key) == Some(&key);
                let failed = self
                    .failed
                    .get(id)
                    .and_then(|(failed, error)| (failed == &key).then_some(error));
                let status = if pending {
                    if cached.is_some() {
                        PreviewArtifactStatus::Updating
                    } else {
                        PreviewArtifactStatus::Building
                    }
                } else if let Some(error) = failed {
                    PreviewArtifactStatus::Failed(error.clone())
                } else if cached.map(|build| &build.key) == Some(&key) {
                    PreviewArtifactStatus::Ready
                } else if cached.is_some() || self.declined.get(id) == Some(&key) {
                    PreviewArtifactStatus::Stale
                } else {
                    PreviewArtifactStatus::Missing
                };
                (id.clone(), status)
            })
            .collect()
    }

    /// Returns the aggregate artifact status and the jobs needed to bring the
    /// currently requested viewport set up to date. Existing artifacts and
    /// in-flight builds outside that set are retained; [`Self::retain_assets`]
    /// is the only ordinary lifecycle eviction point.
    ///
    /// When `auto` is false, out-of-date outputs are counted as stale instead
    /// of dispatched; `force` (an explicit remesh request) dispatches them
    /// regardless and clears any explicit-cancel suppressions.
    fn sync(
        &mut self,
        requests: &[PreviewRequest],
        auto: bool,
        force: bool,
    ) -> (PreviewBuildStatus, Vec<PreviewBuildJob>) {
        if force {
            self.declined.clear();
        }

        let mut jobs = Vec::new();
        let mut stale = 0usize;
        for request in requests {
            let key = PreviewSceneKey::from(request);
            let id = &request.asset_id;
            let cached = self.entities.get(id).map(|build| &build.key) == Some(&key);
            let building = self.pending.get(id).map(|build| &build.key) == Some(&key);
            let failed = self.failed.get(id).map(|(k, _)| k) == Some(&key);
            if cached || building || failed {
                continue;
            }
            if self.declined.get(id) == Some(&key) || (!auto && !force) {
                stale += 1;
                continue;
            }
            // Dispatching a fresh build supersedes any in-flight build for
            // this output; signal its flag so the worker frees up early
            // instead of finishing a mesh nobody will look at.
            if let Some(superseded) = self.pending.remove(id) {
                superseded.cancel.store(true, Ordering::Relaxed);
            }
            self.declined.remove(id);
            self.failed.remove(id);
            let cancel = Arc::new(AtomicBool::new(false));
            self.pending.insert(
                id.clone(),
                PendingBuild {
                    key: key.clone(),
                    cancel: cancel.clone(),
                },
            );
            jobs.push(PreviewBuildJob {
                key,
                request: request.clone(),
                cancel,
            });
        }

        let status = if !self.pending.is_empty() {
            PreviewBuildStatus::Building {
                label: count_outputs(self.pending.len()),
            }
        } else if let Some((_, (key, error))) = self.failed.iter().next() {
            PreviewBuildStatus::Failed {
                label: key.label(),
                error: error.clone(),
            }
        } else if stale > 0 {
            PreviewBuildStatus::Stale {
                label: count_outputs(stale),
            }
        } else if self.entities.is_empty() {
            PreviewBuildStatus::Idle
        } else {
            PreviewBuildStatus::Ready {
                label: count_outputs(self.entities.len()),
            }
        };
        (status, jobs)
    }

    /// Signals every in-flight build's cancel flag, suppressing identical
    /// re-dispatch until the key changes or an explicit remesh. Returns how
    /// many builds were cancelled.
    fn cancel_pending(&mut self) -> usize {
        let count = self.pending.len();
        for (id, build) in self.pending.drain() {
            build.cancel.store(true, Ordering::Relaxed);
            self.declined.insert(id, build.key);
        }
        count
    }

    /// Records a completed build, ignoring results superseded by a newer
    /// request or already-cancelled builds.
    fn accept(&mut self, result: PreviewBuildResult) {
        let PreviewBuildResult { key, result } = result;
        let id = key.asset_id.clone();
        if self.pending.get(&id).map(|build| &build.key) != Some(&key) {
            return; // a newer build for this output was requested; drop this one.
        }
        match result {
            Ok(entity) => {
                self.pending.remove(&id);
                self.failed.remove(&id);
                self.next_revision += 1;
                self.entities.insert(
                    id,
                    CachedBuild {
                        key,
                        entity,
                        revision: self.next_revision,
                    },
                );
            }
            // A cancelled build whose pending entry still matches: someone
            // set the flag without retiring the entry (defensive; the cancel
            // paths all retire it). Retire it now with no failure record —
            // the next sync decides whether to rebuild.
            Err(PreviewBuildError::Cancelled) => {
                self.pending.remove(&id);
            }
            Err(PreviewBuildError::Failed(error)) => {
                self.pending.remove(&id);
                self.failed.insert(id, (key, error));
            }
        }
    }

    fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// The requested outputs' cached builds, in request order: asset id,
    /// build revision (bumps only when a rebuild lands, so GPU residency
    /// re-uploads exactly then), whether the wireframe overlay is requested,
    /// and the built entity. Empty when nothing has materialized yet.
    fn visible<'a>(
        &'a self,
        requests: &'a [PreviewRequest],
    ) -> Vec<(&'a str, u64, bool, &'a PreviewEntity)> {
        requests
            .iter()
            .filter_map(|request| {
                self.entities.get(&request.asset_id).map(|build| {
                    (
                        request.asset_id.as_str(),
                        build.revision,
                        request.wireframe,
                        &build.entity,
                    )
                })
            })
            .collect()
    }
}

pub struct PreviewBuildJob {
    pub(crate) key: PreviewSceneKey,
    pub(crate) request: PreviewRequest,
    /// Cooperative cancel flag; the cache holds the other end.
    pub(crate) cancel: Arc<AtomicBool>,
}

pub struct PreviewBuildResult {
    pub(crate) key: PreviewSceneKey,
    pub(crate) result: Result<PreviewEntity, PreviewBuildError>,
}

pub struct ThumbnailBuildJob {
    pub(crate) source_hash: [u8; 32],
    pub(crate) data: Arc<Vec<u8>>,
    pub(crate) cancel: Arc<AtomicBool>,
}

/// Why a preview build produced no entity.
pub enum PreviewBuildError {
    /// The build observed its cancel flag and stopped; not a failure, and
    /// never surfaced to the user.
    Cancelled,
    Failed(String),
}

/// Where the heavy compute inside background jobs executes: in this process
/// ([`LocalBackend`]) or forwarded to a build daemon
/// ([`crate::remote::RemoteBackend`]). Only the work that scales with model
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
    /// Directly sample a model into a small RGBA thumbnail. Native shells
    /// route this variant to a dedicated worker lane so it overlaps the
    /// project run that produced the artifact.
    BuildThumbnail(ThumbnailBuildJob),
    /// Sample an output for the inspection lightbox (a 2D field raster, or
    /// a slice through a 3D model's channel).
    BuildLightbox {
        asset_id: String,
        mode: LightboxMode,
        data: Arc<Vec<u8>>,
    },
    /// Read a bundled operator's declared metadata for an Add-menu insert.
    /// Reading metadata compiles the operator's wasm module on a cold cache
    /// — over a second for the larger operators in a debug build — which is
    /// why the Add click defers it here instead of blocking the UI thread.
    ReadOperatorMetadata {
        name: String,
    },
    /// Catalog warm scan: read a bundled module's (model or operator)
    /// declared metadata so the Add catalog can display it. Same work as
    /// [`Self::ReadOperatorMetadata`], but idle priority — nothing is
    /// waiting on it, so it runs after previews instead of before them.
    ScanModuleMetadata {
        name: String,
    },
    /// Read a project-embedded operator's declared metadata off the UI
    /// thread. Unlike [`Self::ReadOperatorMetadata`], the bytes travel with
    /// the job — a project import can be a rebuilt or custom operator that
    /// isn't in the bundled registry. Deferred by a cold `operator_import_info`
    /// so the wasm compile never blocks the event loop; low priority, since
    /// only panel decorations wait on it.
    ReadImportMetadata {
        id: String,
        bytes: Vec<u8>,
    },
}

/// A completed unit of background work, routed by [`Session::pre_frame`].
pub enum BackgroundResult {
    ProjectComplete {
        generation: u64,
        result: Result<Vec<volumetric::LoadedAsset>, String>,
        elapsed_ms: u128,
    },
    /// Mid-flight progress from the run tagged `generation` (never terminal;
    /// a `ProjectComplete` always follows).
    RunProgress {
        generation: u64,
        progress: volumetric::BuildProgress,
    },
    /// A preview-capable import or step output became available during a run.
    ArtifactReady {
        generation: u64,
        artifact: volumetric::LoadedAsset,
    },
    ThumbnailComplete {
        source_hash: [u8; 32],
        result: Result<Option<volumetric::direct_preview::DirectPreviewRaster>, String>,
    },
    /// Mid-flight progress from the preview build for `asset_id`.
    PreviewProgress {
        asset_id: String,
        progress: volumetric::BuildProgress,
    },
    PreviewComplete(Box<PreviewBuildResult>),
    LightboxComplete {
        asset_id: String,
        mode: LightboxMode,
        result: Result<LightboxData, String>,
    },
    /// A module metadata read finished — from an add-click read or a
    /// catalog warm scan; either way the catalog warms and any pending
    /// add completes.
    ModuleMetadataReady {
        name: String,
        result: Result<volumetric::OperatorMetadata, String>,
    },
    /// A deferred project-embedded operator metadata read finished; routed to
    /// `on_import_metadata`, which fills the import's metadata cache entry.
    ImportMetadataReady {
        id: String,
        result: Result<volumetric::OperatorMetadata, String>,
    },
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
    lightbox: Option<BackgroundJob>,
    metadata: Option<BackgroundJob>,
    /// Deferred project-import metadata reads, FIFO. Several land at once on
    /// project open; the worker takes them one at a time after previews.
    import_metadata: std::collections::VecDeque<BackgroundJob>,
    scan: Option<BackgroundJob>,
}

impl JobQueue {
    pub fn push(&mut self, job: BackgroundJob) {
        match job {
            run @ BackgroundJob::RunProject { .. } => self.run = Some(run),
            BackgroundJob::BuildPreview(preview) => {
                // Newest job per output wins.
                self.previews.insert(preview.key.asset_id.clone(), preview);
            }
            BackgroundJob::BuildThumbnail(_) => {
                unreachable!("thumbnail jobs use the shell's dedicated lane")
            }
            // One lightbox is open at a time; newest wins.
            lightbox @ BackgroundJob::BuildLightbox { .. } => self.lightbox = Some(lightbox),
            // One operator add is in flight at a time (the app serializes).
            metadata @ BackgroundJob::ReadOperatorMetadata { .. } => self.metadata = Some(metadata),
            // One catalog scan is in flight at a time (the catalog serializes).
            scan @ BackgroundJob::ScanModuleMetadata { .. } => self.scan = Some(scan),
            // Import reads accumulate; the app dedups by id, so no coalescing.
            import @ BackgroundJob::ReadImportMetadata { .. } => {
                self.import_metadata.push_back(import)
            }
        }
    }

    /// The next job to execute: a pending run first, then the lightbox (the
    /// user is actively looking at it), then operator metadata (a click is
    /// waiting on it), then one preview, then a deferred import metadata read,
    /// and only then a catalog warm scan (nothing waits on those). Import
    /// reads sit below previews so opening a project shows its geometry before
    /// warming panel decorations — and a preceding run compiles the same
    /// modules, so those reads then hit a warm module cache.
    pub fn pop(&mut self) -> Option<BackgroundJob> {
        if let Some(run) = self.run.take() {
            return Some(run);
        }
        if let Some(lightbox) = self.lightbox.take() {
            return Some(lightbox);
        }
        if let Some(metadata) = self.metadata.take() {
            return Some(metadata);
        }
        if let Some(id) = self.previews.keys().next().cloned() {
            let preview = self.previews.remove(&id).expect("key just observed");
            return Some(BackgroundJob::BuildPreview(preview));
        }
        if let Some(import) = self.import_metadata.pop_front() {
            return Some(import);
        }
        self.scan.take()
    }

    pub fn is_empty(&self) -> bool {
        self.run.is_none()
            && self.previews.is_empty()
            && self.lightbox.is_none()
            && self.metadata.is_none()
            && self.import_metadata.is_empty()
            && self.scan.is_none()
    }
}

/// Executes one background job to completion in-process. Blocking; run it
/// off the UI thread (or accept the stall, as a single-threaded web shell
/// might).
pub fn execute_job(job: BackgroundJob) -> BackgroundResult {
    execute_job_with(job, &LocalBackend)
}

/// [`execute_job`] with the heavy compute (project run, ASN2 meshing) routed
/// through `backend`; everything else executes in-process.
pub fn execute_job_with(job: BackgroundJob, backend: &dyn ExecutionBackend) -> BackgroundResult {
    execute_job_monitored(job, backend, &|_| {})
}

/// [`execute_job_with`], emitting mid-flight [`BackgroundResult::RunProgress`]
/// / [`BackgroundResult::PreviewProgress`] snapshots through `emit` while the
/// job runs (the returned result is still the terminal one). The shell's
/// worker forwards these onto its result channel so the UI can show them.
pub fn execute_job_monitored(
    job: BackgroundJob,
    backend: &dyn ExecutionBackend,
    emit: &dyn Fn(BackgroundResult),
) -> BackgroundResult {
    match job {
        BackgroundJob::RunProject {
            generation,
            project,
            cancel,
        } => {
            // web_time::Instant is std::time::Instant on native and a
            // performance.now() clock on wasm32 (std's panics there).
            let start = web_time::Instant::now();
            let artifacts = std::cell::RefCell::new(Vec::new());
            let result = backend
                .run_project(
                    &project,
                    &cancel,
                    &|progress| {
                        emit(BackgroundResult::RunProgress {
                            generation,
                            progress,
                        })
                    },
                    &|artifact| {
                        if !is_preview_artifact(artifact) {
                            return;
                        }
                        upsert_artifact(&mut artifacts.borrow_mut(), artifact.clone());
                        emit(BackgroundResult::ArtifactReady {
                            generation,
                            artifact: artifact.clone(),
                        });
                    },
                )
                .map(|exports| {
                    let mut all = artifacts.into_inner();
                    for export in exports {
                        upsert_artifact(&mut all, export);
                    }
                    all
                });
            BackgroundResult::ProjectComplete {
                generation,
                result,
                elapsed_ms: start.elapsed().as_millis(),
            }
        }
        BackgroundJob::BuildPreview(job) => {
            let asset_id = job.key.asset_id.clone();
            let progress = |progress: volumetric::BuildProgress| {
                emit(BackgroundResult::PreviewProgress {
                    asset_id: asset_id.clone(),
                    progress,
                })
            };
            let result = match build_preview_scene_monitored(
                &job.request,
                &job.cancel,
                backend,
                &progress,
            ) {
                Ok(Some(entity)) => Ok(entity),
                Ok(None) => Err(PreviewBuildError::Cancelled),
                Err(error) => Err(PreviewBuildError::Failed(error)),
            };
            BackgroundResult::PreviewComplete(Box::new(PreviewBuildResult {
                key: job.key,
                result,
            }))
        }
        BackgroundJob::BuildThumbnail(job) => {
            let result = volumetric::direct_preview::render_model_thumbnail(
                &job.data,
                96,
                96,
                40,
                &job.cancel,
            )
            .map_err(format_error_chain);
            BackgroundResult::ThumbnailComplete {
                source_hash: job.source_hash,
                result,
            }
        }
        BackgroundJob::BuildLightbox {
            asset_id,
            mode,
            data,
        } => {
            let result = match &mode {
                LightboxMode::Sketch => build_lightbox_data(&data),
                LightboxMode::Slice {
                    axis,
                    frac_percent,
                    channel,
                } => build_slice_lightbox_data(&data, *axis, *frac_percent, channel),
            };
            BackgroundResult::LightboxComplete {
                asset_id,
                mode,
                result,
            }
        }
        BackgroundJob::ReadOperatorMetadata { name } => {
            let result = match volumetric_assets::get_operator(&name) {
                Some(asset) => volumetric::operator_metadata_from_wasm_bytes(asset.bytes)
                    .map_err(|err| err.to_string()),
                None => Err(format!("missing bundled operator {name}")),
            };
            BackgroundResult::ModuleMetadataReady { name, result }
        }
        BackgroundJob::ScanModuleMetadata { name } => {
            let result = match volumetric_assets::get_asset(&name) {
                Some(asset) => volumetric::operator_metadata_from_wasm_bytes(asset.bytes)
                    .map_err(|err| err.to_string()),
                None => Err(format!("missing bundled module {name}")),
            };
            BackgroundResult::ModuleMetadataReady { name, result }
        }
        BackgroundJob::ReadImportMetadata { id, bytes } => {
            let result = volumetric::operator_metadata_from_wasm_bytes(&bytes)
                .map_err(|err| err.to_string());
            BackgroundResult::ImportMetadataReady { id, result }
        }
    }
}

fn upsert_artifact(
    artifacts: &mut Vec<volumetric::LoadedAsset>,
    artifact: volumetric::LoadedAsset,
) {
    if let Some(existing) = artifacts
        .iter_mut()
        .find(|existing| existing.id() == artifact.id())
    {
        *existing = artifact;
    } else {
        artifacts.push(artifact);
    }
}

/// Intermediate project values become UI artifacts only when the viewport has
/// a representation for them. Explicit exports are still retained below even
/// when they are opaque, preserving the project's declared public result.
fn is_preview_artifact(artifact: &volumetric::LoadedAsset) -> bool {
    matches!(
        artifact.type_hint(),
        Some(
            AssetTypeHint::Model
                | AssetTypeHint::FeaMesh
                | AssetTypeHint::TriMesh
                | AssetTypeHint::Subspace
        ) | None
    )
}

/// Samples a 2D model for the inspection lightbox: the colormapped raster
/// (pixel-exact, top row first, ready for texture upload) plus the
/// engineering analytics rows. This is where new statistics get added.
pub fn build_lightbox_data(wasm_bytes: &[u8]) -> Result<LightboxData, String> {
    const RESOLUTION: usize = 512;
    let raster = volumetric::rasterize_sketch_from_bytes(wasm_bytes, RESOLUTION)
        .map_err(format_error_chain)?;
    let binary = raster.is_binary();

    // Colormap the raster, image row order (top row first). Occupancy
    // masks draw black-on-white like the CLI PNG; scalar fields draw
    // viridis over the sampled range; NaN cells scream magenta.
    let span = (raster.value_max - raster.value_min).max(f32::EPSILON);
    let mut rgba = Vec::with_capacity(raster.width * raster.height * 4);
    for yi in (0..raster.height).rev() {
        for xi in 0..raster.width {
            let v = raster.value(xi, yi);
            let pixel: [u8; 3] = if binary {
                if raster.cell(xi, yi) {
                    [0; 3]
                } else {
                    [255; 3]
                }
            } else if !v.is_finite() {
                [255, 0, 255]
            } else {
                let c = volumetric::viridis((v - raster.value_min) / span);
                c.map(|ch| (ch * 255.0).round() as u8)
            };
            rgba.extend(pixel);
            rgba.push(255);
        }
    }

    // Analytics over the sampled grid. Cell area weights the integral, so
    // a pressure map's integral reads as total force.
    let (width_m, height_m) = (
        f64::from(raster.bounds_max.0 - raster.bounds_min.0),
        f64::from(raster.bounds_max.1 - raster.bounds_min.1),
    );
    let cell_area = (width_m / raster.width as f64) * (height_m / raster.height as f64);
    let cells = raster.values.len();
    let mut finite = 0usize;
    let mut nan_cells = 0usize;
    let mut positive = 0usize;
    let mut occupied = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &v in &raster.values {
        if !v.is_finite() {
            nan_cells += 1;
            continue;
        }
        finite += 1;
        let v = f64::from(v);
        sum += v;
        sum_sq += v * v;
        if v > 0.0 {
            positive += 1;
        }
        if v > f64::from(volumetric::OCCUPANCY_THRESHOLD) {
            occupied += 1;
        }
    }
    let mean = if finite > 0 { sum / finite as f64 } else { 0.0 };
    let rms = if finite > 0 {
        (sum_sq / finite as f64).sqrt()
    } else {
        0.0
    };
    let percent = |n: usize| 100.0 * n as f64 / cells.max(1) as f64;

    let mut analytics: Vec<(String, String)> = vec![
        (
            "raster".to_string(),
            format!("{} x {} cells", raster.width, raster.height),
        ),
        (
            "bounds".to_string(),
            format!(
                "x [{:.4}, {:.4}] · y [{:.4}, {:.4}]",
                raster.bounds_min.0, raster.bounds_max.0, raster.bounds_min.1, raster.bounds_max.1
            ),
        ),
        (
            "area".to_string(),
            format!("{:.6} (bounds rectangle)", width_m * height_m),
        ),
    ];
    if binary {
        analytics.push((
            "occupied".to_string(),
            format!(
                "{:.1}% · area {:.6}",
                percent(occupied),
                occupied as f64 * cell_area
            ),
        ));
    } else {
        analytics.push((
            "range".to_string(),
            format!("{:.6} .. {:.6}", raster.value_min, raster.value_max),
        ));
        analytics.push(("mean".to_string(), format!("{mean:.6}")));
        analytics.push(("rms".to_string(), format!("{rms:.6}")));
        analytics.push((
            "integral".to_string(),
            format!("{:.6} (sum · cell area)", sum * cell_area),
        ));
        analytics.push((
            "coverage (v > 0)".to_string(),
            format!("{:.1}%", percent(positive)),
        ));
    }
    if nan_cells > 0 {
        analytics.push((
            "non-finite cells".to_string(),
            format!("{nan_cells} ({:.1}%, magenta)", percent(nan_cells)),
        ));
    }

    Ok(LightboxData {
        rgba,
        width: raster.width as u32,
        height: raster.height as u32,
        bounds_min: raster.bounds_min,
        bounds_max: raster.bounds_max,
        binary,
        value_min: raster.value_min,
        value_max: raster.value_max,
        analytics,
    })
}

/// Samples an axis-aligned slice through a 3D model's declared channel for
/// the inspection lightbox: `position[axis]` is fixed at `min + frac% ·
/// extent` and the chosen channel is sampled over the two lateral axes
/// (ascending order, matching the FEA lateral-axis convention). Occupied
/// cells colormap viridis over their value range; unoccupied cells stay
/// white; the occupancy channel itself renders as a black-on-white mask.
pub fn build_slice_lightbox_data(
    wasm_bytes: &[u8],
    axis: usize,
    frac_percent: u16,
    channel: &str,
) -> Result<LightboxData, String> {
    const RESOLUTION: usize = 256;

    let mut executor =
        volumetric::wasm::create_model_executor(wasm_bytes).map_err(|err| err.to_string())?;
    let dimensions = executor.dimensions().map_err(|err| err.to_string())?;
    if dimensions != 3 {
        return Err(format!(
            "slice inspection needs a 3D model (got {dimensions}D)"
        ));
    }
    if axis > 2 {
        return Err(format!("slice axis {axis} out of range"));
    }
    let channel_idx = executor
        .sample_format()
        .map_err(|err| err.to_string())?
        .channels
        .iter()
        .position(|c| c.name == channel)
        .ok_or_else(|| format!("model declares no channel {channel:?}"))?;
    let masked = channel_idx != 0;

    let bounds = executor.get_bounds_nd().map_err(|err| err.to_string())?;
    let plane =
        bounds.min(axis) + (bounds.max(axis) - bounds.min(axis)) * f64::from(frac_percent) / 100.0;
    let lateral: [usize; 2] = match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let (u_min, u_max) = (bounds.min(lateral[0]), bounds.max(lateral[0]));
    let (v_min, v_max) = (bounds.min(lateral[1]), bounds.max(lateral[1]));
    let (u_extent, v_extent) = ((u_max - u_min).max(1e-9), (v_max - v_min).max(1e-9));

    // Cells proportional to the slice's aspect ratio, longer side RESOLUTION.
    let (width, height) = if u_extent >= v_extent {
        let h = ((RESOLUTION as f64 * v_extent / u_extent).round() as usize).clamp(8, RESOLUTION);
        (RESOLUTION, h)
    } else {
        let w = ((RESOLUTION as f64 * u_extent / v_extent).round() as usize).clamp(8, RESOLUTION);
        (w, RESOLUTION)
    };

    // Sample the grid at cell centers: value channel + occupancy mask.
    let mut values = vec![f32::NAN; width * height];
    let mut occupied_mask = vec![false; width * height];
    let mut position = [0.0f64; 3];
    position[axis] = plane;
    for vi in 0..height {
        position[lateral[1]] = v_min + v_extent * ((vi as f64 + 0.5) / height as f64);
        for ui in 0..width {
            position[lateral[0]] = u_min + u_extent * ((ui as f64 + 0.5) / width as f64);
            let row = executor
                .sample_channels_nd(&position)
                .map_err(|err| err.to_string())?;
            let occupied = row.first().is_some_and(|&occ| volumetric::is_occupied(occ));
            values[vi * width + ui] = row.get(channel_idx).copied().unwrap_or(f32::NAN);
            occupied_mask[vi * width + ui] = occupied;
        }
    }

    // Value range over occupied, finite cells (the colormap domain).
    let mut value_min = f32::INFINITY;
    let mut value_max = f32::NEG_INFINITY;
    let mut occupied = 0usize;
    let mut nan_cells = 0usize;
    let mut sum = 0.0f64;
    for (idx, &v) in values.iter().enumerate() {
        if !occupied_mask[idx] {
            continue;
        }
        occupied += 1;
        if !v.is_finite() {
            nan_cells += 1;
            continue;
        }
        value_min = value_min.min(v);
        value_max = value_max.max(v);
        sum += f64::from(v);
    }
    if occupied == 0 || value_min > value_max {
        (value_min, value_max) = (0.0, 0.0);
    }
    let span = (value_max - value_min).max(f32::EPSILON);

    // Colormap, image row order (top row = max lateral-v).
    let mut rgba = Vec::with_capacity(width * height * 4);
    for vi in (0..height).rev() {
        for ui in 0..width {
            let idx = vi * width + ui;
            let pixel: [u8; 3] = if !occupied_mask[idx] {
                [255; 3]
            } else if !masked {
                [0; 3]
            } else if !values[idx].is_finite() {
                [255, 0, 255]
            } else {
                let c = volumetric::viridis((values[idx] - value_min) / span);
                c.map(|ch| (ch * 255.0).round() as u8)
            };
            rgba.extend(pixel);
            rgba.push(255);
        }
    }

    let cells = width * height;
    let cell_area = (u_extent / width as f64) * (v_extent / height as f64);
    let finite_occupied = occupied - nan_cells;
    let mean = if finite_occupied > 0 {
        sum / finite_occupied as f64
    } else {
        0.0
    };
    let axis_name = ["x", "y", "z"][axis];
    let mut analytics: Vec<(String, String)> = vec![
        (
            "slice".to_string(),
            format!(
                "{axis_name} = {plane:.4} ({frac_percent}% of [{:.4}, {:.4}])",
                bounds.min(axis),
                bounds.max(axis)
            ),
        ),
        ("raster".to_string(), format!("{width} x {height} cells")),
        (
            "bounds".to_string(),
            format!("u [{u_min:.4}, {u_max:.4}] · v [{v_min:.4}, {v_max:.4}]"),
        ),
        (
            "occupied".to_string(),
            format!(
                "{:.1}% · area {:.6}",
                100.0 * occupied as f64 / cells.max(1) as f64,
                occupied as f64 * cell_area
            ),
        ),
    ];
    if masked && occupied > 0 {
        analytics.push((
            "range".to_string(),
            format!("{value_min:.6} .. {value_max:.6}"),
        ));
        analytics.push(("mean (occupied)".to_string(), format!("{mean:.6}")));
        analytics.push((
            "integral".to_string(),
            format!("{:.6} (sum · cell area)", sum * cell_area),
        ));
    }
    if nan_cells > 0 {
        analytics.push((
            "non-finite cells".to_string(),
            format!("{nan_cells} (magenta)"),
        ));
    }

    Ok(LightboxData {
        rgba,
        width: width as u32,
        height: height as u32,
        bounds_min: (u_min as f32, v_min as f32),
        bounds_max: (u_max as f32, v_max as f32),
        binary: !masked,
        value_min,
        value_max,
        analytics,
    })
}

pub fn build_preview_scene(request: &PreviewRequest) -> Result<PreviewEntity, String> {
    static NEVER: AtomicBool = AtomicBool::new(false);
    build_preview_scene_cancellable(request, &NEVER)
        .map(|entity| entity.expect("a never-set cancel flag cannot cancel the build"))
}

/// [`build_preview_scene`] with cooperative cancellation. `Ok(None)` means
/// the flag was observed set and the build was abandoned. Only the ASN2
/// meshing path (by far the longest-running plan) checks mid-build; the
/// other plans check once up front.
pub fn build_preview_scene_cancellable(
    request: &PreviewRequest,
    cancel: &AtomicBool,
) -> Result<Option<PreviewEntity>, String> {
    build_preview_scene_with(request, cancel, &LocalBackend)
}

/// [`build_preview_scene_cancellable`] with the ASN2 meshing pass routed
/// through `backend` (scene assembly, colormapping, and the cheap plans stay
/// in-process — they need the local model executor or are not worth a round
/// trip).
pub fn build_preview_scene_with(
    request: &PreviewRequest,
    cancel: &AtomicBool,
    backend: &dyn ExecutionBackend,
) -> Result<Option<PreviewEntity>, String> {
    build_preview_scene_monitored(request, cancel, backend, &|_| {})
}

/// [`build_preview_scene_with`] forwarding meshing progress to `progress`
/// (only the ASN2 plan reports; the cheap plans finish too fast to matter).
pub fn build_preview_scene_monitored(
    request: &PreviewRequest,
    cancel: &AtomicBool,
    backend: &dyn ExecutionBackend,
    progress: &dyn Fn(volumetric::BuildProgress),
) -> Result<Option<PreviewEntity>, String> {
    match preview_prelude(request, cancel)? {
        None => Ok(None),
        Some(PreviewStage::Done(entity)) => Ok(Some(*entity)),
        Some(PreviewStage::NeedsMesh(pending)) => {
            let Some(mesh) =
                backend.mesh_model(request.data.as_slice(), &pending.config, cancel, progress)?
            else {
                return Ok(None);
            };
            Ok(Some(preview_postlude(request, pending, mesh)))
        }
    }
}

/// Output of [`preview_prelude`]: either a finished preview, or an ASN2
/// plan whose meshing pass the caller must run — through
/// [`ExecutionBackend::mesh_model`] on a worker thread natively, or an
/// awaited daemon fetch on the web — before finishing with
/// [`preview_postlude`]. Split this way so the single-threaded web shell
/// can suspend at the one point that crosses the wire.
pub enum PreviewStage {
    Done(Box<PreviewEntity>),
    NeedsMesh(PendingMesh),
}

/// The ASN2 meshing request plus everything the postlude needs to finish
/// the preview once a mesh exists.
pub struct PendingMesh {
    pub config: volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2,
    color_channel: Option<String>,
    stats: OutputStats,
    build_start: web_time::Instant,
}

/// Everything of a preview build except the ASN2 meshing pass. `Ok(None)`
/// means the cancel flag was observed.
pub fn preview_prelude(
    request: &PreviewRequest,
    cancel: &AtomicBool,
) -> Result<Option<PreviewStage>, String> {
    let build_start = web_time::Instant::now();
    let mut stats = OutputStats::default();

    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }

    let done = |entity: PreviewEntity| Some(PreviewStage::Done(Box::new(entity)));

    // Explicit mesh values are data, not sampleable models: draw them
    // directly, ignoring the model mesh plans.
    if request.type_hint == Some(AssetTypeHint::FeaMesh) {
        return build_fea_mesh_preview(request, build_start).map(done);
    }
    if request.type_hint == Some(AssetTypeHint::TriMesh) {
        return build_tri_mesh_preview(request, build_start).map(done);
    }
    if request.type_hint == Some(AssetTypeHint::Subspace) {
        return build_subspace_preview(request, build_start).map(done);
    }

    // 2D sketches get a flat raster preview; the 3D mesh plans don't apply.
    let dims = volumetric::model_dimensions_from_bytes(request.data.as_slice())
        .map_err(format_error_chain)?;
    if dims == 2 {
        return build_sketch_preview(request, build_start).map(done);
    }

    // The plan normally matches the runtime dimensionality (the UI probes
    // it statically); if a model defeats the static probe, fall back to a
    // cheap default plan rather than failing.
    let fallback_plan;
    let (mesh_plan, color_channel) = match &request.plan {
        PreviewPlan::Model3d {
            mesh,
            color_channel,
            ..
        } => (mesh, color_channel.as_deref()),
        _ => {
            fallback_plan = PreviewMeshPlan::PointCloud { resolution: 24 };
            (&fallback_plan, None)
        }
    };
    let (scene, wireframe_lines, bounds_min, bounds_max) = match mesh_plan {
        PreviewMeshPlan::PointCloud { resolution } => {
            let (points, bounds_min, bounds_max) =
                volumetric::sample_model_from_bytes(request.data.as_slice(), *resolution)
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
                    request.data.as_slice(),
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
            let config = mesh_plan
                .adaptive_surface_nets_config()
                .ok_or_else(|| "missing adaptive surface nets config".to_string())?;
            return Ok(Some(PreviewStage::NeedsMesh(PendingMesh {
                config,
                color_channel: color_channel.map(str::to_string),
                stats,
                build_start,
            })));
        }
    };

    Ok(Some(PreviewStage::Done(Box::new(finish_preview_scene(
        request,
        scene,
        wireframe_lines,
        (bounds_min, bounds_max),
        color_channel.map(str::to_string),
        stats,
        build_start,
    )))))
}

/// Finishes a preview once the ASN2 mesh exists: scene assembly plus the
/// shared colormap/stats tail. Purely local — the pending state carries
/// everything the prelude gathered.
pub fn preview_postlude(
    request: &PreviewRequest,
    pending: PendingMesh,
    mesh: Arc<volumetric::AdaptiveMeshV2Result>,
) -> PreviewEntity {
    let PendingMesh {
        config: _,
        color_channel,
        mut stats,
        build_start,
    } = pending;
    stats.triangles = mesh.indices.len() / 3;
    stats.samples = mesh.stats.total_samples;
    stats.detail = asn2_stage_lines(&mesh.stats);
    // Tripwire: build results are supposed to carry one finite unit
    // normal per vertex (local builds do; a stale remote daemon once
    // shipped all-degenerate normals that rendered as uniform white).
    // Substitute +Z for anything degenerate and surface the counts.
    let mut degenerate_normals = 0usize;
    let mut nonfinite_positions = 0usize;
    if mesh.normals.len() != mesh.vertices.len() {
        stats.detail.push(format!(
            "preview guard: normals/vertices length mismatch {} vs {}",
            mesh.normals.len(),
            mesh.vertices.len()
        ));
    }
    let vertices: Vec<renderer::MeshVertex> = mesh
        .vertices
        .iter()
        .zip(mesh.normals.iter())
        .map(|(position, normal)| {
            if !(position.0.is_finite() && position.1.is_finite() && position.2.is_finite()) {
                nonfinite_positions += 1;
            }
            let len2 = normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2;
            let normal = if len2.is_finite() && len2 > 1e-20 {
                *normal
            } else {
                degenerate_normals += 1;
                (0.0, 0.0, 1.0)
            };
            renderer::MeshVertex::new((*position).into(), normal.into())
        })
        .collect();
    if degenerate_normals > 0 || nonfinite_positions > 0 {
        stats.detail.push(format!(
            "preview guard: {degenerate_normals} degenerate normals substituted,              {nonfinite_positions} non-finite positions"
        ));
    }
    let wireframe = mesh_edge_lines(&vertices, Some(&mesh.indices));
    let mut scene = renderer::SceneData::new();
    scene.add_mesh(
        renderer::MeshData {
            vertices,
            indices: Some(mesh.indices.clone()),
        },
        glam::Mat4::IDENTITY,
        renderer::MaterialId(0),
    );
    finish_preview_scene(
        request,
        scene,
        Some(wireframe),
        (mesh.bounds_min, mesh.bounds_max),
        color_channel,
        stats,
        build_start,
    )
}

/// The tail every 3D preview shares: channel discovery + colormap, timing,
/// and entity assembly.
fn finish_preview_scene(
    request: &PreviewRequest,
    mut scene: renderer::SceneData,
    wireframe_lines: Option<renderer::LineData>,
    (bounds_min, bounds_max): ((f32, f32, f32), (f32, f32, f32)),
    color_channel: Option<String>,
    mut stats: OutputStats,
    build_start: web_time::Instant,
) -> PreviewEntity {
    // Channel discovery + colormap: mirror the declared channels into the
    // stats (feeds the "Color by" picker and the slice lightbox), and when
    // a channel is selected, colormap the built points/vertices by sampling
    // it. The module is already in the executor cache from the meshing pass,
    // so this executor is cheap to create.
    match volumetric::wasm::create_model_executor(request.data.as_slice()) {
        Ok(mut executor) => {
            stats.model_channels = executor
                .sample_format()
                .map(|format| format.channels.iter().map(|c| c.name.clone()).collect())
                .unwrap_or_default();
            if let Some(channel) = &color_channel {
                match colormap_scene_by_channel(&mut scene, &mut executor, channel) {
                    Ok((value_min, value_max)) => stats.detail.push(format!(
                        "Color: {channel} in [{value_min:.4}, {value_max:.4}]"
                    )),
                    Err(err) => stats.detail.push(format!("Color: {err}")),
                }
            } else if let Some(trio) = executor
                .sample_format()
                .ok()
                .and_then(|format| format.color_trio())
            {
                // No scalar channel picked, but the model carries true
                // surface colors (e.g. a styled STEP import): render them.
                match truecolor_scene_by_trio(&mut scene, &mut executor, trio) {
                    Ok(()) => stats.detail.push("Color: model surface colors".to_string()),
                    Err(err) => stats.detail.push(format!("Color: {err}")),
                }
            } else if matches!(
                &request.plan,
                crate::PreviewPlan::Model3d {
                    tint_uncolored: true,
                    ..
                }
            ) {
                // Uncolored model with part tinting on: a muted tint keyed
                // by the output id keeps flush-fitting parts apart when
                // several are pinned into one viewport.
                let tint = part_tint(&request.asset_id);
                tint_scene(&mut scene, tint);
                stats
                    .detail
                    .push(format!("Color: part tint ({})", request.asset_id));
            }
        }
        Err(err) => {
            if color_channel.is_some() {
                stats.detail.push(format!("Color: {err}"));
            }
        }
    }

    stats.mesh_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    PreviewEntity {
        scene,
        bounds: PreviewBounds {
            min: bounds_min,
            max: bounds_max,
        },
        stats,
        wireframe_lines,
        subspace: None,
    }
}

/// Colormaps every point instance and mesh vertex of a built preview scene
/// by the named sample channel: viridis over the sampled range, magenta for
/// non-finite samples. Returns the value range the colormap spans.
fn colormap_scene_by_channel(
    scene: &mut renderer::SceneData,
    executor: &mut impl volumetric::wasm::ModelExecutor,
    channel: &str,
) -> Result<(f32, f32), String> {
    let channel_idx = executor
        .sample_format()
        .map_err(|err| err.to_string())?
        .channels
        .iter()
        .position(|c| c.name == channel)
        .ok_or_else(|| format!("channel {channel:?} not declared"))?;

    let mut sample = |position: [f32; 3]| -> Result<f32, String> {
        let row = executor
            .sample_channels_nd(&[
                f64::from(position[0]),
                f64::from(position[1]),
                f64::from(position[2]),
            ])
            .map_err(|err| err.to_string())?;
        Ok(row.get(channel_idx).copied().unwrap_or(f32::NAN))
    };

    // Sample everything first: the colormap needs the whole range.
    let mut point_values: Vec<Vec<f32>> = Vec::new();
    for (points, _, _) in &scene.points {
        let mut values = Vec::with_capacity(points.points.len());
        for point in &points.points {
            values.push(sample(point.position)?);
        }
        point_values.push(values);
    }
    let mut vertex_values: Vec<Vec<f32>> = Vec::new();
    for (mesh, _, _) in &scene.meshes {
        let mut values = Vec::with_capacity(mesh.vertices.len());
        for vertex in &mesh.vertices {
            values.push(sample(vertex.position)?);
        }
        vertex_values.push(values);
    }

    let mut value_min = f32::INFINITY;
    let mut value_max = f32::NEG_INFINITY;
    for &v in point_values.iter().chain(vertex_values.iter()).flatten() {
        if v.is_finite() {
            value_min = value_min.min(v);
            value_max = value_max.max(v);
        }
    }
    if value_min > value_max {
        (value_min, value_max) = (0.0, 0.0);
    }
    let span = (value_max - value_min).max(f32::EPSILON);
    let color_of = |v: f32| -> [f32; 4] {
        if !v.is_finite() {
            return [1.0, 0.0, 1.0, 1.0];
        }
        let c = volumetric::viridis((v - value_min) / span);
        [c[0], c[1], c[2], 1.0]
    };

    for ((points, _, _), values) in scene.points.iter_mut().zip(&point_values) {
        for (point, &v) in points.points.iter_mut().zip(values) {
            point.color = color_of(v);
        }
    }
    for ((mesh, _, _), values) in scene.meshes.iter_mut().zip(&vertex_values) {
        for (vertex, &v) in mesh.vertices.iter_mut().zip(values) {
            vertex.color = color_of(v);
        }
    }
    Ok((value_min, value_max))
}

/// Colors every point instance and mesh vertex by the model's declared
/// sRGB surface-color trio, sampled at each position and converted to
/// linear RGB for the renderer. Non-finite components fall back to white
/// (the unstyled-surface convention).
fn truecolor_scene_by_trio(
    scene: &mut renderer::SceneData,
    executor: &mut impl volumetric::wasm::ModelExecutor,
    trio: [usize; 3],
) -> Result<(), String> {
    let mut color_at = |position: [f32; 3]| -> Result<[f32; 4], String> {
        let row = executor
            .sample_channels_nd(&[
                f64::from(position[0]),
                f64::from(position[1]),
                f64::from(position[2]),
            ])
            .map_err(|err| err.to_string())?;
        let mut rgba = [1.0f32; 4];
        for (out, &idx) in rgba.iter_mut().zip(&trio) {
            let v = row.get(idx).copied().unwrap_or(f32::NAN);
            if v.is_finite() {
                *out = srgb_to_linear(v.clamp(0.0, 1.0));
            }
        }
        Ok(rgba)
    };
    for (points, _, _) in &mut scene.points {
        for point in &mut points.points {
            point.color = color_at(point.position)?;
        }
    }
    for (mesh, _, _) in &mut scene.meshes {
        for vertex in &mut mesh.vertices {
            vertex.color = color_at(vertex.position)?;
        }
    }
    Ok(())
}

/// One sRGB component to linear, the renderer's working space.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A muted, deterministic tint for an uncolored part: FNV-1a over the
/// output id (stable across sessions and platforms — pin-set changes
/// never recolor a part) picks one of twelve pastel hues, converted to
/// linear RGBA for the renderer.
pub(crate) fn part_tint(id: &str) -> [f32; 4] {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    let hue = (hash % 12) as f32 * 30.0;

    // HSL at fixed saturation/lightness: distinct but never garish.
    let (saturation, lightness) = (0.45, 0.72);
    let chroma = (1.0 - (2.0 * lightness - 1.0f32).abs()) * saturation;
    let hue_prime = hue / 60.0;
    let x = chroma * (1.0 - (hue_prime % 2.0 - 1.0).abs());
    let (r, g, b) = match hue_prime as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let m = lightness - chroma / 2.0;
    [
        srgb_to_linear(r + m),
        srgb_to_linear(g + m),
        srgb_to_linear(b + m),
        1.0,
    ]
}

/// Sets every point instance and mesh vertex of a built scene to one flat
/// color (the part-tint path; per-face colors go through the channel
/// machinery instead).
fn tint_scene(scene: &mut renderer::SceneData, color: [f32; 4]) {
    for (points, _, _) in &mut scene.points {
        for point in &mut points.points {
            point.color = color;
        }
    }
    for (mesh, _, _) in &mut scene.meshes {
        for vertex in &mut mesh.vertices {
            vertex.color = color;
        }
    }
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
            "S1 discovery {:.1} ms · {} samples · {} cells{}",
            ms(stats.stage1_time_secs),
            stats.stage1_samples,
            stats.stage1_mixed_cells,
            if stats.stage1_probe_seeds > 0 {
                format!(" · {} probe seeds", stats.stage1_probe_seeds)
            } else {
                String::new()
            }
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
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let (resolution, color_channel) = match &request.plan {
        PreviewPlan::Sketch {
            resolution,
            color_channel,
        } => (*resolution, color_channel.as_deref()),
        // Kind/dims mismatch (static probe defeated): default raster.
        _ => (256, None),
    };
    let raster = volumetric::rasterize_sketch_channel_from_bytes(
        request.data.as_slice(),
        resolution,
        color_channel,
    )
    .map_err(format_error_chain)?;

    let cell_w = (raster.bounds_max.0 - raster.bounds_min.0) / raster.width as f32;
    let cell_h = (raster.bounds_max.1 - raster.bounds_min.1) / raster.height as f32;

    let mut vertices: Vec<renderer::MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut emit_quad = |x0: f32, x1: f32, y0: f32, y1: f32, color: [f32; 4]| {
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
        for (normal, winding) in [
            ([0.0f32, 0.0, 1.0], [0u32, 1, 2, 0, 2, 3]),
            ([0.0f32, 0.0, -1.0], [0u32, 2, 1, 0, 3, 2]),
        ] {
            let base = vertices.len() as u32;
            for (x, y) in corners {
                vertices.push(renderer::MeshVertex::colored([x, y, 0.0], normal, color));
            }
            indices.extend(winding.iter().map(|i| base + i));
        }
    };

    // Cell classification for per-row run-length merging: None = empty,
    // Some(level) = draw with that level's color. Occupancy sketches keep
    // the mask look (untinted, holes where empty); scalar fields draw
    // every finite cell colormapped over the sampled value range.
    let binary = raster.is_binary();
    const LEVELS: usize = 48;
    let value_span = (raster.value_max - raster.value_min).max(f32::MIN_POSITIVE);
    let classify = |xi: usize, yi: usize| -> Option<usize> {
        if binary {
            raster.cell(xi, yi).then_some(LEVELS)
        } else {
            let v = raster.value(xi, yi);
            if !v.is_finite() {
                return None;
            }
            let t = (v - raster.value_min) / value_span;
            Some(((t * LEVELS as f32) as usize).min(LEVELS - 1))
        }
    };
    let level_color = |level: usize| -> [f32; 4] {
        if binary {
            [1.0; 4]
        } else {
            let [r, g, b] = volumetric::viridis((level as f32 + 0.5) / LEVELS as f32);
            [r, g, b, 1.0]
        }
    };

    for yi in 0..raster.height {
        let y0 = raster.bounds_min.1 + cell_h * yi as f32;
        let y1 = y0 + cell_h;
        let mut run: Option<(usize, usize)> = None; // (start, level)
        for xi in 0..=raster.width {
            let class = (xi < raster.width).then(|| classify(xi, yi)).flatten();
            if let Some((start, level)) = run
                && class != Some(level)
            {
                let x0 = raster.bounds_min.0 + cell_w * start as f32;
                let x1 = raster.bounds_min.0 + cell_w * xi as f32;
                emit_quad(x0, x1, y0, y1, level_color(level));
                run = None;
            }
            if run.is_none()
                && let Some(level) = class
            {
                run = Some((xi, level));
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

    let mut detail = vec![format!(
        "2D sketch raster {}x{}",
        raster.width, raster.height
    )];
    if !binary {
        let field = color_channel.unwrap_or("field");
        detail.push(format!(
            "{field} {:.4} .. {:.4} (viridis)",
            raster.value_min, raster.value_max
        ));
    }
    // Mirror the declared channels into the stats, like the 3D path —
    // this feeds the "Color by" picker for channeled 2D models (e.g. a
    // planar slice of a density model). The module is already in the
    // executor cache from the raster pass.
    let model_channels = volumetric::wasm::create_model_executor(request.data.as_slice())
        .ok()
        .and_then(|mut executor| {
            use volumetric::wasm::ModelExecutor;
            executor
                .sample_format()
                .map(|format| format.channels.iter().map(|c| c.name.clone()).collect())
                .ok()
        })
        .unwrap_or_default();
    let stats = OutputStats {
        triangles,
        samples: (raster.width * raster.height) as u64,
        detail,
        model_channels,
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
        subspace: None,
    })
}

/// Preview of an FEA mesh output: the mesh's boundary faces, flat-shaded
/// or colormapped by a chosen field (viridis over the field's range),
/// optionally in the deformed configuration (displacement x exaggeration),
/// with the wireframe overlay tracing element edges (quad edges only — no
/// triangulation diagonals).
fn build_fea_mesh_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let mut mesh = volumetric::fea::decode_fea_mesh(request.data.as_slice())?;
    let faces = mesh.boundary_faces();

    let (want_deformed, exaggeration, color_field) = match &request.plan {
        PreviewPlan::FeaMesh {
            deformed,
            exaggeration_tenths,
            color_field,
        } => (
            *deformed,
            f64::from(*exaggeration_tenths) / 10.0,
            color_field.clone(),
        ),
        // Plan/kind mismatch (shouldn't happen): the default view.
        _ => (true, 1.0, None),
    };

    // Every colormappable field, mirrored to the settings popover's picker
    // through the stats.
    let fea_fields: Vec<String> = mesh
        .node_fields
        .iter()
        .filter(|f| f.components == 1 || f.components == 3)
        .map(|f| format!("node:{}", f.name))
        .chain(
            mesh.element_fields
                .iter()
                .filter(|f| f.components == 1)
                .map(|f| format!("element:{}", f.name)),
        )
        .collect();

    // Resolve the colormapped field to one scalar per node or per element
    // (3-component node fields color by magnitude).
    enum ColorSource {
        Node(Vec<f64>),
        Element(Vec<f64>),
    }
    let mut extra_detail = Vec::new();
    let color_source = color_field.as_deref().and_then(|qualified| {
        let (container, name) = qualified.split_once(':')?;
        let scalars = |field: &volumetric::fea::FeaField| -> Option<Vec<f64>> {
            match field.components {
                1 => Some(field.data.clone()),
                3 => Some(
                    field
                        .data
                        .chunks_exact(3)
                        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                        .collect(),
                ),
                _ => None,
            }
        };
        let source = match container {
            "node" => mesh
                .node_fields
                .iter()
                .find(|f| f.name == name)
                .and_then(scalars)
                .map(ColorSource::Node),
            "element" => mesh
                .element_fields
                .iter()
                .find(|f| f.name == name && f.components == 1)
                .map(|f| ColorSource::Element(f.data.clone())),
            _ => None,
        };
        if source.is_none() {
            extra_detail.push(format!("colormap field {qualified} not in this mesh"));
        }
        source
    });
    let color_range = color_source.as_ref().map(|source| {
        let values = match source {
            ColorSource::Node(v) | ColorSource::Element(v) => v,
        };
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in values {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if lo > hi { (0.0, 0.0) } else { (lo, hi) }
    });
    if let (Some(field), Some((lo, hi))) = (&color_field, color_range) {
        extra_detail.push(format!("colormap {field} {lo:.4} .. {hi:.4} (viridis)"));
    }
    let color_for = |value: f64| -> [f32; 4] {
        let (lo, hi) = color_range.unwrap_or((0.0, 0.0));
        let t = if hi > lo {
            ((value - lo) / (hi - lo)) as f32
        } else {
            0.5
        };
        let [r, g, b] = volumetric::viridis(t);
        [r, g, b, 1.0]
    };

    // Deformed configuration: positions + displacement x exaggeration;
    // connectivity (and thus the boundary) is unchanged.
    if want_deformed
        && let Some(displacement) = mesh
            .node_fields
            .iter()
            .find(|f| f.name == "displacement" && f.components == 3)
    {
        let max_u = displacement
            .data
            .chunks_exact(3)
            .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
            .fold(0.0f64, f64::max);
        if (exaggeration - 1.0).abs() > 1e-9 {
            extra_detail.push(format!("deformed x{exaggeration} · max |u| = {max_u:.4}"));
        } else {
            extra_detail.push(format!("deformed view · max |u| = {max_u:.4}"));
        }
        let data = displacement.data.clone();
        for (p, u) in mesh.node_positions.iter_mut().zip(&data) {
            *p += u * exaggeration;
        }
    }
    if let Some(contact) = mesh
        .node_fields
        .iter()
        .find(|f| f.name == "contact_force" && f.components == 3)
    {
        let total_fz: f64 = contact.data.chunks_exact(3).map(|f| f[2]).sum();
        let touching = contact
            .data
            .chunks_exact(3)
            .filter(|f| f[0] != 0.0 || f[1] != 0.0 || f[2] != 0.0)
            .count();
        if touching > 0 {
            extra_detail.push(format!("contact Fz = {total_fz:.4} over {touching} nodes"));
        }
    }

    let position = |node: u32| -> [f32; 3] {
        let p = mesh.node_position(node as usize);
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };

    // Flat-shaded triangle soup: two triangles per boundary quad, each with
    // its own face normal (deformed meshes can have non-planar quads),
    // corners carrying the colormap color (white when no field is chosen).
    let mut vertices: Vec<renderer::MeshVertex> = Vec::with_capacity(faces.len() * 6);
    let mut emit_triangle = |corners: [([f32; 3], [f32; 4]); 3]| {
        let [(a, _), (b, _), (c, _)] = corners;
        let (ab, ac) = (Vec3::from(b) - Vec3::from(a), Vec3::from(c) - Vec3::from(a));
        let normal = ab.cross(ac).normalize_or_zero().to_array();
        for (p, color) in corners {
            vertices.push(renderer::MeshVertex::colored(p, normal, color));
        }
    };
    for (element, quad) in &faces {
        let corner = |slot: usize| -> ([f32; 3], [f32; 4]) {
            let node = quad[slot];
            let color = match &color_source {
                Some(ColorSource::Node(values)) => color_for(values[node as usize]),
                Some(ColorSource::Element(values)) => color_for(values[*element as usize]),
                None => [1.0; 4],
            };
            (position(node), color)
        };
        let [a, b, c, d] = [corner(0), corner(1), corner(2), corner(3)];
        emit_triangle([a, b, c]);
        emit_triangle([a, c, d]);
    }

    // Wireframe from the quads' perimeter edges, deduplicated by node pair.
    let mut seen = std::collections::HashSet::new();
    let mut segments = Vec::new();
    for (_, quad) in &faces {
        for (a, b) in [
            (quad[0], quad[1]),
            (quad[1], quad[2]),
            (quad[2], quad[3]),
            (quad[3], quad[0]),
        ] {
            if seen.insert((a.min(b), a.max(b))) {
                segments.push(renderer::LineSegment {
                    start: position(a),
                    end: position(b),
                    color: [0.05, 0.06, 0.08, 0.9],
                });
            }
        }
    }

    // Bar2 strut meshes have no boundary faces: draw every strut as a
    // hexagonal capped prism at the mesh's own `radius` field (the base
    // radius — the scale-to-radius exponent belongs to the realization
    // operator's config, so the designed radii are what the strut_model
    // output's preview shows; colormap element:stiffness_scale to see the
    // design here). Radial side normals shade the prisms as round tubes;
    // joints rely on overlap at shared nodes rather than sphere blending.
    if mesh.element_kind == volumetric::fea::FeaElementKind::Bar2 {
        let radius_field = mesh
            .element_fields
            .iter()
            .find(|f| f.name == "radius" && f.components == 1);
        // Fallback for meshes without radii: a tenth of the mean strut
        // length, so the structure still reads.
        let mean_length = {
            let total: f32 = (0..mesh.element_count())
                .map(|e| {
                    let p = position(mesh.element(e)[0]);
                    let q = position(mesh.element(e)[1]);
                    (Vec3::from(q) - Vec3::from(p)).length()
                })
                .sum();
            total / mesh.element_count().max(1) as f32
        };
        let fallback_radius = (mean_length / 10.0).max(1e-6);

        const SIDES: usize = 6;
        for e in 0..mesh.element_count() {
            let [na, nb] = [mesh.element(e)[0], mesh.element(e)[1]];
            let a = Vec3::from(position(na));
            let b = Vec3::from(position(nb));
            let axis = b - a;
            let length = axis.length();
            if !(length.is_finite() && length > 0.0) {
                continue;
            }
            let axis = axis / length;
            let radius = radius_field
                .map(|f| f.data[e] as f32)
                .filter(|r| r.is_finite() && *r > 0.0)
                .unwrap_or(fallback_radius);
            let (color_a, color_b) = match &color_source {
                Some(ColorSource::Element(values)) => {
                    let c = color_for(values[e]);
                    (c, c)
                }
                Some(ColorSource::Node(values)) => (
                    color_for(values[na as usize]),
                    color_for(values[nb as usize]),
                ),
                None => {
                    let c = [0.82, 0.85, 0.9, 1.0];
                    (c, c)
                }
            };

            // An orthonormal ring basis perpendicular to the strut.
            let seed = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let u = axis.cross(seed).normalize();
            let w = axis.cross(u);
            let ring_dir = |k: usize| -> Vec3 {
                let theta = std::f32::consts::TAU * k as f32 / SIDES as f32;
                u * theta.cos() + w * theta.sin()
            };

            // Sides: radial (smooth) normals, colors interpolating the
            // strut's ends.
            for k in 0..SIDES {
                let (d0, d1) = (ring_dir(k), ring_dir((k + 1) % SIDES));
                let quad = [
                    (a + d0 * radius, d0, color_a),
                    (b + d0 * radius, d0, color_b),
                    (b + d1 * radius, d1, color_b),
                    (a + d1 * radius, d1, color_a),
                ];
                for idx in [0, 1, 2, 0, 2, 3] {
                    let (p, n, c) = quad[idx];
                    vertices.push(renderer::MeshVertex::colored(p.to_array(), n.to_array(), c));
                }
            }
            // Flat end caps (fans anchored at ring vertex 0).
            for k in 1..SIDES - 1 {
                for (p0, p1, p2, normal, color) in [
                    (
                        a + ring_dir(0) * radius,
                        a + ring_dir(k + 1) * radius,
                        a + ring_dir(k) * radius,
                        -axis,
                        color_a,
                    ),
                    (
                        b + ring_dir(0) * radius,
                        b + ring_dir(k) * radius,
                        b + ring_dir(k + 1) * radius,
                        axis,
                        color_b,
                    ),
                ] {
                    for p in [p0, p1, p2] {
                        vertices.push(renderer::MeshVertex::colored(
                            p.to_array(),
                            normal.to_array(),
                            color,
                        ));
                    }
                }
            }
        }
    }

    // Point1 clouds: one small octahedron marker per point, sized from
    // the typical per-point volume so the cloud reads at any scale.
    // Very large clouds are stride-decimated for the preview only (the
    // full data still flows downstream); the detail line reports it.
    let mut point_preview_note = None;
    if mesh.element_kind == volumetric::fea::FeaElementKind::Point1 {
        const MAX_MARKERS: usize = 150_000;
        let count = mesh.element_count();
        let stride = count.div_ceil(MAX_MARKERS).max(1);
        if stride > 1 {
            point_preview_note = Some(format!(
                "preview shows {} of {count} points",
                count.div_ceil(stride)
            ));
        }

        // Extents from the (undeformed-independent) node positions.
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for n in 0..mesh.node_count() {
            let p = position(n as u32);
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let extent: [f32; 3] = std::array::from_fn(|a| (hi[a] - lo[a]).max(0.0));
        let volume = extent.iter().product::<f32>();
        let spacing = if volume > 0.0 {
            (volume / count.max(1) as f32).cbrt()
        } else {
            // Degenerate (coplanar/collinear/single) clouds: fall back to
            // the largest extent, or unit scale for a single point.
            let longest = extent.iter().cloned().fold(0.0f32, f32::max);
            if longest > 0.0 { longest / 20.0 } else { 1.0 }
        };
        let radius = (spacing * 0.15).max(1e-6);

        // Octahedron: 6 vertices, 8 faces, flat-shaded by face normal.
        const AXES: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for e in (0..count).step_by(stride) {
            let node = mesh.element(e)[0];
            let c = Vec3::from(position(node));
            let color = match &color_source {
                Some(ColorSource::Node(values)) => color_for(values[node as usize]),
                Some(ColorSource::Element(values)) => color_for(values[e]),
                None => [0.82, 0.85, 0.9, 1.0],
            };
            for sx in [1.0f32, -1.0] {
                for sy in [1.0f32, -1.0] {
                    for sz in [1.0f32, -1.0] {
                        let x = c + Vec3::from(AXES[0]) * (radius * sx);
                        let y = c + Vec3::from(AXES[1]) * (radius * sy);
                        let z = c + Vec3::from(AXES[2]) * (radius * sz);
                        // Wind outward: flip vertex order for negative-
                        // parity octants.
                        let (a, b) = if sx * sy * sz > 0.0 { (y, z) } else { (z, y) };
                        let normal = ((a - x).cross(b - x)).normalize_or_zero();
                        for p in [x, a, b] {
                            vertices.push(renderer::MeshVertex::colored(
                                p.to_array(),
                                normal.to_array(),
                                color,
                            ));
                        }
                    }
                }
            }
        }
    }

    let mut bounds = PreviewBounds {
        min: (f32::INFINITY, f32::INFINITY, f32::INFINITY),
        max: (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
    };
    for n in 0..mesh.node_count() {
        let p = position(n as u32);
        bounds.min = (
            bounds.min.0.min(p[0]),
            bounds.min.1.min(p[1]),
            bounds.min.2.min(p[2]),
        );
        bounds.max = (
            bounds.max.0.max(p[0]),
            bounds.max.1.max(p[1]),
            bounds.max.2.max(p[2]),
        );
    }
    if mesh.node_count() == 0 {
        bounds = PreviewBounds {
            min: (-1.0, -1.0, -1.0),
            max: (1.0, 1.0, 1.0),
        };
    }

    let mut detail = vec![match mesh.element_kind {
        volumetric::fea::FeaElementKind::Bar2 => format!(
            "FEA strut mesh: {} nodes · {} struts",
            mesh.node_count(),
            mesh.element_count()
        ),
        volumetric::fea::FeaElementKind::Point1 => {
            format!("Point cloud: {} points", mesh.element_count())
        }
        volumetric::fea::FeaElementKind::Hex8 => format!(
            "FEA mesh: {} nodes · {} elements · {} boundary faces",
            mesh.node_count(),
            mesh.element_count(),
            faces.len()
        ),
    }];
    detail.extend(point_preview_note);
    detail.extend(extra_detail);
    let stats = OutputStats {
        triangles: vertices.len() / 3,
        detail,
        fea_fields,
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };

    let mut scene = renderer::SceneData::new();
    if !vertices.is_empty() {
        scene.add_mesh(
            renderer::MeshData {
                vertices,
                indices: None,
            },
            glam::Mat4::IDENTITY,
            renderer::MaterialId(0),
        );
    }

    Ok(PreviewEntity {
        scene,
        bounds,
        stats,
        wireframe_lines: Some(renderer::LineData { segments }),
        subspace: None,
    })
}

/// Preview of a general triangle mesh: the triangles exactly as they are,
/// drawn double-sided so open and non-manifold meshes (a scan with holes, a
/// single free triangle) render cleanly from every angle, with a wireframe
/// of the unique edges.
fn build_tri_mesh_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let mesh = volumetric::trimesh::decode_tri_mesh(request.data.as_slice())?;

    let position = |vertex: u32| -> [f32; 3] {
        let p = mesh.position(vertex as usize);
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };

    let mut vertices: Vec<renderer::MeshVertex> = Vec::with_capacity(mesh.triangle_count() * 6);
    let mut emit = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
        let (ab, ac) = (Vec3::from(b) - Vec3::from(a), Vec3::from(c) - Vec3::from(a));
        let normal = ab.cross(ac).normalize_or_zero().to_array();
        for p in [a, b, c] {
            vertices.push(renderer::MeshVertex::new(p, normal));
        }
    };
    for t in 0..mesh.triangle_count() {
        let [i, j, k] = mesh.triangle(t);
        let (a, b, c) = (position(i), position(j), position(k));
        emit(a, b, c);
        emit(a, c, b); // back face, so open meshes show from both sides
    }

    let mut seen = std::collections::HashSet::new();
    let mut segments = Vec::new();
    for t in 0..mesh.triangle_count() {
        let [i, j, k] = mesh.triangle(t);
        for (a, b) in [(i, j), (j, k), (k, i)] {
            if seen.insert((a.min(b), a.max(b))) {
                segments.push(renderer::LineSegment {
                    start: position(a),
                    end: position(b),
                    color: [0.05, 0.06, 0.08, 0.9],
                });
            }
        }
    }

    let bounds = match mesh.bounds() {
        Some(b) => PreviewBounds {
            min: (b[0] as f32, b[2] as f32, b[4] as f32),
            max: (b[1] as f32, b[3] as f32, b[5] as f32),
        },
        None => PreviewBounds {
            min: (-1.0, -1.0, -1.0),
            max: (1.0, 1.0, 1.0),
        },
    };

    let stats = OutputStats {
        triangles: vertices.len() / 3,
        detail: vec![format!(
            "triangle mesh: {} vertices · {} triangles",
            mesh.vertex_count(),
            mesh.triangle_count()
        )],
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };

    let mut scene = renderer::SceneData::new();
    scene.add_mesh(
        renderer::MeshData {
            vertices,
            indices: None,
        },
        glam::Mat4::IDENTITY,
        renderer::MaterialId(0),
    );

    Ok(PreviewEntity {
        scene,
        bounds,
        stats,
        wireframe_lines: Some(renderer::LineData { segments }),
        subspace: None,
    })
}

/// Preview for a Subspace value: decode and validate now, but bake no
/// geometry — the gizmo is generated per frame in `submit_scene`, sized
/// to the whole scene. The placeholder bounds around the chart origin
/// give the camera something to frame when the gizmo is alone.
fn build_subspace_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let subspace = volumetric::subspace::decode_subspace(request.data.as_slice())?;
    if subspace.ambient() > 3 {
        return Err(format!(
            "cannot draw a subspace of {}-space in the 3D viewport",
            subspace.ambient()
        ));
    }
    let kind = match subspace.rank() {
        0 => "point",
        1 => "line",
        2 => "plane",
        _ => "frame",
    };
    let stats = OutputStats {
        detail: vec![format!(
            "subspace: {kind} (rank {}) in {}-space",
            subspace.rank(),
            subspace.ambient()
        )],
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };
    let origin = pad3(&subspace.origin);
    Ok(PreviewEntity {
        scene: renderer::SceneData::new(),
        bounds: PreviewBounds {
            min: (origin.x - 1.0, origin.y - 1.0, origin.z - 1.0),
            max: (origin.x + 1.0, origin.y + 1.0, origin.z + 1.0),
        },
        stats,
        wireframe_lines: None,
        subspace: Some(subspace),
    })
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
            out.push(renderer::MeshVertex::new([v.0, v.1, v.2], normal));
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

    /// The catalog warm-scan job reads a bundled *model's* declared
    /// metadata through the real executor — the path the Add catalog
    /// fills itself from.
    #[test]
    fn scan_job_reads_model_metadata() {
        let result = execute_job(BackgroundJob::ScanModuleMetadata {
            name: "simple_sphere_model".to_string(),
        });
        let BackgroundResult::ModuleMetadataReady { name, result } = result else {
            panic!("scan job must produce ModuleMetadataReady");
        };
        assert_eq!(name, "simple_sphere_model");
        let metadata = result.expect("sphere metadata reads");
        assert_eq!(metadata.display_name, "Simple Sphere");
        assert!(metadata.inputs.is_empty() && metadata.outputs.is_empty());
    }

    /// A monitored ASN2 preview build must emit `PreviewProgress` snapshots
    /// through the local backend (meshing stage transitions report
    /// unconditionally). Per-step run progress is covered by
    /// `Project::run_monitored`'s own tests; the default project here has an
    /// empty timeline, so it must report nothing.
    #[test]
    fn monitored_jobs_emit_progress() {
        use std::sync::Mutex;

        let sphere = volumetric_assets::get_model("simple_sphere_model")
            .expect("bundled sphere model")
            .bytes
            .to_vec();
        let source_hash = volumetric::content_fingerprint(&sphere);
        let request = PreviewRequest {
            asset_id: "sphere".to_string(),
            source_hash,
            data: Arc::new(sphere),
            type_hint: Some(volumetric::AssetTypeHint::Model),
            precursor_ids: vec![],
            plan: PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::AdaptiveSurfaceNets2 {
                    target_resolution: 32,
                    base_resolution: 8,
                    max_depth: 2,
                    settings: crate::Asn2Settings::default(),
                },
                color_channel: None,
                tint_uncolored: false,
            },
            wireframe: false,
            show_grid: false,
            show_bounds: false,
            ssao: false,
            ssao_radius: 0.5,
            ssao_bias: 0.025,
            ssao_strength: 1.0,
            stale: false,
        };
        let job = PreviewBuildJob {
            key: (&request).into(),
            request,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        let phases: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let result =
            execute_job_monitored(BackgroundJob::BuildPreview(job), &LocalBackend, &|result| {
                if let BackgroundResult::PreviewProgress { asset_id, progress } = result {
                    assert_eq!(asset_id, "sphere");
                    phases.lock().unwrap().push(progress.phase);
                }
            });
        assert!(matches!(result, BackgroundResult::PreviewComplete(_)));
        let phases = phases.into_inner().unwrap();
        assert!(
            phases.iter().any(|p| p.starts_with("refining")),
            "expected a refinement phase, got {phases:?}"
        );

        let project = crate::VolumetricUiV2::default().project().clone();
        let steps: Mutex<Vec<volumetric::BuildProgress>> = Mutex::new(Vec::new());
        let result = execute_job_monitored(
            BackgroundJob::RunProject {
                generation: 1,
                project,
                cancel: Arc::new(AtomicBool::new(false)),
            },
            &LocalBackend,
            &|result| {
                if let BackgroundResult::RunProgress { progress, .. } = result {
                    steps.lock().unwrap().push(progress);
                }
            },
        );
        assert!(matches!(
            result,
            BackgroundResult::ProjectComplete { result: Ok(_), .. }
        ));
        assert!(
            steps.into_inner().unwrap().is_empty(),
            "an empty timeline has no steps to report"
        );
    }

    #[test]
    fn project_job_stages_previewable_intermediates_before_terminal_result() {
        use std::sync::Mutex;

        let echo = r#"(module
            (import "host" "get_input_len" (func $len (param i32) (result i32)))
            (import "host" "get_input_data" (func $data (param i32 i32 i32)))
            (import "host" "post_output" (func $post (param i32 i32 i32)))
            (memory (export "memory") 16)
            (func (export "run")
                (local $n i32)
                (local.set $n (call $len (i32.const 0)))
                (call $data (i32.const 0) (i32.const 0) (local.get $n))
                (call $post (i32.const 0) (i32.const 0) (local.get $n))))"#
            .as_bytes()
            .to_vec();
        let model = volumetric_assets::get_model("simple_sphere_model")
            .expect("bundled sphere model")
            .bytes
            .to_vec();
        let mut project = volumetric::Project::new();
        project
            .imports_mut()
            .push(volumetric::ImportedAsset::operator("echo".into(), echo));
        project.imports_mut().push(volumetric::ImportedAsset::new(
            "source".into(),
            model,
            Some(AssetTypeHint::Model),
        ));
        project.timeline_mut().push(volumetric::ExecutionStep {
            operator_id: "echo".into(),
            inputs: vec![volumetric::ExecutionInput::AssetRef("source".into())],
            outputs: vec!["intermediate".into()],
        });
        project.exports_mut().push("intermediate".into());

        let staged = Mutex::new(Vec::new());
        let terminal = execute_job_monitored(
            BackgroundJob::RunProject {
                generation: 7,
                project,
                cancel: Arc::new(AtomicBool::new(false)),
            },
            &LocalBackend,
            &|event| {
                if let BackgroundResult::ArtifactReady {
                    generation,
                    artifact,
                } = event
                {
                    assert_eq!(generation, 7);
                    staged.lock().unwrap().push(artifact.id().to_string());
                }
            },
        );

        assert_eq!(&*staged.lock().unwrap(), &["source", "intermediate"]);
        let BackgroundResult::ProjectComplete {
            result: Ok(artifacts),
            ..
        } = terminal
        else {
            panic!("project must complete successfully");
        };
        assert_eq!(
            artifacts.iter().map(|asset| asset.id()).collect::<Vec<_>>(),
            ["source", "intermediate"]
        );
    }

    /// The GPU-limit HUD warning names each dropped geometry class and the
    /// limit it hit.
    #[test]
    fn overflow_message_reads_well() {
        let message = overflow_message(&renderer::GeometryOverflow {
            dropped_triangles: 1_951_244,
            total_triangles: 1_951_244,
            dropped_lines: 1_200,
            dropped_points: 0,
            max_buffer_bytes: 256 * 1024 * 1024,
        });
        assert_eq!(
            message,
            "over the 256 MiB GPU buffer limit — dropped 2.0M of 2.0M triangles, \
             1.2k lines; reduce preview resolution"
        );
    }

    /// A single unit hex with a scalar node field, a vector node field
    /// (displacement) and a scalar element field, for FEA preview tests.
    fn one_hex_mesh() -> volumetric::fea::FeaMesh {
        use volumetric::fea::{FeaField, FeaMesh};
        let positions: Vec<f64> = (0..8)
            .flat_map(|i| {
                [
                    f64::from(i & 1),
                    f64::from((i >> 1) & 1),
                    f64::from((i >> 2) & 1),
                ]
            })
            .collect();
        FeaMesh {
            element_kind: volumetric::fea::FeaElementKind::Hex8,
            node_positions: positions,
            connectivity: (0..8).collect(),
            node_fields: vec![
                FeaField {
                    name: "temp".to_string(),
                    components: 1,
                    data: (0..8).map(f64::from).collect(),
                },
                FeaField {
                    name: "displacement".to_string(),
                    components: 3,
                    data: (0..8).flat_map(|_| [0.0, 0.0, 0.5]).collect(),
                },
            ],
            element_fields: vec![FeaField {
                name: "stiffness_scale".to_string(),
                components: 1,
                data: vec![0.25],
            }],
        }
    }

    fn fea_request(plan: PreviewPlan) -> PreviewRequest {
        let data = volumetric::fea::encode_fea_mesh(&one_hex_mesh());
        PreviewRequest {
            asset_id: "fea".to_string(),
            source_hash: volumetric::content_fingerprint(&data),
            data: Arc::new(data),
            type_hint: Some(AssetTypeHint::FeaMesh),
            precursor_ids: Vec::new(),
            plan,
            wireframe: false,
            show_bounds: false,
            show_grid: true,
            ssao: false,
            ssao_radius: 0.5,
            ssao_bias: 0.025,
            ssao_strength: 1.0,
            stale: false,
        }
    }

    /// A 2D model over [0,1]^2 whose sample is `lo` where x < 0.5 and `hi`
    /// elsewhere.
    fn step_field(lo: f32, hi: f32) -> Vec<u8> {
        wat::parse_str(format!(
            r#"(module
                (memory (export "memory") 1)
                (func (export "get_dimensions") (result i32) (i32.const 2))
                (func (export "get_io_ptr") (result i32) (i32.const 1024))
                (func (export "get_bounds") (param $out i32)
                    (f64.store (local.get $out) (f64.const 0))
                    (f64.store offset=8 (local.get $out) (f64.const 1))
                    (f64.store offset=16 (local.get $out) (f64.const 0))
                    (f64.store offset=24 (local.get $out) (f64.const 1)))
                (func (export "sample") (param $pos i32) (result f32)
                    (select (f32.const {lo}) (f32.const {hi})
                        (f64.lt (f64.load (local.get $pos)) (f64.const 0.5))))
            )"#
        ))
        .expect("field module assembles")
    }

    fn analytic(data: &LightboxData, label: &str) -> String {
        data.analytics
            .iter()
            .find(|(l, _)| l == label)
            .unwrap_or_else(|| panic!("missing analytics row {label}: {:?}", data.analytics))
            .1
            .clone()
    }

    /// A 3D model over [0,1]^3, occupied where x < 0.5, declaring a
    /// `density` channel equal to the sample's z coordinate.
    fn density_model() -> Vec<u8> {
        let format = volumetric::encode_sample_format(&volumetric::SampleFormat {
            channels: vec![
                volumetric::SampleChannel {
                    name: "occupancy".to_string(),
                    kind: volumetric::ChannelKind::Occupancy,
                },
                volumetric::SampleChannel {
                    name: "density".to_string(),
                    kind: volumetric::ChannelKind::Density,
                },
            ],
        });
        let data: String = format.iter().map(|b| format!("\\{b:02x}")).collect();
        let packed = 2048_i64 | ((format.len() as i64) << 32);
        wat::parse_str(format!(
            r#"(module
                (memory (export "memory") 1)
                (data (i32.const 2048) "{data}")
                (func (export "get_dimensions") (result i32) (i32.const 3))
                (func (export "get_io_ptr") (result i32) (i32.const 1024))
                (func (export "get_bounds") (param $out i32)
                    (f64.store (local.get $out) (f64.const 0))
                    (f64.store offset=8 (local.get $out) (f64.const 1))
                    (f64.store offset=16 (local.get $out) (f64.const 0))
                    (f64.store offset=24 (local.get $out) (f64.const 1))
                    (f64.store offset=32 (local.get $out) (f64.const 0))
                    (f64.store offset=40 (local.get $out) (f64.const 1)))
                (func $occ (param $pos i32) (result f32)
                    (select (f32.const 1) (f32.const 0)
                        (f64.lt (f64.load (local.get $pos)) (f64.const 0.5))))
                (func (export "sample") (param $pos i32) (result f32)
                    (call $occ (local.get $pos)))
                (func (export "get_sample_format") (result i64) (i64.const {packed}))
                (func (export "sample_channels") (param $pos i32) (param $out i32)
                    (f32.store (local.get $out) (call $occ (local.get $pos)))
                    (f32.store offset=4 (local.get $out)
                        (f32.demote_f64 (f64.load offset=16 (local.get $pos)))))
            )"#
        ))
        .expect("density model assembles")
    }

    #[test]
    fn slice_lightbox_masks_by_occupancy_and_reads_the_channel() {
        let model = density_model();

        // z midplane: density == 0.5 across the occupied half (x < 0.5).
        let data = build_slice_lightbox_data(&model, 2, 50, "density").expect("slice builds");
        assert!(!data.binary);
        assert!((data.value_min - 0.5).abs() < 1e-6);
        assert!((data.value_max - 0.5).abs() < 1e-6);
        assert!(analytic(&data, "occupied").starts_with("50.0%"));
        assert!(analytic(&data, "slice").starts_with("z = 0.5"));
        // Left half viridis, right half white (unoccupied).
        let row = data.width as usize;
        assert_ne!(&data.rgba[0..3], &[255, 255, 255]);
        assert_eq!(
            &data.rgba[(row - 1) * 4..(row - 1) * 4 + 3],
            &[255, 255, 255]
        );

        // x slice inside the occupied half: density spans z.
        let data = build_slice_lightbox_data(&model, 0, 25, "density").expect("slice builds");
        assert!(analytic(&data, "occupied").starts_with("100.0%"));
        assert!(data.value_min < 0.01 && data.value_max > 0.99);

        // x slice in the empty half: nothing occupied, no value rows.
        let data = build_slice_lightbox_data(&model, 0, 75, "density").expect("slice builds");
        assert!(analytic(&data, "occupied").starts_with("0.0%"));
        assert!(!data.analytics.iter().any(|(l, _)| l == "range"));

        // The occupancy channel itself renders as a black-on-white mask.
        let data = build_slice_lightbox_data(&model, 2, 50, "occupancy").expect("slice builds");
        assert!(data.binary);
        assert!(
            data.rgba
                .chunks_exact(4)
                .all(|px| px[0..3] == [0, 0, 0] || px[0..3] == [255, 255, 255])
        );

        // Unknown channels fail with a readable error.
        let err = build_slice_lightbox_data(&model, 2, 50, "nope").unwrap_err();
        assert!(err.contains("no channel"), "{err}");
    }

    #[test]
    fn preview_colors_points_by_declared_channel() {
        let model = density_model();
        let request = PreviewRequest {
            asset_id: "density_points".to_string(),
            source_hash: volumetric::content_fingerprint(&model),
            data: Arc::new(model),
            type_hint: None,
            precursor_ids: vec![],
            plan: PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::PointCloud { resolution: 8 },
                color_channel: Some("density".to_string()),
                tint_uncolored: false,
            },
            wireframe: false,
            show_bounds: false,
            show_grid: false,
            ssao: false,
            ssao_radius: 0.1,
            ssao_bias: 0.02,
            ssao_strength: 1.0,
            stale: false,
        };
        let entity = build_preview_scene(&request).expect("preview builds");

        assert_eq!(
            entity.stats.model_channels,
            vec!["occupancy".to_string(), "density".to_string()],
            "declared channels mirror into the stats"
        );
        assert!(
            entity
                .stats
                .detail
                .iter()
                .any(|line| line.starts_with("Color: density")),
            "colormap range reported: {:?}",
            entity.stats.detail
        );

        // Points along the density (z) gradient get distinct colors.
        let (points, _, _) = &entity.scene.points[0];
        let mut colors: Vec<[u8; 3]> = points
            .points
            .iter()
            .map(|p| {
                p.color[0..3]
                    .iter()
                    .map(|c| (c * 255.0) as u8)
                    .collect::<Vec<_>>()
            })
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        colors.sort();
        colors.dedup();
        assert!(colors.len() > 2, "expected a gradient, got {colors:?}");

        // An undeclared channel degrades to a detail note, not a failure.
        let request = PreviewRequest {
            plan: PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::PointCloud { resolution: 8 },
                color_channel: Some("nope".to_string()),
                tint_uncolored: false,
            },
            ..request
        };
        let entity = build_preview_scene(&request).expect("preview still builds");
        assert!(
            entity
                .stats
                .detail
                .iter()
                .any(|line| line.contains("not declared")),
            "{:?}",
            entity.stats.detail
        );
    }

    #[test]
    fn part_tint_distinguishes_uncolored_models() {
        // The density model declares extra channels but no color trio —
        // an "uncolored" model for tinting purposes.
        let model = density_model();
        let request = |id: &str, tint: bool| PreviewRequest {
            asset_id: id.to_string(),
            source_hash: volumetric::content_fingerprint(&model),
            data: Arc::new(model.clone()),
            type_hint: None,
            precursor_ids: vec![],
            plan: PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::PointCloud { resolution: 8 },
                color_channel: None,
                tint_uncolored: tint,
            },
            wireframe: false,
            show_bounds: false,
            show_grid: false,
            ssao: false,
            ssao_radius: 0.1,
            ssao_bias: 0.02,
            ssao_strength: 1.0,
            stale: false,
        };

        let flat_color = |entity: &PreviewEntity| -> [f32; 4] {
            let (points, _, _) = &entity.scene.points[0];
            let first = points.points[0].color;
            assert!(
                points.points.iter().all(|p| p.color == first),
                "part tint must be flat across the model"
            );
            first
        };

        // Disabled: points keep the default position-gradient colors
        // (not one flat tint).
        let entity = build_preview_scene(&request("tray", false)).expect("preview builds");
        let (points, _, _) = &entity.scene.points[0];
        assert!(
            points.points.iter().any(|p| p.color != points.points[0].color),
            "tint off must leave the default gradient"
        );

        // Enabled: the tint is the deterministic per-id pastel, muted
        // (never white or a saturated primary), and distinct between ids
        // that land in different hue buckets ("tray", "lid", "board" do —
        // see part_tint's FNV-mod-12 bucketing).
        let mut tints = Vec::new();
        for id in ["tray", "lid", "board"] {
            let entity = build_preview_scene(&request(id, true)).expect("preview builds");
            let color = flat_color(&entity);
            assert_eq!(color, part_tint(id), "tint keyed by output id");
            let (lo, hi) = color[0..3]
                .iter()
                .fold((f32::INFINITY, 0.0f32), |(lo, hi), &c| {
                    (lo.min(c), hi.max(c))
                });
            assert!(hi < 0.9 && lo > 0.05, "not muted: {color:?}");
            assert!(
                entity
                    .stats
                    .detail
                    .iter()
                    .any(|line| line.starts_with("Color: part tint")),
                "{:?}",
                entity.stats.detail
            );
            tints.push(color.map(|c| (c * 255.0) as u8));
        }
        tints.sort();
        tints.dedup();
        assert_eq!(tints.len(), 3, "three parts, three tints: {tints:?}");
    }

    #[test]
    fn lightbox_scalar_field_analytics() {
        let data = build_lightbox_data(&step_field(1.0, 3.0)).expect("lightbox builds");
        assert!(!data.binary);
        assert_eq!((data.value_min, data.value_max), (1.0, 3.0));
        assert_eq!((data.width, data.height), (512, 512));
        assert_eq!(data.rgba.len(), 512 * 512 * 4);

        // The step sits exactly on the midline: mean 2, integral 2 over the
        // unit square, everything positive.
        assert!(analytic(&data, "mean").starts_with("2.000000"));
        assert!(analytic(&data, "integral").starts_with("2.000000"));
        assert_eq!(analytic(&data, "coverage (v > 0)"), "100.0%");
        assert!(analytic(&data, "range").contains("1.000000 .. 3.000000"));
        // No NaN row for a clean field.
        assert!(!data.analytics.iter().any(|(l, _)| l.contains("non-finite")));

        // Both step levels appear as distinct viridis colors.
        let first = &data.rgba[0..3];
        let last = &data.rgba[data.rgba.len() - 4..data.rgba.len() - 1];
        assert_ne!(first, last, "step levels should colormap differently");
    }

    #[test]
    fn lightbox_binary_mask_analytics() {
        let data = build_lightbox_data(&step_field(0.0, 1.0)).expect("lightbox builds");
        assert!(data.binary);
        assert!(analytic(&data, "occupied").starts_with("50.0%"));
        // Mask pixels are pure black/white.
        assert!(
            data.rgba
                .chunks_exact(4)
                .all(|px| px[0..3] == [0, 0, 0] || px[0..3] == [255, 255, 255])
        );
    }

    #[test]
    fn fea_preview_reports_colorable_fields_and_colors_nodes() {
        let request = fea_request(PreviewPlan::FeaMesh {
            deformed: false,
            exaggeration_tenths: 10,
            color_field: Some("node:temp".to_string()),
        });
        let entity = build_preview_scene(&request).expect("fea preview builds");

        assert_eq!(
            entity.stats.fea_fields,
            vec![
                "node:temp".to_string(),
                "node:displacement".to_string(),
                "element:stiffness_scale".to_string()
            ]
        );
        assert!(
            entity
                .stats
                .detail
                .iter()
                .any(|line| line.contains("colormap node:temp") && line.contains("viridis")),
            "colormap range missing from stats: {:?}",
            entity.stats.detail
        );

        // Node values 0..7 over the range: corner colors must vary.
        let mesh = &entity.scene.meshes[0].0;
        let first = mesh.vertices[0].color;
        assert!(
            mesh.vertices.iter().any(|v| v.color != first),
            "node colormap produced uniform vertex colors"
        );

        // Undeformed: bounds stay the unit cube.
        assert!((entity.bounds.max.2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fea_preview_element_colormap_and_exaggerated_deformation() {
        let request = fea_request(PreviewPlan::FeaMesh {
            deformed: true,
            exaggeration_tenths: 20,
            color_field: Some("element:stiffness_scale".to_string()),
        });
        let entity = build_preview_scene(&request).expect("fea preview builds");

        // One element: every face gets the same (mid-range) color.
        let mesh = &entity.scene.meshes[0].0;
        let first = mesh.vertices[0].color;
        assert!(
            mesh.vertices.iter().all(|v| v.color == first),
            "single-element colormap should be uniform"
        );

        // displacement (0, 0, 0.5) x2 exaggeration lifts the top to z = 2.
        assert!(
            (entity.bounds.max.2 - 2.0).abs() < 1e-6,
            "exaggerated deformation missing: max z = {}",
            entity.bounds.max.2
        );
    }

    #[test]
    fn bar2_preview_draws_struts_as_capsule_prisms() {
        use volumetric::fea::{FeaElementKind, FeaField, FeaMesh, encode_fea_mesh};
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0,
            ],
            connectivity: vec![0, 1, 1, 2],
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![0.1, 0.2],
            }],
        };
        let mut request = fea_request(PreviewPlan::FeaMesh {
            deformed: false,
            exaggeration_tenths: 10,
            color_field: Some("element:radius".to_string()),
        });
        request.data = Arc::new(encode_fea_mesh(&mesh));
        let entity = build_preview_scene(&request).expect("bar2 preview builds");

        // Each strut renders as a capped hexagonal prism: 12 side + 8 cap
        // triangles, 60 vertices.
        let scene_mesh = &entity.scene.meshes[0].0;
        assert_eq!(scene_mesh.vertices.len(), 2 * 60);
        assert_eq!(entity.stats.triangles, 2 * 20);

        // The element colormap tints each strut uniformly, and the two
        // struts (radius 0.1 vs 0.2) differently.
        let strut_color = |strut: usize| -> [f32; 4] {
            let base = strut * 60;
            let color = scene_mesh.vertices[base].color;
            assert!(
                scene_mesh.vertices[base..base + 60]
                    .iter()
                    .all(|v| v.color == color),
                "strut {strut} not uniformly colored"
            );
            color
        };
        assert_ne!(
            strut_color(0),
            strut_color(1),
            "element colormap should tint the two struts differently"
        );

        // Prism cross-sections honor the radius field: the first strut's
        // ring vertices sit 0.1 off its axis (y/z extent), the second 0.2.
        let max_lateral = |strut: usize| -> f32 {
            scene_mesh.vertices[strut * 60..(strut + 1) * 60]
                .iter()
                .map(|v| v.position[1].abs().max(v.position[2].abs()))
                .fold(0.0f32, f32::max)
        };
        assert!((max_lateral(0) - 0.1).abs() < 1e-5);

        assert!(
            entity
                .stats
                .detail
                .iter()
                .any(|line| line.contains("2 struts")),
            "strut count missing from stats: {:?}",
            entity.stats.detail
        );
        assert_eq!(
            entity.stats.fea_fields,
            vec!["element:radius".to_string()],
            "radius should be offered as a colormap field"
        );
    }

    #[test]
    fn fea_preview_missing_field_falls_back_to_plain() {
        let request = fea_request(PreviewPlan::FeaMesh {
            deformed: false,
            exaggeration_tenths: 10,
            color_field: Some("node:not_a_field".to_string()),
        });
        let entity = build_preview_scene(&request).expect("fea preview builds");
        let mesh = &entity.scene.meshes[0].0;
        assert!(
            mesh.vertices.iter().all(|v| v.color == [1.0; 4]),
            "missing field should draw untinted"
        );
        assert!(
            entity
                .stats
                .detail
                .iter()
                .any(|line| line.contains("not in this mesh")),
            "missing-field note absent: {:?}",
            entity.stats.detail
        );
    }

    fn request(id: &str, resolution: usize) -> PreviewRequest {
        let data = vec![1, 2, 3];
        PreviewRequest {
            asset_id: id.to_string(),
            source_hash: volumetric::content_fingerprint(&data),
            data: Arc::new(data),
            type_hint: None,
            precursor_ids: Vec::new(),
            plan: PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::PointCloud { resolution },
                color_channel: None,
                tint_uncolored: false,
            },
            wireframe: false,
            show_bounds: false,
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
            subspace: None,
        }
    }

    #[test]
    fn zoom_limits_scale_with_the_scene() {
        // A 0.1^3 part: the minimum orbit distance must land well under the
        // old fixed 0.1, or the camera can never approach the surface.
        let tiny = PreviewBounds {
            min: (0.0, 0.0, 0.0),
            max: (0.1, 0.1, 0.1),
        };
        let (min_radius, max_radius) = zoom_limits(Some(tiny));
        assert!(min_radius < 0.01, "min radius was {min_radius}");
        assert!(max_radius > 1.0);

        // Nothing framed (or degenerate bounds): the legacy fixed range.
        assert_eq!(zoom_limits(None), (0.1, 1000.0));
        let point = PreviewBounds {
            min: (1.0, 1.0, 1.0),
            max: (1.0, 1.0, 1.0),
        };
        assert_eq!(zoom_limits(Some(point)), (0.1, 1000.0));
    }

    #[test]
    fn bounds_box_is_twelve_edges_through_every_corner() {
        let bounds = PreviewBounds {
            min: (-1.0, -2.0, -3.0),
            max: (1.0, 2.0, 3.0),
        };
        let lines = bounds_box_lines(bounds);
        assert_eq!(lines.segments.len(), 12);

        // Each of the 8 corners terminates exactly 3 edges.
        let mut corner_uses = HashMap::new();
        for segment in &lines.segments {
            for p in [segment.start, segment.end] {
                assert!(
                    (p[0] == -1.0 || p[0] == 1.0)
                        && (p[1] == -2.0 || p[1] == 2.0)
                        && (p[2] == -3.0 || p[2] == 3.0),
                    "endpoint {p:?} is not a box corner"
                );
                *corner_uses
                    .entry((p[0].to_bits(), p[1].to_bits(), p[2].to_bits()))
                    .or_insert(0usize) += 1;
            }
        }
        assert_eq!(corner_uses.len(), 8);
        assert!(corner_uses.values().all(|&n| n == 3));
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
        req.data = Arc::new(sketch);
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

    #[test]
    fn fea_mesh_outputs_get_a_boundary_preview() {
        use volumetric::fea::{FeaElementKind, FeaField, FeaMesh, encode_fea_mesh};

        // A single unit hex at the origin.
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Hex8,
            node_positions: vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
            connectivity: (0..8).collect(),
            node_fields: vec![],
            element_fields: vec![],
        };

        let mut req = request("fea", 64);
        req.data = Arc::new(encode_fea_mesh(&mesh));
        req.type_hint = Some(AssetTypeHint::FeaMesh);
        let entity = build_preview_scene(&req).expect("fea preview");

        // 6 boundary quads = 12 flat-shaded triangles, 12 unique cube edges.
        assert_eq!(entity.stats.triangles, 12);
        let wireframe = entity.wireframe_lines.expect("fea preview has wireframe");
        assert_eq!(wireframe.segments.len(), 12);
        assert_eq!(entity.bounds.min, (0.0, 0.0, 0.0));
        assert_eq!(entity.bounds.max, (1.0, 1.0, 1.0));
        assert!(
            entity.stats.detail.iter().any(|l| l.contains("FEA mesh")),
            "stats should describe the mesh: {:?}",
            entity.stats.detail
        );

        // A solved mesh draws deformed: a uniform +1 z displacement shifts
        // the drawn bounds, and the detail lines report it.
        let mut solved = mesh.clone();
        solved.node_fields.push(FeaField {
            name: "displacement".to_string(),
            components: 3,
            data: (0..8).flat_map(|_| [0.0, 0.0, 1.0]).collect(),
        });
        let mut req = request("fea_solved", 64);
        req.data = Arc::new(encode_fea_mesh(&solved));
        req.type_hint = Some(AssetTypeHint::FeaMesh);
        let entity = build_preview_scene(&req).expect("solved fea preview");
        assert_eq!(entity.bounds.min, (0.0, 0.0, 1.0));
        assert_eq!(entity.bounds.max, (1.0, 1.0, 2.0));
        assert!(
            entity
                .stats
                .detail
                .iter()
                .any(|l| l.contains("deformed view")),
            "detail should mention deformation: {:?}",
            entity.stats.detail
        );

        // A triangle mesh — including a non-manifold single triangle —
        // renders double-sided with a full wireframe.
        use volumetric::trimesh::{TriMesh, encode_tri_mesh};
        let tri = TriMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            indices: vec![0, 1, 2],
            vertex_fields: vec![],
            face_fields: vec![],
        };
        let mut req = request("tri", 64);
        req.data = Arc::new(encode_tri_mesh(&tri));
        req.type_hint = Some(AssetTypeHint::TriMesh);
        let entity = build_preview_scene(&req).expect("tri mesh preview");
        assert_eq!(entity.stats.triangles, 2, "front + back face");
        assert_eq!(
            entity.wireframe_lines.as_ref().unwrap().segments.len(),
            3,
            "three unique edges"
        );
        assert_eq!(entity.bounds.min, (0.0, 0.0, 0.0));
        assert_eq!(entity.bounds.max, (1.0, 1.0, 0.0));

        // Garbage bytes fail with a decode error instead of a wasm error.
        let mut junk = request("junk", 64);
        junk.data = Arc::new(vec![0xff, 0x00, 0x13]);
        junk.type_hint = Some(AssetTypeHint::FeaMesh);
        let Err(err) = build_preview_scene(&junk) else {
            panic!("junk must fail");
        };
        assert!(err.contains("FEA mesh"), "unexpected error: {err}");
    }

    #[test]
    fn point1_outputs_get_a_marker_preview() {
        use volumetric::fea::{FeaElementKind, FeaField, FeaMesh, encode_fea_mesh};

        // Three points on the x axis, with a colormappable weight field.
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            connectivity: vec![0, 1, 2],
            node_fields: vec![FeaField {
                name: "weight".to_string(),
                components: 1,
                data: vec![0.0, 0.5, 1.0],
            }],
            element_fields: vec![],
        };
        let mut req = request("points", 64);
        req.data = Arc::new(encode_fea_mesh(&mesh));
        req.type_hint = Some(AssetTypeHint::FeaMesh);
        let entity = build_preview_scene(&req).expect("point cloud preview");

        // One octahedron (8 triangles) per point.
        assert_eq!(entity.stats.triangles, 24);
        // Bounds cover the raw points (markers are display-only).
        assert_eq!(entity.bounds.min, (0.0, 0.0, 0.0));
        assert_eq!(entity.bounds.max, (2.0, 0.0, 0.0));
        assert!(
            entity
                .stats
                .detail
                .iter()
                .any(|l| l.contains("Point cloud: 3 points")),
            "stats should describe the cloud: {:?}",
            entity.stats.detail
        );
        assert!(
            entity.stats.fea_fields.contains(&"node:weight".to_string()),
            "weight should be colormappable: {:?}",
            entity.stats.fea_fields
        );
    }

    /// Wireframe is display-only: toggling it flips the `visible` flag the
    /// viewport reads (it submits the prebuilt, GPU-resident edge lines)
    /// without scheduling any rebuild or bumping the build revision — a
    /// revision bump would needlessly re-upload the dense buffers.
    #[test]
    fn wireframe_toggle_flags_without_rebuilding() {
        let mut cache = PreviewCache::default();
        let requests = vec![request("a", 32)];
        let (_, jobs) = cache.sync(&requests, true, false);
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
        let visible = cache.visible(&requests);
        assert!(!visible[0].2, "wireframe off");
        let revision = visible[0].1;

        let mut wire_requests = requests.clone();
        wire_requests[0].wireframe = true;
        let (_, jobs) = cache.sync(&wire_requests, true, false);
        assert!(jobs.is_empty(), "toggling wireframe must not rebuild");
        let visible = cache.visible(&wire_requests);
        assert!(visible[0].2, "wireframe on");
        assert!(visible[0].3.wireframe_lines.is_some());
        assert_eq!(
            visible[0].1, revision,
            "display-only toggle keeps the GPU-resident revision"
        );
    }

    #[test]
    fn sync_meshes_each_output_then_exposes_all() {
        let mut cache = PreviewCache::default();
        let requests = vec![request("a", 8), request("b", 8)];

        let (status, jobs) = cache.sync(&requests, true, false);
        assert_eq!(jobs.len(), 2, "one build job per output");
        assert!(matches!(status, PreviewBuildStatus::Building { .. }));
        assert!(cache.visible(&requests).is_empty(), "nothing built yet");

        for job in jobs {
            accept_ok(&mut cache, job);
        }

        let (status, jobs) = cache.sync(&requests, true, false);
        assert!(jobs.is_empty(), "everything cached, no rebuild");
        assert!(matches!(status, PreviewBuildStatus::Ready { .. }));

        let visible = cache.visible(&requests);
        assert_eq!(visible.len(), 2, "both outputs render together");
        let ids: Vec<&str> = visible.iter().map(|(id, ..)| *id).collect();
        assert_eq!(ids, vec!["a", "b"]);
        assert_ne!(
            visible[0].1, visible[1].1,
            "each build gets its own revision"
        );

        let stats = cache.output_stats();
        assert_eq!(
            stats["a"].bounds,
            Some(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
            "stats carry the entity's bounds for the dimension readout"
        );
    }

    #[test]
    fn per_output_artifact_status_tracks_build_update_and_failure() {
        let mut cache = PreviewCache::default();
        let coarse = vec![request("a", 8)];
        assert_eq!(
            cache.artifact_statuses(&coarse)["a"],
            PreviewArtifactStatus::Missing
        );

        let (_, jobs) = cache.sync(&coarse, true, false);
        assert_eq!(
            cache.artifact_statuses(&coarse)["a"],
            PreviewArtifactStatus::Building
        );
        accept_ok(&mut cache, jobs.into_iter().next().unwrap());
        assert_eq!(
            cache.artifact_statuses(&coarse)["a"],
            PreviewArtifactStatus::Ready
        );

        let fine = vec![request("a", 32)];
        let (_, jobs) = cache.sync(&fine, true, false);
        assert_eq!(
            cache.artifact_statuses(&fine)["a"],
            PreviewArtifactStatus::Updating
        );
        let job = jobs.into_iter().next().unwrap();
        cache.accept(PreviewBuildResult {
            key: job.key,
            result: Err(PreviewBuildError::Failed("mesher stopped".to_string())),
        });
        assert_eq!(
            cache.artifact_statuses(&fine)["a"],
            PreviewArtifactStatus::Failed("mesher stopped".to_string())
        );
    }

    /// Changing an output's resolution requeues its build but keeps the last
    /// good mesh on screen — at its old revision, so the viewport keeps its
    /// GPU residency — until the new one arrives, which bumps the revision.
    #[test]
    fn resolution_change_keeps_stale_output_until_rebuilt() {
        let mut cache = PreviewCache::default();
        let coarse = vec![request("a", 8)];
        let (_, jobs) = cache.sync(&coarse, true, false);
        accept_ok(&mut cache, jobs.into_iter().next().unwrap());
        let coarse_revision = cache.visible(&coarse)[0].1;

        let fine = vec![request("a", 32)];
        let (status, jobs) = cache.sync(&fine, true, false);
        assert_eq!(jobs.len(), 1, "the finer resolution requeues a build");
        assert!(matches!(status, PreviewBuildStatus::Building { .. }));

        let visible = cache.visible(&fine);
        assert_eq!(visible.len(), 1, "stale mesh still shown");
        assert_eq!(visible[0].1, coarse_revision, "still the old upload");

        accept_ok(&mut cache, jobs.into_iter().next().unwrap());
        assert_ne!(
            cache.visible(&fine)[0].1,
            coarse_revision,
            "the rebuild bumps the revision so the viewport re-uploads"
        );
    }

    /// Artifact identity follows bytes, not an allocator address: remote
    /// results and fresh project runs often wrap identical bytes in a new Arc.
    #[test]
    fn reallocated_identical_asset_reuses_cached_artifact() {
        let mut cache = PreviewCache::default();
        let first = request("a", 8);
        let (_, jobs) = cache.sync(std::slice::from_ref(&first), true, false);
        accept_ok(&mut cache, jobs.into_iter().next().unwrap());

        let mut reallocated = first.clone();
        reallocated.data = Arc::new(first.data.as_ref().clone());
        assert_ne!(
            Arc::as_ptr(&first.data),
            Arc::as_ptr(&reallocated.data),
            "test requires a distinct allocation"
        );
        let (_, jobs) = cache.sync(&[reallocated], true, false);
        assert!(jobs.is_empty(), "same content must remain a cache hit");
    }

    /// Equal-length replacements cannot alias a previous artifact merely
    /// because an allocator happens to reuse its address.
    #[test]
    fn changed_same_length_asset_rebuilds() {
        let mut cache = PreviewCache::default();
        let first = request("a", 8);
        let (_, jobs) = cache.sync(std::slice::from_ref(&first), true, false);
        accept_ok(&mut cache, jobs.into_iter().next().unwrap());

        let mut changed = first.clone();
        changed.data = Arc::new(vec![9, 8, 7]);
        changed.source_hash = volumetric::content_fingerprint(&changed.data);
        let (_, jobs) = cache.sync(&[changed], true, false);
        assert_eq!(jobs.len(), 1, "changed content must rebuild");
    }

    /// A successful project rerun may reuse an output id for new bytes. The
    /// old scene must not be exposed to export/stats as if it represented the
    /// new artifact; its canonical ASN2 mesh remains available in the lower
    /// content-addressed mesh cache.
    #[test]
    fn materialized_content_change_invalidates_output_scene() {
        let mut cache = PreviewCache::default();
        let first = request("a", 8);
        let (_, jobs) = cache.sync(std::slice::from_ref(&first), true, false);
        accept_ok(&mut cache, jobs.into_iter().next().unwrap());
        assert!(cache.entity_scene("a").is_some());

        let changed_hash = volumetric::content_fingerprint(&[9, 8, 7]);
        cache.retain_assets(&std::collections::HashMap::from([("a", changed_hash)]));
        assert!(cache.entity_scene("a").is_none());
    }

    /// Viewport visibility does not own mesh artifacts: switching from a
    /// two-output view to one output and back reuses both completed builds.
    #[test]
    fn visibility_change_retains_completed_artifacts() {
        let mut cache = PreviewCache::default();
        // Clone shared requests so each output keeps one stable Arc (and thus a
        // stable cache key) across syncs, mirroring the app's `data_arc()`.
        let (a, b) = (request("a", 8), request("b", 8));
        let both = vec![a.clone(), b.clone()];
        let (_, jobs) = cache.sync(&both, true, false);
        for job in jobs {
            accept_ok(&mut cache, job);
        }

        let only_a = vec![a.clone()];
        let (_, jobs) = cache.sync(&only_a, true, false);
        assert!(jobs.is_empty());
        let visible = cache.visible(&only_a);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].0, "a");

        // "b" remained a valid project artifact while it was not visible.
        let (_, jobs) = cache.sync(&both, true, false);
        assert!(jobs.is_empty(), "showing b again reuses its mesh");
        assert_eq!(cache.visible(&both).len(), 2);
    }

    /// Project output ownership, unlike visibility, is an eviction boundary.
    #[test]
    fn removing_a_materialized_asset_evicts_its_artifact() {
        let mut cache = PreviewCache::default();
        let (a, b) = (request("a", 8), request("b", 8));
        let both = vec![a.clone(), b.clone()];
        let (_, jobs) = cache.sync(&both, true, false);
        for job in jobs {
            accept_ok(&mut cache, job);
        }

        cache.retain_assets(&std::collections::HashMap::from([("a", a.source_hash)]));
        let (_, jobs) = cache.sync(&both, true, false);
        assert_eq!(jobs.len(), 1, "removed b must be rebuilt if it returns");
        assert_eq!(jobs[0].key.asset_id, "b");
    }

    /// A build result for a superseded request (its output was re-requested with
    /// different settings) is discarded rather than shown.
    #[test]
    fn superseded_build_result_is_ignored() {
        let mut cache = PreviewCache::default();
        let (_, jobs) = cache.sync(&[request("a", 8)], true, false);
        let stale_job = jobs.into_iter().next().unwrap();

        // Re-request "a" at a new resolution before the first build returns.
        let fine = vec![request("a", 32)];
        let (_, jobs) = cache.sync(&fine, true, false);
        let fresh_job = jobs.into_iter().next().unwrap();

        // The stale build lands first and must be dropped.
        accept_ok(&mut cache, stale_job);
        assert!(cache.visible(&fine).is_empty(), "stale result discarded");

        accept_ok(&mut cache, fresh_job);
        assert!(!cache.visible(&fine).is_empty(), "fresh result accepted");
    }

    /// With auto-remesh off, out-of-date outputs are counted stale rather
    /// than dispatched; an explicit remesh (force) dispatches them anyway.
    #[test]
    fn auto_remesh_off_holds_builds_until_forced() {
        let mut cache = PreviewCache::default();
        let requests = vec![request("a", 8)];

        let (status, jobs) = cache.sync(&requests, false, false);
        assert!(jobs.is_empty(), "auto off: nothing dispatched");
        assert!(matches!(status, PreviewBuildStatus::Stale { .. }));

        let (status, jobs) = cache.sync(&requests, false, true);
        assert_eq!(jobs.len(), 1, "explicit remesh dispatches");
        assert!(matches!(status, PreviewBuildStatus::Building { .. }));
    }

    /// Superseding an in-flight build (same output, new settings) signals
    /// the old build's cancel flag so the serial worker abandons it instead
    /// of finishing a mesh nobody will look at.
    #[test]
    fn superseding_an_inflight_build_signals_its_cancel_flag() {
        let mut cache = PreviewCache::default();
        let (_, jobs) = cache.sync(&[request("a", 8)], true, false);
        let old = jobs.into_iter().next().unwrap();
        assert!(!old.cancel.load(Ordering::Relaxed));

        let (_, jobs) = cache.sync(&[request("a", 32)], true, false);
        assert_eq!(jobs.len(), 1, "the new settings dispatch a fresh build");
        assert!(
            old.cancel.load(Ordering::Relaxed),
            "superseded build signalled"
        );
    }

    /// Hiding an output does not abandon its expensive in-flight artifact.
    #[test]
    fn hiding_an_output_keeps_its_inflight_build() {
        let mut cache = PreviewCache::default();
        let (_, jobs) = cache.sync(&[request("a", 8)], true, false);
        let job = jobs.into_iter().next().unwrap();

        cache.sync(&[], true, false);
        assert!(!job.cancel.load(Ordering::Relaxed));
    }

    /// Removing the underlying materialized output does abandon its build.
    #[test]
    fn removing_a_materialized_asset_cancels_its_inflight_build() {
        let mut cache = PreviewCache::default();
        let (_, jobs) = cache.sync(&[request("a", 8)], true, false);
        let job = jobs.into_iter().next().unwrap();

        cache.retain_assets(&std::collections::HashMap::new());
        assert!(job.cancel.load(Ordering::Relaxed));
    }

    /// An explicit cancel suppresses re-dispatch of the identical build
    /// (otherwise the next frame would immediately requeue it), until the
    /// key changes or a remesh forces it. The cancelled result itself is
    /// dropped without a failure record.
    #[test]
    fn explicit_cancel_suppresses_identical_redispatch() {
        let mut cache = PreviewCache::default();
        let requests = vec![request("a", 8)];
        let (_, jobs) = cache.sync(&requests, true, false);
        let job = jobs.into_iter().next().unwrap();

        assert_eq!(cache.cancel_pending(), 1);
        assert!(job.cancel.load(Ordering::Relaxed));
        cache.accept(PreviewBuildResult {
            key: job.key,
            result: Err(PreviewBuildError::Cancelled),
        });

        let (status, jobs) = cache.sync(&requests, true, false);
        assert!(jobs.is_empty(), "same key stays suppressed");
        assert!(
            matches!(status, PreviewBuildStatus::Stale { .. }),
            "cancelled build reads as stale, not failed"
        );

        // A settings change (new key) lifts the suppression naturally...
        let (_, jobs) = cache.sync(&[request("a", 32)], true, false);
        assert_eq!(jobs.len(), 1);
    }

    /// ...and so does an explicit remesh at the unchanged key.
    #[test]
    fn explicit_remesh_clears_cancel_suppression() {
        let mut cache = PreviewCache::default();
        let requests = vec![request("a", 8)];
        let (_, jobs) = cache.sync(&requests, true, false);
        drop(jobs);
        cache.cancel_pending();

        let (_, jobs) = cache.sync(&requests, true, true);
        assert_eq!(jobs.len(), 1);
    }

    fn preview_job(id: &str, resolution: usize) -> BackgroundJob {
        let request = request(id, resolution);
        BackgroundJob::BuildPreview(PreviewBuildJob {
            key: PreviewSceneKey::from(&request),
            request,
            cancel: Arc::new(AtomicBool::new(false)),
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
            previews.push((preview.key.asset_id.clone(), preview.request.plan));
        }
        previews.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(previews.len(), 2, "one coalesced job per output");
        assert_eq!(
            previews[0],
            (
                "a".to_string(),
                PreviewPlan::Model3d {
                    mesh: PreviewMeshPlan::PointCloud { resolution: 32 },
                    color_channel: None,
                    tint_uncolored: false,
                }
            ),
            "newest job per output wins"
        );
        assert!(queue.is_empty());
    }
}
