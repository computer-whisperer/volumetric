use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, Sender},
};
use std::thread;

use damascene_core::prelude::*;
use damascene_wgpu::Runner;
use damascene_winit_wgpu::host::input::{key_modifiers, map_key, pointer_button, winit_cursor};
use glam::{Vec2, Vec3};
use volumetric_renderer as renderer;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::{
    PreviewBuildStatus, PreviewMeshPlan, PreviewRenderMode, PreviewRequest, RunState, VIEWPORT_KEY,
    ViewportCameraCommand, VolumetricUiV2,
};

pub fn run(
    title: &'static str,
    viewport: Rect,
    app: VolumetricUiV2,
) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut host = Host {
        title,
        viewport,
        app,
        gfx: None,
        last_pointer: None,
        modifiers: KeyModifiers::default(),
        last_cursor: Cursor::Default,
        pending_resize: None,
        last_viewport_rect: None,
        viewport_buttons: ViewportPointerButtons::default(),
        last_camera_pointer: None,
        worker: BackgroundWorker::new(),
        run_generation: 0,
        active_run: None,
    };
    event_loop.run_app(&mut host)?;
    Ok(())
}

struct Host {
    title: &'static str,
    viewport: Rect,
    app: VolumetricUiV2,
    gfx: Option<Gfx>,
    last_pointer: Option<(f32, f32)>,
    modifiers: KeyModifiers,
    last_cursor: Cursor,
    pending_resize: Option<PhysicalSize<u32>>,
    last_viewport_rect: Option<Rect>,
    viewport_buttons: ViewportPointerButtons,
    last_camera_pointer: Option<(f32, f32)>,
    /// Shared worker thread for project execution and preview meshing.
    worker: BackgroundWorker,
    /// Monotonic id for the most recently dispatched project run. Results whose
    /// generation is older than the active run are discarded (superseded or
    /// cancelled).
    run_generation: u64,
    /// The in-flight run's generation and its cooperative cancel flag.
    active_run: Option<(u64, Arc<AtomicBool>)>,
}

#[derive(Clone, Copy, Default)]
struct ViewportPointerButtons {
    left: bool,
    middle: bool,
    right: bool,
}

impl ViewportPointerButtons {
    fn set(&mut self, button: MouseButton, down: bool) {
        match button {
            MouseButton::Left => self.left = down,
            MouseButton::Middle => self.middle = down,
            MouseButton::Right => self.right = down,
            _ => {}
        }
    }

    fn any(self) -> bool {
        self.left || self.middle || self.right
    }
}

struct Gfx {
    viewport_renderer: ViewportRenderer,
    damascene: Runner,
    surface: wgpu::Surface<'static>,
    queue: wgpu::Queue,
    device: wgpu::Device,
    window: Arc<Window>,
    config: wgpu::SurfaceConfiguration,
}

impl ApplicationHandler for Host {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title(self.title)
            .with_inner_size(PhysicalSize::new(
                self.viewport.w as u32,
                self.viewport.h as u32,
            ));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no compatible adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("volumetric_ui_v2::device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device");

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Fifo)
        {
            wgpu::PresentMode::Fifo
        } else {
            surface_caps.present_modes[0]
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        let mut damascene = Runner::new(&device, &queue, format);
        damascene.set_theme(self.app.theme());
        damascene.set_surface_size(config.width, config.height);
        for shader in self.app.shaders() {
            damascene.register_shader_with(
                &device,
                shader.name,
                shader.wgsl,
                shader.samples_backdrop,
                shader.samples_time,
            );
        }

        self.gfx = Some(Gfx {
            viewport_renderer: ViewportRenderer::new(&device, &queue, format),
            damascene,
            surface,
            queue,
            device,
            window,
            config,
        });
        self.gfx.as_ref().unwrap().window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.gfx.take();
                event_loop.exit();
            }
            event => {
                let Some(gfx) = self.gfx.as_mut() else {
                    return;
                };
                let scale = gfx.window.scale_factor() as f32;

                match event {
                    WindowEvent::Resized(size) => {
                        let w = size.width.max(1);
                        let h = size.height.max(1);
                        let same_as_current = self.pending_resize.is_none()
                            && w == gfx.config.width
                            && h == gfx.config.height;
                        if same_as_current {
                            return;
                        }
                        self.pending_resize = Some(PhysicalSize::new(w, h));
                        gfx.window.request_redraw();
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let lx = position.x as f32 / scale;
                        let ly = position.y as f32 / scale;
                        self.last_pointer = Some((lx, ly));
                        if let Some((last_x, last_y)) = self.last_camera_pointer
                            && self.viewport_buttons.any()
                        {
                            let input = renderer::CameraInputState {
                                left_down: self.viewport_buttons.left,
                                middle_down: self.viewport_buttons.middle,
                                right_down: self.viewport_buttons.right,
                                shift_down: self.modifiers.shift,
                                ctrl_down: self.modifiers.ctrl,
                                alt_down: self.modifiers.alt,
                                mouse_delta: Vec2::new(lx - last_x, ly - last_y),
                                scroll_delta: 0.0,
                            };
                            if gfx
                                .viewport_renderer
                                .apply_camera_input(&input, self.app.camera_control_scheme())
                            {
                                gfx.window.request_redraw();
                            }
                            self.last_camera_pointer = Some((lx, ly));
                        }
                        let moved = gfx.damascene.pointer_moved(Pointer::moving(lx, ly));
                        for event in moved.events {
                            dispatch_event(&mut self.app, &gfx.damascene, event);
                        }
                        if moved.needs_redraw {
                            gfx.window.request_redraw();
                        }
                    }
                    WindowEvent::CursorLeft { .. } => {
                        self.last_pointer = None;
                        self.last_camera_pointer = None;
                        self.viewport_buttons = ViewportPointerButtons::default();
                        for event in gfx.damascene.pointer_left() {
                            dispatch_event(&mut self.app, &gfx.damascene, event);
                        }
                        gfx.window.request_redraw();
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        self.viewport_buttons
                            .set(button, state == ElementState::Pressed);
                        let Some(button) = pointer_button(button) else {
                            return;
                        };
                        let Some((lx, ly)) = self.last_pointer else {
                            return;
                        };
                        if state == ElementState::Pressed
                            && pointer_in_rect(self.last_viewport_rect, lx, ly)
                        {
                            self.last_camera_pointer = Some((lx, ly));
                        } else if state == ElementState::Released && !self.viewport_buttons.any() {
                            self.last_camera_pointer = None;
                        }
                        match state {
                            ElementState::Pressed => {
                                for event in
                                    gfx.damascene.pointer_down(Pointer::mouse(lx, ly, button))
                                {
                                    dispatch_event(&mut self.app, &gfx.damascene, event);
                                }
                            }
                            ElementState::Released => {
                                for event in
                                    gfx.damascene.pointer_up(Pointer::mouse(lx, ly, button))
                                {
                                    dispatch_event(&mut self.app, &gfx.damascene, event);
                                }
                            }
                        }
                        gfx.window.request_redraw();
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let Some((lx, ly)) = self.last_pointer else {
                            return;
                        };
                        let (dy, camera_scroll_delta) = match delta {
                            MouseScrollDelta::LineDelta(_, y) => (-y * 50.0, y),
                            MouseScrollDelta::PixelDelta(p) => {
                                let logical_y = p.y as f32 / scale;
                                (-logical_y, logical_y / 50.0)
                            }
                        };
                        if pointer_in_rect(self.last_viewport_rect, lx, ly) {
                            gfx.viewport_renderer
                                .zoom_camera(camera_scroll_delta, self.app.camera_control_scheme());
                            gfx.window.request_redraw();
                        }
                        if gfx.damascene.pointer_wheel(lx, ly, dy) {
                            gfx.window.request_redraw();
                        }
                    }
                    WindowEvent::ModifiersChanged(modifiers) => {
                        self.modifiers = key_modifiers(modifiers.state());
                        gfx.damascene.set_modifiers(self.modifiers);
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            key_event @ winit::event::KeyEvent {
                                state: ElementState::Pressed,
                                ..
                            },
                        is_synthetic: false,
                        ..
                    } => {
                        if let Some(key) = map_key(&key_event.logical_key) {
                            for event in
                                gfx.damascene
                                    .key_down(key, self.modifiers, key_event.repeat)
                            {
                                dispatch_event(&mut self.app, &gfx.damascene, event);
                            }
                        }
                        if let Some(text) = &key_event.text
                            && let Some(event) = gfx.damascene.text_input(text.to_string())
                        {
                            dispatch_event(&mut self.app, &gfx.damascene, event);
                        }
                        gfx.window.request_redraw();
                    }
                    WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                        if let Some(event) = gfx.damascene.text_input(text) {
                            dispatch_event(&mut self.app, &gfx.damascene, event);
                        }
                        gfx.window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => self.redraw(),
                    _ => {}
                }
            }
        }
    }
}

impl Host {
    fn redraw(&mut self) {
        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };

        if let Some(size) = self.pending_resize.take() {
            gfx.config.width = size.width;
            gfx.config.height = size.height;
            gfx.surface.configure(&gfx.device, &gfx.config);
            gfx.damascene
                .set_surface_size(gfx.config.width, gfx.config.height);
            gfx.viewport_renderer
                .resize(&gfx.device, gfx.config.width, gfx.config.height);
        }

        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                gfx.surface.configure(&gfx.device, &gfx.config);
                return;
            }
            other => {
                eprintln!("surface unavailable: {other:?}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Drain completed background work and route it to the app / viewport.
        for result in self.worker.drain_results() {
            match result {
                BackgroundResult::ProjectComplete {
                    generation,
                    result,
                    elapsed_ms,
                } => {
                    if self.active_run.as_ref().map(|(g, _)| *g) == Some(generation) {
                        self.active_run = None;
                        self.app.apply_run_result(result, elapsed_ms);
                    }
                    // Otherwise the run was superseded or cancelled; discard it.
                }
                BackgroundResult::PreviewComplete(preview) => {
                    gfx.viewport_renderer.accept_preview_result(preview);
                }
            }
        }

        // Honor a cancel request for the in-flight run: signal the worker and
        // bump the generation so its (abandoned) result is ignored on arrival.
        if self.app.take_cancel_request() {
            if let Some((_, cancel)) = self.active_run.take() {
                cancel.store(true, Ordering::Relaxed);
            }
            self.run_generation += 1;
            self.app.on_run_cancelled();
        }

        // Dispatch a queued run to the worker.
        if self.app.take_pending_run() {
            self.run_generation += 1;
            let generation = self.run_generation;
            let cancel = Arc::new(AtomicBool::new(false));
            self.active_run = Some((generation, cancel.clone()));
            self.app.set_run_state(RunState::Running);
            self.worker.send(BackgroundJob::RunProject {
                generation,
                project: self.app.project().clone(),
                cancel,
            });
        }

        let preview_requests = self.app.preview_requests();
        let (preview_status, preview_jobs) = gfx
            .viewport_renderer
            .sync_preview_requests(&preview_requests);
        self.app.set_preview_build_status(preview_status);
        for job in preview_jobs {
            self.worker.send(BackgroundJob::BuildPreview(job));
        }
        if let Some(command) = self.app.take_camera_command() {
            gfx.viewport_renderer.apply_camera_command(command);
        }
        if let Some(rect) = self.last_viewport_rect {
            gfx.viewport_renderer.ensure_target_for_rect(
                &gfx.device,
                rect,
                gfx.window.scale_factor() as f32,
            );
        }
        self.app
            .set_viewport_texture(gfx.viewport_renderer.app_texture());
        self.app.before_build();
        let theme = self.app.theme();
        let palette = theme.palette().clone();
        let cx = damascene_core::BuildCx::new(&theme);
        let mut tree = self.app.build(&cx);

        gfx.damascene.set_theme(theme);
        gfx.damascene.set_hotkeys(self.app.hotkeys());
        gfx.damascene.set_selection(self.app.selection());
        gfx.damascene.push_toasts(self.app.drain_toasts());

        let scale_factor = gfx.window.scale_factor() as f32;
        let viewport = Rect::new(
            0.0,
            0.0,
            gfx.config.width as f32 / scale_factor,
            gfx.config.height as f32 / scale_factor,
        );
        let prepare =
            gfx.damascene
                .prepare(&gfx.device, &gfx.queue, &mut tree, viewport, scale_factor);

        let cursor = gfx.damascene.ui_state().cursor(&tree);
        if cursor != self.last_cursor {
            gfx.window.set_cursor(winit_cursor(cursor));
            self.last_cursor = cursor;
        }

        let mut encoder = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("volumetric_ui_v2::encoder"),
            });

        let viewport_rect = gfx.damascene.rect_of_key(VIEWPORT_KEY);
        self.last_viewport_rect = viewport_rect;

        let viewport_resized = gfx.viewport_renderer.render(ViewportRenderParams {
            device: &gfx.device,
            queue: &gfx.queue,
            encoder: &mut encoder,
            logical_rect: viewport_rect,
            scale_factor,
            clear_color: bg_color(&palette),
            preview_requests,
        });
        gfx.damascene.render(
            &gfx.device,
            &mut encoder,
            &frame.texture,
            &view,
            None,
            wgpu::LoadOp::Clear(bg_color(&palette)),
        );

        gfx.queue.submit(Some(encoder.finish()));
        frame.present();

        if prepare.needs_redraw
            || viewport_resized
            || gfx.viewport_renderer.has_pending_preview()
            || self.active_run.is_some()
        {
            gfx.window.request_redraw();
        }
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

struct ViewportRenderParams<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    encoder: &'a mut wgpu::CommandEncoder,
    logical_rect: Option<Rect>,
    scale_factor: f32,
    clear_color: wgpu::Color,
    preview_requests: Vec<PreviewRequest>,
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

    fn resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // The renderer's real size is the keyed viewport rect, not the full window.
        // It is resized during `render` once Damascene layout has resolved that rect.
    }

    fn app_texture(&self) -> AppTexture {
        self.target.app_texture.clone()
    }

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

    fn apply_camera_command(&mut self, command: ViewportCameraCommand) {
        match command {
            ViewportCameraCommand::FramePreview => self.frame_preview(),
        }
    }

    /// The Frame command re-centers the camera on the current scene. The actual
    /// framing happens in `resolve_scene`, which owns the union bounds; here we
    /// only raise the request so it fires even if the scene isn't cached yet.
    fn frame_preview(&mut self) {
        self.pending_frame_preview = true;
    }

    fn has_pending_preview(&self) -> bool {
        self.preview_cache.has_pending()
    }

    /// Reconciles the desired output set with the mesh cache and returns the
    /// aggregate build status plus any per-output jobs to enqueue on the shared
    /// worker. (The host owns the worker so project runs and preview meshing
    /// share one thread and one result channel.)
    fn sync_preview_requests(
        &mut self,
        requests: &[PreviewRequest],
    ) -> (PreviewBuildStatus, Vec<PreviewBuildJob>) {
        self.preview_cache.sync(requests)
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

/// A single built output: its meshed geometry and world-space bounds.
#[derive(Clone)]
struct PreviewEntity {
    scene: renderer::SceneData,
    bounds: PreviewBounds,
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
    /// Memoized composite, rebuilt only when the contributing keys change.
    composite: Option<(Vec<PreviewSceneKey>, renderer::SceneData, PreviewBounds)>,
}

impl PreviewCache {
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
                keys.push(key.clone());
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

struct PreviewBuildJob {
    key: PreviewSceneKey,
    request: PreviewRequest,
}

struct PreviewBuildResult {
    key: PreviewSceneKey,
    result: Result<PreviewEntity, String>,
}

/// A unit of background work for [`BackgroundWorker`].
enum BackgroundJob {
    RunProject {
        generation: u64,
        project: volumetric::Project,
        cancel: Arc<AtomicBool>,
    },
    BuildPreview(PreviewBuildJob),
}

/// A completed unit of background work.
enum BackgroundResult {
    ProjectComplete {
        generation: u64,
        result: Result<Vec<volumetric::LoadedAsset>, String>,
        elapsed_ms: u128,
    },
    PreviewComplete(PreviewBuildResult),
}

/// A single worker thread that serially executes project runs and preview mesh
/// builds. The newest queued run is coalesced (a burst of edits collapses to one
/// rebuild); preview jobs are coalesced per output id, so several outputs can be
/// meshed while re-requesting one output supersedes its stale job. Runs are
/// always drained before previews so a following preview sees fresh assets.
struct BackgroundWorker {
    jobs: Sender<BackgroundJob>,
    results: Receiver<BackgroundResult>,
}

impl BackgroundWorker {
    fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<BackgroundJob>();
        let (result_tx, result_rx) = mpsc::channel::<BackgroundResult>();

        thread::Builder::new()
            .name("volumetric-background-worker".to_string())
            .spawn(move || {
                let mut pending_run: Option<(u64, volumetric::Project, Arc<AtomicBool>)> = None;
                let mut pending_previews: HashMap<String, PreviewBuildJob> = HashMap::new();

                loop {
                    // Block for work only when nothing is already pending.
                    if pending_run.is_none() && pending_previews.is_empty() {
                        match job_rx.recv() {
                            Ok(job) => stash_job(&mut pending_run, &mut pending_previews, job),
                            Err(_) => break,
                        }
                    }
                    // Coalesce everything queued right now (newest run, newest job per output).
                    while let Ok(job) = job_rx.try_recv() {
                        stash_job(&mut pending_run, &mut pending_previews, job);
                    }

                    // Run the project first so a following preview sees fresh assets;
                    // loop back afterwards so a newly queued run preempts any previews.
                    if let Some((generation, project, cancel)) = pending_run.take() {
                        let start = std::time::Instant::now();
                        let result = project
                            .run_cancellable(&mut volumetric::Environment::new(), &cancel)
                            .map_err(|err| err.to_string());
                        let elapsed_ms = start.elapsed().as_millis();
                        if result_tx
                            .send(BackgroundResult::ProjectComplete {
                                generation,
                                result,
                                elapsed_ms,
                            })
                            .is_err()
                        {
                            break;
                        }
                        continue;
                    }

                    // Mesh one output per iteration, re-checking the job queue between
                    // outputs so a fresh run jumps ahead of the remaining previews.
                    if let Some(id) = pending_previews.keys().next().cloned() {
                        let job = pending_previews.remove(&id).expect("key just observed");
                        let result = build_preview_scene(&job.request);
                        if result_tx
                            .send(BackgroundResult::PreviewComplete(PreviewBuildResult {
                                key: job.key,
                                result,
                            }))
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            })
            .expect("spawn background worker");

        Self {
            jobs: job_tx,
            results: result_rx,
        }
    }

    fn send(&self, job: BackgroundJob) {
        let _ = self.jobs.send(job);
    }

    fn drain_results(&self) -> Vec<BackgroundResult> {
        let mut results = Vec::new();
        while let Ok(result) = self.results.try_recv() {
            results.push(result);
        }
        results
    }
}

fn stash_job(
    run: &mut Option<(u64, volumetric::Project, Arc<AtomicBool>)>,
    previews: &mut HashMap<String, PreviewBuildJob>,
    job: BackgroundJob,
) {
    match job {
        BackgroundJob::RunProject {
            generation,
            project,
            cancel,
        } => *run = Some((generation, project, cancel)),
        BackgroundJob::BuildPreview(preview_job) => {
            // Newest job per output wins.
            previews.insert(preview_job.key.asset_id.clone(), preview_job);
        }
    }
}

fn build_preview_scene(request: &PreviewRequest) -> Result<PreviewEntity, String> {
    let (scene, bounds_min, bounds_max) = match &request.mesh_plan {
        PreviewMeshPlan::PointCloud { resolution } => {
            let (points, bounds_min, bounds_max) =
                volumetric::sample_model_from_bytes(request.wasm_bytes.as_slice(), *resolution)
                    .map_err(format_error_chain)?;
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
            (scene, bounds_min, bounds_max)
        }
        PreviewMeshPlan::MarchingCubes { resolution } => {
            let (triangles, bounds_min, bounds_max) =
                volumetric::generate_marching_cubes_mesh_from_bytes(
                    request.wasm_bytes.as_slice(),
                    *resolution,
                )
                .map_err(format_error_chain)?;
            let mut scene = renderer::SceneData::new();
            scene.add_mesh(
                renderer::MeshData {
                    vertices: triangles_to_mesh_vertices(&triangles),
                    indices: None,
                },
                glam::Mat4::IDENTITY,
                renderer::MaterialId(0),
            );
            (scene, bounds_min, bounds_max)
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
            let vertices = mesh
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
            let mut scene = renderer::SceneData::new();
            scene.add_mesh(
                renderer::MeshData {
                    vertices,
                    indices: Some(mesh.indices),
                },
                glam::Mat4::IDENTITY,
                renderer::MaterialId(0),
            );
            (scene, mesh.bounds_min, mesh.bounds_max)
        }
    };

    Ok(PreviewEntity {
        scene,
        bounds: PreviewBounds {
            min: bounds_min,
            max: bounds_max,
        },
    })
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
        if !request.show_grid {
            settings.grid.planes = renderer::GridPlanes::NONE;
        }
    }

    settings
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

fn pointer_in_rect(rect: Option<Rect>, x: f32, y: f32) -> bool {
    rect.is_some_and(|rect| {
        x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h
    })
}

/// Dispatch a UI event to the app, attaching the runner's `UiState` so
/// geometry accessors on `EventCx` can answer.
fn dispatch_event(app: &mut VolumetricUiV2, runner: &Runner, event: UiEvent) {
    let cx = EventCx::new().with_ui_state(runner.ui_state());
    app.on_event(event, &cx);
}

fn bg_color(palette: &damascene_core::Palette) -> wgpu::Color {
    let c = palette.background;
    wgpu::Color {
        r: srgb_to_linear(c.r as f64 / 255.0),
        g: srgb_to_linear(c.g as f64 / 255.0),
        b: srgb_to_linear(c.b as f64 / 255.0),
        a: c.a as f64 / 255.0,
    }
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
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
            show_grid: true,
            ssao: true,
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
        }
    }

    fn accept_ok(cache: &mut PreviewCache, job: PreviewBuildJob) {
        cache.accept(PreviewBuildResult {
            key: job.key,
            result: Ok(entity()),
        });
    }

    /// The milestone: two requested outputs each get a build job, and once both
    /// land they composite into a single multi-entity scene.
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

    /// End-to-end check that the shared worker executes a project off-thread and
    /// reports the result back through the channel, tagged with its generation.
    #[test]
    fn background_worker_runs_project_off_thread() {
        let worker = BackgroundWorker::new();
        let project = VolumetricUiV2::default().project().clone();
        let cancel = Arc::new(AtomicBool::new(false));
        worker.send(BackgroundJob::RunProject {
            generation: 7,
            project,
            cancel,
        });

        let mut completion = None;
        for _ in 0..300 {
            for result in worker.drain_results() {
                if let BackgroundResult::ProjectComplete {
                    generation, result, ..
                } = result
                {
                    completion = Some((generation, result));
                }
            }
            if completion.is_some() {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }

        let (generation, result) = completion.expect("worker reported a project result");
        assert_eq!(generation, 7);
        let assets = result.expect("default project runs cleanly");
        assert_eq!(assets.len(), 1);
    }
}
