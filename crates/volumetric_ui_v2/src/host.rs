//! Native winit shell for the v2 UI.
//!
//! Everything here is window-system plumbing: the event loop, surface, and
//! input mapping; a worker thread pumping the session's background jobs; and
//! the blocking `rfd` dialogs that fulfill the app's queued file actions. All
//! platform-neutral behavior (preview cache, viewport rendering, camera,
//! run/cancel bookkeeping) lives in [`crate::session`], which a future web
//! shell will drive instead.

use std::sync::{
    Arc,
    mpsc::{self, Receiver, Sender},
};
use std::thread;

use damascene_core::prelude::*;
use damascene_wgpu::Runner;
use damascene_winit_wgpu::host::input::{key_modifiers, map_key, pointer_button, winit_cursor};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::session::{
    BackgroundJob, BackgroundResult, JobQueue, Session, ViewportRenderParams, execute_job,
};
use crate::{FileAction, VIEWPORT_KEY, VolumetricUiV2};

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
        worker: BackgroundWorker::new(),
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
    /// Shared worker thread for project execution and preview meshing.
    worker: BackgroundWorker,
}

struct Gfx {
    session: Session,
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
            session: Session::new(&device, &queue, format),
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
                        if gfx.session.pointer_moved(
                            (lx, ly),
                            self.modifiers,
                            self.app.camera_control_scheme(),
                        ) {
                            gfx.window.request_redraw();
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
                        gfx.session.pointer_left();
                        for event in gfx.damascene.pointer_left() {
                            dispatch_event(&mut self.app, &gfx.damascene, event);
                        }
                        gfx.window.request_redraw();
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        let Some(button) = pointer_button(button) else {
                            return;
                        };
                        let Some((lx, ly)) = self.last_pointer else {
                            return;
                        };
                        match state {
                            ElementState::Pressed => {
                                gfx.session.pointer_down((lx, ly), button);
                                for event in
                                    gfx.damascene.pointer_down(Pointer::mouse(lx, ly, button))
                                {
                                    dispatch_event(&mut self.app, &gfx.damascene, event);
                                }
                            }
                            ElementState::Released => {
                                gfx.session.pointer_up(button);
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
                        if gfx.session.wheel(
                            (lx, ly),
                            camera_scroll_delta,
                            self.app.camera_control_scheme(),
                        ) {
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
            // The viewport render target follows the keyed widget rect, not the
            // window size; the session resizes it once layout resolves.
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

        gfx.session
            .pre_frame(&mut self.app, self.worker.drain_results());

        // Run a queued file action (native dialogs block the loop, like v1).
        // Before the session sync so an opened project's run starts this frame.
        if let Some(action) = self.app.take_file_action() {
            handle_file_action(&mut self.app, &gfx.session, action);
        }

        let scale_factor = gfx.window.scale_factor() as f32;
        let preview_requests = self.app.preview_requests();
        for job in gfx
            .session
            .sync(&mut self.app, &gfx.device, &preview_requests, scale_factor)
        {
            self.worker.send(job);
        }

        self.app.before_build();
        let theme = self.app.theme();
        let palette = theme.palette().clone();
        let cx = damascene_core::BuildCx::new(&theme);
        let mut tree = self.app.build(&cx);

        gfx.damascene.set_theme(theme);
        gfx.damascene.set_hotkeys(self.app.hotkeys());
        gfx.damascene.set_selection(self.app.selection());
        gfx.damascene.push_toasts(self.app.drain_toasts());

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
        let viewport_resized = gfx.session.render(ViewportRenderParams {
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
            || gfx.session.has_pending_preview()
            || gfx.session.run_in_flight()
        {
            gfx.window.request_redraw();
        }
    }
}

/// Runs a queued file action: native dialogs first, then the outcome is
/// routed back into the app. The blocking dialogs stall the event loop while
/// open — same trade-off as the v1 egui app.
fn handle_file_action(app: &mut VolumetricUiV2, session: &Session, action: FileAction) {
    match action {
        FileAction::OpenProject => {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Project", &["vproj"])
                .pick_file()
            {
                app.open_project_file(&path);
            }
        }
        FileAction::SaveProject => {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Project", &["vproj"])
                .save_file()
            {
                app.save_project_file(&path);
            }
        }
        FileAction::ExportStl(id) => {
            let triangles = session.preview_triangles(&id);
            if triangles.is_empty() {
                app.set_status(format!(
                    "no preview mesh for {id} — view it in a mesh render mode first"
                ));
                return;
            }
            let Some(path) = rfd::FileDialog::new()
                .add_filter("STL", &["stl"])
                .set_file_name(format!("{id}.stl"))
                .save_file()
            else {
                return;
            };
            match volumetric::stl::write_binary_stl(&path, &triangles, "volumetric") {
                Ok(()) => app.set_status(format!(
                    "exported {} triangles to {}",
                    triangles.len(),
                    path.display()
                )),
                Err(err) => app.set_status(format!("failed to export STL: {err}")),
            }
        }
        FileAction::ExportWasm(id) => {
            // Grab the bytes (stable Arc) up front so the borrow doesn't span
            // the blocking dialog.
            let Some(bytes) = app
                .runtime_assets()
                .iter()
                .find(|asset| asset.id() == id)
                .map(|asset| asset.data_arc())
            else {
                app.set_status(format!("no runtime asset {id} — run the project first"));
                return;
            };
            let Some(path) = rfd::FileDialog::new()
                .add_filter("WASM", &["wasm"])
                .set_file_name(format!("{id}.wasm"))
                .save_file()
            else {
                return;
            };
            match std::fs::write(&path, bytes.as_slice()) {
                Ok(()) => app.set_status(format!("exported {}", path.display())),
                Err(err) => app.set_status(format!("failed to export WASM: {err}")),
            }
        }
        FileAction::ImportWasm => {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("WASM", &["wasm"])
                .pick_file()
            else {
                return;
            };
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("model")
                .to_string();
            match std::fs::read(&path) {
                Ok(bytes) => app.import_model_wasm(&name, bytes),
                Err(err) => app.set_status(format!("failed to read WASM file: {err}")),
            }
        }
        FileAction::ImportStl => {
            import_blob_file(app, ("STL", &["stl"]), "stl_import_operator", "stl_import");
        }
        FileAction::ImportHeightmap => {
            import_blob_file(
                app,
                ("Images", &["png", "jpg", "jpeg", "bmp", "gif"]),
                "heightmap_extrude_operator",
                "heightmap",
            );
        }
    }
}

/// Shared pick-and-read path for Blob-input imports (STL, heightmap).
fn import_blob_file(
    app: &mut VolumetricUiV2,
    (filter_name, extensions): (&str, &[&str]),
    operator_name: &str,
    output_base: &str,
) {
    let Some(path) = rfd::FileDialog::new()
        .add_filter(filter_name, extensions)
        .pick_file()
    else {
        return;
    };
    match std::fs::read(&path) {
        Ok(bytes) => app.import_blob_asset(operator_name, output_base, bytes),
        Err(err) => app.set_status(format!("failed to read {}: {err}", path.display())),
    }
}

/// A single worker thread that serially executes the session's background
/// jobs through its coalescing [`JobQueue`]: the newest queued run wins,
/// preview jobs coalesce per output id, and the queue is re-filled from the
/// channel before each job so a fresh run preempts remaining previews.
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
                let mut queue = JobQueue::default();
                loop {
                    // Block for work only when nothing is already pending.
                    if queue.is_empty() {
                        match job_rx.recv() {
                            Ok(job) => queue.push(job),
                            Err(_) => break,
                        }
                    }
                    // Coalesce everything queued right now before picking the
                    // next job (a run pops ahead of any pending previews).
                    while let Ok(job) = job_rx.try_recv() {
                        queue.push(job);
                    }
                    if let Some(job) = queue.pop()
                        && result_tx.send(execute_job(job)).is_err()
                    {
                        break;
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
    use std::sync::atomic::AtomicBool;

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
