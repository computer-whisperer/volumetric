//! Native winit shell for the v2 UI.
//!
//! Everything here is window-system plumbing: the event loop, surface, and
//! input mapping; a worker thread pumping the session's background jobs; and
//! a file worker that runs the `rfd` dialogs and all disk I/O off the UI
//! thread (a blocked loop stops answering the Wayland ping and the
//! compositor kills the connection). All platform-neutral behavior (preview
//! cache, viewport rendering, camera, run/cancel bookkeeping) lives in
//! [`crate::session`], which a future web shell will drive instead.

use std::path::PathBuf;
use std::sync::{
    Arc,
    mpsc::{self, Receiver, Sender},
};
use std::thread;

use damascene_core::clipboard;
use damascene_core::prelude::*;
use damascene_core::widgets::text_input::{self, ClipboardKind};
use damascene_wgpu::Runner;
use damascene_winit_wgpu::host::input::{key_modifiers, map_key, pointer_button, winit_cursor};
use volumetric::Project;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use crate::remote::RemoteBackend;
use crate::session::{
    BackgroundJob, BackgroundResult, ExecutionBackend, JobQueue, LocalBackend, Session,
    ViewportRenderParams, execute_job_monitored,
};
use crate::settings::UiSettings;
use crate::{ExecutorChoice, FileAction, VIEWPORT_KEY, VolumetricUiV2};

pub fn run(
    title: &'static str,
    viewport: Rect,
    mut app: VolumetricUiV2,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load persisted preferences before the window exists so the initial
    // window size and executor choice honor them. `viewport` is the
    // caller's fallback for a first run with no recorded size.
    let settings_path = UiSettings::config_path();
    let last_settings = settings_path.as_deref().and_then(UiSettings::load);
    let mut viewport = viewport;
    if let Some(loaded) = &last_settings {
        loaded.apply(&mut app);
        if loaded.window_width > 0 && loaded.window_height > 0 {
            viewport = Rect::new(
                0.0,
                0.0,
                loaded.window_width as f32,
                loaded.window_height as f32,
            );
        }
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let mut host = Host {
        title,
        viewport,
        app,
        gfx: None,
        last_pointer: None,
        modifiers: KeyModifiers::default(),
        last_cursor: Cursor::Default,
        pending_resize: None,
        worker: BackgroundWorker::new(Arc::new(LocalBackend)),
        files: FileWorker::new(proxy),
        clipboard: arboard::Clipboard::new().ok(),
        last_selection: Selection::default(),
        last_primary: String::new(),
        settings_path,
        last_settings,
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
    /// Per-action worker threads for file dialogs and disk I/O.
    files: FileWorker,
    /// Best-effort native clipboard. Initialization can fail (headless,
    /// missing portal); text copy/cut/paste degrade to no-ops then.
    clipboard: Option<arboard::Clipboard>,
    /// Selection last mirrored to the Linux primary buffer, with the text
    /// that was synced — dedups the per-frame sync in `redraw`.
    last_selection: Selection,
    last_primary: String,
    /// Where preferences persist; `None` when the platform has no config
    /// dir. `last_settings` is the snapshot already on disk, so `redraw`
    /// only rewrites the file when something changed.
    settings_path: Option<PathBuf>,
    last_settings: Option<UiSettings>,
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
    /// The file worker pings the loop when an outcome is ready, so results
    /// land promptly even while the loop is idle in `Wait`.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        if let Some(gfx) = self.gfx.as_ref() {
            gfx.window.request_redraw();
        }
    }

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
            // Preview meshes of fine models (e.g. lattices) routinely pass
            // wgpu's 256 MiB default buffer ceiling; take whatever the
            // hardware allows. The renderer clamps to the granted limit and
            // reports overflow rather than panicking.
            required_limits: wgpu::Limits {
                max_buffer_size: adapter.limits().max_buffer_size,
                ..wgpu::Limits::default()
            },
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
                                    let event =
                                        attach_primary_selection_text(event, &mut self.clipboard);
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
                                match text_input::clipboard_request(&event) {
                                    Some(ClipboardKind::Copy) => {
                                        copy_current_selection(&gfx.damascene, &mut self.clipboard);
                                        dispatch_event(&mut self.app, &gfx.damascene, event);
                                    }
                                    Some(ClipboardKind::Cut) => {
                                        copy_current_selection(&gfx.damascene, &mut self.clipboard);
                                        let delete = clipboard::delete_selection_event(event);
                                        dispatch_event(&mut self.app, &gfx.damascene, delete);
                                    }
                                    Some(ClipboardKind::Paste) => {
                                        // No clipboard text: fall through with the raw
                                        // key event so the widget can ignore it.
                                        let event = match paste_text_from_clipboard(
                                            event.clone(),
                                            &mut self.clipboard,
                                        ) {
                                            Some(paste) => paste,
                                            None => event,
                                        };
                                        dispatch_event(&mut self.app, &gfx.damascene, event);
                                    }
                                    None => dispatch_event(&mut self.app, &gfx.damascene, event),
                                }
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

        // Swap the execution backend after results are routed and before
        // jobs dispatch, so this frame's re-queued work lands on the new
        // worker. The old worker thread exits after its current job; the app
        // cancelled that job's bookkeeping when it requested the swap, so
        // its (lost) results were already written off.
        if let Some(choice) = self.app.take_executor_request() {
            let backend: Arc<dyn ExecutionBackend> = match &choice {
                ExecutorChoice::Local => Arc::new(LocalBackend),
                ExecutorChoice::Remote(address) => Arc::new(RemoteBackend::new(address)),
            };
            self.worker = BackgroundWorker::new(backend);
        }

        // Apply finished file operations, then hand any newly queued action
        // to the file worker. Both before the session sync so an opened
        // project's run starts this frame.
        for outcome in self.files.drain() {
            apply_file_outcome(&mut self.app, outcome);
        }
        if let Some(action) = self.app.take_file_action() {
            if self.files.in_flight() {
                self.app
                    .set_status("a file operation is already in progress");
            } else if let Some(task) = gather_file_task(&mut self.app, &gfx.session, action) {
                self.files.spawn(task);
            }
        }

        // Persist preference changes: snapshot the persisted subset and
        // rewrite the file when it differs from what's on disk. Saves are a
        // few hundred bytes through tmp+rename, so writing on every changed
        // frame (address keystrokes, panel drags) is fine.
        if let Some(path) = self.settings_path.as_deref() {
            let snapshot = UiSettings::from_app(&self.app, gfx.config.width, gfx.config.height);
            if self.last_settings.as_ref() != Some(&snapshot) {
                snapshot.save(path);
                self.last_settings = Some(snapshot);
            }
        }

        let scale_factor = gfx.window.scale_factor() as f32;
        let preview_requests = self.app.preview_requests();
        for job in gfx.session.sync(
            &mut self.app,
            &gfx.device,
            &gfx.queue,
            &preview_requests,
            scale_factor,
        ) {
            self.worker.send(job);
        }

        self.app.before_build();
        let theme = self.app.theme();
        let palette = theme.palette().clone();
        let cx = damascene_core::BuildCx::new(&theme);
        let mut tree = self.app.build(&cx);

        gfx.damascene.set_theme(theme);
        gfx.damascene.set_hotkeys(self.app.hotkeys());
        let selection = self.app.selection();
        if selection != self.last_selection {
            self.last_selection = selection.clone();
            // Mirror the new selection's text into the Linux primary buffer
            // (middle-click paste). Reading it here, one frame event batch
            // late at worst, matches what `prepare` is about to draw.
            let text = gfx
                .damascene
                .selected_text_for(&self.last_selection)
                .filter(|s| !s.is_empty())
                .unwrap_or_default();
            if text != self.last_primary {
                if !text.is_empty() {
                    primary::set(&mut self.clipboard, &text);
                }
                self.last_primary = text;
            }
        }
        gfx.damascene.set_selection(selection);
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

/// Append the expected extension when a save dialog returns a bare name —
/// the Linux dialogs don't enforce their filter's extension. An explicit
/// different extension typed by the user is left alone.
fn ensure_extension(path: std::path::PathBuf, ext: &str) -> std::path::PathBuf {
    if path.extension().is_none() {
        path.with_extension(ext)
    } else {
        path
    }
}

/// A file operation with everything it needs gathered up front, so the
/// worker thread never touches app or session state.
enum FileTask {
    OpenProject,
    /// `path: None` asks with a save dialog; `Some` re-saves in place.
    SaveProject {
        project: Project,
        path: Option<PathBuf>,
    },
    ExportStl {
        id: String,
        triangles: Vec<volumetric::Triangle>,
    },
    ExportWasm {
        id: String,
        bytes: Arc<Vec<u8>>,
    },
    ImportWasm,
    ImportBlob {
        filter_name: &'static str,
        extensions: &'static [&'static str],
        operator_name: &'static str,
        output_base: &'static str,
    },
}

/// What a finished file task reports back; applied to the app on the UI
/// thread by [`apply_file_outcome`]. Every task resolves to exactly one
/// outcome (`Dismissed` when its dialog was cancelled) so the worker's
/// in-flight flag always clears.
enum FileOutcome {
    OpenedProject {
        path: PathBuf,
        result: Result<Project, String>,
    },
    SavedProject {
        path: PathBuf,
        result: Result<(), String>,
    },
    ImportedWasm {
        name: String,
        bytes: Vec<u8>,
    },
    ImportedBlob {
        operator_name: &'static str,
        output_base: &'static str,
        bytes: Vec<u8>,
    },
    /// A message for the status line (export results, read failures).
    Status(String),
    /// Dialog cancelled — nothing to apply.
    Dismissed,
}

/// Converts a queued [`FileAction`] into a self-contained [`FileTask`],
/// snapshotting whatever app/session state the task needs (preview
/// triangles, asset bytes, the project itself). Returns `None` when the
/// action can't proceed (with the reason on the status line).
fn gather_file_task(
    app: &mut VolumetricUiV2,
    session: &Session,
    action: FileAction,
) -> Option<FileTask> {
    match action {
        FileAction::OpenProject => Some(FileTask::OpenProject),
        FileAction::SaveProject => Some(FileTask::SaveProject {
            project: app.project().clone(),
            path: None,
        }),
        FileAction::SaveProjectTo(path) => Some(FileTask::SaveProject {
            project: app.project().clone(),
            path: Some(path),
        }),
        FileAction::ExportStl(id) => {
            let triangles = session.preview_triangles(&id);
            if triangles.is_empty() {
                app.set_status(format!(
                    "no preview mesh for {id} — view it in a mesh render mode first"
                ));
                return None;
            }
            Some(FileTask::ExportStl { id, triangles })
        }
        FileAction::ExportWasm(id) => {
            let Some(bytes) = app
                .runtime_assets()
                .iter()
                .find(|asset| asset.id() == id)
                .map(|asset| asset.data_arc())
            else {
                app.set_status(format!("no runtime asset {id} — run the project first"));
                return None;
            };
            Some(FileTask::ExportWasm { id, bytes })
        }
        FileAction::ImportWasm => Some(FileTask::ImportWasm),
        FileAction::ImportStl => Some(FileTask::ImportBlob {
            filter_name: "STL",
            extensions: &["stl"],
            operator_name: "stl_import_operator",
            output_base: "stl_import",
        }),
        FileAction::ImportImage => Some(FileTask::ImportBlob {
            filter_name: "Images",
            extensions: &["png", "jpg", "jpeg", "bmp", "gif"],
            operator_name: "image_model_operator",
            output_base: "image",
        }),
    }
}

/// Runs a file task to completion: dialog (if any), then disk I/O. Executes
/// on a file-worker thread — never on the UI thread, where a long dialog or
/// a slow disk would stall the event loop past the Wayland ping deadline.
/// (rfd's portal-backed Linux dialogs are fine off the main thread; macOS
/// would want dialogs on the main thread per rfd's docs.)
fn perform_file_task(task: FileTask) -> FileOutcome {
    match task {
        FileTask::OpenProject => {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("Project", &["vproj"])
                .pick_file()
            else {
                return FileOutcome::Dismissed;
            };
            let result = Project::load_from_file(&path).map_err(|err| err.to_string());
            FileOutcome::OpenedProject { path, result }
        }
        FileTask::SaveProject { project, path } => {
            let path = match path {
                Some(path) => path,
                None => {
                    let Some(picked) = rfd::FileDialog::new()
                        .add_filter("Project", &["vproj"])
                        .set_file_name("project.vproj")
                        .save_file()
                    else {
                        return FileOutcome::Dismissed;
                    };
                    ensure_extension(picked, "vproj")
                }
            };
            let result = project.save_to_file(&path).map_err(|err| err.to_string());
            FileOutcome::SavedProject { path, result }
        }
        FileTask::ExportStl { id, triangles } => {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("STL", &["stl"])
                .set_file_name(format!("{id}.stl"))
                .save_file()
            else {
                return FileOutcome::Dismissed;
            };
            let path = ensure_extension(path, "stl");
            match volumetric::stl::write_binary_stl(&path, &triangles, "volumetric") {
                Ok(()) => FileOutcome::Status(format!(
                    "exported {} triangles to {}",
                    triangles.len(),
                    path.display()
                )),
                Err(err) => FileOutcome::Status(format!("failed to export STL: {err}")),
            }
        }
        FileTask::ExportWasm { id, bytes } => {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("WASM", &["wasm"])
                .set_file_name(format!("{id}.wasm"))
                .save_file()
            else {
                return FileOutcome::Dismissed;
            };
            let path = ensure_extension(path, "wasm");
            match std::fs::write(&path, bytes.as_slice()) {
                Ok(()) => FileOutcome::Status(format!("exported {}", path.display())),
                Err(err) => FileOutcome::Status(format!("failed to export WASM: {err}")),
            }
        }
        FileTask::ImportWasm => {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("WASM", &["wasm"])
                .pick_file()
            else {
                return FileOutcome::Dismissed;
            };
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("model")
                .to_string();
            match std::fs::read(&path) {
                Ok(bytes) => FileOutcome::ImportedWasm { name, bytes },
                Err(err) => FileOutcome::Status(format!("failed to read WASM file: {err}")),
            }
        }
        FileTask::ImportBlob {
            filter_name,
            extensions,
            operator_name,
            output_base,
        } => {
            let Some(path) = rfd::FileDialog::new()
                .add_filter(filter_name, extensions)
                .pick_file()
            else {
                return FileOutcome::Dismissed;
            };
            match std::fs::read(&path) {
                Ok(bytes) => FileOutcome::ImportedBlob {
                    operator_name,
                    output_base,
                    bytes,
                },
                Err(err) => {
                    FileOutcome::Status(format!("failed to read {}: {err}", path.display()))
                }
            }
        }
    }
}

/// Routes a finished file operation back into the app (UI thread).
fn apply_file_outcome(app: &mut VolumetricUiV2, outcome: FileOutcome) {
    match outcome {
        FileOutcome::OpenedProject { path, result } => app.apply_opened_project(path, result),
        FileOutcome::SavedProject { path, result } => app.apply_saved_project(path, result),
        FileOutcome::ImportedWasm { name, bytes } => app.import_model_wasm(&name, bytes),
        FileOutcome::ImportedBlob {
            operator_name,
            output_base,
            bytes,
        } => app.import_blob_asset(operator_name, output_base, bytes),
        FileOutcome::Status(status) => app.set_status(status),
        FileOutcome::Dismissed => {}
    }
}

/// Runs file tasks on short-lived worker threads, one at a time. Each task
/// reports exactly one [`FileOutcome`] through the channel and pings the
/// event loop proxy so the result is applied promptly even from `Wait`.
struct FileWorker {
    outcome_tx: Sender<FileOutcome>,
    outcomes: Receiver<FileOutcome>,
    proxy: EventLoopProxy<()>,
    in_flight: bool,
}

impl FileWorker {
    fn new(proxy: EventLoopProxy<()>) -> Self {
        let (outcome_tx, outcomes) = mpsc::channel();
        Self {
            outcome_tx,
            outcomes,
            proxy,
            in_flight: false,
        }
    }

    fn in_flight(&self) -> bool {
        self.in_flight
    }

    fn spawn(&mut self, task: FileTask) {
        self.in_flight = true;
        let tx = self.outcome_tx.clone();
        let proxy = self.proxy.clone();
        thread::Builder::new()
            .name("volumetric-file-worker".to_string())
            .spawn(move || {
                let _ = tx.send(perform_file_task(task));
                let _ = proxy.send_event(());
            })
            .expect("spawn file worker");
    }

    fn drain(&mut self) -> Vec<FileOutcome> {
        let mut outcomes = Vec::new();
        while let Ok(outcome) = self.outcomes.try_recv() {
            outcomes.push(outcome);
        }
        // One task in flight at a time: any outcome means it finished.
        if !outcomes.is_empty() {
            self.in_flight = false;
        }
        outcomes
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
    /// Spawns the worker over an execution backend — [`LocalBackend`], or a
    /// [`RemoteBackend`] when remote build is enabled. Replacing the worker
    /// drops this sender; the thread exits after its current job.
    fn new(backend: Arc<dyn ExecutionBackend>) -> Self {
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
                        && result_tx
                            .send(execute_job_monitored(job, backend.as_ref(), &|progress| {
                                let _ = result_tx.send(progress);
                            }))
                            .is_err()
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

/// Copy the current damascene text selection to the system clipboard.
fn copy_current_selection(runner: &Runner, clipboard: &mut Option<arboard::Clipboard>) {
    let Some(text) = runner.selected_text() else {
        return;
    };
    if let Some(cb) = clipboard {
        let _ = cb.set_text(text);
    }
}

/// Rewrite a paste-request key event into a text-input event carrying the
/// system clipboard's text, or `None` when the clipboard is empty/non-text.
fn paste_text_from_clipboard(
    event: UiEvent,
    clipboard: &mut Option<arboard::Clipboard>,
) -> Option<UiEvent> {
    let text = clipboard.as_mut()?.get_text().ok()?;
    Some(clipboard::paste_text_event(event, text))
}

/// Attach the primary-selection text to middle-click events so text inputs
/// can paste it (Linux convention; a no-op elsewhere).
fn attach_primary_selection_text(
    mut event: UiEvent,
    clipboard: &mut Option<arboard::Clipboard>,
) -> UiEvent {
    if event.kind == UiEventKind::MiddleClick {
        event.text = primary::get(clipboard);
    }
    event
}

/// Linux primary selection buffer; select-to-copy, middle-click-to-paste.
mod primary {
    #[cfg(target_os = "linux")]
    pub fn set(clipboard: &mut Option<arboard::Clipboard>, text: &str) {
        use arboard::{LinuxClipboardKind, SetExtLinux};
        if let Some(cb) = clipboard {
            let _ = cb.set().clipboard(LinuxClipboardKind::Primary).text(text);
        }
    }

    #[cfg(target_os = "linux")]
    pub fn get(clipboard: &mut Option<arboard::Clipboard>) -> Option<String> {
        use arboard::{GetExtLinux, LinuxClipboardKind};
        let cb = clipboard.as_mut()?;
        cb.get().clipboard(LinuxClipboardKind::Primary).text().ok()
    }

    #[cfg(not(target_os = "linux"))]
    pub fn set(_clipboard: &mut Option<arboard::Clipboard>, _text: &str) {}

    #[cfg(not(target_os = "linux"))]
    pub fn get(_clipboard: &mut Option<arboard::Clipboard>) -> Option<String> {
        None
    }
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
        let worker = BackgroundWorker::new(Arc::new(LocalBackend));
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

    /// The dialog-less re-save task writes the file and reports the saved
    /// path — the whole known-path Save flow minus the UI thread.
    #[test]
    fn save_task_with_known_path_writes_off_thread() {
        let path = std::env::temp_dir().join(format!(
            "volumetric_ui_v2_file_worker_{}.vproj",
            std::process::id()
        ));
        let project = VolumetricUiV2::default().project().clone();

        let outcome = perform_file_task(FileTask::SaveProject {
            project,
            path: Some(path.clone()),
        });

        let FileOutcome::SavedProject {
            path: saved,
            result,
        } = outcome
        else {
            panic!("save task should report SavedProject");
        };
        assert_eq!(saved, path);
        result.expect("save succeeds");
        assert!(path.exists());

        let mut app = VolumetricUiV2::default();
        apply_file_outcome(
            &mut app,
            FileOutcome::SavedProject {
                path: saved,
                result: Ok(()),
            },
        );
        // A later Save re-saves in place through the worker (no dialog).
        app.on_event(
            UiEvent::synthetic_click(crate::SAVE_PROJECT_KEY),
            &EventCx::new(),
        );
        assert_eq!(
            app.take_file_action(),
            Some(FileAction::SaveProjectTo(path.clone()))
        );
        std::fs::remove_file(&path).ok();
    }
}
