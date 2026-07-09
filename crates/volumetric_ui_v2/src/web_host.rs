//! Browser (wasm32) shell for the v2 UI.
//!
//! The platform seams mirror `host.rs` exactly — same [`Session`] frame
//! protocol, same [`crate::FileAction`] fulfillment, same input mapping —
//! with the native pieces swapped for their browser equivalents:
//!
//! - window → an existing `#volumetric_canvas` element, driven through
//!   winit's web backend (`EventLoop::spawn_app`, canvas resize tracked by a
//!   `ResizeObserver` because winit-web does not reliably translate CSS
//!   resizes into `Resized` events);
//! - `pollster::block_on` GPU setup → an async task
//!   (`wasm_bindgen_futures::spawn_local`) probing WebGPU with a WebGL2
//!   fallback, after damascene-web's host;
//! - the background worker thread → an inline executor: one job per
//!   `setTimeout(0)` turn on the main thread. Long jobs freeze the tab for
//!   their duration (same trade-off the v1 egui web build ships); the
//!   deferral at least lets the frame that queued the job paint its
//!   "running" status first. Cancellation takes effect between jobs only —
//!   the engine's epoch interruption is a wasmtime facility.
//! - blocking `rfd` dialogs + `std::fs` → `rfd::AsyncFileDialog` picks
//!   (bytes in) and Blob/anchor downloads (bytes out). Project save always
//!   downloads; the browser owns the destination.
//!
//! Remote builds are unavailable here: the daemon client is blocking HTTP
//! over TCP. A fetch-based client would slot into the same
//! `ExecutionBackend` seam later.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use damascene_core::prelude::*;
use damascene_wgpu::{Runner, RunnerCaps};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::Closure;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::{Window, WindowId};

use crate::session::{
    BackgroundResult, ExecutionBackend, JobQueue, LocalBackend, Session, ViewportRenderParams,
    execute_job_monitored,
};
use crate::{ExecutorChoice, FileAction, VIEWPORT_KEY, VolumetricUiV2};

/// The canvas element the shell binds to; declared by `index.html`.
const CANVAS_ID: &str = "volumetric_canvas";

/// Start the browser shell. Returns once winit's web event loop is spawned
/// (the browser drives frames from then on) — call from the crate's
/// `#[wasm_bindgen(start)]` entry.
pub fn run(title: &str, viewport: Rect, app: VolumetricUiV2) {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    if let Some(document) = web_sys::window().and_then(|w| w.document()) {
        document.set_title(title);
    }

    let event_loop = EventLoop::new().expect("EventLoop::new");
    let host = WebHost::new(viewport, app);
    event_loop.spawn_app(host);
}

struct WebHost {
    viewport: Rect,
    app: VolumetricUiV2,
    /// `RefCell<Option<..>>` because the async GPU setup fills it after
    /// `resumed` returns; `Rc` so the setup task can reach the slot.
    gfx: Rc<RefCell<Option<Gfx>>>,
    /// Set once `resumed` has bound the canvas, so a second `resumed` (or
    /// one racing the still-async GPU setup) doesn't re-install anything.
    started: bool,
    last_pointer: Option<(f32, f32)>,
    modifiers: KeyModifiers,
    last_cursor: Cursor,
    /// Physical size measured by the ResizeObserver, applied at the top of
    /// the next redraw (mirrors the native host's `pending_resize`).
    pending_resize: Rc<Cell<Option<(u32, u32)>>>,
    /// Keep the observer's JS closure alive for the page's lifetime. This
    /// is a full-page app — there is no teardown path.
    _resize_closure: Option<Closure<dyn FnMut()>>,
    _resize_observer: Option<web_sys::ResizeObserver>,
    /// Inline executor state: the session's coalescing queue plus the
    /// results the pump produced since the last frame.
    jobs: Rc<RefCell<JobQueue>>,
    job_results: Rc<RefCell<Vec<BackgroundResult>>>,
    /// One pump `setTimeout` in flight at a time.
    pump_scheduled: Rc<Cell<bool>>,
    /// Async file-task state, the web analogue of the native `FileWorker`.
    file_outcomes: Rc<RefCell<Vec<FileOutcome>>>,
    file_in_flight: Rc<Cell<bool>>,
}

struct Gfx {
    session: Session,
    damascene: Runner,
    surface: wgpu::Surface<'static>,
    queue: wgpu::Queue,
    device: wgpu::Device,
    window: Arc<Window>,
    config: wgpu::SurfaceConfiguration,
    /// Format for render-target views and pipelines. Differs from
    /// `config.format` when the swapchain offers only linear formats
    /// (Chromium WebGPU) and we re-view them as sRGB.
    render_format: wgpu::TextureFormat,
    canvas: web_sys::HtmlCanvasElement,
}

impl WebHost {
    fn new(viewport: Rect, app: VolumetricUiV2) -> Self {
        Self {
            viewport,
            app,
            gfx: Rc::new(RefCell::new(None)),
            started: false,
            last_pointer: None,
            modifiers: KeyModifiers::default(),
            last_cursor: Cursor::Default,
            pending_resize: Rc::new(Cell::new(None)),
            _resize_closure: None,
            _resize_observer: None,
            jobs: Rc::new(RefCell::new(JobQueue::default())),
            job_results: Rc::new(RefCell::new(Vec::new())),
            pump_scheduled: Rc::new(Cell::new(false)),
            file_outcomes: Rc::new(RefCell::new(Vec::new())),
            file_in_flight: Rc::new(Cell::new(false)),
        }
    }
}

/// Locate the host page's canvas element.
fn locate_canvas() -> web_sys::HtmlCanvasElement {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(CANVAS_ID))
        .unwrap_or_else(|| panic!("missing #{CANVAS_ID} canvas element"))
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap_or_else(|_| panic!("#{CANVAS_ID} is not a canvas"))
}

/// The canvas's CSS-laid-out box in physical pixels (device pixel ratio
/// applied) — what the swapchain backing buffer must match.
fn measure_canvas(canvas: &web_sys::HtmlCanvasElement, fallback: Rect) -> (u32, u32) {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0)
        .max(1.0);
    let css_w = if canvas.client_width() > 0 {
        canvas.client_width() as f64
    } else {
        fallback.w.max(1.0) as f64
    };
    let css_h = if canvas.client_height() > 0 {
        canvas.client_height() as f64
    } else {
        fallback.h.max(1.0) as f64
    };
    (((css_w * dpr).round() as u32).max(1), ((css_h * dpr).round() as u32).max(1))
}

/// sRGB-tagged sibling for a linear `*8Unorm` swapchain format, so the
/// hardware applies the sRGB encode on store (Chromium's WebGPU surface
/// offers only linear formats). `None` when the format has no sibling.
fn srgb_view_of(format: wgpu::TextureFormat) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    match format {
        F::Rgba8Unorm => Some(F::Rgba8UnormSrgb),
        F::Bgra8Unorm => Some(F::Bgra8UnormSrgb),
        _ => None,
    }
}

impl ApplicationHandler for WebHost {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.started {
            return;
        }
        self.started = true;

        let canvas = locate_canvas();

        // Bind the winit window to the existing canvas. Don't request an
        // inner size — winit-web would force the canvas backing buffer to
        // it and fight the CSS layout; the measure below and the
        // ResizeObserver own the buffer size instead.
        let attrs = Window::default_attributes().with_canvas(Some(canvas.clone()));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        // Size the backing buffer to the CSS box now, so the async surface
        // setup reads sensible initial dimensions instead of the canvas
        // default (300×150).
        let (w, h) = measure_canvas(&canvas, self.viewport);
        canvas.set_width(w);
        canvas.set_height(h);

        // Track CSS layout changes for the lifetime of the page. The
        // observer only records the measurement; the redraw applies it (the
        // native host's pending_resize pattern), so surface reconfiguration
        // stays on the frame path.
        let viewport = self.viewport;
        let pending = self.pending_resize.clone();
        let canvas_for_observer = canvas.clone();
        let window_for_observer = window.clone();
        let resize_closure: Closure<dyn FnMut()> = Closure::new(move || {
            pending.set(Some(measure_canvas(&canvas_for_observer, viewport)));
            window_for_observer.request_redraw();
        });
        let observer = web_sys::ResizeObserver::new(resize_closure.as_ref().unchecked_ref())
            .expect("ResizeObserver::new");
        observer.observe(&canvas);
        self._resize_closure = Some(resize_closure);
        self._resize_observer = Some(observer);

        // Adapter/device acquisition is async in the browser. Probe WebGPU
        // and fall back to WebGL2; stash the finished Gfx for the event
        // handlers to find. Failures land in the console — the canvas stays
        // blank, which is the honest outcome when the browser offers no
        // usable GPU path.
        let shaders = self.app.shaders();
        let theme = self.app.theme();
        let gfx_slot = self.gfx.clone();
        let window_for_async = window.clone();
        let canvas_for_async = canvas;
        wasm_bindgen_futures::spawn_local(async move {
            let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
            instance_desc.backends = wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL;
            let instance = wgpu::util::new_instance_with_webgpu_detection(instance_desc).await;
            let surface = match instance.create_surface(window_for_async.clone()) {
                Ok(surface) => surface,
                Err(err) => {
                    log::error!("could not create a rendering surface: {err}");
                    return;
                }
            };
            let adapter = match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
            {
                Ok(adapter) => adapter,
                Err(err) => {
                    log::error!(
                        "no compatible GPU adapter — neither usable WebGPU nor WebGL2 ({err})"
                    );
                    return;
                }
            };
            let info = adapter.get_info();
            log::info!(
                "adapter selected — backend={:?} name={:?} device_type={:?}",
                info.backend,
                info.name,
                info.device_type,
            );

            // naga's GLSL ES target rejects some WGSL features at module
            // creation; the runner has to know up front what to downlevel.
            let caps = RunnerCaps::from_adapter(&adapter);

            // WebGL2's envelope is the baseline both browser backends can
            // satisfy; cap resolution limits at the adapter's real ones.
            // Preview meshes of fine models routinely pass the 256 MiB
            // default buffer ceiling (same rationale as the native host),
            // so take whatever the adapter allows there.
            let mut limits =
                wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
            limits.max_buffer_size = adapter.limits().max_buffer_size;

            let (device, queue) = match adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("volumetric_ui_v2::web_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                })
                .await
            {
                Ok(pair) => pair,
                Err(err) => {
                    log::error!("GPU device creation failed: {err}");
                    return;
                }
            };

            let surface_caps = surface.get_capabilities(&adapter);
            let format = surface_caps
                .formats
                .iter()
                .copied()
                .find(|format| format.is_srgb())
                .unwrap_or(surface_caps.formats[0]);
            let render_format = srgb_view_of(format).unwrap_or(format);
            let view_formats = if render_format != format {
                vec![render_format]
            } else {
                Vec::new()
            };
            // COPY_SRC feeds backdrop-sampling shaders; WebGL2 surfaces may
            // not offer it, in which case those shaders are skipped below.
            let want_copy_src = surface_caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
            let usage = if want_copy_src {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT
            };
            let present_mode = if surface_caps
                .present_modes
                .contains(&wgpu::PresentMode::Fifo)
            {
                wgpu::PresentMode::Fifo
            } else {
                surface_caps.present_modes[0]
            };
            let inner = window_for_async.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage,
                format,
                width: inner.width.max(1),
                height: inner.height.max(1),
                present_mode,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats,
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&device, &config);

            let mut damascene = Runner::with_caps(&device, &queue, render_format, 1, caps);
            damascene.set_theme(theme);
            damascene.set_surface_size(config.width, config.height);
            for shader in shaders {
                if shader.samples_backdrop && !want_copy_src {
                    continue;
                }
                damascene.register_shader_with(
                    &device,
                    shader.name,
                    shader.wgsl,
                    shader.samples_backdrop,
                    shader.samples_time,
                );
            }

            *gfx_slot.borrow_mut() = Some(Gfx {
                session: Session::new(&device, &queue, render_format),
                damascene,
                surface,
                queue,
                device,
                window: window_for_async.clone(),
                config,
                render_format,
                canvas: canvas_for_async,
            });
            window_for_async.request_redraw();
        });
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Clone the cell so the borrow isn't tied to `&self.gfx` — redraw
        // and the input arms need `&mut self` alongside the borrowed Gfx.
        let gfx_cell = self.gfx.clone();
        let mut gfx_borrow = gfx_cell.borrow_mut();
        let Some(gfx) = gfx_borrow.as_mut() else {
            // GPU setup hasn't finished; the post-setup request_redraw
            // brings us back.
            return;
        };
        let scale = gfx.window.scale_factor() as f32;

        match event {
            // Browsers have no CloseRequested; winit-web resize reports are
            // folded into the same pending_resize path the observer uses.
            // Skip echoes of the current size — redraw's canvas
            // set_width/set_height bounces one back through winit.
            WindowEvent::Resized(size) => {
                let w = size.width.max(1);
                let h = size.height.max(1);
                let same_as_current = self.pending_resize.get().is_none()
                    && w == gfx.config.width
                    && h == gfx.config.height;
                if same_as_current {
                    return;
                }
                self.pending_resize.set(Some((w, h)));
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
                        for event in gfx.damascene.pointer_down(Pointer::mouse(lx, ly, button)) {
                            dispatch_event(&mut self.app, &gfx.damascene, event);
                        }
                    }
                    ElementState::Released => {
                        gfx.session.pointer_up(button);
                        for event in gfx.damascene.pointer_up(Pointer::mouse(lx, ly, button)) {
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
                    for event in gfx
                        .damascene
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
            WindowEvent::RedrawRequested => self.redraw(gfx),
            _ => {}
        }
    }
}

impl WebHost {
    /// One frame — the same pump order as the native host's redraw, with
    /// the worker channels replaced by the shared inline-executor cells.
    fn redraw(&mut self, gfx: &mut Gfx) {
        if let Some((w, h)) = self.pending_resize.take() {
            // Keep the canvas backing buffer in lockstep with the surface —
            // winit-web reads inner_size from canvas width/height.
            gfx.canvas.set_width(w);
            gfx.canvas.set_height(h);
            gfx.config.width = w;
            gfx.config.height = h;
            gfx.surface.configure(&gfx.device, &gfx.config);
            gfx.damascene
                .set_surface_size(gfx.config.width, gfx.config.height);
        }

        let frame = match gfx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                gfx.surface.configure(&gfx.device, &gfx.config);
                // Unlike native (where OS expose events and the worker
                // thread keep things moving), a dropped web frame must
                // re-arm itself or queued jobs/results stall until the
                // next user input.
                gfx.window.request_redraw();
                return;
            }
            other => {
                log::error!("surface unavailable: {other:?}");
                gfx.window.request_redraw();
                return;
            }
        };
        // Render through the sRGB view of a linear swapchain format (no-op
        // when the swapchain is already sRGB).
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(gfx.render_format),
            ..Default::default()
        });

        gfx.session
            .pre_frame(&mut self.app, self.job_results.borrow_mut().drain(..));

        // The browser shell always executes locally; a Remote request can
        // only arrive through a stale code path (the toggle is hidden on
        // wasm32) — answer it honestly rather than silently.
        if let Some(choice) = self.app.take_executor_request() {
            if matches!(choice, ExecutorChoice::Remote(_)) {
                self.app
                    .set_status("remote build isn't available in the browser; building locally");
            }
        }

        // Apply finished file operations, then launch any newly queued
        // action — before the session sync so an opened project's run
        // starts this frame.
        for outcome in self.file_outcomes.borrow_mut().drain(..) {
            apply_file_outcome(&mut self.app, outcome);
        }
        if let Some(action) = self.app.take_file_action() {
            if self.file_in_flight.get() {
                self.app
                    .set_status("a file operation is already in progress");
            } else if let Some(task) = gather_file_task(&mut self.app, &gfx.session, action) {
                self.spawn_file_task(gfx.window.clone(), task);
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
            self.jobs.borrow_mut().push(job);
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
                label: Some("volumetric_ui_v2::web_encoder"),
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

        // Pump one queued background job on a later task, so this frame
        // (with its freshly painted "running" status) reaches the
        // compositor before the main thread blocks on the job.
        self.schedule_job_pump(gfx.window.clone());
    }

    /// Arrange for one queued job to execute on a `setTimeout(0)` turn.
    /// Reentrancy-guarded; each completed job requests a redraw, and that
    /// redraw reschedules the pump while jobs remain.
    fn schedule_job_pump(&self, window: Arc<Window>) {
        if self.pump_scheduled.get() || self.jobs.borrow().is_empty() {
            return;
        }
        self.pump_scheduled.set(true);
        let jobs = self.jobs.clone();
        let results = self.job_results.clone();
        let scheduled = self.pump_scheduled.clone();
        let closure = Closure::once_into_js(move || {
            let job = jobs.borrow_mut().pop();
            if let Some(job) = job {
                // Progress snapshots pile up in the shared cell during the
                // (blocking) execution and land in the UI together with the
                // terminal result — a single-threaded shell can't repaint
                // mid-job anyway.
                let backend = LocalBackend;
                let emit = |progress: BackgroundResult| {
                    results.borrow_mut().push(progress);
                };
                let result = execute_job_monitored(job, &backend as &dyn ExecutionBackend, &emit);
                // The UI surfaces these too, but a console line carries the
                // full text — the status chips truncate.
                match &result {
                    BackgroundResult::ProjectComplete {
                        result: Err(err), ..
                    } => log::error!("project run failed: {err}"),
                    BackgroundResult::PreviewComplete(preview) => {
                        if let Err(crate::session::PreviewBuildError::Failed(err)) =
                            &preview.result
                        {
                            log::error!("preview build failed: {err}");
                        }
                    }
                    _ => {}
                }
                results.borrow_mut().push(result);
            }
            scheduled.set(false);
            window.request_redraw();
        });
        if let Some(js_window) = web_sys::window() {
            let _ = js_window.set_timeout_with_callback_and_timeout_and_arguments_0(
                closure.unchecked_ref(),
                0,
            );
        }
    }

    /// Launch an async file task; its outcome lands in `file_outcomes` and
    /// wakes the frame loop.
    fn spawn_file_task(&self, window: Arc<Window>, task: FileTask) {
        self.file_in_flight.set(true);
        let outcomes = self.file_outcomes.clone();
        let in_flight = self.file_in_flight.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let outcome = perform_file_task(task).await;
            outcomes.borrow_mut().push(outcome);
            in_flight.set(false);
            window.request_redraw();
        });
    }
}

/// A file operation with everything it needs gathered up front. Unlike the
/// native shell, saves carry their serialized bytes (no dialog on the way
/// out — the browser download picks the destination).
enum FileTask {
    OpenProject,
    SaveProject {
        bytes: Result<Vec<u8>, String>,
        name: String,
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

/// What a finished file task reports back; applied on the frame path by
/// [`apply_file_outcome`]. Same shape as the native shell's outcomes.
enum FileOutcome {
    OpenedProject {
        path: PathBuf,
        result: Result<volumetric::Project, String>,
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
    Status(String),
    Dismissed,
}

/// Converts a queued [`FileAction`] into a self-contained [`FileTask`],
/// snapshotting the app/session state the async task needs.
fn gather_file_task(
    app: &mut VolumetricUiV2,
    session: &Session,
    action: FileAction,
) -> Option<FileTask> {
    let save_task = |app: &VolumetricUiV2, name: String| FileTask::SaveProject {
        bytes: app.project().to_cbor().map_err(|err| err.to_string()),
        name,
    };
    match action {
        FileAction::OpenProject => Some(FileTask::OpenProject),
        FileAction::SaveProject => Some(save_task(app, "project.vproj".to_string())),
        FileAction::SaveProjectTo(path) => {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("project.vproj")
                .to_string();
            Some(save_task(app, name))
        }
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

/// Runs a file task to completion: async picker for reads, Blob/anchor
/// download for writes. Every task resolves to exactly one outcome
/// (`Dismissed` when the picker was cancelled) so `file_in_flight` always
/// clears.
async fn perform_file_task(task: FileTask) -> FileOutcome {
    match task {
        FileTask::OpenProject => {
            let Some(handle) = rfd::AsyncFileDialog::new()
                .add_filter("Project", &["vproj"])
                .pick_file()
                .await
            else {
                return FileOutcome::Dismissed;
            };
            let name = handle.file_name();
            let bytes = handle.read().await;
            let result = volumetric::Project::from_cbor(&bytes).map_err(|err| err.to_string());
            FileOutcome::OpenedProject {
                path: PathBuf::from(name),
                result,
            }
        }
        FileTask::SaveProject { bytes, name } => {
            let result = bytes.and_then(|bytes| trigger_download(&name, &bytes));
            FileOutcome::SavedProject {
                path: PathBuf::from(name),
                result,
            }
        }
        FileTask::ExportStl { id, triangles } => {
            let bytes = volumetric::stl::triangles_to_binary_stl_bytes(&triangles, "volumetric");
            let name = format!("{id}.stl");
            match trigger_download(&name, &bytes) {
                Ok(()) => FileOutcome::Status(format!(
                    "exported {} triangles as {name}",
                    triangles.len()
                )),
                Err(err) => FileOutcome::Status(format!("failed to export STL: {err}")),
            }
        }
        FileTask::ExportWasm { id, bytes } => {
            let name = format!("{id}.wasm");
            match trigger_download(&name, bytes.as_slice()) {
                Ok(()) => FileOutcome::Status(format!("exported {name}")),
                Err(err) => FileOutcome::Status(format!("failed to export WASM: {err}")),
            }
        }
        FileTask::ImportWasm => {
            let Some(handle) = rfd::AsyncFileDialog::new()
                .add_filter("WASM", &["wasm"])
                .pick_file()
                .await
            else {
                return FileOutcome::Dismissed;
            };
            let file_name = handle.file_name();
            let name = std::path::Path::new(&file_name)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("model")
                .to_string();
            let bytes = handle.read().await;
            FileOutcome::ImportedWasm { name, bytes }
        }
        FileTask::ImportBlob {
            filter_name,
            extensions,
            operator_name,
            output_base,
        } => {
            let Some(handle) = rfd::AsyncFileDialog::new()
                .add_filter(filter_name, extensions)
                .pick_file()
                .await
            else {
                return FileOutcome::Dismissed;
            };
            let bytes = handle.read().await;
            FileOutcome::ImportedBlob {
                operator_name,
                output_base,
                bytes,
            }
        }
    }
}

/// Routes a finished file operation back into the app (frame path).
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

/// Save `bytes` as a browser download named `name` (Blob → object URL →
/// synthetic anchor click).
fn trigger_download(name: &str, bytes: &[u8]) -> Result<(), String> {
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or("no document")?;
    let array = js_sys::Array::new();
    array.push(&js_sys::Uint8Array::from(bytes));
    let blob = web_sys::Blob::new_with_u8_array_sequence(&array)
        .map_err(|err| format!("Blob creation failed: {err:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|err| format!("object URL creation failed: {err:?}"))?;
    let anchor = document
        .create_element("a")
        .map_err(|err| format!("anchor creation failed: {err:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "anchor element has the wrong type".to_string())?;
    anchor.set_href(&url);
    anchor.set_download(name);
    // Clicks on detached anchors have historically been flaky in Firefox;
    // attach for the click.
    let body = document.body().ok_or("no body")?;
    let _ = body.append_child(&anchor);
    anchor.click();
    anchor.remove();
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

// ---- winit → damascene input mapping ----
//
// `damascene_winit_wgpu::host::input` provides these on native, but that
// crate carries native-only dependencies, so the browser shell keeps its
// own copies (same winit major version, same tables).

fn map_key(key: &winit::keyboard::Key) -> Option<UiKey> {
    use winit::keyboard::{Key, NamedKey};
    match key {
        Key::Named(NamedKey::Enter) => Some(UiKey::Enter),
        Key::Named(NamedKey::Escape) => Some(UiKey::Escape),
        Key::Named(NamedKey::Tab) => Some(UiKey::Tab),
        Key::Named(NamedKey::Space) => Some(UiKey::Space),
        Key::Named(NamedKey::ArrowUp) => Some(UiKey::ArrowUp),
        Key::Named(NamedKey::ArrowDown) => Some(UiKey::ArrowDown),
        Key::Named(NamedKey::ArrowLeft) => Some(UiKey::ArrowLeft),
        Key::Named(NamedKey::ArrowRight) => Some(UiKey::ArrowRight),
        Key::Named(NamedKey::Backspace) => Some(UiKey::Backspace),
        Key::Named(NamedKey::Delete) => Some(UiKey::Delete),
        Key::Named(NamedKey::Home) => Some(UiKey::Home),
        Key::Named(NamedKey::End) => Some(UiKey::End),
        Key::Named(NamedKey::PageUp) => Some(UiKey::PageUp),
        Key::Named(NamedKey::PageDown) => Some(UiKey::PageDown),
        Key::Character(s) => Some(UiKey::Character(s.to_string())),
        Key::Named(named) => Some(UiKey::Other(format!("{named:?}"))),
        _ => None,
    }
}

fn pointer_button(b: winit::event::MouseButton) -> Option<PointerButton> {
    use winit::event::MouseButton;
    match b {
        MouseButton::Left => Some(PointerButton::Primary),
        MouseButton::Right => Some(PointerButton::Secondary),
        MouseButton::Middle => Some(PointerButton::Middle),
        _ => None,
    }
}

fn key_modifiers(mods: winit::keyboard::ModifiersState) -> KeyModifiers {
    KeyModifiers {
        shift: mods.shift_key(),
        ctrl: mods.control_key(),
        alt: mods.alt_key(),
        logo: mods.super_key(),
    }
}

/// winit-web writes the icon through to the canvas's CSS `cursor`.
fn winit_cursor(cursor: Cursor) -> winit::window::CursorIcon {
    use winit::window::CursorIcon;
    match cursor {
        Cursor::Default => CursorIcon::Default,
        Cursor::Pointer => CursorIcon::Pointer,
        Cursor::Text => CursorIcon::Text,
        Cursor::NotAllowed => CursorIcon::NotAllowed,
        Cursor::Grab => CursorIcon::Grab,
        Cursor::Grabbing => CursorIcon::Grabbing,
        Cursor::Move => CursorIcon::Move,
        Cursor::EwResize => CursorIcon::EwResize,
        Cursor::NsResize => CursorIcon::NsResize,
        Cursor::NwseResize => CursorIcon::NwseResize,
        Cursor::NeswResize => CursorIcon::NeswResize,
        Cursor::ColResize => CursorIcon::ColResize,
        Cursor::RowResize => CursorIcon::RowResize,
        Cursor::Crosshair => CursorIcon::Crosshair,
        _ => CursorIcon::Default,
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
