use std::sync::Arc;

use aetna_core::prelude::*;
use aetna_core::{Cursor, KeyModifiers, PointerButton, UiKey};
use aetna_wgpu::Runner;
use glam::{Vec2, Vec3};
use volumetric_renderer as renderer;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, Window, WindowId};

use crate::{PreviewMeshPlan, PreviewRenderMode, PreviewRequest, VIEWPORT_KEY, VolumetricUiV2};

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
    aetna: Runner,
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

        let mut aetna = Runner::new(&device, &queue, format);
        aetna.set_theme(self.app.theme());
        aetna.set_surface_size(config.width, config.height);
        for shader in self.app.shaders() {
            aetna.register_shader_with(
                &device,
                shader.name,
                shader.wgsl,
                shader.samples_backdrop,
                shader.samples_time,
            );
        }

        self.gfx = Some(Gfx {
            viewport_renderer: ViewportRenderer::new(&device, &queue, format),
            aetna,
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
                        let moved = gfx.aetna.pointer_moved(lx, ly);
                        for event in moved.events {
                            self.app.on_event(event);
                        }
                        if moved.needs_redraw {
                            gfx.window.request_redraw();
                        }
                    }
                    WindowEvent::CursorLeft { .. } => {
                        self.last_pointer = None;
                        self.last_camera_pointer = None;
                        self.viewport_buttons = ViewportPointerButtons::default();
                        gfx.aetna.pointer_left();
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
                                for event in gfx.aetna.pointer_down(lx, ly, button) {
                                    self.app.on_event(event);
                                }
                            }
                            ElementState::Released => {
                                for event in gfx.aetna.pointer_up(lx, ly, button) {
                                    self.app.on_event(event);
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
                        if gfx.aetna.pointer_wheel(lx, ly, dy) {
                            gfx.window.request_redraw();
                        }
                    }
                    WindowEvent::ModifiersChanged(modifiers) => {
                        self.modifiers = key_modifiers(modifiers.state());
                        gfx.aetna.set_modifiers(self.modifiers);
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
                            for event in gfx.aetna.key_down(key, self.modifiers, key_event.repeat) {
                                self.app.on_event(event);
                            }
                        }
                        if let Some(text) = &key_event.text
                            && let Some(event) = gfx.aetna.text_input(text.to_string())
                        {
                            self.app.on_event(event);
                        }
                        gfx.window.request_redraw();
                    }
                    WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                        if let Some(event) = gfx.aetna.text_input(text) {
                            self.app.on_event(event);
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
            gfx.aetna
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

        self.app.before_build();
        let theme = self.app.theme();
        let palette = theme.palette().clone();
        let cx = aetna_core::BuildCx::new(&theme);
        let mut tree = self.app.build(&cx);

        gfx.aetna.set_theme(theme);
        gfx.aetna.set_hotkeys(self.app.hotkeys());
        gfx.aetna.set_selection(self.app.selection());
        gfx.aetna.push_toasts(self.app.drain_toasts());

        let scale_factor = gfx.window.scale_factor() as f32;
        let viewport = Rect::new(
            0.0,
            0.0,
            gfx.config.width as f32 / scale_factor,
            gfx.config.height as f32 / scale_factor,
        );
        let prepare = gfx
            .aetna
            .prepare(&gfx.device, &gfx.queue, &mut tree, viewport, scale_factor);

        let cursor = gfx.aetna.ui_state().cursor(&tree);
        if cursor != self.last_cursor {
            gfx.window.set_cursor(winit_cursor(cursor));
            self.last_cursor = cursor;
        }

        let mut encoder = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("volumetric_ui_v2::encoder"),
            });

        let viewport_rect = gfx.aetna.rect_of_key(VIEWPORT_KEY);
        self.last_viewport_rect = viewport_rect;

        gfx.viewport_renderer.render(
            &gfx.device,
            &gfx.queue,
            &mut encoder,
            &view,
            viewport_rect,
            scale_factor,
            (gfx.config.width, gfx.config.height),
            bg_color(&palette),
            self.app.preview_request(),
        );
        gfx.aetna.render(
            &gfx.device,
            &mut encoder,
            &frame.texture,
            &view,
            None,
            wgpu::LoadOp::Load,
        );

        gfx.queue.submit(Some(encoder.finish()));
        frame.present();

        if prepare.needs_redraw {
            gfx.window.request_redraw();
        }
    }
}

struct ViewportRenderer {
    renderer: renderer::Renderer,
    target: ViewportTarget,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    surface_format: wgpu::TextureFormat,
    scene_cache: Option<PreviewSceneCache>,
    camera: Option<renderer::Camera>,
}

impl ViewportRenderer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let mut renderer = renderer::Renderer::new(format);
        renderer.initialize(device, queue, None);

        let target = ViewportTarget::new(device, (1, 1), format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetric_ui_v2::viewport_blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("volumetric_ui_v2::viewport_blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let blit_bind_group =
            create_blit_bind_group(device, &blit_bind_group_layout, &target.view, &sampler);
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric_ui_v2::viewport_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(VIEWPORT_BLIT_WGSL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetric_ui_v2::viewport_blit_layout"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric_ui_v2::viewport_blit_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            renderer,
            target,
            blit_pipeline,
            blit_bind_group_layout,
            blit_bind_group,
            sampler,
            surface_format: format,
            scene_cache: None,
            camera: None,
        }
    }

    fn resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // The renderer's real size is the keyed viewport rect, not the full window.
        // It is resized during `render` once Aetna layout has resolved that rect.
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        logical_rect: Option<Rect>,
        scale_factor: f32,
        surface_extent: (u32, u32),
        clear_color: wgpu::Color,
        preview_request: Option<PreviewRequest>,
    ) {
        clear_viewport(encoder, target_view, clear_color);

        let Some(rect) = logical_rect else {
            return;
        };

        let Some((x, y, w, h)) = physical_rect(rect, scale_factor, surface_extent) else {
            return;
        };

        if self.target.extent != (w, h) {
            self.target = ViewportTarget::new(device, (w, h), self.surface_format);
            self.blit_bind_group = create_blit_bind_group(
                device,
                &self.blit_bind_group_layout,
                &self.target.view,
                &self.sampler,
            );
        }

        self.renderer.set_viewport_size(device, w, h);
        let (scene, camera) = self.scene_for_request(preview_request.as_ref());
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

        let settings = render_settings(preview_request.as_ref(), clear_color);
        self.renderer.render(
            device,
            queue,
            encoder,
            &camera,
            &settings,
            &self.target.view,
        );
        self.renderer.end_frame();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volumetric_ui_v2::viewport_blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_scissor_rect(x, y, w, h);
        pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &self.blit_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn apply_camera_input(
        &mut self,
        input: &renderer::CameraInputState,
        scheme: renderer::CameraControlScheme,
    ) -> bool {
        let Some(camera) = &mut self.camera else {
            return false;
        };

        match camera_action_with_left_fallback(scheme, input) {
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

    fn scene_for_request(
        &mut self,
        request: Option<&PreviewRequest>,
    ) -> (renderer::SceneData, renderer::Camera) {
        let Some(request) = request else {
            if self.scene_cache.is_some() {
                self.scene_cache = None;
                self.camera = Some(renderer::test_scenes::create_test_camera());
            }
            let camera = self
                .camera
                .get_or_insert_with(renderer::test_scenes::create_test_camera)
                .clone();
            return (renderer::test_scenes::create_test_scene(), camera);
        };

        let key = PreviewSceneKey::from(request);
        let rebuild = self
            .scene_cache
            .as_ref()
            .is_none_or(|cache| cache.key != key);
        if rebuild {
            self.scene_cache = Some(PreviewSceneCache {
                key,
                result: build_preview_scene(request),
            });
            self.camera = self
                .scene_cache
                .as_ref()
                .and_then(|cache| cache.result.as_ref().ok().map(|(_, camera)| camera.clone()));
        }

        let scene = self
            .scene_cache
            .as_ref()
            .and_then(|cache| cache.result.as_ref().ok().map(|(scene, _)| scene.clone()))
            .unwrap_or_else(renderer::test_scenes::create_test_scene);
        let camera = self
            .camera
            .get_or_insert_with(renderer::test_scenes::create_test_camera)
            .clone();
        (scene, camera)
    }
}

fn camera_action_with_left_fallback(
    scheme: renderer::CameraControlScheme,
    input: &renderer::CameraInputState,
) -> renderer::CameraAction {
    let action = scheme.determine_action(input);
    if action != renderer::CameraAction::None || !input.left_down {
        return action;
    }

    if input.shift_down || input.ctrl_down {
        renderer::CameraAction::Pan
    } else {
        renderer::CameraAction::Orbit
    }
}

struct ViewportTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    extent: (u32, u32),
}

impl ViewportTarget {
    fn new(device: &wgpu::Device, extent: (u32, u32), format: wgpu::TextureFormat) -> Self {
        let extent = (extent.0.max(1), extent.1.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
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
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
            extent,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreviewSceneKey {
    asset_id: String,
    render_mode: PreviewRenderMode,
    mesh_plan: PreviewMeshPlan,
}

impl From<&PreviewRequest> for PreviewSceneKey {
    fn from(request: &PreviewRequest) -> Self {
        Self {
            asset_id: request.asset_id.clone(),
            render_mode: request.render_mode,
            mesh_plan: request.mesh_plan.clone(),
        }
    }
}

type PreviewScene = (renderer::SceneData, renderer::Camera);

struct PreviewSceneCache {
    key: PreviewSceneKey,
    result: Result<PreviewScene, String>,
}

fn build_preview_scene(request: &PreviewRequest) -> Result<PreviewScene, String> {
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

    Ok((scene, camera_for_bounds(bounds_min, bounds_max)))
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

fn camera_for_bounds(bounds_min: (f32, f32, f32), bounds_max: (f32, f32, f32)) -> renderer::Camera {
    let min = Vec3::new(bounds_min.0, bounds_min.1, bounds_min.2);
    let max = Vec3::new(bounds_max.0, bounds_max.1, bounds_max.2);
    let mut camera = renderer::Camera::default();
    camera.focus_on(min, max);
    camera
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

fn create_blit_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    texture_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("volumetric_ui_v2::viewport_blit_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn format_error_chain(error: anyhow::Error) -> String {
    error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

fn clear_viewport(
    encoder: &mut wgpu::CommandEncoder,
    target_view: &wgpu::TextureView,
    clear_color: wgpu::Color,
) {
    let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("volumetric_ui_v2::viewport_clear_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

fn physical_rect(
    rect: Rect,
    scale_factor: f32,
    extent: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    let max_w = extent.0 as f32;
    let max_h = extent.1 as f32;
    let x0 = (rect.x * scale_factor).floor().clamp(0.0, max_w) as u32;
    let y0 = (rect.y * scale_factor).floor().clamp(0.0, max_h) as u32;
    let x1 = ((rect.x + rect.w) * scale_factor).ceil().clamp(0.0, max_w) as u32;
    let y1 = ((rect.y + rect.h) * scale_factor).ceil().clamp(0.0, max_h) as u32;

    let w = x1.checked_sub(x0)?;
    let h = y1.checked_sub(y0)?;
    (w > 0 && h > 0).then_some((x0, y0, w, h))
}

fn pointer_in_rect(rect: Option<Rect>, x: f32, y: f32) -> bool {
    rect.is_some_and(|rect| {
        x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h
    })
}

const VIEWPORT_BLIT_WGSL: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    out.position = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv = vec2<f32>(
        (out.position.x + 1.0) * 0.5,
        1.0 - (out.position.y + 1.0) * 0.5
    );
    return out;
}

@group(0) @binding(0)
var t_source: texture_2d<f32>;

@group(0) @binding(1)
var s_source: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t_source, s_source, in.uv);
}
"#;

fn map_key(key: &Key) -> Option<UiKey> {
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

fn pointer_button(button: MouseButton) -> Option<PointerButton> {
    match button {
        MouseButton::Left => Some(PointerButton::Primary),
        MouseButton::Right => Some(PointerButton::Secondary),
        MouseButton::Middle => Some(PointerButton::Middle),
        _ => None,
    }
}

fn key_modifiers(modifiers: winit::keyboard::ModifiersState) -> KeyModifiers {
    KeyModifiers {
        shift: modifiers.shift_key(),
        ctrl: modifiers.control_key(),
        alt: modifiers.alt_key(),
        logo: modifiers.super_key(),
    }
}

fn winit_cursor(cursor: Cursor) -> CursorIcon {
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

fn bg_color(palette: &aetna_core::Palette) -> wgpu::Color {
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
