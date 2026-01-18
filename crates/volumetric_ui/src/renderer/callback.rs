//! egui callback integration for the unified renderer.
//!
//! Provides a callback that can be used with egui's paint callback system
//! to render 3D content within an egui panel.

#![allow(dead_code)]

use eframe::egui;
use eframe::egui_wgpu;

use super::{
    Camera, LineData, LineStyle, MaterialId, MeshData, PointData, PointStyle, RenderSettings,
    Renderer,
};
use glam::Mat4;

/// Data for a single frame's rendering.
#[derive(Clone)]
pub struct SceneData {
    /// Meshes to render
    pub meshes: Vec<(MeshData, Mat4, MaterialId)>,
    /// Lines to render
    pub lines: Vec<(LineData, Mat4, LineStyle)>,
    /// Points to render
    pub points: Vec<(PointData, Mat4, PointStyle)>,
}

impl Default for SceneData {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            lines: Vec::new(),
            points: Vec::new(),
        }
    }
}

impl SceneData {
    /// Create a new empty scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mesh to the scene.
    pub fn add_mesh(&mut self, mesh: MeshData, transform: Mat4, material: MaterialId) {
        self.meshes.push((mesh, transform, material));
    }

    /// Add lines to the scene.
    pub fn add_lines(&mut self, lines: LineData, transform: Mat4, style: LineStyle) {
        self.lines.push((lines, transform, style));
    }

    /// Add points to the scene.
    pub fn add_points(&mut self, points: PointData, transform: Mat4, style: PointStyle) {
        self.points.push((points, transform, style));
    }

    /// Check if the scene is empty.
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty() && self.lines.is_empty() && self.points.is_empty()
    }

    /// Clear all scene data.
    pub fn clear(&mut self) {
        self.meshes.clear();
        self.lines.clear();
        self.points.clear();
    }
}

/// Draw data passed to the callback.
#[derive(Clone)]
pub struct SceneDrawData {
    /// Scene geometry to render
    pub scene: SceneData,
    /// Camera for viewing
    pub camera: Camera,
    /// Render settings
    pub settings: RenderSettings,
    /// Viewport size in pixels
    pub viewport_size: [u32; 2],
    /// Target texture format
    pub target_format: wgpu::TextureFormat,
}

/// egui callback for rendering 3D scenes.
pub struct SceneCallback {
    pub data: SceneDrawData,
}

impl egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Get or create the renderer
        let renderer = callback_resources
            .entry::<SceneRendererResource>()
            .or_insert_with(|| SceneRendererResource::new(device, queue, self.data.target_format));

        // Resize target texture if viewport changed
        let new_size = (
            self.data.viewport_size[0].max(1),
            self.data.viewport_size[1].max(1),
        );
        if renderer.size != new_size {
            renderer.resize(device, new_size);
        }

        // Ensure renderer viewport matches
        renderer.renderer.set_viewport_size(
            device,
            self.data.viewport_size[0],
            self.data.viewport_size[1],
        );

        // Submit scene data
        for (mesh, transform, material) in &self.data.scene.meshes {
            renderer.renderer.submit_mesh(mesh, *transform, *material);
        }
        for (lines, transform, style) in &self.data.scene.lines {
            renderer.renderer.submit_lines(lines, *transform, style.clone());
        }
        for (points, transform, style) in &self.data.scene.points {
            renderer.renderer.submit_points(points, *transform, style.clone());
        }

        // Render to the offscreen target
        renderer.renderer.render(
            device,
            queue,
            egui_encoder,
            &self.data.camera,
            &self.data.settings,
            &renderer.target_view,
        );

        renderer.renderer.end_frame();

        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(renderer) = callback_resources.get::<SceneRendererResource>() else {
            return;
        };

        // Set viewport
        let vp = info.viewport_in_pixels();
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.from_bottom_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );

        // Blit the rendered scene to the egui target
        render_pass.set_pipeline(&renderer.blit_pipeline);
        render_pass.set_bind_group(0, &renderer.blit_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Resource stored in egui's callback resources.
struct SceneRendererResource {
    renderer: Renderer,
    /// Offscreen render target
    target_texture: wgpu::Texture,
    target_view: wgpu::TextureView,
    /// Blit pipeline to copy to egui's target
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Current size
    size: (u32, u32),
    target_format: wgpu::TextureFormat,
}

impl SceneRendererResource {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
        let size = (1, 1);

        // Create offscreen target
        let (target_texture, target_view) = Self::create_target(device, size, target_format);

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene_blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create blit pipeline
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BLIT_SHADER)),
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_blit_bgl"),
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

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene_blit_pipeline_layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene_blit_pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            // On native, egui's main render pass has a depth attachment (Depth24Plus).
            // On WebGL2/WebGPU, the render pass may not have a depth attachment.
            #[cfg(not(target_arch = "wasm32"))]
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            #[cfg(target_arch = "wasm32")]
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let blit_bind_group = Self::create_blit_bind_group(
            device,
            &blit_bind_group_layout,
            &target_view,
            &sampler,
        );

        // Create and initialize renderer
        let mut renderer = Renderer::new(target_format);
        renderer.initialize(device, queue, None); // No adapter available, but that's OK

        Self {
            renderer,
            target_texture,
            target_view,
            blit_pipeline,
            blit_bind_group,
            blit_bind_group_layout,
            sampler,
            size,
            target_format,
        }
    }

    fn create_target(
        device: &wgpu::Device,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene_target"),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
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
        (texture, view)
    }

    /// Resize the offscreen target texture.
    fn resize(&mut self, device: &wgpu::Device, new_size: (u32, u32)) {
        let (new_texture, new_view) = Self::create_target(device, new_size, self.target_format);
        self.target_texture = new_texture;
        self.target_view = new_view;
        self.blit_bind_group = Self::create_blit_bind_group(
            device,
            &self.blit_bind_group_layout,
            &self.target_view,
            &self.sampler,
        );
        self.size = new_size;
    }

    fn create_blit_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_blit_bg"),
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
}

const BLIT_SHADER: &str = r#"
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
