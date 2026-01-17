use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu;
use glam::Mat4;
use wgpu::util::DeviceExt as _;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeshUniforms {
    view_proj: [[f32; 4]; 4],
    light_dir_world: [f32; 3],
    _pad0: f32,
    base_color: [f32; 3],
    _pad1: f32,
}

#[derive(Clone)]
pub struct MarchingCubesDrawData {
    pub vertices: Arc<Vec<MeshVertex>>,
    pub view_proj: Mat4,
    pub viewport_size_px: [u32; 2],
    pub ssao_enabled: bool,
    pub ssao_radius: f32,
    pub ssao_bias: f32,
    pub ssao_strength: f32,
    pub target_format: wgpu::TextureFormat,
}

pub struct MarchingCubesCallback {
    pub data: MarchingCubesDrawData,
}

impl egui_wgpu::CallbackTrait for MarchingCubesCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu = callback_resources
            .entry::<MarchingCubesGpu>()
            .or_insert_with(|| MarchingCubesGpu::new(device, self.data.target_format));

        gpu.ensure_resources(device, self.data.target_format, self.data.viewport_size_px);
        gpu.update_uniforms(
            queue,
            self.data.view_proj,
            self.data.ssao_enabled,
            self.data.ssao_radius,
            self.data.ssao_bias,
            self.data.ssao_strength,
        );
        gpu.update_vertices(device, queue, &self.data.vertices);
        gpu.encode_gbuffer_and_ssao(egui_encoder, self.data.ssao_enabled);

        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = callback_resources.get::<MarchingCubesGpu>() else {
            return;
        };

        if gpu.vertex_count == 0 {
            return;
        }

        let vp = info.viewport_in_pixels();
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.from_bottom_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );

        render_pass.set_pipeline(&gpu.composite_pipeline);
        render_pass.set_bind_group(0, &gpu.composite_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SsaoUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    screen_size_px: [f32; 2],
    radius: f32,
    bias: f32,
    strength: f32,
    // WGSL uniform layout rounds struct size up to a 16-byte multiple.
    // With the current fields, the shader expects 160 bytes.
    _pad0: [f32; 3],
}

// Keep Rust-side layout in sync with WGSL expectations.
const _: [(); 160] = [(); std::mem::size_of::<SsaoUniforms>()];

struct MarchingCubesGpu {
    target_format: wgpu::TextureFormat,
    viewport_size_px: [u32; 2],

    gbuffer_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,

    mesh_uniform_buffer: wgpu::Buffer,
    mesh_uniform_bind_group: wgpu::BindGroup,

    ssao_uniform_buffer: wgpu::Buffer,
    ssao_uniform_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,

    sampler: wgpu::Sampler,

    g_color_tex: wgpu::Texture,
    g_color_view: wgpu::TextureView,
    g_normal_tex: wgpu::Texture,
    g_normal_view: wgpu::TextureView,
    g_depth_tex: wgpu::Texture,
    g_depth_view: wgpu::TextureView,
    ao_tex: wgpu::Texture,
    ao_view: wgpu::TextureView,
    dummy_ao_tex: wgpu::Texture,
    dummy_ao_view: wgpu::TextureView,
    depth_ds_tex: wgpu::Texture,
    depth_ds_view: wgpu::TextureView,

    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    vertex_count: u32,

    last_vertices_ptr: usize,
    last_vertices_len: usize,
    last_view_proj: Mat4,
    last_ssao_enabled: bool,
    last_ssao_radius: f32,
    last_ssao_bias: f32,
    last_ssao_strength: f32,
}

impl MarchingCubesGpu {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Placeholder; will be recreated on first ensure_resources.
        let viewport_size_px = [1, 1];

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh_ssao_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (gbuffer_pipeline, ssao_pipeline, composite_pipeline, mesh_uniform_buffer, mesh_uniform_bind_group, ssao_uniform_buffer, ssao_uniform_bind_group, composite_bind_group, g_color_tex, g_color_view, g_normal_tex, g_normal_view, g_depth_tex, g_depth_view, ao_tex, ao_view, dummy_ao_tex, dummy_ao_view, depth_ds_tex, depth_ds_view) =
            Self::create_pipelines_and_resources(device, target_format, viewport_size_px, &sampler);

        let vertex_capacity = 1;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("marching_cubes_vertex_buffer"),
            size: (std::mem::size_of::<MeshVertex>() * vertex_capacity) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            target_format,
            viewport_size_px,
            gbuffer_pipeline,
            ssao_pipeline,
            composite_pipeline,
            mesh_uniform_buffer,
            mesh_uniform_bind_group,
            ssao_uniform_buffer,
            ssao_uniform_bind_group,
            composite_bind_group,
            sampler,
            g_color_tex,
            g_color_view,
            g_normal_tex,
            g_normal_view,
            g_depth_tex,
            g_depth_view,
            ao_tex,
            ao_view,
            dummy_ao_tex,
            dummy_ao_view,
            depth_ds_tex,
            depth_ds_view,
            vertex_buffer,
            vertex_capacity,
            vertex_count: 0,
            last_vertices_ptr: 0,
            last_vertices_len: 0,
            last_view_proj: Mat4::IDENTITY,
            last_ssao_enabled: true,
            last_ssao_radius: -1.0,
            last_ssao_bias: -1.0,
            last_ssao_strength: -1.0,
        }
    }

    fn ensure_resources(
        &mut self,
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        viewport_size_px: [u32; 2],
    ) {
        let size_changed = self.viewport_size_px != viewport_size_px;
        let format_changed = self.target_format != target_format;
        if !size_changed && !format_changed {
            return;
        }

        self.target_format = target_format;
        self.viewport_size_px = viewport_size_px;

        let (gbuffer_pipeline, ssao_pipeline, composite_pipeline, mesh_uniform_buffer, mesh_uniform_bind_group, ssao_uniform_buffer, ssao_uniform_bind_group, composite_bind_group, g_color_tex, g_color_view, g_normal_tex, g_normal_view, g_depth_tex, g_depth_view, ao_tex, ao_view, dummy_ao_tex, dummy_ao_view, depth_ds_tex, depth_ds_view) =
            Self::create_pipelines_and_resources(device, target_format, viewport_size_px, &self.sampler);

        self.gbuffer_pipeline = gbuffer_pipeline;
        self.ssao_pipeline = ssao_pipeline;
        self.composite_pipeline = composite_pipeline;
        self.mesh_uniform_buffer = mesh_uniform_buffer;
        self.mesh_uniform_bind_group = mesh_uniform_bind_group;
        self.ssao_uniform_buffer = ssao_uniform_buffer;
        self.ssao_uniform_bind_group = ssao_uniform_bind_group;
        self.composite_bind_group = composite_bind_group;
        self.g_color_tex = g_color_tex;
        self.g_color_view = g_color_view;
        self.g_normal_tex = g_normal_tex;
        self.g_normal_view = g_normal_view;
        self.g_depth_tex = g_depth_tex;
        self.g_depth_view = g_depth_view;
        self.ao_tex = ao_tex;
        self.ao_view = ao_view;
        self.dummy_ao_tex = dummy_ao_tex;
        self.dummy_ao_view = dummy_ao_view;
        self.depth_ds_tex = depth_ds_tex;
        self.depth_ds_view = depth_ds_view;
    }

    #[allow(clippy::type_complexity)]
    fn create_pipelines_and_resources(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        viewport_size_px: [u32; 2],
        sampler: &wgpu::Sampler,
    ) -> (
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::Buffer,
        wgpu::BindGroup,
        wgpu::Buffer,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ) {
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("marching_cubes_gbuffer_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/marching_cubes.wgsl"
            ))),
        });

        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/mesh_ssao.wgsl"
            ))),
        });

        let mesh_uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("marching_cubes_mesh_uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let ssao_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_ssao_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("marching_cubes_gbuffer_pipeline_layout"),
            bind_group_layouts: &[&mesh_uniform_bgl],
            push_constant_ranges: &[],
        });

        let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("marching_cubes_gbuffer_pipeline"),
            layout: Some(&mesh_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: "fs_gbuffer",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let ssao_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_ssao_pipeline_layout"),
            bind_group_layouts: &[&ssao_bgl],
            push_constant_ranges: &[],
        });

        let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_ssao_pipeline"),
            layout: Some(&ssao_layout),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: "fs_ssao",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_ssao_composite_pipeline"),
            layout: Some(&ssao_layout),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: "fs_composite",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            // egui's main render pass typically has a depth attachment (Depth24Plus).
            // Pipelines used inside that pass must declare a compatible depth-stencil state.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let uniforms = MeshUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            light_dir_world: [0.4, 0.7, 0.2],
            _pad0: 0.0,
            base_color: [0.85, 0.9, 1.0],
            _pad1: 0.0,
        };
        let mesh_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("marching_cubes_mesh_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mesh_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("marching_cubes_mesh_uniform_bg"),
            layout: &mesh_uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: mesh_uniform_buffer.as_entire_binding(),
            }],
        });

        let ssao_uniforms = SsaoUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            screen_size_px: [viewport_size_px[0] as f32, viewport_size_px[1] as f32],
            radius: 0.08,
            bias: 0.002,
            strength: 1.6,
            _pad0: [0.0, 0.0, 0.0],
        };
        let ssao_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_ssao_uniform_buffer"),
            contents: bytemuck::bytes_of(&ssao_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let extent = wgpu::Extent3d {
            width: viewport_size_px[0].max(1),
            height: viewport_size_px[1].max(1),
            depth_or_array_layers: 1,
        };

        let g_color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_g_color_tex"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let g_color_view = g_color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let g_normal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_g_normal_tex"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let g_normal_view = g_normal_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let g_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_g_depth_tex"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let g_depth_view = g_depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let ao_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_ao_tex"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ao_view = ao_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let dummy_ao_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_dummy_ao_tex"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_ao_view = dummy_ao_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_ds_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mesh_depth_ds_tex"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_ds_view = depth_ds_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let ssao_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_ssao_bg"),
            layout: &ssao_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ssao_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&g_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&g_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&g_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&dummy_ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_ssao_composite_bg"),
            layout: &ssao_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ssao_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&g_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&g_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&g_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        (
            gbuffer_pipeline,
            ssao_pipeline,
            composite_pipeline,
            mesh_uniform_buffer,
            mesh_uniform_bind_group,
            ssao_uniform_buffer,
            ssao_uniform_bind_group,
            composite_bind_group,
            g_color_tex,
            g_color_view,
            g_normal_tex,
            g_normal_view,
            g_depth_tex,
            g_depth_view,
            ao_tex,
            ao_view,
            dummy_ao_tex,
            dummy_ao_view,
            depth_ds_tex,
            depth_ds_view,
        )
    }

    fn encode_gbuffer_and_ssao(&self, encoder: &mut wgpu::CommandEncoder, ssao_enabled: bool) {
        if self.vertex_count == 0 {
            return;
        }

        // G-buffer pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh_gbuffer_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.g_color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.g_normal_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.g_depth_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_ds_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.gbuffer_pipeline);
            pass.set_bind_group(0, &self.mesh_uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..self.vertex_count, 0..1);
        }

        // SSAO pass (optional)
        if ssao_enabled {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh_ssao_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, &self.ssao_uniform_bind_group, &[]);
            pass.draw(0..3, 0..1);
        } else {
            // Ensure AO is neutral when disabled
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh_ssao_disabled_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
    }

    fn update_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &Arc<Vec<MeshVertex>>,
    ) {
        self.vertex_count = vertices.len() as u32;

        if vertices.is_empty() {
            return;
        }

        let ptr = Arc::as_ptr(vertices) as usize;
        if ptr == self.last_vertices_ptr && vertices.len() == self.last_vertices_len {
            return;
        }
        self.last_vertices_ptr = ptr;
        self.last_vertices_len = vertices.len();

        if vertices.len() > self.vertex_capacity {
            self.vertex_capacity = vertices.len().next_power_of_two().max(1);
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("marching_cubes_vertex_buffer"),
                size: (std::mem::size_of::<MeshVertex>() * self.vertex_capacity) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
    }

    fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        ssao_enabled: bool,
        ssao_radius: f32,
        ssao_bias: f32,
        ssao_strength: f32,
    ) {
        if self.last_view_proj == view_proj
            && self.last_ssao_enabled == ssao_enabled
            && (self.last_ssao_radius - ssao_radius).abs() < f32::EPSILON
            && (self.last_ssao_bias - ssao_bias).abs() < f32::EPSILON
            && (self.last_ssao_strength - ssao_strength).abs() < f32::EPSILON
        {
            return;
        }
        self.last_view_proj = view_proj;
        self.last_ssao_enabled = ssao_enabled;
        self.last_ssao_radius = ssao_radius;
        self.last_ssao_bias = ssao_bias;
        self.last_ssao_strength = ssao_strength;

        let uniforms = MeshUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir_world: [0.4, 0.7, 0.2],
            _pad0: 0.0,
            base_color: [0.85, 0.9, 1.0],
            _pad1: 0.0,
        };

        queue.write_buffer(&self.mesh_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let inv_view_proj = view_proj.inverse();
        let ssao_uniforms = SsaoUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            screen_size_px: [self.viewport_size_px[0] as f32, self.viewport_size_px[1] as f32],
            radius: ssao_radius,
            bias: ssao_bias,
            strength: ssao_strength,
            _pad0: [0.0, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.ssao_uniform_buffer,
            0,
            bytemuck::bytes_of(&ssao_uniforms),
        );
    }
}
