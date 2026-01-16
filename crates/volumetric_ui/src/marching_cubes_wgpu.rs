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
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu = callback_resources
            .entry::<MarchingCubesGpu>()
            .or_insert_with(|| MarchingCubesGpu::new(device, self.data.target_format));

        gpu.ensure_pipeline(device, self.data.target_format);
        gpu.update_uniforms(queue, self.data.view_proj);
        gpu.update_vertices(device, queue, &self.data.vertices);

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

        render_pass.set_pipeline(&gpu.pipeline);
        render_pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
        render_pass.draw(0..gpu.vertex_count, 0..1);
    }
}

struct MarchingCubesGpu {
    target_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    vertex_count: u32,

    last_vertices_ptr: usize,
    last_vertices_len: usize,
    last_view_proj: Mat4,
}

impl MarchingCubesGpu {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let (pipeline, uniform_buffer, uniform_bind_group) =
            Self::create_pipeline_and_uniforms(device, target_format);

        let vertex_capacity = 1;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("marching_cubes_vertex_buffer"),
            size: (std::mem::size_of::<MeshVertex>() * vertex_capacity) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            target_format,
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            vertex_buffer,
            vertex_capacity,
            vertex_count: 0,
            last_vertices_ptr: 0,
            last_vertices_len: 0,
            last_view_proj: Mat4::IDENTITY,
        }
    }

    fn ensure_pipeline(&mut self, device: &wgpu::Device, target_format: wgpu::TextureFormat) {
        if self.target_format == target_format {
            return;
        }

        self.target_format = target_format;
        let (pipeline, uniform_buffer, uniform_bind_group) =
            Self::create_pipeline_and_uniforms(device, target_format);
        self.pipeline = pipeline;
        self.uniform_buffer = uniform_buffer;
        self.uniform_bind_group = uniform_bind_group;
    }

    fn create_pipeline_and_uniforms(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> (wgpu::RenderPipeline, wgpu::Buffer, wgpu::BindGroup) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("marching_cubes_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/marching_cubes.wgsl"
            ))),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("marching_cubes_uniform_bgl"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("marching_cubes_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("marching_cubes_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
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
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            // Use the depth buffer provided by eframe/egui (NativeOptions::depth_buffer).
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

        let uniforms = MeshUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            light_dir_world: [0.4, 0.7, 0.2],
            _pad0: 0.0,
            base_color: [0.85, 0.9, 1.0],
            _pad1: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("marching_cubes_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("marching_cubes_uniform_bg"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        (pipeline, uniform_buffer, uniform_bind_group)
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

    fn update_uniforms(&mut self, queue: &wgpu::Queue, view_proj: Mat4) {
        if self.last_view_proj == view_proj {
            return;
        }
        self.last_view_proj = view_proj;

        let uniforms = MeshUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir_world: [0.4, 0.7, 0.2],
            _pad0: 0.0,
            base_color: [0.85, 0.9, 1.0],
            _pad1: 0.0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}
