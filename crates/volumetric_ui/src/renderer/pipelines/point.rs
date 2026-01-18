//! Point rendering pipeline with instanced quad expansion.
//!
//! Renders points as screen-aligned quads with anti-aliased shapes.
//! Supports circle, square, and diamond shapes, with alpha blending
//! and both screen-space and world-space sizing.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::renderer::{
    DepthMode, DynamicBuffer, PointInstance, PointStyle, QuadVertex, StaticBuffer, WidthMode,
    QUAD_INDICES, QUAD_VERTICES,
};

/// GPU-compatible point instance (with padding for alignment).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuPointInstance {
    pub position: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
}

impl From<&PointInstance> for GpuPointInstance {
    fn from(p: &PointInstance) -> Self {
        Self {
            position: p.position,
            _pad: 0.0,
            color: p.color,
        }
    }
}

/// Uniform data for point rendering.
///
/// WGSL alignment: vec3<f32> has 16-byte alignment, causing implicit padding
/// before _pad in the shader. Total WGSL struct size is 112 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PointUniforms {
    pub view_proj: [[f32; 4]; 4],  // 64 bytes, offset 0
    pub screen_size_px: [f32; 2],  // 8 bytes, offset 64
    pub point_size_px: f32,        // 4 bytes, offset 72
    pub size_mode: u32,            // 4 bytes, offset 76
    pub shape: u32,                // 4 bytes, offset 80
    pub _pad: [f32; 7],            // 28 bytes to reach 112 (matches WGSL vec3 alignment + final pad)
}

// Verify struct size matches WGSL expectations (112 bytes due to vec3 alignment)
const _: [(); 112] = [(); std::mem::size_of::<PointUniforms>()];

impl Default for PointUniforms {
    fn default() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            screen_size_px: [1.0, 1.0],
            point_size_px: 4.0,
            size_mode: 0, // Screen space
            shape: 0,     // Circle
            _pad: [0.0; 7],
        }
    }
}

impl PartialEq for PointUniforms {
    fn eq(&self, other: &Self) -> bool {
        self.view_proj == other.view_proj
            && self.screen_size_px == other.screen_size_px
            && (self.point_size_px - other.point_size_px).abs() < f32::EPSILON
            && self.size_mode == other.size_mode
            && self.shape == other.shape
    }
}

/// Pipeline for rendering points as screen-aligned quads.
pub struct PointPipeline {
    /// Pipeline for depth-tested points
    depth_pipeline: wgpu::RenderPipeline,
    /// Pipeline for overlay points (no depth test)
    overlay_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Static quad vertex buffer
    quad_vertex_buffer: StaticBuffer<QuadVertex>,
    /// Static quad index buffer
    quad_index_buffer: StaticBuffer<u16>,
    /// Dynamic instance buffer for points
    instance_buffer: DynamicBuffer<GpuPointInstance>,
    cached_uniforms: Option<PointUniforms>,
}

impl PointPipeline {
    /// Create a new point pipeline.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/point.wgsl"
            ))),
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point_uniform_bgl"),
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

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex buffer layouts
        let vertex_buffers = [
            // Quad vertex (per-vertex)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<QuadVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0, // corner
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 8,
                        shader_location: 1, // uv
                    },
                ],
            },
            // Point instance (per-instance)
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GpuPointInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 2, // position
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16, // After position (12) + padding (4)
                        shader_location: 3, // color
                    },
                ],
            },
        ];

        // Depth-tested pipeline
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_depth_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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

        // Overlay pipeline (no depth test)
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_overlay_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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

        // Uniform buffer
        let uniforms = PointUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point_uniform_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Static quad buffers
        let quad_vertex_buffer = StaticBuffer::new(
            device,
            &QUAD_VERTICES,
            wgpu::BufferUsages::VERTEX,
            "point_quad_vertex_buffer",
        );
        let quad_index_buffer = StaticBuffer::new(
            device,
            &QUAD_INDICES,
            wgpu::BufferUsages::INDEX,
            "point_quad_index_buffer",
        );

        // Dynamic instance buffer
        let instance_buffer = DynamicBuffer::new(
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "point_instance_buffer",
        );

        Self {
            depth_pipeline,
            overlay_pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group,
            quad_vertex_buffer,
            quad_index_buffer,
            instance_buffer,
            cached_uniforms: None,
        }
    }

    /// Update uniforms if they have changed.
    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, uniforms: &PointUniforms) {
        if self.cached_uniforms.as_ref() == Some(uniforms) {
            return;
        }
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        self.cached_uniforms = Some(*uniforms);
    }

    /// Prepare GPU instances from point data.
    pub fn prepare_instances(points: &[PointInstance]) -> Vec<GpuPointInstance> {
        points.iter().map(GpuPointInstance::from).collect()
    }

    /// Upload point instances to the GPU.
    pub fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[GpuPointInstance],
    ) {
        self.instance_buffer.upload(device, queue, instances);
    }

    /// Create uniforms from view-proj matrix, screen size, and style.
    pub fn create_uniforms(
        view_proj: [[f32; 4]; 4],
        screen_size_px: [f32; 2],
        style: &PointStyle,
    ) -> PointUniforms {
        PointUniforms {
            view_proj,
            screen_size_px,
            point_size_px: style.size,
            size_mode: match style.size_mode {
                WidthMode::ScreenSpace => 0,
                WidthMode::WorldSpace => 1,
            },
            shape: style.shape.to_shader_value(),
            _pad: [0.0; 7],
        }
    }

    /// Record a point render pass.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        depth_mode: DepthMode,
    ) {
        if self.instance_buffer.is_empty() {
            return;
        }

        let pipeline = match depth_mode {
            DepthMode::Normal => &self.depth_pipeline,
            DepthMode::Overlay => &self.overlay_pipeline,
        };

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.buffer().slice(..));
        if let Some(instance_buffer) = self.instance_buffer.buffer() {
            render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        }
        render_pass.set_index_buffer(
            self.quad_index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint16,
        );

        // Draw: 6 indices per quad, N instances
        render_pass.draw_indexed(0..6, 0, 0..self.instance_buffer.len() as u32);
    }

    /// Get the number of point instances currently uploaded.
    pub fn instance_count(&self) -> usize {
        self.instance_buffer.len()
    }
}
