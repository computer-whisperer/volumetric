//! XRay mesh rendering pipeline.
//!
//! Renders semi-transparent meshes with depth testing but no depth writing.
//! Used for X-Ray visualization mode.

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::renderer::{DynamicBuffer, MeshVertex};

/// Uniform data for XRay mesh rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct XRayUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir_world: [f32; 3],
    pub opacity: f32,
    pub base_color: [f32; 3],
    pub _pad0: f32,
}

impl Default for XRayUniforms {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            light_dir_world: [0.4, 0.7, 0.2],
            opacity: 0.3,
            base_color: [0.6, 0.8, 1.0], // Slight blue tint for X-Ray
            _pad0: 0.0,
        }
    }
}

impl PartialEq for XRayUniforms {
    fn eq(&self, other: &Self) -> bool {
        self.view_proj == other.view_proj
            && self.light_dir_world == other.light_dir_world
            && self.opacity == other.opacity
            && self.base_color == other.base_color
    }
}

/// Pipeline for rendering XRay (semi-transparent) meshes.
pub struct XRayPipeline {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: DynamicBuffer<MeshVertex>,
    index_buffer: DynamicBuffer<u32>,
    cached_uniforms: Option<XRayUniforms>,
}

impl XRayPipeline {
    /// Create a new XRay pipeline.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_xray_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/mesh_xray.wgsl"
            ))),
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xray_uniform_bgl"),
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
            label: Some("mesh_xray_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline with transparency
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_xray_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // position
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        // normal
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12, // 3 * sizeof(f32)
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // Premultiplied alpha blending
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling - show both sides
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: false, // Don't write to depth buffer
                depth_compare: wgpu::CompareFunction::LessEqual, // But still test against it
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Uniform buffer
        let uniforms = XRayUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xray_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xray_uniform_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Vertex and index buffers
        let vertex_buffer = DynamicBuffer::new(
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "xray_vertex_buffer",
        );
        let index_buffer = DynamicBuffer::new(
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            "xray_index_buffer",
        );

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group,
            vertex_buffer,
            index_buffer,
            cached_uniforms: None,
        }
    }

    /// Update uniforms if they have changed.
    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, uniforms: &XRayUniforms) {
        if self.cached_uniforms.as_ref() == Some(uniforms) {
            return;
        }
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        self.cached_uniforms = Some(*uniforms);
    }

    /// Upload vertex data.
    pub fn upload_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[MeshVertex],
    ) {
        self.vertex_buffer.upload(device, queue, vertices);
    }

    /// Upload index data.
    pub fn upload_indices(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, indices: &[u32]) {
        self.index_buffer.upload(device, queue, indices);
    }

    /// Record the XRay render pass.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, use_indices: bool) {
        if self.vertex_buffer.is_empty() {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        if let Some(buffer) = self.vertex_buffer.buffer() {
            render_pass.set_vertex_buffer(0, buffer.slice(..));
        }

        if use_indices && !self.index_buffer.is_empty() {
            if let Some(buffer) = self.index_buffer.buffer() {
                render_pass.set_index_buffer(buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.index_buffer.len() as u32, 0, 0..1);
            }
        } else {
            render_pass.draw(0..self.vertex_buffer.len() as u32, 0..1);
        }
    }

    /// Get the number of vertices currently uploaded.
    pub fn vertex_count(&self) -> usize {
        self.vertex_buffer.len()
    }

    /// Get the number of indices currently uploaded.
    pub fn index_count(&self) -> usize {
        self.index_buffer.len()
    }
}
