//! Point rendering pipeline with instanced quad expansion.
//!
//! Renders points as screen-aligned quads with anti-aliased shapes.
//! Supports circle, square, and diamond shapes, with alpha blending
//! and both screen-space and world-space sizing.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{
    DepthMode, DynamicBuffer, PointInstance, PointStyle, QUAD_INDICES, QUAD_VERTICES, QuadVertex,
    StaticBuffer, WidthMode,
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
    pub view_proj: [[f32; 4]; 4], // 64 bytes, offset 0
    pub screen_size_px: [f32; 2], // 8 bytes, offset 64
    pub point_size_px: f32,       // 4 bytes, offset 72
    pub size_mode: u32,           // 4 bytes, offset 76
    pub shape: u32,               // 4 bytes, offset 80
    pub _pad: [f32; 7],           // 28 bytes to reach 112 (matches WGSL vec3 alignment + final pad)
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

/// A point batch resident on the GPU: world-space instances uploaded once
/// (at build time), drawn by reference each frame. Owns its uniform buffer
/// and bind group (per-batch style + per-frame camera); the renderer
/// refreshes the uniforms each frame with a small write. Created by
/// [`PointPipeline::create_retained`].
pub struct GpuPoints {
    instance_buffer: wgpu::Buffer,
    count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub(crate) style: PointStyle,
    /// Instances dropped at the device's buffer size limit.
    pub dropped: usize,
}

impl GpuPoints {
    /// The batch's depth mode (which render pass draws it).
    pub fn depth_mode(&self) -> DepthMode {
        self.style.depth_mode
    }
}

/// Immediate-mode GPU state for one depth mode's point pass. Each pass needs
/// its own buffers: `queue.write_buffer` executes at submit, before any
/// encoded pass runs, so uploads for a later pass into shared buffers would
/// clobber an earlier pass's data.
struct ImmediatePoints {
    instance_buffer: DynamicBuffer<GpuPointInstance>,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cached_uniforms: Option<PointUniforms>,
}

/// Pipeline for rendering points as screen-aligned quads.
pub struct PointPipeline {
    /// Pipeline for depth-tested points
    depth_pipeline: wgpu::RenderPipeline,
    /// Pipeline for overlay points (no depth test)
    overlay_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Static quad vertex buffer
    quad_vertex_buffer: StaticBuffer<QuadVertex>,
    /// Static quad index buffer
    quad_index_buffer: StaticBuffer<u16>,
    /// Per-depth-mode immediate state, indexed by [`slot_index`].
    immediate: [ImmediatePoints; 2],
}

/// The [`PointPipeline::immediate`] slot for a depth mode.
fn slot_index(depth_mode: DepthMode) -> usize {
    match depth_mode {
        DepthMode::Normal => 0,
        DepthMode::Overlay => 1,
    }
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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex buffer layouts
        let vertex_buffers = [
            // Quad vertex (per-vertex)
            Some(wgpu::VertexBufferLayout {
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
            }),
            // Point instance (per-instance)
            Some(wgpu::VertexBufferLayout {
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
                        offset: 16,         // After position (12) + padding (4)
                        shader_location: 3, // color
                    },
                ],
            }),
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
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
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Per-depth-mode immediate buffers
        let immediate = std::array::from_fn(|_| {
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("point_uniform_buffer"),
                contents: bytemuck::bytes_of(&PointUniforms::default()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("point_uniform_bg"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            ImmediatePoints {
                instance_buffer: DynamicBuffer::new(
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    "point_instance_buffer",
                ),
                uniform_buffer,
                bind_group,
                cached_uniforms: None,
            }
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

        Self {
            depth_pipeline,
            overlay_pipeline,
            bind_group_layout,
            quad_vertex_buffer,
            quad_index_buffer,
            immediate,
        }
    }

    /// Update a depth mode's uniforms if they have changed.
    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        uniforms: &PointUniforms,
        depth_mode: DepthMode,
    ) {
        let slot = &mut self.immediate[slot_index(depth_mode)];
        if slot.cached_uniforms.as_ref() == Some(uniforms) {
            return;
        }
        queue.write_buffer(&slot.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        slot.cached_uniforms = Some(*uniforms);
    }

    /// Prepare GPU instances from point data.
    pub fn prepare_instances(points: &[PointInstance]) -> Vec<GpuPointInstance> {
        points.iter().map(GpuPointInstance::from).collect()
    }

    /// Upload point instances for a depth mode's pass to the GPU.
    pub fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[GpuPointInstance],
        depth_mode: DepthMode,
    ) {
        self.immediate[slot_index(depth_mode)]
            .instance_buffer
            .upload(device, queue, instances);
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

    /// Uploads world-space points as a retained batch, clamped to the
    /// device's buffer size limit.
    pub fn create_retained(
        &self,
        device: &wgpu::Device,
        points: &[PointInstance],
        transform: glam::Mat4,
        style: &PointStyle,
    ) -> GpuPoints {
        let max_instances = (device.limits().max_buffer_size
            / std::mem::size_of::<GpuPointInstance>().max(1) as u64)
            as usize;
        let count = points.len().min(max_instances);
        let instances: Vec<GpuPointInstance> = points[..count]
            .iter()
            .map(|p| GpuPointInstance {
                position: transform
                    .transform_point3(glam::Vec3::from(p.position))
                    .into(),
                _pad: 0.0,
                color: p.color,
            })
            .collect();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained_point_instance_buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained_point_uniform_buffer"),
            contents: bytemuck::bytes_of(&PointUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("retained_point_uniform_bg"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        GpuPoints {
            instance_buffer,
            count: count as u32,
            uniform_buffer,
            bind_group,
            style: style.clone(),
            dropped: points.len() - count,
        }
    }

    /// Refresh a retained batch's uniforms for this frame's camera.
    pub fn write_retained_uniforms(
        &self,
        queue: &wgpu::Queue,
        batch: &GpuPoints,
        view_proj: [[f32; 4]; 4],
        screen_size_px: [f32; 2],
    ) {
        let uniforms = Self::create_uniforms(view_proj, screen_size_px, &batch.style);
        queue.write_buffer(&batch.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw a retained batch (in the pass matching its depth mode).
    pub fn render_retained<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        batch: &'a GpuPoints,
    ) {
        if batch.count == 0 {
            return;
        }

        let pipeline = match batch.style.depth_mode {
            DepthMode::Normal => &self.depth_pipeline,
            DepthMode::Overlay => &self.overlay_pipeline,
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &batch.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.buffer().slice(..));
        render_pass.set_vertex_buffer(1, batch.instance_buffer.slice(..));
        render_pass.set_index_buffer(
            self.quad_index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint16,
        );
        render_pass.draw_indexed(0..6, 0, 0..batch.count);
    }

    /// Record a depth mode's point render pass.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, depth_mode: DepthMode) {
        let slot = &self.immediate[slot_index(depth_mode)];
        if slot.instance_buffer.is_empty() {
            return;
        }

        let pipeline = match depth_mode {
            DepthMode::Normal => &self.depth_pipeline,
            DepthMode::Overlay => &self.overlay_pipeline,
        };

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &slot.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.buffer().slice(..));
        if let Some(instance_buffer) = slot.instance_buffer.buffer() {
            render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        }
        render_pass.set_index_buffer(
            self.quad_index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint16,
        );

        // Draw: 6 indices per quad, N instances
        render_pass.draw_indexed(0..6, 0, 0..slot.instance_buffer.len() as u32);
    }

    /// Get the number of point instances currently uploaded for a depth mode.
    pub fn instance_count(&self, depth_mode: DepthMode) -> usize {
        self.immediate[slot_index(depth_mode)].instance_buffer.len()
    }
}
