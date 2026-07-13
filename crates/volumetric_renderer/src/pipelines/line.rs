//! Line rendering pipeline with vertex shader quad expansion.
//!
//! Renders line segments as screen-aligned quads with anti-aliased edges.
//! Supports both screen-space and world-space line widths, as well as
//! dash patterns.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{
    DepthMode, DynamicBuffer, LineInstance, LinePattern, LineSegment, LineStyle, QUAD_INDICES,
    QUAD_VERTICES, QuadVertex, StaticBuffer, WidthMode,
};

/// Uniform data for line rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub screen_size: [f32; 2],
    pub width_mode: u32,
    pub default_width: f32,
    pub dash_length: f32,
    pub gap_length: f32,
    pub _pad: [f32; 2],
}

impl Default for LineUniforms {
    fn default() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            screen_size: [1.0, 1.0],
            width_mode: 0, // Screen space
            default_width: 2.0,
            dash_length: 0.0,
            gap_length: 0.0,
            _pad: [0.0; 2],
        }
    }
}

impl PartialEq for LineUniforms {
    fn eq(&self, other: &Self) -> bool {
        self.view_proj == other.view_proj
            && self.screen_size == other.screen_size
            && self.width_mode == other.width_mode
            && (self.default_width - other.default_width).abs() < f32::EPSILON
            && (self.dash_length - other.dash_length).abs() < f32::EPSILON
            && (self.gap_length - other.gap_length).abs() < f32::EPSILON
    }
}

/// A line batch resident on the GPU: world-space instances uploaded once
/// (at build time), drawn by reference each frame. Owns its uniform buffer
/// and bind group because line uniforms carry per-batch style alongside the
/// per-frame camera; the renderer refreshes them each frame with a small
/// write. Created by [`LinePipeline::create_retained`].
pub struct GpuLines {
    instance_buffer: wgpu::Buffer,
    count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub(crate) style: LineStyle,
    /// Instances dropped at the device's buffer size limit.
    pub dropped: usize,
}

impl GpuLines {
    /// The batch's depth mode (which render pass draws it).
    pub fn depth_mode(&self) -> DepthMode {
        self.style.depth_mode
    }
}

/// Immediate-mode GPU state for one depth mode's line pass. Each pass needs
/// its own buffers: `queue.write_buffer` executes at submit, before any
/// encoded pass runs, so uploads for a later pass into shared buffers would
/// clobber an earlier pass's data.
struct ImmediateLines {
    instance_buffer: DynamicBuffer<LineInstance>,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cached_uniforms: Option<LineUniforms>,
}

/// Pipeline for rendering lines with vertex shader quad expansion.
pub struct LinePipeline {
    /// Pipeline for depth-tested lines
    depth_pipeline: wgpu::RenderPipeline,
    /// Pipeline for overlay lines (no depth test)
    overlay_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Static quad vertex buffer (4 vertices)
    quad_vertex_buffer: StaticBuffer<QuadVertex>,
    /// Static quad index buffer (6 indices for 2 triangles)
    quad_index_buffer: StaticBuffer<u16>,
    /// Per-depth-mode immediate state, indexed by [`slot_index`].
    immediate: [ImmediateLines; 2],
}

/// The [`LinePipeline::immediate`] slot for a depth mode.
fn slot_index(depth_mode: DepthMode) -> usize {
    match depth_mode {
        DepthMode::Normal => 0,
        DepthMode::Overlay => 1,
    }
}

impl LinePipeline {
    /// Create a new line pipeline.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/line.wgsl"
            ))),
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("line_uniform_bgl"),
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
            label: Some("line_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex buffer layouts
        let vertex_buffers = [
            // Quad vertex (per-vertex)
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<QuadVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0, // corner
                }],
            }),
            // Line instance (per-instance)
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 1, // start
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 2, // end
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 24,
                        shader_location: 3, // color
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32,
                        offset: 40,
                        shader_location: 4, // width
                    },
                ],
            }),
        ];

        // Depth-tested pipeline
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_depth_pipeline"),
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
                cull_mode: None, // Lines are double-sided
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
            label: Some("line_overlay_pipeline"),
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
                depth_compare: Some(wgpu::CompareFunction::Always), // No depth test
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
                label: Some("line_uniform_buffer"),
                contents: bytemuck::bytes_of(&LineUniforms::default()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("line_uniform_bg"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            ImmediateLines {
                instance_buffer: DynamicBuffer::new(
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    "line_instance_buffer",
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
            "line_quad_vertex_buffer",
        );
        let quad_index_buffer = StaticBuffer::new(
            device,
            &QUAD_INDICES,
            wgpu::BufferUsages::INDEX,
            "line_quad_index_buffer",
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
        uniforms: &LineUniforms,
        depth_mode: DepthMode,
    ) {
        let slot = &mut self.immediate[slot_index(depth_mode)];
        if slot.cached_uniforms.as_ref() == Some(uniforms) {
            return;
        }
        queue.write_buffer(&slot.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        slot.cached_uniforms = Some(*uniforms);
    }

    /// Prepare line instances from segments with style.
    pub fn prepare_instances(segments: &[LineSegment], style: &LineStyle) -> Vec<LineInstance> {
        segments
            .iter()
            .map(|seg| LineInstance::from_segment(seg, style.width))
            .collect()
    }

    /// Upload line instances for a depth mode's pass to the GPU.
    pub fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[LineInstance],
        depth_mode: DepthMode,
    ) {
        self.immediate[slot_index(depth_mode)]
            .instance_buffer
            .upload(device, queue, instances);
    }

    /// Create uniforms from view-proj matrix, screen size, and style.
    pub fn create_uniforms(
        view_proj: [[f32; 4]; 4],
        screen_size: [f32; 2],
        style: &LineStyle,
    ) -> LineUniforms {
        let (dash_length, gap_length) = match style.pattern {
            LinePattern::Solid => (0.0, 0.0),
            LinePattern::Dashed {
                dash_length,
                gap_length,
            } => (dash_length, gap_length),
            LinePattern::Dotted { spacing } => (0.1, spacing),
        };

        LineUniforms {
            view_proj,
            screen_size,
            width_mode: match style.width_mode {
                WidthMode::ScreenSpace => 0,
                WidthMode::WorldSpace => 1,
            },
            default_width: style.width,
            dash_length,
            gap_length,
            _pad: [0.0; 2],
        }
    }

    /// Uploads world-space segments as a retained batch, clamped to the
    /// device's buffer size limit.
    pub fn create_retained(
        &self,
        device: &wgpu::Device,
        segments: &[LineSegment],
        transform: glam::Mat4,
        style: &LineStyle,
    ) -> GpuLines {
        let max_instances = (device.limits().max_buffer_size
            / std::mem::size_of::<LineInstance>().max(1) as u64)
            as usize;
        let count = segments.len().min(max_instances);
        let instances: Vec<LineInstance> = segments[..count]
            .iter()
            .map(|seg| {
                let seg = LineSegment {
                    start: transform
                        .transform_point3(glam::Vec3::from(seg.start))
                        .into(),
                    end: transform.transform_point3(glam::Vec3::from(seg.end)).into(),
                    color: seg.color,
                };
                LineInstance::from_segment(&seg, style.width)
            })
            .collect();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained_line_instance_buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained_line_uniform_buffer"),
            contents: bytemuck::bytes_of(&LineUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("retained_line_uniform_bg"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        GpuLines {
            instance_buffer,
            count: count as u32,
            uniform_buffer,
            bind_group,
            style: style.clone(),
            dropped: segments.len() - count,
        }
    }

    /// Refresh a retained batch's uniforms for this frame's camera.
    pub fn write_retained_uniforms(
        &self,
        queue: &wgpu::Queue,
        batch: &GpuLines,
        view_proj: [[f32; 4]; 4],
        screen_size: [f32; 2],
    ) {
        let uniforms = Self::create_uniforms(view_proj, screen_size, &batch.style);
        queue.write_buffer(&batch.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw a retained batch (in the pass matching its depth mode).
    pub fn render_retained<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        batch: &'a GpuLines,
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

    /// Record a depth mode's line render pass.
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

    /// Get the number of line instances currently uploaded for a depth mode.
    pub fn instance_count(&self, depth_mode: DepthMode) -> usize {
        self.immediate[slot_index(depth_mode)].instance_buffer.len()
    }
}
