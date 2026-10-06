//! The surfel geometry source: retained surface points drawn as oriented
//! discs into the G-buffer, as `surfel_gbuffer.wgsl` describes.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::{CameraView, GBuffer, MaterialId, ObjectId, SurfelData, SurfelVertex};

/// Uniforms of `surfel_gbuffer.wgsl` shared by every draw of a frame.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    /// The eye with w = 1, or for an orthographic frame the view
    /// direction with w = 0.
    eye: [f32; 4],
}

/// A set of surfels resident on the GPU, in their own coordinates; the
/// transform comes at submission, as for [`super::GpuMesh`].
pub struct GpuSurfels {
    vertex_buffer: wgpu::Buffer,
    count: u32,
    /// Surfels dropped at the device's buffer size limit.
    pub dropped: usize,
    /// The corners of the box around the surfels' discs; `None` when
    /// empty.
    pub bounds: Option<(Vec3, Vec3)>,
}

impl GpuSurfels {
    pub fn new(device: &wgpu::Device, data: &SurfelData) -> Self {
        let max =
            (device.limits().max_buffer_size / std::mem::size_of::<SurfelVertex>() as u64) as usize;
        let kept = data.surfels.len().min(max);
        let surfels = &data.surfels[..kept];
        let bounds = surfels
            .iter()
            .filter(|s| s.position.iter().all(|c| c.is_finite()))
            .fold(None, |bounds: Option<(Vec3, Vec3)>, s| {
                let (p, r) = (Vec3::from(s.position), s.radius.abs());
                Some(bounds.map_or((p - r, p + r), |(min, max)| {
                    (min.min(p - r), max.max(p + r))
                }))
            });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained_surfel_buffer"),
            contents: bytemuck::cast_slice(surfels),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            vertex_buffer,
            count: kept as u32,
            dropped: data.surfels.len() - kept,
            bounds,
        }
    }

    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// One draw's uniforms, padded to the dynamic offset alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DrawUniforms {
    model: [[f32; 4]; 4],
    ids: [u32; 2],
    _pad: [u32; 46],
}

const _: () = assert!(std::mem::size_of::<DrawUniforms>() == 256);

/// Retained surfels submitted for a frame.
pub struct SurfelDraw {
    pub surfels: std::sync::Arc<GpuSurfels>,
    pub transform: Mat4,
    pub object: ObjectId,
    pub material: MaterialId,
}

pub struct SurfelPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    draw_layout: wgpu::BindGroupLayout,
    /// One entry per draw this frame, each at a dynamic offset.
    draws: wgpu::Buffer,
    draws_capacity: usize,
    draw_bind_group: Option<wgpu::BindGroup>,
}

impl SurfelPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surfel_gbuffer_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/surfel_gbuffer.wgsl").into()),
        });
        let uniform_entry = |has_dynamic_offset| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset,
                min_binding_size: None,
            },
            count: None,
        };
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfel_frame_bgl"),
            entries: &[uniform_entry(false)],
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfel_draw_bgl"),
            entries: &[uniform_entry(true)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("surfel_gbuffer_pipeline_layout"),
            bind_group_layouts: &[Some(&frame_layout), Some(&draw_layout)],
            immediate_size: 0,
        });
        let target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("surfel_gbuffer_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SurfelVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: std::mem::offset_of!(SurfelVertex, position) as u64,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: std::mem::offset_of!(SurfelVertex, radius) as u64,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: std::mem::offset_of!(SurfelVertex, normal) as u64,
                            shader_location: 2,
                        },
                    ],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_gbuffer"),
                targets: &[
                    target(GBuffer::ALBEDO_FORMAT),
                    target(GBuffer::NORMAL_FORMAT),
                    target(GBuffer::SURFACE_FORMAT),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GBuffer::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surfel_uniform_buffer"),
            contents: bytemuck::bytes_of(&FrameUniforms {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                eye: [0.0, 0.0, 1.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surfel_frame_bg"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let draws = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surfel_draws"),
            size: std::mem::size_of::<DrawUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            draw_layout,
            draws,
            draws_capacity: 1,
            draw_bind_group: None,
        }
    }

    /// Uploads the frame's camera and one block per draw. Call before the
    /// pass that [`render`](Self::render)s the same draws.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &CameraView,
        draws: &[SurfelDraw],
    ) {
        let camera_to_world = view.view.inverse();
        let orthographic = view.projection.w_axis.w == 1.0;
        let eye = if orthographic {
            (-camera_to_world.z_axis.truncate()).normalize().extend(0.0)
        } else {
            camera_to_world.w_axis.truncate().extend(1.0)
        };
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&FrameUniforms {
                view_proj: view.view_projection().to_cols_array_2d(),
                eye: eye.to_array(),
            }),
        );
        if draws.is_empty() {
            return;
        }
        if draws.len() > self.draws_capacity || self.draw_bind_group.is_none() {
            self.draws_capacity = draws.len().max(self.draws_capacity * 2);
            self.draws = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("surfel_draws"),
                size: (self.draws_capacity * std::mem::size_of::<DrawUniforms>()) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.draw_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surfel_draw_bg"),
                layout: &self.draw_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.draws,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<DrawUniforms>() as u64),
                    }),
                }],
            }));
        }
        let blocks: Vec<DrawUniforms> = draws
            .iter()
            .map(|draw| DrawUniforms {
                model: draw.transform.to_cols_array_2d(),
                ids: [draw.object.0, draw.material.0],
                _pad: [0; 46],
            })
            .collect();
        queue.write_buffer(&self.draws, 0, bytemuck::cast_slice(&blocks));
    }

    /// Draws `draws` into a G-buffer pass, each under the block
    /// [`prepare`](Self::prepare) uploaded for it.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, draws: &'a [SurfelDraw]) {
        let Some(draw_bind_group) = &self.draw_bind_group else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        for (i, draw) in draws.iter().enumerate().take(self.draws_capacity) {
            if draw.surfels.is_empty() {
                continue;
            }
            let offset = (i * std::mem::size_of::<DrawUniforms>()) as u32;
            pass.set_bind_group(1, draw_bind_group, &[offset]);
            pass.set_vertex_buffer(0, draw.surfels.vertex_buffer.slice(..));
            pass.draw(0..6, 0..draw.surfels.count);
        }
    }
}
