//! The gizmo pass: `gizmo.wgsl` drawn over the finished frame.

use bytemuck::{Pod, Zeroable};

use crate::{GizmoPart, ViewGizmo};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GizmoEndUniform {
    disc: [f32; 4],
    color: [f32; 4],
}

/// Uniforms of `gizmo.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GizmoUniforms {
    placement: [f32; 4],
    viewport: [f32; 4],
    ends: [GizmoEndUniform; 6],
}

pub(crate) struct GizmoPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GizmoPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gizmo_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gizmo.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gizmo_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gizmo_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_gizmo"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_gizmo"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo_uniforms"),
            size: std::mem::size_of::<GizmoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gizmo_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            uniform_buffer,
            bind_group,
        }
    }

    /// Draws `gizmo` over `target`, which is `size` pixels.
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gizmo: &ViewGizmo,
        target: &wgpu::TextureView,
        size: (u32, u32),
    ) {
        let ends = gizmo.ends().map(|end| {
            let hovered = gizmo.hovered
                == Some(GizmoPart::End {
                    axis: end.axis,
                    positive: end.positive,
                });
            let [r, g, b] = ViewGizmo::end_color(&end);
            GizmoEndUniform {
                disc: [
                    end.center.x,
                    end.center.y,
                    end.radius,
                    if end.positive {
                        1.0 + end.axis as f32
                    } else {
                        0.0
                    },
                ],
                color: [r, g, b, hovered as u32 as f32],
            }
        });
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&GizmoUniforms {
                placement: [
                    gizmo.center.x,
                    gizmo.center.y,
                    gizmo.radius,
                    gizmo.hovered.is_some() as u32 as f32,
                ],
                viewport: [size.0 as f32, size.1 as f32, 0.0, 0.0],
                ends,
            }),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gizmo_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..4, 0..1);
    }
}
