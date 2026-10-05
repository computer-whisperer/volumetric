//! Full-screen passes over the G-buffer: one triangle, one uniform block,
//! and a list of textures addressed by pixel.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// A full-screen pass: `fullscreen_vs.wgsl` followed by a fragment shader
/// whose group 0 is a uniform block at binding 0 and textures from
/// binding 1 on.
pub(crate) struct FullscreenPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    label: &'static str,
}

impl FullscreenPass {
    /// `fragment_source` is appended to the shared vertex shader;
    /// `textures` gives each texture binding's sample type, in binding
    /// order.
    pub fn new<U: Pod>(
        device: &wgpu::Device,
        label: &'static str,
        fragment_source: &str,
        fragment_entry: &str,
        textures: &[wgpu::TextureSampleType],
        target_format: wgpu::TextureFormat,
        uniforms: &U,
    ) -> Self {
        let source = format!(
            "{}{fragment_source}",
            include_str!("../shaders/fullscreen_vs.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        entries.extend(textures.iter().enumerate().map(|(i, sample_type)| {
            wgpu::BindGroupLayoutEntry {
                binding: i as u32 + 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: *sample_type,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }
        }));
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            label,
        }
    }

    /// Binds the pass's textures, in binding order. Rebuild whenever a
    /// view is recreated.
    pub fn bind(&self, device: &wgpu::Device, views: &[&wgpu::TextureView]) -> wgpu::BindGroup {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: self.uniform_buffer.as_entire_binding(),
        }];
        entries.extend(
            views
                .iter()
                .enumerate()
                .map(|(i, view)| wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: wgpu::BindingResource::TextureView(view),
                }),
        );
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.label),
            layout: &self.bind_group_layout,
            entries: &entries,
        })
    }

    pub fn write_uniforms<U: Pod>(&self, queue: &wgpu::Queue, uniforms: &U) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Draws the pass into `target`, which is cleared to `clear` first.
    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        clear: wgpu::Color,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(self.label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
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
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Uniforms of the ambient occlusion pass (`ao.wgsl`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct AoUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub radius: f32,
    pub bias: f32,
    pub strength: f32,
    pub _pad0: f32,
}

/// Uniforms of the resolve pass (`resolve.wgsl`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct ResolveUniforms {
    pub light_dir_world: [f32; 3],
    pub ao_enabled: u32,
    pub base_tint: [f32; 3],
    pub _pad0: f32,
}

/// Uniforms of the pick pass (`pick.wgsl`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(crate) struct PickUniforms {
    pub pixel: [i32; 2],
    pub _pad0: [i32; 2],
}

const FLOAT: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: false };

pub(crate) fn ao_pass(device: &wgpu::Device) -> FullscreenPass {
    FullscreenPass::new(
        device,
        "ao_pass",
        include_str!("../shaders/ao.wgsl"),
        "fs_ao",
        &[FLOAT, wgpu::TextureSampleType::Uint],
        crate::GBuffer::AO_FORMAT,
        &AoUniforms::zeroed(),
    )
}

pub(crate) fn resolve_pass(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> FullscreenPass {
    FullscreenPass::new(
        device,
        "resolve_pass",
        include_str!("../shaders/resolve.wgsl"),
        "fs_resolve",
        &[FLOAT, FLOAT, wgpu::TextureSampleType::Uint, FLOAT],
        target_format,
        &ResolveUniforms::zeroed(),
    )
}

pub(crate) fn pick_pass(device: &wgpu::Device) -> FullscreenPass {
    FullscreenPass::new(
        device,
        "pick_pass",
        include_str!("../shaders/pick.wgsl"),
        "fs_pick",
        &[wgpu::TextureSampleType::Uint],
        crate::pick::PICK_FORMAT,
        &PickUniforms::zeroed(),
    )
}
