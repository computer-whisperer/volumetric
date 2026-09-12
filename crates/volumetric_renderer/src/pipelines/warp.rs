//! The lens warp: a finished pinhole frame resampled into the frame a
//! real lens would have made.
//!
//! The scene is rendered through an ideal pinhole into an internal frame
//! (the source, usually a little larger than the output so the bent
//! corners are covered), and a final full-screen pass writes each output
//! pixel by sampling the source where that pixel's ray lands. Where it
//! lands is a coarse grid of source positions computed by whoever knows
//! the lens; this pipeline knows no lens model and takes the grid as is.

use bytemuck::{Pod, Zeroable};

/// Where each output pixel samples the pinhole frame.
///
/// `grid` holds `columns` x `rows` source positions, row-major, for the
/// output positions `(i * output.0 / (columns - 1), j * output.1 / (rows
/// - 1))`, so the grid spans the output frame edge to edge and the pass
/// interpolates between the samples bilinearly. A source position is in
/// source pixels, pixel `i` centred at `i + 0.5`. A negative position
/// marks an output pixel with no image in the source (a fisheye's rim,
/// say); it takes the background.
#[derive(Clone, Debug, PartialEq)]
pub struct Warp {
    /// Size of the pinhole frame rendered.
    pub source: (u32, u32),
    /// Size of the warped frame written.
    pub output: (u32, u32),
    pub columns: u32,
    pub rows: u32,
    pub grid: Vec<[f32; 2]>,
}

impl Warp {
    /// The identity: source and output the same size, every pixel its own.
    pub fn identity(width: u32, height: u32) -> Self {
        let (columns, rows) = (2, 2);
        let grid = (0..rows)
            .flat_map(|j| {
                (0..columns).map(move |i| {
                    [
                        i as f32 * width as f32 / (columns - 1) as f32,
                        j as f32 * height as f32 / (rows - 1) as f32,
                    ]
                })
            })
            .collect();
        Self {
            source: (width, height),
            output: (width, height),
            columns,
            rows,
            grid,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.source.0 == 0 || self.source.1 == 0 || self.output.0 == 0 || self.output.1 == 0 {
            return Err("warp frames must be non-empty".to_string());
        }
        if self.columns < 2 || self.rows < 2 {
            return Err("a warp grid needs at least two columns and rows".to_string());
        }
        if self.grid.len() != (self.columns * self.rows) as usize {
            return Err(format!(
                "warp grid holds {} samples for {} x {}",
                self.grid.len(),
                self.columns,
                self.rows
            ));
        }
        if self.grid.iter().flatten().any(|v| !v.is_finite()) {
            return Err("warp grid samples must be finite".to_string());
        }
        Ok(())
    }

    /// The source position an output position samples, interpolated as
    /// the pass does it; `None` where the output has no image.
    pub fn lookup(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let gx = (x / self.output.0 as f32 * (self.columns - 1) as f32)
            .clamp(0.0, (self.columns - 1) as f32);
        let gy =
            (y / self.output.1 as f32 * (self.rows - 1) as f32).clamp(0.0, (self.rows - 1) as f32);
        let (i0, j0) = (gx.floor() as u32, gy.floor() as u32);
        let (i1, j1) = ((i0 + 1).min(self.columns - 1), (j0 + 1).min(self.rows - 1));
        let (fx, fy) = (gx - i0 as f32, gy - j0 as f32);
        let at = |i: u32, j: u32| self.grid[(j * self.columns + i) as usize];
        let taps = [at(i0, j0), at(i1, j0), at(i0, j1), at(i1, j1)];
        if taps.iter().any(|t| t[0] < 0.0 || t[1] < 0.0) {
            return None;
        }
        let lerp =
            |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        Some(lerp(
            lerp(taps[0], taps[1], fx),
            lerp(taps[2], taps[3], fx),
            fy,
        ))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WarpUniforms {
    output_size: [f32; 2],
    grid_cells: [f32; 2],
    source_size: [f32; 2],
    _pad: [f32; 2],
    background: [f32; 4],
}

/// A warp on the GPU: its grid, the internal frame the scene renders
/// into, and the bind group tying them to the pass.
pub struct GpuWarp {
    pub source: (u32, u32),
    pub output: (u32, u32),
    /// Grid cells across and down: columns - 1, rows - 1.
    grid_cells: [f32; 2],
    /// The pinhole frame the scene renders into.
    pub frame_view: wgpu::TextureView,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// The full-screen pass that writes the warped frame.
pub struct WarpPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    target_format: wgpu::TextureFormat,
}

impl WarpPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/warp.wgsl"
            ))),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("warp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warp_bgl"),
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
                // The grid: 32-bit positions, read with textureLoad since
                // 32-bit floats filter nowhere portable and 16-bit ones
                // lose pixels on a 6000-pixel still.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("warp_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("warp_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_warp"),
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
        Self {
            pipeline,
            bind_group_layout,
            sampler,
            target_format,
        }
    }

    /// Upload a warp: its grid as a texture, and a fresh internal frame
    /// at the source size in the target's format.
    pub fn create(&self, device: &wgpu::Device, queue: &wgpu::Queue, warp: &Warp) -> GpuWarp {
        let frame = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("warp_source_frame"),
            size: wgpu::Extent3d {
                width: warp.source.0,
                height: warp.source.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let frame_view = frame.create_view(&Default::default());
        let grid = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("warp_grid"),
            size: wgpu::Extent3d {
                width: warp.columns,
                height: warp.rows,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &grid,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&warp.grid),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(warp.columns * 8),
                rows_per_image: Some(warp.rows),
            },
            wgpu::Extent3d {
                width: warp.columns,
                height: warp.rows,
                depth_or_array_layers: 1,
            },
        );
        let grid_view = grid.create_view(&Default::default());
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warp_uniforms"),
            size: std::mem::size_of::<WarpUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("warp_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&frame_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&grid_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        GpuWarp {
            source: warp.source,
            output: warp.output,
            grid_cells: [(warp.columns - 1) as f32, (warp.rows - 1) as f32],
            frame_view,
            uniforms,
            bind_group,
        }
    }

    /// Write the warped frame into `target` (the output size), the
    /// background where the source has no image.
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        warp: &GpuWarp,
        background: [f32; 4],
        target: &wgpu::TextureView,
    ) {
        let uniforms = WarpUniforms {
            output_size: [warp.output.0 as f32, warp.output.1 as f32],
            grid_cells: warp.grid_cells,
            source_size: [warp.source.0 as f32, warp.source.1 as f32],
            _pad: [0.0; 2],
            background,
        };
        queue.write_buffer(&warp.uniforms, 0, bytemuck::bytes_of(&uniforms));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("warp_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: background[0] as f64,
                        g: background[1] as f64,
                        b: background[2] as f64,
                        a: background[3] as f64,
                    }),
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
        pass.set_bind_group(0, &warp.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
