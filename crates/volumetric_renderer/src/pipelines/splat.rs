//! Gaussian splat pipeline: 3D Gaussians as elliptical footprints (EWA
//! splatting), surfels by intersecting each pixel's ray with the surfel's
//! plane (2DGS), blended back to front over the composited scene.
//!
//! A splat is a retained resident ([`GpuSplat`]): its instance buffer is
//! rewritten in depth order, with the spherical-harmonic colour evaluated
//! for the view, whenever the camera has turned or moved enough for the
//! order to change; between sorts only the camera uniforms are refreshed.
//! Sorting on the CPU keeps the pipeline within WebGL2's limits (no
//! storage buffers, no compute); a scene of six hundred thousand
//! Gaussians re-sorts in a few tens of milliseconds.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::{
    CameraView, QUAD_INDICES, QUAD_VERTICES, QuadVertex, SplatData, SplatStyle, StaticBuffer,
};

/// One primitive as the vertex shader reads it.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuSplatInstance {
    /// World centre and activated opacity.
    pub position: [f32; 4],
    /// First scaled local axis in world coordinates, then an unused lane.
    pub axis_u: [f32; 4],
    /// Second scaled local axis.
    pub axis_v: [f32; 4],
    /// Third scaled local axis; zero for a surfel.
    pub axis_w: [f32; 4],
    /// View-dependent colour (rgb) and an unused lane.
    pub color: [f32; 4],
}

/// Uniforms of the splat shader. WGSL size 160 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SplatUniforms {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub screen_size_px: [f32; 2],
    pub kernel_radius: f32,
    pub opacity_scale: f32,
    /// 1 for surfels (ray–plane intersection), 0 for Gaussians (EWA).
    pub surfels: u32,
    pub _pad: [u32; 3],
}

const _: [(); 160] = [(); std::mem::size_of::<SplatUniforms>()];

/// Turning the view axis by more than this re-sorts.
const RESORT_ANGLE_COS: f32 = 0.999_39; // cos 2°
/// Moving the eye by more than this fraction of the splat's extent re-sorts.
const RESORT_MOVE_FRACTION: f32 = 0.02;

const SH_C0: f32 = 0.282_094_79;
const SH_C1: f32 = 0.488_602_51;
const SH_C2: [f32; 5] = [
    1.092_548_4,
    -1.092_548_4,
    0.315_391_6,
    -1.092_548_4,
    0.546_274_2,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_3,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

/// The colour of one primitive seen along `dir` (unit, from the eye to the
/// centre), from its coefficients laid out as [`SplatData::sh`] describes:
/// the three degree-0 values, then each channel's higher coefficients.
/// Clamped to `[0, 1]`.
pub fn evaluate_sh(degree: u32, coeffs: &[f32], dir: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = dir;
    let rest = ((degree + 1) * (degree + 1)) as usize - 1;
    let mut basis = [0.0f32; 15];
    if degree >= 1 {
        basis[0] = -SH_C1 * y;
        basis[1] = SH_C1 * z;
        basis[2] = -SH_C1 * x;
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        basis[3] = SH_C2[0] * x * y;
        basis[4] = SH_C2[1] * y * z;
        basis[5] = SH_C2[2] * (2.0 * zz - xx - yy);
        basis[6] = SH_C2[3] * x * z;
        basis[7] = SH_C2[4] * (xx - yy);
    }
    if degree >= 3 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        basis[8] = SH_C3[0] * y * (3.0 * xx - yy);
        basis[9] = SH_C3[1] * x * y * z;
        basis[10] = SH_C3[2] * y * (4.0 * zz - xx - yy);
        basis[11] = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
        basis[12] = SH_C3[4] * x * (4.0 * zz - xx - yy);
        basis[13] = SH_C3[5] * z * (xx - yy);
        basis[14] = SH_C3[6] * x * (xx - 3.0 * yy);
    }
    let mut out = [0.0f32; 3];
    for (c, value) in out.iter_mut().enumerate() {
        let mut v = SH_C0 * coeffs[c];
        let channel = &coeffs[3 + c * rest..3 + (c + 1) * rest];
        for k in 0..rest {
            v += basis[k] * channel[k];
        }
        *value = (v + 0.5).clamp(0.0, 1.0);
    }
    out
}

/// The 2D covariance (`xx, xy, yy`, pixels²) of a world Gaussian projected
/// through `view` and `proj` onto a `screen` (x right, y down), before the
/// shader's dilation; `None` behind the camera. The same arithmetic as the
/// vertex shader, kept here so it can be checked against finite
/// differences.
pub fn project_covariance(
    view: Mat4,
    proj: Mat4,
    screen: [f32; 2],
    mean: [f32; 3],
    cov: [f32; 6],
) -> Option<[f32; 3]> {
    let t4 = view * Vec3::from(mean).extend(1.0);
    let clip = proj * t4;
    if clip.w <= 1e-6 {
        return None;
    }
    let w = clip.w;
    let (hw, hh) = (0.5 * screen[0], 0.5 * screen[1]);
    let mut j = [[0.0f32; 3]; 2];
    for k in 0..3 {
        let col = proj.col(k);
        j[0][k] = hw * (col.x * w - clip.x * col.w) / (w * w);
        j[1][k] = -hh * (col.y * w - clip.y * col.w) / (w * w);
    }
    let wr = Mat3::from_mat4(view);
    // m = j * wr, 2 x 3.
    let mut m = [[0.0f32; 3]; 2];
    for r in 0..2 {
        for c in 0..3 {
            m[r][c] = (0..3).map(|k| j[r][k] * wr.col(c)[k]).sum();
        }
    }
    let sigma = Mat3::from_cols(
        Vec3::new(cov[0], cov[1], cov[2]),
        Vec3::new(cov[1], cov[3], cov[4]),
        Vec3::new(cov[2], cov[4], cov[5]),
    );
    let ms = |r: usize| -> Vec3 { sigma * Vec3::new(m[r][0], m[r][1], m[r][2]) };
    let (s0, s1) = (ms(0), ms(1));
    let m0 = Vec3::new(m[0][0], m[0][1], m[0][2]);
    let m1 = Vec3::new(m[1][0], m[1][1], m[1][2]);
    Some([s0.dot(m0), s0.dot(m1), s1.dot(m1)])
}

/// The pixel a world point lands on (x right, y down), for the tests.
fn project_pixel(view: Mat4, proj: Mat4, screen: [f32; 2], p: Vec3) -> Vec2 {
    let clip = proj * view * p.extend(1.0);
    Vec2::new(
        (clip.x / clip.w + 1.0) * 0.5 * screen[0],
        (1.0 - clip.y / clip.w) * 0.5 * screen[1],
    )
}

/// The per-view sort state of a retained splat.
struct SortState {
    /// The view's depth axis and eye at the last sort, `None` before the
    /// first.
    last: Option<(Vec3, Vec3)>,
    keys: Vec<u32>,
    order: Vec<u32>,
    scratch: Vec<u32>,
    staging: Vec<GpuSplatInstance>,
}

/// A splat resident on the GPU, drawn by reference each frame. Created by
/// [`SplatPipeline::create_retained`].
pub struct GpuSplat {
    instance_buffer: wgpu::Buffer,
    count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    data: Arc<SplatData>,
    style: SplatStyle,
    extent: f32,
    sort: Mutex<SortState>,
    /// Primitives dropped at the device's buffer size limit.
    pub dropped: usize,
}

impl GpuSplat {
    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Sorts the primitives back to front for `view` and evaluates their
    /// colours, uploading the result, when the view has turned or moved
    /// enough since the last sort. Returns whether it sorted.
    pub fn sort_for(&self, queue: &wgpu::Queue, view: &CameraView) -> bool {
        if self.count == 0 {
            return false;
        }
        let inverse = view.view.inverse();
        let eye = inverse.w_axis.truncate();
        let axis = Vec3::new(view.view.x_axis.z, view.view.y_axis.z, view.view.z_axis.z)
            .normalize_or_zero();
        let mut sort = self.sort.lock().expect("splat sort state");
        if let Some((last_axis, last_eye)) = sort.last {
            let turned = last_axis.dot(axis).abs() < RESORT_ANGLE_COS;
            let moved = (eye - last_eye).length() > RESORT_MOVE_FRACTION * self.extent.max(1e-3);
            if !turned && !moved {
                return false;
            }
        }
        sort.last = Some((axis, eye));

        let n = self.count as usize;
        let data = &self.data;
        let orthographic = view.projection.w_axis.w == 1.0 && view.projection.z_axis.w == 0.0;
        let row_w = Vec4::new(
            view.projection.x_axis.w,
            view.projection.y_axis.w,
            view.projection.z_axis.w,
            view.projection.w_axis.w,
        );
        let SortState {
            keys,
            order,
            scratch,
            staging,
            ..
        } = &mut *sort;
        keys.clear();
        keys.reserve(n);
        for p in &data.positions[..n] {
            let t = view.view * Vec3::from(*p).extend(1.0);
            let depth = if orthographic { -t.z } else { row_w.dot(t) };
            // Descending depth: far primitives first. Flip the sortable
            // bits so an ascending radix sort yields that.
            keys.push(!sortable_bits(depth));
        }
        radix_sort_indices(keys, order, scratch);

        staging.clear();
        staging.reserve(n);
        let per_point = data.sh_per_point();
        for &i in order.iter() {
            let i = i as usize;
            let p = data.positions[i];
            let dir = (Vec3::from(p) - eye).normalize_or_zero();
            // Trainers fit the SH colour to the photographs' sRGB values and
            // blend those as plain numbers; so does the splat layer, which
            // the composite linearises afterwards.
            let color = evaluate_sh(
                data.sh_degree,
                &data.sh[i * per_point..(i + 1) * per_point],
                dir.to_array(),
            );
            let a = data.axes[i];
            staging.push(GpuSplatInstance {
                position: [p[0], p[1], p[2], data.opacities[i]],
                axis_u: [a[0], a[1], a[2], 0.0],
                axis_v: [a[3], a[4], a[5], 0.0],
                axis_w: [a[6], a[7], a[8], 0.0],
                color: [color[0], color[1], color[2], 1.0],
            });
        }
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(staging));
        true
    }
}

/// Maps an `f32` to a `u32` whose unsigned order is the float's order.
fn sortable_bits(v: f32) -> u32 {
    let bits = v.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Fills `order` with the indices of `keys` in ascending key order (LSD
/// radix sort, four 8-bit passes), using `scratch` as the ping-pong.
fn radix_sort_indices(keys: &[u32], order: &mut Vec<u32>, scratch: &mut Vec<u32>) {
    let n = keys.len();
    order.clear();
    order.extend(0..n as u32);
    scratch.clear();
    scratch.resize(n, 0);
    let mut counts = [0usize; 256];
    for shift in [0u32, 8, 16, 24] {
        counts.fill(0);
        for &i in order.iter() {
            counts[((keys[i as usize] >> shift) & 0xff) as usize] += 1;
        }
        let mut sum = 0;
        for c in counts.iter_mut() {
            let next = sum + *c;
            *c = sum;
            sum = next;
        }
        for &i in order.iter() {
            let bucket = ((keys[i as usize] >> shift) & 0xff) as usize;
            scratch[counts[bucket]] = i;
            counts[bucket] += 1;
        }
        std::mem::swap(order, scratch);
    }
}

/// Lays the splat layer over the scene's target: un-premultiplies,
/// linearises for the sRGB target, blends premultiplied.
pub struct SplatCompositePipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl SplatCompositePipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/splat_composite.wgsl"
            ))),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("splat_composite_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat_composite_bgl"),
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
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_composite_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat_composite_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
        }
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        layer_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat_composite_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(layer_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Pipeline drawing retained splats into the splat layer.
pub struct SplatPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    quad_vertex_buffer: StaticBuffer<QuadVertex>,
    quad_index_buffer: StaticBuffer<u16>,
}

impl SplatPipeline {
    /// `layer_format` is the splat layer's ([`crate::GBuffer::SPLAT_LAYER_FORMAT`]).
    pub fn new(device: &wgpu::Device, layer_format: wgpu::TextureFormat) -> Self {
        let target_format = layer_format;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/splat.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat_uniform_bgl"),
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
            label: Some("splat_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let vertex_buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<QuadVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GpuSplatInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 3,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 32,
                        shader_location: 4,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 48,
                        shader_location: 5,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 64,
                        shader_location: 6,
                    },
                ],
            }),
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat_pipeline"),
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
            // Tested at the centre's depth against the scene, never written:
            // the blend order is the sort's, not the depth buffer's.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let quad_vertex_buffer = StaticBuffer::new(
            device,
            &QUAD_VERTICES,
            wgpu::BufferUsages::VERTEX,
            "splat_quad_vertex_buffer",
        );
        let quad_index_buffer = StaticBuffer::new(
            device,
            &QUAD_INDICES,
            wgpu::BufferUsages::INDEX,
            "splat_quad_index_buffer",
        );
        Self {
            pipeline,
            bind_group_layout,
            quad_vertex_buffer,
            quad_index_buffer,
        }
    }

    /// Uploads a splat as a retained resident with `transform` applied
    /// (to the centres and, through its linear part, the covariances),
    /// clamped to the device's buffer size limit. The instance buffer is
    /// filled at the first [`GpuSplat::sort_for`].
    pub fn create_retained(
        &self,
        device: &wgpu::Device,
        data: Arc<SplatData>,
        transform: Mat4,
        style: &SplatStyle,
    ) -> GpuSplat {
        let data = if transform == Mat4::IDENTITY {
            data
        } else {
            Arc::new(transform_splat(&data, transform))
        };
        let max_instances = (device.limits().max_buffer_size
            / std::mem::size_of::<GpuSplatInstance>() as u64) as usize;
        let count = data.len().min(max_instances);
        let dropped = data.len() - count;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("retained_splat_instance_buffer"),
            size: (count.max(1) * std::mem::size_of::<GpuSplatInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained_splat_uniform_buffer"),
            contents: bytemuck::bytes_of(&SplatUniforms::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("retained_splat_uniform_bg"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let extent = data.extent();
        GpuSplat {
            instance_buffer,
            count: count as u32,
            uniform_buffer,
            bind_group,
            data,
            style: style.clone(),
            extent,
            sort: Mutex::new(SortState {
                last: None,
                keys: Vec::new(),
                order: Vec::new(),
                scratch: Vec::new(),
                staging: Vec::new(),
            }),
            dropped,
        }
    }

    /// Sorts the batch for the view when needed and writes its uniforms.
    pub fn prepare_retained(
        &self,
        queue: &wgpu::Queue,
        batch: &GpuSplat,
        view: &CameraView,
        screen_size_px: [f32; 2],
    ) {
        batch.sort_for(queue, view);
        let uniforms = SplatUniforms {
            view: view.view.to_cols_array_2d(),
            proj: view.projection.to_cols_array_2d(),
            screen_size_px,
            kernel_radius: batch.style.kernel_radius,
            opacity_scale: batch.style.opacity_scale,
            surfels: u32::from(batch.data.surfels),
            _pad: [0; 3],
        };
        queue.write_buffer(&batch.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draws a retained batch.
    pub fn render_retained<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        batch: &'a GpuSplat,
    ) {
        if batch.count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &batch.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.buffer().slice(..));
        render_pass.set_vertex_buffer(1, batch.instance_buffer.slice(..));
        render_pass.set_index_buffer(
            self.quad_index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint16,
        );
        render_pass.draw_indexed(0..6, 0, 0..batch.count);
    }
}

/// A splat moved by `transform`: centres as points, axes by the linear
/// part.
fn transform_splat(data: &SplatData, transform: Mat4) -> SplatData {
    let a = Mat3::from_mat4(transform);
    let positions = data
        .positions
        .iter()
        .map(|p| transform.transform_point3(Vec3::from(*p)).to_array())
        .collect();
    let axes = data
        .axes
        .iter()
        .map(|x| {
            let mut out = [0.0f32; 9];
            for k in 0..3 {
                let v = a * Vec3::new(x[3 * k], x[3 * k + 1], x[3 * k + 2]);
                out[3 * k..3 * k + 3].copy_from_slice(&v.to_array());
            }
            out
        })
        .collect();
    SplatData {
        surfels: data.surfels,
        positions,
        axes,
        opacities: data.opacities.clone(),
        sh_degree: data.sh_degree,
        sh: data.sh.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_sort_orders_keys_ascending_and_descending_by_flip() {
        let values = [3.5f32, -1.0, 0.0, 1e6, -7.25, 2.0];
        let keys: Vec<u32> = values.iter().map(|v| sortable_bits(*v)).collect();
        let (mut order, mut scratch) = (Vec::new(), Vec::new());
        radix_sort_indices(&keys, &mut order, &mut scratch);
        let sorted: Vec<f32> = order.iter().map(|&i| values[i as usize]).collect();
        assert_eq!(sorted, vec![-7.25, -1.0, 0.0, 2.0, 3.5, 1e6]);
        let flipped: Vec<u32> = keys.iter().map(|k| !k).collect();
        radix_sort_indices(&flipped, &mut order, &mut scratch);
        let sorted: Vec<f32> = order.iter().map(|&i| values[i as usize]).collect();
        assert_eq!(sorted, vec![1e6, 3.5, 2.0, 0.0, -1.0, -7.25]);
    }

    #[test]
    fn sh_degree_zero_is_the_constant_and_higher_degrees_vary_with_direction() {
        let coeffs = [1.0 / SH_C0, 0.0, -1.0 / SH_C0];
        assert_eq!(evaluate_sh(0, &coeffs, [0.0, 0.0, 1.0]), [1.0, 0.5, 0.0]);
        // Degree 1 with only the z coefficient of green set: green goes up
        // looking along +z and down along -z.
        let mut coeffs = vec![0.0; 12];
        coeffs[3 + 3 + 1] = 0.5;
        let up = evaluate_sh(1, &coeffs, [0.0, 0.0, 1.0]);
        let down = evaluate_sh(1, &coeffs, [0.0, 0.0, -1.0]);
        assert!(up[1] > 0.5 && down[1] < 0.5, "{up:?} {down:?}");
        assert!((up[1] - 0.5 - SH_C1 * 0.5).abs() < 1e-6);
        // Degree 3 has 48 coefficients and evaluates within range.
        let coeffs = vec![0.3; 48];
        let c = evaluate_sh(3, &coeffs, [0.6, 0.0, 0.8]);
        assert!(c.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    /// The projected covariance matches the pixel footprint of the
    /// Gaussian: for a perspective and a pinhole camera, the 2D covariance
    /// equals J Σ Jᵀ with J from finite differences of the projection.
    #[test]
    fn projected_covariance_matches_finite_differences() {
        let screen = [800.0, 600.0];
        let cases = [
            CameraView::look_at(
                Vec3::new(0.5, 1.0, 2.5),
                Vec3::new(0.0, 0.2, 0.0),
                Vec3::Y,
                0.9,
                800.0 / 600.0,
                0.1,
                10.0,
            ),
            CameraView::pinhole(
                &crate::Pinhole {
                    fx: 900.0,
                    fy: 900.0,
                    cx: 380.0,
                    cy: 310.0,
                    width: 800,
                    height: 600,
                },
                Mat4::from_rotation_translation(
                    glam::Quat::from_rotation_y(0.4),
                    Vec3::new(0.3, 0.1, -1.5),
                ),
                0.1,
                10.0,
            ),
        ];
        let mean = [0.1, 0.25, 0.05];
        // Σ = R diag(s²) Rᵀ for a rotated anisotropic Gaussian.
        let r = Mat3::from_quat(glam::Quat::from_euler(glam::EulerRot::XYZ, 0.3, -0.5, 0.2));
        let s = Mat3::from_diagonal(Vec3::new(0.04, 0.01, 0.002));
        let sigma = r * s * s * r.transpose();
        let cov = [
            sigma.col(0).x,
            sigma.col(1).x,
            sigma.col(2).x,
            sigma.col(1).y,
            sigma.col(2).y,
            sigma.col(2).z,
        ];
        for view in cases {
            let c = project_covariance(view.view, view.projection, screen, mean, cov).unwrap();
            let h = 1e-3;
            let base = project_pixel(view.view, view.projection, screen, Vec3::from(mean));
            let mut j = [[0.0f32; 3]; 2];
            for k in 0..3 {
                let mut p = Vec3::from(mean);
                p[k] += h;
                let d = (project_pixel(view.view, view.projection, screen, p) - base) / h;
                j[0][k] = d.x;
                j[1][k] = d.y;
            }
            let js = |r: usize| sigma * Vec3::new(j[r][0], j[r][1], j[r][2]);
            let j0 = Vec3::new(j[0][0], j[0][1], j[0][2]);
            let j1 = Vec3::new(j[1][0], j[1][1], j[1][2]);
            let expected = [js(0).dot(j0), js(0).dot(j1), js(1).dot(j1)];
            for k in 0..3 {
                assert!(
                    (c[k] - expected[k]).abs() < 0.02 * expected[0].max(expected[2]),
                    "{c:?} vs {expected:?}"
                );
            }
            assert!(c[0] > 1.0, "a 4 cm Gaussian at 2 m covers pixels: {c:?}");
        }
    }

    #[test]
    fn transforming_a_splat_moves_centres_and_scales_covariances() {
        let data = SplatData {
            surfels: false,
            positions: vec![[1.0, 0.0, 0.0]],
            axes: vec![[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]],
            opacities: vec![0.5],
            sh_degree: 0,
            sh: vec![0.0; 3],
        };
        assert_eq!(data.covariance(0), [1.0, 0.0, 0.0, 4.0, 0.0, 9.0]);
        let moved = transform_splat(
            &data,
            Mat4::from_scale_rotation_translation(
                Vec3::splat(2.0),
                glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                Vec3::new(0.0, 0.0, 5.0),
            ),
        );
        let p = moved.positions[0];
        assert!((p[0]).abs() < 1e-6 && (p[1] - 2.0).abs() < 1e-6 && (p[2] - 5.0).abs() < 1e-6);
        let c = moved.covariance(0);
        // x and y swap under the quarter turn, everything scales by 4.
        assert!(
            (c[0] - 16.0).abs() < 1e-4 && (c[3] - 4.0).abs() < 1e-4 && (c[5] - 36.0).abs() < 1e-4
        );
    }
}
