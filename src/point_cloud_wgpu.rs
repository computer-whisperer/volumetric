use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt as _;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct QuadVertex {
    pub corner: [f32; 2],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub color: [f32; 3],
    pub _pad1: f32,
}

impl PointInstance {
    pub fn from_point(p: (f32, f32, f32)) -> Self {
        // Simple gradient similar to the old CPU painter.
        let r = (p.0 + 1.0) * 0.5;
        let g = (p.1 + 1.0) * 0.5;
        let b = (p.2 + 1.0) * 0.5;
        Self {
            position: [p.0, p.1, p.2],
            _pad0: 0.0,
            color: [r, g, b],
            _pad1: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub point_size_px: f32,
    pub _pad0: f32,
    pub screen_size_px: [f32; 2],
}

#[derive(Clone)]
pub struct PointCloudDrawData {
    pub points: Arc<Vec<(f32, f32, f32)>>,
    pub camera_pos: (f32, f32, f32),
    pub view_proj: Mat4,
    pub point_size_px: f32,
    pub target_format: wgpu::TextureFormat,
}

pub struct PointCloudCallback {
    pub data: PointCloudDrawData,
}

impl egui_wgpu::CallbackTrait for PointCloudCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let gpu = callback_resources
            .entry::<PointCloudGpu>()
            .or_insert_with(|| PointCloudGpu::new(device, self.data.target_format));

        gpu.ensure_pipeline(device, self.data.target_format);
        gpu.update_uniforms(
            device,
            queue,
            self.data.view_proj,
            self.data.point_size_px * screen_descriptor.pixels_per_point,
            [
                screen_descriptor.size_in_pixels[0] as f32,
                screen_descriptor.size_in_pixels[1] as f32,
            ],
        );
        gpu.update_vertices(device, queue, &self.data.points, self.data.camera_pos);

        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(gpu) = callback_resources.get::<PointCloudGpu>() else {
            return;
        };

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
        render_pass.set_vertex_buffer(0, gpu.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, gpu.instance_buffer.slice(..));
        render_pass.draw(0..4, 0..gpu.instance_count);
    }
}

struct PointCloudGpu {
    target_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    quad_vertex_buffer: wgpu::Buffer,

    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instance_count: u32,

    last_points_ptr: usize,
    last_points_len: usize,

    last_point_size: f32,
    last_view_proj: Mat4,
    last_screen_size_px: [f32; 2],
}

impl PointCloudGpu {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let (pipeline, uniform_buffer, uniform_bind_group) =
            Self::create_pipeline_and_uniforms(device, target_format);

        let quad_vertices: [QuadVertex; 4] = [
            QuadVertex {
                corner: [-1.0, -1.0],
                uv: [0.0, 0.0],
            },
            QuadVertex {
                corner: [1.0, -1.0],
                uv: [1.0, 0.0],
            },
            QuadVertex {
                corner: [-1.0, 1.0],
                uv: [0.0, 1.0],
            },
            QuadVertex {
                corner: [1.0, 1.0],
                uv: [1.0, 1.0],
            },
        ];
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_cloud_quad_vertex_buffer"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_capacity = 1;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point_cloud_instance_buffer"),
            size: (std::mem::size_of::<PointInstance>() * instance_capacity) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            target_format,
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            quad_vertex_buffer,
            instance_buffer,
            instance_capacity,
            instance_count: 0,
            last_points_ptr: 0,
            last_points_len: 0,
            last_point_size: -1.0,
            last_view_proj: Mat4::IDENTITY,
            last_screen_size_px: [0.0, 0.0],
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
            label: Some("point_cloud_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shaders/point_cloud.wgsl"
            ))),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("point_cloud_uniform_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point_cloud_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_cloud_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
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
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<PointInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 16,
                                shader_location: 3,
                            },
                        ],
                    },
                ],
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
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            // When eframe is configured with a depth buffer, the main render pass includes a
            // depth attachment. Pipelines used inside egui paint callbacks must be compatible.
            // Enable depth testing so points are properly occluded by other geometry and each other.
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

        let uniforms = PointUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            point_size_px: 2.0,
            _pad0: 0.0,
            screen_size_px: [1.0, 1.0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point_cloud_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point_cloud_uniform_bg"),
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
        points: &Arc<Vec<(f32, f32, f32)>>,
        _camera_pos: (f32, f32, f32),
    ) {
        self.instance_count = points.len() as u32;

        if points.is_empty() {
            return;
        }

        // If we're rendering opaque points (no blending), we don't need per-frame depth sorting.
        // Avoid re-uploading if the sampled point set is unchanged.
        let points_ptr = Arc::as_ptr(points) as usize;
        if points_ptr == self.last_points_ptr && points.len() == self.last_points_len {
            return;
        }
        self.last_points_ptr = points_ptr;
        self.last_points_len = points.len();

        if points.len() > self.instance_capacity {
            self.instance_capacity = points.len().next_power_of_two().max(1);
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("point_cloud_instance_buffer"),
                size: (std::mem::size_of::<PointInstance>() * self.instance_capacity) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        let mut instances = Vec::with_capacity(points.len());
        instances.extend(points.iter().copied().map(PointInstance::from_point));

        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instances),
        );
    }

    fn update_uniforms(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        point_size_px: f32,
        screen_size_px: [f32; 2],
    ) {
        if self.last_view_proj == view_proj
            && (self.last_point_size - point_size_px).abs() < f32::EPSILON
            && self.last_screen_size_px == screen_size_px
        {
            return;
        }

        self.last_view_proj = view_proj;
        self.last_point_size = point_size_px;
        self.last_screen_size_px = screen_size_px;

        let uniforms = PointUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            point_size_px,
            _pad0: 0.0,
            screen_size_px,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

pub fn view_proj_from_camera(
    camera_pos: (f32, f32, f32),
    target: (f32, f32, f32),
    fov_y_radians: f32,
    aspect: f32,
    z_near: f32,
    z_far: f32,
) -> Mat4 {
    let eye = Vec3::new(camera_pos.0, camera_pos.1, camera_pos.2);
    let center = Vec3::new(target.0, target.1, target.2);
    let up = Vec3::Y;

    let view = Mat4::look_at_rh(eye, center, up);
    let proj = Mat4::perspective_rh(fov_y_radians, aspect.max(0.001), z_near, z_far);
    proj * view
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn perspective_maps_near_far_to_wgpu_ndc_z_range() {
        // In view-space for a RH camera looking down -Z, points in front have negative Z.
        let z_near = 0.1;
        let z_far = 100.0;

        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 16.0 / 9.0, z_near, z_far);
        let m = proj;

        let clip_near = m * Vec4::new(0.0, 0.0, -z_near, 1.0);
        let ndc_near = clip_near / clip_near.w;
        assert!((ndc_near.z - 0.0).abs() < 1.0e-3, "ndc_near.z = {}", ndc_near.z);

        let clip_far = m * Vec4::new(0.0, 0.0, -z_far, 1.0);
        let ndc_far = clip_far / clip_far.w;
        assert!((ndc_far.z - 1.0).abs() < 1.0e-3, "ndc_far.z = {}", ndc_far.z);
    }
}
