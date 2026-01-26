//! Headless wgpu renderer for generating PNG images

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::HashSet;
use std::path::Path;
use wgpu::util::DeviceExt;

/// Vertex format for the mesh (32 bytes total for alignment)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
}

/// Vertex format for grid lines (32 bytes for alignment)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GridVertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub color: [f32; 3],
    pub _pad1: f32,
}

// ============================================================================
// Point Rendering Types
// ============================================================================

/// Quad vertex for instanced point rendering
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct QuadVertex {
    pub corner: [f32; 2], // -1..+1 quad corners
    pub uv: [f32; 2],     // 0..1 texture coords
}

/// Static quad vertices for point rendering (centered at origin)
pub const POINT_QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex { corner: [-1.0, -1.0], uv: [0.0, 0.0] }, // bottom-left
    QuadVertex { corner: [-1.0,  1.0], uv: [0.0, 1.0] }, // top-left
    QuadVertex { corner: [ 1.0, -1.0], uv: [1.0, 0.0] }, // bottom-right
    QuadVertex { corner: [ 1.0,  1.0], uv: [1.0, 1.0] }, // top-right
];

/// Quad indices for two triangles
pub const POINT_QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];

/// A point instance with position and color
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4], // RGBA
}

impl PointInstance {
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, _pad: 0.0, color }
    }
}

/// Uniform buffer for point rendering
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PointUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub screen_size: [f32; 2],
    pub point_size: f32,
    pub shape: u32, // 0 = circle, 1 = square, 2 = diamond
}

impl Default for PointUniforms {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            screen_size: [1024.0, 1024.0],
            point_size: 8.0,
            shape: 0, // circle
        }
    }
}

/// Point shape options
#[derive(Clone, Copy, Debug, Default)]
pub enum PointShape {
    #[default]
    Circle,
    Square,
    Diamond,
}

impl PointShape {
    pub fn to_shader_value(self) -> u32 {
        match self {
            PointShape::Circle => 0,
            PointShape::Square => 1,
            PointShape::Diamond => 2,
        }
    }
}

/// Uniform buffer layout (must match shader)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub light_dir: [f32; 3],
    pub _pad1: f32,
    pub base_color: [f32; 3],
    pub rim_strength: f32,
    pub sky_color: [f32; 3],
    pub fog_density: f32,
    pub ground_color: [f32; 3],
    pub fog_start: f32,
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 5.0],
            _pad0: 0.0,
            light_dir: Vec3::new(0.5, 0.7, 0.5).normalize().to_array(),
            _pad1: 0.0,
            base_color: [0.4, 0.6, 0.8],
            rim_strength: 0.5,
            sky_color: [0.9, 0.92, 0.95],
            fog_density: 0.02,
            ground_color: [0.3, 0.35, 0.4],
            fog_start: 1.0,
        }
    }
}

/// Options for wireframe rendering
#[derive(Debug, Clone)]
pub struct WireframeOptions {
    pub color: [f32; 3],
}

/// Extract unique edges from triangle indices and convert to line vertices
fn extract_wireframe_edges(
    vertices: &[MeshVertex],
    indices: &[u32],
    color: [f32; 3],
) -> Vec<GridVertex> {
    let mut edges: HashSet<(u32, u32)> = HashSet::new();

    // Collect unique edges (sorted indices to avoid duplicates)
    for tri in indices.chunks(3) {
        if tri.len() != 3 {
            continue;
        }
        let tri_indices = [tri[0], tri[1], tri[2]];

        // Add edges (i0-i1, i1-i2, i2-i0)
        for i in 0..3 {
            let a = tri_indices[i];
            let b = tri_indices[(i + 1) % 3];
            let edge = if a < b { (a, b) } else { (b, a) };
            edges.insert(edge);
        }
    }

    // Convert edges to line vertices
    let mut line_vertices = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        let va = &vertices[a as usize];
        let vb = &vertices[b as usize];

        line_vertices.push(GridVertex {
            position: va.position,
            _pad0: 0.0,
            color,
            _pad1: 0.0,
        });
        line_vertices.push(GridVertex {
            position: vb.position,
            _pad0: 0.0,
            color,
            _pad1: 0.0,
        });
    }

    line_vertices
}

/// Headless wgpu renderer
pub struct HeadlessRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    mesh_pipeline: wgpu::RenderPipeline,
    grid_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    point_bind_group_layout: wgpu::BindGroupLayout,
    /// Static quad vertex buffer for point rendering
    point_quad_vertex_buffer: wgpu::Buffer,
    /// Static quad index buffer for point rendering
    point_quad_index_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl HeadlessRenderer {
    /// Create a new headless renderer
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context("Failed to find a suitable GPU adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Headless Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .context("Failed to create device")?;

        // Load mesh shader
        let mesh_shader_source = include_str!("shaders/headless_mesh.wgsl");
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Headless Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(mesh_shader_source.into()),
        });

        // Load grid shader
        let grid_shader_source = include_str!("shaders/grid.wgsl");
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(grid_shader_source.into()),
        });

        // Bind group layout for uniforms
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
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
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Mesh vertex buffer layout
        let mesh_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 16, // After position + padding
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: "vs_main",
                buffers: &[mesh_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Grid vertex buffer layout (same structure, different semantics)
        let grid_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GridVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grid Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: "vs_main",
                buffers: &[grid_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ====================================================================
        // Point Pipeline
        // ====================================================================

        let point_shader_source = include_str!("shaders/point.wgsl");
        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Shader"),
            source: wgpu::ShaderSource::Wgsl(point_shader_source.into()),
        });

        // Point uniforms have different layout than mesh uniforms
        let point_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Point Uniform Bind Group Layout"),
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

        let point_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Point Pipeline Layout"),
            bind_group_layouts: &[&point_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Quad vertex layout (per-vertex)
        let quad_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0, // corner
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1, // uv
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };

        // Point instance layout (per-instance)
        let point_instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2, // position
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 16, // after position (12) + padding (4)
                    shader_location: 3, // color
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Point Render Pipeline"),
            layout: Some(&point_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &point_shader,
                entry_point: "vs_main",
                buffers: &[quad_vertex_layout, point_instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Points are billboards, no culling
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create static quad buffers for point rendering
        let point_quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&POINT_QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let point_quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Quad Index Buffer"),
            contents: bytemuck::cast_slice(&POINT_QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            device,
            queue,
            mesh_pipeline,
            grid_pipeline,
            point_pipeline,
            bind_group_layout,
            point_bind_group_layout,
            point_quad_vertex_buffer,
            point_quad_index_buffer,
            width,
            height,
        })
    }

    /// Render mesh to an in-memory image with optional overlays.
    pub fn render_to_image(
        &self,
        vertices: &[MeshVertex],
        indices: &[u32],
        uniforms: &Uniforms,
        background_color: [f32; 3],
        grid_vertices: Option<&[GridVertex]>,
        extra_lines: Option<&[GridVertex]>,
        wireframe: Option<&WireframeOptions>,
    ) -> Result<image::RgbaImage> {
        // Create mesh vertex and index buffers
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Create grid vertex buffer if grid is provided
        let grid_buffer = grid_vertices.map(|gv| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Grid Vertex Buffer"),
                    contents: bytemuck::cast_slice(gv),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let grid_vertex_count = grid_vertices.map(|gv| gv.len() as u32).unwrap_or(0);

        // Create wireframe vertices if wireframe mode is enabled
        let wireframe_verts = wireframe.map(|wf| extract_wireframe_edges(vertices, indices, wf.color));
        let wireframe_buffer = wireframe_verts.as_ref().map(|wv| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Wireframe Vertex Buffer"),
                    contents: bytemuck::cast_slice(wv),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let wireframe_vertex_count = wireframe_verts.as_ref().map(|wv| wv.len() as u32).unwrap_or(0);

        let extra_buffer = extra_lines.map(|lines| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Extra Line Vertex Buffer"),
                    contents: bytemuck::cast_slice(lines),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let extra_vertex_count = extra_lines.map(|lines| lines.len() as u32).unwrap_or(0);

        // Create uniform buffer
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create color texture
        let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Output buffer with 256-byte row alignment
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
        let output_buffer_size = (padded_bytes_per_row * self.height) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Record render commands
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color[0] as f64,
                            g: background_color[1] as f64,
                            b: background_color[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw grid first (so mesh renders on top)
            if let Some(ref gb) = grid_buffer {
                render_pass.set_pipeline(&self.grid_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.set_vertex_buffer(0, gb.slice(..));
                render_pass.draw(0..grid_vertex_count, 0..1);
            }

            if let Some(ref wb) = wireframe_buffer {
                // Wireframe mode: draw edges as lines using grid pipeline
                render_pass.set_pipeline(&self.grid_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.set_vertex_buffer(0, wb.slice(..));
                render_pass.draw(0..wireframe_vertex_count, 0..1);
            } else {
                // Normal mode: draw filled mesh
                render_pass.set_pipeline(&self.mesh_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }

            if let Some(ref eb) = extra_buffer {
                render_pass.set_pipeline(&self.grid_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.set_vertex_buffer(0, eb.slice(..));
                render_pass.draw(0..extra_vertex_count, 0..1);
            }
        }

        // Copy texture to buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map buffer and read pixels
        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv()?.context("Failed to map output buffer")?;

        let data = buffer_slice.get_mapped_range();

        // Remove row padding and create image
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);
        for y in 0..self.height {
            let start = (y * padded_bytes_per_row) as usize;
            let end = start + (self.width * bytes_per_pixel) as usize;
            pixels.extend_from_slice(&data[start..end]);
        }

        drop(data);
        output_buffer.unmap();

        // Save as PNG
        let img: image::RgbaImage =
            image::ImageBuffer::from_raw(self.width, self.height, pixels)
                .context("Failed to create image buffer")?;

        Ok(img)
    }

    /// Render mesh to PNG with optional grid overlay and wireframe mode
    pub fn render_to_png(
        &self,
        vertices: &[MeshVertex],
        indices: &[u32],
        uniforms: &Uniforms,
        background_color: [f32; 3],
        grid_vertices: Option<&[GridVertex]>,
        extra_lines: Option<&[GridVertex]>,
        wireframe: Option<&WireframeOptions>,
        output_path: &Path,
    ) -> Result<()> {
        let img = self.render_to_image(
            vertices,
            indices,
            uniforms,
            background_color,
            grid_vertices,
            extra_lines,
            wireframe,
        )?;

        img.save(output_path)
            .context("Failed to save PNG")?;

        Ok(())
    }

    /// Render mesh with points to an in-memory image.
    ///
    /// Points are rendered as screen-space circles that properly depth-test
    /// against mesh geometry.
    pub fn render_with_points_to_image(
        &self,
        vertices: &[MeshVertex],
        indices: &[u32],
        uniforms: &Uniforms,
        background_color: [f32; 3],
        grid_vertices: Option<&[GridVertex]>,
        wireframe: Option<&WireframeOptions>,
        points: Option<&[PointInstance]>,
        point_size: f32,
        point_shape: PointShape,
    ) -> Result<image::RgbaImage> {
        // Create mesh vertex and index buffers
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Grid buffer
        let grid_buffer = grid_vertices.map(|gv| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Grid Vertex Buffer"),
                    contents: bytemuck::cast_slice(gv),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let grid_vertex_count = grid_vertices.map(|gv| gv.len() as u32).unwrap_or(0);

        // Wireframe buffer
        let wireframe_verts = wireframe.map(|wf| extract_wireframe_edges(vertices, indices, wf.color));
        let wireframe_buffer = wireframe_verts.as_ref().map(|wv| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Wireframe Vertex Buffer"),
                    contents: bytemuck::cast_slice(wv),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let wireframe_vertex_count = wireframe_verts.as_ref().map(|wv| wv.len() as u32).unwrap_or(0);

        // Point instance buffer
        let point_buffer = points.filter(|p| !p.is_empty()).map(|pts| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Point Instance Buffer"),
                    contents: bytemuck::cast_slice(pts),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let point_count = points.map(|p| p.len() as u32).unwrap_or(0);

        // Mesh uniform buffer and bind group
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Point uniform buffer and bind group
        let point_uniforms = PointUniforms {
            view_proj: uniforms.view_proj,
            screen_size: [self.width as f32, self.height as f32],
            point_size,
            shape: point_shape.to_shader_value(),
        };

        let point_uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Point Uniform Buffer"),
                contents: bytemuck::bytes_of(&point_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let point_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Point Uniform Bind Group"),
            layout: &self.point_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: point_uniform_buffer.as_entire_binding(),
            }],
        });

        // Create textures
        let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Output buffer
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
        let output_buffer_size = (padded_bytes_per_row * self.height) as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Record render commands
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color[0] as f64,
                            g: background_color[1] as f64,
                            b: background_color[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 1. Draw grid first (behind everything)
            if let Some(ref gb) = grid_buffer {
                render_pass.set_pipeline(&self.grid_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.set_vertex_buffer(0, gb.slice(..));
                render_pass.draw(0..grid_vertex_count, 0..1);
            }

            // 2. Draw mesh (or wireframe)
            if let Some(ref wb) = wireframe_buffer {
                render_pass.set_pipeline(&self.grid_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.set_vertex_buffer(0, wb.slice(..));
                render_pass.draw(0..wireframe_vertex_count, 0..1);
            } else if !indices.is_empty() {
                render_pass.set_pipeline(&self.mesh_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }

            // 3. Draw points (with proper depth testing against mesh)
            if let Some(ref pb) = point_buffer {
                render_pass.set_pipeline(&self.point_pipeline);
                render_pass.set_bind_group(0, &point_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.point_quad_vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, pb.slice(..));
                render_pass.set_index_buffer(
                    self.point_quad_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                // Draw 6 indices per quad, N instances
                render_pass.draw_indexed(0..6, 0, 0..point_count);
            }
        }

        // Copy texture to buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map buffer and read pixels
        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv()?.context("Failed to map output buffer")?;

        let data = buffer_slice.get_mapped_range();

        // Remove row padding and create image
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);
        for y in 0..self.height {
            let start = (y * padded_bytes_per_row) as usize;
            let end = start + (self.width * bytes_per_pixel) as usize;
            pixels.extend_from_slice(&data[start..end]);
        }

        drop(data);
        output_buffer.unmap();

        let img: image::RgbaImage =
            image::ImageBuffer::from_raw(self.width, self.height, pixels)
                .context("Failed to create image buffer")?;

        Ok(img)
    }

    /// Render mesh with points to PNG.
    pub fn render_with_points_to_png(
        &self,
        vertices: &[MeshVertex],
        indices: &[u32],
        uniforms: &Uniforms,
        background_color: [f32; 3],
        grid_vertices: Option<&[GridVertex]>,
        wireframe: Option<&WireframeOptions>,
        points: Option<&[PointInstance]>,
        point_size: f32,
        point_shape: PointShape,
        output_path: &Path,
    ) -> Result<()> {
        let img = self.render_with_points_to_image(
            vertices,
            indices,
            uniforms,
            background_color,
            grid_vertices,
            wireframe,
            points,
            point_size,
            point_shape,
        )?;

        img.save(output_path).context("Failed to save PNG")?;
        Ok(())
    }
}
