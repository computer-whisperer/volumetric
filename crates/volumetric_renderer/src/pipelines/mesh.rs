//! Mesh G-buffer rendering pipeline.
//!
//! Renders triangle meshes to the G-buffer with deferred shading data.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::{DynamicBuffer, MeshData, MeshVertex};

/// A mesh resident on the GPU: world-space vertices (the transform is
/// applied at creation) uploaded once and drawn by reference each frame,
/// so rebuilding a preview is the only time its dense buffers travel to
/// the device. Created by [`GpuMesh::new`]; drawn via
/// `Renderer::submit_retained_mesh`.
pub struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: Option<wgpu::Buffer>,
    draw_count: u32,
    /// Triangles dropped at the device's buffer size limit; 0 when the
    /// whole mesh is resident.
    pub dropped_triangles: usize,
    /// Triangles in the source mesh.
    pub total_triangles: usize,
}

impl GpuMesh {
    /// Uploads `data`, transformed to world space, clamping to the
    /// device's `max_buffer_size` limit (keeping the largest renderable
    /// triangle prefix and reporting what was dropped).
    pub fn new(device: &wgpu::Device, data: &MeshData, transform: Mat4) -> Self {
        let mut vertices: Vec<MeshVertex> = data
            .vertices
            .iter()
            .map(|v| {
                let pos = transform.transform_point3(Vec3::from(v.position));
                // normalize() of a zero/degenerate normal mints NaN, which
                // renders as uniform white downstream; substitute +Z.
                let normal = transform
                    .transform_vector3(Vec3::from(v.normal))
                    .normalize_or_zero();
                let normal = if normal == Vec3::ZERO {
                    Vec3::Z
                } else {
                    normal
                };
                MeshVertex::colored(pos.into(), normal.into(), v.color)
            })
            .collect();
        let mut indices = data.indices.clone();

        let max_buffer = device.limits().max_buffer_size;
        let (dropped_triangles, total_triangles) = clamp_mesh_to_budget(
            &mut vertices,
            &mut indices,
            (max_buffer / std::mem::size_of::<MeshVertex>() as u64) as usize,
            (max_buffer / std::mem::size_of::<u32>() as u64) as usize,
        );

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retained_mesh_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let draw_count = indices
            .as_ref()
            .map_or(vertices.len(), |indices| indices.len()) as u32;
        let index_buffer = indices.map(|indices| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("retained_mesh_index_buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            })
        });

        Self {
            vertex_buffer,
            index_buffer,
            draw_count,
            dropped_triangles,
            total_triangles,
        }
    }
}

/// Clamps a single mesh to per-buffer element budgets, keeping the largest
/// prefix of whole, renderable triangles: vertices are truncated to the
/// vertex budget, and (for indexed meshes) only triangles whose vertices
/// all survived are kept. Returns `(dropped_triangles, total_triangles)`.
fn clamp_mesh_to_budget(
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Option<Vec<u32>>,
    max_vertices: usize,
    max_indices: usize,
) -> (usize, usize) {
    match indices {
        Some(indices) => {
            let total = indices.len() / 3;
            if vertices.len() <= max_vertices && indices.len() <= max_indices {
                return (0, total);
            }
            let kept_vertices = vertices.len().min(max_vertices);
            vertices.truncate(kept_vertices);
            let in_range = kept_vertices as u32;
            let mut kept = Vec::new();
            for triple in indices.chunks_exact(3) {
                if kept.len() + 3 > max_indices {
                    break;
                }
                if triple.iter().all(|&i| i < in_range) {
                    kept.extend_from_slice(triple);
                }
            }
            *indices = kept;
            (total - indices.len() / 3, total)
        }
        None => {
            let total = vertices.len() / 3;
            let kept = (vertices.len().min(max_vertices) / 3) * 3;
            vertices.truncate(kept);
            (total - kept / 3, total)
        }
    }
}

/// Uniform data for mesh rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir_world: [f32; 3],
    pub _pad0: f32,
    pub base_color: [f32; 3],
    pub _pad1: f32,
}

impl Default for MeshUniforms {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            light_dir_world: [0.4, 0.7, 0.2],
            _pad0: 0.0,
            base_color: [0.85, 0.9, 1.0],
            _pad1: 0.0,
        }
    }
}

impl PartialEq for MeshUniforms {
    fn eq(&self, other: &Self) -> bool {
        self.view_proj == other.view_proj
            && self.light_dir_world == other.light_dir_world
            && self.base_color == other.base_color
    }
}

/// Pipeline for rendering meshes to the G-buffer.
pub struct MeshPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: DynamicBuffer<MeshVertex>,
    index_buffer: DynamicBuffer<u32>,
    cached_uniforms: Option<MeshUniforms>,
}

impl MeshPipeline {
    /// Create a new mesh pipeline.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_gbuffer_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/mesh_gbuffer.wgsl"
            ))),
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_uniform_bgl"),
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
            label: Some("mesh_gbuffer_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_gbuffer_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // position
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        // normal — offset 16, not 12: MeshVertex pads the
                        // position out to 16 bytes (_pad0), so the normal
                        // starts one f32 later than a packed layout would.
                        // Reading at 12 fed the shader (_pad0, nx, ny),
                        // which turned exactly-axis-aligned normals into
                        // normalize((0,0,0)) = NaN.
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: std::mem::offset_of!(MeshVertex, normal) as u64,
                            shader_location: 1,
                        },
                        // vertex color (multiplied into the base color)
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: std::mem::offset_of!(MeshVertex, color) as u64,
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
                    // Color output
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Normal output
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Depth output (for SSAO)
                    // Use Rgba16Float on web for better compatibility
                    Some(wgpu::ColorTargetState {
                        #[cfg(target_arch = "wasm32")]
                        format: wgpu::TextureFormat::Rgba16Float,
                        #[cfg(not(target_arch = "wasm32"))]
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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

        // Uniform buffer
        let uniforms = MeshUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_uniform_buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_uniform_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Vertex and index buffers
        let vertex_buffer = DynamicBuffer::new(
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "mesh_vertex_buffer",
        );
        let index_buffer = DynamicBuffer::new(
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            "mesh_index_buffer",
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
    pub fn update_uniforms(&mut self, queue: &wgpu::Queue, uniforms: &MeshUniforms) {
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

    /// Record the G-buffer render pass.
    ///
    /// This renders meshes to the G-buffer textures (color, normal, depth).
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

    /// Draw retained meshes into the G-buffer pass. Shares the immediate
    /// path's pipeline and uniforms (retained vertices are world-space).
    pub fn render_retained<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        meshes: &'a [std::sync::Arc<GpuMesh>],
    ) {
        if meshes.is_empty() {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        for mesh in meshes {
            if mesh.draw_count == 0 {
                continue;
            }
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            match &mesh.index_buffer {
                Some(indices) => {
                    render_pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..mesh.draw_count, 0, 0..1);
                }
                None => render_pass.draw(0..mesh.draw_count, 0..1),
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn verts(n: usize) -> Vec<MeshVertex> {
        vec![MeshVertex::new([0.0; 3], [0.0, 1.0, 0.0]); n]
    }

    /// Meshes within both budgets pass through untouched.
    #[test]
    fn clamp_is_a_no_op_within_budget() {
        let mut vertices = verts(6);
        let mut indices = Some(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            clamp_mesh_to_budget(&mut vertices, &mut indices, 6, 6),
            (0, 2)
        );
        assert_eq!(vertices.len(), 6);
        assert_eq!(indices.unwrap().len(), 6);
    }

    /// Indexed meshes keep only triangles whose vertices all survived the
    /// vertex clamp, then bow to the index budget in whole triangles.
    #[test]
    fn clamp_filters_indexed_triangles_past_the_vertex_budget() {
        let mut vertices = verts(6);
        // Triangles: (0,1,2) survives, (1,2,5) references a clamped vertex.
        let mut indices = Some(vec![0, 1, 2, 1, 2, 5]);
        assert_eq!(
            clamp_mesh_to_budget(&mut vertices, &mut indices, 5, 100),
            (1, 2)
        );
        assert_eq!(vertices.len(), 5);
        assert_eq!(indices.unwrap(), vec![0, 1, 2]);
    }

    /// The index budget truncates to whole triangles.
    #[test]
    fn clamp_truncates_to_the_index_budget() {
        let mut vertices = verts(3);
        let mut indices = Some(vec![0, 1, 2, 2, 1, 0, 1, 0, 2]);
        assert_eq!(
            clamp_mesh_to_budget(&mut vertices, &mut indices, 100, 7),
            (1, 3)
        );
        assert_eq!(indices.unwrap().len(), 6);
    }

    /// Non-indexed soups truncate to whole triangles within the vertex
    /// budget.
    #[test]
    fn clamp_truncates_soups_to_whole_triangles() {
        let mut vertices = verts(9);
        let mut indices = None;
        assert_eq!(
            clamp_mesh_to_budget(&mut vertices, &mut indices, 8, 100),
            (1, 3)
        );
        assert_eq!(vertices.len(), 6);
    }
}
