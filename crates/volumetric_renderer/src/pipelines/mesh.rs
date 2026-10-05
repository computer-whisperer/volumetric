//! The mesh geometry source: retained triangle meshes drawn into the
//! G-buffer.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::{DynamicBuffer, GBuffer, MaterialId, MeshData, MeshVertex, ObjectId};

/// A mesh resident on the GPU: its vertices as given (the transform is
/// applied per draw), uploaded once and drawn by reference each frame with
/// whatever pose the frame submits, so re-posing a part costs nothing and
/// rebuilding a preview is the only time its dense buffers travel to the
/// device. Created by `Renderer::create_retained_mesh`; drawn via
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
    /// The corners of the box around the mesh's vertices, in its own
    /// coordinates; `None` for an empty mesh.
    pub bounds: Option<(Vec3, Vec3)>,
}

impl GpuMesh {
    /// Uploads `data` as given, clamping to the device's `max_buffer_size`
    /// limit (keeping the largest renderable triangle prefix and reporting
    /// what was dropped).
    pub fn new(device: &wgpu::Device, data: &MeshData) -> Self {
        let mut vertices: Vec<MeshVertex> = data
            .vertices
            .iter()
            .map(|v| {
                // normalize() of a zero/degenerate normal mints NaN, which
                // renders as uniform white downstream; substitute +Z.
                let normal = Vec3::from(v.normal).normalize_or_zero();
                let normal = if normal == Vec3::ZERO {
                    Vec3::Z
                } else {
                    normal
                };
                MeshVertex::colored(v.position, normal.into(), v.color)
            })
            .collect();
        let mut indices = data.indices.clone();
        let bounds = vertices
            .iter()
            .map(|v| Vec3::from(v.position))
            .filter(|p| p.is_finite())
            .fold(None, |bounds: Option<(Vec3, Vec3)>, p| {
                Some(bounds.map_or((p, p), |(min, max)| (min.min(p), max.max(p))))
            });

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
            bounds,
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

/// One draw's instance data: the model matrix, column major, then the
/// object id and material index the G-buffer carries for it.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DrawInstance {
    model: [[f32; 4]; 4],
    ids: [u32; 2],
    _pad: [u32; 2],
}

/// A retained mesh submitted for a frame.
pub struct MeshDraw {
    pub mesh: std::sync::Arc<GpuMesh>,
    pub transform: Mat4,
    pub object: ObjectId,
    pub material: MaterialId,
}

/// The mesh geometry source: draws retained meshes into the G-buffer.
pub struct MeshPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// One entry per mesh drawn this frame, in draw order.
    instance_buffer: DynamicBuffer<DrawInstance>,
}

impl MeshPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_gbuffer_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/mesh_gbuffer.wgsl"
            ))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_uniform_bgl"),
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
            label: Some("mesh_gbuffer_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
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
            label: Some("mesh_gbuffer_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<MeshVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        // Offsets come from the struct: MeshVertex pads the
                        // position and normal out to 16 bytes each.
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: std::mem::offset_of!(MeshVertex, position) as u64,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: std::mem::offset_of!(MeshVertex, normal) as u64,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: std::mem::offset_of!(MeshVertex, color) as u64,
                                shader_location: 2,
                            },
                        ],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<DrawInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 4,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 32,
                                shader_location: 5,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 48,
                                shader_location: 6,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32x2,
                                offset: std::mem::offset_of!(DrawInstance, ids) as u64,
                                shader_location: 7,
                            },
                        ],
                    }),
                ],
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
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
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
            label: Some("mesh_uniform_buffer"),
            contents: bytemuck::bytes_of(&Mat4::IDENTITY.to_cols_array_2d()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_uniform_bg"),
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
            instance_buffer: DynamicBuffer::new(
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                "mesh_instance_buffer",
            ),
        }
    }

    /// Uploads the frame's camera and one instance per draw. Call before
    /// the pass that [`render`](Self::render)s the same draws.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        draws: &[MeshDraw],
    ) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&view_proj.to_cols_array_2d()),
        );
        let instances: Vec<DrawInstance> = draws
            .iter()
            .map(|draw| DrawInstance {
                model: draw.transform.to_cols_array_2d(),
                ids: [draw.object.0, draw.material.0],
                _pad: [0; 2],
            })
            .collect();
        self.instance_buffer.upload(device, queue, &instances);
    }

    /// Draws `draws` into a G-buffer pass, each under the instance
    /// [`prepare`](Self::prepare) uploaded for it.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, draws: &'a [MeshDraw]) {
        let Some(instances) = self.instance_buffer.buffer() else {
            return;
        };
        let stride = std::mem::size_of::<DrawInstance>() as u64;
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        for (i, draw) in draws.iter().enumerate().take(self.instance_buffer.len()) {
            let mesh = &draw.mesh;
            if mesh.draw_count == 0 {
                continue;
            }
            let start = i as u64 * stride;
            render_pass.set_vertex_buffer(1, instances.slice(start..start + stride));
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
