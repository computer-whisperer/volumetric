//! The mesh geometry source: retained triangle meshes drawn into the
//! G-buffer, and the transparent ones into the layers peeled after it.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::gbuffer::LayerBuffers;
use crate::texture::{PipelineId, upload};
use crate::{
    DynamicBuffer, GBuffer, MaterialId, MeshData, MeshVertex, ObjectId, SceneTexture, TextureData,
};

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
    /// What the mesh pass binds for it at group 1.
    texture: MeshTexture,
    /// The size the texture was uploaded at when its full size exceeded
    /// the device's `max_texture_dimension_2d` and the top mip levels
    /// were dropped; `None` when it fit (or there is no texture).
    pub texture_downscaled: Option<(u32, u32)>,
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
    /// what was dropped), and binds its texture: the copy in `uploaded`
    /// made from it for this pipeline, or one uploaded now and added there.
    /// `textures` is the mesh pipeline's binding
    /// (`MeshPipeline::textures`); the queue carries the texture's texels.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &MeshTextureBinding,
        uploaded: &mut GpuTextureSet,
        data: &MeshData,
    ) -> Self {
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
                MeshVertex::textured(v.position, normal.into(), v.color, v.uv())
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

        // A texture whose pixels are gone and cannot be made again draws
        // as an untextured mesh does.
        let texture = match data.texture.as_ref().and_then(|source| {
            uploaded.get_or_upload(source, textures.id, || {
                GpuTexture::new(device, queue, textures, source)
            })
        }) {
            Some(texture) => MeshTexture::Texture(texture),
            None => MeshTexture::White(textures.white(device, queue)),
        };
        let texture_downscaled = match &texture {
            MeshTexture::Texture(texture) => texture.downscaled,
            MeshTexture::White(_) => None,
        };

        Self {
            vertex_buffer,
            index_buffer,
            draw_count,
            texture,
            texture_downscaled,
            dropped_triangles,
            total_triangles,
            bounds,
        }
    }
}

/// What a mesh binds at group 1.
enum MeshTexture {
    /// Its texture on the device, shared with every mesh made from the same
    /// scene texture for the same pipeline.
    Texture(Arc<GpuTexture>),
    /// The pipeline's white texel: sampling it leaves the vertex colour
    /// alone.
    White(Arc<wgpu::BindGroup>),
}

impl MeshTexture {
    fn bind_group(&self) -> &wgpu::BindGroup {
        match self {
            MeshTexture::Texture(texture) => &texture.bind_group,
            MeshTexture::White(white) => white,
        }
    }
}

/// A mesh texture on the device, bound for one mesh pipeline: the uploaded
/// counterpart of a [`SceneTexture`], as [`GpuMesh`] is of a mesh. Made on
/// the device's thread when the first mesh that draws it is uploaded, and
/// freed when the last mesh or [`GpuTextureSet`] holding it lets go.
pub struct GpuTexture {
    /// The scene texture it was made from, which names it in a set. Weak,
    /// so the copy keeps neither the pixels nor what reloads them alive;
    /// it still holds the address, which no other texture can then take.
    source: std::sync::Weak<SceneTexture>,
    /// The mesh pipeline it was bound for.
    pipeline: PipelineId,
    /// Group 1 of the mesh pass; it keeps the texture alive.
    bind_group: wgpu::BindGroup,
    /// The size it was uploaded at when its full size exceeded the
    /// device's largest texture side; `None` when it fit.
    downscaled: Option<(u32, u32)>,
}

impl GpuTexture {
    /// Uploads `source`'s pixels for the pipeline `textures` binds for;
    /// `None` when they are gone and cannot be made again.
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &MeshTextureBinding,
        source: &Arc<SceneTexture>,
    ) -> Option<GpuTexture> {
        source.with_pixels(|pixels| {
            let (gpu, downscaled) = upload(device, queue, pixels);
            GpuTexture {
                source: Arc::downgrade(source),
                pipeline: textures.id,
                bind_group: textures.bind(device, &gpu, "retained_mesh_texture_bg"),
                downscaled,
            }
        })
    }
}

/// Device textures a host keeps so they outlive the retained scenes drawn
/// with them: given to [`crate::Renderer::create_retained_scene_with`] (or
/// `create_retained_mesh_with`), a mesh takes the copy here made from its
/// texture for that renderer, and a texture uploaded for it is added. A
/// scene dropped and made again from the same set then uploads its
/// geometry only: no texture is sent, decoded or reloaded. Dropping the
/// set lets its textures go with the last mesh that draws them.
#[derive(Default)]
pub struct GpuTextureSet {
    textures: Vec<Arc<GpuTexture>>,
}

impl GpuTextureSet {
    /// The copy made from `source` for `pipeline`, or the one `make`
    /// uploads now, added to the set.
    fn get_or_upload(
        &mut self,
        source: &Arc<SceneTexture>,
        pipeline: PipelineId,
        make: impl FnOnce() -> Option<GpuTexture>,
    ) -> Option<Arc<GpuTexture>> {
        if let Some(texture) = self.textures.iter().find(|texture| {
            texture.pipeline == pipeline
                && std::ptr::eq(texture.source.as_ptr(), Arc::as_ptr(source))
        }) {
            return Some(texture.clone());
        }
        let texture = Arc::new(make()?);
        self.textures.push(texture.clone());
        Some(texture)
    }

    /// How many device textures the set holds.
    pub fn len(&self) -> usize {
        self.textures.len()
    }

    /// Whether the set holds no device texture.
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }
}

/// What the mesh pass binds at group 1: a mesh's texture and the shared
/// sampler, and the white texel every untextured mesh binds instead.
pub struct MeshTextureBinding {
    /// Names this pipeline to the device textures made for it.
    id: PipelineId,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// A 1×1 white texel and its bind group: sampling it leaves the
    /// vertex colour alone. Made on first use, since uploading it takes
    /// the queue the pipeline is built without.
    white: std::sync::OnceLock<(wgpu::Texture, Arc<wgpu::BindGroup>)>,
}

impl MeshTextureBinding {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_texture_bgl"),
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
        // Trilinear, repeating (UVs outside [0, 1] wrap), anisotropic on
        // native where a photographic atlas seen at a slant blurs
        // otherwise; WebGL2 has no anisotropy without an extension.
        #[cfg(not(target_arch = "wasm32"))]
        let anisotropy_clamp = 8;
        #[cfg(target_arch = "wasm32")]
        let anisotropy_clamp = 1;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh_texture_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp,
            ..Default::default()
        });
        Self {
            id: PipelineId::next(),
            layout,
            sampler,
            white: std::sync::OnceLock::new(),
        }
    }

    /// The bind group of the shared white texel.
    fn white(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Arc<wgpu::BindGroup> {
        self.white
            .get_or_init(|| {
                let (white, _) = upload(device, queue, &TextureData::solid([255; 4]));
                let bind_group = self.bind(device, &white, "mesh_texture_white_bg");
                (white, Arc::new(bind_group))
            })
            .1
            .clone()
    }

    /// A group-1 bind group sampling `texture` with the shared sampler.
    fn bind(&self, device: &wgpu::Device, texture: &wgpu::Texture, label: &str) -> wgpu::BindGroup {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
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
/// object id and material index the G-buffer carries for it, and how
/// opaque the draw is.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DrawInstance {
    model: [[f32; 4]; 4],
    ids: [u32; 2],
    opacity: f32,
    _pad: u32,
}

/// A retained mesh submitted for a frame.
pub struct MeshDraw {
    pub mesh: std::sync::Arc<GpuMesh>,
    pub transform: Mat4,
    pub object: ObjectId,
    pub material: MaterialId,
    /// 1 is opaque; less blends the mesh over what is behind it.
    pub opacity: f32,
}

/// The mesh geometry source: draws retained meshes into the G-buffer,
/// and the transparent ones into a layer at a time.
pub struct MeshPipeline {
    pipeline: wgpu::RenderPipeline,
    /// The same vertices into a layer of the transparent meshes: the
    /// fragments in front of the opaque scene and behind the layer
    /// before. This one writes the layer's surface target alone.
    layer_surface_pipeline: wgpu::RenderPipeline,
    /// And this one the rest of what a G-buffer holds, with how opaque
    /// each surface is.
    layer_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Group 1: each mesh's texture, or the shared white texel.
    textures: MeshTextureBinding,
    /// Group 2 of a layer pass: the surfaces it peels against.
    peel_layout: wgpu::BindGroupLayout,
    /// The layer pass's uniforms for the nearest layer, and for every
    /// one behind it.
    peel_uniforms: [wgpu::Buffer; 2],
    /// One entry per mesh drawn this frame: the opaque draws in draw
    /// order, then the transparent ones.
    instance_buffer: DynamicBuffer<DrawInstance>,
    /// Where the transparent draws' instances start.
    transparent_start: usize,
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
        let textures = MeshTextureBinding::new(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_gbuffer_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&textures.layout)],
            immediate_size: 0,
        });

        let surface_texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Uint,
            },
            count: None,
        };
        let peel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_peel_bgl"),
            entries: &[
                surface_texture(0),
                surface_texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let layer_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mesh_layer_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&bind_group_layout),
                    Some(&textures.layout),
                    Some(&peel_layout),
                ],
                immediate_size: 0,
            });
        let peel_uniforms = [0u32, 1].map(|previous| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_peel_uniforms"),
                contents: bytemuck::bytes_of(&[previous, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        });

        let target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        // The G-buffer's pipeline and the layers' differ in their entry
        // points (the layers' vertex stage is the G-buffer's with an
        // invariant position), in what the fragment stage reads and
        // writes, and in the depth target's format.
        let build = |label,
                     layout: &wgpu::PipelineLayout,
                     vertex_entry,
                     fragment_entry,
                     targets: &[Option<wgpu::ColorTargetState>],
                     depth_format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vertex_entry),
                    buffers: &[
                        Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<MeshVertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            // Offsets come from the struct: the two halves of
                            // the texture coordinate sit where the position
                            // and the normal would otherwise be padded.
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
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32,
                                    offset: std::mem::offset_of!(MeshVertex, u) as u64,
                                    shader_location: 8,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32,
                                    offset: std::mem::offset_of!(MeshVertex, v) as u64,
                                    shader_location: 9,
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
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32,
                                    offset: std::mem::offset_of!(DrawInstance, opacity) as u64,
                                    shader_location: 10,
                                },
                            ],
                        }),
                    ],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fragment_entry),
                    targets,
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
                    format: depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = build(
            "mesh_gbuffer_pipeline",
            &pipeline_layout,
            "vs_main",
            "fs_gbuffer",
            &[
                target(GBuffer::ALBEDO_FORMAT),
                target(GBuffer::NORMAL_FORMAT),
                target(GBuffer::SURFACE_FORMAT),
            ],
            GBuffer::DEPTH_FORMAT,
        );
        let layer_surface_pipeline = build(
            "mesh_layer_surface_pipeline",
            &layer_pipeline_layout,
            "vs_layer",
            "fs_layer_surface",
            &[target(GBuffer::SURFACE_FORMAT)],
            LayerBuffers::DEPTH_FORMAT,
        );
        let layer_pipeline = build(
            "mesh_layer_pipeline",
            &layer_pipeline_layout,
            "vs_layer",
            "fs_layer",
            &[
                target(GBuffer::ALBEDO_FORMAT),
                target(GBuffer::NORMAL_FORMAT),
                target(LayerBuffers::OPACITY_FORMAT),
            ],
            LayerBuffers::DEPTH_FORMAT,
        );

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
            layer_surface_pipeline,
            layer_pipeline,
            uniform_buffer,
            bind_group,
            textures,
            peel_layout,
            peel_uniforms,
            instance_buffer: DynamicBuffer::new(
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                "mesh_instance_buffer",
            ),
            transparent_start: 0,
        }
    }

    /// Group 2 of a layer pass: the opaque scene's surface target, and
    /// the surface target of the layer peeled before (`None` for the
    /// nearest layer, which has nothing in front of it to stay behind).
    /// `unread` stands in for the missing one: any surface target the
    /// pass does not write.
    pub fn bind_peel(
        &self,
        device: &wgpu::Device,
        opaque_surface: &wgpu::TextureView,
        previous_surface: Option<&wgpu::TextureView>,
        unread: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_peel_bg"),
            layout: &self.peel_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(opaque_surface),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        previous_surface.unwrap_or(unread),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.peel_uniforms[previous_surface.is_some() as usize]
                        .as_entire_binding(),
                },
            ],
        })
    }

    /// Uploads a mesh as a GPU resident bound to this pipeline's texture
    /// layout, its texture taken from or added to `uploaded`; see
    /// [`GpuMesh::new`].
    pub fn create_retained(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uploaded: &mut GpuTextureSet,
        data: &MeshData,
    ) -> GpuMesh {
        GpuMesh::new(device, queue, &self.textures, uploaded, data)
    }

    /// Uploads the frame's camera and one instance per draw. Call before
    /// the passes that [`render`](Self::render) `opaque` and
    /// [`render_layer`](Self::render_layer) `transparent`.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        opaque: &[MeshDraw],
        transparent: &[MeshDraw],
    ) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&view_proj.to_cols_array_2d()),
        );
        let instances: Vec<DrawInstance> = opaque
            .iter()
            .chain(transparent)
            .map(|draw| DrawInstance {
                model: draw.transform.to_cols_array_2d(),
                ids: [draw.object.0, draw.material.0],
                opacity: draw.opacity,
                _pad: 0,
            })
            .collect();
        self.instance_buffer.upload(device, queue, &instances);
        self.transparent_start = opaque.len();
    }

    /// Draws the opaque `draws` into a G-buffer pass, each under the
    /// instance [`prepare`](Self::prepare) uploaded for it.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, draws: &'a [MeshDraw]) {
        render_pass.set_pipeline(&self.pipeline);
        self.draw(render_pass, draws, 0);
    }

    /// Draws the transparent `draws` into a layer pass: of their
    /// fragments in front of the opaque scene and behind the layer
    /// `peel` names (see [`bind_peel`](Self::bind_peel)), the pass's
    /// depth test keeps the nearest. `surface_only` writes the layer's
    /// surface target and nothing else; otherwise its albedo, normal and
    /// opacity targets, for the same fragments.
    pub fn render_layer<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        draws: &'a [MeshDraw],
        peel: &'a wgpu::BindGroup,
        surface_only: bool,
    ) {
        render_pass.set_pipeline(if surface_only {
            &self.layer_surface_pipeline
        } else {
            &self.layer_pipeline
        });
        render_pass.set_bind_group(2, peel, &[]);
        self.draw(render_pass, draws, self.transparent_start);
    }

    /// Draws `draws` with the pipeline already set; their instances
    /// start at `first` of the frame's.
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        draws: &'a [MeshDraw],
        first: usize,
    ) {
        let Some(instances) = self.instance_buffer.buffer() else {
            return;
        };
        let stride = std::mem::size_of::<DrawInstance>() as u64;
        let uploaded = self.instance_buffer.len().saturating_sub(first);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        for (i, draw) in draws.iter().enumerate().take(uploaded) {
            let mesh = &draw.mesh;
            if mesh.draw_count == 0 {
                continue;
            }
            let start = (first + i) as u64 * stride;
            render_pass.set_bind_group(1, mesh.texture.bind_group(), &[]);
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

    /// The vertex is 48 bytes with its attributes where the vertex buffer
    /// layout says: position at 0, u at 12, normal at 16, v at 28, colour
    /// at 32.
    #[test]
    fn the_vertex_layout_matches_the_struct() {
        assert_eq!(std::mem::size_of::<MeshVertex>(), 48);
        assert_eq!(std::mem::offset_of!(MeshVertex, position), 0);
        assert_eq!(std::mem::offset_of!(MeshVertex, u), 12);
        assert_eq!(std::mem::offset_of!(MeshVertex, normal), 16);
        assert_eq!(std::mem::offset_of!(MeshVertex, v), 28);
        assert_eq!(std::mem::offset_of!(MeshVertex, color), 32);
        let v = MeshVertex::textured([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [0.5; 4], [0.25, 0.75]);
        let words: &[f32] = bytemuck::cast_slice(bytemuck::bytes_of(&v));
        assert_eq!(
            words,
            &[1.0, 2.0, 3.0, 0.25, 0.0, 0.0, 1.0, 0.75, 0.5, 0.5, 0.5, 0.5]
        );
        assert_eq!(v.uv(), [0.25, 0.75]);
        assert_eq!(MeshVertex::new([0.0; 3], [0.0; 3]).uv(), [0.0; 2]);
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
