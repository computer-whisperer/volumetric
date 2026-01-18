//! Unified rendering engine for volumetric UI.
//!
//! This module provides a high-quality rendering system supporting simultaneous
//! mesh, line, and point rendering on both native and WebGPU backends.
//!
//! # Architecture
//!
//! The renderer uses a multi-pass deferred rendering approach:
//!
//! 1. **Mesh G-Buffer Pass**: Renders meshes to color, normal, and depth buffers
//! 2. **SSAO Pass**: Computes screen-space ambient occlusion
//! 3. **Composite Pass**: Combines mesh color with AO
//! 4. **Grid Pass**: Renders depth-tested grid lines
//! 5. **Line Pass**: Renders depth-tested scene lines
//! 6. **Point Pass**: Renders depth-tested scene points
//! 7. **Overlay Passes**: Renders overlay lines/points (no depth test)
//! 8. **Axis Indicator Pass**: Mini-viewport in corner
//!
//! # Usage
//!
//! ```ignore
//! let mut renderer = Renderer::new(surface_format);
//! renderer.initialize(&device, &queue, &adapter);
//!
//! // Each frame:
//! renderer.submit_mesh(&mesh_data, transform, material);
//! renderer.submit_lines(&line_data, transform, style);
//! renderer.submit_points(&point_data, transform, style);
//! renderer.render(&device, &queue, &camera, &settings, &target_view);
//! renderer.end_frame();
//! ```

mod buffer;
mod callback;
mod camera;
mod conversions;
mod gbuffer;
mod pipelines;
pub mod test_scenes;
mod types;

pub use conversions::{convert_mesh_data, convert_points_to_point_data};

pub use buffer::{DynamicBuffer, QuadVertex, StaticBuffer, UniformBuffer, QUAD_INDICES, QUAD_VERTICES};
pub use callback::{SceneCallback, SceneData, SceneDrawData};
pub use camera::{Camera, CameraAction, CameraControlScheme, CameraInputState, CameraUniforms, ViewDirection};
pub use gbuffer::{AoTexture, GBuffer};
pub use pipelines::{
    CompositePipeline, GpuPointInstance, LinePipeline, LineUniforms, MeshPipeline, MeshUniforms,
    PointPipeline, PointUniforms, SsaoPipeline, SsaoUniforms,
};
pub use types::{
    AxisIndicator, DepthMode, GridPlanes, GridSettings, LineData, LineInstance, LinePattern,
    LineSegment, LineStyle, MaterialId, MeshData, MeshVertex, PointData, PointInstance,
    PointShape, PointStyle, RenderSettings, WidthMode,
};

use glam::{Mat4, Vec3};

/// Renderer capabilities detected from the GPU adapter.
#[derive(Clone, Debug)]
pub struct RendererCapabilities {
    /// Maximum texture dimension
    pub max_texture_size: u32,
    /// Recommended SSAO sample count for this platform
    pub recommended_ssao_samples: u32,
    /// Whether running on WebGPU
    pub is_web: bool,
}

impl RendererCapabilities {
    /// Detect capabilities from the given adapter.
    pub fn detect(adapter: &wgpu::Adapter) -> Self {
        let limits = adapter.limits();
        let is_web = cfg!(target_arch = "wasm32");

        Self {
            max_texture_size: limits.max_texture_dimension_2d,
            recommended_ssao_samples: if is_web { 8 } else { 16 },
            is_web,
        }
    }
}

/// A submitted mesh for the current frame.
struct SubmittedMesh {
    data: MeshData,
    transform: Mat4,
    #[allow(dead_code)]
    material: MaterialId,
}

/// A submitted line batch for the current frame.
struct SubmittedLines {
    data: LineData,
    transform: Mat4,
    style: LineStyle,
}

/// A submitted point batch for the current frame.
struct SubmittedPoints {
    data: PointData,
    transform: Mat4,
    style: PointStyle,
}

/// GPU resources for the renderer.
struct GpuResources {
    // Pipelines
    mesh_pipeline: MeshPipeline,
    ssao_pipeline: SsaoPipeline,
    composite_pipeline: CompositePipeline,
    line_pipeline: LinePipeline,
    point_pipeline: PointPipeline,

    // Textures
    gbuffer: GBuffer,
    ao_texture: AoTexture,

    // Bind groups (recreated on resize)
    ssao_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,

    // Sampler
    sampler: wgpu::Sampler,
}

/// Main renderer that manages all GPU resources and rendering.
///
/// The renderer provides a unified interface for submitting geometry each frame
/// and handles all GPU resource management internally.
pub struct Renderer {
    // Surface format for the render target
    surface_format: wgpu::TextureFormat,

    // Viewport size
    viewport_size: (u32, u32),

    // GPU resources (initialized lazily)
    gpu: Option<GpuResources>,

    // Submitted geometry for current frame
    frame_meshes: Vec<SubmittedMesh>,
    frame_lines: Vec<SubmittedLines>,
    frame_points: Vec<SubmittedPoints>,

    // Grid line cache (regenerated when settings change)
    cached_grid_lines: Vec<LineSegment>,
    cached_grid_settings_hash: u64,

    // Capabilities
    capabilities: Option<RendererCapabilities>,
}

impl Renderer {
    /// Create a new renderer.
    pub fn new(surface_format: wgpu::TextureFormat) -> Self {
        Self {
            surface_format,
            viewport_size: (1, 1),
            gpu: None,
            frame_meshes: Vec::new(),
            frame_lines: Vec::new(),
            frame_points: Vec::new(),
            cached_grid_lines: Vec::new(),
            cached_grid_settings_hash: 0,
            capabilities: None,
        }
    }

    /// Initialize GPU resources. Call this once after creation.
    /// The adapter is optional - if not provided, capabilities detection is skipped.
    pub fn initialize(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        adapter: Option<&wgpu::Adapter>,
    ) {
        self.capabilities = adapter.map(RendererCapabilities::detect);

        // Create pipelines
        let mesh_pipeline = MeshPipeline::new(device, self.surface_format);
        let ssao_pipeline = SsaoPipeline::new(device);
        let composite_pipeline = CompositePipeline::new(device, self.surface_format);
        let line_pipeline = LinePipeline::new(device, self.surface_format);
        let point_pipeline = PointPipeline::new(device, self.surface_format);

        // Create textures
        let gbuffer = GBuffer::new(device, self.viewport_size.0, self.viewport_size.1, self.surface_format);
        let ao_texture = AoTexture::new(device, self.viewport_size.0, self.viewport_size.1);

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("renderer_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind groups
        let ssao_bind_group = ssao_pipeline.create_bind_group(
            device,
            &gbuffer.normal_view,
            &gbuffer.depth_view,
        );
        let composite_bind_group = composite_pipeline.create_bind_group(
            device,
            &gbuffer.color_view,
            &ao_texture.view,
        );

        self.gpu = Some(GpuResources {
            mesh_pipeline,
            ssao_pipeline,
            composite_pipeline,
            line_pipeline,
            point_pipeline,
            gbuffer,
            ao_texture,
            ssao_bind_group,
            composite_bind_group,
            sampler,
        });
    }

    /// Check if the renderer has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.gpu.is_some()
    }

    /// Get the renderer capabilities.
    pub fn capabilities(&self) -> Option<&RendererCapabilities> {
        self.capabilities.as_ref()
    }

    /// Set the viewport size. Call when the window is resized.
    pub fn set_viewport_size(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let new_size = (width.max(1), height.max(1));
        if self.viewport_size == new_size {
            return;
        }

        self.viewport_size = new_size;

        // Resize GPU resources if initialized
        if let Some(gpu) = &mut self.gpu {
            gpu.gbuffer.resize_if_needed(device, new_size.0, new_size.1);
            gpu.ao_texture.resize_if_needed(device, new_size.0, new_size.1);

            // Recreate bind groups with new texture views
            gpu.ssao_bind_group = gpu.ssao_pipeline.create_bind_group(
                device,
                &gpu.gbuffer.normal_view,
                &gpu.gbuffer.depth_view,
            );
            gpu.composite_bind_group = gpu.composite_pipeline.create_bind_group(
                device,
                &gpu.gbuffer.color_view,
                &gpu.ao_texture.view,
            );
        }
    }

    /// Get the current viewport size.
    pub fn viewport_size(&self) -> (u32, u32) {
        self.viewport_size
    }

    /// Submit mesh geometry for this frame.
    pub fn submit_mesh(&mut self, mesh: &MeshData, transform: Mat4, material: MaterialId) {
        if mesh.vertices.is_empty() {
            return;
        }
        self.frame_meshes.push(SubmittedMesh {
            data: mesh.clone(),
            transform,
            material,
        });
    }

    /// Submit line segments for this frame.
    pub fn submit_lines(&mut self, lines: &LineData, transform: Mat4, style: LineStyle) {
        if lines.segments.is_empty() {
            return;
        }
        self.frame_lines.push(SubmittedLines {
            data: lines.clone(),
            transform,
            style,
        });
    }

    /// Submit points for this frame.
    pub fn submit_points(&mut self, points: &PointData, transform: Mat4, style: PointStyle) {
        if points.points.is_empty() {
            return;
        }
        self.frame_points.push(SubmittedPoints {
            data: points.clone(),
            transform,
            style,
        });
    }

    /// Execute all rendering for the frame.
    ///
    /// This performs all render passes in order:
    /// 1. Mesh G-Buffer
    /// 2. SSAO
    /// 3. Composite (to offscreen target)
    /// 4. Grid lines (depth-tested)
    /// 5. Scene lines (depth-tested)
    /// 6. Scene points (depth-tested)
    /// 7. Overlay lines
    /// 8. Overlay points
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        settings: &RenderSettings,
        target: &wgpu::TextureView,
    ) {
        // Update grid lines if settings changed (must be before borrowing self.gpu)
        self.update_grid_cache(settings);

        let Some(gpu) = &mut self.gpu else {
            return;
        };

        let aspect = self.viewport_size.0 as f32 / self.viewport_size.1 as f32;
        let view_proj = camera.view_projection_matrix(aspect);
        let view_proj_array = view_proj.to_cols_array_2d();
        let screen_size = [self.viewport_size.0 as f32, self.viewport_size.1 as f32];

        // =================================================================
        // Pass 1: Mesh G-Buffer
        // =================================================================
        if !self.frame_meshes.is_empty() {
            // Collect all mesh vertices (transformed)
            let mut all_vertices = Vec::new();
            let mut all_indices = Vec::new();
            let mut use_indices = false;

            for submitted in &self.frame_meshes {
                let base_vertex = all_vertices.len() as u32;

                // Transform vertices
                for v in &submitted.data.vertices {
                    let pos = submitted.transform.transform_point3(Vec3::from(v.position));
                    let normal = submitted.transform.transform_vector3(Vec3::from(v.normal)).normalize();
                    all_vertices.push(MeshVertex {
                        position: pos.into(),
                        normal: normal.into(),
                    });
                }

                // Handle indices
                if let Some(indices) = &submitted.data.indices {
                    use_indices = true;
                    for &idx in indices {
                        all_indices.push(base_vertex + idx);
                    }
                }
            }

            // Upload mesh data
            gpu.mesh_pipeline.upload_vertices(device, queue, &all_vertices);
            if use_indices {
                gpu.mesh_pipeline.upload_indices(device, queue, &all_indices);
            }

            // Update mesh uniforms
            let mesh_uniforms = MeshUniforms {
                view_proj: view_proj_array,
                light_dir_world: [0.4, 0.7, 0.2],
                _pad0: 0.0,
                base_color: [0.85, 0.9, 1.0],
                _pad1: 0.0,
            };
            gpu.mesh_pipeline.update_uniforms(queue, &mesh_uniforms);

            // G-buffer render pass
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mesh_gbuffer_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gpu.gbuffer.color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: settings.background_color[0] as f64,
                                    g: settings.background_color[1] as f64,
                                    b: settings.background_color[2] as f64,
                                    a: settings.background_color[3] as f64,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gpu.gbuffer.normal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gpu.gbuffer.depth_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &gpu.gbuffer.depth_stencil_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                gpu.mesh_pipeline.render(&mut pass, use_indices);
            }

            // =================================================================
            // Pass 2: SSAO
            // =================================================================
            if settings.ssao_enabled {
                let ssao_uniforms = SsaoUniforms {
                    view_proj: view_proj_array,
                    inv_view_proj: view_proj.inverse().to_cols_array_2d(),
                    screen_size_px: screen_size,
                    radius: settings.ssao_radius,
                    bias: settings.ssao_bias,
                    strength: settings.ssao_strength,
                    _pad0: [0.0; 7],
                };
                gpu.ssao_pipeline.update_uniforms(queue, &ssao_uniforms);

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ssao_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &gpu.ao_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                gpu.ssao_pipeline.render(&mut pass, &gpu.ssao_bind_group);
            } else {
                // Ensure AO is neutral when disabled
                let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ssao_disabled_clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &gpu.ao_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
        }

        // =================================================================
        // Pass 3: Composite to final target
        // =================================================================
        {
            // When there are no meshes, the depth buffer was never initialized.
            // We need to clear it in that case for subsequent passes (lines/points).
            let depth_load_op = if self.frame_meshes.is_empty() {
                wgpu::LoadOp::Clear(1.0) // Clear to far plane
            } else {
                wgpu::LoadOp::Load // Keep depth from mesh pass
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: settings.background_color[0] as f64,
                            g: settings.background_color[1] as f64,
                            b: settings.background_color[2] as f64,
                            a: settings.background_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.gbuffer.depth_stencil_view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load_op,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if !self.frame_meshes.is_empty() {
                gpu.composite_pipeline.render(&mut pass, &gpu.composite_bind_group);
            }
        }

        // =================================================================
        // Pass 4-6: Lines and Points (depth-tested)
        // =================================================================
        {
            // Collect and prepare all depth-tested line instances
            let mut all_line_segments: Vec<LineSegment> = Vec::new();
            let mut line_style = LineStyle {
                width: 1.0,
                width_mode: WidthMode::ScreenSpace,
                pattern: LinePattern::Solid,
                depth_mode: DepthMode::Normal,
            };

            // Add grid lines
            if !self.cached_grid_lines.is_empty() {
                all_line_segments.extend_from_slice(&self.cached_grid_lines);
            }

            // Add scene lines (depth-tested)
            for submitted in &self.frame_lines {
                if submitted.style.depth_mode == DepthMode::Normal {
                    line_style = submitted.style.clone();
                    for seg in &submitted.data.segments {
                        let start = submitted.transform.transform_point3(Vec3::from(seg.start));
                        let end = submitted.transform.transform_point3(Vec3::from(seg.end));
                        all_line_segments.push(LineSegment {
                            start: start.into(),
                            end: end.into(),
                            color: seg.color,
                        });
                    }
                }
            }

            // Collect and prepare all depth-tested point instances
            let mut all_point_instances: Vec<PointInstance> = Vec::new();
            let mut point_style = PointStyle {
                size: 4.0,
                size_mode: WidthMode::ScreenSpace,
                shape: PointShape::Circle,
                depth_mode: DepthMode::Normal,
            };

            for submitted in &self.frame_points {
                if submitted.style.depth_mode == DepthMode::Normal {
                    point_style = submitted.style.clone();
                    for pt in &submitted.data.points {
                        let pos = submitted.transform.transform_point3(Vec3::from(pt.position));
                        all_point_instances.push(PointInstance {
                            position: pos.into(),
                            color: pt.color,
                        });
                    }
                }
            }

            // Upload all data before starting the render pass (split borrow)
            let GpuResources {
                line_pipeline,
                point_pipeline,
                gbuffer,
                ..
            } = gpu;

            if !all_line_segments.is_empty() {
                let instances = LinePipeline::prepare_instances(&all_line_segments, &line_style);
                line_pipeline.upload_instances(device, queue, &instances);
                let uniforms = LinePipeline::create_uniforms(view_proj_array, screen_size, &line_style);
                line_pipeline.update_uniforms(queue, &uniforms);
            }

            if !all_point_instances.is_empty() {
                let instances = PointPipeline::prepare_instances(&all_point_instances);
                point_pipeline.upload_instances(device, queue, &instances);
                let uniforms = PointPipeline::create_uniforms(view_proj_array, screen_size, &point_style);
                point_pipeline.update_uniforms(queue, &uniforms);
            }

            // Now start the render pass
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("forward_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gbuffer.depth_stencil_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render all uploaded data
            if !all_line_segments.is_empty() {
                line_pipeline.render(&mut pass, DepthMode::Normal);
            }
            if !all_point_instances.is_empty() {
                point_pipeline.render(&mut pass, DepthMode::Normal);
            }
        }

        // =================================================================
        // Pass 7-8: Overlay Lines and Points (no depth test)
        // =================================================================
        {
            // Collect and prepare all overlay line instances
            let mut all_line_segments: Vec<LineSegment> = Vec::new();
            let mut line_style = LineStyle {
                width: 2.0,
                width_mode: WidthMode::ScreenSpace,
                pattern: LinePattern::Solid,
                depth_mode: DepthMode::Overlay,
            };

            for submitted in &self.frame_lines {
                if submitted.style.depth_mode == DepthMode::Overlay {
                    line_style = submitted.style.clone();
                    for seg in &submitted.data.segments {
                        let start = submitted.transform.transform_point3(Vec3::from(seg.start));
                        let end = submitted.transform.transform_point3(Vec3::from(seg.end));
                        all_line_segments.push(LineSegment {
                            start: start.into(),
                            end: end.into(),
                            color: seg.color,
                        });
                    }
                }
            }

            // Collect and prepare all overlay point instances
            let mut all_point_instances: Vec<PointInstance> = Vec::new();
            let mut point_style = PointStyle {
                size: 4.0,
                size_mode: WidthMode::ScreenSpace,
                shape: PointShape::Circle,
                depth_mode: DepthMode::Overlay,
            };

            for submitted in &self.frame_points {
                if submitted.style.depth_mode == DepthMode::Overlay {
                    point_style = submitted.style.clone();
                    for pt in &submitted.data.points {
                        let pos = submitted.transform.transform_point3(Vec3::from(pt.position));
                        all_point_instances.push(PointInstance {
                            position: pos.into(),
                            color: pt.color,
                        });
                    }
                }
            }

            // Upload all data before starting the render pass (split borrow)
            let GpuResources {
                line_pipeline,
                point_pipeline,
                gbuffer,
                ..
            } = gpu;

            if !all_line_segments.is_empty() {
                let instances = LinePipeline::prepare_instances(&all_line_segments, &line_style);
                line_pipeline.upload_instances(device, queue, &instances);
                let uniforms = LinePipeline::create_uniforms(view_proj_array, screen_size, &line_style);
                line_pipeline.update_uniforms(queue, &uniforms);
            }

            if !all_point_instances.is_empty() {
                let instances = PointPipeline::prepare_instances(&all_point_instances);
                point_pipeline.upload_instances(device, queue, &instances);
                let uniforms = PointPipeline::create_uniforms(view_proj_array, screen_size, &point_style);
                point_pipeline.update_uniforms(queue, &uniforms);
            }

            // Now start the render pass
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gbuffer.depth_stencil_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render all uploaded data
            if !all_line_segments.is_empty() {
                line_pipeline.render(&mut pass, DepthMode::Overlay);
            }
            if !all_point_instances.is_empty() {
                point_pipeline.render(&mut pass, DepthMode::Overlay);
            }
        }
    }

    /// Clear frame state for next frame.
    pub fn end_frame(&mut self) {
        self.frame_meshes.clear();
        self.frame_lines.clear();
        self.frame_points.clear();
    }

    /// Update the grid line cache if settings have changed.
    fn update_grid_cache(&mut self, settings: &RenderSettings) {
        let hash = self.hash_grid_settings(&settings.grid);

        if hash != self.cached_grid_settings_hash {
            self.cached_grid_lines = settings.grid.generate_lines();
            self.cached_grid_settings_hash = hash;
        }
    }

    /// Compute a simple hash of grid settings for change detection.
    fn hash_grid_settings(&self, settings: &GridSettings) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        settings.planes.xy.hash(&mut hasher);
        settings.planes.xz.hash(&mut hasher);
        settings.planes.yz.hash(&mut hasher);
        settings.spacing.to_bits().hash(&mut hasher);
        settings.extent.to_bits().hash(&mut hasher);
        settings.subdivisions.hash(&mut hasher);

        hasher.finish()
    }

    /// Get the number of meshes submitted this frame.
    pub fn mesh_count(&self) -> usize {
        self.frame_meshes.len()
    }

    /// Get the number of line batches submitted this frame.
    pub fn line_batch_count(&self) -> usize {
        self.frame_lines.len()
    }

    /// Get the number of point batches submitted this frame.
    pub fn point_batch_count(&self) -> usize {
        self.frame_points.len()
    }

    /// Get total number of line segments submitted this frame.
    pub fn line_segment_count(&self) -> usize {
        self.frame_lines.iter().map(|l| l.data.segments.len()).sum()
    }

    /// Get total number of points submitted this frame.
    pub fn point_count(&self) -> usize {
        self.frame_points.iter().map(|p| p.data.points.len()).sum()
    }

    /// Get the number of grid lines.
    pub fn grid_line_count(&self) -> usize {
        self.cached_grid_lines.len()
    }
}

/// Generate axis indicator lines for rendering.
///
/// Creates three line segments representing the X, Y, and Z axes,
/// transformed by the camera's rotation for display in a corner viewport.
pub fn generate_axis_indicator_lines(camera: &Camera, indicator: &AxisIndicator) -> Vec<LineSegment> {
    // Get the camera's view rotation
    let view = camera.view_matrix();

    // Extract rotation from view matrix (inverse of camera rotation)
    // We want to show world axes as seen from the camera's perspective
    let (_scale, rotation, _translation) = view.to_scale_rotation_translation();
    let inv_rotation = rotation.inverse();

    // Transform world axes to view space
    let x_dir = inv_rotation * Vec3::X;
    let y_dir = inv_rotation * Vec3::Y;
    let z_dir = inv_rotation * Vec3::Z;

    let origin = Vec3::ZERO;
    let len = 1.0; // Normalized length, will be scaled by viewport

    vec![
        LineSegment {
            start: origin.into(),
            end: (origin + x_dir * len).into(),
            color: indicator.x_color,
        },
        LineSegment {
            start: origin.into(),
            end: (origin + y_dir * len).into(),
            color: indicator.y_color,
        },
        LineSegment {
            start: origin.into(),
            end: (origin + z_dir * len).into(),
            color: indicator.z_color,
        },
    ]
}

/// Compute an orthographic projection for the axis indicator.
pub fn axis_indicator_projection() -> Mat4 {
    Mat4::orthographic_rh(-1.5, 1.5, -1.5, 1.5, -10.0, 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let renderer = Renderer::new(wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(renderer.viewport_size(), (1, 1));
        assert_eq!(renderer.mesh_count(), 0);
        assert!(!renderer.is_initialized());
    }

    #[test]
    fn test_submit_mesh() {
        let mut renderer = Renderer::new(wgpu::TextureFormat::Bgra8Unorm);

        let mesh = MeshData {
            vertices: vec![MeshVertex {
                position: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            }],
            indices: None,
        };

        renderer.submit_mesh(&mesh, Mat4::IDENTITY, MaterialId(0));
        assert_eq!(renderer.mesh_count(), 1);

        renderer.end_frame();
        assert_eq!(renderer.mesh_count(), 0);
    }

    #[test]
    fn test_grid_generation() {
        let settings = GridSettings::default();
        let lines = settings.generate_lines();

        // With default settings (extent=10, spacing=1), we should have
        // 21 lines in each direction (from -10 to +10) times 2 (X and Z parallel)
        // = 42 lines on XZ plane
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_axis_indicator_generation() {
        let camera = Camera::default();
        let indicator = AxisIndicator::default();
        let lines = generate_axis_indicator_lines(&camera, &indicator);

        assert_eq!(lines.len(), 3); // X, Y, Z axes
    }
}
