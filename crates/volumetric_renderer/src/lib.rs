//! The rendering engine shared by the GUI viewport and headless renders.
//!
//! A frame is deferred: geometry sources fill a G-buffer (albedo, normal,
//! object id, depth) with no lighting; later passes read it. The design,
//! including what is not built yet, is in `RENDERING_ARCHITECTURE.md`.
//!
//! Passes, in order:
//!
//! 1. **G-buffer fill**: retained meshes.
//! 2. **Ambient occlusion** from depth and normals.
//! 3. **Resolve**: lights the G-buffer into the target.
//! 4. **Depth-tested lines and points** (the grid among them).
//! 5. **Splats**: their own layer, then composited.
//! 6. **Overlay lines and points** (no depth test).
//! 7. **Lens warp**, when the frame is drawn through a real lens.
//!
//! The G-buffer outlives the frame: [`Renderer::request_pick`] reads the
//! object and world point under a pixel of the last frame.
//!
//! ```ignore
//! let mut renderer = Renderer::new(&device, surface_format);
//! let mesh = renderer.create_retained_mesh(&device, &mesh_data);
//!
//! // Each frame:
//! renderer.submit_retained_mesh(&mesh, transform, object, material);
//! renderer.submit_lines(&line_data, transform, style);
//! let info = renderer.render(&device, &queue, &mut encoder, &view, &settings, &target);
//! ```

mod buffer;
mod camera;
mod conversions;
mod gbuffer;
mod navigation;
#[cfg(not(target_arch = "wasm32"))]
pub mod offscreen;
mod pick;
mod pipelines;
mod scene;
pub mod test_scenes;
mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod frame_tests;

pub use conversions::{convert_mesh_data, convert_points_to_point_data};

pub use camera::{Camera, CameraView, OrbitMode, Pinhole, Projection, StandardView};
pub use navigation::{CameraAction, CameraControlScheme, CameraInputState, Cursor, Navigator};
pub use pick::Pick;
pub use pipelines::{
    GpuLines, GpuMesh, GpuPoints, GpuSplat, Warp, evaluate_sh, project_covariance,
};
pub use scene::SceneData;
pub use types::{
    DepthMode, GridPlanes, GridSettings, LineData, LineInstance, LinePattern, LineSegment,
    LineStyle, MaterialId, MeshData, MeshVertex, ObjectId, PointData, PointInstance, PointShape,
    PointStyle, RenderSettings, SplatData, SplatStyle, WidthMode,
};

use std::sync::Arc;

use buffer::{DynamicBuffer, QUAD_INDICES, QUAD_VERTICES, QuadVertex, StaticBuffer};
use bytemuck::Zeroable;
use gbuffer::GBuffer;
use glam::{Mat4, Vec3};
use pick::{FrameRecord, Picker};
use pipelines::{
    AoUniforms, FullscreenPass, GpuPointInstance, GpuWarp, LinePipeline, MeshDraw, MeshPipeline,
    PointPipeline, ResolveUniforms, SplatCompositePipeline, SplatPipeline, WarpPipeline, ao_pass,
    resolve_pass,
};

/// Geometry dropped from a frame because it would have exceeded the
/// device's `max_buffer_size` limit (creating a larger buffer is a wgpu
/// validation panic). The renderer keeps whatever fits and reports the
/// rest here; hosts should surface it so a silently sparse viewport is
/// explainable.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeometryOverflow {
    /// Mesh triangles dropped, across immediate submissions (which drop
    /// whole meshes) and retained meshes (clamped at creation).
    pub dropped_triangles: usize,
    /// Total mesh triangles submitted this frame.
    pub total_triangles: usize,
    /// Line instances dropped across the depth-tested and overlay passes.
    pub dropped_lines: usize,
    /// Point instances dropped across the depth-tested and overlay passes.
    pub dropped_points: usize,
    /// Splat primitives dropped (retained splats are clamped at creation).
    pub dropped_splats: usize,
    /// The device's buffer size limit the frame was clamped to.
    pub max_buffer_bytes: u64,
}

impl GeometryOverflow {
    /// Whether anything was actually dropped.
    pub fn any(&self) -> bool {
        self.dropped_triangles > 0
            || self.dropped_lines > 0
            || self.dropped_points > 0
            || self.dropped_splats > 0
    }
}

/// One scene's geometry uploaded as retained GPU residents (transforms
/// applied at creation): the once-per-rebuild counterpart of submitting a
/// [`SceneData`] every frame. Hold it for as long as the scene should be
/// drawable and submit the handles each frame via
/// [`Renderer::submit_retained_mesh`] and friends.
#[derive(Default)]
pub struct RetainedScene {
    /// Each mesh with the transform it is drawn under.
    pub meshes: Vec<(Arc<GpuMesh>, Mat4)>,
    pub lines: Vec<Arc<GpuLines>>,
    pub points: Vec<Arc<GpuPoints>>,
    pub splats: Vec<Arc<GpuSplat>>,
}

/// What a rendered frame reports back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameInfo {
    /// Geometry dropped at the device's buffer size limit; `None` when
    /// everything fit.
    pub overflow: Option<GeometryOverflow>,
}

/// A line batch submitted for the current frame.
struct SubmittedLines {
    data: LineData,
    transform: Mat4,
    style: LineStyle,
}

/// A point batch submitted for the current frame.
struct SubmittedPoints {
    data: PointData,
    transform: Mat4,
    style: PointStyle,
}

/// The direction the fixed light shines from.
pub(crate) const LIGHT_DIRECTION: [f32; 3] = [0.4, -0.2, 0.7];

/// The grid's lines are their own batch, one pixel wide.
const GRID_STYLE: LineStyle = LineStyle {
    width: 1.0,
    width_mode: WidthMode::ScreenSpace,
    pattern: LinePattern::Solid,
    depth_mode: DepthMode::Normal,
};

fn clear_color(color: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: color[0] as f64,
        g: color[1] as f64,
        b: color[2] as f64,
        a: color[3] as f64,
    }
}

/// The renderer: pipelines, the G-buffer, and the geometry submitted for
/// the next frame.
pub struct Renderer {
    // Viewport size: the frame written
    viewport_size: (u32, u32),

    // The lens warp, when a frame is drawn through a real lens: the
    // scene renders into an internal frame of the warp's source size and
    // the last pass writes the viewport through the warp. GPU state is
    // built on first use and dropped when the warp changes.
    warp: Option<Warp>,
    warp_gpu: Option<GpuWarp>,

    mesh_pipeline: MeshPipeline,
    ao: FullscreenPass,
    resolve: FullscreenPass,
    line_pipeline: LinePipeline,
    point_pipeline: PointPipeline,
    splat_pipeline: SplatPipeline,
    splat_composite_pipeline: SplatCompositePipeline,
    warp_pipeline: WarpPipeline,
    picker: Picker,

    gbuffer: GBuffer,
    // Bind groups over the G-buffer's views (recreated on resize)
    gbuffer_bindings: GBufferBindings,

    // Geometry submitted for the current frame
    frame_meshes: Vec<MeshDraw>,
    frame_lines: Vec<SubmittedLines>,
    frame_points: Vec<SubmittedPoints>,
    frame_retained_lines: Vec<Arc<GpuLines>>,
    frame_retained_points: Vec<Arc<GpuPoints>>,
    frame_retained_splats: Vec<Arc<GpuSplat>>,

    // Grid line cache (regenerated when settings change)
    cached_grid_lines: Vec<LineSegment>,
    cached_grid_settings_hash: u64,

    // The view the G-buffer was last filled with, for picking.
    last_frame: Option<FrameRecord>,
    // A pick asked for while another was in flight; only the newest waits.
    queued_pick: Option<(u32, u32)>,
}

/// The bind groups of the passes that read the G-buffer.
struct GBufferBindings {
    ao: wgpu::BindGroup,
    resolve: wgpu::BindGroup,
    pick: wgpu::BindGroup,
    splat_composite: wgpu::BindGroup,
}

impl Renderer {
    /// Creates a renderer drawing into targets of `surface_format`.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let ao = ao_pass(device);
        let resolve = resolve_pass(device, surface_format);
        let splat_composite_pipeline = SplatCompositePipeline::new(device, surface_format);
        let picker = Picker::new(device);
        let gbuffer = GBuffer::new(device, 1, 1);
        let gbuffer_bindings = Self::bind_gbuffer(
            device,
            &gbuffer,
            &ao,
            &resolve,
            &picker,
            &splat_composite_pipeline,
        );
        Self {
            viewport_size: (1, 1),
            warp: None,
            warp_gpu: None,
            mesh_pipeline: MeshPipeline::new(device),
            ao,
            resolve,
            line_pipeline: LinePipeline::new(device, surface_format),
            point_pipeline: PointPipeline::new(device, surface_format),
            splat_pipeline: SplatPipeline::new(device, GBuffer::SPLAT_LAYER_FORMAT),
            splat_composite_pipeline,
            warp_pipeline: WarpPipeline::new(device, surface_format),
            picker,
            gbuffer,
            gbuffer_bindings,
            frame_meshes: Vec::new(),
            frame_lines: Vec::new(),
            frame_points: Vec::new(),
            frame_retained_lines: Vec::new(),
            frame_retained_points: Vec::new(),
            frame_retained_splats: Vec::new(),
            cached_grid_lines: Vec::new(),
            cached_grid_settings_hash: 0,
            last_frame: None,
            queued_pick: None,
        }
    }

    fn bind_gbuffer(
        device: &wgpu::Device,
        gbuffer: &GBuffer,
        ao: &FullscreenPass,
        resolve: &FullscreenPass,
        picker: &Picker,
        splat_composite: &SplatCompositePipeline,
    ) -> GBufferBindings {
        GBufferBindings {
            ao: ao.bind(device, &[&gbuffer.normal_view, &gbuffer.surface_view]),
            resolve: resolve.bind(
                device,
                &[
                    &gbuffer.albedo_view,
                    &gbuffer.normal_view,
                    &gbuffer.surface_view,
                    &gbuffer.ao_view,
                ],
            ),
            pick: picker.bind(device, &gbuffer.surface_view),
            splat_composite: splat_composite.create_bind_group(device, &gbuffer.splat_view),
        }
    }

    /// Set the viewport size. Call when the window is resized.
    pub fn set_viewport_size(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let new_size = (width.max(1), height.max(1));
        if self.viewport_size == new_size {
            return;
        }
        self.viewport_size = new_size;
        self.resize_internal(device);
    }

    /// Draw the next frames through `warp` (`None`: straight to the
    /// viewport). The viewport must be the warp's output size when a
    /// frame is rendered; the scene renders at its source size. An
    /// invalid warp is refused and the frames go straight.
    pub fn set_warp(&mut self, device: &wgpu::Device, warp: Option<&Warp>) -> Result<(), String> {
        if self.warp.as_ref() == warp {
            return Ok(());
        }
        let checked = match warp {
            Some(warp) => warp.validate().map(|()| Some(warp.clone())),
            None => Ok(None),
        };
        self.warp = checked.clone().unwrap_or(None);
        self.warp_gpu = None;
        self.resize_internal(device);
        checked.map(|_| ())
    }

    pub fn warp(&self) -> Option<&Warp> {
        self.warp.as_ref()
    }

    /// The size the scene renders at: the warp's source, else the viewport.
    fn internal_size(&self) -> (u32, u32) {
        self.warp
            .as_ref()
            .map_or(self.viewport_size, |warp| warp.source)
    }

    /// Size the G-buffer to the internal size and rebind. Its old contents
    /// go with it, so there is nothing to pick until the next frame.
    fn resize_internal(&mut self, device: &wgpu::Device) {
        let (width, height) = self.internal_size();
        if !self.gbuffer.resize_if_needed(device, width, height) {
            return;
        }
        self.gbuffer_bindings = Self::bind_gbuffer(
            device,
            &self.gbuffer,
            &self.ao,
            &self.resolve,
            &self.picker,
            &self.splat_composite_pipeline,
        );
        self.last_frame = None;
    }

    /// Get the current viewport size: the frame written.
    pub fn viewport_size(&self) -> (u32, u32) {
        self.viewport_size
    }

    /// Submit line segments for this frame, drawn with their own style.
    /// For small dynamic batches; retain anything large.
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

    /// Submit points for this frame, drawn with their own style. For
    /// small dynamic batches; retain anything large.
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

    /// Uploads a scene's geometry as retained GPU residents (line, point
    /// and splat transforms applied now, on the CPU, once; a mesh keeps
    /// its transform for submission).
    pub fn create_retained_scene(&self, device: &wgpu::Device, scene: &SceneData) -> RetainedScene {
        RetainedScene {
            meshes: scene
                .meshes
                .iter()
                .map(|(mesh, transform, _)| (self.create_retained_mesh(device, mesh), *transform))
                .collect(),
            lines: scene
                .lines
                .iter()
                .map(|(lines, transform, style)| {
                    self.create_retained_lines(device, lines, *transform, style)
                })
                .collect(),
            points: scene
                .points
                .iter()
                .map(|(points, transform, style)| {
                    Arc::new(self.point_pipeline.create_retained(
                        device,
                        &points.points,
                        *transform,
                        style,
                    ))
                })
                .collect(),
            splats: scene
                .splats
                .iter()
                .map(|(splat, transform, style)| {
                    Arc::new(self.splat_pipeline.create_retained(
                        device,
                        splat.clone(),
                        *transform,
                        style,
                    ))
                })
                .collect(),
        }
    }

    /// Uploads one line batch as a retained GPU resident.
    pub fn create_retained_lines(
        &self,
        device: &wgpu::Device,
        lines: &LineData,
        transform: Mat4,
        style: &LineStyle,
    ) -> Arc<GpuLines> {
        Arc::new(
            self.line_pipeline
                .create_retained(device, &lines.segments, transform, style),
        )
    }

    /// Uploads one mesh as a retained GPU resident, its vertices as
    /// given; the transform comes at submission.
    pub fn create_retained_mesh(&self, device: &wgpu::Device, mesh: &MeshData) -> Arc<GpuMesh> {
        Arc::new(GpuMesh::new(device, mesh))
    }

    /// Submit a retained mesh for this frame, drawn under `transform`.
    /// `object` is what a pick at its pixels reports.
    pub fn submit_retained_mesh(
        &mut self,
        mesh: &Arc<GpuMesh>,
        transform: Mat4,
        object: ObjectId,
        material: MaterialId,
    ) {
        self.frame_meshes.push(MeshDraw {
            mesh: mesh.clone(),
            transform,
            object,
            material,
        });
    }

    /// Submit a retained line batch for this frame.
    pub fn submit_retained_lines(&mut self, lines: &Arc<GpuLines>) {
        self.frame_retained_lines.push(lines.clone());
    }

    /// Submit a retained point batch for this frame.
    pub fn submit_retained_points(&mut self, points: &Arc<GpuPoints>) {
        self.frame_retained_points.push(points.clone());
    }

    /// Submit a retained splat for this frame. It is drawn after the
    /// depth-tested lines and points, blended back to front over them.
    pub fn submit_retained_splat(&mut self, splat: &Arc<GpuSplat>) {
        self.frame_retained_splats.push(splat.clone());
    }

    /// Waits for every submitted splat's order to be current for `view`:
    /// what a frame that is read back needs, where a shown frame would
    /// take the previous order and catch up.
    pub fn settle_splats(&self, queue: &wgpu::Queue, view: &CameraView) {
        for splat in &self.frame_retained_splats {
            splat.settle_for(queue, view);
        }
    }

    /// Renders the submitted geometry with `view` into `target` and
    /// forgets the submissions. `view` is explicit matrices: the orbit
    /// camera's, a posed photograph's pinhole, or any other.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &CameraView,
        settings: &RenderSettings,
        target: &wgpu::TextureView,
    ) -> FrameInfo {
        self.update_grid_cache(settings);

        // Buffers larger than the device limit are a wgpu validation panic,
        // so every geometry class is clamped to it; whatever gets dropped
        // is reported. Retained geometry was clamped at creation.
        let mut overflow = GeometryOverflow {
            max_buffer_bytes: device.limits().max_buffer_size,
            ..GeometryOverflow::default()
        };
        for draw in &self.frame_meshes {
            overflow.dropped_triangles += draw.mesh.dropped_triangles;
            overflow.total_triangles += draw.mesh.total_triangles;
        }
        for lines in &self.frame_retained_lines {
            overflow.dropped_lines += lines.dropped;
        }
        for points in &self.frame_retained_points {
            overflow.dropped_points += points.dropped;
        }
        for splat in &self.frame_retained_splats {
            overflow.dropped_splats += splat.dropped;
        }

        // Through a lens the scene renders into the warp's own frame and
        // the last pass writes the target; otherwise straight to it.
        let internal_size = self.internal_size();
        if let (Some(warp), None) = (&self.warp, &self.warp_gpu) {
            self.warp_gpu = Some(self.warp_pipeline.create(device, queue, warp));
        }
        let final_target = target;
        let target: &wgpu::TextureView = match &self.warp_gpu {
            Some(warp) => &warp.frame_view,
            None => target,
        };

        let view_proj = view.view_projection();
        let view_proj_array = view_proj.to_cols_array_2d();
        let screen_size = [internal_size.0 as f32, internal_size.1 as f32];

        // ---- Uploads. Everything is written before any pass is encoded;
        // each batch has buffers of its own.
        self.mesh_pipeline
            .prepare(device, queue, view_proj, &self.frame_meshes);

        self.line_pipeline.begin_frame();
        let grid: Vec<LineInstance> = self
            .cached_grid_lines
            .iter()
            .map(|segment| LineInstance::from_segment(segment, GRID_STYLE.width))
            .collect();
        overflow.dropped_lines += self.line_pipeline.upload_immediate(
            device,
            queue,
            &grid,
            &GRID_STYLE,
            view_proj_array,
            screen_size,
        );
        for submitted in &self.frame_lines {
            let instances: Vec<LineInstance> = submitted
                .data
                .segments
                .iter()
                .map(|segment| {
                    let world = LineSegment {
                        start: submitted
                            .transform
                            .transform_point3(Vec3::from(segment.start))
                            .into(),
                        end: submitted
                            .transform
                            .transform_point3(Vec3::from(segment.end))
                            .into(),
                        color: segment.color,
                    };
                    LineInstance::from_segment(&world, submitted.style.width)
                })
                .collect();
            overflow.dropped_lines += self.line_pipeline.upload_immediate(
                device,
                queue,
                &instances,
                &submitted.style,
                view_proj_array,
                screen_size,
            );
        }

        self.point_pipeline.begin_frame();
        for submitted in &self.frame_points {
            let instances: Vec<GpuPointInstance> = submitted
                .data
                .points
                .iter()
                .map(|point| GpuPointInstance {
                    position: submitted
                        .transform
                        .transform_point3(Vec3::from(point.position))
                        .into(),
                    _pad: 0.0,
                    color: point.color,
                })
                .collect();
            overflow.dropped_points += self.point_pipeline.upload_immediate(
                device,
                queue,
                &instances,
                &submitted.style,
                view_proj_array,
                screen_size,
            );
        }

        for batch in &self.frame_retained_lines {
            self.line_pipeline
                .write_retained_uniforms(queue, batch, view_proj_array, screen_size);
        }
        for batch in &self.frame_retained_points {
            self.point_pipeline
                .write_retained_uniforms(queue, batch, view_proj_array, screen_size);
        }
        for splat in &self.frame_retained_splats {
            self.splat_pipeline
                .prepare_retained(queue, splat, view, screen_size);
        }

        let ao_enabled = settings.ssao_enabled && !self.frame_meshes.is_empty();
        if ao_enabled {
            self.ao.write_uniforms(
                queue,
                &AoUniforms {
                    view_proj: view_proj_array,
                    inv_view_proj: view_proj.inverse().to_cols_array_2d(),
                    radius: settings.ssao_radius,
                    bias: settings.ssao_bias,
                    strength: settings.ssao_strength,
                    ..AoUniforms::zeroed()
                },
            );
        }
        self.resolve.write_uniforms(
            queue,
            &ResolveUniforms {
                // From above (+Z), a little to the +X, -Y side: the MVP's
                // light, carried over to a z-up world.
                light_dir_world: LIGHT_DIRECTION,
                ao_enabled: ao_enabled as u32,
                base_tint: [0.85, 0.9, 1.0],
                _pad0: 0.0,
            },
        );

        // ---- G-buffer fill. Always run, so a frame with no geometry
        // still leaves a cleared G-buffer for the later passes and picks.
        {
            let attachment = |view, clear| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gbuffer_pass"),
                color_attachments: &[
                    attachment(&self.gbuffer.albedo_view, wgpu::Color::TRANSPARENT),
                    attachment(&self.gbuffer.normal_view, wgpu::Color::TRANSPARENT),
                    attachment(&self.gbuffer.surface_view, GBuffer::SURFACE_CLEAR),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gbuffer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.mesh_pipeline.render(&mut pass, &self.frame_meshes);
        }

        // ---- Ambient occlusion, then the resolve onto the background.
        if ao_enabled {
            self.ao.run(
                encoder,
                &self.gbuffer_bindings.ao,
                &self.gbuffer.ao_view,
                wgpu::Color::WHITE,
            );
        }
        self.resolve.run(
            encoder,
            &self.gbuffer_bindings.resolve,
            target,
            clear_color(settings.background_color),
        );

        // ---- Depth-tested lines and points, over the lit scene.
        self.draw_lines_and_points(encoder, target, DepthMode::Normal);

        // ---- Splats: sorted back to front, depth-tested at their centres
        // against everything drawn so far, blended over it.
        if !self.frame_retained_splats.is_empty() {
            // Into the layer: the trainer's arithmetic, sRGB values blended
            // as numbers.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("splat_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuffer.splat_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.gbuffer.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                for splat in &self.frame_retained_splats {
                    self.splat_pipeline.render_retained(&mut pass, splat);
                }
            }
            // Over the scene, linearised for the target.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("splat_composite_pass"),
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
                self.splat_composite_pipeline
                    .render(&mut pass, &self.gbuffer_bindings.splat_composite);
            }
        }

        // ---- Overlay lines and points (no depth test).
        self.draw_lines_and_points(encoder, target, DepthMode::Overlay);

        // ---- The lens warp: the internal frame written to the target.
        if let Some(warp) = &self.warp_gpu {
            self.warp_pipeline.render(
                queue,
                encoder,
                warp,
                settings.background_color,
                final_target,
            );
        }

        self.last_frame = Some(FrameRecord {
            inv_view_proj: view_proj.inverse(),
            size: internal_size,
        });
        self.frame_meshes.clear();
        self.frame_lines.clear();
        self.frame_points.clear();
        self.frame_retained_lines.clear();
        self.frame_retained_points.clear();
        self.frame_retained_splats.clear();

        FrameInfo {
            overflow: overflow.any().then_some(overflow),
        }
    }

    /// One pass over `target` drawing every line and point batch of
    /// `depth_mode`, immediate then retained.
    fn draw_lines_and_points(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth_mode: DepthMode,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(match depth_mode {
                DepthMode::Normal => "forward_pass",
                DepthMode::Overlay => "overlay_pass",
            }),
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
                view: &self.gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.line_pipeline.render_immediate(&mut pass, depth_mode);
        self.point_pipeline.render_immediate(&mut pass, depth_mode);
        for batch in &self.frame_retained_lines {
            if batch.depth_mode() == depth_mode {
                self.line_pipeline.render_retained(&mut pass, batch);
            }
        }
        for batch in &self.frame_retained_points {
            if batch.depth_mode() == depth_mode {
                self.point_pipeline.render_retained(&mut pass, batch);
            }
        }
    }

    /// Asks what the last rendered frame drew at `pixel` of its target
    /// (origin top-left). The answer arrives through
    /// [`pick_result`](Self::pick_result) a moment later; asking again
    /// before then replaces the question still waiting. Call between
    /// frames, after the frame's commands were submitted: the read is
    /// submitted at once and sees the G-buffer as it then stands.
    pub fn request_pick(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, pixel: (u32, u32)) {
        if self.picker.busy() {
            self.queued_pick = Some(pixel);
            return;
        }
        let Some(frame) = self.last_frame else {
            return;
        };
        // Through a lens the target pixel shows some other pixel of the
        // internal frame, or none.
        let source = match &self.warp {
            Some(warp) => {
                let Some([x, y]) = warp.lookup(pixel.0 as f32 + 0.5, pixel.1 as f32 + 0.5) else {
                    return;
                };
                if x < 0.0 || y < 0.0 {
                    return;
                }
                (x as u32, y as u32)
            }
            None => pixel,
        };
        if source.0 >= frame.size.0 || source.1 >= frame.size.1 {
            return;
        }
        self.picker.request(
            device,
            queue,
            &self.gbuffer_bindings.pick,
            frame,
            pixel,
            source,
        );
    }

    /// The most recent pick to have landed, without waiting; `None` until
    /// the first does. Its `pixel` says which request it answers.
    pub fn pick_result(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Pick> {
        self.picker.poll(device);
        if !self.picker.busy()
            && let Some(pixel) = self.queued_pick.take()
        {
            self.request_pick(device, queue, pixel);
        }
        self.picker.latest()
    }

    /// What the last rendered frame drew at `pixel`, waiting for the
    /// answer: for a press that must know now what it landed on. `None`
    /// when there is no frame or the pixel is outside it.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pick_now(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pixel: (u32, u32),
    ) -> Option<Pick> {
        self.last_frame?;
        // The read asked for here may be queued behind one in flight:
        // landing that one issues ours, which the second wait lands.
        self.request_pick(device, queue, pixel);
        for _ in 0..2 {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(pick) = self.pick_result(device, queue)
                && pick.pixel == pixel
                && !self.picker.busy()
            {
                return Some(pick);
            }
        }
        None
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
}
