//! The rendering engine shared by the GUI viewport and headless renders.
//!
//! A frame is deferred: geometry sources fill a G-buffer (albedo, normal,
//! object id, depth) with no lighting; later passes read it. The design,
//! including what is not built yet, is in `RENDERING_ARCHITECTURE.md`.
//!
//! Passes, in order:
//!
//! 1. **G-buffer fill**: the opaque retained meshes.
//! 2. **Ambient occlusion** from depth and normals, then its blur.
//! 3. **Resolve**: lights the G-buffer and draws edge lines.
//! 4. **Transparent meshes**, when the frame has any: peeled a layer at
//!    a time, nearest first, each filled and resolved like the G-buffer,
//!    then laid over the lit scene together. (Every layer's surface
//!    target is filled before the resolve: edge lines read them.)
//! 5. **Anti-aliasing** writes the lit scene into the target.
//! 6. **Grid**: the ground grid and world axis lines, per pixel.
//! 7. **Depth-tested lines and points**.
//! 8. **Splats**: their own layer, then composited.
//! 9. **Overlay lines and points** (no depth test).
//! 10. **Lens warp**, when the frame is drawn through a real lens.
//! 11. **View gizmo**, over the finished frame.
//!
//! The G-buffer outlives the frame: [`Renderer::request_pick`] reads the
//! object and world point under a pixel of the last frame.
//!
//! ```ignore
//! let mut renderer = Renderer::new(&device, surface_format);
//! let mesh = renderer.create_retained_mesh(&device, &queue, &mesh_data);
//!
//! // Each frame:
//! renderer.submit_retained_mesh(&mesh, transform, object, material, opacity);
//! renderer.submit_lines(&line_data, transform, style);
//! let info = renderer.render(&device, &queue, &mut encoder, &view, &settings, &target);
//! ```

mod buffer;
mod camera;
mod conversions;
mod gbuffer;
mod gizmo;
mod navigation;
#[cfg(not(target_arch = "wasm32"))]
pub mod offscreen;
mod pick;
mod pipelines;
mod scene;
pub mod test_scenes;
mod texture;
mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod frame_tests;

pub use conversions::{convert_mesh_data, convert_points_to_point_data};
pub use gizmo::{GizmoEnd, GizmoPart, ViewGizmo};

pub use camera::{Camera, CameraView, OrbitMode, Pinhole, Projection, StandardView};
pub use navigation::{CameraAction, CameraControlScheme, CameraInputState, Cursor, Navigator};
pub use pick::Pick;
pub use pipelines::{
    GpuLines, GpuMesh, GpuPoints, GpuSplat, GpuSurfels, GpuTexture, GpuTextureSet,
    MeshTextureBinding, Warp, evaluate_sh, project_covariance,
};
pub use scene::SceneData;
pub use types::{
    AXIS_COLORS, AoSettings, DepthMode, EdgeSettings, GridPlane, GridSettings, GridSpacing, Light,
    LightingPreset, LightingRig, LineData, LineInstance, LinePattern, LineSegment, LineStyle,
    MAX_MATERIALS, Material, MaterialId, MeshData, MeshVertex, ObjectId, PointData, PointInstance,
    PointShape, PointStyle, RenderSettings, SceneTexture, SplatData, SplatStyle, SurfelData,
    SurfelVertex, TextureData, WidthMode,
};

use std::sync::Arc;

use buffer::{DynamicBuffer, QUAD_INDICES, QUAD_VERTICES, QuadVertex, StaticBuffer};
use gbuffer::{GBuffer, LayerBuffers};
use glam::{Mat4, Vec3};
use pick::{FrameRecord, Picker};
use pipelines::{
    AoBlurUniforms, AoUniforms, FullscreenPass, FxaaUniforms, GizmoPipeline, GpuPointInstance,
    GpuWarp, GridPipeline, LinePipeline, MeshDraw, MeshPipeline, OTHER_LAYERS, PointPipeline,
    ResolveUniforms, SplatCompositePipeline, SplatPipeline, SurfelDraw, SurfelPipeline,
    WarpPipeline, ao_pass, layer_composite_pass, layer_resolve_pass, resolve_pass,
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
    /// Surfels dropped (clamped at creation).
    pub dropped_surfels: usize,
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
            || self.dropped_surfels > 0
    }
}

// A scene is plain data, built on any thread and sent to the device's to
// be uploaded. These fail the build where something in it is tied to a
// thread (a wgpu object, on threaded wasm).
const _: () = {
    const fn send_and_sync<T: Send + Sync>() {}
    send_and_sync::<SceneData>();
    send_and_sync::<SceneTexture>();
};

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
    /// The distance between the grid's minor lines in world units, when
    /// the grid was drawn: what a scale readout shows.
    pub grid_spacing: Option<f32>,
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

/// The resolve pass's uniforms: the rig's lights turned from the camera's
/// frame into the world, and the material table.
fn resolve_uniforms(
    settings: &RenderSettings,
    view: &CameraView,
    inv_view_proj: [[f32; 4]; 4],
    ao_enabled: bool,
    layers: u32,
) -> ResolveUniforms {
    let [right, up, toward] = view.frame_axes();
    let rgb = |[r, g, b]: [f32; 3]| [r, g, b, 0.0];
    let lights = settings.lighting.lights.map(|light| {
        let [x, y, z] = light.direction;
        let direction = (right * x + up * y + toward * z).normalize_or_zero();
        [direction.extend(0.0).to_array(), rgb(light.color)]
    });
    let first = settings.materials.first().copied().unwrap_or_default();
    let materials = std::array::from_fn(|i| {
        let material = settings.materials.get(i).copied().unwrap_or(first);
        // A Blinn-Phong exponent from roughness: tight at 0, broad at 1.
        let roughness = material.roughness.clamp(0.05, 1.0);
        let exponent = (2.0 / roughness.powi(4) - 2.0).clamp(1.0, 4000.0);
        [
            rgb(material.base_tint),
            [exponent, material.specular.max(0.0), 0.0, 0.0],
        ]
    });
    let edges = &settings.edges;
    let [r, g, b] = edges.color;
    ResolveUniforms {
        inv_view_proj,
        depth_to_distance: view.depth_to_distance(),
        lights,
        sky: rgb(settings.lighting.sky),
        ground: rgb(settings.lighting.ground),
        edge_color: [
            r,
            g,
            b,
            if edges.enabled {
                edges.opacity.clamp(0.0, 1.0)
            } else {
                0.0
            },
        ],
        switches: [
            edges.crease_angle.cos(),
            ao_enabled as u32 as f32,
            settings.pixel_scale.max(1.0).round(),
            layers as f32,
        ],
        materials,
    }
}

/// The least opacity a mesh is drawn at. Below it a mesh cannot change a
/// pixel of an 8-bit sRGB frame by half a level, whatever it is drawn
/// over: half a level is `0.5 / 255 / 12.92` in linear light at the dark
/// end, where the levels are finest.
const MIN_OPACITY: f32 = 0.5 / 255.0 / 12.92;

/// The layers' targets are dropped by this many frames in a row without
/// a transparent mesh: they outlive one fewer. They are 17 bytes a pixel
/// and 8 more a layer, at the frame's internal size. Frames are what is
/// counted, so a host that stops drawing keeps them until it draws
/// again.
const LAYERS_KEPT_FRAMES: u32 = 30;

/// A colour attachment cleared to `clear` when its pass begins.
fn cleared(
    view: &wgpu::TextureView,
    clear: wgpu::Color,
) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(clear),
            store: wgpu::StoreOp::Store,
        },
        depth_slice: None,
    })
}

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
    surfel_pipeline: SurfelPipeline,
    ao: FullscreenPass,
    ao_blur: FullscreenPass,
    /// How many occlusion samples a pixel takes.
    ao_samples: u32,
    resolve: FullscreenPass,
    /// The resolve pass again, for a layer of the transparent meshes.
    layer_resolve: FullscreenPass,
    layer_composite: FullscreenPass,
    fxaa: FullscreenPass,
    grid_pipeline: GridPipeline,
    gizmo_pipeline: GizmoPipeline,
    line_pipeline: LinePipeline,
    point_pipeline: PointPipeline,
    splat_pipeline: SplatPipeline,
    splat_composite_pipeline: SplatCompositePipeline,
    warp_pipeline: WarpPipeline,
    picker: Picker,

    gbuffer: GBuffer,
    // Bind groups over the G-buffer's views (recreated on resize)
    gbuffer_bindings: GBufferBindings,
    // What the transparent meshes are drawn through: made by the first
    // frame that has one, dropped when the frame's size changes or by the
    // `LAYERS_KEPT_FRAMES`th frame in a row without one.
    layers: Option<Layers>,
    frames_without_layers: u32,

    // Geometry submitted for the current frame: the opaque meshes, then
    // the transparent ones.
    frame_meshes: Vec<MeshDraw>,
    frame_transparent: Vec<MeshDraw>,
    frame_surfels: Vec<SurfelDraw>,
    frame_lines: Vec<SubmittedLines>,
    frame_points: Vec<SubmittedPoints>,
    frame_retained_lines: Vec<Arc<GpuLines>>,
    frame_retained_points: Vec<Arc<GpuPoints>>,
    frame_retained_splats: Vec<Arc<GpuSplat>>,

    // The view the G-buffer was last filled with, for picking.
    last_frame: Option<FrameRecord>,
    // A pick asked for while another was in flight; only the newest waits.
    queued_pick: Option<(u32, u32)>,
}

/// The bind groups of the passes that read the G-buffer.
struct GBufferBindings {
    ao: wgpu::BindGroup,
    ao_blur: wgpu::BindGroup,
    resolve: wgpu::BindGroup,
    fxaa: wgpu::BindGroup,
    pick: wgpu::BindGroup,
    splat_composite: wgpu::BindGroup,
}

/// The transparent meshes' targets and the bind groups over them and
/// the G-buffer.
struct Layers {
    buffers: LayerBuffers,
    /// Per layer: what its fill is peeled against.
    peel: Vec<wgpu::BindGroup>,
    /// Per layer: what its resolve reads.
    resolve: Vec<wgpu::BindGroup>,
    /// The opaque scene's resolve, reading the nearest layer as well.
    opaque_resolve: wgpu::BindGroup,
    composite: wgpu::BindGroup,
    /// The pick pass over the opaque scene and every layer.
    pick: wgpu::BindGroup,
}

impl Layers {
    /// How many layers these targets hold.
    fn count(&self) -> u32 {
        self.buffers.surface_views.len() as u32
    }
}

impl Renderer {
    /// Creates a renderer drawing into targets of `surface_format`.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let ao = ao_pass(device);
        let ao_blur = pipelines::ao_blur_pass(device);
        // The WebGL2 fallback takes half the samples.
        let ao_samples = if device.adapter_info().backend == wgpu::Backend::Gl {
            8
        } else {
            16
        };
        let resolve = resolve_pass(device, surface_format);
        let layer_resolve = layer_resolve_pass(device, surface_format);
        let layer_composite = layer_composite_pass(device, surface_format);
        let fxaa = pipelines::fxaa_pass(device, surface_format);
        let splat_composite_pipeline = SplatCompositePipeline::new(device, surface_format);
        let picker = Picker::new(device);
        let gbuffer = GBuffer::new(device, 1, 1, surface_format);
        let gbuffer_bindings = Self::bind_gbuffer(
            device,
            &gbuffer,
            &ao,
            &ao_blur,
            &resolve,
            &fxaa,
            &picker,
            &splat_composite_pipeline,
        );
        Self {
            viewport_size: (1, 1),
            warp: None,
            warp_gpu: None,
            mesh_pipeline: MeshPipeline::new(device),
            surfel_pipeline: SurfelPipeline::new(device),
            ao,
            ao_blur,
            ao_samples,
            resolve,
            layer_resolve,
            layer_composite,
            fxaa,
            grid_pipeline: GridPipeline::new(device, surface_format),
            gizmo_pipeline: GizmoPipeline::new(device, surface_format),
            line_pipeline: LinePipeline::new(device, surface_format),
            point_pipeline: PointPipeline::new(device, surface_format),
            splat_pipeline: SplatPipeline::new(device, GBuffer::SPLAT_LAYER_FORMAT),
            splat_composite_pipeline,
            warp_pipeline: WarpPipeline::new(device, surface_format),
            picker,
            gbuffer,
            gbuffer_bindings,
            layers: None,
            frames_without_layers: 0,
            frame_meshes: Vec::new(),
            frame_transparent: Vec::new(),
            frame_surfels: Vec::new(),
            frame_lines: Vec::new(),
            frame_points: Vec::new(),
            frame_retained_lines: Vec::new(),
            frame_retained_points: Vec::new(),
            frame_retained_splats: Vec::new(),
            last_frame: None,
            queued_pick: None,
        }
    }

    fn bind_gbuffer(
        device: &wgpu::Device,
        gbuffer: &GBuffer,
        ao: &FullscreenPass,
        ao_blur: &FullscreenPass,
        resolve: &FullscreenPass,
        fxaa: &FullscreenPass,
        picker: &Picker,
        splat_composite: &SplatCompositePipeline,
    ) -> GBufferBindings {
        GBufferBindings {
            ao: ao.bind(device, &[&gbuffer.normal_view, &gbuffer.surface_view]),
            ao_blur: ao_blur.bind(
                device,
                &[
                    &gbuffer.ao_raw_view,
                    &gbuffer.normal_view,
                    &gbuffer.surface_view,
                ],
            ),
            // A frame without transparent meshes has no nearest layer to
            // read and does not read one: the last two stand in.
            resolve: resolve.bind(
                device,
                &[
                    &gbuffer.albedo_view,
                    &gbuffer.normal_view,
                    &gbuffer.surface_view,
                    &gbuffer.ao_view,
                    &gbuffer.surface_view,
                    &gbuffer.ao_view,
                ],
            ),
            fxaa: fxaa.bind(device, &[&gbuffer.lit_view]),
            pick: picker.bind(device, &gbuffer.surface_view, &[]),
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
        // Bound over the old G-buffer; the next frame that needs them
        // makes them again.
        self.layers = None;
        self.gbuffer_bindings = Self::bind_gbuffer(
            device,
            &self.gbuffer,
            &self.ao,
            &self.ao_blur,
            &self.resolve,
            &self.fxaa,
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
    /// its transform for submission). The queue carries mesh textures;
    /// meshes that share one share its upload. A texture's copy on the
    /// device lives as long as the scene's meshes do: a host that drops a
    /// scene and makes it again keeps the copies with
    /// [`create_retained_scene_with`](Self::create_retained_scene_with).
    pub fn create_retained_scene(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &SceneData,
    ) -> RetainedScene {
        self.create_retained_scene_with(device, queue, scene, &mut GpuTextureSet::default())
    }

    /// [`create_retained_scene`](Self::create_retained_scene), its mesh
    /// textures taken from `textures` when they were uploaded for this
    /// renderer before and added to it when not, so a host that keeps the
    /// set can drop the scene and make it again without uploading them.
    pub fn create_retained_scene_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &SceneData,
        textures: &mut GpuTextureSet,
    ) -> RetainedScene {
        RetainedScene {
            meshes: scene
                .meshes
                .iter()
                .map(|(mesh, transform, _)| {
                    (
                        self.create_retained_mesh_with(device, queue, mesh, textures),
                        *transform,
                    )
                })
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
    /// given, with its texture (if any), which is uploaded for this mesh
    /// alone and freed with it; meshes share a texture's upload through
    /// [`create_retained_mesh_with`](Self::create_retained_mesh_with). The
    /// transform comes at submission.
    pub fn create_retained_mesh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshData,
    ) -> Arc<GpuMesh> {
        self.create_retained_mesh_with(device, queue, mesh, &mut GpuTextureSet::default())
    }

    /// [`create_retained_mesh`](Self::create_retained_mesh), its texture
    /// taken from `textures` when it was uploaded for this renderer
    /// before and added to it when not.
    pub fn create_retained_mesh_with(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshData,
        textures: &mut GpuTextureSet,
    ) -> Arc<GpuMesh> {
        Arc::new(
            self.mesh_pipeline
                .create_retained(device, queue, textures, mesh),
        )
    }

    /// Submit a retained mesh for this frame, drawn under `transform`.
    /// `object` is what a pick at its pixels reports.
    ///
    /// `opacity` is how much of what is behind the mesh it hides: 1 (or
    /// more) draws it opaque, as every mesh was before the renderer had
    /// transparency; 0 (or too little to change a pixel of an 8-bit
    /// frame) does not draw it at all, nor let it be picked. Between the two it is
    /// transparent: lit and edged like an opaque mesh, then blended at
    /// that weight over everything behind it, pixel by pixel in depth
    /// order, however the transparent meshes of a frame cross each
    /// other and whatever order they are submitted in (up to
    /// [`RenderSettings::transparency_layers`] surfaces deep). It does
    /// not occlude light (ambient occlusion). A pick lands on it when it
    /// is at least [`RenderSettings::pick_opacity`] opaque, and passes
    /// through it to what is behind when it is fainter.
    /// Changing it costs nothing: it is not part of the retained mesh.
    ///
    /// Where two meshes have a surface in exactly the same place and
    /// one of them is transparent, the one with the higher `object` is
    /// in front; give them different objects, or only one of two
    /// transparent ones is drawn there. (Of two opaque meshes the one
    /// submitted last is in front, as before: the same thing, for a host
    /// that numbers its meshes as it submits them.)
    pub fn submit_retained_mesh(
        &mut self,
        mesh: &Arc<GpuMesh>,
        transform: Mat4,
        object: ObjectId,
        material: MaterialId,
        opacity: f32,
    ) {
        if opacity < MIN_OPACITY {
            return;
        }
        let draw = MeshDraw {
            mesh: mesh.clone(),
            transform,
            object,
            material,
            opacity,
        };
        if opacity < 1.0 {
            self.frame_transparent.push(draw);
        } else {
            self.frame_meshes.push(draw);
        }
    }

    /// Uploads surfels as a retained GPU resident; the transform comes at
    /// submission.
    pub fn create_retained_surfels(
        &self,
        device: &wgpu::Device,
        surfels: &SurfelData,
    ) -> Arc<GpuSurfels> {
        Arc::new(GpuSurfels::new(device, surfels))
    }

    /// Submit retained surfels for this frame, drawn under `transform` as
    /// discs; they fill the G-buffer like a mesh, so they are lit, edged,
    /// occluded and picked the same way. `color` is multiplied into the
    /// material base colour for every disc, as a mesh vertex's colour is
    /// (white for the plain look). Its alpha is not read: surfels are
    /// always opaque, because the discs of one surface overlap and would
    /// each count as a surface of their own.
    pub fn submit_retained_surfels(
        &mut self,
        surfels: &Arc<GpuSurfels>,
        transform: Mat4,
        object: ObjectId,
        material: MaterialId,
        color: [f32; 4],
    ) {
        self.frame_surfels.push(SurfelDraw {
            surfels: surfels.clone(),
            transform,
            object,
            material,
            color,
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
        // Buffers larger than the device limit are a wgpu validation panic,
        // so every geometry class is clamped to it; whatever gets dropped
        // is reported. Retained geometry was clamped at creation.
        let mut overflow = GeometryOverflow {
            max_buffer_bytes: device.limits().max_buffer_size,
            ..GeometryOverflow::default()
        };
        for draw in self.frame_meshes.iter().chain(&self.frame_transparent) {
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
        for draw in &self.frame_surfels {
            overflow.dropped_surfels += draw.surfels.dropped;
        }

        let layered = !self.frame_transparent.is_empty();
        let layer_count = settings
            .transparency_layers
            .clamp(1, LayerBuffers::MAX_LAYERS);
        // How many layers this frame peels.
        let frame_layers = if layered { layer_count } else { 0 };
        if !layered {
            // A fade that reaches opaque and comes back finds its targets
            // standing; a scene that has stopped using them gives the
            // memory back.
            self.frames_without_layers = self.frames_without_layers.saturating_add(1);
            if self.frames_without_layers >= LAYERS_KEPT_FRAMES {
                self.layers = None;
            }
        }
        if layered {
            self.frames_without_layers = 0;
            self.ensure_layers(device, layer_count);
            // Rising object ids: of surfaces at exactly one depth, the
            // layer's depth test then keeps the highest id's (see
            // `peeled_away` in the mesh shader), so they are peeled in
            // an order of their own and not the order they were
            // submitted in.
            self.frame_transparent.sort_by_key(|draw| draw.object.0);
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
        let pixel_scale = settings.pixel_scale.max(1.0);
        // Lines and points measure their widths against this size, so a
        // supersampled frame is given the delivered picture's.
        let screen_size = [
            internal_size.0 as f32 / pixel_scale,
            internal_size.1 as f32 / pixel_scale,
        ];

        // ---- Uploads. Everything is written before any pass is encoded;
        // each batch has buffers of its own.
        self.mesh_pipeline.prepare(
            device,
            queue,
            view_proj,
            &self.frame_meshes,
            &self.frame_transparent,
        );
        self.surfel_pipeline
            .prepare(device, queue, view, &self.frame_surfels);

        self.line_pipeline.begin_frame();
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
            self.splat_pipeline.prepare_retained(
                queue,
                splat,
                view,
                [internal_size.0 as f32, internal_size.1 as f32],
            );
        }

        let grid_levels = settings.grid.visible.then(|| {
            self.grid_pipeline
                .prepare(queue, &settings.grid, view, internal_size, pixel_scale)
        });

        // ---- Shading uniforms. The inverse is taken in double
        // precision, as the grid's is.
        let inv_view_proj = view_proj.as_dmat4().inverse().as_mat4().to_cols_array_2d();
        let depth_to_distance = view.depth_to_distance();
        // Occlusion is sized by the meshes it is drawn on.
        let mesh_diagonal = self.mesh_diagonal();
        let ao_enabled = settings.ao.enabled && mesh_diagonal > 0.0;
        if ao_enabled {
            let (pixels_per_unit_at_1, pixels_per_unit) = view.pixels_per_unit(internal_size.1);
            self.ao.write_uniforms(
                queue,
                &AoUniforms {
                    view_proj: view_proj_array,
                    inv_view_proj,
                    depth_to_distance,
                    radius: settings.ao.radius.max(0.0) * mesh_diagonal,
                    samples: self.ao_samples,
                    max_radius_px: internal_size.1 as f32 * 0.15,
                    pixels_per_unit_at_1,
                    pixels_per_unit,
                    _pad0: [0.0; 3],
                },
            );
            self.ao_blur.write_uniforms(
                queue,
                &AoBlurUniforms {
                    depth_to_distance,
                    strength: settings.ao.strength,
                    _pad0: [0.0; 3],
                },
            );
        }
        self.resolve.write_uniforms(
            queue,
            &resolve_uniforms(settings, view, inv_view_proj, ao_enabled, frame_layers),
        );
        if layered {
            // Occlusion is gathered from the opaque scene alone.
            self.layer_resolve.write_uniforms(
                queue,
                &resolve_uniforms(settings, view, inv_view_proj, false, frame_layers),
            );
        }

        // ---- G-buffer fill. Always run, so a frame with no geometry
        // still leaves a cleared G-buffer for the later passes and picks.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gbuffer_pass"),
                color_attachments: &[
                    cleared(&self.gbuffer.albedo_view, wgpu::Color::TRANSPARENT),
                    cleared(&self.gbuffer.normal_view, wgpu::Color::TRANSPARENT),
                    cleared(&self.gbuffer.surface_view, GBuffer::SURFACE_CLEAR),
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
            self.surfel_pipeline.render(&mut pass, &self.frame_surfels);
        }

        // ---- Ambient occlusion, then the resolve onto the background.
        if ao_enabled {
            self.ao.run(
                encoder,
                &self.gbuffer_bindings.ao,
                &self.gbuffer.ao_raw_view,
                wgpu::Color::WHITE,
            );
            self.ao_blur.run(
                encoder,
                &self.gbuffer_bindings.ao_blur,
                &self.gbuffer.ao_view,
                wgpu::Color::WHITE,
            );
        }
        let background = clear_color(settings.background_color);
        // The lit scene: anti-aliasing's input, or the target itself.
        let lit = if settings.antialiasing {
            &self.gbuffer.lit_view
        } else {
            target
        };
        // The transparent meshes are peeled a layer at a time, nearest
        // first. Every layer's surface target is filled before anything
        // is lit: the edge lines of the opaque scene read the nearest
        // layer's, and those of a layer read every other layer's. Then
        // each layer in turn is filled like the G-buffer, resolved like
        // it and blended under the layers gathered so far.
        let layers = self.layers.as_ref().filter(|_| layered);
        if let Some(layers) = layers {
            for layer in 0..layer_count.min(layers.count()) {
                self.fill_layer(encoder, layer, true);
            }
            // The opaque scene's edges read the nearest layer's opacity.
            self.fill_layer(encoder, 0, false);
        }
        let resolve = layers.map_or(&self.gbuffer_bindings.resolve, |layers| {
            &layers.opaque_resolve
        });
        self.resolve.run(encoder, resolve, lit, background);
        if let Some(layers) = layers {
            for layer in 0..layer_count.min(layers.count()) {
                if layer > 0 {
                    self.fill_layer(encoder, layer, false);
                }
                self.resolve_layer(encoder, layer);
            }
            // The gathered layers, over the lit scene.
            self.layer_composite
                .run_onto(encoder, &layers.composite, lit);
        }
        if settings.antialiasing {
            self.fxaa.write_uniforms(
                queue,
                &FxaaUniforms {
                    texel: [1.0 / internal_size.0 as f32, 1.0 / internal_size.1 as f32],
                    _pad0: [0.0; 2],
                },
            );
            self.fxaa
                .run(encoder, &self.gbuffer_bindings.fxaa, target, background);
        }

        // ---- The grid, then depth-tested lines and points, over the lit
        // scene.
        self.draw_lines_and_points(
            encoder,
            target,
            DepthMode::Normal,
            grid_levels.is_some().then_some(settings.grid.axes),
        );

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
        self.draw_lines_and_points(encoder, target, DepthMode::Overlay, None);

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

        // ---- The view gizmo, in the target's own pixels.
        if let Some(gizmo) = &settings.gizmo {
            self.gizmo_pipeline
                .render(queue, encoder, gizmo, final_target, self.viewport_size);
        }

        // Sorted by object above, so the objects come in rising order.
        // A threshold that is not a number is none: every mesh drawn.
        let mut pickable: Vec<u32> = self
            .frame_transparent
            .iter()
            .filter(|draw| settings.pick_opacity.is_nan() || draw.opacity >= settings.pick_opacity)
            .map(|draw| draw.object.0)
            .collect();
        pickable.dedup();
        self.last_frame = Some(FrameRecord {
            inv_view_proj: view_proj.inverse(),
            size: internal_size,
            layers: frame_layers,
            pickable: pickable.into(),
        });
        self.frame_meshes.clear();
        self.frame_transparent.clear();
        self.frame_surfels.clear();
        self.frame_lines.clear();
        self.frame_points.clear();
        self.frame_retained_lines.clear();
        self.frame_retained_points.clear();
        self.frame_retained_splats.clear();

        FrameInfo {
            overflow: overflow.any().then_some(overflow),
            grid_spacing: grid_levels.map(|levels| levels.minor),
        }
    }

    /// Makes the transparent meshes' targets for `count` layers, and the
    /// bind groups over them and the G-buffer, unless they stand already.
    fn ensure_layers(&mut self, device: &wgpu::Device, count: u32) {
        if self
            .layers
            .as_ref()
            .is_some_and(|layers| layers.count() == count)
        {
            return;
        }
        let (width, height) = self.gbuffer.size;
        let buffers = LayerBuffers::new(device, width, height, self.gbuffer.lit_format(), count);
        let opaque = &self.gbuffer.surface_view;
        let surfaces = &buffers.surface_views;
        let peel = (0..surfaces.len())
            .map(|layer| {
                // The nearest layer has none before it; a target it does
                // not write stands in: the last layer's, or with one
                // layer the opaque scene's.
                let unread = if layer + 1 < surfaces.len() {
                    surfaces.last().unwrap_or(opaque)
                } else {
                    opaque
                };
                self.mesh_pipeline.bind_peel(
                    device,
                    opaque,
                    layer.checked_sub(1).map(|before| &surfaces[before]),
                    unread,
                )
            })
            .collect();
        let resolve = (0..surfaces.len())
            .map(|layer| {
                let mut views = vec![
                    &buffers.albedo_view,
                    &buffers.normal_view,
                    &surfaces[layer],
                    &buffers.opacity_view,
                    opaque,
                ];
                // Every other layer's surface target; the layer's own
                // fills the places there is none for.
                let others = (0..surfaces.len()).filter(|&other| other != layer);
                views.extend(
                    others
                        .chain(std::iter::repeat(layer))
                        .take(OTHER_LAYERS)
                        .map(|other| &surfaces[other]),
                );
                self.layer_resolve.bind(device, &views)
            })
            .collect();
        let opaque_resolve = self.resolve.bind(
            device,
            &[
                &self.gbuffer.albedo_view,
                &self.gbuffer.normal_view,
                &self.gbuffer.surface_view,
                &self.gbuffer.ao_view,
                &surfaces[0],
                &buffers.opacity_view,
            ],
        );
        let composite = self.layer_composite.bind(device, &[&buffers.gathered_view]);
        let pick = self.picker.bind(device, opaque, surfaces);
        self.layers = Some(Layers {
            peel,
            resolve,
            opaque_resolve,
            composite,
            pick,
            buffers,
        });
    }

    /// Fills layer `layer` (0 is the nearest) of the frame's transparent
    /// meshes: at each pixel, the nearest of their surfaces that is
    /// behind the layer before and in front of the opaque scene.
    /// `surface_only` fills the layer's own surface target; otherwise
    /// the albedo, normal and opacity targets the layers share, with the
    /// same surfaces.
    fn fill_layer(&self, encoder: &mut wgpu::CommandEncoder, layer: u32, surface_only: bool) {
        let Some(layers) = &self.layers else {
            return;
        };
        let buffers = &layers.buffers;
        let surface = [cleared(
            &buffers.surface_views[layer as usize],
            GBuffer::SURFACE_CLEAR,
        )];
        let rest = [
            cleared(&buffers.albedo_view, wgpu::Color::TRANSPARENT),
            cleared(&buffers.normal_view, wgpu::Color::TRANSPARENT),
            cleared(&buffers.opacity_view, wgpu::Color::TRANSPARENT),
        ];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(if surface_only {
                "layer_surface_pass"
            } else {
                "layer_pass"
            }),
            color_attachments: if surface_only { &surface } else { &rest },
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &buffers.depth_view,
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
        self.mesh_pipeline.render_layer(
            &mut pass,
            &self.frame_transparent,
            &layers.peel[layer as usize],
            surface_only,
        );
    }

    /// Lights layer `layer`, as last filled, under the layers gathered
    /// so far. The layers are resolved nearest first.
    fn resolve_layer(&self, encoder: &mut wgpu::CommandEncoder, layer: u32) {
        let Some(layers) = &self.layers else {
            return;
        };
        let resolve = &layers.resolve[layer as usize];
        let gathered = &layers.buffers.gathered_view;
        if layer == 0 {
            self.layer_resolve
                .run(encoder, resolve, gathered, wgpu::Color::TRANSPARENT);
        } else {
            self.layer_resolve.run_onto(encoder, resolve, gathered);
        }
    }

    /// The diagonal of the box around every mesh and surfel set submitted
    /// for this frame, in world units; 0 with none.
    fn mesh_diagonal(&self) -> f32 {
        let corners = |bounds: Option<(Vec3, Vec3)>, transform: Mat4| {
            let (min, max) = bounds?;
            Some((0..8).map(move |i| {
                transform.transform_point3(Vec3::new(
                    if i & 1 == 0 { min.x } else { max.x },
                    if i & 2 == 0 { min.y } else { max.y },
                    if i & 4 == 0 { min.z } else { max.z },
                ))
            }))
        };
        let bounds = self
            .frame_meshes
            .iter()
            .chain(&self.frame_transparent)
            .filter_map(|draw| corners(draw.mesh.bounds, draw.transform))
            .flatten()
            .chain(
                self.frame_surfels
                    .iter()
                    .filter_map(|draw| corners(draw.surfels.bounds, draw.transform))
                    .flatten(),
            )
            .fold(None, |bounds: Option<(Vec3, Vec3)>, p| {
                Some(bounds.map_or((p, p), |(min, max)| (min.min(p), max.max(p))))
            });
        bounds.map_or(0.0, |(min, max)| (max - min).length())
    }

    /// One pass over `target` drawing every line and point batch of
    /// `depth_mode`, immediate then retained. `grid` draws the grid
    /// first, with the normal axis line when it holds `true`.
    fn draw_lines_and_points(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth_mode: DepthMode,
        grid: Option<bool>,
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
        if let Some(axes) = grid {
            self.grid_pipeline.render(&mut pass, axes);
        }
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
        let Some(frame) = self.last_frame.clone() else {
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
        // A frame with transparent meshes is picked through its layers;
        // the layers of an earlier frame are stale.
        let bind_group = match &self.layers {
            Some(layers) if frame.layers > 0 => &layers.pick,
            _ => &self.gbuffer_bindings.pick,
        };
        self.picker
            .request(device, queue, bind_group, frame, pixel, source);
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
        self.last_frame.as_ref()?;
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
}
