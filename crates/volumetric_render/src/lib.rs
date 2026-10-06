//! The headless frame: draws a model or a project's exports to RGBA through
//! the same preview path as the GUI viewport (`volumetric_preview`): 3D
//! models meshed with the adaptive surface nets plan, 2D sketches as flat
//! rasters, FEA meshes and point clouds as their explicit data, triangle
//! meshes as they are, Subspace values as gizmos sized by the whole scene,
//! view sets as camera frustums, splats as splats. The frame an agent reads
//! headlessly is the frame a person sees in the viewport; the CLI's
//! `render` and the Python bindings are this call.
//!
//! Cameras: preset directions framed to the scene, an explicit pose with a
//! field of view, a pinhole with intrinsics and an OpenCV camera-to-world
//! pose, or a view of a surveyed view set, which is how a scan's
//! photographs are looked through — optionally composited over the
//! photograph.

use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};

use std::sync::atomic::AtomicBool;
use view_core::image::{Rgb, decode_rgb};
use view_core::overlay::compose;

use volumetric::direct_cast::{CastOptions, CastProjection, CastView, DirectCast};
use volumetric::wasm::ParallelModelSampler;
use volumetric::wasm::native::NativeParallelSampler;
use volumetric::{AssetTypeHint, LoadedAsset};
use volumetric_preview::{
    Asn2Settings, MarkLabel, OutputStats, PreviewBounds, PreviewEntity, PreviewMeshPlan,
    PreviewPlan, PreviewRenderMode, PreviewRequest, ViewFrame, build_preview_scene,
    clip_planes_for, mark_labels, observation_lines, srgb_to_linear, submit_subspace_gizmo,
    submit_view_highlight, wireframe_style,
};
use volumetric_renderer::{
    AoSettings, Camera, CameraView, GridPlane, GridSpacing, LineData, MaterialId, ObjectId,
    RenderSettings, SceneData, StandardView, SurfelData, SurfelVertex, Warp, offscreen::Offscreen,
    offscreen::downsample_rgba,
};

pub use view_core::overlay::Overlay;
pub use volumetric_preview::{ColorRange, pose_matrix};
pub use volumetric_renderer::{LightingPreset, Pinhole};

/// How each asset is turned into a preview.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanOptions {
    /// Meshing resolution for 3D models and raster size for 2D sketches.
    pub resolution: usize,
    /// Mesh models with sharp-feature reconstruction.
    pub sharp: bool,
    /// Mesh models with the decimation pass.
    pub simplify: bool,
    /// Colormap models by a declared sample channel.
    pub color_channel: Option<String>,
    /// Colormap FEA meshes and point clouds by a field, e.g. `node:confidence`.
    pub color_field: Option<String>,
    /// With `color_field`: the values the colormap spans (default: the
    /// field's own range).
    pub color_range: Option<ColorRange>,
    /// Overlay mesh edges.
    pub wireframe: bool,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            resolution: 128,
            sharp: true,
            simplify: true,
            color_channel: None,
            color_field: None,
            color_range: None,
            wireframe: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    Perspective,
    Orthographic,
}

/// Everything about a frame but the camera.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderOptions {
    /// Image size; None takes 1024, or a looked-through view's camera size.
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub projection: Projection,
    /// Vertical field of view in degrees (perspective presets and poses).
    pub fov_deg: f32,
    /// Orthographic frame height in world units (0 = fit the scene).
    pub ortho_scale: f32,
    /// Clip distances (default: from the scene).
    pub near: Option<f32>,
    pub far: Option<f32>,
    /// The world's up: orients the preset views, the ground grid and the
    /// default up of an explicit camera (default: the up of a drawn view
    /// set or splat, else +z).
    pub up: Option<Vec3>,
    /// Background as linear RGBA (see [`background_from_hex`]).
    pub background: [f32; 4],
    /// Ground grid minor spacing in metres, a major line every ten
    /// (0 disables).
    pub grid: f32,
    pub ssao: bool,
    pub lighting: LightingPreset,
    /// Draw edge lines at silhouettes, creases and part boundaries.
    pub edges: bool,
    /// Smooth stair-stepped edges (FXAA).
    pub antialias: bool,
    /// Draw a plain frame this many times larger per side and scale it
    /// down (1 = off). Frames through a lens, over a photograph, with
    /// marks, or of a splat are always drawn at their own size.
    pub supersample: u32,
    /// Draw 3D models by casting the model itself (`DIRECT_CASTING_PLAN.md`)
    /// instead of meshing it: exact at every pixel, no mesh artefacts, and
    /// as slow as the model's samples. Other asset kinds are drawn as
    /// usual.
    pub direct: bool,
    pub plan: PlanOptions,
    /// With a `Through` camera: composite the render over the photograph.
    pub overlay: Option<Overlay>,
    /// With a `Through` camera: draw what the view observed (marker
    /// quads, card corners, recorded picks and contours) over the frame,
    /// as the GUI's look-through does.
    pub marks: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            projection: Projection::Perspective,
            fov_deg: 45.0,
            ortho_scale: 0.0,
            near: None,
            far: None,
            up: None,
            background: background_from_hex("2d2d2d").expect("a valid default colour"),
            grid: 1.0,
            ssao: true,
            lighting: LightingPreset::default(),
            edges: true,
            antialias: true,
            supersample: 2,
            direct: false,
            plan: PlanOptions::default(),
            overlay: None,
            marks: false,
        }
    }
}

/// A preset direction, framed to the scene like the viewport's view menu.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ViewPreset {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
    Iso,
    IsoBack,
}

impl ViewPreset {
    pub const ALL: [Self; 8] = [
        Self::Front,
        Self::Back,
        Self::Left,
        Self::Right,
        Self::Top,
        Self::Bottom,
        Self::Iso,
        Self::IsoBack,
    ];

    /// The preset's name, also the file suffix of a multi-view render.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Front => "front",
            Self::Back => "back",
            Self::Left => "left",
            Self::Right => "right",
            Self::Top => "top",
            Self::Bottom => "bottom",
            Self::Iso => "iso",
            Self::IsoBack => "iso-back",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.suffix() == name)
    }

    /// The camera of this preset, framed to `min..max` at `fov_y` for a
    /// frame of `aspect`, in a world whose up is `up`: the viewport's
    /// z-up standard view, turned by the rotation taking +z to `up`, so
    /// `top` looks down `up` and `front` stays horizontal.
    pub fn camera(self, min: Vec3, max: Vec3, fov_y: f32, aspect: f32, up: Vec3) -> Camera {
        let standard = match self {
            Self::Front => StandardView::Front,
            Self::Back => StandardView::Back,
            Self::Left => StandardView::Left,
            Self::Right => StandardView::Right,
            Self::Top => StandardView::Top,
            Self::Bottom => StandardView::Bottom,
            Self::Iso | Self::IsoBack => StandardView::Isometric,
        };
        let mut orientation = standard.orientation();
        if self == Self::IsoBack {
            orientation = Quat::from_rotation_z(std::f32::consts::PI) * orientation;
        }
        let mut camera = Camera {
            orientation: Quat::from_rotation_arc(Vec3::Z, up) * orientation,
            fov_y,
            ..Camera::default()
        };
        camera.frame(min, max, aspect);
        camera
    }
}

/// Parses a comma-separated view list; `all` expands to every preset.
pub fn parse_views(list: &str) -> Result<Vec<ViewPreset>> {
    let mut views = Vec::new();
    for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        if name == "all" {
            views.extend(ViewPreset::ALL);
            continue;
        }
        let preset = ViewPreset::parse(name).with_context(|| {
            format!(
                "unknown view '{name}'; expected one of {}",
                ViewPreset::ALL
                    .iter()
                    .map(|p| p.suffix())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        views.push(preset);
    }
    if views.is_empty() {
        anyhow::bail!("the view list names no view");
    }
    Ok(views)
}

/// Where the frame is drawn from.
#[derive(Clone, Debug, PartialEq)]
pub enum CameraSpec {
    /// One frame per preset, each framed to the scene.
    Presets(Vec<ViewPreset>),
    /// An explicit eye looking at `target` (default: the scene centre)
    /// with `up` (default: the world's up).
    LookAt {
        eye: Vec3,
        target: Option<Vec3>,
        up: Option<Vec3>,
    },
    /// A pinhole with intrinsics in pixels of the frame and an OpenCV
    /// camera-to-world pose.
    Pinhole {
        pinhole: Pinhole,
        camera_to_world: Mat4,
    },
    /// A posed view of a view set among the assets or imports (`asset`
    /// names the set when there are several): its camera, at its size
    /// unless the options give one, with the photograph available for an
    /// overlay.
    Through { asset: Option<String>, view: String },
}

/// One rendered frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// The preset's name when several presets were drawn.
    pub suffix: Option<&'static str>,
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA, 8 bits per channel; opaque unless a splat's edge
    /// fades over the background.
    pub rgba: Vec<u8>,
}

/// What an asset became in the scene.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityReport {
    pub id: String,
    pub triangles: usize,
    pub points: usize,
    pub bounds: PreviewBounds,
    pub mesh_ms: f64,
    pub detail: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderReport {
    pub entities: Vec<EntityReport>,
    pub up: Vec3,
    /// Where the up came from: `options`, an asset's id, or `default`.
    pub up_source: String,
    pub gpu: String,
    /// Advisories: a frame drawn through a view's lens, geometry dropped
    /// at the GPU buffer limit.
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rendered {
    pub frames: Vec<Frame>,
    pub report: RenderReport,
}

/// A hex sRGB colour as the linear RGBA the renderer clears with.
pub fn background_from_hex(hex: &str) -> Result<[f32; 4]> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        anyhow::bail!("expected 6 hex digits, got '{hex}'");
    }
    let channel = |i: usize| -> Result<f32> {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .with_context(|| format!("invalid hex colour '{hex}'"))?;
        Ok(srgb_to_linear(f32::from(byte) / 255.0))
    };
    Ok([channel(0)?, channel(2)?, channel(4)?, 1.0])
}

/// Whether an asset of this kind has a picture.
pub fn is_renderable(asset: &LoadedAsset) -> bool {
    matches!(
        asset.type_hint(),
        Some(
            AssetTypeHint::Model
                | AssetTypeHint::FeaMesh
                | AssetTypeHint::TriMesh
                | AssetTypeHint::Subspace
                | AssetTypeHint::ViewSet
                | AssetTypeHint::Splat
                | AssetTypeHint::Assembly
                | AssetTypeHint::Mechanism
        ) | None
    )
}

/// Keeps the renderable exports, or exactly the `wanted` ids, each of which
/// must exist among the exports or the imports and be renderable. Imports
/// (a scan's view set, say) only draw when asked for by id.
pub fn select_assets(
    assets: Vec<LoadedAsset>,
    imports: &[LoadedAsset],
    wanted: &[String],
) -> Result<Vec<LoadedAsset>> {
    let available = || {
        assets
            .iter()
            .chain(imports)
            .filter(|a| is_renderable(a))
            .map(|a| a.id())
            .collect::<Vec<_>>()
            .join(", ")
    };
    if wanted.is_empty() {
        let renderable: Vec<LoadedAsset> = assets
            .iter()
            .filter(|a| is_renderable(a))
            .cloned()
            .collect();
        if renderable.is_empty() {
            anyhow::bail!(
                "nothing to draw: no model, mesh, cloud, subspace or view set export. Imports draw when named: {}",
                available()
            );
        }
        return Ok(renderable);
    }
    let mut selected = Vec::with_capacity(wanted.len());
    for id in wanted {
        let asset = assets
            .iter()
            .find(|a| a.id() == id)
            .or_else(|| imports.iter().find(|a| a.id() == id))
            .with_context(|| format!("no asset named '{id}'. Available: {}", available()))?;
        if !is_renderable(asset) {
            anyhow::bail!(
                "asset '{id}' is {}, which has no picture. Available: {}",
                asset
                    .type_hint()
                    .map(|h| h.to_string())
                    .unwrap_or_else(|| "untyped".to_string()),
                available()
            );
        }
        selected.push(asset.clone());
    }
    Ok(selected)
}

/// The viewport's recipe for an asset of this kind, with the resolution
/// and colour choices.
pub fn preview_request(asset: &LoadedAsset, options: &PlanOptions) -> PreviewRequest {
    let plan = match asset.type_hint() {
        Some(AssetTypeHint::FeaMesh) => PreviewPlan::FeaMesh {
            deformed: true,
            exaggeration_tenths: 10,
            color_field: options.color_field.clone(),
            color_range: options.color_range,
        },
        Some(AssetTypeHint::TriMesh) => PreviewPlan::TriMesh,
        Some(AssetTypeHint::Subspace) => PreviewPlan::Subspace,
        Some(AssetTypeHint::ViewSet) => PreviewPlan::ViewSet,
        Some(AssetTypeHint::Splat) => PreviewPlan::Splat,
        Some(AssetTypeHint::Mechanism) => PreviewPlan::Mechanism,
        Some(AssetTypeHint::Assembly) => PreviewPlan::Assembly {
            mesh: PreviewMeshPlan::for_mode(
                PreviewRenderMode::AdaptiveSurfaceNets2,
                options.resolution,
                Asn2Settings {
                    sharp_edges: options.sharp,
                    simplify: options.simplify,
                    ..Asn2Settings::default()
                },
            ),
        },
        _ => match volumetric::model_dimensions_static(asset.data()) {
            Some(2) => PreviewPlan::Sketch {
                resolution: options.resolution,
                color_channel: options.color_channel.clone(),
            },
            _ => PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::for_mode(
                    PreviewRenderMode::AdaptiveSurfaceNets2,
                    options.resolution,
                    Asn2Settings {
                        sharp_edges: options.sharp,
                        simplify: options.simplify,
                        ..Asn2Settings::default()
                    },
                ),
                color_channel: options.color_channel.clone(),
                tint_uncolored: false,
            },
        },
    };
    PreviewRequest {
        asset_id: asset.id().to_string(),
        source_hash: asset.content_hash(),
        data: asset.data_arc(),
        type_hint: asset.type_hint(),
        precursor_ids: vec![],
        plan,
        wireframe: options.wireframe,
        show_bounds: false,
        stale: false,
    }
}

/// The world's up for the frame: the option, else the up the drawn view
/// sets and splats were surveyed in (the first found), else +z, with a
/// note of where it came from.
pub fn world_up(up: Option<Vec3>, assets: &[LoadedAsset]) -> Result<(Vec3, String)> {
    if let Some(up) = up {
        let unit = up.normalize_or_zero();
        if unit == Vec3::ZERO {
            anyhow::bail!("up must not be zero");
        }
        return Ok((unit, "options".to_string()));
    }
    for asset in assets {
        let up = match asset.type_hint() {
            Some(AssetTypeHint::Splat) => {
                volumetric::splat::decode_splat(asset.data())
                    .map_err(|err| anyhow::anyhow!("asset '{}': {err}", asset.id()))?
                    .world
                    .up
            }
            Some(AssetTypeHint::ViewSet) => {
                volumetric_abi::viewset::decode_viewset(asset.data())
                    .map_err(|err| anyhow::anyhow!("asset '{}': {err}", asset.id()))?
                    .world
                    .up
            }
            _ => continue,
        };
        let up = Vec3::new(up[0] as f32, up[1] as f32, up[2] as f32).normalize_or_zero();
        if up != Vec3::ZERO {
            return Ok((up, format!("asset '{}'", asset.id())));
        }
    }
    Ok((Vec3::Z, "default".to_string()))
}

/// The ground grid's plane for a world whose up is `up`: the coordinate
/// plane most nearly perpendicular to it.
pub fn grid_plane_for(up: Vec3) -> GridPlane {
    let a = up.abs();
    if a.z >= a.x && a.z >= a.y {
        GridPlane::XY
    } else if a.x >= a.y {
        GridPlane::YZ
    } else {
        GridPlane::XZ
    }
}

/// The most a frame is supersampled, per side.
pub const MAX_SUPERSAMPLE: u32 = 4;

/// The asked-for supersampling, or as much of it as the device can draw
/// at `size`.
fn supersample_factor(asked: u32, size: (u32, u32), max_side: u32) -> u32 {
    (1..=asked.clamp(1, MAX_SUPERSAMPLE))
        .rev()
        .find(|factor| size.0.max(size.1).saturating_mul(*factor) <= max_side)
        .unwrap_or(1)
}

/// The frames to draw: a suffix (for several) and the view for each.
fn frames(
    camera: CameraSpec,
    options: &RenderOptions,
    (width, height): (u32, u32),
    bounds: PreviewBounds,
    world_up: Vec3,
) -> Result<Vec<(Option<&'static str>, CameraView)>> {
    let min = Vec3::from(bounds.min);
    let max = Vec3::from(bounds.max);
    let aspect = width as f32 / height as f32;
    let fov_y = options.fov_deg.to_radians();
    let ortho_height = |default: f32| {
        if options.ortho_scale > 0.0 {
            options.ortho_scale
        } else {
            default
        }
    };
    let clip = |eye: Vec3, forward: Vec3| {
        let scene = clip_planes_for(eye, forward, min, max);
        (
            options.near.unwrap_or(scene.0),
            options.far.unwrap_or(scene.1),
        )
    };
    let look = |eye: Vec3, target: Vec3, up: Vec3| -> CameraView {
        let forward = (target - eye).normalize();
        let (near, far) = clip(eye, forward);
        match options.projection {
            Projection::Perspective => {
                CameraView::look_at(eye, target, up, fov_y, aspect, near, far)
            }
            Projection::Orthographic => CameraView::look_at_orthographic(
                eye,
                target,
                up,
                ortho_height((max - min).length() * 1.1),
                aspect,
                near,
                far,
            ),
        }
    };

    Ok(match camera {
        CameraSpec::Presets(presets) => {
            let several = presets.len() > 1;
            presets
                .into_iter()
                .map(|preset| {
                    // The camera's own up: exact for `top` and `bottom`,
                    // where the world's up is the view direction.
                    let camera = preset.camera(min, max, fov_y, aspect, world_up);
                    (
                        several.then_some(preset.suffix()),
                        look(camera.eye(), camera.focus, camera.up()),
                    )
                })
                .collect()
        }
        CameraSpec::LookAt { eye, target, up } => {
            let up = up.unwrap_or(world_up);
            let target = target.unwrap_or((min + max) * 0.5);
            if (target - eye).normalize_or_zero() == Vec3::ZERO {
                anyhow::bail!("the camera position coincides with the look-at point");
            }
            vec![(None, look(eye, target, up))]
        }
        CameraSpec::Pinhole {
            pinhole,
            camera_to_world,
        } => {
            if options.projection == Projection::Orthographic {
                anyhow::bail!("a pinhole camera is perspective");
            }
            let eye = camera_to_world.transform_point3(Vec3::ZERO);
            let forward = camera_to_world.transform_vector3(Vec3::Z).normalize();
            let (near, far) = clip(eye, forward);
            vec![(
                None,
                CameraView::pinhole(&pinhole, camera_to_world, near, far),
            )]
        }
        CameraSpec::Through { .. } => {
            unreachable!("a Through camera is resolved to a pinhole before framing")
        }
    })
}

/// `src` (premultiplied RGBA over a transparent ground) over `dst`.
fn over(mut dst: Vec<u8>, src: &[u8]) -> Vec<u8> {
    for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        let a = f32::from(s[3]) / 255.0;
        if a <= 0.0 {
            continue;
        }
        for c in 0..3 {
            d[c] = (f32::from(s[c]) + f32::from(d[c]) * (1.0 - a))
                .round()
                .min(255.0) as u8;
        }
        d[3] = (f32::from(s[3]) + f32::from(d[3]) * (1.0 - a))
            .round()
            .min(255.0) as u8;
    }
    dst
}

/// Draw `assets` (a project's selected exports, or one model) from
/// `camera` with `options`. `imports` are the project's imports, where a
/// `Through` camera finds its view set when the set is not among the
/// drawn assets.
static NEVER: AtomicBool = AtomicBool::new(false);

/// A 3D model drawn by casting it: its sampler and the record of what the
/// casts have found.
struct DirectSource {
    id: String,
    sampler: NativeParallelSampler,
    cast: DirectCast,
    bounds: PreviewBounds,
}

impl DirectSource {
    fn new(asset: &LoadedAsset) -> Result<Self> {
        let bytes = asset.as_model().context("not a model")?;
        let sampler = NativeParallelSampler::new(bytes).map_err(|err| anyhow::anyhow!("{err}"))?;
        let cast = DirectCast::for_model(&sampler)?;
        let bounds = sampler.get_bounds()?;
        let (min, max) = bounds.as_f32();
        Ok(Self {
            id: asset.id().to_string(),
            sampler,
            cast,
            bounds: PreviewBounds { min, max },
        })
    }

    /// An entity with no geometry, so the frame is framed on the model's
    /// bounds like any other.
    fn entity(&self) -> PreviewEntity {
        PreviewEntity {
            scene: SceneData::new(),
            bounds: self.bounds,
            stats: OutputStats {
                bounds: Some((self.bounds.min, self.bounds.max)),
                ..OutputStats::default()
            },
            wireframe_lines: None,
            subspace: None,
            mesh_keys: Vec::new(),
            articulated: None,
        }
    }
}

/// The caster's view of a frame drawn `width × height`: the same eye,
/// axes and projection. A sheared frustum (a photograph's off-centre
/// principal point) has no caster equivalent yet.
fn cast_view_of(view: &CameraView, (width, height): (u32, u32)) -> Result<CastView> {
    let camera_to_world = view.view.inverse();
    let axis = |i: usize| camera_to_world.col(i).truncate().as_dvec3().normalize();
    let projection = view.projection.to_cols_array_2d();
    anyhow::ensure!(
        projection[2][0].abs() < 1e-6 && projection[2][1].abs() < 1e-6,
        "a direct cast cannot look through an off-centre camera yet"
    );
    let orthographic = projection[3][3] == 1.0;
    let scale = 1.0 / projection[1][1] as f64;
    Ok(CastView {
        eye: camera_to_world.col(3).truncate().as_dvec3(),
        right: axis(0),
        up: axis(1),
        forward: -axis(2),
        projection: if orthographic {
            CastProjection::Orthographic { half_height: scale }
        } else {
            CastProjection::Perspective {
                tan_half_fov_y: scale,
            }
        },
        width,
        height,
    })
}

pub fn render(
    assets: &[LoadedAsset],
    imports: &[LoadedAsset],
    camera: CameraSpec,
    options: &RenderOptions,
) -> Result<Rendered> {
    let mut notes = Vec::new();
    // A view's camera fixes the image size and the projection; explicit
    // sizes scale its intrinsics so smaller renders stay aligned.
    let mut photo: Option<Rgb> = None;
    let mut marks: Option<LineData> = None;
    // The marks' names, at output pixels.
    let mut labels: Vec<MarkLabel> = Vec::new();
    let mut lens: Option<Warp> = None;
    let mut size = (
        options.width.unwrap_or(1024),
        options.height.unwrap_or(1024),
    );
    let camera = match camera {
        CameraSpec::Through { asset, view } => {
            if options.projection == Projection::Orthographic {
                anyhow::bail!("a view's camera is perspective");
            }
            // Imports and the drawn assets overlap when an import was
            // selected; one copy each.
            let all: Vec<LoadedAsset> = imports
                .iter()
                .chain(
                    assets
                        .iter()
                        .filter(|a| !imports.iter().any(|i| i.id() == a.id())),
                )
                .cloned()
                .collect();
            let set = volumetric::asset_query::viewset(&all, asset.as_deref())
                .map_err(anyhow::Error::msg)?;
            let (view, view_camera) = set.view(&view).with_context(|| {
                format!(
                    "no view '{view}' in the view set. Views: {}",
                    set.views
                        .iter()
                        .map(|v| v.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
            size = (
                options.width.unwrap_or(view_camera.width),
                options.height.unwrap_or(view_camera.height),
            );
            if options.marks {
                marks = Some(LineData {
                    segments: observation_lines(view, view_camera),
                });
                let sx = size.0 as f64 / f64::from(view_camera.width);
                let sy = size.1 as f64 / f64::from(view_camera.height);
                labels = mark_labels(view, view_camera)
                    .into_iter()
                    .map(|mut label| {
                        label.pixel = [label.pixel[0] * sx, label.pixel[1] * sy];
                        label
                    })
                    .collect();
            }
            if options.overlay.is_some() {
                let bytes = view.image.as_deref().with_context(|| {
                    format!(
                        "view {} carries no photograph; import it with images",
                        view.id
                    )
                })?;
                photo = Some(decode_rgb(bytes)?.resized(size.0, size.1)?);
            }
            // The frame is drawn through the view's lens: the scene
            // renders through the overscan pinhole and the warp bends it
            // into the photograph's own pixels.
            let frame = ViewFrame::of(view, view_camera)
                .with_context(|| format!("view '{}' is not posed", view.id))?;
            let framed = frame.framed_stretched(size.0, size.1);
            if let Some(warp) = &framed.warp {
                notes.push(format!(
                    "view {}: drawn through the lens ({} x {} pinhole frame warped to {} x {})",
                    view.id, warp.source.0, warp.source.1, size.0, size.1
                ));
            }
            lens = framed.warp;
            CameraSpec::Pinhole {
                pinhole: framed.pinhole,
                camera_to_world: frame.camera_to_world,
            }
        }
        other => {
            if options.overlay.is_some() {
                anyhow::bail!(
                    "an overlay composites over a view's photograph; look through a view"
                );
            }
            other
        }
    };
    if size.0 == 0 || size.1 == 0 {
        anyhow::bail!("the frame size must be positive");
    }

    let mut entities: Vec<PreviewEntity> = Vec::with_capacity(assets.len());
    let mut reports = Vec::with_capacity(assets.len());
    // Models cast directly, one record each, parallel to `entities`.
    let mut direct: Vec<Option<DirectSource>> = Vec::with_capacity(assets.len());
    for asset in assets {
        if options.direct && asset.type_hint() == Some(AssetTypeHint::Model) {
            match DirectSource::new(asset) {
                Ok(source) => {
                    entities.push(source.entity());
                    reports.push(EntityReport {
                        id: asset.id().to_string(),
                        triangles: 0,
                        points: 0,
                        bounds: source.bounds,
                        mesh_ms: 0.0,
                        detail: vec!["cast directly".to_string()],
                    });
                    direct.push(Some(source));
                    continue;
                }
                Err(err) => notes.push(format!("{}: meshed, not cast: {err}", asset.id())),
            }
        }
        direct.push(None);
        let request = preview_request(asset, &options.plan);
        let entity = build_preview_scene(&request)
            .map_err(|err| anyhow::anyhow!("{}: {err}", asset.id()))?;
        reports.push(EntityReport {
            id: asset.id().to_string(),
            triangles: entity.stats.triangles,
            points: entity.stats.points,
            bounds: entity.bounds,
            mesh_ms: entity.stats.mesh_ms,
            detail: entity.stats.detail.clone(),
        });
        entities.push(entity);
    }
    // Frame the geometry; a Subspace gizmo is infinite and only carries a
    // placeholder box around its chart origin, which sizes the frame when
    // nothing else is drawn.
    let bounds = entities
        .iter()
        .filter(|entity| entity.subspace.is_none())
        .map(|entity| entity.bounds)
        .reduce(PreviewBounds::union)
        .or_else(|| {
            entities
                .iter()
                .map(|entity| entity.bounds)
                .reduce(PreviewBounds::union)
        })
        .context("nothing to draw")?;
    // The marks sit on the picture plane close to the eye; the clip
    // planes must reach them as they do in the GUI, where the view set's
    // frustum is part of the scene.
    let bounds = marks.iter().flat_map(|m| &m.segments).fold(bounds, |b, s| {
        b.union(PreviewBounds {
            min: (s.start[0], s.start[1], s.start[2]),
            max: (s.start[0], s.start[1], s.start[2]),
        })
        .union(PreviewBounds {
            min: (s.end[0], s.end[1], s.end[2]),
            max: (s.end[0], s.end[1], s.end[2]),
        })
    });
    let (up, up_source) = world_up(options.up, assets)?;
    let frames = frames(camera, options, size, bounds, up)?;

    let offscreen = Offscreen::new().map_err(anyhow::Error::msg)?;
    let gpu = offscreen.adapter_name();
    // A plain render is drawn larger and scaled down. A frame through a
    // lens, over a photograph, with marks or of a splat is drawn at its
    // own size: those are compared pixel for pixel with photographs and
    // reference renders.
    let plain = lens.is_none()
        && options.overlay.is_none()
        && !options.marks
        && entities.iter().all(|entity| entity.scene.splats.is_empty());
    let supersample = if plain {
        supersample_factor(options.supersample, size, offscreen.max_frame_side())
    } else {
        1
    };
    let drawn = (size.0 * supersample, size.1 * supersample);
    let mut renderer = offscreen.renderer(drawn.0, drawn.1);
    renderer
        .set_warp(offscreen.device(), lens.as_ref())
        .map_err(anyhow::Error::msg)?;
    let resident: Vec<_> = entities
        .iter()
        .map(|entity| renderer.create_retained_scene(offscreen.device(), &entity.scene))
        .collect();

    // An overlay reads the render's coverage from its alpha channel: the
    // background is transparent black, surfaces are opaque, and a splat's
    // fading edge is in between, so nothing but geometry may touch the
    // frame.
    const SENTINEL: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    let overlay = options.overlay;
    let mut settings = RenderSettings {
        background_color: if overlay.is_some() {
            SENTINEL
        } else {
            options.background
        },
        ao: AoSettings {
            enabled: options.ssao,
            ..AoSettings::default()
        },
        lighting: options.lighting.rig(),
        // An overlay's alpha is the render's coverage, taken as drawn.
        antialiasing: options.antialias && overlay.is_none(),
        ..RenderSettings::default()
    };
    settings.edges.enabled = options.edges;
    settings.pixel_scale = supersample as f32;
    settings.grid.visible = options.grid > 0.0 && overlay.is_none();
    settings.grid.plane = grid_plane_for(up);
    settings.grid.spacing = GridSpacing::Fixed(options.grid);

    let mut out = Vec::with_capacity(frames.len());
    for (suffix, view) in frames {
        // A cast model's surfels are found for this very frame, at the
        // drawn size, so every pixel has a sample of its own.
        let cast_view = if direct.iter().any(Option::is_some) {
            Some(cast_view_of(&view, drawn)?)
        } else {
            None
        };
        // Each mesh is its own object, numbered in drawing order, so the
        // boundary between two parts gets an edge line.
        let mut object = 0;
        for ((scene, entity), source) in resident.iter().zip(&entities).zip(&mut direct) {
            if let (Some(source), Some(cast_view)) = (source, &cast_view) {
                let start = std::time::Instant::now();
                let (image, stats) = source
                    .cast
                    .cast(&source.sampler, cast_view, &CastOptions::default(), &NEVER)
                    .expect("never cancelled");
                let surfels = SurfelData {
                    surfels: source
                        .cast
                        .surfels(Some((cast_view, 1.0)))
                        .iter()
                        .map(|s| SurfelVertex::new(s.position, s.normal, s.radius))
                        .collect(),
                };
                let resident = renderer.create_retained_surfels(offscreen.device(), &surfels);
                object += 1;
                renderer.submit_retained_surfels(
                    &resident,
                    Mat4::IDENTITY,
                    ObjectId(object),
                    MaterialId::default(),
                );
                notes.push(format!(
                    "{}{}: cast directly, {} surfels from {} samples in {:.2} s, {} of {} pixels hit",
                    source.id,
                    suffix.map_or(String::new(), |s| format!(" ({s})")),
                    surfels.surfels.len(),
                    stats.samples,
                    start.elapsed().as_secs_f64(),
                    image.hits.iter().flatten().count(),
                    image.hits.len()
                ));
                continue;
            }
            for (mesh, transform) in &scene.meshes {
                object += 1;
                renderer.submit_retained_mesh(
                    mesh,
                    *transform,
                    ObjectId(object),
                    MaterialId::default(),
                );
            }
            for lines in &scene.lines {
                renderer.submit_retained_lines(lines);
            }
            for points in &scene.points {
                renderer.submit_retained_points(points);
            }
            for splat in &scene.splats {
                renderer.submit_retained_splat(splat);
            }
            if options.plan.wireframe
                && let Some(lines) = &entity.wireframe_lines
            {
                renderer.submit_lines(lines, Mat4::IDENTITY, wireframe_style());
            }
            if let Some(subspace) = &entity.subspace {
                submit_subspace_gizmo(&mut renderer, subspace, bounds);
            }
        }
        let (rgba, info) = offscreen
            .render_rgba(&mut renderer, &view, &settings)
            .map_err(anyhow::Error::msg)?;
        let rgba = downsample_rgba(&rgba, drawn.0, drawn.1, supersample);
        if let Some(overflow) = info.overflow {
            notes.push(format!(
                "dropped {} of {} triangles, {} lines, {} points, {} splat primitives and {} surfels at the GPU buffer limit",
                overflow.dropped_triangles,
                overflow.total_triangles,
                overflow.dropped_lines,
                overflow.dropped_points,
                overflow.dropped_splats,
                overflow.dropped_surfels
            ));
        }
        let rgba = match (&overlay, &photo) {
            (Some(overlay), Some(photo)) => {
                let covered: Vec<f32> = rgba
                    .chunks_exact(4)
                    .map(|px| f32::from(px[3]) / 255.0)
                    .collect();
                // Colours are premultiplied over the transparent background;
                // divide the coverage back out.
                let render = Rgb {
                    width: size.0,
                    height: size.1,
                    pixels: rgba
                        .chunks_exact(4)
                        .flat_map(|px| {
                            let a = f32::from(px[3]).max(1.0);
                            [px[0], px[1], px[2]]
                                .map(|c| (f32::from(c) * 255.0 / a).round().min(255.0) as u8)
                        })
                        .collect(),
                };
                let composed = compose(photo, &render, &covered, *overlay)?;
                composed
                    .pixels
                    .chunks_exact(3)
                    .flat_map(|px| [px[0], px[1], px[2], 255])
                    .collect()
            }
            _ => rgba,
        };
        // The marks go over the finished frame, overlay included, at full
        // strength: a pick is judged against the photograph, not faded
        // with the render. Drawn alone over a transparent ground.
        let rgba = match &marks {
            Some(marks) => {
                submit_view_highlight(&mut renderer, marks);
                let mut marks_settings = RenderSettings {
                    background_color: SENTINEL,
                    ao: AoSettings {
                        enabled: false,
                        ..AoSettings::default()
                    },
                    antialiasing: false,
                    ..RenderSettings::default()
                };
                marks_settings.grid.visible = false;
                let (lines, _) = offscreen
                    .render_rgba(&mut renderer, &view, &marks_settings)
                    .map_err(anyhow::Error::msg)?;
                let mut rgba = over(rgba, &lines);
                // Names beside the marks, the font scaled with the frame
                // so they read when a large frame is viewed small.
                let scale = (size.0 / 600).max(1);
                for label in &labels {
                    let (_, height) = view_core::text::text_size(&label.text, scale);
                    let half = i64::from(height / 2 + scale);
                    let ink = label.kind.srgb8();
                    view_core::text::draw_label(
                        &label.text,
                        label.pixel[0].round() as i64,
                        label.pixel[1].round() as i64 - half,
                        scale,
                        |x, y, lit| {
                            if x < size.0 && y < size.1 {
                                let i = ((y * size.0 + x) * 4) as usize;
                                let color = if lit { ink } else { [20, 20, 20] };
                                rgba[i..i + 3].copy_from_slice(&color);
                                rgba[i + 3] = 255;
                            }
                        },
                    );
                }
                rgba
            }
            None => rgba,
        };
        out.push(Frame {
            suffix,
            width: size.0,
            height: size.1,
            rgba,
        });
    }
    Ok(Rendered {
        frames: out,
        report: RenderReport {
            entities: reports,
            up,
            up_source,
            gpu,
            notes,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(id: &str, type_hint: Option<AssetTypeHint>) -> LoadedAsset {
        LoadedAsset::from_parts(id.to_string(), vec![1, 2, 3], type_hint, vec![])
    }

    #[test]
    fn views_parse() {
        let views = parse_views("front, iso-back").unwrap();
        assert_eq!(views, vec![ViewPreset::Front, ViewPreset::IsoBack]);
        assert_eq!(parse_views("all").unwrap().len(), 8);
        assert!(
            parse_views("sideways")
                .unwrap_err()
                .to_string()
                .contains("sideways")
        );
        assert!(parse_views(" , ").is_err());
    }

    /// Every preset looks at the scene centre from outside the box.
    #[test]
    fn presets_frame_the_scene() {
        let (min, max) = (Vec3::new(-1.0, -2.0, 0.0), Vec3::new(1.0, 2.0, 1.0));
        for preset in ViewPreset::ALL {
            let camera = preset.camera(min, max, 0.8, 1.5, Vec3::Z);
            assert_eq!(camera.focus, (min + max) * 0.5);
            let eye = camera.eye();
            assert!(
                eye.x < min.x
                    || eye.x > max.x
                    || eye.y < min.y
                    || eye.y > max.y
                    || eye.z < min.z
                    || eye.z > max.z,
                "{preset:?} eye {eye} inside"
            );
        }
        // The back three-quarter view is the front one turned half way
        // round the vertical through the centre.
        let centre = (min + max) * 0.5;
        let iso = ViewPreset::Iso.camera(min, max, 0.8, 1.5, Vec3::Z).eye() - centre;
        let back = ViewPreset::IsoBack
            .camera(min, max, 0.8, 1.5, Vec3::Z)
            .eye()
            - centre;
        assert!((iso.x + back.x).abs() < 1e-4 && (iso.y + back.y).abs() < 1e-4);
        assert!((iso.z - back.z).abs() < 1e-4 && iso.z > 0.0);
    }

    /// `top` looks straight down the world's up and `front` stays
    /// horizontal, in the default z-up world and in a y-up one.
    #[test]
    fn presets_follow_the_world_up() {
        let (min, max) = (Vec3::new(-1.0, -2.0, 0.0), Vec3::new(1.0, 2.0, 1.0));
        let centre = (min + max) * 0.5;
        let top = ViewPreset::Top.camera(min, max, 0.8, 1.5, Vec3::Z);
        assert!((top.forward() - Vec3::NEG_Z).length() < 1e-5);
        assert!((top.eye() - centre).truncate().length() < 1e-4);
        assert!(
            top.eye().z > max.z,
            "top eye {} not above the scene",
            top.eye()
        );
        let front = ViewPreset::Front.camera(min, max, 0.8, 1.5, Vec3::Z);
        assert!(
            (front.eye().z - centre.z).abs() < 1e-4,
            "front eye not level"
        );
        assert!((front.forward() - Vec3::Y).length() < 1e-5);

        let top = ViewPreset::Top.camera(min, max, 0.8, 1.5, Vec3::Y);
        assert!((top.forward() - Vec3::NEG_Y).length() < 1e-5);
        let front = ViewPreset::Front.camera(min, max, 0.8, 1.5, Vec3::Y);
        assert!(
            front.forward().y.abs() < 1e-5,
            "front not level in a y-up world"
        );

        assert_eq!(grid_plane_for(Vec3::Z), GridPlane::XY);
        assert_eq!(grid_plane_for(Vec3::Y), GridPlane::XZ);
        assert_eq!(grid_plane_for(Vec3::NEG_X), GridPlane::YZ);
    }

    #[test]
    fn supersampling_stops_at_what_the_device_can_draw() {
        assert_eq!(supersample_factor(2, (1024, 768), 8192), 2);
        assert_eq!(supersample_factor(0, (1024, 768), 8192), 1);
        assert_eq!(supersample_factor(9, (1024, 768), 8192), MAX_SUPERSAMPLE);
        assert_eq!(supersample_factor(4, (3000, 100), 8192), 2);
        assert_eq!(supersample_factor(2, (5000, 100), 8192), 1);
    }

    #[test]
    fn clip_planes_enclose_the_scene() {
        let (min, max) = (Vec3::splat(-1.0), Vec3::splat(1.0));
        let (near, far) = clip_planes_for(Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, min, max);
        assert!(near > 0.0 && near < 4.0, "near {near}");
        assert!(far > 6.0, "far {far}");
        // Inside the box the near plane stays positive.
        let (near, _) = clip_planes_for(Vec3::ZERO, Vec3::NEG_Z, min, max);
        assert!(near > 0.0);
    }

    #[test]
    fn assets_are_selected_by_id_and_kind() {
        let all = vec![
            asset("scan", Some(AssetTypeHint::Model)),
            asset("axis", Some(AssetTypeHint::Subspace)),
            asset("fit", Some(AssetTypeHint::F64Map)),
            asset("cloud", Some(AssetTypeHint::FeaMesh)),
        ];
        let imports = vec![
            asset("views", Some(AssetTypeHint::ViewSet)),
            asset("config", Some(AssetTypeHint::Config)),
        ];
        let ids = |assets: &[LoadedAsset]| {
            assets
                .iter()
                .map(|a| a.id().to_string())
                .collect::<Vec<_>>()
        };
        // Exports draw by default; an import draws only when named.
        assert_eq!(
            ids(&select_assets(all.clone(), &imports, &[]).unwrap()),
            ["scan", "axis", "cloud"]
        );
        assert_eq!(
            ids(&select_assets(
                all.clone(),
                &imports,
                &["cloud".to_string(), "scan".to_string(), "views".to_string()]
            )
            .unwrap()),
            ["cloud", "scan", "views"]
        );
        let missing = select_assets(all.clone(), &imports, &["nope".to_string()]).unwrap_err();
        assert!(
            missing.to_string().contains("scan, axis, cloud, views"),
            "{missing}"
        );
        let wrong = select_assets(all.clone(), &imports, &["fit".to_string()]).unwrap_err();
        assert!(wrong.to_string().contains("F64Map"), "{wrong}");
        let wrong = select_assets(all, &imports, &["config".to_string()]).unwrap_err();
        assert!(wrong.to_string().contains("no picture"), "{wrong}");
        let none = select_assets(
            vec![asset("fit", Some(AssetTypeHint::F64Map))],
            &imports,
            &[],
        )
        .unwrap_err();
        assert!(none.to_string().contains("named: views"), "{none}");
    }

    #[test]
    fn plans_follow_the_asset_kind() {
        let options = PlanOptions {
            resolution: 64,
            sharp: false,
            simplify: true,
            color_channel: None,
            color_field: Some("node:confidence".to_string()),
            color_range: None,
            wireframe: true,
        };
        let sphere = volumetric_assets::get_model("simple_sphere_model").expect("bundled sphere");
        let model = LoadedAsset::from_parts(
            "sphere".to_string(),
            sphere.bytes.to_vec(),
            Some(AssetTypeHint::Model),
            vec![],
        );
        let request = preview_request(&model, &options);
        match request.plan {
            PreviewPlan::Model3d {
                mesh:
                    PreviewMeshPlan::AdaptiveSurfaceNets2 {
                        target_resolution,
                        settings,
                        ..
                    },
                ..
            } => {
                assert_eq!(target_resolution, 64);
                assert!(!settings.sharp_edges && settings.simplify);
            }
            other => panic!("{other:?}"),
        }
        assert!(request.wireframe);

        match preview_request(&asset("cloud", Some(AssetTypeHint::FeaMesh)), &options).plan {
            PreviewPlan::FeaMesh { color_field, .. } => {
                assert_eq!(color_field.as_deref(), Some("node:confidence"))
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            preview_request(&asset("axis", Some(AssetTypeHint::Subspace)), &options).plan,
            PreviewPlan::Subspace
        );
        assert_eq!(
            preview_request(&asset("mesh", Some(AssetTypeHint::TriMesh)), &options).plan,
            PreviewPlan::TriMesh
        );
    }

    /// A model cast directly is the same picture as the model meshed:
    /// the same pixels covered, nearly all of them the same shade. Needs
    /// a GPU; skipped without one.
    #[test]
    fn a_direct_cast_draws_the_mesh_render() {
        if let Err(err) = Offscreen::new() {
            eprintln!("skipped: {err}");
            return;
        }
        let sphere = volumetric_assets::get_model("simple_sphere_model").expect("bundled sphere");
        let model = LoadedAsset::from_parts(
            "sphere".to_string(),
            sphere.bytes.to_vec(),
            Some(AssetTypeHint::Model),
            vec![],
        );
        let options = RenderOptions {
            width: Some(192),
            height: Some(144),
            grid: 0.0,
            supersample: 1,
            ..RenderOptions::default()
        };
        let frame = |direct: bool| {
            let options = RenderOptions {
                direct,
                ..options.clone()
            };
            let rendered = render(
                std::slice::from_ref(&model),
                &[],
                CameraSpec::Presets(vec![ViewPreset::Iso]),
                &options,
            )
            .unwrap();
            assert_eq!(rendered.frames.len(), 1);
            (rendered.frames[0].rgba.clone(), rendered.report)
        };
        let (meshed, _) = frame(false);
        let (cast, report) = frame(true);
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("cast directly")),
            "{:?}",
            report.notes
        );
        let background = options.background.map(|c| (c * 255.0).round() as i32);
        let covered = |px: &[u8]| (0..3).any(|c| (px[c] as i32 - background[c]).abs() > 8);
        let (mut on_mesh, mut on_cast, mut differ) = (0usize, 0usize, 0usize);
        for (a, b) in meshed.chunks_exact(4).zip(cast.chunks_exact(4)) {
            on_mesh += covered(a) as usize;
            on_cast += covered(b) as usize;
            differ += ((0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > 24)) as usize;
        }
        assert!(on_mesh > 3000, "{on_mesh} sphere pixels");
        assert!(
            (on_cast as i64 - on_mesh as i64).abs() * 50 < on_mesh as i64,
            "{on_cast} cast pixels against {on_mesh} meshed"
        );
        assert!(differ * 20 < on_mesh, "{differ} of {on_mesh} pixels differ");
    }

    #[test]
    fn background_is_linearised() {
        let white = background_from_hex("ffffff").unwrap();
        assert!((white[0] - 1.0).abs() < 1e-6);
        let grey = background_from_hex("#808080").unwrap();
        assert!(grey[0] > 0.2 && grey[0] < 0.22, "{grey:?}");
        assert!(background_from_hex("12345").is_err());
    }

    /// The frames a camera spec yields, without a GPU: presets take a
    /// suffix only when several are drawn; a look-at at the target is
    /// refused; a pinhole under orthographic projection is refused.
    #[test]
    fn frames_follow_the_camera_spec() {
        let bounds = PreviewBounds {
            min: (-1.0, -1.0, -1.0),
            max: (1.0, 1.0, 1.0),
        };
        let options = RenderOptions::default();
        let one = frames(
            CameraSpec::Presets(vec![ViewPreset::Iso]),
            &options,
            (640, 480),
            bounds,
            Vec3::Y,
        )
        .unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].0, None);
        let two = frames(
            CameraSpec::Presets(vec![ViewPreset::Top, ViewPreset::Front]),
            &options,
            (640, 480),
            bounds,
            Vec3::Y,
        )
        .unwrap();
        assert_eq!(
            two.iter().map(|f| f.0).collect::<Vec<_>>(),
            [Some("top"), Some("front")]
        );
        assert!(
            frames(
                CameraSpec::LookAt {
                    eye: Vec3::ZERO,
                    target: Some(Vec3::ZERO),
                    up: None
                },
                &options,
                (640, 480),
                bounds,
                Vec3::Y
            )
            .is_err()
        );
        let ortho = RenderOptions {
            projection: Projection::Orthographic,
            ..RenderOptions::default()
        };
        assert!(
            frames(
                CameraSpec::Pinhole {
                    pinhole: Pinhole {
                        fx: 1.0,
                        fy: 1.0,
                        cx: 0.0,
                        cy: 0.0,
                        width: 2,
                        height: 2
                    },
                    camera_to_world: Mat4::IDENTITY
                },
                &ortho,
                (2, 2),
                bounds,
                Vec3::Y
            )
            .is_err()
        );
    }
}
