//! `render`: draws a model, or the exports of a project, to PNG through the
//! same preview path as the GUI viewport (`volumetric_preview`): 3D models
//! meshed with the adaptive surface nets plan, 2D sketches as flat rasters,
//! FEA meshes and point clouds as their explicit data, triangle meshes as
//! they are, Subspace values as gizmos sized by the whole scene. The frame
//! an agent reads headlessly is the frame a person sees in the viewport.
//!
//! Cameras: preset directions framed to the scene, an explicit pose with a
//! field of view, or a pinhole with intrinsics and an OpenCV camera-to-world
//! pose, which is how a scan's photographs are looked through.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use glam::{Mat4, Vec3, Vec4};

use volumetric::{AssetTypeHint, LoadedAsset, Project};
use volumetric_preview::{
    Asn2Settings, PreviewBounds, PreviewEntity, PreviewMeshPlan, PreviewPlan, PreviewRenderMode,
    PreviewRequest, build_preview_scene, srgb_to_linear, submit_subspace_gizmo, wireframe_style,
};
use volumetric_renderer::{
    Camera, CameraView, GridPlanes, Pinhole, RenderSettings, ViewDirection, offscreen::Offscreen,
};

#[derive(Parser, Debug)]
pub struct RenderArgs {
    /// Input file: a .wasm model or a .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// For .vproj inputs: an export to draw (repeatable; default: every
    /// renderable export)
    #[arg(long = "asset")]
    pub assets: Vec<String>,

    /// Output PNG path (a view suffix is added when several views render)
    #[arg(short, long)]
    pub output: PathBuf,

    #[arg(long, default_value_t = 1024)]
    pub width: u32,

    #[arg(long, default_value_t = 1024)]
    pub height: u32,

    /// Comma-separated preset views: front, back, left, right, top, bottom,
    /// iso, iso-back, all
    #[arg(long, default_value = "iso")]
    pub views: String,

    /// Background colour as hex sRGB (e.g. 2d2d2d)
    #[arg(long, default_value = "2d2d2d")]
    pub background: String,

    /// Camera position x,y,z (an explicit camera replaces --views)
    #[arg(long, allow_hyphen_values = true)]
    pub camera_pos: Option<String>,

    /// Look-at point x,y,z (default: the scene centre)
    #[arg(long, allow_hyphen_values = true)]
    pub camera_target: Option<String>,

    /// Up vector x,y,z
    #[arg(long, default_value = "0,1,0", allow_hyphen_values = true)]
    pub camera_up: String,

    /// Vertical field of view in degrees (perspective)
    #[arg(long, default_value_t = 45.0)]
    pub fov: f32,

    /// Pinhole intrinsics fx,fy,cx,cy in pixels of the --width x --height
    /// image (with --pose)
    #[arg(long, allow_hyphen_values = true)]
    pub intrinsics: Option<String>,

    /// Camera-to-world pose as 12 numbers, the rows of a 3x4 matrix, OpenCV
    /// convention: x right, y down, z forward (with --intrinsics)
    #[arg(long, allow_hyphen_values = true)]
    pub pose: Option<String>,

    #[arg(long, value_enum, default_value_t = ProjectionArg::Perspective)]
    pub projection: ProjectionArg,

    /// Orthographic frame height in world units (0 = fit the scene)
    #[arg(long, default_value_t = 0.0)]
    pub ortho_scale: f32,

    /// Near clip distance (default: from the scene)
    #[arg(long)]
    pub near: Option<f32>,

    /// Far clip distance (default: from the scene)
    #[arg(long)]
    pub far: Option<f32>,

    /// Meshing resolution for 3D models and raster size for 2D sketches
    #[arg(long, default_value_t = 128)]
    pub resolution: usize,

    /// Mesh models without sharp-feature reconstruction
    #[arg(long)]
    pub no_sharp: bool,

    /// Mesh models without the decimation pass
    #[arg(long)]
    pub no_simplify: bool,

    /// Colormap models by a declared sample channel
    #[arg(long)]
    pub color_channel: Option<String>,

    /// Colormap FEA meshes and point clouds by a field, e.g. node:confidence
    #[arg(long)]
    pub color_field: Option<String>,

    /// Overlay mesh edges
    #[arg(long)]
    pub wireframe: bool,

    /// Ground grid spacing in metres (0 disables)
    #[arg(long, default_value_t = 1.0)]
    pub grid: f32,

    /// Disable ambient occlusion
    #[arg(long)]
    pub no_ssao: bool,

    /// Suppress per-asset statistics
    #[arg(short, long)]
    pub quiet: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ProjectionArg {
    Perspective,
    Ortho,
}

/// How each asset is turned into a preview.
struct PlanOptions {
    resolution: usize,
    sharp: bool,
    simplify: bool,
    color_channel: Option<String>,
    color_field: Option<String>,
    wireframe: bool,
}

/// A preset direction, framed to the scene like the viewport's view menu.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ViewPreset {
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
    const ALL: [Self; 8] = [
        Self::Front,
        Self::Back,
        Self::Left,
        Self::Right,
        Self::Top,
        Self::Bottom,
        Self::Iso,
        Self::IsoBack,
    ];

    fn suffix(self) -> &'static str {
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

    fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.suffix() == name)
    }

    /// The orbit camera for this preset, framed to `min..max` at `fov_y`.
    fn camera(self, min: Vec3, max: Vec3, fov_y: f32) -> Camera {
        let mut camera = Camera::new((min + max) * 0.5, 1.0);
        camera.fov_y = fov_y;
        let direction = match self {
            Self::Front => ViewDirection::Front,
            Self::Back => ViewDirection::Back,
            Self::Left => ViewDirection::Left,
            Self::Right => ViewDirection::Right,
            Self::Top => ViewDirection::Top,
            Self::Bottom => ViewDirection::Bottom,
            Self::Iso | Self::IsoBack => ViewDirection::Isometric,
        };
        camera.view_from_direction(direction);
        if self == Self::IsoBack {
            camera.theta += std::f32::consts::PI;
        }
        camera.focus_on(min, max);
        camera.fit_clip_planes();
        camera
    }
}

/// Parses a comma-separated view list; `all` expands to every preset.
fn parse_views(list: &str) -> Result<Vec<ViewPreset>> {
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
        anyhow::bail!("--views names no view");
    }
    Ok(views)
}

/// Where the frame is drawn from.
enum CameraMode {
    Presets(Vec<ViewPreset>),
    Pose {
        eye: Vec3,
        target: Option<Vec3>,
        up: Vec3,
    },
    Pinhole {
        pinhole: Pinhole,
        camera_to_world: Mat4,
    },
}

fn camera_mode(args: &RenderArgs) -> Result<CameraMode> {
    match (&args.intrinsics, &args.pose) {
        (Some(intrinsics), Some(pose)) => {
            if args.camera_pos.is_some() {
                anyhow::bail!(
                    "--camera-pos and --intrinsics/--pose are different cameras; give one"
                );
            }
            if args.projection == ProjectionArg::Ortho {
                anyhow::bail!("a pinhole camera is perspective; drop --projection ortho");
            }
            let k = parse_floats(intrinsics, 4).context("Invalid --intrinsics")?;
            let m = parse_floats(pose, 12).context("Invalid --pose")?;
            let pinhole = Pinhole {
                fx: k[0],
                fy: k[1],
                cx: k[2],
                cy: k[3],
                width: args.width,
                height: args.height,
            };
            let camera_to_world = Mat4::from_cols(
                Vec4::new(m[0], m[4], m[8], 0.0),
                Vec4::new(m[1], m[5], m[9], 0.0),
                Vec4::new(m[2], m[6], m[10], 0.0),
                Vec4::new(m[3], m[7], m[11], 1.0),
            );
            Ok(CameraMode::Pinhole {
                pinhole,
                camera_to_world,
            })
        }
        (None, None) => match &args.camera_pos {
            Some(pos) => Ok(CameraMode::Pose {
                eye: parse_vec3(pos).context("Invalid --camera-pos")?,
                target: args
                    .camera_target
                    .as_deref()
                    .map(parse_vec3)
                    .transpose()
                    .context("Invalid --camera-target")?,
                up: parse_vec3(&args.camera_up).context("Invalid --camera-up")?,
            }),
            None => Ok(CameraMode::Presets(parse_views(&args.views)?)),
        },
        _ => anyhow::bail!("--intrinsics and --pose go together"),
    }
}

/// Near and far planes enclosing `min..max` as seen from `eye` along
/// `forward`: the near plane sits at half the nearest corner's depth but
/// never below a thousandth of the scene, the far plane at twice the
/// farthest corner's.
fn clip_planes_for(eye: Vec3, forward: Vec3, min: Vec3, max: Vec3) -> (f32, f32) {
    let extent = (max - min).length().max(1e-6);
    let (mut nearest, mut farthest) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        );
        let depth = (corner - eye).dot(forward);
        nearest = nearest.min(depth);
        farthest = farthest.max(depth);
    }
    let near = (nearest * 0.5).max(extent * 1e-3);
    let far = (farthest * 2.0).max(near * 10.0);
    (near, far)
}

/// The frames to draw: a file suffix (for several) and the view for each.
fn frames(
    mode: CameraMode,
    args: &RenderArgs,
    bounds: PreviewBounds,
) -> Result<Vec<(Option<&'static str>, CameraView)>> {
    let min = Vec3::from(bounds.min);
    let max = Vec3::from(bounds.max);
    let aspect = args.width as f32 / args.height as f32;
    let fov_y = args.fov.to_radians();
    let ortho_height = |default: f32| {
        if args.ortho_scale > 0.0 {
            args.ortho_scale
        } else {
            default
        }
    };
    let clip = |eye: Vec3, forward: Vec3, default: (f32, f32)| {
        let scene = clip_planes_for(eye, forward, min, max);
        (
            args.near
                .unwrap_or(if default.0 > 0.0 { default.0 } else { scene.0 }),
            args.far
                .unwrap_or(if default.1 > 0.0 { default.1 } else { scene.1 }),
        )
    };

    Ok(match mode {
        CameraMode::Presets(presets) => {
            let several = presets.len() > 1;
            presets
                .into_iter()
                .map(|preset| {
                    let camera = preset.camera(min, max, fov_y);
                    let eye = camera.eye_position();
                    let (near, far) = clip(eye, camera.forward(), (camera.near, camera.far));
                    let view = match args.projection {
                        ProjectionArg::Perspective => CameraView::look_at(
                            eye,
                            camera.target,
                            Vec3::Y,
                            fov_y,
                            aspect,
                            near,
                            far,
                        ),
                        ProjectionArg::Ortho => CameraView::look_at_orthographic(
                            eye,
                            camera.target,
                            Vec3::Y,
                            ortho_height((max - min).length() * 1.1),
                            aspect,
                            near,
                            far,
                        ),
                    };
                    (several.then_some(preset.suffix()), view)
                })
                .collect()
        }
        CameraMode::Pose { eye, target, up } => {
            let target = target.unwrap_or((min + max) * 0.5);
            let forward = (target - eye).normalize_or_zero();
            if forward == Vec3::ZERO {
                anyhow::bail!("--camera-pos coincides with the look-at point");
            }
            let (near, far) = clip(eye, forward, (0.0, 0.0));
            let view = match args.projection {
                ProjectionArg::Perspective => {
                    CameraView::look_at(eye, target, up, fov_y, aspect, near, far)
                }
                ProjectionArg::Ortho => CameraView::look_at_orthographic(
                    eye,
                    target,
                    up,
                    ortho_height((max - min).length() * 1.1),
                    aspect,
                    near,
                    far,
                ),
            };
            vec![(None, view)]
        }
        CameraMode::Pinhole {
            pinhole,
            camera_to_world,
        } => {
            let eye = camera_to_world.transform_point3(Vec3::ZERO);
            let forward = camera_to_world.transform_vector3(Vec3::Z).normalize();
            let (near, far) = clip(eye, forward, (0.0, 0.0));
            vec![(
                None,
                CameraView::pinhole(&pinhole, camera_to_world, near, far),
            )]
        }
    })
}

/// The renderable exports of the input: a model file as one asset, a
/// project's exports filtered to the kinds that have a picture and to
/// `wanted` when given.
fn load_renderable_assets(input: &Path, wanted: &[String]) -> Result<Vec<LoadedAsset>> {
    let extension = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let assets = match extension.as_str() {
        "wasm" => {
            let bytes = std::fs::read(input).context("Failed to read WASM file")?;
            crate::assets::ensure_wasm(&bytes, "model", &input.display().to_string())?;
            let id = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string();
            vec![LoadedAsset::from_parts(
                id,
                bytes,
                Some(AssetTypeHint::Model),
                vec![],
            )]
        }
        "vproj" => {
            let project = Project::load_from_file(input).context("Failed to load .vproj file")?;
            crate::project::run_project_exports(project, None)?
        }
        _ => anyhow::bail!(
            "Unknown file extension: {:?}. Expected .wasm or .vproj",
            extension
        ),
    };
    select_assets(assets, wanted)
}

fn is_renderable(asset: &LoadedAsset) -> bool {
    matches!(
        asset.type_hint(),
        Some(
            AssetTypeHint::Model
                | AssetTypeHint::FeaMesh
                | AssetTypeHint::TriMesh
                | AssetTypeHint::Subspace
        ) | None
    )
}

/// Keeps the renderable assets, or exactly the `wanted` ids, each of which
/// must exist and be renderable.
fn select_assets(assets: Vec<LoadedAsset>, wanted: &[String]) -> Result<Vec<LoadedAsset>> {
    let available = || {
        assets
            .iter()
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
            anyhow::bail!("nothing to draw: no model, mesh, cloud or subspace export");
        }
        return Ok(renderable);
    }
    let mut selected = Vec::with_capacity(wanted.len());
    for id in wanted {
        let asset = assets
            .iter()
            .find(|a| a.id() == id)
            .with_context(|| format!("no export named '{id}'. Available: {}", available()))?;
        if !is_renderable(asset) {
            anyhow::bail!(
                "export '{id}' is {}, which has no picture. Available: {}",
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

/// The viewport's recipe for an asset of this kind, with the CLI's
/// resolution and colour choices.
fn preview_request(asset: &LoadedAsset, options: &PlanOptions) -> PreviewRequest {
    let plan = match asset.type_hint() {
        Some(AssetTypeHint::FeaMesh) => PreviewPlan::FeaMesh {
            deformed: true,
            exaggeration_tenths: 10,
            color_field: options.color_field.clone(),
        },
        Some(AssetTypeHint::TriMesh) => PreviewPlan::TriMesh,
        Some(AssetTypeHint::Subspace) => PreviewPlan::Subspace,
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
        show_grid: false,
        show_bounds: false,
        ssao: false,
        ssao_radius: 0.5,
        ssao_bias: 0.025,
        ssao_strength: 1.0,
        stale: false,
    }
}

fn report(id: &str, entity: &PreviewEntity) {
    let (lo, hi) = (entity.bounds.min, entity.bounds.max);
    eprintln!(
        "{id}: {} triangles, {} points, bounds ({:.3}, {:.3}, {:.3})..({:.3}, {:.3}, {:.3}), {:.0} ms",
        entity.stats.triangles,
        entity.stats.points,
        lo.0,
        lo.1,
        lo.2,
        hi.0,
        hi.1,
        hi.2,
        entity.stats.mesh_ms
    );
    for line in &entity.stats.detail {
        eprintln!("  {line}");
    }
}

fn parse_floats(s: &str, count: usize) -> Result<Vec<f32>> {
    let values: Vec<f32> = s
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("expected {count} comma-separated numbers, got '{s}'"))?;
    if values.len() != count {
        anyhow::bail!(
            "expected {count} comma-separated numbers, got {}",
            values.len()
        );
    }
    Ok(values)
}

fn parse_vec3(s: &str) -> Result<Vec3> {
    let v = parse_floats(s, 3)?;
    Ok(Vec3::new(v[0], v[1], v[2]))
}

/// A hex sRGB colour as the linear RGBA the renderer clears with.
fn parse_hex_color(hex: &str) -> Result<[f32; 4]> {
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

/// `base` with `_suffix` before the extension when a suffix is given.
fn output_path(base: &Path, suffix: Option<&str>) -> PathBuf {
    let Some(suffix) = suffix else {
        return base.to_path_buf();
    };
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("render");
    let extension = base.extension().and_then(|e| e.to_str()).unwrap_or("png");
    base.with_file_name(format!("{stem}_{suffix}.{extension}"))
}

pub fn run_render(args: RenderArgs) -> Result<()> {
    let background = parse_hex_color(&args.background).context("Invalid --background")?;
    let mode = camera_mode(&args)?;
    if args.width == 0 || args.height == 0 {
        anyhow::bail!("--width and --height must be positive");
    }

    let assets = load_renderable_assets(&args.input, &args.assets)?;
    let options = PlanOptions {
        resolution: args.resolution,
        sharp: !args.no_sharp,
        simplify: !args.no_simplify,
        color_channel: args.color_channel.clone(),
        color_field: args.color_field.clone(),
        wireframe: args.wireframe,
    };

    let mut entities: Vec<PreviewEntity> = Vec::with_capacity(assets.len());
    for asset in &assets {
        let request = preview_request(asset, &options);
        let entity = build_preview_scene(&request)
            .map_err(|err| anyhow::anyhow!("{}: {err}", asset.id()))?;
        if !args.quiet {
            report(asset.id(), &entity);
        }
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
    let frames = frames(mode, &args, bounds)?;

    let offscreen = Offscreen::new().map_err(anyhow::Error::msg)?;
    if !args.quiet {
        eprintln!("GPU: {}", offscreen.adapter_name());
    }
    let mut renderer = offscreen.renderer(args.width, args.height);
    let resident: Vec<_> = entities
        .iter()
        .map(|entity| renderer.create_retained_scene(offscreen.device(), &entity.scene))
        .collect();

    let mut settings = RenderSettings {
        background_color: background,
        ssao_enabled: !args.no_ssao,
        ..RenderSettings::default()
    };
    if args.grid > 0.0 {
        let extent = (Vec3::from(bounds.max) - Vec3::from(bounds.min)).length();
        settings.grid.planes = GridPlanes::XZ;
        settings.grid.spacing = args.grid;
        settings.grid.extent = (extent * 2.0).max(args.grid * 10.0);
    } else {
        settings.grid.planes = GridPlanes::NONE;
    }

    for (suffix, view) in frames {
        for (scene, entity) in resident.iter().zip(&entities) {
            for mesh in &scene.meshes {
                renderer.submit_retained_mesh(mesh);
            }
            for lines in &scene.lines {
                renderer.submit_retained_lines(lines);
            }
            for points in &scene.points {
                renderer.submit_retained_points(points);
            }
            if args.wireframe
                && let Some(lines) = &entity.wireframe_lines
            {
                renderer.submit_lines(lines, Mat4::IDENTITY, wireframe_style());
            }
            if let Some(subspace) = &entity.subspace {
                submit_subspace_gizmo(&mut renderer, subspace, bounds);
            }
        }
        let rgba = offscreen
            .render_rgba(&mut renderer, &view, &settings)
            .map_err(anyhow::Error::msg)?;
        if let Some(overflow) = renderer.frame_overflow() {
            eprintln!(
                "warning: dropped {} of {} triangles, {} lines and {} points at the GPU buffer limit",
                overflow.dropped_triangles,
                overflow.total_triangles,
                overflow.dropped_lines,
                overflow.dropped_points
            );
        }
        let path = output_path(&args.output, suffix);
        image::RgbaImage::from_raw(args.width, args.height, rgba)
            .context("frame size mismatch")?
            .save(&path)
            .with_context(|| format!("Failed to write {}", path.display()))?;
        println!("Wrote {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(id: &str, type_hint: Option<AssetTypeHint>) -> LoadedAsset {
        LoadedAsset::from_parts(id.to_string(), vec![1, 2, 3], type_hint, vec![])
    }

    #[test]
    fn views_parse_and_name_their_files() {
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

        assert_eq!(
            output_path(Path::new("out/render.png"), Some("top")),
            PathBuf::from("out/render_top.png")
        );
        assert_eq!(
            output_path(Path::new("render.png"), None),
            PathBuf::from("render.png")
        );
    }

    /// Every preset looks at the scene centre from outside the box.
    #[test]
    fn presets_frame_the_scene() {
        let (min, max) = (Vec3::new(-1.0, 0.0, -2.0), Vec3::new(1.0, 1.0, 2.0));
        for preset in ViewPreset::ALL {
            let camera = preset.camera(min, max, 0.8);
            assert_eq!(camera.target, (min + max) * 0.5);
            let eye = camera.eye_position();
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
        let iso = ViewPreset::Iso.camera(min, max, 0.8).eye_position();
        let back = ViewPreset::IsoBack.camera(min, max, 0.8).eye_position();
        assert!((iso.x + back.x).abs() < 1e-4 && (iso.z + back.z).abs() < 1e-4);
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
        let ids = |assets: &[LoadedAsset]| {
            assets
                .iter()
                .map(|a| a.id().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids(&select_assets(all.clone(), &[]).unwrap()),
            ["scan", "axis", "cloud"]
        );
        assert_eq!(
            ids(&select_assets(all.clone(), &["cloud".to_string(), "scan".to_string()]).unwrap()),
            ["cloud", "scan"]
        );
        let missing = select_assets(all.clone(), &["nope".to_string()]).unwrap_err();
        assert!(
            missing.to_string().contains("scan, axis, cloud"),
            "{missing}"
        );
        let wrong = select_assets(all, &["fit".to_string()]).unwrap_err();
        assert!(wrong.to_string().contains("F64Map"), "{wrong}");
        assert!(select_assets(vec![asset("fit", Some(AssetTypeHint::F64Map))], &[]).is_err());
    }

    #[test]
    fn plans_follow_the_asset_kind() {
        let options = PlanOptions {
            resolution: 64,
            sharp: false,
            simplify: true,
            color_channel: None,
            color_field: Some("node:confidence".to_string()),
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

    #[test]
    fn pinhole_pose_rows_build_the_camera_to_world_matrix() {
        let args = RenderArgs::parse_from([
            "render",
            "-i",
            "x.vproj",
            "-o",
            "x.png",
            "--intrinsics",
            "400,410,320,240",
            "--pose",
            "1,0,0,5, 0,1,0,6, 0,0,1,7",
        ]);
        match camera_mode(&args).unwrap() {
            CameraMode::Pinhole {
                pinhole,
                camera_to_world,
            } => {
                assert_eq!(
                    (pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy),
                    (400.0, 410.0, 320.0, 240.0)
                );
                assert_eq!(
                    camera_to_world.transform_point3(Vec3::ZERO),
                    Vec3::new(5.0, 6.0, 7.0)
                );
                assert_eq!(camera_to_world.transform_vector3(Vec3::Z), Vec3::Z);
            }
            _ => panic!("expected a pinhole camera"),
        }
        let half = RenderArgs::parse_from([
            "render",
            "-i",
            "x.vproj",
            "-o",
            "x.png",
            "--pose",
            "1,0,0,0,0,1,0,0,0,0,1,0",
        ]);
        assert!(camera_mode(&half).is_err());
    }

    #[test]
    fn background_is_linearised() {
        let white = parse_hex_color("ffffff").unwrap();
        assert!((white[0] - 1.0).abs() < 1e-6);
        let grey = parse_hex_color("#808080").unwrap();
        assert!(grey[0] > 0.2 && grey[0] < 0.22, "{grey:?}");
        assert!(parse_hex_color("12345").is_err());
    }
}
