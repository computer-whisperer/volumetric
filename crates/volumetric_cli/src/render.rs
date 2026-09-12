//! `render`: draws a model, or the exports of a project, to PNG through
//! `volumetric_render`, the headless frame the GUI viewport also shows
//! (see that crate for what each asset kind becomes). This command is the
//! flags, the PNG files and the printed report.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use glam::Vec3;

use volumetric::{AssetTypeHint, LoadedAsset, Project};
use volumetric_preview::pose_matrix;
use volumetric_render::{
    CameraSpec, ColorRange, Overlay, Pinhole, PlanOptions, Projection, RenderOptions,
    background_from_hex, parse_views, select_assets,
};

#[derive(Parser, Debug)]
pub struct RenderArgs {
    /// Input file: a .wasm model or a .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// For .vproj inputs: an asset to draw (repeatable; default: every
    /// renderable export; imports such as a view set draw only when named)
    #[arg(long = "asset")]
    pub assets: Vec<String>,

    /// Output PNG path (a view suffix is added when several views render)
    #[arg(short, long)]
    pub output: PathBuf,

    /// Image width (default: 1024, or the view's camera with --through)
    #[arg(long)]
    pub width: Option<u32>,

    /// Image height (default: 1024, or the view's camera with --through)
    #[arg(long)]
    pub height: Option<u32>,

    /// Comma-separated preset views: front, back, left, right, top, bottom,
    /// iso, iso-back, all
    #[arg(long, default_value = "iso")]
    pub views: String,

    /// The world's up direction x,y,z: orients the preset views, the ground
    /// grid and the default --camera-up (default: the up of a drawn view set
    /// or splat, else 0,1,0)
    #[arg(long, allow_hyphen_values = true)]
    pub up: Option<String>,

    /// Background colour as hex sRGB (e.g. 2d2d2d)
    #[arg(long, default_value = "2d2d2d")]
    pub background: String,

    /// Camera position x,y,z (an explicit camera replaces --views)
    #[arg(long, allow_hyphen_values = true)]
    pub camera_pos: Option<String>,

    /// Look-at point x,y,z (default: the scene centre)
    #[arg(long, allow_hyphen_values = true)]
    pub camera_target: Option<String>,

    /// Up vector x,y,z of the explicit camera (default: --up)
    #[arg(long, allow_hyphen_values = true)]
    pub camera_up: Option<String>,

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

    /// Look through a view of the project's view set: `<view id>` or
    /// `<views asset>:<view id>`
    #[arg(long)]
    pub through: Option<String>,

    /// With --through: composite the render over the photograph as blend,
    /// edge, side or checker
    #[arg(long)]
    pub overlay: Option<String>,

    /// Render opacity for the blend overlay
    #[arg(long, default_value_t = 0.5)]
    pub overlay_alpha: f32,

    /// Tile size in pixels for the checker overlay
    #[arg(long, default_value_t = 64)]
    pub overlay_tile: u32,

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

    /// With --color-field: the values the colormap spans, lo,hi in the
    /// field's units; values beyond take the end colours (default: the
    /// field's own range)
    #[arg(long, allow_hyphen_values = true)]
    pub color_range: Option<String>,

    /// Overlay mesh edges
    #[arg(long)]
    pub wireframe: bool,

    /// With --through: draw what the view observed (marker quads, card
    /// corners, recorded picks and contours) over the frame
    #[arg(long)]
    pub marks: bool,

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

/// The camera the flags describe, with the flag conflicts refused here so
/// the messages name the flags.
fn camera_mode(args: &RenderArgs) -> Result<CameraSpec> {
    if let Some(through) = &args.through {
        if args.camera_pos.is_some() || args.intrinsics.is_some() || args.pose.is_some() {
            anyhow::bail!(
                "--through is a camera of its own; drop --camera-pos, --intrinsics and --pose"
            );
        }
        if args.projection == ProjectionArg::Ortho {
            anyhow::bail!("a view's camera is perspective; drop --projection ortho");
        }
        let (asset, view) = match through.split_once(':') {
            Some((asset, view)) => (Some(asset.to_string()), view.to_string()),
            None => (None, through.clone()),
        };
        if view.is_empty() {
            anyhow::bail!("--through needs a view id");
        }
        return Ok(CameraSpec::Through { asset, view });
    }
    if args.overlay.is_some() {
        anyhow::bail!("--overlay composites over a view's photograph; give --through");
    }
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
                width: args.width.unwrap_or(1024),
                height: args.height.unwrap_or(1024),
            };
            let rows: [f64; 12] = std::array::from_fn(|i| f64::from(m[i]));
            Ok(CameraSpec::Pinhole {
                pinhole,
                camera_to_world: pose_matrix(&rows),
            })
        }
        (None, None) => match &args.camera_pos {
            Some(pos) => Ok(CameraSpec::LookAt {
                eye: parse_vec3(pos).context("Invalid --camera-pos")?,
                target: args
                    .camera_target
                    .as_deref()
                    .map(parse_vec3)
                    .transpose()
                    .context("Invalid --camera-target")?,
                up: args
                    .camera_up
                    .as_deref()
                    .map(parse_vec3)
                    .transpose()
                    .context("Invalid --camera-up")?,
            }),
            None => Ok(CameraSpec::Presets(
                parse_views(&args.views).context("Invalid --views")?,
            )),
        },
        _ => anyhow::bail!("--intrinsics and --pose go together"),
    }
}

/// The frame options the flags describe.
fn render_options(args: &RenderArgs) -> Result<RenderOptions> {
    let overlay = match &args.overlay {
        Some(name) => Some(
            Overlay::parse(name, args.overlay_alpha, args.overlay_tile).with_context(|| {
                format!("unknown overlay '{name}'; expected blend, edge, side or checker")
            })?,
        ),
        None => None,
    };
    let color_range = match &args.color_range {
        Some(text) => {
            let v = parse_floats(text, 2).context("Invalid --color-range")?;
            Some(
                ColorRange::new(f64::from(v[0]), f64::from(v[1]))
                    .context("--color-range needs lo below hi")?,
            )
        }
        None => None,
    };
    let up = match &args.up {
        Some(up) => {
            let up = parse_vec3(up).context("Invalid --up")?;
            if up.normalize_or_zero() == Vec3::ZERO {
                anyhow::bail!("--up must not be zero");
            }
            Some(up)
        }
        None => None,
    };
    if args.width == Some(0) || args.height == Some(0) {
        anyhow::bail!("--width and --height must be positive");
    }
    Ok(RenderOptions {
        width: args.width,
        height: args.height,
        projection: match args.projection {
            ProjectionArg::Perspective => Projection::Perspective,
            ProjectionArg::Ortho => Projection::Orthographic,
        },
        fov_deg: args.fov,
        ortho_scale: args.ortho_scale,
        near: args.near,
        far: args.far,
        up,
        background: background_from_hex(&args.background).context("Invalid --background")?,
        grid: args.grid,
        ssao: !args.no_ssao,
        plan: PlanOptions {
            resolution: args.resolution,
            sharp: !args.no_sharp,
            simplify: !args.no_simplify,
            color_channel: args.color_channel.clone(),
            color_field: args.color_field.clone(),
            color_range,
            wireframe: args.wireframe,
        },
        overlay,
        marks: args.marks,
    })
}

/// The assets to draw and a project's imports: a model file as one asset,
/// or a project's exports filtered to the kinds that have a picture and to
/// `wanted` when given (which may also name imports).
fn load_renderable_assets(
    input: &Path,
    wanted: &[String],
) -> Result<(Vec<LoadedAsset>, Vec<LoadedAsset>)> {
    let extension = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mut imports = Vec::new();
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
            imports = volumetric::asset_query::imports_as_assets(&project);
            crate::project::run_project_exports(project, None)?
        }
        _ => anyhow::bail!(
            "Unknown file extension: {:?}. Expected .wasm or .vproj",
            extension
        ),
    };
    let selected = select_assets(assets, &imports, wanted)
        .map_err(|err| anyhow::anyhow!("{err:#} (imports draw when named with --asset)"))?;
    Ok((selected, imports))
}

pub(crate) fn parse_floats(s: &str, count: usize) -> Result<Vec<f32>> {
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

pub(crate) fn parse_vec3(s: &str) -> Result<Vec3> {
    let v = parse_floats(s, 3)?;
    Ok(Vec3::new(v[0], v[1], v[2]))
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
    let camera = camera_mode(&args)?;
    let options = render_options(&args)?;
    let (assets, imports) = load_renderable_assets(&args.input, &args.assets)?;
    let rendered = volumetric_render::render(&assets, &imports, camera, &options)?;
    let report = &rendered.report;
    if !args.quiet {
        for entity in &report.entities {
            let (lo, hi) = (entity.bounds.min, entity.bounds.max);
            eprintln!(
                "{}: {} triangles, {} points, bounds ({:.3}, {:.3}, {:.3})..({:.3}, {:.3}, {:.3}), {:.0} ms",
                entity.id,
                entity.triangles,
                entity.points,
                lo.0,
                lo.1,
                lo.2,
                hi.0,
                hi.1,
                hi.2,
                entity.mesh_ms
            );
            for line in &entity.detail {
                eprintln!("  {line}");
            }
        }
        eprintln!(
            "up: ({}, {}, {}) from {}",
            report.up.x,
            report.up.y,
            report.up.z,
            if report.up_source == "options" {
                "--up"
            } else {
                &report.up_source
            }
        );
        eprintln!("GPU: {}", report.gpu);
    }
    for note in &report.notes {
        eprintln!("{note}");
    }
    for frame in &rendered.frames {
        let path = output_path(&args.output, frame.suffix);
        image::RgbaImage::from_raw(frame.width, frame.height, frame.rgba.clone())
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
    use clap::Parser;

    #[test]
    fn several_views_name_their_files() {
        assert_eq!(
            output_path(Path::new("out/render.png"), Some("top")),
            PathBuf::from("out/render_top.png")
        );
        assert_eq!(
            output_path(Path::new("render.png"), None),
            PathBuf::from("render.png")
        );
    }

    fn parse(extra: &[&str]) -> RenderArgs {
        let mut argv = vec!["render", "-i", "x.vproj", "-o", "x.png"];
        argv.extend_from_slice(extra);
        RenderArgs::parse_from(argv)
    }

    #[test]
    fn pinhole_pose_rows_build_the_camera_to_world_matrix() {
        let args = parse(&[
            "--intrinsics",
            "400,410,320,240",
            "--pose",
            "1,0,0,5, 0,1,0,6, 0,0,1,7",
        ]);
        match camera_mode(&args).unwrap() {
            CameraSpec::Pinhole {
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
        assert!(camera_mode(&parse(&["--pose", "1,0,0,0,0,1,0,0,0,0,1,0"])).is_err());
    }

    /// Flag conflicts are refused by name; the rest maps onto the library.
    #[test]
    fn flags_map_onto_the_library_options() {
        let through = parse(&[
            "--through",
            "photos:DSC01",
            "--overlay",
            "edge",
            "--no-ssao",
        ]);
        assert_eq!(
            camera_mode(&through).unwrap(),
            CameraSpec::Through {
                asset: Some("photos".to_string()),
                view: "DSC01".to_string()
            }
        );
        let options = render_options(&through).unwrap();
        assert_eq!(options.overlay, Some(Overlay::Edge));
        assert!(!options.ssao && options.plan.sharp && options.width.is_none());
        let err = camera_mode(&parse(&["--through", "v", "--camera-pos", "0,0,1"])).unwrap_err();
        assert!(err.to_string().contains("--through"), "{err}");
        let err = camera_mode(&parse(&["--overlay", "edge"])).unwrap_err();
        assert!(err.to_string().contains("--through"), "{err}");
        let presets = camera_mode(&parse(&["--views", "top,front"])).unwrap();
        assert!(matches!(presets, CameraSpec::Presets(ref v) if v.len() == 2));
        assert!(render_options(&parse(&["--up", "0,0,0"])).is_err());
        assert!(render_options(&parse(&["--color-range", "1,0"])).is_err());
        let ortho =
            render_options(&parse(&["--projection", "ortho", "--ortho-scale", "2"])).unwrap();
        assert_eq!(ortho.projection, Projection::Orthographic);
        assert_eq!(ortho.ortho_scale, 2.0);
    }
}
