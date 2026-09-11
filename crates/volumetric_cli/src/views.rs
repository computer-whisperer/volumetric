//! View sets on the command line: `view-import` embeds a selection of a
//! posed-image dataset into a `.vviews` file or a project, `view-list`
//! describes one, and `view-residual` compares a model against the depth
//! maps its views carry.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use view_core::image::{decode_depth, decode_mask};
use view_core::residual::{ResidualStats, Search, depth_residual, residual_image};
use view_core::{Eye, Labels, Selection, import_manifest};
use volumetric::wasm::ParallelModelSampler;
use volumetric::{AssetTypeHint, ImportedAsset, LoadedAsset, Project};
use volumetric_abi::viewset::{Distortion, ViewSet, decode_viewset, encode_viewset};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum EyeArg {
    Left,
    Right,
    Both,
}

#[derive(Parser, Debug)]
pub struct ViewImportArgs {
    /// The dataset manifest: the scanner's cameras.json or nerfstudio's
    /// transforms.json (files are read relative to it)
    #[arg(short, long)]
    pub manifest: PathBuf,

    /// Write the view set here (.vviews)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Add the view set to this project as an imported asset
    #[arg(short, long)]
    pub project: Option<PathBuf>,

    /// Asset id in the project
    #[arg(long, default_value = "views")]
    pub asset_id: String,

    /// Keep only this view id (repeatable)
    #[arg(long = "id")]
    pub ids: Vec<String>,

    /// Keep every Nth view after the other filters
    #[arg(long, default_value_t = 1)]
    pub stride: usize,

    /// Keep views whose camera lies within --radius of this point x,y,z
    #[arg(long, allow_hyphen_values = true)]
    pub near: Option<String>,

    #[arg(long, default_value_t = 1.0)]
    pub radius: f64,

    /// At most this many views (0 = no cap)
    #[arg(long, default_value_t = 0)]
    pub max: usize,

    #[arg(long, value_enum, default_value_t = EyeArg::Both)]
    pub eye: EyeArg,

    /// Keep views tagged with this split (train, test)
    #[arg(long)]
    pub split: Option<String>,

    #[arg(long)]
    pub no_images: bool,

    #[arg(long)]
    pub no_depth: bool,

    #[arg(long)]
    pub no_masks: bool,

    /// Provenance: the capture session (default: the manifest's)
    #[arg(long)]
    pub session: Option<String>,

    /// Provenance: the capture system's calibration id
    #[arg(long)]
    pub rig: Option<String>,

    /// Provenance: the tracking field (marker map) id
    #[arg(long)]
    pub field: Option<String>,

    /// Provenance: the subject setup id
    #[arg(long)]
    pub setup: Option<String>,
}

pub fn run_view_import(args: ViewImportArgs) -> Result<()> {
    if args.output.is_none() && args.project.is_none() {
        bail!("give --output, --project, or both");
    }
    let near = args
        .near
        .as_deref()
        .map(|s| {
            let v = crate::render::parse_vec3(s).context("Invalid --near")?;
            Ok::<_, anyhow::Error>((
                [f64::from(v.x), f64::from(v.y), f64::from(v.z)],
                args.radius,
            ))
        })
        .transpose()?;
    let selection = Selection {
        ids: args.ids.clone(),
        stride: args.stride.max(1),
        near,
        max: args.max,
        eye: match args.eye {
            EyeArg::Left => Eye::Left,
            EyeArg::Right => Eye::Right,
            EyeArg::Both => Eye::Both,
        },
        split: args.split.clone(),
        images: !args.no_images,
        depth: !args.no_depth,
        masks: !args.no_masks,
    };
    let labels = Labels {
        session: args.session.clone(),
        rig: args.rig.clone(),
        field: args.field.clone(),
        setup: args.setup.clone(),
    };
    let (set, report) = import_manifest(&args.manifest, &selection, &labels)?;
    let bytes = encode_viewset(&set);
    println!(
        "Imported {} of {} views from {}: {} images, {} depth maps, {} masks, {:.1} MB embedded; {} camera(s), {} markers",
        report.selected,
        report.total,
        report.kind,
        report.with_image,
        report.with_depth,
        report.with_mask,
        report.bytes as f64 / 1e6,
        set.cameras.len(),
        set.markers.len()
    );
    if let Some(output) = &args.output {
        std::fs::write(output, &bytes)
            .with_context(|| format!("Failed to write {}", output.display()))?;
        println!("Wrote {} ({} bytes)", output.display(), bytes.len());
    }
    if let Some(path) = &args.project {
        let mut project = Project::load_from_file(path).context("Failed to load project")?;
        let asset_id = project.unique_asset_id(&args.asset_id);
        project.imports_mut().push(ImportedAsset::new(
            asset_id.clone(),
            bytes,
            Some(AssetTypeHint::ViewSet),
        ));
        project
            .save_to_file(path)
            .with_context(|| format!("Failed to save {}", path.display()))?;
        println!("Added ViewSet asset '{asset_id}' to {}", path.display());
    }
    Ok(())
}

/// A project's imported assets as loaded assets.
pub(crate) fn imports_of(project: &Project) -> Vec<LoadedAsset> {
    project
        .imports()
        .iter()
        .map(|import| {
            LoadedAsset::from_parts(
                import.id.clone(),
                import.data.clone(),
                import.type_hint,
                vec![],
            )
        })
        .collect()
}

/// Every asset of a project as loaded assets: imports first, then the
/// exports of a run when `run` is set.
pub(crate) fn project_assets(path: &Path, run: bool) -> Result<Vec<LoadedAsset>> {
    let project = Project::load_from_file(path).context("Failed to load project")?;
    let mut assets = imports_of(&project);
    if run {
        for export in crate::project::run_project_exports(project, None)? {
            if !assets.iter().any(|a| a.id() == export.id()) {
                assets.push(export);
            }
        }
    }
    Ok(assets)
}

/// The view set asset named by `wanted` among `assets`, or the only one.
pub(crate) fn find_viewset_asset<'a>(
    assets: &'a [LoadedAsset],
    wanted: Option<&str>,
) -> Result<&'a LoadedAsset> {
    let sets: Vec<&LoadedAsset> = assets
        .iter()
        .filter(|a| a.type_hint() == Some(AssetTypeHint::ViewSet))
        .collect();
    match wanted {
        Some(id) => sets
            .iter()
            .find(|a| a.id() == id)
            .copied()
            .ok_or_else(|| anyhow!("no view set asset '{id}'. Available: {}", ids(&sets))),
        None => match sets.as_slice() {
            [] => bail!("no view set asset in the project"),
            [only] => Ok(only),
            _ => bail!("several view sets; choose one with --views: {}", ids(&sets)),
        },
    }
}

/// Asset ids joined for an error message.
fn ids(assets: &[&LoadedAsset]) -> String {
    assets
        .iter()
        .map(|a| a.id().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The view set named by `wanted` among `assets`, or the only one.
pub(crate) fn find_viewset(assets: &[LoadedAsset], wanted: Option<&str>) -> Result<ViewSet> {
    let asset = find_viewset_asset(assets, wanted)?;
    decode_viewset(asset.data()).map_err(|err| anyhow!("asset '{}': {err}", asset.id()))
}

fn load_viewset(input: &Path, asset: Option<&str>) -> Result<ViewSet> {
    let extension = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if extension == "vproj" {
        // Imports carry view sets today; exports may once operators produce
        // them, so run the project only when the imports have none.
        let imports = project_assets(input, false)?;
        if imports
            .iter()
            .any(|a| a.type_hint() == Some(AssetTypeHint::ViewSet))
        {
            return find_viewset(&imports, asset);
        }
        return find_viewset(&project_assets(input, true)?, asset);
    }
    let bytes =
        std::fs::read(input).with_context(|| format!("Failed to read {}", input.display()))?;
    decode_viewset(&bytes).map_err(|err| anyhow!("{} is not a view set: {err}", input.display()))
}

#[derive(Parser, Debug)]
pub struct ViewListArgs {
    /// A .vviews file or a .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// For projects with several view sets: which one
    #[arg(long)]
    pub asset: Option<String>,

    #[arg(long)]
    pub json: bool,
}

#[derive(Serialize)]
struct CameraSummary {
    index: usize,
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    distortion: String,
}

#[derive(Serialize)]
struct ViewSummary {
    id: String,
    camera: u32,
    position: [f64; 3],
    forward: [f64; 3],
    image: bool,
    depth: bool,
    mask: bool,
    tags: Vec<String>,
}

#[derive(Serialize)]
struct SetSummary {
    schema: u32,
    up: [f64; 3],
    provenance: volumetric_abi::viewset::Provenance,
    cameras: Vec<CameraSummary>,
    views: Vec<ViewSummary>,
    markers: usize,
}

fn distortion_label(distortion: &Distortion) -> String {
    match distortion {
        Distortion::None => "none".to_string(),
        Distortion::Radial { k, p } => format!("radial k={k:?} p={p:?}"),
        Distortion::KannalaBrandt { k } => format!("fisheye k={k:?}"),
    }
}

fn summarize(set: &ViewSet) -> SetSummary {
    SetSummary {
        schema: set.schema,
        up: set.world.up,
        provenance: set.provenance.clone(),
        cameras: set
            .cameras
            .iter()
            .enumerate()
            .map(|(index, c)| CameraSummary {
                index,
                width: c.width,
                height: c.height,
                fx: c.fx,
                fy: c.fy,
                cx: c.cx,
                cy: c.cy,
                distortion: distortion_label(&c.distortion),
            })
            .collect(),
        views: set
            .views
            .iter()
            .map(|v| ViewSummary {
                id: v.id.clone(),
                camera: v.camera,
                position: v.position(),
                forward: v.forward(),
                image: v.image.is_some(),
                depth: v.depth.is_some(),
                mask: v.mask.is_some(),
                tags: v.tags.clone(),
            })
            .collect(),
        markers: set.markers.len(),
    }
}

pub fn run_view_list(args: ViewListArgs) -> Result<()> {
    let set = load_viewset(&args.input, args.asset.as_deref())?;
    let summary = summarize(&set);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    let p = &summary.provenance;
    println!(
        "View set schema {}: {} views, {} camera(s), {} markers, up ({}, {}, {})",
        summary.schema,
        summary.views.len(),
        summary.cameras.len(),
        summary.markers,
        summary.up[0],
        summary.up[1],
        summary.up[2]
    );
    println!(
        "Provenance: session {:?}, rig {:?}, field {:?}, setup {:?}, captured {:?}",
        p.session, p.rig, p.field, p.setup, p.captured
    );
    for tool in &p.tools {
        println!("  tool: {tool}");
    }
    for c in &summary.cameras {
        println!(
            "Camera {}: {}x{} fx {:.3} fy {:.3} cx {:.3} cy {:.3}, distortion {}",
            c.index, c.width, c.height, c.fx, c.fy, c.cx, c.cy, c.distortion
        );
    }
    println!("Views:");
    for v in &summary.views {
        println!(
            "  {:<12} cam {} at ({:+.3}, {:+.3}, {:+.3}) looking ({:+.2}, {:+.2}, {:+.2}) {}{}{} {}",
            v.id,
            v.camera,
            v.position[0],
            v.position[1],
            v.position[2],
            v.forward[0],
            v.forward[1],
            v.forward[2],
            if v.image { "image " } else { "" },
            if v.depth { "depth " } else { "" },
            if v.mask { "mask " } else { "" },
            v.tags.join(",")
        );
    }
    Ok(())
}

#[derive(Parser, Debug)]
pub struct ViewResidualArgs {
    /// Project holding the view set and the model
    #[arg(short, long)]
    pub project: PathBuf,

    /// The view set asset (default: the only one)
    #[arg(long)]
    pub views: Option<String>,

    /// The model asset (default: the only model export)
    #[arg(long)]
    pub model: Option<String>,

    /// Compare only this view (repeatable; default: every view with depth)
    #[arg(long = "view")]
    pub view_ids: Vec<String>,

    /// Pixel lattice stride
    #[arg(long, default_value_t = 2)]
    pub stride: u32,

    /// Half-width of the depth band searched around the scan surface, in metres
    #[arg(long, default_value_t = 0.05)]
    pub band: f64,

    /// Marching step within the band, in metres
    #[arg(long, default_value_t = 0.002)]
    pub step: f64,

    /// Compare every measured pixel, not only the subject mask
    #[arg(long)]
    pub no_mask: bool,

    /// Write a residual image per view into this directory
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Residual that saturates the residual image's colours, in metres
    #[arg(long, default_value_t = 0.02)]
    pub scale: f32,

    #[arg(long)]
    pub json: bool,
}

#[derive(Serialize)]
struct ViewResidualReport {
    id: String,
    stats: ResidualStatsJson,
}

#[derive(Serialize)]
struct ResidualStatsJson {
    pixels: usize,
    hits: usize,
    coverage: f64,
    median_abs_m: f64,
    p90_abs_m: f64,
    mean_m: f64,
    rms_m: f64,
}

impl From<&ResidualStats> for ResidualStatsJson {
    fn from(s: &ResidualStats) -> Self {
        Self {
            pixels: s.pixels,
            hits: s.hits,
            coverage: s.coverage,
            median_abs_m: s.median_abs,
            p90_abs_m: s.p90_abs,
            mean_m: s.mean,
            rms_m: s.rms,
        }
    }
}

#[derive(Serialize)]
struct ResidualReport {
    model: String,
    views: Vec<ViewResidualReport>,
    pooled: ResidualStatsJson,
}

fn stats_line(stats: &ResidualStats) -> String {
    format!(
        "{} pixels, coverage {:.1}%, median {:.1} mm, p90 {:.1} mm, mean {:+.1} mm, rms {:.1} mm",
        stats.pixels,
        stats.coverage * 100.0,
        stats.median_abs * 1e3,
        stats.p90_abs * 1e3,
        stats.mean * 1e3,
        stats.rms * 1e3
    )
}

pub fn run_view_residual(args: ViewResidualArgs) -> Result<()> {
    let assets = project_assets(&args.project, true)?;
    let set = find_viewset(&assets, args.views.as_deref())?;

    let models: Vec<&LoadedAsset> = assets.iter().filter(|a| a.as_model().is_some()).collect();
    let model = match &args.model {
        Some(id) => models
            .iter()
            .find(|a| a.id() == id)
            .copied()
            .ok_or_else(|| anyhow!("no model asset '{id}'. Available: {}", ids(&models)))?,
        None => match models.as_slice() {
            [] => bail!("the project has no model"),
            [only] => only,
            _ => bail!("several models; choose one with --model: {}", ids(&models)),
        },
    };
    let sampler = volumetric::wasm::create_parallel_sampler(model.data())
        .map_err(|err| anyhow!("model '{}': {err}", model.id()))?;
    let inside = |p: [f64; 3]| volumetric_abi::is_occupied(sampler.sample(p[0], p[1], p[2]));

    let search = Search {
        band: args.band,
        step: args.step,
        stride: args.stride.max(1),
    };
    if let Some(dir) = &args.output {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create {}", dir.display()))?;
    }

    let mut reports = Vec::new();
    let mut pooled_pixels = 0usize;
    let mut pooled_hits: Vec<f32> = Vec::new();
    for view in &set.views {
        if !args.view_ids.is_empty() && !args.view_ids.contains(&view.id) {
            continue;
        }
        let Some(depth_bytes) = &view.depth else {
            if !args.view_ids.is_empty() {
                eprintln!("warning: view {} has no depth map", view.id);
            }
            continue;
        };
        let camera = set.camera_of(view);
        let depth = decode_depth(depth_bytes, view.depth_unit_m)
            .with_context(|| format!("view {}", view.id))?;
        let mask = if args.no_mask {
            None
        } else {
            view.mask
                .as_deref()
                .map(decode_mask)
                .transpose()
                .with_context(|| format!("view {}", view.id))?
        };
        let residual = depth_residual(view, camera, &depth, mask.as_ref(), &search, &inside);
        if !args.json {
            println!("{}: {}", view.id, stats_line(&residual.stats));
        }
        if let Some(dir) = &args.output {
            let path = dir.join(format!("{}_residual.png", view.id));
            std::fs::write(&path, residual_image(&residual, args.scale).to_png()?)
                .with_context(|| format!("Failed to write {}", path.display()))?;
        }
        pooled_pixels += residual.stats.pixels;
        pooled_hits.extend(residual.values.iter().copied().filter(|v| v.is_finite()));
        reports.push(ViewResidualReport {
            id: view.id.clone(),
            stats: (&residual.stats).into(),
        });
    }
    if reports.is_empty() {
        bail!("no view with a depth map matched");
    }
    if sampler.instantiation_failures() > 0 {
        bail!(
            "model sampling failed on {} thread(s): {}",
            sampler.instantiation_failures(),
            sampler.instantiation_failure_detail().unwrap_or_default()
        );
    }
    if sampler.sample_traps() > 0 {
        eprintln!(
            "warning: {} model sample(s) trapped and were read as outside",
            sampler.sample_traps()
        );
    }
    let pooled = ResidualStats::from_residuals(pooled_pixels, &pooled_hits);
    if args.json {
        let report = ResidualReport {
            model: model.id().to_string(),
            views: reports,
            pooled: (&pooled).into(),
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "pooled over {} view(s) against '{}': {}",
            reports.len(),
            model.id(),
            stats_line(&pooled)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::viewset::{CameraModel, View};

    fn set_asset(id: &str, views: &[&str]) -> LoadedAsset {
        let mut set = ViewSet {
            board: None,
            cameras: vec![CameraModel::pinhole(4, 4, 2.0, 2.0, 2.0, 2.0)],
            ..ViewSet::default()
        };
        for v in views {
            set.views.push(View::posed(
                *v,
                0,
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ));
        }
        LoadedAsset::from_parts(
            id.to_string(),
            encode_viewset(&set),
            Some(AssetTypeHint::ViewSet),
            vec![],
        )
    }

    #[test]
    fn view_sets_are_found_by_id_or_uniqueness() {
        let assets = vec![
            LoadedAsset::from_parts(
                "blob".to_string(),
                vec![1],
                Some(AssetTypeHint::Binary),
                vec![],
            ),
            set_asset("a", &["v1"]),
        ];
        assert_eq!(find_viewset(&assets, None).unwrap().views[0].id, "v1");
        assert!(
            find_viewset(&assets, Some("blob"))
                .unwrap_err()
                .to_string()
                .contains("Available: a")
        );
        let two = vec![set_asset("a", &["v1"]), set_asset("b", &["v2"])];
        assert!(
            find_viewset(&two, None)
                .unwrap_err()
                .to_string()
                .contains("--views")
        );
        assert_eq!(find_viewset(&two, Some("b")).unwrap().views[0].id, "v2");
        assert!(find_viewset(&[], None).is_err());
    }

    #[test]
    fn list_summary_reads_the_set() {
        let asset = set_asset("a", &["v1", "v2"]);
        let set = decode_viewset(asset.data()).unwrap();
        let summary = summarize(&set);
        assert_eq!(summary.views.len(), 2);
        assert_eq!(summary.cameras[0].distortion, "none");
        assert_eq!(summary.views[1].forward, [0.0, 0.0, 1.0]);
        assert!(!summary.views[0].depth);
    }
}
