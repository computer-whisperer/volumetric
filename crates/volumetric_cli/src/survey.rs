//! `view-survey`: from the observations `view-detect` stored on a set's
//! views, solve one camera model per focus setting, a pose per view and
//! the card and swatch corners as points in the world; store the posed
//! set back and report the fit. The bundle is `cv_core::survey`, shared
//! with the `survey` operator.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use cv_core::survey::{SurveyOptions, SurveyReport, survey};
use volumetric::{AssetTypeHint, Project};
use volumetric_abi::viewset::{Board, ViewSet, decode_viewset, encode_viewset};

use crate::observe::read_card_spec;
use crate::views::{find_viewset_asset, project_assets};

#[derive(Parser, Debug)]
pub struct ViewSurveyArgs {
    /// Project whose view set to survey (the posed set is stored back
    /// into it unless --output or --dry-run is given)
    #[arg(short = 'p', long)]
    pub project: Option<PathBuf>,

    /// A standalone .vviews file instead of a project
    #[arg(short = 'i', long)]
    pub input: Option<PathBuf>,

    /// The view set asset in the project (default: the only one)
    #[arg(long)]
    pub views: Option<String>,

    /// The card's spec as JSON, when the set's board should be replaced
    /// (default: the board view-detect stored)
    #[arg(long)]
    pub card: Option<PathBuf>,

    /// The swatches' marker family
    #[arg(long, default_value = "5x5_100")]
    pub dictionary: String,

    /// Swatch ids at or above this are false decodes
    #[arg(long, default_value_t = 60)]
    pub max_swatch_id: u32,

    /// Card corners a view needs to be posed on the card at the start
    #[arg(long, default_value_t = 8)]
    pub min_card: usize,

    /// The soft-L1 scale in pixels
    #[arg(long, default_value_t = 1.5)]
    pub f_scale: f64,

    /// Drop views the bundle leaves above this rms (raise it for a
    /// defocused set)
    #[arg(long, default_value_t = 3.0)]
    pub reject_px: f64,

    /// Bundle rounds (drop, extend, solve again)
    #[arg(long, default_value_t = 4)]
    pub rounds: usize,

    /// Write the posed set here instead of into the project
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Write the report as JSON here
    #[arg(long)]
    pub report: Option<PathBuf>,

    /// Report only; store nothing
    #[arg(long)]
    pub dry_run: bool,

    /// Print the report as JSON
    #[arg(long)]
    pub json: bool,
}

fn load_set(args: &ViewSurveyArgs) -> Result<(ViewSet, Option<String>)> {
    match (&args.project, &args.input) {
        (Some(path), None) => {
            let assets = project_assets(path, false)?;
            let asset = find_viewset_asset(&assets, args.views.as_deref())?;
            let set = decode_viewset(asset.data()).map_err(anyhow::Error::msg)?;
            Ok((set, Some(asset.id().to_string())))
        }
        (None, Some(path)) => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let set = decode_viewset(&bytes)
                .map_err(|err| anyhow::anyhow!("{} is not a view set: {err}", path.display()))?;
            Ok((set, None))
        }
        (None, None) => bail!("give --project or --input"),
        _ => bail!("give exactly one of --project and --input"),
    }
}

pub fn run_view_survey(args: ViewSurveyArgs) -> Result<()> {
    let (mut set, asset_id) = load_set(&args)?;
    if let Some(path) = &args.card {
        let spec = read_card_spec(path)?;
        set.board = Some(Board {
            spec,
            corners: Vec::new(),
        });
    }
    let options = SurveyOptions {
        swatch_family: args.dictionary.clone(),
        max_swatch_id: args.max_swatch_id,
        min_card_corners: args.min_card,
        soft_l1_px: args.f_scale,
        reject_px: args.reject_px,
        rounds: args.rounds.max(1),
        ..SurveyOptions::default()
    };
    let started = Instant::now();
    let report = survey(&mut set, &options).map_err(anyhow::Error::msg)?;
    let seconds = started.elapsed().as_secs_f64();

    let mut saved = None;
    if !args.dry_run && (args.output.is_some() || asset_id.is_some()) {
        let encoded = encode_viewset(&set);
        if let Some(output) = &args.output {
            std::fs::write(output, &encoded)
                .with_context(|| format!("Failed to write {}", output.display()))?;
            saved = Some(output.display().to_string());
        } else if let (Some(path), Some(id)) = (&args.project, &asset_id) {
            let mut project = Project::load_from_file(path).context("Failed to load project")?;
            let import = project
                .imports_mut()
                .iter_mut()
                .find(|i| &i.id == id)
                .context("view set import vanished")?;
            import.data = encoded;
            import.type_hint = Some(AssetTypeHint::ViewSet);
            project
                .save_to_file(path)
                .with_context(|| format!("Failed to save {}", path.display()))?;
            saved = Some(format!("{} asset '{id}'", path.display()));
        }
    }
    if let Some(path) = &args.report {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("Failed to write {}", path.display()))?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report, &set, seconds, saved.as_deref());
    }
    Ok(())
}

fn print_report(r: &SurveyReport, set: &ViewSet, seconds: f64, saved: Option<&str>) {
    for line in &r.log {
        println!("{line}");
    }
    println!();
    for c in &r.cameras {
        println!(
            "camera {} '{}': {} frames, f {:.1} px (± {:.1}), pp ({:.1} ± {:.1}, {:.1} ± {:.1}), k1 {:+.5} k2 {:+.4}",
            c.index, c.label, c.frames, c.f, c.f_std, c.cx, c.cx_std, c.cy, c.cy_std, c.k1, c.k2
        );
    }
    let sigmas: Vec<f64> = set
        .board
        .as_ref()
        .map(|b| b.corners.iter().map(|c| c.sigma_m * 1e3).collect())
        .unwrap_or_default();
    println!(
        "card: {}/{} corners solved; across span {:.3} mm over columns (calipers {:.3}), along span {:.3} mm (calipers {:.3}), planarity {:.3} mm rms, corner sigma median {:.3} mm",
        r.card_corners_solved,
        set.board.as_ref().map_or(0, |b| b.spec.n_corners()),
        r.span_across_mm,
        r.span_across_nominal_mm,
        r.span_along_mm,
        r.span_along_nominal_mm,
        r.card_planarity_mm,
        median(sigmas)
    );
    for s in &r.swatches {
        let m = set.markers.iter().find(|m| m.id == s.id);
        let centre = m.map(|m| {
            let mut c = [0.0; 3];
            for corner in &m.corners {
                for (i, v) in corner.iter().enumerate() {
                    c[i] += v * 0.25e3;
                }
            }
            c
        });
        match centre {
            Some(c) => println!(
                "swatch {:>3}: centre ({:8.1}, {:8.1}, {:6.1}) mm, side {:.2} mm, sigma {:.3} mm",
                s.id, c[0], c[1], c[2], s.side_mm, s.sigma_mm
            ),
            None => println!(
                "swatch {:>3}: side {:.2} mm, sigma {:.3} mm",
                s.id, s.side_mm, s.sigma_mm
            ),
        }
    }
    let mut worst: Vec<&cv_core::survey::FrameReport> = r.frames.iter().collect();
    worst.sort_by(|a, b| b.rms_px.total_cmp(&a.rms_px));
    println!(
        "fit: {} observations, {} parameters, rms {:.3} px, median {:.3}, inliers {:.1} % at {:.3} px, {} iterations, {:.1} s",
        r.observations,
        r.parameters,
        r.rms_px,
        r.median_px,
        100.0 * r.inlier_fraction,
        r.inlier_rms_px,
        r.iterations,
        seconds
    );
    println!(
        "worst frames: {}",
        worst
            .iter()
            .take(5)
            .map(|f| format!("{} {:.2} px", f.id, f.rms_px))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !r.rejected.is_empty() {
        println!("rejected: {}", r.rejected.join(" "));
    }
    let unposed: Vec<&String> = r
        .unposed
        .iter()
        .filter(|id| !r.rejected.contains(id))
        .collect();
    if !unposed.is_empty() {
        println!(
            "not posed: {}",
            unposed
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    if !r.left_out.is_empty() {
        println!(
            "left out (camera with too few frames): {}",
            r.left_out.join(" ")
        );
    }
    match saved {
        Some(where_) => println!("Saved the posed set to {where_}"),
        None => println!("Nothing saved"),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}
