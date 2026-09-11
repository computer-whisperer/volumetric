//! `view-detect`: find the swatches, the survey card's tags and its
//! interior corners in every view of a set (or in loose pictures), and
//! store what was found on each view for the survey to solve from. The
//! pipeline is `cv_core::observe`, shared with the `view_detect`
//! operator; this command adds the file handling, the report and the
//! annotated pictures.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use cv_core::detect::DetectParams;
use cv_core::dict::Dictionary;
use cv_core::gray::Gray;
use cv_core::observe::{ObserveOptions, Observed, observe};
use serde::Serialize;
use view_core::image::{Rgb, decode_rgb};
use view_core::stills::full_picture;
use volumetric::{AssetTypeHint, Project};
use volumetric_abi::viewset::{Board, BoardSpec, ViewSet, decode_viewset, encode_viewset};

use crate::views::{find_viewset_asset, project_assets};

#[derive(Parser, Debug)]
pub struct ViewDetectArgs {
    /// Project whose view set to detect in (the observations are stored
    /// back into it unless --output or --dry-run is given)
    #[arg(short = 'p', long)]
    pub project: Option<PathBuf>,

    /// A standalone .vviews file instead of a project
    #[arg(short = 'i', long)]
    pub input: Option<PathBuf>,

    /// The view set asset in the project (default: the only one)
    #[arg(long)]
    pub views: Option<String>,

    /// Loose pictures to detect in, reported and not stored anywhere
    #[arg(long = "image")]
    pub images: Vec<PathBuf>,

    /// Only these views of the set (default: every view with a picture)
    #[arg(long = "view")]
    pub view_ids: Vec<String>,

    /// The survey card's spec as JSON (this crate's fields, or the
    /// scanner's card.json); default: the survey card
    #[arg(long)]
    pub card: Option<PathBuf>,

    /// Look for no board, only the swatches
    #[arg(long)]
    pub no_card: bool,

    /// Swatch family: 5x5_100, 4x4_50, or none
    #[arg(long, default_value = "5x5_100")]
    pub dictionary: String,

    /// Search for quads on the picture reduced to about this many pixels
    /// on its longer side (0 = full resolution)
    #[arg(long, default_value_t = 1600)]
    pub search_px: u32,

    /// Write the pictures with the detections drawn: swatches in amber,
    /// tags in cyan, card corners as magenta dots. A directory for a set,
    /// a PNG path for a single --image
    #[arg(long)]
    pub annotate: Option<PathBuf>,

    /// Write the updated set here instead of into the project
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Report only; store nothing
    #[arg(long)]
    pub dry_run: bool,

    #[arg(long)]
    pub json: bool,
}

#[derive(Serialize)]
struct MarkerReport {
    id: u32,
    family: String,
    centre: [f64; 2],
    corners: [[f64; 2]; 4],
    rotation: u32,
    distance: u32,
    fit_px: f64,
}

#[derive(Serialize)]
struct CornerReport {
    id: u32,
    pixel: [f64; 2],
    predicted: [f64; 2],
    shift_px: f64,
    tags: u32,
    contrast: f64,
}

#[derive(Serialize)]
struct BlurReport {
    horizontal: Option<f64>,
    vertical: Option<f64>,
    worst: Option<f64>,
    profiles: usize,
}

#[derive(Serialize)]
pub(crate) struct PictureReport {
    /// The view id, or the picture's path.
    picture: String,
    width: u32,
    height: u32,
    markers: Vec<MarkerReport>,
    /// The card's interior corners, when a board was looked for.
    corners: Option<Vec<CornerReport>>,
    blur: BlurReport,
    seconds: f64,
}

#[derive(Serialize)]
struct DetectReport {
    families: Vec<String>,
    board: Option<BoardSpec>,
    pictures: Vec<PictureReport>,
    skipped: Vec<String>,
    saved: Option<String>,
}

/// A card spec from JSON in either this crate's shape or the scanner's
/// `card.json` (`square_m`, `square_x_m`/`square_y_m`, `dictionary`).
pub fn read_card_spec(path: &Path) -> Result<BoardSpec> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))?;
    let spec = if value.get("pitch_x_m").is_some() {
        serde_json::from_value::<BoardSpec>(value)
            .with_context(|| format!("{} is not a board spec", path.display()))?
    } else {
        let num = |key: &str| -> Option<f64> { value.get(key).and_then(|v| v.as_f64()) };
        let int = |key: &str| -> Option<u32> {
            value.get(key).and_then(|v| v.as_u64()).map(|v| v as u32)
        };
        let square = num("square_m");
        let family = value
            .get("dictionary")
            .and_then(|v| v.as_str())
            .context("card.json has no dictionary")?;
        let dict = Dictionary::by_name(family)
            .with_context(|| format!("unknown marker family '{family}' in {}", path.display()))?;
        BoardSpec {
            squares_x: int("squares_x").context("card.json has no squares_x")?,
            squares_y: int("squares_y").context("card.json has no squares_y")?,
            pitch_x_m: num("square_x_m")
                .or(square)
                .context("card.json has no square_m")?,
            pitch_y_m: num("square_y_m")
                .or(square)
                .context("card.json has no square_m")?,
            marker_m: num("marker_m").context("card.json has no marker_m")?,
            family: dict.name.to_string(),
            first_id: int("first_id").unwrap_or(0),
        }
    };
    spec.validate().map_err(anyhow::Error::msg)?;
    Ok(spec)
}

fn options_of(args: &ViewDetectArgs) -> Result<ObserveOptions> {
    let swatches = match args.dictionary.to_ascii_lowercase().as_str() {
        "none" | "" => None,
        name => Some(Dictionary::by_name(name).with_context(|| {
            format!("unknown dictionary '{name}'; expected 5x5_100, 4x4_50, 36h11 or none")
        })?),
    };
    let board = if args.no_card {
        None
    } else {
        Some(match &args.card {
            Some(path) => read_card_spec(path)?,
            None => BoardSpec::survey_card(),
        })
    };
    if swatches.is_none() && board.is_none() {
        bail!("nothing to look for: give a --dictionary or a card");
    }
    Ok(ObserveOptions {
        swatches,
        board,
        detect: DetectParams {
            search_px: args.search_px,
            ..DetectParams::default()
        },
        ..ObserveOptions::default()
    })
}

/// Detects in one picture and reports it.
fn observe_picture(name: &str, photo: &Rgb, options: &ObserveOptions) -> (Observed, PictureReport) {
    let start = Instant::now();
    let gray = Gray::from_rgb8(photo.width, photo.height, &photo.pixels);
    let seen = observe(&gray, options);
    let report = PictureReport {
        picture: name.to_string(),
        width: photo.width,
        height: photo.height,
        markers: seen
            .detections
            .iter()
            .map(|d| MarkerReport {
                id: d.id,
                family: d.family.to_string(),
                centre: d.centre(),
                corners: d.corners,
                rotation: d.rotation,
                distance: d.distance,
                fit_px: d.fit_px,
            })
            .collect(),
        corners: options.board.as_ref().map(|_| {
            seen.corners
                .iter()
                .map(|c| CornerReport {
                    id: c.id,
                    pixel: c.pixel,
                    predicted: c.predicted,
                    shift_px: c.shift_px,
                    tags: c.tags,
                    contrast: c.contrast,
                })
                .collect()
        }),
        blur: BlurReport {
            horizontal: seen.blur.horizontal,
            vertical: seen.blur.vertical,
            worst: seen.blur.worst(),
            profiles: seen.blur.profiles,
        },
        seconds: start.elapsed().as_secs_f64(),
    };
    (seen, report)
}

/// Draws the detections on the photograph.
fn annotate(photo: &Rgb, seen: &Observed, options: &ObserveOptions) -> Rgb {
    let mut out = photo.clone();
    let thickness = ((photo.width.max(photo.height) as f64) / 1000.0)
        .ceil()
        .max(1.0) as i64;
    let board_family = options.board.as_ref().map(|b| b.family.as_str());
    for d in &seen.detections {
        let colour = if Some(d.family) == board_family {
            [89, 217, 242]
        } else {
            [255, 184, 51]
        };
        for i in 0..4 {
            crate::solve::line(
                &mut out,
                d.corners[i],
                d.corners[(i + 1) % 4],
                colour,
                thickness,
            );
        }
        crate::solve::disc(&mut out, d.corners[0], thickness * 3, colour);
    }
    for c in &seen.corners {
        crate::solve::disc(&mut out, c.pixel, thickness * 2, [255, 64, 255]);
    }
    out
}

fn load_set(args: &ViewDetectArgs) -> Result<(ViewSet, Option<String>)> {
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
        (None, None) => bail!("give --project, --input or --image"),
        _ => bail!("give exactly one of --project and --input"),
    }
}

pub fn run_view_detect(args: ViewDetectArgs) -> Result<()> {
    let options = options_of(&args)?;
    let mut report = DetectReport {
        families: options
            .families()
            .iter()
            .map(|d| d.name.to_string())
            .collect(),
        board: options.board.clone(),
        pictures: Vec::new(),
        skipped: Vec::new(),
        saved: None,
    };

    if !args.images.is_empty() {
        if args.project.is_some() || args.input.is_some() {
            bail!("--image reports loose pictures; give it without a set");
        }
        if args.images.len() > 1
            && let Some(dir) = &args.annotate
            && !dir.is_dir()
        {
            bail!("--annotate must be a directory for several pictures");
        }
        for path in &args.images {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let photo = decode_rgb(&bytes)?;
            let (seen, picture) = observe_picture(&path.display().to_string(), &photo, &options);
            if !args.json {
                print_picture(&picture);
            }
            if let Some(target) = &args.annotate {
                let out = if target.is_dir() {
                    target.join(format!("{}_detect.png", stem(path)))
                } else {
                    target.clone()
                };
                std::fs::write(&out, annotate(&photo, &seen, &options).to_png()?)
                    .with_context(|| format!("Failed to write {}", out.display()))?;
            }
            report.pictures.push(picture);
        }
        if args.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        return Ok(());
    }

    let (mut set, asset_id) = load_set(&args)?;
    if let Some(dir) = &args.annotate
        && !dir.is_dir()
    {
        bail!("--annotate must be a directory for a set");
    }
    let (pictures, skipped) = detect_set(
        &mut set,
        &args.view_ids,
        &options,
        args.annotate.as_deref(),
        !args.json,
    )?;
    report.pictures = pictures;
    report.skipped = skipped;

    if !args.dry_run && (args.output.is_some() || asset_id.is_some()) {
        let encoded = encode_viewset(&set);
        if let Some(output) = &args.output {
            std::fs::write(output, &encoded)
                .with_context(|| format!("Failed to write {}", output.display()))?;
            report.saved = Some(output.display().to_string());
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
            report.saved = Some(format!("{} asset '{id}'", path.display()));
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_summary(&report);
    }
    Ok(())
}

/// Detects in the set's views (those named, or every one) and stores the
/// observations on them; the per-view reports and the views skipped for
/// lack of a picture. Annotated pictures go to `annotate` when given;
/// each report is printed as it comes when `print` is set.
pub(crate) fn detect_set(
    set: &mut ViewSet,
    view_ids: &[String],
    options: &ObserveOptions,
    annotate_dir: Option<&Path>,
    print: bool,
) -> Result<(Vec<PictureReport>, Vec<String>)> {
    let mut indices: Vec<usize> = (0..set.views.len()).collect();
    if !view_ids.is_empty() {
        for id in view_ids {
            if !set.views.iter().any(|v| &v.id == id) {
                bail!("no view '{id}' in the set");
            }
        }
        indices.retain(|&i| view_ids.contains(&set.views[i].id));
    }
    let mut pictures = Vec::new();
    let mut skipped = Vec::new();
    for i in indices {
        let id = set.views[i].id.clone();
        if set.views[i].image.is_none() && set.views[i].source.is_none() {
            skipped.push(id);
            continue;
        }
        let photo = match full_picture(set, &set.views[i]).and_then(|bytes| decode_rgb(&bytes)) {
            Ok(photo) => photo,
            Err(err) => {
                eprintln!("view '{id}': no picture to detect in: {err:#}");
                skipped.push(id);
                continue;
            }
        };
        let (seen, picture) = observe_picture(&id, &photo, options);
        if print {
            print_picture(&picture);
        }
        if let Some(dir) = annotate_dir {
            let out = dir.join(format!("{id}_detect.png"));
            std::fs::write(&out, annotate(&photo, &seen, options).to_png()?)
                .with_context(|| format!("Failed to write {}", out.display()))?;
        }
        set.views[i].observations = Some(seen.to_observations());
        pictures.push(picture);
    }
    if let Some(spec) = &options.board {
        // Solved corners survive only for the same card.
        let corners = match set.board.take() {
            Some(board) if &board.spec == spec => board.corners,
            _ => Vec::new(),
        };
        set.board = Some(Board {
            spec: spec.clone(),
            corners,
        });
    }
    Ok((pictures, skipped))
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("picture")
        .to_string()
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(values[values.len() / 2])
}

fn print_picture(p: &PictureReport) {
    let mut families: Vec<(&str, Vec<u32>)> = Vec::new();
    for m in &p.markers {
        match families.iter_mut().find(|(f, _)| *f == m.family) {
            Some((_, ids)) => ids.push(m.id),
            None => families.push((&m.family, vec![m.id])),
        }
    }
    let mut parts: Vec<String> = families
        .iter()
        .map(|(f, ids)| {
            let listed: Vec<String> = ids.iter().take(12).map(|i| i.to_string()).collect();
            let more = if ids.len() > 12 { ", …" } else { "" };
            format!("{} {f} ({}{more})", ids.len(), listed.join(", "))
        })
        .collect();
    if let Some(corners) = &p.corners {
        let shift = median(corners.iter().map(|c| c.shift_px).collect()).unwrap_or(0.0);
        parts.push(format!(
            "{} card corners (shift {shift:.2} px median)",
            corners.len()
        ));
    }
    let blur = match (p.blur.horizontal, p.blur.vertical) {
        (Some(h), Some(v)) => format!("blur {h:.2}/{v:.2} px"),
        (Some(b), None) | (None, Some(b)) => format!("blur {b:.2} px"),
        (None, None) => "no blur measure".to_string(),
    };
    println!(
        "{}: {}x{}, {}, {blur}, {:.1} s",
        p.picture,
        p.width,
        p.height,
        if parts.is_empty() {
            "nothing found".to_string()
        } else {
            parts.join(", ")
        },
        p.seconds
    );
}

fn print_summary(r: &DetectReport) {
    let n = r.pictures.len();
    let with_card = r
        .pictures
        .iter()
        .filter(|p| p.corners.as_ref().is_some_and(|c| !c.is_empty()))
        .count();
    let corners = median(
        r.pictures
            .iter()
            .filter_map(|p| p.corners.as_ref().map(|c| c.len() as f64))
            .collect(),
    );
    let markers = median(r.pictures.iter().map(|p| p.markers.len() as f64).collect());
    let mut line = format!("{n} views detected");
    if let Some(m) = markers {
        line.push_str(&format!(", {m:.0} markers median"));
    }
    if r.board.is_some() {
        line.push_str(&format!(
            ", {with_card} with the card ({} corners median)",
            corners.unwrap_or(0.0)
        ));
    }
    if !r.skipped.is_empty() {
        line.push_str(&format!(
            "; {} without a picture: {}",
            r.skipped.len(),
            r.skipped.join(", ")
        ));
    }
    println!("{line}");
    if let Some(saved) = &r.saved {
        println!("saved to {saved}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::board::{PlacedBoard, Render, render_board};
    use volumetric_abi::viewset::{CameraModel, View};

    #[test]
    fn a_set_gets_observations_and_a_board() {
        // The survey card straight down from 1 m, as a PNG on one view;
        // a second view has no picture.
        let camera = CameraModel::pinhole(1200, 900, 2000.0, 2000.0, 600.0, 450.0);
        let view = View::posed(
            "top",
            0,
            [
                1.0, 0.0, 0.0, 0.0, //
                0.0, -1.0, 0.0, 0.0, //
                0.0, 0.0, -1.0, 1.0, //
            ],
        );
        let board = PlacedBoard::new(
            BoardSpec::survey_card(),
            [-0.11, 0.1, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        );
        let gray = render_board(
            &camera,
            &view,
            &board,
            &Render {
                blur_sigma: 0.8,
                ..Render::default()
            },
        );
        let mut photo = Rgb::new(gray.width, gray.height);
        for y in 0..gray.height {
            for x in 0..gray.width {
                let v = gray.get(x, y);
                photo.set(x, y, [v, v, v]);
            }
        }
        let mut with_picture = view.clone();
        with_picture.image = Some(photo.to_png().unwrap());
        let mut set = ViewSet {
            cameras: vec![camera],
            views: vec![
                with_picture,
                View::posed("blind", 0, view.camera_to_world.unwrap()),
            ],
            ..ViewSet::default()
        };
        let options = ObserveOptions::default();
        let (pictures, skipped) = detect_set(&mut set, &[], &options, None, false).unwrap();
        assert_eq!(pictures.len(), 1);
        assert_eq!(skipped, vec!["blind".to_string()]);
        assert_eq!(pictures[0].corners.as_ref().unwrap().len(), 110);
        assert_eq!(pictures[0].markers.len(), 66);
        let obs = set.views[0].observations.as_ref().unwrap();
        assert_eq!(obs.board.len(), 110);
        assert_eq!(set.board.as_ref().unwrap().spec, BoardSpec::survey_card());
        assert!(set.views[1].observations.is_none());
        // Naming a view that is not there is an error; naming the blind
        // one detects nothing and skips it.
        assert!(detect_set(&mut set, &["nope".to_string()], &options, None, false).is_err());
        let (pictures, skipped) =
            detect_set(&mut set, &["blind".to_string()], &options, None, false).unwrap();
        assert!(pictures.is_empty());
        assert_eq!(skipped, vec!["blind".to_string()]);
    }

    #[test]
    fn card_specs_read_in_both_shapes() {
        let dir = std::env::temp_dir().join(format!("vdetect_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shim = dir.join("card.json");
        std::fs::write(
            &shim,
            r#"{"squares_x": 12, "squares_y": 11, "square_m": 0.01778, "marker_m": 0.0127,
                "dictionary": "DICT_APRILTAG_36h11", "square_x_m": 0.0179443, "square_y_m": 0.0177451, "first_id": 100}"#,
        )
        .unwrap();
        let spec = read_card_spec(&shim).unwrap();
        assert_eq!(spec, BoardSpec::survey_card());
        let ours = dir.join("spec.json");
        std::fs::write(&ours, serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(read_card_spec(&ours).unwrap(), spec);
        std::fs::write(&shim, r#"{"squares_x": 4}"#).unwrap();
        assert!(read_card_spec(&shim).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
