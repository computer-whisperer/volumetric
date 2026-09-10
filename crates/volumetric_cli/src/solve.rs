//! `view-solve`: pose a still from the marker cards it shows, against the
//! marker map a view set carries, and append it to the set.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use cv_core::detect::{DetectParams, Detection, detect};
use cv_core::dict::Dictionary;
use cv_core::exif::{Exif, focal_px_from_fov, read_exif};
use cv_core::gray::Gray;
use cv_core::pnp::{PoseSolve, SolveOptions, solve_view};
use serde::Serialize;
use view_core::image::{Rgb, decode_rgb};
use volumetric::{AssetTypeHint, Project};
use volumetric_abi::viewset::{
    CameraModel, Distortion, View, ViewSet, decode_viewset, encode_viewset,
};

use crate::views::{find_viewset_asset, project_assets};

#[derive(Parser, Debug)]
pub struct ViewSolveArgs {
    /// Project whose view set carries the marker map (the view is appended
    /// to it unless --output or --dry-run is given)
    #[arg(short = 'p', long)]
    pub project: Option<PathBuf>,

    /// A standalone .vviews file instead of a project
    #[arg(short = 'i', long)]
    pub input: Option<PathBuf>,

    /// The view set asset in the project (default: the only one)
    #[arg(long)]
    pub views: Option<String>,

    /// The still to pose (JPEG or PNG)
    #[arg(long)]
    pub image: PathBuf,

    /// Id of the new view (default: the image's file stem)
    #[arg(long)]
    pub id: Option<String>,

    /// Marker dictionary: 5x5_100 (swatches) or 4x4_50 (calibration board)
    #[arg(long, default_value = "5x5_100")]
    pub dictionary: String,

    /// Known intrinsics fx,fy,cx,cy[,k1[,k2]] in pixels; without them the
    /// focal is seeded from EXIF or --fov-deg and solved
    #[arg(long, allow_hyphen_values = true)]
    pub intrinsics: Option<String>,

    /// Horizontal field of view to seed the focal when the picture carries
    /// no 35 mm equivalent
    #[arg(long, default_value_t = 70.0)]
    pub fov_deg: f64,

    /// Solve the focal even when --intrinsics is given
    #[arg(long)]
    pub solve_focal: bool,

    /// Solve the first radial distortion term even when --intrinsics is
    /// given (without them it is solved with the focal)
    #[arg(long)]
    pub solve_distortion: bool,

    /// Write the picture with the detections drawn (map markers in amber,
    /// others in cyan, a dot at each first corner) to this PNG
    #[arg(long)]
    pub annotate: Option<PathBuf>,

    /// Extra tags for the new view
    #[arg(long = "tag")]
    pub tags: Vec<String>,

    /// Write the updated set here instead of into the project
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Report only; append nothing
    #[arg(long)]
    pub dry_run: bool,

    #[arg(long)]
    pub json: bool,
}

#[derive(Serialize)]
struct ExifReport {
    make: Option<String>,
    model: Option<String>,
    focal_mm: Option<f64>,
    focal_35mm: Option<f64>,
}

#[derive(Serialize)]
struct SeedReport {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    source: String,
}

#[derive(Serialize)]
struct DetectionReport {
    id: u32,
    in_map: bool,
    centre: [f64; 2],
    corners: [[f64; 2]; 4],
    rotation: u32,
    distance: u32,
    fit_px: f64,
}

#[derive(Serialize)]
struct MarkerReport {
    id: u32,
    rms_px: f64,
    corners_used: usize,
}

#[derive(Serialize)]
struct EstimateReport {
    value: f64,
    std: f64,
}

#[derive(Serialize)]
struct PoseReport {
    camera_to_world: [f64; 12],
    position: [f64; 3],
    rms_px: f64,
    corners_used: usize,
    markers: Vec<MarkerReport>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    focal: Option<EstimateReport>,
    k1: Option<EstimateReport>,
    warnings: Vec<String>,
}

#[derive(Serialize)]
struct MapMarker {
    id: u32,
    size_m: f64,
    corners: [[f64; 3]; 4],
}

#[derive(Serialize)]
struct SolveReport {
    image: String,
    width: u32,
    height: u32,
    dictionary: String,
    exif: Option<ExifReport>,
    seed: SeedReport,
    /// The set's marker map the pose is solved against.
    map: Vec<MapMarker>,
    detections: Vec<DetectionReport>,
    pose: Option<PoseReport>,
    error: Option<String>,
    view_id: String,
    saved: Option<String>,
}

/// The set to solve against, and the project asset it came from.
fn load_set(args: &ViewSolveArgs) -> Result<(ViewSet, Option<String>)> {
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
        _ => bail!("give exactly one of --project and --input"),
    }
}

/// The camera to start from: explicit intrinsics, else the EXIF 35 mm
/// equivalent, else the field of view; the principal point at the
/// centre.
fn seed_camera(
    args: &ViewSolveArgs,
    exif: Option<&Exif>,
    width: u32,
    height: u32,
) -> Result<(CameraModel, String)> {
    let (cx, cy) = (f64::from(width) * 0.5, f64::from(height) * 0.5);
    if let Some(text) = &args.intrinsics {
        let values: Vec<f64> = text
            .split(',')
            .map(|v| v.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .context("Invalid --intrinsics: expected fx,fy,cx,cy[,k1[,k2]]")?;
        if values.len() < 4 || values.len() > 6 {
            bail!("Invalid --intrinsics: expected fx,fy,cx,cy[,k1[,k2]]");
        }
        let mut camera =
            CameraModel::pinhole(width, height, values[0], values[1], values[2], values[3]);
        if values.len() > 4 {
            camera.distortion = Distortion::Radial {
                k: values[4..].to_vec(),
                p: [0.0, 0.0],
            };
        }
        return Ok((camera, "--intrinsics".to_string()));
    }
    if let Some(f) = exif.and_then(|e| e.focal_px(width, height)) {
        return Ok((
            CameraModel::pinhole(width, height, f, f, cx, cy),
            "EXIF 35 mm equivalent".to_string(),
        ));
    }
    let f = focal_px_from_fov(args.fov_deg, width);
    Ok((
        CameraModel::pinhole(width, height, f, f, cx, cy),
        format!("{}° field of view", args.fov_deg),
    ))
}

/// Draws the detections on the photograph.
fn annotate(photo: &Rgb, detections: &[Detection], set: &ViewSet) -> Rgb {
    let mut out = photo.clone();
    let thickness = ((photo.width.max(photo.height) as f64) / 1000.0)
        .ceil()
        .max(1.0) as i64;
    for d in detections {
        let colour = if set.markers.iter().any(|m| m.id == d.id) {
            [255, 184, 51]
        } else {
            [89, 217, 242]
        };
        for i in 0..4 {
            line(
                &mut out,
                d.corners[i],
                d.corners[(i + 1) % 4],
                colour,
                thickness,
            );
        }
        disc(&mut out, d.corners[0], thickness * 3, colour);
    }
    out
}

fn line(image: &mut Rgb, a: [f64; 2], b: [f64; 2], colour: [u8; 3], thickness: i64) {
    let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        let p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        disc(image, p, thickness, colour);
    }
}

fn disc(image: &mut Rgb, centre: [f64; 2], radius: i64, colour: [u8; 3]) {
    let (cx, cy) = (centre[0].floor() as i64, centre[1].floor() as i64);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let (x, y) = (cx + dx, cy + dy);
            if x >= 0 && y >= 0 && x < i64::from(image.width) && y < i64::from(image.height) {
                image.set(x as u32, y as u32, colour);
            }
        }
    }
}

/// Warnings a reader should weigh the pose by. The residual threshold
/// scales with the picture: a marker map triangulated to a few
/// millimetres leaves a few pixels at 12 MP.
fn warnings(solve: &PoseSolve, width: u32, height: u32) -> Vec<String> {
    let mut out = Vec::new();
    let residual_limit = (0.0015 * f64::from(width.max(height))).max(3.0);
    let used = solve.markers.iter().filter(|m| m.corners_used > 0).count();
    if used < 2 {
        out.push("pose from a single marker: the planar ambiguity is unresolved".to_string());
    }
    if let Some(f) = &solve.focal
        && f.std > 0.05 * f.value
    {
        out.push(format!(
            "focal weakly constrained ({:.0} ± {:.0} px): the cards lie in one plane seen square-on",
            f.value, f.std
        ));
    }
    if solve.rms_px > residual_limit {
        out.push(format!(
            "large residual ({:.1} px rms over {:.1} px): distortion, a moved card, or a map from another setup",
            solve.rms_px, residual_limit
        ));
    }
    out
}

/// The index of a camera equal to `camera` in the set, or the index it
/// would get when appended.
fn camera_index(set: &mut ViewSet, camera: &CameraModel) -> u32 {
    let same = |a: &CameraModel, b: &CameraModel| {
        a.width == b.width
            && a.height == b.height
            && (a.fx - b.fx).abs() < 1e-9
            && (a.fy - b.fy).abs() < 1e-9
            && (a.cx - b.cx).abs() < 1e-9
            && (a.cy - b.cy).abs() < 1e-9
            && a.distortion == b.distortion
    };
    if let Some(i) = set.cameras.iter().position(|c| same(c, camera)) {
        return i as u32;
    }
    set.cameras.push(camera.clone());
    (set.cameras.len() - 1) as u32
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("still")
        .to_string()
}

pub fn run_view_solve(args: ViewSolveArgs) -> Result<()> {
    let (mut set, asset_id) = load_set(&args)?;
    let dict = Dictionary::by_name(&args.dictionary).with_context(|| {
        format!(
            "unknown dictionary '{}'; expected 5x5_100 or 4x4_50",
            args.dictionary
        )
    })?;
    let bytes = std::fs::read(&args.image)
        .with_context(|| format!("Failed to read {}", args.image.display()))?;
    let photo = decode_rgb(&bytes)?;
    let gray = Gray::from_rgb8(photo.width, photo.height, &photo.pixels);
    let exif = read_exif(&bytes);
    let (seed, source) = seed_camera(&args, exif.as_ref(), photo.width, photo.height)?;
    let view_id = args.id.clone().unwrap_or_else(|| stem(&args.image));
    if set.views.iter().any(|v| v.id == view_id) {
        bail!("the set already has a view '{view_id}'; choose another --id");
    }

    let detections = detect(&gray, &dict, &DetectParams::default());
    let in_map = |id: u32| set.markers.iter().any(|m| m.id == id);
    // Unknown intrinsics: the focal and the first radial term come from
    // the cards (a phone's barrel distortion is several pixels at the
    // edges of a 12 MP still).
    let options = SolveOptions {
        solve_focal: args.solve_focal || args.intrinsics.is_none(),
        solve_distortion: args.solve_distortion || args.intrinsics.is_none(),
        ..SolveOptions::default()
    };
    let solved = solve_view(&seed, &set.markers, &detections, &options);

    if let Some(path) = &args.annotate {
        let drawn = annotate(&photo, &detections, &set);
        std::fs::write(path, drawn.to_png()?)
            .with_context(|| format!("Failed to write {}", path.display()))?;
    }

    let mut report = SolveReport {
        image: args.image.display().to_string(),
        width: photo.width,
        height: photo.height,
        dictionary: dict.name.to_string(),
        exif: exif.as_ref().map(|e| ExifReport {
            make: e.make.clone(),
            model: e.model.clone(),
            focal_mm: e.focal_mm,
            focal_35mm: e.focal_35mm,
        }),
        seed: SeedReport {
            fx: seed.fx,
            fy: seed.fy,
            cx: seed.cx,
            cy: seed.cy,
            source,
        },
        map: set
            .markers
            .iter()
            .map(|m| MapMarker {
                id: m.id,
                size_m: m.size_m,
                corners: m.corners,
            })
            .collect(),
        detections: detections
            .iter()
            .map(|d| DetectionReport {
                id: d.id,
                in_map: in_map(d.id),
                centre: d.centre(),
                corners: d.corners,
                rotation: d.rotation,
                distance: d.distance,
                fit_px: d.fit_px,
            })
            .collect(),
        pose: None,
        error: None,
        view_id: view_id.clone(),
        saved: None,
    };

    match &solved {
        Ok(solve) => {
            report.pose = Some(PoseReport {
                camera_to_world: solve.camera_to_world,
                position: solve.position(),
                rms_px: solve.rms_px,
                corners_used: solve.corners_used,
                markers: solve
                    .markers
                    .iter()
                    .map(|m| MarkerReport {
                        id: m.id,
                        rms_px: m.rms_px,
                        corners_used: m.corners_used,
                    })
                    .collect(),
                fx: solve.camera.fx,
                fy: solve.camera.fy,
                cx: solve.camera.cx,
                cy: solve.camera.cy,
                focal: solve.focal.map(|e| EstimateReport {
                    value: e.value,
                    std: e.std,
                }),
                k1: solve.k1.map(|e| EstimateReport {
                    value: e.value,
                    std: e.std,
                }),
                warnings: warnings(solve, photo.width, photo.height),
            });
        }
        Err(err) => report.error = Some(err.clone()),
    }

    // Append the view unless told not to.
    if let Ok(solve) = &solved
        && !args.dry_run
        && (args.output.is_some() || asset_id.is_some())
    {
        let camera = camera_index(&mut set, &solve.camera);
        let mut view = View::posed(view_id.clone(), camera, solve.camera_to_world);
        view.image = Some(bytes.clone());
        view.tags = vec!["still".to_string(), "solved:markers".to_string()];
        view.tags.extend(args.tags.iter().cloned());
        set.views.push(view);
        set.validate().map_err(anyhow::Error::msg)?;
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
        print_report(&report);
    }
    if let Some(err) = report.error {
        bail!("{err}");
    }
    Ok(())
}

fn print_report(r: &SolveReport) {
    let exif = match &r.exif {
        Some(e) => {
            let mut parts = Vec::new();
            if let Some(make) = &e.make {
                parts.push(make.clone());
            }
            if let Some(model) = &e.model {
                parts.push(model.clone());
            }
            if let Some(f) = e.focal_mm {
                parts.push(format!("{f:.2} mm"));
            }
            match e.focal_35mm {
                Some(f) => parts.push(format!("{f:.0} mm equivalent")),
                None => parts.push("no 35 mm equivalent".to_string()),
            }
            format!("EXIF {}", parts.join(", "))
        }
        None => "no EXIF".to_string(),
    };
    println!("{}: {}x{}, {exif}", r.image, r.width, r.height);
    println!(
        "camera seed: f {:.1} px from {}; principal point {:.1}, {:.1}",
        r.seed.fx, r.seed.source, r.seed.cx, r.seed.cy
    );
    let mapped = r.detections.iter().filter(|d| d.in_map).count();
    println!(
        "detections ({}): {} markers, {mapped} in the map",
        r.dictionary,
        r.detections.len()
    );
    for d in &r.detections {
        println!(
            "  id {:<4} centre ({:.0}, {:.0})  fit {:.2} px{}{}",
            d.id,
            d.centre[0],
            d.centre[1],
            d.fit_px,
            if d.distance > 0 {
                format!("  {} bit corrected", d.distance)
            } else {
                String::new()
            },
            if d.in_map { "  in map" } else { "" }
        );
    }
    if let Some(p) = &r.pose {
        println!(
            "pose: rms {:.2} px over {} corners of {} markers; position ({:.3}, {:.3}, {:.3})",
            p.rms_px,
            p.corners_used,
            p.markers.iter().filter(|m| m.corners_used > 0).count(),
            p.position[0],
            p.position[1],
            p.position[2]
        );
        for m in &p.markers {
            println!(
                "  marker {:<4} rms {:.2} px ({} corners)",
                m.id, m.rms_px, m.corners_used
            );
        }
        match &p.focal {
            Some(f) => println!(
                "focal: {:.1} ± {:.1} px (seed {:.1})",
                f.value, f.std, r.seed.fx
            ),
            None => println!("focal: {:.1} px (fixed)", p.fx),
        }
        if let Some(k1) = &p.k1 {
            println!("k1: {:.4} ± {:.4}", k1.value, k1.std);
        }
        for w in &p.warnings {
            println!("warning: {w}");
        }
    }
    if let Some(err) = &r.error {
        println!("no pose: {err}");
    }
    match &r.saved {
        Some(saved) => println!("appended view '{}' to {saved}", r.view_id),
        None => println!(
            "view '{}' not appended (dry run or no destination)",
            r.view_id
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_follow_intrinsics_exif_then_fov() {
        let base = ViewSolveArgs {
            project: None,
            input: None,
            views: None,
            image: PathBuf::from("still.jpg"),
            id: None,
            dictionary: "5x5_100".to_string(),
            intrinsics: None,
            fov_deg: 90.0,
            solve_focal: false,
            solve_distortion: false,
            annotate: None,
            tags: Vec::new(),
            output: None,
            dry_run: true,
            json: false,
        };
        let (cam, source) = seed_camera(&base, None, 1000, 500).unwrap();
        assert!((cam.fx - 500.0).abs() < 1e-9 && (cam.cx, cam.cy) == (500.0, 250.0));
        assert!(source.contains("90"));

        let exif = Exif {
            focal_35mm: Some(36.0),
            ..Exif::default()
        };
        let (cam, source) = seed_camera(&base, Some(&exif), 1000, 500).unwrap();
        assert_eq!(cam.fx, 1000.0);
        assert!(source.contains("EXIF"));

        let explicit = ViewSolveArgs {
            intrinsics: Some("800,810,500,250,0.01".to_string()),
            ..base
        };
        let (cam, _) = seed_camera(&explicit, Some(&exif), 1000, 500).unwrap();
        assert_eq!((cam.fx, cam.fy), (800.0, 810.0));
        assert_eq!(
            cam.distortion,
            Distortion::Radial {
                k: vec![0.01],
                p: [0.0, 0.0]
            }
        );
        let bad = ViewSolveArgs {
            intrinsics: Some("1,2".to_string()),
            ..explicit
        };
        assert!(seed_camera(&bad, None, 10, 10).is_err());
        assert_eq!(stem(Path::new("stills/00007.jpg")), "00007");
    }

    /// A synthetic board rendered to PNG solves against a set carrying its
    /// map, and the view lands in the set with the solved camera.
    #[test]
    fn a_rendered_board_solves_and_appends_a_view() {
        use cv_core::board::{Render, render, square_marker};
        let dir = std::env::temp_dir().join(format!("view_solve_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let truth = CameraModel::pinhole(1280, 960, 1000.0, 1000.0, 640.0, 480.0);
        let pitch: f64 = 50f64.to_radians();
        let (sn, cs) = pitch.sin_cos();
        let (forward, down) = ([0.0, cs, -sn], [0.0, -sn, -cs]);
        let view = View::posed(
            "truth",
            0,
            [
                1.0, down[0], forward[0], 0.1, //
                0.0, down[1], forward[1], -0.4, //
                0.0, down[2], forward[2], 1.2, //
            ],
        );
        let right = [1.0, 0.0, 0.0];
        let floor_down = [0.0, -1.0, 0.0];
        let markers = vec![
            square_marker(0, [-0.45, 0.85, 0.0], 0.12, right, floor_down),
            square_marker(1, [0.3, 0.9, 0.0], 0.12, right, floor_down),
            square_marker(2, [-0.35, 0.45, 0.0], 0.12, right, floor_down),
            square_marker(5, [0.25, 0.4, 0.0], 0.12, right, floor_down),
            square_marker(49, [-0.05, 0.65, 0.0], 0.16, right, floor_down),
        ];
        let dict = Dictionary::aruco_5x5_100();
        let picture = render(&truth, &view, &markers, &dict, &Render::default());
        let mut rgb = Rgb::new(picture.width, picture.height);
        for y in 0..picture.height {
            for x in 0..picture.width {
                let l = picture.get(x, y);
                rgb.set(x, y, [l, l, l]);
            }
        }
        let image = dir.join("board.png");
        std::fs::write(&image, rgb.to_png().unwrap()).unwrap();
        let set = ViewSet {
            schema: 1,
            world: Default::default(),
            provenance: Default::default(),
            cameras: vec![truth.clone()],
            views: Vec::new(),
            markers,
        };
        let input = dir.join("map.vviews");
        std::fs::write(&input, encode_viewset(&set)).unwrap();
        let output = dir.join("solved.vviews");

        // Seeded 15% off, focal and k1 solved.
        let args = ViewSolveArgs {
            project: None,
            input: Some(input),
            views: None,
            image,
            id: Some("board".to_string()),
            dictionary: "5x5_100".to_string(),
            intrinsics: None,
            fov_deg: 2.0 * (640.0f64 / 1150.0).atan().to_degrees(),
            solve_focal: false,
            solve_distortion: false,
            annotate: Some(dir.join("annotated.png")),
            tags: vec!["test".to_string()],
            output: Some(output.clone()),
            dry_run: false,
            json: true,
        };
        run_view_solve(args).unwrap();
        let solved = decode_viewset(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(solved.views.len(), 1);
        let (added, camera) = solved.view("board").unwrap();
        assert!(added.image.is_some());
        assert_eq!(added.tags, vec!["still", "solved:markers", "test"]);
        let (angle, dist) =
            cv_core::pnp::pose_difference(&added.camera_to_world, &view.camera_to_world);
        assert!(angle < 0.01 && dist < 0.02, "{angle} rad, {dist} m");
        assert!((camera.fx - 1000.0).abs() < 15.0, "{}", camera.fx);
        assert!(dir.join("annotated.png").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cameras_are_shared_when_equal() {
        let mut set = ViewSet {
            schema: 1,
            world: Default::default(),
            provenance: Default::default(),
            cameras: vec![CameraModel::pinhole(10, 10, 5.0, 5.0, 5.0, 5.0)],
            views: Vec::new(),
            markers: Vec::new(),
        };
        assert_eq!(
            camera_index(&mut set, &CameraModel::pinhole(10, 10, 5.0, 5.0, 5.0, 5.0)),
            0
        );
        assert_eq!(
            camera_index(&mut set, &CameraModel::pinhole(10, 10, 6.0, 6.0, 5.0, 5.0)),
            1
        );
        assert_eq!(set.cameras.len(), 2);
    }
}
