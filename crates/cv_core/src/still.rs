//! One still through the whole pipeline: seed the camera, find the cards,
//! solve the pose against a view set's map, and append the view. The
//! `view-solve` command and the `view_solve` operator both run this, so
//! a still poses the same way headlessly and in a project.

use volumetric_abi::viewset::{CameraModel, Distortion, View, ViewSet};

use crate::detect::{DetectParams, Detection, detect};
use crate::dict::Dictionary;
use crate::exif::{Exif, focal_px_from_fov, read_exif};
use crate::gray::Gray;
use crate::pnp::{PoseSolve, SolveOptions, solve_view};

/// How a still is solved.
#[derive(Clone, Debug)]
pub struct StillOptions {
    pub dictionary: Dictionary,
    /// Known intrinsics; without them the focal is seeded from EXIF or
    /// `fov_deg` and solved, and so is the first radial term.
    pub intrinsics: Option<CameraModel>,
    /// Horizontal field of view to seed the focal when the picture carries
    /// no 35 mm equivalent, degrees.
    pub fov_deg: f64,
    /// Solve the focal even with known intrinsics.
    pub solve_focal: bool,
    /// Solve the first radial term even with known intrinsics.
    pub solve_distortion: bool,
    pub detect: DetectParams,
}

impl Default for StillOptions {
    fn default() -> Self {
        Self {
            dictionary: Dictionary::aruco_5x5_100(),
            intrinsics: None,
            fov_deg: 70.0,
            solve_focal: false,
            solve_distortion: false,
            detect: DetectParams::default(),
        }
    }
}

/// Everything a still's solve found.
#[derive(Clone, Debug)]
pub struct StillSolve {
    pub exif: Option<Exif>,
    pub seed: CameraModel,
    /// Where the seed came from, for the report.
    pub seed_source: String,
    pub detections: Vec<Detection>,
    pub pose: Result<PoseSolve, String>,
    /// What a reader should weigh the pose by; empty without a pose.
    pub warnings: Vec<String>,
}

impl StillSolve {
    /// Detections of markers the map holds.
    pub fn in_map(&self, set: &ViewSet) -> usize {
        self.detections
            .iter()
            .filter(|d| set.markers.iter().any(|m| m.id == d.id))
            .count()
    }

    /// One line for a status or a warning channel.
    pub fn summary(&self, set: &ViewSet) -> String {
        let cards = format!(
            "{} cards, {} in the map",
            self.detections.len(),
            self.in_map(set)
        );
        match &self.pose {
            Ok(pose) => {
                let used = pose.markers.iter().filter(|m| m.corners_used > 0).count();
                let focal = match pose.focal {
                    Some(f) => format!(
                        "; focal {:.0} ± {:.0} px (seed {:.0})",
                        f.value, f.std, self.seed.fx
                    ),
                    None => String::new(),
                };
                format!(
                    "{cards}; rms {:.2} px over {} corners of {used}{focal}",
                    pose.rms_px, pose.corners_used
                )
            }
            Err(err) => format!("{cards}; no pose: {err}"),
        }
    }
}

/// The camera to start from: known intrinsics, else the EXIF 35 mm
/// equivalent, else the field of view; the principal point at the centre
/// unless given.
pub fn seed_camera(
    options: &StillOptions,
    exif: Option<&Exif>,
    width: u32,
    height: u32,
) -> (CameraModel, String) {
    if let Some(camera) = &options.intrinsics {
        let mut camera = camera.clone();
        camera.width = width;
        camera.height = height;
        return (camera, "given intrinsics".to_string());
    }
    let (cx, cy) = (f64::from(width) * 0.5, f64::from(height) * 0.5);
    if let Some(f) = exif.and_then(|e| e.focal_px(width, height)) {
        return (
            CameraModel::pinhole(width, height, f, f, cx, cy),
            "EXIF 35 mm equivalent".to_string(),
        );
    }
    let f = focal_px_from_fov(options.fov_deg, width);
    (
        CameraModel::pinhole(width, height, f, f, cx, cy),
        format!("{}° field of view", options.fov_deg),
    )
}

/// Warnings a reader should weigh the pose by. The residual threshold
/// scales with the picture: a marker map triangulated to a few
/// millimetres leaves a few pixels at 12 MP.
pub fn warnings(solve: &PoseSolve, width: u32, height: u32) -> Vec<String> {
    let mut out = Vec::new();
    let residual_limit = (0.0015 * f64::from(width.max(height))).max(3.0);
    let used = solve.markers.iter().filter(|m| m.corners_used > 0).count();
    if used < 2 {
        out.push("pose from a single card: the planar ambiguity is unresolved".to_string());
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

/// Solves a still: `gray` is the picture, `file` its bytes (for EXIF; may
/// be empty), `set` supplies the marker map.
pub fn solve_still(gray: &Gray, file: &[u8], set: &ViewSet, options: &StillOptions) -> StillSolve {
    let exif = if file.is_empty() {
        None
    } else {
        read_exif(file)
    };
    let (seed, seed_source) = seed_camera(options, exif.as_ref(), gray.width, gray.height);
    let detections = detect(gray, &[&options.dictionary], &options.detect);
    let unknown = options.intrinsics.is_none();
    let pose = solve_view(
        &seed,
        &set.markers,
        &detections,
        &SolveOptions {
            solve_focal: options.solve_focal || unknown,
            solve_distortion: options.solve_distortion || unknown,
            ..SolveOptions::default()
        },
    );
    let warnings = pose
        .as_ref()
        .map(|p| warnings(p, gray.width, gray.height))
        .unwrap_or_default();
    StillSolve {
        exif,
        seed,
        seed_source,
        detections,
        pose,
        warnings,
    }
}

/// The index of a camera equal to `camera` in the set, or the index it
/// gets when appended.
pub fn camera_index(set: &mut ViewSet, camera: &CameraModel) -> u32 {
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

/// `id` if the set has no view of that name, else `id_2`, `id_3`, …
pub fn unique_view_id(set: &ViewSet, id: &str) -> String {
    if !set.views.iter().any(|v| v.id == id) {
        return id.to_string();
    }
    (2..)
        .map(|n| format!("{id}_{n}"))
        .find(|candidate| !set.views.iter().any(|v| &v.id == candidate))
        .expect("an unused suffix exists")
}

/// Appends the solved still as a view tagged `still` and `solved:markers`
/// (plus the solve's numbers and `tags`), with its picture when given.
/// The id must be unused.
pub fn append_view(
    set: &mut ViewSet,
    id: &str,
    solve: &PoseSolve,
    image: Option<Vec<u8>>,
    tags: &[String],
) -> Result<(), String> {
    if set.views.iter().any(|v| v.id == id) {
        return Err(format!("the set already has a view '{id}'"));
    }
    let camera = camera_index(set, &solve.camera);
    let mut view = View::posed(id, camera, solve.camera_to_world);
    view.image = image;
    view.tags = solve_tags(solve);
    view.tags.extend(tags.iter().cloned());
    set.views.push(view);
    set.validate()
}

/// The tags a solved still carries: how it was posed and how well.
pub fn solve_tags(solve: &PoseSolve) -> Vec<String> {
    let used = solve.markers.iter().filter(|m| m.corners_used > 0).count();
    vec![
        "still".to_string(),
        "solved:markers".to_string(),
        format!("rms:{:.1}px", solve.rms_px),
        format!("cards:{used}"),
    ]
}

/// Known intrinsics from a focal and an optional first radial term, with
/// the principal point at the centre of a `width` x `height` picture.
pub fn intrinsics_from_focal(focal_px: f64, k1: f64, width: u32, height: u32) -> CameraModel {
    let mut camera = CameraModel::pinhole(
        width,
        height,
        focal_px,
        focal_px,
        f64::from(width) * 0.5,
        f64::from(height) * 0.5,
    );
    if k1 != 0.0 {
        camera.distortion = Distortion::Radial {
            k: vec![k1],
            p: [0.0, 0.0],
        };
    }
    camera
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Render, render, square_marker};
    use crate::pnp::pose_difference;

    fn scene() -> (CameraModel, View, ViewSet) {
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
        let set = ViewSet {
            board: None,
            schema: 2,
            world: Default::default(),
            provenance: Default::default(),
            cameras: vec![truth.clone()],
            views: Vec::new(),
            markers,
        };
        (truth, view, set)
    }

    #[test]
    fn a_still_solves_and_joins_the_set() {
        let (truth, view, mut set) = scene();
        let picture = render(
            &truth,
            &view,
            &set.markers,
            &Dictionary::aruco_5x5_100(),
            &Render::default(),
        );
        // Unknown intrinsics: seeded 15% off by a field of view, focal
        // and k1 solved.
        let options = StillOptions {
            fov_deg: 2.0 * (640.0f64 / 1150.0).atan().to_degrees(),
            ..StillOptions::default()
        };
        let solve = solve_still(&picture, &[], &set, &options);
        assert_eq!(solve.detections.len(), 5);
        assert_eq!(solve.in_map(&set), 5);
        assert!(solve.seed_source.contains("field of view"));
        let pose = solve.pose.as_ref().unwrap();
        let (angle, dist) = pose_difference(&pose.camera_to_world, view.pose().unwrap());
        assert!(angle < 0.01 && dist < 0.02, "{angle} rad, {dist} m");
        assert!((pose.camera.fx - 1000.0).abs() < 15.0, "{}", pose.camera.fx);
        assert!(solve.warnings.is_empty(), "{:?}", solve.warnings);
        assert!(
            solve
                .summary(&set)
                .starts_with("5 cards, 5 in the map; rms ")
        );

        append_view(
            &mut set,
            "still",
            pose,
            Some(vec![1, 2, 3]),
            &["mine".to_string()],
        )
        .unwrap();
        assert_eq!(set.views.len(), 1);
        assert_eq!(
            set.cameras.len(),
            2,
            "the solved camera differs from the truth's"
        );
        let (added, camera) = set.view("still").unwrap();
        assert_eq!(
            added.tags[..2],
            ["still".to_string(), "solved:markers".to_string()]
        );
        assert!(added.tags.iter().any(|t| t.starts_with("rms:")));
        assert_eq!(added.tags.last().map(String::as_str), Some("mine"));
        assert!((camera.fx - pose.camera.fx).abs() < 1e-9);
        assert_eq!(unique_view_id(&set, "still"), "still_2");
        assert_eq!(unique_view_id(&set, "other"), "other");
        assert!(append_view(&mut set, "still", pose, None, &[]).is_err());

        // Known intrinsics are used as given (sized to the picture).
        let known = StillOptions {
            intrinsics: Some(intrinsics_from_focal(1000.0, 0.0, 1280, 960)),
            ..StillOptions::default()
        };
        let solve = solve_still(&picture, &[], &set, &known);
        assert_eq!(solve.seed_source, "given intrinsics");
        assert_eq!((solve.seed.width, solve.seed.cx), (1280, 640.0));
        let pose = solve.pose.unwrap();
        assert!(pose.focal.is_none() && pose.k1.is_none());
    }

    #[test]
    fn warnings_name_the_weak_cases() {
        let (truth, view, set) = scene();
        let picture = render(
            &truth,
            &view,
            &set.markers[..1],
            &Dictionary::aruco_5x5_100(),
            &Render::default(),
        );
        // Unknown intrinsics cannot be fixed by one card (four corners,
        // eight unknowns); known ones solve, with the warning.
        let unknown = solve_still(&picture, &[], &set, &StillOptions::default());
        assert!(unknown.pose.is_err() && unknown.warnings.is_empty());
        let known = StillOptions {
            intrinsics: Some(intrinsics_from_focal(1000.0, 0.0, 1280, 960)),
            ..StillOptions::default()
        };
        let solve = solve_still(&picture, &[], &set, &known);
        assert!(solve.pose.is_ok(), "{:?}", solve.pose);
        assert!(
            solve.warnings.iter().any(|w| w.contains("single card")),
            "{:?}",
            solve.warnings
        );
        let cam = CameraModel::pinhole(1, 1, 1.0, 1.0, 0.5, 0.5);
        let (seed, source) = seed_camera(
            &StillOptions::default(),
            Some(&Exif {
                focal_35mm: Some(36.0),
                ..Exif::default()
            }),
            1000,
            500,
        );
        assert_eq!(
            (seed.fx, source.as_str()),
            (1000.0, "EXIF 35 mm equivalent")
        );
        assert_eq!(camera_index(&mut set.clone(), &cam), 1);
    }
}
