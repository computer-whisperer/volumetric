//! View Solve Operator.
//!
//! Poses a photograph from the marker cards it shows, against the marker
//! map a view set carries, and appends it to the set as a view: the still
//! becomes posed evidence beside the scan's own frames. The pipeline is
//! `cv_core::still`, the same one the `view-solve` command runs. See
//! README.md (the operator's docs) for the conventions.
//!
//! Inputs:
//! - Input 0: ViewSet — the set with the marker map (and the views so far)
//! - Input 1: Blob — the picture (JPEG or PNG)
//! - Input 2: CBOR configuration, see [`ViewSolveConfig`].
//!
//! Output 0: ViewSet — the input set with the posed still appended.

use cv_core::dict::Dictionary;
use cv_core::gray::Gray;
use cv_core::still::{
    StillOptions, StillSolve, append_view, intrinsics_from_focal, solve_still, unique_view_id,
};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
use volumetric_abi::viewset::{ViewSet, decode_viewset};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct ViewSolveConfig {
    /// The new view's id; a suffix is added when the set already has it.
    pub id: String,
    /// `5x5_100` (the swatch cards) or `4x4_50` (the calibration board).
    pub dictionary: String,
    /// Horizontal field of view seeding the focal when the picture has no
    /// 35 mm equivalent and `focal_px` is 0, degrees.
    pub fov_deg: f64,
    /// Known focal length in pixels (principal point at the centre); 0
    /// means unknown, so the focal and the first radial term are solved.
    pub focal_px: f64,
    /// Known first radial distortion term, with `focal_px`.
    pub k1: f64,
    /// Solve the focal even when `focal_px` is given.
    pub solve_focal: bool,
    /// Solve the first radial term even when `focal_px` is given.
    pub solve_distortion: bool,
    /// Keep the picture in the view (it is what look-through shows).
    pub embed_image: bool,
}

impl Default for ViewSolveConfig {
    fn default() -> Self {
        Self {
            id: "still".to_string(),
            dictionary: "5x5_100".to_string(),
            fov_deg: 70.0,
            focal_px: 0.0,
            k1: 0.0,
            solve_focal: false,
            solve_distortion: false,
            embed_image: true,
        }
    }
}

/// A solved still appended to its set.
pub struct Solved {
    pub set: ViewSet,
    pub view_id: String,
    pub solve: StillSolve,
}

/// Solves `picture` against the set in `set_bytes` and appends the view.
pub fn solve(set_bytes: &[u8], picture: &[u8], config: &ViewSolveConfig) -> Result<Solved, String> {
    let mut set = decode_viewset(set_bytes)?;
    let dictionary = Dictionary::by_name(&config.dictionary).ok_or_else(|| {
        format!(
            "unknown dictionary '{}'; expected 5x5_100, 4x4_50 or 36h11",
            config.dictionary
        )
    })?;
    let decoded = image::load_from_memory(picture)
        .map_err(|e| format!("the picture does not decode: {e}"))?
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    let gray = Gray::from_rgb8(width, height, decoded.as_raw());
    let options = StillOptions {
        dictionary,
        intrinsics: (config.focal_px > 0.0)
            .then(|| intrinsics_from_focal(config.focal_px, config.k1, width, height)),
        fov_deg: config.fov_deg,
        solve_focal: config.solve_focal,
        solve_distortion: config.solve_distortion,
        ..StillOptions::default()
    };
    let solve = solve_still(&gray, picture, &set, &options);
    let pose = match &solve.pose {
        Ok(pose) => pose.clone(),
        Err(err) => {
            return Err(format!(
                "no pose: {err} ({} cards found, {} in the map)",
                solve.detections.len(),
                solve.in_map(&set)
            ));
        }
    };
    let base = if config.id.trim().is_empty() {
        "still"
    } else {
        config.id.trim()
    };
    let view_id = unique_view_id(&set, base);
    append_view(
        &mut set,
        &view_id,
        &pose,
        config.embed_image.then(|| picture.to_vec()),
        &[],
    )?;
    Ok(Solved {
        set,
        view_id,
        solve,
    })
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let set = read_input(0);
    let picture = read_input(1);
    let config = {
        let cfg = read_input(2);
        if cfg.is_empty() {
            ViewSolveConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&cfg)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    if set.is_empty() {
        report_error("no view set: wire the set whose marker map the still was taken against");
        return;
    }
    if picture.is_empty() {
        report_error("no picture: wire the still's bytes");
        return;
    }
    match solve(&set, &picture, &config) {
        Ok(solved) => {
            post_warning(&format!(
                "view '{}': {}",
                solved.view_id,
                solved.solve.summary(&solved.set)
            ));
            for warning in &solved.solve.warnings {
                post_warning(&format!("view '{}': {warning}", solved.view_id));
            }
            post_output(0, &volumetric_abi::viewset::encode_viewset(&solved.set));
        }
        Err(e) => report_error(&format!("view solve failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ id: tstr .default "still", dictionary: "5x5_100" / "4x4_50" .default "5x5_100", fov_deg: float .ge 1.0 .default 70.0, focal_px: float .ge 0.0 .default 0.0, k1: float .default 0.0, solve_focal: bool .default false, solve_distortion: bool .default false, embed_image: bool .default true }"#
            .to_string();
        OperatorMetadata {
            name: "view_solve_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Solve Still".to_string(),
            description: "Pose a photograph from the marker cards it shows, against the view set's map, and add it to the set.".to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<rect x="3" y="5" width="18" height="14" rx="2"/>"##,
                r##"<circle cx="12" cy="12" r="3.5"/>"##,
                r##"<path d="M7 5l1.5-2h7L17 5"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ViewSet,
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec![
                "Views".to_string(),
                "Picture".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::ViewSet],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::board::{Render, render, square_marker};
    use cv_core::pnp::pose_difference;
    use volumetric_abi::viewset::{CameraModel, View, encode_viewset};

    /// A binary PGM of the picture.
    fn pgm(gray: &Gray) -> Vec<u8> {
        let mut out = format!("P5\n{} {}\n255\n", gray.width, gray.height).into_bytes();
        out.extend_from_slice(&gray.pixels);
        out
    }

    #[test]
    fn a_rendered_still_joins_the_set() {
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
        let picture = pgm(&render(
            &truth,
            &view,
            &markers,
            &Dictionary::aruco_5x5_100(),
            &Render::default(),
        ));
        let set = ViewSet {
            board: None,
            schema: 1,
            world: Default::default(),
            provenance: Default::default(),
            cameras: vec![truth.clone()],
            views: Vec::new(),
            markers,
        };
        let bytes = encode_viewset(&set);

        // Known focal: the pose is exact to the corner accuracy.
        let config = ViewSolveConfig {
            id: "phone".to_string(),
            focal_px: 1000.0,
            ..ViewSolveConfig::default()
        };
        let solved = solve(&bytes, &picture, &config).unwrap();
        assert_eq!(solved.view_id, "phone");
        let (added, camera) = solved.set.view("phone").unwrap();
        let (angle, dist) = pose_difference(&added.camera_to_world, &view.camera_to_world);
        assert!(angle < 0.002 && dist < 0.003, "{angle} rad, {dist} m");
        assert_eq!(camera.fx, 1000.0);
        assert!(added.image.is_some());
        assert!(added.tags.iter().any(|t| t == "solved:markers"));
        assert!(
            solved.solve.warnings.is_empty(),
            "{:?}",
            solved.solve.warnings
        );

        // Unknown focal, seeded 15% off, no picture kept; the id already
        // taken gets a suffix.
        let again = encode_viewset(&solved.set);
        let config = ViewSolveConfig {
            id: "phone".to_string(),
            fov_deg: 2.0 * (640.0f64 / 1150.0).atan().to_degrees(),
            embed_image: false,
            ..ViewSolveConfig::default()
        };
        let solved = solve(&again, &picture, &config).unwrap();
        assert_eq!(solved.view_id, "phone_2");
        let (added, camera) = solved.set.view("phone_2").unwrap();
        assert!(added.image.is_none());
        assert!((camera.fx - 1000.0).abs() < 15.0, "{}", camera.fx);
        assert_eq!(solved.set.views.len(), 2);

        // No cards: an error naming what was found.
        let blank = pgm(&Gray::new(64, 48));
        let err = match solve(&bytes, &blank, &ViewSolveConfig::default()) {
            Ok(_) => panic!("a blank picture must not solve"),
            Err(err) => err,
        };
        assert!(err.contains("0 cards found"), "{err}");
        assert!(solve(&bytes, b"not a picture", &ViewSolveConfig::default()).is_err());
        assert!(
            solve(
                &bytes,
                &picture,
                &ViewSolveConfig {
                    dictionary: "7x7".to_string(),
                    ..ViewSolveConfig::default()
                }
            )
            .is_err()
        );
    }
}
