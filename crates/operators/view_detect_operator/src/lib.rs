//! View Detect Operator.
//!
//! Finds the swatches, the survey card's tags and its interior corners in
//! every picture of a view set and stores the observations on the views.
//! The pipeline is `cv_core::observe`, the same one the `view-detect`
//! command runs. See README.md (the operator's docs) for the conventions.
//!
//! Inputs:
//! - Input 0: ViewSet — the set whose pictures to detect in
//! - Input 1: CBOR configuration, see [`ViewDetectConfig`].
//!
//! Output 0: ViewSet — the input set with observations on its views.

use cv_core::detect::DetectParams;
use cv_core::dict::Dictionary;
use cv_core::gray::Gray;
use cv_core::observe::{ObserveOptions, observe};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
use volumetric_abi::viewset::{Board, BoardSpec, ViewSet, decode_viewset};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct ViewDetectConfig {
    /// `5x5_100` (the swatch cards), `4x4_50` (the calibration board)
    /// or `none`.
    pub dictionary: String,
    /// Look for the survey card.
    pub card: bool,
    pub squares_x: u32,
    pub squares_y: u32,
    pub pitch_x_mm: f64,
    pub pitch_y_mm: f64,
    pub marker_mm: f64,
    /// The card's marker family.
    pub family: String,
    pub first_id: u32,
    /// Quads are searched on the picture reduced to about this many
    /// pixels on its longer side (0 = full resolution).
    pub search_px: u32,
}

impl Default for ViewDetectConfig {
    fn default() -> Self {
        let card = BoardSpec::survey_card();
        Self {
            dictionary: "5x5_100".to_string(),
            card: true,
            squares_x: card.squares_x,
            squares_y: card.squares_y,
            pitch_x_mm: card.pitch_x_m * 1e3,
            pitch_y_mm: card.pitch_y_m * 1e3,
            marker_mm: card.marker_m * 1e3,
            family: card.family,
            first_id: card.first_id,
            search_px: 1600,
        }
    }
}

impl ViewDetectConfig {
    pub fn options(&self) -> Result<ObserveOptions, String> {
        let swatches = match self.dictionary.to_ascii_lowercase().as_str() {
            "none" | "" => None,
            name => Some(Dictionary::by_name(name).ok_or_else(|| {
                format!("unknown dictionary '{name}'; expected 5x5_100, 4x4_50, 36h11 or none")
            })?),
        };
        let board = if self.card {
            let spec = BoardSpec {
                squares_x: self.squares_x,
                squares_y: self.squares_y,
                pitch_x_m: self.pitch_x_mm * 1e-3,
                pitch_y_m: self.pitch_y_mm * 1e-3,
                marker_m: self.marker_mm * 1e-3,
                family: self.family.clone(),
                first_id: self.first_id,
            };
            spec.validate().map_err(|e| format!("card: {e}"))?;
            Dictionary::by_name(&spec.family)
                .ok_or_else(|| format!("unknown card family '{}'", spec.family))?;
            Some(spec)
        } else {
            None
        };
        if swatches.is_none() && board.is_none() {
            return Err("nothing to look for: give a dictionary or a card".to_string());
        }
        Ok(ObserveOptions {
            swatches,
            board,
            detect: DetectParams {
                search_px: self.search_px,
                ..DetectParams::default()
            },
            ..ObserveOptions::default()
        })
    }
}

/// What one view's detection found, for the report.
#[derive(Clone, Debug)]
pub struct ViewSummary {
    pub id: String,
    pub line: String,
}

/// The set with observations, the per-view summaries, and the views
/// skipped for lack of a picture.
pub struct Detected {
    pub set: ViewSet,
    pub views: Vec<ViewSummary>,
    pub skipped: Vec<String>,
}

/// Detects in every picture of the set in `set_bytes`.
pub fn detect(set_bytes: &[u8], config: &ViewDetectConfig) -> Result<Detected, String> {
    let mut set = decode_viewset(set_bytes)?;
    let options = config.options()?;
    let mut views = Vec::new();
    let mut skipped = Vec::new();
    for view in &mut set.views {
        let Some(bytes) = &view.image else {
            skipped.push(view.id.clone());
            continue;
        };
        let decoded = image::load_from_memory(bytes)
            .map_err(|e| format!("view '{}': the picture does not decode: {e}", view.id))?
            .to_rgb8();
        let (width, height) = decoded.dimensions();
        let camera = &set.cameras[view.camera as usize];
        if (width, height) != (camera.width, camera.height) {
            return Err(format!(
                "view '{}': the embedded picture is {width}x{height} but its camera is {}x{}; \
                 detection needs the full picture (import the stills with --embed full, or run \
                 view-detect on the command line, which reads the originals)",
                view.id, camera.width, camera.height
            ));
        }
        let gray = Gray::from_rgb8(width, height, decoded.as_raw());
        let seen = observe(&gray, &options);
        let mut parts: Vec<String> = options
            .families()
            .iter()
            .map(|d| format!("{} {}", seen.of_family(d.name).count(), d.name))
            .collect();
        if options.board.is_some() {
            parts.push(format!("{} card corners", seen.corners.len()));
        }
        match seen.blur.worst() {
            Some(b) => parts.push(format!("blur {b:.2} px")),
            None => parts.push("no blur measure".to_string()),
        }
        views.push(ViewSummary {
            id: view.id.clone(),
            line: parts.join(", "),
        });
        view.observations = Some(seen.to_observations());
    }
    if let Some(spec) = &options.board {
        let corners = match set.board.take() {
            Some(board) if &board.spec == spec => board.corners,
            _ => Vec::new(),
        };
        set.board = Some(Board {
            spec: spec.clone(),
            corners,
        });
    }
    Ok(Detected {
        set,
        views,
        skipped,
    })
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let set = read_input(0);
    let config = {
        let cfg = read_input(1);
        if cfg.is_empty() {
            ViewDetectConfig::default()
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
        report_error("no view set: wire the set whose pictures to detect in");
        return;
    }
    match detect(&set, &config) {
        Ok(found) => {
            for v in &found.views {
                post_warning(&format!("view '{}': {}", v.id, v.line));
            }
            if !found.skipped.is_empty() {
                post_warning(&format!(
                    "{} views without a picture skipped: {}",
                    found.skipped.len(),
                    found.skipped.join(", ")
                ));
            }
            post_output(0, &volumetric_abi::viewset::encode_viewset(&found.set));
        }
        Err(e) => report_error(&format!("view detect failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ dictionary: "5x5_100" / "4x4_50" / "none" .default "5x5_100", card: bool .default true, squares_x: int .ge 2 .default 12, squares_y: int .ge 2 .default 11, pitch_x_mm: float .ge 0.1 .default 17.9443, pitch_y_mm: float .ge 0.1 .default 17.7451, marker_mm: float .ge 0.1 .default 12.7, family: "36h11" / "5x5_100" / "4x4_50" .default "36h11", first_id: int .ge 0 .default 100, search_px: int .ge 0 .default 1600 }"#
            .to_string();
        OperatorMetadata {
            name: "view_detect_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Detect Cards".to_string(),
            description: "Find the swatches, the survey card's tags and its corners in every picture of a view set, and store the observations on the views.".to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<rect x="3" y="3" width="8" height="8" rx="1"/>"##,
                r##"<rect x="13" y="13" width="8" height="8" rx="1"/>"##,
                r##"<path d="M13 3h8v8M3 13v8h8"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ViewSet,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["Views".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ViewSet],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::board::{PlacedBoard, Render, render_scene, square_marker};
    use volumetric_abi::viewset::{CameraModel, View, encode_viewset};

    /// A binary PGM of the picture.
    fn pgm(gray: &Gray) -> Vec<u8> {
        let mut out = format!("P5\n{} {}\n255\n", gray.width, gray.height).into_bytes();
        out.extend_from_slice(&gray.pixels);
        out
    }

    #[test]
    fn a_rendered_card_and_swatches_are_observed_on_the_view() {
        let camera = CameraModel::pinhole(1600, 1200, 2500.0, 2500.0, 800.0, 600.0);
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
            [-0.28, 0.1, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        );
        let swatches = vec![
            square_marker(3, [0.0, 0.2, 0.0], 0.06, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
            square_marker(
                42,
                [0.1, 0.05, 0.0],
                0.06,
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ),
        ];
        let picture = pgm(&render_scene(
            &camera,
            &view,
            &swatches,
            &Dictionary::aruco_5x5_100(),
            std::slice::from_ref(&board),
            &Render {
                blur_sigma: 0.8,
                ..Render::default()
            },
        ));
        let mut with_picture = view.clone();
        with_picture.image = Some(picture);
        let set = ViewSet {
            cameras: vec![camera],
            views: vec![
                with_picture,
                View::posed("blind", 0, view.camera_to_world.unwrap()),
            ],
            ..ViewSet::default()
        };
        let found = detect(&encode_viewset(&set), &ViewDetectConfig::default()).unwrap();
        assert_eq!(found.skipped, vec!["blind".to_string()]);
        assert_eq!(found.views.len(), 1);
        assert!(
            found.views[0].line.contains("2 5x5_100"),
            "{}",
            found.views[0].line
        );
        let obs = found.set.views[0].observations.as_ref().unwrap();
        assert_eq!(
            obs.markers.iter().filter(|m| m.family == "5x5_100").count(),
            2
        );
        assert_eq!(
            obs.markers.iter().filter(|m| m.family == "36h11").count(),
            66
        );
        assert_eq!(obs.board.len(), 110);
        assert!(obs.blur_px.is_some());
        assert_eq!(
            found.set.board.as_ref().unwrap().spec,
            BoardSpec::survey_card()
        );
        assert!(found.set.views[1].observations.is_none());
        // The set round-trips through the value.
        let again = decode_viewset(&encode_viewset(&found.set)).unwrap();
        assert_eq!(again, found.set);
        // Swatches only, and a bad configuration.
        let only = ViewDetectConfig {
            card: false,
            ..ViewDetectConfig::default()
        };
        let found = detect(&encode_viewset(&set), &only).unwrap();
        assert_eq!(
            found.set.views[0]
                .observations
                .as_ref()
                .unwrap()
                .markers
                .len(),
            2
        );
        assert!(found.set.board.is_none());
        assert!(
            ViewDetectConfig {
                dictionary: "none".to_string(),
                card: false,
                ..ViewDetectConfig::default()
            }
            .options()
            .is_err()
        );
        assert!(
            ViewDetectConfig {
                marker_mm: 30.0,
                ..ViewDetectConfig::default()
            }
            .options()
            .is_err()
        );
    }
}
