//! Everything detection finds in one picture: the swatches and the
//! survey card's tags, the card's interior corners, and the edge blur.
//! The `view-detect` command and the `view_detect` operator both run
//! this and store the result on the view.

use volumetric_abi::viewset::{BoardSpec, CornerObs, MarkerObs, Observations};

use crate::blur::{EdgeBlur, edge_blur};
use crate::charuco::{CornerParams, LocatedCorner, locate_corners};
use crate::detect::{DetectParams, Detection, detect};
use crate::dict::Dictionary;
use crate::gray::Gray;

#[derive(Clone, Debug)]
pub struct ObserveOptions {
    /// The swatch family to look for, if any.
    pub swatches: Option<Dictionary>,
    /// The board to look for, if any; its family is detected too.
    pub board: Option<BoardSpec>,
    pub detect: DetectParams,
    pub corners: CornerParams,
}

impl Default for ObserveOptions {
    fn default() -> Self {
        Self {
            swatches: Some(Dictionary::aruco_5x5_100()),
            board: Some(BoardSpec::survey_card()),
            detect: DetectParams::default(),
            corners: CornerParams::default(),
        }
    }
}

impl ObserveOptions {
    /// The families detection runs with: the swatches' and the board's,
    /// once each.
    pub fn families(&self) -> Vec<Dictionary> {
        let mut out: Vec<Dictionary> = self.swatches.iter().cloned().collect();
        if let Some(board) = &self.board
            && let Some(dict) = Dictionary::by_name(&board.family)
            && !out.iter().any(|d| d.name == dict.name)
        {
            out.push(dict);
        }
        out
    }
}

#[derive(Clone, Debug, Default)]
pub struct Observed {
    pub detections: Vec<Detection>,
    pub corners: Vec<LocatedCorner>,
    pub blur: EdgeBlur,
}

impl Observed {
    /// The detections of one family.
    pub fn of_family<'a>(&'a self, family: &'a str) -> impl Iterator<Item = &'a Detection> + 'a {
        self.detections.iter().filter(move |d| d.family == family)
    }

    /// As the view set stores it.
    pub fn to_observations(&self) -> Observations {
        Observations {
            markers: self
                .detections
                .iter()
                .map(|d| MarkerObs {
                    id: d.id,
                    family: d.family.to_string(),
                    corners: d.corners,
                    fit_px: d.fit_px,
                })
                .collect(),
            board: self
                .corners
                .iter()
                .map(|c| CornerObs {
                    id: c.id,
                    pixel: c.pixel,
                    fit_px: c.shift_px,
                })
                .collect(),
            blur_px: self.blur.worst(),
            features: Vec::new(),
            contours: Vec::new(),
        }
    }
}

/// Detects the families asked for, locates the board's corners and
/// measures the blur.
pub fn observe(gray: &Gray, options: &ObserveOptions) -> Observed {
    let families = options.families();
    let refs: Vec<&Dictionary> = families.iter().collect();
    let detections = detect(gray, &refs, &options.detect);
    let corners = match &options.board {
        Some(spec) => locate_corners(gray, spec, &detections, &options.corners),
        None => Vec::new(),
    };
    let blur = edge_blur(gray, &detections);
    Observed {
        detections,
        corners,
        blur,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{PlacedBoard, Render, render_scene, square_marker};
    use volumetric_abi::viewset::{CameraModel, View};

    #[test]
    fn a_card_and_swatches_are_observed_together() {
        // Straight down from 1 m at f = 2500: the card and two swatches
        // beside it.
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
        let picture = render_scene(
            &camera,
            &view,
            &swatches,
            &Dictionary::aruco_5x5_100(),
            std::slice::from_ref(&board),
            &Render {
                blur_sigma: 0.8,
                ..Render::default()
            },
        );
        let options = ObserveOptions::default();
        assert_eq!(options.families().len(), 2);
        let seen = observe(&picture, &options);
        assert_eq!(
            seen.of_family("5x5_100").map(|d| d.id).collect::<Vec<_>>(),
            vec![3, 42]
        );
        assert_eq!(seen.of_family("36h11").count(), 66);
        assert_eq!(seen.corners.len(), 110);
        assert!(
            seen.blur.worst().is_some_and(|b| b > 0.5 && b < 1.3),
            "{:?}",
            seen.blur
        );
        let obs = seen.to_observations();
        assert_eq!(obs.markers.len(), 68);
        assert_eq!(obs.board.len(), 110);
        assert_eq!(obs.blur_px, seen.blur.worst());
        // Without a board only the swatches are looked for, and their
        // family alone counts.
        let only = ObserveOptions {
            board: None,
            ..ObserveOptions::default()
        };
        assert_eq!(only.families().len(), 1);
        let seen = observe(&picture, &only);
        assert_eq!(seen.detections.len(), 2);
        assert!(seen.corners.is_empty());
    }
}
