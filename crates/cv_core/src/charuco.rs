//! ChArUco corner localisation: the interior corners of a chessboard
//! whose white squares carry markers, placed from the decoded markers
//! beside them and refined to the saddle point of the grey picture.
//!
//! Each decoded tag gives a board-to-picture homography; the interior
//! corners of its square are predicted from it, predictions for the same
//! corner averaged, and every corner refined by the gradient-orthogonality
//! iteration (`cornerSubPix`): the point every gradient in a window is
//! perpendicular to. A corner with no decoded tag beside it is predicted
//! from a homography over every decoded tag instead (the board is flat),
//! which the refinement then corrects for the lens; either way a corner is
//! dropped when the refinement leaves the prediction by more than a
//! fraction of the square, or when the four squares around it do not show
//! the chessboard's pattern (a corner under the subject, or off the card).

use std::collections::BTreeMap;

use volumetric_abi::viewset::BoardSpec;

use crate::detect::{Detection, apply_homography, homography, refine_saddle};
use crate::gray::Gray;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CornerParams {
    /// Refinement half-window as a fraction of the square's size in
    /// pixels, clamped to the two bounds below.
    pub window_fraction: f64,
    pub min_half_window: u32,
    pub max_half_window: u32,
    /// A corner that moves further than this fraction of the square from
    /// its prediction (at least `min_shift_px`) is dropped.
    pub max_shift_fraction: f64,
    pub min_shift_px: f64,
    pub iterations: u32,
    /// Stop when a step is shorter than this, pixels.
    pub epsilon: f64,
    /// The four squares around a corner, sampled inside the white
    /// squares' margin around their markers (at least `min_probe_px` in
    /// from the corner), must differ by this many grey levels between
    /// the dark pair and the bright pair.
    pub min_probe_px: f64,
    pub min_contrast: f64,
}

impl Default for CornerParams {
    fn default() -> Self {
        Self {
            window_fraction: 0.1,
            min_half_window: 3,
            max_half_window: 20,
            max_shift_fraction: 0.1,
            min_shift_px: 2.0,
            iterations: 40,
            epsilon: 0.005,
            min_probe_px: 2.5,
            min_contrast: 15.0,
        }
    }
}

/// An interior corner found in a picture.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LocatedCorner {
    pub id: u32,
    pub pixel: [f64; 2],
    /// Where the neighbouring tags predicted it.
    pub predicted: [f64; 2],
    /// How far the refinement moved it, pixels.
    pub shift_px: f64,
    /// Decoded tags beside it (one or two), or 0 when it was predicted
    /// from the whole board's homography.
    pub tags: u32,
    /// Grey levels between the dark and the bright squares around it.
    pub contrast: f64,
}

/// Locates the board's interior corners from the tags detected in the
/// picture (other families in `tags` are ignored), sorted by id.
pub fn locate_corners(
    gray: &Gray,
    spec: &BoardSpec,
    tags: &[Detection],
    params: &CornerParams,
) -> Vec<LocatedCorner> {
    // Predictions per corner: (pixel, square size in pixels, the picture
    // directions of the board's x and y axes there).
    let mut predictions: BTreeMap<u32, Vec<Prediction>> = BTreeMap::new();
    let mut global: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for tag in tags {
        if tag.family != spec.family {
            continue;
        }
        let Some((row, col)) = spec.marker_square(tag.id) else {
            continue;
        };
        let board = spec.marker_corners(tag.id).expect("marker on the board");
        let Some(h) = homography(&board, &tag.corners) else {
            continue;
        };
        let side_px = tag.perimeter() * 0.25;
        let square_px = side_px * (spec.pitch_x_m + spec.pitch_y_m) * 0.5 / spec.marker_m;
        for (id, corner) in spec.corners_of_square(row, col) {
            predictions
                .entry(id)
                .or_default()
                .push(Prediction::at(&h, corner, square_px, spec));
        }
    }
    // Corners beside no decoded tag: predicted from the board as a whole.
    if let Some((h, square_px)) = board_homography(spec, tags) {
        for id in 0..spec.n_corners() {
            if predictions.contains_key(&id) {
                continue;
            }
            let corner = spec.corner(id).expect("interior corner");
            predictions.insert(id, vec![Prediction::at(&h, corner, square_px, spec)]);
            global.insert(id);
        }
    }
    let (w, h) = (f64::from(gray.width), f64::from(gray.height));
    let cols = spec.squares_x - 1;
    let mut out = Vec::with_capacity(predictions.len());
    for (id, preds) in predictions {
        let n = preds.len() as f64;
        let predicted = [
            preds.iter().map(|p| p.pixel[0]).sum::<f64>() / n,
            preds.iter().map(|p| p.pixel[1]).sum::<f64>() / n,
        ];
        let square_px = preds.iter().map(|p| p.square_px).sum::<f64>() / n;
        let axis_x = mean_direction(preds.iter().map(|p| p.axis_x));
        let axis_y = mean_direction(preds.iter().map(|p| p.axis_y));
        // The window must not reach the markers in the white squares:
        // their border starts a margin of (pitch − marker) / 2 in.
        let margin_frac = (spec.pitch_x_m.min(spec.pitch_y_m) - spec.marker_m)
            / (2.0 * spec.pitch_x_m.min(spec.pitch_y_m));
        let max_half = ((0.8 * margin_frac * square_px).floor() as u32).max(params.min_half_window);
        let half = ((params.window_fraction * square_px).round() as u32)
            .clamp(params.min_half_window, params.max_half_window.min(max_half));
        let margin = f64::from(half) + 2.0;
        if predicted[0] < margin
            || predicted[1] < margin
            || predicted[0] > w - margin
            || predicted[1] > h - margin
        {
            continue;
        }
        let Some(pixel) = refine_saddle(gray, predicted, half, params.iterations, params.epsilon)
        else {
            continue;
        };
        let shift_px =
            ((pixel[0] - predicted[0]).powi(2) + (pixel[1] - predicted[1]).powi(2)).sqrt();
        if shift_px > (params.max_shift_fraction * square_px).max(params.min_shift_px) {
            continue;
        }
        // The four squares around the corner: the one at +x +y on the
        // board is (row + 1, col + 1) in squares, dark when that sum is
        // even; the diagonal pair shares a shade.
        let (row, col) = (id / cols + 1, id % cols + 1);
        let dark_at_plus = (row + col) % 2 == 0;
        // The probe sits at 0.6 of that margin so blur does not reach the
        // marker's border.
        let probe = (0.6 * margin_frac * square_px).max(params.min_probe_px);
        let shade = |sx: f64, sy: f64| {
            let p = [
                pixel[0] + probe * (sx * axis_x[0] + sy * axis_y[0]),
                pixel[1] + probe * (sx * axis_x[1] + sy * axis_y[1]),
            ];
            let mut acc = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    acc += gray.sample(p[0] + f64::from(dx), p[1] + f64::from(dy));
                }
            }
            acc / 9.0
        };
        let (pp, mm, pm, mp) = (
            shade(1.0, 1.0),
            shade(-1.0, -1.0),
            shade(1.0, -1.0),
            shade(-1.0, 1.0),
        );
        let (dark, bright) = if dark_at_plus {
            ((pp, mm), (pm, mp))
        } else {
            ((pm, mp), (pp, mm))
        };
        let contrast = (bright.0 + bright.1 - dark.0 - dark.1) * 0.5;
        if contrast < params.min_contrast
            || (dark.0 - dark.1).abs() > contrast
            || (bright.0 - bright.1).abs() > contrast
        {
            continue;
        }
        out.push(LocatedCorner {
            id,
            pixel,
            predicted,
            shift_px,
            tags: if global.contains(&id) {
                0
            } else {
                preds.len() as u32
            },
            contrast,
        });
    }
    out
}

/// Where a homography puts a corner, with the square's size and the
/// board axes' directions in the picture there.
struct Prediction {
    pixel: [f64; 2],
    square_px: f64,
    axis_x: [f64; 2],
    axis_y: [f64; 2],
}

impl Prediction {
    fn at(h: &[[f64; 3]; 3], corner: [f64; 2], square_px: f64, spec: &BoardSpec) -> Self {
        let pixel = apply_homography(h, corner);
        let step = 0.1 * spec.pitch_x_m.min(spec.pitch_y_m);
        let px = apply_homography(h, [corner[0] + step, corner[1]]);
        let py = apply_homography(h, [corner[0], corner[1] + step]);
        Self {
            pixel,
            square_px,
            axis_x: unit([px[0] - pixel[0], px[1] - pixel[1]]),
            axis_y: unit([py[0] - pixel[0], py[1] - pixel[1]]),
        }
    }
}

fn unit(v: [f64; 2]) -> [f64; 2] {
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if n < 1e-12 {
        [1.0, 0.0]
    } else {
        [v[0] / n, v[1] / n]
    }
}

fn mean_direction(dirs: impl Iterator<Item = [f64; 2]>) -> [f64; 2] {
    let mut sum = [0.0, 0.0];
    for d in dirs {
        sum[0] += d[0];
        sum[1] += d[1];
    }
    unit(sum)
}

/// A board-to-picture homography over every decoded tag of the board's
/// family, refitted once without the tags that disagree with it, and the
/// square's size in pixels; `None` under four tags.
fn board_homography(spec: &BoardSpec, tags: &[Detection]) -> Option<([[f64; 3]; 3], f64)> {
    let mut pairs: Vec<(u32, [[f64; 2]; 4], [[f64; 2]; 4])> = tags
        .iter()
        .filter(|t| t.family == spec.family)
        .filter_map(|t| spec.marker_corners(t.id).map(|b| (t.id, b, t.corners)))
        .collect();
    let fit = |pairs: &[(u32, [[f64; 2]; 4], [[f64; 2]; 4])]| {
        let src: Vec<[f64; 2]> = pairs
            .iter()
            .flat_map(|(_, b, _)| b.iter().copied())
            .collect();
        let dst: Vec<[f64; 2]> = pairs
            .iter()
            .flat_map(|(_, _, p)| p.iter().copied())
            .collect();
        homography(&src, &dst)
    };
    if pairs.len() < 4 {
        return None;
    }
    let h = fit(&pairs)?;
    let errors: Vec<f64> = pairs
        .iter()
        .map(|(_, b, p)| {
            (0..4)
                .map(|j| {
                    let q = apply_homography(&h, b[j]);
                    ((q[0] - p[j][0]).powi(2) + (q[1] - p[j][1]).powi(2)).sqrt()
                })
                .sum::<f64>()
                / 4.0
        })
        .collect();
    let mut sorted = errors.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cut = (3.0 * sorted[sorted.len() / 2]).max(2.0);
    let keep: Vec<bool> = errors.iter().map(|e| *e <= cut).collect();
    if keep.iter().filter(|k| **k).count() >= 4 && keep.iter().any(|k| !k) {
        let mut i = 0;
        pairs.retain(|_| {
            let k = keep[i];
            i += 1;
            k
        });
    }
    let h = fit(&pairs)?;
    let side_px = pairs
        .iter()
        .map(|(_, _, p)| {
            (0..4)
                .map(|j| {
                    let (a, b) = (p[j], p[(j + 1) % 4]);
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
                })
                .sum::<f64>()
                / 4.0
        })
        .sum::<f64>()
        / pairs.len() as f64;
    let square_px = side_px * (spec.pitch_x_m + spec.pitch_y_m) * 0.5 / spec.marker_m;
    Some((h, square_px))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{PlacedBoard, Render, render_board};
    use crate::detect::{DetectParams, detect};
    use crate::dict::Dictionary;
    use volumetric_abi::viewset::{CameraModel, View};

    /// The survey card on the floor seen from 0.87 m, 30 degrees off its
    /// normal, 1600 x 1200 at f = 2800: squares of about 57 px, tags of
    /// about 40 by 36.
    fn scene() -> (CameraModel, View, PlacedBoard) {
        let camera = CameraModel::pinhole(1600, 1200, 2800.0, 2800.0, 800.0, 600.0);
        let pitch: f64 = 60f64.to_radians();
        let (s, c) = pitch.sin_cos();
        let forward = [0.0, c, -s];
        let down = [0.0, -s, -c];
        let view = View::posed(
            "cam",
            0,
            [
                1.0, down[0], forward[0], 0.02, //
                0.0, down[1], forward[1], -0.33, //
                0.0, down[2], forward[2], 0.75, //
            ],
        );
        let board = PlacedBoard::new(
            BoardSpec::survey_card(),
            [-0.11, 0.2, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        );
        (camera, view, board)
    }

    fn errors(
        found: &[LocatedCorner],
        camera: &CameraModel,
        view: &View,
        board: &PlacedBoard,
    ) -> Vec<f64> {
        found
            .iter()
            .map(|c| {
                let truth = view
                    .project(camera, board.corner_world(c.id).unwrap())
                    .unwrap();
                ((c.pixel[0] - truth[0]).powi(2) + (c.pixel[1] - truth[1]).powi(2)).sqrt()
            })
            .collect()
    }

    #[test]
    fn corners_are_located_to_a_few_hundredths_of_a_pixel() {
        let (camera, view, board) = scene();
        let tags = Dictionary::apriltag_36h11();
        // A clean render's coverage is quantised to a sixth of a pixel,
        // which the saddle fit feels; real pictures carry half a pixel
        // of blur or more, where it settles below a tenth.
        for (blur, tolerance) in [(0.0, 0.25), (0.8, 0.15), (1.5, 0.15)] {
            let picture = render_board(
                &camera,
                &view,
                &board,
                &Render {
                    supersample: 6,
                    blur_sigma: blur,
                    ..Render::default()
                },
            );
            let detections = detect(&picture, &[&tags], &DetectParams::default());
            assert!(
                detections.len() >= 60,
                "blur {blur}: {} tags",
                detections.len()
            );
            assert!(detections.iter().all(|d| (100..166).contains(&d.id)));
            let found =
                locate_corners(&picture, &board.spec, &detections, &CornerParams::default());
            assert!(found.len() >= 105, "blur {blur}: {} corners", found.len());
            let errs = errors(&found, &camera, &view, &board);
            let worst = errs.iter().cloned().fold(0.0, f64::max);
            let mean = errs.iter().sum::<f64>() / errs.len() as f64;
            assert!(
                worst < tolerance && mean < tolerance * 0.5,
                "blur {blur}: worst {worst:.3} px, mean {mean:.3} px"
            );
            // A single tag's homography extrapolated to the square's
            // corners predicts within a couple of pixels at this tilt.
            assert!(found.iter().all(|c| c.tags >= 1 && c.shift_px < 3.0));
            assert!(found.iter().all(|c| c.contrast > 100.0), "{:?}", found[0]);
            assert!(found.iter().any(|c| c.tags == 2));
        }
    }

    #[test]
    fn one_tag_beside_a_corner_is_enough_and_the_border_is_respected() {
        let (camera, view, board) = scene();
        let tags = Dictionary::apriltag_36h11();
        let picture = render_board(
            &camera,
            &view,
            &board,
            &Render {
                supersample: 6,
                blur_sigma: 0.8,
                ..Render::default()
            },
        );
        let detections = detect(&picture, &[&tags], &DetectParams::default());
        let all = locate_corners(&picture, &board.spec, &detections, &CornerParams::default());
        // Keep the tags of even rows: every corner touches one white
        // square in an even row and one in an odd row, so each keeps
        // exactly one tag beside it.
        let half: Vec<Detection> = detections
            .iter()
            .filter(|d| board.spec.marker_square(d.id).unwrap().0 % 2 == 0)
            .cloned()
            .collect();
        let found = locate_corners(&picture, &board.spec, &half, &CornerParams::default());
        assert!(
            found.len() + 4 >= all.len(),
            "{} of {}",
            found.len(),
            all.len()
        );
        assert!(found.iter().all(|c| c.tags == 1));
        // With three tags left, the rest of the corners come from the
        // board's homography (four tags are needed to fit one).
        let three: Vec<Detection> = detections.iter().take(3).cloned().collect();
        let few = locate_corners(&picture, &board.spec, &three, &CornerParams::default());
        assert!(
            few.len() <= 12 && few.iter().all(|c| c.tags >= 1),
            "{}",
            few.len()
        );
        let four: Vec<Detection> = detections.iter().take(4).cloned().collect();
        let global = locate_corners(&picture, &board.spec, &four, &CornerParams::default());
        assert!(
            global.len() + 4 >= all.len(),
            "{} of {}",
            global.len(),
            all.len()
        );
        assert!(global.iter().filter(|c| c.tags == 0).count() >= 90);
        let errs = errors(&global, &camera, &view, &board);
        assert!(errs.iter().cloned().fold(0.0, f64::max) < 0.15);
        let errs = errors(&found, &camera, &view, &board);
        assert!(errs.iter().cloned().fold(0.0, f64::max) < 0.15);
        // Other families are ignored; no tags, no corners.
        let mut swatch = detections[0].clone();
        swatch.family = "5x5_100";
        swatch.id = 3;
        assert!(
            locate_corners(&picture, &board.spec, &[swatch], &CornerParams::default()).is_empty()
        );
        // A prediction off the picture is skipped, not refined.
        let mut off = detections[0].clone();
        for c in &mut off.corners {
            c[0] -= 2000.0;
        }
        assert!(locate_corners(&picture, &board.spec, &[off], &CornerParams::default()).is_empty());
        // A flat window has no saddle.
        assert!(refine_saddle(&Gray::new(64, 64), [32.0, 32.0], 5, 40, 0.005).is_none());
        // Corners under something are refused: paint over part of the
        // card and the corners there go, the others stay.
        let mut covered = picture.clone();
        let (w, h) = (covered.width, covered.height);
        for y in 0..h {
            for x in 0..w {
                if x > 700 && x < 1000 && y > 400 && y < 700 {
                    covered.set(x, y, 90);
                }
            }
        }
        let found = locate_corners(&covered, &board.spec, &detections, &CornerParams::default());
        let (inside, outside): (Vec<_>, Vec<_>) = all.iter().partition(|c| {
            c.pixel[0] > 700.0 && c.pixel[0] < 1000.0 && c.pixel[1] > 400.0 && c.pixel[1] < 700.0
        });
        assert!(inside.len() > 10 && outside.len() > 40);
        assert!(
            found.iter().all(|c| !inside.iter().any(|i| i.id == c.id)),
            "covered corners kept"
        );
        assert!(
            found.len() + 12 >= outside.len(),
            "{} of {}",
            found.len(),
            outside.len()
        );
    }
}
