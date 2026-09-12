//! Marker and card detection over a view set: every view's picture
//! observed, its observations stored on the view, the card's spec on the
//! set. The CLI's `view-detect` and the Python bindings are this call.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use cv_core::{Dictionary, Gray, ObserveOptions, Observed, observe};
use volumetric_abi::viewset::{Board, BoardSpec, ViewSet};

use crate::image::{Rgb, decode_rgb};
use crate::stills::full_picture;

/// A card spec from JSON in either the ABI's shape or the scanner's
/// `card.json` (`square_m`, `square_x_m`/`square_y_m`, `dictionary`).
pub fn card_spec_from_json(text: &str) -> Result<BoardSpec> {
    let value: serde_json::Value = serde_json::from_str(text).context("the card is not JSON")?;
    let spec = if value.get("pitch_x_m").is_some() {
        serde_json::from_value::<BoardSpec>(value).context("not a board spec")?
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
            .with_context(|| format!("unknown marker family '{family}' in the card"))?;
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

/// What one view's picture showed.
pub struct Detected {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub seen: Observed,
    pub seconds: f64,
}

/// The outcome over a set.
#[derive(Default)]
pub struct Detection {
    pub pictures: Vec<Detected>,
    /// Views with no picture to detect in.
    pub skipped: Vec<String>,
}

/// Observe every view (or the named ones) and store the observations on
/// the views and the card's spec on the set. `on_picture` sees each decoded
/// picture with what was found, for annotation or progress.
pub fn detect_views(
    set: &mut ViewSet,
    view_ids: &[String],
    options: &ObserveOptions,
    mut on_picture: impl FnMut(&Detected, &Rgb),
) -> Result<Detection> {
    let mut indices: Vec<usize> = (0..set.views.len()).collect();
    if !view_ids.is_empty() {
        for id in view_ids {
            if !set.views.iter().any(|v| &v.id == id) {
                bail!("no view '{id}' in the set");
            }
        }
        indices.retain(|&i| view_ids.contains(&set.views[i].id));
    }
    let mut outcome = Detection::default();
    for i in indices {
        let id = set.views[i].id.clone();
        if set.views[i].image.is_none() && set.views[i].source.is_none() {
            outcome.skipped.push(id);
            continue;
        }
        let photo = match full_picture(set, &set.views[i]).and_then(|bytes| decode_rgb(&bytes)) {
            Ok(photo) => photo,
            Err(err) => {
                eprintln!("view '{id}': no picture to detect in: {err:#}");
                outcome.skipped.push(id);
                continue;
            }
        };
        let start = Instant::now();
        let gray = Gray::from_rgb8(photo.width, photo.height, &photo.pixels);
        let seen = observe(&gray, options);
        // Detection replaces what was measured automatically; picks and
        // contours were made by hand and stay.
        let mut observations = seen.to_observations();
        if let Some(previous) = set.views[i].observations.take() {
            observations.features = previous.features;
            observations.contours = previous.contours;
        }
        set.views[i].observations = Some(observations);
        let detected = Detected {
            id,
            width: photo.width,
            height: photo.height,
            seen,
            seconds: start.elapsed().as_secs_f64(),
        };
        on_picture(&detected, &photo);
        outcome.pictures.push(detected);
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
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scanner_and_abi_card_shapes_both_read() {
        let scanner = r#"{"squares_x": 12, "squares_y": 11, "square_m": 0.01778, "marker_m": 0.0127,
            "dictionary": "DICT_APRILTAG_36h11", "square_x_m": 0.0179443, "square_y_m": 0.0177451, "first_id": 100}"#;
        let spec = card_spec_from_json(scanner).unwrap();
        assert_eq!(spec, BoardSpec::survey_card());
        let abi = serde_json::to_string(&BoardSpec::survey_card()).unwrap();
        assert_eq!(card_spec_from_json(&abi).unwrap(), BoardSpec::survey_card());
        assert!(card_spec_from_json("{\"squares_x\": 2}").is_err());
    }
}
