//! Survey Operator.
//!
//! Solves a session's cameras, poses and marker field from the
//! observations Detect Cards stored on a view set. The bundle is
//! `cv_core::survey`, the same one the `view-survey` command runs. See
//! README.md (the operator's docs) for the conventions.
//!
//! Inputs:
//! - Input 0: ViewSet — a detected set (observations on its views, the
//!   card as its board)
//! - Input 1: CBOR configuration, see [`SurveyConfig`].
//!
//! Outputs:
//! - Output 0: ViewSet — the set posed, with the solved cameras, markers
//!   and board.
//! - Output 1: F64Map — the fit report.

use cv_core::survey::{SurveyOptions, SurveyReport, survey};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
use volumetric_abi::viewset::{ViewSet, decode_viewset};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct SurveyConfig {
    /// The swatches' marker family.
    pub dictionary: String,
    /// Swatch ids at or above this are false decodes.
    pub max_swatch_id: u32,
    /// Card corners a view needs to be posed on the card at the start.
    pub min_card: u32,
    /// The soft-L1 scale, pixels.
    pub f_scale: f64,
    /// Views the bundle leaves above this rms are dropped.
    pub reject_px: f64,
    pub rounds: u32,
}

impl Default for SurveyConfig {
    fn default() -> Self {
        let o = SurveyOptions::default();
        Self {
            dictionary: o.swatch_family,
            max_swatch_id: o.max_swatch_id,
            min_card: o.min_card_corners as u32,
            f_scale: o.soft_l1_px,
            reject_px: o.reject_px,
            rounds: o.rounds as u32,
        }
    }
}

impl SurveyConfig {
    pub fn options(&self) -> Result<SurveyOptions, String> {
        if self.f_scale.is_nan()
            || self.f_scale <= 0.0
            || self.reject_px.is_nan()
            || self.reject_px <= 0.0
        {
            return Err("f_scale and reject_px must be positive".to_string());
        }
        Ok(SurveyOptions {
            swatch_family: self.dictionary.clone(),
            max_swatch_id: self.max_swatch_id,
            min_card_corners: self.min_card.max(4) as usize,
            soft_l1_px: self.f_scale,
            reject_px: self.reject_px,
            rounds: self.rounds.max(1) as usize,
            ..SurveyOptions::default()
        })
    }
}

/// Surveys the set in `set_bytes`: the posed set and the report.
pub fn run_survey(
    set_bytes: &[u8],
    config: &SurveyConfig,
) -> Result<(ViewSet, SurveyReport), String> {
    let mut set = decode_viewset(set_bytes)?;
    let options = config.options()?;
    let report = survey(&mut set, &options)?;
    Ok((set, report))
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let set = read_input(0);
    let config = {
        let cfg = read_input(1);
        if cfg.is_empty() {
            SurveyConfig::default()
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
        report_error("no view set: wire a detected set");
        return;
    }
    match run_survey(&set, &config) {
        Ok((set, report)) => {
            for line in &report.log {
                post_warning(line);
            }
            for c in &report.cameras {
                post_warning(&format!(
                    "camera {} '{}': {} frames, f {:.1} px (± {:.1}), pp ({:.1} ± {:.1}, {:.1} ± {:.1}), k1 {:+.5} k2 {:+.4}",
                    c.index,
                    c.label,
                    c.frames,
                    c.f,
                    c.f_std,
                    c.cx,
                    c.cx_std,
                    c.cy,
                    c.cy_std,
                    c.k1,
                    c.k2
                ));
            }
            if !report.rejected.is_empty() {
                post_warning(&format!("rejected: {}", report.rejected.join(" ")));
            }
            post_output(0, &volumetric_abi::viewset::encode_viewset(&set));
            match volumetric_abi::f64_map::encode(&report.to_f64_map()) {
                Ok(bytes) => post_output(1, &bytes),
                Err(e) => report_error(&format!("report encoding failed: {e}")),
            }
        }
        Err(e) => report_error(&format!("survey failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ dictionary: "5x5_100" / "4x4_50" .default "5x5_100", max_swatch_id: int .ge 0 .default 60, min_card: int .ge 4 .default 8, f_scale: float .ge 0.1 .default 1.5, reject_px: float .ge 0.1 .default 3.0, rounds: int .ge 1 .default 4 }"#
            .to_string();
        OperatorMetadata {
            name: "survey_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Survey Field".to_string(),
            description: "Solve a session's cameras, poses and marker field from the cards detected in its stills; the card's calipers set the scale and its plane the world frame.".to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M4 20L12 4l8 16"/>"##,
                r##"<path d="M7 14h10"/>"##,
                r##"<circle cx="12" cy="20" r="1.5"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ViewSet,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["Views".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ViewSet, OperatorMetadataOutput::F64Map],
            output_names: vec!["Views".to_string(), "Report".to_string()],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_maps_onto_the_options_and_is_checked() {
        let options = SurveyConfig::default().options().unwrap();
        assert_eq!(options.swatch_family, "5x5_100");
        assert_eq!(options.reject_px, 3.0);
        let loose = SurveyConfig {
            reject_px: 5.0,
            f_scale: 2.5,
            min_card: 6,
            rounds: 0,
            ..SurveyConfig::default()
        }
        .options()
        .unwrap();
        assert_eq!(
            (
                loose.reject_px,
                loose.soft_l1_px,
                loose.min_card_corners,
                loose.rounds
            ),
            (5.0, 2.5, 6, 1)
        );
        assert!(
            SurveyConfig {
                f_scale: 0.0,
                ..SurveyConfig::default()
            }
            .options()
            .is_err()
        );
        // A set without a board is refused with a pointer to Detect Cards.
        let set = volumetric_abi::viewset::ViewSet::default();
        let err = run_survey(
            &volumetric_abi::viewset::encode_viewset(&set),
            &SurveyConfig::default(),
        )
        .unwrap_err();
        assert!(err.contains("view-detect"), "{err}");
    }
}
