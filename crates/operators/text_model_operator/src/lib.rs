//! Text Model Operator.
//!
//! Renders a string as a filled 2D outline model: `sample(x, y)` is 1.0
//! inside the glyph outlines and 0.0 outside, at any resolution — glyph
//! Béziers are flattened here at conversion time with a size-proportional
//! chord tolerance, and the generated model classifies points exactly
//! against those contours (nonzero winding, the TrueType fill rule). The
//! output composes like any 2D sketch: extrude it into a plaque, subtract
//! it for engraving, or place it with the transform operators.
//!
//! All font work happens here at conversion time: the laid-out contours are
//! baked into an `outline_model_core` payload and patched into the
//! prebuilt, stateless `outline_model_template` module as data segments —
//! the same pattern as `image_model_operator`, with no hand-maintained
//! codegen offsets.
//!
//! Layout is deliberately simple v1 typesetting: lines split on `\n`,
//! horizontal advances plus `kern`-table pair kerning (no full shaping —
//! ligatures and complex scripts are out of scope), `align` for multi-line
//! justification, and `anchor` choosing whether the model is centered on
//! the origin or hangs from the first line's baseline. Characters the font
//! has no glyph for are an error, not tofu.
//!
//! Inputs:
//! - Input 0: CBOR config `{ text, size, align, anchor, line_height,
//!   letter_spacing }` — `size` is the em size in model units; `line_height`
//!   and `letter_spacing` are in em units.
//! - Input 1: Blob (optional) — a TTF/OTF font file replacing the embedded
//!   Liberation Sans Regular (see `fonts/LICENSE`, SIL OFL 1.1).
//!
//! Output 0: ModelWASM (2D).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p outline_model_template
//! cp target/wasm32-unknown-unknown/release/outline_model_template.wasm \
//!    crates/operators/text_model_operator/template/
//! ```

use std::collections::BTreeSet;

use text_render_core::ttf_parser::{Face, GlyphId};
use text_render_core::{outline_glyph_into, pair_kerning, run_width};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/outline_model_template.wasm");

/// The embedded fallback font (SIL OFL 1.1, see `fonts/LICENSE`).
const DEFAULT_FONT: &[u8] = include_bytes!("../fonts/LiberationSans-Regular.ttf");

/// Chord tolerance for Bézier flattening, as a fraction of the em size:
/// 0.2% of the em (10 µm on 5 mm text), well under any print resolution.
const CHORD_TOL_EM: f64 = 1.0 / 512.0;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct TextConfig {
    /// The text to render; `\n` starts a new line.
    text: String,
    /// Em size in model units (glyph capitals come out around 0.7 of this).
    size: f64,
    /// Multi-line justification: "left", "center", or "right".
    align: String,
    /// "center" places the tight bounding box on the origin; "baseline"
    /// puts the first line's baseline at y=0 with the `align` anchor at x=0.
    anchor: String,
    /// Baseline-to-baseline distance in em units.
    line_height: f64,
    /// Extra tracking between adjacent glyphs in em units.
    letter_spacing: f64,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            text: "Text".to_string(),
            size: 1.0,
            align: "center".to_string(),
            anchor: "center".to_string(),
            line_height: 1.2,
            letter_spacing: 0.0,
        }
    }
}

/// Lay the text out and return the flattened contours in model space.
fn text_contours(cfg: &TextConfig, font_bytes: &[u8]) -> Result<Vec<Vec<[f64; 2]>>, String> {
    if !(cfg.size > 0.0 && cfg.size.is_finite()) {
        return Err(format!("size must be > 0, got {}", cfg.size));
    }
    if !(cfg.line_height > 0.0 && cfg.line_height.is_finite()) {
        return Err(format!("line_height must be > 0, got {}", cfg.line_height));
    }
    if !cfg.letter_spacing.is_finite() {
        return Err(format!(
            "letter_spacing must be finite, got {}",
            cfg.letter_spacing
        ));
    }
    if !matches!(cfg.align.as_str(), "left" | "center" | "right") {
        return Err(format!(
            "align must be \"left\", \"center\", or \"right\", got {:?}",
            cfg.align
        ));
    }
    if !matches!(cfg.anchor.as_str(), "center" | "baseline") {
        return Err(format!(
            "anchor must be \"center\" or \"baseline\", got {:?}",
            cfg.anchor
        ));
    }

    let face = Face::parse(font_bytes, 0).map_err(|e| format!("font parse: {e}"))?;
    let upem = face.units_per_em() as f64;
    if upem <= 0.0 {
        return Err("font declares zero units per em".to_string());
    }
    let scale = cfg.size / upem;
    let letter_spacing = cfg.letter_spacing * upem;
    let line_advance = cfg.line_height * upem;

    // Resolve every character to a glyph up front so unsupported characters
    // fail as one complete report instead of one at a time.
    let text = cfg.text.replace("\r\n", "\n").replace('\r', "\n");
    let mut missing = BTreeSet::new();
    let lines: Vec<Vec<GlyphId>> = text
        .split('\n')
        .map(|line| {
            line.chars()
                .filter_map(|c| {
                    let glyph = face.glyph_index(c);
                    if glyph.is_none() {
                        missing.insert(c);
                    }
                    glyph
                })
                .collect()
        })
        .collect();
    if !missing.is_empty() {
        let listed: Vec<String> = missing.iter().map(|c| format!("{c:?}")).collect();
        return Err(format!(
            "font has no glyph for {} — supply a font with coverage on the Font input",
            listed.join(", ")
        ));
    }

    let mut contours: Vec<Vec<[f64; 2]>> = Vec::new();
    for (line_idx, glyphs) in lines.iter().enumerate() {
        // Measure, then place: `align` needs the line width first.
        let width = run_width(&face, glyphs, letter_spacing);
        let line_x = match cfg.align.as_str() {
            "left" => 0.0,
            "center" => -width / 2.0,
            _ => -width,
        };

        let baseline_y = -(line_idx as f64) * line_advance;
        let mut pen_x = line_x;
        for (i, &glyph) in glyphs.iter().enumerate() {
            if i > 0 {
                pen_x += letter_spacing + pair_kerning(&face, glyphs[i - 1], glyph);
            }
            outline_glyph_into(
                &face,
                glyph,
                [pen_x, baseline_y],
                scale,
                CHORD_TOL_EM * upem,
                &mut contours,
            );
            pen_x += text_render_core::advance(&face, glyph);
        }
    }
    if contours.is_empty() {
        return Err("text renders no geometry (no visible glyph outlines)".to_string());
    }

    if cfg.anchor == "center" {
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in contours.iter().flatten() {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        let (cx, cy) = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
        for p in contours.iter_mut().flatten() {
            p[0] -= cx;
            p[1] -= cy;
        }
    }
    Ok(contours)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg = {
        let cfg_buf = read_input(0);
        if cfg_buf.is_empty() {
            TextConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<TextConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    let font_blob = read_input(1);
    let font_bytes: &[u8] = if font_blob.is_empty() {
        DEFAULT_FONT
    } else {
        &font_blob
    };

    let result = text_contours(&cfg, font_bytes)
        .and_then(|contours| outline_model_core::build_payload(&contours))
        .and_then(|payload| outline_model_core::emit::patch_template(TEMPLATE, &payload));
    match result {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("text model generation failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ text: tstr .default \"Text\", size: float .default 1.0, \
                       align: \"left\" / \"center\" / \"right\" .default \"center\", \
                       anchor: \"center\" / \"baseline\" .default \"center\", \
                       line_height: float .default 1.2, \
                       letter_spacing: float .default 0.0 }"
            .to_string();
        OperatorMetadata {
            name: "text_model_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: String::new(),
            display_name: "Text".to_string(),
            description: "Render text as a filled 2D outline model, ready to extrude.".to_string(),
            category: "Primitives".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<polyline points="4 7 4 4 20 4 20 7"/>"##,
                r##"<line x1="9" x2="15" y1="20" y2="20"/>"##,
                r##"<line x1="12" x2="12" y1="4" y2="20"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::CBORConfiguration(schema),
                OperatorMetadataInput::Blob,
            ],
            input_names: vec!["Config".to_string(), "Font (TTF, optional)".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
            output_names: vec![],
        }
    })
}
