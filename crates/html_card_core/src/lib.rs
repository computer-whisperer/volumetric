//! HTML+Tailwind card rendering: an HTML fragment styled with Tailwind
//! utility classes becomes monochrome ink geometry — closed contours for
//! `outline_model_core` — plus the card's plate outline, for fabrication
//! (extrude both and union for embossed labels, subtract the ink for
//! engraving).
//!
//! The pipeline is [`parse`] (strict subset parser) → [`style`] (Tailwind
//! class table) → [`render`] (taffy flexbox/grid/block layout with real
//! text measurement, then an ink-parity paint walk). Everything the
//! subset doesn't cover errors loudly, naming each offending tag or class
//! — the operator's contract is that markup either renders as specified
//! or converges in one authoring round-trip.
//!
//! Embedded faces: Inter Regular and Inter Bold (SIL OFL 1.1, see
//! `fonts/LICENSE`); `font-bold`/`font-semibold` and `<b>`/`<strong>`
//! select the bold face.

pub mod parse;
pub mod render;
pub mod style;
pub mod text;

use text_render_core::ttf_parser::Face;

pub use render::{CardGeometry, Contour, to_model_space};

/// Inter Regular (SIL OFL 1.1, `fonts/LICENSE`).
pub const FONT_REGULAR: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");
/// Inter Bold (SIL OFL 1.1, `fonts/LICENSE`).
pub const FONT_BOLD: &[u8] = include_bytes!("../fonts/Inter-Bold.ttf");

struct EmbeddedFonts {
    regular: Face<'static>,
    bold: Face<'static>,
}

impl text::FaceSource for EmbeddedFonts {
    fn face(&self, bold: bool) -> &Face<'_> {
        if bold { &self.bold } else { &self.regular }
    }
}

/// Render an HTML+Tailwind fragment as card geometry at `width_px` CSS
/// pixels wide (the height follows from the content). All markup errors
/// come back as one newline-separated report.
pub fn render_card(html: &str, width_px: f64) -> Result<CardGeometry, String> {
    if !(width_px > 0.0 && width_px.is_finite()) {
        return Err(format!("width_px must be > 0, got {width_px}"));
    }
    let root = parse::parse_fragment(html)?;
    let fonts = EmbeddedFonts {
        regular: Face::parse(FONT_REGULAR, 0).expect("embedded Inter Regular parses"),
        bold: Face::parse(FONT_BOLD, 0).expect("embedded Inter Bold parses"),
    };
    render::render(&root, width_px, &fonts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_card_renders_end_to_end() {
        let html = r#"
            <div class="p-6 border-2 rounded-xl">
              <div class="flex justify-between items-center">
                <p class="text-2xl font-bold uppercase">Part A-113</p>
                <div class="bg-black rounded px-2 py-1"><p class="text-white text-sm">Rev 4</p></div>
              </div>
              <hr class="border-t-2"/>
              <div class="grid grid-cols-2 gap-2">
                <p class="text-sm">Material</p>
                <p class="text-sm text-right font-bold">PETG</p>
                <p class="text-sm">Torque</p>
                <p class="text-sm text-right font-bold">2.4 Nm</p>
              </div>
            </div>"#;
        let g = render_card(html, 384.0).expect("card renders");
        assert_eq!(g.width_px, 384.0);
        assert!(g.height_px > 80.0, "height {}", g.height_px);
        assert!(g.ink.len() > 30, "contour count {}", g.ink.len());
        assert_eq!(g.plate.len(), 1);
    }

    #[test]
    fn all_errors_report_in_one_pass() {
        let err = render_card(
            r#"<div class="blorp md:flex"><p>ok \u{2603}</p><video></video></div>"#,
            384.0,
        )
        .unwrap_err();
        assert!(err.contains("video"), "{err}");
        // Parse errors preempt style errors (the DOM may be nonsense), but
        // each stage reports all its findings at once.
        let err = render_card(r#"<div class="blorp md:flex"><p>ok</p></div>"#, 384.0).unwrap_err();
        assert!(err.contains("blorp") && err.contains("md:"), "{err}");
    }
}
