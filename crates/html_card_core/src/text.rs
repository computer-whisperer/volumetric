//! Paragraph shaping: inline runs → transformed, glyph-shaped words with
//! px widths, then greedy line breaking. Shared by taffy's measure pass
//! and the paint pass so both always agree on the wrap.
//!
//! HTML whitespace collapses (any run of spaces/tabs/newlines is one
//! space, trimmed at paragraph edges); `&nbsp;` glues its neighbours into
//! one unbreakable word; `<br>` forces a line break.

use crate::style::{TextAlign, TextStyle, TextTransform};
use text_render_core::ttf_parser::{Face, GlyphId};
use text_render_core::{advance, pair_kerning};

/// A shaped word: glyphs plus the px width it occupies.
#[derive(Debug, Clone)]
pub struct Word {
    pub glyphs: Vec<GlyphId>,
    pub style: TextStyle,
    pub width_px: f64,
    /// Width of one space in this word's style (used when joining).
    pub space_px: f64,
    /// Continues the previous word with no intervening space — an inline
    /// run boundary fell mid-word (`<b>Hel</b>lo`). Never a wrap point
    /// and never separated by a space.
    pub joins_prev: bool,
}

#[derive(Debug, Clone)]
pub enum Token {
    Word(Word),
    /// Forced break from `<br>`.
    Break,
}

#[derive(Debug, Clone)]
pub struct Paragraph {
    pub tokens: Vec<Token>,
    pub align: TextAlign,
    /// Line box height: the max over the runs in the paragraph.
    pub line_height_px: f64,
    /// Max ascent/descent in px over the runs (baseline placement).
    pub ascent_px: f64,
    pub descent_px: f64,
}

/// One inline run: text in a single resolved style.
pub struct Run<'a> {
    pub text: &'a str,
    pub style: TextStyle,
}

pub struct FontMetrics {
    pub upem: f64,
    pub ascent_fu: f64,
    pub descent_fu: f64,
}

pub fn font_metrics(face: &Face) -> FontMetrics {
    FontMetrics {
        upem: face.units_per_em() as f64,
        ascent_fu: face.ascender() as f64,
        // ttf descender is negative-down; keep magnitude.
        descent_fu: -(face.descender() as f64),
    }
}

/// Face picker: the style's weight chooses regular or bold.
pub trait FaceSource {
    fn face(&self, bold: bool) -> &Face<'_>;
}

/// Shape a paragraph's runs into tokens. Missing glyphs are collected
/// into `missing` (deduped by the caller); shaping continues so one pass
/// reports every unsupported character.
pub fn shape_paragraph(
    runs: &[Run<'_>],
    breaks_after_run: &[bool],
    align: TextAlign,
    faces: &dyn FaceSource,
    missing: &mut Vec<char>,
) -> Paragraph {
    let mut tokens: Vec<Token> = Vec::new();
    let mut line_height_px = 0.0f64;
    let mut ascent_px = 0.0f64;
    let mut descent_px = 0.0f64;

    // Set when the previous run ended mid-word: the next fragment glues
    // onto it with no space and no wrap opportunity (`<b>Hel</b>lo`).
    let mut pending_join = false;

    for (run_idx, run) in runs.iter().enumerate() {
        let face = faces.face(run.style.bold);
        let metrics = font_metrics(face);
        let scale = run.style.font_px / metrics.upem;
        line_height_px = line_height_px.max(run.style.line_height_px);
        ascent_px = ascent_px.max(metrics.ascent_fu * scale);
        descent_px = descent_px.max(metrics.descent_fu * scale);

        let transformed = transform_text(run.text, run.style.transform);
        // Split on collapsible whitespace; nbsp stays inside words.
        let mut chars = transformed.chars().peekable();
        let mut current = String::new();
        let mut flush = |current: &mut String, tokens: &mut Vec<Token>, pending_join: &mut bool| {
            if current.is_empty() {
                return;
            }
            let word = shape_word(current, run.style, face, &metrics, missing, *pending_join);
            *pending_join = false;
            tokens.push(Token::Word(word));
            current.clear();
        };
        while let Some(c) = chars.next() {
            if c.is_whitespace() && c != '\u{a0}' {
                flush(&mut current, &mut tokens, &mut pending_join);
                // Explicit whitespace cancels any pending cross-run glue.
                pending_join = false;
                while chars
                    .peek()
                    .is_some_and(|c| c.is_whitespace() && *c != '\u{a0}')
                {
                    chars.next();
                }
            } else {
                current.push(c);
            }
        }
        if !current.is_empty() {
            flush(&mut current, &mut tokens, &mut pending_join);
            // The run ended mid-word: the next run's first fragment glues.
            pending_join = true;
        }
        if breaks_after_run.get(run_idx).copied().unwrap_or(false) {
            tokens.push(Token::Break);
            pending_join = false;
        }
    }

    if line_height_px == 0.0 {
        line_height_px = 24.0;
    }
    Paragraph {
        tokens,
        align,
        line_height_px,
        ascent_px,
        descent_px,
    }
}

fn transform_text(text: &str, transform: TextTransform) -> String {
    match transform {
        TextTransform::None => text.to_string(),
        TextTransform::Upper => text.to_uppercase(),
        TextTransform::Lower => text.to_lowercase(),
        TextTransform::Capitalize => {
            let mut out = String::with_capacity(text.len());
            let mut at_word_start = true;
            for c in text.chars() {
                if c.is_whitespace() {
                    at_word_start = true;
                    out.push(c);
                } else if at_word_start {
                    out.extend(c.to_uppercase());
                    at_word_start = false;
                } else {
                    out.push(c);
                }
            }
            out
        }
    }
}

fn shape_word(
    text: &str,
    style: TextStyle,
    face: &Face,
    metrics: &FontMetrics,
    missing: &mut Vec<char>,
    joins_prev: bool,
) -> Word {
    let mut glyphs = Vec::with_capacity(text.chars().count());
    for c in text.chars() {
        // nbsp renders as a space glyph inside the word.
        let lookup = if c == '\u{a0}' { ' ' } else { c };
        match face.glyph_index(lookup) {
            Some(g) => glyphs.push(g),
            None => missing.push(c),
        }
    }
    let scale = style.font_px / metrics.upem;
    let tracking_fu = style.tracking_em * metrics.upem;
    let width_px = text_render_core::run_width(face, &glyphs, tracking_fu) * scale;
    let space_px = face
        .glyph_index(' ')
        .map(|g| advance(face, g) * scale)
        .unwrap_or(style.font_px * 0.25);
    Word {
        glyphs,
        style,
        width_px,
        space_px,
        joins_prev,
    }
}

/// A broken line: the word indices it holds and its total width.
pub struct Line {
    /// (token index, x offset from the line start in px).
    pub words: Vec<(usize, f64)>,
    pub width_px: f64,
}

/// Greedy word wrap at `avail_px` (`f64::INFINITY` for no wrapping).
/// The first word of a line always fits, however wide; glued chains
/// (`joins_prev`) wrap as one unit.
pub fn break_lines(paragraph: &Paragraph, avail_px: f64) -> Vec<Line> {
    let tokens = &paragraph.tokens;
    let mut lines: Vec<Line> = Vec::new();
    let mut current = Line {
        words: Vec::new(),
        width_px: 0.0,
    };
    for (idx, token) in tokens.iter().enumerate() {
        match token {
            Token::Break => {
                lines.push(std::mem::replace(
                    &mut current,
                    Line {
                        words: Vec::new(),
                        width_px: 0.0,
                    },
                ));
            }
            Token::Word(word) => {
                if word.joins_prev && !current.words.is_empty() {
                    // Mid-word continuation: same line, no space.
                    current.words.push((idx, current.width_px));
                    current.width_px += word.width_px;
                    continue;
                }
                // Wrap decision considers the whole glued chain.
                let mut chain = word.width_px;
                let mut j = idx + 1;
                while let Some(Token::Word(next)) = tokens.get(j) {
                    if !next.joins_prev {
                        break;
                    }
                    chain += next.width_px;
                    j += 1;
                }
                let space = if current.words.is_empty() {
                    0.0
                } else {
                    word.space_px
                };
                if !current.words.is_empty() && current.width_px + space + chain > avail_px + 1e-6 {
                    lines.push(std::mem::replace(
                        &mut current,
                        Line {
                            words: Vec::new(),
                            width_px: 0.0,
                        },
                    ));
                    current.words.push((idx, 0.0));
                    current.width_px = word.width_px;
                } else {
                    current.words.push((idx, current.width_px + space));
                    current.width_px += space + word.width_px;
                }
            }
        }
    }
    if !current.words.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

/// Measurement summary for taffy: (max line width, total height).
pub fn measure(paragraph: &Paragraph, avail_px: f64) -> (f64, f64) {
    let lines = break_lines(paragraph, avail_px);
    let width = lines.iter().fold(0.0f64, |w, l| w.max(l.width_px));
    (width, lines.len() as f64 * paragraph.line_height_px)
}

/// Width of the longest unbreakable unit — a word or glued chain
/// (min-content).
pub fn min_content_width(paragraph: &Paragraph) -> f64 {
    let mut longest = 0.0f64;
    let mut chain = 0.0f64;
    for token in &paragraph.tokens {
        match token {
            Token::Word(w) if w.joins_prev && chain > 0.0 => chain += w.width_px,
            Token::Word(w) => {
                longest = longest.max(chain);
                chain = w.width_px;
            }
            Token::Break => {
                longest = longest.max(chain);
                chain = 0.0;
            }
        }
    }
    longest.max(chain)
}

/// Per-glyph positions of one word laid at `(pen_x, baseline_y)` (px,
/// y-down): returns (glyph, x, scale, face-bold) tuples via a callback to
/// avoid materializing.
pub fn place_word(
    word: &Word,
    faces: &dyn FaceSource,
    pen_x: f64,
    mut emit: impl FnMut(GlyphId, f64, f64),
) {
    let face = faces.face(word.style.bold);
    let metrics = font_metrics(face);
    let scale = word.style.font_px / metrics.upem;
    let tracking_fu = word.style.tracking_em * metrics.upem;
    let mut x_fu = 0.0;
    for (i, &glyph) in word.glyphs.iter().enumerate() {
        if i > 0 {
            x_fu += tracking_fu + pair_kerning(face, word.glyphs[i - 1], glyph);
        }
        emit(glyph, pen_x + x_fu * scale, scale);
        x_fu += advance(face, glyph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGULAR: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");
    const BOLD: &[u8] = include_bytes!("../fonts/Inter-Bold.ttf");

    struct Fonts {
        regular: Face<'static>,
        bold: Face<'static>,
    }
    impl FaceSource for Fonts {
        fn face(&self, bold: bool) -> &Face<'_> {
            if bold { &self.bold } else { &self.regular }
        }
    }
    fn fonts() -> Fonts {
        Fonts {
            regular: Face::parse(REGULAR, 0).unwrap(),
            bold: Face::parse(BOLD, 0).unwrap(),
        }
    }

    fn para(text: &str) -> Paragraph {
        let mut missing = Vec::new();
        let runs = [Run {
            text,
            style: TextStyle::default(),
        }];
        let p = shape_paragraph(&runs, &[false], TextAlign::Left, &fonts(), &mut missing);
        assert!(missing.is_empty(), "missing glyphs: {missing:?}");
        p
    }

    #[test]
    fn whitespace_collapses_and_words_shape() {
        let p = para("  hello   world \n test ");
        let words: Vec<usize> = p
            .tokens
            .iter()
            .filter_map(|t| match t {
                Token::Word(w) => Some(w.glyphs.len()),
                Token::Break => None,
            })
            .collect();
        assert_eq!(words, [5, 5, 4]);
    }

    #[test]
    fn wrapping_is_greedy_and_min_content_is_longest_word() {
        let p = para("aa aa aa aa");
        let word_w = match &p.tokens[0] {
            Token::Word(w) => w.width_px,
            _ => unreachable!(),
        };
        let space = match &p.tokens[1] {
            Token::Word(w) => w.space_px,
            _ => unreachable!(),
        };
        // Room for exactly two words per line.
        let lines = break_lines(&p, 2.0 * word_w + space + 0.5);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].words.len(), 2);
        assert!((min_content_width(&p) - word_w).abs() < 0.5);

        // Unbounded: one line.
        let (w, h) = measure(&p, f64::INFINITY);
        assert_eq!(break_lines(&p, f64::INFINITY).len(), 1);
        assert!((w - (4.0 * word_w + 3.0 * space)).abs() < 0.5);
        assert_eq!(h, p.line_height_px);
    }

    #[test]
    fn nbsp_glues_into_one_word() {
        let p = para("a\u{a0}b c");
        let sizes: Vec<usize> = p
            .tokens
            .iter()
            .filter_map(|t| match t {
                Token::Word(w) => Some(w.glyphs.len()),
                Token::Break => None,
            })
            .collect();
        assert_eq!(sizes, [3, 1]); // "a␣b" is one 3-glyph word
    }

    #[test]
    fn breaks_and_transforms() {
        let mut missing = Vec::new();
        let runs = [
            Run {
                text: "up",
                style: TextStyle {
                    transform: TextTransform::Upper,
                    ..TextStyle::default()
                },
            },
            Run {
                text: "low",
                style: TextStyle::default(),
            },
        ];
        let p = shape_paragraph(
            &runs,
            &[true, false],
            TextAlign::Left,
            &fonts(),
            &mut missing,
        );
        assert!(matches!(p.tokens[1], Token::Break));
        // Uppercase "UP" shaped: U and P glyphs differ from lowercase.
        let f = fonts();
        let up = match &p.tokens[0] {
            Token::Word(w) => w.glyphs.clone(),
            _ => unreachable!(),
        };
        assert_eq!(up[0], f.regular.glyph_index('U').unwrap());
    }

    #[test]
    fn adjacent_runs_glue_without_a_space() {
        // <p><b>Hel</b>lo</p>: the run boundary falls mid-word.
        let mut missing = Vec::new();
        let bold = TextStyle {
            bold: true,
            ..TextStyle::default()
        };
        let p = shape_paragraph(
            &[
                Run {
                    text: "Hel",
                    style: bold,
                },
                Run {
                    text: "lo world",
                    style: TextStyle::default(),
                },
            ],
            &[false, false],
            TextAlign::Left,
            &fonts(),
            &mut missing,
        );
        let joins: Vec<bool> = p
            .tokens
            .iter()
            .filter_map(|t| match t {
                Token::Word(w) => Some(w.joins_prev),
                Token::Break => None,
            })
            .collect();
        assert_eq!(joins, [false, true, false]);

        // The glued pair measures as fragment widths with no space, and
        // wraps as one unit: at a width fitting only the chain, "world"
        // drops to line 2 and "Hel"+"lo" stay together.
        let widths: Vec<f64> = p
            .tokens
            .iter()
            .filter_map(|t| match t {
                Token::Word(w) => Some(w.width_px),
                Token::Break => None,
            })
            .collect();
        let chain = widths[0] + widths[1];
        let (w, _) = measure(&p, chain + 1.0);
        // Two lines: the glued chain, then "world" (whichever is wider).
        assert!((w - chain.max(widths[2])).abs() < 1e-9, "measured {w}");
        let lines = break_lines(&p, chain + 1.0);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].words.len(), 2);
        // Second fragment sits flush against the first (no space).
        assert!((lines[0].words[1].1 - widths[0]).abs() < 1e-9);
        // Min-content counts the glued chain as one unbreakable unit.
        assert!((min_content_width(&p) - chain.max(widths[2])).abs() < 1e-9);
    }

    #[test]
    fn missing_glyphs_collect() {
        let mut missing = Vec::new();
        let runs = [Run {
            text: "a\u{2603}b",
            style: TextStyle::default(),
        }];
        shape_paragraph(&runs, &[false], TextAlign::Left, &fonts(), &mut missing);
        assert_eq!(missing, ['\u{2603}']);
    }

    #[test]
    fn bold_face_is_wider() {
        let mut missing = Vec::new();
        let bold_style = TextStyle {
            bold: true,
            ..TextStyle::default()
        };
        let p = shape_paragraph(
            &[
                Run {
                    text: "Weight",
                    style: TextStyle::default(),
                },
                Run {
                    text: "Weight",
                    style: bold_style,
                },
            ],
            &[false, false],
            TextAlign::Left,
            &fonts(),
            &mut missing,
        );
        let widths: Vec<f64> = p
            .tokens
            .iter()
            .filter_map(|t| match t {
                Token::Word(w) => Some(w.width_px),
                Token::Break => None,
            })
            .collect();
        assert!(widths[1] > widths[0] + 0.5, "{widths:?}");
    }
}
