//! Shared glyph-to-contour machinery for text-rendering operators
//! (`text_model_operator`, `html_card_operator`): flattening a glyph's
//! Bézier outline into closed polyline contours ready for
//! `outline_model_core`, plus the advance/kerning arithmetic layouts sum.
//!
//! Everything works in *font units* for decisions (kerning, advances,
//! flattening tolerance) and emits points transformed by a caller-supplied
//! offset (font units) and scale (target units per font unit) — layouts
//! position glyphs in font-unit space and pick the output scale once.
//!
//! Re-exports `ttf_parser` so every consumer shapes with the same version.

use outline_model_core::flatten;
pub use ttf_parser;
use ttf_parser::{Face, GlyphId, OutlineBuilder};

/// Collects one glyph's outline as flattened contours in target space.
///
/// Flattening decisions run in font units (`tol` is a font-unit chord
/// tolerance); emitted points are `(p + offset) * scale`.
struct GlyphSink<'a> {
    offset: [f64; 2],
    scale: f64,
    tol: f64,
    cursor: [f64; 2],
    current: Vec<[f64; 2]>,
    contours: &'a mut Vec<Vec<[f64; 2]>>,
}

impl GlyphSink<'_> {
    fn push(&mut self, p: [f64; 2]) {
        self.current.push([
            (p[0] + self.offset[0]) * self.scale,
            (p[1] + self.offset[1]) * self.scale,
        ]);
    }

    fn flush(&mut self) {
        // Degenerate contours (too few distinct points to enclose area)
        // are dropped rather than failing the whole conversion — some
        // fonts do contain stray anchor-only contours.
        let mut distinct = self.current.clone();
        distinct.dedup();
        if distinct.first() == distinct.last() {
            distinct.pop();
        }
        if distinct.len() >= 3 {
            self.contours.push(std::mem::take(&mut self.current));
        } else {
            self.current.clear();
        }
    }
}

impl OutlineBuilder for GlyphSink<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.flush();
        self.cursor = [x as f64, y as f64];
        self.push(self.cursor);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.cursor = [x as f64, y as f64];
        self.push(self.cursor);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p0 = self.cursor;
        let p1 = [x1 as f64, y1 as f64];
        let p2 = [x as f64, y as f64];
        let mut points = Vec::new();
        flatten::quad(p0, p1, p2, self.tol, &mut points);
        for p in points {
            self.push(p);
        }
        self.cursor = p2;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p0 = self.cursor;
        let p3 = [x as f64, y as f64];
        let mut points = Vec::new();
        flatten::cubic(
            p0,
            [x1 as f64, y1 as f64],
            [x2 as f64, y2 as f64],
            p3,
            self.tol,
            &mut points,
        );
        for p in points {
            self.push(p);
        }
        self.cursor = p3;
    }

    fn close(&mut self) {
        self.flush();
    }
}

/// Flatten `glyph`'s outline into `contours`: points are
/// `(outline_point + offset) * scale`, with `offset` in font units and
/// `tol_fu` the chord tolerance in font units. Glyphs without an outline
/// (spaces) append nothing.
pub fn outline_glyph_into(
    face: &Face,
    glyph: GlyphId,
    offset: [f64; 2],
    scale: f64,
    tol_fu: f64,
    contours: &mut Vec<Vec<[f64; 2]>>,
) {
    let mut sink = GlyphSink {
        offset,
        scale,
        tol: tol_fu,
        cursor: [0.0, 0.0],
        current: Vec::new(),
        contours,
    };
    face.outline_glyph(glyph, &mut sink);
    sink.flush();
}

/// Horizontal advance of `glyph` in font units (0 when the font declares
/// none).
pub fn advance(face: &Face, glyph: GlyphId) -> f64 {
    face.glyph_hor_advance(glyph).unwrap_or(0) as f64
}

/// Horizontal `kern`-table adjustment between two glyphs, in font units.
pub fn pair_kerning(face: &Face, left: GlyphId, right: GlyphId) -> f64 {
    let Some(kern) = face.tables().kern else {
        return 0.0;
    };
    for subtable in kern.subtables {
        if subtable.horizontal && !subtable.variable {
            if let Some(v) = subtable.glyphs_kerning(left, right) {
                return v as f64;
            }
        }
    }
    0.0
}

/// Width of a glyph run in font units: advances plus kerning plus
/// `letter_spacing` (font units) between adjacent glyphs.
pub fn run_width(face: &Face, glyphs: &[GlyphId], letter_spacing: f64) -> f64 {
    let mut width = 0.0;
    for (i, &glyph) in glyphs.iter().enumerate() {
        if i > 0 {
            width += letter_spacing + pair_kerning(face, glyphs[i - 1], glyph);
        }
        width += advance(face, glyph);
    }
    width
}
