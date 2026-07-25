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

pub use ttf_parser;
use ttf_parser::{Face, GlyphId, OutlineBuilder};

/// Collects one glyph's outline as flattened contours in target space.
///
/// Flattening decisions run in font units (`tol2` is a squared font-unit
/// tolerance); emitted points are `(p + offset) * scale`.
struct GlyphSink<'a> {
    offset: [f64; 2],
    scale: f64,
    tol2: f64,
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

    fn flatten_quad(&mut self, p0: [f64; 2], p1: [f64; 2], p2: [f64; 2], depth: u32) {
        // Max deviation of a quadratic from its chord is |p1 - mid(p0,p2)|/2.
        let dx = p1[0] - (p0[0] + p2[0]) * 0.5;
        let dy = p1[1] - (p0[1] + p2[1]) * 0.5;
        if depth >= 16 || (dx * dx + dy * dy) * 0.25 <= self.tol2 {
            self.push(p2);
            return;
        }
        let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let (a, b) = (mid(p0, p1), mid(p1, p2));
        let m = mid(a, b);
        self.flatten_quad(p0, a, m, depth + 1);
        self.flatten_quad(m, b, p2, depth + 1);
    }

    fn flatten_cubic(&mut self, p0: [f64; 2], p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], depth: u32) {
        // Standard cubic flatness bound: deviation² <= (max(d1²)+max(d2²))/16
        // with d1 = 3p1 - 2p0 - p3, d2 = 3p2 - p0 - 2p3 (per component).
        let d1x = 3.0 * p1[0] - 2.0 * p0[0] - p3[0];
        let d1y = 3.0 * p1[1] - 2.0 * p0[1] - p3[1];
        let d2x = 3.0 * p2[0] - p0[0] - 2.0 * p3[0];
        let d2y = 3.0 * p2[1] - p0[1] - 2.0 * p3[1];
        let dev2 = (d1x * d1x).max(d2x * d2x) + (d1y * d1y).max(d2y * d2y);
        if depth >= 16 || dev2 <= 16.0 * self.tol2 {
            self.push(p3);
            return;
        }
        let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let (a, b, c) = (mid(p0, p1), mid(p1, p2), mid(p2, p3));
        let (ab, bc) = (mid(a, b), mid(b, c));
        let m = mid(ab, bc);
        self.flatten_cubic(p0, a, ab, m, depth + 1);
        self.flatten_cubic(m, bc, c, p3, depth + 1);
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
        self.flatten_quad(p0, p1, p2, 0);
        self.cursor = p2;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p0 = self.cursor;
        let p3 = [x as f64, y as f64];
        self.flatten_cubic(p0, [x1 as f64, y1 as f64], [x2 as f64, y2 as f64], p3, 0);
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
        tol2: tol_fu * tol_fu,
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
