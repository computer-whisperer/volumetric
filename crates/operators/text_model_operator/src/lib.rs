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

use ttf_parser::{Face, GlyphId, OutlineBuilder};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
use walrus::{FunctionId, Module, ModuleConfig};

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

/// Collects one glyph's outline as flattened contours in model space.
///
/// Flattening decisions run in font units (`tol2` is a squared font-unit
/// tolerance); emitted points are `(p + offset) * scale`.
struct GlyphContours {
    offset: [f64; 2],
    scale: f64,
    tol2: f64,
    cursor: [f64; 2],
    current: Vec<[f64; 2]>,
    contours: Vec<Vec<[f64; 2]>>,
}

impl GlyphContours {
    fn push(&mut self, p: [f64; 2]) {
        self.current
            .push([(p[0] + self.offset[0]) * self.scale, (p[1] + self.offset[1]) * self.scale]);
    }

    fn flush(&mut self) {
        // Degenerate contours (fewer than 3 distinct points) can't enclose
        // area; drop them rather than failing the whole conversion — some
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

impl OutlineBuilder for GlyphContours {
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

/// Horizontal `kern`-table adjustment between two glyphs, in font units.
fn pair_kerning(face: &Face, left: GlyphId, right: GlyphId) -> f64 {
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

/// Lay the text out and return the flattened contours in model space.
fn text_contours(cfg: &TextConfig, font_bytes: &[u8]) -> Result<Vec<Vec<[f64; 2]>>, String> {
    if !(cfg.size > 0.0 && cfg.size.is_finite()) {
        return Err(format!("size must be > 0, got {}", cfg.size));
    }
    if !(cfg.line_height > 0.0 && cfg.line_height.is_finite()) {
        return Err(format!("line_height must be > 0, got {}", cfg.line_height));
    }
    if !cfg.letter_spacing.is_finite() {
        return Err(format!("letter_spacing must be finite, got {}", cfg.letter_spacing));
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
        let mut width = 0.0f64;
        for (i, &glyph) in glyphs.iter().enumerate() {
            if i > 0 {
                width += letter_spacing + pair_kerning(&face, glyphs[i - 1], glyph);
            }
            width += face.glyph_hor_advance(glyph).unwrap_or(0) as f64;
        }
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
            let mut sink = GlyphContours {
                offset: [pen_x, baseline_y],
                scale,
                tol2: (CHORD_TOL_EM * upem) * (CHORD_TOL_EM * upem),
                cursor: [0.0, 0.0],
                current: Vec::new(),
                contours: std::mem::take(&mut contours),
            };
            face.outline_glyph(glyph, &mut sink);
            sink.flush();
            contours = sink.contours;
            pen_x += face.glyph_hor_advance(glyph).unwrap_or(0) as f64;
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

/// Read the constant a trivial `() -> i32` function returns.
fn const_i32_return(module: &Module, func_id: FunctionId) -> Option<i32> {
    let local = match &module.funcs.get(func_id).kind {
        walrus::FunctionKind::Local(local) => local,
        _ => return None,
    };
    let block = local.block(local.entry_block());
    match block.instrs.as_slice() {
        [(walrus::ir::Instr::Const(c), _)] => match c.value {
            walrus::ir::Value::I32(v) => Some(v),
            _ => None,
        },
        _ => None,
    }
}

fn patch_template(payload: &[u8]) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(TEMPLATE, &config)
        .map_err(|e| format!("failed to parse the embedded template: {e}"))?;

    let memory_id = module
        .exports
        .iter()
        .find(|e| e.name == "memory")
        .and_then(|e| match e.item {
            walrus::ExportItem::Memory(m) => Some(m),
            _ => None,
        })
        .ok_or("template missing memory export")?;

    // The patch slot's address, then drop the helper export — it is not
    // part of the Model ABI.
    let slot_export = module
        .exports
        .iter()
        .find(|e| e.name == "outline_payload_slot")
        .map(|e| (e.id(), e.item))
        .ok_or("template missing outline_payload_slot export")?;
    let slot_addr = match slot_export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(&module, f)
            .ok_or("template outline_payload_slot is not a constant function")?,
        _ => return Err("template outline_payload_slot is not a function".to_string()),
    };
    module.exports.delete(slot_export.0);

    // Payload in freshly reserved pages; base address into the slot.
    let base = {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65536;
        memory.initial += (payload.len() as u64).div_ceil(65536);
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
        base
    };
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(base as i32)),
        },
        payload.to_vec(),
    );
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(slot_addr)),
        },
        (base as u32).to_le_bytes().to_vec(),
    );

    Ok(module.emit_wasm())
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
        .and_then(|payload| patch_template(&payload));
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
        }
    })
}
