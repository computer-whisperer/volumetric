//! DOM → taffy layout → ink contours.
//!
//! # Content model
//!
//! Every element is either a *layout box* (children are block elements;
//! whitespace-only text between them is ignored) or a *text block*
//! (children are text and inline elements: span/b/strong/br). Mixing
//! bare text with block children is an error — the split keeps layout
//! and typesetting cleanly separated and matches how card markup is
//! actually written.
//!
//! # Ink parity
//!
//! The paint walk carries a paper/ink context. Backgrounds paint ink
//! only when `Dark` (light panels on paper stay paper); inside ink, a
//! `White`/`Light` background knocks a hole and flips the context back
//! to paper — so a light card on a dark plate, with dark text on it,
//! renders correctly with no special cases. Text and borders paint on
//! paper unless `White`, and knock out of ink unless `Dark`. Knockout is
//! winding reversal: every contour set is normalized so its outermost
//! contour winds positive for ink, negative for knockout, which the
//! nonzero fill rule then resolves under arbitrary nesting.
//!
//! Output space: y-up px, origin at the card's top-left (y runs
//! negative downward). [`to_model_space`] recenters and scales.

use taffy::prelude::*;
use taffy::{AvailableSpace, NodeId, TaffyTree};

use crate::parse::{Element, Node};
use crate::style::{ResolvedStyle, TextAlign, TextStyle, Tone, resolve_classes};
use crate::text::{
    self, FaceSource, Paragraph, Run, Token, break_lines, min_content_width, shape_paragraph,
};

pub type Contour = Vec<[f64; 2]>;

/// The geometry of a rendered card, in y-up px space (top-left origin).
#[derive(Debug)]
pub struct CardGeometry {
    pub ink: Vec<Contour>,
    pub plate: Vec<Contour>,
    pub width_px: f64,
    pub height_px: f64,
}

const INLINE_TAGS: &[&str] = &["span", "b", "strong", "br"];

/// A built node: resolved style plus either children or a shaped
/// paragraph.
struct BuildNode {
    style: ResolvedStyle,
    kind: BuildKind,
}

enum BuildKind {
    Boxed(Vec<BuildNode>),
    Text(Paragraph),
}

/// Lay out and paint `root` at `width_px` CSS pixels wide.
pub fn render(
    root: &Element,
    width_px: f64,
    fonts: &dyn FaceSource,
) -> Result<CardGeometry, String> {
    let mut errors = Vec::new();
    let mut missing = Vec::new();
    let built = build_node(
        root,
        &TextStyle::default(),
        fonts,
        &mut errors,
        &mut missing,
    );

    missing.sort_unstable();
    missing.dedup();
    if !missing.is_empty() {
        let listed: Vec<String> = missing.iter().map(|c| format!("{c:?}")).collect();
        errors.push(format!(
            "the embedded font has no glyph for {}",
            listed.join(", ")
        ));
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    let Some(mut built) = built else {
        return Err("the root element is hidden".to_string());
    };

    // The card is `width_px` wide unless the root pins its own width.
    if built.style.taffy.size.width == Dimension::auto() {
        built.style.taffy.size.width = length(width_px as f32);
    }

    let mut tree: TaffyTree<Paragraph> = TaffyTree::new();
    tree.disable_rounding();
    let taffy_root = to_taffy(&mut tree, &built);
    tree.compute_layout_with_measure(
        taffy_root,
        Size {
            width: AvailableSpace::Definite(width_px as f32),
            height: AvailableSpace::MaxContent,
        },
        |known, available, _node, ctx, _style| {
            let Some(paragraph) = ctx else {
                return Size::ZERO;
            };
            if paragraph.tokens.is_empty() {
                return Size::ZERO;
            }
            let avail = known.width.map(f64::from).unwrap_or(match available.width {
                AvailableSpace::Definite(w) => w as f64,
                AvailableSpace::MinContent => {
                    return {
                        let w = min_content_width(paragraph);
                        let (_, h) = text::measure(paragraph, w);
                        Size {
                            width: w as f32,
                            height: h as f32,
                        }
                    };
                }
                AvailableSpace::MaxContent => f64::INFINITY,
            });
            let (w, h) = text::measure(paragraph, avail);
            Size {
                width: w as f32,
                height: h as f32,
            }
        },
    )
    .map_err(|e| format!("layout failed: {e}"))?;

    let root_layout = *tree.layout(taffy_root).map_err(|e| e.to_string())?;
    let (card_w, card_h) = (
        root_layout.size.width as f64,
        root_layout.size.height as f64,
    );

    let mut painter = Painter {
        tree: &tree,
        fonts,
        ink: Vec::new(),
    };
    painter.paint(&built, taffy_root, [0.0, 0.0], Ctx::Paper);

    let plate = rounded_rect(
        [0.0, 0.0],
        [card_w, card_h],
        built.style.boxed.radius_px,
        true,
    );

    Ok(CardGeometry {
        ink: painter.ink,
        plate,
        width_px: card_w,
        height_px: card_h,
    })
}

/// Recenter on the card's middle and scale to a model width of
/// `width_model`: (x, y) → ((x - w/2) s, (y + h/2) s).
pub fn to_model_space(
    contours: &[Contour],
    geometry: &CardGeometry,
    width_model: f64,
) -> Vec<Contour> {
    let s = width_model / geometry.width_px;
    contours
        .iter()
        .map(|c| {
            c.iter()
                .map(|p| {
                    [
                        (p[0] - geometry.width_px / 2.0) * s,
                        (p[1] + geometry.height_px / 2.0) * s,
                    ]
                })
                .collect()
        })
        .collect()
}

// --- tree building ---

fn build_node(
    el: &Element,
    inherited: &TextStyle,
    fonts: &dyn FaceSource,
    errors: &mut Vec<String>,
    missing: &mut Vec<char>,
) -> Option<BuildNode> {
    let mut style = resolve_classes(&el.classes, inherited, false, &el.tag, errors);
    if style.boxed.hidden {
        return None;
    }
    if el.tag == "hr" {
        // Tailwind preflight: an hr is a 1px top border (thickness
        // overridable with border classes).
        if style.boxed.border_px == [0.0; 4] {
            style.boxed.border_px = [1.0, 0.0, 0.0, 0.0];
        }
        style.taffy.border.top = length(style.boxed.border_px[0] as f32);
        return Some(BuildNode {
            style,
            kind: BuildKind::Boxed(Vec::new()),
        });
    }
    if el.tag == "br" {
        errors.push("<br> outside a text block does nothing".to_string());
        return None;
    }

    let is_text_block = el.children.iter().any(|child| match child {
        Node::Text(t) => !t.trim().is_empty(),
        Node::Element(child) => INLINE_TAGS.contains(&child.tag.as_str()),
    });

    if is_text_block {
        let mut runs = Vec::new();
        let mut breaks = Vec::new();
        collect_runs(
            &el.children,
            &style.text,
            &el.tag,
            &mut runs,
            &mut breaks,
            errors,
        );
        let paragraph = shape_paragraph(&runs, &breaks, style.text.align, fonts, missing);
        // Border widths participate in taffy sizing.
        apply_border_to_taffy(&mut style);
        return Some(BuildNode {
            style,
            kind: BuildKind::Text(paragraph),
        });
    }

    let mut children = Vec::new();
    for child in &el.children {
        match child {
            Node::Text(_) => {} // whitespace between blocks
            Node::Element(child_el) => {
                if let Some(node) = build_node(child_el, &style.text, fonts, errors, missing) {
                    children.push(node);
                }
            }
        }
    }
    apply_border_to_taffy(&mut style);
    Some(BuildNode {
        style,
        kind: BuildKind::Boxed(children),
    })
}

fn apply_border_to_taffy(style: &mut ResolvedStyle) {
    let [t, r, b, l] = style.boxed.border_px;
    style.taffy.border = Rect {
        top: length(t as f32),
        right: length(r as f32),
        bottom: length(b as f32),
        left: length(l as f32),
    };
}

/// Flatten a text block's inline tree into styled runs (`breaks[i]` marks
/// a `<br>` after run `i`).
fn collect_runs<'a>(
    children: &'a [Node],
    style: &TextStyle,
    where_: &str,
    runs: &mut Vec<Run<'a>>,
    breaks: &mut Vec<bool>,
    errors: &mut Vec<String>,
) {
    for child in children {
        match child {
            Node::Text(t) => {
                runs.push(Run {
                    text: t,
                    style: *style,
                });
                breaks.push(false);
            }
            Node::Element(el) if el.tag == "br" => {
                if breaks.is_empty() {
                    runs.push(Run {
                        text: "",
                        style: *style,
                    });
                    breaks.push(true);
                } else {
                    *breaks.last_mut().unwrap() = true;
                }
            }
            Node::Element(el) if INLINE_TAGS.contains(&el.tag.as_str()) => {
                let mut inner = *style;
                if el.tag == "b" || el.tag == "strong" {
                    inner.bold = true;
                }
                let resolved = resolve_classes(&el.classes, &inner, true, &el.tag, errors);
                collect_runs(&el.children, &resolved.text, &el.tag, runs, breaks, errors);
            }
            Node::Element(el) => {
                errors.push(format!(
                    "{where_}: <{tag}> cannot appear inside a text block \
                     (text and block elements don't mix in one container)",
                    tag = el.tag
                ));
            }
        }
    }
}

fn to_taffy(tree: &mut TaffyTree<Paragraph>, node: &BuildNode) -> NodeId {
    match &node.kind {
        BuildKind::Text(paragraph) => tree
            .new_leaf_with_context(node.style.taffy.clone(), paragraph.clone())
            .expect("taffy leaf"),
        BuildKind::Boxed(children) => {
            let ids: Vec<NodeId> = children.iter().map(|c| to_taffy(tree, c)).collect();
            tree.new_with_children(node.style.taffy.clone(), &ids)
                .expect("taffy node")
        }
    }
}

// --- painting ---

#[derive(Clone, Copy, PartialEq)]
enum Ctx {
    Paper,
    Ink,
}

struct Painter<'a> {
    tree: &'a TaffyTree<Paragraph>,
    fonts: &'a dyn FaceSource,
    ink: Vec<Contour>,
}

impl Painter<'_> {
    /// Paint `node` (whose taffy id is `id`) at the absolute y-down px
    /// offset `abs` of its top-left corner. Children of a taffy node are
    /// positioned parent-relative.
    fn paint(&mut self, node: &BuildNode, id: NodeId, abs: [f64; 2], ctx: Ctx) {
        let layout = *self.tree.layout(id).expect("layout");
        let abs = [
            abs[0] + layout.location.x as f64,
            abs[1] + layout.location.y as f64,
        ];
        let size = [layout.size.width as f64, layout.size.height as f64];
        let boxed = &node.style.boxed;

        // Background: Dark paints ink on paper; White/Light knocks out of
        // ink and flips the context back to paper.
        let mut inner_ctx = ctx;
        if let Some(tone) = boxed.bg {
            match (ctx, tone) {
                (Ctx::Paper, Tone::Dark) => {
                    self.ink
                        .extend(rounded_rect(abs, size, boxed.radius_px, true));
                    inner_ctx = Ctx::Ink;
                }
                (Ctx::Ink, Tone::White | Tone::Light) => {
                    self.ink
                        .extend(rounded_rect(abs, size, boxed.radius_px, false));
                    inner_ctx = Ctx::Paper;
                }
                _ => {}
            }
        }

        // Border ring, over the background, in the post-background context.
        if boxed.border_px.iter().any(|&w| w > 0.0) {
            let positive = match (inner_ctx, boxed.border_tone) {
                (Ctx::Paper, Tone::White) => None,
                (Ctx::Paper, _) => Some(true),
                (Ctx::Ink, Tone::Dark) => None,
                (Ctx::Ink, _) => Some(false),
            };
            if let Some(positive) = positive {
                self.ink.extend(border_ring(
                    abs,
                    size,
                    boxed.border_px,
                    boxed.radius_px,
                    positive,
                ));
            }
        }

        match &node.kind {
            BuildKind::Boxed(children) => {
                // Taffy child order mirrors the build order.
                let child_ids = self.tree.children(id).expect("children");
                for (child, child_id) in children.iter().zip(child_ids) {
                    self.paint(child, child_id, abs, inner_ctx);
                }
            }
            BuildKind::Text(paragraph) => {
                self.paint_paragraph(paragraph, &layout, abs, inner_ctx);
            }
        }
    }

    fn paint_paragraph(
        &mut self,
        paragraph: &Paragraph,
        layout: &taffy::Layout,
        abs: [f64; 2],
        ctx: Ctx,
    ) {
        if paragraph.tokens.is_empty() {
            return;
        }
        let content_x = abs[0] + (layout.border.left + layout.padding.left) as f64;
        let content_y = abs[1] + (layout.border.top + layout.padding.top) as f64;
        let content_w = layout.size.width as f64
            - (layout.border.left
                + layout.border.right
                + layout.padding.left
                + layout.padding.right) as f64;

        let lines = break_lines(paragraph, content_w + 1e-6);
        let lh = paragraph.line_height_px;
        // CSS half-leading: the line box centers the ascent+descent span.
        let half_leading = (lh - (paragraph.ascent_px + paragraph.descent_px)) / 2.0;

        for (line_idx, line) in lines.iter().enumerate() {
            let baseline_y = content_y + line_idx as f64 * lh + half_leading + paragraph.ascent_px;
            let x0 = match paragraph.align {
                TextAlign::Left => content_x,
                TextAlign::Center => content_x + (content_w - line.width_px) / 2.0,
                TextAlign::Right => content_x + content_w - line.width_px,
            };
            for &(token_idx, x_off) in &line.words {
                let Token::Word(word) = &paragraph.tokens[token_idx] else {
                    continue;
                };
                let positive = match (ctx, word.style.tone) {
                    (Ctx::Paper, Tone::White) => continue,
                    (Ctx::Paper, _) => true,
                    (Ctx::Ink, Tone::Dark) => continue,
                    (Ctx::Ink, _) => false,
                };
                let face = self.fonts.face(word.style.bold);
                let upem = face.units_per_em() as f64;
                let ink = &mut self.ink;
                text::place_word(word, self.fonts, x0 + x_off, |glyph, pen_x, scale| {
                    // Emit in y-up space: y_up = -y_down.
                    let mut contours = Vec::new();
                    text_render_core::outline_glyph_into(
                        face,
                        glyph,
                        [pen_x / scale, -baseline_y / scale],
                        scale,
                        upem / 512.0,
                        &mut contours,
                    );
                    normalize_winding(&mut contours, positive);
                    ink.extend(contours);
                });
            }
        }
    }
}

/// Shoelace signed area (y-up convention: counter-clockwise positive).
fn signed_area(contour: &Contour) -> f64 {
    let mut sum = 0.0;
    for i in 0..contour.len() {
        let a = contour[i];
        let b = contour[(i + 1) % contour.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

/// Normalize one element's contour set so its dominant (largest-area)
/// contour winds positive when `positive`, negative otherwise; the
/// relative winding of holes is preserved.
fn normalize_winding(contours: &mut [Contour], positive: bool) {
    let dominant = contours
        .iter()
        .map(signed_area)
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap_or(0.0);
    if (dominant > 0.0) != positive {
        for contour in contours.iter_mut() {
            contour.reverse();
        }
    }
}

/// A rounded rectangle contour for the y-down px rect at `pos`/`size`,
/// emitted in y-up space, wound positive (counter-clockwise) when
/// `positive`.
fn rounded_rect(pos: [f64; 2], size: [f64; 2], radius: f64, positive: bool) -> Vec<Contour> {
    let [w, h] = size;
    if w <= 0.0 || h <= 0.0 {
        return Vec::new();
    }
    let r = radius.min(w / 2.0).min(h / 2.0).max(0.0);
    // Corner centers in y-up coordinates (y = -y_down).
    let (x0, y0) = (pos[0], -pos[1] - h); // bottom-left of the box, y-up
    let (x1, y1) = (pos[0] + w, -pos[1]); // top-right
    let mut points: Vec<[f64; 2]> = Vec::new();

    if r <= 0.0 {
        points.extend([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    } else {
        // CCW starting on the bottom edge; ~0.2px chord tolerance.
        let segments = arc_segments(r);
        let mut corner = |cx: f64, cy: f64, start_angle: f64| {
            for i in 0..=segments {
                let a = start_angle + std::f64::consts::FRAC_PI_2 * i as f64 / segments as f64;
                points.push([cx + r * a.cos(), cy + r * a.sin()]);
            }
        };
        corner(x1 - r, y0 + r, -std::f64::consts::FRAC_PI_2); // bottom-right
        corner(x1 - r, y1 - r, 0.0); // top-right
        corner(x0 + r, y1 - r, std::f64::consts::FRAC_PI_2); // top-left
        corner(x0 + r, y0 + r, std::f64::consts::PI); // bottom-left
    }
    if !positive {
        points.reverse();
    }
    vec![points]
}

/// Segment count for a quarter arc of radius `r` at ~0.2px tolerance.
fn arc_segments(r: f64) -> u32 {
    let tol = 0.2;
    if r <= tol {
        return 2;
    }
    let step = 2.0 * (1.0 - tol / r).acos();
    ((std::f64::consts::FRAC_PI_2 / step).ceil() as u32).clamp(2, 64)
}

/// A border ring: outer rounded rect minus the inner window. Wound so the
/// ring area is `positive`.
fn border_ring(
    pos: [f64; 2],
    size: [f64; 2],
    widths: [f64; 4],
    radius: f64,
    positive: bool,
) -> Vec<Contour> {
    let [t, r_w, b, l] = widths;
    let inner_pos = [pos[0] + l, pos[1] + t];
    let inner_size = [size[0] - l - r_w, size[1] - t - b];
    let mut out = rounded_rect(pos, size, radius, positive);
    if inner_size[0] > 0.0 && inner_size[1] > 0.0 {
        let max_w = t.max(r_w).max(b).max(l);
        let inner_radius = (radius - max_w).max(0.0);
        out.extend(rounded_rect(inner_pos, inner_size, inner_radius, !positive));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_fragment;
    use text_render_core::ttf_parser::Face;

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

    fn render_ok(html: &str) -> CardGeometry {
        let root = parse_fragment(html).expect("parses");
        render(&root, 384.0, &fonts()).expect("renders")
    }

    /// Winding number of `p` over a contour set (nonzero fill test).
    fn winding(contours: &[Contour], p: [f64; 2]) -> i32 {
        let mut w = 0;
        for contour in contours {
            for i in 0..contour.len() {
                let a = contour[i];
                let b = contour[(i + 1) % contour.len()];
                if (a[1] > p[1]) != (b[1] > p[1]) {
                    let t = (p[1] - a[1]) / (b[1] - a[1]);
                    if a[0] + t * (b[0] - a[0]) > p[0] {
                        w += if b[1] > a[1] { 1 } else { -1 };
                    }
                }
            }
        }
        w
    }

    fn inked(g: &CardGeometry, x: f64, y_down: f64) -> bool {
        winding(&g.ink, [x, -y_down]) != 0
    }

    #[test]
    fn plate_matches_card_box_and_content_grows_height() {
        let g = render_ok(r#"<div class="p-4"><p>Hello world</p></div>"#);
        assert_eq!(g.width_px, 384.0);
        // 24px line + 2*16 padding.
        assert!((g.height_px - 56.0).abs() < 0.5, "height {}", g.height_px);
        assert_eq!(winding(&g.plate, [1.0, -1.0]), 1);
        assert_eq!(winding(&g.plate, [-1.0, 1.0]), 0);
    }

    #[test]
    fn dark_chip_knocks_out_white_text() {
        let g = render_ok(
            r#"<div class="p-4"><div class="bg-black p-4"><p class="text-white">OK</p></div></div>"#,
        );
        // p-4 = 16px: the chip spans y [16, 72] full-width. Inside it but
        // left of the text: ink.
        assert!(inked(&g, 20.0, 20.0));
        // Scanning across the caps' midline must cross knocked-out glyph
        // strokes (text content starts at 32, baseline ≈ 49.8).
        let mut holes = 0;
        let mut x = 33.0;
        while x < 70.0 {
            if !inked(&g, x, 44.0) {
                holes += 1;
            }
            x += 0.25;
        }
        assert!(holes > 5, "expected knockout inside glyphs, got {holes}");
    }

    #[test]
    fn light_panel_on_dark_plate_flips_back_to_paper() {
        let g = render_ok(
            r#"<div class="bg-black p-8"><div class="bg-white p-4"><p>A</p></div></div>"#,
        );
        // Outer dark area: ink. Panel interior (starts at 32+): paper.
        assert!(inked(&g, 4.0, 4.0));
        assert!(!inked(&g, 40.0, 40.0));
        // The A prints as ink on the panel (content starts at 48,
        // baseline ≈ 65.8 — scan the caps' midline).
        let mut ink_hits = 0;
        let mut x = 48.0;
        while x < 70.0 {
            if inked(&g, x, 60.0) {
                ink_hits += 1;
            }
            x += 0.25;
        }
        assert!(
            ink_hits > 3,
            "glyph should print on the panel, got {ink_hits}"
        );
    }

    #[test]
    fn border_ring_prints_with_hollow_center() {
        let g = render_ok(r#"<div class="border-2 p-4"><p>x</p></div>"#);
        assert!(inked(&g, 1.0, 20.0), "left border strip");
        assert!(inked(&g, 383.0, 20.0), "right border strip");
        assert!(!inked(&g, 3.0, 20.0), "just inside the ring");
    }

    #[test]
    fn flex_row_places_children_side_by_side() {
        let g = render_ok(
            r#"<div class="flex gap-4"><div class="w-16 h-8 bg-black"></div><div class="w-16 h-8 bg-black"></div></div>"#,
        );
        // w-16 = 64px chips, gap-4 = 16px: chips at [0,64) and [80,144).
        assert!(inked(&g, 8.0, 4.0), "first chip");
        assert!(!inked(&g, 72.0, 4.0), "gap");
        assert!(inked(&g, 90.0, 4.0), "second chip");
        assert!((g.height_px - 32.0).abs() < 0.5);
    }

    #[test]
    fn grid_two_columns_split_the_width() {
        let g = render_ok(
            r#"<div class="grid grid-cols-2 gap-4"><div class="h-8 bg-black"></div><div class="h-8 bg-black"></div></div>"#,
        );
        // Columns: [0,190) and [194,384).
        assert!(inked(&g, 100.0, 4.0));
        assert!(!inked(&g, 192.0, 4.0));
        assert!(inked(&g, 300.0, 4.0));
    }

    #[test]
    fn hr_renders_a_rule() {
        // p-4 = 16px padding: the 2px rule spans y [16, 18).
        let g = render_ok(r#"<div class="p-4"><hr class="border-2"/></div>"#);
        assert!(inked(&g, 100.0, 17.0), "rule strip");
        assert!(!inked(&g, 100.0, 25.0), "below the rule");
        assert!(!inked(&g, 8.0, 17.0), "left padding stays clear");
    }

    #[test]
    fn text_align_and_uppercase_apply() {
        let left = render_ok(r#"<div class="w-64"><p>hi</p></div>"#);
        let right = render_ok(r#"<div class="w-64"><p class="text-right">hi</p></div>"#);
        // Leftmost inked x differs strongly between alignments.
        let first_ink = |g: &CardGeometry| {
            let mut x = 0.0;
            while x < 256.0 {
                if inked(g, x, 12.0) {
                    return x;
                }
                x += 0.5;
            }
            256.0
        };
        assert!(first_ink(&left) < 5.0);
        assert!(first_ink(&right) > 200.0);

        // Uppercase transform: "TALL" runs wider than "tall".
        let ink_extent = |g: &CardGeometry| {
            let mut max_x = 0.0f64;
            for y in [6, 8, 10, 12, 14, 16, 18] {
                let mut x = 0.0;
                while x < 100.0 {
                    if inked(g, x, y as f64) {
                        max_x = max_x.max(x);
                    }
                    x += 0.5;
                }
            }
            max_x
        };
        let lower = render_ok(r#"<div><p>tall</p></div>"#);
        let upper = render_ok(r#"<div><p class="uppercase">tall</p></div>"#);
        assert!(
            ink_extent(&upper) > ink_extent(&lower) + 3.0,
            "upper {} vs lower {}",
            ink_extent(&upper),
            ink_extent(&lower)
        );
    }

    #[test]
    fn mixed_content_and_bad_markup_error() {
        let root = parse_fragment("<div>text<div></div></div>").unwrap();
        let err = render(&root, 384.0, &fonts()).unwrap_err();
        assert!(err.contains("don't mix"), "{err}");

        let root = parse_fragment(r#"<div class="blorp"><p>x</p></div>"#).unwrap();
        let err = render(&root, 384.0, &fonts()).unwrap_err();
        assert!(err.contains("blorp"), "{err}");

        let root = parse_fragment("<div><p>a\u{2603}</p></div>").unwrap();
        let err = render(&root, 384.0, &fonts()).unwrap_err();
        assert!(err.contains('\u{2603}'), "{err}");
    }

    #[test]
    fn model_space_transform_centers() {
        let g = render_ok(r#"<div class="h-16"><p>x</p></div>"#);
        let model = to_model_space(&g.plate, &g, 0.096);
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in model.iter().flatten() {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        assert!((max_x - 0.048).abs() < 1e-9 && (min_x + 0.048).abs() < 1e-9);
        assert!((max_y + min_y).abs() < 1e-9, "y centered");
        assert!((max_y - min_y - 0.096 * 64.0 / 384.0).abs() < 1e-9);
    }
}
