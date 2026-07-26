//! The Tailwind utility-class table: resolves each element's class list
//! into a taffy layout style plus the text/paint properties the renderer
//! needs. The supported families are chosen to cover what real info-card
//! markup uses; anything else is a collected error naming the class —
//! except a short documented list of purely decorative no-ops (shadows,
//! transitions) that have no geometric meaning in monochrome ink.
//!
//! Colors reduce to a three-way [`Tone`]: `White` (white/transparent),
//! `Light` (black-amount below Tailwind shade 500, or luminance >= 0.5
//! for arbitrary hex), `Dark` (the rest). The paint walk turns tones into
//! ink: backgrounds paint only when `Dark`; text and borders paint unless
//! `White` on paper, and knock out of ink when `White`/`Light`.

use taffy::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    White,
    Light,
    Dark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextTransform {
    None,
    Upper,
    Lower,
    Capitalize,
}

/// Inheritable text properties (cascade parent → child).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextStyle {
    pub font_px: f64,
    /// Resolved line height in px (size-class defaults baked in).
    pub line_height_px: f64,
    pub bold: bool,
    pub align: TextAlign,
    pub tracking_em: f64,
    pub transform: TextTransform,
    pub tone: Tone,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_px: 16.0,
            line_height_px: 24.0,
            bold: false,
            align: TextAlign::Left,
            tracking_em: 0.0,
            transform: TextTransform::None,
            tone: Tone::Dark,
        }
    }
}

/// Non-inherited paint properties of one element's box.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxStyle {
    pub bg: Option<Tone>,
    /// Border widths in px: top, right, bottom, left.
    pub border_px: [f64; 4],
    pub border_tone: Tone,
    pub radius_px: f64,
    pub hidden: bool,
}

impl Default for BoxStyle {
    fn default() -> Self {
        Self {
            bg: None,
            border_px: [0.0; 4],
            // Tailwind's default border color is a light gray.
            border_tone: Tone::Light,
            radius_px: 0.0,
            hidden: false,
        }
    }
}

pub struct ResolvedStyle {
    pub taffy: Style,
    pub text: TextStyle,
    pub boxed: BoxStyle,
}

/// Purely decorative classes with no geometric meaning in monochrome ink;
/// accepted as documented no-ops so idiomatic card markup one-shots.
fn is_decorative_noop(class: &str) -> bool {
    class == "antialiased"
        || class == "font-sans"
        || class == "overflow-hidden"
        || class == "select-none"
        || class == "truncate"
        || class.starts_with("shadow")
        || class.starts_with("transition")
        || class.starts_with("duration-")
        || class.starts_with("ease-")
        || class.starts_with("cursor-")
        || class.starts_with("opacity-")
}

/// Tailwind's default color-palette hue names (for `{hue}-{shade}`).
const HUES: &[&str] = &[
    "slate", "gray", "zinc", "neutral", "stone", "red", "orange", "amber", "yellow", "lime",
    "green", "emerald", "teal", "cyan", "sky", "blue", "indigo", "violet", "purple", "fuchsia",
    "pink", "rose",
];

/// Classify a color token (`white`, `slate-200`, `[#1e293b]`, …).
fn parse_tone(token: &str) -> Option<Tone> {
    match token {
        "white" | "transparent" => return Some(Tone::White),
        "black" => return Some(Tone::Dark),
        _ => {}
    }
    if let Some(hex) = token.strip_prefix("[#").and_then(|t| t.strip_suffix(']')) {
        let expand = |s: &str| u8::from_str_radix(&format!("{s}{s}"), 16).ok();
        let (r, g, b) = match hex.len() {
            3 => (
                expand(&hex[0..1])?,
                expand(&hex[1..2])?,
                expand(&hex[2..3])?,
            ),
            6 => (
                u8::from_str_radix(&hex[0..2], 16).ok()?,
                u8::from_str_radix(&hex[2..4], 16).ok()?,
                u8::from_str_radix(&hex[4..6], 16).ok()?,
            ),
            _ => return None,
        };
        let lum = (0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64) / 255.0;
        return Some(if lum >= 0.98 {
            Tone::White
        } else if lum >= 0.5 {
            Tone::Light
        } else {
            Tone::Dark
        });
    }
    let (hue, shade) = token.rsplit_once('-')?;
    if !HUES.contains(&hue) {
        return None;
    }
    let shade: u32 = shade.parse().ok()?;
    if !(50..=950).contains(&shade) {
        return None;
    }
    Some(if shade < 500 { Tone::Light } else { Tone::Dark })
}

/// Tailwind spacing token → px: the 0.25rem scale (`4` → 16px), `px` → 1,
/// fractional steps, and `[Npx]`/`[Nrem]` arbitrary values.
fn parse_spacing(token: &str) -> Option<f64> {
    match token {
        "px" => return Some(1.0),
        "0" => return Some(0.0),
        _ => {}
    }
    if let Some(arb) = token.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
        return parse_arbitrary_px(arb);
    }
    let steps: f64 = token.parse().ok()?;
    (steps.is_finite() && steps >= 0.0).then_some(steps * 4.0)
}

/// `10px` / `1.5rem` / `0` → px (finite, non-negative only).
fn parse_arbitrary_px(value: &str) -> Option<f64> {
    if value == "0" {
        return Some(0.0);
    }
    let px = if let Some(px) = value.strip_suffix("px") {
        px.parse::<f64>().ok()?
    } else {
        value.strip_suffix("rem")?.parse::<f64>().ok()? * 16.0
    };
    (px.is_finite() && px >= 0.0).then_some(px)
}

/// Width/height-style token → taffy dimension (`24`, `full`, `1/2`,
/// `[120px]`, `auto`).
fn parse_dimension(token: &str) -> Option<Dimension> {
    match token {
        "full" => return Some(percent(1.0_f32)),
        "auto" => return Some(auto()),
        _ => {}
    }
    if let Some((num, den)) = token.split_once('/') {
        let (num, den): (f64, f64) = (num.parse().ok()?, den.parse().ok()?);
        if den > 0.0 && num >= 0.0 {
            return Some(percent((num / den) as f32));
        }
        return None;
    }
    parse_spacing(token).map(|px| length(px as f32))
}

/// Tailwind text-size classes: (name, font px, default line-height px).
const TEXT_SIZES: &[(&str, f64, f64)] = &[
    ("xs", 12.0, 16.0),
    ("sm", 14.0, 20.0),
    ("base", 16.0, 24.0),
    ("lg", 18.0, 28.0),
    ("xl", 20.0, 28.0),
    ("2xl", 24.0, 32.0),
    ("3xl", 30.0, 36.0),
    ("4xl", 36.0, 40.0),
    ("5xl", 48.0, 48.0),
    ("6xl", 60.0, 60.0),
];

const ROUNDED: &[(&str, f64)] = &[
    ("none", 0.0),
    ("sm", 2.0),
    ("md", 6.0),
    ("lg", 8.0),
    ("xl", 12.0),
    ("2xl", 16.0),
    ("3xl", 24.0),
    ("full", 1e9),
];

const MAX_W_NAMED: &[(&str, f64)] = &[
    ("xs", 320.0),
    ("sm", 384.0),
    ("md", 448.0),
    ("lg", 512.0),
    ("xl", 576.0),
    ("2xl", 672.0),
];

/// Resolve `classes` against the inherited `parent` text style.
///
/// `inline` marks span/b/strong contexts where only text-styling classes
/// make sense; layout classes there are errors. Unknown or unsupported
/// classes push onto `errors` (prefixed with `where_`) and resolution
/// continues, so one pass reports every offender.
pub fn resolve_classes(
    classes: &[String],
    parent: &TextStyle,
    inline: bool,
    where_: &str,
    errors: &mut Vec<String>,
) -> ResolvedStyle {
    let mut style = Style::DEFAULT;
    // CSS block flow is the default for every supported container tag
    // (taffy's own default is Flex, which is not what HTML markup means).
    style.display = Display::Block;
    let mut text = *parent;
    let mut boxed = BoxStyle::default();
    // `leading-*` must win over a size class's default line height
    // regardless of class order, like CSS specificity does.
    let mut explicit_leading: Option<LeadingSpec> = None;

    #[derive(Clone, Copy)]
    enum LeadingSpec {
        Mult(f64),
        Px(f64),
    }

    let mut err = |msg: String| errors.push(format!("{where_}: {msg}"));

    for class in classes {
        let class = class.as_str();
        if is_decorative_noop(class) {
            continue;
        }
        if let Some((variant, _)) = class.split_once(':') {
            err(format!(
                "class {class:?}: {variant}: variants (responsive/state) are not supported"
            ));
            continue;
        }

        // --- text families (legal everywhere, inline included) ---
        if let Some(rest) = class.strip_prefix("text-") {
            if let Some(&(_, px, lh)) = TEXT_SIZES.iter().find(|(n, ..)| *n == rest) {
                text.font_px = px;
                text.line_height_px = lh;
                continue;
            }
            match rest {
                "left" => {
                    text.align = TextAlign::Left;
                    continue;
                }
                "center" => {
                    text.align = TextAlign::Center;
                    continue;
                }
                "right" => {
                    text.align = TextAlign::Right;
                    continue;
                }
                _ => {}
            }
            if let Some(px) = rest
                .strip_prefix('[')
                .and_then(|t| t.strip_suffix(']'))
                .filter(|v| !v.starts_with('#'))
                .and_then(parse_arbitrary_px)
            {
                text.font_px = px;
                text.line_height_px = px * 1.5;
                continue;
            }
            if let Some(tone) = parse_tone(rest) {
                text.tone = tone;
                continue;
            }
            err(format!("unknown class {class:?}"));
            continue;
        }
        if let Some(rest) = class.strip_prefix("font-") {
            match rest {
                "bold" | "semibold" | "extrabold" | "black" => text.bold = true,
                "normal" | "medium" | "light" | "extralight" | "thin" => text.bold = false,
                _ => err(format!(
                    "unknown class {class:?} (weights map to the regular/bold faces; \
                     other font families are not embedded)"
                )),
            }
            continue;
        }
        if let Some(rest) = class.strip_prefix("leading-") {
            let spec = match rest {
                "none" => Some(LeadingSpec::Mult(1.0)),
                "tight" => Some(LeadingSpec::Mult(1.25)),
                "snug" => Some(LeadingSpec::Mult(1.375)),
                "normal" => Some(LeadingSpec::Mult(1.5)),
                "relaxed" => Some(LeadingSpec::Mult(1.625)),
                "loose" => Some(LeadingSpec::Mult(2.0)),
                _ => parse_spacing(rest).map(LeadingSpec::Px),
            };
            match spec {
                Some(spec) => explicit_leading = Some(spec),
                None => err(format!("unknown class {class:?}")),
            }
            continue;
        }
        if let Some(rest) = class.strip_prefix("tracking-") {
            let em = match rest {
                "tighter" => Some(-0.05),
                "tight" => Some(-0.025),
                "normal" => Some(0.0),
                "wide" => Some(0.025),
                "wider" => Some(0.05),
                "widest" => Some(0.1),
                _ => rest
                    .strip_prefix('[')
                    .and_then(|t| t.strip_suffix("em]"))
                    .and_then(|v| v.parse().ok()),
            };
            match em {
                Some(em) => text.tracking_em = em,
                None => err(format!("unknown class {class:?}")),
            }
            continue;
        }
        match class {
            "uppercase" => {
                text.transform = TextTransform::Upper;
                continue;
            }
            "lowercase" => {
                text.transform = TextTransform::Lower;
                continue;
            }
            "capitalize" => {
                text.transform = TextTransform::Capitalize;
                continue;
            }
            "normal-case" => {
                text.transform = TextTransform::None;
                continue;
            }
            "italic" => {
                err("class \"italic\": no italic face is embedded".to_string());
                continue;
            }
            _ => {}
        }

        // --- box/layout families ---
        if inline {
            err(format!(
                "class {class:?}: layout classes are not supported on inline elements \
                 (span/b/strong take text styling only)"
            ));
            continue;
        }

        if let Some(rest) = class.strip_prefix("bg-") {
            match parse_tone(rest) {
                Some(tone) => boxed.bg = Some(tone),
                None => err(format!("unknown class {class:?}")),
            }
            continue;
        }
        if class == "border" {
            boxed.border_px = [1.0; 4];
            continue;
        }
        if let Some(rest) = class.strip_prefix("border-") {
            let parse_width =
                |w: &str| w.parse::<f64>().ok().filter(|w| w.is_finite() && *w >= 0.0);
            if let Some(width) = parse_width(rest) {
                boxed.border_px = [width; 4];
                continue;
            }
            let (side, width) = match rest.split_once('-') {
                Some((side, w)) => (side, parse_width(w)),
                None => (rest, Some(1.0)),
            };
            let apply: Option<&[usize]> = match side {
                "t" => Some(&[0]),
                "r" => Some(&[1]),
                "b" => Some(&[2]),
                "l" => Some(&[3]),
                "x" => Some(&[1, 3]),
                "y" => Some(&[0, 2]),
                _ => None,
            };
            if let (Some(sides), Some(width)) = (apply, width) {
                for &i in sides {
                    boxed.border_px[i] = width;
                }
                continue;
            }
            match parse_tone(rest) {
                Some(tone) => boxed.border_tone = tone,
                None => err(format!("unknown class {class:?}")),
            }
            continue;
        }
        if class == "rounded" {
            boxed.radius_px = 4.0;
            continue;
        }
        if let Some(rest) = class.strip_prefix("rounded-") {
            let radius = ROUNDED
                .iter()
                .find(|(n, _)| *n == rest)
                .map(|&(_, r)| r)
                .or_else(|| {
                    rest.strip_prefix('[')
                        .and_then(|t| t.strip_suffix(']'))
                        .and_then(parse_arbitrary_px)
                });
            match radius {
                Some(r) => boxed.radius_px = r,
                None => err(format!(
                    "unknown class {class:?} (per-corner rounding is not supported)"
                )),
            }
            continue;
        }

        if resolve_layout_class(class, &mut style, &mut boxed, &mut err) {
            continue;
        }
        err(format!("unknown class {class:?}"));
    }

    if let Some(spec) = explicit_leading {
        text.line_height_px = match spec {
            LeadingSpec::Mult(m) => m * text.font_px,
            LeadingSpec::Px(px) => px,
        };
    }

    ResolvedStyle {
        taffy: style,
        text,
        boxed,
    }
}

/// Layout-family classes; returns false when the class is not a layout
/// class at all (caller reports it unknown).
fn resolve_layout_class(
    class: &str,
    style: &mut Style,
    boxed: &mut BoxStyle,
    err: &mut impl FnMut(String),
) -> bool {
    match class {
        "flex" => style.display = Display::Flex,
        "grid" => style.display = Display::Grid,
        "block" => style.display = Display::Block,
        "hidden" => boxed.hidden = true,
        "flex-row" => style.flex_direction = FlexDirection::Row,
        "flex-col" => style.flex_direction = FlexDirection::Column,
        "flex-wrap" => style.flex_wrap = FlexWrap::Wrap,
        "flex-nowrap" => style.flex_wrap = FlexWrap::NoWrap,
        "flex-1" => {
            style.flex_grow = 1.0;
            style.flex_shrink = 1.0;
            style.flex_basis = length(0.0_f32);
        }
        "flex-auto" => {
            style.flex_grow = 1.0;
            style.flex_shrink = 1.0;
            style.flex_basis = auto();
        }
        "flex-none" => {
            style.flex_grow = 0.0;
            style.flex_shrink = 0.0;
            style.flex_basis = auto();
        }
        "grow" => style.flex_grow = 1.0,
        "grow-0" => style.flex_grow = 0.0,
        "shrink" => style.flex_shrink = 1.0,
        "shrink-0" => style.flex_shrink = 0.0,
        "items-start" => style.align_items = Some(AlignItems::FLEX_START),
        "items-center" => style.align_items = Some(AlignItems::CENTER),
        "items-end" => style.align_items = Some(AlignItems::FLEX_END),
        "items-stretch" => style.align_items = Some(AlignItems::STRETCH),
        "items-baseline" => style.align_items = Some(AlignItems::BASELINE),
        "justify-start" => style.justify_content = Some(JustifyContent::FLEX_START),
        "justify-center" => style.justify_content = Some(JustifyContent::CENTER),
        "justify-end" => style.justify_content = Some(JustifyContent::FLEX_END),
        "justify-between" => style.justify_content = Some(JustifyContent::SPACE_BETWEEN),
        "justify-around" => style.justify_content = Some(JustifyContent::SPACE_AROUND),
        "justify-evenly" => style.justify_content = Some(JustifyContent::SPACE_EVENLY),
        "self-start" => style.align_self = Some(AlignSelf::FLEX_START),
        "self-center" => style.align_self = Some(AlignSelf::CENTER),
        "self-end" => style.align_self = Some(AlignSelf::FLEX_END),
        "self-stretch" => style.align_self = Some(AlignSelf::STRETCH),
        "mx-auto" => {
            style.margin.left = auto();
            style.margin.right = auto();
        }
        "my-auto" => {
            style.margin.top = auto();
            style.margin.bottom = auto();
        }
        "ml-auto" => style.margin.left = auto(),
        "mr-auto" => style.margin.right = auto(),
        "mt-auto" => style.margin.top = auto(),
        "mb-auto" => style.margin.bottom = auto(),
        "m-auto" => style.margin = Rect::auto(),
        _ => return resolve_prefixed_layout_class(class, style, err),
    }
    true
}

fn resolve_prefixed_layout_class(
    class: &str,
    style: &mut Style,
    err: &mut impl FnMut(String),
) -> bool {
    let mut bad = |class: &str| {
        err(format!("unknown class {class:?}"));
        true
    };

    if let Some(rest) = class.strip_prefix("grid-cols-") {
        return match rest.parse::<usize>() {
            Ok(n) if (1..=12).contains(&n) => {
                style.grid_template_columns = vec![fr(1.0_f32); n];
                true
            }
            _ => bad(class),
        };
    }
    if let Some(rest) = class.strip_prefix("col-span-") {
        return match rest.parse::<u16>() {
            Ok(n) if (1..=12).contains(&n) => {
                style.grid_column = Line {
                    start: GridPlacement::Span(n),
                    end: GridPlacement::Auto,
                };
                true
            }
            _ => bad(class),
        };
    }
    if let Some(rest) = class.strip_prefix("row-span-") {
        return match rest.parse::<u16>() {
            Ok(n) if (1..=12).contains(&n) => {
                style.grid_row = Line {
                    start: GridPlacement::Span(n),
                    end: GridPlacement::Auto,
                };
                true
            }
            _ => bad(class),
        };
    }

    if let Some(rest) = class.strip_prefix("gap-") {
        let (axis, token) = match rest.split_once('-') {
            Some(("x", t)) => (Some(0), t),
            Some(("y", t)) => (Some(1), t),
            _ => (None, rest),
        };
        let Some(px) = parse_spacing(token) else {
            return bad(class);
        };
        match axis {
            Some(0) => style.gap.width = length(px as f32),
            Some(1) => style.gap.height = length(px as f32),
            _ => {
                style.gap = Size {
                    width: length(px as f32),
                    height: length(px as f32),
                }
            }
        }
        return true;
    }

    // Padding / margin: p-4, px-2, mt-6, …
    for (prefix, is_margin) in [("p", false), ("m", true)] {
        let Some(rest) = class.strip_prefix(prefix) else {
            continue;
        };
        let (sides, token): (&[usize], &str) = match rest.split_at_checked(1) {
            Some(("-", t)) => (&[0, 1, 2, 3], t),
            Some(("x", t)) => (&[1, 3], t.strip_prefix('-').unwrap_or("!")),
            Some(("y", t)) => (&[0, 2], t.strip_prefix('-').unwrap_or("!")),
            Some(("t", t)) => (&[0], t.strip_prefix('-').unwrap_or("!")),
            Some(("r", t)) => (&[1], t.strip_prefix('-').unwrap_or("!")),
            Some(("b", t)) => (&[2], t.strip_prefix('-').unwrap_or("!")),
            Some(("l", t)) => (&[3], t.strip_prefix('-').unwrap_or("!")),
            _ => continue,
        };
        let Some(px) = parse_spacing(token) else {
            // Not a spacing token (e.g. this was never a p-/m- class at
            // all, like "pointer-events-none") — fall through unless it
            // really looked like one.
            if token == "!"
                || token
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit() || c == '[')
            {
                return bad(class);
            }
            continue;
        };
        for &side in sides {
            let target = match side {
                0 => (&mut style.padding.top, &mut style.margin.top),
                1 => (&mut style.padding.right, &mut style.margin.right),
                2 => (&mut style.padding.bottom, &mut style.margin.bottom),
                _ => (&mut style.padding.left, &mut style.margin.left),
            };
            if is_margin {
                *target.1 = LengthPercentageAuto::length(px as f32);
            } else {
                *target.0 = LengthPercentage::length(px as f32);
            }
        }
        return true;
    }

    // Sizes: w-24, h-full, size-8, min-w-0, max-w-sm, basis-1/2 …
    if let Some(rest) = class.strip_prefix("size-") {
        return match parse_dimension(rest) {
            Some(dim) => {
                style.size = Size {
                    width: dim,
                    height: dim,
                };
                true
            }
            None => bad(class),
        };
    }
    if let Some(rest) = class.strip_prefix("basis-") {
        return match parse_dimension(rest) {
            Some(dim) => {
                style.flex_basis = dim;
                true
            }
            None => bad(class),
        };
    }
    if let Some(rest) = class.strip_prefix("max-w-") {
        if let Some(&(_, px)) = MAX_W_NAMED.iter().find(|(n, _)| *n == rest) {
            style.max_size.width = length(px as f32);
            return true;
        }
        return match parse_dimension(rest) {
            Some(dim) => {
                style.max_size.width = dim;
                true
            }
            None => bad(class),
        };
    }
    for (prefix, is_min, horizontal) in [
        ("min-w-", true, true),
        ("min-h-", true, false),
        ("max-h-", false, false),
    ] {
        if let Some(rest) = class.strip_prefix(prefix) {
            return match parse_dimension(rest) {
                Some(dim) => {
                    let target = if is_min {
                        &mut style.min_size
                    } else {
                        &mut style.max_size
                    };
                    if horizontal {
                        target.width = dim;
                    } else {
                        target.height = dim;
                    }
                    true
                }
                None => bad(class),
            };
        }
    }
    for (prefix, horizontal) in [("w-", true), ("h-", false)] {
        if let Some(rest) = class.strip_prefix(prefix) {
            return match parse_dimension(rest) {
                Some(dim) => {
                    if horizontal {
                        style.size.width = dim;
                    } else {
                        style.size.height = dim;
                    }
                    true
                }
                None => bad(class),
            };
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(classes: &[&str]) -> (ResolvedStyle, Vec<String>) {
        let owned: Vec<String> = classes.iter().map(|s| s.to_string()).collect();
        let mut errors = Vec::new();
        let style = resolve_classes(&owned, &TextStyle::default(), false, "div", &mut errors);
        (style, errors)
    }

    #[test]
    fn layout_classes_map_to_taffy() {
        let (s, errors) = resolve(&[
            "flex",
            "flex-col",
            "items-center",
            "justify-between",
            "gap-2",
            "p-4",
            "px-6",
            "w-64",
            "h-full",
            "mx-auto",
            "grow",
        ]);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(s.taffy.display, Display::Flex);
        assert_eq!(s.taffy.flex_direction, FlexDirection::Column);
        assert_eq!(s.taffy.gap.width, length(8.0_f32));
        assert_eq!(s.taffy.padding.top, length(16.0_f32));
        assert_eq!(s.taffy.padding.left, length(24.0_f32)); // px-6 wins over p-4
        assert_eq!(s.taffy.size.width, length(256.0_f32));
        assert_eq!(s.taffy.size.height, percent(1.0_f32));
        assert_eq!(s.taffy.margin.left, auto());
        assert_eq!(s.taffy.flex_grow, 1.0);
    }

    #[test]
    fn grid_and_spans() {
        let (s, errors) = resolve(&["grid", "grid-cols-3", "col-span-2", "gap-x-1"]);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(s.taffy.display, Display::Grid);
        assert_eq!(s.taffy.grid_template_columns.len(), 3);
        assert_eq!(s.taffy.gap.width, length(4.0_f32));
        assert_eq!(s.taffy.gap.height, length(0.0_f32));
    }

    #[test]
    fn text_classes_cascade_and_leading_wins_over_order() {
        let (s, errors) = resolve(&[
            "leading-tight",
            "text-2xl",
            "font-semibold",
            "text-center",
            "tracking-wide",
            "uppercase",
        ]);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(s.text.font_px, 24.0);
        // leading-tight (1.25 * 24), not text-2xl's default 32.
        assert_eq!(s.text.line_height_px, 30.0);
        assert!(s.text.bold);
        assert_eq!(s.text.align, TextAlign::Center);
        assert_eq!(s.text.transform, TextTransform::Upper);
    }

    #[test]
    fn tones_classify_light_dark_white() {
        let (s, errors) = resolve(&["bg-slate-800", "text-white", "border", "border-gray-200"]);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(s.boxed.bg, Some(Tone::Dark));
        assert_eq!(s.text.tone, Tone::White);
        assert_eq!(s.boxed.border_px, [1.0; 4]);
        assert_eq!(s.boxed.border_tone, Tone::Light);

        let (s, _) = resolve(&["bg-amber-100", "text-[#0f172a]", "bg-[#ffffff]"]);
        assert_eq!(s.boxed.bg, Some(Tone::White)); // last one wins
        assert_eq!(s.text.tone, Tone::Dark);
        let (s, _) = resolve(&["bg-amber-100"]);
        assert_eq!(s.boxed.bg, Some(Tone::Light));
    }

    #[test]
    fn arbitrary_values_parse() {
        let (s, errors) = resolve(&["w-[120px]", "p-[0.5rem]", "text-[10px]", "rounded-[3px]"]);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(s.taffy.size.width, length(120.0_f32));
        assert_eq!(s.taffy.padding.top, length(8.0_f32));
        assert_eq!(s.text.font_px, 10.0);
        assert_eq!(s.text.line_height_px, 15.0);
        assert_eq!(s.boxed.radius_px, 3.0);
    }

    #[test]
    fn unknown_and_variant_classes_error_with_context() {
        let (_, errors) = resolve(&["shadow-md", "md:flex", "bg-shiny", "w-[3vw]", "blorp"]);
        let joined = errors.join("\n");
        assert!(!joined.contains("shadow-md"), "no-op listed: {joined}");
        assert!(joined.contains("md: variants"), "{joined}");
        assert!(joined.contains("\"bg-shiny\""), "{joined}");
        assert!(joined.contains("\"w-[3vw]\""), "{joined}");
        assert!(joined.contains("\"blorp\""), "{joined}");
        assert!(joined.contains("div:"), "{joined}");
    }

    #[test]
    fn inline_rejects_layout_but_takes_text() {
        let owned: Vec<String> = ["font-bold", "text-red-600", "p-2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut errors = Vec::new();
        let s = resolve_classes(&owned, &TextStyle::default(), true, "span", &mut errors);
        assert!(s.text.bold);
        assert_eq!(s.text.tone, Tone::Dark);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("inline"), "{}", errors[0]);
    }

    #[test]
    fn non_finite_and_negative_values_error() {
        for class in [
            "border--2",
            "border-NaN",
            "border-inf",
            "w-[infpx]",
            "rounded-[NaNpx]",
            "p-[-4px]",
        ] {
            let (_, errors) = resolve(&[class]);
            assert!(
                errors.iter().any(|e| e.contains(class)),
                "{class} accepted silently: {errors:?}"
            );
        }
    }

    #[test]
    fn borders_and_sides() {
        let (s, errors) = resolve(&["border-2", "border-t-4", "rounded-full"]);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(s.boxed.border_px, [4.0, 2.0, 2.0, 2.0]);
        assert_eq!(s.boxed.radius_px, 1e9);
    }
}
