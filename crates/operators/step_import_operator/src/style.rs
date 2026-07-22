//! STEP presentation styling: STYLED_ITEM chains resolved to per-target
//! surface colors.
//!
//! The chain in the wild (AP203e2/AP214, OCCT/KiCad/SolidWorks exports):
//!
//! ```text
//! STYLED_ITEM(name, (styles...), item)
//!   -> PRESENTATION_STYLE_ASSIGNMENT((SURFACE_STYLE_USAGE, ...))
//!   -> SURFACE_STYLE_USAGE(side, SURFACE_SIDE_STYLE)
//!   -> SURFACE_SIDE_STYLE(name, (SURFACE_STYLE_FILL_AREA | ..._RENDERING))
//!   -> SURFACE_STYLE_FILL_AREA(FILL_AREA_STYLE)
//!   -> FILL_AREA_STYLE(name, (FILL_AREA_STYLE_COLOUR, ...))
//!   -> FILL_AREA_STYLE_COLOUR(name, colour)
//!   -> COLOUR_RGB(name, r, g, b) | DRAUGHTING_PRE_DEFINED_COLOUR(name)
//! ```
//!
//! `item` targets a representation item — for solid color the
//! MANIFOLD_SOLID_BREP, for per-face color the ADVANCED_FACE. Styling is
//! cosmetic, so nothing here fails an import: malformed or unsupported
//! chains are skipped.

use crate::p21::{Arg, DataSection, Record};
use std::collections::{HashMap, HashSet};

/// Resolve every styled item to a color: styled entity id
/// (MANIFOLD_SOLID_BREP, ADVANCED_FACE, ...) → sRGB components in [0, 1].
/// OVER_RIDING_STYLED_ITEM wins over plain STYLED_ITEM; among equal
/// priority the lowest styled-item entity id wins (deterministic
/// regardless of hash order).
pub fn collect_colors(data: &DataSection) -> HashMap<u64, [f32; 3]> {
    let mut ids: Vec<u64> = data.entities.keys().copied().collect();
    ids.sort_unstable();

    let mut colors = HashMap::new();
    let mut overridden = HashSet::new();
    for (pass, record_name) in ["STYLED_ITEM", "OVER_RIDING_STYLED_ITEM"]
        .into_iter()
        .enumerate()
    {
        for &id in &ids {
            let entity = &data.entities[&id];
            let Some(record) = entity.record(record_name) else {
                continue;
            };
            let Some((target, color)) = styled_item_color(data, record) else {
                continue;
            };
            if pass == 0 {
                colors.entry(target).or_insert(color);
            } else if overridden.insert(target) {
                colors.insert(target, color);
            }
        }
    }
    colors
}

/// `(styled target id, color)` of one STYLED_ITEM record, if any of its
/// styles resolves to a surface color.
fn styled_item_color(data: &DataSection, record: &Record) -> Option<(u64, [f32; 3])> {
    let target = record.args.get(2)?.as_ref_id()?;
    let styles = record.args.get(1)?.as_list()?;
    let color = styles
        .iter()
        .find_map(|style| presentation_style_color(data, style))?;
    Some((target, color))
}

/// Surface color of one PRESENTATION_STYLE_ASSIGNMENT (or its
/// PRESENTATION_STYLE_BY_CONTEXT subtype).
fn presentation_style_color(data: &DataSection, style: &Arg) -> Option<[f32; 3]> {
    let entity = data.deref(style).ok()?;
    let record = entity
        .records
        .iter()
        .find(|r| r.name.starts_with("PRESENTATION_STYLE"))?;
    record
        .args
        .first()?
        .as_list()?
        .iter()
        .find_map(|item| surface_style_color(data, item))
}

/// Surface color of one presentation-style select entry (only
/// SURFACE_STYLE_USAGE carries surface colors; curve/point styles are
/// ignored).
fn surface_style_color(data: &DataSection, item: &Arg) -> Option<[f32; 3]> {
    let entity = data.deref(item).ok()?;
    let usage = entity.record("SURFACE_STYLE_USAGE")?;
    let side_style = data.deref(usage.args.get(1)?).ok()?;
    let side = side_style.record("SURFACE_SIDE_STYLE")?;
    side.args
        .get(1)?
        .as_list()?
        .iter()
        .find_map(|element| surface_style_element_color(data, element))
}

/// Color of one SURFACE_SIDE_STYLE element: fill-area styling, with
/// SURFACE_STYLE_RENDERING(_WITH_PROPERTIES) as a fallback (its second
/// argument is the surface colour directly).
fn surface_style_element_color(data: &DataSection, element: &Arg) -> Option<[f32; 3]> {
    let entity = data.deref(element).ok()?;
    if let Some(fill_area) = entity.record("SURFACE_STYLE_FILL_AREA") {
        let fill = data.deref(fill_area.args.first()?).ok()?;
        let style = fill.record("FILL_AREA_STYLE")?;
        return style.args.get(1)?.as_list()?.iter().find_map(|fill_style| {
            let e = data.deref(fill_style).ok()?;
            let colour_record = e.record("FILL_AREA_STYLE_COLOUR")?;
            colour(data, colour_record.args.get(1)?)
        });
    }
    let rendering = entity
        .records
        .iter()
        .find(|r| r.name.starts_with("SURFACE_STYLE_RENDERING"))?;
    colour(data, rendering.args.get(1)?)
}

/// Resolve a colour entity: COLOUR_RGB, or the (draughting) pre-defined
/// colour name table.
fn colour(data: &DataSection, arg: &Arg) -> Option<[f32; 3]> {
    let entity = data.deref(arg).ok()?;
    if let Some(rgb) = entity.record("COLOUR_RGB") {
        let component = |i: usize| -> Option<f32> {
            let v = rgb.args.get(i)?.as_f64()?;
            v.is_finite().then_some(v.clamp(0.0, 1.0) as f32)
        };
        return Some([component(1)?, component(2)?, component(3)?]);
    }
    let predefined = entity
        .records
        .iter()
        .find(|r| r.name.ends_with("PRE_DEFINED_COLOUR"))?;
    match predefined.args.first()?.as_str()? {
        "red" => Some([1.0, 0.0, 0.0]),
        "green" => Some([0.0, 1.0, 0.0]),
        "blue" => Some([0.0, 0.0, 1.0]),
        "yellow" => Some([1.0, 1.0, 0.0]),
        "magenta" => Some([1.0, 0.0, 1.0]),
        "cyan" => Some([0.0, 1.0, 1.0]),
        "black" => Some([0.0, 0.0, 0.0]),
        "white" => Some([1.0, 1.0, 1.0]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p21;

    /// A minimal DATA section with one full chain per target kind.
    const STYLES: &str = r#"ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#10 = MANIFOLD_SOLID_BREP('',#99);
#11 = ADVANCED_FACE('',(#99),#99,.T.);
#99 = CLOSED_SHELL('',());

#20 = COLOUR_RGB('',0.25,0.5,1.);
#21 = FILL_AREA_STYLE_COLOUR('',#20);
#22 = FILL_AREA_STYLE('',(#21));
#23 = SURFACE_STYLE_FILL_AREA(#22);
#24 = SURFACE_SIDE_STYLE('',(#23));
#25 = SURFACE_STYLE_USAGE(.BOTH.,#24);
#26 = PRESENTATION_STYLE_ASSIGNMENT((#25));
#27 = STYLED_ITEM('color',(#26),#10);

#30 = DRAUGHTING_PRE_DEFINED_COLOUR('green');
#31 = FILL_AREA_STYLE_COLOUR('',#30);
#32 = FILL_AREA_STYLE('',(#31));
#33 = SURFACE_STYLE_FILL_AREA(#32);
#34 = SURFACE_SIDE_STYLE('',(#33));
#35 = SURFACE_STYLE_USAGE(.BOTH.,#34);
#36 = PRESENTATION_STYLE_ASSIGNMENT((#35));
#37 = STYLED_ITEM('color',(#36),#11);
ENDSEC;
END-ISO-10303-21;
"#;

    #[test]
    fn resolves_rgb_and_predefined_chains() {
        let data = p21::parse(STYLES).unwrap();
        let colors = collect_colors(&data);
        assert_eq!(colors.get(&10), Some(&[0.25, 0.5, 1.0]));
        assert_eq!(colors.get(&11), Some(&[0.0, 1.0, 0.0]));
        assert_eq!(colors.len(), 2);
    }

    #[test]
    fn malformed_chains_are_skipped() {
        // Style list referencing a dangling entity: no color, no panic.
        let text = r#"ISO-10303-21;
DATA;
#1 = STYLED_ITEM('color',(#999),#2);
#2 = MANIFOLD_SOLID_BREP('',#3);
#3 = CLOSED_SHELL('',());
ENDSEC;
END-ISO-10303-21;
"#;
        let data = p21::parse(text).unwrap();
        assert!(collect_colors(&data).is_empty());
    }
}
