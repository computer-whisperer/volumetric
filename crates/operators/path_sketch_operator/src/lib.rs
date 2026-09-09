//! Path Sketch Operator: SVG path data → a filled 2D sketch model.
//!
//! The polygon primitive of the construction catalog. `sample(x, y)` is
//! 1.0 inside the drawn outline and 0.0 outside, at any resolution: the
//! path's curves and arcs are flattened to chords here at conversion time
//! (see `outline_model_core::flatten`) and the generated model classifies
//! points exactly against those segments by nonzero winding, exactly as
//! `text_model_operator` does for glyphs. The output composes like any 2D
//! sketch: extrude, revolve, intersect, offset.
//!
//! Inputs:
//! - Input 0: CBOR config
//!   - `path` (SVG `d` syntax, metres, y up): see [`path`] for the grammar
//!   - `flip_y` (default false): mirror about the x axis, for path data
//!     authored on a y-down SVG canvas
//!   - `round` (metres, default 0): fillet radius applied to every corner
//!     between two straight segments, clamped to half the shorter edge
//!   - `chord_tolerance` (metres, default 0 = automatic, 1e-4 of the
//!     bounding-box diagonal): maximum chord deviation when flattening
//!
//! Output 0: ModelWASM (2D).
//!
//! The README (surfaced as the operator's docs) covers the syntax and
//! conventions. The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p outline_model_template
//! cp target/wasm32-unknown-unknown/release/outline_model_template.wasm \
//!    crates/operators/path_sketch_operator/template/
//! ```

pub mod path;

use std::f64::consts::PI;

use outline_model_core::flatten;
use path::{Piece, Point, Subpath};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/outline_model_template.wasm");

/// A 10 mm square: what a freshly added step draws.
pub const DEFAULT_PATH: &str = "M0 0 H0.01 V0.01 H0 Z";

/// Automatic chord tolerance as a fraction of the bounding-box diagonal.
const AUTO_TOLERANCE_FRACTION: f64 = 1e-4;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct SketchConfig {
    pub path: String,
    pub flip_y: bool,
    pub round: f64,
    pub chord_tolerance: f64,
}

impl Default for SketchConfig {
    fn default() -> Self {
        Self {
            path: DEFAULT_PATH.to_string(),
            flip_y: false,
            round: 0.0,
            chord_tolerance: 0.0,
        }
    }
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}

fn length(v: Point) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Points of an SVG endpoint-parameterised arc from `from`, appended after
/// `from` and ending exactly on `end` (SVG 1.1 appendix F.6.5: endpoint to
/// centre conversion, radii scaled up when the endpoints are too far
/// apart). Zero radii degrade to a straight line, as the spec says.
#[allow(clippy::too_many_arguments)]
fn arc_points(
    from: Point,
    rx: f64,
    ry: f64,
    rotation_deg: f64,
    large: bool,
    sweep: bool,
    end: Point,
    tol: f64,
    out: &mut Vec<Point>,
) {
    if from == end {
        return;
    }
    let (mut rx, mut ry) = (rx.abs(), ry.abs());
    if rx == 0.0 || ry == 0.0 {
        out.push(end);
        return;
    }
    let phi = rotation_deg.to_radians();
    let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
    let dx = (from[0] - end[0]) / 2.0;
    let dy = (from[1] - end[1]) / 2.0;
    let x1 = cos_phi * dx + sin_phi * dy;
    let y1 = -sin_phi * dx + cos_phi * dy;
    let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry);
    if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
    }
    let num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1;
    let den = rx * rx * y1 * y1 + ry * ry * x1 * x1;
    let sign = if large != sweep { 1.0 } else { -1.0 };
    let coef = if den > 0.0 {
        sign * (num / den).max(0.0).sqrt()
    } else {
        0.0
    };
    let cx1 = coef * rx * y1 / ry;
    let cy1 = coef * -(ry * x1 / rx);
    let cx = cos_phi * cx1 - sin_phi * cy1 + (from[0] + end[0]) / 2.0;
    let cy = sin_phi * cx1 + cos_phi * cy1 + (from[1] + end[1]) / 2.0;
    let (ux, uy) = ((x1 - cx1) / rx, (y1 - cy1) / ry);
    let (vx, vy) = ((-x1 - cx1) / rx, (-y1 - cy1) / ry);
    let theta1 = uy.atan2(ux);
    let mut dtheta = (ux * vy - uy * vx).atan2(ux * vx + uy * vy);
    if !sweep && dtheta > 0.0 {
        dtheta -= 2.0 * PI;
    } else if sweep && dtheta < 0.0 {
        dtheta += 2.0 * PI;
    }
    flatten::arc([cx, cy], rx, ry, phi, theta1, dtheta, tol, out);
    if let Some(last) = out.last_mut() {
        *last = end;
    }
}

/// Flatten one subpath into a closed polyline. Returns the points and, per
/// point, whether the piece arriving at it was a straight segment (index 0
/// gets the closing piece), which is what corner rounding keys on.
fn subpath_polyline(sp: &Subpath, tol: f64) -> (Vec<Point>, Vec<bool>) {
    let mut points = vec![sp.start];
    let mut straight = vec![true];
    let mut pos = sp.start;
    for piece in &sp.pieces {
        let before = points.len();
        match *piece {
            Piece::Line(end) => points.push(end),
            Piece::Quad { ctrl, end } => flatten::quad(pos, ctrl, end, tol, &mut points),
            Piece::Cubic { c1, c2, end } => flatten::cubic(pos, c1, c2, end, tol, &mut points),
            Piece::Arc {
                rx,
                ry,
                rotation_deg,
                large,
                sweep,
                end,
            } => arc_points(
                pos,
                rx,
                ry,
                rotation_deg,
                large,
                sweep,
                end,
                tol,
                &mut points,
            ),
        }
        let added = points.len() - before;
        straight.extend(std::iter::repeat_n(piece.is_line(), added));
        pos = piece.end();
    }
    // Drop repeated points (zero-length pieces); the survivor keeps its
    // own arrival flag.
    let mut i = 1;
    while i < points.len() {
        if points[i] == points[i - 1] {
            points.remove(i);
            straight.remove(i);
        } else {
            i += 1;
        }
    }
    // Close: a path that returns to its start supplies the closing piece
    // itself; otherwise the implicit closing segment is straight.
    if points.len() > 1 && points[points.len() - 1] == points[0] {
        points.pop();
        straight[0] = straight.pop().unwrap_or(true);
    } else {
        straight[0] = true;
    }
    (points, straight)
}

/// Replace every corner between two straight segments by a tangent arc of
/// radius `r`, clamped so the tangent length never exceeds half of either
/// adjacent edge (neighbouring fillets then meet instead of overlapping).
fn round_corners(points: &[Point], straight: &[bool], r: f64, tol: f64) -> Vec<Point> {
    let n = points.len();
    if n < 3 || r <= 0.0 {
        return points.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = points[i];
        if !(straight[i] && straight[(i + 1) % n]) {
            out.push(b);
            continue;
        }
        let a = points[(i + n - 1) % n];
        let c = points[(i + 1) % n];
        let (ab, cb) = (sub(a, b), sub(c, b));
        let (la, lc) = (length(ab), length(cb));
        if la <= 0.0 || lc <= 0.0 {
            out.push(b);
            continue;
        }
        let u1 = [ab[0] / la, ab[1] / la];
        let u2 = [cb[0] / lc, cb[1] / lc];
        let theta = (u1[0] * u2[0] + u1[1] * u2[1]).clamp(-1.0, 1.0).acos();
        if !(1e-6..=PI - 1e-6).contains(&theta) {
            out.push(b);
            continue;
        }
        let half = theta / 2.0;
        let mut t = r / half.tan();
        let t_max = la.min(lc) / 2.0;
        let radius = if t > t_max {
            t = t_max;
            t * half.tan()
        } else {
            r
        };
        let p1 = [b[0] + u1[0] * t, b[1] + u1[1] * t];
        let p2 = [b[0] + u2[0] * t, b[1] + u2[1] * t];
        let bisector = [u1[0] + u2[0], u1[1] + u2[1]];
        let bl = length(bisector);
        let d = radius / half.sin();
        let center = [b[0] + bisector[0] / bl * d, b[1] + bisector[1] / bl * d];
        let a1 = (p1[1] - center[1]).atan2(p1[0] - center[0]);
        let a2 = (p2[1] - center[1]).atan2(p2[0] - center[0]);
        let mut sweep = a2 - a1;
        while sweep > PI {
            sweep -= 2.0 * PI;
        }
        while sweep <= -PI {
            sweep += 2.0 * PI;
        }
        out.push(p1);
        flatten::arc(center, radius, radius, 0.0, a1, sweep, tol, &mut out);
    }
    out
}

/// Bounding box of every point the path names (ends and control points;
/// arcs contribute their end points), for the automatic tolerance.
fn control_diagonal(subpaths: &[Subpath]) -> f64 {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    let mut take = |p: Point| {
        for axis in 0..2 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    };
    for sp in subpaths {
        take(sp.start);
        for piece in &sp.pieces {
            match *piece {
                Piece::Line(end) | Piece::Arc { end, .. } => take(end),
                Piece::Quad { ctrl, end } => {
                    take(ctrl);
                    take(end);
                }
                Piece::Cubic { c1, c2, end } => {
                    take(c1);
                    take(c2);
                    take(end);
                }
            }
        }
    }
    length(sub(hi, lo))
}

/// The sketch's closed contours in metres, y up (after `flip_y`), ready
/// for `outline_model_core::build_payload`.
pub fn sketch_contours(cfg: &SketchConfig) -> Result<Vec<Vec<Point>>, String> {
    if !(cfg.round.is_finite() && cfg.round >= 0.0) {
        return Err(format!(
            "round must be a finite length >= 0, got {}",
            cfg.round
        ));
    }
    if !(cfg.chord_tolerance.is_finite() && cfg.chord_tolerance >= 0.0) {
        return Err(format!(
            "chord_tolerance must be a finite length >= 0, got {}",
            cfg.chord_tolerance
        ));
    }
    let subpaths = path::parse(&cfg.path)?;
    let tol = if cfg.chord_tolerance > 0.0 {
        cfg.chord_tolerance
    } else {
        (control_diagonal(&subpaths) * AUTO_TOLERANCE_FRACTION).max(1e-9)
    };

    let mut contours = Vec::with_capacity(subpaths.len());
    for (index, sp) in subpaths.iter().enumerate() {
        let (points, straight) = subpath_polyline(sp, tol);
        if points.len() < 3 {
            return Err(format!(
                "subpath {} has only {} distinct point(s); a filled contour needs at least 3",
                index + 1,
                points.len()
            ));
        }
        let mut contour = round_corners(&points, &straight, cfg.round, tol);
        if cfg.flip_y {
            for p in &mut contour {
                p[1] = -p[1];
            }
        }
        contours.push(contour);
    }
    Ok(contours)
}

/// The outline payload for `cfg` (see `outline_model_core`).
pub fn sketch_payload(cfg: &SketchConfig) -> Result<Vec<u8>, String> {
    let contours = sketch_contours(cfg)?;
    outline_model_core::build_payload(&contours)
}

fn generate(cfg: &SketchConfig) -> Result<Vec<u8>, String> {
    let payload = sketch_payload(cfg)?;
    outline_model_core::emit::patch_template(TEMPLATE, &payload)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg = {
        let cfg_buf = read_input(0);
        if cfg_buf.is_empty() {
            SketchConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<SketchConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    match generate(&cfg) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("path sketch failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = format!(
            "{{ path: tstr .default \"{DEFAULT_PATH}\", flip_y: bool .default false, \
             round: float .default 0.0, chord_tolerance: float .default 0.0 }}"
        );
        OperatorMetadata {
            name: "path_sketch_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Path Sketch".to_string(),
            description: "Fill SVG path data (lines, curves, arcs, holes) as a 2D sketch, ready to extrude or revolve."
                .to_string(),
            category: "Primitives".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M4 20 8 8"/>"##,
                r##"<path d="M8 8c3-5 7-3 12 2"/>"##,
                r##"<circle cx="8" cy="8" r="2"/>"##,
                r##"<path d="M20 10v10H4"/>"##,
            )
            .to_string(),
            inputs: vec![OperatorMetadataInput::CBORConfiguration(schema)],
            variadic_input: None,
            input_names: vec!["Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use outline_model_core::PayloadView;

    fn cfg(path: &str) -> SketchConfig {
        SketchConfig {
            path: path.to_string(),
            ..SketchConfig::default()
        }
    }

    /// Classifier over the sketch's built payload.
    fn classifier(cfg: &SketchConfig) -> impl Fn(f64, f64) -> bool + use<> {
        let payload = sketch_payload(cfg).unwrap();
        move |x, y| PayloadView::new(&payload).unwrap().contains(x, y)
    }

    #[test]
    fn square_classifies_inside_and_outside() {
        let inside = classifier(&cfg("M0 0 H1 V1 H0 Z"));
        assert!(inside(0.5, 0.5));
        assert!(!inside(1.5, 0.5));
        assert!(!inside(-0.1, 0.5));
        assert!(!inside(0.5, -0.1));
    }

    #[test]
    fn default_config_draws_a_centimetre_square() {
        let inside = classifier(&SketchConfig::default());
        assert!(inside(0.005, 0.005));
        assert!(!inside(0.011, 0.005));
    }

    #[test]
    fn round_trims_corners_and_keeps_the_flats() {
        let mut c = cfg("M0 0 H1 V1 H0 Z");
        c.round = 0.2;
        let inside = classifier(&c);
        assert!(inside(0.5, 0.5));
        assert!(inside(0.3, 0.01), "flat bottom edge is untouched");
        assert!(
            inside(0.1, 0.1),
            "inside the fillet arc (0.14 from its centre)"
        );
        assert!(
            !inside(0.05, 0.05),
            "outside the fillet arc (0.21 from its centre)"
        );
        assert!(!inside(0.02, 0.02), "the old corner is gone");
        assert!(!inside(0.98, 0.98), "every corner is filleted");
    }

    #[test]
    fn round_is_clamped_to_half_the_shorter_edge() {
        let mut c = cfg("M0 0 H1 V0.2 H0 Z");
        c.round = 1.0;
        let inside = classifier(&c);
        // The short ends become full semicircles of radius 0.1: a stadium.
        assert!(inside(0.5, 0.1));
        assert!(inside(0.005, 0.1));
        assert!(!inside(0.005, 0.005));
        assert!(!inside(0.995, 0.195));
    }

    #[test]
    fn opposite_winding_subpath_cuts_a_hole_and_same_winding_unions() {
        let inside = classifier(&cfg("M0 0 H1 V1 H0 Z M0.25 0.25 V0.75 H0.75 V0.25 Z"));
        assert!(!inside(0.5, 0.5), "hole");
        assert!(inside(0.1, 0.5), "ring");
        let inside = classifier(&cfg("M0 0 H1 V1 H0 Z M0.25 0.25 H0.75 V0.75 H0.25 Z"));
        assert!(inside(0.5, 0.5), "same winding fills");
    }

    #[test]
    fn arc_sweep_flag_picks_the_side_and_points_lie_on_the_circle() {
        // Semicircle from (0,0) to (2,0) about (1,0): sweep=1 turns
        // counterclockwise in this y-up frame, so it bulges to -y.
        let c = cfg("M0 0 A1 1 0 0 1 2 0 Z");
        let contour = &sketch_contours(&c).unwrap()[0];
        assert!(contour.len() > 10);
        for p in contour {
            let d = ((p[0] - 1.0).powi(2) + p[1].powi(2)).sqrt();
            assert!((d - 1.0).abs() < 1e-9, "{p:?} is off the circle");
        }
        let inside = classifier(&c);
        assert!(inside(1.0, -0.5));
        assert!(!inside(1.0, 0.5));
        let inside = classifier(&cfg("M0 0 A1 1 0 0 0 2 0 Z"));
        assert!(inside(1.0, 0.5));
        assert!(!inside(1.0, -0.5));
    }

    #[test]
    fn large_arc_flag_takes_the_long_way() {
        // Chord (1,0)->(0,1) on a unit circle, counterclockwise: the small
        // arc is the quarter turn about (0,0) and the closed shape is the
        // thin circular segment above the chord; the large arc is the
        // three-quarter turn about (1,1) and the shape is most of that disc.
        let small = classifier(&cfg("M1 0 A1 1 0 0 1 0 1 Z"));
        let large = classifier(&cfg("M1 0 A1 1 0 1 1 0 1 Z"));
        assert!(small(0.6, 0.6));
        assert!(!small(1.5, 1.5));
        assert!(!small(0.3, 0.3), "below the chord");
        assert!(large(0.6, 0.6));
        assert!(large(1.5, 1.5));
        assert!(!large(0.3, 0.3), "below the chord");
        assert!(!large(1.0, 2.5), "outside the disc");
    }

    #[test]
    fn flip_y_mirrors_the_sketch_and_the_sweep_direction() {
        let mut c = cfg("M0 0 H1 V1 H0 Z");
        c.flip_y = true;
        let inside = classifier(&c);
        assert!(inside(0.5, -0.5));
        assert!(!inside(0.5, 0.5));
        let mut c = cfg("M0 0 A1 1 0 0 1 2 0 Z");
        c.flip_y = true;
        let inside = classifier(&c);
        assert!(
            inside(1.0, 0.5),
            "on a y-down canvas sweep=1 bulged this way"
        );
    }

    #[test]
    fn automatic_tolerance_is_relative_to_the_sketch_size() {
        let metre = sketch_contours(&cfg("M0 0 A1 1 0 0 1 2 0 Z")).unwrap();
        let millimetre = sketch_contours(&cfg("M0 0 A0.001 0.001 0 0 1 0.002 0 Z")).unwrap();
        assert_eq!(metre[0].len(), millimetre[0].len());
        let mut fine = cfg("M0 0 A1 1 0 0 1 2 0 Z");
        fine.chord_tolerance = 1e-6;
        assert!(sketch_contours(&fine).unwrap()[0].len() > metre[0].len());
    }

    #[test]
    fn curves_flatten_into_the_outline() {
        // A cubic bulge over a straight base.
        let inside = classifier(&cfg("M0 0 C 0 1 1 1 1 0 Z"));
        assert!(inside(0.5, 0.3));
        assert!(!inside(0.5, 0.8));
        assert!(!inside(0.5, -0.1));
    }

    #[test]
    fn degenerate_and_invalid_configs_are_errors() {
        let err = sketch_contours(&cfg("M0 0 L1 1")).unwrap_err();
        assert!(err.contains("subpath 1"), "{err}");
        let mut c = cfg("M0 0 H1 V1 H0 Z");
        c.round = -1.0;
        assert!(sketch_contours(&c).unwrap_err().contains("round"));
        c.round = 0.0;
        c.chord_tolerance = f64::NAN;
        assert!(sketch_contours(&c).unwrap_err().contains("chord_tolerance"));
    }

    #[test]
    fn emitted_module_has_the_model_abi() {
        let wasm = generate(&SketchConfig::default()).unwrap();
        let module = walrus::Module::from_buffer(&wasm).expect("emitted wasm parses");
        let names: Vec<&str> = module.exports.iter().map(|e| e.name.as_str()).collect();
        for required in [
            "sample",
            "get_bounds",
            "get_dimensions",
            "get_io_ptr",
            "memory",
        ] {
            assert!(
                names.contains(&required),
                "missing export {required}: {names:?}"
            );
        }
        assert!(!names.contains(&"outline_payload_slot"));
    }
}
