//! Integration tests for text_model_operator: config text → filled 2D
//! outline model, composing with extrude_operator for 3D solids.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use ciborium::value::Value;
use volumetric::wasm::{NativeModelExecutor, OperatorExecutor, OperatorIo, create_operator_executor};

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing wasm artifact {} ({e}); build it with `cargo build-wasm`",
            path.display()
        )
    })
}

fn run_operator(operator: &str, inputs: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    let operator_wasm = wasm_artifact(operator);
    let mut executor = create_operator_executor(&operator_wasm).expect("create operator executor");
    let result = executor
        .run(OperatorIo::new(inputs))
        .map_err(|e| e.to_string())?;
    result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "operator posted no output".to_string())
}

fn cbor_map(fields: &[(&str, Value)]) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &Value::Map(
            fields
                .iter()
                .map(|(k, v)| (Value::Text((*k).into()), v.clone()))
                .collect(),
        ),
        &mut out,
    )
    .unwrap();
    out
}

fn text_config(text: &str, extra: &[(&str, Value)]) -> Vec<u8> {
    let mut fields = vec![("text", Value::Text(text.into()))];
    fields.extend(extra.iter().cloned());
    cbor_map(&fields)
}

fn text_model(text: &str, extra: &[(&str, Value)]) -> Vec<u8> {
    run_operator("text_model_operator", vec![text_config(text, extra)]).expect("text converts")
}

fn bounds_2d(executor: &mut NativeModelExecutor) -> [f64; 4] {
    let bounds = executor.get_bounds_nd().unwrap();
    [bounds.min(0), bounds.max(0), bounds.min(1), bounds.max(1)]
}

#[test]
fn text_renders_as_a_centered_2d_model() {
    let model = text_model("I", &[]);
    let mut executor = NativeModelExecutor::new(&model).unwrap();
    assert_eq!(executor.dimensions(), 2);

    // Default anchor "center": the tight bounding box centers on the
    // origin; a capital at em size 1.0 stands roughly 0.7 tall.
    let [min_x, max_x, min_y, max_y] = bounds_2d(&mut executor);
    assert!((min_x + max_x).abs() < 1e-9, "x not centered: [{min_x}, {max_x}]");
    assert!((min_y + max_y).abs() < 1e-9, "y not centered: [{min_y}, {max_y}]");
    let (w, h) = (max_x - min_x, max_y - min_y);
    assert!((0.5..0.9).contains(&h), "cap height {h}");
    assert!(w < h, "an I should be narrow, got {w}x{h}");

    // The stem is solid at the origin; beside the glyph is empty.
    let sample = |e: &mut NativeModelExecutor, x: f64, y: f64| e.sample_nd(&[x, y]).unwrap();
    assert_eq!(sample(&mut executor, 0.0, 0.0), 1.0);
    assert_eq!(sample(&mut executor, max_x + 0.1, 0.0), 0.0);
    assert_eq!(sample(&mut executor, 0.0, max_y + 0.1), 0.0);
}

#[test]
fn letter_o_keeps_its_hole() {
    let model = text_model("O", &[]);
    let mut executor = NativeModelExecutor::new(&model).unwrap();
    let [min_x, _, min_y, max_y] = bounds_2d(&mut executor);
    let h = max_y - min_y;

    let sample = |e: &mut NativeModelExecutor, x: f64, y: f64| e.sample_nd(&[x, y]).unwrap();
    // Counter (the hole) is empty; the rim just inside the top and left
    // edges is solid — nonzero winding with the inner contour wound
    // opposite the outer.
    assert_eq!(sample(&mut executor, 0.0, 0.0), 0.0);
    assert_eq!(sample(&mut executor, 0.0, max_y - 0.04 * h), 1.0);
    assert_eq!(sample(&mut executor, min_x + 0.04 * h, 0.0), 1.0);
}

#[test]
fn multiline_right_align_hangs_from_the_baseline() {
    let model = text_model(
        "Hi\nWorld",
        &[
            ("align", Value::Text("right".into())),
            ("anchor", Value::Text("baseline".into())),
        ],
    );
    let mut executor = NativeModelExecutor::new(&model).unwrap();
    let [min_x, max_x, min_y, max_y] = bounds_2d(&mut executor);

    // Right-aligned at the origin: outlines end essentially at x = 0.
    assert!(max_x <= 0.05, "right edge at {max_x}");
    assert!(min_x < -1.0, "\"World\" should reach well left, got {min_x}");
    // First line's caps rise above its baseline at y = 0; the second line
    // sits one line_height (default 1.2 em) below, with no descenders.
    assert!((0.5..0.9).contains(&max_y), "cap top {max_y}");
    assert!((-1.3..-1.1).contains(&min_y), "second baseline {min_y}");
}

#[test]
fn letter_spacing_widens_by_the_gap_count() {
    let width = |extra: &[(&str, Value)]| {
        let model = text_model("AB", extra);
        let mut executor = NativeModelExecutor::new(&model).unwrap();
        let [min_x, max_x, ..] = bounds_2d(&mut executor);
        max_x - min_x
    };
    let base = width(&[]);
    let spaced = width(&[("letter_spacing", Value::Float(0.5))]);
    // One inter-glyph gap in "AB": exactly 0.5 em wider.
    assert!((spaced - base - 0.5).abs() < 1e-9, "base {base}, spaced {spaced}");
}

#[test]
fn explicit_font_blob_matches_the_embedded_default() {
    let font = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("crates/operators/text_model_operator/fonts/LiberationSans-Regular.ttf"),
    )
    .expect("embedded font file");
    let with_blob = run_operator(
        "text_model_operator",
        vec![text_config("Volumetric", &[]), font],
    )
    .expect("text converts with an explicit font");
    assert_eq!(with_blob, text_model("Volumetric", &[]));
}

#[test]
fn garbage_font_is_rejected() {
    let err = run_operator(
        "text_model_operator",
        vec![text_config("Hi", &[]), b"not a font".to_vec()],
    )
    .expect_err("garbage font bytes must fail");
    assert!(err.contains("font"), "{err}");
}

#[test]
fn unsupported_characters_are_reported() {
    let err = run_operator(
        "text_model_operator",
        vec![text_config("a\u{2603}b", &[])],
    )
    .expect_err("snowman is not in Liberation Sans");
    assert!(err.contains('\u{2603}'), "{err}");
}

#[test]
fn whitespace_only_text_is_rejected() {
    let err = run_operator("text_model_operator", vec![text_config("  \n ", &[])])
        .expect_err("whitespace renders nothing");
    assert!(err.contains("no geometry"), "{err}");
}

#[test]
fn invalid_config_values_are_rejected() {
    for (field, value, want) in [
        ("size", Value::Float(0.0), "size"),
        ("align", Value::Text("justified".into()), "align"),
        ("anchor", Value::Text("top".into()), "anchor"),
        ("line_height", Value::Float(-1.0), "line_height"),
    ] {
        let err = run_operator(
            "text_model_operator",
            vec![text_config("Hi", &[(field, value)])],
        )
        .expect_err(field);
        assert!(err.contains(want), "{field}: {err}");
    }
}

#[test]
fn text_extrudes_to_a_3d_solid() {
    let extruded = run_operator(
        "extrude_operator",
        vec![
            text_model("T", &[]),
            cbor_map(&[("height", Value::Float(0.5))]),
        ],
    )
    .expect("extrude text");
    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    assert_eq!(executor.dimensions(), 3);

    // The T's stem crosses the origin; mid-depth is solid, above the slab
    // and beside the glyph are empty.
    let sample = |e: &mut NativeModelExecutor, p: [f64; 3]| e.sample_nd(&p).unwrap();
    assert_eq!(sample(&mut executor, [0.0, 0.0, 0.25]), 1.0);
    assert_eq!(sample(&mut executor, [0.0, 0.0, 0.6]), 0.0);
    assert_eq!(sample(&mut executor, [2.0, 0.0, 0.25]), 0.0);
}
