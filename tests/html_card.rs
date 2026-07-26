//! Integration tests for html_card_operator: HTML+Tailwind blob → ink and
//! plate 2D outline models in one shared, centered coordinate frame.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use ciborium::value::Value;
use volumetric::wasm::{
    NativeModelExecutor, OperatorExecutor, OperatorIo, create_operator_executor,
};

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

/// Runs the operator and returns ALL outputs (indexed).
fn run_operator_all(
    operator: &str,
    inputs: Vec<Vec<u8>>,
) -> Result<std::collections::HashMap<usize, Vec<u8>>, String> {
    let operator_wasm = wasm_artifact(operator);
    let mut executor = create_operator_executor(&operator_wasm).expect("create operator executor");
    let result = executor
        .run(OperatorIo::new(inputs))
        .map_err(|e| e.to_string())?;
    Ok(result.outputs)
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

const CARD: &str = r#"
<div class="p-6 border-2 rounded-xl">
  <div class="flex justify-between items-center">
    <p class="text-2xl font-bold uppercase">Part A-113</p>
    <div class="bg-black rounded px-2 py-1"><p class="text-white text-sm">Rev 4</p></div>
  </div>
  <hr class="border-t-2"/>
  <div class="grid grid-cols-2 gap-2">
    <p class="text-sm">Material</p>
    <p class="text-sm text-right font-bold">PETG</p>
  </div>
</div>"#;

fn card_models(html: &str, config: &[(&str, Value)]) -> (Vec<u8>, Vec<u8>) {
    let outputs = run_operator_all(
        "html_card_operator",
        vec![html.as_bytes().to_vec(), cbor_map(config)],
    )
    .expect("card renders");
    (
        outputs.get(&0).expect("ink output").clone(),
        outputs.get(&1).expect("plate output").clone(),
    )
}

fn bounds_2d(executor: &mut NativeModelExecutor) -> [f64; 4] {
    let bounds = executor.get_bounds_nd().unwrap();
    [bounds.min(0), bounds.max(0), bounds.min(1), bounds.max(1)]
}

#[test]
fn card_produces_registered_ink_and_plate() {
    let (ink, plate) = card_models(CARD, &[]);

    let mut plate_exec = NativeModelExecutor::new(&plate).unwrap();
    assert_eq!(plate_exec.dimensions(), 2);
    let [p_min_x, p_max_x, p_min_y, p_max_y] = bounds_2d(&mut plate_exec);
    // Default config: model width 1.0, centered.
    assert!((p_max_x - 0.5).abs() < 1e-9 && (p_min_x + 0.5).abs() < 1e-9);
    assert!((p_min_y + p_max_y).abs() < 1e-9);
    // Plate is solid at the center, empty outside the rounded corner.
    assert_eq!(plate_exec.sample_nd(&[0.0, 0.0]).unwrap(), 1.0);
    assert_eq!(plate_exec.sample_nd(&[p_min_x, p_min_y]).unwrap(), 0.0);

    let mut ink_exec = NativeModelExecutor::new(&ink).unwrap();
    assert_eq!(ink_exec.dimensions(), 2);
    let [i_min_x, i_max_x, i_min_y, i_max_y] = bounds_2d(&mut ink_exec);
    // Ink stays within the plate (shared frame, registration holds).
    assert!(i_min_x >= p_min_x - 1e-9 && i_max_x <= p_max_x + 1e-9);
    assert!(i_min_y >= p_min_y - 1e-9 && i_max_y <= p_max_y + 1e-9);
    // The border ring is ink: sample just inside the left edge, mid-card.
    let x_border = p_min_x + 1.0 / 384.0;
    assert_eq!(ink_exec.sample_nd(&[x_border, 0.0]).unwrap(), 1.0);
}

#[test]
fn dark_chip_knocks_out_light_text() {
    let html =
        r#"<div class="p-4"><div class="bg-black p-4"><p class="text-white">OK</p></div></div>"#;
    let (ink, _) = card_models(html, &[]);
    let mut exec = NativeModelExecutor::new(&ink).unwrap();

    // px → model helpers (width 1.0 over 384px; card is 88px tall).
    let h_px = 88.0;
    let to_model = |x_px: f64, y_px: f64| [(x_px - 192.0) / 384.0, (h_px / 2.0 - y_px) / 384.0];

    // Inside the chip, left of the text: solid ink.
    let p = to_model(20.0, 44.0);
    assert_eq!(exec.sample_nd(&p).unwrap(), 1.0);
    // Scanning the caps' midline crosses knocked-out glyph strokes.
    let mut holes = 0;
    let mut x_px = 33.0;
    while x_px < 70.0 {
        let p = to_model(x_px, 44.0);
        if exec.sample_nd(&p).unwrap() == 0.0 {
            holes += 1;
        }
        x_px += 0.25;
    }
    assert!(
        holes > 5,
        "expected glyph knockout, got {holes} empty samples"
    );
}

#[test]
fn width_config_scales_both_outputs() {
    let (ink, plate) = card_models(CARD, &[("width", Value::Float(0.2))]);
    let mut plate_exec = NativeModelExecutor::new(&plate).unwrap();
    let [min_x, max_x, ..] = bounds_2d(&mut plate_exec);
    assert!((max_x - 0.1).abs() < 1e-9 && (min_x + 0.1).abs() < 1e-9);
    let mut ink_exec = NativeModelExecutor::new(&ink).unwrap();
    let [i_min_x, i_max_x, ..] = bounds_2d(&mut ink_exec);
    assert!(i_max_x <= 0.1 + 1e-9 && i_min_x >= -0.1 - 1e-9);
}

#[test]
fn unsupported_markup_errors_list_offenders() {
    let html = r#"<div class="shiny-badge md:flex"><p>x</p></div>"#;
    let err = run_operator_all(
        "html_card_operator",
        vec![html.as_bytes().to_vec(), Vec::new()],
    )
    .expect_err("unknown classes must fail");
    assert!(err.contains("shiny-badge"), "{err}");
    assert!(err.contains("md:"), "{err}");

    let err = run_operator_all(
        "html_card_operator",
        vec![b"<video></video>".to_vec(), Vec::new()],
    )
    .expect_err("unsupported tag must fail");
    assert!(err.contains("video"), "{err}");

    let err = run_operator_all("html_card_operator", vec![Vec::new(), Vec::new()])
        .expect_err("empty input must fail");
    assert!(err.contains("no HTML"), "{err}");
}

#[test]
fn blank_card_reports_no_ink() {
    let err = run_operator_all(
        "html_card_operator",
        vec![b"<div class=\"p-4\"></div>".to_vec(), Vec::new()],
    )
    .expect_err("nothing dark to print");
    assert!(err.contains("no ink"), "{err}");
}

#[test]
fn plate_and_ink_extrude_into_3d() {
    let (ink, plate) = card_models(CARD, &[]);
    for (model, name) in [(plate, "plate"), (ink, "ink")] {
        let outputs = run_operator_all(
            "extrude_operator",
            vec![
                model,
                cbor_map(&[("height", Value::Float(0.002))]),
                Vec::new(),
            ],
        )
        .unwrap_or_else(|e| panic!("{name} extrudes: {e}"));
        let exec = NativeModelExecutor::new(outputs.get(&0).unwrap()).unwrap();
        assert_eq!(exec.dimensions(), 3, "{name}");
    }
}
