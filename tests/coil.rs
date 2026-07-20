//! Integration tests for the coil operator: a flat Lua slab rolled into an
//! Archimedean spiral around the world y axis.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

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

/// A 150 x 10 x 1.5 mm sheet: long enough for three wraps at the test's
/// bore radius and pitch.
const SLAB: &str = r#"
function is_inside(x, y, z)
    return x >= 0.0 and x <= 0.15
        and y >= 0.0 and y <= 0.01
        and z >= 0.0 and z <= 0.0015
end

function get_bounds_min_x() return 0.0 end
function get_bounds_max_x() return 0.15 end
function get_bounds_min_y() return 0.0 end
function get_bounds_max_y() return 0.01 end
function get_bounds_min_z() return 0.0 end
function get_bounds_max_z() return 0.0015 end
"#;

fn cbor_floats(fields: &[(&str, f64)]) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(
            fields
                .iter()
                .map(|(k, v)| {
                    (
                        ciborium::value::Value::Text((*k).into()),
                        ciborium::value::Value::Float(*v),
                    )
                })
                .collect(),
        ),
        &mut out,
    )
    .unwrap();
    out
}

fn coiled_slab() -> NativeModelExecutor {
    let slab = run_operator("lua_script_operator", vec![SLAB.as_bytes().to_vec()])
        .expect("compile slab");
    let coil = run_operator(
        "coil_operator",
        vec![
            slab,
            cbor_floats(&[("inner_radius", 0.004), ("gap", 0.001)]),
        ],
    )
    .expect("coil slab");
    NativeModelExecutor::new(&coil).expect("instantiate coil")
}

/// Pitch is thickness + gap = 2.5mm; with a 4mm bore the wraps along the
/// +x ray sit at r in [4, 5.5], [6.5, 8], [9, 10.5], [11.5, 13] mm.
#[test]
fn coil_wraps_gap_and_bore_classify_correctly() {
    let mut coil = coiled_slab();
    assert_eq!(coil.dimensions(), 3);

    // First wrap, the gap beyond it, then the second wrap.
    assert_eq!(coil.sample_nd(&[0.0045, 0.005, 0.0]).unwrap(), 1.0);
    assert_eq!(coil.sample_nd(&[0.006, 0.005, 0.0]).unwrap(), 0.0);
    assert_eq!(coil.sample_nd(&[0.007, 0.005, 0.0]).unwrap(), 1.0);

    // The bore, and beyond the sheet's width along the coil axis.
    assert_eq!(coil.sample_nd(&[0.002, 0.005, 0.0]).unwrap(), 0.0);
    assert_eq!(coil.sample_nd(&[0.0045, 0.02, 0.0]).unwrap(), 0.0);

    // A quarter turn in (+z), the spiral has advanced: sheet spans
    // r in [4.625, 6.125] mm. A point below that at three quarter turns
    // (-z, r = 5mm < 5.875) predates the spiral's start: bore.
    assert_eq!(coil.sample_nd(&[0.0, 0.005, 0.005]).unwrap(), 1.0);
    assert_eq!(coil.sample_nd(&[0.0, 0.005, -0.005]).unwrap(), 0.0);
}

#[test]
fn coil_bounds_cover_the_outermost_wrap() {
    let mut coil = coiled_slab();
    let bounds = coil.get_bounds_nd().unwrap();

    // theta_end = (sqrt(r0^2 + 2*b*L) - r0) / b with b = pitch/tau;
    // outer = r0 + b*theta_end + thickness = 13.06mm.
    let outer = bounds.max(0);
    assert!((0.0125..=0.0135).contains(&outer), "outer = {outer}");
    assert_eq!(bounds.min(0), -outer);
    assert_eq!((bounds.min(1), bounds.max(1)), (0.0, 0.01));
    assert_eq!(bounds.min(2), -outer);
    assert_eq!(bounds.max(2), outer);
}

#[test]
fn coil_rejects_bad_config_and_non_3d_input() {
    let slab = run_operator("lua_script_operator", vec![SLAB.as_bytes().to_vec()])
        .expect("compile slab");
    let err = run_operator(
        "coil_operator",
        vec![slab, cbor_floats(&[("inner_radius", 0.0)])],
    )
    .expect_err("zero inner_radius must be rejected");
    assert!(err.contains("inner_radius"), "unexpected error: {err}");

    let sketch_2d = r#"
function is_inside(x, y)
    return x*x + y*y <= 1.0
end
function get_bounds_min_x() return -1.0 end
function get_bounds_max_x() return 1.0 end
function get_bounds_min_y() return -1.0 end
function get_bounds_max_y() return 1.0 end
"#;
    let sketch = run_operator("lua_script_operator", vec![sketch_2d.as_bytes().to_vec()])
        .expect("compile 2D sketch");
    let err = run_operator(
        "coil_operator",
        vec![sketch, cbor_floats(&[("inner_radius", 0.004)])],
    )
    .expect_err("2D input must be rejected");
    assert!(err.contains("3D"), "unexpected error: {err}");
}
