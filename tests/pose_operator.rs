//! Integration tests for the pose operator: scale and rotation about a
//! pivot followed by a translation, through the real pose_operator.wasm.
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

fn map(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (Value::Text(k.to_string()), v))
            .collect(),
    )
}

fn encode(value: Value) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(&value, &mut out).unwrap();
    out
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

fn float(f: f64) -> Value {
    Value::Float(f)
}

fn pose(model: Vec<u8>, blocks: Vec<(&str, Value)>) -> Result<Vec<u8>, String> {
    run_operator("pose_operator", vec![model, encode(map(blocks))])
}

/// The unit sphere moved by `(dx, dy, dz)` (through translate, so the
/// tests do not rely on the operator under test for their setup).
fn sphere_at(dx: f64, dy: f64, dz: f64) -> Vec<u8> {
    run_operator(
        "translate_operator",
        vec![
            wasm_artifact("simple_sphere_model"),
            encode(map(vec![
                ("dx", float(dx)),
                ("dy", float(dy)),
                ("dz", float(dz)),
            ])),
        ],
    )
    .expect("translate sphere")
}

fn inside(executor: &mut NativeModelExecutor, p: &[f64]) -> bool {
    executor.sample_nd(p).unwrap() > 0.5
}

fn assert_bounds(executor: &mut NativeModelExecutor, axis: usize, min: f64, max: f64) {
    let bounds = executor.get_bounds_nd().unwrap();
    assert!(
        (bounds.min(axis) - min).abs() < 1e-9 && (bounds.max(axis) - max).abs() < 1e-9,
        "axis {axis}: expected [{min}, {max}], got [{}, {}]",
        bounds.min(axis),
        bounds.max(axis)
    );
}

#[test]
fn rotates_about_a_point_pivot() {
    // Sphere at x = 3 turned half a turn about the line x = 1: lands at x = -1.
    let posed = pose(
        sphere_at(3.0, 0.0, 0.0),
        vec![
            (
                "pivot",
                map(vec![("at", text("point")), ("px", float(1.0))]),
            ),
            ("rotate", map(vec![("rz_deg", float(180.0))])),
        ],
    )
    .expect("pose about a point");
    let mut executor = NativeModelExecutor::new(&posed).unwrap();
    assert!(inside(&mut executor, &[-1.0, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[3.0, 0.0, 0.0]));
    assert_bounds(&mut executor, 0, -2.0, 0.0);

    // The default pivot is the origin: the same turn sends x = 3 to x = -3.
    let posed = pose(
        sphere_at(3.0, 0.0, 0.0),
        vec![("rotate", map(vec![("rz_deg", float(180.0))]))],
    )
    .expect("pose about the origin");
    let mut executor = NativeModelExecutor::new(&posed).unwrap();
    assert!(inside(&mut executor, &[-3.0, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[-1.0, 0.0, 0.0]));
}

#[test]
fn center_pivot_uses_the_input_bounds() {
    // Scaling about the sphere's own centre (3, 0, 0) shrinks it in place.
    let posed = pose(
        sphere_at(3.0, 0.0, 0.0),
        vec![
            ("pivot", map(vec![("at", text("center"))])),
            (
                "scale",
                map(vec![
                    ("sx", float(0.5)),
                    ("sy", float(0.5)),
                    ("sz", float(0.5)),
                ]),
            ),
        ],
    )
    .expect("scale about the centre");
    let mut executor = NativeModelExecutor::new(&posed).unwrap();
    assert!(inside(&mut executor, &[3.4, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[3.6, 0.0, 0.0]));
    assert_bounds(&mut executor, 0, 2.5, 3.5);

    // A rotation about its own centre leaves the sphere where it is.
    let posed = pose(
        sphere_at(3.0, 0.0, 0.0),
        vec![
            ("pivot", map(vec![("at", text("center"))])),
            ("rotate", map(vec![("rz_deg", float(90.0))])),
        ],
    )
    .expect("rotate about the centre");
    let mut executor = NativeModelExecutor::new(&posed).unwrap();
    assert!(inside(&mut executor, &[3.0, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[0.0, 3.0, 0.0]));
}

#[test]
fn order_is_scale_then_rotate_then_translate() {
    // Stretch x by 2, turn a quarter about z (the stretch is now along y),
    // then lift by 5.
    let posed = pose(
        wasm_artifact("simple_sphere_model"),
        vec![
            ("scale", map(vec![("sx", float(2.0))])),
            ("rotate", map(vec![("rz_deg", float(90.0))])),
            ("translate", map(vec![("dz", float(5.0))])),
        ],
    )
    .expect("scale, rotate, translate");
    let mut executor = NativeModelExecutor::new(&posed).unwrap();
    assert!(inside(&mut executor, &[0.0, 1.8, 5.0]));
    assert!(!inside(&mut executor, &[1.8, 0.0, 5.0]));
    assert!(!inside(&mut executor, &[0.0, 1.8, 0.0]));
    assert_bounds(&mut executor, 0, -1.0, 1.0);
    assert_bounds(&mut executor, 1, -2.0, 2.0);
    assert_bounds(&mut executor, 2, 4.0, 6.0);
}

const CIRCLE_SKETCH: &str = r#"
function is_inside(x, y)
    if x*x + y*y <= 1.0 then
        return 1.0
    else
        return 0.0
    end
end
function get_bounds_min_x() return -1.5 end
function get_bounds_max_x() return 1.5 end
function get_bounds_min_y() return -1.5 end
function get_bounds_max_y() return 1.5 end
"#;

fn circle_sketch() -> Vec<u8> {
    run_operator(
        "lua_script_operator",
        vec![CIRCLE_SKETCH.as_bytes().to_vec()],
    )
    .expect("compile circle sketch")
}

#[test]
fn sketches_pose_in_plane() {
    // Half a turn about (1.5, 0) sends the origin to (3, 0); then dx = 1.
    // The z components would overrun a 2D buffer if they were not ignored.
    let posed = pose(
        circle_sketch(),
        vec![
            (
                "pivot",
                map(vec![
                    ("at", text("point")),
                    ("px", float(1.5)),
                    ("pz", float(99.0)),
                ]),
            ),
            ("rotate", map(vec![("rz_deg", float(180.0))])),
            ("scale", map(vec![("sz", float(0.0))])),
            (
                "translate",
                map(vec![("dx", float(1.0)), ("dz", float(99.0))]),
            ),
        ],
    )
    .expect("2D pose");
    let mut executor = NativeModelExecutor::new(&posed).unwrap();
    assert_eq!(executor.dimensions(), 2);
    assert!(inside(&mut executor, &[4.0, 0.0]));
    assert!(!inside(&mut executor, &[0.0, 0.0]));
    assert_bounds(&mut executor, 0, 2.5, 5.5);
    assert_bounds(&mut executor, 1, -1.5, 1.5);
    assert!(inside(&mut executor, &[4.0, 0.0]));

    let posed = pose(
        circle_sketch(),
        vec![
            ("pivot", map(vec![("at", text("center"))])),
            ("scale", map(vec![("sx", float(2.0))])),
        ],
    )
    .expect("2D centre pivot");
    let mut executor = NativeModelExecutor::new(&posed).unwrap();
    assert!(inside(&mut executor, &[1.8, 0.0]));
    assert_bounds(&mut executor, 0, -3.0, 3.0);

    let err = pose(
        circle_sketch(),
        vec![("rotate", map(vec![("rx_deg", float(10.0))]))],
    )
    .expect_err("rx on a sketch must fail");
    assert!(err.contains("in-plane"), "{err}");
}

#[test]
fn channels_pass_through_the_posed_position() {
    let posed = pose(
        wasm_artifact("density_gradient_model"),
        vec![("translate", map(vec![("dx", float(0.5))]))],
    )
    .expect("pose gradient");
    let mut executor = NativeModelExecutor::new(&posed).unwrap();
    assert_eq!(executor.sample_format().channels.len(), 2);
    // World x = 0.75 is model x = 0.25: density 0.5 + 0.5 * 0.25.
    let row = executor.sample_channels_nd(&[0.75, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.625).abs() < 1e-6, "density was {}", row[1]);
}

#[test]
fn no_blocks_passes_the_model_through() {
    let same = pose(wasm_artifact("simple_sphere_model"), vec![]).expect("empty pose");
    let mut executor = NativeModelExecutor::new(&same).unwrap();
    assert!(inside(&mut executor, &[0.0, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[1.1, 0.0, 0.0]));
    assert_bounds(&mut executor, 0, -1.0, 1.0);
}

#[test]
fn zero_scale_is_an_error() {
    let err = pose(
        wasm_artifact("simple_sphere_model"),
        vec![("scale", map(vec![("sy", float(0.0))]))],
    )
    .expect_err("a zero factor must fail");
    assert!(err.contains("singular"), "{err}");
}

#[test]
fn metadata_declares_four_optional_blocks() {
    let metadata = volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("pose_operator"))
        .expect("pose metadata");
    assert_eq!(metadata.name, "pose_operator");
    assert_eq!(metadata.category, "Transforms");
    assert_eq!(metadata.inputs.len(), 2);
    assert!(metadata.docs.contains("# Pose"));
    let volumetric_abi::OperatorMetadataInput::CBORConfiguration(schema) = &metadata.inputs[1]
    else {
        panic!("input 1 should be the config");
    };
    let fields = volumetric::operator_config::parse_schema(schema).expect("parse schema");
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["pivot", "scale", "rotate", "translate"]);
    for field in &fields {
        assert!(field.optional, "{} should be optional", field.name);
        assert!(
            matches!(
                field.ty,
                volumetric::operator_config::ConfigFieldType::Group(_)
            ),
            "{} should be a group",
            field.name
        );
    }
}
