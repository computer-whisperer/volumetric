//! Integration tests for the pattern operator: mirror, linear and circular
//! instancing through the real pattern_operator.wasm, sampled with the
//! engine's executor.
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

fn int(i: i64) -> Value {
    Value::Integer(i.into())
}

fn pattern(model: Vec<u8>, blocks: Vec<(&str, Value)>) -> Result<Vec<u8>, String> {
    run_operator("pattern_operator", vec![model, encode(map(blocks))])
}

fn sphere() -> Vec<u8> {
    wasm_artifact("simple_sphere_model")
}

/// The unit sphere moved by `(dx, dy, dz)`.
fn sphere_at(dx: f64, dy: f64, dz: f64) -> Vec<u8> {
    run_operator(
        "translate_operator",
        vec![
            sphere(),
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
fn linear_pattern_places_copies_along_the_step() {
    let row = pattern(
        sphere(),
        vec![("linear", map(vec![("count", int(3)), ("dx", float(3.0))]))],
    )
    .expect("linear pattern");
    let mut executor = NativeModelExecutor::new(&row).unwrap();
    for x in [0.0, 3.0, 6.0] {
        assert!(inside(&mut executor, &[x, 0.0, 0.0]), "copy at x = {x}");
    }
    assert!(!inside(&mut executor, &[1.5, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[9.0, 0.0, 0.0]));
    assert_bounds(&mut executor, 0, -1.0, 7.0);
    assert_bounds(&mut executor, 1, -1.0, 1.0);
}

#[test]
fn mirror_keeps_or_replaces_the_original() {
    let both = pattern(
        sphere_at(2.0, 0.0, 0.0),
        vec![("mirror", map(vec![("axis", text("x"))]))],
    )
    .expect("mirror keeping the original");
    let mut executor = NativeModelExecutor::new(&both).unwrap();
    assert!(inside(&mut executor, &[2.0, 0.0, 0.0]));
    assert!(inside(&mut executor, &[-2.0, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[0.0, 0.0, 0.0]));
    assert_bounds(&mut executor, 0, -3.0, 3.0);

    let image_only = pattern(
        sphere_at(2.0, 0.0, 0.0),
        vec![(
            "mirror",
            map(vec![
                ("axis", text("x")),
                ("offset", float(1.0)),
                ("keep_original", Value::Bool(false)),
            ]),
        )],
    )
    .expect("mirror replacing the original");
    let mut executor = NativeModelExecutor::new(&image_only).unwrap();
    // The plane x = 1 sends the sphere at x = 2 to x = 0.
    assert!(inside(&mut executor, &[0.0, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[2.0, 0.0, 0.0]));
    assert_bounds(&mut executor, 0, -1.0, 1.0);
}

#[test]
fn circular_pattern_spaces_a_full_turn_and_a_partial_sweep() {
    let ring = pattern(
        sphere_at(3.0, 0.0, 0.0),
        vec![(
            "circular",
            map(vec![("count", int(4)), ("axis", text("z"))]),
        )],
    )
    .expect("full-turn ring");
    let mut executor = NativeModelExecutor::new(&ring).unwrap();
    for p in [
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [-3.0, 0.0, 0.0],
        [0.0, -3.0, 0.0],
    ] {
        assert!(inside(&mut executor, &p), "copy at {p:?}");
    }
    let diagonal = 3.0 / 2f64.sqrt();
    assert!(!inside(&mut executor, &[diagonal, diagonal, 0.0]));
    assert_bounds(&mut executor, 0, -4.0, 4.0);
    assert_bounds(&mut executor, 1, -4.0, 4.0);
    assert_bounds(&mut executor, 2, -1.0, 1.0);

    let quarter = pattern(
        sphere_at(3.0, 0.0, 0.0),
        vec![(
            "circular",
            map(vec![("count", int(3)), ("sweep_deg", float(90.0))]),
        )],
    )
    .expect("quarter sweep");
    let mut executor = NativeModelExecutor::new(&quarter).unwrap();
    for p in [[3.0, 0.0, 0.0], [diagonal, diagonal, 0.0], [0.0, 3.0, 0.0]] {
        assert!(inside(&mut executor, &p), "copy at {p:?}");
    }
    assert!(!inside(&mut executor, &[-3.0, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[0.0, -3.0, 0.0]));
    assert_bounds(&mut executor, 0, -1.0, 4.0);
    assert_bounds(&mut executor, 1, -1.0, 4.0);
}

#[test]
fn blocks_compose_as_a_product() {
    // The toy-car wheel case: one wheel at the front-right hub, mirrored
    // across the centre plane and repeated at the rear axle.
    let wheels = pattern(
        sphere_at(2.0, 0.0, 1.0),
        vec![
            ("mirror", map(vec![("axis", text("z"))])),
            ("linear", map(vec![("count", int(2)), ("dx", float(-4.0))])),
        ],
    )
    .expect("mirror x linear");
    let mut executor = NativeModelExecutor::new(&wheels).unwrap();
    for p in [
        [2.0, 0.0, 1.0],
        [2.0, 0.0, -1.0],
        [-2.0, 0.0, 1.0],
        [-2.0, 0.0, -1.0],
    ] {
        assert!(inside(&mut executor, &p), "copy at {p:?}");
    }
    assert!(!inside(&mut executor, &[0.0, 0.0, 1.0]));
    assert!(!inside(&mut executor, &[2.0, 0.0, 3.0]));
    assert_bounds(&mut executor, 0, -3.0, 3.0);
    assert_bounds(&mut executor, 2, -2.0, 2.0);
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
fn sketches_pattern_in_plane() {
    let row = pattern(
        circle_sketch(),
        vec![(
            "linear",
            map(vec![
                ("count", int(2)),
                ("dx", float(3.0)),
                ("dz", float(99.0)),
            ]),
        )],
    )
    .expect("2D linear pattern");
    let mut executor = NativeModelExecutor::new(&row).unwrap();
    assert_eq!(executor.dimensions(), 2);
    assert!(inside(&mut executor, &[0.0, 0.0]));
    assert!(inside(&mut executor, &[3.0, 0.0]));
    assert!(!inside(&mut executor, &[1.5, 0.0]));
    assert_bounds(&mut executor, 0, -1.5, 4.5);
    assert_bounds(&mut executor, 1, -1.5, 1.5);
    // Sample again after get_bounds to catch buffer-overrun corruption.
    assert!(inside(&mut executor, &[3.0, 0.0]));

    let ring = pattern(
        circle_sketch(),
        vec![("circular", map(vec![("count", int(2)), ("cx", float(1.5))]))],
    )
    .expect("2D circular pattern about (1.5, 0)");
    let mut executor = NativeModelExecutor::new(&ring).unwrap();
    assert!(inside(&mut executor, &[3.0, 0.0]));
    assert!(!inside(&mut executor, &[1.5, 1.2]));

    let err = pattern(
        circle_sketch(),
        vec![("mirror", map(vec![("axis", text("z"))]))],
    )
    .expect_err("z mirror of a sketch must fail");
    assert!(err.contains("in-plane"), "{err}");
    let err = pattern(
        circle_sketch(),
        vec![("circular", map(vec![("axis", text("x"))]))],
    )
    .expect_err("x-axis ring of a sketch must fail");
    assert!(err.contains("in-plane"), "{err}");
}

#[test]
fn channels_follow_the_first_containing_instance() {
    // density_gradient: occupancy on [-1, 1]^3, density 0.5 + 0.5 x.
    let row = pattern(
        wasm_artifact("density_gradient_model"),
        vec![("linear", map(vec![("count", int(2)), ("dx", float(1.0))]))],
    )
    .expect("overlapping linear pattern");
    let mut executor = NativeModelExecutor::new(&row).unwrap();
    assert_eq!(executor.sample_format().channels.len(), 2);

    // Both instances contain x = 0.5; the original (local x = 0.5) wins.
    let row = executor.sample_channels_nd(&[0.5, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.75).abs() < 1e-6, "density was {}", row[1]);
    // Only the shifted instance contains x = 1.5 (local x = 0.5).
    let row = executor.sample_channels_nd(&[1.5, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.75).abs() < 1e-6, "density was {}", row[1]);
    // Outside every instance the row reports empty.
    let row = executor.sample_channels_nd(&[3.5, 0.0, 0.0]).unwrap();
    assert_eq!(row[0], 0.0);
    assert!(!inside(&mut executor, &[3.5, 0.0, 0.0]));
}

#[test]
fn no_blocks_passes_the_model_through() {
    let same = pattern(sphere(), vec![]).expect("empty pattern");
    let mut executor = NativeModelExecutor::new(&same).unwrap();
    assert!(inside(&mut executor, &[0.0, 0.0, 0.0]));
    assert!(!inside(&mut executor, &[1.1, 0.0, 0.0]));
    assert_bounds(&mut executor, 0, -1.0, 1.0);
}

#[test]
fn too_many_instances_is_an_error() {
    let err = pattern(
        sphere(),
        vec![(
            "linear",
            map(vec![("count", int(2000)), ("dx", float(3.0))]),
        )],
    )
    .expect_err("2000 instances must be refused");
    assert!(err.contains("1024"), "{err}");
}

#[test]
fn metadata_declares_three_optional_blocks() {
    let metadata =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("pattern_operator"))
            .expect("pattern metadata");
    assert_eq!(metadata.name, "pattern_operator");
    assert_eq!(metadata.category, "Transforms");
    assert_eq!(metadata.inputs.len(), 2);
    assert_eq!(metadata.variadic_slot(), None);
    assert!(metadata.docs.contains("# Pattern"));
    let volumetric_abi::OperatorMetadataInput::CBORConfiguration(schema) = &metadata.inputs[1]
    else {
        panic!("input 1 should be the config");
    };
    let fields = volumetric::operator_config::parse_schema(schema).expect("parse schema");
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["mirror", "linear", "circular"]);
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
