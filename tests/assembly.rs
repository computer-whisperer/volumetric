//! End-to-end tests of the assembly operators through the real wasm
//! pipeline: a Mechanism from its config and a routed axis, an Assembly
//! and its posed union from Assemble, and posed parts from Assembly Model.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use ciborium::value::Value;
use volumetric::mechanism::{JointKind, decode_assembly, decode_mechanism};
use volumetric::subspace::{Subspace, encode_subspace};
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

/// Every output the operator posted, by index.
fn run_operator(
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

fn vec3(v: [f64; 3]) -> Value {
    Value::Array(v.into_iter().map(float).collect())
}

fn strings(items: &[&str]) -> Value {
    Value::Array(items.iter().map(|s| text(s)).collect())
}

/// The unit sphere moved by `(dx, dy, dz)`.
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
    .remove(&0)
    .expect("translated sphere")
}

fn inside(model: &[u8], p: [f64; 3]) -> bool {
    let mut executor = NativeModelExecutor::new(model).unwrap();
    executor.sample_nd(&p).unwrap() > 0.5
}

fn bounds(model: &[u8], axis: usize) -> (f64, f64) {
    let mut executor = NativeModelExecutor::new(model).unwrap();
    let bounds = executor.get_bounds_nd().unwrap();
    (bounds.min(axis), bounds.max(axis))
}

/// Part `a` fixed to the world, part `b` swivelling about the z axis
/// through the origin on `a`, from -180 to 180 degrees.
fn two_part_config(axis: Value) -> Vec<u8> {
    let mut swivel = vec![
        ("name", text("swivel")),
        ("kind", text("revolute")),
        ("parent", text("a")),
        ("child", text("b")),
        ("min", float(-180.0)),
        ("max", float(180.0)),
    ];
    match axis {
        Value::Integer(_) => swivel.push(("axis_input", axis)),
        axis => swivel.push(("axis", axis)),
    }
    encode(map(vec![
        ("parts", strings(&["a", "b"])),
        (
            "joints",
            Value::Array(vec![
                map(vec![("name", text("mount")), ("child", text("a"))]),
                map(swivel),
            ]),
        ),
    ]))
}

fn inline_z_axis() -> Value {
    map(vec![
        ("origin", vec3([0.0, 0.0, 0.0])),
        ("direction", vec3([0.0, 0.0, 1.0])),
    ])
}

fn mechanism(config: Vec<u8>, axes: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    let mut inputs = vec![config];
    inputs.extend(axes);
    run_operator("mechanism_operator", inputs).map(|mut outputs| outputs.remove(&0).unwrap())
}

fn state(entries: Vec<(&str, f64)>) -> Vec<u8> {
    encode(map(entries
        .into_iter()
        .map(|(k, v)| (k, float(v)))
        .collect()))
}

/// Sphere `a` at the origin and sphere `b` at x = 3, assembled at `state`.
fn assemble(state: Vec<u8>) -> Result<std::collections::HashMap<usize, Vec<u8>>, String> {
    let mech = mechanism(two_part_config(inline_z_axis()), vec![]).unwrap();
    run_operator(
        "assemble_operator",
        vec![
            mech,
            sphere_at(0.0, 0.0, 0.0),
            sphere_at(3.0, 0.0, 0.0),
            state,
        ],
    )
}

#[test]
fn a_mechanism_takes_its_axes_inline_or_from_a_subspace() {
    let inline = decode_mechanism(&mechanism(two_part_config(inline_z_axis()), vec![]).unwrap())
        .expect("a valid mechanism");
    assert_eq!(inline.parts, ["a", "b"]);
    assert_eq!(inline.joints[1].kind, JointKind::Revolute);
    assert_eq!(inline.state_keys(), ["swivel"]);

    // The same axis as a line Subspace through (1, 0, 0) along z.
    let line = Subspace::axis_aligned(vec![1.0, 0.0, 0.0], &[2]).unwrap();
    let routed = decode_mechanism(
        &mechanism(
            two_part_config(Value::Integer(0.into())),
            vec![encode_subspace(&line)],
        )
        .unwrap(),
    )
    .expect("a valid mechanism");
    let axis = routed.joints[1].axis.unwrap();
    assert_eq!(axis.origin, [1.0, 0.0, 0.0]);
    assert_eq!(axis.direction, [0.0, 0.0, 1.0]);

    // A routed axis with nothing wired, and a joint hanging from nothing,
    // are named in the error.
    let err = mechanism(two_part_config(Value::Integer(0.into())), vec![]).unwrap_err();
    assert!(err.contains("axis input 0"), "{err}");
    let loose = encode(map(vec![
        ("parts", strings(&["a", "b"])),
        (
            "joints",
            Value::Array(vec![map(vec![
                ("name", text("mount")),
                ("child", text("a")),
            ])]),
        ),
    ]));
    let err = mechanism(loose, vec![]).unwrap_err();
    assert!(err.contains("hangs from nothing"), "{err}");
}

#[test]
fn assemble_poses_the_parts_and_unions_them() {
    let outputs = assemble(state(vec![("swivel", 90.0)])).expect("assemble");
    let assembly = decode_assembly(&outputs[&0]).expect("a valid assembly");
    assert_eq!(assembly.state["swivel"], 90.0);
    assert_eq!(assembly.parts[1].name, "b");
    assert!(assembly.parts[1].model.starts_with(b"\0asm"));

    // Sphere b turned a quarter turn about z sits at y = 3; a stays put.
    let model = &outputs[&1];
    assert!(inside(model, [0.0, 3.0, 0.0]));
    assert!(inside(model, [0.0, 0.0, 0.0]));
    assert!(!inside(model, [3.0, 0.0, 0.0]));
    let (lo, hi) = bounds(model, 1);
    assert!(lo <= -1.0 && hi >= 4.0 - 1e-9, "y bounds {lo}..{hi}");

    // The rest state (an empty map) leaves b at x = 3.
    let rest = assemble(Vec::new()).expect("assemble at rest");
    assert!(inside(&rest[&1], [3.0, 0.0, 0.0]));
    assert_eq!(decode_assembly(&rest[&0]).unwrap().state["swivel"], 0.0);

    // A state outside the joint's range, and a part count that does not
    // match the mechanism, are refused with their reason.
    let err = assemble(state(vec![("swivel", 200.0)])).unwrap_err();
    assert!(err.contains("outside"), "{err}");
    let mech = mechanism(two_part_config(inline_z_axis()), vec![]).unwrap();
    let err = run_operator(
        "assemble_operator",
        vec![mech, sphere_at(0.0, 0.0, 0.0), Vec::new()],
    )
    .unwrap_err();
    assert!(err.contains("model inputs are wired"), "{err}");
}

#[test]
fn assembly_model_extracts_a_part_at_another_state() {
    let assembly = assemble(state(vec![("swivel", 90.0)])).expect("assemble")[&0].clone();

    // Part b alone, at the assembly's own state: at y = 3, and a is absent.
    let b = run_operator(
        "assembly_model_operator",
        vec![
            assembly.clone(),
            encode(map(vec![("part", text("b"))])),
            Vec::new(),
        ],
    )
    .expect("part b")[&0]
        .clone();
    assert!(inside(&b, [0.0, 3.0, 0.0]));
    assert!(!inside(&b, [0.0, 0.0, 0.0]));

    // The union at a half turn instead: b at x = -3.
    let all = run_operator(
        "assembly_model_operator",
        vec![assembly.clone(), Vec::new(), state(vec![("swivel", 180.0)])],
    )
    .expect("all at 180")[&0]
        .clone();
    assert!(inside(&all, [-3.0, 0.0, 0.0]));
    assert!(inside(&all, [0.0, 0.0, 0.0]));
    assert!(!inside(&all, [0.0, 3.0, 0.0]));

    let err = run_operator(
        "assembly_model_operator",
        vec![assembly, encode(map(vec![("part", text("c"))])), Vec::new()],
    )
    .unwrap_err();
    assert!(err.contains("not a part"), "{err}");
}
