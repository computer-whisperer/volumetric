//! Integration tests for fea_deform_operator: a model deformed by an FEA
//! mesh's displacement field.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::wasm::{
    NativeModelExecutor, OperatorExecutor, OperatorIo, create_operator_executor,
};
use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh, encode_fea_mesh};

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

fn cbor_config(entries: &[(&str, ciborium::value::Value)]) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(
            entries
                .iter()
                .map(|(k, v)| (ciborium::value::Value::Text((*k).into()), v.clone()))
                .collect(),
        ),
        &mut out,
    )
    .unwrap();
    out
}

/// A solid box spanning `a..b` via rectangular_prism_operator.
fn prism(a: [f64; 3], b: [f64; 3]) -> Vec<u8> {
    let vec_bytes = |v: [f64; 3]| -> Vec<u8> { v.iter().flat_map(|c| c.to_le_bytes()).collect() };
    run_operator(
        "rectangular_prism_operator",
        vec![
            cbor_config(&[(
                "mode",
                ciborium::value::Value::Text("opposite_corners".into()),
            )]),
            vec_bytes(a),
            vec_bytes(b),
        ],
    )
    .expect("prism runs")
}

/// An n^3 uniform hex grid over `origin..origin+extent` with a node
/// displacement field.
fn grid_mesh(
    n: usize,
    origin: f64,
    extent: f64,
    field_name: &str,
    disp: impl Fn([f64; 3]) -> [f64; 3],
) -> FeaMesh {
    let cell = extent / n as f64;
    let per_axis = n + 1;
    let mut node_positions = Vec::new();
    let mut displacement = Vec::new();
    for x in 0..per_axis {
        for y in 0..per_axis {
            for z in 0..per_axis {
                let p = [
                    origin + x as f64 * cell,
                    origin + y as f64 * cell,
                    origin + z as f64 * cell,
                ];
                node_positions.extend(p);
                displacement.extend(disp(p));
            }
        }
    }
    let node = |x: usize, y: usize, z: usize| -> u32 { ((x * per_axis + y) * per_axis + z) as u32 };
    let mut connectivity = Vec::new();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                connectivity.extend([
                    node(x, y, z),
                    node(x + 1, y, z),
                    node(x + 1, y + 1, z),
                    node(x, y + 1, z),
                    node(x, y, z + 1),
                    node(x + 1, y, z + 1),
                    node(x + 1, y + 1, z + 1),
                    node(x, y + 1, z + 1),
                ]);
            }
        }
    }
    let mesh = FeaMesh {
        element_kind: FeaElementKind::Hex8,
        node_positions,
        connectivity,
        node_fields: vec![FeaField {
            name: field_name.to_string(),
            components: 3,
            data: displacement,
        }],
        element_fields: vec![],
    };
    mesh.validate().expect("test mesh is well-formed");
    mesh
}

fn occupied(executor: &mut NativeModelExecutor, p: [f64; 3]) -> bool {
    volumetric::is_occupied(executor.sample_nd(&p).unwrap())
}

/// A [0,1]^3 box sheared by u_z = 0.4 x: occupancy follows the deformed
/// geometry, the undeformed region empties out, and bounds track the
/// deformed mesh.
#[test]
fn shear_moves_the_geometry_with_the_mesh() {
    let mesh = grid_mesh(2, 0.0, 1.0, "displacement", |p| [0.0, 0.0, 0.4 * p[0]]);
    let output = run_operator(
        "fea_deform_operator",
        vec![prism([0.0; 3], [1.0; 3]), encode_fea_mesh(&mesh), vec![]],
    )
    .expect("deform operator runs");

    let mut executor = NativeModelExecutor::new(&output).expect("output executes");
    assert_eq!(executor.dimensions(), 3);

    // Bounds are the deformed mesh plus the auto skin (half the mean
    // deformed element diagonal; the shear stretches every element's z
    // extent from 0.5 to 0.7).
    let bounds = executor.get_bounds_nd().unwrap();
    let skin = 0.5 * (0.25f64 + 0.25 + 0.49).sqrt();
    assert!((bounds.min(0) - (0.0 - skin)).abs() < 1e-5, "{bounds:?}");
    assert!((bounds.max(2) - (1.4 + skin)).abs() < 1e-5, "{bounds:?}");

    // At x = 0.9 the box's top sits at z = 1.36: below occupied, above
    // (but within the mesh's reach) empty.
    assert!(occupied(&mut executor, [0.9, 0.5, 1.25]));
    assert!(occupied(&mut executor, [0.9, 0.5, 0.5]));
    assert!(!occupied(&mut executor, [0.9, 0.5, 1.45]));
    // The sheared box's bottom at x = 0.9 moved up to z = 0.36.
    assert!(!occupied(&mut executor, [0.9, 0.5, 0.2]));
    // At x ~ 0 nothing moved.
    assert!(occupied(&mut executor, [0.05, 0.5, 0.5]));
    assert!(!occupied(&mut executor, [0.05, 0.5, 1.2]));
    // Far outside the deformed mesh and skin.
    assert!(!occupied(&mut executor, [3.0, 3.0, 3.0]));
}

/// `scale` attenuates the baked displacement and `field` picks the node
/// field by name.
#[test]
fn scale_and_field_name_flow_through() {
    let mesh = grid_mesh(2, 0.0, 1.0, "warp", |p| [0.0, 0.0, 0.4 * p[0]]);
    let deform = |scale: f64| -> Vec<u8> {
        run_operator(
            "fea_deform_operator",
            vec![
                prism([0.0; 3], [1.0; 3]),
                encode_fea_mesh(&mesh),
                cbor_config(&[
                    ("scale", ciborium::value::Value::Float(scale)),
                    ("field", ciborium::value::Value::Text("warp".into())),
                ]),
            ],
        )
        .expect("deform operator runs")
    };

    // Full shear: top of the box at x = 0.9 is z = 1.36.
    let mut full = NativeModelExecutor::new(&deform(1.0)).unwrap();
    assert!(occupied(&mut full, [0.9, 0.5, 1.3]));
    // Half shear: top at z = 1.18, so 1.3 is empty but 1.1 is not.
    let mut half = NativeModelExecutor::new(&deform(0.5)).unwrap();
    assert!(!occupied(&mut half, [0.9, 0.5, 1.3]));
    assert!(occupied(&mut half, [0.9, 0.5, 1.1]));
    // Scale 0: the identity map over the mesh region.
    let mut zero = NativeModelExecutor::new(&deform(0.0)).unwrap();
    assert!(occupied(&mut zero, [0.9, 0.5, 0.9]));
    assert!(!occupied(&mut zero, [0.9, 0.5, 1.1]));
}

/// Channelled inputs keep their layout: sample_channels reads the input's
/// channels at the material point, with occupancy zeroed outside the
/// deformed mesh.
#[test]
fn channels_pass_through_at_the_material_point() {
    // density_gradient_model: [-1,1]^3, density channel = 0.5 + 0.5x.
    // Translate it by +0.25 in x.
    let mesh = grid_mesh(2, -1.0, 2.0, "displacement", |_| [0.25, 0.0, 0.0]);
    let output = run_operator(
        "fea_deform_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            encode_fea_mesh(&mesh),
            vec![],
        ],
    )
    .expect("deform operator runs");

    let mut executor = NativeModelExecutor::new(&output).expect("output executes");
    let channels = executor.sample_format().channels.len();
    assert_eq!(channels, 2, "occupancy + density pass through");

    // The deformed point (0.75, 0, 0) is material (0.5, 0, 0): density
    // 0.75 rides along.
    let row = executor.sample_channels_nd(&[0.75, 0.1, 0.1]).unwrap();
    assert!(volumetric::is_occupied(row[0]), "{row:?}");
    assert!((row[1] - 0.75).abs() < 1e-5, "{row:?}");

    // Outside the deformed mesh and skin: occupancy forced to 0.
    let row = executor.sample_channels_nd(&[-3.0, 0.0, 0.0]).unwrap();
    assert!(!volumetric::is_occupied(row[0]), "{row:?}");

    // sample and sample_channels agree.
    let occ = executor.sample_nd(&[0.75, 0.1, 0.1]).unwrap();
    assert!(volumetric::is_occupied(occ));
}

/// Bad inputs fail with readable errors instead of emitting broken models.
#[test]
fn rejects_bad_meshes_and_config() {
    let box_model = prism([0.0; 3], [1.0; 3]);
    let mut bare = grid_mesh(1, 0.0, 1.0, "displacement", |_| [0.0; 3]);
    bare.node_fields.clear();

    let err = run_operator(
        "fea_deform_operator",
        vec![box_model.clone(), encode_fea_mesh(&bare), vec![]],
    )
    .expect_err("mesh without a displacement field should be rejected");
    assert!(err.contains("displacement"), "{err}");

    let mesh = grid_mesh(1, 0.0, 1.0, "displacement", |_| [0.0; 3]);
    let err = run_operator(
        "fea_deform_operator",
        vec![
            box_model.clone(),
            encode_fea_mesh(&mesh),
            cbor_config(&[("field", ciborium::value::Value::Text("nope".into()))]),
        ],
    )
    .expect_err("unknown field name should be rejected");
    assert!(err.contains("nope"), "{err}");

    let err = run_operator(
        "fea_deform_operator",
        vec![
            box_model.clone(),
            encode_fea_mesh(&mesh),
            cbor_config(&[("boundary_skin", ciborium::value::Value::Float(-1.0))]),
        ],
    )
    .expect_err("negative skin should be rejected");
    assert!(err.contains("boundary_skin"), "{err}");

    // A minimal 2D model (walrus needs binary wasm, so assemble it).
    let flat = wat::parse_str(
        r#"(module
            (memory (export "memory") 1)
            (func (export "get_dimensions") (result i32) (i32.const 2))
            (func (export "get_io_ptr") (result i32) (i32.const 1024))
            (func (export "get_bounds") (param i32))
            (func (export "sample") (param i32) (result f32) (f32.const 1)))"#,
    )
    .unwrap();
    let err = run_operator(
        "fea_deform_operator",
        vec![flat, encode_fea_mesh(&mesh), vec![]],
    )
    .expect_err("2D input should be rejected");
    assert!(err.contains("3D"), "{err}");
}
