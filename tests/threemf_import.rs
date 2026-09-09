//! End-to-end tests of the 3MF pipeline: threemf_import_operator (3MF →
//! TriMesh, in metres) feeding mesh_to_model_operator (TriMesh → sampleable
//! model), plus the round trip through the host-side writer.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p threemf_import_operator -p mesh_to_model_operator

#![cfg(feature = "native")]

use volumetric::threemf::{Unit, read_3mf, write_3mf};
use volumetric::trimesh::{TriMesh, decode_tri_mesh};
use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::is_occupied;

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing wasm artifact {} ({e}); build it with \
             `cargo build --target wasm32-unknown-unknown --release -p {name}`",
            path.display()
        )
    })
}

/// The axis-aligned box [min, max]^3 as a welded 12-triangle mesh.
fn box_mesh(min: f64, max: f64) -> TriMesh {
    let positions: Vec<f64> = (0..8)
        .flat_map(|i| {
            [
                if i & 1 == 0 { min } else { max },
                if i & 2 == 0 { min } else { max },
                if i & 4 == 0 { min } else { max },
            ]
        })
        .collect();
    let quads = [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ];
    let mut indices = Vec::new();
    for q in quads {
        indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]]);
    }
    TriMesh {
        positions,
        indices,
        vertex_fields: vec![],
        face_fields: vec![],
    }
}

fn import_project(package: Vec<u8>, config: Vec<u8>, with_convert: bool) -> Project {
    let mut timeline = vec![ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(package),
            ExecutionInput::Inline(config),
        ],
        outputs: vec!["mesh".to_string()],
    }];
    let mut exports = vec!["mesh".to_string()];
    if with_convert {
        timeline.push(ExecutionStep {
            operator_id: "converter".to_string(),
            inputs: vec![ExecutionInput::AssetRef("mesh".to_string())],
            outputs: vec!["solid".to_string()],
        });
        exports.push("solid".to_string());
    }
    Project {
        version: 2,
        imports: vec![
            ImportedAsset::operator(
                "importer".to_string(),
                wasm_artifact("threemf_import_operator"),
            ),
            ImportedAsset::operator(
                "converter".to_string(),
                wasm_artifact("mesh_to_model_operator"),
            ),
        ],
        timeline,
        exports,
        baked: None,
    }
}

#[test]
fn millimetre_box_imports_in_metres_and_converts_to_a_solid() {
    // A 2 m box authored in millimetres: the importer must land it at ±1 m.
    let package = write_3mf(&box_mesh(-1000.0, 1000.0), Unit::Millimeter, "box").unwrap();
    let project = import_project(package, Vec::new(), true);
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");

    let mesh_asset = exports.iter().find(|e| e.id() == "mesh").unwrap();
    assert_eq!(mesh_asset.type_hint(), Some(AssetTypeHint::TriMesh));
    let mesh = decode_tri_mesh(mesh_asset.data()).expect("mesh decodes");
    assert_eq!(mesh.vertex_count(), 8);
    assert_eq!(mesh.triangle_count(), 12);
    assert_eq!(mesh.bounds(), Some([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]));

    let solid_asset = exports.iter().find(|e| e.id() == "solid").unwrap();
    assert_eq!(solid_asset.type_hint(), Some(AssetTypeHint::Model));
    let mut executor = create_model_executor(solid_asset.data()).expect("solid instantiates");
    assert_eq!(executor.dimensions().unwrap(), 3);
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!(bounds.as_slice(), &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    for (p, inside) in [
        ([0.0, 0.0, 0.0], true),
        ([0.9, -0.9, 0.9], true),
        ([1.1, 0.0, 0.0], false),
        ([2.0, 2.0, 2.0], false),
    ] {
        let sample = executor.sample_nd(&p).unwrap();
        assert_eq!(
            is_occupied(sample),
            inside,
            "sample at {p:?} = {sample}, expected inside={inside}"
        );
    }
}

#[test]
fn config_scale_and_item_selection_apply() {
    let package = write_3mf(&box_mesh(0.0, 1.0), Unit::Meter, "box").unwrap();
    let mut config = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![
            (
                ciborium::value::Value::Text("scale".into()),
                ciborium::value::Value::Float(3.0),
            ),
            (
                ciborium::value::Value::Text("center".into()),
                ciborium::value::Value::Bool(true),
            ),
            (
                ciborium::value::Value::Text("item".into()),
                ciborium::value::Value::Integer(1.into()),
            ),
        ]),
        &mut config,
    )
    .unwrap();
    let project = import_project(package.clone(), config, false);
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let mesh = decode_tri_mesh(exports[0].data()).unwrap();
    assert_eq!(mesh.bounds(), Some([-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]));

    // Asking for a second item the package doesn't have is an error.
    let mut config = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text("item".into()),
            ciborium::value::Value::Integer(2.into()),
        )]),
        &mut config,
    )
    .unwrap();
    let project = import_project(package, config, false);
    let err = project
        .run(&mut Environment::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("build item"), "unexpected error: {err}");
}

#[test]
fn real_freecad_export_round_trips_through_the_operator() {
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/freecad_button.3mf");
    let bytes = std::fs::read(&fixture).unwrap();
    let project = import_project(bytes.clone(), Vec::new(), false);
    let exports = project
        .run(&mut Environment::new())
        .expect("project run failed");
    let imported = decode_tri_mesh(exports[0].data()).unwrap();

    // The host-side reader keeps the file's vertices (FreeCAD writes seam
    // vertices twice); the operator welds them and lands in metres.
    let direct = read_3mf(&bytes).unwrap();
    assert_eq!(direct.unit, Unit::Millimeter);
    let file_mesh = &direct.items[0];
    let welded = TriMesh::from_soup((0..file_mesh.triangle_count()).map(|t| {
        file_mesh
            .triangle(t)
            .map(|v| file_mesh.position(v as usize))
    }));
    assert!(
        welded.vertex_count() < file_mesh.vertex_count(),
        "seams weld"
    );
    assert_eq!(imported.indices, welded.indices);
    for (a, b) in imported.positions.iter().zip(&welded.positions) {
        assert!((a - b * 1e-3).abs() < 1e-12, "{a} vs {b}");
    }
    // Exporting the imported mesh back out in millimetres reproduces it.
    let mut scaled = imported.clone();
    for v in &mut scaled.positions {
        *v *= 1e3;
    }
    let again = read_3mf(&write_3mf(&scaled, Unit::Millimeter, "button").unwrap()).unwrap();
    assert_eq!(again.items[0].indices, welded.indices);
    for (a, b) in again.items[0].positions.iter().zip(&welded.positions) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}
