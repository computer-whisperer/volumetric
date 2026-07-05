//! End-to-end tests of the split STL pipeline: stl_import_operator (STL →
//! TriMesh) and mesh_to_model_operator (TriMesh → sampleable model).
//! Regression suite for the old all-in-one importer, whose generated model
//! had drifted offsets and sampled empty everywhere.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p stl_import_operator -p mesh_to_model_operator

#![cfg(feature = "native")]

use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project,
};
use volumetric_abi::is_occupied;
use volumetric_abi::trimesh::decode_tri_mesh;

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

/// A binary STL of the axis-aligned box [min, max]^3 (12 triangles).
fn box_stl(min: f32, max: f32) -> Vec<u8> {
    let corners: Vec<[f32; 3]> = (0..8)
        .map(|i| {
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
    let mut triangles: Vec<[[f32; 3]; 3]> = Vec::new();
    for q in quads {
        triangles.push([corners[q[0]], corners[q[1]], corners[q[2]]]);
        triangles.push([corners[q[0]], corners[q[2]], corners[q[3]]]);
    }

    let mut out = vec![0u8; 80];
    out.extend((triangles.len() as u32).to_le_bytes());
    for tri in triangles {
        out.extend([0u8; 12]); // normal (ignored)
        for v in tri {
            for c in v {
                out.extend(c.to_le_bytes());
            }
        }
        out.extend([0u8; 2]); // attribute bytes
    }
    out
}

fn import_project(stl: Vec<u8>, with_convert: bool) -> Project {
    let mut timeline = vec![ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(stl),
            ExecutionInput::Inline(Vec::new()),
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
            ImportedAsset::operator("importer".to_string(), wasm_artifact("stl_import_operator")),
            ImportedAsset::operator(
                "converter".to_string(),
                wasm_artifact("mesh_to_model_operator"),
            ),
        ],
        timeline,
        exports,
    }
}

#[test]
fn stl_box_imports_and_converts_to_a_solid() {
    let project = import_project(box_stl(-1.0, 1.0), true);
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");

    // The mesh: welded cube, 8 vertices, 12 triangles.
    let mesh_asset = exports.iter().find(|e| e.id() == "mesh").unwrap();
    assert_eq!(mesh_asset.type_hint(), Some(AssetTypeHint::TriMesh));
    let mesh = decode_tri_mesh(mesh_asset.data()).expect("mesh decodes");
    assert_eq!(mesh.vertex_count(), 8, "corners should weld");
    assert_eq!(mesh.triangle_count(), 12);
    assert_eq!(mesh.bounds(), Some([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]));

    // The solid: a real model with correct bounds and occupancy (the old
    // importer returned zero bounds and sampled empty everywhere).
    let solid_asset = exports.iter().find(|e| e.id() == "solid").unwrap();
    assert_eq!(solid_asset.type_hint(), Some(AssetTypeHint::Model));
    let mut executor = create_model_executor(solid_asset.data()).expect("solid instantiates");
    assert_eq!(executor.dimensions().unwrap(), 3);
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!(bounds.as_slice(), &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);

    for (p, inside) in [
        ([0.0, 0.0, 0.0], true),
        ([0.9, -0.9, 0.9], true),
        ([-0.5, 0.25, -0.75], true),
        ([1.1, 0.0, 0.0], false),
        ([0.0, -1.2, 0.0], false),
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
fn ascii_stl_and_config_transforms_work() {
    // One facet, ASCII, scaled by 2 and translated +10 in x.
    let ascii = b"solid demo
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid demo"
        .to_vec();

    let mut config = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![
            (
                ciborium::value::Value::Text("scale".into()),
                ciborium::value::Value::Float(2.0),
            ),
            (
                ciborium::value::Value::Text("translate".into()),
                ciborium::value::Value::Array(vec![
                    ciborium::value::Value::Float(10.0),
                    ciborium::value::Value::Float(0.0),
                    ciborium::value::Value::Float(0.0),
                ]),
            ),
        ]),
        &mut config,
    )
    .unwrap();

    let mut project = import_project(ascii, false);
    project.timeline[0].inputs[1] = ExecutionInput::Inline(config);

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project run failed");
    let mesh = decode_tri_mesh(exports[0].data()).unwrap();
    assert_eq!(mesh.triangle_count(), 1);
    assert_eq!(mesh.vertex_count(), 3);
    assert_eq!(mesh.bounds(), Some([10.0, 12.0, 0.0, 2.0, 0.0, 0.0]));
}

#[test]
fn non_manifold_stl_imports_as_the_mesh_it_is() {
    // A single free triangle: not a solid, still a perfectly good mesh.
    let mut stl = vec![0u8; 80];
    stl.extend(1u32.to_le_bytes());
    stl.extend([0u8; 12]);
    for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for c in v {
            stl.extend(c.to_le_bytes());
        }
    }
    stl.extend([0u8; 2]);

    let project = import_project(stl, false);
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("open meshes must import");
    let mesh = decode_tri_mesh(exports[0].data()).unwrap();
    assert_eq!(mesh.triangle_count(), 1);
}

#[test]
fn empty_stl_fails_conversion_but_not_import() {
    // Zero triangles: importable as an empty mesh, but meaningless as a
    // solid — the converter must reject it with a clear error.
    let mut stl = vec![0u8; 80];
    stl.extend(0u32.to_le_bytes());

    let project = import_project(stl.clone(), false);
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("empty mesh imports");
    assert_eq!(
        decode_tri_mesh(exports[0].data()).unwrap().triangle_count(),
        0
    );

    let project = import_project(stl, true);
    let mut env = Environment::new();
    let err = project.run(&mut env).expect_err("conversion must fail");
    assert!(
        err.to_string().contains("no triangles"),
        "unexpected error: {err}"
    );
}
