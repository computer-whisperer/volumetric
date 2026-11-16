//! End-to-end test of obj_import_operator (OBJ + texture → TriMesh), chained
//! into mesh_to_model_operator for a solid, executed through the real wasm
//! pipeline; plus a real scanner export from import to a cut solid when
//! `VOLUMETRIC_TEST_OBJ` names one.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p obj_import_operator -p mesh_to_model_operator \
//!     -p translate_operator -p rectangular_prism_operator -p boolean_operator

#![cfg(feature = "native")]

use volumetric::trimesh::{UV_FIELD_NAME, decode_tri_mesh};
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

/// The axis-aligned cube [0, 10]^3 as six textured quads: each face maps
/// the whole unit UV square, written as a scanner would (v/vt corners,
/// an mtllib and a usemtl).
fn cube_obj() -> Vec<u8> {
    let mut text = String::from("# cube\nmtllib cube.mtl\no cube\n");
    for i in 0..8 {
        let (x, y, z) = (
            if i & 1 == 0 { 0 } else { 10 },
            if i & 2 == 0 { 0 } else { 10 },
            if i & 4 == 0 { 0 } else { 10 },
        );
        text.push_str(&format!("v {x} {y} {z}\n"));
    }
    text.push_str("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl scan\n");
    // Outward-wound quads (matching the STL/3MF/PLY test cubes), 1-based.
    for q in [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ] {
        text.push_str(&format!(
            "f {}/1 {}/2 {}/3 {}/4\n",
            q[0] + 1,
            q[1] + 1,
            q[2] + 1,
            q[3] + 1
        ));
    }
    text.into_bytes()
}

fn config(entries: Vec<(&str, ciborium::Value)>) -> Vec<u8> {
    let value = ciborium::Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (ciborium::Value::Text(k.to_string()), v))
            .collect(),
    );
    let mut out = Vec::new();
    ciborium::into_writer(&value, &mut out).unwrap();
    out
}

/// A 2x2 RGB PNG: red, green on the top row; blue, white on the bottom.
const PNG_2X2: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0xFD, 0xD4, 0x9A,
    0x73, 0x00, 0x00, 0x00, 0x12, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0xF8, 0xCF, 0xC0, 0xC0,
    0x00, 0xC2, 0x0C, 0xFF, 0x81, 0x00, 0x00, 0x1F, 0xEE, 0x05, 0xFB, 0xF1, 0xAB, 0xBA, 0x77, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

#[test]
fn obj_mesh_imports_with_uvs_and_texture_and_converts_to_a_solid() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("obj_import_operator"),
    ));
    project.imports.push(ImportedAsset::operator(
        "converter".to_string(),
        wasm_artifact("mesh_to_model_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(cube_obj()),
            ExecutionInput::Inline(PNG_2X2.to_vec()),
            ExecutionInput::Inline(config(vec![
                ("scale", ciborium::Value::Float(0.1)),
                ("center", ciborium::Value::Bool(true)),
                (
                    "texture_name",
                    ciborium::Value::Text("cube.png".to_string()),
                ),
            ])),
        ],
        outputs: vec!["mesh".to_string()],
    });
    project.timeline.push(ExecutionStep {
        operator_id: "converter".to_string(),
        inputs: vec![ExecutionInput::AssetRef("mesh".to_string())],
        outputs: vec!["solid".to_string()],
    });
    project.exports = vec!["mesh".to_string(), "solid".to_string()];

    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");
    assert_eq!(exports.len(), 2);

    let mesh_asset = &exports[0];
    assert_eq!(mesh_asset.type_hint(), Some(AssetTypeHint::TriMesh));
    let mesh = decode_tri_mesh(mesh_asset.data()).unwrap();
    // A vertex is a distinct (v, vt) pair: each cube corner is used by
    // three faces at the face-local texture corners 1..4, which coincide
    // for some of its uses, so the 24 corners weld into 20 vertices (more
    // than the 8 positions, fewer than the 24 corners).
    assert_eq!(mesh.vertex_count(), 20, "corners split along UV seams");
    assert_eq!(mesh.triangle_count(), 12);
    // [0, 10]^3 centred then scaled by 0.1: [-0.5, 0.5]^3.
    assert_eq!(mesh.bounds(), Some([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]));
    let uv = mesh.uv_field().expect("uv field");
    assert_eq!(uv.name, UV_FIELD_NAME);
    assert_eq!(uv.data.len(), 20 * 2);
    assert!(uv.data.iter().all(|v| (0.0..=1.0).contains(v)));
    let texture = mesh.base_color_texture().expect("texture attached");
    assert_eq!(texture.name, "cube.png");
    assert_eq!(texture.media_type, "image/png");
    assert_eq!(texture.bytes, PNG_2X2);

    let solid = &exports[1];
    assert_eq!(solid.type_hint(), Some(AssetTypeHint::Model));
    let mut executor = create_model_executor(solid.data()).expect("model loads");
    assert_eq!(executor.dimensions().unwrap(), 3);
    assert!(is_occupied(executor.sample_nd(&[0.0, 0.0, 0.0]).unwrap()));
    assert!(is_occupied(
        executor.sample_nd(&[0.45, -0.45, 0.45]).unwrap()
    ));
    assert!(!is_occupied(executor.sample_nd(&[0.6, 0.0, 0.0]).unwrap()));
}

/// Operator-level warnings reach the host: a texture for a file without
/// texture coordinates is dropped and said so.
#[test]
fn texture_without_uvs_is_dropped_and_the_mesh_still_imports() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("obj_import_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".to_vec()),
            ExecutionInput::Inline(PNG_2X2.to_vec()),
            ExecutionInput::Inline(Vec::new()),
        ],
        outputs: vec!["mesh".to_string()],
    });
    project.exports = vec!["mesh".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");
    let mesh = decode_tri_mesh(exports[0].data()).unwrap();
    assert_eq!(mesh.triangle_count(), 1);
    assert!(mesh.textures.is_empty());
    assert!(mesh.uv_field().is_none());
}

/// Runs one operator on `inputs` and returns its first output's bytes,
/// the warnings it posted and how long it took.
fn run_step(
    operator: &str,
    inputs: Vec<ExecutionInput>,
) -> (Vec<u8>, Vec<String>, std::time::Duration) {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "op".to_string(),
        wasm_artifact(operator),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "op".to_string(),
        inputs,
        outputs: vec!["out".to_string()],
    });
    project.exports = vec!["out".to_string()];
    let start = std::time::Instant::now();
    let mut env = Environment::new();
    let exports = project
        .run(&mut env)
        .unwrap_or_else(|e| panic!("{operator}: {e}"));
    (
        exports[0].data().to_vec(),
        exports[0].warnings().to_vec(),
        start.elapsed(),
    )
}

fn vec3(v: [f64; 3]) -> ExecutionInput {
    ExecutionInput::Inline(v.iter().flat_map(|c| c.to_le_bytes()).collect())
}

/// The sRGB colour a model gives at `p`.
fn color_at(model: &[u8], p: [f64; 3]) -> [f32; 3] {
    let mut executor = create_model_executor(model).expect("model loads");
    let trio = executor
        .sample_format()
        .expect("sample format")
        .color_trio()
        .expect("the model declares the sRGB trio");
    let row = executor.sample_channels_nd(&p).expect("sample_channels");
    trio.map(|i| row[i])
}

/// A real scanner export, start to finish, when `VOLUMETRIC_TEST_OBJ`
/// names one (the `.obj`, with its material library and texture beside
/// it); skipped otherwise. The sidecars are found as the GUI finds them,
/// the mesh becomes a solid at the default texture size and at full
/// size, and the solid is moved and cut: each stage's time and size is
/// printed, and the colour has to come through every one of them.
///
///   VOLUMETRIC_TEST_OBJ=/path/to/scan.obj cargo test --test obj_import -- --nocapture real_scan
#[test]
fn real_scan_keeps_its_colour_through_a_move_and_a_cut() {
    let Some(obj_path) = std::env::var_os("VOLUMETRIC_TEST_OBJ").map(std::path::PathBuf::from)
    else {
        eprintln!("skipping: VOLUMETRIC_TEST_OBJ is not set");
        return;
    };
    let obj = std::fs::read(&obj_path).expect("VOLUMETRIC_TEST_OBJ is readable");
    let import = volumetric::obj_sidecar::resolve_beside(&obj_path, &obj).into_import(obj);
    eprintln!("sidecars: {}", import.note);
    assert_eq!(import.blobs.len(), 2, "a texture beside the OBJ");
    let texture_bytes = import.blobs[1].clone();
    let mut inputs: Vec<ExecutionInput> = import
        .blobs
        .into_iter()
        .map(ExecutionInput::Inline)
        .collect();
    inputs.push(ExecutionInput::Inline(config(
        import
            .config
            .iter()
            .map(|(key, value)| (key.as_str(), ciborium::Value::Text(value.clone())))
            .collect(),
    )));

    let (scan, warnings, elapsed) = run_step("obj_import_operator", inputs);
    let mesh = decode_tri_mesh(&scan).unwrap();
    eprintln!(
        "import: {} vertices, {} triangles, texture {} bytes, {elapsed:?}; warnings {warnings:?}",
        mesh.vertex_count(),
        mesh.triangle_count(),
        texture_bytes.len()
    );
    assert!(mesh.triangle_count() > 0);
    assert!(mesh.uv_field().is_some());
    let texture = mesh.base_color_texture().expect("texture attached");
    assert_eq!(texture.bytes, texture_bytes);
    assert!(texture.media_type.starts_with("image/"));

    // Is the surface closed? Corners are split along UV seams, so edges
    // are counted between positions, not vertex indices.
    let mut ids = std::collections::HashMap::new();
    let mut position_id = |vertex: u32| -> u32 {
        let key = mesh.position(vertex as usize).map(f64::to_bits);
        let next = ids.len() as u32;
        *ids.entry(key).or_insert(next)
    };
    let mut edges: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for t in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle(t).map(&mut position_id);
        for (i, j) in [(a, b), (b, c), (c, a)] {
            *edges.entry((i.min(j), i.max(j))).or_default() += 1;
        }
    }
    let open = edges.values().filter(|&&n| n == 1).count();
    let shared_by_more = edges.values().filter(|&&n| n > 2).count();
    eprintln!(
        "surface: {} edges, {open} on a boundary (0 = closed), {shared_by_more} shared by more than two triangles",
        edges.len()
    );

    // A point just inside the surface, a third of the way through the
    // triangle list, and the box that keeps its half of the mesh.
    let b = mesh.bounds().unwrap();
    let [i, j, k] = mesh.triangle(mesh.triangle_count() / 3);
    let corners = [i, j, k].map(|v| mesh.position(v as usize));
    let on_surface: [f64; 3] =
        std::array::from_fn(|a| (corners[0][a] + corners[1][a] + corners[2][a]) / 3.0);
    let mid_x = (b[0] + b[1]) / 2.0;
    let (x0, x1) = if on_surface[0] < mid_x {
        (b[0] - 1.0, mid_x + (b[1] - b[0]) * 0.2)
    } else {
        (mid_x - (b[1] - b[0]) * 0.2, b[1] + 1.0)
    };

    for (label, max_size) in [
        ("default texture size", None),
        ("full texture size", Some(0)),
    ] {
        let mut inputs = vec![ExecutionInput::Inline(scan.clone())];
        if let Some(max_size) = max_size {
            inputs.push(ExecutionInput::Inline(config(vec![(
                "texture_max_size",
                ciborium::Value::Integer(max_size.into()),
            )])));
        }
        let (solid, warnings, elapsed) = run_step("mesh_to_model_operator", inputs);
        eprintln!(
            "{label}: Mesh to Model {elapsed:?}, model {:.1} MiB; warnings {warnings:?}",
            solid.len() as f64 / 1048576.0
        );
        let colour = color_at(&solid, on_surface);
        eprintln!("{label}: colour at {on_surface:?} is {colour:?}");
        assert!(colour.iter().all(|c| (0.0..=1.0).contains(c)));

        let shift = [0.25, -0.5, 0.125];
        let (moved, _, elapsed) = run_step(
            "translate_operator",
            vec![
                ExecutionInput::Inline(solid.clone()),
                ExecutionInput::Inline(config(vec![
                    ("dx", ciborium::Value::Float(shift[0])),
                    ("dy", ciborium::Value::Float(shift[1])),
                    ("dz", ciborium::Value::Float(shift[2])),
                ])),
            ],
        );
        eprintln!(
            "{label}: translate {elapsed:?}, model {:.1} MiB",
            moved.len() as f64 / 1048576.0
        );
        let there = std::array::from_fn(|a| on_surface[a] + shift[a]);
        assert_eq!(color_at(&moved, there), colour, "{label}: moved");

        let (keep, _, _) = run_step(
            "rectangular_prism_operator",
            vec![
                ExecutionInput::Inline(config(vec![(
                    "mode",
                    ciborium::Value::Text("opposite_corners".to_string()),
                )])),
                vec3([x0, b[2] - 1.0, b[4] - 1.0]),
                vec3([x1, b[3] + 1.0, b[5] + 1.0]),
            ],
        );
        let (cut, _, elapsed) = run_step(
            "boolean_operator",
            vec![
                ExecutionInput::Inline(solid.clone()),
                ExecutionInput::Inline(keep),
                ExecutionInput::Inline(config(vec![(
                    "op",
                    ciborium::Value::Text("intersect".to_string()),
                )])),
            ],
        );
        eprintln!(
            "{label}: intersect {elapsed:?}, model {:.1} MiB",
            cut.len() as f64 / 1048576.0
        );
        assert_eq!(color_at(&cut, on_surface), colour, "{label}: cut");
        // The cut took the far side away.
        let far = [
            if on_surface[0] < mid_x { b[1] } else { b[0] },
            on_surface[1],
            on_surface[2],
        ];
        let mut executor = create_model_executor(&cut).expect("model loads");
        assert!(!is_occupied(executor.sample_nd(&far).unwrap()));
    }
}
