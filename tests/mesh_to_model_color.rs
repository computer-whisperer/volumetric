//! End-to-end tests of mesh_to_model_operator's surface colour: a coloured
//! mesh becomes a model with the sRGB channel trio (vertex colours via the
//! PLY importer, textures via a hand-built TriMesh), a plain mesh stays
//! occupancy-only, and a step authored before the config slot existed
//! still converts.
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release \
//!     -p trimesh_model_template -p mesh_to_model_operator -p ply_import_operator

#![cfg(feature = "native")]

use volumetric::trimesh::{MeshField, MeshTexture, TriMesh, UV_FIELD_NAME, encode_tri_mesh};
use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{Environment, ExecutionInput, ExecutionStep, ImportedAsset, Project};
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

/// CBOR-encodes a config map from `(key, value)` pairs.
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

/// An ASCII PLY of the cube [0, 1]^3 as six quads, each vertex coloured
/// by its corner (red = x, green = y, blue = z).
fn cube_ply() -> Vec<u8> {
    let mut text = String::from(
        "ply\nformat ascii 1.0\n\
         element vertex 8\nproperty float x\nproperty float y\nproperty float z\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\n\
         element face 6\nproperty list uchar int vertex_indices\nend_header\n",
    );
    for i in 0..8 {
        let (x, y, z) = (i & 1, (i >> 1) & 1, (i >> 2) & 1);
        text.push_str(&format!(
            "{x} {y} {z} {} {} {}\n",
            x * 255,
            y * 255,
            z * 255
        ));
    }
    for q in [
        [0, 4, 6, 2],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 2, 3, 1],
        [4, 5, 7, 6],
    ] {
        text.push_str(&format!("4 {} {} {} {}\n", q[0], q[1], q[2], q[3]));
    }
    text.into_bytes()
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A minimal valid PNG (8-bit RGB, stored/uncompressed deflate) of a
/// top-row-first RGB image: what a texture attached to a mesh looks like.
fn png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), (width * height * 3) as usize);
    let chunk = |out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]| {
        out.extend((data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend(data);
        out.extend(&body);
        out.extend(crc32(&body).to_be_bytes());
    };
    // Scanlines, each prefixed with filter type 0.
    let mut raw = Vec::new();
    for row in rgb.chunks((width * 3) as usize) {
        raw.push(0);
        raw.extend(row);
    }
    // zlib stream: header, one final stored block, adler32.
    let mut zlib = vec![0x78, 0x01, 0x01];
    zlib.extend((raw.len() as u16).to_le_bytes());
    zlib.extend((!(raw.len() as u16)).to_le_bytes());
    zlib.extend(&raw);
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend(((b << 16) | a).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend(width.to_be_bytes());
    ihdr.extend(height.to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]); // 8-bit, truecolour, deflate, no filter, no interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib);
    chunk(&mut out, b"IEND", &[]);
    out
}

/// The unit cube with four vertices per face, every face mapped to the
/// whole texture (`u` along the lower lateral axis, `v` along the higher),
/// carrying `texture` as its base-colour image.
fn uv_cube(texture: Vec<u8>) -> TriMesh {
    let mut positions = Vec::new();
    let mut uv = Vec::new();
    let mut indices = Vec::new();
    for axis in 0..3 {
        let (ua, va) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let right_handed = axis != 1;
        for side in [0.0, 1.0] {
            let base = (positions.len() / 3) as u32;
            for (cu, cv) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
                let mut p = [0.0; 3];
                p[axis] = side;
                p[ua] = cu;
                p[va] = cv;
                positions.extend(p);
                uv.extend([cu, cv]);
            }
            if (side == 1.0) == right_handed {
                indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            } else {
                indices.extend([base, base + 2, base + 1, base, base + 3, base + 2]);
            }
        }
    }
    TriMesh {
        positions,
        indices,
        vertex_fields: vec![MeshField {
            name: UV_FIELD_NAME.to_string(),
            components: 2,
            data: uv,
        }],
        face_fields: vec![],
        textures: vec![MeshTexture {
            name: "checker.png".to_string(),
            media_type: "image/png".to_string(),
            bytes: texture,
        }],
    }
}

/// The plain unit cube: eight shared vertices, no fields.
fn plain_cube() -> TriMesh {
    let mut mesh = uv_cube(vec![]);
    mesh.vertex_fields.clear();
    mesh.textures.clear();
    mesh
}

/// Runs mesh_to_model_operator on `mesh_bytes` with the given config
/// inputs (none = a one-input step) and returns the model's bytes.
fn convert(mesh_bytes: Vec<u8>, config_inputs: Vec<ExecutionInput>) -> Vec<u8> {
    convert_with_warnings(mesh_bytes, config_inputs).0
}

/// [`convert`], also returning the warnings the step posted.
fn convert_with_warnings(
    mesh_bytes: Vec<u8>,
    config_inputs: Vec<ExecutionInput>,
) -> (Vec<u8>, Vec<String>) {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "converter".to_string(),
        wasm_artifact("mesh_to_model_operator"),
    ));
    let mut inputs = vec![ExecutionInput::Inline(mesh_bytes)];
    inputs.extend(config_inputs);
    project.timeline.push(ExecutionStep {
        operator_id: "converter".to_string(),
        inputs,
        outputs: vec!["solid".to_string()],
    });
    project.exports = vec!["solid".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");
    (exports[0].data().to_vec(), exports[0].warnings().to_vec())
}

/// Samples the colour trio at `p`, panicking if the model has none.
fn color_at(executor: &mut impl ModelExecutor, trio: [usize; 3], p: [f64; 3]) -> [f32; 3] {
    let row = executor.sample_channels_nd(&p).expect("sample_channels");
    trio.map(|i| row[i])
}

fn assert_rgb(got: [f32; 3], expect: [f32; 3], tol: f32, what: &str) {
    for k in 0..3 {
        assert!(
            (got[k] - expect[k]).abs() <= tol,
            "{what}: {got:?} vs {expect:?}"
        );
    }
}

#[test]
fn ply_vertex_colours_become_the_srgb_trio() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("ply_import_operator"),
    ));
    project.imports.push(ImportedAsset::operator(
        "converter".to_string(),
        wasm_artifact("mesh_to_model_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![
            ExecutionInput::Inline(cube_ply()),
            ExecutionInput::Inline(Vec::new()),
        ],
        outputs: vec!["mesh".to_string()],
    });
    project.timeline.push(ExecutionStep {
        operator_id: "converter".to_string(),
        inputs: vec![
            ExecutionInput::AssetRef("mesh".to_string()),
            ExecutionInput::Inline(Vec::new()),
        ],
        outputs: vec!["solid".to_string()],
    });
    project.exports = vec!["solid".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).expect("project runs");

    let mut executor = create_model_executor(exports[0].data()).expect("model loads");
    let format = executor.sample_format().expect("sample format");
    let trio = format
        .color_trio()
        .expect("coloured mesh declares the sRGB trio");
    assert_eq!(format.channels.len(), 4);

    // Colour = corner position, interpolated across each face: a probe just
    // off the +x face at (y, z) reads (1, y, z); occupancy agrees with
    // plain sampling on both sides of the face.
    for (p, inside, expect) in [
        ([1.05, 0.3, 0.7], false, [1.0, 0.3, 0.7]),
        ([0.95, 0.3, 0.7], true, [1.0, 0.3, 0.7]),
        ([0.2, -0.1, 0.6], false, [0.2, 0.0, 0.6]),
        ([0.5, 0.5, 1.2], false, [0.5, 0.5, 1.0]),
    ] {
        let row = executor.sample_channels_nd(&p).expect("sample_channels");
        assert_eq!(is_occupied(row[0]), inside, "occupancy at {p:?}: {row:?}");
        assert_eq!(
            is_occupied(executor.sample_nd(&p).unwrap()),
            inside,
            "plain sample at {p:?}"
        );
        assert_rgb(
            trio.map(|i| row[i]),
            expect,
            1e-5,
            &format!("colour at {p:?}"),
        );
    }
}

/// Probes every face of the uv cube at its four quadrant centres against
/// the 2×2 `texels` (row-major, top row first).
fn check_textured_cube(model: &[u8], texels: [[u8; 3]; 4], tol: f32) {
    let mut executor = create_model_executor(model).expect("model loads");
    let trio = executor
        .sample_format()
        .expect("sample format")
        .color_trio()
        .expect("textured mesh declares the sRGB trio");
    for axis in 0..3 {
        let (ua, va) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        for off in [-0.05, 1.05] {
            for (u, v) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let mut p = [0.0; 3];
                p[axis] = off;
                p[ua] = u;
                p[va] = v;
                let col = if u < 0.5 { 0 } else { 1 };
                let row = if v < 0.5 { 1 } else { 0 }; // v counts up, rows down
                let expect = texels[row * 2 + col].map(|c| c as f32 / 255.0);
                let got = color_at(&mut executor, trio, p);
                assert_rgb(got, expect, tol, &format!("axis {axis} at {p:?}"));
            }
        }
    }
    assert!(is_occupied(executor.sample_nd(&[0.5, 0.5, 0.5]).unwrap()));
    assert!(!is_occupied(executor.sample_nd(&[1.5, 0.5, 0.5]).unwrap()));
}

#[test]
fn textured_mesh_reads_its_texture_rgb8() {
    let texels = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
    let mesh = uv_cube(png(2, 2, &texels.concat()));
    let model = convert(
        encode_tri_mesh(&mesh),
        vec![ExecutionInput::Inline(config(vec![(
            "texture",
            ciborium::Value::Text("rgb8".to_string()),
        )]))],
    );
    check_textured_cube(&model, texels, 1e-5);
}

#[test]
fn textured_mesh_reads_its_texture_bc1_by_default() {
    // Two colours exact in RGB565, so the lossy default is exact too.
    let texels = [[255u8, 0, 0], [0, 0, 255], [0, 0, 255], [255, 0, 0]];
    let mesh = uv_cube(png(2, 2, &texels.concat()));
    let bc1 = convert(
        encode_tri_mesh(&mesh),
        vec![ExecutionInput::Inline(Vec::new())],
    );
    check_textured_cube(&bc1, texels, 1e-5);

    // BC1 packs a 2x2 image into one 8-byte block; rgb8 stores 12 bytes.
    let rgb8 = convert(
        encode_tri_mesh(&mesh),
        vec![ExecutionInput::Inline(config(vec![(
            "texture",
            ciborium::Value::Text("rgb8".to_string()),
        )]))],
    );
    assert_eq!(bc1.len(), rgb8.len() - 4);

    // `texture: "none"` on a mesh with no vertex colour drops colour.
    let none = convert(
        encode_tri_mesh(&mesh),
        vec![ExecutionInput::Inline(config(vec![(
            "texture",
            ciborium::Value::Text("none".to_string()),
        )]))],
    );
    let mut executor = create_model_executor(&none).expect("model loads");
    assert!(executor.sample_format().unwrap().color_trio().is_none());
}

#[test]
fn plain_mesh_stays_occupancy_only() {
    let model = convert(
        encode_tri_mesh(&plain_cube()),
        vec![ExecutionInput::Inline(Vec::new())],
    );
    let mut executor = create_model_executor(&model).expect("model loads");
    let format = executor.sample_format().expect("sample format");
    assert!(format.color_trio().is_none());
    assert_eq!(format.channels.len(), 1, "{format:?}");
    assert!(
        format
            .channels
            .iter()
            .all(|c| !c.name.starts_with("color_"))
    );
    assert!(is_occupied(executor.sample_nd(&[0.5, 0.5, 0.5]).unwrap()));
    assert!(!is_occupied(executor.sample_nd(&[0.5, 0.5, 1.5]).unwrap()));
}

/// Projects written before the config slot existed carry one input; the
/// host reads the missing slot as empty and the operator uses defaults.
#[test]
fn a_one_input_step_still_converts() {
    let texels = [[255u8, 0, 0], [0, 0, 255], [0, 0, 255], [255, 0, 0]];
    let mesh = uv_cube(png(2, 2, &texels.concat()));
    let model = convert(encode_tri_mesh(&mesh), vec![]);
    check_textured_cube(&model, texels, 1e-5);

    let plain = convert(encode_tri_mesh(&plain_cube()), vec![]);
    let mut executor = create_model_executor(&plain).expect("model loads");
    assert!(executor.sample_format().unwrap().color_trio().is_none());
    assert!(is_occupied(executor.sample_nd(&[0.5, 0.5, 0.5]).unwrap()));
}

#[test]
fn a_bad_texture_setting_is_reported() {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "converter".to_string(),
        wasm_artifact("mesh_to_model_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "converter".to_string(),
        inputs: vec![
            ExecutionInput::Inline(encode_tri_mesh(&plain_cube())),
            ExecutionInput::Inline(config(vec![(
                "texture",
                ciborium::Value::Text("astc".to_string()),
            )])),
        ],
        outputs: vec!["solid".to_string()],
    });
    project.exports = vec!["solid".to_string()];
    let mut env = Environment::new();
    let err = project.run(&mut env).unwrap_err().to_string();
    assert!(err.contains("astc"), "{err}");
}

/// An undecodable texture is a warning, not a failure: the solid still
/// converts, colour falling back to the vertex field or to none.
#[test]
fn an_undecodable_texture_warns_and_falls_back() {
    // A JPEG header with no body.
    let mut mesh = uv_cube(vec![0xFF, 0xD8, 0xFF, 0xE0]);
    mesh.textures[0].name = "broken.jpg".to_string();
    mesh.textures[0].media_type = "image/jpeg".to_string();
    let (model, warnings) = convert_with_warnings(encode_tri_mesh(&mesh), vec![]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("broken.jpg"), "{warnings:?}");
    let mut executor = create_model_executor(&model).expect("model loads");
    assert!(executor.sample_format().unwrap().color_trio().is_none());
    assert!(is_occupied(executor.sample_nd(&[0.5, 0.5, 0.5]).unwrap()));

    // With a vertex colour field present, that is the fallback.
    mesh.vertex_fields.push(MeshField {
        name: volumetric::fea::COLOR_FIELD_NAME.to_string(),
        components: 3,
        data: vec![0.25; 24 * 3],
    });
    let (model, warnings) = convert_with_warnings(encode_tri_mesh(&mesh), vec![]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    let mut executor = create_model_executor(&model).expect("model loads");
    let trio = executor
        .sample_format()
        .unwrap()
        .color_trio()
        .expect("vertex colour");
    assert_rgb(
        color_at(&mut executor, trio, [0.5, 0.5, 1.2]),
        [0.25, 0.25, 0.25],
        1e-6,
        "fallback vertex colour",
    );
}
