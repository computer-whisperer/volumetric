//! Integration tests for the split image pipeline: image_model_operator
//! (image blob → 2D field model) and heightmap_extrude_operator (2D field →
//! 3D solid), replacing the old all-in-one heightmap importer.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

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

/// A hand-assembled 24-bit grayscale BMP. `rows` is in image orientation
/// (row 0 at the top); each value is one 8-bit gray pixel.
fn gray_bmp(rows: &[&[u8]]) -> Vec<u8> {
    let h = rows.len();
    let w = rows[0].len();
    assert!(rows.iter().all(|r| r.len() == w));
    let row_bytes = (w * 3).div_ceil(4) * 4;
    let file_size = 54 + row_bytes * h;

    let mut out = Vec::with_capacity(file_size);
    // File header
    out.extend(b"BM");
    out.extend((file_size as u32).to_le_bytes());
    out.extend([0u8; 4]);
    out.extend(54u32.to_le_bytes());
    // BITMAPINFOHEADER
    out.extend(40u32.to_le_bytes());
    out.extend((w as i32).to_le_bytes());
    out.extend((h as i32).to_le_bytes()); // positive height: rows bottom-up
    out.extend(1u16.to_le_bytes());
    out.extend(24u16.to_le_bytes());
    out.extend([0u8; 24]); // compression, image size, ppm, palette fields
    // Pixel rows, bottom-up, BGR, padded to 4 bytes
    for row in rows.iter().rev() {
        let mut written = 0;
        for &v in row.iter() {
            out.extend([v, v, v]);
            written += 3;
        }
        out.extend(std::iter::repeat_n(0u8, row_bytes - written));
    }
    assert_eq!(out.len(), file_size);
    out
}

/// The test image: top row [255, 0], bottom row [51, 255]. 51/255 = 0.2.
fn test_image() -> Vec<u8> {
    gray_bmp(&[&[255, 0], &[51, 255]])
}

fn image_field_wasm() -> Vec<u8> {
    run_operator(
        "image_model_operator",
        vec![
            test_image(),
            cbor_floats(&[("width", 2.0), ("height", 1.0)]),
        ],
    )
    .expect("image converts")
}

#[test]
fn image_loads_as_a_2d_field() {
    let field = image_field_wasm();
    let mut executor = NativeModelExecutor::new(&field).unwrap();
    assert_eq!(executor.dimensions(), 2);

    // Centered bounds from the config extents.
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-1.0, 1.0));
    assert_eq!((bounds.min(1), bounds.max(1)), (-0.5, 0.5));

    // Corner-aligned pixels, +y = image-up: the image's top row lands at
    // max_y and nothing is mirrored.
    let sample = |e: &mut NativeModelExecutor, x: f64, y: f64| e.sample_nd(&[x, y]).unwrap() as f64;
    assert!((sample(&mut executor, -1.0, 0.5) - 1.0).abs() < 1e-4); // top-left 255
    assert!(sample(&mut executor, 1.0, 0.5).abs() < 1e-4); // top-right 0
    assert!((sample(&mut executor, -1.0, -0.5) - 0.2).abs() < 1e-4); // bottom-left 51
    assert!((sample(&mut executor, 1.0, -0.5) - 1.0).abs() < 1e-4); // bottom-right 255

    // Bilinear in the middle: mean of the four corners.
    assert!((sample(&mut executor, 0.0, 0.0) - 0.55).abs() < 1e-4);

    // Outside the image rectangle the field is 0.
    assert_eq!(sample(&mut executor, 1.5, 0.0), 0.0);
    assert_eq!(sample(&mut executor, 0.0, 0.7), 0.0);
}

#[test]
fn image_default_height_preserves_aspect() {
    // 2x2 image with width 3.0 and height defaulted: a square plane.
    let field = run_operator(
        "image_model_operator",
        vec![test_image(), cbor_floats(&[("width", 3.0)])],
    )
    .expect("image converts");
    let mut executor = NativeModelExecutor::new(&field).unwrap();
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-1.5, 1.5));
    assert_eq!((bounds.min(1), bounds.max(1)), (-1.5, 1.5));
}

#[test]
fn heightmap_extrude_follows_the_field() {
    let extruded = run_operator(
        "heightmap_extrude_operator",
        vec![image_field_wasm(), cbor_floats(&[("scale", 2.0)])],
    )
    .expect("extrude field");

    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    assert_eq!(executor.dimensions(), 3);

    // x/y pass through; z spans [0, z_max] with z_max defaulting to scale.
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-1.0, 1.0));
    assert_eq!((bounds.min(1), bounds.max(1)), (-0.5, 0.5));
    assert_eq!((bounds.min(2), bounds.max(2)), (0.0, 2.0));

    let sample = |e: &mut NativeModelExecutor, p: [f64; 3]| e.sample_nd(&p).unwrap() as f64;
    // Top-left pixel is full white: surface at 2.0.
    assert_eq!(sample(&mut executor, [-1.0, 0.5, 1.9]), 1.0);
    assert_eq!(sample(&mut executor, [-1.0, 0.5, 2.1]), 0.0);
    // Bottom-left is 0.2: surface at 0.4.
    assert_eq!(sample(&mut executor, [-1.0, -0.5, 0.3]), 1.0);
    assert_eq!(sample(&mut executor, [-1.0, -0.5, 0.5]), 0.0);
    // Top-right is black: no geometry at all (0 doesn't clear the clip).
    assert_eq!(sample(&mut executor, [1.0, 0.5, 0.0]), 0.0);
    // Below the base plane is outside.
    assert_eq!(sample(&mut executor, [-1.0, 0.5, -0.1]), 0.0);
    // Outside the image rectangle the field is 0: outside.
    assert_eq!(sample(&mut executor, [1.5, 0.0, 0.1]), 0.0);
}

#[test]
fn heightmap_extrude_clip_drops_low_values() {
    let extruded = run_operator(
        "heightmap_extrude_operator",
        vec![
            image_field_wasm(),
            cbor_floats(&[("scale", 1.0), ("clip", 0.5)]),
        ],
    )
    .expect("extrude field");
    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    // The 0.2 corner is clipped away entirely; the white corner stays.
    assert_eq!(executor.sample_nd(&[-1.0, -0.5, 0.1]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[-1.0, 0.5, 0.1]).unwrap(), 1.0);
}

#[test]
fn heightmap_extrude_max_height_caps_the_solid() {
    let extruded = run_operator(
        "heightmap_extrude_operator",
        vec![
            image_field_wasm(),
            cbor_floats(&[("scale", 2.0), ("max_height", 1.0)]),
        ],
    )
    .expect("extrude field");
    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(2), bounds.max(2)), (0.0, 1.0));
    // The white corner's 2.0 surface is cut at the 1.0 bound.
    assert_eq!(executor.sample_nd(&[-1.0, 0.5, 0.9]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[-1.0, 0.5, 1.1]).unwrap(), 0.0);
}

#[test]
fn extruding_a_3d_model_is_rejected() {
    let err = run_operator(
        "heightmap_extrude_operator",
        vec![wasm_artifact("simple_sphere_model"), Vec::new()],
    )
    .expect_err("heightmap-extruding a 3D model must fail");
    assert!(err.contains("2D"), "{err}");
}

#[test]
fn garbage_image_is_rejected() {
    let err = run_operator(
        "image_model_operator",
        vec![b"not an image".to_vec(), Vec::new()],
    )
    .expect_err("garbage bytes must fail image decode");
    assert!(err.contains("image"), "{err}");
}
