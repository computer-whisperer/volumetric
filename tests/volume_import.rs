//! End-to-end tests of volume_import_operator: an NRRD volume becomes an
//! occupancy + signed-distance model through the real wasm pipeline.
//!
//! Requires the wasm32 artifact:
//!   cargo build --target wasm32-unknown-unknown --release -p volume_import_operator

#![cfg(feature = "native")]

use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset,
    OperatorMetadataInput, OperatorMetadataOutput, Project,
};
use volumetric_abi::{ChannelKind, SIGNED_DISTANCE_CHANNEL_NAME, TSDF_CHANNEL_KIND, is_occupied};

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

/// A raw little-endian float NRRD: |p| - 5 on a 33^3 grid at 1 mm from
/// -16, i.e. the signed distance of a 5 mm sphere, in millimetres.
fn sphere_nrrd() -> Vec<u8> {
    let mut bytes = b"NRRD0004\ntype: float\ndimension: 3\nsizes: 33 33 33\n\
        space directions: (1,0,0) (0,1,0) (0,0,1)\nspace origin: (-16,-16,-16)\n\
        endian: little\nencoding: raw\n\n"
        .to_vec();
    for z in 0..33 {
        for y in 0..33 {
            for x in 0..33 {
                let p = [x as f64 - 16.0, y as f64 - 16.0, z as f64 - 16.0];
                let r = p.iter().map(|c| c * c).sum::<f64>().sqrt();
                bytes.extend(((r - 5.0) as f32).to_le_bytes());
            }
        }
    }
    bytes
}

/// An ASCII 2D NRRD: r - 3.5 on an 11 x 11 unit grid, with the centre
/// (r < 2) unobserved, as a scan of a hollow-looking closed ring.
fn ring_nrrd() -> Vec<u8> {
    let mut text = String::from(
        "NRRD0004\ntype: double\ndimension: 2\nsizes: 11 11\nspacings: 1 1\n\
         axis mins: -5 -5\nencoding: ascii\n\n",
    );
    for y in -5..=5i32 {
        let row: Vec<String> = (-5..=5i32)
            .map(|x| {
                let r = ((x * x + y * y) as f64).sqrt();
                if r < 2.0 {
                    "nan".to_string()
                } else {
                    (r - 3.5).to_string()
                }
            })
            .collect();
        text.push_str(&row.join(" "));
        text.push('\n');
    }
    text.into_bytes()
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

/// Runs the importer on `file` with `config` and returns the model bytes.
fn import(file: Vec<u8>, config: Vec<u8>) -> Result<Vec<u8>, String> {
    let mut project = Project::new();
    project.imports.push(ImportedAsset::operator(
        "importer".to_string(),
        wasm_artifact("volume_import_operator"),
    ));
    project.timeline.push(ExecutionStep {
        operator_id: "importer".to_string(),
        inputs: vec![ExecutionInput::Inline(file), ExecutionInput::Inline(config)],
        outputs: vec!["solid".to_string()],
    });
    project.exports = vec!["solid".to_string()];
    let mut env = Environment::new();
    let exports = project.run(&mut env).map_err(|e| e.to_string())?;
    let solid = &exports[0];
    assert_eq!(solid.type_hint(), Some(AssetTypeHint::Model));
    Ok(solid.data().to_vec())
}

#[test]
fn metadata_declares_a_blob_config_and_model_output() {
    let metadata =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("volume_import_operator"))
            .expect("metadata");
    assert_eq!(metadata.name, "volume_import_operator");
    assert_eq!(metadata.category, "Import");
    assert!(matches!(metadata.inputs[0], OperatorMetadataInput::Blob));
    assert!(matches!(
        metadata.inputs[1],
        OperatorMetadataInput::CBORConfiguration(_)
    ));
    assert_eq!(metadata.outputs, vec![OperatorMetadataOutput::ModelWASM]);
    assert!(metadata.docs.contains("NRRD"));
}

#[test]
fn a_distance_volume_becomes_a_solid_with_the_tsdf_channel() {
    let model = import(
        sphere_nrrd(),
        config(vec![
            ("scale", ciborium::Value::Float(1e-3)),
            ("band", ciborium::Value::Float(3.0)),
        ]),
    )
    .unwrap();
    let mut executor = create_model_executor(&model).expect("model loads");
    assert_eq!(executor.dimensions().unwrap(), 3);
    // The solid (|p| < 5 mm) plus a 3 mm band and one sample of margin.
    assert_eq!(
        executor.get_bounds_nd().unwrap().as_slice(),
        &[-8e-3, 8e-3, -8e-3, 8e-3, -8e-3, 8e-3]
    );
    let format = executor.sample_format().unwrap();
    assert_eq!(format.channels.len(), 2);
    assert_eq!(format.channels[0].kind, ChannelKind::Occupancy);
    assert_eq!(format.channels[1].name, SIGNED_DISTANCE_CHANNEL_NAME);
    assert_eq!(
        format.channels[1].kind,
        ChannelKind::Custom(TSDF_CHANNEL_KIND.to_string())
    );

    let near = |row: Vec<f32>, occupancy: f32, distance: f64| {
        assert_eq!(row[0], occupancy, "{row:?}");
        assert!(
            (row[1] as f64 - distance).abs() < 1e-7,
            "{row:?} vs {distance}"
        );
    };
    near(executor.sample_channels_nd(&[0.0; 3]).unwrap(), 1.0, -3e-3);
    near(
        executor.sample_channels_nd(&[4.5e-3, 0.0, 0.0]).unwrap(),
        1.0,
        -0.5e-3,
    );
    near(
        executor.sample_channels_nd(&[7e-3, 0.0, 0.0]).unwrap(),
        0.0,
        2e-3,
    );
    // Beyond the volume the field is the band, empty, everywhere.
    near(
        executor.sample_channels_nd(&[0.1, 0.0, 0.0]).unwrap(),
        0.0,
        3e-3,
    );
    assert!(is_occupied(executor.sample_nd(&[0.0; 3]).unwrap()));
    assert!(!is_occupied(
        executor.sample_nd(&[0.0, 5.2e-3, 0.0]).unwrap()
    ));
}

#[test]
fn unobserved_pockets_fill_in_where_the_scan_encloses_them() {
    let model = import(
        ring_nrrd(),
        config(vec![("band", ciborium::Value::Float(1.5))]),
    )
    .unwrap();
    let mut executor = create_model_executor(&model).expect("model loads");
    assert_eq!(executor.dimensions().unwrap(), 2);
    let centre = executor.sample_channels_nd(&[0.0, 0.0]).unwrap();
    assert_eq!(centre, vec![1.0, -1.5], "the enclosed pocket is solid");
    let ring = executor.sample_channels_nd(&[2.5, 0.0]).unwrap();
    assert_eq!(ring[0], 1.0);
    assert!((ring[1] + 1.0).abs() < 1e-6, "{ring:?}");
    assert_eq!(executor.sample_channels_nd(&[4.8, 0.0]).unwrap()[0], 0.0);

    let hollow = import(
        ring_nrrd(),
        config(vec![
            ("band", ciborium::Value::Float(1.5)),
            ("unobserved", ciborium::Value::Text("empty".to_string())),
        ]),
    )
    .unwrap();
    let mut executor = create_model_executor(&hollow).expect("model loads");
    assert_eq!(
        executor.sample_channels_nd(&[0.0, 0.0]).unwrap(),
        vec![0.0, 1.5]
    );
}

#[test]
fn a_non_nrrd_file_fails_with_a_pointer() {
    let err = import(b"ply\nformat ascii 1.0\n".to_vec(), vec![]).unwrap_err();
    assert!(err.contains("NRRD"), "{err}");
}
