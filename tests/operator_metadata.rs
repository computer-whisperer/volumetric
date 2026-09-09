//! Verify every bundled operator's get_metadata() decodes against the host's
//! metadata types (the CBOR contract is name-based, so type drift between an
//! operator and the host shows up here).
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release -p <operator>

#![cfg(feature = "native")]

use volumetric::operator_metadata_from_wasm_bytes;

/// Every operator crate under `crates/operators/` — the directory is the
/// source of truth for what is bundled, so a new operator is covered
/// without editing a list here (a missing artifact fails the test).
fn operator_names() -> Vec<String> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/operators");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("listing {}: {e}", dir.display()))
        .map(|entry| {
            entry
                .expect("directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.ends_with("_operator"))
        .collect();
    names.sort();
    assert!(names.len() > 40, "unexpectedly few operators: {names:?}");
    names
}

#[test]
fn every_operator_metadata_decodes() {
    for name in &operator_names() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/wasm32-unknown-unknown/release")
            .join(format!("{name}.wasm"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "missing wasm artifact {} ({e}); build it with \
                 `cargo build --target wasm32-unknown-unknown --release -p {name}`",
                path.display()
            )
        });

        let metadata = operator_metadata_from_wasm_bytes(&bytes)
            .unwrap_or_else(|e| panic!("{name}: metadata failed to decode: {e}"));

        assert_eq!(&metadata.name, name, "{name}: unexpected metadata name");
        assert!(
            !metadata.inputs.is_empty(),
            "{name}: metadata declares no inputs"
        );
        assert!(
            !metadata.outputs.is_empty(),
            "{name}: metadata declares no outputs"
        );

        // Every declared config schema must parse with the host's CDDL
        // reader — an unparseable schema means no config form in the UI.
        // stl_import is grandfathered: its `translate: [float, float,
        // float]` tuple predates the form parser, which has no fixed-size
        // tuple type yet (the UI falls back to the raw config editor).
        for (idx, input) in metadata.inputs.iter().enumerate() {
            if let volumetric::OperatorMetadataInput::CBORConfiguration(schema) = input {
                let parsed = volumetric::operator_config::parse_schema(schema);
                if name != "stl_import_operator" {
                    parsed.unwrap_or_else(|e| {
                        panic!("{name}: config schema (input {idx}) failed to parse: {e:?}")
                    });
                }
            }
        }
    }
}
