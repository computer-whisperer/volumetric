//! Verify every bundled operator's get_metadata() decodes against the host's
//! metadata types (the CBOR contract is name-based, so type drift between an
//! operator and the host shows up here).
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release -p <operator>

#![cfg(feature = "native")]

use volumetric::operator_metadata_from_wasm_bytes;

const OPERATORS: &[&str] = &[
    "boolean_operator",
    "translate_operator",
    "rotation_operator",
    "scale_operator",
    "lua_script_operator",
    "stl_import_operator",
    "rectangular_prism_operator",
    "heightmap_extrude_operator",
];

#[test]
fn every_operator_metadata_decodes() {
    for name in OPERATORS {
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
    }
}
