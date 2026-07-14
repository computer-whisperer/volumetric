//! Precomputes the catalog metadata CBOR that `get_metadata` in lib.rs
//! serves, so the wasm module carries no runtime CBOR encoder.

fn main() {
    let metadata = volumetric_abi::OperatorMetadata {
        name: "gyroid_lattice_model".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Gyroid Lattice".to_string(),
        description: "Finite chunk of a gyroid lattice surface.".to_string(),
        category: "Lattice".to_string(),
        icon_svg: String::new(),
        inputs: vec![],
        input_names: vec![],
        outputs: vec![],
    };
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(
        out_dir.join("metadata.cbor"),
        volumetric_abi::encode_metadata(&metadata),
    )
    .unwrap();
}
