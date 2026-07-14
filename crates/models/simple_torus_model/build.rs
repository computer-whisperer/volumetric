//! Precomputes the catalog metadata CBOR that `get_metadata` in lib.rs
//! serves, so the wasm module carries no runtime CBOR encoder.

fn main() {
    let metadata = volumetric_abi::OperatorMetadata {
        name: "simple_torus_model".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Simple Torus".to_string(),
        description: "Solid torus centered at the origin.".to_string(),
        category: "Primitives".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<ellipse cx="12" cy="12" rx="9" ry="5"/>"##,
            r##"<ellipse cx="12" cy="12" rx="4" ry="2"/>"##,
        )
        .to_string(),
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
