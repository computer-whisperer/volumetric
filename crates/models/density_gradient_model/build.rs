//! Precomputes the catalog metadata CBOR that `get_metadata` in lib.rs
//! serves, so the wasm module carries no runtime CBOR encoder.

fn main() {
    let metadata = volumetric_abi::OperatorMetadata {
        name: "density_gradient_model".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Density Gradient".to_string(),
        description: "Solid cube with an x-gradient density channel, a playground input for density-driven lattices.".to_string(),
        category: "Lattice".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<path d="M4 4v16"/>"##,
            r##"<path d="M10 4v16"/>"##,
            r##"<path d="M15 4v16"/>"##,
            r##"<path d="M18.5 4v16"/>"##,
            r##"<path d="M21 4v16"/>"##,
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
