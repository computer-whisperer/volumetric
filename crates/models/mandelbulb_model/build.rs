//! Precomputes the catalog metadata CBOR that `get_metadata` in lib.rs
//! serves, so the wasm module carries no runtime CBOR encoder.

fn main() {
    let metadata = volumetric_abi::OperatorMetadata {
        name: "mandelbulb_model".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Mandelbulb".to_string(),
        description: "Mandelbulb fractal solid, a 3D Mandelbrot-like set.".to_string(),
        category: "Primitives".to_string(),
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
