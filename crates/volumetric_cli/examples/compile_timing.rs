// Timing probe: how long do cold operator metadata queries take?
// No args: all bundled operators (the bundled_operator_versions cost).
// With a name: just that operator, cold then warm.
fn main() {
    let total = std::time::Instant::now();
    match std::env::args().nth(1) {
        Some(name) => {
            let asset = volumetric_assets::get_operator(&name).expect("operator");
            let t = std::time::Instant::now();
            let meta =
                volumetric::operator_metadata_from_wasm_bytes(asset.bytes).expect("metadata");
            println!(
                "{name}: {} bytes, cold metadata query {:.0} ms ({} inputs)",
                asset.bytes.len(),
                t.elapsed().as_secs_f64() * 1000.0,
                meta.inputs.len()
            );
            let t = std::time::Instant::now();
            let _ = volumetric::operator_metadata_from_wasm_bytes(asset.bytes).expect("metadata");
            println!("  warm: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        }
        None => {
            for asset in volumetric_assets::operators() {
                let t = std::time::Instant::now();
                let _ =
                    volumetric::operator_metadata_from_wasm_bytes(asset.bytes).expect("metadata");
                println!(
                    "  {}: {:.0} ms ({} KB)",
                    asset.name,
                    t.elapsed().as_secs_f64() * 1000.0,
                    asset.bytes.len() / 1024
                );
            }
            println!(
                "all operators cold: {:.1} s total",
                total.elapsed().as_secs_f64()
            );
        }
    }
}
