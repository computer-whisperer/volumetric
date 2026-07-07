//! Scratch profiler: run ASN2 on a model wasm at a given depth and print the
//! full per-stage report. (Temporary evidence-gathering tool.)

use volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2;
use volumetric::sharp_features::SharpFeatureConfig;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let wasm_path = &args[0];
    let max_depth: usize = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(4);
    let sharp = args.iter().any(|a| a == "--sharp");
    let refine: usize = args
        .iter()
        .position(|a| a == "--refine")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().unwrap())
        .unwrap_or(8);
    let normal_iters: usize = args
        .iter()
        .position(|a| a == "--normal")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().unwrap())
        .unwrap_or(0);

    let simplify_tolerance: Option<f64> = args
        .iter()
        .position(|a| a == "--simplify")
        .map(|i| args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.0));

    let wasm_bytes = std::fs::read(wasm_path).expect("read wasm");
    let config = AdaptiveMeshConfig2 {
        base_resolution: 8,
        discovery_probes: 8,
        max_depth,
        vertex_refinement_iterations: refine,
        normal_sample_iterations: normal_iters,
        normal_epsilon_frac: 0.1,
        num_threads: 0,
        sharp_features: sharp.then(SharpFeatureConfig::default),
        edge_constrained_refinement: args.iter().any(|a| a == "--edge-constrained"),
        decimation: simplify_tolerance.map(|tolerance| {
            volumetric::mesh_decimation::DecimationConfig {
                error_tolerance_cells: tolerance,
                ..Default::default()
            }
        }),
    };
    let result = volumetric::generate_adaptive_mesh_v2_from_bytes(&wasm_bytes, &config)
        .expect("meshing failed");
    result.stats.print_report();
}
