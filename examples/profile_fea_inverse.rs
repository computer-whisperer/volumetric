//! Profile the FEA inverse solve natively on the real inputs from a project
//! file — replays exactly what `fea_inverse_operator` does inside wasm, so
//! native-vs-wasm cost and (with fea_core's parallel feature) threading
//! headroom can be measured on the true workload.
//!
//! Usage: cargo run --release --features native --example profile_fea_inverse \
//!            [project.vproj]   (defaults to fea_strut_example.vproj)

use std::time::Instant;

use volumetric::wasm::native::NativeModelExecutor;
use volumetric::{Environment, ExecutionInput, Project};
use volumetric_abi::fea::decode_fea_mesh;
use volumetric_abi::is_occupied;

/// Mirror of the operator's CBOR config (see fea_inverse_operator).
#[derive(Debug, serde::Deserialize)]
#[serde(default)]
struct InverseOperatorConfig {
    youngs_modulus: f64,
    poissons_ratio: f64,
    fixed_boundary: String,
    max_iterations: u32,
    tolerance: f64,
    exponent: f64,
    min_scale: f64,
    column_size: f64,
}

impl Default for InverseOperatorConfig {
    fn default() -> Self {
        Self {
            youngs_modulus: 1.0,
            poissons_ratio: 0.3,
            fixed_boundary: "zmin".to_string(),
            max_iterations: 20,
            tolerance: 0.02,
            exponent: 0.5,
            min_scale: 0.01,
            column_size: 0.0,
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fea_strut_example.vproj".to_string());
    let mut project = Project::load_from_file(std::path::Path::new(&path)).expect("load project");

    // Find the inverse step, then truncate the timeline before it and export
    // its asset inputs instead — the prior steps produce them natively.
    let step_index = project
        .timeline
        .iter()
        .position(|step| step.operator_id.contains("fea_inverse"))
        .expect("project has no fea_inverse step");
    let step = project.timeline[step_index].clone();

    let mut input_ids = Vec::new();
    let mut config_bytes = Vec::new();
    for input in &step.inputs {
        match input {
            ExecutionInput::AssetRef(id) => input_ids.push(id.clone()),
            ExecutionInput::Inline(data) => config_bytes = data.clone(),
        }
    }
    let [mesh_id, rigid_id, target_id] = input_ids.as_slice() else {
        panic!("expected 3 asset inputs, got {:?}", input_ids);
    };

    project.timeline.truncate(step_index);
    project.exports = vec![mesh_id.clone(), rigid_id.clone(), target_id.clone()];

    let mut env = Environment::new();
    let assets = project.run(&mut env).expect("run precursor steps");
    let bytes_of = |id: &str| {
        assets
            .iter()
            .find(|a| a.id() == id)
            .unwrap_or_else(|| panic!("missing asset {id}"))
            .data()
            .to_vec()
    };

    let mesh = decode_fea_mesh(&bytes_of(mesh_id)).expect("decode FeaMesh");
    let config: InverseOperatorConfig = if config_bytes.is_empty() {
        InverseOperatorConfig::default()
    } else {
        ciborium::de::from_reader(std::io::Cursor::new(&config_bytes)).expect("decode config")
    };
    println!(
        "mesh: {:?}, {} nodes, {} elements",
        mesh.element_kind,
        mesh.node_count(),
        mesh.element_count(),
    );
    println!("config: {config:?}");

    let mut rigid_exec = NativeModelExecutor::new(&bytes_of(rigid_id)).expect("rigid model");
    let mut target_exec = NativeModelExecutor::new(&bytes_of(target_id)).expect("target model");
    let mut rigid = |p: [f64; 3]| {
        rigid_exec
            .sample_nd(&p)
            .map(is_occupied)
            .unwrap_or(false)
    };
    let mut target = |p: [f64; 2]| {
        target_exec
            .sample_nd(&p)
            .map(|s| s as f64)
            .unwrap_or(f64::NAN)
    };

    // Experiment knob: override the CG tolerance (default 1e-8) to measure
    // how tight the inner solves actually need to be.
    let cg_tolerance = std::env::var("PROFILE_CG_TOL")
        .ok()
        .and_then(|s| s.parse().ok());
    let inverse_config = fea_core::InverseConfig {
        solve: fea_core::SolveConfig {
            material: fea_core::Material {
                youngs_modulus: config.youngs_modulus,
                poissons_ratio: config.poissons_ratio,
            },
            fixed_boundary: fea_core::FixedBoundary::parse(&config.fixed_boundary)
                .expect("fixed_boundary"),
            cg_tolerance: cg_tolerance.unwrap_or(fea_core::SolveConfig::default().cg_tolerance),
            ..Default::default()
        },
        max_iterations: config.max_iterations as usize,
        tolerance: config.tolerance,
        exponent: config.exponent,
        min_scale: config.min_scale,
        column_size: config.column_size,
    };

    let timer = Instant::now();
    let result =
        fea_core::solve_inverse(&mesh, &mut rigid, &mut target, &inverse_config).expect("solve");
    let elapsed = timer.elapsed().as_secs_f64();

    println!(
        "solve_inverse: {elapsed:.2}s, {} inverse iterations, converged={}, \
         distribution_error={:.4}",
        result.iterations, result.converged, result.distribution_error,
    );
    println!(
        "final forward solve: {} CG iterations, {} contact iterations, converged={}",
        result.solve.stats.cg_iterations,
        result.solve.stats.contact_iterations,
        result.solve.stats.converged,
    );
}
