//! fea-solve: the external FEA compute line. Consumes a problem bundle
//! (`volumetric_cli fea-export`), solves it with a non-portable backend,
//! and writes a solution bundle for `volumetric_cli fea-import` to adopt —
//! where the engine re-verifies the claimed equilibrium before trusting it
//! (see `fea_core::verify`).
//!
//! v1 backend: `fea_core` compiled natively with threads (rayon + the
//! two-level Schwarz preconditioner). GPU and other accelerator backends
//! belong here too, behind flags — the bundle contract is
//! backend-agnostic, and nothing in this binary is sandboxed or needs to
//! stay deterministic.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use volumetric::fea::{decode_fea_mesh, encode_fea_mesh};
use volumetric::is_occupied;
use volumetric::wasm::native::NativeModelExecutor;

#[derive(Parser, Debug)]
#[command(
    name = "fea-solve",
    about = "Solve an exported FEA problem bundle outside the engine"
)]
struct Args {
    /// Problem bundle (from `volumetric_cli fea-export`)
    problem: PathBuf,

    /// Output solution bundle (defaults to the problem path with a
    /// `.fea-solution.cbor` extension)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let problem_bytes = std::fs::read(&args.problem)
        .with_context(|| format!("Failed to read {}", args.problem.display()))?;
    let problem_blake3 = volumetric::content_fingerprint(&problem_bytes);
    let problem = fea_bundle::decode_problem(&problem_bytes).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        problem.kind == fea_bundle::ProblemKind::Inverse,
        "this solver handles inverse problems; bundle says {:?}",
        problem.kind
    );
    let [case] = problem.load_cases.as_slice() else {
        anyhow::bail!(
            "this solver handles exactly 1 load case; the bundle has {} — \
             multi-load-case solving is not built yet",
            problem.load_cases.len()
        );
    };

    let mesh = decode_fea_mesh(&problem.mesh).map_err(anyhow::Error::msg)?;
    let config =
        fea_bundle::InverseOperatorConfig::decode(&problem.config).map_err(anyhow::Error::msg)?;
    let inverse_config = config.to_inverse_config().map_err(anyhow::Error::msg)?;
    println!(
        "problem: {:?}, {} nodes, {} elements, load case {:?}{}",
        mesh.element_kind,
        mesh.node_count(),
        mesh.element_count(),
        case.name,
        problem
            .source
            .as_deref()
            .map(|s| format!(" (from {s})"))
            .unwrap_or_default(),
    );
    println!("config: {config:?}");

    let mut rigid_exec = NativeModelExecutor::new(&case.rigid_model).context("rigid-body model")?;
    let mut target_exec = NativeModelExecutor::new(&case.target_map).context("target map")?;
    let mut rigid = |p: [f64; 3]| rigid_exec.sample_nd(&p).map(is_occupied).unwrap_or(false);
    // A failed sample surfaces as NaN so fea_core reports the position
    // instead of silently reading a hole in the map as zero pressure.
    let mut target = |p: [f64; 2]| {
        target_exec
            .sample_nd(&p)
            .map(|s| s as f64)
            .unwrap_or(f64::NAN)
    };

    let timer = Instant::now();
    let result = fea_core::solve_inverse(&mesh, &mut rigid, &mut target, &inverse_config)
        .map_err(anyhow::Error::msg)?;
    let wall_seconds = timer.elapsed().as_secs_f64();

    let stats = fea_bundle::SolutionStats {
        backend: "fea_core-native".to_string(),
        wall_seconds,
        inverse_iterations: result.iterations as u32,
        converged: result.converged,
        distribution_error: result.distribution_error,
    };
    println!(
        "solved: {wall_seconds:.2}s, {} inverse iterations, converged={}, \
         distribution_error={:.4} (final solve: {} CG / {} contact iterations)",
        result.iterations,
        result.converged,
        result.distribution_error,
        result.solve.stats.cg_iterations,
        result.solve.stats.contact_iterations,
    );

    let mut solved = mesh;
    fea_bundle::apply_inverse_result(&mut solved, result);
    let solution = fea_bundle::FeaSolution {
        version: fea_bundle::SOLUTION_VERSION,
        problem_blake3: problem_blake3.to_vec(),
        mesh: encode_fea_mesh(&solved),
        stats,
    };
    let bytes = fea_bundle::encode_solution(&solution).map_err(anyhow::Error::msg)?;

    let output = args.output.unwrap_or_else(|| {
        let mut path = args.problem.clone();
        let stem = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("problem")
            .trim_end_matches(".fea-problem.cbor")
            .to_string();
        path.set_file_name(format!("{stem}.fea-solution.cbor"));
        path
    });
    std::fs::write(&output, &bytes)
        .with_context(|| format!("Failed to write {}", output.display()))?;
    println!(
        "wrote {} ({:.1} MiB)",
        output.display(),
        bytes.len() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "adopt it with: volumetric_cli fea-import -p <project.vproj> {}",
        output.display()
    );
    Ok(())
}
