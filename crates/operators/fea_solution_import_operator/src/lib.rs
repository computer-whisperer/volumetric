//! Adopt an externally-solved FEA solution into the timeline — after
//! verifying it.
//!
//! The external compute line (`volumetric_cli fea-export` → `fea-solve` on
//! whatever non-portable hardware → this operator) moves the *search* out
//! of the engine, but never the *judgment*: the solution bundle's claimed
//! equilibrium is re-checked here, deterministically, against the live
//! timeline's mesh and rigid body (see `fea_core::verify` for what
//! "verified" means — the solver's own acceptance criteria). A solution
//! that answers a stale export, was solved sloppily, or was tampered with
//! is rejected with the measured numbers in the error.
//!
//! Output 0 is exactly what `fea_inverse_operator` would have produced
//! (the mesh plus the solved fields), so downstream steps cannot tell the
//! difference.

use volumetric_abi::fea::decode_fea_mesh;
use volumetric_abi::host::{
    input_model_dimensions, input_model_sample, post_output, read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct ImportConfig {
    /// Must match the config the problem was solved under: the verifier
    /// re-derives equilibrium from these.
    youngs_modulus: f64,
    poissons_ratio: f64,
    fixed_boundary: String,
    /// Acceptance threshold on the verifier's relative equilibrium
    /// residual. A solution solved at `cg_tolerance` t verifies near t;
    /// the default accepts the loosest tolerance worth using (1e-4) with
    /// an order of magnitude of headroom.
    residual_tolerance: f64,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            youngs_modulus: 1.0,
            poissons_ratio: 0.3,
            fixed_boundary: "zmin".to_string(),
            residual_tolerance: 1e-3,
        }
    }
}

fn run_import(config: &ImportConfig) -> Result<Vec<u8>, String> {
    let live_mesh = decode_fea_mesh(&read_input(0))?;

    let rigid_dims = input_model_dimensions(1)
        .ok_or_else(|| "input 1 is not a usable rigid-body model".to_string())?;
    if rigid_dims != 3 {
        return Err(format!(
            "the rigid body must be a 3D model; input has {rigid_dims} dimensions"
        ));
    }

    let solution = fea_bundle::decode_solution(&read_input(2))?;
    let solved_mesh = decode_fea_mesh(&solution.mesh)?;

    // The solution must answer *this* timeline's problem: same nodes, same
    // struts. Positions compare exactly — the export carried these bytes
    // verbatim, so any drift means the timeline changed since the export.
    if solved_mesh.element_kind != live_mesh.element_kind
        || solved_mesh.node_positions != live_mesh.node_positions
        || solved_mesh.connectivity != live_mesh.connectivity
    {
        return Err(format!(
            "the solution answers a different problem than the current \
             timeline produces ({} nodes / {} elements vs {} / {}): the \
             mesh changed after the export — re-run fea-export and solve \
             again (solution provenance: backend {}, problem blake3 {})",
            solved_mesh.node_count(),
            solved_mesh.element_count(),
            live_mesh.node_count(),
            live_mesh.element_count(),
            solution.stats.backend,
            hex_prefix(&solution.problem_blake3),
        ));
    }

    let solve_config = fea_core::SolveConfig {
        material: fea_core::Material {
            youngs_modulus: config.youngs_modulus,
            poissons_ratio: config.poissons_ratio,
        },
        fixed_boundary: fea_core::FixedBoundary::parse(&config.fixed_boundary)?,
        ..Default::default()
    };
    let mut rigid = |p: [f64; 3]| {
        input_model_sample(1, &p, 3)
            .map(|samples| is_occupied(samples[0]))
            .unwrap_or(false)
    };
    let report = fea_core::verify::verify_frame_solution(&solved_mesh, &mut rigid, &solve_config)?;

    let mut failures = Vec::new();
    // NaN must fail, not pass: compare in the rejecting direction.
    if report.relative_residual > config.residual_tolerance || report.relative_residual.is_nan() {
        failures.push(format!(
            "equilibrium residual {:.3e} exceeds the acceptance threshold \
             {:.1e}",
            report.relative_residual, config.residual_tolerance
        ));
    }
    if report.contact_set_changes > 0 {
        failures.push(format!(
            "{} node(s) penetrate the rigid body or drifted off its \
             surface",
            report.contact_set_changes
        ));
    }
    if report.contact_releases > 0 {
        failures.push(format!(
            "{} active contact(s) pull on the rigid body",
            report.contact_releases
        ));
    }
    if !failures.is_empty() {
        return Err(format!(
            "solution rejected: {} (backend {}, {} active contacts, solved \
             in {:.1}s, problem blake3 {}) — the solution does not satisfy \
             this timeline's physics; re-export and re-solve",
            failures.join("; "),
            solution.stats.backend,
            report.active_contacts,
            solution.stats.wall_seconds,
            hex_prefix(&solution.problem_blake3),
        ));
    }

    Ok(solution.mesh)
}

fn hex_prefix(digest: &[u8]) -> String {
    digest
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(3);
        if buf.is_empty() {
            ImportConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    match run_import(&config) {
        // The verified bytes pass through as-is; re-encoding the decoded
        // mesh would be equivalent but slower.
        Ok(mesh_bytes) => post_output(0, &mesh_bytes),
        Err(e) => report_error(&format!("FEA solution import failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        OperatorMetadata {
        name: "fea_solution_import_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        docs: String::new(),
        display_name: "FEA Solution Import".to_string(),
        description: "Adopt an externally-solved FEA solution after verifying \
                      its equilibrium against the live mesh and rigid body."
            .to_string(),
        category: "FEA".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<path d="M12 3v12"/>"##,
            r##"<path d="m7 10 5 5 5-5"/>"##,
            r##"<path d="M4 21h16"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::FeaMesh,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::Blob,
            OperatorMetadataInput::CBORConfiguration(
                r#"{ youngs_modulus: float .default 1.0, poissons_ratio: float .default 0.3, fixed_boundary: "zmin" / "zmax" / "xmin" / "xmax" / "ymin" / "ymax" / "none" .default "zmin", residual_tolerance: float .default 1e-3 }"#
                    .to_string(),
            ),
        ],
        input_names: vec![
            "Mesh".to_string(),
            "Rigid body".to_string(),
            "Solution bundle".to_string(),
            "Config".to_string(),
        ],
        outputs: vec![OperatorMetadataOutput::FeaMesh],
        output_names: vec![],
    }
    })
}
