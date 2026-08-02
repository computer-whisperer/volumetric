#![doc = include_str!("../README.md")]

use volumetric_abi::fea::{FeaField, FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{
    input_model_dimensions, input_model_sample, post_output, read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct SolveOperatorConfig {
    youngs_modulus: f64,
    poissons_ratio: f64,
    fixed_boundary: String,
    max_contact_iterations: u32,
    cg_tolerance: f64,
    preconditioner: String,
    schwarz_target_nodes: u32,
    stress_stiffening_passes: u32,
}

impl Default for SolveOperatorConfig {
    fn default() -> Self {
        Self {
            youngs_modulus: 1.0,
            poissons_ratio: 0.3,
            fixed_boundary: "zmin".to_string(),
            max_contact_iterations: 64,
            cg_tolerance: 1e-8,
            preconditioner: "auto".to_string(),
            schwarz_target_nodes: 128,
            stress_stiffening_passes: 0,
        }
    }
}

/// Replace-or-append a field, so re-solving an already-solved mesh doesn't
/// accumulate duplicates.
fn upsert(fields: &mut Vec<FeaField>, field: FeaField) {
    match fields.iter_mut().find(|f| f.name == field.name) {
        Some(existing) => *existing = field,
        None => fields.push(field),
    }
}

fn run_solve(config: &SolveOperatorConfig) -> Result<FeaMesh, String> {
    let mut mesh = decode_fea_mesh(&read_input(0))?;

    let rigid_dims = input_model_dimensions(1)
        .ok_or_else(|| "input 1 is not a usable rigid-body model".to_string())?;
    if rigid_dims != 3 {
        return Err(format!(
            "the rigid body must be a 3D model; input has {rigid_dims} dimensions"
        ));
    }

    let solve_config = fea_core::SolveConfig {
        material: fea_core::Material {
            youngs_modulus: config.youngs_modulus,
            poissons_ratio: config.poissons_ratio,
        },
        fixed_boundary: fea_core::FixedBoundary::parse(&config.fixed_boundary)?,
        max_contact_iterations: config.max_contact_iterations as usize,
        cg_tolerance: config.cg_tolerance,
        preconditioner: fea_core::PrecondChoice::parse(
            &config.preconditioner,
            config.schwarz_target_nodes as usize,
        )?,
        stress_stiffening_passes: config.stress_stiffening_passes as usize,
        ..Default::default()
    };

    let mut rigid = |p: [f64; 3]| {
        input_model_sample(1, &p, 3)
            .map(|samples| is_occupied(samples[0]))
            .unwrap_or(false)
    };
    let result = fea_core::solve(&mesh, &mut rigid, &solve_config)?;

    if !result.stats.converged {
        let hint = match result.stats.failure {
            Some(fea_core::SolveFailure::ContactUnsettled) => format!(
                "the contact active set was still changing after {} contact \
                 iterations (every CG solve converged) — raise \
                 max_contact_iterations",
                result.stats.contact_iterations,
            ),
            Some(fea_core::SolveFailure::CgStalled) | None => format!(
                "CG failed to reach tolerance ({} CG iterations over {} \
                 contact iterations); check that the mesh is held by \
                 fixed_boundary ({}) with no disconnected regions, and that \
                 the rigid body presses toward that face from the opposite \
                 side",
                result.stats.cg_iterations, result.stats.contact_iterations, config.fixed_boundary,
            ),
        };
        return Err(format!(
            "solve did not converge: {hint}; {} active contacts",
            result.stats.active_contacts,
        ));
    }

    upsert(
        &mut mesh.node_fields,
        FeaField {
            name: "displacement".to_string(),
            components: 3,
            data: result.displacement,
        },
    );
    if let Some(rotation) = result.rotation {
        upsert(
            &mut mesh.node_fields,
            FeaField {
                name: "rotation".to_string(),
                components: 3,
                data: rotation,
            },
        );
    }
    upsert(
        &mut mesh.node_fields,
        FeaField {
            name: "contact_force".to_string(),
            components: 3,
            data: result.contact_force,
        },
    );
    upsert(
        &mut mesh.element_fields,
        FeaField {
            name: "strain_energy_density".to_string(),
            components: 1,
            data: result.strain_energy_density,
        },
    );
    mesh.validate()?;
    Ok(mesh)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    // Cancellation is cooperative: the solver polls the host between CG
    // iterations. Mandatory for the threaded variant (the host cannot
    // safely interrupt a guest thread pool) and a faster, cleaner exit
    // for the plain build too.
    fea_core::set_cancel_poll(volumetric_abi::host::cancelled);
    // Threaded builds run the whole body on a host-sized rayon pool torn
    // down before returning; plain builds call straight through.
    volumetric_abi::threading::with_thread_pool(run_body);
}

fn run_body() {
    let config = {
        let buf = read_input(2);
        if buf.is_empty() {
            SolveOperatorConfig::default()
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

    match run_solve(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("FEA solve failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        OperatorMetadata {
        name: "fea_solve_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        docs: include_str!("../README.md").to_string(),
        display_name: "FEA Solve".to_string(),
        description: "Compress an FEA mesh against a rigid implicit body with linear elasticity.".to_string(),
        category: "FEA".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<rect x="4" y="13" width="16" height="8" rx="1"/>"##,
            r##"<path d="M8 3v6"/>"##,
            r##"<path d="m6 7 2 2 2-2"/>"##,
            r##"<path d="M16 3v6"/>"##,
            r##"<path d="m14 7 2 2 2-2"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::FeaMesh,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::CBORConfiguration(
                r#"{ youngs_modulus: float .default 1.0, poissons_ratio: float .default 0.3, fixed_boundary: "zmin" / "zmax" / "xmin" / "xmax" / "ymin" / "ymax" / "none" .default "zmin", max_contact_iterations: int .default 64, cg_tolerance: float .default 1e-8, preconditioner: "auto" / "schwarz" .default "auto", schwarz_target_nodes: int .default 128, stress_stiffening_passes: int .default 0 }"#
                    .to_string(),
            ),
        ],
        input_names: vec!["Mesh".to_string(), "Rigid body".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::FeaMesh],
        output_names: vec![],
    }
    })
}
