//! FEA Solve Operator.
//!
//! Compresses an FEA mesh against a rigid implicit body with linear
//! elasticity (see `fea_core` for scope: uniform hex grids, Hooke's law,
//! quasi-static single pose, vertical active-set contact). The rigid body
//! is sampled where the user placed it — position it already
//! interpenetrating the mesh, in the fully pressed pose.
//!
//! Inputs:
//! - Input 0: FeaMesh (from `fea_grid_mesh_operator`; an optional
//!   `stiffness_scale` element field scales element stiffness)
//! - Input 1: ModelWASM — the rigid body (must be 3D)
//! - Input 2: CBOR configuration: `youngs_modulus` (float, default 1.0),
//!   `poissons_ratio` (float, default 0.3), `fixed_boundary` (enum of
//!   xmin/xmax/ymin/ymax/zmin/zmax/none, default zmin)
//!
//! Output 0: the input FeaMesh plus result fields — per-node `displacement`
//! (3) and `contact_force` (3, the interface force map), per-element
//! `strain_energy_density` (1).

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
}

impl Default for SolveOperatorConfig {
    fn default() -> Self {
        Self {
            youngs_modulus: 1.0,
            poissons_ratio: 0.3,
            fixed_boundary: "zmin".to_string(),
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
        ..Default::default()
    };

    let mut rigid = |p: [f64; 3]| {
        input_model_sample(1, &p, 3)
            .map(|samples| is_occupied(samples[0]))
            .unwrap_or(false)
    };
    let result = fea_core::solve(&mesh, &mut rigid, &solve_config)?;

    if !result.stats.converged {
        return Err(format!(
            "solve did not converge ({} contact iterations, {} CG iterations, \
             {} active contacts); check that the mesh is held by fixed_boundary \
             and the rigid body only presses from above",
            result.stats.contact_iterations,
            result.stats.cg_iterations,
            result.stats.active_contacts
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
        inputs: vec![
            OperatorMetadataInput::FeaMesh,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::CBORConfiguration(
                r#"{ youngs_modulus: float .default 1.0, poissons_ratio: float .default 0.3, fixed_boundary: "zmin" / "zmax" / "xmin" / "xmax" / "ymin" / "ymax" / "none" .default "zmin" }"#
                    .to_string(),
            ),
        ],
        outputs: vec![OperatorMetadataOutput::FeaMesh],
    }
    })
}
