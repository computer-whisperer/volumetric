//! FEA Inverse Operator.
//!
//! Backs out per-element stiffness from a desired interface force
//! distribution: repeatedly compresses the mesh against the rigid body (see
//! `fea_core` for scope: Hooke's law, quasi-static single pose, active-set
//! contact) and scales each lateral column's `stiffness_scale` until the
//! relative contact force distribution matches a target pressure map. Hex
//! grids column by grid cell; Bar2 strut lattices bin struts into lateral
//! columns of `column_size` (0 = twice the mean strut length), so the same
//! loop drives per-strut mechanical properties.
//!
//! Matching is relative: with a prescribed rigid pose, absolute forces just
//! scale with global stiffness, so only the *shape* of the map matters
//! ("where firm, where soft"). Feed hex results to `fea_density_operator`
//! for density assignment; strut results carry their converged per-strut
//! `stiffness_scale` to the lattice-model realization step, which owns the
//! scale-to-radius mapping.
//!
//! Inputs:
//! - Input 0: FeaMesh (hex grid or Bar2 strut lattice; an existing
//!   `stiffness_scale` element field is the starting point)
//! - Input 1: ModelWASM — the rigid body (3D), placed in the pressed pose
//! - Input 2: ModelWASM — the target pressure map (2D), sampled across the
//!   interface footprint: the two non-contact axes in ascending order
//!   (fixed_boundary zmin/zmax → (x, y); ymin/ymax → (x, z); xmin/xmax →
//!   (y, z)). Sample values are relative pressures; <= 0 means no force
//!   wanted there.
//! - Input 3: CBOR configuration: `youngs_modulus` (float, default 1.0),
//!   `poissons_ratio` (float, default 0.3), `fixed_boundary` (enum of
//!   xmin/xmax/ymin/ymax/zmin/zmax/none, default zmin), `max_iterations`
//!   (int, default 20), `tolerance` (float, default 0.02 — total variation
//!   distance between normalized distributions), `exponent` (float, default
//!   0.5 — update damping), `min_scale` (float, default 0.01 — stiffness
//!   floor), `column_size` (float, default 0 — Bar2 lateral bin width,
//!   0 = auto), `max_contact_iterations` (int, default 64 — cap on the
//!   contact active-set sweeps within each forward solve; grazing rims on
//!   curved rigid bodies can need a few dozen).
//!
//! Output 0: the input FeaMesh plus the converged per-element
//! `stiffness_scale` (1), per-node `target_force` (1, the matched
//! distribution scaled to the achieved total, comparable to
//! `contact_force`), and the final solve's `displacement` (3),
//! `contact_force` (3), `strain_energy_density` (1), and `rotation` (3,
//! Bar2 frame meshes only).

use volumetric_abi::fea::{FeaField, FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{
    input_model_dimensions, input_model_sample, post_output, read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

#[derive(Clone, Debug, serde::Deserialize)]
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
    max_contact_iterations: u32,
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
            max_contact_iterations: 64,
        }
    }
}

/// Replace-or-append a field, so re-running on an already-solved mesh
/// doesn't accumulate duplicates.
fn upsert(fields: &mut Vec<FeaField>, field: FeaField) {
    match fields.iter_mut().find(|f| f.name == field.name) {
        Some(existing) => *existing = field,
        None => fields.push(field),
    }
}

fn run_inverse(config: &InverseOperatorConfig) -> Result<FeaMesh, String> {
    let mut mesh = decode_fea_mesh(&read_input(0))?;

    let rigid_dims = input_model_dimensions(1)
        .ok_or_else(|| "input 1 is not a usable rigid-body model".to_string())?;
    if rigid_dims != 3 {
        return Err(format!(
            "the rigid body must be a 3D model; input has {rigid_dims} dimensions"
        ));
    }
    let target_dims = input_model_dimensions(2)
        .ok_or_else(|| "input 2 is not a usable target-map model".to_string())?;
    if target_dims != 2 {
        return Err(format!(
            "the target force map must be a 2D model (sampled across the \
             interface footprint); input has {target_dims} dimensions"
        ));
    }

    let inverse_config = fea_core::InverseConfig {
        solve: fea_core::SolveConfig {
            material: fea_core::Material {
                youngs_modulus: config.youngs_modulus,
                poissons_ratio: config.poissons_ratio,
            },
            fixed_boundary: fea_core::FixedBoundary::parse(&config.fixed_boundary)?,
            max_contact_iterations: config.max_contact_iterations as usize,
            ..Default::default()
        },
        max_iterations: config.max_iterations as usize,
        tolerance: config.tolerance,
        exponent: config.exponent,
        min_scale: config.min_scale,
        column_size: config.column_size,
    };

    let mut rigid = |p: [f64; 3]| {
        input_model_sample(1, &p, 3)
            .map(|samples| is_occupied(samples[0]))
            .unwrap_or(false)
    };
    // A failed sample surfaces as NaN so fea_core reports the position
    // instead of silently reading a hole in the map as zero pressure.
    let mut target = |p: [f64; 2]| {
        input_model_sample(2, &p, 2)
            .map(|samples| samples[0] as f64)
            .unwrap_or(f64::NAN)
    };
    let result = fea_core::solve_inverse(&mesh, &mut rigid, &mut target, &inverse_config)?;

    if !result.converged {
        // The dominant stall mode is scales frozen against the min_scale
        // floor: the scenario demands more stiffness contrast than the
        // floor permits, and no amount of extra iterations moves the
        // clamped fixed point. Diagnose it so the message points at the
        // knob that actually helps.
        let floored = result
            .stiffness_scale
            .iter()
            .filter(|&&s| s <= config.min_scale * 1.0001)
            .count();
        let floor_share = floored as f64 / result.stiffness_scale.len().max(1) as f64;
        let hint = if floor_share > 0.25 {
            format!(
                "{:.0}% of elements sit at the min_scale floor ({}) — the target \
                 needs more stiffness contrast than the floor allows; lower \
                 min_scale",
                floor_share * 100.0,
                config.min_scale
            )
        } else {
            "raise max_iterations, loosen tolerance, or check that the target \
             map covers the contact patch"
                .to_string()
        };
        return Err(format!(
            "did not reach the target distribution in {} iterations (final \
             relative error {:.4}, tolerance {}); {hint}",
            result.iterations, result.distribution_error, config.tolerance
        ));
    }

    upsert(
        &mut mesh.element_fields,
        FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: result.stiffness_scale,
        },
    );
    upsert(
        &mut mesh.node_fields,
        FeaField {
            name: "target_force".to_string(),
            components: 1,
            data: result.target_force,
        },
    );
    upsert(
        &mut mesh.node_fields,
        FeaField {
            name: "displacement".to_string(),
            components: 3,
            data: result.solve.displacement,
        },
    );
    if let Some(rotation) = result.solve.rotation {
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
            data: result.solve.contact_force,
        },
    );
    upsert(
        &mut mesh.element_fields,
        FeaField {
            name: "strain_energy_density".to_string(),
            components: 1,
            data: result.solve.strain_energy_density,
        },
    );
    mesh.validate()?;
    Ok(mesh)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(3);
        if buf.is_empty() {
            InverseOperatorConfig::default()
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

    match run_inverse(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("FEA inverse solve failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        OperatorMetadata {
        name: "fea_inverse_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        inputs: vec![
            OperatorMetadataInput::FeaMesh,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::CBORConfiguration(
                r#"{ youngs_modulus: float .default 1.0, poissons_ratio: float .default 0.3, fixed_boundary: "zmin" / "zmax" / "xmin" / "xmax" / "ymin" / "ymax" / "none" .default "zmin", max_iterations: int .default 20, tolerance: float .default 0.02, exponent: float .default 0.5, min_scale: float .default 0.01, column_size: float .default 0.0, max_contact_iterations: int .default 64 }"#
                    .to_string(),
            ),
        ],
        input_names: vec![
            "Mesh".to_string(),
            "Rigid body".to_string(),
            "Target force map (2D)".to_string(),
            "Config".to_string(),
        ],
        outputs: vec![OperatorMetadataOutput::FeaMesh],
    }
    })
}
