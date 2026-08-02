//! FEA Print Drag Operator.
//!
//! Designs the "spine" a strut lattice needs to survive being swished
//! through a resin bath during large-format printing. The lattice acts as
//! a sieve collecting fluid drag over its whole surface; the integrated
//! load must flow through the strut network to the fixed (build-plate)
//! face, and the struts on that load path carry the summed tension of
//! everything downstream of them. The design loop (see `fea_core::drag`)
//! grows per-strut radii — floored at the as-designed radius — until
//! every strut's tensile fiber stress sits under an allowable, so the
//! reinforcement concentrates along the dominant load paths and leaves
//! the rest of the lattice untouched.
//!
//! The pass is qualitative: only the ratio `drag_pressure /
//! allowable_stress` matters, and it is the severity dial. The drag is
//! `drag_pressure` per unit projected frontal area per strut, lumped to
//! the end nodes, pointing away from the fixed face by default (the
//! pull-out stroke — the tension case that tears parts; push-back is
//! compression, where lattice flexibility is tolerated).
//!
//! Inputs:
//! - Input 0: FeaMesh (Bar2 strut lattice with a `radius` element field;
//!   an existing `stiffness_scale` field participates in load
//!   distribution unchanged)
//! - Input 1: CBOR configuration: `youngs_modulus` (float, default 1.0),
//!   `poissons_ratio` (float, default 0.3), `fixed_boundary` (enum of
//!   xmin/xmax/ymin/ymax/zmin/zmax, default zmin — the build-plate face;
//!   "none" is rejected, the drag load needs a reaction point),
//!   `drag_pressure` (float, default 1.0 — drag force per unit projected
//!   frontal area), `allowable_stress` (float, default 1.0 — tensile
//!   fiber stress ceiling), `max_iterations` (int, default 20 — sizing
//!   iterations, one forward solve each), `tolerance` (float, default
//!   0.02 — accept when max utilization <= 1 + tolerance), `exponent`
//!   (float, default 0.5 — radius update damping), `max_radius_scale`
//!   (float, default 10.0 — growth cap as a multiple of each strut's
//!   original radius), `load_direction_x`/`_y`/`_z` (floats, default all
//!   0.0 = auto: away from the fixed face; need not be unit length),
//!   `cg_tolerance` (float, default
//!   1e-8), `preconditioner` (auto/schwarz, default auto — schwarz needs
//!   the threaded operator build), `schwarz_target_nodes` (int, default
//!   128), `stress_stiffening_passes` (int, default 0 — tension-only
//!   geometric stiffness re-solves; the pull stroke tensions the load
//!   path, so 1-2 passes capture the taut-string effect).
//!
//! Output 0: the input FeaMesh with the designed per-element `radius`
//! (replacing the input design, never below it), plus `utilization` (1,
//! tensile fiber stress over the allowable), `axial_force` (1, tension
//! positive — the load-path visualization), `strain_energy_density` (1),
//! and per-node `displacement` (3), `rotation` (3), `drag_force` (3, the
//! applied load).
//!
//! The result is best-effort: if a strut pins at `max_radius_scale` and
//! stays over-utilized, the lattice topology has no adequate load path to
//! thicken — the loop stops early and emits its best iterate, with the
//! over-unity `utilization` field marking where the demanded spine does
//! not fit, and posts a `host.post_warning` advisory naming the saturated
//! strut count, peak utilization, and the severity ratio to turn down.
//! The fix is upstream geometry or severity, not more iterations.

use volumetric_abi::fea::{FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct PrintDragOperatorConfig {
    youngs_modulus: f64,
    poissons_ratio: f64,
    fixed_boundary: String,
    drag_pressure: f64,
    allowable_stress: f64,
    max_iterations: u32,
    tolerance: f64,
    exponent: f64,
    max_radius_scale: f64,
    load_direction_x: f64,
    load_direction_y: f64,
    load_direction_z: f64,
    cg_tolerance: f64,
    preconditioner: String,
    schwarz_target_nodes: u32,
    stress_stiffening_passes: u32,
}

impl Default for PrintDragOperatorConfig {
    fn default() -> Self {
        Self {
            youngs_modulus: 1.0,
            poissons_ratio: 0.3,
            fixed_boundary: "zmin".to_string(),
            drag_pressure: 1.0,
            allowable_stress: 1.0,
            max_iterations: 20,
            tolerance: 0.02,
            exponent: 0.5,
            max_radius_scale: 10.0,
            load_direction_x: 0.0,
            load_direction_y: 0.0,
            load_direction_z: 0.0,
            cg_tolerance: 1e-8,
            preconditioner: "auto".to_string(),
            schwarz_target_nodes: 128,
            stress_stiffening_passes: 0,
        }
    }
}

fn run_drag(config: &PrintDragOperatorConfig) -> Result<FeaMesh, String> {
    let mut mesh = decode_fea_mesh(&read_input(0))?;

    let drag_config = fea_core::DragConfig {
        solve: fea_core::SolveConfig {
            material: fea_core::Material {
                youngs_modulus: config.youngs_modulus,
                poissons_ratio: config.poissons_ratio,
            },
            fixed_boundary: fea_core::FixedBoundary::parse(&config.fixed_boundary)?,
            cg_tolerance: config.cg_tolerance,
            preconditioner: fea_core::PrecondChoice::parse(
                &config.preconditioner,
                config.schwarz_target_nodes as usize,
            )?,
            stress_stiffening_passes: config.stress_stiffening_passes as usize,
            ..Default::default()
        },
        max_iterations: config.max_iterations as usize,
        drag_pressure: config.drag_pressure,
        allowable_stress: config.allowable_stress,
        exponent: config.exponent,
        max_radius_scale: config.max_radius_scale,
        tolerance: config.tolerance,
        load_direction: {
            let direction = [
                config.load_direction_x,
                config.load_direction_y,
                config.load_direction_z,
            ];
            (direction != [0.0; 3]).then_some(direction)
        },
    };

    // Non-convergence is not an error: a strut pinned at max_radius_scale
    // means the lattice topology lacks an adequate load path — emit the
    // best iterate; the over-unity utilization field carries where the
    // spine falls short. It IS worth a loud advisory: without one, the
    // only symptom is a maxed-out radius field and a garbage-magnitude
    // displacement view.
    let result = fea_core::solve_drag(&mesh, &drag_config)?;
    if !result.converged {
        let saturated = result.saturated;
        let strut_count = mesh.element_count();
        volumetric_abi::host::post_warning(&format!(
            "spine sizing did not converge after {} iteration(s): {saturated} of \
             {strut_count} strut(s) pinned at max_radius_scale ({}x) with peak \
             utilization {:.3e} — the demanded spine does not fit this lattice. \
             The severity dial is drag_pressure / allowable_stress (currently \
             {:.3e}); lower it toward the physical ratio or rework the upstream \
             geometry. More iterations will not help.",
            result.iterations,
            config.max_radius_scale,
            result.max_utilization,
            config.drag_pressure / config.allowable_stress,
        ));
    }

    fea_bundle::apply_drag_result(&mut mesh, result);
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
        let buf = read_input(1);
        if buf.is_empty() {
            PrintDragOperatorConfig::default()
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

    match run_drag(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("FEA print-drag design failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        OperatorMetadata {
        name: "fea_print_drag_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "FEA Print Drag".to_string(),
        description: "Grow the strut radii a lattice needs to survive resin print-drag tension.".to_string(),
        category: "FEA".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<path d="M12 18V5"/>"##,
            r##"<path d="m8 9 4-4 4 4"/>"##,
            r##"<path d="M3 20c1.5-1.2 3-1.2 4.5 0s3 1.2 4.5 0 3-1.2 4.5 0 3 1.2 4.5 0"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::FeaMesh,
            OperatorMetadataInput::CBORConfiguration(
                r#"{ youngs_modulus: float .default 1.0, poissons_ratio: float .default 0.3, fixed_boundary: "zmin" / "zmax" / "xmin" / "xmax" / "ymin" / "ymax" .default "zmin", drag_pressure: float .default 1.0, allowable_stress: float .default 1.0, max_iterations: int .default 20, tolerance: float .default 0.02, exponent: float .default 0.5, max_radius_scale: float .default 10.0, load_direction_x: float .default 0.0, load_direction_y: float .default 0.0, load_direction_z: float .default 0.0, cg_tolerance: float .default 1e-8, preconditioner: "auto" / "schwarz" .default "auto", schwarz_target_nodes: int .default 128, stress_stiffening_passes: int .default 0 }"#
                    .to_string(),
            ),
        ],
        input_names: vec!["Mesh".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::FeaMesh],
        output_names: vec![],
    }
    })
}
