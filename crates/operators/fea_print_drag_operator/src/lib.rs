#![doc = include_str!("../README.md")]

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
            max_iterations: 40,
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

    // Non-convergence is not an error — the best iterate ships either way
    // — but each failure mode deserves its own loud advisory: saturation
    // means the lattice topology cannot fit the demanded spine, a plateau
    // means the design is effectively complete just above the acceptance
    // line, and an iteration-cap exit means sizing was still moving.
    let result = fea_core::solve_drag(&mesh, &drag_config)?;
    if !result.converged {
        let peak = result.max_utilization;
        let message = if result.saturated > 0 {
            format!(
                "spine sizing cannot fit: {} of {} strut(s) pinned at \
                 max_radius_scale ({}x) with peak utilization {peak:.3e} after \
                 {} iteration(s). The severity dial is drag_pressure / \
                 allowable_stress (currently {:.3e}); lower it toward the \
                 physical ratio or rework the upstream geometry — more \
                 iterations will not help.",
                result.saturated,
                mesh.element_count(),
                config.max_radius_scale,
                result.iterations,
                config.drag_pressure / config.allowable_stress,
            )
        } else if result.plateaued {
            format!(
                "spine sizing plateaued at peak utilization {peak:.3e} \
                 (acceptance {:.3}) after {} iteration(s): the design stopped \
                 improving and is effectively complete — accept the best \
                 iterate, or loosen `tolerance` to make this exit quietly.",
                1.0 + config.tolerance,
                result.iterations,
            )
        } else {
            format!(
                "spine sizing stopped at max_iterations ({}) with peak \
                 utilization {peak:.3e} still improving — raise \
                 max_iterations to continue sizing.",
                config.max_iterations,
            )
        };
        volumetric_abi::host::post_warning(&message);
    }

    // The displacement field scales with drag_pressure / youngs_modulus on
    // the as-designed radii; at qualitative parameters it can dwarf the
    // part while the (scale-free) radius/utilization design is perfectly
    // healthy. Say so before someone reads the deformed view literally.
    let max_u = result
        .displacement
        .chunks_exact(3)
        .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
        .fold(0.0f64, f64::max);
    let diagonal = {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in mesh.node_positions.chunks_exact(3) {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        (0..3).map(|c| (hi[c] - lo[c]).powi(2)).sum::<f64>().sqrt()
    };
    if diagonal > 0.0 && max_u > diagonal {
        volumetric_abi::host::post_warning(&format!(
            "the displacement field is qualitative at these parameters: max \
             |u| = {max_u:.3e} exceeds the part (diagonal {diagonal:.3e}). \
             Displacement scales with drag_pressure / youngs_modulus — set \
             youngs_modulus to a physical modulus (~1e9 Pa for green resin) \
             to read real deflections. The radius/utilization design is \
             unaffected.",
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
        docs: include_str!("../README.md").to_string(),
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
                r#"{ youngs_modulus: float .default 1.0, poissons_ratio: float .default 0.3, fixed_boundary: "zmin" / "zmax" / "xmin" / "xmax" / "ymin" / "ymax" .default "zmin", drag_pressure: float .default 1.0, allowable_stress: float .default 1.0, max_iterations: int .default 40, tolerance: float .default 0.02, exponent: float .default 0.5, max_radius_scale: float .default 10.0, load_direction_x: float .default 0.0, load_direction_y: float .default 0.0, load_direction_z: float .default 0.0, cg_tolerance: float .default 1e-8, preconditioner: "auto" / "schwarz" .default "auto", schwarz_target_nodes: int .default 128, stress_stiffening_passes: int .default 0 }"#
                    .to_string(),
            ),
        ],
        variadic_input: None,
        input_names: vec!["Mesh".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::FeaMesh],
        output_names: vec![],
    }
    })
}
