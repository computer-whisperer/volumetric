//! Print-drag spine design over Bar2 strut lattices.
//!
//! Models the resin-print failure mode where the lattice acts as a sieve
//! collecting fluid drag over its whole surface as the plate swishes the
//! part through the bath: the drag on any one strut is trivial, but the
//! integrated load must flow through the strut network to the fixed
//! (build-plate) face, and the struts on that load path see the summed
//! tension of everything they carry. The design loop grows strut radii
//! until every strut's tensile fiber stress sits under an allowable —
//! fully-stressed design with a floor at the as-designed radius, so the
//! result is a reinforced "spine" along the dominant load paths and an
//! untouched lattice everywhere else. Thickened paths attract more load
//! (they stiffen), so the reinforcement self-concentrates over iterations
//! rather than smearing.
//!
//! Deliberately qualitative, matching the rest of the v1 pipeline:
//! - The load is `drag_pressure` per unit projected frontal area
//!   (`2 r L sin(theta)` per strut against the flow direction), lumped
//!   half to each end node. No fixed-end moments: a strut's mid-span
//!   bending under its own drag is not the failure mode — print
//!   sequencing already tolerates local flexing — the accumulated
//!   load-path tension is.
//! - One signed load direction, default pulling the part away from the
//!   fixed face: the pull-out stroke puts the load path in tension, which
//!   is what tears parts; the push-back stroke is compression, where the
//!   lattice's flexibility is tolerated.
//! - The failure metric is the max tensile fiber stress
//!   `N/A + M r / I` over a strut's ends (torsional shear ignored).
//!   Stresses come from internal forces, so a `stiffness_scale` field
//!   from a seating design pass participates in load distribution
//!   without corrupting the stress numbers.
//!
//! Absolute magnitudes are not calibrated: only the ratio
//! `drag_pressure / allowable_stress` matters, and it is the qualitative
//! severity dial. If a strut pins at `max_radius_scale` and stays
//! over-utilized, the lattice topology has no adequate load path to
//! thicken (the returned `saturated` count and `converged` flag report
//! this loudly) — the fix is upstream geometry, not more iterations.
//! Near optimality the fully-stressed update oscillates (thickening the
//! peak strut shifts load to a neighbor whose peak pops right back), so
//! the loop also exits once the best peak stops improving, reported via
//! `plateaued` — that exit is a finished design, not a failure.

use crate::frame::FrameModel;
use crate::{
    Precond, SolveConfig, SolveStats, StiffnessModel, cancel_requested, contact_frame,
    solve_cg_system,
};
#[cfg(feature = "parallel")]
use crate::{PrecondChoice, schwarz};
use volumetric_abi::fea::{FeaElementKind, FeaMesh};

#[derive(Clone, Copy, Debug)]
pub struct DragConfig {
    /// Forward-solve knobs. `fixed_boundary` must name a face (the drag
    /// load needs a reaction point); the contact-specific fields
    /// (`max_contact_iterations`) are unused — there is no rigid body.
    pub solve: SolveConfig,
    /// Cap on sizing iterations (each is one forward solve).
    pub max_iterations: usize,
    /// Drag force per unit projected frontal area.
    pub drag_pressure: f64,
    /// Allowable tensile fiber stress; utilization = stress / allowable.
    pub allowable_stress: f64,
    /// Update damping: radii move by `utilization^exponent` per iteration.
    /// Axial stress at a fixed carried load falls as `1/r^2`, so 0.5 is a
    /// conservative step that lets load redistribution catch up.
    pub exponent: f64,
    /// Cap on radius growth, as a multiple of each strut's original
    /// radius. A strut pinned here while over-utilized means the demanded
    /// spine does not fit the lattice topology.
    pub max_radius_scale: f64,
    /// Acceptance slack: converged when max utilization <= 1 + tolerance.
    pub tolerance: f64,
    /// Flow direction override (need not be unit length). `None` pulls
    /// the part away from the fixed face — the tension stroke.
    pub load_direction: Option<[f64; 3]>,
}

impl Default for DragConfig {
    fn default() -> Self {
        Self {
            solve: SolveConfig::default(),
            max_iterations: 20,
            drag_pressure: 1.0,
            allowable_stress: 1.0,
            exponent: 0.5,
            max_radius_scale: 10.0,
            tolerance: 0.02,
            load_direction: None,
        }
    }
}

/// The lowest-peak-utilization iterate the loop visited. When
/// `max_iterations` runs out (or every over-utilized strut is pinned at
/// `max_radius_scale`) the best effort comes back with `converged`
/// false — `utilization` and `saturated` carry where and how badly the
/// design falls short.
#[derive(Debug)]
pub struct DragResult {
    /// Designed per-strut radius: floored at the original, grown where
    /// the drag tension demands, capped at `max_radius_scale x original`.
    pub radius: Vec<f64>,
    /// Per-strut tensile utilization (max tensile fiber stress over the
    /// allowable) of the returned iterate.
    pub utilization: Vec<f64>,
    /// Per-strut axial force of the returned iterate, tension positive —
    /// the load-path ("spine") visualization field.
    pub axial_force: Vec<f64>,
    /// Per-node applied drag load, xyz interleaved.
    pub drag_force: Vec<f64>,
    /// Per-node displacement / rotation of the returned iterate, xyz
    /// interleaved.
    pub displacement: Vec<f64>,
    pub rotation: Vec<f64>,
    /// Per-strut strain energy per unit strut volume.
    pub strain_energy_density: Vec<f64>,
    /// Forward solves performed (not the index of the returned iterate).
    pub iterations: usize,
    /// Peak utilization of the returned iterate.
    pub max_utilization: f64,
    /// Struts of the returned iterate pinned at `max_radius_scale` while
    /// still over-utilized — nonzero means the lattice has no adequate
    /// load path to thicken.
    pub saturated: usize,
    /// True when max utilization reached `1 + tolerance`.
    pub converged: bool,
    /// True when the loop stopped because the best peak utilization quit
    /// improving (the fully-stressed update oscillates near optimality:
    /// thickening the peak strut shifts load to a neighbor whose peak pops
    /// right back). With `saturated == 0` this means the design is
    /// effectively complete, just above the acceptance line — not a
    /// spine-does-not-fit failure.
    pub plateaued: bool,
    /// Cumulative CG cost over all iterations and stiffening passes.
    pub stats: SolveStats,
}

/// One iterate's full state, kept so the loop can return its best visit.
struct Iterate {
    radius: Vec<f64>,
    utilization: Vec<f64>,
    axial_force: Vec<f64>,
    drag_force: Vec<f64>,
    displacement: Vec<f64>,
    rotation: Vec<f64>,
    strain_energy_density: Vec<f64>,
    max_utilization: f64,
}

/// Per-strut undeformed geometry for the load builder.
struct StrutGeometry {
    nodes: [u32; 2],
    length: f64,
    /// Unit axis, node 0 -> node 1.
    axis: [f64; 3],
}

fn strut_geometry(mesh: &FeaMesh) -> Result<Vec<StrutGeometry>, String> {
    let mut out = Vec::with_capacity(mesh.element_count());
    for e in 0..mesh.element_count() {
        let nodes = mesh.element(e);
        let p1 = mesh.node_position(nodes[0] as usize);
        let p2 = mesh.node_position(nodes[1] as usize);
        let d = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !(length.is_finite() && length > 0.0) {
            return Err(format!(
                "strut {e} has degenerate length {length} (nodes {} and {})",
                nodes[0], nodes[1]
            ));
        }
        out.push(StrutGeometry {
            nodes: [nodes[0], nodes[1]],
            length,
            axis: [d[0] / length, d[1] / length, d[2] / length],
        });
    }
    Ok(out)
}

/// The flow direction: an explicit override, or away from the fixed face
/// (the tension stroke) when none is given.
fn resolve_direction(config: &DragConfig) -> Result<[f64; 3], String> {
    if let Some(d) = config.load_direction {
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !(len.is_finite() && len > 1e-12) {
            return Err(format!(
                "load_direction ({}, {}, {}) is not a usable direction",
                d[0], d[1], d[2]
            ));
        }
        return Ok([d[0] / len, d[1] / len, d[2] / len]);
    }
    let Some((axis, glued_min)) = config.solve.fixed_boundary.axis() else {
        return Err(
            "the drag load needs a reaction point: fixed_boundary cannot be \
             \"none\" (glue the build-plate face)"
                .to_string(),
        );
    };
    let mut dir = [0.0; 3];
    dir[axis] = if glued_min { 1.0 } else { -1.0 };
    Ok(dir)
}

/// Assemble the drag load: per strut, `drag_pressure` times the projected
/// frontal area `2 r L sin(theta)` along `dir`, lumped half to each end.
/// Returns (rhs over `node_count * 6` dofs, per-node 3-component force).
fn build_drag_load(
    geometry: &[StrutGeometry],
    radii: &[f64],
    dir: [f64; 3],
    drag_pressure: f64,
    node_count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut rhs = vec![0.0f64; node_count * 6];
    let mut nodal = vec![0.0f64; node_count * 3];
    for (s, &r) in geometry.iter().zip(radii) {
        let cos = s.axis[0] * dir[0] + s.axis[1] * dir[1] + s.axis[2] * dir[2];
        let sin = (1.0 - cos * cos).max(0.0).sqrt();
        let force = drag_pressure * 2.0 * r * s.length * sin;
        for &node in &s.nodes {
            for c in 0..3 {
                let f = 0.5 * force * dir[c];
                rhs[node as usize * 6 + c] += f;
                nodal[node as usize * 3 + c] += f;
            }
        }
    }
    (rhs, nodal)
}

/// One force-driven forward solve: `K u = rhs` with the glued face held,
/// plus `stress_stiffening_passes` Picard re-solves against the
/// tension-stiffened tangent (the pull stroke tensions the load path, so
/// the taut-string effect is first-order here). Returns the full 6-dof
/// solution and the CG iterations spent.
fn drag_forward(
    mesh: &FeaMesh,
    model: &mut FrameModel,
    constrained: &[bool],
    rhs: &[f64],
    config: &SolveConfig,
) -> Result<(Vec<f64>, usize), String> {
    #[cfg(not(feature = "parallel"))]
    let _ = mesh;
    let mut u = vec![0.0f64; constrained.len()];
    let mut cg_total = 0;
    for pass in 0..=config.stress_stiffening_passes {
        if pass > 0 {
            model.update_prestress(&u);
        }
        let diag = model.diagonal();
        let blocks = model.node_blocks();
        #[cfg(feature = "parallel")]
        let schwarz_precond = match config.preconditioner {
            PrecondChoice::Schwarz(params) => {
                schwarz::SchwarzPrecond::build(mesh, model, constrained, params)
            }
            _ => None,
        };
        #[cfg(feature = "parallel")]
        let precond = match &schwarz_precond {
            Some(schwarz) => Precond::Schwarz(schwarz),
            None => Precond::build(&diag, blocks.as_deref(), 6, constrained),
        };
        #[cfg(not(feature = "parallel"))]
        let precond = Precond::build(&diag, blocks.as_deref(), 6, constrained);
        let (iterations, converged) = solve_cg_system(
            &*model,
            constrained,
            &precond,
            Some(rhs),
            &mut u,
            config.cg_tolerance,
            config.cg_max_iterations,
        );
        cg_total += iterations;
        if cancel_requested() {
            return Err("solve cancelled".to_string());
        }
        if !converged {
            return Err(format!(
                "CG failed to reach tolerance ({iterations} iterations): the \
                 system may be singular — a strut region not connected to the \
                 fixed face cannot react the drag load (run island removal \
                 upstream), or raise cg_max_iterations"
            ));
        }
    }
    Ok((u, cg_total))
}

/// Per-strut (max tensile fiber stress, axial force) at solution `u`.
/// Fiber stress is `N/A + M r / I` taken at the worse end; only tension
/// counts (the tearing metric), so the result is floored at zero.
fn strut_stresses(model: &FrameModel, radii: &[f64], u: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut tensile = Vec::with_capacity(radii.len());
    let mut axial = Vec::with_capacity(radii.len());
    for (e, &r) in radii.iter().enumerate() {
        let [_, m1, f2, m2] = model.strut_local_forces(e, u);
        // Tension shows as a positive axial end force on node 1.
        let n = f2[0];
        let area = std::f64::consts::PI * r * r;
        let inertia = std::f64::consts::PI * r.powi(4) / 4.0;
        let bending = (m1[1] * m1[1] + m1[2] * m1[2])
            .max(m2[1] * m2[1] + m2[2] * m2[2])
            .sqrt();
        tensile.push((n / area + bending * r / inertia).max(0.0));
        axial.push(n);
    }
    (tensile, axial)
}

/// Write `radii` into the mesh's `radius` element field (present by
/// contract — the mesh was validated as a Bar2 frame).
fn set_radius_field(mesh: &mut FeaMesh, radii: &[f64]) {
    let field = mesh
        .element_fields
        .iter_mut()
        .find(|f| f.name == "radius")
        .expect("Bar2 mesh validated with a radius field");
    field.components = 1;
    field.data = radii.to_vec();
}

/// Relative improvement of the best peak utilization that counts as
/// progress; anything less for [`PLATEAU_STALL_ITERATIONS`] consecutive
/// iterations ends the sizing loop. Mid-design sizing improves the peak
/// by whole percents per iteration; the sub-0.3% crawl only appears in
/// the near-optimal whack-a-mole regime, where finishing the last
/// fraction of a percent costs a forward solve per basis point.
const PLATEAU_IMPROVEMENT: f64 = 3e-3;
const PLATEAU_STALL_ITERATIONS: u32 = 3;

/// CG tolerance floor for intermediate sizing solves: the loop only ranks
/// stresses against a few-percent acceptance band, so ~0.1% relative
/// residual is plenty. The final returned fields re-solve at the
/// configured tolerance.
const LOOSE_SIZING_CG_TOLERANCE: f64 = 1e-3;

/// One full design evaluation: assemble the frame at `radii`, apply the
/// drag load, forward-solve at `solve`'s tolerance, and package the
/// resulting fields.
fn evaluate_design(
    work: &mut FeaMesh,
    geometry: &[StrutGeometry],
    radii: &[f64],
    direction: [f64; 3],
    config: &DragConfig,
    solve: &SolveConfig,
    node_count: usize,
) -> Result<(Iterate, usize), String> {
    set_radius_field(work, radii);
    let mut model = FrameModel::new(work, solve.material)?;
    // Glued-face constraints only — contact_frame's scan geometry is
    // unused (there is no rigid body).
    let constrained =
        contact_frame(work, 6, model.length_scale(), solve.fixed_boundary).constrained;
    let (rhs, drag_force) =
        build_drag_load(geometry, radii, direction, config.drag_pressure, node_count);
    let (u, cg_iterations) = drag_forward(work, &mut model, &constrained, &rhs, solve)?;

    let (tensile, axial) = strut_stresses(&model, radii, &u);
    let utilization: Vec<f64> = tensile
        .iter()
        .map(|s| s / config.allowable_stress)
        .collect();
    let max_utilization = utilization.iter().copied().fold(0.0f64, f64::max);

    let mut displacement = Vec::with_capacity(node_count * 3);
    let mut rotation = Vec::with_capacity(node_count * 3);
    for node in 0..node_count {
        displacement.extend_from_slice(&u[node * 6..node * 6 + 3]);
        rotation.extend_from_slice(&u[node * 6 + 3..node * 6 + 6]);
    }
    let strain_energy_density = model.energy_density(&u);
    Ok((
        Iterate {
            radius: radii.to_vec(),
            utilization,
            axial_force: axial,
            drag_force,
            displacement,
            rotation,
            strain_energy_density,
            max_utilization,
        },
        cg_iterations,
    ))
}

/// Design the strut radii that survive the print-drag tension. See the
/// module docs for the model; the returned result carries the final
/// forward solve's fields for inspection.
pub fn solve_drag(mesh: &FeaMesh, config: &DragConfig) -> Result<DragResult, String> {
    if mesh.element_kind != FeaElementKind::Bar2 {
        return Err(format!(
            "print-drag design works on Bar2 strut lattices, got {:?} elements \
             (the sail-and-spine failure mode has no meaning for solid grids)",
            mesh.element_kind
        ));
    }
    mesh.validate()?;
    let nu = config.solve.material.poissons_ratio;
    if !(-1.0 < nu && nu < 0.5) {
        return Err(format!("Poisson's ratio {nu} outside (-1, 0.5)"));
    }
    if config.max_iterations == 0 {
        return Err("max_iterations must be at least 1".to_string());
    }
    if !(config.drag_pressure > 0.0 && config.drag_pressure.is_finite()) {
        return Err(format!(
            "drag_pressure {} must be positive",
            config.drag_pressure
        ));
    }
    if !(config.allowable_stress > 0.0 && config.allowable_stress.is_finite()) {
        return Err(format!(
            "allowable_stress {} must be positive",
            config.allowable_stress
        ));
    }
    if !(config.exponent > 0.0 && config.exponent <= 2.0) {
        return Err(format!(
            "exponent {} outside (0, 2] (1.0 = undamped update)",
            config.exponent
        ));
    }
    if !(config.max_radius_scale >= 1.0 && config.max_radius_scale.is_finite()) {
        return Err(format!(
            "max_radius_scale {} must be at least 1.0 (radii never shrink \
             below the design)",
            config.max_radius_scale
        ));
    }
    if !(config.tolerance > 0.0 && config.tolerance.is_finite()) {
        return Err(format!("tolerance {} must be positive", config.tolerance));
    }
    let direction = resolve_direction(config)?;

    let node_count = mesh.node_count();
    if mesh.element_count() == 0 {
        return Ok(DragResult {
            radius: Vec::new(),
            utilization: Vec::new(),
            axial_force: Vec::new(),
            drag_force: vec![0.0; node_count * 3],
            displacement: vec![0.0; node_count * 3],
            rotation: vec![0.0; node_count * 3],
            strain_energy_density: Vec::new(),
            iterations: 0,
            max_utilization: 0.0,
            saturated: 0,
            converged: true,
            plateaued: false,
            stats: SolveStats {
                converged: true,
                ..Default::default()
            },
        });
    }

    let geometry = strut_geometry(mesh)?;
    // FrameModel::new re-validates radii; this early read feeds the load
    // builder and the growth floor/cap.
    let original: Vec<f64> = mesh
        .element_fields
        .iter()
        .find(|f| f.name == "radius" && f.components == 1)
        .ok_or_else(|| {
            "Bar2 meshes need a scalar `radius` element field (the strut \
             cross-section radius)"
                .to_string()
        })?
        .data
        .clone();

    let mut work = mesh.clone();
    let mut radii = original.clone();
    let mut stats = SolveStats::default();
    let mut iterations = 0;
    let mut best: Option<Iterate> = None;

    // Resolve `auto` by problem size once, for every solve in the loop.
    let mut solve = config.solve;
    solve.preconditioner = solve.preconditioner.resolve(mesh.element_kind, node_count);
    // Intermediate sizing solves only rank stresses against a
    // few-percent acceptance band, so they run at a loosened CG
    // tolerance; the returned fields come from one final solve of the
    // best design at the configured tolerance (below). NOT under stress
    // stiffening: the linear first pass can sag orders of magnitude
    // beyond the stiffened answer, so a loose residual relative to it
    // corrupts the prestress and with it every downstream stress.
    let sizing_cg_tolerance = if solve.stress_stiffening_passes == 0 {
        solve.cg_tolerance.max(LOOSE_SIZING_CG_TOLERANCE)
    } else {
        solve.cg_tolerance
    };
    let sizing_solve = SolveConfig {
        cg_tolerance: sizing_cg_tolerance,
        ..solve
    };

    let mut stalled = 0u32;
    let mut plateaued = false;
    let converged = loop {
        if cancel_requested() {
            return Err("solve cancelled".to_string());
        }
        iterations += 1;
        let (iterate, cg_iterations) = evaluate_design(
            &mut work,
            &geometry,
            &radii,
            direction,
            config,
            &sizing_solve,
            node_count,
        )?;
        stats.cg_iterations += cg_iterations;
        let max_utilization = iterate.max_utilization;
        let utilization = iterate.utilization.clone();

        // Plateau tracking: only a meaningful improvement of the best
        // peak resets the stall counter; the best iterate itself keeps
        // any improvement.
        if best
            .as_ref()
            .is_none_or(|b| max_utilization < b.max_utilization * (1.0 - PLATEAU_IMPROVEMENT))
        {
            stalled = 0;
        } else {
            stalled += 1;
        }
        if best
            .as_ref()
            .is_none_or(|b| max_utilization < b.max_utilization)
        {
            best = Some(iterate);
        }

        if max_utilization <= 1.0 + config.tolerance {
            break true;
        }
        if stalled >= PLATEAU_STALL_ITERATIONS {
            // The peak has quit improving: the update is cycling load
            // between near-critical struts (or creeping below any useful
            // rate). Every further iteration would buy the same nothing.
            plateaued = true;
            break false;
        }
        if iterations >= config.max_iterations {
            break false;
        }

        // Fully-stressed update with a floor at the design radius and a
        // cap at max_radius_scale: over-utilized struts grow, previously
        // grown struts that shed load shrink back toward the floor —
        // that's what concentrates the spine instead of smearing it.
        let mut moved = false;
        for e in 0..radii.len() {
            let updated = (radii[e] * utilization[e].powf(config.exponent))
                .clamp(original[e], original[e] * config.max_radius_scale);
            if (updated - radii[e]).abs() > radii[e] * 1e-9 {
                moved = true;
            }
            radii[e] = updated;
        }
        if !moved {
            // Every over-utilized strut is pinned at the cap: no progress
            // is possible, more iterations would just repeat this solve.
            break false;
        }
    };

    let mut best = best.expect("at least one iterate ran");
    // Acceptance was judged at the loosened tolerance (its stress error
    // sits well inside the acceptance band); refresh the returned fields
    // at the configured tolerance so the emitted result is as accurate
    // as the caller asked for. The loop verdict stands.
    if solve.cg_tolerance < sizing_solve.cg_tolerance {
        let (refreshed, cg_iterations) = evaluate_design(
            &mut work,
            &geometry,
            &best.radius.clone(),
            direction,
            config,
            &solve,
            node_count,
        )?;
        stats.cg_iterations += cg_iterations;
        best = refreshed;
    }
    let saturated = best
        .radius
        .iter()
        .zip(&original)
        .zip(&best.utilization)
        .filter(|((r, orig), u)| {
            **u > 1.0 + config.tolerance && **r >= **orig * config.max_radius_scale * (1.0 - 1e-9)
        })
        .count();
    stats.converged = converged;
    Ok(DragResult {
        radius: best.radius,
        utilization: best.utilization,
        axial_force: best.axial_force,
        drag_force: best.drag_force,
        displacement: best.displacement,
        rotation: best.rotation,
        strain_energy_density: best.strain_energy_density,
        iterations,
        max_utilization: best.max_utilization,
        saturated,
        converged,
        plateaued,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedBoundary, Material};
    use std::f64::consts::PI;
    use volumetric_abi::fea::FeaField;

    const E: f64 = 2.0;

    fn config() -> DragConfig {
        DragConfig {
            solve: SolveConfig {
                material: Material {
                    youngs_modulus: E,
                    poissons_ratio: 0.3,
                },
                fixed_boundary: FixedBoundary::ZMin,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn bar_mesh(nodes: &[[f64; 3]], edges: &[[u32; 2]], radii: &[f64]) -> FeaMesh {
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: nodes.iter().flatten().copied().collect(),
            connectivity: edges.iter().flatten().copied().collect(),
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: radii.to_vec(),
            }],
        }
    }

    /// The T that isolates the load path: a vertical spine strut glued to
    /// the plate carrying a horizontal sail strut. Statically determinate,
    /// so internal forces are exact: the sail collects `2 p r_h` of drag
    /// and the spine carries all of it in tension, plus the bending moment
    /// `p r_h` that the sail's far-end load applies at the joint.
    fn t_mesh(r_spine: f64, r_sail: f64) -> FeaMesh {
        bar_mesh(
            &[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            &[[0, 1], [1, 2]],
            &[r_spine, r_sail],
        )
    }

    /// Exact tensile fiber stress of the T's spine strut: axial `N/A` plus
    /// the joint bending moment's `M r / I`.
    fn t_spine_stress(p: f64, r_spine: f64, r_sail: f64) -> f64 {
        let n = 2.0 * p * r_sail;
        let m = p * r_sail;
        let area = PI * r_spine * r_spine;
        let inertia = PI * r_spine.powi(4) / 4.0;
        n / area + m * r_spine / inertia
    }

    /// Exact tensile fiber stress of the T's sail strut — its own far-end
    /// lumped load bending it at the joint: `(p r L) L r / I = 4 p / (pi
    /// r^2)`. Larger than the spine stress for the radii used here, so it
    /// is the governing (max-utilization) member.
    fn t_sail_stress(p: f64, r_sail: f64) -> f64 {
        4.0 * p / (PI * r_sail * r_sail)
    }

    #[test]
    fn cantilever_root_stress_matches_statics() {
        // A 4-segment cantilever along x, glued at xmin, flow forced along
        // z: a uniformly loaded cantilever via lumped nodal loads. The
        // lumping preserves the total root moment exactly (`M = p r L^2`
        // with L = 1), and the root strut carries no axial force, so the
        // root fiber stress is `M r / I = 4 p / (pi r^2)`. Setting the
        // allowable to exactly that value makes utilization 1.0.
        let (r, p) = (0.05, 0.3);
        let segments = 4;
        let nodes: Vec<[f64; 3]> = (0..=segments)
            .map(|i| [i as f64 / segments as f64, 0.0, 0.0])
            .collect();
        let edges: Vec<[u32; 2]> = (0..segments as u32).map(|i| [i, i + 1]).collect();
        let mesh = bar_mesh(&nodes, &edges, &vec![r; segments]);

        let sigma_root = 4.0 * p / (PI * r * r);
        let result = solve_drag(
            &mesh,
            &DragConfig {
                solve: SolveConfig {
                    fixed_boundary: FixedBoundary::XMin,
                    ..config().solve
                },
                drag_pressure: p,
                allowable_stress: sigma_root,
                load_direction: Some([0.0, 0.0, 1.0]),
                ..config()
            },
        )
        .unwrap();
        assert!(result.converged, "iterations: {}", result.iterations);
        assert_eq!(result.iterations, 1);
        assert!(
            (result.utilization[0] - 1.0).abs() < 1e-6,
            "root utilization {}, expected 1.0",
            result.utilization[0]
        );
        // Transverse load only: no axial force anywhere.
        for (e, n) in result.axial_force.iter().enumerate() {
            assert!(n.abs() < sigma_root * 1e-9, "strut {e} axial {n}");
        }
        // Moments decay toward the tip, so utilization is monotone down.
        for e in 1..segments {
            assert!(
                result.utilization[e] < result.utilization[e - 1],
                "utilization must decay along the span: {:?}",
                result.utilization
            );
        }
        // Radii untouched at utilization <= 1.
        assert_eq!(result.radius, vec![r; segments]);
    }

    #[test]
    fn spine_tension_accumulates_toward_the_plate() {
        // The T: the sail strut collects the drag, the spine carries the
        // integrated load in tension toward the glued face. Flow direction
        // defaults to +z (away from the zmin plate) — the tension stroke.
        let (r_spine, r_sail, p) = (0.04, 0.03, 0.2);
        let mesh = t_mesh(r_spine, r_sail);
        // The sail governs; utilization 1.0 there keeps the first iterate
        // converged so the assertions see the original radii.
        let allowable = t_sail_stress(p, r_sail);
        let result = solve_drag(
            &mesh,
            &DragConfig {
                drag_pressure: p,
                allowable_stress: allowable,
                ..config()
            },
        )
        .unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 1);

        // The spine's axial force is the whole collected drag, in tension.
        let expected_n = 2.0 * p * r_sail;
        assert!(
            (result.axial_force[0] - expected_n).abs() < expected_n * 1e-6,
            "spine axial {}, expected {expected_n}",
            result.axial_force[0]
        );
        // The vertical spine collects no drag itself (parallel to the
        // flow); the joint carries the sail's near half.
        assert!(
            (result.drag_force[3 + 2] - p * r_sail).abs() < p * r_sail * 1e-9,
            "joint drag load {}, expected {}",
            result.drag_force[3 + 2],
            p * r_sail
        );
        // Both struts' utilization match the exact fiber stresses.
        assert!(
            (result.utilization[1] - 1.0).abs() < 1e-6,
            "sail utilization {}, expected 1.0",
            result.utilization[1]
        );
        let expected_spine = t_spine_stress(p, r_spine, r_sail) / allowable;
        assert!(
            (result.utilization[0] - expected_spine).abs() < expected_spine * 1e-6,
            "spine utilization {}, expected {expected_spine}",
            result.utilization[0]
        );
    }

    #[test]
    fn auto_direction_pulls_away_from_a_zmax_plate() {
        // The T hung upside down from a zmax plate: the auto flow flips to
        // -z and the spine must still come out in tension.
        let (r_spine, r_sail, p) = (0.04, 0.03, 0.2);
        let mesh = bar_mesh(
            &[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            &[[0, 1], [1, 2]],
            &[r_spine, r_sail],
        );
        let result = solve_drag(
            &mesh,
            &DragConfig {
                solve: SolveConfig {
                    fixed_boundary: FixedBoundary::ZMax,
                    ..config().solve
                },
                drag_pressure: p,
                allowable_stress: t_sail_stress(p, r_sail),
                ..config()
            },
        )
        .unwrap();
        assert!(result.converged);
        let expected_n = 2.0 * p * r_sail;
        assert!(
            (result.axial_force[0] - expected_n).abs() < expected_n * 1e-6,
            "spine axial {}, expected {expected_n} (tension)",
            result.axial_force[0]
        );
    }

    #[test]
    fn sizing_grows_the_spine_until_it_holds() {
        // Allowable at a quarter of the spine's stress: the loop must grow
        // the spine radius until utilization drops under 1 — and the drag
        // load growing with the thickened radii must not stop it.
        let (r_spine, r_sail, p) = (0.04, 0.03, 0.2);
        let mesh = t_mesh(r_spine, r_sail);
        let allowable = t_spine_stress(p, r_spine, r_sail) / 4.0;
        let result = solve_drag(
            &mesh,
            &DragConfig {
                drag_pressure: p,
                allowable_stress: allowable,
                max_iterations: 40,
                ..config()
            },
        )
        .unwrap();
        assert!(
            result.converged,
            "did not converge: max utilization {} after {} iterations",
            result.max_utilization, result.iterations
        );
        assert!(result.max_utilization <= 1.0 + config().tolerance);
        assert!(
            result.radius[0] > r_spine * 1.3,
            "spine radius {} barely grew from {r_spine}",
            result.radius[0]
        );
        assert!(result.radius[0] >= r_spine && result.radius[1] >= r_sail);
        assert_eq!(result.saturated, 0);
    }

    /// Progress too slow to matter must end the loop, not burn
    /// max_iterations: with a nearly-flat update exponent the peak
    /// improves by well under PLATEAU_IMPROVEMENT per iteration, so the
    /// stall counter fires after a handful of solves and reports the
    /// plateau (distinct from saturation — nothing is pinned).
    #[test]
    fn stalled_progress_plateaus_out_early() {
        let (r_spine, r_sail, p) = (0.04, 0.03, 0.2);
        let mesh = t_mesh(r_spine, r_sail);
        let allowable = t_spine_stress(p, r_spine, r_sail) / 4.0;
        let result = solve_drag(
            &mesh,
            &DragConfig {
                drag_pressure: p,
                allowable_stress: allowable,
                max_iterations: 50,
                exponent: 0.0001,
                ..config()
            },
        )
        .unwrap();
        assert!(!result.converged);
        assert!(result.plateaued, "stalled loop must report the plateau");
        assert_eq!(result.saturated, 0, "nothing is pinned in a crawl");
        assert!(
            result.iterations <= PLATEAU_STALL_ITERATIONS as usize + 2,
            "stall must end the loop after a handful of iterations, ran {}",
            result.iterations
        );
    }

    #[test]
    fn unsatisfiable_demands_saturate_loudly() {
        // A demand no radius under the cap can meet: the loop must stop
        // early (radii pinned, no progress), report converged = false, and
        // count the saturated struts — not spin to max_iterations or grow
        // absurd radii.
        let (r_spine, r_sail, p) = (0.04, 0.03, 0.2);
        let mesh = t_mesh(r_spine, r_sail);
        let result = solve_drag(
            &mesh,
            &DragConfig {
                drag_pressure: p,
                allowable_stress: t_spine_stress(p, r_spine, r_sail) * 1e-6,
                max_radius_scale: 1.5,
                max_iterations: 50,
                ..config()
            },
        )
        .unwrap();
        assert!(!result.converged);
        assert!(result.saturated >= 1, "saturation not reported");
        assert!(
            result.iterations < 50,
            "pinned radii should stop the loop early, ran {}",
            result.iterations
        );
        for (e, (r, orig)) in result.radius.iter().zip([r_spine, r_sail]).enumerate() {
            assert!(
                *r <= orig * 1.5 * (1.0 + 1e-9),
                "strut {e} radius {r} above the cap"
            );
        }
    }

    #[test]
    fn axial_flow_carries_no_load() {
        // A vertical chain with the flow along its own axis: zero frontal
        // area, zero load, immediate convergence with nothing touched.
        let nodes: Vec<[f64; 3]> = (0..=3).map(|i| [0.0, 0.0, i as f64]).collect();
        let edges: Vec<[u32; 2]> = (0..3u32).map(|i| [i, i + 1]).collect();
        let mesh = bar_mesh(&nodes, &edges, &[0.05; 3]);
        let result = solve_drag(&mesh, &config()).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.max_utilization, 0.0);
        assert_eq!(result.radius, vec![0.05; 3]);
        assert!(result.displacement.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn stress_stiffening_keeps_the_load_path_total() {
        // One Picard pass on the T. The geometric stiffness reshapes the
        // displacements (drastically — the linear iterate sags meters),
        // but the flow-axis cut through the spine still carries the whole
        // collected drag: the spine's geometric terms act on its
        // transverse axes, so its axial force stays the exact static
        // total.
        let (r_spine, r_sail, p) = (0.04, 0.03, 0.2);
        let mesh = t_mesh(r_spine, r_sail);
        let result = solve_drag(
            &mesh,
            &DragConfig {
                solve: SolveConfig {
                    stress_stiffening_passes: 1,
                    ..config().solve
                },
                drag_pressure: p,
                allowable_stress: t_sail_stress(p, r_sail) * 4.0,
                ..config()
            },
        )
        .unwrap();
        assert!(result.converged);
        let expected_n = 2.0 * p * r_sail;
        assert!(
            (result.axial_force[0] - expected_n).abs() < expected_n * 1e-6,
            "spine axial {} drifted from {expected_n}",
            result.axial_force[0]
        );
    }

    #[test]
    fn hex_meshes_are_rejected() {
        let mesh = crate::tests::grid_mesh(2, 2, 2, 0.5);
        let err = solve_drag(&mesh, &config()).unwrap_err();
        assert!(err.contains("Bar2"), "unexpected error: {err}");
    }

    #[test]
    fn a_reaction_point_is_required() {
        let mesh = t_mesh(0.04, 0.03);
        let err = solve_drag(
            &mesh,
            &DragConfig {
                solve: SolveConfig {
                    fixed_boundary: FixedBoundary::None,
                    ..config().solve
                },
                ..config()
            },
        )
        .unwrap_err();
        assert!(err.contains("reaction"), "unexpected error: {err}");
    }

    #[test]
    fn degenerate_direction_overrides_are_rejected() {
        let mesh = t_mesh(0.04, 0.03);
        let err = solve_drag(
            &mesh,
            &DragConfig {
                load_direction: Some([0.0, 0.0, 0.0]),
                ..config()
            },
        )
        .unwrap_err();
        assert!(err.contains("load_direction"), "unexpected error: {err}");
    }

    #[test]
    fn disconnected_islands_fail_with_a_pointer_upstream() {
        // A floating strut nowhere near the plate, transverse to the flow
        // so it actually collects load: singular system, and the error
        // should point at island removal rather than reading as a generic
        // CG stall.
        let mesh = bar_mesh(
            &[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [3.0, 3.0, 2.0],
                [4.0, 3.0, 2.0],
            ],
            &[[0, 1], [2, 3]],
            &[0.05, 0.05],
        );
        let err = solve_drag(&mesh, &config()).unwrap_err();
        assert!(err.contains("island"), "unexpected error: {err}");
    }
}
