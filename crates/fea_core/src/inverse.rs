//! Inverse stiffness design: iterate the forward solve, scaling each lateral
//! element column's `stiffness_scale` until the relative interface force
//! distribution matches a target pressure map.
//!
//! Scope, matching the forward solver's v1:
//! - Updates are per lateral column of elements — the first-order
//!   series-spring model: under a prescribed rigid pose, scaling a
//!   column's stiffness scales its share of the interface force. The
//!   cross-column coupling (Poisson, shear) is what the outer fixed
//!   point iterates away. Hex grids column by exact grid cell; Bar2
//!   strut meshes bin struts by lateral midpoint (see
//!   [`solve_inverse_frame`]'s docs for how attribution differs).
//! - Relative distribution matching. With a prescribed pose, absolute
//!   forces just scale with global stiffness, so the target and the achieved
//!   forces are each normalized to unit sum before comparison; the answer is
//!   "where firm, where soft", which is what drives density assignment.
//! - Comparison happens in the actuator's space: achieved nodal forces are
//!   split equally among each contact node's adjacent element columns, and
//!   the target is sampled at each contacted column's lateral center. Column
//!   force and column stiffness then correspond one-to-one — a step in the
//!   target lands on column boundaries instead of leaving node lines
//!   straddling it with unreachable demands, and the equal split reproduces
//!   the tributary-area weighting that makes a uniform map an exact fixed
//!   point under a flat press.

use crate::{
    RigidBody, SolveConfig, SolveFailure, SolveResult, SolveStats, detect_uniform_grid, solve,
    stiffness_scales,
};
use std::collections::HashMap;
use volumetric_abi::fea::{FeaField, FeaMesh};

/// A desired relative contact pressure map over the interface footprint.
///
/// Sampled at the undeformed lateral centers of contacted element columns —
/// the two non-contact axes in ascending order (contact along z → (x, y);
/// along y → (x, z); along x → (y, z)). Only the map's shape matters, not
/// its magnitude; values <= 0 mean "no force wanted here".
pub trait TargetMap {
    fn pressure(&mut self, p: [f64; 2]) -> f64;
}

impl<F: FnMut([f64; 2]) -> f64> TargetMap for F {
    fn pressure(&mut self, p: [f64; 2]) -> f64 {
        self(p)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InverseConfig {
    pub solve: SolveConfig,
    /// Cap on forward solves.
    pub max_iterations: usize,
    /// Convergence threshold on the distribution error (total variation
    /// distance between the normalized target and achieved distributions,
    /// in [0, 1]).
    pub tolerance: f64,
    /// Update damping: column scales move by `ratio^exponent` per iteration.
    /// 1.0 is the exact decoupled-columns update; lower is more damped.
    pub exponent: f64,
    /// Floor on `stiffness_scale` (scales are renormalized so the stiffest
    /// element sits at 1.0).
    pub min_scale: f64,
    /// Bar2 strut meshes only: the lateral bin width that groups struts
    /// into columns (hex grids use their cell size). 0 picks twice the
    /// mean strut length — about one lattice cell for the strut families.
    pub column_size: f64,
}

impl Default for InverseConfig {
    fn default() -> Self {
        Self {
            solve: SolveConfig::default(),
            max_iterations: 20,
            tolerance: 0.02,
            exponent: 0.5,
            min_scale: 0.01,
            column_size: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct InverseResult {
    /// The per-element stiffness multipliers that produced `solve`,
    /// renormalized to peak 1.0 and floored at `min_scale`.
    pub stiffness_scale: Vec<f64>,
    /// Per-node target force: the normalized target distribution scaled to
    /// the achieved total, directly comparable to the final `contact_force`
    /// (zero off the contact set).
    pub target_force: Vec<f64>,
    /// The final forward solve.
    pub solve: SolveResult,
    /// Forward solves performed.
    pub iterations: usize,
    /// Final distribution error (total variation distance, in [0, 1]).
    pub distribution_error: f64,
    /// False when `max_iterations` ran out above `tolerance`.
    pub converged: bool,
}

/// A lateral column key and the fraction/force attributed to it.
type ColumnShares = Vec<((i64, i64), f64)>;

/// Failure-specific message for a forward solve that didn't converge —
/// "the contact set needs more sweeps" and "CG ran out" have opposite
/// remedies, so don't let them read the same.
fn forward_failure(iterations: usize, stats: &SolveStats) -> String {
    let reason = match stats.failure {
        Some(SolveFailure::ContactUnsettled) => format!(
            "the contact active set was still changing after {} contact \
             iterations (every CG solve converged; raise \
             max_contact_iterations)",
            stats.contact_iterations,
        ),
        Some(SolveFailure::CgStalled) | None => format!(
            "CG failed to reach tolerance ({} CG iterations over {} contact \
             iterations; the system may be near-singular — check for \
             disconnected or barely-connected regions and extreme \
             stiffness contrast)",
            stats.cg_iterations, stats.contact_iterations,
        ),
    };
    format!(
        "forward solve did not converge at inverse iteration {iterations}: \
         {reason}; {} active contacts",
        stats.active_contacts,
    )
}

/// Write `scales` into the mesh's `stiffness_scale` element field.
fn set_scale_field(mesh: &mut FeaMesh, scales: &[f64]) {
    let data = scales.to_vec();
    match mesh
        .element_fields
        .iter_mut()
        .find(|f| f.name == "stiffness_scale")
    {
        Some(field) => {
            field.components = 1;
            field.data = data;
        }
        None => mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data,
        }),
    }
}

/// Back out the per-element `stiffness_scale` that makes the interface force
/// distribution match `target`. See the module docs for scope; the returned
/// result carries the final forward solve. Hex grids update per lateral
/// grid column; Bar2 strut meshes bin struts into lateral columns of
/// `column_size` (see [`InverseConfig`]).
pub fn solve_inverse(
    mesh: &FeaMesh,
    rigid: &mut dyn RigidBody,
    target: &mut dyn TargetMap,
    config: &InverseConfig,
) -> Result<InverseResult, String> {
    if config.max_iterations == 0 {
        return Err("max_iterations must be at least 1".to_string());
    }
    if !(config.tolerance > 0.0 && config.tolerance.is_finite()) {
        return Err(format!("tolerance {} must be positive", config.tolerance));
    }
    if !(config.exponent > 0.0 && config.exponent <= 2.0) {
        return Err(format!(
            "exponent {} outside (0, 2] (1.0 = undamped update)",
            config.exponent
        ));
    }
    if !(config.min_scale > 0.0 && config.min_scale <= 1.0) {
        return Err(format!(
            "min_scale {} outside (0, 1]; the solver cannot recover columns \
             scaled to exactly zero",
            config.min_scale
        ));
    }

    match mesh.element_kind {
        volumetric_abi::fea::FeaElementKind::Hex8 => {
            solve_inverse_hex(mesh, rigid, target, config)
        }
        volumetric_abi::fea::FeaElementKind::Bar2 => {
            solve_inverse_frame(mesh, rigid, target, config)
        }
    }
}

/// The hex-grid inverse: exact lateral grid columns, target sampled at
/// column centers (uniform column footprints make pressure and per-column
/// force the same shape).
fn solve_inverse_hex(
    mesh: &FeaMesh,
    rigid: &mut dyn RigidBody,
    target: &mut dyn TargetMap,
    config: &InverseConfig,
) -> Result<InverseResult, String> {
    let h = detect_uniform_grid(mesh)?;
    let (contact_axis, contact_sign) = config.solve.fixed_boundary.contact_axis();
    let lateral: [usize; 2] = match contact_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };

    let node_count = mesh.node_count();
    let mut lo = [f64::INFINITY; 3];
    for node in 0..node_count {
        let p = mesh.node_position(node);
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
        }
    }
    let lat_index = |p: [f64; 3]| -> (i64, i64) {
        (
            ((p[lateral[0]] - lo[lateral[0]]) / h).round() as i64,
            ((p[lateral[1]] - lo[lateral[1]]) / h).round() as i64,
        )
    };

    // Elements grouped by lateral column.
    let mut columns: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for e in 0..mesh.element_count() {
        let base = mesh.node_position(mesh.element(e)[0] as usize);
        columns.entry(lat_index(base)).or_default().push(e);
    }

    let mut work = mesh.clone();
    let mut scales = stiffness_scales(mesh)?;
    let mut iterations = 0;

    loop {
        iterations += 1;
        set_scale_field(&mut work, &scales);
        let result = solve(&work, rigid, &config.solve)?;
        if !result.stats.converged {
            return Err(forward_failure(iterations, &result.stats));
        }

        // Split each contact node's compressive force among its adjacent
        // element columns in proportion to their stiffness — the force a
        // shared node transmits comes from its columns in that proportion,
        // and crediting a soft column with an equal share of a stiff
        // neighbor's force leaves it a residual it can never shed. Equal
        // scales degrade to an equal split, which reproduces the
        // tributary-area weighting (interior nodes feed four columns,
        // footprint edges two, corners one) and keeps a uniform map an
        // exact fixed point.
        let col_scale: HashMap<(i64, i64), f64> = columns
            .iter()
            .map(|(c, elements)| {
                let mean = elements.iter().map(|&e| scales[e]).sum::<f64>() / elements.len() as f64;
                (*c, mean)
            })
            .collect();
        let mut contacts: Vec<(usize, ColumnShares)> = Vec::new();
        let mut col_force: HashMap<(i64, i64), f64> = HashMap::new();
        for node in 0..node_count {
            let force = -contact_sign * result.contact_force[node * 3 + contact_axis];
            if force <= 0.0 {
                continue;
            }
            let lat = lat_index(mesh.node_position(node));
            let adjacent: Vec<(i64, i64)> = [(-1, -1), (-1, 0), (0, -1), (0, 0)]
                .iter()
                .map(|(di, dj)| (lat.0 + di, lat.1 + dj))
                .filter(|c| columns.contains_key(c))
                .collect();
            let total_weight: f64 = adjacent.iter().map(|c| col_scale[c]).sum();
            let shares: ColumnShares = adjacent
                .iter()
                .map(|c| (*c, force * col_scale[c] / total_weight))
                .collect();
            for (c, share) in &shares {
                *col_force.entry(*c).or_default() += share;
            }
            contacts.push((node, shares));
        }
        if col_force.is_empty() {
            return Err(
                "the rigid body makes no contact with the mesh; position it \
                 interpenetrating the face opposite the fixed boundary"
                    .to_string(),
            );
        }

        // The demanded force per contacted column: the map sampled at the
        // column's lateral center (uniform column footprints, so pressure
        // and per-column force have the same shape).
        let mut col_target: HashMap<(i64, i64), f64> = HashMap::new();
        for &c in col_force.keys() {
            let lp = [
                lo[lateral[0]] + (c.0 as f64 + 0.5) * h,
                lo[lateral[1]] + (c.1 as f64 + 0.5) * h,
            ];
            let sample = target.pressure(lp);
            if !sample.is_finite() {
                return Err(format!(
                    "target map returned a non-finite value at ({}, {})",
                    lp[0], lp[1]
                ));
            }
            col_target.insert(c, sample.max(0.0));
        }
        let total_force: f64 = col_force.values().sum();
        let total_target: f64 = col_target.values().sum();
        if total_target <= 0.0 {
            return Err(
                "the target map is zero (or negative) everywhere the rigid body \
                 makes contact, so the desired distribution is undefined; check \
                 that the map covers the contact patch"
                    .to_string(),
            );
        }

        let distribution_error: f64 = 0.5
            * col_force
                .iter()
                .map(|(c, f)| (col_target[c] / total_target - f / total_force).abs())
                .sum::<f64>();

        // Normalized target/achieved ratio per contacted column — the
        // fixed-point residual.
        let ratios: HashMap<(i64, i64), f64> = col_force
            .iter()
            .map(|(c, f)| (*c, (col_target[c] / total_target) / (f / total_force)))
            .collect();

        // Set FEA_INVERSE_DEBUG=1 to watch the outer fixed point: error
        // descending = keep iterating; error frozen with a large floored
        // share = min_scale is binding (lower it).
        if std::env::var("FEA_INVERSE_DEBUG").is_ok() {
            let floored = scales.iter().filter(|s| **s <= config.min_scale).count();
            eprintln!(
                "inverse iter {iterations}: error={distribution_error:.4} floored={floored}/{}",
                scales.len(),
            );
        }

        let converged = distribution_error <= config.tolerance;
        if converged || iterations >= config.max_iterations {
            // Per-node target force, comparable to the achieved contact
            // forces: each node's column shares scaled by their columns'
            // ratios (sums to the achieved total).
            let mut target_force = vec![0.0f64; node_count];
            for (node, shares) in &contacts {
                target_force[*node] = shares.iter().map(|(c, share)| share * ratios[c]).sum();
            }
            return Ok(InverseResult {
                stiffness_scale: scales,
                target_force,
                solve: result,
                iterations,
                distribution_error,
                converged,
            });
        }

        // Fixed-point update, damped by `exponent` and clamped per
        // iteration so one bad linearization can't blow up the scales.
        for (c, ratio) in &ratios {
            let factor = ratio.powf(config.exponent).clamp(0.25, 4.0);
            for &e in &columns[c] {
                scales[e] *= factor;
            }
        }
        // Renormalize so the stiffest element sits at 1.0 — distribution
        // matching is scale-free, and this keeps the SIMP knob in a stable,
        // clampable range.
        let max = scales.iter().copied().fold(0.0f64, f64::max);
        if max > 0.0 {
            for s in &mut scales {
                *s = (*s / max).clamp(config.min_scale, 1.0);
            }
        }
    }
}

/// The strut-lattice inverse. Same fixed point as the hex path, adapted
/// to irregular networks:
///
/// - Columns are lateral bins of width `column_size` (0 = twice the mean
///   strut length); a strut belongs to its midpoint's bin.
/// - A contact node's force splits among the columns of its *incident*
///   struts, in proportion to column stiffness — the strut analog of the
///   hex tributary split (the force a node transmits flows into the
///   struts hanging off it).
/// - The target pressure is sampled at each contact node's lateral
///   position and attributed with the same shares as its force. Summed
///   per column, demand and achievement carry identical footprint
///   weighting, so partially covered boundary bins (or foam's uneven
///   node density) aren't over-demanded, and comparison still happens at
///   column granularity — per-node demands the actuator can't resolve
///   average out inside their bin.
fn solve_inverse_frame(
    mesh: &FeaMesh,
    rigid: &mut dyn RigidBody,
    target: &mut dyn TargetMap,
    config: &InverseConfig,
) -> Result<InverseResult, String> {
    if !(config.column_size >= 0.0 && config.column_size.is_finite()) {
        return Err(format!(
            "column_size {} must be zero (auto) or positive",
            config.column_size
        ));
    }
    if mesh.element_count() == 0 {
        return Err("mesh has no struts".to_string());
    }
    let (contact_axis, contact_sign) = config.solve.fixed_boundary.contact_axis();
    let lateral: [usize; 2] = match contact_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };

    let node_count = mesh.node_count();
    let mut lo = [f64::INFINITY; 3];
    for node in 0..node_count {
        let p = mesh.node_position(node);
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
        }
    }

    let mut total_length = 0.0;
    for e in 0..mesh.element_count() {
        let pair = mesh.element(e);
        let a = mesh.node_position(pair[0] as usize);
        let b = mesh.node_position(pair[1] as usize);
        total_length += (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
    }
    let bin = if config.column_size > 0.0 {
        config.column_size
    } else {
        2.0 * total_length / mesh.element_count() as f64
    };
    if !(bin > 0.0 && bin.is_finite()) {
        return Err(format!("degenerate column bin width {bin}"));
    }

    let col_of = |p: [f64; 3]| -> (i64, i64) {
        (
            ((p[lateral[0]] - lo[lateral[0]]) / bin).floor() as i64,
            ((p[lateral[1]] - lo[lateral[1]]) / bin).floor() as i64,
        )
    };

    // Struts grouped by the lateral column of their midpoint.
    let mut columns: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let mut strut_col: Vec<(i64, i64)> = Vec::with_capacity(mesh.element_count());
    for e in 0..mesh.element_count() {
        let pair = mesh.element(e);
        let a = mesh.node_position(pair[0] as usize);
        let b = mesh.node_position(pair[1] as usize);
        let mid = [
            0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            0.5 * (a[2] + b[2]),
        ];
        let c = col_of(mid);
        columns.entry(c).or_default().push(e);
        strut_col.push(c);
    }

    // Node -> incident struts.
    let mut node_struts: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    for e in 0..mesh.element_count() {
        for &node in mesh.element(e) {
            node_struts[node as usize].push(e);
        }
    }

    let mut work = mesh.clone();
    let mut scales = stiffness_scales(mesh)?;
    let mut iterations = 0;

    loop {
        iterations += 1;
        set_scale_field(&mut work, &scales);
        let result = solve(&work, rigid, &config.solve)?;
        if !result.stats.converged {
            return Err(forward_failure(iterations, &result.stats));
        }

        let col_scale: HashMap<(i64, i64), f64> = columns
            .iter()
            .map(|(c, struts)| {
                let mean = struts.iter().map(|&e| scales[e]).sum::<f64>() / struts.len() as f64;
                (*c, mean)
            })
            .collect();

        // Attribute each contact node's force AND its sampled pressure to
        // the columns of its incident struts with the same shares.
        let mut contacts: Vec<(usize, f64, ColumnShares)> = Vec::new();
        let mut col_force: HashMap<(i64, i64), f64> = HashMap::new();
        let mut col_target: HashMap<(i64, i64), f64> = HashMap::new();
        for node in 0..node_count {
            let force = -contact_sign * result.contact_force[node * 3 + contact_axis];
            if force <= 0.0 {
                continue;
            }
            let mut cols: Vec<(i64, i64)> = node_struts[node]
                .iter()
                .map(|&e| strut_col[e])
                .collect();
            cols.sort_unstable();
            cols.dedup();
            let p = mesh.node_position(node);
            let pressure = target.pressure([p[lateral[0]], p[lateral[1]]]);
            if !pressure.is_finite() {
                return Err(format!(
                    "target map returned a non-finite value at ({}, {})",
                    p[lateral[0]], p[lateral[1]]
                ));
            }
            let pressure = pressure.max(0.0);
            let total_weight: f64 = cols.iter().map(|c| col_scale[c]).sum();
            let fractions: ColumnShares = cols
                .iter()
                .map(|c| (*c, col_scale[c] / total_weight))
                .collect();
            for (c, fraction) in &fractions {
                *col_force.entry(*c).or_default() += force * fraction;
                *col_target.entry(*c).or_default() += pressure * fraction;
            }
            contacts.push((node, force, fractions));
        }
        if col_force.is_empty() {
            return Err(
                "the rigid body makes no contact with the lattice; position it \
                 interpenetrating the face opposite the fixed boundary"
                    .to_string(),
            );
        }

        let total_force: f64 = col_force.values().sum();
        let total_target: f64 = col_target.values().sum();
        if total_target <= 0.0 {
            return Err(
                "the target map is zero (or negative) everywhere the rigid body \
                 makes contact, so the desired distribution is undefined; check \
                 that the map covers the contact patch"
                    .to_string(),
            );
        }

        let distribution_error: f64 = 0.5
            * col_force
                .iter()
                .map(|(c, f)| (col_target[c] / total_target - f / total_force).abs())
                .sum::<f64>();

        let ratios: HashMap<(i64, i64), f64> = col_force
            .iter()
            .map(|(c, f)| (*c, (col_target[c] / total_target) / (f / total_force)))
            .collect();

        // Set FEA_INVERSE_DEBUG=1 to watch the outer fixed point: error
        // descending = keep iterating; error frozen with a large floored
        // share = min_scale is binding (lower it).
        if std::env::var("FEA_INVERSE_DEBUG").is_ok() {
            let floored = scales.iter().filter(|s| **s <= config.min_scale).count();
            eprintln!(
                "inverse iter {iterations}: error={distribution_error:.4} floored={floored}/{}",
                scales.len(),
            );
        }

        let converged = distribution_error <= config.tolerance;
        if converged || iterations >= config.max_iterations {
            let mut target_force = vec![0.0f64; node_count];
            for (node, force, fractions) in &contacts {
                target_force[*node] = fractions
                    .iter()
                    .map(|(c, fraction)| force * fraction * ratios[c])
                    .sum();
            }
            return Ok(InverseResult {
                stiffness_scale: scales,
                target_force,
                solve: result,
                iterations,
                distribution_error,
                converged,
            });
        }

        for (c, ratio) in &ratios {
            let factor = ratio.powf(config.exponent).clamp(0.25, 4.0);
            for &e in &columns[c] {
                scales[e] *= factor;
            }
        }
        let max = scales.iter().copied().fold(0.0f64, f64::max);
        if max > 0.0 {
            for s in &mut scales {
                *s = (*s / max).clamp(config.min_scale, 1.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::grid_mesh;
    use crate::{FixedBoundary, Material};

    fn config(nu: f64, fixed_boundary: FixedBoundary) -> InverseConfig {
        InverseConfig {
            solve: SolveConfig {
                material: Material {
                    youngs_modulus: 1.0,
                    poissons_ratio: nu,
                },
                fixed_boundary,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// A rigid half-space `z > level` (a flat plate pressed straight down).
    fn plate(level: f64) -> impl FnMut([f64; 3]) -> bool {
        move |p: [f64; 3]| p[2] > level
    }

    #[test]
    fn uniform_target_converges_immediately() {
        // A uniform column under a flat plate already produces a uniform
        // pressure: the tributary-area weighting must recognize it as the
        // fixed point on the first solve, leaving the scales untouched.
        let mesh = grid_mesh(4, 4, 2, 0.25);
        let mut target = |_: [f64; 2]| 1.0;
        let result = solve_inverse(
            &mesh,
            &mut plate(0.45),
            &mut target,
            &config(0.0, FixedBoundary::ZMin),
        )
        .unwrap();
        assert!(result.converged, "error {}", result.distribution_error);
        assert_eq!(result.iterations, 1);
        assert!(result.distribution_error < 1e-6);
        assert!(result.stiffness_scale.iter().all(|s| *s == 1.0));
    }

    #[test]
    fn step_target_shapes_the_force_distribution() {
        // 1x1x0.5 slab glued at z=0, flat plate pressed 10% in. Demand
        // three times the pressure on the x > 0.5 half.
        let mesh = grid_mesh(8, 8, 4, 0.125);
        let mut target = |p: [f64; 2]| if p[0] < 0.5 { 1.0 } else { 3.0 };
        let cfg = config(0.3, FixedBoundary::ZMin);
        let result = solve_inverse(&mesh, &mut plate(0.45), &mut target, &cfg).unwrap();
        assert!(
            result.converged,
            "error {} after {} iterations",
            result.distribution_error, result.iterations
        );
        assert!(
            result.iterations > 1,
            "a step target cannot be a fixed point"
        );

        // The achieved distribution matches the demanded 1:3 split.
        let node_count = mesh.node_count();
        let mut split = [0.0f64; 2]; // [x < 0.5, x >= 0.5]
        for node in 0..node_count {
            let f = -result.solve.contact_force[node * 3 + 2];
            if f > 0.0 {
                split[(mesh.node_position(node)[0] >= 0.5) as usize] += f;
            }
        }
        let ratio = split[1] / split[0];
        // Nodes exactly on x = 0.5 sample the stiff side, so the demanded
        // nodal split is a bit above 3; accept the neighborhood.
        assert!(
            (2.5..4.5).contains(&ratio),
            "achieved high/low force ratio {ratio}, expected ~3"
        );

        // The stiffness follows: soft columns on the low-pressure side.
        let mut side_scale = [0.0f64; 2];
        let mut side_count = [0usize; 2];
        for e in 0..mesh.element_count() {
            let cx = mesh.node_position(mesh.element(e)[0] as usize)[0] + 0.0625;
            let side = (cx > 0.5) as usize;
            side_scale[side] += result.stiffness_scale[e];
            side_count[side] += 1;
        }
        let low = side_scale[0] / side_count[0] as f64;
        let high = side_scale[1] / side_count[1] as f64;
        assert!(
            high > 2.0 * low,
            "stiffness contrast missing: low side {low}, high side {high}"
        );
        assert!(
            result
                .stiffness_scale
                .iter()
                .all(|s| (0.01..=1.0).contains(s))
        );

        // target_force mirrors the demanded distribution at the achieved
        // total.
        let total_target: f64 = result.target_force.iter().sum();
        let total_force = split[0] + split[1];
        assert!((total_target - total_force).abs() < 1e-6 * total_force);
    }

    #[test]
    fn contact_axis_follows_the_fixed_boundary() {
        // Glue xmin and press a plate in from +x: the lateral map arguments
        // are (y, z) ascending. Demand the pressure step along z.
        let mesh = grid_mesh(4, 4, 4, 0.25);
        let mut rigid = |p: [f64; 3]| p[0] > 0.9;
        let mut target = |p: [f64; 2]| if p[1] < 0.5 { 3.0 } else { 1.0 };
        let cfg = config(0.0, FixedBoundary::XMin);
        let result = solve_inverse(&mesh, &mut rigid, &mut target, &cfg).unwrap();
        assert!(
            result.converged,
            "error {} after {} iterations",
            result.distribution_error, result.iterations
        );
        let mut side_scale = [0.0f64; 2];
        let mut side_count = [0usize; 2];
        for e in 0..mesh.element_count() {
            let cz = mesh.node_position(mesh.element(e)[0] as usize)[2] + 0.125;
            let side = (cz > 0.5) as usize;
            side_scale[side] += result.stiffness_scale[e];
            side_count[side] += 1;
        }
        let low_z = side_scale[0] / side_count[0] as f64;
        let high_z = side_scale[1] / side_count[1] as f64;
        assert!(
            low_z > 1.5 * high_z,
            "expected stiffer columns at low z: low {low_z}, high {high_z}"
        );
    }

    /// A cubic strut lattice block: (nx+1)(ny+1)(nz+1) grid nodes with
    /// struts along all three axes, uniform radius.
    fn strut_grid(nx: usize, ny: usize, nz: usize, h: f64, radius: f64) -> FeaMesh {
        use volumetric_abi::fea::FeaElementKind;
        let (mx, my) = (nx + 1, ny + 1);
        let node = |i: usize, j: usize, k: usize| (k * my * mx + j * mx + i) as u32;
        let mut node_positions = Vec::new();
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    node_positions.extend([i as f64 * h, j as f64 * h, k as f64 * h]);
                }
            }
        }
        let mut connectivity = Vec::new();
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    if i < nx {
                        connectivity.extend([node(i, j, k), node(i + 1, j, k)]);
                    }
                    if j < ny {
                        connectivity.extend([node(i, j, k), node(i, j + 1, k)]);
                    }
                    if k < nz {
                        connectivity.extend([node(i, j, k), node(i, j, k + 1)]);
                    }
                }
            }
        }
        let strut_count = connectivity.len() / 2;
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions,
            connectivity,
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![radius; strut_count],
            }],
        }
    }

    #[test]
    fn frame_uniform_target_is_a_fixed_point() {
        // A uniform cubic lattice under a flat plate: every top node has
        // an identical vertical strut chain below it, so the contact
        // forces are already uniform. Per-node target attribution must
        // recognize the fixed point on the first solve — including in
        // partially covered boundary bins.
        let mesh = strut_grid(4, 4, 2, 0.25, 0.02);
        let mut target = |_: [f64; 2]| 1.0;
        let result = solve_inverse(
            &mesh,
            &mut plate(0.45),
            &mut target,
            &config(0.3, FixedBoundary::ZMin),
        )
        .unwrap();
        assert!(result.converged, "error {}", result.distribution_error);
        assert_eq!(result.iterations, 1);
        assert!(result.distribution_error < 1e-6);
        assert!(result.stiffness_scale.iter().all(|s| *s == 1.0));
    }

    #[test]
    fn frame_step_target_shapes_forces() {
        // 1x1x0.5 cubic lattice glued at z=0, plate pressed 10% in, three
        // times the pressure demanded on the x > 0.5 half.
        let mesh = strut_grid(8, 8, 4, 0.125, 0.015);
        let mut target = |p: [f64; 2]| if p[0] < 0.5 { 1.0 } else { 3.0 };
        let cfg = config(0.3, FixedBoundary::ZMin);
        let result = solve_inverse(&mesh, &mut plate(0.45), &mut target, &cfg).unwrap();
        assert!(
            result.converged,
            "error {} after {} iterations",
            result.distribution_error, result.iterations
        );
        assert!(
            result.iterations > 1,
            "a step target cannot be a fixed point"
        );

        // The achieved nodal force split matches the demanded 1:3 shape
        // (nodes exactly on x = 0.5 sample the high side, nudging the
        // ratio above 3).
        let mut split = [0.0f64; 2];
        for node in 0..mesh.node_count() {
            let f = -result.solve.contact_force[node * 3 + 2];
            if f > 0.0 {
                split[(mesh.node_position(node)[0] >= 0.5) as usize] += f;
            }
        }
        let ratio = split[1] / split[0];
        assert!(
            (2.3..4.8).contains(&ratio),
            "achieved high/low force ratio {ratio}, expected ~3"
        );

        // Stiffness follows: struts under the low-pressure side go soft.
        let mut side_scale = [0.0f64; 2];
        let mut side_count = [0usize; 2];
        for e in 0..mesh.element_count() {
            let pair = mesh.element(e);
            let a = mesh.node_position(pair[0] as usize);
            let b = mesh.node_position(pair[1] as usize);
            let side = (0.5 * (a[0] + b[0]) > 0.5) as usize;
            side_scale[side] += result.stiffness_scale[e];
            side_count[side] += 1;
        }
        let low = side_scale[0] / side_count[0] as f64;
        let high = side_scale[1] / side_count[1] as f64;
        assert!(
            high > 1.5 * low,
            "stiffness contrast missing: low side {low}, high side {high}"
        );

        // target_force is scaled to the achieved total.
        let total_target: f64 = result.target_force.iter().sum();
        let total_force = split[0] + split[1];
        assert!((total_target - total_force).abs() < 1e-6 * total_force);
    }

    #[test]
    fn zero_target_over_the_contact_is_an_error() {
        let mesh = grid_mesh(2, 2, 2, 0.5);
        let mut target = |_: [f64; 2]| 0.0;
        let err = solve_inverse(
            &mesh,
            &mut plate(0.9),
            &mut target,
            &config(0.0, FixedBoundary::ZMin),
        )
        .unwrap_err();
        assert!(
            err.contains("target map is zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn no_contact_is_an_error() {
        let mesh = grid_mesh(2, 2, 2, 0.5);
        let mut target = |_: [f64; 2]| 1.0;
        let err = solve_inverse(
            &mesh,
            &mut plate(100.0),
            &mut target,
            &config(0.0, FixedBoundary::ZMin),
        )
        .unwrap_err();
        assert!(err.contains("no contact"), "unexpected error: {err}");
    }
}
