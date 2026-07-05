//! Linear-elastic FEA solve on uniform hex8 grids, with active-set contact
//! against a rigid implicit body.
//!
//! Scope (deliberately v1, matching the printed-cushion pipeline):
//! - The mesh must be an axis-aligned uniform hex grid (what
//!   `fea_grid_mesh_operator` emits): one shared element stiffness matrix,
//!   optionally scaled per element by a `stiffness_scale` element field —
//!   the SIMP-style knob the inverse-design loop will drive.
//! - Small-displacement Hooke's law (isotropic E, nu). Quantitatively wrong
//!   for foam at seat strains; fine for relative density assignment.
//! - Quasi-static single pose. The rigid body is sampled where the user
//!   placed it (already interpenetrating the mesh = the pressed pose).
//! - Contact presses toward the glued face: the contact axis is the
//!   `fixed_boundary` axis, with the rigid body approaching from the
//!   opposite side (glued zmin → body presses down from +z; glued ymin →
//!   body presses from +y; no fixed face → down from +z). A penetrating
//!   node's contact-axis displacement is prescribed to the body's near
//!   surface on its line (active-set Dirichlet; nodes with tensile
//!   reactions are released). The constraint reactions are the interface
//!   force map.
//!
//! The linear solves are matrix-free Jacobi-preconditioned conjugate
//! gradient over the free dofs.

pub mod element;

pub use element::{ElementStiffness, Material, cube_stiffness, hex8_stiffness};
use volumetric_abi::fea::{FeaElementKind, FeaMesh};

/// An implicit rigid body, sampled by occupancy (the Model ABI contract).
pub trait RigidBody {
    fn is_inside(&mut self, p: [f64; 3]) -> bool;
}

impl<F: FnMut([f64; 3]) -> bool> RigidBody for F {
    fn is_inside(&mut self, p: [f64; 3]) -> bool {
        self(p)
    }
}

/// Which face of the mesh's bounding box is glued (all dofs zero).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedBoundary {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
    None,
}

impl FixedBoundary {
    /// Parse the operator-config spelling (`"zmin"`, `"none"`, ...).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "xmin" => Ok(Self::XMin),
            "xmax" => Ok(Self::XMax),
            "ymin" => Ok(Self::YMin),
            "ymax" => Ok(Self::YMax),
            "zmin" => Ok(Self::ZMin),
            "zmax" => Ok(Self::ZMax),
            "none" => Ok(Self::None),
            other => Err(format!(
                "unknown fixed boundary {other:?} (expected xmin/xmax/ymin/ymax/zmin/zmax/none)"
            )),
        }
    }

    /// (axis, take-minimum) of the fixed face, if any.
    fn axis(self) -> Option<(usize, bool)> {
        match self {
            Self::XMin => Some((0, true)),
            Self::XMax => Some((0, false)),
            Self::YMin => Some((1, true)),
            Self::YMax => Some((1, false)),
            Self::ZMin => Some((2, true)),
            Self::ZMax => Some((2, false)),
            Self::None => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SolveConfig {
    pub material: Material,
    pub fixed_boundary: FixedBoundary,
    /// Relative residual tolerance for each CG solve.
    pub cg_tolerance: f64,
    pub cg_max_iterations: usize,
    pub max_contact_iterations: usize,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            material: Material {
                youngs_modulus: 1.0,
                poissons_ratio: 0.3,
            },
            fixed_boundary: FixedBoundary::ZMin,
            cg_tolerance: 1e-8,
            cg_max_iterations: 20_000,
            max_contact_iterations: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    /// CG iterations summed over all contact iterations.
    pub cg_iterations: usize,
    pub contact_iterations: usize,
    pub active_contacts: usize,
    /// False when CG or the contact active set hit an iteration cap.
    pub converged: bool,
}

#[derive(Debug)]
pub struct SolveResult {
    /// Per-node displacement, xyz interleaved.
    pub displacement: Vec<f64>,
    /// Per-node force applied by the rigid body (nonzero only on the active
    /// contact set), xyz interleaved. Along the contact axis the sign
    /// presses toward the glued face.
    pub contact_force: Vec<f64>,
    /// Per-element strain energy per unit volume.
    pub strain_energy_density: Vec<f64>,
    pub stats: SolveStats,
}

/// Grid-cell corner offsets in VTK hex8 node order (matches the mesher).
const CELL_CORNERS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// Verify the mesh is an axis-aligned uniform grid and return its cell size.
fn detect_uniform_grid(mesh: &FeaMesh) -> Result<f64, String> {
    if mesh.element_count() == 0 {
        return Err("mesh has no elements".to_string());
    }
    let first = mesh.element(0);
    let origin = mesh.node_position(first[0] as usize);
    let h = mesh.node_position(first[1] as usize)[0] - origin[0];
    if !(h > 0.0 && h.is_finite()) {
        return Err(format!("degenerate grid cell size {h}"));
    }
    let tol = h * 1e-6;
    for e in 0..mesh.element_count() {
        let element = mesh.element(e);
        let base = mesh.node_position(element[0] as usize);
        for (corner, offset) in element.iter().zip(&CELL_CORNERS) {
            let p = mesh.node_position(*corner as usize);
            for axis in 0..3 {
                if (p[axis] - (base[axis] + offset[axis] * h)).abs() > tol {
                    return Err(format!(
                        "element {e} is not an axis-aligned {h}-cube; the solver \
                         requires the uniform grid produced by fea_grid_mesh_operator"
                    ));
                }
            }
        }
    }
    Ok(h)
}

/// Per-element stiffness multipliers from an optional `stiffness_scale`
/// element field (missing field = all 1.0).
fn stiffness_scales(mesh: &FeaMesh) -> Result<Vec<f64>, String> {
    let Some(field) = mesh
        .element_fields
        .iter()
        .find(|f| f.name == "stiffness_scale")
    else {
        return Ok(vec![1.0; mesh.element_count()]);
    };
    if field.components != 1 {
        return Err(format!(
            "stiffness_scale must be a scalar field, has {} components",
            field.components
        ));
    }
    if let Some(bad) = field.data.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
        return Err(format!("stiffness_scale contains invalid value {bad}"));
    }
    Ok(field.data.clone())
}

/// y = K x (matrix-free over elements), then constrained components zeroed.
struct Operator<'a> {
    mesh: &'a FeaMesh,
    ke: &'a ElementStiffness,
    scales: &'a [f64],
    constrained: &'a [bool],
}

impl Operator<'_> {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        for e in 0..self.mesh.element_count() {
            let scale = self.scales[e];
            if scale == 0.0 {
                continue;
            }
            let nodes = self.mesh.element(e);
            let mut xe = [0.0f64; 24];
            for (i, node) in nodes.iter().enumerate() {
                let base = *node as usize * 3;
                xe[i * 3] = x[base];
                xe[i * 3 + 1] = x[base + 1];
                xe[i * 3 + 2] = x[base + 2];
            }
            for (i, node) in nodes.iter().enumerate() {
                let base = *node as usize * 3;
                for c in 0..3 {
                    let row = &self.ke[i * 3 + c];
                    let mut sum = 0.0;
                    for (j, xj) in xe.iter().enumerate() {
                        sum += row[j] * xj;
                    }
                    y[base + c] += scale * sum;
                }
            }
        }
        for (v, constrained) in y.iter_mut().zip(self.constrained) {
            if *constrained {
                *v = 0.0;
            }
        }
    }

    /// K u without masking (for constraint reactions).
    fn apply_unmasked(&self, x: &[f64], y: &mut [f64]) {
        let unmasked = Operator {
            constrained: &[],
            ..*self
        };
        // An empty mask slice means "nothing constrained": zip stops at the
        // shorter side.
        unmasked.apply(x, y);
    }
}

/// Jacobi-preconditioned CG for `K u = 0` with Dirichlet values already
/// written into `u`. Returns (iterations, converged).
fn solve_cg(
    op: &Operator<'_>,
    diag: &[f64],
    u: &mut [f64],
    tolerance: f64,
    max_iterations: usize,
) -> (usize, bool) {
    let n = u.len();
    let mut r = vec![0.0; n];
    // r = -K u on free dofs (external forces are zero; Dirichlet drives).
    op.apply_unmasked(u, &mut r);
    for i in 0..n {
        r[i] = if op.constrained[i] { 0.0 } else { -r[i] };
    }

    let norm0: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm0 == 0.0 {
        return (0, true);
    }
    let target = norm0 * tolerance;

    let precond = |r: &[f64], z: &mut [f64]| {
        for i in 0..r.len() {
            z[i] = if diag[i] > 0.0 { r[i] / diag[i] } else { r[i] };
        }
    };

    let mut z = vec![0.0; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut kp = vec![0.0; n];
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();

    for iteration in 0..max_iterations {
        op.apply(&p, &mut kp);
        let pkp: f64 = p.iter().zip(&kp).map(|(a, b)| a * b).sum();
        if pkp <= 0.0 {
            // Singular or indefinite (e.g. unconstrained mesh): bail out
            // with whatever we have rather than dividing by zero.
            return (iteration, false);
        }
        let alpha = rz / pkp;
        for i in 0..n {
            u[i] += alpha * p[i];
            r[i] -= alpha * kp[i];
        }
        let norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= target {
            return (iteration + 1, true);
        }
        precond(&r, &mut z);
        let rz_next: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_next / rz;
        rz = rz_next;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    (max_iterations, false)
}

/// Find the rigid body's near surface on the line through `inside` along
/// `axis`: the boundary crossing reached by walking from the (inside)
/// starting point *away* from the body's side (`sign` = +1 when the body
/// presses from the axis' positive side). Scans in `h`-steps to bracket the
/// boundary, then bisects. Returns the surface's coordinate along `axis`.
fn rigid_contact_surface(
    rigid: &mut dyn RigidBody,
    inside: [f64; 3],
    axis: usize,
    sign: f64,
    h: f64,
    scan_limit: usize,
) -> Result<f64, String> {
    let mut hi = inside[axis]; // inside the body
    let mut lo = None; // outside, past the surface
    for step in 1..=scan_limit {
        let mut q = inside;
        q[axis] = inside[axis] - sign * step as f64 * h;
        if rigid.is_inside(q) {
            hi = q[axis];
        } else {
            lo = Some(q[axis]);
            break;
        }
    }
    let Some(mut lo) = lo else {
        return Err(format!(
            "rigid body spans the whole mesh along axis {axis} at \
             ({:.4}, {:.4}, {:.4}); it must press against one face, not engulf \
             the mesh",
            inside[0], inside[1], inside[2]
        ));
    };
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let mut q = inside;
        q[axis] = mid;
        if rigid.is_inside(q) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Solve the compression problem. See the module docs for scope.
pub fn solve(
    mesh: &FeaMesh,
    rigid: &mut dyn RigidBody,
    config: &SolveConfig,
) -> Result<SolveResult, String> {
    if mesh.element_kind != FeaElementKind::Hex8 {
        return Err(format!("unsupported element kind {:?}", mesh.element_kind));
    }
    mesh.validate()?;
    let nu = config.material.poissons_ratio;
    if !(-1.0 < nu && nu < 0.5) {
        return Err(format!("Poisson's ratio {nu} outside (-1, 0.5)"));
    }

    let node_count = mesh.node_count();
    let n = node_count * 3;
    if mesh.element_count() == 0 {
        return Ok(SolveResult {
            displacement: vec![0.0; n],
            contact_force: vec![0.0; n],
            strain_energy_density: Vec::new(),
            stats: SolveStats {
                converged: true,
                ..Default::default()
            },
        });
    }

    let h = detect_uniform_grid(mesh)?;
    let scales = stiffness_scales(mesh)?;
    let ke = cube_stiffness(h, config.material)?;

    // Mesh bounding box (for the fixed face and the contact scan limit).
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for node in 0..node_count {
        let p = mesh.node_position(node);
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    // The rigid body presses toward the glued face: contact acts along the
    // fixed boundary's axis, with the body approaching from the opposite
    // side (glued at min → body on the positive side pressing negative).
    // With no fixed face it presses down the z axis, from above.
    let (contact_axis, contact_sign) = match config.fixed_boundary.axis() {
        Some((axis, glued_min)) => (axis, if glued_min { 1.0 } else { -1.0 }),
        None => (2, 1.0),
    };
    let scan_limit = ((hi[contact_axis] - lo[contact_axis]) / h).ceil() as usize + 4;

    // Base constraints: the glued face.
    let mut constrained = vec![false; n];
    let mut prescribed = vec![0.0f64; n];
    let mut fixed_node = vec![false; node_count];
    if let Some((axis, take_min)) = config.fixed_boundary.axis() {
        let face = if take_min { lo[axis] } else { hi[axis] };
        for node in 0..node_count {
            if (mesh.node_position(node)[axis] - face).abs() < h * 1e-3 {
                fixed_node[node] = true;
                for c in 0..3 {
                    constrained[node * 3 + c] = true;
                }
            }
        }
    }

    // Jacobi diagonal (constraint state doesn't affect it; masked entries
    // are never used).
    let mut diag = vec![0.0f64; n];
    for e in 0..mesh.element_count() {
        let nodes = mesh.element(e);
        for (i, node) in nodes.iter().enumerate() {
            for c in 0..3 {
                diag[*node as usize * 3 + c] += scales[e] * ke[i * 3 + c][i * 3 + c];
            }
        }
    }

    let mut u = vec![0.0f64; n];
    let mut active: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut stats = SolveStats::default();
    let mut forces = vec![0.0f64; n];

    for _ in 0..config.max_contact_iterations {
        stats.contact_iterations += 1;

        // Activate/refresh: any node whose deformed position penetrates the
        // rigid body gets its contact-axis displacement prescribed so it
        // sits on the body's near surface *at the node's deformed transverse
        // position* — transverse (Poisson) slide moves nodes to lines where
        // the surface height differs, so an already-active constraint is
        // re-prescribed when it has drifted. Penetration (and drift) within
        // a small slack is tolerated; without both, rim nodes cycle
        // activate/release forever and the active set never settles.
        let slack = h * 1e-3;
        let mut set_changes = 0usize;
        for node in 0..node_count {
            if fixed_node[node] {
                continue;
            }
            let p = mesh.node_position(node);
            let deformed = [
                p[0] + u[node * 3],
                p[1] + u[node * 3 + 1],
                p[2] + u[node * 3 + 2],
            ];
            let mut probe = deformed;
            probe[contact_axis] -= contact_sign * slack;
            if !rigid.is_inside(probe) {
                continue;
            }
            let surface =
                rigid_contact_surface(rigid, deformed, contact_axis, contact_sign, h, scan_limit)?;
            let prescribed_u = surface - p[contact_axis];
            match active.get(&node) {
                Some(current) if (current - prescribed_u).abs() <= slack => {}
                _ => {
                    active.insert(node, prescribed_u);
                    set_changes += 1;
                }
            }
        }

        // Refresh the constraint arrays and warm-started solution.
        for node in 0..node_count {
            let dof = node * 3 + contact_axis;
            if fixed_node[node] {
                continue;
            }
            match active.get(&node) {
                Some(value) => {
                    constrained[dof] = true;
                    prescribed[dof] = *value;
                }
                None => {
                    constrained[dof] = false;
                    prescribed[dof] = 0.0;
                }
            }
        }
        for i in 0..n {
            if constrained[i] {
                u[i] = prescribed[i];
            }
        }

        let op = Operator {
            mesh,
            ke: &ke,
            scales: &scales,
            constrained: &constrained,
        };
        let (iterations, cg_converged) = solve_cg(
            &op,
            &diag,
            &mut u,
            config.cg_tolerance,
            config.cg_max_iterations,
        );
        stats.cg_iterations += iterations;
        if !cg_converged {
            stats.converged = false;
            stats.active_contacts = active.len();
            break;
        }

        // Reactions: external force = K u at constrained dofs. A contact
        // constraint may only press toward the glued face (f·sign <= 0);
        // meaningfully tensile ones release. The threshold is relative to
        // the peak compression so noise-level forces at grazing rim nodes
        // don't cycle the set.
        op.apply_unmasked(&u, &mut forces);
        let peak_compression = active
            .keys()
            .map(|node| -contact_sign * forces[node * 3 + contact_axis])
            .fold(0.0f64, f64::max);
        let release_tol = peak_compression * 1e-3;
        let released: Vec<usize> = active
            .keys()
            .copied()
            .filter(|node| contact_sign * forces[node * 3 + contact_axis] > release_tol)
            .collect();
        for node in &released {
            active.remove(node);
        }

        if set_changes == 0 && released.is_empty() {
            stats.converged = true;
            break;
        }
    }
    stats.active_contacts = active.len();

    // Contact force field: the reaction at each active node.
    op_final_forces(mesh, &ke, &scales, &u, &mut forces);
    let mut contact_force = vec![0.0f64; n];
    for node in active.keys() {
        contact_force[node * 3 + contact_axis] = forces[node * 3 + contact_axis];
    }

    // Strain energy density per element: (1/2) u_e^T K_e u_e / h^3.
    let volume = h * h * h;
    let mut strain_energy_density = Vec::with_capacity(mesh.element_count());
    for e in 0..mesh.element_count() {
        let nodes = mesh.element(e);
        let mut ue = [0.0f64; 24];
        for (i, node) in nodes.iter().enumerate() {
            let base = *node as usize * 3;
            ue[i * 3] = u[base];
            ue[i * 3 + 1] = u[base + 1];
            ue[i * 3 + 2] = u[base + 2];
        }
        let mut energy = 0.0;
        for (i, uei) in ue.iter().enumerate() {
            let row = &ke[i];
            let mut sum = 0.0;
            for (j, uej) in ue.iter().enumerate() {
                sum += row[j] * uej;
            }
            energy += uei * sum;
        }
        strain_energy_density.push(0.5 * scales[e] * energy / volume);
    }

    Ok(SolveResult {
        displacement: u,
        contact_force,
        strain_energy_density,
        stats,
    })
}

/// K u with no masking, via a temporary operator (helper for the final
/// reaction extraction, where the loop's operator has gone out of scope).
fn op_final_forces(
    mesh: &FeaMesh,
    ke: &ElementStiffness,
    scales: &[f64],
    u: &[f64],
    out: &mut [f64],
) {
    let op = Operator {
        mesh,
        ke,
        scales,
        constrained: &[],
    };
    op.apply(u, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::fea::FeaField;

    /// Build the uniform grid mesh the mesher would emit: nx*ny*nz cells of
    /// size h with the origin at (0, 0, 0).
    fn grid_mesh(nx: usize, ny: usize, nz: usize, h: f64) -> FeaMesh {
        let (mx, my) = (nx + 1, ny + 1);
        let mut node_positions = Vec::new();
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    node_positions.extend([i as f64 * h, j as f64 * h, k as f64 * h]);
                }
            }
        }
        let node = |i: usize, j: usize, k: usize| (k * my * mx + j * mx + i) as u32;
        let mut connectivity = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    connectivity.extend([
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    ]);
                }
            }
        }
        FeaMesh {
            element_kind: FeaElementKind::Hex8,
            node_positions,
            connectivity,
            node_fields: vec![],
            element_fields: vec![],
        }
    }

    fn config(nu: f64) -> SolveConfig {
        SolveConfig {
            material: Material {
                youngs_modulus: 1.0,
                poissons_ratio: nu,
            },
            ..Default::default()
        }
    }

    /// A rigid half-space `z > level` (a flat plate pressed straight down).
    fn plate(level: f64) -> impl FnMut([f64; 3]) -> bool {
        move |p: [f64; 3]| p[2] > level
    }

    #[test]
    fn no_contact_solves_to_rest() {
        let mesh = grid_mesh(2, 2, 2, 0.5);
        let result = solve(&mesh, &mut plate(100.0), &config(0.3)).unwrap();
        assert!(result.stats.converged);
        assert_eq!(result.stats.active_contacts, 0);
        assert!(result.displacement.iter().all(|v| v.abs() < 1e-12));
        assert!(result.strain_energy_density.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn column_compression_matches_uniaxial_strain() {
        // 2x2x6 cells of h=0.5: a 1x1x3 column, glued at z=0, pressed by a
        // flat plate to z=2.7 (10% compression). With nu=0 this is exact
        // uniaxial strain: u_z linear in z, no lateral motion, total plate
        // force = E * strain * area.
        let (h, height, level) = (0.5, 3.0, 2.7);
        let mesh = grid_mesh(2, 2, 6, h);
        let result = solve(&mesh, &mut plate(level), &config(0.0)).unwrap();
        assert!(result.stats.converged, "stats: {:?}", result.stats);
        // Only the 3x3 nodes of the top face touch the plate.
        assert_eq!(result.stats.active_contacts, 9);

        let strain = (height - level) / height; // 0.1
        for node in 0..mesh.node_count() {
            let p = mesh.node_position(node);
            let expected_uz = -strain * p[2];
            assert!(
                (result.displacement[node * 3 + 2] - expected_uz).abs() < 1e-6,
                "node {node} at z={} has u_z {}, expected {expected_uz}",
                p[2],
                result.displacement[node * 3 + 2]
            );
            for c in 0..2 {
                assert!(
                    result.displacement[node * 3 + c].abs() < 1e-6,
                    "node {node} moved laterally"
                );
            }
        }

        // Interface force map: everything presses down, summing to E*eps*A.
        let total_fz: f64 = (0..mesh.node_count())
            .map(|node| result.contact_force[node * 3 + 2])
            .sum();
        let expected = -strain; // -E * eps * area, with E = area = 1
        assert!(
            (total_fz - expected).abs() < 1e-6,
            "total contact force {total_fz}, expected {expected}"
        );
        assert!(
            (0..mesh.node_count()).all(|n| result.contact_force[n * 3 + 2] <= 1e-9),
            "contact forces must press down"
        );

        // Uniform strain energy density: E*eps^2/2 everywhere.
        let expected_density = 0.5 * strain * strain;
        for (e, density) in result.strain_energy_density.iter().enumerate() {
            assert!(
                (density - expected_density).abs() < 1e-6,
                "element {e} energy density {density}, expected {expected_density}"
            );
        }
    }

    #[test]
    fn poisson_bulge_appears_and_is_symmetric() {
        let mesh = grid_mesh(2, 2, 6, 0.5);
        let result = solve(&mesh, &mut plate(2.7), &config(0.3)).unwrap();
        assert!(result.stats.converged);

        // The free sides bulge outward at mid-height: outward x at x=1,
        // mirrored at x=0.
        let mid = |x: f64, y: f64, z: f64| {
            (0..mesh.node_count())
                .find(|n| {
                    let p = mesh.node_position(*n);
                    (p[0] - x).abs() < 1e-9 && (p[1] - y).abs() < 1e-9 && (p[2] - z).abs() < 1e-9
                })
                .unwrap()
        };
        let right = mid(1.0, 0.5, 1.5);
        let left = mid(0.0, 0.5, 1.5);
        let ux_right = result.displacement[right * 3];
        let ux_left = result.displacement[left * 3];
        assert!(ux_right > 1e-4, "no outward bulge: {ux_right}");
        assert!(
            (ux_right + ux_left).abs() < 1e-6,
            "bulge asymmetric: {ux_left} vs {ux_right}"
        );
    }

    #[test]
    fn soft_elements_carry_more_strain() {
        // Bottom half at 10% stiffness: it should compress much more than
        // the stiff top half (series springs).
        let mut mesh = grid_mesh(1, 1, 4, 0.5);
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![0.1, 0.1, 1.0, 1.0],
        });
        let result = solve(&mesh, &mut plate(1.8), &config(0.0)).unwrap();
        assert!(result.stats.converged);

        // With nu=0 series springs: strain ratio inverse to stiffness.
        let bottom = result.strain_energy_density[0] / 0.1; // (eps^2)/2 each
        let top = result.strain_energy_density[3] / 1.0;
        let ratio = (bottom / top).sqrt(); // eps_bottom / eps_top
        assert!(
            (ratio - 10.0).abs() < 0.1,
            "strain ratio {ratio}, expected ~10"
        );
    }

    #[test]
    fn sphere_indentation_converges_without_chatter() {
        // Curved indenter: grazing rim nodes carry near-zero force, the
        // regression case for active-set chatter. 1x1x0.5 slab, unit sphere
        // dipping 0.05 into the top center.
        let mesh = grid_mesh(8, 8, 4, 0.125);
        let center = [0.5, 0.5, 0.5 - 0.05 + 1.0];
        let mut sphere =
            move |p: [f64; 3]| (0..3).map(|i| (p[i] - center[i]).powi(2)).sum::<f64>() < 1.0;
        let result = solve(&mesh, &mut sphere, &config(0.3)).unwrap();
        assert!(result.stats.converged, "stats: {:?}", result.stats);
        assert!(result.stats.active_contacts > 0);
        assert!(
            result.stats.contact_iterations < 10,
            "slow active-set convergence: {:?}",
            result.stats
        );

        // Deepest node (top center) is pressed down by the dip depth.
        let top_center = (0..mesh.node_count())
            .find(|n| {
                let p = mesh.node_position(*n);
                (p[0] - 0.5).abs() < 1e-9 && (p[1] - 0.5).abs() < 1e-9 && (p[2] - 0.5).abs() < 1e-9
            })
            .unwrap();
        let uz = result.displacement[top_center * 3 + 2];
        assert!(
            (uz + 0.05).abs() < 1e-4,
            "top-center u_z {uz}, expected -0.05"
        );

        let total_fz: f64 = (0..mesh.node_count())
            .map(|n| result.contact_force[n * 3 + 2])
            .sum();
        assert!(total_fz < 0.0, "net force must press down: {total_fz}");
    }

    #[test]
    fn deep_indentation_with_lateral_slide_converges() {
        // Full 8^3 unit box, unit sphere at (0.5, 0.5, 1.9) — dip 0.1, deep
        // enough that rim nodes slide sideways under the Poisson bulge into
        // columns with a different surface height (the historical active-set
        // livelock case).
        let mesh = grid_mesh(8, 8, 8, 0.125);
        let center = [0.5, 0.5, 1.9];
        let mut sphere =
            move |p: [f64; 3]| (0..3).map(|i| (p[i] - center[i]).powi(2)).sum::<f64>() < 1.0;
        let result = solve(&mesh, &mut sphere, &config(0.3)).unwrap();
        assert!(result.stats.converged, "stats: {:?}", result.stats);
        // Rim nodes slide laterally under the Poisson bulge; the refreshed
        // deformed-column prescription must still settle quickly.
        assert!(
            result.stats.contact_iterations < 10,
            "slow convergence: {:?}",
            result.stats
        );
    }

    #[test]
    fn contact_axis_follows_the_fixed_boundary() {
        // The fea_test.vproj scenario: default 2x2x2 box at the origin,
        // sphere translated along +y to (0, 1.9, 0), glued at ymin. Contact
        // must press along -y (it used to be hardcoded to -z, which turned
        // this into garbage constraints and a livelock).
        let mut mesh = grid_mesh(8, 8, 8, 0.25);
        for v in mesh.node_positions.iter_mut() {
            *v -= 1.0; // shift to [-1, 1]^3
        }
        let center = [0.0, 1.9, 0.0];
        let mut sphere =
            move |p: [f64; 3]| (0..3).map(|i| (p[i] - center[i]).powi(2)).sum::<f64>() < 1.0;
        let solve_config = SolveConfig {
            fixed_boundary: FixedBoundary::YMin,
            ..config(0.3)
        };
        let result = solve(&mesh, &mut sphere, &solve_config).unwrap();
        assert!(result.stats.converged, "stats: {:?}", result.stats);
        assert!(result.stats.active_contacts > 0);

        // The face-center node (0, 1, 0) is pressed 0.1 along -y.
        let face_center = (0..mesh.node_count())
            .find(|node| {
                let p = mesh.node_position(*node);
                p[0].abs() < 1e-9 && (p[1] - 1.0).abs() < 1e-9 && p[2].abs() < 1e-9
            })
            .unwrap();
        let uy = result.displacement[face_center * 3 + 1];
        assert!(
            (uy + 0.1).abs() < 1e-4,
            "face-center u_y {uy}, expected -0.1"
        );

        // Forces act along y (toward the glued face), none along z.
        let total_fy: f64 = (0..mesh.node_count())
            .map(|node| result.contact_force[node * 3 + 1])
            .sum();
        assert!(
            total_fy < 0.0,
            "net force must press toward ymin: {total_fy}"
        );
        assert!(
            (0..mesh.node_count()).all(|node| result.contact_force[node * 3 + 2] == 0.0),
            "no z contact forces expected"
        );
    }

    #[test]
    fn non_uniform_meshes_are_rejected() {
        let mut mesh = grid_mesh(2, 2, 2, 0.5);
        mesh.node_positions[5] += 0.05; // perturb one node
        let err = solve(&mesh, &mut plate(100.0), &config(0.3)).unwrap_err();
        assert!(err.contains("axis-aligned"), "unexpected error: {err}");
    }

    #[test]
    fn engulfing_rigid_body_is_an_error() {
        let mesh = grid_mesh(2, 2, 2, 0.5);
        let err = solve(&mesh, &mut |_p: [f64; 3]| true, &config(0.3)).unwrap_err();
        assert!(err.contains("engulf"), "unexpected error: {err}");
    }
}
