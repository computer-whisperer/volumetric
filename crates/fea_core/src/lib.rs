//! Linear-elastic FEA solve with active-set contact against a rigid
//! implicit body, over two element formulations:
//!
//! - **Hex8 grids** (what `fea_grid_mesh_operator` emits): the mesh must be
//!   an axis-aligned uniform hex grid — one shared element stiffness
//!   matrix, optionally scaled per element by a `stiffness_scale` element
//!   field, the SIMP-style knob the inverse-design loop drives.
//! - **Bar2 strut lattices**: each element is a 3D Euler-Bernoulli frame
//!   member (circular section, radius from the mesh's `radius` element
//!   field, 6 dofs per node), so the solve sees the actual strut network —
//!   including the bending-dominated response a solid-element model can't
//!   represent. `stiffness_scale` multiplies a strut's Young's modulus.
//!
//! Shared scope (deliberately v1, matching the printed-cushion pipeline):
//! - Small-displacement Hooke's law (isotropic E, nu). Quantitatively wrong
//!   for foam at seat strains; fine for relative property assignment.
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
pub mod frame;
pub mod inverse;
#[cfg(feature = "parallel")]
pub mod schwarz;

pub use element::{ElementStiffness, Material, cube_stiffness, hex8_stiffness};
pub use inverse::{InverseConfig, InverseResult, TargetMap, solve_inverse};
#[cfg(feature = "parallel")]
pub use schwarz::SchwarzParams;
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

    /// The contact (axis, sign) implied by this fixed boundary: the rigid
    /// body presses along the fixed face's axis, approaching from the
    /// opposite side (`sign` = +1 when it presses from the axis' positive
    /// side). With no fixed face it presses down the z axis, from above.
    pub fn contact_axis(self) -> (usize, f64) {
        match self.axis() {
            Some((axis, glued_min)) => (axis, if glued_min { 1.0 } else { -1.0 }),
            None => (2, 1.0),
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

/// Which CG preconditioner the solve uses.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum PrecondChoice {
    /// Block-Jacobi when the model assembles node blocks, scalar otherwise.
    #[default]
    Auto,
    /// Two-level additive Schwarz (Bar2 frames only; falls back to Auto for
    /// other element kinds). The scale/parallelism prototype — see the
    /// `schwarz` module docs.
    #[cfg(feature = "parallel")]
    Schwarz(SchwarzParams),
}

#[derive(Clone, Copy, Debug)]
pub struct SolveConfig {
    pub material: Material,
    pub fixed_boundary: FixedBoundary,
    /// Relative residual tolerance for each CG solve.
    pub cg_tolerance: f64,
    pub cg_max_iterations: usize,
    pub max_contact_iterations: usize,
    pub preconditioner: PrecondChoice,
    /// Stress-stiffening (Picard) passes for Bar2 frames: after each full
    /// contact solve, the struts' tensile axial forces feed a geometric
    /// stiffness (hammock/membrane action — a taut surface strut carries
    /// transverse load like a string) and the contact problem is re-solved
    /// against the stiffened tangent. 0 = linear (the historical behavior);
    /// 1-2 passes capture most of the effect. Tension-only: compressive
    /// softening would make K indefinite. Ignored for Hex8 meshes.
    pub stress_stiffening_passes: usize,
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
            // Settled sets exit early, so the cap only costs time on runs
            // that would fail anyway. Fine foam lattices with a grazing
            // contact rim have needed ~21 sweeps; 16 was too tight.
            max_contact_iterations: 64,
            preconditioner: PrecondChoice::Auto,
            stress_stiffening_passes: 0,
        }
    }
}

/// Why a solve reported `converged == false`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveFailure {
    /// A CG solve hit `cg_max_iterations` without reaching tolerance (or
    /// found the system singular/indefinite, e.g. an unconstrained mesh).
    CgStalled,
    /// The contact active set was still changing when
    /// `max_contact_iterations` ran out. Distinct from a solver problem:
    /// every CG solve converged, the activate/release fixed point just
    /// needs more sweeps (grazing rims on curved bodies are the usual
    /// driver) — raising the cap is the remedy.
    ContactUnsettled,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    /// CG iterations summed over all contact iterations.
    pub cg_iterations: usize,
    pub contact_iterations: usize,
    pub active_contacts: usize,
    /// False when CG or the contact active set hit an iteration cap.
    pub converged: bool,
    /// `Some` iff `converged` is false: which cap was hit.
    pub failure: Option<SolveFailure>,
}

#[derive(Debug)]
pub struct SolveResult {
    /// Per-node displacement, xyz interleaved.
    pub displacement: Vec<f64>,
    /// Per-node rotation, xyz interleaved — `Some` for element kinds with
    /// rotational dofs (Bar2 frames), `None` for Hex8.
    pub rotation: Option<Vec<f64>>,
    /// Per-node force applied by the rigid body (nonzero only on the active
    /// contact set), xyz interleaved. Along the contact axis the sign
    /// presses toward the glued face.
    pub contact_force: Vec<f64>,
    /// Per-element strain energy per unit element volume (cell volume for
    /// Hex8, strut volume `A * L` for Bar2).
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
///
/// Public because grid consumers beyond the solver (e.g. the density
/// extractor's cell-grid reconstruction) share the same contract.
pub fn detect_uniform_grid(mesh: &FeaMesh) -> Result<f64, String> {
    if mesh.element_kind != FeaElementKind::Hex8 {
        return Err(format!(
            "expected a Hex8 grid mesh, got {:?} elements",
            mesh.element_kind
        ));
    }
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
pub(crate) fn stiffness_scales(mesh: &FeaMesh) -> Result<Vec<f64>, String> {
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

/// The assembled stiffness action of one mesh under a concrete element
/// formulation — everything the contact driver needs that depends on the
/// element kind.
trait StiffnessModel {
    /// Degrees of freedom per node (3 for Hex8, 6 for Bar2 frames). The
    /// first three dofs of a node are always its xyz translations.
    fn dofs_per_node(&self) -> usize;
    /// y = K x, unmasked (matrix-free over elements).
    fn apply(&self, x: &[f64], y: &mut [f64]);
    /// The assembled diagonal of K, for Jacobi preconditioning.
    fn diagonal(&self) -> Vec<f64>;
    /// The node-diagonal `dpn x dpn` blocks of K (row-major, `node_count`
    /// of them), for block-Jacobi preconditioning — much stronger than the
    /// scalar diagonal when stiffness contrast is high or the per-node dofs
    /// couple (frame translation/rotation). `None` falls back to scalar.
    fn node_blocks(&self) -> Option<Vec<f64>> {
        None
    }
    /// Per-element strain energy per unit element volume at solution `u`.
    fn energy_density(&self, u: &[f64]) -> Vec<f64>;
    /// Characteristic element length: sets the contact scan step and the
    /// fixed-face/slack tolerances.
    fn length_scale(&self) -> f64;
}

/// Uniform-grid hex8 stiffness: one shared element matrix, scaled per
/// element by `stiffness_scale`.
struct HexModel<'a> {
    mesh: &'a FeaMesh,
    ke: ElementStiffness,
    scales: Vec<f64>,
    h: f64,
}

impl StiffnessModel for HexModel<'_> {
    fn dofs_per_node(&self) -> usize {
        3
    }

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
    }

    fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0f64; self.mesh.node_count() * 3];
        for e in 0..self.mesh.element_count() {
            let nodes = self.mesh.element(e);
            for (i, node) in nodes.iter().enumerate() {
                for c in 0..3 {
                    diag[*node as usize * 3 + c] += self.scales[e] * self.ke[i * 3 + c][i * 3 + c];
                }
            }
        }
        diag
    }

    fn energy_density(&self, u: &[f64]) -> Vec<f64> {
        // (1/2) u_e^T K_e u_e / h^3 per element.
        let volume = self.h * self.h * self.h;
        let mut out = Vec::with_capacity(self.mesh.element_count());
        for e in 0..self.mesh.element_count() {
            let nodes = self.mesh.element(e);
            let mut ue = [0.0f64; 24];
            for (i, node) in nodes.iter().enumerate() {
                let base = *node as usize * 3;
                ue[i * 3] = u[base];
                ue[i * 3 + 1] = u[base + 1];
                ue[i * 3 + 2] = u[base + 2];
            }
            let mut energy = 0.0;
            for (i, uei) in ue.iter().enumerate() {
                let row = &self.ke[i];
                let mut sum = 0.0;
                for (j, uej) in ue.iter().enumerate() {
                    sum += row[j] * uej;
                }
                energy += uei * sum;
            }
            out.push(0.5 * self.scales[e] * energy / volume);
        }
        out
    }

    fn length_scale(&self) -> f64 {
        self.h
    }
}

/// The CG preconditioner: scalar Jacobi, per-node block Jacobi when the
/// model can assemble its node-diagonal blocks, or a prebuilt two-level
/// Schwarz preconditioner (borrowed — it outlives the per-contact-iteration
/// scalar/block rebuilds).
enum Precond<'a> {
    /// Reciprocal-safe scalar diagonal.
    Scalar(Vec<f64>),
    /// Per-node inverted blocks (row-major `dpn x dpn`), constraint
    /// rows/cols replaced by identity before inverting — the apply is a
    /// plain per-node matvec, which vectorizes far better than triangular
    /// solves.
    Block { dpn: usize, inverses: Vec<f64> },
    #[cfg(feature = "parallel")]
    Schwarz(&'a schwarz::SchwarzPrecond),
    #[cfg(not(feature = "parallel"))]
    #[allow(dead_code)]
    Never(std::marker::PhantomData<&'a ()>),
}

/// Mask per-node blocks against the constraint set and invert them for
/// block-Jacobi application: constrained (and zero-stiffness) dofs become
/// identity rows/cols so they pass through untouched, numerically degenerate
/// blocks fall back to their reciprocal diagonal.
pub(crate) fn invert_node_blocks(blocks: &[f64], dpn: usize, constrained: &[bool]) -> Vec<f64> {
    let node_count = constrained.len() / dpn;
    let mut inverses = vec![0.0f64; blocks.len()];
    let mut factor = vec![0.0f64; dpn * dpn];
    for node in 0..node_count {
        factor.copy_from_slice(&blocks[node * dpn * dpn..(node + 1) * dpn * dpn]);
        for i in 0..dpn {
            if constrained[node * dpn + i] || factor[i * dpn + i] <= 0.0 {
                for j in 0..dpn {
                    factor[i * dpn + j] = 0.0;
                    factor[j * dpn + i] = 0.0;
                }
                factor[i * dpn + i] = 1.0;
            }
        }
        let inv = &mut inverses[node * dpn * dpn..(node + 1) * dpn * dpn];
        if cholesky_in_place(&mut factor, dpn) {
            // Invert via solves against identity columns.
            let mut col = vec![0.0f64; dpn];
            for j in 0..dpn {
                col.fill(0.0);
                col[j] = 1.0;
                cholesky_solve_in_place(&factor, dpn, &mut col);
                for i in 0..dpn {
                    inv[i * dpn + j] = col[i];
                }
            }
        } else {
            for i in 0..dpn {
                let d = blocks[node * dpn * dpn + i * dpn + i];
                inv[i * dpn + i] = if d > 0.0 { 1.0 / d } else { 1.0 };
            }
        }
    }
    inverses
}

impl Precond<'_> {
    /// Build for the current constraint set. `diag`/`blocks` are the
    /// unmasked assemblies, computed once per solve; constraints change per
    /// contact iteration, so masking and factoring happen here.
    fn build(
        diag: &[f64],
        blocks: Option<&[f64]>,
        dpn: usize,
        constrained: &[bool],
    ) -> Self {
        let Some(blocks) = blocks else {
            return Self::Scalar(diag.to_vec());
        };
        Self::Block {
            dpn,
            inverses: invert_node_blocks(blocks, dpn, constrained),
        }
    }

    fn apply(&self, r: &[f64], z: &mut [f64], constrained: &[bool]) {
        match self {
            Self::Scalar(diag) => {
                for i in 0..r.len() {
                    z[i] = if diag[i] > 0.0 { r[i] / diag[i] } else { r[i] };
                }
            }
            #[cfg(feature = "parallel")]
            Self::Schwarz(precond) => precond.apply(r, z, constrained),
            #[cfg(not(feature = "parallel"))]
            Self::Never(_) => {
                let _ = constrained;
                unreachable!()
            }
            Self::Block { dpn, inverses } => {
                let dpn = *dpn;
                for node in 0..r.len() / dpn {
                    let inv = &inverses[node * dpn * dpn..(node + 1) * dpn * dpn];
                    let rn = &r[node * dpn..(node + 1) * dpn];
                    let zn = &mut z[node * dpn..(node + 1) * dpn];
                    for i in 0..dpn {
                        zn[i] = (0..dpn).map(|j| inv[i * dpn + j] * rn[j]).sum();
                    }
                }
            }
        }
    }
}

/// In-place dense Cholesky (lower triangle, row-major). Returns false when
/// the matrix is not numerically SPD.
fn cholesky_in_place(a: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return false;
                }
                a[i * n + i] = sum.sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
        for j in (i + 1)..n {
            a[i * n + j] = 0.0;
        }
    }
    true
}

/// Solve `L L^T x = b` in place given the lower factor.
fn cholesky_solve_in_place(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * b[k];
        }
        b[i] = sum / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * b[k];
        }
        b[i] = sum / l[i * n + i];
    }
}

/// Preconditioned CG for `K u = 0` with Dirichlet values already
/// written into `u`. Returns (iterations, converged).
fn solve_cg(
    model: &dyn StiffnessModel,
    constrained: &[bool],
    precond: &Precond,
    u: &mut [f64],
    tolerance: f64,
    max_iterations: usize,
) -> (usize, bool) {
    // External forces are zero; the Dirichlet values in `u` drive.
    solve_cg_system(
        model,
        constrained,
        precond,
        None,
        u,
        tolerance,
        max_iterations,
    )
}

/// Preconditioned CG for `K x = rhs` on the free dofs, holding constrained
/// dofs at their current `x` values (`None` rhs means zero). Returns
/// (iterations, converged).
fn solve_cg_system(
    model: &dyn StiffnessModel,
    constrained: &[bool],
    precond: &Precond,
    rhs: Option<&[f64]>,
    x: &mut [f64],
    tolerance: f64,
    max_iterations: usize,
) -> (usize, bool) {
    let u = x;
    let n = u.len();
    let masked_apply = |x: &[f64], y: &mut [f64]| {
        model.apply(x, y);
        for (v, c) in y.iter_mut().zip(constrained) {
            if *c {
                *v = 0.0;
            }
        }
    };
    let mut r = vec![0.0; n];
    // r = (rhs - K x) on free dofs.
    model.apply(u, &mut r);
    for i in 0..n {
        r[i] = if constrained[i] {
            0.0
        } else {
            rhs.map_or(0.0, |b| b[i]) - r[i]
        };
    }

    let norm0: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm0 == 0.0 {
        return (0, true);
    }
    let target = norm0 * tolerance;

    let precond = |r: &[f64], z: &mut [f64]| precond.apply(r, z, constrained);

    let mut z = vec![0.0; n];
    precond(&r, &mut z);
    let mut p = z.clone();
    let mut kp = vec![0.0; n];
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();

    for iteration in 0..max_iterations {
        masked_apply(&p, &mut kp);
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

/// Solve the compression problem. See the module docs for scope; the
/// element formulation is picked by the mesh's element kind.
pub fn solve(
    mesh: &FeaMesh,
    rigid: &mut dyn RigidBody,
    config: &SolveConfig,
) -> Result<SolveResult, String> {
    mesh.validate()?;
    let nu = config.material.poissons_ratio;
    if !(-1.0 < nu && nu < 0.5) {
        return Err(format!("Poisson's ratio {nu} outside (-1, 0.5)"));
    }

    let node_count = mesh.node_count();
    if mesh.element_count() == 0 {
        return Ok(SolveResult {
            displacement: vec![0.0; node_count * 3],
            rotation: (mesh.element_kind == FeaElementKind::Bar2)
                .then(|| vec![0.0; node_count * 3]),
            contact_force: vec![0.0; node_count * 3],
            strain_energy_density: Vec::new(),
            stats: SolveStats {
                converged: true,
                ..Default::default()
            },
        });
    }

    match mesh.element_kind {
        FeaElementKind::Hex8 => {
            let h = detect_uniform_grid(mesh)?;
            let scales = stiffness_scales(mesh)?;
            let ke = cube_stiffness(h, config.material)?;
            let model = HexModel {
                mesh,
                ke,
                scales,
                h,
            };
            Ok(contact_solve(mesh, &model, None, rigid, config)?.0)
        }
        FeaElementKind::Bar2 => Ok(frame_contact_solve(mesh, rigid, config)?.0),
    }
}

/// The Bar2 contact solve with `stress_stiffening_passes` Picard passes:
/// solve, feed the tensile axial forces into the geometric stiffness,
/// re-solve against the stiffened tangent. Returns the model at its final
/// prestress state (what the adjoint must differentiate against) and the
/// last pass's internals; stats accumulate the cost of every pass.
fn frame_contact_solve(
    mesh: &FeaMesh,
    rigid: &mut dyn RigidBody,
    config: &SolveConfig,
) -> Result<(SolveResult, frame::FrameModel, SolveInternals), String> {
    let mut model = frame::FrameModel::new(mesh, config.material)?;
    let (mut result, mut internals) = contact_solve(mesh, &model, Some(&model), rigid, config)?;
    for _ in 0..config.stress_stiffening_passes {
        if !result.stats.converged {
            break;
        }
        model.update_prestress(&internals.u);
        let (next, next_internals) = contact_solve(mesh, &model, Some(&model), rigid, config)?;
        result = SolveResult {
            stats: SolveStats {
                cg_iterations: result.stats.cg_iterations + next.stats.cg_iterations,
                contact_iterations: result.stats.contact_iterations
                    + next.stats.contact_iterations,
                ..next.stats
            },
            ..next
        };
        internals = next_internals;
    }
    Ok((result, model, internals))
}

/// Everything the converged contact state exposes beyond the result fields
/// — what an adjoint solve needs to differentiate through the (frozen)
/// active set: the full dof-space solution, the constraint mask it was
/// solved under, and the active contact nodes.
pub(crate) struct SolveInternals {
    /// Full dof-space solution (dpn dofs per node, prescribed values
    /// included).
    pub(crate) u: Vec<f64>,
    /// Per-dof constraint mask at convergence (fixed face + active
    /// contacts).
    pub(crate) constrained: Vec<bool>,
    /// Active contact node indices. Part of the adjoint contract (an
    /// objective may need the full frozen set, not just the compressive
    /// nodes); currently read by the FD validation harness only.
    #[allow(dead_code)]
    pub(crate) active: Vec<usize>,
    pub(crate) dpn: usize,
}

/// The Bar2 forward solve, keeping the pieces the inverse gradient needs:
/// the frame model (element stiffness access, at final prestress) and the
/// converged contact state. Mirrors what `solve` does for Bar2.
pub(crate) fn solve_frame_internal(
    mesh: &FeaMesh,
    rigid: &mut dyn RigidBody,
    config: &SolveConfig,
) -> Result<(SolveResult, frame::FrameModel, SolveInternals), String> {
    mesh.validate()?;
    let nu = config.material.poissons_ratio;
    if !(-1.0 < nu && nu < 0.5) {
        return Err(format!("Poisson's ratio {nu} outside (-1, 0.5)"));
    }
    frame_contact_solve(mesh, rigid, config)
}

/// Solve the adjoint system `K_ff lambda_f = (K ghat)|_f`, `lambda = 0` on
/// constrained dofs — the same operator, constraint mask, and
/// preconditioner family as the forward solve's final contact iteration.
/// With it, dJ/ds_e = (ghat - lambda)^T (dK_e/ds_e) u for any objective
/// whose reaction-space gradient is `ghat` (zero off the reaction dofs).
pub(crate) fn adjoint_solve(
    model: &dyn StiffnessModel,
    internals: &SolveInternals,
    ghat: &[f64],
    config: &SolveConfig,
) -> Result<Vec<f64>, String> {
    let n = internals.u.len();
    let mut rhs = vec![0.0f64; n];
    model.apply(ghat, &mut rhs);
    let diag = model.diagonal();
    let blocks = model.node_blocks();
    let precond = Precond::build(
        &diag,
        blocks.as_deref(),
        internals.dpn,
        &internals.constrained,
    );
    let mut lambda = vec![0.0f64; n];
    let (_, converged) = solve_cg_system(
        model,
        &internals.constrained,
        &precond,
        Some(&rhs),
        &mut lambda,
        config.cg_tolerance,
        config.cg_max_iterations,
    );
    if !converged {
        return Err("adjoint CG did not converge".to_string());
    }
    Ok(lambda)
}

/// The active-set contact driver: element-kind-independent, working on the
/// assembled stiffness action. Nodes on the fixed face are glued in all
/// dofs; contact prescribes the translational contact-axis dof.
///
/// `frame` is the concrete model when the elements are Bar2 struts — the
/// Schwarz preconditioner assembles subdomain matrices from struts and so
/// cannot be built from the type-erased stiffness action alone.
fn contact_solve(
    mesh: &FeaMesh,
    model: &dyn StiffnessModel,
    frame: Option<&frame::FrameModel>,
    rigid: &mut dyn RigidBody,
    config: &SolveConfig,
) -> Result<(SolveResult, SolveInternals), String> {
    let dpn = model.dofs_per_node();
    let node_count = mesh.node_count();
    let n = node_count * dpn;
    let h = model.length_scale();

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
    let (contact_axis, contact_sign) = config.fixed_boundary.contact_axis();
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
                for c in 0..dpn {
                    constrained[node * dpn + c] = true;
                }
            }
        }
    }

    // Unmasked preconditioner assemblies; the constraint-dependent masking
    // and factoring happen per contact iteration. The Schwarz preconditioner
    // is built ONCE against the glued-face constraints (rebuilding dense
    // subdomain factors per contact iteration would dwarf the CG cost);
    // contact constraints are handled by masking its output.
    let diag = model.diagonal();
    let blocks = model.node_blocks();
    #[cfg(feature = "parallel")]
    let schwarz_precond = match (config.preconditioner, frame) {
        (PrecondChoice::Schwarz(params), Some(frame_model)) => {
            schwarz::SchwarzPrecond::build(mesh, frame_model, &constrained, params)
        }
        _ => None,
    };
    #[cfg(not(feature = "parallel"))]
    let _ = frame;

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
            let base = node * dpn;
            let deformed = [p[0] + u[base], p[1] + u[base + 1], p[2] + u[base + 2]];
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
            let dof = node * dpn + contact_axis;
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

        #[cfg(feature = "parallel")]
        let precond = match &schwarz_precond {
            Some(schwarz) => Precond::Schwarz(schwarz),
            None => Precond::build(&diag, blocks.as_deref(), dpn, &constrained),
        };
        #[cfg(not(feature = "parallel"))]
        let precond = Precond::build(&diag, blocks.as_deref(), dpn, &constrained);
        let (iterations, cg_converged) = solve_cg(
            model,
            &constrained,
            &precond,
            &mut u,
            config.cg_tolerance,
            config.cg_max_iterations,
        );
        stats.cg_iterations += iterations;
        if !cg_converged {
            stats.converged = false;
            stats.failure = Some(SolveFailure::CgStalled);
            stats.active_contacts = active.len();
            break;
        }

        // Reactions: external force = K u at constrained dofs. A contact
        // constraint may only press toward the glued face (f·sign <= 0);
        // meaningfully tensile ones release. The threshold is relative to
        // the peak compression so noise-level forces at grazing rim nodes
        // don't cycle the set.
        model.apply(&u, &mut forces);
        let peak_compression = active
            .keys()
            .map(|node| -contact_sign * forces[node * dpn + contact_axis])
            .fold(0.0f64, f64::max);
        let release_tol = peak_compression * 1e-3;
        let released: Vec<usize> = active
            .keys()
            .copied()
            .filter(|node| contact_sign * forces[node * dpn + contact_axis] > release_tol)
            .collect();
        for node in &released {
            active.remove(node);
        }

        // Set FEA_CONTACT_DEBUG=1 to watch the active-set fixed point: a
        // healthy solve shows released/set_changes decaying to zero; a
        // livelock shows them oscillating. (No env on wasm32 — native only.)
        if std::env::var("FEA_CONTACT_DEBUG").is_ok() {
            eprintln!(
                "contact iter {}: active={} set_changes={} released={} cg={}",
                stats.contact_iterations,
                active.len(),
                set_changes,
                released.len(),
                iterations,
            );
        }

        if set_changes == 0 && released.is_empty() {
            stats.converged = true;
            break;
        }
    }
    if !stats.converged && stats.failure.is_none() {
        stats.failure = Some(SolveFailure::ContactUnsettled);
    }
    stats.active_contacts = active.len();

    // Contact force field: the reaction at each active node (translational,
    // 3 per node regardless of the model's dof count).
    model.apply(&u, &mut forces);
    let mut contact_force = vec![0.0f64; node_count * 3];
    for node in active.keys() {
        contact_force[node * 3 + contact_axis] = forces[node * dpn + contact_axis];
    }

    let strain_energy_density = model.energy_density(&u);

    // Split the solution into translations (+ rotations for 6-dof models).
    let (displacement, rotation) = if dpn == 3 {
        (u.clone(), None)
    } else {
        let mut translations = Vec::with_capacity(node_count * 3);
        let mut rotations = Vec::with_capacity(node_count * 3);
        for node in 0..node_count {
            let base = node * dpn;
            translations.extend_from_slice(&u[base..base + 3]);
            rotations.extend_from_slice(&u[base + 3..base + 6]);
        }
        (translations, Some(rotations))
    };

    let internals = SolveInternals {
        u,
        constrained,
        active: active.keys().copied().collect(),
        dpn,
    };
    Ok((
        SolveResult {
            displacement,
            rotation,
            contact_force,
            strain_energy_density,
            stats,
        },
        internals,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::fea::FeaField;

    /// Build the uniform grid mesh the mesher would emit: nx*ny*nz cells of
    /// size h with the origin at (0, 0, 0). Shared with the inverse module's
    /// tests.
    pub(crate) fn grid_mesh(nx: usize, ny: usize, nz: usize, h: f64) -> FeaMesh {
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
    fn iteration_caps_classify_the_failure() {
        // The sphere press needs several contact sweeps and nontrivial CG
        // work, so either cap can be made the binding one.
        let mesh = grid_mesh(8, 8, 4, 0.125);
        let center = [0.5, 0.5, 0.5 - 0.05 + 1.0];
        let mut sphere =
            move |p: [f64; 3]| (0..3).map(|i| (p[i] - center[i]).powi(2)).sum::<f64>() < 1.0;

        let unsettled = solve(
            &mesh,
            &mut sphere,
            &SolveConfig {
                max_contact_iterations: 1,
                ..config(0.3)
            },
        )
        .unwrap();
        assert!(!unsettled.stats.converged);
        assert_eq!(
            unsettled.stats.failure,
            Some(SolveFailure::ContactUnsettled)
        );

        let stalled = solve(
            &mesh,
            &mut sphere,
            &SolveConfig {
                cg_max_iterations: 2,
                ..config(0.3)
            },
        )
        .unwrap();
        assert!(!stalled.stats.converged);
        assert_eq!(stalled.stats.failure, Some(SolveFailure::CgStalled));

        let healthy = solve(&mesh, &mut sphere, &config(0.3)).unwrap();
        assert!(healthy.stats.converged);
        assert_eq!(healthy.stats.failure, None);
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
    fn grid_detection_rejects_line_meshes() {
        // detect_uniform_grid must not spuriously accept a Bar2 mesh (the
        // corner zip would only check two nodes).
        let mut mesh = grid_mesh(2, 2, 2, 0.5);
        mesh.element_kind = FeaElementKind::Bar2;
        mesh.connectivity = vec![0, 1, 1, 2];
        let err = detect_uniform_grid(&mesh).unwrap_err();
        assert!(err.contains("Hex8"), "unexpected error: {err}");
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
