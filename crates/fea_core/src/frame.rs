//! 3D Euler-Bernoulli frame elements over Bar2 strut meshes.
//!
//! Each strut is a two-node beam with a circular cross-section: axial,
//! torsional, and two bending stiffnesses (`A = pi r^2`, `I = pi r^4 / 4`
//! about both transverse axes, `J = 2 I`, `G = E / (2 (1 + nu))`), giving
//! 6 dofs per node — 3 translations then 3 rotations. Bending stiffness is
//! what keeps strut lattices well-posed: most families (cubic, diamond,
//! foam) are mechanisms as pin-jointed trusses, and real printed foams are
//! bending-dominated.
//!
//! Per-strut inputs from the mesh:
//! - `radius` (scalar element field, required): the cross-section radius.
//! - `stiffness_scale` (scalar element field, optional): multiplies the
//!   strut's Young's modulus — the dimensionless mechanical-property knob
//!   the inverse loop drives; realizing scales as printed radii is the
//!   lattice-model step's job.
//!
//! Everything is precomputed per strut at construction: the local frame
//! (direction cosines) and the classic local stiffness coefficients. The
//! matrix-free apply rotates each node's dof block into the strut frame,
//! evaluates the closed-form local stiffness, and rotates back.

use crate::{Material, StiffnessModel, stiffness_scales};
use volumetric_abi::fea::FeaMesh;

/// Per-strut precomputed data. All stiffness coefficients include the
/// strut's `stiffness_scale`.
struct Strut {
    nodes: [u32; 2],
    /// Rows are the local axes (x' along the strut) in global coordinates.
    r: [[f64; 3]; 3],
    /// Strut length.
    length: f64,
    /// Axial `E A / L`.
    k_ax: f64,
    /// Torsional `G J / L`.
    k_tor: f64,
    /// Bending `12 E I / L^3`.
    b1: f64,
    /// Bending `6 E I / L^2`.
    b2: f64,
    /// Bending `4 E I / L`.
    b3: f64,
    /// Bending `2 E I / L`.
    b4: f64,
    /// Geometric (stress-stiffening) bending terms from the axial prestress
    /// `N`: `6N/5L`, `N/10`, `2NL/15`, `-NL/30` — the consistent
    /// beam-column matrix, same block structure as the elastic terms. Zero
    /// until `update_prestress` runs.
    g1: f64,
    g2: f64,
    g3: f64,
    g4: f64,
    /// Cross-section volume `A * L` (for energy density).
    volume: f64,
}

/// The assembled frame stiffness of a Bar2 mesh.
pub(crate) struct FrameModel {
    struts: Vec<Strut>,
    node_count: usize,
    /// Mean strut length: the contact scan step / tolerance scale.
    mean_length: f64,
    /// CSR node -> incident (strut, end) adjacency for the conflict-free
    /// parallel apply; entries pack `strut << 1 | end`.
    #[cfg(feature = "parallel")]
    incidence_offsets: Vec<u32>,
    #[cfg(feature = "parallel")]
    incidence: Vec<u32>,
}

/// Below this strut count the parallel apply's fork-join and scratch-buffer
/// overhead outweighs the win; the serial path runs instead. Measured with
/// `apply_scaling_bench` (cubic lattices, 8-24 threads): ~0.4-0.55x at 2.7k
/// struts, ~0.95-1.4x at 30k, ~1.2x at 300k+ — the kernel is memory-bandwidth
/// bound, so the crossover sits high and the ceiling is modest.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_STRUTS: usize = 32 * 1024;

/// Build the node -> incident (strut, end) CSR adjacency.
#[cfg(feature = "parallel")]
fn build_incidence(struts: &[Strut], node_count: usize) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; node_count + 1];
    for s in struts {
        for &node in &s.nodes {
            offsets[node as usize + 1] += 1;
        }
    }
    for i in 0..node_count {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor: Vec<u32> = offsets[..node_count].to_vec();
    let mut entries = vec![0u32; offsets[node_count] as usize];
    for (strut, s) in struts.iter().enumerate() {
        for (end, &node) in s.nodes.iter().enumerate() {
            let slot = &mut cursor[node as usize];
            entries[*slot as usize] = (strut as u32) << 1 | end as u32;
            *slot += 1;
        }
    }
    (offsets, entries)
}

impl FrameModel {
    pub(crate) fn new(mesh: &FeaMesh, material: Material) -> Result<Self, String> {
        let radius = mesh
            .element_fields
            .iter()
            .find(|f| f.name == "radius" && f.components == 1)
            .ok_or_else(|| {
                "Bar2 meshes need a scalar `radius` element field (the strut \
                 cross-section radius)"
                    .to_string()
            })?;
        let scales = stiffness_scales(mesh)?;
        let e_modulus = material.youngs_modulus;
        let g_modulus = e_modulus / (2.0 * (1.0 + material.poissons_ratio));

        let mut struts = Vec::with_capacity(mesh.element_count());
        let mut total_length = 0.0;
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
            let r = radius.data[e];
            if !(r.is_finite() && r > 0.0) {
                return Err(format!("strut {e} has invalid radius {r}"));
            }
            total_length += length;

            // Local frame: x' along the strut; y'/z' any orthonormal pair
            // (the section is circular). Seed the cross product with the
            // global axis least aligned with the strut.
            let x_axis = [d[0] / length, d[1] / length, d[2] / length];
            let smallest = (0..3)
                .min_by(|&a, &b| x_axis[a].abs().total_cmp(&x_axis[b].abs()))
                .unwrap();
            let mut seed = [0.0; 3];
            seed[smallest] = 1.0;
            let y_axis = normalize(cross(seed, x_axis));
            let z_axis = cross(x_axis, y_axis);

            let scale = scales[e];
            let area = std::f64::consts::PI * r * r;
            let inertia = std::f64::consts::PI * r.powi(4) / 4.0;
            let ei = e_modulus * scale * inertia;
            struts.push(Strut {
                nodes: [nodes[0], nodes[1]],
                r: [x_axis, y_axis, z_axis],
                length,
                k_ax: e_modulus * scale * area / length,
                k_tor: g_modulus * scale * (2.0 * inertia) / length,
                b1: 12.0 * ei / length.powi(3),
                b2: 6.0 * ei / (length * length),
                b3: 4.0 * ei / length,
                b4: 2.0 * ei / length,
                g1: 0.0,
                g2: 0.0,
                g3: 0.0,
                g4: 0.0,
                volume: area * length,
            });
        }

        #[cfg(feature = "parallel")]
        let (incidence_offsets, incidence) = build_incidence(&struts, mesh.node_count());
        Ok(Self {
            struts,
            node_count: mesh.node_count(),
            mean_length: total_length / mesh.element_count() as f64,
            #[cfg(feature = "parallel")]
            incidence_offsets,
            #[cfg(feature = "parallel")]
            incidence,
        })
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Local end forces for local end displacements, closed form. Blocks are
/// `[u1, theta1, u2, theta2]`, each a 3-vector in the strut frame
/// (component 0 = axial/twist, 1 and 2 = the transverse axes).
///
/// Bending couples deflection along y' with rotation about z' (same-sign
/// coupling) and deflection along z' with rotation about y' (opposite-sign
/// coupling) — the classic Euler-Bernoulli sign structure. Verified by the
/// rigid-body null-space test below.
fn local_forces(s: &Strut, v: &[[f64; 3]; 4]) -> [[f64; 3]; 4] {
    let [u1, th1, u2, th2] = *v;
    let mut f1 = [0.0; 3];
    let mut m1 = [0.0; 3];
    let mut f2 = [0.0; 3];
    let mut m2 = [0.0; 3];

    // Axial and torsional two-node springs.
    f1[0] = s.k_ax * (u1[0] - u2[0]);
    f2[0] = -f1[0];
    m1[0] = s.k_tor * (th1[0] - th2[0]);
    m2[0] = -m1[0];

    // Bending terms: elastic plus the geometric (stress-stiffening) matrix,
    // which shares the elastic block structure — a taut strut resists
    // transverse displacement like a string on top of its bending stiffness.
    let c1 = s.b1 + s.g1;
    let c2 = s.b2 + s.g2;
    let c3 = s.b3 + s.g3;
    let c4 = s.b4 + s.g4;

    // Bending in the x'-y' plane: (u_y, theta_z).
    f1[1] = c1 * u1[1] + c2 * th1[2] - c1 * u2[1] + c2 * th2[2];
    m1[2] = c2 * u1[1] + c3 * th1[2] - c2 * u2[1] + c4 * th2[2];
    f2[1] = -f1[1];
    m2[2] = c2 * u1[1] + c4 * th1[2] - c2 * u2[1] + c3 * th2[2];

    // Bending in the x'-z' plane: (u_z, theta_y), opposite coupling signs.
    f1[2] = c1 * u1[2] - c2 * th1[1] - c1 * u2[2] - c2 * th2[1];
    m1[1] = -c2 * u1[2] + c3 * th1[1] + c2 * u2[2] + c4 * th2[1];
    f2[2] = -f1[2];
    m2[1] = -c2 * u1[2] + c4 * th1[1] + c2 * u2[2] + c3 * th2[1];

    [f1, m1, f2, m2]
}

impl FrameModel {
    pub(crate) fn strut_count(&self) -> usize {
        self.struts.len()
    }

    pub(crate) fn strut_nodes(&self, e: usize) -> [u32; 2] {
        self.struts[e].nodes
    }

    /// The strut's assembled 12x12 global-frame element stiffness (dof
    /// order: node0 translations, node0 rotations, node1 translations,
    /// node1 rotations), column-probed out of local_forces so every
    /// coupling and sign comes from the ground-truth kernel.
    pub(crate) fn element_stiffness(&self, e: usize) -> [[f64; 12]; 12] {
        let s = &self.struts[e];
        // Columns in the strut frame first.
        let mut local = [[0.0f64; 12]; 12];
        for col in 0..12 {
            let mut v = [[0.0f64; 3]; 4];
            v[col / 3][col % 3] = 1.0;
            let f = local_forces(s, &v);
            for row in 0..12 {
                local[row][col] = f[row / 3][row % 3];
            }
        }
        // K_global = Rblk^T K_local Rblk with Rblk = blkdiag(R, R, R, R):
        // every 3x3 quadrant transforms as R^T Q R.
        let mut out = [[0.0f64; 12]; 12];
        for qi in 0..4 {
            for qj in 0..4 {
                for i in 0..3 {
                    for j in 0..3 {
                        let mut sum = 0.0;
                        for k in 0..3 {
                            for l in 0..3 {
                                sum += s.r[k][i] * local[qi * 3 + k][qj * 3 + l] * s.r[l][j];
                            }
                        }
                        out[qi * 3 + i][qj * 3 + j] = sum;
                    }
                }
            }
        }
        out
    }

    /// The strut's global-frame internal end forces `K_e u_e` at the global
    /// solution `x` (dof order as `element_stiffness`: node0 translations,
    /// node0 rotations, node1 translations, node1 rotations). Since every
    /// stiffness coefficient is linear in the strut's `stiffness_scale`,
    /// `dK_e/ds_e u = strut_end_forces(e, u) / s_e` — the element factor of
    /// the adjoint gradient.
    pub(crate) fn strut_end_forces(&self, e: usize, x: &[f64]) -> [f64; 12] {
        let s = &self.struts[e];
        let local = self.gather_local(s, x);
        let f = local_forces(s, &local);
        let mut out = [0.0f64; 12];
        for (block, fl) in f.iter().enumerate() {
            for col in 0..3 {
                out[block * 3 + col] =
                    s.r[0][col] * fl[0] + s.r[1][col] * fl[1] + s.r[2][col] * fl[2];
            }
        }
        out
    }

    /// Cross-section volume `A * L` of one strut.
    pub(crate) fn strut_volume(&self, e: usize) -> f64 {
        self.struts[e].volume
    }

    /// Refresh the geometric (stress-stiffening) terms from the axial
    /// forces at solution `u` — one Picard pass of the beam-column
    /// nonlinearity. The elongation uses the von Karman (second-order)
    /// strain: a lateral strut whose ends sag by different amounts
    /// stretches by `(transverse difference)^2 / 2L` even though its
    /// first-order axial strain is zero — that quadratic term IS the
    /// hammock tension, so a linear strain measure would find nothing to
    /// stiffen. Tension only: compressive geometric stiffness is negative
    /// (buckling softening) and can make K indefinite, which CG cannot
    /// solve; the hammock effect this models lives entirely in the tensile
    /// struts.
    pub(crate) fn update_prestress(&mut self, u: &[f64]) {
        let axial: Vec<f64> = self
            .struts
            .iter()
            .map(|s| {
                let local = self.gather_local(s, u);
                let dt_y = local[2][1] - local[0][1];
                let dt_z = local[2][2] - local[0][2];
                let elongation =
                    (local[2][0] - local[0][0]) + (dt_y * dt_y + dt_z * dt_z) / (2.0 * s.length);
                s.k_ax * elongation
            })
            .collect();
        for (s, n) in self.struts.iter_mut().zip(axial) {
            let n = n.max(0.0);
            let l = s.length;
            s.g1 = 6.0 * n / (5.0 * l);
            s.g2 = n / 10.0;
            s.g3 = 2.0 * n * l / 15.0;
            s.g4 = -n * l / 30.0;
        }
    }

    /// Gather a strut's four dof blocks from the global vector, rotated
    /// into the strut frame.
    fn gather_local(&self, s: &Strut, x: &[f64]) -> [[f64; 3]; 4] {
        let mut local = [[0.0f64; 3]; 4];
        for (block, l) in local.iter_mut().enumerate() {
            let off = s.nodes[block / 2] as usize * 6 + (block % 2) * 3;
            for (row, value) in l.iter_mut().enumerate() {
                *value = s.r[row][0] * x[off]
                    + s.r[row][1] * x[off + 1]
                    + s.r[row][2] * x[off + 2];
            }
        }
        local
    }
}

impl FrameModel {
    fn apply_serial(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        for s in &self.struts {
            let local = self.gather_local(s, x);
            let f = local_forces(s, &local);
            // Scatter back: f_global = R^T f_local per block.
            for (block, fl) in f.iter().enumerate() {
                let off = s.nodes[block / 2] as usize * 6 + (block % 2) * 3;
                for col in 0..3 {
                    y[off + col] +=
                        s.r[0][col] * fl[0] + s.r[1][col] * fl[1] + s.r[2][col] * fl[2];
                }
            }
        }
    }

    /// The apply as two conflict-free parallel passes: per-strut global-frame
    /// end forces into a scratch buffer (blocks `[f1, m1, f2, m2]`, so end 0
    /// owns 0..6 and end 1 owns 6..12), then a per-node gather over the
    /// incidence list. Result is bit-identical to the serial scatter apart
    /// from float addition order.
    #[cfg(feature = "parallel")]
    fn apply_parallel(&self, x: &[f64], y: &mut [f64]) {
        use rayon::prelude::*;

        let mut contributions = vec![0.0f64; self.struts.len() * 12];
        contributions
            .par_chunks_mut(12)
            .zip(self.struts.par_iter())
            .for_each(|(out, s)| {
                let local = self.gather_local(s, x);
                let f = local_forces(s, &local);
                // f_global = R^T f_local per block.
                for (block, fl) in f.iter().enumerate() {
                    for col in 0..3 {
                        out[block * 3 + col] =
                            s.r[0][col] * fl[0] + s.r[1][col] * fl[1] + s.r[2][col] * fl[2];
                    }
                }
            });

        y.par_chunks_mut(6).enumerate().for_each(|(node, block)| {
            block.fill(0.0);
            let start = self.incidence_offsets[node] as usize;
            let end = self.incidence_offsets[node + 1] as usize;
            for &packed in &self.incidence[start..end] {
                let strut = (packed >> 1) as usize;
                let strut_end = (packed & 1) as usize;
                let c = &contributions[strut * 12 + strut_end * 6..][..6];
                for (acc, v) in block.iter_mut().zip(c) {
                    *acc += v;
                }
            }
        });
    }
}

impl StiffnessModel for FrameModel {
    fn dofs_per_node(&self) -> usize {
        6
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        #[cfg(feature = "parallel")]
        if self.struts.len() >= PARALLEL_MIN_STRUTS {
            self.apply_parallel(x, y);
            return;
        }
        self.apply_serial(x, y);
    }

    fn node_blocks(&self) -> Option<Vec<f64>> {
        // q_global = R^T q_local R for a 3x3 quadrant (R rows are the local
        // axes; translations and rotations rotate independently).
        fn rotate(r: &[[f64; 3]; 3], q: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
            let mut qr = [[0.0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    qr[i][j] = (0..3).map(|k| q[i][k] * r[k][j]).sum();
                }
            }
            let mut out = [[0.0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j] = (0..3).map(|k| r[k][i] * qr[k][j]).sum();
                }
            }
            out
        }

        let mut blocks = vec![0.0f64; self.node_count * 36];
        for s in &self.struts {
            for end in 0..2 {
                // Column-probe this end's 6x6 in the strut frame with unit
                // end displacements — local_forces is the ground-truth
                // kernel, so the block picks up every coupling (bending ties
                // translations to rotations at the same node) with no
                // re-derived signs. Rows/cols: (f, m) x (u, th) of this end.
                let mut local = [[0.0f64; 6]; 6];
                for col in 0..6 {
                    let mut v = [[0.0f64; 3]; 4];
                    v[end * 2 + col / 3][col % 3] = 1.0;
                    let f = local_forces(s, &v);
                    for row in 0..6 {
                        local[row][col] = f[end * 2 + row / 3][row % 3];
                    }
                }
                // Rotate each 3x3 quadrant into the global frame and
                // accumulate into the node's block.
                let node = s.nodes[end] as usize;
                let block = &mut blocks[node * 36..(node + 1) * 36];
                for qi in 0..2 {
                    for qj in 0..2 {
                        let mut q = [[0.0f64; 3]; 3];
                        for i in 0..3 {
                            for j in 0..3 {
                                q[i][j] = local[qi * 3 + i][qj * 3 + j];
                            }
                        }
                        let g = rotate(&s.r, &q);
                        for i in 0..3 {
                            for j in 0..3 {
                                block[(qi * 3 + i) * 6 + qj * 3 + j] += g[i][j];
                            }
                        }
                    }
                }
            }
        }
        Some(blocks)
    }

    fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0f64; self.node_count * 6];
        for s in &self.struts {
            // The local per-node blocks are diagonal — translations
            // (k_ax, b1+g1, b1+g1), rotations (k_tor, b3+g3, b3+g3) — so the
            // global diagonal of R^T D R is a weighted sum of squared
            // cosines.
            let translation = [s.k_ax, s.b1 + s.g1, s.b1 + s.g1];
            let rotation = [s.k_tor, s.b3 + s.g3, s.b3 + s.g3];
            for &node in &s.nodes {
                let base = node as usize * 6;
                for col in 0..3 {
                    let mut t = 0.0;
                    let mut w = 0.0;
                    for k in 0..3 {
                        let c2 = s.r[k][col] * s.r[k][col];
                        t += translation[k] * c2;
                        w += rotation[k] * c2;
                    }
                    diag[base + col] += t;
                    diag[base + 3 + col] += w;
                }
            }
        }
        diag
    }

    fn energy_density(&self, u: &[f64]) -> Vec<f64> {
        // (1/2) u_e^T K_e u_e per strut, over the strut volume A L. The
        // quadratic form is frame-invariant, so evaluate it locally.
        self.struts
            .iter()
            .map(|s| {
                let local = self.gather_local(s, u);
                let f = local_forces(s, &local);
                let energy: f64 = local
                    .iter()
                    .zip(&f)
                    .map(|(v, fv)| v[0] * fv[0] + v[1] * fv[1] + v[2] * fv[2])
                    .sum();
                0.5 * energy / s.volume
            })
            .collect()
    }

    fn length_scale(&self) -> f64 {
        self.mean_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedBoundary, SolveConfig, solve, solve_cg};
    use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh};

    const E: f64 = 2.0;
    const NU: f64 = 0.3;

    fn material() -> Material {
        Material {
            youngs_modulus: E,
            poissons_ratio: NU,
        }
    }

    /// A strut chain along `axis_dir` from `origin`, `segments` equal
    /// pieces of total length `length`, radius `r` everywhere.
    fn chain(origin: [f64; 3], axis_dir: [f64; 3], length: f64, segments: usize, r: f64) -> FeaMesh {
        let n = normalize(axis_dir);
        let mut node_positions = Vec::new();
        for i in 0..=segments {
            let t = length * i as f64 / segments as f64;
            node_positions.extend([origin[0] + n[0] * t, origin[1] + n[1] * t, origin[2] + n[2] * t]);
        }
        let connectivity = (0..segments as u32)
            .flat_map(|i| [i, i + 1])
            .collect::<Vec<_>>();
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions,
            connectivity,
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![r; segments],
            }],
        }
    }

    /// A cubic lattice: nodes on an n^3 grid, struts along the three axis
    /// directions — representative strut-mesh topology for apply scaling.
    fn cubic_lattice(n: usize) -> FeaMesh {
        let idx = |i: usize, j: usize, k: usize| (i * n * n + j * n + k) as u32;
        let mut node_positions = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    node_positions.extend([i as f64, j as f64, k as f64]);
                }
            }
        }
        let mut connectivity = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if i + 1 < n {
                        connectivity.extend([idx(i, j, k), idx(i + 1, j, k)]);
                    }
                    if j + 1 < n {
                        connectivity.extend([idx(i, j, k), idx(i, j + 1, k)]);
                    }
                    if k + 1 < n {
                        connectivity.extend([idx(i, j, k), idx(i, j, k + 1)]);
                    }
                }
            }
        }
        let elements = connectivity.len() / 2;
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions,
            connectivity,
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![0.05; elements],
            }],
        }
    }

    /// The Schwarz-preconditioned solve must land on the same solution as
    /// the default preconditioner: the preconditioner changes the CG path,
    /// never the converged answer — including through contact activation.
    #[cfg(feature = "parallel")]
    #[test]
    fn schwarz_solve_matches_default() {
        let mut mesh = cubic_lattice(8);
        // Soft/stiff contrast so the preconditioner actually works for a
        // living: scale field ramping 1e-3..1 along x.
        let scales: Vec<f64> = (0..mesh.connectivity.len() / 2)
            .map(|e| {
                let n0 = mesh.connectivity[e * 2] as usize;
                let x = mesh.node_positions[n0 * 3];
                1e-3_f64.powf(1.0 - x / 7.0)
            })
            .collect();
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: scales,
        });
        // Plate pressing down from above onto the 0..7 cube.
        let mut plate = |p: [f64; 3]| p[2] > 6.3;

        let mut config = SolveConfig {
            fixed_boundary: FixedBoundary::ZMin,
            ..Default::default()
        };
        let base = solve(&mesh, &mut plate, &config).unwrap();
        for direct_local in [true, false] {
            config.preconditioner = crate::PrecondChoice::Schwarz(crate::SchwarzParams {
                target_nodes: 48,
                direct_local,
            });
            let schwarz = solve(&mesh, &mut plate, &config).unwrap();

            assert!(base.stats.converged && schwarz.stats.converged);
            let scale = base
                .displacement
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs()));
            for (i, (a, b)) in base
                .displacement
                .iter()
                .zip(&schwarz.displacement)
                .enumerate()
            {
                assert!(
                    (a - b).abs() <= scale * 1e-5,
                    "dof {i} (direct_local {direct_local}): default {a} vs schwarz {b}"
                );
            }
            assert!(
                schwarz.stats.cg_iterations < base.stats.cg_iterations,
                "schwarz (direct_local {direct_local}) {} vs default {} CG iterations",
                schwarz.stats.cg_iterations,
                base.stats.cg_iterations
            );
        }
    }

    /// Not a correctness test: forward contact solves on contrast-ramped
    /// cubic lattices of growing size, block-Jacobi vs two-level Schwarz —
    /// the scalability question is whether Schwarz iteration counts stay
    /// flat as the mesh grows while the subdomain solves ride the thread
    /// count. Run with RAYON_NUM_THREADS=<t> and:
    ///   cargo test -p fea_core --features parallel --release -- \
    ///       --ignored --nocapture schwarz_scaling_bench
    #[cfg(feature = "parallel")]
    #[test]
    #[ignore]
    fn schwarz_scaling_bench() {
        // Override lattice sizes with SCHWARZ_BENCH_N=12,22,32 when probing
        // larger meshes or isolating one size for thread-scaling runs;
        // SCHWARZ_BENCH_TARGET sets the subdomain size (6 coarse dofs per
        // subdomain, so this trades local against coarse solve cost);
        // SCHWARZ_BENCH_SKIP_BJ=1 skips the block-Jacobi baseline.
        let sizes: Vec<usize> = std::env::var("SCHWARZ_BENCH_N")
            .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![12, 22, 32]);
        let target_nodes: usize = std::env::var("SCHWARZ_BENCH_TARGET")
            .map(|s| s.parse().unwrap())
            .unwrap_or(128);
        let skip_bj = std::env::var("SCHWARZ_BENCH_SKIP_BJ").is_ok();
        for n in sizes {
            let mut mesh = cubic_lattice(n);
            let scales: Vec<f64> = (0..mesh.connectivity.len() / 2)
                .map(|e| {
                    let n0 = mesh.connectivity[e * 2] as usize;
                    let x = mesh.node_positions[n0 * 3];
                    1e-3_f64.powf(1.0 - x / (n - 1) as f64)
                })
                .collect();
            mesh.element_fields.push(FeaField {
                name: "stiffness_scale".to_string(),
                components: 1,
                data: scales,
            });
            let top = (n - 1) as f64 - 0.7;

            let configs = [
                ("block-jacobi", crate::PrecondChoice::Auto),
                (
                    "schwarz",
                    crate::PrecondChoice::Schwarz(crate::SchwarzParams {
                        target_nodes,
                        direct_local: true,
                    }),
                ),
            ];
            for &(label, preconditioner) in configs.iter().skip(if skip_bj { 1 } else { 0 }) {
                let mut plate = |p: [f64; 3]| p[2] > top;
                let config = SolveConfig {
                    fixed_boundary: FixedBoundary::ZMin,
                    cg_tolerance: 1e-6,
                    preconditioner,
                    ..Default::default()
                };
                let timer = std::time::Instant::now();
                let result = solve(&mesh, &mut plate, &config).unwrap();
                println!(
                    "n={n} ({} struts): {label}[{target_nodes}] {:.2}s, {} cg iters, converged={}",
                    mesh.connectivity.len() / 2,
                    timer.elapsed().as_secs_f64(),
                    result.stats.cg_iterations,
                    result.stats.converged,
                );
                // Phase timer dump (see schwarz.rs timers)
                use std::sync::atomic::Ordering::Relaxed;
                let take = |t: &std::sync::atomic::AtomicU64| t.swap(0, Relaxed) as f64 / 1e9;
                println!(
                    "  phases: build_local {:.2}s, build_coarse {:.2}s, local {:.2}s, scatter {:.2}s, coarse {:.2}s",
                    take(&crate::schwarz::T_BUILD_LOCAL),
                    take(&crate::schwarz::T_BUILD_COARSE),
                    take(&crate::schwarz::T_LOCAL),
                    take(&crate::schwarz::T_SCATTER),
                    take(&crate::schwarz::T_COARSE),
                );
            }
        }
    }

    /// Not a correctness test: prints serial vs parallel apply timings over
    /// lattice sizes so PARALLEL_MIN_STRUTS can be set from data. Run with
    ///   cargo test -p fea_core --features parallel --release -- \
    ///       --ignored --nocapture apply_scaling_bench
    #[cfg(feature = "parallel")]
    #[test]
    #[ignore]
    fn apply_scaling_bench() {
        for n in [10, 22, 47, 100] {
            let mesh = cubic_lattice(n);
            let model = FrameModel::new(&mesh, material()).unwrap();
            let dofs = mesh.node_count() * 6;
            let x: Vec<f64> = (0..dofs).map(|i| (i as f64 * 0.37).sin()).collect();
            let mut y = vec![0.0; dofs];

            let struts = model.struts.len();
            let reps = (2_000_000 / struts).max(3);
            let timer = std::time::Instant::now();
            for _ in 0..reps {
                model.apply_serial(&x, &mut y);
            }
            let serial = timer.elapsed().as_secs_f64() / reps as f64;
            let timer = std::time::Instant::now();
            for _ in 0..reps {
                model.apply_parallel(&x, &mut y);
            }
            let parallel = timer.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{struts} struts: serial {:.1}us, parallel {:.1}us, speedup {:.2}x",
                serial * 1e6,
                parallel * 1e6,
                serial / parallel
            );
        }
    }

    /// Parallel and serial applies must agree (up to float addition order)
    /// on a mesh large enough to take the parallel path for real.
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_apply_matches_serial() {
        // A skew chain gives fully general rotation frames; > the
        // PARALLEL_MIN_STRUTS threshold so StiffnessModel::apply dispatches
        // to the parallel path.
        let segments = PARALLEL_MIN_STRUTS + 500;
        let mesh = chain([0.0; 3], [1.0, 2.0, 3.0], 10.0, segments, 0.05);
        let model = FrameModel::new(&mesh, material()).unwrap();

        let n = mesh.node_count() * 6;
        // Deterministic pseudo-random displacements (LCG).
        let mut state = 0x2545F4914F6CDD1Du64;
        let x: Vec<f64> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();

        let mut serial = vec![0.0; n];
        model.apply_serial(&x, &mut serial);
        let mut parallel = vec![0.0; n];
        model.apply(&x, &mut parallel);

        let scale = serial.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (s, p)) in serial.iter().zip(&parallel).enumerate() {
            assert!(
                (s - p).abs() <= scale * 1e-12,
                "dof {i}: serial {s} vs parallel {p}"
            );
        }
    }

    /// Dirichlet-drive a frame model: constrain `fixed` dofs to the given
    /// values, CG the rest, and return (u, K u).
    fn dirichlet_solve(mesh: &FeaMesh, fixed: &[(usize, f64)]) -> (Vec<f64>, Vec<f64>) {
        let model = FrameModel::new(mesh, material()).unwrap();
        let n = mesh.node_count() * 6;
        let mut constrained = vec![false; n];
        let mut u = vec![0.0; n];
        for &(dof, value) in fixed {
            constrained[dof] = true;
            u[dof] = value;
        }
        let precond = crate::Precond::build(
            &model.diagonal(),
            model.node_blocks().as_deref(),
            6,
            &constrained,
        );
        let (_, converged) = solve_cg(&model, &constrained, &precond, &mut u, 1e-12, 50_000);
        assert!(converged, "CG failed to converge");
        let mut forces = vec![0.0; n];
        model.apply(&u, &mut forces);
        (u, forces)
    }

    /// Constrain all 6 dofs of a node to zero.
    fn glue(node: usize) -> Vec<(usize, f64)> {
        (0..6).map(|c| (node * 6 + c, 0.0)).collect()
    }

    /// node_blocks must reproduce the node-diagonal 6x6 blocks of the
    /// assembled K, probed column-by-column out of the ground-truth apply.
    #[test]
    fn node_blocks_match_probed_stiffness() {
        // Skew chain: general rotation frames, shared interior nodes.
        let mesh = chain([0.2, -0.1, 0.4], [1.0, 2.0, 3.0], 2.0, 5, 0.05);
        let model = FrameModel::new(&mesh, material()).unwrap();
        let blocks = model.node_blocks().unwrap();

        let n = mesh.node_count() * 6;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for node in 0..mesh.node_count() {
            for col in 0..6 {
                x[node * 6 + col] = 1.0;
                model.apply(&x, &mut y);
                x[node * 6 + col] = 0.0;
                for row in 0..6 {
                    let expected = y[node * 6 + row];
                    let got = blocks[node * 36 + row * 6 + col];
                    assert!(
                        (expected - got).abs() <= expected.abs().max(1.0) * 1e-12,
                        "node {node} block[{row}][{col}]: probed {expected} vs {got}"
                    );
                }
            }
        }
    }

    /// The geometric stiffness added by `update_prestress` must be the
    /// consistent beam-column matrix: stretch a single strut axially so
    /// N = k_ax * elongation, and check element_stiffness gains exactly
    /// the analytic (6N/5L, N/10, 2NL/15, -NL/30) bending blocks — and
    /// nothing anywhere else. Compression must add nothing (tension-only).
    #[test]
    fn prestress_adds_the_consistent_geometric_matrix() {
        let length = 0.7;
        let mesh = chain([0.0; 3], [1.0, 0.0, 0.0], length, 1, 0.05);
        let mut model = FrameModel::new(&mesh, material()).unwrap();
        let base = model.element_stiffness(0);

        // Stretch: node 1 moves +x by delta -> N = k_ax * delta.
        let delta = 0.01;
        let mut u = vec![0.0f64; 2 * 6];
        u[6] = delta;
        model.update_prestress(&u);
        let n = model.struts[0].k_ax * delta;
        let stiffened = model.element_stiffness(0);

        // The strut lies along global x, so local axes == global axes and
        // the geometric terms sit in the same slots as the elastic bending
        // blocks: (u_y, th_z) and (u_z, th_y) per end.
        let (g1, g2, g3, g4) = (
            6.0 * n / (5.0 * length),
            n / 10.0,
            2.0 * n * length / 15.0,
            -n * length / 30.0,
        );
        // Expected delta, built with the same sign structure as the
        // elastic kernel (probe a strut with only geometric terms).
        let probe = Strut {
            nodes: model.struts[0].nodes,
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            length,
            k_ax: 0.0,
            k_tor: 0.0,
            b1: 0.0,
            b2: 0.0,
            b3: 0.0,
            b4: 0.0,
            g1,
            g2,
            g3,
            g4,
            volume: model.struts[0].volume,
        };
        let mut expect = [[0.0f64; 12]; 12];
        for col in 0..12 {
            let mut v = [[0.0f64; 3]; 4];
            v[col / 3][col % 3] = 1.0;
            let f = local_forces(&probe, &v);
            for row in 0..12 {
                expect[row][col] = f[row / 3][row % 3];
            }
        }
        for row in 0..12 {
            for col in 0..12 {
                let got = stiffened[row][col] - base[row][col];
                assert!(
                    (got - expect[row][col]).abs() < 1e-9 * n,
                    "({row},{col}): geometric delta {got} vs {}",
                    expect[row][col]
                );
            }
        }

        // Compression adds nothing.
        u[6] = -delta;
        model.update_prestress(&u);
        let compressed = model.element_stiffness(0);
        for row in 0..12 {
            for col in 0..12 {
                assert!((compressed[row][col] - base[row][col]).abs() < 1e-12);
            }
        }
    }

    /// The behavioral point of stress stiffening under displacement-driven
    /// contact: a sphere pressed into a slender-strut lattice sags the
    /// surface, the sagging laterals go taut, and the taut membrane resists
    /// the DIFFERENTIAL sag the sphere prescribes — so total contact force
    /// rises, and the share carried where penetration is deepest (the
    /// center) rises with it while the shallow rim sheds. (The "membrane
    /// spreads the load" intuition belongs to force-controlled pressing;
    /// with a prescribed pose, holding the deepest nodes on the surface is
    /// exactly what gets more expensive. Measured: total +30%, center share
    /// 0.569 -> 0.643 with one pass, patch unchanged at 21 nodes.)
    #[test]
    fn stress_stiffening_adds_membrane_resistance() {
        let mesh = cubic_lattice(7); // coords 0..6, very slender struts
        let mut sphere = |p: [f64; 3]| {
            let d = [p[0] - 3.0, p[1] - 3.0, p[2] - 8.8];
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 16.0
        };
        let mut press = |passes: usize| {
            let config = SolveConfig {
                material: material(),
                fixed_boundary: FixedBoundary::ZMin,
                stress_stiffening_passes: passes,
                ..Default::default()
            };
            let result = solve(&mesh, &mut sphere, &config).unwrap();
            assert!(result.stats.converged, "passes={passes} did not converge");
            let mut center = 0.0;
            let mut total = 0.0;
            for node in 0..mesh.node_count() {
                let f = -result.contact_force[node * 3 + 2];
                if f <= 0.0 {
                    continue;
                }
                let p = mesh.node_position(node);
                let r2 = (p[0] - 3.0).powi(2) + (p[1] - 3.0).powi(2);
                total += f;
                if r2 < 1.5 * 1.5 {
                    center += f;
                }
            }
            assert!(total > 0.0);
            (total, center / total)
        };
        let (linear_total, linear_share) = press(0);
        let (stiff_total, stiff_share) = press(1);
        assert!(
            stiff_total > 1.15 * linear_total,
            "membrane resistance missing: total {linear_total:.4} -> {stiff_total:.4}"
        );
        assert!(
            stiff_share > 1.05 * linear_share,
            "deep-penetration concentration missing: center share \
             {linear_share:.4} -> {stiff_share:.4}"
        );
    }

    /// Rigid translations must stay zero-force under prestress — the
    /// geometric matrix has the same translation null space as the elastic
    /// one.
    #[test]
    fn prestressed_rigid_translation_is_zero_force() {
        let mesh = chain([0.2, -0.1, 0.4], [1.0, 2.0, -0.5], 1.3, 4, 0.03);
        let mut model = FrameModel::new(&mesh, material()).unwrap();
        // Stretch the whole chain along its axis to induce tension.
        let axis = normalize([1.0, 2.0, -0.5]);
        let mut u = vec![0.0f64; mesh.node_count() * 6];
        for node in 0..mesh.node_count() {
            let p = mesh.node_position(node);
            let t = p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2];
            for c in 0..3 {
                u[node * 6 + c] = 0.02 * t * axis[c];
            }
        }
        model.update_prestress(&u);
        assert!(model.struts.iter().all(|s| s.g1 > 0.0), "no tension built");

        let mut x = vec![0.0f64; mesh.node_count() * 6];
        for node in 0..mesh.node_count() {
            x[node * 6] = 0.3;
            x[node * 6 + 1] = -0.7;
            x[node * 6 + 2] = 0.1;
        }
        let mut y = vec![0.0f64; x.len()];
        model.apply(&x, &mut y);
        let peak = model.struts.iter().map(|s| s.g1).fold(0.0f64, f64::max);
        assert!(
            y.iter().all(|v| v.abs() < 1e-12 * peak.max(1.0)),
            "rigid translation produces force under prestress"
        );
    }

    /// `strut_end_forces` must equal `element_stiffness . u_e` under the
    /// documented dof order — the adjoint gradient assembly leans on both
    /// the values and the ordering.
    #[test]
    fn strut_end_forces_match_element_stiffness() {
        let mesh = cubic_lattice(2);
        let model = FrameModel::new(&mesh, material()).unwrap();
        let n = mesh.node_count() * 6;
        let x: Vec<f64> = (0..n).map(|i| ((i * 13 % 17) as f64 - 8.0) / 8.0).collect();
        for e in (0..model.strut_count()).step_by(3) {
            let q = model.strut_end_forces(e, &x);
            let ke = model.element_stiffness(e);
            let nodes = model.strut_nodes(e);
            let mut xe = [0.0f64; 12];
            for block in 0..4 {
                let off = nodes[block / 2] as usize * 6 + (block % 2) * 3;
                xe[block * 3..block * 3 + 3].copy_from_slice(&x[off..off + 3]);
            }
            for row in 0..12 {
                let expect: f64 = (0..12).map(|col| ke[row][col] * xe[col]).sum();
                assert!(
                    (q[row] - expect).abs() < 1e-12 * expect.abs().max(1.0),
                    "strut {e} row {row}: {} vs {expect}",
                    q[row]
                );
            }
        }
    }

    #[test]
    fn assembled_stiffness_is_symmetric() {
        // A three-strut Y in general position.
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: vec![
                0.0, 0.0, 0.0, //
                1.0, 0.2, -0.1, //
                0.3, 1.1, 0.4, //
                -0.2, 0.3, 0.9,
            ],
            connectivity: vec![0, 1, 0, 2, 0, 3],
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![0.05, 0.08, 0.03],
            }],
        };
        let model = FrameModel::new(&mesh, material()).unwrap();
        let n = mesh.node_count() * 6;
        let mut k = vec![vec![0.0; n]; n];
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            let mut y = vec![0.0; n];
            model.apply(&x, &mut y);
            k[i] = y;
            x[i] = 0.0;
        }
        let scale = k
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0f64, |a, &b| a.max(b.abs()));
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[i][j] - k[j][i]).abs() <= scale * 1e-12,
                    "K[{i}][{j}] = {} but K[{j}][{i}] = {}",
                    k[i][j],
                    k[j][i]
                );
            }
        }
        // The Jacobi diagonal must match the assembled diagonal.
        let diag = model.diagonal();
        for i in 0..n {
            assert!(
                (diag[i] - k[i][i]).abs() <= scale * 1e-12,
                "diagonal[{i}] = {} but K[{i}][{i}] = {}",
                diag[i],
                k[i][i]
            );
        }
    }

    #[test]
    fn rigid_body_motions_are_zero_energy() {
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: vec![
                0.1, -0.2, 0.3, //
                1.2, 0.4, -0.3, //
                0.7, 1.3, 0.8, //
                -0.4, 0.6, 1.1,
            ],
            connectivity: vec![0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![0.05; 5],
            }],
        };
        let model = FrameModel::new(&mesh, material()).unwrap();
        let n = mesh.node_count() * 6;
        // Reference stiffness magnitude for the zero threshold.
        let diag_max = model.diagonal().iter().fold(0.0f64, |a, &b| a.max(b));

        // Small rigid motion: translation t plus rotation omega about the
        // origin (u = t + omega x p, theta = omega).
        let t = [0.3, -0.7, 0.2];
        let omega = [0.4, 0.1, -0.6];
        let mut u = vec![0.0; n];
        for node in 0..mesh.node_count() {
            let p = mesh.node_position(node);
            u[node * 6] = t[0] + omega[1] * p[2] - omega[2] * p[1];
            u[node * 6 + 1] = t[1] + omega[2] * p[0] - omega[0] * p[2];
            u[node * 6 + 2] = t[2] + omega[0] * p[1] - omega[1] * p[0];
            u[node * 6 + 3] = omega[0];
            u[node * 6 + 4] = omega[1];
            u[node * 6 + 5] = omega[2];
        }
        let mut y = vec![0.0; n];
        model.apply(&u, &mut y);
        for (i, v) in y.iter().enumerate() {
            assert!(
                v.abs() < diag_max * 1e-12,
                "rigid motion produced force {v} at dof {i}"
            );
        }
    }

    #[test]
    fn cantilever_matches_euler_bernoulli() {
        let (length, r, delta) = (1.0, 0.05, 0.01);
        let segments = 4;
        let mesh = chain([0.0; 3], [1.0, 0.0, 0.0], length, segments, r);
        let tip = segments;

        // Glue the root, prescribe the tip's transverse (y) deflection.
        let mut fixed = glue(0);
        fixed.push((tip * 6 + 1, delta));
        let (u, forces) = dirichlet_solve(&mesh, &fixed);

        // Tip point load: F = 3 E I delta / L^3.
        let inertia = std::f64::consts::PI * r.powi(4) / 4.0;
        let expected = 3.0 * E * inertia * delta / length.powi(3);
        let tip_force = forces[tip * 6 + 1];
        assert!(
            (tip_force - expected).abs() < expected * 1e-9,
            "tip reaction {tip_force}, expected {expected}"
        );

        // Interior deflection follows the cubic y(x) = delta (3 (x/L)^2
        // - (x/L)^3) / 2 for a tip point load.
        for i in 1..segments {
            let x = i as f64 / segments as f64;
            let expected_y = delta * (3.0 * x * x - x * x * x) / 2.0;
            let y = u[i * 6 + 1];
            assert!(
                (y - expected_y).abs() < delta * 1e-9,
                "node {i} deflection {y}, expected {expected_y}"
            );
        }
    }

    #[test]
    fn axial_and_torsional_stiffness_match_closed_form() {
        let (length, r) = (2.0, 0.1);
        let segments = 3;
        let mesh = chain([0.0; 3], [1.0, 0.0, 0.0], length, segments, r);
        let tip = segments;
        let area = std::f64::consts::PI * r * r;
        let inertia = std::f64::consts::PI * r.powi(4) / 4.0;
        let g_modulus = E / (2.0 * (1.0 + NU));

        // Axial: F = E A delta / L.
        let delta = 0.02;
        let mut fixed = glue(0);
        fixed.push((tip * 6, delta));
        let (_, forces) = dirichlet_solve(&mesh, &fixed);
        let expected = E * area * delta / length;
        assert!(
            (forces[tip * 6] - expected).abs() < expected * 1e-9,
            "axial reaction {}, expected {expected}",
            forces[tip * 6]
        );

        // Torsion: M = G J theta / L.
        let theta = 0.05;
        let mut fixed = glue(0);
        fixed.push((tip * 6 + 3, theta));
        let (_, forces) = dirichlet_solve(&mesh, &fixed);
        let expected = g_modulus * (2.0 * inertia) * theta / length;
        assert!(
            (forces[tip * 6 + 3] - expected).abs() < expected * 1e-9,
            "torsion reaction {}, expected {expected}",
            forces[tip * 6 + 3]
        );
    }

    #[test]
    fn cantilever_stiffness_is_orientation_invariant() {
        // The same cantilever along a skew axis, tip pressed along a
        // perpendicular direction: the reaction magnitude must match the
        // axis-aligned answer (circular section, isotropic bending).
        let (length, r, delta) = (1.0, 0.05, 0.01);
        let segments = 4;
        let axis = [1.0, 1.0, 1.0];
        let mesh = chain([0.0; 3], axis, length, segments, r);
        let tip = segments;

        // A unit vector perpendicular to the axis.
        let t = normalize(cross(axis, [0.0, 0.0, 1.0]));
        let mut fixed = glue(0);
        for c in 0..3 {
            fixed.push((tip * 6 + c, delta * t[c]));
        }
        let (_, forces) = dirichlet_solve(&mesh, &fixed);
        let magnitude = (0..3)
            .map(|c| forces[tip * 6 + c] * forces[tip * 6 + c])
            .sum::<f64>()
            .sqrt();
        let inertia = std::f64::consts::PI * r.powi(4) / 4.0;
        let expected = 3.0 * E * inertia * delta / length.powi(3);
        assert!(
            (magnitude - expected).abs() < expected * 1e-9,
            "skew cantilever reaction {magnitude}, expected {expected}"
        );
        // No axial component: the reaction is perpendicular to the strut.
        let n = normalize(axis);
        let axial: f64 = (0..3).map(|c| forces[tip * 6 + c] * n[c]).sum();
        assert!(axial.abs() < expected * 1e-9, "axial leakage {axial}");
    }

    /// A rigid half-space `z > level` (a flat plate pressed straight down).
    fn plate(level: f64) -> impl FnMut([f64; 3]) -> bool {
        move |p: [f64; 3]| p[2] > level
    }

    fn solve_config() -> SolveConfig {
        SolveConfig {
            material: material(),
            fixed_boundary: FixedBoundary::ZMin,
            ..Default::default()
        }
    }

    #[test]
    fn strut_column_compression_matches_axial_stiffness() {
        // A vertical 4-segment column, glued at z=0, plate pressing the top
        // node down by 0.1: pure axial compression, F = E A delta / L.
        let (length, r, dip) = (1.0, 0.05, 0.1);
        let segments = 4;
        let mesh = chain([0.0; 3], [0.0, 0.0, 1.0], length, segments, r);
        let result = solve(&mesh, &mut plate(length - dip), &solve_config()).unwrap();
        assert!(result.stats.converged, "stats: {:?}", result.stats);
        assert_eq!(result.stats.active_contacts, 1);

        // Displacement is linear along the column; rotations are zero.
        let rotation = result.rotation.as_ref().expect("frames report rotations");
        for i in 0..=segments {
            let z = length * i as f64 / segments as f64;
            let expected_uz = -dip * z / length;
            assert!(
                (result.displacement[i * 3 + 2] - expected_uz).abs() < 1e-9,
                "node {i} u_z {}, expected {expected_uz}",
                result.displacement[i * 3 + 2]
            );
            for c in 0..3 {
                assert!(rotation[i * 3 + c].abs() < 1e-9, "node {i} rotated");
            }
        }

        let area = std::f64::consts::PI * r * r;
        let expected_force = -E * area * dip / length;
        let tip_force = result.contact_force[segments * 3 + 2];
        assert!(
            (tip_force - expected_force).abs() < expected_force.abs() * 1e-6,
            "contact force {tip_force}, expected {expected_force}"
        );

        // Uniform axial strain energy density: E eps^2 / 2 in every strut.
        let eps = dip / length;
        for (e, density) in result.strain_energy_density.iter().enumerate() {
            let expected = 0.5 * E * eps * eps;
            assert!(
                (density - expected).abs() < expected * 1e-6,
                "strut {e} energy density {density}, expected {expected}"
            );
        }
    }

    #[test]
    fn cube_frame_carries_shear_without_a_mechanism() {
        // A unit cube of 12 edge struts: as a pin-jointed truss this is a
        // mechanism; frame bending stiffness must carry it. Glued at zmin,
        // sphere pressing the top face off-center (asymmetric load).
        let corners: [[f64; 3]; 8] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let edges: [[u32; 2]; 12] = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: corners.iter().flatten().copied().collect(),
            connectivity: edges.iter().flatten().copied().collect(),
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![0.05; 12],
            }],
        };
        // Centered near the (0, 0, 1) corner so a top node actually
        // penetrates (the mesh only has corner nodes to contact).
        let center = [0.1, 0.2, 0.9 + 1.0];
        let mut sphere =
            move |p: [f64; 3]| (0..3).map(|i| (p[i] - center[i]).powi(2)).sum::<f64>() < 1.0;
        let result = solve(&mesh, &mut sphere, &solve_config()).unwrap();
        assert!(result.stats.converged, "stats: {:?}", result.stats);
        assert!(result.stats.active_contacts > 0);
        // Load reaches the plate through the frame: net downward force.
        let total_fz: f64 = (0..8).map(|n| result.contact_force[n * 3 + 2]).sum();
        assert!(total_fz < 0.0, "net contact force must press down");
        // The lateral (bending) response is finite and nonzero — the
        // off-center press shears the frame.
        let max_lateral = (0..8)
            .map(|n| result.displacement[n * 3].abs().max(result.displacement[n * 3 + 1].abs()))
            .fold(0.0f64, f64::max);
        assert!(
            max_lateral.is_finite() && max_lateral > 1e-9,
            "expected lateral frame response, got {max_lateral}"
        );
    }

    #[test]
    fn stiffness_scale_multiplies_strut_forces() {
        let (length, r, dip) = (1.0, 0.05, 0.1);
        let mut mesh = chain([0.0; 3], [0.0, 0.0, 1.0], length, 4, r);
        let base = solve(&mesh, &mut plate(length - dip), &solve_config()).unwrap();
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![2.0; 4],
        });
        let scaled = solve(&mesh, &mut plate(length - dip), &solve_config()).unwrap();
        let ratio = scaled.contact_force[4 * 3 + 2] / base.contact_force[4 * 3 + 2];
        assert!(
            (ratio - 2.0).abs() < 1e-9,
            "doubled stiffness_scale should double the force, ratio {ratio}"
        );
    }

    #[test]
    fn bar2_meshes_require_a_radius_field() {
        let mut mesh = chain([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2, 0.05);
        mesh.element_fields.clear();
        let err = solve(&mesh, &mut plate(0.9), &solve_config()).unwrap_err();
        assert!(err.contains("radius"), "unexpected error: {err}");
    }

    #[test]
    fn invalid_radii_are_rejected() {
        for bad in [0.0, -0.1, f64::NAN] {
            let mut mesh = chain([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2, 0.05);
            mesh.element_fields[0].data[1] = bad;
            let err = solve(&mesh, &mut plate(0.9), &solve_config()).unwrap_err();
            assert!(err.contains("radius"), "unexpected error for {bad}: {err}");
        }
    }
}
