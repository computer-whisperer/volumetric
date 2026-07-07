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
    /// Cross-section volume `A * L` (for energy density).
    volume: f64,
}

/// The assembled frame stiffness of a Bar2 mesh.
pub(crate) struct FrameModel {
    struts: Vec<Strut>,
    node_count: usize,
    /// Mean strut length: the contact scan step / tolerance scale.
    mean_length: f64,
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
                k_ax: e_modulus * scale * area / length,
                k_tor: g_modulus * scale * (2.0 * inertia) / length,
                b1: 12.0 * ei / length.powi(3),
                b2: 6.0 * ei / (length * length),
                b3: 4.0 * ei / length,
                b4: 2.0 * ei / length,
                volume: area * length,
            });
        }

        Ok(Self {
            struts,
            node_count: mesh.node_count(),
            mean_length: total_length / mesh.element_count() as f64,
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

    // Bending in the x'-y' plane: (u_y, theta_z).
    f1[1] = s.b1 * u1[1] + s.b2 * th1[2] - s.b1 * u2[1] + s.b2 * th2[2];
    m1[2] = s.b2 * u1[1] + s.b3 * th1[2] - s.b2 * u2[1] + s.b4 * th2[2];
    f2[1] = -f1[1];
    m2[2] = s.b2 * u1[1] + s.b4 * th1[2] - s.b2 * u2[1] + s.b3 * th2[2];

    // Bending in the x'-z' plane: (u_z, theta_y), opposite coupling signs.
    f1[2] = s.b1 * u1[2] - s.b2 * th1[1] - s.b1 * u2[2] - s.b2 * th2[1];
    m1[1] = -s.b2 * u1[2] + s.b3 * th1[1] + s.b2 * u2[2] + s.b4 * th2[1];
    f2[2] = -f1[2];
    m2[1] = -s.b2 * u1[2] + s.b4 * th1[1] + s.b2 * u2[2] + s.b3 * th2[1];

    [f1, m1, f2, m2]
}

impl FrameModel {
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

impl StiffnessModel for FrameModel {
    fn dofs_per_node(&self) -> usize {
        6
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
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

    fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0f64; self.node_count * 6];
        for s in &self.struts {
            // The local per-node blocks are diagonal — translations
            // (k_ax, b1, b1), rotations (k_tor, b3, b3) — so the global
            // diagonal of R^T D R is a weighted sum of squared cosines.
            let translation = [s.k_ax, s.b1, s.b1];
            let rotation = [s.k_tor, s.b3, s.b3];
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
        let diag = model.diagonal();
        let (_, converged) = solve_cg(&model, &constrained, &diag, &mut u, 1e-12, 50_000);
        assert!(converged, "CG failed to converge");
        let mut forces = vec![0.0; n];
        model.apply(&u, &mut forces);
        (u, forces)
    }

    /// Constrain all 6 dofs of a node to zero.
    fn glue(node: usize) -> Vec<(usize, f64)> {
        (0..6).map(|c| (node * 6 + c, 0.0)).collect()
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
