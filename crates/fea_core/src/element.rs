//! Isoparametric hex8 element stiffness.
//!
//! Standard trilinear hexahedron in VTK node ordering (see
//! `volumetric_abi::fea`), 2x2x2 Gauss quadrature, isotropic linear
//! elasticity. Voigt strain order: [xx, yy, zz, xy, yz, zx].

/// Natural-coordinate sign of each node (VTK ordering).
const NODE_SIGNS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// A 24x24 element stiffness matrix (3 dofs per node, node-major).
pub type ElementStiffness = [[f64; 24]; 24];

/// Isotropic material: Young's modulus and Poisson's ratio.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub youngs_modulus: f64,
    pub poissons_ratio: f64,
}

impl Material {
    /// Lamé parameters (lambda, mu).
    fn lame(self) -> (f64, f64) {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        (lambda, mu)
    }
}

/// Shape-function derivatives w.r.t. natural coordinates at `xi`, per node.
fn shape_gradients_natural(xi: [f64; 3]) -> [[f64; 3]; 8] {
    let mut out = [[0.0; 3]; 8];
    for (node, s) in NODE_SIGNS.iter().enumerate() {
        let f = [1.0 + s[0] * xi[0], 1.0 + s[1] * xi[1], 1.0 + s[2] * xi[2]];
        out[node] = [
            s[0] * f[1] * f[2] / 8.0,
            f[0] * s[1] * f[2] / 8.0,
            f[0] * f[1] * s[2] / 8.0,
        ];
    }
    out
}

fn invert3(m: [[f64; 3]; 3]) -> Option<([[f64; 3]; 3], f64)> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < f64::MIN_POSITIVE * 1e10 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut inv = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = ((i + 1) % 3, (i + 2) % 3);
            let (c, d) = ((j + 1) % 3, (j + 2) % 3);
            // Cofactor transpose (adjugate).
            inv[j][i] = (m[a][c] * m[b][d] - m[a][d] * m[b][c]) * inv_det;
        }
    }
    Some((inv, det))
}

/// The element stiffness matrix for a hex8 with the given node coordinates
/// (VTK order). Fails on degenerate geometry (non-positive Jacobian).
pub fn hex8_stiffness(
    coords: &[[f64; 3]; 8],
    material: Material,
) -> Result<ElementStiffness, String> {
    let (lambda, mu) = material.lame();
    // D in Voigt form: normal block lambda + 2mu diag, shear diag mu.
    let mut d = [[0.0f64; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = lambda;
        }
        d[i][i] = lambda + 2.0 * mu;
        d[i + 3][i + 3] = mu;
    }

    let g = 1.0 / 3.0f64.sqrt();
    let mut k = [[0.0f64; 24]; 24];

    for gauss in 0..8 {
        let xi = [
            g * NODE_SIGNS[gauss][0],
            g * NODE_SIGNS[gauss][1],
            g * NODE_SIGNS[gauss][2],
        ];
        let dn_dxi = shape_gradients_natural(xi);

        // Jacobian J[a][b] = d x_b / d xi_a.
        let mut jac = [[0.0f64; 3]; 3];
        for node in 0..8 {
            for a in 0..3 {
                for b in 0..3 {
                    jac[a][b] += dn_dxi[node][a] * coords[node][b];
                }
            }
        }
        let Some((jac_inv, det)) = invert3(jac) else {
            return Err("degenerate element (singular Jacobian)".to_string());
        };
        if det <= 0.0 {
            return Err(format!("inverted element (Jacobian determinant {det})"));
        }

        // dN/dx = J^-1 dN/dxi.
        let mut dn_dx = [[0.0f64; 3]; 8];
        for node in 0..8 {
            for a in 0..3 {
                for b in 0..3 {
                    dn_dx[node][a] += jac_inv[a][b] * dn_dxi[node][b];
                }
            }
        }

        // B (6x24): strain from nodal displacements.
        let mut b = [[0.0f64; 24]; 6];
        for node in 0..8 {
            let [dx, dy, dz] = dn_dx[node];
            let col = node * 3;
            b[0][col] = dx;
            b[1][col + 1] = dy;
            b[2][col + 2] = dz;
            b[3][col] = dy;
            b[3][col + 1] = dx;
            b[4][col + 1] = dz;
            b[4][col + 2] = dy;
            b[5][col] = dz;
            b[5][col + 2] = dx;
        }

        // K += B^T D B * detJ (unit Gauss weights for 2-point rule).
        let mut db = [[0.0f64; 24]; 6];
        for i in 0..6 {
            for j in 0..24 {
                for l in 0..6 {
                    db[i][j] += d[i][l] * b[l][j];
                }
            }
        }
        for i in 0..24 {
            for j in 0..24 {
                let mut sum = 0.0;
                for l in 0..6 {
                    sum += b[l][i] * db[l][j];
                }
                k[i][j] += sum * det;
            }
        }
    }

    Ok(k)
}

/// The stiffness of an axis-aligned cube element with edge length `h`.
pub fn cube_stiffness(h: f64, material: Material) -> Result<ElementStiffness, String> {
    let coords = std::array::from_fn(|node| {
        [
            (NODE_SIGNS[node][0] + 1.0) / 2.0 * h,
            (NODE_SIGNS[node][1] + 1.0) / 2.0 * h,
            (NODE_SIGNS[node][2] + 1.0) / 2.0 * h,
        ]
    });
    hex8_stiffness(&coords, material)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATERIAL: Material = Material {
        youngs_modulus: 2.5,
        poissons_ratio: 0.3,
    };

    #[test]
    fn stiffness_is_symmetric_with_zero_rigid_body_modes() {
        let k = cube_stiffness(0.5, MATERIAL).unwrap();

        for i in 0..24 {
            for j in 0..24 {
                assert!((k[i][j] - k[j][i]).abs() < 1e-12, "asymmetry at ({i},{j})");
            }
        }

        // Uniform translation along each axis produces zero force.
        for axis in 0..3 {
            let mut u = [0.0f64; 24];
            for node in 0..8 {
                u[node * 3 + axis] = 1.0;
            }
            for (i, row) in k.iter().enumerate() {
                let f: f64 = row.iter().zip(&u).map(|(a, b)| a * b).sum();
                assert!(
                    f.abs() < 1e-10,
                    "translation {axis} residual {f} at dof {i}"
                );
            }
        }
    }

    #[test]
    fn uniaxial_patch_strain_matches_hookes_law() {
        // Uniaxial strain state (not uniaxial stress): u_z = eps * z, lateral
        // displacement zero. Stress_zz = (lambda + 2 mu) * eps; the top-face
        // nodal forces must sum to stress * area.
        let h = 1.0;
        let eps = 0.01;
        let k = cube_stiffness(h, MATERIAL).unwrap();

        let mut u = [0.0f64; 24];
        for node in 0..8 {
            let z = (NODE_SIGNS[node][2] + 1.0) / 2.0 * h;
            u[node * 3 + 2] = eps * z;
        }
        let f: Vec<f64> = k
            .iter()
            .map(|row| row.iter().zip(&u).map(|(a, b)| a * b).sum())
            .collect();

        let e = MATERIAL.youngs_modulus;
        let nu = MATERIAL.poissons_ratio;
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        let expected = (lambda + 2.0 * mu) * eps * h * h;

        let top_force: f64 = (0..8)
            .filter(|node| NODE_SIGNS[*node][2] > 0.0)
            .map(|node| f[node * 3 + 2])
            .sum();
        assert!(
            (top_force - expected).abs() < 1e-10,
            "top force {top_force}, expected {expected}"
        );
    }

    #[test]
    fn general_hex_matches_cube_for_cube_geometry() {
        let h = 0.25;
        let a = cube_stiffness(h, MATERIAL).unwrap();
        let coords = [
            [0.0, 0.0, 0.0],
            [h, 0.0, 0.0],
            [h, h, 0.0],
            [0.0, h, 0.0],
            [0.0, 0.0, h],
            [h, 0.0, h],
            [h, h, h],
            [0.0, h, h],
        ];
        let b = hex8_stiffness(&coords, MATERIAL).unwrap();
        for i in 0..24 {
            for j in 0..24 {
                assert!((a[i][j] - b[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn degenerate_elements_are_rejected() {
        let coords = [[0.0; 3]; 8];
        assert!(hex8_stiffness(&coords, MATERIAL).is_err());
    }
}
