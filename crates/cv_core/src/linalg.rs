//! The small dense linear algebra the detector and solver need: a
//! symmetric eigen-solver for the DLT null vector, a linear solve for the
//! normal equations, 3x3 helpers and the nearest rotation.

pub type Mat3 = [[f64; 3]; 3];

/// Eigenvector of the smallest eigenvalue of a symmetric matrix (cyclic
/// Jacobi rotations; the matrices here are at most 9x9).
pub fn smallest_eigenvector(a: &[Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[i][j] * m[i][j];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k][p], m[k][q]);
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p][k], m[q][k]);
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let smallest = (0..n)
        .min_by(|&i, &j| m[i][i].partial_cmp(&m[j][j]).unwrap())
        .unwrap();
    (0..n).map(|k| v[k][smallest]).collect()
}

/// Solves `a x = b` by Gaussian elimination with partial pivoting; `None`
/// when singular.
pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, rhs)| {
            let mut r = row.clone();
            r.push(*rhs);
            r
        })
        .collect();
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())?;
        if m[pivot][col].abs() < 1e-14 {
            return None;
        }
        m.swap(col, pivot);
        for row in (col + 1)..n {
            let f = m[row][col] / m[col][col];
            for k in col..=n {
                m[row][k] -= f * m[col][k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = m[row][n];
        for k in (row + 1)..n {
            acc -= m[row][k] * x[k];
        }
        x[row] = acc / m[row][row];
    }
    Some(x)
}

/// The inverse of a symmetric positive matrix, column by column; `None`
/// when singular.
pub fn inverse(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut columns = Vec::with_capacity(n);
    for j in 0..n {
        let e: Vec<f64> = (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
        columns.push(solve(a, &e)?);
    }
    Some(
        (0..n)
            .map(|i| (0..n).map(|j| columns[j][i]).collect())
            .collect(),
    )
}

pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub fn mat3_transpose(a: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[j][i];
        }
    }
    out
}

pub fn mat3_vec(a: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

pub fn mat3_det(a: &Mat3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

pub fn mat3_inverse(a: &Mat3) -> Option<Mat3> {
    let det = mat3_det(a);
    if det.abs() < 1e-300 {
        return None;
    }
    let c = |i: usize, j: usize| {
        let r = [(i + 1) % 3, (i + 2) % 3];
        let s = [(j + 1) % 3, (j + 2) % 3];
        a[r[0]][s[0]] * a[r[1]][s[1]] - a[r[0]][s[1]] * a[r[1]][s[0]]
    };
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = c(j, i) / det;
        }
    }
    Some(out)
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalized(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n == 0.0 {
        return a;
    }
    [a[0] / n, a[1] / n, a[2] / n]
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// The rotation nearest to `m` (Higham's iteration to the orthogonal
/// polar factor), with a positive determinant.
pub fn nearest_rotation(m: &Mat3) -> Mat3 {
    let mut r = *m;
    for _ in 0..30 {
        let Some(inv_t) = mat3_inverse(&mat3_transpose(&r)) else {
            break;
        };
        let mut next = [[0.0; 3]; 3];
        let mut delta = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                next[i][j] = 0.5 * (r[i][j] + inv_t[i][j]);
                delta += (next[i][j] - r[i][j]).abs();
            }
        }
        r = next;
        if delta < 1e-15 {
            break;
        }
    }
    if mat3_det(&r) < 0.0 {
        for row in &mut r {
            row[2] = -row[2];
        }
    }
    r
}

/// Rotation matrix of a rotation vector (axis times angle).
pub fn rotation_from_vector(w: [f64; 3]) -> Mat3 {
    let angle = norm(w);
    if angle < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let [x, y, z] = scale(w, 1.0 / angle);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// The angle of a rotation matrix, radians.
pub fn rotation_angle(r: &Mat3) -> f64 {
    ((r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_solve_and_rotations() {
        // Smallest eigenvector of diag(3, 1, 2) is e2.
        let a = vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let v = smallest_eigenvector(&a);
        assert!(
            v[1].abs() > 0.999 && v[0].abs() < 1e-6 && v[2].abs() < 1e-6,
            "{v:?}"
        );

        let x = solve(&[vec![2.0, 1.0], vec![1.0, 3.0]], &[3.0, 5.0]).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
        assert!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[1.0, 2.0]).is_none());
        let inv = inverse(&[vec![2.0, 1.0], vec![1.0, 3.0]]).unwrap();
        assert!((inv[0][0] - 0.6).abs() < 1e-12 && (inv[0][1] + 0.2).abs() < 1e-12);

        let r = rotation_from_vector([0.0, 0.0, std::f64::consts::FRAC_PI_2]);
        let p = mat3_vec(&r, [1.0, 0.0, 0.0]);
        assert!((p[1] - 1.0).abs() < 1e-12 && p[0].abs() < 1e-12);
        assert!((rotation_angle(&r) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        let inv = mat3_inverse(&r).unwrap();
        let identity = mat3_mul(&r, &inv);
        assert!((identity[0][0] - 1.0).abs() < 1e-12 && identity[0][1].abs() < 1e-12);

        // A scaled, slightly sheared rotation snaps back to the rotation.
        let mut skew = r;
        for row in &mut skew {
            for c in row.iter_mut() {
                *c *= 2.5;
            }
        }
        skew[0][1] += 0.05;
        let back = nearest_rotation(&skew);
        assert!((rotation_angle(&mat3_mul(&back, &mat3_transpose(&r)))).abs() < 0.02);
        assert!((mat3_det(&back) - 1.0).abs() < 1e-9);
    }
}
