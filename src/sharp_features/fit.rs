//! Least-squares plane fitting via the covariance eigendecomposition.
//!
//! The fitted normal is the eigenvector of the point covariance matrix with
//! the smallest eigenvalue; the RMS point-to-plane residual is the square root
//! of that eigenvalue. Residuals are the key classification signal downstream:
//! a patch spanning a single smooth face fits tightly, a patch straddling a
//! sharp feature does not.

use glam::DVec3;

#[derive(Clone, Debug)]
pub struct PlaneFit {
    pub centroid: DVec3,
    /// Unit normal. Sign is arbitrary; orient against a reference before use.
    pub normal: DVec3,
    /// Root-mean-square point-to-plane distance.
    pub rms_residual: f64,
}

/// Fit a plane to `points` by PCA. Returns `None` for fewer than 3 points or
/// a degenerate (collinear) configuration.
pub fn fit_plane(points: &[DVec3]) -> Option<PlaneFit> {
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f64;
    let centroid = points.iter().copied().sum::<DVec3>() / n;

    let mut cov = [[0.0f64; 3]; 3];
    for p in points {
        let d = *p - centroid;
        let v = [d.x, d.y, d.z];
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                *entry += v[i] * v[j];
            }
        }
    }
    for row in &mut cov {
        for entry in row.iter_mut() {
            *entry /= n;
        }
    }

    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric3(cov);

    // Index of the smallest eigenvalue, with the middle one for the
    // degeneracy check.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
    let smallest = order[0];
    let middle = order[1];

    // Collinear points: the two smallest eigenvalues are both ~zero relative
    // to the spread, so the normal direction is unconstrained.
    let scale = eigenvalues[order[2]].max(1e-300);
    if eigenvalues[middle] / scale < 1e-12 {
        return None;
    }

    let normal = DVec3::new(
        eigenvectors[0][smallest],
        eigenvectors[1][smallest],
        eigenvectors[2][smallest],
    )
    .normalize();

    Some(PlaneFit {
        centroid,
        normal,
        rms_residual: eigenvalues[smallest].max(0.0).sqrt(),
    })
}

/// Jacobi eigendecomposition of a symmetric 3x3 matrix. Returns
/// `(eigenvalues, eigenvectors)` with eigenvectors as columns, i.e.
/// eigenvector k is `(v[0][k], v[1][k], v[2][k])`.
fn jacobi_eigen_symmetric3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let scale = a
        .iter()
        .flatten()
        .fold(0.0f64, |acc, x| acc.max(x.abs()))
        .max(1e-300);

    for _ in 0..64 {
        // Largest off-diagonal element.
        let (mut p, mut q, mut max) = (0usize, 1usize, a[0][1].abs());
        for (i, j) in [(0usize, 2usize), (1, 2)] {
            if a[i][j].abs() > max {
                max = a[i][j].abs();
                p = i;
                q = j;
            }
        }
        if max < 1e-15 * scale {
            break;
        }

        // Rotation angle that zeroes a[p][q].
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[p][p] - a[q][q]);
        let (s, c) = theta.sin_cos();

        // A <- R^T A R and V <- V R, with R the Givens rotation in the (p, q)
        // plane. Explicit multiplies keep the sign conventions foolproof.
        let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        r[p][p] = c;
        r[q][q] = c;
        r[p][q] = -s;
        r[q][p] = s;

        a = mat_mul(&mat_mul(&transpose(&r), &a), &r);
        v = mat_mul(&v, &r);
    }

    ([a[0][0], a[1][1], a[2][2]], v)
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (j, entry) in out_row.iter_mut().enumerate() {
            *entry = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in a.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            out[j][i] = x;
        }
    }
    out
}

/// Angle between two directions in degrees, ignoring sign (planes have no
/// preferred orientation).
pub fn unsigned_angle_degrees(a: DVec3, b: DVec3) -> f64 {
    a.dot(b).abs().clamp(0.0, 1.0).acos().to_degrees()
}

/// Per-vertex plane fit over a k-ring neighborhood.
#[derive(Clone, Debug)]
pub struct VertexFit {
    /// Fitted plane normal, oriented outward when a reference normal is
    /// available (sign is otherwise the PCA's arbitrary choice).
    pub normal: DVec3,
    /// RMS point-to-plane distance in cell units.
    pub residual_cells: f64,
}

/// Fit a plane to each vertex's k-ring. `orient_refs` supplies outward
/// reference normals (typically the accumulated mesh normals); pass an empty
/// slice to skip orientation. Vertices with fewer than 4 ring members or a
/// degenerate fit get `None`.
pub fn ring_fits(
    positions: &[DVec3],
    adjacency: &crate::sharp_features::adjacency::MeshAdjacency,
    orient_refs: &[DVec3],
    cell: f64,
    k: usize,
) -> Vec<Option<VertexFit>> {
    (0..positions.len() as u32)
        .map(|v| {
            let ring = adjacency.k_ring(v, k);
            if ring.len() < 4 {
                return None;
            }
            let pts: Vec<DVec3> = ring.iter().map(|&u| positions[u as usize]).collect();
            let fit = fit_plane(&pts)?;
            let mut normal = fit.normal;
            if let Some(reference) = orient_refs.get(v as usize)
                && reference.length_squared() > 0.25
                && normal.dot(*reference) < 0.0
            {
                normal = -normal;
            }
            Some(VertexFit {
                normal,
                residual_cells: fit.rms_residual / cell,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_points(n: usize, f: impl Fn(f64, f64) -> DVec3) -> Vec<DVec3> {
        let mut state = 0x243f6a8885a308d3u64;
        let mut rand01 = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(|_| f(rand01() - 0.5, rand01() - 0.5)).collect()
    }

    #[test]
    fn fits_exact_plane() {
        // Plane through (1,2,3) with normal (1,1,1)/sqrt(3).
        let n = DVec3::ONE.normalize();
        let origin = DVec3::new(1.0, 2.0, 3.0);
        let u = n.any_orthonormal_vector();
        let v = n.cross(u);
        let points = pseudo_random_points(50, |a, b| origin + u * a + v * b);
        let fit = fit_plane(&points).unwrap();
        assert!(fit.rms_residual < 1e-12);
        assert!(unsigned_angle_degrees(fit.normal, n) < 1e-6);
    }

    #[test]
    fn fits_noisy_plane() {
        let n = DVec3::new(0.0, 0.0, 1.0);
        let mut state = 7u64;
        let mut noise = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02
        };
        let mut points = pseudo_random_points(200, |a, b| DVec3::new(a, b, 0.0));
        for p in &mut points {
            p.z += noise();
        }
        let fit = fit_plane(&points).unwrap();
        assert!(unsigned_angle_degrees(fit.normal, n) < 2.0);
        assert!(fit.rms_residual < 0.01);
    }

    #[test]
    fn rejects_collinear_points() {
        let points: Vec<DVec3> = (0..10)
            .map(|i| DVec3::new(i as f64, 2.0 * i as f64, 0.5 * i as f64))
            .collect();
        assert!(fit_plane(&points).is_none());
    }

    #[test]
    fn jacobi_recovers_known_eigensystem() {
        // A = Q D Q^T with D = diag(5, 2, 1) and Q a rotation by 30 deg about Z.
        let (s, c) = 30.0f64.to_radians().sin_cos();
        let q = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let d = [[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let a = mat_mul(&mat_mul(&q, &d), &transpose(&q));
        let (vals, vecs) = jacobi_eigen_symmetric3(a);
        let mut sorted = vals;
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 1e-9);
        assert!((sorted[1] - 2.0).abs() < 1e-9);
        assert!((sorted[2] - 5.0).abs() < 1e-9);
        // Eigenvector for eigenvalue 1 must be +/-Z.
        let k = vals.iter().position(|x| (x - 1.0).abs() < 1e-9).unwrap();
        let ev = DVec3::new(vecs[0][k], vecs[1][k], vecs[2][k]);
        assert!(unsigned_angle_degrees(ev, DVec3::Z) < 1e-6);
    }
}
