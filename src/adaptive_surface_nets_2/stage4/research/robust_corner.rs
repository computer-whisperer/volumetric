//! Robust Corner Detection for Binary Samplers
//!
//! Uses iterative RANSAC plane fitting to find 3+ faces meeting at a corner.
//! No gradient computation - works with binary samplers.

use super::sample_cache::{find_crossing_in_direction, SampleCache};

#[derive(Clone, Debug)]
pub struct RobustCornerResult {
    pub position: (f64, f64, f64),
    pub face_normals: Vec<(f64, f64, f64)>,
    pub inlier_counts: Vec<usize>,
    pub residuals: Vec<f64>,
}

/// Robust corner detection using iterative RANSAC plane fitting.
///
/// For binary samplers:
/// 1. Probe many directions, binary search to find surface points
/// 2. RANSAC finds first plane (largest consensus)
/// 3. Remove inliers, repeat for second and third planes
/// 4. Corner = intersection of 3 planes
pub fn robust_corner_detection<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    num_probes: usize,
    max_distance: f64,
    ransac_inlier_threshold: f64,
    ransac_iterations: usize,
    min_planes: usize,  // Usually 3 for a corner
) -> Option<RobustCornerResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Step 1: Find surface points
    let directions = generate_sphere_directions(num_probes);
    let mut surface_points: Vec<(f64, f64, f64)> = Vec::new();

    for dir in &directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache, point, *dir, max_distance, 0.001, 30,
        ) {
            surface_points.push(crossing);
        }
    }

    if surface_points.len() < 15 {
        return None;
    }

    // Step 2: Iteratively find planes using RANSAC
    let mut remaining_points = surface_points.clone();
    let mut planes: Vec<(Vec<(f64, f64, f64)>, (f64, f64, f64), (f64, f64, f64), f64)> = Vec::new();

    // Minimum inliers needed for a valid plane (at least 10% of original points)
    let min_inliers_for_valid_plane = (surface_points.len() / 10).max(5);

    for plane_idx in 0..min_planes.max(4) {
        if remaining_points.len() < min_inliers_for_valid_plane {
            break;
        }

        if let Some((inliers, centroid, normal, residual)) =
            ransac_plane_fit(&remaining_points, ransac_inlier_threshold, ransac_iterations)
        {
            // Stop if this plane has too few inliers (likely noise)
            if inliers.len() < min_inliers_for_valid_plane {
                break;
            }

            // Check if this normal is too similar to an existing plane (likely a split face)
            let is_duplicate = planes.iter().any(|(_, _, existing_normal, _)| {
                let dot_val = dot(normal, *existing_normal).abs();
                dot_val > 0.9  // Within ~25° of each other
            });

            if is_duplicate {
                // This plane is likely a split of an existing face, skip it
                // but remove its inliers so we can continue looking for genuinely different planes
                remaining_points.retain(|p| {
                    !is_inlier(*p, centroid, normal, ransac_inlier_threshold)
                });
                continue;
            }

            // For plane 4+, require it to be nearly as good as earlier planes
            if plane_idx >= 3 && planes.len() >= 3 {
                let avg_prev_inliers = planes.iter().map(|(i, _, _, _)| i.len()).sum::<usize>() / planes.len();
                // 4th plane should have at least 50% of average inlier count
                if inliers.len() < avg_prev_inliers / 2 {
                    break;
                }
            }

            // Remove inliers from remaining points
            remaining_points.retain(|p| {
                !is_inlier(*p, centroid, normal, ransac_inlier_threshold)
            });

            planes.push((inliers, centroid, normal, residual));
        } else {
            break;
        }
    }

    if planes.len() < min_planes {
        return None;
    }

    // Step 3: Orient normals away from query point
    let mut face_normals = Vec::new();
    let mut inlier_counts = Vec::new();
    let mut residuals = Vec::new();

    for (inliers, centroid, normal, residual) in &planes {
        let oriented = orient_away(*normal, *centroid, point);
        face_normals.push(oriented);
        inlier_counts.push(inliers.len());
        residuals.push(*residual);
    }

    // Step 4: Find corner position (intersection of first 3 planes)
    let corner_pos = if planes.len() >= 3 {
        let (_, c0, n0, _) = &planes[0];
        let (_, c1, n1, _) = &planes[1];
        let (_, c2, n2, _) = &planes[2];

        // Solve system: n0·(x-c0)=0, n1·(x-c1)=0, n2·(x-c2)=0
        // Rewritten: n0·x = n0·c0, etc.
        let d0 = dot(*n0, *c0);
        let d1 = dot(*n1, *c1);
        let d2 = dot(*n2, *c2);

        solve_3_planes(*n0, d0, *n1, d1, *n2, d2).unwrap_or(point)
    } else {
        point
    };

    Some(RobustCornerResult {
        position: corner_pos,
        face_normals,
        inlier_counts,
        residuals,
    })
}

fn is_inlier(p: (f64, f64, f64), plane_point: (f64, f64, f64), plane_normal: (f64, f64, f64), threshold: f64) -> bool {
    let dist = dot(sub(p, plane_point), plane_normal).abs();
    dist < threshold
}

/// RANSAC plane fitting - finds the plane with most inliers
fn ransac_plane_fit(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> Option<(Vec<(f64, f64, f64)>, (f64, f64, f64), (f64, f64, f64), f64)> {
    if points.len() < 3 {
        return None;
    }

    let n = points.len();
    let mut best_inliers: Vec<(f64, f64, f64)> = Vec::new();

    for i in 0..iterations {
        // Deterministic "random" point selection
        let idx1 = (i * 7 + 1) % n;
        let idx2 = (i * 13 + 3) % n;
        let idx3 = (i * 19 + 7) % n;

        if idx1 == idx2 || idx2 == idx3 || idx1 == idx3 {
            continue;
        }

        let p1 = points[idx1];
        let p2 = points[idx2];
        let p3 = points[idx3];

        // Compute plane from 3 points
        let v1 = sub(p2, p1);
        let v2 = sub(p3, p1);
        let normal = normalize(cross(v1, v2));

        if length(normal) < 1e-6 {
            continue; // Degenerate triangle
        }

        // Count inliers
        let inliers: Vec<_> = points
            .iter()
            .filter(|p| {
                let dist = dot(sub(**p, p1), normal).abs();
                dist < inlier_threshold
            })
            .cloned()
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }

    if best_inliers.len() < 3 {
        return None;
    }

    // Refit plane to all inliers for better accuracy
    let (refined_centroid, refined_normal, residual) = fit_plane_svd(&best_inliers);

    Some((best_inliers, refined_centroid, refined_normal, residual))
}

/// Fit plane using SVD (smallest eigenvector of covariance matrix)
fn fit_plane_svd(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
    if points.len() < 3 {
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), f64::INFINITY);
    }

    let n = points.len() as f64;
    let centroid = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    );

    let centered: Vec<_> = points.iter().map(|p| sub(*p, centroid)).collect();

    let mut cov = [[0.0; 3]; 3];
    for p in &centered {
        cov[0][0] += p.0 * p.0;
        cov[0][1] += p.0 * p.1;
        cov[0][2] += p.0 * p.2;
        cov[1][1] += p.1 * p.1;
        cov[1][2] += p.1 * p.2;
        cov[2][2] += p.2 * p.2;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // Find smallest eigenvector via power iteration on shifted matrix
    let max_eigenvalue_estimate = cov
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row[i].abs()
                + row.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v.abs()).sum::<f64>()
        })
        .fold(0.0_f64, f64::max);

    let shift = max_eigenvalue_estimate + 1.0;
    let shifted = [
        [shift - cov[0][0], -cov[0][1], -cov[0][2]],
        [-cov[1][0], shift - cov[1][1], -cov[1][2]],
        [-cov[2][0], -cov[2][1], shift - cov[2][2]],
    ];

    let mut v = (1.0, 1.0, 1.0);
    for _ in 0..100 {
        let new_v = (
            shifted[0][0] * v.0 + shifted[0][1] * v.1 + shifted[0][2] * v.2,
            shifted[1][0] * v.0 + shifted[1][1] * v.1 + shifted[1][2] * v.2,
            shifted[2][0] * v.0 + shifted[2][1] * v.1 + shifted[2][2] * v.2,
        );

        let len = length(new_v);
        if len < 1e-12 {
            break;
        }
        v = (new_v.0 / len, new_v.1 / len, new_v.2 / len);
    }

    // Compute RMS residual
    let mut residual_sum = 0.0;
    for p in &centered {
        let dist = dot(*p, v).abs();
        residual_sum += dist * dist;
    }
    let rms_residual = (residual_sum / points.len() as f64).sqrt();

    (centroid, v, rms_residual)
}

/// Solve for intersection of 3 planes
/// Each plane: n·x = d
fn solve_3_planes(
    n0: (f64, f64, f64), d0: f64,
    n1: (f64, f64, f64), d1: f64,
    n2: (f64, f64, f64), d2: f64,
) -> Option<(f64, f64, f64)> {
    // Matrix: [n0; n1; n2] * x = [d0; d1; d2]
    let det = n0.0 * (n1.1 * n2.2 - n1.2 * n2.1)
            - n0.1 * (n1.0 * n2.2 - n1.2 * n2.0)
            + n0.2 * (n1.0 * n2.1 - n1.1 * n2.0);

    if det.abs() < 1e-10 {
        return None; // Planes don't intersect at a point
    }

    let x = (d0 * (n1.1 * n2.2 - n1.2 * n2.1)
           - n0.1 * (d1 * n2.2 - n1.2 * d2)
           + n0.2 * (d1 * n2.1 - n1.1 * d2)) / det;

    let y = (n0.0 * (d1 * n2.2 - n1.2 * d2)
           - d0 * (n1.0 * n2.2 - n1.2 * n2.0)
           + n0.2 * (n1.0 * d2 - d1 * n2.0)) / det;

    let z = (n0.0 * (n1.1 * d2 - d1 * n2.1)
           - n0.1 * (n1.0 * d2 - d1 * n2.0)
           + d0 * (n1.0 * n2.1 - n1.1 * n2.0)) / det;

    Some((x, y, z))
}

fn orient_away(normal: (f64, f64, f64), plane_point: (f64, f64, f64), away_from: (f64, f64, f64)) -> (f64, f64, f64) {
    let to_plane = sub(plane_point, away_from);
    if dot(normal, to_plane) < 0.0 {
        neg(normal)
    } else {
        normal
    }
}

fn generate_sphere_directions(n: usize) -> Vec<(f64, f64, f64)> {
    let mut directions = Vec::with_capacity(n);
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let angle_increment = std::f64::consts::PI * 2.0 * golden_ratio;

    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let phi = angle_increment * i as f64;
        let theta = (1.0 - 2.0 * t).acos();
        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();
        directions.push((x, y, z));
    }

    directions
}

// Vector math
fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn neg(v: (f64, f64, f64)) -> (f64, f64, f64) {
    (-v.0, -v.1, -v.2)
}

fn length(v: (f64, f64, f64)) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = length(v);
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::analytical_cube::AnalyticalRotatedCube;

    fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
        let d = dot(normalize(a), normalize(b)).clamp(-1.0, 1.0);
        d.acos()
    }

    #[test]
    fn benchmark_robust_corner() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let cube_for_sampler = cube.clone();
        let sampler = move |x: f64, y: f64, z: f64| -> f32 {
            let local = cube_for_sampler.world_to_local((x, y, z));
            let h = 0.5;
            if local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h {
                1.0
            } else {
                -1.0
            }
        };

        println!("\n{}", "=".repeat(80));
        println!("ROBUST RANSAC CORNER DETECTION (No Gradients)");
        println!("{}\n", "=".repeat(80));

        // Test configurations
        let configs = [
            (300, 0.01, 200, "300 probes, 0.01 threshold"),
            (300, 0.008, 200, "300 probes, 0.008 threshold"),
            (300, 0.006, 200, "300 probes, 0.006 threshold"),
            (400, 0.008, 300, "400 probes, 0.008 threshold"),
            (400, 0.006, 300, "400 probes, 0.006 threshold"),
        ];

        for (num_probes, threshold, iterations, desc) in &configs {
            println!("Config: {}", desc);
            println!("{:<8} {:>8} {:>10} {:>10} {:>10} {:>10}",
                     "Corner", "Faces", "N1 Err°", "N2 Err°", "N3 Err°", "Pos Err");
            println!("{}", "-".repeat(70));

            let mut all_normal_errors = Vec::new();
            let mut all_position_errors = Vec::new();
            let mut successes = 0;

            for corner_idx in 0..8 {
                let corner = cube.get_corner(corner_idx);

                // Test point offset inside from corner
                let offset_dir = normalize((
                    -corner.position.0,
                    -corner.position.1,
                    -corner.position.2,
                ));
                let test_point = (
                    corner.position.0 + offset_dir.0 * 0.1,
                    corner.position.1 + offset_dir.1 * 0.1,
                    corner.position.2 + offset_dir.2 * 0.1,
                );

                let cache = SampleCache::new(&sampler);

                if let Some(result) = robust_corner_detection(
                    test_point, &cache, *num_probes, 0.5, *threshold, *iterations, 3,
                ) {
                    if result.face_normals.len() >= 3 {
                        // Match detected normals to expected normals
                        let errors = best_corner_normal_matching(
                            &result.face_normals,
                            &corner.face_normals,
                        );

                        let pos_err = length(sub(result.position, corner.position));

                        println!("{:<8} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.4}",
                                 corner_idx, result.face_normals.len(),
                                 errors.0, errors.1, errors.2, pos_err);

                        all_normal_errors.push(errors.0);
                        all_normal_errors.push(errors.1);
                        all_normal_errors.push(errors.2);
                        all_position_errors.push(pos_err);
                        successes += 1;
                    } else {
                        println!("{:<8} {:>8} {:>10} {:>10} {:>10}",
                                 corner_idx, result.face_normals.len(), "—", "—", "—");
                    }
                } else {
                    println!("{:<8} {:>8}", corner_idx, "FAIL");
                }
            }

            if !all_normal_errors.is_empty() {
                let avg_normal = all_normal_errors.iter().sum::<f64>() / all_normal_errors.len() as f64;
                let max_normal = all_normal_errors.iter().cloned().fold(0.0_f64, f64::max);
                let avg_pos = all_position_errors.iter().sum::<f64>() / all_position_errors.len() as f64;
                println!("{}", "-".repeat(70));
                println!("Success: {}/8, Avg normal: {:.2}°, Max normal: {:.2}°, Avg pos: {:.4}\n",
                         successes, avg_normal, max_normal, avg_pos);
            } else {
                println!("All failed\n");
            }
        }
    }

    /// Find best matching between detected and expected normals
    fn best_corner_normal_matching(
        detected: &[(f64, f64, f64)],
        expected: &[(f64, f64, f64)],
    ) -> (f64, f64, f64) {
        // For each expected normal, find the best matching detected normal
        let mut errors = Vec::new();

        for exp in expected.iter().take(3) {
            let mut best_err = f64::MAX;
            for det in detected {
                let err = angle_between(*det, *exp).to_degrees();
                if err < best_err {
                    best_err = err;
                }
            }
            errors.push(best_err);
        }

        while errors.len() < 3 {
            errors.push(f64::NAN);
        }

        (errors[0], errors[1], errors[2])
    }
}
