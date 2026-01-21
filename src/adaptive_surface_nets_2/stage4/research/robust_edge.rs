//! Robust Edge Detection for Binary Samplers
//!
//! Key insight from README: Models return binary values (1 inside, 0 outside),
//! NOT signed distance fields. Central difference gradients are useless.
//!
//! Strategy:
//! 1. Find surface points via binary search (this works correctly)
//! 2. Use RANSAC to find first plane (fits largest consensus set)
//! 3. Remove inliers, RANSAC again to find second plane
//! 4. Edge = intersection of the two planes

use super::analytical_cube::angle_between;
use super::sample_cache::{find_crossing_in_direction, SampleCache};

#[derive(Clone, Debug)]
pub struct RobustEdgeResult {
    pub edge_direction: (f64, f64, f64),
    pub point_on_edge: (f64, f64, f64),
    pub face_a_normal: (f64, f64, f64),
    pub face_b_normal: (f64, f64, f64),
    pub inliers_a: usize,
    pub inliers_b: usize,
    pub residual_a: f64,
    pub residual_b: f64,
}

/// Robust edge detection using RANSAC plane fitting (no gradients).
///
/// For binary samplers, we can't use gradients. Instead:
/// 1. Probe many directions, binary search to find surface points
/// 2. RANSAC finds first plane (largest consensus)
/// 3. Remove inliers, RANSAC finds second plane
/// 4. Planes intersect to give edge
pub fn robust_edge_detection<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    num_probes: usize,
    max_distance: f64,
    ransac_inlier_threshold: f64,
    ransac_iterations: usize,
) -> Option<RobustEdgeResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Step 1: Find surface points
    let directions = generate_sphere_directions(num_probes);
    let mut surface_points = Vec::new();

    for dir in &directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache, point, *dir, max_distance, 0.001, 30,
        ) {
            surface_points.push(crossing);
        }
    }

    if surface_points.len() < 10 {
        return None;
    }

    // Step 2: RANSAC for first plane
    let (plane1_inliers, plane1_centroid, plane1_normal, plane1_residual) =
        ransac_plane_fit(&surface_points, ransac_inlier_threshold, ransac_iterations)?;

    if plane1_inliers.len() < 3 {
        return None;
    }

    // Step 3: Remove inliers, RANSAC for second plane
    let remaining: Vec<_> = surface_points
        .iter()
        .filter(|p| !is_inlier(**p, plane1_centroid, plane1_normal, ransac_inlier_threshold))
        .cloned()
        .collect();

    if remaining.len() < 3 {
        return None;
    }

    let (plane2_inliers, plane2_centroid, plane2_normal, plane2_residual) =
        ransac_plane_fit(&remaining, ransac_inlier_threshold, ransac_iterations)?;

    if plane2_inliers.len() < 3 {
        return None;
    }

    // Orient normals away from query point
    let normal_a = orient_away(plane1_normal, plane1_centroid, point);
    let normal_b = orient_away(plane2_normal, plane2_centroid, point);

    // Step 4: Edge = intersection of planes
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None; // Planes are parallel
    }

    let point_on_edge = find_edge_point(point, plane1_centroid, normal_a, plane2_centroid, normal_b, edge_dir);

    Some(RobustEdgeResult {
        edge_direction: edge_dir,
        point_on_edge,
        face_a_normal: normal_a,
        face_b_normal: normal_b,
        inliers_a: plane1_inliers.len(),
        inliers_b: plane2_inliers.len(),
        residual_a: plane1_residual,
        residual_b: plane2_residual,
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
    let mut best_normal = (0.0, 0.0, 1.0);
    let mut best_centroid = (0.0, 0.0, 0.0);

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
            best_normal = normal;
            best_centroid = p1;
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

    // Compute centroid
    let n = points.len() as f64;
    let centroid = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    );

    // Build covariance matrix
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

fn orient_away(normal: (f64, f64, f64), plane_point: (f64, f64, f64), away_from: (f64, f64, f64)) -> (f64, f64, f64) {
    let to_plane = sub(plane_point, away_from);
    if dot(normal, to_plane) < 0.0 {
        neg(normal)
    } else {
        normal
    }
}

fn find_edge_point(
    query: (f64, f64, f64),
    centroid_a: (f64, f64, f64),
    normal_a: (f64, f64, f64),
    centroid_b: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Iteratively project onto both planes to find their intersection
    let midpoint = (
        (centroid_a.0 + centroid_b.0) / 2.0,
        (centroid_a.1 + centroid_b.1) / 2.0,
        (centroid_a.2 + centroid_b.2) / 2.0,
    );

    let mut p = midpoint;
    for _ in 0..20 {
        let dist_a = dot(sub(p, centroid_a), normal_a);
        p = sub(p, scale(normal_a, dist_a));
        let dist_b = dot(sub(p, centroid_b), normal_b);
        p = sub(p, scale(normal_b, dist_b));
    }

    // Project to be nearest to query along edge direction
    let to_query = sub(query, p);
    let t = dot(to_query, edge_direction);
    (
        p.0 + t * edge_direction.0,
        p.1 + t * edge_direction.1,
        p.2 + t * edge_direction.2,
    )
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

fn scale(v: (f64, f64, f64), s: f64) -> (f64, f64, f64) {
    (v.0 * s, v.1 * s, v.2 * s)
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

    #[test]
    fn benchmark_robust_ransac() {
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
        println!("ROBUST RANSAC EDGE DETECTION (No Gradients)");
        println!("{}\n", "=".repeat(80));

        // Test different inlier thresholds and probe counts
        let configs = [
            (150, 0.02, 200, "150 probes, 0.02 threshold"),
            (200, 0.02, 200, "200 probes, 0.02 threshold"),
            (300, 0.02, 200, "300 probes, 0.02 threshold"),
            (200, 0.01, 200, "200 probes, 0.01 threshold"),
            (200, 0.03, 200, "200 probes, 0.03 threshold"),
            (200, 0.02, 500, "200 probes, 500 iterations"),
        ];

        for (num_probes, threshold, iterations, desc) in &configs {
            println!("Config: {}", desc);
            println!("{:<6} {:>10} {:>10} {:>10} {:>15}", "Edge", "Na Err°", "Nb Err°", "Avg°", "Inliers");
            println!("{}", "-".repeat(60));

            let mut all_errors = Vec::new();
            let mut successes = 0;

            for edge_idx in 0..12 {
                let edge = cube.get_edge(edge_idx);
                let bisector = normalize((
                    edge.face_a_normal.0 + edge.face_b_normal.0,
                    edge.face_a_normal.1 + edge.face_b_normal.1,
                    edge.face_a_normal.2 + edge.face_b_normal.2,
                ));
                let test_point = (
                    edge.point_on_edge.0 - bisector.0 * 0.1,
                    edge.point_on_edge.1 - bisector.1 * 0.1,
                    edge.point_on_edge.2 - bisector.2 * 0.1,
                );

                let cache = SampleCache::new(&sampler);

                if let Some(result) = robust_edge_detection(
                    test_point, &cache, *num_probes, 0.5, *threshold, *iterations,
                ) {
                    let (na_err, nb_err) = best_normal_pairing(
                        (result.face_a_normal, result.face_b_normal),
                        (edge.face_a_normal, edge.face_b_normal),
                    );
                    let avg = (na_err + nb_err) / 2.0;

                    println!("{:<6} {:>10.2} {:>10.2} {:>10.2} {:>7}/{:<7}",
                             edge_idx, na_err, nb_err, avg,
                             result.inliers_a, result.inliers_b);

                    all_errors.push(na_err);
                    all_errors.push(nb_err);
                    successes += 1;
                } else {
                    println!("{:<6} {:>10} {:>10} {:>10}", edge_idx, "FAIL", "FAIL", "FAIL");
                }
            }

            if !all_errors.is_empty() {
                let avg = all_errors.iter().sum::<f64>() / all_errors.len() as f64;
                let max = all_errors.iter().cloned().fold(0.0_f64, f64::max);
                println!("{}", "-".repeat(60));
                println!("Success: {}/12, Avg: {:.2}°, Max: {:.2}°\n",
                         successes, avg, max);
            } else {
                println!("All failed\n");
            }
        }
    }

    fn best_normal_pairing(
        detected: ((f64, f64, f64), (f64, f64, f64)),
        expected: ((f64, f64, f64), (f64, f64, f64)),
    ) -> (f64, f64) {
        let a1 = (
            angle_between(detected.0, expected.0).to_degrees(),
            angle_between(detected.1, expected.1).to_degrees(),
        );
        let a2 = (
            angle_between(detected.0, expected.1).to_degrees(),
            angle_between(detected.1, expected.0).to_degrees(),
        );
        if a1.0 + a1.1 < a2.0 + a2.1 {
            a1
        } else {
            a2
        }
    }
}
