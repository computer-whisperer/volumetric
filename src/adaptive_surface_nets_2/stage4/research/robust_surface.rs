//! Robust Surface/Face Detection for Binary Samplers
//!
//! Uses RANSAC plane fitting to find the dominant face normal,
//! rejecting outliers from adjacent faces.

use super::sample_cache::{find_crossing_in_direction, SampleCache};

#[derive(Clone, Debug)]
pub struct RobustSurfaceResult {
    pub normal: (f64, f64, f64),
    pub point_on_surface: (f64, f64, f64),
    pub inlier_count: usize,
    pub total_points: usize,
    pub residual: f64,
}

/// Robust surface detection using RANSAC plane fitting.
///
/// For points on a flat face, this should give very accurate normals.
/// RANSAC rejects any outliers from adjacent faces.
pub fn robust_surface_detection<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    num_probes: usize,
    max_distance: f64,
    ransac_inlier_threshold: f64,
    ransac_iterations: usize,
) -> Option<RobustSurfaceResult>
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

    if surface_points.len() < 5 {
        return None;
    }

    // Step 2: RANSAC plane fitting
    let (inliers, centroid, normal, residual) =
        ransac_plane_fit(&surface_points, ransac_inlier_threshold, ransac_iterations)?;

    // Orient normal to point outward (query point is inside)
    let oriented_normal = orient_outward(normal, centroid, point);

    Some(RobustSurfaceResult {
        normal: oriented_normal,
        point_on_surface: centroid,
        inlier_count: inliers.len(),
        total_points: surface_points.len(),
        residual,
    })
}

/// RANSAC plane fitting
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
        let idx1 = (i * 7 + 1) % n;
        let idx2 = (i * 13 + 3) % n;
        let idx3 = (i * 19 + 7) % n;

        if idx1 == idx2 || idx2 == idx3 || idx1 == idx3 {
            continue;
        }

        let p1 = points[idx1];
        let p2 = points[idx2];
        let p3 = points[idx3];

        let v1 = sub(p2, p1);
        let v2 = sub(p3, p1);
        let normal = normalize(cross(v1, v2));

        if length(normal) < 1e-6 {
            continue;
        }

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

    let (centroid, normal, residual) = fit_plane_svd(&best_inliers);
    Some((best_inliers, centroid, normal, residual))
}

fn fit_plane_svd(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
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

    let max_ev = cov.iter().enumerate()
        .map(|(i, row)| row[i].abs() + row.iter().enumerate()
            .filter(|&(j, _)| j != i).map(|(_, &v)| v.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    let shift = max_ev + 1.0;
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
        if len < 1e-12 { break; }
        v = (new_v.0 / len, new_v.1 / len, new_v.2 / len);
    }

    let mut residual_sum = 0.0;
    for p in &centered {
        residual_sum += dot(*p, v).powi(2);
    }
    let rms = (residual_sum / points.len() as f64).sqrt();

    (centroid, v, rms)
}

/// Orient normal to point outward (away from the inside query point)
fn orient_outward(normal: (f64, f64, f64), surface_point: (f64, f64, f64), inside_point: (f64, f64, f64)) -> (f64, f64, f64) {
    // Vector from surface to inside point
    let to_inside = sub(inside_point, surface_point);
    // Normal should point opposite to "toward inside" direction
    if dot(normal, to_inside) > 0.0 { neg(normal) } else { normal }
}

fn generate_sphere_directions(n: usize) -> Vec<(f64, f64, f64)> {
    let mut dirs = Vec::with_capacity(n);
    let phi = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt());
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let theta = (1.0 - 2.0 * t).acos();
        let angle = phi * i as f64;
        dirs.push((theta.sin() * angle.cos(), theta.sin() * angle.sin(), theta.cos()));
    }
    dirs
}

fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 { a.0*b.0 + a.1*b.1 + a.2*b.2 }
fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.1*b.2 - a.2*b.1, a.2*b.0 - a.0*b.2, a.0*b.1 - a.1*b.0)
}
fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) { (a.0-b.0, a.1-b.1, a.2-b.2) }
fn neg(v: (f64, f64, f64)) -> (f64, f64, f64) { (-v.0, -v.1, -v.2) }
fn length(v: (f64, f64, f64)) -> f64 { (v.0*v.0 + v.1*v.1 + v.2*v.2).sqrt() }
fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let l = length(v);
    if l > 1e-12 { (v.0/l, v.1/l, v.2/l) } else { (0.0, 0.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::analytical_cube::AnalyticalRotatedCube;

    fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
        dot(normalize(a), normalize(b)).clamp(-1.0, 1.0).acos()
    }

    #[test]
    fn benchmark_robust_surface() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let cube_for_sampler = cube.clone();
        let sampler = move |x: f64, y: f64, z: f64| -> f32 {
            let local = cube_for_sampler.world_to_local((x, y, z));
            let h = 0.5;
            if local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h { 1.0 } else { -1.0 }
        };

        println!("\n{}", "=".repeat(70));
        println!("ROBUST RANSAC SURFACE DETECTION");
        println!("{}\n", "=".repeat(70));

        let configs = [
            (100, 0.02, 100, "100 probes, 0.02 threshold"),
            (100, 0.01, 100, "100 probes, 0.01 threshold"),
            (150, 0.01, 150, "150 probes, 0.01 threshold"),
            (150, 0.005, 150, "150 probes, 0.005 threshold"),
        ];

        for (probes, threshold, iters, desc) in &configs {
            println!("Config: {}", desc);
            println!("{:<12} {:>10} {:>10} {:>12}", "Face", "Error°", "Inliers", "Residual");
            println!("{}", "-".repeat(50));

            let mut errors = Vec::new();

            for face_idx in 0..6 {
                let expected_normal = cube.face_normals[face_idx];
                // Face center is at normal * 0.5 (half-width of cube)
                // Test point 0.1 inside from face center = normal * 0.4
                let test_point = (
                    expected_normal.0 * 0.4,
                    expected_normal.1 * 0.4,
                    expected_normal.2 * 0.4,
                );

                let cache = SampleCache::new(&sampler);
                if let Some(result) = robust_surface_detection(
                    test_point, &cache, *probes, 0.6, *threshold, *iters
                ) {
                    let err = angle_between(result.normal, expected_normal).to_degrees();
                    println!("{:<12} {:>10.3} {:>6}/{:<6} {:>10.6}",
                             format!("Face {}", face_idx), err,
                             result.inlier_count, result.total_points, result.residual);
                    errors.push(err);
                } else {
                    println!("{:<12} {:>10}", format!("Face {}", face_idx), "FAIL");
                }
            }

            if !errors.is_empty() {
                let avg = errors.iter().sum::<f64>() / errors.len() as f64;
                let max = errors.iter().cloned().fold(0.0_f64, f64::max);
                println!("{}", "-".repeat(50));
                println!("Avg: {:.3}°, Max: {:.3}°\n", avg, max);
            }
        }
    }
}
