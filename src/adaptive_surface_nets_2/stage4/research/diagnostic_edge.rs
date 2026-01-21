//! Detailed diagnostic for edge detection error sources
//!
//! Investigates WHY the error is ~6.5° and whether it's fundamental.

use super::analytical_cube::{angle_between, AnalyticalRotatedCube};
use super::improved_reference::compute_gradient_normal;
use super::sample_cache::{find_crossing_in_direction, SampleCache};

/// Diagnose individual edge detection, showing where error comes from
pub fn diagnose_edge_detection() {
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

    println!("\n{}", "=".repeat(90));
    println!("EDGE DETECTION DIAGNOSTIC - Understanding the 6.5° Error");
    println!("{}\n", "=".repeat(90));

    // Pick edge 0 for detailed analysis
    let edge = cube.get_edge(0);
    println!("Analyzing Edge 0:");
    println!("  Direction: ({:.4}, {:.4}, {:.4})", edge.direction.0, edge.direction.1, edge.direction.2);
    println!("  Face A normal: ({:.4}, {:.4}, {:.4})", edge.face_a_normal.0, edge.face_a_normal.1, edge.face_a_normal.2);
    println!("  Face B normal: ({:.4}, {:.4}, {:.4})", edge.face_b_normal.0, edge.face_b_normal.1, edge.face_b_normal.2);
    println!();

    // Test point near edge
    let bisector = normalize((
        edge.face_a_normal.0 + edge.face_b_normal.0,
        edge.face_a_normal.1 + edge.face_b_normal.1,
        edge.face_a_normal.2 + edge.face_b_normal.2,
    ));
    let offset = 0.1;
    let test_point = (
        edge.point_on_edge.0 - bisector.0 * offset,
        edge.point_on_edge.1 - bisector.1 * offset,
        edge.point_on_edge.2 - bisector.2 * offset,
    );

    println!("Test point: ({:.4}, {:.4}, {:.4}) (0.1 inside from edge midpoint)",
             test_point.0, test_point.1, test_point.2);
    println!();

    // Find surface points and analyze their gradient normals
    let cache = SampleCache::new(&sampler);
    let directions = generate_sphere_directions(150);

    println!("Surface points and their gradient normals:");
    println!("{:<50} {:>10} {:>15}", "Position", "Exp Face", "Gradient Err°");
    println!("{}", "-".repeat(80));

    let mut face_a_points: Vec<((f64, f64, f64), (f64, f64, f64))> = Vec::new();
    let mut face_b_points: Vec<((f64, f64, f64), (f64, f64, f64))> = Vec::new();
    let mut face_a_errors = Vec::new();
    let mut face_b_errors = Vec::new();

    for (i, dir) in directions.iter().enumerate() {
        if let Some((crossing, _)) = find_crossing_in_direction(
            &cache, test_point, *dir, 0.5, 0.001, 30,
        ) {
            // Compute gradient at surface point
            let eps = 0.01;
            let grad_normal = compute_gradient_normal(&cache, crossing, eps);

            // Orient away from query point
            let to_query = sub(test_point, crossing);
            let grad_normal = if dot(grad_normal, to_query) < 0.0 {
                grad_normal
            } else {
                neg(grad_normal)
            };

            // Determine which face this belongs to (analytically)
            let err_a = angle_between(grad_normal, edge.face_a_normal).to_degrees();
            let err_b = angle_between(grad_normal, edge.face_b_normal).to_degrees();

            if err_a < err_b {
                face_a_points.push((crossing, grad_normal));
                face_a_errors.push(err_a);
                if i < 10 || err_a > 15.0 {
                    println!(
                        "({:6.3},{:6.3},{:6.3}) grad=({:6.3},{:6.3},{:6.3}) {:>10} {:>12.2}°{}",
                        crossing.0, crossing.1, crossing.2,
                        grad_normal.0, grad_normal.1, grad_normal.2,
                        "Face A",
                        err_a,
                        if err_a > 15.0 { " <-- HIGH" } else { "" }
                    );
                }
            } else {
                face_b_points.push((crossing, grad_normal));
                face_b_errors.push(err_b);
                if i < 10 || err_b > 15.0 {
                    println!(
                        "({:6.3},{:6.3},{:6.3}) grad=({:6.3},{:6.3},{:6.3}) {:>10} {:>12.2}°{}",
                        crossing.0, crossing.1, crossing.2,
                        grad_normal.0, grad_normal.1, grad_normal.2,
                        "Face B",
                        err_b,
                        if err_b > 15.0 { " <-- HIGH" } else { "" }
                    );
                }
            }
        }
    }

    println!("{}", "-".repeat(80));

    // Analysis
    println!("\nAnalysis:");
    println!("  Face A: {} points", face_a_points.len());
    if !face_a_errors.is_empty() {
        let avg_a: f64 = face_a_errors.iter().sum::<f64>() / face_a_errors.len() as f64;
        let max_a: f64 = face_a_errors.iter().cloned().fold(0.0_f64, f64::max);
        println!("    Avg gradient error: {:.2}°", avg_a);
        println!("    Max gradient error: {:.2}°", max_a);

        // Mean gradient direction
        let mean_a = mean_direction(&face_a_points.iter().map(|(_, n)| *n).collect::<Vec<_>>());
        let mean_err_a = angle_between(mean_a, edge.face_a_normal).to_degrees();
        println!("    Mean gradient: ({:.4}, {:.4}, {:.4})", mean_a.0, mean_a.1, mean_a.2);
        println!("    Error of mean: {:.2}°", mean_err_a);
    }

    println!("  Face B: {} points", face_b_points.len());
    if !face_b_errors.is_empty() {
        let avg_b: f64 = face_b_errors.iter().sum::<f64>() / face_b_errors.len() as f64;
        let max_b: f64 = face_b_errors.iter().cloned().fold(0.0_f64, f64::max);
        println!("    Avg gradient error: {:.2}°", avg_b);
        println!("    Max gradient error: {:.2}°", max_b);

        let mean_b = mean_direction(&face_b_points.iter().map(|(_, n)| *n).collect::<Vec<_>>());
        let mean_err_b = angle_between(mean_b, edge.face_b_normal).to_degrees();
        println!("    Mean gradient: ({:.4}, {:.4}, {:.4})", mean_b.0, mean_b.1, mean_b.2);
        println!("    Error of mean: {:.2}°", mean_err_b);
    }

    // Now try plane fitting instead of gradient averaging
    if !face_a_points.is_empty() && !face_b_points.is_empty() {
        println!("\nPlane Fitting Comparison:");

        let points_a: Vec<_> = face_a_points.iter().map(|(p, _)| *p).collect();
        let points_b: Vec<_> = face_b_points.iter().map(|(p, _)| *p).collect();

        let (_, plane_normal_a, residual_a) = fit_plane(&points_a);
        let (_, plane_normal_b, residual_b) = fit_plane(&points_b);

        // Orient plane normals
        let plane_normal_a = if dot(plane_normal_a, edge.face_a_normal) > 0.0 {
            plane_normal_a
        } else {
            neg(plane_normal_a)
        };
        let plane_normal_b = if dot(plane_normal_b, edge.face_b_normal) > 0.0 {
            plane_normal_b
        } else {
            neg(plane_normal_b)
        };

        let plane_err_a = angle_between(plane_normal_a, edge.face_a_normal).to_degrees();
        let plane_err_b = angle_between(plane_normal_b, edge.face_b_normal).to_degrees();

        println!("  Face A plane fit:");
        println!("    Normal: ({:.4}, {:.4}, {:.4})", plane_normal_a.0, plane_normal_a.1, plane_normal_a.2);
        println!("    Error: {:.2}° (vs {:.2}° from gradient mean)", plane_err_a,
                 angle_between(mean_direction(&face_a_points.iter().map(|(_, n)| *n).collect::<Vec<_>>()),
                              edge.face_a_normal).to_degrees());
        println!("    Residual: {:.6}", residual_a);

        println!("  Face B plane fit:");
        println!("    Normal: ({:.4}, {:.4}, {:.4})", plane_normal_b.0, plane_normal_b.1, plane_normal_b.2);
        println!("    Error: {:.2}° (vs {:.2}° from gradient mean)", plane_err_b,
                 angle_between(mean_direction(&face_b_points.iter().map(|(_, n)| *n).collect::<Vec<_>>()),
                              edge.face_b_normal).to_degrees());
        println!("    Residual: {:.6}", residual_b);
    }

    // Check how gradient eps affects accuracy
    println!("\nGradient Epsilon Effect:");
    let epsilons = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

    // Pick a single surface point on face A for testing
    if let Some((crossing, _)) = find_crossing_in_direction(
        &cache, test_point, edge.face_a_normal, 0.5, 0.0001, 50,
    ) {
        println!("  Testing at surface point: ({:.4}, {:.4}, {:.4})", crossing.0, crossing.1, crossing.2);
        println!("  Expected normal: ({:.4}, {:.4}, {:.4})", edge.face_a_normal.0, edge.face_a_normal.1, edge.face_a_normal.2);
        println!();

        println!("{:>10} {:>12} {:>30}", "Epsilon", "Error°", "Gradient");
        println!("{}", "-".repeat(55));

        for &eps in &epsilons {
            let grad = compute_gradient_normal(&cache, crossing, eps);
            // Orient
            let grad = if dot(grad, edge.face_a_normal) > 0.0 { grad } else { neg(grad) };
            let err = angle_between(grad, edge.face_a_normal).to_degrees();
            println!("{:>10.4} {:>12.2} ({:>8.4}, {:>8.4}, {:>8.4})",
                     eps, err, grad.0, grad.1, grad.2);
        }
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

fn mean_direction(normals: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let sum = normals.iter().fold((0.0, 0.0, 0.0), |acc, n| {
        (acc.0 + n.0, acc.1 + n.1, acc.2 + n.2)
    });
    normalize(sum)
}

fn fit_plane(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
    if points.len() < 3 {
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), f64::INFINITY);
    }

    let n = points.len() as f64;
    let center = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    );

    let centered: Vec<_> = points.iter().map(|p| sub(*p, center)).collect();

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

    // Power iteration to find smallest eigenvector
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

    // Compute residual
    let mut residual_sum = 0.0;
    for p in &centered {
        let dist = dot(*p, v).abs();
        residual_sum += dist * dist;
    }
    let rms_residual = (residual_sum / points.len() as f64).sqrt();

    (center, v, rms_residual)
}

fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
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

    #[test]
    fn test_diagnose_edge() {
        diagnose_edge_detection();
    }
}
