//! Optimization experiments for the gradient-based edge detection
//!
//! The gradient threshold approach was the best in benchmarks (~6.5° error).
//! This module explores parameter tuning and algorithmic improvements.

use super::analytical_cube::{angle_between, AnalyticalRotatedCube};
use super::improved_reference::compute_gradient_normal;
use super::sample_cache::{find_crossing_in_direction, SampleCache};

/// Optimized gradient-based edge detection with refined clustering
#[derive(Clone, Debug)]
pub struct OptimizedGradientResult {
    pub edge_direction: (f64, f64, f64),
    pub point_on_edge: (f64, f64, f64),
    pub face_a_normal: (f64, f64, f64),
    pub face_b_normal: (f64, f64, f64),
    pub cluster_a_size: usize,
    pub cluster_b_size: usize,
}

/// Mean-shift clustering for normals on the unit sphere
///
/// Unlike greedy clustering, this finds the actual cluster centers
/// by iteratively shifting toward higher density regions.
pub fn edge_detection_meanshift<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    num_probes: usize,
    max_distance: f64,
    gradient_epsilon: f64,
    bandwidth: f64, // Angular bandwidth in radians for mean-shift kernel
) -> Option<OptimizedGradientResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Step 1: Find surface points with their normals
    let directions = generate_sphere_directions(num_probes);
    let mut point_normals: Vec<((f64, f64, f64), (f64, f64, f64))> = Vec::new();

    for dir in &directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache, point, *dir, max_distance, 0.005, 25,
        ) {
            let normal = compute_gradient_normal(cache, crossing, gradient_epsilon);
            let to_query = sub(point, crossing);
            let oriented = if dot(normal, to_query) < 0.0 {
                normal
            } else {
                neg(normal)
            };
            point_normals.push((crossing, oriented));
        }
    }

    if point_normals.len() < 6 {
        return None;
    }

    // Step 2: Mean-shift clustering on unit sphere
    let normals: Vec<_> = point_normals.iter().map(|(_, n)| *n).collect();
    let cluster_assignments = spherical_meanshift(&normals, bandwidth, 50);

    // Find the two most common cluster IDs
    let mut cluster_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &c in &cluster_assignments {
        *cluster_counts.entry(c).or_insert(0) += 1;
    }

    let mut counts: Vec<_> = cluster_counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));

    if counts.len() < 2 {
        return None;
    }

    let cluster_id_a = counts[0].0;
    let cluster_id_b = counts[1].0;

    // Separate points into clusters
    let cluster_a: Vec<_> = point_normals
        .iter()
        .zip(cluster_assignments.iter())
        .filter(|(_, c)| **c == cluster_id_a)
        .map(|(pn, _)| *pn)
        .collect();

    let cluster_b: Vec<_> = point_normals
        .iter()
        .zip(cluster_assignments.iter())
        .filter(|(_, c)| **c == cluster_id_b)
        .map(|(pn, _)| *pn)
        .collect();

    if cluster_a.len() < 3 || cluster_b.len() < 3 {
        return None;
    }

    // Step 3: Compute face normals as mean of cluster gradient normals
    let normal_a = mean_direction(&cluster_a.iter().map(|(_, n)| *n).collect::<Vec<_>>());
    let normal_b = mean_direction(&cluster_b.iter().map(|(_, n)| *n).collect::<Vec<_>>());

    // Edge direction
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None;
    }

    // Find point on edge
    let centroid_a = centroid(&cluster_a.iter().map(|(p, _)| *p).collect::<Vec<_>>());
    let centroid_b = centroid(&cluster_b.iter().map(|(p, _)| *p).collect::<Vec<_>>());
    let point_on_edge = find_edge_point(point, centroid_a, normal_a, centroid_b, normal_b, edge_dir);

    Some(OptimizedGradientResult {
        edge_direction: edge_dir,
        point_on_edge,
        face_a_normal: normal_a,
        face_b_normal: normal_b,
        cluster_a_size: cluster_a.len(),
        cluster_b_size: cluster_b.len(),
    })
}

/// Spherical mean-shift clustering
/// Returns cluster assignment for each input normal
fn spherical_meanshift(normals: &[(f64, f64, f64)], bandwidth: f64, max_iterations: usize) -> Vec<usize> {
    let n = normals.len();
    let mut modes: Vec<(f64, f64, f64)> = normals.to_vec();

    // Shift each point toward local mode
    for _ in 0..max_iterations {
        let mut converged = true;

        for i in 0..n {
            let old_mode = modes[i];

            // Compute weighted mean of nearby normals
            let mut sum = (0.0, 0.0, 0.0);
            let mut weight_sum = 0.0;

            for &normal in normals {
                let angle = angle_between(old_mode, normal);
                if angle < bandwidth {
                    // Gaussian kernel
                    let weight = (-angle * angle / (2.0 * bandwidth * bandwidth)).exp();
                    sum.0 += normal.0 * weight;
                    sum.1 += normal.1 * weight;
                    sum.2 += normal.2 * weight;
                    weight_sum += weight;
                }
            }

            if weight_sum > 1e-10 {
                let new_mode = normalize((sum.0 / weight_sum, sum.1 / weight_sum, sum.2 / weight_sum));
                let shift = angle_between(old_mode, new_mode);
                if shift > 0.001 {
                    converged = false;
                }
                modes[i] = new_mode;
            }
        }

        if converged {
            break;
        }
    }

    // Assign cluster IDs: modes within bandwidth are same cluster
    let mut cluster_ids = vec![usize::MAX; n];
    let mut next_cluster = 0;

    for i in 0..n {
        if cluster_ids[i] != usize::MAX {
            continue;
        }

        cluster_ids[i] = next_cluster;

        // Find all other points with similar mode
        for j in (i + 1)..n {
            if cluster_ids[j] == usize::MAX {
                let angle = angle_between(modes[i], modes[j]);
                if angle < bandwidth {
                    cluster_ids[j] = next_cluster;
                }
            }
        }

        next_cluster += 1;
    }

    cluster_ids
}

/// Higher precision edge detection using more samples and smaller epsilon
pub fn edge_detection_high_precision<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
) -> Option<OptimizedGradientResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // High precision settings
    edge_detection_meanshift(
        point,
        cache,
        300,   // More probes
        0.5,   // Search distance
        0.005, // Smaller gradient epsilon
        0.3,   // ~17° bandwidth for mean-shift
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

fn mean_direction(normals: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let sum = normals.iter().fold((0.0, 0.0, 0.0), |acc, n| {
        (acc.0 + n.0, acc.1 + n.1, acc.2 + n.2)
    });
    normalize(sum)
}

fn centroid(points: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let n = points.len() as f64;
    (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    )
}

fn find_edge_point(
    query: (f64, f64, f64),
    centroid_a: (f64, f64, f64),
    normal_a: (f64, f64, f64),
    centroid_b: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
) -> (f64, f64, f64) {
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

    let to_query = sub(query, p);
    let t = dot(to_query, edge_direction);
    (
        p.0 + t * edge_direction.0,
        p.1 + t * edge_direction.1,
        p.2 + t * edge_direction.2,
    )
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

/// Parameter sweep to find optimal settings
pub fn run_parameter_sweep() {
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
    println!("PARAMETER SWEEP - Gradient Edge Detection");
    println!("{}\n", "=".repeat(80));

    // Parameter combinations to test
    let probe_counts = [100, 200, 300];
    let epsilons = [0.02, 0.01, 0.005];
    let bandwidths = [0.5, 0.4, 0.3, 0.25]; // radians

    println!(
        "{:<8} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "Probes", "Epsilon", "Bandwidth°", "Avg Nrm°", "Max Nrm°", "Samples"
    );
    println!("{}", "-".repeat(80));

    let mut best_avg = f64::MAX;
    let mut best_config = (0, 0.0, 0.0);

    for &probes in &probe_counts {
        for &eps in &epsilons {
            for &bw in &bandwidths {
                let mut normal_errors = Vec::new();
                let mut total_samples = 0u64;

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

                    if let Some(result) = edge_detection_meanshift(
                        test_point, &cache, probes, 0.5, eps, bw,
                    ) {
                        let (na_err, nb_err) = best_normal_pairing(
                            (result.face_a_normal, result.face_b_normal),
                            (edge.face_a_normal, edge.face_b_normal),
                        );
                        normal_errors.push(na_err);
                        normal_errors.push(nb_err);
                    }

                    total_samples += cache.stats().actual_samples();
                }

                if !normal_errors.is_empty() {
                    let avg = normal_errors.iter().sum::<f64>() / normal_errors.len() as f64;
                    let max = normal_errors.iter().cloned().fold(0.0_f64, f64::max);
                    let avg_samples = total_samples as f64 / 12.0;

                    println!(
                        "{:<8} {:>10.3} {:>12.1} {:>12.2} {:>12.2} {:>10.0}",
                        probes, eps, bw.to_degrees(), avg, max, avg_samples
                    );

                    if avg < best_avg {
                        best_avg = avg;
                        best_config = (probes, eps, bw);
                    }
                }
            }
        }
    }

    println!("{}", "-".repeat(80));
    println!(
        "Best config: {} probes, eps={}, bandwidth={:.1}°",
        best_config.0,
        best_config.1,
        best_config.2.to_degrees()
    );
    println!("Best avg normal error: {:.2}°", best_avg);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_sweep() {
        run_parameter_sweep();
    }
}
