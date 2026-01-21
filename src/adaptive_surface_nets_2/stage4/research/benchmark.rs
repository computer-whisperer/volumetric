//! Benchmark comparing edge detection approaches
//!
//! Run with: cargo test --features native benchmark_edge_detection -- --nocapture

use super::analytical_cube::{angle_between, AnalyticalRotatedCube};
use super::improved_reference::{
    edge_detection_gradient, edge_detection_normal_clustering, edge_detection_ransac,
    edge_detection_two_pass,
};
use super::reference_edge::{reference_find_nearest_edge, EdgeFindingConfig};
use super::sample_cache::SampleCache;

/// Results for one edge detection approach
#[derive(Debug)]
pub struct ApproachResults {
    pub name: String,
    pub successes: usize,
    pub failures: usize,
    pub direction_errors: Vec<f64>,
    pub normal_a_errors: Vec<f64>,
    pub normal_b_errors: Vec<f64>,
    pub position_errors: Vec<f64>,
    pub samples_used: Vec<u64>,
}

impl ApproachResults {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            successes: 0,
            failures: 0,
            direction_errors: Vec::new(),
            normal_a_errors: Vec::new(),
            normal_b_errors: Vec::new(),
            position_errors: Vec::new(),
            samples_used: Vec::new(),
        }
    }

    fn avg_direction_error(&self) -> f64 {
        if self.direction_errors.is_empty() {
            f64::NAN
        } else {
            self.direction_errors.iter().sum::<f64>() / self.direction_errors.len() as f64
        }
    }

    fn avg_normal_error(&self) -> f64 {
        let all: Vec<_> = self
            .normal_a_errors
            .iter()
            .chain(self.normal_b_errors.iter())
            .cloned()
            .collect();
        if all.is_empty() {
            f64::NAN
        } else {
            all.iter().sum::<f64>() / all.len() as f64
        }
    }

    fn max_normal_error(&self) -> f64 {
        self.normal_a_errors
            .iter()
            .chain(self.normal_b_errors.iter())
            .cloned()
            .fold(0.0_f64, f64::max)
    }

    fn avg_samples(&self) -> f64 {
        if self.samples_used.is_empty() {
            f64::NAN
        } else {
            self.samples_used.iter().sum::<u64>() as f64 / self.samples_used.len() as f64
        }
    }
}

fn rotated_cube_sampler(cube: &AnalyticalRotatedCube) -> impl Fn(f64, f64, f64) -> f32 + '_ {
    move |x, y, z| {
        let local = cube.world_to_local((x, y, z));
        let h = 0.5;
        let inside = local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h;
        if inside {
            1.0
        } else {
            -1.0
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

fn distance_to_line(p: (f64, f64, f64), a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
    let ap = (p.0 - a.0, p.1 - a.1, p.2 - a.2);
    let t = (ap.0 * ab.0 + ap.1 * ab.1 + ap.2 * ab.2) / (ab.0 * ab.0 + ab.1 * ab.1 + ab.2 * ab.2);
    let t_clamped = t.clamp(0.0, 1.0);
    let closest = (
        a.0 + t_clamped * ab.0,
        a.1 + t_clamped * ab.1,
        a.2 + t_clamped * ab.2,
    );
    ((p.0 - closest.0).powi(2) + (p.1 - closest.1).powi(2) + (p.2 - closest.2).powi(2)).sqrt()
}

fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 1.0)
    }
}

/// Run the benchmark comparing all edge detection approaches
pub fn run_edge_detection_benchmark() {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let sampler = rotated_cube_sampler(&cube);

    println!("\n{}", "=".repeat(70));
    println!("EDGE DETECTION BENCHMARK - Rotated Cube");
    println!("{}\n", "=".repeat(70));

    let mut results: Vec<ApproachResults> = vec![
        ApproachResults::new("Original (k-means position)"),
        ApproachResults::new("Normal Clustering"),
        ApproachResults::new("RANSAC"),
        ApproachResults::new("Gradient Threshold"),
        ApproachResults::new("Two-Pass (Grad+Plane)"),
    ];

    // Test each of the 12 edges
    for edge_idx in 0..12 {
        let edge = cube.get_edge(edge_idx);
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

        // Test each approach
        // Approach 0: Original k-means
        {
            let cache = SampleCache::new(&sampler);
            let config = EdgeFindingConfig {
                num_probes: 150,
                max_search_distance: 0.5,
                search_step: 0.01,
                binary_iterations: 20,
                kmeans_iterations: 30,
            };

            if let Some(result) = reference_find_nearest_edge(test_point, &cache, &config) {
                let dir_err1 = angle_between(result.edge_direction, edge.direction).to_degrees();
                let dir_err2 = angle_between(
                    (-result.edge_direction.0, -result.edge_direction.1, -result.edge_direction.2),
                    edge.direction,
                )
                .to_degrees();
                let dir_err = dir_err1.min(dir_err2);

                let (na_err, nb_err) = best_normal_pairing(
                    (result.face_a_normal, result.face_b_normal),
                    (edge.face_a_normal, edge.face_b_normal),
                );

                let pos_err = distance_to_line(result.point_on_edge, edge.endpoints.0, edge.endpoints.1);

                results[0].successes += 1;
                results[0].direction_errors.push(dir_err);
                results[0].normal_a_errors.push(na_err);
                results[0].normal_b_errors.push(nb_err);
                results[0].position_errors.push(pos_err);
                results[0].samples_used.push(cache.stats().actual_samples());
            } else {
                results[0].failures += 1;
            }
        }

        // Approach 1: Normal Clustering
        {
            let cache = SampleCache::new(&sampler);
            if let Some(result) =
                edge_detection_normal_clustering(test_point, &cache, 150, 0.5, 0.01)
            {
                let dir_err1 = angle_between(result.edge_direction, edge.direction).to_degrees();
                let dir_err2 = angle_between(
                    (-result.edge_direction.0, -result.edge_direction.1, -result.edge_direction.2),
                    edge.direction,
                )
                .to_degrees();
                let dir_err = dir_err1.min(dir_err2);

                let (na_err, nb_err) = best_normal_pairing(
                    (result.face_a_normal, result.face_b_normal),
                    (edge.face_a_normal, edge.face_b_normal),
                );

                let pos_err = distance_to_line(result.point_on_edge, edge.endpoints.0, edge.endpoints.1);

                results[1].successes += 1;
                results[1].direction_errors.push(dir_err);
                results[1].normal_a_errors.push(na_err);
                results[1].normal_b_errors.push(nb_err);
                results[1].position_errors.push(pos_err);
                results[1].samples_used.push(cache.stats().actual_samples());
            } else {
                results[1].failures += 1;
            }
        }

        // Approach 2: RANSAC
        {
            let cache = SampleCache::new(&sampler);
            if let Some(result) = edge_detection_ransac(test_point, &cache, 150, 0.5, 0.02, 100) {
                let dir_err1 = angle_between(result.edge_direction, edge.direction).to_degrees();
                let dir_err2 = angle_between(
                    (-result.edge_direction.0, -result.edge_direction.1, -result.edge_direction.2),
                    edge.direction,
                )
                .to_degrees();
                let dir_err = dir_err1.min(dir_err2);

                let (na_err, nb_err) = best_normal_pairing(
                    (result.face_a_normal, result.face_b_normal),
                    (edge.face_a_normal, edge.face_b_normal),
                );

                let pos_err = distance_to_line(result.point_on_edge, edge.endpoints.0, edge.endpoints.1);

                results[2].successes += 1;
                results[2].direction_errors.push(dir_err);
                results[2].normal_a_errors.push(na_err);
                results[2].normal_b_errors.push(nb_err);
                results[2].position_errors.push(pos_err);
                results[2].samples_used.push(cache.stats().actual_samples());
            } else {
                results[2].failures += 1;
            }
        }

        // Approach 3: Gradient Threshold
        {
            let cache = SampleCache::new(&sampler);
            // 30 degrees threshold for merging normals
            if let Some(result) =
                edge_detection_gradient(test_point, &cache, 150, 0.5, 0.01, 0.52)
            {
                let dir_err1 = angle_between(result.edge_direction, edge.direction).to_degrees();
                let dir_err2 = angle_between(
                    (-result.edge_direction.0, -result.edge_direction.1, -result.edge_direction.2),
                    edge.direction,
                )
                .to_degrees();
                let dir_err = dir_err1.min(dir_err2);

                let (na_err, nb_err) = best_normal_pairing(
                    (result.face_a_normal, result.face_b_normal),
                    (edge.face_a_normal, edge.face_b_normal),
                );

                let pos_err = distance_to_line(result.point_on_edge, edge.endpoints.0, edge.endpoints.1);

                results[3].successes += 1;
                results[3].direction_errors.push(dir_err);
                results[3].normal_a_errors.push(na_err);
                results[3].normal_b_errors.push(nb_err);
                results[3].position_errors.push(pos_err);
                results[3].samples_used.push(cache.stats().actual_samples());
            } else {
                results[3].failures += 1;
            }
        }

        // Approach 4: Two-Pass
        {
            let cache = SampleCache::new(&sampler);
            if let Some(result) = edge_detection_two_pass(test_point, &cache, 150, 0.5, 0.01) {
                let dir_err1 = angle_between(result.edge_direction, edge.direction).to_degrees();
                let dir_err2 = angle_between(
                    (-result.edge_direction.0, -result.edge_direction.1, -result.edge_direction.2),
                    edge.direction,
                )
                .to_degrees();
                let dir_err = dir_err1.min(dir_err2);

                let (na_err, nb_err) = best_normal_pairing(
                    (result.face_a_normal, result.face_b_normal),
                    (edge.face_a_normal, edge.face_b_normal),
                );

                let pos_err = distance_to_line(result.point_on_edge, edge.endpoints.0, edge.endpoints.1);

                results[4].successes += 1;
                results[4].direction_errors.push(dir_err);
                results[4].normal_a_errors.push(na_err);
                results[4].normal_b_errors.push(nb_err);
                results[4].position_errors.push(pos_err);
                results[4].samples_used.push(cache.stats().actual_samples());
            } else {
                results[4].failures += 1;
            }
        }
    }

    // Print results
    println!("RESULTS (12 edges tested, offset 0.1 inside from edge midpoint)");
    println!("{}", "-".repeat(90));
    println!(
        "{:<28} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Approach", "Success", "Avg Dir°", "Avg Nrm°", "Max Nrm°", "Samples"
    );
    println!("{}", "-".repeat(90));

    for r in &results {
        println!(
            "{:<28} {:>5}/{:<2} {:>10.2} {:>10.2} {:>10.2} {:>10.0}",
            r.name,
            r.successes,
            r.successes + r.failures,
            r.avg_direction_error(),
            r.avg_normal_error(),
            r.max_normal_error(),
            r.avg_samples()
        );
    }

    println!("{}", "-".repeat(90));
    println!("\nTarget: Direction error < 1°, Normal error < 1°");

    // Print detailed per-edge results for the best approach
    let best_idx = results
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.avg_normal_error()
                .partial_cmp(&b.avg_normal_error())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("\nBest approach: {}", results[best_idx].name);
    println!(
        "  Avg normal error: {:.2}°",
        results[best_idx].avg_normal_error()
    );
    println!(
        "  Avg direction error: {:.2}°",
        results[best_idx].avg_direction_error()
    );

    // Check if any approach meets the target
    let meets_target = results
        .iter()
        .any(|r| r.avg_normal_error() < 1.0 && r.avg_direction_error() < 1.0);
    println!(
        "\nAny approach meets <1° target: {}",
        if meets_target { "YES" } else { "NO" }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_edge_detection() {
        run_edge_detection_benchmark();
    }
}
