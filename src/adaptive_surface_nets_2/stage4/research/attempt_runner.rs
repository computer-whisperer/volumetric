//! Attempt benchmark harness (binary-safe)
//!
//! Runs a candidate vertex processing function against the analytical rotated
//! cube validation points, reporting classification accuracy, normal errors,
//! edge direction errors, and sample counts.

use super::analytical_cube::{angle_between, AnalyticalRotatedCube};
use super::attempt_0::{CrossingCountConfig, GeometryType, VertexGeometry};
use super::sample_cache::SampleCache;
use super::validation::{generate_validation_points, ExpectedClassification, ValidationPoint};

pub fn run_attempt_0_benchmark() {
    let config = CrossingCountConfig::default();
    run_attempt_benchmark("Attempt 0 (crossing count + RANSAC)", |point, hint, sampler| {
        let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
        let before = cache.stats().actual_samples();
        let result = super::attempt_0::process_vertex(point, hint, 1.0, &cache, &config);
        let after = cache.stats().actual_samples();
        (result, after - before)
    });
}

pub fn run_attempt_benchmark<F>(name: &str, attempt: F)
where
    F: Fn((f64, f64, f64), (f64, f64, f64), &dyn Fn(f64, f64, f64) -> f32) -> (VertexGeometry, u64),
{
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points(&cube);

    let mut stats = AttemptStats::new(name);
    for point in &points {
        let hint = hint_from_expected(&point.expected);
        let sampler = |x: f64, y: f64, z: f64| -> f32 {
            let local = cube.world_to_local((x, y, z));
            let h = 0.5;
            if local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h {
                1.0
            } else {
                -1.0
            }
        };
        let (result, samples_used) = attempt(point.position, hint, &sampler);
        stats.samples_used.push(samples_used);
        stats.record(point, &result);
    }

    println!("\n{}", "=".repeat(72));
    println!("ATTEMPT BENCHMARK (Analytical Rotated Cube)");
    println!("{}", "=".repeat(72));
    stats.report();
}

struct AttemptStats {
    name: String,
    face_points: usize,
    edge_points: usize,
    corner_points: usize,
    face_success: usize,
    edge_success: usize,
    corner_success: usize,
    face_errors: Vec<f64>,
    edge_normal_errors: Vec<(f64, f64)>,
    edge_direction_errors: Vec<f64>,
    corner_errors: Vec<Vec<f64>>,
    samples_used: Vec<u64>,
}

impl AttemptStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            face_points: 0,
            edge_points: 0,
            corner_points: 0,
            face_success: 0,
            edge_success: 0,
            corner_success: 0,
            face_errors: Vec::new(),
            edge_normal_errors: Vec::new(),
            edge_direction_errors: Vec::new(),
            corner_errors: Vec::new(),
            samples_used: Vec::new(),
        }
    }

    fn record(&mut self, point: &ValidationPoint, result: &VertexGeometry) {
        match &point.expected {
            ExpectedClassification::OnFace { expected_normal, .. } => {
                self.face_points += 1;
                let err = best_match_error(&result.normals, *expected_normal);
                self.face_errors.push(err);
                if result.classification == GeometryType::Face {
                    self.face_success += 1;
                }
            }
            ExpectedClassification::OnEdge {
                expected_direction,
                expected_normals,
                ..
            } => {
                self.edge_points += 1;
                let (a_err, b_err) = best_normal_pairing(
                    result.normals.get(0).copied(),
                    result.normals.get(1).copied(),
                    *expected_normals,
                );
                self.edge_normal_errors.push((a_err, b_err));
                if let Some(edge_dir) = result.edge_direction {
                    self.edge_direction_errors.push(edge_direction_error(edge_dir, *expected_direction));
                } else {
                    self.edge_direction_errors.push(180.0);
                }
                if result.classification == GeometryType::Edge {
                    self.edge_success += 1;
                }
            }
            ExpectedClassification::OnCorner { expected_normals, .. } => {
                self.corner_points += 1;
                let errs = match_normals_to_expected(&result.normals, expected_normals);
                self.corner_errors.push(errs);
                if result.classification == GeometryType::Corner {
                    self.corner_success += 1;
                }
            }
        }
    }

    fn report(&self) {
        println!("\n{}", self.name);
        println!("  Face success: {}/{}", self.face_success, self.face_points);
        println!("  Edge success: {}/{}", self.edge_success, self.edge_points);
        println!("  Corner success: {}/{}", self.corner_success, self.corner_points);

        if !self.face_errors.is_empty() {
            let avg = self.face_errors.iter().sum::<f64>() / self.face_errors.len() as f64;
            println!("  Face normal avg error: {:.2} deg", avg);
        }

        if !self.edge_normal_errors.is_empty() {
            let avg: f64 = self.edge_normal_errors.iter().map(|(a, b)| a + b).sum::<f64>()
                / (self.edge_normal_errors.len() as f64 * 2.0);
            println!("  Edge normal avg error: {:.2} deg", avg);
        }

        if !self.edge_direction_errors.is_empty() {
            let avg = self.edge_direction_errors.iter().sum::<f64>()
                / self.edge_direction_errors.len() as f64;
            println!("  Edge direction avg error: {:.2} deg", avg);
        }

        if !self.corner_errors.is_empty() {
            let mut all = Vec::new();
            for errs in &self.corner_errors {
                all.extend_from_slice(errs);
            }
            if !all.is_empty() {
                let avg = all.iter().sum::<f64>() / all.len() as f64;
                println!("  Corner normal avg error: {:.2} deg", avg);
            }
        }

        if !self.samples_used.is_empty() {
            let avg = self.samples_used.iter().sum::<u64>() as f64 / self.samples_used.len() as f64;
            println!("  Avg samples used: {:.1}", avg);
        }
    }
}

fn hint_from_expected(expected: &ExpectedClassification) -> (f64, f64, f64) {
    match expected {
        ExpectedClassification::OnFace { expected_normal, .. } => *expected_normal,
        ExpectedClassification::OnEdge { expected_normals, .. } => {
            normalize(add(expected_normals.0, expected_normals.1))
        }
        ExpectedClassification::OnCorner { expected_normals, .. } => {
            let sum = (
                expected_normals[0].0 + expected_normals[1].0 + expected_normals[2].0,
                expected_normals[0].1 + expected_normals[1].1 + expected_normals[2].1,
                expected_normals[0].2 + expected_normals[1].2 + expected_normals[2].2,
            );
            normalize(sum)
        }
    }
}

fn best_match_error(normals: &[(f64, f64, f64)], expected: (f64, f64, f64)) -> f64 {
    if normals.is_empty() {
        return 180.0;
    }
    normals
        .iter()
        .map(|n| angle_between(*n, expected).to_degrees())
        .fold(180.0_f64, f64::min)
}

fn best_normal_pairing(
    detected_a: Option<(f64, f64, f64)>,
    detected_b: Option<(f64, f64, f64)>,
    expected: ((f64, f64, f64), (f64, f64, f64)),
) -> (f64, f64) {
    let Some(a) = detected_a else {
        return (180.0, 180.0);
    };
    let Some(b) = detected_b else {
        return (180.0, 180.0);
    };

    let a1 = (
        angle_between(a, expected.0).to_degrees(),
        angle_between(b, expected.1).to_degrees(),
    );
    let a2 = (
        angle_between(a, expected.1).to_degrees(),
        angle_between(b, expected.0).to_degrees(),
    );
    if a1.0 + a1.1 < a2.0 + a2.1 {
        a1
    } else {
        a2
    }
}

fn match_normals_to_expected(
    normals: &[(f64, f64, f64)],
    expected: &[(f64, f64, f64); 3],
) -> Vec<f64> {
    if normals.len() < 3 {
        return vec![180.0, 180.0, 180.0];
    }
    let perms = [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut best = vec![180.0, 180.0, 180.0];
    for perm in perms {
        let errs = vec![
            angle_between(normals[perm[0]], expected[0]).to_degrees(),
            angle_between(normals[perm[1]], expected[1]).to_degrees(),
            angle_between(normals[perm[2]], expected[2]).to_degrees(),
        ];
        if errs.iter().sum::<f64>() < best.iter().sum::<f64>() {
            best = errs;
        }
    }
    best
}

fn edge_direction_error(dir: (f64, f64, f64), expected: (f64, f64, f64)) -> f64 {
    let e1 = angle_between(dir, expected).to_degrees();
    let e2 = angle_between(neg(dir), expected).to_degrees();
    e1.min(e2)
}

fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = length(v);
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 1.0)
    }
}

fn length(v: (f64, f64, f64)) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

fn neg(v: (f64, f64, f64)) -> (f64, f64, f64) {
    (-v.0, -v.1, -v.2)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_attempt_0_against_reference() {
        run_attempt_0_benchmark();
    }
}
