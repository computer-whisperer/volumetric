//! Attempt benchmark harness (binary-safe)
//!
//! Runs a candidate vertex processing function against the analytical rotated
//! cube validation points, reporting classification accuracy, normal errors,
//! edge direction errors, and sample counts.

use super::analytical_cube::{angle_between, AnalyticalRotatedCube};
use super::oracle::{OracleBenchmarkCase, OracleClassification, OracleHit};
use super::attempt_0::{CrossingCountConfig, GeometryType, VertexGeometry};
use super::attempt_1::{Attempt1Config, Attempt1Diag};
use super::attempt_2::{Attempt2Config, Attempt2Diag};
use super::experiments::hermite_microgrid::{hermite_edge_from_microgrid, HermiteMicrogridConfig};
use super::experiments::ml_policy::run_ml_policy_experiment;
use super::sample_cache::{begin_sample_recording, end_sample_recording, SampleCache};
use crate::sample_cloud::{SampleCloudDump, SampleCloudSet};
use super::validation::{generate_validation_points, ExpectedClassification, ValidationPoint};

pub fn run_attempt_0_benchmark() {
    let config = CrossingCountConfig::default();
    if read_oracle_mode() {
        run_attempt_benchmark_oracle("Attempt 0 (crossing count + RANSAC)", |point, hint, sampler| {
            let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
            let before = cache.stats().actual_samples();
            let result = super::attempt_0::process_vertex(point, hint, 1.0, &cache, &config);
            let after = cache.stats().actual_samples();
            (result, after - before)
        });
    } else {
        run_attempt_benchmark("Attempt 0 (crossing count + RANSAC)", |point, hint, sampler| {
            let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
            let before = cache.stats().actual_samples();
            let result = super::attempt_0::process_vertex(point, hint, 1.0, &cache, &config);
            let after = cache.stats().actual_samples();
            (result, after - before)
        });
    }
}

pub fn run_attempt_1_benchmark() {
    let config = Attempt1Config::default();
    let diag_budget = read_diag_budget();
    if read_oracle_mode() {
        run_attempt_benchmark_oracle("Attempt 1 (adaptive RANSAC + crossing count)", |point, hint, sampler| {
            let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
            let before = cache.stats().actual_samples();
            let result = super::attempt_1::process_vertex(point, hint, 1.0, &cache, &config);
            let after = cache.stats().actual_samples();
            (result, after - before)
        });
    } else if diag_budget > 0 {
        run_attempt_benchmark_with_diag(
            "Attempt 1 (adaptive RANSAC + crossing count)",
            diag_budget,
            |point, hint, sampler| {
                let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
                let before = cache.stats().actual_samples();
                let (result, diag) =
                    super::attempt_1::process_vertex_with_diag(point, hint, 1.0, &cache, &config);
                let after = cache.stats().actual_samples();
                (result, after - before, diag)
            },
        );
    } else {
        run_attempt_benchmark("Attempt 1 (adaptive RANSAC + crossing count)", |point, hint, sampler| {
            let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
            let before = cache.stats().actual_samples();
            let result = super::attempt_1::process_vertex(point, hint, 1.0, &cache, &config);
            let after = cache.stats().actual_samples();
            (result, after - before)
        });
    }
}

pub fn run_attempt_2_benchmark() {
    let config = Attempt2Config::default();
    if read_attempt2_diag_mode() {
        run_attempt_2_benchmark_with_diag(&config);
    } else if read_oracle_mode() {
        run_attempt_benchmark_oracle("Attempt 2 (fixed RANSAC budgets)", |point, hint, sampler| {
            let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
            let before = cache.stats().actual_samples();
            let result = super::attempt_2::process_vertex(point, hint, 1.0, &cache, &config);
            let after = cache.stats().actual_samples();
            (result, after - before)
        });
    } else {
        run_attempt_benchmark("Attempt 2 (fixed RANSAC budgets)", |point, hint, sampler| {
            let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
            let before = cache.stats().actual_samples();
            let result = super::attempt_2::process_vertex(point, hint, 1.0, &cache, &config);
            let after = cache.stats().actual_samples();
            (result, after - before)
        });
    }
}

pub fn dump_attempt_sample_cloud(attempt: u8, output_path: &std::path::Path) {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points(&cube);
    let mut dump = SampleCloudDump::new();

    let attempt_name = match attempt {
        0 => "attempt0",
        1 => "attempt1",
        2 => "attempt2",
        _ => "attempt",
    };

    for (idx, point) in points.iter().enumerate() {
        let sampler = |x: f64, y: f64, z: f64| -> f32 {
            let local = cube.world_to_local((x, y, z));
            let h = 0.5;
            if local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h {
                1.0
            } else {
                -1.0
            }
        };

        let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
        let hint = hint_normal(point.expected.clone());
        begin_sample_recording();

        let result = match attempt {
            0 => {
                let config = CrossingCountConfig::default();
                super::attempt_0::process_vertex(point.position, hint, 1.0, &cache, &config)
            }
            1 => {
                let config = Attempt1Config::default();
                super::attempt_1::process_vertex(point.position, hint, 1.0, &cache, &config)
            }
            2 => {
                let config = Attempt2Config::default();
                super::attempt_2::process_vertex(point.position, hint, 1.0, &cache, &config)
            }
            _ => {
                let config = CrossingCountConfig::default();
                super::attempt_0::process_vertex(point.position, hint, 1.0, &cache, &config)
            }
        };

        let samples = end_sample_recording();
        let mut set = SampleCloudSet::new(idx as u64, to_f32(point.position), to_f32(hint));
        set.label = Some(format!("{}: {}", attempt_name, point.description));
        set.points = samples;
        set.meta.samples_used = Some(result.samples_used);
        dump.add_set(set);
    }

    if let Err(err) = dump.save(output_path) {
        eprintln!("Failed to write sample cloud: {err}");
    } else {
        println!(
            "Wrote {} sample sets to {}",
            dump.sets.len(),
            output_path.display()
        );
    }
}

pub fn run_hermite_microgrid_experiment() {
    let kmeans = HermiteMicrogridConfig::default();
    let mut ransac = HermiteMicrogridConfig::default();
    ransac.fit_strategy = super::experiments::hermite_microgrid::PlaneFitStrategy::Ransac;
    ransac.ransac_iterations = 120;
    ransac.ransac_inlier_threshold = 0.04;
    let mut edge_aligned = HermiteMicrogridConfig::default();
    edge_aligned.fit_strategy = super::experiments::hermite_microgrid::PlaneFitStrategy::EdgeAlignedKMeans;
    let mut line_ransac = HermiteMicrogridConfig::default();
    line_ransac.fit_strategy = super::experiments::hermite_microgrid::PlaneFitStrategy::EdgeAlignedLineRansac;
    line_ransac.ransac_iterations = 160;
    line_ransac.line_inlier_threshold = 0.04;

    run_hermite_microgrid_experiment_with_config("Hermite Microgrid (k-means)", &kmeans);
    run_hermite_microgrid_experiment_with_config("Hermite Microgrid (RANSAC)", &ransac);
    run_hermite_microgrid_experiment_with_config("Hermite Microgrid (edge-aligned k-means)", &edge_aligned);
    run_hermite_microgrid_experiment_with_config("Hermite Microgrid (edge-line RANSAC)", &line_ransac);
}

pub fn run_ml_policy_experiment_runner() {
    run_ml_policy_experiment();
}


fn run_hermite_microgrid_experiment_with_config(
    name: &str,
    config: &HermiteMicrogridConfig,
) {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points(&cube);

    let mut total = 0usize;
    let mut success = 0usize;
    let mut normal_errors = Vec::new();
    let mut direction_errors = Vec::new();
    let mut samples_used = Vec::new();
    let mut crossing_counts = Vec::new();

    for point in &points {
        let ExpectedClassification::OnEdge {
            expected_direction,
            expected_normals,
            ..
        } = point.expected
        else {
            continue;
        };

        total += 1;
        let sampler = |x: f64, y: f64, z: f64| -> f32 {
            let local = cube.world_to_local((x, y, z));
            let h = 0.5;
            if local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h {
                1.0
            } else {
                -1.0
            }
        };

        let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
        let before = cache.stats().actual_samples();
        let result = hermite_edge_from_microgrid(point.position, &cache, config);
        let after = cache.stats().actual_samples();
        samples_used.push(after - before);

        if let Some(result) = result {
            success += 1;
            let (a_err, b_err) = best_normal_pairing(
                Some(result.face_a_normal),
                Some(result.face_b_normal),
                expected_normals,
            );
            normal_errors.push(a_err);
            normal_errors.push(b_err);
            direction_errors.push(edge_direction_error(result.edge_direction, expected_direction));
            crossing_counts.push(result.crossing_points as u64);
        } else {
            normal_errors.push(180.0);
            normal_errors.push(180.0);
            direction_errors.push(180.0);
        }
    }

    println!("\n{}", "=".repeat(72));
    println!("EXPERIMENT ({})", name);
    println!("{}", "=".repeat(72));
    println!("  Edge success: {}/{}", success, total);
    if !normal_errors.is_empty() {
        let avg = normal_errors.iter().sum::<f64>() / normal_errors.len() as f64;
        println!("  Edge normal avg error: {:.2} deg", avg);
    }
    if !direction_errors.is_empty() {
        let avg = direction_errors.iter().sum::<f64>() / direction_errors.len() as f64;
        println!("  Edge direction avg error: {:.2} deg", avg);
    }
    if !samples_used.is_empty() {
        let avg = samples_used.iter().sum::<u64>() as f64 / samples_used.len() as f64;
        println!("  Avg samples used: {:.1}", avg);
    }
    if !crossing_counts.is_empty() {
        let avg = crossing_counts.iter().sum::<u64>() as f64 / crossing_counts.len() as f64;
        println!("  Avg crossing points: {:.1}", avg);
    }
}

fn run_attempt_2_benchmark_with_diag(config: &Attempt2Config) {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points(&cube);

    let mut stats = AttemptStats::new("Attempt 2 (fixed RANSAC budgets)");
    let mut edge_counts = [0usize; 5];
    let mut edge_hint_counts = 0usize;
    let mut edge_diag_samples = 0usize;
    let max_edge_diag = read_attempt2_diag_samples();

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
        let cache = SampleCache::new(|x, y, z| sampler(x, y, z));
        let before = cache.stats().actual_samples();
        let (result, diag) =
            super::attempt_2::process_vertex_with_diag(point.position, hint, 1.0, &cache, config);
        let after = cache.stats().actual_samples();
        stats.samples_used.push(after - before);
        stats.record(point, &result);

        if let ExpectedClassification::OnEdge { .. } = point.expected {
            let idx = diag.crossing_count.min(4);
            edge_counts[idx] += 1;
            if diag.hint_available {
                edge_hint_counts += 1;
            }
            if edge_diag_samples < max_edge_diag {
                edge_diag_samples += 1;
                print_attempt2_diag_sample(&point, &result, &diag);
            }
        }
    }

    println!("\n{}", "=".repeat(72));
    println!("ATTEMPT BENCHMARK (Analytical Rotated Cube)");
    println!("{}", "=".repeat(72));
    stats.report();
    println!(
        "Edge crossing counts: [c0={}, c1={}, c2={}, c3={}, c4={}]",
        edge_counts[0],
        edge_counts[1],
        edge_counts[2],
        edge_counts[3],
        edge_counts[4]
    );
    println!(
        "Edge hint availability: {}/{}",
        edge_hint_counts,
        edge_counts.iter().sum::<usize>()
    );
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

pub fn run_attempt_benchmark_oracle<F>(name: &str, attempt: F)
where
    F: Fn((f64, f64, f64), (f64, f64, f64), &dyn Fn(f64, f64, f64) -> f32) -> (VertexGeometry, u64),
{
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let oracle_case = OracleBenchmarkCase { shape: &cube, seed: 0 };
    let points = oracle_case.points();

    let mut stats = AttemptStats::new(name);
    for point in &points {
        let hint = (0.0, 1.0, 0.0);
        let sampler = |x: f64, y: f64, z: f64| -> f32 {
            let local = cube.world_to_local((x, y, z));
            let h = 0.5;
            if local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h {
                1.0
            } else {
                -1.0
            }
        };
        let (result, samples_used) = attempt(*point, hint, &sampler);
        stats.samples_used.push(samples_used);
        let expected = expected_from_oracle(oracle_case.expected(*point));
        let vpoint = ValidationPoint {
            position: *point,
            expected,
            description: "oracle".to_string(),
        };
        stats.record(&vpoint, &result);
    }

    println!("\n{}", "=".repeat(72));
    println!("ATTEMPT BENCHMARK (Oracle)");
    println!("{}", "=".repeat(72));
    stats.report();
}

pub fn run_attempt_benchmark_with_diag<F>(name: &str, diag_budget: usize, attempt: F)
where
    F: Fn((f64, f64, f64), (f64, f64, f64), &dyn Fn(f64, f64, f64) -> f32) -> (VertexGeometry, u64, Attempt1Diag),
{
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points(&cube);

    let mut stats = AttemptStats::new(name);
    let mut printed = 0usize;
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
        let (result, samples_used, diag) = attempt(point.position, hint, &sampler);
        stats.samples_used.push(samples_used);
        stats.record(point, &result);

        if printed < diag_budget {
            let is_match = matches_expected(point, &result);
            if !is_match {
                printed += 1;
                print_diag(point, &result, &diag);
            }
        }
    }

    println!("\n{}", "=".repeat(72));
    println!("ATTEMPT BENCHMARK (Analytical Rotated Cube)");
    println!("{}", "=".repeat(72));
    stats.report();
}

fn read_diag_budget() -> usize {
    std::env::var("ATTEMPT1_DIAG")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

fn read_oracle_mode() -> bool {
    std::env::var("ORACLE_BENCH")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn read_attempt2_diag_mode() -> bool {
    std::env::var("ATTEMPT2_CROSSING_DIAG")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn read_attempt2_diag_samples() -> usize {
    std::env::var("ATTEMPT2_CROSSING_SAMPLES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5)
}

fn print_attempt2_diag_sample(point: &ValidationPoint, result: &VertexGeometry, diag: &Attempt2Diag) {
    println!("\nATTEMPT2_CROSSING_DIAG sample");
    println!("  expected={:?} result={:?}", point.expected, result.classification);
    println!(
        "  crossing_count={} hint_available={} surface_points={} edge_points={}",
        diag.crossing_count, diag.hint_available, diag.surface_points, diag.edge_points
    );
    if let Some(hist) = diag.hint_hist {
        println!(
            "  hint_dot_hist=[<0, <0.25, <0.5, <0.75, <0.9, >=0.9]=[{}, {}, {}, {}, {}, {}]",
            hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]
        );
    }
}

fn expected_from_oracle(hit: OracleHit) -> ExpectedClassification {
    match hit.classification {
        OracleClassification::Face => ExpectedClassification::OnFace {
            face_index: 0,
            expected_normal: hit.normals.get(0).copied().unwrap_or((0.0, 1.0, 0.0)),
        },
        OracleClassification::Edge => ExpectedClassification::OnEdge {
            edge_index: 0,
            expected_direction: hit.edge_direction.unwrap_or((1.0, 0.0, 0.0)),
            expected_normals: (
                hit.normals.get(0).copied().unwrap_or((0.0, 1.0, 0.0)),
                hit.normals.get(1).copied().unwrap_or((1.0, 0.0, 0.0)),
            ),
        },
        OracleClassification::Corner | OracleClassification::Unknown => {
            let n0 = hit.normals.get(0).copied().unwrap_or((1.0, 0.0, 0.0));
            let n1 = hit.normals.get(1).copied().unwrap_or((0.0, 1.0, 0.0));
            let n2 = hit.normals.get(2).copied().unwrap_or((0.0, 0.0, 1.0));
            ExpectedClassification::OnCorner {
                corner_index: 0,
                expected_normals: [n0, n1, n2],
            }
        }
    }
}

fn matches_expected(point: &ValidationPoint, result: &VertexGeometry) -> bool {
    match point.expected {
        ExpectedClassification::OnFace { .. } => result.classification == GeometryType::Face,
        ExpectedClassification::OnEdge { .. } => result.classification == GeometryType::Edge,
        ExpectedClassification::OnCorner { .. } => result.classification == GeometryType::Corner,
    }
}

fn print_diag(point: &ValidationPoint, result: &VertexGeometry, diag: &Attempt1Diag) {
    let expected_label = match point.expected {
        ExpectedClassification::OnFace { .. } => "Face",
        ExpectedClassification::OnEdge { .. } => "Edge",
        ExpectedClassification::OnCorner { .. } => "Corner",
    };
    println!("\nATTEMPT1_DIAG mismatch");
    println!("  Expected: {}", expected_label);
    println!("  Result: {:?}", result.classification);
    println!(
        "  Crossing count: {} | samples_used: {}",
        diag.crossing_count, diag.samples_used
    );
    if let Some(residual) = diag.face_residual {
        println!("  Face: points={} residual={:.6}", diag.face_points, residual);
    }
    if let Some((ra, rb)) = diag.edge_residuals {
        println!(
            "  Edge: points={} inliers={:?} angle={:?} residuals=({:.6},{:.6})",
            diag.edge_points, diag.edge_inliers, diag.edge_angle_deg, ra, rb
        );
    }
    if let Some(residuals) = diag.corner_residuals.as_ref() {
        println!(
            "  Corner: points={} residuals={:?}",
            diag.corner_points, residuals
        );
    }
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

fn hint_normal(expected: ExpectedClassification) -> (f64, f64, f64) {
    match expected {
        ExpectedClassification::OnFace { expected_normal, .. } => expected_normal,
        ExpectedClassification::OnEdge { expected_normals, .. } => {
            normalize(add(expected_normals.0, expected_normals.1))
        }
        ExpectedClassification::OnCorner { expected_normals, .. } => normalize(add(
            add(expected_normals[0], expected_normals[1]),
            expected_normals[2],
        )),
    }
}

fn to_f32(v: (f64, f64, f64)) -> [f32; 3] {
    [v.0 as f32, v.1 as f32, v.2 as f32]
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

    #[test]
    fn benchmark_attempt_1_against_reference() {
        run_attempt_1_benchmark();
    }

    #[test]
    fn benchmark_attempt_2_against_reference() {
        run_attempt_2_benchmark();
    }

    #[test]
    fn experiment_hermite_microgrid() {
        run_hermite_microgrid_experiment();
    }

    #[test]
    fn experiment_ml_policy_mvp() {
        run_ml_policy_experiment_runner();
    }

}
