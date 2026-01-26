//! MVP: Learned sampling policy vs octant policy on analytical cube.

use crate::adaptive_surface_nets_2::stage4::research::analytical_cube::AnalyticalRotatedCube;
use crate::adaptive_surface_nets_2::stage4::research::oracle::OracleClassification;
use crate::adaptive_surface_nets_2::stage4::research::sample_cache::{
    begin_sample_recording, end_sample_recording, find_crossing_in_direction, SampleCache,
};
use crate::adaptive_surface_nets_2::stage4::research::validation::{
    generate_validation_points, ExpectedClassification, ValidationPoint,
};
use crate::sample_cloud::{SampleCloudDump, SampleCloudSet};
use std::path::Path;

const FEATURE_LEN: usize = 8;
const BUDGET: usize = 50;
const MAX_DISTANCE: f64 = 1.0;
const INITIAL_STEP: f64 = 0.05;
const BINARY_ITERS: usize = 12;

const TRAIN_EPOCHS: usize = 40;
const LEARNING_RATE: f64 = 0.05;
const DISCOUNT: f64 = 0.98;
const LAMBDA_SAMPLE: f64 = 0.02;

const W_CLS: f64 = 1.0;
const W_NORM: f64 = 0.5;
const W_EDGE: f64 = 0.5;

const RANSAC_ITERS: usize = 60;
const RANSAC_THRESHOLD: f64 = 0.01;

const DIR_COUNT: usize = 32;

pub fn run_ml_policy_experiment() {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points(&cube);

    let mut rng = Rng::new(12345);
    let directions = fibonacci_sphere_directions(DIR_COUNT);

    let mut directional = Policy::new(DIR_COUNT, FEATURE_LEN);
    train_policy(
        &mut directional,
        &cube,
        &points,
        &directions,
        PolicyMode::Directional,
        &mut rng,
    );

    let mut octant = Policy::new(8, FEATURE_LEN);
    train_policy(
        &mut octant,
        &cube,
        &points,
        &directions,
        PolicyMode::Octant,
        &mut rng,
    );

    println!("\n{}", "=".repeat(72));
    println!("ML POLICY MVP (ROTATED CUBE)");
    println!("{}", "=".repeat(72));

    let dir_eval = evaluate_policy(
        &directional,
        &cube,
        &points,
        &directions,
        PolicyMode::Directional,
        OctantEval::Argmax,
    );
    println!("{}", dir_eval.summary("Directional policy"));

    let oct_argmax = evaluate_policy(
        &octant,
        &cube,
        &points,
        &directions,
        PolicyMode::Octant,
        OctantEval::Argmax,
    );
    println!("{}", oct_argmax.summary("Octant policy (argmax)"));

    let oct_lerp = evaluate_policy(
        &octant,
        &cube,
        &points,
        &directions,
        PolicyMode::Octant,
        OctantEval::Lerp,
    );
    println!("{}", oct_lerp.summary("Octant policy (lerp)"));
}

#[derive(Clone, Copy, Debug)]
pub enum MlPolicyDumpKind {
    Directional,
    OctantArgmax,
    OctantLerp,
}

pub fn dump_ml_policy_sample_cloud(kind: MlPolicyDumpKind, output_path: &Path) {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points(&cube);
    let mut rng = Rng::new(12345);
    let directions = fibonacci_sphere_directions(DIR_COUNT);

    let mut directional = Policy::new(DIR_COUNT, FEATURE_LEN);
    train_policy(
        &mut directional,
        &cube,
        &points,
        &directions,
        PolicyMode::Directional,
        &mut rng,
    );

    let mut octant = Policy::new(8, FEATURE_LEN);
    train_policy(
        &mut octant,
        &cube,
        &points,
        &directions,
        PolicyMode::Octant,
        &mut rng,
    );

    let (policy, mode, oct_eval, label) = match kind {
        MlPolicyDumpKind::Directional => (&directional, PolicyMode::Directional, OctantEval::Argmax, "ml-directional"),
        MlPolicyDumpKind::OctantArgmax => (&octant, PolicyMode::Octant, OctantEval::Argmax, "ml-octant-argmax"),
        MlPolicyDumpKind::OctantLerp => (&octant, PolicyMode::Octant, OctantEval::Lerp, "ml-octant-lerp"),
    };

    let mut dump = SampleCloudDump::new();

    for (idx, point) in points.iter().enumerate() {
        begin_sample_recording();
        let mut rng_local = Rng::new(999 + idx as u64);
        let episode = run_episode(
            policy,
            &cube,
            point,
            &directions,
            mode,
            &mut rng_local,
            false,
            oct_eval,
        );
        let samples = end_sample_recording();

        let mut set = SampleCloudSet::new(idx as u64, to_f32(point.position));
        set.label = Some(format!("{}: {}", label, point.description));
        set.points = samples;
        set.meta.samples_used = Some(episode.samples_used as u32);
        set.meta.note = Some(label.to_string());
        // Add initial normal as a named vector
        let init_normal = initial_normal(point.expected.clone());
        set.add_vector("initial_normal", to_f32(point.position), to_f32(init_normal), Some([0.5, 0.5, 1.0, 1.0]));
        dump.add_set(set);
    }

    if let Err(err) = dump.save(output_path) {
        eprintln!("Failed to write ML sample cloud: {err}");
    } else {
        println!(
            "Wrote {} sample sets to {}",
            dump.sets.len(),
            output_path.display()
        );
    }
}

#[derive(Clone, Copy)]
enum PolicyMode {
    Directional,
    Octant,
}

#[derive(Clone, Copy)]
enum OctantEval {
    Argmax,
    Lerp,
}

#[derive(Clone, Debug)]
struct Policy {
    weights: Vec<f64>,
    biases: Vec<f64>,
    action_count: usize,
    feature_len: usize,
}

impl Policy {
    fn new(action_count: usize, feature_len: usize) -> Self {
        Self {
            weights: vec![0.0; action_count * feature_len],
            biases: vec![0.0; action_count],
            action_count,
            feature_len,
        }
    }

    fn logits(&self, features: &[f64]) -> Vec<f64> {
        let mut logits = vec![0.0; self.action_count];
        for action in 0..self.action_count {
            let mut v = self.biases[action];
            let base = action * self.feature_len;
            for i in 0..self.feature_len {
                v += self.weights[base + i] * features[i];
            }
            logits[action] = v;
        }
        logits
    }
}

fn train_policy(
    policy: &mut Policy,
    cube: &AnalyticalRotatedCube,
    points: &[ValidationPoint],
    directions: &[(f64, f64, f64)],
    mode: PolicyMode,
    rng: &mut Rng,
) {
    for _ in 0..TRAIN_EPOCHS {
        for point in points {
            let episode = run_episode(policy, cube, point, directions, mode, rng, true, OctantEval::Argmax);
            if episode.steps.is_empty() {
                continue;
            }

            let returns = compute_returns(&episode.rewards, DISCOUNT);
            let baseline = returns.iter().sum::<f64>() / returns.len() as f64;

            for (step, g_t) in episode.steps.iter().zip(returns.iter()) {
                let advantage = g_t - baseline;
                let mut probs = step.probs.clone();
                let action = step.action;

                let sum_exp: f64 = probs.iter().sum();
                if sum_exp < 1e-12 {
                    continue;
                }

                for p in &mut probs {
                    *p /= sum_exp;
                }

                for a in 0..policy.action_count {
                    let coeff = if a == action {
                        1.0 - probs[a]
                    } else {
                        -probs[a]
                    };
                    let base = a * policy.feature_len;
                    for i in 0..policy.feature_len {
                        policy.weights[base + i] += LEARNING_RATE * advantage * coeff * step.features[i];
                    }
                    policy.biases[a] += LEARNING_RATE * advantage * coeff;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct EpisodeStep {
    features: Vec<f64>,
    probs: Vec<f64>,
    action: usize,
}

#[derive(Clone, Debug)]
struct EpisodeTrace {
    steps: Vec<EpisodeStep>,
    rewards: Vec<f64>,
    samples_used: u64,
    final_loss: f64,
    predicted_class: OracleClassification,
}

fn run_episode(
    policy: &Policy,
    cube: &AnalyticalRotatedCube,
    point: &ValidationPoint,
    directions: &[(f64, f64, f64)],
    mode: PolicyMode,
    rng: &mut Rng,
    training: bool,
    octant_eval: OctantEval,
) -> EpisodeTrace {
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
    let midpoint = point.position;
    let hint_normal = initial_normal(point.expected.clone());
    let inside_point = find_inside_point(midpoint, hint_normal, &cache);

    let mut samples = Vec::new();
    let mut total_attempts = 0usize;
    let mut crossings_found = 0usize;
    let mut last_dir = hint_normal;

    for dir in [hint_normal, neg(hint_normal)] {
        total_attempts += 1;
        if let Some((crossing, _)) = find_crossing_in_direction(
            &cache,
            inside_point,
            dir,
            MAX_DISTANCE,
            INITIAL_STEP,
            BINARY_ITERS,
        ) {
            samples.push(crossing);
            crossings_found += 1;
            last_dir = dir;
        }
    }

    let mut latent = [0.0; FEATURE_LEN];
    let mut rewards = Vec::new();
    let mut steps = Vec::new();
    let mut prev_loss = loss_for(point, &samples);

    for step_idx in 0..(BUDGET - 2) {
        let features = build_features(
            step_idx,
            &samples,
            total_attempts,
            crossings_found,
            last_dir,
            hint_normal,
            &latent,
        );
        update_latent(&mut latent, &features);

        let logits = policy.logits(&features);
        let probs = softmax(&logits);

        let action = if training {
            sample_categorical(&probs, rng)
        } else {
            argmax(&probs)
        };

        let direction = match mode {
            PolicyMode::Directional => directions[action],
            PolicyMode::Octant => match octant_eval {
                OctantEval::Argmax => octant_direction(action),
                OctantEval::Lerp => {
                    let mut dir = (0.0, 0.0, 0.0);
                    for (idx, p) in probs.iter().enumerate() {
                        let o = octant_direction(idx);
                        dir = add(dir, scale(o, *p));
                    }
                    normalize(dir)
                }
            },
        };

        total_attempts += 1;
        if let Some((crossing, _)) = find_crossing_in_direction(
            &cache,
            inside_point,
            direction,
            MAX_DISTANCE,
            INITIAL_STEP,
            BINARY_ITERS,
        ) {
            samples.push(crossing);
            crossings_found += 1;
            last_dir = direction;
        }

        let loss = loss_for(point, &samples);
        let reward = (prev_loss - loss) - LAMBDA_SAMPLE;
        prev_loss = loss;

        rewards.push(reward);
        steps.push(EpisodeStep {
            features,
            probs,
            action,
        });
    }

    let (scores, _) = estimate_scores(point, &samples);
    let predicted_class = if scores.face >= scores.edge && scores.face >= scores.corner {
        OracleClassification::Face
    } else if scores.edge >= scores.corner {
        OracleClassification::Edge
    } else {
        OracleClassification::Corner
    };

    EpisodeTrace {
        steps,
        rewards,
        samples_used: cache.stats().actual_samples(),
        final_loss: prev_loss,
        predicted_class,
    }
}

#[derive(Clone, Debug)]
struct EvalStats {
    classification_accuracy: f64,
    avg_loss: f64,
    avg_samples: f64,
}

impl EvalStats {
    fn summary(&self, name: &str) -> String {
        format!(
            "  {}: accuracy={:.2}%, avg_loss={:.3}, avg_samples={:.1}",
            name,
            self.classification_accuracy * 100.0,
            self.avg_loss,
            self.avg_samples
        )
    }
}

fn evaluate_policy(
    policy: &Policy,
    cube: &AnalyticalRotatedCube,
    points: &[ValidationPoint],
    directions: &[(f64, f64, f64)],
    mode: PolicyMode,
    octant_eval: OctantEval,
) -> EvalStats {
    let mut rng = Rng::new(999);
    let mut correct = 0usize;
    let mut loss_sum = 0.0;
    let mut samples_sum = 0.0;

    for point in points {
        let episode = run_episode(policy, cube, point, directions, mode, &mut rng, false, octant_eval);
        if episode.predicted_class == expected_class(point) {
            correct += 1;
        }
        loss_sum += episode.final_loss;
        samples_sum += episode.samples_used as f64;
    }

    EvalStats {
        classification_accuracy: correct as f64 / points.len() as f64,
        avg_loss: loss_sum / points.len() as f64,
        avg_samples: samples_sum / points.len() as f64,
    }
}

fn expected_class(point: &ValidationPoint) -> OracleClassification {
    match point.expected {
        ExpectedClassification::OnFace { .. } => OracleClassification::Face,
        ExpectedClassification::OnEdge { .. } => OracleClassification::Edge,
        ExpectedClassification::OnCorner { .. } => OracleClassification::Corner,
    }
}

fn loss_for(point: &ValidationPoint, samples: &[(f64, f64, f64)]) -> f64 {
    let (scores, errors) = estimate_scores(point, samples);
    let probs = normalize_scores(scores);
    let expected = expected_class(point);
    let ce = match expected {
        OracleClassification::Face => -probs.face.ln(),
        OracleClassification::Edge => -probs.edge.ln(),
        OracleClassification::Corner => -probs.corner.ln(),
        OracleClassification::Unknown => 1.0,
    };

    let normal_err = errors.normal_error / 90.0;
    let edge_err = errors.edge_dir_error / 90.0;

    W_CLS * ce + W_NORM * normal_err + W_EDGE * edge_err
}

#[derive(Clone, Copy)]
struct Scores {
    face: f64,
    edge: f64,
    corner: f64,
}

#[derive(Clone, Copy)]
struct Errors {
    normal_error: f64,
    edge_dir_error: f64,
}

fn estimate_scores(
    point: &ValidationPoint,
    samples: &[(f64, f64, f64)],
) -> (Scores, Errors) {
    let face_fit = fit_plane_svd(samples);
    let edge_fit = fit_edge_ransac(samples);
    let corner_fit = fit_corner_ransac(samples);

    let face_score = if let Some(ref fit) = face_fit {
        (-fit.residual / 0.01).exp()
    } else {
        1e-6
    };
    let edge_score = if let Some(ref fit) = edge_fit {
        let angle = angle_between(fit.normal_a, fit.normal_b).to_degrees() / 90.0;
        (-fit.residual / 0.02).exp() * angle.max(0.0)
    } else {
        1e-6
    };
    let corner_score = if let Some(ref fit) = corner_fit {
        let min_angle = min_pair_angle(&fit.normals) / 90.0;
        (-fit.residual / 0.03).exp() * min_angle.max(0.0)
    } else {
        1e-6
    };

    let (normal_error, edge_dir_error) = match point.expected {
        ExpectedClassification::OnFace { expected_normal, .. } => {
            let err = face_fit
                .map(|f| angle_between(f.normal, expected_normal).to_degrees())
                .unwrap_or(180.0);
            (err, 0.0)
        }
        ExpectedClassification::OnEdge {
            expected_normals,
            expected_direction,
            ..
        } => {
            if let Some(fit) = edge_fit {
                let (err_a, err_b) = best_normal_pairing(
                    (fit.normal_a, fit.normal_b),
                    expected_normals,
                );
                let dir_err = edge_direction_error(fit.edge_direction, expected_direction);
                ((err_a + err_b) * 0.5, dir_err)
            } else {
                (180.0, 180.0)
            }
        }
        ExpectedClassification::OnCorner { expected_normals, .. } => {
            if let Some(fit) = corner_fit {
                let errs = match_normals_to_expected(&fit.normals, &expected_normals);
                let avg = errs.iter().sum::<f64>() / errs.len() as f64;
                (avg, 0.0)
            } else {
                (180.0, 0.0)
            }
        }
    };

    (
        Scores {
            face: face_score,
            edge: edge_score,
            corner: corner_score,
        },
        Errors {
            normal_error,
            edge_dir_error,
        },
    )
}

fn normalize_scores(scores: Scores) -> Scores {
    let sum = scores.face + scores.edge + scores.corner + 1e-12;
    Scores {
        face: (scores.face / sum).max(1e-6),
        edge: (scores.edge / sum).max(1e-6),
        corner: (scores.corner / sum).max(1e-6),
    }
}

#[derive(Clone)]
struct PlaneFit {
    centroid: (f64, f64, f64),
    normal: (f64, f64, f64),
    residual: f64,
}

#[derive(Clone)]
struct EdgeFit {
    normal_a: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
    residual: f64,
}

#[derive(Clone)]
struct CornerFit {
    normals: [(f64, f64, f64); 3],
    residual: f64,
}

fn fit_plane_svd(points: &[(f64, f64, f64)]) -> Option<PlaneFit> {
    if points.len() < 3 {
        return None;
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

    let max_estimate = cov
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row[i].abs()
                + row
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &v)| v.abs())
                    .sum::<f64>()
        })
        .fold(0.0_f64, f64::max);
    let shift = max_estimate + 1.0;
    let shifted = [
        [shift - cov[0][0], -cov[0][1], -cov[0][2]],
        [-cov[1][0], shift - cov[1][1], -cov[1][2]],
        [-cov[2][0], -cov[2][1], shift - cov[2][2]],
    ];

    let mut v = normalize((0.6, 0.8, 0.3));
    for _ in 0..80 {
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

    let mut residual_sum = 0.0;
    for p in &centered {
        let dist = dot(*p, v).abs();
        residual_sum += dist * dist;
    }
    let residual = (residual_sum / points.len() as f64).sqrt();

    Some(PlaneFit {
        centroid,
        normal: v,
        residual,
    })
}

fn fit_edge_ransac(points: &[(f64, f64, f64)]) -> Option<EdgeFit> {
    let plane_a = ransac_plane_fit(points)?;
    let remaining: Vec<_> = points
        .iter()
        .cloned()
        .filter(|p| !is_inlier(*p, plane_a.centroid, plane_a.normal, RANSAC_THRESHOLD))
        .collect();
    if remaining.len() < 3 {
        return None;
    }
    let plane_b = ransac_plane_fit(&remaining)?;
    let edge_dir = normalize(cross(plane_a.normal, plane_b.normal));
    if length(edge_dir) < 1e-6 {
        return None;
    }
    Some(EdgeFit {
        normal_a: plane_a.normal,
        normal_b: plane_b.normal,
        edge_direction: edge_dir,
        residual: (plane_a.residual + plane_b.residual) * 0.5,
    })
}

fn fit_corner_ransac(points: &[(f64, f64, f64)]) -> Option<CornerFit> {
    let mut remaining: Vec<_> = points.to_vec();
    let mut normals = Vec::new();
    let mut residuals = Vec::new();

    for _ in 0..3 {
        let fit = ransac_plane_fit(&remaining)?;
        let duplicate = normals.iter().any(|n| angle_between(*n, fit.normal).to_degrees() < 10.0);
        if duplicate {
            return None;
        }
        normals.push(fit.normal);
        residuals.push(fit.residual);
        remaining = remaining
            .into_iter()
            .filter(|p| !is_inlier(*p, fit.centroid, fit.normal, RANSAC_THRESHOLD))
            .collect();
        if remaining.len() < 3 && normals.len() < 3 {
            return None;
        }
    }

    Some(CornerFit {
        normals: [normals[0], normals[1], normals[2]],
        residual: (residuals[0] + residuals[1] + residuals[2]) / 3.0,
    })
}

fn ransac_plane_fit(points: &[(f64, f64, f64)]) -> Option<PlaneFit> {
    if points.len() < 3 {
        return None;
    }
    let mut rng = Rng::new(777);
    let mut best_inliers: Vec<(f64, f64, f64)> = Vec::new();

    for _ in 0..RANSAC_ITERS {
        let i1 = rng.next_usize(points.len());
        let i2 = rng.next_usize(points.len());
        let i3 = rng.next_usize(points.len());
        if i1 == i2 || i1 == i3 || i2 == i3 {
            continue;
        }
        let p1 = points[i1];
        let p2 = points[i2];
        let p3 = points[i3];
        let normal = normalize(cross(sub(p2, p1), sub(p3, p1)));
        if length(normal) < 1e-6 {
            continue;
        }
        let mut inliers = Vec::new();
        for p in points {
            let dist = dot(sub(*p, p1), normal).abs();
            if dist < RANSAC_THRESHOLD {
                inliers.push(*p);
            }
        }
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }

    if best_inliers.len() < 3 {
        return None;
    }

    let refined = fit_plane_svd(&best_inliers)?;
    Some(PlaneFit {
        centroid: refined.centroid,
        normal: refined.normal,
        residual: refined.residual,
    })
}

fn is_inlier(
    p: (f64, f64, f64),
    plane_point: (f64, f64, f64),
    plane_normal: (f64, f64, f64),
    threshold: f64,
) -> bool {
    dot(sub(p, plane_point), plane_normal).abs() < threshold
}

fn min_pair_angle(normals: &[(f64, f64, f64); 3]) -> f64 {
    let a = angle_between(normals[0], normals[1]).to_degrees();
    let b = angle_between(normals[1], normals[2]).to_degrees();
    let c = angle_between(normals[0], normals[2]).to_degrees();
    a.min(b).min(c)
}

fn best_normal_pairing(
    detected: ((f64, f64, f64), (f64, f64, f64)),
    expected: ((f64, f64, f64), (f64, f64, f64)),
) -> (f64, f64) {
    let (a1, b1) = (
        angle_between(detected.0, expected.0).to_degrees(),
        angle_between(detected.1, expected.1).to_degrees(),
    );
    let (a2, b2) = (
        angle_between(detected.0, expected.1).to_degrees(),
        angle_between(detected.1, expected.0).to_degrees(),
    );
    if a1 + b1 <= a2 + b2 {
        (a1, b1)
    } else {
        (a2, b2)
    }
}

fn match_normals_to_expected(
    detected: &[(f64, f64, f64); 3],
    expected: &[(f64, f64, f64); 3],
) -> Vec<f64> {
    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut best = vec![180.0; 3];
    for perm in perms {
        let errs = vec![
            angle_between(detected[perm[0]], expected[0]).to_degrees(),
            angle_between(detected[perm[1]], expected[1]).to_degrees(),
            angle_between(detected[perm[2]], expected[2]).to_degrees(),
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

fn initial_normal(expected: ExpectedClassification) -> (f64, f64, f64) {
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

fn find_inside_point<F>(
    position: (f64, f64, f64),
    hint_normal: (f64, f64, f64),
    cache: &SampleCache<F>,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    let step = 0.2;
    let hint = normalize(hint_normal);
    let candidates = [
        hint,
        neg(hint),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ];

    for dir in candidates {
        let probe = add(position, scale(dir, step));
        if cache.is_inside(probe.0, probe.1, probe.2) {
            return probe;
        }
    }
    position
}

fn build_features(
    step_idx: usize,
    samples: &[(f64, f64, f64)],
    total_attempts: usize,
    crossings_found: usize,
    last_dir: (f64, f64, f64),
    hint_normal: (f64, f64, f64),
    latent: &[f64; FEATURE_LEN],
) -> Vec<f64> {
    let attempts = total_attempts as f64;
    let found = crossings_found as f64;
    let hit_ratio = if attempts > 0.0 { found / attempts } else { 0.0 };
    let avg_dist = if samples.is_empty() {
        0.0
    } else {
        samples
            .iter()
            .map(|p| length(*p))
            .sum::<f64>()
            / samples.len() as f64
    };

    let mut features = vec![
        1.0,
        step_idx as f64 / BUDGET as f64,
        samples.len() as f64 / BUDGET as f64,
        hit_ratio,
        (avg_dist / MAX_DISTANCE).min(1.0),
        dot(normalize(last_dir), normalize(hint_normal)),
        latent[0],
        latent[1],
    ];
    features.resize(FEATURE_LEN, 0.0);
    features
}

fn update_latent(latent: &mut [f64; FEATURE_LEN], features: &[f64]) {
    for i in 0..FEATURE_LEN {
        latent[i] = latent[i] * 0.8 + features[i] * 0.2;
    }
}

fn octant_direction(index: usize) -> (f64, f64, f64) {
    let x = if index & 1 == 0 { -1.0 } else { 1.0 };
    let y = if index & 2 == 0 { -1.0 } else { 1.0 };
    let z = if index & 4 == 0 { -1.0 } else { 1.0 };
    normalize((x, y, z))
}

fn compute_returns(rewards: &[f64], discount: f64) -> Vec<f64> {
    let mut returns = vec![0.0; rewards.len()];
    let mut g = 0.0;
    for (i, r) in rewards.iter().enumerate().rev() {
        g = r + discount * g;
        returns[i] = g;
    }
    returns
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-12 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|v| v / sum).collect()
}

fn sample_categorical(probs: &[f64], rng: &mut Rng) -> usize {
    let mut r = rng.next_f64();
    for (i, p) in probs.iter().enumerate() {
        if r <= *p {
            return i;
        }
        r -= p;
    }
    probs.len().saturating_sub(1)
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, v) in values.iter().enumerate() {
        if *v > best_val {
            best_val = *v;
            best = i;
        }
    }
    best
}

fn fibonacci_sphere_directions(n: usize) -> Vec<(f64, f64, f64)> {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let angle_increment = std::f64::consts::PI * 2.0 * golden_ratio;

    (0..n)
        .map(|i| {
            let t = (i as f64 + 0.5) / n as f64;
            let phi = angle_increment * i as f64;
            let theta = (1.0 - 2.0 * t).acos();
            (
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            )
        })
        .collect()
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

fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
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

fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let denom = length(a) * length(b);
    if denom < 1e-12 {
        return 0.0;
    }
    let mut v = dot(a, b) / denom;
    if v > 1.0 {
        v = 1.0;
    } else if v < -1.0 {
        v = -1.0;
    }
    v.acos()
}

#[derive(Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f64(&mut self) -> f64 {
        let v = self.next_u32() as f64 / u32::MAX as f64;
        v.min(0.999999)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max.max(1)
    }
}
