//! RNN-Based Sampling Policy MVP
//!
//! This module implements an RNN policy that learns WHERE to sample in a cubic
//! volume around a vertex, replacing the fixed-direction sampling approaches.
//!
//! ## Architecture
//!
//! - **GRU Updater**: `(latent[32], input[~10]) → new_latent[32]`
//!   - Takes normalized position, in/out status, oracle distance, budget fraction
//!   - GRU chosen over LSTM (simpler, sufficient for short episodes ~50 steps)
//!
//! - **Chooser Head**: `latent[32] → octant_weights[8] → position`
//!   - Linear layer outputs 8 weights for cube corners
//!   - Position = weighted average of corners (vertex ± 0.5*cell_size)
//!
//! ## Training
//!
//! - REINFORCE with baseline
//! - Adam optimizer, lr ~0.001
//! - Gradient clipping at 1.0
//! - Full BPTT (episodes are short, no truncation needed)
//!
//! ## Usage
//!
//! ```ignore
//! use rnn_policy::{run_rnn_policy_experiment, RnnPolicyDumpKind, dump_rnn_policy_sample_cloud};
//!
//! // Run the full experiment
//! run_rnn_policy_experiment();
//!
//! // Or dump sample clouds for visualization
//! dump_rnn_policy_sample_cloud(RnnPolicyDumpKind::Trained, &path);
//! ```

pub mod chooser;
pub mod classifier;
pub mod classifier_heads;
pub mod gradients;
pub mod gru;
pub mod math;
pub mod policy;
pub mod reward;
pub mod training;

use std::path::Path;

use crate::adaptive_surface_nets_2::stage4::research::analytical_cube::AnalyticalRotatedCube;
use crate::adaptive_surface_nets_2::stage4::research::sample_cache::{
    begin_sample_recording, end_sample_recording,
};
use crate::adaptive_surface_nets_2::stage4::research::validation::generate_validation_points_randomized;
use crate::sample_cloud::{SampleCloudDump, SampleCloudSet};

use gru::Rng;
use policy::{run_episode, run_episode_ex, RnnPolicy};
use reward::RewardConfig;
use training::{evaluate_policy, train_policy, EvaluationResult, TrainingConfig};

/// Which variant of the RNN policy to use for sample cloud dumps.
#[derive(Clone, Copy, Debug)]
pub enum RnnPolicyDumpKind {
    /// Untrained (random) policy.
    Untrained,
    /// Trained policy (trains fresh).
    Trained,
    /// Load from file.
    FromFile,
}

/// Default path for saved RNN policy weights.
pub const DEFAULT_MODEL_PATH: &str = "rnn_policy_trained.bin";

/// Run the full RNN policy experiment.
///
/// This trains the policy on validation points and reports accuracy.
pub fn run_rnn_policy_experiment() {
    run_rnn_policy_experiment_with_epochs(1000, 200);
}

/// Run the RNN policy experiment with configurable epochs.
pub fn run_rnn_policy_experiment_with_epochs(epochs: usize, print_every: usize) {
    use classifier_heads::{compute_classifier_loss, ClassifierLossConfig, ExpectedGeometry, GeometryType};

    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);

    let mut rng = Rng::new(12345);
    let reward_config = RewardConfig::for_classifier_training();

    println!("\n{}", "=".repeat(72));
    println!("RNN POLICY WITH NEURAL CLASSIFIER HEADS");
    println!("{}", "=".repeat(72));

    // Create and train policy
    let mut policy = RnnPolicy::new(&mut rng);
    println!(
        "\nPolicy parameters: {} (GRU: {}, Chooser: {}, Heads: {})",
        policy.param_count(),
        policy.gru.param_count(),
        policy.chooser.param_count(),
        policy.classifier_heads.param_count()
    );
    println!("Training points: {}", points.len());

    // Helper to evaluate classifier accuracy
    // Uses stochastic=true for jittered samples to match training distribution
    let eval_classifier = |policy: &RnnPolicy, cube: &AnalyticalRotatedCube, points: &[crate::adaptive_surface_nets_2::stage4::research::validation::ValidationPoint], rng: &mut Rng| {
        let mut correct = 0;
        let mut total_loss = 0.0;
        let loss_config = ClassifierLossConfig::default();

        for (idx, point) in points.iter().enumerate() {
            // Use stochastic=true to get jittered samples matching training
            let episode = run_episode_ex(policy, cube, point, idx, rng, true, true, &reward_config);
            let predictions = policy.classify(&episode.h_final);

            let expected_geom = match &point.expected {
                crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification::OnFace { expected_normal, .. } => {
                    ExpectedGeometry::Face { normal: *expected_normal }
                }
                crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification::OnEdge { expected_normals, expected_direction, .. } => {
                    ExpectedGeometry::Edge {
                        normal_a: expected_normals.0,
                        normal_b: expected_normals.1,
                        direction: *expected_direction,
                    }
                }
                crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification::OnCorner { expected_normals, .. } => {
                    ExpectedGeometry::Corner { normals: *expected_normals }
                }
            };

            let predicted_type = predictions.predicted_type();
            let expected_type = expected_geom.geometry_type();
            if predicted_type == expected_type {
                correct += 1;
            }

            let (loss, _, _, _) = compute_classifier_loss(&predictions, &expected_geom, &loss_config);
            total_loss += loss;
        }

        (correct as f64 / points.len() as f64, total_loss / points.len() as f64)
    };

    // Evaluate before training
    let (acc_before, loss_before) = eval_classifier(&policy, &cube, &points, &mut rng);
    println!("\nBefore training:");
    println!("  Classifier accuracy: {:.1}%, loss: {:.3}", acc_before * 100.0, loss_before);

    // Also show RANSAC-based eval for comparison
    let eval_before = evaluate_policy(&policy, &cube, &points, &reward_config, &mut rng);
    println!("  RANSAC fit rate: {:.1}%", eval_before.fit_rate * 100.0);

    // Train
    println!("\nTraining with rotation augmentation...");
    let config = TrainingConfig {
        epochs,
        print_every,
        lr: 0.001,
        use_classifier_heads: true,
        use_rotation_augmentation: true,
        ..Default::default()
    };
    let _stats = train_policy(&mut policy, &cube, &points, &config, &reward_config, &mut rng);

    // Evaluate after training (on standard cube - no rotation)
    let (acc_after, loss_after) = eval_classifier(&policy, &cube, &points, &mut rng);
    println!("\nAfter training:");
    println!("  Classifier accuracy: {:.1}%, loss: {:.3}", acc_after * 100.0, loss_after);

    let eval_after = evaluate_policy(&policy, &cube, &points, &reward_config, &mut rng);
    println!("  RANSAC fit rate: {:.1}%", eval_after.fit_rate * 100.0);

    // Breakdown by classification
    println!("\nBreakdown by geometry type:");
    print_breakdown(&eval_after);

    // Show classifier confidence breakdown
    println!("\nClassifier confidence analysis:");
    print_classifier_breakdown(&policy, &cube, &points, &reward_config, &mut rng);
}

fn print_breakdown(eval: &EvaluationResult) {
    let face_points: Vec<_> = eval
        .points
        .iter()
        .filter(|p| matches!(
            p.expected_class,
            crate::adaptive_surface_nets_2::stage4::research::oracle::OracleClassification::Face
        ))
        .collect();
    let edge_points: Vec<_> = eval
        .points
        .iter()
        .filter(|p| matches!(
            p.expected_class,
            crate::adaptive_surface_nets_2::stage4::research::oracle::OracleClassification::Edge
        ))
        .collect();
    let corner_points: Vec<_> = eval
        .points
        .iter()
        .filter(|p| matches!(
            p.expected_class,
            crate::adaptive_surface_nets_2::stage4::research::oracle::OracleClassification::Corner
        ))
        .collect();

    // Report fit success rate, normal accuracy, and surface point counts per category
    fn category_stats(points: &[&training::PointEvaluation]) -> (usize, f64, f64) {
        let fit_count = points.iter().filter(|p| p.fit_success).count();
        let avg_normal = if points.is_empty() {
            0.0
        } else {
            points.iter().map(|p| p.normal_reward).sum::<f64>() / points.len() as f64
        };
        let avg_surface_pts = if points.is_empty() {
            0.0
        } else {
            points.iter().map(|p| p.surface_points_count as f64).sum::<f64>() / points.len() as f64
        };
        (fit_count, avg_normal, avg_surface_pts)
    }

    let (face_fit, face_normal, face_surface) = category_stats(&face_points);
    let (edge_fit, edge_normal, edge_surface) = category_stats(&edge_points);
    let (corner_fit, corner_normal, corner_surface) = category_stats(&corner_points);

    println!(
        "  Face:   {}/{} fit ({:.1}%), normal={:.3}, surface_pts={:.1}",
        face_fit,
        face_points.len(),
        if face_points.is_empty() { 0.0 } else { face_fit as f64 / face_points.len() as f64 * 100.0 },
        face_normal,
        face_surface
    );
    println!(
        "  Edge:   {}/{} fit ({:.1}%), normal={:.3}, surface_pts={:.1}",
        edge_fit,
        edge_points.len(),
        if edge_points.is_empty() { 0.0 } else { edge_fit as f64 / edge_points.len() as f64 * 100.0 },
        edge_normal,
        edge_surface
    );
    println!(
        "  Corner: {}/{} fit ({:.1}%), normal={:.3}, surface_pts={:.1}",
        corner_fit,
        corner_points.len(),
        if corner_points.is_empty() { 0.0 } else { corner_fit as f64 / corner_points.len() as f64 * 100.0 },
        corner_normal,
        corner_surface
    );

    // Average crossings found
    let avg_crossings = eval
        .points
        .iter()
        .map(|p| p.crossings_found as f64)
        .sum::<f64>()
        / eval.points.len() as f64;
    println!("  Avg crossings per episode: {:.1}", avg_crossings);
}

fn print_classifier_breakdown(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    points: &[crate::adaptive_surface_nets_2::stage4::research::validation::ValidationPoint],
    reward_config: &RewardConfig,
    rng: &mut Rng,
) {
    use classifier_heads::GeometryType;
    use crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification;

    let mut face_correct = 0;
    let mut face_total = 0;
    let mut edge_correct = 0;
    let mut edge_total = 0;
    let mut corner_correct = 0;
    let mut corner_total = 0;

    let mut face_conf_sum = 0.0;
    let mut edge_conf_sum = 0.0;
    let mut corner_conf_sum = 0.0;

    for (idx, point) in points.iter().enumerate() {
        // Use stochastic=true for jittered samples matching training
        let episode = run_episode_ex(policy, cube, point, idx, rng, true, true, reward_config);
        let predictions = policy.classify(&episode.h_final);
        let predicted_type = predictions.predicted_type();

        let expected_type = match &point.expected {
            ExpectedClassification::OnFace { .. } => GeometryType::Face,
            ExpectedClassification::OnEdge { .. } => GeometryType::Edge,
            ExpectedClassification::OnCorner { .. } => GeometryType::Corner,
        };

        match expected_type {
            GeometryType::Face => {
                face_total += 1;
                face_conf_sum += predictions.face.confidence;
                if predicted_type == expected_type {
                    face_correct += 1;
                }
            }
            GeometryType::Edge => {
                edge_total += 1;
                edge_conf_sum += predictions.edge.confidence;
                if predicted_type == expected_type {
                    edge_correct += 1;
                }
            }
            GeometryType::Corner => {
                corner_total += 1;
                corner_conf_sum += predictions.corner.confidence;
                if predicted_type == expected_type {
                    corner_correct += 1;
                }
            }
        }
    }

    if face_total > 0 {
        println!(
            "  Face:   {}/{} correct ({:.1}%), avg conf: {:.3}",
            face_correct, face_total,
            face_correct as f64 / face_total as f64 * 100.0,
            face_conf_sum / face_total as f64
        );
    }
    if edge_total > 0 {
        println!(
            "  Edge:   {}/{} correct ({:.1}%), avg conf: {:.3}",
            edge_correct, edge_total,
            edge_correct as f64 / edge_total as f64 * 100.0,
            edge_conf_sum / edge_total as f64
        );
    }
    if corner_total > 0 {
        println!(
            "  Corner: {}/{} correct ({:.1}%), avg conf: {:.3}",
            corner_correct, corner_total,
            corner_correct as f64 / corner_total as f64 * 100.0,
            corner_conf_sum / corner_total as f64
        );
    }
}

/// Train the RNN policy and save weights to a file.
pub fn train_and_save_rnn_policy(model_path: &Path) {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);

    let mut rng = Rng::new(12345);
    let reward_config = RewardConfig::default();

    println!("Creating RNN policy...");
    let mut policy = RnnPolicy::new(&mut rng);
    println!(
        "Policy parameters: {} (GRU: {}, Chooser: {})",
        policy.param_count(),
        policy.gru.param_count(),
        policy.chooser.param_count()
    );

    // Evaluate before training
    let eval_before = evaluate_policy(&policy, &cube, &points, &reward_config, &mut rng);
    println!("Before training: {}", eval_before.summary("Random"));
    print_breakdown(&eval_before);

    // Train
    println!("\nTraining (6000 epochs)...");
    let config = TrainingConfig {
        epochs: 6000,
        print_every: 1000,
        lr: 0.001,
        ..Default::default()
    };
    train_policy(&mut policy, &cube, &points, &config, &reward_config, &mut rng);

    // Evaluate after training
    let eval_after = evaluate_policy(&policy, &cube, &points, &reward_config, &mut rng);
    println!("\nAfter training: {}", eval_after.summary("Trained"));
    print_breakdown(&eval_after);

    // Save
    if let Err(e) = policy.save(model_path) {
        eprintln!("Failed to save model: {}", e);
    } else {
        println!("\nSaved model to {}", model_path.display());
    }
}

/// Load a trained RNN policy and dump sample clouds.
pub fn load_and_dump_rnn_policy(model_path: &Path, output_path: &Path, use_discrete: bool) {
    let policy = match RnnPolicy::load(model_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load model from {}: {}", model_path.display(), e);
            return;
        }
    };
    println!("Loaded model from {}", model_path.display());

    let label = if use_discrete { "rnn-trained-discrete" } else { "rnn-trained" };
    dump_rnn_policy_sample_cloud_with_policy(&policy, label, output_path, use_discrete);
}

/// Dump sample clouds from the RNN policy for visualization.
pub fn dump_rnn_policy_sample_cloud(kind: RnnPolicyDumpKind, output_path: &Path) {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);

    let mut rng = Rng::new(12345);
    let reward_config = RewardConfig::default();

    let mut policy = RnnPolicy::new(&mut rng);

    let label = match kind {
        RnnPolicyDumpKind::Untrained => "rnn-untrained",
        RnnPolicyDumpKind::Trained => {
            // Train the policy first
            let config = TrainingConfig {
                epochs: 200,
                print_every: 50,
                ..Default::default()
            };
            println!("Training RNN policy for sample cloud dump...");
            train_policy(&mut policy, &cube, &points, &config, &reward_config, &mut rng);
            "rnn-trained"
        }
        RnnPolicyDumpKind::FromFile => {
            // This variant should use load_and_dump_rnn_policy instead
            eprintln!("Use load_and_dump_rnn_policy for FromFile variant");
            return;
        }
    };

    dump_rnn_policy_sample_cloud_with_policy(&policy, label, output_path, true);
}

/// Dump sample clouds using a pre-loaded policy.
fn dump_rnn_policy_sample_cloud_with_policy(policy: &RnnPolicy, label: &str, output_path: &Path, use_discrete: bool) {
    use classifier_heads::GeometryType;

    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);

    let mut rng = Rng::new(12345);
    let reward_config = RewardConfig::default();

    let mut dump = SampleCloudDump::new();

    // Vector visualization scale (relative to cell size)
    let vec_scale = policy::CELL_SIZE as f32 * 0.4;

    for (idx, point) in points.iter().enumerate() {
        begin_sample_recording();

        // stochastic=false (use argmax), use_discrete controls corner vs weighted position
        let episode = run_episode_ex(policy, &cube, point, idx, &mut rng, false, use_discrete, &reward_config);

        let samples = end_sample_recording();

        // Run classifier on final hidden state
        let predictions = policy.classify(&episode.h_final);
        let predicted_type = predictions.predicted_type();
        let confs = predictions.confidences();

        let hint = hint_from_expected(&point.expected);
        let mut set = SampleCloudSet::with_cell_size(
            idx as u64,
            to_f32(point.position),
            policy::CELL_SIZE as f32,
        );
        set.label = Some(format!("{}: {}", label, point.description));
        set.points = samples;
        set.meta.samples_used = Some(episode.samples_used as u32);

        // Build note with classifier confidence values
        let type_str = match predicted_type {
            GeometryType::Face => "Face",
            GeometryType::Edge => "Edge",
            GeometryType::Corner => "Corner",
        };
        set.meta.note = Some(format!(
            "{}\nPredicted: {}\nConf: F={:.1}% E={:.1}% C={:.1}%",
            label,
            type_str,
            confs[0] * 100.0,
            confs[1] * 100.0,
            confs[2] * 100.0,
        ));

        // Add hint normal as a named vector (light blue, from vertex)
        set.add_vector(
            "hint",
            to_f32(point.position),
            scale_vec(to_f32(hint), vec_scale),
            Some([0.5, 0.5, 1.0, 1.0]),
        );

        // Add predicted vectors based on geometry type
        let vertex = to_f32(point.position);
        match predicted_type {
            GeometryType::Face => {
                // Face: single normal (green)
                set.add_vector(
                    "pred_normal",
                    vertex,
                    scale_vec(to_f32(predictions.face.normal), vec_scale),
                    Some([0.3, 1.0, 0.3, 1.0]),
                );
            }
            GeometryType::Edge => {
                // Edge: two normals (red, orange) and edge direction (yellow)
                set.add_vector(
                    "pred_normal_a",
                    vertex,
                    scale_vec(to_f32(predictions.edge.normal_a), vec_scale),
                    Some([1.0, 0.3, 0.3, 1.0]),
                );
                set.add_vector(
                    "pred_normal_b",
                    vertex,
                    scale_vec(to_f32(predictions.edge.normal_b), vec_scale),
                    Some([1.0, 0.6, 0.2, 1.0]),
                );
                set.add_vector(
                    "pred_edge_dir",
                    vertex,
                    scale_vec(to_f32(predictions.edge.direction), vec_scale),
                    Some([1.0, 1.0, 0.3, 1.0]),
                );
            }
            GeometryType::Corner => {
                // Corner: three normals (red, green, blue)
                let colors = [
                    [1.0, 0.3, 0.3, 1.0],
                    [0.3, 1.0, 0.3, 1.0],
                    [0.3, 0.3, 1.0, 1.0],
                ];
                for (i, normal) in predictions.corner.normals.iter().enumerate() {
                    set.add_vector(
                        &format!("pred_normal_{}", i),
                        vertex,
                        scale_vec(to_f32(*normal), vec_scale),
                        Some(colors[i]),
                    );
                }
            }
        }

        dump.add_set(set);
    }

    if let Err(err) = dump.save(output_path) {
        eprintln!("Failed to write RNN sample cloud: {err}");
    } else {
        println!(
            "Wrote {} sample sets to {}",
            dump.sets.len(),
            output_path.display()
        );
    }
}

fn hint_from_expected(
    expected: &crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification,
) -> (f64, f64, f64) {
    use crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification;
    match expected {
        ExpectedClassification::OnFace { expected_normal, .. } => *expected_normal,
        ExpectedClassification::OnEdge { expected_normals, .. } => {
            math::normalize_3d(math::add_3d(expected_normals.0, expected_normals.1))
        }
        ExpectedClassification::OnCorner { expected_normals, .. } => {
            let sum = math::add_3d(
                math::add_3d(expected_normals[0], expected_normals[1]),
                expected_normals[2],
            );
            math::normalize_3d(sum)
        }
    }
}

fn to_f32(v: (f64, f64, f64)) -> [f32; 3] {
    [v.0 as f32, v.1 as f32, v.2 as f32]
}

fn scale_vec(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_policy_experiment() {
        // Run a quick version of the experiment
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);
        let subset: Vec<_> = points.into_iter().take(5).collect();

        let mut rng = Rng::new(42);
        let reward_config = RewardConfig::default();

        let mut policy = RnnPolicy::new(&mut rng);

        let config = TrainingConfig {
            epochs: 10,
            print_every: 5,
            ..Default::default()
        };
        train_policy(&mut policy, &cube, &subset, &config, &reward_config, &mut rng);

        let eval = evaluate_policy(&policy, &cube, &subset, &reward_config, &mut rng);
        assert!(eval.fit_rate >= 0.0 && eval.fit_rate <= 1.0);
    }
}
