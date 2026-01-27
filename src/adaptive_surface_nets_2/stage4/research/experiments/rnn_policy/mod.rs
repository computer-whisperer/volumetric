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

use classifier_heads::{compute_classifier_loss, ClassifierLossConfig, ExpectedGeometry, GeometryType};
use gru::Rng;
use policy::{run_episode, RnnPolicy};
use reward::RewardConfig;
use training::{train_policy, evaluate_policy, TrainingConfig};
use crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification;

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
            // Use stochastic=true to match training behavior
            let episode = run_episode(policy, cube, point, idx, rng, true, &reward_config);
            let predictions = policy.classify(&episode.h_final);

            let expected_geom = ExpectedGeometry::from_classification(&point.expected, point.position);

            let predicted_type = predictions.predicted_type();
            let expected_type = expected_geom.geometry_type();
            if predicted_type == expected_type {
                correct += 1;
            }

            let (loss, _, _, _, _) = compute_classifier_loss(&predictions, &expected_geom, &loss_config);
            total_loss += loss;
        }

        (correct as f64 / points.len() as f64, total_loss / points.len() as f64)
    };

    // Evaluate before training
    let (acc_before, loss_before) = eval_classifier(&policy, &cube, &points, &mut rng);
    println!("\nBefore training:");
    println!("  Classifier accuracy: {:.1}% (loss: {:.3})", acc_before * 100.0, loss_before);

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
    println!("  Classifier accuracy: {:.1}% (loss: {:.3})", acc_after * 100.0, loss_after);

    // Show classifier confidence breakdown
    print_classifier_breakdown(&policy, &cube, &points, &reward_config, &mut rng);
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
        // Use stochastic=true to match training behavior
        let episode = run_episode(policy, cube, point, idx, rng, true, reward_config);
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
        "Policy parameters: {} (GRU: {}, Chooser: {}, Heads: {})",
        policy.param_count(),
        policy.gru.param_count(),
        policy.chooser.param_count(),
        policy.classifier_heads.param_count()
    );

    // Evaluate classifier heads before training
    let eval_before = evaluate_classifier_heads(&policy, &cube, &points, &reward_config, &mut rng);
    println!("\nBefore training:");
    println!("  Classifier accuracy: {:.1}% (loss: {:.3})", eval_before.accuracy * 100.0, eval_before.loss);
    println!("  {:10} {:>8} {:>8} {:>12}", "Class", "TypeAcc", "Conf", "NormalLoss");
    println!("  {:10} {:>3}/{:<2}={:3.0}% {:>7.3} {:>12.3}", "Face",
        eval_before.face_correct, eval_before.face_total,
        if eval_before.face_total > 0 { eval_before.face_correct as f64 / eval_before.face_total as f64 * 100.0 } else { 0.0 },
        eval_before.face_conf, eval_before.face_normal_loss);
    println!("  {:10} {:>3}/{:<2}={:3.0}% {:>7.3} {:>12.3}", "Edge",
        eval_before.edge_correct, eval_before.edge_total,
        if eval_before.edge_total > 0 { eval_before.edge_correct as f64 / eval_before.edge_total as f64 * 100.0 } else { 0.0 },
        eval_before.edge_conf, eval_before.edge_normal_loss);
    println!("  {:10} {:>3}/{:<2}={:3.0}% {:>7.3} {:>12.3}", "Corner",
        eval_before.corner_correct, eval_before.corner_total,
        if eval_before.corner_total > 0 { eval_before.corner_correct as f64 / eval_before.corner_total as f64 * 100.0 } else { 0.0 },
        eval_before.corner_conf, eval_before.corner_normal_loss);

    // Train with classifier heads
    println!("\nTraining (24000 epochs, lr=0.0003, hidden=64)...");
    let config = TrainingConfig {
        epochs: 24000,
        print_every: 1000,
        lr: 0.0003,
        use_classifier_heads: true,
        ..Default::default()
    };
    train_policy(&mut policy, &cube, &points, &config, &reward_config, &mut rng);

    // Evaluate classifier heads after training
    let eval_after = evaluate_classifier_heads(&policy, &cube, &points, &reward_config, &mut rng);
    println!("\nAfter training:");
    println!("  Classifier accuracy: {:.1}% (loss: {:.3})", eval_after.accuracy * 100.0, eval_after.loss);
    println!("  {:10} {:>8} {:>8} {:>12}", "Class", "TypeAcc", "Conf", "NormalLoss");
    println!("  {:10} {:>3}/{:<2}={:3.0}% {:>7.3} {:>12.3}", "Face",
        eval_after.face_correct, eval_after.face_total,
        if eval_after.face_total > 0 { eval_after.face_correct as f64 / eval_after.face_total as f64 * 100.0 } else { 0.0 },
        eval_after.face_conf, eval_after.face_normal_loss);
    println!("  {:10} {:>3}/{:<2}={:3.0}% {:>7.3} {:>12.3}", "Edge",
        eval_after.edge_correct, eval_after.edge_total,
        if eval_after.edge_total > 0 { eval_after.edge_correct as f64 / eval_after.edge_total as f64 * 100.0 } else { 0.0 },
        eval_after.edge_conf, eval_after.edge_normal_loss);
    println!("  {:10} {:>3}/{:<2}={:3.0}% {:>7.3} {:>12.3}", "Corner",
        eval_after.corner_correct, eval_after.corner_total,
        if eval_after.corner_total > 0 { eval_after.corner_correct as f64 / eval_after.corner_total as f64 * 100.0 } else { 0.0 },
        eval_after.corner_conf, eval_after.corner_normal_loss);

    // Save
    if let Err(e) = policy.save(model_path) {
        eprintln!("Failed to save model: {}", e);
    } else {
        println!("\nSaved model to {}", model_path.display());
    }
}

/// Compare training with and without exploration schedule.
/// Runs both approaches with the same initial seed and compares results.
pub fn compare_exploration_schedule() {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);
    let reward_config = RewardConfig::default();

    // Shorter training for comparison (still substantial)
    let epochs = 6000;
    let print_every = 500;
    let lr = 0.0003;

    println!("=== Exploration Schedule Comparison ===");
    println!("Training points: {}", points.len());
    println!("Epochs: {}, LR: {}", epochs, lr);
    println!();

    // --- Baseline (no exploration schedule) ---
    println!(">>> BASELINE (no exploration schedule) <<<");
    let mut rng_baseline = Rng::new(12345);
    let mut policy_baseline = RnnPolicy::new(&mut rng_baseline);

    let eval_before = evaluate_classifier_heads(&policy_baseline, &cube, &points, &reward_config, &mut rng_baseline);
    println!("Before: acc={:.1}%, loss={:.3}", eval_before.accuracy * 100.0, eval_before.loss);

    let config_baseline = TrainingConfig {
        epochs,
        print_every,
        lr,
        use_classifier_heads: true,
        use_exploration_schedule: false,
        ..Default::default()
    };
    train_policy(&mut policy_baseline, &cube, &points, &config_baseline, &reward_config, &mut rng_baseline);

    let eval_baseline = evaluate_classifier_heads(&policy_baseline, &cube, &points, &reward_config, &mut rng_baseline);
    println!("\nBaseline final results:");
    println!("  Accuracy: {:.1}% (loss: {:.3})", eval_baseline.accuracy * 100.0, eval_baseline.loss);
    println!("  Face:   {}/{} = {:.0}%, normal_loss={:.3}",
        eval_baseline.face_correct, eval_baseline.face_total,
        if eval_baseline.face_total > 0 { eval_baseline.face_correct as f64 / eval_baseline.face_total as f64 * 100.0 } else { 0.0 },
        eval_baseline.face_normal_loss);
    println!("  Edge:   {}/{} = {:.0}%, normal_loss={:.3}",
        eval_baseline.edge_correct, eval_baseline.edge_total,
        if eval_baseline.edge_total > 0 { eval_baseline.edge_correct as f64 / eval_baseline.edge_total as f64 * 100.0 } else { 0.0 },
        eval_baseline.edge_normal_loss);
    println!("  Corner: {}/{} = {:.0}%, normal_loss={:.3}",
        eval_baseline.corner_correct, eval_baseline.corner_total,
        if eval_baseline.corner_total > 0 { eval_baseline.corner_correct as f64 / eval_baseline.corner_total as f64 * 100.0 } else { 0.0 },
        eval_baseline.corner_normal_loss);

    println!();

    // --- With exploration schedule ---
    println!(">>> WITH EXPLORATION SCHEDULE (100% -> 0%) <<<");
    let mut rng_explore = Rng::new(12345); // Same seed for fair comparison
    let mut policy_explore = RnnPolicy::new(&mut rng_explore);

    let config_explore = TrainingConfig {
        epochs,
        print_every,
        lr,
        use_classifier_heads: true,
        use_exploration_schedule: true,
        exploration_start: 1.0,
        exploration_end: 0.0,
        ..Default::default()
    };
    train_policy(&mut policy_explore, &cube, &points, &config_explore, &reward_config, &mut rng_explore);

    let eval_explore = evaluate_classifier_heads(&policy_explore, &cube, &points, &reward_config, &mut rng_explore);
    println!("\nExploration schedule final results:");
    println!("  Accuracy: {:.1}% (loss: {:.3})", eval_explore.accuracy * 100.0, eval_explore.loss);
    println!("  Face:   {}/{} = {:.0}%, normal_loss={:.3}",
        eval_explore.face_correct, eval_explore.face_total,
        if eval_explore.face_total > 0 { eval_explore.face_correct as f64 / eval_explore.face_total as f64 * 100.0 } else { 0.0 },
        eval_explore.face_normal_loss);
    println!("  Edge:   {}/{} = {:.0}%, normal_loss={:.3}",
        eval_explore.edge_correct, eval_explore.edge_total,
        if eval_explore.edge_total > 0 { eval_explore.edge_correct as f64 / eval_explore.edge_total as f64 * 100.0 } else { 0.0 },
        eval_explore.edge_normal_loss);
    println!("  Corner: {}/{} = {:.0}%, normal_loss={:.3}",
        eval_explore.corner_correct, eval_explore.corner_total,
        if eval_explore.corner_total > 0 { eval_explore.corner_correct as f64 / eval_explore.corner_total as f64 * 100.0 } else { 0.0 },
        eval_explore.corner_normal_loss);

    // Summary comparison
    println!("\n=== COMPARISON SUMMARY ===");
    println!("                    Baseline    Exploration");
    println!("  Accuracy:         {:>6.1}%      {:>6.1}%",
        eval_baseline.accuracy * 100.0, eval_explore.accuracy * 100.0);
    println!("  Loss:             {:>6.3}       {:>6.3}",
        eval_baseline.loss, eval_explore.loss);
    println!("  Face normal:      {:>6.3}       {:>6.3}",
        eval_baseline.face_normal_loss, eval_explore.face_normal_loss);
    println!("  Edge normal:      {:>6.3}       {:>6.3}",
        eval_baseline.edge_normal_loss, eval_explore.edge_normal_loss);
    println!("  Corner normal:    {:>6.3}       {:>6.3}",
        eval_baseline.corner_normal_loss, eval_explore.corner_normal_loss);

    let acc_diff = eval_explore.accuracy - eval_baseline.accuracy;
    let loss_diff = eval_baseline.loss - eval_explore.loss; // positive = explore is better
    println!("\n  Accuracy improvement: {:+.1}%", acc_diff * 100.0);
    println!("  Loss improvement:     {:+.3}", loss_diff);
}

/// Train without exploration schedule (baseline) and dump sample clouds.
pub fn dump_baseline_sample_cloud(output_path: &Path) {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);
    let reward_config = RewardConfig::default();

    let epochs = 6000;
    let lr = 0.0003;

    println!("=== Training Baseline (no exploration) for Sample Cloud Dump ===");
    println!("Epochs: {}, LR: {}", epochs, lr);

    let mut rng = Rng::new(12345);
    let mut policy = RnnPolicy::new(&mut rng);

    let config = TrainingConfig {
        epochs,
        print_every: 1000,
        lr,
        use_classifier_heads: true,
        use_exploration_schedule: false,
        ..Default::default()
    };
    train_policy(&mut policy, &cube, &points, &config, &reward_config, &mut rng);

    // Evaluate
    let eval = evaluate_classifier_heads(&policy, &cube, &points, &reward_config, &mut rng);
    println!("\nFinal results:");
    println!("  Accuracy: {:.1}% (loss: {:.3})", eval.accuracy * 100.0, eval.loss);
    println!("  Face:   {}/{} = {:.0}%", eval.face_correct, eval.face_total,
        if eval.face_total > 0 { eval.face_correct as f64 / eval.face_total as f64 * 100.0 } else { 0.0 });
    println!("  Edge:   {}/{} = {:.0}%", eval.edge_correct, eval.edge_total,
        if eval.edge_total > 0 { eval.edge_correct as f64 / eval.edge_total as f64 * 100.0 } else { 0.0 });
    println!("  Corner: {}/{} = {:.0}%", eval.corner_correct, eval.corner_total,
        if eval.corner_total > 0 { eval.corner_correct as f64 / eval.corner_total as f64 * 100.0 } else { 0.0 });

    // Dump sample clouds (use_discrete=false for weighted lerp positions)
    println!("\nDumping sample clouds to {}...", output_path.display());
    dump_rnn_policy_sample_cloud_with_policy(&policy, "rnn-baseline", output_path);
    println!("Done!");
}

/// Train with exploration schedule and dump sample clouds.
pub fn dump_exploration_schedule_sample_cloud(output_path: &Path) {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);
    let reward_config = RewardConfig::default();

    let epochs = 6000;
    let lr = 0.0003;

    println!("=== Training with Exploration Schedule for Sample Cloud Dump ===");
    println!("Epochs: {}, LR: {}", epochs, lr);

    let mut rng = Rng::new(12345);
    let mut policy = RnnPolicy::new(&mut rng);

    let config = TrainingConfig {
        epochs,
        print_every: 1000,
        lr,
        use_classifier_heads: true,
        use_exploration_schedule: true,
        exploration_start: 1.0,
        exploration_end: 0.0,
        ..Default::default()
    };
    train_policy(&mut policy, &cube, &points, &config, &reward_config, &mut rng);

    // Evaluate
    let eval = evaluate_classifier_heads(&policy, &cube, &points, &reward_config, &mut rng);
    println!("\nFinal results:");
    println!("  Accuracy: {:.1}% (loss: {:.3})", eval.accuracy * 100.0, eval.loss);
    println!("  Face:   {}/{} = {:.0}%", eval.face_correct, eval.face_total,
        if eval.face_total > 0 { eval.face_correct as f64 / eval.face_total as f64 * 100.0 } else { 0.0 });
    println!("  Edge:   {}/{} = {:.0}%", eval.edge_correct, eval.edge_total,
        if eval.edge_total > 0 { eval.edge_correct as f64 / eval.edge_total as f64 * 100.0 } else { 0.0 });
    println!("  Corner: {}/{} = {:.0}%", eval.corner_correct, eval.corner_total,
        if eval.corner_total > 0 { eval.corner_correct as f64 / eval.corner_total as f64 * 100.0 } else { 0.0 });

    // Dump sample clouds (use_discrete=false for weighted lerp positions)
    println!("\nDumping sample clouds to {}...", output_path.display());
    dump_rnn_policy_sample_cloud_with_policy(&policy, "rnn-exploration-schedule", output_path);
    println!("Done!");
}

/// Classifier evaluation result with both accuracy and confidence.
#[derive(Debug)]
pub struct ClassifierEvalResult {
    pub accuracy: f64,
    pub loss: f64,
    pub face_correct: usize,
    pub face_total: usize,
    pub face_conf: f64, // Avg confidence of face head on face points
    pub face_normal_loss: f64, // Avg normal cosine loss for face predictions
    pub edge_correct: usize,
    pub edge_total: usize,
    pub edge_conf: f64,
    pub edge_normal_loss: f64,
    pub corner_correct: usize,
    pub corner_total: usize,
    pub corner_conf: f64,
    pub corner_normal_loss: f64, // Avg normal cosine loss for corner predictions
}

/// Compute cosine loss between predicted and expected normals.
/// Returns 1 - |dot(pred, exp)|, so 0 = perfect, 1 = perpendicular.
fn normal_cosine_loss(pred: (f64, f64, f64), exp: (f64, f64, f64)) -> f64 {
    let d = pred.0 * exp.0 + pred.1 * exp.1 + pred.2 * exp.2;
    1.0 - d.abs().min(1.0)
}

/// Compute best matching loss for 3 normals (greedy assignment).
fn corner_normal_loss(pred: &[(f64, f64, f64); 3], exp: &[(f64, f64, f64); 3]) -> f64 {
    let mut total = 0.0;
    let mut used = [false; 3];

    for p in pred {
        let mut best = f64::MAX;
        let mut best_idx = 0;
        for (i, e) in exp.iter().enumerate() {
            if !used[i] {
                let loss = normal_cosine_loss(*p, *e);
                if loss < best {
                    best = loss;
                    best_idx = i;
                }
            }
        }
        used[best_idx] = true;
        total += best;
    }
    total / 3.0
}

/// Evaluate classifier head accuracy and confidence.
fn evaluate_classifier_heads(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    points: &[crate::adaptive_surface_nets_2::stage4::research::validation::ValidationPoint],
    reward_config: &RewardConfig,
    rng: &mut Rng,
) -> ClassifierEvalResult {
    let loss_config = ClassifierLossConfig::default();

    let mut correct = 0;
    let mut total_loss = 0.0;
    let mut face_correct = 0;
    let mut face_total = 0;
    let mut face_conf_sum = 0.0;
    let mut face_normal_loss_sum = 0.0;
    let mut edge_correct = 0;
    let mut edge_total = 0;
    let mut edge_conf_sum = 0.0;
    let mut edge_normal_loss_sum = 0.0;
    let mut corner_correct = 0;
    let mut corner_total = 0;
    let mut corner_conf_sum = 0.0;
    let mut corner_normal_loss_sum = 0.0;

    for (idx, point) in points.iter().enumerate() {
        let episode = run_episode(policy, cube, point, idx, rng, false, reward_config);
        let predictions = policy.classify(&episode.h_final);

        let expected_geom = ExpectedGeometry::from_classification(&point.expected, point.position);

        let predicted_type = predictions.predicted_type();
        let expected_type = expected_geom.geometry_type();

        // Track per-type accuracy, confidence, and normal quality
        match &expected_geom {
            ExpectedGeometry::Face { normal, .. } => {
                face_total += 1;
                face_conf_sum += predictions.face.confidence;
                face_normal_loss_sum += normal_cosine_loss(predictions.face.normal, *normal);
                if predicted_type == expected_type {
                    face_correct += 1;
                    correct += 1;
                }
            }
            ExpectedGeometry::Edge { normal_a, normal_b, .. } => {
                edge_total += 1;
                edge_conf_sum += predictions.edge.confidence;
                // Best matching of 2 normals
                let loss1 = normal_cosine_loss(predictions.edge.normal_a, *normal_a)
                    + normal_cosine_loss(predictions.edge.normal_b, *normal_b);
                let loss2 = normal_cosine_loss(predictions.edge.normal_a, *normal_b)
                    + normal_cosine_loss(predictions.edge.normal_b, *normal_a);
                edge_normal_loss_sum += loss1.min(loss2) / 2.0;
                if predicted_type == expected_type {
                    edge_correct += 1;
                    correct += 1;
                }
            }
            ExpectedGeometry::Corner { normals, .. } => {
                corner_total += 1;
                corner_conf_sum += predictions.corner.confidence;
                corner_normal_loss_sum += corner_normal_loss(&predictions.corner.normals, normals);
                if predicted_type == expected_type {
                    corner_correct += 1;
                    correct += 1;
                }
            }
        }

        let (loss, _, _, _, _) = compute_classifier_loss(&predictions, &expected_geom, &loss_config);
        total_loss += loss;
    }

    ClassifierEvalResult {
        accuracy: correct as f64 / points.len() as f64,
        loss: total_loss / points.len() as f64,
        face_correct,
        face_total,
        face_conf: if face_total > 0 { face_conf_sum / face_total as f64 } else { 0.0 },
        face_normal_loss: if face_total > 0 { face_normal_loss_sum / face_total as f64 } else { 0.0 },
        edge_correct,
        edge_total,
        edge_conf: if edge_total > 0 { edge_conf_sum / edge_total as f64 } else { 0.0 },
        edge_normal_loss: if edge_total > 0 { edge_normal_loss_sum / edge_total as f64 } else { 0.0 },
        corner_correct,
        corner_total,
        corner_conf: if corner_total > 0 { corner_conf_sum / corner_total as f64 } else { 0.0 },
        corner_normal_loss: if corner_total > 0 { corner_normal_loss_sum / corner_total as f64 } else { 0.0 },
    }
}

/// Load a trained RNN policy and dump sample clouds.
pub fn load_and_dump_rnn_policy(model_path: &Path, output_path: &Path) {
    let policy = match RnnPolicy::load(model_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load model from {}: {}", model_path.display(), e);
            return;
        }
    };
    println!("Loaded model from {}", model_path.display());

    dump_rnn_policy_sample_cloud_with_policy(&policy, "rnn-trained", output_path);
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

    dump_rnn_policy_sample_cloud_with_policy(&policy, label, output_path);
}

/// Dump sample clouds using a pre-loaded policy.
fn dump_rnn_policy_sample_cloud_with_policy(policy: &RnnPolicy, label: &str, output_path: &Path) {
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

        // stochastic=false (use argmax) for deterministic evaluation
        let episode = run_episode(policy, &cube, point, idx, &mut rng, false, &reward_config);

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
        // Offset represents signed distance from vertex to plane (negative = inside surface)
        // We visualize by showing a point on the surface: vertex + normal * (-offset)
        let vertex = to_f32(point.position);
        match predicted_type {
            GeometryType::Face => {
                // Face: single normal (green) with surface point (cyan)
                set.add_vector(
                    "pred_normal",
                    vertex,
                    scale_vec(to_f32(predictions.face.normal), vec_scale),
                    Some([0.3, 1.0, 0.3, 1.0]),
                );
                // Show predicted surface point: move along normal by -offset * cell_size
                let surface_pt = offset_point(vertex, to_f32(predictions.face.normal), predictions.face.offset, policy::CELL_SIZE as f32);
                set.add_vector(
                    "pred_surface",
                    vertex,
                    [surface_pt[0] - vertex[0], surface_pt[1] - vertex[1], surface_pt[2] - vertex[2]],
                    Some([0.0, 1.0, 1.0, 1.0]), // cyan
                );
            }
            GeometryType::Edge => {
                // Edge: two normals (red, orange), edge direction (yellow), and surface points
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
                // Show predicted surface points for both faces
                let surface_a = offset_point(vertex, to_f32(predictions.edge.normal_a), predictions.edge.offset_a, policy::CELL_SIZE as f32);
                let surface_b = offset_point(vertex, to_f32(predictions.edge.normal_b), predictions.edge.offset_b, policy::CELL_SIZE as f32);
                set.add_vector(
                    "pred_surface_a",
                    vertex,
                    [surface_a[0] - vertex[0], surface_a[1] - vertex[1], surface_a[2] - vertex[2]],
                    Some([1.0, 0.5, 0.5, 1.0]), // light red
                );
                set.add_vector(
                    "pred_surface_b",
                    vertex,
                    [surface_b[0] - vertex[0], surface_b[1] - vertex[1], surface_b[2] - vertex[2]],
                    Some([1.0, 0.8, 0.5, 1.0]), // light orange
                );
            }
            GeometryType::Corner => {
                // Corner: three normals (red, green, blue) with surface points
                let colors = [
                    [1.0, 0.3, 0.3, 1.0],
                    [0.3, 1.0, 0.3, 1.0],
                    [0.3, 0.3, 1.0, 1.0],
                ];
                let surface_colors = [
                    [1.0, 0.6, 0.6, 1.0], // light red
                    [0.6, 1.0, 0.6, 1.0], // light green
                    [0.6, 0.6, 1.0, 1.0], // light blue
                ];
                for (i, normal) in predictions.corner.normals.iter().enumerate() {
                    set.add_vector(
                        &format!("pred_normal_{}", i),
                        vertex,
                        scale_vec(to_f32(*normal), vec_scale),
                        Some(colors[i]),
                    );
                    // Show predicted surface point
                    let surface_pt = offset_point(vertex, to_f32(*normal), predictions.corner.offsets[i], policy::CELL_SIZE as f32);
                    set.add_vector(
                        &format!("pred_surface_{}", i),
                        vertex,
                        [surface_pt[0] - vertex[0], surface_pt[1] - vertex[1], surface_pt[2] - vertex[2]],
                        Some(surface_colors[i]),
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

/// Compute surface point from vertex, normal, and offset.
/// offset is normalized by cell_size, so surface = vertex - offset * cell_size * normal
fn offset_point(vertex: [f32; 3], normal: [f32; 3], offset: f64, cell_size: f32) -> [f32; 3] {
    let off = (offset as f32) * cell_size;
    [
        vertex[0] - off * normal[0],
        vertex[1] - off * normal[1],
        vertex[2] - off * normal[2],
    ]
}

/// Result of a single sweep configuration.
#[derive(Debug)]
pub struct SweepResult {
    pub name: String,
    pub accuracy: f64,
    pub loss: f64,
    pub face_correct: usize,
    pub face_total: usize,
    pub face_conf: f64,
    pub face_normal_loss: f64, // Cosine loss for face normals (0=perfect, 1=perpendicular)
    pub edge_correct: usize,
    pub edge_total: usize,
    pub edge_conf: f64,
    pub edge_normal_loss: f64,
    pub corner_correct: usize,
    pub corner_total: usize,
    pub corner_conf: f64,
    pub corner_normal_loss: f64,
}

impl SweepResult {
    pub fn face_pct(&self) -> f64 {
        if self.face_total == 0 { 0.0 } else { self.face_correct as f64 / self.face_total as f64 * 100.0 }
    }
    pub fn edge_pct(&self) -> f64 {
        if self.edge_total == 0 { 0.0 } else { self.edge_correct as f64 / self.edge_total as f64 * 100.0 }
    }
    pub fn corner_pct(&self) -> f64 {
        if self.corner_total == 0 { 0.0 } else { self.corner_correct as f64 / self.corner_total as f64 * 100.0 }
    }
}

/// Run a sweep of different reward configurations.
pub fn run_reward_sweep() {
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);

    println!("\n{}", "=".repeat(80));
    println!("REWARD CONFIGURATION SWEEP");
    println!("{}", "=".repeat(80));
    println!("Points: {} (Face: 12, Edge: 24, Corner: 16)", points.len());
    println!("Epochs: 3000 per config");
    println!();

    // Define configurations to test
    let configs: Vec<(&str, RewardConfig)> = vec![
        ("baseline", RewardConfig::default()),
        ("classifier_training", RewardConfig::for_classifier_training()),
        ("direction_diversity", RewardConfig::with_direction_diversity()),
        ("high_crossing", RewardConfig::with_high_crossing_bonus()),
        // Variations on entropy
        ("entropy_0.1", RewardConfig {
            w_entropy: 0.1,
            ..RewardConfig::default()
        }),
        ("entropy_0.3", RewardConfig {
            w_entropy: 0.3,
            ..RewardConfig::default()
        }),
        // Variations on direction diversity
        ("dir_div_0.1", RewardConfig {
            w_direction_diversity: 0.1,
            w_outside_spread: 0.1,
            ..RewardConfig::default()
        }),
        ("dir_div_0.2", RewardConfig {
            w_direction_diversity: 0.2,
            w_outside_spread: 0.2,
            ..RewardConfig::default()
        }),
        // Combined strategies
        ("entropy+div", RewardConfig {
            w_entropy: 0.1,
            w_direction_diversity: 0.1,
            w_outside_spread: 0.1,
            ..RewardConfig::default()
        }),
        // Lower crossing bonus (was causing reward hacking before)
        ("no_crossing", RewardConfig {
            crossing_bonus: 0.0,
            w_spread: 0.2,
            ..RewardConfig::default()
        }),
        // High spread
        ("high_spread", RewardConfig {
            w_spread: 0.3,
            crossing_bonus: 0.2,
            ..RewardConfig::default()
        }),
    ];

    let training_config = TrainingConfig {
        epochs: 3000,
        print_every: 3001, // Don't print per-epoch
        lr: 0.001,
        use_classifier_heads: true,
        ..Default::default()
    };

    let mut results: Vec<SweepResult> = Vec::new();

    for (name, reward_config) in &configs {
        print!("Testing {:20} ... ", name);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let mut rng = Rng::new(12345); // Same seed for fair comparison
        let mut policy = RnnPolicy::new(&mut rng);

        // Train
        train_policy(&mut policy, &cube, &points, &training_config, reward_config, &mut rng);

        // Evaluate
        let eval = evaluate_classifier_heads(&policy, &cube, &points, reward_config, &mut rng);

        let result = SweepResult {
            name: name.to_string(),
            accuracy: eval.accuracy,
            loss: eval.loss,
            face_correct: eval.face_correct,
            face_total: eval.face_total,
            face_conf: eval.face_conf,
            face_normal_loss: eval.face_normal_loss,
            edge_correct: eval.edge_correct,
            edge_total: eval.edge_total,
            edge_conf: eval.edge_conf,
            edge_normal_loss: eval.edge_normal_loss,
            corner_correct: eval.corner_correct,
            corner_total: eval.corner_total,
            corner_conf: eval.corner_conf,
            corner_normal_loss: eval.corner_normal_loss,
        };

        // Show type accuracy and normal loss
        println!(
            "TypeAcc: F={:2}/{:2} E={:2}/{:2} C={:2}/{:2} | NormLoss: F={:.3} E={:.3} C={:.3}",
            result.face_correct, result.face_total,
            result.edge_correct, result.edge_total,
            result.corner_correct, result.corner_total,
            result.face_normal_loss, result.edge_normal_loss, result.corner_normal_loss,
        );

        results.push(result);
    }

    // Summary - Accuracy table
    println!("\n{}", "-".repeat(90));
    println!("ACCURACY (sorted by overall)");
    println!("{}", "-".repeat(90));

    results.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

    println!("{:20} {:>6} {:>10} {:>10} {:>10}", "Config", "Total", "Face", "Edge", "Corner");
    println!("{}", "-".repeat(60));
    for r in &results {
        println!(
            "{:20} {:5.1}% {:>4}/{:<2}={:4.0}% {:>4}/{:<2}={:4.0}% {:>4}/{:<2}={:4.0}%",
            r.name,
            r.accuracy * 100.0,
            r.face_correct, r.face_total, r.face_pct(),
            r.edge_correct, r.edge_total, r.edge_pct(),
            r.corner_correct, r.corner_total, r.corner_pct(),
        );
    }

    // Normal loss table (sorted by corner normal loss - lower is better)
    println!("\n{}", "-".repeat(90));
    println!("NORMAL LOSS (cosine loss, 0=perfect, 1=perpendicular) - sorted by corner");
    println!("{}", "-".repeat(90));

    results.sort_by(|a, b| a.corner_normal_loss.partial_cmp(&b.corner_normal_loss).unwrap());

    println!("{:20} {:>12} {:>12} {:>12}", "Config", "Face Loss", "Edge Loss", "Corner Loss");
    println!("{}", "-".repeat(60));
    for r in &results {
        println!(
            "{:20} {:>12.3} {:>12.3} {:>12.3}",
            r.name,
            r.face_normal_loss,
            r.edge_normal_loss,
            r.corner_normal_loss,
        );
    }

    // Confidence table
    println!("\n{}", "-".repeat(90));
    println!("CONFIDENCE for correct class (sorted by corner conf)");
    println!("{}", "-".repeat(90));

    results.sort_by(|a, b| b.corner_conf.partial_cmp(&a.corner_conf).unwrap());

    println!("{:20} {:>10} {:>10} {:>10}", "Config", "Face Conf", "Edge Conf", "Corner Conf");
    println!("{}", "-".repeat(55));
    for r in &results {
        println!(
            "{:20} {:>10.3} {:>10.3} {:>10.3}",
            r.name,
            r.face_conf,
            r.edge_conf,
            r.corner_conf,
        );
    }

    // Find best for each category
    println!("\nBest accuracy per category:");
    if let Some(best_face) = results.iter().max_by(|a, b| a.face_pct().partial_cmp(&b.face_pct()).unwrap()) {
        println!("  Face:   {} ({:.1}%)", best_face.name, best_face.face_pct());
    }
    if let Some(best_edge) = results.iter().max_by(|a, b| a.edge_pct().partial_cmp(&b.edge_pct()).unwrap()) {
        println!("  Edge:   {} ({:.1}%)", best_edge.name, best_edge.edge_pct());
    }
    if let Some(best_corner) = results.iter().max_by(|a, b| a.corner_pct().partial_cmp(&b.corner_pct()).unwrap()) {
        println!("  Corner: {} ({:.1}%)", best_corner.name, best_corner.corner_pct());
    }

    println!("\nBest confidence per category:");
    if let Some(best) = results.iter().max_by(|a, b| a.face_conf.partial_cmp(&b.face_conf).unwrap()) {
        println!("  Face:   {} ({:.3})", best.name, best.face_conf);
    }
    if let Some(best) = results.iter().max_by(|a, b| a.edge_conf.partial_cmp(&b.edge_conf).unwrap()) {
        println!("  Edge:   {} ({:.3})", best.name, best.edge_conf);
    }
    if let Some(best) = results.iter().max_by(|a, b| a.corner_conf.partial_cmp(&b.corner_conf).unwrap()) {
        println!("  Corner: {} ({:.3})", best.name, best.corner_conf);
    }

    println!("\nBest normal loss per category (lower = better):");
    if let Some(best) = results.iter().min_by(|a, b| a.face_normal_loss.partial_cmp(&b.face_normal_loss).unwrap()) {
        println!("  Face:   {} ({:.3})", best.name, best.face_normal_loss);
    }
    if let Some(best) = results.iter().min_by(|a, b| a.edge_normal_loss.partial_cmp(&b.edge_normal_loss).unwrap()) {
        println!("  Edge:   {} ({:.3})", best.name, best.edge_normal_loss);
    }
    if let Some(best) = results.iter().min_by(|a, b| a.corner_normal_loss.partial_cmp(&b.corner_normal_loss).unwrap()) {
        println!("  Corner: {} ({:.3})", best.name, best.corner_normal_loss);
    }
}

/// Run a diagnostic sweep to understand why corners aren't being classified.
pub fn run_corner_diagnostic() {
    use classifier_heads::{GeometryType, ClassifierLossConfig};

    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);

    println!("\n{}", "=".repeat(80));
    println!("CORNER CLASSIFICATION DIAGNOSTIC");
    println!("{}", "=".repeat(80));

    let reward_config = RewardConfig::with_direction_diversity();

    let training_config = TrainingConfig {
        epochs: 6000,
        print_every: 1000,
        lr: 0.001,
        use_classifier_heads: true,
        ..Default::default()
    };

    let mut rng = Rng::new(12345);
    let mut policy = RnnPolicy::new(&mut rng);

    // Evaluate before training
    println!("\n=== BEFORE TRAINING ===");
    print_confidence_accuracy_table(&policy, &cube, &points, &reward_config, &mut rng);

    println!("\nTraining for 6000 epochs...");
    train_policy(&mut policy, &cube, &points, &training_config, &reward_config, &mut rng);

    println!("\n=== AFTER TRAINING ===");
    print_confidence_accuracy_table(&policy, &cube, &points, &reward_config, &mut rng);
}

/// Print a table showing both confidence (for correct class) and accuracy.
fn print_confidence_accuracy_table(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    points: &[crate::adaptive_surface_nets_2::stage4::research::validation::ValidationPoint],
    reward_config: &RewardConfig,
    rng: &mut Rng,
) {
    use classifier_heads::GeometryType;

    // Track per-class: correct count, total count, sum of correct-class confidence
    let mut face_correct = 0usize;
    let mut face_total = 0usize;
    let mut face_conf_sum = 0.0f64; // Sum of face confidence for face points

    let mut edge_correct = 0usize;
    let mut edge_total = 0usize;
    let mut edge_conf_sum = 0.0f64;

    let mut corner_correct = 0usize;
    let mut corner_total = 0usize;
    let mut corner_conf_sum = 0.0f64;

    // Also track what corners are predicted as
    let mut corner_pred_dist = [0usize; 3]; // [face, edge, corner]

    for (idx, point) in points.iter().enumerate() {
        let episode = run_episode(policy, cube, point, idx, rng, false, reward_config);
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
                if predicted_type == GeometryType::Face {
                    face_correct += 1;
                }
            }
            GeometryType::Edge => {
                edge_total += 1;
                edge_conf_sum += predictions.edge.confidence;
                if predicted_type == GeometryType::Edge {
                    edge_correct += 1;
                }
            }
            GeometryType::Corner => {
                corner_total += 1;
                corner_conf_sum += predictions.corner.confidence;
                if predicted_type == GeometryType::Corner {
                    corner_correct += 1;
                }
                // Track what corners are predicted as
                match predicted_type {
                    GeometryType::Face => corner_pred_dist[0] += 1,
                    GeometryType::Edge => corner_pred_dist[1] += 1,
                    GeometryType::Corner => corner_pred_dist[2] += 1,
                }
            }
        }
    }

    // Print table
    println!("\n{:10} {:>10} {:>10} {:>15}", "Class", "Accuracy", "Avg Conf", "Correct Conf");
    println!("{}", "-".repeat(50));

    let face_acc = if face_total > 0 { face_correct as f64 / face_total as f64 * 100.0 } else { 0.0 };
    let face_avg_conf = if face_total > 0 { face_conf_sum / face_total as f64 } else { 0.0 };
    println!("{:10} {:>9.1}% {:>10.3} {:>6}/{:>2} = {:.1}%",
        "Face", face_acc, face_avg_conf, face_correct, face_total,
        face_acc);

    let edge_acc = if edge_total > 0 { edge_correct as f64 / edge_total as f64 * 100.0 } else { 0.0 };
    let edge_avg_conf = if edge_total > 0 { edge_conf_sum / edge_total as f64 } else { 0.0 };
    println!("{:10} {:>9.1}% {:>10.3} {:>6}/{:>2} = {:.1}%",
        "Edge", edge_acc, edge_avg_conf, edge_correct, edge_total,
        edge_acc);

    let corner_acc = if corner_total > 0 { corner_correct as f64 / corner_total as f64 * 100.0 } else { 0.0 };
    let corner_avg_conf = if corner_total > 0 { corner_conf_sum / corner_total as f64 } else { 0.0 };
    println!("{:10} {:>9.1}% {:>10.3} {:>6}/{:>2} = {:.1}%",
        "Corner", corner_acc, corner_avg_conf, corner_correct, corner_total,
        corner_acc);

    // Show corner prediction distribution
    if corner_total > 0 {
        println!("\nCorner points predicted as:");
        println!("  Face:   {:2}/{:2} ({:.1}%)", corner_pred_dist[0], corner_total,
            corner_pred_dist[0] as f64 / corner_total as f64 * 100.0);
        println!("  Edge:   {:2}/{:2} ({:.1}%)", corner_pred_dist[1], corner_total,
            corner_pred_dist[1] as f64 / corner_total as f64 * 100.0);
        println!("  Corner: {:2}/{:2} ({:.1}%)", corner_pred_dist[2], corner_total,
            corner_pred_dist[2] as f64 / corner_total as f64 * 100.0);
    }
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

    /// Run exploration schedule comparison.
    /// This test is ignored by default as it takes several minutes.
    /// Run with: cargo test --release --features native compare_exploration -- --ignored --nocapture
    #[test]
    #[ignore]
    fn compare_exploration() {
        compare_exploration_schedule();
    }

    /// Dump sample cloud from exploration schedule training.
    /// Run with: cargo test --release --features native test_dump_exploration_sample_cloud -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_dump_exploration_sample_cloud() {
        let output_path = std::path::Path::new("sample_cloud_rnn_exploration.cbor");
        super::dump_exploration_schedule_sample_cloud(output_path);
    }

    /// Dump sample cloud from baseline (no exploration) training.
    /// Run with: cargo test --release --features native test_dump_baseline_sample_cloud -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_dump_baseline_sample_cloud() {
        let output_path = std::path::Path::new("sample_cloud_rnn_baseline.cbor");
        super::dump_baseline_sample_cloud(output_path);
    }
}
