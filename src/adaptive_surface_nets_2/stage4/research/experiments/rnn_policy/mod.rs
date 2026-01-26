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
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);

    let mut rng = Rng::new(12345);
    let reward_config = RewardConfig::default();

    println!("\n{}", "=".repeat(72));
    println!("RNN POLICY MVP (ROTATED CUBE)");
    println!("{}", "=".repeat(72));

    // Create and train policy
    let mut policy = RnnPolicy::new(&mut rng);
    println!(
        "\nPolicy parameters: {} (GRU: {}, Chooser: {})",
        policy.param_count(),
        policy.gru.param_count(),
        policy.chooser.param_count()
    );

    // Evaluate before training
    let eval_before = evaluate_policy(&policy, &cube, &points, &reward_config, &mut rng);
    println!("\nBefore training:");
    println!("{}", eval_before.summary("Random policy"));

    // Train
    println!("\nTraining...");
    let config = TrainingConfig {
        epochs: 200,
        print_every: 40,
        lr: 0.001,
        ..Default::default()
    };
    let _stats = train_policy(&mut policy, &cube, &points, &config, &reward_config, &mut rng);

    // Evaluate after training
    let eval_after = evaluate_policy(&policy, &cube, &points, &reward_config, &mut rng);
    println!("\nAfter training:");
    println!("{}", eval_after.summary("Trained policy"));

    // Breakdown by classification
    println!("\nBreakdown:");
    print_breakdown(&eval_after);
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
    let cube = AnalyticalRotatedCube::standard_test_cube();
    let points = generate_validation_points_randomized(&cube, policy::CELL_SIZE * 0.2, 2, 42);

    let mut rng = Rng::new(12345);
    let reward_config = RewardConfig::default();

    let mut dump = SampleCloudDump::new();

    for (idx, point) in points.iter().enumerate() {
        begin_sample_recording();

        // stochastic=false (use argmax), use_discrete controls corner vs weighted position
        let episode = run_episode_ex(policy, &cube, point, idx, &mut rng, false, use_discrete, &reward_config);

        let samples = end_sample_recording();

        let hint = hint_from_expected(&point.expected);
        let mut set = SampleCloudSet::with_cell_size(
            idx as u64,
            to_f32(point.position),
            to_f32(hint),
            policy::CELL_SIZE as f32,
        );
        set.label = Some(format!("{}: {}", label, point.description));
        set.points = samples;
        set.meta.samples_used = Some(episode.samples_used as u32);
        set.meta.note = Some(label.to_string());
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
        assert!(eval.accuracy >= 0.0 && eval.accuracy <= 1.0);
    }
}
