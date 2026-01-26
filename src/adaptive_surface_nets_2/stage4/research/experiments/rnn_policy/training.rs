//! Training loop for RNN policy.
//!
//! Uses Adam optimizer with gradient clipping for stable training.

use crate::adaptive_surface_nets_2::stage4::research::analytical_cube::AnalyticalRotatedCube;
use crate::adaptive_surface_nets_2::stage4::research::oracle::OracleClassification;
use crate::adaptive_surface_nets_2::stage4::research::validation::{
    ExpectedClassification, ValidationPoint,
};

use super::gradients::{compute_batch_gradient, PolicyGradients};
use super::gru::{Rng, HIDDEN_DIM};
use super::policy::{run_episode, RnnPolicy, BUDGET, CELL_SIZE, INPUT_DIM};
use super::reward::{compute_terminal_reward, RewardConfig, TerminalRewardConfig};

/// Training configuration.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Discount factor for returns.
    pub discount: f64,
    /// Gradient clipping threshold.
    pub grad_clip: f64,
    /// Adam beta1.
    pub adam_beta1: f64,
    /// Adam beta2.
    pub adam_beta2: f64,
    /// Adam epsilon.
    pub adam_eps: f64,
    /// Print progress every N epochs.
    pub print_every: usize,
    /// Whether to use terminal (classifier-based) rewards.
    pub use_terminal_reward: bool,
    /// Terminal reward configuration.
    pub terminal_reward: TerminalRewardConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 200,
            lr: 0.001,
            discount: 0.98,
            grad_clip: 1.0,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            print_every: 20,
            use_terminal_reward: true, // Re-enabled with boosted accuracy weights
            terminal_reward: TerminalRewardConfig::default(),
        }
    }
}

/// Adam optimizer state.
pub struct AdamState {
    /// First moment (mean of gradients).
    m_gru_w_z: Vec<f64>,
    m_gru_u_z: Vec<f64>,
    m_gru_b_z: Vec<f64>,
    m_gru_w_r: Vec<f64>,
    m_gru_u_r: Vec<f64>,
    m_gru_b_r: Vec<f64>,
    m_gru_w_h: Vec<f64>,
    m_gru_u_h: Vec<f64>,
    m_gru_b_h: Vec<f64>,
    m_chooser_w: Vec<f64>,
    m_chooser_b: Vec<f64>,

    /// Second moment (variance of gradients).
    v_gru_w_z: Vec<f64>,
    v_gru_u_z: Vec<f64>,
    v_gru_b_z: Vec<f64>,
    v_gru_w_r: Vec<f64>,
    v_gru_u_r: Vec<f64>,
    v_gru_b_r: Vec<f64>,
    v_gru_w_h: Vec<f64>,
    v_gru_u_h: Vec<f64>,
    v_gru_b_h: Vec<f64>,
    v_chooser_w: Vec<f64>,
    v_chooser_b: Vec<f64>,

    /// Timestep.
    t: usize,
}

impl AdamState {
    /// Create new Adam state for the given policy.
    pub fn new(_policy: &RnnPolicy) -> Self {
        let h = HIDDEN_DIM;
        let i = INPUT_DIM;
        let o = 8; // NUM_OCTANTS

        Self {
            m_gru_w_z: vec![0.0; h * i],
            m_gru_u_z: vec![0.0; h * h],
            m_gru_b_z: vec![0.0; h],
            m_gru_w_r: vec![0.0; h * i],
            m_gru_u_r: vec![0.0; h * h],
            m_gru_b_r: vec![0.0; h],
            m_gru_w_h: vec![0.0; h * i],
            m_gru_u_h: vec![0.0; h * h],
            m_gru_b_h: vec![0.0; h],
            m_chooser_w: vec![0.0; o * h],
            m_chooser_b: vec![0.0; o],

            v_gru_w_z: vec![0.0; h * i],
            v_gru_u_z: vec![0.0; h * h],
            v_gru_b_z: vec![0.0; h],
            v_gru_w_r: vec![0.0; h * i],
            v_gru_u_r: vec![0.0; h * h],
            v_gru_b_r: vec![0.0; h],
            v_gru_w_h: vec![0.0; h * i],
            v_gru_u_h: vec![0.0; h * h],
            v_gru_b_h: vec![0.0; h],
            v_chooser_w: vec![0.0; o * h],
            v_chooser_b: vec![0.0; o],

            t: 0,
        }
    }

    /// Update policy using Adam optimizer.
    pub fn update(&mut self, policy: &mut RnnPolicy, grads: &PolicyGradients, config: &TrainingConfig) {
        self.t += 1;
        let lr = config.lr;
        let b1 = config.adam_beta1;
        let b2 = config.adam_beta2;
        let eps = config.adam_eps;

        // Bias correction factors
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);

        // Update each parameter group
        adam_step(&mut policy.gru.w_z, &grads.gru.dw_z, &mut self.m_gru_w_z, &mut self.v_gru_w_z, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.gru.u_z, &grads.gru.du_z, &mut self.m_gru_u_z, &mut self.v_gru_u_z, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.gru.b_z, &grads.gru.db_z, &mut self.m_gru_b_z, &mut self.v_gru_b_z, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.gru.w_r, &grads.gru.dw_r, &mut self.m_gru_w_r, &mut self.v_gru_w_r, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.gru.u_r, &grads.gru.du_r, &mut self.m_gru_u_r, &mut self.v_gru_u_r, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.gru.b_r, &grads.gru.db_r, &mut self.m_gru_b_r, &mut self.v_gru_b_r, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.gru.w_h, &grads.gru.dw_h, &mut self.m_gru_w_h, &mut self.v_gru_w_h, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.gru.u_h, &grads.gru.du_h, &mut self.m_gru_u_h, &mut self.v_gru_u_h, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.gru.b_h, &grads.gru.db_h, &mut self.m_gru_b_h, &mut self.v_gru_b_h, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.chooser.w, &grads.chooser.dw, &mut self.m_chooser_w, &mut self.v_chooser_w, lr, b1, b2, eps, bc1, bc2);
        adam_step(&mut policy.chooser.b, &grads.chooser.db, &mut self.m_chooser_b, &mut self.v_chooser_b, lr, b1, b2, eps, bc1, bc2);
    }
}

/// Apply one Adam step to a parameter vector.
fn adam_step(
    weights: &mut [f64],
    grads: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    bc1: f64,
    bc2: f64,
) {
    for i in 0..weights.len() {
        // Update biased first moment estimate
        m[i] = b1 * m[i] + (1.0 - b1) * grads[i];
        // Update biased second raw moment estimate
        v[i] = b2 * v[i] + (1.0 - b2) * grads[i] * grads[i];
        // Compute bias-corrected estimates
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        // Update parameters (gradient ascent: + instead of -)
        weights[i] += lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// Training statistics for one epoch.
#[derive(Clone, Debug)]
pub struct EpochStats {
    pub epoch: usize,
    pub avg_return: f64,
    pub avg_reward: f64,
    pub grad_norm: f64,
    pub avg_samples: f64,
}

/// Train the RNN policy.
pub fn train_policy(
    policy: &mut RnnPolicy,
    cube: &AnalyticalRotatedCube,
    points: &[ValidationPoint],
    config: &TrainingConfig,
    reward_config: &RewardConfig,
    rng: &mut Rng,
) -> Vec<EpochStats> {
    let mut adam = AdamState::new(policy);
    let mut stats = Vec::new();

    for epoch in 0..config.epochs {
        // Collect episodes for all points
        let mut episodes = Vec::new();
        for (idx, point) in points.iter().enumerate() {
            let mut episode = run_episode(policy, cube, point, idx, rng, true, reward_config);

            // Compute terminal reward if enabled
            if config.use_terminal_reward {
                let terminal_result = compute_terminal_reward(
                    &episode.final_samples,
                    &episode.inside_flags,
                    point.position,
                    &point.expected,
                    CELL_SIZE,
                    &config.terminal_reward,
                );
                episode.terminal_reward = terminal_result.reward;

                // Add terminal reward to the last step's reward
                if let Some(last_reward) = episode.rewards.last_mut() {
                    *last_reward += terminal_result.reward;
                }
            }

            episodes.push(episode);
        }

        // Compute batch gradient
        let mut grads = compute_batch_gradient(policy, &episodes, config.discount);

        // Clip gradients
        grads.clip(config.grad_clip);
        let grad_norm = grads.norm();

        // Update policy
        adam.update(policy, &grads, config);

        // Compute statistics
        let avg_return = episodes
            .iter()
            .map(|e| e.rewards.iter().sum::<f64>())
            .sum::<f64>()
            / episodes.len() as f64;
        let avg_reward = episodes
            .iter()
            .flat_map(|e| e.rewards.iter())
            .sum::<f64>()
            / (episodes.len() * BUDGET) as f64;
        let avg_samples = episodes.iter().map(|e| e.samples_used as f64).sum::<f64>()
            / episodes.len() as f64;

        let epoch_stats = EpochStats {
            epoch,
            avg_return,
            avg_reward,
            grad_norm,
            avg_samples,
        };
        stats.push(epoch_stats.clone());

        if (epoch + 1) % config.print_every == 0 || epoch == 0 {
            println!(
                "  Epoch {:3}: return={:.3}, reward={:.4}, grad_norm={:.4}, samples={:.1}",
                epoch + 1,
                avg_return,
                avg_reward,
                grad_norm,
                avg_samples
            );
        }
    }

    stats
}

/// Evaluation result for a single point.
#[derive(Clone, Debug)]
pub struct PointEvaluation {
    pub point_idx: usize,
    pub description: String,
    pub expected_class: OracleClassification,
    /// Whether the oracle-appropriate classifier fit succeeded.
    pub fit_success: bool,
    /// Normal accuracy reward (higher = fitted normals closer to oracle).
    pub normal_reward: f64,
    /// Residual reward (higher = lower fitting residual).
    pub residual_reward: f64,
    /// Total reward for this point.
    pub total_reward: f64,
    pub samples_used: u64,
    pub crossings_found: usize,
    pub surface_points_count: usize,
}

/// Evaluation results.
#[derive(Clone, Debug)]
pub struct EvaluationResult {
    pub points: Vec<PointEvaluation>,
    /// Fraction of points where classifier fit succeeded.
    pub fit_rate: f64,
    /// Average normal accuracy reward (0-1 scale, higher = better).
    pub avg_normal_reward: f64,
    /// Average residual reward (0-1 scale, higher = better).
    pub avg_residual_reward: f64,
    pub avg_reward: f64,
    pub avg_samples: f64,
}

impl EvaluationResult {
    pub fn summary(&self, name: &str) -> String {
        format!(
            "  {}: fit_rate={:.1}%, normal_acc={:.3}, residual={:.3}, avg_reward={:.3}",
            name,
            self.fit_rate * 100.0,
            self.avg_normal_reward,
            self.avg_residual_reward,
            self.avg_reward,
        )
    }
}

/// Evaluate the policy without training.
///
/// Uses the oracle-selected RANSAC classifier for each point and measures
/// how well the fitted geometry matches the oracle ground truth.
pub fn evaluate_policy(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    points: &[ValidationPoint],
    reward_config: &RewardConfig,
    rng: &mut Rng,
) -> EvaluationResult {
    let terminal_config = TerminalRewardConfig::default();
    let mut evaluations = Vec::new();
    let mut total_reward = 0.0;
    let mut total_samples = 0u64;
    let mut fit_successes = 0;
    let mut total_normal_reward = 0.0;
    let mut total_residual_reward = 0.0;

    for (idx, point) in points.iter().enumerate() {
        let episode = run_episode(policy, cube, point, idx, rng, false, reward_config);

        let expected_class = expected_class_from_point(point);

        // Use oracle-selected RANSAC classifier to evaluate fit quality
        let terminal_result = compute_terminal_reward(
            &episode.final_samples,
            &episode.inside_flags,
            point.position,
            &point.expected,
            CELL_SIZE,
            &terminal_config,
        );

        if terminal_result.fit_success {
            fit_successes += 1;
        }
        total_normal_reward += terminal_result.normal_reward;
        total_residual_reward += terminal_result.residual_reward;

        let point_reward: f64 = episode.rewards.iter().sum();
        total_reward += point_reward;
        total_samples += episode.samples_used;

        let crossings_found = episode
            .steps
            .iter()
            .filter(|s| s.is_crossing)
            .count();

        evaluations.push(PointEvaluation {
            point_idx: idx,
            description: point.description.clone(),
            expected_class,
            fit_success: terminal_result.fit_success,
            normal_reward: terminal_result.normal_reward,
            residual_reward: terminal_result.residual_reward,
            total_reward: point_reward,
            samples_used: episode.samples_used,
            crossings_found,
            surface_points_count: terminal_result.surface_points_count,
        });
    }

    let n = points.len() as f64;
    EvaluationResult {
        fit_rate: fit_successes as f64 / n,
        avg_normal_reward: total_normal_reward / n,
        avg_residual_reward: total_residual_reward / n,
        avg_reward: total_reward / n,
        avg_samples: total_samples as f64 / n,
        points: evaluations,
    }
}

fn expected_class_from_point(point: &ValidationPoint) -> OracleClassification {
    match &point.expected {
        ExpectedClassification::OnFace { .. } => OracleClassification::Face,
        ExpectedClassification::OnEdge { .. } => OracleClassification::Edge,
        ExpectedClassification::OnCorner { .. } => OracleClassification::Corner,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_surface_nets_2::stage4::research::validation::generate_validation_points_randomized;

    #[test]
    fn test_training_loop() {
        let mut rng = Rng::new(42);
        let mut policy = RnnPolicy::new(&mut rng);
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let points = generate_validation_points_randomized(&cube, CELL_SIZE * 0.2, 2, 42);

        // Use a subset of points for faster test
        let subset: Vec<_> = points.into_iter().take(5).collect();

        let config = TrainingConfig {
            epochs: 5,
            print_every: 2,
            ..Default::default()
        };
        let reward_config = RewardConfig::default();

        let stats = train_policy(&mut policy, &cube, &subset, &config, &reward_config, &mut rng);

        assert_eq!(stats.len(), 5);
        // Returns should be finite
        for s in &stats {
            assert!(s.avg_return.is_finite());
        }
    }

    #[test]
    fn test_evaluation() {
        let mut rng = Rng::new(42);
        let policy = RnnPolicy::new(&mut rng);
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let points = generate_validation_points_randomized(&cube, CELL_SIZE * 0.2, 2, 42);

        let subset: Vec<_> = points.into_iter().take(5).collect();
        let reward_config = RewardConfig::default();

        let eval = evaluate_policy(&policy, &cube, &subset, &reward_config, &mut rng);

        assert_eq!(eval.points.len(), 5);
        assert!(eval.fit_rate >= 0.0 && eval.fit_rate <= 1.0);
        assert!(eval.avg_normal_reward >= 0.0);
    }
}
