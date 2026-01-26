//! Gradient computation for RNN policy via BPTT.
//!
//! Uses REINFORCE with baseline for policy gradient estimation,
//! combined with backpropagation through time for the GRU.

use super::chooser::{chooser_backward_policy_gradient, ChooserGradients};
use super::classifier_heads::ClassifierHeadGradients;
use super::gru::{gru_backward, GruGradients, HIDDEN_DIM};
use super::policy::{Episode, RnnPolicy, INPUT_DIM};
use super::reward::{compute_advantages, compute_baseline, compute_returns};

/// Combined gradients for the entire policy.
#[derive(Clone)]
pub struct PolicyGradients {
    pub gru: GruGradients,
    pub chooser: ChooserGradients,
    pub classifier_heads: ClassifierHeadGradients,
}

impl PolicyGradients {
    /// Create zero-initialized gradients.
    pub fn zeros() -> Self {
        Self {
            gru: GruGradients::zeros(INPUT_DIM),
            chooser: ChooserGradients::zeros(),
            classifier_heads: ClassifierHeadGradients::zeros(),
        }
    }

    /// Add another gradient.
    pub fn add(&mut self, other: &PolicyGradients) {
        self.gru.add(&other.gru);
        self.chooser.add(&other.chooser);
        self.classifier_heads.add(&other.classifier_heads);
    }

    /// Scale gradients.
    pub fn scale(&mut self, s: f64) {
        self.gru.scale(s);
        self.chooser.scale(s);
        self.classifier_heads.scale(s);
    }

    /// Clip gradient norms.
    pub fn clip(&mut self, max_norm: f64) {
        self.gru.clip(max_norm);
        self.chooser.clip(max_norm);
        self.classifier_heads.clip(max_norm);
    }

    /// Compute total gradient norm.
    pub fn norm(&self) -> f64 {
        let mut sum = 0.0;
        for &v in &self.gru.dw_z {
            sum += v * v;
        }
        for &v in &self.gru.du_z {
            sum += v * v;
        }
        for &v in &self.gru.db_z {
            sum += v * v;
        }
        for &v in &self.gru.dw_r {
            sum += v * v;
        }
        for &v in &self.gru.du_r {
            sum += v * v;
        }
        for &v in &self.gru.db_r {
            sum += v * v;
        }
        for &v in &self.gru.dw_h {
            sum += v * v;
        }
        for &v in &self.gru.du_h {
            sum += v * v;
        }
        for &v in &self.gru.db_h {
            sum += v * v;
        }
        for &v in &self.chooser.dw {
            sum += v * v;
        }
        for &v in &self.chooser.db {
            sum += v * v;
        }
        // Include classifier head gradients
        for &v in &self.classifier_heads.face_dw {
            sum += v * v;
        }
        for &v in &self.classifier_heads.face_db {
            sum += v * v;
        }
        for &v in &self.classifier_heads.edge_dw {
            sum += v * v;
        }
        for &v in &self.classifier_heads.edge_db {
            sum += v * v;
        }
        for &v in &self.classifier_heads.corner_dw {
            sum += v * v;
        }
        for &v in &self.classifier_heads.corner_db {
            sum += v * v;
        }
        sum.sqrt()
    }
}

/// Compute policy gradient for a single episode using REINFORCE with baseline.
///
/// Uses full BPTT since episodes are short (~50 steps).
pub fn compute_episode_gradient(
    policy: &RnnPolicy,
    episode: &Episode,
    discount: f64,
) -> PolicyGradients {
    if episode.steps.is_empty() {
        return PolicyGradients::zeros();
    }

    // Compute returns and advantages
    let returns = compute_returns(&episode.rewards, discount);
    let baseline = compute_baseline(&returns);
    let advantages = compute_advantages(&returns, baseline);

    // Initialize accumulated gradients
    let mut grads = PolicyGradients::zeros();

    // Backward pass through time
    // We accumulate dL/dh as we go backward
    let mut dh_next = vec![0.0; HIDDEN_DIM];

    for (t, step) in episode.steps.iter().enumerate().rev() {
        let advantage = advantages[t];

        // Chooser backward: get gradients and dL/dh from chooser
        let (chooser_grad, dh_from_chooser) =
            chooser_backward_policy_gradient(&policy.chooser, &step.chooser_cache, advantage);

        // Combine dL/dh from chooser and from next timestep
        let dh: Vec<f64> = dh_from_chooser
            .iter()
            .zip(dh_next.iter())
            .map(|(&c, &n)| c + n)
            .collect();

        // GRU backward: get gradients and dL/dh_prev
        let (gru_grad, dh_prev) = gru_backward(&policy.gru, &step.gru_cache, &dh);

        // Accumulate gradients
        grads.gru.add(&gru_grad);
        grads.chooser.add(&chooser_grad);

        // Pass gradient to previous timestep
        dh_next = dh_prev;
    }

    grads
}

/// Compute average policy gradient over multiple episodes.
pub fn compute_batch_gradient(
    policy: &RnnPolicy,
    episodes: &[Episode],
    discount: f64,
) -> PolicyGradients {
    if episodes.is_empty() {
        return PolicyGradients::zeros();
    }

    let mut total_grads = PolicyGradients::zeros();

    for episode in episodes {
        let grads = compute_episode_gradient(policy, episode, discount);
        total_grads.add(&grads);
    }

    // Average over episodes
    total_grads.scale(1.0 / episodes.len() as f64);

    total_grads
}

/// Apply gradients to policy (gradient ascent for maximizing return).
pub fn apply_gradients(policy: &mut RnnPolicy, grads: &PolicyGradients, lr: f64) {
    // GRU weights
    apply_grad_vec(&mut policy.gru.w_z, &grads.gru.dw_z, lr);
    apply_grad_vec(&mut policy.gru.u_z, &grads.gru.du_z, lr);
    apply_grad_vec(&mut policy.gru.b_z, &grads.gru.db_z, lr);
    apply_grad_vec(&mut policy.gru.w_r, &grads.gru.dw_r, lr);
    apply_grad_vec(&mut policy.gru.u_r, &grads.gru.du_r, lr);
    apply_grad_vec(&mut policy.gru.b_r, &grads.gru.db_r, lr);
    apply_grad_vec(&mut policy.gru.w_h, &grads.gru.dw_h, lr);
    apply_grad_vec(&mut policy.gru.u_h, &grads.gru.du_h, lr);
    apply_grad_vec(&mut policy.gru.b_h, &grads.gru.db_h, lr);

    // Chooser weights
    apply_grad_vec(&mut policy.chooser.w, &grads.chooser.dw, lr);
    apply_grad_vec(&mut policy.chooser.b, &grads.chooser.db, lr);
}

fn apply_grad_vec(weights: &mut [f64], grads: &[f64], lr: f64) {
    for (w, g) in weights.iter_mut().zip(grads.iter()) {
        *w += lr * g; // Gradient ascent: + instead of -
    }
}

/// Numerical gradient check for debugging.
#[cfg(test)]
#[allow(dead_code)]
pub fn numerical_gradient_check(
    policy: &mut RnnPolicy,
    episode: &Episode,
    discount: f64,
    epsilon: f64,
) -> f64 {
    use super::reward::compute_returns;

    // Helper to compute total return
    fn total_return(episode: &Episode, discount: f64) -> f64 {
        let returns = compute_returns(&episode.rewards, discount);
        if returns.is_empty() {
            0.0
        } else {
            returns[0]
        }
    }

    // Compute analytical gradient
    let analytical = compute_episode_gradient(policy, episode, discount);

    // Sample a few parameters to check
    let mut max_error: f64 = 0.0;
    let num_checks = 10;

    // Check GRU w_z
    for i in 0..num_checks.min(policy.gru.w_z.len()) {
        let orig = policy.gru.w_z[i];

        policy.gru.w_z[i] = orig + epsilon;
        let loss_plus = total_return(episode, discount);

        policy.gru.w_z[i] = orig - epsilon;
        let loss_minus = total_return(episode, discount);

        policy.gru.w_z[i] = orig;

        let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        let error = (numerical - analytical.gru.dw_z[i]).abs();
        max_error = max_error.max(error);
    }

    max_error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_surface_nets_2::stage4::research::analytical_cube::AnalyticalRotatedCube;
    use crate::adaptive_surface_nets_2::stage4::research::validation::{
        ExpectedClassification, ValidationPoint,
    };
    use super::super::gru::Rng;
    use super::super::policy::run_episode;
    use super::super::reward::RewardConfig;

    #[test]
    fn test_gradient_computation() {
        let mut rng = Rng::new(42);
        let policy = RnnPolicy::new(&mut rng);
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let point = ValidationPoint {
            position: cube.face_center(0),
            expected: ExpectedClassification::OnFace {
                face_index: 0,
                expected_normal: cube.face_normals[0],
            },
            description: "test".to_string(),
        };
        let reward_config = RewardConfig::default();

        let episode = run_episode(&policy, &cube, &point, 0, &mut rng, true, &reward_config);
        let grads = compute_episode_gradient(&policy, &episode, 0.99);

        // Gradients should be non-zero
        let norm = grads.norm();
        assert!(norm > 0.0, "Gradient norm should be positive");
    }

    #[test]
    fn test_batch_gradient() {
        let mut rng = Rng::new(42);
        let policy = RnnPolicy::new(&mut rng);
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let reward_config = RewardConfig::default();

        let mut episodes = Vec::new();
        for i in 0..3 {
            let face_idx = i % 6;
            let point = ValidationPoint {
                position: cube.face_center(face_idx),
                expected: ExpectedClassification::OnFace {
                    face_index: face_idx,
                    expected_normal: cube.face_normals[face_idx],
                },
                description: format!("face {}", face_idx),
            };
            let episode = run_episode(&policy, &cube, &point, i, &mut rng, true, &reward_config);
            episodes.push(episode);
        }

        let grads = compute_batch_gradient(&policy, &episodes, 0.99);
        assert!(grads.norm() > 0.0);
    }
}
