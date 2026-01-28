//! Training loop for RNN policy.
//!
//! Uses Adam optimizer with gradient clipping for stable training.
//! When the `native` feature is enabled, episode collection is parallelized using rayon.

use crate::adaptive_surface_nets_2::stage4::research::analytical_cube::AnalyticalRotatedCube;
use crate::adaptive_surface_nets_2::stage4::research::oracle::OracleClassification;
use crate::adaptive_surface_nets_2::stage4::research::validation::{
    ExpectedClassification, ValidationPoint,
};

use super::classifier_heads::{
    compute_classifier_gradients, compute_classifier_loss, ClassifierHeadGradients,
    ClassifierLossConfig, ExpectedGeometry, CORNER_HEAD_OUTPUTS, EDGE_HEAD_OUTPUTS,
    FACE_HEAD_OUTPUTS,
};
use super::gradients::{compute_batch_gradient, PolicyGradients};
use super::gru::{Rng, HIDDEN_DIM};
use super::policy::{run_episode, run_episode_with_exploration, Episode, RnnPolicy, BUDGET, CELL_SIZE, INPUT_DIM};
use super::reward::{compute_terminal_reward, RewardBreakdown, RewardConfig, TerminalRewardConfig};

#[cfg(feature = "native")]
use rayon::prelude::*;

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
    /// Whether to use terminal (classifier-based) rewards (legacy RANSAC mode).
    pub use_terminal_reward: bool,
    /// Terminal reward configuration (legacy RANSAC mode).
    pub terminal_reward: TerminalRewardConfig,
    /// Whether to use neural classifier heads for training.
    pub use_classifier_heads: bool,
    /// Classifier head loss configuration.
    pub classifier_loss: ClassifierLossConfig,
    /// Whether to use rotation augmentation during training.
    pub use_rotation_augmentation: bool,
    /// Whether to use exploration schedule (random sampling that cools down).
    /// When enabled, sample selection starts fully random and transitions to policy-based.
    pub use_exploration_schedule: bool,
    /// Starting exploration rate (fraction of random samples at epoch 0).
    /// Only used when use_exploration_schedule is true.
    pub exploration_start: f64,
    /// Ending exploration rate (fraction of random samples at final epoch).
    /// Only used when use_exploration_schedule is true.
    pub exploration_end: f64,
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
            use_terminal_reward: false, // Disabled in favor of classifier heads
            terminal_reward: TerminalRewardConfig::default(),
            use_classifier_heads: true, // New default: use neural classifier heads
            classifier_loss: ClassifierLossConfig::default(),
            use_rotation_augmentation: true, // Enable rotation by default
            use_exploration_schedule: false, // Disabled by default for backwards compatibility
            exploration_start: 1.0, // 100% random at start
            exploration_end: 0.0, // 0% random at end (fully policy-based)
        }
    }
}

impl TrainingConfig {
    /// Create a config with exploration schedule enabled.
    /// This starts training with fully random sampling and gradually transitions
    /// to policy-based sampling, allowing the classifier heads to learn from
    /// diverse samples early in training.
    pub fn with_exploration_schedule() -> Self {
        Self {
            use_exploration_schedule: true,
            exploration_start: 1.0,
            exploration_end: 0.0,
            ..Default::default()
        }
    }

    /// Compute the exploration rate for a given epoch.
    /// Returns 0.0 if exploration schedule is disabled.
    /// Otherwise returns a linearly interpolated value between exploration_start and exploration_end.
    pub fn exploration_rate(&self, epoch: usize) -> f64 {
        if !self.use_exploration_schedule {
            return 0.0;
        }
        if self.epochs <= 1 {
            return self.exploration_end;
        }
        let t = epoch as f64 / (self.epochs - 1) as f64;
        self.exploration_start + t * (self.exploration_end - self.exploration_start)
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
    // Classifier heads
    m_face_w: Vec<f64>,
    m_face_b: Vec<f64>,
    m_edge_w: Vec<f64>,
    m_edge_b: Vec<f64>,
    m_corner_w: Vec<f64>,
    m_corner_b: Vec<f64>,

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
    // Classifier heads
    v_face_w: Vec<f64>,
    v_face_b: Vec<f64>,
    v_edge_w: Vec<f64>,
    v_edge_b: Vec<f64>,
    v_corner_w: Vec<f64>,
    v_corner_b: Vec<f64>,

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
            m_face_w: vec![0.0; FACE_HEAD_OUTPUTS * h],
            m_face_b: vec![0.0; FACE_HEAD_OUTPUTS],
            m_edge_w: vec![0.0; EDGE_HEAD_OUTPUTS * h],
            m_edge_b: vec![0.0; EDGE_HEAD_OUTPUTS],
            m_corner_w: vec![0.0; CORNER_HEAD_OUTPUTS * h],
            m_corner_b: vec![0.0; CORNER_HEAD_OUTPUTS],

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
            v_face_w: vec![0.0; FACE_HEAD_OUTPUTS * h],
            v_face_b: vec![0.0; FACE_HEAD_OUTPUTS],
            v_edge_w: vec![0.0; EDGE_HEAD_OUTPUTS * h],
            v_edge_b: vec![0.0; EDGE_HEAD_OUTPUTS],
            v_corner_w: vec![0.0; CORNER_HEAD_OUTPUTS * h],
            v_corner_b: vec![0.0; CORNER_HEAD_OUTPUTS],

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

        // Classifier heads (gradient descent for loss minimization)
        adam_step_descent(&mut policy.classifier_heads.face.w, &grads.classifier_heads.face_dw, &mut self.m_face_w, &mut self.v_face_w, lr, b1, b2, eps, bc1, bc2);
        adam_step_descent(&mut policy.classifier_heads.face.b, &grads.classifier_heads.face_db, &mut self.m_face_b, &mut self.v_face_b, lr, b1, b2, eps, bc1, bc2);
        adam_step_descent(&mut policy.classifier_heads.edge.w, &grads.classifier_heads.edge_dw, &mut self.m_edge_w, &mut self.v_edge_w, lr, b1, b2, eps, bc1, bc2);
        adam_step_descent(&mut policy.classifier_heads.edge.b, &grads.classifier_heads.edge_db, &mut self.m_edge_b, &mut self.v_edge_b, lr, b1, b2, eps, bc1, bc2);
        adam_step_descent(&mut policy.classifier_heads.corner.w, &grads.classifier_heads.corner_dw, &mut self.m_corner_w, &mut self.v_corner_w, lr, b1, b2, eps, bc1, bc2);
        adam_step_descent(&mut policy.classifier_heads.corner.b, &grads.classifier_heads.corner_db, &mut self.m_corner_b, &mut self.v_corner_b, lr, b1, b2, eps, bc1, bc2);
    }
}

/// Apply one Adam step to a parameter vector (gradient ascent for RL).
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

/// Apply one Adam step to a parameter vector (gradient descent for supervised learning).
fn adam_step_descent(
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
        m[i] = b1 * m[i] + (1.0 - b1) * grads[i];
        v[i] = b2 * v[i] + (1.0 - b2) * grads[i] * grads[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        // Gradient descent: - instead of +
        weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
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
    /// Exploration rate used for this epoch (0.0 if exploration schedule disabled).
    pub exploration_rate: f64,
}

/// Breakdown of classifier loss components for diagnostics.
#[derive(Clone, Debug, Default)]
pub struct ClassifierLossBreakdown {
    pub total: f64,
    pub conf_loss: f64,
    pub normal_loss: f64,
    pub offset_loss: f64,
}

impl ClassifierLossBreakdown {
    /// Accumulate another breakdown into this one.
    pub fn accumulate(&mut self, other: &ClassifierLossBreakdown) {
        self.total += other.total;
        self.conf_loss += other.conf_loss;
        self.normal_loss += other.normal_loss;
        self.offset_loss += other.offset_loss;
    }

    /// Scale all components by a factor.
    pub fn scale(&mut self, factor: f64) {
        self.total *= factor;
        self.conf_loss *= factor;
        self.normal_loss *= factor;
        self.offset_loss *= factor;
    }
}

/// Result of processing a single episode (for parallel collection).
struct EpisodeResult {
    episode: Episode,
    classifier_grads: Option<ClassifierHeadGradients>,
    /// Gradient of classifier loss w.r.t. final hidden state (for BPTT).
    classifier_dh: Vec<f64>,
    classifier_loss: f64,
    classifier_breakdown: ClassifierLossBreakdown,
}

/// Collect episodes in parallel using rayon (native feature).
#[cfg(feature = "native")]
fn collect_episodes_parallel(
    policy: &RnnPolicy,
    training_cube: &AnalyticalRotatedCube,
    training_points: &[ValidationPoint],
    config: &TrainingConfig,
    reward_config: &RewardConfig,
    epoch: usize,
    exploration_rate: f64,
) -> Vec<EpisodeResult> {
    let num_points = training_points.len();

    training_points
        .par_iter()
        .enumerate()
        .map(|(idx, point)| {
            // Create thread-local RNG with deterministic seed
            let seed = (epoch as u64 * num_points as u64 + idx as u64) ^ 0xDEADBEEF;
            let mut rng = Rng::new(seed);

            let mut episode = run_episode_with_exploration(
                policy, training_cube, point, idx, &mut rng, true, exploration_rate, reward_config
            );

            // Compute terminal reward using legacy RANSAC if enabled
            if config.use_terminal_reward && !config.use_classifier_heads {
                let terminal_result = compute_terminal_reward(
                    &episode.final_samples,
                    &episode.inside_flags,
                    point.position,
                    &point.expected,
                    CELL_SIZE,
                    &config.terminal_reward,
                );
                episode.terminal_reward = terminal_result.reward;

                if let Some(last_reward) = episode.rewards.last_mut() {
                    *last_reward += terminal_result.reward;
                }
            }

            // Compute classifier head loss and gradients if enabled
            let (classifier_grads, classifier_dh, classifier_loss, classifier_breakdown) = if config.use_classifier_heads {
                let expected_geom = expected_to_geometry(&point.expected, point.position);
                let predictions = policy.classify(&episode.h_final);

                let (loss, conf_loss, normal_loss, offset_loss, _wrong_loss) = compute_classifier_loss(
                    &predictions,
                    &expected_geom,
                    &config.classifier_loss,
                );

                let breakdown = ClassifierLossBreakdown {
                    total: loss,
                    conf_loss,
                    normal_loss,
                    offset_loss,
                };

                let (head_grads, dh) = compute_classifier_gradients(
                    &policy.classifier_heads,
                    &episode.h_final,
                    &predictions,
                    &expected_geom,
                    &config.classifier_loss,
                );

                let classifier_reward = -loss * 0.5;
                if let Some(last_reward) = episode.rewards.last_mut() {
                    *last_reward += classifier_reward;
                }
                episode.terminal_reward = classifier_reward;

                (Some(head_grads), dh, loss, breakdown)
            } else {
                (None, vec![0.0; HIDDEN_DIM], 0.0, ClassifierLossBreakdown::default())
            };

            EpisodeResult {
                episode,
                classifier_grads,
                classifier_dh,
                classifier_loss,
                classifier_breakdown,
            }
        })
        .collect()
}

/// Collect episodes sequentially (fallback when native feature is disabled).
#[cfg(not(feature = "native"))]
fn collect_episodes_parallel(
    policy: &RnnPolicy,
    training_cube: &AnalyticalRotatedCube,
    training_points: &[ValidationPoint],
    config: &TrainingConfig,
    reward_config: &RewardConfig,
    epoch: usize,
    exploration_rate: f64,
) -> Vec<EpisodeResult> {
    let num_points = training_points.len();

    training_points
        .iter()
        .enumerate()
        .map(|(idx, point)| {
            let seed = (epoch as u64 * num_points as u64 + idx as u64) ^ 0xDEADBEEF;
            let mut rng = Rng::new(seed);

            let mut episode = run_episode_with_exploration(
                policy, training_cube, point, idx, &mut rng, true, exploration_rate, reward_config
            );

            if config.use_terminal_reward && !config.use_classifier_heads {
                let terminal_result = compute_terminal_reward(
                    &episode.final_samples,
                    &episode.inside_flags,
                    point.position,
                    &point.expected,
                    CELL_SIZE,
                    &config.terminal_reward,
                );
                episode.terminal_reward = terminal_result.reward;

                if let Some(last_reward) = episode.rewards.last_mut() {
                    *last_reward += terminal_result.reward;
                }
            }

            let (classifier_grads, classifier_dh, classifier_loss, classifier_breakdown) = if config.use_classifier_heads {
                let expected_geom = expected_to_geometry(&point.expected, point.position);
                let predictions = policy.classify(&episode.h_final);

                let (loss, conf_loss, normal_loss, offset_loss, _wrong_loss) = compute_classifier_loss(
                    &predictions,
                    &expected_geom,
                    &config.classifier_loss,
                );

                let breakdown = ClassifierLossBreakdown {
                    total: loss,
                    conf_loss,
                    normal_loss,
                    offset_loss,
                };

                let (head_grads, dh) = compute_classifier_gradients(
                    &policy.classifier_heads,
                    &episode.h_final,
                    &predictions,
                    &expected_geom,
                    &config.classifier_loss,
                );

                let classifier_reward = -loss * 0.5;
                if let Some(last_reward) = episode.rewards.last_mut() {
                    *last_reward += classifier_reward;
                }
                episode.terminal_reward = classifier_reward;

                (Some(head_grads), dh, loss, breakdown)
            } else {
                (None, vec![0.0; HIDDEN_DIM], 0.0, ClassifierLossBreakdown::default())
            };

            EpisodeResult {
                episode,
                classifier_grads,
                classifier_dh,
                classifier_loss,
                classifier_breakdown,
            }
        })
        .collect()
}

/// Train the RNN policy.
///
/// When the `native` feature is enabled, episode collection is parallelized
/// across CPU cores using rayon, providing significant speedup for training.
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
        // Apply rotation augmentation if enabled
        let (training_cube, training_points) = if config.use_rotation_augmentation {
            let rotation = rng.next_rotation_matrix();
            let rotated_cube = AnalyticalRotatedCube::from_rotation_matrix(rotation);
            let rotated_points: Vec<ValidationPoint> = points
                .iter()
                .map(|p| p.rotate(&rotation))
                .collect();
            (rotated_cube, rotated_points)
        } else {
            (cube.clone(), points.to_vec())
        };

        // Compute exploration rate for this epoch (cooling schedule)
        let exploration_rate = config.exploration_rate(epoch);

        // Collect episodes for all points (parallelized when native feature enabled)
        let results = collect_episodes_parallel(
            policy,
            &training_cube,
            &training_points,
            config,
            reward_config,
            epoch,
            exploration_rate,
        );

        // Aggregate results from parallel collection
        let mut episodes = Vec::with_capacity(results.len());
        let mut classifier_grads_sum = ClassifierHeadGradients::zeros();
        let mut classifier_dh_list = Vec::with_capacity(results.len());
        let mut total_classifier_loss = 0.0;
        let mut classifier_breakdown_sum = ClassifierLossBreakdown::default();

        for result in results {
            if let Some(grads) = result.classifier_grads {
                classifier_grads_sum.add(&grads);
            }
            classifier_dh_list.push(result.classifier_dh);
            total_classifier_loss += result.classifier_loss;
            classifier_breakdown_sum.accumulate(&result.classifier_breakdown);
            episodes.push(result.episode);
        }

        // Compute batch gradient for policy (GRU + Chooser)
        // Pass classifier dh gradients to backpropagate through GRU
        // IMPORTANT: Negate classifier_dh because:
        // - classifier_dh is d_loss/d_h (gradient for minimizing loss)
        // - policy gradient uses gradient ascent (maximizing reward)
        // - reward = -loss, so d_reward/d_h = -d_loss/d_h
        let classifier_dh_negated: Vec<Vec<f64>> = if config.use_classifier_heads {
            classifier_dh_list
                .iter()
                .map(|dh| dh.iter().map(|&x| -x).collect())
                .collect()
        } else {
            Vec::new()
        };
        let classifier_dh_ref = if config.use_classifier_heads {
            Some(classifier_dh_negated.as_slice())
        } else {
            None
        };
        let mut grads = compute_batch_gradient(policy, &episodes, config.discount, classifier_dh_ref);

        // Add classifier head gradients (averaged)
        if config.use_classifier_heads {
            classifier_grads_sum.scale(1.0 / episodes.len() as f64);
            grads.classifier_heads = classifier_grads_sum;
        }

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
        let avg_classifier_loss = if config.use_classifier_heads {
            total_classifier_loss / episodes.len() as f64
        } else {
            0.0
        };

        // Average classifier breakdown
        let mut avg_clf_breakdown = classifier_breakdown_sum;
        avg_clf_breakdown.scale(1.0 / episodes.len() as f64);

        // Aggregate reward breakdown
        let mut avg_breakdown = RewardBreakdown::default();
        for ep in &episodes {
            avg_breakdown.accumulate(&ep.reward_breakdown);
        }
        avg_breakdown.scale(1.0 / episodes.len() as f64);

        let epoch_stats = EpochStats {
            epoch,
            avg_return,
            avg_reward,
            grad_norm,
            avg_samples,
            exploration_rate,
        };
        stats.push(epoch_stats.clone());

        if (epoch + 1) % config.print_every == 0 || epoch == 0 {
            let explore_str = if config.use_exploration_schedule {
                format!(", explore={:.0}%", exploration_rate * 100.0)
            } else {
                String::new()
            };
            if config.use_classifier_heads {
                println!(
                    "  Epoch {:3}: return={:.3}, clf_loss={:.3}, grad_norm={:.4}, samples={:.1}{}",
                    epoch + 1,
                    avg_return,
                    avg_classifier_loss,
                    grad_norm,
                    avg_samples,
                    explore_str
                );
                println!(
                    "             clf breakdown: conf={:.3}, normal={:.3}, offset={:.3}",
                    avg_clf_breakdown.conf_loss,
                    avg_clf_breakdown.normal_loss,
                    avg_clf_breakdown.offset_loss
                );
                println!(
                    "             reward breakdown: surface={:.3}, spread={:.3}, crossing={:.3}, lambda={:.3}",
                    avg_breakdown.surface,
                    avg_breakdown.spread,
                    avg_breakdown.crossing,
                    avg_breakdown.lambda
                );
            } else {
                println!(
                    "  Epoch {:3}: return={:.3}, reward={:.4}, grad_norm={:.4}, samples={:.1}{}",
                    epoch + 1,
                    avg_return,
                    avg_reward,
                    grad_norm,
                    avg_samples,
                    explore_str
                );
            }
        }
    }

    stats
}

/// Convert ExpectedClassification to ExpectedGeometry for classifier loss.
fn expected_to_geometry(
    expected: &ExpectedClassification,
    vertex_position: (f64, f64, f64),
) -> ExpectedGeometry {
    ExpectedGeometry::from_classification(expected, vertex_position)
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
///
/// When the `native` feature is enabled, evaluation is parallelized.
pub fn evaluate_policy(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    points: &[ValidationPoint],
    reward_config: &RewardConfig,
    _rng: &mut Rng,
) -> EvaluationResult {
    let evaluations = evaluate_points_parallel(policy, cube, points, reward_config);

    let n = points.len() as f64;
    let fit_successes = evaluations.iter().filter(|e| e.fit_success).count();
    let total_normal_reward: f64 = evaluations.iter().map(|e| e.normal_reward).sum();
    let total_residual_reward: f64 = evaluations.iter().map(|e| e.residual_reward).sum();
    let total_reward: f64 = evaluations.iter().map(|e| e.total_reward).sum();
    let total_samples: u64 = evaluations.iter().map(|e| e.samples_used).sum();

    EvaluationResult {
        fit_rate: fit_successes as f64 / n,
        avg_normal_reward: total_normal_reward / n,
        avg_residual_reward: total_residual_reward / n,
        avg_reward: total_reward / n,
        avg_samples: total_samples as f64 / n,
        points: evaluations,
    }
}

/// Evaluate points in parallel (native feature).
#[cfg(feature = "native")]
fn evaluate_points_parallel(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    points: &[ValidationPoint],
    reward_config: &RewardConfig,
) -> Vec<PointEvaluation> {
    let terminal_config = TerminalRewardConfig::default();

    points
        .par_iter()
        .enumerate()
        .map(|(idx, point)| {
            let seed = (idx as u64) ^ 0xCAFEBABE;
            let mut rng = Rng::new(seed);

            let episode = run_episode(policy, cube, point, idx, &mut rng, false, reward_config);
            let expected_class = expected_class_from_point(point);

            let terminal_result = compute_terminal_reward(
                &episode.final_samples,
                &episode.inside_flags,
                point.position,
                &point.expected,
                CELL_SIZE,
                &terminal_config,
            );

            let point_reward: f64 = episode.rewards.iter().sum();
            let crossings_found = episode.steps.iter().filter(|s| s.is_crossing).count();

            PointEvaluation {
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
            }
        })
        .collect()
}

/// Evaluate points sequentially (fallback when native feature is disabled).
#[cfg(not(feature = "native"))]
fn evaluate_points_parallel(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    points: &[ValidationPoint],
    reward_config: &RewardConfig,
) -> Vec<PointEvaluation> {
    let terminal_config = TerminalRewardConfig::default();

    points
        .iter()
        .enumerate()
        .map(|(idx, point)| {
            let seed = (idx as u64) ^ 0xCAFEBABE;
            let mut rng = Rng::new(seed);

            let episode = run_episode(policy, cube, point, idx, &mut rng, false, reward_config);
            let expected_class = expected_class_from_point(point);

            let terminal_result = compute_terminal_reward(
                &episode.final_samples,
                &episode.inside_flags,
                point.position,
                &point.expected,
                CELL_SIZE,
                &terminal_config,
            );

            let point_reward: f64 = episode.rewards.iter().sum();
            let crossings_found = episode.steps.iter().filter(|s| s.is_crossing).count();

            PointEvaluation {
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
            }
        })
        .collect()
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
