//! Reward computation for RNN sampling policy.
//!
//! The reward structure encourages:
//! 1. Sampling close to the surface
//! 2. Spatial exploration (spread out samples)
//! 3. Finding in/out transitions
//! 4. Efficiency (penalty per sample)
//! 5. Good classifier fits (terminal reward)
//! 6. Directional diversity for outside samples (helps edge/corner fitting)

use super::classifier::{
    best_corner_normal_errors, best_edge_normal_errors, extract_surface_points,
    fit_corner_from_samples, fit_edge_from_samples, fit_face_from_samples, normal_error_degrees,
    ClassifierConfig,
};
use super::math::{distance_3d, dot, length, normalize, sub};
use crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification;

/// Reward weights and parameters.
#[derive(Clone, Debug)]
pub struct RewardConfig {
    /// Weight for surface proximity reward.
    pub w_surface: f64,
    /// Weight for spread (min distance to previous samples).
    pub w_spread: f64,
    /// Bonus for finding an in/out transition.
    pub crossing_bonus: f64,
    /// Per-sample cost (negative reward).
    pub lambda: f64,
    /// Scale factor for exponential decay (larger = sharper).
    pub surface_decay: f64,
    /// Weight for entropy bonus (encourages diverse octant selection).
    pub w_entropy: f64,
    /// Weight for directional diversity of outside samples.
    /// Encourages sampling in directions orthogonal to previous outside samples,
    /// which helps edge/corner fitting by spreading samples across multiple faces.
    pub w_direction_diversity: f64,
    /// Weight for spatial spread of outside samples.
    /// Encourages outside samples to be far from each other spatially,
    /// which helps edge/corner fitting by spreading samples across multiple faces.
    pub w_outside_spread: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            w_surface: 0.2,      // Encourage sampling near surface
            w_spread: 0.1,       // Encourage exploration
            crossing_bonus: 0.5, // Strong bonus for finding crossings
            lambda: 0.01,        // Small per-sample cost
            surface_decay: 5.0,
            w_entropy: 0.0,      // Disabled by default
            w_direction_diversity: 0.0, // Disabled by default
            w_outside_spread: 0.0, // Disabled by default
        }
    }
}

impl RewardConfig {
    /// Config optimized for classifier head training.
    /// Emphasizes exploration and crossing detection with entropy bonus.
    pub fn for_classifier_training() -> Self {
        Self {
            w_surface: 0.15,     // Moderate surface proximity reward
            w_spread: 0.1,       // Good exploration reward
            crossing_bonus: 1.0, // Strong crossing bonus
            lambda: 0.005,       // Lower per-sample cost
            surface_decay: 4.0,  // Slightly softer decay
            w_entropy: 0.1,      // Entropy bonus for action diversity
            w_direction_diversity: 0.0, // Disabled for now
            w_outside_spread: 0.0, // Disabled for now
        }
    }

    /// Config with directional diversity enabled for edge/corner improvement.
    /// This encourages outside samples to spread across multiple face planes.
    pub fn with_direction_diversity() -> Self {
        Self {
            w_surface: 0.15,
            w_spread: 0.1,
            crossing_bonus: 0.5,
            lambda: 0.005,
            surface_decay: 4.0,
            w_entropy: 0.0,
            w_direction_diversity: 0.05, // Small diversity bonus
            w_outside_spread: 0.15, // Stronger spatial spread for outside samples
        }
    }

    /// Config with high crossing bonus to encourage finding transitions.
    /// This may help edge/corner classification by encouraging exploration
    /// across the surface in multiple directions.
    pub fn with_high_crossing_bonus() -> Self {
        Self {
            w_surface: 0.1,       // Lower surface weight
            w_spread: 0.15,       // Higher spread weight
            crossing_bonus: 2.0,  // Very high crossing bonus
            lambda: 0.01,
            surface_decay: 4.0,
            w_entropy: 0.0,
            w_direction_diversity: 0.0,
            w_outside_spread: 0.0,
        }
    }
}

/// Compute directional diversity bonus for an outside sample.
///
/// Returns a value in [0, 1] indicating how different this sample's direction
/// (from vertex) is compared to previous outside samples. Higher values mean
/// the sample explores a new direction, which helps edge/corner fitting.
///
/// For the first outside sample, returns 0.5 (neutral).
/// For subsequent samples, returns the minimum angular difference (normalized)
/// from any previous outside sample direction.
pub fn compute_direction_diversity(
    vertex: (f64, f64, f64),
    sample_pos: (f64, f64, f64),
    prev_outside_samples: &[(f64, f64, f64)],
) -> f64 {
    if prev_outside_samples.is_empty() {
        return 0.5; // Neutral for first outside sample
    }

    // Compute direction from vertex to current sample
    let dir = sub(sample_pos, vertex);
    let len = length(dir);
    if len < 1e-10 {
        return 0.0; // Sample at vertex, no diversity
    }
    let dir_norm = normalize(dir);

    // Find minimum angular difference to any previous outside sample
    let mut min_cos_sim: f64 = 1.0; // cos(0) = 1, meaning identical direction

    for &prev in prev_outside_samples {
        let prev_dir = sub(prev, vertex);
        let prev_len = length(prev_dir);
        if prev_len < 1e-10 {
            continue;
        }
        let prev_norm = normalize(prev_dir);

        // Cosine similarity: 1 = same direction, 0 = orthogonal, -1 = opposite
        let cos_sim: f64 = dot(dir_norm, prev_norm).abs(); // abs because opposite is also similar
        min_cos_sim = min_cos_sim.min(cos_sim);
    }

    // Convert to diversity score: orthogonal (cos=0) gives max score (1.0)
    // Same direction (cos=1) gives min score (0.0)
    1.0 - min_cos_sim
}

/// Compute spatial spread bonus for an outside sample.
///
/// Returns a value in [0, 1] indicating how far this outside sample is from
/// previous outside samples (normalized by cell_size). This encourages outside
/// samples to spread out spatially, which helps edge/corner fitting.
///
/// For the first outside sample, returns 0.5 (neutral).
pub fn compute_outside_spread(
    sample_pos: (f64, f64, f64),
    prev_outside_samples: &[(f64, f64, f64)],
    cell_size: f64,
) -> f64 {
    if prev_outside_samples.is_empty() {
        return 0.5; // Neutral for first outside sample
    }

    // Find minimum distance to any previous outside sample
    let min_dist = prev_outside_samples
        .iter()
        .map(|&prev| distance_3d(sample_pos, prev))
        .fold(f64::INFINITY, f64::min);

    // Normalize by cell_size, cap at 1.0
    // Reward increases linearly with distance up to cell_size
    (min_dist / cell_size).min(1.0)
}

/// Compute reward for a single sampling step.
///
/// # Arguments
/// * `sample_pos` - Position that was sampled
/// * `oracle_distance` - Distance to closest surface point (from analytical oracle)
/// * `prev_samples` - List of previous sample positions
/// * `is_crossing` - Whether this sample found an in/out transition
/// * `cell_size` - Size of the cell (for normalizing distances)
/// * `config` - Reward configuration
pub fn compute_step_reward(
    sample_pos: (f64, f64, f64),
    oracle_distance: f64,
    prev_samples: &[(f64, f64, f64)],
    is_crossing: bool,
    cell_size: f64,
    config: &RewardConfig,
) -> f64 {
    // Surface proximity: exp(-distance/cell_size * decay)
    let normalized_dist = oracle_distance / cell_size;
    let surface_reward = (-normalized_dist * config.surface_decay).exp();

    // Spread: minimum distance to any previous sample
    let spread_reward = if prev_samples.is_empty() {
        0.5 // Neutral reward for first sample
    } else {
        let min_dist = prev_samples
            .iter()
            .map(|&p| distance_3d(sample_pos, p))
            .fold(f64::INFINITY, f64::min);
        // Normalize by cell_size, cap at 1.0
        (min_dist / cell_size).min(1.0)
    };

    // Crossing bonus
    let crossing = if is_crossing { config.crossing_bonus } else { 0.0 };

    // Total reward
    config.w_surface * surface_reward
        + config.w_spread * spread_reward
        + crossing
        - config.lambda
}

/// Compute reward for a single sampling step with entropy bonus.
///
/// Extended version that includes entropy bonus from action probabilities
/// and directional diversity bonus for outside samples.
///
/// # Arguments
/// * `sample_pos` - Position that was sampled
/// * `oracle_distance` - Distance to closest surface point
/// * `prev_samples` - List of previous sample positions
/// * `prev_inside_flags` - Inside/outside flag for each previous sample
/// * `is_inside` - Whether current sample is inside
/// * `is_crossing` - Whether this sample found an in/out transition
/// * `vertex` - The vertex position (for directional diversity)
/// * `cell_size` - Size of the cell
/// * `action_probs` - Action probabilities (for entropy bonus)
/// * `config` - Reward configuration
pub fn compute_step_reward_with_entropy(
    sample_pos: (f64, f64, f64),
    oracle_distance: f64,
    prev_samples: &[(f64, f64, f64)],
    prev_inside_flags: &[bool],
    is_inside: bool,
    is_crossing: bool,
    vertex: (f64, f64, f64),
    cell_size: f64,
    action_probs: &[f64],
    config: &RewardConfig,
) -> f64 {
    let base_reward = compute_step_reward(
        sample_pos,
        oracle_distance,
        prev_samples,
        is_crossing,
        cell_size,
        config,
    );

    // Entropy bonus: -sum(p * log(p))
    // Normalized by max entropy (log(n) for uniform distribution)
    let entropy = compute_entropy(action_probs);
    let max_entropy = (action_probs.len() as f64).ln();
    let normalized_entropy = if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    };

    // Collect previous outside samples (shared by both diversity bonuses)
    let prev_outside: Vec<_> = if !is_inside
        && (config.w_direction_diversity > 0.0 || config.w_outside_spread > 0.0)
    {
        prev_samples
            .iter()
            .zip(prev_inside_flags.iter())
            .filter(|(_, inside)| !**inside)
            .map(|(pos, _)| *pos)
            .collect()
    } else {
        Vec::new()
    };

    // Directional diversity bonus for outside samples
    // This encourages outside samples to spread across different faces,
    // which helps edge/corner fitting
    let direction_diversity = if !is_inside && config.w_direction_diversity > 0.0 {
        compute_direction_diversity(vertex, sample_pos, &prev_outside)
    } else {
        0.0
    };

    // Spatial spread bonus for outside samples
    // This encourages outside samples to be far from each other,
    // which helps edge/corner fitting
    let outside_spread = if !is_inside && config.w_outside_spread > 0.0 {
        compute_outside_spread(sample_pos, &prev_outside, cell_size)
    } else {
        0.0
    };

    base_reward
        + config.w_entropy * normalized_entropy
        + config.w_direction_diversity * direction_diversity
        + config.w_outside_spread * outside_spread
}

/// Compute entropy of a probability distribution.
fn compute_entropy(probs: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &p in probs {
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Compute discounted returns from rewards.
///
/// G_t = r_t + gamma * G_{t+1}
pub fn compute_returns(rewards: &[f64], discount: f64) -> Vec<f64> {
    let mut returns = vec![0.0; rewards.len()];
    let mut g = 0.0;
    for (i, &r) in rewards.iter().enumerate().rev() {
        g = r + discount * g;
        returns[i] = g;
    }
    returns
}

/// Compute baseline (average return) for variance reduction.
pub fn compute_baseline(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        0.0
    } else {
        returns.iter().sum::<f64>() / returns.len() as f64
    }
}

/// Compute advantages: returns - baseline.
pub fn compute_advantages(returns: &[f64], baseline: f64) -> Vec<f64> {
    returns.iter().map(|&g| g - baseline).collect()
}

/// Configuration for terminal (classifier-based) reward.
#[derive(Clone, Debug)]
pub struct TerminalRewardConfig {
    /// Weight for successful fit (fitted something vs nothing).
    pub w_fit_success: f64,
    /// Weight for residual quality (lower residual = higher reward).
    pub w_residual: f64,
    /// Weight for normal accuracy (lower error = higher reward).
    pub w_normal_accuracy: f64,
    /// Weight for edge direction accuracy.
    pub w_edge_direction: f64,
    /// Maximum expected residual (for normalization).
    pub max_residual: f64,
    /// Maximum expected normal error in degrees (for normalization).
    pub max_normal_error: f64,
    /// Classifier configuration.
    pub classifier: ClassifierConfig,
}

impl Default for TerminalRewardConfig {
    fn default() -> Self {
        Self {
            w_fit_success: 1.0,        // Small bonus for successful fit
            w_residual: 5.0,           // Good residual quality matters
            w_normal_accuracy: 20.0,   // Dominant: accuracy vs oracle normals
            w_edge_direction: 5.0,     // Edge direction accuracy
            max_residual: 0.1,         // Relative to cell_size
            max_normal_error: 45.0,    // Degrees
            classifier: ClassifierConfig::default(),
        }
    }
}

/// Result of terminal reward computation.
#[derive(Clone, Debug)]
pub struct TerminalRewardResult {
    /// Total terminal reward.
    pub reward: f64,
    /// Whether the fit succeeded.
    pub fit_success: bool,
    /// Residual quality component.
    pub residual_reward: f64,
    /// Normal accuracy component.
    pub normal_reward: f64,
    /// Number of surface points extracted.
    pub surface_points_count: usize,
}

/// Compute terminal reward based on classifier fit quality.
///
/// This function:
/// 1. Extracts appropriate points based on geometry type
/// 2. Runs the appropriate classifier based on oracle label
/// 3. Computes reward based on fit quality and normal accuracy
///
/// For faces: uses midpoints (surface approximations)
/// For edges/corners: uses outside samples directly (they lie on actual face planes)
pub fn compute_terminal_reward(
    samples: &[(f64, f64, f64)],
    inside_flags: &[bool],
    vertex: (f64, f64, f64),
    expected: &ExpectedClassification,
    cell_size: f64,
    config: &TerminalRewardConfig,
) -> TerminalRewardResult {
    match expected {
        ExpectedClassification::OnFace { expected_normal, .. } => {
            // For faces: use midpoints (surface approximations work well for single plane)
            let surface_points = extract_surface_points(samples, inside_flags, vertex, cell_size);
            if surface_points.len() < 3 {
                return TerminalRewardResult {
                    reward: 0.0,
                    fit_success: false,
                    residual_reward: 0.0,
                    normal_reward: 0.0,
                    surface_points_count: surface_points.len(),
                };
            }
            compute_face_terminal_reward(&surface_points, *expected_normal, cell_size, config)
        }
        ExpectedClassification::OnEdge {
            expected_direction,
            expected_normals,
            ..
        } => {
            // For edges: use outside samples directly (they lie on the two face planes)
            let outside_samples: Vec<_> = samples
                .iter()
                .zip(inside_flags.iter())
                .filter(|&(_, &is_inside)| !is_inside)
                .map(|(p, _)| *p)
                .collect();
            if outside_samples.len() < 6 {
                // Need enough points for two planes
                return TerminalRewardResult {
                    reward: 0.0,
                    fit_success: false,
                    residual_reward: 0.0,
                    normal_reward: 0.0,
                    surface_points_count: outside_samples.len(),
                };
            }
            compute_edge_terminal_reward(
                &outside_samples,
                *expected_direction,
                *expected_normals,
                cell_size,
                config,
            )
        }
        ExpectedClassification::OnCorner {
            expected_normals, ..
        } => {
            // For corners: use outside samples directly (they lie on the three face planes)
            let outside_samples: Vec<_> = samples
                .iter()
                .zip(inside_flags.iter())
                .filter(|&(_, &is_inside)| !is_inside)
                .map(|(p, _)| *p)
                .collect();
            if outside_samples.len() < 9 {
                // Need enough points for three planes
                return TerminalRewardResult {
                    reward: 0.0,
                    fit_success: false,
                    residual_reward: 0.0,
                    normal_reward: 0.0,
                    surface_points_count: outside_samples.len(),
                };
            }
            compute_corner_terminal_reward(&outside_samples, expected_normals, cell_size, config)
        }
    }
}

/// Compute terminal reward for face classification.
fn compute_face_terminal_reward(
    surface_points: &[(f64, f64, f64)],
    expected_normal: (f64, f64, f64),
    cell_size: f64,
    config: &TerminalRewardConfig,
) -> TerminalRewardResult {
    let fit = fit_face_from_samples(surface_points, cell_size, &config.classifier);

    match fit {
        None => TerminalRewardResult {
            reward: 0.0,
            fit_success: false,
            residual_reward: 0.0,
            normal_reward: 0.0,
            surface_points_count: surface_points.len(),
        },
        Some(result) => {
            // Residual reward: lower is better
            let normalized_residual = (result.residual / cell_size) / config.max_residual;
            let residual_reward = config.w_residual * (1.0 - normalized_residual.min(1.0));

            // Normal accuracy reward
            let normal_error = normal_error_degrees(result.normal, expected_normal);
            let normalized_error = normal_error / config.max_normal_error;
            let normal_reward = config.w_normal_accuracy * (1.0 - normalized_error.min(1.0));

            let total = config.w_fit_success + residual_reward + normal_reward;

            TerminalRewardResult {
                reward: total,
                fit_success: true,
                residual_reward,
                normal_reward,
                surface_points_count: surface_points.len(),
            }
        }
    }
}

/// Compute terminal reward for edge classification.
fn compute_edge_terminal_reward(
    surface_points: &[(f64, f64, f64)],
    expected_direction: (f64, f64, f64),
    expected_normals: ((f64, f64, f64), (f64, f64, f64)),
    cell_size: f64,
    config: &TerminalRewardConfig,
) -> TerminalRewardResult {
    let fit = fit_edge_from_samples(surface_points, cell_size, &config.classifier);

    match fit {
        None => TerminalRewardResult {
            reward: 0.0,
            fit_success: false,
            residual_reward: 0.0,
            normal_reward: 0.0,
            surface_points_count: surface_points.len(),
        },
        Some(result) => {
            // Residual reward: average of both planes
            let avg_residual = (result.residual_a + result.residual_b) / 2.0;
            let normalized_residual = (avg_residual / cell_size) / config.max_residual;
            let residual_reward = config.w_residual * (1.0 - normalized_residual.min(1.0));

            // Normal accuracy reward: best pairing of detected to expected
            let (err_a, err_b) = best_edge_normal_errors(
                (result.normal_a, result.normal_b),
                expected_normals,
            );
            let avg_normal_error = (err_a + err_b) / 2.0;
            let normalized_error = avg_normal_error / config.max_normal_error;
            let normal_reward = config.w_normal_accuracy * (1.0 - normalized_error.min(1.0));

            // Edge direction reward
            let dir_error = normal_error_degrees(result.edge_direction, expected_direction);
            let normalized_dir_error = dir_error / config.max_normal_error;
            let dir_reward = config.w_edge_direction * (1.0 - normalized_dir_error.min(1.0));

            let total = config.w_fit_success + residual_reward + normal_reward + dir_reward;

            TerminalRewardResult {
                reward: total,
                fit_success: true,
                residual_reward,
                normal_reward: normal_reward + dir_reward,
                surface_points_count: surface_points.len(),
            }
        }
    }
}

/// Compute terminal reward for corner classification.
fn compute_corner_terminal_reward(
    surface_points: &[(f64, f64, f64)],
    expected_normals: &[(f64, f64, f64); 3],
    cell_size: f64,
    config: &TerminalRewardConfig,
) -> TerminalRewardResult {
    let fit = fit_corner_from_samples(surface_points, cell_size, &config.classifier);

    match fit {
        None => TerminalRewardResult {
            reward: 0.0,
            fit_success: false,
            residual_reward: 0.0,
            normal_reward: 0.0,
            surface_points_count: surface_points.len(),
        },
        Some(result) => {
            // Residual reward: average of all planes
            let avg_residual: f64 =
                result.residuals.iter().sum::<f64>() / result.residuals.len() as f64;
            let normalized_residual = (avg_residual / cell_size) / config.max_residual;
            let residual_reward = config.w_residual * (1.0 - normalized_residual.min(1.0));

            // Normal accuracy reward: best matching
            let errors = best_corner_normal_errors(&result.normals, expected_normals);
            let avg_normal_error = if errors.is_empty() {
                config.max_normal_error
            } else {
                errors.iter().sum::<f64>() / errors.len() as f64
            };
            let normalized_error = avg_normal_error / config.max_normal_error;
            let normal_reward = config.w_normal_accuracy * (1.0 - normalized_error.min(1.0));

            // Bonus for finding all 3 planes
            let completeness_bonus = if result.normals.len() >= 3 { 1.0 } else { 0.0 };

            let total = config.w_fit_success + residual_reward + normal_reward + completeness_bonus;

            TerminalRewardResult {
                reward: total,
                fit_success: true,
                residual_reward,
                normal_reward,
                surface_points_count: surface_points.len(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_reward_decay() {
        let config = RewardConfig::default();
        let cell_size = 0.05;
        let prev = vec![];

        // On surface: high reward
        let r_on = compute_step_reward((0.0, 0.0, 0.0), 0.0, &prev, false, cell_size, &config);

        // Half cell away: medium reward
        let r_half = compute_step_reward(
            (0.0, 0.0, 0.0),
            cell_size * 0.5,
            &prev,
            false,
            cell_size,
            &config,
        );

        // One cell away: low reward
        let r_one = compute_step_reward(
            (0.0, 0.0, 0.0),
            cell_size,
            &prev,
            false,
            cell_size,
            &config,
        );

        assert!(r_on > r_half);
        assert!(r_half > r_one);
    }

    #[test]
    fn test_spread_reward() {
        let config = RewardConfig::default();
        let cell_size = 0.05;
        let prev = vec![(0.0, 0.0, 0.0)];

        // Close to previous: low spread reward
        let r_close = compute_step_reward(
            (0.001, 0.0, 0.0),
            0.0,
            &prev,
            false,
            cell_size,
            &config,
        );

        // Far from previous: high spread reward
        let r_far = compute_step_reward(
            (cell_size, 0.0, 0.0),
            0.0,
            &prev,
            false,
            cell_size,
            &config,
        );

        assert!(r_far > r_close);
    }

    #[test]
    fn test_crossing_bonus() {
        let config = RewardConfig::default();
        let cell_size = 0.05;
        let prev = vec![];

        let r_no_cross = compute_step_reward((0.0, 0.0, 0.0), 0.0, &prev, false, cell_size, &config);
        let r_cross = compute_step_reward((0.0, 0.0, 0.0), 0.0, &prev, true, cell_size, &config);

        assert!((r_cross - r_no_cross - config.crossing_bonus).abs() < 1e-10);
    }

    #[test]
    fn test_returns() {
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        let discount = 0.9;
        let returns = compute_returns(&rewards, discount);

        // G_3 = 1
        // G_2 = 1 + 0.9 * 1 = 1.9
        // G_1 = 1 + 0.9 * 1.9 = 2.71
        // G_0 = 1 + 0.9 * 2.71 = 3.439
        assert!((returns[3] - 1.0).abs() < 1e-10);
        assert!((returns[2] - 1.9).abs() < 1e-10);
        assert!((returns[1] - 2.71).abs() < 1e-10);
        assert!((returns[0] - 3.439).abs() < 1e-10);
    }

    #[test]
    fn test_terminal_reward_face() {
        // Simulate samples with crossings that form a plane
        let samples = vec![
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.01),  // Crossing
            (0.0, 0.1, -0.01), // Crossing back
            (0.1, 0.1, 0.01),  // Crossing
            (-0.1, 0.0, -0.01), // Crossing back
        ];
        let inside_flags = vec![true, false, true, false, true];
        let vertex = (0.0, 0.0, 0.0);
        let expected = ExpectedClassification::OnFace {
            face_index: 0,
            expected_normal: (0.0, 0.0, 1.0),
        };

        let config = TerminalRewardConfig::default();
        let result = compute_terminal_reward(
            &samples,
            &inside_flags,
            vertex,
            &expected,
            0.1,
            &config,
        );

        // Should get some reward for finding crossings
        assert!(result.surface_points_count > 0);
    }
}
