//! Reward computation for RNN sampling policy.
//!
//! The reward structure encourages:
//! 1. Sampling close to the surface
//! 2. Spatial exploration (spread out samples)
//! 3. Finding in/out transitions
//! 4. Efficiency (penalty per sample)

use super::math::distance_3d;

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
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            w_surface: 1.0,
            w_spread: 0.3,
            crossing_bonus: 0.5,
            lambda: 0.02,
            surface_decay: 5.0,
        }
    }
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
}
