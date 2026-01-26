//! RNN Policy: combines GRU + Chooser for sampling decisions.
//!
//! The policy takes the current state and hidden representation,
//! updates the hidden state via GRU, and outputs a sample position.

use crate::adaptive_surface_nets_2::stage4::research::analytical_cube::AnalyticalRotatedCube;
use crate::adaptive_surface_nets_2::stage4::research::sample_cache::SampleCache;
use crate::adaptive_surface_nets_2::stage4::research::validation::{ExpectedClassification, ValidationPoint};

use super::chooser::{
    chooser_forward, octant_corners, position_from_probs, sample_categorical, ChooserCache,
    ChooserWeights,
};
use super::gru::{gru_forward, GruCache, GruWeights, Rng, HIDDEN_DIM};
use super::math::{argmax, distance_3d, normalize_3d, add_3d};
use super::reward::{compute_step_reward, RewardConfig};

/// Input dimension for the GRU.
/// Features: normalized_pos(3), is_inside(1), oracle_dist(1), budget_frac(1),
///           sample_count(1), spread_metric(1), last_reward(1), bias(1)
pub const INPUT_DIM: usize = 10;

/// Maximum samples per episode.
pub const BUDGET: usize = 50;

/// Cell size for research experiments.
pub const CELL_SIZE: f64 = 0.05;

/// RNN Policy combining GRU and Chooser head.
#[derive(Clone)]
pub struct RnnPolicy {
    pub gru: GruWeights,
    pub chooser: ChooserWeights,
}

impl RnnPolicy {
    /// Create a new policy with random initialization.
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            gru: GruWeights::new(INPUT_DIM, rng),
            chooser: ChooserWeights::new(rng),
        }
    }

    /// Total number of parameters.
    pub fn param_count(&self) -> usize {
        self.gru.param_count() + self.chooser.param_count()
    }

    /// Save policy weights to a file.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Write magic and version
        file.write_all(b"RNNP")?;
        file.write_all(&1u32.to_le_bytes())?;

        // Write dimensions
        file.write_all(&(INPUT_DIM as u32).to_le_bytes())?;
        file.write_all(&(HIDDEN_DIM as u32).to_le_bytes())?;

        // Helper to write a Vec<f64>
        let write_vec = |f: &mut std::fs::File, v: &[f64]| -> std::io::Result<()> {
            f.write_all(&(v.len() as u32).to_le_bytes())?;
            for &x in v {
                f.write_all(&x.to_le_bytes())?;
            }
            Ok(())
        };

        // GRU weights
        write_vec(&mut file, &self.gru.w_z)?;
        write_vec(&mut file, &self.gru.u_z)?;
        write_vec(&mut file, &self.gru.b_z)?;
        write_vec(&mut file, &self.gru.w_r)?;
        write_vec(&mut file, &self.gru.u_r)?;
        write_vec(&mut file, &self.gru.b_r)?;
        write_vec(&mut file, &self.gru.w_h)?;
        write_vec(&mut file, &self.gru.u_h)?;
        write_vec(&mut file, &self.gru.b_h)?;

        // Chooser weights
        write_vec(&mut file, &self.chooser.w)?;
        write_vec(&mut file, &self.chooser.b)?;

        Ok(())
    }

    /// Load policy weights from a file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;

        // Read and verify magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"RNNP" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid RNN policy file magic",
            ));
        }

        // Read version
        let mut version = [0u8; 4];
        file.read_exact(&mut version)?;
        let version = u32::from_le_bytes(version);
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported RNN policy version: {}", version),
            ));
        }

        // Read dimensions
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let input_dim = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4)?;
        let hidden_dim = u32::from_le_bytes(buf4) as usize;

        if input_dim != INPUT_DIM || hidden_dim != HIDDEN_DIM {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Dimension mismatch: file has input={} hidden={}, expected input={} hidden={}",
                    input_dim, hidden_dim, INPUT_DIM, HIDDEN_DIM
                ),
            ));
        }

        // Helper to read a Vec<f64>
        let read_vec = |f: &mut std::fs::File| -> std::io::Result<Vec<f64>> {
            let mut buf4 = [0u8; 4];
            f.read_exact(&mut buf4)?;
            let len = u32::from_le_bytes(buf4) as usize;
            let mut vec = Vec::with_capacity(len);
            let mut buf8 = [0u8; 8];
            for _ in 0..len {
                f.read_exact(&mut buf8)?;
                vec.push(f64::from_le_bytes(buf8));
            }
            Ok(vec)
        };

        // GRU weights
        let w_z = read_vec(&mut file)?;
        let u_z = read_vec(&mut file)?;
        let b_z = read_vec(&mut file)?;
        let w_r = read_vec(&mut file)?;
        let u_r = read_vec(&mut file)?;
        let b_r = read_vec(&mut file)?;
        let w_h = read_vec(&mut file)?;
        let u_h = read_vec(&mut file)?;
        let b_h = read_vec(&mut file)?;

        // Chooser weights
        let chooser_w = read_vec(&mut file)?;
        let chooser_b = read_vec(&mut file)?;

        Ok(Self {
            gru: GruWeights {
                input_dim: INPUT_DIM,
                w_z,
                u_z,
                b_z,
                w_r,
                u_r,
                b_r,
                w_h,
                u_h,
                b_h,
            },
            chooser: super::chooser::ChooserWeights {
                w: chooser_w,
                b: chooser_b,
            },
        })
    }
}

/// A single step in an episode, with all information needed for BPTT.
#[derive(Clone)]
pub struct EpisodeStep {
    /// Input features for this step.
    pub input: Vec<f64>,
    /// GRU cache for backprop.
    pub gru_cache: GruCache,
    /// Chooser cache for backprop.
    pub chooser_cache: ChooserCache,
    /// Action taken (octant index).
    pub action: usize,
    /// Sample position.
    pub sample_pos: (f64, f64, f64),
    /// Whether sample was inside.
    pub is_inside: bool,
    /// Oracle distance to surface.
    pub oracle_distance: f64,
    /// Whether this was a crossing.
    pub is_crossing: bool,
    /// Reward received.
    pub reward: f64,
}

/// Complete episode trace for training.
#[derive(Clone)]
pub struct Episode {
    /// Validation point being processed.
    pub point_idx: usize,
    /// All steps in the episode.
    pub steps: Vec<EpisodeStep>,
    /// All rewards.
    pub rewards: Vec<f64>,
    /// Total samples used.
    pub samples_used: u64,
    /// Final accumulated samples.
    pub final_samples: Vec<(f64, f64, f64)>,
    /// Initial hidden state.
    pub h_init: Vec<f64>,
}

/// Run a single episode with the policy.
///
/// # Arguments
/// * `policy` - The RNN policy
/// * `cube` - Analytical cube for oracle queries
/// * `point` - Validation point to process
/// * `point_idx` - Index of the point (for tracking)
/// * `rng` - Random number generator
/// * `training` - Whether to sample stochastically (true) or use argmax (false)
/// * `reward_config` - Reward configuration
pub fn run_episode(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    point: &ValidationPoint,
    point_idx: usize,
    rng: &mut Rng,
    training: bool,
    reward_config: &RewardConfig,
) -> Episode {
    // Create sampler for the cube
    let sampler = |x: f64, y: f64, z: f64| -> f32 {
        let local = cube.world_to_local((x, y, z));
        let h = 0.5;
        if local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h {
            1.0
        } else {
            -1.0
        }
    };
    let cache = SampleCache::new(sampler);

    // Initialize state
    let mut h = vec![0.0; HIDDEN_DIM];
    let h_init = h.clone();
    let mut samples: Vec<(f64, f64, f64)> = Vec::new();
    let mut steps = Vec::new();
    let mut rewards = Vec::new();
    let mut prev_inside: Option<bool> = None;
    let mut last_reward = 0.0;

    let corners = octant_corners(point.position, CELL_SIZE);
    let hint_normal = hint_from_expected(&point.expected);

    // Run episode
    for step_idx in 0..BUDGET {
        // Build input features
        let input = build_input(
            point.position,
            &samples,
            step_idx,
            prev_inside,
            last_reward,
            hint_normal,
            CELL_SIZE,
        );

        // GRU forward
        let (new_h, gru_cache) = gru_forward(&policy.gru, &input, &h);
        h = new_h;

        // Chooser forward
        let (_logits, probs, mut chooser_cache) = chooser_forward(&policy.chooser, &h);

        // Select action
        let action = if training {
            sample_categorical(&probs, rng)
        } else {
            argmax(&probs)
        };
        chooser_cache.action = action;

        // Compute sample position
        let sample_pos = if training {
            // During training, use the sampled corner
            corners[action]
        } else {
            // During evaluation, use weighted average for smoother behavior
            position_from_probs(&probs, &corners)
        };

        // Query the sampler
        let is_inside = cache.is_inside(sample_pos.0, sample_pos.1, sample_pos.2);

        // Check for crossing
        let is_crossing = if let Some(prev) = prev_inside {
            prev != is_inside
        } else {
            false
        };
        prev_inside = Some(is_inside);

        // Get oracle distance
        let closest = cube.closest_surface_point(sample_pos);
        let oracle_distance = closest.distance;

        // Compute reward
        let reward = compute_step_reward(
            sample_pos,
            oracle_distance,
            &samples,
            is_crossing,
            CELL_SIZE,
            reward_config,
        );
        last_reward = reward;

        samples.push(sample_pos);
        rewards.push(reward);
        steps.push(EpisodeStep {
            input,
            gru_cache,
            chooser_cache,
            action,
            sample_pos,
            is_inside,
            oracle_distance,
            is_crossing,
            reward,
        });
    }

    Episode {
        point_idx,
        steps,
        rewards,
        samples_used: cache.stats().actual_samples(),
        final_samples: samples,
        h_init,
    }
}

/// Build input features for one step.
fn build_input(
    vertex: (f64, f64, f64),
    samples: &[(f64, f64, f64)],
    step_idx: usize,
    prev_inside: Option<bool>,
    last_reward: f64,
    hint_normal: (f64, f64, f64),
    cell_size: f64,
) -> Vec<f64> {
    // Normalized position relative to vertex
    let rel_pos = if samples.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let last = samples.last().unwrap();
        (
            (last.0 - vertex.0) / cell_size,
            (last.1 - vertex.1) / cell_size,
            (last.2 - vertex.2) / cell_size,
        )
    };

    // Inside/outside indicator
    let inside_indicator = match prev_inside {
        Some(true) => 1.0,
        Some(false) => -1.0,
        None => 0.0,
    };

    // Spread metric: average distance between samples
    let spread = if samples.len() < 2 {
        0.0
    } else {
        let mut total_dist = 0.0;
        let mut count = 0;
        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                total_dist += distance_3d(samples[i], samples[j]);
                count += 1;
            }
        }
        if count > 0 {
            (total_dist / count as f64 / cell_size).min(2.0)
        } else {
            0.0
        }
    };

    vec![
        rel_pos.0.tanh(),                          // 0: normalized x
        rel_pos.1.tanh(),                          // 1: normalized y
        rel_pos.2.tanh(),                          // 2: normalized z
        inside_indicator,                          // 3: last sample inside/outside
        (step_idx as f64 / BUDGET as f64),         // 4: budget fraction
        (samples.len() as f64 / BUDGET as f64),    // 5: sample count fraction
        spread,                                     // 6: spread metric
        last_reward.tanh(),                        // 7: last reward (normalized)
        hint_normal.0,                             // 8: hint normal x
        1.0,                                       // 9: bias
    ]
}

/// Extract hint normal from expected classification.
fn hint_from_expected(expected: &ExpectedClassification) -> (f64, f64, f64) {
    match expected {
        ExpectedClassification::OnFace { expected_normal, .. } => *expected_normal,
        ExpectedClassification::OnEdge { expected_normals, .. } => {
            normalize_3d(add_3d(expected_normals.0, expected_normals.1))
        }
        ExpectedClassification::OnCorner { expected_normals, .. } => {
            let sum = add_3d(
                add_3d(expected_normals[0], expected_normals[1]),
                expected_normals[2],
            );
            normalize_3d(sum)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_param_count() {
        let mut rng = Rng::new(42);
        let policy = RnnPolicy::new(&mut rng);
        let count = policy.param_count();
        // Should be in the ~4500 range
        assert!(count > 4000 && count < 5000);
    }

    #[test]
    fn test_build_input() {
        let vertex = (0.0, 0.0, 0.0);
        let samples = vec![(0.01, 0.0, 0.0)];
        let hint = (0.0, 1.0, 0.0);
        let input = build_input(vertex, &samples, 5, Some(true), 0.5, hint, CELL_SIZE);

        assert_eq!(input.len(), INPUT_DIM);
        // Budget fraction at step 5
        assert!((input[4] - 5.0 / BUDGET as f64).abs() < 1e-10);
        // One sample
        assert!((input[5] - 1.0 / BUDGET as f64).abs() < 1e-10);
    }

    #[test]
    fn test_run_episode() {
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

        assert_eq!(episode.steps.len(), BUDGET);
        assert_eq!(episode.rewards.len(), BUDGET);
        assert_eq!(episode.final_samples.len(), BUDGET);
    }
}
