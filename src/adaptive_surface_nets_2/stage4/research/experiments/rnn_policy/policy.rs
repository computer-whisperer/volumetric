//! RNN Policy: combines GRU + Chooser for sampling decisions.
//!
//! The policy takes the current state and hidden representation,
//! updates the hidden state via GRU, and outputs a sample position.

use crate::adaptive_surface_nets_2::stage4::research::analytical_cube::AnalyticalRotatedCube;
use crate::adaptive_surface_nets_2::stage4::research::sample_cache::SampleCache;
use crate::adaptive_surface_nets_2::stage4::research::validation::ValidationPoint;

use super::chooser::{
    chooser_forward, octant_corners, position_from_probs, sample_categorical, ChooserCache,
    ChooserWeights,
};
use super::classifier_heads::{ClassifierHeads, ClassifierPredictions};
use super::gru::{gru_forward, GruCache, GruWeights, Rng, HIDDEN_DIM};
use super::math::argmax;
use super::reward::{compute_step_reward_with_entropy, RewardConfig};

/// Input dimension for the GRU.
/// Features: prev_offset(3), in_out_sign(1), budget_remaining(1), crossings_found(1)
pub const INPUT_DIM: usize = 6;

/// Maximum samples per episode.
pub const BUDGET: usize = 50;

/// Cell size for research experiments.
pub const CELL_SIZE: f64 = 0.05;

/// RNN Policy combining GRU, Chooser head, and Classifier heads.
#[derive(Clone)]
pub struct RnnPolicy {
    pub gru: GruWeights,
    pub chooser: ChooserWeights,
    pub classifier_heads: ClassifierHeads,
}

impl RnnPolicy {
    /// Create a new policy with random initialization.
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            gru: GruWeights::new(INPUT_DIM, rng),
            chooser: ChooserWeights::new(rng),
            classifier_heads: ClassifierHeads::new(rng),
        }
    }

    /// Total number of parameters.
    pub fn param_count(&self) -> usize {
        self.gru.param_count() + self.chooser.param_count() + self.classifier_heads.param_count()
    }

    /// Run classifier heads on the given hidden state.
    pub fn classify(&self, hidden: &[f64]) -> ClassifierPredictions {
        self.classifier_heads.forward(hidden)
    }

    /// Save policy weights to a file.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Write magic and version (bump to v2 for classifier heads)
        file.write_all(b"RNNP")?;
        file.write_all(&2u32.to_le_bytes())?;

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

        // Classifier head weights (v2+)
        write_vec(&mut file, &self.classifier_heads.face.w)?;
        write_vec(&mut file, &self.classifier_heads.face.b)?;
        write_vec(&mut file, &self.classifier_heads.edge.w)?;
        write_vec(&mut file, &self.classifier_heads.edge.b)?;
        write_vec(&mut file, &self.classifier_heads.corner.w)?;
        write_vec(&mut file, &self.classifier_heads.corner.b)?;

        Ok(())
    }

    /// Load policy weights from a file.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        use super::classifier_heads::{HeadWeights, FACE_HEAD_OUTPUTS, EDGE_HEAD_OUTPUTS, CORNER_HEAD_OUTPUTS};

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
        if version != 1 && version != 2 {
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

        // Classifier head weights (v2+), or initialize randomly for v1
        let classifier_heads = if version >= 2 {
            let face_w = read_vec(&mut file)?;
            let face_b = read_vec(&mut file)?;
            let edge_w = read_vec(&mut file)?;
            let edge_b = read_vec(&mut file)?;
            let corner_w = read_vec(&mut file)?;
            let corner_b = read_vec(&mut file)?;

            ClassifierHeads {
                face: HeadWeights {
                    w: face_w,
                    b: face_b,
                    output_dim: FACE_HEAD_OUTPUTS,
                },
                edge: HeadWeights {
                    w: edge_w,
                    b: edge_b,
                    output_dim: EDGE_HEAD_OUTPUTS,
                },
                corner: HeadWeights {
                    w: corner_w,
                    b: corner_b,
                    output_dim: CORNER_HEAD_OUTPUTS,
                },
            }
        } else {
            // v1 file: initialize classifier heads randomly
            let mut rng = Rng::new(42);
            ClassifierHeads::new(&mut rng)
        };

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
            classifier_heads,
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
    /// Inside/outside flags for each sample (for terminal reward).
    pub inside_flags: Vec<bool>,
    /// Initial hidden state.
    pub h_init: Vec<f64>,
    /// Final hidden state (for classifier heads).
    pub h_final: Vec<f64>,
    /// Terminal reward (classifier-based, computed after episode).
    pub terminal_reward: f64,
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
    run_episode_ex(policy, cube, point, point_idx, rng, training, false, reward_config)
}

/// Run a single episode with the policy (extended version).
///
/// # Arguments
/// * `policy` - The RNN policy
/// * `cube` - Analytical cube for oracle queries
/// * `point` - Validation point to process
/// * `point_idx` - Index of the point (for tracking)
/// * `rng` - Random number generator
/// * `stochastic` - Whether to sample stochastically (true) or use argmax (false)
/// * `use_discrete` - Whether to use discrete corners (true) or weighted positions (false)
/// * `reward_config` - Reward configuration
pub fn run_episode_ex(
    policy: &RnnPolicy,
    cube: &AnalyticalRotatedCube,
    point: &ValidationPoint,
    point_idx: usize,
    rng: &mut Rng,
    stochastic: bool,
    use_discrete: bool,
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
    let mut inside_flags: Vec<bool> = Vec::new();
    let mut steps = Vec::new();
    let mut rewards = Vec::new();
    let mut prev_inside: Option<bool> = None;
    let mut prev_sample: Option<(f64, f64, f64)> = None;
    let mut crossings_found: usize = 0;

    let corners = octant_corners(point.position, CELL_SIZE);

    // Run episode
    for step_idx in 0..BUDGET {
        // Build input features
        let input = build_input(
            point.position,
            prev_sample,
            prev_inside,
            step_idx,
            crossings_found,
            CELL_SIZE,
        );

        // GRU forward
        let (new_h, gru_cache) = gru_forward(&policy.gru, &input, &h);
        h = new_h;

        // Chooser forward
        let (_logits, probs, mut chooser_cache) = chooser_forward(&policy.chooser, &h);

        // Select action
        let action = if stochastic {
            sample_categorical(&probs, rng)
        } else {
            argmax(&probs)
        };
        chooser_cache.action = action;

        // Compute sample position
        let sample_pos = if use_discrete || stochastic {
            // Use discrete corner with jitter during training
            let base = corners[action];
            if stochastic {
                // Add small random jitter to break cache degeneracy
                // Jitter is ~5% of cell size to stay near the corner
                let jitter_scale = CELL_SIZE * 0.05;
                (
                    base.0 + (rng.next_f64() - 0.5) * jitter_scale,
                    base.1 + (rng.next_f64() - 0.5) * jitter_scale,
                    base.2 + (rng.next_f64() - 0.5) * jitter_scale,
                )
            } else {
                base
            }
        } else {
            // Use weighted average for smoother behavior
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
        if is_crossing {
            crossings_found += 1;
        }
        prev_inside = Some(is_inside);
        prev_sample = Some(sample_pos);

        // Get oracle distance
        let closest = cube.closest_surface_point(sample_pos);
        let oracle_distance = closest.distance;

        // Compute reward with entropy bonus
        let reward = compute_step_reward_with_entropy(
            sample_pos,
            oracle_distance,
            &samples,
            is_crossing,
            CELL_SIZE,
            &probs,
            reward_config,
        );

        samples.push(sample_pos);
        inside_flags.push(is_inside);
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
        inside_flags,
        h_init,
        h_final: h,
        terminal_reward: 0.0, // Computed later by training loop
    }
}

/// Build input features for one step.
///
/// All spatial values are normalized by cell_size so the policy operates
/// in a unit-cube reference frame.
fn build_input(
    vertex: (f64, f64, f64),
    prev_sample: Option<(f64, f64, f64)>,
    prev_inside: Option<bool>,
    step_idx: usize,
    crossings_found: usize,
    cell_size: f64,
) -> Vec<f64> {
    // Previous sample offset from vertex, normalized by cell_size
    // Range: approximately [-1, 1] since samples are within the cell
    let prev_offset = match prev_sample {
        Some(s) => (
            (s.0 - vertex.0) / cell_size,
            (s.1 - vertex.1) / cell_size,
            (s.2 - vertex.2) / cell_size,
        ),
        None => (0.0, 0.0, 0.0),
    };

    // Inside/outside sign: +1 inside, -1 outside, 0 if no previous sample
    let in_out_sign = match prev_inside {
        Some(true) => 1.0,
        Some(false) => -1.0,
        None => 0.0,
    };

    // Budget remaining (1.0 at start, 0.0 at end)
    let budget_remaining = 1.0 - (step_idx as f64 / BUDGET as f64);

    // Crossings found so far, normalized (expect ~3-10 crossings for good fit)
    let crossings_norm = (crossings_found as f64 / 5.0).min(2.0);

    vec![
        prev_offset.0,      // 0: previous sample X offset / cell_size
        prev_offset.1,      // 1: previous sample Y offset / cell_size
        prev_offset.2,      // 2: previous sample Z offset / cell_size
        in_out_sign,        // 3: previous sample inside(+1) / outside(-1) / none(0)
        budget_remaining,   // 4: fraction of budget remaining
        crossings_norm,     // 5: crossings found (normalized)
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_surface_nets_2::stage4::research::validation::ExpectedClassification;

    #[test]
    fn test_policy_param_count() {
        let mut rng = Rng::new(42);
        let policy = RnnPolicy::new(&mut rng);
        let count = policy.param_count();
        // With INPUT_DIM=6, HIDDEN_DIM=32, NUM_OCTANTS=8:
        // GRU: 3 * (32*6 + 32*32 + 32) = 3 * (192 + 1024 + 32) = 3 * 1248 = 3744
        // Chooser: 8*32 + 8 = 264
        // Classifier heads:
        //   Face: 4*32 + 4 = 132
        //   Edge: 10*32 + 10 = 330
        //   Corner: 10*32 + 10 = 330
        //   Total: 792
        // Grand total: 3744 + 264 + 792 = 4800
        assert!(count > 4500 && count < 5200, "param_count={}", count);
    }

    #[test]
    fn test_build_input() {
        let vertex = (0.0, 0.0, 0.0);
        let prev_sample = Some((0.01, 0.02, -0.01));
        let input = build_input(vertex, prev_sample, Some(true), 5, 2, CELL_SIZE);

        assert_eq!(input.len(), INPUT_DIM);
        // prev_offset = sample / cell_size
        assert!((input[0] - 0.01 / CELL_SIZE).abs() < 1e-10);
        assert!((input[1] - 0.02 / CELL_SIZE).abs() < 1e-10);
        assert!((input[2] - -0.01 / CELL_SIZE).abs() < 1e-10);
        // in_out_sign = +1 for inside
        assert!((input[3] - 1.0).abs() < 1e-10);
        // budget_remaining at step 5
        assert!((input[4] - (1.0 - 5.0 / BUDGET as f64)).abs() < 1e-10);
        // crossings_norm = 2 / 5 = 0.4
        assert!((input[5] - 0.4).abs() < 1e-10);
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
