# RNN-Based Sampling Policy MVP

## Overview

This module implements a recurrent neural network policy that learns **where** to sample in a cubic volume around a vertex, replacing fixed-direction sampling approaches. The policy observes the results of previous samples and adaptively chooses the next sample location to maximize information gain about the local surface geometry.

## Motivation

Previous sampling approaches (attempts 0-2, ml_policy) use predetermined sampling patterns:
- Fixed directional grids
- Uniform random sampling
- Direction-based ML policies that pick from predefined vectors

These approaches don't adapt based on what they've already learned. The RNN policy addresses this by:
1. Maintaining a hidden state that summarizes sampling history
2. Choosing sample locations dynamically based on observed in/out patterns
3. Learning to explore efficiently through reinforcement learning

## Architecture

### Network Components

```
Input[10] → GRU[32] → Linear[8] → Softmax → Position
```

#### GRU Updater
- **Input dimension**: 10 features
- **Hidden dimension**: 32
- **Output**: Updated hidden state

The GRU was chosen over LSTM because:
- Simpler (fewer parameters)
- Sufficient for short episodes (~50 steps)
- Faster to train

#### Chooser Head
- **Input**: 32-dimensional hidden state
- **Output**: 8 weights (one per octant)
- **Position calculation**: Weighted average of cube corners

### Input Features

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0-2 | Normalized position | [-1, 1] | Last sample pos relative to vertex, normalized by cell_size, tanh-squashed |
| 3 | Inside indicator | {-1, 0, 1} | Last sample: inside=1, outside=-1, unknown=0 |
| 4 | Budget fraction | [0, 1] | step_idx / BUDGET |
| 5 | Sample count fraction | [0, 1] | num_samples / BUDGET |
| 6 | Spread metric | [0, 2] | Average pairwise distance / cell_size |
| 7 | Last reward | [-1, 1] | Previous step reward, tanh-squashed |
| 8 | Hint normal x | [-1, 1] | X component of expected normal direction |
| 9 | Bias | 1.0 | Constant bias term |

### Parameter Count

| Component | Parameters |
|-----------|------------|
| GRU W_z, W_r, W_h | 3 × (32 × 10) = 960 |
| GRU U_z, U_r, U_h | 3 × (32 × 32) = 3,072 |
| GRU b_z, b_r, b_h | 3 × 32 = 96 |
| Chooser W | 8 × 32 = 256 |
| Chooser b | 8 |
| **Total** | **4,392** |

## Octant System

The policy outputs weights for 8 octants (cube corners):

```
Index  Signs   Corner Position (relative to vertex)
  0    ---    (-0.5, -0.5, -0.5) × cell_size
  1    +--    (+0.5, -0.5, -0.5) × cell_size
  2    -+-    (-0.5, +0.5, -0.5) × cell_size
  3    ++-    (+0.5, +0.5, -0.5) × cell_size
  4    --+    (-0.5, -0.5, +0.5) × cell_size
  5    +-+    (+0.5, -0.5, +0.5) × cell_size
  6    -++    (-0.5, +0.5, +0.5) × cell_size
  7    +++    (+0.5, +0.5, +0.5) × cell_size
```

During training, actions are sampled from the softmax distribution. During evaluation, the position is computed as a weighted average of all corners.

## Reward Structure

Each sampling step receives a reward:

```
reward = W_SURFACE × exp(-distance/cell_size × 5)   // Surface proximity
       + W_SPREAD × min_dist_to_prev_samples        // Exploration bonus
       + crossing_bonus                              // In/out transition found
       - LAMBDA                                      // Per-sample cost
```

### Default Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| W_SURFACE | 1.0 | Weight for surface proximity |
| W_SPREAD | 0.3 | Weight for spatial exploration |
| crossing_bonus | 0.5 | Bonus for finding in/out transition |
| LAMBDA | 0.02 | Per-sample cost (encourages efficiency) |
| surface_decay | 5.0 | Exponential decay rate |

### Reward Components

**Surface proximity**: Rewards samples close to the actual surface. Uses oracle distance (in research setting) to compute `exp(-d/cell_size × 5)`.

**Spread bonus**: Rewards exploring new areas. Computed as `min(distances_to_previous_samples) / cell_size`, capped at 1.0.

**Crossing bonus**: Flat bonus when a sample transitions between inside and outside the surface.

**Per-sample cost**: Small penalty per sample to encourage finding information quickly.

## Training

### Algorithm: REINFORCE with Baseline

1. Roll out episode using current policy (stochastic sampling)
2. Compute discounted returns: G_t = r_t + γ × G_{t+1}
3. Compute baseline (average return)
4. Compute advantages: A_t = G_t - baseline
5. Compute policy gradient via BPTT
6. Update with Adam optimizer

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Learning rate | 0.001 |
| Discount (γ) | 0.98 |
| Gradient clipping | 1.0 |
| Adam β₁ | 0.9 |
| Adam β₂ | 0.999 |
| Budget (max samples) | 50 |

### Gradient Computation

Full backpropagation through time (BPTT) is used since episodes are short. The gradient flows:

```
Loss → Chooser gradients → GRU gradients (backward through time)
```

For each timestep t (in reverse order):
1. Compute chooser gradient: ∇ log π(a_t) × advantage_t
2. Backpropagate through GRU to get dL/dh_{t-1}
3. Accumulate parameter gradients

## Current Results

### Validation Set
- 44 points on analytical rotated cube
- 12 face points, 24 edge points, 8 corner points
- Cell size: 0.05 (5% of unit cube edge)

### Training Metrics (200 epochs)

| Metric | Epoch 1 | Epoch 200 |
|--------|---------|-----------|
| Average return | 16.5 | 22.8 |
| Average reward/step | 0.33 | 0.46 |
| Gradient norm | 2.0 | 2.1 |
| Samples used | 50 | 50 |

### Classification Accuracy

| Classification | Before Training | After Training |
|----------------|-----------------|----------------|
| Face | ~33% | **100%** |
| Edge | ~12% | 12.5% |
| Corner | ~12% | 12.5% |
| **Overall** | 36.4% | 36.4% |

### Analysis

**Face points (100% accuracy)**: The policy learns to sample near the surface effectively for simple planar regions.

**Edge/corner points (low accuracy)**: The current evaluation uses a placeholder classifier that just queries the oracle at the average sample position. This doesn't use the sample distribution to actually fit geometry. Proper edge/corner detection would require:
- Fitting multiple planes to sample clusters
- Detecting in/out transition patterns indicative of edges
- Using crossing points for geometry reconstruction

**Samples per episode**: The policy currently uses the full budget (50 samples). The per-sample cost (LAMBDA=0.02) may need to be increased, or an early-stopping mechanism could be added when sufficient crossings are found.

## File Organization

```
rnn_policy/
├── mod.rs          # Public API: run_rnn_policy_experiment()
├── math.rs         # Vector/matrix operations
├── gru.rs          # GRU forward/backward passes
├── chooser.rs      # Octant weights → position mapping
├── policy.rs       # RnnPolicy struct, episode rollout
├── gradients.rs    # BPTT gradient computation
├── reward.rs       # Reward function
├── training.rs     # Adam optimizer, training loop
└── README.md       # This file
```

## Usage

### Run the experiment
```rust
use experiments::rnn_policy::run_rnn_policy_experiment;
run_rnn_policy_experiment();
```

### Generate sample clouds for visualization
```rust
use experiments::rnn_policy::{dump_rnn_policy_sample_cloud, RnnPolicyDumpKind};
dump_rnn_policy_sample_cloud(RnnPolicyDumpKind::Trained, Path::new("samples.cbor"));
```

### Run via cargo test
```bash
cargo test experiment_rnn_policy_mvp --lib -- --nocapture
```

## Future Improvements

### Short-term
1. **Better classifier**: Use sample distributions and crossing patterns to fit geometry instead of oracle queries
2. **Early stopping**: End episodes when sufficient crossings found
3. **Entropy bonus**: Add entropy regularization to encourage exploration during training
4. **Curriculum learning**: Start with face points, gradually add edges and corners

### Medium-term
1. **Attention mechanism**: Replace GRU with transformer for better long-range dependencies
2. **Multi-head output**: Separate heads for face/edge/corner classification
3. **Position encoding**: Better representation of 3D sample positions
4. **Larger hidden dimension**: Try 64 or 128 for more capacity

### Long-term
1. **Imitation learning**: Pre-train on successful sampling trajectories from existing algorithms
2. **Multi-task learning**: Jointly learn sampling policy and geometry fitting
3. **Generalization**: Train on diverse shapes beyond rotated cube
4. **Integration**: Replace sample selection in actual surface nets algorithm

## References

- GRU: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
- REINFORCE: Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)
- Adam: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)
