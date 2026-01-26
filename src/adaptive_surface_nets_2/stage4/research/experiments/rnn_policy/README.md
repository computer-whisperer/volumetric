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
Input[6] → GRU[32] → Linear[8] → Softmax → Weighted Position
```

#### GRU Updater
- **Input dimension**: 6 features (clean, no oracle cheating)
- **Hidden dimension**: 32
- **Output**: Updated hidden state

The GRU was chosen over LSTM because:
- Simpler (fewer parameters)
- Sufficient for short episodes (~50 steps)
- Faster to train

#### Chooser Head
- **Input**: 32-dimensional hidden state
- **Output**: 8 weights (one per octant corner)
- **Position calculation**: Weighted average of cube corners via softmax

### Input Features (v2 - Clean Design)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0-2 | prev_offset | ~[-1, 1] | `(prev_sample - vertex) / cell_size` |
| 3 | in_out_sign | {-1, 0, 1} | Previous sample: inside=+1, outside=-1, first=0 |
| 4 | budget_remaining | [0, 1] | `1 - step_idx / BUDGET` |
| 5 | crossings_norm | [0, 2] | `crossings_found / 5` (capped) |

**Key design decisions:**
- No oracle information in inputs (previous version cheated with hint_normal)
- All spatial values normalized by cell_size
- Minimal feature set - only information available at inference time

### Parameter Count

| Component | Parameters |
|-----------|------------|
| GRU W_z, W_r, W_h | 3 × (32 × 6) = 576 |
| GRU U_z, U_r, U_h | 3 × (32 × 32) = 3,072 |
| GRU b_z, b_r, b_h | 3 × 32 = 96 |
| Chooser W | 8 × 32 = 256 |
| Chooser b | 8 |
| **Total** | **4,008** |

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

The sampling cell is a cube of side `cell_size` centered on the control vertex.

**Sampling modes:**
- **Training (stochastic)**: Sample octant from softmax, use discrete corner
- **Evaluation (weighted)**: Compute position as weighted average of all corners

## Reward Structure

Each sampling step receives a reward:

```
reward = W_SURFACE × exp(-oracle_dist/cell_size × decay)  // Surface proximity
       + W_SPREAD × min_dist_to_prev / cell_size          // Exploration bonus
       + crossing_bonus (if transition found)              // In/out transition
       - LAMBDA                                            // Per-sample cost
```

**Note:** The reward uses oracle distance (analytical ground truth) to shape learning. This is acceptable because:
- The reward is the training signal, not a policy input
- The policy must learn to achieve high reward using only its clean inputs
- Once trained, the policy generalizes without needing oracle access

### Default Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| W_SURFACE | 1.0 | Weight for surface proximity |
| W_SPREAD | 0.3 | Weight for spatial exploration |
| crossing_bonus | 0.5 | Bonus for finding in/out transition |
| LAMBDA | 0.02 | Per-sample cost (encourages efficiency) |
| surface_decay | 5.0 | Exponential decay rate |

## Training

### Algorithm: REINFORCE with Baseline

1. Roll out episode using current policy (stochastic sampling)
2. Compute discounted returns: G_t = r_t + γ × G_{t+1}
3. Compute baseline (average return)
4. Compute advantages: A_t = G_t - baseline
5. Compute policy gradient via BPTT
6. Update with Adam optimizer

### Current Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 6000 |
| Learning rate | 0.001 |
| Discount (γ) | 0.98 |
| Gradient clipping | 1.0 |
| Adam β₁ | 0.9 |
| Adam β₂ | 0.999 |
| Budget (max samples) | 50 |
| Validation offset | 0.2 × cell_size |

### Validation Set

- 52 points on analytical rotated cube
- 12 face samples, 24 edge samples, 16 corner samples (2 per feature)
- Random 3D offset within `0.2 × cell_size` of surface (easier start)
- Deterministic seed for reproducibility

## Results (v2 - 2026-01-26)

### Training Progress (6000 epochs, ~3.5 min)

| Metric | Epoch 1 | Epoch 3000 | Epoch 6000 |
|--------|---------|------------|------------|
| Return | 22.0 | 29.7 | 33.3 |
| Reward/step | 0.44 | 0.59 | 0.67 |
| Gradient norm | 2.0 | 2.4 | 3.3 |
| Samples used | 8 | 5 | 5 |

**Key improvements:**
- Return increased 50% (22 → 33)
- Still learning at 6000 epochs (gradients healthy)
- More sample-efficient (8 → 5 samples)

### Sample Exploration

| Version | Sample Spread per Axis |
|---------|----------------------|
| v1 (before) | ~0.001 (degenerate) |
| v2 (after) | ~0.05 (full cell) |

The policy learned to explore the entire cell volume instead of clustering in a tiny region.

## Changelog

### v2 (2026-01-26) - Major Refactor

**Input features redesigned (10 → 6 dims):**
- Removed oracle hint_normal (was cheating)
- Removed last_reward, spread metric, extra budget features
- Clean minimal design: prev_offset, in_out_sign, budget_remaining, crossings_found

**Validation points improved:**
- Added randomized 3D offsets (not just normal-aligned)
- Reduced offset to 0.2 × cell_size for easier start
- Added both inside and outside surface points

**Training scaled up:**
- 200 → 6000 epochs
- Training time ~3.5 minutes

**Sample cloud format (v2):**
- Added `cell_bounds: Option<BBox>` to `SampleCloudSet`
- UI can now render the sampling cell bounding box

**Results:**
- 50% improvement in training reward
- 50x improvement in sample spread (actual exploration)

### v1 (Initial)

- 10-dim inputs including oracle hint_normal
- 200 epochs
- Samples clustered in tiny region (~0.001 spread)

## File Organization

```
rnn_policy/
├── mod.rs          # Public API, training entry points
├── math.rs         # Vector/matrix operations
├── gru.rs          # GRU forward/backward passes
├── chooser.rs      # Octant weights → position mapping
├── policy.rs       # RnnPolicy struct, episode rollout, input building
├── gradients.rs    # BPTT gradient computation
├── reward.rs       # Reward function
├── training.rs     # Adam optimizer, training loop, evaluation
└── README.md       # This file
```

## Usage

### Train and save model
```bash
cargo run --release --bin sample_cloud_dump -- --rnn-policy train
# Saves to rnn_policy_trained.bin
```

### Dump sample cloud for visualization
```bash
cargo run --release --bin sample_cloud_dump -- --rnn-policy dump --out samples.cbor
# Use --discrete for corner sampling instead of weighted
```

### Visualize with CLI
```bash
cargo run --release -p volumetric_cli -- \
  --sample-cloud samples.cbor \
  --sample-cloud-mode overlay \
  analytical_cube.wasm
```

### Inspect sample cloud
```bash
cargo run --release --bin sample_cloud_inspect -- --file samples.cbor --set 0
```

## Future Work

### Short-term
1. **Better classifier**: Replace oracle-based placeholder with actual geometry fitting from samples
2. **Scale up offset**: Gradually increase validation offset toward full cell_size
3. **More shapes**: Train on cylinder, chamfered box, CSG combinations
4. **Early stopping**: End episodes when sufficient crossings found

### Medium-term
1. **Curriculum learning**: Start with faces, add edges, then corners
2. **Entropy bonus**: Regularization to maintain exploration
3. **Larger capacity**: Try hidden_dim=64 or 128
4. **Attention**: Consider transformer architecture for sample history

### Long-term
1. **Integration**: Replace sampling in actual surface nets algorithm
2. **Multi-resolution**: Adapt to different cell sizes
3. **Real-time inference**: Optimize for production use

## References

- GRU: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
- REINFORCE: Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)
- Adam: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)
