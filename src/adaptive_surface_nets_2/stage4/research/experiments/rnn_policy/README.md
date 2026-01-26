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

### Per-Step Reward

Each sampling step receives a reward:

```
reward = W_SURFACE × exp(-oracle_dist/cell_size × decay)  // Surface proximity
       + W_SPREAD × min_dist_to_prev / cell_size          // Exploration bonus
       + crossing_bonus (if transition found)              // In/out transition
       - LAMBDA                                            // Per-sample cost
```

### Terminal Reward (Classifier-Based)

At episode end, samples are evaluated by running the appropriate RANSAC classifier
based on the oracle label (face/edge/corner):

```
terminal_reward = W_FIT_SUCCESS (if fit succeeded)
                + W_RESIDUAL × (1 - normalized_residual)
                + W_NORMAL_ACCURACY × (1 - normalized_normal_error)
                + [edge only] W_EDGE_DIRECTION × (1 - normalized_direction_error)
                + [corner only] completeness_bonus (if 3 planes found)
```

The terminal reward encourages the policy to collect samples that:
1. Allow successful geometry fitting (face/edge/corner detection)
2. Produce low residuals (samples tightly fit the expected geometry)
3. Yield accurate normals relative to oracle ground truth

**Training flow:**
1. Policy collects samples during episode (only clean inputs used)
2. At episode end, oracle label determines which classifier to run
3. Classifier fit quality contributes to terminal reward
4. Policy learns to sample in ways that produce good classifier fits

### Default Parameters (v4)

**Per-step:** (reduced to let terminal reward dominate)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| W_SURFACE | 0.1 | Weight for surface proximity (reduced) |
| W_SPREAD | 0.05 | Weight for spatial exploration (reduced) |
| crossing_bonus | 0.0 | **Disabled** - was causing reward hacking |
| LAMBDA | 0.01 | Per-sample cost |
| surface_decay | 5.0 | Exponential decay rate |

**Terminal:** (boosted to be dominant factor)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| W_FIT_SUCCESS | 1.0 | Small bonus for successful fit |
| W_RESIDUAL | 5.0 | Weight for residual quality |
| W_NORMAL_ACCURACY | 20.0 | **Dominant**: accuracy vs oracle normals |
| W_EDGE_DIRECTION | 5.0 | Weight for edge direction accuracy |
| max_residual | 0.1 | Max expected residual (relative to cell_size) |
| max_normal_error | 45.0 | Max expected normal error (degrees) |

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

## Results (v4 - 2026-01-26)

### Classification Fit Rates by Geometry Type

| Geometry | Before Training | After Training |
|----------|-----------------|----------------|
| Face | 0% (0/12) | **41.7%** (5/12) |
| Edge | 0% (0/24) | 0% (0/24) |
| Corner | 0% (0/16) | 0% (0/16) |
| **Overall** | 0% | **9.6%** |

### Fit Quality (when fits succeed)

| Metric | Value |
|--------|-------|
| Normal accuracy reward | 1.556 (of max 20) |
| Residual reward | ~0.3 (of max 5) |
| Avg crossings/episode | 2.2 |
| Avg surface points | 67 (face), 38 (edge), 43 (corner) |

### Key Findings

1. **Faces work well**: 42% fit rate with reasonable normal accuracy
2. **Edges/corners fail**: 0% fit rate despite having 38-43 outside samples
3. **Root cause identified**: Outside samples cluster on ONE face, not spread across multiple faces needed for edge/corner fitting

### Sample Exploration (improved from v2)

| Version | Sample Spread per Axis |
|---------|----------------------|
| v1 | ~0.001 (degenerate) |
| v2 | ~0.05 (full cell) |
| v3 | ~0.003 (reward hacking - narrow strip) |
| v4 | ~0.04 (recovered after fixing rewards) |

## Changelog

### v4 (2026-01-26) - Reward Rebalancing & Edge/Corner Investigation

**Problem identified: Reward hacking**
- Policy learned to alternate samples across surface in a narrow strip
- This maximized crossing bonus while producing degenerate geometry
- Samples had ~0.003 spread on one axis (collapsed to a line)

**Reward structure overhauled:**
- **Disabled** crossing_bonus (was being exploited)
- **Reduced** per-step rewards (w_surface: 1.0→0.1, w_spread: 0.3→0.05)
- **Boosted** terminal reward to dominate (w_normal_accuracy: 3→20)

**Evaluation metric fixed:**
- Removed broken RANSAC discrimination (tried to classify face/edge/corner from samples alone)
- Now uses oracle-selected classifier and reports fit quality metrics
- Tracks: fit_rate, normal_accuracy, residual_quality per geometry type

**Edge/corner classification approach:**
- For faces: use midpoints of inside/outside pairs (works well)
- For edges/corners: use raw outside samples directly (they lie on actual face planes)
- Relaxed RANSAC thresholds: inlier 10%→20%, parallel check 0.95→0.85

**Current status:**
- Faces: 42% fit rate, good normal accuracy ✓
- Edges: 0% fit rate - samples cluster on one face, not both
- Corners: 0% fit rate - same issue, samples don't span all three faces

**Next steps to investigate:**
- Why do outside samples cluster on one face instead of spreading across adjacent faces?
- May need geometry-aware exploration reward or curriculum learning

### v3 (2026-01-26) - Classifier-Based Terminal Reward

**Added classifier module (`classifier.rs`):**
- `fit_face_from_samples()` - RANSAC single-plane fit
- `fit_edge_from_samples()` - RANSAC two-plane fit
- `fit_corner_from_samples()` - RANSAC three-plane fit
- Error metrics: `normal_error_degrees()`, `best_edge_normal_errors()`, `best_corner_normal_errors()`

**Terminal reward integration:**
- `TerminalRewardConfig` for classifier-based reward weights
- `compute_terminal_reward()` evaluates samples against oracle label
- Episode struct now tracks `inside_flags` and `terminal_reward`
- Training loop computes terminal reward and adds to last step

**Training configuration:**
- New `use_terminal_reward` flag (default: true)
- New `terminal_reward` config block

This enables the policy to learn sampling strategies that produce good
classifier fits, not just samples near the surface.

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
├── reward.rs       # Per-step + terminal reward functions
├── classifier.rs   # RANSAC classifiers for face/edge/corner fitting
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

### Immediate (Edge/Corner Classification)
1. **Investigate sample clustering**: Why do outside samples cluster on one face?
   - Visualize sample distributions for edge/corner vertices
   - Check if policy is biased toward certain octants
2. **Geometry-aware exploration**: Add reward for sampling on multiple faces
   - Could use dot product of sample directions to encourage diversity
   - Or partial credit for finding samples on different faces
3. **Curriculum learning**: Train on faces first, then gradually add edges/corners

### Short-term
1. **Scale up offset**: Gradually increase validation offset toward full cell_size
2. **More shapes**: Train on cylinder, chamfered box, CSG combinations
3. **Early stopping**: End episodes when sufficient quality achieved

### Medium-term
1. **Entropy bonus**: Regularization to maintain exploration
2. **Larger capacity**: Try hidden_dim=64 or 128
3. **Attention**: Consider transformer architecture for sample history

### Long-term
1. **Integration**: Replace sampling in actual surface nets algorithm
2. **Multi-resolution**: Adapt to different cell sizes
3. **Real-time inference**: Optimize for production use

## References

- GRU: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
- REINFORCE: Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)
- Adam: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)
