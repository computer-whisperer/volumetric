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
Input[5] → GRU[64] → Linear[8] → Softmax → Weighted Position
                  ↓
            Classifier Heads → Face/Edge/Corner predictions
```

#### GRU Updater
- **Input dimension**: 5 features (clean, no oracle cheating)
- **Hidden dimension**: 64
- **Output**: Updated hidden state

The GRU was chosen over LSTM because:
- Simpler (fewer parameters)
- Sufficient for short episodes (~50 steps)
- Faster to train

#### Chooser Head
- **Input**: 32-dimensional hidden state
- **Output**: 8 weights (one per octant corner)
- **Position calculation**: Weighted average of cube corners via softmax

### Input Features (v5 - Minimal Design)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0-2 | prev_offset | ~[-1, 1] | `(prev_sample - vertex) / cell_size` |
| 3 | in_out_sign | {-1, 0, 1} | Previous sample: inside=+1, outside=-1, first=0 |
| 4 | budget_remaining | [0, 1] | `1 - step_idx / BUDGET` |

**Key design decisions:**
- No oracle information in inputs (previous version cheated with hint_normal)
- All spatial values normalized by cell_size
- Minimal feature set - only information available at inference time
- Removed `crossings_found` (v5) - model should learn from spatial patterns, not counts

### Parameter Count

| Component | Parameters |
|-----------|------------|
| GRU W_z, W_r, W_h | 3 × (64 × 5) = 960 |
| GRU U_z, U_r, U_h | 3 × (64 × 64) = 12,288 |
| GRU b_z, b_r, b_h | 3 × 64 = 192 |
| Chooser W | 8 × 64 = 512 |
| Chooser b | 8 |
| Face head | 5 × 64 + 5 = 325 |
| Edge head | 12 × 64 + 12 = 780 |
| Corner head | 13 × 64 + 13 = 845 |
| **Total** | **15,910** |

#### Classifier Head Outputs

| Head | Outputs | Description |
|------|---------|-------------|
| Face | 5 | confidence + normal(3) + offset(1) |
| Edge | 12 | confidence + normal_a(3) + offset_a + normal_b(3) + offset_b + direction(3) |
| Corner | 13 | confidence + [normal(3) + offset] × 3 |

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

### Parallel Training

When the `native` feature is enabled, episode collection and evaluation are
parallelized using rayon. This provides ~2.5x speedup on multi-core systems:

```bash
# Build with parallel training
cargo build --release --features native

# Training time: ~1.5 minutes (vs ~4 minutes sequential)
cargo run --release --features native --bin sample_cloud_dump -- --rnn-policy train
```

Each parallel worker uses a deterministic RNG seeded by `(epoch * num_points + point_idx)`,
ensuring reproducible results across runs.

### Current Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 24000 |
| Learning rate | 0.0003 |
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

## Results (v7 - 2026-01-27)

### Neural Classifier Accuracy (with Asymmetric Margin Loss)

| Geometry | Type Accuracy | Confidence | Normal Loss |
|----------|---------------|------------|-------------|
| Face | 42% (5/12) | 0.241 | 0.496 |
| Edge | 29% (7/24) | 0.255 | 0.274 |
| Corner | **94%** (15/16) | 0.352 | 0.269 |
| **Overall** | **51.9%** | - | - |

### Key Findings

1. **Corner classification solved**: 94% accuracy (up from 0%!)
2. **Balanced confidence**: All heads now have similar confidence levels (0.24-0.35)
3. **Trade-off with edges**: Edge accuracy dropped to 29% (from 92% in v6)
4. **Face improved**: 42% (up from 25% in v6)

### What Fixed Corners

The corner confidence problem was solved with two changes:
1. **Asymmetric margin loss**: Only applied when ground truth is corner
   - Requires corner confidence to exceed other heads by margin=0.1
   - Doesn't hurt other classes when they're correct
2. **Higher wrong confidence penalty**: w_wrong_conf=4.0 (up from 2.0)
   - Penalizes overconfident wrong predictions
   - Prevents any single head from dominating

### Remaining Challenge: Edge Classification

Edges are now the weakest class (29%). The model sometimes misclassifies edges as:
- Corners (when corner confidence is boosted by margin loss)
- Faces (simpler geometry is sometimes preferred)

This suggests the edge/corner distinction needs more work, but the fundamental
corner confidence calibration problem is solved.

### Previous Results (v4 - RANSAC-based)

| Geometry | Before Training | After Training |
|----------|-----------------|----------------|
| Face | 0% (0/12) | **41.7%** (5/12) |
| Edge | 0% (0/24) | 0% (0/24) |
| Corner | 0% (0/16) | 0% (0/16) |
| **Overall** | 0% | **9.6%** |

## Changelog

### v7 (2026-01-27) - Margin Loss & Confidence Calibration Fix

**Problem solved: Corner confidence calibration**
- Corners now correctly classified at 94% (was 0%)
- Overall accuracy improved to 51.9%

**Asymmetric margin loss (only for corners):**
- Added hinge-style margin loss: `max(0, margin - (correct_conf - wrong_conf))`
- Only activated when ground truth is Corner (not Face/Edge)
- Parameters: w_margin=3.0, margin=0.1
- This pushes corner confidence above other heads without destabilizing other classes

**Increased wrong confidence penalty:**
- w_wrong_conf increased from 2.0 to 4.0
- Penalizes heads for being confident when they're wrong
- Prevents any single head from dominating

**Loss configuration (final):**
```rust
w_correct_conf: 1.0,
w_normal: 5.0,
w_offset: 2.0,
w_wrong_conf: 4.0,  // increased
w_margin: 3.0,      // new
margin: 0.1,        // new
class_weights: [1.0, 1.0, 1.0],  // equal
```

**Results progression:**
| Version | Face | Edge | Corner | Overall |
|---------|------|------|--------|---------|
| v6 (baseline) | 25% | 92% | 0% | 48% |
| v7 + symmetric margin | 67% | 0% | 75% | 27% |
| v7 + asymmetric margin | 42% | 29% | 94% | **52%** |

**Key insight:**
The corner problem was not about model capacity or architecture - it was about
the training objective. Independent sigmoid confidences don't compete directly.
The margin loss creates the necessary competition for corners to win.

### v6 (2026-01-27) - Model Scale-Up Experiment

**Capacity increase:**
- Scaled HIDDEN_DIM from 32 to 64 (3.2x more parameters: 4,902 → 15,910)
- Extended training from 6k to 24k epochs
- Reduced learning rate from 0.001 to 0.0003 for stability

**Training observations:**
- With lr=0.001: unstable, high gradient norms
- With lr=0.0003: smooth descent, loss reached 3.683 at epoch 23k (best)
- Training still improving at end, suggesting even longer training could help

**Results:**
- Edge accuracy improved: 83% → 92%
- Face accuracy regressed: 50% → 25%
- Corner accuracy unchanged: 0%
- Corner confidence improved (0.251 → 0.330) but still loses to edge (0.532)

**Key insight:**
More capacity helps edge classification but doesn't solve the corner confidence problem.
The model learns to be a very confident edge classifier, making it harder for corners to win.
This is not a capacity problem - it's a training objective/architecture problem.

### v5 (2026-01-27) - Neural Classifier Heads & Offset Prediction

**Major architecture change: Neural classifier heads**
- Replaced RANSAC-based classification with learned neural heads
- Three parallel heads from GRU hidden state: Face, Edge, Corner
- Each head predicts: confidence + normals + offsets
- Training uses cross-entropy for type classification + cosine loss for normals + L1 for offsets

**Classifier head outputs:**
- Face: confidence + 1 normal + 1 offset (5 outputs)
- Edge: confidence + 2 normals + 2 offsets + direction (12 outputs)
- Corner: confidence + 3 normals + 3 offsets (13 outputs)

**Offset prediction:**
- Offset = signed distance from vertex to plane, normalized by cell_size
- Computed as: `(dot(vertex, normal) - 0.5) / cell_size`
- Enables direct surface position prediction without RANSAC fitting

**Input simplification:**
- Removed `crossings_found` from inputs (INPUT_DIM: 6 → 5)
- Model should learn from spatial patterns, not explicit crossing counts

**Order-invariant loss:**
- Edge normals: best matching of 2 predicted to 2 expected
- Corner normals: greedy assignment of 3 predicted to 3 expected
- Offset loss uses same assignment as normal loss

**Results:**
- Edge classification: 83% accuracy (major improvement over RANSAC's 0%)
- Face classification: 50% accuracy
- Corner classification: 0% (confidence calibration problem)
- Corner normal quality is actually BEST (0.255 loss) but confidence always loses

**Current investigation:**
- Why does corner confidence (~0.25) always lose to edge confidence (~0.45)?
- Possible causes: class imbalance (16 corners vs 24 edges), architecture bias
- The model predicts good corner normals but refuses to classify as corner

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
├── mod.rs              # Public API, training entry points, sweep experiments
├── math.rs             # Vector/matrix operations
├── gru.rs              # GRU forward/backward passes
├── chooser.rs          # Octant weights → position mapping
├── policy.rs           # RnnPolicy struct, episode rollout, input building
├── gradients.rs        # BPTT gradient computation
├── reward.rs           # Per-step + terminal reward functions
├── classifier.rs       # RANSAC classifiers for face/edge/corner fitting (legacy)
├── classifier_heads.rs # Neural classifier heads (Face/Edge/Corner prediction)
├── training.rs         # Adam optimizer, training loop, evaluation
└── README.md           # This file
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

### Immediate (Edge Classification Problem)
Corner problem solved (94% accuracy). Now edges are the weakest class (29%).

1. **Edge-specific improvements**:
   - Add asymmetric margin for edges (similar to what fixed corners)
   - Try edge-specific class weight increase
   - Investigate edge→corner misclassification patterns
2. **Hierarchical classification**:
   - First: Face vs Multi-plane (Edge/Corner)
   - Second: Edge vs Corner
   - May help disambiguate 2-plane vs 3-plane geometry
3. **Feature engineering**:
   - Add angle between predicted normals as auxiliary feature
   - Edge geometry: normals should be ~90° apart
   - Corner geometry: normals should span 3D

### Short-term
1. **Scale up offset**: Gradually increase validation offset toward full cell_size
2. **More shapes**: Train on cylinder, chamfered box, CSG combinations
3. **Longer training**: Loss still improving at 24k epochs

### Solved
- ✅ **Corner confidence problem** (v7): Asymmetric margin loss + higher wrong_conf penalty
- ✅ Larger capacity (hidden_dim=64): Now 15,910 parameters
- ✅ Lower learning rate (0.0003): Stable training
- ✅ Longer training (24k epochs): Continued improvement

### Long-term
1. **Integration**: Replace sampling in actual surface nets algorithm
2. **Multi-resolution**: Adapt to different cell sizes
3. **Real-time inference**: Optimize for production use

## References

- GRU: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
- REINFORCE: Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)
- Adam: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)
