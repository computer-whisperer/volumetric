# Edge Refinement Research (Living)

Updated: 2026-01-26

This document is the living knowledge for sharp edge detection on binary samplers.
It intentionally omits archived details and historical algorithms. If something is
missing, check git history or `PHASE4_ARCHIVE.md`.

---

## Core Problem

Given a point P on or near a geometric edge, determine:
1. Classification: Face vs Edge vs Corner
2. Face normals (1/2/3 as applicable)
3. Edge direction and intersection point (for edges)

This supports:
- Case 1: vertex straddling detection (P is a mesh vertex)
- Case 2: edge crossing detection (P lies along a mesh edge)

---

## Constraints (Non-Negotiable)

- Models are binary samplers, not SDFs. Central-difference gradients are unusable.
- Oracle data is never available to attempts at runtime.
- Sampling budgets must be measured in sampler cache misses.

---

## Verified Findings (Keep)

*Note: Quantitative claims below should be re-verified after fixing scale invariance.*

- Binary fields break gradient-based normals. Probing must be geometric.
- Edge vertices often have degenerate accumulated normals due to transition triangles.
- Sampling from *inside* the surface yields much cleaner plane separation than sampling
  directly on the surface point ("thin shell" problem).
- Crossing-count during surface location is a strong edge signal: 2 crossings correlate
  with edge candidates in archived runs. *(Needs re-verification)*
- RANSAC plane fitting is highly sensitive to inlier thresholds; tight thresholds
  are required to keep face samples pure. *(Threshold values need to scale with cell_size)*

---

## Critiques of Current Methodology (Actionable)

- **[CRITICAL] Scale invariance broken**: Hardcoded absolute thresholds in attempt
  algorithms mean results depend on cell_size. Must fix before any further benchmarking.
- **[CRITICAL] Benchmark methodology was flawed**: Previous results used cell_size=1.0
  and fixed offsets, not representative of real surface nets behavior.
- Over-reliance on a single benchmark shape (rotated cube). Add cylinders and chamfers
  to prevent overfitting to planar edges.
- Per-vertex probe + RANSAC assumes mixed samples can be cleanly separated after the fact;
  most failures are actually caused by the sampling distribution.
- Sampling from the surface point yields coplanar clouds and mixed face populations.
- Lack of neighborhood coherence: edges are global features but are inferred locally.

---

## Attempt Status (Snapshot)

Results at realistic scale (`cell_size=0.05`, proportional offset `0.5 * cell_size`).
All attempts have scale-invariant thresholds (multiplied by cell_size).

| Attempt | Face | Edge | Corner | Face Err | Edge Err | Samples |
|---------|------|------|--------|----------|----------|---------|
| 0 | 12/12 | 2/24 | 0/8 | 44° | 169° | 1443 |
| 1 | 12/12 | 0/24 | 0/8 | 21° | 180° | 808 |
| 2 | 6/12 | 14/24 | 0/8 | 52° | 108° | 1594 |

- **Attempt 0**: Classification works but measurement fails. **Not viable.**
- **Attempt 1**: Best face error but zero edge detection. Already had scale-invariant
  thresholds. **Not viable.**
- **Attempt 2**: After threshold fix, detects edges (14/24) but misclassifies faces.
  Trade-off between face/edge detection. **Not viable.**

All three approaches fail at realistic scale. The fundamental issue is that
local RANSAC plane fitting doesn't work reliably in small sample neighborhoods.

See attempt notes in `src/adaptive_surface_nets_2/stage4/research/ATTEMPT_*.md`.

---

## Living Directions (Near-Term)

1. Use crossing-count or surface-location results to seed two face directions
   instead of sampling from the surface point with a single normal.
2. Introduce neighborhood coherence: edge classifications should align along
   mesh adjacency, not fluctuate per-vertex.
3. Add additional oracle shapes (cylinder, chamfered box) to validate generality.
4. Explore micro-grid Hermite sampling as an alternative to random probing.

---

## Debugging Tools

Sample cloud visualization is available for debugging sampling patterns:

```bash
# Generate sample cloud dump for an attempt
cargo run --bin sample_cloud_dump -- --attempt 0

# Render with CLI (overlay mode shows depth-tested points on mesh)
cargo run -p volumetric_cli -- render \
  -i rotated_cube.wasm/rotated_cube.wasm -o debug.png \
  --sample-cloud sample_cloud_attempt0.cbor \
  --sample-cloud-mode overlay --sample-cloud-set 5
```

See `SAMPLE_CLOUD_DEBUG.md` for full documentation of the sample cloud format,
rendering options, and color coding by sample kind (crossing/inside/outside).

---

## Log (Timestamped)

2026-01-25 16:50 EST - Document trimmed to living knowledge, older experimental
history removed to reduce confusion. Added critiques and clarified near-term
research directions.

2026-01-25 17:05 EST - Hermite micro-grid experiment added (grid edge crossings
with k-means, RANSAC, and edge-aligned k-means plane fits). All variants find
edges but produce high normal/direction errors, indicating mixed face samples
in the local grid are still too ambiguous.

2026-01-25 17:10 EST - Added edge-line RANSAC on micro-grid crossings. Edge
direction error improves (~28°) but normals remain very poor (~98°), suggesting
the crossing cloud is still face-mixed even with line-constrained fitting.

2026-01-25 17:18 EST - Crossing-cone experiment (using crossing directions from
locate_surface and sampling cones around them) performed poorly: ~113° normal
error, ~118° edge direction error, and high sample cost (~1k). Likely still
mixing face populations and/or using unreliable crossing directions.

2026-01-25 17:20 EST - Retired the crossing-cone experiment (code removed) to
reduce clutter; results retained above for reference.

2026-01-25 18:40 EST - Added ML sampling MVP notes in `ML_SAMPLING_APPROACH.md`
and sample cloud debug notes in `SAMPLE_CLOUD_DEBUG.md` for ongoing work.

2026-01-25 21:00 EST - **CRITICAL: BENCHMARK METHODOLOGY INVALIDATED**

Investigation into sample cloud visualization revealed that ALL previous
benchmark results were obtained under incorrect scaling conditions. Two
compounding issues were discovered:

### Issue 1: Hardcoded cell_size = 1.0

The research benchmarks and sample cloud dumps passed `cell_size = 1.0` to
all attempt algorithms. For a unit cube (side 1.0), this meant:
- `search_distance = 0.5 * 1.0 = 0.5` (half the entire model!)
- `probe_epsilon = 0.1 * 1.0 = 0.1` (10% of the entire model!)

Sample clouds spanned the entire cube instead of a small cell neighborhood.
The algorithms appeared to work because they over-sampled everything.

### Issue 2: Fixed validation point offsets

The validation points used a fixed offset of `0.1` units inside the surface,
regardless of cell_size. In real surface nets, vertices are within ~1 cell of
the actual surface. The offset should scale with cell_size (e.g., `0.5 * cell_size`).

### Issue 3: Hardcoded absolute thresholds (SCALE INVARIANCE BROKEN)

Testing revealed the algorithms are NOT scale-invariant. With proportional
offsets (`0.5 * cell_size`), results differed dramatically:

| cell_size | offset | Face Error | Edge Success |
|-----------|--------|------------|--------------|
| 1.0       | 0.5    | 1.85°      | 6/24         |
| 0.2       | 0.1    | 44.66°     | 6/24         |

Root cause: Multiple hardcoded absolute thresholds in `attempt_0.rs`:
```rust
let tight_threshold = 0.006;              // Lines 573, 688
const FACE_RESIDUAL_THRESHOLD: f64 = 0.002;   // Line 900
let outlier_threshold = 0.01;             // Line 922
const EDGE_RESIDUAL_THRESHOLD: f64 = 0.02;    // Line 975
const CORNER_RESIDUAL_THRESHOLD: f64 = 0.005; // Line 1000
```

These thresholds are in world units, not relative to cell_size. When
cell_size shrinks, these thresholds become relatively larger, breaking the
algorithm's assumptions about plane fitting tolerance.

### Implications

**All previous benchmark results in this document and attempt notes are
suspect.** The "good" face detection (~1.68° error) was achieved only with
the oversized cell_size=1.0. The actual algorithm behavior with realistic
cell sizes is much worse.

To fix:
1. All thresholds must be made relative to cell_size
2. Validation point offsets must scale with cell_size
3. Benchmarks must be re-run after fixes to establish true baselines

See `attempt_runner.rs:RESEARCH_CELL_SIZE` and the `generate_validation_points_with_offset()` function.

2026-01-26 - **Attempt 0 re-tested at realistic scale**

Set `RESEARCH_CELL_SIZE = 0.05` (5% of cube edge) and made all thresholds in
`attempt_0.rs` scale with cell_size:
- `tight_threshold = 0.006 * cell_size`
- `face_residual_threshold = 0.002 * cell_size`
- `outlier_threshold = 0.01 * cell_size`
- `edge_residual_threshold = 0.02 * cell_size`
- `corner_residual_threshold = 0.005 * cell_size`
- `corner_threshold = config.ransac_inlier_threshold * cell_size * 0.6`

Results:
| Metric | cell_size=1.0 | cell_size=0.05 |
|--------|---------------|----------------|
| Face success | 12/12 | 12/12 |
| Face error | 1.85° | 43.94° |
| Edge success | 6/24 | 2/24 |
| Corner success | 0/8 | 0/8 |
| Samples | 1076 | 1443 |

The crossing-count classification still works, but plane fitting fails at
realistic scale. The algorithm was designed and tuned for the unrealistic
cell_size=1.0 setup where sample clouds spanned the entire model. The unit
tests in `attempt_0.rs` still use hardcoded cell_size=1.0 and don't test
realistic behavior.

**Conclusion**: Attempt 0 is not viable. Moving on to re-test attempts 1 and 2.

2026-01-26 - **Attempts 1 and 2 re-tested at realistic scale**

Attempt 1 already had scale-invariant thresholds. Results:
- Face: 12/12, 21° error (best of all attempts)
- Edge: 0/24, 180° error (complete failure)
- Samples: 808

Attempt 2 needed threshold scaling fix (added `* cell_size` to fit_face,
fit_edge, fit_corner). Results after fix:
- Face: 6/12, 52° error (threshold change broke face classification)
- Edge: 14/24, 108° error (improved from 0/24 before fix)
- Corner: 0/8
- Samples: 1594

The threshold scaling in attempt 2 reveals a trade-off: tighter thresholds
help edge detection but cause false edge classifications on faces.

**Conclusion**: All three attempts fail at realistic scale. The RANSAC-based
approach doesn't work reliably in small local neighborhoods. New approaches
needed.

2026-01-26 - **RNN Sampling Policy v2 - Major Progress**

Refactored the RNN-based sampling policy with significant improvements:

### Input Feature Redesign (10 → 6 dims)
Previous inputs included oracle hint_normal (cheating). Replaced with clean
minimal design:
- `prev_offset[3]`: `(prev_sample - vertex) / cell_size`
- `in_out_sign`: +1 inside, -1 outside, 0 first
- `budget_remaining`: `1 - step/BUDGET`
- `crossings_norm`: `crossings_found / 5` (capped)

No oracle information in inputs. Reward still uses oracle distance for training
signal (acceptable - shapes learning, not used at inference).

### Validation Point Improvements
- Random 3D offsets within cell (not just normal-aligned)
- Reduced offset to `0.2 × cell_size` for easier start
- Deterministic seeding for reproducibility

### Training Results (6000 epochs, ~3.5 min)

| Metric | Before | After |
|--------|--------|-------|
| Return | 22.0 | 33.3 |
| Reward/step | 0.44 | 0.67 |
| Sample spread | ~0.001 | ~0.05 |

**Key achievement**: Policy learned to explore the full cell volume instead of
clustering in a tiny region (50x improvement in sample spread).

### Sample Cloud Format v2
Added `cell_bounds: Option<BBox>` to `SampleCloudSet` for UI rendering of the
sampling cell bounding box.

See `src/adaptive_surface_nets_2/stage4/research/experiments/rnn_policy/README.md`
for full documentation.

**Next steps**: Scale up validation offset toward full cell_size, add more
shapes, replace placeholder classifier with actual geometry fitting.
