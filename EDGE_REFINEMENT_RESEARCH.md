# Edge Refinement Research (Living)

Updated: 2026-01-25 21:00 EST

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

**WARNING**: Previous assessments below were made with incorrect scaling (cell_size=1.0,
fixed offset=0.1). These need re-evaluation after fixing scale invariance issues.
See Log entry 2026-01-25 21:00 EST for details.

- Attempt 0: ~~strong classification~~, unstable second normal, high sampling cost.
  *Re-test needed with realistic cell_size.*
- Attempt 1: adaptive RANSAC collapses onto single face; false edges on corners.
  *Re-test needed.*
- Attempt 2: fixed-budget RANSAC improves faces/corners but fails edges due to mixed samples.
  *Re-test needed.*

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
