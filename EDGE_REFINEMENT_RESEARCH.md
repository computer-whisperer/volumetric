# Edge Refinement Algorithm Research

## Problem Statement

Given a point P known to be on or near a sharp geometric edge, determine:
1. **Is there a sharp edge?** (two-face vs single-face classification)
2. **What are the two face normals?** (NormA, NormB)
3. **What is the edge direction?** (cross product of normals)
4. **Where exactly is the edge?** (intersection point)

This algorithm serves both:
- **Case 1**: Vertex straddling detection (P is a mesh vertex)
- **Case 2**: Edge crossing detection (P is interpolated along a mesh edge)

---

## Current Best Results

| Method | TPR | FPR | NormA Error | NormB Error | Samples |
|--------|-----|-----|-------------|-------------|---------|
| **Standard (production)** | 100% | 0% | 4.2° | 8.5° | 73 |
| Bisection | 100% | 0% | 3.7° | 7.5° | 457 |
| Multiradius | 100% | 0% | 3.0° | 8.1° | 912 |
| Boundary Bisection | 63.6% | 11.2% | 36.6° | 23.1° | 34 |
| Max Uncertainty | 95.5% | 53.0% | 34.6° | 30.1° | 106 |
| Adaptive Grid | 86.4% | 35.2% | 31.8° | 28.7° | 95 |

**Test geometry:** Rotated cube (rx=35.264°, ry=45°, rz=0°), max-depth 2

---

## Core Algorithm (Production)

```
Input: surface_pos, initial_normal, probe_epsilon, search_distance

1. SETUP
   - Normalize initial_normal → n
   - Compute tangent basis (t1, t2) perpendicular to n

2. PHASE 1: Cardinal Probes (4 probes)
   - Probe at ±t1, ±t2 directions
   - For each: offset by probe_epsilon, search along n for surface
   - EARLY EXIT: If residual < threshold AND normal agrees → smooth surface

3. PHASE 2: Additional Probes (8 more = 12 total)
   - Probe at 30° intervals (excluding cardinals)
   - Check residual for edge escalation

4. PHASE 3: Edge Escalation (12 more = 24 total)
   - If residual suggests edge, add golden-ratio distributed probes
   - Cluster points into two groups
   - Fit planes to each cluster

5. OUTPUT
   - If sharp: return (normal_a, normal_b, edge_direction)
   - If smooth: return blended normal
```

### Key Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| probe_epsilon | 0.1 × cell_size | Distance to probe from vertex |
| search_distance | 1.0 × cell_size | How far to search for surface |
| binary_search_iterations | 12 | Surface point precision |
| angle_threshold | 30° | Minimum for sharp edge |

---

## Identified Issues

### 1. NormB Asymmetry — INVESTIGATION COMPLETE, ROOT CAUSE UNKNOWN

All methods show ~2x higher error on NormB (7-8°) vs NormA (3-4°).

**Key Finding:** The error distribution is **bimodal** — one normal is nearly perfect (~0°), the other is poor (~10°). Which normal is "A" vs "B" depends on arbitrary clustering assignment.

**Empirical Evidence (22 sharp edge vertices):**
```
A more accurate: 15 vertices (A=0.5°, B=11.8°)
B more accurate: 7 vertices (B=1.5°, A=12.3°)

Sample data (sizes, errors):
  #1: (7,6) A=0.1°, B=0.1°  — BOTH accurate (rare case)
  #3: (7,6) A=0.1°, B=17.9° — A perfect, B poor
  #5: (14,11) A=5.0°, B=0.1° — A poor, B perfect (despite A larger)
  #7: (16,9) A=12.2°, B=0.2° — A poor, B perfect (despite A much larger)
```

**Hypotheses Tested and REJECTED:**

1. **Anchor point bias (`surface_pos` in cluster_a):** Removing it from cluster_a had no effect.
2. **Adding anchor to cluster_b:** Made things WORSE (NormB: 8.5° → 13.6°) because `surface_pos` is on the edge, not on face B's plane.
3. **Search direction alignment:** No correlation (10 vs 12 split, essentially random).
4. **Cluster size:** Cases #5, #7 show larger cluster can have WORSE accuracy.
5. **More probes (36 total):** Mixed results (NormA worse, NormB slightly better).

**Remaining Hypotheses (Untested):**
1. Edge geometry interaction with discrete probe pattern
2. Numerical precision in SVD plane fitting for certain orientations
3. Probe epsilon relative to edge position

**Note:** The reference method (258 probes) achieves balanced errors (5.6° vs 5.2°), suggesting sufficient probe density overcomes whatever causes the asymmetry.

### 2. Sample-by-Sample Accuracy Gap
The experimental algorithms (34-106 samples) have 30-37° error vs 4.2° for production.

**Root causes identified:**
1. **Binary search iterations:** 6 vs 12 (64x precision loss)
2. **Sparse surface points:** 12 max crossings vs 24+ probes
3. **Same clustering thresholds** for sparse/dense data

### 3. Probe Epsilon vs Edge Distance
Fixed probe_epsilon may be suboptimal:
- Too large: probes cross to other face
- Too small: insufficient face coverage

**Unexplored:** Adaptive epsilon based on detected edge proximity.

---

## Experimental Algorithms Summary

### Bisection (457 samples, 3.7° NormA)
- Initial 24-probe clustering
- Binary search angular transitions between faces
- Re-sample within detected face regions

**Strength:** Best vs-reference metrics
**Weakness:** High sample count, relies on initial clustering quality

### Multiradius (912 samples, 3.0° NormA)
- Probe at 3 radii × 16 angles
- Check radius consistency for face assignment
- Re-fit using only consistent probes

**Strength:** Best NormA accuracy
**Weakness:** Very high sample count, NormB still 8.1°

### Boundary Bisection (34 samples, 36.6° NormA)
- Bootstrap with 9 samples
- Angular scan for inside/outside transitions
- Binary search transition angles

**Strength:** Lowest sample count
**Weakness:** Poor accuracy, 63.6% TPR

### Max Uncertainty (106 samples, 34.6° NormA)
- Polar belief grid (16 angular × 3 radial)
- Sample at P(inside) ≈ 0.5 locations
- Neighbor smoothing propagation

**Strength:** Information-theoretic sampling
**Weakness:** 53% FPR from smoothing bias

### Adaptive Grid (95 samples, 31.8° NormA)
- Coarse 4×2 polar grid
- Refine only "mixed" cells (both inside/outside)
- Up to 3 refinement levels

**Strength:** Focuses samples on boundary
**Weakness:** 35% FPR, cell containment edge cases

---

## Infrastructure

### SampleCache
```rust
struct CachedSample {
    position: (f64, f64, f64),
    is_inside: bool,
}

struct SampleCache {
    samples: Vec<CachedSample>,
}
```

Tracks all boolean samples for sample-by-sample algorithms.

### AnalyticalRotatedCube
Computes ground truth normals for rotated cube via Euler rotation matrix.
Enables validation beyond probe-based reference (which has ~5.6° inherent error).

### Clustering (cluster_points_two)
Tries 4 splitting strategies:
1. Split by fitted plane normal
2. Split by cardinal axes (X, Y, Z)
3. Split by direction of maximum spread (PCA)
4. Split by cross-product of displacement vectors

Selects partition with lowest combined residual.

---

## Research Directions

### A. Fix Sample-by-Sample Accuracy

**Hypothesis:** Increasing binary search iterations from 6 to 12 will halve error.

**Test:**
1. Modify `find_surface_crossing_quick` to use 12 iterations
2. Re-run diagnostics
3. Measure accuracy/sample tradeoff

**Expected:** 30-37° → ~15-20° error

### B. Adaptive Clustering Thresholds

**Hypothesis:** Sparse data needs stricter edge detection criteria.

**Current thresholds (same for all):**
- improvement > 2.0
- angle_between > 30°
- cluster_fits < 2.0 × base_threshold

**Test:** Scale thresholds by `sqrt(num_points / 24)` to account for noise.

### C. ~~Fix Anchor Point Bias~~ TESTED - FAILED

**Status:** Hypothesis tested and REJECTED.

**Original hypothesis:** `surface_pos` always goes to cluster_a, giving it an anchor point that cluster_b lacks.

**Tests performed:**
1. Remove `surface_pos` from cluster_a → No improvement
2. Add `surface_pos` to cluster_b → Made things WORSE (NormB: 8.5° → 13.6°)

**Why it failed:** `surface_pos` is on the EDGE (intersection of two planes), not on either face's plane. Adding an edge point to a face's cluster corrupts the plane fit.

**Lesson learned:** The asymmetry is NOT caused by the anchor point.

### D. NormB Refinement Pass

**Hypothesis:** After detecting edge, re-probe specifically for weaker face.

**Algorithm:**
1. Detect edge, get initial NormA, NormB
2. Identify which normal has fewer supporting points
3. Probe in directions perpendicular to that normal
4. Re-fit weaker face with additional points

### D. Hybrid Detection + Refinement

**Hypothesis:** Use cheap detection, expensive refinement only when needed.

**Algorithm:**
1. Sample-by-sample detection (30-50 samples)
2. If edge detected with confidence > threshold:
   - Switch to full 24-probe refinement for normals
3. If smooth with confidence > threshold:
   - Return single-plane normal

**Expected:** ~40-60 samples for smooth, ~70-90 for edges (vs 73 uniform)

### E. Multi-Epsilon Probing

**Hypothesis:** Probing at multiple radii improves edge localization.

**Algorithm:**
1. Probe at 0.5×, 0.75×, 1.0× epsilon
2. If all radii agree on face: confident assignment
3. If radii disagree: edge crosses that direction
4. Use smallest agreeing radius for normal fitting

**Note:** Similar to multiradius but with smarter radius selection.

### F. Search Direction Per-Face

**Hypothesis:** Searching along initial blended normal biases toward one face.

**Algorithm:**
1. Initial detection gives candidate NormA, NormB
2. For face A probes: search along NormA (not blended n)
3. For face B probes: search along NormB
4. Re-cluster with face-specific samples

**Risk:** Error in initial normals propagates to search directions.

---

## Test Protocol

### Rotated Cube Diagnostic
```bash
cargo run -p volumetric_cli --release --features volumetric/edge-diagnostic -- \
  mesh -i rotated_cube.wasm/rotated_cube.wasm -o /tmp/test.stl \
  --sharp-edges --max-depth 2
```

### Visual Validation
```bash
cargo run -p volumetric_cli --release -- render \
  -i rotated_cube.wasm/rotated_cube.wasm -o edge_test.png \
  --sharp-edges --recalc-normals --max-depth 2 \
  --camera-pos 1.5,0.8,3 --camera-target 0.6,0.1,0.3 \
  --projection ortho --ortho-scale 0.8 \
  --width 1024 --height 1024 --grid 0
```

### Metrics
1. **NormA/NormB error** vs analytical ground truth (primary)
2. **TPR/FPR** for edge detection
3. **Samples per vertex** (efficiency)
4. **Visual quality** (scalloping, edge straightness)

---

## Experimental Results Log

### 2026-01-21: NormB Asymmetry Investigation

**Objective:** Understand why NormB consistently has ~2x higher error than NormA.

**Method:**
1. Added `cluster_sizes` field to `VertexSharpInfo` struct for tracking
2. Added search direction correlation tracking (dot product of initial normal with each face)
3. Tested multiple fix hypotheses experimentally

**Initial Observations:**

| Metric | Value |
|--------|-------|
| Total sharp vertices | 22 |
| A more accurate | 15 (68%) |
| B more accurate | 7 (32%) |
| Larger cluster more accurate | 16 (73%) |
| Smaller cluster more accurate | 6 (27%) |
| Search dir closer to better | 10 (45%) |
| Search dir closer to worse | 12 (55%) |

**Bimodal Error Distribution:**
- When A is better: A = 0.5°, B = 11.8°
- When B is better: B = 1.5°, A = 12.3°
- One normal is nearly perfect, one is poor

**Experiments Conducted:**

1. **Remove `surface_pos` from cluster_a** (make clusters equal)
   - Result: No change (NormA=4.1°, NormB=8.5°)

2. **Add `surface_pos` to cluster_b** (give both an anchor)
   - Result: WORSE (NormB: 8.5° → 13.6°)
   - Reason: `surface_pos` is on the EDGE, not on face B's plane

3. **Add 12 more probes (36 total) after edge detection**
   - Result: Mixed (NormA: 4.2° → 5.0° worse, NormB: 8.5° → 8.2° slightly better)

**Conclusion:** Root cause remains unknown. The asymmetry is NOT caused by anchor point, search direction, or cluster size. More investigation needed.

---

## Next Steps

### Immediate
1. [ ] Increase `find_surface_crossing_quick` iterations: 6 → 12 (for sample-by-sample methods)
2. [ ] Implement adaptive clustering thresholds for sparse data
3. [ ] Test hybrid detection + refinement approach

### Short-term
4. [ ] Investigate NormB asymmetry further:
   - Test with different edge orientations
   - Analyze SVD numerical stability for different point configurations
   - Check if issue is specific to tangent-plane probe distribution
5. [ ] Design unified algorithm for Case 1 and Case 2

### Medium-term
6. [ ] Optimize for common case (smooth surfaces)
7. [ ] Test on diverse geometry (spheres, cylinders, CSG)

---

## Open Questions

1. **Why does the reference (258 probes) have 5.6° error?**
   - Is this a fundamental limit of probe-based estimation on boolean fields?
   - Could continuous SDF improve this?
   - Note: Reference has balanced NormA/NormB (5.6°/5.2°), suggesting sufficient probes overcomes the asymmetry

2. **Why is one normal consistently much better than the other?** (PARTIALLY INVESTIGATED)
   - NOT caused by: anchor point, search direction, cluster size
   - Bimodal distribution: one normal ~0°, one ~10°
   - More probes (258 vs 24) eliminates the asymmetry
   - Likely related to how probe pattern interacts with edge geometry

3. **Is clustering optimal for face assignment?**
   - Could direct geometric reasoning (edge direction estimation) work if done correctly?
   - The failed edge-specific approach may have implementation issues vs fundamental flaws.

3. **What's the theoretical minimum samples needed?**
   - Information-theoretic lower bound for edge detection?
   - For a given accuracy target, what's achievable?

4. **How does edge angle affect difficulty?**
   - 90° edges (cube) vs shallow edges (30°)?
   - Does optimal epsilon vary with edge angle?
