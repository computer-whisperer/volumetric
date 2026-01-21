# Normal Probing Architecture Report

## Executive Summary

The normal probing system estimates surface normals at mesh vertices by sampling nearby surface points and fitting planes. After implementing adaptive multi-phase probing, the system now achieves:

- **9° normal error** (down from 16-18°)
- **100% true positive rate** for sharp edge detection
- **0% false positive rate**
- **73 samples/vertex** (down from ~180)

The key insight from testing is that **clustering-based point assignment consistently outperforms edge-direction-based assignment**, even when the edge direction is estimated from PCA.

---

## Current Algorithm

### Overview

The normal probing system has two variants:
1. **`refine_normal_via_probing`** - Standard normal refinement without edge detection
2. **`refine_normal_via_probing_with_sharp_detection`** - Normal refinement with Case 1 sharp edge detection

Both follow the same basic pattern: probe nearby surface points, fit planes, blend with prior.

### Detailed Flow (Current Adaptive Implementation)

```
Input: surface_pos, initial_normal (from accumulated face normals)

1. SETUP
   - Normalize initial_normal → n
   - Compute tangent basis (t1, t2) perpendicular to n
   - probe_epsilon = 0.1 * min_cell_size
   - search_distance = max_cell_size

2. PHASE 1: Cardinal Probes (4 probes)
   For each direction in [+t1, -t1, +t2, -t2]:
     - probe_pos = surface_pos + dir * probe_epsilon
     - Search along n for surface crossing
     - Binary search to refine crossing position
     - Add to surface_points[]

   EARLY EXIT CHECK:
     - Fit single plane to 4 points
     - If residual < 1e-6 AND angle < 5°: DONE (smooth surface)

3. PHASE 2: Additional Probes (8 more = 12 total)
   For angles [30°, 60°, 120°, 150°, 210°, 240°, 300°, 330°]:
     - dir = t1*cos(θ) + t2*sin(θ)
     - Same probing process as Phase 1

4. PHASE 3: Edge Detection Check
   - Try clustering into two groups
   - Check sharp criteria:
     * improvement > 2.0 (2-plane fit 2x better than 1-plane)
     * angle_between > threshold (default 30°)
   - If POTENTIAL SHARP EDGE detected: escalate to Phase 4
   - If clearly smooth: Bayesian blend and return

5. PHASE 4: High-Precision Probing (12 more = 24 total)
   For 12 golden-ratio distributed directions:
     - angle = i * π * (3 - √5)  // Golden angle
     - dir = t1*cos(angle) + t2*sin(angle)
     - Same probing process

   Re-cluster all 24 points for final edge detection

6. FINAL ANALYSIS
   a. If sharp edge confirmed:
      - Return both normals + projected position on edge
   b. If not sharp:
      - Bayesian blend with prior

Output: refined_normal, [new_position], sharp_info
```

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `probe_epsilon` | 0.1 * cell_size | Distance to probe from vertex |
| `search_distance` | 1.0 * cell_size | How far to search for surface |
| `binary_search_iterations` | 12 | Precision of surface finding |
| `normal_sample_iterations` | 12 | (Same as above) |
| `angle_threshold` | 30° | Minimum angle for sharp edge |
| `prior_sigma` | 1° | Assumed topology normal error |

### Probe Distribution

**Phase 1-2 (12 probes):**
```
        90° (+t2)
         |
   120°  |  60°
     \   |   /
150°  \  |  /  30°
       \ | /
180° ----+---- 0° (+t1)
       / | \
210°  /  |  \  330°
     /   |   \
   240°  |  300°
         |
       270° (-t2)
```
- 4 cardinal: 0°, 90°, 180°, 270°
- 8 additional: 30°, 60°, 120°, 150°, 210°, 240°, 300°, 330°

**Phase 4 (12 additional probes):**
- Golden-ratio distribution: `angle = i * π * (3 - √5)`
- Provides uniform angular coverage that fills gaps in the fixed distribution
- Combined with Phase 1-2: 24 total probes for edge vertices

### Clustering Algorithm

`cluster_points_two()` tries 4 splitting strategies:
1. Split by fitted plane normal
2. Split by cardinal directions (X, Y, Z)
3. Split by direction of maximum point spread
4. Split by cross-product of displacement vectors

Each strategy partitions points by signed distance from a plane through the center point. The partition with lowest combined residual wins.

**Minimum cluster size: 3 points each** (for reliable plane fitting)

---

## Identified Problems (Original)

### Problem 1: Insufficient Probe Count ✅ ADDRESSED
- **12 probes** vs **32 probes** in reference
- When split into 2 clusters: ~6 points each
- Plane fitting with 6 points is noisy
- **Original result: 16-18° normal error**
- **Fix:** Escalate to 24 probes for edge vertices → 12 points per cluster

### Problem 2: Fixed Probe Distribution ✅ ADDRESSED
- Cardinal + 30° spacing leaves angular gaps
- If edge aligns with gap, clustering may fail
- **Fix:** Added golden-ratio distribution for Phase 4 probes

### Problem 3: Search Direction Bias ⚠️ PARTIALLY ADDRESSED
- All probes search along initial normal `n`
- If `n` is already blended (from accumulated face normals), probes may systematically favor one face
- **Status:** Tested per-face probing (see Implementation Results), but clustering-based assignment works better in practice

### Problem 4: No Adaptive Escalation ✅ ADDRESSED
- Always does 12 probes regardless of convergence
- No early exit for smooth surfaces (wasted samples)
- No escalation for difficult cases (insufficient samples)
- **Fix:** 3-phase adaptive probing with early exit and escalation

### Problem 5: Clustering Limitations ✅ MITIGATED
- Splitting by plane normal assumes edge is planar
- For vertices near but not on edge, split may be poor
- Minimum 3 points per cluster is strict with only 12 probes
- **Mitigation:** With 24 probes, minimum 3 per cluster is easily met

---

## Proposed Approaches

### Approach A: Increased Probe Count with Better Distribution

**Concept:** Use 24-32 probes with golden-ratio distribution, matching the reference method.

**Changes:**
```rust
// Replace fixed cardinal + 30° with golden-ratio spiral
let probe_dirs = generate_uniform_tangent_directions(n, 24);
```

**Pros:**
- Simple change, proven to work (reference method)
- Better angular coverage
- More points per cluster → better plane fits

**Cons:**
- 2x sample cost increase
- Still uses same tangent plane basis
- Doesn't address search direction bias

**Expected improvement:** Normal error 16-18° → ~8-10° (estimated from 2x more samples)

---

### Approach B: Adaptive Multi-Phase Probing

**Concept:** Start cheap, escalate if needed based on residual.

**Algorithm:**
```
Phase 1: Quick check (4 probes)
  - If residual < threshold AND angle deviation < 5°: DONE (smooth surface)

Phase 2: Standard probing (8 more probes = 12 total)
  - Try clustering
  - If clear separation: DONE (edge detected)
  - If ambiguous (residual still high, no clear clusters): escalate

Phase 3: High-precision (12 more probes = 24 total)
  - Golden-ratio distribution for remaining probes
  - Re-cluster with all points
```

**Pros:**
- Cheap for smooth surfaces (most vertices)
- Extra samples only where needed
- Better average-case performance

**Cons:**
- More complex logic
- Need to tune escalation thresholds
- Multiple clustering attempts

**Expected improvement:** Similar accuracy to Approach A, but 30-50% fewer samples on average

---

### Approach C: Per-Face Directional Probing

**Concept:** Once we suspect an edge, probe specifically in directions that stay on each face.

**Algorithm:**
```
Phase 1: Initial probing (12 probes as now)
  - Detect potential edge via clustering
  - Get preliminary normal_a, normal_b

Phase 2: Face-specific probing
  - Compute face-aligned tangent bases:
    - (t1a, t2a) perpendicular to normal_a
    - (t1b, t2b) perpendicular to normal_b
  - Probe 4-8 more points in each face's tangent plane
  - Search along respective face normal (not blended n)

Phase 3: Refined plane fitting
  - Fit planes using face-specific points
  - Much better normal accuracy
```

**Pros:**
- Addresses search direction bias directly
- Points guaranteed to be on correct face
- Better plane fits with targeted sampling

**Cons:**
- Requires two-phase approach
- More complex implementation
- Extra samples for edge vertices

**Expected improvement:** Normal error → ~5-8° for edge vertices

---

### Approach D: Gradient-Based Normal Estimation

**Concept:** Estimate normal from local density gradient instead of plane fitting.

**Algorithm:**
```
For each axis (X, Y, Z):
  - Sample at surface_pos ± delta along axis
  - Compute finite difference: grad_i = (density(+) - density(-)) / (2*delta)

Normal = -normalize(gradient)  // Points toward lower density (outside)
```

**Pros:**
- Only 6 samples needed
- No plane fitting or clustering
- Works for any surface geometry

**Cons:**
- Requires non-binary density function (current models return 0/1)
- Sensitive to delta choice
- Doesn't directly detect sharp edges

**Expected improvement:** Depends on density function smoothness. Not applicable to current binary samplers.

---

### Approach E: Hybrid Topology + Probing

**Concept:** Use mesh topology to guide probing directions.

**Algorithm:**
```
1. From mesh topology, identify adjacent faces to this vertex
2. Compute face normals from triangle geometry
3. If faces have similar normals: smooth surface
   - Use face normal directly, minimal probing for refinement
4. If faces have different normals: potential edge
   - Use face normals as initial guesses for the two surfaces
   - Probe in face-aligned directions to refine
5. Verify with geometry sampling
```

**Pros:**
- Leverages existing mesh topology information
- Targeted probing based on actual geometry
- Can detect edges from topology alone

**Cons:**
- Topology normals may be inaccurate (accumulated, not geometric)
- Requires access to mesh connectivity
- More complex integration

**Expected improvement:** Could be very good for edges that align with mesh structure

---

### Approach F: Multi-Scale Probing

**Concept:** Probe at multiple epsilon values to handle varying edge distances.

**Algorithm:**
```
Epsilon values: [0.05, 0.1, 0.2] * cell_size

For each epsilon:
  - Probe 8 directions
  - Fit plane, record residual

Analysis:
  - If all scales have low residual: smooth surface
  - If small epsilon low, large epsilon high: edge nearby but not at vertex
  - If all scales high: vertex on edge

For edge detection:
  - Use smallest epsilon with clear clustering
  - Larger epsilon may cross to other face
```

**Pros:**
- Handles varying vertex-to-edge distances
- Smaller epsilon = more precise for vertices on edge
- Larger epsilon = better for vertices near edge

**Cons:**
- 3x sample cost
- Complex analysis logic
- May still have clustering issues

**Expected improvement:** Better edge detection for vertices at varying distances from geometric edge

---

## Implementation Results

### Implemented: Approach B (Adaptive Multi-Phase Probing)

**Status: ✅ SUCCESSFUL**

The adaptive multi-phase probing was implemented and achieved significant improvements:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Normal Error (NormA) | 16-18° | 9.0° | ~50% reduction |
| Normal Error (NormB) | 16-18° | 9.6° | ~47% reduction |
| Samples/vertex | ~180 | 73 | 59% reduction |
| TPR | 100% | 100% | Maintained |
| FPR | 0% | 0% | Maintained |

**Key implementation details:**
- Phase 1: 4 cardinal probes with early-exit for smooth surfaces
- Phase 2: 8 additional probes (30° spacing)
- Phase 3: Edge detection check - if potential edge, escalate
- Phase 4: 12 golden-ratio probes for high-precision edge handling

### Attempted: Edge-Specific Probing (Variant of Approach C)

**Status: ❌ DID NOT IMPROVE OVER CLUSTERING**

An alternative approach was tested that used geometric edge direction estimation:

**Algorithm:**
```
1. After initial probing (12 points), estimate edge direction:
   - Compute centroid of all probe points
   - Run PCA on displacement vectors from centroid
   - Direction of MAXIMUM spread = perpendicular to edge
   - Edge direction = perp × initial_normal

2. Probe specifically for each face:
   - For each side (+perp and -perp):
     - Offset probe position away from edge
     - Search along initial normal for surface
     - Assign found points to that face's cluster

3. Fit planes to each face's points separately
```

**Results:**
| Attempt | NormA Error | NormB Error | Notes |
|---------|-------------|-------------|-------|
| Baseline (clustering) | 9.0° | 9.6° | Best result |
| Edge-specific v1 | 18.6° | 20.8° | Wrong direction relationship |
| Edge-specific v2 (fixed) | 10.4° | 11.9° | After fixing PCA interpretation |
| Hybrid (cluster then refine) | 14-18° | 14-18° | Refinement made it worse |
| Face-normal search direction | 11-12° | 11-12° | Still worse than clustering |

**Why Clustering Outperforms Edge-Specific:**

1. **Error propagation:** Edge-specific probing relies on accurate edge direction estimation from PCA. Any error in this estimate (~10-15°) propagates to face assignment errors.

2. **Clustering is adaptive:** The clustering algorithm tries multiple splitting strategies and picks the one with lowest residual. It finds the optimal split based on actual point positions, not estimated geometry.

3. **Search direction is less important than assignment:** Even when probing in theoretically better directions (along face normals), the assignment of points to faces is what matters most. Clustering does this optimally.

**Key Insight:**

> The fundamental difference is how points get assigned to faces:
> - **Clustering:** Assigns each point to the cluster that minimizes plane-fitting residual
> - **Edge-specific:** Assigns based on which side of the estimated edge the probe started from
>
> Clustering's adaptive assignment corrects for probe position errors, while edge-specific compounds them.

### Remaining Opportunities

1. **Better clustering algorithms:** The current clustering tries 4 splitting strategies. More sophisticated methods (k-means, spectral clustering) might improve further.

2. **Iterative refinement with feedback:** After initial clustering, could re-probe in directions informed by the found face normals, then re-cluster. This differs from the failed hybrid approach by using clustering for assignment at each step.

3. **Smaller probe epsilon:** Current 0.1 * cell_size might cross edge for tight geometry. Adaptive epsilon based on detected edge proximity could help.

4. **More probes for edges:** Current 24 may still be insufficient. Could escalate to 32-48 for vertices with high residual after 24 probes.

---

## Recommendation (Updated)

**Completed:** ✅ Adaptive Multi-Phase Probing (Approach B)
- Early-exit for smooth surfaces
- Escalation to 24 probes for edges
- Golden-ratio distribution in Phase 4

**Current state:** 9° normal error is acceptable for most use cases, and TPR/FPR are perfect.

**Next priorities (if further improvement needed):**

1. **More aggressive escalation:** For vertices with residual still high after 24 probes, escalate to 32-48 probes. This is a straightforward extension of the current approach.

2. **Improve clustering:** The clustering algorithm is the key to accuracy. Better splitting strategies or iterative refinement could reduce error below 9°.

3. **Adaptive probe epsilon:** For very tight edges, smaller epsilon could keep probes on the correct face. This would help with vertices where the edge passes very close.

**Not recommended:** Edge-specific probing based on estimated edge direction. Testing showed this consistently performs worse than clustering due to error propagation.

---

## Metrics for Evaluation

When testing changes, measure:

1. **Normal accuracy** (degrees error vs 32-probe reference)
   - ~~Original: 16-18° mean~~
   - **Current: 9.0°/9.6° (NormA/NormB)** ✅
   - Target: <10° mean, <5° for smooth surfaces

2. **Sharp edge detection**
   - True positive rate: **100%** ✅
   - False positive rate: **0%** ✅

3. **Crossing position error** (% of cell size)
   - Current: 2.67% mean, 5.32% P95
   - Target: <2% mean, <4% P95

4. **Sample efficiency**
   - ~~Original: ~180 samples/vertex~~
   - **Current: 73 samples/vertex** ✅
   - Target: Similar or better for same accuracy

5. **Visual quality**
   - Render with `--recalc-normals` and check for scalloping
   - Edge straightness at depth 1-2

---

## Test Commands

```bash
# Run edge diagnostics
cargo run -p volumetric_cli --release --features volumetric/edge-diagnostic -- \
  mesh -i rotated_cube.wasm/rotated_cube.wasm -o /tmp/test.stl \
  --sharp-edges --max-depth 2

# Render to check visual quality
cargo run -p volumetric_cli --release -- render \
  -i rotated_cube.wasm/rotated_cube.wasm -o edge_test.png \
  --sharp-edges --recalc-normals --max-depth 2 \
  --camera-pos 1.5,0.8,3 --camera-target 0.6,0.1,0.3 \
  --projection ortho --ortho-scale 0.8 \
  --width 1024 --height 1024 --grid 0
```
