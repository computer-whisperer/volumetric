# Normal Probing Architecture Report

## Executive Summary

The normal probing system estimates surface normals at mesh vertices by sampling nearby surface points and fitting planes. After implementing adaptive multi-phase probing and experimental algorithms, the system achieves:

**Standard Algorithm (Production):**
- **4.2° normal error** vs analytical ground truth (NormA)
- **100% true positive rate** for sharp edge detection
- **0% false positive rate**
- **73 samples/vertex**

**Experimental Algorithms:**
- **Bisection:** 3.7° error, 457 samples/vertex
- **Multiradius:** 3.0° error, 912 samples/vertex

**Key Insights:**
1. **Clustering-based point assignment consistently outperforms edge-direction-based assignment**, even when the edge direction is estimated from PCA.
2. **The probe-based reference diagnostic has inherent error** (~5.6° vs analytical truth), making "vs reference" metrics misleading. Always validate against analytical ground truth when available.
3. **More probes help, but with diminishing returns.** The standard 73-sample method achieves 4.2° error; the 912-sample multiradius achieves 3.0° — only 1.2° improvement for 12x the cost.

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

## Experimental Algorithms (2024)

Two new experimental edge detection algorithms were implemented and tested against both the probe-based reference and analytical ground truth.

### Algorithm: Face-Assignment Bisection (`edge_detect_bisection`)

**Concept:** Binary search around the vertex to find angles where face assignment changes, then sample densely on each side.

**Algorithm:**
```
1. Initial probing (12 directions at 30° spacing)
   - Classify each probe point to face A or B via clustering

2. Edge angle detection
   - For each adjacent pair of probes with different face assignments:
     - Binary search (8 iterations) to find precise transition angle
   - Collect 2-4 edge crossing angles

3. Face-specific sampling
   - For each detected face region:
     - Sample 8 additional probes within that angular range
     - Fit plane to face-specific points

4. Final plane fitting with all points per face
```

**Results:**
| Metric | Value |
|--------|-------|
| vs Reference NormA | 2.7° |
| vs Reference NormB | 4.4° |
| vs Analytical NormA | 3.7° |
| vs Analytical NormB | 7.5° |
| Samples/vertex | 457 |
| TPR | 100% |
| FPR | 0% |

**Analysis:** Bisection achieves the best "vs reference" metrics because it uses a similar sampling strategy. However, analytical validation shows it's only marginally better than the standard method (3.7° vs 4.2°) at 6x the sample cost.

---

### Algorithm: Multi-Radius Probing (`edge_detect_multiradius`)

**Concept:** Probe at multiple radii (0.5x, 0.75x, 1.0x, 1.5x epsilon) and use consistency across radii to detect edges and improve normal estimates.

**Algorithm:**
```
1. Multi-radius probing
   - For each of 16 directions (golden-ratio distributed):
     - Probe at 4 different radii
     - Track which probes find surface successfully

2. Consistency analysis
   - For each direction, check if all radii agree on face assignment
   - Directions with inconsistent assignments indicate edge proximity

3. Clustering with radius weighting
   - Weight points by radius (smaller radius = higher confidence)
   - Cluster using weighted residual

4. Edge detection
   - If clustering shows clear separation with angle > threshold:
     - Mark as sharp edge
     - Return both face normals
```

**Results:**
| Metric | Value |
|--------|-------|
| vs Reference NormA | 3.2° |
| vs Reference NormB | 7.4° |
| vs Analytical NormA | 3.0° |
| vs Analytical NormB | 8.1° |
| Samples/vertex | 912 |
| TPR | 100% |
| FPR | 0% |

**Analysis:** Multiradius achieves the best analytical accuracy for NormA (3.0°) but at very high sample cost (912 samples). The NormB error is higher, suggesting the second face normal is harder to estimate accurately.

---

## Analytical Ground Truth Validation

### The Problem with Probe-Based References

The original diagnostic compared detection methods against a "reference" computed with 258 probes. However, this reference itself has error because:

1. **Same fundamental limitations:** The reference uses the same probe-and-cluster approach, just with more probes
2. **Search direction bias:** All methods search along the initial blended normal
3. **Discretization effects:** Probing a boolean field has inherent quantization

### Analytical Ground Truth

For the rotated cube test case (rx=35.264°, ry=45°, rz=0°), we can compute the **exact** face normals analytically by applying the rotation matrix to the 6 unit cube face normals.

**Implementation:** `AnalyticalRotatedCube` struct in `adaptive_surface_nets_2.rs`

```rust
// Euler XYZ rotation: R = Rz * Ry * Rx
let analytical = AnalyticalRotatedCube::new(35.264, 45.0, 0.0);
let (error_a, error_b) = analytical.compute_pair_errors(detected_normal_a, detected_normal_b);
```

### Validation Results

| Method | vs Reference A | vs Reference B | vs Analytical A | vs Analytical B | Samples |
|--------|---------------|----------------|-----------------|-----------------|---------|
| Reference (258-probe) | — | — | **5.6°** | **5.2°** | 9,034 |
| Standard | 13.4° | 13.4° | **4.2°** | 8.5° | 73 |
| Bisection | 2.7° | 4.4° | **3.7°** | 7.5° | 457 |
| Multiradius | 3.2° | 7.4° | **3.0°** | 8.1° | 912 |

### Key Findings

1. **The reference has ~5.6° error** against analytical truth. This means any method can appear to have up to ~11° error "vs reference" even if it's actually quite accurate.

2. **Standard method outperforms reference on NormA** (4.2° vs 5.6°). The "13.4° vs reference" metric was misleading — the standard method is actually doing well.

3. **Diminishing returns on samples:** Going from 73 to 912 samples only improves NormA from 4.2° to 3.0° (1.2° gain for 12x cost).

4. **NormB is consistently harder:** All methods show higher error on the second face normal (7.5-8.5° vs 3.0-4.2° for NormA). This may be due to:
   - Smaller angular coverage on one side of the edge
   - Systematic bias in the initial normal direction
   - Fewer probe points landing on the "far" face

### Recommendations

- **Always validate against analytical ground truth** when testing on known geometry
- **Don't trust "vs reference" metrics alone** — the reference has its own errors
- **Focus optimization efforts on NormB** — there's more room for improvement there
- **Consider the cost/benefit tradeoff** — 73 samples at 4.2° error may be better than 912 samples at 3.0° error for most use cases

---

## Recommendation (Updated)

**Completed:** ✅ Adaptive Multi-Phase Probing (Approach B)
- Early-exit for smooth surfaces
- Escalation to 24 probes for edges
- Golden-ratio distribution in Phase 4

**Completed:** ✅ Experimental Algorithms (2024)
- Face-Assignment Bisection (3.7° analytical error, 457 samples)
- Multi-Radius Probing (3.0° analytical error, 912 samples)

**Completed:** ✅ Sample-by-Sample Algorithms (2026)
- Boundary Bisection (36.6° analytical error, 34 samples)
- Maximum Uncertainty (34.6° analytical error, 106 samples)
- Adaptive Grid (31.8° analytical error, 95 samples)
- Infrastructure: `SampleCache`, `CachedSample`, bootstrap phase, surface crossing finder

**Completed:** ✅ Analytical Ground Truth Validation
- `AnalyticalRotatedCube` struct for rotated cube test case
- Diagnostics now report both vs-reference and vs-analytical metrics

**Current state:**
- Standard method achieves **4.2° analytical error** at 73 samples/vertex
- TPR/FPR are perfect (100%/0%)
- This is likely sufficient for most use cases
- Sample-by-sample algorithms achieve lower sample counts (34-106) but with reduced accuracy (31-37°)

**Next priorities (if further improvement needed):**

1. **Improve NormB accuracy:** All methods show ~2x higher error on NormB (7-8°) compared to NormA (3-4°). Investigate why and address the asymmetry.

2. **Refine sample-by-sample algorithms:** The infrastructure is in place. Key improvements:
   - Increase binary search iterations (6 → 10-12) for better surface points
   - Better edge detection criteria to reduce FPR
   - Hybrid approach: sample-by-sample detection + traditional probing for refinement

3. **Adaptive probe epsilon:** For very tight edges, smaller epsilon could keep probes on the correct face.

4. **Test on more geometry:** Current validation uses only the rotated cube. Test on spheres, cylinders, and complex CSG to ensure generalization.

**Not recommended:**
- Edge-specific probing based on estimated edge direction (consistently worse than clustering)
- Massive sample increases for marginal gains (912 samples → 3.0° vs 73 samples → 4.2°)
- Using current sample-by-sample algorithms in production (accuracy gap too large)

---

## Metrics for Evaluation

When testing changes, measure:

1. **Normal accuracy vs analytical ground truth** (primary metric for known geometry)
   - **Current: 4.2°/8.5° (NormA/NormB)** ✅
   - Target: <5° NormA, <8° NormB

2. **Normal accuracy vs reference** (secondary metric, useful for relative comparison)
   - Current: 13.4°/13.4° (NormA/NormB)
   - Note: Reference itself has ~5.6° error vs analytical

3. **Sharp edge detection**
   - True positive rate: **100%** ✅
   - False positive rate: **0%** ✅

4. **Crossing position error** (% of cell size)
   - Current: 2.00% mean, 5.32% P95
   - Target: <2% mean, <4% P95

5. **Sample efficiency**
   - ~~Original: ~180 samples/vertex~~
   - **Current: 73 samples/vertex** ✅
   - Target: Similar or better for same accuracy

6. **Visual quality**
   - Render with `--recalc-normals` and check for scalloping
   - Edge straightness at depth 1-2

---

## Test Commands

```bash
# Run edge diagnostics (includes analytical ground truth validation)
cargo run -p volumetric_cli --release --features volumetric/edge-diagnostic -- \
  mesh -i rotated_cube.wasm/rotated_cube.wasm -o /tmp/test.stl \
  --sharp-edges --max-depth 2

# Expected output includes:
#   Reference vs analytical: NormA=5.6°  NormB=5.2°
#   standard: TPR=100.0%  FPR=0.0%  NormA=13.4°  NormB=13.4°  samples/vert=73
#            vs analytical: NormA=4.2°  NormB=8.5°
#   bisection: TPR=100.0%  FPR=0.0%  NormA=2.7°  NormB=4.4°  samples/vert=457
#            vs analytical: NormA=3.7°  NormB=7.5°
#   multiradius: TPR=100.0%  FPR=0.0%  NormA=3.2°  NormB=7.4°  samples/vert=912
#            vs analytical: NormA=3.0°  NormB=8.1°
#   boundary_bisection: TPR=63.6%  FPR=11.2%  samples/vert=34
#            vs analytical: NormA=36.6°  NormB=23.1°
#   max_uncertainty: TPR=95.5%  FPR=53.0%  samples/vert=106
#            vs analytical: NormA=34.6°  NormB=30.1°
#   adaptive_grid: TPR=86.4%  FPR=35.2%  samples/vert=95
#            vs analytical: NormA=31.8°  NormB=28.7°

# Render to check visual quality
cargo run -p volumetric_cli --release -- render \
  -i rotated_cube.wasm/rotated_cube.wasm -o edge_test.png \
  --sharp-edges --recalc-normals --max-depth 2 \
  --camera-pos 1.5,0.8,3 --camera-target 0.6,0.1,0.3 \
  --projection ortho --ortho-scale 0.8 \
  --width 1024 --height 1024 --grid 0
```

## Enhanced Reference Diagnostic

The reference diagnostic was upgraded to use 258 probes per vertex:
- 64 directions (golden-ratio distributed)
- 4 radii per direction (0.5x, 0.75x, 1.0x, 1.25x epsilon)
- 2 additional probes along the normal direction
- 32 binary search iterations (up from 12)

Despite this thorough sampling, the reference still has ~5.6° error against analytical ground truth, demonstrating the inherent limitations of probe-based normal estimation on boolean fields.

---

## Sample-by-Sample Edge Detection (2026)

### Motivation

The existing edge detection algorithms consume many samples internally per "probe":
- 1 origin sample
- 2 coarse search samples
- ~12 binary search samples

This means each of the ~12-24 probe directions uses ~14 samples, but only the final surface crossing point is kept. All intermediate data is discarded.

**New Approach:** Track every `(position, is_inside)` boolean sample. Make decisions about the next sample based on all accumulated data. This enables more sample-efficient algorithms.

### Infrastructure Added

```rust
// Cached boolean sample
struct CachedSample {
    position: (f64, f64, f64),
    is_inside: bool,
}

// Sample cache with helper methods
struct SampleCache {
    samples: Vec<CachedSample>,
}

impl SampleCache {
    fn sample(&mut self, pos, sampler, stats) -> &CachedSample;
    fn partition(&self) -> (inside_samples, outside_samples);
    fn has_mixed_samples(&self) -> bool;
    fn count(&self) -> usize;
}
```

**Shared Bootstrap Phase (9 samples):**
1. Vertex position (1 sample)
2. 4 cardinal directions at probe_epsilon in tangent plane (4 samples)
3. ±normal at 0.5×epsilon and 1.0×epsilon (4 samples)

Bootstrap returns early if all samples are same class (smooth surface detected).

### Algorithm 1: Boundary Bisection (`edge_detect_boundary_bisection`)

**Strategy:** Angular scan to find inside/outside transitions, then binary search to refine transition angles.

**Phases:**
```
1. Bootstrap (9 samples)
   - If all same class → smooth surface, done

2. Angular scan (8 samples)
   - Sample 8 directions at probe_epsilon
   - Track (angle, is_inside) for each

3. Transition bisection (6 iterations each)
   - For each adjacent pair with different classifications
   - Binary search to find precise transition angle

4. Perpendicular sampling
   - Sample on each side of detected edge for normal refinement

5. Surface crossing refinement
   - Find accurate surface crossings via binary search between inside/outside pairs
   - Fit planes to refined surface points
```

**Results:**
| Metric | Value |
|--------|-------|
| TPR | 63.6% |
| FPR | 11.2% |
| vs Analytical NormA | 36.6° |
| vs Analytical NormB | 23.1° |
| Samples/vertex | 34 |

---

### Algorithm 2: Maximum Uncertainty (`edge_detect_max_uncertainty`)

**Strategy:** Maintain a polar belief grid, iteratively sample where uncertainty is highest (P(inside) ≈ 0.5).

**Phases:**
```
1. Bootstrap (9 samples)
   - If all same class → smooth surface, done

2. Initialize belief grid
   - 16 angular bins × 3 radial bins
   - Each cell stores (sum_of_inside, count)

3. Uncertainty-guided sampling (up to 25 iterations)
   - Find cell with highest uncertainty: 0.5 - |P(inside) - 0.5|
   - Sample at cell center
   - Update belief grid with smoothing to neighbors
   - Stop when max uncertainty < 0.1

4. Surface crossing refinement
   - Binary search between inside/outside pairs
   - Plane fitting on refined surface points
```

**Results:**
| Metric | Value |
|--------|-------|
| TPR | 95.5% |
| FPR | 53.0% |
| vs Analytical NormA | 34.6° |
| vs Analytical NormB | 30.1° |
| Samples/vertex | 106 |

---

### Algorithm 3: Adaptive Grid (`edge_detect_adaptive_grid`)

**Strategy:** Start with coarse grid, refine only cells that contain the surface boundary.

**Phases:**
```
1. Bootstrap (9 samples)
   - If all same class → smooth surface, done

2. Initialize coarse grid
   - 4 angular × 2 radial = 8 cells
   - Track (has_inside, has_outside, sampled) per cell

3. Adaptive refinement (up to 5 iterations, max 50 samples)
   - Sample unsampled cells
   - Identify "mixed" cells (has both inside and outside samples)
   - Subdivide mixed cells into 4 subcells (up to level 3)
   - Stop when no mixed cells remain

4. Surface crossing refinement
   - Binary search between inside/outside pairs
   - Plane fitting on refined surface points
```

**Results:**
| Metric | Value |
|--------|-------|
| TPR | 86.4% |
| FPR | 35.2% |
| vs Analytical NormA | 31.8° |
| vs Analytical NormB | 28.7° |
| Samples/vertex | 95 |

---

### Comparative Results Summary

| Method | TPR | FPR | Analytical A | Analytical B | Samples |
|--------|-----|-----|--------------|--------------|---------|
| **Standard** | 100% | 0% | 4.2° | 8.5° | 73 |
| **Bisection** | 100% | 0% | 3.7° | 7.5° | 457 |
| **Multiradius** | 100% | 0% | 3.0° | 8.1° | 912 |
| **Boundary Bisection** | 63.6% | 11.2% | 36.6° | 23.1° | 34 |
| **Max Uncertainty** | 95.5% | 53.0% | 34.6° | 30.1° | 106 |
| **Adaptive Grid** | 86.4% | 35.2% | 31.8° | 28.7° | 95 |

### Analysis

**Sample Efficiency vs Accuracy Tradeoff:**

The sample-by-sample algorithms achieve significantly lower sample counts (34-106 vs 457-912) but with reduced accuracy and detection reliability:

1. **TPR degradation:** The new algorithms miss some sharp edges (63-95% TPR vs 100%)
2. **FPR issues:** Higher false positive rates, especially max_uncertainty (53%)
3. **Normal accuracy:** 30-37° error vs 3-4° for existing methods

**Why the Accuracy Gap?**

1. **Surface point quality:** Existing algorithms use full binary search (12 iterations) along the normal direction to find precise surface crossings. Sample-by-sample algorithms interpolate between inside/outside samples or use quick 6-iteration searches, giving less accurate surface points.

2. **Sparse data:** With only 34-106 samples covering a 2D tangent disk, there aren't enough surface points per cluster for accurate plane fitting. Existing methods get 60+ surface points per face.

3. **Clustering sensitivity:** The clustering algorithm needs sufficient well-distributed points to distinguish two planes from one. Sparse, unevenly distributed points lead to poor cluster assignments.

**Potential Improvements:**

1. **Increase surface crossing iterations:** Change from 6 to 10-12 iterations would improve surface point accuracy at modest sample cost.

2. **Smarter pair selection:** Only binary search between pairs that are geometrically meaningful (not redundant pairs from same angular region).

3. **Better edge detection criteria:** Current residual-based detection has high FPR. Could use the angular distribution of inside/outside samples more directly.

4. **Hybrid approach:** Use sample-by-sample for detection, then switch to traditional probing for normal refinement if an edge is detected.

### Test Command

```bash
cargo run -p volumetric_cli --release --features volumetric/edge-diagnostic -- \
  mesh -i rotated_cube.wasm/rotated_cube.wasm -o /tmp/test.stl \
  --sharp-edges --max-depth 2

# Output includes all methods:
#   standard: TPR=100.0%  FPR=0.0%  samples/vert=73
#            vs analytical: NormA=4.2°  NormB=8.5°
#   bisection: TPR=100.0%  FPR=0.0%  samples/vert=457
#            vs analytical: NormA=3.7°  NormB=7.5°
#   multiradius: TPR=100.0%  FPR=0.0%  samples/vert=912
#            vs analytical: NormA=3.0°  NormB=8.1°
#   boundary_bisection: TPR=63.6%  FPR=11.2%  samples/vert=34
#            vs analytical: NormA=36.6°  NormB=23.1°
#   max_uncertainty: TPR=95.5%  FPR=53.0%  samples/vert=106
#            vs analytical: NormA=34.6°  NormB=30.1°
#   adaptive_grid: TPR=86.4%  FPR=35.2%  samples/vert=95
#            vs analytical: NormA=31.8°  NormB=28.7°
```

### Conclusion

The sample-by-sample approach shows promise for sample efficiency but doesn't yet achieve competitive accuracy. The key bottleneck is getting enough high-quality surface points for reliable plane fitting.

**Recommendation:** The standard method (73 samples, 4.2° error, 100% TPR, 0% FPR) remains the best choice for production use. The sample-by-sample algorithms provide a foundation for future exploration of more sample-efficient edge detection, but need further refinement before they can match the existing methods' reliability.
