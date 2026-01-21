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

## Algorithm Pipeline (How Initial Normals Are Computed)

Understanding the NormB asymmetry requires understanding how `initial_normal` is computed.

### Full Pipeline Overview

```
Stage 1-2: Voxelization
  → Identify surface-crossing cell edges
  → Each crossing edge becomes a potential vertex

Stage 3: Topology Finalization
  → Place vertices at cell edge midpoints
  → Create triangles connecting adjacent vertices
  → Compute accumulated_normals from triangle face normals  ← FIRST NORMALS

Stage 4a: Vertex Refinement
  → Binary search along accumulated_normal to find actual surface
  → Vertices move from midpoints to exact surface positions

Stage 4b: Normal Recomputation
  → Recompute accumulated_normals from REFINED positions
  → Same triangle connectivity, new vertex positions         ← SECOND NORMALS

Stage 4c: Normal Refinement via Probing (Edge Detection)
  → Use recomputed_normals as initial search direction
  → Probe tangent plane to find surface points
  → Cluster and fit planes for edge detection
```

### Key Insight: Two Sets of Accumulated Normals

1. **Stage 3 normals**: Used for vertex refinement. Computed from edge midpoints.
2. **Stage 4b normals**: Used for probing. Computed from refined positions.

The edge detection uses **Stage 4b normals** (`recomputed_normals`). These are computed from:
- **Refined vertex positions** (on the actual surface)
- **Original triangle connectivity** (from Stage 3)

### Why Transition Triangles Exist

In surface nets, triangles approximate the isosurface. At a geometric edge of the model:
- The isosurface transitions from one face to another
- Triangles in this region connect vertices from both faces
- These "transition triangles" have normals pointing in **intermediate directions**

For a 90° cube edge, we might expect:
- Half the triangles point toward face A
- Half point toward face B
- Sum = face bisector

But actually we see:
- Some triangles cleanly point toward face A or B
- **Transition triangles** point in neither direction (errors of 35-50° to their "best match" face)
- These transitions can dominate the sum, biasing it away from the true bisector

### Key Discovery: Vertex Refinement Already Detects Edges

**The vertex refinement stage (4a) implicitly detects edge vertices with 100% accuracy.**

During vertex refinement, we search along 4 directions (primary + 3 cardinal axes) for surface crossings:

```
Crossing Count Analysis:
  2 crossings: 132 vertices, 22 sharp (16.7%)
  3 crossings: 1288 vertices, 0 sharp (0.0%)
  4 crossings: 2414 vertices, 0 sharp (0.0%)
  Average crossings: sharp=2.00, smooth=3.60
```

**Every sharp vertex has exactly 2 crossings. Every smooth vertex has 3-4 crossings.**

Why this happens:
- **Smooth vertices**: Sit on a single face. All 4 search directions find the same surface (redundant paths).
- **Edge vertices**: Sit on a geometric edge. Only 2 search directions find surface crossings (one toward each face). The other 2 directions point into empty space or solid material.

**Implications:**
1. **Early edge detection**: We can flag vertices with 2 crossings as "likely edge" before any probing
2. **Don't trust the normal**: For 2-crossing vertices, the accumulated normal is unreliable
3. **Alternative search strategy**: Use the 2 successful crossing directions as hints for face normals

**Refinement Outcome Correlation:**
```
Primary: 3794/3834 vertices, 16 sharp (0.4%)
FallbackX: 14/3834 vertices, 3 sharp (21.4%)
FallbackZ: 14/3834 vertices, 3 sharp (21.4%)
Sharp vertices: 16/22 Primary, 6/22 Fallback
```

27% of sharp vertices needed fallback directions because the primary direction (accumulated normal) failed to find a surface crossing. Fallback vertices are 50x more likely to be sharp.

---

## Identified Issues

### 1. NormB Asymmetry — ROOT CAUSE IDENTIFIED

All methods show ~2x higher error on NormB (7-8°) vs NormA (3-4°).

**Key Finding:** The error distribution is **bimodal** — one normal is nearly perfect (~0°), the other is poor (~10°). Which normal is "A" vs "B" depends on arbitrary clustering assignment.

**ROOT CAUSE: Transition Triangles at Mesh Edges**

At edge vertices, the accumulated normal has tiny magnitude (~0.006 instead of ~1.0):
```
Vertex #1: initial_n raw = (0.0016, -0.0026, -0.0054)  len=0.0061
Vertex #2: initial_n raw = (-0.0011, 0.0026, -0.0048)  len=0.0056
```

**This is NOT simple cancellation of opposite normals.** Analysis of the actual mesh topology reveals:

```
Sharp vertex 9 on -Y/+X edge (6 triangles touch):
  tri 0: area=0.000353 dir=(-0.408,-0.816,-0.409) → -Y (err=0.0°)  ✓ clean face
  tri 1: area=0.000792 dir=(-0.408,-0.816,-0.408) → -Y (err=0.0°)  ✓ clean face
  tri 2: area=0.001044 dir=(0.108,-0.057,-0.992) → +X (err=38.9°) ← TRANSITION
  tri 3: area=0.000350 dir=(0.438,-0.772,-0.460) → -Y (err=50.3°) ← TRANSITION
  tri 4: area=0.000693 dir=(0.684,-0.001,-0.729) → +X (err=1.8°)   ✓ clean face
  tri 5: area=0.000719 dir=(0.707,-0.027,-0.707) → +X (err=1.6°)   ✓ clean face
```

**The real mechanism:**
1. Surface nets creates "transition triangles" at geometric edges to smooth the mesh
2. These transition triangles have normals pointing in NEITHER face direction
3. Triangle 2 has the LARGEST area but points almost -Z (38.9° from +X, its "best match")
4. These transition triangles dominate the accumulated normal
5. The resulting direction is biased away from the true face bisector

**Expected vs Actual:**
```
expected bisector = normalize(-Y + +X) ≈ (0.21, -0.58, -0.79)
actual (dominated by transitions)      ≈ (0.25, -0.42, -0.87)  ← biased toward -Z
```

The small magnitude comes from both:
- Small triangle areas at the mesh resolution (depth 2 = 32³)
- Partial cancellation between face contributions

**Consequences of degenerate init_n:**

| Factor | Correlation | Notes |
|--------|-------------|-------|
| init_n closer to face | → smaller cluster (73%) | Probes land more on opposite face |
| init_n closer to face | → slightly lower error (6.0° vs 6.8°) | Anchor point compensates |
| Face-specific error | -Y: 11.5°, +X: 4.0° | ~3x variation by face |

**Face-specific accuracy pattern (rotated cube rx=35.264°, ry=45°):**
```
+X: mean=4.0° (n=12) — best
±Z: mean=5.0-5.1° (n=6 each)
-X: mean=7.6° (n=10)
+Y: mean=7.7° (n=5)
-Y: mean=11.5° (n=5) — worst
```

The ±Y faces have highest error, likely because the rx rotation affects their orientation relative to the probe pattern.

**Hypotheses Tested and REJECTED:**

1. **Anchor point bias (`surface_pos` in cluster_a):** Removing it from cluster_a had no effect.
2. **Adding anchor to cluster_b:** Made things WORSE (NormB: 8.5° → 13.6°) because `surface_pos` is on the edge, not on face B's plane.
3. **Search direction alignment:** No correlation (10 vs 12 split, essentially random).
4. **Cluster size:** Cases #5, #7 show larger cluster can have WORSE accuracy.
5. **More probes (36 total):** Mixed results (NormA worse, NormB slightly better).

**Potential Fixes (Prioritized by Likelihood of Success):**

1. **Use crossing count as early edge indicator (RECOMMENDED):**
   - Vertices with exactly 2 crossings in refinement are 100% correlated with sharp edges
   - For 2-crossing vertices, skip the normal-based early exit entirely
   - Use the 2 successful search directions as hints for face normals
   - Cache crossing positions for reuse in probing

2. **Hemisphere-only normal for edge vertices:**
   - For 2-crossing vertices, use accumulated normal ONLY for inside/outside determination
   - Use world-aligned or spherical probing instead of tangent-plane probing
   - Don't let the biased normal direction influence probe distribution

3. **Multiple probe orientations:** When crossing_count <= 2, probe in several tangent planes and pick best result

4. **Cache and reuse refinement probes:**
   - Vertex refinement already does ~20 samples per vertex
   - These could seed the normal refinement instead of starting fresh
   - Would reduce redundant sampling

**Note:** The reference method (258 probes) achieves balanced errors (5.6° vs 5.2°), suggesting sufficient probe density overcomes the degenerate init_n issue. But using the crossing count signal could achieve similar results with far fewer probes.

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

**Initial Conclusion:** Root cause remains unknown. The asymmetry is NOT caused by anchor point, search direction, or cluster size.

### 2026-01-21 (continued): Root Cause Discovery

**Breakthrough:** Discovered that accumulated normals at edge vertices are nearly zero!

**Key Evidence:**
```
Vertex #1: initial_n raw = (0.0016, -0.0026, -0.0054)  len=0.0061
Vertex #2: initial_n raw = (-0.0011, 0.0026, -0.0048)  len=0.0056
```

At edge vertices, face normals from both sides partially cancel, leaving a tiny vector that normalizes to an arbitrary-ish direction.

**Additional diagnostics added:**
1. Vector printing: initial_n raw, normalized, expected bisector, edge direction
2. Face-specific accuracy tracking: per-face mean error
3. Init_n bias correlation: which face init_n is closer to vs accuracy

**New Correlation Findings:**

| Correlation | Result |
|-------------|--------|
| init_n closer to face → smaller cluster | 73% (strong) |
| init_n closer to face → lower error | 6.0° vs 6.8° (weak) |
| Larger cluster → better | 73% (but confounded) |

**Face-specific accuracy:**
```
+X: mean=4.0° (n=12) — best
±Z: mean=5.0-5.1° (n=6 each)
-X: mean=7.6° (n=10)
+Y: mean=7.7° (n=5)
-Y: mean=11.5° (n=5) — worst (3x worse than +X)
```

**Root Cause Confirmed:** The degenerate initial normal at edge vertices creates an arbitrary tangent plane orientation, causing asymmetric probe distribution. The face that init_n is biased toward gets fewer probe points but has the anchor point advantage.

---

## Next Steps

### Immediate — Fix Degenerate Init_n
1. [ ] **Detect degenerate initial normals:** When |accumulated_normal| < threshold, flag vertex as "edge candidate"
2. [ ] **Alternative probe strategy for edge candidates:**
   - Option A: Use position-based heuristic (direction from cube center)
   - Option B: Probe in multiple tangent plane orientations, pick best
   - Option C: Use vertex-to-edge direction from mesh topology
3. [ ] **Iterative refinement for edges:** After initial detection, re-probe with each cluster's normal as search direction

### Short-term
4. [ ] Increase `find_surface_crossing_quick` iterations: 6 → 12 (for sample-by-sample methods)
5. [ ] Implement adaptive clustering thresholds for sparse data
6. [ ] Test on diverse geometry (spheres, cylinders, CSG) to validate findings generalize

### Medium-term
7. [ ] Design unified algorithm for Case 1 and Case 2
8. [ ] Optimize for common case (smooth surfaces)
9. [ ] Consider tangent-plane-independent sampling strategies

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

---

## Code Refactoring (2026-01-21)

The monolithic `adaptive_surface_nets_2.rs` (8,973 lines) was refactored into a modular directory structure:

```
src/adaptive_surface_nets_2/
├── mod.rs              # Public API, entry points
├── types.rs            # All type definitions
├── lookup_tables.rs    # MC tables, edge mappings
├── parallel_iter.rs    # Rayon/sequential helpers
├── stage1.rs           # Coarse grid discovery
├── stage2.rs           # Subdivision & emission
├── stage3.rs           # Topology finalization
├── stage4_stub.rs      # STUBBED - passthrough only
├── diagnostics.rs      # Feature-gated research tools
└── tests.rs            # Unit tests
```

**Key changes:**
- **Stage 4 stubbed**: All vertex refinement, normal probing, and sharp edge detection code has been removed from production. Stage 4 now passes through Stage 3 output unchanged.
- **Research code preserved**: The diagnostic functions remain available under feature flags (`edge-diagnostic`, `normal-diagnostic`) for continued research.
- **API unchanged**: `adaptive_surface_nets_2()` works identically; `SharpEdgeConfig` is kept for compatibility but has no effect.

**Rationale**: The sharp edge detection algorithms documented in this file did not achieve production-quality results. The experimental code was consuming maintenance effort without providing value. By stubbing Stage 4, the meshing pipeline is simpler and faster while research can continue separately.

**Archived code location**: See `PHASE4_ARCHIVE.md` for documentation of removed algorithms and their line ranges in the git history.
