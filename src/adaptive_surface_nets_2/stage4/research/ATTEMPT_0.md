# Attempt 0 Notes

Date/Time: 2026-01-25 16:50 EST (reviewed)

> **WARNING (2026-01-25 21:00 EST):** Results below were obtained with incorrect
> scaling parameters (cell_size=1.0, fixed offset=0.1). The algorithm is NOT
> scale-invariant due to hardcoded absolute thresholds. All metrics need
> re-verification. See `EDGE_REFINEMENT_RESEARCH.md` for details.

## Summary
Implementation of residual-based geometry classification with full-sphere edge detection.

## Edge Detection Results

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| Classification | 12/12 (100%) | 100% | ✅ |
| N_A error | 1.01° avg | <1° | ⚠️ Close |
| N_B error | 12.31° avg | <1° | ❌ Needs work |
| Samples | 913 avg | <80 | ❌ Over budget |

Per-edge breakdown:
```
Edge  0:  0.1° /  0.0°  (1205 samples)
Edge  1:  0.1° /  2.9°  ( 539 samples)
Edge  2:  0.1° / 75.1°  ( 679 samples) ← N_B problem
Edge  3:  0.0° /  0.0°  (1173 samples)
Edge  4:  0.3° /  0.0°  (1171 samples)
Edge  5:  0.0° /  0.0°  (1185 samples)
Edge  6:  0.0° / 11.1°  ( 667 samples)
Edge  7:  0.1° / 57.3°  ( 541 samples) ← N_B problem
Edge  8: 10.8° /  1.1°  ( 699 samples) ← N_A problem
Edge  9:  0.5° /  0.3°  ( 713 samples)
Edge 10:  0.1° /  0.0°  (1187 samples)
Edge 11:  0.0° /  0.0°  (1201 samples)
```

## Face Detection Results

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| Error | 1.68° avg | <1° | ⚠️ Close |
| Samples | 269 avg | <35 | ❌ Over budget |

## Key Findings

1. Probe position matters critically:
   - Probing from ON the surface gives a thin coplanar shell (all points ~same distance)
   - Probing from INSIDE (0.1 offset) gives good plane separation for RANSAC
2. Full-sphere probing outperforms biased hemispheres.
3. RANSAC threshold of 0.006 (from `robust_edge.rs`) provides good plane separation.
4. Second normal (N_B) accuracy is inconsistent; remaining points are sometimes insufficient.

## Critique (Why It Is Not Enough Yet)

- Sampling cost is far above budget; the method is accurate but not viable for production.
- N_B instability likely stems from mixed samples and probe origin bias; current clustering
  does not guarantee face-pure populations.
- The approach is strongly shape-tuned (rotated cube); generalization is unproven.

## Algorithm Flow

```
process_vertex(midpoint, accumulated_normal):
  1. locate_surface() → find surface position, track crossing directions
  2. measure_face() from surface position → if residual < 0.002, return Face
  3. Find outliers from face fit → derive second face hint
  4. measure_edge() with biased hemispheres from MIDPOINT (not surface)
     - If angle < 30°, fallback to measure_edge_full_sphere()
  5. If edge residual < 0.02 and angle > 30° → return Edge
  6. Try corner detection, then fallback logic
```

## Next Steps (from Attempt 0)

1. Reduce sample count (target <80): fewer full-sphere directions, skip unnecessary fallbacks.
2. Improve N_B accuracy: better second-plane fitting, iterative refinement.
3. Lower face threshold (0.002) as needed.
