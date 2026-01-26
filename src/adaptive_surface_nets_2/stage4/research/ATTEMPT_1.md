# Attempt 1 Notes

Date/Time: 2026-01-25 16:50 EST (reviewed)

> **WARNING (2026-01-25 21:00 EST):** Results below were obtained with incorrect
> scaling parameters (cell_size=1.0, fixed offset=0.1). The algorithm is NOT
> scale-invariant due to hardcoded absolute thresholds. All metrics need
> re-verification. See `EDGE_REFINEMENT_RESEARCH.md` for details.

## Summary
Adaptive RANSAC + diagnostics with crossing-count preclassification.

## Results (rotated cube)

- Face success: 11/12
- Edge success: 16/24
- Corner success: 0/8
- Face normal avg error: ~61.6°
- Edge normal avg error: ~109.3°
- Edge direction avg error: ~79.6°
- Avg samples used: ~968

## Diagnostics Summary

- Edge points often get extremely low face residuals → face fit looks “perfect” even when it should be edge.
- Edge RANSAC frequently yields small separation angles (15–26°), indicating both planes latch onto the same face.
- Corner points are frequently classified as edges; corner RANSAC rarely “wins.”
- Reducing search distance or using full-sphere vs hemisphere sampling did not fix the core issue.

## Critique (Why It Failed)

- Adaptive sampling assumes the hint direction is reliable; edge vertices often have
  degenerate normals, so the sampler collapses onto a single face.
- RANSAC is fitting the sampling bias, not separating faces; this is a sampling geometry
  failure more than a fitting failure.
- Corner handling is insufficient; corners are repeatedly misclassified as edges.

## Conclusion
Attempt 1’s adaptive sampling strategy is fundamentally misaligned with the geometry. It tends to collapse onto a single face and does not reliably separate face populations, even with high sample counts. Continuing to tune Attempt 1 is likely diminishing returns.
