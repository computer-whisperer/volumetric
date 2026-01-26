# Attempt 2 Notes

Date/Time: 2026-01-25 16:50 EST (reviewed)

> **WARNING (2026-01-25 21:00 EST):** Results below were obtained with incorrect
> scaling parameters (cell_size=1.0, fixed offset=0.1). The algorithm is NOT
> scale-invariant due to hardcoded absolute thresholds. All metrics need
> re-verification. See `EDGE_REFINEMENT_RESEARCH.md` for details.

## Summary
Fixed-budget, full-sphere RANSAC with crossing-count routing. Sampling is reused across passes.

## Results (rotated cube, latest)

- Face success: 12/12
- Edge success: 2/24 (often 0/24 depending on sampling variant)
- Corner success: 6/8
- Face normal avg error: ~0.00°
- Edge normal avg error: ~165° (classification failures)
- Edge direction avg error: ~165°
- Corner normal avg error: ~45°
- Avg samples used: ~1.4k–2.0k

## Diagnostics Findings

- Edge fits often show low residuals and ~90° separation, but one plane normal is diagonal/corner-like.
- Hint availability is high (22/24 edge points), yet edge sampling is not aligned with hints.
- `hint_dot_hist` shows almost no samples with strong alignment to hints (>=0.9 bin is 0), meaning probes are still too mixed.
- Tightened cone angles (10°) did not improve alignment.
- Offset origin along hint directions often yields zero edge samples; likely starting outside and missing crossings.
- Increasing `max_distance` did not recover edge samples from offset origins.

## Critique (Why It Failed)

- Fixed budgets stabilize face/corner fits but do not solve mixed-sample contamination.
- Hint-driven sampling still produces diagonal planes; sampling strategy is misaligned
  with edge geometry rather than a RANSAC instability.
- Offset-origin sampling must guarantee inside starts; single-sided offsets are brittle.

## Key Takeaways

- Edge failures are driven by mixed sample sets, not RANSAC instability.
- Hint-based sampling is not producing face-pure points; the geometry near edges yields diagonal planes.
- Offset-origin sampling needs to start inside the shape (or test both +/- hint) to guarantee crossings.

## Open Questions

- How to reliably generate face-pure samples near edges without oracle data?
- Should edge probing start from two inside offsets along +/- hint instead of from the surface point?
- Is a dedicated sampling strategy needed that explicitly rejects diagonal-plane samples?
