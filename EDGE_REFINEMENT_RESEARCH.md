# Edge Refinement Research (Living)

Updated: 2026-01-25 16:50 EST

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

- Binary fields break gradient-based normals. Probing must be geometric.
- Edge vertices often have degenerate accumulated normals due to transition triangles.
- Sampling from *inside* the surface yields much cleaner plane separation than sampling
  directly on the surface point ("thin shell" problem).
- Crossing-count during surface location is a strong edge signal: 2 crossings correlate
  with edge candidates in archived runs.
- RANSAC plane fitting is highly sensitive to inlier thresholds; tight thresholds
  are required to keep face samples pure.

---

## Critiques of Current Methodology (Actionable)

- Over-reliance on a single benchmark shape (rotated cube). Add cylinders and chamfers
  to prevent overfitting to planar edges.
- Per-vertex probe + RANSAC assumes mixed samples can be cleanly separated after the fact;
  most failures are actually caused by the sampling distribution.
- Sampling from the surface point yields coplanar clouds and mixed face populations.
- Lack of neighborhood coherence: edges are global features but are inferred locally.

---

## Attempt Status (Snapshot)

- Attempt 0: strong classification, unstable second normal, high sampling cost.
- Attempt 1: adaptive RANSAC collapses onto single face; false edges on corners.
- Attempt 2: fixed-budget RANSAC improves faces/corners but fails edges due to mixed samples.

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
