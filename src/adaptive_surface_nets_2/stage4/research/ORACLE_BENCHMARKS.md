# Oracle Benchmark Requirements

This document defines the oracle benchmark system for edge/feature detection.
The oracle is the ground truth reference; it must be **independent** from any
attempt implementation and must remain correct even if slow.

## Goals

- Provide a reliable, theoretically correct reference for face/edge/corner
  classification and normals on binary samplers.
- Enable comparisons across attempts with consistent metrics and datasets.
- Scale from analytical primitives (e.g., rotated cube) to richer shapes
  (cylinders, chamfers, CSG, meshes) without changing benchmark semantics.

## Non-Goals

- The oracle is **not** used inside attempt implementations.
- The oracle is **not** optimized for performance.

## Hard Constraints

- Oracle classifiers must be geometry-derived, not probe-derived.
- Attempts must **never** receive oracle hints or ground truth data at runtime.
- Benchmarks must be reproducible (deterministic seeds, stable sampling).

## Oracle Responsibilities

For a query point P (near surface), the oracle returns:

- classification: Face | Edge | Corner | Unknown
- surface_position: exact closest point on the surface
- normals:
  - Face: 1 normal
  - Edge: 2 face normals
  - Corner: 3 face normals
- edge_direction (if Edge)
- corner_position (if Corner)

The oracle must be defined per shape using analytical geometry or exact CSG
geometry. If a shape is defined procedurally, the oracle still must compute
exact intersections and normals rather than sampling.

## Benchmark Requirements

- **Two modes**:
  - `oracle_only`: oracle generates expected values and metrics.
  - `attempt_compare`: attempt output is compared to oracle.
- **No hints** from oracle to attempts by default.
- **Sampling budgets** must be reported as actual sampler calls (cache misses).
- **Metrics** per classification:
  - classification accuracy
  - normal error (avg, p95, max)
  - edge direction error (avg, p95, max)
  - sample counts (avg, p95, max)

## Dataset Requirements

- Fixed validation points per shape (deterministic seed).
- Randomized rotations/scales per run only in stress mode.
- Points must include face, edge, and corner regions at multiple offsets.

## Separation Guarantees

Attempts must not access:

- oracle classifiers
- oracle normals
- oracle surface positions
- validation labels

The only shared input is the sampler function (binary field).

## Future Shapes

The oracle must support additional shapes without changing attempt APIs:

- primitives: sphere, cylinder, cone, chamfered box
- analytical CSG: unions/intersections/differences
- imported meshes with precomputed analytic planes (if available)

