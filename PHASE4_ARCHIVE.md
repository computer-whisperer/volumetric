# Phase 4 Archive - Removed Vertex Refinement Implementation

This document archives the Phase 4 implementation that was removed from
`adaptive_surface_nets_2.rs` during the modular refactoring. The code
remains available in `src/old_adaptive_surface_nets_2.rs` for reference.

## What Was Removed

The original Stage 4 implementation included approximately 2,350 lines of code
implementing the following phases:

### Phase 4a: Binary Search Vertex Refinement

Moved vertices from edge midpoints to actual surface crossings by:
1. Searching along the accumulated normal direction
2. Falling back to cardinal axes (+X, +Y, +Z) if primary direction failed
3. Using 12-iteration binary search for precise positioning

**Key Functions Removed:**
- `find_crossing_along_direction()` - Find a crossing bracket along a direction
- `binary_search_crossing()` - Refine bracket to precise position
- `refine_vertex_position()` - Main vertex refinement entry point
- `RefineOutcome` enum - Track which search direction succeeded

### Phase 4b: Normal Recomputation

After vertex positions were refined, face normals were recomputed from the
new positions to improve accuracy.

### Phase 4c: Bayesian Normal Probing with Sharp Detection (Case 1)

For each vertex, probed the surface in tangent directions to:
1. Find nearby surface points using binary search
2. Fit a plane to the discovered points
3. Blend with the topology normal using inverse-variance weighting
4. Detect "Case 1" sharp edges where high residual indicated straddling

**Key Functions Removed:**
- `refine_normal_via_probing()` - Basic Bayesian normal refinement
- `refine_normal_via_probing_with_sharp_detection()` - With Case 1 detection
- `find_surface_point_along_direction()` - Find surface along probe direction
- `fit_plane_to_points()` - Least-squares plane fitting
- `orthonormal_basis_perpendicular_to()` - Compute tangent basis
- `try_sharp_edge_detection()` - Clustering-based edge detection
- `cluster_points_two()` - Split surface points into two clusters
- `try_split_by_direction()` - Helper for clustering
- `project_to_plane_intersection()` - Find edge line position

### Phase 4d: Case 2 Edge Crossing Detection

Identified mesh edges that crossed geometric sharp edges by:
1. Comparing endpoint normals of mesh edges
2. Computing crossing positions on the geometric edge (plane intersection)

**Key Functions Removed:**
- `detect_edge_crossings()` - Find Case 2 crossings
- `compute_edge_crossing_position()` - Compute crossing point
- `extract_unique_edges()` - Get unique edges from triangle indices
- `EdgeCrossing` struct - Store crossing information

### Phase 4e: Re-triangulation with Crossings

Split triangles that contained Case 2 edge crossings to properly
represent the sharp edge geometry.

### Phase 4f: Sharp Vertex Duplication

Duplicated vertices at sharp edges so that each face could have its
own normal direction, preventing smooth interpolation across edges.

**Key Functions Removed:**
- `stage4_5_sharp_edge_processing()` - Main sharp edge entry point
- `VertexSharpInfo` struct - Per-vertex sharp edge information
- `SharpStats` struct - Statistics for sharp edge processing

## Why It Was Removed

The sharp edge detection algorithms (4c-4f) were experimental research code
that did not achieve production-quality results:

1. **Probe-based detection was unreliable**: The Bayesian clustering approach
   required many probes per vertex and still had high false positive rates
   on smooth-but-curved surfaces.

2. **Position error compounded**: Binary search vertex refinement introduced
   position errors that were then amplified by normal probing, leading to
   edge detection artifacts.

3. **Computational cost was prohibitive**: Full sharp edge detection required
   256+ probes per vertex for the reference computation, which was too
   expensive for real-time use.

4. **Mesh topology didn't match edge geometry**: Even when edges were correctly
   detected, the mesh triangulation didn't align with the geometric edges,
   causing visual artifacts.

See `EDGE_REFINEMENT_RESEARCH.md` for detailed experimental results and analysis.

## What Remains

The stubbed Stage 4 (`stage4_stub.rs`) provides a simple passthrough that:
1. Takes the Stage 3 result unchanged
2. Converts f64 positions to f32
3. Normalizes the accumulated normals

This produces usable meshes with edge midpoint vertex positions and
accumulated face normals. While not as refined as the full implementation,
this approach is fast, deterministic, and produces consistent results.

## Future Directions

For improved sharp edge handling, consider:

1. **Sample-by-sample edge detection**: Detect edges during the initial
   sampling phases rather than as a post-process. See algorithms in
   `EDGE_REFINEMENT_RESEARCH.md`.

2. **Analytical edge hints**: For SDFs or other analytically-defined surfaces,
   the model itself could provide edge location hints.

3. **Normal discontinuity mesh**: Separate mesh topology from sharp edge
   handling by allowing per-face-corner normals in the output format.

4. **Resolution increase**: Higher subdivision depth naturally captures
   more edge detail, though at computational cost.

## Reference Code Location

The complete Phase 4 implementation can be found in:
- `src/old_adaptive_surface_nets_2.rs` (backup file, will be deleted after verification)

Key line ranges in the original file:
- Lines 1778-4132: Phase 4 implementation
- Lines 4154-7823: Diagnostic functions (feature-gated)
