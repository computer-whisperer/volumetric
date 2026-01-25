# Adaptive Surface Nets v2 - Architecture & Implementation

A modular implementation of the adaptive surface net algorithm for extracting
triangle meshes from binary volumetric data.

## Problem Statement

Given a volumetric sampling function `is_inside(x: f64, y: f64, z: f64) -> f32`:
- Returns `> 0.0` for points inside the model
- Returns `0.0` for points in empty space
- **NOT an SDF** - only the sign/threshold matters, not the magnitude

We must produce a triangle mesh that:
1. Is **airtight** (no holes or gaps) - unless the surface intersects the bounding box
2. Is **non-self-intersecting**
3. Has **consistent winding order** (all normals point outward)
4. May have non-manifold edges (4+ triangles sharing an edge is acceptable)

**Boundary Handling**: If the surface intersects the bounding box boundary, we return
an open surface. The frontier expansion simply stops at the edge of the box rather
than attempting to close the mesh artificially.

## Key Constraints

- **Minimize sampler calls**: `is_inside` is expensive (microseconds per call, increasing
  with geometry complexity). This is the primary optimization target.
- **Accuracy over vertex count**: Spend sample budget on vertex position refinement
  and normal estimation rather than raw triangle count.
- **Multithreading**: Leverage parallel sampling. Accept occasional redundant samples
  rather than complex synchronization. `is_inside` is deterministic.
- **Integer-based topology**: ALL position-derived IDs (CuboidId, EdgeId) MUST be
  computed from integer cell coordinates, NEVER from floating-point positions.
  This ensures deterministic vertex welding across adjacent cells.

## Canonical Corner Indexing System

We define a canonical mapping from corner index (0-7) to position offset within a cell.
This system is used consistently throughout the entire pipeline.

```text
Corner Index → (dx, dy, dz) offset from cell minimum corner:
  0 → (0, 0, 0)  "---"
  1 → (1, 0, 0)  "+--"
  2 → (0, 1, 0)  "-+-"
  3 → (1, 1, 0)  "++-"
  4 → (0, 0, 1)  "--+"
  5 → (1, 0, 1)  "+-+"
  6 → (0, 1, 1)  "-++"
  7 → (1, 1, 1)  "+++"

Bit encoding: index = (z << 2) | (y << 1) | x
  - Bit 0 (LSB): X offset (0 or 1)
  - Bit 1: Y offset (0 or 1)
  - Bit 2: Z offset (0 or 1)
```

The CornerMask uses this same bit ordering: bit N is set if corner N is inside.

## Canonical Edge Indexing System

Each cell has 12 edges. Edges are identified by:
- The minimum corner position (in finest-level integer units)
- The axis direction (0=X, 1=Y, 2=Z)

```text
Edge Index → (corner_a, corner_b, axis):
  Edges along X-axis (axis=0):
    0: corners 0-1, min=(x, y, z)
    1: corners 2-3, min=(x, y+1, z)
    2: corners 4-5, min=(x, y, z+1)
    3: corners 6-7, min=(x, y+1, z+1)
  Edges along Y-axis (axis=1):
    4: corners 0-2, min=(x, y, z)
    5: corners 1-3, min=(x+1, y, z)
    6: corners 4-6, min=(x, y, z+1)
    7: corners 5-7, min=(x+1, y, z+1)
  Edges along Z-axis (axis=2):
    8: corners 0-4, min=(x, y, z)
    9: corners 1-5, min=(x+1, y, z)
   10: corners 2-6, min=(x, y+1, z)
   11: corners 3-7, min=(x+1, y+1, z)
```

## Algorithm Overview (4 Stages)

### Stage 1: Coarse Grid Discovery

Sample the volume at low resolution to find regions containing the surface.
- Grid of cubic cells using `base_cell_size` (base cell count is derived per axis)
- Identify "mixed" edges (inside→outside transitions)
- Output: Initial work queue of coarse mixed cells WITH pre-sampled corners

### Stage 2: Parallel Subdivision & Triangle Emission

Process cuboids in parallel, subdividing and expanding the frontier:

```text
while work_queue not empty:
    work_item = work_queue.pop()  // includes CuboidId + [Option<bool>; 8] known corners
    corners = complete_corner_samples(work_item)  // only sample unknown corners
    if is_mixed(corners):
        if depth < max_depth:
            subdivide into 8 children
            for each child: propagate known corners, add to queue
        else:  // at max depth, emit geometry
            for each transitioning_edge:
                edge_id = compute_edge_id(cell_coords, edge_index)  // INTEGER ONLY
                emit_triangle(edge_ids, corner_states)
            for each neighbor (within bounds):
                if neighbor_is_mixed(corners):
                    work_queue.add_deduplicated(neighbor, propagated_corners)
```

**Key Concepts**:
- **Pre-packed corner samples**: Each work queue entry carries `[Option<bool>; 8]`
  with already-known corner samples. This replaces the sample cache entirely and
  is simpler, more effective, and avoids cache synchronization issues.
- **Deterministic Edge IDs**: Computed ONLY from integer cell coordinates.
  EdgeId = (cell_x + dx, cell_y + dy, cell_z + dz, axis) where dx/dy/dz are
  integer offsets from the canonical edge table. Adjacent cells sharing an edge
  will compute identical EdgeIds, causing automatic vertex welding.
- Triangles are stored with their sparse edge IDs for later index rewriting.
- Frontier expansion stops at bounding box edges (open surfaces allowed).
- Output: Collection of triangles with sparse edge-based vertex IDs

### Stage 3: Topology Finalization

Convert sparse edge IDs to monotonic vertex indices:
- Iterate over all unique edge IDs encountered, assign each a monotonic index (0, 1, 2, ...)
- Rewrite triangle indices from sparse edge IDs to monotonic vertex indices
- Accumulate face normals per vertex (sum of adjacent triangle normals)
- Output: IndexedMesh with proper indices and accumulated normals per vertex

Note: This is NOT deduplication - each edge ID is already unique. We are simply
converting from a sparse ID space (edge positions) to a dense monotonic space
(vertex buffer indices).

### Stage 4: Vertex Refinement & Normal Estimation (STUBBED)

The original Stage 4 implementation included:
- Binary search vertex position refinement
- Normal recomputation from refined positions
- Bayesian normal probing with plane fitting
- Sharp edge detection (Case 1 and Case 2)
- Vertex duplication for sharp edges

**Current Status**: Stage 4 is currently stubbed out as a passthrough. The Stage 3
output (edge midpoints with accumulated face normals) is used directly. See
`PHASE4_ARCHIVE.md` for details on what was removed and why.

## Winding Consistency & Ambiguous Cases

We use a lookup table approach similar to Marching Cubes, indexed by the 8-bit CornerMask.
This gives us 256 possible configurations.

**Winding Rule**: Triangles use CCW winding when viewed from outside the surface,
matching the project-wide convention documented in README.md. The face normal
(computed as cross(AB, AC)) points outward from the solid. This is achieved by
reversing the vertex order from the standard MC tables.

**Ambiguous Cases**: Some CornerMask values (e.g., 0x3C, 0x69, 0x96, 0xC3) have
topologically ambiguous configurations where the surface could be connected in
multiple valid ways. Our resolution strategy:
1. Use a fixed, deterministic choice for each ambiguous case in the lookup table
2. Prioritize configurations that avoid creating holes or self-intersections
3. The same choice is made consistently across all cells with the same CornerMask

This ensures global consistency: adjacent cells with compatible corner states will
always produce compatible triangle configurations.

## Integer Size Analysis

**CuboidId coordinates (i32)**:
- At max_depth, resolution = base_cells * 2^max_depth (per axis)
- Example: base_cells=8, max_depth=20: 8 * 2^20 = 8,388,608 cells per axis
- i32 range: ±2,147,483,647 → sufficient for ~256x this resolution
- **Verdict**: i32 is adequate for any practical use case

**EdgeId coordinates (i32)**:
- Same analysis as CuboidId - edges are at finest-level resolution
- **Verdict**: i32 is adequate

**Monotonic vertex index (u32)**:
- Maximum vertices ≈ number of transitioning edges
- Worst case: every cell at max depth has ~12 edges, but most are shared
- Realistic estimate: ~1-2 vertices per surface cell
- With 8M³ cells, surface area scales as N², so ~64 trillion surface cells max
- This exceeds u32! However, practical meshes are much smaller.
- **Verdict**: u32 sufficient for meshes up to ~4 billion vertices.
  For extreme cases, could upgrade to u64 indices.

**Triangle indices in output (u32)**:
- Standard GPU vertex buffer index type
- **Verdict**: u32 is the practical choice; matches GPU expectations

## Module Structure

```
src/adaptive_surface_nets_2/
├── mod.rs              # Public API, entry points, re-exports
├── types.rs            # All types, configs, output structures
├── lookup_tables.rs    # CORNER_OFFSETS, EDGE_TABLE, MC tables
├── parallel_iter.rs    # Conditional parallel iteration helpers
├── stage1.rs           # Stage 1: Coarse Grid Discovery
├── stage2.rs           # Stage 2: Subdivision & Triangle Emission
├── stage3.rs           # Stage 3: Topology Finalization
├── stage4_stub.rs      # Stage 4: STUBBED passthrough
├── diagnostics.rs      # Feature-gated diagnostic functions
└── tests.rs            # All unit tests
```

## Public API

```rust
// Main function
pub fn adaptive_surface_nets_2<F>(...) -> MeshingResult2

// Types
pub struct AdaptiveMeshConfig2
pub struct SharpEdgeConfig
pub struct MeshingStats2
pub struct IndexedMesh2
pub struct MeshingResult2

// Feature-gated diagnostics
pub fn run_normal_diagnostics(...)   // #[cfg(feature = "normal-diagnostic")]
pub fn run_edge_diagnostics(...)     // #[cfg(feature = "edge-diagnostic")]
pub fn run_crossing_diagnostics(...) // #[cfg(feature = "edge-diagnostic")]
```
