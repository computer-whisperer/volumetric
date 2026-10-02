# Mesher back half: target design

Status: design, written 2026-10-01 before any deletion. Nothing below is
built yet. Step 1 of the same arc (near-cubic cells, `MeshGrid`) has landed
and every number here was measured after it.

This file is the ground truth for the rework. If the design changes while
building, change this file first.

## Why

The mesher places the surface well. With sharp features on and decimation
off, the keychain sleeve mesh is within 6 µm of the model everywhere. The
defects come from what happens after placement, and mostly from the order
in which it happens.

Today, with sharp features on (the GUI and `render` default):

```
stage 4    refine vertices onto the surface
stage 4.5  fit planes -> grow regions -> snap -> weld -> CREASE SPLIT
stage 5    decimate -> re-accumulate normals
```

The crease split gives every crease vertex one copy per adjacent region so
each side can shade with its own normal. The copies share a position but
not an index, so to the decimator each side of a crease is a separate open
border. It collapses the two borders independently, they stop matching
vertex for vertex, and the mesh tears. Decimation then recomputes every
normal from the new triangles, which discards the split stage's normals and
smears whatever blends remain across triangles a hundred cells long.

Two patches already compensate for this order (the "pocket quarantine" in
`crease.rs`, the "shading pin" in `mesh_decimation.rs`). Both keep the
order and limit its damage. This design removes the cause instead.

Measured today, `mesh --sharp-edges --normal-refinement 0`, default
tolerance (sleeve and bumper 512, toy car 256):

| | open edges | non-manifold edges | folds > 135° | p99 deviation |
|---|---:|---:|---:|---:|
| sleeve, sharp, no simplify | 0 | 4 | 273 | 0.006 mm |
| sleeve, sharp + simplify | 4,289 | 5 | 67 | 0.144 mm |
| bumper, sharp, no simplify | 0 | 1 | 584 | |
| bumper, sharp + simplify | 8,601 | 10 | 88 | |
| car, sharp, no simplify | 0 | 75 | 1,847 | 0.056 mm |
| car, sharp + simplify | 5,894 | 41 | 401 | 0.356 mm |

Open edges are counted with vertices merged by position, so crease copies
do not count: every open edge is a real gap.

## Target

```
stage 4    refine vertices onto the surface                    (unchanged)
stage 4.5  fit planes -> grow regions -> snap -> weld         (crease split removed)
stage 4.6  classify feature edges on the welded mesh          (new)
stage 5    decimate one connected mesh; features constrain it (changed)
stage 6    split normals at feature edges                     (new, last)
```

The mesh stays one connected, index-shared surface from stage 4 until the
last step. Nothing before stage 6 duplicates a vertex, so nothing can tear.
Shading is derived once, from the final triangles.

All of this applies when `sharp_features` is on. With it off the pipeline
keeps today's behaviour exactly (no feature edges, smooth re-accumulated
normals, the shading pin). Lattices are meshed that way, and their output
should not change by a byte.

### Feature edges (stage 4.6)

An edge is a feature edge when the angle between its two faces exceeds the
crease angle. This is a property of the geometry alone. It does not use
region labels, so a crease whose two sides belong to the same smooth region
(the sleeve's keyring hole) is treated like any other.

- Classified once, on the fine welded mesh, where a curved surface turns a
  few degrees per edge and a crease turns tens. Classifying after
  decimation would be wrong: a 2.5 mm hole decimated to a 0.165 mm budget
  works out to about 42° between facets and would read as creases.
- Crease angle: one config value, default 30°. It replaces nothing in
  `SegmentationConfig`; the segmentation's 15° normal-jump gate keeps its
  own meaning.
- A vertex's class follows from its feature edges: 0 = smooth, 2 = crease,
  anything else = corner.
- Where snapping failed, the sawtooth is itself full of feature edges and
  its vertices are mostly corners. They stay put at cell pitch and each
  tooth shades flat. That is the honest picture of unresolved geometry, and
  it replaces the shading pin's role on this path.

Representation: a set of edge keys (the decimator's `edge_key`), plus the
per-vertex class. New module `src/sharp_features/feature_edges.rs`.

### Decimation with features (stage 5)

The decimator already has this mechanism for open borders: a border vertex
may only collapse along a border edge. Feature edges get the same rule.

- A corner never moves.
- A crease vertex may collapse only onto the neighbour at the other end of
  one of its feature edges.
- A smooth vertex may collapse onto anything, as now.
- No extra constraint planes are needed. A crease vertex already carries
  the planes of both faces, so sliding along a straight crease costs
  nothing and along a curved one costs its sagitta, which the budget
  bounds.
- The feature-edge set is maintained through collapses: when `a` collapses
  onto `b`, each feature edge `(a, c)` becomes `(b, c)`. (The border set is
  static today, which only lets a border shrink along original edges. Do
  the same maintenance for borders while there.)
- Link condition, flip guard, subset placement, the parallel region path:
  unchanged. They already assume one connected surface.

`decimate_mesh` takes the feature set as an argument and returns it,
renumbered to the surviving vertices. On this path the shading pin is not
computed.

### Normals (stage 6)

At each vertex, the faces around it are divided into fans by the feature
edges that meet there. Each fan gets one vertex copy whose normal is the
weighted sum of its faces' normals. A vertex with no feature edge has one
fan and is not copied.

- Runs last, on whatever mesh reaches it (decimated or not).
- Output format is unchanged: `IndexedMesh2` with duplicated vertices at
  creases, positions bit-identical. STL is a triangle soup, and the 3MF
  writer already merges identical positions (`TriMesh::from_soup`), so an
  exported sharp mesh is watertight.
- Weighting: corner-angle weighting, not area. After decimation one fan
  mixes triangles whose areas differ by orders of magnitude. To be
  confirmed against area weighting on the sleeve before it is fixed.

New module `src/sharp_features/normals.rs`.

## What gets deleted

- `src/sharp_features/crease.rs`, whole file: region-keyed copies,
  triangle-to-region assignment, the `MIN_RESOLVE_DOT` sweep, the pocket
  quarantine, the normal re-derivation.
- In `apply_sharp_features`: the label and normal carry-through for the
  split. Its output becomes welded positions, indices and the snapped
  flags.
- `SharpFeatureStats::crease_splits` and `MeshingStats2::sharp_crease_splits`
  keep their meaning (copies made) but are filled by stage 6.
- The test `fold_triangle_stays_region_less` and the other `crease.rs`
  tests go with the file.
- Kept: the shading pin and its two tests, for the sharp-off path only.

Delete first, then build. After the deletion and before stage 6 exists,
sharp meshes shade smooth across creases; that intermediate state is
expected.

## Order of work

1. Record the reference numbers below on the commit before the first
   deletion.
2. Delete `crease.rs` and its call. Sharp + simplify must already show 0
   open edges at this point (there are no copies left to pull apart). This
   is the cheap test of the diagnosis: if edges still open, the tearing has
   another cause and the design stops here.
3. Feature-edge classification, with tests on a cube (12 edges, 8 corners),
   a cylinder (two rims, no corners), and a sphere (none).
4. Stage 6 normals. Sharp without simplify must look as it does today.
5. Decimation rule and feature-set maintenance.
6. Measure against the table below, render the close-ups, update the
   README and the module docs.

## Done means

Same commands as the table above.

| | today | target |
|---|---:|---:|
| open edges, sharp + simplify (sleeve / bumper / car) | 4,289 / 8,601 / 5,894 | 0 / 0 / 0 |
| non-manifold edges, sharp + simplify | 5 / 10 / 41 | no more than sharp alone (4 / 1 / 75) |
| streaks from the lug hole in `render` close-ups | present | gone |
| sharp, no simplify: p99 deviation (sleeve) | 0.006 mm | unchanged |
| sharp off: lattice_test output | reference | byte-identical |
| cube and sphere triangle counts after simplify | reference | within 10 % |

A 3MF of the sleeve exported with sharp + simplify must pass
`superslicer --info` as one manifold part.

## Not in this design

- Snapping a crease whose sides share a region (the hole rim stays a
  sawtooth; this design only makes it shade and decimate sanely). That is
  the next step: find the two sides by local connectivity inside the
  gather ball instead of by global label.
- The debris the snap and weld leave behind without decimation (the
  non-manifold edges, zero-area triangles and folds in the "no simplify"
  rows, and the spikes on the car). Zero-area triangles should fall to the
  first decimation pass once the mesh is connected; the rest needs its own
  look at the weld.
- Decimation defaults. Tolerance is one cell, which on the sleeve costs
  1.9 % of the volume; 0.05 cell costs 0.11 % for 8 % more triangles. No
  triangle-shape term, so half the output triangles are needles. Both are
  contained changes and independent of the order fixed here.
- Using feature edges on the sharp-off path to replace the shading pin
  there too. Possible once this is measured, not assumed.

## Open questions

- Crease angle default (30° proposed). A strut two cells in radius turns
  28° per edge, so anything thinner reads as creased. That only matters
  with sharp features on, where such geometry is already unreliable.
- Whether corners should be exactly "not two feature edges", or should
  also include vertices the snap stage placed as corners. Start with the
  geometric rule alone.
