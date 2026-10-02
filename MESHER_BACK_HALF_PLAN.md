# Mesher back half: target design

Status: built, 2026-10-02. The design below was written on 2026-10-01
before any deletion and is kept as written, except where "As built" at the
end says the build departed from it. Results are there too.

Step 1 of the same arc (near-cubic cells, `MeshGrid`) landed first, and
every "today" number in the design sections was measured after it, at the
decimation budget of one cell that was the default then.

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
  (Checked: area weighting is the right one. See "As built".)

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

## As built

Commands: `mesh --sharp-edges` with and without `--no-simplify`, defaults
otherwise (decimation budget 0.1 cell, no normal probing); sleeve and
bumper at 512, toy car at 256, cube, sphere and tray at 256.

### Results

Sharp + simplify, before (crease split ahead of decimation) and after:

| | triangles | open edges | non-manifold edges | needles (< 1°) |
|---|---:|---:|---:|---:|
| sleeve | 7,188 → 1,796 | 4,262 → 0 | 5 → 4 | 2,110 → 247 |
| bumper | 12,874 → 2,976 | 8,699 → 0 | 12 → 0 | 2,172 → 228 |
| toy car | 28,850 → 21,592 | 5,496 → 0 | 52 → 59 | 1,739 → 991 |
| cube | 2,770 → 12 | 2,192 → 0 | 0 → 0 | 1,872 → 0 |
| sphere | 10,104 → 10,104 | 0 → 0 | 0 → 0 | 0 → 0 |
| tray | 10,958 → 3,780 | 5,169 → 0 | 15 → 4 | 1,254 → 164 |

- Non-manifold edges stay at or below sharp without simplify (4 / 1 / 75 /
  0 / 0 / 4).
- The kill-test in step 2 of the order of work passed: with only the
  crease split deleted, all six models already had 0 open edges.
- `superslicer --info` on the sharp + simplify 3MF: sleeve and bumper
  manifold, one part each.
- Sharp off: lattice_test (edge-constrained, with and without simplify),
  smooth sleeve and smooth car are byte-identical to before (triangles in
  canonical order).
- Sharp without simplify: geometry unchanged; zero-area triangles 17 / 2 /
  6 / 38 (sleeve / bumper / car / tray) → 0.
- Surface area more than one cell off the model, and area facing the wrong
  way, are unchanged within noise (each at most 0.03 % on every model).
- The streaks from the sleeve's lug hole are gone and the bumper's hole no
  longer has gashes (close-up renders).
- Lattice and fractal models with sharp on still decimate, and no longer
  tear: lattice_test 9.7M → 4.9M triangles with 0 open edges, mandelbulb
  1.37M → 1.22M, gyroid 1.31M → 29k. (Sharp on an under-resolved lattice
  is still unreliable in other ways: 25k non-manifold edges there.)

The "cube and sphere within 10 %" criterion was written as a guard against
a blow-up. The sphere is identical; the cube went from 2,770 to its 12.

### Where the build departed from the design

1. **Fan normals are area-weighted, not angle-weighted.** At the unsnapped
   hole rim a flat-side vertex shares its fan with sub-cell tilted facets
   and is also a corner of triangles a hundred cells long. By angle the
   facet tilts the normal and each long triangle shows a streak; by area it
   is outvoted. Measured on the sleeve.
2. **A sharp turn of a crease is a corner too** (`feature_kinks` in
   `mesh_decimation.rs`; answers the second open question). Counting
   feature edges alone lost box corners: snap debris can hide one of the
   three creases at a corner, the corner then has two feature edges, and it
   slid away along one of them. A crease vertex whose two feature edges
   leave it less than 135° apart is frozen. This also holds the sawtooth of
   an unsnapped feature in place, which the design left to "mostly
   corners".
3. **Zero-area caps are flipped away in the weld** (`flip_zero_area_caps`
   in `cleanup.rs`). The design expected decimation to absorb zero-area
   triangles. It cannot: their edges have no angle, so they are classified
   as feature edges (otherwise they would join the faces either side of a
   crease into one fan), and that pins their vertices. All of the ones the
   weld leaves are three collinear vertices along a crease; flipping the
   long edge removes them without changing the surface.
4. **Border edges are not maintained through collapses.** The design said
   to do it "while there". It would change sharp-off output on open
   meshes, and that path was to stay identical.
5. **Normals are re-derived only near features.** Stage 6 recomputes the
   normals of feature vertices and of vertices sharing a triangle with one,
   and keeps the rest (which may be probe-refined, or decimation's).

### Still open

- Snap debris: the folds and non-manifold edges in the sharp meshes, the
  spikes on the car, and corners cut by up to two cells where the snap
  leaves a folded pocket (seen on an off-grid box at 64). Decimation now
  neither grows nor removes these.
- Creases whose two sides share a region are still not snapped (the lug
  hole's rim is a clean-shaded sawtooth). Done since: see "Next step: local
  sides in the snap stage" below.
- Needles: no triangle-shape term in decimation.

## Next step: local sides in the snap stage

Status: built, 2026-10-02. The design below was written before the first
edit and is kept as written; "As built: local sides" after it has the
results and the three places the build departed from it.

### Why

The snap stage moves a feature-zone vertex onto the line where two smooth
faces meet. It finds the faces by gathering the nearby vertices that
segmentation assigned to a smooth region, grouping them **by region
label**, and fitting one plane per label (`gather_side_planes` in
`sharp_features/snap.rs`).

A region label is global. Region growth steps from vertex to vertex wherever
the fitted normals turn by less than 15°, so two faces that meet at a sharp
crease in one place get the same label if they are joined smoothly anywhere
else. Then the crease has one label on both sides, the two faces are pooled
into one "side", its plane fit fails, and the vertex is left where it was.

Measured on the keychain sleeve at 512 (`wasm_mesh_view --debug-corner` at
three points round the lug hole's top rim): the hole's wall and the lug's
top face both carry region 0, the two-vertex band between them is unclaimed,
and none of it snaps. The whole mesh has 435 candidates rejected for having
fewer than two sides. Any part with an edge that is filleted along some of
its length and sharp along the rest has the same problem.

### Target

A side is local: a set of nearby on-face vertices that are connected to
each other *inside the gather ball* without crossing a crease.

1. **One home for the growth gate.** `segmentation.rs` gets a function for
   "these two fitted vertices may belong to one face" (fitted normals
   within `max_normal_jump_deg`). Region growth uses it, and so does snap.
2. **Sides by local connectivity.** In `gather_side_planes`, the claimed
   vertices inside the ball are split into connected components over the
   mesh edges that pass the growth gate. Each component is a side: plane
   fit, support gate and residual gate as today. Labels are only used to
   tell claimed from unclaimed.
   Two claimed neighbours have different labels only if the gate fails
   between them, so local components are always a refinement of the labels:
   nothing that was two sides becomes one.
3. **Candidates.** Today: unclaimed vertices, plus claimed vertices with a
   neighbour of a different label (a grid-aligned crease leaves no unclaimed
   band). The second rule becomes "a claimed neighbour the growth gate
   fails against", which covers the same-label case too.
4. Everything after the sides (intersection, movement clamp, refinement
   against the sampler, verification, weld, feature edges, decimation,
   normal split) is unchanged.

`snap_feature_vertices` takes the fits and the segmentation config in
addition to the labels.

### The one known risk, and how it is decided

Splitting can also separate two pieces of the *same* face that today pool
under one label, when they are connected only outside the ball (a groove
narrower than the ball, an unclaimed streak across a face). Two such pieces
could each fall under the six-vertex support gate, or be picked as the two
best-supported sides and rejected as parallel.

This is measured before anything is built on it: dump the per-vertex snap
outcome from the old and the new code on the six reference models and count
the vertices that snapped before and no longer do. If that count is not
negligible, pieces are merged again where they fit one plane within the
side residual gate; the first candidate rule for that is "pool by label
first, split a label only when its pooled fit fails".

### Done means

- The lug hole's rim on the sleeve snaps (512), and the count of candidates
  rejected for sides drops.
- No vertex that snapped before stays unsnapped, bar a negligible count
  explained case by case.
- Reference table (six models, sharp and sharp + simplify): open edges stay
  0; non-manifold edges, folds and off-surface area do not grow beyond what
  the newly snapped creases account for; volume moves toward the model.
- Sharp-off outputs stay byte-identical (this stage does not run there).
- Tests: a part-filleted crease whose two faces share one label snaps onto
  the crease line; a single-label crease with no unclaimed band snaps.

### Not in this step

- Small-radius curved sides. A side is a plane fit with a residual gate of
  0.1 cell over a three-cell ball, which a cylinder passes only above a
  radius of about 14 cells. The lug hole is 15 cells at 512 and 7.5 at 256,
  so at 256 its rim is expected to stay unsnapped for that separate reason.
- Snap and weld debris, needles (see "Still open" above).

### As built: local sides

Commands as for the back half: `mesh --sharp-edges` with and without
`--no-simplify`; sleeve and bumper at 512, the rest at 256. Per-vertex snap
outcomes come from `wasm_mesh_view --debug-corner 9,9,9,0 --dump-snaps
<file>` run with the build before and after.

#### Results

Snapped vertices, before → after, and what changed per vertex:

| | snapped | no longer snapped | newly snapped | same vertex, other target |
|---|---:|---:|---:|---:|
| sleeve | 9,673 → 9,876 | 0 | 203 | 0 |
| bumper | 16,573 → 19,439 | 0 | 2,866 | 4 |
| toy car | 12,391 → 12,981 | 0 | 590 | 2 |
| cube | 6,000 → 6,000 | 0 | 0 | 0 |
| tray | 11,061 → 11,085 | 0 | 24 | 0 |

(The six "other target" vertices were edge snaps and are now corner snaps:
a third side appeared.)

- The lug hole's top rim on the sleeve snaps all the way round (193 of the
  203; close-up renders before and after: ragged rim, clean circle).
  Candidates rejected for sides on the sleeve: 435 → 239.
- The bumper's gains lie along straight creases at three heights (z = 3.3,
  10.2 and 13.6 mm), down its long sides and across its ends, that were left
  as sawtooth before.
- Reference table, sharp and sharp + simplify: open edges 0 throughout;
  non-manifold edges unchanged on every model (sleeve 4, bumper 1 / 0, car
  75 / 59, tray 4); cube and sphere unchanged; the four sharp-off outputs
  byte-identical.
- Hard folds rise with the snapped count: sleeve 273 → 318, bumper 584 →
  624, car 1,846 → 1,880, tray 356 → 376 (sharp, no simplify). On the
  sleeve all 45 new ones are on the newly snapped rim, which now has 44;
  the three rims that already snapped have 36, 46 and 48. A snapped curved
  rim carries that many folds today wherever it is; that is the snap and
  weld debris still open above, not something new.
- Largest deviation on the sleeve 0.08 → 0.37 cell (99th percentile
  unchanged at 0.025). Bumper sharp + simplify volume 5,705.6 → 5,716.3
  against 5,711.0 for the undecimated mesh: from 0.09 % under to 0.09 %
  over. Not chased.
- At 256 the lug hole's rim is unchanged: three of the seven or eight
  feature-zone vertices at each point checked snap, before and after. The
  hole is 7.5 cells in radius there and its wall fails the plane fit, as
  "Not in this step" expected.

#### Where the build departed from the design

1. **Sides are pooled by label first and split only when the pooled fit
   fails.** Splitting every label into local pieces (the design) gained the
   same creases but lost 67 snapped vertices on the car and 15 on the tray:
   one face broken into pieces inside the ball, each short of support or
   picked as a parallel pair. This was the fallback the design named.
2. **A piece of a split region is a side only if its plane passes within
   the movement clamp of the vertex.** With label-first sides the car still
   lost 12: a piece of a face three cells away, with the most support, was
   picked ahead of the near side. Applying the same test to every side (not
   only pieces) was tried and rejected: it gains about 950 more snaps on the
   car and takes its non-manifold edges from 75 to 160.
3. **A snap that uses a piece must find every side's surface at the target**
   (`sides_meet_at` in `snap.rs`). At this scale a tight fillet looks like a
   crease. On the tray, pockets end in a fillet of about four cells; its two
   flanks came out as two sides, their planes meet 1.4 cells beyond the
   fillet, on the top face, and the rim vertices were pulled there (four new
   non-manifold edges, 45 folds, from 13 snaps). The existing check along the
   mean normal passes because the top face does run through that point. The
   new check probes each side separately, on whichever side of the other
   faces its surface is (behind them at a convex edge, in front at a concave
   one). With it the tray's non-manifold edges are back to 4.

   Applying this check to every snap, not only to pieces, would reject 196
   (sleeve), 375 (car) and 282 (tray) vertices that snap today and takes the
   tray's folds from 356 to 249. That is a lead for the debris work, not
   part of this step.

#### Also noticed

- `refine_target` shifts its probe line to the material side of the other
  sides. That is right at a convex edge. At a concave edge the probe then
  runs inside the other face's material, finds no boundary, and that side
  contributes no correction. On the sleeve about half of all snaps (4,867
  of 9,673) have a side that goes unrefined; with the shift tried both ways
  196 do. Flat faces lose nothing by it; a concave edge against a curved
  face keeps the plane fit's secant error. `sides_meet_at` tries both
  shifts; refinement could do the same.

## Next step: snap and weld debris

Status: built, 2026-10-02. The design below was written after the
measurements it quotes and before the rebuild, and is kept as written; "As
built: debris" after it has the results and where the build departed from it.
(The measurements came from throwaway experiments in the working tree.)

### What the debris is

"Debris" is what a sharp mesh carries that the model does not have: hard
folds (two neighbouring triangles more than 135° apart), triangles facing
into the material, edges shared by more than two triangles, and the spikes
and cut corners these show up as. After the last step the reference models
had 318 / 624 / 1,880 / 376 folds (sleeve / bumper / toy car / tray; sharp,
no simplify) and 4 / 1 / 75 / 4 non-manifold edges.

Every fold and non-manifold edge on those four models was sorted by what
the snap stage had done to the vertices around it (`wasm_mesh_view
--dump-stages`, then a script over the dump). There are four causes, and
one experiment per cause confirmed each:

1. **Slivers along a crease** (79–87 % of the folds on sleeve, bumper and
   car; 28 % on the tray). Both rows of the band between two faces are
   snapped onto the crease line. Where two of them land more than the weld
   radius apart they stay separate vertices, and the band triangle between
   them has all three corners on the crease. It has next to no area and an
   arbitrary normal. The weld already removes these when they have exactly
   zero area (`flip_zero_area_caps`); on a curved rim, or with the last
   digits of the bisection, they have a little. Treating every triangle with
   all three corners snapped and a height under the weld radius the same way
   took the folds from 328 / 613 / 1,913 / 506 to 71 / 114 / 522 / 384. The
   3–4 % that were left were cut off by the function's eight-round limit:
   it flips at most every second or third sliver of a chain per round.
2. **Triangle pairs folded onto each other** (67 of the car's 73
   non-manifold edges, and all of the others'). The weld joins two snapped
   vertices that were not connected by an edge, and the two triangles
   between them end up on the same three vertices, facing opposite ways.
   Removing such pairs took the non-manifold edges from 2 / 1 / 73 / 4 to
   0 / 0 / 6 / 0.
3. **Targets the model does not have.** A side's fitted plane can be
   extended past the end of its face. The clearest case is the tray's
   pocket corners, seen from above: a vertex on the fillet arc gets the
   straight wall beyond the tangent point as a side and is moved 1.3 cells
   onto the extension of that wall, into the top face. The existing check
   (inside just behind the target, outside just in front, along the mean of
   the side normals) passes, because the target is on the top face. This is
   the false corner of the last step again, for sides that are not pieces.
   Requiring every side's surface to be found at the target removed all 74
   folds of this kind on the tray and rejected 231 / 9 / 729 / 280 of the
   snaps.
4. **Snaps that overtake a neighbour.** A vertex is moved up to 1.5 cells,
   and its neighbours move differently or not at all, so it can cross the
   far edge of one of its own triangles and turn that triangle over. Two
   common shapes: a vertex one row back from the crease that was a snap
   candidate while its row-mates were not; and a vertex pulled to a corner
   past a nearer vertex that only reached the edge. Undoing the snaps that
   turn a triangle over (the vertex that moved furthest, repeated until
   nothing is turned) took triangles facing into the material from
   28 / 41 / 250 / 105 to 1 / 5 / 1 / 5 and the folds to 8 / 23 / 37 / 10,
   for 17–33 / 33–37 / 215–238 / 83–116 snaps undone. It does not cascade.
   What remains is beside the slivers cut off in cause 1.

### Target

Two changes, one in each stage.

**Snap: a target is where the model's surfaces meet, or it is rejected.**
`refine_target`, the mean-normal check and `sides_meet_at` become one step,
`locate_on_sides`. Each side is measured: the occupancy boundary is bisected
along the side's normal, on a probe line shifted off the target to whichever
side of the other faces has this face's surface alone (behind them at a
convex edge, in front at a concave one). The target moves to where the
measured surfaces meet.

- A side with no boundary inside the bracket has no surface there, and the
  snap is rejected. That is cause 3.
- A measured target beyond the movement clamp is rejected. (Today the
  unmeasured plane target is kept in that case.)
- Concave sides are measured like convex ones. Today about half of all
  snaps have a side that is never refined (see "Also noticed" above).
- The mean-normal check is deleted. With every side measured it rejected
  one or two vertices per model in the experiment.
- `SidePlane::split` and `SnapConfig::verify_delta_cells` go with it. The
  probe that orients a side normal keeps its 0.6 cell as a constant.

**Weld: the snap's result is made into a valid mesh, in four steps.**
`cleanup.rs` is rebuilt around this; the present `weld_snapped_vertices`
and `flip_zero_area_caps` are deleted first.

1. *Retract.* Snapped vertices within the weld radius are clustered as
   today. Then every stage-4 triangle that touches a snapped vertex is
   compared with itself before the snap. If it has turned by more than 90°
   and is not about to be removed by steps 2–4, the snapped vertex of it
   that moved furthest is put back where the mesher had it. Cluster and
   check again until nothing is turned. This is the last gate of the snap's
   contract: a snap that cannot be fitted into the mesh is not made.
2. *Weld.* One vertex per cluster, triangles with two corners in one
   cluster dropped. As today.
3. *Cancel.* Two triangles on the same three vertices with opposite
   winding are both removed. Every edge of the pair loses two faces, so no
   edge is opened.
4. *Flip caps.* A cap is a triangle with a snapped vertex within the weld
   radius of the opposite edge, and not within that radius of either end of
   it: the vertex is on that edge, and the triangle is what is left of a
   face that the snap flattened. The edge is flipped, so the cap and the
   face across the edge become two triangles that cover that face and meet
   at the vertex. This is `flip_zero_area_caps` with "zero area" replaced by
   "within the weld radius", which is the same tolerance the weld uses for
   two vertices being one. It covers cause 1 and the thin turned-over
   triangles of cause 4 (a vertex pulled just across an edge of a face it
   lies in). It runs from a work list over an edge map of the triangles
   near snapped vertices, so a chain of any length is finished, and a flip
   that would turn a triangle over is not made.

### To be decided by measurement

- **Corner capture.** When a vertex is pulled to a corner from distance
  *d*, every edge-snapped vertex whose target is within *d* of that corner
  would go to the corner too, so that order along the edge is kept. In the
  experiment this kept more corner snaps but did not change the folds, and
  the retraction still lost 7 of the tray's 38 corners with it (9 without).
  Those corners are where three faces meet around a notch, and what turns
  over there is a thin triangle of the kind step 4 now flips. Measure the
  corners kept with step 4 in place, with and without capture, and keep
  capture only if it saves corners.

### Done means

Same commands and models as before.

- Reference table, sharp and sharp + simplify: open edges 0; non-manifold
  edges at most 0 / 0 / 2 / 0; folds on sharp, no simplify, under a tenth
  of today's.
- Triangles facing into the material (new column, from the stage dump): at
  most a handful per model.
- No corner of the cube, sleeve or tray is lost: the count of distinct
  corner points does not drop except where a lost corner is shown to be a
  false one.
- Snaps lost against today are accounted for by the two rejections above,
  with a sample of them looked at.
- Sharp-off outputs byte-identical.
- Tests: a sliver chain along a curved crease is flipped away, whatever
  its length; a cancelling pair is removed and the mesh stays closed; a
  target on the extension of a wall past a fillet is rejected; a concave
  side is measured; a snap that overtakes a fixed neighbour is retracted
  and its neighbours' snaps stay.

### Not in this step

- Small-radius curved sides, needles, the remote daemon (see above).
- Snapping more of what is rejected today. The car at 256 has about as
  many unsnapped candidates as snapped ones (features one or two cells
  thick); that is resolution, not debris.

### As built: debris

Two instruments. The reference table is the `mesh` CLI's STL output, as
before. The stage counts come from `wasm_mesh_view --dump-stages` and
`crates/meshing_lab/scripts/sharp_stages.py` (new, see its header), run on
the build before and after; they see each vertex's snap outcome, which an STL
does not. The two count folds slightly differently (the tray: 38 and 22).

#### Results

Reference table, before → after:

| | non-manifold edges | folds > 135° | needles (< 1°) | triangles |
|---|---:|---:|---:|---:|
| sleeve, sharp | 4 → 0 | 318 → 8 | 341 → 1 | 918,154 → 918,150 |
| sleeve, sharp + simplify | 4 → 0 | 154 → 2 | 281 → 181 | 1,716 → 1,362 |
| bumper, sharp | 1 → 0 | 624 → 3 | 468 → 1 | 708,196 → 708,208 |
| bumper, sharp + simplify | 0 → 0 | 292 → 3 | 238 → 69 | 3,054 → 2,476 |
| toy car, sharp | 75 → 1 | 1,880 → 56 | 2,161 → 25 | 300,448 → 300,514 |
| toy car, sharp + simplify | 59 → 1 | 1,158 → 15 | 992 → 184 | 21,662 → 18,690 |
| tray, sharp | 4 → 0 | 376 → 38 | 107 → 9 | 335,502 → 335,510 |
| tray, sharp + simplify | 4 → 0 | 265 → 24 | 173 → 131 | 3,790 → 3,128 |

- Open edges 0 throughout. Cube and sphere unchanged (the cube is still 12
  triangles). The four sharp-off outputs are byte-identical.
- `superslicer --info` on the sharp + simplify 3MF: all four are manifold and
  one part. Before, the tray came out as two parts and the car as not
  manifold (15 open edges, three parts).
- The sharp stage takes 347 → 209 ms on the sleeve, 130 → 109 ms on the
  tray and 111 → 155 ms on the car (median of five). Total samples are up
  2 % on the sleeve and 6 % on the car.
- Decimated meshes are 14–21 % smaller on the four part models. The likely
  reason, not measured: a sliver's edges are feature edges, so the crease
  vertices beside it had more than two and were held as corners.
- Volume after decimation is closer to the undecimated mesh: sleeve 0.14 %
  under → 0.11 % under, bumper 0.09 % over → 0.01 % under.
- Largest deviation on the sleeve 0.37 → 0.17 cell. The car's 99th
  percentile 0.23 → 0.17 cell, and its sample points with no model surface
  within four cells (the spikes) 5 → 1 of 3,000.

Stage counts, before → after:

| | snapped | retracted | facing inward | slivers | corner points |
|---|---:|---:|---:|---:|---:|
| sleeve | 9,876 → 9,853 | 22 | 55 → 2 | 449 → 4 | 22 → 22 |
| bumper | 19,439 → 19,413 | 25 | 96 → 1 | 674 → 16 | 6 → 5 |
| toy car | 12,981 → 12,620 | 404 | 488 → 16 | 3,232 → 31 | 74 → 75 |
| tray | 11,085 → 10,965 | 71 | 205 → 8 | 290 → 12 | 38 → 36 |
| cube | 6,000 → 6,000 | 0 | 0 → 0 | 0 → 0 | 8 → 8 |

- Snaps lost to the side measurement: 1 / 1 / 56 / 89. The tray's are the
  pocket-corner targets of cause 3, in groups of three round each pocket.
- Snaps kept with a different target: 294 / 1,923 / 2,968 / 391 moved by more
  than 0.05 cell, and 0 / 6 / 212 / 181 by more than a quarter cell. The
  tray's 181 are along vertical edges at its right-hand end, where one wall
  curves away from the edge: the plane fitted to that wall put the old
  target a quarter cell out in the open.
- What is left facing inward is thin (under a quarter cell): caps whose flip
  was refused.

#### Against "Done means"

Met: open edges; non-manifold edges (0 / 0 / 1 / 0); triangles facing inward;
sharp-off outputs; the cube's and the sleeve's corners; every test listed.
Folds are under a tenth of before on three models and at a tenth on the tray.

Not met: **corners**. Three on the tray, one on the bumper and three on the
car are gone (others appeared: the car's total went up by one). They are real
corners, each made by a single corner snap, and each now ends about half a
cell short. The sequence at the one followed through (tray, a notch in the
outer wall): a vertex one row back from the notch has only two qualifying
sides, and they give it an edge a full cell away instead of the one it is
0.4 from; that snap turns a triangle and is retracted; with it back in
place, a triangle of the corner vertex beside it is turned, and the corner
snap is retracted too. Open.

#### Where the build departed from the design

1. **A side is measured with a probe that adapts.** The design took over
   the fixed probe of `sides_meet_at` (a quarter cell off the target, three
   quarters deep either way). Requiring that of every snap lost real
   creases, found by sampling the model where the lost snaps were:
   - *Edges sharper than a right angle.* The sleeve has a 65° edge, 16 mm
     long. Behind either face the material is a thin wedge, and a probe
     three quarters of a cell deep comes out through the other face. All
     196 snaps along it and its two corners were rejected. Now each end of
     the probe is halved until it is in material (or in the open), and the
     shift is taken within the side's own plane.
   - *Targets outside the edge.* Where a wall curves away below the edge,
     its fitted plane is tilted and the plane intersection lies up to a
     third of a cell outside the real edge. The other face is then not
     under the target, nor under a quarter-cell shift from it. Now shifts of
     a half and a whole cell are tried as well. (The old code accepted such
     targets where they were: off the surface.)

   The estimate in the last step's record, that this check would reject 196
   / 375 / 282 snaps and that those were debris, was wrong about what they
   were: most were real creases the probe could not see.
2. **The check for turned triangles runs on the welded positions**, inside
   the cleanup stage, with the weld's clusters recomputed after each round of
   retractions. The design had the same order; the point is that it shares
   the weld's own test for "will be dropped" and the flip's own test for "is
   a cap", so the three steps cannot disagree about a triangle.
3. **Cap flips end by never remaking an edge a flip has removed.** Four
   snapped vertices on one line, with two flattened triangles back to back,
   trade places for ever otherwise. (A first version with a flip budget
   spent it there and left other caps unflipped; a second, requiring each
   flip to shorten the caps' long edges, refused the chains of flips that
   carry a vertex across a fan of thin triangles.)
4. **A corner that is not found in the model falls back to the edge** of the
   two best-supported sides, as a corner out of range already did. Sound,
   tested, and without measurable effect on the reference models.
5. **Corner capture was not kept.** With the cap flip in place it saved two
   corners (one on the tray, one on the bumper) of 139, left the folds
   unchanged, and raised the car's retractions from 143 to 168.
6. `cleanup::weld_snapped_vertices` is now `cleanup::clean_up_snaps`, and
   reports the retracted vertices. `SharpFeatureStats` and `MeshingStats2`
   count them, and count as snapped only what stays snapped.

#### Still open

- The seven corners above. The root is the choice of sides: the two
  best-supported, not the two nearest.
- Caps whose flip is refused because the face across the long edge is as
  thin as the cap (36 on the car, 11 on the tray, 6 on the sleeve, 1 on the
  bumper): they are what is left facing inward.
- Edge snaps that move more than a cell (129 / 26 / 1,087 / 278 kept): rows
  behind the band collapsing onto the crease. Harmless where they are kept,
  and the source of most retractions (22 of 22, 17 of 25, 239 of 404 and 68
  of 71 retracted snaps had moved more than three quarters of a cell).
- Small-radius curved sides, needles in decimated meshes (181 / 69 / 184 /
  131 left), the remote daemon: as before.

