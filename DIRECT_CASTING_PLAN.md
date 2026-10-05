# Direct casting: drawing a model without meshing it

Status: **ratified 2026-10-05; step 1 of 4 built** (the search library,
`cast-bench`; §5.1), with the `sample-bench` instrument (`6aa4e2e`). This file is the target design; when
a decision here changes, rewrite it and record what was rejected and why.

## 1. What this is for

Two uses, in priority order (user ruling 2026-10-05):

1. **Feedback after an edit**, in the cases the mesher serves badly:
   - parts where 128 cells is too coarse for real work and even 512 leaves
     the user guessing whether a flaw is in the model or in the mesh;
   - lattices, where a mesh fine enough to catch the geometry reliably
     takes over a minute;
   - zooming in on a feature, which should develop more exact samples
     there, so nothing on screen is a mesh defect.
2. **High-quality stills** of a model in a pose.

It complements the mesh viewport; it does not replace it. Export always
goes through the mesher.

## 2. Measurements this rests on

Release build, 24 threads, 2026-10-05. Mesher runs are the full pipeline
with sharp edges. Sample rates are for points near the surface
(`volumetric_cli sample-bench`).

| Model | ns/sample, 1 thread | samples/s, all threads | mesh at 128 | mesh at 512 |
|---|---|---|---|---|
| keychain sleeve | 75 | 170M | 0.09 s | |
| comet racer | 107 | 118M | 0.18 s | |
| housing D | 169 | 74M | 0.22 s | 2.9 s |
| toy car | 294 | 40M | 0.18 s | 2.1 s |
| cushion struts | 493 | 32M | 0.56 s | |
| chair base | 504 | 23M | 0.09 s | 0.8 s |
| lattice_test | 834 | 19M | 8.7 s | 37 s at 256 |
| Raspberry Pi STEP | 9,127 | 1.6M | 1.2 s | 24.7 s |

- Threads scale 12–17× on 24.
- Mesher time and samples grow about 4× per doubling of resolution.
- Toy car at 512: sampling is about 45% of mesher time, sharp-edge
  reconstruction 32%, decimation and the rest about 20%.
- Not measured: the GUI's time from an edit to new pixels.

## 3. The constraint

A model is a wasm occupancy function: one point per call, on the CPU, no
distance, no gradient, no bound on how fast it changes. Two consequences
shape everything below.

- **Finding the surface along a ray is a search.** Step until a sample is
  inside, then bisect. Bisection is cheap and exact (10 samples buy three
  decimal digits of the bracket).
- **Empty space can only be certified by sampling it.** To know there is
  nothing thicker than `s` in a region, the region must have been sampled
  at pitch `s`. No trick removes this; the mesher is bound by it too. A
  megapixel view searched at pixel pitch through a model a thousand pixels
  deep is a billion samples: 8 s for the comet racer, 50 s for the lattice.

So the caster cannot be both instant and certain. The design makes it
**honest and progressive** instead: a useful image at once, certainty that
grows while the view rests, and a record of how certain each region is so
no work is repeated.

## 4. Design

Sections 4.1 and 4.2 describe what is built (`src/direct_cast.rs`); 4.3
and 4.4 are still the target.

### 4.1 One tree per model

A `DirectCast` is an octree over the model's bounds, in the model's own
frame, so a pose change moves it rigidly and an edit to another part
leaves it alone. It holds both things the proposal called for:

- **Surfels.** Each surface point found (bisected position, normal, disc
  radius) is stored in the node whose level matches the pitch it was found
  at: a node is four pitches wide. The tree is therefore a level-of-detail
  hierarchy of the surface. A coarse surfel is marked superseded when a
  finer one is found inside its disc, and `surfels(view)` picks, for each
  part of the surface, the level that suits that view's pixels.
- **The search record.** Per node: whether surface has been found in or
  under it, and the finest pitch its volume has been searched at and found
  empty. Rays skip nodes searched finely enough for their pixel.

A node counts as searched by a pass when at least 70% of the rays its
outline should catch went all the way through it without a hit. Rays that
stopped at nearer surface do not count, so a node partly in shadow is not
claimed. A node searched at up to 1.5 times the pitch a ray wants is taken
as searched; without that allowance a view slightly closer than the last
would search everything again.

Rejected: a screen-space cache only (the depth image of the last frame).
It is lost on every camera move, cannot serve a pose change, and cannot
say what has been searched.

Rejected: counting samples per node and calling a node searched at
`size / cbrt(samples)`. A node half hidden behind nearer surface gets all
its samples in the visible half and would be claimed whole.

### 4.2 The search

A pass casts one lattice of rays, some whole number of pixels apart. Each
ray walks the tree front to back, steps through the nodes its pass mode
selects at a stride of one pitch (ray spacing times pixel footprint), and
on an inside sample bisects ten times back to the surface. Three modes:

- **Discover** steps every node not yet searched at the pass's pitch.
  This is the pass that finds things, and where the time goes.
- **Refine** steps only nodes known to hold surface, their unsearched
  siblings, and nodes marked for a closer look. It redraws known surface
  on a finer lattice, cheaply.
- **Chase** steps only the marked nodes, in front of what the image
  already shows.

`cast()` orders them the way a viewport wants: Discover at 8 px, Refine at
4, 2 and 1 px (a full-resolution image of everything found so far), Chase
until nothing new, and only then Discover at 4, 2 and 1 px, each followed
by a Refine and Chase at 1 px if it found anything.

**Following thin things.** A hit whose neighbouring rays missed, or hit at
quite another depth, is on something thin or at the edge of what is known.
The 26 nodes around it are marked; the next pass steps through them, and a
Chase pass marks the neighbours of whatever it finds in turn. A strut
touched once by the coarse search is followed along its whole length
without any fine search of empty space. A marked node gets one look.

**Boundaries are sampled.** Skipping is decided node by node, so wherever
stepping starts or stops the boundary point itself is sampled, and the
point where a ray enters the model's bounds. Without this a ray crosses
unseen through anything lying partly in a stepped node (too little of it
to meet a step) and partly in a skipped one; that lost 3–10% of the
pixels of every thin rod in the test scene. A node in which any sample was
inside the model is marked as holding surface even when bisection put the
surface point in the node before it.

**Normals** are those of the plane through the hit and two more points
bisected half a pitch to the side and above in the image. A hit within
half a pitch of a surfel already stored at the same pitch reuses its
normal and costs no probes. Where a probe finds no surface (a silhouette,
something thinner than a pitch) the normal faces the viewer.

Not built from the proposal: marking such surfels as edges and taking the
normal from one side. Known limit: the three points are one-sided, so on
a curved surface the normal leans by about the angle the surface turns in
half a pitch; near a silhouette, where half a pixel is a long way round,
that reaches a couple of degrees at 160 px across a sphere.

### 4.3 Drawing

Surfels are drawn as opaque oriented discs into the G-buffer (albedo,
normal, object id, depth), so lighting, edge lines, occlusion, picking and
the grid treat them exactly as they treat meshes. This is the image source
of `RENDERING_ARCHITECTURE.md` §9, realised as geometry rather than as a
depth image, which is what lets it survive camera motion.

Rejected: blended Gaussians. They need sorting, write no depth and soften
exactly the edges the search made exact.

Open for step 2: a pass also returns its rays' hits as an image, exact per
pixel. A finished still could be drawn from that image instead of from
discs, which overhang silhouettes by up to their radius. Decide when both
can be looked at.

**Showing certainty.** The viewport says how finely the visible region has
been searched (for example "searched to 4 px" falling to "1 px"), so a
blank region is never mistaken for a certified one.

### 4.4 Stills

The same search run to completion at the output's pixel pitch (or finer,
for supersampling), headless, behind `render --direct` and the Python
`render`. No time limit, so no progressive display, but the same record,
the same G-buffer pass, the same shading.

## 5. Build order

Each step ends with something that can be checked by running it.

1. **The search as a library.** BUILT 2026-10-05; results in §5.1.
2. **The surfel pass** in `volumetric_renderer` and `render --direct`.
   Checked by comparing stills with mesh renders of the same view
   (depth agreement where both have surface), on the Vulkan and GLES test
   adapters.
3. **The viewport mode**: progressive passes on a worker, cancel on edit,
   coarse during motion, the certainty readout, per-part records for
   assemblies. Includes not re-tracing pixels the stored surfels already
   cover after a camera move, which step 1 does not do.
4. **Tuning on the two hard cases**: `lattice_test` and the Raspberry Pi
   STEP import (9 µs samples).

### 5.1 Step 1 as built

`src/direct_cast.rs` with nine tests on analytic models: a sphere and a
torus found to under 0.01 px with normals within the limit above; a grid
of 25 rods two pixels thick found whole both by the full search and by the
coarse search plus following, at under half the samples; a second view
costing under two thirds of a fresh one; zooming in producing finer
surfels; determinism; cancellation; a model filling its bounds.

The thumbnail marcher in `src/direct_preview.rs` is deleted; thumbnails
are a small cast shaded from the normals found. The module stays as the
thumbnail client (it also draws 2D sketches).

`volumetric_cli cast-bench` runs it on real models. At 1024 × 1024,
perspective, model framed on its bounds, 24 threads:

| Model | full-resolution image | full 1 px search | follow only (`--search inf`) |
|---|---|---|---|
| comet racer | 0.27 s | 1.15 s | 0.31 s |
| toy car | | 2.3 s (then 1.2 s per view 10° on) | |
| lattice_test | 1.4 s | 3.4 s | 1.6 s |
| lattice_test, zoom 6 | | 9.5 s | |
| Raspberry Pi STEP | 8.2 s (2 px image at 2.1 s) | 28.6 s | |

The first image, at 8 px spacing, takes 5–15 ms (0.12 s for the STEP
import). For comparison the mesher takes 8.7 s on the lattice at 128 cells
and 37 s at 256.

Checks against the models themselves: 99.1–99.9% of surfels have the model
outside a quarter radius along the normal and inside the same way back.
Against the mesher on the comet racer at 512 cells: of 230,463 vertices
not hidden behind nearer surface, the cast shows surface within 2 px of
99.48%, median depth difference 0.15 px. The 1,189 it does not show are
not yet explained (grazing vertices at silhouettes are the likely cause;
not checked).

Known limits at the end of step 1:

- Every view re-traces its hits; stored surfels are reused for their
  normals only. Reuse of the search record roughly halves a nearby view.
- Empty nodes partly hidden behind nearer surface, or cut by the edge of
  the frame, never count as searched and are searched again by each
  Discover pass that reaches them.
- A pass walks every ray through the tree even when few have anything to
  step, about 40–70 ms per megapixel pass. Chase passes pay this each.
- `surfels(view)` does not cull to the frame, and a coarse level that was
  only ever cast in part is drawn in part.
- The zoomed lattice view costs 168 samples per pixel; not yet looked at.
- A few isolated dark pixels appear on flat faces of the zoomed lattice
  and above the Raspberry Pi board; not yet explained.

## 6. Open questions

- **Memory.** A megapixel of surfels is small (tens of MB); surfels
  accumulated over many views and zoom levels are not, and neither is the
  tree (1.5M nodes for one view of the lattice). Nothing is evicted yet.
- **Interior faces.** Occupancy is boolean, so a ray starting inside the
  model (camera inside, or a section view) needs the outside-to-inside
  rule reversed. Section views are not in scope for the first build.
- **Web.** The browser has one sampling thread. The search will run there
  but slowly; whether that is worth shipping is undecided.
- **Sharing the pyramid with the mesher.** It is the same information the
  mesher's discovery stage computes. Not planned; noted because it is the
  obvious next question.
- **Material and colour channels.** Models can declare sample channels;
  surfels could carry them. Not in the first build.
