# Direct casting: drawing a model without meshing it

Status: **ratified 2026-10-05; steps 1 to 3 of 4 built** (the search
library and `cast-bench`, §5.1; the disc pass and `render --direct`, §5.2;
the viewport mode, §5.3), with the `sample-bench` instrument (`6aa4e2e`).
Step 4, tuning on the hard cases, remains. This file is the target design; when
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

Sections 4.1 to 4.5 describe what is built.

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

**The stride has a floor** of 1/5000 of the model's half-size, whatever
the pixel footprint, and an eye inside the model sees nothing of it.
(Fixed 2026-10-05 after the first GUI run: a perspective ray starting at
the eye, inside the bounds, had a footprint of zero, so its step was zero
and the thread spun forever, past the slot being dropped, since
cancellation is checked per ray.) The floor applies to the stride only;
the pitch a surfel is stored at is always the pixel's. (The first version
floored both, so every surfel of a view closer than the model filling the
frame counted as too coarse for it and was dropped once the final pass
landed: the viewport went dark but for edge highlights, or the near half
of a model did.)

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
normal, object id, depth) by the renderer's surfel source
(`crates/volumetric_renderer/src/pipelines/surfel.rs`), so lighting, edge
lines, occlusion, picking and the grid treat them exactly as they treat
meshes. This is the image source of `RENDERING_ARCHITECTURE.md` §9,
realised as geometry rather than as a depth image, which is what lets it
survive camera motion. A disc is a quad per surfel in its tangent plane,
cut round in the fragment shader, back faces culled; its depth is the
plane's, so the discs of one surface meet without gaps.

`DirectCast::surfels(view)` chooses the surfels for a view: at each part
of the surface the level whose pitch suits the view's pixels at the
node's nearest point, and a coarser surfel only where no finer one has
been found beneath it (looked up by position when the surfels are
collected). Once a pass has produced an image of the view,
`surfels_shown` also drops every coarse surfel whose pixel's ray hit
surface at or in front of it: the finer point that replaces it, or what
hides it. (Added 2026-10-05 from a screenshot: after zooming in on a
lattice, the farther view's surfels stood along the near view's
silhouettes and creases, where the tree lookup alone finds nothing finer
beneath their centres.) Rejected: a stored
"superseded" flag set when a finer surfel is inserted within a coarse
one's disc. It missed coarse surfels in neighbouring nodes and at
silhouettes, where normals turn fast, and their oversized discs stood out
of every outline (user-visible on the first `render --direct`).

Rejected: blended Gaussians. They need sorting, write no depth and soften
exactly the edges the search made exact.

Open: a pass also returns its rays' hits as an image, exact per pixel. A
finished still could be drawn from that image instead of from discs, which
overhang silhouettes by up to their radius and give the edge-line pass
disc rims to find. Discs were built first because the viewport needs them
regardless; compare before deciding what `render --direct` uses.

**Showing certainty.** The viewport says how finely the visible region has
been searched (for example "searched to 4 px" falling to "1 px"), so a
blank region is never mistaken for a certified one.

### 4.5 The viewport

A fourth preview mode, **Direct**, beside points, marching cubes and
surface nets. Choosing it builds a `PreviewEntity` like any other, but
with no geometry: one empty mesh per part, keyed by the part's model hash,
with the model bytes alongside (`PreviewEntity::direct_models`). Everything
that handles entities (the cache, residency by key, assemblies' poses and
drags, object ids, framing) is unchanged; the viewport session keeps one
`DirectSlot` per model hash beside its per-part meshes.

A slot owns a thread with the model's sampler and `DirectCast`. Each frame
the session hands it the frame's camera, in the part's own frame
(`cast_view_in_frame`, so a posed or dragged part is cast without moving
its record); a changed view cancels the pass in flight and starts a
`CastRun` for the new one. After every pass the thread sends the surfels
to draw (`DirectCast::surfels` for that view) and the run's state; the
session uploads them as retained surfels and submits them under the
part's transform and object id, so they are lit, edged, occluded and
picked like the mesh they replace. The HUD badge reads "direct cast:
searched to 8 px, drawing at 1 px" and so on; the shell keeps painting
while any slot has passes left.

Reuse between views is `DirectCast::coverage`: every pass starts from
what the stored surfels already show of the new view, a refine pass
traces only the pixels they do not cover, and the other passes stop each
ray at the covered depth. Orbiting a cast model therefore costs a coarse
search per frame and, at rest, the pixels the move uncovered.

Not in the browser: the thread is `std::thread`; on `wasm32` the slot has
no worker and the badge says so.

### 4.4 Stills

The same search run to completion at the drawn frame's pixel pitch, which
with the default supersampling is twice the output's, headless, behind
`render --direct` and the Python `render(direct=True)`. No time limit, so
no progressive display, but the same record, the same G-buffer pass, the
same shading. Only `Model` assets are cast; other kinds (assemblies among
them) are meshed as before, and a photograph's off-centre camera is
refused, since the caster's view has a symmetric frustum.

## 5. Build order

Each step ends with something that can be checked by running it.

1. **The search as a library.** BUILT 2026-10-05; results in §5.1.
2. **The surfel pass** in `volumetric_renderer` and `render --direct`.
   BUILT 2026-10-05; results in §5.2.
3. **The viewport mode.** BUILT 2026-10-05; §4.5 and §5.3.
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

### 5.2 Step 2 as built

The disc source, with a frame test on both test adapters: discs over a
sphere give the same picture as the sphere's mesh (coverage within 2%,
under 5% of pixels differing by more than a shade), and a pick through
them reports their object and a point on the sphere. `render --direct` and
the Python keyword, each with a test rendering the bundled model both ways
and comparing. Dropped surfels are reported with the other overflow.

At 1024 × 1024 output, supersampled 2× (so cast at 2048 × 2048), the
comet racer takes 7.2 s against 0.9 s meshed; the lattice 16.5 s against
9.4 s; the Raspberry Pi STEP import 28.7 s without supersampling. Looked
at: the comet racer's wheels show their grooves and spokes cleanly where
the 128-cell mesh shows faceting and torn edges; the lattice shows its
actual cells, square faces and holes where the mesh shows a blob. The
dark dot at the centre of every lattice face was probed
(`cast-bench --probe`): the model is outside for three pixels either side
of the face plane along those rays, so the holes are in the model, and the
full search draws the interior surface seen through them.

Known limits at the end of step 2:

- Edge lines along fine creases come out dashed, and a little speckle
  sits along silhouettes: the edge pass finds the rims of discs seen at
  grazing angles. A few disc "crumbs" remain at silhouettes where a coarse
  surfel has no finer one beneath it.
- Supersampling multiplies the cast's cost by four; `--supersample 1` is
  the cheap still.
- Not cast: assemblies and other non-`Model` assets, and views through a
  photograph's camera.

### 5.3 Step 3 as built

`CastRun` (the pass schedule as a state machine, one pass per step) and
`DirectCast::coverage` in the library; `PreviewRenderMode::Direct`,
`PreviewMeshPlan::Direct` and `cast_view_of`/`cast_view_in_frame` in
`volumetric_preview` (the headless render uses the same conversion);
`DirectSlot` and `DirectWorker` in the GUI session; the mode button, the
HUD badge, and the resolution row hidden for Direct.

Checked: a GPU-backed session test previews the bundled sphere in Direct
mode, drives frames until the badge says "searched to 1 px", and picks
its surface at the sphere's radius. The part-frame conversion is tested
against the world-frame projection for a scaled, rotated, translated
part, perspective and orthographic. With coverage, the library's second
view costs 28% of a fresh one (was 55%).

Not checked: the GUI itself has not been run with a Direct output, so the
badge, the mode button and the feel of orbiting a cast model are unseen;
an assembly in Direct mode has not been tried; the browser build only
type-checks.

Known limits at the end of step 3:

- The cast is at the viewport's physical size; a 4K viewport of a slow
  model refines slowly, and there is no setting for a coarser final
  spacing.
- Nothing is evicted from a slot's record while the output is shown; a
  long session orbiting a lattice grows it without bound.
- Each pass re-uploads every surfel of the view (tens of MB for a
  megapixel), rather than the ones that changed.
- Part drags re-cast nothing (the record moves with the part) but the
  coverage is recomputed from scratch each view change.

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
