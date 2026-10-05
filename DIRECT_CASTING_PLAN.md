# Direct casting: drawing a model without meshing it

Status: **proposed 2026-10-05, not yet ratified, nothing built** except the
`sample-bench` instrument (`6aa4e2e`). This file is the target design; when
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

### 4.1 Two world-space caches per model

Both live in the model's own frame, keyed like the mesh cache by the
model's content hash, so a pose change moves them rigidly and an edit to
another part leaves them alone.

- **Search pyramid.** A sparse octree over the model's bounds. Each node
  records the finest pitch its volume has been searched at and whether any
  sample in it was inside. It answers "how far along this ray is already
  known empty at the pitch this pixel needs?" Rays skip what is certified
  and search only the rest, then write back what they searched.
- **Surfels.** Each surface point found: position (bisected), normal, and
  the pitch it was found at (its disc radius). These are what is drawn.

The pyramid is what makes zoom adaptive: the pitch a pixel needs is its
footprint at that depth, so zooming in asks for a finer pitch in the
visible region only, and the pyramid shows it has not been searched that
finely yet.

Rejected: a screen-space cache only (the depth image of the last frame).
It is lost on every camera move, cannot serve a pose change, and cannot
say what has been searched.

### 4.2 The search

Per ray, front to back through the model's bounds:

1. Skip spans the pyramid certifies empty at this pixel's pitch.
2. Step the rest at the current pass's stride, with a per-ray offset so
   successive passes test new depths rather than the same ones.
3. On an inside sample, bisect back to the surface, emit a surfel.
4. Record the searched spans in the pyramid.

Passes halve the stride (and the pixel spacing) until both reach the pixel
footprint. The first pass is coarse in both: every 8th pixel at an 8-pixel
stride is 1/512 of the full search.

**Hit-guided refinement.** A hit whose neighbouring rays missed is the
signature of something thin. Those neighbours are searched immediately at
fine stride around the hit's depth, before the general passes get there.
A lattice strut first appears as scattered dots and is completed along its
length within the same pass.

**Normals** come from neighbouring surface points: the hit plus two more
bisected a fraction of a pixel away in the image plane. Where the three
disagree in depth by more than the footprint (a silhouette or a step), the
surfel is marked as an edge and takes its normal from the side it lies on.

### 4.3 Drawing

Surfels are drawn as opaque oriented discs into the G-buffer (albedo,
normal, object id, depth), so lighting, edge lines, occlusion, picking and
the grid treat them exactly as they treat meshes. This is the image source
of `RENDERING_ARCHITECTURE.md` §9, realised as geometry rather than as a
depth image, which is what lets it survive camera motion.

Rejected: blended Gaussians. They need sorting, write no depth and soften
exactly the edges the search made exact.

**Showing certainty.** The viewport says how finely the visible region has
been searched (for example "searched to 4 px" falling to "1 px"), so a
blank region is never mistaken for a certified one.

### 4.4 Stills

The same search run to completion at the output's pixel pitch (or finer,
for supersampling), headless, behind `render --direct` and the Python
`render`. No time limit, so no progressive display, but the same caches,
the same G-buffer pass, the same shading.

## 5. Build order

Each step ends with something that can be checked by running it.

1. **The search as a library** (in the `volumetric` crate, beside the
   mesher): rays in, surfels out, with the pyramid. Checked against
   analytic models (sphere, torus: position and normal error), against the
   mesher's vertices on real projects, and for determinism.
2. **The surfel pass** in `volumetric_renderer` and `render --direct`.
   Checked by comparing stills with mesh renders of the same view
   (depth agreement where both have surface), on the Vulkan and GLES test
   adapters.
3. **The viewport mode**: progressive passes on a worker, cancel on edit,
   coarse during motion, the certainty readout, per-part caches for
   assemblies.
4. **Tuning on the two hard cases**: `lattice_test` (does hit-guided
   refinement find every strut, and how long to certainty) and the
   Raspberry Pi STEP import (9 µs samples: is the first image fast enough).

`src/direct_preview.rs` (the thumbnail marcher) is replaced by the library
in step 1 and deleted.

## 6. Open questions

- **Memory.** A megapixel of surfels is small (tens of MB); surfels
  accumulated over many views and zoom levels are not. Needs an eviction
  rule (coarser surfels superseded by finer ones in the same place).
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
