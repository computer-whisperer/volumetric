# Volumetric Renderer: Target Design

Status: **built** (written 2026-10-05 as the target design; all five steps
of section 11 were built the same day). This file replaces the May 2026 MVP
architecture notes and now describes the renderer as it is; where it was
revised while building, the revision and its reason are recorded in place.
Direct model raytracing (section 9) is the follow-up project and is not
built. Transparent meshes (section 4) were added on 2026-11-17.

## 1. Scope

`volumetric_renderer` is the wgpu renderer shared by the GUI viewport
(`volumetric_ui_v2`, native and web), the headless `render` command and the
Python bindings (both through `volumetric_render`).

In scope for this overhaul:

- a real deferred frame with a G-buffer that more than one kind of geometry
  source can fill;
- neutral studio-lit CAD shading, with room to grow;
- CAD-standard navigation: Z-up, orbit about the point under the cursor,
  zoom to the cursor, exact standard views, orthographic projection;
- scene widgets that adapt to scale and are configurable: ground grid, world
  axis lines, an interactive view gizmo;
- cursor picking (depth and object) as a renderer service.

Designed for, built later (the follow-up project): direct model raytracing
in the viewport, bypassing the mesher. Section 9 states the contract the
frame keeps open for it.

Out of scope: splat rendering and the lens warp keep their current design
and only move to their place in the new frame.

## 2. Ratified decisions

| Decision | Chose | Because | Rejected |
|---|---|---|---|
| World frame | Right-handed, **Z-up**, metres, everywhere | Print and CAD convention; the viewport (Y-up), the thumbnail caster (Z-up) and `render --up` currently disagree | Configurable up axis: three conventions to keep consistent for no user benefit |
| Web | WebGL2 fallback stays supported; features may degrade there | Broadest reach for the daemon-hosted UI | Dropping WebGL2 to gain compute shaders |
| Direct raytracing | Follow-up project; this design only reserves its seat | Keeps this arc to one window | Building both at once |
| Default look | Neutral studio-lit CAD | Reads shape reliably; a base to add features to | Physically based materials as the starting point |
| View gizmo | Drawn by the renderer, interactive | It must sit in the same frame as the camera it reports; the old indicator was settings with no draw path | A UI-toolkit overlay: free text and hit-testing, but lags the viewport texture and is absent from headless frames |
| Lighting output | One resolve pass writes display-referred colour | Nothing in scope needs an HDR intermediate | HDR target plus separate tone-map pass: revisit when bloom or exposure control arrives |
| Anti-aliasing | FXAA in the viewport, supersampling for headless renders | Works with a deferred G-buffer and on WebGL2 | MSAA: multiplies every G-buffer target |

## 3. Coordinates and camera

World: right-handed, +Z up, the ground plane is XY, units are metres.

### Camera state

```rust
pub struct Camera {
    /// The point the view is centred on.
    pub focus: Vec3,
    /// Camera-to-world rotation (camera looks down its -Z, +Y is screen up).
    pub orientation: Quat,
    /// Eye-to-focus distance.
    pub distance: f32,
    pub projection: Projection, // Perspective | Orthographic
    pub fov_y: f32,             // default 35 degrees
}
```

- The eye is `focus - forward * distance`.
- The orthographic frame is `2 * distance * tan(fov_y / 2)` tall, so
  switching projection keeps the apparent size at the focus.
- Orientation is stored, never derived from a look-at with a fixed up
  vector, so looking straight down is an ordinary pose: Top is exactly top.
- Clip planes follow `distance` (near at 0.005 of it), with the far plane
  pushed out to hold the scene's bounds; an orthographic frame also sees
  behind the eye. (Revised 2026-10-05 from "fitted to the scene bounds
  only": zooming about a point scales `distance` together with the eye's
  distance to that point, so `distance` stays a good measure of how
  closely the user is looking, and a distance-relative near plane is what
  keeps an approached surface from being clipped at any part scale.)

`CameraView` (view and projection matrices) and `Pinhole` stay as they are:
they are how look-through and the CLI hand the renderer a view.

### Standard views

| View | Eye sits on | Screen up |
|---|---|---|
| Front | -Y | +Z |
| Back | +Y | +Z |
| Right | +X | +Z |
| Left | -X | +Z |
| Top | +Z | +Y |
| Bottom | -Z | -Y |
| Isometric (default) | +X, -Y, +Z | +Z |

Changing to a standard view is animated (about 200 ms, shortest rotation)
and keeps `focus` and `distance`.

### Navigation

Every gesture is defined by a **gesture point** `P`: the world point under
the cursor when the gesture starts, from the pick service (section 6). When
the cursor is over background, the centre `C` of the scene's bounds stands
in for the missing surface: an orbit takes `P = C`, so the model turns in
place wherever the press landed (with `C` outside the view, `P` is the
middle of the view at `C`'s depth, so what is on screen stays there); a pan
or zoom takes `P` where the cursor ray reaches `C`'s depth. Without a
scene, or with `C` behind a perspective eye, `focus` takes `C`'s place.

Rejected: anchoring a background orbit on the focus-plane point under the
cursor (built first, 2026-10-05). It keeps one rule for every gesture, but
the pivot is then a point of empty space, far from the part once cursor
zooms have moved the focus, and the model swings around it.

| Gesture | Behaviour | Invariant (unit-tested) |
|---|---|---|
| Orbit | Rotate the camera rigidly about `P`. Turntable: yaw about world Z, pitch about the camera's right axis, elevation limited to exactly ±90 degrees | `P` does not move on screen; in turntable mode the horizon stays level |
| Zoom (wheel or drag) | Scale by `s` about `P`: `distance *= s`, `focus = P + s * (focus - P)` | `P` stays under the cursor, in both projections |
| Pan | Translate in the view plane at `P`'s depth | `P` tracks the cursor 1:1 |
| Frame | Fit the bounding sphere to the narrower of the two field-of-view angles | The whole scene is inside the frame at any aspect ratio |
| Standard view | Animated reorientation | Ends on the exact pose |
| Projection toggle | Perspective ↔ orthographic | Size at the focus is unchanged |

Rotation about `P` by `R` is `focus = P + R * (focus - P)`,
`orientation = R * orientation`.

Orbit has two modes, a user setting: **turntable** (the default, above) and
**free**, where the drag rotates about the camera's own right and up axes
through `P` with no elevation limit and no level horizon. Both keep `P`
fixed on screen. Standard views and Frame behave the same in either mode.

### Where the code lives

- `Camera`: state, matrices, the operations above. Pure math.
- `Navigator`: the gesture state machine. It takes pointer and wheel events,
  a pick result and a time step, and mutates a `Camera`. No wgpu; fully unit
  tested with the invariants above.
- `CameraControlScheme` keeps its job: mapping buttons and modifiers to
  gestures (Blender, OnShape, Fusion 360, SolidWorks, Maya).
- Hosts forward events to the `Navigator` and repaint while it reports an
  animation in progress.

## 4. The frame

```
 geometry sources ──► G-buffer ──► AO ──► resolve ──► FXAA ──► scene target
   mesh rasteriser      albedo            lighting ▲            │
   surfel discs         normal            edges    │            ▼
                        object id         tone map │     grid, lines, points
                        depth                      │     splats
                                                   │     overlay lines, points
 transparent meshes ──► a layer ──► resolve ──► gathered   lens warp (look-through)
   (per layer,          as the      as above    layers     view gizmo
    nearest first)      G-buffer
```

### G-buffer

| Target | Format | Contents |
|---|---|---|
| Albedo | `Rgba8Unorm` | rgb: base colour; a: material index |
| Normal | `Rgb10a2Unorm` | rgb: world normal; a: 0 means "no normal supplied, reconstruct from depth" |
| Surface | `Rg32Uint` | r: `ObjectId` of the draw, 0 is background; g: the bits of the fragment's depth as `f32`, 1.0 is background |
| Depth | `Depth24Plus` | Hardware depth, for depth testing only |

No lighting happens while the G-buffer is filled.

Depth is carried twice, and later passes read it from the surface target,
never from the depth attachment. (Revised 2026-10-05, replacing "passes
sample the depth attachment": measured on a GLES adapter under WebGL2
limits, wgpu cannot translate a plain read of a depth texture to GLSL,
neither `textureLoad` nor a non-comparison sampler. An integer colour
target is renderable and loadable on every backend with no extension, and
holds the value exactly. Rejected: a float colour target for depth, which
needs `EXT_color_buffer_float` on WebGL2.)

### Geometry sources

A source is anything that can write those four targets with depth testing,
so sources compose with each other and with everything drawn afterwards.

1. **Mesh rasteriser** (this arc). Retained `GpuMesh` handles drawn with a
   per-draw transform, `ObjectId` and `MaterialId`.
2. **Surfel discs** (`surfel_gbuffer.wgsl`, built 2026-10-05 for the direct
   caster). Retained `GpuSurfels`: surface points with normals and radii,
   each drawn as an opaque disc in its tangent plane, back faces culled,
   under the same per-draw transform and ids as a mesh, coloured by the
   surfel's colour times one per draw (the part tint, 2026-10-06), stored
   in the G-buffer as a mesh vertex's colour is.
   The G-buffer gets
   the disc's own depth, so discs of one surface meet without gaps and
   everything after the fill treats them as it treats a mesh. (Replaces
   the planned full-screen image source of section 9; see
   `DIRECT_CASTING_PLAN.md` §4.3 for why geometry rather than an image.)

### Passes

1. **G-buffer fill** from every source: the opaque geometry.
2. **Ambient occlusion** (`ao.wgsl`), followed by a blur that keeps to
   one surface (`ao_blur.wgsl`). Distances are compared in world units in
   front of the eye, so the result does not depend on the clip planes. The
   radius is a fraction of the diagonal of the frame's meshes (the
   renderer keeps each mesh's bounds), clamped to 15 % of the frame's
   height on screen, so it is right for a 20 mm part and a 2 m assembly
   alike; a frame test renders one scene at 1 m and at 1 mm and compares.
   (The MVP's shader compared non-linear depth against a fixed bias, which
   by arithmetic registered no local occlusion with the viewport's clip
   planes.)
3. **Resolve**: lighting × AO, edge lines, background, tone map, into a
   display-referred intermediate.
4. **Transparent meshes**, in a frame that has any: peeled a layer at a
   time and laid over the intermediate (below). Their surface targets
   are filled before pass 3.
5. **FXAA** into the scene target (built 2026-10-05, ahead of the rest of
   step 4; `RenderSettings::antialiasing`).
6. **Grid** (section 5), depth-tested and blended.
7. **Depth-tested lines and points.**
8. **Splats**: own layer, then composited (unchanged).
9. **Overlay lines and points** (no depth test).
10. **Lens warp** when looking through a photograph (unchanged).
11. **View gizmo** on the final target.

The **pick pass** (section 6) runs on demand, outside this sequence.

### Transparent meshes

Built 2026-11-17. A mesh is submitted with an opacity. At 1 it is opaque
and takes the G-buffer as before; at 0 (or too little to change a pixel
of an 8-bit frame) it is not drawn; in between it is transparent, and
drawn after the opaque scene is lit.

The G-buffer holds one surface per pixel and everything after it reads
that, so a transparent surface cannot be put in it. The transparent
meshes get the same treatment a layer at a time instead (depth peeling).
A layer is, at each pixel, the nearest of their surfaces that is in front
of the opaque scene and behind the layer before: the fragment shader
discards the rest (reading the G-buffer's surface target and the layer
before's) and the layer's own depth test keeps the nearest. The first
layer is therefore the nearest transparent surface at each pixel, the
second the next one behind it, and so on.

1. **Every layer's surface target is filled** (`fs_layer_surface`):
   object and depth, as the G-buffer's surface target holds them. The
   edge lines of the opaque scene and of every layer read these (below),
   so they come first.
2. **Each layer in turn, nearest first, is filled again**, this time its
   albedo, normal and opacity (`fs_layer`), and **resolved**:
   `resolve_layer.wgsl` lights it with the resolve pass's own lighting
   (`resolve_common.wgsl`: the same rig, materials and tone curve) and
   draws its edge lines. Its output, premultiplied by the opacity, is
   blended *under* the layers gathered so far, so each layer is dimmed by
   what is in front of it.
3. After `RenderSettings::transparency_layers` layers (4 by default, 8 at
   most) the gathered layers are laid **over the lit scene**, before
   anti-aliasing.

The opaque scene is resolved (pass 3) after step 1 and after the nearest
layer's second fill, whose opacity its edge lines read.

| Target | Format | Contents |
|---|---|---|
| Albedo, normal, depth | as the G-buffer's | the layer being drawn; reused by every layer |
| Opacity | `R8Unorm` | the square root of how opaque the layer's surface is; reused likewise |
| Surface, one a layer | `Rg32Uint` | as the G-buffer's. The next layer is peeled against it, every other layer's edge lines look for their surfaces in it, and a pick reads the nearest layer's |
| Gathered | the frame's | the layers lit so far, premultiplied by coverage |

That is 17 bytes a pixel and 8 more for each layer: 49 a pixel at four
layers, beside the G-buffer's 34. They are made by the first frame that
draws a transparent mesh and dropped when the frame's size changes; a
frame without transparent meshes runs no pass and reads no target of
theirs, and its pixels are the same bytes as before transparency was
built (28 frames compared, on both adapters).

Opacity is stored as its square root because 8 bits of it, blended in
linear light, are coarse where a fade is faintest: the first step of a
white mesh over black would be ten levels of the frame. The least opacity
drawn is the least that could change a pixel by half a level.

What this gives:

- The blend is right per pixel in depth order, for meshes that pass
  through or enclose each other and whatever order they are submitted
  in. (What a fade between a model and the model it was made from needs:
  the two occupy the same space.)
- Surfaces in exactly the same place are both drawn. The higher
  `ObjectId` is in front, between two transparent meshes and between a
  transparent one and an opaque one; between two opaque meshes it is the
  one submitted last, as before, which is the same thing for a host that
  numbers its meshes as it submits them.
- Opacity is a value of the draw, not of the retained mesh: fading costs
  no upload.
- A transparent mesh is lit, edged and picked like an opaque one.
- Nothing needs more than WebGL2 gives: the fills are unblended and the
  blends have one target each. A layer's resolve reads twelve textures of
  the sixteen WebGL2 guarantees, which is what limits the layers to 8.

#### Edge lines

Edge lines come from a G-buffer's discontinuities, and a layer is not a
G-buffer of one scene: it holds whichever surface is so many surfaces
deep at each pixel. A surface is one layer farther wherever something
transparent in front of it begins, and one nearer wherever that ends, so
a layer's neighbouring pixels may hold different surfaces with nothing
ending between them.

The rule that settles every case is that **a transparent surface counts,
in the edge test, as much as it is opaque**. Wherever the answer depends
on a transparent surface, the test is made both ways (that surface
opaque, that surface absent) and the mark is the blend of the two by its
opacity. Then the marks are the all-opaque frame's when every opacity is
1, a mesh fading out takes its marks on other meshes with it, and a mesh
keeps its own outline whatever fades in around it.

In a layer (`edge_in_layer`, `mark_toward`):

- Beside a surface is the layer's own next pixel, or the opaque scene's
  where the layer is empty there (the opaque scene is behind every
  layer).
- If what is beside it is farther, or nothing, the surface has ended,
  unless it is found there in another layer (the same object, at the
  depth it would have): then what was in front of it ended, and it did
  not.
- If another object is beside it, as near or nearer, and the surface is
  found there in another layer, it passes behind that object or through
  it. The mark is that object's outline, or the line where the two cross,
  and is as strong as that object is opaque.
- If another object is beside it and the surface is nowhere there, it
  ends against that object: marked toward the right and down, and toward
  the left and up (where the opaque scene leaves the mark to the other
  object's pixel) as far as that object is not opaque.
- The test extrapolates a surface from the pixel a step the other way.
  Where the surface is in another layer there, it is extrapolated from
  that; where it is behind a transparent surface there, the answer is
  blended with the one an opaque surface would leave.

In the opaque scene (`edge_beside_transparent`), the G-buffer does not
hold the transparent meshes, but the nearest of them stand beside its
surfaces as neighbours all the same. Toward a neighbouring pixel that has
a transparent surface, and away from one, the opaque resolve blends the
mark the G-buffer alone gives with the mark that surface would give if it
were in the G-buffer, by its opacity.

Frame tests hold these on both adapters (0.999 is stored as opacity 1, so
"just short of opaque" is the limit itself, reached through the
transparent path):

- every mesh just short of opaque: the all-opaque frame;
- one mesh just short of opaque among opaque ones, in front of them,
  behind them and passing through them: the all-opaque frame;
- a surface behind a transparent mesh of one, two or three walls takes no
  line where that mesh ends, and is edged alike just short of opaque and
  opaque, on either side of the frame and when it slopes steeply;
- a mesh too faint to see leaves no mark on a mesh it passes through,
  lies close to or ends flush against, and no gap in that mesh's outline;
- a transparent surface lying close in front of or behind another
  object's (a hundredth of a metre, three metres off) is edged at its
  edges only.

A second measure is a set of randomised scenes (boxes, spheres,
double-sided meshes, oblique cameras) that a review of this work wrote;
they are not in the repository. On them, 2026-11-17: one mesh just short
of opaque among opaque ones, 0 of 480 scenes differ from the all-opaque
frame; a surface behind one transparent mesh, 12 of 120 differ, by 4
pixels at most; a faint mesh over another, 68 of 360 differ, by 26 pixels
at most (128×96 frames).

What is not exact, and why:

- **Creases where a surface changes layer.** A crease needs the
  neighbour's normal, and a layer's normals are held only while that
  layer is lit. Where a transparent mesh's outline crosses a crease of
  the surface behind it, a pixel of the crease can be missing. This is
  most of the remainder above.
- **Stacks of three.** The opaque scene reads the nearest layer only, so
  a mark it owes to a transparent mesh can be missing where another
  transparent mesh is in front of that one: one side of a two-pixel
  outline, along part of it.
- **Surfaces that weave through each other** (a model and a slightly
  different one) get a line wherever they cross, as strong as the other
  is opaque; when both are opaque they get the same lines at full
  strength.
- The opaque edge test marks a surface steeper than about 2 % of its
  distance per pixel all over, and a layer inherits that.

#### Limits, as built

- **Depth.** A pixel shows the nearest `transparency_layers` transparent
  surfaces; any behind them are left out. Each layer is two passes over
  the transparent meshes (one of them writing two words a pixel) and one
  resolve. Measured 2026-11-17 on a double-sided mesh of 2.8 million
  triangles at 1920×1080: about 0.5 ms a layer on a Radeon RX 6800 XT,
  about 6 ms a layer on a Ryzen 7950X3D's integrated graphics through
  GLES.
- **No ambient occlusion.** Occlusion is gathered from the opaque scene:
  a transparent mesh neither casts nor takes it, so it appears as the
  opacity reaches 1. The occlusion radius does follow the transparent
  meshes' size, as it follows the opaque ones'.
- **Lines, points, the grid and splats** are drawn afterwards against the
  opaque scene's depth, as before: a transparent mesh neither hides nor
  dims them. Dimming what is behind one needs these passes to read the
  nearest layer's depth and the gathered coverage.
- **Surfels are always opaque.** The discs of one surface overlap, and a
  peel would count each as a surface of its own.
- **Opacity is per draw.** A vertex colour's alpha and a texture's alpha
  are still not read. The opacity target is per pixel, so they can be
  multiplied in there.
- **Coincident surfaces of one object** (the same `ObjectId` twice in
  exactly one place) are drawn once. Two transparent surfaces closer
  together than the depth buffer resolves can lose the farther one, as
  two opaque ones z-fight.

#### Rejected

- *Blending in a forward pass, meshes sorted back to front.* One pass,
  but meshes that cross or enclose each other have no order, so the
  picture changes as the view turns; the lighting model would be written
  a second time in the forward shader, and edge lines, which come from
  the G-buffer, would not exist on a transparent mesh.
- *Weighted blended order-independent transparency.* One pass and no
  order, but the result is a weighted average, not occlusion: a nearly
  opaque surface does not hide what is behind it, so a mesh does not
  arrive at its opaque look as its opacity reaches 1. It also wants a
  different blend on each of two targets, which WebGL2 lacks (wgpu's
  `INDEPENDENT_BLEND`), and a float target, which is an extension there.
- *Screen-door transparency* (discarding a pattern of pixels in the
  G-buffer fill). No new pass, but the pattern is a depth step at every
  pixel to the edge lines and noise to anti-aliasing.
- *Filling and resolving each layer once, in turn* (built first). Half
  the passes over the meshes, but a layer is then resolved before the
  layers behind it exist, and its edge test cannot tell a surface that
  ends against another transparent object from one that passes behind
  it: either a mesh's outline broke where a faint mesh lay beside it, or
  a faint mesh left full lines where it crossed another.
- *Keeping every layer's albedo and normal* so that each is filled once
  and creases can be followed from layer to layer: 17 bytes a pixel a
  layer in place of 8.
- *Marking the vertex shader's position `@invariant`*, for the peel's
  comparison of a fragment's depth with its own from an earlier pass. The
  attribute moved opaque pixels by a few levels on the GLES adapter, and
  the peel holds without it on both adapters.
- *Opacity in 8 bits as it is.* See above: a fade's faint end steps.

### Shading

The resolve pass implements one model, driven by a `LightingRig` value so
later looks are data rather than new passes:

- three directional lights given in the camera's frame (key from the upper
  left, a weaker fill, a rim), so no face is ever unlit as the view turns.
  The frame's axes are read from the view-projection, so a photograph's
  pinhole view lights the same way as the orbit camera;
- a hemisphere ambient term along world Z (`sky`, `ground`);
- a Blinn-Phong highlight per material;
- a material table of up to 16 entries (`MaterialId` indexes it: base
  tint, roughness, specular strength). Vertex colour multiplies the base
  tint;
- occlusion takes the ambient light whole and the direct light by half;
- a shoulder above 0.8 so highlights roll off instead of clipping.

Edge lines come from discontinuities in the G-buffer, in the resolve pass:

- a step in depth, marked on its near side: the neighbour is farther than
  the plane through this pixel and the opposite neighbour would put it
  (depth is linear across a plane on screen, so the plane is extrapolated
  in depth and compared in world distance);
- a crease sharper than `EdgeSettings::crease_angle`;
- a change of `ObjectId`. Both hosts number each drawn mesh, so the parts
  of an assembly are outlined against each other.

They are one pixel wide, drawn before anti-aliasing, and any geometry
source gets them.

### Supersampling

A headless frame is drawn at twice its delivered size per side and scaled
down in linear light (`offscreen::downsample_rgba`).
`RenderSettings::pixel_scale` tells the renderer how many target pixels
make one delivered pixel; every width given in pixels (lines, points, grid
lines, edge lines) is multiplied by it, so the picture is the same and
only smoother. FXAA still runs first.

Only plain frames are supersampled. A frame drawn through a lens, over a
photograph, with marks, or of a splat keeps its own size: those are
compared pixel for pixel with photographs and reference renders, and a
splat's footprint is defined in the pixels of the frame it was trained at.

Reserved for later, and the reason the id target exists beyond picking:
hover and selection outlines, per-object visibility and section planes.

### Lines and points

The pipelines keep their vertex-expansion design. One correction: every
immediate batch keeps its own style. Today all immediate batches of a depth
mode are merged and drawn with the last batch's style, and the grid is in
the same merge.

Meshes are always retained. The immediate mesh path (flatten on the CPU
every frame) is removed; immediate submission remains for small dynamic
lines and points such as gizmos and bounds boxes.

## 5. Scene widgets

### Grid

A full-screen pass (`grid.wgsl`) intersects each pixel's ray with the grid
plane and draws analytic, anti-aliased lines; there is no line geometry and
no extent.

- Three levels are drawn: minor lines, major lines every ten, super-major
  lines every hundred. With automatic spacing the minor spacing is the
  power of ten that keeps a minor cell between 16 and 160 logical pixels
  at the view's focus depth (`GridSpacing::MIN_CELL_PX`; the first build's
  8 physical pixels read as too fine in the GUI). A level's weight depends only on how large its cells
  are on screen, so when the view zooms across a decade each level takes
  over the weight the next one had and nothing pops.
- Per pixel, each direction's lines fade out as they draw closer than a few
  pixels, which is what clears the horizon and grazing views.
- The pass is depth-tested against the scene with the depth of the plane
  intersection, pushed a pixel's width along the ray so a surface lying in
  the plane wins. It does not write depth (revised from the first draft:
  a transparent plane in the depth buffer would hide lines and splats
  behind cells where nothing is drawn). The grid is dimmer from below.
- The frame reports the minor spacing (`FrameInfo::grid_spacing`); the GUI
  shows it as a scale readout ("grid 10 mm").

Automatic spacing needs the depth the view is looking at, which the view's
matrices do not hold, and the display's scale factor: the host passes both
(`GridSpacing::Auto { focus_depth, min_cell_px }`).

### World axis lines

The two world axes in the grid plane are grid lines in their axis colours
(X red, Y green on the default plane). The third (Z, blue) is drawn by a
second instance of the same pass: each pixel finds the point of the axis
its ray passes closest to and is covered by its distance from that point's
projection. It has no extent either, which the line pipeline cannot do: it
does not clip segments that cross behind the eye.

### Settings

```rust
pub struct GridSettings {
    pub visible: bool,
    pub plane: GridPlane,      // XY (default), XZ, YZ
    pub spacing: GridSpacing,  // Auto { focus_depth } | Fixed(metres)
    pub axes: bool,
    pub opacity: f32,
    pub line_color: [f32; 3],
    pub axis_colors: [[f32; 3]; 3],
}
```

### View gizmo

Three world axes drawn in the top-right corner of the viewport, following
the camera's orientation only. Positive ends are labelled discs (X, Y, Z in
red, green, blue) joined to the centre by an arm; negative ends are smaller
ringed discs.

It is interactive:

- the part under the pointer is lit;
- clicking an end animates to that standard view, and clicking the end
  already facing the viewer goes to the opposite view;
- dragging on the gizmo orbits about the scene.

`ViewGizmo::ends` lays the gizmo out from the camera orientation, a centre
and a radius. Drawing and `ViewGizmo::hit_test` both use that layout, so
what is clickable is exactly what is drawn. The host routes primary-button
presses to it before the `Navigator`.

It is drawn by its own pass over the finished frame (`gizmo.wgsl`): every
shape, the letters included, is a distance function of the pixel, painted
back to front. Revised from the first draft, which stroked it with the
line pipeline: lines and points are drawn in separate groups there, so
letters could not be laid over their discs.

A photograph's viewpoint (look-through) and headless renders have no
gizmo.

A view cube is a possible later replacement; the layout-plus-hit-test
contract would not change.

## 6. Picking

`Renderer::request_pick(pixel)` copies one texel of the retained G-buffer's
surface target (object id and depth) into a 1×1 `Rgba32Uint` target and
reads it back asynchronously. `Renderer::pick_result()` returns the latest answer as
`Pick { object: ObjectId, world: Option<Vec3> }`.

- The G-buffer of the last rendered frame is retained, so picking does not
  need a new frame.
- Where the frame drew a transparent mesh, the texel comes from the
  nearest layer's surface target: a pick lands on the nearest surface,
  transparent or not.
- A native host waits for the pick at the press (`pick_now`, well under a
  frame). The web cannot wait, so it requests a pick as the pointer moves
  over the viewport and a press takes the answer already there.
- An integer colour target is used because WebGL2 cannot read back depth
  attachments but guarantees integer read-back. The path is the same on
  every backend.

This replaces the per-press CPU ray-versus-every-triangle test for part
drags, and it works for any geometry source.

Known limit: lines, points and splats are not geometry sources. They do
not write the surface target, so a pick over a point-cloud or splat
preview finds nothing and navigation treats it as background (section 3). Giving
them a depth-only source is the fix if that fallback feels wrong in use;
it needs a second pass for them, because WebGL2 cannot blend one colour
target while writing another unblended.

## 7. Host API

```rust
renderer.submit_retained_mesh(&mesh, transform, object_id, material_id, opacity);
renderer.submit_retained_lines(&lines);   // points, splats likewise
renderer.submit_lines(&data, transform, style); // small dynamic batches
let info: FrameInfo = renderer.render(&device, &queue, &mut encoder, &view, &settings, &target);
renderer.request_pick(pixel);
```

`FrameInfo` carries what a frame reports back: geometry dropped at device
buffer limits (today's `GeometryOverflow`) and the grid's current spacing.

`RenderSettings` groups the `LightingRig`, the material table, occlusion,
edges, anti-aliasing, `GridSettings`, the gizmo, the background, the
pixel scale and how many layers deep transparent meshes are drawn.

Both hosts submit every mesh at opacity 1: nothing in the GUI or the
headless render sets one yet.

The GUI holds a `ViewportSettings` value (projection, grid visibility,
axes and density, gizmo, `LightingPreset`, edges and their strength,
occlusion and its radius and strength, anti-aliasing). The view settings
panel behind the viewport toolbar's gear edits it, with the navigation
settings (mouse controls, orbit mode) beside it; the settings file keeps
it; the session turns it into `RenderSettings` each frame. Viewport-wide
settings used to ride on each output's preview request, where the first
output's copy won and a viewport with no outputs fell back to defaults.

Headless renders set the same things from `RenderOptions`: `lighting`,
`edges`, `antialias`, `supersample`, with matching CLI flags and Python
keywords.

## 8. WebGL2

Nothing in the core frame needs compute shaders or float colour targets.

| Feature | WebGL2 behaviour |
|---|---|
| G-buffer (3 colour targets + depth) | Same |
| Grid, edges, FXAA, gizmo | Same |
| Picking | Same (integer read-back) |
| Transparent meshes | Same, up to 8 layers |
| Ambient occlusion | Half the samples (8) |
| Image sources / direct raytracing | Unavailable at first: the caster is CPU-parallel and the web build has no workers |

Measured 2026-10-05 on a GLES adapter under WebGL2 limits (the frame
tests run every scenario there as well as on the default adapter):
`Rgb10a2Unorm` and `Rg32Uint` render targets and the `Rgba32Uint`
read-back work; sampling `Depth24Plus` does not, which is why depth lives
in the surface target. A native GLES adapter is the nearest stand-in for
WebGL2 that runs headless, not WebGL2 itself: the browser build still
needs one manual check.

## 9. Contract for direct raytracing (follow-up)

Models are wasm occupancy functions: no distance field, one sample per
call, on the CPU. The viewport caster will therefore be a progressive CPU
renderer, and the renderer's obligations to it are fixed here:

- an **image source** accepts a depth image in the frame's own view and
  projection, with optional normal, colour and id images, at a resolution
  that may be lower than the frame's;
- it is written with fragment depth and into the surface target, so it
  composes with meshes, the grid, lines and overlays;
- where no normal is supplied, the resolve pass reconstructs it from depth,
  so a depth-only caster is fully shaded, occluded and edge-lined;
- picking works on it unchanged;
- the source is replaced without re-rendering other sources' buffers, so a
  coarse image shown during camera motion can be refined at rest.

The caster is built to `DIRECT_CASTING_PLAN.md` (repo root): the search
library (`src/direct_cast.rs`, which the icon thumbnails use) and the surfel
disc source above, which `render --direct` draws with. The viewport mode is
that plan's step 3.

## 10. What is deleted

- The frame in `Renderer::render_view`, the lit "G-buffer" mesh shader, the
  SSAO and composite passes, and the float depth colour target.
- The spherical `Camera` (theta, phi, radius, Y-up), `ViewDirection`'s Y-up
  poses, `fit_clip_planes`, `CameraUniforms`.
- `GridSettings::generate_lines` and the grid line cache.
- `AxisIndicator`, `generate_axis_indicator_lines`,
  `axis_indicator_projection`: settings and helpers nothing draws.
- The immediate mesh path and the unused `ssao_samples` and
  `RendererCapabilities::recommended_ssao_samples`.
- The crate-wide `#![allow(dead_code)]` attributes.
- In `volumetric_render`: the `--up` rotation trick around a Y-up camera.
  `--up` defaults to Z.

## 11. Build order

Each step lands on its own with its tests. Headless frames through the
existing `Offscreen` harness are the verification instrument: tests assert
measured properties of rendered frames, not stored images.

1. **Frame core (built 2026-10-05).** Delete the old frame; build the G-buffer, the mesh
   source with object and material ids, a resolve pass at lighting parity
   with today, per-batch line and point styles, and the pick pass. Verify
   the three WebGL2 items in section 8.
2. **Camera and navigation (built 2026-10-05).** Z-up everywhere; the new `Camera` and
   `Navigator`; pick-driven orbit (turntable and free), zoom and pan;
   standard views, projection toggle and framing; GUI controls and shortcuts.
   Left for later steps: hosts still submit every mesh as
   `ObjectId::NONE`, and dragging an assembly part still finds the part
   with the CPU ray test; the toolbar's View menu (standard views,
   projection, orbit mode, reset) is a placeholder for step 5's panel; the
   web build's hover-tracked pick has not been exercised in a browser.
3. **Widgets (built 2026-10-05).** Grid pass, axis lines, view gizmo with
   hit-testing, scale readout. Left for step 5: every grid and gizmo
   setting but the grid's visibility is fixed in code; the headless
   `--grid` is still a fixed spacing in metres.
4. **Shading (built 2026-10-05).** Lighting rig and material table, AO
   rewrite, edge lines, FXAA. Left for step 5: the rig, materials and
   edge settings are fixed in code; the GUI still calls occlusion "SSAO"
   and exposes only its radius (now a fraction of the scene) and strength;
   hosts give every mesh material 0. Not built: half-resolution occlusion
   on WebGL2 (it takes half the samples instead). Supersampling is a
   fixed 2x in the headless render, with no option yet.
5. **Settings and parity (built 2026-10-05).** The GUI's view settings
   panel (section 7) and its persistence; the lighting, edge, anti-aliasing
   and supersampling options of the headless render, the CLI and the
   Python binding; documentation.

Left open at the end of the arc:

- every mesh is drawn with material 0: nothing assigns materials yet, and
  the panel has no material controls;
- dragging an assembly part still finds the part with a CPU ray test
  instead of the pick's object id;
- picks do not see lines, points or splats (section 6);
- the headless `--grid` is a fixed spacing in metres, not the automatic
  one;
- on WebGL2 occlusion takes half the samples instead of running at half
  resolution;
- the web build's hover-tracked pick and the WebGL2 path as a whole have
  been exercised on a native GLES adapter, not in a browser.

## 12. Rulings on former open questions (2026-10-05)

1. Free orbit is built in this arc, as a setting beside turntable.
2. `render --up` defaults to Z; nothing outside this repository depends on
   the Y default, so the break is made without a compatibility path.
3. The grid is always at z = 0. No print-bed mode.
4. The viewport configuration menu is in scope and is more than a list of
   toggles: step 5 designs it as a proper panel (navigation, lighting,
   grid, gizmo, quality) when the settings it exposes exist.
