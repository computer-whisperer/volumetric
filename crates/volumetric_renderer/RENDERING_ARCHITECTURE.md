# Volumetric Renderer: Target Design

Status: **target design, being built** (written 2026-10-05; steps 1 to 3
of section 11 are built). This file
replaces the May 2026 MVP architecture notes. It is the ground truth for the
renderer overhaul: where the code and this file disagree during the rebuild,
this file wins until it is deliberately revised.

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
   mesh rasteriser      albedo            lighting              │
   image sources*       normal            edges                 ▼
                        object id         tone map       grid, lines, points
                        depth                            splats
                                                         overlay lines, points
 * follow-up                                             lens warp (look-through)
                                                         view gizmo
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
2. **Image source** (follow-up, section 9). A depth image with optional
   normal, colour and id images, produced elsewhere, written by a
   full-screen pass that sets fragment depth.

### Passes

1. **G-buffer fill** from every source.
2. **Ambient occlusion** on linear depth, followed by a depth-aware blur.
   The radius is a fraction of the scene's bounding diagonal, clamped in
   screen space, so it is right for a 20 mm part and a 2 m assembly alike.
   (Until step 4 the MVP's occlusion shader is kept, ported to the new
   G-buffer. It compares non-linear depth against a fixed bias. With the
   viewport's clip planes (near 0.005 and far 100 orbit distances) the
   default bias of 0.025 is only exceeded by an occluder nearer than about
   a sixth of the orbit distance, so by that arithmetic it registers no
   local occlusion at all. The frame test has to lower the bias to see an
   effect.)
3. **Resolve**: lighting × AO, edge lines, background, tone map, into a
   display-referred intermediate.
4. **FXAA** into the scene target (built 2026-10-05, ahead of the rest of
   step 4; `RenderSettings::antialiasing`).
5. **Grid** (section 5), depth-tested and blended.
6. **Depth-tested lines and points.**
7. **Splats**: own layer, then composited (unchanged).
8. **Overlay lines and points** (no depth test).
9. **Lens warp** when looking through a photograph (unchanged).
10. **View gizmo** on the final target.

The **pick pass** (section 6) runs on demand, outside this sequence.

### Shading

The resolve pass implements one model, driven by a `LightingRig` value so
later looks are data rather than new passes:

- lights are attached to the camera (key from upper left, a weaker fill, a
  rim), so no face is ever unlit as the view turns;
- a hemisphere ambient term along world Z;
- a specular lobe per material;
- a small material table (`MaterialId` indexes it: base tint, roughness,
  specular strength). Vertex colour multiplies the base tint.

Edge lines come from discontinuities in depth, normal and object id in the
resolve pass: silhouettes, creases and part boundaries, one pixel wide,
switchable. Because they are computed from the G-buffer, any source gets
them.

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
renderer.submit_retained_mesh(&mesh, transform, object_id, material_id);
renderer.submit_retained_lines(&lines);   // points, splats likewise
renderer.submit_lines(&data, transform, style); // small dynamic batches
let info: FrameInfo = renderer.render(&mut encoder, &FrameDesc { view, settings, target });
renderer.request_pick(pixel);
```

`FrameInfo` carries what a frame reports back: geometry dropped at device
buffer limits (today's `GeometryOverflow`) and the grid's current spacing.

`RenderSettings` groups `LightingRig`, AO, edges, anti-aliasing,
`GridSettings`, gizmo and background settings. The GUI persists them and
exposes them in a viewport settings panel; headless renders set them from
options.

## 8. WebGL2

Nothing in the core frame needs compute shaders or float colour targets.

| Feature | WebGL2 behaviour |
|---|---|
| G-buffer (3 colour targets + depth) | Same |
| Grid, edges, FXAA, gizmo | Same |
| Picking | Same (integer read-back) |
| Ambient occlusion | Fewer samples, half resolution |
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

`src/direct_preview.rs` (the icon thumbnails) is the seed of that caster and
is not touched by this arc, except that it already agrees with Z-up.

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
4. **Shading.** Lighting rig and material table, AO rewrite, edge lines,
   supersampled headless renders. (FXAA is built.)
5. **Settings and parity.** Viewport settings panel and persistence
   (migrating the stored SSAO radius from metres to a relative value); CLI
   and Python options; documentation.

## 12. Rulings on former open questions (2026-10-05)

1. Free orbit is built in this arc, as a setting beside turntable.
2. `render --up` defaults to Z; nothing outside this repository depends on
   the Y default, so the break is made without a compatibility path.
3. The grid is always at z = 0. No print-bed mode.
4. The viewport configuration menu is in scope and is more than a list of
   toggles: step 5 designs it as a proper panel (navigation, lighting,
   grid, gizmo, quality) when the settings it exposes exist.
