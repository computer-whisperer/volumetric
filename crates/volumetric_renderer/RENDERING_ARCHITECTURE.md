# Volumetric Renderer: Target Design

Status: **target design, not yet built** (written 2026-10-05). This file
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
- Clip planes are fitted to the scene bounds as seen from the eye
  (`volumetric_preview::clip_planes_for` already does this for look-through
  and the CLI). The orbit-radius-relative planes go away, because zooming to
  the cursor makes `distance` a poor proxy for where the scene is.

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
the cursor is over background, `P` is where the cursor ray meets the plane
through `focus` facing the camera.

| Gesture | Behaviour | Invariant (unit-tested) |
|---|---|---|
| Orbit | Rotate the camera rigidly about `P`. Turntable: yaw about world Z, pitch about the camera's right axis, elevation limited to exactly ±90 degrees | `P` does not move on screen; the horizon stays level |
| Zoom (wheel or drag) | Scale by `s` about `P`: `distance *= s`, `focus = P + s * (focus - P)` | `P` stays under the cursor, in both projections |
| Pan | Translate in the view plane at `P`'s depth | `P` tracks the cursor 1:1 |
| Frame | Fit the bounding sphere to the narrower of the two field-of-view angles | The whole scene is inside the frame at any aspect ratio |
| Standard view | Animated reorientation | Ends on the exact pose |
| Projection toggle | Perspective ↔ orthographic | Size at the focus is unchanged |

Rotation about `P` by `R` is `focus = P + R * (focus - P)`,
`orientation = R * orientation`.

Free (trackball) orbit is not built now. Storing a quaternion keeps it
possible, and lets a look-through handover keep the photograph's roll.

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
| Object id | `R32Uint` | `ObjectId` of the draw; 0 is background |
| Depth | `Depth24Plus`, sampled | Hardware depth; linear depth is recovered through the inverse projection |

No lighting happens while the G-buffer is filled. The separate float
"depth as colour" target of the MVP is gone: passes sample the depth
attachment directly, which WebGL2 allows.

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
3. **Resolve**: lighting × AO, edge lines, background, tone map, into a
   display-referred intermediate.
4. **FXAA** into the scene target.
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

A full-screen pass intersects each pixel's ray with the grid plane and draws
analytic, anti-aliased lines; there is no line geometry and no extent.

- Spacing is chosen per frame from the size of a pixel on the plane, in
  powers of ten, with ten minor cells per major cell. Adjacent decades
  cross-fade, so zooming never pops.
- Lines fade toward the horizon and at grazing angles.
- The pass writes depth from the plane intersection and is depth-tested, so
  geometry occludes the grid and the grid is dimmer seen from below.
- The frame reports the current minor spacing so the UI can show a scale
  readout ("10 mm").

### World axis lines

The X axis (red) and Y axis (green) are drawn by the grid pass on the ground
plane. The Z axis (blue) is a depth-tested line through the origin.

### Settings

```rust
pub struct GridSettings {
    pub visible: bool,
    pub plane: GridPlane,      // XY (default), XZ, YZ
    pub spacing: GridSpacing,  // Auto | Fixed(metres)
    pub axes: bool,
    pub opacity: f32,
    pub colors: GridColors,
}
```

### View gizmo

Three world axes drawn in a corner of the viewport with an orthographic
projection that follows the camera's orientation only. Positive ends are
labelled discs (X, Y, Z in red, green, blue); negative ends are small
unlabelled discs. The letters are stroked with the line pipeline, so the
renderer needs no text system.

It is interactive:

- clicking an end animates to that standard view;
- dragging on the gizmo orbits.

One function lays the gizmo out from the camera orientation and the corner
rectangle. Drawing and `ViewGizmo::hit_test(pointer)` both use that layout,
so what is clickable is exactly what is drawn. The host routes pointer
events to it before the `Navigator`.

A view cube is a possible later replacement; the layout-plus-hit-test
contract would not change.

## 6. Picking

`Renderer::request_pick(pixel)` renders one texel of the retained G-buffer
(object id and depth) into a 1×1 `Rgba32Uint` target and reads it back
asynchronously. `Renderer::pick_result()` returns the latest answer as
`Pick { object: ObjectId, world: Option<Vec3> }`.

- The G-buffer of the last rendered frame is retained, so picking does not
  need a new frame.
- The host requests a pick as the pointer moves over the viewport, so a
  button press finds the answer already there instead of waiting a frame.
- An integer colour target is used because WebGL2 cannot read back depth
  attachments but guarantees integer read-back. The path is the same on
  every backend.

This replaces the per-press CPU ray-versus-every-triangle test for part
drags, and it works for any geometry source.

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

To verify on the GL backend before relying on them: sampling `Depth24Plus`
in a later pass, `Rgb10a2Unorm` as a render target, and `Rgba32Uint`
read-back.

## 9. Contract for direct raytracing (follow-up)

Models are wasm occupancy functions: no distance field, one sample per
call, on the CPU. The viewport caster will therefore be a progressive CPU
renderer, and the renderer's obligations to it are fixed here:

- an **image source** accepts a depth image in the frame's own view and
  projection, with optional normal, colour and id images, at a resolution
  that may be lower than the frame's;
- it is written with fragment depth, so it composes with meshes, the grid,
  lines and overlays;
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

1. **Frame core.** Delete the old frame; build the G-buffer, the mesh
   source with object and material ids, a resolve pass at lighting parity
   with today, per-batch line and point styles, and the pick pass. Verify
   the three WebGL2 items in section 8.
2. **Camera and navigation.** Z-up everywhere; the new `Camera` and
   `Navigator`; pick-driven orbit, zoom and pan; standard views, projection
   toggle and framing; GUI controls and shortcuts.
3. **Widgets.** Grid pass, axis lines, view gizmo with hit-testing, scale
   readout.
4. **Shading.** Lighting rig and material table, AO rewrite, edge lines,
   FXAA, supersampled headless renders.
5. **Settings and parity.** Viewport settings panel and persistence
   (migrating the stored SSAO radius from metres to a relative value); CLI
   and Python options; documentation.

## 12. Open questions

1. Turntable is the default orbit. Is a free-orbit mode wanted in this arc
   or left for later?
2. Changing `render --up` to default to Z changes the output of scripts
   that rely on the Y default. Examples in this repository will be updated;
   is there anything outside it?
3. Should the grid sit at z = 0 always, or optionally at the bottom of the
   scene bounds (a "print bed" mode)?
