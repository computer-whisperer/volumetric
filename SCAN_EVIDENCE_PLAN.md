# Scan Evidence — Design and Plan

Status: ratified 2026-09-09; reframed 2026-09-11 for stills-only capture,
the survey card and splats. Step 0 landed 2026-09-09. Step A landed
2026-09-09 (value, import, CLI, GUI). Step B landed 2026-09-10 (cv_core,
view-solve, view_solve_operator, GUI Import Still). C2 (card detection)
landed 2026-09-11. C3 (survey bundle) landed 2026-09-11 together with the
parts of C1 it needed (schema 2, `view-import --stills`, Sony focus keys);
C1's remaining audits fold into D. S1 (splat value, import, listing,
points) and S2 (splat rendering) landed 2026-09-11, P1+P2 (Python bindings:
projects, cv, view sets, splats, measuring, sampling, intake) 2026-09-12.
C4, S3, P3 and D (audits) pending; order C4, D, P3, S3.

## Why

Volumetric's purpose is to let an agent that cannot hold a tape measure do
ordinary modelling work from physical evidence a human collects. The
evidence is now a set of stills: a camera (a Sony a6700 with a 50 mm lens,
or a phone) photographs the subject on and around a marker field, and one
rigid reference in the field, the survey card, gives the scale and the
datum. A bundle over the card's chessboard corners and the swatch corners
solves the cameras, the poses and every corner as a free point, to a few
hundredths of a millimetre on the card and a few tenths across a metre of
field. Nothing but the camera, the printed card and a caliper is needed.
(Until 2026-09-10 the evidence came from a Valve Index used as a stereo
scanner on a lighthouse-tracked pose; the stills survey beats its map by
an order of magnitude and needs no rig, so the Index path is dropped.)

The scanning project (`~/workspace/playground/index_scanner`, Python) is
a temporary shim: it is where each stage is prototyped, and its cached
outputs are the oracles for the Rust version. What lands here is the
mature form of the same concepts, and the shim is retired stage by stage.
The one stage that stays outside is Gaussian-splat training (gsplat on a
CUDA node, with SAM 2 masks): it is the GPU service of the pipeline.
Volumetric prepares its input, imports its output, and represents and
renders the splat itself, so the agent can see and audit what came back.

The goal state: an agent is handed volumetric and a directory of stills
and manages everything from there. The current dogfood prioritizes useful
modelling from photographs alone; the GPU service is an optional source
of additional evidence. Two interfaces, one code path. The CLI is the agent's interface: it
must let the agent see, measure, build and verify headlessly. The GUI is
the human's audit of what the agent accepted and produced. Whatever the
CLI renders or solves, the GUI shows through the same crates. Python
bindings expose the same kernels for exploration and validation, and
never carry logic of their own.

Acceptance task updated 2026-09-12 (user): dogfood a clean-slate chair-base
model from `sessions/chairbase-dslr-1`, starting with the upper mounting
interface. Priority order: improve volumetric's agent tooling, establish
a repeatable workflow requiring minimal human measurements, then progress
the specialty chair accepting the user's wheelchair cushions. Survey the
44 bright JPEGs in volumetric, recover features from multiple views, build
in WGSL, and check the geometry against the photos. The previous
`examples/chair` work is deliberately excluded; the new example lives in
`examples/chair_photo`. Splats and clouds in `runs/chairbase-dslr-1-all-refuse`
remain available, but their use is not a prerequisite. This supersedes the
older session-0 acceptance sequence requiring remote training, because
the DSLR capture is easier to repeat than CUDA reconstruction or manual
measurement feedback. All evidence paths above are under
`/ceph/christian/index_scanner`.

## Decision record

1. **Posed image sets, not single views (reverses the arc-4 design of
   2026-09-09).** Rejected: a `View` value per image. Chose: a `ViewSet`
   value holding intrinsics, poses, image/depth/mask blobs for a chosen
   subset of a dataset, and the marker map. Because: real datasets are
   1542 views, 1.7 GB of images and 340 MB of depth; a project embeds a
   curated subset, and the marker map is what poses a new still.
2. **The marker pipeline comes into volumetric (revised 2026-09-11).** In:
   still intake with EXIF and maker-note camera keys, ArUco and AprilTag
   detection with sub-pixel refinement, ChArUco corner localisation, PnP
   and focal solving, radial and fisheye (Kannala-Brandt) models, the
   survey bundle (cameras per focus setting, poses, corners as free
   points, scale from the card), the trainer's dataset export, and the
   trained splat as a value with rendering and photometric audits. Out:
   capture, device calibration, structure from motion and SIFT seeding,
   SAM 2, splat training. Dropped from the 2026-09-09 list: marker-map
   refinement against the lighthouse track (no lighthouse; the survey
   replaces it) and TSDF fusion from posed depth (no depth source any
   more; fusion happens in the trainer, and may return here as depth
   rendered from the splat, see "Later"). Because: the target card and
   the survey card are what turn photographs into evidence, and none of
   the "in" items needs hardware or a GPU.
3. **Provenance and audits are first-class.** Evidence values carry a
   provenance record (session, rig calibration hash with noise model,
   marker field id, subject setup id, tool versions). Combinators refuse
   to merge evidence from different fields or setups; crossing frames is
   an explicit alignment step with a residual. Statistical audits compare
   residuals to the rig's declared noise and surface as warnings on the
   existing warning channel; categorical mismatches are step errors.
4. **Formats at the boundary.** Heavy artefacts use standards: PLY (clouds
   and the 3DGS splat layout), NRRD and numpy `.npz` for grids, 16-bit
   PNG depth, PNG/JPEG images. The posed-image manifest is imported from
   the shim's `cameras.json` (schema 2) and from nerfstudio
   `transforms.json` with two documented extensions (world units are
   metres, `depth_unit_m`), and exported in the schema-2 layout for the
   trainer. The marker map is produced by volumetric, by the survey, not
   imported. Specs live in this repo, since it is the consumer.
5. **Values print.** Any small value (Subspace, F64Map) is readable from
   the CLI without a decoder. The agent reads numbers, not sizes.
6. **The shim is the oracle, then retired (2026-09-11).** Each stage lands
   validated against the shim's cached output for the same session
   (`detections.json`, `survey.json`, the trainer's `metrics.json`), the
   way step B was validated against COLMAP and OpenCV. Once a stage is
   validated the shim's version is not extended; the shim keeps capture
   and GPU dispatch until a GPU-service step exists here.
7. **Splats are an evidence value (2026-09-11).** A `Splat` value (3D
   Gaussians or 2D surfels: means, scales, rotations, opacities, spherical
   harmonics, provenance naming the view set it was trained from) is
   imported from the trainer's PLY, drawn in the viewport and by `render`
   with sorted alpha blending, looked through over the photographs, and
   converted to a point cloud with normals for the fit tools. Rejected:
   treating the splat as an opaque blob and working only from the
   trainer's TSDF surface. Because: the splat is the pipeline's product,
   the photographs are its ground truth, and comparing the two per view
   is the audit that says whether the geometry can be trusted.
8. **Python bindings over the same kernels (2026-09-11).** A PyO3 module
   exposes detection, pose solving, the survey, view sets, splats and
   project runs on numpy types. The CLI stays the canonical agent surface
   and the operator the durable pipeline stage; the bindings are the
   workbench for validation loops and plots, and the way the shim can
   call volumetric's kernels instead of its own. Because: the exploration
   half of this work is Python by reflex, and a per-call process with
   bespoke flags and JSON shapes is friction that in-process arrays are
   not.

## Steps

| Step | Content | Size | Status |
|---|---|---|---|
| 0 | Values in `project-run --json`; one `render` for a whole project scene with an explicit pinhole camera | 2 days | landed |
| A | `view_core`: ViewSet value with provenance, manifest import, look-through render, depth residual | 4 days | landed |
| B | `cv_core`: ArUco detection, PnP, focal; `view-solve` CLI, operator, GUI drop-a-still | 5 days | landed |
| C1 | Still intake: a view set from a directory of stills, camera keys from EXIF and the Sony maker note, intake audits | 2 days | landed with C3 (audits → D) |
| C2 | Card detection: AprilTag 36h11 as data, ChArUco corners refined, pyramid search for 26 MP frames, observations stored per view | 3 days | landed |
| C3 | Survey bundle: cameras per focus key, poses, corners free, scale from the card, uncertainties; `view-survey`, operator, GUI | 4 days | landed |
| C4 | Trainer bridges: dataset export in the schema-2 layout with a box seed; `.npz` TSDF import | 2 days | pending |
| S1 | Splat value, 3DGS PLY import, `splat-list`, splat to point cloud | 2 days | landed |
| S2 | Splat rendering in the viewport and `render`, look-through over the photograph | 4 days | landed |
| S3 | Photometric audit: the splat rendered through each view against its photograph | 2 days | pending |
| P | Python bindings (PyO3, maturin) over cv_core, view_core, the survey, splats and projects | 3 days | P1+P2 landed 2026-09-12; P3 (render, add_views, crop, set_config, observations value) pending |
| D | Evidence audits for still sets: intake, detection, survey, setup, coverage, photometric, physical | 3 days | pending |

Proposed order: C2, C3 (the survey is the stage the agent starts from and
its oracle is fresh), S1, S2 (the splat is what it ends with and cannot
be seen here today), P (once there are kernels worth binding), C4, D,
S3. (C1 was to follow P, but C3 has no input without it: the stills
intake landed with C3.) Items that need no format decision (the frame-datum operator and
`measure` command under "Later") can land at any time.

## Step 0 — see and read

### 0a. Values in `project-run`

`project-run --json` gains a `value` field per export for Subspace
(`dimensions`, `origin`, `basis` as rows) and F64Map (the map). The text
output prints the same. `project-export` is unchanged; the JSON is the
agent's path. Baked projects already open hot, so no separate
`project-show` command.

### 0b. One render path

The CLI's `render` gets rebuilt on the GUI's preview path instead of its
own renderer (`headless_renderer.rs`, `camera.rs`, `shaders/`, which are
deleted). The preview builders move out of `volumetric_ui_v2::session`
into a new `volumetric_preview` crate (types `PreviewRequest`,
`PreviewPlan`, `PreviewMeshPlan`, `Asn2Settings`, `OutputStats`,
`PreviewEntity`, `PreviewBounds`, `ExecutionBackend`, `LocalBackend`, the
per-kind builders, and `submit_subspace_gizmo`); the UI re-exports them so
its paths do not change. The crate depends on `volumetric` and
`volumetric_renderer` only, and builds for the web target like the UI.

`volumetric_renderer` gains an explicit camera: `render` accepts a
`CameraView { view, projection }` built from the orbit `Camera` or from a
pinhole (`fx, fy, cx, cy, width, height` and an OpenCV camera-to-world
pose), plus a native-only offscreen helper that renders to RGBA bytes.

New `render` surface:

```
render -i <model.wasm | project.vproj> -o out.png
       [--asset id]...              renderable exports to draw (default: all)
       [--views iso,front,...]      preset directions, one file per view
       [--camera-pos x,y,z --camera-target x,y,z --camera-up x,y,z --fov deg]
       [--intrinsics fx,fy,cx,cy --pose m00,...,m23]   pinhole through an OpenCV pose
       [--projection perspective|ortho --ortho-scale s]
       [--width --height --background hex]
       [--resolution N --no-sharp --no-simplify]      model meshing plan
       [--color-channel name | --color-field node:name]
       [--wireframe --grid m --no-ssao] [-q]
```

Models are meshed with the GUI's ASN2 plan, triangle meshes and 2D
sketches draw as they do in the viewport, point clouds draw as points with
their colours, Subspaces draw as gizmos sized by the union bounds. Per
asset, the stats (triangles, points, bounds) go to stderr. The four
`headless_*_preview` examples in the UI crate are superseded and deleted.

Tests are GPU-free: the pinhole camera projects a known world point to a
known pixel; asset selection, plan choice and pose parsing are
unit-tested. Verified on real data: the chairbase1 cloud rendered through
scan view 600's intrinsics and pose overlays the photograph, so the
OpenCV convention is right end to end.

## Step A — ViewSet

Status: landed 2026-09-09 (value, import and CLI, then the GUI).
Verified on chairbase1: eight left views imported in 10 MB; the TSDF model
rendered through view 600 overlays its photograph (blend, edge, checker);
`view-residual` against the same model reports 82% coverage and a 5 mm
median on view 600, which is the per-frame stereo depth noise the scanner
documents, not a model error.

### Value

`volumetric_abi::viewset` (CBOR, `AssetTypeHint::ViewSet`,
`OperatorMetadataInput/Output::ViewSet`), schema 1:

- `world`: up axis (world units are always metres).
- `provenance`: `session`, `rig` (calibration id or hash), `field` (marker
  field id), `setup` (subject setup id), `tools` (producer strings),
  `captured` (ISO time). Empty strings mean unknown; step D's combinators
  will refuse to merge sets whose non-empty `field` or `setup` differ.
- `cameras`: intrinsics shared by views — `width`, `height`, `fx`, `fy`,
  `cx`, `cy` and a `distortion` enum: `None`, `Radial { k, p }` (OpenCV
  k1..k3, p1, p2), `KannalaBrandt { k }` (the Index's fisheye).
- `views`: `id`, `camera` index, `camera_to_world` (3x4 row-major, OpenCV:
  x right, y down, z forward), optional `time` (seconds), optional
  embedded `image` (PNG/JPEG bytes), `depth` (16-bit PNG bytes) with
  `depth_unit_m`, `mask` (PNG, nonzero = subject), and `tags` (eye,
  split, whatever the source knew).
- `markers`: `id`, `size_m`, four world `corners`.

Camera math lives beside the type so operators can use it: project a
camera-space point to a pixel (with distortion), a pixel to a unit ray
(inverting the distortion), and the pose helpers (position, forward,
world projection, unprojection at a depth).

### Import

`crates/view_core` (native): reads the scanner's `cameras.json` (schema
2, with `depth/` and optional `masks/` files beside `images/`) and
nerfstudio `transforms.json` (OpenGL camera axes converted; `fl_x`..`cy`
global or per frame; `depth_file_path`; units assumed metres unless
`depth_unit_m` and `world_unit_m` extensions say otherwise), selects a
subset (`ids`, `stride`, `near` point + radius, `max`, `eye`, `split`),
and embeds the chosen files. The 1542-view chair dataset is 2 GB; a
project carries the twenty to forty views a task needs.

### CLI

- `view-import --manifest <json> [-o set.vviews] [-p project --asset-id
  views]` with the selection flags and provenance labels.
- `view-list -i <set | project> [--asset id] [--json]`: cameras, views
  (id, position, forward, what is embedded), markers, provenance.
- `render --through <views asset>:<view id> [--overlay blend|edge|side|
  checker] [--overlay-alpha a]`: the frame through that view's camera,
  composited with its photograph; the image size defaults to the
  camera's.
- `view-residual -i project --views <asset> --model <asset> [--view id]...
  [--stride n] [--band m] [--json] [-o dir]`: for each view with depth,
  the model's surface along every valid pixel's ray within `band` of the
  scan depth, found by marching occupancy and bisecting the transition.
  Reports coverage (pixels whose ray meets the model near the scan
  surface), median and p90 absolute residual, mean (bias) and rms, per
  view and pooled, and writes a residual image per view (blue in front,
  red behind, grey unmatched).

### GUI

Landed 2026-09-09, over the same preview path the CLI's `render` uses (`render --asset views` draws the frustums headlessly; `chairbase1/demo/6_frustums_iso_top_through.jpg` shows them beside the model and through view 3):

- **Frustums.** `PreviewPlan::ViewSet` builds a `PreviewEntity` of retained
  lines: per view the four edges from the eye to the image corners at a
  fixed 0.1 m, the image rectangle, and a tick on its top edge (which way is
  up); a point at each eye; each marker as its square. Bounds are the union
  of eyes and corners, so Frame includes the cameras. A ViewSet counts as
  renderable everywhere an asset is pinned, selected or rendered, in the
  viewport and in `render`.
- **Views section.** The project panel grows a "Views" section whenever a
  ViewSet is in the viewport: a provenance line (views, cameras, field,
  setup), then a row per view with a thumbnail, id, tags, depth and mask
  marks, and a Look button. Thumbnails decode lazily, a couple per frame,
  and are cached by content hash and view id as Damascene images.
- **Look-through.** App state names the view being looked through and a
  photo opacity (0–100% picker). The viewport then renders through the
  view's pinhole, its intrinsics scaled and shifted into the letterbox the
  viewport gives the camera's aspect, and the photograph sits over the
  viewport in the same letterbox (Damascene `Contain` fit) at that opacity,
  unkeyed so it never takes pointer input. The looked-through frustum is
  drawn highlighted. Clip planes come from the scene bounds along the view
  axis, as in `render`. Any orbit, pan or zoom leaves look-through and
  seeds the orbit camera from the view's pose, so the user continues from
  the photograph's viewpoint; Look again or Reset also leaves it.
- **Caches.** Decoded ViewSets are cached per content hash in the app, like
  the model dimension probe; the session reuses the retained frustum lines
  across frames like any other entity.

## Step B — solve a still from the cards

Status: landed 2026-09-10 — `cv_core`, `view-solve`, the
`view_solve_operator` (the same `cv_core::still` pipeline, 1.3 s per
12 MP still through the wasm path) and the GUI's Import Still flow (a
Solve Still step wired to the project's view set; the step's warnings
carry the solve summary, and the Views section lists the still with its
`rms` and `cards` tags). Validated on the 22 chair-phone stills COLMAP
registered:
every card OpenCV found is found (plus two it missed), corners agree with
OpenCV's to 0.4–1.1 px median, poses agree with COLMAP's to 1.3 cm median
/ 3.2 cm worst and 0.27° median / 0.85° worst with the focal and k1
solved from the cards alone (rms 2.4–5.9 px, the map's own accuracy);
0.6 s per 12 MP still. The first run "failed" at 165 px rms because the
stills were taken in session chairbase2, whose cards sit 17 cm from
chairbase1's — the evidence-doesn't-make-sense case step D's `setup`
check is for, caught here by hand.

### Why this shape

A still from a phone is the cheapest evidence a human can add: one
picture of the subject with the cards in view. Posing it needs the marker
map the scan already carries, a detector, and a pose solver; none of that
needs OpenCV once the dictionaries are data in the crate. Everything runs
on the full-resolution picture (a 12 MP still holds 100–300 px markers,
and the corner accuracy is the pose accuracy).

### `crates/cv_core` (pure Rust, native and wasm32)

- `gray`: 8-bit luma image with bilinear sampling and an integral image.
- `dict`: ArUco dictionaries as data — `DICT_5X5_100` (the swatches) and
  `DICT_4X4_50` (the calibration board), each code with its four
  rotations precomputed, generated from OpenCV 5.0 by
  `gen/aruco_dicts.py`. Identification takes the rotation that matches
  within the tolerance OpenCV uses (0.6 × the dictionary's maximum
  correction bits): 1 bit for the 5x5 set, 0 for the 4x4.
- `detect`: adaptive threshold at three window sizes scaled to the
  picture (integral-image local mean, constant 7), connected dark
  components, outer-border tracing, Douglas–Peucker to a convex quad with
  a minimum perimeter, perspective sampling of the (n+2)² cells with an
  Otsu split, border check, dictionary lookup, then corner refinement by
  edge fitting: sub-pixel gradient crossings along each side, a robust
  line per side, corners at the intersections. Duplicates across windows
  collapse to the best fit. Output: id, four corners in the marker's
  canonical order (top-left first, clockwise), the rotation used, the
  Hamming distance and the fit residual.
- `pnp`: pose from correspondences between the map's marker corners and
  the detections. Initial pose from the best single-marker homography
  (DLT with normalisation, decomposition against K, polar
  orthonormalisation, the solution with the marker in front), then
  Gauss–Newton over all corners with Huber weights and a residual cut,
  rotation updated on the manifold. Optionally the focal joins the
  unknowns (fx = fy, principal point at the image centre) and its
  standard error from the normal matrix says whether the picture
  constrains it — all cards on one plane seen square-on does not, and the
  solve says so. Output: camera-to-world, rms, per-marker residuals and
  which markers were used.
- `exif`: the JPEG APP1 reader for FocalLength, FocalLengthIn35mmFilm and
  the maker and model, to seed the focal (35 mm equivalent when present,
  else a configurable field of view, default 70°).
- `board`: a synthetic renderer (markers at world poses through a pinhole,
  supersampled, optional blur) with exact ground truth for the tests of
  every stage above.

### CLI

`view-solve -p project [--views asset] --image still.jpg [--id name]
[--dictionary 5x5_100|4x4_50] [--intrinsics fx,fy,cx,cy | --fov-deg d]
[--solve-focal] [--annotate out.png] [--json] [-o set.vviews]`: detects
the cards, solves the pose against the set's marker map, reports detected
and used ids, rms, per-marker residuals and the focal (with its standard
error when solved), draws the detections when asked, and appends the view
(image embedded, tags `still`, `solved:markers`) to the set in the project
or to a standalone file.

Validation: the 22 chair-phone stills COLMAP registered into the scan
world (`work/chair-phone/poses.json`, f = 3004 px) — position and
rotation agreement, and detection agreement with the scanner's OpenCV
corners (`obs_stills.json`).

### Operator and GUI

`view_solve_operator` (ViewSet + picture blob + config → ViewSet with the
new view) runs `cv_core::still`, the pipeline the command runs, and
posts the solve summary and cautions as step warnings. The GUI's Import
Still (Solve Still in the Add catalog) opens a file dialog and adds the
step wired to the project's first view set, with a view-set picker in
the step editor for any other; the posed still then appears in the Views
section with its `rms` and `cards` tags and can be looked through, where
the map's marker squares over the photograph's cards are the visual
check of the solve.

## Step C — the survey (replaces the lighthouse map refinement)

### Why this shape

The shim's `idx-survey` showed what a still session can give: 73 of 79
frames posed at 0.72 px rms, the glued card planar to 0.031 mm over 110
corners, fourteen swatches at 0.1–0.34 mm sigma across a 0.9 m field, the
card's caliper span reproduced to 0.06 %. Every part of that is CPU work
on detections, and `cv_core` already has the quad finder, the marker
decoder, sub-pixel edge refinement, PnP and a robust Gauss-Newton. What
is missing is the AprilTag family, chessboard corners, triangulation, a
bundle with an analytic Jacobian, and the intake and export around them.

### C1. Still intake

- `view-import --stills DIR [--glob '*.JPG'] [--embed full|preview|none]
  [--session s --setup s --field s]`: one view per still, unposed. Each
  view carries `source` (the path, relative to the set's origin) and the
  original's content hash in provenance; `--embed preview` (default)
  keeps a quarter-scale JPEG for look-through and thumbnails, `full` the
  original, `none` only the reference. Commands that need full resolution
  (detection, export) read `source` when the embedded image is reduced
  and fail with the path when the file is gone.
- ViewSet schema 2: `camera_to_world` becomes optional (unposed views draw
  no frustum and are listed as such), `shot: Option<Shot>` (iso,
  exposure_s, aperture, focus mode, focus position, stabilisation,
  orientation) and `observations: Option<Observations>` (C2) per view.
  Schema-1 sets decode as posed with no shot.
- Camera keys: `make:model:focus` from EXIF, with the Sony maker note
  read for the focus position (block 0x9402 deciphered, byte 0x2d), focus
  mode (tag 0x201b) and SteadyShot, as `sonyexif.py` does; one
  `CameraModel` per key seeded from the focal length and the sensor width
  (23.5 mm for the a6700; the 35 mm equivalent or `--fov-deg` otherwise),
  principal point at the centre, k1 = k2 = 0. Autofocus frames each get
  their own key.
- Intake warnings: autofocus frames (cannot share a model), stabilisation
  on (the principal point moves per shot), ISO above a threshold, mixed
  orientations, frames without EXIF. The report lists frames per key.

Status: landed 2026-09-11 with C3, except the intake audits, which go
with D. `view-import --stills DIR [--ext jpg] [--embed preview|full|none]
[--preview-px 1600] [--sensor-mm] [--fov-deg]`; `view_core::stills`.
ViewSet schema 2: `camera_to_world: Option`, `shot: Option<Shot>`,
`source: Option<String>` relative to `provenance.origin`,
`CameraModel.label` (the camera key); schema-1 sets decode as schema 2.
`cv_core::exif` reads the exposure tags and the Sony maker note (focus
mode 0x201b, SteadyShot 0xb026, focus position from the deciphered block
0x9402 at byte 0x2d); keys are `make:model:lens:mode:position`, an
autofocus frame's key ends in its own id. On chairbase-dslr-0 the keys
match the shim's for all 79 frames (one at manual:154, 78 at manual:170;
the a6700 seeds 13 833 px from 50 mm over 23.5 mm times 1.05 for focus
breathing). The intake warns on autofocus, stabilisation, ISO above
3200, mixed orientations, missing EXIF, and keys with under three
frames. `view-detect` reads the original through `source` when the
embedded picture is reduced; the operator refuses reduced pictures.

### C2. Card detection

Status: landed 2026-09-11 (`cv_core::{dict, detect, charuco, blur,
observe}`, `view-detect`, `view_detect_operator`, the observation overlay
in look-through). Validated against the shim on chairbase-dslr-0 (79
stills): every card corner OpenCV found is found (1984 of 1984) plus 1361
more, agreeing to 0.13 px median, 0.98 px at the 99th percentile; swatch
corners agree to 1.0 px median, of which about 0.5 px is OpenCV's
`cornerSubPix` pulling a blurred L-corner inward (measured on renders:
0.3σ of the blur), so the shim's survey, fitted to those corners, is
itself offset; the caliper check on the swatches' printed size in C3 is
the arbiter. On pcb-dslr-1 (the board on the card, f/4) 0.10 px median
with the defocused frames' corners disagreeing by tens of pixels on both
sides. The chair-phone stills still pose within 1.3 cm median of COLMAP.
Detection is 0.2 s per 26 MP still. Two things the validation taught:
the contour's first pixel in raster order must not be kept as a quad
vertex (on a level edge it sits tens of pixels from the corner), and a
straight edge fit is wrong on real cards, which curl by several pixels
over a side; the marker corners come from the edge points nearest each
corner.

- `cv_core::dict` gains AprilTag 36h11 as data (587 codes of 36 bits in a
  6x6 grid, generated by `gen/aruco_dicts.py` from OpenCV's
  `DICT_APRILTAG_36h11`, tolerance from the family's correction bits as
  for the ArUco sets); the quad pipeline is unchanged. Since the card's
  tags carry ids 100–165 and the swatches 0–59, a detection run reports
  both families from one pass over the quads.
- `cv_core::charuco`: a `BoardSpec` (squares across and down, pitch across
  and down from the calipers, marker size, family, first id; the card is
  12x11 at 17.944 x 17.745 mm, ids from 100) and the corner localisation.
  Each decoded tag gives a board-to-image homography; the interior
  corners of its neighbouring squares are predicted from it and refined
  by the saddle-point iteration (gradient orthogonality over a window
  scaled to the square's size in pixels, the `cornerSubPix` method); a
  corner with one decoded tag beside it is accepted (the board-on-the-card
  case that took pcb-dslr-1 from 16 to 50 usable frames); a corner whose
  refinement leaves the prediction by more than a fraction of the square
  is dropped. Output per corner: id (row-major over the interior),
  pixel, and the fit residual.
- Pyramid: quads are found on a reduction to about 1600 px on the long
  side (a 26 MP frame at 0.5–1 m holds cells the adaptive threshold's
  windows cannot see at full size) and every sample and refinement runs
  at full resolution. The swatch detector gets the same treatment.
- `Observations { markers: [(id, corners, fit_px)], board: [(corner_id,
  pixel, fit_px)], blur_px: Option<f64> }` stored per view, so the GUI
  can draw detections over the photograph in look-through without
  detecting again. `blur_px` is the error-function edge sigma over the
  detected markers' edges (the shim's `edge_blur`), the frame-quality
  number step D uses.
- `view-detect -p project --views set [--card spec.json] [--annotate dir]
  [--json]` and `view_detect_operator` (ViewSet + config → ViewSet) run
  it for every view; the CLI prints per-view counts and residuals.
- Oracle: `work/chairbase-dslr-0/detections.json` (79 frames) and
  `work/pcb-dslr-1/detections.json` (one tag per corner). Targets: the
  same corner ids found, card corners within 0.3 px median of the shim's,
  swatch corners within 0.5 px (the shim measured 0.5–0.6 px scatter
  frame to frame, so agreement beyond that is not measurable).

### C3. Survey bundle

- `cv_core::survey`. Unknowns: per camera key f, cx, cy, k1, k2; per view
  a rotation vector and translation; per point (card corners, swatch
  corners) a free 3D position. Residuals: reprojection of every
  observation, and one scale residual, the mean of the solved column
  spans of the card against the caliper pitch times the span, heavily
  weighted. Gauge: one card frame held during the solve, then the world
  re-based by a rigid fit of the nominal card onto the solved corners
  (origin at the first interior corner, x along the columns, z from the
  card towards the cameras), so no single corner defines the frame.
- Initialisation as the shim does it: PnP on the nominal card for frames
  with enough card corners (the planar solver on all points, then robust
  refinement with the worst dropped; the mirrored planar solution is the
  failure to guard against), swatch corners triangulated from posed
  frames (linear multi-view, in front of every camera), remaining frames
  posed on solved points, repeated until nothing grows. Points seen once
  stay out of the bundle (known only along a ray).
- Solver: Levenberg-Marquardt with an analytic Jacobian and dense normal
  equations (about a thousand unknowns for eighty stills; the Schur
  complement over points when phone video brings hundreds of frames),
  soft-L1 at 1.5 px, rounds that drop frames over `reject_px` rms and
  extend again. Uncertainty from (JᵀJ)⁺σ² with the gauge freed: per point
  sigma, per camera the focal's standard error.
- Output: the ViewSet posed, its cameras replaced by the solved model per
  key, the swatches in `markers` with solved corners and sizes, and a
  `board` entry (spec, solved corners with sigma, planarity) for the card;
  plus an F64Map report: observations, rms, median, inlier rms, per-frame
  rms, card planarity, span across against the caliper, swatch side
  lengths, per-camera intrinsics with their standard errors, frames
  rejected. Views posed get tags `solved:survey` and `rms:<px>`.
- `view-survey -p project --views set [--card spec] [--f-scale 1.5]
  [--reject-px 3] [--min-card 8] [--json]` and `survey_operator`
  (ViewSet + config → ViewSet, F64Map). GUI: a catalog entry; the Views
  section shows the rms tag; frustums, swatches and the card draw once
  posed.
- Oracle: `work/chairbase-dslr-0/survey.json` (73 posed, 0.72 px rms,
  0.31 median, planarity 0.031 mm, along span 179.34 mm against 179.44
  by caliper, swatches 59.66–59.99 mm, f 13 960 px at focus 170) and
  `work/pcb-dslr-1/survey.json` (`--reject-px 5 --f-scale 2.5 --min-card
  6`: 36 frames, 2.37 px). Targets: poses within 0.2 mm and 0.02°, map
  corners within 0.1 mm, the same frames rejected.

Status: landed 2026-09-11. `cv_core::survey` (`SurveyOptions`,
`survey(&mut ViewSet) -> SurveyReport`, `SurveyReport::to_f64_map`),
`view-survey`, `survey_operator` (Survey Field; ViewSet + config →
ViewSet, F64Map). As designed, with three additions found on the way:
single observations beyond `outlier_px` (10) are dropped before frames
over `reject_px`, since one detector slip of 28 px on a swatch corner
otherwise costs the whole frame; the uncertainties are expressed in the
card's datum (the S-transformation of the gauge-free pseudo-inverse onto
inner constraints over the card's corners), which a Monte Carlo over the
noise confirms (card scatter 0.037 mm against sigma 0.035, swatches 0.36
against 0.32, f 7.6 against 9.9) where inner constraints over all points
had put a uniform 0.25 mm on every card corner; and each view gets a
position and angle sigma from its 6x6 covariance block, so a frame on
two swatches at two metres (5 mm, 0.15°) is told apart from one on the
card (3 mm, 0.05°, dominated by the principal point). The planar-field
degeneracy is the principal point: ±9 px in cy at 0.5 px noise, and every
pose tilts with it; the plan's pose target of 0.2 mm and 0.02° was not
achievable by either solver and is replaced by agreement within the
reported sigmas. Validation on chairbase-dslr-0: 74 of 79 posed (the shim
73; we also pose DSC00151), 0.90 px rms on our detections, f 13 985 ± 12
(shim 13 960), pp (3081 ± 3, 2092 ± 9) (shim 3079, 2107), card corners
0.024 mm median from the shim's, planarity 0.026 mm, along span 179.36
(calipers 179.44), swatch sides 0.05–0.2 mm larger than the shim's (our
corners sit ~0.5 px outside OpenCV's saddle-biased ones; a caliper on a
swatch is the arbiter). On the shim's own detections: 0.70 px rms (shim
0.72), card 0.003 mm, poses 1.7 mm median apart along the pp tilt. Five
seconds natively for 74 stills (dense normal equations, ~940 unknowns).
The synthetic tests cover the Jacobian against central differences,
Cholesky, Horn's rigid fit on planar points, the field to truth, the
Monte Carlo, and a frame with scrambled observations. Noted for D:
blurred frames (DSC00149, DSC00169) get many more card corners than
OpenCV accepts, with shifts up to 16 px, and land at 2.3–2.6 px rms.

### C4. Trainer bridges

- `splat-export -p project --views set -o dir [--scale 0.5] [--holdout
  10] [--seed box|none] [--near-marker id]`, native: the trainer's
  dataset. `cameras.json` schema 2 with one K (the first camera key's
  model at `scale`, distortion removed), every view remapped through its
  own model into that pinhole as `images/NNNNN_l.jpg`, `mask_l.png` the
  valid area, `init.ply` a box seed (a 1 cm grid over the card and field
  bounds to 0.25 m up), `markers.json` with the card's outer corners as
  pseudo-marker 1000 so `--near-markers 1000` works, and a holdout split.
  The agent runs the cluster script itself until a GPU-service step
  exists (see "Later").
- Import back: `splat.ply` (S1), `cloud.ply` (the PLY import), and
  `tsdf.npz` through `volume-import --npz`: a reader for the zip-of-npy
  container (`tsdf`, `weight`, `origin`, `voxel`, `trunc`) into the
  volume value the NRRD import produces, with a crop to given bounds or a
  stride at import, since the chairbase-dslr-0 grid is 2.7 GB at 1 mm.
  The TSDF is the route to a mesh through volumetric's own mesher; the
  trainer's surface cloud is the usual route to measurements.

## Step S — splats

### Why this shape

The trained splat is the pipeline's product. Today nothing outside the
trainer can show it; its held-out renders are the only audit, and the
TSDF surface it fuses is a derivative with its own noise (0.5–0.8 mm and
a bulge on grazing flanks, per the shim's measurements). Drawing the
splat in the viewport over the photograph it was trained on, through the
camera that took it, is the check that says where the geometry can be
trusted. The same renderer, run through every view, is the photometric
audit, and its depth output is how fusion can return here later.

### S1. Value and import

- `volumetric_abi::splat`, CBOR, `AssetTypeHint::Splat`,
  `OperatorMetadataInput/Output::Splat`, schema 1: `world`,
  `provenance` (plus the content hash of the view set it was trained
  from and the trainer's arguments), `kind` (`Gaussian3d` or
  `Surfel2d`), `sh_degree`, `count`, and packed f32 columns: `means`
  (3N), `scales` (3N, logarithmic; surfels carry a third scale near
  zero), `quats` (4N, w x y z), `opacities` (N, logit), `sh0` (3N),
  `sh_rest` (3N·((d+1)²−1), channel-major as the 3DGS PLY stores it),
  `normals` optional (3N). Six hundred thousand Gaussians at degree 2 are
  about 90 MB, an ordinary asset since the blob format landed.
- `splat_import_operator` (blob + config → Splat) reads the 3DGS PLY
  layout through `ply_core` (`x y z`, `f_dc_*`, `f_rest_*`, `opacity`,
  `scale_*`, `rot_*`, `nx ny nz` when present); `kind` from the config
  with a default from the third scale. The GUI's Import menu and the
  CLI's project add both take it.
- `splat-list -i <splat | project> [--asset id] [--json]`: count, kind,
  SH degree, bounds at the 1st and 99th percentile of the means, opacity
  and scale quantiles, provenance.
- `splat_points_operator` (Splat + config → TriMesh point cloud): centres
  with normals (a surfel's third axis; a Gaussian's smallest-scale axis)
  and the SH0 colour, filtered by opacity, scale and a bounds or
  near-marker radius, so `cloud_fit`, `cloud_normals` and the section
  tools apply to the splat directly.

**S1 status (landed 2026-09-11).** `volumetric_abi::splat` as designed,
with the columns as little-endian `f32` byte strings through a serde
adapter (a CBOR float array would be five bytes a value and slow to
decode); `normals` is an empty column rather than an option; the world
frame and `Provenance` are the view set's types. Deviations: the
importer takes the training view set as an optional second input and
copies its world frame and provenance and hashes its bytes (blake3, the
engine's content fingerprint) into `views_hash`, so the link to the
evidence is made by the DAG rather than typed; the config strings
override where set. The points operator emits a Point1 `FeaMesh` (the
engine's point-cloud value; the plan's "TriMesh point cloud" was a
slip), with `opacity` and `scale` fields beside `normal` and `color`,
and takes the view set as an optional input for `near.marker`. Its
`bounds` group is six scalars because the config parser has no
fixed-length arrays (`stl_import`'s `translate: [float, float, float]`
fails the same parse today; unchanged here). File extension `.vsplat`;
`project-add-asset` validates it; the GUI's Import menu has Splat
(`.ply`) and the step editor a splat slot. Findings: the
chairbase-dslr-0-2dgs run's third log-scale is a full third axis
(median ratio 1.01 to the others), so `auto` reads it as 3D Gaussians,
which is what the parameters are; its normals are all zero and are
dropped; its quaternions are unnormalised (median norm 1.19) and are
normalised at use. The survey now sets the set's `world.up` to the card
normal (it left the intake's default), which the splat inherits.
Validation: import, listing and points in 2 s; 345 k of 595 k primitives
at or above half opacity, 1.2 mm median from the trainer's TSDF surface
cloud and it from them (4.5 mm at p99); SH0 colours match the trainer's
`red green blue`. Demo project
`sessions/chairbase-dslr-0/demo/chairbase_dslr0_splat.vproj`.

### S2. Rendering

- `volumetric_renderer` gains a `splat` pipeline: per-Gaussian instance
  data (mean, scale, rotation, opacity, SH coefficients), the 2D
  covariance from the projected 3D covariance in the vertex shader (the
  EWA splatting Jacobian), the SH colour evaluated there from the view
  direction, a quad sized to three sigmas, a Gaussian falloff times the
  opacity in the fragment, premultiplied alpha blended back to front,
  depth tested against the scene's depth at the centre and never written,
  drawn after the composite pass with the lines and points (the point
  pipeline's overlay variant is the template). Sorting: view-space depth
  keys radix-sorted on the CPU into the index buffer, redone when the
  view direction turns by a few degrees or the eye moves, kept otherwise;
  a GPU sort when a scene demands it. The same code runs on the web
  target with the sort on the main thread under a budget.
- `volumetric_preview`: `PreviewPlan::Splat` builds a retained GPU splat
  with bounds from the means' percentiles and the count in the stats.
- GUI and CLI: a splat draws wherever a cloud would, in the viewport and
  in `render`. Look-through with the photograph at 50 % over the splat is
  the visual audit; `render --through set:view --asset splat` produces
  the trainer's evaluation render headlessly.
- Surfels are drawn as thin ellipsoids first; the exact ray–surfel
  intersection of 2DGS is a follow-up if the thin-ellipsoid result is
  visibly wrong at grazing angles.

**S2 status (landed 2026-09-11).** As designed, with these findings.
Sorting is on the CPU (radix on the view-space depth, the SH colour
evaluated per primitive at the same pass) into a rewritten instance
buffer, re-done when the view axis turns 2° or the eye moves 2 % of the
extent; the 3DGS instance layout carries the three scaled world axes
(80 bytes) rather than a covariance, so the vertex shader forms the 2D
covariance as the outer-product sum of the projected axes and the
fragment can intersect a surfel's plane. Surfels use the exact 2DGS
evaluation from the start: the thin-ellipsoid draft drew an edge-on
surfel as a line the length of its 3σ footprint (the projected
covariance integrates the disc's density along the ray; the intersection
lands far from the centre and contributes nothing), which was not
"visibly wrong at grazing angles" but wrong on every silhouette, so the
follow-up came forward. Two colour-space facts settled by measurement
against gsplat's held-out renders: trainers fit and blend sRGB values as
plain numbers, so the splats blend on their own half-float layer in that
space and the layer is linearised once over the scene (blending in
linear light was 1.5 dB and 40 % of mean brightness off); and the
chairbase-dslr-0-2dgs run must be read as surfels, since gsplat's 2DGS
trains two scales and exports an untrained third (drawn as Gaussians it
was 7 points of silhouette IoU and 3 dB worse). The CLI's overlay now
takes the render's coverage from the alpha channel (transparent-black
sentinel, colours un-premultiplied) instead of a magenta test, so a
splat's fading edge blends by its opacity. A stray-primitive cull (before
the near plane, outside 1.3 × the frame, or wider than two frames) is in
the vertex shader. Validation: `render --intrinsics --pose` with the
trainer's own cameras against its held-out panels (frames 133, 143, 154,
166; 3096 × 2064; the grid off): 34.9, 34.8, 33.0, 36.2 dB inside the
subject mask, 40.0, 41.3, 38.9, 46.1 dB over the frame, silhouette IoU
0.93, 0.93, 0.89, 0.89; affine-fitted to the photographs 27.8, 27.0,
30.4, 33.8 dB against the trainer's 28.6, 27.4, 31.7, 34.8 (its
`metrics.json` mean 28.6 over seven). The surfel intersection is the
last 2 dB of that. Follow-up the same day on chairbase-dslr-1-refuse
(4.17 M surfels, no subject mask): a 550 ms single-threaded re-sort on
every two degrees of orbit gave 1–2 fps, and the CLI's offscreen device
was dropping 817 k primitives at wgpu's default 256 MiB buffer limit.
Now the first sort runs in place in parallel (255 ms), later sorts on a
background thread (~200 ms, frames stay at 12–36 ms), the GUI keeps
painting while one is in flight, `render` settles the order before a
readback, and the offscreen device takes the adapter's buffer limit as
the GUI does. Instances are 80 bytes (334 MB for that splat); halving
that with half-float axes is the next step if uploads dominate on a
weaker link. Not done: the viewport's reference grid lies in the
renderer's XZ plane while a surveyed world is z-up, so the grid cuts
through a splat and a view set at an angle (it also drew the "streaks"
in the first comparisons); a settings toggle for kernel radius and
opacity scale; a GPU sort for scenes past a few million primitives.

### S3. Photometric audit

`view-photometric -p project --views set --splat splat [--view id]...
[--holdout] [--scale s] [--json] [-o dir]`: the splat rendered through
each view at the training scale against its photograph, PSNR and mean
absolute error inside the mask (or the whole frame), the affine-fitted
PSNR the trainer reports, and a residual image per view. On
chairbase-dslr-0 the trainer's `metrics.json` says 28.2 dB on the seven
held-out views and 32.0 on the training views; the command reproduces
those within the JPEG round trip. An operator variant emits the numbers
as an F64Map for step D.

## Step P — Python bindings

`crates/volumetric_py`: a PyO3 module `volumetric` built with maturin
(abi3, Python ≥ 3.12; pyo3 0.29 + numpy 0.29, Python 3.14 in the shim's
venv). The bindings wrap crate functions one to one; a feature that
exists only in Python is a bug, and so is logic copied out of the CLI:
where the CLI held the only copy (operator input coercion, asset value
decoding, add-op naming) it moves into the `volumetric` crate as
`volumetric::project_edit`, and the CLI calls that. The shim's tests
against OpenCV become the bindings' tests.

Layout: `Cargo.toml` (`[lib] name = "_volumetric"`, cdylib, `test =
false` — an `extension-module` cdylib cannot link a test binary, tests
are pytest), `pyproject.toml` (maturin, `python-source = "python"`,
module `volumetric._volumetric`), `python/volumetric/__init__.py`
(re-export only), `src/{lib,project,cv,viewset,splat}.rs`, `tests/*.py`.
Build: `maturin develop --release -m crates/volumetric_py/Cargo.toml`
inside the venv (maturin is pip-installed there).

Surface, first increment (P1, landed 2026-09-12; 16 pytest cases, the
chair demo set surveyed from Python matches the CLI's poses within 4 mm):

- `Project()`, `Project.open(path)`, `.save(path)`, `.to_bytes()`,
  `Project.from_bytes(b)`; `.asset_ids()`, `.exports`, `.steps()`.
- `.add_model(wasm, id=None) -> id`; `.add_asset(data, kind, id=None) ->
  id` with kind one of `lua wgsl config f64map blob viewset splat` (the
  CLI's names; ViewSet/Splat/F64Map bytes are validated, a JSON dict for
  `f64map` is encoded).
- `.add_op(operator, inputs, output=None, export=True) -> [output ids]`:
  `operator` a bundled name or a path; each input is an asset id (`str`),
  `None` (unwired), `bytes` (raw), or a JSON-like value (`dict`, `list`,
  number, string) coerced by the slot's declared type exactly as the CLI's
  `json:` form is.
- `.run(remote=None) -> dict[id, Asset]` (the exports); `Asset.id`,
  `.kind` (the type hint's display name), `.bytes`, `.warnings`, `.value`
  (F64Map → dict, VecF64 → float64 array, Subspace → dict with `origin`
  and `basis` arrays, else None), `.mesh()` (FeaMesh/TriMesh → `Mesh`
  with `nodes` (n,3), `elements` (m,k) uint32, `node_fields` and
  `element_fields` dicts of (n,c) arrays), `.viewset()`, `.splat()`.
- `operators() -> [name]`, `operator_info(name) -> dict` (name, version,
  description, inputs as labels, outputs, docs, variadic).
- `Gray(array)` from a uint8 (h,w) array, `Gray.decode(bytes)` from a
  JPEG/PNG; `.array` (h,w) uint8 view copy; `.width/.height`.
- `detect(gray, families=["5x5_100"], **DetectParams) -> [dict]`: `id,
  family, corners, rotation, distance, fit_px` (dicts: detections are
  small, and the shim's code already reads them that way).
- `observe(gray, swatches="5x5_100", board="survey_card") -> Observed`:
  `.markers` (Detections), `.corners` (board corners: `id, pixel (2,),
  fit_px`), `.observations` (the view-set Observations as a dict).
- `ViewSet.load(path)`, `.decode(bytes)`, `.save(path)`, `.encode()`;
  `.views` (`View`: `id, camera, camera_to_world (3,4) or None, tags,
  time, source, shot dict, observations dict, image() -> (h,w,3) uint8
  decoded from the embedded picture, picture bytes`), `.cameras`
  (`Camera`: `label width height fx fy cx cy distortion dict`),
  `.markers` (`id`, `size_m`, `corners (4,3)`), `.board` (spec dict +
  corners (id, position, sigma)), `.poses()` (n,3,4) with NaN rows for
  unposed views, `.world_up`, `.provenance` dict.
- `solve_still(gray, file_bytes, viewset, **StillOptions) -> StillSolve`
  (`seed` Camera, `seed_source`, `detections`, `pose` (camera_to_world,
  camera, rms_px, markers used, focal/k1 estimates) or `error`,
  `warnings`); `survey(viewset, **SurveyOptions) -> (ViewSet, report
  dict)` where the report is `SurveyReport` as JSON.
- `Splat.load(path)`, `.decode(bytes)`: `kind, sh_degree, count, means
  (n,3), scales (n,3), quats (n,4), opacities (n,), sh0 (n,3), sh_rest
  (n,k,3), normals (n,3) or None, world_up, provenance`.
- `render_board(...)` from cv_core's synthetic renderer, so the shim's
  synthetic-scene tests can be ported without OpenCV.

P2 (landed 2026-09-12): `View.project/cast/ray`, `ViewSet.triangulate`
(`view_core::measure`, shared with `view-pick`/`view-triangulate`),
`Asset.sample/occupied/bounds/dimensions`, `import_stills(dir, **StillsOptions)`,
`ViewSet.detect(...)` (`view_core::detect`, shared with `view-detect`),
`card_spec`. The chair_photo measurement report replays exactly.

Deferred to P3: `render(...)` through the native offscreen path (needs
the CLI's render command factored into a library entry first),
`Project.add_views` (view-select), `View.crop`, `Project.set_config`,
`ViewSet` mutation (poses from Python), observations as a first-class
value (named feature picks and contours on the view set — an ABI decision).

Tests (`crates/volumetric_py/tests`, pytest in the shim's venv): a
cylinder built and run, its mesh bounds checked; a synthetic board
rendered, detected and observed; a real chairbase still detected and
solved against `chair_views.vviews`; survey on the ten-view demo set
reproducing the CLI's numbers.

## Step D — evidence audits

| Check | Signal | Catches |
|---|---|---|
| Intake | Camera key per view, autofocus frames, stabilisation, ISO, orientation | Frames that cannot share a camera model |
| Detection | Corners per frame, `blur_px` from marker edges, card corners seen once | Defocused frames (pcb-dslr-1 at f/4), motion blur |
| Survey | Per-frame rms trend, per-point sigma, card planarity against 0.03 mm, span against the caliper, swatch sides against the sheet spec, focal standard error | A wrong pitch, a moved swatch, a mirrored pose, a mixed focus setting |
| Setup | The still's cards against the set's map (step B's rms) | A map from another setup (the chairbase1/chairbase2 mix-up) |
| Coverage | Per subject region: view count, elevation and azimuth spread, before training | A one-elevation capture (the chair-leg phantom ridge) |
| Photometric | S3 per view: PSNR outliers against the set's median | Views the splat cannot explain: motion, exposure, a moved subject |
| Physical | The card's normal is up, swatch sizes, plausible scale | Unit and convention errors |

Audit operators emit an F64Map of named metrics plus warnings; the same
checks run natively on a directory of stills for the capture-time loop.

## Later, not scheduled

- A GPU-service step: a project step whose execution is a remote job on
  the daemon's progress and cancel channels, so training becomes a step
  the agent runs from here. Until then the agent runs the shim's cluster
  script on the exported dataset.
- Fusion returns through the splat: expected depth rendered from the
  splat through every view, integrated into a TSDF here, with the
  cosine-of-incidence weighting the shim's results ask for.
- SIFT seeding (COLMAP triangulation against fixed poses) stays outside;
  the box seed is the default since seed density did not move the shim's
  results.
- Exact 2DGS ray–surfel rendering; compressed splat formats.
- `frame_align` operator: pose a model from one rank-3 Subspace frame to
  another, so a scan can be brought into a datum frame built from fits.
- `measure` command: volume, area, centroid, clearance and interference
  between two models.
- Pick and project through a view; magnified crop with a pixel grid.
- Colour projection from views onto models; visual hull.

## References

- Shim (oracle) code: `index_scanner/survey.py` (bundle, gauge, report),
  `board.py` (card spec, ChArUco detection), `sonyexif.py` (focus key),
  `phone.py` (`edge_blur`, `export_dataset`), `splat.py`
  (`write_splat_ply`, cameras.json schema 2), `splat_train.py` (what a
  run writes: `splat.ply`, `cloud.ply`, `tsdf.npz`, `metrics.json`,
  `renders/`), `calib/card.json` (the card with its caliper pitches).
- Real data, all under `/ceph/christian/index_scanner`:
  `sessions/chairbase-dslr-0` and `sessions/pcb-dslr-1` (stills),
  `work/<name>/{detections,survey,markers}.json`, `runs/chairbase-dslr-0-2dgs`.
  Earlier: `sessions/chairbase1` and `chairbase2`, `work/chair-phone`.
- Previous arcs (PLY, NRRD, cloud fit): commits 2bfa345, 3f2ebf1, e226286.
