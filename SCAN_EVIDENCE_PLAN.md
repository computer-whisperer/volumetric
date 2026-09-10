# Scan Evidence — Design and Plan

Status: ratified 2026-09-09. Step 0 landed 2026-09-09. Step A landed 2026-09-09 (value, import, CLI, GUI). Step B landed 2026-09-10 (cv_core, view-solve, view_solve_operator, GUI Import Still); C–D pending.

## Why

Volumetric's purpose is to let an agent that cannot hold a tape measure do
ordinary modelling work from physical evidence a human collects. The
scanning project (`~/workspace/playground/index_scanner`, a Valve Index
used as a stereo scanner on a lighthouse-tracked pose, with printed ArUco
target cards on and around the subject) is the evidence pipeline; its
Python scripts are the interim exploration of what is practical. The
persistent tooling for everything from "files on disk" to a model lives
here.

Two interfaces, one code path. The CLI is the agent's interface: it must
let the agent see, measure, build and verify headlessly. The GUI is the
human's audit of what the agent accepted and produced. Whatever the CLI
renders or solves, the GUI shows through the same crates, so the frame the
agent looked at is the frame the human sees.

Acceptance task: "build the rest of the office chair" from the chairbase1
scan (`/ceph/christian/index_scanner/sessions/chairbase1`, plus the run
`runs/chairbase1-2dgs-mask` and the phone stills in
`work/chair-phone`). The scan covers the star base, casters, gas lift and
tilt mechanism with its seat-mount plate. The task is to add seat, back
and arms that mount on that plate, verified against the scan and the
photos, and to export a printable or manufacturable mesh.

## Decision record

1. **Posed image sets, not single views (reverses the arc-4 design of
   2026-09-09).** Rejected: a `View` value per image. Chose: a `ViewSet`
   value holding intrinsics, poses, image/depth/mask blobs for a chosen
   subset of a dataset, and the marker map. Because: real datasets are
   1542 views, 1.7 GB of images and 340 MB of depth; a project embeds a
   curated subset, and the marker map is what poses a new still.
2. **The marker pipeline comes into volumetric.** In: ArUco detection with
   sub-pixel refinement, PnP and focal solving, radial and fisheye
   (Kannala-Brandt) models, marker-map refinement against a pose track,
   TSDF fusion from posed depth, later stereo matching. Out: capture,
   device calibration, hand-eye, structure from motion (the cards replace
   it), splat training (CUDA). Because: the target cards are what turn a
   photo into evidence, and none of the "in" items needs hardware or a GPU.
3. **Provenance and audits are first-class.** Evidence values carry a
   provenance record (session, rig calibration hash with noise model,
   marker field id, subject setup id, tool versions). Combinators refuse
   to merge evidence from different fields or setups; crossing frames is
   an explicit alignment step with a residual. Statistical audits compare
   residuals to the rig's declared noise and surface as warnings on the
   existing warning channel; categorical mismatches are step errors.
4. **Formats at the boundary.** Heavy artefacts use standards: PLY, NRRD,
   16-bit PNG depth, PNG/JPEG images. The posed-image manifest is imported
   from the scanner's `cameras.json` (schema 2) and from nerfstudio
   `transforms.json` with two documented extensions (world units are
   metres, `depth_unit_m`). The marker map is produced by volumetric, not
   imported, once step C lands. Specs live in this repo, since it is the
   consumer.
5. **Values print.** Any small value (Subspace, F64Map) is readable from
   the CLI without a decoder. The agent reads numbers, not sizes.

## Steps

| Step | Content | Size | Status |
|---|---|---|---|
| 0 | Values in `project-run --json`; one `render` for a whole project scene with an explicit pinhole camera | 2 days | landed |
| A | `view_core`: ViewSet value with provenance, manifest import, look-through render, depth residual | 4 days | landed |
| B | `cv_core`: ArUco detection, PnP, focal; `view-solve` CLI, operator, GUI drop-a-still | 5 days | landed |
| C | Marker-map refinement with the pose track; TSDF fusion operator | 5 days | pending |
| D | Evidence audits: coverage, subject motion, frame quality, grouping | 3 days | pending |

Items that need no format decision (0, and the frame-datum operator and
`measure` command listed under "later") can land at any time. A, B and D
build on the ViewSet value.

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

## Step C — marker map and fusion

Marker-map refinement: per-frame poses and marker world corners solved
jointly against the lighthouse pose track with a loose prior, robust loss,
hand-rolled sparse Levenberg–Marquardt (tens of markers, thousands of
frames). Replaces the scanner's scan step and produces the per-frame
residuals the audits need. TSDF fusion from a ViewSet with depth as an
operator, removing the npz-to-NRRD detour.

## Step D — evidence audits

| Check | Signal | Catches |
|---|---|---|
| Card drift | Per-marker world position as a time series, step detection against corner noise | A swatch kicked or a sheet curling |
| Pose against markers | Lighthouse pose versus marker-solved pose per frame | Base station moved, universe change, occlusion |
| Calibration | Reprojection residual trend with radius, stereo epipolar residual | Wrong or stale intrinsics and extrinsics |
| Subject motion | Fuse first and second half separately, register, compare to surface noise | Subject nudged mid-capture |
| Time offset | Residual correlated with camera velocity | Frame and pose clock skew |
| Coverage | View count and angular spread per surface region of the subject bound | Unrepresentative capture, answered before fusion |
| Frame quality | Blur sigma, exposure, near-duplicate poses | Frames that add noise, not information |
| Physical sanity | Floor plane exists and is up, marker sizes match the sheet spec, plausible scale | Unit and convention errors |

Audit operators emit an F64Map of named metrics plus warnings; the same
checks run natively on a raw session directory for the capture-time loop.

## Later, not scheduled

- `frame_align` operator: pose a model from one rank-3 Subspace frame to
  another, so a scan can be brought into a datum frame built from fits.
- `measure` command: volume, area, centroid, clearance and interference
  between two models.
- Pick and project through a view; magnified crop with a pixel grid.
- Stereo matching in Rust, native only.
- Colour projection from views onto models; visual hull.

## References

- Scanner session layout and formats: `index_scanner/README.md`,
  `index_scanner/splat.py` (cameras.json schema 2), `index_scanner/phone.py`.
- Real data: `/ceph/christian/index_scanner/{sessions,runs,datasets}/chairbase1`.
- Previous arcs (PLY, NRRD, cloud fit): commits 2bfa345, 3f2ebf1, e226286.
