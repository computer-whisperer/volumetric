# Scan Evidence — Design and Plan

Status: ratified 2026-09-09. Step 0 landed 2026-09-09. Step A: value, import and CLI landed 2026-09-09; GUI pending. Steps B–D pending.

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
| A | `view_core`: ViewSet value with provenance, manifest import, look-through render, depth residual | 4 days | CLI landed; GUI pending |
| B | `cv_core`: ArUco detection, PnP, focal; `view-solve` CLI, operator, GUI drop-a-still | 5 days | pending |
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

Status: value, import and CLI landed 2026-09-09; the GUI panel is pending.
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

Views panel listing a ViewSet's views with thumbnails, frustum gizmos in
the viewport, and look-through: the viewport camera takes the view's
pinhole and the photograph underlays the frame at a chosen opacity.
Sequenced last in this step; the value, import and CLI land first so the
acceptance task can start.

## Step B — solve a still from the cards

`cv_core`: ArUco 5x5 (the scanner's swatch dictionary, `DICT_5X5_100`) and
4x4 (its calibration board) detection, quad extraction, decoding with
Hamming tolerance, sub-pixel corner refinement; planar PnP per marker,
joint least squares over all corners; focal from the markers when the
intrinsics are unknown (well-posed with markers on two planes, reported as
weak when all markers share one plane seen near fronto-parallel); EXIF
focal as the seed. Tests on a synthetic board renderer with exact ground
truth, then the chair-phone stills against their COLMAP poses in
`work/chair-phone/colmap`. Surfaces: `view-solve` CLI, a `view_solve`
operator (ViewSet + image blob → ViewSet with the new view), and the GUI
drop-a-still flow showing detected markers and the reprojection residual.

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
