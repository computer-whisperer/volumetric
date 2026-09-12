# Chair mounting interface from photographs

Clean-slate dogfood of volumetric, independent of `examples/chair`.

Priority: improve agent modelling tools, make the photo-to-CAD process
repeatable, then recover useful chair geometry. This first milestone is
the upper mounting interface, checked in multiple photographs. Hidden
geometry and motion are assumptions until observed.

## Evidence and scope

- Input: the 44 developed JPEGs in
  `/ceph/christian/index_scanner/sessions/chairbase-dslr-1/`.
- Scale reference: the measured survey card specification in `card.json`,
  copied from index_scanner's calibration metadata (not chair measurements).
- Run intake, detection, and survey with volumetric itself. Do not import
  scanner camera solutions, splats, clouds, or the previous chair model.
- Fit consequential features from photos, preserve pixel observations,
  and check projections in additional views.
- Generated data goes in `work/` and is excluded from Git.

## Replay

From the repository root, with Rust's `wasm32-unknown-unknown` target and
Python 3.11 or newer with NumPy and SciPy installed:

```sh
cargo build --release --target wasm32-unknown-unknown \
    -p wgsl_script_operator -p pose_operator -p subspace_operator
cargo build --release -p volumetric_cli
bash examples/chair_photo/survey.sh
python3 examples/chair_photo/measure.py
python3 examples/chair_photo/build.py --render
```

`survey.sh [photo-directory [work-directory]]` accepts a relocated copy
of the same photographs and verifies their hashes against `photo-manifest.json`.
Both Python scripts accept `--work`. For a new
subject, collect new observations and labels: the saved chair pixel picks
are specific to this session. The process carries over; they do not.

The Python code orchestrates the CLI and computes dimensions and a local
frame. Camera calibration, distortion, ray casting, and triangulation
remain volumetric kernels. No index_scanner executable or reconstructed
geometry is used. The survey card's physical calibration is the sole
external measurement.

Outputs in `work/`:

- `survey.vviews`, `survey.json`, `survey.log`: the full surveyed evidence.
- `measurements.json`: recovered points, ray misses, additional-view
  reprojection errors, sensitivity probes, dimensions, and local datums.
- `parameters.json`: measured and assumed model inputs, in metres.
- `mount.vproj`: two WGSL-generated parts, their measured frame, and
  three photographs. Open in volumetric's GUI for an audit.
- `check-DSC007*.png`: picked pixels (cyan) and projected 3D centers
  (magenta); labels use original full-resolution image coordinates.
- `overlay-DSC007*.png`, `mount_iso.png`, `mount_top.png`: render outputs
  when `--render` is supplied. Overlay photographs are rectified to the
  pinhole camera used by the renderer; their coordinates differ from the
  original-pixel `check-*` crops.

## What the first measurement establishes

The native survey poses all 44 images with 0.371 px reprojection RMS.
The fitted card planarity residual is 0.030 mm. Those describe the target
fit, not accuracy of the chair geometry; scale is constrained by the
card measurement and is not independently verified by that residual.

Two views (DSC00755 and DSC00758) locate six aperture centers. DSC00756
provides an additional-view consistency check: its feature picks are
excluded from triangulation, but its targets participate in the shared
camera survey. Predicted feature positions were inspected during picking;
this is not blind validation. The current check errors are 2.9–6.7 px,
and the largest fit-ray miss is 0.44 mm.

Approximate center distances:

| Quantity | Millimetres |
|---|---:|
| Crossbar outer slots | 213.6 |
| Crossbar inner holes | 109.8 |
| Rail slots | 29.9 |
| Rail-slot midpoint to crossbar inner midpoint, along local Y | 174.6 |

The local origin is the midpoint of the two inner crossbar holes. X runs
toward `cross_a_inner`; Y runs toward the rail-slot midpoint, perpendicular
to X; Z is X × Y. The outer pads are about 3.5 and 3.7 mm above this datum.
`observations.json` preserves the approximate rim-center picks;
`outlines.json` preserves silhouette and aperture boundary picks. The
crossbar ends are projected onto their respective raised planes.

The +/-3 px perturbation report measures sensitivity to one coordinate
at a time. It is not a confidence interval or accuracy bound: it excludes
camera uncertainty and systematic errors in identifying aperture rims.
`measurement-report.json` is the committed snapshot of this run; the full
photo → survey → measurement → project replay produced identical points
and parameters in a separate work directory.

## Model scope and assumptions

The model contains the crossbar and main upper rail, with six apertures.
It is an initial interface reference, not the complete chair-base assembly
or a fabrication drawing. The gas lift, star base, casters, controls,
rear cap, and mechanism housing are not yet modelled.

The hole centers and end-pad heights come from triangulation. Aperture
sizes and silhouettes come from approximate photo picks; matching slots
share their averaged dimensions, and the two inner holes share a diameter.
The crossbar retains asymmetric X bounds; its front/back edges are averaged
to straight edges. The rail is a centered trapezoid, flattened onto the
datum (its two slot centers differ by about 0.4 mm in normal coordinate).
The following are assumed, not measured: 2.5 mm sheet thickness, the
crossbar's linear bend transitions at |X| = 83–94 mm, and the rail's
front termination at Y = 14 mm. Small stampings, lips, fillets, fasteners,
and exact contact surfaces remain unresolved.

## Tool friction and direction

- **Fixed:** `view-crop` had an unlabelled image grid. It now prints original
  pixel coordinates on the image itself and retains readable labels when
  magnified. Its console crop description also had misordered arguments.
- **Fixed:** rendered photo overlays ignored lens distortion, although
  measurement commands honoured it. A shared `view_core` rectification
  function now supplies both CLI overlays and GUI look-through, matching
  the renderer's pinhole projection. Pixels outside the source are black.
- **Documented:** WGSL is the preferred authoring path. Lua is deprecated
  and frozen for existing projects, reaffirmed by the user on 2026-09-12.
  This example uses WGSL and routed `F64Map` parameters.
- **Deferred legacy issue:** the first, abandoned Lua draft exposed that
  one bounds getter cannot call another (`unknown function`). A compiler
  change was discarded after the WGSL direction was clarified; the legacy
  operator is unchanged by this task.
- **Remaining intake gap:** the developed JPEGs have no EXIF. Intake warns
  and seeds one unknown camera from its default field of view. Survey
  successfully recovers this session's camera, but a future multi-focus
  capture needs preserved metadata or explicit grouping. No metadata was
  guessed or copied from a differently cropped RAW image.
- **Remaining workflow gap:** collecting feature correspondences still
  involves manual visual picking by the agent. The JSON observations and
  replay scripts make that work auditable, but automatic aperture/edge
  fitting and a native persistent feature-observation value would make
  the next project easier. No human chair measurements were requested.
- **Environment limitation:** rendering in this sandbox selected llvmpipe
  and failed wgpu validation because its R32Float attachment was not
  renderable. Rendering with host GPU access selected the Radeon RX 6900 XT
  and succeeded. Software-renderer compatibility remains unresolved.
- **Separate existing test failure:** `cargo test -p volumetric --test
  operator_metadata` fails because `cloud_distance_operator` declares
  `band: float .gt 0.0`, while the host config parser supports inclusive
  bounds only. Neither that operator nor the config parser was changed
  here; support for strict bounds remains a follow-up tooling bug.

## Verification

- Replayed all three stages in a separate directory; recovered points and
  model parameters were identical. The six apertures are visible in the
  rendered WGSL parts and were inspected in three rectified photo overlays.
- Rectification tests cover radial and fisheye models, half-resolution
  previews, off-center intrinsics, tangential distortion, anisotropic
  resizing, bilinear interpolation, and pixel-center conventions. The GUI
  regression checks a known pixel after look-through rectification.
- Browser GUI check: `cargo check -p volumetric_ui_v2 --target
  wasm32-unknown-unknown --no-default-features --features web` passes.
- Full workspace tests were run; the metadata failure above prevents an
  all-green result. The sandbox run also encountered localhost binding
  failures, the software-GPU failure above, and a UI worker stack overflow.
  All nine daemon integration tests, the complete UI target (146 tests),
  and the renderer regression pass with host networking/GPU access.
  The stack overflow did not reproduce in that UI rerun; its cause is
  unresolved. Logs are retained under `work/`.
