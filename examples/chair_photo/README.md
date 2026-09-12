# Chair base from photographs

Clean-slate dogfood of volumetric, independent of `examples/chair`.

Priority: improve agent modelling tools, make the photo-to-CAD process
repeatable, then recover useful chair geometry. The accepted upper mounting
interface is preserved in `mount.vproj`; the second milestone extends it to
a representative complete base in `base.vproj`, with a separately fitted
backrest receiver. Hidden geometry and motion remain assumptions.

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
  when `--render` is supplied. Since 2026-09-12 the renderer draws the
  model through the camera's lens, so overlay photographs are shown as
  shot and share the original-pixel coordinates of the `check-*` crops
  (before that, overlays were rectified to the renderer's pinhole and the
  two differed).

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
or a fabrication drawing. The extended `base.vproj` adds the gas lift, star base, casters, controls,
receiver, and mechanism housing as described below.

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


## Extended assembly and backrest receiver

Continue after the original measurement stage:

```sh
python3 examples/chair_photo/fit_receiver.py --audit
python3 examples/chair_photo/measure_assembly.py
python3 examples/chair_photo/build_assembly.py --render
python3 examples/chair_photo/audit_assembly.py
```

All four scripts accept `--work`. The build refreshes `mount.vproj` from
its existing measurements, then writes `base.vproj` with 16 separate WGSL
exports. The accepted six aperture positions and datum are unchanged.
The project embeds six audit photographs. Outputs include `base_iso.png`,
four `base-overlay-*.png` photos, and `backrest-detail_{iso,top}.png`.
Native sampling checks that all exported solids leave the six mounting
holes and four receiver-axis probe positions clear; positive probes also
check that the receiver walls and column actually exist.

The receiver is a hollow socket, with a lead-in around its entry mouth,
a side clamp knob, and separately exported visible pin/fastener geometry.
`backrest_mouth_frame` records its fitted entry plane.
`rear_pin_reference_frame` records the observed transverse pin station;
it does not assert that this is the complete tilt mechanism.

`receiver-observations.json` traces the outer entry rim in three fitted
views (55, 58, 60), checked in 56. `fit_receiver.py` fits a common plane
and rounded rectangular profile without pretending that contour samples
are matching physical points across images. Every ray/plane intersection
uses volumetric; SciPy fits the Euclidean plane/profile parameters.

Approximate receiver results:

| Quantity | Result |
|---|---:|
| Entry mouth width × depth | 39 × 18 mm |
| Mouth center in accepted mounting frame | X −0.1, Y 274.9, Z 7.0 mm |
| Fitted mouth-plane rotation about local X | 6.9° |
| Visible rear pin-cap center | X −34.3, Y 260.0, Z −35.0 mm |
| Shared rim-fit RMS, by view | 0.49–0.68 mm |

The radius reaches the capsule limit (half the short dimension); this is
an active profile constraint, not an independent radius measurement.
Fitting each view separately on the common plane gives widths about
37.0–39.6 mm and depths 17.5–19.4 mm. That disagreement is more informative
than the shared residual alone: these are approximate mouth dimensions,
not a proven mating clearance. Residuals are equally weighted millimetres
on the plane, not equally weighted pixel errors under foreshortening.
Predictions were inspected while refining the manual traces; the fourth
view is a consistency check, not blind validation.

The narrower throat below the rounded lead-in is not resolved. The model
assumes a 1 mm inset per side, a 3 mm lead-in depth, a 65 mm open socket
length, and a throat perpendicular to the mouth. The mouth normal itself
does **not** establish the insertion direction. These parameters are
explicitly named `assumed_*`; the 65 mm opening in the model is not a
claim of usable insertion depth. The receiver's actual internal stop,
clamp screw intrusion, tilt range, and linkage remain unresolved.
Opposite pin caps and shafts are symmetric approximations; the visible
cap positions retain all three measured coordinates.

The remaining base is intentionally representative. Measurements support
about 336 mm star radius, a 48 mm lower column, a 27 mm upper rod, and a
266 mm column collar height above the survey datum. Four caster joints
constrain a common circle (2.5 mm radial RMS); fivefold symmetry supplies
the occluded fifth arm. Floor height, curved arm sections, hub shape,
wheel dimensions and caster fork details are approximate. Four caster
yaws come from apparent wheel centers at an assumed 25 mm axle height;
the hidden fifth yaw and all caster trails are assumed. Control-paddle
centers were triangulated, while stems and paddle profiles are simplified.
No dynamic articulation or internal spring/locking mechanism is asserted.

`receiver-report.json` and `assembly-report.json` preserve measurement
snapshots. The replay in a separate work directory reproduces the fitted
receiver, assembly measurements, and routed model parameters.

## Additional tooling findings and verification

- **Fixed:** photo measurement arguments were parsed as f32 before being
  passed to f64 kernels. They now retain f64 precision and reject nonfinite
  coordinates. The regression covers a millimetre offset at a large world
  coordinate and fractional image coordinates; all 29 CLI tests pass.
- **Documented WGSL trap:** a typed f64 `let` does not prevent `select`
  from concretizing literal arguments as f32. Explicit `float(...)`
  arguments fix it; the authoring guide now gives the working form.
- **Fixed diagnostic:** with multiple embedded view sets, `render --through`
  needs `views:DSC00755` or `assembly_photos:DSC00762`. The shared CLI error
  previously suggested `--views`, which selects preset render views in this
  command. It now explains both selection forms; the assembly script uses
  qualified IDs.
- **Remaining authoring gap:** persistent multi-view feature and contour
  observations still live in example JSON, and the fitter orchestrates many
  CLI calls. A native observation/fit value would improve reuse.
- **Rendering:** final audit renders disable mesh sharpening and
  simplification for denser curved surfaces. An initial suspicion of missing
  longitudinal column surfaces was withdrawn after isolated comparisons:
  the silhouette is continuous in all four modes. Cap/collar faceting differs;
  these images do not establish a missing-surface bug.
- **Verification:** critical void checks and positive occupancy controls pass.
  Full workspace tests were run with host networking and GPU access; only
  the existing `operator_metadata` strict-bound parser failure remains.
  The earlier UI stack overflow did not recur in this run. Logs and all
  diagnostic images remain under the ignored `work/` directory.

## Python workbench and persisted observations

[The P1–P5 dogfood report](DOGFOOD.md) documents the in-process receiver fit,
observation-bearing project, labelled lens-true overlays, reproduction command,
and remaining workflow gaps. `dogfood.py` compares against this example's
accepted geometry and writes separate artifacts under `work/dogfood/`.
