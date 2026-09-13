# Python and persisted-evidence dogfood — 2026-09-12

This pass exercises Fable's P1–P5 additions against the accepted chair base.
This historical report describes the 16-export baseline. The script now accepts
the 22-part articulated base described in ARTICULATION.md and extracts its
parts through Assembly Model. It reuses the photos-only camera survey and existing observations; it adds no
new chair measurements and does not change the accepted geometry.

## Repeat

First complete the survey, measurements and `build_assembly.py` from README.md.
Install the Python package using `crates/volumetric_py/README.md`, plus NumPy,
SciPy and Pillow. Then:

```sh
python3 examples/chair_photo/dogfood.py --benchmark-cli --render
```

`--work` selects the existing baseline directory. Results go in its `dogfood/`:

- `evidence.vviews`: four full-photo views, six named hole picks with fit/check
  roles, and the traced receiver mouth in four views.
- `base-evidence.vproj`: the current WGSL assembly with that view set and a
  provenance blob containing the original observation descriptions and roles.
- `receiver-python.json`, optionally `receiver-cli.json`: the same SciPy
  optimization through either native Python ray casting or the CLI.
- `report.json`: numerical comparisons, execution timings and occupancy audit.
  `render-report.json` retains the last render run, including camera/GPU notes.
- `overlay-DSC00755.png`, `overlay-DSC00756.png`, `overlay-DSC00760.png`:
  lens-true edge overlays with saved marks and labels, when rendering.

The fitter itself accepts `--backend python`, `--views PATH` (reads the saved
`receiver_mouth` contour), and `--output PATH`. The CLI remains available as
an independent transport baseline. Both use the same optimization math.

For this run, maturin was not installed. We built the extension with
`cargo build --release -p volumetric_py`, copied the package's `__init__.py`
into `work/python/volumetric`, and linked `_volumetric.abi3.so` there to
`target/release/lib_volumetric.so`. Commands used
`PYTHONPATH=examples/chair_photo/work/python`. This exercised the current
extension, not the documented maturin installation procedure.

## Observed results

The first comparison took 0.528 s through Python and 10.428 s through the CLI,
including subprocess startup for each whole fit: about 20x faster. This is a
single paired measurement, not a controlled performance study. A later run
concurrent with workspace compilation had very different timing; logs retain
both. Dimensions and center agree to about 2e-12 mm; the maximum world-basis
component difference is 3e-13. These numbers measure implementation agreement,
not physical measurement accuracy. Receiver uncertainty and assumptions in
ASSEMBLY_NOTES.md still apply.

All six fitted hole centers agree with the saved baseline within 7e-12 mm.
View-set save/load preserves the picks and contours; project save/load preserves
its serialized bytes. All 16 exported solids leave all ten critical void probes
clear, and receiver-wall and column positive controls pass through Python.

Three overlays were inspected. Labels make the evidence immediately legible,
and the original-image coordinates remove the former rectified-image mental
conversion. The receiver contour is visibly useful in all three views. The
check-view rail labels overlap in DSC00756. These images also still expose
approximate shell, control and caster geometry; this pass does not refit them.

## Friction and next priorities

1. **Fixed:** JSON pixel lists were rejected by tuple-only Python extraction.
   Picks, contours, triangulation, crop centers and crop marks now accept
   two-element sequences; existing tuple callers remain supported. Regression
   coverage includes real JSON observations and malformed coordinate lengths.
2. **WGSL parameter editing:** `Project.set_config` reports
   `wgsl_script_operator declares no configuration input`. This is consistent
   with its config-only contract, but WGSL's F64Map parameter input needs an
   equally direct editing API for dimension-fitting loops. A dimensions update
   should not require reconstructing the project or editing serialized bytes.
3. **Contour roles:** unlike feature picks, contours carry no fit/check role.
   The receiver fitter still reads membership from receiver-observations.json;
   the project also stores that metadata as a blob. The receiver contour appears
   green even in the check view. Check here means additional-view consistency,
   not blind validation: all cameras share the survey and predictions informed
   the original picks.
4. **Label layout:** collision avoidance or selective labels would help the
   crowded rail-hole view. Automatic target marks can also dominate small crops.
5. **Reusable render context:** rendering a Project invokes project execution
   each time (with build caching). Passing existing exported Assets avoids that,
   but Python exposes no direct accessor for imported evidence Assets to combine
   with those exports. A run result retaining imports would simplify this loop.
6. **GPU failure reporting:** sandbox rendering raises PyO3 PanicException for
   an unsupported R32Float render target. Host Radeon rendering works. This is
   the existing software-adapter limitation, now surfaced through Python too.

Also fixed the survey integration test missed by the Observations field migration,
using defaults for fields it does not populate.

35 Python tests pass with host GPU access. The sandbox workspace run reached
the daemon integration tests, where binding localhost ports was denied.
`cargo test --workspace` passed with host access, including doctests. The host
run is recorded in work/workspace-tests-host.log; the sandbox log is
work/workspace-tests.log. The change introduces no unrelated cleanup.

## Downstream chair design dogfood — concept A

The next pass used 18 new DSLR photographs and the same physical card, now
identified by its allocated AprilTag IDs through `examples/calibration-targets.json`.
The inventory points at the existing anisotropic calibration instead of copying
another set of scale values. It is an example workflow, not an engine registry.
The replay verifies source/coding-stream hashes and requires a new survey if the
stored card identity or numeric calibration hash changes. Re-extracting with
LibRaw changed a few EXIF bytes while all 18 compressed image streams remained
identical; `intake_backrest.py` verifies those streams without recompression.

Native intake, detection, survey, triangulation, projection, persisted picks,
WGSL compilation, boolean subtraction, assembly import, driven joints, posed
part extraction and rendering all ran successfully. No new engine patch was
needed to build the concept. It has 53 parts including cushion and backrest-shell
proxies, and 16 independent states. All 22 base part models are byte-identical
to the existing assembly. Changing chair adjustment state also preserves all
53 rest part models; attached interface sidecars follow their owners.

Geometry checks caught actual modelling errors before review: the first arm
brace crossed the pan flange and underside; the first mast/yoke crossed the
retained clamp and plate. The revised layout has deliberate flange relief and
a mast behind the retained hardware. The committed audit samples those mating
pairs and independently checks translations/coupled inset. It does not prove
absence of every collision or establish structural fitness.

Remaining tool friction seen in this pass:

- **Assembly composition:** importing a base into a downstream chair required
  copying its part bytes and flattening its joint tree. A namespaced subassembly
  instance with attached interface frames would make this a much smaller and
  less fragile operation. Current code preserves names and uses decoded inline
  axes; it does not duplicate the base's geometry implementation.
- **Collision feedback:** useful occupied-point evidence required scripting
  pairwise samples. An engine-level clearance query with witness points and
  explicit contact exceptions would shorten this design loop. The current
  audit is a deterministic smoke test, not an exact collision solver.
- **Thin features:** resolution 96 renders badly broke up the 4 mm shell and thin
  tubes. At 256 the shell and tubes become legible, but sheet flange artifacts
  remain. This is observed render behaviour; its precise meshing cause has not
  been diagnosed or fixed. Geometry decisions used native occupancy checks.
- **Evidence confidence:** the card fits at 0.108 px RMS while the hardware bolt
  checks disagree by up to 75 px. A good survey fit must not be presented as a
  hardware accuracy certificate. Those checks remain visible in the report.
- **Identity reuse:** the inventory works here, but native intake has no general
  physical-target calibration catalog API. Recognition currently performs tag
  detection before the board detector runs again for survey. Exposing selection
  of a calibrated target from existing detections would remove that duplicate
  work. A reprinted target needs a fresh identity and physical calibration.

Cleanup in this pass: updated the stale chair-design brief that still said the
cushion sizes and fabrication processes were awaiting input. No unrelated
subsystem cleanup or mesher changes were made.
