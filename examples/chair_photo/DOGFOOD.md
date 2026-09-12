# Python and persisted-evidence dogfood — 2026-09-12

This pass exercises Fable's P1–P5 additions against the accepted chair base.
It reuses the photos-only camera survey and existing observations; it adds no
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
- `base-evidence.vproj`: the 16-part WGSL assembly with that view set and a
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
The host rerun is recorded in work/workspace-tests-host.log; the sandbox log
is work/workspace-tests.log. The change introduces no unrelated cleanup.
