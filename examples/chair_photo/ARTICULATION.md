# Articulated chair base — 2026-09-12

The canonical base is now a 22-part Assembly and its union Model, replacing
16 independently exported solids. Parts remain authored in the photographed
world frame. The gas lift is split into body/rod; each caster into fork/wheels.
All other parts retain the accepted geometry. The rod extends 80 mm down
inside the solid body so positive lift does not expose an artificial gap;
this hidden extension leaves the rest union unchanged.

## Joint tree and assumptions

Star fixed to world → lift body fixed to star → rod translates vertically →
housing swivels on rod → upper parts fixed to housing. Five forks swivel
vertically on the star; each wheel pair rolls about its axle on its fork.
There are twelve independent states: lift, swivel and five caster swivel/roll
pairs. Swivel and roll are continuous; angles are degrees, lift is metres.
Axes are in world coordinates at rest. The lift/swivel axis routes through
the measured column datum as a Subspace input to Mechanism.

The -20..+80 mm lift exploration range is provisional, not measured stroke.
Twin wheels share a roll state as a representative simplification. There is
no ground-contact constraint, rolling-without-slip condition or stability model.
Caster yaw and axle details retain the approximation level of the old model.

The backrest receiver remains fixed to the upper structure. Its mouth plane
and the observed rear-pin frame are reference datums, not proof of an insertion
axis or the tilt mechanism's kinematics. No synchro ratio or tilt range is
inferred from a single static pose. A future confirmed tilt joint must assign
the actual moving parts and their datums together.

## Repeat and inspect

Use the volumetric Python package plus NumPy/SciPy/Pillow. This workspace's
local extension setup is described in DOGFOOD.md. After the photo measurements:

```sh
python3 examples/chair_photo/build_assembly.py --render
python3 examples/chair_photo/audit_assembly.py
python3 examples/chair_photo/build_assembly.py --state examples/chair_photo/work/motion-state.json --render
python3 examples/chair_photo/audit_assembly.py --posed examples/chair_photo/work/base-posed.vasm
```

The rest build writes `work/base.vproj`, `.vasm`, `.vmech`,
`base-interfaces.json`, `base_iso.png`, three marked `base-overlay-*.png`
photos and `base-backrest-detail.png`. The pose option writes `base-posed.*`
and similarly prefixed renders, preserving the canonical rest build. In the
GUI, select the `chair_base` Assembly for articulated preview and its state
form/part dragging. `chair_base_model` is the union for downstream booleans and
occupancy. Headless renders select the Assembly explicitly to avoid drawing
the union over its parts. Interactive dragging itself was not manually tested
in this pass; the underlying pull solver was exercised through Python.

`base-mechanism.json` records the joint tree. `base-interfaces.json` pairs each
seat/mouth/pin datum with its owning part and includes rest and posed frames.
The project stores those attachments as a blob; existing Subspace outputs stay
as rest survey datums. An attached-datum value is not yet an engine feature.

For the one-time comparison against the saved pre-migration build:

```sh
python3 examples/chair_photo/audit_assembly.py \
  --reference examples/chair_photo/work/pre-assembly/base.vproj \
  --posed examples/chair_photo/work/base-posed.vasm
```

## Verification

- 12,000 deterministic samples per old component, including positive occupied
  samples: 192,000 total, zero occupancy mismatches against the new grouped
  parts. This is a sampled regression, not a proof about every surface.
- All 22 parts and the rest union leave six mounting-hole probes and four
  receiver void probes clear. Receiver walls, column body and rod positive
  controls pass. The voids remain clear under the tested lifted/swivelled pose.
- Independent rotation/translation oracles pass for 60 mm lift, 37° chair
  swivel, 43° caster swivel and 725° wheel roll. Roll direction is checked by
  the transform oracle; the wheel is axisymmetric and its sampled material
  point does not independently demonstrate roll direction.
- Pull reaches the crossbar target within 1e-7 m (observed about 7e-12 m).
  All twelve velocity derivatives agree with central differences to 1e-8.
- Rest and posed assemblies contain identical part WASM. Poses and all three
  attached interface-frame exports agree; canonical state remains zero.
- Original photo dogfood still replays after extracting assembly parts. The
  Python assembly test passes. The workspace run exposed a missing prerequisite:
  `fea_threaded::schwarz_config_matches_auto_solution_end_to_end` requires the
  packed threaded solver WASM, which is absent here. The no-fail-fast run is
  recorded in `work/assembly-workspace-all-tests.log`: all other non-ignored tests
  and doctests passed. The first run is
  `work/assembly-workspace-tests.log`. No FEA toolchain changes are part of this migration.
- Inspected rest and moved assembly renders, the DSC00760 rest overlay and
  the receiver closeup. Known meshing faceting remains; geometry was not refit.

The reproducible numerical result is saved in `articulation-report.json`;
full generated artifacts and logs are ignored under work/.

## Tooling findings and cleanup

The joint tree fits this mechanism well; the existing world-frame part design
migrates directly, and Assembly Model supplies per-part native sampling.
Part splitting is necessary to make rigid motion meaningful: a monolithic gas
lift cannot telescope, nor can a fused caster fork/wheel roll independently.

Two workflow gaps remain: datums cannot yet belong to a moving part as a native
value (the example exports ownership/posed coordinates), and joint-axis guides
are long and clutter photo overlays. Selective or hidden guides would improve
inspection. The prototype does not change those engine APIs.

Cleanup: corrected the assembly plan’s state count (13 with independent tilt,
12 without). Shared the accepted mounting-parameter calculation between build.py
and the new native build; removed the old flat CLI build path and updated its
audit/dogfood callers. Corrected historical metadata-test status in the notes.
Unrelated toy_car edits and existing screen/print artifacts remain untouched.

See CHAIR_DESIGN.md for the downstream carrier/adapter/backrest discussion.
