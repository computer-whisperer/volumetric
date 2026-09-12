# Chair — dogfooding plan

Status: opened 2026-09-11. The infrastructure arcs of `SCAN_EVIDENCE_PLAN.md`
(survey, splat value, splat rendering) are landed; this file is the ledger
of using them. Ground truth for the chair work; the evidence plan stays
ground truth for the tooling.

## Goal

Build a chair with specific properties on top of the office-chair base we
have been photographing. The first deliverable is a model of the base
itself: the five-arm star, the gas lift, and the tilt mechanism whose plate
the seat mounts on. It is the datum everything else attaches to, and
modelling it from the evidence is the forcing function for the tooling:
every place where an agent cannot see, measure, build or verify from the
CLI is a gap, recorded here and fixed in volumetric as it is met.

Evidence: session `chairbase-dslr-1` (44 a6700 stills, DSC00723–00766, no
EXIF: the JPEGs come from the raw developer), the shim's survey and
dataset (`datasets/chairbase-dslr-1`, 43 views at half resolution), the
trained splat `runs/chairbase-dslr-1-2dgs` (2DGS, 15 000 steps, subject
mask, 422 338 surfels, initialised from the unmasked run's TSDF cloud) with
its TSDF surface cloud (542 730 points at 1 mm) and `tsdf.npz` (2 GB). All
under `/ceph/christian/index_scanner`.

## Steps

| Step | What | Status |
|---|---|---|
| E | Evidence project: survey the stills here, import the splat as surfels, the TSDF cloud and the splat's points; check our frame against the trainer's | in progress |
| L | Look: renders of the splat and cloud from canonical directions and through the photographs, enough to name every part and its rough size | pending |
| M | Measure: ground plane, lift axis and radius, hub, arm count/length/rise, caster positions and wheel size, column heights, plate outline and hole pattern | pending |
| B | Build: `examples/chair/base.sh` → `chair_base.vproj`, parts from the catalog (sketch, extrude, revolve, cylinder, booleans, offset) placed on the measured datums | pending |
| V | Verify: model over the photographs through the surveyed views; point-to-model distances from the cloud; residuals recorded | pending |

Gaps found on the way go in the table below, with the step that met them.
Each is fixed in volumetric as a separate commit when it blocks; noted
when it does not.

## Frames

The survey re-bases the world on the card: origin at the first interior
corner, x along the columns, z from the card towards the cameras (up). The
trainer's dataset was exported from the shim's survey of the same stills,
so the splat lives in the shim's card frame; step E measures how far ours
is from it (expected: millimetres over the field, dominated by the
principal-point tilt).

## Measurements (step M, in progress)

World: the survey's card frame, z up, ground (card plane) at z = 0. All in
metres unless noted. Source: the trainer's TSDF surface cloud (542 730
points at 1 mm) unless noted; the cloud is what the cameras saw, so
undersides are missing and every "bottom" below is a silhouette.

- Lift axis (cloud_fit cylinder, seed line near the column, 2 mm
  tolerance): through (0.2569, 0.1509), direction (0.0023, 0.0019, 1.0),
  0.13° off z; outer tube radius 25.95 mm, 9346 inliers, 0.96 mm rms.
- Column profile (median radius of points within 60 mm of the axis, per
  10 mm of height): hub r 52–54 mm from z 0.13 to 0.165, chamfering to
  the top at z ≈ 0.18; lift outer tube r 25.5 mm from z 0.19 to 0.265;
  inner tube (with a bellows) r 14–17 mm from z 0.27 to 0.37; mechanism
  from z 0.41; its top faces peak at z 0.455–0.460.
- Arms: five, azimuths −153.8°, −82.9°, −10.4°, 63.3°, 135.8° about the
  axis (spacing 70.9–73.7°, mean 72.4°; two arms are contaminated by the
  carpet patch and the levers, take 72° nominal). Arm top from z 0.168 at
  the hub to 0.140 at r 0.27, then down to the caster socket at r ≈ 0.31;
  visible thickness 15–20 mm, width 50–70 mm across (widest at r ≈ 0.25).
- Casters: wheel bodies z 0 to 0.073 (diameter ≈ 65–70 mm, swivelled, so
  the wheel centre sits 0.30–0.34 from the axis); lowest points −8 mm
  (carpet compression).
- Mechanism (points above z 0.40): centre (0.297, 0.239), i.e. 65 mm from
  the lift axis; a rail ≈ 310 mm long at 65.9° azimuth with holes and
  slots along it, a cross bar with a bolt hole at each end, two levers on
  bellows stalks and a tension knob (see `cb1/sec_top_plate_splat.png`).

## Gaps

| # | Met in | Gap | Fix |
|---|---|---|---|
| 1 | L | `render` presets and grid assume y up; a surveyed world is z up, so `top` gave an elevation | `--up`, defaulted from the drawn view set or splat |
| 2 | L, M | No numeric probe of a cloud from the CLI: radial profile about an axis, azimuth histogram, wedge/slab percentiles, cluster centres; done with numpy on the PLY (oracle) | design after B, from the list of queries actually needed |
| 3 | B | Operators without READMEs: cylinder, revolve, extrude, subspace, sweep, slice, model_bound, mesh_to_model | write as each is used |
| 4 | B | No path sweep (a tube along a curve): the arms need one | two-view intersection (toy car) or an SDF script; decide in B |

## Log

- 2026-09-11: survey here of the 44 stills: all posed, 0.371 px rms, f
  6904.4 ± 1.1 px (the shim: 6904.4), card along span 179.32 mm against
  179.44 by caliper, planarity 0.030 mm.
