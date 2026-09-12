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
| E | Evidence project: survey the stills here, import the splat as surfels, the TSDF cloud and the splat's points; check our frame against the trainer's | done (`evidence.sh`; poses 0.33 mm median from the trainer's) |
| L | Look: renders of the splat and cloud from canonical directions and through the photographs, enough to name every part and its rough size | done (presets, through-views, ortho slab sections) |
| M | Measure: ground plane, lift axis and radius, hub, arm count/length/rise, caster positions and wheel size, column heights, plate outline and hole pattern | done except the bracket's hole pattern and the levers |
| B | Build: `examples/chair/base.sh` → `chair_base.vproj`, parts from the catalog (sketch, extrude, revolve, cylinder, booleans, offset) placed on the measured datums | first version: column, five arms with casters, rail and bracket boxes |
| V | Verify: model over the photographs through the surveyed views; point-to-model distances from the cloud; residuals recorded | first version: edge overlays through three views and sections over the cloud (`verify.sh`); numeric residuals pending |

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
- Mechanism: the top faces selected with a box (rectangular_prism z
  0.448–0.49 over the mechanism, `mesh_clip` keep inside, 43 363 points),
  then `cloud_fit`: line (20 mm tolerance) → the rail's direction (0.638,
  0.770), azimuth 50.4°, extent 303 mm, through (0.299, 0.222) at z 0.455,
  18 091 inliers; plane (4 mm) → the top face at z 0.456, level to 1.6°,
  33 068 inliers. (A first PCA over every point above z 0.40 said 65.9°;
  the seat bracket across the rail's end biased it. Fit the feature, not
  the cloud.) In the rail frame (u along, v across, origin on the lift
  axis): rail u −0.07..0.23, v −0.017..0.043; bracket at u ≈ −0.075, 230
  long, ≈ 30 wide, top at 0.456 (photograph DSC00742 for the bracket);
  two levers on bellows stalks and a tension knob not modelled.
- Mechanism, second pass (the mounting features, from the photographs).
  Frame: u along the rail (50.4° in the world), v across to the left, w
  up, origin on the lift axis at z 0.456. Hole centres read on
  magnified crops of DSC00742 and DSC00760 and triangulated (rays meet
  within 0.6 mm; the picture scale at the mechanism is 0.136 mm per
  pixel):

  | feature | u | v | w | size |
  |---|---|---|---|---|
  | bracket slot A (+v end tab) | −51.0 | +104.6 | +7.8 | 10 × 12.5 along v |
  | bracket hole 1 | −52.7 | +53.1 | (+6.9) | Ø8 |
  | bracket hole 2 | −54.0 | −56.8 | (+6.9) | Ø8 |
  | bracket slot B (−v end tab) | −54.5 | −108.9 | +5.5 | 10 × 12.5 along v |
  | rail slot, +v | +121.3 | +11.0 | −1.9 | 18 × 8 along u |
  | rail slot, −v | +121.0 | −18.4 | −2.0 | 18 × 8 along u |
  | dimple A (not a hole) | +45.8 | −2.8 | −3.3 at the bottom | Ø14 |
  | dimple B (not a hole) | +99.4 | −5.0 | −2.6 at the bottom | Ø14 |

  All in millimetres. Surfaces from plane fits on clipped patches of the
  refuse2 cloud (0.8–1.1 mm rms): the bracket's middle is level, top at
  w +6.9, 26 wide at u −51.5 and skewed −1.0°; its end tabs are formed,
  A rising 3.4° outward from a crease at v +85, B stepping down 2.5 mm at
  v −88 and rising 6.7°. The rail's flat top is 50 wide (v −23..+27),
  from u −33 to +187, level: the two slots triangulate to one height, and
  the 3.65° cross tilt the top-face fits showed came from the dimples and
  the folds (the rail-top patches disagreed with each other by 2–3 mm).
  Triangulated hole "centres" sit a millimetre or two below the surface
  (the far inner edge is what one sees), so heights of surfaces come from
  the patch fits and positions from the triangulations.

## Gaps

| # | Met in | Gap | Fix |
|---|---|---|---|
| 1 | L | `render` presets and grid assume y up; a surveyed world is z up, so `top` gave an elevation | `--up`, defaulted from the drawn view set or splat |
| 2 | L, M | No numeric probe of a cloud from the CLI: radial profile about an axis, azimuth histogram, wedge/slab percentiles, cluster centres; done with numpy on the PLY (oracle). Region selection does exist: a box model + `mesh_clip` keeps the points inside, and `cloud_fit` on the selection gives lines, planes, cylinders with extents | design after B, from the list of queries actually needed |
| 5 | B | `pattern` (and every wrapper reading dimensionality statically) refused a boolean's output: the glue passed `get_dimensions` through by a call | boolean emits the first model's constant (model_merge_core `const_i32_export`) |
| 6 | V | No way to take some views of a surveyed set into a project: `view-import`'s filters apply to manifests only, and a `.vviews` could not be subset or have its pictures re-embedded | `view-select` (ids in order, posed, tags, nearness, stride, cap; `--embed keep/full/preview/none` through each view's source) |
| 7 | V | `render --through` ignores the surveyed lens distortion (k1 −0.147, k2 0.255 on the a6700 at 50 mm), so an overlay is a few pixels off away from the centre at 1548 px wide | open: warp the render through the camera's distortion, or undistort the photograph once at import |
| 8 | V | No numeric cloud-to-model residual: an operator can sample only occupancy, not a model's channels, so the SDF operator's distance could not be read at the cloud's points | `cloud_distance` operator: bakes a signed distance lattice from occupancy (exact EDT in cloud_core), reads it at every node as the `distance` field, F64Map summary with band fractions; `audit.sh` |
| 9 | V | The field colormap spans the data's range, so one unmodelled part (a lever 180 mm out) hides every other residual | `render --color-range lo,hi`, a clamped span carried in the preview plan (the GUI's control is still to come) |
| 10 | M | A bolt hole is crisp in a photograph and blurred to nothing in the cloud, and nothing let an agent read a picture at full resolution with coordinates, or turn a pixel into a world point | `view-crop` (magnified crop with a pixel grid, crosses at pixels and at projected world points), `view-pick` (pixels cast onto a plane, world points into pixels, results in a frame's chart, distortion honoured), `view-triangulate` (a feature picked in two or more views, with each ray's miss distance) |
| 3 | B | Operators without READMEs: cylinder, revolve, extrude, subspace, sweep, slice, model_bound, mesh_to_model | write as each is used |
| 4 | B | No path sweep (a tube along a curve): the arms need one | two-view intersection (toy car) or an SDF script; decide in B |

## Log

- 2026-09-11: survey here of the 44 stills: all posed, 0.371 px rms, f
  6904.4 ± 1.1 px (the shim: 6904.4), card along span 179.32 mm against
  179.44 by caliper, planarity 0.030 mm.
- 2026-09-11: `base.sh` first version (25 steps, 8.6 s): the edge overlay
  through DSC00730/00742/00750 follows the arms, hub, lift and casters to
  within a few pixels at 1548 px wide; the rail and bracket outlines sit
  on the photograph's after the line fit replaced the PCA azimuth. Not
  modelled: the caster hoods and swivel state, the levers and their
  stalks, the tension knob, the rail's holes and slots, the bracket's
  bolt holes. Next: the bracket's hole pattern (what the seat mounts to),
  the underside of the arms (unseen by every camera; assume the tube
  section symmetric about the visible top), numeric residuals cloud →
  model.
- 2026-09-11: the full shoot `chairbase-dslr-1-all` (192 stills, the 44
  among them) surveyed here: 182 posed at 1.17 px rms (the shim 186 at
  1.49), f 6900 ± 1.4 px, card corners 0.033 mm median from the shim's,
  poses 2.0 mm / 0.08° median from the shim's over 180 frames (the dark
  and blurred frames of the wider set). Ten of its views (the top-down
  DSC00742, low DSC00730/54/23, the high ring DSC00739/49/27/45/60/31)
  are loaded into `chair_base.vproj` by `base.sh`, previews embedded, so
  `render --through` works on the base project itself and the GUI shows
  the photographs. Through them: arms, hub and casters follow the
  photographs; the near arm in DSC00754 shows the arm's underside curving
  up towards the caster where the model's is a straight offset, and the
  arm is wider than modelled near the tip.
- 2026-09-12: the residual audit against the unmasked full-run cloud
  `runs/chairbase-dslr-1-refuse2/cloud.ply` (481 113 points; clipped to
  the base's box above the carpet, 238 934) at 3.5 mm cells. What it
  showed of the first base: the arms were too wide at the hub (the cloud's
  edges 5–10 mm inside the model), too high mid-arm, and ended 20–30 mm
  short of the caster sockets (the socket band and casters 15 mm and more
  outside); the hub, lift and rail within a cell; the bracket's ends and
  the levers, knob and caster bodies unmodelled (yellow beyond 15 mm).
  The arm rebuilt to the cloud's own profile, measured per arm in a 20 mm
  strip along its centreline (r 0.06 → 0.34: top 166, 165, 162, 156, 150,
  144, 140, 137, 134, 128, 115, 90 mm; width 28, 30, 44, 50, 44, 30; the
  socket at r 0.34, top 70–90 mm; wheels at r 0.36 below 50 mm): the arms
  now read within 5 mm along their length; the sockets 5–10 mm. Arm
  azimuths from this cloud: −153.7°, −82.8°, −11.0°, 61.8°, 135.7°
  (spacing 70.9–73.9°); the spread is the one-sided surface coverage
  biasing each centreline by a few millimetres, not the casting, and 72°
  stays the model's spacing. Star centre from the hub rim's circle (0.260,
  0.152) against the lift tube's (0.256, 0.149): 4 mm apart, the same
  bias; the lift axis stays the datum.
- 2026-09-12: `audit.sh` measures per zone (an annulus for the arms, a
  cylinder for the column, a box above z 0.40 for the mechanism; each
  zone's cloud clipped and measured on its own). First base against the
  corrected one, fraction of the zone's points within 5 / 10 / 20 mm of
  the model, median absolute distance, fraction inside the model:

  | zone | points | first: 5 / 10 / 20 mm | p50 | inside | corrected: 5 / 10 / 20 mm | p50 | inside |
  |---|---|---|---|---|---|---|---|
  | arms | 110 834 | 0.70 / 0.87 / 0.95 | 3.0 | 0.35 | 0.69 / 0.91 / 0.95 | 3.1 | 0.22 |
  | column | 33 911 | 0.81 / 0.91 / 0.94 | 1.9 | 0.48 | 0.80 / 0.91 / 0.94 | 1.9 | 0.43 |
  | mechanism | 76 147 | 0.35 / 0.49 / 0.67 | 10.7 | 0.24 | 0.35 / 0.49 / 0.67 | 10.7 | 0.29 |
  | all | 238 934 | 0.55 / 0.68 / 0.77 | 4.2 | 0.30 | 0.54 / 0.70 / 0.78 | 4.3 | 0.25 |

  Reading: the arm rewrite moved the arms' 10 mm band from 0.87 to 0.91
  and their inside fraction from 0.35 to 0.22 (the first arms were fat;
  a surface cloud on a right model reads about half inside), but the
  5 mm band did not move: the corrected arms now sit a few millimetres
  small, most likely the 5 mm opening at 1.4 mm cells and the cloud's
  own outward bias, to be checked with the arm slab section before
  touching the numbers again. The column is right to a cell. The
  mechanism is the worst zone by far (median 10.7 mm, a third within
  5 mm): the boxes are crude, the levers, knob and bracket ends are
  absent; it is the next target, then the casters (bodies and hoods,
  wheels at r 0.36).
- 2026-09-12: the full-set cloud `runs/chairbase-dslr-1-all-refuse/
  cloud.ply` (2 367 623 points, 192 stills) against the same base. Its
  base surfaces read the same as refuse2's (the arm slab sections are
  indistinguishable; the column median 1.8 against 1.7 mm) but it carries
  far more room: carpet fuzz to 60 mm high inside the base's box, mounds
  at the box corners, floor patches. With the zones on the carpet the
  arms' 5 mm band read 0.58 against refuse2's 0.70 for that reason alone.
  Zones now start above the carpet (arms from z 0.04, column from 0.12)
  and a shell zone (points within 30 mm of the model, by `offset`) gives
  the number for the modelled parts on either cloud. The arm slab at
  ±10 mm shows the arm within 5 mm along its length, 3 mm small around
  r 0.15–0.20, and the model beyond the cloud over the last 60 mm to the
  socket (the tip dips too early); the caster hoods are not in the model.
  With the arms zone above the sockets (z 0.08) and the casters in a
  zone of their own, fraction within 5 / 10 / 20 mm and median:

  | zone | refuse2 | all-refuse |
  |---|---|---|
  | arms | 0.75 / 0.98 / 1.00, 2.8 mm (99 048) | 0.66 / 0.92 / 0.97, 3.4 mm (179 947) |
  | column | 0.90 / 1.00 / 1.00, 1.7 mm | 0.85 / 0.98 / 1.00, 1.8 mm |
  | mechanism | 0.35 / 0.49 / 0.67, 10.7 mm | 0.25 / 0.41 / 0.63, 14.2 mm |
  | casters | 0.16 / 0.27 / 0.52, 19.2 mm | 0.15 / 0.25 / 0.39, 29.7 mm |

  The arms and column are right to a cell on both clouds; the full-set
  cloud carries nearly twice the points in the arm zone (haze along the
  arms and carpet near the sockets), which is its extra residual. The
  casters and the mechanism are the model's remaining errors.
- 2026-09-12: the mechanism rebuilt around its mounting features (user:
  the dimples are not through-holes; the four bracket holes and the two
  rail slots are the real ones). Measured with the new `view-crop`,
  `view-pick` and `view-triangulate` (gap 10) and patch fits, table in
  Measurements. Every feature projected back into DSC00760 lands inside
  its hole. Two wrong turns on the way, both caught by the tools: the
  first hole sizes assumed 0.2 mm per pixel where the picture is 0.136
  at this depth (fixed from the triangulated depth); the first heights
  came from casting onto the whole top's fitted plane, whose 3.45° tilt
  was the bracket's end tabs and the dimples, and the oblique view showed
  slot B 6 mm off until the features were triangulated instead.
  Mechanism zone within 5 / 10 / 20 mm: 0.35 / 0.49 / 0.67 (boxes) →
  0.55 / 0.63 / 0.70, median 10.7 → 3.4 mm; the rest of that zone is the
  levers, the knob and the rail's internals, none of them mating
  surfaces. Not modelled on purpose: the rail's internal parts, the lever
  stalks, the knob, the bracket's underside.
