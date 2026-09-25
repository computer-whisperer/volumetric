# Keychain bodies for a Bluetooth e-paper tag

Two keychain bodies, a rigid slide-in sleeve and a TPU bumper, for a white Bluetooth e-paper shelf label, designed
from 21 raw stills of the tag lying display-down on the survey card
(`/ceph/christian/Photos/library/2026/2026-09-25/DSC02137.ARW` to
`DSC02157.ARW`, Sony a6700 + Viltrox 25 mm).

## Replay

From the repository root. `survey.sh` develops the raws with index_scanner's
`develop` (LibRaw; its venv supplies rawpy and OpenCV, override with
`PYTHON=`); everything else is volumetric.

```sh
cargo build --release --target wasm32-unknown-unknown -p wgsl_script_operator -p pose_operator
cargo build --release -p volumetric_cli
bash examples/keychain_tag/survey.sh            # raws -> survey.vviews
python examples/keychain_tag/hull.py            # visual-hull sections
python examples/keychain_tag/measure.py         # -> work/measurements.json
python examples/keychain_tag/build.py [--render] # -> work/keychain.vproj, keychain_{sleeve,bumper}.{3mf,stl}
```

A full replay from the raws into a fresh work directory reproduced every
number and the mesh (2026-09-25). `measurement-report.json` and
`parameters.json` are the committed snapshots of that run.

## Evidence

- Survey: 18 of 21 frames posed at 0.451 px RMS (DSC02155 rejected at
  3 px, DSC02156/57 not posed). The card's spans agree with its calipers
  (159.706 / 179.200 vs 179.443 mm), and it is planar to 0.037 mm. Scale
  comes from the card's calipers (`card.json`, copied from index_scanner).
- The raws report in-body stabilisation on, so the principal point moves
  per shot. The 0.45 px fit says that cost is small here. For
  measurement sessions, switch SteadyShot off.
- The tag frame: origin on the display face at the outline's centre,
  (111.20, −97.95) mm in the card frame, yaw +0.38°. x runs along the
  length (+x is the plain end, −x the end with the hanger slot), z comes
  out of the back.

The measurements are three independent readings, each through
`view-rectify`:

| Quantity | Reading | mm |
|---|---|---:|
| Back face height | logo plane sweep, 6 views | 13.66 |
| Face outline | contact lines at z = 0, edges on sides facing a camera | 69.5 × 34.7 |
| Face outline | visual hull, lowest section (strict) | 69.46 × 34.42 |
| Largest outline | visual hull, strict, over all heights | 70.37 × 35.70 |
| Back face outline | silhouette edges at z = 13.66 | 57.6 × 32.8 |
| Back face outline | visual hull at z = 13.75 (strict) | 57.4 × 31.0 |

The visual hull (`hull.py`) takes each view's occlusion "shadow" on the
card: the photo rectified onto z = 0 against the rendered nominal
ChArUco card. A point is inside when every view's ray through it lands in
that view's shadow; the robust variant lets 2 views dissent, to absorb
segmentation slips at square edges. The hull bounds the tag from outside,
so the pocket is sized from it. Overlays of the envelope through
DSC02143 and DSC02150 (`build.py --render`) sit just outside the tag's
silhouette all round.

Seen but not modelled: three latch tabs on the −y side (z ≈ 9.5–14 mm), a
comb of clips over a slot on the +y side, and a hanger tab with a through
slot on the −x end (z ≈ 4.8–8.8 mm). All lie inside the envelope, and
the sleeve covers them.

## The sleeve

`keychain.wgsl` part 1, overall 82.3 × 39.9 × 16.6 mm, 10.2 cm³:

- Pocket: the largest outline plus 0.3 mm clearance (71.0 × 36.3 ×
  13.96 mm), the same at every height, open at +x. The tag slides in
  display first against the front lip.
- Front lip 1.0 mm thick, overlapping the face's measured outline by
  1.0 mm on three sides; the display window is the rest.
- Back plate 1.6 mm, with a 16 mm tongue between two 0.8 mm slits at the
  open end. A 1.4 mm detent under it, ramped for insertion, drops 1.0 mm
  behind the tag's back face and holds the tag in. Lift the tongue to
  slide the tag out.
- Walls 1.8 mm, 0.8 mm edge rounds, keyring lug on the closed end (4 mm
  thick, Ø5 mm hole).
- Check: 0 of 49,576 points of the tag envelope fall inside the sleeve
  (`work/fit.json`). The superslicer check passes: manifold, one part.

Print back-plate down; the front lip then overhangs the pocket by about
1.3 mm, which prints without support. PETG gives the tongue its spring.

## The bumper (TPU)

`keychain.wgsl` part 2, overall 82.1 × 38.7 × 15.9 mm, 5.7 cm³, with a
full surround and no entry slot:

- Pocket: the measured envelope itself (the tapered two-stage shape of
  part 0) at 0 clearance. The envelope is an outer bound of the tag, so
  that is snug, not tight. Raise `assumed_tpu_clearance` if a print grips
  too hard.
- Walls 1.5 mm following the taper, 0.6 mm chamfers at both faces.
- Front lip 1.0 mm thick, all round the display face, overlapping its
  outline by 1.0 mm (window 67.5 × 32.4 mm).
- Back frame 1.2 mm thick, overlapping the back face's outline by 2.0 mm
  (opening 53.6 × 28.8 mm). Fitting: seat the display in the front lip,
  then stretch the back frame over. The opening's perimeter is about
  163 mm against the tag's largest outline of about 207 mm, so the TPU
  stretches roughly 27 % while the tag goes in. Soft TPU (95A or lower)
  helps.
- Keyring lug off the −x end on the display side, 3.5 mm thick, Ø5 mm
  hole.
- Check: 0 of 49,576 envelope points fall inside the bumper. The
  superslicer check passes: manifold, one part. Decimation leaves the
  volume within 0.01 % of the unsimplified mesh.

Print display face down: the lug and front lip lie on the bed, the walls
lean in at most about 31° from vertical, and the back frame's 2 mm
overhang is the only bridge.

## Assumed, not observed

- The display face was on the card, so its bezel width is unseen. The
  1.0 mm lip assumes the display's active area starts at least that far
  inside the face outline.
- Clearances (0.3 mm sleeve, 0 bumper), walls, lips, the tongue and
  detent, the lugs, and the tag's 3 mm corner radius are design choices
  (`assumed_*` in the WGSL).
- The envelope model (part 0) is a two-stage taper through the hull's
  outlines. It is a fit check, not the tag's true shape.

## Tooling added while doing this

- `view-rectify`: resamples a view onto any world plane as a metric
  image with a millimetre grid. It gives plan and elevation views, height
  plane sweeps, and edges read directly in millimetres.
- Stills intake reads EXIF from a raw of the same name beside a
  developed JPEG. Sony's maker note is also read bare, as it sits in an
  ARW, so focus and SteadyShot keys come through for raw-only sessions.

Lessons:

- Reading coordinates off downscaled crops by eye went wrong twice in
  this session; casts and rectifications did not.
- An edge detector locked onto a card square boundary
  (4 × 17.944 = 71.78 mm) and reported it as the tag's end. `measure.py`
  now flags edges within 0.4 mm of a card line.
