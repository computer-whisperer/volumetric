# Keychain bodies for Bluetooth e-paper tags

Keychain bodies for two Bluetooth e-paper shelf labels, each designed from
raw stills of the tag on the survey card. Both get a rigid slide-in sleeve
and a TPU bumper:

- `stellar_l3n/`: the Hanshow Stellar-L3N, photographed 2026-11-10 display
  down and display up.
- `tag_b/`: the white tag with a blue "B" on its back, photographed
  2026-09-25 display down.

## Replay

From the repository root. `survey.sh` develops the raws with index_scanner's
`develop` (LibRaw; its venv supplies rawpy and OpenCV, override with
`PYTHON=`). The Python scripts need OpenCV too (the same venv). Everything
else is volumetric.

```sh
cargo build --release --target wasm32-unknown-unknown -p wgsl_script_operator -p pose_operator
cargo build --release -p volumetric_cli
# Stellar-L3N
bash examples/keychain_tag/survey.sh examples/keychain_tag/stellar_l3n    # raws -> work/survey.vviews
python examples/keychain_tag/hull.py examples/keychain_tag/stellar_l3n down_a
python examples/keychain_tag/hull.py examples/keychain_tag/stellar_l3n down_e
python examples/keychain_tag/stellar_l3n/measure.py                       # -> work/parameters.json
python examples/keychain_tag/build.py examples/keychain_tag/stellar_l3n [--render]
# B tag
bash examples/keychain_tag/survey.sh examples/keychain_tag/tag_b
python examples/keychain_tag/hull.py examples/keychain_tag/tag_b down
python examples/keychain_tag/tag_b/measure.py
python examples/keychain_tag/build.py examples/keychain_tag/tag_b [--render]
```

`build.py` writes `work/keychain.vproj` (bodies, the tag's envelope posed
into the survey with a few photographs for overlays), `work/fit.json` and
the printable `work/keychain_{sleeve,bumper}.{3mf,stl}` in millimetres.
`measurement-report.json` and `parameters.json` in each tag directory are
the committed snapshots of a run.

## Layout

- `survey.sh <tag-dir>`: verify the raws against the tag's
  `photo-manifest.json`, develop, intake, detect the card, survey.
- `hull.py <tag-dir> <pose>`: visual hull sections of one pose from card
  occlusion; `--shadows` gives every view's occlusion, for grouping views
  into poses.
- `cameras.py`: the survey's cameras in Python. It projects world points
  into the developed photographs exactly as volumetric's camera model does
  (checked against `view-rectify` to 0.005 px), and renders the nominal
  card at world points.
- `build.py <tag-dir>`: the project, the fit check and the meshes, from the
  tag's `parameters.json` and the `build` section of its `tag.json`.
- Per tag: `tag.json` (poses, windows, build), `photo-manifest.json`,
  `measure.py`, `keychain.wgsl` (part 0 the envelope, 1 the sleeve, 2 the
  bumper).
- `card.json`: the survey card's calipers (copied from index_scanner).

## Stellar-L3N

### Evidence

40 raws, `/ceph/christian/Photos/library/2026/2026-11-10/DSC03238.ARW` to
`DSC03277.ARW` (Sony a6700, Viltrox 25 mm at f/5.6, ISO 100, AF-S,
SteadyShot off). DSC03237 (ISO 5000, white) was a settings shot and is
left out.

- Survey: all 40 frames posed at 0.147 px RMS. The card is planar to
  0.025 mm, and its long span reads 179.21 mm against the calipers'
  179.44, as in September (179.20).
- The frames are autofocus stills, a camera each. The survey now solves
  their focal lengths per frame, against one principal point and
  distortion for the lens; they run 6904 to 6932 px, each to 1 to 6 px.
- Poses: the tag moved between groups of frames. A view belongs to a
  pose when adding it leaves the pose's hull near the card intact (a view
  of another pose carves 100 to 1500 mm² from it). Three poses:
  - `down_a`, display down, DSC03238 to 47 in room light (DSC03242 to 44
    overexposed);
  - `down_e`, display down, DSC03264 to 77 under a lamp;
  - `up_c`, display up, DSC03250 to 57, all from one side.

  DSC03248/49 and DSC03258 to 63 are display up too, but the tag was
  nudged between nearly every frame; they are not used.

The tag frame: origin on the display face at the centre of its outline,
x toward the end whose back carries the model label, y toward the side
with a strip of clips, z out of the back. The back body is flush with
the face at +x and +y, and steps in at -x and -y onto ledges.

### Measurements

`measure.py` reads everything through `cameras.py` (`measurement-report.json`
has each view's reading):

| Quantity | Reading | down_a | down_e | Used |
|---|---|---:|---:|---:|
| Face outline | contact lines on white card squares | 90.44 × 40.61 | 90.58 × 41.04 | 90.44 × 40.61 |
| Back face (label) height | parallax zero-crossings | 12.40 | 12.37 | 12.38 |
| Logo panel height | parallax zero-crossings | 13.15 | 13.25 | 13.20 |
| Back body's top face | edge triangulation | | | x −37.15…+44.85, y −12.34…+19.26 |
| Logo panel | edge triangulation | | | x …+17.11, y −10.12…+17.16 |
| Ledge heights, −x / −y | brightness up the face's wall | 3.70 / 5.10 | 3.80 / 5.20 | 3.75 / 5.15 |
| Face corner radius | circles on the hull's lowest sections | 2.66 (2.33…2.99) | | 2.66 |
| Clip strip, stands out from +y | hull over it less hull beside it | 1.16 | 1.00 | 1.16 |
| Display (active area) | edges on the face plane, display up | | | x −33.42…+34.12, y −14.89…+14.84 |

All lengths in millimetres. In detail:

- Contact lines: from outside inward, the first place a facing view stops
  showing the card. The face's dark edge strip reads like a black card
  square, so only rows over white squares count. In `down_a` the views
  agree to 0.12 to 0.40 mm per side. In `down_e` the lamp's shadow on the
  card adds 0.4 mm on one side, so the outline comes from `down_a`; its
  length agrees (90.58).
- Heights: the label (or panel) rectified onto two planes in every view
  and phase-correlated against a reference view. Each view's shift grows
  linearly with the plane's error, and vanishes at its reading. Label
  views agree to 0.02 mm (IQR).
- Edges on the back: a straight edge's offset in planes at several
  heights, per view, follows that view's parallax; the height where the
  views agree is the edge's. Per edge, the pose whose views agree best
  (0.02 to 0.07 mm) is used.
- The display, display up: its edges agree across the eight views to
  0.1 mm. They are placed against the face's far edge (a silhouette) and
  the face's measured length, then turned over. The display is centred on
  the face to 0.4 mm. The barcode and the status LED sit at the −x end.

### The sleeve

`keychain.wgsl` part 1 (PETG), overall 102.7 × 45.6 × 16.1 mm, 14.5 cm³:

- Pocket: the face's outline plus 0.3 mm clearance (91.04 × 41.21 mm), up
  to the logo panel plus 0.3 mm (13.50 mm), open at −x. The tag slides in
  +x end first, display against the front lip.
- A groove in the +y wall lets the clip strip through: 1.16 mm beyond the
  pocket, 7.5 to 12.8 mm up, from the open end to past the strip's seat.
  The wall there is 1.0 mm.
- Front lip 1.0 mm thick, overlapping the face by 1.5 mm on three sides
  (window 87.4 × 37.6 mm, open at −x). It clears the display by 3.9 mm or
  more, and the LED and barcode stay visible at the open end.
- Back plate 1.6 mm, with a 16 mm tongue between two 0.8 mm slits, 25 mm
  long from the open end. Under it, a detent hangs into the −x step: its
  catch face is 0.5 mm beyond the back body's −x wall, its foot 1.8 mm
  below the back face, ramped 36° for the tag going in. The tongue lifts
  2.6 mm while the panel passes under it, about 1.0 % strain in PETG. Lift
  the tongue to slide the tag out.
- Walls 2.2 mm, 0.8 mm edge rounds, keyring lug on the closed +x end
  (4 mm thick, Ø5 mm hole).
- Check: 0 of 47,650 points of the envelope fall inside the sleeve
  (`work/fit.json`). The superslicer check passes: manifold, one part.

Print back plate down; the front lip then overhangs the pocket by 1.8 mm.

### The bumper (TPU)

`keychain.wgsl` part 2, overall 102.9 × 44.8 × 14.6 mm, 8.1 cm³:

- Pocket: the face's outline at 0 clearance at every height, bulging
  1.16 mm over the clip strip. The steps at −x and −y are air under the
  back frame.
- Walls 1.5 mm, 0.6 mm chamfers at both faces. Front lip 1.0 mm thick,
  overlapping the face by 1.5 mm all round (window 87.4 × 37.6 mm).
- Back frame 1.2 mm thick on the back face, overlapping the back body's
  top face by 2 mm at +x and −y, 1.8 mm at +y beside the panel. Its opening clears the logo panel by
  0.3 mm; the panel stands 0.82 mm proud and sits 0.38 mm below the
  frame's top. At −x the panel nearly reaches the body's edge, so there
  the frame only bridges the step.
- Fitting: stretch the back frame over the tag. The opening's perimeter
  is about 216 mm against the face's 258 mm, about 20 % stretch (the B
  tag's bumper: 27 %).
- Keyring lug off the +x end on the back side, 3.5 mm thick, Ø5 mm hole.
- Check: 0 of 47,650 envelope points fall inside the bumper. The
  superslicer check passes: manifold, one part.

Print back frame down: it bridges up to 10 mm over the steps, so it goes
on the bed, with the lug. The front lip's 1.5 mm overhang prints last.

### Assumed, not observed

- The clip strip's extent, x −11.5…+11.5 and 8.5 to 11.8 mm up, comes from
  two photographs (DSC03274, DSC03238); how far it stands out is measured.
  The sleeve's groove and the bumper's bulge add 1 mm all round.
- The back body's top edges are rounded; its walls are taken 1.0 mm
  outside the measured top-face edges (the detent's place, the envelope).
- The panel's −x edge was not separated from the body's; it is taken
  0.4 mm inside.
- The face's far edge is rounded; it is taken 0.5 mm inside the outline
  when placing the display, which is only a check.
- The corner radius is read from the hull, not from contact lines: no
  corner lay on white card squares.
- Clearances, walls, lips, the tongue and detent and the lugs are design
  choices (`assumed_*` in the WGSL).

### Lessons

- Autofocus stills made every frame a camera of its own, and the survey
  left them all out. It now groups them by lens (see Tooling).
- Plane sweeps found false minima twice: the first window held card and
  wall at other heights, and the first range missed the label (12.4 mm, not
  15). The parallax zero-crossing localises a height from two planes and
  says when it is unsure.
- The visual hull here bounds the tag but cannot size it. No view looks
  steeper than 68°, so it overshoots a vertical wall by z/tan 68° (2 mm at
  5 mm up), and it fills the steps. A lamp's shadow on the card enters it
  near z = 0 in every view. Contact lines and edge triangulation carried
  the outline.
- For the next session: a few views near overhead and a few low ones would
  let the hull carve walls; diffuse light avoids cast shadows; keep the tag
  still within a group of frames.

## The B tag

Designed from 21 raw stills of the tag lying display down on the survey card
(`/ceph/christian/Photos/library/2026/2026-09-25/DSC02137.ARW` to
`DSC02157.ARW`, Sony a6700 + Viltrox 25 mm).

### Evidence

- Survey: 18 of 21 frames posed at 0.451 px RMS (DSC02155 rejected at
  3 px, DSC02156/57 not posed). The card's spans agree with its calipers
  (159.706 / 179.200 vs 179.443 mm), and it is planar to 0.037 mm.
- The raws report in-body stabilisation on, so the principal point moves
  per shot. The 0.45 px fit says that cost is small here.
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
| Face outline | visual hull, lowest section (strict) | 69.14 × 34.40 |
| Largest outline | visual hull, strict, over all heights | 70.36 × 35.70 |
| Back face outline | silhouette edges at z = 13.66 | 57.6 × 32.8 |
| Back face outline | visual hull at z = 13.75 (strict) | 57.4 × 31.0 |

The visual hull takes each view's occlusion "shadow" on the card: the
photo rectified onto z = 0 against the rendered nominal ChArUco card. A
point is inside when every view's ray through it lands in that view's
shadow; the robust variant lets 2 views dissent, to absorb segmentation
slips at square edges. The hull bounds the tag from outside, so the pocket
is sized from it.

Seen but not modelled: three latch tabs on the −y side (z ≈ 9.5–14 mm), a
comb of clips over a slot on the +y side, and a hanger tab with a through
slot on the −x end (z ≈ 4.8–8.8 mm). All lie inside the envelope, and
the sleeve covers them.

Replayed 2026-11-11 through the shared scripts: the survey reproduces to
the last digit, and every parameter is within 0.04 mm of the September
snapshot except the lowest hull section's length, 0.32 mm shorter (69.14
against 69.46 mm). The generic `hull.py` fits each view's card gain over a
larger window, and the lowest section is the sensitive one. It sets the
display windows, now 0.32 mm shorter.

### The sleeve

`tag_b/keychain.wgsl` part 1, overall 82.3 × 39.9 × 16.6 mm, 10.2 cm³:
pocket the largest outline plus 0.3 mm (71.0 × 36.3 × 13.96 mm), open at
+x; front lip 1.0 mm overlapping the face's outline by 1.0 mm on three
sides; back plate 1.6 mm with a 16 mm tongue between 0.8 mm slits and a
1.4 mm detent that drops 1.0 mm behind the back face; walls 1.8 mm, keyring
lug on the closed −x end. 0 of 51,182 envelope points inside; manifold,
one part. Print back plate down (front lip overhang 1.3 mm), PETG.

### The bumper (TPU)

`tag_b/keychain.wgsl` part 2, overall 82.1 × 38.7 × 15.9 mm, 5.7 cm³:
pocket the measured envelope (a two-stage taper) at 0 clearance; walls
1.5 mm following the taper, 0.6 mm chamfers; front lip 1.0 mm overlapping
the face by 1.0 mm; back frame 1.2 mm overlapping the back face by 2.0 mm
(about 27 % stretch to fit); keyring lug off −x on the display side.
0 of 51,182 envelope points inside; manifold, one part. Print display face
down.

### Assumed, not observed

- The display face was on the card, so its bezel width is unseen. The
  1.0 mm lip assumes the display's active area starts at least that far
  inside the face outline.
- Clearances, walls, lips, the tongue and detent, the lugs, and the tag's
  3 mm corner radius are design choices.
- The envelope is a two-stage taper through the hull's outlines: a fit
  check, not the tag's true shape.

## Tooling added along the way

- 2026-11: the survey groups autofocus stills by lens. Each frame keeps
  its own focal length; the frames of one body and lens share the
  principal point and distortion, so a frame that could not calibrate
  alone is solved against all of them. A weak prior holds each focal
  length to the lens's median with a 3 % spread, for a frame that sees
  the card square on (focal length and distance trade there). Held-focus
  sessions solve as before.
- 2026-09: `view-rectify` (a metric image of any world plane: plans,
  elevations, plane sweeps), and stills intake reading EXIF from a raw of
  the same name beside a developed JPEG (Sony's maker note read bare).
