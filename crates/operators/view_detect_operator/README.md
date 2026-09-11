# Detect Cards

Finds the swatch markers, the survey card's tags and the card's interior
corners in every picture a view set carries, and stores what it found on
each view. The survey solves cameras, poses and the marker map from these
observations; the viewport draws them over the photograph in look-through
so a reader can see what the solve rests on.

The pipeline is `cv_core::observe`, the same one the `view-detect` command
runs: quads are searched on the picture reduced to about 1600 px, then
every corner is refined and every cell read at full resolution; the card's
corners are placed from the decoded tags beside them and settled on the
saddle point of the grey picture; edge blur is measured across the
markers' edges.

## Inputs

1. **Views** — the view set. Views without a picture are skipped.
2. **Config** — see below.

## Output

The same view set, each detected view carrying `observations`: the
markers of every family with their corners and edge-fit residual, the
card's corners with the distance the refinement moved them, and the blur
in pixels. The set's `board` records the card spec the corners refer to.

## Configuration

- `dictionary` (`5x5_100`, `4x4_50` or `none`): the swatch family.
- `card` (default on): look for the survey card.
- `squares_x`, `squares_y`, `pitch_x_mm`, `pitch_y_mm`, `marker_mm`,
  `family`, `first_id`: the card's spec; the defaults are the survey
  card (12 x 11 squares at the caliper pitches 17.944 x 17.745 mm,
  12.7 mm AprilTag 36h11 tags from id 100).
- `search_px` (default 1600): the reduction quads are searched at
  (0 = full resolution).

## Reading the result

The step reports one line per view: markers per family, card corners,
blur. Views without a picture are named. A view with no card corners in
a session that has the card in view is worth looking at through the
viewport: the card may be out of focus (the a6700 at f/4 holds a
centimetre of depth of field), or lie under the subject.

A 26 MP still takes about a quarter of a second to detect in, plus the
JPEG decode; through the wasm path count a few seconds per view.
