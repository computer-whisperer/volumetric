# Survey Field

Solves a session of stills from the observations Detect Cards stored on
them: one camera model per focus setting (focal length, principal point,
two radial terms), a pose per view, and every card corner and swatch
corner as a point in the world. The card's caliper pitches set the scale;
the card's plane and axes define the world frame, with z pointing from
the card towards the cameras.

The bundle is `cv_core::survey`, the same one the `view-survey` command
runs: each camera is first calibrated on its frames that show the whole
card, frames are posed on the card, swatch corners triangulated and
further frames posed on them until nothing grows, then everything is
refined together under a soft-L1 loss, with frames the fit leaves above
`reject_px` dropped and the rest solved again.

## Inputs

1. **Views** — a view set with observations on its views and the card as
   its `board` (run Detect Cards first). Views without observations are
   left as they are.
2. **Config** — see below.

## Outputs

1. **Views** — the set posed: cameras replaced by the solved models,
   every surveyed view carrying its pose and the tags `solved:survey` and
   `rms:<px>`, `markers` holding the solved swatches (corners and side),
   `board` holding the card's solved corners with their uncertainties.
   Views the survey could not pose stay unposed.
2. **Report** — an F64Map: `rms_px`, `median_px`, `inlier_rms_px`,
   `frames_posed`, `frames_rejected`, `card_planarity_mm`,
   `span_across_mm` against `span_across_nominal_mm`, and per camera
   `camera.<i>.f` with `camera.<i>.f_std`, per swatch
   `swatch.<id>.side_mm` and `swatch.<id>.sigma_mm`, per frame
   `frame.<id>.rms_px`.

## Configuration

- `dictionary` (default `5x5_100`): the swatches' family; other marker
  observations (the card's tags) are not points.
- `max_swatch_id` (default 60): ids at or above this are false decodes.
- `min_card` (default 8): card corners a view needs to be posed on the
  card at the start.
- `f_scale` (default 1.5 px): the soft-L1 scale.
- `reject_px` (default 3): views the bundle leaves above this rms are
  dropped. Raise it for a defocused set.
- `rounds` (default 4): drop, extend and solve again this many times at
  most.

## Reading the result

A good DSLR session lands under a pixel rms with the card planar to a few
hundredths of a millimetre and the across span within 0.1 % of the
calipers. A camera's `f_std` above a few pixels means its frames do not
constrain the focal: too few frames, or all from one distance. A frame in
the rejected list is worth a look through the viewport: motion blur, the
card under the subject, or a swatch decoded with the wrong id.

Eighty stills take a few seconds natively and longer through the wasm
path; the normal equations are dense, so sessions of several hundred
frames are for the command line.
