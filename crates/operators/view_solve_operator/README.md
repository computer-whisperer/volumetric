# Solve Still

Poses a photograph from the marker cards it shows and adds it to a view
set as a posed view, picture embedded. A phone still of the subject with
the cards in frame becomes evidence beside the scan's own frames: look
through it in the viewport, check a model against it, measure from it.

The set supplies the marker map (each card's id and four world corners,
as the scan solved them). The operator finds the cards in the picture,
matches them to the map by id, and solves the camera pose that projects
the map's corners onto the detected ones. The pipeline is the same one
the `view-solve` command runs.

## Inputs

1. **Views** — the view set with the marker map. Its views and cameras
   are kept; the still is appended.
2. **Picture** — the still's bytes (JPEG or PNG). Import the file as a
   blob and wire it here.
3. **Config** — see below.

## Output

The same view set with one more view, tagged `still`, `solved:markers`,
`rms:<px>` (the reprojection residual) and `cards:<n>` (the cards the
pose rests on). The picture is embedded unless `embed_image` is off. The
camera the still was solved with joins the set's cameras (or reuses an
equal one).

## Configuration

- `id` (default `still`): the new view's id; `_2`, `_3`… when taken.
- `dictionary` (`5x5_100` or `4x4_50`): the swatch cards or the
  calibration board.
- `focal_px` (default 0 = unknown): a known focal length in pixels, with
  the principal point at the picture's centre. Unknown, the focal is
  seeded from the picture's EXIF 35 mm equivalent, or from `fov_deg`
  when there is none, and solved from the cards together with the first
  radial distortion term.
- `k1` (default 0): a known first radial term, used with `focal_px`.
- `fov_deg` (default 70): the horizontal field of view that seeds an
  unknown focal without EXIF.
- `solve_focal`, `solve_distortion`: solve those even with `focal_px`.
- `embed_image` (default on): keep the picture in the view.

## Reading the result

The step reports one line per solve — cards found and in the map, the
residual over the corners used, the focal with its standard error — and
warnings a reader should weigh the pose by:

- **single card**: the pose rests on one card, whose planar ambiguity
  is unresolved; add cards or take the picture from further off axis.
- **focal weakly constrained**: every card lies in one plane seen
  square-on, so the focal trades against the distance; the standard
  error says by how much. Tilt the camera or add a card off the plane.
- **large residual**: the corners do not fit the map to the accuracy the
  picture allows — lens distortion beyond one radial term, a card that
  moved since the scan, or a map from a different setup. A set whose
  `setup` label differs from the picture's session is the first thing to
  check.

Detection runs on the full picture (a 12 MP phone still holds 100–300 px
cards); corners are refined to a fraction of a pixel by fitting each
card's edges.
