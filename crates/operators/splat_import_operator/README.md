# Splat Import

Reads a trained Gaussian splat from the 3DGS PLY layout as a `Splat`
value: the centres, log-scales, rotation quaternions, logit opacities
and spherical-harmonic colours as the trainer stored them, in the world
frame of the view set the splat was trained from. The result is what
`splat-list` describes, Splat Points turns into a cloud, and the viewport
draws over the photographs.

Formats: PLY (ASCII or binary, either byte order) with the 3DGS vertex
properties. Any trainer that writes that layout (the reference 3DGS code,
gsplat, 2DGS) is read; a PLY without `f_dc_*`, `opacity`, `scale_*` and
`rot_*` is a point cloud, not a splat, and Point Cloud Import reads it.

## What is honoured

- `x y z`: the centre, metres.
- `f_dc_0..2`: the degree-0 colour; `f_rest_*`: the higher-degree
  coefficients, 9, 24 or 45 of them for SH degree 1, 2 or 3 (none for
  degree 0), channel-major as the file stores them.
- `opacity`: the logit opacity.
- `scale_0..2`: the log-scales. A file with only two (a 2DGS surfel
  export) reads as surfels with a third scale of one nanometre.
- `rot_0..3`: the quaternion, `w x y z`, normalised when used.
- `nx ny nz`: normals, kept when any is nonzero. The reference exporter
  writes zeros; those are dropped and the normal is derived from the
  orientation instead.

## Inputs

1. **Splat file** — the PLY.
2. **Views** — optional: the view set the splat was trained from. The
   splat takes its world frame and provenance (session, field, setup) and
   records the set's content hash as `views_hash`, so a later audit can
   tell which evidence the splat explains. Leave it unwired to set those
   through the config instead.
3. **Config** — see below.

## Config

- `kind` (default `auto`): `gaussian`, `surfel`, or `auto`, which reads
  surfels when the third scale is under a hundredth of the other two for
  most primitives (or absent). gsplat's 2DGS mode trains two scales and
  exports the third untrained at its initial value, so its files look
  like Gaussians to `auto`: pass `surfel` for a run from it (the
  trainer's `args.json` says which mode it ran).
- `scale` (default 1): multiplier applied to the centres and the scales
  on the way in, for a splat trained in a scaled frame. Must be positive.
- `center` (default off): translate the centres so their bounding box is
  centred on the origin, before `scale`.
- `session`, `field`, `setup` (default empty): the provenance of the
  capture; each overrides the view set's when non-empty.
- `views_hash` (default empty): the content hash of the view set the
  splat was trained from, when the set is not wired in.
- `training` (default empty): the trainer and its arguments, as text.

## Warnings

The file's faces (a splat has none) and all-zero normals are reported
and ignored; the detected kind is reported when `kind` is `auto`.
