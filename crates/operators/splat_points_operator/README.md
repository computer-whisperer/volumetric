# Splat Points

Turns a splat into a `Point1` cloud: one point per primitive at its
centre, with its normal (a surfel's third axis, a Gaussian's
smallest-scale axis, or the trainer's normal when it stored one), its
degree-0 colour, its opacity and its largest scale as node fields, so
Cloud Fit, Cloud Normals, Mesh Clip and the section tools apply to the
splat directly and a measurement can be taken from it.

## Inputs

1. **Splat** — the trained splat.
2. **Views** — optional; the view set whose solved markers `near.marker`
   refers to. Leave it unwired otherwise.
3. **Config** — see below.

## Output

A `Point1` cloud with the node fields `normal` (3), `color` (3),
`opacity` (1, in `[0, 1]`) and `scale` (1, the largest standard
deviation in metres).

## Configuration

- `min_opacity` (default 0.5): primitives below this opacity are left
  out. A trainer keeps many faint primitives that fill haze and shadow;
  the surface is in the opaque ones.
- `max_scale` (default 0, unlimited): primitives whose largest scale is
  above this, in metres, are left out; the large ones model background
  and sky.
- `min_scale` (default 0): primitives whose largest scale is below this
  are left out.
- `stride` (default 1): keep every nth primitive that passes the
  filters.
- `bounds` (optional): keep only centres inside the axis-aligned box
  `min_x..max_x` × `min_y..max_y` × `min_z..max_z`, metres.
- `near` (optional): keep only centres within `radius` metres of a
  point: the centre of the solved marker `marker` in the view set when
  it is named, else `(x, y, z)`.

The filters compose: a primitive passes all of them or is left out.
