# Volume Import

Reads a sampled scalar volume and emits a solid. The field's isosurface
becomes the model's occupancy, and the field itself becomes its
`signed_distance` channel: the same two channels Generate SDF bakes, so
meshing, Offset and everything downstream treat an imported scan like any
other model. A fused scan (TSDF), a thresholded CT stack or any regular
grid of numbers comes in the same door.

The file format is NRRD ("nearly raw raster data"), the plain container
3D Slicer, ITK, teem and pynrrd exchange: a text header naming the sample
type, sizes, spacing and origin, followed by the samples raw, gzipped or
as text. Write one from numpy with pynrrd, or from ITK/Slicer directly.

## Conventions

- **Axes.** Every axis must be spatial and along its own coordinate axis
  (`space directions` diagonal and positive, or `spacings`); a rotated,
  sheared or flipped volume needs resampling first. Channel axes (`kinds:
  list`, `vector`, ...) are refused: write one scalar volume per channel.
  A 2-axis file gives a 2D model, a 3-axis file a 3D one.
- **Unit.** NRRD carries no length unit. Positions and field values are
  taken in the file's unit and multiplied by `scale`, so a grid in
  millimetres with distances in millimetres imports with `scale: 0.001`.
- **Surface and sign.** The surface is where the field equals
  `threshold`. With `inside: below` (the default, for signed distance
  fields) samples below it are solid; with `above` samples above it are
  solid, as for a density.
- **Band.** The field is clamped to `±band` from the threshold, and the
  volume returns `+band` (empty) everywhere beyond its samples. `band: 0`
  uses the largest magnitude in the data. For a truncated distance field
  set it to the truncation distance; for a density it is only the scale of
  the channel.
- **Unobserved samples.** A NaN sample is one the volume never observed:
  a scan's interior, or space no camera covered. `unobserved: enclosed`
  (the default) fills a NaN region as solid when observed samples enclose
  it and leaves it empty when it reaches the volume's edge, so a closed
  scan becomes a solid object and an open one stays open. `solid` and
  `empty` force one answer.
- **Crop and stride.** `crop` keeps the solid plus one band of margin,
  which is all the field needs, and shrinks a scan grid that is mostly
  empty space by orders of magnitude. `stride` keeps every n-th sample
  along each axis, for grids finer than the model needs.

The imported model's channel is exactly `volumetric.tsdf.v1`: a signed
distance for a distance-field file, and a re-signed, clamped field value
for anything else.

## Config

- `scale` (default 1): metres per file unit.
- `center` (default off): centre the solid's bounding box on the origin,
  before `scale`.
- `threshold` (default 0): the isovalue.
- `inside` (default `below`): `below` or `above`.
- `band` (default 0 = largest magnitude): clamp distance, file units.
- `unobserved` (default `enclosed`): `enclosed`, `solid` or `empty`.
- `crop` (default on).
- `stride` (default 1).
