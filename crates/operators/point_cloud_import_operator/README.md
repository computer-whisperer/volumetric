# Point Cloud Import

Reads a point-cloud file as a `Point1` cloud: one point per vertex, with
its colour and normal when the file carries them. The result feeds the
point operators (Mesh Clip, Mesh Transform, Voronoi Skeleton, Mesh
Remaster) and renders as a coloured cloud in the viewport.

Formats: PLY (ASCII or binary, either byte order). A PLY that also has
faces imports its vertices alone; use PLY Import for the mesh.

## What is honoured

- Positions `x y z` (required), normals `nx ny nz` as the `normal` node
  field, and colours (`red green blue`, `r g b` or `diffuse_*`) as the
  `color` node field, integer channels normalised to `[0, 1]`.
- Any other scalar vertex property, by name, through `fields`: a scan's
  `confidence`, a splat's `opacity`, and so on.

Coordinates are taken as metres, the engine's canonical unit; use
`scale` for a file authored in millimetres.

## Config

- `scale` (default 1): multiplier applied on the way in.
- `center` (default off): translate the cloud so its bounding box is
  centred on the origin, before `scale`.
- `stride` (default 1): keep every nth point. A cheap way to thin a
  multi-million-point scan at import; `1` keeps everything.
- `fields` (default none): vertex property names to carry through as
  one-component node fields.
