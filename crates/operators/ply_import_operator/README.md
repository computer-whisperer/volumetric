# PLY Import

Reads a PLY file (the Stanford polygon format that Blender, MeshLab and
most scan tools exchange) as an explicit triangle mesh. Like STL and 3MF
Import, the result is a `TriMesh` value, not a sampleable model: chain
Mesh To Model to get a solid that booleans, offsets, and meshing can use.

The file must carry faces. A PLY with vertices only is a point cloud;
import it with Point Cloud Import instead, which emits a `Point1` cloud
for the point operators (clip, transform, fill, Voronoi, remaster).

## What is honoured

- ASCII and binary bodies of either byte order, every scalar type, and
  list properties.
- Polygons of any size, fan-triangulated from their first vertex with the
  file's winding kept. Faces with fewer than three vertices are dropped
  and counted in a warning.
- Per-vertex normals (`nx ny nz`) as the `normal` vertex field and
  colours (`red green blue`, `r g b` or `diffuse_*`) as the `color`
  field, integer channels normalised to `[0, 1]`. The preview renders
  the colour field directly.
- Any other scalar vertex property, by name, through `fields`.

Coordinates are taken as metres, the engine's canonical unit: PLY carries
no unit, so use `scale` for a file authored in millimetres. Vertices are
kept as the file indexes them (no welding), since a PLY already shares
vertices between faces.

## Config

- `scale` (default 1): multiplier applied on the way in.
- `center` (default off): translate the result so its bounding box is
  centred on the origin, before `scale`.
- `fields` (default none): vertex property names to carry through as
  one-component vertex fields, e.g. `confidence` or `quality`.
