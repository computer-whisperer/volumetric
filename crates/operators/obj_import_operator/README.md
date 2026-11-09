# OBJ Import

Reads a Wavefront OBJ file as an explicit triangle mesh, with its texture
when the host supplies one. Like STL, 3MF and PLY Import, the result is a
`TriMesh` value, not a sampleable model: chain Mesh To Model to get a
solid that booleans, offsets and meshing can use. Mesh To Model keeps the
texture as the model's surface-colour channels, so the colour survives
the operations stacked on top.

This is the import for photogrammetry and phone-scanner output
(Scaniverse, Polycam, RealityCapture), which is one OBJ, one MTL and one
large photographic atlas. The OBJ convention is that the user picks the
`.obj` and the program finds the sidecars: the host reads the `mtllib`
line, the MTL's `map_Kd`, and the image, and wires the image into the
second input. The operator itself never touches a filesystem.

## What is honoured

- `v` (with an optional per-vertex colour, as MeshLab writes it), `vt`,
  `vn`, and `f` corners in every form (`v`, `v/vt`, `v//vn`, `v/vt/vn`),
  1-based and negative (relative) indices, polygons of any size
  fan-triangulated from their first corner with the file's winding kept,
  `\` line continuations and `#` comments.
- Texture coordinates as the two-component `uv` vertex field (OBJ
  convention: `v` runs bottom to top). A corner is a vertex: positions
  shared by corners with different `vt` or `vn` are split, which is what
  a GPU needs and what Mesh To Model expects.
- Normals as the `normal` vertex field when every corner has one; vertex
  colours as the `color` field when every vertex has one.
- One texture: the bytes in the second input, attached as the mesh's
  first texture with its media type sniffed from the bytes (JPEG, PNG,
  BMP, GIF, WebP). `usemtl` names are read only to warn when the file uses
  more than one material, since only one atlas is carried.
- `o`, `g`, `s`, `mtllib` and parameter-space (`vp`, curves, surfaces)
  lines are ignored.

Coordinates are taken as metres, the engine's canonical unit: OBJ carries
no unit, so use `scale` for a file authored in millimetres or inches.
Scanner exports are already metres.

## Config

| Field | Default | Meaning |
|---|---|---|
| `scale` | `1.0` | Multiplies every coordinate after centring. |
| `center` | `false` | Translate the bounding-box centre to the origin first. |
| `texture_name` | `""` | The file name to record for the attached texture (the host fills in the `map_Kd` name); `texture` when empty. |

## Warnings

Dropped faces (fewer than three corners), corners without a `vt` when
others have one (they read `(0, 0)`), corners without a `vn` when others
have one (the field is omitted), more than one `usemtl`, and a texture
supplied for a file with no texture coordinates (the texture is dropped,
since nothing could address it).
