# Mesh to Model

Turns an explicit triangle mesh (`TriMesh`, as STL, PLY, 3MF and OBJ
Import produce) into a sampleable solid: point-in-mesh by ray-crossing
parity over a BVH built once at conversion time. The output is an
ordinary model, so booleans, offsets, posing and meshing all apply.

A watertight mesh behaves as the solid it encloses. Open meshes are not
rejected — a point counts as inside when an odd number of triangles lie
between it and infinity along a fixed ray — but only closed surfaces
give meaningful solids.

## Surface colour

A mesh that carries colour comes out as a coloured model. Two sources
are recognised, in this order of preference:

- a base-colour texture (the mesh's first image) addressed by the `uv`
  vertex field — what OBJ Import produces from `map_Kd`;
- the `color` vertex field (sRGB in 0..1) — what PLY and 3MF Import
  produce.

The model then declares the sRGB channel trio `color_r`, `color_g`,
`color_b` next to occupancy. The colour at any point is that of the
nearest surface point (barycentric interpolation of the corner
attributes; bilinear, repeat-wrapped texel fetch for a texture), so
samples slightly off a face, or deep inside the solid, read the nearest
face. Operators that propagate channels (boolean, pose, translate,
pattern...) carry the colour through, and the preview's true-colour mode
shows it. A mesh with neither source yields the plain occupancy-only
model and declares no colour channels.

A texture that cannot be decoded (truncated file, unsupported format) is
a warning, not a failure: the step reports it by name and converts the
geometry without it, falling back to vertex colour or to no colour.

Textures are decoded (JPEG, PNG) and stored inside the model, by default
BC1-compressed (4 bits per texel, 6x smaller than raw RGB, lossy in the
way GPU textures usually are). A scanner's 8192² atlas is 32 MiB as BC1
and 192 MiB raw; use `texture_max_size` when the model is going to be
meshed far coarser than the atlas resolves.

## Config

The second input is optional: a step with only the mesh input (as every
project written before this slot existed has) gets the defaults.

- `texture` (default `"bc1"`): `"bc1"`, `"rgb8"` (lossless, 24 bits per
  texel) or `"none"` (ignore the texture; fall back to vertex colour, or
  to no colour).
- `texture_max_size` (default 0 = full size): box-filter the texture
  down by halves until both extents are at most this many texels.
