# 3MF Import

Reads a 3MF package (the ZIP-of-XML format slicers and CAD tools exchange)
as an explicit triangle mesh. Like STL Import, the result is a `TriMesh`
value, not a sampleable model: chain Mesh To Model to get a solid that
booleans, offsets, and meshing can use.

## What is honoured

- The model's `unit` attribute. Geometry lands in metres, the engine's
  canonical unit, whatever the file was authored in — no scale guessing.
- Every build item, with its placement transform, and every component
  the item's object is assembled from, recursively. Components that live
  in other parts of the package (the production extension's `p:path`, as
  Bambu Studio writes) are followed. Mirroring placements flip the
  triangle winding so faces stay outward.
- Objects of every type. FreeCAD tags closed solids `type="surface"`.

Materials, colours, textures, thumbnails, and slicer settings are ignored.
Vertices that coincide exactly are welded into one, as STL Import does:
CAD exporters often write seam vertices twice, which would leave a closed
surface topologically open for Mesh To Model.

## Config

- `scale` (default 1): multiplier applied after the unit conversion, for
  resizing on the way in.
- `center` (default off): translate the result so its bounding box is
  centred on the origin, before `scale`.
- `item` (default 0): which build item to import. `0` merges all of them
  into one mesh; `n` takes the nth item (1-based) alone.
