# Voronoi Skeleton

Builds the Voronoi edge skeleton of a point cloud: every input point
becomes a cell seed, and the output Bar2 FeaMesh is the strut network
along the boundaries where three or more cells meet — the same
construction behind the built-in foam lattice, but with the site set under
user control. A point-fill bcc cloud at seed 0 reproduces the foam
family's skeleton exactly; edited clouds (clipped, merged, transformed)
make foams the built-in family can't.

Removing sites from a cloud makes the *neighboring cells grow* into the
vacated space (locally coarser foam); it does not cut holes. For holes,
clip the Bar2 output with the mesh-clip operator instead.

## Boundary handling

Hull cells are infinite; `boundary` picks what happens to their outward
edges:

- `"trim"` (default) keeps only edges supported by genuine Voronoi
  vertices on both ends AND within `max_reach` local spacings of their
  cell's site — infinite hull edges vanish and the hull shell's ballooning
  vertices (near-coplanar site slivers with huge empty circumspheres) are
  cut, so the skeleton simply ends at the cloud.
- `"box"` truncates hull edges at the cloud's bounding box plus `padding`
  (endpoints on the box, no reach cap), for when a downstream mesh-clip
  should cut the rays to a real domain surface as skin-contact stubs.

There is deliberately no domain input: clip the output against a model
with the mesh-clip operator — before or after editing — which also welds
the boundary stubs clipping creates.

## Inputs

1. **Sites** — FeaMesh (must be Point1): the cell seed sites, e.g. from
   the point-fill operator.
2. **Config** — CBOR configuration:

## Configuration

- `boundary` (default `"trim"`) — see above.
- `max_reach` (default 1.5) — trim only: cap on edge reach in units of
  each cell's nearest-neighbor distance; 0 disables.
- `radius` (default 0.0 = typical spacing / 10) — the uniform strut radius
  written to the output's `radius` element field, in metres. **Downstream
  FEA and the strut-model renderer both read this field** — set it to the
  radius you intend to print (e.g. 0.0003 for 0.3 mm struts) so analysis
  and geometry agree; it also scales the weld threshold below.
- `weld_factor` (default 1.0) — welds struts shorter than
  `weld_factor × radius`; 0 disables.
- `padding` (default 0.0 = two typical spacings) — box mode only.

## Output

CBOR-encoded Bar2 FeaMesh with a uniform scalar `radius` element field.
