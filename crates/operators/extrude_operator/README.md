# Extrude

Sweeps a profile along a plane's normal: a 2D sketch becomes a 3D slab of
a given height. The everyday sketch-and-extrude of CAD.

## Inputs

| Slot | Type | Meaning |
|---|---|---|
| Profile | Model (k-D, usually a 2D sketch) | the section to sweep |
| Config | CBOR | `height` (metres, default 1.0) |
| Plane | Subspace, optional | where the sketch lies and which way it sweeps |

The plane is a Subspace of rank k in (k+1)-space; for a 2D sketch, a
plane in 3-space (`subspace_operator` with `kind: plane`, or a
`cloud_fit` of kind `plane`). Sketch x runs along the plane's first basis
vector, sketch y along the second, both from the plane's origin; the
sweep goes from the plane to `height` along the plane's normal, which is
first basis × second basis. Unwired, the plane is the world xy plane at
z = 0 with a +z normal and the profile must be 2D.

## Output

A model one dimension higher than the profile: a point is inside when its
normal coordinate lies in [0, height] and its in-plane coordinates are
inside the profile. Sample channels pass through; outside the slab the
occupancy is forced to zero. Bounds are the profile's chart box swept
through the height.

## Notes

- `height` is always positive. To extrude the other way, flip the plane:
  swap its two basis vectors (or negate one), which reverses the normal.
- Because the normal is basis₁ × basis₂, a plane given as
  `origin, (1,0,0), (0,1,0)` extrudes towards +z, and as
  `origin, (0,1,0), (1,0,0)` towards −z. The toy car's plan-view extrude
  relies on this to sweep downward.
- A slab through a scan at a fitted plane: wire the fit's Subspace in;
  its origin is the fit's centroid, so centre the sketch at 0,0.
