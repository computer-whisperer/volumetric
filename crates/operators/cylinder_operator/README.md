# Cylinder

An analytic cylinder or capsule between two points: the round primitive
for shafts, bosses, pins, holes to subtract, and anything a scan's
cylinder fit hands back as an axis and a radius.

## Inputs

| Slot | Type | Meaning |
|---|---|---|
| Config | CBOR | `radius` (metres, default 5 mm), `cap` |
| Endpoint A | VecF64(3) | one end of the axis |
| Endpoint B | VecF64(3) | the other end |

`cap` is `flat` (default): a true cylinder, ending in the two planes
through the endpoints. `round`: a capsule, hemispheres of the same radius
past each endpoint, so the overall length is the axis length plus two
radii.

## Output

A 3D model. A point is inside when its distance from the axis line is at
most `radius` and, for flat caps, its projection lies between the
endpoints; a capsule tests the distance to the segment instead. The
sample is exact (the quadric is evaluated at sample time, no mesh), so
the surface is as round as the meshing resolution allows.

Bounds are the box of the endpoints padded by the radius on every side,
exact for a capsule and slightly generous for a flat cylinder that is not
axis-aligned.

## Notes

- Coincident endpoints are an error; a zero-length capsule is a sphere,
  make it with the sphere primitive instead.
- Orientation comes from the endpoints alone; to place a cylinder on a
  fitted axis, use the fit's origin and direction to compute the two
  endpoints (origin ± half-length along the direction).
- CLI: `--input 'json:{"radius":0.026}' --input 'json:[0,0,0]' --input
  'json:[0,0,0.1]'`; the endpoint vectors are the three components.
