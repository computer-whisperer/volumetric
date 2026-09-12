# Revolve

Spins a profile around an axis into a solid of revolution: a 2D sketch
becomes a 3D body. Hubs, collars, wheels, tapered columns, anything a
lathe would make.

## Inputs

| Slot | Type | Meaning |
|---|---|---|
| Profile | Model (k-D, usually a 2D sketch) | the section to spin |
| Axis | Subspace, optional | the axis of revolution |

The profile's first coordinate is the distance from the axis (the
radius), its remaining coordinates run along the axis. For the usual 2D
sketch: sketch x is the radius, sketch y is the position along the axis.
Only the x ≥ 0 half of the sketch contributes, so draw the section beside
the axis, not across it.

The axis is a Subspace of rank k−1 in (k+1)-space; for a 2D profile, a
line in 3-space (`subspace_operator` with `kind: line`, or a `cloud_fit`
of kind `line` or `cylinder`). The line's origin is where sketch y = 0
and its direction is where sketch y grows. Unwired, the axis is the world
z axis through the origin and the profile must be 2D: sketch x is radial,
sketch y is world z.

## Output

A model one dimension higher than the profile. Sampling rewrites each
point into (radius, along-axis coordinates) and asks the profile, so the
result is exact wherever the profile is; sample channels pass through
unchanged. Bounds are the profile's chart box swept around the axis.

## Notes

- A profile that crosses the axis (x < 0 parts) is clipped at the axis,
  not mirrored.
- To revolve about an axis found in a scan, wire the `cloud_fit`
  Subspace straight in; its origin sits where the fit put it, so the
  sketch's y = 0 lands there.
- With `path_sketch_operator` the section is one SVG path in metres, for
  example a stepped column: `M 0 0 H 0.026 V 0.08 H 0.015 V 0.18 H 0 Z`.
