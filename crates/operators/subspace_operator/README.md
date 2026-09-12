# Subspace

Builds a Subspace value in 3-space from numbers: a point, a line, a plane
or a full frame. Subspaces are the datums the construction operators
build on (Extrude's plane, Revolve's axis, Slice's chart, Span and
Intersect's operands) and what `cloud_fit` returns from a scan; this
operator is how one is written down by hand.

## Inputs

| Slot | Type | Meaning |
|---|---|---|
| Config | CBOR | `kind`: `point`, `line`, `plane` (default) or `frame` |
| Origin | VecF64(3) | the chart origin |
| Primary | VecF64(3) | first direction (line, plane, frame) |
| Secondary | VecF64(3) | second direction (plane, frame) |

Directions need not be unit or orthogonal: `primary` is normalised, and
`secondary` is orthonormalised against it (Gram–Schmidt), so it only has
to be non-parallel. A `frame` adds the third axis as primary × secondary,
so frames are always right-handed. Unused slots may be left unwired.

## Output

The Subspace: rank 0 (point), 1 (line), 2 (plane) or 3 (frame) in
3-space, with the origin and an orthonormal basis. A plane's normal is
primary × secondary, which is the direction Extrude sweeps along.

## Errors

A zero direction, or two parallel directions where two are needed, is an
error rather than a silently substituted axis.

## Notes

- CLI: `--input 'json:{"kind":"plane"}' --input 'json:[0,0,0.16]'
  --input 'json:[1,0,0]' --input 'json:[0,1,0]'` is the horizontal plane
  at z = 160 mm facing +z; `--input none` for the unused secondary of a
  line.
- To put a sketch on a plane found in a scan, skip this operator and
  wire the `cloud_fit` result in directly; use this one when the datum
  is a design decision, not a measurement.
