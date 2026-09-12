# Cloud Distance

Measures a point cloud against a model: the signed distance from every
node to the model's surface, stored as a node field, with a summary of
how far the cloud sits from the model. This is the audit that says
where a model built from a scan disagrees with the scan, and by how
much: colour the cloud by the field in the viewport or with
`render --color-field node:distance`, and read the summary for the
numbers.

## Inputs

| Slot | Type | Meaning |
|---|---|---|
| Cloud | FeaMesh | the points (any element kind; the nodes are measured) |
| Model | Model | the surface to measure against |
| Config | CBOR | `resolution`, `field`, `band` |

## How

Occupancy is sampled on a regular lattice over the union of the cloud's
and the model's bounds, `resolution` cells along the longest side (default
128, at most 256), padded by two cells. An exact Euclidean distance
transform runs over the occupied cells and over the empty ones, giving
each cell its signed distance to the nearest cell of the other kind,
negative inside. Each node reads the field trilinearly; beyond the lattice
the distance to it is added.

Precision is the cell: the surface lies somewhere within the cell that
changes state, so distances are good to about half a cell plus the
interpolation. At 128 cells over a 0.7 m base that is 5.5 mm cells, 3 mm
distances; raise `resolution` for finer work (256 cells cost eight times
the samples of 128).

## Outputs

1. The cloud with the `field` node field (default `distance`, metres,
   negative inside), replacing one of that name.
2. An F64Map: `count`, `inside` (nodes with negative distance),
   `inside_fraction`, `mean` (signed), `abs_p50`, `abs_p90`, `abs_p99`,
   `max_outside`, `max_inside`, `cell` (the lattice spacing), and the
   fractions `within_band`, `within_2band`, `within_4band` of nodes
   within `band` (default 5 mm), twice and four times it. The fractions
   are the numbers to watch while a model is corrected: percentiles stay
   pinned by whatever is not modelled at all (a lever, the floor), the
   fraction within a band moves with every part that lands.

## Reading it

- A cloud that sits on the model reads within a cell of zero everywhere.
- Positive distances mark scanned surface the model does not reach: a
  part too small, too short, or missing.
- Negative distances mark model volume the scan saw no surface in: a
  part too fat, or in the wrong place (then the true surface shows up
  positive nearby).
- The scan's own noise and unseen undersides show as a few millimetres
  either way; look for the regions, not the scatter.
