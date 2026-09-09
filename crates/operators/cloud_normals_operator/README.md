# Cloud Normals

Estimates a unit normal at every node of a point cloud and writes it as
the `normal` node field, replacing one the cloud already has. Each
normal is that of the plane through the node's `neighbours` nearest
points (principal component analysis: the direction of least spread), so
it needs a cloud dense enough that a node's neighbours lie on the same
surface patch, which a scan at a few points per millimetre is.

Cloud Fit needs this field to fit a cylinder without a seed axis, and a
Point Cloud Import that carried normals from its file already has it.

## Orientation

Principal components have no sign, so `orient` picks one:

- `outward` (default): away from the cloud's centroid, right for a scan
  of a convex-ish object seen from outside and consistent enough for
  fitting anywhere;
- `up`: towards +y, the engine's up, for terrain-like clouds and table
  tops;
- `none`: whatever the analysis produced, when only the line of the
  normal matters.

Nodes with fewer than three neighbours, or with collinear ones, get a
zero normal and are counted in a warning.

## Config

- `neighbours` (default 16, at least 3): points used per normal.
  More smooths noise and blurs edges.
- `orient` (default `outward`).
