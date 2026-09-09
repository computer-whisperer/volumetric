# Cloud Fit

Fits a geometric feature to a point cloud and hands it on as a Subspace,
so a scanned face, edge, hole or shaft becomes something Slice, Extrude,
Revolve, Span and Intersect can build on. The cloud is the nodes of any
FeaMesh: a Point1 cloud from Point Cloud Import, or a mesh's vertices.

RANSAC finds the feature most points agree with, least squares refines
it on those inliers (re-selecting them as the feature moves), and the
second output reports what the Subspace cannot carry.

## Kinds

| `kind` | Feature output | Extra numbers |
|---|---|---|
| `plane` | rank-2 plane, facing as `normal` says | `extent_0`, `extent_1` of the inliers along the two basis vectors |
| `line` | rank-1 line, sign canonical (first component positive) | `extent_0` along the line |
| `point` | rank-0 point: the centroid of the considered points | |
| `sphere` | rank-0 point: the centre | `radius` |
| `cylinder` | rank-1 line: the axis | `radius`, `extent_0` along the axis |

Every fit reports `points` (considered), `inliers`, `rms` and `max`
(inlier distances from the feature) and the `tolerance` used.

A cylinder hypothesis needs a direction. The cloud's `normal` node field
supplies it (two points with normals fix an axis); run Cloud Normals
first if the cloud has none. A seed line fixes the direction instead,
and is refined with the rest.

## Seed

The optional Subspace input steers the search:

- a **point** narrows the fit to the points within `search_radius` of
  it (0 = 10% of the cloud's bounding diagonal), for picking one face
  among many;
- a **line** (for `line`) or **plane** (for `plane`) is the estimate to
  refine: its inliers are taken directly, no search, so a plane you
  already know moves onto the scan;
- a **line** for `cylinder` gives the axis direction (see above).

With a line or plane seed, `search_radius` above 0 further limits the
points to those within that distance of the seed.

## Config

- `kind` (default `plane`).
- `normal` (default `outward`): which way a plane faces. `outward` is
  away from the whole cloud's centroid (outward on a scanned object,
  away from the object for the table it stands on), `inward` the
  opposite, `up` towards +y.
- `tolerance` (default 0 = 0.5% of the considered points' bounding
  diagonal): a point within this distance of the feature is an inlier.
  Set it to the scan's noise level for a tight fit.
- `search_radius` (default 0): see Seed.
- `trials` (default 500): the most RANSAC hypotheses tried; the search
  stops early once the best one is unlikely to be beaten.

The fit is deterministic: the same cloud, seed and config always give
the same feature.
