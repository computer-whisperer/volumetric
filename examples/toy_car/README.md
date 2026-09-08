# Toy car (dogfooding exercise)

A hand-sized toy car (80 x 32 x 37 mm) built entirely from the operator
catalog, as a capability test of the CAD-style construction path:
2D path sketches -> extrude / revolve -> booleans -> offset rounding.
Every sketch is one `path_sketch_operator` step with inline SVG path data
(the exercise's first gap, originally hand-written Lua half-planes).
`build.sh` rebuilds `toy_car.vproj` at the repo root from scratch
(`cargo build --release -p volumetric_cli` first) and runs it.

Coordinate frame (engine convention): +x right = car length, nose at
+x; +y up; +z forward = car width. Wheels turn about z. Ground at y = 0.

## Construction

Body:
- Side sketch (x, y): chassis slab y 6..23 mm with a full-round nose
  (arc r 8.5 at x 31.5), cabin trapezoid y 23..37 with a raked
  windscreen (6,23)->(-2,37) and rear slope (-28,23)->(-24,37).
  Extruded across z via a `subspace` plane at z = -16 mm, height 32.
- Plan sketch (x, z): 80 x 32 rectangle with `round` 6 mm corners.
  Extruded down (-y normal) through the side extrude; intersect gives
  the two-view body.
- Rounding: `offset` -1.5 mm then +1.5 mm (morphological opening) at
  resolution 256 rounds every convex edge r 1.5. Done before any cut so
  the cuts stay crisp.

Cuts (all subtract from the rounded body):
- Wheel wells: one flat-cap cylinder r 10.5 along z, z 9..20, at the
  front-right axle position (x 25, y 9), patterned to all four corners
  (mirror across z = 0, then repeated 50 mm back). Central spine z -9..9
  stays solid.
- Side windows: a two-subpath sketch (x, y) extruded z -25..25, minus
  an inner slab |z| <= 14.8, leaving 1.2 mm-deep recesses on both flanks.
- Windscreen / rear window: axis-aligned prisms rotated about z to the
  face rake (29.7 deg / 164.1 deg) and translated to the face midpoint;
  half sits outside, the inner half recesses 1.2 mm.

Additions (union):
- Axles: a cylinder r 1.5 along z, z -17.5..17.5, at (25, 9), repeated
  at x -25.
- Wheels: a tyre section sketch (r, a) revolved about z at the origin:
  tyre r 9, width 8, r 1.5 shoulder arcs, two tread grooves, a hub
  dish on the outer face with a centre boss. The wheel is translated to
  the front-right hub (25, 9, 14) and patterned like the wells; the
  mirror across z = 0 turns the dish outward on the left side.
- Headlights: a round-cap cylinder r 2.5 along x at the nose, y 17,
  z 9, mirrored across z = 0.

Exports: `body` (rounded, cut body), `wheel` (one wheel at the origin),
`car` (the assembly).

Grille: three slot subpaths sketched in the front view (chart (z, y))
on a plane at x = 45 mm whose basis order gives a -x normal, extruded
9 mm back into the nose: 0.8 mm slots below the headlights.

## Meshing and printing

- Smooth pipeline (`mesh` defaults): watertight, manifold, Euler 2, at
  256^3 (`--base-resolution 16 --max-depth 4`). That is the printable
  STL. Its normals smear across creases (wheel arch, hub dish, grooves).
- `--sharp-edges`: clean creases in renders, but the decimated STL has
  ~11k open edges (sharp + simplify tearing) and even `--no-simplify`
  leaves a few hundred non-manifold edges on the tyre groove crease
  circles. The `wheel` export alone reproduces it (72 non-manifold
  edges at 256^3, count grows with resolution), so it is a sharp-
  pipeline bug, not a sampling limit.
- Render the pretty pictures with `--sharp-edges`; print the smooth one.

## CLI notes from the exercise

- An optional operator input (the revolve axis) is left unwired with
  `--input none` on `project-add-op`.
- Booleans take any number of models before the config (a variadic
  slot; repeat `--input`), so the five body cuts and the four-part
  assembly are one step each instead of chains.
- Symmetric parts are stated once: `pattern_operator` places the wheel
  well, wheel, axle and headlight copies (mirror across the centre plane
  and a linear repeat to the rear axle) as one single-memory step each.
- Operator modules are shared between steps that run the same build
  (`Project::insert_operation` reuses a byte-identical import, and
  loading merges duplicates from older files), so this project is a
  fraction of the size it would be with one operator copy per step.
- `json:` float fields must be float literals; `sketch-raster` is the
  fast way to check a profile before extruding, but it only reaches
  exported assets, so drop `--no-export` on the sketch step while
  iterating.
