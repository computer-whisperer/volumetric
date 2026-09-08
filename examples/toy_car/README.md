# Toy car (dogfooding exercise)

A hand-sized toy car (80 x 32 x 37 mm) built entirely from the operator
catalog, as a capability test of the CAD-style construction path:
2D Lua sketches -> extrude / revolve -> booleans -> offset rounding.
`build.sh` rebuilds `toy_car.vproj` at the repo root from scratch
(`cargo build --release -p volumetric_cli` first) and runs it.

Coordinate frame (engine convention): +x right = car length, nose at
+x; +y up; +z forward = car width. Wheels turn about z. Ground at y = 0.

## Construction

Body:
- `side_profile.lua` (x, y): chassis slab y 6..23 mm with a full-round
  nose (disk r 8.5 at x 31.5), cabin trapezoid y 23..37 with a raked
  windscreen (6,23)->(-2,37) and rear slope (-28,23)->(-24,37).
  Extruded across z via a `subspace` plane at z = -16 mm, height 32.
- `plan_profile.lua` (x, z): rounded rectangle 80 x 32, corner r 6.
  Extruded down (-y normal) through the side extrude; intersect gives
  the two-view body.
- Rounding: `offset` -1.5 mm then +1.5 mm (morphological opening) at
  resolution 256 rounds every convex edge r 1.5. Done before any cut so
  the cuts stay crisp.

Cuts (all subtract from the rounded body):
- Wheel wells: four flat-cap cylinders r 10.5 along z, |z| 9..20, at
  the axle positions (x +-25, y 9). Central spine z -9..9 stays solid.
- Side windows: `windows.lua` (x, y) extruded z -25..25, minus an inner
  slab |z| <= 14.8, leaving 1.2 mm-deep recesses on both flanks.
- Windscreen / rear window: axis-aligned prisms rotated about z to the
  face rake (29.7 deg / 164.1 deg) and translated to the face midpoint;
  half sits outside, the inner half recesses 1.2 mm.

Additions (union):
- Axles: cylinders r 1.5 along z, z -17.5..17.5, at (+-25, 9).
- Wheels: `wheel_profile.lua` (r, a) revolved about z at the origin:
  tyre r 9, width 8, shoulder round r 1.5, two tread grooves, a hub
  dish on the outer face with a centre boss. Right wheels are
  translated to z +14; left wheels are `scale` sz = -1 first (mirror)
  so the dish faces out.
- Headlights: round-cap cylinders r 2.5 along x at the nose, y 17,
  z +-9.

Exports: `body` (rounded, cut body), `wheel` (one wheel at the origin),
`car` (the assembly).

Grille: `grille.lua` sketched in the front view (chart (z, y)) on a
plane at x = 45 mm whose basis order gives a -x normal, extruded 9 mm
back into the nose: three 0.8 mm slots below the headlights.

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

- An optional operator input (the revolve axis) has no unwired spelling
  on `project-add-op`; `--input 'data:'` (empty bytes) is what works.
- Each `project-add-op` embeds a fresh copy of the operator module, so a
  55-step project is ~99 MB on disk (the housing examples are the same).
- `json:` float fields must be float literals; `sketch-raster` on a
  Lua sketch step is the fast way to check a profile before extruding.
