# Path Sketch

Fills SVG path data as a 2D sketch: `sample(x, y)` is 1.0 inside the
drawn outline and 0.0 outside, at any resolution. The output composes like
any 2D model — extrude it, revolve it, intersect it with another sketch,
or offset it — and `sketch-raster` previews it from the CLI.

## Path syntax

`path` takes the SVG `d` attribute grammar: `M L H V C S Q T A Z`, absolute
(upper case) and relative (lower case), implicit repetition (`L 1 2 3 4`),
compact number runs (`10-5`), and adjacent arc flags (`A 1 1 0 01 5 5`).
Curves and arcs are flattened to chords at conversion time, so the model
classifies points exactly against straight segments.

Coordinates are metres, y up. Every subpath is filled closed, whether or
not it ends in `Z`. The fill rule is nonzero winding: a subpath wound the
opposite way to the one enclosing it cuts a hole, and subpaths wound the
same way union.

Arcs follow the SVG endpoint form `A rx ry rotation large sweep x y`. In
this y-up frame `sweep=1` turns counterclockwise, which is the opposite of
what the same flag draws on a y-down SVG canvas. Set `flip_y` when pasting
path data authored in an SVG editor: the sketch is mirrored about the x
axis, which restores the on-screen orientation and sweep directions
together.

## Rounding

`round` (metres, default 0) fillets every corner where two straight
segments meet, convex and concave alike. The tangent length is clamped to
half the shorter adjacent edge, so a radius larger than a short edge can
carry degrades to a full round instead of overlapping its neighbour.
Corners involving a curve or arc are left as drawn; author those fillets
as explicit arcs.

## Tolerance

`chord_tolerance` (metres) bounds how far any chord may deviate from its
curve. The default 0 means automatic: one ten-thousandth of the sketch's
bounding-box diagonal, so a 90 mm part flattens to 9 µm and a 1 m part to
0.1 mm.

## Example

The toy car side profile from `examples/toy_car`, a chassis slab with a
full-round nose and a raked cabin:

```
M -0.040 0.006 H 0.0315 A 0.0085 0.0085 0 0 1 0.0315 0.023
H 0.006 L -0.002 0.037 H -0.024 L -0.028 0.023 H -0.040 Z
```
