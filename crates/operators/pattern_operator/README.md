# Pattern

Repeats a model as a set of placed copies in one step: a mirror image, a
row of copies along a direction, a ring of copies about an axis, or any
product of the three. The result is a single model with the input's memory
and sample format; nothing is merged, so a hundred copies cost no more
module memory than one.

## Blocks

The config has three optional blocks. Each block patterns the result of
the blocks before it, in this order, so enabling several multiplies the
instance count (mirror x linear x circular). The original placement is
always instance 0.

- `mirror { axis, offset, keep_original }` reflects across the plane
  `axis = offset` (for a 2D sketch, the line). With `keep_original`
  (default) both the model and its image are kept; without it only the
  image, which makes the step a plain mirror.
- `linear { count, dx, dy, dz }` places `count` copies at every multiple of
  the step `(dx, dy, dz)`, the original included.
- `circular { count, axis, cx, cy, cz, sweep_deg }` places `count` copies
  rotated about the line through `(cx, cy, cz)` parallel to `axis`. A full
  `sweep_deg` of 360 (the default) spaces them `360 / count` apart; any
  other sweep is inclusive of both ends, `sweep / (count - 1)` apart.

At most 1024 instances are allowed; a step with no block enabled passes
the model through.

## Dimensions

Only the spatial prefix of the model transforms (three axes, or two for a
sketch). A 2D sketch can mirror across `x` or `y` and rotate about `z`
only; asking for `z` mirror or an `x`/`y` circular axis is an error, while
a linear `dz` and a circular `cz` are simply ignored.

## Sampling

A point is inside where any instance is inside. Instances are tried in
order and the first hit wins, so a model with typed channels reports the
channels of the first instance that contains the point; where no instance
does, the row is whatever the last instance reported there. Bounds are the
enclosure of every instance's box. Each sample costs one inverse transform
and one input sample per instance until the first hit, so put the densest
block first when the order is free.

## Example

Four wheels from one wheel placed at the front-right hub: mirror across the
car's centre plane `z = 0`, then a second copy of the pair 50 mm back.

```sh
volumetric_cli project-add-op -p car.vproj --operator pattern_operator \
  --input asset:wheel_fr \
  --input 'json:{"mirror":{"axis":"z"},"linear":{"count":2,"dx":-0.05}}' \
  --output-id wheels
```
