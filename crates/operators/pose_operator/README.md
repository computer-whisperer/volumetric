# Pose

Places a model in one step: scales it about a pivot, rotates it about the
same pivot, then moves it. Translate, Rotation and Scale each do one of
those about the origin; Pose does all three about a point of your choice,
so a part built at the origin lands where it belongs without a chain of
steps. The result keeps the input's memory and sample format.

## Blocks

Every block is optional and is the identity when off.

- `pivot { at, px, py, pz }`: the point scaling and rotation happen about.
  `at` is `origin` (default), `center` (the centre of the input model's
  bounds, read when the step runs), or `point` (`px`, `py`, `pz`).
- `scale { sx, sy, sz }`: per-axis factors about the pivot; negative
  factors reflect, zero is an error.
- `rotate { rx_deg, ry_deg, rz_deg }`: Euler angles in degrees about the
  pivot, applied X then Y then Z (the same convention as Rotation and Mesh
  Transform).
- `translate { dx, dy, dz }`: the final move.

A point `p` of the input lands at `R S (p - c) + c + t`.

## Dimensions

Only the spatial prefix of the model transforms: three axes, or two for a
sketch. A 2D sketch poses in-plane: `pz`, `dz` and `sz` are ignored, the
pivot is `(px, py)` or the sketch's bounds centre, and a nonzero `rx_deg`
or `ry_deg` is an error rather than silently dropped.

## Bounds

Pure translations and scalings keep the bounds exact; a rotation encloses
the rotated box, the same as Rotation.

## Example

A windscreen prism built at the origin, raked 30 degrees about z and moved
to the face midpoint:

```sh
volumetric_cli project-add-op -p car.vproj --operator pose_operator \
  --input asset:ws_box \
  --input 'json:{"rotate":{"rz_deg":29.745},"translate":{"dx":0.002,"dy":0.030}}' \
  --output-id windscreen
```
