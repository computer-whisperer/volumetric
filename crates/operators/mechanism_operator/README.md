# Mechanism

Describes how the parts of an assembly move: a tree of parts joined by
fixed, revolute and prismatic joints, with the ranges and defaults of
each. The result is a Mechanism value that Assemble pairs with the part
models; on its own the viewport draws its joint axes.

## Frames

Parts are authored in the world frame at the rest state, exactly as they
are measured and built, and every axis is written in world coordinates at
rest. A part's pose is the product of its chain's joint motions, root
first, so the joint after a swivel turns with the swivel. There are no
local frames to author, and the rest state is the state as photographed.

## Config

- `parts`: the part names, in the order Assemble's part inputs are wired.
- `joints`: one entry per part (every part hangs from exactly one joint):
  - `name`: the joint's name and, for a moving joint, its state key.
  - `kind`: `fixed` (default), `revolute` (degrees about the axis) or
    `prismatic` (metres along the axis).
  - `parent`: a part name, or `world` (default).
  - `child`: the part the joint attaches.
  - `axis { origin, direction }`: the axis in world coordinates at rest,
    for a moving joint. Or `axis_input: k`: take it from the k-th Subspace
    wired to the Axes block (a line's origin and direction, or a frame's
    origin and first basis vector), so a measured datum such as a fitted
    lift axis drives the joint directly. With every axis inline, leave the
    Axes slot unwired (`--input none` from the CLI).
  - `min`, `max`, `default`: the range and rest value of a moving joint
    (defaults 0). The state form's sliders and a drag's clamp use them.
  - `drive { joint, ratio, offset }`: a coupling. The joint's value is
    `ratio * value(joint) + offset` and it is not a state of its own: a
    synchro-tilt back that follows the seat at half the angle.

## Example

A column that swivels about a fitted lift axis, a seat that tilts about a
pivot on it:

```json
{ "parts": ["column", "seat"],
  "joints": [
    { "name": "swivel", "kind": "revolute", "child": "column",
      "axis_input": 0, "min": -180.0, "max": 180.0 },
    { "name": "tilt", "kind": "revolute", "parent": "column", "child": "seat",
      "axis": { "origin": [0.26, 0.15, 0.45], "direction": [1.0, 0.0, 0.0] },
      "min": -5.0, "max": 20.0 } ] }
```
