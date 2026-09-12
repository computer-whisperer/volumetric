# Boolean

Combines one or more models by union, intersection, or subtraction. The
step takes any number of `Model` inputs followed by the config; the models
are combined in a single flat node rather than a chain, so an assembly of
nine parts is one union step.

## Operations

`op` selects the rule, applied to every wired model in slot order:

- `union` (default): occupied where any model is occupied.
- `intersect`: occupied where every model is occupied.
- `subtract`: the first model minus the union of all the others. Which
  model comes first matters only here.

A single model passes through unchanged under every operation. Unwired
model entries are skipped; the step fails when no model is wired.

## Bounds and dimensions

The result's dimension count is the first model's; every input must agree
(a 2D sketch unions with 2D sketches, a solid with solids). When the first
model states its count as a constant, so does the result, which is what
Pattern, Pose, Extrude and the other wrappers read, so a boolean can sit
anywhere in a chain. Bounds are the
enclosure of every input for `union`, the intersection of every input's
box for `intersect`, and the first model's box for `subtract`.

## Channels

Typed sample channels follow the first model: when it declares a sample
format, the output keeps that format and channel row with channel 0
replaced by the combined occupancy. The other models contribute occupancy
only, so in regions where only a later model is solid the extra channels
hold whatever the first model reports there. A first model without a
declared format yields the plain occupancy-only output regardless of what
the others declare.

## Performance

Each sample evaluates the models in order and stops early: a union stops
at the first occupied model, an intersection at the first empty one, and a
subtraction at the first model that carves the point away. Put the model
most likely to decide the answer first.

Every input keeps its own linear memory inside the merged module, and the
WebAssembly validator allows 100 memories per module. A flat union of
many parts and a chain of unions hit that ceiling at the same total, so
combine very large assemblies from sub-assemblies whose inputs are baked
(an offset or lattice step produces a single-memory model).

## Example

CLI, uniting a body with four wheels then carving two cuts:

```sh
volumetric_cli project-add-op -p car.vproj --operator boolean_operator \
  --input asset:body --input asset:wheel_fl --input asset:wheel_fr \
  --input asset:wheel_rl --input asset:wheel_rr \
  --input 'json:{"op":"union"}' --output-id assembly
volumetric_cli project-add-op -p car.vproj --operator boolean_operator \
  --input asset:assembly --input asset:window_cut --input asset:grille_cut \
  --input 'json:{"op":"subtract"}' --output-id car
```
