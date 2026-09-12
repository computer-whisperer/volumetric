# Assemble

The model group: a Mechanism, one model per part and a state make an
Assembly, and the parts posed at that state make one model.

## Inputs

- `Mechanism`: from the Mechanism operator.
- `Parts`: one model per part, in the mechanism's part order, each
  authored in the world frame at the rest state.
- `State`: an F64Map keyed by joint name (degrees for a revolute joint,
  metres for a prismatic one). Missing joints take their defaults; a
  value outside a joint's range is an error. Leave it empty for the rest
  state.

## Outputs

- `assembly`: the Assembly value. The viewport meshes each part once and
  draws it under its pose with the joints' axes, so changing the state
  re-meshes nothing; Assembly Model extracts posed parts from it.
- `model`: every part posed and unioned, a plain model for booleans,
  audits against a cloud, renders through a photograph, and export.

The parts themselves are unchanged by the assembly: for printing, export
the part inputs.
