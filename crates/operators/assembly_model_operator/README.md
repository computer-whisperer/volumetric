# Assembly Model

One part of an assembly, or all of them, posed as a model: a part on its
own for a per-part audit, or the union at a state other than the
assembly's without re-running the steps before it.

## Inputs

- `Assembly`: from Assemble.
- `Config`: `part`, a part name or `all` (default) for the union.
- `State`: an optional F64Map; empty keeps the assembly's own state.

## Output

- `model`: the posed model.
