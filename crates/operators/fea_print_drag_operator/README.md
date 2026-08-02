# FEA Print Drag

Designs the "spine" a strut lattice needs to survive being swished through
a resin bath during large-format printing. The lattice acts as a sieve
collecting fluid drag over its whole surface; the drag on any one strut is
trivial, but the integrated load must flow through the strut network to the
fixed (build-plate) face, and the struts on that load path carry the summed
tension of everything downstream of them. The design loop grows per-strut
radii — floored at the as-designed radius — until every strut's tensile
fiber stress sits under an allowable, so the reinforcement concentrates
along the dominant load paths and leaves the rest of the lattice untouched.

## The severity dial — read this first

The pass is **qualitative**: for the sizing, only the ratio
`drag_pressure / allowable_stress` matters, and it is the severity dial.
Physically motivated values put it around **1e-5 to 1e-4**: swishing a part
through resin loads its struts with a few tens of pascals of drag pressure,
while green-state (just-cured) resin tolerates a few MPa of tension. At the
defaults (ratio = 1) the demanded spine is thousands of times beyond any
lattice — every strut pins at the growth cap and the result is a uniform
blob, not a design.

The emitted **`displacement` field is a separate dial**: it scales with
`drag_pressure / youngs_modulus` on the as-designed radii, so at the
default `youngs_modulus = 1.0` (1 Pa — a qualitative placeholder) the
deflections can dwarf the part while the sizing is perfectly healthy. Set
`youngs_modulus` to a physical modulus (~1e9 Pa for green resin) if you
want to read the deformed view literally; the radius/utilization design
does not depend on it.

## Model

- The load is `drag_pressure` per unit projected frontal area
  (`2·r·L·sin(θ)` per strut against the flow direction), lumped half to
  each end node. No fixed-end moments: a strut's mid-span bending under its
  own drag is not the failure mode — print sequencing already tolerates
  local flexing — the accumulated load-path tension is.
- One signed load direction, default pulling the part away from the fixed
  face: the pull-out stroke puts the load path in tension, which is what
  tears parts; the push-back stroke is compression, where the lattice's
  flexibility is tolerated.
- The failure metric is the max tensile fiber stress `N/A + M·r/I` over a
  strut's ends (torsional shear ignored). Stresses come from internal
  forces, so a `stiffness_scale` field from a seating design pass
  participates in load distribution without corrupting the stress numbers.

## Inputs

1. **Mesh** — FeaMesh: a Bar2 strut lattice with a scalar `radius` element
   field (metres). An existing `stiffness_scale` element field participates
   in load distribution unchanged.
2. **Config** — CBOR configuration, all fields optional:

## Configuration

- `drag_pressure` (default 1.0) — drag force per unit projected frontal
  area, in Pa when working physically. Typical resin-bath value: tens of
  Pa. Only its ratio to `allowable_stress` affects the sizing.
- `allowable_stress` (default 1.0) — tensile fiber stress ceiling, in Pa
  when working physically. Green-state resin: a few MPa (1e6–1e7).
- `youngs_modulus` (default 1.0) — sets the displacement scale only (the
  sizing is stiffness-ratio driven). Use ~1e9 Pa (green resin) to make the
  emitted `displacement` field physically readable.
- `poissons_ratio` (default 0.3).
- `fixed_boundary` (default `"zmin"`) — the build-plate face
  (xmin/xmax/ymin/ymax/zmin/zmax); the drag load needs a reaction point, so
  `"none"` is rejected.
- `load_direction_x/y/z` (default all 0.0) — flow direction override; all
  zeros means auto: away from the fixed face (the tension stroke). Need not
  be unit length.
- `max_iterations` (default 40) — cap on sizing iterations (one forward
  solve each). The loop usually exits far earlier: on acceptance, or when
  the peak utilization stops improving (plateau).
- `tolerance` (default 0.02) — accept when max utilization ≤ 1 + tolerance.
- `exponent` (default 0.5) — radius update damping (`r ← r·u^exponent`).
- `max_radius_scale` (default 10.0) — growth cap as a multiple of each
  strut's original radius. Struts pinned here while over-utilized mean the
  lattice topology cannot carry the demanded load.
- `cg_tolerance` (default 1e-8) — relative CG residual for the returned
  fields. Intermediate sizing solves run at a loosened tolerance
  automatically (they only rank stresses against the acceptance band).
- `preconditioner` (default `"auto"`) — `auto` picks the two-level Schwarz
  solver for large frames in the threaded build and block-Jacobi
  otherwise; `"schwarz"` forces it (`schwarz_target_nodes`, default 128,
  sizes its subdomains).
- `stress_stiffening_passes` (default 0) — tension-only geometric-stiffness
  re-solves; the pull stroke tensions the load path, so 1–2 passes capture
  the taut-string effect.

## Output

The input FeaMesh with the designed per-element `radius` (replacing the
input design, never below it), plus element fields `utilization` (tensile
fiber stress over the allowable — over-unity marks where the spine falls
short), `axial_force` (tension positive — the load-path visualization),
`strain_energy_density`, and node fields `displacement`, `rotation`, and
`drag_force` (the applied load).

## Warnings

The result is best-effort: when sizing stops short of acceptance, the best
iterate ships with an advisory naming the failure mode —

- **saturation**: struts pinned at `max_radius_scale` while over-utilized.
  The spine does not fit this lattice; lower the severity ratio or rework
  the upstream geometry. More iterations will not help.
- **plateau**: the peak utilization quit improving just above acceptance.
  The design is effectively complete; accept it or loosen `tolerance`.
- **iteration cap**: sizing was still improving; raise `max_iterations`.

A second advisory flags a displacement field that exceeds the part size —
see the severity-dial section above.
