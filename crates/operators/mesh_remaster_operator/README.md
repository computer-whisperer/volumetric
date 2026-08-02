# Mesh Remaster

Reworks a strut lattice against composable *requirements*, solved jointly
in one operator so their fixes cannot silently undo each other. Each
requirement is an optional config block; at least one must be present.

## Requirements

- **`surface`** — fold the parts of the network that protrude outside a
  model down onto that model's surface (the drape pass). The outside
  segments become a surface-conforming net (the "skin"), so a trimmed
  Voronoi lattice ends in a smooth printable face instead of open cell
  rims — without adding material the way a separate conforming surface
  lattice would. Draping relocates the outer cells' own struts and the net
  inherits the skeleton's degree-3 vertices: a polygonal, sub-isostatic
  net that deforms by strut bending, far softer in-plane than a
  triangulated skin. `skin_radius_factor` tunes what stiffness remains
  (bending scales as radius⁴) and the `skin` element flag hands the net to
  downstream optimization.
- **`connectivity`** — no floating fragments (they make FEA singular and
  fall off prints). Fix `"reconnect"` re-drapes the cheapest arcs the
  surface pass dropped until everything is one component (a spanning
  forest over components, shortest arcs first), then synthesizes direct
  ties for pieces with no dropped arc to reuse (up to `max_new_strut` ×
  the median strut length); whatever nothing can reach is pruned. Fix
  `"prune"` keeps the largest component only. Reconnection struts are
  flagged in a `tie` element field.
- **`support`** — printable along the build axis on a resin printer:
  overhangs are fine, hooks toward the bed are not. A piece fails exactly
  where it appears in a slice unattached to already-cured material; on the
  strut graph that is sub-level-set connectivity — ascend the build axis
  and every node must connect to the bed through nodes at or below its own
  height (`max_descent` degrees of per-strut slack models in-slice
  cohesion; the slack deliberately does not compound across struts). Fix
  `"raise"` projects the descent out of invalid struts — unsupported nodes
  rise straight up (x/y preserved) by the minimum that makes every support
  path monotone, a hanging hook flattening into a fan; the raise distance
  lands in a `raise` node field, and struts the raise collapses (a
  vertical hook lands exactly on its supporter) weld into the joint. Fix
  `"drop"` removes unsupported struts instead — transitively, like the
  voxel island remover, but exact and graph-aware.

Note that `max_descent: 0.0` (the default) is the strictest setting: *no*
descent is tolerated anywhere, so every downward-facing region flattens
into horizontal plateaus. A physical overhang angle (30–45°) preserves the
lattice shape far better; use 0 only when the print process truly cannot
bridge at all.

## Pass order

Surface first (it decides which outside arcs survive and hands the dropped
ones to reconnection), connectivity second, support last. Raising only
moves nodes (plus a weld of what it collapses), so it can break neither of
the first two — where it pulls skin off the surface, the printer is
overruling cosmetics. The one destructive interaction — `support.fix:
"drop"` can split a component by removing a bridge — is closed by
re-running connectivity once.

## Inputs

1. **Mesh** — FeaMesh (Bar2, or Point1 for surface-only).
2. **Surface** — ModelWASM (must be 3D) to drape onto; only required when
   the `surface` block is present. The model is a binary occupancy oracle,
   so projection estimates a direction from a signed stencil of occupancy
   samples, marches to bracket the surface, and bisects — batched, one
   host call per round.
3. **Config** — CBOR configuration, every block optional, at least one:

## Configuration

- `surface.outside` (default `"project"`) — `"project"` drapes struts with
  both nodes outside onto the surface; `"drop"` removes them (the removed
  arcs stay available to reconnection).
- `surface.skin_radius_factor` (default 1.0) — multiplies the `radius`
  field on skin struts.
- `surface.chord_tolerance` (default 0.0 = each strut's own radius) —
  draped chords subdivide until they sag off the surface by less than
  this (metres).
- `surface.inset_factor` (default 0.0) — sink skin nodes below the surface
  by this many strut radii.
- `surface.max_distance` (default 0.0 = 4 × the median strut length) —
  nodes farther than this from the surface drop with their struts.
- `surface.weld_factor` (default 1.0) — welds struts shorter than
  `weld_factor × radius`; 0 disables.
- `connectivity.fix` (default `"reconnect"`), `connectivity.max_new_strut`
  (default 1.5 × the median strut length; 0 never synthesizes).
- `support.axis` (default `"auto"` = z), `support.extreme` (default
  `"min"` — which end of the axis the bed is on).
- `support.max_descent` (default 0.0, degrees; see note above).
- `support.bed_tolerance` (default 0.0 = 1e-4 × the axis extent) — nodes
  this close to the extreme seed as bed-supported.
- `support.fix` (default `"raise"`).

## Output

CBOR-encoded FeaMesh. The surface pass adds a scalar `skin` element field
(1.0 on draped elements), connectivity adds `tie` (1.0 on reconnection
struts), and support raising adds a `raise` node field (the distance each
node rose).

Point1 clouds accept the `surface` requirement only; Hex8 meshes are
rejected — volume elements cannot fold. Node positions are always 3D.
