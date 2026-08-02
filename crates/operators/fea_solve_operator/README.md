# FEA Solve

Compresses an FEA mesh against a rigid implicit body with linear
elasticity (Hooke's law, quasi-static single pose, active-set contact).
The rigid body is sampled where the user placed it — position it already
interpenetrating the mesh, in the fully pressed pose. Contact presses
along the `fixed_boundary` axis, toward the glued face (glue zmin, press
with a body from +z; glue ymin, press from +y; …).

The element formulation follows the mesh's kind:

- **Hex8** (from the FEA grid-mesh operator): uniform-grid solid elements;
  an optional `stiffness_scale` element field scales element stiffness.
- **Bar2** (an explicit strut lattice): 3D frame elements with a circular
  section from the required `radius` element field (metres);
  `stiffness_scale` multiplies a strut's Young's modulus.

## Units

The solve is qualitative unless you make it physical: displacements scale
inversely with `youngs_modulus`, so at the default 1.0 (1 Pa) the deformed
view is shape-only. Set a physical modulus (foam ~1e5–1e7 Pa, green resin
~1e9 Pa) to read displacements and contact forces in real units. Geometry
is metres throughout the pipeline.

## Inputs

1. **Mesh** — FeaMesh (Hex8 or Bar2).
2. **Rigid body** — ModelWASM (must be 3D), placed in the pressed pose.
3. **Config** — CBOR configuration:

## Configuration

- `youngs_modulus` (default 1.0), `poissons_ratio` (default 0.3).
- `fixed_boundary` (default `"zmin"`) — the glued face
  (xmin/xmax/ymin/ymax/zmin/zmax/none).
- `max_contact_iterations` (default 64) — cap on the contact active-set
  sweeps; grazing rims on curved rigid bodies can need a few dozen. If the
  cap is hit, the best iterate is returned.
- `cg_tolerance` (default 1e-8) — relative residual per CG solve; 1e-4 is
  measurably ~3x faster and usually converges to the same answers.
- `preconditioner` (default `"auto"`) — `auto` picks the two-level Schwarz
  solver for large Bar2 frames in the threaded operator build and
  block-Jacobi otherwise; `"schwarz"` forces it (`schwarz_target_nodes`,
  default 128, sizes its subdomains).
- `stress_stiffening_passes` (default 0) — tension-only geometric
  stiffness re-solves for Bar2 frames; 1–2 passes capture most hammocking
  (a taut surface strut carries transverse load like a string).

## Output

The input FeaMesh plus result fields — per-node `displacement` (3),
`contact_force` (3, the interface force map), and `rotation` (3, Bar2
frame meshes only), per-element `strain_energy_density` (energy per unit
element volume: cell volume for Hex8, strut volume for Bar2).
