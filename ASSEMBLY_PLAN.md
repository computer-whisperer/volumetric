# Articulated assemblies — design

Status: RATIFIED 2026-09-12 (user: "agreed, let's ratify and implement").
Ground truth for the arc. Open questions resolved at ratification: the
assemble step always emits the union beside the Assembly (one step fewer
in every script; the rebuild is a wrapper rewrite); joint axes come from
variadic Subspace slots named by index in the config, inline axes allowed.

## What the system is, as it bears on this (survey 2026-09-12)

- A project is imports + a timeline of operator steps + exports. Values are
  self-contained blobs with a type hint. Explicit-data values (TriMesh,
  Subspace, ViewSet, Splat, F64Map) each live in `volumetric_abi` with
  encode/decode, and each got a preview builder, CLI surface and Python
  class when it landed. That is the recipe a new value follows.
- The DAG is the grouping mechanism already: a variadic step over N models
  is how `boolean` groups parts. "Model group" = a step with variadic
  model inputs, not a new project construct.
- Rigid/affine transforms are wasm rewrites (`model_wrap_core::Affine`,
  query points get the inverse, bounds the map) baked at build time.
  `pose_operator` takes CBOR config only; nothing routes an F64Map into a
  transform. A state vector cannot drive a pose today except inside a
  WGSL script through `override` parameters, which constant-fold: every
  change recompiles the model and re-meshes the whole thing.
- `Subspace` already models axes and frames (rank 1 = a line with origin
  and direction; rank 3 = a rigid frame), comes out of measured datums
  (`cloud_fit`, `subspace_operator`) and has a gizmo. A joint axis is a
  rank-1 Subspace.
- Preview: one entity per asset. ASN2 meshes are cached by model content
  hash plus recipe (`mesh_cache`). Retained GPU meshes bake their
  transform into the vertices at upload (`GpuMesh::new(device, mesh,
  transform)`); immediate meshes carry a transform per submission. There
  is no picking or drag machinery beyond the camera and panel resize.
- The WGSL dialect lowers no matrices (`restrict.rs`) and routes scalar
  f64 overrides only.
- The GUI re-runs the project after an edit (`mark_project_dirty` with
  auto-rebuild); the step cache confines the work to steps whose inputs
  changed. A state edit therefore costs the assemble step plus whatever
  the preview must rebuild.
- Concrete need: the chair base has a swivel about the lift axis, the
  gas lift (prismatic), the tilt on the mechanism's pivot, and five
  casters each with a swivel and a roll: twelve states. The toy car's
  wheels roll. All revolute or prismatic; the chair's synchro-tilt (back
  and seat at a fixed ratio) is a coupling, not a new joint kind. No
  closed chain anywhere in sight.

## Decision: a joint tree as data, not a mobility module

Two candidates were weighed.

**Mobility module (a new wasm role: `pose(state) -> transforms`,
`jacobian(state)`)**. For: arbitrary kinematics as code, in the
model-as-code idiom; derivatives supplied by the code. Against: the pose
is evaluated once per state, never per sample, so nothing is gained from
it being wasm; an operator cannot run a wasm input, so the assemble step
would need a new host import (`input_mechanism_pose`) serviced by the
native, probe and web hosts, and the preview, GUI and Python would need
a third executor role; the only compiler we have (WGSL) lacks matrices;
and nothing on the roadmap needs kinematics a tree with couplings cannot
express.

**Joint tree as an explicit value** evaluated by shared Rust
(`volumetric_abi::mechanism`, compiled into the operator and native into
the preview, GUI, CLI and Python alike, the way `cv_core` and
`subspace` are). For: no new executor role or host import; analytic
Jacobians in closed form; the GUI can draw joints as gizmos and build
the state form from the joint ranges; axes come straight from measured
Subspaces. Against: a closed set of joint kinds.

CHOSEN: the joint tree. REJECTED for now: the mobility module. Reopen
condition: a mechanism a tree with linear couplings cannot express (a
closed chain, a cam). If that day comes, `Mechanism` gains a `Module`
variant and everything downstream, which only calls `pose` and
`velocity`, stays.

## Conventions

- Parts are authored in the world frame at the rest state, as they are
  measured and built today; joint axes are given in world coordinates at
  rest. The pose of a part is the product of its chain's joint motions,
  root first: `T_p(q) = M_1(q_1) · … · M_k(q_k)`, each `M_i` the rigid
  motion about joint i's rest axis. No local frames to author, and the
  rest state is the state as photographed.
- The state is an `F64Map` keyed by joint name. Missing keys take the
  joint's default; unknown keys are an error. Coupled joints carry
  `drive: { joint, ratio, offset }` and are not states.
- Joint kinds: `fixed`, `revolute` (angle in degrees about an axis),
  `prismatic` (distance along an axis). A cylindrical joint is two joints
  on one axis. A revolute joint marked `continuous` turns without limit:
  its range is ignored, any angle is a state, and a drag wraps the angle
  into (-180, 180] (a typed value is kept as typed). Chosen over a
  sin/cos pair state, which would only move the seam into a unit-circle
  constraint the solver, the form and the config would all carry. Each joint: `name`, `parent` (a part or `world`), `child`,
  `axis` (`{ origin, direction }` inline, or a Subspace input by index),
  `min`, `max`, `default`, optional `drive`. The tree must be a tree with
  every part reached once.

## Values (`volumetric_abi::mechanism`)

- `Mechanism`: `{ parts: [name], joints: [Joint] }`, CBOR,
  `AssetTypeHint::Mechanism`. Validation: names unique, tree, finite
  axes, direction normalised, ranges contain defaults.
- `Assembly`: `{ mechanism, parts: [{ name, model: bytes }], state }`,
  CBOR, `AssetTypeHint::Assembly`. Self-contained like ViewSet (which
  embeds pictures): parts travel with it. Part bytes hash to the same
  mesh-cache key whatever the state, so re-posing never re-meshes.
- Kinematics: `pose(&Mechanism, &F64Map) -> Vec<Rigid>` (one 3x4 per
  part, world <- part), `velocity(&Mechanism, &F64Map, part, point) ->
  Vec<[f64; 3]>` (the point's world velocity per unit rate of each
  state: `ω × (x − o)` for a revolute joint on the chain, `d` for a
  prismatic one, zero otherwise; couplings by the chain rule). That is
  the Jacobian a drag needs, and the matrix Jacobian is derivable from
  it. `parameter_specs(&Mechanism) -> Vec<ParameterSpec>` gives the
  state form through the existing `annotations::schema_cddl`.
- `Rigid` stays in the ABI (`[[f64; 4]; 3]`); the operator converts to
  `model_wrap_core::Affine` (linear + offset) at the boundary.

## Operators

- `mechanism_operator`: config (the joint list above) plus variadic
  optional Subspace inputs for axes named by index. Output: Mechanism.
- `assemble_operator` (the model group): inputs Mechanism, variadic
  ModelWASM parts in the mechanism's part order, optional F64Map state.
  Outputs: 0 `assembly` (Assembly), 1 `model` (the posed union: each
  part wrapped by its pose through `Wrapper::apply_affine`, merged by the
  boolean glue). Both are cheap: no sampling at build time.
- `assembly_model_operator`: Assembly + config `{ part: name | "all",
  state?: F64Map override }` → ModelWASM. Posed single parts for per-part
  audits; the union at another state without re-running the DAG.
- The unposed part is the input asset already, so printing needs
  nothing new.

## Preview, GUI, CLI, Python

- Preview (`volumetric_preview`): `build_assembly_preview` meshes each
  part through the ASN2 path (mesh cache hits on the part's bytes) and
  emits one mesh per part with its pose as the `Mat4`. Renderer change:
  retained meshes keep their vertices in part space and take the
  transform at submission (`submit_retained_mesh_with(mesh, transform)`),
  so a state change re-uploads nothing. Joint axes drawn per frame like
  the subspace gizmo, at their posed positions.
- GUI: the assemble step's F64Map slot shows the state form built from
  the mechanism's ranges (sliders with min/max/default, the same inline
  F64Map form scripts get). Editing re-runs; the step cache leaves the
  parts alone; the preview re-poses. Drag (P3): pick the part under the
  cursor by a CPU ray test against its preview mesh, take the grabbed
  point, and per mouse move solve damped least squares
  `J Δq ≈ Δx` with `J` from `velocity`, clamp to ranges, write the state
  into the step's inline F64Map, and let the ordinary edit path re-run.
- CLI: `op --operator mechanism_operator / assemble_operator` with json
  configs, nothing bespoke; `render` draws Assembly assets through the
  shared preview; `assets`/`info` describe them.
- Python: `Mechanism` (pose, velocity, parts, joints) and `Assembly`
  (mechanism, parts, state) wrapping the ABI one to one; `render`
  already goes through `volumetric_render`.

## Phases

- P1 (value + operators + preview): `volumetric_abi::mechanism` with
  tests (chain products against hand-computed poses; `velocity` against
  finite differences); the three operators with READMEs; preview of an
  Assembly with per-part transforms and the renderer's submit-time
  transform; CLI `op` path works by construction; README. Dogfood: the
  chair base gets its twelve states in `base.sh` and a render at a
  tilted, swivelled state through a surveyed view.
- P2 (GUI): state form from the mechanism; joint gizmos; Assembly in
  the artifact browser; Python classes.
- P3 (drag): pick, grab, Jacobian solve, clamp, write-back.

## P1 as built (2026-09-12)

- `volumetric_abi::mechanism`: `Mechanism` (parts, joints: `Joint { name,
  kind, parent, child, axis, min, max, default, drive }`), `Assembly`
  (mechanism, parts with model bytes, state), `Rigid` (3x3 + offset,
  `rotation_about`, `then`, `to_cols_array_f32`), `pose`, `velocity`
  (checked against central differences to 1e-6 for every part and state
  of a swivel + lift + tilt + driven-back chain), `joint_values`,
  `state_keys`, `default_state`, `parameter_specs`, `chain`, validation
  (tree, unique names, degenerate axes, ranges, drives). Schema 1. Type
  hints `Mechanism` and `Assembly` through the host, CLI (`vmech`,
  `vasm`, `project-add-asset --type`), Python kind names, GUI slot kinds,
  and the renderable lists.
- `model_merge_core::combine` (moved out of `boolean_operator`, which is
  now a thin caller): `Combine`, `combine_models`.
- `assembly_core`: `posed_model`, `posed_parts`, `posed_part`,
  `posed_union` (wrap by `Affine` from the pose, union by the glue).
- Operators (category Assembly): `mechanism_operator` (config + variadic
  Subspace axes by `axis_input`; the Axes slot is `none` when every axis
  is inline), `assemble_operator` (Mechanism, variadic parts, F64Map
  state → `assembly` + `model`), `assembly_model_operator` (Assembly +
  `{ part }` + optional state → model). READMEs; bundled; in
  `build-wasm`; README tables; ABI.md slot lists.
- Preview: `PreviewPlan::Assembly { mesh }` and `PreviewPlan::Mechanism`;
  `PendingMesh` now carries one `MeshJob` per part (the web host loops its
  remote meshing over them); `assembly.rs` meshes each part by its own
  bytes (mesh-cache key), places it under its pose tinted by part name,
  posts the joints' axes as lines (orange revolute, blue prismatic) sized
  to the scene, and unions the posed bounds; a Mechanism alone draws its
  axes at rest. GUI: `OutputKind::{Assembly, Mechanism}`,
  `OutputRender::Assembly { mode, resolution, wireframe }` with the
  mode/resolution rows shared with models.
- Deferred to P2: the renderer still bakes a retained mesh's transform at
  upload, so a state change re-meshes nothing but re-uploads the parts;
  submit-time transforms come with the live state form. The GUI's F64Map
  slot on an assemble step is the plain inline map until the mechanism's
  `parameter_specs` feed it.
- Dogfood: `examples/chair/base.sh` builds the base as thirteen parts
  posed into the survey frame (arms, hub and tube, piston, mechanism,
  five stems, five wheels) and a mechanism of twelve states (lift
  prismatic −20..80 mm assumed, swivel, and a swivel and a roll per
  caster); exports `chair` (Assembly, 7.3 MB) and `chair_model` (the old
  `base_world`; verify.sh and audit.sh renamed). Rendered at rest with
  the axes, at a moved state through `assembly_model`, and through
  DSC00742 as an edge overlay (the parts' world poses reproduce the
  monolithic pose).
- Tests: `tests/assembly.rs` (mechanism inline/routed axes and errors;
  assemble poses and unions, rest state, range and count errors;
  assembly_model part and state override), the ABI unit tests, a preview
  test for posed joint axes; `operator_metadata` sees the three
  operators.

## P2 as built (2026-09-12)

- Renderer: a retained mesh keeps its vertices as given and takes its
  transform per draw (an instance vertex buffer of model matrices, slot 0
  the identity for the immediate soup; normals take the rotation, so
  rigid or uniform-scale poses only). `RetainedScene.meshes` pairs each
  mesh with its transform; `submit_retained_mesh(mesh, transform)`;
  `create_retained_mesh(device, &MeshData)`.
- Preview: `PreviewEntity.mesh_keys`, a stable identity per mesh (an
  assembly part's mesh-cache key folded with its name). The session keeps
  `part_meshes` by key across entity revisions and only re-submits a
  re-posed part under its new transform; a state change re-meshes and
  re-uploads nothing. The joint axes (and the wireframe edges) are still
  rebuilt per state; both are cheap.
- GUI: an assemble step's inline F64Map slot shows a form built from the
  wired mechanism's `parameter_specs` (one ranged number per state, the
  range in the label, the same form scripts get), once the mechanism is
  built or imported; a run that produces the mechanism gives an open
  editor its form without touching an existing form's buffers. Sliders
  were not added: the config form's numeric fields are text inputs
  everywhere, and a slider widget is a form-wide feature for its own arc.
- Python: `Mechanism` (load/decode/save/encode, parts, joints,
  state_keys, default_state, ranges, pose(state) → (n,3,4),
  velocity(part, point, state) → (k,3)) and `Assembly` (mechanism, parts,
  part_model, state, poses); `Asset.mechanism()` / `.assembly()`;
  `test_assembly.py` builds the two-sphere assembly through the operators
  and checks the kinematics.
- Not done: joint gizmos regenerated per frame (they are sized to the
  assembly's own bounds and drawn as retained lines; nothing needs the
  per-frame form yet).

## P3 as built (2026-09-12)

- `Mechanism::pull(state, part, local, target, iterations)`: the state
  that brings a point of a part (in the part's rest coordinates) nearest a
  world target, Levenberg-Marquardt on `velocity` with Marquardt's
  per-joint diagonal damping (a shared scalar damping crushed the
  per-degree revolute rows under the per-metre prismatic ones), every step
  clamped to the ranges. `clamp_state`. Python `Mechanism.pull`.
- Preview: `PreviewEntity.articulated` (`Articulated { assembly,
  part_of_mesh }`, `mesh_transforms(state)`), `joint_axis_lines`,
  `joint_axis_half`, `joint_axis_style` exported.
- Session: the primary button pressed over a part of a resident
  assembly, when the control scheme leaves that button to the scene
  (every scheme but Maya's Alt-drag), starts a part drag: the last
  frame's camera gives the pointer ray, the nearest triangle hit (all
  resident assemblies, at their current, possibly overridden, poses)
  names the part and the grabbed point, and each pointer move pulls the
  grabbed point to the pointer on the plane of the grab (same NDC depth)
  with six solver steps, drawing the parts under the solved poses and the
  axes as immediate lines at once. Release hands the state to the app;
  the override keeps drawing the dragged pose until the rebuilt entity
  (a new revision) lands.
- App: `apply_assembly_state` writes the state into the inline F64Map of
  the step producing the output (a routed state is refused with a status
  line), refreshes an open editor's form, and marks the project dirty so
  the ordinary auto-rebuild re-runs the assemble step. Viewport hint:
  "drag a part to pose it".
- Tests: `pull` (reaches, clamps, ignores a point on the axis, refuses a
  non-part); the pointer ray and the mesh hit against an orthographic
  camera; the dragged state landing in the assemble step's slot with the
  form refreshed and the project dirty.
- Not done: per-part resolution scaling (each part still meshes at its
  own grid over its own bounds; fine at the default 64, several times the
  fused model's cost at 256).

## Continuous joints (2026-09-12, b7ac290)

`Joint.continuous` (revolute only; validated), `Joint::has_range`,
`Joint::wrap_degrees`; validation, `joint_values`, `parameter_specs` and
`clamp_state` honour it; the mechanism operator's config and README carry
it; the chair's swivel, caster swivels and rolls are continuous. Test:
any angle accepted, no form bounds, a pull from 170° to a target at 200°
lands at −160° instead of stopping at 180°.

## Ledger

- 2026-09-12: ratified; P1 landed 203fbca; P2 landed 79367af; P3 landed 4f78e4d.
- 2026-09-12: the viewport never requested an Assembly (its request
  filter kept its own kind list): fixed f5a53ad, the three copies of the
  list collapsed onto `runtime_asset_is_renderable`, pinned by a test.
  Found by the user; the headless render had hidden it.
