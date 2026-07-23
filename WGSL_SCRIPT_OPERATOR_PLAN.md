# WGSL Script Operator — Design

Status: P1 + P2 LANDED 2026-07-22 — operator + template + tests +
example ports green; tray port meshes via CLI in ~1s (16k triangles,
bounding box exact); UI editor generalized (ScriptForm, language-tagged)
with WGSL parameter forms, reset-to-template, and catalog entry. P3
(extensions) remains; in-editor span-highlighted diagnostics are open
polish (naga's annotated text errors already flow through report_error).
Spike evidence reproduced below so this document stands alone.

P1 delivery notes (what shipped vs the plan):
- `crates/wgsl_model_template` (libm kernels + IO buffer) and
  `crates/operators/wgsl_script_operator` (naga → restrict → walrus).
  Template binary committed under the operator's `template/`; both crates
  in the `build-wasm` alias.
- Static array frames sit above the template's exported `__heap_base`;
  read-only const arrays (module consts and function-local) bake as data
  segments; mutable local arrays zero/const-init per call. Value-position
  const-array indexing loads from segments (wgsl-in does NOT spill const
  arrays to vars the way glsl-in did in the spike).
- Multi-selector switch cases arrive as empty-body fall-through cases and
  are grouped; real fallthrough stays rejected.
- WGSL abstract-float gotcha worth documenting for authors: an
  unannotated `let x = 0.012;` concretizes to f32 — annotate module-alias
  `float` (f64) on `let`/`var` bindings with literal initializers.
- `volumetric_abi::annotations` now hosts the shared @param option
  grammar; `lua_parameters` delegates to it, `wgsl_parameters` is the
  twin. `WgslSource` metadata input + `AssetTypeHint::WgslSource` +
  CLI `--type wgsl` (and `.wgsl` extension default) landed with minimal
  UI arms (full editor wiring is P2).
- Tests: 19 wasmtime execution tests including a probe harness checking
  every transcendental against Rust std at 1e-12, WGSL integer div/mod
  semantics (x/0==x), per-call array reset, and geometry probes on both
  example ports (`examples/*.wgsl`).

## Decision record

**Ratified direction:** a new `wgsl_script_operator` compiles a restricted
WGSL module into a model-ABI wasm module, replacing the planned Lua
overhaul as the scripting path for models. The frontend is naga 30's
`wgsl-in` (already in the workspace via wgpu 30); the backend is our own
naga-IR → walrus lowering; f64 math kernels come from a Rust/libm template
crate following the existing `*_model_template` pattern.

**Ruling 2026-07-22 (user): `scene` returns `bool`.** The entry point is
a presence predicate, not a distance function. Rationale: the occupancy
model is the primary model form across the whole system (ABI.md: models
are *not* SDFs; magnitude carries no geometric meaning), and the
scripting dialect should reinforce that. Distance, density, and FEA
properties are case-specific *side channels* layered over the base
presence protocol — in this dialect, future per-channel optional
functions (e.g. `fn signed_distance(p) -> f64`) — never the primary
return value. SDF-corpus code ports by appending `<= 0.0`.

**Why WGSL** (spike-verified, naga 30.0.0):

- Full f64 SDF scene — aliases, unsuffixed literals, `override`, const
  arrays with runtime indexing, swizzles, for loops, f64 trig — parses
  and validates under `Capabilities::FLOAT64`.
- Literals and const-folding are exact f64 (`0.1` → `F64(0.1)` bit-exact;
  `0.1 + 0.2` folds at f64). WGSL abstract-float literals are f64-backed.
- Every builtin we need validates on f64: sin, cos, tan, atan2, exp, log,
  pow, sqrt, floor, clamp, mix, smoothstep, length, dot, normalize,
  cross, `%`, division.
- `override` is a native routed-parameter mechanism (name + typed default
  in the IR arena; we substitute values during lowering).
- Recursion is rejected at parse ("declaration of `a` is cyclic") — the
  no-recursion guarantee is free.
- naga with `wgsl-in` compiles clean for `wasm32-unknown-unknown`, so the
  compiler lives inside the operator like full_moon does today.
- `wgsl-in` is naga's first-class frontend; no deprecation risk.

**REJECTED — GLSL via naga `glsl-in`:** every float literal is stored
`Literal::F32` (even initializing a `double`: confirmed bit-identical to
`0.1f32 as f64`); `lf` literals die in the preprocessor
(`NotSupported64BitLiteral`); recovering precision needs a span-reparse
hack that cannot reach inside naga's f32-folded const expressions; GLSL
spec has no double trig (`sin(double)` → "Unknown function"), forcing a
"float secretly means f64" dialect fiction; glsl-in is second-tier
upstream with periodic deprecation debate. Corpus familiarity was its
only advantage; porting shadertoy-style SDF code to WGSL is mechanical.

**REJECTED — Lua overhaul (two-stage interpreter + compiled kernel):**
the design was subsetting a dynamic language down to a typed numeric
kernel; WGSL *is* that kernel language natively, with typing, validation,
and vector types supplied by naga instead of hand-rolled. The Lua
operator stays frozen for existing projects (housing_d et al). Known Lua
traps for the record: `^`/`math.pow` are catastrophically wrong
(`2^2 == 2` — a 1-term exp approximation), Taylor trig is unguarded
outside |x| < π, and chained `and`/`or` re-evaluate their lhs twice
(exponential re-execution in chains). Decide separately whether to fix
or tombstone these.

## Crates and files

- `crates/operators/wgsl_script_operator/` — the operator.
  - `src/lib.rs` (or split: `restrict.rs`, `lower.rs`) — naga parse →
    validate → restriction pass → walrus lowering → model wasm.
  - `template/wgsl_model_template.wasm` — committed template binary,
    regenerated manually (doc comment carries the command), same as
    `mesh_to_model_operator`.
- `crates/wgsl_model_template/` — Rust → wasm32-unknown-unknown template:
  - `#[no_mangle]` f64 math kernels wrapping `libm`: sin, cos, tan, asin,
    acos, atan, atan2, exp, exp2, log, log2, pow, sinh, cosh, tanh
    (~15 exports, prefixed `wgsl_`).
  - Static IO buffer (8 f64s) + exported `get_io_ptr`.
  - Rust's `__stack_pointer` global comes along for free and backs
    function-local array frames later.
- `volumetric_abi`:
  - `OperatorMetadataInput::WgslSource(String)` + `AssetTypeHint` entry.
  - Generalize `lua_parameters` into a shared annotation-parameter core
    with a comment-prefix parameter (`--` vs `//`); `lua_parameters`
    keeps its API as a thin wrapper, `wgsl_parameters` is the twin.
- `.cargo/config.toml`: add both crates to the `build-wasm` alias.
- `examples/`: WGSL ports of `fidget_spinner` and `raspberry_pi_4_tray`.

## Source conventions (the dialect)

A script is a library-style WGSL module: no entry points, no bindings.

Required functions:

```wgsl
fn scene(p: vec3<f64>) -> bool  // presence predicate: true = inside
fn bounds_min() -> vec3<f64>
fn bounds_max() -> vec3<f64>
```

- `scene` returns `bool` (RULED, see decision record): the base protocol
  is presence. Boolean CSG (`&&`, `||`, `!`, bool-returning helpers)
  spike-verified: parses and validates; naga lowers short-circuit
  `&&`/`||` into `If`/`Store` control flow, so the backend needs no
  logical binary ops. SDF-style code ports by ending with `<= 0.0`.
- Dimensionality from `scene`'s parameter: `vec2<f64>` compiles a 2D
  sketch (bounds functions then return `vec2<f64>`), `vec3<f64>` a
  volume. Mirrors the Lua arity convention.
- Future channels are per-channel optional functions over the same
  positions (P3): e.g. `fn signed_distance(p: vec3<f64>) -> f64`
  declared alongside `scene` causes the operator to declare the
  corresponding `SampleFormat` and emit `sample_channels`. Each channel
  is opt-in and carries its own semantics; none replaces `scene`.
- Bounds are compiled to wasm functions called by `get_bounds` at
  runtime (not const-evaluated), so they may reference overrides —
  WGSL `const` cannot, and bounds usually depend on routed radii.

Routed parameters:

```wgsl
override bearing_radius: f64 = 0.011; // @param key="case.bearing_radius" min=0.005 max=0.05
```

- Only annotated overrides are routed from the F64Map input; values are
  validated against min/max exactly like Lua `@param`. Unannotated
  overrides keep their WGSL defaults. Annotated overrides must be scalar
  f64 (WGSL restricts overrides to scalars already).
- Overrides are substituted as literals during lowering, so routed
  values constant-fold into the generated code. Parameter changes
  recompile the model; the step cache memoizes per parameter set,
  unchanged from Lua.

Starter template (the "new node" content) teaches the conventions and
carries visible aliases — no implicit prelude, so user line numbers equal
naga span line numbers and vanilla-WGSL expectations hold:

```wgsl
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;

override radius: float = 1.0; // @param key="sphere.radius" min=0.000001
const margin: float = 0.5;

fn scene(p: vec3d) -> bool {
    return length(p) <= radius;
}
fn bounds_min() -> vec3d { return vec3d(-(radius + margin)); }
fn bounds_max() -> vec3d { return vec3d(radius + margin); }
```

## Restriction pass

Runs after `Validator::validate(Capabilities::FLOAT64)` succeeds. The
naga validator accepts legal *shader* constructs we must ban (spike:
texture declarations and derivatives validate fine), so this pass is
where our dialect gets its edges and its good error messages:

- No entry points (`@compute` etc.).
- No global `var` in any address space: uniform, storage, workgroup,
  handle (textures/samplers), **and private** — `var<private>` mutation
  validates in naga (spike-confirmed) but breaks the stateless-model
  contract. `const` and `override` are the module-scope vocabulary.
- No expressions: image sample/load, derivatives, ray query, subgroup,
  atomics, `ArrayLength` (runtime-sized arrays are unreachable without
  storage buffers anyway).
- No statements: barriers, image store, atomics, `Kill` (discard).
- No f16 (scalar width 2), no pointers as user-declared function
  parameters or in structs (naga's internal Load/Store pointer exprs on
  locals are of course fine).
- Required-function signature checks: `scene` present and
  bool-returning, bounds pair present, arities and types consistent,
  dimensionality 2 or 3.
- Diagnostics: map naga `Span`s to line/column with the source in hand;
  errors go through `report_error` like the Lua operator's.

Recursion needs no check (parse-rejected). Loops always terminate in
WGSL? No — `loop` without break is legal; we do NOT attempt termination
analysis. Runaway scripts are already handled by the host's wasmtime
epoch interruption (mid-operator cancel infra).

## Backend: naga IR → walrus

The shape of naga IR (from the spike dump): per function, a flat typed
expression arena + structured statements (`Emit`, `Store`, `If`, `Loop
{ body, continuing, break_if }`, `Switch`, `Call`, `Return`, `Break`,
`Continue`, `Block`). This maps directly onto wasm structured control
flow — no relooping, no CFG reconstruction.

- **Values:** scalars map 1:1 (f64→f64, f32→f32, i32/u32→i32, bool→i32).
  Vectors (and later matrices) scalarize: an IR handle of type
  `vec3<f64>` owns 3 wasm locals. Expression results cache in locals at
  their `Emit` point, honoring naga's evaluation-order contract.
- **Composite ops:** `Compose`, `Splat`, `Swizzle`, `AccessIndex` are
  local shuffles. `Select` → wasm `select` per component. `As`
  conversions → wasm promote/demote/convert; float→int uses the
  saturating `trunc_sat` forms (matches WGSL semantics).
- **`%` on floats** (WGSL: truncated, `x - y*trunc(x/y)`) → inline
  expansion; wasm has no frem. Integer `%` → `i32.rem_s`/`rem_u`.
- **Module-scope const arrays** → data segments; runtime `Access`
  becomes an index-clamped linear-memory load (WGSL sanctions clamping
  for out-of-bounds). Placed after the template's own data segments.
- **Function-local arrays / structs:** flattened when statically
  indexed; dynamically-indexed local arrays spill to shadow-stack frames
  using the template's `__stack_pointer`. If this turns out fiddly it
  slips to Phase 3 without hurting the example ports (neither needs it).
- **Math lowering:** native wasm ops inline (abs, sqrt, floor, ceil,
  trunc, round/nearest, min, max, sign via copysign/select); the ~15
  transcendentals call template kernels; composites (mix, clamp, step,
  smoothstep, fract, length, distance, normalize, dot, cross, reflect,
  …) expand inline over components. f32 transcendentals promote →
  f64 kernel → demote; no f32 kernel set.
- **Overrides** are replaced by literal values (routed or default)
  during lowering.
- **Assembly:** load the template with `walrus::Module::from_buffer`,
  build all script functions into it, then add ABI exports —
  `get_dimensions` (const), `get_bounds` (calls bounds fns, interleaved
  min/max write), `sample` (load position f64s, call `scene`, select
  the canonical 1.0/0.0 f32 from the bool, per ABI.md), reuse the
  template's `get_io_ptr`/`memory`. Internalize (un-export) the kernel
  functions afterward.

## Testing

Follow the Lua operator's wasmtime execution-test pattern (compile,
instantiate, write position, call `sample`), which exists precisely
because compile-only tests missed behavioral bugs:

1. **Golden scenes, executed:** sphere, box, 2D sketch, override routing
   (routed + default + out-of-range rejection), const-array hole table
   with a for loop, trig wave surface, boolean CSG chains (short-circuit
   lowering), SDF-style helpers ending in `<= 0.0`,
   swizzle/select/mix/clamp/smoothstep coverage, f32 and i32
   arithmetic, float `%`, nested helper calls.
2. **Differential ports:** the two example scripts ported to WGSL,
   sampled on a probe grid, compared point-for-point against the
   Lua-compiled originals (both avoid the broken Lua math, so they are
   honest cross-checks).
3. **Kernel accuracy:** template kernels executed in wasmtime vs Rust
   `std` across value sweeps (libm tracks std closely; assert tight ULP
   bounds, exact for sqrt/floor-class).
4. **Restriction pass negatives:** one test per banned construct;
   missing/duplicate entry points; wrong signatures.
5. **Wasm validity:** every compiled module reparses under walrus (the
   Lua suite's `assert_valid_wasm` trick) in addition to executing.

## Integration (Phase 2)

- UI: `WgslSource` mirrors the `LuaSource` editor paths in
  `volumetric_ui_v2` (~7 touch points found; editor state, template
  seeding, asset type hint label "WGSL").
- CLI: `info.rs` / `project.rs` handling mirrors Lua; JSON configs keep
  the float-literal requirement for F64Map params.
- Catalog: category "Scripting", display name "WGSL Script".
- Daemon: nothing special — output is ordinary ModelWASM; version-skew
  rules unchanged.

## Phasing

- **P1 — core operator** (target: tray port meshes via CLI): template
  crate + operator + restriction pass + lowering for
  scalars/vectors/control flow/math/overrides/const arrays + ABI
  wrapper + test suite + both example ports.
- **P2 — integration:** ABI variant, UI editor, CLI, catalog, starter
  template polish, span-mapped error display.
- **P3 — by demand:** matrices, dynamically-indexed local arrays and
  structs via shadow stack, an `sd_*` helper library shipped as a
  visible include, per-channel optional functions — e.g.
  `fn signed_distance(p) -> f64` → declared `SampleFormat` +
  `sample_channels` (`Custom("volumetric.tsdf.v1")`: the channel
  machinery `sdf_operator` uses for *baked* TSDFs would here carry
  *analytic* distance), gradient/normal channel for meshing hints.

## Open questions

1. Operator naming: `wgsl_script_operator` / "WGSL Script" for symmetry
   with the Lua operator — or "Shader Script" for discoverability?
2. Lua operator disposition: freeze silently, or land the two small trap
   fixes (`^`/`math.pow` correctness, `and`/`or` scratch-local) so
   existing scripts stop being landmines while frozen?
3. Distance channel (P3): worth pulling into P1/P2 for the remaster
   arc's benefit, or leave until a consumer exists?
