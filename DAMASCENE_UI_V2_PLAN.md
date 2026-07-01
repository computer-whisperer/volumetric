# Volumetric UI v2: Damascene Port Plan

## Goal

Port `volumetric_ui` from the current `eframe`/`egui` shell to a Damascene-based UI,
while preserving the existing volumetric engine, project system, operators, WASM
asset bundling, background meshing, and custom 3D renderer.

This should be treated as a v2 UI effort, not an in-place rewrite of the current
UI. The existing egui UI remains the usable baseline until the Damascene version can
load projects, render model exports, and perform the main editing/export flows.

## Why Damascene

Damascene is a better fit for the next UI because it provides:

- a more application-shaped widget vocabulary than egui;
- controlled widgets and app-owned state, which matches Volumetric's project model;
- keyed layout rects, which let the app reserve a viewport region for custom rendering;
- direct `wgpu` integration without requiring Damascene to own the whole render graph;
- headless layout/artifact dumps that should make UI iteration easier once the port is
  structured.

The key architectural fit is host-composed rendering: Damascene paints the app chrome,
and Volumetric paints the 3D viewport into a keyed rect in the same frame.

## Current Baseline

The current `volumetric_ui` combines several responsibilities in one large
`main.rs`:

- project load/save and project DAG editing;
- bundled and filesystem fallback model/operator discovery;
- operator metadata parsing and generated config forms;
- background project execution and per-asset resampling;
- render mode and meshing configuration;
- camera input;
- egui panels and controls;
- egui/wgpu callback glue for the 3D renderer.

The custom renderer itself is already usefully separated under
`crates/volumetric_ui/src/renderer/`. It is tied to the egui callback path only
through `renderer/callback.rs`; the core renderer types and render passes should be
portable to a custom Damascene host.

## First Milestone: Dependency Modernization

Before starting the Damascene port, update Volumetric's dependency stack across the
workspace. This is useful regardless of the port and reduces the risk of fighting
two migrations at once.

Priority updates:

- align UI-side `wgpu` with Damascene's current `wgpu` version;
- review whether the CLI renderer should remain on its own `wgpu` version or move
  at the same time;
- update `wasmtime`, `wasmparser`, `wasm-encoder`, `walrus`, `eframe`, `rfd`, and
  related platform crates where practical;
- keep WASM model/operator builds working for `wasm32-unknown-unknown`;
- keep `cargo build-wasm`, `cargo check --workspace`, and `cargo test --workspace`
  green after each dependency slice.

The likely hard requirement for the Damascene UI is that the v2 renderer and Damascene use
the same `wgpu` crate version, because they must share a device, queue, texture
views, command encoder, and render target.

## Target Architecture

Add a v2 UI path alongside the existing UI.

Recommended shape:

- `volumetric_ui` remains the current egui app until v2 reaches parity.
- Add either:
  - a new crate, `crates/volumetric_ui_v2`, or
  - a new binary/module path inside `crates/volumetric_ui`.
- Extract reusable app/domain state out of egui-specific code before rebuilding
  controls in Damascene.

The v2 host should own:

- `winit` event loop;
- `wgpu` instance, adapter, device, queue, surface, and surface configuration;
- one `damascene_wgpu::Runner`;
- one Volumetric 3D renderer instance;
- frame ordering between Volumetric viewport rendering and Damascene UI rendering.

Frame sketch:

1. Poll background work and app state.
2. Build the Damascene `El` tree.
3. Call `damascene_runner.prepare(...)`.
4. Query `damascene_runner.rect_of_key("viewport")`.
5. Render the Volumetric 3D scene into the app-owned viewport texture.
6. Let Damascene composite that texture through the keyed `surface()` element.
7. Submit and present.

## Viewport Strategy

The central risk is the 3D viewport.

Start with the simplest robust path:

- reserve a keyed Damascene element for the viewport;
- render the Volumetric scene to an offscreen texture sized to the viewport rect;
- expose that texture to Damascene as an app-owned `surface()` with opaque alpha;
- let normal Damascene z-order, clipping, hit-testing, and overlays handle composition.

This keeps the existing renderer's viewport-sized G-buffer, SSAO, grid, line, and
point passes intact. A later optimization can render directly into a scissored
surface region if the offscreen path becomes a performance issue.

Camera input should be handled by the custom host or app event layer, not hidden in
the Damascene widget tree. Pointer events over the viewport rect should update the
existing orbit/pan/zoom camera state.

## UI Shell Sketch

Viewport-dominant workbench: the 3D view is ~90% of the screen and everything
else packs into a thin menubar, floating viewport chrome, and one project
panel.

- top bar (single thin strip):
  - menubar: `File` (new/open/save/export actions) and `Add` (bundled model +
    operator catalogs as menu groups — adding is write-once, no permanent
    panel);
  - run/cancel, auto-rebuild toggle, run + preview status chips;
- center: keyed 3D viewport filling everything else, with floating chrome
  composed via `stack` (only keyed nodes hit-test, so camera input passes
  through):
  - top-right cluster: grid/SSAO/frame toggles plus `select` pickers for
    default render mode, resolution, and camera scheme;
  - bottom HUD: unkeyed one-line badges for scene counts and status;
- right project panel (the one fat sidebar):
  - pipeline accordion: imports → steps → exports, with select/delete rows;
  - Outputs list: per-output visibility dot, pin, view, and (Phase 3b)
    per-output render mode/resolution overrides;
  - inspector for the current selection: step editor, operator config form,
    Lua source.

Damascene widgets in play: `menubar`, `accordion`, `select`, `popover`,
`toolbar`, `card`, `field_row`, `switch`, `text_input`, `text_area`;
`resize_handle` for the panel width and `sheet` for a wide Lua editor are
candidates for Phase 3c.

## Migration Slices

### Current Progress

- Slice 0 is complete: the workspace is dependency-modernized and CI covers
  rustfmt, clippy, tests, and WASM asset builds.
- Slice 1 has started with `crates/volumetric_ui_v2`, a native Damascene shell that
  opens via `damascene_winit_wgpu::run`, reserves a keyed `viewport` region, and can
  dump Damascene bundle artifacts through `dump_v2_shell`.
- The v2 shell now owns real app state: it initializes a `Project` with a bundled
  model, renders bundled model/operator catalogs from `volumetric_assets`, routes
  Damascene click/activation events, and exposes a project summary for the future
  renderer host.
- Project details are now represented in v2 with imports, operation timeline,
  and exports lists plus initial select, delete, move, and add-export actions.
  This is the starting point for rebuilding the old project sequence editor;
  selected operation steps can also retarget their first input and update output
  export wiring.
- The v2 shell has been reworked toward Damascene's stock app vocabulary: sidebar
  groups and menu buttons, toolbar headers/actions, card anatomy, table rows,
  field rows, icon buttons, and tooltips now carry the main layout instead of
  hand-rolled panel and row styling.
- The shell is shifting toward a denser CAD/workbench shape: narrower rails,
  tighter table rows, compact catalog/action controls, and controlled viewport
  preview settings for render mode, preview resolution, grid visibility, and
  SSAO are now part of app state.
- The v2 app can now run the current `Project`, retain materialized runtime
  exports, report elapsed time/errors, and expose the selected runtime asset as
  the handoff point for the future renderer/export host.
- The renderer handoff is now explicit as a `PreviewRequest`: selected runtime
  WASM bytes, render mode, mesh plan, grid/SSAO flags, and stale state are
  packaged from app state instead of being inferred from scattered UI controls.
- The native v2 binary now uses a custom winit/wgpu host built on
  `damascene_wgpu::Runner`: it prepares Damascene layout, resolves `VIEWPORT_KEY`, renders
  `volumetric_renderer::Renderer` into a viewport-sized app-owned texture, and
  exposes that texture as a keyed Damascene `surface()` element for composition.
- The old egui renderer module has been extracted into `crates/volumetric_renderer`.
  Core renderer types/pipelines are usable without egui, while the previous egui
  paint callback remains behind an `egui-callback` feature for `volumetric_ui`.
- The v2 host can convert `PreviewRequest` into cached renderer `SceneData` using
  point cloud, marching-cubes, or Adaptive Surface Nets v2 mesh generation. When
  no runtime preview is available it falls back to the renderer test scene, so
  the viewport remains structurally representative.
- The v2 crate now depends on the published `damascene-{core,wgpu,winit-wgpu}`
  crates from crates.io, so CI no longer needs a sibling checkout of the
  Damascene repo.
- Project execution is now asynchronous. A single host-owned `BackgroundWorker`
  thread serially runs project execution and preview meshing, coalescing jobs
  per kind so bursts of edits collapse to one rebuild. The app no longer
  executes inline: a Run click (or, when the auto-rebuild toggle is on, any
  project edit) queues a run that the host dispatches and drains per frame.
  Cancellation is cooperative (`Project::run_cancellable` checks a flag between
  timeline steps) and generation-tagged results discard superseded/cancelled
  runs. Run status, Run/Cancel, and the auto-rebuild toggle are surfaced in the
  shell.
- Incremental project edits now keep the last good run output on screen instead
  of clearing it: edits only mark the output stale (and queue a run when
  auto-rebuild is on), a failed rerun keeps the previous preview while surfacing
  the error, and only a full project replacement (new/open) clears the runtime.
  Adding a new model/operator still transitions the viewport to that new node's
  (building) preview, since selection follows the freshly added output.
- Operators are now configurable. Config schema handling lives in
  `volumetric::operator_config` (parse CDDL, encode/decode the CBOR map, seed
  defaults). Adding an operator reads its metadata and wires one step input per
  declared input; selecting an operator step shows an inspector config form with
  editable fields (float/int/tstr via controlled text inputs, bool via switch,
  string enums via a button group) that commit into the step's CBOR blob and
  mark the project dirty. Each `ModelWASM` input slot has its own selector, so
  multi-input operators (e.g. boolean) can wire every model input; retargeting
  the primary slot renames the output as before. `LuaSource` inputs are edited
  in a `text_area` whose contents are written straight back to the step's input
  bytes. Not yet editable in the form: `VecF64` literal/asset and `Blob` file
  picker.
- The viewport now renders a *set* of outputs, not a single selected asset —
  the capability the egui v1 UI had (`asset_render_data` multi-entity rendering)
  that the initial port dropped. `preview_requests()` emits one request per
  renderable runtime export; a host-side `PreviewCache` meshes each output on the
  background worker (coalesced per output id), keeps the last good mesh per
  output while a rebuild is in flight, evicts outputs that leave the set, and
  composites the cached geometry into one `SceneData` (memoized against the
  contributing keys). Camera framing changed accordingly: instead of snapping on
  every rebuild, the camera re-frames the union bounds when the set of rendered
  outputs changes or the Frame command is issued, and otherwise leaves the
  user's view alone. This is Phase 1 of the render/selection redesign; the
  selection surface (follow-selected-node + per-output pins and render modes) and
  the layout rework are Phases 2–3.
- Phase 2 wires the selection surface: the viewport now follows the selected
  pipeline node (selecting an import/step/export previews its output) plus any
  pinned outputs, dissolving the old disconnect where clicking a node never
  changed what rendered. `selected_render_id` maps the selection to a runtime
  output; `pinned_outputs` keeps chosen outputs on screen across selection
  changes; the Outputs inspector card gained a per-output pin toggle, a View
  action, and a visibility dot. Selecting a node with no materialized output
  (e.g. an unexported step) renders only the pins and shows a "run to preview"
  hint. Runtime output rendering resolves through `runtime_assets` (stable
  `Arc`s), so previewing an import that hasn't been run/exported is deferred.
  Per-output render *mode/resolution* overrides remain global for now and land
  with the Outputs-list layout in Phase 3.
- Phase 3a restructured the chrome around a viewport-dominant workbench: the
  left sidebar and the four inspector cards are gone. A single thin top bar
  carries the menubar (`File` → New Project; `Add` → one-click bundled model /
  operator entries via `add:model:{name}` / `add:operator:{name}`), the
  run/status chips, and the auto-rebuild toggle. View controls (grid, SSAO,
  frame, and `select`-based mode / resolution / camera pickers) float over the
  viewport's top-right corner — hit-testing only targets keyed nodes, so camera
  input passes through everywhere else — with a one-line unkeyed HUD along the
  bottom. All project structure lives in one right panel: a pipeline accordion
  (imports/steps/exports with counts), the Outputs list, and the selection
  inspector. Catalog "selection" state (`selected_model`/`selected_operator`)
  was deleted along with the dead Open/Save/Export STL buttons; per-output
  render overrides, panel resize, and wiring real file actions are Phase 3b/3c.
- Phase 3b delivered the per-output render overrides deferred from Phase 2: an
  `OutputRender { mode, resolution }` map keyed by asset id feeds
  `render_request_for_asset`, so each output can mesh with its own mode and
  resolution while the viewport pickers stay the defaults for un-overridden
  outputs (matching egui v1's per-export Render combo, minus the None=hidden
  mode, which pins/eye already cover). Each Outputs row shows its effective
  `mode · res` and a settings popover (gear; highlighted when overriding)
  with mode/resolution buttons and a "use viewport defaults" reset — picks
  keep the panel open, outside click or Escape dismisses. Overrides prune
  with pins when a run drops an output. Remaining for 3c: panel resize
  handle, wide Lua sheet, real file actions.
- Phase 3c wired the real file actions and the panel divider. The app queues a
  `FileAction` (Open/Save/ExportStl) and the host drains it each frame,
  showing blocking rfd dialogs (same trade-off as v1) and routing outcomes
  back: Open replaces the project, clears runtime state, and queues a run;
  Save writes `.vproj` via the engine's `save_to_file`; Export STL (in the
  per-output settings popover) converts the host's cached preview mesh —
  what's actually on screen, transforms applied — to `volumetric::Triangle`s
  and writes binary STL, with a status hint when the output has no mesh (e.g.
  points mode). The project panel is now resizable via a `resize_handle`
  divider (240–560px, `Side::End`), with a small row gap so the handle's grab
  band stays off the viewport's hit target. Still open: the wide Lua `sheet`
  (deferred until the in-panel editor actually pinches) and remembering the
  save path for one-click re-save.
- The v1-parity sweep closed the remaining functional gaps found by comparing
  against the egui app's full feature inventory: per-output Export WASM;
  Add > Import for external model WASM / STL meshes / heightmap images (the
  two Blob-input import operators are hidden from the plain operator list and
  reached via Import, with config defaults derived from the operator schema
  instead of v1's hand-encoded CBOR); per-output ASN2 quality settings
  (vertex/normal refinement, sharp edge detection with angle + residual
  steppers) — the base/depth split stays derived from the single resolution
  knob, unlike v1's two coupled drags; SSAO radius/bias/strength steppers in
  a viewport-cluster popover; camera Reset; a 16-256 resolution ladder;
  output rename in the step editor (also rewrites downstream AssetRef inputs,
  pins, overrides, and selection — v1 left references dangling); Save vs Save
  As with a remembered project path; Lua reset-to-template; precursor lineage
  in the export inspector; and meshing statistics (time, triangle/point/
  sample counts in the output popover with ASN2 per-stage profiling lines,
  plus scene totals in the viewport HUD). Not yet ported: the wasm32/web
  build — v2's host is native-only *for now, but a web version is expected to
  follow shortly*, so new host-side code must keep platform assumptions
  behind narrow seams (the `FileAction` queue and bytes-based app methods are
  the pattern). Known web-hostile spots to fix when the port starts: blocking
  `rfd::FileDialog` calls (web needs `AsyncFileDialog` + polling, as in v1),
  the `BackgroundWorker` std::thread, `std::time::Instant` in preview stats,
  and path-based project open/save (web needs bytes + browser download).
  Also deliberately changed from v1: ladder resolution presets instead of an
  arbitrary-resolution slider.
- The custom host was reviewed against upstream Damascene's hosted-app path
  (`WinitWgpuApp` + `run_host_app_with_config`) and kept: pixel-exact,
  DPI-scaled viewport rendering needs the host-side `rect_of_key` surface,
  which hosted apps cannot reach today (BuildCx viewport metrics are a
  documented future addition; the hosted `AppTexture` path stretches a
  fixed-size texture). In preparation for the web shell, the host was split
  into `session.rs` — the platform-neutral core (preview cache, viewport
  renderer/target, camera input typed over Damascene's
  `PointerButton`/`KeyModifiers`, run/cancel generations, `JobQueue` +
  `execute_job` for background work) — and `host.rs`, now only winit plumbing:
  event loop, input mapping, the worker thread pumping the job queue, and the
  blocking `rfd` dialogs. A web shell plugs a canvas event source, async
  dialogs, and its own job execution strategy into the same `Session`.

### Slice 0: Dependency Update

Modernize dependencies and keep the current UI building. This is the dependency
cleanup effort that should happen before v2 starts in earnest.

Acceptance:

- `cargo build-wasm` passes;
- `cargo check --workspace` passes;
- `cargo test --workspace` passes;
- current egui UI still launches.

### Slice 1: Damascene Host Prototype

Create a minimal v2 app that opens a window and renders Damascene chrome.

Acceptance:

- window opens;
- basic sidebar + viewport placeholder layout renders;
- Damascene artifact dump exists for the shell.

### Slice 2: Renderer Integration

Port the existing renderer to the dependency-updated `wgpu` API and render a static
test mesh in the keyed viewport.

Acceptance:

- viewport rect is resolved from Damascene layout;
- renderer draws into that rect at correct scale;
- resizing keeps the viewport non-distorted;
- no egui callback dependency remains in the v2 path.

### Slice 3: Bundled Model Smoke Path

Load a bundled demo model, mesh it, and render it in v2.

Acceptance:

- bundled sphere loads automatically or via a simple button;
- background resampling completes;
- point cloud and ASN v2 mesh modes render;
- orbit/pan/zoom work over the viewport.

### Slice 4: Project and Operator Flows

Rebuild the existing project workflows using Damascene.

Acceptance:

- load/save project;
- add bundled model;
- add operator with generated config form;
- run project;
- inspect exports;
- export STL/WASM.

### Slice 5: Parity and Polish

Only after the core flows work, improve layout, density, keyboard support, and
visual polish.

Acceptance:

- current egui workflow can be completed in v2;
- errors/progress are visible and non-blocking;
- common controls are keyboard reachable;
- v2 is comfortable enough to become the default UI.

## Open Questions

- Should v2 be a new crate (`volumetric_ui_v2`) or a second binary inside
  `volumetric_ui`?
- Should the CLI renderer be upgraded to the same `wgpu` version now, or kept
  isolated until needed?
- Should project/domain state be extracted into a shared crate/module before the
  v2 crate is introduced?
- How much web support should v2 target initially? Native-first is the pragmatic
  path, with web after the custom host is stable.

## Non-Goals For This Port

- Do not resume the sharp-edge research as part of the UI port.
- Do not rewrite the volumetric engine or project DAG.
- Do not replace the custom 3D renderer with Damascene primitives.
- Do not chase full egui feature parity before proving the Damascene host and viewport.
- Do not remove the current egui UI until v2 is clearly usable.
