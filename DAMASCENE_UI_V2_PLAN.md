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

Use an application workbench layout:

- left sidebar:
  - project status and file actions;
  - bundled demo models;
  - operator toolbox;
  - project timeline;
- center:
  - keyed 3D viewport;
  - lightweight viewport HUD for asset/triangle/sample counts;
- right inspector/sheet:
  - selected timeline entry editor;
  - operator input/config editing;
- lower or sidebar section:
  - project exports;
  - render mode and meshing controls;
  - profiling stats;
  - export STL/WASM actions.

Damascene widgets that should map well:

- `sidebar`, `toolbar`, `card`, `tabs_list`, `field_row`, `form_item`;
- `select`, `switch`, `checkbox`, `slider`, `text_input`, `text_area`;
- `resize_handle` for adjustable sidebar/inspector widths;
- toasts or alerts for errors and completion messages.

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
  mark the project dirty. Not yet editable in the form: Lua source (`text_area`),
  `VecF64` literal/asset, `Blob` file picker, and per-slot model selectors for
  multi-input operators (only the first model input is retargetable so far).

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
