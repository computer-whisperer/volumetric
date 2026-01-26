# Sample Cloud Debug System

This document describes the sample cloud dump format and the rendering tools for
visualizing sampling behavior in the edge detection research pipeline.

## Format (CBOR)

- **File**: `SampleCloudDump` (CBOR serialized)
- **Fields**:
  - `version`: format version (current: 1)
  - `sets`: list of `SampleCloudSet`

`SampleCloudSet`:
- `id`: u64 identifier
- `label`: optional string description
- `vertex`: [f32; 3] original vertex position being analyzed
- `hint_normal`: [f32; 3] hint normal used for sampling
- `points`: ordered `SamplePoint` list (in sampling order)
- `meta`: `SampleCloudMeta` (samples_used, note)

`SamplePoint`:
- `position`: [f32; 3]
- `kind`: `Unknown | Probe | Crossing | Inside | Outside`

Implementation: `src/sample_cloud.rs`

## Recording Behavior

- Every `SampleCache::sample()` logs an `Inside` or `Outside` point in order.
- `find_crossing_in_direction()` logs the final `Crossing` point.
- Ordered samples are intended for post-hoc debugging (timelines, clustering).

Implementation: `src/adaptive_surface_nets_2/stage4/research/sample_cache.rs`

## Generating Dumps

Attempt dumps:
```bash
cargo run --bin sample_cloud_dump -- --attempt 0
cargo run --bin sample_cloud_dump -- --attempt 1
cargo run --bin sample_cloud_dump -- --attempt 2
```

ML policy dumps:
```bash
cargo run --bin sample_cloud_dump -- --ml-policy directional
cargo run --bin sample_cloud_dump -- --ml-policy octant-argmax
cargo run --bin sample_cloud_dump -- --ml-policy octant-lerp
```

Custom output path:
```bash
cargo run --bin sample_cloud_dump -- --attempt 0 --out my_cloud.cbor
```

Inspect metadata:
```bash
cargo run --bin sample_cloud_inspect -- --file sample_cloud_attempt0.cbor --set 0
```

## CLI Rendering

The CLI renderer uses a GPU-accelerated point rendering pipeline with proper depth
testing against mesh geometry.

### Model Alignment

**Important:** Sample clouds are generated using `AnalyticalRotatedCube::standard_test_cube()`,
which is a unit cube (±0.5) centered at origin, rotated by rx=35.264°, ry=45°, rz=0°.

To visualize sample clouds correctly, use a model that matches this geometry:
- `analytical_cube.wasm` - Lua-based model matching the analytical cube exactly
- Do NOT use `rotated_cube.wasm` - it has different bounds/position and will appear misaligned

### Basic Usage

```bash
volumetric_cli render \
  -i analytical_cube.wasm \
  -o cloud.png \
  --sample-cloud sample_cloud_attempt0.cbor \
  --sample-cloud-mode overlay \
  --sample-cloud-set 0
```

### Render Modes

- `overlay`: Sample points rendered on top of mesh with depth testing
- `split`: Side-by-side view (mesh left, cloud right)
- `cloud-only`: Only the sample cloud, no mesh

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sample-cloud <path>` | - | Path to CBOR sample cloud file |
| `--sample-cloud-set <n>` | 0 | Set index to render (0-based) |
| `--sample-cloud-id <id>` | - | Set ID to render (alternative to index) |
| `--sample-cloud-mode <mode>` | overlay | Render mode: overlay, split, cloud-only |
| `--sample-cloud-size <f>` | 0.02 | Point size as fraction of scene size |
| `--sample-cloud-color <hex>` | 33ccff | Default point color (for Probe/Unknown) |
| `--sample-cloud-vertex-color <hex>` | ff3366 | Vertex marker color |

### Point Color Coding

Points are automatically colored by their `SamplePointKind`:

| Kind | Color | Description |
|------|-------|-------------|
| Crossing | Green (#33ff66) | Surface crossing point |
| Inside | Yellow (#ffcc33) | Point inside the surface |
| Outside | Orange (#ff6633) | Point outside the surface |
| Probe | Cyan (configurable) | Generic probe point |
| Unknown | Cyan (configurable) | Unclassified point |
| Vertex | Pink (configurable) | The vertex being analyzed (first point) |

### Example Commands

Overlay with wireframe mesh:
```bash
volumetric_cli render \
  -i analytical_cube.wasm \
  -o overlay.png --wireframe --grid 0 \
  --sample-cloud sample_cloud_attempt0.cbor \
  --sample-cloud-mode overlay \
  --sample-cloud-set 5
```

Split view for comparison:
```bash
volumetric_cli render \
  -i analytical_cube.wasm \
  -o split.png --wireframe --grid 0 \
  --sample-cloud sample_cloud_attempt0.cbor \
  --sample-cloud-mode split \
  --sample-cloud-set 5
```

Cloud only with larger points:
```bash
volumetric_cli render \
  -i analytical_cube.wasm \
  -o cloud.png \
  --sample-cloud sample_cloud_attempt0.cbor \
  --sample-cloud-mode cloud-only \
  --sample-cloud-size 0.04
```

## UI Rendering

The UI application (`volumetric_ui`) also supports sample cloud visualization.

- Load via **Sample Cloud** panel
- Select set index and view mode
- Overlay/split/cloud-only supported
- Adjustable point size via slider

## Implementation Details

### CLI Point Pipeline

The CLI uses a GPU-accelerated instanced point rendering pipeline:

- **Shader**: `crates/volumetric_cli/src/shaders/point.wgsl`
- **Pipeline**: `crates/volumetric_cli/src/headless_renderer.rs`
- **Integration**: `crates/volumetric_cli/src/render.rs`

Features:
- Screen-space point sizing (constant pixel size regardless of depth)
- Anti-aliased circle shapes with smooth edges
- Proper depth testing against mesh geometry
- Alpha blending for transparency
- Instanced rendering for efficiency

### UI Point Pipeline

The UI uses a similar but separate implementation:

- **Shader**: `crates/volumetric_ui/src/renderer/shaders/point.wgsl`
- **Pipeline**: `crates/volumetric_ui/src/renderer/pipelines/point.rs`

## Known Issues

- **Split/cloud-only camera framing**: These modes use mesh bounds for camera
  framing, so sample clouds may render off-center if the vertex being analyzed
  is far from the mesh center.

- **Model alignment**: Sample clouds are generated from the analytical test cube
  (`AnalyticalRotatedCube::standard_test_cube()`). Using a different model (like
  `rotated_cube.wasm`) will cause the sample points to appear misaligned with the
  mesh surface. Always use `analytical_cube.wasm` for visualization.

---

## Historical Note: Scaling Issues (2026-01-25)

**WARNING:** Sample cloud dumps generated before 2026-01-25 21:00 EST were created
with incorrect scaling parameters. If you have old `.cbor` files, regenerate them.

### What Was Wrong

The research benchmarks used `cell_size = 1.0` for all attempt algorithms. For a
unit cube (side 1.0), this resulted in massively oversized sample clouds:

- `search_distance = 0.5 * 1.0 = 0.5` (half the entire model!)
- `probe_epsilon = 0.1 * 1.0 = 0.1` (10% of the entire model!)

Sample clouds appeared to span the entire cube instead of being confined to a
small cell neighborhood as intended. The algorithms appeared to work because
they over-sampled everything, but this was not representative of real surface
nets behavior.

### Current Status

The `RESEARCH_CELL_SIZE` constant in `attempt_runner.rs` controls this parameter.
For realistic benchmarking, use smaller cell sizes (e.g., 0.2 or smaller). Note
that the attempt algorithms also have hardcoded absolute thresholds that break
scale invariance - see `EDGE_REFINEMENT_RESEARCH.md` for details.

### Regenerating Sample Clouds

After any changes to cell_size or algorithm parameters, regenerate sample clouds:

```bash
cargo run --bin sample_cloud_dump -- --attempt 0 --out sample_cloud_attempt0.cbor
```

Verify the sample cloud looks correct by rendering it with overlay mode - points
should cluster tightly around the vertex being analyzed, not span the entire model.

## Planned Improvements

- Add `--sample-cloud-center` flag to center camera on sample cloud bounds
- Add cloud-bounds camera framing for split/cloud-only modes
- Support rendering multiple sample sets simultaneously for comparison

## References

- Core format: `src/sample_cloud.rs`
- Dump tool: `src/bin/sample_cloud_dump.rs`
- Inspect tool: `src/bin/sample_cloud_inspect.rs`
- CLI renderer: `crates/volumetric_cli/src/render.rs`
- CLI point shader: `crates/volumetric_cli/src/shaders/point.wgsl`
- CLI headless renderer: `crates/volumetric_cli/src/headless_renderer.rs`
- UI renderer: `crates/volumetric_ui/src/main.rs`
- UI point pipeline: `crates/volumetric_ui/src/renderer/pipelines/point.rs`
- Analytical cube model: `analytical_cube.wasm` (Lua-based, matches `AnalyticalRotatedCube::standard_test_cube()`)
- Analytical cube Lua source: `analytical_cube.lua` (includes regeneration instructions)
- Analytical cube definition: `src/adaptive_surface_nets_2/stage4/research/analytical_cube.rs`
