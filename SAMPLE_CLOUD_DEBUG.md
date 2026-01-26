# Sample Cloud Debug System

This document describes the sample cloud dump format and the current rendering
tools, plus known issues found during visualization.

## Format (CBOR)

- **File**: `SampleCloudDump` (CBOR)
- **Fields**:
  - `version`: format version (current: 1)
  - `sets`: list of `SampleCloudSet`

`SampleCloudSet`:
- `id`: u64
- `label`: optional string
- `vertex`: [f32; 3] original vertex
- `hint_normal`: [f32; 3]
- `points`: ordered `SamplePoint` list
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
```
cargo run --bin sample_cloud_dump -- --attempt 0
cargo run --bin sample_cloud_dump -- --attempt 1
cargo run --bin sample_cloud_dump -- --attempt 2
```

ML policy dumps:
```
cargo run --bin sample_cloud_dump -- --ml-policy directional
cargo run --bin sample_cloud_dump -- --ml-policy octant-argmax
cargo run --bin sample_cloud_dump -- --ml-policy octant-lerp
```

Inspect metadata:
```
cargo run --bin sample_cloud_inspect -- --file sample_cloud_attempt0.cbor --set 0
```

## Rendering

CLI:
```
volumetric_cli render \
  -i rotated_cube.wasm/rotated_cube.wasm \
  -o cloud.png --wireframe --max-depth 2 --grid 0 \
  --sample-cloud sample_cloud_attempt0.cbor \
  --sample-cloud-mode split \
  --sample-cloud-set 0
```

Modes:
- `overlay`: mesh + cloud
- `split`: mesh left, cloud right
- `cloud-only`: cloud only

UI:
- Load via **Sample Cloud** panel.
- Select set index and view mode.
- Overlay/split/cloud-only supported.

## Known Issues (Current)

- **Split view framing**: right pane uses mesh camera framing, so clouds often
  render off-frame. The data is present but not visible.
- **Cloud-only framing**: same camera target as mesh, which can place the cloud
  off-center.

## Next Fixes

- Add cloud-bounds camera framing for split/cloud-only modes.
- Optionally add a `--sample-cloud-center` flag in CLI to target the cloud.

## References

- `src/sample_cloud.rs`
- `src/bin/sample_cloud_dump.rs`
- `src/bin/sample_cloud_inspect.rs`
- CLI renderer: `crates/volumetric_cli/src/render.rs`
- UI renderer: `crates/volumetric_ui/src/main.rs`
