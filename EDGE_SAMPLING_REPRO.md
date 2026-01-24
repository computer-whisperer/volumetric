# Edge Sampling Reproduction Procedure

Temporary file documenting how to reproduce rotated cube edge renders for edge sampling analysis.

## Prerequisites

```bash
# Build WASM modules
cargo build-wasm

# Build CLI
cargo build -p volumetric_cli --release
```

Note: `--sharp-edges` is currently stubbed in production (Stage 4 passthrough).
It is included below for historical parity with earlier runs.

## Step 1: Create Rotated Cube Project

The existing `cube_attempt.vproj` already has the setup, but to recreate from scratch:

```bash
# Create new project with rectangular prism operator
cargo run -p volumetric_cli --release -- project-new \
  -i target/wasm32-unknown-unknown/release/rectangular_prism_operator.wasm \
  -o rotated_cube_test.vproj

# Add rotation operator (35.264° X, 45° Y gives "isometric" orientation)
cargo run -p volumetric_cli --release -- project-add-op \
  -p cube_attempt.vproj \
  --operator target/wasm32-unknown-unknown/release/rotation_operator.wasm \
  -i 'asset:rectangular_prism_output' \
  -i 'json:{"rx_deg": 35.264, "ry_deg": 45.0, "rz_deg": 0.0}' \
  --output-id rotated_cube
```

## Step 2: Export Rotated Cube

The project exports both original and rotated. Export just the rotated one:

```bash
cargo run -p volumetric_cli --release -- project-export \
  -p cube_attempt.vproj \
  --asset rotated_cube \
  -o rotated_cube.wasm
```

This creates `rotated_cube.wasm/rotated_cube.wasm` (1633 bytes).

## Step 3: Verify Rotation

```bash
# Front view should show hexagon shape (proves rotation)
cargo run -p volumetric_cli --release -- render \
  -i rotated_cube.wasm/rotated_cube.wasm \
  -o verify_rotation.png \
  --wireframe --sharp-edges \
  --views front \
  --projection ortho --ortho-scale 5 \
  --width 1024 --height 1024 --grid 1
```

## Step 4: Edge Close-up Renders

### Best render for seeing individual triangles (16³ resolution):

```bash
cargo run -p volumetric_cli --release -- render \
  -i rotated_cube.wasm/rotated_cube.wasm \
  -o rot_edge_depth1.png \
  --wireframe --sharp-edges \
  --max-depth 1 \
  --camera-pos 0,0,5 --camera-target 0,0,0 \
  --projection ortho --ortho-scale 0.6 \
  --near 3.8 \
  --width 1024 --height 1024 --grid 0
```

### Corner view at 32³:

```bash
cargo run -p volumetric_cli --release -- render \
  -i rotated_cube.wasm/rotated_cube.wasm \
  -o rot_edge_corner.png \
  --wireframe --sharp-edges \
  --max-depth 2 \
  --camera-pos 2,2,5 --camera-target 0.5,0.5,0.5 \
  --projection ortho --ortho-scale 0.6 \
  --near 4.5 \
  --width 1024 --height 1024 --grid 0
```

### Tight edge view at 32³:

```bash
cargo run -p volumetric_cli --release -- render \
  -i rotated_cube.wasm/rotated_cube.wasm \
  -o rot_edge_tight.png \
  --wireframe --sharp-edges \
  --max-depth 2 \
  --camera-pos 0,0,5 --camera-target 0,0,0 \
  --projection ortho --ortho-scale 0.4 \
  --near 4.2 \
  --width 1024 --height 1024 --grid 0
```

## Key CLI Options

- `--near <dist>`: Near clipping plane - use this to slice through mesh and see edge vertices
- `--max-depth <n>`: Lower = coarser mesh, easier to see individual triangles (1=16³, 2=32³, 3=64³, 4=128³)
- `--projection ortho`: Orthographic for consistent triangle sizes in view
- `--ortho-scale <n>`: Smaller = more zoomed in
- `--sharp-edges`: Sharp edge detection flag (currently no-op; kept for compatibility)

## Current State Observations

1. Edge straightness is good (improved from screen03.png)
2. Triangle sizes vary near edges - some much larger than neighbors
3. Triangulation pattern not symmetric across edge boundaries
4. Issues most visible at low resolution (depth 1-2)

## Files Created

- `rotated_cube.wasm/rotated_cube.wasm` - exported model
- `rot_edge_depth1.png` - best for seeing individual triangles
- `rot_edge_corner.png` - corner view
- `rot_edge_tight.png` - tight edge zoom
- `rotated_front.png` - verification of rotation (hexagon shape)
