# Volumetric

A high-performance volumetric modeling engine where models and operations are defined as portable WebAssembly (WASM) modules.

## Core Concept

The project explores a "model-as-code" paradigm where a 3D physical model is defined by a simple function: `sample(position) -> f32` returning a density value. By leveraging WebAssembly, we can define complex, parametric models that are:
- **Portable**: Run anywhere with a WASM runtime.
- **Fast**: Near-native execution speed.
- **Composable**: Models can be transformed and combined using Operator modules that manipulate WASM bytecode.

## Project Structure

- `src/`: The host application (Orchestrator). Built with Rust, `wasmtime` for execution, and `egui` for the UI.
- `crates/models/`: Example model definitions (Sphere, Torus, Mandelbulb, etc.).
- `crates/operators/`: Modules that transform or combine models. Includes **Transform Operators** (translate, scale, rotation, boolean) and **Generator Operators** (rectangular_prism, stl_import, heightmap_extrude, lua_script).

## Architecture

The system is divided into three main components:

### 1. Model WASM Modules
A Model module is a WASM artifact that exports:
- `get_dimensions() -> u32`: Returns the number of spatial dimensions (typically 3).
- `get_bounds(out_ptr: i32)`: Writes axis-aligned bounding box to memory as interleaved `[min₀, max₀, min₁, max₁, ...]` f64 values.
- `sample(pos_ptr: i32) -> f32`: Reads position from memory and returns a density value.
- `memory`: Linear memory export for I/O buffers.

This N-dimensional ABI supports models of any dimensionality and uses pointer-based I/O for efficient batch sampling. See [ABI.md](ABI.md) for the complete specification including memory layouts and reserved offsets.

**Important: Models are NOT Signed Distance Functions (SDFs).** The `sample` function returns an occupancy/density value where only the sign matters for geometry extraction (`> 0` = inside, `<= 0` = outside). Current demo models return binary values (`1.0` for inside, `0.0` for outside). This means:
- You cannot compute surface normals via gradient (central differences yield zero almost everywhere)
- The meshing algorithm uses binary classification and edge-crossing detection, not gradient descent
- Vertex refinement uses binary search along candidate directions, not gradient-based optimization

### 2. Operator WASM Modules
Operators are the "compilers" of the volumetric world. They take existing models or configurations as input and produce a new Model WASM as output. 

Operators export:
- `get_metadata() -> i64`: Returns a pointer/length to a CBOR-encoded `OperatorMetadata` struct, describing inputs (e.g., Model WASM, CDDL-schema-defined configurations).
- `run()`: The execution entry point where it pulls inputs via host imports and pushes results.

### 3. The Orchestrator (Host)
The host application manages the lifecycle of models and operators:
- **Project DAG**: Manages a sequence of operations to build complex scenes.
- **Rendering**: Implements both Point Cloud sampling and Marching Cubes (CPU-based) to visualize the WASM-defined volumes.
- **Bytecode Manipulation**: Orchestrates the execution of Operators to generate new model bytecode on the fly.

## Available Operators

Operators are divided into two categories:

### Transform Operators
Transform operators take an existing model and produce a modified version:

| Operator | Description | Configuration |
|----------|-------------|---------------|
| `translate` | Moves a model in 3D space | `{dx, dy, dz}` - displacement in each axis |
| `scale` | Scales a model uniformly or non-uniformly | `{sx, sy, sz}` - scale factors |
| `rotation` | Rotates a model around an axis | `{angle, axis}` - angle in radians and axis vector |
| `boolean` | Combines two models with CSG operations | `{operation}` - "union", "subtract", or "intersect" |

### Generator Operators
Generator operators create new models from configuration or external data:

| Operator | Description | Inputs |
|----------|-------------|--------|
| `rectangular_prism` | Creates a box shape | CBOR config: `{width, height, depth}` |
| `stl_import` | Converts STL mesh to volumetric model | STL blob + CBOR config: `{scale, translate, center}` |
| `heightmap_extrude` | Extrudes a heightmap image to 3D | Image blob + CBOR config: `{width, depth, height, clip}` |
| `lua_script` | Custom density function via Lua script | Model WASM + Lua source code |

## Demo Models

The project includes several example models in `crates/models/`:

| Model | Description |
|-------|-------------|
| `simple_sphere_model` | Unit sphere centered at origin |
| `simple_torus_model` | Torus (donut shape) |
| `mandelbulb_model` | 3D Mandelbrot fractal |
| `rounded_box_model` | Box with rounded edges |
| `gyroid_lattice_model` | Triply periodic minimal surface lattice |

## Getting Started

### Prerequisites

You need the Rust toolchain and the WASM target:
```bash
rustup target add wasm32-unknown-unknown
```

### Building

Build the host application:
```bash
cargo build --release
```

Build all WASM demo models and operators:
```bash
cargo build-wasm
```
*(This uses a Cargo alias defined in `.cargo/config.toml`)*

### Running the GUI

```bash
cargo run -p volumetric_ui --release
```

In the UI:
1.  **Demos**: Load pre-built models from the "Demo" panel.
2.  **Operations**: Apply operators like "Translate" or "Boolean" to transform your models.
3.  **Visualization**: Toggle between Point Cloud and Marching Cubes rendering modes.

### Running the CLI

For batch processing and profiling, use the command-line tool:

```bash
cargo run -p volumetric_cli --release -- <COMMAND>
```

#### Mesh Command

Generate STL meshes from volumetric models:

```bash
volumetric_cli mesh -i <file> -o <output.stl>
```

**Arguments:**
- `-i, --input <file>` - Input file: either a `.wasm` model or a `.vproj` project file
- `-o, --output <file>` - Output STL file path

**Options:**
- `--base-cell-size <f>` - Coarse grid cell size in world units (default: 0.25)
- `--max-depth <n>` - Refinement depth (default: 4). Finest cell size = base-cell-size / 2^depth
- `--vertex-refinement <n>` - Vertex position refinement iterations (default: 12)
- `--normal-refinement <n>` - Normal estimation iterations (default: 12, use 0 to disable)
- `--normal-epsilon <f>` - Normal probe distance as fraction of cell size (default: 0.1)
- `--sharp-edges` - Enable sharp edge detection and vertex duplication for hard edges (currently no-op; Stage 4 is stubbed)
- `--sharp-angle <degrees>` - Angle threshold for sharp edge detection (default: 20, currently no-op)
- `--sharp-residual <f>` - Residual multiplier for sharp edge clustering (default: 4.0, currently no-op)
- `-q, --quiet` - Suppress profiling output

**Examples:**
```bash
# Mesh a WASM model with default settings
volumetric_cli mesh -i simple_torus_model.wasm -o torus.stl

# Faster meshing with lower resolution and no normal refinement
volumetric_cli mesh -i model.wasm -o output.stl --max-depth 3 --normal-refinement 0

# Mesh a project file
volumetric_cli mesh -i scene.vproj -o scene.stl

# Legacy sharp edge flags (currently no effect; kept for compatibility)
volumetric_cli mesh -i box.wasm -o box.stl --sharp-edges --sharp-angle 30
```

#### Render Command

Generate PNG images from volumetric models using headless wgpu rendering:

```bash
volumetric_cli render -i <file> -o <output.png>
```

**Arguments:**
- `-i, --input <file>` - Input file: either a `.wasm` model or a `.vproj` project file
- `-o, --output <file>` - Output PNG file path (view suffix added for multiple views)

**Options:**
- `--width <n>` - Image width in pixels (default: 1024)
- `--height <n>` - Image height in pixels (default: 1024)
- `--views <views>` - Comma-separated views: front, back, left, right, top, bottom, iso, iso-back, all (default: iso)
- `--background <hex>` - Background color as hex, e.g., 2d2d2d (default: 2d2d2d)
- `--color <hex>` - Mesh base color as hex, e.g., 6699cc (default: 6699cc)
- `--grid <spacing>` - Reference grid spacing in meters, 0 to disable (default: 1.0)
- `--grid-color <hex>` - Grid color as hex, e.g., 555555 (default: 555555)
- `--base-cell-size <f>`, `--max-depth <n>`, `--sharp-edges`, etc. - Same meshing options as the mesh command (sharp-edge flags currently no-op)
- `-q, --quiet` - Suppress profiling output

**Projection & Camera Options:**
- `--projection <type>` - Projection type: `perspective` or `ortho` (default: perspective)
- `--fov <degrees>` - Field of view for perspective projection (default: 45)
- `--ortho-scale <units>` - Orthographic vertical scale in world units (auto-computed if 0)
- `--camera-pos <x,y,z>` - Custom camera position (overrides --views)
- `--camera-target <x,y,z>` - Look-at point (default: model center)
- `--camera-up <x,y,z>` - Up vector (default: 0,1,0)
- `--near <distance>` - Near clipping plane distance (default: auto-computed)
- `--far <distance>` - Far clipping plane distance (default: auto-computed)

**Rendering Mode Options:**
- `--wireframe` - Render mesh edges instead of filled triangles
- `--wireframe-color <hex>` - Wireframe line color (default: ffffff)
- `--recalc-normals` - Recompute smooth normals from mesh geometry (useful for imported STLs with bad normals)

**Examples:**
```bash
# Single isometric view
volumetric_cli render -i model.wasm -o render.png

# Multiple views
volumetric_cli render -i model.wasm -o render.png --views front,iso,top

# All views at high resolution
volumetric_cli render -i model.wasm -o render.png --views all --width 2048 --height 2048

# Custom colors
volumetric_cli render -i model.wasm -o out.png --background ffffff --color 4488aa

# With 0.5m reference grid
volumetric_cli render -i model.wasm -o out.png --grid 0.5

# Without grid
volumetric_cli render -i model.wasm -o out.png --grid 0

# Orthographic projection (parallel lines don't converge)
volumetric_cli render -i model.wasm -o out.png --projection ortho

# Wide-angle perspective (90° FOV)
volumetric_cli render -i model.wasm -o out.png --fov 90

# Wireframe rendering
volumetric_cli render -i model.wasm -o out.png --wireframe --wireframe-color 00ff00

# Custom camera position
volumetric_cli render -i model.wasm -o out.png --camera-pos 5,3,5 --camera-target 0,0,0

# Combined: orthographic + wireframe + custom camera
volumetric_cli render -i model.wasm -o out.png --projection ortho --wireframe --camera-pos 10,10,10

# Close-up wireframe inspection with custom clip planes
volumetric_cli render -i model.wasm -o out.png --wireframe --camera-pos 0.5,0.5,0.5 --near 0.01 --far 10
```

The CLI outputs detailed profiling statistics showing per-stage timing and sample counts, useful for performance analysis.

#### Project Commands

Manage volumetric projects (`.vproj` files) that chain operators together:

```bash
# Create a new project from a model WASM
volumetric_cli project-new -i sphere.wasm -o scene.vproj

# Add another model to the project
volumetric_cli project-add-model -p scene.vproj -i torus.wasm

# Apply an operator (e.g., translate the first model)
volumetric_cli project-add-op -p scene.vproj --operator translate --input 0 --config '{"dx": 2.0, "dy": 0.0, "dz": 0.0}'

# List all assets in a project
volumetric_cli project-list -p scene.vproj

# Run the project DAG and show exported assets
volumetric_cli project-run -p scene.vproj

# Export a specific asset to a standalone WASM file
volumetric_cli project-export -p scene.vproj --asset 2 -o translated_sphere.wasm
```

#### Diagnostic Commands

Inspect WASM models and operators:

```bash
# Show model/operator info (ABI type, dimensions, bounds, metadata)
volumetric_cli info -i model.wasm

# Query bounds of a model
volumetric_cli bounds -i model.wasm

# Sample density at a specific point
volumetric_cli sample -i model.wasm --point 0.5,0.5,0.5
```

## Coordinate System and Units

The volumetric engine uses a **right-handed** coordinate system with the following conventions:

### Unit Scale
- **1 unit = 1 meter** - All coordinates in the `sample()` function are interpreted as meters
- The reference grid in the renderer defaults to 1-meter spacing
- When designing models, dimensions should be specified in meters (e.g., a 2-meter sphere has bounds from -1 to +1)

### Axis Orientation
- **+X**: Right
- **+Y**: Up
- **+Z**: Forward (toward camera in default view)

### Reference Grid
The CLI renderer includes an optional reference grid on the XZ plane at y=0 (or at the model's minimum Y if below zero):
- **Major lines**: Every 5 grid units (brighter)
- **Minor lines**: At the specified grid spacing (dimmer)
- Use `--grid 0` to disable the grid
- Default grid spacing is 1.0 meter

## Mesh Conventions

All mesh generation algorithms in this project follow these conventions:

### Winding Order
- **Counter-clockwise (CCW)** when viewed from outside the surface
- For a triangle with vertices A, B, C in CCW order, the face normal points outward
- Face normal is computed as: `normal = (B - A) × (C - A)`

### Normals
- All normals point **outward** from the solid (from inside toward outside)
- Per-vertex normals are used for smooth shading
- Since `sample()` returns binary values (not an SDF), normals are estimated by accumulating face normals from the mesh topology, then optionally refined via tangent-plane probing

### Renderer Expectations
- The GPU renderer uses `FrontFace::Ccw` with backface culling
- Triangles wound clockwise when viewed from outside will be culled (not rendered)

## Performance Note

While sampling a function via WASM for every voxel in a grid is computationally intensive, this project demonstrates that with efficient runtimes (like `wasmtime`) and optimized sampling strategies, it is a highly viable approach for flexible 3D modeling.
