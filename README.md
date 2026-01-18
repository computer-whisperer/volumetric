# Volumetric

A high-performance volumetric modeling engine where models and operations are defined as portable WebAssembly (WASM) modules.

## Core Concept

The project explores a "model-as-code" paradigm where a 3D physical model is defined by a simple function: `is_inside(x: f64, y: f64, z: f64) -> f32` returning a density value. By leveraging WebAssembly, we can define complex, parametric models that are:
- **Portable**: Run anywhere with a WASM runtime.
- **Fast**: Near-native execution speed.
- **Composable**: Models can be transformed and combined using Operator modules that manipulate WASM bytecode.

## Project Structure

- `src/`: The host application (Orchestrator). Built with Rust, `wasmtime` for execution, and `egui` for the UI.
- `crates/models/`: Example model definitions (Sphere, Torus, Mandelbulb, etc.).
- `crates/operators/`: Modules that transform or combine models (Translate, Boolean Union/Subtract/Intersect).

## Architecture

The system is divided into three main components:

### 1. Model WASM Modules
A Model module is a WASM artifact that exports:
- `is_inside(x: f64, y: f64, z: f64) -> f32`: Returns a density value indicating whether the point is inside the model.
- `get_bounds_min_x/y/z() -> f64` & `get_bounds_max_x/y/z() -> f64`: Defines the axis-aligned bounding box.

**Important: `is_inside` is NOT a Signed Distance Function (SDF).** Current demo models return binary values (`1.0` for inside, `0.0` for outside). Future models may use continuous densities for effects like soft blending, but the return value represents *occupancy/density*, not distance to the surface. This means:
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
- `--base-resolution <n>` - Coarse grid resolution (default: 8)
- `--max-depth <n>` - Refinement depth (default: 4). Effective resolution = base × 2^depth
- `--vertex-refinement <n>` - Vertex position refinement iterations (default: 12)
- `--normal-refinement <n>` - Normal estimation iterations (default: 12, use 0 to disable)
- `--normal-epsilon <f>` - Normal probe distance as fraction of cell size (default: 0.1)
- `-q, --quiet` - Suppress profiling output

**Examples:**
```bash
# Mesh a WASM model with default settings (128³ effective resolution)
volumetric_cli mesh -i simple_torus_model.wasm -o torus.stl

# Faster meshing with lower resolution and no normal refinement
volumetric_cli mesh -i model.wasm -o output.stl --max-depth 3 --normal-refinement 0

# Mesh a project file
volumetric_cli mesh -i scene.vproj -o scene.stl
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
- `--background <hex>` - Background color as hex, e.g., f0f0f0 (default: f0f0f0)
- `--color <hex>` - Mesh base color as hex, e.g., 6699cc (default: 6699cc)
- `--base-resolution <n>`, `--max-depth <n>`, etc. - Same meshing options as the mesh command
- `-q, --quiet` - Suppress profiling output

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
```

The CLI outputs detailed profiling statistics showing per-stage timing and sample counts, useful for performance analysis.

## Mesh Conventions

All mesh generation algorithms in this project follow these conventions:

### Winding Order
- **Counter-clockwise (CCW)** when viewed from outside the surface
- For a triangle with vertices A, B, C in CCW order, the face normal points outward
- Face normal is computed as: `normal = (B - A) × (C - A)`

### Normals
- All normals point **outward** from the solid (from inside toward outside)
- Per-vertex normals are used for smooth shading
- Since `is_inside()` is binary (not an SDF), normals are estimated by accumulating face normals from the mesh topology, then optionally refined via tangent-plane probing

### Renderer Expectations
- The GPU renderer uses `FrontFace::Ccw` with backface culling
- Triangles wound clockwise when viewed from outside will be culled (not rendered)

## Performance Note

While sampling a function via WASM for every voxel in a grid is computationally intensive, this project demonstrates that with efficient runtimes (like `wasmtime`) and optimized sampling strategies, it is a highly viable approach for flexible 3D modeling.