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
- `crates/operators/`: Modules that transform or combine models. Includes **Transform Operators** (translate, scale, rotation, boolean) and **Generator Operators** (rectangular_prism, stl_import, threemf_import, heightmap_extrude, lua_script).

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

Operators are grouped by the kind of data they transform or produce:

### Transform Operators
Transform operators take an existing model and produce a modified version:

| Operator | Description | Configuration |
|----------|-------------|---------------|
| `translate` | Moves a model in 3D space | `{dx, dy, dz}` - displacement in each axis |
| `scale` | Scales a model uniformly or non-uniformly | `{sx, sy, sz}` - scale factors |
| `rotation` | Rotates a model about the origin | `{rx_deg, ry_deg, rz_deg}` - Euler angles in degrees, applied X then Y then Z |
| `pattern` | Repeats a model as mirrored, linear and circular copies in one single-memory step | `{mirror, linear, circular}` - optional blocks, composed in that order |
| `pose` | Scales and rotates a model about a pivot (origin, bounds centre, or a point), then moves it | `{pivot, scale, rotate, translate}` - optional blocks |
| `boolean` | Union, intersection, or subtraction of one or more models (one flat step, not a chain) | `{op}` - "union", "subtract", or "intersect" |
| `coil` | Rolls a flat model into an Archimedean spiral around the y axis (x becomes arc length, z radial depth) | `{inner_radius, gap}` - bore radius and inter-wrap clearance |

### Generator Operators
Generator operators create new models from configuration or external data:

| Operator | Description | Inputs |
|----------|-------------|--------|
| `rectangular_prism` | Creates a box shape | CBOR config: `{width, height, depth}` |
| `stl_import` | Reads an STL file as an explicit triangle mesh (`TriMesh`; chain `mesh_to_model` for a solid) | STL blob + CBOR config: `{scale, translate, center}` |
| `threemf_import` | Reads a 3MF package as a triangle mesh in metres (file unit honoured; build items, components, and cross-part `p:path` references resolved) | 3MF blob + CBOR config: `{scale, center, item}` |
| `ply_import` | Reads a PLY mesh (ASCII or binary) as a triangle mesh; vertex colours and normals ride along as the `color` and `normal` vertex fields | PLY blob + CBOR config: `{scale, center, fields}` |
| `point_cloud_import` | Reads a point-cloud file (PLY) as a Point1 cloud with `color` and `normal` node fields, for the point operators and the coloured cloud preview | PLY blob + CBOR config: `{scale, center, stride, fields}` |
| `volume_import` | Reads a sampled volume (NRRD: a TSDF, a CT threshold, any regular grid) as a solid with the `signed_distance` channel Generate SDF bakes; NaN samples are unobserved and fill in where the scan encloses them | NRRD blob + CBOR config: `{scale, center, threshold, inside, band, unobserved, crop, stride}` |
| `cloud_fit` | Fits a plane, line, point, sphere or cylinder to a point cloud (RANSAC + least squares) as a Subspace for Slice/Extrude/Revolve/Span/Intersect, plus an F64Map of radius, extents, inliers and rms; an optional seed Subspace narrows the search or is refined | FeaMesh + optional Subspace + CBOR config: `{kind, normal, tolerance, search_radius, trials}` |
| `cloud_normals` | Estimates a unit normal per node from its nearest neighbours and stores it as the `normal` field (what a cylinder fit needs) | FeaMesh + CBOR config: `{neighbours, orient}` |
| `heightmap_extrude` | Extrudes a heightmap image to 3D | Image blob + CBOR config: `{width, depth, height, clip}` |
| `lua_script` | Custom occupancy model via restricted Lua | Lua source + optional routed `F64Map` parameters |
| `path_sketch` | Fills SVG path data (lines, curves, arcs, holes) as a 2D sketch for extrude/revolve | CBOR config: `{path, flip_y, round, chord_tolerance}` |

### Data Operators

Data operators compose typed values in the project DAG without invoking the model evaluator:

| Operator | Description | Inputs |
|----------|-------------|--------|
| `f64_map_merge` | Composes shared numeric project data; later maps override earlier keys | Any number of routed `F64Map` inputs (one variadic slot) |

The construction catalog also includes `subspace_fit`, which locates a local point, edge line,
or tangent plane from a seed using bounded model probes and optional metric-grid snapping.

The analysis catalog includes `sdf`, which preserves a model's occupancy while adding a
world-space truncated signed-distance channel. Its configured band extends outside the part
bounds, and the positive clamp value remains defined globally beyond the sampled field.

Demand-driven Lua examples live in `examples/`. `fidget_spinner.lua` exercises compact radial
parts, while `raspberry_pi_4_tray.lua` exercises routed mechanical clearances, thin walls,
mounting standoffs, irregular connector openings, and repeated ventilation features.
`raspberry_pi_4_fit.vproj` places a vendored Pi 4B reference assembly in that tray and exports
the tray, aligned board, combined assembly, and their intersection. The `fit_interference`
export is intended as a clearance check and meshes to an empty surface at the documented
alignment.

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
cargo run -p volumetric_ui_v2 --release
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

Generate STL or 3MF meshes from volumetric models:

```bash
volumetric_cli mesh -i <file> -o <output.stl>
volumetric_cli mesh -i <file> -o <output.3mf> --unit mm
```

**Arguments:**
- `-i, --input <file>` - Input file: either a `.wasm` model or a `.vproj` project file
- `-o, --output <file>` - Output mesh path; a `.3mf` extension writes a 3MF package, anything else binary STL

**Options:**
- `--unit <m|mm|cm|in>` - Output length unit (default: `m`). Geometry is in metres; coordinates are scaled to this unit, and a 3MF is labelled with it so slicers open it at true size. STL carries no unit and slicers read it as millimetres, so `--unit mm` exports true size there too
- `--base-resolution <n>` - Coarse grid resolution (default: 8)
- `--max-depth <n>` - Refinement depth (default: 4). Effective resolution = base × 2^depth
- `--vertex-refinement <n>` - Vertex position refinement iterations (default: 12)
- `--normal-refinement <n>` - Normal estimation iterations (default: 12, use 0 to disable)
- `--normal-epsilon <f>` - Normal probe distance as fraction of cell size (default: 0.1)
- `--sharp-edges` - Enable sharp edge detection and vertex duplication for hard edges
- `--sharp-angle <degrees>` - Angle threshold for sharp edge detection (default: 20)
- `--sharp-residual <f>` - Residual multiplier for sharp edge clustering (default: 4.0)
- `-q, --quiet` - Suppress profiling output

**Examples:**
```bash
# Mesh a WASM model with default settings (128³ effective resolution)
volumetric_cli mesh -i simple_torus_model.wasm -o torus.stl

# Faster meshing with lower resolution and no normal refinement
volumetric_cli mesh -i model.wasm -o output.stl --max-depth 3 --normal-refinement 0

# Mesh a project file
volumetric_cli mesh -i scene.vproj -o scene.stl

# A slicer-ready 3MF, labelled in millimetres
volumetric_cli mesh -i scene.vproj -o scene.3mf --unit mm

# High-quality mesh with sharp edge preservation (for CAD-like models)
volumetric_cli mesh -i box.wasm -o box.stl --sharp-edges --sharp-angle 30
```

#### Render Command

Draw a model, or the exports of a project, to PNG through the same preview
path as the GUI viewport: 3D models are meshed with the adaptive surface
nets plan, 2D sketches raster to flat quads, point clouds draw as coloured
points, FEA and triangle meshes as their explicit data, and Subspace values
as gizmos sized by the scene.

```bash
volumetric_cli render -i <model.wasm | project.vproj> -o <output.png>
```

**Scene:**
- `--asset <id>` - Draw only this asset (repeatable; default: every renderable export; an import such as a view set draws only when named)
- `--resolution <N>` - Meshing resolution for models and raster size for sketches (default: 128)
- `--no-sharp`, `--no-simplify` - Mesh without sharp-feature reconstruction or decimation
- `--color-channel <name>` - Colormap models by a declared sample channel
- `--color-field node:<name>` - Colormap FEA meshes and point clouds by a field
- `--wireframe` - Overlay mesh edges
- `--grid <m>` - Ground grid spacing in metres (default: 1.0; 0 disables)
- `--no-ssao` - Disable ambient occlusion
- `--background <hex>` - Background colour (default: 2d2d2d)
- `--width`, `--height` - Image size (default: 1024 x 1024)

**Camera** (one of):
- `--views <list>` - Preset directions framed to the scene: front, back, left, right, top, bottom, iso, iso-back, all (default: iso; several views write one file each, suffixed)
- `--camera-pos x,y,z [--camera-target x,y,z] [--camera-up x,y,z] [--fov deg]` - An explicit pose
- `--intrinsics fx,fy,cx,cy --pose m00,...,m23` - A pinhole camera in OpenCV convention (pixel origin top-left, camera z forward); the pose is the rows of its 3x4 camera-to-world matrix and the image is `--width` x `--height`
- `--through <view id>` (or `<views asset>:<view id>`) - The camera of a view in the project's view set, at the view's image size unless `--width`/`--height` scale it
- `--overlay blend|edge|side|checker [--overlay-alpha a] [--overlay-tile px]` - With `--through`, composite the render over the view's photograph
- `--projection ortho [--ortho-scale h]` - Orthographic instead of perspective (presets and poses)
- `--near`, `--far` - Clip planes (default: from the scene)

**Examples:**
```bash
# Every export of a project from the isometric preset
volumetric_cli render -i chair.vproj -o chair.png

# Three views of a scan and a fitted axis
volumetric_cli render -i chair.vproj --asset scan --asset lift_axis -o chair.png --views front,top,iso

# Through one of the scan's rectified photographs (960 x 960)
volumetric_cli render -i chair.vproj -o view.png --width 960 --height 960 \
    --intrinsics 414.8,415.1,484.2,488.6 \
    --pose 0.726,-0.475,0.497,0.881,-0.014,-0.733,-0.681,-1.031,0.687,0.487,-0.538,-2.186

# The same through the project's view set, blended over the photograph
volumetric_cli render -i chair.vproj -o view.png --through 00600_l --overlay blend
```

#### View Set Commands

Posed images with depth are evidence a model can be checked against. A
view set (`ViewSet` asset, `.vviews` file) carries cameras, per-view
poses in the OpenCV convention, the embedded photograph, 16-bit depth map
and subject mask of each view, the marker map that posed them, and a
provenance record.

```bash
# Embed a selection of a dataset: the scanner's cameras.json or nerfstudio's transforms.json
volumetric_cli view-import --manifest datasets/chair/cameras.json -p chair.vproj \
    --eye left --near 1.8,-1.9,-3.0 --radius 1.4 --stride 60 --max 8 --field chair-markers

# Or write a standalone file, by view id
volumetric_cli view-import --manifest cameras.json -o views.vviews --id 00600_l --id 00003_l

# What a set holds
volumetric_cli view-list -i chair.vproj [--asset views] [--json]

# How far the model's surface sits from each view's depth map: coverage,
# median and p90 residual per view and pooled, with residual images
volumetric_cli view-residual -p chair.vproj --model chair_solid -o residuals/ [--json]

# Pose a phone still from the marker cards it shows, against the set's map,
# and append it to the set (focal and first radial term solved when the
# intrinsics are unknown; --annotate draws the detections)
volumetric_cli view-solve -p chair.vproj --image IMG_0042.jpg [--intrinsics fx,fy,cx,cy[,k1]] \
    [--fov-deg 70] [--annotate cards.png] [--dry-run] [--json]

# Find the swatches, the survey card's tags and its interior corners in every
# picture of a set and store them on the views (the survey solves from them);
# or report on loose pictures with --image
volumetric_cli view-detect -p chair.vproj [--views set] [--card card.json | --no-card] \
    [--dictionary 5x5_100|4x4_50|none] [--search-px 1600] [--annotate dir/] [--dry-run] [--json]
volumetric_cli view-detect --image DSC00123.JPG [--image ...] --json
```

`view-solve` detects ArUco markers (`5x5_100` swatches or the `4x4_50`
board) with sub-pixel corners, solves the pose against the map with a
robust fit, and reports per-marker residuals, the focal with its standard
error, and warnings when the pose rests on one card, the focal is weakly
constrained (all cards on one plane seen square-on), or the residual says
the map belongs to another setup. On the chair-phone stills it agrees with
COLMAP to 1.3 cm median (3 cm worst) and 0.3 degrees, at 0.6 s per 12 MP
picture.

The same solve is the `view_solve_operator` (Solve Still), so a still
can be a project step: inputs are the view set, the picture blob and a
config (`id`, `dictionary`, `focal_px`, `k1`, `fov_deg`, `solve_focal`,
`solve_distortion`, `embed_image`); the output is the set with the posed
view appended, and the step's warnings carry the solve's summary line and
its cautions. In the GUI, Solve Still in the Add catalog (or Import Still)
opens a file dialog, adds the step wired to the project's view set, and
after a run the Views section lists the still with its `rms` and `cards`
tags, ready to look through.

`view-detect` is the survey's first stage. It reads every family in one
pass (the `5x5_100` swatches and the survey card's AprilTag `36h11` tags;
`4x4_50` for the calibration board), searching for quads on the picture
reduced to about 1600 px and reading every corner and cell at full
resolution; the card's interior corners are placed from the decoded tags
beside them (or from the board as a whole where none is) and settled on
the saddle point of the grey picture, and refused where the four squares
around them do not show the chessboard; marker corners come from robust
quadratic fits of each edge, so a curled card or the lens bending an edge
does not shift them; edge blur is measured across the markers as a
frame-quality number. Each view gets `observations` (markers with their
fit residual, card corners with the distance the refinement moved them,
the blur), and the set records the card spec. The default card is the
survey card; `--card` reads either this crate's spec or the scanner's
`card.json`. On the 79 stills of the chairbase-dslr-0 session it finds
every card corner OpenCV's ChArUco detector found and 69 % more,
agreeing with it to 0.13 px median; swatch corners agree to 1 px, with
half of that OpenCV's own inward bias on blurred corners; a 26 MP still
takes 0.2 s. The same detection is the `view_detect_operator` (Detect
Cards): view set and config in, view set with observations out, one
report line per view in the step's warnings. In the GUI a detected view's
row shows its corner or marker count, and looking through the view draws
the observations over the photograph: swatches amber, tags cyan, card
corners as magenta crosses.

```bash
volumetric_cli project-add-asset -p chair.vproj -i IMG_0042.jpg --type blob --asset-id still_42
volumetric_cli project-add-op -p chair.vproj --operator view_solve_operator \
    -i asset:views -i asset:still_42 -i 'json:{"id":"phone_42"}' --output-id views_with_still
```

Selection flags: `--id` (repeatable), `--stride`, `--near x,y,z --radius r`,
`--max`, `--eye left|right|both`, `--split train|test`, `--no-images`,
`--no-depth`, `--no-masks`; provenance labels `--session`, `--rig`,
`--field`, `--setup`. `project-add-asset --type viewset` adds a `.vviews`
file to a project.

A view set in the viewport draws every view's frustum (0.1 m deep, with a
tick marking the picture's up) and the marker squares, and the project
panel's Views section lists the views with thumbnails, tags and depth or
mask marks. The eye button looks through a view: the viewport takes the
view's camera, letterboxed to its aspect, with the photograph over the
frame at a chosen opacity (0–100%) and that frustum highlighted. Orbiting,
panning or zooming leaves the view and continues from its viewpoint; Reset
or the Leave button leaves it too. Headlessly, `render --asset views` draws
the same frustums beside whichever exports are named (imports draw only
when asked for by id).

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
