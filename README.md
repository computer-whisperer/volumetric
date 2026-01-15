Here is the basic idea:

Suppose we define a physical 3d model as a simple wasm function. This function will take in a 3d point and return a simple boolean indicating whether the point is inside the model or not.

This is attractive for a few reasons, but the main potential flaw is that sampling the model will be expensive. The question is, how slow? Given the reality of many projects that could benefit from a simpler model definition paradigm, and how cheap compute is in reality, is this viable?


Therefore, here we will be creating the MVP. We will need to divide this into two parts: a wasm module that defines the model (currently should be defined in crates/test_model) and a root module that will a) render the model and b) convert the model to various other 3d formats.

## Demo models

This repo includes a few small WASM demo model crates (one crate per demo), each exporting:

- `is_inside(x: f32, y: f32, z: f32) -> i32` (1 = inside, 0 = outside)
- `get_bounds_min_x/y/z() -> f32`
- `get_bounds_max_x/y/z() -> f32`

Current demos:

- `simple_sphere_model` (sphere)
- `simple_torus_model` (torus)
- `rounded_box_model` (rounded box)
- `gyroid_lattice_model` (gyroid lattice shell)
- `mandelbulb_model` (3D fractal / mandelbulb)

### Building demo WASM artifacts

You need the WASM target installed:

```bash
rustup target add wasm32-unknown-unknown
```

Build all demos (release):

```bash
cargo build --release --target wasm32-unknown-unknown -p simple_sphere_model -p simple_torus_model -p rounded_box_model -p gyroid_lattice_model -p mandelbulb_model
```

The app looks for demo outputs at:

- `target/wasm32-unknown-unknown/release/<crate>.wasm` (preferred)
- `target/wasm32-unknown-unknown/debug/<crate>.wasm` (fallback)

### Using demos in the UI

Run the app, then in the left panel:

1. Pick a demo under "Demo"
2. Click "Load demo"

You can still load any external `.wasm` file via "Load WASM…".