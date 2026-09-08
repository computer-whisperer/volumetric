# Comet racer

An original, chunky open-wheel toy race car, authored from scratch in
`car.wgsl`. Approximately 102 mm long, 62 mm wide and 38 mm tall, with
a 60 mm wheelbase and 25 mm diameter wheels. This is a static display
solid with attached wheels, not a rolling assembly.

Features: tapered rounded monocoque, open cockpit with seat and steering
wheel, rear roll hoop, engine cooling slots, twin recessed hood stripes,
sidepods, front splitter, rear diffuser, wing with endplates, and four
rounded tyres with circumferential grooves and six-spoke hubs.

The geometry uses analytic occupancy functions compiled by the engine's
`wgsl_script_operator`; no imported meshes or baked distance fields.
Coordinates inside the design are millimetres, converted at the ABI
boundary to metres. +x points toward the nose, +y is up, and z spans
the axles. The annotated `comet.scale` override scales the entire car.
The WGSL model dialect uses f64 precision with `scene`, `bounds_min`,
and `bounds_max` entry functions.
To build at another scale, replace the operator's `json:{}` input in
`build.sh` with, for example, `json:{"comet.scale":0.5}`.

From the repository root:

```sh
cargo build --release -p volumetric_cli
bash examples/comet_racer/build.sh
```

The build recreates `comet_racer.vproj` and exports `car.wasm`, `car.stl`
and four PNG views under `target/comet_racer/`. `car_iso-back.png` shows
the nose and cockpit. `car_front.png` is the side view in engine axes.
The STL is in metres: use a 1000x import scale in a millimetre-based slicer.
Meshing uses 384³ resolution with simplification disabled; sharp-edge
reconstruction is used only for the preview images.

The final STL was checked after welding identical vertex coordinates:
1,892,224 triangles, one connected component, zero boundary edges, zero
non-manifold edges, and zero zero-area triangles. Its measured bounds are
102.0 x 38.0 x 61.6 mm in engine x/y/z order. Occupancy probes also check
that the cockpit is empty above its solid floor and the steering rim exists.
The WGSL migration preserves exactly the same vertex coordinates and
triangles as the original Lua export (ignoring triangle ordering). The
routed half-scale override was also checked for bounds and occupancy.

Dogfooding observations:

- Analytic rounding and a shared wheel function make this compact to
  author without repeatedly baking an offset field.
- A mesh-health CLI command would make checking topology and degenerate
  faces a normal part of this build script.
- The preview CLI has a single base color. Per-part colors would help
  distinguish tyres, bodywork and cockpit details.
- Sharp previews show small crease artifacts; inspect the smooth STL
  separately when evaluating the export.
