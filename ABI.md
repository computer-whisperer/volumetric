# Volumetric ABI Specification

This document describes the Application Binary Interface (ABI) for WASM modules in the volumetric project, covering both **Models** (volumetric density functions) and **Operators** (transformations and generators).

## Table of Contents

1. [Model ABI](#model-abi)
2. [Operator ABI](#operator-abi)
3. [Data Types and Memory Layout](#data-types-and-memory-layout)

---

## Model ABI

Models are WASM modules that define a volumetric occupancy function. They are stateless and can be executed in parallel across multiple threads.

> **Note:** Volumetric models are **not** signed distance fields (SDFs). The `sample` function returns an occupancy/density value classified against `volumetric_abi::OCCUPANCY_THRESHOLD` (0.5): strictly greater is inside, anything else (including NaN) is outside. Models emit the canonical values `1.0`/`0.0`. The magnitude carries no geometric meaning — it is reserved for future density use cases, not distance to surface.

### Required Exports

| Export | Type | Signature | Purpose |
|--------|------|-----------|---------|
| `get_dimensions` | function | `() -> u32` | Return number of dimensions |
| `get_bounds` | function | `(out_ptr: i32)` | Write bounds to memory |
| `sample` | function | `(pos_ptr: i32) -> f32` | Sample density at position |
| `memory` | memory | — | Linear memory for I/O buffers |

### Optional Exports

| Function | Signature | Purpose |
|----------|-----------|---------|
| `get_metadata` | `() -> i64` | Catalog metadata, same format as the operator export (empty `inputs`/`outputs`). Read by host catalogs; never called during execution. Only authoritative for modules loaded from source files — pipeline outputs may carry a stale copy inherited from a wrapped input. |

### Function Semantics

#### `get_dimensions() -> u32`

Returns the number of dimensions for this model. Most models are 3D and return `3`.

#### `get_bounds(out_ptr: i32)`

Writes `2n` f64 values to memory at `out_ptr`, where `n` is the number of dimensions. The values are interleaved as `[min₀, max₀, min₁, max₁, ..., minₙ₋₁, maxₙ₋₁]`.

For a 3D model, this writes 48 bytes (6 × 8 bytes):
- Offset 0: `min_x` (f64)
- Offset 8: `max_x` (f64)
- Offset 16: `min_y` (f64)
- Offset 24: `max_y` (f64)
- Offset 32: `min_z` (f64)
- Offset 40: `max_z` (f64)

#### `sample(pos_ptr: i32) -> f32`

Reads `n` f64 values from memory at `pos_ptr` (where `n` is the number of dimensions), and returns a density value.

**Important: These are NOT signed distance fields (SDFs).**

For determining geometry and mesh boundaries, the **only** relevant information is the sign:
- `> 0.5` (`volumetric_abi::OCCUPANCY_THRESHOLD`, via `is_occupied`) → point is **inside** the solid (material present)
- anything else, including `NaN` → point is **outside** the solid (empty space)

The magnitude of the return value represents **material density**, not distance to surface. This density value:
- Has no geometric meaning for surface extraction
- Does not indicate proximity to the boundary
- Is reserved for future use cases (e.g., variable material properties, multi-material models)

Most models return binary values: `1.0` (solid) or `0.0` (empty).

For a 3D model, the host writes 24 bytes at `pos_ptr`:
- Offset 0: `x` (f64)
- Offset 8: `y` (f64)
- Offset 16: `z` (f64)

### Memory Layout

Models must export their linear memory. The host uses specific offsets for I/O buffers:

| Offset | Size | Purpose |
|--------|------|---------|
| 0 | n × 8 bytes | Position input buffer (host writes before `sample` call) |
| 256 | 2n × 8 bytes | Bounds output buffer (`get_bounds` writes here) |

Models may use memory beyond these reserved regions for their own data (e.g., BVH structures, heightmaps).

### Example Implementation (Rust)

```rust
// A unit sphere: returns 1.0 (inside) or 0.0 (outside)
const RADIUS: f64 = 1.0;

#[unsafe(no_mangle)]
pub extern "C" fn get_dimensions() -> u32 {
    3
}

#[unsafe(no_mangle)]
pub extern "C" fn get_bounds(out_ptr: i32) {
    let ptr = out_ptr as *mut f64;
    unsafe {
        *ptr.add(0) = -RADIUS;  // min_x
        *ptr.add(1) = RADIUS;   // max_x
        *ptr.add(2) = -RADIUS;  // min_y
        *ptr.add(3) = RADIUS;   // max_y
        *ptr.add(4) = -RADIUS;  // min_z
        *ptr.add(5) = RADIUS;   // max_z
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sample(pos_ptr: i32) -> f32 {
    let ptr = pos_ptr as *const f64;
    let (x, y, z) = unsafe { (*ptr, *ptr.add(1), *ptr.add(2)) };

    // Binary occupancy: 1.0 = inside, 0.0 = outside
    let inside = x * x + y * y + z * z <= RADIUS * RADIUS;
    if inside { 1.0 } else { 0.0 }
}
```

### Execution Model

1. Host loads WASM bytes and creates a wasmtime Module
2. Host instantiates the module and gets the exported memory
3. Host calls `get_dimensions()` to determine dimensionality
4. Host calls `get_bounds(256)` and reads bounds from memory offset 256
5. For each sample:
   - Host writes position coordinates to memory offset 0
   - Host calls `sample(0)` to get density value
6. For parallel execution, each thread maintains its own Store/Instance (Module is shared via `Arc<Module>`)

---

## Operator ABI

Operators are WASM modules that transform or generate models. They read inputs from the host and produce outputs.

### Host Imports (Module: `"host"`)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `get_input_len` | `(idx: i32) -> u32` | Get byte length of input at index |
| `get_input_data` | `(idx: i32, ptr: i32, len: i32)` | Copy input data to WASM memory |
| `get_input_count` | `() -> i32` | Optional: how many input slots the step carries (variadic operators size their input block from it; 0 while probing metadata) |
| `post_output` | `(output_idx: i32, ptr: i32, len: i32)` | Post output data to host |

### Required Exports

| Function | Signature | Purpose |
|----------|-----------|---------|
| `run` | `()` | Execute the operator |
| `get_metadata` | `() -> i64` | Return operator metadata |

### Metadata Format

`get_metadata()` returns a packed `i64`:
- Bits 0-31: Pointer to CBOR data in linear memory
- Bits 32-63: Length of CBOR data

The CBOR data encodes an `OperatorMetadata` struct (authoritative
definition and field docs in `crates/volumetric_abi`):

```rust
struct OperatorMetadata {
    name: String,
    version: String,
    // Display fields for host catalogs; all serde-defaulted, so metadata
    // from modules built before they existed decodes with them empty.
    display_name: String,       // catalog name; hosts fall back to `name`
    description: String,        // one-line catalog summary
    category: String,           // free-form catalog grouping
    icon_svg: String,           // monochrome SVG (24×24, currentColor)
    inputs: Vec<OperatorMetadataInput>,
    variadic_input: Option<usize>, // serde-defaulted: index of the one slot that
                                // accepts one or more inputs; a step's extra
                                // inputs sit at that position and later slots
                                // shift along (helpers on the struct do the math)
    input_names: Vec<String>,   // labels parallel to `inputs`
    input_constraints: Vec<InputConstraint>, // serde-defaulted, parallel to
                                // `inputs`: what a slot takes beyond its kind
                                // (see "Input constraints" below)
    outputs: Vec<OperatorMetadataOutput>,
    output_names: Vec<String>,  // labels parallel to `outputs` (serde-defaulted);
                                // hosts label output slots and derive default
                                // asset ids from them for multi-output operators
}

enum OperatorMetadataInput {
    ModelWASM,
    CBORConfiguration(String),  // CDDL schema
    LuaSource(String),          // Template script
    Blob,                       // Raw binary data
    VecF64(usize),              // N f64 values
    FeaMesh,                    // CBOR FEA mesh (explicit data)
    TriMesh,                    // CBOR triangle mesh (explicit data)
    Subspace,                   // CBOR affine subspace (explicit data)
    ViewSet,                    // CBOR posed images (explicit data)
    Splat,                      // CBOR Gaussian splat (explicit data)
    Mechanism,                  // CBOR joint tree (explicit data)
    Assembly,                   // CBOR mechanism + part models + state (explicit data)
}

enum OperatorMetadataOutput {
    ModelWASM,
    FeaMesh,
    TriMesh,
    Subspace,
    F64Map,
    ViewSet,
    Splat,
    Mechanism,
    Assembly,
}
```

```rust
struct InputConstraint {        // every field serde-defaulted
    subspace: SubspaceShape,    // for a Subspace slot; ignored on any other
}

struct SubspaceShape {          // every bound optional; all absent = any
    ambient: Option<u32>,       // the n-space the subspace must live in
    min_rank: Option<u32>,      // lowest rank taken
    max_rank: Option<u32>,      // highest rank taken
}
```

`icon_svg` is a complete SVG document: 24×24 viewBox, lucide-style 2px
round strokes, every paint treated as `currentColor` so hosts tint it
like a built-in icon. The `volumetric_abi::icon_svg!` macro wraps bare
shape elements in the canonical header (usable in `const` position, and
in the build scripts no_std models precompute their metadata CBOR from).
Hosts fall back to a per-category stock glyph when it is empty or fails
to parse — a bad icon never breaks cataloging.

#### Input constraints

A slot's kind says how its value is encoded. `input_constraints[i]` says
which values of that kind slot `i` takes, so a host can describe the slot
and offer the right tool for it before anything is wired: `brim_operator`'s
bed is `{ ambient: 3, min_rank: 2, max_rank: 2 }`, a plane in 3-space, and
`mechanism_operator`'s axes are `{ ambient: 3, min_rank: 1 }`, a line or
more. Rank 0 is a point, 1 a line, 2 a plane.

- The list is parallel to `inputs` and may be short or absent; a slot with
  no entry declares nothing. The variadic slot's entry covers every value
  wired into its block.
- It is a promise to the host, not a check the host makes. The operator
  still refuses a value it cannot use, with `SubspaceShape::check` against
  the same constant it declares, so the declaration and the check cannot
  drift and a wrong shape is worded alike everywhere ("expected plane in
  3-space, got line in 3-space").
- It is an outer bound. A value outside the shape is refused when the
  operator comes to use it; a value inside may still be refused for a
  reason the shape cannot say. `cloud_fit_operator`'s seed is declared as
  rank 0 to 2, and a plane is refused when the fit is of a cylinder.
- An unwired slot (empty bytes) is the operator's to accept or refuse, as
  before: a declared shape says nothing about whether the slot is optional.
- A shape that depends on another input cannot be declared.
  `extrude_operator`'s plane has one rank less than the space its profile
  is lifted into, so it declares nothing and checks at run time;
  `slice_operator` declares the rank it can (1 or more) and checks the
  ambient dimension against its model.
- Compatibility: metadata without the field decodes with nothing declared,
  a host built before the field skips it, and an operator that declares
  nothing does not write it. `OperatorMetadata::subspace_shape(idx, count)`
  reads the shape for a step input.

### Operator Categories

**Transform Operators** (translate, scale, rotate, boolean):
- Take model WASM as input
- Parse and modify the WASM to wrap ABI functions
- Output transformed model WASM
- Must preserve the model ABI (get_dimensions, get_bounds, sample, memory)

**Generator Operators** (rectangular_prism, stl_import, heightmap_extrude, lua_script):
- Take configuration/data as input
- Generate complete model WASM from scratch
- Output new model WASM implementing the full model ABI

### Transform Operator Pattern

Transform operators typically:
1. Parse the input model WASM using walrus or wasm-encoder
2. Rename existing ABI functions with a unique suffix (e.g., `sample` → `sample_abc123`)
3. Generate new wrapper functions that:
   - Read position from memory
   - Apply inverse transform to position
   - Write transformed position to a scratch buffer
   - Call the original function
   - (For bounds) Apply forward transform to the result
4. Export the new wrapper functions with the standard ABI names
5. Pass through the memory export from the input model

---

## Data Types and Memory Layout

### Primitive Types

| Type | Size | Encoding |
|------|------|----------|
| `f64` | 8 bytes | IEEE 754 double, little-endian |
| `f32` | 4 bytes | IEEE 754 single, little-endian |
| `i32` | 4 bytes | Two's complement, little-endian |
| `u32` | 4 bytes | Unsigned, little-endian |

### Composite Types

- **Position buffer**: `n × 8` bytes of contiguous little-endian f64 values
- **Bounds buffer**: `2n × 8` bytes, interleaved as `[min₀, max₀, min₁, max₁, ...]`
- **CBOR**: Variable-length binary encoding (ciborium library)
- **WASM**: Standard WebAssembly binary format

### Reserved Memory Offsets

| Offset | Purpose |
|--------|---------|
| 0-255 | Position I/O buffer (host writes position here) |
| 256-511 | Bounds I/O buffer (get_bounds writes here) |
| 512+ | Available for model/operator data |

Transform operators use offset 512 as a scratch buffer for transformed positions.
