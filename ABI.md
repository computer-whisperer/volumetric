# Volumetric ABI Specification

This document describes the Application Binary Interface (ABI) for WASM modules in the volumetric project, covering both **Models** (volumetric density functions) and **Operators** (transformations and generators).

## Table of Contents

1. [Model ABI](#model-abi)
2. [Operator ABI](#operator-abi)
3. [Data Types and Memory Layout](#data-types-and-memory-layout)

---

## Model ABI

Models are WASM modules that define a volumetric occupancy function. They are stateless and can be executed in parallel across multiple threads.

> **Note:** Volumetric models are **not** signed distance fields (SDFs). The `sample` function returns an occupancy/density value where only the sign matters for geometry extraction (`> 0` = inside, `<= 0` = outside). The magnitude represents material density for future use cases, not distance to surface.

### Required Exports

| Export | Type | Signature | Purpose |
|--------|------|-----------|---------|
| `get_dimensions` | function | `() -> u32` | Return number of dimensions |
| `get_bounds` | function | `(out_ptr: i32)` | Write bounds to memory |
| `sample` | function | `(pos_ptr: i32) -> f32` | Sample density at position |
| `memory` | memory | — | Linear memory for I/O buffers |

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
- `> 0.0` → point is **inside** the solid (material present)
- `<= 0.0` → point is **outside** the solid (empty space)

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

The CBOR data encodes an `OperatorMetadata` struct:

```rust
struct OperatorMetadata {
    name: String,
    version: String,
    inputs: Vec<OperatorMetadataInput>,
    outputs: Vec<OperatorMetadataOutput>,
}

enum OperatorMetadataInput {
    ModelWASM,
    CBORConfiguration(String),  // CDDL schema
    LuaSource(String),          // Template script
    Blob,                       // Raw binary data
    VecF64(usize),              // N f64 values
}

enum OperatorMetadataOutput {
    ModelWASM,
}
```

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
