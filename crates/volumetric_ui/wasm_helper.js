// WASM Helper for nested WASM module execution
// This file provides JavaScript functions for instantiating and calling WASM model modules
// from within the main volumetric_ui WASM module.
//
// Models use the N-dimensional ABI:
//   get_dimensions() -> u32
//   get_io_ptr() -> i32        -- model-owned IO buffer (>= 2 * dims f64s)
//   get_bounds(out_ptr: i32)   -- writes interleaved min/max f64 pairs
//   sample(pos_ptr: i32) -> f32
//   memory export
//
// The host writes positions into (and reads bounds from) the buffer the model
// returns from get_io_ptr, so the model's own layout decides where it lives.

// Map of handle -> WASM instance
const wasmInstances = new Map();
let nextHandle = 1;

// Create a model WASM instance from bytes (synchronous, for use from WASM)
// Returns a handle (number) that can be used with other functions, 0 on error
window.wasmModelCreateSync = function(bytes) {
    try {
        const module = new WebAssembly.Module(bytes);
        const instance = new WebAssembly.Instance(module, {});
        const exports = instance.exports;
        if (!exports.get_dimensions || !exports.get_io_ptr || !exports.get_bounds
            || !exports.sample || !exports.memory) {
            console.error("WASM model does not export the N-dimensional ABI");
            return 0;
        }
        const dimensions = exports.get_dimensions();
        const ioPtr = exports.get_io_ptr();
        if (ioPtr <= 0 || ioPtr + dimensions * 2 * 8 > exports.memory.buffer.byteLength) {
            console.error("WASM model returned an invalid IO buffer pointer:", ioPtr);
            return 0;
        }
        const handle = nextHandle++;
        wasmInstances.set(handle, { instance, module, dimensions, ioPtr });
        return handle;
    } catch (e) {
        console.error("Failed to create WASM model:", e);
        return 0;
    }
};

// Get the number of dimensions of a model, 0 on error
window.wasmModelGetDimensions = function(handle) {
    const entry = wasmInstances.get(handle);
    if (!entry) {
        console.error("Invalid WASM handle:", handle);
        return 0;
    }
    return entry.dimensions;
};

// Get the bounding box of a model
// Returns interleaved [min_0, max_0, min_1, max_1, ...] (2 * dims) or null on error
window.wasmModelGetBounds = function(handle) {
    const entry = wasmInstances.get(handle);
    if (!entry) {
        console.error("Invalid WASM handle:", handle);
        return null;
    }

    try {
        const exports = entry.instance.exports;
        exports.get_bounds(entry.ioPtr);
        const view = new Float64Array(
            exports.memory.buffer,
            entry.ioPtr,
            entry.dimensions * 2,
        );
        // Copy: the view aliases WASM memory, which may move on growth
        return new Float64Array(view);
    } catch (e) {
        console.error("Failed to get bounds:", e);
        return null;
    }
};

// Sample the density at a point (extra dimensions are zeroed)
// Returns the density value, or NaN on error
window.wasmModelSample = function(handle, x, y, z) {
    const entry = wasmInstances.get(handle);
    if (!entry) {
        console.error("Invalid WASM handle:", handle);
        return NaN;
    }

    try {
        const exports = entry.instance.exports;
        const pos = new Float64Array(
            exports.memory.buffer,
            entry.ioPtr,
            entry.dimensions,
        );
        pos.fill(0);
        pos[0] = x;
        if (entry.dimensions > 1) pos[1] = y;
        if (entry.dimensions > 2) pos[2] = z;
        return exports.sample(entry.ioPtr);
    } catch (e) {
        console.error("Failed to sample:", e);
        return NaN;
    }
};

// Destroy a model instance and free resources
window.wasmModelDestroy = function(handle) {
    wasmInstances.delete(handle);
};

// ============================================================================
// Operator WASM support
// Operators are more complex - they require host function imports for I/O
// ============================================================================

// Map of handle -> operator instance with state
const operatorInstances = new Map();
let nextOperatorHandle = 1;

// Create an operator WASM instance with inputs
// inputs: Array of Uint8Array (input data for each slot)
// Returns a handle or 0 on error
window.wasmOperatorCreate = function(bytes, inputs) {
    try {
        // Store inputs for this operator instance
        const state = {
            inputs: inputs || [],
            outputs: new Map(),
            instance: null,
            module: null,
            error: null,
        };

        // Create host imports that close over the state
        const hostImports = {
            host: {
                // Get the length of input at index
                get_input_len: function(idx) {
                    const input = state.inputs[idx];
                    return input ? input.length : 0;
                },

                // Copy input data into WASM memory at ptr
                get_input_data: function(idx, ptr, len) {
                    const input = state.inputs[idx];
                    if (!input || !state.instance) return;

                    const memory = state.instance.exports.memory;
                    const dest = new Uint8Array(memory.buffer, ptr, len);
                    const copyLen = Math.min(len, input.length);
                    dest.set(input.subarray(0, copyLen));
                },

                // Post output data from WASM memory
                post_output: function(outputIdx, ptr, len) {
                    if (!state.instance) return;

                    const memory = state.instance.exports.memory;
                    const src = new Uint8Array(memory.buffer, ptr, len);
                    // Copy the data (don't just reference it, as memory may change)
                    state.outputs.set(outputIdx, new Uint8Array(src));
                },

                // Report a failure (UTF-8 message in WASM memory).
                // Only the first reported error is kept.
                post_error: function(ptr, len) {
                    if (!state.instance || state.error !== null) return;

                    const memory = state.instance.exports.memory;
                    const src = new Uint8Array(memory.buffer, ptr, len);
                    state.error = new TextDecoder().decode(src);
                },
            },
        };

        const module = new WebAssembly.Module(bytes);
        const instance = new WebAssembly.Instance(module, hostImports);
        state.instance = instance;
        state.module = module;

        const handle = nextOperatorHandle++;
        operatorInstances.set(handle, state);
        return handle;
    } catch (e) {
        console.error("Failed to create WASM operator:", e);
        return 0;
    }
};

// Run the operator
// Returns true on success, false on error
window.wasmOperatorRun = function(handle) {
    const state = operatorInstances.get(handle);
    if (!state || !state.instance) {
        console.error("Invalid operator handle:", handle);
        return false;
    }

    try {
        state.instance.exports.run();
        return true;
    } catch (e) {
        console.error("Failed to run operator:", e);
        return false;
    }
};

// Get the error message the operator reported via host.post_error, if any
// Returns a string or null
window.wasmOperatorGetError = function(handle) {
    const state = operatorInstances.get(handle);
    if (!state) return null;
    return state.error;
};

// Get the number of outputs produced
window.wasmOperatorGetOutputCount = function(handle) {
    const state = operatorInstances.get(handle);
    if (!state) return 0;
    return state.outputs.size;
};

// Get output data at index
// Returns Uint8Array or null
window.wasmOperatorGetOutput = function(handle, idx) {
    const state = operatorInstances.get(handle);
    if (!state) return null;
    return state.outputs.get(idx) || null;
};

// Get all output indices that have data
window.wasmOperatorGetOutputIndices = function(handle) {
    const state = operatorInstances.get(handle);
    if (!state) return [];
    return Array.from(state.outputs.keys());
};

// Get metadata from operator
// Returns Uint8Array or null on error
window.wasmOperatorGetMetadata = function(handle) {
    const state = operatorInstances.get(handle);
    if (!state || !state.instance) {
        console.error("Invalid operator handle:", handle);
        return null;
    }

    try {
        // get_metadata returns a packed i64: (len << 32) | ptr
        const packed = state.instance.exports.get_metadata();
        // JavaScript BigInt handling for i64
        const packedBigInt = BigInt(packed);
        const ptr = Number(packedBigInt & BigInt(0xFFFFFFFF));
        const len = Number(packedBigInt >> BigInt(32));

        const memory = state.instance.exports.memory;
        const data = new Uint8Array(memory.buffer, ptr, len);
        // Return a copy
        return new Uint8Array(data);
    } catch (e) {
        console.error("Failed to get operator metadata:", e);
        return null;
    }
};

// Destroy an operator instance
window.wasmOperatorDestroy = function(handle) {
    operatorInstances.delete(handle);
};

// Log that the helper is loaded
console.log("WASM helper loaded");
