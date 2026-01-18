// WASM Helper for nested WASM module execution
// This file provides JavaScript functions for instantiating and calling WASM model modules
// from within the main volumetric_ui WASM module.

// Map of handle -> WASM instance
const wasmInstances = new Map();
let nextHandle = 1;

// Create a model WASM instance from bytes
// Returns a handle (number) that can be used with other functions
window.wasmModelCreate = async function(bytes) {
    try {
        const module = await WebAssembly.compile(bytes);
        const instance = await WebAssembly.instantiate(module, {});
        const handle = nextHandle++;
        wasmInstances.set(handle, { instance, module });
        return handle;
    } catch (e) {
        console.error("Failed to create WASM model:", e);
        return 0; // 0 indicates error
    }
};

// Synchronous version for use from WASM (blocks until complete)
window.wasmModelCreateSync = function(bytes) {
    try {
        const module = new WebAssembly.Module(bytes);
        const instance = new WebAssembly.Instance(module, {});
        const handle = nextHandle++;
        wasmInstances.set(handle, { instance, module });
        return handle;
    } catch (e) {
        console.error("Failed to create WASM model:", e);
        return 0;
    }
};

// Get the bounding box of a model
// Returns [minX, minY, minZ, maxX, maxY, maxZ] or null on error
window.wasmModelGetBounds = function(handle) {
    const entry = wasmInstances.get(handle);
    if (!entry) {
        console.error("Invalid WASM handle:", handle);
        return null;
    }

    try {
        const exports = entry.instance.exports;
        return new Float64Array([
            exports.get_bounds_min_x(),
            exports.get_bounds_min_y(),
            exports.get_bounds_min_z(),
            exports.get_bounds_max_x(),
            exports.get_bounds_max_y(),
            exports.get_bounds_max_z(),
        ]);
    } catch (e) {
        console.error("Failed to get bounds:", e);
        return null;
    }
};

// Sample the density at a point
// Returns the density value (> 0 means inside), or NaN on error
window.wasmModelIsInside = function(handle, x, y, z) {
    const entry = wasmInstances.get(handle);
    if (!entry) {
        console.error("Invalid WASM handle:", handle);
        return NaN;
    }

    try {
        return entry.instance.exports.is_inside(x, y, z);
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
