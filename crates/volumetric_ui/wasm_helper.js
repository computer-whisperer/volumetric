// WASM Helper for nested WASM module execution
// This file provides JavaScript functions for instantiating and calling WASM model modules
// from within the main volumetric_ui WASM module.
//
// Copied into crates/volumetric_ui_v2/wasm_helper.js — the two files implement
// the same engine JS-bridge contract (src/wasm/web/js_bindings.rs) and must
// stay in sync.
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

// Content-keyed caches of compiled WebAssembly.Modules, mirroring the
// native backend's module_cache.rs: compiling is by far the most expensive
// part of creating an instance, and the preview loop keeps recreating
// executors for the same bytes (mesh sampler + channel executor per build,
// one executor per lightbox slice, every operator per run). Modules are
// cached; instances are always fresh (concurrent executors must not share
// an IO buffer). FIFO-bounded like the native cache: interactive editing
// keeps producing new merged-model blobs, so an unbounded cache would grow
// for the lifetime of the page. Keys are the exact content — a hash picks
// the bucket, a byte compare confirms (native keys by the full bytes too).
const MODULE_CACHE_CAPACITY = 32;

function newModuleCache() {
    return { buckets: new Map(), order: [], hits: 0, misses: 0 };
}

const modelModuleCache = newModuleCache();
const operatorModuleCache = newModuleCache();

// Two interleaved FNV-1a-style 32-bit hashes plus the length; collisions
// only cost a wasted bucket scan, correctness comes from sameBytes.
function moduleCacheKey(bytes) {
    let h1 = 0x811c9dc5 | 0;
    let h2 = 0x01000193 | 0;
    for (let i = 0; i < bytes.length; i++) {
        h1 = Math.imul(h1 ^ bytes[i], 0x01000193);
        h2 = Math.imul(h2 ^ bytes[i], 0x0100019b);
    }
    return bytes.length + ":" + (h1 >>> 0) + ":" + (h2 >>> 0);
}

function sameBytes(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

// Return the compiled module for bytes, compiling on a miss. Compile
// failures propagate (and are not cached, matching the native cache).
function getOrCompileModule(cache, bytes) {
    const key = moduleCacheKey(bytes);
    const bucket = cache.buckets.get(key);
    if (bucket) {
        for (const entry of bucket) {
            if (sameBytes(entry.bytes, bytes)) {
                cache.hits++;
                return entry.module;
            }
        }
    }
    cache.misses++;
    const module = new WebAssembly.Module(bytes);
    // Copy the key bytes: the caller's array may be a view into wasm memory.
    const entry = { key, bytes: bytes.slice(), module };
    if (bucket) {
        bucket.push(entry);
    } else {
        cache.buckets.set(key, [entry]);
    }
    cache.order.push(entry);
    while (cache.order.length > MODULE_CACHE_CAPACITY) {
        const evicted = cache.order.shift();
        const evictedBucket = cache.buckets.get(evicted.key);
        const idx = evictedBucket.indexOf(evicted);
        evictedBucket.splice(idx, 1);
        if (evictedBucket.length === 0) cache.buckets.delete(evicted.key);
    }
    return module;
}

// Cache hit/miss counters, for debugging and driving tests.
window.wasmModuleCacheStats = function() {
    return {
        modelHits: modelModuleCache.hits,
        modelMisses: modelModuleCache.misses,
        operatorHits: operatorModuleCache.hits,
        operatorMisses: operatorModuleCache.misses,
    };
};

// Map of handle -> WASM instance
const wasmInstances = new Map();
let nextHandle = 1;

// Instantiate a model module and validate the N-dimensional ABI.
// Returns { instance, module, dimensions, ioPtr, sampleFormatBytes,
// hasSampleChannels } or null on error.
// Shared by the model executor path (wasmModelCreateSync) and the
// operators' lazily-created input models (input_model_* host imports).
function createModelEntry(bytes) {
    try {
        const module = getOrCompileModule(modelModuleCache, bytes);
        const instance = new WebAssembly.Instance(module, {});
        const exports = instance.exports;
        if (!exports.get_dimensions || !exports.get_io_ptr || !exports.get_bounds
            || !exports.sample || !exports.memory) {
            console.error("WASM model does not export the N-dimensional ABI");
            return null;
        }
        const dimensions = exports.get_dimensions();
        const ioPtr = exports.get_io_ptr();
        if (ioPtr <= 0 || ioPtr + dimensions * 2 * 8 > exports.memory.buffer.byteLength) {
            console.error("WASM model returned an invalid IO buffer pointer:", ioPtr);
            return null;
        }
        // Optional typed-channel exports (absent means occupancy-only). The
        // declared format is a ptr|len-packed CBOR blob; copy it out at
        // creation like the native executor does, and fail creation if the
        // export exists but its region is bogus.
        let sampleFormatBytes = null;
        if (exports.get_sample_format) {
            const packed = BigInt(exports.get_sample_format());
            const ptr = Number(packed & BigInt(0xFFFFFFFF));
            const len = Number(packed >> BigInt(32));
            if (ptr + len > exports.memory.buffer.byteLength) {
                console.error("get_sample_format returned an out-of-bounds region:", ptr, len);
                return null;
            }
            sampleFormatBytes = new Uint8Array(exports.memory.buffer, ptr, len).slice();
        }
        const hasSampleChannels = !!exports.sample_channels;
        return { instance, module, dimensions, ioPtr, sampleFormatBytes, hasSampleChannels };
    } catch (e) {
        console.error("Failed to create WASM model:", e);
        return null;
    }
}

// Create a model WASM instance from bytes (synchronous, for use from WASM)
// Returns a handle (number) that can be used with other functions, 0 on error
window.wasmModelCreateSync = function(bytes) {
    const entry = createModelEntry(bytes);
    if (!entry) return 0;
    const handle = nextHandle++;
    wasmInstances.set(handle, entry);
    return handle;
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

// Sample the density at an N-dimensional position (a Float64Array of the
// model's dimension count; missing trailing dimensions are zeroed, extras
// ignored). Returns the density value, or NaN on error.
window.wasmModelSampleNd = function(handle, position) {
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
        const n = Math.min(entry.dimensions, position.length);
        for (let d = 0; d < n; d++) pos[d] = position[d];
        return exports.sample(entry.ioPtr);
    } catch (e) {
        console.error("Failed to sample:", e);
        return NaN;
    }
};

// Raw CBOR bytes of the model's declared SampleFormat, captured at creation.
// Returns null when the model has no get_sample_format export (occupancy-only
// default) or the handle is invalid.
window.wasmModelGetSampleFormat = function(handle) {
    const entry = wasmInstances.get(handle);
    if (!entry) {
        console.error("Invalid WASM handle:", handle);
        return null;
    }
    return entry.sampleFormatBytes;
};

// Whether the model exports sample_channels.
window.wasmModelHasSampleChannels = function(handle) {
    const entry = wasmInstances.get(handle);
    return entry ? entry.hasSampleChannels : false;
};

// Sample every declared channel at an N-dimensional position (same position
// convention as wasmModelSampleNd). The position goes in the first half of
// the IO buffer and the model writes one f32 per channel into the second
// half; channelCount is the caller's decoded channel count (its fit in the
// buffer's n*8-byte output half is validated executor-side at creation).
// Returns a Float32Array of channelCount values, or null on error.
window.wasmModelSampleChannelsNd = function(handle, position, channelCount) {
    const entry = wasmInstances.get(handle);
    if (!entry) {
        console.error("Invalid WASM handle:", handle);
        return null;
    }
    if (!entry.hasSampleChannels || channelCount * 4 > entry.dimensions * 8) {
        console.error("sample_channels unavailable or channel count too large:", channelCount);
        return null;
    }

    try {
        const exports = entry.instance.exports;
        const pos = new Float64Array(
            exports.memory.buffer,
            entry.ioPtr,
            entry.dimensions,
        );
        pos.fill(0);
        const n = Math.min(entry.dimensions, position.length);
        for (let d = 0; d < n; d++) pos[d] = position[d];
        const outPtr = entry.ioPtr + entry.dimensions * 8;
        exports.sample_channels(entry.ioPtr, outPtr);
        // Re-view after the call: memory may move on growth.
        const view = new Float32Array(exports.memory.buffer, outPtr, channelCount);
        return new Float32Array(view);
    } catch (e) {
        console.error("Failed to sample channels:", e);
        return null;
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
            // Lazily-created model entries backing the input_model_*
            // sampling imports, keyed by input slot. null records a failed
            // creation so a bad input isn't recompiled on every call.
            models: new Map(),
        };

        // Get (or lazily create) the model entry for input slot idx.
        function modelFor(idx) {
            if (!state.models.has(idx)) {
                const input = state.inputs[idx];
                state.models.set(idx, input ? createModelEntry(input) : null);
            }
            return state.models.get(idx);
        }

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

                // Number of dimensions of the model in an input slot
                // (0 when the slot doesn't hold a usable model).
                input_model_dimensions: function(idx) {
                    const m = modelFor(idx);
                    return m ? m.dimensions : 0;
                },

                // Write a model input's 2n interleaved f64 bounds into
                // operator memory at outPtr. Returns 1 on success, 0 on
                // failure. Byte copies throughout: outPtr need not be
                // 8-aligned, and typed-array constructors throw on
                // out-of-range pointers (caught below → 0, matching the
                // native host's bounds checks).
                input_model_bounds: function(idx, outPtr) {
                    const m = modelFor(idx);
                    if (!m || !state.instance) return 0;
                    try {
                        const exports = m.instance.exports;
                        exports.get_bounds(m.ioPtr);
                        const bytes = m.dimensions * 2 * 8;
                        const src = new Uint8Array(exports.memory.buffer, m.ioPtr, bytes);
                        new Uint8Array(state.instance.exports.memory.buffer, outPtr, bytes)
                            .set(src);
                        return 1;
                    } catch (e) {
                        console.error("input_model_bounds failed:", e);
                        return 0;
                    }
                },

                // Sample a model input at count positions (n f64s each,
                // read at posPtr), writing one occupancy f32 per position
                // at outPtr. Failed individual samples follow the ABI
                // convention and read as 0.0 (outside); returns 1 on
                // success, 0 when the slot is not a model or a pointer is
                // out of range.
                input_model_sample: function(idx, posPtr, count, outPtr) {
                    const m = modelFor(idx);
                    if (!m || !state.instance) return 0;
                    try {
                        const opMemory = state.instance.exports.memory;
                        const n = m.dimensions;
                        // Copy out of operator memory first (posPtr may be
                        // unaligned; the model's sample may not touch
                        // operator memory but stay defensive anyway).
                        const posBytes =
                            new Uint8Array(opMemory.buffer, posPtr, count * n * 8).slice();
                        const positions = new Float64Array(posBytes.buffer);
                        const out = new Float32Array(count);
                        const exports = m.instance.exports;
                        for (let i = 0; i < count; i++) {
                            // Re-view per call: model memory may move on growth.
                            const pos = new Float64Array(exports.memory.buffer, m.ioPtr, n);
                            for (let d = 0; d < n; d++) pos[d] = positions[i * n + d];
                            let sample = 0.0;
                            try {
                                sample = exports.sample(m.ioPtr);
                            } catch (e) {
                                // Failed sample reads as 0.0 (outside),
                                // matching the native host.
                            }
                            out[i] = sample;
                        }
                        new Uint8Array(opMemory.buffer, outPtr, count * 4)
                            .set(new Uint8Array(out.buffer));
                        return 1;
                    } catch (e) {
                        console.error("input_model_sample failed:", e);
                        return 0;
                    }
                },
            },
        };

        const module = getOrCompileModule(operatorModuleCache, bytes);
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
