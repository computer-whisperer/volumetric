//! JavaScript bindings for web WASM execution.
//!
//! This module provides the wasm-bindgen extern declarations for calling
//! JavaScript functions that execute nested WASM modules in the browser.
//!
//! The companion JavaScript file (wasm_helper.js) implements:
//!
//! ## Model Functions (N-dimensional ABI)
//! - `wasmModelCreateSync(bytes: Uint8Array): number` - Create a model instance synchronously
//! - `wasmModelGetDimensions(handle: number): number` - Get the number of dimensions
//! - `wasmModelGetBounds(handle: number): Float64Array` - Get interleaved [min_0, max_0, min_1, max_1, ...]
//! - `wasmModelSample(handle: number, x: f64, y: f64, z: f64): f32` - Sample density (extra dims zero)
//! - `wasmModelSampleNd(handle: number, position: Float64Array): f32` - Sample density at an N-dimensional position
//! - `wasmModelDestroy(handle: number)` - Free the model instance
//!
//! ## Operator Functions
//! - `wasmOperatorCreate(bytes: Uint8Array, inputs: Array<Uint8Array>): number` - Create operator
//! - `wasmOperatorRun(handle: number): boolean` - Run the operator
//! - `wasmOperatorGetError(handle: number): string | null` - Get reported error, if any
//! - `wasmOperatorGetOutput(handle: number, idx: number): Uint8Array | null` - Get output data
//! - `wasmOperatorGetOutputIndices(handle: number): number[]` - Get indices of outputs
//! - `wasmOperatorGetMetadata(handle: number): Uint8Array | null` - Get metadata
//! - `wasmOperatorDestroy(handle: number)` - Free the operator instance

#![allow(dead_code)]

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

/// Handle type for WASM instances managed by JavaScript.
#[cfg(feature = "web")]
pub type JsWasmHandle = u32;

// ============================================================================
// Model bindings
// ============================================================================

#[cfg(feature = "web")]
#[wasm_bindgen]
extern "C" {
    /// Create a model WASM instance from bytes (synchronous).
    /// Returns a handle that can be used with other model functions.
    /// Returns 0 on error.
    #[wasm_bindgen(js_name = wasmModelCreateSync)]
    pub fn wasm_model_create_sync(bytes: &[u8]) -> JsWasmHandle;

    /// Get the number of dimensions of a model. Returns 0 on error.
    #[wasm_bindgen(js_name = wasmModelGetDimensions)]
    pub fn wasm_model_get_dimensions(handle: JsWasmHandle) -> u32;

    /// Get the bounding box of a model.
    /// Returns a Float64Array with interleaved [min_0, max_0, min_1, max_1, ...]
    /// (2 * dimensions values). Returns null/undefined on error.
    #[wasm_bindgen(js_name = wasmModelGetBounds)]
    pub fn wasm_model_get_bounds(handle: JsWasmHandle) -> Option<Vec<f64>>;

    /// Sample the density at a point (extra dimensions are zeroed).
    /// Returns the density value, or NaN on error.
    #[wasm_bindgen(js_name = wasmModelSample)]
    pub fn wasm_model_sample(handle: JsWasmHandle, x: f64, y: f64, z: f64) -> f32;

    /// Sample the density at an N-dimensional position (missing trailing
    /// dimensions are zeroed, extras ignored). Returns NaN on error.
    #[wasm_bindgen(js_name = wasmModelSampleNd)]
    pub fn wasm_model_sample_nd(handle: JsWasmHandle, position: &[f64]) -> f32;

    /// Destroy a model instance and free resources.
    #[wasm_bindgen(js_name = wasmModelDestroy)]
    pub fn wasm_model_destroy(handle: JsWasmHandle);
}

// ============================================================================
// Operator bindings
// ============================================================================

#[cfg(feature = "web")]
#[wasm_bindgen]
extern "C" {
    /// Create an operator WASM instance with input data.
    /// inputs: Array of Uint8Array (input data for each slot)
    /// Returns a handle or 0 on error.
    #[wasm_bindgen(js_name = wasmOperatorCreate)]
    pub fn wasm_operator_create(bytes: &[u8], inputs: js_sys::Array) -> JsWasmHandle;

    /// Run the operator.
    /// Returns true on success, false on error.
    #[wasm_bindgen(js_name = wasmOperatorRun)]
    pub fn wasm_operator_run(handle: JsWasmHandle) -> bool;

    /// Get the error message the operator reported via `host.post_error`,
    /// if any. Returns None when the operator did not report an error.
    #[wasm_bindgen(js_name = wasmOperatorGetError)]
    pub fn wasm_operator_get_error(handle: JsWasmHandle) -> Option<String>;

    /// Get the output data at the given index.
    /// Returns the output bytes or None if no output at that index.
    #[wasm_bindgen(js_name = wasmOperatorGetOutput)]
    pub fn wasm_operator_get_output(handle: JsWasmHandle, idx: u32) -> Option<Vec<u8>>;

    /// Get all output indices that have data.
    /// Returns an array of indices.
    #[wasm_bindgen(js_name = wasmOperatorGetOutputIndices)]
    pub fn wasm_operator_get_output_indices(handle: JsWasmHandle) -> Vec<u32>;

    /// Get the operator's metadata.
    /// Returns the metadata bytes or None on error.
    #[wasm_bindgen(js_name = wasmOperatorGetMetadata)]
    pub fn wasm_operator_get_metadata(handle: JsWasmHandle) -> Option<Vec<u8>>;

    /// Destroy an operator instance and free resources.
    #[wasm_bindgen(js_name = wasmOperatorDestroy)]
    pub fn wasm_operator_destroy(handle: JsWasmHandle);
}
