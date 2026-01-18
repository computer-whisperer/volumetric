//! JavaScript bindings for web WASM execution.
//!
//! This module provides the wasm-bindgen extern declarations for calling
//! JavaScript functions that execute nested WASM modules in the browser.
//!
//! The companion JavaScript file (wasm_helper.js) implements:
//!
//! ## Model Functions
//! - `wasmModelCreateSync(bytes: Uint8Array): number` - Create a model instance synchronously
//! - `wasmModelGetBounds(handle: number): Float64Array` - Get [minX, minY, minZ, maxX, maxY, maxZ]
//! - `wasmModelIsInside(handle: number, x: f64, y: f64, z: f64): f32` - Sample density
//! - `wasmModelDestroy(handle: number)` - Free the model instance
//!
//! ## Operator Functions
//! - `wasmOperatorCreate(bytes: Uint8Array, inputs: Array<Uint8Array>): number` - Create operator
//! - `wasmOperatorRun(handle: number): boolean` - Run the operator
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

    /// Get the bounding box of a model.
    /// Returns a Float64Array with [minX, minY, minZ, maxX, maxY, maxZ].
    /// Returns null/undefined on error.
    #[wasm_bindgen(js_name = wasmModelGetBounds)]
    pub fn wasm_model_get_bounds(handle: JsWasmHandle) -> Option<Vec<f64>>;

    /// Sample the density at a point.
    /// Returns the density value (> 0 means inside).
    #[wasm_bindgen(js_name = wasmModelIsInside)]
    pub fn wasm_model_is_inside(handle: JsWasmHandle, x: f64, y: f64, z: f64) -> f32;

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
