//! JavaScript bindings for web WASM execution.
//!
//! This module provides the wasm-bindgen extern declarations for calling
//! JavaScript functions that execute WASM modules in the browser.
//!
//! The companion JavaScript file should implement:
//! - `wasmModelCreate(bytes: Uint8Array): number` - Create a model instance
//! - `wasmModelGetBounds(handle: number): Float64Array` - Get [minX, minY, minZ, maxX, maxY, maxZ]
//! - `wasmModelIsInside(handle: number, x: f64, y: f64, z: f64): f32` - Sample density
//! - `wasmModelDestroy(handle: number)` - Free the model instance
//! - `wasmOperatorCreate(bytes: Uint8Array): number` - Create an operator instance
//! - `wasmOperatorRun(handle: number, inputs: Array<Uint8Array>): Array<Uint8Array>` - Run operator
//! - `wasmOperatorGetMetadata(handle: number): Uint8Array` - Get CBOR metadata
//! - `wasmOperatorDestroy(handle: number)` - Free the operator instance

#![allow(dead_code)]

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

/// Handle type for WASM instances managed by JavaScript.
#[cfg(feature = "web")]
pub type JsWasmHandle = u32;

#[cfg(feature = "web")]
#[wasm_bindgen]
extern "C" {
    /// Create a model WASM instance from bytes.
    /// Returns a handle that can be used with other model functions.
    #[wasm_bindgen(js_name = wasmModelCreate)]
    pub fn wasm_model_create(bytes: &[u8]) -> JsWasmHandle;

    /// Get the bounding box of a model.
    /// Returns a Float64Array with [minX, minY, minZ, maxX, maxY, maxZ].
    #[wasm_bindgen(js_name = wasmModelGetBounds)]
    pub fn wasm_model_get_bounds(handle: JsWasmHandle) -> Vec<f64>;

    /// Sample the density at a point.
    /// Returns the density value (> 0 means inside).
    #[wasm_bindgen(js_name = wasmModelIsInside)]
    pub fn wasm_model_is_inside(handle: JsWasmHandle, x: f64, y: f64, z: f64) -> f32;

    /// Destroy a model instance and free resources.
    #[wasm_bindgen(js_name = wasmModelDestroy)]
    pub fn wasm_model_destroy(handle: JsWasmHandle);

    /// Create an operator WASM instance from bytes.
    #[wasm_bindgen(js_name = wasmOperatorCreate)]
    pub fn wasm_operator_create(bytes: &[u8]) -> JsWasmHandle;

    /// Run an operator with the given inputs.
    /// Returns an array of output byte arrays.
    #[wasm_bindgen(js_name = wasmOperatorRun)]
    pub fn wasm_operator_run(handle: JsWasmHandle, inputs: Vec<js_sys::Uint8Array>) -> Vec<js_sys::Uint8Array>;

    /// Get the operator's CBOR metadata.
    #[wasm_bindgen(js_name = wasmOperatorGetMetadata)]
    pub fn wasm_operator_get_metadata(handle: JsWasmHandle) -> Vec<u8>;

    /// Destroy an operator instance and free resources.
    #[wasm_bindgen(js_name = wasmOperatorDestroy)]
    pub fn wasm_operator_destroy(handle: JsWasmHandle);
}
