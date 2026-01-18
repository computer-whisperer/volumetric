//! Web WASM backend using JavaScript bridge.
//!
//! This module provides implementations for web-based WASM execution.
//! The actual WASM instantiation and execution is delegated to JavaScript,
//! which has native WebAssembly support via the browser's WebAssembly API.
//!
//! # Architecture
//!
//! Since the main application is itself compiled to WASM, we can't use wasmtime.
//! Instead, we use wasm-bindgen to call JavaScript functions that use the browser's
//! WebAssembly API to instantiate and execute nested WASM modules (the model files).
//!
//! The JavaScript helper (wasm_helper.js) maintains a handle map of WASM instances
//! and provides synchronous functions for creating instances and calling exports.

mod js_bindings;

#[cfg(feature = "web")]
pub use js_bindings::*;

use crate::wasm::error::WasmBackendError;
use crate::wasm::traits::{ModelBounds, ModelExecutor, OperatorExecutor, OperatorIo, ParallelModelSampler};

#[cfg(feature = "web")]
use js_bindings::{
    wasm_model_create_sync, wasm_model_get_bounds, wasm_model_is_inside, wasm_model_destroy,
    wasm_operator_create, wasm_operator_run, wasm_operator_get_output,
    wasm_operator_get_output_indices, wasm_operator_get_metadata, wasm_operator_destroy,
    JsWasmHandle,
};

/// Web model executor using JavaScript bridge.
///
/// This implementation uses the browser's WebAssembly API via JavaScript
/// to instantiate and execute model WASM modules.
#[cfg(feature = "web")]
pub struct WebModelExecutor {
    handle: JsWasmHandle,
}

#[cfg(feature = "web")]
impl WebModelExecutor {
    /// Create a new executor from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let handle = wasm_model_create_sync(wasm_bytes);
        if handle == 0 {
            return Err(WasmBackendError::Instantiation(
                "JavaScript failed to create WASM instance".to_string(),
            ));
        }
        Ok(Self { handle })
    }
}

#[cfg(feature = "web")]
impl Drop for WebModelExecutor {
    fn drop(&mut self) {
        wasm_model_destroy(self.handle);
    }
}

#[cfg(feature = "web")]
impl ModelExecutor for WebModelExecutor {
    fn get_bounds(&mut self) -> Result<ModelBounds, WasmBackendError> {
        let bounds = wasm_model_get_bounds(self.handle)
            .ok_or_else(|| WasmBackendError::Execution("Failed to get bounds from WASM".to_string()))?;

        if bounds.len() != 6 {
            return Err(WasmBackendError::Execution(format!(
                "Expected 6 bounds values, got {}",
                bounds.len()
            )));
        }

        Ok(ModelBounds::new(
            (bounds[0], bounds[1], bounds[2]),
            (bounds[3], bounds[4], bounds[5]),
        ))
    }

    fn is_inside(&mut self, x: f64, y: f64, z: f64) -> Result<f32, WasmBackendError> {
        let result = wasm_model_is_inside(self.handle, x, y, z);
        if result.is_nan() {
            return Err(WasmBackendError::Execution("WASM is_inside returned NaN".to_string()));
        }
        Ok(result)
    }
}

/// Web parallel sampler using JavaScript bridge.
///
/// Since wasm32 is single-threaded, this implementation is simpler than
/// the native version - we just use a single WASM instance without any
/// thread-local storage or synchronization.
#[cfg(feature = "web")]
pub struct WebParallelSampler {
    handle: JsWasmHandle,
    bounds: ModelBounds,
}

#[cfg(feature = "web")]
impl WebParallelSampler {
    /// Create a new parallel sampler from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let handle = wasm_model_create_sync(wasm_bytes);
        if handle == 0 {
            return Err(WasmBackendError::Instantiation(
                "JavaScript failed to create WASM instance".to_string(),
            ));
        }

        // Get bounds immediately and cache them
        let bounds_vec = wasm_model_get_bounds(handle)
            .ok_or_else(|| WasmBackendError::Execution("Failed to get bounds from WASM".to_string()))?;

        if bounds_vec.len() != 6 {
            wasm_model_destroy(handle);
            return Err(WasmBackendError::Execution(format!(
                "Expected 6 bounds values, got {}",
                bounds_vec.len()
            )));
        }

        let bounds = ModelBounds::new(
            (bounds_vec[0], bounds_vec[1], bounds_vec[2]),
            (bounds_vec[3], bounds_vec[4], bounds_vec[5]),
        );

        Ok(Self { handle, bounds })
    }
}

#[cfg(feature = "web")]
impl Drop for WebParallelSampler {
    fn drop(&mut self) {
        wasm_model_destroy(self.handle);
    }
}

#[cfg(feature = "web")]
impl ParallelModelSampler for WebParallelSampler {
    fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        let result = wasm_model_is_inside(self.handle, x, y, z);
        if result.is_nan() {
            0.0 // Return 0 (outside) on error
        } else {
            result
        }
    }

    fn get_bounds(&self) -> Result<ModelBounds, WasmBackendError> {
        Ok(self.bounds.clone())
    }
}

// SAFETY: wasm32 is single-threaded, so Send/Sync are safe
#[cfg(feature = "web")]
unsafe impl Send for WebParallelSampler {}
#[cfg(feature = "web")]
unsafe impl Sync for WebParallelSampler {}

/// Web operator executor using JavaScript bridge.
///
/// Operators are WASM modules that transform inputs (model WASM, configuration)
/// into outputs (transformed model WASM). This implementation uses the browser's
/// WebAssembly API via JavaScript to handle the complex input/output buffer
/// management required for operators.
#[cfg(feature = "web")]
pub struct WebOperatorExecutor {
    wasm_bytes: Vec<u8>,
}

#[cfg(feature = "web")]
impl WebOperatorExecutor {
    /// Create a new operator executor from WASM bytes.
    ///
    /// Note: Unlike model executors, we don't create the WASM instance immediately.
    /// Instead, we store the bytes and create a fresh instance for each run() call,
    /// since operators need different inputs for each execution.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        Ok(Self {
            wasm_bytes: wasm_bytes.to_vec(),
        })
    }
}

#[cfg(feature = "web")]
impl OperatorExecutor for WebOperatorExecutor {
    fn run(&mut self, io: OperatorIo) -> Result<OperatorIo, WasmBackendError> {
        // Convert inputs to JavaScript Array of Uint8Array
        let js_inputs = js_sys::Array::new();
        for input in &io.inputs {
            let js_array = js_sys::Uint8Array::from(input.as_slice());
            js_inputs.push(&js_array);
        }

        // Create operator instance with inputs
        let handle = wasm_operator_create(&self.wasm_bytes, js_inputs);
        if handle == 0 {
            return Err(WasmBackendError::Instantiation(
                "JavaScript failed to create operator WASM instance".to_string(),
            ));
        }

        // Run the operator
        let success = wasm_operator_run(handle);
        if !success {
            wasm_operator_destroy(handle);
            return Err(WasmBackendError::Execution(
                "Operator run() failed".to_string(),
            ));
        }

        // Collect outputs
        let mut result = OperatorIo {
            inputs: io.inputs,
            outputs: std::collections::HashMap::new(),
        };

        let output_indices = wasm_operator_get_output_indices(handle);
        for idx in output_indices {
            if let Some(data) = wasm_operator_get_output(handle, idx) {
                result.outputs.insert(idx as usize, data);
            }
        }

        // Clean up
        wasm_operator_destroy(handle);

        Ok(result)
    }

    fn get_metadata(&mut self) -> Result<Vec<u8>, WasmBackendError> {
        // Create a temporary instance with empty inputs just to get metadata
        let js_inputs = js_sys::Array::new();
        let handle = wasm_operator_create(&self.wasm_bytes, js_inputs);
        if handle == 0 {
            return Err(WasmBackendError::Instantiation(
                "JavaScript failed to create operator WASM instance".to_string(),
            ));
        }

        let metadata = wasm_operator_get_metadata(handle);
        wasm_operator_destroy(handle);

        metadata.ok_or_else(|| {
            WasmBackendError::Execution("Failed to get operator metadata".to_string())
        })
    }
}
