//! WASM backend abstraction layer.
//!
//! This module provides traits and implementations for executing WASM modules
//! in different environments (native desktop via wasmtime, web via JavaScript bridge).
//!
//! Models use the N-dimensional ABI:
//! - `get_dimensions() -> u32`
//! - `get_bounds(out_ptr: i32)` — writes interleaved min/max f64 pairs
//! - `sample(pos_ptr: i32) -> f32` — reads `dims` f64 values
//! - `memory` export
//!
//! # Feature Flags
//!
//! - `native` (default): Enables the native wasmtime backend
//! - `web`: Enables the web JavaScript bridge backend
//!
//! # Usage
//!
//! Use the factory functions to create executors:
//!
//! ```ignore
//! use volumetric::wasm::{create_model_executor, create_parallel_sampler, create_operator_executor};
//!
//! // Single-threaded model executor
//! let mut executor = create_model_executor(wasm_bytes)?;
//! let bounds = executor.get_bounds()?;
//! let density = executor.is_inside(0.0, 0.0, 0.0)?;
//!
//! // Thread-safe parallel sampler
//! let sampler = create_parallel_sampler(wasm_bytes)?;
//! // Can be used from multiple threads via rayon
//! let density = sampler.sample(0.0, 0.0, 0.0);
//!
//! // Operator executor
//! let mut op_executor = create_operator_executor(operator_wasm)?;
//! let metadata = op_executor.get_metadata()?;
//! let result = op_executor.run(io)?;
//! ```

pub mod error;
pub mod traits;

#[cfg(feature = "native")]
pub mod native;

#[cfg(feature = "web")]
pub mod web;

// Re-export main types
pub use error::WasmBackendError;
pub use traits::{
    ModelBounds, ModelBoundsNd, ModelExecutor, OperatorExecutor, OperatorIo, ParallelModelSampler,
};

// Re-export native types when available
#[cfg(feature = "native")]
pub use native::{
    ModuleCache, NativeModelExecutor, NativeOperatorExecutor, NativeParallelSampler, model_cache,
    operator_cache,
};

// Re-export web types when available
#[cfg(feature = "web")]
pub use web::{WebModelExecutor, WebOperatorExecutor, WebParallelSampler};

/// Reject models built against the removed legacy 3D ABI with a clear error.
///
/// Legacy models exported `is_inside(f64, f64, f64) -> f32` plus six
/// `get_bounds_*` getters. Support was removed; every model in this repo (and
/// every operator output) uses the N-dimensional ABI. Without this check a
/// legacy blob from an old saved project would fail with a bare
/// "missing export: get_dimensions".
fn reject_legacy_model(wasm_bytes: &[u8]) -> Result<(), WasmBackendError> {
    let mut has_sample = false;
    let mut has_is_inside = false;

    for payload in wasmparser::Parser::new(0).parse_all(wasm_bytes) {
        let payload = payload.map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;
        if let wasmparser::Payload::ExportSection(section) = payload {
            for export in section {
                let export = export.map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;
                match export.name {
                    "sample" => has_sample = true,
                    "is_inside" => has_is_inside = true,
                    _ => {}
                }
            }
        }
    }

    if has_is_inside && !has_sample {
        return Err(WasmBackendError::Instantiation(
            "model exports the legacy 3D ABI (is_inside/get_bounds_*), which is no longer \
             supported; rebuild the model against the N-dimensional ABI \
             (get_dimensions/get_bounds/sample/memory)"
                .to_string(),
        ));
    }
    Ok(())
}

/// Create a model executor from WASM bytes.
#[cfg(feature = "native")]
pub fn create_model_executor(wasm_bytes: &[u8]) -> Result<impl ModelExecutor, WasmBackendError> {
    reject_legacy_model(wasm_bytes)?;
    NativeModelExecutor::new(wasm_bytes)
}

/// Create a model executor from WASM bytes (web backend).
#[cfg(all(feature = "web", not(feature = "native")))]
pub fn create_model_executor(wasm_bytes: &[u8]) -> Result<impl ModelExecutor, WasmBackendError> {
    reject_legacy_model(wasm_bytes)?;
    WebModelExecutor::new(wasm_bytes)
}

/// Create a model executor from WASM bytes (no backend available).
#[cfg(not(any(feature = "native", feature = "web")))]
pub fn create_model_executor(_wasm_bytes: &[u8]) -> Result<impl ModelExecutor, WasmBackendError> {
    Err::<DummyExecutor, _>(WasmBackendError::Unavailable(
        "No WASM backend available. Enable 'native' or 'web' feature.".to_string(),
    ))
}

/// Create a parallel model sampler from WASM bytes.
///
/// Returns a thread-safe sampler that can be used from multiple threads.
#[cfg(feature = "native")]
pub fn create_parallel_sampler(
    wasm_bytes: &[u8],
) -> Result<impl ParallelModelSampler, WasmBackendError> {
    reject_legacy_model(wasm_bytes)?;
    NativeParallelSampler::new(wasm_bytes)
}

/// Create a parallel model sampler from WASM bytes (web backend).
#[cfg(all(feature = "web", not(feature = "native")))]
pub fn create_parallel_sampler(
    wasm_bytes: &[u8],
) -> Result<impl ParallelModelSampler, WasmBackendError> {
    reject_legacy_model(wasm_bytes)?;
    WebParallelSampler::new(wasm_bytes)
}

/// Create a parallel model sampler from WASM bytes (no backend available).
#[cfg(not(any(feature = "native", feature = "web")))]
pub fn create_parallel_sampler(
    _wasm_bytes: &[u8],
) -> Result<impl ParallelModelSampler, WasmBackendError> {
    Err::<DummySampler, _>(WasmBackendError::Unavailable(
        "No WASM backend available. Enable 'native' or 'web' feature.".to_string(),
    ))
}

/// Create an operator executor from WASM bytes.
#[cfg(feature = "native")]
pub fn create_operator_executor(
    wasm_bytes: &[u8],
) -> Result<impl OperatorExecutor, WasmBackendError> {
    NativeOperatorExecutor::new(wasm_bytes)
}

/// Create an operator executor from WASM bytes (web backend).
#[cfg(all(feature = "web", not(feature = "native")))]
pub fn create_operator_executor(
    wasm_bytes: &[u8],
) -> Result<impl OperatorExecutor, WasmBackendError> {
    WebOperatorExecutor::new(wasm_bytes)
}

/// Create an operator executor from WASM bytes (no backend available).
#[cfg(not(any(feature = "native", feature = "web")))]
pub fn create_operator_executor(
    _wasm_bytes: &[u8],
) -> Result<impl OperatorExecutor, WasmBackendError> {
    Err::<DummyOperator, _>(WasmBackendError::Unavailable(
        "No WASM backend available. Enable 'native' or 'web' feature.".to_string(),
    ))
}

// Dummy types for when no backend is available (needed for type inference)
#[cfg(not(any(feature = "native", feature = "web")))]
struct DummyExecutor;
#[cfg(not(any(feature = "native", feature = "web")))]
impl ModelExecutor for DummyExecutor {
    fn get_bounds(&mut self) -> Result<ModelBounds, WasmBackendError> {
        unreachable!()
    }
    fn is_inside(&mut self, _x: f64, _y: f64, _z: f64) -> Result<f32, WasmBackendError> {
        unreachable!()
    }
}

#[cfg(not(any(feature = "native", feature = "web")))]
struct DummySampler;
#[cfg(not(any(feature = "native", feature = "web")))]
impl ParallelModelSampler for DummySampler {
    fn sample(&self, _x: f64, _y: f64, _z: f64) -> f32 {
        unreachable!()
    }
    fn get_bounds(&self) -> Result<ModelBounds, WasmBackendError> {
        unreachable!()
    }
}

#[cfg(not(any(feature = "native", feature = "web")))]
struct DummyOperator;
#[cfg(not(any(feature = "native", feature = "web")))]
impl OperatorExecutor for DummyOperator {
    fn run(&mut self, _io: OperatorIo) -> Result<OperatorIo, WasmBackendError> {
        unreachable!()
    }
    fn get_metadata(&mut self) -> Result<Vec<u8>, WasmBackendError> {
        unreachable!()
    }
}
