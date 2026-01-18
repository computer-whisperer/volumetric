//! WASM backend abstraction layer.
//!
//! This module provides traits and implementations for executing WASM modules
//! in different environments (native desktop via wasmtime, web via JavaScript bridge).
//!
//! # Feature Flags
//!
//! - `native` (default): Enables the native wasmtime backend
//! - `web`: Enables the web JavaScript bridge backend (future)
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
pub use traits::{ModelBounds, ModelExecutor, OperatorExecutor, OperatorIo, ParallelModelSampler};

// Re-export native types when available
#[cfg(feature = "native")]
pub use native::{NativeModelExecutor, NativeOperatorExecutor, NativeParallelSampler};

// Re-export web types when available
#[cfg(feature = "web")]
pub use web::{WebModelExecutor, WebOperatorExecutor, WebParallelSampler};

/// Create a model executor from WASM bytes.
///
/// Returns the appropriate executor for the current build configuration.
#[cfg(feature = "native")]
pub fn create_model_executor(
    wasm_bytes: &[u8],
) -> Result<impl ModelExecutor, WasmBackendError> {
    NativeModelExecutor::new(wasm_bytes)
}

/// Create a model executor from WASM bytes (web backend).
#[cfg(all(feature = "web", not(feature = "native")))]
pub fn create_model_executor(
    wasm_bytes: &[u8],
) -> Result<impl ModelExecutor, WasmBackendError> {
    WebModelExecutor::new(wasm_bytes)
}

/// Create a model executor from WASM bytes (no backend available).
#[cfg(not(any(feature = "native", feature = "web")))]
pub fn create_model_executor(
    _wasm_bytes: &[u8],
) -> Result<impl ModelExecutor, WasmBackendError> {
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
    NativeParallelSampler::new(wasm_bytes)
}

/// Create a parallel model sampler from WASM bytes (web backend).
#[cfg(all(feature = "web", not(feature = "native")))]
pub fn create_parallel_sampler(
    wasm_bytes: &[u8],
) -> Result<impl ParallelModelSampler, WasmBackendError> {
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
