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
pub use traits::{ModelBounds, ModelBoundsNd, ModelExecutor, OperatorExecutor, OperatorIo, ParallelModelSampler};

// Re-export native types when available
#[cfg(feature = "native")]
pub use native::{NativeModelExecutor, NativeModelExecutorNd, NativeOperatorExecutor, NativeParallelSampler, NativeParallelSamplerNd};

// Re-export web types when available
#[cfg(feature = "web")]
pub use web::{WebModelExecutor, WebOperatorExecutor, WebParallelSampler};

/// Wrapper enum for model executors that auto-detects ABI version.
#[cfg(feature = "native")]
pub enum AutoModelExecutor {
    Legacy(NativeModelExecutor),
    Nd(NativeModelExecutorNd),
}

#[cfg(feature = "native")]
impl ModelExecutor for AutoModelExecutor {
    fn get_bounds(&mut self) -> Result<ModelBounds, WasmBackendError> {
        match self {
            AutoModelExecutor::Legacy(e) => e.get_bounds(),
            AutoModelExecutor::Nd(e) => e.get_bounds(),
        }
    }

    fn is_inside(&mut self, x: f64, y: f64, z: f64) -> Result<f32, WasmBackendError> {
        match self {
            AutoModelExecutor::Legacy(e) => e.is_inside(x, y, z),
            AutoModelExecutor::Nd(e) => e.is_inside(x, y, z),
        }
    }
}

/// Create a model executor from WASM bytes.
///
/// Returns the appropriate executor for the current build configuration.
/// Automatically detects whether the WASM uses the legacy 3D ABI or the new N-dimensional ABI.
#[cfg(feature = "native")]
pub fn create_model_executor(
    wasm_bytes: &[u8],
) -> Result<impl ModelExecutor, WasmBackendError> {
    match detect_abi_version(wasm_bytes)? {
        AbiVersion::Nd => {
            let executor = NativeModelExecutorNd::new(wasm_bytes)?;
            Ok(AutoModelExecutor::Nd(executor))
        }
        AbiVersion::Legacy => {
            let executor = NativeModelExecutor::new(wasm_bytes)?;
            Ok(AutoModelExecutor::Legacy(executor))
        }
    }
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

/// Wrapper enum for parallel samplers that auto-detects ABI version.
#[cfg(feature = "native")]
pub enum AutoParallelSampler {
    Legacy(NativeParallelSampler),
    Nd(NativeParallelSamplerNd),
}

#[cfg(feature = "native")]
impl ParallelModelSampler for AutoParallelSampler {
    fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        match self {
            AutoParallelSampler::Legacy(s) => s.sample(x, y, z),
            AutoParallelSampler::Nd(s) => s.sample(x, y, z),
        }
    }

    fn get_bounds(&self) -> Result<ModelBounds, WasmBackendError> {
        match self {
            AutoParallelSampler::Legacy(s) => s.get_bounds(),
            AutoParallelSampler::Nd(s) => s.get_bounds(),
        }
    }
}

/// Create a parallel model sampler from WASM bytes.
///
/// Returns a thread-safe sampler that can be used from multiple threads.
/// Automatically detects whether the WASM uses the legacy 3D ABI or the new N-dimensional ABI.
#[cfg(feature = "native")]
pub fn create_parallel_sampler(
    wasm_bytes: &[u8],
) -> Result<impl ParallelModelSampler, WasmBackendError> {
    match detect_abi_version(wasm_bytes)? {
        AbiVersion::Nd => {
            let sampler = NativeParallelSamplerNd::new(wasm_bytes)?;
            Ok(AutoParallelSampler::Nd(sampler))
        }
        AbiVersion::Legacy => {
            let sampler = NativeParallelSampler::new(wasm_bytes)?;
            Ok(AutoParallelSampler::Legacy(sampler))
        }
    }
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

/// Create a parallel model sampler for N-dimensional ABI from WASM bytes.
///
/// Returns a thread-safe sampler that can be used from multiple threads.
/// This sampler works with the new N-dimensional ABI:
/// - `get_dimensions() -> u32`
/// - `get_bounds(out_ptr: i32)` - writes interleaved min/max
/// - `sample(pos_ptr: i32) -> f32`
/// - `memory` export
#[cfg(feature = "native")]
pub fn create_parallel_sampler_nd(
    wasm_bytes: &[u8],
) -> Result<NativeParallelSamplerNd, WasmBackendError> {
    NativeParallelSamplerNd::new(wasm_bytes)
}

/// Create an N-dimensional model executor from WASM bytes.
#[cfg(feature = "native")]
pub fn create_model_executor_nd(
    wasm_bytes: &[u8],
) -> Result<NativeModelExecutorNd, WasmBackendError> {
    NativeModelExecutorNd::new(wasm_bytes)
}

/// Detect which ABI version a WASM module uses.
///
/// Returns `AbiVersion::Nd` if the module exports `sample` and `get_dimensions`,
/// otherwise returns `AbiVersion::Legacy` if it exports `is_inside`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbiVersion {
    /// Legacy 3D ABI: is_inside(x, y, z) and separate get_bounds_* functions
    Legacy,
    /// New N-dimensional ABI: sample(pos_ptr), get_bounds(out_ptr), get_dimensions()
    Nd,
}

/// Detect ABI version by examining WASM exports.
#[cfg(feature = "native")]
pub fn detect_abi_version(wasm_bytes: &[u8]) -> Result<AbiVersion, WasmBackendError> {
    use wasmtime::{Engine, Module};

    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes)
        .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

    let has_sample = module.exports().any(|e| e.name() == "sample");
    let has_get_dimensions = module.exports().any(|e| e.name() == "get_dimensions");
    let has_memory = module.exports().any(|e| e.name() == "memory");

    if has_sample && has_get_dimensions && has_memory {
        Ok(AbiVersion::Nd)
    } else {
        Ok(AbiVersion::Legacy)
    }
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
