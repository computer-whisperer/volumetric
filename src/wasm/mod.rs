//! WASM backend abstraction layer.
//!
//! This module provides traits and implementations for executing WASM modules
//! in different environments (native desktop via wasmtime, web via JavaScript bridge).
//!
//! Models use the N-dimensional ABI:
//! - `get_dimensions() -> u32`
//! - `get_io_ptr() -> i32` — model-owned IO buffer (>= `2 * dims` f64s); the
//!   host writes positions into it and reads bounds from it
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
pub mod variant;

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

/// Reject models built against outdated ABIs with a clear error.
///
/// Legacy models exported `is_inside(f64, f64, f64) -> f32` plus six
/// `get_bounds_*` getters. Support was removed; every model in this repo (and
/// every operator output) uses the N-dimensional ABI. Without this check a
/// legacy blob from an old saved project would fail with a bare
/// "missing export: get_dimensions".
///
/// Early N-dimensional models predate `get_io_ptr` (the host used to write
/// into hardcoded low-memory offsets); those also need a rebuild.
fn reject_legacy_model(wasm_bytes: &[u8]) -> Result<(), WasmBackendError> {
    let mut has_sample = false;
    let mut has_is_inside = false;
    let mut has_get_io_ptr = false;

    for payload in wasmparser::Parser::new(0).parse_all(wasm_bytes) {
        let payload = payload.map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;
        if let wasmparser::Payload::ExportSection(section) = payload {
            for export in section {
                let export = export.map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;
                match export.name {
                    "sample" => has_sample = true,
                    "is_inside" => has_is_inside = true,
                    "get_io_ptr" => has_get_io_ptr = true,
                    _ => {}
                }
            }
        }
    }

    if has_is_inside && !has_sample {
        return Err(WasmBackendError::Instantiation(
            "model exports the legacy 3D ABI (is_inside/get_bounds_*), which is no longer \
             supported; rebuild the model against the N-dimensional ABI \
             (get_dimensions/get_io_ptr/get_bounds/sample/memory)"
                .to_string(),
        ));
    }
    if has_sample && !has_get_io_ptr {
        return Err(WasmBackendError::Instantiation(
            "model predates the model-owned IO buffer ABI (missing `get_io_ptr` export); \
             rebuild the model against the current N-dimensional ABI \
             (get_dimensions/get_io_ptr/get_bounds/sample/memory)"
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a minimal module whose export section names `names` as
    /// functions. Only the export scan runs on it, so the (dangling)
    /// function indices never get validated.
    fn module_exporting(names: &[&str]) -> Vec<u8> {
        let mut body = vec![names.len() as u8];
        for name in names {
            body.push(name.len() as u8);
            body.extend_from_slice(name.as_bytes());
            body.push(0x00); // kind: func
            body.push(0x00); // index 0
        }
        let mut wasm = b"\0asm\x01\0\0\0".to_vec();
        wasm.push(7); // export section id
        wasm.push(body.len() as u8);
        wasm.extend_from_slice(&body);
        wasm
    }

    #[test]
    fn legacy_3d_models_are_rejected() {
        let err = reject_legacy_model(&module_exporting(&["is_inside", "get_bounds_min_x"]))
            .expect_err("legacy models must be rejected");
        assert!(err.to_string().contains("legacy 3D ABI"), "{err}");
    }

    #[test]
    fn models_without_get_io_ptr_are_rejected() {
        let err = reject_legacy_model(&module_exporting(&["get_dimensions", "sample"]))
            .expect_err("models predating get_io_ptr must be rejected");
        assert!(err.to_string().contains("get_io_ptr"), "{err}");
    }

    #[test]
    fn current_abi_models_pass_the_scan() {
        reject_legacy_model(&module_exporting(&[
            "get_dimensions",
            "get_io_ptr",
            "sample",
        ]))
        .expect("current-ABI models must pass");
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
