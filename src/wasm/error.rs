//! Error types for WASM backend abstraction.

use std::fmt;

/// Error type for WASM backend operations.
#[derive(Debug)]
pub enum WasmBackendError {
    /// Failed to compile or instantiate a WASM module.
    Instantiation(String),
    /// Failed to call a WASM function.
    Execution(String),
    /// A required export was not found in the WASM module.
    MissingExport(String),
    /// Memory access error (out of bounds, etc.).
    Memory(String),
    /// CBOR encoding/decoding error.
    Cbor(String),
    /// The backend is not available (e.g., native backend in web context).
    Unavailable(String),
}

impl fmt::Display for WasmBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WasmBackendError::Instantiation(msg) => write!(f, "WASM instantiation error: {}", msg),
            WasmBackendError::Execution(msg) => write!(f, "WASM execution error: {}", msg),
            WasmBackendError::MissingExport(name) => write!(f, "Missing WASM export: {}", name),
            WasmBackendError::Memory(msg) => write!(f, "WASM memory error: {}", msg),
            WasmBackendError::Cbor(msg) => write!(f, "CBOR error: {}", msg),
            WasmBackendError::Unavailable(msg) => write!(f, "Backend unavailable: {}", msg),
        }
    }
}

impl std::error::Error for WasmBackendError {}

#[cfg(feature = "native")]
impl From<wasmtime::Error> for WasmBackendError {
    fn from(err: wasmtime::Error) -> Self {
        WasmBackendError::Execution(err.to_string())
    }
}
