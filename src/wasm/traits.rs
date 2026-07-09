//! Trait definitions for WASM backend abstraction.
//!
//! These traits define the interface for executing WASM modules, allowing
//! different backends (native wasmtime, web JavaScript bridge) to be used
//! interchangeably.

use super::error::WasmBackendError;
use std::collections::HashMap;

/// Bounding box for a volumetric model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelBounds {
    pub min: (f64, f64, f64),
    pub max: (f64, f64, f64),
}

impl ModelBounds {
    /// Create new model bounds from min/max corners.
    pub fn new(min: (f64, f64, f64), max: (f64, f64, f64)) -> Self {
        Self { min, max }
    }

    /// Convert to f32 tuples.
    pub fn as_f32(&self) -> ((f32, f32, f32), (f32, f32, f32)) {
        (
            (self.min.0 as f32, self.min.1 as f32, self.min.2 as f32),
            (self.max.0 as f32, self.max.1 as f32, self.max.2 as f32),
        )
    }
}

/// N-dimensional bounding box for volumetric models.
///
/// Stores bounds as interleaved min/max pairs: [min_0, max_0, min_1, max_1, ...]
#[derive(Clone, Debug, PartialEq)]
pub struct ModelBoundsNd {
    /// Interleaved min/max bounds: [min_0, max_0, min_1, max_1, ..., min_n-1, max_n-1]
    bounds: Vec<f64>,
}

impl ModelBoundsNd {
    /// Create new n-dimensional bounds from interleaved min/max values.
    ///
    /// The input should be `[min_0, max_0, min_1, max_1, ...]` with length `2 * n`.
    pub fn new(bounds: Vec<f64>) -> Self {
        assert!(bounds.len() % 2 == 0, "bounds must have even length");
        Self { bounds }
    }

    /// Create 3D bounds from min/max corners.
    pub fn from_3d(min: (f64, f64, f64), max: (f64, f64, f64)) -> Self {
        Self {
            bounds: vec![min.0, max.0, min.1, max.1, min.2, max.2],
        }
    }

    /// Returns the number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.bounds.len() / 2
    }

    /// Get the minimum bound for a given dimension.
    pub fn min(&self, dim: usize) -> f64 {
        self.bounds[dim * 2]
    }

    /// Get the maximum bound for a given dimension.
    pub fn max(&self, dim: usize) -> f64 {
        self.bounds[dim * 2 + 1]
    }

    /// Get the raw interleaved bounds.
    pub fn as_slice(&self) -> &[f64] {
        &self.bounds
    }

    /// Convert to 3D ModelBounds.
    ///
    /// Panics if dimensions < 3.
    pub fn to_3d(&self) -> ModelBounds {
        assert!(
            self.dimensions() >= 3,
            "need at least 3 dimensions for to_3d"
        );
        ModelBounds::new(
            (self.min(0), self.min(1), self.min(2)),
            (self.max(0), self.max(1), self.max(2)),
        )
    }

    /// Convert to f32 tuples (3D subset).
    ///
    /// Panics if dimensions < 3.
    pub fn as_f32(&self) -> ((f32, f32, f32), (f32, f32, f32)) {
        self.to_3d().as_f32()
    }
}

/// Executor for volumetric model WASM modules.
///
/// This trait represents a single-threaded executor for WASM modules that
/// export the N-dimensional model interface (`get_dimensions`, `get_io_ptr`,
/// `get_bounds`, `sample`) documented in the `volumetric_abi` crate.
pub trait ModelExecutor: Send {
    /// Get the bounding box of the model.
    fn get_bounds(&mut self) -> Result<ModelBounds, WasmBackendError>;

    /// Sample the occupancy at a point.
    ///
    /// Classify the result with [`volumetric_abi::is_occupied`] (inside iff
    /// `> 0.5`); the magnitude carries no meaning. Models emit the canonical
    /// values `1.0`/`0.0`.
    fn is_inside(&mut self, x: f64, y: f64, z: f64) -> Result<f32, WasmBackendError>;

    /// The model's declared per-sample format.
    ///
    /// Models without a `get_sample_format` export get the default
    /// occupancy-only format.
    fn sample_format(&mut self) -> Result<volumetric_abi::SampleFormat, WasmBackendError> {
        Ok(volumetric_abi::SampleFormat::default())
    }

    /// The model's number of dimensions.
    fn dimensions(&mut self) -> Result<u32, WasmBackendError> {
        Err(WasmBackendError::Unavailable(
            "N-dimensional access not supported by this backend yet".to_string(),
        ))
    }

    /// The model's N-dimensional bounding box.
    fn get_bounds_nd(&mut self) -> Result<ModelBoundsNd, WasmBackendError> {
        Err(WasmBackendError::Unavailable(
            "N-dimensional access not supported by this backend yet".to_string(),
        ))
    }

    /// Sample the occupancy at an N-dimensional position (classify with
    /// [`volumetric_abi::is_occupied`]).
    fn sample_nd(&mut self, _position: &[f64]) -> Result<f32, WasmBackendError> {
        Err(WasmBackendError::Unavailable(
            "N-dimensional access not supported by this backend yet".to_string(),
        ))
    }
}

/// Thread-safe sampler for parallel volumetric model sampling.
///
/// This trait represents a sampler that can be safely used from multiple threads
/// concurrently. Implementations typically use thread-local WASM instances that
/// share a pre-compiled module.
///
/// The `sample` method intentionally returns `f32` directly (not `Result`) because:
/// 1. It's called millions of times during meshing and error handling overhead matters
/// 2. The implementation can log/handle errors internally and return a default (0.0)
/// 3. The meshing algorithms are designed to handle occasional bad samples gracefully
pub trait ParallelModelSampler: Send + Sync {
    /// Sample the occupancy at a point.
    ///
    /// Classify the result with [`volumetric_abi::is_occupied`] (inside iff
    /// `> 0.5`). On error, implementations return 0.0 — errors read as
    /// "outside".
    fn sample(&self, x: f64, y: f64, z: f64) -> f32;

    /// Get the bounding box of the model.
    fn get_bounds(&self) -> Result<ModelBounds, WasmBackendError>;
}

/// I/O state for operator execution.
#[derive(Clone, Debug, Default)]
pub struct OperatorIo {
    /// Input data indexed by input slot.
    pub inputs: Vec<Vec<u8>>,
    /// Output data indexed by output slot, populated after execution.
    pub outputs: HashMap<usize, Vec<u8>>,
}

impl OperatorIo {
    /// Create new I/O state with the given inputs.
    pub fn new(inputs: Vec<Vec<u8>>) -> Self {
        Self {
            inputs,
            outputs: HashMap::new(),
        }
    }

    /// Get the length of input at the given index.
    pub fn get_input_len(&self, idx: usize) -> usize {
        self.inputs.get(idx).map(|v| v.len()).unwrap_or(0)
    }

    /// Get a slice of input data at the given index.
    pub fn get_input_data(&self, idx: usize) -> Option<&[u8]> {
        self.inputs.get(idx).map(|v| v.as_slice())
    }

    /// Store output data for the given index.
    pub fn post_output(&mut self, idx: usize, data: Vec<u8>) {
        self.outputs.insert(idx, data);
    }
}

/// Executor for operator WASM modules.
///
/// Operators are WASM modules that transform inputs (model WASM, configuration)
/// into outputs (transformed model WASM).
///
/// # Required WASM Exports
/// - `run()` - Execute the operator
/// - `get_metadata() -> i64` - Return ptr|len packed metadata
///
/// # Required WASM Imports (from "host")
/// - `get_input_len(i32) -> u32` - Get length of input at index
/// - `get_input_data(i32, i32, i32)` - Copy input data to WASM memory
/// - `post_output(i32, i32, i32)` - Post output data from WASM memory
///
/// # Optional WASM Imports (from "host")
/// - `post_error(ptr: i32, len: i32)` - Report a failure with a UTF-8 message.
///   A run that posts an error fails with the message instead of returning
///   outputs; only the first posted error is kept.
/// - `input_model_dimensions(i32) -> i32`, `input_model_bounds(i32, i32) ->
///   i32`, `input_model_sample(i32, i32, i32, i32) -> i32` - Evaluate a
///   `ModelWASM` input without instantiating it inside the operator; the
///   host services the calls with a native model executor. See the
///   `volumetric_abi` crate docs for the full contract. Currently native
///   only - the web backend's JS bridge does not provide them yet, so
///   operators that import them fail to instantiate there.
pub trait OperatorExecutor: Send {
    /// Execute the operator with the given I/O state.
    fn run(&mut self, io: OperatorIo) -> Result<OperatorIo, WasmBackendError>;

    /// [`run`](Self::run), interruptible: aborts with
    /// [`WasmBackendError::Cancelled`] shortly after `cancel` becomes true,
    /// even mid-execution. Backends without an interruption mechanism (the
    /// web bridge) fall back to an uninterruptible run.
    fn run_cancellable(
        &mut self,
        io: OperatorIo,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<OperatorIo, WasmBackendError> {
        let _ = cancel;
        self.run(io)
    }

    /// Get the operator's metadata as CBOR bytes.
    fn get_metadata(&mut self) -> Result<Vec<u8>, WasmBackendError>;
}
