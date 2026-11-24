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
//! Models use the N-dimensional ABI
//! (get_dimensions/get_io_ptr/get_bounds/sample/memory), plus the optional
//! typed-channel exports (get_sample_format/sample_channels).
//!
//! # Threads
//!
//! In a threaded build (wasm32 with atomics, Web Workers over one shared
//! memory) each worker has a JS global of its own, so the embedder must
//! load the helper in every worker that runs a model, and a handle only
//! means something on the thread that created it. The model types
//! therefore keep the model's bytes, not a handle, and each thread creates
//! its own instance the first time it uses one (see `PerThreadModel`). The
//! native parallel sampler also gives each thread its own instance; unlike
//! it, a thread here keeps one per live model and compiles the module
//! itself, in its own helper's cache.

mod js_bindings;

use crate::wasm::error::WasmBackendError;
use crate::wasm::traits::{
    ModelBounds, ModelBoundsNd, ModelExecutor, OperatorExecutor, OperatorIo, ParallelModelSampler,
};

#[cfg(feature = "web")]
use js_bindings::{
    JsWasmHandle, wasm_model_create_sync, wasm_model_destroy, wasm_model_get_bounds,
    wasm_model_get_dimensions, wasm_model_get_sample_format, wasm_model_has_sample_channels,
    wasm_model_sample, wasm_model_sample_channels_nd, wasm_model_sample_nd, wasm_operator_create,
    wasm_operator_destroy, wasm_operator_get_error, wasm_operator_get_metadata,
    wasm_operator_get_output, wasm_operator_get_output_indices, wasm_operator_get_warnings,
    wasm_operator_run,
};
#[cfg(feature = "web")]
use std::cell::RefCell;
#[cfg(feature = "web")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "web")]
use std::sync::{Arc, Weak};
#[cfg(feature = "web")]
use volumetric_abi::SampleFormat;

/// What the JS helper's failure to create an instance reads as.
#[cfg(feature = "web")]
const CREATE_FAILED: &str = "JavaScript failed to create WASM instance";

/// Fetch a model's N-dimensional bounds through the JS bridge.
#[cfg(feature = "web")]
fn fetch_bounds_nd(handle: JsWasmHandle) -> Result<ModelBoundsNd, WasmBackendError> {
    let dimensions = wasm_model_get_dimensions(handle) as usize;
    if dimensions == 0 {
        return Err(WasmBackendError::Execution(
            "model reported 0 dimensions".to_string(),
        ));
    }

    let bounds = wasm_model_get_bounds(handle)
        .ok_or_else(|| WasmBackendError::Execution("Failed to get bounds from WASM".to_string()))?;

    if bounds.len() != dimensions * 2 {
        return Err(WasmBackendError::Execution(format!(
            "Expected {} bounds values, got {}",
            dimensions * 2,
            bounds.len()
        )));
    }

    Ok(ModelBoundsNd::new(bounds))
}

/// Convert N-dimensional bounds to 3D, erroring on lower-dimensional models.
#[cfg(feature = "web")]
fn bounds_3d(bounds: &ModelBoundsNd) -> Result<ModelBounds, WasmBackendError> {
    if bounds.dimensions() < 3 {
        return Err(WasmBackendError::Execution(format!(
            "model has only {} dimensions, need at least 3",
            bounds.dimensions()
        )));
    }
    Ok(bounds.to_3d())
}

/// Source of fresh [`PerThreadModel`] ids.
#[cfg(feature = "web")]
static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

/// One thread's instance of one [`PerThreadModel`].
#[cfg(feature = "web")]
struct ThreadInstance {
    id: u64,
    /// Dead once the owner is dropped. The thread that drops the owner
    /// destroys its own instance then; any other thread destroys its
    /// instance on its next miss.
    owner: Weak<()>,
    /// `None` records that instantiation failed on this thread, so it is
    /// counted once and not retried per call.
    handle: Option<JsWasmHandle>,
}

#[cfg(feature = "web")]
thread_local! {
    /// This thread's instances, a few entries long: one per model in use.
    static THREAD_INSTANCES: RefCell<Vec<ThreadInstance>> = const { RefCell::new(Vec::new()) };
}

/// A model's bytes plus a lazily created instance on each thread that uses
/// it. Handles live in a thread-local table keyed by this model's id, so the
/// value itself is plain shareable data.
#[cfg(feature = "web")]
struct PerThreadModel {
    id: u64,
    wasm_bytes: Vec<u8>,
    /// Liveness token; threads hold a `Weak` to it.
    alive: Arc<()>,
    /// Threads on which instantiation failed.
    init_failures: AtomicU64,
}

#[cfg(feature = "web")]
impl PerThreadModel {
    /// Keep `wasm_bytes` and instantiate them on the calling thread,
    /// returning that thread's handle for the caller to validate.
    fn new(wasm_bytes: &[u8]) -> Result<(Self, JsWasmHandle), WasmBackendError> {
        let model = Self {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            wasm_bytes: wasm_bytes.to_vec(),
            alive: Arc::new(()),
            init_failures: AtomicU64::new(0),
        };
        let handle = model
            .handle()
            .ok_or_else(|| WasmBackendError::Instantiation(CREATE_FAILED.to_string()))?;
        Ok((model, handle))
    }

    /// The calling thread's handle, created on first use. `None` when
    /// instantiation failed on this thread.
    #[inline]
    fn handle(&self) -> Option<JsWasmHandle> {
        let hit = THREAD_INSTANCES.with(|table| {
            table
                .borrow()
                .iter()
                .find(|entry| entry.id == self.id)
                .map(|entry| entry.handle)
        });
        match hit {
            Some(handle) => handle,
            None => self.instantiate(),
        }
    }

    /// Miss path: destroy this thread's instances whose owner is gone, then
    /// create one for `self`.
    ///
    /// The table is never borrowed across a call into the helper. Such a
    /// call can throw (a worker that never loaded the helper), and a throw
    /// leaves the frames above it without running their destructors, so a
    /// borrow held here would stay held for the rest of the thread's life.
    #[cold]
    fn instantiate(&self) -> Option<JsWasmHandle> {
        let mut dead = Vec::new();
        THREAD_INSTANCES.with(|table| {
            table.borrow_mut().retain(|entry| {
                let live = entry.owner.strong_count() > 0;
                if !live {
                    dead.extend(entry.handle);
                }
                live
            });
        });
        for handle in dead {
            wasm_model_destroy(handle);
        }
        let handle = match wasm_model_create_sync(&self.wasm_bytes) {
            0 => {
                self.init_failures.fetch_add(1, Ordering::Relaxed);
                None
            }
            handle => Some(handle),
        };
        THREAD_INSTANCES.with(|table| {
            table.borrow_mut().push(ThreadInstance {
                id: self.id,
                owner: Arc::downgrade(&self.alive),
                handle,
            });
        });
        handle
    }
}

#[cfg(feature = "web")]
impl Drop for PerThreadModel {
    /// Destroy the dropping thread's instance now; other threads' go on
    /// their next miss, since only the creating thread can destroy a handle.
    fn drop(&mut self) {
        // `try_with`: the table may already be gone during thread teardown.
        // The entry is taken out before the helper is called, for the
        // reason given on `instantiate`.
        let entry = THREAD_INSTANCES.try_with(|table| {
            let mut table = table.borrow_mut();
            let i = table.iter().position(|entry| entry.id == self.id)?;
            Some(table.swap_remove(i))
        });
        if let Ok(Some(ThreadInstance {
            handle: Some(handle),
            ..
        })) = entry
        {
            wasm_model_destroy(handle);
        }
    }
}

/// Web model executor using JavaScript bridge.
///
/// This implementation uses the browser's WebAssembly API via JavaScript
/// to instantiate and execute model WASM modules. Each thread that uses
/// the executor gets its own instance (see `PerThreadModel`).
#[cfg(feature = "web")]
pub struct WebModelExecutor {
    model: PerThreadModel,
    bounds: ModelBoundsNd,
    sample_format: SampleFormat,
    has_sample_channels: bool,
}

#[cfg(feature = "web")]
impl WebModelExecutor {
    /// Create a new executor from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let (model, handle) = PerThreadModel::new(wasm_bytes)?;
        Self::load(model, handle)
    }

    /// Fetch and validate the model's metadata from a freshly created
    /// handle, mirroring the checks the native executor makes at load.
    /// On error, dropping `model` destroys the handle.
    fn load(model: PerThreadModel, handle: JsWasmHandle) -> Result<Self, WasmBackendError> {
        let bounds = fetch_bounds_nd(handle)?;
        let sample_format = match wasm_model_get_sample_format(handle) {
            None => SampleFormat::default(),
            Some(bytes) => {
                volumetric_abi::decode_sample_format(&bytes).map_err(WasmBackendError::Execution)?
            }
        };
        let has_sample_channels = wasm_model_has_sample_channels(handle);
        if sample_format.channels.len() > 1 {
            if !has_sample_channels {
                return Err(WasmBackendError::MissingExport(format!(
                    "sample_channels (format declares {} channels)",
                    sample_format.channels.len()
                )));
            }
            // Channel output goes in the second half of the IO buffer (the
            // first n f64s hold the position), so it must fit there.
            if sample_format.channels.len() * 4 > bounds.dimensions() * 8 {
                return Err(WasmBackendError::Execution(format!(
                    "{} channels exceed the IO buffer's output capacity ({} f32s)",
                    sample_format.channels.len(),
                    bounds.dimensions() * 2
                )));
            }
        }
        Ok(Self {
            model,
            bounds,
            sample_format,
            has_sample_channels,
        })
    }

    /// This thread's handle, or an instantiation error.
    fn handle(&self) -> Result<JsWasmHandle, WasmBackendError> {
        self.model
            .handle()
            .ok_or_else(|| WasmBackendError::Instantiation(CREATE_FAILED.to_string()))
    }
}

#[cfg(feature = "web")]
impl ModelExecutor for WebModelExecutor {
    fn get_bounds(&mut self) -> Result<ModelBounds, WasmBackendError> {
        bounds_3d(&self.bounds)
    }

    fn is_inside(&mut self, x: f64, y: f64, z: f64) -> Result<f32, WasmBackendError> {
        let result = wasm_model_sample(self.handle()?, x, y, z);
        if result.is_nan() {
            return Err(WasmBackendError::Execution(
                "WASM sample returned NaN".to_string(),
            ));
        }
        Ok(result)
    }

    fn sample_format(&mut self) -> Result<SampleFormat, WasmBackendError> {
        Ok(self.sample_format.clone())
    }

    fn dimensions(&mut self) -> Result<u32, WasmBackendError> {
        Ok(self.bounds.dimensions() as u32)
    }

    fn get_bounds_nd(&mut self) -> Result<ModelBoundsNd, WasmBackendError> {
        Ok(self.bounds.clone())
    }

    fn sample_nd(&mut self, position: &[f64]) -> Result<f32, WasmBackendError> {
        let result = wasm_model_sample_nd(self.handle()?, position);
        if result.is_nan() {
            return Err(WasmBackendError::Execution(
                "WASM sample returned NaN".to_string(),
            ));
        }
        Ok(result)
    }

    fn sample_channels_nd(&mut self, position: &[f64]) -> Result<Vec<f32>, WasmBackendError> {
        if !self.has_sample_channels {
            // Occupancy-only model: plain `sample` is the whole row (same
            // fallback the native executor uses).
            return Ok(vec![self.sample_nd(position)?]);
        }
        wasm_model_sample_channels_nd(
            self.handle()?,
            position,
            self.sample_format.channels.len() as u32,
        )
        .ok_or_else(|| WasmBackendError::Execution("WASM sample_channels failed".to_string()))
    }
}

/// Web parallel sampler using JavaScript bridge.
///
/// Each thread that samples gets its own instance (see `PerThreadModel`):
/// the creating thread's is made by `new`, any other thread's on its first
/// sample. A thread whose helper cannot create one is counted once in
/// `instantiation_failures` and its samples read as "outside".
#[cfg(feature = "web")]
pub struct WebParallelSampler {
    model: PerThreadModel,
    bounds: ModelBoundsNd,
    /// Samples the JS bridge reported as errored (NaN), read as "outside".
    errors: AtomicU64,
}

#[cfg(feature = "web")]
impl WebParallelSampler {
    /// Create a new parallel sampler from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let (model, handle) = PerThreadModel::new(wasm_bytes)?;
        // On error, dropping `model` destroys the handle.
        let bounds = fetch_bounds_nd(handle)?;
        Ok(Self {
            model,
            bounds,
            errors: AtomicU64::new(0),
        })
    }
}

#[cfg(feature = "web")]
impl ParallelModelSampler for WebParallelSampler {
    fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        let Some(handle) = self.model.handle() else {
            // No instance on this thread (counted in `instantiation_failures`).
            return 0.0;
        };
        let result = wasm_model_sample(handle, x, y, z);
        if result.is_nan() {
            self.errors.fetch_add(1, Ordering::Relaxed);
            0.0 // Return 0 (outside) on error
        } else {
            result
        }
    }

    fn get_bounds(&self) -> Result<ModelBounds, WasmBackendError> {
        bounds_3d(&self.bounds)
    }

    fn instantiation_failures(&self) -> u64 {
        self.model.init_failures.load(Ordering::Relaxed)
    }

    fn instantiation_failure_detail(&self) -> Option<String> {
        // The helper logs the cause to the console and reports only failure.
        (self.instantiation_failures() > 0).then(|| CREATE_FAILED.to_string())
    }

    fn sample_traps(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }
}

// The traits require these; they hold by structure (bytes, metadata, atomics).
#[cfg(feature = "web")]
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<WebParallelSampler>();
    send::<WebModelExecutor>();
};

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
        let reported = wasm_operator_get_error(handle);
        if !success || reported.is_some() {
            wasm_operator_destroy(handle);
            return Err(match reported {
                Some(msg) => WasmBackendError::OperatorReported(msg),
                None => WasmBackendError::Execution("Operator run() failed".to_string()),
            });
        }

        // Collect outputs and any posted warnings
        let mut result = OperatorIo {
            inputs: io.inputs,
            outputs: std::collections::HashMap::new(),
            warnings: wasm_operator_get_warnings(handle),
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
