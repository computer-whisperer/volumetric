//! Native (wasmtime) implementation of ParallelModelSampler.
//!
//! This implementation uses thread-local WASM instances that share a pre-compiled
//! module, enabling efficient parallel sampling without mutex contention.

use crate::wasm::error::WasmBackendError;
use crate::wasm::native::module_cache::model_cache;
use crate::wasm::traits::{ModelBounds, ModelBoundsNd, ParallelModelSampler};
use std::sync::atomic::{AtomicU64, Ordering};
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

/// Global counter for assigning unique IDs to samplers.
static SAMPLER_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Thread-local WASM execution context.
struct ThreadLocalContext {
    store: Store<()>,
    memory: Memory,
    dimensions: u32,
    io_ptr: i32,
    sample: TypedFunc<i32, f32>,
}

impl ThreadLocalContext {
    fn new(engine: &Engine, module: &Module, dimensions: u32) -> Result<Self, WasmBackendError> {
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, module, &[])
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| WasmBackendError::MissingExport("memory".to_string()))?;
        let get_io_ptr = instance
            .get_typed_func::<(), i32>(&mut store, "get_io_ptr")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_io_ptr: {e}")))?;
        let sample = instance
            .get_typed_func::<i32, f32>(&mut store, "sample")
            .map_err(|e| WasmBackendError::MissingExport(format!("sample: {e}")))?;

        let io_ptr = get_io_ptr
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        super::model_executor::validate_io_ptr(io_ptr, dimensions, memory.data_size(&store))?;

        Ok(Self {
            store,
            memory,
            dimensions,
            io_ptr,
            sample,
        })
    }

    fn sample(&mut self, x: f64, y: f64, z: f64) -> Result<f32, wasmtime::Error> {
        // Write position into the model's IO buffer (pad extra dims with zeros)
        {
            let mem_data = self.memory.data_mut(&mut self.store);
            let offset = self.io_ptr as usize;

            // Write x, y, z
            mem_data[offset..offset + 8].copy_from_slice(&x.to_le_bytes());
            mem_data[offset + 8..offset + 16].copy_from_slice(&y.to_le_bytes());
            mem_data[offset + 16..offset + 24].copy_from_slice(&z.to_le_bytes());

            // Zero out extra dimensions if needed
            for i in 3..self.dimensions as usize {
                let start = offset + i * 8;
                mem_data[start..start + 8].copy_from_slice(&0.0f64.to_le_bytes());
            }
        }

        // Call sample. Traps unwind only this activation; the store stays
        // usable for subsequent calls.
        self.sample.call(&mut self.store, self.io_ptr)
    }
}

/// Native parallel model sampler using wasmtime with thread-local instances.
///
/// Each thread maintains its own WASM Store and Instance, initialized lazily
/// on first sample call. This avoids the overhead of creating a new instance
/// for every sample while still allowing parallel access.
///
/// The sampler uses a unique ID to detect when a new sampler is created,
/// ensuring thread-locals are re-initialized when switching between different
/// WASM modules.
pub struct NativeParallelSampler {
    id: u64,
    engine: Engine,
    module: Module,
    dimensions: u32,
    bounds: ModelBoundsNd,
    /// Threads whose sampling instance failed to instantiate (each such
    /// thread's samples all read as "outside" — see the trait docs; bulk
    /// consumers must reject their output when this is nonzero).
    init_failures: AtomicU64,
    /// The first instantiation error, for diagnostics.
    init_failure_detail: std::sync::Mutex<Option<String>>,
    /// Sample calls that trapped inside the model (read as "outside").
    traps: AtomicU64,
}

impl NativeParallelSampler {
    /// Create a new parallel sampler from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let cache = model_cache();
        let engine = cache.engine().clone();
        let module = cache.get_or_compile(wasm_bytes)?;

        // Get dimensions and bounds from a temporary instance
        let (dimensions, bounds) = Self::fetch_dimensions_and_bounds(&engine, &module)?;

        // The (x, y, z) sampler interface is inherently 3D; feeding it a 2D
        // sketch would silently ignore z. Sketches get meshed only after an
        // extrude-style operator lifts them to 3D.
        if dimensions < 3 {
            return Err(WasmBackendError::Execution(format!(
                "model has {dimensions} dimensions; 3D sampling needs at least 3 \
                 (extrude 2D sketches before meshing)"
            )));
        }

        let id = SAMPLER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            id,
            engine,
            module,
            dimensions,
            bounds,
            init_failures: AtomicU64::new(0),
            init_failure_detail: std::sync::Mutex::new(None),
            traps: AtomicU64::new(0),
        })
    }

    fn fetch_dimensions_and_bounds(
        engine: &Engine,
        module: &Module,
    ) -> Result<(u32, ModelBoundsNd), WasmBackendError> {
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, module, &[])
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| WasmBackendError::MissingExport("memory".to_string()))?;

        let get_dimensions = instance
            .get_typed_func::<(), u32>(&mut store, "get_dimensions")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_dimensions: {}", e)))?;

        let get_io_ptr = instance
            .get_typed_func::<(), i32>(&mut store, "get_io_ptr")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_io_ptr: {}", e)))?;

        let get_bounds = instance
            .get_typed_func::<i32, ()>(&mut store, "get_bounds")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds: {}", e)))?;

        // Get dimensions
        let dimensions = get_dimensions
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        // Ask the model where its IO buffer lives
        let io_ptr = get_io_ptr
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        super::model_executor::validate_io_ptr(io_ptr, dimensions, memory.data_size(&store))?;

        // Get bounds
        get_bounds
            .call(&mut store, io_ptr)
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        // Read bounds from memory
        let n = dimensions as usize;
        let byte_count = n * 2 * 8;
        let mut bounds_vec = vec![0.0f64; n * 2];

        let mem_data = memory.data(&store);
        let offset = io_ptr as usize;
        if offset + byte_count > mem_data.len() {
            return Err(WasmBackendError::Execution(
                "bounds buffer exceeds memory".to_string(),
            ));
        }

        for i in 0..(n * 2) {
            let start = offset + i * 8;
            let bytes: [u8; 8] = mem_data[start..start + 8]
                .try_into()
                .map_err(|_| WasmBackendError::Execution("failed to read bounds".to_string()))?;
            bounds_vec[i] = f64::from_le_bytes(bytes);
        }

        Ok((dimensions, ModelBoundsNd::new(bounds_vec)))
    }

    /// Get the number of dimensions.
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Get the N-dimensional bounds.
    pub fn get_bounds_nd(&self) -> &ModelBoundsNd {
        &self.bounds
    }
}

impl ParallelModelSampler for NativeParallelSampler {
    fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        // Thread-local storage for the WASM context.
        // Stores (sampler_id, Option<context>) so we can detect when to
        // reinitialize; `None` context records a failed instantiation so a
        // doomed run doesn't retry it once per sample. `init_failures`
        // therefore counts failing threads, not failing samples.
        thread_local! {
            static CONTEXT: std::cell::RefCell<Option<(u64, Option<ThreadLocalContext>)>> =
                const { std::cell::RefCell::new(None) };
        }

        CONTEXT.with(|cell| {
            let mut opt = cell.borrow_mut();

            // Check if we need to (re)initialize the context
            let needs_init = match &*opt {
                Some((cached_id, _)) => *cached_id != self.id,
                None => true,
            };

            if needs_init {
                let ctx = match ThreadLocalContext::new(&self.engine, &self.module, self.dimensions)
                {
                    Ok(ctx) => Some(ctx),
                    Err(e) => {
                        self.init_failures.fetch_add(1, Ordering::Relaxed);
                        self.init_failure_detail
                            .lock()
                            .unwrap()
                            .get_or_insert_with(|| e.to_string());
                        None
                    }
                };
                *opt = Some((self.id, ctx));
            }

            match opt.as_mut() {
                Some((_, Some(ctx))) => match ctx.sample(x, y, z) {
                    Ok(value) => value,
                    Err(_) => {
                        self.traps.fetch_add(1, Ordering::Relaxed);
                        0.0
                    }
                },
                // Instantiation failed on this thread (recorded above):
                // there is no instance to consult, so the value is
                // fabricated. Bulk callers reject the run via
                // `instantiation_failures`.
                _ => 0.0,
            }
        })
    }

    fn get_bounds(&self) -> Result<ModelBounds, WasmBackendError> {
        if self.bounds.dimensions() < 3 {
            return Err(WasmBackendError::Execution(format!(
                "model has only {} dimensions, need at least 3",
                self.bounds.dimensions()
            )));
        }
        Ok(self.bounds.to_3d())
    }

    fn instantiation_failures(&self) -> u64 {
        self.init_failures.load(Ordering::Relaxed)
    }

    fn instantiation_failure_detail(&self) -> Option<String> {
        self.init_failure_detail.lock().unwrap().clone()
    }

    fn sample_traps(&self) -> u64 {
        self.traps.load(Ordering::Relaxed)
    }
}
