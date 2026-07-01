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

/// Memory buffer offset for position input.
/// Must be nonzero: address 0 is a null pointer to the model's Rust code, and
/// debug builds trap on null-pointer dereference.
const POS_BUFFER_OFFSET: i32 = 8;
/// Memory buffer offset for bounds output.
const BOUNDS_BUFFER_OFFSET: i32 = 256;

/// Thread-local WASM execution context.
struct ThreadLocalContext {
    store: Store<()>,
    memory: Memory,
    dimensions: u32,
    sample: TypedFunc<i32, f32>,
}

impl ThreadLocalContext {
    fn new(engine: &Engine, module: &Module, dimensions: u32) -> Option<Self> {
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, module, &[]).ok()?;

        let memory = instance.get_memory(&mut store, "memory")?;
        let sample = instance
            .get_typed_func::<i32, f32>(&mut store, "sample")
            .ok()?;

        Some(Self {
            store,
            memory,
            dimensions,
            sample,
        })
    }

    fn sample(&mut self, x: f64, y: f64, z: f64) -> f32 {
        // Write position to memory (pad with zeros for extra dimensions)
        {
            let mem_data = self.memory.data_mut(&mut self.store);
            let offset = POS_BUFFER_OFFSET as usize;

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

        // Call sample
        self.sample
            .call(&mut self.store, POS_BUFFER_OFFSET)
            .unwrap_or(0.0)
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
}

impl NativeParallelSampler {
    /// Create a new parallel sampler from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let cache = model_cache();
        let engine = cache.engine().clone();
        let module = cache.get_or_compile(wasm_bytes)?;

        // Get dimensions and bounds from a temporary instance
        let (dimensions, bounds) = Self::fetch_dimensions_and_bounds(&engine, &module)?;

        let id = SAMPLER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            id,
            engine,
            module,
            dimensions,
            bounds,
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

        let get_bounds = instance
            .get_typed_func::<i32, ()>(&mut store, "get_bounds")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds: {}", e)))?;

        // Get dimensions
        let dimensions = get_dimensions
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        // Get bounds
        get_bounds
            .call(&mut store, BOUNDS_BUFFER_OFFSET)
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        // Read bounds from memory
        let n = dimensions as usize;
        let byte_count = n * 2 * 8;
        let mut bounds_vec = vec![0.0f64; n * 2];

        let mem_data = memory.data(&store);
        let offset = BOUNDS_BUFFER_OFFSET as usize;
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
        // Stores (sampler_id, context) so we can detect when to reinitialize.
        thread_local! {
            static CONTEXT: std::cell::RefCell<Option<(u64, ThreadLocalContext)>> =
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
                match ThreadLocalContext::new(&self.engine, &self.module, self.dimensions) {
                    Some(ctx) => *opt = Some((self.id, ctx)),
                    None => return 0.0,
                }
            }

            if let Some((_, ctx)) = opt.as_mut() {
                ctx.sample(x, y, z)
            } else {
                0.0
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
}
