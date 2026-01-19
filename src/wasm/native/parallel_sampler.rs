//! Native (wasmtime) implementation of ParallelModelSampler.
//!
//! This implementation uses thread-local WASM instances that share a pre-compiled
//! module, enabling efficient parallel sampling without mutex contention.

use crate::wasm::error::WasmBackendError;
use crate::wasm::traits::{ModelBounds, ModelBoundsNd, ParallelModelSampler};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

/// Global counter for assigning unique IDs to samplers.
static SAMPLER_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Thread-local WASM execution context.
struct ThreadLocalContext {
    store: Store<()>,
    is_inside: TypedFunc<(f64, f64, f64), f32>,
}

impl ThreadLocalContext {
    fn new(engine: &Engine, module: &Module) -> Option<Self> {
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, module, &[]).ok()?;
        let is_inside = instance
            .get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")
            .ok()?;
        Some(Self { store, is_inside })
    }

    fn sample(&mut self, x: f64, y: f64, z: f64) -> f32 {
        self.is_inside
            .call(&mut self.store, (x, y, z))
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
    engine: Arc<Engine>,
    module: Arc<Module>,
    bounds: ModelBounds,
}

impl NativeParallelSampler {
    /// Create a new parallel sampler from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Get bounds from a temporary instance
        let bounds = Self::fetch_bounds(&engine, &module)?;

        let id = SAMPLER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            id,
            engine: Arc::new(engine),
            module: Arc::new(module),
            bounds,
        })
    }

    fn fetch_bounds(engine: &Engine, module: &Module) -> Result<ModelBounds, WasmBackendError> {
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, module, &[])
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let get_min_x = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_min_x: {}", e)))?;
        let get_min_y = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_min_y: {}", e)))?;
        let get_min_z = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_min_z: {}", e)))?;
        let get_max_x = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_max_x: {}", e)))?;
        let get_max_y = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_max_y: {}", e)))?;
        let get_max_z = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_max_z: {}", e)))?;

        let min_x = get_min_x
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let min_y = get_min_y
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let min_z = get_min_z
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let max_x = get_max_x
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let max_y = get_max_y
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let max_z = get_max_z
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        Ok(ModelBounds::new(
            (min_x, min_y, min_z),
            (max_x, max_y, max_z),
        ))
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
                match ThreadLocalContext::new(&self.engine, &self.module) {
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
        Ok(self.bounds)
    }
}

// =============================================================================
// N-Dimensional Parallel Sampler
// =============================================================================

/// Memory buffer offset for position input in N-dimensional ABI
const POS_BUFFER_OFFSET_ND: i32 = 0;
/// Memory buffer offset for bounds output in N-dimensional ABI
const BOUNDS_BUFFER_OFFSET_ND: i32 = 256;

/// Thread-local WASM execution context for N-dimensional ABI.
struct ThreadLocalContextNd {
    store: Store<()>,
    memory: Memory,
    dimensions: u32,
    sample: TypedFunc<i32, f32>,
}

impl ThreadLocalContextNd {
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
            let offset = POS_BUFFER_OFFSET_ND as usize;

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
            .call(&mut self.store, POS_BUFFER_OFFSET_ND)
            .unwrap_or(0.0)
    }
}

/// Native parallel model sampler for N-dimensional ABI using wasmtime.
///
/// Each thread maintains its own WASM Store and Instance, initialized lazily
/// on first sample call. This avoids the overhead of creating a new instance
/// for every sample while still allowing parallel access.
pub struct NativeParallelSamplerNd {
    id: u64,
    engine: Arc<Engine>,
    module: Arc<Module>,
    dimensions: u32,
    bounds: ModelBoundsNd,
}

impl NativeParallelSamplerNd {
    /// Create a new N-dimensional parallel sampler from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Get dimensions and bounds from a temporary instance
        let (dimensions, bounds) = Self::fetch_dimensions_and_bounds(&engine, &module)?;

        let id = SAMPLER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            id,
            engine: Arc::new(engine),
            module: Arc::new(module),
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
            .call(&mut store, BOUNDS_BUFFER_OFFSET_ND)
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        // Read bounds from memory
        let n = dimensions as usize;
        let byte_count = n * 2 * 8;
        let mut bounds_vec = vec![0.0f64; n * 2];

        let mem_data = memory.data(&store);
        let offset = BOUNDS_BUFFER_OFFSET_ND as usize;
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

impl ParallelModelSampler for NativeParallelSamplerNd {
    fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        // Thread-local storage for the WASM context.
        thread_local! {
            static CONTEXT_ND: std::cell::RefCell<Option<(u64, ThreadLocalContextNd)>> =
                const { std::cell::RefCell::new(None) };
        }

        CONTEXT_ND.with(|cell| {
            let mut opt = cell.borrow_mut();

            // Check if we need to (re)initialize the context
            let needs_init = match &*opt {
                Some((cached_id, _)) => *cached_id != self.id,
                None => true,
            };

            if needs_init {
                match ThreadLocalContextNd::new(&self.engine, &self.module, self.dimensions) {
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
