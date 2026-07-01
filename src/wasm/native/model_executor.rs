//! Native (wasmtime) implementation of ModelExecutor.

use crate::wasm::error::WasmBackendError;
use crate::wasm::native::module_cache::model_cache;
use crate::wasm::traits::{ModelBounds, ModelBoundsNd, ModelExecutor};
use wasmtime::{Instance, Memory, Store, TypedFunc};

/// Native model executor using wasmtime.
///
/// Models use the N-dimensional ABI:
/// - `get_dimensions() -> u32`: Returns number of dimensions
/// - `get_bounds(out_ptr: i32)`: Writes 2n f64 values (interleaved min/max)
/// - `sample(pos_ptr: i32) -> f32`: Reads n f64 values, returns density
/// - `memory` export required
pub struct NativeModelExecutor {
    store: Store<()>,
    memory: Memory,
    dimensions: u32,
    get_bounds: TypedFunc<i32, ()>,
    sample: TypedFunc<i32, f32>,
}

/// Memory buffer offsets for N-dimensional ABI.
/// The position offset must be nonzero: address 0 is a null pointer to the
/// model's Rust code, and debug builds trap on null-pointer dereference.
const POS_BUFFER_OFFSET: i32 = 8;
const BOUNDS_BUFFER_OFFSET: i32 = 256;

impl NativeModelExecutor {
    /// Create a new executor from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let cache = model_cache();
        let module = cache.get_or_compile(wasm_bytes)?;

        let mut store = Store::new(cache.engine(), ());
        let instance = Instance::new(&mut store, &module, &[])
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

        let sample = instance
            .get_typed_func::<i32, f32>(&mut store, "sample")
            .map_err(|e| WasmBackendError::MissingExport(format!("sample: {}", e)))?;

        // Get the number of dimensions
        let dimensions = get_dimensions
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        Ok(Self {
            store,
            memory,
            dimensions,
            get_bounds,
            sample,
        })
    }

    /// Get the number of dimensions.
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Get the N-dimensional bounding box.
    pub fn get_bounds_nd(&mut self) -> Result<ModelBoundsNd, WasmBackendError> {
        // Call get_bounds to write bounds to memory
        self.get_bounds
            .call(&mut self.store, BOUNDS_BUFFER_OFFSET)
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        // Read bounds from memory (2n f64 values)
        let n = self.dimensions as usize;
        let byte_count = n * 2 * 8;
        let mut bounds = vec![0.0f64; n * 2];

        let mem_data = self.memory.data(&self.store);
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
            bounds[i] = f64::from_le_bytes(bytes);
        }

        Ok(ModelBoundsNd::new(bounds))
    }

    /// Sample the density at the given N-dimensional position.
    pub fn sample_nd(&mut self, position: &[f64]) -> Result<f32, WasmBackendError> {
        let n = self.dimensions as usize;
        if position.len() != n {
            return Err(WasmBackendError::Execution(format!(
                "position has {} dimensions, expected {}",
                position.len(),
                n
            )));
        }

        // Write position to memory
        {
            let mem_data = self.memory.data_mut(&mut self.store);
            let offset = POS_BUFFER_OFFSET as usize;
            for (i, &val) in position.iter().enumerate() {
                let start = offset + i * 8;
                mem_data[start..start + 8].copy_from_slice(&val.to_le_bytes());
            }
        }

        // Call sample
        self.sample
            .call(&mut self.store, POS_BUFFER_OFFSET)
            .map_err(|e| WasmBackendError::Execution(e.to_string()))
    }
}

impl ModelExecutor for NativeModelExecutor {
    fn get_bounds(&mut self) -> Result<ModelBounds, WasmBackendError> {
        let bounds_nd = self.get_bounds_nd()?;
        if bounds_nd.dimensions() < 3 {
            return Err(WasmBackendError::Execution(format!(
                "model has only {} dimensions, need at least 3",
                bounds_nd.dimensions()
            )));
        }
        Ok(bounds_nd.to_3d())
    }

    fn is_inside(&mut self, x: f64, y: f64, z: f64) -> Result<f32, WasmBackendError> {
        // Pad position with zeros if model has more than 3 dimensions
        let n = self.dimensions as usize;
        let mut pos = vec![0.0f64; n];
        pos[0] = x;
        pos[1] = y;
        pos[2] = z;
        self.sample_nd(&pos)
    }
}
