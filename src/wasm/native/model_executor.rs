//! Native (wasmtime) implementation of ModelExecutor.

use crate::wasm::error::WasmBackendError;
use crate::wasm::traits::{ModelBounds, ModelBoundsNd, ModelExecutor};
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

/// Native model executor using wasmtime.
pub struct NativeModelExecutor {
    store: Store<()>,
    is_inside: TypedFunc<(f64, f64, f64), f32>,
    get_bounds_min_x: TypedFunc<(), f64>,
    get_bounds_min_y: TypedFunc<(), f64>,
    get_bounds_min_z: TypedFunc<(), f64>,
    get_bounds_max_x: TypedFunc<(), f64>,
    get_bounds_max_y: TypedFunc<(), f64>,
    get_bounds_max_z: TypedFunc<(), f64>,
}

impl NativeModelExecutor {
    /// Create a new executor from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let is_inside = instance
            .get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")
            .map_err(|e| WasmBackendError::MissingExport(format!("is_inside: {}", e)))?;

        let get_bounds_min_x = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_min_x: {}", e)))?;
        let get_bounds_min_y = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_min_y: {}", e)))?;
        let get_bounds_min_z = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_min_z: {}", e)))?;
        let get_bounds_max_x = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_max_x: {}", e)))?;
        let get_bounds_max_y = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_max_y: {}", e)))?;
        let get_bounds_max_z = instance
            .get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_bounds_max_z: {}", e)))?;

        Ok(Self {
            store,
            is_inside,
            get_bounds_min_x,
            get_bounds_min_y,
            get_bounds_min_z,
            get_bounds_max_x,
            get_bounds_max_y,
            get_bounds_max_z,
        })
    }
}

impl ModelExecutor for NativeModelExecutor {
    fn get_bounds(&mut self) -> Result<ModelBounds, WasmBackendError> {
        let min_x = self
            .get_bounds_min_x
            .call(&mut self.store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let min_y = self
            .get_bounds_min_y
            .call(&mut self.store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let min_z = self
            .get_bounds_min_z
            .call(&mut self.store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let max_x = self
            .get_bounds_max_x
            .call(&mut self.store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let max_y = self
            .get_bounds_max_y
            .call(&mut self.store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        let max_z = self
            .get_bounds_max_z
            .call(&mut self.store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        Ok(ModelBounds::new(
            (min_x, min_y, min_z),
            (max_x, max_y, max_z),
        ))
    }

    fn is_inside(&mut self, x: f64, y: f64, z: f64) -> Result<f32, WasmBackendError> {
        self.is_inside
            .call(&mut self.store, (x, y, z))
            .map_err(|e| WasmBackendError::Execution(e.to_string()))
    }
}

/// Native model executor for N-dimensional ABI using wasmtime.
///
/// This executor supports the new N-dimensional ABI:
/// - `get_dimensions() -> u32`: Returns number of dimensions
/// - `get_bounds(out_ptr: i32)`: Writes 2n f64 values (interleaved min/max)
/// - `sample(pos_ptr: i32) -> f32`: Reads n f64 values, returns density
/// - `memory` export required
pub struct NativeModelExecutorNd {
    store: Store<()>,
    memory: Memory,
    dimensions: u32,
    get_dimensions: TypedFunc<(), u32>,
    get_bounds: TypedFunc<i32, ()>,
    sample: TypedFunc<i32, f32>,
}

/// Memory buffer offsets for N-dimensional ABI
const POS_BUFFER_OFFSET: i32 = 0;
const BOUNDS_BUFFER_OFFSET: i32 = 256;

impl NativeModelExecutorNd {
    /// Create a new N-dimensional executor from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let mut store = Store::new(&engine, ());
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
            get_dimensions,
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

impl ModelExecutor for NativeModelExecutorNd {
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
