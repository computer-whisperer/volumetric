//! Native (wasmtime) implementation of ModelExecutor.

use crate::wasm::error::WasmBackendError;
use crate::wasm::traits::{ModelBounds, ModelExecutor};
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

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
