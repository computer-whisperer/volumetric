//! Native (wasmtime) implementation of ModelExecutor.

use crate::wasm::error::WasmBackendError;
use crate::wasm::native::module_cache::model_cache;
use crate::wasm::traits::{ModelBounds, ModelBoundsNd, ModelExecutor};
use volumetric_abi::SampleFormat;
use wasmtime::{Instance, Memory, Store, TypedFunc};

/// Native model executor using wasmtime.
///
/// Models use the N-dimensional ABI:
/// - `get_dimensions() -> u32`: Returns number of dimensions
/// - `get_io_ptr() -> i32`: Returns the model-owned IO buffer (>= 2n f64s)
/// - `get_bounds(out_ptr: i32)`: Writes 2n f64 values (interleaved min/max)
/// - `sample(pos_ptr: i32) -> f32`: Reads n f64 values, returns occupancy
///   (inside iff > 0.5; see `volumetric_abi::is_occupied`)
/// - `memory` export required
///
/// Optional typed-channel exports (see the `volumetric_abi` docs):
/// - `get_sample_format() -> i64`: ptr|len-packed CBOR `SampleFormat`
/// - `sample_channels(pos_ptr: i32, out_ptr: i32)`: one f32 per channel
pub struct NativeModelExecutor {
    store: Store<()>,
    memory: Memory,
    dimensions: u32,
    io_ptr: i32,
    get_bounds: TypedFunc<i32, ()>,
    sample: TypedFunc<i32, f32>,
    sample_format: SampleFormat,
    sample_channels: Option<TypedFunc<(i32, i32), ()>>,
}

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

        let get_io_ptr = instance
            .get_typed_func::<(), i32>(&mut store, "get_io_ptr")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_io_ptr: {}", e)))?;

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

        // Ask the model where its IO buffer lives
        let io_ptr = get_io_ptr
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
        validate_io_ptr(io_ptr, dimensions, memory.data_size(&store))?;

        // Optional typed-channel exports; absent means occupancy-only.
        let sample_format =
            match instance.get_typed_func::<(), i64>(&mut store, "get_sample_format") {
                Err(_) => SampleFormat::default(),
                Ok(get_sample_format) => {
                    let packed = get_sample_format
                        .call(&mut store, ())
                        .map_err(|e| WasmBackendError::Execution(e.to_string()))?;
                    let (ptr, len) = volumetric_abi::unpack_ptr_len(packed);
                    let mem_data = memory.data(&store);
                    let end = ptr.checked_add(len).filter(|&end| end <= mem_data.len());
                    let Some(end) = end else {
                        return Err(WasmBackendError::Execution(format!(
                            "get_sample_format returned out-of-bounds region ({ptr}+{len})"
                        )));
                    };
                    volumetric_abi::decode_sample_format(&mem_data[ptr..end])
                        .map_err(WasmBackendError::Execution)?
                }
            };

        let sample_channels = instance
            .get_typed_func::<(i32, i32), ()>(&mut store, "sample_channels")
            .ok();
        if sample_format.channels.len() > 1 {
            if sample_channels.is_none() {
                return Err(WasmBackendError::MissingExport(format!(
                    "sample_channels (format declares {} channels)",
                    sample_format.channels.len()
                )));
            }
            // Channel output goes in the second half of the IO buffer (the
            // first n f64s hold the position), so it must fit there.
            if sample_format.channels.len() * 4 > dimensions as usize * 8 {
                return Err(WasmBackendError::Execution(format!(
                    "{} channels exceed the IO buffer's output capacity ({} f32s)",
                    sample_format.channels.len(),
                    dimensions * 2
                )));
            }
        }

        Ok(Self {
            store,
            memory,
            dimensions,
            io_ptr,
            get_bounds,
            sample,
            sample_format,
            sample_channels,
        })
    }

    /// The model's declared per-sample format (default: occupancy-only).
    pub fn sample_format(&self) -> &SampleFormat {
        &self.sample_format
    }

    /// Get the number of dimensions.
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Get the N-dimensional bounding box.
    pub fn get_bounds_nd(&mut self) -> Result<ModelBoundsNd, WasmBackendError> {
        // Call get_bounds to write bounds into the model's IO buffer
        self.get_bounds
            .call(&mut self.store, self.io_ptr)
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        // Read bounds from memory (2n f64 values)
        let n = self.dimensions as usize;
        let byte_count = n * 2 * 8;
        let mut bounds = vec![0.0f64; n * 2];

        let mem_data = self.memory.data(&self.store);
        let offset = self.io_ptr as usize;
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

        // Write position into the model's IO buffer
        {
            let mem_data = self.memory.data_mut(&mut self.store);
            let offset = self.io_ptr as usize;
            for (i, &val) in position.iter().enumerate() {
                let start = offset + i * 8;
                mem_data[start..start + 8].copy_from_slice(&val.to_le_bytes());
            }
        }

        // Call sample
        self.sample
            .call(&mut self.store, self.io_ptr)
            .map_err(|e| WasmBackendError::Execution(e.to_string()))
    }

    /// Sample every declared channel at the given N-dimensional position,
    /// in format order (channel 0 is occupancy).
    pub fn sample_channels_nd(&mut self, position: &[f64]) -> Result<Vec<f32>, WasmBackendError> {
        let Some(ref sample_channels) = self.sample_channels else {
            // Occupancy-only model: plain `sample` is the whole row.
            return Ok(vec![self.sample_nd(position)?]);
        };

        let n = self.dimensions as usize;
        if position.len() != n {
            return Err(WasmBackendError::Execution(format!(
                "position has {} dimensions, expected {}",
                position.len(),
                n
            )));
        }

        // Position in the first half of the IO buffer, channel output in the
        // second half (capacity checked at load).
        let out_ptr = self.io_ptr + (n * 8) as i32;
        {
            let mem_data = self.memory.data_mut(&mut self.store);
            let offset = self.io_ptr as usize;
            for (i, &val) in position.iter().enumerate() {
                let start = offset + i * 8;
                mem_data[start..start + 8].copy_from_slice(&val.to_le_bytes());
            }
        }

        sample_channels
            .call(&mut self.store, (self.io_ptr, out_ptr))
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        let mem_data = self.memory.data(&self.store);
        let offset = out_ptr as usize;
        self.sample_format
            .channels
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let start = offset + i * 4;
                mem_data[start..start + 4]
                    .try_into()
                    .map(f32::from_le_bytes)
                    .map_err(|_| {
                        WasmBackendError::Execution("failed to read channel output".to_string())
                    })
            })
            .collect()
    }
}

/// Sanity-check the pointer a model returned from `get_io_ptr`.
///
/// The buffer must be nonzero (address 0 is a null pointer to the model's own
/// Rust code) and hold `2 * dims` f64s within the exported memory.
pub(crate) fn validate_io_ptr(
    io_ptr: i32,
    dimensions: u32,
    memory_size: usize,
) -> Result<(), WasmBackendError> {
    let needed = dimensions as usize * 2 * 8;
    if io_ptr <= 0 || (io_ptr as usize).saturating_add(needed) > memory_size {
        return Err(WasmBackendError::Execution(format!(
            "model returned invalid IO buffer pointer {io_ptr} \
             (need {needed} bytes within {memory_size} bytes of memory)"
        )));
    }
    Ok(())
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
        let n = self.dimensions as usize;
        if n < 3 {
            return Err(WasmBackendError::Execution(format!(
                "model has {n} dimensions; 3D sampling needs at least 3 \
                 (extrude 2D sketches before meshing)"
            )));
        }
        // Pad position with zeros if model has more than 3 dimensions
        let mut pos = vec![0.0f64; n];
        pos[0] = x;
        pos[1] = y;
        pos[2] = z;
        self.sample_nd(&pos)
    }

    fn sample_format(&mut self) -> Result<SampleFormat, WasmBackendError> {
        Ok(self.sample_format.clone())
    }

    fn dimensions(&mut self) -> Result<u32, WasmBackendError> {
        Ok(self.dimensions)
    }

    fn get_bounds_nd(&mut self) -> Result<ModelBoundsNd, WasmBackendError> {
        NativeModelExecutor::get_bounds_nd(self)
    }

    fn sample_nd(&mut self, position: &[f64]) -> Result<f32, WasmBackendError> {
        NativeModelExecutor::sample_nd(self, position)
    }

    fn sample_channels_nd(&mut self, position: &[f64]) -> Result<Vec<f32>, WasmBackendError> {
        NativeModelExecutor::sample_channels_nd(self, position)
    }
}
