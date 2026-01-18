//! Native (wasmtime) implementation of OperatorExecutor.

use crate::wasm::error::WasmBackendError;
use crate::wasm::traits::{OperatorExecutor, OperatorIo};
use std::collections::HashMap;
use wasmtime::{Caller, Engine, Linker, Module, Store};

/// State held in the WASM Store during operator execution.
struct OperatorState {
    inputs: Vec<Vec<u8>>,
    outputs: HashMap<usize, Vec<u8>>,
}

/// Native operator executor using wasmtime.
pub struct NativeOperatorExecutor {
    engine: Engine,
    module: Module,
}

impl NativeOperatorExecutor {
    /// Create a new executor from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let engine = Engine::new(wasmtime::Config::new().debug_info(true))
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        Ok(Self { engine, module })
    }

    fn create_linker(&self) -> Result<Linker<OperatorState>, WasmBackendError> {
        let mut linker = Linker::new(&self.engine);

        // Host function: get the length of an input
        linker
            .func_wrap(
                "host",
                "get_input_len",
                |caller: Caller<'_, OperatorState>, idx: i32| -> u32 {
                    caller
                        .data()
                        .inputs
                        .get(idx as usize)
                        .map(|v| v.len() as u32)
                        .unwrap_or(0)
                },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Host function: copy input data into WASM memory
        linker
            .func_wrap(
                "host",
                "get_input_data",
                |mut caller: Caller<'_, OperatorState>, idx: i32, ptr: i32, len: i32| {
                    let data = caller
                        .data()
                        .inputs
                        .get(idx as usize)
                        .cloned();

                    if let Some(src_data) = data {
                        let copy_len = (len as usize).min(src_data.len());
                        if let Some(memory) =
                            caller.get_export("memory").and_then(|e| e.into_memory())
                        {
                            let mem_data = memory.data_mut(&mut caller);
                            let dest_start = ptr as usize;
                            let dest_end = dest_start + copy_len;
                            if dest_end <= mem_data.len() {
                                mem_data[dest_start..dest_end]
                                    .copy_from_slice(&src_data[..copy_len]);
                            }
                        }
                    }
                },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Host function: post output data from WASM memory
        linker
            .func_wrap(
                "host",
                "post_output",
                |mut caller: Caller<'_, OperatorState>, output_idx: i32, ptr: i32, len: i32| {
                    if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory())
                    {
                        let mem_data = memory.data(&caller);
                        let src_start = ptr as usize;
                        let src_end = src_start + len as usize;
                        if src_end <= mem_data.len() {
                            let output_bytes = mem_data[src_start..src_end].to_vec();
                            caller
                                .data_mut()
                                .outputs
                                .insert(output_idx as usize, output_bytes);
                        }
                    }
                },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        Ok(linker)
    }
}

impl OperatorExecutor for NativeOperatorExecutor {
    fn run(&mut self, io: OperatorIo) -> Result<OperatorIo, WasmBackendError> {
        let linker = self.create_linker()?;

        let state = OperatorState {
            inputs: io.inputs,
            outputs: HashMap::new(),
        };
        let mut store = Store::new(&self.engine, state);

        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let run_func = instance
            .get_typed_func::<(), ()>(&mut store, "run")
            .map_err(|e| WasmBackendError::MissingExport(format!("run: {}", e)))?;

        run_func
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        let state = store.into_data();
        Ok(OperatorIo {
            inputs: state.inputs,
            outputs: state.outputs,
        })
    }

    fn get_metadata(&mut self) -> Result<Vec<u8>, WasmBackendError> {
        // Create a linker with stub host functions for metadata retrieval
        let mut linker = Linker::new(&self.engine);

        linker
            .func_wrap(
                "host",
                "get_input_len",
                |_caller: Caller<'_, ()>, _idx: i32| -> u32 { 0 },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        linker
            .func_wrap(
                "host",
                "get_input_data",
                |_caller: Caller<'_, ()>, _idx: i32, _ptr: i32, _len: i32| {},
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        linker
            .func_wrap(
                "host",
                "post_output",
                |_caller: Caller<'_, ()>, _idx: i32, _ptr: i32, _len: i32| {},
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let mut store = Store::new(&self.engine, ());
        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let metadata_func = instance
            .get_typed_func::<(), i64>(&mut store, "get_metadata")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_metadata: {}", e)))?;

        let packed = metadata_func
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))? as u64;

        let ptr = (packed & 0xFFFF_FFFF) as usize;
        let len = (packed >> 32) as usize;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| WasmBackendError::MissingExport("memory".to_string()))?;

        let mem_data = memory.data(&store);
        let end = ptr
            .checked_add(len)
            .ok_or_else(|| WasmBackendError::Memory("Metadata pointer overflow".to_string()))?;

        if end > mem_data.len() {
            return Err(WasmBackendError::Memory(
                "Metadata points outside linear memory".to_string(),
            ));
        }

        Ok(mem_data[ptr..end].to_vec())
    }
}
