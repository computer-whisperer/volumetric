//! Native (wasmtime) implementation of OperatorExecutor.

use crate::wasm::error::WasmBackendError;
use crate::wasm::native::module_cache::operator_cache;
use crate::wasm::traits::{ModelExecutor, OperatorExecutor, OperatorIo};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;
use wasmtime::{Caller, Engine, Linker, Module, Store, UpdateDeadline};

/// How often a run's watchdog polls its cancel flag. Bounds both the
/// cancellation latency and the extra wall time a completed run spends
/// waiting for its watchdog to notice and exit.
const CANCEL_POLL_PERIOD: Duration = Duration::from_millis(50);

/// State held in the WASM Store during operator execution.
struct OperatorState {
    inputs: Vec<Vec<u8>>,
    outputs: HashMap<usize, Vec<u8>>,
    /// First error message the operator reported via `host.post_error`.
    error: Option<String>,
    /// Lazily-created model executors backing the `input_model_*` sampling
    /// imports, keyed by input slot. `None` records a failed creation so a
    /// bad input isn't recompiled on every call.
    models: HashMap<usize, Option<Box<dyn ModelExecutor>>>,
    /// Set by this run's watchdog when its cancel flag fires. The engine
    /// epoch is shared by every concurrently-running operator, so on an
    /// epoch bump each store's deadline callback consults this to decide
    /// whether *it* is the run being cancelled.
    cancelled: Arc<AtomicBool>,
}

/// Get (or lazily create) the model executor for input slot `idx`.
fn model_for(state: &mut OperatorState, idx: usize) -> Option<&mut Box<dyn ModelExecutor>> {
    if !state.models.contains_key(&idx) {
        // The concrete executor type (not create_model_executor's impl-trait
        // return, which captures the byte slice's lifetime); same legacy
        // rejection as the public constructor.
        let created = state.inputs.get(idx).and_then(|bytes| {
            crate::wasm::reject_legacy_model(bytes).ok()?;
            crate::wasm::native::NativeModelExecutor::new(bytes)
                .ok()
                .map(|executor| Box::new(executor) as Box<dyn ModelExecutor>)
        });
        state.models.insert(idx, created);
    }
    state.models.get_mut(&idx).and_then(|slot| slot.as_mut())
}

/// Native operator executor using wasmtime.
pub struct NativeOperatorExecutor {
    engine: Engine,
    module: Module,
}

impl NativeOperatorExecutor {
    /// Create a new executor from WASM bytes.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let cache = operator_cache();
        let engine = cache.engine().clone();
        let module = cache.get_or_compile(wasm_bytes)?;

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
                    let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory())
                    else {
                        return;
                    };
                    // Split-borrow the memory and host state so the input can
                    // be copied straight into WASM memory without cloning it.
                    let (mem_data, state) = memory.data_and_store_mut(&mut caller);
                    let Some(src_data) = state.inputs.get(idx as usize) else {
                        return;
                    };
                    let copy_len = (len as usize).min(src_data.len());
                    let dest_start = ptr as usize;
                    let Some(dest_end) = dest_start.checked_add(copy_len) else {
                        return;
                    };
                    if let Some(dest) = mem_data.get_mut(dest_start..dest_end) {
                        dest.copy_from_slice(&src_data[..copy_len]);
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

        // Host function: number of dimensions of the model in an input slot
        // (0 when the slot doesn't hold a usable model).
        linker
            .func_wrap(
                "host",
                "input_model_dimensions",
                |mut caller: Caller<'_, OperatorState>, idx: i32| -> i32 {
                    model_for(caller.data_mut(), idx as usize)
                        .and_then(|model| model.dimensions().ok())
                        .map(|n| n as i32)
                        .unwrap_or(0)
                },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Host function: write a model input's 2n interleaved f64 bounds
        // into operator memory. Returns 1 on success, 0 on failure.
        linker
            .func_wrap(
                "host",
                "input_model_bounds",
                |mut caller: Caller<'_, OperatorState>, idx: i32, out_ptr: i32| -> i32 {
                    let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory())
                    else {
                        return 0;
                    };
                    let (mem_data, state) = memory.data_and_store_mut(&mut caller);
                    let Some(model) = model_for(state, idx as usize) else {
                        return 0;
                    };
                    let Ok(bounds) = model.get_bounds_nd() else {
                        return 0;
                    };
                    let bytes: Vec<u8> = bounds
                        .as_slice()
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    let start = out_ptr as usize;
                    let Some(end) = start.checked_add(bytes.len()) else {
                        return 0;
                    };
                    let Some(dest) = mem_data.get_mut(start..end) else {
                        return 0;
                    };
                    dest.copy_from_slice(&bytes);
                    1
                },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Host function: sample a model input at `count` positions (n f64s
        // each, read at pos_ptr), writing one occupancy f32 per position at
        // out_ptr. Failed individual samples follow the ABI convention and
        // read as 0.0 (outside); returns 1 on success, 0 when the slot is
        // not a model or a pointer is out of range.
        linker
            .func_wrap(
                "host",
                "input_model_sample",
                |mut caller: Caller<'_, OperatorState>,
                 idx: i32,
                 pos_ptr: i32,
                 count: i32,
                 out_ptr: i32|
                 -> i32 {
                    let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory())
                    else {
                        return 0;
                    };
                    let (mem_data, state) = memory.data_and_store_mut(&mut caller);
                    let Some(model) = model_for(state, idx as usize) else {
                        return 0;
                    };
                    let Ok(n) = model.dimensions() else {
                        return 0;
                    };
                    let (n, count) = (n as usize, count as usize);

                    let pos_start = pos_ptr as usize;
                    let Some(pos_len) = count.checked_mul(n * 8) else {
                        return 0;
                    };
                    let Some(pos_end) = pos_start.checked_add(pos_len) else {
                        return 0;
                    };
                    let out_start = out_ptr as usize;
                    let Some(out_end) = out_start.checked_add(count * 4) else {
                        return 0;
                    };
                    if pos_end > mem_data.len() || out_end > mem_data.len() {
                        return 0;
                    }

                    let positions: Vec<f64> = mem_data[pos_start..pos_end]
                        .chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let mut samples = Vec::with_capacity(count);
                    for pos in positions.chunks_exact(n) {
                        samples.push(model.sample_nd(pos).unwrap_or(0.0));
                    }
                    for (i, sample) in samples.iter().enumerate() {
                        mem_data[out_start + i * 4..out_start + (i + 1) * 4]
                            .copy_from_slice(&sample.to_le_bytes());
                    }
                    1
                },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Host function: report a failure (UTF-8 message in WASM memory).
        // Only the first reported error is kept.
        linker
            .func_wrap(
                "host",
                "post_error",
                |mut caller: Caller<'_, OperatorState>, ptr: i32, len: i32| {
                    if caller.data().error.is_some() {
                        return;
                    }
                    if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory())
                    {
                        let mem_data = memory.data(&caller);
                        let src_start = ptr as usize;
                        let src_end = src_start.saturating_add(len as usize);
                        if src_end <= mem_data.len() {
                            let msg =
                                String::from_utf8_lossy(&mem_data[src_start..src_end]).into_owned();
                            caller.data_mut().error = Some(msg);
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
        static NEVER: AtomicBool = AtomicBool::new(false);
        self.run_cancellable(io, &NEVER)
    }

    fn run_cancellable(
        &mut self,
        io: OperatorIo,
        cancel: &AtomicBool,
    ) -> Result<OperatorIo, WasmBackendError> {
        let linker = self.create_linker()?;

        let cancelled = Arc::new(AtomicBool::new(false));
        let state = OperatorState {
            inputs: io.inputs,
            outputs: HashMap::new(),
            error: None,
            models: HashMap::new(),
            cancelled: Arc::clone(&cancelled),
        };
        let mut store = Store::new(&self.engine, state);
        // The engine has epoch interruption enabled, so the store needs a
        // deadline or it traps at the first check. The epoch only advances
        // when some run's watchdog cancels, at which point every running
        // store hits its callback: the cancelled one traps out, the rest
        // extend their deadline and keep going.
        store.set_epoch_deadline(1);
        store.epoch_deadline_callback(|ctx| {
            if ctx.data().cancelled.load(Ordering::Relaxed) {
                Err(wasmtime::Error::msg("operator run cancelled"))
            } else {
                Ok(UpdateDeadline::Continue(1))
            }
        });

        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let run_func = instance
            .get_typed_func::<(), ()>(&mut store, "run")
            .map_err(|e| WasmBackendError::MissingExport(format!("run: {}", e)))?;

        // Watchdog: polls the caller's cancel flag while wasm runs; on
        // cancel it marks this store and bumps the engine epoch, which is
        // the only thing that can interrupt a wasm call in progress.
        // Dropping the sender wakes the watchdog as soon as the run ends.
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let engine = &self.engine;
        let watchdog_cancelled = Arc::clone(&cancelled);
        let call_result = std::thread::scope(|scope| {
            scope.spawn(move || {
                loop {
                    match done_rx.recv_timeout(CANCEL_POLL_PERIOD) {
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if cancel.load(Ordering::Relaxed) {
                                watchdog_cancelled.store(true, Ordering::Relaxed);
                                engine.increment_epoch();
                                return;
                            }
                        }
                        _ => return,
                    }
                }
            });
            let result = run_func.call(&mut store, ());
            drop(done_tx);
            result
        });

        if let Err(e) = call_result {
            return Err(if cancelled.load(Ordering::Relaxed) {
                WasmBackendError::Cancelled
            } else {
                WasmBackendError::Execution(e.to_string())
            });
        }

        let state = store.into_data();
        if let Some(msg) = state.error {
            return Err(WasmBackendError::OperatorReported(msg));
        }
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

        linker
            .func_wrap(
                "host",
                "post_error",
                |_caller: Caller<'_, ()>, _ptr: i32, _len: i32| {},
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Sampling imports stub to failure during metadata retrieval (there
        // are no inputs to sample).
        linker
            .func_wrap(
                "host",
                "input_model_dimensions",
                |_caller: Caller<'_, ()>, _idx: i32| -> i32 { 0 },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        linker
            .func_wrap(
                "host",
                "input_model_bounds",
                |_caller: Caller<'_, ()>, _idx: i32, _out_ptr: i32| -> i32 { 0 },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        linker
            .func_wrap(
                "host",
                "input_model_sample",
                |_caller: Caller<'_, ()>,
                 _idx: i32,
                 _pos_ptr: i32,
                 _count: i32,
                 _out_ptr: i32|
                 -> i32 { 0 },
            )
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let mut store = Store::new(&self.engine, ());
        // Metadata retrieval is never cancelled, but the epoch-interruption
        // engine still requires a deadline, and a concurrent run being
        // cancelled bumps the shared epoch — keep this store alive through
        // that.
        store.set_epoch_deadline(1);
        store.epoch_deadline_callback(|_| Ok(UpdateDeadline::Continue(1)));
        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let metadata_func = instance
            .get_typed_func::<(), i64>(&mut store, "get_metadata")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_metadata: {}", e)))?;

        let packed = metadata_func
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        let (ptr, len) = volumetric_abi::unpack_ptr_len(packed);

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

#[cfg(test)]
mod tests {
    use super::*;

    /// An operator whose `run` spins forever — only an epoch trap ends it.
    fn spinning_operator() -> NativeOperatorExecutor {
        let wasm = wat::parse_str(
            r#"(module
                (memory (export "memory") 1)
                (func (export "run") (loop $spin (br $spin))))"#,
        )
        .unwrap();
        NativeOperatorExecutor::new(&wasm).unwrap()
    }

    #[test]
    fn cancel_interrupts_a_running_operator() {
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            flag.store(true, Ordering::Relaxed);
        });

        let result = spinning_operator().run_cancellable(OperatorIo::new(vec![]), &cancel);
        assert!(
            matches!(result, Err(WasmBackendError::Cancelled)),
            "{result:?}"
        );
    }

    #[test]
    fn uncancelled_runs_complete_normally() {
        let wasm = wat::parse_str(
            r#"(module
                (memory (export "memory") 1)
                (func (export "run")))"#,
        )
        .unwrap();
        let cancel = AtomicBool::new(false);
        NativeOperatorExecutor::new(&wasm)
            .unwrap()
            .run_cancellable(OperatorIo::new(vec![]), &cancel)
            .expect("run should succeed");
    }

    /// Cancelling one run must not trap other runs sharing the engine: the
    /// epoch bump reaches every live store, and the deadline callback keeps
    /// the uncancelled ones going.
    #[test]
    fn cancelling_one_run_leaves_concurrent_runs_alive() {
        let spawn_run = |cancel: &Arc<AtomicBool>| {
            let cancel = Arc::clone(cancel);
            std::thread::spawn(move || {
                spinning_operator().run_cancellable(OperatorIo::new(vec![]), &cancel)
            })
        };
        let (cancel_a, cancel_b) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        );
        let run_a = spawn_run(&cancel_a);
        let run_b = spawn_run(&cancel_b);
        // Let both runs reach their spin loops before bumping the epoch.
        std::thread::sleep(Duration::from_millis(100));

        cancel_a.store(true, Ordering::Relaxed);
        let result_a = run_a.join().unwrap();
        assert!(
            matches!(result_a, Err(WasmBackendError::Cancelled)),
            "{result_a:?}"
        );

        // B saw A's epoch bump; had its callback mis-trapped it, it would
        // have finished with an Execution error by now.
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !run_b.is_finished(),
            "run B should have survived A's cancel"
        );

        cancel_b.store(true, Ordering::Relaxed);
        let result_b = run_b.join().unwrap();
        assert!(
            matches!(result_b, Err(WasmBackendError::Cancelled)),
            "{result_b:?}"
        );
    }
}
