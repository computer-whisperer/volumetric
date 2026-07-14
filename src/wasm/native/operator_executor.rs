//! Native (wasmtime) implementation of OperatorExecutor.
//!
//! Two module shapes run here. Plain operators (wasm32-unknown-unknown)
//! own their linear memory and import only the `host.*` functions.
//! Threaded operator variants (wasm32-wasip1-threads) additionally import
//! a *shared* linear memory, a hand-stubbed sliver of
//! `wasi_snapshot_preview1`, and `wasi.thread-spawn` — implemented in this
//! module rather than via `wasmtime-wasi-threads` so every spawned store
//! joins the engine-wide epoch/cancel protocol (see `module_cache.rs`) and
//! the operator import surface stays closed and host-controlled.

use crate::wasm::error::WasmBackendError;
use crate::wasm::native::module_cache::operator_cache;
use crate::wasm::traits::{ModelExecutor, OperatorExecutor, OperatorIo};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;
use wasmtime::{
    Caller, Engine, Extern, Linker, MemoryType, Module, SharedMemory, Store, UpdateDeadline,
};

/// How often a run's watchdog polls its cancel flag. Bounds both the
/// cancellation latency and the extra wall time a completed run spends
/// waiting for its watchdog to notice and exit.
const CANCEL_POLL_PERIOD: Duration = Duration::from_millis(50);

/// How long teardown waits for a threaded run's workers to finish before
/// kicking the epoch, and again after the kick before detaching them.
const THREAD_JOIN_GRACE: Duration = Duration::from_secs(2);

/// State held in the WASM Store during operator execution. Threaded runs
/// create one per instance (main + each spawned thread), and the IO surface
/// (`inputs`, `outputs`, `error`) is shared across a run's stores so guest
/// code can call host imports from whichever pool thread it lands on —
/// rayon's `install` moves execution onto workers. Only `models` stays
/// per-store (each thread lazily instantiates its own sampler).
struct OperatorState {
    inputs: Arc<Vec<Vec<u8>>>,
    outputs: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
    /// First failure of the run: an operator `host.post_error` message or a
    /// dead worker thread's report. First write wins.
    error: Arc<Mutex<Option<String>>>,
    /// Lazily-created model executors backing the `input_model_*` sampling
    /// imports, keyed by input slot. `None` records a failed creation so a
    /// bad input isn't recompiled on every call.
    models: HashMap<usize, Option<Box<dyn ModelExecutor>>>,
    /// Set by this run's watchdog when its cancel flag fires (or by a dying
    /// worker thread). The engine epoch is shared by every concurrently-
    /// running operator, so on an epoch bump each store's deadline callback
    /// consults this to decide whether *it* is the run being cancelled.
    /// Exposed to the guest via `host.cancelled` for cooperative exits.
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

/// A guest module's linear memory: plain, or the shared memory of a
/// threaded variant. Shared memories are a distinct wasmtime extern with
/// interior-mutable contents; copying through them is sound here because
/// the ABI contract makes IO buffers the property of the thread making the
/// host call for the call's duration.
enum GuestMem {
    Plain(wasmtime::Memory),
    Shared(SharedMemory),
}

impl GuestMem {
    fn from_export(export: Option<Extern>) -> Option<Self> {
        match export {
            Some(Extern::Memory(memory)) => Some(Self::Plain(memory)),
            Some(Extern::SharedMemory(memory)) => Some(Self::Shared(memory)),
            _ => None,
        }
    }

    fn from_caller<T>(caller: &mut Caller<'_, T>) -> Option<Self> {
        Self::from_export(caller.get_export("memory"))
    }

    /// Copy `len` guest bytes at `ptr` out of the module's memory.
    fn read(&self, ctx: impl wasmtime::AsContext, ptr: usize, len: usize) -> Option<Vec<u8>> {
        let end = ptr.checked_add(len)?;
        match self {
            Self::Plain(memory) => memory.data(&ctx).get(ptr..end).map(<[u8]>::to_vec),
            Self::Shared(memory) => {
                let cells = memory.data();
                if end > cells.len() {
                    return None;
                }
                let mut out = vec![0u8; len];
                if len > 0 {
                    // Contiguous UnsafeCell<u8> slice; layout matches u8.
                    unsafe {
                        std::ptr::copy_nonoverlapping(cells[ptr].get(), out.as_mut_ptr(), len);
                    }
                }
                Some(out)
            }
        }
    }

    /// Copy `bytes` into the module's memory at `ptr`.
    fn write(&self, mut ctx: impl wasmtime::AsContextMut, ptr: usize, bytes: &[u8]) -> bool {
        let Some(end) = ptr.checked_add(bytes.len()) else {
            return false;
        };
        match self {
            Self::Plain(memory) => match memory.data_mut(&mut ctx).get_mut(ptr..end) {
                Some(dest) => {
                    dest.copy_from_slice(bytes);
                    true
                }
                None => false,
            },
            Self::Shared(memory) => {
                let cells = memory.data();
                if end > cells.len() {
                    return false;
                }
                if !bytes.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            cells[ptr].get(),
                            bytes.len(),
                        );
                    }
                }
                true
            }
        }
    }
}

/// Everything a threaded run shares across its instances: the plumbing a
/// spawned guest thread needs to build its own instance of the module, and
/// the run-wide failure/teardown state.
struct ThreadCtx {
    engine: Engine,
    module: Module,
    memory: SharedMemory,
    inputs: Arc<Vec<Vec<u8>>>,
    outputs: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
    error: Arc<Mutex<Option<String>>>,
    cancelled: Arc<AtomicBool>,
    /// Guest stderr, captured via the `fd_write` stub — Rust panic messages
    /// from the guest arrive here.
    stderr: Arc<Mutex<Vec<u8>>>,
    joins: Mutex<Vec<std::thread::JoinHandle<()>>>,
    next_tid: AtomicI32,
}

/// Body of one spawned guest thread: instantiate the module against the
/// run's shared memory and call its `wasi_thread_start`.
fn spawned_thread_main(ctx: Arc<ThreadCtx>, tid: i32, start_arg: i32) {
    let result = (|| -> Result<(), wasmtime::Error> {
        let mut linker = build_linker(&ctx.engine, Some(&ctx))
            .map_err(|e| wasmtime::Error::msg(e.to_string()))?;
        let state = OperatorState {
            inputs: Arc::clone(&ctx.inputs),
            outputs: Arc::clone(&ctx.outputs),
            error: Arc::clone(&ctx.error),
            models: HashMap::new(),
            cancelled: Arc::clone(&ctx.cancelled),
        };
        let mut store = Store::new(&ctx.engine, state);
        store.set_epoch_deadline(1);
        // Never trap workers on cancel: guest thread pools synchronize
        // through futexes the epoch cannot interrupt, so killing a worker
        // mid-region would strand the threads waiting on its work — the
        // exact deadlock cooperative cancellation (`host.cancelled`) exists
        // to avoid. Cancelled runs exit through the guest's own poll (or
        // the main store's trap as a backstop); workers just finish their
        // current slice.
        store.epoch_deadline_callback(|_| Ok(UpdateDeadline::Continue(1)));
        linker.define(&store, "env", "memory", ctx.memory.clone())?;
        let instance = linker.instantiate(&mut store, &ctx.module)?;
        let start = instance.get_typed_func::<(i32, i32), ()>(&mut store, "wasi_thread_start")?;
        start.call(&mut store, (tid, start_arg))
    })();

    if let Err(e) = result {
        // A worker dying with work on its queue would deadlock the main
        // thread, so fail the whole run. Suppressed when the run is already
        // cancelled — every store traps out then by design.
        if !ctx.cancelled.swap(true, Ordering::Relaxed) {
            let stderr = String::from_utf8_lossy(&ctx.stderr.lock().unwrap())
                .trim()
                .to_string();
            let mut msg = format!("operator worker thread {tid} died: {e}");
            if !stderr.is_empty() {
                msg = format!("{msg}\nguest stderr:\n{stderr}");
            }
            ctx.error.lock().unwrap().get_or_insert(msg);
            ctx.engine.increment_epoch();
        }
    }
}

/// Join a finished run's worker threads. Well-behaved operators tear their
/// thread pool down before returning, making this immediate. A thread
/// parked in a guest futex (`memory.atomic.wait`) cannot be interrupted
/// from the host, and trapping a computing worker would strand its pool's
/// waiters — so stragglers get a bounded grace period and are then
/// detached with a warning rather than hanging the executor.
fn drain_threads(ctx: &ThreadCtx) {
    let handles: Vec<_> = std::mem::take(&mut *ctx.joins.lock().unwrap());
    if handles.is_empty() {
        return;
    }

    let deadline = std::time::Instant::now() + THREAD_JOIN_GRACE;
    while handles.iter().any(|h| !h.is_finished()) {
        if std::time::Instant::now() > deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    let mut leaked = 0usize;
    for handle in handles {
        if handle.is_finished() {
            let _ = handle.join();
        } else {
            leaked += 1;
        }
    }
    if leaked > 0 {
        log::warn!("{leaked} operator worker thread(s) left parked in guest futexes; detached");
    }
}

const ERRNO_SUCCESS: i32 = 0;
const ERRNO_INVAL: i32 = 28;

/// The environ block handed to threaded guests: NUL-terminated KEY=VALUE
/// entries. Thread-pool sizing is the only thing the guest needs from us —
/// `available_parallelism` inside wasm has nothing to measure. A
/// `VOLUMETRIC_THREADS` variable in the host's own environment overrides
/// the core count (daemon tuning, debugging).
fn wasi_env_block() -> Vec<Vec<u8>> {
    let threads = std::env::var("VOLUMETRIC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1)
        });
    [
        format!("RAYON_NUM_THREADS={threads}"),
        format!("VOLUMETRIC_THREADS={threads}"),
    ]
    .into_iter()
    .map(|entry| {
        let mut bytes = entry.into_bytes();
        bytes.push(0);
        bytes
    })
    .collect()
}

/// Define the hand-stubbed `wasi_snapshot_preview1` sliver that
/// wasm32-wasip1-threads modules import: environ (thread-count hints),
/// clocks, deterministic `random_get`, stderr capture, yield, and a
/// trapping `proc_exit`. No filesystem, no args, no real entropy — the
/// import surface stays closed.
fn add_wasi_stubs<T: Send + 'static>(
    linker: &mut Linker<T>,
    env_block: Arc<Vec<Vec<u8>>>,
    stderr: Arc<Mutex<Vec<u8>>>,
) -> Result<(), WasmBackendError> {
    let instantiation = |e: wasmtime::Error| WasmBackendError::Instantiation(e.to_string());

    let env = Arc::clone(&env_block);
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "environ_sizes_get",
            move |mut caller: Caller<'_, T>, count_ptr: i32, size_ptr: i32| -> i32 {
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return ERRNO_INVAL;
                };
                let count = (env.len() as u32).to_le_bytes();
                let size = (env.iter().map(Vec::len).sum::<usize>() as u32).to_le_bytes();
                if mem.write(&mut caller, count_ptr as usize, &count)
                    && mem.write(&mut caller, size_ptr as usize, &size)
                {
                    ERRNO_SUCCESS
                } else {
                    ERRNO_INVAL
                }
            },
        )
        .map_err(instantiation)?;

    let env = Arc::clone(&env_block);
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "environ_get",
            move |mut caller: Caller<'_, T>, environ_ptr: i32, buf_ptr: i32| -> i32 {
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return ERRNO_INVAL;
                };
                let mut entry_ptr = buf_ptr as u32;
                for (i, entry) in env.iter().enumerate() {
                    let slot = environ_ptr as usize + i * 4;
                    if !mem.write(&mut caller, slot, &entry_ptr.to_le_bytes())
                        || !mem.write(&mut caller, entry_ptr as usize, entry)
                    {
                        return ERRNO_INVAL;
                    }
                    entry_ptr += entry.len() as u32;
                }
                ERRNO_SUCCESS
            },
        )
        .map_err(instantiation)?;

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "clock_time_get",
            move |mut caller: Caller<'_, T>, id: i32, _precision: i64, out_ptr: i32| -> i32 {
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return ERRNO_INVAL;
                };
                let nanos: u64 = if id == 0 {
                    // CLOCK_REALTIME
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0)
                } else {
                    // CLOCK_MONOTONIC and friends: nanos since first use.
                    static START: std::sync::OnceLock<std::time::Instant> =
                        std::sync::OnceLock::new();
                    START
                        .get_or_init(std::time::Instant::now)
                        .elapsed()
                        .as_nanos() as u64
                };
                if mem.write(&mut caller, out_ptr as usize, &nanos.to_le_bytes()) {
                    ERRNO_SUCCESS
                } else {
                    ERRNO_INVAL
                }
            },
        )
        .map_err(instantiation)?;

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "random_get",
            move |mut caller: Caller<'_, T>, buf_ptr: i32, len: i32| -> i32 {
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return ERRNO_INVAL;
                };
                // Deterministic splitmix64 stream: guests use this for
                // HashMap seeds, not cryptography, and determinism keeps
                // repeated runs reproducible.
                static SEED: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
                let mut bytes = vec![0u8; len.max(0) as usize];
                for chunk in bytes.chunks_mut(8) {
                    let mut x = SEED.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
                    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    x ^= x >> 31;
                    let n = chunk.len();
                    chunk.copy_from_slice(&x.to_le_bytes()[..n]);
                }
                if mem.write(&mut caller, buf_ptr as usize, &bytes) {
                    ERRNO_SUCCESS
                } else {
                    ERRNO_INVAL
                }
            },
        )
        .map_err(instantiation)?;

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_write",
            move |mut caller: Caller<'_, T>,
                  _fd: i32,
                  iovs_ptr: i32,
                  iovs_len: i32,
                  nwritten_ptr: i32|
                  -> i32 {
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return ERRNO_INVAL;
                };
                let Some(iovs) = mem.read(&caller, iovs_ptr as usize, iovs_len.max(0) as usize * 8)
                else {
                    return ERRNO_INVAL;
                };
                let mut written = 0u32;
                for iov in iovs.chunks_exact(8) {
                    let ptr = u32::from_le_bytes(iov[0..4].try_into().unwrap()) as usize;
                    let len = u32::from_le_bytes(iov[4..8].try_into().unwrap()) as usize;
                    let Some(bytes) = mem.read(&caller, ptr, len) else {
                        return ERRNO_INVAL;
                    };
                    let mut sink = stderr.lock().unwrap();
                    // Both stdout and stderr land in one capture buffer,
                    // bounded so a chatty guest can't balloon the host.
                    if sink.len() < 64 << 10 {
                        sink.extend_from_slice(&bytes);
                    }
                    written += len as u32;
                }
                if mem.write(&mut caller, nwritten_ptr as usize, &written.to_le_bytes()) {
                    ERRNO_SUCCESS
                } else {
                    ERRNO_INVAL
                }
            },
        )
        .map_err(instantiation)?;

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "proc_exit",
            |_caller: Caller<'_, T>, code: i32| -> Result<(), wasmtime::Error> {
                Err(wasmtime::Error::msg(format!(
                    "operator called proc_exit({code})"
                )))
            },
        )
        .map_err(instantiation)?;

    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "sched_yield",
            |_caller: Caller<'_, T>| -> i32 {
                std::thread::yield_now();
                ERRNO_SUCCESS
            },
        )
        .map_err(instantiation)?;

    Ok(())
}

/// Build the full operator linker: the `host.*` imports, and for threaded
/// runs (`threads` present) the wasi stubs plus a live `thread-spawn`.
fn build_linker(
    engine: &Engine,
    threads: Option<&Arc<ThreadCtx>>,
) -> Result<Linker<OperatorState>, WasmBackendError> {
    let mut linker = Linker::new(engine);
    let instantiation = |e: wasmtime::Error| WasmBackendError::Instantiation(e.to_string());

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
        .map_err(instantiation)?;

    // Host function: copy input data into WASM memory
    linker
        .func_wrap(
            "host",
            "get_input_data",
            |mut caller: Caller<'_, OperatorState>, idx: i32, ptr: i32, len: i32| {
                match GuestMem::from_caller(&mut caller) {
                    // Split-borrow the memory and host state so the input is
                    // copied straight into WASM memory without cloning it.
                    Some(GuestMem::Plain(memory)) => {
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
                    }
                    // Shared memories don't borrow the store, so the state
                    // borrow and the copy don't conflict.
                    Some(GuestMem::Shared(memory)) => {
                        let Some(src_data) = caller.data().inputs.get(idx as usize) else {
                            return;
                        };
                        let copy_len = (len as usize).min(src_data.len());
                        let dest_start = ptr as usize;
                        let Some(dest_end) = dest_start.checked_add(copy_len) else {
                            return;
                        };
                        let cells = memory.data();
                        if dest_end > cells.len() {
                            return;
                        }
                        if copy_len > 0 {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    src_data.as_ptr(),
                                    cells[dest_start].get(),
                                    copy_len,
                                );
                            }
                        }
                    }
                    None => {}
                }
            },
        )
        .map_err(instantiation)?;

    // Host function: post output data from WASM memory
    linker
        .func_wrap(
            "host",
            "post_output",
            |mut caller: Caller<'_, OperatorState>, output_idx: i32, ptr: i32, len: i32| {
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return;
                };
                if let Some(bytes) = mem.read(&caller, ptr as usize, len.max(0) as usize) {
                    caller
                        .data()
                        .outputs
                        .lock()
                        .unwrap()
                        .insert(output_idx as usize, bytes);
                }
            },
        )
        .map_err(instantiation)?;

    // Host function: cooperative-cancellation poll. Long solves check this
    // between iterations and unwind cleanly — for threaded operators it is
    // the *primary* cancel path, because guest thread pools synchronize
    // through futexes the epoch trap cannot interrupt.
    linker
        .func_wrap(
            "host",
            "cancelled",
            |caller: Caller<'_, OperatorState>| -> i32 {
                caller.data().cancelled.load(Ordering::Relaxed) as i32
            },
        )
        .map_err(instantiation)?;

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
        .map_err(instantiation)?;

    // Host function: write a model input's 2n interleaved f64 bounds
    // into operator memory. Returns 1 on success, 0 on failure.
    linker
        .func_wrap(
            "host",
            "input_model_bounds",
            |mut caller: Caller<'_, OperatorState>, idx: i32, out_ptr: i32| -> i32 {
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return 0;
                };
                let Some(model) = model_for(caller.data_mut(), idx as usize) else {
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
                mem.write(&mut caller, out_ptr as usize, &bytes) as i32
            },
        )
        .map_err(instantiation)?;

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
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return 0;
                };
                let count = count.max(0) as usize;
                let Some(model) = model_for(caller.data_mut(), idx as usize) else {
                    return 0;
                };
                let Ok(n) = model.dimensions() else {
                    return 0;
                };
                let n = n as usize;

                let Some(pos_len) = count.checked_mul(n * 8) else {
                    return 0;
                };
                let Some(pos_bytes) = mem.read(&caller, pos_ptr as usize, pos_len) else {
                    return 0;
                };
                let positions: Vec<f64> = pos_bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();

                let state = caller.data_mut();
                let Some(model) = model_for(state, idx as usize) else {
                    return 0;
                };
                let mut samples = Vec::with_capacity(count * 4);
                for pos in positions.chunks_exact(n) {
                    let sample = model.sample_nd(pos).unwrap_or(0.0);
                    samples.extend_from_slice(&sample.to_le_bytes());
                }
                mem.write(&mut caller, out_ptr as usize, &samples) as i32
            },
        )
        .map_err(instantiation)?;

    // Host function: report a failure (UTF-8 message in WASM memory).
    // Only the first reported error is kept.
    linker
        .func_wrap(
            "host",
            "post_error",
            |mut caller: Caller<'_, OperatorState>, ptr: i32, len: i32| {
                let Some(mem) = GuestMem::from_caller(&mut caller) else {
                    return;
                };
                let Some(bytes) = mem.read(&caller, ptr as usize, len.max(0) as usize) else {
                    return;
                };
                let msg = String::from_utf8_lossy(&bytes).into_owned();
                caller.data().error.lock().unwrap().get_or_insert(msg);
            },
        )
        .map_err(instantiation)?;

    if let Some(ctx) = threads {
        add_wasi_stubs(
            &mut linker,
            Arc::new(wasi_env_block()),
            Arc::clone(&ctx.stderr),
        )?;

        let ctx = Arc::clone(ctx);
        linker
            .func_wrap(
                "wasi",
                "thread-spawn",
                move |_caller: Caller<'_, OperatorState>, start_arg: i32| -> i32 {
                    let tid = ctx.next_tid.fetch_add(1, Ordering::Relaxed);
                    if tid <= 0 {
                        return -1;
                    }
                    let thread_ctx = Arc::clone(&ctx);
                    match std::thread::Builder::new()
                        .name(format!("operator-worker-{tid}"))
                        .spawn(move || spawned_thread_main(thread_ctx, tid, start_arg))
                    {
                        Ok(handle) => {
                            ctx.joins.lock().unwrap().push(handle);
                            tid
                        }
                        Err(_) => -1,
                    }
                },
            )
            .map_err(instantiation)?;
    }

    Ok(linker)
}

/// Native operator executor using wasmtime.
pub struct NativeOperatorExecutor {
    engine: Engine,
    module: Module,
    /// `Some` when the module imports a shared `env.memory` — the
    /// wasm32-wasip1-threads variant shape. Holds the declared memory type
    /// so each run can allocate a fresh shared memory of the right size.
    shared_memory: Option<MemoryType>,
}

impl NativeOperatorExecutor {
    /// Create a new executor from WASM bytes. Packed operators (see
    /// `wasm::variant`) run their embedded wasm32-wasip1-threads build;
    /// set `VOLUMETRIC_DISABLE_THREADED_OPERATORS=1` to force the
    /// single-threaded baseline for debugging or benchmarking.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, WasmBackendError> {
        let wasm_bytes = match crate::wasm::variant::threaded_variant(wasm_bytes) {
            Some(variant)
                if std::env::var_os("VOLUMETRIC_DISABLE_THREADED_OPERATORS").is_none() =>
            {
                variant
            }
            _ => wasm_bytes,
        };
        let cache = operator_cache();
        let engine = cache.engine().clone();
        let module = cache.get_or_compile(wasm_bytes)?;

        let shared_memory = module.imports().find_map(|import| {
            match (import.module(), import.name(), import.ty()) {
                ("env", "memory", wasmtime::ExternType::Memory(ty)) if ty.is_shared() => Some(ty),
                _ => None,
            }
        });

        Ok(Self {
            engine,
            module,
            shared_memory,
        })
    }

    /// Allocate this run's shared memory and thread context, when the
    /// module is a threaded variant.
    fn create_thread_ctx(
        &self,
        state: &OperatorState,
    ) -> Result<Option<Arc<ThreadCtx>>, WasmBackendError> {
        let Some(memory_type) = &self.shared_memory else {
            return Ok(None);
        };
        let memory = SharedMemory::new(&self.engine, memory_type.clone())
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;
        Ok(Some(Arc::new(ThreadCtx {
            engine: self.engine.clone(),
            module: self.module.clone(),
            memory,
            inputs: Arc::clone(&state.inputs),
            outputs: Arc::clone(&state.outputs),
            error: Arc::clone(&state.error),
            cancelled: Arc::clone(&state.cancelled),
            stderr: Arc::new(Mutex::new(Vec::new())),
            joins: Mutex::new(Vec::new()),
            // wasi-libc's main thread claims the low TIDs, and musl-style
            // locks embed the owner's TID — a spawned thread reusing the
            // main thread's ID makes those locks spin forever ("owner" is
            // never not-self). Start well clear; the wasi-threads spec caps
            // valid TIDs at 0x1FFFFFFF.
            next_tid: AtomicI32::new(1024),
        })))
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
        let inputs = Arc::new(io.inputs);
        let outputs = Arc::new(Mutex::new(HashMap::new()));
        let error = Arc::new(Mutex::new(None));
        let cancelled = Arc::new(AtomicBool::new(false));
        let state = OperatorState {
            inputs: Arc::clone(&inputs),
            outputs: Arc::clone(&outputs),
            error: Arc::clone(&error),
            models: HashMap::new(),
            cancelled: Arc::clone(&cancelled),
        };
        let thread_ctx = self.create_thread_ctx(&state)?;
        let mut linker = build_linker(&self.engine, thread_ctx.as_ref())?;

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

        if let Some(ctx) = &thread_ctx {
            linker
                .define(&store, "env", "memory", ctx.memory.clone())
                .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;
        }

        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        // Reactor protocol: crt initialization (wasi-libc's thread runtime,
        // C constructors) must run before any other export. Only on the
        // main instance — spawned threads initialize via wasi_thread_start.
        if let Ok(init) = instance.get_typed_func::<(), ()>(&mut store, "_initialize") {
            init.call(&mut store, ())
                .map_err(|e| WasmBackendError::Execution(format!("_initialize: {e}")))?;
        }

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

        // Workers must be done before the error slot is read: a dying
        // worker's report races the main call's return otherwise.
        if let Some(ctx) = &thread_ctx {
            drain_threads(ctx);
        }

        if let Err(e) = call_result {
            let reported = error.lock().unwrap().take();
            return Err(if cancel.load(Ordering::Relaxed) {
                WasmBackendError::Cancelled
            } else if let Some(msg) = reported {
                WasmBackendError::OperatorReported(msg)
            } else {
                WasmBackendError::Execution(e.to_string())
            });
        }

        // A cooperative exit (the guest saw host.cancelled and returned) is
        // still a cancellation, whatever it wrote on its way out.
        if cancel.load(Ordering::Relaxed) {
            return Err(WasmBackendError::Cancelled);
        }
        if let Some(msg) = error.lock().unwrap().take() {
            return Err(WasmBackendError::OperatorReported(msg));
        }
        // Release the store's and thread context's handles so the input
        // arc unwraps without copying.
        drop(store);
        drop(thread_ctx);
        Ok(OperatorIo {
            inputs: Arc::try_unwrap(inputs).unwrap_or_else(|arc| arc.as_ref().clone()),
            outputs: std::mem::take(&mut *outputs.lock().unwrap()),
        })
    }

    fn get_metadata(&mut self) -> Result<Vec<u8>, WasmBackendError> {
        // Create a linker with stub host functions for metadata retrieval
        let mut linker = Linker::new(&self.engine);
        let instantiation = |e: wasmtime::Error| WasmBackendError::Instantiation(e.to_string());

        linker
            .func_wrap(
                "host",
                "get_input_len",
                |_caller: Caller<'_, ()>, _idx: i32| -> u32 { 0 },
            )
            .map_err(instantiation)?;

        linker
            .func_wrap(
                "host",
                "get_input_data",
                |_caller: Caller<'_, ()>, _idx: i32, _ptr: i32, _len: i32| {},
            )
            .map_err(instantiation)?;

        linker
            .func_wrap(
                "host",
                "post_output",
                |_caller: Caller<'_, ()>, _idx: i32, _ptr: i32, _len: i32| {},
            )
            .map_err(instantiation)?;

        linker
            .func_wrap(
                "host",
                "post_error",
                |_caller: Caller<'_, ()>, _ptr: i32, _len: i32| {},
            )
            .map_err(instantiation)?;

        linker
            .func_wrap("host", "cancelled", |_caller: Caller<'_, ()>| -> i32 { 0 })
            .map_err(instantiation)?;

        // Sampling imports stub to failure during metadata retrieval (there
        // are no inputs to sample).
        linker
            .func_wrap(
                "host",
                "input_model_dimensions",
                |_caller: Caller<'_, ()>, _idx: i32| -> i32 { 0 },
            )
            .map_err(instantiation)?;

        linker
            .func_wrap(
                "host",
                "input_model_bounds",
                |_caller: Caller<'_, ()>, _idx: i32, _out_ptr: i32| -> i32 { 0 },
            )
            .map_err(instantiation)?;

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
            .map_err(instantiation)?;

        // Threaded variants import the wasi sliver and a shared memory even
        // for metadata; thread-spawn reports failure — metadata retrieval
        // must not compute.
        if let Some(memory_type) = &self.shared_memory {
            add_wasi_stubs(
                &mut linker,
                Arc::new(wasi_env_block()),
                Arc::new(Mutex::new(Vec::new())),
            )?;
            linker
                .func_wrap(
                    "wasi",
                    "thread-spawn",
                    |_caller: Caller<'_, ()>, _start_arg: i32| -> i32 { -1 },
                )
                .map_err(instantiation)?;
            let memory = SharedMemory::new(&self.engine, memory_type.clone())
                .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;
            linker
                .define(&Store::new(&self.engine, ()), "env", "memory", memory)
                .map_err(instantiation)?;
        }

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

        // Reactor protocol, as in run_cancellable.
        if let Ok(init) = instance.get_typed_func::<(), ()>(&mut store, "_initialize") {
            init.call(&mut store, ())
                .map_err(|e| WasmBackendError::Execution(format!("_initialize: {e}")))?;
        }

        let metadata_func = instance
            .get_typed_func::<(), i64>(&mut store, "get_metadata")
            .map_err(|e| WasmBackendError::MissingExport(format!("get_metadata: {}", e)))?;

        let packed = metadata_func
            .call(&mut store, ())
            .map_err(|e| WasmBackendError::Execution(e.to_string()))?;

        let (ptr, len) = volumetric_abi::unpack_ptr_len(packed);

        let mem = GuestMem::from_export(instance.get_export(&mut store, "memory"))
            .ok_or_else(|| WasmBackendError::MissingExport("memory".to_string()))?;
        mem.read(&store, ptr, len).ok_or_else(|| {
            WasmBackendError::Memory("Metadata points outside linear memory".to_string())
        })
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

    /// A bounded spin (~hundreds of millions of iterations) that traps with
    /// `unreachable` when exhausted, so a broken mechanism fails the test
    /// instead of hanging it. `$addr`'s atomic value is polled against
    /// `$target`.
    const WAIT_FOR_COUNTER: &str = r#"
        (func $wait_for (param $target i32)
            (local $fuel i32)
            (local.set $fuel (i32.const 0x20000000))
            (loop $wait
                (br_if 1 (i32.ge_u
                    (i32.atomic.load (i32.const 0))
                    (local.get $target)))
                (local.set $fuel (i32.sub (local.get $fuel) (i32.const 1)))
                (if (i32.eqz (local.get $fuel)) (then unreachable))
                (br $wait)))"#;

    /// Threaded-variant happy path: `run` spawns three guest threads, each
    /// bumps a shared atomic counter, and `run` busy-waits until all three
    /// have run on the shared memory.
    #[test]
    fn threaded_module_spawns_workers_on_shared_memory() {
        let wasm = wat::parse_str(format!(
            r#"(module
                (import "env" "memory" (memory 1 4 shared))
                (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
                (export "memory" (memory 0))
                {WAIT_FOR_COUNTER}
                (func (export "wasi_thread_start") (param $tid i32) (param $arg i32)
                    (drop (i32.atomic.rmw.add (i32.const 0) (i32.const 1))))
                (func (export "run")
                    (local $i i32)
                    (block $spawned
                        (loop $more
                            (br_if $spawned (i32.eq (local.get $i) (i32.const 3)))
                            (if (i32.le_s (call $spawn (local.get $i)) (i32.const 0))
                                (then unreachable))
                            (local.set $i (i32.add (local.get $i) (i32.const 1)))
                            (br $more)))
                    (call $wait_for (i32.const 3))))"#
        ))
        .unwrap();
        let cancel = AtomicBool::new(false);
        NativeOperatorExecutor::new(&wasm)
            .unwrap()
            .run_cancellable(OperatorIo::new(vec![]), &cancel)
            .expect("threaded run should succeed");
    }

    /// A worker that traps must fail the whole run (not deadlock the main
    /// thread waiting on work that will never finish).
    #[test]
    fn dead_worker_fails_the_run() {
        let wasm = wat::parse_str(format!(
            r#"(module
                (import "env" "memory" (memory 1 4 shared))
                (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
                (export "memory" (memory 0))
                {WAIT_FOR_COUNTER}
                (func (export "wasi_thread_start") (param $tid i32) (param $arg i32)
                    unreachable)
                (func (export "run")
                    (drop (call $spawn (i32.const 0)))
                    ;; Never satisfied: the worker dies without counting.
                    (call $wait_for (i32.const 1))))"#
        ))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let result = NativeOperatorExecutor::new(&wasm)
            .unwrap()
            .run_cancellable(OperatorIo::new(vec![]), &cancel);
        match result {
            Err(WasmBackendError::OperatorReported(msg)) => {
                assert!(msg.contains("worker thread"), "{msg}");
            }
            other => panic!("expected a worker-death failure, got {other:?}"),
        }
    }

    /// The host IO imports must work against a shared memory (a distinct
    /// wasmtime extern from plain memories): echo input 0 to output 0.
    #[test]
    fn threaded_module_io_roundtrips_through_shared_memory() {
        let wasm = wat::parse_str(
            r#"(module
                (import "env" "memory" (memory 1 4 shared))
                (import "host" "get_input_len" (func $len (param i32) (result i32)))
                (import "host" "get_input_data" (func $data (param i32 i32 i32)))
                (import "host" "post_output" (func $post (param i32 i32 i32)))
                (export "memory" (memory 0))
                (func (export "wasi_thread_start") (param i32 i32))
                (func (export "run")
                    (local $n i32)
                    (local.set $n (call $len (i32.const 0)))
                    (call $data (i32.const 0) (i32.const 1024) (local.get $n))
                    (call $post (i32.const 0) (i32.const 1024) (local.get $n))))"#,
        )
        .unwrap();
        let cancel = AtomicBool::new(false);
        let io = NativeOperatorExecutor::new(&wasm)
            .unwrap()
            .run_cancellable(OperatorIo::new(vec![b"hello".to_vec()]), &cancel)
            .expect("threaded IO run should succeed");
        assert_eq!(io.outputs.get(&0).map(Vec::as_slice), Some(&b"hello"[..]));
    }

    /// Host IO from a spawned thread: guest code runs on whichever pool
    /// thread rayon's install lands it on, so workers see the run's inputs
    /// and their posted outputs must reach the caller.
    #[test]
    fn worker_threads_share_the_runs_io_surface() {
        let wasm = wat::parse_str(format!(
            r#"(module
                (import "env" "memory" (memory 1 4 shared))
                (import "wasi" "thread-spawn" (func $spawn (param i32) (result i32)))
                (import "host" "get_input_len" (func $len (param i32) (result i32)))
                (import "host" "get_input_data" (func $data (param i32 i32 i32)))
                (import "host" "post_output" (func $post (param i32 i32 i32)))
                (export "memory" (memory 0))
                {WAIT_FOR_COUNTER}
                (func (export "wasi_thread_start") (param $tid i32) (param $arg i32)
                    (local $n i32)
                    (local.set $n (call $len (i32.const 0)))
                    (call $data (i32.const 0) (i32.const 1024) (local.get $n))
                    (call $post (i32.const 1) (i32.const 1024) (local.get $n))
                    (drop (i32.atomic.rmw.add (i32.const 0) (i32.const 1))))
                (func (export "run")
                    (if (i32.le_s (call $spawn (i32.const 0)) (i32.const 0))
                        (then unreachable))
                    (call $wait_for (i32.const 1))))"#
        ))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let io = NativeOperatorExecutor::new(&wasm)
            .unwrap()
            .run_cancellable(OperatorIo::new(vec![b"shared".to_vec()]), &cancel)
            .expect("worker-IO run should succeed");
        assert_eq!(io.outputs.get(&1).map(Vec::as_slice), Some(&b"shared"[..]));
    }

    /// `host.cancelled` flips once the caller's cancel flag is seen; a
    /// guest polling it exits and the run reports Cancelled.
    #[test]
    fn cooperative_cancel_poll_reaches_the_guest() {
        // Spins until host.cancelled() returns nonzero, then returns
        // cleanly — no trap needed. Bounded by the same fuel scheme.
        let wasm = wat::parse_str(
            r#"(module
                (import "host" "cancelled" (func $cancelled (result i32)))
                (memory (export "memory") 1)
                (func (export "run")
                    (local $fuel i64)
                    (local.set $fuel (i64.const 0x200000000))
                    (loop $poll
                        (local.set $fuel (i64.sub (local.get $fuel) (i64.const 1)))
                        (if (i64.eqz (local.get $fuel)) (then unreachable))
                        (br_if $poll (i32.eqz (call $cancelled))))))"#,
        )
        .unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            flag.store(true, Ordering::Relaxed);
        });
        let result = NativeOperatorExecutor::new(&wasm)
            .unwrap()
            .run_cancellable(OperatorIo::new(vec![]), &cancel);
        assert!(
            matches!(result, Err(WasmBackendError::Cancelled)),
            "{result:?}"
        );
    }

    /// A packed blob (baseline + embedded threaded variant) must run the
    /// variant: the two modules post different bytes to output 0.
    #[test]
    fn packed_blob_runs_the_embedded_variant() {
        let post_const = |value: &str, threaded: bool| {
            let memory = if threaded {
                r#"(import "env" "memory" (memory 1 2 shared))
                   (export "memory" (memory 0))
                   (func (export "wasi_thread_start") (param i32 i32))"#
                    .to_string()
            } else {
                r#"(memory (export "memory") 1)"#.to_string()
            };
            wat::parse_str(format!(
                r#"(module
                    (import "host" "post_output" (func $post (param i32 i32 i32)))
                    {memory}
                    (data (i32.const 0) "{value}")
                    (func (export "run")
                        (call $post (i32.const 0) (i32.const 0) (i32.const 4))))"#
            ))
            .unwrap()
        };
        let packed = crate::wasm::variant::embed_threaded_variant(
            &post_const("base", false),
            &post_const("fast", true),
        )
        .unwrap();

        let cancel = AtomicBool::new(false);
        let io = NativeOperatorExecutor::new(&packed)
            .unwrap()
            .run_cancellable(OperatorIo::new(vec![]), &cancel)
            .expect("packed run should succeed");
        assert_eq!(io.outputs.get(&0).map(Vec::as_slice), Some(&b"fast"[..]));
    }

    /// Metadata retrieval must read back through a shared memory export.
    #[test]
    fn threaded_module_metadata_reads_shared_memory() {
        let wasm = wat::parse_str(
            r#"(module
                (import "env" "memory" (memory 1 2 shared))
                (export "memory" (memory 0))
                (data (i32.const 64) "meta")
                (func (export "run"))
                (func (export "wasi_thread_start") (param i32 i32))
                (func (export "get_metadata") (result i64)
                    ;; len 4 in the high half, ptr 64 in the low half.
                    (i64.or (i64.shl (i64.const 4) (i64.const 32)) (i64.const 64))))"#,
        )
        .unwrap();
        let metadata = NativeOperatorExecutor::new(&wasm)
            .unwrap()
            .get_metadata()
            .expect("metadata should read back");
        assert_eq!(metadata, b"meta");
    }
}
