//! Scoped thread-pool entry point for threaded operator builds.
//!
//! Threaded operators (wasm32-wasip1-threads variants) run their whole
//! body inside [`with_thread_pool`]: the host's IO imports work from any
//! thread, so guest code doesn't care which pool member it lands on. The
//! pool is sized by the `VOLUMETRIC_THREADS` environ hint the host
//! provides (inside wasm, `available_parallelism` has nothing to measure)
//! and torn down before returning — spawned wasi threads must not outlive
//! the operator run, or the host detaches and leaks them.
//!
//! Without the `threaded` feature this is a plain call, so operator code
//! is identical in both builds.

/// Run `f` on a host-sized scoped thread pool (`threaded` builds) or
/// directly (plain builds).
#[cfg(feature = "threaded")]
pub fn with_thread_pool<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    let threads = std::env::var("VOLUMETRIC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);
    if threads <= 1 {
        return f();
    }
    // NOT use_current_thread(): on a local pool it leaks the registry —
    // the workers never join, and the host would detach the wasi threads
    // at teardown. `install` runs `f` on a worker instead (host IO works
    // from any thread) while this thread parks until `f` completes; the
    // pool then drops and every worker exits before run() returns.
    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

/// Run `f` on a host-sized scoped thread pool (`threaded` builds) or
/// directly (plain builds).
#[cfg(not(feature = "threaded"))]
pub fn with_thread_pool<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    f()
}
