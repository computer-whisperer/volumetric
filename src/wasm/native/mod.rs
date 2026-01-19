//! Native WASM backend using wasmtime.
//!
//! This module provides the native implementation of the WASM execution traits
//! using the wasmtime runtime. It's the default backend for desktop applications.

mod model_executor;
mod operator_executor;
mod parallel_sampler;

pub use model_executor::{NativeModelExecutor, NativeModelExecutorNd};
pub use operator_executor::NativeOperatorExecutor;
pub use parallel_sampler::{NativeParallelSampler, NativeParallelSamplerNd};
