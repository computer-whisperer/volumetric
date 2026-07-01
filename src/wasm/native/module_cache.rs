//! Content-keyed cache of compiled wasmtime modules.
//!
//! Compiling a module is by far the most expensive part of creating an
//! executor, and the interactive preview loop creates executors for the same
//! model bytes over and over. Each cache owns a single shared [`Engine`];
//! both `Engine` and `Module` are internally reference-counted, so clones
//! handed out here are cheap.
//!
//! The cache is bounded with FIFO eviction: interactive editing keeps
//! producing new merged-model blobs, so an unbounded cache would grow for the
//! lifetime of the process.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use wasmtime::{Config, Engine, Module};

use crate::wasm::error::WasmBackendError;

/// Maximum number of compiled modules kept per cache.
const CACHE_CAPACITY: usize = 32;

/// A bounded cache of compiled modules, keyed by the raw wasm bytes.
pub struct ModuleCache {
    engine: Engine,
    entries: Mutex<Entries>,
}

#[derive(Default)]
struct Entries {
    modules: HashMap<Arc<[u8]>, Module>,
    /// Insertion order for FIFO eviction.
    order: VecDeque<Arc<[u8]>>,
}

impl ModuleCache {
    fn new(engine: Engine) -> Self {
        Self {
            engine,
            entries: Mutex::new(Entries::default()),
        }
    }

    /// The engine all modules in this cache are compiled against.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Return the compiled module for `wasm_bytes`, compiling on a miss.
    pub fn get_or_compile(&self, wasm_bytes: &[u8]) -> Result<Module, WasmBackendError> {
        {
            let entries = self.entries.lock().unwrap();
            if let Some(module) = entries.modules.get(wasm_bytes) {
                return Ok(module.clone());
            }
        }

        // Compile outside the lock so concurrent misses on different modules
        // don't serialize. A racing compile of the same bytes is wasted work
        // but harmless.
        let module = Module::new(&self.engine, wasm_bytes)
            .map_err(|e| WasmBackendError::Instantiation(e.to_string()))?;

        let mut entries = self.entries.lock().unwrap();
        if !entries.modules.contains_key(wasm_bytes) {
            let key: Arc<[u8]> = wasm_bytes.into();
            entries.order.push_back(Arc::clone(&key));
            entries.modules.insert(key, module.clone());
            while entries.order.len() > CACHE_CAPACITY {
                if let Some(evicted) = entries.order.pop_front() {
                    entries.modules.remove(&evicted);
                }
            }
        }

        Ok(module)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.lock().unwrap().modules.len()
    }
}

/// Shared cache for model modules (default engine config).
pub fn model_cache() -> &'static ModuleCache {
    static CACHE: OnceLock<ModuleCache> = OnceLock::new();
    CACHE.get_or_init(|| ModuleCache::new(Engine::default()))
}

/// Shared cache for operator modules.
///
/// Operators keep `debug_info` enabled so failures reported through
/// `host.post_error` and traps come with usable backtraces.
pub fn operator_cache() -> &'static ModuleCache {
    static CACHE: OnceLock<ModuleCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        let engine = Engine::new(Config::new().debug_info(true))
            .expect("wasmtime engine with debug_info should construct");
        ModuleCache::new(engine)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest valid wasm module, with a custom section to vary content.
    fn tiny_module(tag: u8) -> Vec<u8> {
        // custom section: id 0, payload = [name_len=1, name=tag]
        let mut bytes = b"\0asm\x01\0\0\0".to_vec();
        bytes.extend_from_slice(&[0, 2, 1, tag]);
        bytes
    }

    #[test]
    fn caches_by_content_and_evicts_fifo() {
        let cache = ModuleCache::new(Engine::default());

        let a = tiny_module(b'a');
        cache.get_or_compile(&a).unwrap();
        cache.get_or_compile(&a).unwrap();
        assert_eq!(cache.len(), 1, "same bytes should share an entry");

        for i in 0..CACHE_CAPACITY {
            cache.get_or_compile(&tiny_module(b'0' + i as u8)).unwrap();
        }
        assert_eq!(cache.len(), CACHE_CAPACITY, "cache should be bounded");

        // 'a' was inserted first, so it should have been evicted.
        let entries = cache.entries.lock().unwrap();
        assert!(!entries.modules.contains_key(a.as_slice()));
    }

    #[test]
    fn invalid_wasm_is_an_error_and_not_cached() {
        let cache = ModuleCache::new(Engine::default());
        assert!(cache.get_or_compile(b"not wasm").is_err());
        assert_eq!(cache.len(), 0);
    }
}
