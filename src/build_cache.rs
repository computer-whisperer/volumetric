//! Content-addressed memoization of timeline step results.
//!
//! A timeline step is a pure function of bytes: the operator's wasm module
//! and every resolved input blob (the host import surface — `get_input_*`,
//! `post_*`, `input_model_*` — exposes no clock or randomness). That makes
//! step outputs safe to memoize by content hash: the key chains through
//! intermediates, so editing one step's config re-runs exactly that step
//! and its downstream while everything upstream (and on independent
//! branches) is served from cache. A rebuilt operator module is a new hash,
//! so staleness is impossible by construction.
//!
//! Entries hold the produced output blobs behind `Arc`s shared with the
//! `Environment`, so a hit costs no copies — and repeated runs hand back
//! pointer-identical blobs, which downstream identity-keyed caches (the
//! UI's preview mesh cache) piggyback on.
//!
//! The cache is in-memory only and bounded by a byte budget with LRU
//! eviction. One process-global instance ([`global`]) serves every run in
//! the process — the native UI's worker, all daemon jobs (content
//! addressing makes cross-client sharing safe), and the browser's inline
//! executor — mirroring the global compiled-module caches in
//! `wasm::native::module_cache`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::AssetTypeHint;

/// Cache key: hash of the operator module's content hash followed by each
/// resolved input's content hash, in slot order. Inline and asset-ref
/// inputs with identical bytes hash identically, as they should.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StepKey([u8; 32]);

impl StepKey {
    /// Builds the key for one step from the operator's and each input's
    /// content hash (32-byte blocks, so the encoding is unambiguous
    /// without length framing).
    pub fn new(operator_hash: &[u8; 32], input_hashes: &[[u8; 32]]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(operator_hash);
        for hash in input_hashes {
            hasher.update(hash);
        }
        Self(*hasher.finalize().as_bytes())
    }
}

/// The memoized result of one executed step: every output the operator
/// produced (by output slot, with its content hash so consumers never
/// re-hash cached blobs) plus the operator's declared output types.
/// Stored per (operator, inputs) — independent of any step's output-id
/// mapping, which the hit path applies per project.
pub struct CachedStep {
    /// Produced outputs by slot index. Sparse: operators may skip slots.
    pub outputs: HashMap<usize, (Arc<Vec<u8>>, [u8; 32])>,
    /// Output type hints declared by the operator's metadata (may be
    /// shorter than the produced slots; consumers fall back to `Model`).
    pub declared_outputs: Vec<AssetTypeHint>,
}

impl CachedStep {
    fn cost_bytes(&self) -> usize {
        // Fixed overhead keeps a flood of tiny entries from escaping the
        // budget on payload bytes alone.
        const ENTRY_OVERHEAD: usize = 256;
        const OUTPUT_OVERHEAD: usize = 64;
        ENTRY_OVERHEAD
            + self
                .outputs
                .values()
                .map(|(data, _)| data.len() + OUTPUT_OVERHEAD)
                .sum::<usize>()
    }
}

/// Point-in-time counters for observability and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub bytes: usize,
    pub budget: usize,
}

struct Entry {
    step: Arc<CachedStep>,
    bytes: usize,
    last_used: u64,
}

struct Inner {
    entries: HashMap<StepKey, Entry>,
    total_bytes: usize,
    budget: usize,
    /// Monotonic access clock for LRU ordering.
    tick: u64,
    hits: u64,
    misses: u64,
}

impl Inner {
    /// Evicts least-recently-used entries until the total fits the budget.
    /// Linear scans are fine here: entries are step-sized (typically MBs),
    /// so the map stays in the hundreds at most.
    fn evict_to_budget(&mut self) {
        while self.total_bytes > self.budget {
            let Some((&key, _)) = self.entries.iter().min_by_key(|(_, entry)| entry.last_used)
            else {
                break;
            };
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes -= entry.bytes;
            }
        }
    }
}

/// Byte-budgeted LRU memo of step results. Interior-mutable and shareable;
/// the mutex guards bookkeeping only (blob bytes live behind `Arc`s), so
/// contention is negligible next to step execution.
pub struct BuildCache {
    inner: Mutex<Inner>,
}

impl BuildCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                total_bytes: 0,
                budget: budget_bytes,
                tick: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// Looks up a step result, refreshing its LRU position on a hit.
    pub fn get(&self, key: &StepKey) -> Option<Arc<CachedStep>> {
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;
        match inner.entries.get_mut(key) {
            Some(entry) => {
                entry.last_used = tick;
                let step = Arc::clone(&entry.step);
                inner.hits += 1;
                Some(step)
            }
            None => {
                inner.misses += 1;
                None
            }
        }
    }

    /// Records a step result. Entries larger than the whole budget are not
    /// stored (a zero-budget cache thus stores nothing). Racing inserts of
    /// the same key overwrite each other — identical content, harmless.
    pub fn insert(&self, key: StepKey, step: CachedStep) {
        let bytes = step.cost_bytes();
        let mut inner = self.inner.lock().unwrap();
        if bytes > inner.budget {
            return;
        }
        inner.tick += 1;
        let entry = Entry {
            step: Arc::new(step),
            bytes,
            last_used: inner.tick,
        };
        if let Some(old) = inner.entries.insert(key, entry) {
            inner.total_bytes -= old.bytes;
        }
        inner.total_bytes += bytes;
        inner.evict_to_budget();
    }

    /// Adjusts the byte budget, evicting immediately if lowered.
    pub fn set_budget(&self, budget_bytes: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.budget = budget_bytes;
        inner.evict_to_budget();
    }

    /// Drops every entry (counters are kept).
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.clear();
        inner.total_bytes = 0;
    }

    pub fn stats(&self) -> BuildCacheStats {
        let inner = self.inner.lock().unwrap();
        BuildCacheStats {
            hits: inner.hits,
            misses: inner.misses,
            entries: inner.entries.len(),
            bytes: inner.total_bytes,
            budget: inner.budget,
        }
    }
}

/// Default budget: enough to keep a few heavy FEA chains' intermediates
/// resident on desktop; conservative inside a browser tab.
#[cfg(target_arch = "wasm32")]
pub const DEFAULT_BUDGET_BYTES: usize = 256 << 20;
#[cfg(not(target_arch = "wasm32"))]
pub const DEFAULT_BUDGET_BYTES: usize = 2 << 30;

/// The process-wide cache used by `Project::run*` unless a caller supplies
/// its own via `run_monitored_with_cache`.
pub fn global() -> &'static BuildCache {
    static GLOBAL: OnceLock<BuildCache> = OnceLock::new();
    GLOBAL.get_or_init(|| BuildCache::new(DEFAULT_BUDGET_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step_with_bytes(len: usize) -> CachedStep {
        CachedStep {
            outputs: HashMap::from([(0, (Arc::new(vec![0u8; len]), [0u8; 32]))]),
            declared_outputs: Vec::new(),
        }
    }

    fn key(n: u8) -> StepKey {
        StepKey::new(&[n; 32], &[])
    }

    #[test]
    fn hit_returns_shared_arc_and_counts() {
        let cache = BuildCache::new(1 << 20);
        assert!(cache.get(&key(1)).is_none());
        cache.insert(key(1), step_with_bytes(100));
        let a = cache.get(&key(1)).expect("hit");
        let b = cache.get(&key(1)).expect("hit");
        assert!(Arc::ptr_eq(&a.outputs[&0].0, &b.outputs[&0].0));
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (2, 1, 1));
    }

    #[test]
    fn key_depends_on_operator_and_input_order() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        assert_eq!(StepKey::new(&a, &[b]), StepKey::new(&a, &[b]));
        assert_ne!(StepKey::new(&a, &[b]), StepKey::new(&b, &[a]));
        assert_ne!(StepKey::new(&a, &[a, b]), StepKey::new(&a, &[b, a]));
        assert_ne!(StepKey::new(&a, &[]), StepKey::new(&a, &[a]));
    }

    #[test]
    fn evicts_least_recently_used_first() {
        // Budget fits two payloads (plus overheads), not three.
        let cache = BuildCache::new(2 * (1000 + 64 + 256) + 100);
        cache.insert(key(1), step_with_bytes(1000));
        cache.insert(key(2), step_with_bytes(1000));
        cache.get(&key(1)); // refresh 1 so 2 is now LRU
        cache.insert(key(3), step_with_bytes(1000));
        assert!(cache.get(&key(1)).is_some());
        assert!(cache.get(&key(2)).is_none(), "LRU entry evicted");
        assert!(cache.get(&key(3)).is_some());
    }

    #[test]
    fn oversized_entries_and_zero_budget_store_nothing() {
        let cache = BuildCache::new(500);
        cache.insert(key(1), step_with_bytes(1000));
        assert!(cache.get(&key(1)).is_none());

        let disabled = BuildCache::new(0);
        disabled.insert(key(2), step_with_bytes(1));
        assert!(disabled.get(&key(2)).is_none());
        assert_eq!(disabled.stats().bytes, 0);
    }

    #[test]
    fn lowering_budget_evicts() {
        let cache = BuildCache::new(1 << 20);
        cache.insert(key(1), step_with_bytes(1000));
        cache.insert(key(2), step_with_bytes(1000));
        cache.set_budget(1);
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.bytes), (0, 0));
    }
}
