//! Content-addressed, byte-budgeted cache for generated ASN2 meshes.
//!
//! Meshes are expensive build artifacts: changing which output a viewport is
//! looking at must not determine their lifetime.  This cache sits below UI
//! scene assembly and is shared by local meshing callers; build daemons create
//! their own instance so repeated remote requests reuse the same geometry.
//! Entries are keyed by model content plus the complete geometry recipe and
//! bounded by an in-memory LRU budget.  Persistence is deliberately outside
//! this layer.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use crate::AdaptiveMeshV2Result;
use crate::adaptive_surface_nets_2::AdaptiveMeshConfig2;

/// Default generated-mesh budget: large enough for several serious desktop
/// previews, conservative enough for a browser tab.
#[cfg(target_arch = "wasm32")]
pub const DEFAULT_BUDGET_BYTES: usize = 256 << 20;
#[cfg(not(target_arch = "wasm32"))]
pub const DEFAULT_BUDGET_BYTES: usize = 2 << 30;

/// Bump when the mesher changes in a way that can change output for the same
/// model/configuration. This is part of every cache key.
const ASN2_ARTIFACT_VERSION: u32 = 1;

/// Content identity of one ASN2 mesh artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshCacheKey([u8; 32]);

impl MeshCacheKey {
    pub fn new(model_wasm: &[u8], config: &AdaptiveMeshConfig2) -> Self {
        // Thread count is an execution-resource choice, not part of the
        // generated geometry recipe. ASN2 is deterministic across worker
        // counts, so normalize it before serializing the remaining recipe.
        let mut recipe = config.clone();
        recipe.num_threads = 0;
        let mut recipe_bytes = Vec::new();
        ciborium::into_writer(&recipe, &mut recipe_bytes)
            .expect("AdaptiveMeshConfig2 always serializes to CBOR");

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"volumetric/asn2-mesh-artifact\0");
        hasher.update(&ASN2_ARTIFACT_VERSION.to_le_bytes());
        hasher.update(blake3::hash(model_wasm).as_bytes());
        hasher.update(&(recipe_bytes.len() as u64).to_le_bytes());
        hasher.update(&recipe_bytes);
        Self(*hasher.finalize().as_bytes())
    }
}

/// Point-in-time cache counters for hosts and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshCacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Cold callers that joined an identical build already in flight.
    pub coalesced: u64,
    pub entries: usize,
    pub bytes: usize,
    pub budget: usize,
}

struct Entry {
    mesh: Arc<AdaptiveMeshV2Result>,
    bytes: usize,
    last_used: u64,
}

struct Inner {
    entries: HashMap<MeshCacheKey, Entry>,
    total_bytes: usize,
    budget: usize,
    tick: u64,
    hits: u64,
    misses: u64,
    coalesced: u64,
}

#[derive(Default)]
struct FlightState {
    done: bool,
    /// Populated for every successful build, including artifacts too large
    /// for the configured LRU budget. This lets concurrent waiters share the
    /// just-built value even when it will not remain cached afterward.
    result: Option<Arc<AdaptiveMeshV2Result>>,
}

#[derive(Default)]
struct Flight {
    state: Mutex<FlightState>,
    finished: Condvar,
}

/// How a caller obtained a generated mesh artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshCacheSource {
    Hit,
    Shared,
    Built,
}

/// A generated mesh plus the cache path that delivered it.
pub struct MeshCacheArtifact {
    pub mesh: Arc<AdaptiveMeshV2Result>,
    pub source: MeshCacheSource,
}

impl Inner {
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

/// In-memory LRU of immutable generated meshes.
pub struct MeshCache {
    inner: Mutex<Inner>,
    flights: Mutex<HashMap<MeshCacheKey, Arc<Flight>>>,
}

impl MeshCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                total_bytes: 0,
                budget: budget_bytes,
                tick: 0,
                hits: 0,
                misses: 0,
                coalesced: 0,
            }),
            flights: Mutex::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &MeshCacheKey) -> Option<Arc<AdaptiveMeshV2Result>> {
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;
        match inner.entries.get_mut(key) {
            Some(entry) => {
                entry.last_used = tick;
                let mesh = Arc::clone(&entry.mesh);
                inner.hits += 1;
                Some(mesh)
            }
            None => {
                inner.misses += 1;
                None
            }
        }
    }

    /// Inserts a completed artifact and returns the shared cache value. An
    /// artifact larger than the entire budget is returned but not retained.
    /// If another worker inserted the same deterministic artifact first, its
    /// existing value wins and the duplicate is dropped.
    pub fn insert(
        &self,
        key: MeshCacheKey,
        mesh: AdaptiveMeshV2Result,
    ) -> Arc<AdaptiveMeshV2Result> {
        let mesh = Arc::new(mesh);
        let bytes = mesh_cost_bytes(&mesh);
        let mut inner = self.inner.lock().unwrap();
        inner.tick += 1;
        let tick = inner.tick;

        if let Some(existing) = inner.entries.get_mut(&key) {
            existing.last_used = tick;
            return Arc::clone(&existing.mesh);
        }
        if bytes > inner.budget {
            return mesh;
        }

        inner.entries.insert(
            key,
            Entry {
                mesh: Arc::clone(&mesh),
                bytes,
                last_used: tick,
            },
        );
        inner.total_bytes += bytes;
        inner.evict_to_budget();
        mesh
    }

    /// Returns an existing artifact or elects exactly one cold caller to run
    /// `build`; concurrent callers for the same key wait for and share that
    /// result. Waiting is cancellation-aware. A failed or cancelled leader
    /// wakes its followers without poisoning the key, allowing one of them to
    /// retry with its own build closure.
    pub fn get_or_build<E, F>(
        &self,
        key: MeshCacheKey,
        cancel: &AtomicBool,
        build: F,
    ) -> Result<Option<MeshCacheArtifact>, E>
    where
        F: FnOnce() -> Result<Option<AdaptiveMeshV2Result>, E>,
    {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        if let Some(mesh) = self.get(&key) {
            return Ok(Some(MeshCacheArtifact {
                mesh,
                source: MeshCacheSource::Hit,
            }));
        }

        let mut build = Some(build);
        loop {
            let (flight, leader) = {
                let mut flights = self.flights.lock().unwrap();
                match flights.get(&key) {
                    Some(flight) => (Arc::clone(flight), false),
                    None => {
                        let flight = Arc::new(Flight::default());
                        flights.insert(key, Arc::clone(&flight));
                        (flight, true)
                    }
                }
            };

            if leader {
                let mut guard = FlightGuard {
                    cache: self,
                    key,
                    flight,
                };
                let Some(mesh) = build.take().expect("one caller becomes leader")()? else {
                    return Ok(None);
                };
                let mesh = self.insert(key, mesh);
                guard.publish(Arc::clone(&mesh));
                return Ok(Some(MeshCacheArtifact {
                    mesh,
                    source: MeshCacheSource::Built,
                }));
            }

            self.inner.lock().unwrap().coalesced += 1;
            let mut state = flight.state.lock().unwrap();
            while !state.done {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(None);
                }
                let (next, _) = flight
                    .finished
                    .wait_timeout(state, Duration::from_millis(50))
                    .expect("mesh flight mutex poisoned");
                state = next;
            }
            if let Some(mesh) = &state.result {
                return Ok(Some(MeshCacheArtifact {
                    mesh: Arc::clone(mesh),
                    source: MeshCacheSource::Shared,
                }));
            }
            drop(state);
            if cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
            // The leader failed or cancelled. Its guard removed the flight,
            // so loop and let one remaining caller become the retry leader.
        }
    }

    pub fn set_budget(&self, budget_bytes: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.budget = budget_bytes;
        inner.evict_to_budget();
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.clear();
        inner.total_bytes = 0;
    }

    pub fn stats(&self) -> MeshCacheStats {
        let inner = self.inner.lock().unwrap();
        MeshCacheStats {
            hits: inner.hits,
            misses: inner.misses,
            coalesced: inner.coalesced,
            entries: inner.entries.len(),
            bytes: inner.total_bytes,
            budget: inner.budget,
        }
    }
}

struct FlightGuard<'a> {
    cache: &'a MeshCache,
    key: MeshCacheKey,
    flight: Arc<Flight>,
}

impl FlightGuard<'_> {
    fn publish(&mut self, mesh: Arc<AdaptiveMeshV2Result>) {
        self.flight.state.lock().unwrap().result = Some(mesh);
    }
}

impl Drop for FlightGuard<'_> {
    fn drop(&mut self) {
        {
            let mut flights = self.cache.flights.lock().unwrap();
            if flights
                .get(&self.key)
                .is_some_and(|flight| Arc::ptr_eq(flight, &self.flight))
            {
                flights.remove(&self.key);
            }
        }
        let mut state = self.flight.state.lock().unwrap();
        state.done = true;
        drop(state);
        self.flight.finished.notify_all();
    }
}

fn mesh_cost_bytes(mesh: &AdaptiveMeshV2Result) -> usize {
    const ENTRY_OVERHEAD: usize = 512;
    ENTRY_OVERHEAD
        + mesh.vertices.len() * std::mem::size_of::<(f32, f32, f32)>()
        + mesh.normals.len() * std::mem::size_of::<(f32, f32, f32)>()
        + mesh.indices.len() * std::mem::size_of::<u32>()
}

/// Process-wide cache used by in-process callers such as the native UI.
pub fn global() -> &'static MeshCache {
    static GLOBAL: OnceLock<MeshCache> = OnceLock::new();
    GLOBAL.get_or_init(|| MeshCache::new(DEFAULT_BUDGET_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn mesh(vertex_count: usize) -> AdaptiveMeshV2Result {
        AdaptiveMeshV2Result {
            vertices: vec![(0.0, 0.0, 0.0); vertex_count],
            normals: vec![(0.0, 0.0, 1.0); vertex_count],
            indices: vec![0, 0, 0],
            bounds_min: (0.0, 0.0, 0.0),
            bounds_max: (1.0, 1.0, 1.0),
            stats: Default::default(),
        }
    }

    #[test]
    fn key_tracks_content_and_geometry_recipe_but_not_thread_count() {
        let config = AdaptiveMeshConfig2::default();
        let key = MeshCacheKey::new(b"model-a", &config);

        let mut threads = config.clone();
        threads.num_threads = 3;
        assert_eq!(key, MeshCacheKey::new(b"model-a", &threads));

        let mut resolution = config.clone();
        resolution.max_depth += 1;
        assert_ne!(key, MeshCacheKey::new(b"model-a", &resolution));
        assert_ne!(key, MeshCacheKey::new(b"model-b", &config));
    }

    #[test]
    fn caches_by_content_and_evicts_to_budget() {
        let config = AdaptiveMeshConfig2::default();
        let a = MeshCacheKey::new(b"a", &config);
        let b = MeshCacheKey::new(b"b", &config);
        let one_cost = mesh_cost_bytes(&mesh(4));
        let cache = MeshCache::new(one_cost);

        let first = cache.insert(a, mesh(4));
        assert!(Arc::ptr_eq(&first, &cache.get(&a).unwrap()));
        cache.insert(b, mesh(4));
        assert!(cache.get(&a).is_none(), "least-recent artifact evicted");
        assert!(cache.get(&b).is_some());
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn oversized_mesh_is_delivered_but_not_retained() {
        let config = AdaptiveMeshConfig2::default();
        let key = MeshCacheKey::new(b"large", &config);
        let cache = MeshCache::new(1);
        let result = cache.insert(key, mesh(4));
        assert_eq!(result.vertices.len(), 4);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn simultaneous_cold_callers_share_one_build() {
        let cache = Arc::new(MeshCache::new(1 << 20));
        let key = MeshCacheKey::new(b"single-flight", &AdaptiveMeshConfig2::default());
        let builds = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let leader_cache = Arc::clone(&cache);
        let leader_builds = Arc::clone(&builds);
        let leader = std::thread::spawn(move || {
            leader_cache
                .get_or_build(key, &AtomicBool::new(false), || {
                    leader_builds.fetch_add(1, Ordering::Relaxed);
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok::<_, String>(Some(mesh(8)))
                })
                .unwrap()
                .unwrap()
        });
        started_rx.recv().unwrap();

        let follower_cache = Arc::clone(&cache);
        let follower_builds = Arc::clone(&builds);
        let follower = std::thread::spawn(move || {
            follower_cache
                .get_or_build(key, &AtomicBool::new(false), || {
                    follower_builds.fetch_add(1, Ordering::Relaxed);
                    Ok::<_, String>(Some(mesh(8)))
                })
                .unwrap()
                .unwrap()
        });

        while cache.stats().coalesced == 0 {
            std::thread::yield_now();
        }
        release_tx.send(()).unwrap();
        let leader = leader.join().unwrap();
        let follower = follower.join().unwrap();

        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert_eq!(leader.source, MeshCacheSource::Built);
        assert_eq!(follower.source, MeshCacheSource::Shared);
        assert!(Arc::ptr_eq(&leader.mesh, &follower.mesh));
    }

    #[test]
    fn waiting_caller_can_cancel_without_cancelling_leader() {
        let cache = Arc::new(MeshCache::new(1 << 20));
        let key = MeshCacheKey::new(b"cancel-wait", &AdaptiveMeshConfig2::default());
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let leader_cache = Arc::clone(&cache);
        let leader = std::thread::spawn(move || {
            leader_cache
                .get_or_build(key, &AtomicBool::new(false), || {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok::<_, String>(Some(mesh(8)))
                })
                .unwrap()
                .unwrap()
        });
        started_rx.recv().unwrap();

        let follower_cache = Arc::clone(&cache);
        let follower_cancel = Arc::new(AtomicBool::new(false));
        let cancel = Arc::clone(&follower_cancel);
        let follower = std::thread::spawn(move || {
            follower_cache
                .get_or_build(
                    key,
                    &cancel,
                    || -> Result<Option<AdaptiveMeshV2Result>, String> {
                        panic!("a follower must not run while the leader is active")
                    },
                )
                .unwrap()
        });
        while cache.stats().coalesced == 0 {
            std::thread::yield_now();
        }
        follower_cancel.store(true, Ordering::Relaxed);
        assert!(follower.join().unwrap().is_none());

        release_tx.send(()).unwrap();
        assert_eq!(leader.join().unwrap().source, MeshCacheSource::Built);
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn failed_leader_wakes_a_follower_to_retry() {
        let cache = Arc::new(MeshCache::new(1 << 20));
        let key = MeshCacheKey::new(b"retry", &AdaptiveMeshConfig2::default());
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let leader_cache = Arc::clone(&cache);
        let leader = std::thread::spawn(move || {
            leader_cache.get_or_build(key, &AtomicBool::new(false), || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Err::<Option<AdaptiveMeshV2Result>, _>("leader failed")
            })
        });
        started_rx.recv().unwrap();

        let follower_cache = Arc::clone(&cache);
        let follower = std::thread::spawn(move || {
            follower_cache
                .get_or_build(key, &AtomicBool::new(false), || {
                    Ok::<_, &str>(Some(mesh(8)))
                })
                .unwrap()
                .unwrap()
        });
        while cache.stats().coalesced == 0 {
            std::thread::yield_now();
        }
        release_tx.send(()).unwrap();

        assert!(matches!(leader.join().unwrap(), Err("leader failed")));
        assert_eq!(follower.join().unwrap().source, MeshCacheSource::Built);
        assert!(cache.get(&key).is_some());
    }
}
