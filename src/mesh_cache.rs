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
use std::sync::{Arc, Mutex, OnceLock};

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
            }),
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
            entries: inner.entries.len(),
            bytes: inner.total_bytes,
            budget: inner.budget,
        }
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
}
