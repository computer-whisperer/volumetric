//! The module catalog: what the Add UI lists, fed by each module's own
//! declared metadata (`get_metadata()`) rather than a static side table.
//!
//! Reading metadata means compiling the module's wasm — far too slow for a
//! menu (seconds per module on a cold debug cache) — so the catalog warms
//! in the background: entries start as name-only stubs enumerated from the
//! bundled registry, a persisted content-hash-keyed cache instantly fills
//! every module seen before, and the session scans the rest one at a time
//! through idle-priority background jobs, persisting each result as it
//! lands. Keying by content hash makes staleness impossible by
//! construction: a rebuilt module is a new hash, which is a fresh scan.
//!
//! Persistence is native-only (same gap as settings — the web shell
//! rescans per session). The bundled registry is only a byte source here;
//! runtime-loaded modules join the same enumerate → hash → cache/scan path
//! in a later chunk.

use sha2::{Digest, Sha256};
use volumetric::OperatorMetadata;
use volumetric_assets::AssetCategory;

/// A module's declared metadata, as far as the catalog currently knows.
#[derive(Clone, Debug, PartialEq)]
pub enum CatalogMetadata {
    /// Not yet read: no cache hit, scan still pending or in flight.
    Pending,
    /// Declared metadata, from the cache or a completed read.
    Ready(OperatorMetadata),
    /// The read failed; kept so the scan queue doesn't retry forever.
    /// (A rebuilt module re-enters as a new hash and scans fresh.)
    Failed(String),
}

/// One Add-catalog entry: a module known by name, identified by content
/// hash, displayed from its own declared metadata once known.
#[derive(Clone, Debug)]
pub struct CatalogEntry {
    /// The crate/module name — the stable id add routes address.
    pub name: String,
    /// Bundled collection the bytes came from (model vs operator).
    pub kind: AssetCategory,
    /// Hex SHA-256 of the module bytes — the metadata cache key.
    pub hash: String,
    /// Declared metadata state.
    pub metadata: CatalogMetadata,
}

impl CatalogEntry {
    /// The name the UI shows: declared `display_name`, falling back to the
    /// module name until metadata is known (transient on first launch).
    pub fn display_name(&self) -> &str {
        match &self.metadata {
            CatalogMetadata::Ready(metadata) => metadata.catalog_name(),
            _ => &self.name,
        }
    }

    /// Declared metadata, if known.
    pub fn ready(&self) -> Option<&OperatorMetadata> {
        match &self.metadata {
            CatalogMetadata::Ready(metadata) => Some(metadata),
            _ => None,
        }
    }
}

/// The Add catalog. Owned by the app; the session drains
/// [`Catalog::take_scan_request`] into background jobs and routes results
/// back through [`Catalog::on_metadata`].
#[derive(Debug)]
pub struct Catalog {
    entries: Vec<CatalogEntry>,
    /// Where the metadata cache persists; `None` (tests, unattached hosts)
    /// scans every session and never touches disk.
    #[cfg(not(target_arch = "wasm32"))]
    cache_path: Option<std::path::PathBuf>,
    /// Cached metadata for hashes beyond the current entries (other builds,
    /// removed modules) — carried through saves so parallel workspaces
    /// don't evict each other.
    #[cfg(not(target_arch = "wasm32"))]
    foreign_cache: std::collections::HashMap<String, OperatorMetadata>,
    /// Entry name a warm scan is currently reading, if any.
    scan_inflight: Option<String>,
}

impl Default for Catalog {
    /// A catalog over the bundled registry with no persistence attached.
    fn default() -> Self {
        let entries = volumetric_assets::models()
            .iter()
            .chain(volumetric_assets::operators())
            .map(|asset| CatalogEntry {
                name: asset.name.to_string(),
                kind: asset.category,
                hash: hex_sha256(asset.bytes),
                metadata: CatalogMetadata::Pending,
            })
            .collect();
        Self {
            entries,
            #[cfg(not(target_arch = "wasm32"))]
            cache_path: None,
            #[cfg(not(target_arch = "wasm32"))]
            foreign_cache: std::collections::HashMap::new(),
            scan_inflight: None,
        }
    }
}

impl Catalog {
    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    pub fn get(&self, name: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// Display name for a module by name; the raw name when unknown.
    pub fn display_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.get(name)
            .map(CatalogEntry::display_name)
            .unwrap_or(name)
    }

    /// The next module a warm scan should read: the first pending entry,
    /// one at a time (scans are idle-priority; serial keeps them out of
    /// the way of real work). Returns `None` while one is in flight.
    pub fn take_scan_request(&mut self) -> Option<String> {
        if self.scan_inflight.is_some() {
            return None;
        }
        let name = self
            .entries
            .iter()
            .find(|entry| entry.metadata == CatalogMetadata::Pending)
            .map(|entry| entry.name.clone())?;
        self.scan_inflight = Some(name.clone());
        Some(name)
    }

    /// A metadata read finished (warm scan or add-click — any read warms
    /// the catalog). Fills the entry and persists successes to the cache.
    pub fn on_metadata(&mut self, name: &str, result: &Result<OperatorMetadata, String>) {
        if self.scan_inflight.as_deref() == Some(name) {
            self.scan_inflight = None;
        }
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.name == name) else {
            return;
        };
        match result {
            Ok(metadata) => {
                entry.metadata = CatalogMetadata::Ready(metadata.clone());
                self.persist();
            }
            // Don't clobber good (cached) metadata with a transient failure.
            Err(err) => {
                if entry.ready().is_none() {
                    entry.metadata = CatalogMetadata::Failed(err.clone());
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn persist(&self) {}
}

/// The persisted metadata cache (native shells only; the web shell rescans
/// per session, the same open gap as settings persistence).
#[cfg(not(target_arch = "wasm32"))]
mod persistence {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use volumetric::OperatorMetadata;

    use super::{Catalog, CatalogMetadata};

    /// Persisted shape: content hash → declared metadata.
    #[derive(serde::Deserialize, serde::Serialize)]
    struct CacheFile {
        version: u32,
        modules: HashMap<String, OperatorMetadata>,
    }

    /// The cache format version; bump to invalidate wholesale on layout
    /// change.
    const CACHE_VERSION: u32 = 1;

    impl Catalog {
        /// `<cache dir>/volumetric/module-metadata.json`; `None` when the
        /// platform has no cache directory (then every session rescans).
        pub fn default_cache_path() -> Option<PathBuf> {
            dirs::cache_dir().map(|dir| dir.join("volumetric").join("module-metadata.json"))
        }

        /// Attach the persisted cache: load it and fill every entry whose
        /// content hash it covers. Hosts call this once at startup; tests
        /// skip it and stay off the real cache file.
        pub fn attach_cache(&mut self, path: PathBuf) {
            let mut cached = load_cache(&path);
            self.cache_path = Some(path);
            for entry in &mut self.entries {
                if let Some(metadata) = cached.remove(&entry.hash) {
                    entry.metadata = CatalogMetadata::Ready(metadata);
                }
            }
            self.foreign_cache = cached;
        }

        /// Write the cache file: every Ready entry plus the foreign hashes
        /// the load carried. Same tmp + rename discipline as settings — a
        /// cache write must never take the UI down, so failures just log.
        pub(super) fn persist(&self) {
            let Some(path) = self.cache_path.as_deref() else {
                return;
            };
            let mut modules = self.foreign_cache.clone();
            for entry in &self.entries {
                if let Some(metadata) = entry.ready() {
                    modules.insert(entry.hash.clone(), metadata.clone());
                }
            }
            let file = CacheFile {
                version: CACHE_VERSION,
                modules,
            };
            let json = match serde_json::to_vec(&file) {
                Ok(json) => json,
                Err(_) => return,
            };
            let result = (|| {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let tmp = path.with_extension("json.tmp");
                std::fs::write(&tmp, &json)?;
                std::fs::rename(&tmp, path)
            })();
            if let Err(err) = result {
                log::warn!(
                    "failed to persist module metadata cache to {}: {err}",
                    path.display()
                );
            }
        }
    }

    /// Read a cache file; missing, malformed, or version-mismatched files
    /// are an empty cache (the catalog rescans — never an error).
    fn load_cache(path: &Path) -> HashMap<String, OperatorMetadata> {
        let Ok(bytes) = std::fs::read(path) else {
            return HashMap::new();
        };
        match serde_json::from_slice::<CacheFile>(&bytes) {
            Ok(file) if file.version == CACHE_VERSION => file.modules,
            Ok(file) => {
                log::info!(
                    "module metadata cache at {} is version {} (want {CACHE_VERSION}); rescanning",
                    path.display(),
                    file.version
                );
                HashMap::new()
            }
            Err(err) => {
                log::warn!(
                    "ignoring malformed module metadata cache at {}: {err}",
                    path.display()
                );
                HashMap::new()
            }
        }
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready_metadata(name: &str) -> OperatorMetadata {
        OperatorMetadata {
            name: name.to_string(),
            version: "0.0.0".to_string(),
            display_name: "Pretty Name".to_string(),
            description: "A test module.".to_string(),
            category: "Testing".to_string(),
            icon_svg: String::new(),
            inputs: vec![],
            input_names: vec![],
            outputs: vec![],
        }
    }

    /// Bundled entries stub in as Pending, and scan requests hand them out
    /// one at a time.
    #[test]
    fn scan_requests_are_serial_and_cover_pending_entries() {
        let mut catalog = Catalog::default();
        if catalog.entries().is_empty() {
            return; // wasm bundle absent in this build; nothing to check
        }
        let first = catalog.take_scan_request().expect("pending entries");
        // In flight: no second request until the first resolves.
        assert_eq!(catalog.take_scan_request(), None);

        catalog.on_metadata(&first, &Ok(ready_metadata(&first)));
        assert_eq!(catalog.display_name(&first), "Pretty Name");

        // Resolution frees the slot for the next pending entry.
        if catalog
            .entries()
            .iter()
            .any(|entry| entry.metadata == CatalogMetadata::Pending)
        {
            let second = catalog.take_scan_request().expect("more pending");
            assert_ne!(second, first);
        }
    }

    /// A failed read parks the entry as Failed (no retry loop) but never
    /// clobbers previously cached metadata.
    #[test]
    fn failures_park_but_do_not_clobber() {
        let mut catalog = Catalog::default();
        let Some(name) = catalog.entries().first().map(|e| e.name.clone()) else {
            return;
        };
        catalog.on_metadata(&name, &Err("boom".to_string()));
        assert!(matches!(
            catalog.get(&name).unwrap().metadata,
            CatalogMetadata::Failed(_)
        ));

        catalog.on_metadata(&name, &Ok(ready_metadata(&name)));
        catalog.on_metadata(&name, &Err("late failure".to_string()));
        assert_eq!(catalog.display_name(&name), "Pretty Name");
    }

    /// Ready metadata persists through the cache file and pre-fills a new
    /// catalog over the same bytes; foreign hashes survive the rewrite.
    #[test]
    fn cache_round_trips_by_content_hash() {
        let dir =
            std::env::temp_dir().join(format!("volumetric-catalog-test-{}", std::process::id()));
        let path = dir.join("module-metadata.json");
        let _ = std::fs::remove_file(&path);

        let mut catalog = Catalog::default();
        let Some(name) = catalog.entries().first().map(|e| e.name.clone()) else {
            return;
        };
        catalog.attach_cache(path.clone());
        catalog.foreign_cache.insert(
            "not-a-current-module".to_string(),
            ready_metadata("foreign"),
        );
        catalog.on_metadata(&name, &Ok(ready_metadata(&name)));

        let mut reloaded = Catalog::default();
        reloaded.attach_cache(path.clone());
        assert_eq!(reloaded.display_name(&name), "Pretty Name");
        assert!(reloaded.foreign_cache.contains_key("not-a-current-module"));
        // The cache hit means nothing to scan for that entry.
        assert_ne!(reloaded.take_scan_request(), Some(name));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
