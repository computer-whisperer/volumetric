//! Persisting built step results inside a project file ("baked" projects).
//!
//! A baked project is an ordinary `.vproj` carrying an optional snapshot of
//! its step-cache entries ([`Project::baked`]): the content-addressed keys
//! of its timeline steps together with the output blobs they produced. On
//! open, [`Project::seed_build_cache`] moves the snapshot into the process
//! [`BuildCache`](crate::BuildCache), so the next build serves every baked
//! step from cache — a heavy FEA project opens without re-running a single
//! solve.
//!
//! Because step keys chain content hashes through every intermediate (see
//! the [`build_cache`](crate::build_cache) module docs), a baked entry can
//! never be *wrong*, only *unused*: editing a step diverges the key chain
//! from that step onward, and the stale entries are simply never hit while
//! everything upstream still is. Baked files therefore stay fully editable
//! projects — tools that load, mutate, and re-save them preserve a bake
//! that remains correct with no invalidation logic.
//!
//! The snapshot is collected without executing anything
//! ([`Project::collect_baked`] replays the key chain against the cache),
//! and it only asserts what the cache itself asserted: blob hashes are
//! re-verified on seed, and a corrupted blob just drops the steps that
//! reference it back to re-execution. Trust-wise a bake is equivalent to
//! the file's author having run the build locally — fine for opening your
//! own files, which is why the remote daemon never seeds its shared cache
//! from client-supplied bakes.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::build_cache::{BuildCache, CachedStep, StepKey};
use crate::{AssetTypeHint, ExecutionInput, Project};

/// One output blob, stored once regardless of how many steps produced it.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct BakedBlob {
    /// blake3 of `data`; re-verified on seed.
    pub hash: [u8; 32],
    /// The raw blob bytes (byte-string encoded: blobs run to hundreds of
    /// MB and CBOR per-element arrays would roughly double them).
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

/// One memoized step: its cache key and every output slot it produced,
/// referencing blobs by hash in the shared [`BakedResults::blobs`] table.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct BakedStep {
    /// The [`StepKey`] bytes (blake3 chain of operator and input hashes).
    pub key: [u8; 32],
    /// Produced output slots as `(slot index, blob hash)`, sorted by slot.
    /// All produced slots are kept, not just the ones this project maps to
    /// ids — the memo is a pure function of (operator, inputs).
    pub outputs: Vec<(u32, [u8; 32])>,
    /// Output type hints declared by the operator's metadata.
    pub declared_outputs: Vec<AssetTypeHint>,
}

/// The optional step-result snapshot embedded in a `.vproj`.
///
/// Serialization is deterministic for a given project and cache state:
/// steps in timeline order, blobs sorted by hash.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct BakedResults {
    /// Content-addressed blob store, deduplicated by hash.
    pub blobs: Vec<BakedBlob>,
    /// Baked timeline steps (steps missing from the cache at collection
    /// time are absent; they re-run on open).
    pub steps: Vec<BakedStep>,
}

impl BakedResults {
    /// Total payload bytes across all blobs (the dominant file-size term).
    pub fn blob_bytes(&self) -> usize {
        self.blobs.iter().map(|blob| blob.data.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// How much of a project's timeline [`Project::collect_baked`] covered.
///
/// Coverage is prefix-closed per dependency chain: a step missing from the
/// cache makes its outputs' hashes unknowable, so downstream steps can't be
/// keyed (or baked) either.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BakeCoverage {
    /// Steps whose results made it into the snapshot.
    pub baked_steps: usize,
    /// Steps in the project timeline.
    pub total_steps: usize,
}

impl BakeCoverage {
    pub fn is_complete(&self) -> bool {
        self.baked_steps == self.total_steps
    }
}

/// What [`Project::seed_build_cache`] accepted into the cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeedReport {
    /// Step entries inserted into the cache.
    pub seeded_steps: usize,
    /// Budget cost of the inserted entries.
    pub seeded_bytes: usize,
    /// Blobs whose bytes did not match their recorded hash (file damage).
    pub corrupt_blobs: usize,
    /// Steps dropped because a blob they reference was corrupt or absent;
    /// they simply re-execute on the next build.
    pub skipped_steps: usize,
}

impl Project {
    /// Snapshots this project's step results out of `cache` without
    /// executing anything.
    ///
    /// Replays the timeline's key chain exactly like the executor's hit
    /// path: imports are hashed, each step is keyed from the hashes known
    /// so far, and a cache hit contributes both the snapshot entry and the
    /// output hashes that key the steps downstream. A miss (never run, or
    /// evicted) leaves that step and its dependents out of the snapshot —
    /// see the returned [`BakeCoverage`].
    pub fn collect_baked(&self, cache: &BuildCache) -> (BakedResults, BakeCoverage) {
        // Content hash of every asset id currently "in scope", updated in
        // timeline order so id shadowing resolves exactly as at runtime.
        let mut hashes: HashMap<String, [u8; 32]> = HashMap::new();
        for import in &self.imports {
            hashes.insert(import.id.clone(), *blake3::hash(&import.data).as_bytes());
        }

        let mut blobs: BTreeMap<[u8; 32], Arc<Vec<u8>>> = BTreeMap::new();
        let mut steps = Vec::new();

        for step in &self.timeline {
            let cached = (|| {
                let operator_hash = hashes.get(&step.operator_id)?;
                let mut input_hashes = Vec::with_capacity(step.inputs.len());
                for input in &step.inputs {
                    match input {
                        ExecutionInput::AssetRef(id) => input_hashes.push(*hashes.get(id)?),
                        ExecutionInput::Inline(data) => {
                            input_hashes.push(*blake3::hash(data).as_bytes())
                        }
                    }
                }
                let key = StepKey::new(operator_hash, &input_hashes);
                cache.get(&key).map(|entry| (key, entry))
            })();

            let Some((key, entry)) = cached else {
                // Unbakeable: this step's outputs have unknown hashes. Ids
                // it would shadow must not keep their stale hashes, or a
                // dependent step would bake under a key that can never be
                // hit at runtime.
                for output_id in &step.outputs {
                    hashes.remove(output_id);
                }
                continue;
            };

            let mut outputs: Vec<(u32, [u8; 32])> = entry
                .outputs
                .iter()
                .map(|(&slot, (data, hash))| {
                    blobs.entry(*hash).or_insert_with(|| Arc::clone(data));
                    (slot as u32, *hash)
                })
                .collect();
            outputs.sort_unstable_by_key(|&(slot, _)| slot);
            steps.push(BakedStep {
                key: *key.as_bytes(),
                outputs,
                declared_outputs: entry.declared_outputs.clone(),
            });

            for (idx, output_id) in step.outputs.iter().enumerate() {
                match entry.outputs.get(&idx) {
                    Some((_, hash)) => {
                        hashes.insert(output_id.clone(), *hash);
                    }
                    // The operator skipped this slot, so the id never
                    // materializes at runtime; drop anything it shadows.
                    None => {
                        hashes.remove(output_id);
                    }
                }
            }
        }

        let coverage = BakeCoverage {
            baked_steps: steps.len(),
            total_steps: self.timeline.len(),
        };
        let blobs = blobs
            .into_iter()
            .map(|(hash, data)| BakedBlob {
                hash,
                // Unshareable copy: BakedBlob owns plain bytes for serde.
                data: data.as_ref().clone(),
            })
            .collect();
        (BakedResults { blobs, steps }, coverage)
    }

    /// Moves this project's baked results (if any) into `cache`, verifying
    /// every blob's hash on the way in. Returns what was accepted.
    ///
    /// Consumes `self.baked` so a heavy snapshot is never resident twice
    /// (once in the project, once behind the cache's `Arc`s); re-saving a
    /// bake is a fresh [`collect_baked`](Self::collect_baked), which the
    /// seeded cache serves cheaply.
    ///
    /// The cache budget is raised as needed ([`BuildCache::reserve`]) so
    /// seeding never evicts resident entries and is never refused for
    /// exceeding the budget.
    pub fn seed_build_cache(&mut self, cache: &BuildCache) -> SeedReport {
        let Some(baked) = self.baked.take() else {
            return SeedReport::default();
        };
        let mut report = SeedReport::default();

        let mut blob_map: HashMap<[u8; 32], Arc<Vec<u8>>> =
            HashMap::with_capacity(baked.blobs.len());
        for blob in baked.blobs {
            if *blake3::hash(&blob.data).as_bytes() == blob.hash {
                blob_map.insert(blob.hash, Arc::new(blob.data));
            } else {
                report.corrupt_blobs += 1;
            }
        }

        // Assemble every entry before touching the cache so the budget
        // reservation covers exactly what gets inserted.
        let mut entries = Vec::with_capacity(baked.steps.len());
        for step in baked.steps {
            let outputs: Option<HashMap<usize, (Arc<Vec<u8>>, [u8; 32])>> = step
                .outputs
                .iter()
                .map(|&(slot, hash)| {
                    let data = blob_map.get(&hash)?;
                    Some((slot as usize, (Arc::clone(data), hash)))
                })
                .collect();
            // A partially-assembled entry would falsely claim the operator
            // skipped the missing slots, so the whole step is dropped.
            let Some(outputs) = outputs else {
                report.skipped_steps += 1;
                continue;
            };
            entries.push((
                StepKey::from_bytes(step.key),
                CachedStep {
                    outputs,
                    declared_outputs: step.declared_outputs,
                },
            ));
        }

        cache.reserve(entries.iter().map(|(_, step)| step.cost_bytes()).sum());
        for (key, step) in entries {
            report.seeded_steps += 1;
            report.seeded_bytes += step.cost_bytes();
            cache.insert(key, step);
        }
        report
    }
}
