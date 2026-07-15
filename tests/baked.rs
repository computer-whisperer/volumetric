//! Integration tests for baked projects: step results collected out of the
//! build cache, persisted through the `.vproj` CBOR round trip, and seeded
//! back into a fresh cache so a rebuild executes nothing.
//!
//! Hand-written WAT operators (wasmtime compiles text modules directly)
//! keep these independent of the wasm32 artifact builds, mirroring
//! `tests/build_cache.rs`.

#![cfg(feature = "native")]

use std::sync::atomic::AtomicBool;

use volumetric::{
    BuildCache, Environment, ExecutionInput, ExecutionStep, ImportedAsset, LoadedAsset, Project,
};

/// Echoes input `idx` to output 0.
fn echo_operator(idx: u32) -> Vec<u8> {
    format!(
        r#"(module
            (import "host" "get_input_len" (func $len (param i32) (result i32)))
            (import "host" "get_input_data" (func $data (param i32 i32 i32)))
            (import "host" "post_output" (func $post (param i32 i32 i32)))
            (memory (export "memory") 16)
            (func (export "run")
                (local $n i32)
                (local.set $n (call $len (i32.const {idx})))
                (call $data (i32.const {idx}) (i32.const 0) (local.get $n))
                (call $post (i32.const 0) (i32.const 0) (local.get $n))))"#
    )
    .into_bytes()
}

/// Posts a fixed 4-byte blob to output 0, ignoring all inputs.
fn const_operator() -> Vec<u8> {
    r#"(module
        (import "host" "post_output" (func $post (param i32 i32 i32)))
        (memory (export "memory") 1)
        (data (i32.const 0) "\de\ad\be\ef")
        (func (export "run")
            (call $post (i32.const 0) (i32.const 0) (i32.const 4))))"#
        .as_bytes()
        .to_vec()
}

fn step(operator_id: &str, inputs: Vec<ExecutionInput>, output: &str) -> ExecutionStep {
    ExecutionStep {
        operator_id: operator_id.to_string(),
        inputs,
        outputs: vec![output.to_string()],
    }
}

fn run(project: &Project, cache: &BuildCache) -> Vec<LoadedAsset> {
    let never = AtomicBool::new(false);
    project
        .run_monitored_with_cache(&mut Environment::new(), cache, &never, &|_| {})
        .expect("project runs")
}

/// A two-step chain: `echo0(src) -> a`, then `const(a) -> b`. Distinct
/// operators keep the two step keys distinct.
fn chain_project() -> Project {
    let mut project = Project::new();
    project
        .imports_mut()
        .push(ImportedAsset::operator("echo0".into(), echo_operator(0)));
    project
        .imports_mut()
        .push(ImportedAsset::operator("konst".into(), const_operator()));
    project
        .imports_mut()
        .push(ImportedAsset::new("src".into(), b"hello".to_vec(), None));
    project.timeline_mut().extend([
        step("echo0", vec![ExecutionInput::AssetRef("src".into())], "a"),
        step("konst", vec![ExecutionInput::AssetRef("a".into())], "b"),
    ]);
    project.exports_mut().extend(["a".into(), "b".into()]);
    project
}

/// The whole feature end to end: build once, collect a complete bake,
/// round-trip it through the `.vproj` CBOR encoding, seed a fresh
/// zero-budget cache in a "new process", and rebuild without executing a
/// single step.
#[test]
fn baked_round_trip_rebuilds_from_cache_alone() {
    let project = chain_project();
    let cache = BuildCache::new(64 << 20);
    let built = run(&project, &cache);
    assert_eq!(built[0].data(), b"hello");
    assert_eq!(built[1].data(), b"\xde\xad\xbe\xef");

    let (baked, coverage) = project.collect_baked(&cache);
    assert!(coverage.is_complete());
    assert_eq!((coverage.baked_steps, coverage.total_steps), (2, 2));
    assert_eq!(baked.steps.len(), 2);
    // "hello" and "\xde\xad\xbe\xef" — two distinct blobs.
    assert_eq!(baked.blobs.len(), 2);
    assert_eq!(baked.blob_bytes(), 5 + 4);

    let mut saved = project.clone();
    saved.baked = Some(baked);
    let bytes = saved.to_cbor().expect("serializes");

    // "New process": fresh cache with no budget of its own.
    let mut reopened = Project::from_cbor(&bytes).expect("deserializes");
    let fresh = BuildCache::new(0);
    let report = reopened.seed_build_cache(&fresh);
    assert_eq!(
        (
            report.seeded_steps,
            report.corrupt_blobs,
            report.skipped_steps
        ),
        (2, 0, 0)
    );
    assert!(
        reopened.baked.is_none(),
        "seeding consumes the in-memory bake"
    );
    let stats = fresh.stats();
    assert_eq!(stats.entries, 2);
    assert!(
        stats.budget >= stats.bytes,
        "reserve raised the budget to fit the bake"
    );

    let rebuilt = run(&reopened, &fresh);
    let stats = fresh.stats();
    assert_eq!(
        (stats.hits, stats.misses),
        (2, 0),
        "every step must come from the seeded cache"
    );
    assert_eq!(rebuilt[0].data(), b"hello");
    assert_eq!(rebuilt[1].data(), b"\xde\xad\xbe\xef");
}

/// Collecting against a cache that only saw a prefix of the timeline bakes
/// exactly that prefix: the missing step's outputs can't be keyed, so it
/// and its dependents stay unbaked and re-run on open.
#[test]
fn partial_cache_bakes_prefix_and_reports_coverage() {
    let full = chain_project();
    let mut prefix = full.clone();
    prefix.timeline_mut().truncate(1);
    prefix.exports_mut().clear();
    prefix.exports_mut().push("a".into());

    let cache = BuildCache::new(64 << 20);
    run(&prefix, &cache);

    let (baked, coverage) = full.collect_baked(&cache);
    assert_eq!((coverage.baked_steps, coverage.total_steps), (1, 2));
    assert!(!coverage.is_complete());

    let mut reopened = full.clone();
    reopened.baked = Some(baked);
    let fresh = BuildCache::new(0);
    let report = reopened.seed_build_cache(&fresh);
    assert_eq!(report.seeded_steps, 1);

    let rebuilt = run(&reopened, &fresh);
    let stats = fresh.stats();
    assert_eq!(
        (stats.hits, stats.misses),
        (1, 1),
        "baked prefix hits, unbaked tail re-executes"
    );
    assert_eq!(rebuilt[1].data(), b"\xde\xad\xbe\xef");
}

/// A blob whose bytes don't match its recorded hash is rejected on seed,
/// the steps referencing it are dropped, and the rebuild recomputes them —
/// file damage costs time, never correctness.
#[test]
fn corrupt_blob_is_rejected_and_recomputed() {
    let project = chain_project();
    let cache = BuildCache::new(64 << 20);
    run(&project, &cache);
    let (mut baked, _) = project.collect_baked(&cache);

    // Damage the "hello" blob (the 5-byte one); only step 1 references it.
    let blob = baked
        .blobs
        .iter_mut()
        .find(|blob| blob.data.len() == 5)
        .expect("hello blob present");
    blob.data[0] ^= 0xff;

    let mut reopened = project.clone();
    reopened.baked = Some(baked);
    let fresh = BuildCache::new(0);
    let report = reopened.seed_build_cache(&fresh);
    assert_eq!(
        (
            report.seeded_steps,
            report.corrupt_blobs,
            report.skipped_steps
        ),
        (1, 1, 1)
    );

    let rebuilt = run(&reopened, &fresh);
    let stats = fresh.stats();
    // Step 1 recomputes (miss); its output hash matches the original, so
    // step 2's key still hits the seeded entry.
    assert_eq!((stats.hits, stats.misses), (1, 1));
    assert_eq!(rebuilt[0].data(), b"hello");
    assert_eq!(rebuilt[1].data(), b"\xde\xad\xbe\xef");
}

/// The bake is deterministic: collecting twice from the same cache yields
/// byte-identical project files (steps in timeline order, blobs sorted by
/// hash), so re-saving an unchanged built copy is a no-op diff.
#[test]
fn collection_is_deterministic() {
    let project = chain_project();
    let cache = BuildCache::new(64 << 20);
    run(&project, &cache);

    let mut first = project.clone();
    first.baked = Some(project.collect_baked(&cache).0);
    let mut second = project.clone();
    second.baked = Some(project.collect_baked(&cache).0);
    assert_eq!(
        first.to_cbor().unwrap(),
        second.to_cbor().unwrap(),
        "identical cache state must serialize identically"
    );
}

/// Projects without a bake seed nothing, and a cold cache bakes nothing —
/// both edges of the feature are quiet no-ops.
#[test]
fn missing_bake_and_cold_cache_are_no_ops() {
    let mut project = chain_project();
    let cache = BuildCache::new(64 << 20);

    let report = project.seed_build_cache(&cache);
    assert_eq!(report, volumetric::SeedReport::default());

    let (baked, coverage) = project.collect_baked(&cache);
    assert!(baked.is_empty());
    assert_eq!((coverage.baked_steps, coverage.total_steps), (0, 2));
}

/// A pre-bake `.vproj` (no `baked` field) still loads, and a baked file's
/// project fields survive the round trip untouched.
#[test]
fn cbor_compatibility_with_unbaked_files() {
    let project = chain_project();
    let bytes = project.to_cbor().expect("serializes without bake");
    let loaded = Project::from_cbor(&bytes).expect("loads without bake");
    assert!(loaded.baked.is_none());
    assert_eq!(loaded.timeline().len(), 2);
}
