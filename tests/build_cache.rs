//! Integration tests for step-result caching in `Project::run_monitored_with_cache`.
//!
//! Hand-written WAT operators (wasmtime compiles text modules directly)
//! keep these independent of the wasm32 artifact builds: the cache keys on
//! operator/input bytes, not on what the operators compute.

#![cfg(feature = "native")]

use std::{cell::RefCell, sync::atomic::AtomicBool};

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

#[test]
fn artifact_stream_is_ordered_and_identical_on_cache_hits() {
    let mut project = Project::new();
    project
        .imports_mut()
        .push(ImportedAsset::operator("echo".into(), echo_operator(0)));
    project.imports_mut().push(ImportedAsset::new(
        "src".into(),
        b"model bytes".to_vec(),
        Some(volumetric::AssetTypeHint::Model),
    ));
    project.timeline_mut().extend([
        step("echo", vec![ExecutionInput::AssetRef("src".into())], "a"),
        step("echo", vec![ExecutionInput::AssetRef("a".into())], "b"),
    ]);
    project.exports_mut().push("b".into());

    let cache = BuildCache::new(64 << 20);
    for expected_hits in [1, 3] {
        let ready = RefCell::new(Vec::new());
        let exports = project
            .run_monitored_with_cache_and_artifacts(
                &mut Environment::new(),
                &cache,
                &AtomicBool::new(false),
                &|_| {},
                &|artifact| ready.borrow_mut().push(artifact.id().to_string()),
            )
            .expect("project runs");

        assert_eq!(&*ready.borrow(), &["echo", "src", "a", "b"]);
        assert_eq!(exports[0].id(), "b");
        assert_eq!(cache.stats().hits, expected_hits);
    }
}

/// A two-step chain re-run unchanged: every step is served from cache, and
/// the exported blob is the identical allocation (pointer-equal `Arc`), which
/// downstream identity-keyed caches rely on.
///
/// The steps use two distinct echo operators: chaining one identity operator
/// through its own output would give both steps the same memo key (see
/// `identical_steps_deduplicate_within_one_run`).
#[test]
fn second_run_serves_every_step_from_cache() {
    let mut project = Project::new();
    project
        .imports_mut()
        .push(ImportedAsset::operator("echo0".into(), echo_operator(0)));
    project
        .imports_mut()
        .push(ImportedAsset::operator("echo1".into(), echo_operator(1)));
    project
        .imports_mut()
        .push(ImportedAsset::new("src".into(), b"hello".to_vec(), None));
    project.timeline_mut().extend([
        step("echo0", vec![ExecutionInput::AssetRef("src".into())], "a"),
        step(
            "echo1",
            vec![
                ExecutionInput::Inline(b"pad".to_vec()),
                ExecutionInput::AssetRef("a".into()),
            ],
            "b",
        ),
    ]);
    project.exports_mut().push("b".into());

    let cache = BuildCache::new(64 << 20);
    let first = run(&project, &cache);
    let stats = cache.stats();
    assert_eq!((stats.hits, stats.misses), (0, 2));
    assert_eq!(first[0].data(), b"hello");

    let second = run(&project, &cache);
    let stats = cache.stats();
    assert_eq!((stats.hits, stats.misses), (2, 2));
    assert_eq!(second[0].data(), b"hello");
    assert!(
        std::sync::Arc::ptr_eq(&first[0].data_arc(), &second[0].data_arc()),
        "cached export must be the same allocation across runs"
    );
}

/// Two byte-identical steps in one run share one execution: an identity
/// operator fed its own output reproduces its input, so the second step's
/// key (same operator, same input bytes) is the first step's memo.
#[test]
fn identical_steps_deduplicate_within_one_run() {
    let mut project = Project::new();
    project
        .imports_mut()
        .push(ImportedAsset::operator("echo".into(), echo_operator(0)));
    project
        .imports_mut()
        .push(ImportedAsset::new("src".into(), b"hello".to_vec(), None));
    project.timeline_mut().extend([
        step("echo", vec![ExecutionInput::AssetRef("src".into())], "a"),
        step("echo", vec![ExecutionInput::AssetRef("a".into())], "b"),
    ]);
    project.exports_mut().push("b".into());

    let cache = BuildCache::new(64 << 20);
    let exports = run(&project, &cache);
    assert_eq!(exports[0].data(), b"hello");
    assert_eq!((cache.stats().hits, cache.stats().misses), (1, 1));
}

/// Editing a downstream step's inline config leaves the upstream step
/// cached — the heavy-FEA-chain scenario: only the edited step re-runs.
#[test]
fn downstream_edit_keeps_upstream_cached() {
    let make_project = |config: &[u8]| {
        let mut project = Project::new();
        project
            .imports_mut()
            .push(ImportedAsset::operator("echo0".into(), echo_operator(0)));
        project
            .imports_mut()
            .push(ImportedAsset::operator("echo1".into(), echo_operator(1)));
        project
            .imports_mut()
            .push(ImportedAsset::new("src".into(), b"payload".to_vec(), None));
        project.timeline_mut().extend([
            step("echo0", vec![ExecutionInput::AssetRef("src".into())], "a"),
            // Echoes the config, so its output genuinely depends on it.
            step(
                "echo1",
                vec![
                    ExecutionInput::AssetRef("a".into()),
                    ExecutionInput::Inline(config.to_vec()),
                ],
                "b",
            ),
        ]);
        project.exports_mut().push("b".into());
        project
    };

    let cache = BuildCache::new(64 << 20);
    let first = run(&make_project(b"config v1"), &cache);
    assert_eq!(first[0].data(), b"config v1");
    assert_eq!((cache.stats().hits, cache.stats().misses), (0, 2));

    let second = run(&make_project(b"config v2"), &cache);
    assert_eq!(second[0].data(), b"config v2");
    let stats = cache.stats();
    assert_eq!(
        (stats.hits, stats.misses),
        (1, 3),
        "step 'a' hits, edited step 'b' re-runs"
    );
}

/// Early cutoff: when an edited step happens to produce byte-identical
/// output, everything downstream of it still hits the cache.
#[test]
fn identical_intermediate_cuts_off_downstream_reruns() {
    let make_project = |config: &[u8]| {
        let mut project = Project::new();
        project
            .imports_mut()
            .push(ImportedAsset::operator("const".into(), const_operator()));
        project
            .imports_mut()
            .push(ImportedAsset::operator("echo".into(), echo_operator(0)));
        project.timeline_mut().extend([
            // Takes the config as an input but ignores it: the step's key
            // changes with the config while its output does not.
            step("const", vec![ExecutionInput::Inline(config.to_vec())], "a"),
            step("echo", vec![ExecutionInput::AssetRef("a".into())], "b"),
        ]);
        project.exports_mut().push("b".into());
        project
    };

    let cache = BuildCache::new(64 << 20);
    run(&make_project(b"config v1"), &cache);
    assert_eq!((cache.stats().hits, cache.stats().misses), (0, 2));

    let second = run(&make_project(b"config v2"), &cache);
    assert_eq!(second[0].data(), b"\xde\xad\xbe\xef");
    let stats = cache.stats();
    assert_eq!(
        (stats.hits, stats.misses),
        (1, 3),
        "'const' re-runs but 'echo' hits on the unchanged intermediate"
    );
}
