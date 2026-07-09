//! End-to-end daemon tests: a real daemon on an ephemeral port, driven
//! through the protocol crate's HTTP client. Operators and models are tiny
//! WAT modules (wasmtime compiles text directly), so no wasm32 artifacts
//! are needed.

use std::net::SocketAddr;
use std::time::Duration;

use volumetric::{ExecutionStep, ImportedAsset, Project};
use volumetric_protocol::{DaemonClient, JobOutcome, JobOutput, JobRequest, JobState};

fn start_daemon(max_jobs: usize) -> (volumetric_daemon::DaemonHandle, DaemonClient) {
    let handle = volumetric_daemon::start(volumetric_daemon::DaemonConfig {
        bind: SocketAddr::from(([127, 0, 0, 1], 0)),
        max_concurrent_jobs: max_jobs,
        ..Default::default()
    })
    .expect("daemon starts on an ephemeral port");
    let client = DaemonClient::new(&format!("http://{}", handle.addr()));
    (handle, client)
}

/// An operator that posts one 4-byte blob to output 0.
fn blob_operator() -> Vec<u8> {
    r#"(module
        (import "host" "post_output" (func $post_output (param i32 i32 i32)))
        (memory (export "memory") 1)
        (data (i32.const 0) "\de\ad\be\ef")
        (func (export "run")
            (call $post_output (i32.const 0) (i32.const 0) (i32.const 4))))"#
        .as_bytes()
        .to_vec()
}

/// An operator that spins ~a second before posting, so cancellation has a
/// window to land between timeline steps.
fn slow_operator() -> Vec<u8> {
    r#"(module
        (import "host" "post_output" (func $post_output (param i32 i32 i32)))
        (memory (export "memory") 1)
        (func (export "run")
            (local $i i64)
            (local.set $i (i64.const 2000000000))
            (block (loop
                (br_if 1 (i64.eqz (local.get $i)))
                (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                (br 0)))
            (call $post_output (i32.const 0) (i32.const 0) (i32.const 4))))"#
        .as_bytes()
        .to_vec()
}

/// A model of the axis-aligned box |x|,|y|,|z| <= 0.4 inside ±0.5 bounds.
/// Assembled to binary: the model pipeline (unlike the operator pipeline)
/// pre-validates bytes with wasmparser, which rejects the text format.
fn box_model() -> Vec<u8> {
    wat::parse_str(
        r#"(module
        (memory (export "memory") 1)
        (func (export "get_dimensions") (result i32) (i32.const 3))
        ;; Model-owned IO buffer (>= 2*dims f64s): host writes positions
        ;; here and reads bounds from it.
        (func (export "get_io_ptr") (result i32) (i32.const 1024))
        (func (export "get_bounds") (param $out i32)
            (f64.store (local.get $out) (f64.const -0.5))
            (f64.store offset=8 (local.get $out) (f64.const 0.5))
            (f64.store offset=16 (local.get $out) (f64.const -0.5))
            (f64.store offset=24 (local.get $out) (f64.const 0.5))
            (f64.store offset=32 (local.get $out) (f64.const -0.5))
            (f64.store offset=40 (local.get $out) (f64.const 0.5)))
        (func (export "sample") (param $p i32) (result f32)
            (if (result f32)
                (i32.and
                    (i32.and
                        (f64.le (f64.abs (f64.load (local.get $p))) (f64.const 0.4))
                        (f64.le (f64.abs (f64.load offset=8 (local.get $p))) (f64.const 0.4)))
                    (f64.le (f64.abs (f64.load offset=16 (local.get $p))) (f64.const 0.4)))
                (then (f32.const 1.0))
                (else (f32.const 0.0)))))"#,
    )
    .expect("box model WAT assembles")
}

fn one_step_project(operator: Vec<u8>, steps: usize) -> Project {
    let timeline = (0..steps)
        .map(|i| ExecutionStep {
            operator_id: "op".to_string(),
            inputs: vec![],
            outputs: vec![format!("out_{i}")],
        })
        .collect();
    Project {
        version: 2,
        imports: vec![ImportedAsset::operator("op".to_string(), operator)],
        timeline,
        exports: (0..steps).map(|i| format!("out_{i}")).collect(),
    }
}

const NEVER_CANCEL: &dyn Fn() -> bool = &|| false;
const IGNORE_PROGRESS: &dyn Fn(volumetric_protocol::BuildProgress) = &|_| {};

/// `run_monitored` reports one snapshot per timeline step with a rising
/// fraction — the payload the daemon's status endpoint serves to pollers.
#[test]
fn run_monitored_reports_each_step() {
    let project = one_step_project(blob_operator(), 3);
    let steps: std::cell::RefCell<Vec<volumetric::BuildProgress>> =
        std::cell::RefCell::new(Vec::new());
    let never = std::sync::atomic::AtomicBool::new(false);
    project
        .run_monitored(&mut volumetric::Environment::new(), &never, &|p| {
            steps.borrow_mut().push(p)
        })
        .expect("project runs");
    let steps = steps.into_inner();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0].phase, "op (1/3)");
    assert_eq!(steps[0].fraction, Some(0.0));
    assert_eq!(steps[2].phase, "op (3/3)");
    assert_eq!(steps[2].fraction, Some(2.0 / 3.0));
}

#[test]
fn info_reports_matching_protocol_version() {
    let (handle, client) = start_daemon(1);
    let info = client.info().expect("info answers and versions match");
    assert_eq!(info.name, "volumetric_daemon");
    handle.shutdown();
}

#[test]
fn run_project_returns_exports() {
    let (handle, client) = start_daemon(1);

    let request = JobRequest::RunProject {
        project: one_step_project(blob_operator(), 1),
    };
    let outcome = client.run(&request, NEVER_CANCEL, IGNORE_PROGRESS).expect("job completes");

    let JobOutcome::Success { output, .. } = outcome else {
        panic!("expected success, got {outcome:?}");
    };
    let JobOutput::RunProject { exports } = output else {
        panic!("wrong output kind");
    };
    assert_eq!(exports.len(), 1);
    assert_eq!(exports[0].id, "out_0");
    assert_eq!(exports[0].data, vec![0xde, 0xad, 0xbe, 0xef]);
    // The wire asset reassembles into the engine type the UI consumes.
    let loaded = exports[0].clone().into_loaded();
    assert_eq!(loaded.id(), "out_0");
    assert_eq!(loaded.data(), &[0xde, 0xad, 0xbe, 0xef]);

    handle.shutdown();
}

#[test]
fn run_project_reports_execution_errors() {
    let (handle, client) = start_daemon(1);

    // The step references an operator asset that doesn't exist.
    let project = Project {
        version: 2,
        imports: vec![],
        timeline: vec![ExecutionStep {
            operator_id: "missing".to_string(),
            inputs: vec![],
            outputs: vec!["out".to_string()],
        }],
        exports: vec![],
    };
    let outcome = client
        .run(&JobRequest::RunProject { project }, NEVER_CANCEL, IGNORE_PROGRESS)
        .expect("job completes");

    let JobOutcome::Failed { error } = outcome else {
        panic!("expected failure, got {outcome:?}");
    };
    assert!(error.contains("missing"), "unhelpful error: {error}");

    handle.shutdown();
}

#[test]
fn mesh_model_returns_a_plausible_box_mesh() {
    let (handle, client) = start_daemon(1);

    let config = volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2 {
        base_resolution: 8,
        max_depth: 2,
        ..Default::default()
    };
    let request = JobRequest::MeshModel {
        model_wasm: box_model(),
        config,
    };
    let outcome = client.run(&request, NEVER_CANCEL, IGNORE_PROGRESS).expect("job completes");

    let JobOutcome::Success { output, .. } = outcome else {
        panic!("expected success, got {outcome:?}");
    };
    let JobOutput::MeshModel { mesh, stats } = output else {
        panic!("wrong output kind");
    };
    let positions = mesh.unpack_positions().expect("valid positions");
    let indices = mesh.unpack_indices().expect("valid indices");
    assert!(!positions.is_empty(), "box surface produced no vertices");
    assert_eq!(indices.len() % 3, 0);
    assert_eq!(stats.total_vertices, positions.len());
    // Every vertex sits near the ±0.4 box surface, inside declared bounds.
    for &(x, y, z) in &positions {
        for v in [x, y, z] {
            assert!(v.abs() <= 0.5, "vertex escaped bounds: {x} {y} {z}");
        }
        let m = x.abs().max(y.abs()).max(z.abs());
        assert!(
            (m - 0.4).abs() < 0.1,
            "vertex far from box surface: {x} {y} {z}"
        );
    }

    handle.shutdown();
}

#[test]
fn cancellation_stops_queued_and_running_jobs() {
    let (handle, client) = start_daemon(1);

    // Job A: two slow steps, so a cancel can land between them.
    let a = client
        .submit(&JobRequest::RunProject {
            project: one_step_project(slow_operator(), 2),
        })
        .expect("submit A");
    // Don't submit B until A holds the single slot, so B is deterministically
    // the queued one.
    loop {
        let status = client.status(a, 100).expect("status A");
        if status.state == JobState::Running {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let b = client
        .submit(&JobRequest::RunProject {
            project: one_step_project(slow_operator(), 2),
        })
        .expect("submit B");

    client.cancel(b).expect("cancel B");
    client.cancel(a).expect("cancel A");

    let outcome_b = client.wait(b, NEVER_CANCEL, IGNORE_PROGRESS).expect("await B");
    assert!(
        matches!(outcome_b, JobOutcome::Cancelled),
        "queued job should cancel without running, got {outcome_b:?}"
    );
    let outcome_a = client.wait(a, NEVER_CANCEL, IGNORE_PROGRESS).expect("await A");
    assert!(
        matches!(outcome_a, JobOutcome::Cancelled),
        "running job should cancel between steps, got {outcome_a:?}"
    );

    handle.shutdown();
}

#[test]
fn unknown_job_is_a_clean_api_error() {
    let (handle, client) = start_daemon(1);
    let error = client.status(999_999, 0).expect_err("no such job");
    assert!(
        matches!(
            error,
            volumetric_protocol::ClientError::Api { status: 404, .. }
        ),
        "expected 404, got {error:?}"
    );
    handle.shutdown();
}
