//! Volumetric processing daemon: executes build work (project runs, model
//! meshing) on behalf of remote clients, over the HTTP/CBOR protocol defined
//! in `volumetric_protocol`.
//!
//! Structure: a fixed pool of accept threads pulls HTTP requests off one
//! `tiny_http::Server` and answers against shared [`DaemonState`]. Each
//! submitted job gets its own thread, gated by a counting semaphore so at
//! most `max_concurrent_jobs` execute at once (jobs beyond that sit
//! `Queued`). Long-poll status requests park on a condvar that job
//! completion notifies.
//!
//! Job-thread panics are caught and reported as job failures — one broken
//! operator or mesher input must not take the daemon down. Finished
//! outcomes are retained for `result_ttl` and then swept, so a client that
//! disappeared doesn't pin hundreds of MB of mesh forever.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use volumetric_protocol::{
    DaemonInfo, ExportedAsset, JobOutcome, JobOutput, JobRequest, JobState, JobStatus, JobTicket,
    MeshPayload, PROTOCOL_VERSION, from_cbor, to_cbor,
};

pub struct DaemonConfig {
    /// Address to bind, e.g. `127.0.0.1:7373`. Port 0 picks an ephemeral
    /// port (see [`DaemonHandle::addr`]).
    pub bind: SocketAddr,
    /// Jobs executing at once; further jobs queue. Meshing jobs each fan out
    /// over the shared rayon pool internally, so this bounds oversubscription
    /// rather than core usage.
    pub max_concurrent_jobs: usize,
    /// How long finished job outcomes are kept for pickup.
    pub result_ttl: Duration,
    /// HTTP accept threads. Each long-poll status request occupies one for
    /// its wait, so this bounds concurrent pollers, not throughput.
    pub accept_threads: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], volumetric_protocol::DEFAULT_PORT)),
            max_concurrent_jobs: 4,
            result_ttl: Duration::from_secs(600),
            accept_threads: 8,
        }
    }
}

struct JobEntry {
    state: JobState,
    cancel: Arc<AtomicBool>,
    outcome: Option<JobOutcome>,
    finished_at: Option<Instant>,
}

struct DaemonState {
    jobs: Mutex<HashMap<u64, JobEntry>>,
    /// Notified whenever any job finishes; status long-polls wait here.
    finished: Condvar,
    next_job_id: AtomicU64,
    /// Counting semaphore for execution slots.
    running: Mutex<usize>,
    slot_freed: Condvar,
    config: DaemonConfig,
}

/// A running daemon. Dropping the handle does not stop it; call
/// [`DaemonHandle::shutdown`] (used by tests) or let the process exit.
pub struct DaemonHandle {
    server: Arc<tiny_http::Server>,
    addr: SocketAddr,
    accept_threads: Vec<std::thread::JoinHandle<()>>,
}

impl DaemonHandle {
    /// The actually-bound address (resolves port 0).
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stops accepting requests and joins the accept threads. Jobs already
    /// executing are detached and finish on their own.
    pub fn shutdown(self) {
        // Each unblock() queues exactly one Control::Unblock, waking exactly
        // one recv() — issue one per accept thread. Queued messages persist,
        // so a thread busy with a request picks its unblock up when it
        // returns to recv().
        for _ in &self.accept_threads {
            self.server.unblock();
        }
        for thread in self.accept_threads {
            let _ = thread.join();
        }
    }

    /// Blocks until the daemon stops accepting (i.e. forever, absent
    /// [`Self::shutdown`]).
    pub fn wait(self) {
        for thread in self.accept_threads {
            let _ = thread.join();
        }
    }
}

/// Starts a daemon serving on `config.bind`.
pub fn start(config: DaemonConfig) -> anyhow::Result<DaemonHandle> {
    let server = Arc::new(
        tiny_http::Server::http(config.bind)
            .map_err(|e| anyhow::anyhow!("binding {} failed: {e}", config.bind))?,
    );
    let addr = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr,
        #[allow(unreachable_patterns)]
        other => anyhow::bail!("unexpected listen address {other:?}"),
    };
    let accept_thread_count = config.accept_threads.max(1);
    let state = Arc::new(DaemonState {
        jobs: Mutex::new(HashMap::new()),
        finished: Condvar::new(),
        next_job_id: AtomicU64::new(1),
        running: Mutex::new(0),
        slot_freed: Condvar::new(),
        config,
    });

    let accept_threads = (0..accept_thread_count)
        .map(|_| {
            let server = Arc::clone(&server);
            let state = Arc::clone(&state);
            std::thread::spawn(move || {
                while let Ok(request) = server.recv() {
                    handle_request(&state, request);
                }
            })
        })
        .collect();

    Ok(DaemonHandle {
        server,
        addr,
        accept_threads,
    })
}

// ---------------------------------------------------------------------------
// HTTP layer
// ---------------------------------------------------------------------------

fn handle_request(state: &Arc<DaemonState>, mut request: tiny_http::Request) {
    sweep_expired(state);

    let url = request.url().to_string();
    let (path, query) = match url.split_once('?') {
        Some((path, query)) => (path, query),
        None => (url.as_str(), ""),
    };
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let method = request.method().clone();

    let response = match (method.as_str(), segments.as_slice()) {
        ("GET", ["v1", "info"]) => cbor_response(&DaemonInfo {
            name: "volumetric_daemon".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: PROTOCOL_VERSION,
        }),
        ("POST", ["v1", "jobs"]) => match read_body(&mut request) {
            Ok(body) => submit_job(state, &body),
            Err(e) => text_response(400, &format!("unreadable request body: {e}")),
        },
        ("GET", ["v1", "jobs", id]) => match id.parse::<u64>() {
            Ok(id) => job_status(state, id, wait_ms_from_query(query)),
            Err(_) => text_response(400, "job id must be an integer"),
        },
        ("POST", ["v1", "jobs", id, "cancel"]) => match id.parse::<u64>() {
            Ok(id) => cancel_job(state, id),
            Err(_) => text_response(400, "job id must be an integer"),
        },
        _ => text_response(404, &format!("no such endpoint: {} {path}", method)),
    };

    let _ = request.respond(response);
}

fn read_body(request: &mut tiny_http::Request) -> std::io::Result<Vec<u8>> {
    let mut body = Vec::new();
    request.as_reader().read_to_end(&mut body)?;
    Ok(body)
}

fn wait_ms_from_query(query: &str) -> u64 {
    query
        .split('&')
        .find_map(|pair| pair.strip_prefix("wait_ms="))
        .and_then(|v| v.parse().ok())
        // Bound server-side so a stuck client can't park an accept thread
        // for minutes.
        .map(|ms: u64| ms.min(30_000))
        .unwrap_or(0)
}

type HttpResponse = tiny_http::Response<std::io::Cursor<Vec<u8>>>;

fn cbor_response<T: serde::Serialize>(value: &T) -> HttpResponse {
    let body = to_cbor(value);
    tiny_http::Response::from_data(body).with_header(
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/cbor"[..])
            .expect("static header"),
    )
}

fn text_response(status: u16, message: &str) -> HttpResponse {
    tiny_http::Response::from_data(message.as_bytes().to_vec()).with_status_code(status)
}

// ---------------------------------------------------------------------------
// Job lifecycle
// ---------------------------------------------------------------------------

fn submit_job(state: &Arc<DaemonState>, body: &[u8]) -> HttpResponse {
    let request: JobRequest = match from_cbor(body) {
        Ok(request) => request,
        Err(e) => return text_response(400, &format!("undecodable job request: {e}")),
    };

    let job_id = state.next_job_id.fetch_add(1, Ordering::Relaxed);
    let cancel = Arc::new(AtomicBool::new(false));
    state.jobs.lock().unwrap().insert(
        job_id,
        JobEntry {
            state: JobState::Queued,
            cancel: Arc::clone(&cancel),
            outcome: None,
            finished_at: None,
        },
    );

    let state = Arc::clone(state);
    std::thread::spawn(move || run_job(&state, job_id, request, &cancel));

    cbor_response(&JobTicket { job_id })
}

fn run_job(state: &Arc<DaemonState>, job_id: u64, request: JobRequest, cancel: &AtomicBool) {
    acquire_slot(state);

    let outcome = if cancel.load(Ordering::Relaxed) {
        JobOutcome::Cancelled
    } else {
        set_job_state(state, job_id, JobState::Running);
        let started = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute(&request, cancel)
        }));
        match result {
            Ok(Ok(Some(output))) => JobOutcome::Success {
                output,
                elapsed_ms: started.elapsed().as_millis() as u64,
            },
            Ok(Ok(None)) => JobOutcome::Cancelled,
            Ok(Err(error)) => JobOutcome::Failed { error },
            Err(panic) => JobOutcome::Failed {
                error: format!("job panicked: {}", panic_message(&panic)),
            },
        }
    };

    {
        let mut jobs = state.jobs.lock().unwrap();
        if let Some(entry) = jobs.get_mut(&job_id) {
            entry.state = JobState::Finished;
            entry.outcome = Some(outcome);
            entry.finished_at = Some(Instant::now());
        }
    }
    state.finished.notify_all();
    release_slot(state);
}

/// Runs one job payload. `Ok(None)` means cancelled.
fn execute(request: &JobRequest, cancel: &AtomicBool) -> Result<Option<JobOutput>, String> {
    match request {
        JobRequest::RunProject { project } => {
            let mut env = volumetric::Environment::new();
            match project.run_cancellable(&mut env, cancel) {
                Ok(exports) => Ok(Some(JobOutput::RunProject {
                    exports: exports.iter().map(ExportedAsset::from_loaded).collect(),
                })),
                Err(volumetric::ExecutionError::Cancelled) => Ok(None),
                Err(e) => Err(e.to_string()),
            }
        }
        JobRequest::MeshModel { model_wasm, config } => {
            match volumetric::generate_adaptive_mesh_v2_from_bytes_cancellable(
                model_wasm, config, cancel,
            ) {
                Ok(Some(result)) => Ok(Some(JobOutput::MeshModel {
                    mesh: MeshPayload::pack(
                        &result.vertices,
                        &result.normals,
                        &result.indices,
                        result.bounds_min.into(),
                        result.bounds_max.into(),
                    ),
                    stats: result.stats,
                })),
                Ok(None) => Ok(None),
                // {:#} flattens the anyhow context chain into one line.
                Err(e) => Err(format!("{e:#}")),
            }
        }
    }
}

fn job_status(state: &Arc<DaemonState>, job_id: u64, wait_ms: u64) -> HttpResponse {
    let deadline = Instant::now() + Duration::from_millis(wait_ms);
    let mut jobs = state.jobs.lock().unwrap();
    loop {
        let Some(entry) = jobs.get(&job_id) else {
            return text_response(404, &format!("no job {job_id} (expired or never existed)"));
        };
        if entry.outcome.is_some() {
            return cbor_response(&JobStatus {
                state: entry.state,
                outcome: entry.outcome.clone(),
            });
        }
        let now = Instant::now();
        if now >= deadline {
            return cbor_response(&JobStatus {
                state: entry.state,
                outcome: None,
            });
        }
        let (guard, _timeout) = state
            .finished
            .wait_timeout(jobs, deadline - now)
            .expect("daemon state mutex poisoned");
        jobs = guard;
    }
}

fn cancel_job(state: &Arc<DaemonState>, job_id: u64) -> HttpResponse {
    let jobs = state.jobs.lock().unwrap();
    match jobs.get(&job_id) {
        Some(entry) => {
            entry.cancel.store(true, Ordering::Relaxed);
            text_response(200, "cancellation requested")
        }
        None => text_response(404, &format!("no job {job_id} (expired or never existed)")),
    }
}

fn set_job_state(state: &Arc<DaemonState>, job_id: u64, job_state: JobState) {
    if let Some(entry) = state.jobs.lock().unwrap().get_mut(&job_id) {
        entry.state = job_state;
    }
}

fn sweep_expired(state: &Arc<DaemonState>) {
    let ttl = state.config.result_ttl;
    state
        .jobs
        .lock()
        .unwrap()
        .retain(|_, entry| match entry.finished_at {
            Some(finished_at) => finished_at.elapsed() < ttl,
            None => true,
        });
}

fn acquire_slot(state: &Arc<DaemonState>) {
    let mut running = state.running.lock().unwrap();
    while *running >= state.config.max_concurrent_jobs.max(1) {
        running = state
            .slot_freed
            .wait(running)
            .expect("daemon slot mutex poisoned");
    }
    *running += 1;
}

fn release_slot(state: &Arc<DaemonState>) {
    *state.running.lock().unwrap() -= 1;
    state.slot_freed.notify_one();
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}
