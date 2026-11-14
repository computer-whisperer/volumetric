//! Blocking HTTP client for a volumetric daemon.
//!
//! Deliberately synchronous: every consumer runs build work on a dedicated
//! background thread already (the UI's background worker, the CLI's main
//! thread), so a blocking client slots in where a local
//! `Project::run_cancellable` call sits today.

use std::io::Read;
use std::time::Duration;

use crate::error::POLL_WAIT_MS;
use crate::{
    ClientError, DaemonInfo, JobOutcome, JobRequest, JobState, JobStatus, JobTicket,
    PROTOCOL_VERSION, from_cbor, to_cbor,
};

/// Blocking client for one daemon endpoint.
pub struct DaemonClient {
    base: String,
    agent: ureq::Agent,
}

impl DaemonClient {
    /// `base_url` like `http://buildbox:7373` (scheme required, no trailing
    /// slash needed).
    pub fn new(base_url: &str) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(10)))
            // A long-poll intentionally idles server-side before the
            // response headers; results can be large, so the body gets a
            // budget of its own.
            .timeout_recv_response(Some(Duration::from_secs(120)))
            .timeout_recv_body(Some(Duration::from_secs(600)))
            // Error statuses come back as responses, so their bodies (the
            // daemon's message) can be read.
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            base: base_url.trim_end_matches('/').to_string(),
            agent,
        }
    }

    /// Fetches daemon identity and verifies the protocol version matches.
    pub fn info(&self) -> Result<DaemonInfo, ClientError> {
        let info: DaemonInfo = self.get(&format!("{}/v1/info", self.base))?;
        if info.protocol_version != PROTOCOL_VERSION {
            return Err(ClientError::VersionMismatch {
                daemon: info.protocol_version,
                client: PROTOCOL_VERSION,
            });
        }
        Ok(info)
    }

    /// Submits a job; returns its id.
    pub fn submit(&self, request: &JobRequest) -> Result<u64, ClientError> {
        let ticket: JobTicket = self.post(&format!("{}/v1/jobs", self.base), to_cbor(request))?;
        Ok(ticket.job_id)
    }

    /// One status request, holding server-side up to `wait_ms` for the job
    /// to finish.
    pub fn status(&self, job_id: u64, wait_ms: u64) -> Result<JobStatus, ClientError> {
        self.get(&format!("{}/v1/jobs/{job_id}?wait_ms={wait_ms}", self.base))
    }

    /// Requests cancellation. The job still runs to its next checkpoint and
    /// must be awaited for its (now likely `Cancelled`) outcome.
    pub fn cancel(&self, job_id: u64) -> Result<(), ClientError> {
        let response = self
            .agent
            .post(&format!("{}/v1/jobs/{job_id}/cancel", self.base))
            .send_empty()
            .map_err(transport_error)?;
        // Drain so the connection is reusable.
        read_body(check_status(response)?)?;
        Ok(())
    }

    /// Long-polls until the job finishes. `cancel_requested` is consulted
    /// between polls; the first time it reports true, cancellation is
    /// forwarded to the daemon and the wait continues until the daemon
    /// acknowledges with a final outcome. `on_progress` receives the job's
    /// latest [`crate::BuildProgress`] whenever a poll returns one — at the
    /// polling cadence, so an indicator rather than a stream.
    pub fn wait(
        &self,
        job_id: u64,
        cancel_requested: &dyn Fn() -> bool,
        on_progress: &dyn Fn(crate::BuildProgress),
    ) -> Result<JobOutcome, ClientError> {
        let mut cancel_sent = false;
        loop {
            if !cancel_sent && cancel_requested() {
                self.cancel(job_id)?;
                cancel_sent = true;
            }
            let status = self.status(job_id, POLL_WAIT_MS)?;
            if let Some(outcome) = status.outcome {
                return Ok(outcome);
            }
            if let Some(progress) = status.progress {
                on_progress(progress);
            }
            debug_assert!(matches!(status.state, JobState::Queued | JobState::Running));
        }
    }

    /// Submit + wait in one call.
    pub fn run(
        &self,
        request: &JobRequest,
        cancel_requested: &dyn Fn() -> bool,
        on_progress: &dyn Fn(crate::BuildProgress),
    ) -> Result<JobOutcome, ClientError> {
        let job_id = self.submit(request)?;
        self.wait(job_id, cancel_requested, on_progress)
    }

    fn get<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, ClientError> {
        let response = self.agent.get(url).call().map_err(transport_error)?;
        decode_body(response)
    }

    fn post<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        body: Vec<u8>,
    ) -> Result<T, ClientError> {
        let response = self
            .agent
            .post(url)
            .content_type("application/cbor")
            .send(&body[..])
            .map_err(transport_error)?;
        decode_body(response)
    }
}

type Response = ureq::http::Response<ureq::Body>;

fn decode_body<T: serde::de::DeserializeOwned>(response: Response) -> Result<T, ClientError> {
    let bytes = read_body(check_status(response)?)?;
    from_cbor(&bytes).map_err(ClientError::Decode)
}

/// The whole body. Unlimited, unlike ureq's `read_to_vec` (10 MB): build
/// results are routinely larger.
fn read_body(response: Response) -> Result<Vec<u8>, ClientError> {
    let mut bytes = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| ClientError::Transport(e.to_string()))?;
    Ok(bytes)
}

/// A non-2xx response as [`ClientError::Api`], carrying the daemon's
/// message from the body.
fn check_status(response: Response) -> Result<Response, ClientError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let message = read_body(response)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_else(|| "<unreadable error body>".to_string());
    Err(ClientError::Api {
        status: status.as_u16(),
        message,
    })
}

fn transport_error(error: ureq::Error) -> ClientError {
    ClientError::Transport(error.to_string())
}
