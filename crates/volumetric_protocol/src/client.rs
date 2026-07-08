//! Blocking HTTP client for a volumetric daemon.
//!
//! Deliberately synchronous: every consumer runs build work on a dedicated
//! background thread already (the UI's background worker, the CLI's main
//! thread), so a blocking client slots in where a local
//! `Project::run_cancellable` call sits today.

use std::io::Read;
use std::time::Duration;

use crate::{
    DaemonInfo, JobOutcome, JobRequest, JobState, JobStatus, JobTicket, PROTOCOL_VERSION,
    from_cbor, to_cbor,
};

/// How long one status request asks the daemon to hold before answering.
/// Short enough that a cancel request is forwarded promptly.
const POLL_WAIT_MS: u64 = 2_000;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// The daemon could not be reached (connection refused, socket error,
    /// timeout).
    #[error("daemon unreachable: {0}")]
    Transport(String),
    /// The daemon answered with an error status.
    #[error("daemon rejected the request ({status}): {message}")]
    Api { status: u16, message: String },
    /// The daemon answered 200 but the body did not decode; usually a
    /// protocol version mismatch.
    #[error("undecodable daemon response: {0}")]
    Decode(String),
    /// The daemon speaks a different protocol version.
    #[error("daemon speaks protocol v{daemon}, this client speaks v{client}")]
    VersionMismatch { daemon: u32, client: u32 },
}

/// Blocking client for one daemon endpoint.
pub struct DaemonClient {
    base: String,
    agent: ureq::Agent,
}

impl DaemonClient {
    /// `base_url` like `http://buildbox:7373` (scheme required, no trailing
    /// slash needed).
    pub fn new(base_url: &str) -> Self {
        let agent = ureq::AgentBuilder::new()
            // Covers connection setup; reads get their own generous timeout
            // since a long-poll intentionally idles and results can be large.
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(120))
            .build();
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
        self.get(&format!(
            "{}/v1/jobs/{job_id}?wait_ms={wait_ms}",
            self.base
        ))
    }

    /// Requests cancellation. The job still runs to its next checkpoint and
    /// must be awaited for its (now likely `Cancelled`) outcome.
    pub fn cancel(&self, job_id: u64) -> Result<(), ClientError> {
        let response = self
            .agent
            .post(&format!("{}/v1/jobs/{job_id}/cancel", self.base))
            .call()
            .map_err(map_ureq_error)?;
        // Drain so the connection is reusable.
        let _ = response.into_reader().read_to_end(&mut Vec::new());
        Ok(())
    }

    /// Long-polls until the job finishes. `cancel_requested` is consulted
    /// between polls; the first time it reports true, cancellation is
    /// forwarded to the daemon and the wait continues until the daemon
    /// acknowledges with a final outcome.
    pub fn wait(
        &self,
        job_id: u64,
        cancel_requested: &dyn Fn() -> bool,
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
            debug_assert!(matches!(status.state, JobState::Queued | JobState::Running));
        }
    }

    /// Submit + wait in one call.
    pub fn run(
        &self,
        request: &JobRequest,
        cancel_requested: &dyn Fn() -> bool,
    ) -> Result<JobOutcome, ClientError> {
        let job_id = self.submit(request)?;
        self.wait(job_id, cancel_requested)
    }

    fn get<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, ClientError> {
        let response = self.agent.get(url).call().map_err(map_ureq_error)?;
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
            .set("Content-Type", "application/cbor")
            .send_bytes(&body)
            .map_err(map_ureq_error)?;
        decode_body(response)
    }
}

fn decode_body<T: serde::de::DeserializeOwned>(response: ureq::Response) -> Result<T, ClientError> {
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| ClientError::Transport(e.to_string()))?;
    from_cbor(&bytes).map_err(ClientError::Decode)
}

fn map_ureq_error(error: ureq::Error) -> ClientError {
    match error {
        ureq::Error::Status(status, response) => {
            let message = response
                .into_string()
                .unwrap_or_else(|_| "<unreadable error body>".to_string());
            ClientError::Api { status, message }
        }
        ureq::Error::Transport(t) => ClientError::Transport(t.to_string()),
    }
}
