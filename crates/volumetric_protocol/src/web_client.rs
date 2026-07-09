//! Browser (fetch) client for a volumetric daemon.
//!
//! The async twin of [`crate::DaemonClient`]: same endpoints, same CBOR
//! bodies, same submit → long-poll → outcome state machine — but every
//! request is a browser `fetch`, awaited on the wasm event loop instead of
//! blocking a worker thread (wasm32 has none to block). Timeouts are the
//! browser's own; the long-poll returns within the daemon's `wait_ms` cap,
//! so requests never intentionally idle longer than that.

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::error::POLL_WAIT_MS;
use crate::{
    ClientError, DaemonInfo, JobOutcome, JobRequest, JobState, JobStatus, JobTicket,
    PROTOCOL_VERSION, from_cbor, to_cbor,
};

/// Fetch-based client for one daemon endpoint.
pub struct WebDaemonClient {
    base: String,
}

impl WebDaemonClient {
    /// `base_url` like `http://buildbox:7373` (scheme required, no trailing
    /// slash needed).
    pub fn new(base_url: &str) -> Self {
        Self {
            base: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Fetches daemon identity and verifies the protocol version matches.
    pub async fn info(&self) -> Result<DaemonInfo, ClientError> {
        let info: DaemonInfo = self.get(&format!("{}/v1/info", self.base)).await?;
        if info.protocol_version != PROTOCOL_VERSION {
            return Err(ClientError::VersionMismatch {
                daemon: info.protocol_version,
                client: PROTOCOL_VERSION,
            });
        }
        Ok(info)
    }

    /// Submits a job; returns its id.
    pub async fn submit(&self, request: &JobRequest) -> Result<u64, ClientError> {
        let ticket: JobTicket = self
            .post(&format!("{}/v1/jobs", self.base), Some(to_cbor(request)))
            .await?;
        Ok(ticket.job_id)
    }

    /// One status request, holding server-side up to `wait_ms` for the job
    /// to finish.
    pub async fn status(&self, job_id: u64, wait_ms: u64) -> Result<JobStatus, ClientError> {
        self.get(&format!(
            "{}/v1/jobs/{job_id}?wait_ms={wait_ms}",
            self.base
        ))
        .await
    }

    /// Requests cancellation. The job still runs to its next checkpoint and
    /// must be awaited for its (now likely `Cancelled`) outcome.
    pub async fn cancel(&self, job_id: u64) -> Result<(), ClientError> {
        let response = self
            .fetch("POST", &format!("{}/v1/jobs/{job_id}/cancel", self.base), None)
            .await?;
        check_status(&response).await?;
        Ok(())
    }

    /// Long-polls until the job finishes. `cancel_requested` is consulted
    /// between polls; the first time it reports true, cancellation is
    /// forwarded to the daemon and the wait continues until the daemon
    /// acknowledges with a final outcome. `on_progress` receives the job's
    /// latest [`crate::BuildProgress`] whenever a poll returns one — at the
    /// polling cadence, so an indicator rather than a stream.
    pub async fn wait(
        &self,
        job_id: u64,
        cancel_requested: &dyn Fn() -> bool,
        on_progress: &dyn Fn(crate::BuildProgress),
    ) -> Result<JobOutcome, ClientError> {
        let mut cancel_sent = false;
        loop {
            if !cancel_sent && cancel_requested() {
                self.cancel(job_id).await?;
                cancel_sent = true;
            }
            let status = self.status(job_id, POLL_WAIT_MS).await?;
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
    pub async fn run(
        &self,
        request: &JobRequest,
        cancel_requested: &dyn Fn() -> bool,
        on_progress: &dyn Fn(crate::BuildProgress),
    ) -> Result<JobOutcome, ClientError> {
        let job_id = self.submit(request).await?;
        self.wait(job_id, cancel_requested, on_progress).await
    }

    async fn get<T: serde::de::DeserializeOwned>(&self, url: &str) -> Result<T, ClientError> {
        let response = self.fetch("GET", url, None).await?;
        decode_body(response).await
    }

    async fn post<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        body: Option<Vec<u8>>,
    ) -> Result<T, ClientError> {
        let response = self.fetch("POST", url, body).await?;
        decode_body(response).await
    }

    /// One fetch, resolved to a `Response`. Network-level failures (daemon
    /// down, CORS rejection, bad URL) surface as `Transport`.
    async fn fetch(
        &self,
        method: &str,
        url: &str,
        body: Option<Vec<u8>>,
    ) -> Result<web_sys::Response, ClientError> {
        let has_body = body.is_some();
        let init = web_sys::RequestInit::new();
        init.set_method(method);
        if let Some(body) = body {
            let js_body: wasm_bindgen::JsValue = js_sys::Uint8Array::from(body.as_slice()).into();
            init.set_body(&js_body);
        }
        let request = web_sys::Request::new_with_str_and_init(url, &init)
            .map_err(|e| ClientError::Transport(js_error_text(&e)))?;
        if has_body {
            request
                .headers()
                .set("Content-Type", "application/cbor")
                .map_err(|e| ClientError::Transport(js_error_text(&e)))?;
        }
        let window = web_sys::window()
            .ok_or_else(|| ClientError::Transport("no browser window".to_string()))?;
        let response = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| ClientError::Transport(js_error_text(&e)))?;
        response
            .dyn_into::<web_sys::Response>()
            .map_err(|_| ClientError::Transport("fetch resolved to a non-Response".to_string()))
    }
}

async fn check_status(response: &web_sys::Response) -> Result<(), ClientError> {
    if response.ok() {
        return Ok(());
    }
    let message = match response.text() {
        Ok(promise) => JsFuture::from(promise)
            .await
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_else(|| "<unreadable error body>".to_string()),
        Err(_) => "<unreadable error body>".to_string(),
    };
    Err(ClientError::Api {
        status: response.status(),
        message,
    })
}

async fn decode_body<T: serde::de::DeserializeOwned>(
    response: web_sys::Response,
) -> Result<T, ClientError> {
    check_status(&response).await?;
    let buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|e| ClientError::Transport(js_error_text(&e)))?,
    )
    .await
    .map_err(|e| ClientError::Transport(js_error_text(&e)))?;
    let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
    from_cbor(&bytes).map_err(ClientError::Decode)
}

fn js_error_text(error: &wasm_bindgen::JsValue) -> String {
    error
        .as_string()
        .or_else(|| {
            error
                .dyn_ref::<js_sys::Error>()
                .map(|e| e.message().as_string().unwrap_or_default())
        })
        .unwrap_or_else(|| format!("{error:?}"))
}
