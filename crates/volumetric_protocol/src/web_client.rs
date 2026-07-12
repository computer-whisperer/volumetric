//! Browser (fetch) client for a volumetric daemon.
//!
//! The async twin of [`crate::DaemonClient`]: same endpoints, same CBOR
//! bodies, same submit → long-poll → outcome state machine — but every
//! request is a browser `fetch`, awaited on the wasm event loop instead of
//! blocking a worker thread (wasm32 has none to block). Every request runs
//! under an abort timer matching the native client's read timeout; the
//! long-poll itself returns within the daemon's `wait_ms` cap, so requests
//! never intentionally idle longer than that.

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::error::POLL_WAIT_MS;
use crate::{
    ClientError, DaemonInfo, JobOutcome, JobRequest, JobState, JobStatus, JobTicket,
    PROTOCOL_VERSION, from_cbor, to_cbor,
};

/// Abort deadline per request, headers through body — the fetch analogue
/// of the native client's 120s read timeout. Generous because results can
/// be large; the long-poll returns within ~2s regardless.
const FETCH_TIMEOUT_MS: i32 = 120_000;

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
        self.get(&format!("{}/v1/jobs/{job_id}?wait_ms={wait_ms}", self.base))
            .await
    }

    /// Requests cancellation. The job still runs to its next checkpoint and
    /// must be awaited for its (now likely `Cancelled`) outcome.
    pub async fn cancel(&self, job_id: u64) -> Result<(), ClientError> {
        let (status, bytes) = self
            .fetch(
                "POST",
                &format!("{}/v1/jobs/{job_id}/cancel", self.base),
                None,
            )
            .await?;
        check_status((status, &bytes))
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
        decode_body(self.fetch("GET", url, None).await?)
    }

    async fn post<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        body: Option<Vec<u8>>,
    ) -> Result<T, ClientError> {
        decode_body(self.fetch("POST", url, body).await?)
    }

    /// One fetch, run to a fully-read body under the timeout. Network-level
    /// failures (daemon down, CORS rejection, bad URL, timeout) surface as
    /// `Transport`.
    async fn fetch(
        &self,
        method: &str,
        url: &str,
        body: Option<Vec<u8>>,
    ) -> Result<(u16, Vec<u8>), ClientError> {
        let transport = |e: &wasm_bindgen::JsValue| ClientError::Transport(js_error_text(e));

        let has_body = body.is_some();
        let init = web_sys::RequestInit::new();
        init.set_method(method);
        if let Some(body) = body {
            let js_body: wasm_bindgen::JsValue = js_sys::Uint8Array::from(body.as_slice()).into();
            init.set_body(&js_body);
        }

        // Browsers impose no fetch deadline of their own: without an abort
        // timer, a half-open connection (network path change mid long-poll)
        // suspends this future forever — and with it the web shell's whole
        // job pump. Same ceiling as the native client's read timeout. The
        // signal covers body streaming too, so the timer is cleared only
        // after the body is fully read.
        let controller = web_sys::AbortController::new().map_err(|e| transport(&e))?;
        init.set_signal(Some(&controller.signal()));

        let request =
            web_sys::Request::new_with_str_and_init(url, &init).map_err(|e| transport(&e))?;
        if has_body {
            request
                .headers()
                .set("Content-Type", "application/cbor")
                .map_err(|e| transport(&e))?;
        }
        let window = web_sys::window()
            .ok_or_else(|| ClientError::Transport("no browser window".to_string()))?;
        let abort = controller.clone();
        let on_timeout = wasm_bindgen::closure::Closure::once_into_js(move || abort.abort());
        let timer = window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                on_timeout.unchecked_ref(),
                FETCH_TIMEOUT_MS,
            )
            .map_err(|e| transport(&e))?;

        let result = async {
            let response = JsFuture::from(window.fetch_with_request(&request))
                .await
                .map_err(|e| transport(&e))?;
            let response = response.dyn_into::<web_sys::Response>().map_err(|_| {
                ClientError::Transport("fetch resolved to a non-Response".to_string())
            })?;
            let status = response.status();
            let buffer = JsFuture::from(response.array_buffer().map_err(|e| transport(&e))?)
                .await
                .map_err(|e| transport(&e))?;
            Ok((status, js_sys::Uint8Array::new(&buffer).to_vec()))
        }
        .await;
        window.clear_timeout_with_handle(timer);

        if controller.signal().aborted() {
            return Err(ClientError::Transport(format!(
                "no response within {}s",
                FETCH_TIMEOUT_MS / 1000
            )));
        }
        result
    }
}

fn decode_body<T: serde::de::DeserializeOwned>(
    (status, bytes): (u16, Vec<u8>),
) -> Result<T, ClientError> {
    check_status((status, &bytes))?;
    from_cbor(&bytes).map_err(ClientError::Decode)
}

fn check_status((status, bytes): (u16, &[u8])) -> Result<(), ClientError> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    Err(ClientError::Api {
        status,
        message: String::from_utf8_lossy(bytes).into_owned(),
    })
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
