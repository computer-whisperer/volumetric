//! Error and polling constants shared by the daemon clients (blocking
//! `client` and browser `web-client`).

/// How long one status request asks the daemon to hold before answering.
/// Short enough that a cancel request is forwarded promptly.
pub(crate) const POLL_WAIT_MS: u64 = 2_000;

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
