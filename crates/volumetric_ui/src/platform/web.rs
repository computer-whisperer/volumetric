//! Web platform implementation using poll-promise and web-sys.

use super::{FileError, FileFilter, FileResult, PickedFile};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wasm_bindgen::JsCast;

// =============================================================================
// Background Worker (Poll-Promise based)
// =============================================================================

/// A cancellation token that can be shared between the UI and background task.
#[derive(Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Background worker state for web (uses poll-promise)
///
/// On web, we don't have real threads. Instead, we execute tasks synchronously
/// within the event loop using poll-promise. This may cause brief UI freezes
/// for long-running operations, but it's the simplest approach without Web Workers.
pub struct BackgroundWorker<Task, Result> {
    pending_task: Option<(Task, CancellationToken)>,
    pending_result: Option<Result>,
    worker_fn: Box<dyn Fn(Task, &CancellationToken) -> Result>,
}

impl<Task, Result> BackgroundWorker<Task, Result>
where
    Task: 'static,
    Result: 'static,
{
    /// Create a new background worker with a synchronous task executor.
    ///
    /// Note: On web, the `_receiver_setup` function is not used. Instead,
    /// we need a direct task executor function.
    pub fn new_with_executor<F>(executor: F) -> Self
    where
        F: Fn(Task, &CancellationToken) -> Result + 'static,
    {
        Self {
            pending_task: None,
            pending_result: None,
            worker_fn: Box::new(executor),
        }
    }

    pub fn send_task(&mut self, task: Task, cancel_token: CancellationToken) {
        self.pending_task = Some((task, cancel_token));
    }

    /// Poll for results. On web, this executes the task synchronously if one is pending.
    pub fn try_recv_result(&mut self) -> Option<Result> {
        // First check if we have a buffered result
        if self.pending_result.is_some() {
            return self.pending_result.take();
        }

        // Execute any pending task synchronously
        if let Some((task, cancel_token)) = self.pending_task.take() {
            let result = (self.worker_fn)(task, &cancel_token);
            return Some(result);
        }

        None
    }
}

// =============================================================================
// File I/O Operations (Async with web dialogs)
// =============================================================================

/// Pending file pick operation
pub struct PendingFilePick {
    promise: poll_promise::Promise<Option<PickedFile>>,
}

impl PendingFilePick {
    /// Poll the promise for completion
    pub fn poll(&self) -> Option<Option<PickedFile>> {
        self.promise.ready().cloned()
    }
}

/// Start picking a file (async on web)
pub fn pick_file_async(filter: &FileFilter) -> PendingFilePick {
    let filter_name = filter.name.to_string();
    let filter_exts: Vec<String> = filter.extensions.iter().map(|s| s.to_string()).collect();

    let promise = poll_promise::Promise::spawn_local(async move {
        let file = rfd::AsyncFileDialog::new()
            .add_filter(&filter_name, &filter_exts.iter().map(|s| s.as_str()).collect::<Vec<_>>())
            .pick_file()
            .await?;

        let data = file.read().await;
        let name = file.file_name();

        Some(PickedFile { name, data })
    });

    PendingFilePick { promise }
}

/// Pick a file (blocking stub for web - prefer pick_file_async)
pub fn pick_file(_filter: &FileFilter) -> FileResult<PickedFile> {
    Err(FileError::NotSupported)
}

/// Pick a file and return path (not supported on web)
pub fn pick_file_with_path(_filter: &FileFilter) -> FileResult<(std::path::PathBuf, Vec<u8>)> {
    Err(FileError::NotSupported)
}

/// Save a file by triggering a browser download
pub fn save_file(_filter: &FileFilter, filename: &str, data: &[u8]) -> FileResult<()> {
    trigger_download(filename, data)
}

/// Save a file with path (not applicable on web, just triggers download)
pub fn save_file_with_path(
    _filter: &FileFilter,
    filename: &str,
    data: &[u8],
) -> FileResult<std::path::PathBuf> {
    trigger_download(filename, data)?;
    // Return a dummy path since web doesn't have real paths
    Ok(std::path::PathBuf::from(filename))
}

/// Trigger a browser download for the given data
fn trigger_download(filename: &str, data: &[u8]) -> FileResult<()> {
    let window = web_sys::window().ok_or_else(|| FileError::Io("No window".to_string()))?;
    let document = window
        .document()
        .ok_or_else(|| FileError::Io("No document".to_string()))?;

    // Create a Uint8Array from the data
    let uint8_array = js_sys::Uint8Array::new_with_length(data.len() as u32);
    uint8_array.copy_from(data);

    // Create a Blob from the array
    let array = js_sys::Array::new();
    array.push(&uint8_array.buffer());

    let blob = web_sys::Blob::new_with_u8_array_sequence(&array)
        .map_err(|_| FileError::Io("Failed to create blob".to_string()))?;

    // Create object URL
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|_| FileError::Io("Failed to create object URL".to_string()))?;

    // Create and click a download link
    let a = document
        .create_element("a")
        .map_err(|_| FileError::Io("Failed to create element".to_string()))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| FileError::Io("Failed to cast to anchor".to_string()))?;

    a.set_href(&url);
    a.set_download(filename);
    a.click();

    // Revoke the URL to free memory
    let _ = web_sys::Url::revoke_object_url(&url);

    Ok(())
}

/// Read a file from filesystem (not supported on web)
pub fn read_file(_path: &std::path::Path) -> FileResult<Vec<u8>> {
    Err(FileError::NotSupported)
}

/// Check if a file exists (always returns false on web)
pub fn file_exists(_path: &std::path::Path) -> bool {
    false
}

// =============================================================================
// Demo Models (Embedded for web)
// =============================================================================

/// Get an embedded demo model by name.
/// On web, demo models must be embedded at compile time or fetched from a server.
pub fn get_embedded_demo_model(name: &str) -> Option<&'static [u8]> {
    match name {
        // These would be populated if we embed models at compile time
        // "simple_sphere_model" => Some(include_bytes!("../../../models/simple_sphere_model.wasm")),
        _ => None,
    }
}
