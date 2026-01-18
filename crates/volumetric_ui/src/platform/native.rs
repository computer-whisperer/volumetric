//! Native platform implementation using std::thread and rfd.

use super::{FileError, FileFilter, FileResult, PickedFile};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

// =============================================================================
// Background Worker (Thread-based)
// =============================================================================

/// A cancellation token that can be shared between the UI and background worker.
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

/// State of the background worker
pub struct BackgroundWorker<Task, Result> {
    /// Channel to send tasks to the worker
    task_sender: Sender<(Task, CancellationToken)>,
    /// Channel to receive results from the worker
    result_receiver: Receiver<Result>,
    /// Handle to the worker thread
    _thread_handle: JoinHandle<()>,
}

impl<Task, Result> BackgroundWorker<Task, Result>
where
    Task: Send + 'static,
    Result: Send + 'static,
{
    pub fn new<F>(worker_fn: F) -> Self
    where
        F: FnOnce(Receiver<(Task, CancellationToken)>, Sender<Result>) + Send + 'static,
    {
        let (task_sender, task_receiver) = mpsc::channel::<(Task, CancellationToken)>();
        let (result_sender, result_receiver) = mpsc::channel::<Result>();

        let thread_handle = thread::spawn(move || {
            worker_fn(task_receiver, result_sender);
        });

        Self {
            task_sender,
            result_receiver,
            _thread_handle: thread_handle,
        }
    }

    pub fn send_task(&self, task: Task, cancel_token: CancellationToken) {
        // Ignore send errors - if the worker is dead, we'll notice when we try to receive
        let _ = self.task_sender.send((task, cancel_token));
    }

    pub fn try_recv_result(&self) -> Option<Result> {
        self.result_receiver.try_recv().ok()
    }
}

// =============================================================================
// File I/O Operations
// =============================================================================

/// Pick a file using the system file dialog (blocking)
pub fn pick_file(filter: &FileFilter) -> FileResult<PickedFile> {
    let path = rfd::FileDialog::new()
        .add_filter(filter.name, filter.extensions)
        .pick_file()
        .ok_or(FileError::Cancelled)?;

    let data = std::fs::read(&path)?;
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("file")
        .to_string();

    Ok(PickedFile { name, data })
}

/// Pick a file and return the path (for project files that need path persistence)
pub fn pick_file_with_path(filter: &FileFilter) -> FileResult<(PathBuf, Vec<u8>)> {
    let path = rfd::FileDialog::new()
        .add_filter(filter.name, filter.extensions)
        .pick_file()
        .ok_or(FileError::Cancelled)?;

    let data = std::fs::read(&path)?;
    Ok((path, data))
}

/// Save a file using the system file dialog (blocking)
pub fn save_file(filter: &FileFilter, suggested_name: &str, data: &[u8]) -> FileResult<()> {
    let path = rfd::FileDialog::new()
        .add_filter(filter.name, filter.extensions)
        .set_file_name(suggested_name)
        .save_file()
        .ok_or(FileError::Cancelled)?;

    std::fs::write(&path, data)?;
    Ok(())
}

/// Save a file and return the chosen path
pub fn save_file_with_path(
    filter: &FileFilter,
    suggested_name: &str,
    data: &[u8],
) -> FileResult<PathBuf> {
    let path = rfd::FileDialog::new()
        .add_filter(filter.name, filter.extensions)
        .set_file_name(suggested_name)
        .save_file()
        .ok_or(FileError::Cancelled)?;

    std::fs::write(&path, data)?;
    Ok(path)
}

/// Read a file from the filesystem (native only)
pub fn read_file(path: &std::path::Path) -> FileResult<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

/// Check if a file exists (native only)
pub fn file_exists(path: &std::path::Path) -> bool {
    std::fs::metadata(path).is_ok()
}
