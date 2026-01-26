//! Platform abstraction layer for native/web compatibility.
//!
//! This module provides abstractions for:
//! - Background task execution
//! - File I/O operations (open/save dialogs, file downloads)
//!
//! Note: This module is currently unused scaffolding - main.rs has its own implementations.

#![allow(dead_code)]

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(not(target_arch = "wasm32"))]
pub use native::save_file;
#[cfg(target_arch = "wasm32")]
pub use web::save_file;

/// Result type alias for file operations
pub type FileResult<T> = Result<T, FileError>;

/// Errors that can occur during file operations
#[derive(Debug, thiserror::Error)]
pub enum FileError {
    #[error("User cancelled the operation")]
    Cancelled,
    #[error("I/O error: {0}")]
    Io(String),
    #[error("Operation not supported on this platform")]
    NotSupported,
}

impl From<std::io::Error> for FileError {
    fn from(e: std::io::Error) -> Self {
        FileError::Io(e.to_string())
    }
}

/// A file that was picked by the user
#[derive(Clone)]
pub struct PickedFile {
    /// The file name (without path)
    pub name: String,
    /// The file contents
    pub data: Vec<u8>,
}

/// Filter for file dialogs
pub struct FileFilter {
    pub name: &'static str,
    pub extensions: &'static [&'static str],
}

impl FileFilter {
    pub const WASM: FileFilter = FileFilter {
        name: "WASM",
        extensions: &["wasm"],
    };

    pub const PROJECT: FileFilter = FileFilter {
        name: "Project",
        extensions: &["vproj"],
    };

    pub const STL: FileFilter = FileFilter {
        name: "STL",
        extensions: &["stl"],
    };

    pub const PNG: FileFilter = FileFilter {
        name: "PNG Image",
        extensions: &["png"],
    };
}
