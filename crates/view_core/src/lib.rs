//! Native tooling around the [`ViewSet`](volumetric_abi::viewset::ViewSet)
//! value: importing posed-image datasets from their manifests and stills
//! straight from a camera, decoding the images a set carries, comparing a
//! model against a view's depth map, and compositing a render over its
//! photograph.
//!
//! The value itself and its camera math live in `volumetric_abi::viewset`
//! so operators share them; this crate is what the CLI and the GUI use
//! around it.

pub mod image;
pub mod manifest;
pub mod overlay;
pub mod residual;
pub mod stills;
pub mod text;

pub mod crop;
pub mod detect;
pub mod measure;
pub mod subset;

pub use manifest::{
    Eye, ImportReport, Labels, ManifestKind, Selection, import_manifest, select_views,
};
pub use stills::{
    Embed, StillsOptions, StillsReport, embed_pictures, full_picture, import_stills, preview_jpeg,
};
