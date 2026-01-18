//! Render pipeline implementations.
//!
//! This module contains the GPU pipeline implementations for each render pass:
//! - Mesh G-buffer rendering
//! - SSAO computation
//! - Final compositing
//! - Line rendering
//! - Point rendering

mod composite;
mod line;
mod mesh;
mod point;
mod ssao;

pub use composite::CompositePipeline;
pub use line::{LinePipeline, LineUniforms};
pub use mesh::{MeshPipeline, MeshUniforms};
pub use point::{GpuPointInstance, PointPipeline, PointUniforms};
pub use ssao::{SsaoPipeline, SsaoUniforms};
