//! Render pipeline implementations.
//!
//! This module contains the GPU pipeline implementations for each render pass:
//! - Mesh G-buffer rendering
//! - SSAO computation
//! - Final compositing
//! - Line rendering
//! - Point rendering

#![allow(dead_code)]

mod composite;
mod line;
mod mesh;
mod point;
mod ssao;

pub use composite::CompositePipeline;
pub use line::{GpuLines, LinePipeline};
pub use mesh::{GpuMesh, MeshPipeline, MeshUniforms};
pub use point::{GpuPoints, PointPipeline};
pub use ssao::{SsaoPipeline, SsaoUniforms};
