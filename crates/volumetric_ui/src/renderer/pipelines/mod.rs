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
pub use line::LinePipeline;
pub use mesh::{MeshPipeline, MeshUniforms};
pub use point::PointPipeline;
pub use ssao::{SsaoPipeline, SsaoUniforms};
