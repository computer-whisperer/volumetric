//! GPU pipelines, one per kind of pass.

mod fullscreen;
mod line;
mod mesh;
mod point;
mod splat;
mod warp;

pub(crate) use fullscreen::{
    AoUniforms, FullscreenPass, PickUniforms, ResolveUniforms, ao_pass, pick_pass, resolve_pass,
};
pub use line::{GpuLines, LinePipeline};
pub use mesh::{GpuMesh, MeshDraw, MeshPipeline};
pub use point::{GpuPointInstance, GpuPoints, PointPipeline};
pub use splat::{GpuSplat, SplatCompositePipeline, SplatPipeline, evaluate_sh, project_covariance};
pub use warp::{GpuWarp, Warp, WarpPipeline};
