use std::sync::Arc;

use glam::Mat4;

use crate::{
    Camera, LineData, LineStyle, MaterialId, MeshData, PointData, PointStyle, RenderSettings,
    SplatData, SplatStyle,
};

/// Data for a single frame's rendering.
#[derive(Clone, Default)]
pub struct SceneData {
    /// Meshes to render.
    pub meshes: Vec<(MeshData, Mat4, MaterialId)>,
    /// Lines to render.
    pub lines: Vec<(LineData, Mat4, LineStyle)>,
    /// Points to render.
    pub points: Vec<(PointData, Mat4, PointStyle)>,
    /// Gaussian splats to render (shared, since a splat is tens of
    /// megabytes and scenes are cloned).
    pub splats: Vec<(Arc<SplatData>, Mat4, SplatStyle)>,
}

impl SceneData {
    /// Create a new empty scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mesh to the scene.
    pub fn add_mesh(&mut self, mesh: MeshData, transform: Mat4, material: MaterialId) {
        self.meshes.push((mesh, transform, material));
    }

    /// Add lines to the scene.
    pub fn add_lines(&mut self, lines: LineData, transform: Mat4, style: LineStyle) {
        self.lines.push((lines, transform, style));
    }

    /// Add points to the scene.
    pub fn add_points(&mut self, points: PointData, transform: Mat4, style: PointStyle) {
        self.points.push((points, transform, style));
    }

    /// Add a splat to the scene.
    pub fn add_splat(&mut self, splat: Arc<SplatData>, transform: Mat4, style: SplatStyle) {
        self.splats.push((splat, transform, style));
    }

    /// Check if the scene is empty.
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
            && self.lines.is_empty()
            && self.points.is_empty()
            && self.splats.is_empty()
    }

    /// Clear all scene data.
    pub fn clear(&mut self) {
        self.meshes.clear();
        self.lines.clear();
        self.points.clear();
        self.splats.clear();
    }
}

/// Draw data for rendering a scene into a target viewport.
#[derive(Clone)]
pub struct SceneDrawData {
    /// Scene geometry to render.
    pub scene: SceneData,
    /// Camera for viewing.
    pub camera: Camera,
    /// Render settings.
    pub settings: RenderSettings,
    /// Viewport size in pixels.
    pub viewport_size: [u32; 2],
    /// Target texture format.
    pub target_format: wgpu::TextureFormat,
}
