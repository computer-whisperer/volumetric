//! Test scenes for smoke testing the renderer.
//!
//! Provides sample geometry for verifying the renderer works correctly.

#![allow(dead_code)]

use super::{
    Camera, DepthMode, LineData, LineSegment, LineStyle, MaterialId, MeshData, MeshVertex,
    PointData, PointInstance, PointShape, PointStyle, RenderSettings, SceneData, WidthMode,
};
use glam::{Mat4, Vec3};

/// Create a simple cube mesh for testing.
pub fn create_test_cube(size: f32) -> MeshData {
    let h = size * 0.5;

    // Cube vertices with normals
    let vertices = vec![
        // Front face (+Z)
        MeshVertex::new([-h, -h, h], [0.0, 0.0, 1.0]),
        MeshVertex::new([h, -h, h], [0.0, 0.0, 1.0]),
        MeshVertex::new([h, h, h], [0.0, 0.0, 1.0]),
        MeshVertex::new([-h, h, h], [0.0, 0.0, 1.0]),
        // Back face (-Z)
        MeshVertex::new([h, -h, -h], [0.0, 0.0, -1.0]),
        MeshVertex::new([-h, -h, -h], [0.0, 0.0, -1.0]),
        MeshVertex::new([-h, h, -h], [0.0, 0.0, -1.0]),
        MeshVertex::new([h, h, -h], [0.0, 0.0, -1.0]),
        // Top face (+Y)
        MeshVertex::new([-h, h, h], [0.0, 1.0, 0.0]),
        MeshVertex::new([h, h, h], [0.0, 1.0, 0.0]),
        MeshVertex::new([h, h, -h], [0.0, 1.0, 0.0]),
        MeshVertex::new([-h, h, -h], [0.0, 1.0, 0.0]),
        // Bottom face (-Y)
        MeshVertex::new([-h, -h, -h], [0.0, -1.0, 0.0]),
        MeshVertex::new([h, -h, -h], [0.0, -1.0, 0.0]),
        MeshVertex::new([h, -h, h], [0.0, -1.0, 0.0]),
        MeshVertex::new([-h, -h, h], [0.0, -1.0, 0.0]),
        // Right face (+X)
        MeshVertex::new([h, -h, h], [1.0, 0.0, 0.0]),
        MeshVertex::new([h, -h, -h], [1.0, 0.0, 0.0]),
        MeshVertex::new([h, h, -h], [1.0, 0.0, 0.0]),
        MeshVertex::new([h, h, h], [1.0, 0.0, 0.0]),
        // Left face (-X)
        MeshVertex::new([-h, -h, -h], [-1.0, 0.0, 0.0]),
        MeshVertex::new([-h, -h, h], [-1.0, 0.0, 0.0]),
        MeshVertex::new([-h, h, h], [-1.0, 0.0, 0.0]),
        MeshVertex::new([-h, h, -h], [-1.0, 0.0, 0.0]),
    ];

    let indices = vec![
        0, 1, 2, 2, 3, 0,       // Front
        4, 5, 6, 6, 7, 4,       // Back
        8, 9, 10, 10, 11, 8,    // Top
        12, 13, 14, 14, 15, 12, // Bottom
        16, 17, 18, 18, 19, 16, // Right
        20, 21, 22, 22, 23, 20, // Left
    ];

    MeshData {
        vertices,
        indices: Some(indices),
    }
}

/// Create test lines forming a 3D cross/axis at origin.
pub fn create_test_axes(length: f32) -> LineData {
    LineData {
        segments: vec![
            // X axis (red)
            LineSegment {
                start: [0.0, 0.0, 0.0],
                end: [length, 0.0, 0.0],
                color: [1.0, 0.2, 0.2, 1.0],
            },
            // Y axis (green)
            LineSegment {
                start: [0.0, 0.0, 0.0],
                end: [0.0, length, 0.0],
                color: [0.2, 1.0, 0.2, 1.0],
            },
            // Z axis (blue)
            LineSegment {
                start: [0.0, 0.0, 0.0],
                end: [0.0, 0.0, length],
                color: [0.2, 0.2, 1.0, 1.0],
            },
        ],
    }
}

/// Create a wireframe box as lines.
pub fn create_wireframe_box(min: Vec3, max: Vec3, color: [f32; 4]) -> LineData {
    let corners = [
        [min.x, min.y, min.z],
        [max.x, min.y, min.z],
        [max.x, max.y, min.z],
        [min.x, max.y, min.z],
        [min.x, min.y, max.z],
        [max.x, min.y, max.z],
        [max.x, max.y, max.z],
        [min.x, max.y, max.z],
    ];

    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4), // Top face
        (0, 4), (1, 5), (2, 6), (3, 7), // Vertical edges
    ];

    LineData {
        segments: edges
            .iter()
            .map(|&(a, b)| LineSegment {
                start: corners[a],
                end: corners[b],
                color,
            })
            .collect(),
    }
}

/// Create a set of random points for testing.
pub fn create_test_points(count: usize, bounds: f32) -> PointData {
    use std::f32::consts::PI;

    let mut points = Vec::with_capacity(count);
    for i in 0..count {
        // Generate somewhat random but deterministic positions
        let t = i as f32 / count as f32;
        let phi = t * PI * 20.0;
        let theta = t * PI * 2.0 * 7.0;
        let r = bounds * (0.3 + 0.7 * t);

        let x = r * phi.sin() * theta.cos();
        let y = r * phi.cos();
        let z = r * phi.sin() * theta.sin();

        // Color based on position
        let color = [
            (x / bounds + 1.0) * 0.5,
            (y / bounds + 1.0) * 0.5,
            (z / bounds + 1.0) * 0.5,
            1.0,
        ];

        points.push(PointInstance {
            position: [x, y, z],
            color,
        });
    }

    PointData { points }
}

/// Create a complete test scene with mesh, lines, and points.
pub fn create_test_scene() -> SceneData {
    let mut scene = SceneData::new();

    // Add a cube
    scene.add_mesh(
        create_test_cube(1.0),
        Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)),
        MaterialId(0),
    );

    // Add coordinate axes
    scene.add_lines(
        create_test_axes(2.0),
        Mat4::IDENTITY,
        LineStyle {
            width: 2.0,
            width_mode: WidthMode::ScreenSpace,
            pattern: super::LinePattern::Solid,
            depth_mode: DepthMode::Normal,
        },
    );

    // Add a wireframe box
    scene.add_lines(
        create_wireframe_box(Vec3::new(-1.5, 0.0, -1.5), Vec3::new(1.5, 2.0, 1.5), [0.5, 0.5, 0.5, 0.5]),
        Mat4::IDENTITY,
        LineStyle {
            width: 1.0,
            width_mode: WidthMode::ScreenSpace,
            pattern: super::LinePattern::Dashed {
                dash_length: 0.2,
                gap_length: 0.1,
            },
            depth_mode: DepthMode::Normal,
        },
    );

    // Add some test points
    scene.add_points(
        create_test_points(100, 2.0),
        Mat4::IDENTITY,
        PointStyle {
            size: 6.0,
            size_mode: WidthMode::ScreenSpace,
            shape: PointShape::Circle,
            depth_mode: DepthMode::Normal,
        },
    );

    scene
}

/// Create a default camera positioned to view the test scene.
pub fn create_test_camera() -> Camera {
    Camera {
        target: Vec3::new(0.0, 0.5, 0.0),
        radius: 5.0,
        theta: std::f32::consts::FRAC_PI_4,
        phi: std::f32::consts::FRAC_PI_4,
        fov_y: std::f32::consts::FRAC_PI_3,
        near: 0.1,
        far: 100.0,
    }
}

/// Create default render settings for testing.
pub fn create_test_settings() -> RenderSettings {
    RenderSettings {
        ssao_enabled: true,
        ssao_samples: 16,
        ssao_radius: 0.5,
        ssao_bias: 0.025,
        ssao_strength: 1.0,
        grid: super::GridSettings::default(),
        axis_indicator: super::AxisIndicator::default(),
        show_axis_indicator: true,
        background_color: [0.1, 0.1, 0.12, 1.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_cube() {
        let cube = create_test_cube(2.0);
        assert_eq!(cube.vertices.len(), 24); // 6 faces * 4 vertices
        assert!(cube.indices.is_some());
        assert_eq!(cube.indices.as_ref().unwrap().len(), 36); // 6 faces * 2 triangles * 3 indices
    }

    #[test]
    fn test_create_axes() {
        let axes = create_test_axes(1.0);
        assert_eq!(axes.segments.len(), 3); // X, Y, Z
    }

    #[test]
    fn test_create_wireframe_box() {
        let bbox = create_wireframe_box(Vec3::ZERO, Vec3::ONE, [1.0; 4]);
        assert_eq!(bbox.segments.len(), 12); // 12 edges
    }

    #[test]
    fn test_create_points() {
        let points = create_test_points(50, 1.0);
        assert_eq!(points.points.len(), 50);
    }

    #[test]
    fn test_create_scene() {
        let scene = create_test_scene();
        assert!(!scene.is_empty());
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.lines.len(), 2);
        assert_eq!(scene.points.len(), 1);
    }
}
