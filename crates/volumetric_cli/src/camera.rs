//! Camera setup and auto-framing for headless rendering

use glam::{Mat4, Vec3};

/// Projection type for rendering
#[derive(Debug, Clone, Copy)]
pub enum ProjectionType {
    /// Perspective projection with field of view in radians
    Perspective { fov_y: f32 },
    /// Orthographic projection with vertical scale in world units
    Orthographic { scale: f32 },
}

impl Default for ProjectionType {
    fn default() -> Self {
        ProjectionType::Perspective {
            fov_y: std::f32::consts::FRAC_PI_4,
        }
    }
}

/// Camera parameters for rendering
pub struct CameraSetup {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub projection: ProjectionType,
    pub near: f32,
    pub far: f32,
}

impl CameraSetup {
    /// Create a view-projection matrix for the given aspect ratio
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.position, self.target, self.up);
        let proj = match self.projection {
            ProjectionType::Perspective { fov_y } => {
                Mat4::perspective_rh(fov_y, aspect, self.near, self.far)
            }
            ProjectionType::Orthographic { scale } => {
                let half_height = scale * 0.5;
                let half_width = half_height * aspect;
                Mat4::orthographic_rh(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    self.near,
                    self.far,
                )
            }
        };
        proj * view
    }

    /// Auto-frame camera to fit model bounds with padding
    pub fn auto_frame(
        bounds_min: Vec3,
        bounds_max: Vec3,
        view_angle: ViewAngle,
        projection: ProjectionType,
    ) -> Self {
        let center = (bounds_min + bounds_max) * 0.5;
        let size = bounds_max - bounds_min;
        let max_dim = size.x.max(size.y).max(size.z);
        let padding = 1.2;

        let (direction, up) = view_angle.direction_and_up();

        let (distance, projection) = match projection {
            ProjectionType::Perspective { fov_y } => {
                // Distance to fit model in view with padding
                let dist = (max_dim * padding) / (fov_y * 0.5).tan();
                (dist, ProjectionType::Perspective { fov_y })
            }
            ProjectionType::Orthographic { scale } => {
                // Use provided scale or auto-compute from bounds
                let ortho_scale = if scale <= 0.0 {
                    max_dim * padding
                } else {
                    scale
                };
                // For orthographic, distance doesn't affect size, but we need
                // enough distance to see the model properly
                let dist = max_dim * 2.0;
                (dist, ProjectionType::Orthographic { scale: ortho_scale })
            }
        };

        let position = center - direction * distance;

        // Near/far based on scene size
        let near = distance * 0.01;
        let far = distance * 10.0;

        CameraSetup {
            position,
            target: center,
            up,
            projection,
            near,
            far,
        }
    }

    /// Create camera from explicit pose (position, target, up)
    pub fn from_pose(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        projection: ProjectionType,
        bounds_min: Vec3,
        bounds_max: Vec3,
    ) -> Self {
        let size = bounds_max - bounds_min;
        let max_dim = size.x.max(size.y).max(size.z);
        let distance = (position - target).length();

        // Auto-compute orthographic scale if not specified
        let projection = match projection {
            ProjectionType::Orthographic { scale } if scale <= 0.0 => {
                ProjectionType::Orthographic {
                    scale: max_dim * 1.2,
                }
            }
            other => other,
        };

        // Near/far based on distance and scene size
        let near = (distance * 0.01).min(0.1);
        let far = distance + max_dim * 5.0;

        CameraSetup {
            position,
            target,
            up,
            projection,
            near,
            far,
        }
    }
}

/// Predefined view angles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewAngle {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
    Iso,
    IsoBack,
}

impl ViewAngle {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "front" => Some(ViewAngle::Front),
            "back" => Some(ViewAngle::Back),
            "left" => Some(ViewAngle::Left),
            "right" => Some(ViewAngle::Right),
            "top" => Some(ViewAngle::Top),
            "bottom" => Some(ViewAngle::Bottom),
            "iso" => Some(ViewAngle::Iso),
            "iso-back" | "isoback" => Some(ViewAngle::IsoBack),
            _ => None,
        }
    }

    /// Get all standard views
    pub fn all() -> Vec<ViewAngle> {
        vec![
            ViewAngle::Front,
            ViewAngle::Back,
            ViewAngle::Left,
            ViewAngle::Right,
            ViewAngle::Top,
            ViewAngle::Bottom,
            ViewAngle::Iso,
            ViewAngle::IsoBack,
        ]
    }

    /// Get camera direction (towards target) and up vector
    pub fn direction_and_up(&self) -> (Vec3, Vec3) {
        match self {
            ViewAngle::Front => (Vec3::NEG_Z, Vec3::Y),
            ViewAngle::Back => (Vec3::Z, Vec3::Y),
            ViewAngle::Left => (Vec3::NEG_X, Vec3::Y),
            ViewAngle::Right => (Vec3::X, Vec3::Y),
            ViewAngle::Top => (Vec3::NEG_Y, Vec3::Z),
            ViewAngle::Bottom => (Vec3::Y, Vec3::NEG_Z),
            ViewAngle::Iso => {
                // Isometric: 45° azimuth, ~35° elevation
                let dir = Vec3::new(1.0, -1.0, 1.0).normalize();
                (dir, Vec3::Y)
            }
            ViewAngle::IsoBack => {
                // Isometric from back
                let dir = Vec3::new(-1.0, -1.0, -1.0).normalize();
                (dir, Vec3::Y)
            }
        }
    }

    /// Get suffix for filename
    pub fn suffix(&self) -> &'static str {
        match self {
            ViewAngle::Front => "front",
            ViewAngle::Back => "back",
            ViewAngle::Left => "left",
            ViewAngle::Right => "right",
            ViewAngle::Top => "top",
            ViewAngle::Bottom => "bottom",
            ViewAngle::Iso => "iso",
            ViewAngle::IsoBack => "iso-back",
        }
    }
}

/// Parse view angles from comma-separated string
pub fn parse_views(views_str: &str) -> Vec<ViewAngle> {
    if views_str.to_lowercase() == "all" {
        return ViewAngle::all();
    }

    views_str
        .split(',')
        .filter_map(|s| ViewAngle::from_str(s.trim()))
        .collect()
}
