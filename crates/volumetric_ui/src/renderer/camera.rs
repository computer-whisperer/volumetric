//! Camera system with orbit, pan, and zoom controls.
//!
//! Uses spherical coordinates for intuitive 3D navigation around a target point.

#![allow(dead_code)]

use glam::{Mat4, Vec2, Vec3};

/// A camera that orbits around a target point.
///
/// Uses spherical coordinates (radius, theta, phi) relative to the target
/// for intuitive 3D navigation.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Point the camera orbits around / looks at
    pub target: Vec3,

    /// Distance from target (spherical radius)
    pub radius: f32,

    /// Azimuth angle in radians (rotation around Y axis)
    pub theta: f32,

    /// Elevation angle in radians (from Y axis, 0 = top, PI = bottom)
    pub phi: f32,

    /// Vertical field of view in radians
    pub fov_y: f32,

    /// Near clip plane distance
    pub near: f32,

    /// Far clip plane distance
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            radius: 5.0,
            theta: std::f32::consts::FRAC_PI_4,       // 45 degrees
            phi: std::f32::consts::FRAC_PI_4,         // 45 degrees from top
            fov_y: std::f32::consts::FRAC_PI_3,       // 60 degrees
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Camera {
    /// Create a new camera looking at the given target from the specified distance.
    pub fn new(target: Vec3, radius: f32) -> Self {
        Self {
            target,
            radius,
            ..Default::default()
        }
    }

    /// Compute eye position from spherical coordinates.
    ///
    /// Uses standard spherical coordinate conversion:
    /// - theta: azimuth angle (rotation around Y)
    /// - phi: polar angle from Y axis
    pub fn eye_position(&self) -> Vec3 {
        let sin_phi = self.phi.sin();
        let cos_phi = self.phi.cos();
        let sin_theta = self.theta.sin();
        let cos_theta = self.theta.cos();

        let x = self.radius * sin_phi * sin_theta;
        let y = self.radius * cos_phi;
        let z = self.radius * sin_phi * cos_theta;

        self.target + Vec3::new(x, y, z)
    }

    /// Compute the view matrix (world to camera transform).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye_position(), self.target, Vec3::Y)
    }

    /// Compute the projection matrix.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    /// Compute combined view-projection matrix.
    pub fn view_projection_matrix(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    /// Get the camera's forward direction (pointing toward target).
    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye_position()).normalize()
    }

    /// Get the camera's right direction.
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// Get the camera's up direction (may not be exactly Y due to tilt).
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }

    /// Orbit the camera around the target.
    ///
    /// - `delta_theta`: Change in azimuth (horizontal rotation)
    /// - `delta_phi`: Change in elevation (vertical rotation)
    pub fn orbit(&mut self, delta_theta: f32, delta_phi: f32) {
        self.theta += delta_theta;

        // Clamp phi to avoid gimbal lock at poles
        const MIN_PHI: f32 = 0.01;
        const MAX_PHI: f32 = std::f32::consts::PI - 0.01;
        self.phi = (self.phi + delta_phi).clamp(MIN_PHI, MAX_PHI);
    }

    /// Pan the camera (translate target in the view plane).
    ///
    /// - `delta_screen`: Mouse delta in screen pixels
    /// - `_viewport_size`: Viewport dimensions in pixels (unused)
    pub fn pan(&mut self, delta_screen: Vec2, _viewport_size: Vec2) {
        // Use camera's own coordinate axes
        let right = self.right();
        let up = self.up();

        // Scale movement by distance for consistent feel at any zoom level
        // The 0.002 factor provides reasonable sensitivity
        let scale = self.radius * 0.002;

        // Move target opposite to drag direction (scene follows mouse)
        self.target -= right * (delta_screen.x * scale);
        self.target += up * (delta_screen.y * scale);
    }

    /// Zoom the camera (adjust distance from target).
    ///
    /// - `delta`: Positive to zoom in, negative to zoom out
    pub fn zoom(&mut self, delta: f32) {
        const MIN_RADIUS: f32 = 0.1;
        const MAX_RADIUS: f32 = 1000.0;

        // Multiplicative zoom for consistent feel
        let factor = 1.0 - delta * 0.1;
        self.radius = (self.radius * factor).clamp(MIN_RADIUS, MAX_RADIUS);
    }

    /// Zoom with explicit min/max radius.
    pub fn zoom_clamped(&mut self, delta: f32, min_radius: f32, max_radius: f32) {
        let factor = 1.0 - delta * 0.1;
        self.radius = (self.radius * factor).clamp(min_radius, max_radius);
    }

    /// Focus the camera on a bounding box.
    ///
    /// Centers the target on the box and adjusts distance to fit the box in view.
    pub fn focus_on(&mut self, min: Vec3, max: Vec3) {
        // Center target on bounding box
        self.target = (min + max) * 0.5;

        // Calculate diagonal and set radius to fit
        let diagonal = (max - min).length();
        self.radius = diagonal * 1.5;

        // Clamp radius to reasonable range
        self.radius = self.radius.clamp(0.1, 1000.0);
    }

    /// Focus on a point with specified distance.
    pub fn focus_on_point(&mut self, point: Vec3, distance: f32) {
        self.target = point;
        self.radius = distance.clamp(0.1, 1000.0);
    }

    /// Reset camera to default orientation while keeping target and distance.
    pub fn reset_orientation(&mut self) {
        self.theta = std::f32::consts::FRAC_PI_4;
        self.phi = std::f32::consts::FRAC_PI_4;
    }

    /// Set camera to view from a specific direction.
    pub fn view_from_direction(&mut self, direction: ViewDirection) {
        match direction {
            ViewDirection::Front => {
                self.theta = 0.0;
                self.phi = std::f32::consts::FRAC_PI_2;
            }
            ViewDirection::Back => {
                self.theta = std::f32::consts::PI;
                self.phi = std::f32::consts::FRAC_PI_2;
            }
            ViewDirection::Left => {
                self.theta = -std::f32::consts::FRAC_PI_2;
                self.phi = std::f32::consts::FRAC_PI_2;
            }
            ViewDirection::Right => {
                self.theta = std::f32::consts::FRAC_PI_2;
                self.phi = std::f32::consts::FRAC_PI_2;
            }
            ViewDirection::Top => {
                self.theta = 0.0;
                self.phi = 0.01; // Nearly straight down
            }
            ViewDirection::Bottom => {
                self.theta = 0.0;
                self.phi = std::f32::consts::PI - 0.01; // Nearly straight up
            }
            ViewDirection::Isometric => {
                self.theta = std::f32::consts::FRAC_PI_4;
                self.phi = std::f32::consts::FRAC_PI_4;
            }
        }
    }
}

/// Preset view directions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ViewDirection {
    /// Looking along +Z axis
    Front,
    /// Looking along -Z axis
    Back,
    /// Looking along +X axis
    Left,
    /// Looking along -X axis
    Right,
    /// Looking down along -Y axis
    Top,
    /// Looking up along +Y axis
    Bottom,
    /// Isometric view (45 degrees)
    Isometric,
}

/// Camera control schemes matching popular 3D applications.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CameraControlScheme {
    /// Blender-style: Middle-drag orbits, Shift+Middle pans, Scroll zooms
    #[default]
    Blender,
    /// OnShape-style: Right-drag orbits, Middle-drag pans, Scroll zooms
    OnShape,
    /// Fusion 360-style: Middle-drag orbits, Shift+Middle pans, Scroll zooms (same as Blender)
    Fusion360,
    /// SolidWorks-style: Middle-drag orbits, Ctrl+Middle pans, Scroll zooms
    SolidWorks,
    /// Maya-style: Alt+Left orbits, Alt+Middle pans, Alt+Right or Scroll zooms
    Maya,
}

impl CameraControlScheme {
    /// All available control schemes.
    pub const ALL: &'static [CameraControlScheme] = &[
        CameraControlScheme::Blender,
        CameraControlScheme::OnShape,
        CameraControlScheme::Fusion360,
        CameraControlScheme::SolidWorks,
        CameraControlScheme::Maya,
    ];

    /// Human-readable name for the control scheme.
    pub fn name(&self) -> &'static str {
        match self {
            CameraControlScheme::Blender => "Blender",
            CameraControlScheme::OnShape => "OnShape",
            CameraControlScheme::Fusion360 => "Fusion 360",
            CameraControlScheme::SolidWorks => "SolidWorks",
            CameraControlScheme::Maya => "Maya",
        }
    }

    /// Determine the camera action based on input state.
    pub fn determine_action(&self, input: &CameraInputState) -> CameraAction {
        match self {
            CameraControlScheme::Blender | CameraControlScheme::Fusion360 => {
                // Middle-drag orbits, Shift+Middle pans, Scroll zooms
                if input.middle_down {
                    if input.shift_down {
                        CameraAction::Pan
                    } else {
                        CameraAction::Orbit
                    }
                } else if input.scroll_delta != 0.0 {
                    CameraAction::Zoom
                } else {
                    CameraAction::None
                }
            }
            CameraControlScheme::OnShape => {
                // Right-drag orbits, Middle-drag pans, Scroll zooms
                if input.right_down {
                    CameraAction::Orbit
                } else if input.middle_down {
                    CameraAction::Pan
                } else if input.scroll_delta != 0.0 {
                    CameraAction::Zoom
                } else {
                    CameraAction::None
                }
            }
            CameraControlScheme::SolidWorks => {
                // Middle-drag orbits, Ctrl+Middle pans, Scroll zooms
                if input.middle_down {
                    if input.ctrl_down {
                        CameraAction::Pan
                    } else {
                        CameraAction::Orbit
                    }
                } else if input.scroll_delta != 0.0 {
                    CameraAction::Zoom
                } else {
                    CameraAction::None
                }
            }
            CameraControlScheme::Maya => {
                // Alt+Left orbits, Alt+Middle pans, Alt+Right or Scroll zooms
                if input.alt_down {
                    if input.left_down {
                        CameraAction::Orbit
                    } else if input.middle_down {
                        CameraAction::Pan
                    } else if input.right_down {
                        CameraAction::Zoom
                    } else {
                        CameraAction::None
                    }
                } else if input.scroll_delta != 0.0 {
                    CameraAction::Zoom
                } else {
                    CameraAction::None
                }
            }
        }
    }
}

/// Camera action to perform based on input.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CameraAction {
    /// No camera action
    #[default]
    None,
    /// Orbit around the target
    Orbit,
    /// Pan in the view plane
    Pan,
    /// Zoom in/out
    Zoom,
}

/// Input state for determining camera action.
#[derive(Clone, Debug, Default)]
pub struct CameraInputState {
    /// Left mouse button is down
    pub left_down: bool,
    /// Middle mouse button is down
    pub middle_down: bool,
    /// Right mouse button is down
    pub right_down: bool,
    /// Shift modifier is held
    pub shift_down: bool,
    /// Ctrl modifier is held
    pub ctrl_down: bool,
    /// Alt modifier is held
    pub alt_down: bool,
    /// Mouse delta since last frame
    pub mouse_delta: Vec2,
    /// Scroll wheel delta (positive = zoom in)
    pub scroll_delta: f32,
}

/// Camera uniform data for GPU upload.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Inverse view-projection matrix (for unprojection)
    pub inv_view_proj: [[f32; 4]; 4],
    /// View matrix
    pub view: [[f32; 4]; 4],
    /// Camera position in world space
    pub eye_position: [f32; 3],
    pub _pad0: f32,
    /// Camera forward direction
    pub forward: [f32; 3],
    pub _pad1: f32,
}

impl CameraUniforms {
    /// Create camera uniforms from a camera and aspect ratio.
    pub fn from_camera(camera: &Camera, aspect: f32) -> Self {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();
        let eye = camera.eye_position();
        let forward = camera.forward();

        Self {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            eye_position: eye.into(),
            _pad0: 0.0,
            forward: forward.into(),
            _pad1: 0.0,
        }
    }
}

impl PartialEq for CameraUniforms {
    fn eq(&self, other: &Self) -> bool {
        self.view_proj == other.view_proj
            && self.inv_view_proj == other.inv_view_proj
            && self.view == other.view
            && self.eye_position == other.eye_position
            && self.forward == other.forward
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_default() {
        let camera = Camera::default();
        assert_eq!(camera.target, Vec3::ZERO);
        assert_eq!(camera.radius, 5.0);
    }

    #[test]
    fn test_camera_orbit() {
        let mut camera = Camera::default();
        let initial_theta = camera.theta;
        let initial_phi = camera.phi;

        camera.orbit(0.1, 0.1);

        assert!((camera.theta - (initial_theta + 0.1)).abs() < 0.001);
        assert!((camera.phi - (initial_phi + 0.1)).abs() < 0.001);
    }

    #[test]
    fn test_camera_zoom() {
        let mut camera = Camera::default();
        let initial_radius = camera.radius;

        camera.zoom(1.0); // Zoom in

        assert!(camera.radius < initial_radius);
    }

    #[test]
    fn test_phi_clamping() {
        let mut camera = Camera::default();

        // Try to go past top
        camera.phi = 0.0;
        camera.orbit(0.0, -1.0);
        assert!(camera.phi > 0.0);

        // Try to go past bottom
        camera.phi = std::f32::consts::PI;
        camera.orbit(0.0, 1.0);
        assert!(camera.phi < std::f32::consts::PI);
    }
}
