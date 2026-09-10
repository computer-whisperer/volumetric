//! Camera system with orbit, pan, and zoom controls.
//!
//! Uses spherical coordinates for intuitive 3D navigation around a target point.

#![allow(dead_code)]

use glam::{Mat4, Vec2, Vec3, Vec4};

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
            theta: std::f32::consts::FRAC_PI_4, // 45 degrees
            phi: std::f32::consts::FRAC_PI_4,   // 45 degrees from top
            fov_y: std::f32::consts::FRAC_PI_3, // 60 degrees
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
    /// True 1:1 grab: one pixel of drag maps to the world span of one pixel
    /// at the focus (target) depth — the frustum is `2·radius·tan(fov_y/2)`
    /// world units tall across `viewport_size.y` pixels, and the scale is
    /// uniform across x/y for square pixels. The grabbed point tracks the
    /// cursor exactly at any zoom or window size.
    ///
    /// - `delta_screen`: Mouse delta, in the same pixel space as `viewport_size`
    /// - `viewport_size`: Viewport dimensions in pixels
    pub fn pan(&mut self, delta_screen: Vec2, viewport_size: Vec2) {
        if viewport_size.y <= 0.0 {
            return;
        }

        // Use camera's own coordinate axes
        let right = self.right();
        let up = self.up();

        // World span of one pixel at the focus depth.
        let world_per_pixel = 2.0 * self.radius * (self.fov_y * 0.5).tan() / viewport_size.y;

        // Move target opposite to drag direction (scene follows mouse)
        self.target -= right * (delta_screen.x * world_per_pixel);
        self.target += up * (delta_screen.y * world_per_pixel);
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

    /// Fit the clip planes to the current orbit distance.
    ///
    /// Radius-relative planes make zooming scale-free: the near plane stays
    /// a fixed fraction of the orbit distance, so a surface being approached
    /// never crosses it — at any part scale. The 2e4 near:far ratio is well
    /// within Depth24Plus precision.
    pub fn fit_clip_planes(&mut self) {
        self.near = self.radius * 0.005;
        self.far = self.radius * 100.0;
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

    /// Orbit `target` from `eye`: the pose a posed photograph hands over
    /// when the user leaves it for the orbit camera. Roll is lost (the
    /// orbit camera keeps world +y up) and the pitch stays off the poles.
    pub fn look_from(&mut self, eye: Vec3, target: Vec3) {
        let offset = eye - target;
        let radius = offset.length();
        if radius.is_nan() || radius <= 1e-6 {
            return;
        }
        self.target = target;
        self.radius = radius;
        self.phi = (offset.y / radius)
            .clamp(-1.0, 1.0)
            .acos()
            .clamp(0.01, std::f32::consts::PI - 0.01);
        self.theta = offset.x.atan2(offset.z);
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
    fn clip_planes_follow_the_orbit_radius() {
        let mut camera = Camera {
            radius: 0.26, // a framed 0.1^3 part
            ..Default::default()
        };
        camera.fit_clip_planes();
        assert!(
            camera.near < 0.26 * 0.01,
            "near {} must sit well inside the orbit radius",
            camera.near
        );
        assert!(camera.far > 0.26, "far {} must cover the scene", camera.far);

        // Zooming in pulls the near plane along, so the approached surface
        // never crosses it.
        camera.radius = 0.01;
        camera.fit_clip_planes();
        assert!(camera.near < 0.001);
        assert!(camera.near > 0.0);
    }

    /// Projects a world point to pointer-space pixels (x right, y down).
    fn project_to_screen(camera: &Camera, world: Vec3, viewport: Vec2) -> Vec2 {
        let clip = camera.view_projection_matrix(viewport.x / viewport.y) * world.extend(1.0);
        let ndc = clip / clip.w;
        Vec2::new(
            (ndc.x * 0.5 + 0.5) * viewport.x,
            (0.5 - ndc.y * 0.5) * viewport.y,
        )
    }

    #[test]
    fn pan_is_one_to_one_at_focus_depth() {
        let mut camera = Camera::default();
        let viewport = Vec2::new(800.0, 600.0);
        // Grab the point at the focus depth (the target) and drag: it must
        // track the cursor exactly, at any zoom or viewport size.
        let grabbed = camera.target;
        let before = project_to_screen(&camera, grabbed, viewport);

        camera.pan(Vec2::new(60.0, 24.0), viewport);

        let moved = project_to_screen(&camera, grabbed, viewport) - before;
        assert!(
            (moved.x - 60.0).abs() < 1e-2 && (moved.y - 24.0).abs() < 1e-2,
            "grabbed point moved {moved:?} for a (60, 24) px drag"
        );

        // Still 1:1 after zooming in and at a different window size.
        camera.zoom(3.0);
        let viewport = Vec2::new(333.0, 1111.0);
        let grabbed = camera.target;
        let before = project_to_screen(&camera, grabbed, viewport);

        camera.pan(Vec2::new(-17.0, 5.0), viewport);

        let moved = project_to_screen(&camera, grabbed, viewport) - before;
        assert!(
            (moved.x + 17.0).abs() < 1e-2 && (moved.y - 5.0).abs() < 1e-2,
            "grabbed point moved {moved:?} for a (-17, 5) px drag"
        );
    }

    #[test]
    fn test_phi_clamping() {
        // Try to go past top
        let mut camera = Camera {
            phi: 0.0,
            ..Default::default()
        };
        camera.orbit(0.0, -1.0);
        assert!(camera.phi > 0.0);

        // Try to go past bottom
        let mut camera = Camera {
            phi: std::f32::consts::PI,
            ..Default::default()
        };
        camera.orbit(0.0, 1.0);
        assert!(camera.phi < std::f32::consts::PI);
    }
}

/// What a frame is drawn with: the world-to-camera transform and the
/// projection, as matrices. The orbit [`Camera`] produces one per frame; a
/// [`Pinhole`] from a posed photograph produces one directly, which is how
/// a scan's views are looked through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraView {
    pub view: Mat4,
    pub projection: Mat4,
}

impl CameraView {
    pub fn from_camera(camera: &Camera, aspect: f32) -> Self {
        Self {
            view: camera.view_matrix(),
            projection: camera.projection_matrix(aspect),
        }
    }

    /// A perspective camera at `eye` looking at `target`, `fov_y` in
    /// radians.
    pub fn look_at(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            view: Mat4::look_at_rh(eye, target, up),
            projection: Mat4::perspective_rh(fov_y, aspect, near, far),
        }
    }

    /// An orthographic camera at `eye` looking at `target`, whose frame is
    /// `height` world units tall.
    pub fn look_at_orthographic(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        height: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let half_h = height * 0.5;
        let half_w = half_h * aspect;
        Self {
            view: Mat4::look_at_rh(eye, target, up),
            projection: Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, near, far),
        }
    }

    /// A pinhole camera posed by `camera_to_world` (OpenCV convention, see
    /// [`Pinhole`]).
    pub fn pinhole(pinhole: &Pinhole, camera_to_world: Mat4, near: f32, far: f32) -> Self {
        Self {
            view: camera_to_world.inverse(),
            projection: pinhole.projection(near, far),
        }
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection * self.view
    }

    /// The pixel a world point lands on in a `width` x `height` image
    /// (origin top-left, +y down), or `None` when it is behind the camera.
    pub fn project(&self, world: Vec3, width: u32, height: u32) -> Option<Vec2> {
        let clip = self.view_projection() * world.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new(
            (ndc.x + 1.0) * 0.5 * width as f32,
            (1.0 - ndc.y) * 0.5 * height as f32,
        ))
    }
}

/// A pinhole camera in OpenCV convention: pixel coordinates have their
/// origin at the image's top-left corner with +u right and +v down, and
/// camera coordinates are +x right, +y down, +z forward into the scene.
/// Scan and photogrammetry tools export intrinsics and camera-to-world
/// poses in exactly this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pinhole {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl Pinhole {
    /// The projection from OpenCV camera coordinates to wgpu clip space
    /// (x right, y up, depth 0 at `near` and 1 at `far`). The principal
    /// point offset shears the frustum, so an off-centre `cx, cy` is
    /// honoured exactly.
    pub fn projection(&self, near: f32, far: f32) -> Mat4 {
        let (w, h) = (self.width as f32, self.height as f32);
        Mat4::from_cols(
            Vec4::new(2.0 * self.fx / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -2.0 * self.fy / h, 0.0, 0.0),
            Vec4::new(
                2.0 * self.cx / w - 1.0,
                1.0 - 2.0 * self.cy / h,
                far / (far - near),
                1.0,
            ),
            Vec4::new(0.0, 0.0, -far * near / (far - near), 0.0),
        )
    }

    /// Vertical field of view in radians.
    pub fn fov_y(&self) -> f32 {
        2.0 * (self.height as f32 / (2.0 * self.fy)).atan()
    }
}

#[cfg(test)]
mod look_from_tests {
    use super::*;

    #[test]
    fn look_from_round_trips_the_eye() {
        let mut camera = Camera::default();
        let eye = Vec3::new(1.5, 0.8, -2.0);
        let target = Vec3::new(0.2, 0.1, 0.3);
        camera.look_from(eye, target);
        assert!(
            (camera.eye_position() - eye).length() < 1e-4,
            "{:?}",
            camera.eye_position()
        );
        assert!((camera.target - target).length() < 1e-6);
        assert!((camera.forward() - (target - eye).normalize()).length() < 1e-4);

        // Straight down keeps off the pole; a zero distance is ignored.
        camera.look_from(Vec3::new(0.0, 3.0, 0.0), Vec3::ZERO);
        assert!(camera.phi >= 0.01 && camera.radius == 3.0);
        let before = camera.clone();
        camera.look_from(Vec3::ONE, Vec3::ONE);
        assert_eq!(camera.radius, before.radius);
    }
}

#[cfg(test)]
mod view_tests {
    use super::*;

    fn pinhole() -> Pinhole {
        Pinhole {
            fx: 1000.0,
            fy: 1000.0,
            cx: 480.0,
            cy: 270.0,
            width: 960,
            height: 540,
        }
    }

    /// A point in front of an unposed pinhole lands on the OpenCV pixel
    /// `(fx x / z + cx, fy y / z + cy)`, +v down.
    #[test]
    fn pinhole_projects_to_opencv_pixels() {
        let view = CameraView::pinhole(&pinhole(), Mat4::IDENTITY, 0.1, 10.0);
        let pixel = view
            .project(Vec3::new(0.1, 0.05, 2.0), 960, 540)
            .expect("in front");
        assert!((pixel.x - 530.0).abs() < 1e-3, "{pixel}");
        assert!((pixel.y - 295.0).abs() < 1e-3, "{pixel}");
        assert!(view.project(Vec3::new(0.0, 0.0, -1.0), 960, 540).is_none());
    }

    /// The pose places the camera: a world point given in the posed
    /// camera's own coordinates lands where the unposed one would.
    #[test]
    fn pinhole_pose_is_camera_to_world() {
        let camera_to_world = Mat4::from_rotation_translation(
            glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let in_camera = Vec3::new(0.1, 0.05, 2.0);
        let world = camera_to_world.transform_point3(in_camera);
        let view = CameraView::pinhole(&pinhole(), camera_to_world, 0.1, 10.0);
        let pixel = view.project(world, 960, 540).expect("in front");
        assert!((pixel.x - 530.0).abs() < 1e-2, "{pixel}");
        assert!((pixel.y - 295.0).abs() < 1e-2, "{pixel}");
    }

    /// Depth runs from 0 at the near plane to 1 at the far plane, and an
    /// off-centre principal point moves the optical axis, not the frame.
    #[test]
    fn pinhole_depth_and_principal_point() {
        let projection = pinhole().projection(0.5, 8.0);
        let at = |z: f32| {
            let clip = projection * Vec4::new(0.0, 0.0, z, 1.0);
            clip.z / clip.w
        };
        assert!(at(0.5).abs() < 1e-6);
        assert!((at(8.0) - 1.0).abs() < 1e-6);

        let shifted = Pinhole {
            cx: 100.0,
            ..pinhole()
        };
        let view = CameraView::pinhole(&shifted, Mat4::IDENTITY, 0.1, 10.0);
        let axis = view.project(Vec3::new(0.0, 0.0, 1.0), 960, 540).unwrap();
        assert!((axis.x - 100.0).abs() < 1e-3 && (axis.y - 270.0).abs() < 1e-3);
    }

    /// The orbit camera and its explicit view agree.
    #[test]
    fn camera_view_matches_the_orbit_camera() {
        let camera = Camera::new(Vec3::new(1.0, 0.5, -2.0), 3.0);
        let view = CameraView::from_camera(&camera, 1.5);
        assert_eq!(view.view_projection(), camera.view_projection_matrix(1.5));
    }
}
