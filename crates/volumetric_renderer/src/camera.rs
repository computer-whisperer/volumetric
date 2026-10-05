//! The viewport camera: where it looks from, and the operations that move
//! it. World space is right-handed with +Z up.
//!
//! The camera is a focus point, an orientation and a distance. Every
//! operation is defined about a world point that must not move on screen
//! (the point under the cursor), which is what makes orbiting, zooming
//! and panning feel anchored; [`crate::Navigator`] turns pointer input
//! into these operations.

use std::f32::consts::FRAC_PI_2;

use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

/// How the camera projects.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    #[default]
    Perspective,
    Orthographic,
}

/// How an orbit drag turns the camera.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum OrbitMode {
    /// Yaw about world Z, pitch about the camera's right axis, elevation
    /// limited to straight up and straight down. The horizon stays level.
    #[default]
    Turntable,
    /// Rotation about the camera's own up and right axes, unlimited.
    Free,
}

/// The six axis views and the default three-quarter view.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StandardView {
    /// From -Y, looking along +Y.
    Front,
    /// From +Y.
    Back,
    /// From +X.
    Right,
    /// From -X.
    Left,
    /// From +Z looking down, +Y up the screen.
    Top,
    /// From -Z looking up, -Y up the screen.
    Bottom,
    /// From (+X, -Y, +Z).
    Isometric,
}

impl StandardView {
    pub const ALL: [Self; 7] = [
        Self::Front,
        Self::Back,
        Self::Right,
        Self::Left,
        Self::Top,
        Self::Bottom,
        Self::Isometric,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Front => "Front",
            Self::Back => "Back",
            Self::Right => "Right",
            Self::Left => "Left",
            Self::Top => "Top",
            Self::Bottom => "Bottom",
            Self::Isometric => "Isometric",
        }
    }

    /// The view's azimuth and elevation, radians (see [`level_orientation`]).
    fn azimuth_elevation(self) -> (f32, f32) {
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        match self {
            Self::Front => (0.0, 0.0),
            Self::Back => (PI, 0.0),
            Self::Right => (FRAC_PI_2, 0.0),
            Self::Left => (-FRAC_PI_2, 0.0),
            Self::Top => (0.0, FRAC_PI_2),
            Self::Bottom => (0.0, -FRAC_PI_2),
            Self::Isometric => (FRAC_PI_4, (1.0 / 3.0f32.sqrt()).asin()),
        }
    }

    /// The camera-to-world rotation of this view.
    pub fn orientation(self) -> Quat {
        let (azimuth, elevation) = self.azimuth_elevation();
        level_orientation(azimuth, elevation)
    }
}

/// The camera-to-world rotation with a level horizon at `azimuth` (the
/// angle of the camera's right axis from +X, about +Z) and `elevation`
/// (how far above the horizontal the eye sits; `PI/2` looks straight
/// down). Built from the basis directly, so the poles are ordinary poses.
fn level_orientation(azimuth: f32, elevation: f32) -> Quat {
    let right = Vec3::new(azimuth.cos(), azimuth.sin(), 0.0);
    let level_forward = Vec3::Z.cross(right);
    let forward = level_forward * elevation.cos() - Vec3::Z * elevation.sin();
    let up = level_forward * elevation.sin() + Vec3::Z * elevation.cos();
    Quat::from_mat3(&Mat3::from_cols(right, up, -forward)).normalize()
}

/// The viewport camera.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// The point the view is centred on.
    pub focus: Vec3,
    /// Camera-to-world rotation: the camera looks down its -Z, +X is
    /// screen right, +Y is screen up.
    pub orientation: Quat,
    /// Eye-to-focus distance. Under an orthographic projection the eye is
    /// still that far back, and the distance sets the frame's size.
    pub distance: f32,
    pub projection: Projection,
    /// Vertical field of view in radians.
    pub fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            orientation: StandardView::Isometric.orientation(),
            distance: 5.0,
            projection: Projection::Perspective,
            fov_y: 35.0f32.to_radians(),
        }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        self.orientation * Vec3::NEG_Z
    }

    pub fn right(&self) -> Vec3 {
        self.orientation * Vec3::X
    }

    pub fn up(&self) -> Vec3 {
        self.orientation * Vec3::Y
    }

    pub fn eye(&self) -> Vec3 {
        self.focus - self.forward() * self.distance
    }

    /// World to camera.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.orientation, self.eye()).inverse()
    }

    /// Height of the frame at the focus, world units: what an
    /// orthographic projection shows, and what a perspective one shows at
    /// the focus depth. Switching projection keeps it.
    pub fn frame_height(&self) -> f32 {
        2.0 * self.distance * (self.fov_y * 0.5).tan()
    }

    /// Clip planes for the current pose. They follow the distance, so a
    /// surface being approached never crosses the near plane at any part
    /// scale, and reach far enough to hold `scene` (its bounds) whole. An
    /// orthographic frame also sees what is behind the eye.
    pub fn clip_planes(&self, scene: Option<(Vec3, Vec3)>) -> (f32, f32) {
        let mut far = self.distance * 100.0;
        if let Some((min, max)) = scene {
            let reach = (0..8)
                .map(|i| {
                    let corner = Vec3::new(
                        if i & 1 == 0 { min.x } else { max.x },
                        if i & 2 == 0 { min.y } else { max.y },
                        if i & 4 == 0 { min.z } else { max.z },
                    );
                    (corner - self.eye()).length()
                })
                .fold(0.0f32, f32::max);
            if reach.is_finite() {
                far = far.max(reach * 1.5);
            }
        }
        match self.projection {
            Projection::Perspective => (self.distance * 0.005, far),
            Projection::Orthographic => (-far, far),
        }
    }

    pub fn projection_matrix(&self, aspect: f32, near: f32, far: f32) -> Mat4 {
        match self.projection {
            Projection::Perspective => Mat4::perspective_rh(self.fov_y, aspect, near, far),
            Projection::Orthographic => {
                let half_h = self.frame_height() * 0.5;
                let half_w = half_h * aspect;
                Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, near, far)
            }
        }
    }

    /// The matrices a frame is drawn with, clip planes fitted to `scene`.
    pub fn view(&self, aspect: f32, scene: Option<(Vec3, Vec3)>) -> CameraView {
        let (near, far) = self.clip_planes(scene);
        CameraView {
            view: self.view_matrix(),
            projection: self.projection_matrix(aspect, near, far),
        }
    }

    /// The world point seen at `ndc` (x right, y up, each -1..1) at
    /// `depth` along the view direction from the eye.
    pub fn point_at(&self, ndc: Vec2, aspect: f32, depth: f32) -> Vec3 {
        let half_h = match self.projection {
            Projection::Perspective => depth * (self.fov_y * 0.5).tan(),
            Projection::Orthographic => self.frame_height() * 0.5,
        };
        self.eye()
            + self.forward() * depth
            + self.right() * (ndc.x * half_h * aspect)
            + self.up() * (ndc.y * half_h)
    }

    /// How far `point` is in front of the eye, along the view direction.
    pub fn depth_of(&self, point: Vec3) -> f32 {
        (point - self.eye()).dot(self.forward())
    }

    /// Turns the camera rigidly about `pivot`, which therefore stays
    /// where it is on screen. Positive `yaw` carries the eye
    /// anticlockwise seen from above; positive `pitch` raises it.
    ///
    /// Turntable: yaw is about world Z, pitch about the camera's right
    /// axis, and the elevation stops at straight down and straight up. A
    /// camera carrying roll (from a free orbit or a photograph's pose) is
    /// levelled by its first turntable orbit.
    pub fn orbit(&mut self, pivot: Vec3, yaw: f32, pitch: f32, mode: OrbitMode) {
        let turned = match mode {
            OrbitMode::Turntable => {
                let right = self.right();
                let azimuth = right.y.atan2(right.x);
                let elevation = (-self.forward().z)
                    .atan2(self.up().z)
                    .clamp(-FRAC_PI_2, FRAC_PI_2);
                level_orientation(
                    azimuth + yaw,
                    (elevation + pitch).clamp(-FRAC_PI_2, FRAC_PI_2),
                )
            }
            OrbitMode::Free => {
                let turn = Quat::from_axis_angle(self.up(), yaw)
                    * Quat::from_axis_angle(self.right(), -pitch);
                (turn * self.orientation).normalize()
            }
        };
        let rotation = turned * self.orientation.inverse();
        self.focus = pivot + rotation * (self.focus - pivot);
        self.orientation = turned;
    }

    /// Scales the view by `factor` about `point` (below 1 zooms in),
    /// which stays where it is on screen under either projection. The
    /// distance is kept within `limits`.
    pub fn zoom_about(&mut self, point: Vec3, factor: f32, limits: (f32, f32)) {
        if !(factor.is_finite() && factor > 0.0 && self.distance > 0.0) {
            return;
        }
        let distance = (self.distance * factor).clamp(limits.0, limits.1);
        let factor = distance / self.distance;
        self.focus = point + (self.focus - point) * factor;
        self.distance = distance;
    }

    /// Slides the view by `delta_px` (x right, y down) of a viewport
    /// `viewport_height_px` tall, so that a point `depth` in front of the
    /// eye follows the pointer exactly.
    pub fn pan(&mut self, delta_px: Vec2, viewport_height_px: f32, depth: f32) {
        if viewport_height_px <= 0.0 {
            return;
        }
        let frame_height = match self.projection {
            Projection::Perspective => 2.0 * depth * (self.fov_y * 0.5).tan(),
            Projection::Orthographic => self.frame_height(),
        };
        let world_per_px = frame_height / viewport_height_px;
        self.focus += (self.up() * delta_px.y - self.right() * delta_px.x) * world_per_px;
    }

    /// Centres on the box `min..max` and backs off until its bounding
    /// sphere fits the frame, whatever the aspect ratio. The orientation
    /// is kept.
    pub fn frame(&mut self, min: Vec3, max: Vec3, aspect: f32) {
        let radius = (max - min).length() * 0.5;
        if !(radius.is_finite() && radius > 0.0 && aspect > 0.0) {
            return;
        }
        const MARGIN: f32 = 1.1;
        let half_tan = (self.fov_y * 0.5).tan();
        self.focus = (min + max) * 0.5;
        self.distance = MARGIN
            * match self.projection {
                // The sphere must fit the narrower of the two half-angles.
                Projection::Perspective => {
                    let narrow = half_tan.min(half_tan * aspect).atan();
                    radius / narrow.sin()
                }
                Projection::Orthographic => radius / (half_tan * aspect.min(1.0)),
            };
    }

    /// Turns to a standard view, keeping the focus and distance.
    pub fn set_view(&mut self, view: StandardView) {
        self.orientation = view.orientation();
    }

    /// Takes the pose of a camera at `eye` looking at `target` with `up`
    /// up the screen: how a photograph's viewpoint is handed over, roll
    /// included. Ignored when the pose is degenerate.
    pub fn look_from(&mut self, eye: Vec3, target: Vec3, up: Vec3) {
        let offset = target - eye;
        let distance = offset.length();
        let forward = offset / distance;
        let right = forward.cross(up).normalize_or_zero();
        if !(distance.is_finite() && distance > 1e-6) || right == Vec3::ZERO {
            return;
        }
        let up = right.cross(forward);
        self.orientation = Quat::from_mat3(&Mat3::from_cols(right, up, -forward)).normalize();
        self.focus = target;
        self.distance = distance;
    }

    /// The pose a fraction `t` of the way from `self` to `to`: the focus
    /// moves in a straight line, the distance geometrically, the
    /// orientation by the shortest rotation.
    pub fn interpolate(&self, to: &Camera, t: f32) -> Camera {
        Camera {
            focus: self.focus.lerp(to.focus, t),
            orientation: self.orientation.slerp(to.orientation, t).normalize(),
            distance: self.distance * (to.distance / self.distance).powf(t),
            projection: to.projection,
            fov_y: self.fov_y + (to.fov_y - self.fov_y) * t,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIEWPORT: Vec2 = Vec2::new(800.0, 600.0);

    /// Where a world point lands, in pixels (x right, y down).
    fn on_screen(camera: &Camera, world: Vec3) -> Vec2 {
        camera
            .view(VIEWPORT.x / VIEWPORT.y, None)
            .project(world, VIEWPORT.x as u32, VIEWPORT.y as u32)
            .expect("in front of the camera")
    }

    fn both_projections(test: impl Fn(Camera)) {
        for projection in [Projection::Perspective, Projection::Orthographic] {
            test(Camera {
                focus: Vec3::new(0.2, -0.1, 0.3),
                distance: 2.5,
                projection,
                ..Camera::default()
            });
        }
    }

    /// Each standard view looks from where its name says, with the stated
    /// axis up the screen; Top and Bottom are exactly on the Z axis.
    #[test]
    fn standard_views_look_from_where_they_say() {
        let cases = [
            (StandardView::Front, Vec3::NEG_Y, Vec3::Z),
            (StandardView::Back, Vec3::Y, Vec3::Z),
            (StandardView::Right, Vec3::X, Vec3::Z),
            (StandardView::Left, Vec3::NEG_X, Vec3::Z),
            (StandardView::Top, Vec3::Z, Vec3::Y),
            (StandardView::Bottom, Vec3::NEG_Z, Vec3::NEG_Y),
        ];
        for (view, eye_side, screen_up) in cases {
            let mut camera = Camera::default();
            camera.set_view(view);
            let eye = camera.eye().normalize();
            assert!((eye - eye_side).length() < 1e-6, "{view:?} eye {eye}");
            assert!((camera.up() - screen_up).length() < 1e-6, "{view:?}");
            // Right-handed: X runs to the right in Front and Top.
            assert!(camera.right().cross(camera.up()).dot(-camera.forward()) > 0.999);
        }
        let camera = Camera::default();
        let eye = camera.eye().normalize();
        assert!(
            eye.x > 0.5 && eye.y < -0.5 && eye.z > 0.5,
            "isometric {eye}"
        );
        assert!(camera.right().z.abs() < 1e-6);
    }

    /// The orbit pivot does not move on screen, wherever it is in the
    /// frame, in either mode and projection.
    #[test]
    fn orbit_keeps_its_pivot_on_screen() {
        both_projections(|start| {
            for mode in [OrbitMode::Turntable, OrbitMode::Free] {
                let mut camera = start.clone();
                let pivot = Vec3::new(0.6, 0.3, 0.1);
                let before = on_screen(&camera, pivot);
                for _ in 0..5 {
                    camera.orbit(pivot, 0.31, -0.17, mode);
                }
                let after = on_screen(&camera, pivot);
                assert!(
                    (after - before).length() < 0.05,
                    "{mode:?}: {before} -> {after}"
                );
                assert!((camera.orientation.length() - 1.0).abs() < 1e-5);
            }
        });
    }

    /// Turntable orbits keep the horizon level and stop exactly at the
    /// poles, where yaw still turns the view; free orbits pass over them.
    #[test]
    fn turntable_is_level_and_stops_at_the_poles() {
        let mut camera = Camera::default();
        camera.orbit(Vec3::ZERO, 0.4, 10.0, OrbitMode::Turntable);
        assert!(
            (camera.forward() - Vec3::NEG_Z).length() < 1e-6,
            "{}",
            camera.forward()
        );
        assert!(camera.right().z.abs() < 1e-6);
        let right = camera.right();
        camera.orbit(Vec3::ZERO, 0.5, 0.0, OrbitMode::Turntable);
        assert!((camera.forward() - Vec3::NEG_Z).length() < 1e-6);
        assert!((right.angle_between(camera.right()) - 0.5).abs() < 1e-5);
        camera.orbit(Vec3::ZERO, 0.0, -10.0, OrbitMode::Turntable);
        assert!((camera.forward() - Vec3::Z).length() < 1e-6);

        let mut free = Camera::default();
        free.set_view(StandardView::Top);
        free.orbit(Vec3::ZERO, 0.0, 0.3, OrbitMode::Free);
        // Past the pole: the eye has come over the top.
        assert!(free.up().z < -0.2, "{}", free.up());
    }

    /// The point zoomed about stays under the cursor in both projections,
    /// and the distance respects its limits.
    #[test]
    fn zoom_keeps_its_point_on_screen() {
        both_projections(|start| {
            let mut camera = start.clone();
            let point = Vec3::new(-0.4, 0.5, 0.0);
            let before = on_screen(&camera, point);
            camera.zoom_about(point, 0.4, (0.01, 100.0));
            let after = on_screen(&camera, point);
            assert!((after - before).length() < 0.05, "{before} -> {after}");
            assert!((camera.distance - 1.0).abs() < 1e-5);
            // Nearer things got bigger: a neighbour moved away from it.
            let neighbour = point + camera.right() * 0.1;
            let spread = (on_screen(&camera, neighbour) - after).length();
            let spread_before = (on_screen(&start, neighbour) - before).length();
            assert!(spread > spread_before * 1.5);

            camera.zoom_about(point, 1e-6, (0.5, 100.0));
            assert_eq!(camera.distance, 0.5);
            assert!((on_screen(&camera, point) - before).length() < 0.05);
        });
    }

    /// A point at the pan's depth follows the pointer pixel for pixel.
    #[test]
    fn pan_is_one_to_one_at_the_grabbed_depth() {
        both_projections(|start| {
            let mut camera = start.clone();
            let grabbed = camera.point_at(Vec2::new(0.3, -0.2), VIEWPORT.x / VIEWPORT.y, 1.7);
            let before = on_screen(&camera, grabbed);
            camera.pan(Vec2::new(60.0, 24.0), VIEWPORT.y, camera.depth_of(grabbed));
            let moved = on_screen(&camera, grabbed) - before;
            assert!((moved - Vec2::new(60.0, 24.0)).length() < 0.05, "{moved}");
        });
    }

    /// `point_at` is the inverse of projection.
    #[test]
    fn point_at_inverts_projection() {
        both_projections(|camera| {
            let aspect = VIEWPORT.x / VIEWPORT.y;
            let ndc = Vec2::new(-0.6, 0.45);
            let px = on_screen(&camera, camera.point_at(ndc, aspect, 3.0));
            let expected = Vec2::new(
                (ndc.x + 1.0) * 0.5 * VIEWPORT.x,
                (1.0 - ndc.y) * 0.5 * VIEWPORT.y,
            );
            assert!((px - expected).length() < 0.05, "{px} vs {expected}");
        });
    }

    /// Framing puts every corner of the box inside the frame, at wide and
    /// tall aspect ratios, without leaving it tiny.
    #[test]
    fn frame_fits_the_box_at_any_aspect() {
        let (min, max) = (Vec3::new(-0.3, -0.1, 0.0), Vec3::new(0.5, 0.2, 0.1));
        for projection in [Projection::Perspective, Projection::Orthographic] {
            for (w, h) in [(1600u32, 400u32), (400, 1600), (800, 600)] {
                let mut camera = Camera {
                    projection,
                    ..Camera::default()
                };
                let aspect = w as f32 / h as f32;
                camera.frame(min, max, aspect);
                let view = camera.view(aspect, Some((min, max)));
                let mut extent = 0.0f32;
                for i in 0..8 {
                    let corner = Vec3::new(
                        if i & 1 == 0 { min.x } else { max.x },
                        if i & 2 == 0 { min.y } else { max.y },
                        if i & 4 == 0 { min.z } else { max.z },
                    );
                    let px = view.project(corner, w, h).expect("in front");
                    assert!(
                        px.x >= 0.0 && px.x <= w as f32 && px.y >= 0.0 && px.y <= h as f32,
                        "{projection:?} {w}x{h}: corner at {px}"
                    );
                    let centre = Vec2::new(w as f32, h as f32) * 0.5;
                    extent = extent.max((px - centre).length());
                }
                // The farthest corner reaches at least a third of the way
                // to the frame's nearer edge.
                assert!(extent > w.min(h) as f32 * 0.5 / 3.0, "{w}x{h}: {extent}");
            }
        }
    }

    /// Switching projection keeps what the focus plane shows.
    #[test]
    fn projections_agree_at_the_focus() {
        let perspective = Camera {
            distance: 2.0,
            ..Camera::default()
        };
        let orthographic = Camera {
            projection: Projection::Orthographic,
            ..perspective.clone()
        };
        let at_focus = perspective.focus + perspective.right() * 0.3 + perspective.up() * 0.2;
        let delta = on_screen(&perspective, at_focus) - on_screen(&orthographic, at_focus);
        assert!(delta.length() < 0.05, "{delta}");
    }

    /// The near plane follows the distance; the far plane holds the scene.
    #[test]
    fn clip_planes_follow_the_distance_and_hold_the_scene() {
        let mut camera = Camera {
            distance: 0.01,
            ..Camera::default()
        };
        let (near, far) = camera.clip_planes(None);
        assert!(near > 0.0 && near < 0.001 && far > 0.5);
        let scene = (Vec3::splat(-50.0), Vec3::splat(50.0));
        let (_, far) = camera.clip_planes(Some(scene));
        assert!(far > 86.0, "far {far} must reach the scene's far corner");
        camera.projection = Projection::Orthographic;
        let (near, far) = camera.clip_planes(Some(scene));
        assert!(near < -86.0 && far > 86.0);
    }

    /// A handed-over pose is kept exactly, roll included.
    #[test]
    fn look_from_keeps_the_pose() {
        let mut camera = Camera::default();
        let eye = Vec3::new(1.5, 0.8, -2.0);
        let target = Vec3::new(0.2, 0.1, 0.3);
        let up = Vec3::new(0.3, 1.0, 0.1).normalize();
        camera.look_from(eye, target, up);
        assert!((camera.eye() - eye).length() < 1e-5);
        assert!((camera.focus - target).length() < 1e-6);
        assert!(camera.up().dot(up) > 0.9);
        assert!(camera.right().dot(up).abs() < 1e-5, "roll was lost");
        let before = camera.clone();
        camera.look_from(Vec3::ONE, Vec3::ONE, Vec3::Z);
        assert_eq!(camera, before);
    }

    /// Interpolation starts and ends on the given poses.
    #[test]
    fn interpolation_ends_on_the_target_pose() {
        let from = Camera::default();
        let mut to = Camera {
            focus: Vec3::new(1.0, 2.0, 3.0),
            distance: 0.5,
            ..Camera::default()
        };
        to.set_view(StandardView::Top);
        let end = from.interpolate(&to, 1.0);
        assert!((end.focus - to.focus).length() < 1e-6);
        assert!((end.distance - to.distance).abs() < 1e-6);
        assert!(end.orientation.dot(to.orientation).abs() > 0.999_999);
        let mid = from.interpolate(&to, 0.5);
        assert!((mid.distance - (5.0f32 * 0.5).sqrt()).abs() < 1e-4);
    }
}

/// What a frame is drawn with: the world-to-camera transform and the
/// projection, as matrices. The viewport [`Camera`] produces one per frame; a
/// [`Pinhole`] from a posed photograph produces one directly, which is how
/// a scan's views are looked through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraView {
    pub view: Mat4,
    pub projection: Mat4,
}

impl CameraView {
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
}
