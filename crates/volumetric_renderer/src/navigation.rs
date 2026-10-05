//! Navigation: pointer input turned into camera motion.
//!
//! A [`CameraControlScheme`] says which buttons mean which gesture; the
//! [`Navigator`] carries a gesture from press to release. Every gesture is
//! anchored on the world point under the cursor when it starts (from a
//! pick of the last frame, or the focus plane where the pick found
//! nothing), and that point stays under the cursor while the gesture runs.

use glam::{Vec2, Vec3};

use crate::{Camera, OrbitMode};

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

/// Radians of orbit per pixel of drag.
const ORBIT_SPEED: f32 = 0.01;
/// Zoom factor per wheel step, and per 50 px of a zoom drag.
const ZOOM_STEP: f32 = 0.9;
/// Seconds an animated view change takes.
const TRANSITION_SECONDS: f32 = 0.22;

/// What the pointer is over when a gesture starts.
#[derive(Copy, Clone, Debug)]
pub struct Cursor {
    /// Normalised device coordinates in the viewport: x right, y up, each
    /// -1..1.
    pub ndc: Vec2,
    /// The world point the last frame drew there, if it drew a surface.
    pub picked: Option<Vec3>,
}

struct Gesture {
    action: CameraAction,
    /// The world point the gesture is anchored on.
    anchor: Vec3,
    /// Its depth in front of the eye when the gesture began.
    depth: f32,
}

struct Transition {
    from: Camera,
    to: Camera,
    elapsed: f32,
}

/// Carries camera gestures and animated view changes.
#[derive(Default)]
pub struct Navigator {
    gesture: Option<Gesture>,
    transition: Option<Transition>,
}

impl Navigator {
    /// The point a gesture starting at `cursor` is anchored on: the
    /// picked surface point when it is in front of the eye, else where
    /// the cursor's ray meets the plane through the focus.
    fn anchor(camera: &Camera, cursor: Cursor, aspect: f32) -> (Vec3, f32) {
        if let Some(point) = cursor.picked {
            let depth = camera.depth_of(point);
            if depth.is_finite() && depth > 0.0 {
                return (point, depth);
            }
        }
        (
            camera.point_at(cursor.ndc, aspect, camera.distance),
            camera.distance,
        )
    }

    /// Advances a drag by `delta_px` (x right, y down). `cursor` is where
    /// the pointer was before this move; it anchors the gesture when one
    /// starts, which is on the first move and whenever `action` changes
    /// mid-drag. Returns whether the camera moved.
    #[allow(clippy::too_many_arguments)]
    pub fn drag(
        &mut self,
        camera: &mut Camera,
        action: CameraAction,
        cursor: Cursor,
        delta_px: Vec2,
        viewport_px: Vec2,
        orbit_mode: OrbitMode,
        zoom_limits: (f32, f32),
    ) -> bool {
        if action == CameraAction::None || viewport_px.x <= 0.0 || viewport_px.y <= 0.0 {
            self.gesture = None;
            return false;
        }
        self.transition = None;
        if self.gesture.as_ref().is_none_or(|g| g.action != action) {
            let (anchor, depth) = Self::anchor(camera, cursor, viewport_px.x / viewport_px.y);
            self.gesture = Some(Gesture {
                action,
                anchor,
                depth,
            });
        }
        let gesture = self.gesture.as_ref().expect("set above");
        match action {
            CameraAction::Orbit => camera.orbit(
                gesture.anchor,
                -delta_px.x * ORBIT_SPEED,
                delta_px.y * ORBIT_SPEED,
                orbit_mode,
            ),
            CameraAction::Pan => camera.pan(delta_px, viewport_px.y, gesture.depth),
            CameraAction::Zoom => camera.zoom_about(
                gesture.anchor,
                ZOOM_STEP.powf(delta_px.x / 50.0),
                zoom_limits,
            ),
            CameraAction::None => unreachable!("returned above"),
        }
        true
    }

    /// Ends the drag in progress, if any.
    pub fn release(&mut self) {
        self.gesture = None;
    }

    /// Zooms by `steps` wheel steps (positive zooms in) about what is
    /// under `cursor`.
    pub fn wheel(
        &mut self,
        camera: &mut Camera,
        steps: f32,
        cursor: Cursor,
        aspect: f32,
        zoom_limits: (f32, f32),
    ) {
        self.transition = None;
        let (anchor, _) = Self::anchor(camera, cursor, aspect);
        camera.zoom_about(anchor, ZOOM_STEP.powf(steps), zoom_limits);
    }

    /// Starts an animated change from `camera`'s pose to `to`.
    pub fn transition_to(&mut self, camera: &Camera, to: Camera) {
        self.gesture = None;
        self.transition = Some(Transition {
            from: camera.clone(),
            to,
            elapsed: 0.0,
        });
    }

    /// Whether an animated change is running: the host keeps drawing
    /// frames while it is.
    pub fn animating(&self) -> bool {
        self.transition.is_some()
    }

    /// Advances the running animation by `dt` seconds, easing in and out,
    /// and lands exactly on its target pose.
    pub fn tick(&mut self, camera: &mut Camera, dt: f32) {
        let Some(transition) = &mut self.transition else {
            return;
        };
        transition.elapsed += dt.max(0.0);
        let t = (transition.elapsed / TRANSITION_SECONDS).min(1.0);
        if t >= 1.0 {
            *camera = transition.to.clone();
            self.transition = None;
        } else {
            let eased = t * t * (3.0 - 2.0 * t);
            *camera = transition.from.interpolate(&transition.to, eased);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Projection, StandardView};

    const VIEWPORT: Vec2 = Vec2::new(800.0, 600.0);
    const LIMITS: (f32, f32) = (0.01, 100.0);

    fn on_screen(camera: &Camera, world: Vec3) -> Vec2 {
        camera
            .view(VIEWPORT.x / VIEWPORT.y, None)
            .project(world, VIEWPORT.x as u32, VIEWPORT.y as u32)
            .expect("in front of the camera")
    }

    fn ndc_of(px: Vec2) -> Vec2 {
        Vec2::new(px.x / VIEWPORT.x * 2.0 - 1.0, 1.0 - px.y / VIEWPORT.y * 2.0)
    }

    fn cameras() -> [Camera; 2] {
        [Projection::Perspective, Projection::Orthographic].map(|projection| Camera {
            projection,
            distance: 2.0,
            ..Camera::default()
        })
    }

    /// An orbit drag pivots on the picked point: it stays under the
    /// cursor for the whole drag.
    #[test]
    fn orbit_drags_pivot_on_the_picked_point() {
        for mut camera in cameras() {
            let picked = Vec3::new(0.4, -0.2, 0.3);
            let at = on_screen(&camera, picked);
            let cursor = Cursor {
                ndc: ndc_of(at),
                picked: Some(picked),
            };
            let mut navigator = Navigator::default();
            for _ in 0..10 {
                assert!(navigator.drag(
                    &mut camera,
                    CameraAction::Orbit,
                    cursor,
                    Vec2::new(13.0, -7.0),
                    VIEWPORT,
                    OrbitMode::Turntable,
                    LIMITS,
                ));
            }
            assert!((on_screen(&camera, picked) - at).length() < 0.1);
            assert!(camera.right().z.abs() < 1e-5, "horizon tilted");
        }
    }

    /// A pan drag carries the picked point with the pointer, pixel for
    /// pixel, whatever its depth.
    #[test]
    fn pan_drags_carry_the_picked_point() {
        for mut camera in cameras() {
            let picked = camera.focus + camera.forward() * 0.7 + camera.right() * 0.2;
            let at = on_screen(&camera, picked);
            let cursor = Cursor {
                ndc: ndc_of(at),
                picked: Some(picked),
            };
            let mut navigator = Navigator::default();
            for _ in 0..4 {
                navigator.drag(
                    &mut camera,
                    CameraAction::Pan,
                    cursor,
                    Vec2::new(25.0, 10.0),
                    VIEWPORT,
                    OrbitMode::Turntable,
                    LIMITS,
                );
            }
            let moved = on_screen(&camera, picked) - at;
            assert!((moved - Vec2::new(100.0, 40.0)).length() < 0.1, "{moved}");
        }
    }

    /// The wheel zooms about the picked point; over background it zooms
    /// about the point of the focus plane under the cursor.
    #[test]
    fn the_wheel_zooms_about_what_is_under_the_cursor() {
        for start in cameras() {
            let mut camera = start.clone();
            let picked = Vec3::new(-0.3, 0.1, 0.2);
            let at = on_screen(&camera, picked);
            let mut navigator = Navigator::default();
            navigator.wheel(
                &mut camera,
                3.0,
                Cursor {
                    ndc: ndc_of(at),
                    picked: Some(picked),
                },
                VIEWPORT.x / VIEWPORT.y,
                LIMITS,
            );
            assert!(camera.distance < start.distance * 0.75);
            assert!((on_screen(&camera, picked) - at).length() < 0.1);

            // Background: the focus-plane point under the cursor stays.
            let mut camera = start.clone();
            let ndc = Vec2::new(0.5, 0.25);
            let on_plane = camera.point_at(ndc, VIEWPORT.x / VIEWPORT.y, camera.distance);
            let at = on_screen(&camera, on_plane);
            navigator.wheel(
                &mut camera,
                -2.0,
                Cursor { ndc, picked: None },
                VIEWPORT.x / VIEWPORT.y,
                LIMITS,
            );
            assert!(camera.distance > start.distance * 1.2);
            assert!((on_screen(&camera, on_plane) - at).length() < 0.1);
        }
    }

    /// A pick behind the eye is not an anchor; the focus plane is used.
    #[test]
    fn a_pick_behind_the_eye_falls_back_to_the_focus_plane() {
        let camera = Camera::default();
        let behind = camera.eye() - camera.forward();
        let cursor = Cursor {
            ndc: Vec2::ZERO,
            picked: Some(behind),
        };
        let (anchor, depth) = Navigator::anchor(&camera, cursor, 1.0);
        assert!((anchor - camera.focus).length() < 1e-5);
        assert_eq!(depth, camera.distance);
    }

    /// Changing the action mid-drag re-anchors; releasing ends the drag.
    #[test]
    fn a_changed_action_starts_a_new_gesture() {
        let mut camera = Camera::default();
        let mut navigator = Navigator::default();
        let first = Cursor {
            ndc: Vec2::ZERO,
            picked: Some(Vec3::new(0.1, 0.0, 0.0)),
        };
        let second = Cursor {
            ndc: Vec2::ZERO,
            picked: Some(Vec3::new(0.0, 0.0, 0.5)),
        };
        let drag = |navigator: &mut Navigator, camera: &mut Camera, action, cursor| {
            navigator.drag(
                camera,
                action,
                cursor,
                Vec2::ONE,
                VIEWPORT,
                OrbitMode::Free,
                LIMITS,
            )
        };
        drag(&mut navigator, &mut camera, CameraAction::Orbit, first);
        // Same action: the anchor stays where the gesture began.
        drag(&mut navigator, &mut camera, CameraAction::Orbit, second);
        assert_eq!(
            navigator.gesture.as_ref().unwrap().anchor,
            first.picked.unwrap()
        );
        drag(&mut navigator, &mut camera, CameraAction::Pan, second);
        assert_eq!(
            navigator.gesture.as_ref().unwrap().anchor,
            second.picked.unwrap()
        );
        assert!(!drag(
            &mut navigator,
            &mut camera,
            CameraAction::None,
            second
        ));
        assert!(navigator.gesture.is_none());
    }

    /// A transition eases to its target and lands on it exactly; a drag
    /// cancels it.
    #[test]
    fn transitions_land_exactly_and_yield_to_input() {
        let mut camera = Camera::default();
        let mut to = camera.clone();
        to.set_view(StandardView::Top);
        let mut navigator = Navigator::default();
        navigator.transition_to(&camera, to.clone());
        assert!(navigator.animating());
        navigator.tick(&mut camera, TRANSITION_SECONDS * 0.5);
        assert!(navigator.animating());
        assert!(camera.forward().z < Camera::default().forward().z - 0.05);
        assert!(camera.forward().z > -0.999);
        navigator.tick(&mut camera, TRANSITION_SECONDS);
        assert!(!navigator.animating());
        assert_eq!(camera, to);

        navigator.transition_to(&camera, Camera::default());
        navigator.drag(
            &mut camera,
            CameraAction::Pan,
            Cursor {
                ndc: Vec2::ZERO,
                picked: None,
            },
            Vec2::ONE,
            VIEWPORT,
            OrbitMode::Turntable,
            LIMITS,
        );
        assert!(!navigator.animating());
    }
}
