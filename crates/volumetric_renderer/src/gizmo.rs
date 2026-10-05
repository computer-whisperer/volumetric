//! The view gizmo: the world axes as the camera sees them, in a corner of
//! the viewport. It shows which way is up and takes clicks and drags.
//!
//! [`ViewGizmo::ends`] lays the gizmo out. Drawing (`gizmo.wgsl`) and
//! [`ViewGizmo::hit_test`] both read that layout, so what is clickable is
//! exactly what is drawn.

use glam::{Quat, Vec2, Vec3};

use crate::{AXIS_COLORS, StandardView};

/// Where the gizmo is drawn and how it looks this frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ViewGizmo {
    /// The camera-to-world rotation of the view it reflects
    /// ([`crate::Camera::orientation`]).
    pub orientation: Quat,
    /// Its centre in target pixels, origin top-left.
    pub center: Vec2,
    /// Its radius in target pixels.
    pub radius: f32,
    /// The part the pointer is over, drawn highlighted.
    pub hovered: Option<GizmoPart>,
}

/// A part of the gizmo a pointer can be over.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GizmoPart {
    /// The disc at one end of world axis `axis` (0 = X, 1 = Y, 2 = Z).
    End { axis: usize, positive: bool },
    /// Anywhere else inside the gizmo's circle.
    Body,
}

/// One end of an axis, laid out.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GizmoEnd {
    pub axis: usize,
    pub positive: bool,
    /// The disc's centre in target pixels.
    pub center: Vec2,
    /// The disc's radius in target pixels.
    pub radius: f32,
    /// How far the end points toward the viewer, -1 (straight away) to 1
    /// (straight at them).
    pub toward: f32,
}

impl GizmoEnd {
    /// The standard view that looks at the scene from this end.
    pub fn view(&self) -> StandardView {
        match (self.axis, self.positive) {
            (0, true) => StandardView::Right,
            (0, false) => StandardView::Left,
            (1, true) => StandardView::Back,
            (1, false) => StandardView::Front,
            (_, true) => StandardView::Top,
            (_, false) => StandardView::Bottom,
        }
    }
}

/// Disc radii as fractions of the gizmo's radius.
const POSITIVE_DISC: f32 = 0.23;
const NEGATIVE_DISC: f32 = 0.15;

impl ViewGizmo {
    /// The six ends, back to front: the order they are painted in.
    pub fn ends(&self) -> [GizmoEnd; 6] {
        let to_camera = self.orientation.inverse();
        let arm = self.radius * (1.0 - POSITIVE_DISC);
        let mut ends: [GizmoEnd; 6] = std::array::from_fn(|i| {
            let (axis, positive) = (i / 2, i % 2 == 0);
            let sign = if positive { 1.0 } else { -1.0 };
            // Camera coordinates: x right, y up, z toward the viewer.
            let seen: Vec3 = to_camera * (Vec3::AXES[axis] * sign);
            GizmoEnd {
                axis,
                positive,
                center: self.center + Vec2::new(seen.x, -seen.y) * arm,
                radius: self.radius
                    * if positive {
                        POSITIVE_DISC
                    } else {
                        NEGATIVE_DISC
                    },
                toward: seen.z,
            }
        });
        ends.sort_by(|a, b| a.toward.total_cmp(&b.toward));
        ends
    }

    /// What of the gizmo is under `at` (target pixels): the frontmost end
    /// whose disc holds it, else the body inside the gizmo's circle.
    pub fn hit_test(&self, at: Vec2) -> Option<GizmoPart> {
        if at.distance(self.center) > self.radius {
            return None;
        }
        let end = self
            .ends()
            .into_iter()
            .rev()
            .find(|end| at.distance(end.center) <= end.radius);
        Some(match end {
            Some(end) => GizmoPart::End {
                axis: end.axis,
                positive: end.positive,
            },
            None => GizmoPart::Body,
        })
    }

    /// The view a click on `part` asks for: the one looking from that end,
    /// or from the opposite end when the camera is already there, so a
    /// second click flips the view.
    pub fn view_for(&self, part: GizmoPart) -> Option<StandardView> {
        let GizmoPart::End { axis, positive } = part else {
            return None;
        };
        let ends = self.ends();
        let find = |positive: bool| {
            ends.iter()
                .find(|end| end.axis == axis && end.positive == positive)
                .expect("every axis has both ends")
        };
        let end = find(positive);
        let already_there = end.toward > 0.999;
        Some(if already_there { find(!positive) } else { end }.view())
    }

    /// The colour an end is drawn in: its axis's, dimmed where it points
    /// away from the viewer.
    pub(crate) fn end_color(end: &GizmoEnd) -> [f32; 3] {
        let shade = if end.toward < 0.0 { 0.72 } else { 1.0 };
        AXIS_COLORS[end.axis].map(|c| c * shade)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Camera;

    fn gizmo(view: StandardView) -> ViewGizmo {
        ViewGizmo {
            orientation: view.orientation(),
            center: Vec2::new(500.0, 80.0),
            radius: 60.0,
            hovered: None,
        }
    }

    /// Each end names the standard view that puts it at the centre of the
    /// gizmo, facing the viewer.
    #[test]
    fn an_end_names_the_view_that_faces_it() {
        for view in StandardView::ALL {
            if view == StandardView::Isometric {
                continue;
            }
            let gizmo = gizmo(view);
            let front = gizmo.ends()[5];
            assert_eq!(front.view(), view);
            assert!(front.toward > 0.999);
            assert!(front.center.distance(gizmo.center) < 1e-3);
        }
    }

    /// The layout follows the camera: from the default three-quarter view
    /// +X points right, +Y points right and away, +Z points up the screen.
    #[test]
    fn the_layout_follows_the_camera() {
        let gizmo = ViewGizmo {
            orientation: Camera::default().orientation,
            ..gizmo(StandardView::Front)
        };
        let end = |axis: usize| {
            *gizmo
                .ends()
                .iter()
                .find(|end| end.axis == axis && end.positive)
                .unwrap()
        };
        assert!(end(0).center.x > gizmo.center.x);
        assert!(end(0).toward > 0.0);
        assert!(end(1).center.x > gizmo.center.x);
        assert!(end(1).toward < 0.0);
        assert!(end(2).center.y < gizmo.center.y);
        // Painted back to front.
        let ends = gizmo.ends();
        assert!(ends.windows(2).all(|pair| pair[0].toward <= pair[1].toward));
    }

    /// A point is over the frontmost disc that holds it, the body inside
    /// the circle otherwise, and nothing outside.
    #[test]
    fn hit_tests_read_the_layout() {
        let gizmo = gizmo(StandardView::Front);
        // From the front -Y faces the viewer at the centre, over +Y.
        assert_eq!(
            gizmo.hit_test(gizmo.center),
            Some(GizmoPart::End {
                axis: 1,
                positive: false
            })
        );
        for end in gizmo.ends() {
            if end.axis == 1 {
                continue;
            }
            assert_eq!(
                gizmo.hit_test(end.center),
                Some(GizmoPart::End {
                    axis: end.axis,
                    positive: end.positive
                })
            );
        }
        let between = gizmo.center + Vec2::new(0.4, 0.4) * gizmo.radius;
        assert_eq!(gizmo.hit_test(between), Some(GizmoPart::Body));
        let outside = gizmo.center + Vec2::new(1.1, 0.0) * gizmo.radius;
        assert_eq!(gizmo.hit_test(outside), None);
    }

    /// Clicking an end goes to its view; clicking the end already faced
    /// goes to the opposite one.
    #[test]
    fn a_click_on_the_faced_end_flips_the_view() {
        let gizmo = gizmo(StandardView::Top);
        let top = GizmoPart::End {
            axis: 2,
            positive: true,
        };
        let right = GizmoPart::End {
            axis: 0,
            positive: true,
        };
        assert_eq!(gizmo.view_for(top), Some(StandardView::Bottom));
        assert_eq!(gizmo.view_for(right), Some(StandardView::Right));
        assert_eq!(gizmo.view_for(GizmoPart::Body), None);
    }
}
