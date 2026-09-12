//! Views (posed cameras with intrinsics) in the scene: the frustum entity a
//! ViewSet draws as, the frame the viewport takes when looking through one
//! view, and the highlight of that view among the others.

use glam::{Mat4, Vec3, Vec4};
use volumetric::viewset::{CameraModel, View, ViewSet, decode_viewset};
use volumetric_renderer as renderer;
use volumetric_renderer::{CameraView, Pinhole};

use crate::{OutputStats, PreviewBounds, PreviewEntity, PreviewRequest};

/// How far from the eye a view's image rectangle is drawn, in metres.
pub const FRUSTUM_DEPTH_M: f64 = 0.1;

/// Frustum and marker palette: views in cyan, the looked-through view and
/// the markers in amber (the same amber as the bounds box and normals).
const VIEW_COLOR: [f32; 4] = [0.35, 0.85, 0.95, 0.9];
const VIEW_HIGHLIGHT_COLOR: [f32; 4] = [1.0, 0.72, 0.2, 1.0];
const MARKER_COLOR: [f32; 4] = [1.0, 0.72, 0.2, 0.85];
/// Observed swatches in look-through: amber like the map's markers.
const SWATCH_OBS_COLOR: [f32; 4] = [1.0, 0.72, 0.2, 0.95];
/// Observed card tags in look-through.
const TAG_OBS_COLOR: [f32; 4] = [0.35, 0.85, 0.95, 0.95];
/// Observed card corners in look-through.
const CORNER_OBS_COLOR: [f32; 4] = [1.0, 0.3, 1.0, 0.95];
/// Half-size of a corner's cross, pixels of the photograph.
const CORNER_CROSS_PX: f64 = 6.0;

fn v3(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

fn segment(start: Vec3, end: Vec3, color: [f32; 4]) -> renderer::LineSegment {
    renderer::LineSegment {
        start: start.to_array(),
        end: end.to_array(),
        color,
    }
}

/// The world point a pixel sees at `depth`, with the lateral extent clamped
/// to about 76 degrees off axis so a fisheye's undistorted corners stay on
/// the picture.
fn corner_at_depth(view: &View, camera: &CameraModel, pixel: [f64; 2], depth: f64) -> Vec3 {
    let [x, y, z] = camera.point_at_depth(pixel, depth);
    let limit = depth * 4.0;
    let clamped = [x.clamp(-limit, limit), y.clamp(-limit, limit), z];
    v3(view.to_world(clamped).expect("a posed view"))
}

/// A view's frustum: the four edges from the eye to the image corners at
/// `depth`, the image rectangle, and a tick off the middle of its top edge
/// showing which way is up in the picture. An unposed view has none.
pub fn frustum_segments(
    view: &View,
    camera: &CameraModel,
    depth: f64,
    color: [f32; 4],
) -> Vec<renderer::LineSegment> {
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    let Some(eye) = view.position() else {
        return Vec::new();
    };
    let eye = v3(eye);
    let corners =
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].map(|px| corner_at_depth(view, camera, px, depth));
    let mut out = Vec::with_capacity(9);
    for corner in corners {
        out.push(segment(eye, corner, color));
    }
    for i in 0..4 {
        out.push(segment(corners[i], corners[(i + 1) % 4], color));
    }
    let top = (corners[0] + corners[1]) * 0.5;
    let bottom = (corners[2] + corners[3]) * 0.5;
    out.push(segment(top, top + (top - bottom) * 0.15, color));
    out
}

/// The looked-through view's frustum, in the highlight colour.
pub fn highlight_lines(view: &View, camera: &CameraModel) -> renderer::LineData {
    renderer::LineData {
        segments: frustum_segments(view, camera, FRUSTUM_DEPTH_M, VIEW_HIGHLIGHT_COLOR),
    }
}

/// What detection found in the photograph, as lines on the view's
/// picture plane at the frustum's depth: each marker's quad (swatches
/// amber, the card's tags cyan) and a cross at each card corner. Looked
/// through the view they land on the photograph's own cards, which is
/// the visual check of the detection; from elsewhere they sit in the
/// frustum's rectangle.
pub fn observation_lines(view: &View, camera: &CameraModel) -> Vec<renderer::LineSegment> {
    let Some(obs) = &view.observations else {
        return Vec::new();
    };
    if view.pose().is_none() {
        return Vec::new();
    }
    let at = |px: [f64; 2]| corner_at_depth(view, camera, px, FRUSTUM_DEPTH_M);
    let mut out = Vec::with_capacity(4 * obs.markers.len() + 2 * obs.board.len());
    for m in &obs.markers {
        let color = if m.family == "36h11" {
            TAG_OBS_COLOR
        } else {
            SWATCH_OBS_COLOR
        };
        for i in 0..4 {
            out.push(segment(at(m.corners[i]), at(m.corners[(i + 1) % 4]), color));
        }
    }
    for c in &obs.board {
        let [x, y] = c.pixel;
        let r = CORNER_CROSS_PX;
        out.push(segment(at([x - r, y]), at([x + r, y]), CORNER_OBS_COLOR));
        out.push(segment(at([x, y - r]), at([x, y + r]), CORNER_OBS_COLOR));
    }
    out
}

/// Draws a view's highlight over everything, per frame.
pub fn submit_view_highlight(renderer: &mut renderer::Renderer, lines: &renderer::LineData) {
    renderer.submit_lines(
        lines,
        Mat4::IDENTITY,
        renderer::LineStyle {
            width: 2.5,
            width_mode: renderer::WidthMode::ScreenSpace,
            pattern: renderer::LinePattern::Solid,
            depth_mode: renderer::DepthMode::Overlay,
        },
    );
}

fn extend(bounds: &mut Option<PreviewBounds>, p: Vec3) {
    let point = PreviewBounds {
        min: (p.x, p.y, p.z),
        max: (p.x, p.y, p.z),
    };
    *bounds = Some(match *bounds {
        Some(b) => b.union(point),
        None => point,
    });
}

/// Preview for a ViewSet value: every view's frustum and eye, and every
/// marker's square, as retained lines and points. The bounds enclose the
/// eyes and image corners, so framing the scene includes the cameras.
pub(crate) fn build_viewset_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let set = decode_viewset(request.data.as_slice())?;
    let mut segments = Vec::with_capacity(set.views.len() * 9 + set.markers.len() * 4);
    let mut points = Vec::with_capacity(set.views.len());
    let mut bounds: Option<PreviewBounds> = None;
    for view in &set.views {
        let camera = set.camera_of(view);
        let frustum = frustum_segments(view, camera, FRUSTUM_DEPTH_M, VIEW_COLOR);
        for s in &frustum {
            extend(&mut bounds, Vec3::from(s.start));
            extend(&mut bounds, Vec3::from(s.end));
        }
        segments.extend(frustum);
        if let Some(eye) = view.position() {
            points.push(renderer::PointInstance {
                position: v3(eye).to_array(),
                color: VIEW_COLOR,
            });
        }
    }
    for marker in &set.markers {
        let corners = marker.corners.map(v3);
        for i in 0..4 {
            extend(&mut bounds, corners[i]);
            segments.push(segment(corners[i], corners[(i + 1) % 4], MARKER_COLOR));
        }
    }

    let mut scene = renderer::SceneData::new();
    scene.add_lines(
        renderer::LineData { segments },
        Mat4::IDENTITY,
        renderer::LineStyle {
            width: 1.5,
            width_mode: renderer::WidthMode::ScreenSpace,
            pattern: renderer::LinePattern::Solid,
            depth_mode: renderer::DepthMode::Normal,
        },
    );
    scene.add_points(
        renderer::PointData { points },
        Mat4::IDENTITY,
        renderer::PointStyle {
            size: 6.0,
            size_mode: renderer::WidthMode::ScreenSpace,
            shape: renderer::PointShape::Circle,
            depth_mode: renderer::DepthMode::Normal,
        },
    );

    let stats = OutputStats {
        detail: viewset_detail(&set),
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };
    Ok(PreviewEntity {
        scene,
        bounds: bounds.unwrap_or(PreviewBounds {
            min: (-1.0, -1.0, -1.0),
            max: (1.0, 1.0, 1.0),
        }),
        stats,
        wireframe_lines: None,
        subspace: None,
    })
}

/// The lines the viewport's statistics show for a view set: counts and
/// the provenance labels that are known.
pub fn viewset_detail(set: &ViewSet) -> Vec<String> {
    let with = |f: fn(&View) -> bool| set.views.iter().filter(|v| f(v)).count();
    let mut detail = vec![
        format!(
            "views: {} ({} with image, {} with depth, {} with mask)",
            set.views.len(),
            with(|v| v.image.is_some()),
            with(|v| v.depth.is_some()),
            with(|v| v.mask.is_some()),
        ),
        format!(
            "cameras: {}; markers: {}",
            set.cameras.len(),
            set.markers.len()
        ),
    ];
    let p = &set.provenance;
    let labels: Vec<String> = [
        ("session", &p.session),
        ("rig", &p.rig),
        ("field", &p.field),
        ("setup", &p.setup),
    ]
    .into_iter()
    .filter(|(_, value)| !value.is_empty())
    .map(|(name, value)| format!("{name} {value}"))
    .collect();
    if !labels.is_empty() {
        detail.push(labels.join(" · "));
    }
    detail
}

/// One view's camera as the viewport takes it: the pinhole at the image's
/// own size and the camera-to-world pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewFrame {
    pub pinhole: Pinhole,
    pub camera_to_world: Mat4,
}

impl ViewFrame {
    /// `None` for an unposed view: there is nowhere to look from.
    pub fn of(view: &View, camera: &CameraModel) -> Option<Self> {
        Some(Self {
            pinhole: Pinhole {
                fx: camera.fx as f32,
                fy: camera.fy as f32,
                cx: camera.cx as f32,
                cy: camera.cy as f32,
                width: camera.width,
                height: camera.height,
            },
            camera_to_world: pose_matrix(view.pose()?),
        })
    }

    pub fn eye(&self) -> Vec3 {
        self.camera_to_world.transform_point3(Vec3::ZERO)
    }

    /// The optical axis (OpenCV +z), in world space.
    pub fn forward(&self) -> Vec3 {
        self.camera_to_world.transform_vector3(Vec3::Z).normalize()
    }

    /// The picture's up (OpenCV -y), in world space.
    pub fn up(&self) -> Vec3 {
        self.camera_to_world
            .transform_vector3(Vec3::NEG_Y)
            .normalize()
    }

    /// The pinhole for a `width` x `height` frame: the camera's image scaled
    /// uniformly to fit and centred, so the picture sits in a letterbox and
    /// the projection stays true.
    pub fn letterboxed(&self, width: u32, height: u32) -> Pinhole {
        let (cw, ch) = (self.pinhole.width as f32, self.pinhole.height as f32);
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        let scale = (w / cw).min(h / ch);
        let ox = (w - cw * scale) * 0.5;
        let oy = (h - ch * scale) * 0.5;
        Pinhole {
            fx: self.pinhole.fx * scale,
            fy: self.pinhole.fy * scale,
            cx: self.pinhole.cx * scale + ox,
            cy: self.pinhole.cy * scale + oy,
            width: width.max(1),
            height: height.max(1),
        }
    }

    /// The near and far planes that keep `bounds` in view from the eye.
    pub fn clip_planes(&self, bounds: PreviewBounds) -> (f32, f32) {
        clip_planes_for(
            self.eye(),
            self.forward(),
            bounds.min_vec3(),
            bounds.max_vec3(),
        )
    }

    /// The view and projection for a `width` x `height` frame that keeps
    /// `bounds` between the clip planes.
    pub fn camera_view(&self, width: u32, height: u32, bounds: PreviewBounds) -> CameraView {
        let (near, far) = self.clip_planes(bounds);
        CameraView::pinhole(
            &self.letterboxed(width, height),
            self.camera_to_world,
            near,
            far,
        )
    }
}

/// A row-major 3x4 camera-to-world pose as a matrix.
pub fn pose_matrix(m: &[f64; 12]) -> Mat4 {
    let m: Vec<f32> = m.iter().map(|v| *v as f32).collect();
    Mat4::from_cols(
        Vec4::new(m[0], m[4], m[8], 0.0),
        Vec4::new(m[1], m[5], m[9], 0.0),
        Vec4::new(m[2], m[6], m[10], 0.0),
        Vec4::new(m[3], m[7], m[11], 1.0),
    )
}

/// Clip planes for a camera at `eye` looking along `forward`: half the
/// nearest bounds corner's depth to twice the farthest, kept off zero by
/// the bounds extent.
pub fn clip_planes_for(eye: Vec3, forward: Vec3, min: Vec3, max: Vec3) -> (f32, f32) {
    let extent = (max - min).length().max(1e-6);
    let (mut nearest, mut farthest) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..8 {
        let corner = Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        );
        let depth = (corner - eye).dot(forward);
        nearest = nearest.min(depth);
        farthest = farthest.max(depth);
    }
    let near = (nearest * 0.5).max(extent * 1e-3);
    let far = (farthest * 2.0).max(near * 10.0);
    (near, far)
}

/// What the viewport needs to look through one view: its frame and the
/// frustum to highlight.
#[derive(Clone)]
pub struct LookThrough {
    pub frame: ViewFrame,
    /// The highlighted frustum and the view's observations, drawn over
    /// everything.
    pub frustum: renderer::LineData,
}

impl LookThrough {
    /// `None` for an unposed view.
    pub fn of(view: &View, camera: &CameraModel) -> Option<Self> {
        let frame = ViewFrame::of(view, camera)?;
        let mut frustum = highlight_lines(view, camera);
        frustum.segments.extend(observation_lines(view, camera));
        Some(Self { frame, frustum })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use volumetric::AssetTypeHint;
    use volumetric::viewset::{Marker, Provenance, WorldFrame, encode_viewset};

    #[test]
    fn observations_draw_on_the_picture_plane() {
        use volumetric::viewset::{CornerObs, MarkerObs, Observations};
        let camera = CameraModel::pinhole(640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut view = looking_away();
        assert!(observation_lines(&view, &camera).is_empty());
        view.observations = Some(Observations {
            markers: vec![
                MarkerObs {
                    id: 3,
                    family: "5x5_100".to_string(),
                    corners: [
                        [100.0, 100.0],
                        [200.0, 100.0],
                        [200.0, 200.0],
                        [100.0, 200.0],
                    ],
                    fit_px: 0.2,
                },
                MarkerObs {
                    id: 100,
                    family: "36h11".to_string(),
                    corners: [
                        [300.0, 300.0],
                        [340.0, 300.0],
                        [340.0, 340.0],
                        [300.0, 340.0],
                    ],
                    fit_px: 0.2,
                },
            ],
            board: vec![CornerObs {
                id: 0,
                pixel: [400.5, 120.5],
                fit_px: 0.3,
            }],
            blur_px: None,
            features: Vec::new(),
            contours: Vec::new(),
        });
        let lines = observation_lines(&view, &camera);
        assert_eq!(lines.len(), 4 + 4 + 2);
        // Every segment end projects back to the pixel it came from.
        let world = |p: [f32; 3]| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
        let back = view.project(&camera, world(lines[0].start)).unwrap();
        assert!(
            (back[0] - 100.0).abs() < 1e-3 && (back[1] - 100.0).abs() < 1e-3,
            "{back:?}"
        );
        let back = view.project(&camera, world(lines[8].start)).unwrap();
        assert!(
            (back[0] - (400.5 - CORNER_CROSS_PX)).abs() < 1e-3,
            "{back:?}"
        );
        assert_eq!(lines[0].color, SWATCH_OBS_COLOR);
        assert_eq!(lines[4].color, TAG_OBS_COLOR);
        assert_eq!(lines[8].color, CORNER_OBS_COLOR);
        // Look-through carries them with the frustum.
        let look = LookThrough::of(&view, &camera).unwrap();
        assert_eq!(look.frustum.segments.len(), 9 + 10);
        // Unposed, the view draws nothing and cannot be looked through.
        let mut raw = view.clone();
        raw.camera_to_world = None;
        assert!(observation_lines(&raw, &camera).is_empty());
        assert!(frustum_segments(&raw, &camera, 1.0, VIEW_COLOR).is_empty());
        assert!(LookThrough::of(&raw, &camera).is_none());
    }

    /// A camera at z = -2 looking along world -z, with the picture's up
    /// along world +y (OpenCV: camera y is down, so the pose flips y).
    fn looking_away() -> View {
        View::posed(
            "v",
            0,
            [
                1.0, 0.0, 0.0, 0.0, //
                0.0, -1.0, 0.0, 0.0, //
                0.0, 0.0, -1.0, -2.0, //
            ],
        )
    }

    fn set() -> ViewSet {
        let mut view = looking_away();
        view.image = Some(vec![1, 2, 3]);
        ViewSet {
            board: None,
            schema: 2,
            world: WorldFrame::default(),
            provenance: Provenance {
                field: "cards".to_string(),
                ..Default::default()
            },
            cameras: vec![CameraModel::pinhole(400, 300, 400.0, 400.0, 200.0, 150.0)],
            views: vec![view],
            markers: vec![Marker {
                id: 7,
                size_m: 0.1,
                corners: [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.1, 0.1, 0.0],
                    [0.0, 0.1, 0.0],
                ],
            }],
        }
    }

    #[test]
    fn frustum_has_the_eye_corners_and_up_tick() {
        let set = set();
        let (view, camera) = set.view("v").unwrap();
        let segments = frustum_segments(view, camera, 0.1, VIEW_COLOR);
        assert_eq!(segments.len(), 9);
        let eye = Vec3::new(0.0, 0.0, -2.0);
        for s in &segments[..4] {
            assert_eq!(Vec3::from(s.start), eye);
            let corner = Vec3::from(s.end);
            // Corners sit 0.1 m along the axis (world -z), half the image
            // width and height off it at focal length 400.
            assert!((corner.z - (-2.1)).abs() < 1e-5, "{corner:?}");
            assert!((corner.x.abs() - 0.05).abs() < 1e-5, "{corner:?}");
            assert!((corner.y.abs() - 0.0375).abs() < 1e-5, "{corner:?}");
        }
        // The up tick leaves the top edge (image y = 0 maps to world +y
        // through the flipped pose) and points further up.
        let tick = &segments[8];
        assert!(
            tick.start[1] > 0.0 && tick.end[1] > tick.start[1],
            "{tick:?}"
        );
    }

    #[test]
    fn viewset_entity_carries_lines_points_and_bounds() {
        let set = set();
        let request = PreviewRequest {
            asset_id: "views".to_string(),
            source_hash: [0; 32],
            data: Arc::new(encode_viewset(&set)),
            type_hint: Some(AssetTypeHint::ViewSet),
            precursor_ids: Vec::new(),
            plan: crate::PreviewPlan::ViewSet,
            wireframe: false,
            show_grid: false,
            show_bounds: false,
            ssao: false,
            ssao_radius: 0.5,
            ssao_bias: 0.025,
            ssao_strength: 1.0,
            stale: false,
        };
        let entity = build_viewset_preview(&request, web_time::Instant::now()).unwrap();
        assert_eq!(entity.scene.lines[0].0.segments.len(), 9 + 4);
        assert_eq!(entity.scene.points[0].0.points.len(), 1);
        assert!((entity.bounds.min.2 - (-2.1)).abs() < 1e-5);
        assert!((entity.bounds.max.2 - 0.0).abs() < 1e-5);
        assert!((entity.bounds.max.0 - 0.1).abs() < 1e-5);
        assert_eq!(
            entity.stats.detail,
            vec![
                "views: 1 (1 with image, 0 with depth, 0 with mask)".to_string(),
                "cameras: 1; markers: 1".to_string(),
                "field cards".to_string(),
            ]
        );
    }

    #[test]
    fn letterbox_keeps_the_projection_true() {
        let set = set();
        let (view, camera) = set.view("v").unwrap();
        let frame = ViewFrame::of(view, camera).unwrap();
        assert_eq!(frame.eye(), Vec3::new(0.0, 0.0, -2.0));
        assert_eq!(frame.forward(), Vec3::NEG_Z);
        assert_eq!(frame.up(), Vec3::Y);

        // A 1000 x 300 viewport: the 400 x 300 picture scales by 1 and
        // sits 300 px in.
        let boxed = frame.letterboxed(1000, 300);
        assert_eq!((boxed.fx, boxed.fy), (400.0, 400.0));
        assert_eq!((boxed.cx, boxed.cy), (500.0, 150.0));
        // A 200 x 600 viewport: scale 0.5, the picture is 200 x 150 at
        // y = 225.
        let boxed = frame.letterboxed(200, 600);
        assert_eq!((boxed.fx, boxed.cx, boxed.cy), (200.0, 100.0, 300.0));

        // A world point that lands on camera pixel (300, 150) lands on
        // the same picture pixel scaled and offset in the viewport.
        let world = [0.5, 0.0, -4.0];
        assert_eq!(view.project(camera, world), Some([300.0, 150.0]));
        let bounds = PreviewBounds {
            min: (-1.0, -1.0, -5.0),
            max: (1.0, 1.0, -3.0),
        };
        let cv = frame.camera_view(1000, 300, bounds);
        let px = cv
            .project(Vec3::from(world.map(|v| v as f32)), 1000, 300)
            .unwrap();
        assert!(
            (px.x - 600.0).abs() < 1e-2 && (px.y - 150.0).abs() < 1e-2,
            "{px:?}"
        );

        let (near, far) = frame.clip_planes(bounds);
        assert!(near > 0.0 && near < 1.0 && far > 3.0, "{near} {far}");
    }
}
