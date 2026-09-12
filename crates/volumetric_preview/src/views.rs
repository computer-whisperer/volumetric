//! Views (posed cameras with intrinsics) in the scene: the frustum entity a
//! ViewSet draws as, the frame the viewport takes when looking through one
//! view, and the highlight of that view among the others.

use glam::{Mat4, Vec3, Vec4};
use volumetric::viewset::{CameraModel, Distortion, PickRole, View, ViewSet, decode_viewset};
use volumetric_renderer as renderer;
use volumetric_renderer::{CameraView, Pinhole, Warp};

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
/// A recorded fit pick: an upright cross in green.
const PICK_FIT_COLOR: [f32; 4] = [0.4, 1.0, 0.45, 1.0];
/// A recorded check pick: a diagonal cross in orange, so a held-out
/// pick reads differently from one the feature is fitted from.
const PICK_CHECK_COLOR: [f32; 4] = [1.0, 0.5, 0.15, 1.0];
/// A recorded contour, as a polyline through its pixels.
const CONTOUR_COLOR: [f32; 4] = [0.4, 1.0, 0.45, 0.9];
/// Half-size of a pick's cross as a fraction of the picture's width, with
/// a floor in pixels: a pick is one of a few and is what the eye is asked
/// to judge, so it stays legible when a 6000-pixel still is shown at
/// screen size (40 px on a DSLR frame, 8 px on a webcam's).
const PICK_CROSS_FRACTION: f64 = 1.0 / 150.0;
const PICK_CROSS_MIN_PX: f64 = 8.0;

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

/// A name drawn beside a mark in the photograph: the GUI sets it as
/// text over the viewport, the headless frame draws it into the pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkLabel {
    pub text: String,
    /// The label's left edge, vertically centred, in the photograph's
    /// pixels: just right of the cross, or at a contour's first point.
    pub pixel: [f64; 2],
    pub kind: MarkKind,
}

/// What a mark is, for its colour and shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkKind {
    Fit,
    Check,
    Contour,
}

impl MarkKind {
    /// The mark's colour in linear RGBA, as the lines are drawn.
    pub fn color(self) -> [f32; 4] {
        match self {
            MarkKind::Fit => PICK_FIT_COLOR,
            MarkKind::Check => PICK_CHECK_COLOR,
            MarkKind::Contour => CONTOUR_COLOR,
        }
    }

    /// The mark's colour as 8-bit sRGB, for text drawn into pixels.
    pub fn srgb8(self) -> [u8; 3] {
        let c = self.color();
        [c[0], c[1], c[2]].map(|v| (v.max(0.0).powf(1.0 / 2.2) * 255.0).round() as u8)
    }
}

/// Half-size of a pick's cross in the photograph's pixels.
pub fn pick_cross_px(camera: &CameraModel) -> f64 {
    (f64::from(camera.width) * PICK_CROSS_FRACTION).max(PICK_CROSS_MIN_PX)
}

/// The names of a view's recorded picks and contours, placed beside
/// their marks. Nothing for an unposed view or one without observations.
pub fn mark_labels(view: &View, camera: &CameraModel) -> Vec<MarkLabel> {
    let Some(obs) = &view.observations else {
        return Vec::new();
    };
    if view.pose().is_none() {
        return Vec::new();
    }
    let gap = pick_cross_px(camera) * 1.3;
    let mut out = Vec::with_capacity(obs.features.len() + obs.contours.len());
    for f in &obs.features {
        out.push(MarkLabel {
            text: f.name.clone(),
            pixel: [f.pixel[0] + gap, f.pixel[1]],
            kind: match f.role {
                PickRole::Fit => MarkKind::Fit,
                PickRole::Check => MarkKind::Check,
            },
        });
    }
    for c in &obs.contours {
        if let Some(first) = c.pixels.first() {
            out.push(MarkLabel {
                text: c.name.clone(),
                pixel: [first[0] + gap * 0.5, first[1]],
                kind: MarkKind::Contour,
            });
        }
    }
    out
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
    // Recorded picks: a fit pick is an upright cross, a check pick a
    // diagonal one, each with a gap at the centre so the pixel picked
    // stays visible under it.
    for f in &obs.features {
        let [x, y] = f.pixel;
        let r = pick_cross_px(camera);
        let gap = r * 0.25;
        let (color, arms): ([f32; 4], [[[f64; 2]; 2]; 4]) = match f.role {
            PickRole::Fit => (
                PICK_FIT_COLOR,
                [
                    [[x - r, y], [x - gap, y]],
                    [[x + gap, y], [x + r, y]],
                    [[x, y - r], [x, y - gap]],
                    [[x, y + gap], [x, y + r]],
                ],
            ),
            PickRole::Check => {
                let (d, g) = (
                    r * std::f64::consts::FRAC_1_SQRT_2,
                    gap * std::f64::consts::FRAC_1_SQRT_2,
                );
                (
                    PICK_CHECK_COLOR,
                    [
                        [[x - d, y - d], [x - g, y - g]],
                        [[x + g, y + g], [x + d, y + d]],
                        [[x - d, y + d], [x - g, y + g]],
                        [[x + g, y - g], [x + d, y - d]],
                    ],
                )
            }
        };
        for [a, b] in arms {
            out.push(segment(at(a), at(b), color));
        }
    }
    for c in &obs.contours {
        for pair in c.pixels.windows(2) {
            out.push(segment(at(pair[0]), at(pair[1]), CONTOUR_COLOR));
        }
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
        mesh_keys: Vec::new(),
        articulated: None,
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
/// own size, the lens it was shot through, and the camera-to-world pose.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewFrame {
    pub pinhole: Pinhole,
    pub camera: CameraModel,
    pub camera_to_world: Mat4,
}

/// A frame drawn through a view's lens: the pinhole the scene renders
/// through, and the warp that bends that render into the photograph's
/// own pixels. `warp` is `None` for an ideal lens, when the pinhole is
/// the output frame itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Framed {
    pub pinhole: Pinhole,
    pub warp: Option<Warp>,
}

/// Rays further off axis than this, in normalised camera coordinates
/// (about 76 degrees), have no pinhole image: a fisheye's rim.
const LENS_LIMIT: f64 = 4.0;
/// Output pixels between warp grid samples; the lens is smooth at this
/// scale, so bilinear interpolation between samples is exact to well
/// under a pixel.
const WARP_STEP_PX: u32 = 16;
/// Points along each edge of the output frame whose undistorted images
/// bound the pinhole frame.
const WARP_EDGE_SAMPLES: u32 = 64;

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
            camera: camera.clone(),
            camera_to_world: pose_matrix(view.pose()?),
        })
    }

    /// The frame for a `width` x `height` output with the picture
    /// letterboxed in it (the GUI's `Contain` fit), drawn through the lens.
    pub fn framed(&self, width: u32, height: u32) -> Framed {
        let (cw, ch) = (
            f64::from(self.pinhole.width),
            f64::from(self.pinhole.height),
        );
        let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
        let scale = (w / cw).min(h / ch);
        let ox = (w - cw * scale) * 0.5;
        let oy = (h - ch * scale) * 0.5;
        self.framed_with(width.max(1), height.max(1), [scale, scale], [ox, oy])
    }

    /// The frame for a `width` x `height` output with the picture
    /// stretched to fill it (the headless render's scaled frame), drawn
    /// through the lens.
    pub fn framed_stretched(&self, width: u32, height: u32) -> Framed {
        let (cw, ch) = (
            f64::from(self.pinhole.width),
            f64::from(self.pinhole.height),
        );
        let (w, h) = (f64::from(width.max(1)), f64::from(height.max(1)));
        self.framed_with(width.max(1), height.max(1), [w / cw, h / ch], [0.0, 0.0])
    }

    /// The frame for an output in which camera pixel `q` sits at `q *
    /// scale + offset`. An ideal lens gives that pinhole and no warp. A
    /// real lens gives the pinhole of the *overscan* frame, the ideal
    /// image of the output's boundary, and the warp from the output's
    /// pixels into it.
    fn framed_with(&self, width: u32, height: u32, scale: [f64; 2], offset: [f64; 2]) -> Framed {
        let camera = &self.camera;
        // Camera pixel `q` sits at `(q - origin) * scale + shift` in a
        // frame: the output has the letterbox shift, the overscan frame
        // its own origin and no shift.
        let scaled =
            |frame_width: u32, frame_height: u32, origin: [f64; 2], shift: [f64; 2]| Pinhole {
                fx: (camera.fx * scale[0]) as f32,
                fy: (camera.fy * scale[1]) as f32,
                cx: ((camera.cx - origin[0]) * scale[0] + shift[0]) as f32,
                cy: ((camera.cy - origin[1]) * scale[1] + shift[1]) as f32,
                width: frame_width,
                height: frame_height,
            };
        if camera.distortion == Distortion::None {
            return Framed {
                pinhole: scaled(width, height, [0.0, 0.0], offset),
                warp: None,
            };
        }
        // The ideal pinhole pixel an output pixel's ray goes through, in
        // camera pixel units; `None` off the lens's limit or where the
        // lens model does not invert cleanly (far outside the picture).
        let ideal = |x: f64, y: f64| -> Option<[f64; 2]> {
            let q = [(x - offset[0]) / scale[0], (y - offset[1]) / scale[1]];
            let [xn, yn, _] = camera.point_at_depth(q, 1.0);
            if !(xn.is_finite() && yn.is_finite()) || xn.abs() > LENS_LIMIT || yn.abs() > LENS_LIMIT
            {
                return None;
            }
            let back = camera.project([xn, yn, 1.0])?;
            if (back[0] - q[0]).abs() > 0.05 || (back[1] - q[1]).abs() > 0.05 {
                return None;
            }
            Some([camera.fx * xn + camera.cx, camera.fy * yn + camera.cy])
        };
        // The overscan: the ideal image of the output's boundary, padded
        // a pixel, in camera pixels.
        let (w, h) = (f64::from(width), f64::from(height));
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for i in 0..=WARP_EDGE_SAMPLES {
            let t = f64::from(i) / f64::from(WARP_EDGE_SAMPLES);
            for p in [[t * w, 0.0], [t * w, h], [0.0, t * h], [w, t * h]] {
                if let Some(u) = ideal(p[0], p[1]) {
                    min = [min[0].min(u[0]), min[1].min(u[1])];
                    max = [max[0].max(u[0]), max[1].max(u[1])];
                }
            }
        }
        // A boundary wholly beyond the lens's limit (a fisheye wider than
        // the limit) leaves the overscan at the limit itself; a partly
        // mapped one is clipped to it.
        let limit = [
            [
                camera.cx - camera.fx * LENS_LIMIT,
                camera.cy - camera.fy * LENS_LIMIT,
            ],
            [
                camera.cx + camera.fx * LENS_LIMIT,
                camera.cy + camera.fy * LENS_LIMIT,
            ],
        ];
        if !(min[0].is_finite() && max[0].is_finite() && min[1].is_finite() && max[1].is_finite()) {
            (min, max) = limit.into();
        }
        min = [min[0].max(limit[0][0]), min[1].max(limit[0][1])];
        max = [max[0].min(limit[1][0]), max[1].min(limit[1][1])];
        let origin = [min[0] - 1.0, min[1] - 1.0];
        let source = (
            ((max[0] + 1.0 - origin[0]) * scale[0]).ceil().max(1.0) as u32,
            ((max[1] + 1.0 - origin[1]) * scale[1]).ceil().max(1.0) as u32,
        );
        let columns = width.div_ceil(WARP_STEP_PX) + 1;
        let rows = height.div_ceil(WARP_STEP_PX) + 1;
        let mut grid = Vec::with_capacity((columns * rows) as usize);
        for j in 0..rows {
            let y = f64::from(j) * h / f64::from(rows - 1);
            for i in 0..columns {
                let x = f64::from(i) * w / f64::from(columns - 1);
                grid.push(match ideal(x, y) {
                    Some(u) => [
                        ((u[0] - origin[0]) * scale[0]) as f32,
                        ((u[1] - origin[1]) * scale[1]) as f32,
                    ],
                    None => [-1.0, -1.0],
                });
            }
        }
        Framed {
            pinhole: scaled(source.0, source.1, origin, [0.0, 0.0]),
            warp: Some(Warp {
                source,
                output: (width, height),
                columns,
                rows,
                grid,
            }),
        }
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

    /// The near and far planes that keep `bounds` in view from the eye.
    pub fn clip_planes(&self, bounds: PreviewBounds) -> (f32, f32) {
        clip_planes_for(
            self.eye(),
            self.forward(),
            bounds.min_vec3(),
            bounds.max_vec3(),
        )
    }

    /// The view and projection for a frame that keeps `bounds` between
    /// the clip planes: a `Framed`'s pinhole, from [`framed`](Self::framed)
    /// or [`framed_stretched`](Self::framed_stretched).
    pub fn camera_view(&self, framed: &Framed, bounds: PreviewBounds) -> CameraView {
        let (near, far) = self.clip_planes(bounds);
        CameraView::pinhole(&framed.pinhole, self.camera_to_world, near, far)
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
        use volumetric::viewset::{ContourObs, CornerObs, FeatureObs, MarkerObs, Observations};
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
            features: vec![
                FeatureObs {
                    name: "hole_a".to_string(),
                    pixel: [50.0, 60.0],
                    role: PickRole::Fit,
                },
                FeatureObs {
                    name: "hole_b".to_string(),
                    pixel: [70.0, 60.0],
                    role: PickRole::Check,
                },
            ],
            contours: vec![ContourObs {
                name: "rim".to_string(),
                pixels: vec![[10.0, 10.0], [20.0, 10.0], [20.0, 20.0]],
            }],
        });
        let lines = observation_lines(&view, &camera);
        // Marker quads, the corner's cross, four arms per pick, and the
        // contour's two segments.
        assert_eq!(lines.len(), 4 + 4 + 2 + 4 + 4 + 2);
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
        // The fit pick's first arm ends short of the pixel; the check
        // pick's arms are diagonal; the contour follows its pixels.
        assert_eq!(lines[10].color, PICK_FIT_COLOR);
        let back = view.project(&camera, world(lines[10].end)).unwrap();
        assert!(
            (back[0] - (50.0 - PICK_CROSS_MIN_PX * 0.25)).abs() < 1e-3
                && (back[1] - 60.0).abs() < 1e-3,
            "{back:?}"
        );
        assert_eq!(lines[14].color, PICK_CHECK_COLOR);
        let back = view.project(&camera, world(lines[14].start)).unwrap();
        assert!(
            (back[0] - 70.0).abs() > 5.0 && (back[1] - 60.0).abs() > 5.0,
            "{back:?}"
        );
        assert_eq!(lines[18].color, CONTOUR_COLOR);
        let back = view.project(&camera, world(lines[19].end)).unwrap();
        assert!(
            (back[0] - 20.0).abs() < 1e-3 && (back[1] - 20.0).abs() < 1e-3,
            "{back:?}"
        );
        // Look-through carries them with the frustum.
        let look = LookThrough::of(&view, &camera).unwrap();
        assert_eq!(look.frustum.segments.len(), 9 + 20);
        // Labels sit just right of each cross and at a contour's start.
        let labels = mark_labels(&view, &camera);
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0].text, "hole_a");
        assert_eq!(labels[0].kind, MarkKind::Fit);
        assert!(labels[0].pixel[0] > 50.0 + PICK_CROSS_MIN_PX && labels[0].pixel[1] == 60.0);
        assert_eq!(labels[1].kind, MarkKind::Check);
        assert_eq!(
            (labels[2].text.as_str(), labels[2].kind),
            ("rim", MarkKind::Contour)
        );
        assert_eq!(MarkKind::Fit.srgb8(), [168, 255, 177]);
        // Unposed, the view draws nothing and cannot be looked through.
        let mut raw = view.clone();
        raw.camera_to_world = None;
        assert!(observation_lines(&raw, &camera).is_empty());
        assert!(mark_labels(&raw, &camera).is_empty());
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
        // sits 300 px in. An ideal lens needs no warp.
        let framed = frame.framed(1000, 300);
        let boxed = &framed.pinhole;
        assert_eq!((boxed.fx, boxed.fy), (400.0, 400.0));
        assert_eq!((boxed.cx, boxed.cy), (500.0, 150.0));
        assert_eq!(framed.warp, None);
        // A 200 x 600 viewport: scale 0.5, the picture is 200 x 150 at
        // y = 225.
        let boxed = frame.framed(200, 600).pinhole;
        assert_eq!((boxed.fx, boxed.cx, boxed.cy), (200.0, 100.0, 300.0));
        // Stretched to 800 x 300: x doubles, y stays.
        let stretched = frame.framed_stretched(800, 300).pinhole;
        assert_eq!(
            (stretched.fx, stretched.fy, stretched.cx, stretched.cy),
            (800.0, 400.0, 400.0, 150.0)
        );

        // A world point that lands on camera pixel (300, 150) lands on
        // the same picture pixel scaled and offset in the viewport.
        let world = [0.5, 0.0, -4.0];
        assert_eq!(view.project(camera, world), Some([300.0, 150.0]));
        let bounds = PreviewBounds {
            min: (-1.0, -1.0, -5.0),
            max: (1.0, 1.0, -3.0),
        };
        let cv = frame.camera_view(&frame.framed(1000, 300), bounds);
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

    /// Through a barrel lens the frame renders with an overscan and a
    /// warp: a world point's pixel in the photograph, taken through the
    /// warp, lands on the point's ideal pinhole pixel in the overscan
    /// frame, both letterboxed and stretched; a straight line stays
    /// straight in the overscan frame, so the render bends only at the
    /// warp.
    #[test]
    fn a_lens_frame_warps_the_output_onto_the_pinhole_render() {
        let mut set = set();
        set.cameras[0].distortion = Distortion::Radial {
            k: vec![-0.3, 0.1],
            p: [0.001, -0.0005],
        };
        let (view, camera) = set.view("v").unwrap();
        let frame = ViewFrame::of(view, camera).unwrap();
        // Barrel: the corners pull in, so the ideal image of the frame's
        // boundary reaches beyond the picture.
        let framed = frame.framed(400, 300);
        let warp = framed.warp.as_ref().expect("a real lens warps");
        assert_eq!(warp.output, (400, 300));
        assert!(
            warp.source.0 > 400 && warp.source.1 > 300,
            "{:?}",
            warp.source
        );
        assert_eq!(framed.pinhole.width, warp.source.0);
        assert!(framed.pinhole.cx > 200.0, "the overscan shifts the centre");
        let bounds = PreviewBounds {
            min: (-2.0, -2.0, -6.0),
            max: (2.0, 2.0, -3.0),
        };
        for (framed, size, offset) in [
            (frame.framed(400, 300), (400u32, 300u32), (0.0f64, 0.0f64)),
            (frame.framed(1000, 300), (1000, 300), (300.0, 0.0)),
            (frame.framed_stretched(800, 300), (800, 300), (0.0, 0.0)),
        ] {
            let warp = framed.warp.as_ref().unwrap();
            let cv = frame.camera_view(&framed, bounds);
            for world in [[0.5, 0.0, -4.0], [-0.8, 0.6, -4.0], [0.4, -0.3, -3.2]] {
                // Where the photograph has the point.
                let [u, v] = view.project(camera, world).unwrap();
                let sx = f64::from(size.0 - 2 * offset.0 as u32) / 400.0;
                let sy = f64::from(size.1) / 300.0;
                let out = [u * sx + offset.0, v * sy + offset.1];
                // Where the warp reads for that output pixel.
                let src = warp
                    .lookup(out[0] as f32, out[1] as f32)
                    .expect("inside the lens limit");
                // Where the pinhole render has the point.
                let px = cv
                    .project(
                        Vec3::from(world.map(|c| c as f32)),
                        warp.source.0,
                        warp.source.1,
                    )
                    .unwrap();
                // Within a tenth of a pixel: the grid interpolates the
                // lens between samples 16 px apart, and this lens is
                // twice as strong at the corners as the chair's DSLR.
                assert!(
                    (f64::from(px.x) - f64::from(src[0])).abs() < 0.1
                        && (f64::from(px.y) - f64::from(src[1])).abs() < 0.1,
                    "{size:?} {world:?}: warp reads {src:?}, render has {px:?}"
                );
            }
        }
        // The lens limit: a wildly off-axis output pixel has no image.
        let mut fish = set.clone();
        fish.cameras[0].distortion = Distortion::KannalaBrandt {
            k: [0.0, 0.0, 0.0, 0.0],
        };
        // 180 degrees across the 400 px width.
        fish.cameras[0].fx = 127.0;
        fish.cameras[0].fy = 127.0;
        let (view, camera) = fish.view("v").unwrap();
        let framed = ViewFrame::of(view, camera).unwrap().framed(400, 300);
        let warp = framed.warp.as_ref().unwrap();
        assert!(warp.lookup(200.0, 150.0).is_some());
        assert!(warp.lookup(1.0, 1.0).is_none(), "beyond 76 degrees");
    }
}
