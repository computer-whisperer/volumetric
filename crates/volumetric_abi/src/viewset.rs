//! Posed images with depth: the `ViewSet` value.
//!
//! A view set is physical evidence about a scene — photographs whose
//! cameras are known — in the form an engine can use: shared intrinsics
//! (with a distortion model), one camera-to-world pose per view in the
//! OpenCV convention (x right, y down, z forward; pixel origin top-left,
//! +v down), and the encoded image, depth map and subject mask each view
//! carries. The marker map that posed the views rides along, as does a
//! provenance record naming the capture so combinators can refuse to mix
//! evidence from different setups.
//!
//! Images stay encoded (PNG or JPEG bytes) so the value can travel through
//! the DAG and the project file; decoding is the consumer's job. Depth maps
//! are 16-bit PNGs in units of `depth_unit_m` and hold z-depth (distance
//! along the camera axis), zero meaning no measurement.
//!
//! The camera math lives here, beside the type, so operators that pose or
//! fuse views share one definition with the hosts that draw them.

use serde::{Deserialize, Serialize};

/// The schema this module writes and accepts.
pub const VIEWSET_SCHEMA: u32 = 1;

/// Posed images, their cameras, the marker map that posed them, and where
/// they came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewSet {
    pub schema: u32,
    pub world: WorldFrame,
    pub provenance: Provenance,
    /// Intrinsics shared by views; a view names its camera by index.
    pub cameras: Vec<CameraModel>,
    pub views: Vec<View>,
    /// The marker map: printed target cards at known world positions.
    pub markers: Vec<Marker>,
}

/// The world the poses live in. Units are always metres.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldFrame {
    /// The gravity-opposed direction, unit length.
    pub up: [f64; 3],
}

impl Default for WorldFrame {
    fn default() -> Self {
        Self {
            up: [0.0, 1.0, 0.0],
        }
    }
}

/// Where the evidence came from. Empty strings mean unknown; a non-empty
/// `field` or `setup` that differs between two sets means they must not be
/// combined without an explicit alignment.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// The capture session.
    pub session: String,
    /// The capture system's calibration (an id or content hash).
    pub rig: String,
    /// The tracking field: marker map and base stations that define the
    /// world frame.
    pub field: String,
    /// The subject's placement within the field.
    pub setup: String,
    /// The tools that produced this set, newest last.
    pub tools: Vec<String>,
    /// When the capture happened (ISO 8601), if known.
    pub captured: String,
}

/// A pinhole camera in the OpenCV convention, with its lens distortion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraModel {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub distortion: Distortion,
}

/// Lens distortion applied to normalised camera coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum Distortion {
    /// A rectified or ideal image.
    None,
    /// OpenCV's radial-tangential model: `k` holds k1, k2 and optionally k3;
    /// `p` holds p1 and p2.
    Radial { k: Vec<f64>, p: [f64; 2] },
    /// The Kannala-Brandt fisheye model with four polynomial terms in the
    /// incidence angle (OpenCV's `fisheye` module, the Index's cameras).
    KannalaBrandt { k: [f64; 4] },
}

/// One photograph and everything measured with it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct View {
    /// Unique within the set.
    pub id: String,
    /// Index into [`ViewSet::cameras`].
    pub camera: u32,
    /// Camera-to-world transform, the rows of a 3x4 matrix: camera axes as
    /// columns, camera position as the last column.
    pub camera_to_world: [f64; 12],
    /// Seconds from the session's start, when known.
    pub time: Option<f64>,
    /// The photograph, PNG or JPEG encoded.
    #[serde(with = "serde_bytes")]
    pub image: Option<Vec<u8>>,
    /// Z-depth per pixel as a 16-bit PNG in units of `depth_unit_m`; zero
    /// means no measurement.
    #[serde(with = "serde_bytes")]
    pub depth: Option<Vec<u8>>,
    pub depth_unit_m: f64,
    /// Subject mask as a PNG, nonzero where the subject is.
    #[serde(with = "serde_bytes")]
    pub mask: Option<Vec<u8>>,
    /// What the source knew about the view: eye, split, and so on.
    pub tags: Vec<String>,
}

/// A printed target card at a known place in the world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Marker {
    pub id: u32,
    pub size_m: f64,
    /// The four corners in world coordinates, in the dictionary's order.
    pub corners: [[f64; 3]; 4],
}

impl Default for ViewSet {
    fn default() -> Self {
        Self {
            schema: VIEWSET_SCHEMA,
            world: WorldFrame::default(),
            provenance: Provenance::default(),
            cameras: Vec::new(),
            views: Vec::new(),
            markers: Vec::new(),
        }
    }
}

impl ViewSet {
    /// Structural checks: schema, camera references, unique ids, finite
    /// and positive intrinsics, near-orthonormal poses, positive depth units
    /// and marker sizes.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != VIEWSET_SCHEMA {
            return Err(format!(
                "view set schema {} is not the supported {VIEWSET_SCHEMA}",
                self.schema
            ));
        }
        let up_len = norm(self.world.up);
        if !(up_len.is_finite() && (up_len - 1.0).abs() < 1e-6) {
            return Err("world up axis must be a unit vector".to_string());
        }
        for (i, camera) in self.cameras.iter().enumerate() {
            camera
                .validate()
                .map_err(|err| format!("camera {i}: {err}"))?;
        }
        let mut ids = std::collections::HashSet::with_capacity(self.views.len());
        for view in &self.views {
            if view.id.is_empty() {
                return Err("a view has an empty id".to_string());
            }
            if !ids.insert(view.id.as_str()) {
                return Err(format!("view id {:?} is used twice", view.id));
            }
            if view.camera as usize >= self.cameras.len() {
                return Err(format!(
                    "view {:?} names camera {} but the set has {}",
                    view.id,
                    view.camera,
                    self.cameras.len()
                ));
            }
            view.validate_pose()
                .map_err(|err| format!("view {:?}: {err}", view.id))?;
            if view.depth.is_some() && !(view.depth_unit_m.is_finite() && view.depth_unit_m > 0.0) {
                return Err(format!(
                    "view {:?} carries depth with a non-positive unit",
                    view.id
                ));
            }
        }
        for marker in &self.markers {
            if !(marker.size_m.is_finite() && marker.size_m > 0.0) {
                return Err(format!("marker {} has a non-positive size", marker.id));
            }
            if marker.corners.iter().flatten().any(|c| !c.is_finite()) {
                return Err(format!("marker {} has a non-finite corner", marker.id));
            }
        }
        Ok(())
    }

    /// A view and its camera, by id.
    pub fn view(&self, id: &str) -> Option<(&View, &CameraModel)> {
        let view = self.views.iter().find(|v| v.id == id)?;
        Some((view, self.camera_of(view)))
    }

    /// The camera a view names. The set must have been validated.
    pub fn camera_of(&self, view: &View) -> &CameraModel {
        &self.cameras[view.camera as usize]
    }
}

impl CameraModel {
    /// An ideal pinhole with no distortion.
    pub fn pinhole(width: u32, height: u32, fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            distortion: Distortion::None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err("image size must be positive".to_string());
        }
        for (name, value) in [("fx", self.fx), ("fy", self.fy)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        for (name, value) in [("cx", self.cx), ("cy", self.cy)] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        match &self.distortion {
            Distortion::None => {}
            Distortion::Radial { k, p } => {
                if k.len() > 3 || k.iter().chain(p.iter()).any(|c| !c.is_finite()) {
                    return Err(
                        "radial distortion takes up to three finite k and two p".to_string()
                    );
                }
            }
            Distortion::KannalaBrandt { k } => {
                if k.iter().any(|c| !c.is_finite()) {
                    return Err("fisheye coefficients must be finite".to_string());
                }
            }
        }
        Ok(())
    }

    /// Vertical field of view in radians of the undistorted image.
    pub fn fov_y(&self) -> f64 {
        2.0 * (f64::from(self.height) / (2.0 * self.fy)).atan()
    }

    /// Applies the lens to normalised coordinates `(x/z, y/z)`.
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        match &self.distortion {
            Distortion::None => (x, y),
            Distortion::Radial { k, p } => {
                let r2 = x * x + y * y;
                let radial = 1.0
                    + k.first().copied().unwrap_or(0.0) * r2
                    + k.get(1).copied().unwrap_or(0.0) * r2 * r2
                    + k.get(2).copied().unwrap_or(0.0) * r2 * r2 * r2;
                let (p1, p2) = (p[0], p[1]);
                (
                    x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                    y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
                )
            }
            Distortion::KannalaBrandt { k } => {
                let r = (x * x + y * y).sqrt();
                if r < 1e-12 {
                    return (x, y);
                }
                let theta = r.atan();
                let t2 = theta * theta;
                let theta_d = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
                let scale = theta_d / r;
                (x * scale, y * scale)
            }
        }
    }

    /// Inverts [`distort`](Self::distort): the normalised coordinates that
    /// the lens maps to `(xd, yd)`.
    fn undistort(&self, xd: f64, yd: f64) -> (f64, f64) {
        match &self.distortion {
            Distortion::None => (xd, yd),
            Distortion::Radial { .. } => {
                // Fixed-point iteration on x = (xd - tangential(x)) / radial(x),
                // which converges for the mild distortion of real lenses.
                let (mut x, mut y) = (xd, yd);
                for _ in 0..25 {
                    let (dx, dy) = self.distort(x, y);
                    let (nx, ny) = (x + (xd - dx), y + (yd - dy));
                    let moved = (nx - x).abs().max((ny - y).abs());
                    (x, y) = (nx, ny);
                    if moved < 1e-12 {
                        break;
                    }
                }
                (x, y)
            }
            Distortion::KannalaBrandt { k } => {
                let rd = (xd * xd + yd * yd).sqrt();
                if rd < 1e-12 {
                    return (xd, yd);
                }
                // Newton on theta_d(theta) = rd, starting from the ideal lens.
                let mut theta = rd;
                for _ in 0..20 {
                    let t2 = theta * theta;
                    let f =
                        theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - rd;
                    let df = 1.0
                        + t2 * (3.0 * k[0]
                            + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])));
                    let step = f / df;
                    theta -= step;
                    if step.abs() < 1e-14 {
                        break;
                    }
                }
                let scale = theta.tan() / rd;
                (xd * scale, yd * scale)
            }
        }
    }

    /// The pixel a camera-space point lands on, or `None` when it is not in
    /// front of the camera.
    pub fn project(&self, point: [f64; 3]) -> Option<[f64; 2]> {
        let [x, y, z] = point;
        if z.is_nan() || z <= 0.0 {
            return None;
        }
        let (xd, yd) = self.distort(x / z, y / z);
        Some([self.fx * xd + self.cx, self.fy * yd + self.cy])
    }

    /// The unit direction in camera space that a pixel looks along.
    pub fn ray(&self, pixel: [f64; 2]) -> [f64; 3] {
        let (xn, yn) = self.undistort(
            (pixel[0] - self.cx) / self.fx,
            (pixel[1] - self.cy) / self.fy,
        );
        normalized([xn, yn, 1.0])
    }

    /// The camera-space point a pixel sees at z-depth `depth`.
    pub fn point_at_depth(&self, pixel: [f64; 2], depth: f64) -> [f64; 3] {
        let (xn, yn) = self.undistort(
            (pixel[0] - self.cx) / self.fx,
            (pixel[1] - self.cy) / self.fy,
        );
        [xn * depth, yn * depth, depth]
    }
}

impl View {
    /// A view at `camera_to_world` with nothing embedded.
    pub fn posed(id: impl Into<String>, camera: u32, camera_to_world: [f64; 12]) -> Self {
        Self {
            id: id.into(),
            camera,
            camera_to_world,
            time: None,
            image: None,
            depth: None,
            depth_unit_m: 0.0,
            mask: None,
            tags: Vec::new(),
        }
    }

    /// The camera position in the world.
    pub fn position(&self) -> [f64; 3] {
        let m = &self.camera_to_world;
        [m[3], m[7], m[11]]
    }

    /// Camera axis `i` (0 = right, 1 = down, 2 = forward) in the world.
    pub fn axis(&self, i: usize) -> [f64; 3] {
        let m = &self.camera_to_world;
        [m[i], m[4 + i], m[8 + i]]
    }

    /// The direction the camera looks along, in the world.
    pub fn forward(&self) -> [f64; 3] {
        self.axis(2)
    }

    /// A camera-space point in the world.
    pub fn to_world(&self, point: [f64; 3]) -> [f64; 3] {
        let m = &self.camera_to_world;
        let [x, y, z] = point;
        [
            m[0] * x + m[1] * y + m[2] * z + m[3],
            m[4] * x + m[5] * y + m[6] * z + m[7],
            m[8] * x + m[9] * y + m[10] * z + m[11],
        ]
    }

    /// A world point in camera space (the rotation is orthonormal, so its
    /// transpose inverts it).
    pub fn to_camera(&self, point: [f64; 3]) -> [f64; 3] {
        let m = &self.camera_to_world;
        let d = [point[0] - m[3], point[1] - m[7], point[2] - m[11]];
        [
            m[0] * d[0] + m[4] * d[1] + m[8] * d[2],
            m[1] * d[0] + m[5] * d[1] + m[9] * d[2],
            m[2] * d[0] + m[6] * d[1] + m[10] * d[2],
        ]
    }

    /// The pixel a world point lands on through `camera`.
    pub fn project(&self, camera: &CameraModel, world: [f64; 3]) -> Option<[f64; 2]> {
        camera.project(self.to_camera(world))
    }

    /// The world point a pixel sees at z-depth `depth`.
    pub fn unproject(&self, camera: &CameraModel, pixel: [f64; 2], depth: f64) -> [f64; 3] {
        self.to_world(camera.point_at_depth(pixel, depth))
    }

    /// The world-space unit direction a pixel looks along.
    pub fn ray(&self, camera: &CameraModel, pixel: [f64; 2]) -> [f64; 3] {
        let d = camera.ray(pixel);
        let m = &self.camera_to_world;
        normalized([
            m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
            m[4] * d[0] + m[5] * d[1] + m[6] * d[2],
            m[8] * d[0] + m[9] * d[1] + m[10] * d[2],
        ])
    }

    fn validate_pose(&self) -> Result<(), String> {
        if self.camera_to_world.iter().any(|v| !v.is_finite()) {
            return Err("pose has a non-finite entry".to_string());
        }
        // Columns of the rotation must be orthonormal.
        for i in 0..3 {
            for j in i..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot(self.axis(i), self.axis(j)) - expected).abs() > 1e-4 {
                    return Err("pose rotation is not orthonormal".to_string());
                }
            }
        }
        Ok(())
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn normalized(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n > 0.0 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}

/// Encodes a view set as CBOR. Images travel as byte strings.
pub fn encode_viewset(set: &ViewSet) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(set, &mut out).expect("view set CBOR serialization should not fail");
    out
}

/// Decodes and validates a view set.
pub fn decode_viewset(bytes: &[u8]) -> Result<ViewSet, String> {
    let set: ViewSet = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode view set CBOR: {e}"))?;
    set.validate()?;
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_pose() -> [f64; 12] {
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    }

    /// Rotation of 90 degrees about y (camera looks along world -x) at
    /// position (1, 2, 3).
    fn posed() -> [f64; 12] {
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, -1.0, 0.0, 0.0, 3.0]
    }

    fn cameras() -> Vec<CameraModel> {
        vec![
            CameraModel::pinhole(960, 960, 414.0, 415.0, 484.0, 488.0),
            CameraModel {
                distortion: Distortion::Radial {
                    k: vec![0.05, -0.01, 0.002],
                    p: [0.001, -0.0005],
                },
                ..CameraModel::pinhole(4000, 3000, 3100.0, 3100.0, 2000.0, 1500.0)
            },
            CameraModel {
                distortion: Distortion::KannalaBrandt {
                    k: [0.1855, 0.0503, -0.2343, 0.0956],
                },
                ..CameraModel::pinhole(960, 960, 414.0, 415.0, 484.0, 488.0)
            },
        ]
    }

    #[test]
    fn round_trips_with_embedded_bytes() {
        let image = vec![200u8; 10_000];
        let mut set = ViewSet {
            cameras: cameras(),
            markers: vec![Marker {
                id: 7,
                size_m: 0.06,
                corners: [
                    [0.0; 3],
                    [0.06, 0.0, 0.0],
                    [0.06, 0.0, 0.06],
                    [0.0, 0.0, 0.06],
                ],
            }],
            ..ViewSet::default()
        };
        set.provenance.session = "chairbase1".to_string();
        let mut view = View::posed("00600_l", 0, posed());
        view.image = Some(image.clone());
        view.depth = Some(vec![1, 2, 3]);
        view.depth_unit_m = 1e-4;
        view.tags = vec!["left".to_string()];
        set.views.push(view);
        set.views.push(View::posed("00600_r", 2, identity_pose()));

        let bytes = encode_viewset(&set);
        let decoded = decode_viewset(&bytes).expect("decodes");
        assert_eq!(decoded, set);
        // Bytes are byte strings, not integer arrays: the image costs its
        // own length plus a small header, not two CBOR bytes per byte.
        assert!(bytes.len() < image.len() + 1_500, "{} bytes", bytes.len());
        assert_eq!(decoded.view("00600_l").unwrap().1.width, 960);
    }

    #[test]
    fn projection_inverts_the_ray_for_every_lens() {
        for camera in cameras() {
            for &(u, v) in &[(10.0, 20.0), (480.0, 488.0), (900.0, 150.0), (300.0, 800.0)] {
                let pixel = [u, v];
                let point = camera.point_at_depth(pixel, 2.5);
                let back = camera.project(point).expect("in front");
                assert!(
                    (back[0] - u).abs() < 1e-6 && (back[1] - v).abs() < 1e-6,
                    "{:?}: {pixel:?} -> {point:?} -> {back:?}",
                    camera.distortion
                );
                let ray = camera.ray(pixel);
                let along = [ray[0] * 3.0 / ray[2], ray[1] * 3.0 / ray[2], 3.0];
                let back = camera.project(along).unwrap();
                assert!((back[0] - u).abs() < 1e-6 && (back[1] - v).abs() < 1e-6);
            }
        }
        assert!(cameras()[0].project([0.0, 0.0, -1.0]).is_none());
        assert!((cameras()[0].fov_y() - 2.0 * (480.0f64 / 415.0).atan()).abs() < 1e-12);
    }

    #[test]
    fn poses_move_points_both_ways() {
        let view = View::posed("v", 0, posed());
        let camera = &cameras()[0];
        assert_eq!(view.position(), [1.0, 2.0, 3.0]);
        assert_eq!(view.forward(), [1.0, 0.0, -0.0].map(|v| v * 1.0));
        let world = view.unproject(camera, [484.0, 488.0], 2.0);
        // The optical axis looks along world +x from (1, 2, 3).
        assert!(
            (world[0] - 3.0).abs() < 1e-12
                && (world[1] - 2.0).abs() < 1e-12
                && (world[2] - 3.0).abs() < 1e-12,
            "{world:?}"
        );
        let pixel = view.project(camera, world).unwrap();
        assert!((pixel[0] - 484.0).abs() < 1e-9 && (pixel[1] - 488.0).abs() < 1e-9);
        let back = view.to_camera(view.to_world([0.3, -0.2, 1.7]));
        assert!(
            (back[0] - 0.3).abs() < 1e-12
                && (back[1] + 0.2).abs() < 1e-12
                && (back[2] - 1.7).abs() < 1e-12
        );
        let ray = view.ray(camera, [484.0, 488.0]);
        assert!((ray[0] - 1.0).abs() < 1e-9, "{ray:?}");
    }

    #[test]
    fn validation_catches_structural_faults() {
        let mut set = ViewSet {
            cameras: cameras(),
            ..ViewSet::default()
        };
        set.views.push(View::posed("a", 0, identity_pose()));
        set.validate().expect("sound");

        let mut bad = set.clone();
        bad.views.push(View::posed("a", 0, identity_pose()));
        assert!(bad.validate().unwrap_err().contains("twice"));

        let mut bad = set.clone();
        bad.views[0].camera = 9;
        assert!(bad.validate().unwrap_err().contains("camera 9"));

        let mut bad = set.clone();
        bad.views[0].camera_to_world[0] = 2.0;
        assert!(bad.validate().unwrap_err().contains("orthonormal"));

        let mut bad = set.clone();
        bad.views[0].depth = Some(vec![0]);
        assert!(bad.validate().unwrap_err().contains("unit"));

        let mut bad = set.clone();
        bad.cameras[0].fx = -1.0;
        assert!(bad.validate().unwrap_err().contains("fx"));

        let mut bad = set;
        bad.schema = 2;
        assert!(decode_viewset(&encode_viewset(&bad)).is_err());
    }
}
