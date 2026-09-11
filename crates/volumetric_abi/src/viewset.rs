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

/// The schema this module writes. Schema 1 (every view posed, no shot
/// state) is read and upgraded.
pub const VIEWSET_SCHEMA: u32 = 2;

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
    /// The survey card, when the set was detected or surveyed against one.
    #[serde(default)]
    pub board: Option<Board>,
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
    /// The directory the views' `source` paths are relative to, when the
    /// pictures live outside the set.
    #[serde(default)]
    pub origin: String,
}

/// A pinhole camera in the OpenCV convention, with its lens distortion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraModel {
    /// What the camera is, for people: the body, lens and focus setting
    /// the intake keyed it by. Empty when unknown.
    #[serde(default)]
    pub label: String,
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
    /// columns, camera position as the last column. `None` for a view not
    /// yet posed (a still straight from the camera).
    #[serde(default)]
    pub camera_to_world: Option<[f64; 12]>,
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
    /// What detection found in the photograph.
    #[serde(default)]
    pub observations: Option<Observations>,
    /// The camera's state when the picture was taken, from its metadata.
    #[serde(default)]
    pub shot: Option<Shot>,
    /// Where the original picture is, relative to the provenance's
    /// `origin`, when the embedded picture is reduced or absent.
    #[serde(default)]
    pub source: Option<String>,
}

/// A camera's state for one exposure, as its metadata reports it. Zero
/// or empty means unknown.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Shot {
    pub make: String,
    pub model: String,
    pub lens: String,
    /// Physical focal length, millimetres.
    pub focal_mm: f64,
    pub f_number: f64,
    pub exposure_s: f64,
    pub iso: u32,
    /// `manual`, `dmf`, `af-s`, `af-c`, `af-a`, or empty.
    pub focus_mode: String,
    /// The lens focus encoder, when the maker note gives it (Sony: 80 to
    /// 255, 255 at infinity).
    pub focus_position: Option<u32>,
    /// Image stabilisation, when the maker note gives it.
    pub stabilisation: Option<bool>,
    /// The EXIF orientation value, 1 for upright; 0 when absent.
    pub orientation: u32,
}

impl Shot {
    /// Focus that stays put between frames: manual and direct manual
    /// focus, or any setting whose mode is unknown.
    pub fn focus_is_held(&self) -> bool {
        !self.focus_mode.starts_with("af")
    }

    /// The key frames with the same intrinsics share: body, lens, focus
    /// mode and focus position. An autofocus frame focuses anew each time
    /// and gets a key of its own, from `view_id`.
    pub fn camera_key(&self, view_id: &str) -> String {
        let focus = match self.focus_position {
            Some(p) => format!("{}:{p}", self.focus_mode),
            None => self.focus_mode.clone(),
        };
        let mut key = format!("{}:{}:{}:{focus}", self.make, self.model, self.lens);
        if !self.focus_is_held() {
            key.push(':');
            key.push_str(view_id);
        }
        key
    }
}

/// A printed target card at a known place in the world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Marker {
    pub id: u32,
    pub size_m: f64,
    /// The four corners in world coordinates, in the dictionary's order.
    pub corners: [[f64; 3]; 4],
}

/// A ChArUco board: a chessboard whose white squares carry a marker each.
/// Board coordinates have the origin at the board's top-left outer corner,
/// x along the columns and y down the rows, in metres, with the printed
/// pitch measured separately along each axis.
///
/// Squares are black where row + column is even; the markers sit centred
/// in the white squares in row-major order from `first_id`; the interior
/// corners are numbered row-major, corner `row * (squares_x - 1) + col`
/// at `((col + 1) * pitch_x, (row + 1) * pitch_y)`. This is OpenCV's
/// `CharucoBoard` layout, which the printed cards follow.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoardSpec {
    pub squares_x: u32,
    pub squares_y: u32,
    /// Square pitch along x (the columns), metres.
    pub pitch_x_m: f64,
    /// Square pitch along y (the rows), metres.
    pub pitch_y_m: f64,
    /// Marker side, metres.
    pub marker_m: f64,
    /// The marker family: `36h11`, `5x5_100` or `4x4_50`.
    pub family: String,
    /// The first marker's id; ids run row-major over the white squares.
    pub first_id: u32,
}

impl BoardSpec {
    /// The survey card: 12 x 11 squares of 0.7 in with 12.7 mm AprilTag
    /// 36h11 markers from id 100, at the pitches the calipers measured on
    /// the glued print (along the feed 17.944 mm, across 17.745 mm).
    pub fn survey_card() -> Self {
        Self {
            squares_x: 12,
            squares_y: 11,
            pitch_x_m: 0.0179443,
            pitch_y_m: 0.0177451,
            marker_m: 0.0127,
            family: "36h11".to_string(),
            first_id: 100,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.squares_x < 2 || self.squares_y < 2 {
            return Err("a board needs at least two squares each way".to_string());
        }
        for (name, value) in [
            ("pitch_x_m", self.pitch_x_m),
            ("pitch_y_m", self.pitch_y_m),
            ("marker_m", self.marker_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("board {name} must be finite and positive"));
            }
        }
        if self.marker_m >= self.pitch_x_m.min(self.pitch_y_m) {
            return Err("board markers must be smaller than the squares".to_string());
        }
        if self.family.is_empty() {
            return Err("board family is empty".to_string());
        }
        Ok(())
    }

    pub fn n_markers(&self) -> u32 {
        self.squares_x * self.squares_y / 2
    }

    /// Interior corners: `(squares_x - 1) * (squares_y - 1)`.
    pub fn n_corners(&self) -> u32 {
        (self.squares_x - 1) * (self.squares_y - 1)
    }

    /// The board's outer size, metres.
    pub fn size_m(&self) -> [f64; 2] {
        [
            f64::from(self.squares_x) * self.pitch_x_m,
            f64::from(self.squares_y) * self.pitch_y_m,
        ]
    }

    /// The (row, column) of the white square carrying the marker.
    pub fn marker_square(&self, id: u32) -> Option<(u32, u32)> {
        let k = id.checked_sub(self.first_id)?;
        if k >= self.n_markers() {
            return None;
        }
        // Each row holds squares_x / 2 white squares, alternating start.
        let per_row = self.squares_x / 2;
        let (row, col) = if self.squares_x % 2 == 0 {
            let row = k / per_row;
            (row, 2 * (k % per_row) + (row + 1) % 2)
        } else {
            // Odd widths alternate between (w - 1) / 2 and (w + 1) / 2
            // white squares per row; walk the rows.
            let mut remaining = k;
            let mut row = 0;
            loop {
                let count = if row % 2 == 0 {
                    self.squares_x / 2
                } else {
                    self.squares_x.div_ceil(2)
                };
                if remaining < count {
                    break (row, 2 * remaining + (row + 1) % 2);
                }
                remaining -= count;
                row += 1;
            }
        };
        Some((row, col))
    }

    /// The marker's id in the white square at (row, column), if any.
    pub fn marker_at(&self, row: u32, col: u32) -> Option<u32> {
        if row >= self.squares_y || col >= self.squares_x || (row + col) % 2 == 0 {
            return None;
        }
        let mut k = 0;
        for r in 0..row {
            k += if r % 2 == 0 {
                self.squares_x / 2
            } else {
                self.squares_x.div_ceil(2)
            };
        }
        k += col / 2;
        Some(self.first_id + k)
    }

    /// A marker's corners in board coordinates, in the dictionary's order
    /// (top-left, top-right, bottom-right, bottom-left as printed).
    pub fn marker_corners(&self, id: u32) -> Option<[[f64; 2]; 4]> {
        let (row, col) = self.marker_square(id)?;
        let inset_x = (self.pitch_x_m - self.marker_m) * 0.5;
        let inset_y = (self.pitch_y_m - self.marker_m) * 0.5;
        let x0 = f64::from(col) * self.pitch_x_m + inset_x;
        let y0 = f64::from(row) * self.pitch_y_m + inset_y;
        let m = self.marker_m;
        Some([[x0, y0], [x0 + m, y0], [x0 + m, y0 + m], [x0, y0 + m]])
    }

    /// An interior corner's board coordinates.
    pub fn corner(&self, id: u32) -> Option<[f64; 2]> {
        if id >= self.n_corners() {
            return None;
        }
        let cols = self.squares_x - 1;
        let (row, col) = (id / cols, id % cols);
        Some([
            f64::from(col + 1) * self.pitch_x_m,
            f64::from(row + 1) * self.pitch_y_m,
        ])
    }

    /// The interior corners at the four corners of a square, as
    /// (corner id, board coordinates), clockwise from the square's
    /// top-left; a square on the board's edge has fewer.
    pub fn corners_of_square(&self, row: u32, col: u32) -> Vec<(u32, [f64; 2])> {
        let cols = self.squares_x - 1;
        let mut out = Vec::with_capacity(4);
        for (dr, dc) in [(0, 0), (0, 1), (1, 1), (1, 0)] {
            let (gy, gx) = (row + dr, col + dc);
            if gy == 0 || gx == 0 || gy > self.squares_y - 1 || gx > self.squares_x - 1 {
                continue;
            }
            let id = (gy - 1) * cols + (gx - 1);
            out.push((id, self.corner(id).expect("interior corner")));
        }
        out
    }
}

/// The survey card in a set: its spec, and its interior corners in the
/// world once a survey has solved them (empty until then).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Board {
    pub spec: BoardSpec,
    pub corners: Vec<BoardCorner>,
}

/// A solved interior corner of the board.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoardCorner {
    pub id: u32,
    pub position: [f64; 3],
    /// Standard error of the position from the survey, metres.
    pub sigma_m: f64,
}

/// What detection found in one photograph, in the set's pixel
/// convention (pixel centres at +0.5).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Observations {
    /// Markers of every family, swatches and the card's tags alike.
    pub markers: Vec<MarkerObs>,
    /// The board's interior corners.
    pub board: Vec<CornerObs>,
    /// Edge blur across the markers' edges, pixels (the worse of the two
    /// picture axes), when there were edges to measure.
    pub blur_px: Option<f64>,
}

/// A marker seen in a photograph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarkerObs {
    pub id: u32,
    /// `36h11`, `5x5_100` or `4x4_50`.
    pub family: String,
    /// In the dictionary's order (top-left first, clockwise as printed).
    pub corners: [[f64; 2]; 4],
    /// RMS distance of the fitted edge points to the fitted sides.
    pub fit_px: f64,
}

/// A board corner seen in a photograph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CornerObs {
    pub id: u32,
    pub pixel: [f64; 2],
    /// How far the sub-pixel refinement moved the corner from where the
    /// neighbouring markers predicted it.
    pub fit_px: f64,
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
            board: None,
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
        if let Some(board) = &self.board {
            board.spec.validate()?;
            for corner in &board.corners {
                if corner.id >= board.spec.n_corners() {
                    return Err(format!(
                        "board corner {} is beyond the board's {} corners",
                        corner.id,
                        board.spec.n_corners()
                    ));
                }
                if corner.position.iter().any(|c| !c.is_finite()) || !corner.sigma_m.is_finite() {
                    return Err(format!("board corner {} is not finite", corner.id));
                }
            }
        }
        for view in &self.views {
            let Some(obs) = &view.observations else {
                continue;
            };
            for m in &obs.markers {
                if m.corners.iter().flatten().any(|c| !c.is_finite()) || !m.fit_px.is_finite() {
                    return Err(format!(
                        "view {:?}: marker {} observation is not finite",
                        view.id, m.id
                    ));
                }
            }
            for c in &obs.board {
                if c.pixel.iter().any(|v| !v.is_finite()) || !c.fit_px.is_finite() {
                    return Err(format!(
                        "view {:?}: board corner {} observation is not finite",
                        view.id, c.id
                    ));
                }
                if let Some(board) = &self.board
                    && c.id >= board.spec.n_corners()
                {
                    return Err(format!(
                        "view {:?}: board corner {} is beyond the board's {} corners",
                        view.id,
                        c.id,
                        board.spec.n_corners()
                    ));
                }
            }
            if obs.blur_px.is_some_and(|b| !(b.is_finite() && b >= 0.0)) {
                return Err(format!("view {:?}: blur is not finite", view.id));
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
            label: String::new(),
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
            camera_to_world: Some(camera_to_world),
            ..Self::unposed(id, camera)
        }
    }

    /// A view with no pose yet and nothing embedded.
    pub fn unposed(id: impl Into<String>, camera: u32) -> Self {
        Self {
            id: id.into(),
            camera,
            camera_to_world: None,
            time: None,
            image: None,
            depth: None,
            depth_unit_m: 0.0,
            mask: None,
            tags: Vec::new(),
            observations: None,
            shot: None,
            source: None,
        }
    }

    /// The camera-to-world pose, when the view has one.
    pub fn pose(&self) -> Option<&[f64; 12]> {
        self.camera_to_world.as_ref()
    }

    /// The camera position in the world.
    pub fn position(&self) -> Option<[f64; 3]> {
        self.pose().map(|m| [m[3], m[7], m[11]])
    }

    /// Camera axis `i` (0 = right, 1 = down, 2 = forward) in the world.
    pub fn axis(&self, i: usize) -> Option<[f64; 3]> {
        self.pose().map(|m| [m[i], m[4 + i], m[8 + i]])
    }

    /// The direction the camera looks along, in the world.
    pub fn forward(&self) -> Option<[f64; 3]> {
        self.axis(2)
    }

    /// A camera-space point in the world.
    pub fn to_world(&self, point: [f64; 3]) -> Option<[f64; 3]> {
        let m = self.pose()?;
        let [x, y, z] = point;
        Some([
            m[0] * x + m[1] * y + m[2] * z + m[3],
            m[4] * x + m[5] * y + m[6] * z + m[7],
            m[8] * x + m[9] * y + m[10] * z + m[11],
        ])
    }

    /// A world point in camera space (the rotation is orthonormal, so its
    /// transpose inverts it).
    pub fn to_camera(&self, point: [f64; 3]) -> Option<[f64; 3]> {
        let m = self.pose()?;
        let d = [point[0] - m[3], point[1] - m[7], point[2] - m[11]];
        Some([
            m[0] * d[0] + m[4] * d[1] + m[8] * d[2],
            m[1] * d[0] + m[5] * d[1] + m[9] * d[2],
            m[2] * d[0] + m[6] * d[1] + m[10] * d[2],
        ])
    }

    /// The pixel a world point lands on through `camera`; `None` when the
    /// view is unposed or the point is behind it.
    pub fn project(&self, camera: &CameraModel, world: [f64; 3]) -> Option<[f64; 2]> {
        camera.project(self.to_camera(world)?)
    }

    /// The world point a pixel sees at z-depth `depth`.
    pub fn unproject(&self, camera: &CameraModel, pixel: [f64; 2], depth: f64) -> Option<[f64; 3]> {
        self.to_world(camera.point_at_depth(pixel, depth))
    }

    /// The world-space unit direction a pixel looks along.
    pub fn ray(&self, camera: &CameraModel, pixel: [f64; 2]) -> Option<[f64; 3]> {
        let d = camera.ray(pixel);
        let m = self.pose()?;
        Some(normalized([
            m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
            m[4] * d[0] + m[5] * d[1] + m[6] * d[2],
            m[8] * d[0] + m[9] * d[1] + m[10] * d[2],
        ]))
    }

    fn validate_pose(&self) -> Result<(), String> {
        let Some(m) = &self.camera_to_world else {
            return Ok(());
        };
        if m.iter().any(|v| !v.is_finite()) {
            return Err("pose has a non-finite entry".to_string());
        }
        // Columns of the rotation must be orthonormal.
        for i in 0..3 {
            for j in i..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let a = [m[i], m[4 + i], m[8 + i]];
                let b = [m[j], m[4 + j], m[8 + j]];
                if (dot(a, b) - expected).abs() > 1e-4 {
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

/// Decodes and validates a view set. A schema-1 set (every view posed,
/// no shot state) reads as schema 2.
pub fn decode_viewset(bytes: &[u8]) -> Result<ViewSet, String> {
    let mut set: ViewSet = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode view set CBOR: {e}"))?;
    if set.schema == 1 {
        set.schema = VIEWSET_SCHEMA;
    }
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
            board: None,
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
        assert_eq!(view.position(), Some([1.0, 2.0, 3.0]));
        assert_eq!(view.forward(), Some([1.0, 0.0, -0.0].map(|v| v * 1.0)));
        let world = view.unproject(camera, [484.0, 488.0], 2.0).unwrap();
        // The optical axis looks along world +x from (1, 2, 3).
        assert!(
            (world[0] - 3.0).abs() < 1e-12
                && (world[1] - 2.0).abs() < 1e-12
                && (world[2] - 3.0).abs() < 1e-12,
            "{world:?}"
        );
        let pixel = view.project(camera, world).unwrap();
        assert!((pixel[0] - 484.0).abs() < 1e-9 && (pixel[1] - 488.0).abs() < 1e-9);
        let back = view
            .to_camera(view.to_world([0.3, -0.2, 1.7]).unwrap())
            .unwrap();
        assert!(
            (back[0] - 0.3).abs() < 1e-12
                && (back[1] + 0.2).abs() < 1e-12
                && (back[2] - 1.7).abs() < 1e-12
        );
        let ray = view.ray(camera, [484.0, 488.0]).unwrap();
        assert!((ray[0] - 1.0).abs() < 1e-9, "{ray:?}");
    }

    #[test]
    fn the_survey_card_lays_out_as_opencv_prints_it() {
        // Values from OpenCV's CharucoBoard for the card (legacy pattern
        // off): squares 17.78 mm nominal, markers 12.7 mm.
        let mut card = BoardSpec::survey_card();
        card.pitch_x_m = 0.01778;
        card.pitch_y_m = 0.01778;
        assert_eq!((card.n_markers(), card.n_corners()), (66, 110));
        let close =
            |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9;
        assert!(close(card.corner(0).unwrap(), [0.01778, 0.01778]));
        assert!(close(card.corner(1).unwrap(), [0.03556, 0.01778]));
        assert!(close(card.corner(11).unwrap(), [0.01778, 0.03556]));
        assert!(close(card.corner(109).unwrap(), [0.19558, 0.1778]));
        assert!(card.corner(110).is_none());
        // Marker 100 sits in square (row 0, col 1); 106 starts row 1 at
        // col 0; 112 starts row 2 at col 1; 165 is the last, (10, 11).
        assert_eq!(card.marker_square(100), Some((0, 1)));
        assert_eq!(card.marker_square(103), Some((0, 7)));
        assert_eq!(card.marker_square(106), Some((1, 0)));
        assert_eq!(card.marker_square(111), Some((1, 10)));
        assert_eq!(card.marker_square(112), Some((2, 1)));
        assert_eq!(card.marker_square(165), Some((10, 11)));
        assert_eq!(card.marker_square(166), None);
        assert_eq!(card.marker_square(99), None);
        for id in 100..166 {
            let (r, c) = card.marker_square(id).unwrap();
            assert_eq!(card.marker_at(r, c), Some(id));
        }
        assert_eq!(card.marker_at(0, 0), None);
        let m100 = card.marker_corners(100).unwrap();
        assert!(close(m100[0], [0.02032, 0.00254]), "{m100:?}");
        assert!(close(m100[2], [0.03302, 0.01524]), "{m100:?}");
        let m106 = card.marker_corners(106).unwrap();
        assert!(close(m106[0], [0.00254, 0.02032]), "{m106:?}");
        // Square (0, 1) has interior corners at its bottom-right and
        // bottom-left only: ids 1 and 0.
        let around = card.corners_of_square(0, 1);
        assert_eq!(
            around.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![1, 0]
        );
        let around = card.corners_of_square(3, 4);
        assert_eq!(
            around.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![2 * 11 + 3, 2 * 11 + 4, 3 * 11 + 4, 3 * 11 + 3]
        );
        // An odd width alternates the white count per row.
        let odd = BoardSpec {
            squares_x: 7,
            squares_y: 5,
            first_id: 0,
            ..BoardSpec::survey_card()
        };
        assert_eq!(odd.n_markers(), 17);
        assert_eq!(odd.marker_square(0), Some((0, 1)));
        assert_eq!(odd.marker_square(2), Some((0, 5)));
        assert_eq!(odd.marker_square(3), Some((1, 0)));
        assert_eq!(odd.marker_square(6), Some((1, 6)));
        assert_eq!(odd.marker_square(7), Some((2, 1)));
        for id in 0..17 {
            let (r, c) = odd.marker_square(id).unwrap();
            assert_eq!((r + c) % 2, 1);
            assert_eq!(odd.marker_at(r, c), Some(id));
        }
        assert!(BoardSpec::survey_card().validate().is_ok());
        assert!(
            BoardSpec {
                marker_m: 0.02,
                ..BoardSpec::survey_card()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn observations_and_the_board_round_trip_and_default_absent() {
        let mut set = ViewSet {
            cameras: vec![CameraModel::pinhole(100, 80, 90.0, 90.0, 50.0, 40.0)],
            views: vec![View::posed("a", 0, identity_pose())],
            board: Some(Board {
                spec: BoardSpec::survey_card(),
                corners: vec![BoardCorner {
                    id: 3,
                    position: [0.1, 0.2, 0.0],
                    sigma_m: 1e-4,
                }],
            }),
            ..ViewSet::default()
        };
        set.views[0].observations = Some(Observations {
            markers: vec![MarkerObs {
                id: 100,
                family: "36h11".to_string(),
                corners: [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
                fit_px: 0.1,
            }],
            board: vec![CornerObs {
                id: 7,
                pixel: [10.5, 20.5],
                fit_px: 0.3,
            }],
            blur_px: Some(1.2),
        });
        let back = decode_viewset(&encode_viewset(&set)).unwrap();
        assert_eq!(back, set);
        // A corner beyond the board is caught, in the set and in a view.
        let mut bad = set.clone();
        bad.board.as_mut().unwrap().corners[0].id = 110;
        assert!(bad.validate().is_err());
        let mut bad = set.clone();
        bad.views[0].observations.as_mut().unwrap().board[0].id = 110;
        assert!(bad.validate().is_err());
        // Without the fields (an older encoding) both come back absent.
        let mut value: ciborium::Value =
            ciborium::de::from_reader(std::io::Cursor::new(encode_viewset(&set))).unwrap();
        let strip = |map: &mut Vec<(ciborium::Value, ciborium::Value)>, key: &str| {
            map.retain(|(k, _)| k.as_text() != Some(key));
        };
        if let ciborium::Value::Map(map) = &mut value {
            strip(map, "board");
            for (k, v) in map.iter_mut() {
                if k.as_text() == Some("schema") {
                    *v = ciborium::Value::Integer(1.into());
                }
                if k.as_text() == Some("views")
                    && let ciborium::Value::Array(views) = v
                {
                    for view in views {
                        if let ciborium::Value::Map(fields) = view {
                            strip(fields, "observations");
                            strip(fields, "shot");
                            strip(fields, "source");
                        }
                    }
                }
            }
        }
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&value, &mut bytes).unwrap();
        let old = decode_viewset(&bytes).unwrap();
        assert_eq!(old.schema, VIEWSET_SCHEMA);
        assert!(old.board.is_none());
        assert!(old.views[0].observations.is_none());
        assert!(old.views[0].shot.is_none() && old.views[0].source.is_none());
        assert_eq!(old.views[0].camera_to_world, Some(identity_pose()));
    }

    #[test]
    fn unposed_views_and_shots_round_trip() {
        let mut set = ViewSet {
            cameras: vec![CameraModel::pinhole(100, 80, 90.0, 90.0, 50.0, 40.0)],
            views: vec![View::unposed("raw", 0)],
            ..ViewSet::default()
        };
        set.views[0].source = Some("DSC00001.JPG".to_string());
        set.views[0].shot = Some(Shot {
            make: "SONY".to_string(),
            model: "ILCE-6700".to_string(),
            lens: "FE 50mm F1.8".to_string(),
            focal_mm: 50.0,
            f_number: 8.0,
            exposure_s: 0.004,
            iso: 5000,
            focus_mode: "manual".to_string(),
            focus_position: Some(170),
            stabilisation: Some(false),
            orientation: 1,
        });
        set.provenance.origin = "/stills".to_string();
        let view = &set.views[0];
        assert!(view.pose().is_none() && view.position().is_none());
        assert!(view.project(&set.cameras[0], [0.0, 0.0, 1.0]).is_none());
        assert!(view.ray(&set.cameras[0], [1.0, 1.0]).is_none());
        let shot = view.shot.as_ref().unwrap();
        assert_eq!(
            shot.camera_key("raw"),
            "SONY:ILCE-6700:FE 50mm F1.8:manual:170"
        );
        let af = Shot {
            focus_mode: "af-s".to_string(),
            ..shot.clone()
        };
        assert_eq!(
            af.camera_key("raw"),
            "SONY:ILCE-6700:FE 50mm F1.8:af-s:170:raw"
        );
        assert!(!af.focus_is_held() && shot.focus_is_held());
        let back = decode_viewset(&encode_viewset(&set)).unwrap();
        assert_eq!(back, set);
    }

    #[test]
    fn validation_catches_structural_faults() {
        let mut set = ViewSet {
            board: None,
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
        bad.views[0].camera_to_world.as_mut().unwrap()[0] = 2.0;
        assert!(bad.validate().unwrap_err().contains("orthonormal"));

        let mut bad = set.clone();
        bad.views[0].depth = Some(vec![0]);
        assert!(bad.validate().unwrap_err().contains("unit"));

        let mut bad = set.clone();
        bad.cameras[0].fx = -1.0;
        assert!(bad.validate().unwrap_err().contains("fx"));

        let mut bad = set;
        bad.schema = 3;
        assert!(decode_viewset(&encode_viewset(&bad)).is_err());
    }
}
