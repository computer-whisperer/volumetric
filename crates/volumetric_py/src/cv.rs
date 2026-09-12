//! Pictures and markers: `cv_core` one to one. Detections, corners, poses
//! and reports come back as dicts; options are keyword arguments over the
//! crate's defaults.

use cv_core::board::{PlacedBoard, Render};
use cv_core::{
    CornerParams, DetectParams, Dictionary, ObserveOptions, StillOptions, SurveyOptions,
};
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use volumetric_abi::viewset::{BoardSpec, Marker};

use crate::viewset::{Camera, ViewSet};
use crate::{
    array2, from_py, invalid, options_from_dict, options_from_kwargs, pose_from_py, runtime, to_py,
    vec3_from_py,
};

/// A greyscale picture, row-major uint8.
#[pyclass(module = "volumetric")]
pub struct Gray {
    pub(crate) inner: cv_core::Gray,
}

#[pymethods]
impl Gray {
    /// From an (h,w) uint8 array.
    #[new]
    fn new(array: PyReadonlyArray2<'_, u8>) -> Self {
        let view = array.as_array();
        let (height, width) = view.dim();
        let mut inner = cv_core::Gray::new(width as u32, height as u32);
        inner.pixels = view.iter().copied().collect();
        Self { inner }
    }

    /// From an (h,w,3) uint8 RGB array, by luma.
    #[staticmethod]
    fn from_rgb(array: PyReadonlyArray3<'_, u8>) -> PyResult<Self> {
        let view = array.as_array();
        let (height, width, channels) = view.dim();
        if channels != 3 {
            return Err(invalid("an RGB array is (h,w,3)"));
        }
        let rgb: Vec<u8> = view.iter().copied().collect();
        Ok(Self {
            inner: cv_core::Gray::from_rgb8(width as u32, height as u32, &rgb),
        })
    }

    /// From a JPEG or PNG's bytes.
    #[staticmethod]
    fn decode(data: &[u8]) -> PyResult<Self> {
        let rgb = view_core::image::decode_rgb(data).map_err(runtime)?;
        Ok(Self {
            inner: cv_core::Gray::from_rgb8(rgb.width, rgb.height, &rgb.pixels),
        })
    }

    /// Read a picture file.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(runtime)?;
        Self::decode(&bytes)
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    /// The pixels as an (h,w) uint8 array (a copy).
    #[getter]
    fn array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        array2(py, self.inner.pixels.clone(), self.inner.width as usize)
    }

    /// Downsampled by an integer factor (box filter).
    fn downsampled(&self, factor: u32) -> Self {
        Self {
            inner: self.inner.downsampled(factor),
        }
    }

    /// Gaussian-blurred.
    fn blurred(&self, sigma: f64) -> Self {
        Self {
            inner: self.inner.blurred(sigma),
        }
    }

    fn __repr__(&self) -> String {
        format!("Gray({}x{})", self.inner.width, self.inner.height)
    }
}

fn dictionary(name: &str) -> PyResult<Dictionary> {
    Dictionary::by_name(name).ok_or_else(|| {
        invalid(format!(
            "unknown marker family `{name}` (5x5_100, 4x4_50, 36h11)"
        ))
    })
}

/// A board spec from `"survey_card"` or an absent argument (the survey
/// card), `False` (no board) or a dict of `BoardSpec` fields.
fn board_spec(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<BoardSpec>> {
    match value {
        None => Ok(Some(BoardSpec::survey_card())),
        Some(v) if v.is_none() => Ok(Some(BoardSpec::survey_card())),
        Some(v) if matches!(v.extract::<bool>(), Ok(false)) => Ok(None),
        Some(v) => {
            if let Ok(name) = v.extract::<String>() {
                if name == "survey_card" {
                    return Ok(Some(BoardSpec::survey_card()));
                }
                return Err(invalid(format!(
                    "unknown board `{name}`; pass \"survey_card\", False or a dict of BoardSpec fields"
                )));
            }
            Ok(Some(from_py::<BoardSpec>(v)?))
        }
    }
}

/// Find markers of the given families. Keyword arguments are
/// `DetectParams` fields. Each detection is a dict: `id`, `family`,
/// `corners` (top-left, top-right, bottom-right, bottom-left), `rotation`,
/// `distance`, `fit_px`.
#[pyfunction]
#[pyo3(signature = (gray, families=None, **params))]
fn detect<'py>(
    py: Python<'py>,
    gray: &Gray,
    families: Option<Vec<String>>,
    params: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let families = families.unwrap_or_else(|| vec!["5x5_100".to_string()]);
    let dicts = families
        .iter()
        .map(|f| dictionary(f))
        .collect::<PyResult<Vec<_>>>()?;
    let refs: Vec<&Dictionary> = dicts.iter().collect();
    let params: DetectParams = options_from_kwargs(params)?;
    let found = py.detach(|| cv_core::detect(&gray.inner, &refs, &params));
    to_py(py, &found)
}

/// Everything a survey measures in one picture: the swatch markers, the
/// board's markers and its located corners, and the edge blur. Returns a
/// dict with `detections`, `corners`, `blur` and `observations` (the
/// latter in the view-set form `View.observations` carries). `board` is
/// the survey card unless a dict of `BoardSpec` fields or `False` (no
/// board); `detect` and `corners` take dicts of `DetectParams` /
/// `CornerParams` overrides.
#[pyfunction]
#[pyo3(signature = (gray, swatches="5x5_100", board=None, detect=None, corners=None))]
fn observe<'py>(
    py: Python<'py>,
    gray: &Gray,
    swatches: Option<&str>,
    board: Option<&Bound<'py, PyAny>>,
    detect: Option<&Bound<'py, PyDict>>,
    corners: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let options = ObserveOptions {
        swatches: swatches.map(dictionary).transpose()?,
        board: board_spec(board)?,
        detect: options_from_dict::<DetectParams>(detect)?,
        corners: options_from_dict::<CornerParams>(corners)?,
    };
    let observed = py.detach(|| cv_core::observe(&gray.inner, &options));
    let d = PyDict::new(py);
    d.set_item("detections", to_py(py, &observed.detections)?)?;
    d.set_item("corners", to_py(py, &observed.corners)?)?;
    d.set_item("blur", to_py(py, &observed.blur)?)?;
    d.set_item("observations", to_py(py, &observed.to_observations())?)?;
    Ok(d)
}

/// Pose one photograph against a view set's marker field. `file` is the
/// picture file's bytes (for its EXIF; may be empty). Returns a dict:
/// `exif`, `seed` (the camera the solve started from), `seed_source`,
/// `detections`, `pose` (camera_to_world, camera, rms_px, markers,
/// corners_used, focal, k1) or None with `error`, and `warnings`.
#[pyfunction]
#[pyo3(signature = (gray, viewset, file=None, family="5x5_100", intrinsics=None, fov_deg=70.0, solve_focal=false, solve_distortion=false, detect=None))]
#[allow(clippy::too_many_arguments)]
fn solve_still<'py>(
    py: Python<'py>,
    gray: &Gray,
    viewset: &ViewSet,
    file: Option<&[u8]>,
    family: &str,
    intrinsics: Option<&Camera>,
    fov_deg: f64,
    solve_focal: bool,
    solve_distortion: bool,
    detect: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let options = StillOptions {
        dictionary: dictionary(family)?,
        intrinsics: intrinsics.map(|c| c.inner.clone()),
        fov_deg,
        solve_focal,
        solve_distortion,
        detect: options_from_dict::<DetectParams>(detect)?,
    };
    let set = viewset.inner.clone();
    let file = file.unwrap_or(&[]);
    let solve = py.detach(|| cv_core::solve_still(&gray.inner, file, &set, &options));
    let d = PyDict::new(py);
    d.set_item("exif", to_py(py, &solve.exif)?)?;
    d.set_item("seed", Camera::from(solve.seed.clone()))?;
    d.set_item("seed_source", &solve.seed_source)?;
    d.set_item("detections", to_py(py, &solve.detections)?)?;
    match &solve.pose {
        Ok(pose) => {
            d.set_item("pose", to_py(py, pose)?)?;
            d.set_item("error", py.None())?;
        }
        Err(error) => {
            d.set_item("pose", py.None())?;
            d.set_item("error", error)?;
        }
    }
    d.set_item("warnings", &solve.warnings)?;
    Ok(d)
}

/// The survey: solve the marker field, the cameras and every view's pose
/// from the views' observations. Keyword arguments are `SurveyOptions`
/// fields. Returns the solved view set and the report as a dict.
#[pyfunction]
#[pyo3(signature = (viewset, **options))]
fn survey<'py>(
    py: Python<'py>,
    viewset: &ViewSet,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<(ViewSet, Bound<'py, PyAny>)> {
    let options: SurveyOptions = options_from_kwargs(options)?;
    let mut set = (*viewset.inner).clone();
    let report = py
        .detach(|| cv_core::survey(&mut set, &options))
        .map_err(runtime)?;
    Ok((ViewSet::wrap(set), to_py(py, &report)?))
}

fn posed_view(camera_to_world: &Bound<'_, PyAny>) -> PyResult<volumetric_abi::viewset::View> {
    let pose = pose_from_py(camera_to_world)?;
    Ok(volumetric_abi::viewset::View {
        id: "render".to_string(),
        camera: 0,
        camera_to_world: Some(pose),
        time: None,
        image: None,
        depth: None,
        depth_unit_m: 0.001,
        mask: None,
        tags: Vec::new(),
        observations: None,
        shot: None,
        source: None,
    })
}

/// Render markers (dicts with `id`, `size_m`, `corners` (4,3), as
/// `ViewSet.markers` gives them) as a camera at `camera_to_world` (3,4)
/// sees them. Keyword arguments are `Render` fields (background, ink,
/// paper, supersample, blur_sigma, margin_cells).
#[pyfunction]
#[pyo3(signature = (camera, camera_to_world, markers, family="5x5_100", **options))]
fn render_markers<'py>(
    py: Python<'py>,
    camera: &Camera,
    camera_to_world: &Bound<'py, PyAny>,
    markers: &Bound<'py, PyList>,
    family: &str,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Gray> {
    let markers: Vec<Marker> = from_py(markers.as_any())?;
    let dict = dictionary(family)?;
    let options: Render = options_from_kwargs(options)?;
    let view = posed_view(camera_to_world)?;
    let picture =
        py.detach(|| cv_core::board::render(&camera.inner, &view, &markers, &dict, &options));
    Ok(Gray { inner: picture })
}

/// Render a board (the survey card unless `spec` is a dict of `BoardSpec`
/// fields) placed with its `origin` corner and unit `right` / `down`
/// directions in the world, as a camera at `camera_to_world` sees it.
/// Keyword arguments are `Render` fields.
#[pyfunction]
#[pyo3(signature = (camera, camera_to_world, origin, right, down, spec=None, **options))]
#[allow(clippy::too_many_arguments)]
fn render_board<'py>(
    py: Python<'py>,
    camera: &Camera,
    camera_to_world: &Bound<'py, PyAny>,
    origin: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
    down: &Bound<'py, PyAny>,
    spec: Option<&Bound<'py, PyAny>>,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Gray> {
    let spec = board_spec(spec)?.ok_or_else(|| invalid("render_board needs a board"))?;
    let board = PlacedBoard::new(
        spec,
        vec3_from_py(origin, "origin")?,
        vec3_from_py(right, "right")?,
        vec3_from_py(down, "down")?,
    );
    let options: Render = options_from_kwargs(options)?;
    let view = posed_view(camera_to_world)?;
    let picture =
        py.detach(|| cv_core::board::render_board(&camera.inner, &view, &board, &options));
    Ok(Gray { inner: picture })
}

/// The survey card's spec as a dict.
#[pyfunction]
fn survey_card<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &BoardSpec::survey_card())
}

/// A board's corner ids and positions on its plane (right = x, down = y,
/// metres from the origin corner), from a spec (the survey card unless a
/// dict of `BoardSpec` fields).
#[pyfunction]
#[pyo3(signature = (spec=None))]
fn board_corners<'py>(
    py: Python<'py>,
    spec: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let spec = board_spec(spec)?.ok_or_else(|| invalid("board_corners needs a board"))?;
    let placed = PlacedBoard::new(spec, [0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let count = placed.spec.n_corners();
    let corners: Vec<(u32, [f64; 3])> = (0..count)
        .filter_map(|id| placed.corner_world(id).map(|p| (id, p)))
        .collect();
    to_py(py, &corners)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Gray>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(observe, m)?)?;
    m.add_function(wrap_pyfunction!(solve_still, m)?)?;
    m.add_function(wrap_pyfunction!(survey, m)?)?;
    m.add_function(wrap_pyfunction!(render_markers, m)?)?;
    m.add_function(wrap_pyfunction!(render_board, m)?)?;
    m.add_function(wrap_pyfunction!(survey_card, m)?)?;
    m.add_function(wrap_pyfunction!(board_corners, m)?)?;
    Ok(())
}
