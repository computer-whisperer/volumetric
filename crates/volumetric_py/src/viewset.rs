//! View sets: posed photographs, their cameras, and the marker field.

use std::path::Path;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use view_core::crop::{CropOptions, crop, picture_of};
use view_core::manifest::{Eye, Selection};
use view_core::measure::{Plane, cast, triangulate_picks};
use view_core::subset::{Reembed, SubsetOptions, subset};
use volumetric_abi::viewset::{self as abi, CameraModel, Distortion};

use crate::{array2, array3, from_py, invalid, runtime, to_py, vec3_from_py};

/// Rows of an (n,k) array as fixed-size points.
pub(crate) fn rows<const K: usize>(
    array: &PyReadonlyArray2<'_, f64>,
    what: &str,
) -> PyResult<Vec<[f64; K]>> {
    let view = array.as_array();
    if view.ncols() != K {
        return Err(invalid(format!("{what} is an (n,{K}) array")));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|r| {
            let mut out = [0.0; K];
            for (o, v) in out.iter_mut().zip(r.iter()) {
                *o = *v;
            }
            out
        })
        .collect())
}

/// A plane from `plane=(point, normal)` or `z=height`.
fn plane_arg(plane: Option<&Bound<'_, PyAny>>, z: Option<f64>) -> PyResult<Plane> {
    match (plane, z) {
        (Some(p), None) => {
            let (point, normal): (Bound<'_, PyAny>, Bound<'_, PyAny>) = p
                .extract()
                .map_err(|_| invalid("plane is a (point, normal) pair"))?;
            Ok(Plane {
                point: vec3_from_py(&point, "the plane's point")?,
                normal: vec3_from_py(&normal, "the plane's normal")?,
            })
        }
        (None, Some(z)) => Ok(Plane::at_z(z)),
        (None, None) => Err(invalid("casting needs plane=(point, normal) or z=height")),
        (Some(_), Some(_)) => Err(invalid("give plane= or z=, not both")),
    }
}

/// A view set: cameras, views (photographs with poses and observations)
/// and the marker field they were measured against.
#[pyclass(module = "volumetric")]
pub struct ViewSet {
    pub(crate) inner: Arc<abi::ViewSet>,
}

impl ViewSet {
    pub(crate) fn decode(bytes: &[u8]) -> PyResult<Self> {
        let set = abi::decode_viewset(bytes).map_err(runtime)?;
        Ok(Self {
            inner: Arc::new(set),
        })
    }

    pub(crate) fn wrap(set: abi::ViewSet) -> Self {
        Self {
            inner: Arc::new(set),
        }
    }
}

#[pymethods]
impl ViewSet {
    /// Load a `.vviews`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(Path::new(path)).map_err(runtime)?;
        Self::decode(&bytes)
    }

    /// A view set from its encoded bytes.
    #[staticmethod]
    #[pyo3(name = "decode")]
    fn decode_py(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Save as a `.vviews`.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(Path::new(path), abi::encode_viewset(&self.inner)).map_err(runtime)
    }

    /// The encoded bytes.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &abi::encode_viewset(&self.inner))
    }

    #[getter]
    fn schema(&self) -> u32 {
        self.inner.schema
    }

    /// The world's up direction (3,).
    #[getter]
    fn world_up<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.world.up)
    }

    #[getter]
    fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.provenance)
    }

    #[getter]
    fn cameras(&self) -> Vec<Camera> {
        self.inner
            .cameras
            .iter()
            .cloned()
            .map(Camera::from)
            .collect()
    }

    #[getter]
    fn views(&self) -> Vec<View> {
        (0..self.inner.views.len())
            .map(|index| View {
                set: Arc::clone(&self.inner),
                index,
            })
            .collect()
    }

    /// The view with this id.
    fn view(&self, id: &str) -> PyResult<View> {
        let index = self
            .inner
            .views
            .iter()
            .position(|v| v.id == id)
            .ok_or_else(|| invalid(format!("no view `{id}`")))?;
        Ok(View {
            set: Arc::clone(&self.inner),
            index,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.views.len()
    }

    /// Every view's camera-to-world as (n,3,4); NaN rows for unposed views.
    fn poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let mut data = Vec::with_capacity(self.inner.views.len() * 12);
        for view in &self.inner.views {
            match view.camera_to_world {
                Some(pose) => data.extend_from_slice(&pose),
                None => data.extend(std::iter::repeat_n(f64::NAN, 12)),
            }
        }
        array3(py, data, 3, 4)
    }

    /// The markers as dicts (`id`, `size_m`, `corners`, ...).
    #[getter]
    fn markers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.markers)
    }

    /// The marker ids (n,) uint32, in the order of `marker_corners`.
    fn marker_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_iter(py, self.inner.markers.iter().map(|m| m.id))
    }

    /// The markers' corners in the world, (n,4,3).
    fn marker_corners<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let data: Vec<f64> = self
            .inner
            .markers
            .iter()
            .flat_map(|m| m.corners.iter().flatten().copied())
            .collect();
        array3(py, data, 4, 3)
    }

    /// A feature picked in two or more views, `{view_id: (u, v)}`,
    /// triangulated: the world point (3,) and each pick's ray miss in
    /// metres, in the dict's order.
    fn triangulate<'py>(
        &self,
        py: Python<'py>,
        picks: &Bound<'py, PyDict>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let mut list = Vec::new();
        for (key, value) in picks.iter() {
            let id: String = key
                .extract()
                .map_err(|_| invalid("pick keys are view ids"))?;
            let pixel: (f64, f64) = value
                .extract()
                .map_err(|_| invalid(format!("pick for `{id}` is a (u, v) pixel")))?;
            list.push((id, [pixel.0, pixel.1]));
        }
        let (point, gaps) = triangulate_picks(&self.inner, &list).map_err(invalid)?;
        Ok((
            PyArray1::from_slice(py, &point),
            PyArray1::from_vec(py, gaps),
        ))
    }

    /// Detect markers (and the card) in every view's picture — or the
    /// named `ids` — and return the set with the observations stored, plus
    /// one report dict per picture (`id, width, height, detections,
    /// corners, blur, seconds`) and the ids skipped for want of a picture.
    /// `swatches` is a family name or None; `card` is the survey card
    /// unless a spec dict, JSON text, a path to a card.json, or False;
    /// `detect` and `corners` take parameter overrides.
    #[pyo3(signature = (ids=None, swatches="5x5_100", card=None, detect=None, corners=None))]
    fn detect<'py>(
        &self,
        py: Python<'py>,
        ids: Option<Vec<String>>,
        swatches: Option<&str>,
        card: Option<&Bound<'py, PyAny>>,
        detect: Option<&Bound<'py, PyDict>>,
        corners: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(ViewSet, Vec<Bound<'py, PyDict>>, Vec<String>)> {
        let options = cv_core::ObserveOptions {
            swatches: swatches.map(crate::cv::dictionary).transpose()?,
            board: crate::cv::card_arg(card)?,
            detect: crate::options_from_dict(detect)?,
            corners: crate::options_from_dict(corners)?,
        };
        if options.swatches.is_none() && options.board.is_none() {
            return Err(invalid("nothing to look for: give swatches or a card"));
        }
        let mut set = (*self.inner).clone();
        let ids = ids.unwrap_or_default();
        let outcome = py
            .detach(|| view_core::detect::detect_views(&mut set, &ids, &options, |_, _| {}))
            .map_err(runtime)?;
        let reports = outcome
            .pictures
            .iter()
            .map(|d| {
                let r = PyDict::new(py);
                r.set_item("id", &d.id)?;
                r.set_item("width", d.width)?;
                r.set_item("height", d.height)?;
                r.set_item("detections", to_py(py, &d.seen.detections)?)?;
                r.set_item("corners", to_py(py, &d.seen.corners)?)?;
                r.set_item("blur", to_py(py, &d.seen.blur)?)?;
                r.set_item("seconds", d.seconds)?;
                Ok(r)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((ViewSet::wrap(set), reports, outcome.skipped))
    }

    /// A subset of the views as a set of its own (cameras, markers and the
    /// card shared), the views a project carries for look-through: by
    /// `ids` (in that order), every `stride`th, within `radius` metres of
    /// `near`, at most `max` (0 = all), `posed` only, carrying every one of
    /// `tags`; `embed` says what each kept view carries: `"keep"`,
    /// `"full"`, `"preview"` (a JPEG `preview_px` wide) or `"none"`.
    #[pyo3(signature = (ids=None, stride=1, near=None, radius=0.5, max=0, posed=false, tags=None, eye="both", split=None, embed="keep", preview_px=1600))]
    #[allow(clippy::too_many_arguments)]
    fn select<'py>(
        &self,
        py: Python<'py>,
        ids: Option<Vec<String>>,
        stride: usize,
        near: Option<&Bound<'py, PyAny>>,
        radius: f64,
        max: usize,
        posed: bool,
        tags: Option<Vec<String>>,
        eye: &str,
        split: Option<String>,
        embed: &str,
        preview_px: u32,
    ) -> PyResult<ViewSet> {
        let options = SubsetOptions {
            selection: Selection {
                ids: ids.unwrap_or_default(),
                stride: stride.max(1),
                near: near
                    .map(|n| vec3_from_py(n, "near").map(|p| (p, radius)))
                    .transpose()?,
                max,
                eye: match eye {
                    "left" => Eye::Left,
                    "right" => Eye::Right,
                    "both" => Eye::Both,
                    other => return Err(invalid(format!("eye `{other}`: left, right or both"))),
                },
                split,
                ..Selection::default()
            },
            posed,
            tags: tags.unwrap_or_default(),
            embed: match embed {
                "keep" => Reembed::Keep,
                "full" => Reembed::Full,
                "preview" => Reembed::Preview,
                "none" => Reembed::None,
                other => {
                    return Err(invalid(format!(
                        "embed `{other}`: keep, full, preview or none"
                    )));
                }
            },
            preview_px,
        };
        let set = Arc::clone(&self.inner);
        let (selected, _) = py.detach(move || subset(&set, &options)).map_err(invalid)?;
        Ok(ViewSet::wrap(selected))
    }

    /// The survey card (spec and solved corners) as a dict, or None.
    #[getter]
    fn board<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner
            .board
            .as_ref()
            .map(|board| to_py(py, board))
            .transpose()
    }

    fn __repr__(&self) -> String {
        let posed = self
            .inner
            .views
            .iter()
            .filter(|v| v.camera_to_world.is_some())
            .count();
        format!(
            "ViewSet({} views, {} posed, {} cameras, {} markers)",
            self.inner.views.len(),
            posed,
            self.inner.cameras.len(),
            self.inner.markers.len()
        )
    }
}

/// A camera's intrinsics.
#[pyclass(module = "volumetric", skip_from_py_object)]
#[derive(Clone)]
pub struct Camera {
    pub(crate) inner: CameraModel,
}

impl From<CameraModel> for Camera {
    fn from(inner: CameraModel) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Camera {
    /// `distortion` is a dict like `{"model": "radial", "k": [k1, k2],
    /// "p": [p1, p2]}` or `{"model": "kannala_brandt", "k": [..4]}`; None
    /// is an ideal camera.
    #[new]
    #[pyo3(signature = (width, height, fx, fy, cx, cy, label="", distortion=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        label: &str,
        distortion: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let distortion = match distortion {
            Some(value) if !value.is_none() => from_py::<Distortion>(value)?,
            _ => Distortion::None,
        };
        Ok(Self {
            inner: CameraModel {
                label: label.to_string(),
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                distortion,
            },
        })
    }

    #[getter]
    fn label(&self) -> &str {
        &self.inner.label
    }
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }
    #[getter]
    fn fx(&self) -> f64 {
        self.inner.fx
    }
    #[getter]
    fn fy(&self) -> f64 {
        self.inner.fy
    }
    #[getter]
    fn cx(&self) -> f64 {
        self.inner.cx
    }
    #[getter]
    fn cy(&self) -> f64 {
        self.inner.cy
    }
    #[getter]
    fn distortion<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.distortion)
    }

    /// Every field as a dict.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Camera({:?}, {}x{}, f {:.1}/{:.1}, c {:.1},{:.1})",
            self.inner.label,
            self.inner.width,
            self.inner.height,
            self.inner.fx,
            self.inner.fy,
            self.inner.cx,
            self.inner.cy
        )
    }
}

/// One photograph in a view set.
#[pyclass(module = "volumetric")]
pub struct View {
    set: Arc<abi::ViewSet>,
    index: usize,
}

impl View {
    fn view(&self) -> &abi::View {
        &self.set.views[self.index]
    }

    fn view_and_camera(&self) -> PyResult<(&abi::View, &CameraModel)> {
        let view = self.view();
        let camera = self
            .set
            .cameras
            .get(view.camera as usize)
            .ok_or_else(|| runtime("view names a camera the set does not have"))?;
        Ok((view, camera))
    }
}

#[pymethods]
impl View {
    #[getter]
    fn id(&self) -> &str {
        &self.view().id
    }

    /// Index into the set's cameras.
    #[getter]
    fn camera_index(&self) -> u32 {
        self.view().camera
    }

    /// The view's camera.
    #[getter]
    fn camera(&self) -> PyResult<Camera> {
        self.set
            .cameras
            .get(self.view().camera as usize)
            .cloned()
            .map(Camera::from)
            .ok_or_else(|| runtime("view names a camera the set does not have"))
    }

    /// Camera-to-world as (3,4), or None when unposed.
    #[getter]
    fn camera_to_world<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.view()
            .camera_to_world
            .map(|pose| array2(py, pose.to_vec(), 4))
            .transpose()
    }

    /// The camera's position in the world (3,), or None.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.view().position().map(|p| PyArray1::from_slice(py, &p))
    }

    #[getter]
    fn tags(&self) -> Vec<String> {
        self.view().tags.clone()
    }
    #[getter]
    fn time(&self) -> Option<f64> {
        self.view().time
    }
    #[getter]
    fn source(&self) -> Option<String> {
        self.view().source.clone()
    }
    #[getter]
    fn depth_unit_m(&self) -> f64 {
        self.view().depth_unit_m
    }

    /// The camera settings the picture was taken with, as a dict, or None.
    #[getter]
    fn shot<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.view().shot.as_ref().map(|s| to_py(py, s)).transpose()
    }

    /// What was measured in the picture (markers, board corners, blur) as
    /// a dict, or None.
    #[getter]
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.view()
            .observations
            .as_ref()
            .map(|o| to_py(py, o))
            .transpose()
    }

    /// World points (n,3) projected through the camera and its
    /// distortion, (n,2); NaN rows for points behind the camera.
    fn project<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (view, camera) = self.view_and_camera()?;
        let mut out = Vec::new();
        for p in rows::<3>(&points, "points")? {
            match view.project(camera, p) {
                Some(px) => out.extend_from_slice(&px),
                None => out.extend([f64::NAN, f64::NAN]),
            }
        }
        array2(py, out, 2)
    }

    /// The world-space unit directions pixels (n,2) look along, (n,3).
    fn ray<'py>(
        &self,
        py: Python<'py>,
        pixels: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (view, camera) = self.view_and_camera()?;
        let mut out = Vec::new();
        for px in rows::<2>(&pixels, "pixels")? {
            let d = view
                .ray(camera, px)
                .ok_or_else(|| invalid("the view is not posed"))?;
            out.extend_from_slice(&d);
        }
        array2(py, out, 3)
    }

    /// Pixels (n,2) cast onto a plane — `plane=(point, normal)` or
    /// `z=height` — as world points (n,3) and their depths (n,).
    #[pyo3(signature = (pixels, plane=None, z=None))]
    fn cast<'py>(
        &self,
        py: Python<'py>,
        pixels: PyReadonlyArray2<'py, f64>,
        plane: Option<&Bound<'py, PyAny>>,
        z: Option<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        let plane = plane_arg(plane, z)?;
        let (view, camera) = self.view_and_camera()?;
        let mut world = Vec::new();
        let mut depths = Vec::new();
        for px in rows::<2>(&pixels, "pixels")? {
            let hit = cast(view, camera, px, &plane).map_err(invalid)?;
            world.extend_from_slice(&hit.world);
            depths.push(hit.depth);
        }
        Ok((array2(py, world, 3)?, PyArray1::from_vec(py, depths)))
    }

    /// A magnified crop of the view's original picture (read through its
    /// source when only a preview is embedded) around `center` (u, v):
    /// `size` (w, h) original pixels at integer `scale`, a labelled grid
    /// every `grid` pixels (0 = none), cyan crosses at `marks` pixels and
    /// magenta crosses where `world_marks` points land. Returns a `Crop`.
    #[pyo3(signature = (center, size=(600, 400), scale=3, grid=50, marks=None, world_marks=None))]
    #[allow(clippy::too_many_arguments)]
    fn crop<'py>(
        &self,
        py: Python<'py>,
        center: &Bound<'py, PyAny>,
        size: (u32, u32),
        scale: u32,
        grid: u32,
        marks: Option<Vec<(f64, f64)>>,
        world_marks: Option<Vec<Bound<'py, PyAny>>>,
    ) -> PyResult<Crop> {
        let centre: (f64, f64) = center
            .extract()
            .map_err(|_| invalid("center is a (u, v) pixel"))?;
        let options = CropOptions {
            centre: [centre.0, centre.1],
            size,
            scale,
            grid,
            marks: marks
                .unwrap_or_default()
                .into_iter()
                .map(|(u, v)| [u, v])
                .collect(),
            world_marks: world_marks
                .unwrap_or_default()
                .iter()
                .map(|p| vec3_from_py(p, "a world mark"))
                .collect::<PyResult<_>>()?,
        };
        let (view, camera) = self.view_and_camera()?;
        let picture = picture_of(&self.set, view).map_err(runtime)?;
        let out = crop(view, camera, &picture, &options).map_err(invalid)?;
        Ok(Crop {
            image: array3(py, out.image.pixels, out.image.width as usize, 3)?.unbind(),
            origin: out.origin,
            end: out.end,
            scale: out.scale,
            verticals: out.verticals,
            horizontals: out.horizontals,
            projected: out
                .projected
                .iter()
                .map(|p| p.map(|p| (p[0], p[1])))
                .collect(),
        })
    }

    /// The embedded picture's bytes (JPEG/PNG), or None.
    fn picture<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.view().image.as_ref().map(|b| PyBytes::new(py, b))
    }

    /// The embedded picture decoded, (h,w,3) uint8.
    fn image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let bytes =
            self.view().image.as_ref().ok_or_else(|| {
                invalid(format!("view `{}` has no embedded picture", self.view().id))
            })?;
        let rgb = view_core::image::decode_rgb(bytes).map_err(runtime)?;
        array3(py, rgb.pixels, rgb.width as usize, 3)
    }

    /// The embedded depth in metres, (h,w) float32, or None.
    fn depth<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        let view = self.view();
        let Some(bytes) = view.depth.as_ref() else {
            return Ok(None);
        };
        let depth = view_core::image::decode_depth(bytes, view.depth_unit_m).map_err(runtime)?;
        Ok(Some(array2(py, depth.metres, depth.width as usize)?))
    }

    /// The embedded mask, (h,w) bool, or None.
    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<bool>>>> {
        let Some(bytes) = self.view().mask.as_ref() else {
            return Ok(None);
        };
        let mask = view_core::image::decode_mask(bytes).map_err(runtime)?;
        Ok(Some(array2(py, mask.inside, mask.width as usize)?))
    }

    /// Every field but the pictures, as a dict.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut view = self.view().clone();
        view.image = None;
        view.depth = None;
        view.mask = None;
        let d = to_py(py, &view)?;
        let d = d
            .cast_into::<PyDict>()
            .map_err(|_| runtime("view did not serialize as a dict"))?;
        d.set_item("has_image", self.view().image.is_some())?;
        d.set_item("has_depth", self.view().depth.is_some())?;
        d.set_item("has_mask", self.view().mask.is_some())?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let v = self.view();
        format!(
            "View({:?}, camera {}, {}, {})",
            v.id,
            v.camera,
            if v.camera_to_world.is_some() {
                "posed"
            } else {
                "unposed"
            },
            if v.image.is_some() {
                "with picture"
            } else {
                "no picture"
            }
        )
    }
}

/// A crop of a view's picture: `image` (h,w,3) uint8, the original pixels
/// it covers as `origin`..`end`, its `scale`, the grid lines' original
/// coordinates (`verticals`, `horizontals`) and each world mark's pixel
/// (`projected`, None when behind the camera).
#[pyclass(module = "volumetric")]
pub struct Crop {
    #[pyo3(get)]
    image: Py<PyArray3<u8>>,
    #[pyo3(get)]
    origin: (u32, u32),
    #[pyo3(get)]
    end: (u32, u32),
    #[pyo3(get)]
    scale: u32,
    #[pyo3(get)]
    verticals: Vec<u32>,
    #[pyo3(get)]
    horizontals: Vec<u32>,
    #[pyo3(get)]
    projected: Vec<Option<(f64, f64)>>,
}

#[pymethods]
impl Crop {
    /// The crop as PNG bytes.
    fn png<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let image = self.image.bind(py).readonly();
        let view = image.as_array();
        let (h, w, _) = view.dim();
        let rgb = view_core::image::Rgb {
            width: w as u32,
            height: h as u32,
            pixels: view.iter().copied().collect(),
        };
        Ok(PyBytes::new(py, &rgb.to_png().map_err(runtime)?))
    }

    /// Write the crop as a PNG.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let bytes = self.png(py)?;
        std::fs::write(path, bytes.as_bytes()).map_err(runtime)
    }

    fn __repr__(&self) -> String {
        format!(
            "Crop(({}, {})..({}, {}) at {}x)",
            self.origin.0, self.origin.1, self.end.0, self.end.1, self.scale
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Crop>()?;
    m.add_class::<ViewSet>()?;
    m.add_class::<Camera>()?;
    m.add_class::<View>()?;
    Ok(())
}
