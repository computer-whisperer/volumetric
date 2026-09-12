//! View sets: posed photographs, their cameras, and the marker field.

use std::path::Path;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use volumetric_abi::viewset::{self as abi, CameraModel, Distortion};

use crate::{array2, array3, from_py, invalid, runtime, to_py};

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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ViewSet>()?;
    m.add_class::<Camera>()?;
    m.add_class::<View>()?;
    Ok(())
}
