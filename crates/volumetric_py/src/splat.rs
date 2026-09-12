//! Splats: a trained Gaussian or surfel field as its columns.

use std::path::Path;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use volumetric_abi::splat as abi;

use crate::{array2, array3, runtime, to_py};

/// A splat: `count` primitives with `means` (n,3), `scales` (n,3) (log),
/// `quats` (n,4), `opacities` (n,) (logit), `sh0` (n,3), `sh_rest`
/// (n,3,k) channel-major, and `normals` (n,3) or None.
#[pyclass(module = "volumetric")]
pub struct Splat {
    inner: Arc<abi::Splat>,
}

impl Splat {
    pub(crate) fn decode(bytes: &[u8]) -> PyResult<Self> {
        let splat = abi::decode_splat(bytes).map_err(runtime)?;
        Ok(Self {
            inner: Arc::new(splat),
        })
    }
}

#[pymethods]
impl Splat {
    /// Load a `.vsplat`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(Path::new(path)).map_err(runtime)?;
        Self::decode(&bytes)
    }

    /// A splat from its encoded bytes.
    #[staticmethod]
    #[pyo3(name = "decode")]
    fn decode_py(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Save as a `.vsplat`.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(Path::new(path), abi::encode_splat(&self.inner)).map_err(runtime)
    }

    /// The encoded bytes.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &abi::encode_splat(&self.inner))
    }

    /// `gaussian` or `surfel`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind.name()
    }
    #[getter]
    fn sh_degree(&self) -> u32 {
        self.inner.sh_degree
    }
    #[getter]
    fn count(&self) -> u32 {
        self.inner.count
    }
    #[getter]
    fn training(&self) -> &str {
        &self.inner.training
    }
    #[getter]
    fn views_hash(&self) -> &str {
        &self.inner.views_hash
    }
    #[getter]
    fn world_up<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.world.up)
    }
    #[getter]
    fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.provenance)
    }

    #[getter]
    fn means<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        array2(py, self.inner.means.clone(), 3)
    }
    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        array2(py, self.inner.scales.clone(), 3)
    }
    #[getter]
    fn quats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        array2(py, self.inner.quats.clone(), 4)
    }
    #[getter]
    fn opacities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.opacities)
    }
    #[getter]
    fn sh0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        array2(py, self.inner.sh0.clone(), 3)
    }
    #[getter]
    fn sh_rest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let per_channel = abi::sh_rest_per_point(self.inner.sh_degree) / 3;
        array3(py, self.inner.sh_rest.clone(), 3, per_channel.max(1))
    }
    #[getter]
    fn normals<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        if self.inner.normals.is_empty() {
            return Ok(None);
        }
        Ok(Some(array2(py, self.inner.normals.clone(), 3)?))
    }

    fn __len__(&self) -> usize {
        self.inner.count as usize
    }

    fn __repr__(&self) -> String {
        format!(
            "Splat({} {} primitives, SH degree {})",
            self.inner.count,
            self.inner.kind.name(),
            self.inner.sh_degree
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Splat>()?;
    Ok(())
}
