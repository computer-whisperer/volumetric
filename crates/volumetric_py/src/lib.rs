//! `volumetric._volumetric`: the extension module. Each submodule wraps one
//! crate's functions one to one — projects (`volumetric`), pictures and
//! markers (`cv_core`), view sets and splats (`volumetric_abi`). Small
//! results come back as dicts and lists (`pythonize`), large ones as numpy
//! arrays; options structs take keyword arguments over their defaults.

use numpy::{Element, IntoPyArray, PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Serialize;
use serde::de::DeserializeOwned;

mod cv;
mod project;
mod render;
mod splat;
mod viewset;

/// An error from a crate function, as a RuntimeError with its message.
pub(crate) fn runtime<E: std::fmt::Display>(error: E) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A bad argument, as a ValueError with its message.
pub(crate) fn invalid<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Any serializable value as Python objects (structs as dicts).
pub(crate) fn to_py<'py, T: Serialize>(py: Python<'py>, value: &T) -> PyResult<Bound<'py, PyAny>> {
    pythonize::pythonize(py, value).map_err(PyErr::from)
}

/// A Python object as a serde value (dicts as structs).
pub(crate) fn from_py<T: DeserializeOwned>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    pythonize::depythonize(value).map_err(PyErr::from)
}

/// An options struct from keyword arguments over its default: every key
/// names a field (unknown keys are refused by the struct's serde
/// attributes), values take the field's type.
pub(crate) fn options_from_kwargs<T: Serialize + DeserializeOwned + Default>(
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<T> {
    let mut base = serde_json::to_value(T::default()).map_err(runtime)?;
    if let Some(kwargs) = kwargs {
        let over: serde_json::Value = from_py(kwargs.as_any())?;
        let (Some(base), Some(over)) = (base.as_object_mut(), over.as_object()) else {
            return Err(invalid("options must be keyword arguments"));
        };
        for (key, value) in over {
            base.insert(key.clone(), value.clone());
        }
    }
    serde_json::from_value(base).map_err(invalid)
}

/// An options struct from an optional dict of overrides (the same rule as
/// [`options_from_kwargs`] for nested option blocks).
pub(crate) fn options_from_dict<T: Serialize + DeserializeOwned + Default>(
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<T> {
    options_from_kwargs(dict)
}

/// A flat row-major vector as an (rows, cols) array.
pub(crate) fn array2<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let rows = data.len().checked_div(cols).unwrap_or(0);
    if rows * cols != data.len() {
        return Err(runtime(format!(
            "{} values do not fill rows of {cols}",
            data.len()
        )));
    }
    data.into_pyarray(py).reshape([rows, cols])
}

/// A flat vector as a (planes, rows, cols) array.
pub(crate) fn array3<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray3<T>>> {
    let block = rows * cols;
    let planes = data.len().checked_div(block).unwrap_or(0);
    if planes * block != data.len() {
        return Err(runtime(format!(
            "{} values do not fill blocks of {rows}x{cols}",
            data.len()
        )));
    }
    data.into_pyarray(py).reshape([planes, rows, cols])
}

/// A 3x4 camera-to-world (row-major, as the ABI stores it) from a (3,4)
/// array-like.
pub(crate) fn pose_from_py(value: &Bound<'_, PyAny>) -> PyResult<[f64; 12]> {
    let rows: Vec<Vec<f64>> = value
        .extract()
        .map_err(|_| invalid("a pose is a (3,4) array of camera-to-world"))?;
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 4) {
        return Err(invalid("a pose is a (3,4) array of camera-to-world"));
    }
    let mut out = [0.0; 12];
    for (i, row) in rows.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(row);
    }
    Ok(out)
}

/// A 3-vector from an array-like.
pub(crate) fn vec3_from_py(value: &Bound<'_, PyAny>, what: &str) -> PyResult<[f64; 3]> {
    let v: Vec<f64> = value
        .extract()
        .map_err(|_| invalid(format!("{what} is three numbers")))?;
    <[f64; 3]>::try_from(v).map_err(|_| invalid(format!("{what} is three numbers")))
}

#[pymodule]
fn _volumetric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    project::register(m)?;
    cv::register(m)?;
    viewset::register(m)?;
    splat::register(m)?;
    render::register(m)?;
    let names: Vec<String> = m
        .dir()?
        .iter()
        .filter_map(|n| n.extract::<String>().ok())
        .filter(|n| !n.starts_with('_'))
        .collect();
    m.add("__all__", names)?;
    Ok(())
}
