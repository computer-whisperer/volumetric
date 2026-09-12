//! Mechanisms and assemblies: the joint tree with its kinematics, and an
//! assembly's parts and state.

use std::path::Path;
use std::sync::Arc;

use numpy::{PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use volumetric_abi::f64_map::F64Map;
use volumetric_abi::mechanism as abi;

use crate::{array2, array3, from_py, invalid, runtime, to_py};

/// A mechanism: parts joined by fixed, revolute and prismatic joints. Parts
/// are authored in the world frame at rest and axes given in world
/// coordinates at rest; a part's pose is the product of its chain's joint
/// motions, root first. States are keyed by joint name: degrees for a
/// revolute joint, metres for a prismatic one.
#[pyclass(module = "volumetric")]
pub struct Mechanism {
    inner: Arc<abi::Mechanism>,
}

impl Mechanism {
    pub(crate) fn decode(bytes: &[u8]) -> PyResult<Self> {
        let mechanism = abi::decode_mechanism(bytes).map_err(runtime)?;
        Ok(Self {
            inner: Arc::new(mechanism),
        })
    }

    fn state_of(&self, state: Option<&Bound<'_, PyDict>>) -> PyResult<F64Map> {
        match state {
            Some(state) => from_py(state.as_any()),
            None => Ok(self.inner.default_state()),
        }
    }
}

/// Every part's pose as an (n, 3, 4) array: rows of the linear part with
/// the offset as the fourth column.
fn poses_array<'py>(py: Python<'py>, poses: &[abi::Rigid]) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let mut flat = Vec::with_capacity(poses.len() * 12);
    for pose in poses {
        for (row, offset) in pose.linear.iter().zip(pose.offset) {
            flat.extend_from_slice(row);
            flat.push(offset);
        }
    }
    array3(py, flat, 3, 4)
}

#[pymethods]
impl Mechanism {
    /// Load a `.vmech`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(Path::new(path)).map_err(runtime)?;
        Self::decode(&bytes)
    }

    /// A mechanism from its encoded bytes.
    #[staticmethod]
    #[pyo3(name = "decode")]
    fn decode_py(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Save as a `.vmech`.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(Path::new(path), abi::encode_mechanism(&self.inner)).map_err(runtime)
    }

    /// The encoded bytes.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &abi::encode_mechanism(&self.inner))
    }

    /// The part names, in the order an assembly's models are given.
    #[getter]
    fn parts(&self) -> Vec<String> {
        self.inner.parts.clone()
    }

    /// The joints as dicts: `name`, `kind`, `parent`, `child`, `axis`
    /// (`origin`, `direction`) or None, `min`, `max`, `default`, `drive`.
    #[getter]
    fn joints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.joints)
    }

    /// The state keys: the moving, undriven joints in joint order (the
    /// order `velocity` reports in).
    #[getter]
    fn state_keys(&self) -> Vec<String> {
        self.inner
            .state_keys()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Every state at its default.
    #[getter]
    fn default_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.default_state())
    }

    /// The states' ranges as dicts (`key`, `default`, `min`, `max`).
    #[getter]
    fn ranges<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .parameter_specs()
            .iter()
            .map(|spec| {
                let d = PyDict::new(py);
                d.set_item("key", &spec.key)?;
                d.set_item("default", spec.default)?;
                d.set_item("min", spec.min)?;
                d.set_item("max", spec.max)?;
                Ok(d)
            })
            .collect()
    }

    /// Every part's pose at `state` (the defaults when omitted; missing
    /// keys take their default) as an (n, 3, 4) array, world <- part.
    #[pyo3(signature = (state=None))]
    fn pose<'py>(
        &self,
        py: Python<'py>,
        state: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let state = self.state_of(state)?;
        let poses = self.inner.pose(&state).map_err(invalid)?;
        poses_array(py, &poses)
    }

    /// The world velocity of `point` (world coordinates at `state`) on
    /// `part` per unit rate of each state, in `state_keys` order, as a
    /// (k, 3) array: the Jacobian a drag solves against.
    #[pyo3(signature = (part, point, state=None))]
    fn velocity<'py>(
        &self,
        py: Python<'py>,
        part: &str,
        point: [f64; 3],
        state: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let state = self.state_of(state)?;
        let rows = self.inner.velocity(&state, part, point).map_err(invalid)?;
        array2(py, rows.into_iter().flatten().collect(), 3)
    }

    /// The state that brings `local`, a point of `part` in the part's own
    /// (rest) coordinates, as near `target` (world) as the joints allow,
    /// from `state` (the assembly's defaults when omitted): what a drag
    /// solves each time the pointer moves.
    #[pyo3(signature = (part, local, target, state=None, iterations=8))]
    fn pull<'py>(
        &self,
        py: Python<'py>,
        part: &str,
        local: [f64; 3],
        target: [f64; 3],
        state: Option<&Bound<'py, PyDict>>,
        iterations: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state_of(state)?;
        let pulled = self
            .inner
            .pull(&state, part, local, target, iterations)
            .map_err(invalid)?;
        to_py(py, &pulled)
    }

    fn __repr__(&self) -> String {
        format!(
            "Mechanism({} parts, {} joints, states {:?})",
            self.inner.parts.len(),
            self.inner.joints.len(),
            self.inner.state_keys()
        )
    }
}

/// An assembly: a mechanism with its part models and one state.
#[pyclass(module = "volumetric")]
pub struct Assembly {
    inner: Arc<abi::Assembly>,
}

impl Assembly {
    pub(crate) fn decode(bytes: &[u8]) -> PyResult<Self> {
        let assembly = abi::decode_assembly(bytes).map_err(runtime)?;
        Ok(Self {
            inner: Arc::new(assembly),
        })
    }
}

#[pymethods]
impl Assembly {
    /// Load a `.vasm`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(Path::new(path)).map_err(runtime)?;
        Self::decode(&bytes)
    }

    /// An assembly from its encoded bytes.
    #[staticmethod]
    #[pyo3(name = "decode")]
    fn decode_py(data: &[u8]) -> PyResult<Self> {
        Self::decode(data)
    }

    /// Save as a `.vasm`.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(Path::new(path), abi::encode_assembly(&self.inner)).map_err(runtime)
    }

    /// The encoded bytes.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &abi::encode_assembly(&self.inner))
    }

    /// The mechanism.
    #[getter]
    fn mechanism(&self) -> Mechanism {
        Mechanism {
            inner: Arc::new(self.inner.mechanism.clone()),
        }
    }

    /// The part names, in order.
    #[getter]
    fn parts(&self) -> Vec<String> {
        self.inner.parts.iter().map(|p| p.name.clone()).collect()
    }

    /// A part's unposed model, its wasm bytes.
    fn part_model<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        let part = self
            .inner
            .parts
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| invalid(format!("`{name}` is not a part of the assembly")))?;
        Ok(PyBytes::new(py, &part.model))
    }

    /// The state the assembly is posed at, every state key present.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.state)
    }

    /// Every part's pose at the assembly's state, an (n, 3, 4) array.
    fn poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        poses_array(py, &self.inner.poses())
    }

    fn __len__(&self) -> usize {
        self.inner.parts.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Assembly({} parts, state {:?})",
            self.inner.parts.len(),
            self.inner.state
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mechanism>()?;
    m.add_class::<Assembly>()?;
    Ok(())
}
