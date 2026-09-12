//! Projects: build a pipeline of steps, run it, read the exports.

use std::path::Path;

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use volumetric::project_edit::{self, InputValue};
use volumetric::{AssetTypeHint, Environment, ExecutionInput, LoadedAsset};
use volumetric_abi::fea::decode_fea_mesh;
use volumetric_abi::trimesh::decode_tri_mesh;

use crate::{array2, from_py, invalid, runtime, to_py};

/// A `.vproj`: imports, a timeline of operator steps and the ids exported.
#[pyclass(module = "volumetric")]
pub struct Project {
    pub(crate) inner: volumetric::Project,
}

fn op_bytes(operator: &Bound<'_, PyAny>) -> PyResult<(String, Vec<u8>)> {
    if let Ok(bytes) = operator.extract::<Vec<u8>>() {
        return Ok(("operator".to_string(), bytes));
    }
    let spec: String = operator
        .extract()
        .map_err(|_| invalid("operator is a bundled name, a .wasm path or wasm bytes"))?;
    if let Some(asset) = volumetric_assets::get_operator(&spec) {
        return Ok((asset.name.to_string(), asset.bytes.to_vec()));
    }
    let path = Path::new(&spec);
    if path.exists() {
        let bytes = std::fs::read(path).map_err(runtime)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("operator")
            .to_string();
        return Ok((name, bytes));
    }
    let known: Vec<&str> = volumetric_assets::operators()
        .iter()
        .map(|a| a.name)
        .collect();
    Err(invalid(format!(
        "unknown operator `{spec}`; bundled: {}",
        known.join(", ")
    )))
}

fn input_value(value: &Bound<'_, PyAny>) -> PyResult<InputValue> {
    if value.is_none() {
        return Ok(InputValue::Unwired);
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(InputValue::Bytes(bytes.as_bytes().to_vec()));
    }
    if let Ok(id) = value.extract::<String>() {
        return Ok(InputValue::Asset(id));
    }
    Ok(InputValue::Json(from_py(value)?))
}

#[pymethods]
impl Project {
    /// An empty project.
    #[new]
    fn new() -> Self {
        Self {
            inner: volumetric::Project::new(),
        }
    }

    /// Load a `.vproj`.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let inner = volumetric::Project::load_from_file(Path::new(path)).map_err(runtime)?;
        Ok(Self { inner })
    }

    /// A project from its CBOR bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = volumetric::Project::from_cbor(data).map_err(runtime)?;
        Ok(Self { inner })
    }

    /// Save as a `.vproj`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save_to_file(Path::new(path)).map_err(runtime)
    }

    /// The project's CBOR bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.to_cbor().map_err(runtime)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Every asset id the project declares (imports and step outputs) with
    /// its kind.
    fn asset_ids(&self) -> Vec<(String, Option<String>)> {
        self.inner
            .declared_assets()
            .into_iter()
            .map(|(id, hint)| (id, hint.map(|h| h.to_string())))
            .collect()
    }

    /// The ids `run` returns.
    #[getter]
    fn exports(&self) -> Vec<String> {
        self.inner.exports().to_vec()
    }

    /// The timeline: one dict per step with `operator`, `inputs` (asset
    /// ids, or `None` for an unwired slot, or the inline byte count) and
    /// `outputs`.
    fn steps<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .timeline()
            .iter()
            .map(|step| {
                let d = PyDict::new(py);
                d.set_item("operator", &step.operator_id)?;
                let inputs: Vec<Py<PyAny>> = step
                    .inputs
                    .iter()
                    .map(|input| match input {
                        ExecutionInput::AssetRef(id) => id.as_str().into_py_any(py),
                        ExecutionInput::Inline(bytes) if bytes.is_empty() => Ok(py.None()),
                        ExecutionInput::Inline(bytes) => bytes.len().into_py_any(py),
                    })
                    .collect::<PyResult<_>>()?;
                d.set_item("inputs", inputs)?;
                d.set_item("outputs", &step.outputs)?;
                Ok(d)
            })
            .collect()
    }

    /// Import a model (a wasm module) and return its id.
    #[pyo3(signature = (wasm, id=None))]
    fn add_model(&mut self, wasm: &[u8], id: Option<&str>) -> PyResult<String> {
        if !wasm.starts_with(b"\0asm") {
            return Err(invalid("a model is a wasm module"));
        }
        Ok(self
            .inner
            .insert_model(id.unwrap_or("model"), wasm.to_vec()))
    }

    /// Import bytes as an asset of `kind` (`lua`, `wgsl`, `config`,
    /// `f64map`, `blob`, `viewset` or `splat`) and return its id. An
    /// `f64map` may be given as a dict of numbers.
    #[pyo3(signature = (data, kind="blob", id=None))]
    fn add_asset(
        &mut self,
        data: &Bound<'_, PyAny>,
        kind: &str,
        id: Option<&str>,
    ) -> PyResult<String> {
        let kind = project_edit::parse_asset_kind(kind).map_err(invalid)?;
        let bytes = if let Ok(bytes) = data.cast::<PyBytes>() {
            bytes.as_bytes().to_vec()
        } else if kind == AssetTypeHint::F64Map {
            let json: serde_json::Value = from_py(data)?;
            project_edit::encode_json_f64_map(&json, "F64Map asset").map_err(invalid)?
        } else {
            return Err(invalid("asset data is bytes (or a dict for an f64map)"));
        };
        let base = id.unwrap_or(match kind {
            AssetTypeHint::LuaSource => "lua",
            AssetTypeHint::WgslSource => "wgsl",
            AssetTypeHint::Config => "config",
            AssetTypeHint::F64Map => "values",
            AssetTypeHint::ViewSet => "views",
            AssetTypeHint::Splat => "splat",
            AssetTypeHint::Mechanism => "mechanism",
            AssetTypeHint::Assembly => "assembly",
            _ => "asset",
        });
        project_edit::add_asset(&mut self.inner, base, kind, bytes).map_err(invalid)
    }

    /// Import a view set (for look-through, audits and `render(through=)`)
    /// and return its id.
    #[pyo3(signature = (views, id="views"))]
    fn add_views(&mut self, views: &crate::viewset::ViewSet, id: &str) -> PyResult<String> {
        let bytes = volumetric_abi::viewset::encode_viewset(&views.inner);
        project_edit::add_asset(&mut self.inner, id, AssetTypeHint::ViewSet, bytes).map_err(invalid)
    }

    /// Merge `updates` (field path → value, checked against the operator's
    /// declared schema; group sub-fields use dotted paths) into the
    /// configuration of the step `step` picks: a 0-based index, or a
    /// substring of the step's operator id matching exactly one step.
    /// Returns what changed as `{path: (previous, value)}`.
    fn set_config<'py>(
        &mut self,
        py: Python<'py>,
        step: &Bound<'py, PyAny>,
        updates: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let selector = if let Ok(index) = step.extract::<usize>() {
            index.to_string()
        } else {
            step.extract::<String>()
                .map_err(|_| invalid("step is an index or an operator-id substring"))?
        };
        let updates: serde_json::Value = from_py(updates.as_any())?;
        let serde_json::Value::Object(entries) = updates else {
            return Err(invalid("updates is a dict of field: value"));
        };
        let (_, changes) =
            project_edit::set_config(&mut self.inner, &selector, &entries).map_err(invalid)?;
        let out = PyDict::new(py);
        for change in changes {
            out.set_item(change.path, (change.previous, change.value))?;
        }
        Ok(out)
    }

    /// Append a step. `operator` is a bundled name, a `.wasm` path or wasm
    /// bytes; `inputs` has one entry per declared slot: an asset id
    /// (`str`), `None` to leave an optional slot unwired, `bytes` to pass
    /// raw, or a JSON-like value (dict, list, number) coerced by the slot's
    /// type — a dict for a CBOR configuration or an F64Map, a list for a
    /// VecF64. `output` names the first output (the rest take its declared
    /// suffixes). Returns the output ids, exported unless `export` is
    /// false.
    #[pyo3(signature = (operator, inputs, output=None, export=true))]
    fn add_op(
        &mut self,
        operator: &Bound<'_, PyAny>,
        inputs: Vec<Bound<'_, PyAny>>,
        output: Option<String>,
        export: bool,
    ) -> PyResult<Vec<String>> {
        let (name, bytes) = op_bytes(operator)?;
        let inputs = inputs
            .iter()
            .map(input_value)
            .collect::<PyResult<Vec<_>>>()?;
        let added =
            project_edit::add_operation(&mut self.inner, &name, bytes, inputs, output, export)
                .map_err(invalid)?;
        Ok(added.output_ids)
    }

    /// Structural problems, one string each (empty when the project is
    /// sound).
    fn validate(&self) -> Vec<String> {
        self.inner
            .validate()
            .iter()
            .map(|issue| issue.to_string())
            .collect()
    }

    /// Run every step and return the exports as `{id: Asset}`. With
    /// `remote`, the run happens on that daemon (`http://host:port`).
    #[pyo3(signature = (remote=None))]
    fn run(&self, py: Python<'_>, remote: Option<&str>) -> PyResult<Py<PyDict>> {
        let mut project = self.inner.clone();
        let exports = py.detach(move || run_exports(&mut project, remote))?;
        let out = PyDict::new(py);
        for asset in exports {
            out.set_item(asset.id().to_string(), Asset { inner: asset })?;
        }
        Ok(out.unbind())
    }
}

pub(crate) fn run_exports(
    project: &mut volumetric::Project,
    remote: Option<&str>,
) -> PyResult<Vec<LoadedAsset>> {
    let Some(address) = remote else {
        project.seed_build_cache(volumetric::build_cache::global());
        let mut env = Environment::new();
        let never = std::sync::atomic::AtomicBool::new(false);
        return project
            .run_monitored_with_artifacts(&mut env, &never, &|_| {}, &|_| {})
            .map_err(|e| runtime(format!("project execution failed: {e}")));
    };
    project.baked = None;
    let client = volumetric_protocol::DaemonClient::new(address);
    client
        .info()
        .map_err(|e| runtime(format!("remote daemon at {address} is not usable: {e}")))?;
    let outcome = client
        .run(
            &volumetric_protocol::JobRequest::RunProject {
                project: project.clone(),
            },
            &|| false,
            &|_| {},
        )
        .map_err(|e| runtime(format!("remote run on {address} failed: {e}")))?;
    match outcome {
        volumetric_protocol::JobOutcome::Success {
            output: volumetric_protocol::JobOutput::RunProject { exports },
            ..
        } => Ok(exports
            .into_iter()
            .map(volumetric_protocol::ExportedAsset::into_loaded)
            .collect()),
        volumetric_protocol::JobOutcome::Success { .. } => Err(runtime(
            "daemon returned the wrong output kind for a project run",
        )),
        volumetric_protocol::JobOutcome::Failed { error } => {
            Err(runtime(format!("project execution failed: {error}")))
        }
        volumetric_protocol::JobOutcome::Cancelled => Err(runtime("remote run was cancelled")),
    }
}

/// One exported asset of a run.
#[pyclass(module = "volumetric")]
pub struct Asset {
    pub(crate) inner: LoadedAsset,
}

#[pymethods]
impl Asset {
    #[getter]
    fn id(&self) -> &str {
        self.inner.id()
    }

    /// The kind's name: `Model`, `FeaMesh`, `TriMesh`, `F64Map`,
    /// `VecF64(n)`, `Subspace`, `ViewSet`, `Splat`, `Binary`, ...
    #[getter]
    fn kind(&self) -> String {
        self.inner
            .type_hint()
            .map(|h| h.to_string())
            .unwrap_or_else(|| "Binary".to_string())
    }

    /// The raw bytes (a model's wasm, a mesh's encoding, a blob).
    #[getter]
    fn bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.data())
    }

    /// Advisories the producing operator posted.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings().to_vec()
    }

    /// The decoded value of a small typed asset: an `F64Map` as a dict, a
    /// `VecF64` as a float array, a `Subspace` as a dict with `origin` and
    /// `basis` arrays; `None` for the rest.
    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.inner.type_hint() {
            Some(AssetTypeHint::VecF64(_)) => {
                let values = project_edit::vec_f64(self.inner.data());
                Ok(Some(PyArray1::from_vec(py, values).into_any()))
            }
            Some(AssetTypeHint::Subspace) => {
                let subspace = volumetric_abi::subspace::decode_subspace(self.inner.data())
                    .map_err(runtime)?;
                let d = PyDict::new(py);
                d.set_item("dimensions", subspace.dimensions)?;
                d.set_item("rank", subspace.rank())?;
                let ambient = subspace.ambient().max(1);
                d.set_item("origin", PyArray1::from_vec(py, subspace.origin.clone()))?;
                d.set_item("basis", array2(py, subspace.basis.clone(), ambient)?)?;
                Ok(Some(d.into_any()))
            }
            Some(AssetTypeHint::F64Map) => {
                let map = volumetric_abi::f64_map::decode(self.inner.data()).map_err(runtime)?;
                Ok(Some(to_py(py, &map)?))
            }
            _ => Ok(None),
        }
    }

    /// A `FeaMesh` or `TriMesh` as arrays; a `Model` meshed with the
    /// adaptive surface nets the CLI's `mesh` uses, at `base_resolution ·
    /// 2^max_depth` cells across its bounds (128 by default), with sharp
    /// edges recovered when `sharp_edges` (creases sharper than
    /// `sharp_angle` degrees) and simplified to `simplify` cells of error
    /// unless None. A model's mesh carries its normals as the `normal`
    /// node field.
    #[pyo3(signature = (base_resolution=8, max_depth=4, sharp_edges=false, sharp_angle=15.0, simplify=Some(1.0)))]
    fn mesh(
        &self,
        py: Python<'_>,
        base_resolution: usize,
        max_depth: usize,
        sharp_edges: bool,
        sharp_angle: f64,
        simplify: Option<f64>,
    ) -> PyResult<Mesh> {
        match self.inner.type_hint() {
            Some(AssetTypeHint::Model) => {
                let sharp_features = sharp_edges.then(|| {
                    let mut config = volumetric::sharp_features::SharpFeatureConfig::default();
                    config.segmentation.max_normal_jump_deg = sharp_angle;
                    config
                });
                let config = volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2 {
                    base_resolution,
                    max_depth,
                    sharp_features,
                    decimation: simplify.map(|tolerance| {
                        volumetric::mesh_decimation::DecimationConfig {
                            error_tolerance_cells: tolerance,
                            ..Default::default()
                        }
                    }),
                    ..Default::default()
                };
                let wasm = self.inner.data_arc();
                let result = py
                    .detach(move || {
                        volumetric::generate_adaptive_mesh_v2_from_bytes(&wasm, &config)
                    })
                    .map_err(runtime)?;
                let nodes: Vec<f64> = result
                    .vertices
                    .iter()
                    .flat_map(|v| [f64::from(v.0), f64::from(v.1), f64::from(v.2)])
                    .collect();
                let normals: Vec<f64> = result
                    .normals
                    .iter()
                    .flat_map(|n| [f64::from(n.0), f64::from(n.1), f64::from(n.2)])
                    .collect();
                let node_fields = PyDict::new(py);
                node_fields.set_item("normal", array2(py, normals, 3)?)?;
                Ok(Mesh {
                    kind: "Tri3".to_string(),
                    nodes: array2(py, nodes, 3)?.unbind(),
                    elements: array2(py, result.indices, 3)?.unbind(),
                    node_fields: node_fields.unbind(),
                    element_fields: PyDict::new(py).unbind(),
                })
            }
            Some(AssetTypeHint::FeaMesh) => {
                let mesh = decode_fea_mesh(self.inner.data()).map_err(runtime)?;
                let per = mesh.element_kind.node_count();
                Ok(Mesh {
                    kind: format!("{:?}", mesh.element_kind),
                    nodes: array2(py, mesh.node_positions, 3)?.unbind(),
                    elements: array2(py, mesh.connectivity, per)?.unbind(),
                    node_fields: fields(py, &mesh.node_fields)?,
                    element_fields: fields(py, &mesh.element_fields)?,
                })
            }
            Some(AssetTypeHint::TriMesh) => {
                let mesh = decode_tri_mesh(self.inner.data()).map_err(runtime)?;
                Ok(Mesh {
                    kind: "Tri3".to_string(),
                    nodes: array2(py, mesh.positions, 3)?.unbind(),
                    elements: array2(py, mesh.indices, 3)?.unbind(),
                    node_fields: fields(py, &mesh.vertex_fields)?,
                    element_fields: fields(py, &mesh.face_fields)?,
                })
            }
            other => Err(invalid(format!(
                "{} is {}, not a mesh",
                self.inner.id(),
                other
                    .map(|h| h.to_string())
                    .unwrap_or_else(|| "Binary".into())
            ))),
        }
    }

    /// A model's dimensionality (3 for a volume, 2 for a sketch).
    #[getter]
    fn dimensions(&self) -> PyResult<u32> {
        Ok(self.executor()?.dimensions())
    }

    /// A model's bounds as (min, max) arrays of its dimensionality.
    fn bounds<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let mut executor = self.executor()?;
        let bounds = executor.get_bounds_nd().map_err(runtime)?;
        let n = bounds.dimensions();
        let lo: Vec<f64> = (0..n).map(|i| bounds.min(i)).collect();
        let hi: Vec<f64> = (0..n).map(|i| bounds.max(i)).collect();
        Ok((PyArray1::from_vec(py, lo), PyArray1::from_vec(py, hi)))
    }

    /// A model's occupancy value at points (n,k), (n,) float32: inside
    /// where > 0.5 (see `occupied`).
    fn sample<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_vec(py, self.values(py, points)?))
    }

    /// Whether a model is occupied at points (n,k), (n,) bool.
    fn occupied<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let flags: Vec<bool> = self
            .values(py, points)?
            .into_iter()
            .map(volumetric::is_occupied)
            .collect();
        Ok(PyArray1::from_vec(py, flags))
    }

    /// A `ViewSet` asset decoded.
    fn viewset(&self) -> PyResult<crate::viewset::ViewSet> {
        crate::viewset::ViewSet::decode(self.inner.data())
    }

    /// A `Splat` asset decoded.
    fn splat(&self) -> PyResult<crate::splat::Splat> {
        crate::splat::Splat::decode(self.inner.data())
    }

    fn __repr__(&self) -> String {
        format!(
            "Asset({:?}, {}, {} bytes)",
            self.inner.id(),
            self.kind(),
            self.inner.data().len()
        )
    }
}

impl Asset {
    fn values(&self, py: Python<'_>, points: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<f32>> {
        let mut executor = self.executor()?;
        let k = executor.dimensions() as usize;
        let view = points.as_array();
        if view.ncols() != k {
            return Err(invalid(format!(
                "{} is {k}D: points are (n,{k})",
                self.inner.id()
            )));
        }
        let pts: Vec<Vec<f64>> = view.rows().into_iter().map(|r| r.to_vec()).collect();
        py.detach(move || {
            pts.iter()
                .map(|p| executor.sample_nd(p))
                .collect::<Result<Vec<f32>, _>>()
        })
        .map_err(runtime)
    }

    fn executor(&self) -> PyResult<volumetric::wasm::native::NativeModelExecutor> {
        if self.inner.type_hint() != Some(AssetTypeHint::Model) {
            return Err(invalid(format!(
                "{} is {}, not a model",
                self.inner.id(),
                self.kind()
            )));
        }
        volumetric::wasm::native::NativeModelExecutor::new(self.inner.data())
            .map_err(|e| runtime(format!("failed to instantiate {}: {e}", self.inner.id())))
    }
}

fn fields(py: Python<'_>, fields: &[volumetric_abi::fea::FeaField]) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    for field in fields {
        d.set_item(
            &field.name,
            array2(py, field.data.clone(), field.components.max(1))?,
        )?;
    }
    Ok(d.unbind())
}

/// A mesh as arrays: `nodes` (n,3) float64, `elements` (m,k) uint32 with
/// k nodes per element (`kind` says which: `Tri3`, `Hex8`, `Bar2`,
/// `Point1`), and per-node / per-element fields as `{name: (n,c)}`.
#[pyclass(module = "volumetric")]
pub struct Mesh {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    nodes: Py<PyArray2<f64>>,
    #[pyo3(get)]
    elements: Py<PyArray2<u32>>,
    #[pyo3(get)]
    node_fields: Py<PyDict>,
    #[pyo3(get)]
    element_fields: Py<PyDict>,
}

#[pymethods]
impl Mesh {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Mesh({}, {} nodes, {} elements)",
            self.kind,
            self.nodes.bind(py).shape()[0],
            self.elements.bind(py).shape()[0]
        )
    }
}

/// The bundled operators' names.
#[pyfunction]
fn operators() -> Vec<String> {
    volumetric_assets::operators()
        .iter()
        .map(|a| a.name.to_string())
        .collect()
}

/// The bundled models' names.
#[pyfunction]
fn models() -> Vec<String> {
    volumetric_assets::models()
        .iter()
        .map(|a| a.name.to_string())
        .collect()
}

/// A bundled model's wasm bytes, for `Project.add_model`.
#[pyfunction]
fn model_bytes<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
    let asset = volumetric_assets::get_model(name)
        .ok_or_else(|| invalid(format!("unknown bundled model `{name}`")))?;
    Ok(PyBytes::new(py, asset.bytes))
}

/// An operator's metadata as a dict: `name`, `version`, `display_name`,
/// `description`, `category`, `docs`, `inputs` (one dict per slot with
/// `name` and `type`), `variadic` (the slot that takes one or more, or
/// None) and `outputs` (names).
#[pyfunction]
fn operator_info<'py>(
    py: Python<'py>,
    operator: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let (_, bytes) = op_bytes(operator)?;
    let metadata = volumetric::operator_metadata_from_wasm_bytes(&bytes).map_err(runtime)?;
    let d = PyDict::new(py);
    d.set_item("name", &metadata.name)?;
    d.set_item("version", &metadata.version)?;
    d.set_item("display_name", &metadata.display_name)?;
    d.set_item("description", &metadata.description)?;
    d.set_item("category", &metadata.category)?;
    d.set_item("docs", &metadata.docs)?;
    let inputs: Vec<Bound<'py, PyDict>> = metadata
        .inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let slot = PyDict::new(py);
            slot.set_item("name", metadata.input_name(i))?;
            slot.set_item("type", project_edit::input_type_label(input))?;
            if let volumetric::OperatorMetadataInput::CBORConfiguration(cddl) = input {
                slot.set_item("cddl", cddl)?;
            }
            Ok(slot)
        })
        .collect::<PyResult<_>>()?;
    d.set_item("inputs", inputs)?;
    d.set_item("variadic", metadata.variadic_slot())?;
    d.set_item("outputs", &metadata.output_names)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Project>()?;
    m.add_class::<Asset>()?;
    m.add_class::<Mesh>()?;
    m.add_function(wrap_pyfunction!(operators, m)?)?;
    m.add_function(wrap_pyfunction!(models, m)?)?;
    m.add_function(wrap_pyfunction!(model_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(operator_info, m)?)?;
    Ok(())
}
