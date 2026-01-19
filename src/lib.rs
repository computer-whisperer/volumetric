use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;

// wasmtime imports are now used through the wasm module

pub mod wasm;

/// A triangle in 3D space with vertices and per-vertex normal vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct Triangle {
    /// The three vertices of the triangle.
    pub vertices: [(f32, f32, f32); 3],
    /// Per-vertex normal vectors (should be unit length).
    /// Each normal corresponds to the vertex at the same index.
    pub normals: [(f32, f32, f32); 3],
}

impl Triangle {
    /// Create a new triangle with the given vertices and compute the face normal automatically.
    /// The same face normal is assigned to all three vertices.
    pub fn new(vertices: [(f32, f32, f32); 3]) -> Self {
        let normal = Self::compute_face_normal(&vertices);
        Self { vertices, normals: [normal, normal, normal] }
    }

    /// Create a new triangle with explicit vertices and a single normal for all vertices.
    pub fn with_normal(vertices: [(f32, f32, f32); 3], normal: (f32, f32, f32)) -> Self {
        Self { vertices, normals: [normal, normal, normal] }
    }

    /// Create a new triangle with explicit per-vertex normals.
    pub fn with_vertex_normals(vertices: [(f32, f32, f32); 3], normals: [(f32, f32, f32); 3]) -> Self {
        Self { vertices, normals }
    }

    /// Get the face normal (average of vertex normals, or computed from geometry if degenerate).
    pub fn face_normal(&self) -> (f32, f32, f32) {
        let avg = (
            self.normals[0].0 + self.normals[1].0 + self.normals[2].0,
            self.normals[0].1 + self.normals[1].1 + self.normals[2].1,
            self.normals[0].2 + self.normals[1].2 + self.normals[2].2,
        );
        let len2 = avg.0 * avg.0 + avg.1 * avg.1 + avg.2 * avg.2;
        if len2 > 1.0e-24 {
            let inv_len = 1.0 / len2.sqrt();
            (avg.0 * inv_len, avg.1 * inv_len, avg.2 * inv_len)
        } else {
            Self::compute_face_normal(&self.vertices)
        }
    }

    /// Compute the unit face normal for a triangle from its vertices using the right-hand rule.
    pub fn compute_face_normal(vertices: &[(f32, f32, f32); 3]) -> (f32, f32, f32) {
        let (ax, ay, az) = vertices[0];
        let (bx, by, bz) = vertices[1];
        let (cx, cy, cz) = vertices[2];

        let ab = (bx - ax, by - ay, bz - az);
        let ac = (cx - ax, cy - ay, cz - az);
        let n = (
            ab.1 * ac.2 - ab.2 * ac.1,
            ab.2 * ac.0 - ab.0 * ac.2,
            ab.0 * ac.1 - ab.1 * ac.0,
        );
        let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
        // Use 1e-24 threshold (equivalent to len > 1e-12 in original code)
        if len2 <= 1.0e-24 {
            return (0.0, 0.0, 0.0);
        }
        let inv_len = 1.0 / len2.sqrt();
        (n.0 * inv_len, n.1 * inv_len, n.2 * inv_len)
    }

    /// Legacy method for compatibility - returns the face normal.
    #[deprecated(note = "Use face_normal() or access normals directly for per-vertex normals")]
    pub fn compute_normal(vertices: &[(f32, f32, f32); 3]) -> (f32, f32, f32) {
        Self::compute_face_normal(vertices)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("No loaded asset with id: {0}")]
    NoSuchAssetId(String),
    #[error("Invalid input index: {0}")]
    InvalidInputIndex(usize),
    #[error("Invalid output index: {0}")]
    InvalidOutputIndex(usize),
    #[error("Wasmtime error: {0}")]
    Wasmtime(String),
    #[error("WASM backend error: {0}")]
    WasmBackend(String),
}

// =============================================================================
// Project Format V2: Flat Imports/Exports and Blob-Based Assets
// =============================================================================

/// Type hint for assets - purely informational, not enforced at execution time.
/// The UI uses these hints for validation and display purposes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum AssetTypeHint {
    /// Volumetric boolean field (WASM with is_inside/get_bounds)
    Model,
    /// Transform operator (WASM with run/get_metadata)
    Operator,
    /// CBOR-encoded configuration
    Config,
    /// Lua script (UTF-8 text)
    LuaSource,
    /// Unknown/generic binary data
    Binary,
}

impl std::fmt::Display for AssetTypeHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetTypeHint::Model => write!(f, "Model"),
            AssetTypeHint::Operator => write!(f, "Operator"),
            AssetTypeHint::Config => write!(f, "Config"),
            AssetTypeHint::LuaSource => write!(f, "LuaSource"),
            AssetTypeHint::Binary => write!(f, "Binary"),
        }
    }
}

/// An asset imported into the project (available at start of execution).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ImportedAsset {
    /// Unique identifier for this asset within the project.
    pub id: String,
    /// Raw binary data (the asset blob).
    pub data: Vec<u8>,
    /// Optional type hint for UI validation and display.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub type_hint: Option<AssetTypeHint>,
}

impl ImportedAsset {
    pub fn new(id: String, data: Vec<u8>, type_hint: Option<AssetTypeHint>) -> Self {
        Self { id, data, type_hint }
    }

    pub fn model(id: String, data: Vec<u8>) -> Self {
        Self::new(id, data, Some(AssetTypeHint::Model))
    }

    pub fn operator(id: String, data: Vec<u8>) -> Self {
        Self::new(id, data, Some(AssetTypeHint::Operator))
    }
}

/// Input to an execution step.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ExecutionInput {
    /// Reference to an asset by ID.
    AssetRef(String),
    /// Inline embedded data.
    Inline(Vec<u8>),
}

impl ExecutionInput {
    /// Returns a display-friendly description of this input.
    pub fn display(&self) -> String {
        match self {
            ExecutionInput::AssetRef(id) => format!("Asset: {}", id),
            ExecutionInput::Inline(data) => format!("Inline ({} bytes)", data.len()),
        }
    }
}

/// A single step in the execution timeline.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ExecutionStep {
    /// ID of the operator asset to execute.
    pub operator_id: String,
    /// Inputs to the operator.
    pub inputs: Vec<ExecutionInput>,
    /// Output asset IDs (no types - just IDs).
    pub outputs: Vec<String>,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum OperatorMetadataInput {
    ModelWASM,
    /// A CBOR-encoded configuration blob.
    ///
    /// The `String` is a CDDL snippet describing the expected CBOR structure.
    ///
    /// v0 convention (current host support): a single record/map like:
    /// `{ dx: float, dy: float, dz: float }`.
    ///
    /// The host UI uses this to generate widgets and encodes a CBOR map from field names to
    /// primitive values.
    CBORConfiguration(String),
    /// A Lua script source input.
    ///
    /// The `String` is a template/stub script showing the required function signatures.
    /// The host UI displays a multiline text editor pre-populated with this template.
    /// The script is passed as UTF-8 bytes to the operator.
    LuaSource(String),
    /// Raw binary data input (e.g., STL file data).
    ///
    /// The host UI should display a file picker allowing the user to select a file.
    /// The file contents are passed as raw bytes to the operator.
    Blob,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum OperatorMetadataOutput {
    ModelWASM,
}

// TODO: The operators should emit this as a cbor-encoded response to get_metadata()
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct OperatorMetadata {
    pub name: String,
    pub version: String,
    pub inputs: Vec<OperatorMetadataInput>,
    pub outputs: Vec<OperatorMetadataOutput>,
}

/// Load `OperatorMetadata` from an operator WASM module via its `get_metadata()` export.
///
/// ABI contract:
/// - The operator exports `get_metadata() -> i64` (or `u64`) where the return value packs
///   `(ptr: u32, len: u32)` as `ptr | (len << 32)`.
/// - The referenced bytes are CBOR encoded and match `OperatorMetadata`.
#[cfg(any(feature = "native", feature = "web"))]
pub fn operator_metadata_from_wasm_bytes(wasm_bin: &[u8]) -> Result<OperatorMetadata, ExecutionError> {
    use wasm::OperatorExecutor;

    let mut executor = wasm::create_operator_executor(wasm_bin)
        .map_err(|e| ExecutionError::WasmBackend(e.to_string()))?;

    let bytes = executor
        .get_metadata()
        .map_err(|e| ExecutionError::WasmBackend(e.to_string()))?;

    let mut cursor = std::io::Cursor::new(bytes);
    ciborium::de::from_reader(&mut cursor)
        .map_err(|e| ExecutionError::WasmBackend(format!("Failed to decode operator metadata CBOR: {e}")))
}

pub mod stl;
pub mod marching_cubes_cpu;
pub mod adaptive_surface_nets_2;

/// Sample points from the WASM volumetric model
#[cfg(feature = "native")]
pub fn sample_model(wasm_path: &Path, resolution: usize) -> anyhow::Result<(Vec<(f32, f32, f32)>, (f32, f32, f32), (f32, f32, f32))> {
    let wasm_bytes = std::fs::read(wasm_path)?;
    sample_model_from_bytes(&wasm_bytes, resolution)
}

/// Sample points from WASM bytes (in-memory model)
#[cfg(any(feature = "native", feature = "web"))]
pub fn sample_model_from_bytes(wasm_bytes: &[u8], resolution: usize) -> anyhow::Result<(Vec<(f32, f32, f32)>, (f32, f32, f32), (f32, f32, f32))> {
    use wasm::ModelExecutor;

    let mut executor = wasm::create_model_executor(wasm_bytes)
        .context("Failed to create model executor")?;

    let bounds = executor.get_bounds()?;
    let (bounds_min, bounds_max) = bounds.as_f32();

    let (min_x, min_y, min_z) = bounds_min;
    let (max_x, max_y, max_z) = bounds_max;

    let mut points = Vec::new();

    for z_idx in 0..resolution {
        let z = min_z + (max_z - min_z) * (z_idx as f32 / (resolution - 1).max(1) as f32);
        for y_idx in 0..resolution {
            let y = min_y + (max_y - min_y) * (y_idx as f32 / (resolution - 1).max(1) as f32);
            for x_idx in 0..resolution {
                let x = min_x + (max_x - min_x) * (x_idx as f32 / (resolution - 1).max(1) as f32);
                let density = executor.is_inside(x as f64, y as f64, z as f64)?;
                if density > 0.5 {
                    points.push((x, y, z));
                }
            }
        }
    }

    Ok((points, bounds_min, bounds_max))
}

/// Generate a mesh using marching cubes algorithm from the WASM volumetric model
#[cfg(feature = "native")]
pub fn generate_marching_cubes_mesh(wasm_path: &Path, resolution: usize) -> anyhow::Result<(Vec<Triangle>, (f32, f32, f32), (f32, f32, f32))> {
    let wasm_bytes = std::fs::read(wasm_path)?;
    generate_marching_cubes_mesh_from_bytes(&wasm_bytes, resolution)
}

/// Generate a mesh using marching cubes from WASM bytes
#[cfg(any(feature = "native", feature = "web"))]
pub fn generate_marching_cubes_mesh_from_bytes(wasm_bytes: &[u8], resolution: usize) -> anyhow::Result<(Vec<Triangle>, (f32, f32, f32), (f32, f32, f32))> {
    use wasm::ModelExecutor;

    let mut executor = wasm::create_model_executor(wasm_bytes)
        .context("Failed to create model executor")?;

    let bounds = executor.get_bounds()?;
    let (bounds_min, bounds_max) = bounds.as_f32();

    // Wrap executor in RefCell for interior mutability in the closure
    let executor = std::cell::RefCell::new(executor);

    let triangles = marching_cubes_cpu::marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
        let d = executor.borrow_mut().is_inside(p.0 as f64, p.1 as f64, p.2 as f64)?;
        Ok(d)
    })?;

    Ok((triangles, bounds_min, bounds_max))
}

/// Result from adaptive mesh v2 generation including mesh data, bounds, and profiling stats.
pub struct AdaptiveMeshV2Result {
    pub vertices: Vec<(f32, f32, f32)>,
    pub normals: Vec<(f32, f32, f32)>,
    pub indices: Vec<u32>,
    pub bounds_min: (f32, f32, f32),
    pub bounds_max: (f32, f32, f32),
    pub stats: adaptive_surface_nets_2::MeshingStats2,
}

/// Generate an indexed mesh using the new Adaptive Surface Nets v2 algorithm.
/// Returns a result struct containing mesh data, bounds, and detailed profiling statistics.
#[cfg(feature = "native")]
pub fn generate_adaptive_mesh_v2_from_bytes(
    wasm_bytes: &[u8],
    config: &adaptive_surface_nets_2::AdaptiveMeshConfig2,
) -> anyhow::Result<AdaptiveMeshV2Result> {
    use wasm::ParallelModelSampler;

    let wasm_sampler = wasm::create_parallel_sampler(wasm_bytes)
        .context("Failed to create parallel sampler")?;

    let bounds = wasm_sampler.get_bounds()?;
    let (bounds_min, bounds_max) = bounds.as_f32();

    // Create a closure-based sampler that wraps the ParallelModelSampler
    let sampler = move |x: f64, y: f64, z: f64| -> f32 {
        wasm_sampler.sample(x, y, z)
    };

    let result = adaptive_surface_nets_2::adaptive_surface_nets_2(sampler, bounds_min, bounds_max, config);

    Ok(AdaptiveMeshV2Result {
        vertices: result.mesh.vertices,
        normals: result.mesh.normals,
        indices: result.mesh.indices,
        bounds_min,
        bounds_max,
        stats: result.stats,
    })
}

/// Generate an indexed mesh using Adaptive Surface Nets v2 (web version).
/// Uses the WebParallelSampler for single-threaded web execution.
#[cfg(all(feature = "web", not(feature = "native")))]
pub fn generate_adaptive_mesh_v2_from_bytes(
    wasm_bytes: &[u8],
    config: &adaptive_surface_nets_2::AdaptiveMeshConfig2,
) -> anyhow::Result<AdaptiveMeshV2Result> {
    use wasm::ParallelModelSampler;

    let wasm_sampler = wasm::create_parallel_sampler(wasm_bytes)
        .context("Failed to create parallel sampler")?;

    let bounds = wasm_sampler.get_bounds()?;
    let (bounds_min, bounds_max) = bounds.as_f32();

    // Create a closure-based sampler that wraps the WebParallelSampler
    let sampler = move |x: f64, y: f64, z: f64| -> f32 {
        wasm_sampler.sample(x, y, z)
    };

    let result = adaptive_surface_nets_2::adaptive_surface_nets_2(sampler, bounds_min, bounds_max, config);

    Ok(AdaptiveMeshV2Result {
        vertices: result.mesh.vertices,
        normals: result.mesh.normals,
        indices: result.mesh.indices,
        bounds_min,
        bounds_max,
        stats: result.stats,
    })
}


// =============================================================================
// Project V2: New Flat Structure
// =============================================================================

/// Project file format version 2 with flat imports/exports and blob-based assets.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Project {
    /// Format version (2 for this format).
    pub version: u32,

    /// Assets embedded in the project (available at start of execution).
    pub imports: Vec<ImportedAsset>,

    /// Execution timeline (operator invocations only).
    pub timeline: Vec<ExecutionStep>,

    /// Asset IDs to return after execution.
    pub exports: Vec<String>,
}

/// A loaded asset in the execution environment.
#[derive(Clone, Debug)]
pub struct LoadedAsset {
    /// The asset ID.
    id: String,
    /// Raw binary data.
    data: Arc<Vec<u8>>,
    /// Optional type hint.
    type_hint: Option<AssetTypeHint>,
    /// IDs of assets that were used to create this asset.
    precursor_ids: Vec<String>,
}

impl LoadedAsset {
    /// Returns the asset ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the raw bytes of this asset.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns the data wrapped in Arc.
    pub fn data_arc(&self) -> Arc<Vec<u8>> {
        Arc::clone(&self.data)
    }

    /// Returns the type hint if available.
    pub fn type_hint(&self) -> Option<AssetTypeHint> {
        self.type_hint
    }

    /// Returns the IDs of assets that were used to create this asset.
    pub fn precursor_ids(&self) -> &[String] {
        &self.precursor_ids
    }

    /// Returns the raw bytes if this looks like a model (by type hint).
    pub fn as_model(&self) -> Option<&[u8]> {
        match self.type_hint {
            Some(AssetTypeHint::Model) | None => Some(&self.data),
            _ => None,
        }
    }
}

/// Execution environment that holds loaded assets during project execution.
pub struct Environment {
    assets: HashMap<String, LoadedAsset>,
}

impl Environment {
    /// Creates a new empty environment.
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
        }
    }

    /// Returns a reference to a loaded asset by ID.
    pub fn get(&self, id: &str) -> Option<&LoadedAsset> {
        self.assets.get(id)
    }

    /// Returns all loaded asset IDs.
    pub fn asset_ids(&self) -> impl Iterator<Item = &str> {
        self.assets.keys().map(|s| s.as_str())
    }

    /// Inserts an asset into the environment.
    fn insert(&mut self, asset: LoadedAsset) {
        self.assets.insert(asset.id.clone(), asset);
    }

    /// Removes and returns an asset by ID.
    fn remove(&mut self, id: &str) -> Option<LoadedAsset> {
        self.assets.remove(id)
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Project {
    /// Creates a new empty project.
    pub fn new() -> Self {
        Self {
            version: 2,
            imports: Vec::new(),
            timeline: Vec::new(),
            exports: Vec::new(),
        }
    }

    /// Creates a simple project that imports a model and exports it.
    pub fn from_model(id: String, data: Vec<u8>) -> Self {
        Self {
            version: 2,
            imports: vec![ImportedAsset::model(id.clone(), data)],
            timeline: Vec::new(),
            exports: vec![id],
        }
    }

    /// Returns all declared asset IDs with their type hints.
    ///
    /// This includes both imported assets and outputs from execution steps.
    pub fn declared_assets(&self) -> Vec<(String, Option<AssetTypeHint>)> {
        let mut out: Vec<(String, Option<AssetTypeHint>)> = Vec::new();
        let mut seen: HashMap<String, usize> = HashMap::new();

        // Add imported assets
        for import in &self.imports {
            if let Some(idx) = seen.get(&import.id).copied() {
                out[idx].1 = import.type_hint;
            } else {
                seen.insert(import.id.clone(), out.len());
                out.push((import.id.clone(), import.type_hint));
            }
        }

        // Add execution step outputs (type hint unknown for outputs)
        for step in &self.timeline {
            for output_id in &step.outputs {
                if !seen.contains_key(output_id) {
                    seen.insert(output_id.clone(), out.len());
                    // Outputs from operators are typically models
                    out.push((output_id.clone(), Some(AssetTypeHint::Model)));
                }
            }
        }

        out
    }

    /// Generates a unique asset ID based on the given base name.
    pub fn unique_asset_id(&self, base: &str) -> String {
        let declared: std::collections::HashSet<String> = self
            .declared_assets()
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        if !declared.contains(base) {
            return base.to_string();
        }

        for i in 2..=10_000 {
            let candidate = format!("{base}_{i}");
            if !declared.contains(&candidate) {
                return candidate;
            }
        }

        format!("{base}_{}", declared.len() + 1)
    }

    /// Generates a reasonable default output name for an operator.
    pub fn default_output_name(&self, operator_crate_name: &str, primary_input: Option<&str>) -> String {
        let suffix = match operator_crate_name {
            "translate_operator" => "translated",
            "rotation_operator" => "rotated",
            "scale_operator" => "scaled",
            "boolean_operator" => "boolean_result",
            "lua_script_operator" => "scripted",
            _ => {
                let base = operator_crate_name.strip_suffix("_operator").unwrap_or(operator_crate_name);
                return self.unique_asset_id(&format!("{}_output", base));
            }
        };

        let base = match primary_input {
            Some(input) => format!("{}_{}", input, suffix),
            None => suffix.to_string(),
        };

        self.unique_asset_id(&base)
    }

    /// Inserts a new model into the project.
    /// Returns the final asset ID (may differ from base if collision occurred).
    pub fn insert_model(&mut self, id_base: &str, data: Vec<u8>) -> String {
        let id = self.unique_asset_id(id_base);
        self.imports.push(ImportedAsset::model(id.clone(), data));
        self.exports.push(id.clone());
        id
    }

    /// Inserts an operator and execution step into the project.
    pub fn insert_operation(
        &mut self,
        op_id_base: &str,
        op_data: Vec<u8>,
        inputs: Vec<ExecutionInput>,
        output_ids: Vec<String>,
        export_id: String,
    ) {
        let op_id = self.unique_asset_id(op_id_base);

        // Add the operator as an import
        self.imports.push(ImportedAsset::operator(op_id.clone(), op_data));

        // Add the execution step
        self.timeline.push(ExecutionStep {
            operator_id: op_id,
            inputs,
            outputs: output_ids,
        });

        // Add to exports
        if !self.exports.contains(&export_id) {
            self.exports.push(export_id);
        }
    }

    /// Returns the imports.
    pub fn imports(&self) -> &[ImportedAsset] {
        &self.imports
    }

    /// Returns mutable access to imports.
    pub fn imports_mut(&mut self) -> &mut Vec<ImportedAsset> {
        &mut self.imports
    }

    /// Returns the timeline.
    pub fn timeline(&self) -> &[ExecutionStep] {
        &self.timeline
    }

    /// Returns mutable access to timeline.
    pub fn timeline_mut(&mut self) -> &mut Vec<ExecutionStep> {
        &mut self.timeline
    }

    /// Returns the exports.
    pub fn exports(&self) -> &[String] {
        &self.exports
    }

    /// Returns mutable access to exports.
    pub fn exports_mut(&mut self) -> &mut Vec<String> {
        &mut self.exports
    }

    /// Runs the project and returns exported assets.
    ///
    /// The executor is type-agnostic: no type enforcement at execution time.
    #[cfg(any(feature = "native", feature = "web"))]
    pub fn run(&self, env: &mut Environment) -> Result<Vec<LoadedAsset>, ExecutionError> {
        use wasm::{OperatorExecutor, OperatorIo};

        // Load all imports into environment
        for import in &self.imports {
            env.insert(LoadedAsset {
                id: import.id.clone(),
                data: Arc::new(import.data.clone()),
                type_hint: import.type_hint,
                precursor_ids: vec![],
            });
        }

        // Execute timeline steps
        for step in &self.timeline {
            // Get operator bytes
            let op_data = env.get(&step.operator_id)
                .ok_or_else(|| ExecutionError::NoSuchAssetId(step.operator_id.clone()))?
                .data_arc();

            // Resolve inputs to bytes
            let mut input_bytes = Vec::new();
            let mut precursor_ids = Vec::new();

            for input in &step.inputs {
                let bytes = match input {
                    ExecutionInput::AssetRef(id) => {
                        let asset = env.get(id)
                            .ok_or_else(|| ExecutionError::NoSuchAssetId(id.clone()))?;
                        precursor_ids.push(id.clone());
                        asset.data().to_vec()
                    }
                    ExecutionInput::Inline(data) => data.clone(),
                };
                input_bytes.push(bytes);
            }

            // Execute the operator
            let mut executor = wasm::create_operator_executor(&op_data)
                .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

            let io = OperatorIo::new(input_bytes);
            let result = executor.run(io)
                .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

            // Store outputs
            for (idx, output_id) in step.outputs.iter().enumerate() {
                if let Some(output_bytes) = result.outputs.get(&idx) {
                    env.insert(LoadedAsset {
                        id: output_id.clone(),
                        data: Arc::new(output_bytes.clone()),
                        type_hint: Some(AssetTypeHint::Model), // Outputs are typically models
                        precursor_ids: precursor_ids.clone(),
                    });
                }
            }
        }

        // Collect exports
        let mut exported = Vec::new();
        for export_id in &self.exports {
            let asset = env.remove(export_id)
                .ok_or_else(|| ExecutionError::NoSuchAssetId(export_id.clone()))?;
            exported.push(asset);
        }

        Ok(exported)
    }

    #[cfg(not(any(feature = "native", feature = "web")))]
    pub fn run(&self, _env: &mut Environment) -> Result<Vec<LoadedAsset>, ExecutionError> {
        Err(ExecutionError::Wasmtime("WASM execution requires the 'native' or 'web' feature".to_string()))
    }

    /// Serializes the project to CBOR format.
    pub fn to_cbor(&self) -> Result<Vec<u8>, std::io::Error> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(bytes)
    }

    /// Deserializes a project from CBOR format.
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, std::io::Error> {
        ciborium::from_reader(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    /// Saves the project to a file in CBOR format.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let bytes = self.to_cbor()?;
        std::fs::write(path, bytes)
    }

    /// Loads a project from a CBOR file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let bytes = std::fs::read(path)?;
        Self::from_cbor(&bytes)
    }
}

impl Default for Project {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exporting_same_asset_twice_is_an_error() {
        let project = Project {
            version: 2,
            imports: vec![ImportedAsset::model("a".to_string(), vec![1, 2, 3])],
            timeline: vec![],
            exports: vec!["a".to_string(), "a".to_string()],
        };

        let mut env = Environment::new();
        let err = project.run(&mut env).unwrap_err();
        match err {
            ExecutionError::NoSuchAssetId(id) => assert_eq!(id, "a"),
            other => panic!("expected NoSuchAssetId, got: {other:?}"),
        }
    }

    #[test]
    fn project_declared_assets_includes_outputs() {
        let project = Project {
            version: 2,
            imports: vec![ImportedAsset::model("model_a".to_string(), vec![1, 2, 3])],
            timeline: vec![ExecutionStep {
                operator_id: "op".to_string(),
                inputs: vec![ExecutionInput::AssetRef("model_a".to_string())],
                outputs: vec!["model_b".to_string()],
            }],
            exports: vec![],
        };

        let assets = project.declared_assets();
        assert!(assets.iter().any(|(id, hint)| id == "model_a" && *hint == Some(AssetTypeHint::Model)));
        assert!(assets.iter().any(|(id, _)| id == "model_b"));
    }

    #[test]
    fn insert_model_makes_ids_unique() {
        let mut p = Project::new();
        let id1 = p.insert_model("model", vec![1, 2, 3]);
        let id2 = p.insert_model("model", vec![4, 5, 6]);

        assert_eq!(id1, "model");
        assert_eq!(id2, "model_2");
        assert_eq!(p.imports.len(), 2);
        assert_eq!(p.exports.len(), 2);
    }
}
