use std::collections::HashMap;
use std::fmt::Display;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use wasmtime::{Caller, Engine, Instance, Module, Store};

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
    #[error("Asset with id {0} is not of type {1}, but {2}")]
    WrongAssetType(String, AssetType, AssetType),
    #[error("Invalid input index: {0}")]
    InvalidInputIndex(usize),
    #[error("Invalid output index: {0}")]
    InvalidOutputIndex(usize),
    #[error("Wasmtime error: {0}")]
    Wasmtime(String),
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
pub fn operator_metadata_from_wasm_bytes(wasm_bin: &[u8]) -> Result<OperatorMetadata, ExecutionError> {
    let engine = wasmtime::Engine::new(wasmtime::Config::new().debug_info(true))
        .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;
    let module = wasmtime::Module::new(&engine, wasm_bin)
        .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

    let mut linker = wasmtime::Linker::new(&engine);

    // The operator module may import the host IO functions even if `get_metadata()` doesn't use them.
    // Provide stubs so instantiation succeeds.
    linker.func_wrap("host", "get_input_len", |_caller: Caller<'_, ()>, _arg: i32| -> u32 { 0 })
        .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;
    linker.func_wrap(
        "host",
        "get_input_data",
        |_caller: Caller<'_, ()>, _arg: i32, _ptr: i32, _len: i32| {},
    )
    .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;
    linker.func_wrap(
        "host",
        "post_output",
        |_caller: Caller<'_, ()>, _output_idx: i32, _ptr: i32, _len: i32| {},
    )
    .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

    let mut store = Store::new(&engine, ());
    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

    let metadata_func = instance
        .get_typed_func::<(), i64>(&mut store, "get_metadata")
        .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

    let packed = metadata_func
        .call(&mut store, ())
        .map_err(|e| ExecutionError::Wasmtime(e.to_string()))? as u64;
    let ptr = (packed & 0xFFFF_FFFF) as usize;
    let len = (packed >> 32) as usize;

    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| ExecutionError::Wasmtime("Operator module does not export `memory`".to_string()))?;
    let mem_data = memory.data(&store);
    let end = ptr
        .checked_add(len)
        .ok_or_else(|| ExecutionError::Wasmtime("Operator metadata pointer overflow".to_string()))?;
    if end > mem_data.len() {
        return Err(ExecutionError::Wasmtime("Operator metadata points outside of linear memory".to_string()));
    }
    let bytes = &mem_data[ptr..end];

    let mut cursor = std::io::Cursor::new(bytes);
    ciborium::de::from_reader(&mut cursor)
        .map_err(|e| ExecutionError::Wasmtime(format!("Failed to decode operator metadata CBOR: {e}")))
}

pub mod stl;
pub mod marching_cubes_cpu;
pub mod adaptive_surface_nets;
pub mod adaptive_surface_nets_2;

/// Sample points from the WASM volumetric model
pub fn sample_model(wasm_path: &Path, resolution: usize) -> anyhow::Result<(Vec<(f32, f32, f32)>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm_path)?;

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let is_inside = instance.get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")?;

    let get_bounds_min_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")?;

    let min_x = get_bounds_min_x.call(&mut store, ())? as f32;
    let min_y = get_bounds_min_y.call(&mut store, ())? as f32;
    let min_z = get_bounds_min_z.call(&mut store, ())? as f32;
    let max_x = get_bounds_max_x.call(&mut store, ())? as f32;
    let max_y = get_bounds_max_y.call(&mut store, ())? as f32;
    let max_z = get_bounds_max_z.call(&mut store, ())? as f32;

    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);

    let mut points = Vec::new();

    for z_idx in 0..resolution {
        let z = min_z + (max_z - min_z) * (z_idx as f32 / (resolution - 1).max(1) as f32);
        for y_idx in 0..resolution {
            let y = min_y + (max_y - min_y) * (y_idx as f32 / (resolution - 1).max(1) as f32);
            for x_idx in 0..resolution {
                let x = min_x + (max_x - min_x) * (x_idx as f32 / (resolution - 1).max(1) as f32);
                let density = is_inside.call(&mut store, (x as f64, y as f64, z as f64))?;
                if density > 0.5 {
                    points.push((x, y, z));
                }
            }
        }
    }

    Ok((points, bounds_min, bounds_max))
}

/// Sample points from WASM bytes (in-memory model)
pub fn sample_model_from_bytes(wasm_bytes: &[u8], resolution: usize) -> anyhow::Result<(Vec<(f32, f32, f32)>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to load WASM module from bytes")?;

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let is_inside = instance.get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")?;

    let get_bounds_min_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")?;

    let min_x = get_bounds_min_x.call(&mut store, ())? as f32;
    let min_y = get_bounds_min_y.call(&mut store, ())? as f32;
    let min_z = get_bounds_min_z.call(&mut store, ())? as f32;
    let max_x = get_bounds_max_x.call(&mut store, ())? as f32;
    let max_y = get_bounds_max_y.call(&mut store, ())? as f32;
    let max_z = get_bounds_max_z.call(&mut store, ())? as f32;

    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);

    let mut points = Vec::new();

    for z_idx in 0..resolution {
        let z = min_z + (max_z - min_z) * (z_idx as f32 / (resolution - 1).max(1) as f32);
        for y_idx in 0..resolution {
            let y = min_y + (max_y - min_y) * (y_idx as f32 / (resolution - 1).max(1) as f32);
            for x_idx in 0..resolution {
                let x = min_x + (max_x - min_x) * (x_idx as f32 / (resolution - 1).max(1) as f32);
                let density = is_inside.call(&mut store, (x as f64, y as f64, z as f64))?;
                if density > 0.5 {
                    points.push((x, y, z));
                }
            }
        }
    }

    Ok((points, bounds_min, bounds_max))
}

/// Generate a mesh using marching cubes algorithm from the WASM volumetric model
pub fn generate_marching_cubes_mesh(wasm_path: &Path, resolution: usize) -> anyhow::Result<(Vec<Triangle>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm_path)?;

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let is_inside = instance.get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")?;

    let get_bounds_min_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")?;

    let min_x = get_bounds_min_x.call(&mut store, ())? as f32;
    let min_y = get_bounds_min_y.call(&mut store, ())? as f32;
    let min_z = get_bounds_min_z.call(&mut store, ())? as f32;
    let max_x = get_bounds_max_x.call(&mut store, ())? as f32;
    let max_y = get_bounds_max_y.call(&mut store, ())? as f32;
    let max_z = get_bounds_max_z.call(&mut store, ())? as f32;

    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);

    let triangles = marching_cubes_cpu::marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
        let d = is_inside.call(&mut store, (p.0 as f64, p.1 as f64, p.2 as f64))?;
        Ok(d)
    })?;

    Ok((triangles, bounds_min, bounds_max))
}

/// Generate a mesh using marching cubes from WASM bytes
pub fn generate_marching_cubes_mesh_from_bytes(wasm_bytes: &[u8], resolution: usize) -> anyhow::Result<(Vec<Triangle>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to load WASM module from bytes")?;

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let is_inside = instance.get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")?;

    let get_bounds_min_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")?;

    let min_x = get_bounds_min_x.call(&mut store, ())? as f32;
    let min_y = get_bounds_min_y.call(&mut store, ())? as f32;
    let min_z = get_bounds_min_z.call(&mut store, ())? as f32;
    let max_x = get_bounds_max_x.call(&mut store, ())? as f32;
    let max_y = get_bounds_max_y.call(&mut store, ())? as f32;
    let max_z = get_bounds_max_z.call(&mut store, ())? as f32;

    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);

    let triangles = marching_cubes_cpu::marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
        let d = is_inside.call(&mut store, (p.0 as f64, p.1 as f64, p.2 as f64))?;
        Ok(d)
    })?;

    Ok((triangles, bounds_min, bounds_max))
}

/// Generate an adaptive mesh from WASM bytes
pub fn generate_adaptive_mesh_from_bytes(wasm_bytes: &[u8], config: &adaptive_surface_nets::AdaptiveMeshConfig) -> anyhow::Result<(Vec<Triangle>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes)?;

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let get_bounds_min_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")?;

    let min_x = get_bounds_min_x.call(&mut store, ())? as f32;
    let min_y = get_bounds_min_y.call(&mut store, ())? as f32;
    let min_z = get_bounds_min_z.call(&mut store, ())? as f32;
    let max_x = get_bounds_max_x.call(&mut store, ())? as f32;
    let max_y = get_bounds_max_y.call(&mut store, ())? as f32;
    let max_z = get_bounds_max_z.call(&mut store, ())? as f32;

    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);

    let triangles = adaptive_surface_nets::adaptive_surface_nets_mesh(wasm_bytes, bounds_min, bounds_max, config)?;

    Ok((triangles, bounds_min, bounds_max))
}

/// WASM sampler for ASN2 that creates a fresh instance for each sampling call.
/// This is needed because wasmtime's Store requires mutable access but the
/// ASN2 API expects a Fn (not FnMut) closure.
struct Asn2WasmSampler {
    engine: std::sync::Arc<Engine>,
    module: std::sync::Arc<Module>,
}

impl Asn2WasmSampler {
    fn new(wasm_bytes: &[u8]) -> anyhow::Result<Self> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes)?;
        Ok(Self {
            engine: std::sync::Arc::new(engine),
            module: std::sync::Arc::new(module),
        })
    }

    fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        // Create a fresh store and instance for each sample
        // This is not efficient but allows us to use Fn interface
        // For production, we should use thread_local! storage
        thread_local! {
            static CONTEXT: std::cell::RefCell<Option<(Store<()>, wasmtime::TypedFunc<(f64, f64, f64), f32>)>> = const { std::cell::RefCell::new(None) };
        }

        // This is a simplified approach - for real usage we'd want proper thread-local caching
        let mut store = Store::new(&self.engine, ());
        let instance = match Instance::new(&mut store, &self.module, &[]) {
            Ok(i) => i,
            Err(_) => return 0.0,
        };
        let is_inside = match instance.get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside") {
            Ok(f) => f,
            Err(_) => return 0.0,
        };

        is_inside.call(&mut store, (x, y, z)).unwrap_or(0.0)
    }
}

/// Generate an indexed mesh using the new Adaptive Surface Nets v2 algorithm.
/// Returns (vertices, normals, indices, bounds_min, bounds_max).
pub fn generate_adaptive_mesh_v2_from_bytes(
    wasm_bytes: &[u8],
    config: &adaptive_surface_nets_2::AdaptiveMeshConfig2,
) -> anyhow::Result<(Vec<(f32, f32, f32)>, Vec<(f32, f32, f32)>, Vec<u32>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes)?;

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let get_bounds_min_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")?;

    let min_x = get_bounds_min_x.call(&mut store, ())? as f32;
    let min_y = get_bounds_min_y.call(&mut store, ())? as f32;
    let min_z = get_bounds_min_z.call(&mut store, ())? as f32;
    let max_x = get_bounds_max_x.call(&mut store, ())? as f32;
    let max_y = get_bounds_max_y.call(&mut store, ())? as f32;
    let max_z = get_bounds_max_z.call(&mut store, ())? as f32;

    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);

    // Create sampler using thread-local caching approach
    let wasm_sampler = Asn2WasmSampler::new(wasm_bytes)?;
    let sampler = move |x: f64, y: f64, z: f64| -> f32 {
        wasm_sampler.sample(x, y, z)
    };

    let mesh = adaptive_surface_nets_2::adaptive_surface_nets_2(sampler, bounds_min, bounds_max, config);

    Ok((mesh.vertices, mesh.normals, mesh.indices, bounds_min, bounds_max))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum AssetType {
    ModelWASM,
    OperationWASM,
}

impl Display for AssetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetType::ModelWASM => write!(f, "ModelWASM"),
            AssetType::OperationWASM => write!(f, "OperationWASM"),
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum Asset {
    ModelWASM(Vec<u8>),
    OperationWASM(Vec<u8>),
}

impl Asset {
    /// Returns the type of this asset
    pub fn asset_type(&self) -> AssetType {
        match self {
            Asset::ModelWASM(_) => AssetType::ModelWASM,
            Asset::OperationWASM(_) => AssetType::OperationWASM,
        }
    }

    /// Returns the raw WASM bytes
    pub fn bytes(&self) -> &[u8] {
        match self {
            Asset::ModelWASM(bytes) => bytes,
            Asset::OperationWASM(bytes) => bytes,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct LoadAssetEntry {
    asset_id: String,
    asset: Asset,
}

impl LoadAssetEntry {
    pub fn new(asset_id: String, asset: Asset) -> Self {
        Self { asset_id, asset }
    }

    /// Returns the asset ID
    pub fn asset_id(&self) -> &str {
        &self.asset_id
    }

    /// Returns the asset type
    pub fn asset_type(&self) -> AssetType {
        self.asset.asset_type()
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ExecuteWasmInput {
    AssetByID(String),
    String(String),
    Data(Vec<u8>)
}

impl ExecuteWasmInput {
    /// Returns a display-friendly description of this input
    pub fn display(&self) -> String {
        match self {
            ExecuteWasmInput::AssetByID(id) => format!("Asset: {}", id),
            ExecuteWasmInput::String(s) => {
                if s.len() > 20 {
                    format!("\"{}...\"", &s[..17])
                } else {
                    format!("\"{}\"", s)
                }
            }
            ExecuteWasmInput::Data(_) => "Data".to_string(),
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ExecuteWasmOutput {
    pub asset_id: String,
    pub asset_type: AssetType,
}

impl ExecuteWasmOutput {
    pub fn new(asset_id: String, asset_type: AssetType) -> Self {
        Self { asset_id, asset_type }
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ExecuteWasmEntry {
    asset_id: String,
    inputs: Vec<ExecuteWasmInput>,
    outputs: Vec<ExecuteWasmOutput>
}

impl ExecuteWasmEntry {
    pub fn new(asset_id: String, inputs: Vec<ExecuteWasmInput>, outputs: Vec<ExecuteWasmOutput>) -> Self {
        Self { asset_id, inputs, outputs }
    }

    /// Returns the asset ID of the WASM operation to execute
    pub fn asset_id(&self) -> &str {
        &self.asset_id
    }

    /// Returns the inputs for this operation
    pub fn inputs(&self) -> &[ExecuteWasmInput] {
        &self.inputs
    }

    /// Returns the number of outputs this operation produces
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
    
    /// Returns the outputs for this operation
    pub fn outputs(&self) -> &[ExecuteWasmOutput] {
        &self.outputs
    }
}

/// Runtime state for WASM execution, holding inputs and collecting outputs
struct WasmExecutionState {
    /// The input specifications
    inputs: Vec<ExecuteWasmInput>,
    /// Pre-fetched assets needed for inputs
    input_assets: HashMap<String, LoadedAsset>,
    /// Output specifications (asset_id, asset_type)
    output_specs: Vec<ExecuteWasmOutput>,
    /// Collected output data (index -> bytes)
    output_data: HashMap<usize, Vec<u8>>,
    /// Precursor asset IDs for tracking lineage
    precursor_asset_ids: Vec<String>,
}

impl ExecuteWasmEntry {
    pub fn run(&self, environment: &mut Environment) -> Result<(), ExecutionError> {
        let wasm_asset = environment.loaded_assets.get(&self.asset_id)
            .ok_or(ExecutionError::NoSuchAssetId(self.asset_id.clone()))?;

        // Pre-fetch needed assets
        let mut input_assets = HashMap::new();
        let mut precursor_asset_ids = vec![];
        for input in &self.inputs {
            if let ExecuteWasmInput::AssetByID(asset_id) = input {
                input_assets.insert(
                    asset_id.clone(),
                    environment.loaded_assets.get(asset_id)
                        .ok_or(ExecutionError::NoSuchAssetId(asset_id.clone()))?.clone()
                );
                precursor_asset_ids.push(asset_id.clone());
            }
        }

        let wasm_bin = match wasm_asset.asset.as_ref() {
            Asset::OperationWASM(wasm_bin) => wasm_bin,
            other => return Err(ExecutionError::WrongAssetType(
                self.asset_id.clone(),
                AssetType::OperationWASM,
                other.asset_type(),
            )),
        };

        let engine = wasmtime::Engine::new(wasmtime::Config::new().debug_info(true))
            .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;
        let module = wasmtime::Module::new(&engine, wasm_bin)
            .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

        let mut linker = wasmtime::Linker::new(&engine);
        
        let state = WasmExecutionState {
            inputs: self.inputs.clone(),
            input_assets,
            output_specs: self.outputs.clone(),
            output_data: HashMap::new(),
            precursor_asset_ids: precursor_asset_ids.clone(),
        };
        let mut store = Store::new(&engine, state);

        // Host function: get the length of an input
        linker.func_wrap("host", "get_input_len", |caller: Caller<'_, WasmExecutionState>, arg: i32| -> u32 {
            let idx = arg as usize;
            match caller.data().inputs.get(idx) {
                Some(ExecuteWasmInput::AssetByID(asset_id)) => {
                    if let Some(asset) = caller.data().input_assets.get(asset_id) {
                        match asset.asset.as_ref() {
                            Asset::ModelWASM(wasm) => wasm.len() as u32,
                            Asset::OperationWASM(wasm) => wasm.len() as u32,
                        }
                    } else {
                        0
                    }
                },
                Some(ExecuteWasmInput::String(s)) => s.len() as u32,
                Some(ExecuteWasmInput::Data(d)) => d.len() as u32,
                None => 0,
            }
        }).map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

        // Host function: copy input data into WASM memory
        linker.func_wrap("host", "get_input_data", |mut caller: Caller<'_, WasmExecutionState>, arg: i32, ptr: i32, len: i32| {
            let idx = arg as usize;
            // First, copy the data we need to avoid borrow conflicts
            let data: Option<Vec<u8>> = match caller.data().inputs.get(idx) {
                Some(ExecuteWasmInput::AssetByID(asset_id)) => {
                    caller.data().input_assets.get(asset_id).map(|asset| {
                        match asset.asset.as_ref() {
                            Asset::ModelWASM(wasm) => wasm.clone(),
                            Asset::OperationWASM(wasm) => wasm.clone(),
                        }
                    })
                },
                Some(ExecuteWasmInput::String(s)) => Some(s.as_bytes().to_vec()),
                Some(ExecuteWasmInput::Data(d)) => Some(d.clone()),
                None => None,
            };

            if let Some(src_data) = data {
                let copy_len = (len as usize).min(src_data.len());
                if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                    let mem_data = memory.data_mut(&mut caller);
                    let dest_start = ptr as usize;
                    let dest_end = dest_start + copy_len;
                    if dest_end <= mem_data.len() {
                        mem_data[dest_start..dest_end].copy_from_slice(&src_data[..copy_len]);
                    }
                }
            }
        }).map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

        // Host function: post output data from WASM memory
        linker.func_wrap("host", "post_output", |mut caller: Caller<'_, WasmExecutionState>, output_idx: i32, ptr: i32, len: i32| {
            let idx = output_idx as usize;
            if idx >= caller.data().output_specs.len() {
                return;
            }

            if let Some(memory) = caller.get_export("memory").and_then(|e| e.into_memory()) {
                let mem_data = memory.data(&caller);
                let src_start = ptr as usize;
                let src_end = src_start + len as usize;
                if src_end <= mem_data.len() {
                    let output_bytes = mem_data[src_start..src_end].to_vec();
                    caller.data_mut().output_data.insert(idx, output_bytes);
                }
            }
        }).map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

        let instance = linker.instantiate(&mut store, &module)
            .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

        let run_func = instance.get_typed_func::<(), ()>(&mut store, "run")
            .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;
        run_func.call(&mut store, ())
            .map_err(|e| ExecutionError::Wasmtime(e.to_string()))?;

        // Collect outputs and add them to the environment
        let state = store.into_data();
        for (idx, output_spec) in self.outputs.iter().enumerate() {
            if let Some(output_bytes) = state.output_data.get(&idx) {
                let asset = match output_spec.asset_type {
                    AssetType::ModelWASM => Asset::ModelWASM(output_bytes.clone()),
                    AssetType::OperationWASM => Asset::OperationWASM(output_bytes.clone()),
                };
                environment.loaded_assets.insert(output_spec.asset_id.clone(), LoadedAsset {
                    asset_id: output_spec.asset_id.clone(),
                    asset: Arc::new(asset),
                    precursor_asset_ids: precursor_asset_ids.clone(),
                });
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ProjectEntry {
    LoadAsset(LoadAssetEntry),
    ExecuteWASM(ExecuteWasmEntry),
    ExportAsset(String),
}

impl ProjectEntry {
    pub fn run(&self, environment: &mut Environment) -> Result<Vec<LoadedAsset>, ExecutionError> {
        match self {
            ProjectEntry::LoadAsset(entry) => {
                environment.loaded_assets.insert(entry.asset_id.clone(), LoadedAsset {
                    asset_id: entry.asset_id.clone(),
                    asset: Arc::new(entry.asset.clone()),
                    precursor_asset_ids: vec![],
                });
                Ok(vec![])
            }
            ProjectEntry::ExecuteWASM(entry) => {
                entry.run(environment)?;
                Ok(vec![])
            }
            ProjectEntry::ExportAsset(asset_id) => {
                Ok(vec![environment.loaded_assets.remove(asset_id).ok_or(ExecutionError::NoSuchAssetId(asset_id.clone()))?])
            }
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Project {
    entries: Vec<ProjectEntry>,
}

#[derive(Clone, Debug)]
pub struct LoadedAsset {
    asset_id: String,
    asset: Arc<Asset>,
    precursor_asset_ids: Vec<String>,
}

impl LoadedAsset {
    /// Returns the asset ID
    pub fn asset_id(&self) -> &str {
        &self.asset_id
    }

    /// Returns a reference to the asset
    pub fn asset(&self) -> &Asset {
        &self.asset
    }

    /// Returns the asset wrapped in Arc
    pub fn asset_arc(&self) -> Arc<Asset> {
        Arc::clone(&self.asset)
    }

    /// Returns the IDs of assets that were used to create this asset
    pub fn precursor_asset_ids(&self) -> &[String] {
        &self.precursor_asset_ids
    }

    /// Returns the raw WASM bytes if this is a ModelWASM asset
    pub fn as_model_wasm(&self) -> Option<&[u8]> {
        match self.asset.as_ref() {
            Asset::ModelWASM(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Returns the raw WASM bytes if this is an OperationWASM asset
    pub fn as_operation_wasm(&self) -> Option<&[u8]> {
        match self.asset.as_ref() {
            Asset::OperationWASM(bytes) => Some(bytes),
            _ => None,
        }
    }
}

pub struct Environment {
    loaded_assets: HashMap<String, LoadedAsset>,
}

impl Environment {
    /// Creates a new empty environment
    pub fn new() -> Self {
        Self {
            loaded_assets: HashMap::new(),
        }
    }

    /// Returns a reference to a loaded asset by ID
    pub fn get_asset(&self, asset_id: &str) -> Option<&LoadedAsset> {
        self.loaded_assets.get(asset_id)
    }

    /// Returns all loaded asset IDs
    pub fn asset_ids(&self) -> impl Iterator<Item = &str> {
        self.loaded_assets.keys().map(|s| s.as_str())
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Project {
    /// Creates a new project from a list of entries
    pub fn new(entries: Vec<ProjectEntry>) -> Self {
        Self { entries }
    }

    /// Creates a simple project that loads a ModelWASM and exports it
    /// This is the standard way to import a raw WASM binary into the project system
    pub fn from_model_wasm(asset_id: String, wasm_bytes: Vec<u8>) -> Self {
        Self {
            entries: vec![
                ProjectEntry::LoadAsset(LoadAssetEntry {
                    asset_id: asset_id.clone(),
                    asset: Asset::ModelWASM(wasm_bytes),
                }),
                ProjectEntry::ExportAsset(asset_id),
            ],
        }
    }

    fn first_export_index(&self) -> usize {
        self.entries
            .iter()
            .position(|e| matches!(e, ProjectEntry::ExportAsset(_)))
            .unwrap_or(self.entries.len())
    }

    fn last_declaration_index_for_asset_id(&self, asset_id: &str) -> Option<usize> {
        let mut last: Option<usize> = None;
        for (idx, entry) in self.entries.iter().enumerate() {
            match entry {
                ProjectEntry::LoadAsset(a) => {
                    if a.asset_id() == asset_id {
                        last = Some(idx);
                    }
                }
                ProjectEntry::ExecuteWASM(e) => {
                    if e.outputs.iter().any(|o| o.asset_id == asset_id) {
                        last = Some(idx);
                    }
                }
                ProjectEntry::ExportAsset(_) => {}
            }
        }
        last
    }

    /// Generates a reasonable default output name for an operator based on its type and primary input.
    ///
    /// The name is derived from the operator crate name (e.g., "translate_operator" -> "translated")
    /// and the primary input asset ID. The result is then made unique via `unique_asset_id`.
    pub fn default_output_name(&self, operator_crate_name: &str, primary_input: Option<&str>) -> String {
        // Map operator crate names to descriptive past-tense suffixes
        let suffix = match operator_crate_name {
            "translate_operator" => "translated",
            "rotation_operator" => "rotated",
            "scale_operator" => "scaled",
            "boolean_operator" => "boolean_result",
            "lua_script_operator" => "scripted",
            _ => {
                // For unknown operators, use the crate name with "_output" suffix
                let base = operator_crate_name.strip_suffix("_operator").unwrap_or(operator_crate_name);
                return self.unique_asset_id(&format!("{}_output", base));
            }
        };

        // If we have a primary input, combine it with the suffix
        let base = match primary_input {
            Some(input) => format!("{}_{}", input, suffix),
            None => suffix.to_string(),
        };

        self.unique_asset_id(&base)
    }

    /// Generates a unique asset ID based on the given base name.
    ///
    /// If `base` is not already declared, it is returned as-is. Otherwise, a numeric suffix
    /// (`_2`, `_3`, etc.) is appended until a unique name is found.
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

        // Extremely unlikely; fall back to a timestamp-ish suffix.
        format!("{base}_{}", declared.len() + 1)
    }

    /// Inserts a new `ModelWASM` into this project without replacing existing entries.
    ///
    /// The new `LoadAsset` entry will be inserted before the first `ExportAsset` entry (if any),
    /// so that exports remain at the end of the timeline. A corresponding `ExportAsset` is also
    /// inserted so the model will render.
    ///
    /// Returns the final asset id used (may differ from `asset_id_base` if it collided).
    pub fn insert_model_wasm(&mut self, asset_id_base: &str, wasm_bytes: Vec<u8>) -> String {
        let asset_id = self.unique_asset_id(asset_id_base);
        let insert_at = self.first_export_index();
        self.entries.insert(
            insert_at,
            ProjectEntry::LoadAsset(LoadAssetEntry::new(
                asset_id.clone(),
                Asset::ModelWASM(wasm_bytes),
            )),
        );
        self.entries
            .insert(insert_at + 1, ProjectEntry::ExportAsset(asset_id.clone()));
        asset_id
    }

    /// Inserts a new operator into this project, placing it after all declared input dependencies.
    ///
    /// The operator's `LoadAsset` and `ExecuteWASM` entries are inserted together, followed by an
    /// `ExportAsset` for the primary output.
    pub fn insert_operation(
        &mut self,
        op_asset_id_base: &str,
        wasm_bytes: Vec<u8>,
        inputs: Vec<ExecuteWasmInput>,
        outputs: Vec<ExecuteWasmOutput>,
        export_asset_id: String,
    ) {
        let op_asset_id = self.unique_asset_id(op_asset_id_base);

        let mut dep_after = 0usize;
        for input in &inputs {
            if let ExecuteWasmInput::AssetByID(id) = input {
                if let Some(idx) = self.last_declaration_index_for_asset_id(id.as_str()) {
                    dep_after = dep_after.max(idx + 1);
                }
            }
        }

        let first_export = self.first_export_index();
        let insert_at = first_export.max(dep_after);

        self.entries.insert(
            insert_at,
            ProjectEntry::LoadAsset(LoadAssetEntry::new(
                op_asset_id.clone(),
                Asset::OperationWASM(wasm_bytes),
            )),
        );
        self.entries.insert(
            insert_at + 1,
            ProjectEntry::ExecuteWASM(ExecuteWasmEntry::new(
                op_asset_id,
                inputs,
                outputs,
            )),
        );
        self.entries
            .insert(insert_at + 2, ProjectEntry::ExportAsset(export_asset_id));
    }

    /// Returns the project entries
    pub fn entries(&self) -> &[ProjectEntry] {
        &self.entries
    }

    /// Returns all asset IDs that are declared by this project, along with their `AssetType`.
    ///
    /// This includes both explicitly loaded assets (`ProjectEntry::LoadAsset`) and assets
    /// produced by operation steps (`ProjectEntry::ExecuteWASM` outputs).
    pub fn declared_assets(&self) -> Vec<(String, AssetType)> {
        let mut out: Vec<(String, AssetType)> = Vec::new();
        let mut index_by_id: HashMap<String, usize> = HashMap::new();

        for entry in &self.entries {
            match entry {
                ProjectEntry::LoadAsset(a) => {
                    let id = a.asset_id().to_string();
                    let ty = a.asset_type();
                    if let Some(idx) = index_by_id.get(&id).copied() {
                        out[idx].1 = ty;
                    } else {
                        index_by_id.insert(id.clone(), out.len());
                        out.push((id, ty));
                    }
                }
                ProjectEntry::ExecuteWASM(e) => {
                    for o in &e.outputs {
                        let id = o.asset_id.clone();
                        let ty = o.asset_type;
                        if let Some(idx) = index_by_id.get(&id).copied() {
                            out[idx].1 = ty;
                        } else {
                            index_by_id.insert(id.clone(), out.len());
                            out.push((id, ty));
                        }
                    }
                }
                ProjectEntry::ExportAsset(_) => {}
            }
        }

        out
    }

    /// Returns a mutable reference to the project entries.
    ///
    /// This is primarily intended for UI code to insert new operations into an existing project.
    pub fn entries_mut(&mut self) -> &mut Vec<ProjectEntry> {
        &mut self.entries
    }

    /// Runs the project, executing all entries in order
    pub fn run(&self, environment: &mut Environment) -> Result<Vec<LoadedAsset>, ExecutionError> {
        let mut exported_assets = vec![];
        for entry in &self.entries {
            exported_assets.extend(entry.run(environment)?);
        }
        Ok(exported_assets)
    }

    /// Serializes the project to CBOR format
    pub fn to_cbor(&self) -> Result<Vec<u8>, std::io::Error> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(bytes)
    }

    /// Deserializes a project from CBOR format
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, std::io::Error> {
        ciborium::from_reader(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    /// Saves the project to a file in CBOR format
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let bytes = self.to_cbor()?;
        std::fs::write(path, bytes)
    }

    /// Loads a project from a CBOR file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let bytes = std::fs::read(path)?;
        Self::from_cbor(&bytes)
    }

    /// Moves the entry at the given index up by one position (swaps with the previous entry).
    ///
    /// Returns `true` if the move was performed, `false` if the index is 0 or out of bounds.
    pub fn move_entry_up(&mut self, index: usize) -> bool {
        if index == 0 || index >= self.entries.len() {
            return false;
        }
        self.entries.swap(index, index - 1);
        true
    }

    /// Moves the entry at the given index down by one position (swaps with the next entry).
    ///
    /// Returns `true` if the move was performed, `false` if the index is the last element or out of bounds.
    pub fn move_entry_down(&mut self, index: usize) -> bool {
        if index >= self.entries.len() - 1 {
            return false;
        }
        self.entries.swap(index, index + 1);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exporting_same_asset_twice_is_an_error() {
        let project = Project::new(vec![
            ProjectEntry::LoadAsset(LoadAssetEntry::new(
                "a".to_string(),
                Asset::ModelWASM(vec![1, 2, 3]),
            )),
            ProjectEntry::ExportAsset("a".to_string()),
            ProjectEntry::ExportAsset("a".to_string()),
        ]);

        let mut env = Environment::new();
        let err = project.run(&mut env).unwrap_err();
        match err {
            ExecutionError::NoSuchAssetId(id) => assert_eq!(id, "a"),
            other => panic!("expected NoSuchAssetId, got: {other:?}"),
        }
    }

    #[test]
    fn project_declared_assets_includes_execute_outputs_and_types() {
        let project = Project::new(vec![
            ProjectEntry::LoadAsset(LoadAssetEntry::new(
                "model_a".to_string(),
                Asset::ModelWASM(vec![1, 2, 3]),
            )),
            ProjectEntry::ExecuteWASM(ExecuteWasmEntry::new(
                "op_identity".to_string(),
                vec![ExecuteWasmInput::AssetByID("model_a".to_string())],
                vec![ExecuteWasmOutput::new(
                    "model_b".to_string(),
                    AssetType::ModelWASM,
                )],
            )),
        ]);

        let assets = project.declared_assets();
        assert!(assets
            .iter()
            .any(|(id, ty)| id == "model_a" && *ty == AssetType::ModelWASM));
        assert!(assets
            .iter()
            .any(|(id, ty)| id == "model_b" && *ty == AssetType::ModelWASM));
    }

    #[test]
    fn insert_model_wasm_appends_without_replacing_and_makes_ids_unique() {
        let mut p = Project::new(vec![]);
        let id1 = p.insert_model_wasm("model", vec![1, 2, 3]);
        let id2 = p.insert_model_wasm("model", vec![4, 5, 6]);

        assert_eq!(id1, "model");
        assert_eq!(id2, "model_2");
        assert!(p.entries().len() >= 4);

        let declared: Vec<String> = p.declared_assets().into_iter().map(|(id, _)| id).collect();
        assert!(declared.contains(&"model".to_string()));
        assert!(declared.contains(&"model_2".to_string()));
    }

    #[test]
    fn insert_operation_is_placed_after_input_dependencies() {
        let mut p = Project::new(vec![]);
        let a = p.insert_model_wasm("a", vec![1]);
        let b = p.insert_model_wasm("b", vec![2]);

        // Insert an operation that depends on `b`.
        p.insert_operation(
            "op_identity_operator",
            vec![9, 9, 9],
            vec![ExecuteWasmInput::AssetByID(b.clone())],
            vec![ExecuteWasmOutput::new("out".to_string(), AssetType::ModelWASM)],
            "out".to_string(),
        );

        let mut idx_load_b = None;
        let mut idx_exec = None;
        for (idx, e) in p.entries().iter().enumerate() {
            match e {
                ProjectEntry::LoadAsset(le) if le.asset_id() == b => idx_load_b = Some(idx),
                ProjectEntry::ExecuteWASM(_) => idx_exec = Some(idx),
                _ => {}
            }
        }

        assert!(idx_load_b.is_some());
        assert!(idx_exec.is_some());
        assert!(idx_exec.unwrap() > idx_load_b.unwrap());

        // Sanity: earlier model still present.
        let declared: Vec<String> = p.declared_assets().into_iter().map(|(id, _)| id).collect();
        assert!(declared.contains(&a));
        assert!(declared.contains(&b));
    }
}
