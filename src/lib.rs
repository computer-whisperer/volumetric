use std::collections::HashMap;
use std::fmt::Display;
use std::sync::Arc;
use wasmtime::{Caller, Store};

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

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
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

    /// Returns the project entries
    pub fn entries(&self) -> &[ProjectEntry] {
        &self.entries
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
}
