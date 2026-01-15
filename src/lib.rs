use std::collections::HashMap;
use std::fmt::Display;
use std::sync::Arc;
use wasmtime::{Caller, Instance, Store};

#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("No loaded asset with id: {0}")]
    NoSuchAssetId(String),
    #[error("Asset with id {0} is not of type {1}, but {2}")]
    WrongAssetType(String, AssetType, AssetType),
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
    pub fn asset_type(&self) -> AssetType {
        match self {
            Asset::ModelWASM(_) => AssetType::ModelWASM,
            Asset::OperationWASM(_) => AssetType::OperationWASM,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct LoadAssetEntry {
    asset_id: String,
    asset: Asset,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ExecuteWasmInput {
    AssetByID(String),
    String(String),
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
struct ExecuteWasmOutput {
    asset_id: String,
    asset_type: AssetType,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ExecuteWasmEntry {
    asset_id: String,
    inputs: Vec<ExecuteWasmInput>,
    outputs: Vec<ExecuteWasmOutput>
}

impl ExecuteWasmEntry {
    pub fn run(&self, environment: &mut Environment) -> Result<(), ExecutionError> {
        let wasm_asset = environment.loaded_assets.get(&self.asset_id).ok_or(ExecutionError::NoSuchAssetId(self.asset_id.clone()))?;

        // Pre-fetch needed assets
        let mut needed_assets = HashMap::new();
        let mut precursor_asset_ids = vec![];
        for input in &self.inputs {
            match input {
                ExecuteWasmInput::AssetByID(asset_id) => {
                    needed_assets.insert(
                        asset_id.clone(),
                        environment.loaded_assets.get(asset_id).ok_or(ExecutionError::NoSuchAssetId(asset_id.clone()))?.clone()
                    );
                    precursor_asset_ids.push(asset_id.clone());
                },
                ExecuteWasmInput::String(_) => (),
            }
        }

        let wasm_bin = if let Asset::OperationWASM(wasm_bin) = wasm_asset.asset.as_ref() { wasm_bin } else { return Err(ExecutionError::NoSuchAssetId(self.asset_id.clone())) };
        let engine = wasmtime::Engine::new(wasmtime::Config::new().debug_info(true)).unwrap();
        let module = wasmtime::Module::new(&engine, &wasm_bin).unwrap();

        let mut linker = wasmtime::Linker::new(&engine);
        let mut store = Store::new(&engine, (&self.inputs, needed_assets, vec![]));


        linker.func_wrap("host", "get_input_len", |caller: Caller<'_, (&Vec<ExecuteWasmInput>, HashMap<String, LoadedAsset>, Vec<()>)>, arg: i32|{
            match caller.data().0.get(arg as usize) {
                Some(ExecuteWasmInput::AssetByID(asset_id)) => {
                    let asset = caller.data().1.get(asset_id).ok_or(ExecutionError::NoSuchAssetId(asset_id.clone()))?;
                    match asset.asset.as_ref() {
                        Asset::ModelWASM(wasm) => {
                            wasm.len() as u32
                        }
                        Asset::OperationWASM(wasm) => {
                            wasm.len() as u32
                        }
                    }
                },
                Some(ExecuteWasmInput::String(arg_str)) => arg_str.len() as u32,
                None => panic!("Invalid argument index"),
            }
        }).unwrap();

        linker.func_wrap("host", "get_input_data", |caller: Caller<'_, (&Vec<ExecuteWasmInput>, HashMap<String, LoadedAsset>, Vec<()>)>, arg: i32, ptr: i32, len: i32|{
            // Copy data from input into the indicated location
            match caller.data().0.get(arg as usize) {
                Some(ExecuteWasmInput::AssetByID(asset_id)) => {
                    todo!();
                },
                Some(ExecuteWasmInput::String(arg_str)) => arg_str,
                None => panic!("Invalid argument index"),
            };
            // TODO
        }).unwrap();

        linker.func_wrap("host", "post_output", |caller: Caller<'_, (&Vec<ExecuteWasmInput>, HashMap<String, LoadedAsset>, Vec<()>)>, arg: i32, ptr: i32, len: i32|{
            // Copy from the indicated location into a new LoadedAsset
            todo!();
        }).unwrap();

        let instance = linker.instantiate(&mut store, &module).unwrap();

        let run_func = instance.get_typed_func::<(), ()>(&mut store, "run").unwrap();
        run_func.call(&mut store, ()).unwrap();
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
                    asset: entry.asset.clone(),
                    precursor_asset_ids: vec![],
                });
                Ok(vec![])
            }
            ProjectEntry::ExecuteWASM(entry) => {
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

pub struct Environment {
    loaded_assets: HashMap<String, LoadedAsset>,
}

impl Project {
    pub fn run(&self, environment: &mut Environment) -> Result<Vec<LoadedAsset>, ExecutionError> {
        let mut exported_assets = vec![];
        for entry in &self.entries {
            exported_assets.extend(entry.run(environment)?);
        }
        Ok(vec![])
    }
}
