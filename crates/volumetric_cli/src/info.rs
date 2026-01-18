//! Info subcommands for inspecting WASM models, operators, and projects.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;
use wasmtime::{Engine, Instance, Module, Store};

use volumetric::{
    operator_metadata_from_wasm_bytes, ExecuteWasmInput, OperatorMetadata,
    OperatorMetadataInput, OperatorMetadataOutput, Project, ProjectEntry,
};

// === WASM Type Detection ===

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WasmType {
    Model,
    Operator,
    Unknown,
}

impl std::fmt::Display for WasmType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmType::Model => write!(f, "Model"),
            WasmType::Operator => write!(f, "Operator"),
            WasmType::Unknown => write!(f, "Unknown"),
        }
    }
}

fn detect_wasm_type(wasm_bytes: &[u8]) -> Result<WasmType> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;

    let has_is_inside = module.exports().any(|e| e.name() == "is_inside");
    let has_get_metadata = module.exports().any(|e| e.name() == "get_metadata");

    if has_is_inside {
        Ok(WasmType::Model)
    } else if has_get_metadata {
        Ok(WasmType::Operator)
    } else {
        Ok(WasmType::Unknown)
    }
}

// === Bounds ===

#[derive(Debug, Clone, Serialize)]
pub struct Bounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Bounds {
    pub fn dimensions(&self) -> [f64; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
}

fn get_model_bounds(wasm_bytes: &[u8]) -> Result<Bounds> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;
    let mut store = Store::new(&engine, ());
    let instance =
        Instance::new(&mut store, &module, &[]).context("Failed to instantiate WASM module")?;

    let min_x = instance
        .get_typed_func::<(), f64>(&mut store, "get_bounds_min_x")
        .context("Missing get_bounds_min_x export")?
        .call(&mut store, ())?;
    let min_y = instance
        .get_typed_func::<(), f64>(&mut store, "get_bounds_min_y")
        .context("Missing get_bounds_min_y export")?
        .call(&mut store, ())?;
    let min_z = instance
        .get_typed_func::<(), f64>(&mut store, "get_bounds_min_z")
        .context("Missing get_bounds_min_z export")?
        .call(&mut store, ())?;
    let max_x = instance
        .get_typed_func::<(), f64>(&mut store, "get_bounds_max_x")
        .context("Missing get_bounds_max_x export")?
        .call(&mut store, ())?;
    let max_y = instance
        .get_typed_func::<(), f64>(&mut store, "get_bounds_max_y")
        .context("Missing get_bounds_max_y export")?
        .call(&mut store, ())?;
    let max_z = instance
        .get_typed_func::<(), f64>(&mut store, "get_bounds_max_z")
        .context("Missing get_bounds_max_z export")?
        .call(&mut store, ())?;

    Ok(Bounds {
        min: [min_x, min_y, min_z],
        max: [max_x, max_y, max_z],
    })
}

// === Sampling ===

fn sample_model(wasm_bytes: &[u8], points: &[(f64, f64, f64)]) -> Result<Vec<f32>> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;
    let mut store = Store::new(&engine, ());
    let instance =
        Instance::new(&mut store, &module, &[]).context("Failed to instantiate WASM module")?;
    let is_inside = instance
        .get_typed_func::<(f64, f64, f64), f32>(&mut store, "is_inside")
        .context("Missing is_inside export")?;

    points
        .iter()
        .map(|&(x, y, z)| {
            is_inside
                .call(&mut store, (x, y, z))
                .context("is_inside call failed")
        })
        .collect()
}

fn parse_point(s: &str) -> Result<(f64, f64, f64)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        anyhow::bail!("Point must have exactly 3 coordinates: x,y,z");
    }
    let x: f64 = parts[0]
        .trim()
        .parse()
        .context("Failed to parse x coordinate")?;
    let y: f64 = parts[1]
        .trim()
        .parse()
        .context("Failed to parse y coordinate")?;
    let z: f64 = parts[2]
        .trim()
        .parse()
        .context("Failed to parse z coordinate")?;
    Ok((x, y, z))
}

// === Info Subcommand ===

#[derive(Parser, Debug)]
pub struct InfoArgs {
    /// Input file: .wasm model/operator or .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output as JSON for machine parsing
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum InfoOutput {
    Model {
        file_size: usize,
        bounds: Bounds,
        dimensions: [f64; 3],
    },
    Operator {
        file_size: usize,
        metadata: OperatorMetadataJson,
    },
    Project {
        entries: Vec<EntryInfo>,
        declared_assets: Vec<DeclaredAsset>,
    },
    Unknown {
        file_size: usize,
        message: String,
    },
}

#[derive(Debug, Serialize)]
struct OperatorMetadataJson {
    name: String,
    version: String,
    inputs: Vec<InputInfo>,
    outputs: Vec<OutputInfo>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputInfo {
    ModelWasm,
    CborConfiguration { cddl: String },
    LuaSource { template: String },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputInfo {
    ModelWasm,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum EntryInfo {
    LoadAsset {
        asset_id: String,
        asset_type: String,
        size_bytes: usize,
    },
    ExecuteWasm {
        operator_id: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    Export {
        asset_id: String,
    },
}

#[derive(Debug, Serialize)]
struct DeclaredAsset {
    asset_id: String,
    asset_type: String,
}

fn metadata_to_json(meta: &OperatorMetadata) -> OperatorMetadataJson {
    OperatorMetadataJson {
        name: meta.name.clone(),
        version: meta.version.clone(),
        inputs: meta
            .inputs
            .iter()
            .map(|i| match i {
                OperatorMetadataInput::ModelWASM => InputInfo::ModelWasm,
                OperatorMetadataInput::CBORConfiguration(cddl) => InputInfo::CborConfiguration {
                    cddl: cddl.clone(),
                },
                OperatorMetadataInput::LuaSource(template) => InputInfo::LuaSource {
                    template: template.clone(),
                },
            })
            .collect(),
        outputs: meta
            .outputs
            .iter()
            .map(|o| match o {
                OperatorMetadataOutput::ModelWASM => OutputInfo::ModelWasm,
            })
            .collect(),
    }
}

fn format_input(input: &ExecuteWasmInput) -> String {
    match input {
        ExecuteWasmInput::AssetByID(id) => format!("asset:{}", id),
        ExecuteWasmInput::String(s) => {
            if s.len() > 30 {
                format!("string:\"{}...\"", &s[..27])
            } else {
                format!("string:\"{}\"", s)
            }
        }
        ExecuteWasmInput::Data(d) => format!("data:{} bytes", d.len()),
    }
}

fn get_project_info(project: &Project) -> Result<InfoOutput> {
    let entries: Vec<EntryInfo> = project
        .entries()
        .iter()
        .map(|e| match e {
            ProjectEntry::LoadAsset(a) => {
                let asset_bytes = match a.asset_type() {
                    volumetric::AssetType::ModelWASM => "ModelWASM",
                    volumetric::AssetType::OperationWASM => "OperationWASM",
                };
                // We can't directly access the asset bytes from LoadAssetEntry's public API
                // So we'll use 0 as a placeholder - in practice the asset is embedded
                EntryInfo::LoadAsset {
                    asset_id: a.asset_id().to_string(),
                    asset_type: asset_bytes.to_string(),
                    size_bytes: 0, // Not directly accessible
                }
            }
            ProjectEntry::ExecuteWASM(exec) => EntryInfo::ExecuteWasm {
                operator_id: exec.asset_id().to_string(),
                inputs: exec.inputs().iter().map(format_input).collect(),
                outputs: exec
                    .outputs()
                    .iter()
                    .map(|o| o.asset_id.clone())
                    .collect(),
            },
            ProjectEntry::ExportAsset(id) => EntryInfo::Export {
                asset_id: id.clone(),
            },
        })
        .collect();

    let declared = project
        .declared_assets()
        .into_iter()
        .map(|(id, ty)| DeclaredAsset {
            asset_id: id,
            asset_type: format!("{:?}", ty),
        })
        .collect();

    Ok(InfoOutput::Project {
        entries,
        declared_assets: declared,
    })
}

fn print_info_human(output: &InfoOutput) {
    match output {
        InfoOutput::Model {
            file_size,
            bounds,
            dimensions,
        } => {
            println!("=== Model Info ===");
            println!("Type: Model");
            println!("File size: {} bytes", file_size);
            println!(
                "Bounds: ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3})",
                bounds.min[0],
                bounds.min[1],
                bounds.min[2],
                bounds.max[0],
                bounds.max[1],
                bounds.max[2]
            );
            println!(
                "Dimensions: {:.3} x {:.3} x {:.3}",
                dimensions[0], dimensions[1], dimensions[2]
            );
        }
        InfoOutput::Operator {
            file_size,
            metadata,
        } => {
            println!("=== Operator Info ===");
            println!("Type: Operator");
            println!("File size: {} bytes", file_size);
            println!("Name: {}", metadata.name);
            println!("Version: {}", metadata.version);
            println!("Inputs:");
            for (i, input) in metadata.inputs.iter().enumerate() {
                match input {
                    InputInfo::ModelWasm => println!("  [{}] ModelWASM", i),
                    InputInfo::CborConfiguration { cddl } => {
                        println!("  [{}] CBOR Configuration:", i);
                        for line in cddl.lines() {
                            println!("      {}", line);
                        }
                    }
                    InputInfo::LuaSource { template } => {
                        println!("  [{}] Lua Source (template):", i);
                        for line in template.lines().take(5) {
                            println!("      {}", line);
                        }
                        if template.lines().count() > 5 {
                            println!("      ...");
                        }
                    }
                }
            }
            println!("Outputs:");
            for (i, output) in metadata.outputs.iter().enumerate() {
                match output {
                    OutputInfo::ModelWasm => println!("  [{}] ModelWASM", i),
                }
            }
        }
        InfoOutput::Project {
            entries,
            declared_assets,
        } => {
            println!("=== Project Info ===");
            println!("Entries ({}):", entries.len());
            for (i, entry) in entries.iter().enumerate() {
                match entry {
                    EntryInfo::LoadAsset {
                        asset_id,
                        asset_type,
                        size_bytes,
                    } => {
                        if *size_bytes > 0 {
                            println!(
                                "  [{}] LoadAsset: {} ({}, {} bytes)",
                                i, asset_id, asset_type, size_bytes
                            );
                        } else {
                            println!("  [{}] LoadAsset: {} ({})", i, asset_id, asset_type);
                        }
                    }
                    EntryInfo::ExecuteWasm {
                        operator_id,
                        inputs,
                        outputs,
                    } => {
                        println!("  [{}] ExecuteWASM: {}", i, operator_id);
                        println!("       Inputs: {}", inputs.join(", "));
                        println!("       Outputs: {}", outputs.join(", "));
                    }
                    EntryInfo::Export { asset_id } => {
                        println!("  [{}] Export: {}", i, asset_id);
                    }
                }
            }
            println!();
            println!("Declared assets ({}):", declared_assets.len());
            for asset in declared_assets {
                println!("  - {} ({})", asset.asset_id, asset.asset_type);
            }
        }
        InfoOutput::Unknown { file_size, message } => {
            println!("=== Unknown WASM ===");
            println!("Type: Unknown");
            println!("File size: {} bytes", file_size);
            println!("Note: {}", message);
        }
    }
}

pub fn run_info(args: InfoArgs) -> Result<()> {
    let extension = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let output = match extension.as_str() {
        "wasm" => {
            let wasm_bytes =
                std::fs::read(&args.input).context("Failed to read WASM file")?;
            let wasm_type = detect_wasm_type(&wasm_bytes)?;

            match wasm_type {
                WasmType::Model => {
                    let bounds = get_model_bounds(&wasm_bytes)?;
                    let dimensions = bounds.dimensions();
                    InfoOutput::Model {
                        file_size: wasm_bytes.len(),
                        bounds,
                        dimensions,
                    }
                }
                WasmType::Operator => {
                    let metadata = operator_metadata_from_wasm_bytes(&wasm_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to get operator metadata: {}", e))?;
                    InfoOutput::Operator {
                        file_size: wasm_bytes.len(),
                        metadata: metadata_to_json(&metadata),
                    }
                }
                WasmType::Unknown => InfoOutput::Unknown {
                    file_size: wasm_bytes.len(),
                    message: "WASM file does not appear to be a model or operator".to_string(),
                },
            }
        }
        "vproj" => {
            let project =
                Project::load_from_file(&args.input).context("Failed to load .vproj file")?;
            get_project_info(&project)?
        }
        _ => {
            anyhow::bail!(
                "Unknown file extension: {:?}. Expected .wasm or .vproj",
                extension
            );
        }
    };

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize to JSON")?
        );
    } else {
        print_info_human(&output);
    }

    Ok(())
}

// === Bounds Subcommand ===

#[derive(Parser, Debug)]
pub struct BoundsArgs {
    /// Input file: .wasm model or .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct BoundsOutput {
    bounds: Bounds,
    dimensions: [f64; 3],
}

fn print_bounds_human(bounds: &Bounds) {
    let dims = bounds.dimensions();
    println!(
        "Bounds: ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3})",
        bounds.min[0], bounds.min[1], bounds.min[2], bounds.max[0], bounds.max[1], bounds.max[2]
    );
    println!("Dimensions: {:.3} x {:.3} x {:.3}", dims[0], dims[1], dims[2]);
}

pub fn run_bounds(args: BoundsArgs) -> Result<()> {
    let wasm_bytes = crate::load_wasm_bytes(&args.input)?;
    let wasm_type = detect_wasm_type(&wasm_bytes)?;

    if wasm_type != WasmType::Model {
        anyhow::bail!(
            "Input is not a model (detected type: {}). Cannot query bounds.",
            wasm_type
        );
    }

    let bounds = get_model_bounds(&wasm_bytes)?;

    if args.json {
        let output = BoundsOutput {
            dimensions: bounds.dimensions(),
            bounds,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize to JSON")?
        );
    } else {
        print_bounds_human(&bounds);
    }

    Ok(())
}

// === Sample Subcommand ===

#[derive(Parser, Debug)]
pub struct SampleArgs {
    /// Input file: .wasm model or .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// Points to sample as "x,y,z" (can specify multiple)
    #[arg(short, long)]
    pub point: Vec<String>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct SampleOutput {
    samples: Vec<SampleResult>,
}

#[derive(Debug, Serialize)]
struct SampleResult {
    point: [f64; 3],
    value: f32,
}

fn print_samples_human(points: &[(f64, f64, f64)], values: &[f32]) {
    for (point, value) in points.iter().zip(values.iter()) {
        println!(
            "({:.3}, {:.3}, {:.3}) -> {:.6}",
            point.0, point.1, point.2, value
        );
    }
}

pub fn run_sample(args: SampleArgs) -> Result<()> {
    if args.point.is_empty() {
        anyhow::bail!("At least one point must be specified with -p/--point");
    }

    let wasm_bytes = crate::load_wasm_bytes(&args.input)?;
    let wasm_type = detect_wasm_type(&wasm_bytes)?;

    if wasm_type != WasmType::Model {
        anyhow::bail!(
            "Input is not a model (detected type: {}). Cannot sample.",
            wasm_type
        );
    }

    let points: Vec<(f64, f64, f64)> = args
        .point
        .iter()
        .enumerate()
        .map(|(i, s)| parse_point(s).with_context(|| format!("Invalid point at index {}", i)))
        .collect::<Result<_>>()?;

    let values = sample_model(&wasm_bytes, &points)?;

    if args.json {
        let output = SampleOutput {
            samples: points
                .iter()
                .zip(values.iter())
                .map(|((x, y, z), v)| SampleResult {
                    point: [*x, *y, *z],
                    value: *v,
                })
                .collect(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize to JSON")?
        );
    } else {
        print_samples_human(&points, &values);
    }

    Ok(())
}
