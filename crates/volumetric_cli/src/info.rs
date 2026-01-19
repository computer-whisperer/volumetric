//! Info subcommands for inspecting WASM models, operators, and projects.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;
use wasmtime::{Engine, Instance, Module, Store};

use volumetric::{
    operator_metadata_from_wasm_bytes, ExecutionInput, OperatorMetadata,
    OperatorMetadataInput, OperatorMetadataOutput, Project,
};

// === WASM Type Detection ===

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WasmType {
    Model,
    Operator,
    Unknown,
}

/// ABI version for models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelAbiVersion {
    /// Legacy ABI: is_inside(x, y, z) -> f32, get_bounds_min_x() etc.
    Legacy,
    /// N-dimensional ABI: sample(pos_ptr) -> f32, get_bounds(out_ptr), get_dimensions() -> u32
    Nd,
}

impl std::fmt::Display for ModelAbiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelAbiVersion::Legacy => write!(f, "Legacy (3D)"),
            ModelAbiVersion::Nd => write!(f, "N-dimensional"),
        }
    }
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

/// Detect the ABI version for a model WASM module
fn detect_model_abi(module: &Module) -> ModelAbiVersion {
    let has_sample = module.exports().any(|e| e.name() == "sample");
    let has_get_dimensions = module.exports().any(|e| e.name() == "get_dimensions");
    let has_memory = module.exports().any(|e| e.name() == "memory");

    if has_sample && has_get_dimensions && has_memory {
        ModelAbiVersion::Nd
    } else {
        ModelAbiVersion::Legacy
    }
}

fn detect_wasm_type(wasm_bytes: &[u8]) -> Result<WasmType> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;

    // Check for new N-dimensional ABI first
    let has_sample = module.exports().any(|e| e.name() == "sample");
    let has_get_dimensions = module.exports().any(|e| e.name() == "get_dimensions");
    // Legacy ABI
    let has_is_inside = module.exports().any(|e| e.name() == "is_inside");
    let has_get_metadata = module.exports().any(|e| e.name() == "get_metadata");

    if has_sample && has_get_dimensions {
        Ok(WasmType::Model)
    } else if has_is_inside {
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

/// Get bounds using the legacy ABI (separate get_bounds_min_x etc. functions)
fn get_model_bounds_legacy(
    store: &mut Store<()>,
    instance: &Instance,
) -> Result<Bounds> {
    let min_x = instance
        .get_typed_func::<(), f64>(&mut *store, "get_bounds_min_x")
        .context("Missing get_bounds_min_x export")?
        .call(&mut *store, ())?;
    let min_y = instance
        .get_typed_func::<(), f64>(&mut *store, "get_bounds_min_y")
        .context("Missing get_bounds_min_y export")?
        .call(&mut *store, ())?;
    let min_z = instance
        .get_typed_func::<(), f64>(&mut *store, "get_bounds_min_z")
        .context("Missing get_bounds_min_z export")?
        .call(&mut *store, ())?;
    let max_x = instance
        .get_typed_func::<(), f64>(&mut *store, "get_bounds_max_x")
        .context("Missing get_bounds_max_x export")?
        .call(&mut *store, ())?;
    let max_y = instance
        .get_typed_func::<(), f64>(&mut *store, "get_bounds_max_y")
        .context("Missing get_bounds_max_y export")?
        .call(&mut *store, ())?;
    let max_z = instance
        .get_typed_func::<(), f64>(&mut *store, "get_bounds_max_z")
        .context("Missing get_bounds_max_z export")?
        .call(&mut *store, ())?;

    Ok(Bounds {
        min: [min_x, min_y, min_z],
        max: [max_x, max_y, max_z],
    })
}

/// Get bounds using the N-dimensional ABI (get_bounds(out_ptr) with memory)
fn get_model_bounds_nd(
    store: &mut Store<()>,
    instance: &Instance,
) -> Result<(u32, Bounds)> {
    let memory = instance
        .get_memory(&mut *store, "memory")
        .context("Missing memory export")?;

    let get_dimensions = instance
        .get_typed_func::<(), u32>(&mut *store, "get_dimensions")
        .context("Missing get_dimensions export")?;

    let get_bounds = instance
        .get_typed_func::<i32, ()>(&mut *store, "get_bounds")
        .context("Missing get_bounds export")?;

    let dimensions = get_dimensions.call(&mut *store, ())?;

    // Buffer offset for bounds output (after position buffer)
    const BOUNDS_BUFFER_OFFSET: i32 = 256;

    // Call get_bounds to write bounds to memory
    get_bounds.call(&mut *store, BOUNDS_BUFFER_OFFSET)?;

    // Read bounds from memory (interleaved: min_x, max_x, min_y, max_y, ...)
    let mut bounds_data = vec![0u8; (dimensions as usize) * 2 * 8];
    memory
        .read(&mut *store, BOUNDS_BUFFER_OFFSET as usize, &mut bounds_data)
        .context("Failed to read bounds from memory")?;

    // Parse interleaved bounds - for 3D: [min_x, max_x, min_y, max_y, min_z, max_z]
    let min_x = if dimensions >= 1 {
        f64::from_le_bytes(bounds_data[0..8].try_into().unwrap())
    } else {
        0.0
    };
    let max_x = if dimensions >= 1 {
        f64::from_le_bytes(bounds_data[8..16].try_into().unwrap())
    } else {
        0.0
    };
    let min_y = if dimensions >= 2 {
        f64::from_le_bytes(bounds_data[16..24].try_into().unwrap())
    } else {
        0.0
    };
    let max_y = if dimensions >= 2 {
        f64::from_le_bytes(bounds_data[24..32].try_into().unwrap())
    } else {
        0.0
    };
    let min_z = if dimensions >= 3 {
        f64::from_le_bytes(bounds_data[32..40].try_into().unwrap())
    } else {
        0.0
    };
    let max_z = if dimensions >= 3 {
        f64::from_le_bytes(bounds_data[40..48].try_into().unwrap())
    } else {
        0.0
    };

    Ok((
        dimensions,
        Bounds {
            min: [min_x, min_y, min_z],
            max: [max_x, max_y, max_z],
        },
    ))
}

fn get_model_bounds(wasm_bytes: &[u8]) -> Result<Bounds> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;
    let mut store = Store::new(&engine, ());
    let instance =
        Instance::new(&mut store, &module, &[]).context("Failed to instantiate WASM module")?;

    let abi_version = detect_model_abi(&module);

    match abi_version {
        ModelAbiVersion::Nd => {
            let (_, bounds) = get_model_bounds_nd(&mut store, &instance)?;
            Ok(bounds)
        }
        ModelAbiVersion::Legacy => get_model_bounds_legacy(&mut store, &instance),
    }
}

/// Get model bounds along with ABI version info
fn get_model_bounds_with_abi(wasm_bytes: &[u8]) -> Result<(ModelAbiVersion, u32, Bounds)> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;
    let mut store = Store::new(&engine, ());
    let instance =
        Instance::new(&mut store, &module, &[]).context("Failed to instantiate WASM module")?;

    let abi_version = detect_model_abi(&module);

    match abi_version {
        ModelAbiVersion::Nd => {
            let (dims, bounds) = get_model_bounds_nd(&mut store, &instance)?;
            Ok((abi_version, dims, bounds))
        }
        ModelAbiVersion::Legacy => {
            let bounds = get_model_bounds_legacy(&mut store, &instance)?;
            Ok((abi_version, 3, bounds))
        }
    }
}

// === Sampling ===

/// Sample using the legacy ABI (is_inside(x, y, z) -> f32)
fn sample_model_legacy(
    store: &mut Store<()>,
    instance: &Instance,
    points: &[(f64, f64, f64)],
) -> Result<Vec<f32>> {
    let is_inside = instance
        .get_typed_func::<(f64, f64, f64), f32>(&mut *store, "is_inside")
        .context("Missing is_inside export")?;

    let mut results = Vec::with_capacity(points.len());
    for &(x, y, z) in points {
        let value = is_inside
            .call(&mut *store, (x, y, z))
            .context("is_inside call failed")?;
        results.push(value);
    }
    Ok(results)
}

/// Sample using the N-dimensional ABI (sample(pos_ptr) -> f32 with memory)
fn sample_model_nd(
    store: &mut Store<()>,
    instance: &Instance,
    points: &[(f64, f64, f64)],
) -> Result<Vec<f32>> {
    let memory = instance
        .get_memory(&mut *store, "memory")
        .context("Missing memory export")?;

    let sample = instance
        .get_typed_func::<i32, f32>(&mut *store, "sample")
        .context("Missing sample export")?;

    // Position buffer at offset 0
    const POS_BUFFER_OFFSET: i32 = 0;

    let mut results = Vec::with_capacity(points.len());
    for &(x, y, z) in points {
        // Write position to memory
        let mut pos_bytes = [0u8; 24];
        pos_bytes[0..8].copy_from_slice(&x.to_le_bytes());
        pos_bytes[8..16].copy_from_slice(&y.to_le_bytes());
        pos_bytes[16..24].copy_from_slice(&z.to_le_bytes());

        memory
            .write(&mut *store, POS_BUFFER_OFFSET as usize, &pos_bytes)
            .context("Failed to write position to memory")?;

        // Call sample
        let value = sample
            .call(&mut *store, POS_BUFFER_OFFSET)
            .context("sample call failed")?;
        results.push(value);
    }

    Ok(results)
}

fn sample_model(wasm_bytes: &[u8], points: &[(f64, f64, f64)]) -> Result<Vec<f32>> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;
    let mut store = Store::new(&engine, ());
    let instance =
        Instance::new(&mut store, &module, &[]).context("Failed to instantiate WASM module")?;

    let abi_version = detect_model_abi(&module);

    match abi_version {
        ModelAbiVersion::Nd => sample_model_nd(&mut store, &instance, points),
        ModelAbiVersion::Legacy => sample_model_legacy(&mut store, &instance, points),
    }
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
        abi_version: ModelAbiVersion,
        model_dimensions: u32,
        bounds: Bounds,
        size_dimensions: [f64; 3],
    },
    Operator {
        file_size: usize,
        metadata: OperatorMetadataJson,
    },
    Project {
        version: u32,
        imports: Vec<ImportInfo>,
        timeline: Vec<TimelineStepInfo>,
        exports: Vec<String>,
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
    VecF64 { dimension: usize },
    Blob,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputInfo {
    ModelWasm,
}

#[derive(Debug, Serialize)]
struct ImportInfo {
    id: String,
    type_hint: String,
    size_bytes: usize,
}

#[derive(Debug, Serialize)]
struct TimelineStepInfo {
    operator_id: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
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
                OperatorMetadataInput::VecF64(dim) => InputInfo::VecF64 {
                    dimension: *dim,
                },
                OperatorMetadataInput::Blob => InputInfo::Blob,
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

fn format_input(input: &ExecutionInput) -> String {
    match input {
        ExecutionInput::AssetRef(id) => format!("asset:{}", id),
        ExecutionInput::Inline(d) => format!("inline:{} bytes", d.len()),
    }
}

fn get_project_info(project: &Project) -> Result<InfoOutput> {
    let imports: Vec<ImportInfo> = project
        .imports()
        .iter()
        .map(|i| ImportInfo {
            id: i.id.clone(),
            type_hint: i
                .type_hint
                .map(|h| h.to_string())
                .unwrap_or_else(|| "Binary".to_string()),
            size_bytes: i.data.len(),
        })
        .collect();

    let timeline: Vec<TimelineStepInfo> = project
        .timeline()
        .iter()
        .map(|step| TimelineStepInfo {
            operator_id: step.operator_id.clone(),
            inputs: step.inputs.iter().map(format_input).collect(),
            outputs: step.outputs.clone(),
        })
        .collect();

    Ok(InfoOutput::Project {
        version: project.version,
        imports,
        timeline,
        exports: project.exports().to_vec(),
    })
}

fn print_info_human(output: &InfoOutput) {
    match output {
        InfoOutput::Model {
            file_size,
            abi_version,
            model_dimensions,
            bounds,
            size_dimensions,
        } => {
            println!("=== Model Info ===");
            println!("Type: Model");
            println!("File size: {} bytes", file_size);
            println!("ABI: {} ({}D)", abi_version, model_dimensions);
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
                "Size: {:.3} x {:.3} x {:.3}",
                size_dimensions[0], size_dimensions[1], size_dimensions[2]
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
                    InputInfo::VecF64 { dimension } => {
                        println!("  [{}] VecF64({})", i, dimension);
                    }
                    InputInfo::Blob => {
                        println!("  [{}] Blob (binary data)", i);
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
            version,
            imports,
            timeline,
            exports,
        } => {
            println!("=== Project Info (v{}) ===", version);
            println!();
            println!("Imports ({}):", imports.len());
            for import in imports {
                println!("  {} ({}, {} bytes)", import.id, import.type_hint, import.size_bytes);
            }
            println!();
            println!("Timeline ({} steps):", timeline.len());
            for (i, step) in timeline.iter().enumerate() {
                println!("  [{}] {} -> {:?}", i, step.operator_id, step.outputs);
                println!("       Inputs: {}", step.inputs.join(", "));
            }
            println!();
            println!("Exports ({}):", exports.len());
            for id in exports {
                println!("  {}", id);
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
                    let (abi_version, model_dims, bounds) = get_model_bounds_with_abi(&wasm_bytes)?;
                    let size_dimensions = bounds.dimensions();
                    InfoOutput::Model {
                        file_size: wasm_bytes.len(),
                        abi_version,
                        model_dimensions: model_dims,
                        bounds,
                        size_dimensions,
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
