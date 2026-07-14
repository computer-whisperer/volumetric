//! Info subcommands for inspecting WASM models, operators, and projects.
//!
//! Model access goes through [`volumetric::wasm::ModelExecutor`], so these
//! commands see exactly what the meshing/preview pipeline sees — including
//! 2D sketches and typed sample channels.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;
use wasmtime::{Engine, Module};

use volumetric::{
    ExecutionInput, OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, Project,
    operator_metadata_from_wasm_bytes,
    wasm::{ModelExecutor, create_model_executor},
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
    let module = Module::new(&engine, wasm_bytes)
        .map_err(|err| anyhow::anyhow!("Failed to parse WASM module: {err}"))?;

    let has_export = |name: &str| module.exports().any(|e| e.name() == name);

    if has_export("sample")
        && has_export("get_dimensions")
        && has_export("get_io_ptr")
        && has_export("memory")
    {
        Ok(WasmType::Model)
    } else if has_export("get_metadata") && has_export("run") {
        Ok(WasmType::Operator)
    } else {
        Ok(WasmType::Unknown)
    }
}

// === Model inspection via ModelExecutor ===

/// Per-dimension bounds of an N-dimensional model.
#[derive(Debug, Clone, Serialize)]
pub struct BoundsNd {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

impl BoundsNd {
    pub fn size(&self) -> Vec<f64> {
        self.min
            .iter()
            .zip(&self.max)
            .map(|(lo, hi)| hi - lo)
            .collect()
    }

    fn format_tuple(values: &[f64]) -> String {
        let parts: Vec<String> = values.iter().map(|v| format!("{v:.3}")).collect();
        format!("({})", parts.join(", "))
    }
}

#[derive(Debug, Serialize)]
struct ChannelInfo {
    name: String,
    kind: String,
}

fn channel_kind_name(kind: &volumetric_abi::ChannelKind) -> String {
    match kind {
        volumetric_abi::ChannelKind::Occupancy => "occupancy".to_string(),
        volumetric_abi::ChannelKind::Density => "density".to_string(),
        volumetric_abi::ChannelKind::Custom(name) => format!("custom:{name}"),
    }
}

/// Everything `info` and `bounds` report about a model, gathered in one
/// executor instantiation.
struct ModelReport {
    dimensions: u32,
    bounds: BoundsNd,
    channels: Vec<ChannelInfo>,
}

fn inspect_model(wasm_bytes: &[u8]) -> Result<ModelReport> {
    let mut executor = create_model_executor(wasm_bytes).context("Failed to instantiate model")?;

    let dimensions = executor.dimensions().context("get_dimensions failed")?;
    let nd = executor.get_bounds_nd().context("get_bounds failed")?;
    let bounds = BoundsNd {
        min: (0..nd.dimensions()).map(|d| nd.min(d)).collect(),
        max: (0..nd.dimensions()).map(|d| nd.max(d)).collect(),
    };
    let format = executor
        .sample_format()
        .context("get_sample_format failed")?;
    let channels = format
        .channels
        .iter()
        .map(|c| ChannelInfo {
            name: c.name.clone(),
            kind: channel_kind_name(&c.kind),
        })
        .collect();

    Ok(ModelReport {
        dimensions,
        bounds,
        channels,
    })
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
        dimensions: u32,
        bounds: BoundsNd,
        size: Vec<f64>,
        channels: Vec<ChannelInfo>,
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
        issues: Vec<String>,
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
    #[serde(skip_serializing_if = "String::is_empty")]
    display_name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    description: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    category: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    icon_svg: String,
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
    FeaMesh,
    TriMesh,
    Subspace,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputInfo {
    ModelWasm,
    FeaMesh,
    TriMesh,
    Subspace,
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
        display_name: meta.display_name.clone(),
        description: meta.description.clone(),
        category: meta.category.clone(),
        icon_svg: meta.icon_svg.clone(),
        inputs: meta
            .inputs
            .iter()
            .map(|i| match i {
                OperatorMetadataInput::ModelWASM => InputInfo::ModelWasm,
                OperatorMetadataInput::CBORConfiguration(cddl) => {
                    InputInfo::CborConfiguration { cddl: cddl.clone() }
                }
                OperatorMetadataInput::LuaSource(template) => InputInfo::LuaSource {
                    template: template.clone(),
                },
                OperatorMetadataInput::VecF64(dim) => InputInfo::VecF64 { dimension: *dim },
                OperatorMetadataInput::Blob => InputInfo::Blob,
                OperatorMetadataInput::FeaMesh => InputInfo::FeaMesh,
                OperatorMetadataInput::TriMesh => InputInfo::TriMesh,
                OperatorMetadataInput::Subspace => InputInfo::Subspace,
            })
            .collect(),
        outputs: meta
            .outputs
            .iter()
            .map(|o| match o {
                OperatorMetadataOutput::ModelWASM => OutputInfo::ModelWasm,
                OperatorMetadataOutput::FeaMesh => OutputInfo::FeaMesh,
                OperatorMetadataOutput::TriMesh => OutputInfo::TriMesh,
                OperatorMetadataOutput::Subspace => OutputInfo::Subspace,
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

    let issues = project
        .validate()
        .into_iter()
        .map(|i| i.to_string())
        .collect();

    Ok(InfoOutput::Project {
        version: project.version,
        imports,
        timeline,
        exports: project.exports().to_vec(),
        issues,
    })
}

fn print_info_human(output: &InfoOutput) {
    match output {
        InfoOutput::Model {
            file_size,
            dimensions,
            bounds,
            size,
            channels,
        } => {
            println!("=== Model Info ===");
            println!("Type: Model ({}D)", dimensions);
            println!("File size: {} bytes", file_size);
            println!(
                "Bounds: {} to {}",
                BoundsNd::format_tuple(&bounds.min),
                BoundsNd::format_tuple(&bounds.max)
            );
            let size_parts: Vec<String> = size.iter().map(|v| format!("{v:.3}")).collect();
            println!("Size: {}", size_parts.join(" x "));
            println!("Channels ({}):", channels.len());
            for (i, c) in channels.iter().enumerate() {
                println!("  [{}] {} ({})", i, c.name, c.kind);
            }
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
            if !metadata.display_name.is_empty() {
                println!("Display name: {}", metadata.display_name);
            }
            if !metadata.description.is_empty() {
                println!("Description: {}", metadata.description);
            }
            if !metadata.category.is_empty() {
                println!("Category: {}", metadata.category);
            }
            if !metadata.icon_svg.is_empty() {
                println!("Icon: {} bytes of SVG", metadata.icon_svg.len());
            }
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
                    InputInfo::FeaMesh => {
                        println!("  [{}] FEA Mesh", i);
                    }
                    InputInfo::TriMesh => {
                        println!("  [{}] Triangle Mesh", i);
                    }
                    InputInfo::Subspace => {
                        println!("  [{}] Subspace", i);
                    }
                }
            }
            println!("Outputs:");
            for (i, output) in metadata.outputs.iter().enumerate() {
                match output {
                    OutputInfo::ModelWasm => println!("  [{}] ModelWASM", i),
                    OutputInfo::FeaMesh => println!("  [{}] FEA Mesh", i),
                    OutputInfo::TriMesh => println!("  [{}] Triangle Mesh", i),
                    OutputInfo::Subspace => println!("  [{}] Subspace", i),
                }
            }
        }
        InfoOutput::Project {
            version,
            imports,
            timeline,
            exports,
            issues,
        } => {
            println!("=== Project Info (v{}) ===", version);
            println!();
            println!("Imports ({}):", imports.len());
            for import in imports {
                println!(
                    "  {} ({}, {} bytes)",
                    import.id, import.type_hint, import.size_bytes
                );
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
            if !issues.is_empty() {
                println!();
                println!("Issues ({}):", issues.len());
                for issue in issues {
                    println!("  {}", issue);
                }
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
            let wasm_bytes = std::fs::read(&args.input).context("Failed to read WASM file")?;
            let wasm_type = detect_wasm_type(&wasm_bytes)?;

            match wasm_type {
                WasmType::Model => {
                    let report = inspect_model(&wasm_bytes)?;
                    InfoOutput::Model {
                        file_size: wasm_bytes.len(),
                        dimensions: report.dimensions,
                        size: report.bounds.size(),
                        bounds: report.bounds,
                        channels: report.channels,
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
                    message: "WASM file exports neither the N-dimensional model ABI \
                              (sample/get_bounds/get_dimensions/get_io_ptr/memory) nor the \
                              operator ABI (run/get_metadata)"
                        .to_string(),
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

    /// For .vproj inputs with multiple exports: which exported asset to use
    #[arg(long)]
    pub asset: Option<String>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct BoundsOutput {
    dimensions: u32,
    bounds: BoundsNd,
    size: Vec<f64>,
}

pub fn run_bounds(args: BoundsArgs) -> Result<()> {
    let wasm_bytes = crate::load_wasm_bytes(&args.input, args.asset.as_deref())?;
    let report = inspect_model(&wasm_bytes)?;

    if args.json {
        let output = BoundsOutput {
            dimensions: report.dimensions,
            size: report.bounds.size(),
            bounds: report.bounds,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize to JSON")?
        );
    } else {
        println!("Dimensions: {}D", report.dimensions);
        println!(
            "Bounds: {} to {}",
            BoundsNd::format_tuple(&report.bounds.min),
            BoundsNd::format_tuple(&report.bounds.max)
        );
        let size_parts: Vec<String> = report
            .bounds
            .size()
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect();
        println!("Size: {}", size_parts.join(" x "));
    }

    Ok(())
}

// === Sample Subcommand ===

#[derive(Parser, Debug)]
pub struct SampleArgs {
    /// Input file: .wasm model or .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// For .vproj inputs with multiple exports: which exported asset to use
    #[arg(long)]
    pub asset: Option<String>,

    /// Points to sample as comma-separated coordinates matching the model's
    /// dimensionality: "x,y" for a 2D sketch, "x,y,z" for a volume
    #[arg(short, long, allow_hyphen_values = true)]
    pub point: Vec<String>,

    /// Sample N evenly spaced points along a segment, endpoints included:
    /// "x0,y0,z0 : x1,y1,z1 : N" (2D models take "x0,y0 : x1,y1 : N")
    #[arg(short, long, allow_hyphen_values = true)]
    pub line: Vec<String>,

    /// Print every declared sample channel, not just occupancy
    #[arg(short, long)]
    pub channels: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct SampleOutput {
    dimensions: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    channels: Option<Vec<String>>,
    samples: Vec<SampleResult>,
}

#[derive(Debug, Serialize)]
struct SampleResult {
    point: Vec<f64>,
    value: f32,
    occupied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    values: Option<Vec<f32>>,
}

fn parse_point(s: &str) -> Result<Vec<f64>> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .with_context(|| format!("Failed to parse coordinate '{}'", part.trim()))
        })
        .collect()
}

/// Expand a "start : end : N" line spec into N evenly spaced points
/// (endpoints included; N=1 yields the start point).
fn parse_line(s: &str) -> Result<Vec<Vec<f64>>> {
    let parts: Vec<&str> = s.split(':').collect();
    let [start, end, count] = parts.as_slice() else {
        anyhow::bail!(
            "line spec must be \"x0,y0,z0 : x1,y1,z1 : N\" (three ':'-separated parts), got '{s}'"
        );
    };
    let start = parse_point(start)?;
    let end = parse_point(end)?;
    if start.len() != end.len() {
        anyhow::bail!(
            "line endpoints disagree on dimensionality ({} vs {} coordinates)",
            start.len(),
            end.len()
        );
    }
    let count: usize = count
        .trim()
        .parse()
        .with_context(|| format!("Failed to parse sample count '{}'", count.trim()))?;
    if count == 0 {
        anyhow::bail!("line sample count must be at least 1");
    }
    Ok((0..count)
        .map(|i| {
            let t = if count == 1 {
                0.0
            } else {
                i as f64 / (count - 1) as f64
            };
            start
                .iter()
                .zip(&end)
                .map(|(a, b)| a + (b - a) * t)
                .collect()
        })
        .collect())
}

pub fn run_sample(args: SampleArgs) -> Result<()> {
    if args.point.is_empty() && args.line.is_empty() {
        anyhow::bail!("Specify points with -p/--point and/or lines with -l/--line");
    }

    let wasm_bytes = crate::load_wasm_bytes(&args.input, args.asset.as_deref())?;

    let mut points: Vec<Vec<f64>> = args
        .point
        .iter()
        .enumerate()
        .map(|(i, s)| parse_point(s).with_context(|| format!("Invalid point at index {}", i)))
        .collect::<Result<_>>()?;
    for (i, spec) in args.line.iter().enumerate() {
        points.extend(parse_line(spec).with_context(|| format!("Invalid line at index {}", i))?);
    }

    let mut executor = volumetric::wasm::native::NativeModelExecutor::new(&wasm_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to instantiate model: {e}"))?;
    let dimensions = executor.dimensions();

    for point in &points {
        if point.len() != dimensions as usize {
            anyhow::bail!(
                "model is {}D but point {} has {} coordinate(s)",
                dimensions,
                BoundsNd::format_tuple(point),
                point.len()
            );
        }
    }

    let channel_names: Option<Vec<String>> = args.channels.then(|| {
        executor
            .sample_format()
            .channels
            .iter()
            .map(|c| c.name.clone())
            .collect()
    });

    let mut samples = Vec::with_capacity(points.len());
    for point in points {
        let (value, values) = if args.channels {
            let row = executor
                .sample_channels_nd(&point)
                .context("sample_channels failed")?;
            (row[0], Some(row))
        } else {
            (executor.sample_nd(&point).context("sample failed")?, None)
        };
        samples.push(SampleResult {
            point,
            value,
            occupied: volumetric_abi::is_occupied(value),
            values,
        });
    }

    if args.json {
        let output = SampleOutput {
            dimensions,
            channels: channel_names,
            samples,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize to JSON")?
        );
    } else {
        if let Some(names) = &channel_names {
            println!("channels: [{}]", names.join(", "));
        }
        for s in &samples {
            let channel_suffix = s
                .values
                .as_ref()
                .map(|row| {
                    let cells: Vec<String> = row.iter().map(|v| format!("{v:.6}")).collect();
                    format!(" [{}]", cells.join(", "))
                })
                .unwrap_or_default();
            println!(
                "{} -> {:.6} ({}){channel_suffix}",
                BoundsNd::format_tuple(&s.point),
                s.value,
                if s.occupied { "inside" } else { "outside" }
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_parse_at_any_arity() {
        assert_eq!(parse_point("1,2").unwrap(), vec![1.0, 2.0]);
        assert_eq!(parse_point(" 1 , 2 , 3 ").unwrap(), vec![1.0, 2.0, 3.0]);
        assert!(parse_point("1,two").is_err());
    }

    #[test]
    fn lines_expand_to_evenly_spaced_points() {
        let points = parse_line("-1,0,0 : 1,0,0 : 3").unwrap();
        assert_eq!(
            points,
            vec![
                vec![-1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0]
            ]
        );
        // A single sample sits at the start; endpoints always included.
        assert_eq!(parse_line("2,3 : 4,5 : 1").unwrap(), vec![vec![2.0, 3.0]]);
        let two = parse_line("0,0,0 : 1,2,4 : 2").unwrap();
        assert_eq!(two, vec![vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 4.0]]);

        assert!(parse_line("0,0,0 : 1,1,1").is_err(), "missing count");
        assert!(parse_line("0,0 : 1,1,1 : 4").is_err(), "arity mismatch");
        assert!(parse_line("0,0,0 : 1,1,1 : 0").is_err(), "zero count");
    }
}
