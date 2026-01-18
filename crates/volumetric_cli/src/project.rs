//! Project manipulation subcommands for creating and modifying .vproj files.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

use volumetric::{
    AssetType, Environment, ExecuteWasmInput, ExecuteWasmOutput, Project, ProjectEntry,
};

// === Project New ===

#[derive(Parser, Debug)]
pub struct ProjectNewArgs {
    /// Input WASM model file
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output .vproj file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Asset ID for the model (defaults to filename without extension)
    #[arg(long)]
    pub asset_id: Option<String>,
}

pub fn run_project_new(args: ProjectNewArgs) -> Result<()> {
    let wasm_bytes = std::fs::read(&args.input).context("Failed to read WASM file")?;

    let asset_id = args.asset_id.unwrap_or_else(|| {
        args.input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string()
    });

    let project = Project::from_model_wasm(asset_id.clone(), wasm_bytes);
    project
        .save_to_file(&args.output)
        .context("Failed to save project")?;

    println!("Created project with model '{}'", asset_id);
    println!("Saved to {:?}", args.output);
    Ok(())
}

// === Project Add Model ===

#[derive(Parser, Debug)]
pub struct ProjectAddModelArgs {
    /// Project file to modify
    #[arg(short, long)]
    pub project: PathBuf,

    /// WASM model file to add
    #[arg(short, long)]
    pub input: PathBuf,

    /// Asset ID for the model (defaults to filename without extension)
    #[arg(long)]
    pub asset_id: Option<String>,

    /// Output project file (defaults to overwriting input)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Don't auto-export this asset (useful for intermediate results)
    #[arg(long)]
    pub no_export: bool,
}

pub fn run_project_add_model(args: ProjectAddModelArgs) -> Result<()> {
    let mut project =
        Project::load_from_file(&args.project).context("Failed to load project")?;
    let wasm_bytes = std::fs::read(&args.input).context("Failed to read WASM file")?;

    let asset_id_base = args.asset_id.unwrap_or_else(|| {
        args.input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string()
    });

    let asset_id = project.insert_model_wasm(&asset_id_base, wasm_bytes);

    // Remove the auto-added export if --no-export was specified
    if args.no_export {
        project.entries_mut().retain(|e| {
            !matches!(e, ProjectEntry::ExportAsset(id) if id == &asset_id)
        });
    }

    let output_path = args.output.unwrap_or(args.project);
    project
        .save_to_file(&output_path)
        .context("Failed to save project")?;

    println!("Added model '{}' to project", asset_id);
    println!("Saved to {:?}", output_path);
    Ok(())
}

// === Project Add Operator ===

#[derive(Parser, Debug)]
pub struct ProjectAddOpArgs {
    /// Project file to modify
    #[arg(short, long)]
    pub project: PathBuf,

    /// Operator WASM file
    #[arg(long)]
    pub operator: PathBuf,

    /// Input asset IDs or literal values (format: "asset:id", "string:value", "json:{...}", or "data:base64")
    #[arg(short, long)]
    pub input: Vec<String>,

    /// Output asset ID (auto-generated if not specified)
    #[arg(long)]
    pub output_id: Option<String>,

    /// Output project file (defaults to overwriting input)
    #[arg(short = 'O', long)]
    pub output: Option<PathBuf>,

    /// Don't auto-export the output asset (useful for intermediate results)
    #[arg(long)]
    pub no_export: bool,
}

fn parse_input(s: &str) -> Result<ExecuteWasmInput> {
    if let Some(rest) = s.strip_prefix("asset:") {
        Ok(ExecuteWasmInput::AssetByID(rest.to_string()))
    } else if let Some(rest) = s.strip_prefix("string:") {
        Ok(ExecuteWasmInput::String(rest.to_string()))
    } else if let Some(rest) = s.strip_prefix("json:") {
        // Parse JSON and convert to CBOR
        let json_value: serde_json::Value =
            serde_json::from_str(rest).context("Failed to parse JSON")?;
        let mut cbor_bytes = Vec::new();
        ciborium::into_writer(&json_value, &mut cbor_bytes)
            .context("Failed to convert JSON to CBOR")?;
        Ok(ExecuteWasmInput::Data(cbor_bytes))
    } else if let Some(rest) = s.strip_prefix("data:") {
        // Assume base64 encoded data
        use base64::{engine::general_purpose::STANDARD, Engine};
        let bytes = STANDARD
            .decode(rest)
            .context("Failed to decode base64 data")?;
        Ok(ExecuteWasmInput::Data(bytes))
    } else {
        // Default: treat as asset ID
        Ok(ExecuteWasmInput::AssetByID(s.to_string()))
    }
}

pub fn run_project_add_op(args: ProjectAddOpArgs) -> Result<()> {
    let mut project =
        Project::load_from_file(&args.project).context("Failed to load project")?;
    let op_bytes = std::fs::read(&args.operator).context("Failed to read operator WASM")?;

    let op_name = args
        .operator
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("operator")
        .to_string();

    let inputs: Vec<ExecuteWasmInput> = args
        .input
        .iter()
        .map(|s| parse_input(s))
        .collect::<Result<_>>()?;

    // Get primary input for naming
    let primary_input = inputs.iter().find_map(|i| match i {
        ExecuteWasmInput::AssetByID(id) => Some(id.as_str()),
        _ => None,
    });

    let output_id = args
        .output_id
        .unwrap_or_else(|| project.default_output_name(&op_name, primary_input));

    let outputs = vec![ExecuteWasmOutput::new(
        output_id.clone(),
        AssetType::ModelWASM,
    )];

    project.insert_operation(&op_name, op_bytes, inputs, outputs, output_id.clone());

    // Remove the auto-added export if --no-export was specified
    if args.no_export {
        project.entries_mut().retain(|e| {
            !matches!(e, ProjectEntry::ExportAsset(id) if id == &output_id)
        });
    }

    let output_path = args.output.unwrap_or(args.project);
    project
        .save_to_file(&output_path)
        .context("Failed to save project")?;

    println!("Added operator '{}' with output '{}'", op_name, output_id);
    println!("Saved to {:?}", output_path);
    Ok(())
}

// === Project Export ===

#[derive(Parser, Debug)]
pub struct ProjectExportArgs {
    /// Project file to export from
    #[arg(short, long)]
    pub project: PathBuf,

    /// Output directory for exported WASM files
    #[arg(short, long)]
    pub output: PathBuf,

    /// Specific asset IDs to export (exports all if not specified)
    #[arg(long)]
    pub asset: Vec<String>,

    /// Output as JSON (list of exported files)
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct ExportResult {
    exports: Vec<ExportedAsset>,
}

#[derive(Debug, Serialize)]
struct ExportedAsset {
    asset_id: String,
    asset_type: String,
    file_path: String,
    size_bytes: usize,
}

pub fn run_project_export(args: ProjectExportArgs) -> Result<()> {
    let project = Project::load_from_file(&args.project).context("Failed to load project")?;

    // Create output directory if needed
    std::fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    // Run the project to get exported assets
    let mut env = Environment::new();
    let exports = project
        .run(&mut env)
        .map_err(|e| anyhow::anyhow!("Project execution failed: {}", e))?;

    // Filter exports if specific assets requested
    let filtered_exports: Vec<_> = if args.asset.is_empty() {
        exports
    } else {
        exports
            .into_iter()
            .filter(|e| args.asset.contains(&e.asset_id().to_string()))
            .collect()
    };

    let mut results = Vec::new();

    for export in filtered_exports {
        let asset_id = export.asset_id();
        let (ext, asset_type_str) = match export.asset().asset_type() {
            AssetType::ModelWASM => ("wasm", "ModelWASM"),
            AssetType::OperationWASM => ("wasm", "OperationWASM"),
        };

        let file_name = format!("{}.{}", asset_id, ext);
        let file_path = args.output.join(&file_name);
        let bytes = export.asset().bytes();

        std::fs::write(&file_path, bytes)
            .with_context(|| format!("Failed to write {}", file_path.display()))?;

        results.push(ExportedAsset {
            asset_id: asset_id.to_string(),
            asset_type: asset_type_str.to_string(),
            file_path: file_path.display().to_string(),
            size_bytes: bytes.len(),
        });
    }

    if args.json {
        let output = ExportResult { exports: results };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize JSON")?
        );
    } else {
        println!("Exported {} asset(s):", results.len());
        for result in &results {
            println!(
                "  {} ({}) -> {} ({} bytes)",
                result.asset_id, result.asset_type, result.file_path, result.size_bytes
            );
        }
    }

    Ok(())
}

// === Project Run ===

#[derive(Parser, Debug)]
pub struct ProjectRunArgs {
    /// Project file to run
    #[arg(short, long)]
    pub project: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct RunResult {
    success: bool,
    exports: Vec<RunExport>,
}

#[derive(Debug, Serialize)]
struct RunExport {
    asset_id: String,
    asset_type: String,
    size_bytes: usize,
}

pub fn run_project_run(args: ProjectRunArgs) -> Result<()> {
    let project = Project::load_from_file(&args.project).context("Failed to load project")?;

    let mut env = Environment::new();
    let exports = project
        .run(&mut env)
        .map_err(|e| anyhow::anyhow!("Project execution failed: {}", e))?;

    let results: Vec<RunExport> = exports
        .iter()
        .map(|e| RunExport {
            asset_id: e.asset_id().to_string(),
            asset_type: format!("{:?}", e.asset().asset_type()),
            size_bytes: e.asset().bytes().len(),
        })
        .collect();

    if args.json {
        let output = RunResult {
            success: true,
            exports: results,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize JSON")?
        );
    } else {
        println!("Project executed successfully");
        println!("Exports ({}):", results.len());
        for result in &results {
            println!(
                "  {} ({}, {} bytes)",
                result.asset_id, result.asset_type, result.size_bytes
            );
        }
    }

    Ok(())
}

// === Project List Assets ===

#[derive(Parser, Debug)]
pub struct ProjectListArgs {
    /// Project file to inspect
    #[arg(short, long)]
    pub project: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct ListResult {
    declared_assets: Vec<DeclaredAsset>,
    exported_assets: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DeclaredAsset {
    asset_id: String,
    asset_type: String,
}

pub fn run_project_list(args: ProjectListArgs) -> Result<()> {
    let project = Project::load_from_file(&args.project).context("Failed to load project")?;

    let declared: Vec<DeclaredAsset> = project
        .declared_assets()
        .into_iter()
        .map(|(id, ty)| DeclaredAsset {
            asset_id: id,
            asset_type: format!("{:?}", ty),
        })
        .collect();

    let exported: Vec<String> = project
        .entries()
        .iter()
        .filter_map(|e| match e {
            ProjectEntry::ExportAsset(id) => Some(id.clone()),
            _ => None,
        })
        .collect();

    if args.json {
        let output = ListResult {
            declared_assets: declared,
            exported_assets: exported,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize JSON")?
        );
    } else {
        println!("Declared assets ({}):", declared.len());
        for asset in &declared {
            let is_exported = if exported.contains(&asset.asset_id) {
                " [exported]"
            } else {
                ""
            };
            println!("  {} ({}){}", asset.asset_id, asset.asset_type, is_exported);
        }
    }

    Ok(())
}
