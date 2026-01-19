//! Project manipulation subcommands for creating and modifying .vproj files.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

use volumetric::{AssetTypeHint, Environment, ExecutionInput, Project};

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

    let project = Project::from_model(asset_id.clone(), wasm_bytes);
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

    let asset_id = project.insert_model(&asset_id_base, wasm_bytes);

    // Remove the auto-added export if --no-export was specified
    if args.no_export {
        project.exports_mut().retain(|id| id != &asset_id);
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

    /// Input asset IDs or literal values (format: "asset:id", "json:{...}", or "data:base64")
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

fn parse_input(s: &str) -> Result<ExecutionInput> {
    if let Some(rest) = s.strip_prefix("asset:") {
        Ok(ExecutionInput::AssetRef(rest.to_string()))
    } else if let Some(rest) = s.strip_prefix("json:") {
        // Parse JSON and convert to CBOR
        let json_value: serde_json::Value =
            serde_json::from_str(rest).context("Failed to parse JSON")?;
        let mut cbor_bytes = Vec::new();
        ciborium::into_writer(&json_value, &mut cbor_bytes)
            .context("Failed to convert JSON to CBOR")?;
        Ok(ExecutionInput::Inline(cbor_bytes))
    } else if let Some(rest) = s.strip_prefix("data:") {
        // Assume base64 encoded data
        use base64::{engine::general_purpose::STANDARD, Engine};
        let bytes = STANDARD
            .decode(rest)
            .context("Failed to decode base64 data")?;
        Ok(ExecutionInput::Inline(bytes))
    } else {
        // Default: treat as asset ID
        Ok(ExecutionInput::AssetRef(s.to_string()))
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

    let inputs: Vec<ExecutionInput> = args
        .input
        .iter()
        .map(|s| parse_input(s))
        .collect::<Result<_>>()?;

    // Get primary input for naming
    let primary_input = inputs.iter().find_map(|i| match i {
        ExecutionInput::AssetRef(id) => Some(id.as_str()),
        _ => None,
    });

    let output_id = args
        .output_id
        .unwrap_or_else(|| project.default_output_name(&op_name, primary_input));

    // Output IDs for the execution step (just the output ID, no types)
    let output_ids = vec![output_id.clone()];

    project.insert_operation(&op_name, op_bytes, inputs, output_ids, output_id.clone());

    // Remove the auto-added export if --no-export was specified
    if args.no_export {
        project.exports_mut().retain(|id| id != &output_id);
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
    type_hint: String,
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
            .filter(|e| args.asset.contains(&e.id().to_string()))
            .collect()
    };

    let mut results = Vec::new();

    for export in filtered_exports {
        let asset_id = export.id();
        let type_hint = export.type_hint().unwrap_or(AssetTypeHint::Binary);
        let ext = match type_hint {
            AssetTypeHint::Model | AssetTypeHint::Operator => "wasm",
            AssetTypeHint::LuaSource => "lua",
            _ => "bin",
        };

        let file_name = format!("{}.{}", asset_id, ext);
        let file_path = args.output.join(&file_name);
        let bytes = export.data();

        std::fs::write(&file_path, bytes)
            .with_context(|| format!("Failed to write {}", file_path.display()))?;

        results.push(ExportedAsset {
            asset_id: asset_id.to_string(),
            type_hint: type_hint.to_string(),
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
                result.asset_id, result.type_hint, result.file_path, result.size_bytes
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
    type_hint: String,
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
            asset_id: e.id().to_string(),
            type_hint: e
                .type_hint()
                .map(|h| h.to_string())
                .unwrap_or_else(|| "Binary".to_string()),
            size_bytes: e.data().len(),
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
                result.asset_id, result.type_hint, result.size_bytes
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
    version: u32,
    imports: Vec<ImportInfo>,
    timeline_steps: usize,
    exports: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ImportInfo {
    id: String,
    type_hint: String,
    size_bytes: usize,
}

pub fn run_project_list(args: ProjectListArgs) -> Result<()> {
    let project = Project::load_from_file(&args.project).context("Failed to load project")?;

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

    let exports = project.exports().to_vec();

    if args.json {
        let output = ListResult {
            version: project.version,
            imports,
            timeline_steps: project.timeline().len(),
            exports,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize JSON")?
        );
    } else {
        println!("Project version: {}", project.version);
        println!();
        println!("Imports ({}):", imports.len());
        for import in &imports {
            println!("  {} ({}, {} bytes)", import.id, import.type_hint, import.size_bytes);
        }
        println!();
        println!("Timeline steps: {}", project.timeline().len());
        for (idx, step) in project.timeline().iter().enumerate() {
            println!(
                "  {}. {} -> {:?}",
                idx + 1,
                step.operator_id,
                step.outputs
            );
        }
        println!();
        println!("Exports ({}):", exports.len());
        for id in &exports {
            println!("  {}", id);
        }
    }

    Ok(())
}
