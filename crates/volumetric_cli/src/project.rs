//! Project manipulation subcommands for creating and modifying .vproj files.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::path::PathBuf;

use volumetric::{AssetTypeHint, Environment, ExecutionInput, ImportedAsset, Project};

use crate::assets::{resolve_model_spec, resolve_operator_spec};

/// Print every structural problem `Project::validate` finds as a warning.
///
/// The project is saved regardless — a half-built pipeline (e.g. an operator
/// added before its config asset) is legitimate intermediate state — but the
/// user should hear about it at edit time, not at run time.
fn warn_validation_issues(project: &Project) {
    for issue in project.validate() {
        eprintln!("warning: {issue}");
    }
}

fn save_project(project: &Project, path: &PathBuf) -> Result<()> {
    warn_validation_issues(project);
    project.save_to_file(path).context("Failed to save project")
}

// === Project New ===

#[derive(Parser, Debug)]
pub struct ProjectNewArgs {
    /// Model to seed the project with: a .wasm path or a bundled model name.
    /// Omit to create an empty project (e.g. one that starts with a sketch).
    #[arg(short, long)]
    pub input: Option<String>,

    /// Output .vproj file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Asset ID for the model (defaults to filename without extension)
    #[arg(long)]
    pub asset_id: Option<String>,
}

pub fn run_project_new(args: ProjectNewArgs) -> Result<()> {
    let project = match &args.input {
        Some(spec) => {
            let (name, wasm_bytes) = resolve_model_spec(spec)?;
            let asset_id = args.asset_id.unwrap_or(name);
            let project = Project::from_model(asset_id.clone(), wasm_bytes);
            println!("Created project with model '{}'", asset_id);
            project
        }
        None => {
            if args.asset_id.is_some() {
                anyhow::bail!("--asset-id requires --input");
            }
            println!("Created empty project");
            Project::new()
        }
    };

    save_project(&project, &args.output)?;
    println!("Saved to {:?}", args.output);
    Ok(())
}

// === Project Add Model ===

#[derive(Parser, Debug)]
pub struct ProjectAddModelArgs {
    /// Project file to modify
    #[arg(short, long)]
    pub project: PathBuf,

    /// Model to add: a .wasm path or a bundled model name
    #[arg(short, long)]
    pub input: String,

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
    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let (name, wasm_bytes) = resolve_model_spec(&args.input)?;

    let asset_id_base = args.asset_id.unwrap_or(name);
    let asset_id = project.insert_model(&asset_id_base, wasm_bytes);

    // Remove the auto-added export if --no-export was specified
    if args.no_export {
        project.exports_mut().retain(|id| id != &asset_id);
    }

    let output_path = args.output.unwrap_or(args.project);
    save_project(&project, &output_path)?;

    println!("Added model '{}' to project", asset_id);
    println!("Saved to {:?}", output_path);
    Ok(())
}

// === Project Add Asset (non-model imports) ===

/// Type hint for `project-add-asset`.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum AssetTypeArg {
    Lua,
    Config,
    Blob,
}

impl From<AssetTypeArg> for AssetTypeHint {
    fn from(value: AssetTypeArg) -> Self {
        match value {
            AssetTypeArg::Lua => AssetTypeHint::LuaSource,
            AssetTypeArg::Config => AssetTypeHint::Config,
            AssetTypeArg::Blob => AssetTypeHint::Binary,
        }
    }
}

#[derive(Parser, Debug)]
pub struct ProjectAddAssetArgs {
    /// Project file to modify
    #[arg(short, long)]
    pub project: PathBuf,

    /// File whose bytes become the asset (e.g. a .lua sketch source)
    #[arg(short, long)]
    pub input: PathBuf,

    /// Asset type (defaults from the file extension: .lua -> lua, else blob)
    #[arg(long, value_enum)]
    pub r#type: Option<AssetTypeArg>,

    /// Asset ID (defaults to filename without extension)
    #[arg(long)]
    pub asset_id: Option<String>,

    /// Output project file (defaults to overwriting input)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub fn run_project_add_asset(args: ProjectAddAssetArgs) -> Result<()> {
    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let bytes = std::fs::read(&args.input)
        .with_context(|| format!("Failed to read {}", args.input.display()))?;

    let extension = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let type_hint: AssetTypeHint = args
        .r#type
        .unwrap_or(match extension.as_str() {
            "lua" => AssetTypeArg::Lua,
            "cbor" => AssetTypeArg::Config,
            _ => AssetTypeArg::Blob,
        })
        .into();

    // Adding a wasm module as lua/blob is almost certainly a mistyped command.
    if bytes.starts_with(b"\0asm") {
        anyhow::bail!(
            "{} is a WASM module; use project-add-model (or project-add-op for operators)",
            args.input.display()
        );
    }

    let asset_id_base = args.asset_id.unwrap_or_else(|| {
        args.input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("asset")
            .to_string()
    });
    let asset_id = project.unique_asset_id(&asset_id_base);
    project
        .imports_mut()
        .push(ImportedAsset::new(asset_id.clone(), bytes, Some(type_hint)));

    let output_path = args.output.unwrap_or(args.project);
    save_project(&project, &output_path)?;

    println!("Added {} asset '{}' to project", type_hint, asset_id);
    println!("Saved to {:?}", output_path);
    Ok(())
}

// === Project Add Operator ===

#[derive(Parser, Debug)]
pub struct ProjectAddOpArgs {
    /// Project file to modify
    #[arg(short, long)]
    pub project: PathBuf,

    /// Operator: a .wasm path or a bundled operator name (see `assets`)
    #[arg(long)]
    pub operator: String,

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
    } else if let Some(rest) = s.strip_prefix("file:") {
        // Read raw bytes from file
        let bytes =
            std::fs::read(rest).with_context(|| format!("Failed to read file: {}", rest))?;
        Ok(ExecutionInput::Inline(bytes))
    } else if let Some(rest) = s.strip_prefix("data:") {
        // Assume base64 encoded data
        use base64::{Engine, engine::general_purpose::STANDARD};
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
    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let (op_name, op_bytes) = resolve_operator_spec(&args.operator)?;

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
    save_project(&project, &output_path)?;

    println!("Added operator '{}' with output '{}'", op_name, output_id);
    println!("Saved to {:?}", output_path);
    Ok(())
}

// === Project Validate ===

#[derive(Parser, Debug)]
pub struct ProjectValidateArgs {
    /// Project file to check
    #[arg(short, long)]
    pub project: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct ValidateResult {
    valid: bool,
    issues: Vec<String>,
}

pub fn run_project_validate(args: ProjectValidateArgs) -> Result<()> {
    let project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let issues: Vec<String> = project
        .validate()
        .into_iter()
        .map(|i| i.to_string())
        .collect();

    if args.json {
        let output = ValidateResult {
            valid: issues.is_empty(),
            issues: issues.clone(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize JSON")?
        );
    } else if issues.is_empty() {
        println!("Project is structurally sound");
    } else {
        println!("Found {} issue(s):", issues.len());
        for issue in &issues {
            println!("  {issue}");
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        // Non-zero exit so scripts can gate on validity
        std::process::exit(1);
    }
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
            AssetTypeHint::FeaMesh => "vfea",
            AssetTypeHint::TriMesh => "vmesh",
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
            println!(
                "  {} ({}, {} bytes)",
                import.id, import.type_hint, import.size_bytes
            );
        }
        println!();
        println!("Timeline steps: {}", project.timeline().len());
        for (idx, step) in project.timeline().iter().enumerate() {
            println!("  {}. {} -> {:?}", idx + 1, step.operator_id, step.outputs);
        }
        println!();
        println!("Exports ({}):", exports.len());
        for id in &exports {
            println!("  {}", id);
        }
    }

    Ok(())
}
