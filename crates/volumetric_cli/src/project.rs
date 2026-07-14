//! Project manipulation subcommands for creating and modifying .vproj files.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::path::PathBuf;

use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ImportedAsset, OperatorMetadata,
    OperatorMetadataInput, Project,
};

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

/// A CLI input spec, parsed but not yet checked against the operator's
/// declared input slot type.
enum ParsedInput {
    Asset(String),
    Json(serde_json::Value),
    /// Raw bytes from `file:` or `data:`, with the spec form kept for errors.
    Bytes(Vec<u8>, &'static str),
}

fn parse_input(s: &str) -> Result<ParsedInput> {
    if let Some(rest) = s.strip_prefix("asset:") {
        Ok(ParsedInput::Asset(rest.to_string()))
    } else if let Some(rest) = s.strip_prefix("json:") {
        let json_value: serde_json::Value =
            serde_json::from_str(rest).context("Failed to parse JSON")?;
        Ok(ParsedInput::Json(json_value))
    } else if let Some(rest) = s.strip_prefix("file:") {
        let bytes =
            std::fs::read(rest).with_context(|| format!("Failed to read file: {}", rest))?;
        Ok(ParsedInput::Bytes(bytes, "file:"))
    } else if let Some(rest) = s.strip_prefix("data:") {
        use base64::{Engine, engine::general_purpose::STANDARD};
        let bytes = STANDARD
            .decode(rest)
            .context("Failed to decode base64 data")?;
        Ok(ParsedInput::Bytes(bytes, "data:"))
    } else {
        // Default: treat as asset ID
        Ok(ParsedInput::Asset(s.to_string()))
    }
}

/// Short human label for a declared operator input slot type.
pub fn input_type_label(input: &OperatorMetadataInput) -> String {
    match input {
        OperatorMetadataInput::ModelWASM => "ModelWASM".to_string(),
        OperatorMetadataInput::CBORConfiguration(_) => "CBOR configuration".to_string(),
        OperatorMetadataInput::LuaSource(_) => "Lua source".to_string(),
        OperatorMetadataInput::Blob => "Blob".to_string(),
        OperatorMetadataInput::VecF64(dim) => format!("VecF64({dim})"),
        OperatorMetadataInput::FeaMesh => "FeaMesh".to_string(),
        OperatorMetadataInput::TriMesh => "TriMesh".to_string(),
        OperatorMetadataInput::Subspace => "Subspace".to_string(),
    }
}

/// One line per declared input slot, for count-mismatch errors.
fn describe_declared_inputs(metadata: &OperatorMetadata) -> String {
    metadata
        .inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let name = metadata
                .input_name(i)
                .map(|n| format!("{n} "))
                .unwrap_or_default();
            format!("  [{i}] {name}({})", input_type_label(input))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Check a parsed input against its declared slot type, coercing where the
/// intent is unambiguous (JSON array -> VecF64 raw bytes, JSON string ->
/// Lua source bytes). Asset references always pass — they resolve at run
/// time.
fn coerce_input(
    parsed: ParsedInput,
    slot: &OperatorMetadataInput,
    slot_desc: &str,
) -> Result<ExecutionInput> {
    let inline = |bytes| Ok(ExecutionInput::Inline(bytes));
    match (parsed, slot) {
        (ParsedInput::Asset(id), _) => Ok(ExecutionInput::AssetRef(id)),

        // VecF64: raw little-endian f64s. Accept a JSON array of the right
        // arity, or raw bytes of exactly the right length.
        (
            ParsedInput::Json(serde_json::Value::Array(items)),
            OperatorMetadataInput::VecF64(dim),
        ) => {
            if items.len() != *dim {
                anyhow::bail!(
                    "{slot_desc} expects VecF64({dim}) but the JSON array has {} element(s)",
                    items.len()
                );
            }
            let mut bytes = Vec::with_capacity(dim * 8);
            for (i, item) in items.iter().enumerate() {
                let v = item.as_f64().with_context(|| {
                    format!("{slot_desc}: JSON array element {i} ({item}) is not a number")
                })?;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            inline(bytes)
        }
        (ParsedInput::Json(other), OperatorMetadataInput::VecF64(dim)) => {
            anyhow::bail!(
                "{slot_desc} expects VecF64({dim}): pass json:[x,y,..] with {dim} numbers, \
                 got JSON {other}"
            );
        }
        (ParsedInput::Bytes(bytes, form), OperatorMetadataInput::VecF64(dim)) => {
            if bytes.len() != dim * 8 {
                anyhow::bail!(
                    "{slot_desc} expects VecF64({dim}) = {} raw little-endian bytes, but the \
                     {form} input has {} bytes (tip: json:[x,y,..] also works)",
                    dim * 8,
                    bytes.len()
                );
            }
            inline(bytes)
        }

        // CBOR configuration: JSON converts, raw bytes pass through as
        // pre-encoded CBOR.
        (ParsedInput::Json(value), OperatorMetadataInput::CBORConfiguration(_)) => {
            let mut cbor_bytes = Vec::new();
            ciborium::into_writer(&value, &mut cbor_bytes)
                .context("Failed to convert JSON to CBOR")?;
            inline(cbor_bytes)
        }

        // Lua source: a JSON string is the script text; raw bytes pass.
        (
            ParsedInput::Json(serde_json::Value::String(source)),
            OperatorMetadataInput::LuaSource(_),
        ) => inline(source.into_bytes()),

        // Binary slot types can't be built from JSON literals.
        (ParsedInput::Json(_), slot_type) => {
            anyhow::bail!(
                "{slot_desc} expects {} — pass an asset reference (asset:<id>) or raw bytes \
                 (file:<path> / data:<base64>), not JSON",
                input_type_label(slot_type)
            );
        }

        (ParsedInput::Bytes(bytes, _), _) => inline(bytes),
    }
}

pub fn run_project_add_op(args: ProjectAddOpArgs) -> Result<()> {
    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let (op_name, op_bytes) = resolve_operator_spec(&args.operator)?;

    let metadata = volumetric::operator_metadata_from_wasm_bytes(&op_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to read operator metadata: {e}"))?;
    if args.input.len() != metadata.inputs.len() {
        anyhow::bail!(
            "{op_name} expects {} input(s), got {}:\n{}",
            metadata.inputs.len(),
            args.input.len(),
            describe_declared_inputs(&metadata)
        );
    }

    let inputs: Vec<ExecutionInput> = args
        .input
        .iter()
        .zip(metadata.inputs.iter().enumerate())
        .map(|(spec, (idx, slot))| {
            let name = metadata
                .input_name(idx)
                .map(|n| format!(" ({n})"))
                .unwrap_or_default();
            let slot_desc = format!("input [{idx}]{name}");
            coerce_input(parse_input(spec)?, slot, &slot_desc)
        })
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

/// Runs a project to its exports, locally or — when `remote` names a daemon
/// base URL — on a remote build daemon over `volumetric_protocol`.
fn run_project_exports(
    mut project: Project,
    remote: Option<&str>,
) -> Result<Vec<volumetric::LoadedAsset>> {
    let Some(address) = remote else {
        // A built copy opens hot: its bake seeds the process cache and the
        // run below serves those steps without executing them.
        project.seed_build_cache(volumetric::build_cache::global());
        let mut env = Environment::new();
        return project
            .run(&mut env)
            .map_err(|e| anyhow::anyhow!("Project execution failed: {}", e));
    };

    // The daemon's cache is shared across clients and must not trust
    // client-supplied results; a bake would only bloat the upload.
    project.baked = None;

    let client = volumetric_protocol::DaemonClient::new(address);
    client
        .info()
        .with_context(|| format!("remote daemon at {address} is not usable"))?;
    let outcome = client
        .run(
            &volumetric_protocol::JobRequest::RunProject { project },
            &|| false,
            &|progress| eprintln!("remote: {}", progress.phase),
        )
        .with_context(|| format!("remote run on {address} failed"))?;
    match outcome {
        volumetric_protocol::JobOutcome::Success {
            output: volumetric_protocol::JobOutput::RunProject { exports },
            ..
        } => Ok(exports
            .into_iter()
            .map(volumetric_protocol::ExportedAsset::into_loaded)
            .collect()),
        volumetric_protocol::JobOutcome::Success { .. } => Err(anyhow::anyhow!(
            "daemon returned the wrong output kind for a project run"
        )),
        volumetric_protocol::JobOutcome::Failed { error } => {
            Err(anyhow::anyhow!("Project execution failed: {}", error))
        }
        volumetric_protocol::JobOutcome::Cancelled => {
            Err(anyhow::anyhow!("remote run was cancelled"))
        }
    }
}

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

    /// Execute on a remote build daemon (base URL, e.g. http://buildbox:7373)
    #[arg(long)]
    pub remote: Option<String>,
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
    let exports = run_project_exports(project, args.remote.as_deref())?;

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

    /// Execute on a remote build daemon (base URL, e.g. http://buildbox:7373)
    #[arg(long)]
    pub remote: Option<String>,
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

    let exports = run_project_exports(project, args.remote.as_deref())?;

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

// === Project Bake ===

#[derive(Parser, Debug)]
pub struct ProjectBakeArgs {
    /// Project file to bake
    #[arg(short, long)]
    pub project: PathBuf,

    /// Where to write the built copy (defaults to baking in place)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Builds the project locally and saves a copy with every step result
/// embedded (a built copy): opening it later serves the whole timeline from
/// cache instead of re-executing. An existing bake in the input is reused,
/// so re-baking an already-built copy executes nothing.
pub fn run_project_bake(args: ProjectBakeArgs) -> Result<()> {
    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let cache = volumetric::build_cache::global();

    let seeded = project.seed_build_cache(cache);
    if seeded.corrupt_blobs > 0 {
        eprintln!(
            "warning: dropped {} corrupt blob(s) from the input file's bake",
            seeded.corrupt_blobs
        );
    }
    if seeded.seeded_steps > 0 {
        eprintln!(
            "Reusing {} baked step(s) from the input file",
            seeded.seeded_steps
        );
    }

    // Build whatever the (possibly seeded) cache can't already serve.
    if !project.collect_baked(cache).1.is_complete() {
        let never = std::sync::atomic::AtomicBool::new(false);
        project
            .run_monitored(&mut Environment::new(), &never, &|progress| {
                eprintln!("build: {}", progress.phase)
            })
            .map_err(|e| anyhow::anyhow!("Project execution failed: {}", e))?;
    }

    let (baked, coverage) = project.collect_baked(cache);
    if !coverage.is_complete() {
        eprintln!(
            "warning: only {}/{} steps fit the build cache budget; the rest re-run on open",
            coverage.baked_steps, coverage.total_steps
        );
    }
    let blob_bytes = baked.blob_bytes();
    project.baked = (!baked.is_empty()).then_some(baked);

    let output = args.output.unwrap_or(args.project);
    save_project(&project, &output)?;
    println!(
        "Baked {}/{} step(s), {:.1} MB of results -> {}",
        coverage.baked_steps,
        coverage.total_steps,
        blob_bytes as f64 / (1024.0 * 1024.0),
        output.display()
    );
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

    /// Show each timeline step's inputs with inline values decoded per the
    /// operator's declared slot types (CBOR configs as JSON, VecF64 as
    /// numbers)
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, Serialize)]
struct ListResult {
    version: u32,
    imports: Vec<ImportInfo>,
    timeline_steps: usize,
    exports: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeline: Option<Vec<StepDetail>>,
}

#[derive(Debug, Serialize)]
struct StepDetail {
    operator: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

/// Render one execution input for display. Inline bytes are decoded per the
/// operator's declared slot type when metadata is available.
fn describe_step_input(
    input: &ExecutionInput,
    slot: Option<&OperatorMetadataInput>,
    name: Option<&str>,
) -> String {
    let label = name.map(|n| format!("{n} = ")).unwrap_or_default();
    let body = match input {
        ExecutionInput::AssetRef(id) => format!("asset:{id}"),
        ExecutionInput::Inline(bytes) => match slot {
            Some(OperatorMetadataInput::CBORConfiguration(_)) => {
                match ciborium::from_reader::<ciborium::value::Value, _>(bytes.as_slice())
                    .ok()
                    .and_then(|v| serde_json::to_string(&v).ok())
                {
                    Some(json) if json.len() <= 200 => json,
                    Some(json) => format!("{}… ({} bytes of CBOR)", &json[..200], bytes.len()),
                    None => format!("<{} bytes of CBOR>", bytes.len()),
                }
            }
            Some(OperatorMetadataInput::VecF64(dim)) if bytes.len() == dim * 8 => {
                let values: Vec<String> = bytes
                    .chunks_exact(8)
                    .map(|c| format!("{}", f64::from_le_bytes(c.try_into().unwrap())))
                    .collect();
                format!("[{}]", values.join(", "))
            }
            Some(OperatorMetadataInput::LuaSource(_)) => {
                let first_line = std::str::from_utf8(bytes)
                    .ok()
                    .and_then(|s| s.lines().find(|l| !l.trim().is_empty()))
                    .unwrap_or("<non-utf8>");
                format!("<{} bytes of Lua: {first_line}…>", bytes.len())
            }
            Some(other) => format!("<{} bytes ({})>", bytes.len(), input_type_label(other)),
            None => format!("<{} bytes>", bytes.len()),
        },
    };
    format!("{label}{body}")
}

/// Detailed step listing: decode each step's operator metadata (from the
/// operator asset stored in the project) to label and decode its inputs.
fn describe_steps(project: &Project) -> Vec<StepDetail> {
    project
        .timeline()
        .iter()
        .map(|step| {
            let metadata = project
                .imports()
                .iter()
                .find(|import| import.id == step.operator_id)
                .and_then(|import| {
                    volumetric::operator_metadata_from_wasm_bytes(&import.data).ok()
                });
            let inputs = step
                .inputs
                .iter()
                .enumerate()
                .map(|(idx, input)| {
                    let slot = metadata.as_ref().and_then(|m| m.inputs.get(idx));
                    let name = metadata.as_ref().and_then(|m| m.input_name(idx));
                    describe_step_input(input, slot, name)
                })
                .collect();
            StepDetail {
                operator: step.operator_id.clone(),
                inputs,
                outputs: step.outputs.clone(),
            }
        })
        .collect()
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
    let details = args.verbose.then(|| describe_steps(&project));

    if args.json {
        let output = ListResult {
            version: project.version,
            imports,
            timeline_steps: project.timeline().len(),
            exports,
            timeline: details,
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
        match &details {
            Some(steps) => {
                for (idx, step) in steps.iter().enumerate() {
                    println!("  {}. {} -> {:?}", idx + 1, step.operator, step.outputs);
                    for input in &step.inputs {
                        println!("       {input}");
                    }
                }
            }
            None => {
                for (idx, step) in project.timeline().iter().enumerate() {
                    println!("  {}. {} -> {:?}", idx + 1, step.operator_id, step.outputs);
                }
            }
        }
        println!();
        println!("Exports ({}):", exports.len());
        for id in &exports {
            println!("  {}", id);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3_slot() -> OperatorMetadataInput {
        OperatorMetadataInput::VecF64(3)
    }

    #[test]
    fn json_arrays_coerce_to_vecf64_bytes() {
        let parsed = parse_input("json:[0.25,-0.26,0.25]").unwrap();
        let ExecutionInput::Inline(bytes) =
            coerce_input(parsed, &vec3_slot(), "input [1]").unwrap()
        else {
            panic!("expected inline bytes");
        };
        assert_eq!(bytes.len(), 24);
        let decoded: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, vec![0.25, -0.26, 0.25]);
    }

    #[test]
    fn vecf64_inputs_reject_shape_mismatches() {
        let wrong_arity = parse_input("json:[1,2]").unwrap();
        let err = coerce_input(wrong_arity, &vec3_slot(), "input [1]")
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 element(s)"), "{err}");

        let not_an_array = parse_input("json:{\"x\":1}").unwrap();
        assert!(coerce_input(not_an_array, &vec3_slot(), "input [1]").is_err());

        let wrong_len = ParsedInput::Bytes(vec![0u8; 23], "data:");
        let err = coerce_input(wrong_len, &vec3_slot(), "input [1]")
            .unwrap_err()
            .to_string();
        assert!(err.contains("23 bytes"), "{err}");
    }

    #[test]
    fn json_is_rejected_for_binary_slots() {
        let parsed = parse_input("json:{\"op\":\"union\"}").unwrap();
        let err = coerce_input(parsed, &OperatorMetadataInput::ModelWASM, "input [0]")
            .unwrap_err()
            .to_string();
        assert!(err.contains("ModelWASM"), "{err}");
    }

    #[test]
    fn config_and_lua_slots_accept_json() {
        let config = parse_input("json:{\"op\":\"intersect\"}").unwrap();
        let ExecutionInput::Inline(cbor) = coerce_input(
            config,
            &OperatorMetadataInput::CBORConfiguration(String::new()),
            "c",
        )
        .unwrap() else {
            panic!("expected inline");
        };
        let value: ciborium::value::Value = ciborium::from_reader(cbor.as_slice()).unwrap();
        assert!(format!("{value:?}").contains("intersect"));

        let lua = parse_input("json:\"return 1\"").unwrap();
        let ExecutionInput::Inline(source) =
            coerce_input(lua, &OperatorMetadataInput::LuaSource(String::new()), "l").unwrap()
        else {
            panic!("expected inline");
        };
        assert_eq!(source, b"return 1");
    }
}
