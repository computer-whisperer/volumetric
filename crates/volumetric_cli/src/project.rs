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

pub(crate) fn save_project(project: &Project, path: &std::path::Path) -> Result<()> {
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
    Wgsl,
    Config,
    F64Map,
    Blob,
}

impl From<AssetTypeArg> for AssetTypeHint {
    fn from(value: AssetTypeArg) -> Self {
        match value {
            AssetTypeArg::Lua => AssetTypeHint::LuaSource,
            AssetTypeArg::Wgsl => AssetTypeHint::WgslSource,
            AssetTypeArg::Config => AssetTypeHint::Config,
            AssetTypeArg::F64Map => AssetTypeHint::F64Map,
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
    let mut bytes = std::fs::read(&args.input)
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
            "wgsl" => AssetTypeArg::Wgsl,
            "cbor" => AssetTypeArg::Config,
            _ => AssetTypeArg::Blob,
        })
        .into();

    if type_hint == AssetTypeHint::F64Map {
        if extension == "json" {
            let json: serde_json::Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("Failed to parse {} as JSON", args.input.display()))?;
            bytes = encode_json_f64_map(&json, "F64Map asset")?;
        } else {
            volumetric_abi::f64_map::decode(&bytes)
                .map_err(anyhow::Error::msg)
                .context("Invalid F64Map asset")?;
        }
    }

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

    /// One per declared input slot: "asset:id", "json:{...}", "file:path",
    /// "data:base64", or "none" to leave an optional slot unwired. Repeat
    /// for a slot that accepts one or more inputs (see `info`)
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
    /// `none`: the slot stays unwired (empty inline bytes, which is what
    /// the GUI stores for a slot it has nothing to wire and what operators
    /// with optional inputs test for). Any slot type accepts it; a slot the
    /// operator requires fails at run time with the operator's message.
    Unwired,
}

/// The spelling that leaves a slot unwired. A bare word, so an asset that
/// happens to be called `none` needs the `asset:` prefix.
const UNWIRED_SPEC: &str = "none";

fn parse_input(s: &str) -> Result<ParsedInput> {
    if s == UNWIRED_SPEC {
        Ok(ParsedInput::Unwired)
    } else if let Some(rest) = s.strip_prefix("asset:") {
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
        OperatorMetadataInput::WgslSource(_) => "WGSL source".to_string(),
        OperatorMetadataInput::F64Map => "F64Map".to_string(),
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
            let arity = if metadata.variadic_slot() == Some(i) {
                ", one or more"
            } else {
                ""
            };
            format!("  [{i}] {name}({}{arity})", input_type_label(input))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// "3" or "at least 3": how many inputs the operator's declaration takes.
fn expected_input_count(metadata: &OperatorMetadata) -> String {
    match metadata.variadic_slot() {
        Some(_) => format!("at least {}", metadata.inputs.len()),
        None => metadata.inputs.len().to_string(),
    }
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
        (ParsedInput::Unwired, _) => inline(Vec::new()),

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

        (ParsedInput::Json(value), OperatorMetadataInput::F64Map) => {
            inline(encode_json_f64_map(&value, slot_desc)?)
        }

        // Lua source: a JSON string is the script text; raw bytes pass.
        (
            ParsedInput::Json(serde_json::Value::String(source)),
            OperatorMetadataInput::LuaSource(_),
        ) => inline(source.into_bytes()),

        (ParsedInput::Bytes(bytes, _), OperatorMetadataInput::F64Map) => {
            volumetric_abi::f64_map::decode(&bytes)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("{slot_desc} is not a valid F64Map"))?;
            inline(bytes)
        }

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

fn encode_json_f64_map(value: &serde_json::Value, context: &str) -> Result<Vec<u8>> {
    let serde_json::Value::Object(entries) = value else {
        anyhow::bail!("{context} expects a JSON object whose values are finite numbers");
    };
    let mut values = volumetric_abi::f64_map::F64Map::new();
    for (key, value) in entries {
        let number = value
            .as_f64()
            .with_context(|| format!("{context}: value for `{key}` is not a number"))?;
        values.insert(key.clone(), number);
    }
    volumetric_abi::f64_map::encode(&values)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("{context} is invalid"))
}

pub fn run_project_add_op(args: ProjectAddOpArgs) -> Result<()> {
    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let (op_name, op_bytes) = resolve_operator_spec(&args.operator)?;

    let metadata = volumetric::operator_metadata_from_wasm_bytes(&op_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to read operator metadata: {e}"))?;
    let count = args.input.len();
    if !metadata.accepts_input_count(count) {
        anyhow::bail!(
            "{op_name} expects {} input(s), got {count}:\n{}\n(pass `{UNWIRED_SPEC}` for an \
             optional slot you want to leave unwired)",
            expected_input_count(&metadata),
            describe_declared_inputs(&metadata)
        );
    }

    let inputs: Vec<ExecutionInput> = args
        .input
        .iter()
        .enumerate()
        .map(|(idx, spec)| {
            let slot = metadata
                .input_type(idx, count)
                .expect("input count was checked against the declaration");
            let name = metadata
                .input_label(idx, count)
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

    // One output id per declared output: --output-id names slot 0, the
    // rest get its declared-name suffixes (e.g. `card`, `card_plate`).
    let output_ids = project.output_ids_for(output_id, &metadata);

    let import_id = project.insert_operation(&op_name, op_bytes, inputs, output_ids.clone());

    // Remove the auto-added exports if --no-export was specified
    if args.no_export {
        project.exports_mut().retain(|id| !output_ids.contains(id));
    }

    let output_path = args.output.unwrap_or(args.project);
    save_project(&project, &output_path)?;

    println!(
        "Added operator '{}' (import '{}') with output{} '{}'",
        op_name,
        import_id,
        if output_ids.len() > 1 { "s" } else { "" },
        output_ids.join("', '")
    );
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

/// Prints a step's non-fatal warnings to stderr, attributed to the asset
/// that carries them. Shared by every path that surfaces a built asset.
pub(crate) fn print_asset_warnings(asset: &volumetric::LoadedAsset) {
    for warning in asset.warnings() {
        eprintln!("warning [{}]: {}", asset.id(), warning);
    }
}

/// Runs a project to its exports, locally or — when `remote` names a daemon
/// base URL — on a remote build daemon over `volumetric_protocol`.
pub(crate) fn run_project_exports(
    mut project: Project,
    remote: Option<&str>,
) -> Result<Vec<volumetric::LoadedAsset>> {
    let Some(address) = remote else {
        // A built copy opens hot: its bake seeds the process cache and the
        // run below serves those steps without executing them.
        project.seed_build_cache(volumetric::build_cache::global());
        let mut env = Environment::new();
        let never = std::sync::atomic::AtomicBool::new(false);
        return project
            .run_monitored_with_artifacts(&mut env, &never, &|_| {}, &print_asset_warnings)
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
            .inspect(print_asset_warnings)
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
            AssetTypeHint::WgslSource => "wgsl",
            AssetTypeHint::F64Map | AssetTypeHint::Config => "cbor",
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
            .run_monitored_with_artifacts(
                &mut Environment::new(),
                &never,
                &|progress| eprintln!("build: {}", progress.phase),
                &print_asset_warnings,
            )
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
        ExecutionInput::Inline(bytes) if bytes.is_empty() => {
            format!("{UNWIRED_SPEC} (unwired)")
        }
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
            Some(OperatorMetadataInput::F64Map) => {
                match volumetric_abi::f64_map::decode(bytes)
                    .ok()
                    .and_then(|map| serde_json::to_string(&map).ok())
                {
                    Some(json) if json.len() <= 200 => json,
                    Some(json) => format!("{}… ({} bytes of CBOR)", &json[..200], bytes.len()),
                    None => format!("<{} bytes of invalid F64Map>", bytes.len()),
                }
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
            let count = step.inputs.len();
            let inputs = step
                .inputs
                .iter()
                .enumerate()
                .map(|(idx, input)| {
                    let slot = metadata.as_ref().and_then(|m| m.input_type(idx, count));
                    let name = metadata.as_ref().and_then(|m| m.input_label(idx, count));
                    describe_step_input(input, slot, name.as_deref())
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

// === Project Set Config ===

#[derive(Parser, Debug)]
pub struct ProjectSetConfigArgs {
    /// Project file to modify
    #[arg(short, long)]
    pub project: PathBuf,

    /// Step selector: a 0-based timeline index, or a substring of the
    /// step's operator id matching exactly one step
    #[arg(long)]
    pub step: String,

    /// JSON object of config fields to merge into the step's configuration,
    /// e.g. '{"preconditioner":"schwarz","cg_tolerance":1e-4}'. Values are
    /// checked against the operator's declared schema (integers are accepted
    /// for float fields); group sub-fields use dotted paths.
    pub config: String,

    /// Output project file (defaults to overwriting input)
    #[arg(short = 'O', long)]
    pub output: Option<PathBuf>,
}

/// Coerce a JSON literal to a schema-typed config value. Integers promote to
/// floats for `Float` fields — the raw JSON→CBOR path can't do this (it has
/// no schema), and operators reject CBOR ints in f64 fields.
fn json_config_value(
    field: &volumetric::operator_config::ConfigField,
    value: &serde_json::Value,
    path: &str,
) -> Result<volumetric::operator_config::ConfigValue> {
    use volumetric::operator_config::{ConfigFieldType, ConfigValue};

    fn scalar(ty: &ConfigFieldType, value: &serde_json::Value, path: &str) -> Result<ConfigValue> {
        match ty {
            ConfigFieldType::Bool => value
                .as_bool()
                .map(ConfigValue::Bool)
                .with_context(|| format!("{path}: expected a bool, got {value}")),
            ConfigFieldType::Int => value
                .as_i64()
                .map(ConfigValue::Int)
                .with_context(|| format!("{path}: expected an integer, got {value}")),
            ConfigFieldType::Float => value
                .as_f64()
                .map(ConfigValue::Float)
                .with_context(|| format!("{path}: expected a number, got {value}")),
            ConfigFieldType::Text => value
                .as_str()
                .map(|s| ConfigValue::Text(s.to_string()))
                .with_context(|| format!("{path}: expected a string, got {value}")),
            ConfigFieldType::Enum(options) => {
                let s = value
                    .as_str()
                    .with_context(|| format!("{path}: expected a string, got {value}"))?;
                anyhow::ensure!(
                    options.iter().any(|o| o == s),
                    "{path}: {s:?} is not one of {}",
                    options.join("/")
                );
                Ok(ConfigValue::Text(s.to_string()))
            }
            ConfigFieldType::List { element, min_len } => {
                let items = value
                    .as_array()
                    .with_context(|| format!("{path}: expected an array, got {value}"))?;
                anyhow::ensure!(
                    items.len() >= *min_len,
                    "{path}: needs at least {min_len} element(s), got {}",
                    items.len()
                );
                items
                    .iter()
                    .map(|item| scalar(element, item, path))
                    .collect::<Result<Vec<_>>>()
                    .map(ConfigValue::List)
            }
            ConfigFieldType::Group(_) => {
                // A bool toggles an optional group's enablement marker;
                // sub-fields are set through their dotted paths.
                value.as_bool().map(ConfigValue::Bool).with_context(|| {
                    format!(
                        "{path} is a config group: pass true/false to toggle it, \
                         or set sub-fields via dotted paths ({path}.<field>)"
                    )
                })
            }
        }
    }

    let value = scalar(&field.ty, value, path)?;
    anyhow::ensure!(
        field.in_bounds(&value),
        "{path}: {value:?} is outside the declared bounds [{}, {}]",
        field.min.map_or("-inf".into(), |v| v.to_string()),
        field.max.map_or("+inf".into(), |v| v.to_string()),
    );
    Ok(value)
}

/// Select a timeline step by 0-based index or operator-id substring (which
/// must match exactly one step).
pub(crate) fn select_step(project: &Project, selector: &str) -> Result<usize> {
    if let Ok(idx) = selector.parse::<usize>() {
        anyhow::ensure!(
            idx < project.timeline.len(),
            "step index {idx} out of range; the timeline has {} step(s)",
            project.timeline.len()
        );
        return Ok(idx);
    }
    let matches: Vec<usize> = project
        .timeline
        .iter()
        .enumerate()
        .filter(|(_, s)| s.operator_id.contains(selector))
        .map(|(i, _)| i)
        .collect();
    match matches.as_slice() {
        [only] => Ok(*only),
        [] => anyhow::bail!(
            "no timeline step's operator id contains {selector:?}; steps: {}",
            project
                .timeline
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{i}:{}", s.operator_id))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        many => anyhow::bail!(
            "{selector:?} matches {} steps ({}); use an index",
            many.len(),
            many.iter()
                .map(|i| format!("{i}:{}", project.timeline[*i].operator_id))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub fn run_project_set_config(args: ProjectSetConfigArgs) -> Result<()> {
    use volumetric::operator_config;

    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let step_index = select_step(&project, &args.step)?;

    let operator_id = project.timeline[step_index].operator_id.clone();
    let op_bytes = project
        .imports
        .iter()
        .find(|a| a.id == operator_id)
        .map(|a| a.data.clone())
        .with_context(|| format!("operator asset {operator_id:?} not found in project imports"))?;
    let metadata = volumetric::operator_metadata_from_wasm_bytes(&op_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to read operator metadata: {e}"))?;

    let (slot, cddl) = metadata
        .inputs
        .iter()
        .enumerate()
        .find_map(|(i, input)| match input {
            OperatorMetadataInput::CBORConfiguration(schema) => Some((i, schema.clone())),
            _ => None,
        })
        .with_context(|| format!("{operator_id} declares no configuration input"))?;
    let fields = operator_config::parse_schema(&cddl)
        .map_err(|e| anyhow::anyhow!("{operator_id}'s config schema failed to parse: {e}"))?;

    let step = &mut project.timeline[step_index];
    anyhow::ensure!(
        metadata.accepts_input_count(step.inputs.len()),
        "step {step_index} has {} input(s) but {operator_id} declares {}; \
         the project predates the operator version — re-add the step first",
        step.inputs.len(),
        expected_input_count(&metadata)
    );
    let slot = metadata.input_of_slot(slot, step.inputs.len());
    let mut values = match &step.inputs[slot] {
        ExecutionInput::Inline(bytes) if !bytes.is_empty() => operator_config::decode(bytes),
        _ => operator_config::default_values(&fields),
    };

    let updates: serde_json::Value =
        serde_json::from_str(&args.config).context("config is not valid JSON")?;
    let serde_json::Value::Object(entries) = updates else {
        anyhow::bail!("config must be a JSON object of field: value pairs");
    };
    anyhow::ensure!(
        !entries.is_empty(),
        "config object is empty; nothing to set"
    );

    for (path, json_value) in &entries {
        let field = operator_config::find_field(&fields, path).with_context(|| {
            format!(
                "{operator_id} has no config field {path:?}; fields: {}",
                fields
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let value = json_config_value(field, json_value, path)?;
        let previous = values.insert(path.clone(), value.clone());
        println!(
            "{path}: {} -> {value:?}",
            previous.map_or("(unset)".to_string(), |p| format!("{p:?}"))
        );
    }

    step.inputs[slot] = ExecutionInput::Inline(operator_config::encode(&fields, &values));

    let output = args.output.unwrap_or(args.project);
    save_project(&project, &output)?;
    println!(
        "Updated step {step_index} ({operator_id}) config in {}",
        output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec3_slot() -> OperatorMetadataInput {
        OperatorMetadataInput::VecF64(3)
    }

    /// Count-mismatch errors name the variadic slot and the "at least"
    /// arity, so the hint matches what `--input` repetition means.
    #[test]
    fn variadic_declarations_describe_their_arity() {
        let metadata = OperatorMetadata {
            name: "nary".to_string(),
            version: "0.0.0".to_string(),
            display_name: String::new(),
            description: String::new(),
            category: String::new(),
            icon_svg: String::new(),
            docs: String::new(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration("{ op: tstr }".to_string()),
            ],
            variadic_input: Some(0),
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![],
            output_names: vec![],
        };
        assert_eq!(expected_input_count(&metadata), "at least 2");
        let described = describe_declared_inputs(&metadata);
        assert!(described.contains("[0] Model (ModelWASM, one or more)"), "{described}");
        assert!(described.contains("[1] Config (CBOR configuration)"), "{described}");
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
    fn none_leaves_any_slot_unwired() {
        for slot in [
            OperatorMetadataInput::Subspace,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::Blob,
            OperatorMetadataInput::CBORConfiguration(String::new()),
        ] {
            let parsed = parse_input("none").unwrap();
            let input = coerce_input(parsed, &slot, "input [1]").unwrap();
            assert!(
                matches!(&input, ExecutionInput::Inline(bytes) if bytes.is_empty()),
                "{slot:?} -> {input:?}"
            );
        }
        // The bare word is reserved; an asset called `none` takes the prefix.
        let parsed = parse_input("asset:none").unwrap();
        assert!(matches!(parsed, ParsedInput::Asset(ref id) if id == "none"));
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

    #[test]
    fn f64_map_slots_accept_numeric_json_objects() {
        let parsed =
            parse_input("json:{\"spinner.bearing_pitch\":0.04,\"global.scale\":2}").unwrap();
        let ExecutionInput::Inline(bytes) =
            coerce_input(parsed, &OperatorMetadataInput::F64Map, "parameters").unwrap()
        else {
            panic!("expected inline F64Map");
        };
        let values = volumetric_abi::f64_map::decode(&bytes).unwrap();
        assert_eq!(values["spinner.bearing_pitch"], 0.04);
        assert_eq!(values["global.scale"], 2.0);

        let invalid = parse_input("json:{\"x\":\"not numeric\"}").unwrap();
        assert!(coerce_input(invalid, &OperatorMetadataInput::F64Map, "parameters").is_err());
    }
}
