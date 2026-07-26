//! FEA externalization subcommands: export a solve step's inputs as a
//! problem bundle for non-portable compute backends, and adopt the returned
//! solution back into the project (see the `fea_bundle` crate for the
//! format and the trust model).

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

use volumetric::{
    AssetTypeHint, ExecutionInput, ExecutionStep, ImportedAsset, OperatorMetadataInput, Project,
    operator_config,
};

use crate::assets::resolve_operator_spec;
use crate::project::{run_project_exports, save_project, select_step};

/// Hex-encode a blake3 digest for display.
fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

// === fea-export ===

#[derive(Parser, Debug)]
pub struct FeaExportArgs {
    /// Project file to export from
    #[arg(short, long)]
    pub project: PathBuf,

    /// Step selector: a 0-based timeline index, or an operator-id substring
    /// matching exactly one step
    #[arg(long, default_value = "fea_inverse")]
    pub step: String,

    /// Output problem bundle (defaults to `<project>.fea-problem.cbor`)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub fn run_fea_export(args: FeaExportArgs) -> Result<()> {
    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;
    let step_index = select_step(&project, &args.step)?;
    let step = project.timeline[step_index].clone();

    anyhow::ensure!(
        step.operator_id.contains("fea_inverse"),
        "step {step_index} runs {}; fea-export currently understands \
         fea_inverse steps",
        step.operator_id
    );

    // The inverse step's shape: [mesh, rigid body, target map, config].
    let mut asset_ids = Vec::new();
    let mut config = Vec::new();
    for input in &step.inputs {
        match input {
            ExecutionInput::AssetRef(id) => asset_ids.push(id.clone()),
            ExecutionInput::Inline(bytes) => config = bytes.clone(),
        }
    }
    let [mesh_id, rigid_id, target_id] = asset_ids.as_slice() else {
        anyhow::bail!(
            "expected the step to reference 3 assets (mesh, rigid body, \
             target map), found {}",
            asset_ids.len()
        );
    };
    let (mesh_id, rigid_id, target_id) = (mesh_id.clone(), rigid_id.clone(), target_id.clone());

    // Run the precursor timeline so the mesh (and any derived models) exist,
    // then collect the three inputs.
    project.timeline.truncate(step_index);
    project.exports = vec![mesh_id.clone(), rigid_id.clone(), target_id.clone()];
    let assets = run_project_exports(project, None)?;
    let bytes_of = |id: &str| -> Result<Vec<u8>> {
        assets
            .iter()
            .find(|a| a.id() == id)
            .map(|a| a.data().to_vec())
            .with_context(|| format!("running the precursor steps produced no asset {id:?}"))
    };

    let problem = fea_bundle::FeaProblem {
        version: fea_bundle::PROBLEM_VERSION,
        kind: fea_bundle::ProblemKind::Inverse,
        mesh: bytes_of(&mesh_id)?,
        load_cases: vec![fea_bundle::LoadCase {
            name: "seating".to_string(),
            rigid_model: bytes_of(&rigid_id)?,
            target_map: bytes_of(&target_id)?,
        }],
        config,
        source: Some(format!("{} step {step_index}", args.project.display())),
    };
    let bytes = fea_bundle::encode_problem(&problem).map_err(anyhow::Error::msg)?;

    let output = args.output.unwrap_or_else(|| {
        let mut path = args.project.clone();
        path.set_extension("fea-problem.cbor");
        path
    });
    std::fs::write(&output, &bytes)
        .with_context(|| format!("Failed to write {}", output.display()))?;

    let digest = volumetric::content_fingerprint(&bytes);
    println!(
        "Exported {} step {step_index} -> {} ({:.1} MiB, 1 load case)",
        args.project.display(),
        output.display(),
        bytes.len() as f64 / (1024.0 * 1024.0),
    );
    println!("problem blake3: {}", hex(&digest));
    println!("solve it with: fea-solve {}", output.display());
    Ok(())
}

// === fea-import ===

/// The import operator's asset id inside a project, and the bundled
/// operator resolved for it.
const IMPORT_OPERATOR_ID: &str = "fea_solution_import_operator";
/// The solution bundle's asset id inside a project.
const SOLUTION_ASSET_ID: &str = "fea_solution";

#[derive(Parser, Debug)]
pub struct FeaImportArgs {
    /// Project file to modify
    #[arg(short, long)]
    pub project: PathBuf,

    /// Solution bundle (from `fea-solve`)
    pub solution: PathBuf,

    /// Step selector: the fea_inverse step to replace (or an existing
    /// fea_solution_import step to refresh)
    #[arg(long, default_value = "fea_inverse")]
    pub step: String,

    /// Output project file (defaults to overwriting input)
    #[arg(short = 'O', long)]
    pub output: Option<PathBuf>,
}

/// Insert-or-replace an imported asset's bytes.
fn upsert_import(project: &mut Project, asset: ImportedAsset) {
    match project.imports.iter_mut().find(|a| a.id == asset.id) {
        Some(existing) => *existing = asset,
        None => project.imports.push(asset),
    }
}

pub fn run_fea_import(args: FeaImportArgs) -> Result<()> {
    let mut project = Project::load_from_file(&args.project).context("Failed to load project")?;

    let solution_bytes = std::fs::read(&args.solution)
        .with_context(|| format!("Failed to read {}", args.solution.display()))?;
    // Decode now for early format errors and the provenance line; the
    // operator re-decodes and physically verifies at run time.
    let solution = fea_bundle::decode_solution(&solution_bytes).map_err(anyhow::Error::msg)?;

    let step_index = select_step(&project, &args.step)?;
    let step = project.timeline[step_index].clone();

    if step.operator_id == IMPORT_OPERATOR_ID {
        // Refresh: the step is already an import; swap the solution bytes.
        let ExecutionInput::AssetRef(solution_id) = &step.inputs[2] else {
            anyhow::bail!(
                "step {step_index} is a solution import but its solution \
                 input is inline; re-create it with fea-import"
            );
        };
        upsert_import(
            &mut project,
            ImportedAsset::new(
                solution_id.clone(),
                solution_bytes,
                Some(AssetTypeHint::Binary),
            ),
        );
    } else if step.operator_id.contains("fea_inverse") {
        // Swap the inverse step for a verified import, keeping the output
        // ids so downstream steps are untouched.
        anyhow::ensure!(
            step.inputs.len() == 4,
            "expected the fea_inverse step to have 4 inputs (mesh, rigid, \
             target, config), found {}",
            step.inputs.len()
        );
        let (op_name, op_bytes) = resolve_operator_spec(IMPORT_OPERATOR_ID)?;
        let metadata = volumetric::operator_metadata_from_wasm_bytes(&op_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to read operator metadata: {e}"))?;

        // Carry the physics fields of the original config over to the
        // import operator's schema (same names); verification knobs seed
        // from their declared defaults.
        let cddl = metadata
            .inputs
            .iter()
            .find_map(|input| match input {
                OperatorMetadataInput::CBORConfiguration(schema) => Some(schema.clone()),
                _ => None,
            })
            .context("the import operator declares no configuration input")?;
        let fields = operator_config::parse_schema(&cddl)
            .map_err(|e| anyhow::anyhow!("import operator config schema: {e}"))?;
        let mut values = operator_config::default_values(&fields);
        if let ExecutionInput::Inline(original) = &step.inputs[3]
            && !original.is_empty()
        {
            for (path, value) in operator_config::decode(original) {
                if operator_config::find_field(&fields, &path).is_some() {
                    values.insert(path, value);
                }
            }
        }
        let config_bytes = operator_config::encode(&fields, &values);

        upsert_import(&mut project, ImportedAsset::operator(op_name, op_bytes));
        upsert_import(
            &mut project,
            ImportedAsset::new(
                SOLUTION_ASSET_ID.to_string(),
                solution_bytes,
                Some(AssetTypeHint::Binary),
            ),
        );
        project.timeline[step_index] = ExecutionStep {
            operator_id: IMPORT_OPERATOR_ID.to_string(),
            inputs: vec![
                step.inputs[0].clone(),
                step.inputs[1].clone(),
                ExecutionInput::AssetRef(SOLUTION_ASSET_ID.to_string()),
                ExecutionInput::Inline(config_bytes),
            ],
            outputs: step.outputs.clone(),
        };
    } else {
        anyhow::bail!(
            "step {step_index} runs {}; fea-import replaces fea_inverse \
             steps (or refreshes an existing {IMPORT_OPERATOR_ID} step)",
            step.operator_id
        );
    }

    let output = args.output.unwrap_or(args.project);
    save_project(&project, &output)?;
    println!(
        "Adopted {} into {} step {step_index} (backend {}, {} inverse \
         iterations, converged={}, distribution_error={:.4}, solved in \
         {:.1}s, problem blake3 {})",
        args.solution.display(),
        output.display(),
        solution.stats.backend,
        solution.stats.inverse_iterations,
        solution.stats.converged,
        solution.stats.distribution_error,
        solution.stats.wall_seconds,
        hex(&solution.problem_blake3[..8.min(solution.problem_blake3.len())]),
    );
    println!(
        "The solution is verified against the live mesh and rigid body \
         when the project runs."
    );
    Ok(())
}
