//! Volumetric CLI - Command-line mesh generation and rendering tool
//!
//! Provides subcommands for:
//! - `mesh`: Generate STL files from volumetric models
//! - `render`: Generate PNG images using headless wgpu rendering
//! - `sketch-raster`: Rasterize a 2D sketch model to PNG or ASCII
//! - `info`: Inspect WASM models, operators, and projects
//! - `bounds`: Query bounding box of a model
//! - `sample`: Sample occupancy values at points
//! - `assets`: List the models and operators bundled into the binary
//! - `project-*`: Create, validate and manipulate .vproj project files
//!
//! Model and operator arguments accept either a filesystem path or the name
//! of a bundled asset (see `assets`).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use volumetric::{
    Environment, Project, Triangle,
    adaptive_surface_nets_2::{AdaptiveMeshConfig2, MeshingStats2},
    generate_adaptive_mesh_v2_from_bytes,
    mesh_decimation::DecimationConfig,
    sharp_features::SharpFeatureConfig,
    stl,
};

mod assets;
mod camera;
mod headless_renderer;
mod info;
mod project;
mod raster;
mod render;

#[derive(Parser, Debug)]
#[command(name = "volumetric_cli")]
#[command(about = "Generate meshes and renders from volumetric models", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate an STL mesh from a volumetric model
    Mesh(MeshArgs),
    /// Render a volumetric model to PNG image(s)
    Render(render::RenderArgs),
    /// Rasterize a 2D sketch model to a PNG image or ASCII art
    #[command(name = "sketch-raster")]
    SketchRaster(raster::SketchRasterArgs),
    /// Get information about a WASM model, operator, or project
    Info(info::InfoArgs),
    /// Query the bounding box of a model
    Bounds(info::BoundsArgs),
    /// Sample occupancy values at specified points
    Sample(info::SampleArgs),
    /// List models and operators bundled into this binary
    Assets(assets::AssetsArgs),
    /// Create a new project, optionally seeded with a model
    #[command(name = "project-new")]
    ProjectNew(project::ProjectNewArgs),
    /// Add a model to an existing project
    #[command(name = "project-add-model")]
    ProjectAddModel(project::ProjectAddModelArgs),
    /// Add a non-model asset (Lua source, config, blob) to a project
    #[command(name = "project-add-asset")]
    ProjectAddAsset(project::ProjectAddAssetArgs),
    /// Add an operator execution to a project
    #[command(name = "project-add-op")]
    ProjectAddOp(project::ProjectAddOpArgs),
    /// Check a project for structural problems without running it
    #[command(name = "project-validate")]
    ProjectValidate(project::ProjectValidateArgs),
    /// Export assets from a project to WASM files
    #[command(name = "project-export")]
    ProjectExport(project::ProjectExportArgs),
    /// Run a project and show exported assets
    #[command(name = "project-run")]
    ProjectRun(project::ProjectRunArgs),
    /// List assets in a project
    #[command(name = "project-list")]
    ProjectList(project::ProjectListArgs),
}

#[derive(Parser, Debug)]
pub struct MeshArgs {
    /// Input file: either a .wasm model or a .vproj project file
    #[arg(short, long)]
    input: PathBuf,

    /// For .vproj inputs with multiple exports: which exported asset to mesh
    #[arg(long)]
    asset: Option<String>,

    /// Output STL file path
    #[arg(short, long)]
    output: PathBuf,

    /// Base resolution for coarse grid discovery (default: 8)
    #[arg(long, default_value = "8")]
    base_resolution: usize,

    /// Maximum refinement depth (default: 4, effective resolution = base * 2^depth)
    #[arg(long, default_value = "4")]
    max_depth: usize,

    /// Vertex refinement iterations (default: 12)
    #[arg(long, default_value = "12")]
    vertex_refinement: usize,

    /// Normal refinement iterations (default: 12, 0 to disable)
    #[arg(long, default_value = "12")]
    normal_refinement: usize,

    /// Normal epsilon fraction (default: 0.1)
    #[arg(long, default_value = "0.1")]
    normal_epsilon: f32,

    /// Enable sharp feature reconstruction (region-based edge/corner snapping)
    #[arg(long)]
    sharp_edges: bool,

    /// Sharp features: max same-region normal jump between adjacent vertices
    /// in degrees (default: 15)
    #[arg(long, default_value = "15.0")]
    sharp_angle: f64,

    /// Disable the decimation post-pass (stage 5 quadric simplification)
    #[arg(long)]
    no_simplify: bool,

    /// Decimation error budget, as a fraction of the finest cell size
    #[arg(long, default_value = "1.0")]
    simplify_tolerance: f64,

    /// Constrain vertex refinement to each vertex's own grid edge. Prevents
    /// refinement from capturing a neighboring parallel surface — use for
    /// thin-walled lattices whose sheets visually bond together
    #[arg(long)]
    edge_constrained: bool,

    /// Suppress profiling output
    #[arg(short, long)]
    quiet: bool,
}

/// Load model WASM bytes from a `.wasm` file or a `.vproj` project.
///
/// For projects, `asset` selects which export to use. With no selector the
/// project must have exactly one model export — silently picking the first
/// of several meshes the wrong thing.
pub fn load_wasm_bytes(path: &PathBuf, asset: Option<&str>) -> Result<Vec<u8>> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "wasm" => {
            if let Some(asset) = asset {
                anyhow::bail!(
                    "--asset '{asset}' only applies to .vproj inputs; {path:?} is a .wasm file"
                );
            }
            eprintln!("Loading WASM model from {:?}", path);
            let bytes = std::fs::read(path).context("Failed to read WASM file")?;
            assets::ensure_wasm(&bytes, "model", &path.display().to_string())?;
            Ok(bytes)
        }
        "vproj" => {
            eprintln!("Loading project from {:?}", path);
            let project = Project::load_from_file(path).context("Failed to load .vproj file")?;

            // Run the project to get the exported assets
            let mut env = Environment::new();
            let exports = project
                .run(&mut env)
                .map_err(|e| anyhow::anyhow!("Project execution failed: {}", e))?;

            let model_exports: Vec<_> = exports.iter().filter(|e| e.as_model().is_some()).collect();

            let export = match asset {
                Some(id) => model_exports
                    .iter()
                    .find(|e| e.id() == id)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "project has no model export named '{id}'. Available: {}",
                            join_ids(&model_exports)
                        )
                    })?,
                None => match model_exports.as_slice() {
                    [] => anyhow::bail!("project has no model exports"),
                    [only] => only,
                    _ => anyhow::bail!(
                        "project has {} model exports; select one with --asset. Available: {}",
                        model_exports.len(),
                        join_ids(&model_exports)
                    ),
                },
            };

            eprintln!("Using export '{}'", export.id());
            Ok(export
                .as_model()
                .expect("filtered to model exports")
                .to_vec())
        }
        _ => Err(anyhow::anyhow!(
            "Unknown file extension: {:?}. Expected .wasm or .vproj",
            extension
        )),
    }
}

fn join_ids(exports: &[&volumetric::LoadedAsset]) -> String {
    exports
        .iter()
        .map(|e| e.id())
        .collect::<Vec<_>>()
        .join(", ")
}

fn indexed_mesh_to_triangles(
    vertices: &[(f32, f32, f32)],
    normals: &[(f32, f32, f32)],
    indices: &[u32],
) -> Vec<Triangle> {
    indices
        .chunks(3)
        .filter_map(|tri| {
            if tri.len() != 3 {
                return None;
            }
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let v0 = vertices.get(i0)?;
            let v1 = vertices.get(i1)?;
            let v2 = vertices.get(i2)?;
            let n0 = normals.get(i0)?;
            let n1 = normals.get(i1)?;
            let n2 = normals.get(i2)?;

            Some(Triangle::with_vertex_normals(
                [*v0, *v1, *v2],
                [*n0, *n1, *n2],
            ))
        })
        .collect()
}

fn print_stats_summary(stats: &MeshingStats2) {
    println!();
    println!("=== Meshing Statistics ===");
    println!("Total time:      {:.2}ms", stats.total_time_secs * 1000.0);
    println!("Total samples:   {}", stats.total_samples);
    println!(
        "Avg sample time: {:.2}µs",
        stats.total_time_secs * 1_000_000.0 / stats.total_samples as f64
    );
    println!("Vertices:        {}", stats.total_vertices);
    println!("Triangles:       {}", stats.total_triangles);
    println!("Resolution:      {}³", stats.effective_resolution);
    println!();
    println!("Stage breakdown:");
    println!(
        "  1. Discovery:   {:>7.2}ms ({:>5.1}%) - {} samples, {} mixed cells",
        stats.stage1_time_secs * 1000.0,
        stats.stage1_time_secs / stats.total_time_secs * 100.0,
        stats.stage1_samples,
        stats.stage1_mixed_cells
    );
    println!(
        "  2. Subdivision: {:>7.2}ms ({:>5.1}%) - {} samples, {} cuboids",
        stats.stage2_time_secs * 1000.0,
        stats.stage2_time_secs / stats.total_time_secs * 100.0,
        stats.stage2_samples,
        stats.stage2_cuboids_processed
    );
    println!(
        "  3. Topology:    {:>7.2}ms ({:>5.1}%) - {} unique vertices",
        stats.stage3_time_secs * 1000.0,
        stats.stage3_time_secs / stats.total_time_secs * 100.0,
        stats.stage3_unique_vertices
    );
    println!(
        "  4. Refinement:  {:>7.2}ms ({:>5.1}%) - {} samples",
        stats.stage4_time_secs * 1000.0,
        stats.stage4_time_secs / stats.total_time_secs * 100.0,
        stats.stage4_samples
    );

    // Print refinement diagnostics if available
    let total_refine = stats.stage4_refine_primary_hit
        + stats.stage4_refine_fallback_x_hit
        + stats.stage4_refine_fallback_y_hit
        + stats.stage4_refine_fallback_z_hit
        + stats.stage4_refine_miss;

    if total_refine > 0 {
        println!();
        println!("Vertex refinement outcomes:");
        println!(
            "  Primary hit:    {:>10} ({:>5.2}%)",
            stats.stage4_refine_primary_hit,
            stats.stage4_refine_primary_hit as f64 / total_refine as f64 * 100.0
        );
        println!(
            "  Fallback X:     {:>10} ({:>5.2}%)",
            stats.stage4_refine_fallback_x_hit,
            stats.stage4_refine_fallback_x_hit as f64 / total_refine as f64 * 100.0
        );
        println!(
            "  Fallback Y:     {:>10} ({:>5.2}%)",
            stats.stage4_refine_fallback_y_hit,
            stats.stage4_refine_fallback_y_hit as f64 / total_refine as f64 * 100.0
        );
        println!(
            "  Fallback Z:     {:>10} ({:>5.2}%)",
            stats.stage4_refine_fallback_z_hit,
            stats.stage4_refine_fallback_z_hit as f64 / total_refine as f64 * 100.0
        );
        println!(
            "  MISS:           {:>10} ({:>5.2}%)",
            stats.stage4_refine_miss,
            stats.stage4_refine_miss as f64 / total_refine as f64 * 100.0
        );
    }

    // Print sharp feature stats if any processing occurred
    if stats.sharp_candidates > 0 || stats.sharp_regions > 0 {
        println!();
        println!(
            "Stage 4.5 (Sharp Features): {:>7.2}ms ({:>5.1}%)",
            stats.stage4_5_time_secs * 1000.0,
            stats.stage4_5_time_secs / stats.total_time_secs * 100.0
        );
        println!("  Smooth regions:      {}", stats.sharp_regions);
        println!("  Feature candidates:  {}", stats.sharp_candidates);
        println!(
            "  Snapped:             {} edge + {} corner",
            stats.sharp_snapped_edges, stats.sharp_snapped_corners
        );
        println!("  Welded vertices:     {}", stats.sharp_welded_vertices);
        println!("  Dropped triangles:   {}", stats.sharp_dropped_triangles);
        println!("  Crease splits:       {}", stats.sharp_crease_splits);
    }

    println!("==========================");
}

#[allow(clippy::too_many_arguments)]
pub fn build_mesh_config(
    base_resolution: usize,
    max_depth: usize,
    vertex_refinement: usize,
    normal_refinement: usize,
    normal_epsilon: f32,
    sharp_edges: bool,
    sharp_angle: f64,
    simplify_tolerance: Option<f64>,
    edge_constrained: bool,
) -> AdaptiveMeshConfig2 {
    let sharp_features = sharp_edges.then(|| {
        let mut config = SharpFeatureConfig::default();
        config.segmentation.max_normal_jump_deg = sharp_angle;
        config
    });

    AdaptiveMeshConfig2 {
        base_resolution,
        max_depth,
        vertex_refinement_iterations: vertex_refinement,
        normal_sample_iterations: normal_refinement,
        normal_epsilon_frac: normal_epsilon,
        num_threads: 0,
        sharp_features,
        edge_constrained_refinement: edge_constrained,
        decimation: simplify_tolerance.map(|tolerance| DecimationConfig {
            error_tolerance_cells: tolerance,
            ..Default::default()
        }),
    }
}

fn run_mesh(args: MeshArgs) -> Result<()> {
    // Load WASM bytes
    let wasm_bytes = load_wasm_bytes(&args.input, args.asset.as_deref())?;
    println!("Loaded {} bytes", wasm_bytes.len());

    // Configure meshing
    let config = build_mesh_config(
        args.base_resolution,
        args.max_depth,
        args.vertex_refinement,
        args.normal_refinement,
        args.normal_epsilon,
        args.sharp_edges,
        args.sharp_angle,
        (!args.no_simplify).then_some(args.simplify_tolerance),
        args.edge_constrained,
    );

    let effective_res = config.base_resolution * (1 << config.max_depth);
    println!(
        "Meshing with resolution {}³ (base={}, depth={})",
        effective_res, config.base_resolution, config.max_depth
    );

    // Generate mesh
    let result = generate_adaptive_mesh_v2_from_bytes(&wasm_bytes, &config)
        .context("Mesh generation failed")?;

    // Print statistics
    if !args.quiet {
        print_stats_summary(&result.stats);
    }

    // Convert to triangles and export STL
    let triangles = indexed_mesh_to_triangles(&result.vertices, &result.normals, &result.indices);

    if triangles.is_empty() {
        eprintln!(
            "warning: the mesh is empty. If the model is a sparse lattice, coarse discovery \
             may have missed its struts entirely — try a higher --max-depth."
        );
    }
    println!(
        "Exporting {} triangles to {:?}",
        triangles.len(),
        args.output
    );
    stl::write_binary_stl(&args.output, &triangles, "volumetric_cli")
        .context("Failed to write STL file")?;

    println!("Done!");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Mesh(args) => run_mesh(args),
        Commands::Render(args) => render::run_render(args),
        Commands::SketchRaster(args) => raster::run_sketch_raster(args),
        Commands::Info(args) => info::run_info(args),
        Commands::Bounds(args) => info::run_bounds(args),
        Commands::Sample(args) => info::run_sample(args),
        Commands::Assets(args) => assets::run_assets(args),
        Commands::ProjectNew(args) => project::run_project_new(args),
        Commands::ProjectAddModel(args) => project::run_project_add_model(args),
        Commands::ProjectAddAsset(args) => project::run_project_add_asset(args),
        Commands::ProjectAddOp(args) => project::run_project_add_op(args),
        Commands::ProjectValidate(args) => project::run_project_validate(args),
        Commands::ProjectExport(args) => project::run_project_export(args),
        Commands::ProjectRun(args) => project::run_project_run(args),
        Commands::ProjectList(args) => project::run_project_list(args),
    }
}
