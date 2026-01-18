//! Volumetric CLI - Command-line mesh generation and rendering tool
//!
//! Provides subcommands for:
//! - `mesh`: Generate STL files from volumetric models
//! - `render`: Generate PNG images using headless wgpu rendering
//! - `info`: Inspect WASM models, operators, and projects
//! - `bounds`: Query bounding box of a model
//! - `sample`: Sample is_inside values at points
//! - `project-*`: Create and manipulate .vproj project files

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use volumetric::{
    adaptive_surface_nets_2::{AdaptiveMeshConfig2, MeshingStats2},
    generate_adaptive_mesh_v2_from_bytes, stl, Environment, Project, Triangle,
};

mod camera;
mod headless_renderer;
mod info;
mod project;
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
    /// Get information about a WASM model, operator, or project
    Info(info::InfoArgs),
    /// Query the bounding box of a model
    Bounds(info::BoundsArgs),
    /// Sample is_inside values at specified points
    Sample(info::SampleArgs),
    /// Create a new project from a model WASM
    #[command(name = "project-new")]
    ProjectNew(project::ProjectNewArgs),
    /// Add a model to an existing project
    #[command(name = "project-add-model")]
    ProjectAddModel(project::ProjectAddModelArgs),
    /// Add an operator execution to a project
    #[command(name = "project-add-op")]
    ProjectAddOp(project::ProjectAddOpArgs),
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

    /// Suppress profiling output
    #[arg(short, long)]
    quiet: bool,
}

pub fn load_wasm_bytes(path: &PathBuf) -> Result<Vec<u8>> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "wasm" => {
            println!("Loading WASM model from {:?}", path);
            std::fs::read(path).context("Failed to read WASM file")
        }
        "vproj" => {
            println!("Loading project from {:?}", path);
            let project = Project::load_from_file(path).context("Failed to load .vproj file")?;

            // Run the project to get the exported assets
            let mut env = Environment::new();
            let exports = project
                .run(&mut env)
                .map_err(|e| anyhow::anyhow!("Project execution failed: {}", e))?;

            // Take the first exported model
            let export = exports
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Project has no exported assets"))?;

            export
                .as_model_wasm()
                .map(|b| b.to_vec())
                .ok_or_else(|| anyhow::anyhow!("Exported asset is not a ModelWASM"))
        }
        _ => Err(anyhow::anyhow!(
            "Unknown file extension: {:?}. Expected .wasm or .vproj",
            extension
        )),
    }
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

    println!("==========================");
}

pub fn build_mesh_config(
    base_resolution: usize,
    max_depth: usize,
    vertex_refinement: usize,
    normal_refinement: usize,
    normal_epsilon: f32,
) -> AdaptiveMeshConfig2 {
    AdaptiveMeshConfig2 {
        base_resolution,
        max_depth,
        vertex_refinement_iterations: vertex_refinement,
        normal_sample_iterations: normal_refinement,
        normal_epsilon_frac: normal_epsilon,
        num_threads: 0,
    }
}

fn run_mesh(args: MeshArgs) -> Result<()> {
    // Load WASM bytes
    let wasm_bytes = load_wasm_bytes(&args.input)?;
    println!("Loaded {} bytes", wasm_bytes.len());

    // Configure meshing
    let config = build_mesh_config(
        args.base_resolution,
        args.max_depth,
        args.vertex_refinement,
        args.normal_refinement,
        args.normal_epsilon,
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
        Commands::Info(args) => info::run_info(args),
        Commands::Bounds(args) => info::run_bounds(args),
        Commands::Sample(args) => info::run_sample(args),
        Commands::ProjectNew(args) => project::run_project_new(args),
        Commands::ProjectAddModel(args) => project::run_project_add_model(args),
        Commands::ProjectAddOp(args) => project::run_project_add_op(args),
        Commands::ProjectExport(args) => project::run_project_export(args),
        Commands::ProjectRun(args) => project::run_project_run(args),
        Commands::ProjectList(args) => project::run_project_list(args),
    }
}
