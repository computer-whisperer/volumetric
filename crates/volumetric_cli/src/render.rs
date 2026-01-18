//! Render subcommand for generating PNG images from volumetric models

use anyhow::{Context, Result};
use clap::Parser;
use glam::Vec3;
use std::path::PathBuf;

use volumetric::generate_adaptive_mesh_v2_from_bytes;

use crate::camera::{parse_views, CameraSetup, ViewAngle};
use crate::headless_renderer::{HeadlessRenderer, MeshVertex, Uniforms};
use crate::{build_mesh_config, load_wasm_bytes};

#[derive(Parser, Debug)]
pub struct RenderArgs {
    /// Input file: either a .wasm model or a .vproj project file
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output PNG file path (view suffix added for multiple views)
    #[arg(short, long)]
    pub output: PathBuf,

    /// Image width in pixels
    #[arg(long, default_value = "1024")]
    pub width: u32,

    /// Image height in pixels
    #[arg(long, default_value = "1024")]
    pub height: u32,

    /// Comma-separated views: front, back, left, right, top, bottom, iso, iso-back, all
    #[arg(long, default_value = "iso")]
    pub views: String,

    /// Background color as hex (e.g., f0f0f0)
    #[arg(long, default_value = "f0f0f0")]
    pub background: String,

    /// Mesh base color as hex (e.g., 6699cc)
    #[arg(long, default_value = "6699cc")]
    pub color: String,

    /// Base resolution for coarse grid discovery
    #[arg(long, default_value = "8")]
    pub base_resolution: usize,

    /// Maximum refinement depth
    #[arg(long, default_value = "4")]
    pub max_depth: usize,

    /// Vertex refinement iterations
    #[arg(long, default_value = "12")]
    pub vertex_refinement: usize,

    /// Normal refinement iterations
    #[arg(long, default_value = "12")]
    pub normal_refinement: usize,

    /// Normal epsilon fraction
    #[arg(long, default_value = "0.1")]
    pub normal_epsilon: f32,

    /// Suppress profiling output
    #[arg(short, long)]
    pub quiet: bool,
}

fn parse_hex_color(hex: &str) -> Result<[f32; 3]> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        anyhow::bail!("Invalid hex color: expected 6 characters, got {}", hex.len());
    }
    let r = u8::from_str_radix(&hex[0..2], 16).context("Invalid red component")?;
    let g = u8::from_str_radix(&hex[2..4], 16).context("Invalid green component")?;
    let b = u8::from_str_radix(&hex[4..6], 16).context("Invalid blue component")?;
    Ok([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0])
}

fn compute_bounds(vertices: &[(f32, f32, f32)]) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for &(x, y, z) in vertices {
        min = min.min(Vec3::new(x, y, z));
        max = max.max(Vec3::new(x, y, z));
    }

    (min, max)
}

fn convert_to_gpu_vertices(
    vertices: &[(f32, f32, f32)],
    normals: &[(f32, f32, f32)],
) -> Vec<MeshVertex> {
    vertices
        .iter()
        .zip(normals.iter())
        .map(|(&(px, py, pz), &(nx, ny, nz))| MeshVertex {
            position: [px, py, pz],
            _pad0: 0.0,
            normal: [nx, ny, nz],
            _pad1: 0.0,
        })
        .collect()
}

fn output_path_for_view(base_path: &PathBuf, view: ViewAngle, num_views: usize) -> PathBuf {
    if num_views == 1 {
        return base_path.clone();
    }

    let stem = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("render");
    let ext = base_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");

    let new_name = format!("{}_{}.{}", stem, view.suffix(), ext);

    base_path
        .parent()
        .map(|p| p.join(&new_name))
        .unwrap_or_else(|| PathBuf::from(&new_name))
}

pub fn run_render(args: RenderArgs) -> Result<()> {
    // Parse colors
    let background_color = parse_hex_color(&args.background).context("Invalid background color")?;
    let base_color = parse_hex_color(&args.color).context("Invalid mesh color")?;

    // Parse views
    let views = parse_views(&args.views);
    if views.is_empty() {
        anyhow::bail!("No valid views specified");
    }

    println!("Rendering {} view(s): {:?}", views.len(), views.iter().map(|v| v.suffix()).collect::<Vec<_>>());

    // Load WASM and generate mesh
    let wasm_bytes = load_wasm_bytes(&args.input)?;
    println!("Loaded {} bytes", wasm_bytes.len());

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

    let mesh_result = generate_adaptive_mesh_v2_from_bytes(&wasm_bytes, &config)
        .context("Mesh generation failed")?;

    if !args.quiet {
        println!(
            "Generated {} vertices, {} triangles",
            mesh_result.vertices.len(),
            mesh_result.indices.len() / 3
        );
    }

    // Convert to GPU format
    let gpu_vertices = convert_to_gpu_vertices(&mesh_result.vertices, &mesh_result.normals);
    let (bounds_min, bounds_max) = compute_bounds(&mesh_result.vertices);

    if !args.quiet {
        println!(
            "Bounds: min=({:.3}, {:.3}, {:.3}) max=({:.3}, {:.3}, {:.3})",
            bounds_min.x, bounds_min.y, bounds_min.z, bounds_max.x, bounds_max.y, bounds_max.z
        );
    }

    // Initialize renderer
    println!("Initializing headless renderer ({}x{})", args.width, args.height);
    let renderer = HeadlessRenderer::new(args.width, args.height)?;

    let fov_y = std::f32::consts::FRAC_PI_4; // 45 degrees
    let aspect = args.width as f32 / args.height as f32;

    // Light direction: upper-front-right
    let light_dir = Vec3::new(0.5, 0.7, 0.5).normalize();

    // Compute fog parameters based on scene size
    let scene_size = (bounds_max - bounds_min).length();

    // Render each view
    for view in &views {
        let output_path = output_path_for_view(&args.output, *view, views.len());
        println!("Rendering {} view to {:?}", view.suffix(), output_path);

        let camera = CameraSetup::auto_frame(bounds_min, bounds_max, *view, fov_y);
        let view_proj = camera.view_proj(aspect);

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera.position.to_array(),
            _pad0: 0.0,
            light_dir: light_dir.to_array(),
            _pad1: 0.0,
            base_color,
            rim_strength: 0.4,
            sky_color: [0.95, 0.96, 0.98],
            fog_density: 0.5 / scene_size,
            ground_color: [0.4, 0.42, 0.45],
            fog_start: scene_size * 0.5,
        };

        renderer.render_to_png(
            &gpu_vertices,
            &mesh_result.indices,
            &uniforms,
            background_color,
            &output_path,
        )?;
    }

    println!("Done!");
    Ok(())
}
