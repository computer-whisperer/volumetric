//! Render subcommand for generating PNG images from volumetric models

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use glam::Vec3;
use std::path::PathBuf;

use volumetric::generate_adaptive_mesh_v2_from_bytes;

use crate::camera::{parse_views, CameraSetup, ProjectionType, ViewAngle};
use crate::headless_renderer::{GridVertex, HeadlessRenderer, MeshVertex, Uniforms, WireframeOptions};
use crate::{build_mesh_config, load_wasm_bytes};

/// Projection type for CLI argument parsing
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum ProjectionArg {
    #[default]
    Perspective,
    Ortho,
}

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

    /// Background color as hex (e.g., 2d2d2d)
    #[arg(long, default_value = "2d2d2d")]
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

    /// Enable sharp edge detection and vertex duplication
    #[arg(long)]
    pub sharp_edges: bool,

    /// Sharp edge angle threshold in degrees (default: 20)
    #[arg(long, default_value = "20.0")]
    pub sharp_angle: f64,

    /// Sharp edge residual multiplier (default: 4.0)
    #[arg(long, default_value = "4.0")]
    pub sharp_residual: f64,

    /// Suppress profiling output
    #[arg(short, long)]
    pub quiet: bool,

    /// Reference grid spacing in meters (0 to disable)
    #[arg(long, default_value = "1.0")]
    pub grid: f32,

    /// Grid color as hex (e.g., 555555)
    #[arg(long, default_value = "555555")]
    pub grid_color: String,

    // === New rendering mode options ===

    /// Projection type: perspective or ortho
    #[arg(long, value_enum, default_value = "perspective")]
    pub projection: ProjectionArg,

    /// Orthographic vertical scale in world units (auto-computed if omitted or 0)
    #[arg(long, default_value = "0.0")]
    pub ortho_scale: f32,

    /// Field of view for perspective projection in degrees
    #[arg(long, default_value = "45.0")]
    pub fov: f32,

    /// Render edges instead of filled triangles
    #[arg(long)]
    pub wireframe: bool,

    /// Wireframe line color as hex (e.g., ffffff)
    #[arg(long, default_value = "ffffff")]
    pub wireframe_color: String,

    /// Recompute smooth normals from mesh geometry
    #[arg(long)]
    pub recalc_normals: bool,

    /// Camera position as x,y,z (overrides --views)
    #[arg(long)]
    pub camera_pos: Option<String>,

    /// Look-at point as x,y,z (default: model center)
    #[arg(long)]
    pub camera_target: Option<String>,

    /// Up vector as x,y,z (default: 0,1,0)
    #[arg(long, default_value = "0,1,0")]
    pub camera_up: String,

    /// Near clipping plane distance (default: auto-computed from scene)
    #[arg(long)]
    pub near: Option<f32>,

    /// Far clipping plane distance (default: auto-computed from scene)
    #[arg(long)]
    pub far: Option<f32>,
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

/// Parse a Vec3 from "x,y,z" format
fn parse_vec3(s: &str) -> Result<Vec3> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        anyhow::bail!("Invalid vec3 format: expected 'x,y,z', got '{}'", s);
    }
    let x: f32 = parts[0].trim().parse().context("Invalid x component")?;
    let y: f32 = parts[1].trim().parse().context("Invalid y component")?;
    let z: f32 = parts[2].trim().parse().context("Invalid z component")?;
    Ok(Vec3::new(x, y, z))
}

/// Recalculate smooth normals from mesh geometry using area-weighted face normals
fn recalculate_normals(
    vertices: &[(f32, f32, f32)],
    indices: &[u32],
) -> Vec<(f32, f32, f32)> {
    let mut normals = vec![Vec3::ZERO; vertices.len()];

    // Accumulate area-weighted face normals for each vertex
    for tri in indices.chunks(3) {
        if tri.len() != 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let v0 = Vec3::new(vertices[i0].0, vertices[i0].1, vertices[i0].2);
        let v1 = Vec3::new(vertices[i1].0, vertices[i1].1, vertices[i1].2);
        let v2 = Vec3::new(vertices[i2].0, vertices[i2].1, vertices[i2].2);

        let e1 = v1 - v0;
        let e2 = v2 - v0;

        // Cross product gives area-weighted normal (length = 2 * triangle area)
        let face_normal = e1.cross(e2);

        // Accumulate to each vertex of the triangle
        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    // Normalize all vertex normals
    normals
        .into_iter()
        .map(|n| {
            let normalized = n.normalize_or_zero();
            // Fall back to up vector if zero normal
            let n = if normalized == Vec3::ZERO {
                Vec3::Y
            } else {
                normalized
            };
            (n.x, n.y, n.z)
        })
        .collect()
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

/// Generate grid line vertices on the XZ plane at y=0
fn generate_grid_vertices(
    bounds_min: Vec3,
    bounds_max: Vec3,
    spacing: f32,
    color: [f32; 3],
) -> Vec<GridVertex> {
    let mut vertices = Vec::new();

    // Extend grid slightly beyond model bounds, snapped to grid spacing
    let margin = spacing;
    let x_min = ((bounds_min.x - margin) / spacing).floor() * spacing;
    let x_max = ((bounds_max.x + margin) / spacing).ceil() * spacing;
    let z_min = ((bounds_min.z - margin) / spacing).floor() * spacing;
    let z_max = ((bounds_max.z + margin) / spacing).ceil() * spacing;

    // Place grid at y=0 or at the bottom of model bounds if model is above y=0
    let y = 0.0_f32.min(bounds_min.y);

    // Determine which lines are major (every 5 units) vs minor
    let major_spacing = spacing * 5.0;
    let major_color = color;
    let minor_color = [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6];

    // Generate lines parallel to Z axis (varying X)
    let mut x = x_min;
    while x <= x_max + 0.001 {
        let is_major = (x / major_spacing).abs().fract() < 0.01 || (x / major_spacing).abs().fract() > 0.99;
        let line_color = if is_major { major_color } else { minor_color };

        vertices.push(GridVertex {
            position: [x, y, z_min],
            _pad0: 0.0,
            color: line_color,
            _pad1: 0.0,
        });
        vertices.push(GridVertex {
            position: [x, y, z_max],
            _pad0: 0.0,
            color: line_color,
            _pad1: 0.0,
        });

        x += spacing;
    }

    // Generate lines parallel to X axis (varying Z)
    let mut z = z_min;
    while z <= z_max + 0.001 {
        let is_major = (z / major_spacing).abs().fract() < 0.01 || (z / major_spacing).abs().fract() > 0.99;
        let line_color = if is_major { major_color } else { minor_color };

        vertices.push(GridVertex {
            position: [x_min, y, z],
            _pad0: 0.0,
            color: line_color,
            _pad1: 0.0,
        });
        vertices.push(GridVertex {
            position: [x_max, y, z],
            _pad0: 0.0,
            color: line_color,
            _pad1: 0.0,
        });

        z += spacing;
    }

    vertices
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
    let grid_color = parse_hex_color(&args.grid_color).context("Invalid grid color")?;
    let wireframe_color = parse_hex_color(&args.wireframe_color).context("Invalid wireframe color")?;

    // Build projection type from CLI args
    let projection = match args.projection {
        ProjectionArg::Perspective => {
            let fov_y = args.fov.to_radians();
            ProjectionType::Perspective { fov_y }
        }
        ProjectionArg::Ortho => {
            // Warn if user specified fov with ortho
            if args.fov != 45.0 {
                eprintln!("Warning: --fov is ignored with orthographic projection");
            }
            ProjectionType::Orthographic { scale: args.ortho_scale }
        }
    };

    // Parse custom camera if specified
    let custom_camera_pos = args.camera_pos.as_ref()
        .map(|s| parse_vec3(s))
        .transpose()
        .context("Invalid --camera-pos")?;
    let custom_camera_target = args.camera_target.as_ref()
        .map(|s| parse_vec3(s))
        .transpose()
        .context("Invalid --camera-target")?;
    let camera_up = parse_vec3(&args.camera_up).context("Invalid --camera-up")?;

    // Determine if we're using custom camera or predefined views
    let use_custom_camera = custom_camera_pos.is_some();

    // Parse views (only used if not using custom camera)
    let views = if use_custom_camera {
        if args.views != "iso" {
            eprintln!("Warning: --views is ignored when using --camera-pos");
        }
        vec![ViewAngle::Iso] // Dummy, won't be used
    } else {
        let v = parse_views(&args.views);
        if v.is_empty() {
            anyhow::bail!("No valid views specified");
        }
        v
    };

    if use_custom_camera {
        println!("Using custom camera position");
    } else {
        println!("Rendering {} view(s): {:?}", views.len(), views.iter().map(|v| v.suffix()).collect::<Vec<_>>());
    }

    // Load WASM and generate mesh
    let wasm_bytes = load_wasm_bytes(&args.input)?;
    println!("Loaded {} bytes", wasm_bytes.len());

    let config = build_mesh_config(
        args.base_resolution,
        args.max_depth,
        args.vertex_refinement,
        args.normal_refinement,
        args.normal_epsilon,
        args.sharp_edges,
        args.sharp_angle,
        args.sharp_residual,
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

    // Apply normal recalculation if requested
    let normals = if args.recalc_normals {
        if !args.quiet {
            println!("Recalculating normals from mesh geometry");
        }
        recalculate_normals(&mesh_result.vertices, &mesh_result.indices)
    } else {
        mesh_result.normals.clone()
    };

    // Convert to GPU format
    let gpu_vertices = convert_to_gpu_vertices(&mesh_result.vertices, &normals);
    let (bounds_min, bounds_max) = compute_bounds(&mesh_result.vertices);

    if !args.quiet {
        println!(
            "Bounds: min=({:.3}, {:.3}, {:.3}) max=({:.3}, {:.3}, {:.3})",
            bounds_min.x, bounds_min.y, bounds_min.z, bounds_max.x, bounds_max.y, bounds_max.z
        );
    }

    // Generate grid vertices if grid spacing > 0 and not wireframe mode
    let grid_vertices = if args.grid > 0.0 && !args.wireframe {
        let gv = generate_grid_vertices(bounds_min, bounds_max, args.grid, grid_color);
        if !args.quiet {
            println!("Generated {} grid line segments", gv.len() / 2);
        }
        Some(gv)
    } else {
        if args.wireframe && args.grid > 0.0 && !args.quiet {
            println!("Note: Grid is disabled in wireframe mode");
        }
        None
    };

    // Build wireframe options
    let wireframe_options = if args.wireframe {
        if !args.quiet {
            println!("Wireframe mode enabled");
        }
        Some(WireframeOptions {
            color: wireframe_color,
        })
    } else {
        None
    };

    // Initialize renderer
    println!("Initializing headless renderer ({}x{})", args.width, args.height);
    let renderer = HeadlessRenderer::new(args.width, args.height)?;

    let aspect = args.width as f32 / args.height as f32;

    // Light direction: upper-front-right
    let light_dir = Vec3::new(0.5, 0.7, 0.5).normalize();

    // Compute fog parameters based on scene size
    let scene_size = (bounds_max - bounds_min).length();
    let center = (bounds_min + bounds_max) * 0.5;

    // Render based on camera mode
    if use_custom_camera {
        // Custom camera mode - single output
        let camera_pos = custom_camera_pos.unwrap();
        let camera_target = custom_camera_target.unwrap_or(center);

        let camera = CameraSetup::from_pose(
            camera_pos,
            camera_target,
            camera_up,
            projection,
            bounds_min,
            bounds_max,
        )
        .with_clip_planes(args.near, args.far);
        let view_proj = camera.view_proj(aspect);

        println!("Rendering to {:?}", args.output);

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
            grid_vertices.as_deref(),
            wireframe_options.as_ref(),
            &args.output,
        )?;
    } else {
        // Predefined views mode
        for view in &views {
            let output_path = output_path_for_view(&args.output, *view, views.len());
            println!("Rendering {} view to {:?}", view.suffix(), output_path);

            let camera = CameraSetup::auto_frame(bounds_min, bounds_max, *view, projection)
                .with_clip_planes(args.near, args.far);
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
                grid_vertices.as_deref(),
                wireframe_options.as_ref(),
                &output_path,
            )?;
        }
    }

    println!("Done!");
    Ok(())
}
