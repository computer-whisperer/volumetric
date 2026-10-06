//! `cast-bench`: runs the direct caster on a model and reports what it
//! cost and how well the result agrees with the model and the mesher.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use clap::Parser;
use glam::{DVec3, Vec3};
use volumetric::direct_cast::{CastImage, CastOptions, CastProjection, CastView, DirectCast};
use volumetric::wasm::ParallelModelSampler;

#[derive(Parser, Debug)]
pub struct CastBenchArgs {
    /// Input file: .wasm model or .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// For .vproj inputs with multiple exports: which exported asset to use
    #[arg(long)]
    pub asset: Option<String>,

    /// Image width in pixels
    #[arg(long, default_value_t = 1024)]
    pub width: u32,

    /// Image height in pixels
    #[arg(long, default_value_t = 1024)]
    pub height: u32,

    /// View direction as "yaw,pitch" in degrees: yaw about +Z from +X,
    /// pitch above the horizon. The view is framed on the model's bounds
    #[arg(long, default_value = "35,25", allow_hyphen_values = true)]
    pub view: String,

    /// Magnification about the centre of the bounds
    #[arg(long, default_value_t = 1.0)]
    pub zoom: f64,

    /// Orthographic instead of perspective
    #[arg(long)]
    pub ortho: bool,

    /// Ray spacing of the finished image in pixels
    #[arg(long, default_value_t = 1.0)]
    pub spacing: f64,

    /// Finest pitch empty space is searched at, in pixels. "inf" searches
    /// only at the coarsest spacing and follows what that finds
    #[arg(long, default_value_t = 1.0)]
    pub search: f64,

    /// Cast this many views in turn, each this many degrees of yaw on from
    /// the last, to see what a view reuses of the ones before
    #[arg(long, default_value_t = 1)]
    pub views: u32,

    /// Yaw between successive views, in degrees
    #[arg(long, default_value_t = 10.0)]
    pub yaw_step: f64,

    /// Write the last view's image, shaded from the normals found
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Also mesh the model at this refinement depth (as `mesh --max-depth`)
    /// and compare the last view with the mesh's vertices
    #[arg(long)]
    pub mesh_depth: Option<usize>,

    /// Print the hit under this pixel "x,y" of the last view and the hits
    /// around it, with the model sampled along the ray through it
    #[arg(long)]
    pub probe: Option<String>,
}

fn view_for(
    args: &CastBenchArgs,
    min: DVec3,
    max: DVec3,
    yaw_degrees: f64,
    pitch_degrees: f64,
) -> CastView {
    let center = (min + max) * 0.5;
    let radius = (max - min).length() * 0.5;
    let (yaw, pitch) = (yaw_degrees.to_radians(), pitch_degrees.to_radians());
    let toward_eye = DVec3::new(
        yaw.cos() * pitch.cos(),
        yaw.sin() * pitch.cos(),
        pitch.sin(),
    );
    const TAN_HALF_FOV: f64 = 0.35;
    // Far enough that the bounding sphere fills the shorter side.
    let fit = (args.height.min(args.width) as f64 / args.height as f64).min(1.0);
    let (distance, projection) = if args.ortho {
        (
            radius * 3.0,
            CastProjection::Orthographic {
                half_height: radius * 1.05 / fit / args.zoom,
            },
        )
    } else {
        (
            radius * 1.05 / (TAN_HALF_FOV.atan().sin() * fit),
            CastProjection::Perspective {
                tan_half_fov_y: TAN_HALF_FOV / args.zoom,
            },
        )
    };
    CastView::look_at(
        center + toward_eye * distance,
        center,
        DVec3::Z,
        projection,
        args.width,
        args.height,
    )
}

pub fn run_cast_bench(args: CastBenchArgs) -> Result<()> {
    let (yaw, pitch) = args
        .view
        .split_once(',')
        .and_then(|(yaw, pitch)| {
            Some((
                yaw.trim().parse::<f64>().ok()?,
                pitch.trim().parse::<f64>().ok()?,
            ))
        })
        .context("--view takes \"yaw,pitch\" in degrees")?;

    let wasm_bytes = crate::load_wasm_bytes(&args.input, args.asset.as_deref())?;
    let sampler = volumetric::wasm::create_parallel_sampler(&wasm_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to instantiate model: {e}"))?;
    let bounds = sampler
        .get_bounds()
        .map_err(|e| anyhow::anyhow!("Failed to read bounds: {e}"))?;
    let (min, max) = (DVec3::from(bounds.min), DVec3::from(bounds.max));
    let mut cast = DirectCast::new(min, max)?;
    let options = CastOptions {
        spacing: args.spacing,
        search: args.search,
        ..Default::default()
    };
    let never = AtomicBool::new(false);

    let mut last = None;
    for index in 0..args.views {
        let view = view_for(&args, min, max, yaw + index as f64 * args.yaw_step, pitch);
        let first_pass = cast.history.len();
        let start = std::time::Instant::now();
        let (image, total) = cast
            .cast(&sampler, &view, &options, &never)
            .expect("never cancelled");
        let seconds = start.elapsed().as_secs_f64();

        println!("View {} of {}", index + 1, args.views);
        println!(
            "  {:<9} {:>7} {:>9} {:>11} {:>9} {:>9} {:>9}",
            "pass", "spacing", "rays", "samples", "hits", "new", "seconds"
        );
        for (mode, spacing, stats) in &cast.history[first_pass..] {
            println!(
                "  {:<9} {:>7} {:>9} {:>11} {:>9} {:>9} {:>9.3}",
                format!("{mode:?}"),
                spacing,
                stats.rays,
                stats.samples,
                stats.hits,
                stats.fresh,
                stats.seconds
            );
        }
        let pixels = args.width as f64 * args.height as f64;
        println!(
            "  total: {} samples in {:.2} s ({:.1} M/s), {:.1} per pixel, {:.1} per hit pixel",
            total.samples,
            seconds,
            total.samples as f64 / seconds / 1e6,
            total.samples as f64 / pixels,
            total.samples as f64 / image.hits.iter().flatten().count().max(1) as f64,
        );
        last = Some((view, image));
    }
    let (view, image) = last.context("--views must be at least 1")?;

    // Every surfel should sit on the surface: outside the model a little
    // way along its normal, inside the same way back.
    let surfels = cast.surfels(None);
    let mut off = 0usize;
    for surfel in &surfels {
        let p = Vec3::from(surfel.position).as_dvec3();
        let step = Vec3::from(surfel.normal).as_dvec3() * (surfel.radius as f64 * 0.25);
        let inside = |p: DVec3| volumetric_abi::is_occupied(sampler.sample(p.x, p.y, p.z));
        off += (inside(p + step) || !inside(p - step)) as usize;
    }
    println!(
        "Record: {} nodes, {} surfels at the finest level; {} ({:.2}%) do not straddle the surface a quarter radius either way",
        cast.node_count(),
        surfels.len(),
        off,
        100.0 * off as f64 / surfels.len().max(1) as f64
    );

    if let Some(depth) = args.mesh_depth {
        compare_with_mesh(&wasm_bytes, depth, &view, &image)?;
    }
    if let Some(probe) = &args.probe {
        let (x, y) = probe
            .split_once(',')
            .and_then(|(x, y)| Some((x.trim().parse::<u32>().ok()?, y.trim().parse::<u32>().ok()?)))
            .context("--probe takes \"x,y\" in pixels")?;
        probe_pixel(&sampler, &view, &image, x, y);
    }
    if let Some(path) = &args.output {
        write_shaded(&view, &image, path)?;
        println!("Wrote {}", path.display());
    }
    if sampler.instantiation_failures() > 0 {
        anyhow::bail!(
            "{} thread(s) could not instantiate the model, so the result is wrong: {}",
            sampler.instantiation_failures(),
            sampler.instantiation_failure_detail().unwrap_or_default()
        );
    }
    Ok(())
}

/// Where each mesh vertex falls in the image, the cast should show
/// surface at the vertex's depth, or something nearer that hides it.
fn compare_with_mesh(
    wasm_bytes: &[u8],
    depth: usize,
    view: &CastView,
    image: &CastImage,
) -> Result<()> {
    let config = crate::build_mesh_config(8, depth, 8, 12, 0, 0.1, false, 15.0, None, false);
    let start = std::time::Instant::now();
    let mesh = volumetric::generate_adaptive_mesh_v2_from_bytes(wasm_bytes, &config)
        .context("Mesh generation failed")?;
    println!(
        "Mesh at {} cells: {} vertices in {:.2} s (no sharp edges, no decimation)",
        8usize << depth,
        mesh.vertices.len(),
        start.elapsed().as_secs_f64()
    );

    const TOLERANCE: f64 = 2.0;
    let (mut hidden, mut missing, mut offsets) = (0usize, 0usize, Vec::new());
    for &(x, y, z) in &mesh.vertices {
        let vertex = DVec3::new(x as f64, y as f64, z as f64);
        let Some((px, py)) = view.pixel_of(vertex) else {
            continue;
        };
        let pixel = view.footprint_at(vertex);
        let depth = match view.projection {
            CastProjection::Perspective { .. } => (vertex - view.eye).length(),
            CastProjection::Orthographic { .. } => (vertex - view.eye).dot(view.forward),
        };
        // The nearest in depth of the rays around the vertex's pixel.
        let (cx, cy) = ((px / image.spacing) as i32, (py / image.spacing) as i32);
        let mut best: Option<f64> = None;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= image.width as i32 || y >= image.height as i32 {
                    continue;
                }
                if let Some(hit) = image.hit(x as u32, y as u32) {
                    let offset = (hit.t - depth) / pixel;
                    if best.is_none_or(|b| offset.abs() < b.abs()) {
                        best = Some(offset);
                    }
                }
            }
        }
        match best {
            Some(offset) if offset.abs() <= TOLERANCE => offsets.push(offset.abs()),
            Some(offset) if offset < 0.0 => hidden += 1,
            _ => missing += 1,
        }
    }
    offsets.sort_by(f64::total_cmp);
    let at = |q: f64| {
        offsets
            .get(((offsets.len() as f64 * q) as usize).min(offsets.len().saturating_sub(1)))
            .copied()
            .unwrap_or(f64::NAN)
    };
    let seen = offsets.len() + missing;
    println!(
        "  of {seen} vertices not hidden behind nearer surface ({hidden} hidden), the cast shows surface within {TOLERANCE} px of {} ({:.2}%); {missing} it does not",
        offsets.len(),
        100.0 * offsets.len() as f64 / seen.max(1) as f64
    );
    println!(
        "  depth difference where it does: median {:.3} px, 99th percentile {:.3} px",
        at(0.5),
        at(0.99)
    );
    Ok(())
}

/// Prints what the rays around pixel (`x`, `y`) found, and the model's
/// answer along the centre ray a quarter pixel at a time around its hit.
fn probe_pixel(
    sampler: &impl ParallelModelSampler,
    view: &CastView,
    image: &CastImage,
    x: u32,
    y: u32,
) {
    for dy in -2..=2i32 {
        for dx in -2..=2i32 {
            let (px, py) = (x as i32 + dx, y as i32 + dy);
            if px < 0 || py < 0 || px >= image.width as i32 || py >= image.height as i32 {
                continue;
            }
            match image.hit(px as u32, py as u32) {
                Some(hit) => println!(
                    "  ({px},{py}) t {:.6} normal [{:.3} {:.3} {:.3}] at [{:.6} {:.6} {:.6}]",
                    hit.t,
                    hit.normal.x,
                    hit.normal.y,
                    hit.normal.z,
                    hit.position.x,
                    hit.position.y,
                    hit.position.z
                ),
                None => println!("  ({px},{py}) miss"),
            }
        }
    }
    // A missed pixel is probed at the depth of the nearest hit around it.
    let near = (-2..=2i32)
        .flat_map(|dy| (-2..=2i32).map(move |dx| (dx, dy)))
        .filter_map(|(dx, dy)| {
            let (px, py) = (x as i32 + dx, y as i32 + dy);
            let inside = px >= 0 && py >= 0 && px < image.width as i32 && py < image.height as i32;
            inside
                .then(|| {
                    image
                        .hit(px as u32, py as u32)
                        .map(|hit| (dx * dx + dy * dy, hit))
                })
                .flatten()
        })
        .min_by_key(|(d, _)| *d);
    let Some((_, hit)) = near else { return };
    let ray = view.ray(
        (x as f64 + 0.5) * image.spacing,
        (y as f64 + 0.5) * image.spacing,
    );
    let pixel = view.footprint_at(hit.position);
    let line: String = (-12..=12)
        .map(|i| {
            let p = ray.at(hit.t + i as f64 * pixel * 0.25);
            if volumetric_abi::is_occupied(sampler.sample(p.x, p.y, p.z)) {
                '#'
            } else {
                '.'
            }
        })
        .collect();
    println!(
        "  along the ray through ({x},{y}), a quarter pixel apart about t {:.6}: {line}",
        hit.t
    );
}

fn write_shaded(view: &CastView, image: &CastImage, path: &PathBuf) -> Result<()> {
    let light = DVec3::new(-0.45, 0.55, 1.0).normalize();
    let mut out = image::RgbImage::from_pixel(image.width, image.height, image::Rgb([24, 26, 30]));
    for (hit, pixel) in image.hits.iter().zip(out.pixels_mut()) {
        let Some(hit) = hit else { continue };
        let normal = hit.normal.as_dvec3();
        let normal = DVec3::new(
            normal.dot(view.right),
            normal.dot(view.up),
            -normal.dot(view.forward),
        );
        let level = (0.18 + 0.82 * normal.dot(light).max(0.0)).sqrt();
        *pixel = image::Rgb([0.80, 0.82, 0.86].map(|c: f64| (c * level * 255.0).round() as u8));
    }
    out.save(path)
        .with_context(|| format!("writing {}", path.display()))
}
