//! `cast-bench`: runs the direct caster on a model and reports what it
//! cost and how well the result agrees with the model and the mesher.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
use clap::Parser;
use glam::{DVec3, Vec3};
use volumetric::direct_cast::{
    CastImage, CastOptions, CastProjection, CastRun, CastView, DirectCast,
};
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

    /// Magnification of each successive view over the last
    #[arg(long, default_value_t = 1.0)]
    pub zoom_step: f64,

    /// Run only this many passes of each view but the last, as a viewport
    /// does while the view keeps moving
    #[arg(long)]
    pub passes: Option<usize>,

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

    /// Also write the last view as the viewport would draw it: the
    /// surfels shown for it, splatted as discs
    #[arg(long)]
    pub splat: Option<PathBuf>,

    /// Check every hit's normal of the last view against the model, by
    /// sampling a sphere of one pixel about the hit; hits whose normal is
    /// more than 60 degrees off are drawn red in the output image
    #[arg(long)]
    pub check_normals: bool,
}

fn view_for(
    args: &CastBenchArgs,
    min: DVec3,
    max: DVec3,
    yaw_degrees: f64,
    pitch_degrees: f64,
    zoom: f64,
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
                half_height: radius * 1.05 / fit / zoom,
            },
        )
    } else {
        (
            radius * 1.05 / (TAN_HALF_FOV.atan().sin() * fit),
            CastProjection::Perspective {
                tan_half_fov_y: TAN_HALF_FOV / zoom,
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
        let view = view_for(
            &args,
            min,
            max,
            yaw + index as f64 * args.yaw_step,
            pitch,
            args.zoom * args.zoom_step.powi(index as i32),
        );
        let first_pass = cast.history.len();
        let start = std::time::Instant::now();
        let mut run = CastRun::new(&options);
        let mut steps = 0;
        let image = loop {
            let last = index + 1 == args.views;
            if !last && args.passes.is_some_and(|passes| steps >= passes) {
                break run.into_image();
            }
            if !run
                .step(&mut cast, &sampler, &view, &never)
                .expect("never cancelled")
            {
                break run.into_image();
            }
            steps += 1;
        };
        let total: volumetric::direct_cast::PassStats =
            cast.history[first_pass..]
                .iter()
                .fold(Default::default(), |mut sum, (_, _, stats)| {
                    sum += *stats;
                    sum
                });
        let seconds = start.elapsed().as_secs_f64();

        println!("View {} of {}", index + 1, args.views);
        println!(
            "  {:<9} {:>7} {:>9} {:>11} {:>9} {:>9} {:>10} {:>9}",
            "pass", "spacing", "rays", "samples", "hits", "new", "backed off", "seconds"
        );
        for (mode, spacing, stats) in &cast.history[first_pass..] {
            println!(
                "  {:<9} {:>7} {:>9} {:>11} {:>9} {:>9} {:>10} {:>9.3}",
                format!("{mode:?}"),
                spacing,
                stats.rays,
                stats.samples,
                stats.hits,
                stats.fresh,
                stats.backed_off,
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
            total.samples as f64
                / image
                    .as_ref()
                    .map_or(0, |image| image.hits.iter().flatten().count())
                    .max(1) as f64,
        );
        last = Some((view, image));
    }
    let (view, image) = last.context("--views must be at least 1")?;
    let image = image.context("the last view did not reach its final spacing")?;

    // Every surfel should sit on the surface: outside the model a little
    // way along its normal, inside the same way back. One that does not
    // is looked for along its normal, out to four radii either way.
    let surfels = cast.surfels_within((min + max) * 0.5, (max - min).length());
    let mut off = 0usize;
    let mut adrift = [0usize; 4];
    let mut farthest = 0.0f64;
    let (mut afloat, mut askew) = (0usize, 0usize);
    let sphere = fibonacci_sphere(60);
    for surfel in &surfels {
        let p = Vec3::from(surfel.position).as_dvec3();
        let normal = Vec3::from(surfel.normal).as_dvec3();
        let radius = surfel.radius as f64;
        let inside = |p: DVec3| volumetric_abi::is_occupied(sampler.sample(p.x, p.y, p.z));
        let step = normal * (radius * 0.25);
        if !(inside(p + step) || !inside(p - step)) {
            continue;
        }
        off += 1;
        // The nearest change of occupancy along the normal, in radii.
        let here = inside(p);
        let mut found = None;
        for k in 1..=16 {
            let d = k as f64 * 0.25;
            if inside(p + normal * (d * radius)) != here
                || inside(p - normal * (d * radius)) != here
            {
                found = Some(d);
                break;
            }
        }
        match found {
            Some(d) if d <= 0.5 => adrift[0] += 1,
            Some(d) if d <= 1.0 => adrift[1] += 1,
            Some(d) if d <= 2.0 => adrift[2] += 1,
            Some(d) => {
                adrift[3] += 1;
                farthest = farthest.max(d);
            }
            None => {
                adrift[3] += 1;
                farthest = f64::INFINITY;
                // Off the surface in every direction, or only along the
                // normal?
                let clear = |d: f64| {
                    sphere
                        .iter()
                        .all(|dir| inside(p + *dir * (d * radius)) == here)
                };
                let floating = [0.5, 1.0, 2.0, 4.0]
                    .into_iter()
                    .take_while(|d| clear(*d))
                    .last();
                match floating {
                    Some(d) => {
                        afloat += 1;
                        if afloat <= 8 {
                            println!(
                                "  afloat: radius {:.3e} at [{:.6} {:.6} {:.6}] normal [{:.3} {:.3} {:.3}]{} is {} in {} radii clear of surface in every direction",
                                radius,
                                p.x,
                                p.y,
                                p.z,
                                normal.x,
                                normal.y,
                                normal.z,
                                if surfel.guessed { " (guessed)" } else { "" },
                                if here { "inside" } else { "outside" },
                                d
                            );
                        }
                    }
                    None => askew += 1,
                }
            }
        }
    }
    println!(
        "Record: {} nodes, {} surfels at all levels; {} ({:.2}%) do not straddle the surface a quarter radius either way",
        cast.node_count(),
        surfels.len(),
        off,
        100.0 * off as f64 / surfels.len().max(1) as f64
    );
    println!(
        "  of those, the surface lies along the normal within half a radius for {}, a radius for {}, two for {}, farther or not found for {} (farthest {} radii)",
        adrift[0], adrift[1], adrift[2], adrift[3], farthest
    );
    println!(
        "  of the not found, {afloat} are clear of surface in every direction (afloat), {askew} have surface beside them (normal askew)"
    );

    if let Some(depth) = args.mesh_depth {
        compare_with_mesh(&wasm_bytes, depth, &view, &image)?;
    }
    if let Some(probe) = &args.probe {
        let (x, y) = probe
            .split_once(',')
            .and_then(|(x, y)| Some((x.trim().parse::<u32>().ok()?, y.trim().parse::<u32>().ok()?)))
            .context("--probe takes \"x,y\" in pixels")?;
        probe_pixel(&sampler, &cast, &view, &image, x, y);
    }
    let wrong = args
        .check_normals
        .then(|| check_normals(&sampler, &view, &image));
    if let Some(path) = &args.output {
        write_shaded(&view, &image, wrong.as_deref(), path)?;
        println!("Wrote {}", path.display());
    }
    if let Some(path) = &args.splat {
        let shown = cast.surfels_shown(&view, args.spacing, &image);
        let coarse = shown
            .iter()
            .filter(|s| {
                let pitch = args.spacing * view.footprint_at(Vec3::from(s.position).as_dvec3());
                s.radius as f64 > 0.75 * pitch * 1.2
            })
            .count();
        let in_frame: Vec<_> = shown
            .iter()
            .filter(|s| {
                let p = Vec3::from(s.position).as_dvec3();
                let pitch = args.spacing * view.footprint_at(p);
                s.radius as f64 > 0.75 * pitch * 1.2
                    && view
                        .pixel_of(p)
                        .is_some_and(|(x, y)| x < args.width as f64 && y < args.height as f64)
            })
            .copied()
            .collect();
        println!(
            "Shown: {} surfels, {} of them coarser than the view's pixels ({} in frame), {} guessed",
            shown.len(),
            coarse,
            in_frame.len(),
            shown.iter().filter(|s| s.guessed).count()
        );
        write_splat(&view, &shown, None, path)?;
        println!("Wrote {}", path.display());
        let coarse_path = path.with_extension("coarse.png");
        write_splat(&view, &in_frame, None, &coarse_path)?;
        println!(
            "Wrote {} (the coarse ones in frame alone)",
            coarse_path.display()
        );
        if args.check_normals {
            // The shown surfels in frame whose normal is more than 60
            // degrees off the model's, drawn red over the rest.
            let sphere = fibonacci_sphere(60);
            let askew: Vec<bool> = shown
                .iter()
                .map(|s| {
                    let p = Vec3::from(s.position).as_dvec3();
                    let normal = Vec3::from(s.normal).as_dvec3();
                    if normal.dot(view.direction_at(p)) >= 0.0
                        || !view
                            .pixel_of(p)
                            .is_some_and(|(x, y)| x < args.width as f64 && y < args.height as f64)
                    {
                        return false;
                    }
                    let radius = s.radius as f64;
                    let mut oracle = DVec3::ZERO;
                    for d in &sphere {
                        let q = p + *d * radius;
                        let inside = volumetric_abi::is_occupied(sampler.sample(q.x, q.y, q.z));
                        oracle += if inside { -*d } else { *d };
                    }
                    oracle
                        .try_normalize()
                        .is_some_and(|o| o.dot(normal) < 60f64.to_radians().cos())
                })
                .collect();
            println!(
                "  {} shown surfels in frame, facing the view, have a normal more than 60 degrees off the model's ({} of them guessed)",
                askew.iter().filter(|a| **a).count(),
                askew
                    .iter()
                    .zip(&shown)
                    .filter(|(a, s)| **a && s.guessed)
                    .count()
            );
            let askew_path = path.with_extension("askew.png");
            write_splat(&view, &shown, Some(&askew), &askew_path)?;
            println!("Wrote {} (those in red)", askew_path.display());
        }
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
    cast: &DirectCast,
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
                    "  ({px},{py}) t {:.6} normal [{:.3} {:.3} {:.3}]{} at [{:.6} {:.6} {:.6}]",
                    hit.t,
                    hit.normal.x,
                    hit.normal.y,
                    hit.normal.z,
                    if hit.guessed { " (guessed)" } else { "" },
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
    let pixel = view.footprint_at(hit.position);
    if let Some(own) = image.hit(x, y) {
        // The normal found again, with its probes written out.
        let ray = view.ray(
            (x as f64 + 0.5) * image.spacing,
            (y as f64 + 0.5) * image.spacing,
        );
        let mut log = Vec::new();
        let mut samples = 0;
        let (normal, guessed) = DirectCast::normal_at(
            sampler,
            view,
            &ray,
            own.t,
            pixel * image.spacing,
            &mut samples,
            Some(&mut log),
        );
        println!(
            "  the normal at ({x},{y}) found again: [{:.3} {:.3} {:.3}]{} from {samples} samples:",
            normal.x,
            normal.y,
            normal.z,
            if guessed { " (guessed)" } else { "" }
        );
        for line in log {
            println!("    {line}");
        }
    }
    println!("  surfels within 2 px of the hit at ({x},{y}), radius in px:");
    let mut around = cast.surfels_within(hit.position, 2.0 * pixel);
    around.sort_by(|a, b| a.radius.total_cmp(&b.radius));
    for s in &around {
        let p = Vec3::from(s.position).as_dvec3();
        println!(
            "    radius {:.2} normal [{:.3} {:.3} {:.3}]{} {:.2} px away, {:.2} px in depth",
            s.radius as f64 / pixel,
            s.normal[0],
            s.normal[1],
            s.normal[2],
            if s.guessed { " (guessed)" } else { "" },
            (p - hit.position).length() / pixel,
            (p - hit.position).dot(view.direction_at(hit.position)) / pixel,
        );
    }
    let ray = view.ray(
        (x as f64 + 0.5) * image.spacing,
        (y as f64 + 0.5) * image.spacing,
    );
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

/// `n` directions spread evenly over the sphere.
fn fibonacci_sphere(n: usize) -> Vec<DVec3> {
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let phi = i as f64 * std::f64::consts::PI * (3.0 - 5f64.sqrt());
            DVec3::new(r * phi.cos(), r * phi.sin(), z)
        })
        .collect()
}

/// The model's own normal at each hit, from the occupancy of 60 points on
/// a sphere one pixel wide about it (the empty ones pull the normal
/// their way), against the normal the cast found. Returns which hits are
/// more than 60 degrees off; at an edge the two differ by up to 45.
fn check_normals(
    sampler: &(impl ParallelModelSampler + Sync),
    view: &CastView,
    image: &CastImage,
) -> Vec<bool> {
    let directions = fibonacci_sphere(60);
    let angle_of = |hit: &volumetric::direct_cast::RayHit| -> Option<f64> {
        let radius = view.footprint_at(hit.position) * image.spacing;
        let mut oracle = DVec3::ZERO;
        for d in &directions {
            let p = hit.position + *d * radius;
            let inside = volumetric_abi::is_occupied(sampler.sample(p.x, p.y, p.z));
            oracle += if inside { -*d } else { *d };
        }
        let oracle = oracle.try_normalize()?;
        Some(
            oracle
                .dot(hit.normal.as_dvec3())
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees(),
        )
    };
    let rows = image.height as usize;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let per = rows.div_ceil(threads);
    let angles: Vec<Vec<Option<f64>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                let angle_of = &angle_of;
                scope.spawn(move || {
                    (k * per..((k + 1) * per).min(rows))
                        .flat_map(|row| {
                            (0..image.width)
                                .map(move |x| image.hit(x, row as u32).and_then(angle_of))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let angles: Vec<Option<f64>> = angles.into_iter().flatten().collect();
    let checked = angles.iter().flatten().count();
    let over = |limit: f64| angles.iter().flatten().filter(|a| **a > limit).count();
    println!(
        "Normals: of {checked} hits, {} ({:.3}%) are more than 30 degrees off the model's, {} ({:.3}%) more than 60, {} ({:.3}%) more than 90",
        over(30.0),
        100.0 * over(30.0) as f64 / checked.max(1) as f64,
        over(60.0),
        100.0 * over(60.0) as f64 / checked.max(1) as f64,
        over(90.0),
        100.0 * over(90.0) as f64 / checked.max(1) as f64,
    );
    let guessed = image.hits.iter().flatten().filter(|h| h.guessed).count();
    println!("  {guessed} hits have a guessed normal");
    let mut listed = 0;
    for (index, angle) in angles.iter().enumerate() {
        let Some(angle) = angle else { continue };
        let hit = image.hits[index].unwrap();
        if *angle > 60.0 && !hit.guessed {
            if listed < 12 {
                println!(
                    "  found normal {:.0} degrees off at ({},{})",
                    angle,
                    index as u32 % image.width,
                    index as u32 / image.width
                );
            }
            listed += 1;
        }
    }
    println!("  {listed} hits with a found normal are more than 60 degrees off");
    angles.iter().map(|a| a.is_some_and(|a| a > 60.0)).collect()
}

/// Draws the surfels as the viewport does, as discs with a depth test,
/// shaded from their normals; back-facing ones are not drawn.
fn write_splat(
    view: &CastView,
    surfels: &[volumetric::direct_cast::Surfel],
    marked: Option<&[bool]>,
    path: &PathBuf,
) -> Result<()> {
    let light = DVec3::new(-0.45, 0.55, 1.0).normalize();
    let (w, h) = (view.width as usize, view.height as usize);
    let mut depth = vec![f64::INFINITY; w * h];
    let mut out = image::RgbImage::from_pixel(view.width, view.height, image::Rgb([24, 26, 30]));
    for (index, s) in surfels.iter().enumerate() {
        let position = Vec3::from(s.position).as_dvec3();
        let normal = Vec3::from(s.normal).as_dvec3();
        let dir = view.direction_at(position);
        if normal.dot(dir) >= 0.0 {
            continue;
        }
        let Some((cx, cy)) = view.pixel_of(position) else {
            continue;
        };
        let t = match view.projection {
            CastProjection::Perspective { .. } => (position - view.eye).length(),
            CastProjection::Orthographic { .. } => (position - view.eye).dot(view.forward),
        };
        let radius = s.radius as f64 / view.footprint_at(position);
        let shade = {
            let n = DVec3::new(
                normal.dot(view.right),
                normal.dot(view.up),
                -normal.dot(view.forward),
            );
            (0.18 + 0.82 * n.dot(light).max(0.0)).sqrt()
        };
        let marked = marked.is_some_and(|marked| marked[index]);
        let colour = if marked {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([0.80, 0.82, 0.86].map(|c: f64| (c * shade * 255.0).round() as u8))
        };
        // A marked disc is drawn over whatever is at its depth, and at
        // least a pixel and a half wide, to be seen.
        let r = if marked {
            radius.max(1.5)
        } else {
            radius.max(0.5)
        };
        let t = if marked { t - 1e-6 } else { t };
        let (x0, x1) = (
            (cx - r).floor().max(0.0) as usize,
            ((cx + r).ceil() as usize).min(w.saturating_sub(1)),
        );
        let (y0, y1) = (
            (cy - r).floor().max(0.0) as usize,
            ((cy + r).ceil() as usize).min(h.saturating_sub(1)),
        );
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                if dx * dx + dy * dy > r * r {
                    continue;
                }
                let slot = &mut depth[y * w + x];
                if t < *slot {
                    *slot = t;
                    out.put_pixel(x as u32, y as u32, colour);
                }
            }
        }
    }
    out.save(path)
        .with_context(|| format!("writing {}", path.display()))
}

fn write_shaded(
    view: &CastView,
    image: &CastImage,
    wrong: Option<&[bool]>,
    path: &PathBuf,
) -> Result<()> {
    let light = DVec3::new(-0.45, 0.55, 1.0).normalize();
    let mut out = image::RgbImage::from_pixel(image.width, image.height, image::Rgb([24, 26, 30]));
    for (index, (hit, pixel)) in image.hits.iter().zip(out.pixels_mut()).enumerate() {
        let Some(hit) = hit else { continue };
        if wrong.is_some_and(|wrong| wrong[index]) {
            *pixel = image::Rgb([255, 0, 0]);
            continue;
        }
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
