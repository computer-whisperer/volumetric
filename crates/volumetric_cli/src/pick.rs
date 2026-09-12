//! Measuring in a photograph: `view-crop` writes a magnified crop of a
//! view's original picture with a pixel grid, so a detail can be read at
//! full resolution with coordinates; `view-pick` turns pixels of a view
//! into world points on a plane (and world points into pixels), through
//! the surveyed camera with its lens distortion. Together they are how an
//! agent measures a feature it can see in a picture but the scan blurs: a
//! bolt hole, an edge, a mark.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use serde::Serialize;
use view_core::full_picture;
use view_core::image::{Rgb, decode_rgb};
use volumetric::AssetTypeHint;
use volumetric_abi::subspace::{Subspace, decode_subspace};
use volumetric_abi::viewset::{CameraModel, View, ViewSet};

use crate::views::{load_viewset, project_assets};

#[derive(Parser, Debug)]
pub struct ViewCropArgs {
    /// A .vviews file or a .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// For projects with several view sets: which one
    #[arg(long)]
    pub views: Option<String>,

    /// The view whose original picture to crop
    #[arg(long)]
    pub view: String,

    /// Centre of the crop, u,v in the original's pixels
    #[arg(long, allow_hyphen_values = true)]
    pub center: String,

    /// Size of the crop in the original's pixels, WxH
    #[arg(long, default_value = "600x400")]
    pub size: String,

    /// Magnification (integer, nearest neighbour)
    #[arg(long, default_value_t = 3)]
    pub scale: u32,

    /// Grid spacing in the original's pixels (0 = no grid); every fifth
    /// line is brighter, the lines' coordinates are printed
    #[arg(long, default_value_t = 50)]
    pub grid: u32,

    /// Draw a cross at this pixel of the original (repeatable)
    #[arg(long = "mark", allow_hyphen_values = true)]
    pub marks: Vec<String>,

    /// Draw a cross where this world point x,y,z lands in the picture
    /// (repeatable): the check that a modelled feature sits on the real one
    #[arg(long = "mark-world", allow_hyphen_values = true)]
    pub world_marks: Vec<String>,

    /// Output PNG
    #[arg(short, long)]
    pub output: PathBuf,
}

const GRID: [u8; 3] = [255, 170, 40];
const GRID_MAJOR: [u8; 3] = [255, 230, 120];
const MARK: [u8; 3] = [60, 220, 255];
const WORLD_MARK: [u8; 3] = [255, 70, 200];

fn parse2(s: &str, what: &str) -> Result<[f64; 2]> {
    let v = crate::render::parse_floats(s, 2).with_context(|| format!("Invalid {what}"))?;
    Ok([f64::from(v[0]), f64::from(v[1])])
}

fn parse3(s: &str, what: &str) -> Result<[f64; 3]> {
    let v = crate::render::parse_floats(s, 3).with_context(|| format!("Invalid {what}"))?;
    Ok([f64::from(v[0]), f64::from(v[1]), f64::from(v[2])])
}

/// The view and its camera, posed or not.
fn view_of<'a>(set: &'a ViewSet, id: &str) -> Result<(&'a View, &'a CameraModel)> {
    set.view(id).with_context(|| {
        format!(
            "no view '{id}' in the view set. Views: {}",
            set.views
                .iter()
                .map(|v| v.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

pub fn run_view_crop(args: ViewCropArgs) -> Result<()> {
    let set = load_viewset(&args.input, args.views.as_deref())?;
    let (view, camera) = view_of(&set, &args.view)?;
    let picture = decode_rgb(&full_picture(&set, view)?)?;
    if (picture.width, picture.height) != (camera.width, camera.height) {
        bail!(
            "the picture is {}x{} but the camera is {}x{}",
            picture.width,
            picture.height,
            camera.width,
            camera.height
        );
    }
    let centre = parse2(&args.center, "--center")?;
    let (w, h) = args
        .size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .filter(|(w, h)| *w > 0 && *h > 0)
        .context("--size must be WxH in pixels")?;
    let scale = args.scale.max(1);
    let u0 = (centre[0] - f64::from(w) / 2.0)
        .round()
        .clamp(0.0, f64::from(picture.width - 1)) as u32;
    let v0 = (centre[1] - f64::from(h) / 2.0)
        .round()
        .clamp(0.0, f64::from(picture.height - 1)) as u32;
    let u1 = (u0 + w).min(picture.width);
    let v1 = (v0 + h).min(picture.height);
    let (cw, ch) = ((u1 - u0) * scale, (v1 - v0) * scale);
    let mut out = Rgb::new(cw, ch);
    for y in 0..ch {
        for x in 0..cw {
            out.set(x, y, picture.get(u0 + x / scale, v0 + y / scale));
        }
    }
    // The grid, on the original's multiples so coordinates read directly.
    let mut verticals = Vec::new();
    let mut horizontals = Vec::new();
    if args.grid > 0 {
        let g = args.grid;
        let mut u = u0.div_ceil(g) * g;
        while u < u1 {
            let major = (u / g) % 5 == 0;
            let x = (u - u0) * scale;
            for y in 0..ch {
                out.set(x, y, if major { GRID_MAJOR } else { GRID });
            }
            verticals.push(u);
            u += g;
        }
        let mut v = v0.div_ceil(g) * g;
        while v < v1 {
            let major = (v / g) % 5 == 0;
            let y = (v - v0) * scale;
            for x in 0..cw {
                out.set(x, y, if major { GRID_MAJOR } else { GRID });
            }
            horizontals.push(v);
            v += g;
        }
    }
    let cross = |out: &mut Rgb, p: [f64; 2], colour: [u8; 3]| {
        let x = ((p[0] - f64::from(u0)) * f64::from(scale)).round();
        let y = ((p[1] - f64::from(v0)) * f64::from(scale)).round();
        let arm = 6 * scale as i64;
        for d in -arm..=arm {
            for (px, py) in [(x as i64 + d, y as i64), (x as i64, y as i64 + d)] {
                if px >= 0 && py >= 0 && (px as u32) < cw && (py as u32) < ch && d.abs() > 1 {
                    out.set(px as u32, py as u32, colour);
                }
            }
        }
    };
    for mark in &args.marks {
        cross(&mut out, parse2(mark, "--mark")?, MARK);
    }
    for mark in &args.world_marks {
        let world = parse3(mark, "--mark-world")?;
        match view.project(camera, world) {
            Some(pixel) => {
                println!(
                    "world ({:.4}, {:.4}, {:.4}) -> pixel ({:.1}, {:.1})",
                    world[0], world[1], world[2], pixel[0], pixel[1]
                );
                cross(&mut out, pixel, WORLD_MARK);
            }
            None => println!(
                "world ({:.4}, {:.4}, {:.4}) is behind the camera or the view is unposed",
                world[0], world[1], world[2]
            ),
        }
    }
    std::fs::write(&args.output, out.to_png()?)
        .with_context(|| format!("Failed to write {}", args.output.display()))?;
    println!(
        "Wrote {}: {} pixels ({u0}, {v0})..({u1}, {v1}) of {} at {scale}x, {cw}x{ch}",
        args.output.display(),
        args.view,
        w * h
    );
    if args.grid > 0 {
        println!(
            "grid every {} px; verticals at u = {}; horizontals at v = {} (every fifth brighter)",
            args.grid,
            verticals
                .iter()
                .map(|u| u.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            horizontals
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}

#[derive(Parser, Debug)]
pub struct ViewPickArgs {
    /// A .vviews file or a .vproj project (a project also supplies planes
    /// and frames by asset id)
    #[arg(short, long)]
    pub input: PathBuf,

    /// For projects with several view sets: which one
    #[arg(long)]
    pub views: Option<String>,

    /// The view to look through
    #[arg(long)]
    pub view: String,

    /// A pixel u,v of the original picture to cast onto the plane
    /// (repeatable)
    #[arg(long = "pixel", allow_hyphen_values = true)]
    pub pixels: Vec<String>,

    /// A world point x,y,z to project into the picture (repeatable)
    #[arg(long = "point", allow_hyphen_values = true)]
    pub points: Vec<String>,

    /// The plane the pixels land on: a horizontal plane at this height
    #[arg(long, allow_hyphen_values = true, conflicts_with_all = ["plane", "plane_point"])]
    pub plane_z: Option<f64>,

    /// The plane the pixels land on: a project asset holding a plane
    /// Subspace (a cloud_fit plane, say)
    #[arg(long)]
    pub plane: Option<String>,

    /// The plane the pixels land on: a point on it (with --plane-normal)
    #[arg(long, allow_hyphen_values = true, requires = "plane_normal")]
    pub plane_point: Option<String>,

    #[arg(long, allow_hyphen_values = true)]
    pub plane_normal: Option<String>,

    /// Also give every result in this project asset's chart: a frame or
    /// plane Subspace (coordinates along its basis vectors, then along its
    /// normal for a plane)
    #[arg(long)]
    pub frame: Option<String>,

    #[arg(long)]
    pub json: bool,
}

#[derive(Serialize)]
struct Picked {
    pixel: [f64; 2],
    world: [f64; 3],
    depth: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    chart: Option<Vec<f64>>,
}

#[derive(Serialize)]
struct Projected {
    world: [f64; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pixel: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chart: Option<Vec<f64>>,
}

/// A project's Subspace export or import by id.
fn load_subspace(input: &Path, id: &str) -> Result<Subspace> {
    let extension = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if extension != "vproj" {
        bail!("a plane or frame asset needs a .vproj input");
    }
    let assets = project_assets(input, true)?;
    let asset = assets
        .iter()
        .find(|a| a.id() == id)
        .with_context(|| format!("no asset '{id}' in the project"))?;
    if asset.type_hint() != Some(AssetTypeHint::Subspace) {
        bail!("asset '{id}' is not a Subspace");
    }
    decode_subspace(asset.data()).map_err(|e| anyhow!("asset '{id}': {e}"))
}

fn plane_of(subspace: &Subspace, id: &str) -> Result<([f64; 3], [f64; 3])> {
    if subspace.ambient() != 3 || subspace.rank() != 2 {
        bail!(
            "asset '{id}' is a rank {} subspace in {}-space, not a plane",
            subspace.rank(),
            subspace.ambient()
        );
    }
    let n = subspace.normal().context("the plane has no normal")?;
    Ok((
        [subspace.origin[0], subspace.origin[1], subspace.origin[2]],
        [n[0], n[1], n[2]],
    ))
}

/// Coordinates of `p` in a subspace's chart: along each basis vector,
/// then along the normal for a plane in 3-space.
fn chart(subspace: &Subspace, p: [f64; 3]) -> Vec<f64> {
    let d = [
        p[0] - subspace.origin[0],
        p[1] - subspace.origin[1],
        p[2] - subspace.origin[2],
    ];
    let mut out: Vec<f64> = (0..subspace.rank())
        .map(|i| {
            let b = subspace.basis_vector(i);
            b[0] * d[0] + b[1] * d[1] + b[2] * d[2]
        })
        .collect();
    if subspace.rank() == 2
        && let Some(n) = subspace.normal()
    {
        out.push(n[0] * d[0] + n[1] * d[1] + n[2] * d[2]);
    }
    out
}

pub fn run_view_pick(args: ViewPickArgs) -> Result<()> {
    let set = load_viewset(&args.input, args.views.as_deref())?;
    let (view, camera) = view_of(&set, &args.view)?;
    if view.pose().is_none() {
        bail!("view '{}' is not posed", view.id);
    }
    let frame = match &args.frame {
        Some(id) => {
            let s = load_subspace(&args.input, id)?;
            if s.ambient() != 3 || s.rank() < 2 {
                bail!("asset '{id}' must be a plane or a frame in 3-space");
            }
            Some(s)
        }
        None => None,
    };
    let plane: Option<([f64; 3], [f64; 3])> = if let Some(z) = args.plane_z {
        Some(([0.0, 0.0, z], [0.0, 0.0, 1.0]))
    } else if let Some(id) = &args.plane {
        Some(plane_of(&load_subspace(&args.input, id)?, id)?)
    } else if let (Some(p), Some(n)) = (&args.plane_point, &args.plane_normal) {
        Some((parse3(p, "--plane-point")?, parse3(n, "--plane-normal")?))
    } else {
        None
    };
    if !args.pixels.is_empty() && plane.is_none() {
        bail!(
            "casting pixels needs a plane: --plane-z, --plane, or --plane-point with --plane-normal"
        );
    }
    let eye = view.position().context("the view is not posed")?;
    let mut picked = Vec::new();
    for text in &args.pixels {
        let pixel = parse2(text, "--pixel")?;
        let (p0, n) = plane.expect("checked above");
        let d = view.ray(camera, pixel).context("the view is not posed")?;
        let denom = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
        if denom.abs() < 1e-9 {
            bail!("pixel ({}, {}) looks along the plane", pixel[0], pixel[1]);
        }
        let t =
            (n[0] * (p0[0] - eye[0]) + n[1] * (p0[1] - eye[1]) + n[2] * (p0[2] - eye[2])) / denom;
        if t <= 0.0 {
            bail!(
                "pixel ({}, {}) meets the plane behind the camera",
                pixel[0],
                pixel[1]
            );
        }
        let world = [eye[0] + t * d[0], eye[1] + t * d[1], eye[2] + t * d[2]];
        let depth = view.to_camera(world).map(|c| c[2]).unwrap_or(t);
        picked.push(Picked {
            pixel,
            world,
            depth,
            chart: frame.as_ref().map(|f| chart(f, world)),
        });
    }
    let mut projected = Vec::new();
    for text in &args.points {
        let world = parse3(text, "--point")?;
        projected.push(Projected {
            world,
            pixel: view.project(camera, world),
            chart: frame.as_ref().map(|f| chart(f, world)),
        });
    }
    if args.json {
        #[derive(Serialize)]
        struct Out<'a> {
            view: &'a str,
            picked: Vec<Picked>,
            projected: Vec<Projected>,
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&Out {
                view: &view.id,
                picked,
                projected
            })?
        );
        return Ok(());
    }
    let fmt_chart = |c: &Option<Vec<f64>>| match c {
        Some(c) => format!(
            "  chart ({})",
            c.iter()
                .map(|v| format!("{:+.4}", v))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        None => String::new(),
    };
    for p in &picked {
        println!(
            "pixel ({:.1}, {:.1}) -> world ({:.4}, {:.4}, {:.4}) at depth {:.3} m{}",
            p.pixel[0],
            p.pixel[1],
            p.world[0],
            p.world[1],
            p.world[2],
            p.depth,
            fmt_chart(&p.chart)
        );
    }
    for p in &projected {
        match p.pixel {
            Some(px) => println!(
                "world ({:.4}, {:.4}, {:.4}) -> pixel ({:.1}, {:.1}){}",
                p.world[0],
                p.world[1],
                p.world[2],
                px[0],
                px[1],
                fmt_chart(&p.chart)
            ),
            None => println!(
                "world ({:.4}, {:.4}, {:.4}) is behind the camera",
                p.world[0], p.world[1], p.world[2]
            ),
        }
    }
    Ok(())
}

#[derive(Parser, Debug)]
pub struct ViewTriangulateArgs {
    /// A .vviews file or a .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// For projects with several view sets: which one
    #[arg(long)]
    pub views: Option<String>,

    /// The same feature seen in a view, as VIEW:u,v (at least two)
    #[arg(long = "ray", required = true)]
    pub rays: Vec<String>,

    /// Also give the result in this project asset's chart (a plane or
    /// frame Subspace)
    #[arg(long)]
    pub frame: Option<String>,

    #[arg(long)]
    pub json: bool,
}

/// The point nearest every ray in the least-squares sense, and each
/// ray's distance from it.
pub fn triangulate(rays: &[([f64; 3], [f64; 3])]) -> Option<([f64; 3], Vec<f64>)> {
    // Sum over rays of (I - d dᵀ)(x - o) = 0.
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for (o, d) in rays {
        for i in 0..3 {
            for j in 0..3 {
                let p = if i == j { 1.0 } else { 0.0 } - d[i] * d[j];
                a[i][j] += p;
                b[i] += p * o[j];
            }
        }
    }
    let x = cloud_core::solve(a.iter().map(|r| r.to_vec()).collect(), b.to_vec())?;
    let x = [x[0], x[1], x[2]];
    let gaps = rays
        .iter()
        .map(|(o, d)| {
            let v = [x[0] - o[0], x[1] - o[1], x[2] - o[2]];
            let t = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
            let r = [v[0] - t * d[0], v[1] - t * d[1], v[2] - t * d[2]];
            (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()
        })
        .collect();
    Some((x, gaps))
}

pub fn run_view_triangulate(args: ViewTriangulateArgs) -> Result<()> {
    if args.rays.len() < 2 {
        bail!("triangulation needs at least two --ray VIEW:u,v");
    }
    let set = load_viewset(&args.input, args.views.as_deref())?;
    let frame = match &args.frame {
        Some(id) => Some(load_subspace(&args.input, id)?),
        None => None,
    };
    let mut rays = Vec::new();
    let mut labels = Vec::new();
    for text in &args.rays {
        let (id, pixel) = text
            .split_once(':')
            .with_context(|| format!("--ray must be VIEW:u,v, got '{text}'"))?;
        let (view, camera) = view_of(&set, id)?;
        let pixel = parse2(pixel, "--ray pixel")?;
        let origin = view
            .position()
            .with_context(|| format!("view '{id}' is not posed"))?;
        let direction = view.ray(camera, pixel).context("unposed")?;
        rays.push((origin, direction));
        labels.push((id.to_string(), pixel));
    }
    let (point, gaps) = triangulate(&rays).context("the rays are parallel")?;
    let chart_coords = frame.as_ref().map(|f| chart(f, point));
    if args.json {
        #[derive(Serialize)]
        struct Out {
            world: [f64; 3],
            gaps: Vec<f64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            chart: Option<Vec<f64>>,
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&Out {
                world: point,
                gaps,
                chart: chart_coords
            })?
        );
        return Ok(());
    }
    println!(
        "world ({:.4}, {:.4}, {:.4}){}",
        point[0],
        point[1],
        point[2],
        match &chart_coords {
            Some(c) => format!(
                "  chart ({})",
                c.iter()
                    .map(|v| format!("{:+.4}", v))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None => String::new(),
        }
    );
    for ((id, pixel), gap) in labels.iter().zip(&gaps) {
        println!(
            "  {id} ({:.1}, {:.1}): ray passes {:.1} mm from the point",
            pixel[0],
            pixel[1],
            gap * 1000.0
        );
    }
    Ok(())
}
