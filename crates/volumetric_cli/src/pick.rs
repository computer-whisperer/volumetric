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
use view_core::crop::{CropOptions, crop, picture_of};
use view_core::measure::{Plane, cast, chart, fit_feature, plane_of, triangulate_picks};
use volumetric::AssetTypeHint;
use volumetric_abi::subspace::{Subspace, decode_subspace};
use volumetric_abi::viewset::{CameraModel, PickRole, View, ViewSet};

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

// Measurement kernels use f64; routing their coordinates through the
// renderer's f32 parser loses small changes during iterative fitting.
fn parse_coordinates<const N: usize>(s: &str, what: &str) -> Result<[f64; N]> {
    let values = s
        .split(',')
        .map(|part| part.trim().parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("Invalid {what}: expected {N} comma-separated numbers"))?;
    if values.iter().any(|v| !v.is_finite()) {
        bail!("Invalid {what}: coordinates must be finite");
    }
    values.try_into().map_err(|v: Vec<f64>| {
        anyhow!(
            "Invalid {what}: expected {N} comma-separated numbers, got {}",
            v.len()
        )
    })
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
    let picture = picture_of(&set, view)?;
    let (w, h) = args
        .size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .filter(|(w, h)| *w > 0 && *h > 0)
        .context("--size must be WxH in pixels")?;
    let options = CropOptions {
        centre: parse_coordinates::<2>(&args.center, "--center")?,
        size: (w, h),
        scale: args.scale,
        grid: args.grid,
        marks: args
            .marks
            .iter()
            .map(|m| parse_coordinates::<2>(m, "--mark"))
            .collect::<Result<_>>()?,
        world_marks: args
            .world_marks
            .iter()
            .map(|m| parse_coordinates::<3>(m, "--mark-world"))
            .collect::<Result<_>>()?,
    };
    let out = crop(view, camera, &picture, &options)?;
    for (world, pixel) in options.world_marks.iter().zip(&out.projected) {
        match pixel {
            Some(pixel) => println!(
                "world ({:.4}, {:.4}, {:.4}) -> pixel ({:.1}, {:.1})",
                world[0], world[1], world[2], pixel[0], pixel[1]
            ),
            None => println!(
                "world ({:.4}, {:.4}, {:.4}) is behind the camera or the view is unposed",
                world[0], world[1], world[2]
            ),
        }
    }
    std::fs::write(&args.output, out.image.to_png()?)
        .with_context(|| format!("Failed to write {}", args.output.display()))?;
    println!(
        "Wrote {}: pixels ({}, {})..({}, {}) of {} at {}x, {}x{}",
        args.output.display(),
        out.origin.0,
        out.origin.1,
        out.end.0,
        out.end.1,
        args.view,
        out.scale,
        out.image.width,
        out.image.height
    );
    if args.grid > 0 {
        println!(
            "grid every {} px; verticals at u = {}; horizontals at v = {} (every fifth brighter)",
            args.grid,
            out.verticals
                .iter()
                .map(|u| u.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            out.horizontals
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

    /// Record the one --pixel as a feature pick of this name in the view
    /// (replacing a pick of that name there) and save the set: a .vviews
    /// input is rewritten (or --output), a project's asset is updated
    #[arg(long)]
    pub record: Option<String>,

    /// With --record: the pick is a check (held out of triangulation and
    /// compared against the fitted point), not a fit
    #[arg(long)]
    pub check: bool,

    /// With --record: write the set here instead of over the input
    #[arg(short, long)]
    pub output: Option<PathBuf>,

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
    let plane: Option<Plane> = if let Some(z) = args.plane_z {
        Some(Plane::at_z(z))
    } else if let Some(id) = &args.plane {
        Some(plane_of(&load_subspace(&args.input, id)?).map_err(|e| anyhow!("asset '{id}': {e}"))?)
    } else if let (Some(p), Some(n)) = (&args.plane_point, &args.plane_normal) {
        Some(Plane {
            point: parse_coordinates::<3>(p, "--plane-point")?,
            normal: parse_coordinates::<3>(n, "--plane-normal")?,
        })
    } else {
        None
    };
    if let Some(name) = &args.record {
        let [pixel] = args.pixels.as_slice() else {
            bail!("--record names one pick: give exactly one --pixel");
        };
        let pixel = parse_coordinates::<2>(pixel, "--pixel")?;
        let role = if args.check {
            PickRole::Check
        } else {
            PickRole::Fit
        };
        let mut set = set.clone();
        set.record_pick(&view.id, name, pixel, role)
            .map_err(anyhow::Error::msg)?;
        let where_ = crate::views::store_viewset(
            &set,
            &args.input,
            args.views.as_deref(),
            args.output.as_deref(),
        )?;
        println!(
            "recorded {} pick {name:?} at ({:.1}, {:.1}) in view {} -> {where_}",
            if args.check { "check" } else { "fit" },
            pixel[0],
            pixel[1],
            view.id
        );
        if plane.is_none() {
            return Ok(());
        }
    }
    if !args.pixels.is_empty() && plane.is_none() {
        bail!(
            "casting pixels needs a plane: --plane-z, --plane, or --plane-point with --plane-normal"
        );
    }
    let mut picked = Vec::new();
    for text in &args.pixels {
        let pixel = parse_coordinates::<2>(text, "--pixel")?;
        let plane = plane.as_ref().expect("checked above");
        let hit = cast(view, camera, pixel, plane).map_err(anyhow::Error::msg)?;
        picked.push(Picked {
            pixel,
            world: hit.world,
            depth: hit.depth,
            chart: frame.as_ref().map(|f| chart(f, hit.world)),
        });
    }
    let mut projected = Vec::new();
    for text in &args.points {
        let world = parse_coordinates::<3>(text, "--point")?;
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

    /// Fit this recorded feature from its picks (repeatable; see
    /// `view-pick --record`)
    #[arg(long = "feature")]
    pub features: Vec<String>,

    /// Fit every recorded feature
    #[arg(long)]
    pub all_features: bool,

    /// The same feature seen in a view, as VIEW:u,v (at least two)
    #[arg(long = "ray")]
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
pub fn run_view_triangulate(args: ViewTriangulateArgs) -> Result<()> {
    let set = load_viewset(&args.input, args.views.as_deref())?;
    let frame = match &args.frame {
        Some(id) => Some(load_subspace(&args.input, id)?),
        None => None,
    };
    if args.all_features || !args.features.is_empty() {
        if !args.rays.is_empty() {
            bail!("--ray and recorded features are different inputs; give one");
        }
        return fit_recorded(&set, &args, frame.as_ref());
    }
    if args.rays.len() < 2 {
        bail!("triangulation needs at least two --ray VIEW:u,v (or --feature / --all-features)");
    }
    let mut labels = Vec::new();
    for text in &args.rays {
        let (id, pixel) = text
            .split_once(':')
            .with_context(|| format!("--ray must be VIEW:u,v, got '{text}'"))?;
        let pixel = parse_coordinates::<2>(pixel, "--ray pixel")?;
        labels.push((id.to_string(), pixel));
    }
    let (point, gaps) = triangulate_picks(&set, &labels).map_err(anyhow::Error::msg)?;
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

/// Fit recorded features and print each with its picks' misses and errors.
fn fit_recorded(set: &ViewSet, args: &ViewTriangulateArgs, frame: Option<&Subspace>) -> Result<()> {
    let names: Vec<String> = if args.all_features {
        set.picks().keys().cloned().collect()
    } else {
        args.features.clone()
    };
    if names.is_empty() {
        bail!("no feature picks recorded in the set");
    }
    #[derive(Serialize)]
    struct Out {
        name: String,
        world: [f64; 3],
        #[serde(skip_serializing_if = "Option::is_none")]
        chart: Option<Vec<f64>>,
        max_gap: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_check_px: Option<f64>,
        picks: Vec<view_core::measure::PickReport>,
    }
    let mut fits = Vec::new();
    for name in &names {
        // Every recorded feature: one that cannot be fitted yet is noted,
        // not fatal; a named feature must fit.
        let fit = match fit_feature(set, name) {
            Ok(fit) => fit,
            Err(err) if args.all_features => {
                eprintln!("{name}: {err}");
                continue;
            }
            Err(err) => bail!(err),
        };
        fits.push(Out {
            name: fit.name.clone(),
            world: fit.world,
            chart: frame.map(|f| chart(f, fit.world)),
            max_gap: fit.max_gap(),
            max_check_px: fit.max_check_px(),
            picks: fit.picks,
        });
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&fits)?);
        return Ok(());
    }
    for fit in &fits {
        println!(
            "{}: world ({:.4}, {:.4}, {:.4}){}; worst ray miss {:.2} mm{}",
            fit.name,
            fit.world[0],
            fit.world[1],
            fit.world[2],
            match &fit.chart {
                Some(c) => format!(
                    "  chart ({})",
                    c.iter()
                        .map(|v| format!("{:+.4}", v))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                None => String::new(),
            },
            fit.max_gap * 1000.0,
            match fit.max_check_px {
                Some(px) => format!(", worst check {px:.1} px"),
                None => String::new(),
            }
        );
        for pick in &fit.picks {
            println!(
                "  {} {:?} ({:.1}, {:.1}): {}",
                pick.view,
                pick.role,
                pick.pixel[0],
                pick.pixel[1],
                match (pick.gap, pick.error_px) {
                    (Some(gap), Some(px)) => format!(
                        "ray passes {:.2} mm from the point, reprojects {px:.1} px off",
                        gap * 1000.0
                    ),
                    (None, Some(px)) => format!("reprojects {px:.1} px off"),
                    _ => "behind the camera".to_string(),
                }
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurement_coordinates_preserve_f64_precision() {
        assert_eq!(
            parse_coordinates::<3>("1000000.001, -0.000000001, 6191.123456789", "point").unwrap(),
            [1000000.001_f64, -0.000000001, 6191.123456789]
        );
        for bad in ["NaN,0", "inf,0", "1", "1,2,3", "x,0"] {
            assert!(parse_coordinates::<2>(bad, "pixel").is_err());
        }
    }
}
