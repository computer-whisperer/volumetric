//! `splat-list`: what a trained splat holds.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use serde::Serialize;
use volumetric::splat::{Splat, decode_splat, quantile_sorted};
use volumetric::{AssetTypeHint, LoadedAsset};

use crate::views::project_assets;

#[derive(Parser, Debug)]
pub struct SplatListArgs {
    /// A .vsplat file or a .vproj project
    #[arg(short, long)]
    pub input: PathBuf,

    /// For projects with several splats: which one
    #[arg(long)]
    pub asset: Option<String>,

    #[arg(long)]
    pub json: bool,
}

/// The 5th, 50th and 95th percentiles of a quantity over the primitives.
#[derive(Serialize)]
pub struct Quantiles {
    pub p5: f64,
    pub p50: f64,
    pub p95: f64,
}

fn quantiles(mut values: Vec<f32>) -> Quantiles {
    values.sort_by(f32::total_cmp);
    Quantiles {
        p5: f64::from(quantile_sorted(&values, 0.05)),
        p50: f64::from(quantile_sorted(&values, 0.5)),
        p95: f64::from(quantile_sorted(&values, 0.95)),
    }
}

#[derive(Serialize)]
pub struct SplatSummary {
    pub schema: u32,
    pub kind: String,
    pub sh_degree: u32,
    pub count: usize,
    pub normals: bool,
    /// Per-axis bounds of the centres at the 1st and 99th percentiles.
    pub bounds_p1: [f64; 3],
    pub bounds_p99: [f64; 3],
    pub bounds_min: [f64; 3],
    pub bounds_max: [f64; 3],
    pub opacity: Quantiles,
    /// The largest standard deviation per primitive, millimetres.
    pub extent_mm: Quantiles,
    /// The smallest standard deviation per primitive, millimetres.
    pub thickness_mm: Quantiles,
    /// Primitives at or above half opacity.
    pub opaque: usize,
    pub up: [f64; 3],
    pub provenance: volumetric::viewset::Provenance,
    pub views_hash: String,
    pub training: String,
}

pub fn summarize(splat: &Splat) -> SplatSummary {
    let n = splat.len();
    let (bounds_p1, bounds_p99) = splat
        .bounds_quantile(0.01, 0.99)
        .unwrap_or(([0.0; 3], [0.0; 3]));
    let (bounds_min, bounds_max) = splat
        .bounds_quantile(0.0, 1.0)
        .unwrap_or(([0.0; 3], [0.0; 3]));
    let opacities: Vec<f32> = (0..n).map(|i| splat.opacity(i)).collect();
    let opaque = opacities.iter().filter(|o| **o >= 0.5).count();
    let extents: Vec<f32> = (0..n).map(|i| splat.extent(i) * 1000.0).collect();
    let thickness: Vec<f32> = (0..n)
        .map(|i| {
            let s = splat.scale(i);
            s[0].min(s[1]).min(s[2]) * 1000.0
        })
        .collect();
    SplatSummary {
        schema: splat.schema,
        kind: splat.kind.name().to_string(),
        sh_degree: splat.sh_degree,
        count: n,
        normals: !splat.normals.is_empty(),
        bounds_p1,
        bounds_p99,
        bounds_min,
        bounds_max,
        opacity: quantiles(opacities),
        extent_mm: quantiles(extents),
        thickness_mm: quantiles(thickness),
        opaque,
        up: splat.world.up,
        provenance: splat.provenance.clone(),
        views_hash: splat.views_hash.clone(),
        training: splat.training.clone(),
    }
}

/// The splat asset named by `wanted` among `assets`, or the only one.
pub(crate) fn find_splat(assets: &[LoadedAsset], wanted: Option<&str>) -> Result<Splat> {
    let splats: Vec<&LoadedAsset> = assets
        .iter()
        .filter(|a| a.type_hint() == Some(AssetTypeHint::Splat))
        .collect();
    let ids = || {
        splats
            .iter()
            .map(|a| a.id().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let asset = match wanted {
        Some(id) => splats
            .iter()
            .find(|a| a.id() == id)
            .copied()
            .ok_or_else(|| anyhow!("no splat asset '{id}'. Available: {}", ids()))?,
        None => match splats.as_slice() {
            [] => bail!("no splat in the project: import one with splat_import_operator"),
            [only] => only,
            _ => bail!("several splats; choose one with --asset: {}", ids()),
        },
    };
    decode_splat(asset.data()).map_err(|err| anyhow!("asset '{}': {err}", asset.id()))
}

/// A splat from a `.vsplat` file or a project (an imported `.vsplat`
/// asset, else the project's exports after running it).
pub(crate) fn load_splat(input: &Path, asset: Option<&str>) -> Result<Splat> {
    let extension = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if extension == "vproj" {
        let imports = project_assets(input, false)?;
        if imports
            .iter()
            .any(|a| a.type_hint() == Some(AssetTypeHint::Splat))
        {
            return find_splat(&imports, asset);
        }
        return find_splat(&project_assets(input, true)?, asset);
    }
    let bytes =
        std::fs::read(input).with_context(|| format!("Failed to read {}", input.display()))?;
    decode_splat(&bytes).map_err(|err| anyhow!("{} is not a splat: {err}", input.display()))
}

pub fn run_splat_list(args: SplatListArgs) -> Result<()> {
    let splat = load_splat(&args.input, args.asset.as_deref())?;
    let summary = summarize(&splat);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    print_summary(&summary);
    Ok(())
}

fn print_summary(s: &SplatSummary) {
    println!(
        "Splat schema {}: {} {}s, SH degree {}, {} at or above half opacity, normals {}",
        s.schema,
        s.count,
        s.kind,
        s.sh_degree,
        s.opaque,
        if s.normals {
            "stored"
        } else {
            "from the orientation"
        }
    );
    let axis = |name: &str, k: usize| {
        println!(
            "  {name}: {:+.4} .. {:+.4} m (1st–99th percentile), {:+.4} .. {:+.4} in all",
            s.bounds_p1[k], s.bounds_p99[k], s.bounds_min[k], s.bounds_max[k]
        );
    };
    println!("Bounds of the centres:");
    axis("x", 0);
    axis("y", 1);
    axis("z", 2);
    let q = |name: &str, q: &Quantiles, unit: &str| {
        println!(
            "  {name}: p5 {:.3}{unit}, median {:.3}{unit}, p95 {:.3}{unit}",
            q.p5, q.p50, q.p95
        );
    };
    println!("Per primitive:");
    q("opacity", &s.opacity, "");
    q("largest scale", &s.extent_mm, " mm");
    q("smallest scale", &s.thickness_mm, " mm");
    let p = &s.provenance;
    println!(
        "Provenance: session {:?}, field {:?}, setup {:?}, up ({}, {}, {})",
        p.session, p.field, p.setup, s.up[0], s.up[1], s.up[2]
    );
    if !s.views_hash.is_empty() {
        println!("  trained from view set {}", s.views_hash);
    }
    if !s.training.is_empty() {
        println!("  training: {}", s.training);
    }
    for tool in &p.tools {
        println!("  tool: {tool}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric::splat::{SplatKind, encode_splat, sh_rest_per_point};

    #[test]
    fn summarizes_a_file() {
        let mut s = Splat::empty(SplatKind::Surfel2d, 1);
        s.count = 3;
        s.means = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
        s.scales = [(0.01f32).ln(), (0.02f32).ln(), -20.0].repeat(3);
        s.quats = [1.0, 0.0, 0.0, 0.0].repeat(3);
        s.opacities = vec![-10.0, 0.0, 10.0];
        s.sh0 = vec![0.0; 9];
        s.sh_rest = vec![0.0; 3 * sh_rest_per_point(1)];
        s.views_hash = "deadbeef".to_string();
        let dir = std::env::temp_dir().join(format!("splat_list_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.vsplat");
        std::fs::write(&path, encode_splat(&s)).unwrap();
        let summary = summarize(&load_splat(&path, None).unwrap());
        assert_eq!(summary.kind, "surfel");
        assert_eq!(summary.count, 3);
        assert_eq!(summary.opaque, 2);
        assert_eq!(summary.bounds_min, [0.0, 0.0, 0.0]);
        assert_eq!(summary.bounds_max, [2.0, 4.0, 6.0]);
        assert!((summary.extent_mm.p50 - 20.0).abs() < 1e-3);
        assert!((summary.opacity.p50 - 0.5).abs() < 1e-6);
        assert_eq!(summary.views_hash, "deadbeef");
        std::fs::remove_dir_all(&dir).unwrap();
        run_splat_list(SplatListArgs {
            input: dir.join("missing.vsplat"),
            asset: None,
            json: false,
        })
        .unwrap_err();
    }
}
