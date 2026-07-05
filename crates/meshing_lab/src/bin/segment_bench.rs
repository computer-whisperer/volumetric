//! Region segmentation benchmark.
//!
//! Questions under test, all against closed-form oracle truth on real mesher
//! output:
//! - Does seeded region growing recover exactly the true smooth faces
//!   (box 6, sphere 1, cylinder 3), with no fragmentation and no bridging?
//! - Are claimed labels pure (majority face_id agreement)?
//! - Is the unclaimed feature zone thin and confined to true features?
//!
//! Usage: segment_bench [depth...] [--render <dir>]

use glam::DVec3;
use meshing_lab::harness::{BUCKET_NAMES, MeshedShape, mesh_shape};
use meshing_lab::oracle::{
    BoxShape, CylinderShape, OracleShape, Rotated, SphereShape, standard_rotation,
};
use meshing_lab::render::{RenderConfig, region_color, render_png};
use meshing_lab::segmentation::{Segmentation, SegmentationConfig, segment_regions};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    let mut depths: Vec<usize> = Vec::new();
    let mut render_dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--render" {
            render_dir = Some(PathBuf::from(
                args.next().expect("--render requires a directory"),
            ));
        } else {
            depths.push(arg.parse().expect("depth args must be integers"));
        }
    }
    if depths.is_empty() {
        depths = vec![3, 4];
    }
    if let Some(dir) = &render_dir {
        std::fs::create_dir_all(dir).expect("create render dir");
    }

    let shapes: Vec<(Box<dyn OracleShape>, usize, &str)> = vec![
        (
            Box::new(Rotated::new(
                BoxShape {
                    half: DVec3::splat(0.5),
                },
                standard_rotation(),
            )),
            6,
            "box",
        ),
        (Box::new(SphereShape { radius: 0.5 }), 1, "sphere"),
        (
            Box::new(Rotated::new(
                CylinderShape {
                    radius: 0.4,
                    half_height: 0.5,
                },
                standard_rotation(),
            )),
            3,
            "cylinder",
        ),
    ];

    for (shape, expected_regions, slug) in &shapes {
        for &depth in &depths {
            run_case(
                shape.as_ref(),
                depth,
                *expected_regions,
                slug,
                render_dir.as_deref(),
            );
        }
    }
}

fn run_case(
    shape: &dyn OracleShape,
    max_depth: usize,
    expected_regions: usize,
    slug: &str,
    render_dir: Option<&std::path::Path>,
) {
    let m = mesh_shape(shape, max_depth);
    let fits = m.ring_fits(1);
    let config = SegmentationConfig::default();
    let seg = segment_regions(&m.adjacency, &fits, &config);

    println!(
        "\n=== {} | depth {} | {} vertices ===",
        shape.name(),
        max_depth,
        m.positions.len()
    );
    println!(
        "regions: {} (expected {}) | sizes: {:?}",
        seg.region_count,
        expected_regions,
        top_sizes(&seg.region_sizes, 8)
    );

    // Purity: map each region to its majority true face, then measure
    // agreement over all claimed vertices. Fragmentation shows up as several
    // regions mapping to one face; bridging as low purity.
    let mut majority: HashMap<u32, HashMap<u32, usize>> = HashMap::new();
    for v in 0..m.positions.len() {
        if let Some(region) = seg.labels[v] {
            *majority
                .entry(region)
                .or_default()
                .entry(m.truths[v].face_id)
                .or_default() += 1;
        }
    }
    let region_to_face: HashMap<u32, u32> = majority
        .iter()
        .map(|(&region, counts)| {
            let (&face, _) = counts.iter().max_by_key(|entry| *entry.1).unwrap();
            (region, face)
        })
        .collect();
    let mut faces_covered: Vec<u32> = region_to_face.values().copied().collect();
    faces_covered.sort_unstable();
    faces_covered.dedup();

    let claimed = seg.labels.iter().flatten().count();
    let mislabeled = (0..m.positions.len())
        .filter(|&v| {
            seg.labels[v].is_some_and(|region| region_to_face[&region] != m.truths[v].face_id)
        })
        .count();
    println!(
        "claimed: {}/{} ({:.1}%) | label purity: {:.3}% mislabeled ({} vertices) | true faces covered: {}/{}",
        claimed,
        m.positions.len(),
        100.0 * claimed as f64 / m.positions.len() as f64,
        100.0 * mislabeled as f64 / claimed.max(1) as f64,
        mislabeled,
        faces_covered.len(),
        expected_regions
    );

    // Coverage per feature-distance bucket: interiors should be fully
    // claimed, the unclaimed zone should concentrate at features.
    for bucket in 0..3 {
        let members: Vec<usize> = (0..m.positions.len())
            .filter(|&v| m.bucket(v) == bucket)
            .collect();
        if members.is_empty() {
            continue;
        }
        let unclaimed = members.iter().filter(|&&v| seg.labels[v].is_none()).count();
        println!(
            "  {:<16} {:>7} vertices | unclaimed {:>6} ({:.2}%)",
            BUCKET_NAMES[bucket],
            members.len(),
            unclaimed,
            100.0 * unclaimed as f64 / members.len() as f64
        );
    }

    // Boundary localization: adjacent claimed pairs with different regions
    // should only occur near true features.
    let mut boundary_dists_cells: Vec<f64> = Vec::new();
    let mut far_boundary_pairs = 0usize;
    for v in 0..m.positions.len() as u32 {
        for &u in m.adjacency.neighbors(v) {
            if u <= v {
                continue;
            }
            if let (Some(a), Some(b)) = (seg.labels[v as usize], seg.labels[u as usize])
                && a != b
            {
                let d = m.truths[v as usize]
                    .dist_to_sharp
                    .min(m.truths[u as usize].dist_to_sharp)
                    / m.cell;
                if d >= 2.0 {
                    far_boundary_pairs += 1;
                }
                boundary_dists_cells.push(d);
            }
        }
    }
    if boundary_dists_cells.is_empty() {
        println!("  region-contact pairs: none");
    } else {
        boundary_dists_cells.sort_by(f64::total_cmp);
        let median = boundary_dists_cells[boundary_dists_cells.len() / 2];
        let p95 =
            boundary_dists_cells[((boundary_dists_cells.len() - 1) as f64 * 0.95).round() as usize];
        println!(
            "  region-contact pairs: {} | dist-to-true-feature median {:.2}c p95 {:.2}c | false (>=2c) {}",
            boundary_dists_cells.len(),
            median,
            p95,
            far_boundary_pairs
        );
    }

    if let Some(dir) = render_dir {
        let path = dir.join(format!("{}_d{}_segments.png", slug, max_depth));
        render_segmentation(&m, &seg, &path);
        println!("  render: {}", path.display());
    }
}

/// Render with per-region colors; triangles touching any unclaimed vertex are
/// bright red so the feature zone stands out.
fn render_segmentation(m: &MeshedShape, seg: &Segmentation, path: &std::path::Path) {
    let (lo, hi) = m.shape.world_bounds();
    let center = (lo + hi) / 2.0;
    let radius = (hi - lo).length() / 2.0;
    let config = RenderConfig {
        camera_pos: center + DVec3::new(1.0, 0.62, 1.05).normalize() * radius * 2.6,
        target: center,
        ..RenderConfig::default()
    };
    let tri_color = |tri: usize| -> [f64; 3] {
        let vs = &m.indices[tri * 3..tri * 3 + 3];
        let labels: Vec<Option<u32>> = vs.iter().map(|&v| seg.labels[v as usize]).collect();
        if labels.iter().any(|l| l.is_none()) {
            return [0.95, 0.15, 0.12]; // feature zone
        }
        region_color(labels[0].unwrap())
    };
    if let Err(err) = render_png(&m.positions, &m.indices, &tri_color, &config, path) {
        eprintln!("render failed: {err}");
    }
}

fn top_sizes(sizes: &[usize], n: usize) -> Vec<usize> {
    let mut sorted = sizes.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    sorted.truncate(n);
    sorted
}
