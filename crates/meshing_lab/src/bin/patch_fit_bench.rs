//! Patch-fit normal quality benchmark.
//!
//! Question under test: on *real* adaptive-surface-nets output at realistic
//! resolution, how accurate are per-vertex normals obtained by fitting a plane
//! to the k-ring of refined vertex positions — and do fit residuals plus
//! adjacent-vertex normal jumps separate smooth regions from sharp features
//! cleanly enough to drive region segmentation?
//!
//! Everything is measured against closed-form oracles, bucketed by true
//! distance to the nearest sharp feature in units of the finest cell size.

use glam::DVec3;
use meshing_lab::adjacency::unique_edges;
use meshing_lab::fit::VertexFit;
use meshing_lab::fit::unsigned_angle_degrees;
use meshing_lab::harness::{BUCKET_NAMES, mesh_shape};
use meshing_lab::oracle::{
    BoxShape, CylinderShape, OracleShape, Rotated, SphereShape, standard_rotation,
};

const KS: [usize; 3] = [1, 2, 3];

fn main() {
    let shapes: Vec<Box<dyn OracleShape>> = vec![
        Box::new(Rotated::new(
            BoxShape {
                half: DVec3::splat(0.5),
            },
            standard_rotation(),
        )),
        Box::new(SphereShape { radius: 0.5 }),
        Box::new(Rotated::new(
            CylinderShape {
                radius: 0.4,
                half_height: 0.5,
            },
            standard_rotation(),
        )),
    ];

    let depths: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("depth args must be integers"))
        .collect();
    let depths = if depths.is_empty() {
        vec![3, 4]
    } else {
        depths
    };

    for shape in &shapes {
        for &depth in &depths {
            run_case(shape.as_ref(), depth);
        }
    }
}

fn run_case(shape: &dyn OracleShape, max_depth: usize) {
    let m = mesh_shape(shape, max_depth);
    println!(
        "\n=== {} | depth {} (cell={:.5}) ===",
        shape.name(),
        max_depth,
        m.cell
    );
    println!(
        "mesh: {} vertices, {} triangles | refine miss {} | total samples {}",
        m.positions.len(),
        m.indices.len() / 3,
        m.refine_misses,
        m.total_samples
    );

    let fits: Vec<Vec<Option<VertexFit>>> = KS.iter().map(|&k| m.ring_fits(k)).collect();

    println!("\n  normal error vs oracle (degrees, oriented) and fit RMS residual (cell units):");
    println!(
        "  {:<16} {:>7} | {:>13} | {:>21} {:>21} {:>21}",
        "bucket",
        "count",
        "accum med/p95",
        "k=1 med/p95 (res)",
        "k=2 med/p95 (res)",
        "k=3 med/p95 (res)"
    );
    for bucket in 0..3 {
        let members: Vec<usize> = (0..m.positions.len())
            .filter(|&v| m.bucket(v) == bucket)
            .collect();
        if members.is_empty() {
            continue;
        }
        let accum_errs: Vec<f64> = members
            .iter()
            .filter(|&&v| m.mesh_normals[v].length_squared() > 0.25)
            .map(|&v| oriented_angle_degrees(m.mesh_normals[v], m.truths[v].normal))
            .collect();
        let mut cols = String::new();
        for k_fits in &fits {
            let errs: Vec<f64> = members
                .iter()
                .filter_map(|&v| {
                    k_fits[v]
                        .as_ref()
                        .map(|f| oriented_angle_degrees(f.normal, m.truths[v].normal))
                })
                .collect();
            let residuals: Vec<f64> = members
                .iter()
                .filter_map(|&v| k_fits[v].as_ref().map(|f| f.residual_cells))
                .collect();
            cols.push_str(&format!(
                " {:>6.2}/{:>6.2} ({:.3})",
                median(&errs),
                p95(&errs),
                median(&residuals)
            ));
        }
        println!(
            "  {:<16} {:>7} | {:>6.2}/{:>6.2} |{}",
            BUCKET_NAMES[bucket],
            members.len(),
            median(&accum_errs),
            p95(&accum_errs),
            cols
        );
    }

    // Adjacent-pair normal jumps: the segmentation discriminator. A mesh edge
    // "crosses" a sharp feature when its endpoints' nearest smooth faces
    // differ. Same-face pairs are split by proximity to a feature so the noise
    // floor near edges is visible separately.
    let edges = unique_edges(&m.indices);
    println!("\n  adjacent-vertex normal jump (degrees, unsigned):");
    println!(
        "  {:<22} {:>8} | {:>13} | {:>13} | {:>13}",
        "pair group", "count", "accum med/p95", "k=1 med/p95", "k=2 med/p95"
    );
    let group_of = |a: usize, b: usize| -> usize {
        let min_dist = m.truths[a].dist_to_sharp.min(m.truths[b].dist_to_sharp);
        if m.truths[a].face_id != m.truths[b].face_id {
            2 // crossing a sharp feature
        } else if min_dist / m.cell < 2.0 {
            1 // same face, near a feature
        } else {
            0 // same face, interior
        }
    };
    const GROUP_NAMES: [&str; 3] = ["same-face interior", "same-face near", "feature-crossing"];
    for group in 0..3 {
        let pairs: Vec<&(u32, u32)> = edges
            .iter()
            .filter(|&&(a, b)| group_of(a as usize, b as usize) == group)
            .collect();
        if pairs.is_empty() {
            println!("  {:<22} {:>8} | (none)", GROUP_NAMES[group], 0);
            continue;
        }
        let accum: Vec<f64> = pairs
            .iter()
            .map(|&&(a, b)| {
                unsigned_angle_degrees(m.mesh_normals[a as usize], m.mesh_normals[b as usize])
            })
            .collect();
        let mut cols = String::new();
        for k_fits in fits.iter().take(2) {
            let jumps: Vec<f64> = pairs
                .iter()
                .filter_map(
                    |&&(a, b)| match (&k_fits[a as usize], &k_fits[b as usize]) {
                        (Some(fa), Some(fb)) => Some(unsigned_angle_degrees(fa.normal, fb.normal)),
                        _ => None,
                    },
                )
                .collect();
            cols.push_str(&format!(" | {:>6.2}/{:>6.2}", median(&jumps), p95(&jumps)));
        }
        println!(
            "  {:<22} {:>8} | {:>6.2}/{:>6.2}{}",
            GROUP_NAMES[group],
            pairs.len(),
            median(&accum),
            p95(&accum),
            cols
        );
    }
}

/// Angle in degrees between oriented vectors (can exceed 90).
fn oriented_angle_degrees(a: DVec3, b: DVec3) -> f64 {
    let denom = a.length() * b.length();
    if denom < 1e-30 {
        return f64::NAN;
    }
    (a.dot(b) / denom).clamp(-1.0, 1.0).acos().to_degrees()
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

fn median(values: &[f64]) -> f64 {
    percentile(values, 0.5)
}

fn p95(values: &[f64]) -> f64 {
    percentile(values, 0.95)
}
