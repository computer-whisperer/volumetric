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
use meshing_lab::adjacency::{MeshAdjacency, unique_edges};
use meshing_lab::fit::{fit_plane, unsigned_angle_degrees};
use meshing_lab::oracle::{
    BoxShape, CylinderShape, OracleShape, Rotated, SphereShape, standard_rotation,
};
use volumetric::adaptive_surface_nets_2::{AdaptiveMeshConfig2, adaptive_surface_nets_2};

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
    let (lo, hi) = shape.world_bounds();
    let margin = (hi - lo).max_element() * 0.12;
    let lo = lo - DVec3::splat(margin);
    let hi = hi + DVec3::splat(margin);

    let config = AdaptiveMeshConfig2 {
        base_resolution: 8,
        max_depth,
        vertex_refinement_iterations: 12,
        normal_sample_iterations: 0,
        normal_epsilon_frac: 0.1,
        num_threads: 0,
        sharp_edge_config: None,
    };
    let finest_cells = config.base_resolution * (1 << config.max_depth);
    let cell = (hi - lo).max_element() / finest_cells as f64;

    let result = adaptive_surface_nets_2(
        |x, y, z| {
            if shape.is_inside(DVec3::new(x, y, z)) {
                1.0
            } else {
                0.0
            }
        },
        (lo.x as f32, lo.y as f32, lo.z as f32),
        (hi.x as f32, hi.y as f32, hi.z as f32),
        &config,
    );
    let mesh = &result.mesh;
    let stats = &result.stats;

    println!(
        "\n=== {} | depth {} ({}^3 cells, cell={:.5}) ===",
        shape.name(),
        max_depth,
        finest_cells,
        cell
    );
    println!(
        "mesh: {} vertices, {} triangles | refine miss {} | total samples {}",
        mesh.vertices.len(),
        mesh.indices.len() / 3,
        stats.stage4_refine_miss,
        stats.total_samples
    );

    let positions: Vec<DVec3> = mesh
        .vertices
        .iter()
        .map(|&(x, y, z)| DVec3::new(x as f64, y as f64, z as f64))
        .collect();
    let mesh_normals: Vec<DVec3> = mesh
        .normals
        .iter()
        .map(|&(x, y, z)| DVec3::new(x as f64, y as f64, z as f64))
        .collect();
    let adjacency = MeshAdjacency::build(positions.len(), &mesh.indices);

    let truths: Vec<_> = positions.iter().map(|&p| shape.truth(p)).collect();

    // Per-vertex fits for each ring size. Fit normals are oriented outward by
    // the accumulated mesh normal so errors past 90 degrees stay visible.
    let mut fit_normals: Vec<Vec<Option<DVec3>>> = vec![vec![None; positions.len()]; KS.len()];
    let mut fit_residuals: Vec<Vec<Option<f64>>> = vec![vec![None; positions.len()]; KS.len()];
    for (ki, &k) in KS.iter().enumerate() {
        for v in 0..positions.len() as u32 {
            let ring = adjacency.k_ring(v, k);
            if ring.len() < 4 {
                continue;
            }
            let pts: Vec<DVec3> = ring.iter().map(|&u| positions[u as usize]).collect();
            if let Some(fit) = fit_plane(&pts) {
                let mut n = fit.normal;
                let reference = mesh_normals[v as usize];
                if reference.length_squared() > 0.25 && n.dot(reference) < 0.0 {
                    n = -n;
                }
                fit_normals[ki][v as usize] = Some(n);
                fit_residuals[ki][v as usize] = Some(fit.rms_residual);
            }
        }
    }

    // Buckets by true distance to the nearest sharp feature, in cell units.
    let bucket_of = |dist: f64| -> usize {
        let cells = dist / cell;
        if cells >= 2.0 {
            0 // interior
        } else if cells >= 1.0 {
            1 // near
        } else {
            2 // at-feature
        }
    };
    const BUCKET_NAMES: [&str; 3] = ["interior (>=2c)", "near (1-2c)", "feature (<1c)"];

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
        let members: Vec<usize> = (0..positions.len())
            .filter(|&v| bucket_of(truths[v].dist_to_sharp) == bucket)
            .collect();
        if members.is_empty() {
            continue;
        }
        let accum_errs: Vec<f64> = members
            .iter()
            .filter(|&&v| mesh_normals[v].length_squared() > 0.25)
            .map(|&v| oriented_angle_degrees(mesh_normals[v], truths[v].normal))
            .collect();
        let mut cols = String::new();
        for ki in 0..KS.len() {
            let errs: Vec<f64> = members
                .iter()
                .filter_map(|&v| {
                    fit_normals[ki][v].map(|n| oriented_angle_degrees(n, truths[v].normal))
                })
                .collect();
            let residuals: Vec<f64> = members
                .iter()
                .filter_map(|&v| fit_residuals[ki][v])
                .map(|r| r / cell)
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
    let edges = unique_edges(&mesh.indices);
    println!("\n  adjacent-vertex normal jump (degrees, unsigned):");
    println!(
        "  {:<22} {:>8} | {:>13} | {:>13} | {:>13}",
        "pair group", "count", "accum med/p95", "k=1 med/p95", "k=2 med/p95"
    );
    let group_of = |a: usize, b: usize| -> usize {
        let min_dist = truths[a].dist_to_sharp.min(truths[b].dist_to_sharp);
        if truths[a].face_id != truths[b].face_id {
            2 // crossing a sharp feature
        } else if min_dist / cell < 2.0 {
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
                unsigned_angle_degrees(mesh_normals[a as usize], mesh_normals[b as usize])
            })
            .collect();
        let mut cols = String::new();
        for ki in 0..2 {
            let jumps: Vec<f64> = pairs
                .iter()
                .filter_map(|&&(a, b)| {
                    match (fit_normals[ki][a as usize], fit_normals[ki][b as usize]) {
                        (Some(na), Some(nb)) => Some(unsigned_angle_degrees(na, nb)),
                        _ => None,
                    }
                })
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
