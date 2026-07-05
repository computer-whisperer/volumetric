//! Feature-snap benchmark: accuracy against oracles, validity everywhere.
//!
//! Accuracy (oracle shapes only): how far snapped vertices sit from the true
//! feature curve before vs after snapping, and from true corner points for
//! corner snaps.
//!
//! Validity (all shapes, most importantly the mandelbulb): the snap stage must
//! never produce non-finite positions, out-of-bound moves, or newly inverted
//! triangles — pathological geometry has to degrade to the unsnapped mesh.
//!
//! Usage: snap_bench [depth...] [--render <dir>]

use glam::{DQuat, DVec3};
use meshing_lab::harness::{MeshedShape, mesh_shape};
use meshing_lab::oracle::{
    BoxShape, CylinderShape, MandelbulbShape, OracleShape, Rotated, SphereShape, standard_rotation,
};
use meshing_lab::render::{render_plain, render_segments};
use meshing_lab::segmentation::{SegmentationConfig, segment_regions};
use meshing_lab::snap::{SnapConfig, SnapKind, SnapResult, snap_feature_vertices};
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
        depths = vec![4];
    }
    if let Some(dir) = &render_dir {
        std::fs::create_dir_all(dir).expect("create render dir");
    }

    let rot = standard_rotation();
    let shapes: Vec<(Box<dyn OracleShape>, &str, bool)> = vec![
        (
            Box::new(Rotated::new(
                BoxShape {
                    half: DVec3::splat(0.5),
                },
                rot,
            )),
            "box",
            true,
        ),
        (Box::new(SphereShape { radius: 0.5 }), "sphere", true),
        (
            Box::new(Rotated::new(
                CylinderShape {
                    radius: 0.4,
                    half_height: 0.5,
                },
                rot,
            )),
            "cylinder",
            true,
        ),
        (
            Box::new(MandelbulbShape { iterations: 10 }),
            "mandelbulb",
            false,
        ),
    ];

    for (shape, slug, has_truth) in &shapes {
        for &depth in &depths {
            run_case(
                shape.as_ref(),
                depth,
                slug,
                *has_truth,
                rot,
                render_dir.as_deref(),
            );
        }
    }
}

fn run_case(
    shape: &dyn OracleShape,
    max_depth: usize,
    slug: &str,
    has_truth: bool,
    box_rotation: DQuat,
    render_dir: Option<&std::path::Path>,
) {
    let m = mesh_shape(shape, max_depth);
    let fits = m.ring_fits(1);
    let seg = segment_regions(&m.adjacency, &fits, &SegmentationConfig::default());

    let sampler = |p: DVec3| shape.is_inside(p);
    let result = snap_feature_vertices(
        &m.positions,
        &m.adjacency,
        &seg.labels,
        m.cell,
        &SnapConfig::default(),
        Some(&sampler),
    );
    let s = &result.stats;

    println!(
        "\n=== {} | depth {} | {} vertices, {} unclaimed candidates ===",
        shape.name(),
        max_depth,
        m.positions.len(),
        s.candidates
    );
    println!(
        "snapped: {} edge + {} corner ({:.1}% of candidates) | rejections: sides {}, parallel {}, move {}, verify {}, nonfinite {} | corner fallbacks {}",
        s.snapped_edges,
        s.snapped_corners,
        100.0 * (s.snapped_edges + s.snapped_corners) as f64 / s.candidates.max(1) as f64,
        s.rejected_sides,
        s.rejected_parallel,
        s.rejected_move,
        s.rejected_verify,
        s.rejected_nonfinite,
        s.corner_fallbacks,
    );

    // -------- Accuracy vs oracle truth --------
    if has_truth {
        let snapped_verts: Vec<usize> = (0..m.positions.len())
            .filter(|&v| result.snapped[v].is_some())
            .collect();
        if !snapped_verts.is_empty() {
            let before: Vec<f64> = snapped_verts
                .iter()
                .map(|&v| shape.truth(m.positions[v]).dist_to_sharp / m.cell)
                .collect();
            let after: Vec<f64> = snapped_verts
                .iter()
                .map(|&v| shape.truth(result.positions[v]).dist_to_sharp / m.cell)
                .collect();
            println!(
                "  dist to true feature (cells): before med {:.3} p95 {:.3} -> after med {:.4} p95 {:.4}",
                median(&before),
                p95(&before),
                median(&after),
                p95(&after)
            );
        }

        // Corner-snap accuracy for the box: distance to the nearest true
        // (rotated) cube corner.
        if slug == "box" {
            let corners: Vec<DVec3> = (0..8)
                .map(|i| {
                    box_rotation
                        * DVec3::new(
                            if i & 1 == 0 { -0.5 } else { 0.5 },
                            if i & 2 == 0 { -0.5 } else { 0.5 },
                            if i & 4 == 0 { -0.5 } else { 0.5 },
                        )
                })
                .collect();
            let corner_dist = |p: DVec3| -> f64 {
                corners
                    .iter()
                    .map(|&c| (p - c).length())
                    .fold(f64::INFINITY, f64::min)
                    / m.cell
            };
            let corner_verts: Vec<usize> = (0..m.positions.len())
                .filter(|&v| result.snapped[v] == Some(SnapKind::Corner))
                .collect();
            if corner_verts.is_empty() {
                println!("  corner snaps: none");
            } else {
                let before: Vec<f64> = corner_verts
                    .iter()
                    .map(|&v| corner_dist(m.positions[v]))
                    .collect();
                let after: Vec<f64> = corner_verts
                    .iter()
                    .map(|&v| corner_dist(result.positions[v]))
                    .collect();
                println!(
                    "  corner snaps: {} | dist to true corner (cells): before med {:.3} -> after med {:.4} max {:.4}",
                    corner_verts.len(),
                    median(&before),
                    median(&after),
                    after.iter().copied().fold(0.0, f64::max)
                );
            }
        }
    }

    // -------- Validity (all shapes) --------
    let moves: Vec<f64> = (0..m.positions.len())
        .filter(|&v| result.snapped[v].is_some())
        .map(|v| (result.positions[v] - m.positions[v]).length() / m.cell)
        .collect();
    let nonfinite = result.positions.iter().filter(|p| !p.is_finite()).count();
    let (flipped, degenerate) = triangle_damage(&m, &result);
    println!(
        "  validity: moves med {:.3} max {:.3} cells | non-finite {} | flipped tris {} | degenerate tris {} (of {})",
        median(&moves),
        moves.iter().copied().fold(0.0, f64::max),
        nonfinite,
        flipped,
        degenerate,
        m.indices.len() / 3
    );

    if let Some(dir) = render_dir {
        let bounds = shape.world_bounds();
        let before = dir.join(format!("{}_d{}_before.png", slug, max_depth));
        let after = dir.join(format!("{}_d{}_after.png", slug, max_depth));
        render_plain(&m.positions, &m.indices, bounds, &before);
        render_plain(&result.positions, &m.indices, bounds, &after);
        let segments = dir.join(format!("{}_d{}_after_segments.png", slug, max_depth));
        render_segments(
            &result.positions,
            &m.indices,
            &seg.labels,
            bounds,
            &segments,
        );
        println!(
            "  renders: {} / {} / {}",
            before.display(),
            after.display(),
            segments.display()
        );
    }
}

/// Count triangles inverted (normal flipped past 90 degrees) or collapsed to
/// near-zero area by snapping.
fn triangle_damage(m: &MeshedShape, result: &SnapResult) -> (usize, usize) {
    let mut flipped = 0;
    let mut degenerate = 0;
    let area_floor = 1e-6 * m.cell * m.cell;
    for tri in m.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if result.snapped[a].is_none() && result.snapped[b].is_none() && result.snapped[c].is_none()
        {
            continue;
        }
        let n_before = (m.positions[b] - m.positions[a]).cross(m.positions[c] - m.positions[a]);
        let n_after = (result.positions[b] - result.positions[a])
            .cross(result.positions[c] - result.positions[a]);
        if n_after.length() / 2.0 < area_floor {
            degenerate += 1;
        } else if n_before.dot(n_after) < 0.0 {
            flipped += 1;
        }
    }
    (flipped, degenerate)
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
