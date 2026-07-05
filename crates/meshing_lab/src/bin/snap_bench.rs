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
use meshing_lab::cleanup::{
    CleanupConfig, boundary_edge_count, inward_facing_count, weld_snapped_vertices,
};
use meshing_lab::crease::split_crease_vertices;
use meshing_lab::harness::mesh_shape_with_margin;
use meshing_lab::oracle::{
    BoxShape, CylinderShape, MandelbulbShape, OracleShape, PolygonPrism, Rotated, SphereShape,
    standard_rotation,
};
use meshing_lab::render::{frame_bounds, render_plain, render_segments, render_smooth_png};
use meshing_lab::segmentation::{SegmentationConfig, segment_regions};
use meshing_lab::snap::{SnapConfig, SnapKind, snap_feature_vertices};
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
    // (shape, slug, has_truth, bounds margin fraction). The 0.015625 margin
    // mimics the production pad (~2 cells at depth 4), where grid alignment
    // collapses the sawtooth and regions touch without an unclaimed band.
    let shapes: Vec<(Box<dyn OracleShape>, &str, bool, f64)> = vec![
        (
            Box::new(Rotated::new(
                BoxShape {
                    half: DVec3::splat(0.5),
                },
                rot,
            )),
            "box",
            true,
            0.12,
        ),
        // Axis-aligned: features on grid planes -- the degenerate case users
        // hit first (a plain cube), untested by rotated shapes.
        (
            Box::new(BoxShape {
                half: DVec3::splat(0.5),
            }),
            "box_axis",
            true,
            0.12,
        ),
        (
            Box::new(BoxShape {
                half: DVec3::splat(0.5),
            }),
            "box_axis_tight",
            true,
            0.015625,
        ),
        (Box::new(SphereShape { radius: 0.5 }), "sphere", true, 0.12),
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
            0.12,
        ),
        (
            Box::new(Rotated::new(PolygonPrism::l_shape(0.5, 0.4), rot)),
            "lprism",
            true,
            0.12,
        ),
        (
            Box::new(MandelbulbShape { iterations: 10 }),
            "mandelbulb",
            false,
            0.12,
        ),
    ];

    for (shape, slug, has_truth, margin) in &shapes {
        for &depth in &depths {
            run_case(
                shape.as_ref(),
                depth,
                slug,
                *has_truth,
                *margin,
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
    margin_frac: f64,
    box_rotation: DQuat,
    render_dir: Option<&std::path::Path>,
) {
    let m = mesh_shape_with_margin(shape, max_depth, margin_frac);
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

        // Corner-snap accuracy for the boxes: distance to the nearest true
        // cube corner.
        if slug.starts_with("box") {
            let corner_rot = if slug == "box" {
                box_rotation
            } else {
                DQuat::IDENTITY
            };
            let corners: Vec<DVec3> = (0..8)
                .map(|i| {
                    corner_rot
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

    // -------- Cleanup: weld the collapsed band --------
    let cleaned = weld_snapped_vertices(
        &result.positions,
        &m.indices,
        &result.snapped,
        m.cell,
        &CleanupConfig::default(),
    );
    // Carry outward reference normals through the remap; cluster members
    // agree in orientation, so summing is safe.
    let mut cleaned_normals: Vec<DVec3> = vec![DVec3::ZERO; cleaned.positions.len()];
    for v in 0..m.positions.len() {
        cleaned_normals[cleaned.remap[v] as usize] += m.mesh_normals[v];
    }
    println!(
        "  cleanup: welded {} vertices, dropped {} triangles",
        cleaned.welded_vertices, cleaned.dropped_triangles
    );

    // -------- Crease split: per-region shading normals --------
    let mut cleaned_labels: Vec<Option<u32>> = vec![None; cleaned.positions.len()];
    let mut is_crease = vec![false; cleaned.positions.len()];
    for v in 0..m.positions.len() {
        if let Some(l) = seg.labels[v] {
            cleaned_labels[cleaned.remap[v] as usize] = Some(l);
        }
        if result.snapped[v].is_some() {
            is_crease[cleaned.remap[v] as usize] = true;
        }
    }
    let split = split_crease_vertices(
        &cleaned.positions,
        &cleaned_normals,
        &cleaned.indices,
        &cleaned_labels,
        &is_crease,
    );

    // Geometric seal: the split adds topological boundary edges along
    // creases, but every one must have a position-coincident partner, or the
    // surface has a real crack.
    let unsealed = unsealed_boundary_edges(&split.positions, &split.indices);

    // Crease shading accuracy: each crease-copy normal vs the oracle normal
    // at the centroid of a triangle referencing it (the centroid sits inside
    // that side's face, so the oracle picks the correct smooth face).
    let mut crease_normal_errs: Vec<f64> = Vec::new();
    if has_truth {
        for tri in split.indices.chunks_exact(3) {
            let (pa, pb, pc) = (
                split.positions[tri[0] as usize],
                split.positions[tri[1] as usize],
                split.positions[tri[2] as usize],
            );
            // Slivers along the crease have their centroid on the feature
            // line, where the oracle's nearest face is a coin flip; they also
            // cover no pixels. Skip them.
            if (pb - pa).cross(pc - pa).length() / 2.0 < 0.01 * m.cell * m.cell {
                continue;
            }
            let centroid = (pa + pb + pc) / 3.0;
            let truth_normal = shape.truth(centroid).normal;
            for &v in tri {
                if (v as usize) < is_crease.len() && !is_crease[v as usize] {
                    continue; // only crease slots (originals or copies)
                }
                let n = split.normals[v as usize];
                if n.length_squared() > 1e-20 {
                    crease_normal_errs.push(oriented_angle_degrees_v(n.normalize(), truth_normal));
                }
            }
        }
    }
    print!(
        "  crease split: {} copies | unsealed boundary edges {}",
        split.split_vertices, unsealed
    );
    if crease_normal_errs.is_empty() {
        println!();
    } else {
        println!(
            " | copy normal vs oracle med {:.2} deg p95 {:.2} deg",
            median(&crease_normal_errs),
            p95(&crease_normal_errs)
        );
    }

    // Post-pipeline shading normals for NON-crease vertices (the carried
    // ones), bucketed by true feature distance: this is what the viewport
    // actually interpolates, and where edge smearing hides if crease copies
    // alone look clean.
    if has_truth {
        println!("  carried shading normal error vs oracle (deg):");
        for bucket in 0..3 {
            let errs: Vec<f64> = (0..cleaned.positions.len())
                .filter(|&v| !is_crease[v])
                .filter(|&v| {
                    let cells = shape.truth(split.positions[v]).dist_to_sharp / m.cell;
                    match bucket {
                        0 => cells >= 2.0,
                        1 => (1.0..2.0).contains(&cells),
                        _ => cells < 1.0,
                    }
                })
                .filter_map(|v| {
                    let n = split.normals[v];
                    (n.length_squared() > 1e-20).then(|| {
                        oriented_angle_degrees_v(
                            n.normalize(),
                            shape.truth(split.positions[v]).normal,
                        )
                    })
                })
                .collect();
            if !errs.is_empty() {
                println!(
                    "    {:<16} {:>7} vertices | med {:>6.2} p95 {:>6.2} max {:>6.2}",
                    ["interior (>=2c)", "near (1-2c)", "feature (<1c)"][bucket],
                    errs.len(),
                    median(&errs),
                    p95(&errs),
                    errs.iter().copied().fold(0.0, f64::max)
                );
            }
        }
    }

    // -------- Validity (all shapes) --------
    let moves: Vec<f64> = (0..m.positions.len())
        .filter(|&v| result.snapped[v].is_some())
        .map(|v| (result.positions[v] - m.positions[v]).length() / m.cell)
        .collect();
    let nonfinite = cleaned.positions.iter().filter(|p| !p.is_finite()).count();
    let boundary_before = boundary_edge_count(&m.indices);
    let boundary_after = boundary_edge_count(&cleaned.indices);
    // Winding damage among triangles big enough for their normal to mean
    // anything (>= 1% of a cell face); sub-sliver normals are noise.
    let min_area = 0.01 * m.cell * m.cell;
    let inward_before = inward_facing_count(&m.positions, &m.indices, &m.mesh_normals, min_area);
    let inward_after = inward_facing_count(
        &cleaned.positions,
        &cleaned.indices,
        &cleaned_normals,
        min_area,
    );
    let degenerate_before = degenerate_count(&m.positions, &m.indices, m.cell);
    let degenerate_after = degenerate_count(&cleaned.positions, &cleaned.indices, m.cell);
    println!(
        "  validity: moves med {:.3} max {:.3} cells | non-finite {} | boundary edges {} -> {} | inward-facing {} -> {} | degenerate {} -> {} (of {})",
        median(&moves),
        moves.iter().copied().fold(0.0, f64::max),
        nonfinite,
        boundary_before,
        boundary_after,
        inward_before,
        inward_after,
        degenerate_before,
        degenerate_after,
        cleaned.indices.len() / 3
    );

    if let Some(dir) = render_dir {
        let bounds = shape.world_bounds();
        let before = dir.join(format!("{}_d{}_before.png", slug, max_depth));
        let after = dir.join(format!("{}_d{}_after.png", slug, max_depth));
        render_plain(&m.positions, &m.indices, bounds, &before);
        render_plain(&cleaned.positions, &cleaned.indices, bounds, &after);

        // Smooth-shaded pair: blended single normals vs per-region copies.
        // This is where crease shading quality shows; flat renders hide it.
        let blended = dir.join(format!("{}_d{}_smooth_blended.png", slug, max_depth));
        let crisp = dir.join(format!("{}_d{}_smooth_split.png", slug, max_depth));
        let smooth_config = frame_bounds(bounds.0, bounds.1);
        if let Err(err) = render_smooth_png(
            &cleaned.positions,
            &cleaned_normals,
            &cleaned.indices,
            &smooth_config,
            &blended,
        ) {
            eprintln!("render failed: {err}");
        }
        if let Err(err) = render_smooth_png(
            &split.positions,
            &split.normals,
            &split.indices,
            &smooth_config,
            &crisp,
        ) {
            eprintln!("render failed: {err}");
        }

        let segments = dir.join(format!("{}_d{}_after_segments.png", slug, max_depth));
        render_segments(
            &cleaned.positions,
            &cleaned.indices,
            &cleaned_labels,
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

/// Triangles with near-zero area (thin slivers left along feature lines).
fn degenerate_count(positions: &[DVec3], indices: &[u32], cell: f64) -> usize {
    let area_floor = 1e-6 * cell * cell;
    indices
        .chunks_exact(3)
        .filter(|tri| {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            (positions[b] - positions[a])
                .cross(positions[c] - positions[a])
                .length()
                / 2.0
                < area_floor
        })
        .count()
}

/// Boundary edges (used by exactly one triangle) whose endpoint positions are
/// not shared by any other edge of the mesh. Crease splits create boundary
/// edges by design, but each coincides positionally with the opposite side's
/// edge (which may itself be a regular interior edge when only one side
/// split); an edge alone at its position is a real crack.
fn unsealed_boundary_edges(positions: &[DVec3], indices: &[u32]) -> usize {
    use std::collections::HashMap;
    let key = |v: u32| -> [u64; 3] {
        let p = positions[v as usize];
        [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
    };
    let pos_key = |a: u32, b: u32| -> [[u64; 3]; 2] {
        let (ka, kb) = (key(a), key(b));
        if ka <= kb { [ka, kb] } else { [kb, ka] }
    };
    // Index-level edge use counts and position-level totals.
    let mut index_counts: HashMap<(u32, u32), u32> = HashMap::new();
    let mut position_totals: HashMap<[[u64; 3]; 2], u32> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            *index_counts.entry((a.min(b), a.max(b))).or_default() += 1;
            *position_totals.entry(pos_key(a, b)).or_default() += 1;
        }
    }
    index_counts
        .iter()
        .filter(|&(&(a, b), &c)| c == 1 && position_totals[&pos_key(a, b)] == 1)
        .count()
}

/// Oriented angle between unit vectors, degrees.
fn oriented_angle_degrees_v(a: DVec3, b: DVec3) -> f64 {
    a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
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
