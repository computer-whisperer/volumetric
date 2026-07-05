//! Render the production asn2 mesher's output for a real WASM model.
//!
//! Bridges user-reported GUI defects into the lab: run the exact production
//! entry point (`generate_adaptive_mesh_v2_from_bytes`, pad+clamp and all) on
//! an exported model, then inspect the result with the lab renderer and
//! validity metrics.
//!
//! Usage: wasm_mesh_view <model.wasm> <out_dir> [--no-sharp] [--depth N]
//!
//! For a z-aligned unit cylinder (r=1, z 0..1), `--cyl-truth` adds exact
//! surface/rim error metrics and dumps the top-rim chain to rim_chain.csv;
//! `--debug-rim` (with `--no-sharp`) re-runs the pipeline stages manually and
//! reports per-vertex snap outcomes for the rim band to rim_debug.txt.

use glam::DVec3;
use meshing_lab::render;
use std::collections::HashMap;
use std::path::PathBuf;
use volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2;
use volumetric::sharp_features::{SharpFeatureConfig, adjacency, fit, segmentation, snap};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let depth_value_pos = args.iter().position(|a| a == "--depth").map(|i| i + 1);
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && Some(*i) != depth_value_pos)
        .map(|(_, a)| a)
        .collect();
    if positional.len() != 2 {
        eprintln!("usage: wasm_mesh_view <model.wasm> <out_dir> [--no-sharp] [--depth N]");
        std::process::exit(1);
    }
    let wasm_path = PathBuf::from(positional[0]);
    let out_dir = PathBuf::from(positional[1]);
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let sharp = !args.iter().any(|a| a == "--no-sharp");
    let max_depth = args
        .iter()
        .position(|a| a == "--depth")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().expect("--depth takes an integer"))
        .unwrap_or(4);

    let wasm_bytes = std::fs::read(&wasm_path).expect("read wasm");
    // GUI-default settings: refine 8, no normal probing, sharp defaults.
    let config = AdaptiveMeshConfig2 {
        base_resolution: 8,
        max_depth,
        vertex_refinement_iterations: 8,
        normal_sample_iterations: 0,
        normal_epsilon_frac: 0.1,
        num_threads: 0,
        sharp_features: sharp.then(SharpFeatureConfig::default),
    };

    let result = volumetric::generate_adaptive_mesh_v2_from_bytes(&wasm_bytes, &config)
        .expect("meshing failed");

    let positions: Vec<DVec3> = result
        .vertices
        .iter()
        .map(|&(x, y, z)| DVec3::new(x as f64, y as f64, z as f64))
        .collect();
    let normals: Vec<DVec3> = result
        .normals
        .iter()
        .map(|&(x, y, z)| DVec3::new(x as f64, y as f64, z as f64))
        .collect();
    let indices = &result.indices;

    let s = &result.stats;
    println!(
        "mesh: {} verts, {} tris | sharp: {} regions, {} candidates, {} edge + {} corner snaps, {} welded, {} dropped, {} splits",
        positions.len(),
        indices.len() / 3,
        s.sharp_regions,
        s.sharp_candidates,
        s.sharp_snapped_edges,
        s.sharp_snapped_corners,
        s.sharp_welded_vertices,
        s.sharp_dropped_triangles,
        s.sharp_crease_splits,
    );

    // Validity: boundary edges (position-keyed, so crease splits don't count),
    // degenerate triangles.
    let quant = |p: DVec3| {
        (
            (p.x * 1e7).round() as i64,
            (p.y * 1e7).round() as i64,
            (p.z * 1e7).round() as i64,
        )
    };
    let mut edge_mult: HashMap<((i64, i64, i64), (i64, i64, i64)), u32> = HashMap::new();
    let mut degenerate = 0usize;
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let area2 = (positions[b] - positions[a])
            .cross(positions[c] - positions[a])
            .length();
        if area2 < 1e-12 {
            degenerate += 1;
        }
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let (ku, kv) = (quant(positions[u]), quant(positions[v]));
            let key = if ku <= kv { (ku, kv) } else { (kv, ku) };
            *edge_mult.entry(key).or_insert(0) += 1;
        }
    }
    let unsealed = edge_mult.values().filter(|&&c| c == 1).count();
    println!("validity: {unsealed} unsealed boundary edges, {degenerate} degenerate tris");

    // Optional exact-truth check for a z-aligned cylinder (r=1, z in 0..1):
    // per-vertex distance to the true surface, plus rim-band accuracy.
    if args.iter().any(|a| a == "--cyl-truth") {
        let cell = 3.0 / (8usize << max_depth) as f64; // xy span / finest cells
        let mut surf: Vec<f64> = Vec::new();
        let mut rim: Vec<f64> = Vec::new();
        for p in &positions {
            let rho = (p.x * p.x + p.y * p.y).sqrt();
            // Distance to the capped cylinder surface.
            let dr = rho - 1.0;
            let dz = (p.z - 0.5).abs() - 0.5;
            let d = if dr <= 0.0 && dz <= 0.0 {
                -(dr.abs().min(dz.abs())) // inside: nearest face
            } else {
                let (ar, az) = (dr.max(0.0), dz.max(0.0));
                (ar * ar + az * az).sqrt()
            };
            surf.push(d.abs() / cell);
            // Distance to the nearest rim circle (rho=1, z=0 or 1).
            let dzr = (p.z - 1.0).abs().min(p.z.abs());
            let drim = ((rho - 1.0).powi(2) + dzr * dzr).sqrt();
            if drim < 1.5 * cell {
                rim.push(drim / cell);
            }
        }
        let stats = |v: &mut Vec<f64>| -> (f64, f64, f64) {
            v.sort_by(f64::total_cmp);
            let n = v.len();
            (
                v.iter().sum::<f64>() / n as f64,
                v[(n as f64 * 0.95) as usize],
                *v.last().unwrap(),
            )
        };
        let (sm, s95, smax) = stats(&mut surf);
        println!("surface dist (cells): mean {sm:.4} p95 {s95:.4} max {smax:.4}");
        if !rim.is_empty() {
            let (rm, r95, rmax) = stats(&mut rim);
            println!(
                "rim-band dist (cells, {} verts): mean {rm:.4} p95 {r95:.4} max {rmax:.4}",
                rim.len()
            );
            // Histogram: snapped rim vertices should pile up near zero; a
            // second lobe below ~1 cell is rim wobble.
            let edges = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5];
            let mut lo = 0.0;
            for &e in &edges {
                let n = rim.iter().filter(|&&d| d >= lo && d < e).count();
                println!("  rim dist [{lo:.2}, {e:.2}): {n}");
                lo = e;
            }
            // Dump the top-rim chain (vertices near z=1 and near rho=1) as
            // angle vs signed radial error, for offline analysis.
            let mut csv = String::from("theta_deg,radial_err_cells,z_err_cells\n");
            for p in &positions {
                let rho = (p.x * p.x + p.y * p.y).sqrt();
                if (p.z - 1.0).abs() < 0.3 * cell && (rho - 1.0).abs() < 0.6 * cell {
                    let theta = p.y.atan2(p.x).to_degrees();
                    csv.push_str(&format!(
                        "{theta:.3},{:.5},{:.5}\n",
                        (rho - 1.0) / cell,
                        (p.z - 1.0) / cell
                    ));
                }
            }
            std::fs::write(out_dir.join("rim_chain.csv"), csv).expect("write csv");
        }
    }

    // Per-vertex snap diagnosis for the top rim of a z-aligned cylinder:
    // re-run the pipeline stages manually on the stage-4 mesh (sharp must be
    // off so `positions` IS the stage-4 mesh) and report what each rim-band
    // vertex did.
    if args.iter().any(|a| a == "--debug-rim") {
        assert!(!sharp, "--debug-rim requires --no-sharp");
        let cell = 3.1875 / (8usize << max_depth) as f64; // padded xy span
        let sampler_impl = volumetric::wasm::create_parallel_sampler(&wasm_bytes).unwrap();
        let (bmin, bmax) = (result.bounds_min, result.bounds_max);
        let is_inside = |p: DVec3| -> bool {
            use volumetric::wasm::ParallelModelSampler;
            if p.x < bmin.0 as f64
                || p.x > bmax.0 as f64
                || p.y < bmin.1 as f64
                || p.y > bmax.1 as f64
                || p.z < bmin.2 as f64
                || p.z > bmax.2 as f64
            {
                return false;
            }
            sampler_impl.sample(p.x, p.y, p.z) > 0.5
        };
        let adj = adjacency::MeshAdjacency::build(positions.len(), indices);
        let fits = fit::ring_fits(&positions, &adj, &normals, cell, 1);
        let seg_cfg = segmentation::SegmentationConfig::default();
        let seg = segmentation::segment_regions(&adj, &fits, &seg_cfg);
        let snap_cfg = snap::SnapConfig::default();
        let snapped = snap::snap_feature_vertices(
            &positions,
            &adj,
            &seg.labels,
            cell,
            &snap_cfg,
            Some(&is_inside),
        );
        println!(
            "debug-rim: {} regions | candidates {} snapped {}+{} rejects: sides {} parallel {} move {} verify {} nonfinite {}",
            seg.region_count,
            snapped.stats.candidates,
            snapped.stats.snapped_edges,
            snapped.stats.snapped_corners,
            snapped.stats.rejected_sides,
            snapped.stats.rejected_parallel,
            snapped.stats.rejected_move,
            snapped.stats.rejected_verify,
            snapped.stats.rejected_nonfinite,
        );
        let mut lines: Vec<(f64, String)> = Vec::new();
        for v in 0..positions.len() {
            let p = positions[v];
            let q = snapped.positions[v];
            let rho_after = (q.x * q.x + q.y * q.y).sqrt();
            let rim_dist = ((rho_after - 1.0).powi(2) + (q.z - 1.0).powi(2)).sqrt() / cell;
            // Top rim band: near z=1 either before or after snapping.
            if (p.z - 1.0).abs() > 1.2 * cell || (p.x * p.x + p.y * p.y).sqrt() < 1.0 - 3.0 * cell {
                continue;
            }
            let theta = q.y.atan2(q.x).to_degrees();
            let kind = match snapped.snapped[v] {
                Some(snap::SnapKind::Edge) => "edge",
                Some(snap::SnapKind::Corner) => "corner",
                None => "-",
            };
            let label = match seg.labels[v] {
                Some(l) => l.to_string(),
                None => "unclaimed".into(),
            };
            let moved = (q - p).length() / cell;
            let fit_res = fits[v]
                .as_ref()
                .map(|f| format!("{:.3}", f.residual_cells))
                .unwrap_or_else(|| "nofit".into());
            lines.push((
                theta,
                format!(
                    "theta {theta:+8.2} label {label:>9} snap {kind:>6} moved {moved:5.2} rim_dist {rim_dist:6.3} res {fit_res} z_before {:+.3} rho_before {:+.4}",
                    (p.z - 1.0) / cell,
                    (p.x * p.x + p.y * p.y).sqrt() - 1.0,
                ),
            ));
        }
        lines.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut dbg = String::new();
        for (_, l) in &lines {
            dbg.push_str(l);
            dbg.push('\n');
        }
        std::fs::write(out_dir.join("rim_debug.txt"), dbg).expect("write debug");
        println!("rim debug ({} verts) -> rim_debug.txt", lines.len());
    }

    let lo = DVec3::new(
        result.bounds_min.0 as f64,
        result.bounds_min.1 as f64,
        result.bounds_min.2 as f64,
    );
    let hi = DVec3::new(
        result.bounds_max.0 as f64,
        result.bounds_max.1 as f64,
        result.bounds_max.2 as f64,
    );

    // Full views plus rim close-ups (the cylinder axis is Z for extrudes).
    let center = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 0.5;
    let rim = DVec3::new(center.x + (hi.x - lo.x) * 0.5, center.y, hi.z);
    let views: [(&str, DVec3, DVec3, f64); 6] = [
        ("persp", center, DVec3::new(2.2, 1.6, 2.2), 2.6),
        ("side", center, DVec3::new(3.0, 0.35, 0.0), 2.6),
        ("axis", center, DVec3::new(0.15, 0.4, 3.0), 2.6),
        ("rim_zoom", rim, DVec3::new(1.4, 0.9, 1.8), 0.7),
        ("rim_graze", rim, DVec3::new(0.2, 1.0, 0.25), 0.7),
        ("cap_graze", rim, DVec3::new(-0.9, 0.5, 0.12), 0.7),
    ];
    for (name, target, dir, dist) in views {
        let mut cfg = render::frame_bounds(lo, hi);
        cfg.camera_pos = target + dir.normalize() * radius * dist;
        cfg.target = target;
        render::render_smooth_png(
            &positions,
            &normals,
            indices,
            &cfg,
            &out_dir.join(format!("{name}_smooth.png")),
        )
        .expect("render");
        render::render_png(
            &positions,
            indices,
            &|_| [0.75, 0.75, 0.78],
            &cfg,
            &out_dir.join(format!("{name}_flat.png")),
        )
        .expect("render");
    }
    println!("renders written to {}", out_dir.display());
}
