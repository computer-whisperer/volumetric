//! Heatmap of shading-normal vs face-normal divergence over a meshed model.
//!
//! Colors each triangle by the worst angle between a corner's shading normal
//! and the triangle's geometric normal: green ~0 deg -> red >= 30 deg. On
//! flat-dominated models (trays, enclosures) hot triangles mark exactly the
//! crease-residue artifacts; on curved regions some divergence is legitimate
//! smooth shading, so judge hot spots against the flat faces around them.
//!
//! Usage: crease_heat <model.wasm> <out_dir> [--no-simplify] [--depth N]

use glam::DVec3;
use meshing_lab::render;
use std::path::PathBuf;
use volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2;
use volumetric::sharp_features::SharpFeatureConfig;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = {
        let value_positions: Vec<usize> = ["--depth", "--probe"]
            .iter()
            .filter_map(|flag| args.iter().position(|a| a == flag).map(|i| i + 1))
            .collect();
        args.iter()
            .enumerate()
            .filter(|(i, a)| !a.starts_with("--") && !value_positions.contains(i))
            .map(|(_, a)| a)
            .collect()
    };
    if positional.len() != 2 {
        eprintln!("usage: crease_heat <model.wasm> <out_dir> [--no-simplify] [--depth N]");
        std::process::exit(1);
    }
    let wasm_bytes = std::fs::read(positional[0]).expect("read wasm");
    let out_dir = PathBuf::from(positional[1]);
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let max_depth = args
        .iter()
        .position(|a| a == "--depth")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().expect("--depth takes an integer"))
        .unwrap_or(4);
    let simplify = !args.iter().any(|a| a == "--no-simplify");

    let config = AdaptiveMeshConfig2 {
        base_resolution: 8,
        discovery_probes: 8,
        max_depth,
        vertex_refinement_iterations: 8,
        normal_sample_iterations: 0,
        normal_epsilon_frac: 0.1,
        num_threads: 0,
        sharp_features: Some(SharpFeatureConfig::default()),
        edge_constrained_refinement: false,
        decimation: simplify.then(volumetric::mesh_decimation::DecimationConfig::default),
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

    // Per-triangle worst corner-vs-face angle in degrees.
    let tri_count = indices.len() / 3;
    let mut tri_angle = vec![0.0f64; tri_count];
    let mut hot = 0usize;
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let Some(face) = (positions[b] - positions[a])
            .cross(positions[c] - positions[a])
            .try_normalize()
        else {
            continue;
        };
        let worst = tri
            .iter()
            .filter_map(|&v| normals[v as usize].try_normalize())
            .map(|n| n.dot(face).clamp(-1.0, 1.0).acos().to_degrees())
            .fold(0.0, f64::max);
        tri_angle[t] = worst;
        if worst > 15.0 {
            hot += 1;
        }
    }
    println!(
        "{} verts, {tri_count} tris | {hot} tris with a corner normal > 15 deg off face",
        positions.len()
    );

    // --pin-hist: histogram of per-vertex max divergence between the shading
    // normal and incident face planes (the decimation pin's discriminator).
    // Run with --no-simplify so the mesh is the decimation entry mesh.
    if args.iter().any(|a| a == "--pin-hist") {
        let mut max_div = vec![0.0f64; positions.len()];
        for tri in indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let Some(face) = (positions[b] - positions[a])
                .cross(positions[c] - positions[a])
                .try_normalize()
            else {
                continue;
            };
            for &v in tri {
                if let Some(n) = normals[v as usize].try_normalize() {
                    let ang = n.dot(face).clamp(-1.0, 1.0).acos().to_degrees();
                    if ang > max_div[v as usize] {
                        max_div[v as usize] = ang;
                    }
                }
            }
        }
        let edges = [
            5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 55.0, 65.0, 80.0, 100.0, 180.0,
        ];
        let mut lo = 0.0;
        println!("pin histogram (per-vertex max shading-vs-face divergence):");
        for &e in &edges {
            let n = max_div.iter().filter(|&&d| d >= lo && d < e).count();
            println!("  [{lo:5.1}, {e:5.1}): {n}");
            lo = e;
        }
    }

    // --probe x,y,z,r: dump every vertex within r (inf-norm) with its normal,
    // plus each incident triangle's face normal and area, to see exactly what
    // tilts an accumulated normal.
    if let Some(spec) = args
        .iter()
        .position(|a| a == "--probe")
        .and_then(|i| args.get(i + 1))
    {
        let vals: Vec<f64> = spec.split(',').map(|v| v.parse().unwrap()).collect();
        let target = DVec3::new(vals[0], vals[1], vals[2]);
        let r = vals[3];
        // Approximate finest cell from bounds (matches wasm_mesh_view).
        let finest = (8usize << max_depth) as f64;
        let ext = [
            (result.bounds_max.0 - result.bounds_min.0) as f64,
            (result.bounds_max.1 - result.bounds_min.1) as f64,
            (result.bounds_max.2 - result.bounds_min.2) as f64,
        ];
        let cell = ext.iter().cloned().fold(0.0f64, f64::max) / finest;
        for (v, p) in positions.iter().enumerate() {
            if (*p - target).abs().max_element() >= r {
                continue;
            }
            let n = normals[v].normalize_or_zero();
            println!(
                "v{v} pos=({:+.5},{:+.5},{:+.5}) n=({:+.3},{:+.3},{:+.3})",
                p.x, p.y, p.z, n.x, n.y, n.z
            );
            for (t, tri) in indices.chunks_exact(3).enumerate() {
                if !tri.contains(&(v as u32)) {
                    continue;
                }
                let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                let cross = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
                let fnorm = cross.normalize_or_zero();
                let angle = n.dot(fnorm).clamp(-1.0, 1.0).acos().to_degrees();
                println!(
                    "    t{t} fn=({:+.3},{:+.3},{:+.3}) area={:.3}c2 vs_n={:5.1}deg corners=[{} {} {}]",
                    fnorm.x,
                    fnorm.y,
                    fnorm.z,
                    0.5 * cross.length() / (cell * cell),
                    angle,
                    tri[0],
                    tri[1],
                    tri[2],
                );
            }
        }
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
    let center = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 0.5;
    let tri_color = |t: usize| -> [f64; 3] {
        let x = (tri_angle[t] / 30.0).min(1.0);
        [0.2 + 0.7 * x, 0.7 - 0.5 * x, 0.25]
    };
    let views: [(&str, DVec3, f64); 4] = [
        ("persp", DVec3::new(2.2, 1.6, 2.2), 2.6),
        ("back", DVec3::new(-2.2, -1.6, 2.2), 2.6),
        ("front", DVec3::new(0.3, -2.5, 1.4), 2.2),
        ("top", DVec3::new(0.15, 0.4, 3.0), 2.6),
    ];
    for (name, dir, dist) in views {
        let mut cfg = render::frame_bounds(lo, hi);
        cfg.camera_pos = center + dir.normalize() * radius * dist;
        cfg.target = center;
        render::render_png(
            &positions,
            indices,
            &tri_color,
            &cfg,
            &out_dir.join(format!("{name}_heat.png")),
        )
        .expect("render");
    }
    println!("heatmaps written to {}", out_dir.display());
}
