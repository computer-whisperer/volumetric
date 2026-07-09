//! Grazing-rim stress case for the inverse update dynamics
//! (fea_strut_example_2 at 1/10 scale): a sphere pressed into a jittered
//! strut lattice so the contact patch lies strictly inside the top face —
//! rim columns demand force where penetration vanishes, i.e. unbounded,
//! unsatisfiable stiffness contrast. This is the case that made the old
//! renormalize-by-max update ratchet the satisfied bulk into the min_scale
//! floor; it runs in seconds instead of the real project's ~30 minutes, so
//! update-rule changes can be measured before an fea_strut_example_2
//! confirmation run (examples/profile_fea_inverse.rs).
//!
//! Usage: cargo run -p fea_core --release --example grazing_rim
//! Knobs: FEA_INVERSE_DEBUG=1|2 (per-iteration trace),
//!        GRAZE_MAX_ITER (default 30), GRAZE_MIN_SCALE (default 0.01),
//!        GRAZE_JITTER (default 0.4, in cells — how foam-like the mesh is),
//!        GRAZE_TOL (default 0.005 — deliberately below the geometric
//!        limit, so the loop runs in the unreachable-tolerance regime
//!        where the instability lived).

use std::time::Instant;

use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh};

/// A cubic strut lattice block: (nx+1)(ny+1)(nz+1) grid nodes with struts
/// along all three axes, uniform radius (mirrors fea_core's test helper).
fn strut_grid(nx: usize, ny: usize, nz: usize, h: f64, radius: f64) -> FeaMesh {
    let (mx, my) = (nx + 1, ny + 1);
    let node = |i: usize, j: usize, k: usize| (k * my * mx + j * mx + i) as u32;
    let mut node_positions = Vec::new();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                node_positions.extend([i as f64 * h, j as f64 * h, k as f64 * h]);
            }
        }
    }
    let mut connectivity = Vec::new();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                if i < nx {
                    connectivity.extend([node(i, j, k), node(i + 1, j, k)]);
                }
                if j < ny {
                    connectivity.extend([node(i, j, k), node(i, j + 1, k)]);
                }
                if k < nz {
                    connectivity.extend([node(i, j, k), node(i, j, k + 1)]);
                }
            }
        }
    }
    // Jitter every node above the glued bottom layer (deterministic
    // xorshift) — the real failing meshes are foam lattices whose irregular
    // nodes land arbitrarily close to the grazing rim, where penetration
    // (and thus achievable force) vanishes. A regular grid keeps ~one cell
    // of clearance and never sees the unbounded demand ratios.
    let jitter = env_f64("GRAZE_JITTER", 0.4) * h;
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut rand_unit = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for n in 0..node_positions.len() / 3 {
        if node_positions[n * 3 + 2] > 0.0 {
            for axis in 0..3 {
                node_positions[n * 3 + axis] += jitter * rand_unit();
            }
        } else {
            // Keep the glued layer flat; burn draws to stay deterministic
            // under changing jitter amplitudes.
            for _ in 0..3 {
                rand_unit();
            }
        }
    }
    let strut_count = connectivity.len() / 2;
    FeaMesh {
        element_kind: FeaElementKind::Bar2,
        node_positions,
        connectivity,
        node_fields: vec![],
        element_fields: vec![FeaField {
            name: "radius".to_string(),
            components: 1,
            data: vec![radius; strut_count],
        }],
    }
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // 1x1 lateral, 0.3 tall, 20x20x6 cells: ~3.1k nodes / ~9.2k struts.
    let mesh = strut_grid(20, 20, 6, 0.05, 0.01);

    // Sphere r=0.5 dipping 0.1 into the top: patch radius
    // sqrt(0.25 - 0.16) = 0.3, strictly inside the face's half-width 0.5 —
    // penetration falls 0.1 -> 0 across the patch (the grazing rim).
    let center = [0.5f64, 0.5, 0.7];
    let mut rigid = |p: [f64; 3]| {
        (0..3).map(|a| (p[a] - center[a]).powi(2)).sum::<f64>() < 0.25
    };
    // Uniform demand over a disk (r=0.45) that fully covers the patch,
    // like example_2's target: uniform force demanded where penetration
    // varies, so rim demand is unsatisfiable at any stiffness.
    let mut target = |p: [f64; 2]| {
        if (p[0] - 0.5).powi(2) + (p[1] - 0.5).powi(2) <= 0.45 * 0.45 {
            1.0
        } else {
            0.2
        }
    };

    let config = fea_core::InverseConfig {
        solve: fea_core::SolveConfig {
            fixed_boundary: fea_core::FixedBoundary::ZMin,
            cg_tolerance: 1e-4,
            ..Default::default()
        },
        max_iterations: env_f64("GRAZE_MAX_ITER", 30.0) as usize,
        tolerance: env_f64("GRAZE_TOL", 0.005),
        exponent: 0.5,
        min_scale: env_f64("GRAZE_MIN_SCALE", 0.01),
        column_size: 0.0,
    };

    println!(
        "mesh: {} nodes, {} struts",
        mesh.node_count(),
        mesh.element_count(),
    );
    let timer = Instant::now();
    let result =
        fea_core::solve_inverse(&mesh, &mut rigid, &mut target, &config).expect("solve_inverse");
    println!(
        "{:.1}s, {} iterations, converged={}, best distribution_error={:.4}",
        timer.elapsed().as_secs_f64(),
        result.iterations,
        result.converged,
        result.distribution_error,
    );
}
