//! Integration test against the representative Onshape hybrid export
//! (exact BRep shell + tessellated 'ConvergentBody' face glued along a
//! B-spline seam ring). Skips when the fixture directory is absent so CI
//! without it stays green.
//!
//! Ground truth is the watertight STL that Onshape exported from the
//! same Part Studio, classified by an independent Möller–Trumbore parity
//! oracle written here (majority over three skewed rays; ambiguous or
//! near-surface points are skipped).

use brep_core::ir::Surface;
use brep_core::payload::{PayloadView, build_payload};
use step_import_operator::{StepConfig, import};

fn fixture_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = std::path::PathBuf::from(home).join("workspace/playground/Cushions_2026");
    dir.is_dir().then_some(dir)
}

/// Minimal binary STL reader: triangles as f64 vertex triples, in the
/// file's millimetres.
fn read_stl(path: &std::path::Path) -> Vec<[[f64; 3]; 3]> {
    let raw = std::fs::read(path).expect("read STL");
    let n = u32::from_le_bytes(raw[80..84].try_into().unwrap()) as usize;
    let mut tris = Vec::with_capacity(n);
    for i in 0..n {
        let base = 84 + i * 50 + 12; // skip the normal
        let v = |k: usize| -> [f64; 3] {
            let off = base + k * 12;
            core::array::from_fn(|c| {
                f32::from_le_bytes(raw[off + c * 4..off + c * 4 + 4].try_into().unwrap()) as f64
            })
        };
        tris.push([v(0), v(1), v(2)]);
    }
    tris
}

/// Möller–Trumbore: does the ray from `p` along `dir` cross the triangle
/// at t > 0? (Boundary hits resolve arbitrarily — the majority vote and
/// the distance filter absorb them.)
fn ray_crosses(p: [f64; 3], dir: [f64; 3], t: &[[f64; 3]; 3]) -> bool {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let e1 = sub(t[1], t[0]);
    let e2 = sub(t[2], t[0]);
    let h = cross(dir, e2);
    let det = dot(e1, h);
    if det.abs() < 1e-14 {
        return false;
    }
    let inv = 1.0 / det;
    let s = sub(p, t[0]);
    let u = dot(s, h) * inv;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = cross(s, e1);
    let w = dot(dir, q) * inv;
    if w < 0.0 || u + w > 1.0 {
        return false;
    }
    dot(e2, q) * inv > 0.0
}

fn point_triangle_dist(p: [f64; 3], t: &[[f64; 3]; 3]) -> f64 {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let seg = |p: [f64; 3], a: [f64; 3], b: [f64; 3]| {
        let e = sub(b, a);
        let l2 = dot(e, e);
        let s = if l2 > 0.0 {
            (dot(sub(p, a), e) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = [a[0] + s * e[0], a[1] + s * e[1], a[2] + s * e[2]];
        let d = sub(p, q);
        dot(d, d).sqrt()
    };
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let n2 = dot(n, n);
    if n2 > 0.0 {
        let dp = dot(sub(p, t[0]), n) / n2;
        let q = [p[0] - dp * n[0], p[1] - dp * n[1], p[2] - dp * n[2]];
        let inside = dot(cross(sub(t[1], t[0]), sub(q, t[0])), n) >= 0.0
            && dot(cross(sub(t[2], t[1]), sub(q, t[1])), n) >= 0.0
            && dot(cross(sub(t[0], t[2]), sub(q, t[2])), n) >= 0.0;
        if inside {
            return dp.abs() * n2.sqrt();
        }
    }
    seg(p, t[0], t[1])
        .min(seg(p, t[1], t[2]))
        .min(seg(p, t[2], t[0]))
}

#[test]
fn imports_the_hybrid_cushion() {
    let Some(dir) = fixture_dir() else {
        eprintln!("cushion fixture directory not present; skipping");
        return;
    };
    let text = std::fs::read_to_string(dir.join("airline_cushion.step")).expect("read STEP");
    let model = import(&text, &StepConfig::default()).expect("import cushion");

    // One hybrid body: 7 exact faces (5 planes + 2 B-spline surfaces)
    // plus one mesh face, all styled the same blue.
    assert_eq!(model.solids.len(), 1);
    assert_eq!(model.instances.len(), 1);
    let faces = &model.solids[0].faces;
    assert_eq!(faces.len(), 8);
    let mesh_faces: Vec<_> = faces
        .iter()
        .filter(|f| matches!(f.surface, Surface::Mesh(_)))
        .collect();
    assert_eq!(mesh_faces.len(), 1);
    if let Surface::Mesh(m) = &mesh_faces[0].surface {
        assert!(m.tris.len() > 30_000, "{} triangles", m.tris.len());
    }
    for f in faces {
        let c = f.color.expect("styled face");
        assert!(
            (c[0] - 0.2314).abs() < 0.01 && (c[1] - 0.3804).abs() < 0.01,
            "unexpected color {c:?}"
        );
    }

    let payload = build_payload(&model).expect("payload");
    let view = PayloadView::new(&payload).unwrap();

    // Bounds against the sibling STL of the same body (mm -> m), with
    // sampling slack.
    let stl = read_stl(&dir.join("Part Studio 1 - Cushion.stl"));
    let mut stl_bb = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for t in &stl {
        for v in t {
            for axis in 0..3 {
                stl_bb[axis * 2] = stl_bb[axis * 2].min(v[axis]);
                stl_bb[axis * 2 + 1] = stl_bb[axis * 2 + 1].max(v[axis]);
            }
        }
    }
    const M_PER_MM: f64 = 1e-3;
    let b = view.bounds();
    for axis in 0..3 {
        let lo = stl_bb[axis * 2] * M_PER_MM;
        let hi = stl_bb[axis * 2 + 1] * M_PER_MM;
        assert!(
            b[axis * 2] <= lo + 1e-4 && b[axis * 2] >= lo - 2e-3,
            "axis {axis} min: ours {} vs STL {lo}",
            b[axis * 2]
        );
        assert!(
            b[axis * 2 + 1] >= hi - 1e-4 && b[axis * 2 + 1] <= hi + 2e-3,
            "axis {axis} max: ours {} vs STL {hi}",
            b[axis * 2 + 1]
        );
    }

    // Parity oracle: deterministic pseudo-random points in the inflated
    // box, keeping those the STL classifies unambiguously (3-ray
    // majority is unanimous) and that sit at least 0.5 mm off the STL
    // surface. The mesh region agrees exactly (same triangles); the
    // exact faces agree wherever the STL's own chordal error can't
    // flip the answer — hence the distance filter.
    const DIRS: [[f64; 3]; 3] = [
        [1.0, 1.618e-4, 2.718e-4],
        [-2.236e-4, 1.0, 1.414e-4],
        [3.141e-4, -2.718e-4, -1.0],
    ];
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut tested, mut skipped) = (0usize, 0usize);
    while tested < 400 {
        let p_mm = [
            stl_bb[0] - 10.0 + (stl_bb[1] - stl_bb[0] + 20.0) * rand(),
            stl_bb[2] - 10.0 + (stl_bb[3] - stl_bb[2] + 20.0) * rand(),
            stl_bb[4] - 10.0 + (stl_bb[5] - stl_bb[4] + 20.0) * rand(),
        ];
        let d_surf = stl
            .iter()
            .map(|t| point_triangle_dist(p_mm, t))
            .fold(f64::INFINITY, f64::min);
        if d_surf < 0.5 {
            skipped += 1;
            assert!(skipped < 4000, "too many near-surface skips");
            continue;
        }
        let votes: Vec<bool> = DIRS
            .iter()
            .map(|d| stl.iter().filter(|t| ray_crosses(p_mm, *d, t)).count() % 2 == 1)
            .collect();
        if votes[0] != votes[1] || votes[1] != votes[2] {
            skipped += 1;
            continue;
        }
        let p_m = p_mm.map(|v| v * M_PER_MM);
        assert_eq!(
            view.is_inside(p_m),
            votes[0],
            "at {p_mm:?} mm (surface distance {d_surf:.3} mm)"
        );
        tested += 1;
    }
    eprintln!("cushion oracle: {tested} points agreed, {skipped} skipped");

    // The ~1.6 MB hybrid payload must survive template patching (memory
    // growth + data-segment injection) and still parse as wasm.
    let wasm = step_import_operator::patch_template(&payload, true).expect("patch template");
    walrus::Module::from_buffer(&wasm).expect("patched module parses");
}
