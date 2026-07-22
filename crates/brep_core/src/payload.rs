//! Payload serialization (build time) and zero-copy classification
//! (sample time). Byte layout is documented in `lib.rs`; the writer and
//! reader live together here so the format cannot drift.

use crate::bvh::{self, LEAF_FLAG};
use crate::ir::{BRepModel, Face, Solid, Surface};
use crate::math::{self, Affine, Vec3, aabb_union, norm, sub};
use crate::nurbs::{self, MAX_DEGREE};
use crate::surface::{NurbsRaw, NurbsView, Profile2, SurfaceView, f32_at, f64_at, u32_at};
use crate::trim::{LoopSet, Region, TrimLoop};
use crate::{MAGIC, RAY_DIRS};

const HEADER_LEN: usize = 72;
const INSTANCE_LEN: usize = 152;
const SOLID_HEADER_LEN: usize = 24;
const NODE_LEN: usize = 32;
const FACE_HEADER_LEN: usize = 48;

pub const TYPE_PLANE: u32 = 0;
pub const TYPE_CYLINDER: u32 = 1;
pub const TYPE_CONE: u32 = 2;
pub const TYPE_SPHERE: u32 = 3;
pub const TYPE_TORUS: u32 = 4;
pub const TYPE_EXTRUSION: u32 = 5;
pub const TYPE_NURBS: u32 = 6;
pub const TYPE_MESH: u32 = 7;

// ---------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------

struct W {
    buf: Vec<u8>,
}

impl W {
    fn new() -> W {
        W { buf: Vec::new() }
    }
    fn pos(&self) -> usize {
        self.buf.len()
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn pad8(&mut self) {
        while !self.buf.len().is_multiple_of(8) {
            self.buf.push(0);
        }
    }
    fn patch_u32(&mut self, at: usize, v: u32) {
        self.buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// Serialize a model. Fails on structural problems (empty model, bad
/// transforms, invalid NURBS, degenerate trims) — never panics on valid
/// IR.
pub fn build_payload(model: &BRepModel) -> Result<Vec<u8>, String> {
    if model.instances.is_empty() {
        return Err("model has no instances".into());
    }
    if model.solids.is_empty() {
        return Err("model has no solids".into());
    }
    for inst in &model.instances {
        if inst.solid >= model.solids.len() {
            return Err(format!(
                "instance '{}' references solid {} of {}",
                inst.label,
                inst.solid,
                model.solids.len()
            ));
        }
    }

    let mut solid_blobs = Vec::with_capacity(model.solids.len());
    for (i, solid) in model.solids.iter().enumerate() {
        solid_blobs.push(serialize_solid(solid).map_err(|e| format!("solid {i}: {e}"))?);
    }

    let mut world = math::EMPTY_AABB;
    let mut instance_data = Vec::with_capacity(model.instances.len());
    for inst in &model.instances {
        let blob = &solid_blobs[inst.solid];
        let inv = inst
            .local_to_world
            .rigid_inverse()
            .map_err(|e| format!("instance '{}': {e}", inst.label))?;
        let mut aabb = inst.local_to_world.apply_aabb(blob.aabb);
        let margin = blob.eps_boundary * 8.0;
        for axis in 0..3 {
            aabb[axis * 2] -= margin;
            aabb[axis * 2 + 1] += margin;
        }
        world = aabb_union(world, aabb);
        instance_data.push((inst.solid as u32, aabb, inv));
    }

    let mut w = W::new();
    w.u32(MAGIC);
    w.u32(0); // payload_len, patched at the end
    w.u32(model.instances.len() as u32);
    w.u32(model.solids.len() as u32);
    for v in world {
        w.f64(v);
    }
    let instances_off_at = w.pos();
    w.u32(0);
    let solids_off_at = w.pos();
    w.u32(0);
    debug_assert_eq!(w.pos(), HEADER_LEN);

    let instances_off = w.pos();
    w.patch_u32(instances_off_at, instances_off as u32);
    for (solid, aabb, inv) in &instance_data {
        w.u32(*solid);
        w.u32(0);
        for v in aabb {
            w.f64(*v);
        }
        for v in inv.0 {
            w.f64(v);
        }
    }

    let solids_off = w.pos();
    w.patch_u32(solids_off_at, solids_off as u32);
    let table_at = w.pos();
    for _ in &solid_blobs {
        w.u32(0);
    }
    w.pad8();
    for (i, blob) in solid_blobs.iter().enumerate() {
        w.pad8();
        w.patch_u32(table_at + i * 4, w.pos() as u32);
        w.buf.extend_from_slice(&blob.bytes);
    }

    let len = w.pos();
    if len > u32::MAX as usize {
        return Err(format!("payload too large: {len} bytes"));
    }
    w.patch_u32(4, len as u32);
    Ok(w.buf)
}

struct SolidBlob {
    bytes: Vec<u8>,
    aabb: [f64; 6],
    eps_boundary: f64,
}

fn serialize_solid(solid: &Solid) -> Result<SolidBlob, String> {
    if solid.faces.is_empty() {
        return Err("solid has no faces".into());
    }
    for (i, face) in solid.faces.iter().enumerate() {
        validate_face(face).map_err(|e| format!("face {i}: {e}"))?;
    }

    // Conservative 3D bounds per face, then the solid scale, then the
    // scale-derived tolerances.
    let mut aabbs: Vec<[f64; 6]> = solid.faces.iter().map(face_aabb).collect();
    let mut solid_aabb = math::EMPTY_AABB;
    for bb in &aabbs {
        solid_aabb = aabb_union(solid_aabb, *bb);
    }
    let diag = norm([
        solid_aabb[1] - solid_aabb[0],
        solid_aabb[3] - solid_aabb[2],
        solid_aabb[5] - solid_aabb[4],
    ]);
    if !diag.is_finite() {
        return Err("non-finite face bounds".into());
    }
    let t_eps = (diag * 1e-9).max(1e-12);
    let eps_boundary = (diag * 1e-7).max(1e-9);
    for bb in &mut aabbs {
        for axis in 0..3 {
            bb[axis * 2] -= eps_boundary * 4.0;
            bb[axis * 2 + 1] += eps_boundary * 4.0;
        }
    }

    let tree = bvh::build(&aabbs);

    let mut w = W::new();
    w.u32(solid.faces.len() as u32);
    w.u32(tree.nodes.len() as u32);
    w.f64(t_eps);
    w.f64(eps_boundary);
    debug_assert_eq!(w.pos(), SOLID_HEADER_LEN);
    for n in &tree.nodes {
        for v in n.min {
            w.f32(v);
        }
        for v in n.max {
            w.f32(v);
        }
        w.u32(n.a);
        w.u32(n.b);
    }
    let slots_at = w.pos();
    for _ in 0..solid.faces.len() {
        w.u32(0);
    }
    w.pad8();

    let mut face_offsets = vec![0u32; solid.faces.len()];
    for (i, face) in solid.faces.iter().enumerate() {
        w.pad8();
        face_offsets[i] = w.pos() as u32;
        write_face(&mut w, face, eps_boundary).map_err(|e| format!("face {i}: {e}"))?;
    }
    for (s, &face) in tree.slots.iter().enumerate() {
        w.patch_u32(slots_at + s * 4, face_offsets[face as usize]);
    }

    Ok(SolidBlob {
        bytes: w.buf,
        aabb: solid_aabb,
        eps_boundary,
    })
}

/// Face color as stored in the face record: 0 = unstyled, otherwise
/// `0xFF_rr_gg_bb` with 8-bit sRGB components (clamped to [0, 1]; the
/// high byte distinguishes a styled black from "no color").
fn pack_color(color: Option<[f32; 3]>) -> u32 {
    let Some(c) = color else { return 0 };
    let byte = |v: f32| {
        let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        (v * 255.0).round() as u32
    };
    0xFF00_0000 | (byte(c[0]) << 16) | (byte(c[1]) << 8) | byte(c[2])
}

/// Inverse of [`pack_color`].
pub(crate) fn unpack_color(packed: u32) -> Option<[f32; 3]> {
    if packed & 0xFF00_0000 == 0 {
        return None;
    }
    Some([
        ((packed >> 16) & 0xFF) as f32 / 255.0,
        ((packed >> 8) & 0xFF) as f32 / 255.0,
        (packed & 0xFF) as f32 / 255.0,
    ])
}

fn validate_face(face: &Face) -> Result<(), String> {
    if let Surface::Mesh(m) = &face.surface {
        // A mesh face is bounded by its triangulation; UV trims don't
        // apply to it.
        if !face.trims.is_empty() {
            return Err("mesh face with trim loops".into());
        }
        return m.validate();
    }
    if face.trims.is_empty() {
        return Err("face has no trim loops".into());
    }
    for lp in &face.trims {
        if lp.len() < 3 {
            return Err(format!("trim loop with {} points", lp.len()));
        }
        for p in lp {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err("non-finite trim point".into());
            }
        }
    }
    match &face.surface {
        Surface::Cylinder { radius, .. } | Surface::Sphere { radius, .. } => {
            if !matches!(radius.partial_cmp(&0.0), Some(core::cmp::Ordering::Greater)) {
                return Err(format!("non-positive radius {radius}"));
            }
        }
        Surface::Cone { radius, .. } => {
            if matches!(
                radius.partial_cmp(&0.0),
                None | Some(core::cmp::Ordering::Less)
            ) {
                return Err(format!("negative cone radius {radius}"));
            }
        }
        Surface::Torus { major, minor, .. } => {
            if [major, minor]
                .iter()
                .any(|r| !matches!(r.partial_cmp(&&0.0), Some(core::cmp::Ordering::Greater)))
            {
                return Err(format!("non-positive torus radii {major}/{minor}"));
            }
        }
        Surface::ExtrusionPolyline { profile, .. } => {
            if profile.len() < 2 {
                return Err("extrusion profile with < 2 points".into());
            }
        }
        Surface::Nurbs(n) => {
            n.validate()?;
            if n.degree_u > MAX_DEGREE || n.degree_v > MAX_DEGREE {
                return Err(format!(
                    "nurbs degree {}x{} exceeds supported maximum {MAX_DEGREE}",
                    n.degree_u, n.degree_v
                ));
            }
        }
        Surface::Plane { .. } => {}
        Surface::Mesh(_) => unreachable!("handled above"),
    }
    Ok(())
}

/// Open boundary segments of a mesh face — edges used by exactly one
/// triangle — in first-encounter order (deterministic payload bytes).
fn mesh_boundary(m: &crate::ir::MeshSurface) -> Vec<[u32; 2]> {
    use std::collections::HashMap;
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    let edges = |t: &[u32; 3]| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
    for t in &m.tris {
        for (a, b) in edges(t) {
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut out = Vec::new();
    for t in &m.tris {
        for (a, b) in edges(t) {
            let key = (a.min(b), a.max(b));
            if count[&key] == 1 && seen.insert(key) {
                out.push([a, b]);
            }
        }
    }
    out
}

/// Geometric seam-band estimate, per boundary segment: the mesh boundary
/// is a chordal sampling of the exact seam curves of the neighboring
/// faces, so the crack along a chord is bounded by that chord's sagitta.
/// Estimated as the *smaller* second difference of the segment's two
/// endpoints — for a densely sampled smooth curve both are ~4x the sag;
/// at a genuine seam corner the kink inflates only the corner vertex's
/// difference, and the min keeps straight chords beside it at zero
/// rather than swallowing them in the corner angle.
fn mesh_seam_bands(m: &crate::ir::MeshSurface, boundary: &[[u32; 2]]) -> Vec<f64> {
    use std::collections::HashMap;
    // Up to two boundary neighbors per boundary vertex; junction
    // vertices (more than two) keep their first two — the estimate is a
    // heuristic bound, not an exact measure.
    let mut nbrs: HashMap<u32, ([u32; 2], u8)> = HashMap::new();
    for s in boundary {
        for (v, o) in [(s[0], s[1]), (s[1], s[0])] {
            let e = nbrs.entry(v).or_insert(([0, 0], 0));
            if (e.1 as usize) < 2 {
                e.0[e.1 as usize] = o;
                e.1 += 1;
            }
        }
    }
    let vertex_sd = |v: u32| -> f64 {
        let Some(&(nb, n)) = nbrs.get(&v) else {
            return 0.0;
        };
        if n != 2 {
            return 0.0;
        }
        let (a, b, c) = (
            m.verts[nb[0] as usize],
            m.verts[v as usize],
            m.verts[nb[1] as usize],
        );
        norm([
            a[0] - 2.0 * b[0] + c[0],
            a[1] - 2.0 * b[1] + c[1],
            a[2] - 2.0 * b[2] + c[2],
        ])
    };
    boundary
        .iter()
        .map(|s| vertex_sd(s[0]).min(vertex_sd(s[1])) * 0.5)
        .collect()
}

/// UV extent of a face's trim loops.
fn trim_uv_aabb(face: &Face) -> [f64; 4] {
    let mut bb = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for lp in &face.trims {
        for p in lp {
            bb[0] = bb[0].min(p[0]);
            bb[1] = bb[1].max(p[0]);
            bb[2] = bb[2].min(p[1]);
            bb[3] = bb[3].max(p[1]);
        }
    }
    bb
}

/// Conservative 3D bounds of the trimmed face. Parity correctness
/// depends on faces never escaping their box (a missed crossing flips
/// the answer), so every branch here must over- rather than
/// under-estimate.
fn face_aabb(face: &Face) -> [f64; 6] {
    if let Surface::Mesh(m) = &face.surface {
        // Vertices bound the triangles exactly; inflate by the seam
        // band so crack-adjacent rays still route into this face's
        // kernel (the suspicion-scale margin is added per solid).
        let mut bb = math::EMPTY_AABB;
        for v in &m.verts {
            grow(&mut bb, *v);
        }
        let band = mesh_seam_bands(m, &mesh_boundary(m))
            .into_iter()
            .fold(0.0f64, f64::max);
        for axis in 0..3 {
            bb[axis * 2] -= band;
            bb[axis * 2 + 1] += band;
        }
        return bb;
    }
    let view = SurfaceView::from_ir(&face.surface);
    let uv = trim_uv_aabb(face);
    let mut bb = math::EMPTY_AABB;
    match &face.surface {
        // Linear in UV: the trim points' images bound the region exactly.
        Surface::Plane { .. } => {
            for lp in &face.trims {
                for p in lp {
                    grow(&mut bb, view.eval(p[0], p[1]));
                }
            }
            bb
        }
        // Linear in v, piecewise linear in u: profile corners at both v
        // extremes bound the sheet.
        Surface::ExtrusionPolyline { frame, profile } => {
            for p in profile {
                for v in [uv[2], uv[3]] {
                    grow(&mut bb, frame.to_world([p[0], p[1], v]));
                }
            }
            bb
        }
        // Convex-hull property: control points bound the whole surface
        // (weights are validated positive).
        Surface::Nurbs(n) => {
            for c in &n.ctrl {
                grow(&mut bb, [c[0], c[1], c[2]]);
            }
            bb
        }
        // Curved analytics: sample a grid over the trim UV range and
        // inflate by the largest second difference — for these
        // constant-curvature surfaces that is ~4x the single-cell
        // sagitta, so the true surface between samples stays inside
        // with real headroom, without the gross slack of a
        // cell-diagonal margin.
        _ => {
            const N: usize = 17;
            let mut samples = [[0.0f64; 3]; N * N];
            for i in 0..N {
                for j in 0..N {
                    let u = uv[0] + (uv[1] - uv[0]) * i as f64 / (N - 1) as f64;
                    let v = uv[2] + (uv[3] - uv[2]) * j as f64 / (N - 1) as f64;
                    let p = view.eval(u, v);
                    samples[i * N + j] = p;
                    grow(&mut bb, p);
                }
            }
            let mut margin = 0.0f64;
            let mut dev = |a: [f64; 3], m: [f64; 3], b: [f64; 3]| {
                let mid = [
                    (a[0] + b[0]) * 0.5,
                    (a[1] + b[1]) * 0.5,
                    (a[2] + b[2]) * 0.5,
                ];
                margin = margin.max(dist(m, mid));
            };
            for i in 0..N {
                for j in 0..N {
                    if i + 2 < N {
                        dev(
                            samples[i * N + j],
                            samples[(i + 1) * N + j],
                            samples[(i + 2) * N + j],
                        );
                    }
                    if j + 2 < N {
                        dev(
                            samples[i * N + j],
                            samples[i * N + j + 1],
                            samples[i * N + j + 2],
                        );
                    }
                }
            }
            // The second difference is ~4x the single-cell sagitta;
            // half of it still covers the worst-case diagonal
            // deviation (u sagitta + v sagitta) with headroom.
            let margin = margin * 0.5;
            for axis in 0..3 {
                bb[axis * 2] -= margin;
                bb[axis * 2 + 1] += margin;
            }
            bb
        }
    }
}

fn grow(bb: &mut [f64; 6], p: Vec3) {
    for axis in 0..3 {
        bb[axis * 2] = bb[axis * 2].min(p[axis]);
        bb[axis * 2 + 1] = bb[axis * 2 + 1].max(p[axis]);
    }
}

fn dist(a: Vec3, b: Vec3) -> f64 {
    norm(sub(a, b))
}

/// The 3D length of a unit UV step along each axis, by central
/// differences at several spots across the trim range (the metric can
/// vary strongly — a sphere trim reaching toward a pole) — converts
/// the 3D suspicion tolerance into UV units for the trim boundary
/// test. The *smallest* metric wins: it yields the widest uv_eps, so
/// suspicion errs toward re-casting.
fn uv_metric(view: &SurfaceView, uv: [f64; 4]) -> [f64; 2] {
    let hu = ((uv[1] - uv[0]) * 1e-3).max(1e-9);
    let hv = ((uv[3] - uv[2]) * 1e-3).max(1e-9);
    let mut metric = [f64::INFINITY; 2];
    for (fu, fv) in [(0.5, 0.5), (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)] {
        let cu = uv[0] + (uv[1] - uv[0]) * fu;
        let cv = uv[2] + (uv[3] - uv[2]) * fv;
        let mu = dist(view.eval(cu + hu, cv), view.eval(cu - hu, cv)) / (2.0 * hu);
        let mv = dist(view.eval(cu, cv + hv), view.eval(cu, cv - hv)) / (2.0 * hv);
        if mu > 1e-12 {
            metric[0] = metric[0].min(mu);
        }
        if mv > 1e-12 {
            metric[1] = metric[1].min(mv);
        }
    }
    [
        metric[0].clamp(1e-12, f64::MAX),
        metric[1].clamp(1e-12, f64::MAX),
    ]
}

fn write_face(w: &mut W, face: &Face, eps_boundary: f64) -> Result<(), String> {
    if let Surface::Mesh(m) = &face.surface {
        return write_mesh_face(w, m, face.color, eps_boundary);
    }
    let face_base = w.pos();
    let view = SurfaceView::from_ir(&face.surface);
    let uv = trim_uv_aabb(face);
    let metric = uv_metric(&view, uv);
    let (u_period, v_period) = face.surface.periods();

    let type_id = match &face.surface {
        Surface::Plane { .. } => TYPE_PLANE,
        Surface::Cylinder { .. } => TYPE_CYLINDER,
        Surface::Cone { .. } => TYPE_CONE,
        Surface::Sphere { .. } => TYPE_SPHERE,
        Surface::Torus { .. } => TYPE_TORUS,
        Surface::ExtrusionPolyline { .. } => TYPE_EXTRUSION,
        Surface::Nurbs(_) => TYPE_NURBS,
        Surface::Mesh(_) => unreachable!("mesh faces take the write_mesh_face path"),
    };
    w.u32(type_id);
    w.u32(pack_color(face.color));
    w.f64(eps_boundary / metric[0]);
    w.f64(eps_boundary / metric[1]);
    w.f64(u_period);
    w.f64(v_period);
    let trims_off_at = w.pos();
    w.u32(0);
    w.u32(0);
    debug_assert_eq!(w.pos() - face_base, FACE_HEADER_LEN);

    match &face.surface {
        Surface::Plane { frame } => {
            for v in frame.flat() {
                w.f64(v);
            }
        }
        Surface::Cylinder { frame, radius } => {
            for v in frame.flat() {
                w.f64(v);
            }
            w.f64(*radius);
        }
        Surface::Cone {
            frame,
            radius,
            half_angle,
        } => {
            for v in frame.flat() {
                w.f64(v);
            }
            w.f64(*radius);
            w.f64(half_angle.tan());
        }
        Surface::Sphere { frame, radius } => {
            for v in frame.flat() {
                w.f64(v);
            }
            w.f64(*radius);
        }
        Surface::Torus {
            frame,
            major,
            minor,
        } => {
            for v in frame.flat() {
                w.f64(v);
            }
            w.f64(*major);
            w.f64(*minor);
        }
        Surface::ExtrusionPolyline { frame, profile } => {
            for v in frame.flat() {
                w.f64(v);
            }
            w.u32(profile.len() as u32);
            w.u32(0);
            for p in profile {
                w.f64(p[0]);
                w.f64(p[1]);
            }
        }
        Surface::Nurbs(n) => {
            let (seed_nu, seed_nv) = seed_grid(n);
            w.u32(n.degree_u as u32);
            w.u32(n.degree_v as u32);
            w.u32(n.nctrl_u as u32);
            w.u32(n.nctrl_v as u32);
            w.u32(n.knots_u.len() as u32);
            w.u32(n.knots_v.len() as u32);
            w.u32(seed_nu as u32);
            w.u32(seed_nv as u32);
            for &k in &n.knots_u {
                w.f64(k);
            }
            for &k in &n.knots_v {
                w.f64(k);
            }
            for c in &n.ctrl {
                for v in c {
                    w.f64(*v);
                }
            }
            write_seed_boxes(w, n, seed_nu, seed_nv);
        }
        Surface::Mesh(_) => unreachable!("mesh faces take the write_mesh_face path"),
    }

    w.pad8();
    let trims_off = w.pos() - face_base;
    w.patch_u32(trims_off_at, trims_off as u32);

    let trim_base = w.pos();
    w.u32(face.trims.len() as u32);
    w.u32(0);
    for v in uv {
        w.f64(v);
    }
    let loop_table_at = w.pos();
    for _ in &face.trims {
        w.u32(0);
    }
    w.pad8();
    for (i, lp) in face.trims.iter().enumerate() {
        w.pad8();
        w.patch_u32(loop_table_at + i * 4, (w.pos() - trim_base) as u32);
        w.u32(lp.len() as u32);
        w.u32(0);
        for p in lp {
            w.f64(p[0]);
            w.f64(p[1]);
        }
    }
    Ok(())
}

/// Serialize one mesh face (layout in `lib.rs`). Triangles and boundary
/// segments are stored in BVH leaf-contiguous order so leaves index the
/// arrays directly — no slot table.
fn write_mesh_face(
    w: &mut W,
    m: &crate::ir::MeshSurface,
    color: Option<[f32; 3]>,
    eps_boundary: f64,
) -> Result<(), String> {
    let boundary = mesh_boundary(m);
    let bands: Vec<f64> = mesh_seam_bands(m, &boundary)
        .into_iter()
        .map(|b| b.max(eps_boundary * 4.0))
        .collect();
    let band_max = bands.iter().copied().fold(0.0f64, f64::max);

    let tri_pad = eps_boundary * 4.0;
    let tri_aabbs: Vec<[f64; 6]> = m
        .tris
        .iter()
        .map(|t| {
            let mut bb = math::EMPTY_AABB;
            for &i in t {
                grow(&mut bb, m.verts[i as usize]);
            }
            for axis in 0..3 {
                bb[axis * 2] -= tri_pad;
                bb[axis * 2 + 1] += tri_pad;
            }
            bb
        })
        .collect();
    let tri_bvh = bvh::build(&tri_aabbs);
    let tris: Vec<[u32; 3]> = tri_bvh
        .slots
        .iter()
        .map(|&s| m.tris[s as usize])
        .collect();

    let (seg_bvh, segs) = if boundary.is_empty() {
        (None, Vec::new())
    } else {
        let seg_aabbs: Vec<[f64; 6]> = boundary
            .iter()
            .zip(&bands)
            .map(|(s, &band)| {
                let mut bb = math::EMPTY_AABB;
                for &i in s {
                    grow(&mut bb, m.verts[i as usize]);
                }
                let pad = band + eps_boundary * 4.0;
                for axis in 0..3 {
                    bb[axis * 2] -= pad;
                    bb[axis * 2 + 1] += pad;
                }
                bb
            })
            .collect();
        let b = bvh::build(&seg_aabbs);
        let segs: Vec<([u32; 2], f64)> = b
            .slots
            .iter()
            .map(|&s| (boundary[s as usize], bands[s as usize]))
            .collect();
        (Some(b), segs)
    };
    let seg_node_count = seg_bvh.as_ref().map_or(0, |b| b.nodes.len());

    let face_base = w.pos();
    w.u32(TYPE_MESH);
    w.u32(pack_color(color));
    w.f64(0.0); // uv_eps (no UV space)
    w.f64(0.0);
    w.f64(0.0); // periods
    w.f64(0.0);
    let trims_off_at = w.pos();
    w.u32(0);
    w.u32(0);
    debug_assert_eq!(w.pos() - face_base, FACE_HEADER_LEN);

    w.u32(m.verts.len() as u32);
    w.u32(tris.len() as u32);
    w.u32(segs.len() as u32);
    w.u32(tri_bvh.nodes.len() as u32);
    w.f64(band_max);
    w.u32(seg_node_count as u32);
    w.u32(0);
    let write_nodes = |w: &mut W, nodes: &[bvh::Node]| {
        for n in nodes {
            for v in n.min {
                w.f32(v);
            }
            for v in n.max {
                w.f32(v);
            }
            w.u32(n.a);
            w.u32(n.b);
        }
    };
    write_nodes(w, &tri_bvh.nodes);
    if let Some(b) = &seg_bvh {
        write_nodes(w, &b.nodes);
    }
    for v in &m.verts {
        for c in v {
            w.f64(*c);
        }
    }
    for t in &tris {
        for &i in t {
            w.u32(i);
        }
    }
    for (s, band) in &segs {
        w.u32(s[0]);
        w.u32(s[1]);
        w.f32(up(*band)); // round toward wider: the band must contain the crack
    }

    // Empty trim blob, so the record shape matches every other face.
    w.pad8();
    let trims_off = w.pos() - face_base;
    w.patch_u32(trims_off_at, trims_off as u32);
    w.u32(0);
    w.u32(0);
    for _ in 0..4 {
        w.f64(0.0);
    }
    Ok(())
}

fn seed_grid(n: &crate::ir::NurbsSurface) -> (usize, usize) {
    let nu = ((n.nctrl_u - n.degree_u) * 2).clamp(4, 12);
    let nv = ((n.nctrl_v - n.degree_v) * 2).clamp(4, 12);
    (nu, nv)
}

/// Newton seed boxes: a UV grid of cells, each with a conservative AABB
/// of its surface patch (5x5 samples inflated by the largest
/// adjacent-sample distance) and the cell's UV center as the Newton
/// starting point.
fn write_seed_boxes(w: &mut W, n: &crate::ir::NurbsSurface, seed_nu: usize, seed_nv: usize) {
    let dom = n.domain();
    for i in 0..seed_nu {
        for j in 0..seed_nv {
            let u0 = dom[0] + (dom[1] - dom[0]) * i as f64 / seed_nu as f64;
            let u1 = dom[0] + (dom[1] - dom[0]) * (i + 1) as f64 / seed_nu as f64;
            let v0 = dom[2] + (dom[3] - dom[2]) * j as f64 / seed_nv as f64;
            let v1 = dom[2] + (dom[3] - dom[2]) * (j + 1) as f64 / seed_nv as f64;
            const S: usize = 5;
            let mut bb = math::EMPTY_AABB;
            let mut samples = [[0.0f64; 3]; S * S];
            for a in 0..S {
                for b in 0..S {
                    let u = u0 + (u1 - u0) * a as f64 / (S - 1) as f64;
                    let v = v0 + (v1 - v0) * b as f64 / (S - 1) as f64;
                    let p = nurbs::surface_eval(n, u, v).0;
                    samples[a * S + b] = p;
                    grow(&mut bb, p);
                }
            }
            let mut cell = 0.0f64;
            for a in 0..S {
                for b in 0..S {
                    if a + 1 < S {
                        cell = cell.max(dist(samples[a * S + b], samples[(a + 1) * S + b]));
                    }
                    if b + 1 < S {
                        cell = cell.max(dist(samples[a * S + b], samples[a * S + b + 1]));
                    }
                }
            }
            w.f32(down(bb[0] - cell));
            w.f32(down(bb[2] - cell));
            w.f32(down(bb[4] - cell));
            w.f32(up(bb[1] + cell));
            w.f32(up(bb[3] + cell));
            w.f32(up(bb[5] + cell));
            w.f32(((u0 + u1) * 0.5) as f32);
            w.f32(((v0 + v1) * 0.5) as f32);
        }
    }
}

fn up(x: f64) -> f32 {
    let f = x as f32;
    if (f as f64) < x {
        f32::from_bits(if f > 0.0 {
            f.to_bits() + 1
        } else if f < 0.0 {
            f.to_bits() - 1
        } else {
            1
        })
    } else {
        f
    }
}

fn down(x: f64) -> f32 {
    -up(-x)
}

// ---------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------

/// Zero-copy view over payload bytes; all reads are bounds-checked by
/// slicing (a corrupt payload panics rather than reading out of bounds —
/// the template treats a failed [`PayloadView::new`] as an empty model,
/// and the builder-verified structure guarantees in-bounds offsets).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
}

impl<'a> PayloadView<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<PayloadView<'a>, String> {
        if bytes.len() < HEADER_LEN {
            return Err(format!("payload too short: {}", bytes.len()));
        }
        if u32_at(bytes, 0) != MAGIC {
            return Err("bad payload magic".into());
        }
        let len = u32_at(bytes, 4) as usize;
        if len > bytes.len() {
            return Err(format!(
                "payload length field {len} exceeds buffer {}",
                bytes.len()
            ));
        }
        Ok(PayloadView { bytes })
    }

    pub fn bounds(&self) -> [f64; 6] {
        core::array::from_fn(|i| f64_at(self.bytes, 16 + i * 8))
    }

    pub fn instance_count(&self) -> usize {
        u32_at(self.bytes, 8) as usize
    }

    /// Point-in-model: union over instances of per-solid ray parity.
    pub fn is_inside(&self, p: Vec3) -> bool {
        let instances_off = u32_at(self.bytes, 64) as usize;
        let solids_off = u32_at(self.bytes, 68) as usize;
        for i in 0..self.instance_count() {
            let base = instances_off + i * INSTANCE_LEN;
            let mut outside = false;
            for (axis, &c) in p.iter().enumerate() {
                let lo = f64_at(self.bytes, base + 8 + axis * 16);
                let hi = f64_at(self.bytes, base + 16 + axis * 16);
                if c < lo || c > hi {
                    outside = true;
                    break;
                }
            }
            if outside {
                continue;
            }
            let w2l = Affine(core::array::from_fn(|k| {
                f64_at(self.bytes, base + 56 + k * 8)
            }));
            let q = w2l.apply(p);
            let solid = u32_at(self.bytes, base) as usize;
            let solid_off = u32_at(self.bytes, solids_off + solid * 4) as usize;
            if self.solid_parity(solid_off, q) {
                return true;
            }
        }
        false
    }

    /// Parity for one solid in solid-local coordinates, with re-casts on
    /// suspect classifications. The last cast's parity stands when every
    /// direction is suspect — by then the point is within tolerance of a
    /// boundary and either answer is acceptable.
    fn solid_parity(&self, solid_off: usize, p: Vec3) -> bool {
        let mut parity = false;
        for dir in RAY_DIRS {
            let (par, suspect) = self.cast(solid_off, p, dir);
            parity = par;
            if !suspect {
                break;
            }
        }
        parity
    }

    fn cast(&self, solid_off: usize, p: Vec3, dir: Vec3) -> (bool, bool) {
        let bytes = self.bytes;
        let face_count = u32_at(bytes, solid_off) as usize;
        let node_count = u32_at(bytes, solid_off + 4) as usize;
        let t_eps = f64_at(bytes, solid_off + 8);
        let eps_boundary = f64_at(bytes, solid_off + 16);
        let nodes_off = solid_off + SOLID_HEADER_LEN;
        let slots_off = nodes_off + node_count * NODE_LEN;

        let mut crossings = 0usize;
        let mut suspect = false;
        let t_floor = -4.0 * t_eps;

        let mut stack = [0u32; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;
        while top > 0 {
            top -= 1;
            let node = stack[top] as usize;
            let base = nodes_off + node * NODE_LEN;
            let mut t_min = t_floor;
            let mut t_max = f64::INFINITY;
            for axis in 0..3 {
                let inv = 1.0 / dir[axis];
                let mut lo = (f32_at(bytes, base + axis * 4) as f64 - p[axis]) * inv;
                let mut hi = (f32_at(bytes, base + 12 + axis * 4) as f64 - p[axis]) * inv;
                if lo > hi {
                    core::mem::swap(&mut lo, &mut hi);
                }
                t_min = t_min.max(lo);
                t_max = t_max.min(hi);
            }
            if t_min > t_max {
                continue;
            }
            let a = u32_at(bytes, base + 24);
            let b = u32_at(bytes, base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for s in first..first + b as usize {
                    debug_assert!(s < face_count);
                    let face_off = solid_off + u32_at(bytes, slots_off + s * 4) as usize;
                    self.face_cast(
                        face_off,
                        p,
                        dir,
                        t_eps,
                        eps_boundary,
                        &mut crossings,
                        &mut suspect,
                    );
                }
            } else {
                stack[top] = a;
                top += 1;
                stack[top] = b;
                top += 1;
            }
        }
        (crossings % 2 == 1, suspect)
    }

    #[allow(clippy::too_many_arguments)]
    fn face_cast(
        &self,
        face_off: usize,
        p: Vec3,
        dir: Vec3,
        t_eps: f64,
        eps_boundary: f64,
        crossings: &mut usize,
        suspect: &mut bool,
    ) {
        if u32_at(self.bytes, face_off) == TYPE_MESH {
            MeshRaw::at(self.bytes, face_off).cast(p, dir, t_eps, eps_boundary, crossings, suspect);
            return;
        }
        let region = self.region_at(face_off);
        let view = self.surface_view(face_off);
        view.ray_hits(p, dir, eps_boundary, &mut |hit| {
            if hit.t < -4.0 * t_eps && !hit.suspect {
                return;
            }
            let th = region.contains(hit.u, hit.v);
            if hit.counts && th.inside && hit.t > t_eps {
                *crossings += 1;
                if hit.suspect || th.near_boundary {
                    *suspect = true;
                }
                return;
            }
            // Non-counting markers, behind-the-origin grazes, on-surface
            // queries, and near-trim-boundary misses: suspicion only.
            if hit.suspect && (th.inside || th.near_boundary) {
                *suspect = true;
            }
            if hit.counts && th.near_boundary {
                *suspect = true;
            }
            if hit.counts && th.inside && hit.t.abs() <= t_eps {
                *suspect = true;
            }
        });
    }

    /// The trim region of the face record at `face_off`.
    fn region_at(&self, face_off: usize) -> Region<RawLoops<'a>> {
        let bytes = self.bytes;
        let uv_eps = [f64_at(bytes, face_off + 8), f64_at(bytes, face_off + 16)];
        let u_period = f64_at(bytes, face_off + 24);
        let v_period = f64_at(bytes, face_off + 32);
        let trims_off = face_off + u32_at(bytes, face_off + 40) as usize;

        let loop_count = u32_at(bytes, trims_off) as usize;
        let uv_aabb: [f64; 4] = core::array::from_fn(|i| f64_at(bytes, trims_off + 8 + i * 8));
        Region {
            loops: RawLoops {
                bytes,
                base: trims_off,
                count: loop_count,
            },
            uv_aabb,
            u_period,
            v_period,
            uv_eps,
        }
    }

    /// Display color of the model's nearest face at `p` (sRGB components
    /// in [0, 1]); unstyled faces read as white. Distances are measured
    /// to the *trimmed* faces (projections landing outside a face's trim
    /// region are clamped to its trim boundary), across every instance.
    /// `None` only for a payload with no reachable faces.
    ///
    /// This ranks faces by proximity — the NURBS distance is a polished
    /// local minimum, not a certified bound — which is exactly enough
    /// for color lookup at or near the surface, and deliberately plays
    /// no part in occupancy classification.
    pub fn nearest_color(&self, p: Vec3) -> Option<[f32; 3]> {
        let instances_off = u32_at(self.bytes, 64) as usize;
        let solids_off = u32_at(self.bytes, 68) as usize;
        let mut best: Option<(f64, u32)> = None; // (world distance, packed color)
        for i in 0..self.instance_count() {
            let base = instances_off + i * INSTANCE_LEN;
            // World-AABB prune: skip instances that cannot beat the
            // running best (most of them, in a populated assembly).
            let mut aabb_d2 = 0.0f64;
            for (axis, &c) in p.iter().enumerate() {
                let lo = f64_at(self.bytes, base + 8 + axis * 16);
                let hi = f64_at(self.bytes, base + 16 + axis * 16);
                let gap = (lo - c).max(c - hi).max(0.0);
                aabb_d2 += gap * gap;
            }
            if let Some((bd, _)) = best
                && aabb_d2 >= bd * bd
            {
                continue;
            }
            let w2l = Affine(core::array::from_fn(|k| {
                f64_at(self.bytes, base + 56 + k * 8)
            }));
            // world-to-local is rigid + uniform scale: its row norm is
            // 1/s, and local distances scale back to world by s.
            let inv_s = norm([w2l.0[0], w2l.0[1], w2l.0[2]]);
            if !(inv_s > 0.0 && inv_s.is_finite()) {
                continue;
            }
            let s = 1.0 / inv_s;
            let q = w2l.apply(p);
            let solid = u32_at(self.bytes, base) as usize;
            let solid_off = u32_at(self.bytes, solids_off + solid * 4) as usize;
            let limit_local = best.map_or(f64::INFINITY, |(d, _)| d / s);
            if let Some((d_local, packed)) = self.solid_nearest_face(solid_off, q, limit_local) {
                best = Some((d_local * s, packed));
            }
        }
        best.map(|(_, packed)| unpack_color(packed).unwrap_or([1.0, 1.0, 1.0]))
    }

    /// Nearest face of one solid (solid-local coordinates): `(distance,
    /// packed color)`, only when some face beats `limit`. Best-first BVH
    /// descent pruned against the running best.
    fn solid_nearest_face(&self, solid_off: usize, p: Vec3, limit: f64) -> Option<(f64, u32)> {
        let bytes = self.bytes;
        let node_count = u32_at(bytes, solid_off + 4) as usize;
        let nodes_off = solid_off + SOLID_HEADER_LEN;
        let slots_off = nodes_off + node_count * NODE_LEN;
        if node_count == 0 {
            return None;
        }
        let node_dist = |node: usize| -> f64 {
            let base = nodes_off + node * NODE_LEN;
            let mut d2 = 0.0f64;
            for (axis, &c) in p.iter().enumerate() {
                let lo = f32_at(bytes, base + axis * 4) as f64;
                let hi = f32_at(bytes, base + 12 + axis * 4) as f64;
                let gap = (lo - c).max(c - hi).max(0.0);
                d2 += gap * gap;
            }
            d2.sqrt()
        };

        let mut best: Option<(f64, u32)> = None;
        let mut bound = limit;
        let mut stack = [(0u32, 0.0f64); 64];
        stack[0] = (0, node_dist(0));
        let mut top = 1usize;
        while top > 0 {
            top -= 1;
            let (node, d) = stack[top];
            if d >= bound {
                continue;
            }
            let base = nodes_off + node as usize * NODE_LEN;
            let a = u32_at(bytes, base + 24);
            let b = u32_at(bytes, base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for s in first..first + b as usize {
                    let face_off = solid_off + u32_at(bytes, slots_off + s * 4) as usize;
                    let (fd, packed) = self.face_distance(face_off, p);
                    if fd < bound {
                        bound = fd;
                        best = Some((fd, packed));
                    }
                }
            } else {
                // Nearer child popped first; the farther one re-checks
                // its distance against the (possibly tightened) bound.
                let (da, db) = (node_dist(a as usize), node_dist(b as usize));
                let ordered = if da <= db { [(b, db), (a, da)] } else { [(a, da), (b, db)] };
                for (child, dist) in ordered {
                    if dist < bound {
                        debug_assert!(top < stack.len(), "BVH deeper than the traversal stack");
                        if top < stack.len() {
                            stack[top] = (child, dist);
                            top += 1;
                        }
                    }
                }
            }
        }
        best
    }

    /// Distance from `p` to one trimmed face, with its packed color.
    /// Off-trim projections clamp to the trim boundary (approximately —
    /// the clamp is exact in UV, and re-measured in 3D).
    fn face_distance(&self, face_off: usize, p: Vec3) -> (f64, u32) {
        if u32_at(self.bytes, face_off) == TYPE_MESH {
            let d = MeshRaw::at(self.bytes, face_off).distance(p);
            return (d, u32_at(self.bytes, face_off + 4));
        }
        let view = self.surface_view(face_off);
        let (uv, d_surf) = view.closest(p);
        let region = self.region_at(face_off);
        let d = if region.contains(uv[0], uv[1]).inside {
            d_surf
        } else {
            let c = region.nearest_boundary_uv(uv[0], uv[1]);
            norm(sub(p, view.eval(c[0], c[1])))
        };
        (d, u32_at(self.bytes, face_off + 4))
    }

    fn surface_view(&self, face_off: usize) -> SurfaceView<'a> {
        let bytes = self.bytes;
        let type_id = u32_at(bytes, face_off);
        let data = face_off + FACE_HEADER_LEN;
        let frame = || {
            let f: [f64; 12] = core::array::from_fn(|i| f64_at(bytes, data + i * 8));
            math::Frame::from_flat(&f)
        };
        match type_id {
            TYPE_PLANE => SurfaceView::Plane { frame: frame() },
            TYPE_CYLINDER => SurfaceView::Cylinder {
                frame: frame(),
                radius: f64_at(bytes, data + 96),
            },
            TYPE_CONE => SurfaceView::Cone {
                frame: frame(),
                radius: f64_at(bytes, data + 96),
                tan_half: f64_at(bytes, data + 104),
            },
            TYPE_SPHERE => SurfaceView::Sphere {
                frame: frame(),
                radius: f64_at(bytes, data + 96),
            },
            TYPE_TORUS => SurfaceView::Torus {
                frame: frame(),
                major: f64_at(bytes, data + 96),
                minor: f64_at(bytes, data + 104),
            },
            TYPE_EXTRUSION => SurfaceView::Extrusion {
                frame: frame(),
                profile: Profile2::Raw {
                    bytes,
                    offset: data + 104,
                    count: u32_at(bytes, data + 96) as usize,
                },
            },
            TYPE_NURBS => {
                let h = |i: usize| u32_at(bytes, data + i * 4) as usize;
                let (degree_u, degree_v) = (h(0), h(1));
                let (nctrl_u, nctrl_v) = (h(2), h(3));
                let (nknot_u, nknot_v) = (h(4), h(5));
                let (seed_nu, seed_nv) = (h(6), h(7));
                let knots_u_off = data + 32;
                let knots_v_off = knots_u_off + nknot_u * 8;
                let ctrl_off = knots_v_off + nknot_v * 8;
                let seeds_off = ctrl_off + nctrl_u * nctrl_v * 32;
                SurfaceView::Nurbs(NurbsView::Raw(NurbsRaw {
                    bytes,
                    degree_u,
                    degree_v,
                    nctrl_u,
                    nctrl_v,
                    knots_u_off,
                    knots_v_off,
                    ctrl_off,
                    seeds_off,
                    seed_nu,
                    seed_nv,
                }))
            }
            _ => {
                debug_assert!(false, "unknown surface type {type_id}");
                SurfaceView::Plane {
                    frame: math::Frame::IDENTITY,
                }
            }
        }
    }
}

/// A mesh face read from payload bytes (layout in `lib.rs`).
///
/// # Parity safety
///
/// Same contract as the analytic kernels: transversal interior crossings
/// count; hits within `eps` of a triangle edge, grazing-incidence hits,
/// and near-misses beside an edge set `suspect` (an exactly-on-edge
/// crossing may count twice or not at all — the re-cast resolves it).
/// Rays passing within the seam band of an open boundary segment set
/// `suspect` too: the band covers the chordal crack between this mesh
/// and the exact faces it glues against, where a crossing can be missed
/// entirely (odd count) without any triangle noticing.
#[derive(Clone, Copy)]
struct MeshRaw<'a> {
    bytes: &'a [u8],
    band: f64,
    tri_count: usize,
    seg_count: usize,
    nodes_off: usize,
    seg_nodes_off: usize,
    verts_off: usize,
    tris_off: usize,
    segs_off: usize,
}

impl<'a> MeshRaw<'a> {
    fn at(bytes: &'a [u8], face_off: usize) -> MeshRaw<'a> {
        let data = face_off + FACE_HEADER_LEN;
        let vert_count = u32_at(bytes, data) as usize;
        let tri_count = u32_at(bytes, data + 4) as usize;
        let seg_count = u32_at(bytes, data + 8) as usize;
        let node_count = u32_at(bytes, data + 12) as usize;
        let band = f64_at(bytes, data + 16);
        let seg_node_count = u32_at(bytes, data + 24) as usize;
        let nodes_off = data + 32;
        let seg_nodes_off = nodes_off + node_count * NODE_LEN;
        let verts_off = seg_nodes_off + seg_node_count * NODE_LEN;
        let tris_off = verts_off + vert_count * 24;
        let segs_off = tris_off + tri_count * 12;
        MeshRaw {
            bytes,
            band,
            tri_count,
            seg_count,
            nodes_off,
            seg_nodes_off,
            verts_off,
            tris_off,
            segs_off,
        }
    }

    fn vert(&self, i: usize) -> Vec3 {
        let base = self.verts_off + i * 24;
        [
            f64_at(self.bytes, base),
            f64_at(self.bytes, base + 8),
            f64_at(self.bytes, base + 16),
        ]
    }

    fn tri(&self, i: usize) -> [Vec3; 3] {
        let base = self.tris_off + i * 12;
        [
            self.vert(u32_at(self.bytes, base) as usize),
            self.vert(u32_at(self.bytes, base + 4) as usize),
            self.vert(u32_at(self.bytes, base + 8) as usize),
        ]
    }

    /// Boundary segment endpoints and its seam band.
    fn seg(&self, i: usize) -> ([Vec3; 2], f64) {
        let base = self.segs_off + i * 12;
        (
            [
                self.vert(u32_at(self.bytes, base) as usize),
                self.vert(u32_at(self.bytes, base + 4) as usize),
            ],
            f32_at(self.bytes, base + 8) as f64,
        )
    }

    /// Ray slab test against a BVH node, `(t_min, t_max)` clipped to
    /// `[t_floor, inf)`; `None` when the ray misses the box.
    fn node_span(
        &self,
        nodes_off: usize,
        node: usize,
        p: Vec3,
        dir: Vec3,
        t_floor: f64,
    ) -> Option<(u32, u32)> {
        let base = nodes_off + node * NODE_LEN;
        let mut t_min = t_floor;
        let mut t_max = f64::INFINITY;
        for axis in 0..3 {
            let inv = 1.0 / dir[axis];
            let mut lo = (f32_at(self.bytes, base + axis * 4) as f64 - p[axis]) * inv;
            let mut hi = (f32_at(self.bytes, base + 12 + axis * 4) as f64 - p[axis]) * inv;
            if lo > hi {
                core::mem::swap(&mut lo, &mut hi);
            }
            t_min = t_min.max(lo);
            t_max = t_max.min(hi);
        }
        if t_min > t_max {
            return None;
        }
        Some((
            u32_at(self.bytes, base + 24),
            u32_at(self.bytes, base + 28),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn cast(
        &self,
        p: Vec3,
        dir: Vec3,
        t_eps: f64,
        eps: f64,
        crossings: &mut usize,
        suspect: &mut bool,
    ) {
        let t_floor = -4.0 * t_eps;
        let dir_len = norm(dir);

        let mut stack = [0u32; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;
        while top > 0 {
            top -= 1;
            let Some((a, b)) = self.node_span(self.nodes_off, stack[top] as usize, p, dir, t_floor)
            else {
                continue;
            };
            if a & LEAF_FLAG == 0 {
                stack[top] = a;
                top += 1;
                stack[top] = b;
                top += 1;
                continue;
            }
            let first = (a & !LEAF_FLAG) as usize;
            for i in first..first + b as usize {
                debug_assert!(i < self.tri_count);
                let [v0, v1, v2] = self.tri(i);
                let e1 = sub(v1, v0);
                let e2 = sub(v2, v0);
                let n = math::cross(e1, e2);
                let n_len = norm(n);
                if n_len == 0.0 {
                    continue; // degenerate (rejected at build; belt and braces)
                }
                let denom = math::dot(n, dir);
                if denom.abs() < 1e-9 * n_len * dir_len {
                    // In-plane ray: never a transversal crossing; a skim
                    // along the triangle itself earns a re-cast.
                    if math::point_triangle_dist(p, v0, v1, v2) < eps * 4.0 {
                        *suspect = true;
                    }
                    continue;
                }
                let t = math::dot(n, sub(v0, p)) / denom;
                let q = [p[0] + t * dir[0], p[1] + t * dir[1], p[2] + t * dir[2]];
                let inside = math::dot(math::cross(e1, sub(q, v0)), n) >= 0.0
                    && math::dot(math::cross(sub(v2, v1), sub(q, v1)), n) >= 0.0
                    && math::dot(math::cross(sub(v0, v2), sub(q, v2)), n) >= 0.0;
                let d_edge = math::point_segment_dist(q, v0, v1)
                    .min(math::point_segment_dist(q, v1, v2))
                    .min(math::point_segment_dist(q, v2, v0));
                if !inside && d_edge >= eps {
                    continue;
                }
                let grazing = denom.abs() < 1e-4 * n_len * dir_len;
                let suspect_hit = grazing || d_edge < eps;
                if t < t_floor && !suspect_hit {
                    continue;
                }
                if inside && t > t_eps {
                    *crossings += 1;
                    if suspect_hit {
                        *suspect = true;
                    }
                } else if suspect_hit || (inside && t.abs() <= t_eps) {
                    // Near-miss beside an edge, grazing hit behind the
                    // origin, or an on-surface query: re-cast.
                    *suspect = true;
                }
            }
        }

        // Seam pass: a ray through the boundary crack crosses the solid's
        // surface without crossing any face — flag it. Skipped when this
        // cast is already suspect.
        if *suspect || self.seg_count == 0 {
            return;
        }
        let reach = self.band + 4.0 * t_eps;
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;
        while top > 0 {
            top -= 1;
            let Some((a, b)) =
                self.node_span(self.seg_nodes_off, stack[top] as usize, p, dir, t_floor - reach)
            else {
                continue;
            };
            if a & LEAF_FLAG == 0 {
                stack[top] = a;
                top += 1;
                stack[top] = b;
                top += 1;
                continue;
            }
            let first = (a & !LEAF_FLAG) as usize;
            for i in first..first + b as usize {
                debug_assert!(i < self.seg_count);
                let ([sa, sb], band) = self.seg(i);
                let (t_ray, d) = ray_segment_approach(p, dir, sa, sb);
                if d < band && t_ray > t_floor - reach {
                    *suspect = true;
                    return;
                }
            }
        }
    }

    /// Distance from `p` to the mesh: best-first BVH descent.
    fn distance(&self, p: Vec3) -> f64 {
        let node_dist = |node: usize| -> f64 {
            let base = self.nodes_off + node * NODE_LEN;
            let mut d2 = 0.0f64;
            for (axis, &c) in p.iter().enumerate() {
                let lo = f32_at(self.bytes, base + axis * 4) as f64;
                let hi = f32_at(self.bytes, base + 12 + axis * 4) as f64;
                let gap = (lo - c).max(c - hi).max(0.0);
                d2 += gap * gap;
            }
            d2.sqrt()
        };
        let mut best = f64::INFINITY;
        let mut stack = [(0u32, 0.0f64); 64];
        stack[0] = (0, node_dist(0));
        let mut top = 1usize;
        while top > 0 {
            top -= 1;
            let (node, d) = stack[top];
            if d >= best {
                continue;
            }
            let base = self.nodes_off + node as usize * NODE_LEN;
            let a = u32_at(self.bytes, base + 24);
            let b = u32_at(self.bytes, base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for i in first..first + b as usize {
                    let [v0, v1, v2] = self.tri(i);
                    best = best.min(math::point_triangle_dist(p, v0, v1, v2));
                }
            } else {
                let (da, db) = (node_dist(a as usize), node_dist(b as usize));
                let ordered = if da <= db {
                    [(b, db), (a, da)]
                } else {
                    [(a, da), (b, db)]
                };
                for (child, dist) in ordered {
                    if dist < best {
                        debug_assert!(top < stack.len(), "BVH deeper than the traversal stack");
                        if top < stack.len() {
                            stack[top] = (child, dist);
                            top += 1;
                        }
                    }
                }
            }
        }
        best
    }
}

/// Closest approach between the ray `p + t * dir` (t unbounded here;
/// the caller applies its floor) and the segment `[a, b]`: `(t at the
/// approach, distance)`.
fn ray_segment_approach(p: Vec3, dir: Vec3, a: Vec3, b: Vec3) -> (f64, f64) {
    let u = sub(b, a);
    let w0 = sub(p, a);
    let aa = math::dot(dir, dir);
    let bb = math::dot(dir, u);
    let cc = math::dot(u, u);
    let dd = math::dot(dir, w0);
    let ee = math::dot(u, w0);
    let den = aa * cc - bb * bb;
    let s = if den > 1e-30 {
        ((aa * ee - bb * dd) / den).clamp(0.0, 1.0)
    } else if cc > 0.0 {
        // Parallel: any s gives the same distance; project the origin.
        (ee / cc).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let q = [a[0] + s * u[0], a[1] + s * u[1], a[2] + s * u[2]];
    let t = if aa > 0.0 {
        math::dot(sub(q, p), dir) / aa
    } else {
        0.0
    };
    let at = [p[0] + t * dir[0], p[1] + t * dir[1], p[2] + t * dir[2]];
    (t, norm(sub(q, at)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BRepModel, Face, Instance, MeshSurface, Solid};
    use crate::math::Frame;

    fn identity_instance() -> Instance {
        Instance {
            solid: 0,
            local_to_world: Affine::IDENTITY,
            label: String::new(),
        }
    }

    /// Unit-cube triangle mesh `[0,1]^3`, optionally without the top
    /// (z = 1) so the shell has an open square boundary ring.
    fn cube_mesh(open_top: bool) -> MeshSurface {
        let verts: Vec<[f64; 3]> = (0..8)
            .map(|i| {
                [
                    (i & 1) as f64,
                    ((i >> 1) & 1) as f64,
                    ((i >> 2) & 1) as f64,
                ]
            })
            .collect();
        // Each face as two triangles (winding irrelevant: parity is
        // orientation-blind).
        let mut quads = vec![
            [0u32, 1, 3, 2], // z = 0
            [0, 1, 5, 4],    // y = 0
            [2, 3, 7, 6],    // y = 1
            [0, 2, 6, 4],    // x = 0
            [1, 3, 7, 5],    // x = 1
        ];
        if !open_top {
            quads.push([4, 5, 7, 6]); // z = 1
        }
        let tris = quads
            .into_iter()
            .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect();
        MeshSurface { verts, tris }
    }

    fn mesh_face(m: MeshSurface, color: Option<[f32; 3]>) -> Face {
        Face {
            surface: Surface::Mesh(m),
            trims: vec![],
            color,
        }
    }

    fn classify_grid(view: &PayloadView, expect: impl Fn([f64; 3]) -> Option<bool>) {
        let n = 23;
        let mut tested = 0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let p = [
                        -0.3 + 1.6 * i as f64 / (n - 1) as f64,
                        -0.3 + 1.6 * j as f64 / (n - 1) as f64,
                        -0.3 + 1.6 * k as f64 / (n - 1) as f64,
                    ];
                    let Some(want) = expect(p) else { continue };
                    tested += 1;
                    assert_eq!(view.is_inside(p), want, "at {p:?}");
                }
            }
        }
        assert!(tested > 8000, "only {tested} points tested");
    }

    fn cube_truth(p: [f64; 3]) -> Option<bool> {
        let d = (p[0] - 0.5)
            .abs()
            .max((p[1] - 0.5).abs())
            .max((p[2] - 0.5).abs())
            - 0.5;
        if d.abs() < 1e-9 { None } else { Some(d < 0.0) }
    }

    #[test]
    fn closed_mesh_cube_classifies() {
        let model = BRepModel {
            solids: vec![Solid {
                faces: vec![mesh_face(cube_mesh(false), None)],
            }],
            instances: vec![identity_instance()],
        };
        let payload = build_payload(&model).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        classify_grid(&view, cube_truth);
    }

    #[test]
    fn hybrid_mesh_with_exact_cap_classifies() {
        // The cushion topology in miniature: an open-top mesh box glued
        // along its boundary ring to an exact trimmed plane cap.
        let cap = Face {
            surface: Surface::Plane {
                frame: Frame {
                    origin: [0.0, 0.0, 1.0],
                    ..Frame::IDENTITY
                },
            },
            trims: vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            color: None,
        };
        let model = BRepModel {
            solids: vec![Solid {
                faces: vec![mesh_face(cube_mesh(true), None), cap],
            }],
            instances: vec![identity_instance()],
        };
        let payload = build_payload(&model).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        classify_grid(&view, cube_truth);
        // Points straddling the seam ring specifically.
        for (p, want) in [
            ([0.5, 0.001, 0.999], true),
            ([0.5, -0.001, 0.999], false),
            ([0.5, 0.001, 1.001], false),
            ([0.999, 0.999, 0.999], true),
            ([1.001, 0.999, 0.999], false),
        ] {
            assert_eq!(view.is_inside(p), want, "at {p:?}");
        }
    }

    #[test]
    fn mesh_face_color_and_distance() {
        let model = BRepModel {
            solids: vec![Solid {
                faces: vec![mesh_face(cube_mesh(false), Some([0.2, 0.4, 0.8]))],
            }],
            instances: vec![identity_instance()],
        };
        let payload = build_payload(&model).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        let c = view.nearest_color([0.5, 0.5, 1.2]).unwrap();
        assert!((c[0] - 0.2).abs() < 0.01 && (c[1] - 0.4).abs() < 0.01 && (c[2] - 0.8).abs() < 0.01);
    }

    #[test]
    fn mesh_boundary_and_bands() {
        let closed = cube_mesh(false);
        assert!(mesh_boundary(&closed).is_empty());
        let open = cube_mesh(true);
        let boundary = mesh_boundary(&open);
        assert_eq!(boundary.len(), 4, "open square rim");
        // Every rim vertex is a corner here, so every segment carries
        // the corner's second difference (erring wide is safe — it only
        // forces re-casts).
        for b in mesh_seam_bands(&open, &boundary) {
            assert!((b - core::f64::consts::SQRT_2 * 0.5).abs() < 1e-12, "{b}");
        }

        // A 2x1 flat strip: boundary chords with at least one straight
        // (collinear-neighbor) endpoint drop to zero band — the corner
        // kink must not swallow the chords beside it.
        let strip = MeshSurface {
            verts: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            tris: vec![[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]],
        };
        let boundary = mesh_boundary(&strip);
        assert_eq!(boundary.len(), 6);
        let bands = mesh_seam_bands(&strip, &boundary);
        for (s, b) in boundary.iter().zip(&bands) {
            let mid = |v: u32| v == 1 || v == 4; // straight mid-edge vertices
            if s.iter().any(|&v| mid(v)) {
                assert_eq!(*b, 0.0, "segment {s:?}");
            } else {
                assert!(*b > 0.4, "segment {s:?}: {b}");
            }
        }
    }
}

/// Trim loops read from payload bytes.
#[derive(Clone, Copy)]
struct RawLoops<'a> {
    bytes: &'a [u8],
    /// Trim blob base (the loop offset table lives at base + 40).
    base: usize,
    count: usize,
}

#[derive(Clone, Copy)]
struct RawLoop<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl TrimLoop for RawLoop<'_> {
    fn len(&self) -> usize {
        u32_at(self.bytes, self.off) as usize
    }
    fn point(&self, i: usize) -> [f64; 2] {
        let base = self.off + 8 + i * 16;
        [f64_at(self.bytes, base), f64_at(self.bytes, base + 8)]
    }
}

impl LoopSet for RawLoops<'_> {
    type Loop<'b>
        = RawLoop<'b>
    where
        Self: 'b;
    fn len(&self) -> usize {
        self.count
    }
    fn at(&self, i: usize) -> RawLoop<'_> {
        let rel = u32_at(self.bytes, self.base + 40 + i * 4) as usize;
        RawLoop {
            bytes: self.bytes,
            off: self.base + rel,
        }
    }
}
