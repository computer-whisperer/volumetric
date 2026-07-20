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
    }
    Ok(())
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
