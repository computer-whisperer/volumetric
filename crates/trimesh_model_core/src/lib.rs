//! The triangle-mesh model payload: the data contract between
//! `mesh_to_model_operator` (which builds it) and `trimesh_model_template`
//! (which traverses it at sample time).
//!
//! The operator does all the work up front — BVH construction and
//! serialization happen at conversion time in [`build_payload`] — so the
//! generated model is stateless: [`PayloadView`] only reads. Both sides are
//! this one crate, natively unit-tested, so the layout can't drift the way
//! hand-emitted codegen offsets did in the old STL importer.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (64 bytes):
//!    0  magic          u32   "TRM1" (0x314D_5254)
//!    4  triangle_count u32
//!    8  node_count     u32
//!   12  payload_len    u32   total byte length, header included
//!   16  bounds         6xf64 [min_x, max_x, min_y, max_y, min_z, max_z]
//! Nodes (node_count x 32 bytes at offset 64):
//!    0  aabb_min  3xf32
//!   12  aabb_max  3xf32
//!   24  a         u32    internal: left child index
//!                        leaf: 0x8000_0000 | first triangle index
//!   28  b         u32    internal: right child index; leaf: triangle count
//! Triangles (triangle_count x 36 bytes after the nodes): 9xf32 v0 v1 v2,
//! reordered so each leaf's triangles are contiguous.
//! ```
//!
//! # Inside/outside semantics
//!
//! [`PayloadView::is_inside`] casts a ray in a fixed near-+x direction and
//! classifies by crossing parity. For a watertight mesh this is the usual
//! point-in-solid test. Open meshes get parity's literal behavior — a
//! point is "inside" when an odd number of triangles lie between it and
//! infinity along the ray — which is well-defined but only meaningful for
//! closed surfaces; the conversion operator documents this rather than
//! rejecting open input. The ray is deliberately skewed off the x axis
//! (by irrational-ish factors) so the axis-aligned edges and face
//! diagonals that grid samplers constantly line up with never produce
//! parity ties; points exactly on a surface may still classify either way
//! (no exact arithmetic).

use volumetric_abi::trimesh::TriMesh;

pub const MAGIC: u32 = 0x314D_5254; // "TRM1"
const HEADER_LEN: usize = 64;
const NODE_LEN: usize = 32;
const TRI_LEN: usize = 36;
const LEAF_FLAG: u32 = 0x8000_0000;
const MAX_LEAF_TRIS: usize = 4;

/// Build the payload for a mesh: BVH + reordered triangles, serialized.
///
/// Fails on an empty mesh (a solid with no surface is meaningless) and on
/// invalid meshes.
pub fn build_payload(mesh: &TriMesh) -> Result<Vec<u8>, String> {
    mesh.validate()?;
    let triangle_count = mesh.triangle_count();
    if triangle_count == 0 {
        return Err("mesh has no triangles".to_string());
    }

    // f32 triangle copies (the sample-time precision) and their bounds.
    let tri = |t: usize| -> [[f32; 3]; 3] {
        let idx = mesh.triangle(t);
        std::array::from_fn(|v| {
            let p = mesh.position(idx[v] as usize);
            [p[0] as f32, p[1] as f32, p[2] as f32]
        })
    };

    #[derive(Clone, Copy)]
    struct TriBounds {
        min: [f32; 3],
        max: [f32; 3],
        centroid: [f32; 3],
    }
    let tri_bounds: Vec<TriBounds> = (0..triangle_count)
        .map(|t| {
            let v = tri(t);
            let mut min = v[0];
            let mut max = v[0];
            for p in &v[1..] {
                for axis in 0..3 {
                    min[axis] = min[axis].min(p[axis]);
                    max[axis] = max[axis].max(p[axis]);
                }
            }
            TriBounds {
                min,
                max,
                centroid: std::array::from_fn(|a| (min[a] + max[a]) * 0.5),
            }
        })
        .collect();

    struct Node {
        min: [f32; 3],
        max: [f32; 3],
        a: u32,
        b: u32,
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut order: Vec<u32> = (0..triangle_count as u32).collect();

    // Median-split recursion over `order[range]`; leaves reference the
    // final (reordered) triangle positions directly.
    fn build(
        nodes: &mut Vec<Node>,
        order: &mut [u32],
        base: usize,
        tri_bounds: &[TriBounds],
    ) -> u32 {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for &t in order.iter() {
            for axis in 0..3 {
                min[axis] = min[axis].min(tri_bounds[t as usize].min[axis]);
                max[axis] = max[axis].max(tri_bounds[t as usize].max[axis]);
            }
        }
        let node_index = nodes.len() as u32;
        nodes.push(Node {
            min,
            max,
            a: 0,
            b: 0,
        });

        if order.len() <= MAX_LEAF_TRIS {
            nodes[node_index as usize].a = LEAF_FLAG | base as u32;
            nodes[node_index as usize].b = order.len() as u32;
            return node_index;
        }

        let extent: [f32; 3] = std::array::from_fn(|a| max[a] - min[a]);
        let axis = (0..3)
            .max_by(|&x, &y| extent[x].partial_cmp(&extent[y]).unwrap())
            .unwrap();
        order.sort_unstable_by(|&p, &q| {
            tri_bounds[p as usize].centroid[axis]
                .partial_cmp(&tri_bounds[q as usize].centroid[axis])
                .unwrap()
        });
        let mid = order.len() / 2;
        let (left, right) = order.split_at_mut(mid);
        let a = build(nodes, left, base, tri_bounds);
        let b = build(nodes, right, base + mid, tri_bounds);
        nodes[node_index as usize].a = a;
        nodes[node_index as usize].b = b;
        node_index
    }
    build(&mut nodes, &mut order, 0, &tri_bounds);

    let bounds = mesh.bounds().expect("non-empty mesh has bounds");

    let payload_len = HEADER_LEN + nodes.len() * NODE_LEN + triangle_count * TRI_LEN;
    let mut out = Vec::with_capacity(payload_len);
    out.extend(MAGIC.to_le_bytes());
    out.extend((triangle_count as u32).to_le_bytes());
    out.extend((nodes.len() as u32).to_le_bytes());
    out.extend((payload_len as u32).to_le_bytes());
    for v in bounds {
        out.extend(v.to_le_bytes());
    }
    for node in &nodes {
        for v in node.min {
            out.extend(v.to_le_bytes());
        }
        for v in node.max {
            out.extend(v.to_le_bytes());
        }
        out.extend(node.a.to_le_bytes());
        out.extend(node.b.to_le_bytes());
    }
    for &t in &order {
        for v in tri(t as usize) {
            for c in v {
                out.extend(c.to_le_bytes());
            }
        }
    }
    debug_assert_eq!(out.len(), payload_len);
    Ok(out)
}

/// A read-only view over a serialized payload (sample-time side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    triangle_count: usize,
    node_count: usize,
}

impl<'a> PayloadView<'a> {
    /// Validate the header and structural sizes.
    pub fn new(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_LEN {
            return Err("payload shorter than header");
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err("bad payload magic");
        }
        let triangle_count = u32_at(4) as usize;
        let node_count = u32_at(8) as usize;
        let payload_len = u32_at(12) as usize;
        let expected = HEADER_LEN + node_count * NODE_LEN + triangle_count * TRI_LEN;
        if payload_len != expected || bytes.len() < expected {
            return Err("payload length mismatch");
        }
        Ok(Self {
            bytes,
            triangle_count,
            node_count,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.triangle_count
    }

    /// `[min_x, max_x, min_y, max_y, min_z, max_z]`.
    pub fn bounds(&self) -> [f64; 6] {
        std::array::from_fn(|i| {
            f64::from_le_bytes(self.bytes[16 + i * 8..24 + i * 8].try_into().unwrap())
        })
    }

    fn f32_at(&self, off: usize) -> f32 {
        f32::from_le_bytes(self.bytes[off..off + 4].try_into().unwrap())
    }

    fn u32_at(&self, off: usize) -> u32 {
        u32::from_le_bytes(self.bytes[off..off + 4].try_into().unwrap())
    }

    fn triangle(&self, t: usize) -> [[f32; 3]; 3] {
        let base = HEADER_LEN + self.node_count * NODE_LEN + t * TRI_LEN;
        std::array::from_fn(|v| std::array::from_fn(|c| self.f32_at(base + (v * 3 + c) * 4)))
    }

    /// Height of the mesh surface over a lateral point: the extreme
    /// coordinate along `axis` (max when `top`, min otherwise) where the
    /// axis-aligned line through `lat` crosses a triangle. `lat` holds the
    /// other two coordinates in ascending axis order (axis 2 → (x, y),
    /// axis 1 → (x, z), axis 0 → (y, z)). `None` when the line misses every
    /// triangle. Triangles parallel to the line (vertical walls) don't
    /// define a height and are skipped.
    pub fn height_at(&self, lat: [f64; 2], axis: usize, top: bool) -> Option<f64> {
        let (u, v) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            2 => (0, 1),
            _ => panic!("axis out of range: {axis}"),
        };
        let mut origin = [0.0f64; 3];
        origin[u] = lat[0];
        origin[v] = lat[1];
        let mut dir = [0.0f64; 3];
        dir[axis] = 1.0;

        if self.node_count == 0 {
            return None;
        }
        let mut best: Option<f64> = None;
        let mut stack = [0u32; 64];
        let mut top_of_stack = 0usize;
        stack[top_of_stack] = 0;
        top_of_stack += 1;

        while top_of_stack > 0 {
            top_of_stack -= 1;
            let node = stack[top_of_stack] as usize;
            let base = HEADER_LEN + node * NODE_LEN;
            // The line is axis-aligned and unbounded, so the node test is a
            // point-in-rect over the two lateral axes.
            let hit = [u, v].iter().zip(lat).all(|(&a, l)| {
                l >= self.f32_at(base + a * 4) as f64 && l <= self.f32_at(base + 12 + a * 4) as f64
            });
            if !hit {
                continue;
            }
            let a = self.u32_at(base + 24);
            let b = self.u32_at(base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for t in first..first + b as usize {
                    if let Some(h) = line_hit(self.triangle(t), origin, dir) {
                        best = Some(match best {
                            Some(prev) if top => prev.max(h),
                            Some(prev) => prev.min(h),
                            None => h,
                        });
                    }
                }
            } else {
                stack[top_of_stack] = a;
                top_of_stack += 1;
                stack[top_of_stack] = b;
                top_of_stack += 1;
            }
        }
        best
    }

    /// Point-in-mesh by ray parity along [`RAY_DIR`] (see the module docs
    /// for open-mesh and on-surface semantics).
    pub fn is_inside(&self, p: [f64; 3]) -> bool {
        if self.node_count == 0 {
            return false;
        }

        let mut crossings = 0usize;
        let mut stack = [0u32; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;

        while top > 0 {
            top -= 1;
            let node = stack[top] as usize;
            let base = HEADER_LEN + node * NODE_LEN;
            // Slab test; all RAY_DIR components are positive, so no swaps.
            let mut t_min = 0.0f64;
            let mut t_max = f64::INFINITY;
            for axis in 0..3 {
                let inv = 1.0 / RAY_DIR[axis];
                let lo = (self.f32_at(base + axis * 4) as f64 - p[axis]) * inv;
                let hi = (self.f32_at(base + 12 + axis * 4) as f64 - p[axis]) * inv;
                t_min = t_min.max(lo);
                t_max = t_max.min(hi);
            }
            if t_min > t_max {
                continue;
            }
            let a = self.u32_at(base + 24);
            let b = self.u32_at(base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for t in first..first + b as usize {
                    if ray_crosses(self.triangle(t), p) {
                        crossings += 1;
                    }
                }
            } else {
                // Depth is bounded by ~log2(n) + a slack for uneven splits;
                // 64 covers any mesh whose index fits in u32.
                stack[top] = a;
                top += 1;
                stack[top] = b;
                top += 1;
            }
        }
        crossings % 2 == 1
    }
}

/// The parity ray's direction: near +x, skewed so axis-aligned edges and
/// face diagonals never line up with it.
const RAY_DIR: [f64; 3] = [1.0, 1.618_033_988_7e-4, 2.718_281_828_4e-4];

/// Möller–Trumbore for the [`RAY_DIR`] ray from `p`: does the ray cross
/// this triangle at t > 0?
fn ray_crosses(v: [[f32; 3]; 3], p: [f64; 3]) -> bool {
    matches!(line_hit(v, p, RAY_DIR), Some(t) if t > 0.0)
}

/// Möller–Trumbore for the unbounded line `origin + t * dir`: the `t` where
/// the line crosses this triangle, any sign. `None` when it misses or when
/// the triangle is parallel to the line.
fn line_hit(v: [[f32; 3]; 3], origin: [f64; 3], dir: [f64; 3]) -> Option<f64> {
    // All in f64 for headroom; vertex inputs are exact f32 values.
    let (v0, v1, v2) = (v[0], v[1], v[2]);
    let e1 = [
        (v1[0] - v0[0]) as f64,
        (v1[1] - v0[1]) as f64,
        (v1[2] - v0[2]) as f64,
    ];
    let e2 = [
        (v2[0] - v0[0]) as f64,
        (v2[1] - v0[1]) as f64,
        (v2[2] - v0[2]) as f64,
    ];
    let h = [
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
    ];
    let det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if det.abs() < 1e-12 {
        return None; // line parallel to the triangle plane
    }
    let inv_det = 1.0 / det;
    let s = [
        origin[0] - v0[0] as f64,
        origin[1] - v0[1] as f64,
        origin[2] - v0[2] as f64,
    ];
    let u = (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = [
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0],
    ];
    let w = (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]) * inv_det;
    if w < 0.0 || u + w > 1.0 {
        return None;
    }
    Some((e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv_det)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(min: f64, max: f64) -> TriMesh {
        let corners: Vec<[f64; 3]> = (0..8)
            .map(|i| {
                [
                    if i & 1 == 0 { min } else { max },
                    if i & 2 == 0 { min } else { max },
                    if i & 4 == 0 { min } else { max },
                ]
            })
            .collect();
        // Faces as (a, b, c, d) quads with outward CCW winding.
        let quads = [
            [0, 4, 6, 2], // -x
            [1, 3, 7, 5], // +x
            [0, 1, 5, 4], // -y
            [2, 6, 7, 3], // +y
            [0, 2, 3, 1], // -z
            [4, 5, 7, 6], // +z
        ];
        let mut indices = Vec::new();
        for q in quads {
            indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]]);
        }
        TriMesh {
            positions: corners.into_iter().flatten().collect(),
            indices,
            vertex_fields: vec![],
            face_fields: vec![],
        }
    }

    /// Brute-force parity over all triangles (no BVH) — the oracle.
    fn brute_force_inside(mesh: &TriMesh, p: [f64; 3]) -> bool {
        let mut crossings = 0;
        for t in 0..mesh.triangle_count() {
            let idx = mesh.triangle(t);
            let v: [[f32; 3]; 3] = std::array::from_fn(|i| {
                let q = mesh.position(idx[i] as usize);
                [q[0] as f32, q[1] as f32, q[2] as f32]
            });
            if ray_crosses(v, p) {
                crossings += 1;
            }
        }
        crossings % 2 == 1
    }

    #[test]
    fn cube_payload_classifies_correctly() {
        let payload = build_payload(&cube(-1.0, 1.0)).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.triangle_count(), 12);
        assert_eq!(view.bounds(), [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);

        assert!(view.is_inside([0.0, 0.0, 0.0]));
        assert!(view.is_inside([0.9, -0.9, 0.9]));
        assert!(!view.is_inside([1.1, 0.0, 0.0]));
        assert!(!view.is_inside([0.0, -1.2, 0.0]));
        assert!(!view.is_inside([5.0, 5.0, 5.0]));
    }

    /// A unit UV sphere with enough triangles to force a real tree.
    fn uv_sphere(rings: usize, segments: usize) -> TriMesh {
        let mut positions: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 0.0, -1.0];
        for r in 1..rings {
            let phi = std::f64::consts::PI * r as f64 / rings as f64;
            for s in 0..segments {
                let theta = std::f64::consts::TAU * s as f64 / segments as f64;
                positions.extend([phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()]);
            }
        }
        let ring = |r: usize, s: usize| (2 + (r - 1) * segments + (s % segments)) as u32;
        let mut indices: Vec<u32> = Vec::new();
        for s in 0..segments {
            indices.extend([0, ring(1, s), ring(1, s + 1)]);
            indices.extend([1, ring(rings - 1, s + 1), ring(rings - 1, s)]);
        }
        for r in 1..rings - 1 {
            for s in 0..segments {
                let (a, b, c, d) = (
                    ring(r, s),
                    ring(r, s + 1),
                    ring(r + 1, s + 1),
                    ring(r + 1, s),
                );
                indices.extend([a, b, c, a, c, d]);
            }
        }
        TriMesh {
            positions,
            indices,
            vertex_fields: vec![],
            face_fields: vec![],
        }
    }

    #[test]
    fn bvh_matches_brute_force_on_a_sphere() {
        let mesh = uv_sphere(16, 24);
        let payload = build_payload(&mesh).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        // Deterministic pseudo-random probes, compared against the
        // no-BVH oracle; also sanity-check against the true sphere away
        // from the tessellation skin.
        let mut state = 0x12345678u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 3.0 - 1.5
        };
        for _ in 0..1000 {
            let p = [rand(), rand(), rand()];
            assert_eq!(
                view.is_inside(p),
                brute_force_inside(&mesh, p),
                "BVH disagrees with brute force at {p:?}"
            );
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if r < 0.95 {
                assert!(view.is_inside(p), "sphere interior misclassified at {p:?}");
            }
            if r > 1.05 {
                assert!(!view.is_inside(p), "sphere exterior misclassified at {p:?}");
            }
        }
    }

    #[test]
    fn cube_height_query_hits_the_faces() {
        let payload = build_payload(&cube(-1.0, 1.0)).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        // Top/bottom along each axis; the lateral point (0.3, -0.2) is well
        // inside every face.
        for axis in 0..3 {
            assert_eq!(view.height_at([0.3, -0.2], axis, true), Some(1.0));
            assert_eq!(view.height_at([0.3, -0.2], axis, false), Some(-1.0));
        }
        // Outside the footprint the line misses.
        assert_eq!(view.height_at([1.5, 0.0], 2, true), None);
        assert_eq!(view.height_at([0.0, -1.5], 0, true), None);
    }

    #[test]
    fn sphere_height_query_matches_the_analytic_surface() {
        let mesh = uv_sphere(16, 24);
        let payload = build_payload(&mesh).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        // Inscribed tessellation: heights sit slightly under sqrt(1 - r^2).
        for &(x, y) in &[(0.0, 0.0), (0.3, 0.1), (-0.5, 0.4), (0.2, -0.6)] {
            let analytic = (1.0f64 - x * x - y * y).sqrt();
            let top = view.height_at([x, y], 2, true).expect("line hits sphere");
            let bottom = view.height_at([x, y], 2, false).expect("line hits sphere");
            assert!(
                (top - analytic).abs() < 0.05,
                "top at ({x}, {y}): {top} vs {analytic}"
            );
            assert!(
                (bottom + analytic).abs() < 0.05,
                "bottom at ({x}, {y}): {bottom} vs {analytic}"
            );
        }
        assert_eq!(view.height_at([1.2, 0.0], 2, true), None);
    }

    #[test]
    fn height_query_matches_brute_force() {
        let mesh = uv_sphere(16, 24);
        let payload = build_payload(&mesh).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        // No-BVH oracle: fold line_hit over every triangle.
        let brute = |lat: [f64; 2], axis: usize, top: bool| -> Option<f64> {
            let (u, v) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let mut origin = [0.0f64; 3];
            origin[u] = lat[0];
            origin[v] = lat[1];
            let mut dir = [0.0f64; 3];
            dir[axis] = 1.0;
            let mut best: Option<f64> = None;
            for t in 0..mesh.triangle_count() {
                let idx = mesh.triangle(t);
                let tri: [[f32; 3]; 3] = std::array::from_fn(|i| {
                    let q = mesh.position(idx[i] as usize);
                    [q[0] as f32, q[1] as f32, q[2] as f32]
                });
                if let Some(h) = line_hit(tri, origin, dir) {
                    best = Some(match best {
                        Some(prev) if top => prev.max(h),
                        Some(prev) => prev.min(h),
                        None => h,
                    });
                }
            }
            best
        };

        let mut state = 0xdeadbeefu64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 3.0 - 1.5
        };
        for _ in 0..500 {
            let lat = [rand(), rand()];
            for axis in 0..3 {
                for top in [true, false] {
                    assert_eq!(
                        view.height_at(lat, axis, top),
                        brute(lat, axis, top),
                        "BVH height disagrees with brute force at {lat:?} axis {axis} top {top}"
                    );
                }
            }
        }
    }

    #[test]
    fn open_mesh_has_documented_parity_semantics() {
        // A single triangle in the x = 1 plane: points with the triangle
        // between them and +infinity along x count one crossing -> "inside".
        let mesh = TriMesh {
            positions: vec![
                1.0, -1.0, -1.0, //
                1.0, 1.0, -1.0, //
                1.0, 0.0, 1.0,
            ],
            indices: vec![0, 1, 2],
            vertex_fields: vec![],
            face_fields: vec![],
        };
        let payload = build_payload(&mesh).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert!(view.is_inside([0.0, 0.0, 0.0]), "behind the triangle");
        assert!(!view.is_inside([2.0, 0.0, 0.0]), "past the triangle");
        assert!(!view.is_inside([0.0, 5.0, 0.0]), "outside the silhouette");
    }

    #[test]
    fn empty_and_garbage_payloads_are_rejected() {
        let empty = TriMesh {
            positions: vec![],
            indices: vec![],
            vertex_fields: vec![],
            face_fields: vec![],
        };
        assert!(build_payload(&empty).is_err());
        assert!(PayloadView::new(&[0u8; 16]).is_err());
        let mut bad = build_payload(&cube(0.0, 1.0)).unwrap();
        bad[0] ^= 0xFF; // corrupt magic
        assert!(PayloadView::new(&bad).is_err());
    }
}
