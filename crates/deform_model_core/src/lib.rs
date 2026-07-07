//! The FEA-deformation model payload: the data contract between
//! `fea_deform_operator` (which builds it) and `deform_model_template`
//! (which evaluates it at sample time).
//!
//! The payload encodes a hex8 FEA mesh twice — undeformed node positions
//! and deformed ones (`position + scale * displacement`, prebaked here so
//! the sample side never sees the scale) — plus a BVH over the deformed
//! elements' AABBs. [`PayloadView::pull_back`] maps a point in deformed
//! space to its material (undeformed) point: find the deformed element
//! containing the query, Newton-invert its trilinear map to local
//! coordinates, and evaluate the undeformed element at those coordinates.
//! The wrapped model is then sampled at the material point (occupancy is
//! invariant under the pullback, so nothing needs rescaling).
//!
//! Queries outside every deformed element get a skin fallback: the nearest
//! element within `skin` extends its (extrapolated) trilinear inverse, so
//! geometry that pokes slightly past the mesh — the grid mesher keeps only
//! cells whose *centers* are occupied, so the true surface can sit up to
//! about half a cell outside — deforms continuously with its neighborhood
//! instead of being chopped off. Beyond the skin the pullback is `None`
//! (outside).
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (72 bytes):
//!    0  magic           u32   "DFM1" (0x314D_4644)
//!    4  element_count   u32
//!    8  bvh_node_count  u32
//!   12  payload_len     u32   total byte length, header included
//!   16  bounds          6xf64 [min_x, max_x, ...] deformed + skin pad
//!   64  skin            f32
//!   68  mesh_node_count u32
//! BVH nodes (bvh_node_count x 32 bytes at offset 72):
//!    0  aabb_min  3xf32  deformed element bounds
//!   12  aabb_max  3xf32
//!   24  a         u32    internal: left child index
//!                        leaf: 0x8000_0000 | first element index
//!   28  b         u32    internal: right child index; leaf: element count
//! Undeformed node positions (mesh_node_count x 12 bytes): 3xf32 each
//! Deformed node positions   (mesh_node_count x 12 bytes): 3xf32 each
//! Connectivity (element_count x 32 bytes): 8xu32 VTK-ordered node
//! indices, elements reordered so each leaf's range is contiguous.
//! ```

use volumetric_abi::fea::{FeaElementKind, FeaMesh};

pub const MAGIC: u32 = 0x314D_4644; // "DFM1"
const HEADER_LEN: usize = 72;
const NODE_LEN: usize = 32;
const LEAF_FLAG: u32 = 0x8000_0000;
const MAX_LEAF_ELEMENTS: usize = 4;

/// Build the payload for a solved mesh: the displacement field is applied
/// at `scale` (deformed = position + scale * displacement) and baked in.
///
/// `skin` is the fallback distance for queries outside every deformed
/// element; `None` picks half the mean deformed-element diagonal (about
/// the overhang the cell-center grid mesher can leave).
pub fn build_payload(
    mesh: &FeaMesh,
    displacement_field: &str,
    scale: f64,
    skin: Option<f64>,
) -> Result<Vec<u8>, String> {
    mesh.validate()?;
    let FeaElementKind::Hex8 = mesh.element_kind;
    let element_count = mesh.element_count();
    if element_count == 0 {
        return Err("mesh has no elements".to_string());
    }

    let field = mesh
        .node_fields
        .iter()
        .find(|f| f.name == displacement_field)
        .ok_or_else(|| {
            let available: Vec<&str> = mesh.node_fields.iter().map(|f| f.name.as_str()).collect();
            format!("mesh has no node field {displacement_field:?} (available: {available:?})")
        })?;
    if field.components != 3 {
        return Err(format!(
            "node field {displacement_field:?} has {} components, need 3",
            field.components
        ));
    }

    let node_count = mesh.node_count();
    let undeformed: Vec<f32> = mesh.node_positions.iter().map(|&v| v as f32).collect();
    let deformed: Vec<f32> = (0..node_count * 3)
        .map(|i| (mesh.node_positions[i] + scale * field.data[i]) as f32)
        .collect();

    #[derive(Clone, Copy)]
    struct ElementBounds {
        min: [f32; 3],
        max: [f32; 3],
        centroid: [f32; 3],
    }
    let element_bounds: Vec<ElementBounds> = (0..element_count)
        .map(|e| {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for &n in mesh.element(e) {
                for axis in 0..3 {
                    let v = deformed[n as usize * 3 + axis];
                    min[axis] = min[axis].min(v);
                    max[axis] = max[axis].max(v);
                }
            }
            ElementBounds {
                min,
                max,
                centroid: std::array::from_fn(|a| (min[a] + max[a]) * 0.5),
            }
        })
        .collect();

    let skin = match skin {
        Some(s) => s,
        None => {
            let total: f64 = element_bounds
                .iter()
                .map(|b| {
                    (0..3)
                        .map(|a| f64::from(b.max[a] - b.min[a]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum();
            0.5 * total / element_count as f64
        }
    };
    if !(skin.is_finite() && skin >= 0.0) {
        return Err(format!("skin must be finite and non-negative, got {skin}"));
    }

    struct Node {
        min: [f32; 3],
        max: [f32; 3],
        a: u32,
        b: u32,
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut order: Vec<u32> = (0..element_count as u32).collect();

    // Median-split recursion over `order[range]`, same scheme as the
    // trimesh BVH; leaves reference final (reordered) element positions.
    fn build(
        nodes: &mut Vec<Node>,
        order: &mut [u32],
        base: usize,
        bounds: &[ElementBounds],
    ) -> u32 {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for &e in order.iter() {
            for axis in 0..3 {
                min[axis] = min[axis].min(bounds[e as usize].min[axis]);
                max[axis] = max[axis].max(bounds[e as usize].max[axis]);
            }
        }
        let node_index = nodes.len() as u32;
        nodes.push(Node {
            min,
            max,
            a: 0,
            b: 0,
        });

        if order.len() <= MAX_LEAF_ELEMENTS {
            nodes[node_index as usize].a = LEAF_FLAG | base as u32;
            nodes[node_index as usize].b = order.len() as u32;
            return node_index;
        }

        let extent: [f32; 3] = std::array::from_fn(|a| max[a] - min[a]);
        let axis = (0..3)
            .max_by(|&x, &y| extent[x].partial_cmp(&extent[y]).unwrap())
            .unwrap();
        order.sort_unstable_by(|&p, &q| {
            bounds[p as usize].centroid[axis]
                .partial_cmp(&bounds[q as usize].centroid[axis])
                .unwrap()
        });
        let mid = order.len() / 2;
        let (left, right) = order.split_at_mut(mid);
        let a = build(nodes, left, base, bounds);
        let b = build(nodes, right, base + mid, bounds);
        nodes[node_index as usize].a = a;
        nodes[node_index as usize].b = b;
        node_index
    }
    build(&mut nodes, &mut order, 0, &element_bounds);

    // Sample domain: deformed extent padded by the skin, interleaved
    // min/max per axis (the model ABI's get_bounds order).
    let bounds: [f64; 6] = std::array::from_fn(|i| {
        let axis = i / 2;
        if i % 2 == 0 {
            f64::from(nodes[0].min[axis]) - skin
        } else {
            f64::from(nodes[0].max[axis]) + skin
        }
    });

    let payload_len = HEADER_LEN + nodes.len() * NODE_LEN + node_count * 24 + element_count * 32;
    let mut out = Vec::with_capacity(payload_len);
    out.extend(MAGIC.to_le_bytes());
    out.extend((element_count as u32).to_le_bytes());
    out.extend((nodes.len() as u32).to_le_bytes());
    out.extend((payload_len as u32).to_le_bytes());
    for v in bounds {
        out.extend(v.to_le_bytes());
    }
    out.extend((skin as f32).to_le_bytes());
    out.extend((node_count as u32).to_le_bytes());
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
    for &v in &undeformed {
        out.extend(v.to_le_bytes());
    }
    for &v in &deformed {
        out.extend(v.to_le_bytes());
    }
    for &e in &order {
        for &n in mesh.element(e as usize) {
            out.extend(n.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), payload_len);
    Ok(out)
}

/// A read-only view over a serialized payload (sample-time side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    element_count: usize,
    bvh_node_count: usize,
    mesh_node_count: usize,
    skin: f64,
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
        let element_count = u32_at(4) as usize;
        let bvh_node_count = u32_at(8) as usize;
        let payload_len = u32_at(12) as usize;
        let mesh_node_count = u32_at(68) as usize;
        let expected =
            HEADER_LEN + bvh_node_count * NODE_LEN + mesh_node_count * 24 + element_count * 32;
        if payload_len != expected || bytes.len() < expected {
            return Err("payload length mismatch");
        }
        let skin = f64::from(f32::from_le_bytes(bytes[64..68].try_into().unwrap()));
        if !(skin.is_finite() && skin >= 0.0) {
            return Err("bad skin distance");
        }
        Ok(Self {
            bytes,
            element_count,
            bvh_node_count,
            mesh_node_count,
            skin,
        })
    }

    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// `[min_x, max_x, min_y, max_y, min_z, max_z]`, skin pad included.
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

    /// The 8 corners of element `e`, from the undeformed or deformed
    /// position table.
    fn corners(&self, e: usize, deformed: bool) -> [[f64; 3]; 8] {
        let positions = HEADER_LEN
            + self.bvh_node_count * NODE_LEN
            + if deformed {
                self.mesh_node_count * 12
            } else {
                0
            };
        let conn = HEADER_LEN + self.bvh_node_count * NODE_LEN + self.mesh_node_count * 24 + e * 32;
        std::array::from_fn(|c| {
            let n = self.u32_at(conn + c * 4) as usize;
            std::array::from_fn(|axis| f64::from(self.f32_at(positions + (n * 3 + axis) * 4)))
        })
    }

    /// Map a point in deformed space back to its material (undeformed)
    /// point, or `None` when it lies outside every deformed element and
    /// the skin.
    pub fn pull_back(&self, p: [f64; 3]) -> Option<[f64; 3]> {
        if self.bvh_node_count == 0 {
            return None;
        }
        let skin = self.skin;
        let skin2 = skin * skin;
        let mut best: Option<(f64, [f64; 3])> = None;

        let mut stack = [0u32; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;

        while top > 0 {
            top -= 1;
            let node = stack[top] as usize;
            let base = HEADER_LEN + node * NODE_LEN;
            let hit = (0..3).all(|axis| {
                p[axis] >= f64::from(self.f32_at(base + axis * 4)) - skin
                    && p[axis] <= f64::from(self.f32_at(base + 12 + axis * 4)) + skin
            });
            if !hit {
                continue;
            }
            let a = self.u32_at(base + 24);
            let b = self.u32_at(base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for e in first..first + b as usize {
                    let corners = self.corners(e, true);
                    let Some(local) = invert_trilinear(&corners, p) else {
                        continue;
                    };
                    const INSIDE_EPS: f64 = 1e-6;
                    if local
                        .iter()
                        .all(|&c| (-INSIDE_EPS..=1.0 + INSIDE_EPS).contains(&c))
                    {
                        return Some(trilinear(&self.corners(e, false), local));
                    }
                    // Skin fallback: distance from the query to the
                    // element (via its clamped local point); the nearest
                    // element extends its map by extrapolation.
                    let clamped: [f64; 3] = std::array::from_fn(|i| local[i].clamp(0.0, 1.0));
                    let surface = trilinear(&corners, clamped);
                    let d2: f64 = (0..3).map(|i| (p[i] - surface[i]).powi(2)).sum();
                    if d2 <= skin2 && best.is_none_or(|(bd2, _)| d2 < bd2) {
                        best = Some((d2, trilinear(&self.corners(e, false), local)));
                    }
                }
            } else {
                stack[top] = a;
                top += 1;
                stack[top] = b;
                top += 1;
            }
        }
        best.map(|(_, x)| x)
    }
}

/// The hex8 trilinear basis (VTK node ordering) at local `(u, v, w)`.
fn weights(local: [f64; 3]) -> [f64; 8] {
    let [u, v, w] = local;
    [
        (1.0 - u) * (1.0 - v) * (1.0 - w),
        u * (1.0 - v) * (1.0 - w),
        u * v * (1.0 - w),
        (1.0 - u) * v * (1.0 - w),
        (1.0 - u) * (1.0 - v) * w,
        u * (1.0 - v) * w,
        u * v * w,
        (1.0 - u) * v * w,
    ]
}

/// Evaluate the trilinear map of a hex at local coordinates.
fn trilinear(corners: &[[f64; 3]; 8], local: [f64; 3]) -> [f64; 3] {
    let weight = weights(local);
    std::array::from_fn(|axis| (0..8).map(|i| weight[i] * corners[i][axis]).sum())
}

/// The Jacobian columns `[dT/du, dT/dv, dT/dw]` of the trilinear map.
fn jacobian(corners: &[[f64; 3]; 8], local: [f64; 3]) -> [[f64; 3]; 3] {
    let [u, v, w] = local;
    let du = [
        -(1.0 - v) * (1.0 - w),
        (1.0 - v) * (1.0 - w),
        v * (1.0 - w),
        -v * (1.0 - w),
        -(1.0 - v) * w,
        (1.0 - v) * w,
        v * w,
        -v * w,
    ];
    let dv = [
        -(1.0 - u) * (1.0 - w),
        -u * (1.0 - w),
        u * (1.0 - w),
        (1.0 - u) * (1.0 - w),
        -(1.0 - u) * w,
        -u * w,
        u * w,
        (1.0 - u) * w,
    ];
    let dw = [
        -(1.0 - u) * (1.0 - v),
        -u * (1.0 - v),
        -u * v,
        -(1.0 - u) * v,
        (1.0 - u) * (1.0 - v),
        u * (1.0 - v),
        u * v,
        (1.0 - u) * v,
    ];
    let column = |d: &[f64; 8]| -> [f64; 3] {
        std::array::from_fn(|axis| (0..8).map(|i| d[i] * corners[i][axis]).sum())
    };
    [column(&du), column(&dv), column(&dw)]
}

/// Newton-invert the trilinear map: the local coordinates where the hex
/// maps to `target`. Converges in 1-2 iterations for mildly warped
/// elements (exact in one step for affine ones). `None` on a degenerate
/// Jacobian or non-convergence; local coordinates outside `[0,1]^3` are
/// returned as-is (the caller decides how far outside is acceptable —
/// they extrapolate the element's map continuously).
fn invert_trilinear(corners: &[[f64; 3]; 8], target: [f64; 3]) -> Option<[f64; 3]> {
    let diag2: f64 = {
        let mut min = corners[0];
        let mut max = corners[0];
        for c in &corners[1..] {
            for axis in 0..3 {
                min[axis] = min[axis].min(c[axis]);
                max[axis] = max[axis].max(c[axis]);
            }
        }
        (0..3).map(|a| (max[a] - min[a]).powi(2)).sum()
    };
    if diag2 <= 0.0 {
        return None;
    }

    let mut local = [0.5f64; 3];
    for _ in 0..16 {
        let pos = trilinear(corners, local);
        let residual: [f64; 3] = std::array::from_fn(|i| target[i] - pos[i]);
        let r2: f64 = residual.iter().map(|r| r * r).sum();
        if r2 < diag2 * 1e-20 {
            return Some(local);
        }
        let j = jacobian(corners, local);
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[1][0] * (j[0][1] * j[2][2] - j[0][2] * j[2][1])
            + j[2][0] * (j[0][1] * j[1][2] - j[0][2] * j[1][1]);
        if det.abs() < 1e-12 * diag2.powf(1.5) {
            return None;
        }
        // Cramer's rule on J (columns j[0..3]) * delta = residual.
        let inv_det = 1.0 / det;
        let col_det = |a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]| -> f64 {
            a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
                + c[0] * (a[1] * b[2] - a[2] * b[1])
        };
        let delta = [
            col_det(&residual, &j[1], &j[2]) * inv_det,
            col_det(&j[0], &residual, &j[2]) * inv_det,
            col_det(&j[0], &j[1], &residual) * inv_det,
        ];
        for axis in 0..3 {
            // Keep the iterate near the element; targets this far outside
            // are rejected by the skin test anyway, and the unclamped
            // trilinear map can fold at large distances.
            local[axis] = (local[axis] + delta[axis]).clamp(-4.0, 5.0);
        }
    }
    let pos = trilinear(corners, local);
    let r2: f64 = (0..3).map(|i| (target[i] - pos[i]).powi(2)).sum();
    (r2 < diag2 * 1e-14).then_some(local)
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::fea::FeaField;

    /// An n x n x n uniform hex grid over `[0, extent]^3` with a
    /// displacement field defined per node.
    fn grid_mesh(n: usize, extent: f64, disp: impl Fn([f64; 3]) -> [f64; 3]) -> FeaMesh {
        let cell = extent / n as f64;
        let nodes_per_axis = n + 1;
        let mut node_positions = Vec::new();
        let mut displacement = Vec::new();
        for x in 0..nodes_per_axis {
            for y in 0..nodes_per_axis {
                for z in 0..nodes_per_axis {
                    let p = [x as f64 * cell, y as f64 * cell, z as f64 * cell];
                    node_positions.extend(p);
                    displacement.extend(disp(p));
                }
            }
        }
        let node = |x: usize, y: usize, z: usize| -> u32 {
            ((x * nodes_per_axis + y) * nodes_per_axis + z) as u32
        };
        let mut connectivity = Vec::new();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    connectivity.extend([
                        node(x, y, z),
                        node(x + 1, y, z),
                        node(x + 1, y + 1, z),
                        node(x, y + 1, z),
                        node(x, y, z + 1),
                        node(x + 1, y, z + 1),
                        node(x + 1, y + 1, z + 1),
                        node(x, y + 1, z + 1),
                    ]);
                }
            }
        }
        FeaMesh {
            element_kind: FeaElementKind::Hex8,
            node_positions,
            connectivity,
            node_fields: vec![FeaField {
                name: "displacement".to_string(),
                components: 3,
                data: displacement,
            }],
            element_fields: vec![],
        }
    }

    #[test]
    fn identity_displacement_pulls_back_to_the_query() {
        let mesh = grid_mesh(2, 1.0, |_| [0.0; 3]);
        let payload = build_payload(&mesh, "displacement", 1.0, Some(0.1)).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        for p in [[0.5, 0.5, 0.5], [0.1, 0.9, 0.3], [0.99, 0.01, 0.5]] {
            let x = view.pull_back(p).expect("inside the mesh");
            for axis in 0..3 {
                assert!((x[axis] - p[axis]).abs() < 1e-6, "{p:?} -> {x:?}");
            }
        }
        // Within the skin: extrapolates to (approximately) the query.
        let x = view.pull_back([1.05, 0.5, 0.5]).expect("in the skin");
        assert!((x[0] - 1.05).abs() < 1e-6, "{x:?}");
        // Beyond the skin: outside.
        assert!(view.pull_back([1.2, 0.5, 0.5]).is_none());
        assert!(view.pull_back([5.0, 5.0, 5.0]).is_none());
    }

    #[test]
    fn rigid_translation_pulls_back_by_the_offset() {
        let t = [0.3, -0.2, 0.7];
        let mesh = grid_mesh(2, 1.0, |_| t);
        let payload = build_payload(&mesh, "displacement", 1.0, Some(0.05)).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        let bounds = view.bounds();
        assert!((bounds[0] - (0.0 + t[0] - 0.05)).abs() < 1e-6);
        assert!((bounds[1] - (1.0 + t[0] + 0.05)).abs() < 1e-6);

        for p in [[0.5, 0.5, 0.5], [0.2, 0.8, 0.4]] {
            let q = [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
            let x = view.pull_back(q).expect("inside the deformed mesh");
            for axis in 0..3 {
                assert!((x[axis] - p[axis]).abs() < 1e-6, "{q:?} -> {x:?}");
            }
        }
        // The undeformed region is now void (the mesh moved away).
        assert!(view.pull_back([0.02, 0.02, 0.02]).is_none());
    }

    #[test]
    fn linear_shear_inverts_exactly() {
        // u_z = 0.5 x: linear in x, so the trilinear elements reproduce
        // the field exactly and the pullback must too.
        let mesh = grid_mesh(3, 1.0, |p| [0.0, 0.0, 0.5 * p[0]]);
        let payload = build_payload(&mesh, "displacement", 1.0, Some(0.02)).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        for p in [[0.5, 0.5, 0.5], [0.9, 0.1, 0.2], [0.15, 0.7, 0.85]] {
            let q = [p[0], p[1], p[2] + 0.5 * p[0]];
            let x = view.pull_back(q).expect("inside the sheared mesh");
            for axis in 0..3 {
                assert!((x[axis] - p[axis]).abs() < 1e-5, "{q:?} -> {x:?}");
            }
        }
    }

    #[test]
    fn scale_bakes_into_the_deformation() {
        let mesh = grid_mesh(2, 1.0, |_| [1.0, 0.0, 0.0]);
        let payload = build_payload(&mesh, "displacement", 0.25, Some(0.05)).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        let x = view.pull_back([0.75, 0.5, 0.5]).expect("inside");
        assert!((x[0] - 0.5).abs() < 1e-6, "{x:?}");

        // scale 0: the identity map over the mesh region.
        let payload = build_payload(&mesh, "displacement", 0.0, Some(0.05)).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        let x = view.pull_back([0.75, 0.5, 0.5]).expect("inside");
        assert!((x[0] - 0.75).abs() < 1e-6, "{x:?}");
    }

    #[test]
    fn bvh_agrees_with_brute_force_on_a_warped_grid() {
        // A smoothly warped 4^3 grid; compare pull_back against a linear
        // scan over every element.
        let disp = |p: [f64; 3]| {
            [
                0.1 * (p[1] * 2.0).sin(),
                0.08 * (p[0] * 3.0).cos(),
                0.12 * p[0] * p[1],
            ]
        };
        let mesh = grid_mesh(4, 1.0, disp);
        let payload = build_payload(&mesh, "displacement", 1.0, None).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        let brute = |p: [f64; 3]| -> Option<[f64; 3]> {
            let skin = view.skin;
            let mut best: Option<(f64, [f64; 3])> = None;
            for e in 0..view.element_count() {
                let corners = view.corners(e, true);
                let Some(local) = invert_trilinear(&corners, p) else {
                    continue;
                };
                if local.iter().all(|&c| (-1e-6..=1.0 + 1e-6).contains(&c)) {
                    return Some(trilinear(&view.corners(e, false), local));
                }
                let clamped: [f64; 3] = std::array::from_fn(|i| local[i].clamp(0.0, 1.0));
                let surface = trilinear(&corners, clamped);
                let d2: f64 = (0..3).map(|i| (p[i] - surface[i]).powi(2)).sum();
                if d2 <= skin * skin && best.is_none_or(|(bd2, _)| d2 < bd2) {
                    best = Some((d2, trilinear(&view.corners(e, false), local)));
                }
            }
            best.map(|(_, x)| x)
        };

        let mut state = 0x1234_5678_u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 1.6 - 0.3
        };
        let mut inside = 0usize;
        for _ in 0..500 {
            let p = [rand(), rand(), rand()];
            let via_bvh = view.pull_back(p);
            let via_scan = brute(p);
            match (via_bvh, via_scan) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    inside += 1;
                    // Both must land on the same material point: the mesh
                    // is conforming, so overlapping candidates give the
                    // same continuous map.
                    for axis in 0..3 {
                        assert!((a[axis] - b[axis]).abs() < 1e-5, "{p:?}: {a:?} vs {b:?}");
                    }
                }
                (a, b) => panic!("BVH {a:?} vs scan {b:?} disagree at {p:?}"),
            }
        }
        assert!(inside > 100, "probe set should land inside: {inside}");
    }

    #[test]
    fn missing_or_malformed_fields_are_rejected() {
        let mesh = grid_mesh(1, 1.0, |_| [0.0; 3]);
        let err = build_payload(&mesh, "nope", 1.0, None).unwrap_err();
        assert!(
            err.contains("nope") && err.contains("displacement"),
            "{err}"
        );

        let mut scalar = mesh.clone();
        scalar.node_fields[0].components = 1;
        scalar.node_fields[0].data.truncate(8);
        let err = build_payload(&scalar, "displacement", 1.0, None).unwrap_err();
        assert!(err.contains("3"), "{err}");

        let mut empty = mesh.clone();
        empty.connectivity.clear();
        assert!(build_payload(&empty, "displacement", 1.0, None).is_err());

        assert!(build_payload(&mesh, "displacement", 1.0, Some(f64::NAN)).is_err());
    }

    #[test]
    fn garbage_payloads_are_rejected() {
        assert!(PayloadView::new(&[0u8; 16]).is_err());
        let mesh = grid_mesh(1, 1.0, |_| [0.0; 3]);
        let mut bad = build_payload(&mesh, "displacement", 1.0, None).unwrap();
        bad[0] ^= 0xFF;
        assert!(PayloadView::new(&bad).is_err());
        let mut short = build_payload(&mesh, "displacement", 1.0, None).unwrap();
        short.truncate(short.len() - 8);
        assert!(PayloadView::new(&short).is_err());
    }
}
