//! The strut-lattice model payload: the data contract between
//! `strut_model_operator` (which builds it) and `strut_model_template`
//! (which evaluates it at sample time).
//!
//! A strut lattice realizes as the union of capsules — one per strut,
//! segment between the strut's endpoints swept by its radius, so struts
//! meeting at a node blend through the shared sphere. The operator does
//! all the work up front ([`build_payload`]: BVH construction over the
//! radius-inflated strut boxes, serialization); [`PayloadView`] only
//! reads. Both sides live in this one crate, natively unit-tested, so the
//! layout can't drift.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (64 bytes):
//!    0  magic        u32   "STM1" (0x314D_5453)
//!    4  strut_count  u32
//!    8  node_count   u32
//!   12  payload_len  u32   total byte length, header included
//!   16  bounds       6xf64 [min_x, max_x, min_y, max_y, min_z, max_z]
//! Nodes (node_count x 32 bytes at offset 64):
//!    0  aabb_min  3xf32   capsule boxes, radius included
//!   12  aabb_max  3xf32
//!   24  a         u32     internal: left child; leaf: 0x8000_0000 | first
//!   28  b         u32     internal: right child; leaf: capsule count
//! Capsules (strut_count x 28 bytes after the nodes): 3xf32 a, 3xf32 b,
//! f32 radius — reordered so each leaf's capsules are contiguous.
//! ```

pub const MAGIC: u32 = 0x314D_5453; // "STM1"
const HEADER_LEN: usize = 64;
const NODE_LEN: usize = 32;
const CAPSULE_LEN: usize = 28;
const LEAF_FLAG: u32 = 0x8000_0000;
const MAX_LEAF_CAPSULES: usize = 4;

/// One strut, realized as the segment `[a, b]` swept by `radius`.
#[derive(Clone, Copy, Debug)]
pub struct Capsule {
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub radius: f64,
}

/// Build the payload: BVH over the capsules' radius-inflated boxes, then
/// leaf-ordered capsule records. Fails on an empty list and on non-finite
/// or non-positive geometry.
pub fn build_payload(capsules: &[Capsule]) -> Result<Vec<u8>, String> {
    if capsules.is_empty() {
        return Err("no struts to realize".to_string());
    }
    for (i, c) in capsules.iter().enumerate() {
        if c.a.iter().chain(&c.b).any(|v| !v.is_finite()) {
            return Err(format!("strut {i} has a non-finite endpoint"));
        }
        if !(c.radius.is_finite() && c.radius > 0.0) {
            return Err(format!("strut {i} has invalid radius {}", c.radius));
        }
    }

    #[derive(Clone, Copy)]
    struct Box3 {
        min: [f32; 3],
        max: [f32; 3],
        centroid: [f32; 3],
    }
    let boxes: Vec<Box3> = capsules
        .iter()
        .map(|c| {
            let min: [f32; 3] = std::array::from_fn(|i| (c.a[i].min(c.b[i]) - c.radius) as f32);
            let max: [f32; 3] = std::array::from_fn(|i| (c.a[i].max(c.b[i]) + c.radius) as f32);
            Box3 {
                min,
                max,
                centroid: std::array::from_fn(|i| (min[i] + max[i]) * 0.5),
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
    let mut order: Vec<u32> = (0..capsules.len() as u32).collect();

    // Median-split recursion over `order[range]` (same scheme as the
    // trimesh BVH); leaves reference final reordered capsule positions.
    fn build(nodes: &mut Vec<Node>, order: &mut [u32], base: usize, boxes: &[Box3]) -> u32 {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for &c in order.iter() {
            for axis in 0..3 {
                min[axis] = min[axis].min(boxes[c as usize].min[axis]);
                max[axis] = max[axis].max(boxes[c as usize].max[axis]);
            }
        }
        let node_index = nodes.len() as u32;
        nodes.push(Node {
            min,
            max,
            a: 0,
            b: 0,
        });

        if order.len() <= MAX_LEAF_CAPSULES {
            nodes[node_index as usize].a = LEAF_FLAG | base as u32;
            nodes[node_index as usize].b = order.len() as u32;
            return node_index;
        }

        let extent: [f32; 3] = std::array::from_fn(|a| max[a] - min[a]);
        let axis = (0..3)
            .max_by(|&x, &y| extent[x].partial_cmp(&extent[y]).unwrap())
            .unwrap();
        order.sort_unstable_by(|&p, &q| {
            boxes[p as usize].centroid[axis]
                .partial_cmp(&boxes[q as usize].centroid[axis])
                .unwrap()
        });
        let mid = order.len() / 2;
        let (left, right) = order.split_at_mut(mid);
        let a = build(nodes, left, base, boxes);
        let b = build(nodes, right, base + mid, boxes);
        nodes[node_index as usize].a = a;
        nodes[node_index as usize].b = b;
        node_index
    }
    build(&mut nodes, &mut order, 0, &boxes);

    // f64 bounds from the exact capsule extents.
    let mut bounds = [f64::INFINITY, f64::NEG_INFINITY].repeat(3);
    for c in capsules {
        for axis in 0..3 {
            bounds[axis * 2] = bounds[axis * 2].min(c.a[axis].min(c.b[axis]) - c.radius);
            bounds[axis * 2 + 1] = bounds[axis * 2 + 1].max(c.a[axis].max(c.b[axis]) + c.radius);
        }
    }

    let payload_len = HEADER_LEN + nodes.len() * NODE_LEN + capsules.len() * CAPSULE_LEN;
    let mut out = Vec::with_capacity(payload_len);
    out.extend(MAGIC.to_le_bytes());
    out.extend((capsules.len() as u32).to_le_bytes());
    out.extend((nodes.len() as u32).to_le_bytes());
    out.extend((payload_len as u32).to_le_bytes());
    for v in &bounds {
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
    for &c in &order {
        let capsule = capsules[c as usize];
        for v in capsule.a {
            out.extend((v as f32).to_le_bytes());
        }
        for v in capsule.b {
            out.extend((v as f32).to_le_bytes());
        }
        out.extend((capsule.radius as f32).to_le_bytes());
    }
    debug_assert_eq!(out.len(), payload_len);
    Ok(out)
}

/// A read-only view over a serialized payload (sample-time side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
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
        let strut_count = u32_at(4) as usize;
        let node_count = u32_at(8) as usize;
        let payload_len = u32_at(12) as usize;
        let expected = HEADER_LEN + node_count * NODE_LEN + strut_count * CAPSULE_LEN;
        if payload_len != expected || bytes.len() < expected {
            return Err("payload length mismatch");
        }
        Ok(Self { bytes, node_count })
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

    /// Whether `p` lies inside any capsule (BVH point query).
    pub fn is_inside(&self, p: [f64; 3]) -> bool {
        if self.node_count == 0 {
            return false;
        }
        let capsule_base = HEADER_LEN + self.node_count * NODE_LEN;
        let mut stack = [0u32; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;

        while top > 0 {
            top -= 1;
            let node = stack[top] as usize;
            let base = HEADER_LEN + node * NODE_LEN;
            let hit = (0..3).all(|axis| {
                p[axis] >= self.f32_at(base + axis * 4) as f64
                    && p[axis] <= self.f32_at(base + 12 + axis * 4) as f64
            });
            if !hit {
                continue;
            }
            let a = self.u32_at(base + 24);
            let b = self.u32_at(base + 28);
            if a & LEAF_FLAG != 0 {
                let first = (a & !LEAF_FLAG) as usize;
                for c in first..first + b as usize {
                    let off = capsule_base + c * CAPSULE_LEN;
                    let ca: [f64; 3] = std::array::from_fn(|i| self.f32_at(off + i * 4) as f64);
                    let cb: [f64; 3] =
                        std::array::from_fn(|i| self.f32_at(off + 12 + i * 4) as f64);
                    let radius = self.f32_at(off + 24) as f64;
                    if segment_distance_squared(p, ca, cb) <= radius * radius {
                        return true;
                    }
                }
            } else {
                // Depth bounded by ~log2(n) plus slack; 64 covers any u32
                // index count.
                stack[top] = a;
                top += 1;
                stack[top] = b;
                top += 1;
            }
        }
        false
    }
}

/// Squared distance from `p` to the segment `[a, b]` (degenerate segments
/// read as their point).
fn segment_distance_squared(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let t = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    let s = if tt > 0.0 {
        ((ap[0] * t[0] + ap[1] * t[1] + ap[2] * t[2]) / tt).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let e = [ap[0] - s * t[0], ap[1] - s * t[1], ap[2] - s * t[2]];
    e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_capsule_classifies_analytically() {
        // Unit segment along x at radius 0.1.
        let payload = build_payload(&[Capsule {
            a: [0.0, 0.0, 0.0],
            b: [1.0, 0.0, 0.0],
            radius: 0.1,
        }])
        .unwrap();
        let view = PayloadView::new(&payload).unwrap();

        assert!(view.is_inside([0.5, 0.0, 0.0])); // on the axis
        assert!(view.is_inside([0.5, 0.09, 0.0])); // inside the sweep
        assert!(!view.is_inside([0.5, 0.11, 0.0])); // outside the sweep
        assert!(view.is_inside([-0.05, 0.0, 0.0])); // end cap
        assert!(!view.is_inside([-0.15, 0.0, 0.0])); // past the end cap
        assert!(view.is_inside([1.05, 0.05, 0.0])); // far cap, diagonal
        assert!(!view.is_inside([1.08, 0.08, 0.0])); // sqrt(0.08^2*2) > 0.1

        let bounds = view.bounds();
        assert_eq!(bounds, [-0.1, 1.1, -0.1, 0.1, -0.1, 0.1]);
    }

    #[test]
    fn degenerate_capsule_is_a_sphere() {
        let payload = build_payload(&[Capsule {
            a: [1.0, 2.0, 3.0],
            b: [1.0, 2.0, 3.0],
            radius: 0.5,
        }])
        .unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert!(view.is_inside([1.0, 2.0, 3.4]));
        assert!(!view.is_inside([1.0, 2.0, 3.6]));
    }

    /// A jittered strut field big enough to force a real tree.
    fn strut_field() -> Vec<Capsule> {
        let mut out = Vec::new();
        let mut state = 0x2468_ace0u64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        for _ in 0..300 {
            let a = [rand() * 4.0, rand() * 4.0, rand() * 4.0];
            let d = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
            out.push(Capsule {
                a,
                b: [a[0] + d[0], a[1] + d[1], a[2] + d[2]],
                radius: 0.02 + rand() * 0.08,
            });
        }
        out
    }

    #[test]
    fn bvh_matches_brute_force() {
        let capsules = strut_field();
        let payload = build_payload(&capsules).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        // Brute force over the same f32-rounded capsules the payload holds.
        let brute = |p: [f64; 3]| {
            capsules.iter().any(|c| {
                let a: [f64; 3] = std::array::from_fn(|i| c.a[i] as f32 as f64);
                let b: [f64; 3] = std::array::from_fn(|i| c.b[i] as f32 as f64);
                let r = c.radius as f32 as f64;
                segment_distance_squared(p, a, b) <= r * r
            })
        };

        let mut state = 0x1357_9bdfu64;
        let mut rand = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 * 5.0 - 0.5
        };
        let mut inside = 0usize;
        for _ in 0..2000 {
            let p = [rand(), rand(), rand()];
            assert_eq!(
                view.is_inside(p),
                brute(p),
                "BVH disagrees with brute force at {p:?}"
            );
            inside += view.is_inside(p) as usize;
        }
        assert!(inside > 20, "probe set never hit the struts: {inside}");
    }

    #[test]
    fn payload_round_trips_bounds() {
        let capsules = strut_field();
        let payload = build_payload(&capsules).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        let bounds = view.bounds();
        for c in &capsules {
            for axis in 0..3 {
                assert!(bounds[axis * 2] <= c.a[axis].min(c.b[axis]) - c.radius + 1e-12);
                assert!(bounds[axis * 2 + 1] >= c.a[axis].max(c.b[axis]) + c.radius - 1e-12);
            }
        }
    }

    #[test]
    fn empty_and_garbage_payloads_are_rejected() {
        assert!(build_payload(&[]).is_err());
        assert!(
            build_payload(&[Capsule {
                a: [0.0; 3],
                b: [1.0, 0.0, 0.0],
                radius: 0.0,
            }])
            .is_err()
        );
        assert!(
            build_payload(&[Capsule {
                a: [f64::NAN, 0.0, 0.0],
                b: [1.0, 0.0, 0.0],
                radius: 0.1,
            }])
            .is_err()
        );
        assert!(PayloadView::new(&[0u8; 16]).is_err());
        let mut bad = build_payload(&[Capsule {
            a: [0.0; 3],
            b: [1.0, 0.0, 0.0],
            radius: 0.1,
        }])
        .unwrap();
        bad[0] ^= 0xFF;
        assert!(PayloadView::new(&bad).is_err());
    }
}
