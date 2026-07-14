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
//! Beyond geometry, the payload can carry per-strut *channel* values — the
//! raw FEA element-field scalars the operator passes through — plus the
//! opaque CBOR [`volumetric_abi::SampleFormat`] the template serves from
//! `get_sample_format`. Channels are strut-indexed in the same leaf order as
//! the capsules, so the strut index [`PayloadView::sample_owner`] returns
//! indexes both the capsule and its channel row.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (72 bytes):
//!    0  magic          u32   "STM2" (0x324D_5453)
//!    4  strut_count    u32
//!    8  node_count     u32
//!   12  payload_len    u32   total byte length, header included
//!   16  bounds         6xf64 [min_x, max_x, min_y, max_y, min_z, max_z]
//!   64  channel_count  u32   extra channels per strut (excludes occupancy)
//!   68  format_len     u32   byte length of the trailing CBOR SampleFormat
//! Nodes (node_count x 32 bytes at offset 72):
//!    0  aabb_min  3xf32   capsule boxes, radius included
//!   12  aabb_max  3xf32
//!   24  a         u32     internal: left child; leaf: 0x8000_0000 | first
//!   28  b         u32     internal: right child; leaf: capsule count
//! Capsules (strut_count x 28 bytes after the nodes): 3xf32 a, 3xf32 b,
//! f32 radius — reordered so each leaf's capsules are contiguous.
//! Channel values (strut_count x channel_count x f32 after the capsules):
//!   row for leaf-order strut e is channel_count f32s, same order as the
//!   SampleFormat's extra channels.
//! SampleFormat (format_len bytes at the very end): CBOR, stored verbatim.
//! ```

pub const MAGIC: u32 = 0x324D_5453; // "STM2"
const HEADER_LEN: usize = 72;
const NODE_LEN: usize = 32;
const CAPSULE_LEN: usize = 28;
const LEAF_FLAG: u32 = 0x8000_0000;
const MAX_LEAF_CAPSULES: usize = 4;

/// CBOR of the single-occupancy [`volumetric_abi::SampleFormat`] — the format
/// an unpatched template (or an occupancy-only payload) reports through
/// `get_sample_format`. Held here as literal bytes so the template stays free
/// of the CBOR/ABI dependency; the operator crate guards it against
/// `encode_sample_format(&SampleFormat::default())` in a unit test.
pub const OCCUPANCY_FORMAT_CBOR: &[u8] = &[
    161, 104, 99, 104, 97, 110, 110, 101, 108, 115, 129, 162, 100, 110, 97, 109, 101, 105, 111, 99,
    99, 117, 112, 97, 110, 99, 121, 100, 107, 105, 110, 100, 105, 79, 99, 99, 117, 112, 97, 110,
    99, 121,
];

/// Per-strut channel data to embed alongside the capsules. The caller builds
/// the CBOR [`volumetric_abi::SampleFormat`] (this crate stays ABI-agnostic)
/// and supplies one value row per strut, indexed like `capsules`.
#[derive(Clone, Debug, Default)]
pub struct ChannelPayload {
    /// Number of extra channels per strut (excludes occupancy).
    pub count: usize,
    /// CBOR-encoded `SampleFormat` (occupancy + the extra channels), stored
    /// verbatim and served by the template's `get_sample_format`.
    pub format_cbor: Vec<u8>,
    /// Per-strut values in the SAME order as `capsules`, row-major: strut `e`
    /// occupies `values[e * count .. (e + 1) * count]`. Length must be
    /// `capsules.len() * count`.
    pub values: Vec<f32>,
}

/// One strut, realized as the segment `[a, b]` swept by `radius`.
#[derive(Clone, Copy, Debug)]
pub struct Capsule {
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub radius: f64,
}

/// Build the payload: BVH over the capsules' radius-inflated boxes, then
/// leaf-ordered capsule records and their channel rows. Fails on an empty
/// list, on non-finite or non-positive geometry, and on a channel-value
/// count that doesn't match `strut_count x channel_count`.
pub fn build_payload(capsules: &[Capsule], channels: &ChannelPayload) -> Result<Vec<u8>, String> {
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
    if channels.values.len() != capsules.len() * channels.count {
        return Err(format!(
            "channel values length {} does not match strut count {} x channel count {}",
            channels.values.len(),
            capsules.len(),
            channels.count
        ));
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

    let payload_len = HEADER_LEN
        + nodes.len() * NODE_LEN
        + capsules.len() * CAPSULE_LEN
        + capsules.len() * channels.count * 4
        + channels.format_cbor.len();
    let mut out = Vec::with_capacity(payload_len);
    out.extend(MAGIC.to_le_bytes());
    out.extend((capsules.len() as u32).to_le_bytes());
    out.extend((nodes.len() as u32).to_le_bytes());
    out.extend((payload_len as u32).to_le_bytes());
    for v in &bounds {
        out.extend(v.to_le_bytes());
    }
    out.extend((channels.count as u32).to_le_bytes());
    out.extend((channels.format_cbor.len() as u32).to_le_bytes());
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
    // Channel rows, reordered to match the leaf-order capsules above.
    for &c in &order {
        let base = c as usize * channels.count;
        for v in &channels.values[base..base + channels.count] {
            out.extend(v.to_le_bytes());
        }
    }
    out.extend_from_slice(&channels.format_cbor);
    debug_assert_eq!(out.len(), payload_len);
    Ok(out)
}

/// A read-only view over a serialized payload (sample-time side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    node_count: usize,
    strut_count: usize,
    channel_count: usize,
    format_off: usize,
    format_len: usize,
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
        let channel_count = u32_at(64) as usize;
        let format_len = u32_at(68) as usize;
        let format_off = HEADER_LEN
            + node_count * NODE_LEN
            + strut_count * CAPSULE_LEN
            + strut_count * channel_count * 4;
        let expected = format_off + format_len;
        if payload_len != expected || bytes.len() < expected {
            return Err("payload length mismatch");
        }
        Ok(Self {
            bytes,
            node_count,
            strut_count,
            channel_count,
            format_off,
            format_len,
        })
    }

    /// `[min_x, max_x, min_y, max_y, min_z, max_z]`.
    pub fn bounds(&self) -> [f64; 6] {
        std::array::from_fn(|i| {
            f64::from_le_bytes(self.bytes[16 + i * 8..24 + i * 8].try_into().unwrap())
        })
    }

    /// Number of extra channels per strut (excludes occupancy).
    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    /// `(offset, len)` of the embedded CBOR `SampleFormat` within these
    /// payload bytes. The template adds the payload's base memory address to
    /// the offset to hand `get_sample_format` a linear-memory pointer.
    pub fn format_range(&self) -> (usize, usize) {
        (self.format_off, self.format_len)
    }

    /// The `channel`-th extra channel value (0-based, occupancy excluded) for
    /// the strut at leaf-order index `strut` — the index
    /// [`Self::sample_owner`] returns. Out-of-range indices are the caller's
    /// responsibility (the template only passes what it read from the header).
    pub fn channel_value(&self, strut: usize, channel: usize) -> f32 {
        let base = HEADER_LEN + self.node_count * NODE_LEN + self.strut_count * CAPSULE_LEN;
        self.f32_at(base + (strut * self.channel_count + channel) * 4)
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

    /// The leaf-order index of the strut that "owns" `p`: among the capsules
    /// containing `p`, the one whose axis is nearest (minimum segment
    /// distance), ties going to the lower index. `None` iff `p` is outside
    /// every capsule (equivalently, `!is_inside(p)`). Feed the result to
    /// [`Self::channel_value`]. Unlike [`Self::is_inside`], this visits every
    /// containing capsule (no early-out) to find the nearest.
    pub fn sample_owner(&self, p: [f64; 3]) -> Option<usize> {
        if self.node_count == 0 {
            return None;
        }
        let capsule_base = HEADER_LEN + self.node_count * NODE_LEN;
        let mut stack = [0u32; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;
        let mut best: Option<(usize, f64)> = None;

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
                    let d2 = segment_distance_squared(p, ca, cb);
                    if d2 <= radius * radius && best.is_none_or(|(_, bd)| d2 < bd) {
                        best = Some((c, d2));
                    }
                }
            } else {
                stack[top] = a;
                top += 1;
                stack[top] = b;
                top += 1;
            }
        }
        best.map(|(c, _)| c)
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

    /// Occupancy-only payload (no channels) — the shape the geometry tests want.
    fn occ_payload(capsules: &[Capsule]) -> Result<Vec<u8>, String> {
        build_payload(capsules, &ChannelPayload::default())
    }

    #[test]
    fn single_capsule_classifies_analytically() {
        // Unit segment along x at radius 0.1.
        let payload = occ_payload(&[Capsule {
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
        let payload = occ_payload(&[Capsule {
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
        let payload = occ_payload(&capsules).unwrap();
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
        let payload = occ_payload(&capsules).unwrap();
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
        assert!(occ_payload(&[]).is_err());
        assert!(
            occ_payload(&[Capsule {
                a: [0.0; 3],
                b: [1.0, 0.0, 0.0],
                radius: 0.0,
            }])
            .is_err()
        );
        assert!(
            occ_payload(&[Capsule {
                a: [f64::NAN, 0.0, 0.0],
                b: [1.0, 0.0, 0.0],
                radius: 0.1,
            }])
            .is_err()
        );
        assert!(PayloadView::new(&[0u8; 16]).is_err());
        let mut bad = occ_payload(&[Capsule {
            a: [0.0; 3],
            b: [1.0, 0.0, 0.0],
            radius: 0.1,
        }])
        .unwrap();
        bad[0] ^= 0xFF;
        assert!(PayloadView::new(&bad).is_err());
    }

    #[test]
    fn channels_round_trip_in_leaf_order() {
        // Three well-separated capsules with two channels each. Values are
        // keyed to the strut so we can prove leaf reordering keeps rows glued
        // to their capsule.
        let capsules = vec![
            Capsule {
                a: [0.0, 0.0, 0.0],
                b: [1.0, 0.0, 0.0],
                radius: 0.1,
            },
            Capsule {
                a: [0.0, 5.0, 0.0],
                b: [1.0, 5.0, 0.0],
                radius: 0.1,
            },
            Capsule {
                a: [0.0, 10.0, 0.0],
                b: [1.0, 10.0, 0.0],
                radius: 0.1,
            },
        ];
        let channels = ChannelPayload {
            count: 2,
            format_cbor: b"FORMAT-BYTES".to_vec(),
            // strut e -> [10+e, 100+e]
            values: vec![10.0, 100.0, 11.0, 101.0, 12.0, 102.0],
        };
        let payload = build_payload(&capsules, &channels).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.channel_count(), 2);

        // The owning strut of a point on capsule e carries e's channel row,
        // regardless of how the BVH reordered the capsules internally.
        for (e, c) in capsules.iter().enumerate() {
            let mid = [0.5, c.a[1], 0.0];
            let owner = view.sample_owner(mid).expect("inside a capsule");
            assert_eq!(view.channel_value(owner, 0), 10.0 + e as f32);
            assert_eq!(view.channel_value(owner, 1), 100.0 + e as f32);
        }

        // The embedded format bytes round-trip via the declared range.
        let (off, len) = view.format_range();
        assert_eq!(&payload[off..off + len], b"FORMAT-BYTES");
        assert_eq!(off + len, payload.len());

        // Outside every capsule: no owner.
        assert!(view.sample_owner([0.5, 2.5, 0.0]).is_none());
    }

    #[test]
    fn sample_owner_picks_the_nearest_strut_at_a_shared_node() {
        // Two struts meeting at the origin, forming an L in the xy-plane. A
        // point near the +x arm's axis (but well off the +y arm) must be
        // owned by the +x arm.
        let capsules = vec![
            Capsule {
                a: [0.0, 0.0, 0.0],
                b: [2.0, 0.0, 0.0],
                radius: 0.3,
            },
            Capsule {
                a: [0.0, 0.0, 0.0],
                b: [0.0, 2.0, 0.0],
                radius: 0.3,
            },
        ];
        let channels = ChannelPayload {
            count: 1,
            format_cbor: OCCUPANCY_FORMAT_CBOR.to_vec(),
            values: vec![7.0, 9.0], // +x arm -> 7, +y arm -> 9
        };
        let payload = build_payload(&capsules, &channels).unwrap();
        let view = PayloadView::new(&payload).unwrap();

        // On the +x arm's axis: distance 0 to strut 0, ~1 to strut 1.
        let owner = view.sample_owner([1.0, 0.0, 0.0]).unwrap();
        assert_eq!(view.channel_value(owner, 0), 7.0);
        // On the +y arm's axis.
        let owner = view.sample_owner([0.0, 1.0, 0.0]).unwrap();
        assert_eq!(view.channel_value(owner, 0), 9.0);
    }
}
