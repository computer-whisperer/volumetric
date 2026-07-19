//! Build-time BVH over face AABBs — the `trimesh_model_core` median-split
//! scheme, except leaves index a slot table (face records are
//! variable-size, so they aren't reordered; the slot table is).

pub const LEAF_FLAG: u32 = 0x8000_0000;
const MAX_LEAF_FACES: usize = 4;

pub struct Node {
    pub min: [f32; 3],
    pub max: [f32; 3],
    /// Internal: left child index. Leaf: `LEAF_FLAG | first slot`.
    pub a: u32,
    /// Internal: right child index. Leaf: slot count.
    pub b: u32,
}

pub struct Bvh {
    pub nodes: Vec<Node>,
    /// Face indices in leaf-contiguous order.
    pub slots: Vec<u32>,
}

/// Build over `aabbs` (`[min_x, max_x, min_y, max_y, min_z, max_z]` per
/// face). f32 node bounds round outward so no face escapes its node.
pub fn build(aabbs: &[[f64; 6]]) -> Bvh {
    assert!(!aabbs.is_empty(), "BVH over zero faces");
    let mut slots: Vec<u32> = (0..aabbs.len() as u32).collect();
    let mut nodes = Vec::new();
    split(&mut nodes, &mut slots, 0, aabbs);
    Bvh { nodes, slots }
}

fn split(nodes: &mut Vec<Node>, slots: &mut [u32], base: usize, aabbs: &[[f64; 6]]) -> u32 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &f in slots.iter() {
        let bb = &aabbs[f as usize];
        for axis in 0..3 {
            min[axis] = min[axis].min(bb[axis * 2]);
            max[axis] = max[axis].max(bb[axis * 2 + 1]);
        }
    }
    let idx = nodes.len() as u32;
    nodes.push(Node {
        // Round outward: f32 must contain the f64 box.
        min: core::array::from_fn(|a| next_down(min[a] as f32)),
        max: core::array::from_fn(|a| next_up(max[a] as f32)),
        a: 0,
        b: 0,
    });

    if slots.len() <= MAX_LEAF_FACES {
        nodes[idx as usize].a = LEAF_FLAG | base as u32;
        nodes[idx as usize].b = slots.len() as u32;
        return idx;
    }

    // Split at the centroid median along the widest axis.
    let axis = (0..3)
        .max_by(|&a, &b| (max[a] - min[a]).total_cmp(&(max[b] - min[b])))
        .unwrap();
    let centroid = |f: u32| {
        let bb = &aabbs[f as usize];
        bb[axis * 2] + bb[axis * 2 + 1]
    };
    slots.sort_by(|&x, &y| centroid(x).total_cmp(&centroid(y)));
    let mid = slots.len() / 2;
    let (left, right) = slots.split_at_mut(mid);
    let a = split(nodes, left, base, aabbs);
    let b = split(nodes, right, base + mid, aabbs);
    nodes[idx as usize].a = a;
    nodes[idx as usize].b = b;
    idx
}

fn next_up(x: f32) -> f32 {
    if x.is_finite() {
        f32::from_bits(if x > 0.0 {
            x.to_bits() + 1
        } else if x < 0.0 {
            x.to_bits() - 1
        } else {
            1
        })
    } else {
        x
    }
}

fn next_down(x: f32) -> f32 {
    -next_up(-x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_all_faces_once() {
        let aabbs: Vec<[f64; 6]> = (0..37)
            .map(|i| {
                let x = i as f64;
                [x, x + 0.5, 0.0, 1.0, 0.0, 1.0]
            })
            .collect();
        let bvh = build(&aabbs);
        let mut seen = vec![false; aabbs.len()];
        for n in &bvh.nodes {
            if n.a & LEAF_FLAG != 0 {
                let first = (n.a & !LEAF_FLAG) as usize;
                for s in first..first + n.b as usize {
                    let f = bvh.slots[s] as usize;
                    assert!(!seen[f], "face {f} in two leaves");
                    seen[f] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn nodes_contain_children() {
        let aabbs: Vec<[f64; 6]> = (0..64)
            .map(|i| {
                let x = (i % 4) as f64;
                let y = ((i / 4) % 4) as f64;
                let z = (i / 16) as f64;
                [x, x + 1.0, y, y + 1.0, z, z + 1.0]
            })
            .collect();
        let bvh = build(&aabbs);
        for n in &bvh.nodes {
            if n.a & LEAF_FLAG != 0 {
                let first = (n.a & !LEAF_FLAG) as usize;
                for s in first..first + n.b as usize {
                    let bb = &aabbs[bvh.slots[s] as usize];
                    for axis in 0..3 {
                        assert!(n.min[axis] as f64 <= bb[axis * 2]);
                        assert!(n.max[axis] as f64 >= bb[axis * 2 + 1]);
                    }
                }
            }
        }
    }
}
