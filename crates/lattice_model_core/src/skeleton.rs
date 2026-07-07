//! Explicit strut skeletons of the lattice families — the same networks
//! the implicit occupancy functions thicken, enumerated as nodes + edges.
//!
//! The contract with the implicit side (`lattice_occupied`): both derive
//! from the same lattice definition in the same coordinate convention
//! (nodes at absolute model coordinates that are `cell_size` multiples of
//! the family's lattice points, anchored at the origin — NOT at a domain
//! corner), so a point on an enumerated edge is occupied at any positive
//! density and the explicit skeleton is exactly the implicit one's
//! centerline. Tests below hold the two sides together.
//!
//! Consumers (the strut-pattern operator) clip the enumerated network
//! against a domain model; enumeration covers the queried box expanded by
//! one cell so boundary-crossing struts are present.

/// The strut-lattice families with enumerable skeletons. (The TPMS sheet
/// and honeycomb-wall families have no strut skeleton and are implicit
/// only; foam joins here when its Voronoi edge extraction lands.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkeletonFamily {
    /// Edges of the cubic cell lattice ([`crate::LatticeKind::Struts`]):
    /// axis-aligned struts between adjacent integer lattice points,
    /// interior node degree 6.
    Cubic,
    /// Diamond-bond tetrahedral lattice ([`crate::LatticeKind::Tetra`]):
    /// struts along the bonds of the diamond lattice, every interior node
    /// joining exactly 4 struts, length `sqrt(3)/4 * cell_size`.
    Tetra,
}

/// An enumerated strut network: deduplicated nodes and the edges between
/// them. Node and edge order is deterministic for a given query box.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    /// Node positions in model coordinates.
    pub nodes: Vec<[f64; 3]>,
    /// Node-index pairs, one per strut.
    pub edges: Vec<[u32; 2]>,
}

/// Enumerate a family's skeleton covering the axis-aligned box `[lo, hi]`
/// expanded by one cell on every side. `cell_size` must be positive and
/// finite; degenerate boxes yield an empty skeleton.
pub fn enumerate_skeleton(
    family: SkeletonFamily,
    lo: [f64; 3],
    hi: [f64; 3],
    cell_size: f64,
) -> Skeleton {
    if !(cell_size.is_finite() && cell_size > 0.0)
        || lo.iter().chain(&hi).any(|v| !v.is_finite())
        || lo.iter().zip(&hi).any(|(a, b)| a > b)
    {
        return Skeleton::default();
    }
    // Integer cell range covering the box plus one cell of margin.
    let cell_lo: [i64; 3] = core::array::from_fn(|a| (lo[a] / cell_size).floor() as i64 - 1);
    let cell_hi: [i64; 3] = core::array::from_fn(|a| (hi[a] / cell_size).ceil() as i64 + 1);
    match family {
        SkeletonFamily::Cubic => cubic_skeleton(cell_lo, cell_hi, cell_size),
        SkeletonFamily::Tetra => tetra_skeleton(cell_lo, cell_hi, cell_size),
    }
}

/// A rough strut-count estimate for the query box (used by operators to
/// refuse absurd cell sizes before enumerating).
pub fn estimate_strut_count(
    family: SkeletonFamily,
    lo: [f64; 3],
    hi: [f64; 3],
    cell_size: f64,
) -> u64 {
    if !(cell_size.is_finite() && cell_size > 0.0) {
        return 0;
    }
    let cells: u64 = (0..3)
        .map(|a| ((hi[a] - lo[a]) / cell_size).ceil().max(0.0) as u64 + 3)
        .product();
    match family {
        SkeletonFamily::Cubic => cells.saturating_mul(3),
        SkeletonFamily::Tetra => cells.saturating_mul(16),
    }
}

/// Cubic lattice: nodes at integer points, edges to the +x/+y/+z
/// neighbors (each edge owned by its lower endpoint — no duplicates).
fn cubic_skeleton(cell_lo: [i64; 3], cell_hi: [i64; 3], cell_size: f64) -> Skeleton {
    let counts: [i64; 3] = core::array::from_fn(|a| cell_hi[a] - cell_lo[a] + 1);
    let node_id = |i: i64, j: i64, k: i64| -> u32 {
        (((k - cell_lo[2]) * counts[1] + (j - cell_lo[1])) * counts[0] + (i - cell_lo[0])) as u32
    };

    let mut nodes = Vec::with_capacity((counts[0] * counts[1] * counts[2]) as usize);
    for k in cell_lo[2]..=cell_hi[2] {
        for j in cell_lo[1]..=cell_hi[1] {
            for i in cell_lo[0]..=cell_hi[0] {
                nodes.push([
                    i as f64 * cell_size,
                    j as f64 * cell_size,
                    k as f64 * cell_size,
                ]);
            }
        }
    }

    let mut edges = Vec::new();
    for k in cell_lo[2]..=cell_hi[2] {
        for j in cell_lo[1]..=cell_hi[1] {
            for i in cell_lo[0]..=cell_hi[0] {
                let a = node_id(i, j, k);
                if i < cell_hi[0] {
                    edges.push([a, node_id(i + 1, j, k)]);
                }
                if j < cell_hi[1] {
                    edges.push([a, node_id(i, j + 1, k)]);
                }
                if k < cell_hi[2] {
                    edges.push([a, node_id(i, j, k + 1)]);
                }
            }
        }
    }
    Skeleton { nodes, edges }
}

/// Diamond lattice: atoms live on the quarter-integer grid (A sublattice
/// at `4 * (cell + FCC basis)`, B at A + (1,1,1)); each of a cell's four A
/// atoms owns its four bonds, so every bond is emitted exactly once.
fn tetra_skeleton(cell_lo: [i64; 3], cell_hi: [i64; 3], cell_size: f64) -> Skeleton {
    /// FCC basis in quarter units.
    const BASIS_Q: [[i64; 3]; 4] = [[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0]];
    /// Diamond bond offsets in quarter units (`DIAMOND_BONDS * 4`).
    const BONDS_Q: [[i64; 3]; 4] = [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]];

    let mut ids: std::collections::HashMap<[i64; 3], u32> = std::collections::HashMap::new();
    let mut nodes: Vec<[f64; 3]> = Vec::new();
    let mut edges: Vec<[u32; 2]> = Vec::new();
    let quarter = cell_size / 4.0;
    let mut id_of = |q: [i64; 3], nodes: &mut Vec<[f64; 3]>| -> u32 {
        *ids.entry(q).or_insert_with(|| {
            nodes.push([
                q[0] as f64 * quarter,
                q[1] as f64 * quarter,
                q[2] as f64 * quarter,
            ]);
            (nodes.len() - 1) as u32
        })
    };

    for k in cell_lo[2]..=cell_hi[2] {
        for j in cell_lo[1]..=cell_hi[1] {
            for i in cell_lo[0]..=cell_hi[0] {
                for basis in BASIS_Q {
                    let a = [4 * i + basis[0], 4 * j + basis[1], 4 * k + basis[2]];
                    let a_id = id_of(a, &mut nodes);
                    for bond in BONDS_Q {
                        let b = [a[0] + bond[0], a[1] + bond[1], a[2] + bond[2]];
                        let b_id = id_of(b, &mut nodes);
                        edges.push([a_id, b_id]);
                    }
                }
            }
        }
    }
    Skeleton { nodes, edges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LatticeKind, LatticeParams, lattice_occupied};

    fn segment_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
        let t = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
        let tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
        let s = ((ap[0] * t[0] + ap[1] * t[1] + ap[2] * t[2]) / tt).clamp(0.0, 1.0);
        let e = [ap[0] - s * t[0], ap[1] - s * t[1], ap[2] - s * t[2]];
        (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
    }

    fn min_distance(skeleton: &Skeleton, p: [f64; 3]) -> f64 {
        skeleton
            .edges
            .iter()
            .map(|e| segment_distance(p, skeleton.nodes[e[0] as usize], skeleton.nodes[e[1] as usize]))
            .fold(f64::MAX, f64::min)
    }

    fn implicit_kind(family: SkeletonFamily) -> LatticeKind {
        match family {
            SkeletonFamily::Cubic => LatticeKind::Struts,
            SkeletonFamily::Tetra => LatticeKind::Tetra,
        }
    }

    /// Probe points at odd offsets spanning a couple of cells.
    fn probes(cell: f64) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for i in 0..9 {
            for j in 0..9 {
                for k in 0..9 {
                    out.push([
                        (0.013 + 0.23 * i as f64) * cell,
                        (-0.71 + 0.29 * j as f64) * cell,
                        (0.19 + 0.31 * k as f64) * cell,
                    ]);
                }
            }
        }
        out
    }

    #[test]
    fn explicit_skeleton_matches_implicit_occupancy() {
        // Points ON enumerated struts are occupied at any positive density;
        // points measurably far from every strut are unoccupied at sparse
        // density (strut radii at d=0.05 are well under 0.1 cells for both
        // families). This pins the explicit and implicit lattices to the
        // same network in the same coordinates.
        let cell = 0.25;
        let (lo, hi) = ([0.0; 3], [1.0; 3]);
        let params = LatticeParams::default();
        for family in [SkeletonFamily::Cubic, SkeletonFamily::Tetra] {
            let skeleton = enumerate_skeleton(family, lo, hi, cell);
            let kind = implicit_kind(family);

            let mut on_strut_checked = 0usize;
            for (n, e) in skeleton.edges.iter().enumerate() {
                let a = skeleton.nodes[e[0] as usize];
                let b = skeleton.nodes[e[1] as usize];
                let t = 0.15 + 0.7 * (n % 7) as f64 / 7.0;
                let p = [
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2]),
                ];
                assert!(
                    lattice_occupied(kind, p, cell, 0.05, &params),
                    "{family:?}: point {p:?} on strut {e:?} not occupied"
                );
                on_strut_checked += 1;
            }
            assert!(on_strut_checked > 100, "{family:?}: too few struts");

            let mut far_checked = 0usize;
            for p in probes(cell) {
                if min_distance(&skeleton, p) > 0.1 * cell {
                    assert!(
                        !lattice_occupied(kind, p, cell, 0.05, &params),
                        "{family:?}: point {p:?} far from every strut is occupied"
                    );
                    far_checked += 1;
                }
            }
            assert!(far_checked > 100, "{family:?}: probe set degenerate");
        }
    }

    #[test]
    fn interior_node_degrees_match_the_families() {
        let cell = 0.5;
        for (family, expected_degree) in [(SkeletonFamily::Cubic, 6), (SkeletonFamily::Tetra, 4)] {
            let skeleton = enumerate_skeleton(family, [0.0; 3], [2.0; 3], cell);
            let mut degree = vec![0usize; skeleton.nodes.len()];
            for e in &skeleton.edges {
                degree[e[0] as usize] += 1;
                degree[e[1] as usize] += 1;
            }
            // Nodes well inside the enumerated range have full degree.
            let mut interior = 0usize;
            for (n, p) in skeleton.nodes.iter().enumerate() {
                if p.iter().all(|&v| v > 0.4 && v < 1.6) {
                    assert_eq!(
                        degree[n], expected_degree,
                        "{family:?}: interior node {n} at {p:?} has degree {}",
                        degree[n]
                    );
                    interior += 1;
                }
            }
            assert!(interior > 4, "{family:?}: no interior nodes checked");
        }
    }

    #[test]
    fn edges_are_deduplicated_and_valid() {
        for family in [SkeletonFamily::Cubic, SkeletonFamily::Tetra] {
            let skeleton = enumerate_skeleton(family, [0.0; 3], [1.0; 3], 0.25);
            let mut seen = std::collections::HashSet::new();
            for e in &skeleton.edges {
                assert_ne!(e[0], e[1], "{family:?}: self-loop {e:?}");
                assert!((e[0] as usize) < skeleton.nodes.len());
                assert!((e[1] as usize) < skeleton.nodes.len());
                let key = (e[0].min(e[1]), e[0].max(e[1]));
                assert!(seen.insert(key), "{family:?}: duplicate edge {e:?}");
            }
        }
    }

    #[test]
    fn tetra_struts_have_the_bond_length() {
        let cell = 0.4;
        let skeleton = enumerate_skeleton(SkeletonFamily::Tetra, [0.0; 3], [1.0; 3], cell);
        let expected = 3.0f64.sqrt() / 4.0 * cell;
        for e in &skeleton.edges {
            let a = skeleton.nodes[e[0] as usize];
            let b = skeleton.nodes[e[1] as usize];
            let len = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
            assert!(
                (len - expected).abs() < 1e-12,
                "bond {e:?} has length {len}, expected {expected}"
            );
        }
    }

    #[test]
    fn degenerate_inputs_yield_empty_skeletons() {
        for family in [SkeletonFamily::Cubic, SkeletonFamily::Tetra] {
            for bad_cell in [0.0, -1.0, f64::NAN] {
                let s = enumerate_skeleton(family, [0.0; 3], [1.0; 3], bad_cell);
                assert!(s.nodes.is_empty() && s.edges.is_empty());
            }
            let s = enumerate_skeleton(family, [1.0; 3], [0.0; 3], 0.25);
            assert!(s.nodes.is_empty() && s.edges.is_empty());
        }
    }
}
