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

use crate::cell::ConvexCell;

/// The strut-lattice families with enumerable skeletons. (The TPMS sheet
/// and honeycomb-wall families have no strut skeleton and are implicit
/// only.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SkeletonFamily {
    /// Edges of the cubic cell lattice ([`crate::LatticeKind::Struts`]):
    /// axis-aligned struts between adjacent integer lattice points,
    /// interior node degree 6.
    Cubic,
    /// Diamond-bond tetrahedral lattice ([`crate::LatticeKind::Tetra`]):
    /// struts along the bonds of the diamond lattice, every interior node
    /// joining exactly 4 struts, length `sqrt(3)/4 * cell_size`.
    Tetra,
    /// Voronoi foam ([`crate::LatticeKind::Foam`]): the edge skeleton of
    /// the Voronoi diagram of the jittered-BCC site set — Plateau
    /// geometry, every interior node joining exactly 4 struts.
    /// `irregularity` as in [`crate::LatticeParams`]: 0 is the periodic
    /// Kelvin cell, 1 fully organic (clamped to `[0, 1]`, non-finite
    /// reads 0 — the same site set the implicit foam thickens).
    Foam { irregularity: f64 },
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
        SkeletonFamily::Foam { irregularity } => {
            let jitter = if irregularity.is_finite() {
                irregularity.clamp(0.0, 1.0)
            } else {
                0.0
            };
            foam_skeleton(cell_lo, cell_hi, cell_size, jitter)
        }
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
        // A Kelvin cell has 36 edges shared 3 ways, 2 sites per unit cell.
        SkeletonFamily::Foam { .. } => cells.saturating_mul(24),
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

/// A jittered-BCC site's identity: base cell + coset. Plane/edge/node
/// dedup keys are built from these (never from float positions), so the
/// same Voronoi feature seen from adjacent cells welds exactly.
type Site = ([i64; 3], u8);

/// Half the box the Voronoi cell is clipped from, in cell units. Any
/// point farther than the lattice covering radius (~1.0 at full jitter)
/// from a site has a closer site, so cells never reach this box and the
/// box faces always clip away entirely.
const FOAM_CELL_BOX: f64 = 2.0;

/// Voronoi foam: for every site in range, clip its Voronoi cell against
/// the neighboring sites (both cosets, sorted nearest-first with an
/// early-out once no remaining bisector can reach the shrinking cell) and
/// emit the cell's edges. Edges dedupe by their three generating sites,
/// nodes weld by their four — identity comes from the lattice, not from
/// float positions, so adjacent cells agree exactly.
fn foam_skeleton(cell_lo: [i64; 3], cell_hi: [i64; 3], cell_size: f64, jitter: f64) -> Skeleton {
    // Node identity: the sorted set of sites the Voronoi vertex is
    // equidistant to (4 generically, more only in degenerate configs).
    let mut node_ids: std::collections::HashMap<Vec<Site>, u32> = std::collections::HashMap::new();
    let mut edge_seen: std::collections::HashSet<[Site; 3]> = std::collections::HashSet::new();
    let mut nodes: Vec<[f64; 3]> = Vec::new();
    let mut edges: Vec<[u32; 2]> = Vec::new();

    for k in cell_lo[2]..=cell_hi[2] {
        for j in cell_lo[1]..=cell_hi[1] {
            for i in cell_lo[0]..=cell_hi[0] {
                for coset in 0..2u8 {
                    let site: Site = ([i, j, k], coset);
                    let center = crate::foam_site([i, j, k], coset as usize, jitter);

                    // Neighbor sites of both cosets within +-2 cells,
                    // nearest first.
                    let mut neighbors: Vec<(Site, [f64; 3], f64)> = Vec::with_capacity(249);
                    for dz in -2..=2i64 {
                        for dy in -2..=2i64 {
                            for dx in -2..=2i64 {
                                for nc in 0..2u8 {
                                    if dx == 0 && dy == 0 && dz == 0 && nc == coset {
                                        continue;
                                    }
                                    let base = [i + dx, j + dy, k + dz];
                                    let p = crate::foam_site(base, nc as usize, jitter);
                                    let d2 =
                                        (0..3).map(|a| (p[a] - center[a]).powi(2)).sum::<f64>();
                                    neighbors.push(((base, nc), p, d2));
                                }
                            }
                        }
                    }
                    neighbors.sort_by(|a, b| a.2.total_cmp(&b.2));

                    let r = FOAM_CELL_BOX;
                    let mut cell = ConvexCell::from_aabb(
                        [center[0] - r, center[1] - r, center[2] - r],
                        [center[0] + r, center[1] + r, center[2] + r],
                    );
                    let mut max_d2 = cell.max_dist2(center);
                    for (plane_index, (_, p, d2)) in neighbors.iter().enumerate() {
                        // The bisector sits at distance sqrt(d2)/2: past the
                        // farthest vertex it can't cut anything, and
                        // neighbors only get farther.
                        if *d2 > 4.0 * max_d2 {
                            break;
                        }
                        let n = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                        let mid = [
                            0.5 * (p[0] + center[0]),
                            0.5 * (p[1] + center[1]),
                            0.5 * (p[2] + center[2]),
                        ];
                        let d = n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2];
                        cell.clip(6 + plane_index, n, d);
                        max_d2 = cell.max_dist2(center);
                    }

                    // A plane id below 6 is the clip box: the cell wasn't
                    // closed by its neighbors (can't happen with a sane
                    // site set) — skip the feature rather than emit
                    // box-artifact geometry.
                    let neighbor_site = |plane: usize| -> Option<Site> {
                        Some(neighbors.get(plane.checked_sub(6)?)?.0)
                    };
                    for (va, vb, shared) in cell.edges() {
                        // Only cleanly-two-plane edges: anything else is a
                        // near-degenerate sliver (see cell.rs docs).
                        let [p0, p1] = shared.as_slice() else {
                            continue;
                        };
                        let (Some(n0), Some(n1)) = (neighbor_site(*p0), neighbor_site(*p1)) else {
                            continue;
                        };
                        let mut triple = [site, n0, n1];
                        triple.sort_unstable();
                        if edge_seen.contains(&triple) {
                            continue;
                        }
                        // Drop numeric slivers (near-coincident Voronoi
                        // vertices): real foam edges are O(0.1) cells.
                        let (pa, pb) = (cell.vertices[va as usize], cell.vertices[vb as usize]);
                        let len2 = (0..3).map(|a| (pa[a] - pb[a]).powi(2)).sum::<f64>();
                        if len2 < 1e-12 {
                            continue;
                        }
                        let mut node_of = |v: u32| -> Option<u32> {
                            let mut quad: Vec<Site> = std::iter::once(Some(site))
                                .chain(
                                    cell.generators[v as usize]
                                        .iter()
                                        .map(|&g| neighbor_site(g)),
                                )
                                .collect::<Option<Vec<Site>>>()?;
                            quad.sort_unstable();
                            Some(*node_ids.entry(quad).or_insert_with(|| {
                                let p = cell.vertices[v as usize];
                                nodes.push([p[0] * cell_size, p[1] * cell_size, p[2] * cell_size]);
                                (nodes.len() - 1) as u32
                            }))
                        };
                        let (Some(a), Some(b)) = (node_of(va), node_of(vb)) else {
                            continue;
                        };
                        if a != b {
                            edge_seen.insert(triple);
                            edges.push([a, b]);
                        }
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
            .map(|e| {
                segment_distance(
                    p,
                    skeleton.nodes[e[0] as usize],
                    skeleton.nodes[e[1] as usize],
                )
            })
            .fold(f64::MAX, f64::min)
    }

    fn implicit_kind(family: SkeletonFamily) -> (LatticeKind, LatticeParams) {
        match family {
            SkeletonFamily::Cubic => (LatticeKind::Struts, LatticeParams::default()),
            SkeletonFamily::Tetra => (LatticeKind::Tetra, LatticeParams::default()),
            SkeletonFamily::Foam { irregularity } => (
                LatticeKind::Foam,
                LatticeParams {
                    irregularity,
                    ..LatticeParams::default()
                },
            ),
        }
    }

    fn all_families() -> Vec<SkeletonFamily> {
        vec![
            SkeletonFamily::Cubic,
            SkeletonFamily::Tetra,
            SkeletonFamily::Foam { irregularity: 0.0 },
            SkeletonFamily::Foam { irregularity: 0.35 },
        ]
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
        // density (implicit strut cross-sections at d=0.04 stay well under
        // 0.12 cells for every family). This pins the explicit and implicit
        // lattices to the same network in the same coordinates.
        let cell = 0.25;
        let (lo, hi) = ([0.0; 3], [1.0; 3]);
        for family in all_families() {
            let skeleton = enumerate_skeleton(family, lo, hi, cell);
            let (kind, params) = implicit_kind(family);

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
                    lattice_occupied(kind, p, cell, 0.04, &params),
                    "{family:?}: point {p:?} on strut {e:?} not occupied"
                );
                on_strut_checked += 1;
            }
            assert!(on_strut_checked > 100, "{family:?}: too few struts");

            let mut far_checked = 0usize;
            for p in probes(cell) {
                if min_distance(&skeleton, p) > 0.12 * cell {
                    assert!(
                        !lattice_occupied(kind, p, cell, 0.04, &params),
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
        for family in all_families() {
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
    fn foam_edges_are_voronoi_equidistant() {
        // The defining property, independent of any threshold: every point
        // of a foam strut is equidistant to its three nearest sites, and
        // every node to its four nearest (foam_nearest_three returns the
        // closest three — on a vertex all three ties).
        for irregularity in [0.0, 0.35, 0.8] {
            let skeleton = enumerate_skeleton(
                SkeletonFamily::Foam { irregularity },
                [0.0; 3],
                [2.0; 3],
                1.0,
            );
            assert!(skeleton.edges.len() > 100, "few edges at {irregularity}");
            for (n, e) in skeleton.edges.iter().enumerate() {
                let a = skeleton.nodes[e[0] as usize];
                let b = skeleton.nodes[e[1] as usize];
                let t = 0.2 + 0.6 * (n % 5) as f64 / 5.0;
                let mid = [
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2]),
                ];
                for p in [a, b, mid] {
                    let [d1, d2, d3] = crate::foam_nearest_three(p, irregularity);
                    assert!(
                        d3 - d1 < 1e-9,
                        "irr {irregularity}: point {p:?} on edge {e:?} has site \
                         distances {d1} / {d2} / {d3}"
                    );
                }
            }
        }
    }

    #[test]
    fn kelvin_foam_is_the_truncated_octahedron() {
        // irregularity 0 = plain BCC: every Voronoi edge of the truncated
        // octahedron has length sqrt(2)/4, interior nodes join 4 struts
        // (Plateau), and the density is 24 edges per unit cell.
        let skeleton = enumerate_skeleton(
            SkeletonFamily::Foam { irregularity: 0.0 },
            [0.0; 3],
            [3.0; 3],
            1.0,
        );
        let expected_len = 2.0f64.sqrt() / 4.0;
        for e in &skeleton.edges {
            let a = skeleton.nodes[e[0] as usize];
            let b = skeleton.nodes[e[1] as usize];
            let len = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
            assert!(
                (len - expected_len).abs() < 1e-9,
                "kelvin edge {e:?} has length {len}, expected {expected_len}"
            );
        }

        // Count edges by midpoint in the central 2^3 region.
        let central = skeleton
            .edges
            .iter()
            .filter(|e| {
                let a = skeleton.nodes[e[0] as usize];
                let b = skeleton.nodes[e[1] as usize];
                (0..3).all(|c| {
                    let m = 0.5 * (a[c] + b[c]);
                    (0.5..2.5).contains(&m)
                })
            })
            .count();
        assert_eq!(
            central,
            24 * 8,
            "kelvin edge density off: {central} in 8 cells"
        );
    }

    #[test]
    fn foam_interior_nodes_have_plateau_degree() {
        for irregularity in [0.0, 0.5] {
            let skeleton = enumerate_skeleton(
                SkeletonFamily::Foam { irregularity },
                [0.0; 3],
                [3.0; 3],
                1.0,
            );
            let mut degree = vec![0usize; skeleton.nodes.len()];
            for e in &skeleton.edges {
                degree[e[0] as usize] += 1;
                degree[e[1] as usize] += 1;
            }
            // Window edges off the lattice planes: Kelvin vertices sit at
            // quarter-unit coordinates and a grid-aligned strict window
            // would exclude most of them.
            let mut interior = 0usize;
            for (n, p) in skeleton.nodes.iter().enumerate() {
                if p.iter().all(|&v| v > 0.6 && v < 2.4) {
                    assert_eq!(
                        degree[n], 4,
                        "irr {irregularity}: interior node {n} at {p:?} joins {} struts",
                        degree[n]
                    );
                    interior += 1;
                }
            }
            assert!(
                interior > 30,
                "irr {irregularity}: too few interior nodes ({interior} of {} total; \
                 {} edges)",
                skeleton.nodes.len(),
                skeleton.edges.len()
            );
        }
    }

    #[test]
    fn jittered_foam_is_deterministic_and_reshaped() {
        let run = || {
            enumerate_skeleton(
                SkeletonFamily::Foam { irregularity: 0.4 },
                [0.0; 3],
                [2.0; 3],
                0.5,
            )
        };
        let (a, b) = (run(), run());
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.edges, b.edges);

        // Jitter actually moves the network relative to Kelvin.
        let kelvin = enumerate_skeleton(
            SkeletonFamily::Foam { irregularity: 0.0 },
            [0.0; 3],
            [2.0; 3],
            0.5,
        );
        let moved = a
            .nodes
            .iter()
            .filter(|p| {
                kelvin
                    .nodes
                    .iter()
                    .all(|q| (0..3).map(|c| (p[c] - q[c]).powi(2)).sum::<f64>() > 1e-6)
            })
            .count();
        assert!(
            moved > a.nodes.len() / 2,
            "jitter should move most nodes ({moved} of {})",
            a.nodes.len()
        );
    }

    #[test]
    fn degenerate_inputs_yield_empty_skeletons() {
        for family in all_families() {
            for bad_cell in [0.0, -1.0, f64::NAN] {
                let s = enumerate_skeleton(family, [0.0; 3], [1.0; 3], bad_cell);
                assert!(s.nodes.is_empty() && s.edges.is_empty());
            }
            let s = enumerate_skeleton(family, [1.0; 3], [0.0; 3], 0.25);
            assert!(s.nodes.is_empty() && s.edges.is_empty());
        }
    }
}
