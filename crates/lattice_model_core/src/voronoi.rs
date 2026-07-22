//! General Voronoi edge skeleton of an arbitrary 3D site set — the
//! machinery behind `voronoi_skeleton_operator`, generalizing the foam
//! family's cell clipping (`skeleton.rs`) off its jittered-BCC
//! assumptions.
//!
//! Per site: clip a padded bounding box against the bisector planes of
//! nearby sites, nearest first through a uniform grid hash, stopping once
//! the next candidate is provably too far to cut the shrunken cell
//! (`d > 2 * max vertex distance`). Emitted edges are the cell edges
//! whose supporting planes are all site bisectors (two or more): pure
//! box-artifact edges vanish, while genuine Voronoi edges truncated *at*
//! the box survive with an endpoint on it.
//!
//! Where the foam welds features by lattice identity (impossible here —
//! there is no lattice), nodes weld by position tolerance: adjacent cells
//! compute the same Voronoi vertex through different clip sequences and
//! land within float noise of each other, far below any genuine feature
//! separation; edges then dedupe by welded node pair. This also makes
//! degenerate inputs come out clean rather than as garbage: a perfect
//! cubic point grid (every Voronoi vertex 8-fold cospherical) yields
//! exactly the cubic strut lattice, its coincident vertices welded and
//! its multiply-supported edges deduped.
//!
//! All work happens in spacing-normalized coordinates (sites scaled so
//! the typical neighbor distance is O(1)), giving the shared
//! [`crate::cell`] tolerances the same meaning as in the foam's
//! cell-unit coordinates regardless of input scale.

use crate::cell::ConvexCell;
use crate::skeleton::Skeleton;

/// Position-weld tolerance in normalized (unit-spacing) coordinates:
/// far above accumulated clip float noise (~1e-12), far below any
/// genuine vertex separation of interest (~1e-2).
const WELD_TOL: f64 = 1e-6;

/// Near-coincident input sites (closer than this, normalized) collapse
/// to their first representative: duplicate sites make every downstream
/// bisector degenerate, and a user merging two point clouds shouldn't
/// have to hand-dedupe overlaps.
const SITE_DEDUP_TOL: f64 = 1e-6;

/// What happens where the diagram leaves the site cloud: hull cells are
/// infinite, so their outward edges either vanish or truncate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary {
    /// Keep only edges whose *both* endpoints are genuine Voronoi
    /// vertices (supported by three or more site bisectors): infinite
    /// hull edges vanish entirely and the skeleton ends at the cloud.
    /// The natural lattice semantics — no hairs reaching for the box.
    Trim,
    /// Truncate infinite hull edges at the padded bounding box (endpoint
    /// on the box). Use when something downstream cuts the rays to a
    /// real boundary — e.g. `mesh_clip` against a domain model, which
    /// turns them into skin-contact stubs.
    Box,
}

/// Knobs for [`voronoi_skeleton`].
pub struct VoronoiOptions {
    /// How far past the sites' bounding box the clip box sits (world
    /// units); <= 0 picks two typical spacings.
    pub padding: f64,
    /// What happens to infinite hull-cell edges (see [`Boundary`]).
    pub boundary: Boundary,
    /// [`Boundary::Trim`] only: drop edges reaching farther than
    /// `max_reach` times the generating cell's nearest-neighbor distance
    /// from its site. A vertex's site distance is its empty-circumsphere
    /// radius — interior vertices sit at ~0.6-1.0 local spacings, while
    /// the genuine-but-bulging vertices of a filled cloud's hull shell
    /// (near-coplanar site slivers, huge empty spheres) sit at several —
    /// so this is what makes the skeleton *end at the cloud* instead of
    /// ballooning past it. The bound is per-cell, so density-graded
    /// clouds trim correctly. <= 0 disables.
    pub max_reach: f64,
}

impl Default for VoronoiOptions {
    fn default() -> Self {
        Self {
            padding: 0.0,
            boundary: Boundary::Trim,
            max_reach: 1.5,
        }
    }
}

/// A computed Voronoi skeleton plus the site statistics consumers need
/// to pick sensible defaults.
pub struct VoronoiResult {
    /// Nodes in the input's coordinates; edges between them.
    pub skeleton: Skeleton,
    /// The typical site spacing (cube root of volume per site) — e.g.
    /// the natural default strut-radius scale.
    pub spacing: f64,
    /// Sites surviving the near-coincident dedup.
    pub site_count: usize,
}

/// Build the Voronoi edge skeleton of `sites` (see [`VoronoiOptions`]
/// for the boundary/padding/reach knobs; interior cells never feel
/// them). Errors on non-finite input; too-few or fully-degenerate site
/// sets yield an empty skeleton (0 edges) rather than an error so
/// callers can phrase the failure.
pub fn voronoi_skeleton(
    sites: &[[f64; 3]],
    options: &VoronoiOptions,
) -> Result<VoronoiResult, String> {
    let padding = options.padding;
    let boundary = options.boundary;
    if let Some(bad) = sites.iter().find(|p| !p.iter().all(|c| c.is_finite())) {
        return Err(format!("non-finite site {bad:?}"));
    }
    if sites.is_empty() {
        return Err("no sites (the input point cloud is empty)".to_string());
    }
    if !(padding.is_finite()) {
        return Err(format!("padding must be finite, got {padding}"));
    }

    // Bounds and the typical-spacing normalization scale.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in sites {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let extent: [f64; 3] = core::array::from_fn(|a| hi[a] - lo[a]);
    let volume: f64 = extent.iter().product();
    let longest = extent.iter().cloned().fold(0.0f64, f64::max);
    let spacing_for = |count: usize| -> f64 {
        if volume > 0.0 {
            (volume / count as f64).cbrt()
        } else if longest > 0.0 {
            // Planar/collinear clouds: fall back to the longest extent.
            longest / (count as f64).cbrt()
        } else {
            1.0 // all sites coincident; dedup leaves one, no edges
        }
    };

    // Near-coincident dedup first (world units, provisional spacing for
    // the tolerance), so the spacing estimate — and with it the
    // normalization and clip box — sees deduped sites only: feeding the
    // same cloud twice changes nothing downstream.
    let mut dedup = NodeWelder::new(SITE_DEDUP_TOL * spacing_for(sites.len()));
    for p in sites {
        dedup.intern(*p);
    }
    let kept_world = dedup.nodes;

    // Normalized sites on a unit-spacing grid hash.
    let spacing = spacing_for(kept_world.len());
    let scale = 1.0 / spacing;
    let kept: Vec<[f64; 3]> = kept_world
        .iter()
        .map(|p| core::array::from_fn(|a| p[a] * scale))
        .collect();
    let dims: [usize; 3] = core::array::from_fn(|a| (extent[a] * scale).ceil() as usize + 1);
    let cell_coords = |p: [f64; 3]| -> [i64; 3] {
        core::array::from_fn(|a| ((p[a] - lo[a] * scale) as i64).clamp(0, dims[a] as i64 - 1))
    };
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
    for (i, q) in kept.iter().enumerate() {
        let c = cell_coords(*q);
        buckets[(c[2] as usize * dims[1] + c[1] as usize) * dims[0] + c[0] as usize]
            .push(i as u32);
    }

    // The clip box: normalized bounds plus the padding.
    let pad = if padding > 0.0 { padding * scale } else { 2.0 };
    let box_lo: [f64; 3] = core::array::from_fn(|a| lo[a] * scale - pad);
    let box_hi: [f64; 3] = core::array::from_fn(|a| hi[a] * scale + pad);

    let mut welder = NodeWelder::new(WELD_TOL);
    let mut edges: Vec<[u32; 2]> = Vec::new();
    let mut edge_seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let max_ring = dims.iter().max().copied().unwrap_or(1) as i64 + 1;

    for (si, &site) in kept.iter().enumerate() {
        let mut cell = ConvexCell::from_aabb(box_lo, box_hi);
        let mut max_d2 = cell.max_dist2(site);
        let home = cell_coords(site);
        let mut plane_id = 6usize;
        // The cell's nearest-neighbor distance (squared): the nearest
        // site is always among the clipped candidates (its bisector
        // bounds the cell, so the security radius can never skip it, and
        // the ring floors can never skip its ring).
        let mut nn_d2 = f64::INFINITY;

        // Expanding Chebyshev rings of grid cells, candidates within a
        // ring processed nearest-first. A site anywhere in ring r is at
        // least (r - 1) grid cells away, so once that floor passes the
        // security radius nothing further out can cut.
        'rings: for r in 0..=max_ring {
            if r >= 1 {
                let ring_floor = (r - 1) as f64;
                if ring_floor * ring_floor > 4.0 * max_d2 {
                    break 'rings;
                }
            }
            let mut candidates: Vec<(f64, u32)> = Vec::new();
            for_ring_cells(home, r, &dims, |idx| {
                for &j in &buckets[idx] {
                    if j as usize == si {
                        continue;
                    }
                    let p = kept[j as usize];
                    let d2: f64 = (0..3).map(|a| (p[a] - site[a]).powi(2)).sum();
                    candidates.push((d2, j));
                }
            });
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (d2, j) in candidates {
                // The bisector sits at distance sqrt(d2)/2: past the
                // farthest vertex it can't cut anything, and the rest of
                // this ring is farther still.
                if d2 > 4.0 * max_d2 {
                    break;
                }
                let p = kept[j as usize];
                nn_d2 = nn_d2.min(d2);
                let n: [f64; 3] = core::array::from_fn(|a| p[a] - site[a]);
                let mid: [f64; 3] = core::array::from_fn(|a| 0.5 * (p[a] + site[a]));
                let d = (0..3).map(|a| n[a] * mid[a]).sum();
                cell.clip(plane_id, n, d);
                plane_id += 1;
                max_d2 = cell.max_dist2(site);
            }
        }

        for (va, vb, shared) in cell.edges() {
            // A genuine (possibly box-truncated) Voronoi edge is
            // supported by site bisectors alone — two generically, more
            // in degenerate configurations. Any box plane in the support
            // means the edge lies on the box: an artifact, skip.
            if shared.len() < 2 || shared.iter().any(|&p| p < 6) {
                continue;
            }
            // Trim mode: an endpoint whose generators include a box
            // plane is a truncation of an infinite edge, not a Voronoi
            // vertex — the edge is unsupported on that side; drop it.
            if boundary == Boundary::Trim
                && [va, vb].iter().any(|&v| {
                    cell.generators[v as usize].iter().any(|&p| p < 6)
                })
            {
                continue;
            }
            // Reach cap (Trim only): drop edges whose endpoints balloon
            // past the cell — but an edge is kept if ANY of its cells
            // emits it, so the bound is a union over the (up to three)
            // adjacent cells' local scales.
            if boundary == Boundary::Trim && options.max_reach > 0.0 {
                let reach2 = options.max_reach * options.max_reach * nn_d2;
                let far = |v: u32| -> bool {
                    let p = cell.vertices[v as usize];
                    (0..3).map(|a| (p[a] - site[a]).powi(2)).sum::<f64>() > reach2
                };
                if far(va) || far(vb) {
                    continue;
                }
            }
            let (pa, pb) = (cell.vertices[va as usize], cell.vertices[vb as usize]);
            let len2: f64 = (0..3).map(|a| (pa[a] - pb[a]).powi(2)).sum();
            if len2 < WELD_TOL * WELD_TOL {
                continue; // endpoints would weld anyway
            }
            let (a, b) = (welder.intern(pa), welder.intern(pb));
            if a != b && edge_seen.insert((a.min(b), a.max(b))) {
                edges.push([a, b]);
            }
        }
    }

    let nodes = welder
        .nodes
        .into_iter()
        .map(|p| core::array::from_fn(|a| p[a] * spacing))
        .collect();
    Ok(VoronoiResult {
        skeleton: Skeleton { nodes, edges },
        spacing,
        site_count: kept.len(),
    })
}

/// Visit the bucket indices of the Chebyshev ring at distance `r` around
/// `home` (r = 0 is the home cell itself), clamped to the grid.
fn for_ring_cells(home: [i64; 3], r: i64, dims: &[usize; 3], mut visit: impl FnMut(usize)) {
    let idx_of = |c: [i64; 3]| -> Option<usize> {
        for a in 0..3 {
            if c[a] < 0 || c[a] >= dims[a] as i64 {
                return None;
            }
        }
        Some((c[2] as usize * dims[1] + c[1] as usize) * dims[0] + c[0] as usize)
    };
    if r == 0 {
        if let Some(i) = idx_of(home) {
            visit(i);
        }
        return;
    }
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                    continue;
                }
                if let Some(i) = idx_of([home[0] + dx, home[1] + dy, home[2] + dz]) {
                    visit(i);
                }
            }
        }
    }
}

/// Position-tolerance node interning: nodes within `tol` of an existing
/// node reuse its id. Buckets are `tol`-sized, so checking the 27
/// surrounding buckets covers every candidate within `tol`.
struct NodeWelder {
    tol: f64,
    map: std::collections::HashMap<[i64; 3], Vec<u32>>,
    nodes: Vec<[f64; 3]>,
}

impl NodeWelder {
    fn new(tol: f64) -> Self {
        Self {
            tol,
            map: std::collections::HashMap::new(),
            nodes: Vec::new(),
        }
    }

    fn intern(&mut self, p: [f64; 3]) -> u32 {
        let key: [i64; 3] = core::array::from_fn(|a| (p[a] / self.tol).floor() as i64);
        for dz in -1..=1i64 {
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let k = [key[0] + dx, key[1] + dy, key[2] + dz];
                    if let Some(ids) = self.map.get(&k) {
                        for &id in ids {
                            let q = self.nodes[id as usize];
                            let d2: f64 = (0..3).map(|a| (p[a] - q[a]).powi(2)).sum();
                            if d2 < self.tol * self.tol {
                                return id;
                            }
                        }
                    }
                }
            }
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(p);
        self.map.entry(key).or_default().push(id);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random point in the unit cube (splitmix-style
    /// hash; no RNG dependency, stable across runs).
    fn hash_point(i: u64) -> [f64; 3] {
        let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut out = [0.0; 3];
        for o in &mut out {
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            *o = (x & 0xFFFF_FFFF) as f64 / u32::MAX as f64;
        }
        out
    }

    fn boxed_options() -> VoronoiOptions {
        VoronoiOptions {
            boundary: Boundary::Box,
            ..Default::default()
        }
    }

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
    }

    /// Circumcenter of 4 points (equidistant to all), or None if the
    /// tetrahedron is (near-)degenerate.
    fn circumcenter(p: [[f64; 3]; 4]) -> Option<[f64; 3]> {
        // 2 (p_i - p_0) . x = |p_i|^2 - |p_0|^2, i = 1..3.
        let mut m = [[0.0f64; 3]; 3];
        let mut rhs = [0.0f64; 3];
        let n0: f64 = p[0].iter().map(|c| c * c).sum();
        for i in 0..3 {
            for a in 0..3 {
                m[i][a] = 2.0 * (p[i + 1][a] - p[0][a]);
            }
            rhs[i] = p[i + 1].iter().map(|c| c * c).sum::<f64>() - n0;
        }
        // Cramer's rule.
        let det = |m: &[[f64; 3]; 3]| -> f64 {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        let d = det(&m);
        if d.abs() < 1e-6 {
            return None; // sliver tetrahedron: circumcenter too imprecise
        }
        let mut out = [0.0; 3];
        for a in 0..3 {
            let mut ma = m;
            for i in 0..3 {
                ma[i][a] = rhs[i];
            }
            out[a] = det(&ma) / d;
        }
        Some(out)
    }

    #[test]
    fn matches_brute_force_on_random_sites() {
        let sites: Vec<[f64; 3]> = (0..24).map(|i| hash_point(i + 1)).collect();
        // Box mode: the completeness claim covers every genuine vertex,
        // including sparse-pocket ones Trim's reach cap may drop.
        let result = voronoi_skeleton(&sites, &boxed_options()).unwrap();
        let nodes = &result.skeleton.nodes;
        assert!(!result.skeleton.edges.is_empty());

        // Brute force: every empty-circumsphere 4-subset is a Voronoi
        // vertex; every one inside the bbox must appear as a node.
        let mut expected = 0usize;
        for i in 0..sites.len() {
            for j in i + 1..sites.len() {
                for k in j + 1..sites.len() {
                    for l in k + 1..sites.len() {
                        let Some(c) = circumcenter([sites[i], sites[j], sites[k], sites[l]])
                        else {
                            continue;
                        };
                        let r = dist(c, sites[i]);
                        let empty = sites
                            .iter()
                            .enumerate()
                            .filter(|(m, _)| ![i, j, k, l].contains(m))
                            .all(|(_, s)| dist(c, *s) > r + 1e-9);
                        if !empty {
                            continue;
                        }
                        // Only vertices inside the sites' own bbox: farther
                        // out they may be truncated by the clip box.
                        let in_bbox = (0..3).all(|a| {
                            let (l, h) = (
                                sites.iter().map(|s| s[a]).fold(f64::INFINITY, f64::min),
                                sites.iter().map(|s| s[a]).fold(f64::NEG_INFINITY, f64::max),
                            );
                            (l..h).contains(&c[a])
                        });
                        if !in_bbox {
                            continue;
                        }
                        expected += 1;
                        assert!(
                            nodes.iter().any(|n| dist(*n, c) < 1e-6),
                            "brute-force Voronoi vertex {c:?} missing from output"
                        );
                    }
                }
            }
        }
        assert!(expected > 10, "degenerate test data: {expected} vertices");
    }

    #[test]
    fn edge_points_are_equidistant_to_three_nearest_sites() {
        let sites: Vec<[f64; 3]> = (0..40).map(|i| hash_point(i + 100)).collect();
        let result = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap();
        let skeleton = &result.skeleton;

        // Sample interior edges (both endpoints well inside the cloud's
        // bounds) at the midpoint: the three nearest sites are equally
        // near, the fourth strictly farther.
        let interior = |p: &[f64; 3]| p.iter().all(|&c| (0.15..0.85).contains(&c));
        let mut checked = 0usize;
        for e in &skeleton.edges {
            let (a, b) = (skeleton.nodes[e[0] as usize], skeleton.nodes[e[1] as usize]);
            if !(interior(&a) && interior(&b)) {
                continue;
            }
            let mid: [f64; 3] = core::array::from_fn(|i| 0.5 * (a[i] + b[i]));
            let mut d: Vec<f64> = sites.iter().map(|s| dist(mid, *s)).collect();
            d.sort_by(f64::total_cmp);
            assert!(
                d[2] - d[0] < 1e-9,
                "edge midpoint {mid:?}: nearest three sites not equidistant \
                 ({} vs {})",
                d[0],
                d[2]
            );
            assert!(
                d[3] - d[0] > 1e-9,
                "edge midpoint {mid:?} is a vertex, not an edge point"
            );
            checked += 1;
        }
        assert!(checked > 20, "too few interior edges: {checked}");
    }

    #[test]
    fn bcc_sites_reproduce_the_foam_skeleton() {
        use crate::skeleton::{SkeletonFamily, enumerate_skeleton};
        let jitter = 0.3;

        // The same site set the foam family uses, well beyond the
        // comparison window so interior cells are lattice-genuine.
        let mut sites = Vec::new();
        for i in -3..=7i64 {
            for j in -3..=7i64 {
                for k in -3..=7i64 {
                    for coset in 0..2 {
                        sites.push(crate::foam_site_seeded([i, j, k], coset, jitter, 0));
                    }
                }
            }
        }
        let general = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap().skeleton;
        let foam = enumerate_skeleton(
            SkeletonFamily::Foam { irregularity: jitter },
            [0.0; 3],
            [4.0; 3],
            1.0,
        );

        // Compare edge sets restricted to a window both cover fully.
        let inside = |p: &[f64; 3]| p.iter().all(|&c| (0.5..3.5).contains(&c));
        let edge_set = |sk: &Skeleton| -> Vec<([f64; 3], [f64; 3])> {
            let mut out = Vec::new();
            for e in &sk.edges {
                let (a, b) = (sk.nodes[e[0] as usize], sk.nodes[e[1] as usize]);
                if inside(&a) && inside(&b) {
                    out.push(if (a[0], a[1], a[2]) <= (b[0], b[1], b[2]) {
                        (a, b)
                    } else {
                        (b, a)
                    });
                }
            }
            out
        };
        let foam_edges = edge_set(&foam);
        let general_edges = edge_set(&general);
        assert!(foam_edges.len() > 200, "window too small: {}", foam_edges.len());

        let matches = |from: &[([f64; 3], [f64; 3])], to: &[([f64; 3], [f64; 3])]| {
            from.iter()
                .filter(|(a, b)| {
                    to.iter()
                        .any(|(c, d)| dist(*a, *c) < 1e-6 && dist(*b, *d) < 1e-6)
                })
                .count()
        };
        assert_eq!(
            matches(&foam_edges, &general_edges),
            foam_edges.len(),
            "foam edges missing from the general skeleton"
        );
        assert_eq!(
            matches(&general_edges, &foam_edges),
            general_edges.len(),
            "general skeleton invented edges the foam doesn't have"
        );
    }

    #[test]
    fn perfect_cubic_grid_degenerates_to_the_cubic_lattice() {
        // Every Voronoi vertex of a cubic grid is 8-fold cospherical —
        // the identity-based foam approach would emit nothing here; the
        // position-welded general skeleton must produce the clean cubic
        // strut lattice (nodes at half-integer corners, degree 6,
        // unit axis-aligned edges).
        let mut sites = Vec::new();
        for i in 0..=6i64 {
            for j in 0..=6i64 {
                for k in 0..=6i64 {
                    sites.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        let result = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap();
        let skeleton = &result.skeleton;

        let inside = |p: &[f64; 3]| p.iter().all(|&c| (0.4..5.6).contains(&c));
        let mut degree: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut interior_edges = 0usize;
        for e in &skeleton.edges {
            let (a, b) = (skeleton.nodes[e[0] as usize], skeleton.nodes[e[1] as usize]);
            if !(inside(&a) && inside(&b)) {
                continue;
            }
            interior_edges += 1;
            *degree.entry(e[0]).or_default() += 1;
            *degree.entry(e[1]).or_default() += 1;
            // Unit-length, axis-aligned, at half-integer corners.
            let d: [f64; 3] = core::array::from_fn(|i| (a[i] - b[i]).abs());
            let mut sorted = d;
            sorted.sort_by(f64::total_cmp);
            assert!(
                sorted[0] < 1e-9 && sorted[1] < 1e-9 && (sorted[2] - 1.0).abs() < 1e-9,
                "edge {a:?} -> {b:?} is not a unit axis edge"
            );
            for p in [a, b] {
                for c in p {
                    assert!(
                        ((c - 0.5).round() - (c - 0.5)).abs() < 1e-9,
                        "node {p:?} off the half-integer corners"
                    );
                }
            }
        }
        assert!(interior_edges > 100, "too few edges: {interior_edges}");
        // Fully interior nodes (all six neighbors inside the window)
        // have degree exactly 6.
        let deep = |p: &[f64; 3]| p.iter().all(|&c| (1.4..4.6).contains(&c));
        let mut deep_nodes = 0usize;
        for (node, d) in &degree {
            if deep(&skeleton.nodes[*node as usize]) {
                assert_eq!(*d, 6, "node {:?} degree {d}", skeleton.nodes[*node as usize]);
                deep_nodes += 1;
            }
        }
        assert!(deep_nodes > 8, "too few deep nodes: {deep_nodes}");
    }

    #[test]
    fn trim_drops_exactly_the_unsupported_hull_edges() {
        let sites: Vec<[f64; 3]> = (0..40).map(|i| hash_point(i + 300)).collect();
        let trim_result = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap();
        let (trim, spacing) = (trim_result.skeleton, trim_result.spacing);
        let boxed = voronoi_skeleton(&sites, &boxed_options()).unwrap().skeleton;

        // Every trimmed node is a genuine Voronoi vertex: equidistant to
        // its four nearest sites (the empty sphere is implied by
        // "nearest"). Box mode keeps truncation endpoints that fail this.
        let vertex_spread = |node: &[f64; 3]| -> f64 {
            let mut d: Vec<f64> = sites.iter().map(|s| dist(*node, *s)).collect();
            d.sort_by(f64::total_cmp);
            d[3] - d[0]
        };
        for node in &trim.nodes {
            assert!(
                vertex_spread(node) < 1e-9,
                "trim kept non-vertex {node:?} (spread {})",
                vertex_spread(node)
            );
        }
        let truncated = boxed
            .nodes
            .iter()
            .filter(|node| vertex_spread(node) > 1e-6)
            .count();
        assert!(truncated > 0, "box mode should keep truncated endpoints");

        // Trim only ever drops: every trimmed edge exists in box mode.
        let canonical = |sk: &Skeleton, e: &[u32; 2]| -> ([f64; 3], [f64; 3]) {
            let (a, b) = (sk.nodes[e[0] as usize], sk.nodes[e[1] as usize]);
            if (a[0], a[1], a[2]) <= (b[0], b[1], b[2]) {
                (a, b)
            } else {
                (b, a)
            }
        };
        assert!(!trim.edges.is_empty());
        assert!(trim.edges.len() < boxed.edges.len());
        for e in &trim.edges {
            let (a, b) = canonical(&trim, e);
            assert!(
                boxed.edges.iter().any(|f| {
                    let (c, d) = canonical(&boxed, f);
                    dist(a, c) < 1e-9 && dist(b, d) < 1e-9
                }),
                "trim invented edge {a:?} -> {b:?}"
            );
        }

        // The exact boundary claim: box mode has truncation nodes ON the
        // clip box (bbox +- two spacings, the auto padding); trim has
        // none. (Genuine vertices may legitimately sit far outside a
        // scattered cloud — hull-sliver circumcenters — so "stays near
        // the cloud" is only a property of *filled* clouds, not a rule.)
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for s in &sites {
            for a in 0..3 {
                lo[a] = lo[a].min(s[a]);
                hi[a] = hi[a].max(s[a]);
            }
        }
        let on_box = |p: &[f64; 3]| {
            (0..3).any(|a| {
                (p[a] - (lo[a] - 2.0 * spacing)).abs() < 1e-6
                    || (p[a] - (hi[a] + 2.0 * spacing)).abs() < 1e-6
            })
        };
        assert!(boxed.nodes.iter().any(on_box), "box mode should reach the box");
        assert!(!trim.nodes.iter().any(on_box), "trim kept a box node");
    }

    #[test]
    fn deterministic_and_dedup_tolerant() {
        let mut sites: Vec<[f64; 3]> = (0..30).map(|i| hash_point(i + 7)).collect();
        let a = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap();
        let b = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap();
        assert_eq!(a.skeleton.nodes, b.skeleton.nodes);
        assert_eq!(a.skeleton.edges, b.skeleton.edges);

        // Exact duplicates collapse and change nothing.
        sites.push(sites[3]);
        sites.push(sites[17]);
        let c = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap();
        assert_eq!(c.site_count, 30);
        assert_eq!(a.skeleton.nodes, c.skeleton.nodes);
        assert_eq!(a.skeleton.edges, c.skeleton.edges);
    }

    #[test]
    fn tiny_and_degenerate_inputs_do_not_panic() {
        // 1 or 2 sites: no all-bisector edges survive -> empty skeleton.
        for n in 1..=2 {
            let sites: Vec<[f64; 3]> = (0..n).map(|i| hash_point(i + 50)).collect();
            let r = voronoi_skeleton(&sites, &boxed_options()).unwrap();
            assert!(
                r.skeleton.edges.is_empty(),
                "{n} sites produced {} edges",
                r.skeleton.edges.len()
            );
        }
        // 3 generic sites: the tri-junction line, box-truncated on both
        // ends — so Box keeps it and Trim drops it.
        let sites: Vec<[f64; 3]> = (0..3).map(|i| hash_point(i + 50)).collect();
        let r = voronoi_skeleton(&sites, &boxed_options()).unwrap();
        assert_eq!(r.skeleton.edges.len(), 1);
        let r = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap();
        assert!(r.skeleton.edges.is_empty());
        // 4 sites in general position: one vertex, four truncated rays —
        // all unsupported on the far side, so Trim keeps nothing.
        let sites: Vec<[f64; 3]> = (0..4).map(|i| hash_point(i + 60)).collect();
        let r = voronoi_skeleton(&sites, &boxed_options()).unwrap();
        assert_eq!(r.skeleton.edges.len(), 4);
        let r = voronoi_skeleton(&sites, &VoronoiOptions::default()).unwrap();
        assert!(r.skeleton.edges.is_empty());

        // Collinear and coplanar clouds have no finite Voronoi vertices;
        // everything truncates at the box (all-bisector rule may keep
        // face-to-face edges) — just require no panic and validity.
        let line: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
        voronoi_skeleton(&line, &VoronoiOptions::default()).unwrap();
        let plane: Vec<[f64; 3]> = (0..25)
            .map(|i| [(i % 5) as f64, (i / 5) as f64, 0.0])
            .collect();
        voronoi_skeleton(&plane, &VoronoiOptions::default()).unwrap();

        // All-coincident collapses to one site.
        let same = vec![[1.0, 1.0, 1.0]; 5];
        let r = voronoi_skeleton(&same, &VoronoiOptions::default()).unwrap();
        assert_eq!(r.site_count, 1);
        assert!(r.skeleton.edges.is_empty());

        assert!(voronoi_skeleton(&[], &VoronoiOptions::default()).is_err());
        assert!(voronoi_skeleton(&[[f64::NAN, 0.0, 0.0]], &VoronoiOptions::default()).is_err());
    }
}
