//! Implicit lattice structures modulated by a local density value.
//!
//! Shared by `lattice_model_template` (compiled to wasm32 and merged into
//! density models by `lattice_operator`) and native unit tests. New lattice
//! families are plain Rust additions here: an implicit occupancy function
//! of position, cell size, and the local relative density.
//!
//! # Density mapping contract
//!
//! Every lattice maps the local relative density `d` (clamped to `[0, 1]`,
//! non-finite reads as 0) onto its thickness parameter so that:
//!
//! - `d <= 0` is empty and `d >= 1` is fully solid,
//! - occupancy is monotone in `d` (raising density never removes material).
//!
//! Between the endpoints the mapping is *approximate* — the realized volume
//! fraction tracks `d` monotonically but is not calibrated to equal it.
//! Calibrating per-family `d -> parameter` curves (so printed foam density
//! matches the FEA-requested density) is expected follow-up work.

pub mod skeleton;

/// The lattice families the operator offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatticeKind {
    /// Gyroid TPMS sheet: `|sin X cos Y + sin Y cos Z + sin Z cos X| <= 1.5 d`.
    /// Smooth, self-supporting, near-linear density response.
    Gyroid = 0,
    /// Schwarz-P TPMS sheet: `|cos X + cos Y + cos Z| <= 3 d`. More open,
    /// axis-aligned tunnels.
    SchwarzP = 1,
    /// Cubic strut lattice: material within radius `r(d)` of the cell-edge
    /// skeleton — the vendor-style "fixed pattern, modulated strut
    /// diameter" structure.
    Struts = 2,
    /// Hexagonal honeycomb: vertical cell walls (prisms along z, the
    /// compression axis), walls on the Voronoi boundaries of a triangular
    /// lattice with site spacing `cell_size`. Compression is carried by
    /// wall buckling (foam-like plateau) and the closed cells decouple
    /// surface shear — the classic seat-cushion honeycomb. Thin-wall
    /// volume fraction tracks `d` exactly.
    Honeycomb = 3,
    /// Tetrahedral strut lattice: uniform-radius struts along the bonds
    /// of the diamond lattice — every node joins four struts in
    /// tetrahedral directions. Connected at any positive density, so it
    /// supports very sparse fills (a few percent); thin-strut volume
    /// fraction tracks `d` exactly. The vendor-style tetrahedral tiling.
    Tetra = 4,
    /// Foam lattice: struts along the edge skeleton of the Voronoi
    /// diagram of a jittered BCC point set — polyhedral cells with
    /// pentagon/hexagon faces, three faces meeting at every strut and
    /// four struts at every vertex (Plateau geometry, like a dried real
    /// foam). `LatticeParams::irregularity` 0 gives the periodic Kelvin
    /// cell (truncated octahedra); higher values give organic,
    /// non-repeating cells.
    Foam = 5,
}

impl LatticeKind {
    /// Decode the config-slot encoding (see `lattice_model_template`).
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Gyroid),
            1 => Some(Self::SchwarzP),
            2 => Some(Self::Struts),
            3 => Some(Self::Honeycomb),
            4 => Some(Self::Tetra),
            5 => Some(Self::Foam),
            _ => None,
        }
    }
}

/// Per-family shape parameters beyond the density (all families ignore
/// the ones they don't use).
#[derive(Clone, Copy, Debug)]
pub struct LatticeParams {
    /// Foam site jitter: 0 keeps the periodic Kelvin cells, 1 fully
    /// randomizes cell shapes. Clamped to `[0, 1]`; non-finite reads 0.
    pub irregularity: f64,
}

impl Default for LatticeParams {
    fn default() -> Self {
        Self { irregularity: 0.3 }
    }
}

/// How the local relative density transforms before the family mapping —
/// the calibration knobs between the FEA-requested density and the printed
/// structure's thickness parameter:
///
/// `d_eff = clamp(min + (max - min) * d^gamma, 0, 1)`
///
/// - `gamma` bends the response curve (>1 thins the mid-range, <1 fattens
///   it) without moving the endpoints,
/// - `min` sets a thickness floor so zero-density regions still print
///   structure,
/// - `max` caps the thickness so peak-density regions stay porous.
///
/// Non-finite or non-positive `gamma` reads as 1; the all-zero (unpatched
/// config slot) map yields 0 everywhere — empty, never a misleading solid.
#[derive(Clone, Copy, Debug)]
pub struct DensityMap {
    pub gamma: f32,
    pub min: f32,
    pub max: f32,
}

impl Default for DensityMap {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            min: 0.0,
            max: 1.0,
        }
    }
}

/// Apply a [`DensityMap`] to a raw density sample.
pub fn map_density(map: DensityMap, density: f32) -> f32 {
    let d = if density.is_finite() {
        density.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let gamma = if map.gamma.is_finite() && map.gamma > 0.0 {
        map.gamma
    } else {
        1.0
    };
    let min = if map.min.is_finite() { map.min } else { 0.0 };
    let max = if map.max.is_finite() { map.max } else { 1.0 };
    (min + (max - min) * d.powf(gamma)).clamp(0.0, 1.0)
}

const TAU: f64 = core::f64::consts::TAU;

/// Whether the lattice contains `pos`, given the local relative density.
/// `cell_size` is the pattern period in model units; non-positive cell
/// sizes read as empty (an unpatched or corrupt config must not produce a
/// misleading solid).
pub fn lattice_occupied(
    kind: LatticeKind,
    pos: [f64; 3],
    cell_size: f64,
    density: f32,
    params: &LatticeParams,
) -> bool {
    let d = if density.is_finite() {
        f64::from(density).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if d <= 0.0 || cell_size <= 0.0 || !cell_size.is_finite() {
        return false;
    }
    if d >= 1.0 {
        return true;
    }
    let [x, y, z] = pos.map(|v| v * TAU / cell_size);
    match kind {
        LatticeKind::Gyroid => {
            // g ranges over [-1.5, 1.5]; the sheet |g| <= t fills the cell
            // as t -> 1.5.
            let g = x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos();
            g.abs() <= 1.5 * d
        }
        LatticeKind::SchwarzP => {
            // s ranges over [-3, 3].
            let s = x.cos() + y.cos() + z.cos();
            s.abs() <= 3.0 * d
        }
        LatticeKind::Struts => {
            // Distance (in cell units) from the position to the nearest
            // edge of the cubic cell skeleton: for each axis family, the
            // lateral distance to the nearest integer lattice line pair.
            let u = frac_dist(x / TAU);
            let v = frac_dist(y / TAU);
            let w = frac_dist(z / TAU);
            let to_x_edges = (v * v + w * w).sqrt(); // edges along x
            let to_y_edges = (u * u + w * w).sqrt();
            let to_z_edges = (u * u + v * v).sqrt();
            let dist = to_x_edges.min(to_y_edges).min(to_z_edges);
            dist <= strut_radius(d)
        }
        LatticeKind::Honeycomb => {
            // Vertical walls: z never enters. Distance (in cell units,
            // where 1 = triangular-lattice site spacing) from the xy
            // point to the nearest hexagonal cell wall.
            hex_wall_distance(x / TAU, y / TAU) <= honeycomb_half_thickness(d)
        }
        LatticeKind::Tetra => {
            // Uniform-radius struts along the bonds of the diamond
            // lattice: every node joins exactly four struts in tetrahedral
            // directions, and the network is connected at any positive
            // radius — thin struts everywhere, down to ~1% fill.
            let p = [x / TAU, y / TAU, z / TAU];
            diamond_strut_distance(p) <= diamond_strut_radius(d)
        }
        LatticeKind::Foam => {
            // Struts live where three Voronoi cells (nearly) meet: both
            // the second and third site distances are within the
            // threshold of the first. On a face (two cells) only d2
            // closes in, so faces stay open; along the cell edges d2 and
            // d3 both approach d1 and the region thickens into a strut.
            let p = [x / TAU, y / TAU, z / TAU];
            let jitter = if params.irregularity.is_finite() {
                params.irregularity.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let [d1, d2, d3] = foam_nearest_three(p, jitter);
            let t = foam_threshold(d);
            d2 - d1 <= t && d3 - d1 <= t
        }
    }
}

/// Deterministic integer hash (splitmix64-style finalizer) for foam site
/// jitter: same cell always jitters the same way, with no runtime RNG.
fn foam_hash(x: i64, y: i64, z: i64, coset: u64) -> u64 {
    let mut h = (x as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((z as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
        .wrapping_add(coset.wrapping_mul(0x94D0_49BB_1331_11EB));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}

/// The jittered BCC site for a cell: base + coset offset + a hash-derived
/// displacement of up to `0.25 * jitter` per axis (sites stay separated
/// even at full irregularity; the BCC nearest-neighbor gap is sqrt(3)/2).
fn foam_site(base: [i64; 3], coset: usize, jitter: f64) -> [f64; 3] {
    let coset_offset = if coset == 0 { 0.0 } else { 0.5 };
    let h = foam_hash(base[0], base[1], base[2], coset as u64);
    let amp = 0.25 * jitter;
    let unit = |bits: u64| -> f64 { (bits & 0x1F_FFFF) as f64 / 0xF_FFFF as f64 - 1.0 };
    [
        base[0] as f64 + coset_offset + amp * unit(h),
        base[1] as f64 + coset_offset + amp * unit(h >> 21),
        base[2] as f64 + coset_offset + amp * unit(h >> 42),
    ]
}

/// Distances to the three nearest jittered-BCC sites (each coset searched
/// in the 3x3x3 cell neighborhood around the position — ample for the
/// third-nearest site anywhere the strut test can pass).
fn foam_nearest_three(p: [f64; 3], jitter: f64) -> [f64; 3] {
    let mut best = [f64::MAX; 3];
    for coset in 0..2 {
        let offset = if coset == 0 { 0.0 } else { 0.5 };
        let center = [
            (p[0] - offset).round() as i64,
            (p[1] - offset).round() as i64,
            (p[2] - offset).round() as i64,
        ];
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let base = [center[0] + dx, center[1] + dy, center[2] + dz];
                    let s = foam_site(base, coset, jitter);
                    let e = [p[0] - s[0], p[1] - s[1], p[2] - s[2]];
                    let d2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
                    if d2 < best[2] {
                        best[2] = d2;
                        if best[2] < best[1] {
                            best.swap(1, 2);
                            if best[1] < best[0] {
                                best.swap(0, 1);
                            }
                        }
                    }
                }
            }
        }
    }
    best.map(f64::sqrt)
}

/// Foam strut threshold for a target relative density: the numerically
/// measured sparse-range fit `VF = 5.39 * t^1.8` inverted (fitted at the
/// default irregularity; realized VF tracks `d` within ~1% for d <= 0.2
/// and ~10% to d = 0.4, drifting mildly with irregularity), blended
/// linearly to a threshold past every point's largest gap once struts
/// merge.
fn foam_threshold(d: f64) -> f64 {
    const BLEND_START: f64 = 0.6;
    const FIT_COEFF: f64 = 5.39;
    const FIT_EXPONENT: f64 = 1.8;
    let thin = |d: f64| (d / FIT_COEFF).powf(1.0 / FIT_EXPONENT);
    if d <= BLEND_START {
        thin(d)
    } else {
        let t = (d - BLEND_START) / (1.0 - BLEND_START);
        thin(BLEND_START) + t * (1.6 - thin(BLEND_START))
    }
}

/// The four diamond bond offsets from an A-sublattice atom to its B
/// neighbors (quarter-cell tetrahedral directions, length sqrt(3)/4).
const DIAMOND_BONDS: [[f64; 3]; 4] = [
    [0.25, 0.25, 0.25],
    [-0.25, -0.25, 0.25],
    [-0.25, 0.25, -0.25],
    [0.25, -0.25, -0.25],
];

/// The FCC coset offsets within the unit cell (the A sublattice; the B
/// sublattice adds (1/4, 1/4, 1/4)).
const FCC_BASIS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
];

/// Nearest point of the lattice `Z^3 + offset` to `p` (exact: each
/// coordinate rounds independently).
fn nearest_coset_point(p: [f64; 3], offset: [f64; 3]) -> ([f64; 3], f64) {
    let mut site = [0.0; 3];
    let mut d2 = 0.0;
    for axis in 0..3 {
        let q = p[axis] - offset[axis];
        let n = q.round();
        site[axis] = offset[axis] + n;
        let e = q - n;
        d2 += e * e;
    }
    (site, d2)
}

/// Nearest FCC atom to `p`, with the sublattice shift `shift` added to
/// every coset offset (0 for the A sublattice, 1/4 for B).
fn nearest_fcc_atom(p: [f64; 3], shift: f64) -> [f64; 3] {
    let mut best = [0.0; 3];
    let mut best_d2 = f64::MAX;
    for basis in FCC_BASIS {
        let offset = [basis[0] + shift, basis[1] + shift, basis[2] + shift];
        let (site, d2) = nearest_coset_point(p, offset);
        if d2 < best_d2 {
            best_d2 = d2;
            best = site;
        }
    }
    best
}

/// Distance from `p` to the segment `[a, a + t]`.
fn segment_distance(p: [f64; 3], a: [f64; 3], t: [f64; 3]) -> f64 {
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    let s = ((ap[0] * t[0] + ap[1] * t[1] + ap[2] * t[2]) / tt).clamp(0.0, 1.0);
    let e = [ap[0] - s * t[0], ap[1] - s * t[1], ap[2] - s * t[2]];
    (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
}

/// Distance (in cell units) from `p` to the diamond-lattice bond network.
///
/// Any point within reach of a strut has that strut's endpoints as its
/// nearest A- and B-sublattice atoms (the next atom of either sublattice
/// is at least sqrt(2)/2 away, versus sqrt(3)/8 to a bond midpoint from
/// its endpoint), so checking the four bonds of each nearest atom is
/// exact wherever the answer can be within any usable strut radius.
fn diamond_strut_distance(p: [f64; 3]) -> f64 {
    let a = nearest_fcc_atom(p, 0.0);
    let b = nearest_fcc_atom(p, 0.25);
    let mut dist = f64::MAX;
    for t in DIAMOND_BONDS {
        dist = dist.min(segment_distance(p, a, t));
        dist = dist.min(segment_distance(p, b, [-t[0], -t[1], -t[2]]));
    }
    dist
}

/// Diamond strut radius (in cell units) for a target relative density:
/// the analytic thin-strut inverse (16 bonds of length sqrt(3)/4 per unit
/// cell, fraction = 4 sqrt(3) pi r^2), blended linearly to 1/2 — beyond
/// every point's distance to the network — once struts start merging.
fn diamond_strut_radius(d: f64) -> f64 {
    const BLEND_START: f64 = 0.6;
    // 4 * sqrt(3) * pi
    const THIN_COEFF: f64 = 21.765_659_582_517_43;
    let thin = |d: f64| (d / THIN_COEFF).sqrt();
    if d <= BLEND_START {
        thin(d)
    } else {
        let t = (d - BLEND_START) / (1.0 - BLEND_START);
        thin(BLEND_START) + t * (0.5 - thin(BLEND_START))
    }
}

/// Distance from an xy point (in site-spacing units) to the nearest wall
/// of the hexagonal tiling whose cells are the Voronoi regions of the unit
/// triangular lattice.
///
/// Exact: find the nearest lattice site, then take the minimum distance to
/// the perpendicular bisectors against its six neighbors (the hexagon's
/// six walls).
fn hex_wall_distance(x: f64, y: f64) -> f64 {
    // Triangular lattice basis: a1 = (1, 0), a2 = (1/2, sqrt(3)/2).
    const ROW_HEIGHT: f64 = 0.866_025_403_784_438_6; // sqrt(3)/2
    let fj = y / ROW_HEIGHT;
    let fi = x - 0.5 * fj;

    // Nearest site: search the skewed-coordinate neighborhood.
    let i0 = fi.floor() as i64;
    let j0 = fj.floor() as i64;
    let mut nearest = (0.0, 0.0);
    let mut nearest_d2 = f64::MAX;
    for j in (j0 - 1)..=(j0 + 2) {
        for i in (i0 - 1)..=(i0 + 2) {
            let sx = i as f64 + 0.5 * j as f64;
            let sy = j as f64 * ROW_HEIGHT;
            let d2 = (x - sx) * (x - sx) + (y - sy) * (y - sy);
            if d2 < nearest_d2 {
                nearest_d2 = d2;
                nearest = (sx, sy);
            }
        }
    }

    // Distance to the Voronoi boundary: min over the six unit-distance
    // neighbors b of the signed distance to the bisector plane,
    // (|p - b|^2 - |p - a|^2) / (2 |b - a|) with |b - a| = 1.
    const NEIGHBORS: [(f64, f64); 6] = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, ROW_HEIGHT),
        (-0.5, ROW_HEIGHT),
        (0.5, -ROW_HEIGHT),
        (-0.5, -ROW_HEIGHT),
    ];
    let mut wall = f64::MAX;
    for (nx, ny) in NEIGHBORS {
        let bx = nearest.0 + nx;
        let by = nearest.1 + ny;
        let d2 = (x - bx) * (x - bx) + (y - by) * (y - by);
        wall = wall.min((d2 - nearest_d2) * 0.5);
    }
    wall
}

/// Honeycomb wall half-thickness (in site-spacing units) for a target
/// relative density: thin-wall volume fraction is exactly `4 * t_half`
/// (wall length 2 per unit area at unit spacing), blended linearly to the
/// hexagon inradius 1/2 — where the cell is fully solid — once walls
/// start merging.
fn honeycomb_half_thickness(d: f64) -> f64 {
    const BLEND_START: f64 = 0.6;
    let thin = |d: f64| d * 0.25;
    if d <= BLEND_START {
        thin(d)
    } else {
        let t = (d - BLEND_START) / (1.0 - BLEND_START);
        thin(BLEND_START) + t * (0.5 - thin(BLEND_START))
    }
}

/// Strut radius (in cell units) for a target relative density: the
/// analytic inverse of the thin-strut volume fraction (three edge families
/// per cell, fraction ≈ 3π r²) while struts stay thin, blended linearly to
/// `sqrt(2)/2` — the body-center distance, where the cell is fully solid —
/// once they start merging.
fn strut_radius(d: f64) -> f64 {
    const BLEND_START: f64 = 0.6;
    let thin = |d: f64| (d / (3.0 * core::f64::consts::PI)).sqrt();
    if d <= BLEND_START {
        thin(d)
    } else {
        let t = (d - BLEND_START) / (1.0 - BLEND_START);
        thin(BLEND_START) + t * (core::f64::consts::FRAC_1_SQRT_2 - thin(BLEND_START))
    }
}

/// Distance from `u` to the nearest integer, in the same units.
fn frac_dist(u: f64) -> f64 {
    let f = u - u.floor();
    f.min(1.0 - f)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [LatticeKind; 6] = [
        LatticeKind::Gyroid,
        LatticeKind::SchwarzP,
        LatticeKind::Struts,
        LatticeKind::Honeycomb,
        LatticeKind::Tetra,
        LatticeKind::Foam,
    ];

    /// A grid of probe points spanning several cells at odd offsets.
    fn probes() -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                for k in 0..7 {
                    out.push([
                        0.013 + 0.37 * i as f64,
                        -0.71 + 0.29 * j as f64,
                        0.19 + 0.41 * k as f64,
                    ]);
                }
            }
        }
        out
    }

    #[test]
    fn density_endpoints_are_empty_and_solid() {
        for kind in KINDS {
            for pos in probes() {
                assert!(
                    !lattice_occupied(kind, pos, 0.25, 0.0, &LatticeParams::default()),
                    "{kind:?} at {pos:?} should be empty at density 0"
                );
                assert!(
                    lattice_occupied(kind, pos, 0.25, 1.0, &LatticeParams::default()),
                    "{kind:?} at {pos:?} should be solid at density 1"
                );
                assert!(
                    !lattice_occupied(kind, pos, 0.25, f32::NAN, &LatticeParams::default()),
                    "{kind:?} at {pos:?} should treat NaN density as empty"
                );
            }
        }
    }

    #[test]
    fn occupancy_is_monotone_in_density() {
        let densities = [0.1, 0.25, 0.4, 0.6, 0.8, 0.95];
        for kind in KINDS {
            for pos in probes() {
                let mut was_occupied = false;
                for d in densities {
                    let occupied = lattice_occupied(kind, pos, 0.25, d, &LatticeParams::default());
                    assert!(
                        occupied || !was_occupied,
                        "{kind:?} at {pos:?} lost material raising density to {d}"
                    );
                    was_occupied = occupied;
                }
            }
        }
    }

    #[test]
    fn volume_fraction_grows_with_density_and_stays_sane() {
        for kind in KINDS {
            let fraction = |d: f32| -> f64 {
                let probes = probes();
                let hits = probes
                    .iter()
                    .filter(|&&p| lattice_occupied(kind, p, 0.25, d, &LatticeParams::default()))
                    .count();
                hits as f64 / probes.len() as f64
            };
            let low = fraction(0.2);
            let high = fraction(0.7);
            assert!(
                low > 0.02 && low < 0.75,
                "{kind:?} fraction at d=0.2 out of range: {low}"
            );
            assert!(
                high > low,
                "{kind:?} fraction should grow with density ({low} -> {high})"
            );
            assert!(high < 1.0, "{kind:?} should not be solid at d=0.7: {high}");
        }
    }

    #[test]
    fn struts_cover_the_cell_edges() {
        // Points on the cell-edge skeleton are occupied at modest density.
        for t in [0.05, 0.3, 0.77] {
            assert!(lattice_occupied(
                LatticeKind::Struts,
                [t, 0.0, 0.0],
                0.25,
                0.2,
                &LatticeParams::default()
            ));
            // (0.5, 0.25) is a lattice corner in cell units (2, 1): on a
            // z-parallel strut for every z.
            assert!(lattice_occupied(
                LatticeKind::Struts,
                [0.5, 0.25, t],
                0.25,
                0.2,
                &LatticeParams::default()
            ));
        }
        // The body center is the last point to fill.
        assert!(!lattice_occupied(
            LatticeKind::Struts,
            [0.125, 0.125, 0.125],
            0.25,
            0.5,
            &LatticeParams::default()
        ));
        assert!(lattice_occupied(
            LatticeKind::Struts,
            [0.125, 0.125, 0.125],
            0.25,
            1.0,
            &LatticeParams::default()
        ));
    }

    #[test]
    fn degenerate_cell_sizes_read_empty() {
        for kind in KINDS {
            assert!(!lattice_occupied(
                kind,
                [0.1; 3],
                0.0,
                0.8,
                &LatticeParams::default()
            ));
            assert!(!lattice_occupied(
                kind,
                [0.1; 3],
                -1.0,
                0.8,
                &LatticeParams::default()
            ));
            assert!(!lattice_occupied(
                kind,
                [0.1; 3],
                f64::NAN,
                0.8,
                &LatticeParams::default()
            ));
        }
    }

    #[test]
    fn honeycomb_walls_are_vertical_prisms() {
        // Occupancy is independent of z: probing the same xy at many
        // heights always agrees.
        for pos in probes() {
            let base = lattice_occupied(
                LatticeKind::Honeycomb,
                pos,
                0.25,
                0.35,
                &LatticeParams::default(),
            );
            for dz in [-3.7, -0.2, 1.9, 12.3] {
                let shifted = [pos[0], pos[1], pos[2] + dz];
                assert_eq!(
                    lattice_occupied(
                        LatticeKind::Honeycomb,
                        shifted,
                        0.25,
                        0.35,
                        &LatticeParams::default()
                    ),
                    base,
                    "honeycomb occupancy must not depend on z ({pos:?} + {dz})"
                );
            }
        }
    }

    #[test]
    fn honeycomb_geometry_walls_and_cell_centers() {
        // A lattice site (hex cell center) is the farthest point from the
        // walls: empty at any thin-wall density, filled last as d -> 1
        // (the exact center reaches the wall threshold only at d = 1).
        let center = [0.0, 0.0, 0.1];
        assert!(!lattice_occupied(
            LatticeKind::Honeycomb,
            center,
            1.0,
            0.5,
            &LatticeParams::default()
        ));
        assert!(lattice_occupied(
            LatticeKind::Honeycomb,
            center,
            1.0,
            1.0,
            &LatticeParams::default()
        ));
        let near_center = [0.06, 0.03, 0.1];
        assert!(!lattice_occupied(
            LatticeKind::Honeycomb,
            near_center,
            1.0,
            0.5,
            &LatticeParams::default()
        ));
        assert!(lattice_occupied(
            LatticeKind::Honeycomb,
            near_center,
            1.0,
            0.999,
            &LatticeParams::default()
        ));
        // The midpoint between two adjacent sites lies on a wall: occupied
        // at any positive density.
        let wall = [0.5, 0.0, -2.0];
        assert!(lattice_occupied(
            LatticeKind::Honeycomb,
            wall,
            1.0,
            0.05,
            &LatticeParams::default()
        ));
        // Thin-wall volume fraction tracks the density closely (exact up
        // to junction overlap).
        let fraction = |d: f32| -> f64 {
            let mut hits = 0usize;
            let mut total = 0usize;
            for i in 0..60 {
                for j in 0..60 {
                    let pos = [0.013 + i as f64 * 0.061, -0.71 + j as f64 * 0.053, 0.0];
                    total += 1;
                    if lattice_occupied(
                        LatticeKind::Honeycomb,
                        pos,
                        1.0,
                        d,
                        &LatticeParams::default(),
                    ) {
                        hits += 1;
                    }
                }
            }
            hits as f64 / total as f64
        };
        for d in [0.2, 0.4] {
            let f = fraction(d);
            assert!(
                (f - f64::from(d)).abs() < 0.05,
                "honeycomb VF at d={d} should be ~{d}, got {f}"
            );
        }
    }

    #[test]
    fn tetra_geometry_bonds_and_voids() {
        // The bond midpoint between the atoms at (0,0,0) and
        // (1/4,1/4,1/4) is on a strut: occupied at very sparse density.
        let on_bond = [0.125, 0.125, 0.125];
        assert!(lattice_occupied(
            LatticeKind::Tetra,
            on_bond,
            1.0,
            0.02,
            &LatticeParams::default()
        ));
        // Atoms (strut nodes) are occupied too.
        assert!(lattice_occupied(
            LatticeKind::Tetra,
            [0.0; 3],
            1.0,
            0.02,
            &LatticeParams::default()
        ));
        // The tetrahedral void center (1/2,1/2,1/2) sits sqrt(3)/4 from
        // the nearest atom: empty at mid density, filled near solid.
        let void = [0.5, 0.5, 0.5];
        assert!(!lattice_occupied(
            LatticeKind::Tetra,
            void,
            1.0,
            0.5,
            &LatticeParams::default()
        ));
        assert!(lattice_occupied(
            LatticeKind::Tetra,
            void,
            1.0,
            0.999,
            &LatticeParams::default()
        ));
    }

    #[test]
    fn tetra_thin_strut_fraction_tracks_density() {
        // The analytic thin-strut inverse keeps the realized volume
        // fraction close to d across the sparse range — including the
        // few-percent fills seat cushions actually print.
        let fraction = |d: f32| -> f64 {
            let mut hits = 0usize;
            let mut total = 0usize;
            for i in 0..17 {
                for j in 0..17 {
                    for k in 0..17 {
                        let pos = [
                            0.013 + i as f64 * 0.0603,
                            -0.71 + j as f64 * 0.0567,
                            0.19 + k as f64 * 0.0611,
                        ];
                        total += 1;
                        if lattice_occupied(
                            LatticeKind::Tetra,
                            pos,
                            1.0,
                            d,
                            &LatticeParams::default(),
                        ) {
                            hits += 1;
                        }
                    }
                }
            }
            hits as f64 / total as f64
        };
        for d in [0.02f32, 0.1, 0.3] {
            let f = fraction(d);
            assert!(
                (f - f64::from(d)).abs() < 0.02 + f64::from(d) * 0.25,
                "tetra VF at d={d} should be ~{d}, got {f}"
            );
        }
    }

    #[test]
    fn map_density_default_is_identity() {
        let map = DensityMap::default();
        for d in [0.0, 0.2, 0.5, 0.8, 1.0] {
            assert!((map_density(map, d) - d).abs() < 1e-6);
        }
        assert_eq!(map_density(map, f32::NAN), 0.0);
        assert_eq!(map_density(map, -3.0), 0.0);
        assert_eq!(map_density(map, 7.0), 1.0);
    }

    #[test]
    fn map_density_gamma_bends_without_moving_endpoints() {
        let steep = DensityMap {
            gamma: 2.0,
            ..Default::default()
        };
        let shallow = DensityMap {
            gamma: 0.5,
            ..Default::default()
        };
        assert_eq!(map_density(steep, 0.0), 0.0);
        assert_eq!(map_density(steep, 1.0), 1.0);
        assert!((map_density(steep, 0.5) - 0.25).abs() < 1e-6);
        assert!((map_density(shallow, 0.25) - 0.5).abs() < 1e-6);
        // Degenerate gamma reads as identity.
        for bad in [0.0, -2.0, f32::NAN, f32::INFINITY] {
            let map = DensityMap {
                gamma: bad,
                ..Default::default()
            };
            assert!((map_density(map, 0.3) - 0.3).abs() < 1e-6, "gamma {bad}");
        }
    }

    #[test]
    fn map_density_min_max_remap_the_range() {
        let map = DensityMap {
            gamma: 1.0,
            min: 0.2,
            max: 0.8,
        };
        assert!((map_density(map, 0.0) - 0.2).abs() < 1e-6);
        assert!((map_density(map, 0.5) - 0.5).abs() < 1e-6);
        assert!((map_density(map, 1.0) - 0.8).abs() < 1e-6);
        // The all-zero (unpatched config slot) map yields empty.
        let zeroed = DensityMap {
            gamma: 0.0,
            min: 0.0,
            max: 0.0,
        };
        assert_eq!(map_density(zeroed, 0.9), 0.0);
    }
    #[test]
    fn foam_volume_fraction_tracks_density() {
        // The fitted threshold keeps realized volume fraction close to d
        // across the sparse range, at any irregularity.
        for irr in [0.0f64, 0.3, 0.7] {
            let params = LatticeParams { irregularity: irr };
            for d in [0.05f32, 0.1, 0.2] {
                let mut hits = 0usize;
                let mut total = 0usize;
                for i in 0..24 {
                    for j in 0..24 {
                        for k in 0..24 {
                            let pos = [
                                0.013 + i as f64 * 0.1013,
                                -0.71 + j as f64 * 0.0997,
                                0.19 + k as f64 * 0.1051,
                            ];
                            total += 1;
                            if lattice_occupied(LatticeKind::Foam, pos, 1.0, d, &params) {
                                hits += 1;
                            }
                        }
                    }
                }
                let f = hits as f64 / total as f64;
                assert!(
                    (f - f64::from(d)).abs() < 0.015 + f64::from(d) * 0.2,
                    "foam VF at irr={irr} d={d} should be ~{d}, got {f}"
                );
            }
        }
    }

    #[test]
    fn foam_zero_irregularity_is_periodic_kelvin() {
        // Without jitter the sites are plain BCC, so occupancy repeats
        // with the unit cell in every direction.
        let params = LatticeParams { irregularity: 0.0 };
        for pos in probes() {
            let base = lattice_occupied(LatticeKind::Foam, pos, 0.25, 0.25, &params);
            for shift in [[0.25, 0.0, 0.0], [0.0, -0.5, 0.0], [0.75, 0.25, -0.5]] {
                let moved = [pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2]];
                assert_eq!(
                    lattice_occupied(LatticeKind::Foam, moved, 0.25, 0.25, &params),
                    base,
                    "kelvin foam must repeat per cell ({pos:?} + {shift:?})"
                );
            }
        }
    }

    #[test]
    fn foam_irregularity_changes_the_structure() {
        // Jitter must actually move material around (and non-finite
        // irregularity must read as 0, not poison the sites).
        let regular = LatticeParams { irregularity: 0.0 };
        let organic = LatticeParams { irregularity: 0.6 };
        let differing = probes()
            .iter()
            .filter(|&&p| {
                lattice_occupied(LatticeKind::Foam, p, 0.25, 0.25, &regular)
                    != lattice_occupied(LatticeKind::Foam, p, 0.25, 0.25, &organic)
            })
            .count();
        assert!(
            differing > probes().len() / 20,
            "jitter should visibly reshape the foam ({differing} probes differ)"
        );
        let nan = LatticeParams {
            irregularity: f64::NAN,
        };
        for pos in probes() {
            assert_eq!(
                lattice_occupied(LatticeKind::Foam, pos, 0.25, 0.25, &nan),
                lattice_occupied(LatticeKind::Foam, pos, 0.25, 0.25, &regular),
                "NaN irregularity should read as 0"
            );
        }
    }
}
