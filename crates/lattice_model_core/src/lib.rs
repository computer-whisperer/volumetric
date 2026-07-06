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
}

impl LatticeKind {
    /// Decode the config-slot encoding (see `lattice_model_template`).
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Gyroid),
            1 => Some(Self::SchwarzP),
            2 => Some(Self::Struts),
            _ => None,
        }
    }
}

const TAU: f64 = core::f64::consts::TAU;

/// Whether the lattice contains `pos`, given the local relative density.
/// `cell_size` is the pattern period in model units; non-positive cell
/// sizes read as empty (an unpatched or corrupt config must not produce a
/// misleading solid).
pub fn lattice_occupied(kind: LatticeKind, pos: [f64; 3], cell_size: f64, density: f32) -> bool {
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

    const KINDS: [LatticeKind; 3] = [
        LatticeKind::Gyroid,
        LatticeKind::SchwarzP,
        LatticeKind::Struts,
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
                    !lattice_occupied(kind, pos, 0.25, 0.0),
                    "{kind:?} at {pos:?} should be empty at density 0"
                );
                assert!(
                    lattice_occupied(kind, pos, 0.25, 1.0),
                    "{kind:?} at {pos:?} should be solid at density 1"
                );
                assert!(
                    !lattice_occupied(kind, pos, 0.25, f32::NAN),
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
                    let occupied = lattice_occupied(kind, pos, 0.25, d);
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
                    .filter(|&&p| lattice_occupied(kind, p, 0.25, d))
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
                0.2
            ));
            // (0.5, 0.25) is a lattice corner in cell units (2, 1): on a
            // z-parallel strut for every z.
            assert!(lattice_occupied(
                LatticeKind::Struts,
                [0.5, 0.25, t],
                0.25,
                0.2
            ));
        }
        // The body center is the last point to fill.
        assert!(!lattice_occupied(
            LatticeKind::Struts,
            [0.125, 0.125, 0.125],
            0.25,
            0.5
        ));
        assert!(lattice_occupied(
            LatticeKind::Struts,
            [0.125, 0.125, 0.125],
            0.25,
            1.0
        ));
    }

    #[test]
    fn degenerate_cell_sizes_read_empty() {
        for kind in KINDS {
            assert!(!lattice_occupied(kind, [0.1; 3], 0.0, 0.8));
            assert!(!lattice_occupied(kind, [0.1; 3], -1.0, 0.8));
            assert!(!lattice_occupied(kind, [0.1; 3], f64::NAN, 0.8));
        }
    }
}
