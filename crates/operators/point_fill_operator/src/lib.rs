//! Point Fill Operator.
//!
//! Fills a 3D domain model with a point cloud — a Point1 [`FeaMesh`] of
//! sites kept where the domain's occupancy says inside. The primary
//! consumer is Voronoi lattice generation (the sites become cell seeds),
//! with the mesh clip/transform operators editing the cloud in between;
//! any Point1-consuming operator works.
//!
//! Patterns (both from `lattice_model_core`'s foam site machinery, so the
//! cloud is exactly the site set the built-in foam family uses):
//! - `grid`: one site per cubic cell (coset 0) — a cubic point grid.
//! - `bcc`: both cosets — the jittered-BCC set behind the foam lattice.
//!   At `seed` 0 this reproduces the foam family's sites bit-exactly, so
//!   `point_fill (bcc) -> voronoi skeleton` matches the built-in foam.
//!
//! `irregularity` jitters either pattern (0 = regular, 1 = fully organic;
//! displacement <= 0.25 cells per axis, so sites never collide); `seed`
//! selects alternate jitters (0 = the foam family's).
//!
//! Inputs:
//! - Input 0: ModelWASM (must be 3D) — the domain to fill
//! - Input 1: CBOR configuration:
//!   `{ pattern: "grid" / "bcc" .default "bcc", cell_size: float .default
//!   0.05, irregularity: float .default 0.3, seed: int .default 0 }`
//!
//! Output 0: CBOR-encoded Point1 `FeaMesh` (no fields; per-site data such
//! as Voronoi weights can be attached by downstream operators).

use lattice_model_core::foam_site_seeded;
use volumetric_abi::fea::{FeaElementKind, FeaMesh, encode_fea_mesh};
use volumetric_abi::host::{
    input_model_bounds, input_model_dimensions, input_model_sample, post_output, read_input,
    report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// Refuse fills past this many candidate sites (matches the strut
/// operator's cap; Voronoi consumers want far fewer).
const MAX_POINTS: u64 = 2_000_000;

/// Occupancy samples per batched host call.
const SAMPLE_CHUNK: usize = 8192;

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum PatternConfig {
    /// Cubic point grid (one site per cell).
    Grid,
    /// Jittered BCC (two sites per cell) — the foam family's site set.
    Bcc,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct FillConfig {
    pattern: PatternConfig,
    cell_size: f64,
    /// Site jitter, 0 (regular) ..= 1 (fully organic). Matches the foam
    /// lattice's knob.
    irregularity: f64,
    /// Jitter-hash salt: 0 is the foam family's site set, other values
    /// alternate jitters of the same pattern.
    seed: i64,
}

impl Default for FillConfig {
    fn default() -> Self {
        Self {
            pattern: PatternConfig::Bcc,
            cell_size: 0.05,
            irregularity: 0.3,
            seed: 0,
        }
    }
}

/// Enumerate every candidate site (model coordinates) whose cell touches
/// the box `[lo, hi]` expanded by one cell — jitter and the coset offset
/// keep a cell's site within one cell of its base, so no site that could
/// land inside the box is missed. Deterministic order.
fn enumerate_sites(lo: [f64; 3], hi: [f64; 3], config: &FillConfig) -> Vec<[f64; 3]> {
    let cell = config.cell_size;
    let cell_lo: [i64; 3] = core::array::from_fn(|a| (lo[a] / cell).floor() as i64 - 1);
    let cell_hi: [i64; 3] = core::array::from_fn(|a| (hi[a] / cell).ceil() as i64 + 1);
    let cosets: &[usize] = match config.pattern {
        PatternConfig::Grid => &[0],
        PatternConfig::Bcc => &[0, 1],
    };
    let mut sites = Vec::new();
    for k in cell_lo[2]..=cell_hi[2] {
        for j in cell_lo[1]..=cell_hi[1] {
            for i in cell_lo[0]..=cell_hi[0] {
                for &coset in cosets {
                    let s =
                        foam_site_seeded([i, j, k], coset, config.irregularity, config.seed as u64);
                    sites.push([s[0] * cell, s[1] * cell, s[2] * cell]);
                }
            }
        }
    }
    sites
}

/// Assemble the kept sites into a Point1 mesh (one element per point).
fn points_mesh(sites: Vec<[f64; 3]>) -> Result<FeaMesh, String> {
    if sites.is_empty() {
        return Err(
            "no points inside the domain (is cell_size much larger than the model, \
             or the model empty?)"
                .to_string(),
        );
    }
    let connectivity: Vec<u32> = (0..sites.len() as u32).collect();
    let mesh = FeaMesh {
        element_kind: FeaElementKind::Point1,
        node_positions: sites.into_iter().flatten().collect(),
        connectivity,
        node_fields: vec![],
        element_fields: vec![],
    };
    mesh.validate()?;
    Ok(mesh)
}

fn build_points(config: &FillConfig) -> Result<FeaMesh, String> {
    if !(config.cell_size.is_finite() && config.cell_size > 0.0) {
        return Err(format!(
            "cell_size must be positive, got {}",
            config.cell_size
        ));
    }
    if !(config.irregularity.is_finite() && (0.0..=1.0).contains(&config.irregularity)) {
        return Err(format!(
            "irregularity must be in 0..=1, got {}",
            config.irregularity
        ));
    }
    if config.seed < 0 {
        return Err(format!("seed must be non-negative, got {}", config.seed));
    }

    let dims =
        input_model_dimensions(0).ok_or_else(|| "input 0 is not a usable model".to_string())?;
    if dims != 3 {
        return Err(format!(
            "point fill needs a 3D domain model; input has {dims} dimensions"
        ));
    }
    let bounds =
        input_model_bounds(0, 3).ok_or_else(|| "failed to read model bounds".to_string())?;
    let lo = [bounds[0], bounds[2], bounds[4]];
    let hi = [bounds[1], bounds[3], bounds[5]];
    if lo.iter().chain(&hi).any(|v| !v.is_finite()) || lo.iter().zip(&hi).any(|(a, b)| a > b) {
        return Err(format!("degenerate model bounds {lo:?} .. {hi:?}"));
    }

    let sites_per_cell = match config.pattern {
        PatternConfig::Grid => 1,
        PatternConfig::Bcc => 2,
    };
    let estimate: u64 = (0..3)
        .map(|a| ((hi[a] - lo[a]) / config.cell_size).ceil().max(0.0) as u64 + 3)
        .product::<u64>()
        .saturating_mul(sites_per_cell);
    if estimate > MAX_POINTS {
        return Err(format!(
            "cell_size {} would enumerate ~{estimate} points (cap {MAX_POINTS}); \
             raise cell_size",
            config.cell_size
        ));
    }

    let candidates = enumerate_sites(lo, hi, config);

    // Batched occupancy of every candidate site.
    let mut kept = Vec::new();
    for chunk in candidates.chunks(SAMPLE_CHUNK) {
        let positions: Vec<f64> = chunk.iter().flatten().copied().collect();
        let samples = input_model_sample(0, &positions, 3)
            .ok_or_else(|| "sampling the domain model failed".to_string())?;
        kept.extend(
            chunk
                .iter()
                .zip(&samples)
                .filter(|&(_, &s)| is_occupied(s))
                .map(|(&p, _)| p),
        );
    }

    points_mesh(kept)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(1);
        if buf.is_empty() {
            FillConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    match build_points(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("point fill failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ pattern: "grid" / "bcc" .default "bcc", cell_size: float .default 0.05, irregularity: float .default 0.3, seed: int .default 0 }"#
            .to_string();
        OperatorMetadata {
            name: "point_fill_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Point Fill".to_string(),
            description: "Fill a 3D domain model with a point cloud (Point1 mesh) — \
                          Voronoi cell seeds and other site sets."
                .to_string(),
            category: "Lattice".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<circle cx="5" cy="5" r="1"/>"##,
                r##"<circle cx="12" cy="6" r="1"/>"##,
                r##"<circle cx="19" cy="5" r="1"/>"##,
                r##"<circle cx="6" cy="12" r="1"/>"##,
                r##"<circle cx="13" cy="13" r="1"/>"##,
                r##"<circle cx="19" cy="12" r="1"/>"##,
                r##"<circle cx="5" cy="19" r="1"/>"##,
                r##"<circle cx="12" cy="18" r="1"/>"##,
                r##"<circle cx="19" cy="19" r="1"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Domain model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_filter(sites: Vec<[f64; 3]>, c: [f64; 3], r: f64) -> Vec<[f64; 3]> {
        sites
            .into_iter()
            .filter(|p| (0..3).map(|i| (p[i] - c[i]).powi(2)).sum::<f64>() < r * r)
            .collect()
    }

    #[test]
    fn grid_without_jitter_sits_on_cell_multiples() {
        let config = FillConfig {
            pattern: PatternConfig::Grid,
            cell_size: 0.5,
            irregularity: 0.0,
            seed: 0,
        };
        let sites = enumerate_sites([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], &config);
        assert!(!sites.is_empty());
        for p in &sites {
            for c in p {
                let cells = c / 0.5;
                assert!(
                    (cells - cells.round()).abs() < 1e-12,
                    "site {p:?} is off the grid"
                );
            }
        }
    }

    #[test]
    fn bcc_without_jitter_has_the_bcc_nearest_gap() {
        let config = FillConfig {
            pattern: PatternConfig::Bcc,
            cell_size: 1.0,
            irregularity: 0.0,
            seed: 0,
        };
        let sites = enumerate_sites([0.0, 0.0, 0.0], [3.0, 3.0, 3.0], &config);
        // Nearest-neighbor distance in plain BCC is sqrt(3)/2 * cell.
        let expected = 3.0f64.sqrt() / 2.0;
        let p = sites
            .iter()
            .find(|p| p.iter().all(|c| (c - 1.5).abs() < 0.26)) // a coset-1 site well interior
            .expect("an interior site");
        let nearest = sites
            .iter()
            .filter(|q| *q != p)
            .map(|q| (0..3).map(|i| (p[i] - q[i]).powi(2)).sum::<f64>().sqrt())
            .fold(f64::MAX, f64::min);
        assert!(
            (nearest - expected).abs() < 1e-9,
            "nearest {nearest}, expected {expected}"
        );
    }

    #[test]
    fn jitter_moves_sites_and_seeds_pick_different_jitters() {
        let base = FillConfig {
            pattern: PatternConfig::Bcc,
            cell_size: 1.0,
            irregularity: 0.8,
            seed: 0,
        };
        let lo = [0.0, 0.0, 0.0];
        let hi = [4.0, 4.0, 4.0];
        let regular = enumerate_sites(
            lo,
            hi,
            &FillConfig {
                irregularity: 0.0,
                ..base
            },
        );
        let jittered = enumerate_sites(lo, hi, &base);
        assert_eq!(regular.len(), jittered.len());
        let moved = regular
            .iter()
            .zip(&jittered)
            .filter(|(a, b)| a != b)
            .count();
        assert!(moved > regular.len() / 2, "jitter moved only {moved} sites");
        // Jitter is bounded: no site strays more than 0.25 cells per axis.
        for (a, b) in regular.iter().zip(&jittered) {
            for i in 0..3 {
                assert!((a[i] - b[i]).abs() <= 0.25 + 1e-12);
            }
        }

        let reseeded = enumerate_sites(lo, hi, &FillConfig { seed: 7, ..base });
        assert_ne!(jittered, reseeded, "seed 7 should re-jitter");
        // Determinism: the same config enumerates the same sites.
        assert_eq!(jittered, enumerate_sites(lo, hi, &base));
    }

    #[test]
    fn filtered_sites_assemble_into_a_valid_point1_mesh() {
        let config = FillConfig {
            pattern: PatternConfig::Bcc,
            cell_size: 0.25,
            irregularity: 0.3,
            seed: 0,
        };
        let all = enumerate_sites([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], &config);
        let kept = sphere_filter(all.clone(), [0.0, 0.0, 0.0], 1.0);
        assert!(!kept.is_empty() && kept.len() < all.len());

        let count = kept.len();
        let mesh = points_mesh(kept).unwrap();
        assert_eq!(mesh.element_kind, FeaElementKind::Point1);
        assert_eq!(mesh.node_count(), count);
        assert_eq!(mesh.element_count(), count);
        for i in 0..count {
            let p = mesh.node_position(i);
            assert!(p.iter().map(|c| c * c).sum::<f64>() < 1.0);
        }

        assert!(points_mesh(Vec::new()).is_err());
    }
}
