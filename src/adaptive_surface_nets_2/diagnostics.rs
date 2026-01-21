//! Diagnostic functions for algorithm analysis.
//!
//! These functions are feature-gated and only available when building with
//! specific diagnostic features enabled. They are primarily used for research
//! and algorithm validation.
//!
//! # Feature Flags
//!
//! - `normal-diagnostic` + `native`: Enables `run_normal_diagnostics()`
//! - `edge-diagnostic` + `native`: Enables `run_edge_diagnostics()` and `run_crossing_diagnostics()`
//!
//! # Current Status
//!
//! The diagnostic functions are currently stubbed as the underlying Phase 4
//! implementation has been archived. The functions remain for API compatibility
//! and will print a message indicating they are not available.
//!
//! See PHASE4_ARCHIVE.md and EDGE_REFINEMENT_RESEARCH.md for research context.

#[cfg(feature = "normal-diagnostic")]
use crate::adaptive_surface_nets_2::types::{NormalDiagnosticEntry, SamplerFn, SamplingStats};

#[cfg(feature = "edge-diagnostic")]
use crate::adaptive_surface_nets_2::types::{EdgeDiagnosticEntry, SamplerFn, SamplingStats, SharpEdgeConfig};

/// Analytical ground truth for the rotated cube test case.
///
/// The rotated cube has rotation parameters: rx=35.264°, ry=45°, rz=0° (Euler XYZ)
/// This computes the 6 analytical face normals after rotation.
#[cfg(feature = "edge-diagnostic")]
pub struct AnalyticalRotatedCube {
    /// The 6 rotated face normals: +X, -X, +Y, -Y, +Z, -Z (after rotation)
    pub face_normals: [(f64, f64, f64); 6],
}

#[cfg(feature = "edge-diagnostic")]
impl AnalyticalRotatedCube {
    /// Create analytical ground truth for rotated cube with given Euler angles (degrees).
    /// Uses XYZ rotation order: R = Rz * Ry * Rx
    pub fn new(rx_deg: f64, ry_deg: f64, rz_deg: f64) -> Self {
        let rx = rx_deg.to_radians();
        let ry = ry_deg.to_radians();
        let rz = rz_deg.to_radians();

        let (sx, cx) = (rx.sin(), rx.cos());
        let (sy, cy) = (ry.sin(), ry.cos());
        let (sz, cz) = (rz.sin(), rz.cos());

        // Combined rotation matrix R = Rz * Ry * Rx
        // Row-major: r[row][col]
        let r = [
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy],
        ];

        // Original face normals (unit cube faces)
        let original_normals = [
            (1.0, 0.0, 0.0),  // +X
            (-1.0, 0.0, 0.0), // -X
            (0.0, 1.0, 0.0),  // +Y
            (0.0, -1.0, 0.0), // -Y
            (0.0, 0.0, 1.0),  // +Z
            (0.0, 0.0, -1.0), // -Z
        ];

        // Apply rotation to each normal
        let mut face_normals = [(0.0, 0.0, 0.0); 6];
        for (i, &(nx, ny, nz)) in original_normals.iter().enumerate() {
            face_normals[i] = (
                r[0][0] * nx + r[0][1] * ny + r[0][2] * nz,
                r[1][0] * nx + r[1][1] * ny + r[1][2] * nz,
                r[2][0] * nx + r[2][1] * ny + r[2][2] * nz,
            );
        }

        Self { face_normals }
    }

    /// Create with the standard test cube rotation (rx=35.264°, ry=45°, rz=0°).
    pub fn standard_test_cube() -> Self {
        Self::new(35.264, 45.0, 0.0)
    }

    /// Find the best-matching analytical normal for a given normal.
    /// Returns (best_normal, error_degrees).
    pub fn find_best_match(&self, normal: (f64, f64, f64)) -> ((f64, f64, f64), f64) {
        let mut best_normal = self.face_normals[0];
        let mut best_error = 180.0f64;

        for &face_normal in &self.face_normals {
            let error = angular_error_degrees_f64(normal, face_normal);
            if error < best_error {
                best_error = error;
                best_normal = face_normal;
            }
        }

        (best_normal, best_error)
    }

    /// Given two normals from a sharp edge detection, find the best pair of
    /// analytical normals and return the errors.
    /// Handles face swapping to find optimal assignment.
    pub fn compute_pair_errors(
        &self,
        normal_a: (f64, f64, f64),
        normal_b: Option<(f64, f64, f64)>,
    ) -> (f64, Option<f64>) {
        let (_, error_a) = self.find_best_match(normal_a);

        let error_b = normal_b.map(|nb| {
            let (_, err) = self.find_best_match(nb);
            err
        });

        (error_a, error_b)
    }
}

/// Compute angular error between two normals in degrees (f64 version).
#[cfg(feature = "edge-diagnostic")]
fn angular_error_degrees_f64(n1: (f64, f64, f64), n2: (f64, f64, f64)) -> f64 {
    let dot = n1.0 * n2.0 + n1.1 * n2.1 + n1.2 * n2.2;
    let clamped = dot.clamp(-1.0, 1.0);
    clamped.acos().to_degrees()
}

/// Run normal diagnostics: compute reference normals and compare various iteration levels.
///
/// Tests iteration counts: 0 (topology only), 4, 8, 12, 16, 24
/// Returns error statistics for each level compared to high-precision (32-iter) reference.
///
/// Note: This diagnostic feature requires the `native` feature for parallel iteration.
///
/// # Current Status
///
/// This function is currently stubbed as the underlying Phase 4 implementation
/// has been archived. Returns an empty vector.
#[cfg(all(feature = "normal-diagnostic", feature = "native"))]
pub fn run_normal_diagnostics<F>(
    _refined_positions: &[(f64, f64, f64)],
    _recomputed_normals: &[(f64, f64, f64)],
    _probe_epsilon: f64,
    _search_distance: f64,
    _sampler: &F,
    _stats: &SamplingStats,
) -> Vec<NormalDiagnosticEntry>
where
    F: SamplerFn,
{
    eprintln!("WARNING: run_normal_diagnostics() is stubbed - Phase 4 implementation archived");
    eprintln!("         See PHASE4_ARCHIVE.md and EDGE_REFINEMENT_RESEARCH.md for details");
    Vec::new()
}

/// Run edge detection diagnostics.
///
/// Tests various probe counts and residual thresholds against high-precision reference.
/// Returns error statistics for each method.
///
/// Also computes analytical ground truth validation using the rotated cube test case
/// (rotation rx=35.264°, ry=45°, rz=0°).
///
/// # Current Status
///
/// This function is currently stubbed as the underlying Phase 4 implementation
/// has been archived. Returns an empty vector.
#[cfg(all(feature = "edge-diagnostic", feature = "native"))]
pub fn run_edge_diagnostics<F>(
    _refined_positions: &[(f64, f64, f64)],
    _recomputed_normals: &[(f64, f64, f64)],
    _indices: &[u32],
    _probe_epsilon: f64,
    _search_distance: f64,
    _sampler: &F,
    _stats: &SamplingStats,
    _sharp_config: &SharpEdgeConfig,
) -> Vec<EdgeDiagnosticEntry>
where
    F: SamplerFn,
{
    eprintln!("WARNING: run_edge_diagnostics() is stubbed - Phase 4 implementation archived");
    eprintln!("         See PHASE4_ARCHIVE.md and EDGE_REFINEMENT_RESEARCH.md for details");
    Vec::new()
}

/// Diagnostic entry for Case 2 crossing position analysis.
#[cfg(feature = "edge-diagnostic")]
#[derive(Clone, Debug, Default)]
pub struct CrossingDiagnosticEntry {
    /// Method being tested
    pub method: String,
    /// Number of crossings analyzed
    pub crossing_count: usize,
    /// Mean position error (world units)
    pub position_error_mean: f64,
    /// Median position error
    pub position_error_median: f64,
    /// 95th percentile position error
    pub position_error_p95: f64,
    /// Maximum position error
    pub position_error_max: f64,
    /// Mean error as fraction of cell size
    pub error_cell_fraction_mean: f64,
}

/// Run Case 2 crossing position diagnostics.
///
/// Compares the cheap crossing position estimate against an expensive reference
/// computed using more iterations and multiple search directions.
///
/// # Current Status
///
/// This function is currently stubbed as the underlying Phase 4 implementation
/// has been archived. Returns a default entry.
#[cfg(all(feature = "edge-diagnostic", feature = "native"))]
pub fn run_crossing_diagnostics<F>(
    _crossings: &[()], // Placeholder type since EdgeCrossing is removed
    _vertices: &[(f64, f64, f64)],
    _normals: &[(f64, f64, f64)],
    _cell_size: (f64, f64, f64),
    _sampler: &F,
    _stats: &SamplingStats,
    _angle_threshold: f64,
) -> CrossingDiagnosticEntry
where
    F: SamplerFn,
{
    eprintln!("WARNING: run_crossing_diagnostics() is stubbed - Phase 4 implementation archived");
    eprintln!("         See PHASE4_ARCHIVE.md and EDGE_REFINEMENT_RESEARCH.md for details");
    CrossingDiagnosticEntry::default()
}
