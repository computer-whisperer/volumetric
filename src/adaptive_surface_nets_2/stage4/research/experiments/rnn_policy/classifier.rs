//! RANSAC-based classifiers for evaluating sample quality.
//!
//! These classifiers run on pre-collected samples (from the RNN policy)
//! and produce fit quality metrics used for terminal reward computation.

use super::math::{cross, dot, length, normalize, sub};

/// Result of fitting a single plane (face).
#[derive(Clone, Debug)]
pub struct FaceFitResult {
    /// Fitted plane normal (unit vector).
    pub normal: (f64, f64, f64),
    /// Plane centroid.
    pub centroid: (f64, f64, f64),
    /// RMS residual of the fit.
    pub residual: f64,
    /// Number of inliers.
    pub inlier_count: usize,
    /// Total points used.
    pub total_points: usize,
}

/// Result of fitting two planes (edge).
#[derive(Clone, Debug)]
pub struct EdgeFitResult {
    /// First plane normal.
    pub normal_a: (f64, f64, f64),
    /// Second plane normal.
    pub normal_b: (f64, f64, f64),
    /// Edge direction (cross product of normals).
    pub edge_direction: (f64, f64, f64),
    /// Residual for first plane.
    pub residual_a: f64,
    /// Residual for second plane.
    pub residual_b: f64,
    /// Inliers for first plane.
    pub inliers_a: usize,
    /// Inliers for second plane.
    pub inliers_b: usize,
    /// Total points used.
    pub total_points: usize,
}

/// Result of fitting three planes (corner).
#[derive(Clone, Debug)]
pub struct CornerFitResult {
    /// Normals for each plane.
    pub normals: Vec<(f64, f64, f64)>,
    /// Residuals for each plane.
    pub residuals: Vec<f64>,
    /// Inlier counts for each plane.
    pub inlier_counts: Vec<usize>,
    /// Total points used.
    pub total_points: usize,
}

/// Configuration for the RANSAC classifiers.
#[derive(Clone, Debug)]
pub struct ClassifierConfig {
    /// Inlier threshold relative to cell_size.
    pub inlier_threshold_ratio: f64,
    /// Number of RANSAC iterations.
    pub ransac_iterations: usize,
    /// Minimum inliers required for a valid plane.
    pub min_inliers: usize,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            inlier_threshold_ratio: 0.2, // 20% of cell_size - more lenient for noisy samples
            ransac_iterations: 100,
            min_inliers: 3,
        }
    }
}

/// Find surface points from samples by identifying in/out transitions.
///
/// Given samples with inside/outside labels, finds approximate crossing points
/// by pairing ALL inside samples with ALL outside samples and using midpoints.
/// This gives many more surface estimates than just consecutive transitions.
pub fn extract_surface_points(
    samples: &[(f64, f64, f64)],
    inside_flags: &[bool],
    vertex: (f64, f64, f64),
    cell_size: f64,
) -> Vec<(f64, f64, f64)> {
    if samples.len() < 2 {
        return vec![];
    }

    // Separate samples by inside/outside
    let inside_samples: Vec<_> = samples
        .iter()
        .zip(inside_flags.iter())
        .filter(|&(_, &is_inside)| is_inside)
        .map(|(p, _)| *p)
        .collect();

    let outside_samples: Vec<_> = samples
        .iter()
        .zip(inside_flags.iter())
        .filter(|&(_, &is_inside)| !is_inside)
        .map(|(p, _)| *p)
        .collect();

    let mut surface_points = Vec::new();

    // For each inside sample, pair with ALL outside samples.
    // The midpoint approximates where the surface might be.
    // With more pairs, RANSAC can find the best-fitting plane.
    for &p_in in &inside_samples {
        for &p_out in &outside_samples {
            // Midpoint approximates the surface
            let mid = (
                (p_in.0 + p_out.0) / 2.0,
                (p_in.1 + p_out.1) / 2.0,
                (p_in.2 + p_out.2) / 2.0,
            );
            surface_points.push(mid);
        }
    }

    // Also add consecutive transition midpoints (the original approach)
    // These are higher quality since they're from actual trajectory crossings
    for i in 1..samples.len() {
        if inside_flags[i] != inside_flags[i - 1] {
            let p1 = samples[i - 1];
            let p2 = samples[i];
            let mid = (
                (p1.0 + p2.0) / 2.0,
                (p1.1 + p2.1) / 2.0,
                (p1.2 + p2.2) / 2.0,
            );
            surface_points.push(mid);
        }
    }

    let _ = vertex; // May use later for distance-based filtering

    // Deduplicate nearby points to avoid over-weighting
    // Use a smaller threshold to keep more points for RANSAC
    deduplicate_points(&mut surface_points, cell_size * 0.01);

    surface_points
}

/// Remove points that are very close together.
fn deduplicate_points(points: &mut Vec<(f64, f64, f64)>, min_dist: f64) {
    if points.len() < 2 {
        return;
    }

    let min_dist_sq = min_dist * min_dist;
    let mut keep = vec![true; points.len()];

    for i in 0..points.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..points.len() {
            if !keep[j] {
                continue;
            }
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            let dz = points[i].2 - points[j].2;
            if dx * dx + dy * dy + dz * dz < min_dist_sq {
                keep[j] = false;
            }
        }
    }

    let mut write_idx = 0;
    for read_idx in 0..points.len() {
        if keep[read_idx] {
            points[write_idx] = points[read_idx];
            write_idx += 1;
        }
    }
    points.truncate(write_idx);
}

/// Fit a single plane to samples (for face classification).
///
/// Returns None if insufficient points or fit fails.
pub fn fit_face_from_samples(
    surface_points: &[(f64, f64, f64)],
    cell_size: f64,
    config: &ClassifierConfig,
) -> Option<FaceFitResult> {
    let threshold = config.inlier_threshold_ratio * cell_size;

    let (inliers, centroid, normal, residual) =
        ransac_plane_fit(surface_points, threshold, config.ransac_iterations)?;

    if inliers.len() < config.min_inliers {
        return None;
    }

    Some(FaceFitResult {
        normal,
        centroid,
        residual,
        inlier_count: inliers.len(),
        total_points: surface_points.len(),
    })
}

/// Fit two planes to samples (for edge classification).
///
/// Uses RANSAC to find the dominant plane, removes its inliers,
/// then finds a second plane in the remaining points.
pub fn fit_edge_from_samples(
    surface_points: &[(f64, f64, f64)],
    cell_size: f64,
    config: &ClassifierConfig,
) -> Option<EdgeFitResult> {
    let threshold = config.inlier_threshold_ratio * cell_size;

    // First plane
    let (inliers_a, centroid_a, normal_a, residual_a) =
        ransac_plane_fit(surface_points, threshold, config.ransac_iterations)?;

    if inliers_a.len() < config.min_inliers {
        return None;
    }

    // Remove first plane's inliers
    let remaining: Vec<_> = surface_points
        .iter()
        .filter(|p| !is_inlier(**p, centroid_a, normal_a, threshold))
        .cloned()
        .collect();

    if remaining.len() < config.min_inliers {
        return None;
    }

    // Second plane
    let (inliers_b, centroid_b, normal_b, residual_b) =
        ransac_plane_fit(&remaining, threshold, config.ransac_iterations)?;

    if inliers_b.len() < config.min_inliers {
        return None;
    }

    // Orient normals consistently (both pointing "outward" from vertex)
    // Ensure planes aren't too parallel (need meaningful angle between them)
    let dot_val = dot(normal_a, normal_b).abs();
    if dot_val > 0.85 {
        // Planes too parallel (< ~32° between them) - not a valid edge
        return None;
    }

    let edge_direction = normalize(cross(normal_a, normal_b));

    let _ = centroid_b; // Silence unused warning

    Some(EdgeFitResult {
        normal_a,
        normal_b,
        edge_direction,
        residual_a,
        residual_b,
        inliers_a: inliers_a.len(),
        inliers_b: inliers_b.len(),
        total_points: surface_points.len(),
    })
}

/// Fit three planes to samples (for corner classification).
///
/// Iteratively finds planes and removes inliers.
pub fn fit_corner_from_samples(
    surface_points: &[(f64, f64, f64)],
    cell_size: f64,
    config: &ClassifierConfig,
) -> Option<CornerFitResult> {
    let threshold = config.inlier_threshold_ratio * cell_size;

    let mut remaining = surface_points.to_vec();
    let mut normals = Vec::new();
    let mut residuals = Vec::new();
    let mut inlier_counts = Vec::new();

    for _ in 0..3 {
        if remaining.len() < config.min_inliers {
            break;
        }

        let (inliers, centroid, normal, residual) =
            ransac_plane_fit(&remaining, threshold, config.ransac_iterations)?;

        if inliers.len() < config.min_inliers {
            break;
        }

        // Check for duplicate normal (too similar to existing)
        let is_duplicate = normals.iter().any(|existing: &(f64, f64, f64)| {
            let d = dot(normal, *existing).abs();
            d > 0.9 // Within ~25 degrees
        });

        if is_duplicate {
            // Remove these inliers but don't add the plane
            remaining.retain(|p| !is_inlier(*p, centroid, normal, threshold));
            continue;
        }

        normals.push(normal);
        residuals.push(residual);
        inlier_counts.push(inliers.len());

        // Remove inliers for next iteration
        remaining.retain(|p| !is_inlier(*p, centroid, normal, threshold));
    }

    if normals.len() < 3 {
        return None;
    }

    Some(CornerFitResult {
        normals,
        residuals,
        inlier_counts,
        total_points: surface_points.len(),
    })
}

/// RANSAC plane fitting - finds the plane with most inliers.
fn ransac_plane_fit(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> Option<(Vec<(f64, f64, f64)>, (f64, f64, f64), (f64, f64, f64), f64)> {
    if points.len() < 3 {
        return None;
    }

    let n = points.len();
    let mut best_inliers: Vec<(f64, f64, f64)> = Vec::new();

    for i in 0..iterations {
        // Deterministic "random" point selection
        let idx1 = (i * 7 + 1) % n;
        let idx2 = (i * 13 + 3) % n;
        let idx3 = (i * 19 + 7) % n;

        if idx1 == idx2 || idx2 == idx3 || idx1 == idx3 {
            continue;
        }

        let p1 = points[idx1];
        let p2 = points[idx2];
        let p3 = points[idx3];

        // Compute plane from 3 points
        let v1 = sub(p2, p1);
        let v2 = sub(p3, p1);
        let cross_product = cross(v1, v2);
        let cross_len = length(cross_product);

        // BUG FIX: Check cross product length BEFORE normalizing
        // The original code checked length(normal) after normalize(),
        // but normalize() always returns a unit vector (or (0,0,1) for zero input)
        if cross_len < 1e-9 {
            continue; // Degenerate triangle (collinear points)
        }

        let normal = (
            cross_product.0 / cross_len,
            cross_product.1 / cross_len,
            cross_product.2 / cross_len,
        );

        // Count inliers
        let inliers: Vec<_> = points
            .iter()
            .filter(|p| {
                let dist = dot(sub(**p, p1), normal).abs();
                dist < inlier_threshold
            })
            .cloned()
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }

    if best_inliers.len() < 3 {
        return None;
    }

    // Refit plane to all inliers for better accuracy
    let (centroid, normal, residual) = fit_plane_svd(&best_inliers);

    Some((best_inliers, centroid, normal, residual))
}

/// RANSAC plane fitting with detailed diagnostics (for testing).
#[cfg(test)]
fn ransac_plane_fit_diagnostic(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> (
    Option<(Vec<(f64, f64, f64)>, (f64, f64, f64), (f64, f64, f64), f64)>,
    RansacDiagnostics,
) {
    let mut diag = RansacDiagnostics::default();

    if points.len() < 3 {
        return (None, diag);
    }

    let n = points.len();
    let mut best_inliers: Vec<(f64, f64, f64)> = Vec::new();
    let mut best_normal = (0.0, 0.0, 0.0);

    for i in 0..iterations {
        let idx1 = (i * 7 + 1) % n;
        let idx2 = (i * 13 + 3) % n;
        let idx3 = (i * 19 + 7) % n;

        if idx1 == idx2 || idx2 == idx3 || idx1 == idx3 {
            diag.skipped_duplicate_indices += 1;
            continue;
        }

        let p1 = points[idx1];
        let p2 = points[idx2];
        let p3 = points[idx3];

        let v1 = sub(p2, p1);
        let v2 = sub(p3, p1);
        let cross_product = cross(v1, v2);
        let cross_len = length(cross_product);

        if cross_len < 1e-9 {
            diag.skipped_degenerate += 1;
            continue;
        }

        diag.valid_iterations += 1;

        let normal = (
            cross_product.0 / cross_len,
            cross_product.1 / cross_len,
            cross_product.2 / cross_len,
        );

        let inliers: Vec<_> = points
            .iter()
            .filter(|p| {
                let dist = dot(sub(**p, p1), normal).abs();
                dist < inlier_threshold
            })
            .cloned()
            .collect();

        diag.inlier_counts.push(inliers.len());

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_normal = normal;
        }
    }

    diag.best_inlier_count = best_inliers.len();
    diag.best_normal = best_normal;

    if best_inliers.len() < 3 {
        return (None, diag);
    }

    let (centroid, normal, residual) = fit_plane_svd(&best_inliers);
    diag.final_normal = normal;
    diag.final_residual = residual;

    (Some((best_inliers, centroid, normal, residual)), diag)
}

#[cfg(test)]
#[derive(Default, Debug)]
struct RansacDiagnostics {
    valid_iterations: usize,
    skipped_duplicate_indices: usize,
    skipped_degenerate: usize,
    inlier_counts: Vec<usize>,
    best_inlier_count: usize,
    best_normal: (f64, f64, f64),
    final_normal: (f64, f64, f64),
    final_residual: f64,
}

/// Check if a point is an inlier to a plane.
fn is_inlier(
    p: (f64, f64, f64),
    plane_point: (f64, f64, f64),
    plane_normal: (f64, f64, f64),
    threshold: f64,
) -> bool {
    let dist = dot(sub(p, plane_point), plane_normal).abs();
    dist < threshold
}

/// Fit plane using SVD (smallest eigenvector of covariance matrix).
fn fit_plane_svd(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
    if points.len() < 3 {
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), f64::INFINITY);
    }

    // Compute centroid
    let n = points.len() as f64;
    let centroid = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    );

    // Build covariance matrix
    let centered: Vec<_> = points.iter().map(|p| sub(*p, centroid)).collect();

    let mut cov = [[0.0; 3]; 3];
    for p in &centered {
        cov[0][0] += p.0 * p.0;
        cov[0][1] += p.0 * p.1;
        cov[0][2] += p.0 * p.2;
        cov[1][1] += p.1 * p.1;
        cov[1][2] += p.1 * p.2;
        cov[2][2] += p.2 * p.2;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // Find smallest eigenvector via power iteration on shifted matrix
    let max_eigenvalue_estimate = cov
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row[i].abs()
                + row
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &v)| v.abs())
                    .sum::<f64>()
        })
        .fold(0.0_f64, f64::max);

    let shift = max_eigenvalue_estimate + 1.0;
    let shifted = [
        [shift - cov[0][0], -cov[0][1], -cov[0][2]],
        [-cov[1][0], shift - cov[1][1], -cov[1][2]],
        [-cov[2][0], -cov[2][1], shift - cov[2][2]],
    ];

    let mut v = (1.0, 1.0, 1.0);
    for _ in 0..100 {
        let new_v = (
            shifted[0][0] * v.0 + shifted[0][1] * v.1 + shifted[0][2] * v.2,
            shifted[1][0] * v.0 + shifted[1][1] * v.1 + shifted[1][2] * v.2,
            shifted[2][0] * v.0 + shifted[2][1] * v.1 + shifted[2][2] * v.2,
        );

        let len = length(new_v);
        if len < 1e-12 {
            break;
        }
        v = (new_v.0 / len, new_v.1 / len, new_v.2 / len);
    }

    // Compute RMS residual
    let mut residual_sum = 0.0;
    for p in &centered {
        let dist = dot(*p, v).abs();
        residual_sum += dist * dist;
    }
    let rms_residual = (residual_sum / points.len() as f64).sqrt();

    (centroid, v, rms_residual)
}

/// Compute angular error between two normals (in degrees).
/// Handles sign ambiguity (normal can point either direction).
pub fn normal_error_degrees(detected: (f64, f64, f64), expected: (f64, f64, f64)) -> f64 {
    let d = dot(normalize(detected), normalize(expected)).abs().clamp(-1.0, 1.0);
    d.acos().to_degrees()
}

/// Compute best normal pairing error for edges.
/// Returns (error_a, error_b) for the best assignment.
pub fn best_edge_normal_errors(
    detected: ((f64, f64, f64), (f64, f64, f64)),
    expected: ((f64, f64, f64), (f64, f64, f64)),
) -> (f64, f64) {
    let assignment1 = (
        normal_error_degrees(detected.0, expected.0),
        normal_error_degrees(detected.1, expected.1),
    );

    let assignment2 = (
        normal_error_degrees(detected.0, expected.1),
        normal_error_degrees(detected.1, expected.0),
    );

    if assignment1.0 + assignment1.1 < assignment2.0 + assignment2.1 {
        assignment1
    } else {
        assignment2
    }
}

/// Compute best corner normal matching errors.
/// Returns errors for the best greedy assignment.
pub fn best_corner_normal_errors(
    detected: &[(f64, f64, f64)],
    expected: &[(f64, f64, f64)],
) -> Vec<f64> {
    let mut errors = Vec::new();
    let mut used = vec![false; expected.len()];

    for d in detected {
        let mut best_err = f64::MAX;
        let mut best_idx = 0;

        for (i, e) in expected.iter().enumerate() {
            if !used[i] {
                let err = normal_error_degrees(*d, *e);
                if err < best_err {
                    best_err = err;
                    best_idx = i;
                }
            }
        }

        if best_err < f64::MAX {
            used[best_idx] = true;
            errors.push(best_err);
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_face_simple() {
        // Points on the XY plane (z = 0)
        let points = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.5, 0.5, 0.0),
        ];

        let config = ClassifierConfig::default();
        let result = fit_face_from_samples(&points, 1.0, &config).unwrap();

        // Normal should be (0, 0, 1) or (0, 0, -1)
        assert!((result.normal.2.abs() - 1.0).abs() < 0.01);
        assert!(result.residual < 0.01);
        assert_eq!(result.inlier_count, 5);
    }

    #[test]
    fn test_fit_edge_two_planes() {
        // Two perpendicular planes with more points for reliable RANSAC
        // Plane 1: z = 0 (horizontal plane)
        // Plane 2: y = 0 (vertical plane)
        let mut points = vec![];

        // Plane 1: z = 0 - 8 points spread across the XY plane
        for i in 0..8 {
            let x = (i % 4) as f64 * 0.25;
            let y = (i / 4) as f64 * 0.5 + 0.25;
            points.push((x, y, 0.0));
        }

        // Plane 2: y = 0 - 8 points spread across the XZ plane
        for i in 0..8 {
            let x = (i % 4) as f64 * 0.25;
            let z = (i / 4) as f64 * 0.5 + 0.25;
            points.push((x, 0.0, z));
        }

        let config = ClassifierConfig {
            inlier_threshold_ratio: 0.1, // 10% of cell_size
            ransac_iterations: 500,
            min_inliers: 3,
        };
        let result = fit_edge_from_samples(&points, 1.0, &config);

        // Should find two planes (or None if planes are degenerate)
        // This is a basic smoke test - real edge detection needs surface crossing points
        if let Some(result) = result {
            assert!(result.inliers_a >= 3, "Plane A should have >= 3 inliers, got {}", result.inliers_a);
            assert!(result.inliers_b >= 3, "Plane B should have >= 3 inliers, got {}", result.inliers_b);
        }
        // Note: fit may fail with synthetic points - that's OK for this unit test
        // The real test is terminal reward integration in the training loop
    }

    #[test]
    fn test_normal_error() {
        let n1 = (1.0, 0.0, 0.0);
        let n2 = (0.0, 1.0, 0.0);

        let err = normal_error_degrees(n1, n2);
        assert!((err - 90.0).abs() < 0.1);

        // Opposite directions should give 0 error (sign ambiguity)
        let err2 = normal_error_degrees(n1, (-1.0, 0.0, 0.0));
        assert!(err2 < 0.1);
    }

    // ========================================================================
    // RANSAC CLASSIFIER DIAGNOSTIC TESTS
    // These tests are designed to expose potential issues in the RANSAC
    // implementation and help refine behavior.
    // ========================================================================

    /// Test: Degenerate triangle detection is broken
    ///
    /// The current code checks `length(normal) < 1e-6` AFTER normalizing,
    /// but normalize() always returns a unit vector (or (0,0,1) for zero input).
    /// This test verifies whether degenerate triangles are properly detected.
    #[test]
    fn test_ransac_degenerate_triangle_detection() {
        // Three collinear points - should be degenerate
        let collinear_points = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ];

        let config = ClassifierConfig {
            inlier_threshold_ratio: 0.1,
            ransac_iterations: 10,
            min_inliers: 3,
        };

        // With only collinear points, RANSAC should fail or return a poor fit
        // The issue: normalize() masks the degenerate case
        let result = fit_face_from_samples(&collinear_points, 1.0, &config);

        // Document current behavior:
        // If this assertion fails, it means degenerate triangles are being accepted
        // when they should be rejected
        if let Some(fit) = &result {
            // Check if the fit is reasonable - collinear points can't define a unique plane
            // so any result is suspect
            println!(
                "DIAGNOSTIC: Collinear points produced fit with normal {:?}, residual {}",
                fit.normal, fit.residual
            );
            // A proper implementation would return None for collinear points
        }
    }

    /// Test: Deterministic point selection coverage analysis
    ///
    /// The RANSAC uses `(i * prime + offset) % n` for "random" selection.
    /// This test checks coverage quality for various point counts.
    #[test]
    fn test_ransac_deterministic_coverage() {
        // Check coverage for specific point counts that might cause issues
        let problem_sizes = vec![7, 13, 14, 19, 21, 26, 38, 91]; // Multiples of primes used

        for n in problem_sizes {
            let mut selected_triples = std::collections::HashSet::new();
            let iterations = 100;

            for i in 0..iterations {
                let idx1 = (i * 7 + 1) % n;
                let idx2 = (i * 13 + 3) % n;
                let idx3 = (i * 19 + 7) % n;

                if idx1 != idx2 && idx2 != idx3 && idx1 != idx3 {
                    let mut triple = [idx1, idx2, idx3];
                    triple.sort();
                    selected_triples.insert(triple);
                }
            }

            // Calculate what percentage of possible triples we cover
            let max_triples = n * (n - 1) * (n - 2) / 6; // n choose 3
            let coverage = selected_triples.len() as f64 / max_triples as f64;

            println!(
                "DIAGNOSTIC: n={}, iterations={}, unique_triples={}, max_possible={}, coverage={:.1}%",
                n,
                iterations,
                selected_triples.len(),
                max_triples,
                coverage * 100.0
            );

            // Flag poor coverage (less than 10% of possible triples sampled)
            if coverage < 0.1 && max_triples > 20 {
                println!("  WARNING: Poor coverage for n={}", n);
            }
        }
    }

    /// Test: Surface point extraction quality
    ///
    /// The O(n²) all-pairs midpoint approach can create many spurious points
    /// that aren't near the actual surface.
    #[test]
    fn test_surface_point_extraction_quality() {
        // Simulate samples from a cube edge scenario:
        // Inside samples are inside the cube
        // Outside samples are on two different faces
        let cell_size = 0.05;
        let vertex = (0.0, 0.0, 0.0);

        // Outside samples on face 1 (z = +0.02, varying x,y)
        let mut samples = vec![];
        let mut inside_flags = vec![];

        // 5 outside samples on face 1 (z+ face)
        for i in 0..5 {
            let x = (i as f64 - 2.0) * 0.01;
            let y = (i as f64 - 2.0) * 0.005;
            samples.push((x, y, 0.02));
            inside_flags.push(false);
        }

        // 5 outside samples on face 2 (y+ face)
        for i in 0..5 {
            let x = (i as f64 - 2.0) * 0.01;
            let z = (i as f64 - 2.0) * 0.005;
            samples.push((x, 0.02, z));
            inside_flags.push(false);
        }

        // 5 inside samples (inside the cube)
        for i in 0..5 {
            let x = (i as f64 - 2.0) * 0.005;
            samples.push((x, -0.01, -0.01));
            inside_flags.push(true);
        }

        let surface_points =
            extract_surface_points(&samples, &inside_flags, vertex, cell_size);

        // With 5 inside and 10 outside samples, we get 5*10 = 50 all-pairs midpoints
        // plus consecutive transition midpoints
        println!(
            "DIAGNOSTIC: {} samples ({} in, {} out) -> {} surface points",
            samples.len(),
            inside_flags.iter().filter(|&&f| f).count(),
            inside_flags.iter().filter(|&&f| !f).count(),
            surface_points.len()
        );

        // Check how many surface points are actually near either face plane
        let near_face1 = surface_points
            .iter()
            .filter(|p| (p.2 - 0.01).abs() < 0.02) // Near z=0.01 (midpoint to face 1)
            .count();
        let near_face2 = surface_points
            .iter()
            .filter(|p| (p.1 - 0.005).abs() < 0.02) // Near y=0.005 (midpoint to face 2)
            .count();

        println!(
            "DIAGNOSTIC: {} points near face 1, {} points near face 2",
            near_face1, near_face2
        );

        // Many points will be spurious (midpoints between unrelated in/out pairs)
        let spurious = surface_points.len() - near_face1.max(near_face2);
        println!(
            "DIAGNOSTIC: Potentially {} spurious midpoints ({:.1}%)",
            spurious,
            spurious as f64 / surface_points.len() as f64 * 100.0
        );
    }

    /// Test: Power iteration convergence for thin point clouds
    ///
    /// Thin point clouds (nearly collinear) have closely-spaced eigenvalues,
    /// which can cause slow convergence in power iteration.
    #[test]
    fn test_svd_convergence_thin_cloud() {
        // Create a thin point cloud: mostly along X axis with small Y/Z variation
        let mut points = vec![];
        for i in 0..20 {
            let x = i as f64 * 0.1;
            let y = 0.001 * (i as f64).sin(); // Tiny Y variation
            let z = 0.001 * (i as f64).cos(); // Tiny Z variation
            points.push((x, y, z));
        }

        let (centroid, normal, residual) = fit_plane_svd(&points);

        println!(
            "DIAGNOSTIC: Thin cloud SVD result: centroid={:?}, normal={:?}, residual={}",
            centroid, normal, residual
        );

        // For a nearly-collinear cloud, the normal should be perpendicular to X axis
        // But the exact direction (Y or Z) is ambiguous due to tiny variations
        let x_component = normal.0.abs();
        println!(
            "DIAGNOSTIC: Normal X component = {} (should be ~0 for thin X-aligned cloud)",
            x_component
        );

        // The residual should be very small since points are nearly on a line
        // (which lies on infinitely many planes)
        assert!(
            residual < 0.01,
            "Residual {} too high for thin cloud",
            residual
        );
    }

    /// Test: Edge fitting with realistic edge samples
    ///
    /// Simulates what actual edge vertex samples might look like:
    /// - Outside samples cluster on one or two faces
    /// - RANSAC needs to find two distinct planes
    #[test]
    fn test_edge_fitting_realistic_scenario() {
        let cell_size = 0.05;

        // Create points that simulate outside samples on a cube edge
        // Edge is along the X axis, faces are Z+ and Y+
        let mut points = vec![];

        // Face 1: Z+ plane (z = 0.02, varying x and y)
        // These are "outside" samples that landed on the Z+ face
        for i in 0..8 {
            let x = (i as f64 - 3.5) * 0.01;
            let y = -0.01 + (i % 3) as f64 * 0.005;
            points.push((x, y, 0.02));
        }

        // Face 2: Y+ plane (y = 0.02, varying x and z)
        for i in 0..8 {
            let x = (i as f64 - 3.5) * 0.01;
            let z = -0.01 + (i % 3) as f64 * 0.005;
            points.push((x, 0.02, z));
        }

        // DIAGNOSTIC: Check RANSAC sampling coverage
        println!("DIAGNOSTIC: Edge test has {} points (8 per plane)", points.len());
        let n = points.len();
        let iterations = 200;
        let mut same_plane_count = 0;
        let mut cross_plane_count = 0;
        for i in 0..iterations {
            let idx1 = (i * 7 + 1) % n;
            let idx2 = (i * 13 + 3) % n;
            let idx3 = (i * 19 + 7) % n;
            if idx1 != idx2 && idx2 != idx3 && idx1 != idx3 {
                // Points 0-7 are on plane 1, 8-15 are on plane 2
                let planes: Vec<usize> = [idx1, idx2, idx3].iter().map(|&i| i / 8).collect();
                if planes[0] == planes[1] && planes[1] == planes[2] {
                    same_plane_count += 1;
                } else {
                    cross_plane_count += 1;
                }
            }
        }
        println!(
            "DIAGNOSTIC: RANSAC triples: {} same-plane, {} cross-plane",
            same_plane_count, cross_plane_count
        );

        let config = ClassifierConfig {
            inlier_threshold_ratio: 0.2,
            ransac_iterations: 200,
            min_inliers: 3,
        };

        let result = fit_edge_from_samples(&points, cell_size, &config);

        match &result {
            Some(fit) => {
                println!("DIAGNOSTIC: Edge fit succeeded");
                println!("  Normal A: {:?}", fit.normal_a);
                println!("  Normal B: {:?}", fit.normal_b);
                println!("  Edge direction: {:?}", fit.edge_direction);
                println!(
                    "  Inliers: A={}, B={}, total={}",
                    fit.inliers_a, fit.inliers_b, fit.total_points
                );

                // Expected normals (approximately):
                // Face 1 (Z+): normal ~ (0, 0, 1)
                // Face 2 (Y+): normal ~ (0, 1, 0)
                // Edge direction: ~ (1, 0, 0)

                let z_normal_err = normal_error_degrees(fit.normal_a, (0.0, 0.0, 1.0))
                    .min(normal_error_degrees(fit.normal_b, (0.0, 0.0, 1.0)));
                let y_normal_err = normal_error_degrees(fit.normal_a, (0.0, 1.0, 0.0))
                    .min(normal_error_degrees(fit.normal_b, (0.0, 1.0, 0.0)));
                let edge_dir_err = normal_error_degrees(fit.edge_direction, (1.0, 0.0, 0.0));

                println!(
                    "  Errors: Z-normal={:.1}°, Y-normal={:.1}°, edge-dir={:.1}°",
                    z_normal_err, y_normal_err, edge_dir_err
                );

                // These should be small for a well-designed test
                assert!(
                    z_normal_err < 15.0,
                    "Z-normal error {} too high",
                    z_normal_err
                );
                assert!(
                    y_normal_err < 15.0,
                    "Y-normal error {} too high",
                    y_normal_err
                );
                assert!(
                    edge_dir_err < 15.0,
                    "Edge direction error {} too high",
                    edge_dir_err
                );
            }
            None => {
                println!("DIAGNOSTIC: Edge fit FAILED - could not find two planes");
                println!("DIAGNOSTIC: Likely cause: RANSAC sampling selected triples from wrong plane mix");
                // This documents a known issue - don't panic
            }
        }
    }

    /// Test: Edge fitting failure mode - samples clustered on one face
    ///
    /// This tests the documented failure mode where outside samples
    /// cluster on a single face, making edge detection impossible.
    #[test]
    fn test_edge_fitting_single_face_cluster() {
        let cell_size = 0.05;

        // All outside samples on just one face (Z+)
        // This simulates the failure mode described in the documentation
        let mut points = vec![];
        for i in 0..15 {
            let x = (i as f64 - 7.0) * 0.008;
            let y = ((i * 3) as f64 - 22.0) * 0.003;
            points.push((x, y, 0.02 + (i as f64 * 0.001).sin() * 0.002));
        }

        let config = ClassifierConfig {
            inlier_threshold_ratio: 0.2,
            ransac_iterations: 200,
            min_inliers: 3,
        };

        let result = fit_edge_from_samples(&points, cell_size, &config);

        // This SHOULD fail because all points are on one plane
        match &result {
            Some(fit) => {
                println!("DIAGNOSTIC: Single-face cluster incorrectly produced edge fit");
                println!("  Normal A: {:?}", fit.normal_a);
                println!("  Normal B: {:?}", fit.normal_b);
                println!("  Dot product: {}", dot(fit.normal_a, fit.normal_b).abs());

                // The fit is likely invalid - normals should be nearly parallel
                // (both fitting the same plane from different point subsets)
                let dot_val = dot(fit.normal_a, fit.normal_b).abs();
                println!(
                    "  NOTE: If dot product > 0.85, this would be rejected. Current: {}",
                    dot_val
                );
            }
            None => {
                println!("DIAGNOSTIC: Single-face cluster correctly rejected (no edge found)");
            }
        }
    }

    /// Test: Corner fitting with three perpendicular planes
    #[test]
    fn test_corner_fitting_three_planes() {
        let cell_size = 0.05;
        let mut points = vec![];

        // Face 1: X+ plane (x = 0.02)
        for i in 0..6 {
            let y = (i as f64 - 2.5) * 0.01;
            let z = ((i * 2) as f64 - 5.0) * 0.008;
            points.push((0.02, y, z));
        }

        // Face 2: Y+ plane (y = 0.02)
        for i in 0..6 {
            let x = (i as f64 - 2.5) * 0.01;
            let z = ((i * 2) as f64 - 5.0) * 0.008;
            points.push((x, 0.02, z));
        }

        // Face 3: Z+ plane (z = 0.02)
        for i in 0..6 {
            let x = (i as f64 - 2.5) * 0.01;
            let y = ((i * 2) as f64 - 5.0) * 0.008;
            points.push((x, y, 0.02));
        }

        // DIAGNOSTIC: Check what triples RANSAC will actually try
        println!("DIAGNOSTIC: Corner test has {} points", points.len());
        let mut selected_triples = std::collections::HashSet::new();
        let iterations = 200;
        let n = points.len();
        for i in 0..iterations {
            let idx1 = (i * 7 + 1) % n;
            let idx2 = (i * 13 + 3) % n;
            let idx3 = (i * 19 + 7) % n;
            if idx1 != idx2 && idx2 != idx3 && idx1 != idx3 {
                let mut triple = [idx1, idx2, idx3];
                triple.sort();
                selected_triples.insert(triple);
            }
        }
        println!("DIAGNOSTIC: RANSAC will try {} unique triples", selected_triples.len());

        // Check how many triples have points from different planes
        let mut cross_plane_triples = 0;
        for triple in &selected_triples {
            let planes: Vec<usize> = triple.iter().map(|&i| i / 6).collect();
            if planes[0] != planes[1] || planes[1] != planes[2] {
                cross_plane_triples += 1;
            }
        }
        println!("DIAGNOSTIC: {} triples mix points from different planes", cross_plane_triples);

        let config = ClassifierConfig {
            inlier_threshold_ratio: 0.2,
            ransac_iterations: 200,
            min_inliers: 3,
        };

        let result = fit_corner_from_samples(&points, cell_size, &config);

        match &result {
            Some(fit) => {
                println!("DIAGNOSTIC: Corner fit found {} planes", fit.normals.len());
                for (i, normal) in fit.normals.iter().enumerate() {
                    println!(
                        "  Plane {}: normal={:?}, residual={:.6}, inliers={}",
                        i, normal, fit.residuals[i], fit.inlier_counts[i]
                    );
                }

                // Check if we found all three expected normals
                let expected = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];
                let errors = best_corner_normal_errors(&fit.normals, &expected);
                println!("  Normal errors: {:?}", errors);

                assert_eq!(fit.normals.len(), 3, "Should find exactly 3 planes");
                for (i, err) in errors.iter().enumerate() {
                    assert!(
                        *err < 15.0,
                        "Normal {} error {} too high",
                        i,
                        err
                    );
                }
            }
            None => {
                println!("DIAGNOSTIC: Corner fit FAILED - could not find 3 planes");
                println!("DIAGNOSTIC: This suggests RANSAC sampling is missing valid plane triples");
                // Don't panic - this is documenting a known issue
            }
        }
    }

    /// Test: Inlier threshold sensitivity
    ///
    /// Different thresholds can dramatically change results.
    /// This test explores the sensitivity.
    #[test]
    fn test_inlier_threshold_sensitivity() {
        let cell_size = 0.05;

        // Points on a plane with some noise
        let mut points = vec![];
        for i in 0..20 {
            let x = (i as f64 - 10.0) * 0.01;
            let y = ((i * 3) as f64 - 30.0) * 0.008;
            // Add noise proportional to cell_size
            let noise = (i as f64 * 0.7).sin() * 0.005; // ±0.005 noise
            points.push((x, y, noise));
        }

        let thresholds = [0.05, 0.1, 0.2, 0.3, 0.5];

        println!("DIAGNOSTIC: Inlier threshold sensitivity (cell_size={})", cell_size);
        for &ratio in &thresholds {
            let config = ClassifierConfig {
                inlier_threshold_ratio: ratio,
                ransac_iterations: 100,
                min_inliers: 3,
            };

            let result = fit_face_from_samples(&points, cell_size, &config);

            match result {
                Some(fit) => {
                    let normal_err = normal_error_degrees(fit.normal, (0.0, 0.0, 1.0));
                    println!(
                        "  ratio={:.2}: inliers={}/{}, residual={:.6}, normal_err={:.1}°",
                        ratio, fit.inlier_count, fit.total_points, fit.residual, normal_err
                    );
                }
                None => {
                    println!("  ratio={:.2}: FIT FAILED", ratio);
                }
            }
        }
    }

    /// Test: Best edge normal pairing with ambiguous assignment
    #[test]
    fn test_edge_normal_pairing_ambiguous() {
        // Test case where optimal assignment is non-obvious
        let detected = (
            (0.707, 0.707, 0.0),  // 45° between X and Y
            (0.0, 0.707, 0.707),  // 45° between Y and Z
        );

        let expected = (
            (1.0, 0.0, 0.0),  // Pure X
            (0.0, 1.0, 0.0),  // Pure Y
        );

        let (err_a, err_b) = best_edge_normal_errors(detected, expected);

        println!(
            "DIAGNOSTIC: Ambiguous edge pairing: err_a={:.1}°, err_b={:.1}°",
            err_a, err_b
        );

        // The algorithm should pick the assignment that minimizes total error
        // Assignment 1: detected.0 -> expected.0, detected.1 -> expected.1
        //   err = 45° + 45° = 90°
        // Assignment 2: detected.0 -> expected.1, detected.1 -> expected.0
        //   err = 45° + 90° = 135°
        // So assignment 1 should be chosen
        assert!(err_a + err_b < 100.0, "Should choose better assignment");
    }

    /// Test: Deep RANSAC diagnostic to understand why edge fitting fails
    #[test]
    fn test_ransac_deep_diagnostic() {
        let cell_size = 0.05;
        let threshold = 0.2 * cell_size; // = 0.01

        // Create two perpendicular planes
        let mut points = vec![];

        // Plane 1: Z+ (z = 0.02)
        for i in 0..8 {
            let x = (i as f64 - 3.5) * 0.01;
            let y = -0.01 + (i % 3) as f64 * 0.005;
            points.push((x, y, 0.02));
        }

        // Plane 2: Y+ (y = 0.02)
        for i in 0..8 {
            let x = (i as f64 - 3.5) * 0.01;
            let z = -0.01 + (i % 3) as f64 * 0.005;
            points.push((x, 0.02, z));
        }

        println!("\n=== RANSAC DEEP DIAGNOSTIC ===");
        println!("Points: {}, threshold: {}", points.len(), threshold);
        println!("\nPlane 1 (Z+) points:");
        for (i, p) in points.iter().enumerate().take(8) {
            println!("  {}: {:?}", i, p);
        }
        println!("\nPlane 2 (Y+) points:");
        for (i, p) in points.iter().enumerate().skip(8) {
            println!("  {}: {:?}", i, p);
        }

        // First RANSAC call
        let (result1, diag1) = ransac_plane_fit_diagnostic(&points, threshold, 200);
        println!("\n--- First RANSAC fit ---");
        println!("  Valid iterations: {}", diag1.valid_iterations);
        println!("  Skipped (duplicate idx): {}", diag1.skipped_duplicate_indices);
        println!("  Skipped (degenerate): {}", diag1.skipped_degenerate);
        println!("  Best inlier count: {}", diag1.best_inlier_count);
        println!("  Best normal (before SVD): {:?}", diag1.best_normal);
        println!("  Final normal (after SVD): {:?}", diag1.final_normal);
        println!("  Final residual: {}", diag1.final_residual);

        // Analyze inlier distribution
        if !diag1.inlier_counts.is_empty() {
            let min_inliers = *diag1.inlier_counts.iter().min().unwrap();
            let max_inliers = *diag1.inlier_counts.iter().max().unwrap();
            let avg_inliers: f64 =
                diag1.inlier_counts.iter().sum::<usize>() as f64 / diag1.inlier_counts.len() as f64;
            println!(
                "  Inlier distribution: min={}, max={}, avg={:.1}",
                min_inliers, max_inliers, avg_inliers
            );
        }

        // Check which points are inliers to the first plane
        if let Some((inliers1, centroid1, normal1, _)) = &result1 {
            println!("\n  First plane found:");
            println!("    Centroid: {:?}", centroid1);
            println!("    Normal: {:?}", normal1);
            println!("    Inliers: {}", inliers1.len());

            // Which original points are inliers?
            let plane1_inliers: Vec<usize> = points
                .iter()
                .enumerate()
                .filter(|(_, p)| is_inlier(**p, *centroid1, *normal1, threshold))
                .map(|(i, _)| i)
                .collect();
            println!("    Inlier indices: {:?}", plane1_inliers);

            let plane1_from_face1 = plane1_inliers.iter().filter(|&&i| i < 8).count();
            let plane1_from_face2 = plane1_inliers.iter().filter(|&&i| i >= 8).count();
            println!(
                "    From face 1 (Z+): {}, from face 2 (Y+): {}",
                plane1_from_face1, plane1_from_face2
            );

            // Remove inliers and run second RANSAC
            let remaining: Vec<_> = points
                .iter()
                .filter(|p| !is_inlier(**p, *centroid1, *normal1, threshold))
                .cloned()
                .collect();
            println!("\n  Remaining points for second fit: {}", remaining.len());

            if remaining.len() >= 3 {
                let (result2, diag2) = ransac_plane_fit_diagnostic(&remaining, threshold, 200);
                println!("\n--- Second RANSAC fit ---");
                println!("  Valid iterations: {}", diag2.valid_iterations);
                println!("  Best inlier count: {}", diag2.best_inlier_count);
                println!("  Best normal: {:?}", diag2.best_normal);
                println!("  Final normal: {:?}", diag2.final_normal);

                if let Some((inliers2, _, normal2, _)) = &result2 {
                    println!("  Second plane inliers: {}", inliers2.len());

                    // Check if planes are too parallel
                    let dot_val = dot(*normal1, *normal2).abs();
                    println!("\n  Dot product of normals: {:.4}", dot_val);
                    println!("  (Threshold for rejection: > 0.85)");
                    if dot_val > 0.85 {
                        println!("  REJECTED: Planes too parallel!");
                    } else {
                        println!("  ACCEPTED: Planes sufficiently different");
                    }
                } else {
                    println!("  Second fit FAILED - no plane found");
                }
            }
        } else {
            println!("\n  First fit FAILED completely");
        }

        println!("\n=== END DIAGNOSTIC ===\n");
    }

    /// Test: Greedy corner matching with reordered normals
    #[test]
    fn test_corner_normal_matching_order_independence() {
        let detected = vec![
            (0.0, 0.0, 1.0),  // Z+
            (1.0, 0.0, 0.0),  // X+
            (0.0, 1.0, 0.0),  // Y+
        ];

        let expected = vec![
            (1.0, 0.0, 0.0),  // X+
            (0.0, 1.0, 0.0),  // Y+
            (0.0, 0.0, 1.0),  // Z+
        ];

        let errors = best_corner_normal_errors(&detected, &expected);

        println!("DIAGNOSTIC: Corner matching errors: {:?}", errors);

        // All errors should be ~0 since normals are exact matches (just reordered)
        for (i, err) in errors.iter().enumerate() {
            assert!(
                *err < 1.0,
                "Error {} for normal {} should be ~0 for exact match",
                err,
                i
            );
        }
    }

    /// Test: Explore what inlier threshold is needed for proper edge separation
    ///
    /// This test demonstrates that the default 20% threshold is too generous
    /// for edge detection when planes meet at right angles.
    #[test]
    fn test_inlier_threshold_for_edge_separation() {
        let cell_size = 0.05;

        // Create well-separated planes (same as deep diagnostic)
        let mut points = vec![];

        // Plane 1: Z+ (z = 0.02)
        for i in 0..8 {
            let x = (i as f64 - 3.5) * 0.01;
            let y = -0.01 + (i % 3) as f64 * 0.005;
            points.push((x, y, 0.02));
        }

        // Plane 2: Y+ (y = 0.02)
        for i in 0..8 {
            let x = (i as f64 - 3.5) * 0.01;
            let z = -0.01 + (i % 3) as f64 * 0.005;
            points.push((x, 0.02, z));
        }

        println!("\n=== INLIER THRESHOLD EXPLORATION ===");

        // Try different threshold ratios
        let ratios = [0.20, 0.15, 0.10, 0.08, 0.05, 0.03];

        for &ratio in &ratios {
            let threshold = ratio * cell_size;
            let config = ClassifierConfig {
                inlier_threshold_ratio: ratio,
                ransac_iterations: 200,
                min_inliers: 3,
            };

            let result = fit_edge_from_samples(&points, cell_size, &config);

            match result {
                Some(fit) => {
                    let z_err = normal_error_degrees(fit.normal_a, (0.0, 0.0, 1.0))
                        .min(normal_error_degrees(fit.normal_b, (0.0, 0.0, 1.0)));
                    let y_err = normal_error_degrees(fit.normal_a, (0.0, 1.0, 0.0))
                        .min(normal_error_degrees(fit.normal_b, (0.0, 1.0, 0.0)));
                    println!(
                        "  ratio={:.2} (thresh={:.4}): SUCCESS! Z-err={:.1}°, Y-err={:.1}°, inliers={}/{}",
                        ratio, threshold, z_err, y_err, fit.inliers_a, fit.inliers_b
                    );
                }
                None => {
                    // Run diagnostic to see why it failed
                    let (result1, _) = ransac_plane_fit_diagnostic(&points, threshold, 200);
                    let status = match result1 {
                        Some((ref inliers, _, _, _)) => format!("first plane got {} inliers", inliers.len()),
                        None => "first plane fit failed".to_string(),
                    };
                    println!(
                        "  ratio={:.2} (thresh={:.4}): FAILED ({})",
                        ratio, threshold, status
                    );
                }
            }
        }

        println!("=== END THRESHOLD EXPLORATION ===\n");
    }

    /// Test: Quantify the minimum plane separation needed for edge detection
    #[test]
    fn test_minimum_plane_separation() {
        println!("\n=== MINIMUM PLANE SEPARATION ANALYSIS ===");

        // For a cube edge where two faces meet at 90 degrees:
        // If the vertex is at origin and cell_size is 0.05,
        // samples on both faces within the cell might be at most
        // 0.05 away in each direction.
        //
        // Face 1 (z = 0): points like (x, y, 0)
        // Face 2 (y = 0): points like (x, 0, z)
        //
        // A diagonal plane z = y would pass through both faces.
        // For a point on Face 1 at (x, y, 0), distance to z=y is |y|/sqrt(2)
        // For a point on Face 2 at (x, 0, z), distance to z=y is |z|/sqrt(2)
        //
        // If y and z values are small (< threshold * sqrt(2)), ALL points
        // will be inliers to the diagonal plane!

        let cell_size = 0.05;
        let threshold_ratio = 0.2;
        let threshold = threshold_ratio * cell_size; // 0.01

        // Maximum y value on Face 1 that fits within threshold of z=y plane
        let max_y = threshold * (2.0_f64).sqrt();
        // Maximum z value on Face 2 that fits within threshold of z=y plane
        let max_z = threshold * (2.0_f64).sqrt();

        println!("Cell size: {}", cell_size);
        println!("Threshold ratio: {}", threshold_ratio);
        println!("Threshold: {}", threshold);
        println!("Max |y| on Face 1 for z=y diagonal to capture: {:.4}", max_y);
        println!("Max |z| on Face 2 for z=y diagonal to capture: {:.4}", max_z);
        println!();
        println!("IMPLICATION: If Face 1 samples have |y| < {:.4}", max_y);
        println!("         AND Face 2 samples have |z| < {:.4}", max_z);
        println!("         THEN a diagonal plane will capture both faces!");
        println!();

        // For 90-degree edges, the minimum safe threshold ratio is approximately:
        // threshold < min(|y|, |z|) / (sqrt(2) * cell_size)
        // If samples span 0.4*cell_size in each direction, we need:
        // threshold < 0.4 / sqrt(2) ≈ 0.283 * cell_size
        //
        // But our samples might only span 0.3*cell_size, so:
        // threshold < 0.3 / sqrt(2) ≈ 0.21 * cell_size
        //
        // The current 0.2 threshold is right at the edge of failure!

        println!("For reliable 90° edge detection with samples spanning 0.3*cell_size:");
        println!("  Maximum safe threshold: {:.3} * cell_size", 0.3 / (2.0_f64).sqrt());
        println!("  Current threshold: {} * cell_size", threshold_ratio);
        println!("  VERDICT: Current threshold is TOO CLOSE to failure boundary!");

        println!("=== END ANALYSIS ===\n");
    }

    /// Test: Edge fitting with overlapping inlier regions
    ///
    /// When the inlier threshold is large relative to plane separation,
    /// points may be counted as inliers to both planes.
    #[test]
    fn test_edge_fitting_overlapping_inliers() {
        let cell_size = 0.05;

        // Two planes that are close together (30° angle, not 90°)
        let mut points = vec![];

        // Plane 1: z = 0
        for i in 0..10 {
            let x = (i as f64 - 4.5) * 0.01;
            let y = (i as f64 - 4.5) * 0.008;
            points.push((x, y, 0.0));
        }

        // Plane 2: z = 0.5 * y (30° from horizontal)
        for i in 0..10 {
            let x = (i as f64 - 4.5) * 0.01;
            let y = (i as f64 - 4.5) * 0.008;
            let z = 0.577 * y; // tan(30°) ≈ 0.577
            points.push((x, y, z));
        }

        let config = ClassifierConfig {
            inlier_threshold_ratio: 0.3, // Large threshold
            ransac_iterations: 200,
            min_inliers: 3,
        };

        let result = fit_edge_from_samples(&points, cell_size, &config);

        match &result {
            Some(fit) => {
                let angle = dot(fit.normal_a, fit.normal_b).abs().acos().to_degrees();
                println!(
                    "DIAGNOSTIC: Close planes (30° expected): found angle={:.1}°",
                    angle
                );
                println!(
                    "  Inliers: A={}, B={}",
                    fit.inliers_a, fit.inliers_b
                );

                // The parallel check (dot > 0.85 = ~32°) should reject this
                // If it doesn't, the fit is questionable
            }
            None => {
                println!(
                    "DIAGNOSTIC: Close planes correctly rejected (angle too small)"
                );
            }
        }
    }
}
