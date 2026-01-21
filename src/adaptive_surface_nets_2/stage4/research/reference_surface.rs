//! Reference Implementation: Find Nearest Flat Surface
//!
//! Given a point P near the surface, finds:
//! - The closest point on the surface
//! - The surface normal at that point
//! - Confidence metric (how flat is this region?)
//!
//! # Approach
//!
//! Dense spherical probing + plane fitting:
//! 1. Probe in many directions from P
//! 2. Binary search each direction to find surface crossing
//! 3. Fit plane to all surface points using least squares
//! 4. Residual indicates flatness (low residual = flat surface)
//!
//! This is expensive (50-200+ samples) but provably correct for flat surfaces.

use super::sample_cache::{find_crossing_in_direction, SampleCache};

/// Result of reference surface finding
#[derive(Clone, Debug)]
pub struct SurfaceFindingResult {
    /// The closest point found on the surface
    pub closest_point: (f64, f64, f64),
    /// The surface normal at the closest point (outward-facing)
    pub normal: (f64, f64, f64),
    /// RMS residual from plane fitting (lower = flatter)
    pub flatness_residual: f64,
    /// Number of surface points found
    pub points_found: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Configuration for reference surface finding
#[derive(Clone, Debug)]
pub struct SurfaceFindingConfig {
    /// Number of probe directions (more = more accurate but slower)
    pub num_probes: usize,
    /// Maximum distance to search for surface crossing
    pub max_search_distance: f64,
    /// Step size for initial crossing search
    pub search_step: f64,
    /// Number of binary search iterations for crossing refinement
    pub binary_iterations: usize,
}

impl Default for SurfaceFindingConfig {
    fn default() -> Self {
        Self {
            num_probes: 50,
            max_search_distance: 1.0,
            search_step: 0.05,
            binary_iterations: 20,
        }
    }
}

/// Find the nearest flat surface from a given point.
///
/// Uses dense spherical probing followed by plane fitting.
/// This is an expensive reference implementation (50+ samples).
///
/// # Arguments
/// * `point` - Starting point for the search
/// * `cache` - Sample cache for the underlying sampler
/// * `config` - Configuration parameters
///
/// # Returns
/// SurfaceFindingResult with the closest point, normal, and confidence metrics
pub fn reference_find_nearest_surface<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    config: &SurfaceFindingConfig,
) -> SurfaceFindingResult
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Generate uniformly distributed directions on a sphere
    let directions = generate_sphere_directions(config.num_probes);

    // Find surface crossings in each direction
    let mut surface_points = Vec::with_capacity(config.num_probes);

    for dir in &directions {
        if let Some((crossing, _dist)) = find_crossing_in_direction(
            cache,
            point,
            *dir,
            config.max_search_distance,
            config.search_step,
            config.binary_iterations,
        ) {
            surface_points.push(crossing);
        }
    }

    if surface_points.is_empty() {
        return SurfaceFindingResult {
            closest_point: point,
            normal: (0.0, 0.0, 1.0),
            flatness_residual: f64::INFINITY,
            points_found: 0,
            confidence: 0.0,
        };
    }

    // Fit plane to surface points
    let (centroid, normal, residual) = fit_plane_to_points(&surface_points);

    // Orient normal to point away from the query point
    let to_centroid = sub(centroid, point);
    let oriented_normal = if dot(normal, to_centroid) < 0.0 {
        (-normal.0, -normal.1, -normal.2)
    } else {
        normal
    };

    // Find closest point to query point among surface points
    let mut closest = surface_points[0];
    let mut min_dist = distance_sq(point, closest);
    for &sp in &surface_points[1..] {
        let d = distance_sq(point, sp);
        if d < min_dist {
            min_dist = d;
            closest = sp;
        }
    }

    // Compute confidence based on:
    // 1. Number of points found (more is better)
    // 2. Residual (lower is better for flat surfaces)
    let point_ratio = surface_points.len() as f64 / config.num_probes as f64;
    let residual_score = (-residual * 100.0).exp(); // Exponential decay for residual
    let confidence = (point_ratio * 0.5 + residual_score * 0.5).clamp(0.0, 1.0);

    SurfaceFindingResult {
        closest_point: closest,
        normal: oriented_normal,
        flatness_residual: residual,
        points_found: surface_points.len(),
        confidence,
    }
}

/// Generate uniformly distributed directions on a sphere using Fibonacci lattice
fn generate_sphere_directions(n: usize) -> Vec<(f64, f64, f64)> {
    let mut directions = Vec::with_capacity(n);
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let angle_increment = std::f64::consts::PI * 2.0 * golden_ratio;

    for i in 0..n {
        // Fibonacci lattice for even distribution
        let t = (i as f64 + 0.5) / n as f64;
        let phi = angle_increment * i as f64;

        // Convert to spherical coordinates
        let theta = (1.0 - 2.0 * t).acos();
        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();

        directions.push((x, y, z));
    }

    directions
}

/// Fit a plane to a set of 3D points using SVD (via eigendecomposition of covariance)
///
/// Returns (centroid, normal, rms_residual)
fn fit_plane_to_points(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
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

    // Center the points
    let centered: Vec<(f64, f64, f64)> = points.iter().map(|p| sub(*p, centroid)).collect();

    // Build covariance matrix (symmetric 3x3)
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

    // Find eigenvector corresponding to smallest eigenvalue
    // Using power iteration on (max_eigenvalue * I - cov) to find min eigenvector
    let normal = find_smallest_eigenvector(&cov);

    // Compute RMS residual (distance to fitted plane)
    let mut residual_sum = 0.0;
    for p in &centered {
        let dist = dot(*p, normal).abs();
        residual_sum += dist * dist;
    }
    let rms_residual = (residual_sum / n).sqrt();

    (centroid, normal, rms_residual)
}

/// Find the eigenvector corresponding to the smallest eigenvalue of a 3x3 symmetric matrix
///
/// Uses iterative approach: find largest eigenvector first, then deflate, repeat
fn find_smallest_eigenvector(cov: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    // Estimate max eigenvalue using Gershgorin circle theorem
    let max_eigenvalue_estimate = cov
        .iter()
        .enumerate()
        .map(|(i, row)| row[i].abs() + row.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    // Create shifted matrix: M = lambda_max * I - cov
    // The smallest eigenvector of cov is the largest eigenvector of M
    let shift = max_eigenvalue_estimate + 1.0;
    let shifted = [
        [shift - cov[0][0], -cov[0][1], -cov[0][2]],
        [-cov[1][0], shift - cov[1][1], -cov[1][2]],
        [-cov[2][0], -cov[2][1], shift - cov[2][2]],
    ];

    // Power iteration to find largest eigenvector of shifted matrix
    let mut v = (1.0, 1.0, 1.0);
    for _ in 0..50 {
        // Matrix-vector multiply
        let new_v = (
            shifted[0][0] * v.0 + shifted[0][1] * v.1 + shifted[0][2] * v.2,
            shifted[1][0] * v.0 + shifted[1][1] * v.1 + shifted[1][2] * v.2,
            shifted[2][0] * v.0 + shifted[2][1] * v.1 + shifted[2][2] * v.2,
        );

        // Normalize
        let len = (new_v.0 * new_v.0 + new_v.1 * new_v.1 + new_v.2 * new_v.2).sqrt();
        if len < 1e-12 {
            break;
        }
        v = (new_v.0 / len, new_v.1 / len, new_v.2 / len);
    }

    v
}

// Vector math helpers

fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn distance_sq(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sphere(x: f64, y: f64, z: f64) -> f32 {
        (1.0 - x * x - y * y - z * z) as f32
    }

    fn unit_cube(x: f64, y: f64, z: f64) -> f32 {
        let half = 0.5;
        let inside_x = x.abs() < half;
        let inside_y = y.abs() < half;
        let inside_z = z.abs() < half;
        if inside_x && inside_y && inside_z {
            1.0
        } else {
            -1.0
        }
    }

    #[test]
    fn test_sphere_directions_coverage() {
        let dirs = generate_sphere_directions(100);

        // All should be unit length
        for dir in &dirs {
            let len = (dir.0 * dir.0 + dir.1 * dir.1 + dir.2 * dir.2).sqrt();
            assert!((len - 1.0).abs() < 1e-10);
        }

        // Should have reasonable coverage (not all in same hemisphere)
        let pos_x: usize = dirs.iter().filter(|d| d.0 > 0.0).count();
        let pos_y: usize = dirs.iter().filter(|d| d.1 > 0.0).count();
        let pos_z: usize = dirs.iter().filter(|d| d.2 > 0.0).count();

        // Each hemisphere should have roughly half the points
        assert!(pos_x > 30 && pos_x < 70, "X coverage: {}", pos_x);
        assert!(pos_y > 30 && pos_y < 70, "Y coverage: {}", pos_y);
        assert!(pos_z > 30 && pos_z < 70, "Z coverage: {}", pos_z);
    }

    #[test]
    fn test_plane_fitting() {
        // Points on z=0 plane with some noise
        let points = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.5, 0.5, 0.0),
        ];

        let (centroid, normal, residual) = fit_plane_to_points(&points);

        // Centroid should be near (0.5, 0.5, 0.0)
        assert!((centroid.0 - 0.5).abs() < 0.01);
        assert!((centroid.1 - 0.5).abs() < 0.01);
        assert!(centroid.2.abs() < 0.01);

        // Normal should be (0, 0, ±1)
        assert!(normal.0.abs() < 0.01);
        assert!(normal.1.abs() < 0.01);
        assert!((normal.2.abs() - 1.0).abs() < 0.01);

        // Residual should be near zero for points on a plane
        assert!(residual < 0.001);
    }

    #[test]
    fn test_find_surface_on_sphere() {
        let cache = SampleCache::new(unit_sphere);

        // Point inside sphere
        let config = SurfaceFindingConfig {
            num_probes: 50,
            max_search_distance: 2.0,
            search_step: 0.1,
            binary_iterations: 15,
        };

        let result = reference_find_nearest_surface((0.0, 0.0, 0.0), &cache, &config);

        // Should find points
        assert!(result.points_found > 30, "Found {} points", result.points_found);

        // For a sphere from center, all points are at radius 1
        // The "flatness" should be poor because sphere is curved
        // (residual will be high)
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_find_surface_on_cube_face() {
        let cache = SampleCache::new(unit_cube);

        // Point just inside a face, should find a flat surface
        let config = SurfaceFindingConfig {
            num_probes: 50,
            max_search_distance: 0.2,
            search_step: 0.02,
            binary_iterations: 15,
        };

        let result = reference_find_nearest_surface((0.4, 0.0, 0.0), &cache, &config);

        // Should find points
        assert!(result.points_found > 0, "Found {} points", result.points_found);

        // Normal should be close to +X direction (we're near +X face)
        let normal_x = result.normal.0;
        assert!(
            normal_x.abs() > 0.9,
            "Expected normal near X axis, got {:?}",
            result.normal
        );
    }
}
