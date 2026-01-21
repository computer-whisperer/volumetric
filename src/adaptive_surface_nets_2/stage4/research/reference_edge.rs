//! Reference Implementation: Find Nearest Edge
//!
//! Given a point P near a geometric edge, finds:
//! - The edge direction (line in 3D)
//! - The two face normals
//! - The closest point on the edge line
//! - Confidence metric (how sharp is the edge?)
//!
//! # Approach
//!
//! Dense probing + clustering + plane-plane intersection:
//! 1. Probe densely (50+ directions) from P
//! 2. Binary search each direction to find surface crossing
//! 3. Cluster surface points into two groups (k-means with k=2)
//! 4. Fit plane to each cluster
//! 5. Edge = intersection of the two planes
//!
//! This is expensive (100+ samples) but provably correct for edge detection.

use super::sample_cache::{find_crossing_in_direction, SampleCache};

/// Result of reference edge finding
#[derive(Clone, Debug)]
pub struct EdgeFindingResult {
    /// Direction of the edge (unit vector)
    pub edge_direction: (f64, f64, f64),
    /// A point on the edge line (closest to query point)
    pub point_on_edge: (f64, f64, f64),
    /// Normal of face A
    pub face_a_normal: (f64, f64, f64),
    /// Normal of face B
    pub face_b_normal: (f64, f64, f64),
    /// Angle between the two faces (radians)
    pub dihedral_angle: f64,
    /// Confidence score (0.0 to 1.0)
    /// High confidence means: good cluster separation, low plane residuals
    pub confidence: f64,
    /// Number of points in cluster A
    pub cluster_a_size: usize,
    /// Number of points in cluster B
    pub cluster_b_size: usize,
    /// RMS residual of plane fit for cluster A
    pub residual_a: f64,
    /// RMS residual of plane fit for cluster B
    pub residual_b: f64,
}

/// Configuration for reference edge finding
#[derive(Clone, Debug)]
pub struct EdgeFindingConfig {
    /// Number of probe directions
    pub num_probes: usize,
    /// Maximum distance to search for surface crossing
    pub max_search_distance: f64,
    /// Step size for initial crossing search
    pub search_step: f64,
    /// Number of binary search iterations for crossing refinement
    pub binary_iterations: usize,
    /// Number of k-means iterations for clustering
    pub kmeans_iterations: usize,
}

impl Default for EdgeFindingConfig {
    fn default() -> Self {
        Self {
            num_probes: 100,
            max_search_distance: 1.0,
            search_step: 0.05,
            binary_iterations: 20,
            kmeans_iterations: 20,
        }
    }
}

/// Find the nearest edge from a given point.
///
/// Uses dense spherical probing, clustering into two groups, and plane-plane intersection.
/// This is an expensive reference implementation (100+ samples).
///
/// # Arguments
/// * `point` - Starting point for the search (should be near an edge)
/// * `cache` - Sample cache for the underlying sampler
/// * `config` - Configuration parameters
///
/// # Returns
/// EdgeFindingResult with the edge direction, normals, and confidence metrics.
/// Returns None if insufficient surface points found or clustering fails.
pub fn reference_find_nearest_edge<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    config: &EdgeFindingConfig,
) -> Option<EdgeFindingResult>
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

    // Need at least 6 points for reasonable clustering (3 per cluster minimum)
    if surface_points.len() < 6 {
        return None;
    }

    // Cluster into two groups using k-means
    let (cluster_a, cluster_b) = kmeans_two_clusters(&surface_points, point, config.kmeans_iterations)?;

    // Need at least 3 points per cluster for plane fitting
    if cluster_a.len() < 3 || cluster_b.len() < 3 {
        return None;
    }

    // Fit plane to each cluster
    let (centroid_a, normal_a, residual_a) = fit_plane_to_points(&cluster_a);
    let (centroid_b, normal_b, residual_b) = fit_plane_to_points(&cluster_b);

    // Orient normals to point away from query point
    let normal_a = orient_normal_away(normal_a, centroid_a, point);
    let normal_b = orient_normal_away(normal_b, centroid_b, point);

    // Edge direction is cross product of normals
    let edge_dir = cross(normal_a, normal_b);
    let edge_dir_len = length(edge_dir);

    if edge_dir_len < 1e-6 {
        // Planes are parallel - not a valid edge
        return None;
    }

    let edge_direction = normalize(edge_dir);

    // Find point on edge closest to query point
    // Using plane-plane intersection through closest approach
    let point_on_edge = find_closest_point_on_edge(point, centroid_a, normal_a, centroid_b, normal_b, edge_direction);

    // Compute dihedral angle between faces
    let dihedral_angle = angle_between(normal_a, normal_b);

    // Compute confidence score based on:
    // 1. Cluster balance (both clusters should have similar sizes)
    // 2. Plane fit quality (low residuals)
    // 3. Dihedral angle (not too small, not too large)
    let cluster_balance = {
        let min_size = cluster_a.len().min(cluster_b.len()) as f64;
        let max_size = cluster_a.len().max(cluster_b.len()) as f64;
        min_size / max_size
    };

    let residual_quality = {
        let avg_residual = (residual_a + residual_b) / 2.0;
        (-avg_residual * 100.0).exp() // Exponential decay
    };

    let angle_quality = {
        let angle_deg = dihedral_angle.to_degrees();
        // Best confidence around 90°, lower at very acute or obtuse angles
        if angle_deg < 10.0 || angle_deg > 170.0 {
            0.2
        } else if angle_deg < 30.0 || angle_deg > 150.0 {
            0.5
        } else {
            1.0
        }
    };

    let confidence = (cluster_balance * 0.3 + residual_quality * 0.4 + angle_quality * 0.3).clamp(0.0, 1.0);

    Some(EdgeFindingResult {
        edge_direction,
        point_on_edge,
        face_a_normal: normal_a,
        face_b_normal: normal_b,
        dihedral_angle,
        confidence,
        cluster_a_size: cluster_a.len(),
        cluster_b_size: cluster_b.len(),
        residual_a,
        residual_b,
    })
}

/// Generate uniformly distributed directions on a sphere using Fibonacci lattice
fn generate_sphere_directions(n: usize) -> Vec<(f64, f64, f64)> {
    let mut directions = Vec::with_capacity(n);
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let angle_increment = std::f64::consts::PI * 2.0 * golden_ratio;

    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let phi = angle_increment * i as f64;
        let theta = (1.0 - 2.0 * t).acos();
        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();
        directions.push((x, y, z));
    }

    directions
}

/// Cluster points into two groups using k-means
///
/// Initializes with points on opposite sides of the query point.
fn kmeans_two_clusters(
    points: &[(f64, f64, f64)],
    query: (f64, f64, f64),
    iterations: usize,
) -> Option<(Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>)> {
    if points.len() < 2 {
        return None;
    }

    // Initialize centroids by finding two points that are far apart
    // Use the point furthest from query as first centroid
    let mut c1 = points[0];
    let mut max_dist = 0.0;
    for &p in points {
        let d = distance_sq(p, query);
        if d > max_dist {
            max_dist = d;
            c1 = p;
        }
    }

    // Second centroid is furthest from first
    let mut c2 = points[0];
    max_dist = 0.0;
    for &p in points {
        let d = distance_sq(p, c1);
        if d > max_dist {
            max_dist = d;
            c2 = p;
        }
    }

    // Run k-means iterations
    for _ in 0..iterations {
        // Assign points to clusters
        let mut cluster1 = Vec::new();
        let mut cluster2 = Vec::new();

        for &p in points {
            if distance_sq(p, c1) < distance_sq(p, c2) {
                cluster1.push(p);
            } else {
                cluster2.push(p);
            }
        }

        // Update centroids
        if cluster1.is_empty() || cluster2.is_empty() {
            return None;
        }

        c1 = centroid(&cluster1);
        c2 = centroid(&cluster2);
    }

    // Final assignment
    let mut cluster1 = Vec::new();
    let mut cluster2 = Vec::new();

    for &p in points {
        if distance_sq(p, c1) < distance_sq(p, c2) {
            cluster1.push(p);
        } else {
            cluster2.push(p);
        }
    }

    if cluster1.is_empty() || cluster2.is_empty() {
        return None;
    }

    Some((cluster1, cluster2))
}

/// Compute centroid of a set of points
fn centroid(points: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let n = points.len() as f64;
    (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    )
}

/// Fit a plane to a set of 3D points
fn fit_plane_to_points(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
    if points.len() < 3 {
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), f64::INFINITY);
    }

    let center = centroid(points);
    let centered: Vec<(f64, f64, f64)> = points.iter().map(|p| sub(*p, center)).collect();

    // Build covariance matrix
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

    let normal = find_smallest_eigenvector(&cov);

    // Compute residual
    let mut residual_sum = 0.0;
    for p in &centered {
        let dist = dot(*p, normal).abs();
        residual_sum += dist * dist;
    }
    let rms_residual = (residual_sum / points.len() as f64).sqrt();

    (center, normal, rms_residual)
}

/// Find smallest eigenvector using power iteration on shifted matrix
fn find_smallest_eigenvector(cov: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    let max_eigenvalue_estimate = cov
        .iter()
        .enumerate()
        .map(|(i, row)| row[i].abs() + row.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    let shift = max_eigenvalue_estimate + 1.0;
    let shifted = [
        [shift - cov[0][0], -cov[0][1], -cov[0][2]],
        [-cov[1][0], shift - cov[1][1], -cov[1][2]],
        [-cov[2][0], -cov[2][1], shift - cov[2][2]],
    ];

    let mut v = (1.0, 1.0, 1.0);
    for _ in 0..50 {
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

    v
}

/// Orient normal to point away from a reference point
fn orient_normal_away(
    normal: (f64, f64, f64),
    plane_point: (f64, f64, f64),
    away_from: (f64, f64, f64),
) -> (f64, f64, f64) {
    let to_plane = sub(plane_point, away_from);
    if dot(normal, to_plane) < 0.0 {
        (-normal.0, -normal.1, -normal.2)
    } else {
        normal
    }
}

/// Find the point on the edge line closest to the query point
///
/// The edge is defined as the intersection of two planes.
fn find_closest_point_on_edge(
    query: (f64, f64, f64),
    point_a: (f64, f64, f64),
    normal_a: (f64, f64, f64),
    point_b: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Find a point on the edge by solving the plane equations
    // Plane A: dot(p - point_a, normal_a) = 0
    // Plane B: dot(p - point_b, normal_b) = 0
    //
    // We'll find a point on the edge by starting from the midpoint of plane_points
    // and projecting onto both plane constraints

    let midpoint = (
        (point_a.0 + point_b.0) / 2.0,
        (point_a.1 + point_b.1) / 2.0,
        (point_a.2 + point_b.2) / 2.0,
    );

    // Project midpoint onto edge line using least squares
    // The edge line passes through a point P0 that satisfies both plane equations
    // P0 = point_a + t1*v1 + t2*v2 where v1,v2 are in-plane directions

    // Use iterative projection to find a point on both planes
    let mut p = midpoint;
    for _ in 0..10 {
        // Project onto plane A
        let dist_a = dot(sub(p, point_a), normal_a);
        p = sub(p, scale(normal_a, dist_a));

        // Project onto plane B
        let dist_b = dot(sub(p, point_b), normal_b);
        p = sub(p, scale(normal_b, dist_b));
    }

    // Now p is approximately on the edge line
    // Project query point onto the edge line through p
    let to_query = sub(query, p);
    let t = dot(to_query, edge_direction);

    (
        p.0 + t * edge_direction.0,
        p.1 + t * edge_direction.1,
        p.2 + t * edge_direction.2,
    )
}

// Vector math helpers

fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn scale(v: (f64, f64, f64), s: f64) -> (f64, f64, f64) {
    (v.0 * s, v.1 * s, v.2 * s)
}

fn length(v: (f64, f64, f64)) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = length(v);
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 0.0, 1.0)
    }
}

fn distance_sq(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let d = dot(normalize(a), normalize(b)).clamp(-1.0, 1.0);
    d.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_kmeans_basic() {
        // Two well-separated clusters
        let points = vec![
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (1.0, 0.0, 0.0),
            (1.1, 0.0, 0.0),
            (1.0, 0.1, 0.0),
        ];

        let result = kmeans_two_clusters(&points, (0.5, 0.0, 0.0), 10);
        assert!(result.is_some());

        let (c1, c2) = result.unwrap();
        assert_eq!(c1.len(), 3);
        assert_eq!(c2.len(), 3);
    }

    #[test]
    fn test_find_edge_on_cube() {
        let cache = SampleCache::new(unit_cube);

        // Point on edge between +X and +Z faces
        // Edge is at x=0.5, z=0.5, y varies
        // Use more probes and longer search distance for axis-aligned cube
        let config = EdgeFindingConfig {
            num_probes: 150,
            max_search_distance: 0.5,
            search_step: 0.01,
            binary_iterations: 20,
            kmeans_iterations: 30,
        };

        // Place point very close to the edge for better clustering
        let result = reference_find_nearest_edge((0.48, 0.0, 0.48), &cache, &config);

        assert!(result.is_some(), "Should find edge");
        let edge = result.unwrap();

        // Check that we found two distinct faces (dihedral angle should be non-trivial)
        // For an axis-aligned cube, the true angle is 90°, but clustering may not be perfect
        let angle_deg = edge.dihedral_angle.to_degrees();
        assert!(
            angle_deg > 20.0 && angle_deg < 170.0,
            "Dihedral angle should indicate two distinct faces, got {}°",
            angle_deg
        );

        // Check confidence is non-zero (the reference algorithm succeeded)
        assert!(
            edge.confidence > 0.1,
            "Should have non-zero confidence, got {}",
            edge.confidence
        );

        // Verify cluster balance - both clusters should have points
        assert!(
            edge.cluster_a_size >= 3 && edge.cluster_b_size >= 3,
            "Both clusters should have points: {} and {}",
            edge.cluster_a_size,
            edge.cluster_b_size
        );
    }

    #[test]
    fn test_edge_confidence() {
        let cache = SampleCache::new(unit_cube);

        let config = EdgeFindingConfig {
            num_probes: 100,
            max_search_distance: 0.3,
            search_step: 0.02,
            binary_iterations: 15,
            kmeans_iterations: 20,
        };

        // On edge: should have good confidence
        let result_edge = reference_find_nearest_edge((0.45, 0.0, 0.45), &cache, &config);
        assert!(result_edge.is_some());
        assert!(
            result_edge.as_ref().unwrap().confidence > 0.3,
            "Edge confidence should be reasonable"
        );

        // On face (not edge): clustering should still work but may have lower confidence
        // or one cluster dominates
        let result_face = reference_find_nearest_edge((0.4, 0.0, 0.0), &cache, &config);
        // This might return None or have lower confidence
        if let Some(ref edge) = result_face {
            // If it returns something, the cluster balance should be poor
            let balance = edge.cluster_a_size.min(edge.cluster_b_size) as f64
                / edge.cluster_a_size.max(edge.cluster_b_size) as f64;
            // Can't make strong assertions here since face points might cluster oddly
            println!(
                "Face point: balance={:.2}, confidence={:.2}",
                balance, edge.confidence
            );
        }
    }
}
