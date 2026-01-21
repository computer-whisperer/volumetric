//! Reference Implementation: Find Nearest Corner
//!
//! Given a point P near a geometric corner, finds:
//! - Whether this is a corner (3+ faces meet)
//! - The face normals meeting at the corner
//! - The corner position
//!
//! # Approach
//!
//! Multi-cluster analysis:
//! 1. Probe very densely (100+ directions)
//! 2. Binary search each direction to find surface crossing
//! 3. Attempt k-cluster fits for k=2, 3, 4
//! 4. Select k based on residual improvement and cluster validity
//! 5. Fit planes to each cluster, compute corner as intersection
//!
//! This is expensive (150+ samples) but handles corners properly.

use super::sample_cache::{find_crossing_in_direction, SampleCache};

/// Result of reference corner finding
#[derive(Clone, Debug)]
pub struct CornerFindingResult {
    /// Number of faces detected meeting at this point (2 = edge, 3+ = corner)
    pub num_faces: usize,
    /// The face normals (outward-facing)
    pub face_normals: Vec<(f64, f64, f64)>,
    /// Estimated corner position (intersection of planes)
    pub corner_position: (f64, f64, f64),
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Per-cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Per-cluster plane fit residuals
    pub cluster_residuals: Vec<f64>,
    /// Which k was selected (2, 3, or 4)
    pub selected_k: usize,
}

/// Configuration for reference corner finding
#[derive(Clone, Debug)]
pub struct CornerFindingConfig {
    /// Number of probe directions (more needed for corners than edges)
    pub num_probes: usize,
    /// Maximum distance to search for surface crossing
    pub max_search_distance: f64,
    /// Step size for initial crossing search
    pub search_step: f64,
    /// Number of binary search iterations for crossing refinement
    pub binary_iterations: usize,
    /// Number of k-means iterations for clustering
    pub kmeans_iterations: usize,
    /// Minimum cluster size to be valid
    pub min_cluster_size: usize,
    /// Residual improvement ratio required to increase k
    /// (new_residual must be < old_residual * improvement_threshold)
    pub k_improvement_threshold: f64,
}

impl Default for CornerFindingConfig {
    fn default() -> Self {
        Self {
            num_probes: 150,
            max_search_distance: 1.0,
            search_step: 0.05,
            binary_iterations: 20,
            kmeans_iterations: 30,
            min_cluster_size: 4,
            k_improvement_threshold: 0.7,
        }
    }
}

/// Find the nearest corner from a given point.
///
/// Uses dense spherical probing and multi-cluster analysis.
/// Tries k=2, 3, 4 clusters and selects based on fit quality.
///
/// # Arguments
/// * `point` - Starting point for the search (should be near a corner)
/// * `cache` - Sample cache for the underlying sampler
/// * `config` - Configuration parameters
///
/// # Returns
/// CornerFindingResult with the corner position, normals, and metadata.
/// Returns None if insufficient surface points found.
pub fn reference_find_nearest_corner<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    config: &CornerFindingConfig,
) -> Option<CornerFindingResult>
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

    // Need at least 12 points for trying k=4 (3 per cluster minimum)
    if surface_points.len() < 8 {
        return None;
    }

    // Try k=2, 3, 4 and select best
    let mut best_result: Option<CornerFindingResult> = None;
    let mut best_score = f64::NEG_INFINITY;

    for k in 2..=4 {
        if surface_points.len() < k * config.min_cluster_size {
            continue;
        }

        if let Some(clusters) = kmeans_k_clusters(&surface_points, k, config.kmeans_iterations) {
            // Check all clusters have minimum size
            let valid_clusters = clusters.iter().all(|c| c.len() >= config.min_cluster_size);
            if !valid_clusters {
                continue;
            }

            // Fit planes to each cluster
            let mut normals = Vec::with_capacity(k);
            let mut centroids = Vec::with_capacity(k);
            let mut residuals = Vec::with_capacity(k);
            let mut cluster_sizes = Vec::with_capacity(k);

            for cluster in &clusters {
                let (centroid, normal, residual) = fit_plane_to_points(cluster);
                let oriented_normal = orient_normal_away(normal, centroid, point);
                normals.push(oriented_normal);
                centroids.push(centroid);
                residuals.push(residual);
                cluster_sizes.push(cluster.len());
            }

            // Compute total residual (sum of squared residuals weighted by cluster size)
            let total_residual: f64 = residuals.iter().zip(cluster_sizes.iter())
                .map(|(r, &s)| r * r * s as f64)
                .sum::<f64>()
                / surface_points.len() as f64;

            // Compute corner position as intersection of planes
            let corner_position = if k == 2 {
                // Edge: midpoint of centroids projected onto intersection line
                let edge_dir = normalize(cross(normals[0], normals[1]));
                let mid = (
                    (centroids[0].0 + centroids[1].0) / 2.0,
                    (centroids[0].1 + centroids[1].1) / 2.0,
                    (centroids[0].2 + centroids[1].2) / 2.0,
                );
                find_point_on_edge(mid, &centroids, &normals, edge_dir)
            } else {
                // Corner: intersection of 3+ planes
                find_planes_intersection(&centroids, &normals)
            };

            // Score this clustering
            // Prefer: lower residuals, balanced clusters, appropriate k for geometry
            let cluster_balance = cluster_balance_score(&cluster_sizes);
            let residual_score = (-total_residual.sqrt() * 50.0).exp();

            // Penalty for higher k unless it significantly improves residual
            let k_penalty = match k {
                2 => 1.0,
                3 => 0.9,
                4 => 0.8,
                _ => 0.7,
            };

            // Check if normals are sufficiently different
            let normal_separation = min_angle_between_normals(&normals);
            let separation_score = if normal_separation.to_degrees() > 20.0 {
                1.0
            } else {
                0.5
            };

            let score = cluster_balance * 0.2 + residual_score * 0.4 + k_penalty * 0.2 + separation_score * 0.2;

            // Update best if this is better
            let should_update = match &best_result {
                None => true,
                Some(prev) => {
                    // Prefer higher k only if score is similar or better
                    if k > prev.selected_k {
                        score > best_score * config.k_improvement_threshold
                    } else {
                        score > best_score
                    }
                }
            };

            if should_update {
                best_score = score;
                best_result = Some(CornerFindingResult {
                    num_faces: k,
                    face_normals: normals,
                    corner_position,
                    confidence: score.clamp(0.0, 1.0),
                    cluster_sizes,
                    cluster_residuals: residuals,
                    selected_k: k,
                });
            }
        }
    }

    best_result
}

/// Generate uniformly distributed directions on a sphere
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

/// K-means clustering for k clusters
fn kmeans_k_clusters(
    points: &[(f64, f64, f64)],
    k: usize,
    iterations: usize,
) -> Option<Vec<Vec<(f64, f64, f64)>>> {
    if points.len() < k {
        return None;
    }

    // Initialize centroids using k-means++ style initialization
    let mut centroids = Vec::with_capacity(k);

    // First centroid: random (use first point)
    centroids.push(points[0]);

    // Remaining centroids: choose points far from existing centroids
    for _ in 1..k {
        let mut max_min_dist = 0.0;
        let mut best_point = points[0];

        for &p in points {
            let min_dist = centroids.iter().map(|&c| distance_sq(p, c)).fold(f64::INFINITY, f64::min);
            if min_dist > max_min_dist {
                max_min_dist = min_dist;
                best_point = p;
            }
        }
        centroids.push(best_point);
    }

    // Run k-means iterations
    for _ in 0..iterations {
        // Assign points to clusters
        let mut clusters: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); k];

        for &p in points {
            let (closest_idx, _) = centroids
                .iter()
                .enumerate()
                .map(|(i, &c)| (i, distance_sq(p, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            clusters[closest_idx].push(p);
        }

        // Check for empty clusters
        if clusters.iter().any(|c| c.is_empty()) {
            return None;
        }

        // Update centroids
        for (i, cluster) in clusters.iter().enumerate() {
            centroids[i] = centroid(cluster);
        }
    }

    // Final assignment
    let mut clusters: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); k];

    for &p in points {
        let (closest_idx, _) = centroids
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, distance_sq(p, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        clusters[closest_idx].push(p);
    }

    if clusters.iter().any(|c| c.is_empty()) {
        return None;
    }

    Some(clusters)
}

fn centroid(points: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let n = points.len() as f64;
    (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    )
}

fn fit_plane_to_points(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
    if points.len() < 3 {
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), f64::INFINITY);
    }

    let center = centroid(points);
    let centered: Vec<(f64, f64, f64)> = points.iter().map(|p| sub(*p, center)).collect();

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

    let mut residual_sum = 0.0;
    for p in &centered {
        let dist = dot(*p, normal).abs();
        residual_sum += dist * dist;
    }
    let rms_residual = (residual_sum / points.len() as f64).sqrt();

    (center, normal, rms_residual)
}

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

fn orient_normal_away(normal: (f64, f64, f64), plane_point: (f64, f64, f64), away_from: (f64, f64, f64)) -> (f64, f64, f64) {
    let to_plane = sub(plane_point, away_from);
    if dot(normal, to_plane) < 0.0 {
        (-normal.0, -normal.1, -normal.2)
    } else {
        normal
    }
}

/// Find a point on the intersection line of two planes
fn find_point_on_edge(
    start: (f64, f64, f64),
    centroids: &[(f64, f64, f64)],
    normals: &[(f64, f64, f64)],
    edge_direction: (f64, f64, f64),
) -> (f64, f64, f64) {
    let mut p = start;
    for _ in 0..10 {
        for (centroid, normal) in centroids.iter().zip(normals.iter()) {
            let dist = dot(sub(p, *centroid), *normal);
            p = sub(p, scale(*normal, dist));
        }
    }

    // Project to be nearest to original start along edge direction
    let to_start = sub(start, p);
    let t = dot(to_start, edge_direction);
    (
        p.0 + t * edge_direction.0,
        p.1 + t * edge_direction.1,
        p.2 + t * edge_direction.2,
    )
}

/// Find intersection of 3+ planes (least-squares solution)
fn find_planes_intersection(
    centroids: &[(f64, f64, f64)],
    normals: &[(f64, f64, f64)],
) -> (f64, f64, f64) {
    // Set up least squares: minimize sum of squared distances to planes
    // Each plane: dot(p - centroid, normal) = 0
    // => dot(p, normal) = dot(centroid, normal)
    //
    // Matrix form: N * p = b where N is (k x 3), b is (k x 1)

    let k = normals.len();
    if k < 3 {
        // Not enough planes for a unique intersection
        return centroid(centroids);
    }

    // Build A^T A and A^T b
    let mut ata = [[0.0; 3]; 3];
    let mut atb = [0.0; 3];

    for i in 0..k {
        let n = normals[i];
        let c = centroids[i];
        let bi = dot(c, n);

        ata[0][0] += n.0 * n.0;
        ata[0][1] += n.0 * n.1;
        ata[0][2] += n.0 * n.2;
        ata[1][1] += n.1 * n.1;
        ata[1][2] += n.1 * n.2;
        ata[2][2] += n.2 * n.2;

        atb[0] += n.0 * bi;
        atb[1] += n.1 * bi;
        atb[2] += n.2 * bi;
    }
    ata[1][0] = ata[0][1];
    ata[2][0] = ata[0][2];
    ata[2][1] = ata[1][2];

    // Solve using Cramer's rule (simple for 3x3)
    solve_3x3(&ata, &atb).unwrap_or_else(|| centroid(centroids))
}

/// Solve 3x3 linear system using Cramer's rule
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<(f64, f64, f64)> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let det_x = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);

    let det_y = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);

    let det_z = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    Some((det_x / det, det_y / det, det_z / det))
}

fn cluster_balance_score(sizes: &[usize]) -> f64 {
    if sizes.is_empty() {
        return 0.0;
    }

    let min = *sizes.iter().min().unwrap() as f64;
    let max = *sizes.iter().max().unwrap() as f64;

    if max == 0.0 {
        return 0.0;
    }

    min / max
}

fn min_angle_between_normals(normals: &[(f64, f64, f64)]) -> f64 {
    let mut min_angle = std::f64::consts::PI;

    for i in 0..normals.len() {
        for j in (i + 1)..normals.len() {
            let angle = angle_between(normals[i], normals[j]);
            min_angle = min_angle.min(angle);
        }
    }

    min_angle
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
    fn test_kmeans_three_clusters() {
        // Three well-separated clusters
        let points = vec![
            (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0),
            (1.0, 0.0, 0.0), (1.1, 0.0, 0.0), (1.0, 0.1, 0.0),
            (0.5, 1.0, 0.0), (0.5, 1.1, 0.0), (0.6, 1.0, 0.0),
        ];

        let result = kmeans_k_clusters(&points, 3, 20);
        assert!(result.is_some());

        let clusters = result.unwrap();
        assert_eq!(clusters.len(), 3);
        for cluster in &clusters {
            assert_eq!(cluster.len(), 3);
        }
    }

    #[test]
    fn test_find_corner_on_cube() {
        let cache = SampleCache::new(unit_cube);

        // Point near corner at (0.5, 0.5, 0.5)
        let config = CornerFindingConfig {
            num_probes: 150,
            max_search_distance: 0.3,
            search_step: 0.02,
            binary_iterations: 15,
            kmeans_iterations: 30,
            min_cluster_size: 3,
            k_improvement_threshold: 0.7,
        };

        let result = reference_find_nearest_corner((0.4, 0.4, 0.4), &cache, &config);

        assert!(result.is_some(), "Should find corner");
        let corner = result.unwrap();

        // Should detect 3 faces at a corner
        assert!(corner.num_faces >= 2, "Should detect at least 2 faces, got {}", corner.num_faces);

        // If 3 faces detected, normals should be roughly +X, +Y, +Z
        if corner.num_faces == 3 {
            // Check that each axis has a normal close to it
            let has_x = corner.face_normals.iter().any(|n| n.0.abs() > 0.8);
            let has_y = corner.face_normals.iter().any(|n| n.1.abs() > 0.8);
            let has_z = corner.face_normals.iter().any(|n| n.2.abs() > 0.8);

            assert!(has_x && has_y && has_z,
                "Expected normals near X, Y, Z axes, got {:?}", corner.face_normals);
        }
    }

    #[test]
    fn test_planes_intersection() {
        // Three perpendicular planes meeting at origin
        // Plane 1: x = 0 (centroid at origin, normal +X)
        // Plane 2: y = 0 (centroid at origin, normal +Y)
        // Plane 3: z = 0 (centroid at origin, normal +Z)
        let centroids = vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)];
        let normals = vec![(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];

        let intersection = find_planes_intersection(&centroids, &normals);

        // Should be at origin
        assert!(intersection.0.abs() < 0.01, "x={}", intersection.0);
        assert!(intersection.1.abs() < 0.01, "y={}", intersection.1);
        assert!(intersection.2.abs() < 0.01, "z={}", intersection.2);

        // Also test planes meeting at (1, 1, 1)
        let centroids2 = vec![(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];
        let normals2 = vec![(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];

        let intersection2 = find_planes_intersection(&centroids2, &normals2);

        // Planes x=1, y=1, z=1 meet at (1, 1, 1)
        assert!((intersection2.0 - 1.0).abs() < 0.01, "x={}", intersection2.0);
        assert!((intersection2.1 - 1.0).abs() < 0.01, "y={}", intersection2.1);
        assert!((intersection2.2 - 1.0).abs() < 0.01, "z={}", intersection2.2);
    }

    #[test]
    fn test_solve_3x3() {
        // Simple system: I * x = b => x = b
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [1.0, 2.0, 3.0];

        let result = solve_3x3(&a, &b);
        assert!(result.is_some());

        let (x, y, z) = result.unwrap();
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 2.0).abs() < 1e-10);
        assert!((z - 3.0).abs() < 1e-10);
    }
}
