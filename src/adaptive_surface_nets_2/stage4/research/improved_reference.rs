//! Improved Reference Implementations
//!
//! Alternative approaches to edge and corner detection that address the
//! limitations of simple k-means clustering.
//!
//! # Approaches
//!
//! 1. **Normal-based clustering**: Cluster surface points by their local normal
//!    direction instead of spatial proximity.
//!
//! 2. **RANSAC plane fitting**: Iteratively fit planes, removing inliers to find
//!    subsequent planes.
//!
//! 3. **Gradient-based assignment**: Compute surface normals via finite differences
//!    at each surface point, then cluster by normal similarity.

use super::sample_cache::{find_crossing_in_direction, SampleCache};

// ============================================================================
// APPROACH 1: Normal-Based Clustering
// ============================================================================

/// Edge detection using normal-based clustering.
///
/// Instead of clustering surface points by position, this approach:
/// 1. Finds surface points via dense probing
/// 2. Computes local surface normal at each point via gradient
/// 3. Clusters points by normal direction using spherical k-means
/// 4. Fits planes to each cluster
#[derive(Clone, Debug)]
pub struct NormalClusteringResult {
    pub edge_direction: (f64, f64, f64),
    pub point_on_edge: (f64, f64, f64),
    pub face_a_normal: (f64, f64, f64),
    pub face_b_normal: (f64, f64, f64),
    pub cluster_a_size: usize,
    pub cluster_b_size: usize,
    pub residual_a: f64,
    pub residual_b: f64,
}

pub fn edge_detection_normal_clustering<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    num_probes: usize,
    max_distance: f64,
    gradient_epsilon: f64,
) -> Option<NormalClusteringResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Step 1: Find surface points
    let directions = generate_sphere_directions(num_probes);
    let mut surface_points = Vec::new();

    for dir in &directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache, point, *dir, max_distance, 0.01, 20,
        ) {
            surface_points.push(crossing);
        }
    }

    if surface_points.len() < 6 {
        return None;
    }

    // Step 2: Compute local normal at each surface point via gradient
    let mut point_normals: Vec<((f64, f64, f64), (f64, f64, f64))> = Vec::new();

    for &sp in &surface_points {
        let normal = compute_gradient_normal(cache, sp, gradient_epsilon);
        // Orient normal to point away from query point
        let to_query = sub(point, sp);
        let oriented = if dot(normal, to_query) < 0.0 {
            normal
        } else {
            neg(normal)
        };
        point_normals.push((sp, oriented));
    }

    // Step 3: Cluster by normal direction using spherical k-means
    let (cluster_a, cluster_b) = spherical_kmeans_2(&point_normals, 30)?;

    if cluster_a.len() < 3 || cluster_b.len() < 3 {
        return None;
    }

    // Step 4: Fit planes to each cluster
    let points_a: Vec<_> = cluster_a.iter().map(|(p, _)| *p).collect();
    let points_b: Vec<_> = cluster_b.iter().map(|(p, _)| *p).collect();

    let (centroid_a, normal_a, residual_a) = fit_plane_to_points(&points_a);
    let (centroid_b, normal_b, residual_b) = fit_plane_to_points(&points_b);

    // Orient normals away from query point
    let normal_a = orient_away(normal_a, centroid_a, point);
    let normal_b = orient_away(normal_b, centroid_b, point);

    // Edge direction from cross product
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None;
    }

    // Find point on edge
    let point_on_edge = find_edge_point(point, centroid_a, normal_a, centroid_b, normal_b, edge_dir);

    Some(NormalClusteringResult {
        edge_direction: edge_dir,
        point_on_edge,
        face_a_normal: normal_a,
        face_b_normal: normal_b,
        cluster_a_size: cluster_a.len(),
        cluster_b_size: cluster_b.len(),
        residual_a,
        residual_b,
    })
}

/// Spherical k-means: cluster unit vectors by angular similarity
fn spherical_kmeans_2(
    point_normals: &[((f64, f64, f64), (f64, f64, f64))],
    iterations: usize,
) -> Option<(Vec<((f64, f64, f64), (f64, f64, f64))>, Vec<((f64, f64, f64), (f64, f64, f64))>)> {
    if point_normals.len() < 2 {
        return None;
    }

    // Initialize centroids: find two normals that are most different
    let mut c1 = point_normals[0].1;
    let mut max_angle = 0.0;
    let mut c2 = c1;

    for &(_, n) in point_normals {
        let angle = angle_between(n, c1);
        if angle > max_angle {
            max_angle = angle;
            c2 = n;
        }
    }

    // Run spherical k-means
    for _ in 0..iterations {
        let mut cluster1: Vec<((f64, f64, f64), (f64, f64, f64))> = Vec::new();
        let mut cluster2: Vec<((f64, f64, f64), (f64, f64, f64))> = Vec::new();

        // Assign points to nearest centroid (by angular distance)
        for &pn in point_normals {
            let angle1 = angle_between(pn.1, c1);
            let angle2 = angle_between(pn.1, c2);
            if angle1 < angle2 {
                cluster1.push(pn);
            } else {
                cluster2.push(pn);
            }
        }

        if cluster1.is_empty() || cluster2.is_empty() {
            return None;
        }

        // Update centroids (mean direction, normalized)
        c1 = mean_direction(&cluster1.iter().map(|(_, n)| *n).collect::<Vec<_>>());
        c2 = mean_direction(&cluster2.iter().map(|(_, n)| *n).collect::<Vec<_>>());
    }

    // Final assignment
    let mut cluster1 = Vec::new();
    let mut cluster2 = Vec::new();

    for &pn in point_normals {
        let angle1 = angle_between(pn.1, c1);
        let angle2 = angle_between(pn.1, c2);
        if angle1 < angle2 {
            cluster1.push(pn);
        } else {
            cluster2.push(pn);
        }
    }

    if cluster1.is_empty() || cluster2.is_empty() {
        return None;
    }

    Some((cluster1, cluster2))
}

fn mean_direction(normals: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let sum = normals.iter().fold((0.0, 0.0, 0.0), |acc, n| {
        (acc.0 + n.0, acc.1 + n.1, acc.2 + n.2)
    });
    normalize(sum)
}

// ============================================================================
// APPROACH 2: RANSAC Plane Fitting
// ============================================================================

/// Edge detection using RANSAC-style iterative plane fitting.
///
/// 1. Find surface points
/// 2. Fit first plane using RANSAC (find largest consensus set)
/// 3. Remove inliers, fit second plane to remaining points
/// 4. Edge = intersection of planes
#[derive(Clone, Debug)]
pub struct RansacResult {
    pub edge_direction: (f64, f64, f64),
    pub point_on_edge: (f64, f64, f64),
    pub face_a_normal: (f64, f64, f64),
    pub face_b_normal: (f64, f64, f64),
    pub inliers_a: usize,
    pub inliers_b: usize,
    pub residual_a: f64,
    pub residual_b: f64,
}

pub fn edge_detection_ransac<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    num_probes: usize,
    max_distance: f64,
    inlier_threshold: f64,
    ransac_iterations: usize,
) -> Option<RansacResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Step 1: Find surface points
    let directions = generate_sphere_directions(num_probes);
    let mut surface_points = Vec::new();

    for dir in &directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache, point, *dir, max_distance, 0.01, 20,
        ) {
            surface_points.push(crossing);
        }
    }

    if surface_points.len() < 6 {
        return None;
    }

    // Step 2: RANSAC for first plane
    let (plane1_normal, plane1_inliers, _) = ransac_plane(&surface_points, inlier_threshold, ransac_iterations)?;

    // Step 3: Remove inliers, fit second plane
    let remaining: Vec<_> = surface_points
        .iter()
        .filter(|p| !plane1_inliers.contains(p))
        .cloned()
        .collect();

    if remaining.len() < 3 {
        return None;
    }

    let (plane2_normal, plane2_inliers, _) = ransac_plane(&remaining, inlier_threshold, ransac_iterations)?;

    // Fit final planes to inlier sets
    let (centroid_a, normal_a, residual_a) = fit_plane_to_points(&plane1_inliers);
    let (centroid_b, normal_b, residual_b) = fit_plane_to_points(&plane2_inliers);

    // Orient normals away from query point
    let normal_a = orient_away(normal_a, centroid_a, point);
    let normal_b = orient_away(normal_b, centroid_b, point);

    // Edge direction
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None;
    }

    let point_on_edge = find_edge_point(point, centroid_a, normal_a, centroid_b, normal_b, edge_dir);

    Some(RansacResult {
        edge_direction: edge_dir,
        point_on_edge,
        face_a_normal: normal_a,
        face_b_normal: normal_b,
        inliers_a: plane1_inliers.len(),
        inliers_b: plane2_inliers.len(),
        residual_a,
        residual_b,
    })
}

/// RANSAC plane fitting
fn ransac_plane(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> Option<((f64, f64, f64), Vec<(f64, f64, f64)>, f64)> {
    if points.len() < 3 {
        return None;
    }

    let mut best_normal = (0.0, 0.0, 1.0);
    let mut best_inliers: Vec<(f64, f64, f64)> = Vec::new();
    let mut best_score = 0;

    // Use deterministic sampling for reproducibility
    for i in 0..iterations {
        // Select 3 points (deterministic pseudo-random based on iteration)
        let idx1 = i % points.len();
        let idx2 = (i * 7 + 3) % points.len();
        let idx3 = (i * 13 + 7) % points.len();

        if idx1 == idx2 || idx2 == idx3 || idx1 == idx3 {
            continue;
        }

        let p1 = points[idx1];
        let p2 = points[idx2];
        let p3 = points[idx3];

        // Compute plane from 3 points
        let v1 = sub(p2, p1);
        let v2 = sub(p3, p1);
        let normal = normalize(cross(v1, v2));

        if length(normal) < 1e-6 {
            continue;
        }

        // Count inliers
        let mut inliers = Vec::new();
        for &p in points {
            let dist = (dot(sub(p, p1), normal)).abs();
            if dist < inlier_threshold {
                inliers.push(p);
            }
        }

        if inliers.len() > best_score {
            best_score = inliers.len();
            best_normal = normal;
            best_inliers = inliers;
        }
    }

    if best_inliers.len() < 3 {
        return None;
    }

    // Refit plane to all inliers
    let (_, refined_normal, residual) = fit_plane_to_points(&best_inliers);

    Some((refined_normal, best_inliers, residual))
}

// ============================================================================
// APPROACH 3: Gradient-Based Direct Assignment
// ============================================================================

/// Edge detection using gradient-based face assignment.
///
/// 1. Find surface points
/// 2. Compute gradient (normal) at each point
/// 3. Cluster normals using angle threshold (not k-means)
/// 4. Each distinct normal direction = one face
#[derive(Clone, Debug)]
pub struct GradientClusterResult {
    pub edge_direction: (f64, f64, f64),
    pub point_on_edge: (f64, f64, f64),
    pub face_a_normal: (f64, f64, f64),
    pub face_b_normal: (f64, f64, f64),
    pub cluster_a_size: usize,
    pub cluster_b_size: usize,
}

pub fn edge_detection_gradient<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    num_probes: usize,
    max_distance: f64,
    gradient_epsilon: f64,
    angle_merge_threshold: f64, // Normals within this angle are same face
) -> Option<GradientClusterResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Step 1: Find surface points with their normals
    let directions = generate_sphere_directions(num_probes);
    let mut point_normals: Vec<((f64, f64, f64), (f64, f64, f64))> = Vec::new();

    for dir in &directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache, point, *dir, max_distance, 0.01, 20,
        ) {
            let normal = compute_gradient_normal(cache, crossing, gradient_epsilon);
            // Orient away from query point
            let to_query = sub(point, crossing);
            let oriented = if dot(normal, to_query) < 0.0 {
                normal
            } else {
                neg(normal)
            };
            point_normals.push((crossing, oriented));
        }
    }

    if point_normals.len() < 6 {
        return None;
    }

    // Step 2: Greedy clustering by normal similarity
    let clusters = greedy_normal_clustering(&point_normals, angle_merge_threshold);

    // Need exactly 2 clusters for edge detection
    if clusters.len() < 2 {
        return None;
    }

    // Take two largest clusters
    let mut sorted_clusters = clusters;
    sorted_clusters.sort_by(|a, b| b.len().cmp(&a.len()));

    let cluster_a = &sorted_clusters[0];
    let cluster_b = &sorted_clusters[1];

    if cluster_a.len() < 3 || cluster_b.len() < 3 {
        return None;
    }

    // Step 3: Compute face normals as mean of cluster normals
    let normal_a = mean_direction(&cluster_a.iter().map(|(_, n)| *n).collect::<Vec<_>>());
    let normal_b = mean_direction(&cluster_b.iter().map(|(_, n)| *n).collect::<Vec<_>>());

    // Edge direction
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None;
    }

    // Find point on edge using cluster centroids
    let centroid_a = centroid(&cluster_a.iter().map(|(p, _)| *p).collect::<Vec<_>>());
    let centroid_b = centroid(&cluster_b.iter().map(|(p, _)| *p).collect::<Vec<_>>());
    let point_on_edge = find_edge_point(point, centroid_a, normal_a, centroid_b, normal_b, edge_dir);

    Some(GradientClusterResult {
        edge_direction: edge_dir,
        point_on_edge,
        face_a_normal: normal_a,
        face_b_normal: normal_b,
        cluster_a_size: cluster_a.len(),
        cluster_b_size: cluster_b.len(),
    })
}

/// Greedy clustering: merge points whose normals are within threshold
fn greedy_normal_clustering(
    point_normals: &[((f64, f64, f64), (f64, f64, f64))],
    angle_threshold: f64,
) -> Vec<Vec<((f64, f64, f64), (f64, f64, f64))>> {
    let mut clusters: Vec<Vec<((f64, f64, f64), (f64, f64, f64))>> = Vec::new();
    let mut assigned = vec![false; point_normals.len()];

    for i in 0..point_normals.len() {
        if assigned[i] {
            continue;
        }

        // Start new cluster with this point
        let mut cluster = vec![point_normals[i]];
        assigned[i] = true;
        let seed_normal = point_normals[i].1;

        // Add all unassigned points with similar normal
        for j in (i + 1)..point_normals.len() {
            if assigned[j] {
                continue;
            }

            let angle = angle_between(point_normals[j].1, seed_normal);
            if angle < angle_threshold {
                cluster.push(point_normals[j]);
                assigned[j] = true;
            }
        }

        clusters.push(cluster);
    }

    clusters
}

// ============================================================================
// APPROACH 4: Two-Pass Refinement (Gradient + Plane Fit)
// ============================================================================

/// Two-pass edge detection:
/// 1. First pass: gradient clustering to identify face membership
/// 2. Second pass: refine with plane fitting to the identified clusters
#[derive(Clone, Debug)]
pub struct TwoPassResult {
    pub edge_direction: (f64, f64, f64),
    pub point_on_edge: (f64, f64, f64),
    pub face_a_normal: (f64, f64, f64),
    pub face_b_normal: (f64, f64, f64),
    pub cluster_a_size: usize,
    pub cluster_b_size: usize,
    pub residual_a: f64,
    pub residual_b: f64,
}

pub fn edge_detection_two_pass<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    num_probes: usize,
    max_distance: f64,
    gradient_epsilon: f64,
) -> Option<TwoPassResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Step 1: Find surface points with gradient normals
    let directions = generate_sphere_directions(num_probes);
    let mut point_normals: Vec<((f64, f64, f64), (f64, f64, f64))> = Vec::new();

    for dir in &directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache, point, *dir, max_distance, 0.01, 20,
        ) {
            let normal = compute_gradient_normal(cache, crossing, gradient_epsilon);
            let to_query = sub(point, crossing);
            let oriented = if dot(normal, to_query) < 0.0 {
                normal
            } else {
                neg(normal)
            };
            point_normals.push((crossing, oriented));
        }
    }

    if point_normals.len() < 6 {
        return None;
    }

    // Step 2: Use spherical k-means on gradient normals
    let (cluster_a, cluster_b) = spherical_kmeans_2(&point_normals, 50)?;

    if cluster_a.len() < 3 || cluster_b.len() < 3 {
        return None;
    }

    // Step 3: Fit planes to clustered points (using positions, not gradient normals)
    let points_a: Vec<_> = cluster_a.iter().map(|(p, _)| *p).collect();
    let points_b: Vec<_> = cluster_b.iter().map(|(p, _)| *p).collect();

    let (centroid_a, plane_normal_a, residual_a) = fit_plane_to_points(&points_a);
    let (centroid_b, plane_normal_b, residual_b) = fit_plane_to_points(&points_b);

    // Orient plane normals using gradient normal consensus
    let grad_normal_a = mean_direction(&cluster_a.iter().map(|(_, n)| *n).collect::<Vec<_>>());
    let grad_normal_b = mean_direction(&cluster_b.iter().map(|(_, n)| *n).collect::<Vec<_>>());

    let normal_a = if dot(plane_normal_a, grad_normal_a) > 0.0 {
        plane_normal_a
    } else {
        neg(plane_normal_a)
    };
    let normal_b = if dot(plane_normal_b, grad_normal_b) > 0.0 {
        plane_normal_b
    } else {
        neg(plane_normal_b)
    };

    // Edge direction
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None;
    }

    let point_on_edge = find_edge_point(point, centroid_a, normal_a, centroid_b, normal_b, edge_dir);

    Some(TwoPassResult {
        edge_direction: edge_dir,
        point_on_edge,
        face_a_normal: normal_a,
        face_b_normal: normal_b,
        cluster_a_size: cluster_a.len(),
        cluster_b_size: cluster_b.len(),
        residual_a,
        residual_b,
    })
}

// ============================================================================
// Shared Utilities
// ============================================================================

/// Compute surface normal via central differences gradient
pub fn compute_gradient_normal<F>(
    cache: &SampleCache<F>,
    point: (f64, f64, f64),
    epsilon: f64,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    let dx = cache.sample(point.0 + epsilon, point.1, point.2) as f64
        - cache.sample(point.0 - epsilon, point.1, point.2) as f64;
    let dy = cache.sample(point.0, point.1 + epsilon, point.2) as f64
        - cache.sample(point.0, point.1 - epsilon, point.2) as f64;
    let dz = cache.sample(point.0, point.1, point.2 + epsilon) as f64
        - cache.sample(point.0, point.1, point.2 - epsilon) as f64;

    normalize((dx, dy, dz))
}

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

fn fit_plane_to_points(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
    if points.len() < 3 {
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), f64::INFINITY);
    }

    let center = centroid(points);
    let centered: Vec<_> = points.iter().map(|p| sub(*p, center)).collect();

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

fn centroid(points: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let n = points.len() as f64;
    (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    )
}

fn orient_away(normal: (f64, f64, f64), plane_point: (f64, f64, f64), away_from: (f64, f64, f64)) -> (f64, f64, f64) {
    let to_plane = sub(plane_point, away_from);
    if dot(normal, to_plane) < 0.0 {
        neg(normal)
    } else {
        normal
    }
}

fn find_edge_point(
    query: (f64, f64, f64),
    centroid_a: (f64, f64, f64),
    normal_a: (f64, f64, f64),
    centroid_b: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
) -> (f64, f64, f64) {
    let midpoint = (
        (centroid_a.0 + centroid_b.0) / 2.0,
        (centroid_a.1 + centroid_b.1) / 2.0,
        (centroid_a.2 + centroid_b.2) / 2.0,
    );

    // Iteratively project onto both planes
    let mut p = midpoint;
    for _ in 0..10 {
        let dist_a = dot(sub(p, centroid_a), normal_a);
        p = sub(p, scale(normal_a, dist_a));
        let dist_b = dot(sub(p, centroid_b), normal_b);
        p = sub(p, scale(normal_b, dist_b));
    }

    // Project to be nearest to query along edge direction
    let to_query = sub(query, p);
    let t = dot(to_query, edge_direction);
    (
        p.0 + t * edge_direction.0,
        p.1 + t * edge_direction.1,
        p.2 + t * edge_direction.2,
    )
}

// Vector math
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

fn neg(v: (f64, f64, f64)) -> (f64, f64, f64) {
    (-v.0, -v.1, -v.2)
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

fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let d = dot(normalize(a), normalize(b)).clamp(-1.0, 1.0);
    d.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube(x: f64, y: f64, z: f64) -> f32 {
        let h = 0.5;
        if x.abs() <= h && y.abs() <= h && z.abs() <= h {
            1.0
        } else {
            -1.0
        }
    }

    #[test]
    fn test_gradient_normal() {
        let cache = SampleCache::new(unit_cube);

        // Point on +X face
        let normal = compute_gradient_normal(&cache, (0.5, 0.0, 0.0), 0.01);
        assert!(normal.0.abs() > 0.9, "Should point in X direction");
    }
}
