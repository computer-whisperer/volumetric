//! Hermite micro-grid experiment.
//!
//! Samples a small grid around a query point, finds sign changes on grid edges,
//! and fits two planes from the resulting Hermite points.

use crate::adaptive_surface_nets_2::stage4::research::sample_cache::binary_search_crossing;
use crate::adaptive_surface_nets_2::stage4::research::SampleCache;

#[derive(Clone, Debug)]
pub enum PlaneFitStrategy {
    KMeans,
    Ransac,
    EdgeAlignedKMeans,
    EdgeAlignedLineRansac,
}

#[derive(Clone, Debug)]
pub struct HermiteMicrogridConfig {
    pub grid_radius: i32,
    pub spacing: f64,
    pub binary_iterations: usize,
    pub kmeans_iterations: usize,
    pub fit_strategy: PlaneFitStrategy,
    pub ransac_inlier_threshold: f64,
    pub ransac_iterations: usize,
    pub line_inlier_threshold: f64,
}

impl Default for HermiteMicrogridConfig {
    fn default() -> Self {
        Self {
            grid_radius: 1,
            spacing: 0.25,
            binary_iterations: 12,
            kmeans_iterations: 15,
            fit_strategy: PlaneFitStrategy::KMeans,
            ransac_inlier_threshold: 0.05,
            ransac_iterations: 80,
            line_inlier_threshold: 0.05,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HermiteMicrogridResult {
    pub face_a_normal: (f64, f64, f64),
    pub face_b_normal: (f64, f64, f64),
    pub edge_direction: (f64, f64, f64),
    pub point_on_edge: (f64, f64, f64),
    pub dihedral_angle: f64,
    pub cluster_a_size: usize,
    pub cluster_b_size: usize,
    pub residual_a: f64,
    pub residual_b: f64,
    pub crossing_points: usize,
}

pub fn hermite_edge_from_microgrid<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    config: &HermiteMicrogridConfig,
) -> Option<HermiteMicrogridResult>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let grid_radius = config.grid_radius.max(1);
    let spacing = config.spacing;
    let size = (grid_radius * 2 + 1) as usize;
    let mut inside = vec![false; size * size * size];

    for iz in 0..size {
        for iy in 0..size {
            for ix in 0..size {
                let pos = grid_position(point, grid_radius, spacing, ix, iy, iz);
                let idx = index(size, ix, iy, iz);
                inside[idx] = cache.is_inside(pos.0, pos.1, pos.2);
            }
        }
    }

    let mut crossings = Vec::new();
    collect_edge_crossings(
        point,
        cache,
        config,
        grid_radius,
        size,
        &inside,
        &mut crossings,
    );

    if crossings.len() < 6 {
        return None;
    }

    let (center_a, normal_a, residual_a, center_b, normal_b, residual_b, size_a, size_b) =
        match config.fit_strategy {
            PlaneFitStrategy::KMeans => {
                let (cluster_a, cluster_b) =
                    kmeans_two_clusters(&crossings, point, config.kmeans_iterations)?;
                if cluster_a.len() < 3 || cluster_b.len() < 3 {
                    return None;
                }
                let (center_a, normal_a, residual_a) = fit_plane_to_points(&cluster_a);
                let (center_b, normal_b, residual_b) = fit_plane_to_points(&cluster_b);
                (
                    center_a,
                    normal_a,
                    residual_a,
                    center_b,
                    normal_b,
                    residual_b,
                    cluster_a.len(),
                    cluster_b.len(),
                )
            }
            PlaneFitStrategy::Ransac => {
                let (plane_a, plane_b) = ransac_two_planes(
                    &crossings,
                    config.ransac_inlier_threshold,
                    config.ransac_iterations,
                )?;
                (
                    plane_a.centroid,
                    plane_a.normal,
                    plane_a.residual,
                    plane_b.centroid,
                    plane_b.normal,
                    plane_b.residual,
                    plane_a.inliers.len(),
                    plane_b.inliers.len(),
                )
            }
            PlaneFitStrategy::EdgeAlignedKMeans => {
                let crossings_center = centroid(&crossings);
                let edge_dir = estimate_edge_direction(&crossings, crossings_center)?;
                let (u, v) = perpendicular_basis(edge_dir)?;
                let projected: Vec<(f64, f64)> = crossings
                    .iter()
                    .map(|p| {
                        let d = sub(*p, crossings_center);
                        (dot(d, u), dot(d, v))
                    })
                    .collect();
                let (cluster_a_idx, cluster_b_idx) =
                    kmeans_two_clusters_2d(&projected, config.kmeans_iterations)?;
                let cluster_a: Vec<_> = cluster_a_idx.iter().map(|&i| crossings[i]).collect();
                let cluster_b: Vec<_> = cluster_b_idx.iter().map(|&i| crossings[i]).collect();
                if cluster_a.len() < 3 || cluster_b.len() < 3 {
                    return None;
                }
                let (center_a, normal_a, residual_a) = fit_plane_to_points(&cluster_a);
                let (center_b, normal_b, residual_b) = fit_plane_to_points(&cluster_b);
                (
                    center_a,
                    normal_a,
                    residual_a,
                    center_b,
                    normal_b,
                    residual_b,
                    cluster_a.len(),
                    cluster_b.len(),
                )
            }
            PlaneFitStrategy::EdgeAlignedLineRansac => {
                let crossings_center = centroid(&crossings);
                let edge_dir = estimate_edge_direction(&crossings, crossings_center)?;
                let (u, v) = perpendicular_basis(edge_dir)?;
                let projected: Vec<(f64, f64)> = crossings
                    .iter()
                    .map(|p| {
                        let d = sub(*p, crossings_center);
                        (dot(d, u), dot(d, v))
                    })
                    .collect();
                let (line_a, inliers_a, line_b, inliers_b) = ransac_two_lines_2d(
                    &projected,
                    config.line_inlier_threshold,
                    config.ransac_iterations,
                )?;
                let cluster_a: Vec<_> = inliers_a.iter().map(|&i| crossings[i]).collect();
                let cluster_b: Vec<_> = inliers_b.iter().map(|&i| crossings[i]).collect();
                if cluster_a.len() < 3 || cluster_b.len() < 3 {
                    return None;
                }
                let t3d_a = add(scale(u, line_a.0), scale(v, line_a.1));
                let t3d_b = add(scale(u, line_b.0), scale(v, line_b.1));
                let normal_a = normalize(cross(edge_dir, t3d_a));
                let normal_b = normalize(cross(edge_dir, t3d_b));
                let center_a = centroid(&cluster_a);
                let center_b = centroid(&cluster_b);
                let residual_a = plane_residual(&cluster_a, center_a, normal_a);
                let residual_b = plane_residual(&cluster_b, center_b, normal_b);
                (
                    center_a,
                    normal_a,
                    residual_a,
                    center_b,
                    normal_b,
                    residual_b,
                    cluster_a.len(),
                    cluster_b.len(),
                )
            }
        };
    let normal_a = orient_normal_away(normal_a, center_a, point);
    let normal_b = orient_normal_away(normal_b, center_b, point);

    let edge_dir_raw = cross(normal_a, normal_b);
    let edge_dir_len = length(edge_dir_raw);
    if edge_dir_len < 1e-6 {
        return None;
    }

    let edge_direction = (
        edge_dir_raw.0 / edge_dir_len,
        edge_dir_raw.1 / edge_dir_len,
        edge_dir_raw.2 / edge_dir_len,
    );

    let point_on_edge = find_closest_point_on_edge(point, center_a, normal_a, center_b, normal_b, edge_direction);
    let dihedral_angle = angle_between(normal_a, normal_b);

    Some(HermiteMicrogridResult {
        face_a_normal: normal_a,
        face_b_normal: normal_b,
        edge_direction,
        point_on_edge,
        dihedral_angle,
        cluster_a_size: size_a,
        cluster_b_size: size_b,
        residual_a,
        residual_b,
        crossing_points: crossings.len(),
    })
}

fn collect_edge_crossings<F>(
    point: (f64, f64, f64),
    cache: &SampleCache<F>,
    config: &HermiteMicrogridConfig,
    grid_radius: i32,
    size: usize,
    inside: &[bool],
    crossings: &mut Vec<(f64, f64, f64)>,
) where
    F: Fn(f64, f64, f64) -> f32,
{
    let spacing = config.spacing;

    for iz in 0..size {
        for iy in 0..size {
            for ix in 0..size.saturating_sub(1) {
                let idx0 = index(size, ix, iy, iz);
                let idx1 = index(size, ix + 1, iy, iz);
                if inside[idx0] != inside[idx1] {
                    let p0 = grid_position(point, grid_radius, spacing, ix, iy, iz);
                    let p1 = grid_position(point, grid_radius, spacing, ix + 1, iy, iz);
                    crossings.push(binary_search_edge(cache, inside[idx0], p0, p1, config.binary_iterations));
                }
            }
        }
    }

    for iz in 0..size {
        for iy in 0..size.saturating_sub(1) {
            for ix in 0..size {
                let idx0 = index(size, ix, iy, iz);
                let idx1 = index(size, ix, iy + 1, iz);
                if inside[idx0] != inside[idx1] {
                    let p0 = grid_position(point, grid_radius, spacing, ix, iy, iz);
                    let p1 = grid_position(point, grid_radius, spacing, ix, iy + 1, iz);
                    crossings.push(binary_search_edge(cache, inside[idx0], p0, p1, config.binary_iterations));
                }
            }
        }
    }

    for iz in 0..size.saturating_sub(1) {
        for iy in 0..size {
            for ix in 0..size {
                let idx0 = index(size, ix, iy, iz);
                let idx1 = index(size, ix, iy, iz + 1);
                if inside[idx0] != inside[idx1] {
                    let p0 = grid_position(point, grid_radius, spacing, ix, iy, iz);
                    let p1 = grid_position(point, grid_radius, spacing, ix, iy, iz + 1);
                    crossings.push(binary_search_edge(cache, inside[idx0], p0, p1, config.binary_iterations));
                }
            }
        }
    }
}

fn binary_search_edge<F>(
    cache: &SampleCache<F>,
    p0_inside: bool,
    p0: (f64, f64, f64),
    p1: (f64, f64, f64),
    iterations: usize,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    if p0_inside {
        binary_search_crossing(cache, p0, p1, iterations)
    } else {
        binary_search_crossing(cache, p1, p0, iterations)
    }
}

fn grid_position(
    point: (f64, f64, f64),
    radius: i32,
    spacing: f64,
    ix: usize,
    iy: usize,
    iz: usize,
) -> (f64, f64, f64) {
    let offset_x = (ix as i32 - radius) as f64 * spacing;
    let offset_y = (iy as i32 - radius) as f64 * spacing;
    let offset_z = (iz as i32 - radius) as f64 * spacing;
    (point.0 + offset_x, point.1 + offset_y, point.2 + offset_z)
}

fn index(size: usize, ix: usize, iy: usize, iz: usize) -> usize {
    (iz * size + iy) * size + ix
}

fn kmeans_two_clusters(
    points: &[(f64, f64, f64)],
    query: (f64, f64, f64),
    iterations: usize,
) -> Option<(Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>)> {
    if points.len() < 2 {
        return None;
    }

    let mut c1 = points[0];
    let mut max_dist = 0.0;
    for &p in points {
        let d = distance_sq(p, query);
        if d > max_dist {
            max_dist = d;
            c1 = p;
        }
    }

    let mut c2 = points[0];
    max_dist = 0.0;
    for &p in points {
        let d = distance_sq(p, c1);
        if d > max_dist {
            max_dist = d;
            c2 = p;
        }
    }

    for _ in 0..iterations {
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

        c1 = centroid(&cluster1);
        c2 = centroid(&cluster2);
    }

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

fn kmeans_two_clusters_2d(
    points: &[(f64, f64)],
    iterations: usize,
) -> Option<(Vec<usize>, Vec<usize>)> {
    if points.len() < 2 {
        return None;
    }

    let mut c1 = points[0];
    let mut max_dist = 0.0;
    for &p in points {
        let d = distance_sq_2d(p, (0.0, 0.0));
        if d > max_dist {
            max_dist = d;
            c1 = p;
        }
    }

    let mut c2 = points[0];
    max_dist = 0.0;
    for &p in points {
        let d = distance_sq_2d(p, c1);
        if d > max_dist {
            max_dist = d;
            c2 = p;
        }
    }

    for _ in 0..iterations {
        let mut cluster1 = Vec::new();
        let mut cluster2 = Vec::new();

        for (idx, &p) in points.iter().enumerate() {
            if distance_sq_2d(p, c1) < distance_sq_2d(p, c2) {
                cluster1.push(idx);
            } else {
                cluster2.push(idx);
            }
        }

        if cluster1.is_empty() || cluster2.is_empty() {
            return None;
        }

        c1 = centroid_2d(points, &cluster1);
        c2 = centroid_2d(points, &cluster2);
    }

    let mut cluster1 = Vec::new();
    let mut cluster2 = Vec::new();
    for (idx, &p) in points.iter().enumerate() {
        if distance_sq_2d(p, c1) < distance_sq_2d(p, c2) {
            cluster1.push(idx);
        } else {
            cluster2.push(idx);
        }
    }

    if cluster1.is_empty() || cluster2.is_empty() {
        return None;
    }

    Some((cluster1, cluster2))
}

fn ransac_two_lines_2d(
    points: &[(f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> Option<((f64, f64), Vec<usize>, (f64, f64), Vec<usize>)> {
    let line_a = ransac_line_fit_2d(points, inlier_threshold, iterations)?;
    let remaining: Vec<usize> = (0..points.len())
        .filter(|idx| !line_a.1.contains(idx))
        .collect();
    if remaining.len() < 3 {
        return None;
    }
    let remaining_points: Vec<(f64, f64)> = remaining.iter().map(|&idx| points[idx]).collect();
    let mut line_b = ransac_line_fit_2d(&remaining_points, inlier_threshold, iterations)?;
    let line_b_indices: Vec<usize> = line_b
        .1
        .iter()
        .map(|&idx| remaining[idx])
        .collect();
    line_b.1 = line_b_indices;
    if line_a.1.len() < 3 || line_b.1.len() < 3 {
        return None;
    }
    Some((line_a.0, line_a.1, line_b.0, line_b.1))
}

fn ransac_line_fit_2d(
    points: &[(f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> Option<((f64, f64), Vec<usize>)> {
    if points.len() < 2 {
        return None;
    }

    let mut best_inliers: Vec<usize> = Vec::new();

    for i in 0..iterations {
        let idx = (i * 7 + 1) % points.len();
        let p = points[idx];
        let len = (p.0 * p.0 + p.1 * p.1).sqrt();
        if len < 1e-8 {
            continue;
        }
        let dir = (p.0 / len, p.1 / len);
        let perp = (-dir.1, dir.0);
        let mut inliers = Vec::new();
        for (pi, &q) in points.iter().enumerate() {
            let dist = (perp.0 * q.0 + perp.1 * q.1).abs();
            if dist < inlier_threshold {
                inliers.push(pi);
            }
        }
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }

    if best_inliers.len() < 2 {
        return None;
    }

    let refined_dir = principal_dir_2d(points, &best_inliers);
    Some((refined_dir, best_inliers))
}

fn principal_dir_2d(points: &[(f64, f64)], indices: &[usize]) -> (f64, f64) {
    let center = centroid_2d(points, indices);
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    for &idx in indices {
        let dx = points[idx].0 - center.0;
        let dy = points[idx].1 - center.1;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    let trace = xx + yy;
    let det = xx * yy - xy * xy;
    let mut lambda = trace / 2.0 + ((trace * trace) / 4.0 - det).sqrt();
    if lambda.is_nan() {
        lambda = trace / 2.0;
    }
    let mut dir = if xy.abs() > 1e-12 {
        (lambda - yy, xy)
    } else if xx >= yy {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
    if len > 1e-12 {
        dir.0 /= len;
        dir.1 /= len;
    } else {
        dir = (1.0, 0.0);
    }
    dir
}

fn centroid_2d(points: &[(f64, f64)], indices: &[usize]) -> (f64, f64) {
    let n = indices.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for &idx in indices {
        sum_x += points[idx].0;
        sum_y += points[idx].1;
    }
    (sum_x / n, sum_y / n)
}

fn distance_sq_2d(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

#[derive(Clone, Debug)]
struct PlaneFit {
    inliers: Vec<(f64, f64, f64)>,
    centroid: (f64, f64, f64),
    normal: (f64, f64, f64),
    residual: f64,
}

fn ransac_two_planes(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> Option<(PlaneFit, PlaneFit)> {
    let plane_a = ransac_plane_fit(points, inlier_threshold, iterations)?;
    let remaining: Vec<_> = points
        .iter()
        .filter(|p| !is_inlier(**p, plane_a.centroid, plane_a.normal, inlier_threshold))
        .cloned()
        .collect();
    let plane_b = ransac_plane_fit(&remaining, inlier_threshold, iterations)?;
    if plane_a.inliers.len() < 3 || plane_b.inliers.len() < 3 {
        return None;
    }
    Some((plane_a, plane_b))
}

fn ransac_plane_fit(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> Option<PlaneFit> {
    if points.len() < 3 {
        return None;
    }

    let n = points.len();
    let mut best_inliers: Vec<(f64, f64, f64)> = Vec::new();

    for i in 0..iterations {
        let idx1 = (i * 7 + 1) % n;
        let idx2 = (i * 13 + 3) % n;
        let idx3 = (i * 19 + 7) % n;

        if idx1 == idx2 || idx2 == idx3 || idx1 == idx3 {
            continue;
        }

        let p1 = points[idx1];
        let p2 = points[idx2];
        let p3 = points[idx3];
        let v1 = sub(p2, p1);
        let v2 = sub(p3, p1);
        let normal = normalize(cross(v1, v2));

        if length(normal) < 1e-6 {
            continue;
        }

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

    let (centroid, normal, residual) = fit_plane_to_points(&best_inliers);

    Some(PlaneFit {
        inliers: best_inliers,
        centroid,
        normal,
        residual,
    })
}

fn is_inlier(
    p: (f64, f64, f64),
    plane_point: (f64, f64, f64),
    plane_normal: (f64, f64, f64),
    threshold: f64,
) -> bool {
    let dist = dot(sub(p, plane_point), plane_normal).abs();
    dist < threshold
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

fn plane_residual(
    points: &[(f64, f64, f64)],
    center: (f64, f64, f64),
    normal: (f64, f64, f64),
) -> f64 {
    if points.is_empty() {
        return f64::INFINITY;
    }
    let mut residual_sum = 0.0;
    for p in points {
        let dist = dot(sub(*p, center), normal).abs();
        residual_sum += dist * dist;
    }
    (residual_sum / points.len() as f64).sqrt()
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

fn estimate_edge_direction(
    points: &[(f64, f64, f64)],
    center: (f64, f64, f64),
) -> Option<(f64, f64, f64)> {
    if points.len() < 3 {
        return None;
    }

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

    let dir = find_largest_eigenvector(&cov);
    let len = length(dir);
    if len < 1e-12 {
        None
    } else {
        Some((dir.0 / len, dir.1 / len, dir.2 / len))
    }
}

fn find_largest_eigenvector(cov: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    let mut v = (1.0, 1.0, 1.0);
    for _ in 0..50 {
        let new_v = (
            cov[0][0] * v.0 + cov[0][1] * v.1 + cov[0][2] * v.2,
            cov[1][0] * v.0 + cov[1][1] * v.1 + cov[1][2] * v.2,
            cov[2][0] * v.0 + cov[2][1] * v.1 + cov[2][2] * v.2,
        );
        let len = length(new_v);
        if len < 1e-12 {
            break;
        }
        v = (new_v.0 / len, new_v.1 / len, new_v.2 / len);
    }
    v
}

fn perpendicular_basis(dir: (f64, f64, f64)) -> Option<((f64, f64, f64), (f64, f64, f64))> {
    let axis = if dir.0.abs() < 0.9 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    };
    let u = normalize(cross(dir, axis));
    if length(u) < 1e-12 {
        return None;
    }
    let v = normalize(cross(dir, u));
    Some((u, v))
}

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

fn find_closest_point_on_edge(
    query: (f64, f64, f64),
    point_a: (f64, f64, f64),
    normal_a: (f64, f64, f64),
    point_b: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
) -> (f64, f64, f64) {
    let midpoint = (
        (point_a.0 + point_b.0) / 2.0,
        (point_a.1 + point_b.1) / 2.0,
        (point_a.2 + point_b.2) / 2.0,
    );

    let mut p = midpoint;
    for _ in 0..10 {
        let dist_a = dot(sub(p, point_a), normal_a);
        p = sub(p, scale(normal_a, dist_a));

        let dist_b = dot(sub(p, point_b), normal_b);
        p = sub(p, scale(normal_b, dist_b));
    }

    let to_query = sub(query, p);
    let t = dot(to_query, edge_direction);

    (
        p.0 + t * edge_direction.0,
        p.1 + t * edge_direction.1,
        p.2 + t * edge_direction.2,
    )
}

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

fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
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
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dz = a.2 - b.2;
    dx * dx + dy * dy + dz * dz
}

fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let denom = length(a) * length(b);
    if denom <= 1e-12 {
        return 0.0;
    }
    let cos = (dot(a, b) / denom).clamp(-1.0, 1.0);
    cos.acos()
}
