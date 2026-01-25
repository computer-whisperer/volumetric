//! Attempt 2: Fixed-Budget RANSAC Using Crossing Count
//!
//! Sample the surface once with a fixed, unbiased full-sphere probe set,
//! then reuse those samples for face/edge/corner fits.

use super::attempt_0::{locate_surface, CrossingCountConfig, GeometryType, VertexGeometry};
use super::sample_cache::{find_crossing_in_direction, SampleCache};

#[derive(Clone, Debug)]
pub struct Attempt2Config {
    pub locate_config: CrossingCountConfig,
    pub face_probes: usize,
    pub edge_probes: usize,
    pub corner_probes: usize,
    pub max_distance_scale: f64,
    pub ransac_iterations: usize,
    pub face_inlier_threshold: f64,
    pub edge_inlier_threshold: f64,
    pub corner_inlier_threshold: f64,
    pub edge_exclusion_scale: f64,
    pub edge_line_distance_min: f64,
    pub edge_cone_angle_deg: f64,
    pub edge_hint_offset: f64,
}

impl Default for Attempt2Config {
    fn default() -> Self {
        let mut locate_config = CrossingCountConfig::default();
        locate_config.binary_search_iterations = 12;
        locate_config.search_distance = 0.5;
        Self {
            locate_config,
            face_probes: 64,
            edge_probes: 64,
            corner_probes: 96,
            max_distance_scale: 1.0,
            ransac_iterations: 200,
            face_inlier_threshold: 0.005,
            edge_inlier_threshold: 0.01,
            corner_inlier_threshold: 0.006,
            edge_exclusion_scale: 3.0,
            edge_line_distance_min: 0.05,
            edge_cone_angle_deg: 10.0,
            edge_hint_offset: 0.1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Attempt2Diag {
    pub crossing_count: usize,
    pub hint_available: bool,
    pub surface_points: usize,
    pub edge_points: usize,
    pub hint_hist: Option<[usize; 6]>,
}

pub fn process_vertex<F>(
    midpoint: (f64, f64, f64),
    accumulated_normal: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &Attempt2Config,
) -> VertexGeometry
where
    F: Fn(f64, f64, f64) -> f32,
{
    let (result, _) = process_vertex_with_diag(
        midpoint,
        accumulated_normal,
        cell_size,
        cache,
        config,
    );
    result
}

pub fn process_vertex_with_diag<F>(
    midpoint: (f64, f64, f64),
    accumulated_normal: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &Attempt2Config,
) -> (VertexGeometry, Attempt2Diag)
where
    F: Fn(f64, f64, f64) -> f32,
{
    let location = locate_surface(midpoint, accumulated_normal, cell_size, cache, &config.locate_config);
    let inside_point = find_inside_point(location.position, accumulated_normal, cell_size, cache);
    let max_distance = cell_size * config.max_distance_scale;

    let edge_hints = if location.crossing_directions.len() >= 2 {
        Some([
            normalize(location.crossing_directions[0]),
            normalize(location.crossing_directions[1]),
        ])
    } else {
        None
    };

    let max_probes = config.face_probes.max(config.corner_probes);
    let directions_full = fibonacci_sphere_directions(max_probes);
    let surface_points = collect_surface_points(
        inside_point,
        &directions_full,
        max_distance,
        config.locate_config.binary_search_iterations,
        cache,
    );

    let edge_points = if let Some(hints) = edge_hints {
        let offset = config.edge_hint_offset * cell_size;
        let origin_a = add(location.position, scale(hints[0], offset));
        let origin_b = add(location.position, scale(hints[1], offset));
        let directions_a = cone_directions(hints[0], hints[0], config.edge_probes / 2, config.edge_cone_angle_deg);
        let directions_b = cone_directions(hints[1], hints[1], config.edge_probes - config.edge_probes / 2, config.edge_cone_angle_deg);

        let mut points = collect_surface_points(
            origin_a,
            &directions_a,
            max_distance,
            config.locate_config.binary_search_iterations,
            cache,
        );
        points.extend(collect_surface_points(
            origin_b,
            &directions_b,
            max_distance,
            config.locate_config.binary_search_iterations,
            cache,
        ));
        points
    } else {
        let directions_edge = fibonacci_sphere_directions(config.edge_probes);
        collect_surface_points(
            inside_point,
            &directions_edge,
            max_distance,
            config.locate_config.binary_search_iterations,
            cache,
        )
    };

    let hint_hist = edge_hints.map(|hints| hint_histogram(&edge_points, location.position, hints));
    let diag = Attempt2Diag {
        crossing_count: location.crossing_count,
        hint_available: edge_hints.is_some(),
        surface_points: surface_points.len(),
        edge_points: edge_points.len(),
        hint_hist,
    };

    let face_fit = || fit_face(&surface_points, config, inside_point);
    let edge_fit = || {
        let points = if edge_points.is_empty() {
            surface_points.as_slice()
        } else {
            edge_points.as_slice()
        };
        fit_edge(
            points,
            config,
            inside_point,
            location.position,
            edge_hints,
            location.crossing_count,
            cell_size,
        )
    };
    let corner_fit = || fit_corner(&surface_points, config, inside_point);

    match location.crossing_count {
        3 | 4 => {
            if let Some(face) = face_fit() {
                return (face, diag);
            }
            if let Some(edge) = edge_fit() {
                return (edge, diag);
            }
        }
        2 => {
            if let Some(edge) = edge_fit() {
                return (edge, diag);
            }
            if let Some(face) = face_fit() {
                return (face, diag);
            }
        }
        _ => {
            if let Some(corner) = corner_fit() {
                return (corner, diag);
            }
            if let Some(edge) = edge_fit() {
                return (edge, diag);
            }
        }
    }

    if let Some(face) = face_fit() {
        return (face, diag);
    }

    let result = VertexGeometry {
        classification: GeometryType::Face,
        normals: vec![normalized_or_fallback(accumulated_normal)],
        edge_direction: None,
        corner_position: None,
        confidence: 0.1,
        samples_used: 0,
    };
    (result, diag)
}

fn fit_face(
    points: &[(f64, f64, f64)],
    config: &Attempt2Config,
    query_point: (f64, f64, f64),
) -> Option<VertexGeometry> {
    let plane = ransac_plane_fit(points, config.face_inlier_threshold, config.ransac_iterations)?;
    let normal = orient_away(plane.normal, plane.centroid, query_point);
    Some(VertexGeometry {
        classification: GeometryType::Face,
        normals: vec![normal],
        edge_direction: None,
        corner_position: None,
        confidence: confidence_from_residual(plane.residual),
        samples_used: 0,
    })
}

fn fit_edge(
    points: &[(f64, f64, f64)],
    config: &Attempt2Config,
    query_point: (f64, f64, f64),
    surface_point: (f64, f64, f64),
    hints: Option<[(f64, f64, f64); 2]>,
    crossing_count: usize,
    cell_size: f64,
) -> Option<VertexGeometry> {
    let mut filtered_points: Vec<(f64, f64, f64)> = points.to_vec();
    if let Some(hints) = hints {
        let edge_dir_hint = normalize(cross(hints[0], hints[1]));
        if length(edge_dir_hint) > 1e-6 {
            let min_dist = config.edge_line_distance_min * cell_size;
            let candidates: Vec<_> = points
                .iter()
                .cloned()
                .filter(|p| distance_to_line(*p, surface_point, edge_dir_hint) >= min_dist)
                .collect();
            if candidates.len() >= 6 {
                filtered_points = candidates;
            }
        }
    }

    let (plane_a, plane_b, mode) = if let Some(hints) = hints {
        let result = two_plane_ransac_partitioned(
            &filtered_points,
            config.edge_inlier_threshold,
            config.ransac_iterations,
            query_point,
            hints,
            config.edge_exclusion_scale,
        )?;
        (result.0, result.1, "partitioned")
    } else {
        let result = two_plane_ransac_exclusion(
            &filtered_points,
            config.edge_inlier_threshold,
            config.ransac_iterations,
            config.edge_exclusion_scale,
        )?;
        (result.0, result.1, "exclusion")
    };

    let normal_a = orient_away(plane_a.normal, plane_a.centroid, query_point);
    let normal_b = orient_away(plane_b.normal, plane_b.centroid, query_point);
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None;
    }

    let _edge_point =
        find_edge_point(surface_point, plane_a.centroid, normal_a, plane_b.centroid, normal_b, edge_dir);

    let geometry = VertexGeometry {
        classification: GeometryType::Edge,
        normals: vec![normal_a, normal_b],
        edge_direction: Some(edge_dir),
        corner_position: None,
        confidence: confidence_from_residual((plane_a.residual + plane_b.residual) * 0.5),
        samples_used: 0,
    };

    diag_edge_fit(
        points,
        &plane_a,
        &plane_b,
        mode,
        crossing_count,
        query_point,
        surface_point,
        hints,
    );
    Some(geometry)
}

fn fit_corner(
    points: &[(f64, f64, f64)],
    config: &Attempt2Config,
    query_point: (f64, f64, f64),
) -> Option<VertexGeometry> {
    let planes = three_plane_ransac(
        points,
        config.corner_inlier_threshold,
        config.ransac_iterations,
    )?;

    let normals: Vec<(f64, f64, f64)> = planes
        .iter()
        .map(|p| orient_away(p.normal, p.centroid, query_point))
        .collect();

    if normals.len() < 3 {
        return None;
    }

    let corner_position = solve_3_planes(
        normals[0], dot(normals[0], planes[0].centroid),
        normals[1], dot(normals[1], planes[1].centroid),
        normals[2], dot(normals[2], planes[2].centroid),
    )
    .unwrap_or(query_point);

    let avg_residual = planes.iter().map(|p| p.residual).sum::<f64>() / planes.len() as f64;

    Some(VertexGeometry {
        classification: GeometryType::Corner,
        normals,
        edge_direction: None,
        corner_position: Some(corner_position),
        confidence: confidence_from_residual(avg_residual),
        samples_used: 0,
    })
}

fn collect_surface_points<F>(
    start: (f64, f64, f64),
    directions: &[(f64, f64, f64)],
    max_distance: f64,
    iterations: usize,
    cache: &SampleCache<F>,
) -> Vec<(f64, f64, f64)>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut points = Vec::new();
    let step = max_distance / 6.0;
    for dir in directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache,
            start,
            *dir,
            max_distance,
            step,
            iterations,
        ) {
            points.push(crossing);
        }
    }
    points
}

fn confidence_from_residual(residual: f64) -> f64 {
    1.0 / (1.0 + residual * 100.0)
}

fn find_inside_point<F>(
    position: (f64, f64, f64),
    hint_normal: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    let step = cell_size * 0.2;
    let hint = normalized_or_fallback(hint_normal);
    let candidates = [
        hint,
        (-hint.0, -hint.1, -hint.2),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ];

    for dir in candidates {
        let probe = (
            position.0 + dir.0 * step,
            position.1 + dir.1 * step,
            position.2 + dir.2 * step,
        );
        if cache.is_inside(probe.0, probe.1, probe.2) {
            return probe;
        }
    }

    position
}

#[derive(Clone, Debug)]
struct PlaneFit {
    inliers: Vec<(f64, f64, f64)>,
    centroid: (f64, f64, f64),
    normal: (f64, f64, f64),
    residual: f64,
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
            .filter(|p| dot(sub(**p, p1), normal).abs() < inlier_threshold)
            .cloned()
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }

    if best_inliers.len() < 3 {
        return None;
    }

    let (centroid, normal, residual) = fit_plane_svd(&best_inliers);
    Some(PlaneFit {
        inliers: best_inliers,
        centroid,
        normal,
        residual,
    })
}

fn two_plane_ransac_exclusion(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
    exclusion_scale: f64,
) -> Option<(PlaneFit, PlaneFit)> {
    let plane_a = ransac_plane_fit(points, inlier_threshold, iterations)?;
    let exclusion_threshold = inlier_threshold * exclusion_scale;
    let remaining: Vec<_> = points
        .iter()
        .filter(|p| !is_inlier(**p, plane_a.centroid, plane_a.normal, exclusion_threshold))
        .cloned()
        .collect();
    if remaining.len() < 3 {
        return None;
    }
    let plane_b = ransac_plane_fit(&remaining, inlier_threshold, iterations)?;
    Some((plane_a, plane_b))
}

fn two_plane_ransac_partitioned(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
    origin: (f64, f64, f64),
    hints: [(f64, f64, f64); 2],
    exclusion_scale: f64,
) -> Option<(PlaneFit, PlaneFit)> {
    let (mut cluster_a, mut cluster_b) = (Vec::new(), Vec::new());
    for p in points {
        let dir = normalize(sub(*p, origin));
        let dot_a = dot(dir, hints[0]);
        let dot_b = dot(dir, hints[1]);
        if dot_a >= dot_b {
            cluster_a.push(*p);
        } else {
            cluster_b.push(*p);
        }
    }

    if cluster_a.len() < 3 || cluster_b.len() < 3 {
        return None;
    }

    let plane_a = ransac_plane_fit(&cluster_a, inlier_threshold, iterations)?;
    let exclusion_threshold = inlier_threshold * exclusion_scale;
    let filtered_b: Vec<_> = cluster_b
        .into_iter()
        .filter(|p| !is_inlier(*p, plane_a.centroid, plane_a.normal, exclusion_threshold))
        .collect();
    if filtered_b.len() < 3 {
        return None;
    }
    let plane_b = ransac_plane_fit(&filtered_b, inlier_threshold, iterations)?;
    Some((plane_a, plane_b))
}

fn three_plane_ransac(
    points: &[(f64, f64, f64)],
    inlier_threshold: f64,
    iterations: usize,
) -> Option<Vec<PlaneFit>> {
    let mut planes = Vec::new();
    let mut remaining = points.to_vec();

    for _ in 0..3 {
        let plane = ransac_plane_fit(&remaining, inlier_threshold, iterations)?;
        let is_duplicate = planes.iter().any(|existing: &PlaneFit| {
            let angle = angle_between(existing.normal, plane.normal).to_degrees();
            angle < 25.0 || angle > 155.0
        });
        remaining.retain(|p| !is_inlier(*p, plane.centroid, plane.normal, inlier_threshold));
        if !is_duplicate {
            planes.push(plane);
        }
        if remaining.len() < 3 {
            break;
        }
    }

    if planes.len() < 3 {
        None
    } else {
        Some(planes)
    }
}

fn fit_plane_svd(points: &[(f64, f64, f64)]) -> ((f64, f64, f64), (f64, f64, f64), f64) {
    let n = points.len() as f64;
    let centroid = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
        points.iter().map(|p| p.2).sum::<f64>() / n,
    );

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

    let max_ev = cov
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row[i].abs() + row.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v.abs()).sum::<f64>()
        })
        .fold(0.0_f64, f64::max);

    let shift = max_ev + 1.0;
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

    let mut residual_sum = 0.0;
    for p in &centered {
        residual_sum += dot(*p, v).powi(2);
    }
    let rms = (residual_sum / points.len() as f64).sqrt();

    (centroid, v, rms)
}

fn is_inlier(
    p: (f64, f64, f64),
    plane_point: (f64, f64, f64),
    plane_normal: (f64, f64, f64),
    threshold: f64,
) -> bool {
    dot(sub(p, plane_point), plane_normal).abs() < threshold
}

fn orient_away(
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

fn find_edge_point(
    query: (f64, f64, f64),
    centroid_a: (f64, f64, f64),
    normal_a: (f64, f64, f64),
    centroid_b: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
) -> (f64, f64, f64) {
    let midpoint = (
        (centroid_a.0 + centroid_b.0) * 0.5,
        (centroid_a.1 + centroid_b.1) * 0.5,
        (centroid_a.2 + centroid_b.2) * 0.5,
    );

    let mut p = midpoint;
    for _ in 0..20 {
        let dist_a = dot(sub(p, centroid_a), normal_a);
        p = sub(p, scale(normal_a, dist_a));
        let dist_b = dot(sub(p, centroid_b), normal_b);
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

fn solve_3_planes(
    n0: (f64, f64, f64),
    d0: f64,
    n1: (f64, f64, f64),
    d1: f64,
    n2: (f64, f64, f64),
    d2: f64,
) -> Option<(f64, f64, f64)> {
    let denom = dot(n0, cross(n1, n2));
    if denom.abs() < 1e-12 {
        return None;
    }

    let p = (
        (d0 * (n1.1 * n2.2 - n1.2 * n2.1)
            + d1 * (n2.1 * n0.2 - n2.2 * n0.1)
            + d2 * (n0.1 * n1.2 - n0.2 * n1.1))
            / denom,
        (d0 * (n1.2 * n2.0 - n1.0 * n2.2)
            + d1 * (n2.2 * n0.0 - n2.0 * n0.2)
            + d2 * (n0.2 * n1.0 - n0.0 * n1.2))
            / denom,
        (d0 * (n1.0 * n2.1 - n1.1 * n2.0)
            + d1 * (n2.0 * n0.1 - n2.1 * n0.0)
            + d2 * (n0.0 * n1.1 - n0.1 * n1.0))
            / denom,
    );

    Some(p)
}

fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    dot(normalize(a), normalize(b)).clamp(-1.0, 1.0).acos()
}

fn distance_to_line(
    p: (f64, f64, f64),
    line_point: (f64, f64, f64),
    line_dir: (f64, f64, f64),
) -> f64 {
    let v = sub(p, line_point);
    let cross_v = cross(v, line_dir);
    length(cross_v)
}

fn fibonacci_sphere_directions(n: usize) -> Vec<(f64, f64, f64)> {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let angle_increment = std::f64::consts::PI * 2.0 * golden_ratio;

    (0..n)
        .map(|i| {
            let t = (i as f64 + 0.5) / n as f64;
            let phi = angle_increment * i as f64;
            let theta = (1.0 - 2.0 * t).acos();

            (
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            )
        })
        .collect()
}

fn edge_orthogonal_directions(hints: [(f64, f64, f64); 2], n: usize) -> Vec<(f64, f64, f64)> {
    let edge_dir = normalize(cross(hints[0], hints[1]));
    if length(edge_dir) < 1e-6 {
        return fibonacci_sphere_directions(n);
    }
    let t1 = normalize(cross(edge_dir, (1.0, 0.0, 0.0)));
    let t1 = if length(t1) < 1e-6 {
        normalize(cross(edge_dir, (0.0, 1.0, 0.0)))
    } else {
        t1
    };
    let t2 = cross(edge_dir, t1);

    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let theta = i as f64 * golden_angle;
            let dir = (
                t1.0 * theta.cos() + t2.0 * theta.sin(),
                t1.1 * theta.cos() + t2.1 * theta.sin(),
                t1.2 * theta.cos() + t2.2 * theta.sin(),
            );
            normalize(dir)
        })
        .collect()
}

fn cone_directions(
    center_a: (f64, f64, f64),
    center_b: (f64, f64, f64),
    n: usize,
    angle_deg: f64,
) -> Vec<(f64, f64, f64)> {
    let half = (n / 2).max(1);
    let angle = angle_deg.to_radians();
    let mut dirs = Vec::with_capacity(n);
    dirs.extend(cone_directions_one(center_a, half, angle));
    dirs.extend(cone_directions_one(center_b, n - half, angle));
    dirs
}

fn cone_directions_one(center: (f64, f64, f64), n: usize, angle: f64) -> Vec<(f64, f64, f64)> {
    let center = normalize(center);
    let (t1, t2) = tangent_basis(center);
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());

    (0..n)
        .map(|i| {
            let t = (i as f64 + 0.5) / n as f64;
            let theta = (t.sqrt() * angle).min(angle);
            let phi = i as f64 * golden_angle;
            let sin_t = theta.sin();
            let dir = (
                center.0 * theta.cos() + t1.0 * sin_t * phi.cos() + t2.0 * sin_t * phi.sin(),
                center.1 * theta.cos() + t1.1 * sin_t * phi.cos() + t2.1 * sin_t * phi.sin(),
                center.2 * theta.cos() + t1.2 * sin_t * phi.cos() + t2.2 * sin_t * phi.sin(),
            );
            normalize(dir)
        })
        .collect()
}

fn tangent_basis(normal: (f64, f64, f64)) -> ((f64, f64, f64), (f64, f64, f64)) {
    let reference = if normal.0.abs() < 0.9 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    };
    let t1 = normalize(cross(normal, reference));
    let t2 = cross(normal, t1);
    (t1, t2)
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

fn normalized_or_fallback(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = length(v);
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 1.0, 0.0)
    }
}

fn diag_edge_fit(
    points: &[(f64, f64, f64)],
    plane_a: &PlaneFit,
    plane_b: &PlaneFit,
    mode: &str,
    crossing_count: usize,
    query_point: (f64, f64, f64),
    surface_point: (f64, f64, f64),
    hints: Option<[(f64, f64, f64); 2]>,
) {
    if std::env::var("ATTEMPT2_DIAG").ok().as_deref() != Some("1") {
        return;
    }
    let angle = angle_between(plane_a.normal, plane_b.normal).to_degrees();
    let dist_a = dot(sub(query_point, plane_a.centroid), plane_a.normal).abs();
    let dist_b = dot(sub(query_point, plane_b.centroid), plane_b.normal).abs();

    println!("\nATTEMPT2_EDGE_DIAG mode={}", mode);
    println!("  points={} crossing_count={}", points.len(), crossing_count);
    println!(
        "  plane_a inliers={} residual={:.6} dist={:.6}",
        plane_a.inliers.len(),
        plane_a.residual,
        dist_a
    );
    println!(
        "  plane_b inliers={} residual={:.6} dist={:.6}",
        plane_b.inliers.len(),
        plane_b.residual,
        dist_b
    );
    println!("  plane angle={:.2} deg", angle);
    println!(
        "  normal_a=({:.3},{:.3},{:.3})",
        plane_a.normal.0,
        plane_a.normal.1,
        plane_a.normal.2
    );
    println!(
        "  normal_b=({:.3},{:.3},{:.3})",
        plane_b.normal.0,
        plane_b.normal.1,
        plane_b.normal.2
    );

    if let Some(hints) = hints {
        let angle_a0 = angle_between(plane_a.normal, hints[0]).to_degrees();
        let angle_a1 = angle_between(plane_a.normal, hints[1]).to_degrees();
        let angle_b0 = angle_between(plane_b.normal, hints[0]).to_degrees();
        let angle_b1 = angle_between(plane_b.normal, hints[1]).to_degrees();
        let hint_angle = angle_between(hints[0], hints[1]).to_degrees();

        let mut counts = [0usize, 0usize];
        let mut hist = [0usize; 6];
        for p in points {
            let dir = normalize(sub(*p, surface_point));
            let dot_a = dot(dir, hints[0]);
            let dot_b = dot(dir, hints[1]);
            if dot_a >= dot_b {
                counts[0] += 1;
            } else {
                counts[1] += 1;
            }
            let best_dot = dot_a.max(dot_b);
            let bin = if best_dot < 0.0 {
                0
            } else if best_dot < 0.25 {
                1
            } else if best_dot < 0.5 {
                2
            } else if best_dot < 0.75 {
                3
            } else if best_dot < 0.9 {
                4
            } else {
                5
            };
            hist[bin] += 1;
        }

        println!(
            "  hint_angle={:.2} deg | cluster_counts=({}, {})",
            hint_angle, counts[0], counts[1]
        );
        println!(
            "  hint_dot_hist=[<0, <0.25, <0.5, <0.75, <0.9, >=0.9]=[{}, {}, {}, {}, {}, {}]",
            hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]
        );
        println!(
            "  plane_a vs hints: {:.2} / {:.2} deg",
            angle_a0, angle_a1
        );
        println!(
            "  plane_b vs hints: {:.2} / {:.2} deg",
            angle_b0, angle_b1
        );
    }
}

fn hint_histogram(
    points: &[(f64, f64, f64)],
    surface_point: (f64, f64, f64),
    hints: [(f64, f64, f64); 2],
) -> [usize; 6] {
    let mut hist = [0usize; 6];
    for p in points {
        let dir = normalize(sub(*p, surface_point));
        let dot_a = dot(dir, hints[0]);
        let dot_b = dot(dir, hints[1]);
        let best_dot = dot_a.max(dot_b);
        let bin = if best_dot < 0.0 {
            0
        } else if best_dot < 0.25 {
            1
        } else if best_dot < 0.5 {
            2
        } else if best_dot < 0.75 {
            3
        } else if best_dot < 0.9 {
            4
        } else {
            5
        };
        hist[bin] += 1;
    }
    hist
}
