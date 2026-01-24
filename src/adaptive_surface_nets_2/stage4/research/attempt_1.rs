//! Attempt 1: Adaptive RANSAC with Crossing-Count Preclassification
//!
//! This attempt uses the crossing-count signal from surface location to
//! preclassify vertices, then applies an adaptive, full-sphere RANSAC
//! pipeline for edges. It avoids tangent-plane bias for edge candidates
//! and adds targeted probes when one plane is under-supported.

use super::attempt_0::{locate_surface, CrossingCountConfig, GeometryType, VertexGeometry};
use super::sample_cache::{find_crossing_in_direction, SampleCache};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Clone, Debug)]
pub struct Attempt1Config {
    pub locate_config: CrossingCountConfig,
    pub locate_use_hint_normal: bool,
    pub face_probe_count: usize,
    pub edge_probe_start: usize,
    pub edge_probe_step: usize,
    pub edge_probe_max: usize,
    pub edge_refine_probes: usize,
    pub corner_probe_count: usize,
    pub ransac_iterations: usize,
    pub face_inlier_threshold: f64,
    pub edge_inlier_threshold: f64,
    pub corner_inlier_threshold: f64,
    pub face_residual_threshold: f64,
    pub edge_residual_threshold: f64,
    pub corner_residual_threshold: f64,
    pub min_edge_angle_deg: f64,
    pub weak_plane_inlier_ratio: f64,
    pub min_edge_inliers: usize,
    pub edge_plane_distance_max: f64,
}

impl Default for Attempt1Config {
    fn default() -> Self {
        let mut locate_config = CrossingCountConfig::default();
        locate_config.binary_search_iterations = 12;
        locate_config.search_distance = 0.35;
        Self {
            locate_config,
            locate_use_hint_normal: false,
            face_probe_count: 24,
            edge_probe_start: 32,
            edge_probe_step: 16,
            edge_probe_max: 80,
            edge_refine_probes: 12,
            corner_probe_count: 80,
            ransac_iterations: 200,
            face_inlier_threshold: 0.005,
            edge_inlier_threshold: 0.01,
            corner_inlier_threshold: 0.006,
            face_residual_threshold: 0.002,
            edge_residual_threshold: 0.02,
            corner_residual_threshold: 0.005,
            min_edge_angle_deg: 30.0,
            weak_plane_inlier_ratio: 0.6,
            min_edge_inliers: 6,
            edge_plane_distance_max: 0.2,
        }
    }
}

// ============================================================================
// Entry Point
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct Attempt1Diag {
    pub crossing_count: usize,
    pub face_points: usize,
    pub edge_points: usize,
    pub corner_points: usize,
    pub face_residual: Option<f64>,
    pub edge_residuals: Option<(f64, f64)>,
    pub edge_inliers: Option<(usize, usize)>,
    pub edge_angle_deg: Option<f64>,
    pub corner_residuals: Option<Vec<f64>>,
    pub samples_used: u32,
}

pub fn process_vertex<F>(
    midpoint: (f64, f64, f64),
    accumulated_normal: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &Attempt1Config,
) -> VertexGeometry
where
    F: Fn(f64, f64, f64) -> f32,
{
    let (result, _diag) = process_vertex_with_diag(
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
    config: &Attempt1Config,
) -> (VertexGeometry, Attempt1Diag)
where
    F: Fn(f64, f64, f64) -> f32,
{
    let location = locate_surface(midpoint, accumulated_normal, cell_size, cache, &config.locate_config);
    let location = if config.locate_use_hint_normal {
        location
    } else {
        locate_surface(midpoint, (0.0, 0.0, 0.0), cell_size, cache, &config.locate_config)
    };
    let mut total_samples = location.samples_used;
    let mut diag = Attempt1Diag {
        crossing_count: location.crossing_count,
        ..Attempt1Diag::default()
    };

    let inside_point = find_inside_point(location.position, accumulated_normal, cell_size, cache);
    let mut face_fit = None;
    if location.crossing_count >= 3 {
        let hint = normalized_or_fallback(accumulated_normal);
        if let Some(fit) = measure_face_ransac(
            inside_point,
            location.position,
            hint,
            cell_size,
            cache,
            config,
        ) {
            total_samples += fit.samples_used;
            diag.face_residual = Some(fit.residual);
            diag.face_points = fit.num_points;
            face_fit = Some(fit);
        }
    }

    if location.crossing_count <= 1 {
        if let Some(corner) = measure_corner_ransac(
            inside_point,
            location.position,
            cell_size,
            cache,
            config,
        ) {
            total_samples += corner.samples_used;
            let corner_residual = corner.residuals.iter().sum::<f64>() / corner.residuals.len() as f64;
            diag.corner_residuals = Some(corner.residuals.clone());
            diag.corner_points = corner.num_points;
            if corner_residual < config.corner_residual_threshold * cell_size && corner.normals.len() >= 3 {
                let result = VertexGeometry {
                    classification: GeometryType::Corner,
                    normals: corner.normals.clone(),
                    edge_direction: None,
                    corner_position: Some(corner.corner_position),
                    confidence: confidence_from_residual(corner_residual),
                    samples_used: total_samples,
                };
                diag.samples_used = total_samples;
                return (result, diag);
            }
        }
    }

    let edge_fit = adaptive_edge_ransac(inside_point, location.position, cell_size, cache, config);
    if let Some(edge) = edge_fit.as_ref() {
        total_samples += edge.samples_used;
        let normal_angle = angle_between(edge.normal_a, edge.normal_b).to_degrees();
        let edge_residual = (edge.residual_a + edge.residual_b) * 0.5;
        diag.edge_residuals = Some((edge.residual_a, edge.residual_b));
        diag.edge_inliers = Some((edge.inliers_a, edge.inliers_b));
        diag.edge_points = edge.surface_points;
        diag.edge_angle_deg = Some(normal_angle);
        if normal_angle >= config.min_edge_angle_deg && edge_residual < config.edge_residual_threshold * cell_size {
            let result = VertexGeometry {
                classification: GeometryType::Edge,
                normals: vec![edge.normal_a, edge.normal_b],
                edge_direction: Some(edge.edge_direction),
                corner_position: None,
                confidence: confidence_from_residual(edge_residual),
                samples_used: total_samples,
            };
            diag.samples_used = total_samples;
            return (result, diag);
        }
    }

    if let Some(edge) = edge_fit {
        let normal_angle = angle_between(edge.normal_a, edge.normal_b).to_degrees();
        if normal_angle >= config.min_edge_angle_deg {
            let edge_residual = (edge.residual_a + edge.residual_b) * 0.5;
            let result = VertexGeometry {
                classification: GeometryType::Edge,
                normals: vec![edge.normal_a, edge.normal_b],
                edge_direction: Some(edge.edge_direction),
                corner_position: None,
                confidence: confidence_from_residual(edge_residual) * 0.7,
                samples_used: total_samples,
            };
            diag.samples_used = total_samples;
            return (result, diag);
        }
    }

    if let Some(face) = face_fit {
        let result = if face.residual < config.face_residual_threshold * cell_size {
            VertexGeometry {
                classification: GeometryType::Face,
                normals: vec![face.normal],
                edge_direction: None,
                corner_position: None,
                confidence: confidence_from_residual(face.residual),
                samples_used: total_samples,
            }
        } else {
            VertexGeometry {
                classification: GeometryType::Face,
                normals: vec![face.normal],
                edge_direction: None,
                corner_position: None,
                confidence: confidence_from_residual(face.residual) * 0.5,
                samples_used: total_samples,
            }
        };
        diag.samples_used = total_samples;
        return (result, diag);
    }

    let result = VertexGeometry {
        classification: GeometryType::Face,
        normals: vec![normalized_or_fallback(accumulated_normal)],
        edge_direction: None,
        corner_position: None,
        confidence: 0.1,
        samples_used: total_samples,
    };
    diag.samples_used = total_samples;
    (result, diag)
}

// ============================================================================
// Face Measurement
// ============================================================================

struct FaceFit {
    normal: (f64, f64, f64),
    residual: f64,
    samples_used: u32,
    num_points: usize,
}

fn measure_face_ransac<F>(
    sample_origin: (f64, f64, f64),
    query_position: (f64, f64, f64),
    normal_hint: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &Attempt1Config,
) -> Option<FaceFit>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    let directions = hemisphere_directions(normal_hint, config.face_probe_count);
    let search_distance = cell_size * config.locate_config.search_distance;

    let (points, used) =
        collect_surface_points(sample_origin, &directions, search_distance, cache, config);
    samples_used += used;

    let threshold = config.face_inlier_threshold * cell_size;
    let plane = ransac_plane_fit(&points, threshold, config.ransac_iterations)?;
    let normal = orient_away(plane.normal, plane.centroid, query_position);

    Some(FaceFit {
        normal,
        residual: plane.residual,
        samples_used,
        num_points: points.len(),
    })
}

// ============================================================================
// Edge Measurement
// ============================================================================

struct EdgeFit {
    normal_a: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
    point_on_edge: (f64, f64, f64),
    inliers_a: usize,
    inliers_b: usize,
    residual_a: f64,
    residual_b: f64,
    samples_used: u32,
    surface_points: usize,
}

fn adaptive_edge_ransac<F>(
    sample_origin: (f64, f64, f64),
    query_position: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &Attempt1Config,
) -> Option<EdgeFit>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    let search_distance = cell_size * config.locate_config.search_distance;
    let directions = fibonacci_sphere_directions(config.edge_probe_max);

    let mut surface_points: Vec<(f64, f64, f64)> = Vec::new();
    let mut target = config.edge_probe_start.min(config.edge_probe_max);
    let mut used_dirs = 0usize;

    while target <= config.edge_probe_max {
        if used_dirs < target {
            let (new_points, used) = collect_surface_points(
            sample_origin,
            &directions[used_dirs..target],
            search_distance,
            cache,
            config,
            );
            samples_used += used;
            surface_points.extend(new_points);
            used_dirs = target;
        }

        if surface_points.len() < config.min_edge_inliers * 2 {
            if target == config.edge_probe_max {
                break;
            }
            target = (target + config.edge_probe_step).min(config.edge_probe_max);
            continue;
        }

        let threshold = config.edge_inlier_threshold * cell_size;
        let (plane_a, plane_b) = match two_plane_ransac(&surface_points, threshold, config.ransac_iterations) {
            Some(value) => value,
            None => {
                if target == config.edge_probe_max {
                    break;
                }
                target = (target + config.edge_probe_step).min(config.edge_probe_max);
                continue;
            }
        };

        let mut plane_a = plane_a;
        let mut plane_b = plane_b;

        if plane_a.inliers.len().min(plane_b.inliers.len()) < config.min_edge_inliers {
            if target == config.edge_probe_max {
                break;
            }
            target = (target + config.edge_probe_step).min(config.edge_probe_max);
            continue;
        }

        let weak_ratio = plane_a.inliers.len().min(plane_b.inliers.len()) as f64
            / plane_a.inliers.len().max(plane_b.inliers.len()) as f64;

        if weak_ratio < config.weak_plane_inlier_ratio && config.edge_refine_probes > 0 {
            let weak_normal = if plane_a.inliers.len() < plane_b.inliers.len() {
                plane_a.normal
            } else {
                plane_b.normal
            };
            let refine_dirs = hemisphere_directions(weak_normal, config.edge_refine_probes);
            let (extra_points, used) = collect_surface_points(
                sample_origin,
                &refine_dirs,
                search_distance,
                cache,
                config,
            );
            samples_used += used;
            if !extra_points.is_empty() {
                surface_points.extend(extra_points);
                if let Some((new_a, new_b)) = two_plane_ransac(&surface_points, threshold, config.ransac_iterations) {
                    plane_a = new_a;
                    plane_b = new_b;
                }
            }
        }

        let plane_distance_limit = config.edge_plane_distance_max * cell_size;
        let dist_a = dot(sub(query_position, plane_a.centroid), plane_a.normal).abs();
        let dist_b = dot(sub(query_position, plane_b.centroid), plane_b.normal).abs();
        if dist_a > plane_distance_limit || dist_b > plane_distance_limit {
            return None;
        }

        let normal_a = orient_away(plane_a.normal, plane_a.centroid, query_position);
        let normal_b = orient_away(plane_b.normal, plane_b.centroid, query_position);
        let edge_direction = normalize(cross(normal_a, normal_b));
        if length(edge_direction) < 1e-6 {
            return None;
        }

        let point_on_edge = find_edge_point(
            query_position,
            plane_a.centroid,
            normal_a,
            plane_b.centroid,
            normal_b,
            edge_direction,
        );

        return Some(EdgeFit {
            normal_a,
            normal_b,
            edge_direction,
            point_on_edge,
            inliers_a: plane_a.inliers.len(),
            inliers_b: plane_b.inliers.len(),
            residual_a: plane_a.residual,
            residual_b: plane_b.residual,
            samples_used,
            surface_points: surface_points.len(),
        });
    }

    None
}

// ============================================================================
// Corner Measurement
// ============================================================================

struct CornerFit {
    normals: Vec<(f64, f64, f64)>,
    residuals: Vec<f64>,
    corner_position: (f64, f64, f64),
    samples_used: u32,
    num_points: usize,
}

fn measure_corner_ransac<F>(
    sample_origin: (f64, f64, f64),
    query_position: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &Attempt1Config,
) -> Option<CornerFit>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    let search_distance = cell_size * config.locate_config.search_distance;
    let directions = fibonacci_sphere_directions(config.corner_probe_count);

    let (points, used) =
        collect_surface_points(sample_origin, &directions, search_distance, cache, config);
    samples_used += used;

    if points.len() < 9 {
        return None;
    }

    let threshold = config.corner_inlier_threshold * cell_size;
    let mut remaining = points.clone();
    let mut planes: Vec<PlaneFit> = Vec::new();

    for _ in 0..3 {
        let plane = ransac_plane_fit(&remaining, threshold, config.ransac_iterations)?;
        let is_duplicate = planes.iter().any(|existing| {
            let angle = angle_between(existing.normal, plane.normal).to_degrees();
            angle < 25.0 || angle > 155.0
        });

        if is_duplicate {
            remaining.retain(|p| !is_inlier(*p, plane.centroid, plane.normal, threshold));
            continue;
        }

        remaining.retain(|p| !is_inlier(*p, plane.centroid, plane.normal, threshold));
        planes.push(plane);
        if remaining.len() < 3 {
            break;
        }
    }

    if planes.len() < 3 {
        return None;
    }

    let normals: Vec<(f64, f64, f64)> = planes
        .iter()
        .map(|p| orient_away(p.normal, p.centroid, query_position))
        .collect();
    let residuals: Vec<f64> = planes.iter().map(|p| p.residual).collect();

    let corner_position = three_plane_intersection(
        (normals[0], planes[0].centroid),
        (normals[1], planes[1].centroid),
        (normals[2], planes[2].centroid),
    )
    .unwrap_or(query_position);

    Some(CornerFit {
        normals,
        residuals,
        corner_position,
        samples_used,
        num_points: points.len(),
    })
}

// ============================================================================
// Sampling Helpers
// ============================================================================

fn collect_surface_points<F>(
    start: (f64, f64, f64),
    directions: &[(f64, f64, f64)],
    max_distance: f64,
    cache: &SampleCache<F>,
    config: &Attempt1Config,
) -> (Vec<(f64, f64, f64)>, u32)
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    let mut points = Vec::new();
    let step = max_distance / 6.0;
    for dir in directions {
        if let Some((crossing, _)) = find_crossing_in_direction(
            cache,
            start,
            *dir,
            max_distance,
            step,
            config.locate_config.binary_search_iterations,
        ) {
            samples_used += 1;
            points.push(crossing);
        } else {
            samples_used += 1;
        }
    }

    (points, samples_used)
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
        neg(hint),
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

fn hemisphere_directions(center: (f64, f64, f64), n: usize) -> Vec<(f64, f64, f64)> {
    let center = normalize(center);
    let (tangent1, tangent2) = build_tangent_basis(center);
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());

    (0..n)
        .map(|i| {
            let t = (i as f64 + 0.5) / n as f64;
            let phi = i as f64 * golden_angle;
            let z = t;
            let r = (1.0 - z * z).sqrt();
            let x = r * phi.cos();
            let y = r * phi.sin();

            normalize((
                tangent1.0 * x + tangent2.0 * y + center.0 * z,
                tangent1.1 * x + tangent2.1 * y + center.1 * z,
                tangent1.2 * x + tangent2.2 * y + center.2 * z,
            ))
        })
        .collect()
}

fn build_tangent_basis(normal: (f64, f64, f64)) -> ((f64, f64, f64), (f64, f64, f64)) {
    let reference = if normal.0.abs() < 0.9 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    };
    let tangent1 = normalize(cross(normal, reference));
    let tangent2 = cross(normal, tangent1);
    (tangent1, tangent2)
}

// ============================================================================
// Plane Fitting
// ============================================================================

#[derive(Clone, Debug)]
struct PlaneFit {
    inliers: Vec<(f64, f64, f64)>,
    centroid: (f64, f64, f64),
    normal: (f64, f64, f64),
    residual: f64,
}

fn two_plane_ransac(
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
    if remaining.len() < 3 {
        return None;
    }
    let plane_b = ransac_plane_fit(&remaining, inlier_threshold, iterations)?;
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

    let (centroid, normal, residual) = fit_plane_svd(&best_inliers);
    Some(PlaneFit {
        inliers: best_inliers,
        centroid,
        normal,
        residual,
    })
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
    let dist = dot(sub(p, plane_point), plane_normal).abs();
    dist < threshold
}

// ============================================================================
// Geometry Utilities
// ============================================================================

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

fn three_plane_intersection(
    plane1: ((f64, f64, f64), (f64, f64, f64)),
    plane2: ((f64, f64, f64), (f64, f64, f64)),
    plane3: ((f64, f64, f64), (f64, f64, f64)),
) -> Option<(f64, f64, f64)> {
    let n1 = plane1.0;
    let n2 = plane2.0;
    let n3 = plane3.0;

    let d1 = dot(n1, plane1.1);
    let d2 = dot(n2, plane2.1);
    let d3 = dot(n3, plane3.1);

    let denom = dot(n1, cross(n2, n3));
    if denom.abs() < 1e-12 {
        return None;
    }

    let p = (
        (d1 * (n2.1 * n3.2 - n2.2 * n3.1)
            + d2 * (n3.1 * n1.2 - n3.2 * n1.1)
            + d3 * (n1.1 * n2.2 - n1.2 * n2.1))
            / denom,
        (d1 * (n2.2 * n3.0 - n2.0 * n3.2)
            + d2 * (n3.2 * n1.0 - n3.0 * n1.2)
            + d3 * (n1.2 * n2.0 - n1.0 * n2.2))
            / denom,
        (d1 * (n2.0 * n3.1 - n2.1 * n3.0)
            + d2 * (n3.0 * n1.1 - n3.1 * n1.0)
            + d3 * (n1.0 * n2.1 - n1.1 * n2.0))
            / denom,
    );

    Some(p)
}

fn orient_away(
    normal: (f64, f64, f64),
    plane_point: (f64, f64, f64),
    away_from: (f64, f64, f64),
) -> (f64, f64, f64) {
    let to_plane = sub(plane_point, away_from);
    if dot(normal, to_plane) < 0.0 {
        neg(normal)
    } else {
        normal
    }
}

fn confidence_from_residual(residual: f64) -> f64 {
    1.0 / (1.0 + residual * 100.0)
}

fn normalized_or_fallback(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let len = length(v);
    if len > 1e-12 {
        (v.0 / len, v.1 / len, v.2 / len)
    } else {
        (0.0, 1.0, 0.0)
    }
}

// ============================================================================
// Vector Math
// ============================================================================

fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let na = normalize(a);
    let nb = normalize(b);
    dot(na, nb).clamp(-1.0, 1.0).acos()
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
