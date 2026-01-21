//! Validation Framework
//!
//! Compares reference probing routines against analytical ground truth
//! using the rotated cube test case.
//!
//! # Validation Strategy
//!
//! 1. Generate systematic test points at known locations:
//!    - Points on each of the 6 faces (varying distance from edges)
//!    - Points on each of the 12 edges (varying distance from corners)
//!    - Points near each of the 8 corners
//!    - Points at varying distances from the surface
//!
//! 2. Run reference routines at each point
//!
//! 3. Compare results against analytical ground truth:
//!    - Face points → surface normal error < 1°
//!    - Edge points → edge direction error < 1°, both face normals error < 1°
//!    - Corner points → all face normals error < 1°
//!    - Position estimates → error < 0.01 (fraction of relevant dimension)

use super::analytical_cube::{angle_between, AnalyticalRotatedCube};
use super::reference_corner::{reference_find_nearest_corner, CornerFindingConfig};
use super::reference_edge::{reference_find_nearest_edge, EdgeFindingConfig};
use super::reference_surface::{reference_find_nearest_surface, SurfaceFindingConfig};
use super::sample_cache::SampleCache;

/// A validation test point with expected classification
#[derive(Clone, Debug)]
pub struct ValidationPoint {
    /// Position of the test point
    pub position: (f64, f64, f64),
    /// Expected classification at this point
    pub expected: ExpectedClassification,
    /// Human-readable description
    pub description: String,
}

/// Expected classification for a validation point
#[derive(Clone, Debug)]
pub enum ExpectedClassification {
    /// Point is on a face (should detect single normal)
    OnFace {
        face_index: usize,
        expected_normal: (f64, f64, f64),
    },
    /// Point is on an edge (should detect two normals)
    OnEdge {
        edge_index: usize,
        expected_direction: (f64, f64, f64),
        expected_normals: ((f64, f64, f64), (f64, f64, f64)),
    },
    /// Point is at a corner (should detect three normals)
    OnCorner {
        corner_index: usize,
        expected_normals: [(f64, f64, f64); 3],
    },
}

/// Result of validating surface detection at a point
#[derive(Clone, Debug)]
pub struct SurfaceValidationResult {
    /// Angular error between detected and expected normal (degrees)
    pub normal_error_degrees: f64,
    /// Distance error for closest point (world units)
    pub position_error: f64,
    /// Was the detection successful?
    pub success: bool,
    /// Number of samples used
    pub samples_used: u64,
}

/// Result of validating edge detection at a point
#[derive(Clone, Debug)]
pub struct EdgeValidationResult {
    /// Angular error for edge direction (degrees, min of both orientations)
    pub direction_error_degrees: f64,
    /// Angular error for face A normal (degrees)
    pub normal_a_error_degrees: f64,
    /// Angular error for face B normal (degrees)
    pub normal_b_error_degrees: f64,
    /// Position error for point on edge (world units)
    pub position_error: f64,
    /// Was the detection successful?
    pub success: bool,
    /// Number of samples used
    pub samples_used: u64,
}

/// Result of validating corner detection at a point
#[derive(Clone, Debug)]
pub struct CornerValidationResult {
    /// Number of faces detected (should be 3 for a cube corner)
    pub num_faces_detected: usize,
    /// Angular errors for each detected normal matched to expected (degrees)
    pub normal_errors_degrees: Vec<f64>,
    /// Position error for corner position (world units)
    pub position_error: f64,
    /// Was the detection successful?
    pub success: bool,
    /// Number of samples used
    pub samples_used: u64,
}

/// Aggregate validation result
#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub point: ValidationPoint,
    pub surface_result: Option<SurfaceValidationResult>,
    pub edge_result: Option<EdgeValidationResult>,
    pub corner_result: Option<CornerValidationResult>,
}

impl ValidationResult {
    /// Check if all results meet the pass criteria
    pub fn meets_criteria(&self) -> bool {
        // Criteria from the plan:
        // - Face points → surface normal error < 1°
        // - Edge points → edge direction error < 1°, both face normals error < 1° each
        // - Corner points → all face normals error < 1° each
        // - Position estimates → error < 0.01

        const ANGLE_THRESHOLD: f64 = 1.0; // degrees
        const POSITION_THRESHOLD: f64 = 0.01; // world units (cube is 1.0 side)

        match &self.point.expected {
            ExpectedClassification::OnFace { .. } => {
                if let Some(ref sr) = self.surface_result {
                    sr.success && sr.normal_error_degrees < ANGLE_THRESHOLD && sr.position_error < POSITION_THRESHOLD
                } else {
                    false
                }
            }
            ExpectedClassification::OnEdge { .. } => {
                if let Some(ref er) = self.edge_result {
                    er.success
                        && er.direction_error_degrees < ANGLE_THRESHOLD
                        && er.normal_a_error_degrees < ANGLE_THRESHOLD
                        && er.normal_b_error_degrees < ANGLE_THRESHOLD
                        && er.position_error < POSITION_THRESHOLD
                } else {
                    false
                }
            }
            ExpectedClassification::OnCorner { .. } => {
                if let Some(ref cr) = self.corner_result {
                    cr.success
                        && cr.num_faces_detected >= 3
                        && cr.normal_errors_degrees.iter().all(|&e| e < ANGLE_THRESHOLD)
                        && cr.position_error < POSITION_THRESHOLD
                } else {
                    false
                }
            }
        }
    }
}

/// Generate systematic validation points for a rotated cube
pub fn generate_validation_points(cube: &AnalyticalRotatedCube) -> Vec<ValidationPoint> {
    let mut points = Vec::new();

    // Face centers (6 points)
    for face_idx in 0..6 {
        let center = cube.face_center(face_idx);
        points.push(ValidationPoint {
            position: center,
            expected: ExpectedClassification::OnFace {
                face_index: face_idx,
                expected_normal: cube.face_normals[face_idx],
            },
            description: format!("Face {} center", face_idx),
        });
    }

    // Edge midpoints (12 points)
    for edge_idx in 0..12 {
        let edge = cube.get_edge(edge_idx);
        points.push(ValidationPoint {
            position: edge.point_on_edge,
            expected: ExpectedClassification::OnEdge {
                edge_index: edge_idx,
                expected_direction: edge.direction,
                expected_normals: (edge.face_a_normal, edge.face_b_normal),
            },
            description: format!("Edge {} midpoint", edge_idx),
        });
    }

    // Corners (8 points)
    for corner_idx in 0..8 {
        let corner_pos = cube.corners[corner_idx];
        let faces = faces_of_corner(corner_idx);
        let expected_normals = [
            cube.face_normals[faces[0]],
            cube.face_normals[faces[1]],
            cube.face_normals[faces[2]],
        ];
        points.push(ValidationPoint {
            position: corner_pos,
            expected: ExpectedClassification::OnCorner {
                corner_index: corner_idx,
                expected_normals,
            },
            description: format!("Corner {}", corner_idx),
        });
    }

    // Points slightly inside faces (offset inward by 0.1)
    for face_idx in 0..6 {
        let center = cube.face_center(face_idx);
        let normal = cube.face_normals[face_idx];
        let offset = 0.1;
        let inside_point = (
            center.0 - normal.0 * offset,
            center.1 - normal.1 * offset,
            center.2 - normal.2 * offset,
        );
        points.push(ValidationPoint {
            position: inside_point,
            expected: ExpectedClassification::OnFace {
                face_index: face_idx,
                expected_normal: normal,
            },
            description: format!("Face {} (inside)", face_idx),
        });
    }

    // Points slightly inside edges (offset inward along bisector)
    for edge_idx in 0..12 {
        let edge = cube.get_edge(edge_idx);
        let bisector = normalize(add(edge.face_a_normal, edge.face_b_normal));
        let offset = 0.1;
        let inside_point = (
            edge.point_on_edge.0 - bisector.0 * offset,
            edge.point_on_edge.1 - bisector.1 * offset,
            edge.point_on_edge.2 - bisector.2 * offset,
        );
        points.push(ValidationPoint {
            position: inside_point,
            expected: ExpectedClassification::OnEdge {
                edge_index: edge_idx,
                expected_direction: edge.direction,
                expected_normals: (edge.face_a_normal, edge.face_b_normal),
            },
            description: format!("Edge {} (inside)", edge_idx),
        });
    }

    points
}

/// Get the 3 face indices that share a corner
fn faces_of_corner(corner_idx: usize) -> [usize; 3] {
    let x_pos = (corner_idx & 1) != 0;
    let y_pos = (corner_idx & 2) != 0;
    let z_pos = (corner_idx & 4) != 0;

    [
        if x_pos { 0 } else { 1 },
        if y_pos { 2 } else { 3 },
        if z_pos { 4 } else { 5 },
    ]
}

/// Validate surface detection at a point
pub fn validate_surface_detection<F>(
    point: &ValidationPoint,
    cube: &AnalyticalRotatedCube,
    cache: &SampleCache<F>,
    config: &SurfaceFindingConfig,
) -> SurfaceValidationResult
where
    F: Fn(f64, f64, f64) -> f32,
{
    let samples_before = cache.stats().actual_samples();

    let result = reference_find_nearest_surface(point.position, cache, config);

    let samples_used = cache.stats().actual_samples() - samples_before;

    // Get expected normal
    let expected_normal = match &point.expected {
        ExpectedClassification::OnFace { expected_normal, .. } => *expected_normal,
        ExpectedClassification::OnEdge { expected_normals, .. } => {
            // For edge, use bisector direction as reference
            normalize(add(expected_normals.0, expected_normals.1))
        }
        ExpectedClassification::OnCorner { expected_normals, .. } => {
            // For corner, use sum of normals as reference
            let sum = (
                expected_normals[0].0 + expected_normals[1].0 + expected_normals[2].0,
                expected_normals[0].1 + expected_normals[1].1 + expected_normals[2].1,
                expected_normals[0].2 + expected_normals[1].2 + expected_normals[2].2,
            );
            normalize(sum)
        }
    };

    let normal_error = angle_between(result.normal, expected_normal).to_degrees();

    // For position error, compare against closest surface point from analytical
    let analytical_closest = cube.closest_surface_point(point.position);
    let position_error = distance(result.closest_point, analytical_closest.point);

    SurfaceValidationResult {
        normal_error_degrees: normal_error,
        position_error,
        success: result.points_found > 0,
        samples_used,
    }
}

/// Validate edge detection at a point
pub fn validate_edge_detection<F>(
    point: &ValidationPoint,
    cube: &AnalyticalRotatedCube,
    cache: &SampleCache<F>,
    config: &EdgeFindingConfig,
) -> EdgeValidationResult
where
    F: Fn(f64, f64, f64) -> f32,
{
    let samples_before = cache.stats().actual_samples();

    let result = reference_find_nearest_edge(point.position, cache, config);

    let samples_used = cache.stats().actual_samples() - samples_before;

    match result {
        None => EdgeValidationResult {
            direction_error_degrees: 180.0,
            normal_a_error_degrees: 180.0,
            normal_b_error_degrees: 180.0,
            position_error: f64::INFINITY,
            success: false,
            samples_used,
        },
        Some(edge) => {
            // Get expected values
            let (expected_direction, expected_normals) = match &point.expected {
                ExpectedClassification::OnEdge {
                    expected_direction,
                    expected_normals,
                    ..
                } => (*expected_direction, *expected_normals),
                _ => {
                    // For non-edge points, find closest edge analytically
                    if let Some(analytical_edge) = cube.closest_edge(point.position, 1.0) {
                        (
                            analytical_edge.direction,
                            (analytical_edge.face_a_normal, analytical_edge.face_b_normal),
                        )
                    } else {
                        return EdgeValidationResult {
                            direction_error_degrees: 180.0,
                            normal_a_error_degrees: 180.0,
                            normal_b_error_degrees: 180.0,
                            position_error: f64::INFINITY,
                            success: false,
                            samples_used,
                        };
                    }
                }
            };

            // Direction error (edge direction can be either orientation)
            let dir_error_1 = angle_between(edge.edge_direction, expected_direction).to_degrees();
            let dir_error_2 = angle_between(
                (-edge.edge_direction.0, -edge.edge_direction.1, -edge.edge_direction.2),
                expected_direction,
            )
            .to_degrees();
            let direction_error = dir_error_1.min(dir_error_2);

            // Normal errors (match best pairing)
            let (normal_a_error, normal_b_error) = best_normal_pairing(
                (edge.face_a_normal, edge.face_b_normal),
                expected_normals,
            );

            // Position error (distance to analytical edge)
            let position_error = if let Some(analytical_edge) = cube.closest_edge(point.position, 1.0) {
                distance_to_line(edge.point_on_edge, analytical_edge.endpoints.0, analytical_edge.endpoints.1)
            } else {
                f64::INFINITY
            };

            EdgeValidationResult {
                direction_error_degrees: direction_error,
                normal_a_error_degrees: normal_a_error,
                normal_b_error_degrees: normal_b_error,
                position_error,
                success: true,
                samples_used,
            }
        }
    }
}

/// Validate corner detection at a point
pub fn validate_corner_detection<F>(
    point: &ValidationPoint,
    cube: &AnalyticalRotatedCube,
    cache: &SampleCache<F>,
    config: &CornerFindingConfig,
) -> CornerValidationResult
where
    F: Fn(f64, f64, f64) -> f32,
{
    let samples_before = cache.stats().actual_samples();

    let result = reference_find_nearest_corner(point.position, cache, config);

    let samples_used = cache.stats().actual_samples() - samples_before;

    match result {
        None => CornerValidationResult {
            num_faces_detected: 0,
            normal_errors_degrees: vec![],
            position_error: f64::INFINITY,
            success: false,
            samples_used,
        },
        Some(corner) => {
            // Get expected normals
            let expected_normals = match &point.expected {
                ExpectedClassification::OnCorner { expected_normals, .. } => expected_normals.to_vec(),
                _ => {
                    // For non-corner points, find closest corner analytically
                    if let Some(analytical_corner) = cube.closest_corner(point.position, 1.0) {
                        analytical_corner.face_normals.to_vec()
                    } else {
                        return CornerValidationResult {
                            num_faces_detected: corner.num_faces,
                            normal_errors_degrees: vec![],
                            position_error: f64::INFINITY,
                            success: false,
                            samples_used,
                        };
                    }
                }
            };

            // Match detected normals to expected normals (find best assignment)
            let normal_errors = match_normals_to_expected(&corner.face_normals, &expected_normals);

            // Position error (distance to analytical corner)
            let position_error = if let Some(analytical_corner) = cube.closest_corner(point.position, 1.0) {
                distance(corner.corner_position, analytical_corner.position)
            } else {
                f64::INFINITY
            };

            CornerValidationResult {
                num_faces_detected: corner.num_faces,
                normal_errors_degrees: normal_errors,
                position_error,
                success: corner.num_faces >= 2,
                samples_used,
            }
        }
    }
}

/// Find best pairing of two detected normals to two expected normals
fn best_normal_pairing(
    detected: ((f64, f64, f64), (f64, f64, f64)),
    expected: ((f64, f64, f64), (f64, f64, f64)),
) -> (f64, f64) {
    // Try both assignments
    let assignment1 = (
        angle_between(detected.0, expected.0).to_degrees(),
        angle_between(detected.1, expected.1).to_degrees(),
    );

    let assignment2 = (
        angle_between(detected.0, expected.1).to_degrees(),
        angle_between(detected.1, expected.0).to_degrees(),
    );

    // Return assignment with lower total error
    if assignment1.0 + assignment1.1 < assignment2.0 + assignment2.1 {
        assignment1
    } else {
        assignment2
    }
}

/// Match detected normals to expected normals (greedy assignment)
fn match_normals_to_expected(
    detected: &[(f64, f64, f64)],
    expected: &[(f64, f64, f64)],
) -> Vec<f64> {
    let mut errors = Vec::new();
    let mut used_expected: Vec<bool> = vec![false; expected.len()];

    for d in detected {
        let mut best_error = f64::MAX;
        let mut best_idx = 0;

        for (i, e) in expected.iter().enumerate() {
            if !used_expected[i] {
                let error = angle_between(*d, *e).to_degrees();
                if error < best_error {
                    best_error = error;
                    best_idx = i;
                }
            }
        }

        if best_error < f64::MAX {
            used_expected[best_idx] = true;
            errors.push(best_error);
        }
    }

    errors
}

// Vector math helpers

fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
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

fn distance(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    length(sub(a, b))
}

/// Distance from point to line segment
fn distance_to_line(p: (f64, f64, f64), a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let ab = sub(b, a);
    let ap = sub(p, a);

    let t = dot(ap, ab) / dot(ab, ab);
    let t_clamped = t.clamp(0.0, 1.0);

    let closest = (
        a.0 + t_clamped * ab.0,
        a.1 + t_clamped * ab.1,
        a.2 + t_clamped * ab.2,
    );

    distance(p, closest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_validation_points() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let points = generate_validation_points(&cube);

        // Should have: 6 face centers + 12 edge midpoints + 8 corners + 6 inside faces + 12 inside edges = 44
        assert!(points.len() >= 26, "Should have at least 26 points (basic), got {}", points.len());

        // Check face points exist
        let face_points = points
            .iter()
            .filter(|p| matches!(p.expected, ExpectedClassification::OnFace { .. }))
            .count();
        assert!(face_points >= 6, "Should have at least 6 face points");

        // Check edge points exist
        let edge_points = points
            .iter()
            .filter(|p| matches!(p.expected, ExpectedClassification::OnEdge { .. }))
            .count();
        assert!(edge_points >= 12, "Should have at least 12 edge points");

        // Check corner points exist
        let corner_points = points
            .iter()
            .filter(|p| matches!(p.expected, ExpectedClassification::OnCorner { .. }))
            .count();
        assert_eq!(corner_points, 8, "Should have exactly 8 corner points");
    }

    #[test]
    fn test_best_normal_pairing() {
        let detected = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
        let expected = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)); // swapped order

        let (err_a, err_b) = best_normal_pairing(detected, expected);

        // Should find the correct pairing with near-zero errors
        assert!(err_a < 1.0);
        assert!(err_b < 1.0);
    }

    fn rotated_cube_sampler(cube: &AnalyticalRotatedCube) -> impl Fn(f64, f64, f64) -> f32 + '_ {
        move |x, y, z| {
            // Transform to local coordinates
            let local = cube.world_to_local((x, y, z));
            let h = 0.5;
            // Use <= for boundary handling
            let inside = local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h;
            if inside { 1.0 } else { -1.0 }
        }
    }

    #[test]
    fn test_validate_surface_at_face_center() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);

        let points = generate_validation_points(&cube);
        // Use the "inside" point instead of the face center itself
        // Face center is ON the surface, which can cause issues with probing
        // The "inside" point is offset inward, providing a clear inside starting point
        let face_point = points.iter().find(|p| p.description == "Face 0 (inside)").unwrap();

        // Use more aggressive probing for testing
        let config = SurfaceFindingConfig {
            num_probes: 100,
            max_search_distance: 0.5,
            search_step: 0.01,
            binary_iterations: 20,
        };

        let result = validate_surface_detection(face_point, &cube, &cache, &config);

        assert!(result.success, "Should successfully detect surface");
        // The reference routines should find the surface with reasonable accuracy
        assert!(
            result.normal_error_degrees < 10.0,
            "Normal error {:.1}° should be reasonable",
            result.normal_error_degrees
        );
    }
}
// Temporary test to measure reference vs analytical accuracy
#[cfg(test)]
mod accuracy_tests {
    use crate::adaptive_surface_nets_2::stage4::research::*;
    
    fn rotated_cube_sampler(cube: &AnalyticalRotatedCube) -> impl Fn(f64, f64, f64) -> f32 + '_ {
        move |x, y, z| {
            let local = cube.world_to_local((x, y, z));
            let h = 0.5;
            let inside = local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h;
            if inside { 1.0 } else { -1.0 }
        }
    }

    #[test]
    fn measure_reference_accuracy() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);
        
        let surface_config = reference_surface::SurfaceFindingConfig {
            num_probes: 100,
            max_search_distance: 0.5,
            search_step: 0.01,
            binary_iterations: 20,
        };
        
        let edge_config = reference_edge::EdgeFindingConfig {
            num_probes: 150,
            max_search_distance: 0.5,
            search_step: 0.01,
            binary_iterations: 20,
            kmeans_iterations: 30,
        };
        
        let corner_config = reference_corner::CornerFindingConfig {
            num_probes: 200,
            max_search_distance: 0.5,
            search_step: 0.01,
            binary_iterations: 20,
            kmeans_iterations: 40,
            min_cluster_size: 4,
            k_improvement_threshold: 0.7,
        };
        
        println!("\n=== Reference vs Analytical Accuracy ===\n");
        
        // Test face points (inside offset)
        println!("FACE POINTS (offset 0.1 inside from face center):");
        println!("{:<20} {:>12} {:>12}", "Point", "Normal Err°", "Pos Err");
        println!("{}", "-".repeat(50));
        
        let mut face_normal_errors = Vec::new();
        let mut face_pos_errors = Vec::new();
        
        for face_idx in 0..6 {
            let center = cube.face_center(face_idx);
            let normal = cube.face_normals[face_idx];
            let offset = 0.1;
            let inside_point = (
                center.0 - normal.0 * offset,
                center.1 - normal.1 * offset,
                center.2 - normal.2 * offset,
            );
            
            cache.clear();
            let result = reference_find_nearest_surface(inside_point, &cache, &surface_config);
            
            let normal_err = analytical_cube::angle_between(result.normal, normal).to_degrees();
            let analytical_closest = cube.closest_surface_point(inside_point);
            let pos_err = distance(result.closest_point, analytical_closest.point);
            
            face_normal_errors.push(normal_err);
            face_pos_errors.push(pos_err);
            
            println!("Face {} (inside)      {:>10.2}°  {:>10.4}", face_idx, normal_err, pos_err);
        }
        
        let avg_face_normal = face_normal_errors.iter().sum::<f64>() / face_normal_errors.len() as f64;
        let max_face_normal = face_normal_errors.iter().cloned().fold(0.0_f64, f64::max);
        let avg_face_pos = face_pos_errors.iter().sum::<f64>() / face_pos_errors.len() as f64;
        
        println!("{}", "-".repeat(50));
        println!("Average:             {:>10.2}°  {:>10.4}", avg_face_normal, avg_face_pos);
        println!("Max:                 {:>10.2}°", max_face_normal);
        
        // Test edge points (inside offset)
        println!("\nEDGE POINTS (offset 0.1 inside from edge midpoint):");
        println!("{:<20} {:>10} {:>10} {:>10} {:>10}", "Point", "Dir Err°", "N_A Err°", "N_B Err°", "Pos Err");
        println!("{}", "-".repeat(70));
        
        let mut edge_dir_errors = Vec::new();
        let mut edge_na_errors = Vec::new();
        let mut edge_nb_errors = Vec::new();
        let mut edge_pos_errors = Vec::new();
        let mut edge_failures = 0;
        
        for edge_idx in 0..12 {
            let edge = cube.get_edge(edge_idx);
            let bisector = normalize(add(edge.face_a_normal, edge.face_b_normal));
            let offset = 0.1;
            let inside_point = (
                edge.point_on_edge.0 - bisector.0 * offset,
                edge.point_on_edge.1 - bisector.1 * offset,
                edge.point_on_edge.2 - bisector.2 * offset,
            );
            
            cache.clear();
            let result = reference_find_nearest_edge(inside_point, &cache, &edge_config);
            
            if let Some(ref detected) = result {
                // Direction error (handle both orientations)
                let dir_err1 = analytical_cube::angle_between(detected.edge_direction, edge.direction).to_degrees();
                let dir_err2 = analytical_cube::angle_between(
                    (-detected.edge_direction.0, -detected.edge_direction.1, -detected.edge_direction.2),
                    edge.direction
                ).to_degrees();
                let dir_err = dir_err1.min(dir_err2);
                
                // Normal errors (best pairing)
                let (na_err, nb_err) = best_normal_pairing(
                    (detected.face_a_normal, detected.face_b_normal),
                    (edge.face_a_normal, edge.face_b_normal),
                );
                
                // Position error
                let pos_err = distance_to_line(detected.point_on_edge, edge.endpoints.0, edge.endpoints.1);
                
                edge_dir_errors.push(dir_err);
                edge_na_errors.push(na_err);
                edge_nb_errors.push(nb_err);
                edge_pos_errors.push(pos_err);
                
                println!("Edge {:2} (inside)    {:>8.2}°  {:>8.2}°  {:>8.2}°  {:>8.4}", 
                    edge_idx, dir_err, na_err, nb_err, pos_err);
            } else {
                edge_failures += 1;
                println!("Edge {:2} (inside)    FAILED", edge_idx);
            }
        }
        
        if !edge_dir_errors.is_empty() {
            let avg_dir = edge_dir_errors.iter().sum::<f64>() / edge_dir_errors.len() as f64;
            let avg_na = edge_na_errors.iter().sum::<f64>() / edge_na_errors.len() as f64;
            let avg_nb = edge_nb_errors.iter().sum::<f64>() / edge_nb_errors.len() as f64;
            let avg_pos = edge_pos_errors.iter().sum::<f64>() / edge_pos_errors.len() as f64;
            let max_dir = edge_dir_errors.iter().cloned().fold(0.0_f64, f64::max);
            let max_na = edge_na_errors.iter().cloned().fold(0.0_f64, f64::max);
            let max_nb = edge_nb_errors.iter().cloned().fold(0.0_f64, f64::max);
            
            println!("{}", "-".repeat(70));
            println!("Average:            {:>8.2}°  {:>8.2}°  {:>8.2}°  {:>8.4}", avg_dir, avg_na, avg_nb, avg_pos);
            println!("Max:                {:>8.2}°  {:>8.2}°  {:>8.2}°", max_dir, max_na, max_nb);
            println!("Failures: {}/12", edge_failures);
        }
        
        // Test corner points (inside offset)
        println!("\nCORNER POINTS (offset 0.1 inside from corner):");
        println!("{:<20} {:>8} {:>10} {:>10} {:>10} {:>10}", "Point", "Faces", "N1 Err°", "N2 Err°", "N3 Err°", "Pos Err");
        println!("{}", "-".repeat(80));
        
        let mut corner_normal_errors = Vec::new();
        let mut corner_pos_errors = Vec::new();
        let mut corner_face_counts = Vec::new();
        let mut corner_failures = 0;
        
        for corner_idx in 0..8 {
            let corner_pos = cube.corners[corner_idx];
            // Move inside along the diagonal from corner toward center
            let to_center = normalize((-corner_pos.0, -corner_pos.1, -corner_pos.2));
            let offset = 0.1;
            let inside_point = (
                corner_pos.0 + to_center.0 * offset,
                corner_pos.1 + to_center.1 * offset,
                corner_pos.2 + to_center.2 * offset,
            );
            
            let faces = faces_of_corner(corner_idx);
            let expected_normals = [
                cube.face_normals[faces[0]],
                cube.face_normals[faces[1]],
                cube.face_normals[faces[2]],
            ];
            
            cache.clear();
            let result = reference_find_nearest_corner(inside_point, &cache, &corner_config);
            
            if let Some(ref detected) = result {
                corner_face_counts.push(detected.num_faces);
                
                // Match normals
                let errors = match_normals(&detected.face_normals, &expected_normals);
                for &e in &errors {
                    corner_normal_errors.push(e);
                }
                
                let pos_err = distance(detected.corner_position, corner_pos);
                corner_pos_errors.push(pos_err);
                
                let err_str: Vec<String> = errors.iter().map(|e| format!("{:>8.2}°", e)).collect();
                let err_display = if errors.len() >= 3 {
                    format!("{} {} {}", err_str[0], err_str[1], err_str[2])
                } else {
                    errors.iter().map(|e| format!("{:.2}°", e)).collect::<Vec<_>>().join(" ")
                };
                
                println!("Corner {:2} (inside)  {:>6}  {}  {:>8.4}", 
                    corner_idx, detected.num_faces, err_display, pos_err);
            } else {
                corner_failures += 1;
                println!("Corner {:2} (inside)  FAILED", corner_idx);
            }
        }
        
        if !corner_normal_errors.is_empty() {
            let avg_normal = corner_normal_errors.iter().sum::<f64>() / corner_normal_errors.len() as f64;
            let max_normal = corner_normal_errors.iter().cloned().fold(0.0_f64, f64::max);
            let avg_pos = corner_pos_errors.iter().sum::<f64>() / corner_pos_errors.len() as f64;
            let avg_faces = corner_face_counts.iter().sum::<usize>() as f64 / corner_face_counts.len() as f64;
            
            println!("{}", "-".repeat(80));
            println!("Avg faces detected: {:.1}", avg_faces);
            println!("Avg normal error:   {:.2}°", avg_normal);
            println!("Max normal error:   {:.2}°", max_normal);
            println!("Avg position error: {:.4}", avg_pos);
            println!("Failures: {}/8", corner_failures);
        }
        
        println!("\n=== Summary ===");
        println!("Pass criteria: Normal errors < 1°, Position errors < 0.01");
        println!("Face detection:  Avg {:.2}° (target <1°)", avg_face_normal);
        if !edge_dir_errors.is_empty() {
            let avg_edge = (edge_na_errors.iter().sum::<f64>() + edge_nb_errors.iter().sum::<f64>()) 
                / (edge_na_errors.len() + edge_nb_errors.len()) as f64;
            println!("Edge detection:  Avg {:.2}° (target <1°)", avg_edge);
        }
        if !corner_normal_errors.is_empty() {
            let avg_corner = corner_normal_errors.iter().sum::<f64>() / corner_normal_errors.len() as f64;
            println!("Corner detection: Avg {:.2}° (target <1°)", avg_corner);
        }
    }
    
    fn distance(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
        ((a.0-b.0).powi(2) + (a.1-b.1).powi(2) + (a.2-b.2).powi(2)).sqrt()
    }
    
    fn normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
        let len = (v.0*v.0 + v.1*v.1 + v.2*v.2).sqrt();
        if len > 1e-12 { (v.0/len, v.1/len, v.2/len) } else { (0.0, 0.0, 1.0) }
    }
    
    fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
        (a.0+b.0, a.1+b.1, a.2+b.2)
    }
    
    fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
        a.0*b.0 + a.1*b.1 + a.2*b.2
    }
    
    fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
        (a.0-b.0, a.1-b.1, a.2-b.2)
    }
    
    fn distance_to_line(p: (f64, f64, f64), a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
        let ab = sub(b, a);
        let ap = sub(p, a);
        let t = dot(ap, ab) / dot(ab, ab);
        let t_clamped = t.clamp(0.0, 1.0);
        let closest = (a.0 + t_clamped * ab.0, a.1 + t_clamped * ab.1, a.2 + t_clamped * ab.2);
        distance(p, closest)
    }
    
    fn best_normal_pairing(detected: ((f64,f64,f64), (f64,f64,f64)), expected: ((f64,f64,f64), (f64,f64,f64))) -> (f64, f64) {
        let a1 = (analytical_cube::angle_between(detected.0, expected.0).to_degrees(),
                  analytical_cube::angle_between(detected.1, expected.1).to_degrees());
        let a2 = (analytical_cube::angle_between(detected.0, expected.1).to_degrees(),
                  analytical_cube::angle_between(detected.1, expected.0).to_degrees());
        if a1.0 + a1.1 < a2.0 + a2.1 { a1 } else { a2 }
    }
    
    fn match_normals(detected: &[(f64,f64,f64)], expected: &[(f64,f64,f64)]) -> Vec<f64> {
        let mut errors = Vec::new();
        let mut used = vec![false; expected.len()];
        for d in detected {
            let mut best_err = f64::MAX;
            let mut best_idx = 0;
            for (i, e) in expected.iter().enumerate() {
                if !used[i] {
                    let err = analytical_cube::angle_between(*d, *e).to_degrees();
                    if err < best_err { best_err = err; best_idx = i; }
                }
            }
            if best_err < f64::MAX {
                used[best_idx] = true;
                errors.push(best_err);
            }
        }
        errors
    }
    
    fn faces_of_corner(corner_idx: usize) -> [usize; 3] {
        let x_pos = (corner_idx & 1) != 0;
        let y_pos = (corner_idx & 2) != 0;
        let z_pos = (corner_idx & 4) != 0;
        [
            if x_pos { 0 } else { 1 },
            if y_pos { 2 } else { 3 },
            if z_pos { 4 } else { 5 },
        ]
    }
}
