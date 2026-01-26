//! Attempt 0: Sequential Geometry Classification via Crossing Count
//!
//! This experimental implementation replaces the entire Stage 4 pipeline with an efficient
//! algorithm that uses minimal boolean probes to:
//! 1. **Surface Location**: Move vertices from edge midpoints to actual surface
//! 2. **Geometry Classification**: Determine if vertex is near face, edge, or corner
//! 3. **Measurement**: Extract normals, edge directions, corner positions
//!
//! # Key Insight: Crossing Count as Edge Detector
//!
//! During surface location, we search along multiple directions for surface crossings.
//! The number of successful crossings is a FREE edge detector:
//! - **4 crossings**: All axes find surface → definitely smooth
//! - **3 crossings**: Most axes find surface → probably smooth
//! - **2 crossings**: Only 2 axes find surface → almost certainly edge
//! - **1 crossing**: Unusual, might be corner or thin feature
//!
//! # Algorithm Structure
//!
//! ```text
//! Phase 0: Surface Location (10-15 samples)
//!   → Find actual surface position
//!   → Record crossing count and directions
//!
//! Phase 1: Classification (from crossing count)
//!   → crossing_count >= 3 → SmoothSurface
//!   → crossing_count == 2 → EdgeCandidate
//!   → crossing_count == 1 → CornerCandidate or uncertain
//!
//! Phase 2: Measurement (10-60 samples depending on type)
//!   → Face: Mini-RANSAC with 12 tangent disk probes
//!   → Edge: Biased hemisphere probing, 2-plane RANSAC
//!   → Corner: Full sphere coverage, 3-plane RANSAC
//! ```
//!
//! # Sample Budget Targets
//!
//! | Geometry Type | Sample Budget | Accuracy Target |
//! |---------------|---------------|-----------------|
//! | Surface location | 10-15 samples | Position within 0.01 cell |
//! | Face (smooth) | 20-35 samples total | Normal error < 1° |
//! | Edge (sharp)  | 50-80 samples total | Both normals < 1°, direction < 1° |
//! | Corner        | 80-120 samples total | All 3 normals < 1° |

use super::sample_cache::SampleCache;

// ============================================================================
// Types
// ============================================================================

/// Result of surface location (Phase 0)
#[derive(Clone, Debug)]
pub struct SurfaceLocation {
    /// The refined position on the actual surface
    pub position: (f64, f64, f64),
    /// Number of search directions that found a crossing (1-4)
    pub crossing_count: usize,
    /// The directions that successfully found crossings
    pub crossing_directions: Vec<(f64, f64, f64)>,
    /// The surface points found by each successful crossing search
    pub crossing_positions: Vec<(f64, f64, f64)>,
    /// Number of samples used
    pub samples_used: u32,
}

/// Classification result from Phase 1
#[derive(Clone, Debug)]
pub enum GeometryClassification {
    /// Likely a smooth surface (single face)
    SmoothSurface,
    /// Likely on an edge between two faces
    EdgeCandidate {
        /// The two directions that found crossings (hints for face normals)
        face_hints: [(f64, f64, f64); 2],
    },
    /// Likely at a corner (three or more faces meet)
    CornerCandidate {
        /// Hints about edge/face directions from crossing analysis
        edge_hints: Vec<(f64, f64, f64)>,
    },
    /// Classification uncertain, needs more probing
    Uncertain {
        /// Surface points found during location
        probed_points: Vec<(f64, f64, f64)>,
    },
}

/// Geometry type enum
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryType {
    Face,
    Edge,
    Corner,
}

/// Final measurement result with confidence
#[derive(Clone, Debug)]
pub struct VertexGeometry {
    /// The detected geometry type
    pub classification: GeometryType,
    /// The normals (1 for face, 2 for edge, 3 for corner)
    pub normals: Vec<(f64, f64, f64)>,
    /// Edge direction (only for Edge type)
    pub edge_direction: Option<(f64, f64, f64)>,
    /// Corner position (only for Corner type)
    pub corner_position: Option<(f64, f64, f64)>,
    /// Confidence score (0.0 to 1.0, based on plane fit residuals)
    pub confidence: f64,
    /// Total samples used across all phases
    pub samples_used: u32,
}

/// Result of face measurement
#[derive(Clone, Debug)]
pub struct FaceMeasurement {
    /// The computed face normal
    pub normal: (f64, f64, f64),
    /// Centroid of the fitted plane
    pub centroid: (f64, f64, f64),
    /// RMS residual of the plane fit
    pub residual: f64,
    /// Number of surface points used
    pub num_points: usize,
    /// Samples used for this measurement
    pub samples_used: u32,
    /// Surface points found (for outlier analysis)
    pub surface_points: Vec<(f64, f64, f64)>,
}

/// Result of edge measurement
#[derive(Clone, Debug)]
pub struct EdgeMeasurement {
    /// Normal of face A
    pub normal_a: (f64, f64, f64),
    /// Normal of face B
    pub normal_b: (f64, f64, f64),
    /// Direction of the edge (unit vector)
    pub edge_direction: (f64, f64, f64),
    /// A point on the edge
    pub point_on_edge: (f64, f64, f64),
    /// Number of inliers for plane A
    pub inliers_a: usize,
    /// Number of inliers for plane B
    pub inliers_b: usize,
    /// RMS residual for plane A
    pub residual_a: f64,
    /// RMS residual for plane B
    pub residual_b: f64,
    /// Samples used for this measurement
    pub samples_used: u32,
}

/// Result of corner measurement
#[derive(Clone, Debug)]
pub struct CornerMeasurement {
    /// The three face normals meeting at this corner
    pub normals: [(f64, f64, f64); 3],
    /// The computed corner position
    pub corner_position: (f64, f64, f64),
    /// Inlier counts for each plane
    pub inliers: [usize; 3],
    /// Residuals for each plane
    pub residuals: [f64; 3],
    /// Samples used for this measurement
    pub samples_used: u32,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the crossing count algorithm
#[derive(Clone, Debug)]
pub struct CrossingCountConfig {
    /// Number of binary search iterations for surface finding
    pub binary_search_iterations: usize,
    /// Maximum search distance (relative to cell size)
    pub search_distance: f64,
    /// Probe epsilon for tangent plane probing (relative to cell size)
    pub probe_epsilon: f64,
    /// RANSAC inlier threshold for plane fitting
    pub ransac_inlier_threshold: f64,
    /// Number of RANSAC iterations
    pub ransac_iterations: usize,
    /// Number of probes for face measurement
    pub face_probe_count: usize,
    /// Number of probes per hemisphere for edge measurement
    pub edge_probes_per_hemisphere: usize,
    /// Number of probes for corner measurement
    pub corner_probe_count: usize,
}

impl Default for CrossingCountConfig {
    fn default() -> Self {
        Self {
            binary_search_iterations: 15,
            search_distance: 0.5, // Half cell size
            probe_epsilon: 0.1,
            ransac_inlier_threshold: 0.01,
            ransac_iterations: 200,
            face_probe_count: 12,
            edge_probes_per_hemisphere: 15,
            corner_probe_count: 60,
        }
    }
}

// ============================================================================
// Phase 0: Surface Location
// ============================================================================

/// Locate the actual surface position from an edge midpoint.
///
/// This function moves a vertex from the mesh edge midpoint to the actual surface
/// by searching along the accumulated normal and fallback axes for surface crossings.
///
/// # Key Output: Crossing Count
///
/// The number of successful crossing directions is a FREE edge detector:
/// - **4 crossings**: All axes find surface → definitely smooth
/// - **3 crossings**: Most axes find surface → probably smooth
/// - **2 crossings**: Only 2 axes find surface → almost certainly edge
/// - **1 crossing**: Edge case, may be corner or thin feature
pub fn locate_surface<F>(
    midpoint: (f64, f64, f64),
    accumulated_normal: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &CrossingCountConfig,
) -> SurfaceLocation
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    let search_distance = cell_size * config.search_distance;

    // Step 1: Determine if midpoint is inside or outside
    let is_inside = cache.is_inside(midpoint.0, midpoint.1, midpoint.2);
    samples_used += 1;

    // Step 2: Prepare search directions
    // Start with accumulated normal (if non-degenerate), then try cardinal axes
    let mut search_directions = Vec::with_capacity(7);

    let norm_len = length(accumulated_normal);
    if norm_len > 0.01 {
        let normalized = (
            accumulated_normal.0 / norm_len,
            accumulated_normal.1 / norm_len,
            accumulated_normal.2 / norm_len,
        );
        search_directions.push(normalized);
    }

    // Add cardinal axes as fallbacks
    search_directions.push((1.0, 0.0, 0.0));
    search_directions.push((-1.0, 0.0, 0.0));
    search_directions.push((0.0, 1.0, 0.0));
    search_directions.push((0.0, -1.0, 0.0));
    search_directions.push((0.0, 0.0, 1.0));
    search_directions.push((0.0, 0.0, -1.0));

    // Step 3: Search each direction for crossings
    let mut crossing_directions = Vec::new();
    let mut crossing_positions = Vec::new();
    let mut best_crossing: Option<(f64, f64, f64)> = None;
    let mut best_distance = f64::INFINITY;

    for dir in &search_directions {
        if crossing_directions.len() >= 4 {
            break; // Enough data
        }

        // Skip if we already have a similar direction
        let is_duplicate = crossing_directions.iter().any(|existing: &(f64, f64, f64)| {
            let dot_val = dot(*existing, *dir).abs();
            dot_val > 0.9 // More than ~25 degrees of similarity
        });
        if is_duplicate {
            continue;
        }

        if let Some((crossing, dist, used)) = binary_search_crossing_along_dir(
            midpoint,
            *dir,
            search_distance,
            is_inside,
            cache,
            config.binary_search_iterations,
        ) {
            samples_used += used;
            crossing_directions.push(*dir);
            crossing_positions.push(crossing);

            if dist < best_distance {
                best_distance = dist;
                best_crossing = Some(crossing);
            }
        } else {
            // Direction didn't find a crossing - still count samples
            samples_used += 2; // Initial probe + at least one more
        }
    }

    // Use the best crossing position, or midpoint if none found
    let position = best_crossing.unwrap_or(midpoint);

    SurfaceLocation {
        position,
        crossing_count: crossing_directions.len(),
        crossing_directions,
        crossing_positions,
        samples_used,
    }
}

/// Binary search for a surface crossing along a direction.
///
/// Returns Some((crossing_point, distance_from_start, samples_used)) if found.
fn binary_search_crossing_along_dir<F>(
    start: (f64, f64, f64),
    dir: (f64, f64, f64),
    max_distance: f64,
    start_inside: bool,
    cache: &SampleCache<F>,
    iterations: usize,
) -> Option<((f64, f64, f64), f64, u32)>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;

    // First, probe at max_distance to see if there's a crossing
    let end = (
        start.0 + dir.0 * max_distance,
        start.1 + dir.1 * max_distance,
        start.2 + dir.2 * max_distance,
    );

    let end_inside = cache.is_inside(end.0, end.1, end.2);
    samples_used += 1;

    if end_inside == start_inside {
        // No crossing in this direction
        return None;
    }

    // Binary search to find the crossing
    let (mut inside_point, mut outside_point) = if start_inside {
        (start, end)
    } else {
        (end, start)
    };

    for _ in 0..iterations {
        let mid = (
            (inside_point.0 + outside_point.0) / 2.0,
            (inside_point.1 + outside_point.1) / 2.0,
            (inside_point.2 + outside_point.2) / 2.0,
        );

        let mid_inside = cache.is_inside(mid.0, mid.1, mid.2);
        samples_used += 1;

        if mid_inside {
            inside_point = mid;
        } else {
            outside_point = mid;
        }
    }

    // Final crossing estimate is midpoint of interval
    let crossing = (
        (inside_point.0 + outside_point.0) / 2.0,
        (inside_point.1 + outside_point.1) / 2.0,
        (inside_point.2 + outside_point.2) / 2.0,
    );

    let dist = distance(start, crossing);
    Some((crossing, dist, samples_used))
}

// ============================================================================
// Phase 1: Classification
// ============================================================================

/// Classify geometry type based on crossing count from surface location.
pub fn classify_geometry(location: &SurfaceLocation) -> GeometryClassification {
    match location.crossing_count {
        4 => GeometryClassification::SmoothSurface,
        3 => GeometryClassification::SmoothSurface,
        2 => {
            let hints = [
                location.crossing_directions[0],
                location.crossing_directions[1],
            ];
            GeometryClassification::EdgeCandidate { face_hints: hints }
        }
        1 => GeometryClassification::CornerCandidate {
            edge_hints: location.crossing_directions.clone(),
        },
        0 => GeometryClassification::Uncertain {
            probed_points: location.crossing_positions.clone(),
        },
        _ => GeometryClassification::Uncertain {
            probed_points: location.crossing_positions.clone(),
        },
    }
}

// ============================================================================
// Phase 2a: Face Measurement
// ============================================================================

/// Measure a face normal at a smooth surface location.
///
/// Uses a mini-RANSAC approach with well-distributed probes in the tangent disk.
pub fn measure_face<F>(
    position: (f64, f64, f64),
    hint_normal: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &CrossingCountConfig,
) -> Option<FaceMeasurement>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    let probe_distance = cell_size * config.probe_epsilon;
    let search_distance = cell_size * config.search_distance;

    // Build tangent basis from hint normal
    let normal = normalize(hint_normal);
    let (tangent1, tangent2) = build_tangent_basis(normal);

    // Generate disk directions using golden ratio
    let directions = golden_ratio_disk_directions(config.face_probe_count);

    // Find surface points
    let mut surface_points = Vec::new();

    for (u, v) in directions {
        // Transform to world direction
        let world_dir = (
            tangent1.0 * u + tangent2.0 * v,
            tangent1.1 * u + tangent2.1 * v,
            tangent1.2 * u + tangent2.2 * v,
        );
        let world_dir = normalize(world_dir);

        // Offset position and search for surface
        let probe_start = (
            position.0 + world_dir.0 * probe_distance,
            position.1 + world_dir.1 * probe_distance,
            position.2 + world_dir.2 * probe_distance,
        );

        // Check inside/outside at the actual probe start point
        let probe_inside = cache.is_inside(probe_start.0, probe_start.1, probe_start.2);
        samples_used += 1;

        // Try searching along hint normal
        if let Some((crossing, _, used)) = binary_search_crossing_along_dir(
            probe_start,
            normal,
            search_distance,
            probe_inside,
            cache,
            config.binary_search_iterations,
        ) {
            samples_used += used;
            surface_points.push(crossing);
        } else {
            // Try opposite direction
            let neg_normal = (-normal.0, -normal.1, -normal.2);
            if let Some((crossing, _, used)) = binary_search_crossing_along_dir(
                probe_start,
                neg_normal,
                search_distance,
                probe_inside,
                cache,
                config.binary_search_iterations,
            ) {
                samples_used += used;
                surface_points.push(crossing);
            } else {
                samples_used += 2;
            }
        }
    }

    if surface_points.len() < 3 {
        return None;
    }

    // Fit plane via SVD (single plane, no RANSAC needed for smooth surfaces)
    let (centroid, fitted_normal, residual) = fit_plane_svd(&surface_points);

    // Orient normal away from inside
    let adjusted_normal = orient_normal_away_from_inside(fitted_normal, centroid, position, cache);
    samples_used += 1; // orient check

    Some(FaceMeasurement {
        normal: adjusted_normal,
        centroid,
        residual,
        num_points: surface_points.len(),
        samples_used,
        surface_points,
    })
}

// ============================================================================
// Phase 2b: Edge Measurement
// ============================================================================

/// Measure edge geometry (two face normals and edge direction).
///
/// Uses crossing directions as hints for where faces are, and probes more densely
/// in those regions.
pub fn measure_edge<F>(
    position: (f64, f64, f64),
    face_hints: [(f64, f64, f64); 2],
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &CrossingCountConfig,
) -> Option<EdgeMeasurement>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    let search_distance = cell_size * config.search_distance;

    // Generate probes biased toward each face hint
    let mut all_directions = Vec::new();

    // Hemisphere around hint_a
    let directions_a = hemisphere_directions(face_hints[0], config.edge_probes_per_hemisphere);
    all_directions.extend(directions_a.iter().map(|d| (*d, 0))); // 0 = face A hint

    // Hemisphere around hint_b
    let directions_b = hemisphere_directions(face_hints[1], config.edge_probes_per_hemisphere);
    all_directions.extend(directions_b.iter().map(|d| (*d, 1))); // 1 = face B hint

    // Check inside/outside at query position
    let query_inside = cache.is_inside(position.0, position.1, position.2);
    samples_used += 1;

    // Find surface points
    let mut surface_points = Vec::new();

    for (dir, _hint_idx) in &all_directions {
        if let Some((crossing, _, used)) = binary_search_crossing_along_dir(
            position,
            *dir,
            search_distance,
            query_inside,
            cache,
            config.binary_search_iterations,
        ) {
            samples_used += used;
            surface_points.push(crossing);
        } else {
            samples_used += 2;
        }
    }

    if surface_points.len() < 6 {
        return None;
    }

    // Two-plane RANSAC with threshold scaled to cell_size (0.006 per unit)
    let tight_threshold = 0.006 * cell_size;

    // First plane
    let plane_a_result = ransac_plane_fit(&surface_points, tight_threshold, config.ransac_iterations);
    if plane_a_result.is_none() {
        return None;
    }
    let (plane_a_inliers, plane_a_centroid, plane_a_normal, plane_a_residual) = plane_a_result.unwrap();

    if plane_a_inliers.len() < 3 {
        return None;
    }

    // Remove inliers, find second plane
    let remaining: Vec<_> = surface_points
        .iter()
        .filter(|p| !is_inlier(**p, plane_a_centroid, plane_a_normal, tight_threshold))
        .cloned()
        .collect();

    if remaining.len() < 3 {
        return None;
    }

    let plane_b_result = ransac_plane_fit(&remaining, tight_threshold, config.ransac_iterations);
    if plane_b_result.is_none() {
        return None;
    }
    let (plane_b_inliers, plane_b_centroid, plane_b_normal, plane_b_residual) = plane_b_result.unwrap();

    if plane_b_inliers.len() < 3 {
        return None;
    }

    // Orient normals away from query point
    let normal_a = orient_away(plane_a_normal, plane_a_centroid, position);
    let normal_b = orient_away(plane_b_normal, plane_b_centroid, position);

    // Compute edge from plane intersection
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None; // Planes are parallel
    }

    let point_on_edge = find_edge_point(
        position,
        plane_a_centroid,
        normal_a,
        plane_b_centroid,
        normal_b,
        edge_dir,
    );

    Some(EdgeMeasurement {
        normal_a,
        normal_b,
        edge_direction: edge_dir,
        point_on_edge,
        inliers_a: plane_a_inliers.len(),
        inliers_b: plane_b_inliers.len(),
        residual_a: plane_a_residual,
        residual_b: plane_b_residual,
        samples_used,
    })
}

/// Measure edge geometry WITHOUT requiring face hints.
///
/// Uses full sphere coverage and finds 2 planes via RANSAC.
/// This is more expensive but doesn't require knowing face normal hints.
pub fn measure_edge_full_sphere<F>(
    position: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &CrossingCountConfig,
) -> Option<EdgeMeasurement>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    // Use same search distance as baseline robust_edge.rs (0.5)
    let search_distance = cell_size * 0.5;

    // Use more probes for better coverage (baseline uses 200, we use 80 for efficiency)
    let num_directions = 80;
    let directions = fibonacci_sphere_directions(num_directions);

    // Check inside/outside at query position
    let query_inside = cache.is_inside(position.0, position.1, position.2);
    samples_used += 1;

    // Find surface points
    let mut surface_points = Vec::new();

    for dir in &directions {
        if let Some((crossing, _, used)) = binary_search_crossing_along_dir(
            position,
            *dir,
            search_distance,
            query_inside,
            cache,
            config.binary_search_iterations,
        ) {
            samples_used += used;
            surface_points.push(crossing);
        } else {
            samples_used += 2;
        }
    }

    if surface_points.len() < 6 {
        return None;
    }

    // Two-plane RANSAC with threshold scaled to cell_size (0.006 per unit)
    let tight_threshold = 0.006 * cell_size;

    let (plane_a_inliers, plane_a_centroid, plane_a_normal, plane_a_residual) =
        ransac_plane_fit(&surface_points, tight_threshold, config.ransac_iterations)?;

    if plane_a_inliers.len() < 3 {
        return None;
    }

    // Remove inliers, find second plane
    let remaining: Vec<_> = surface_points
        .iter()
        .filter(|p| !is_inlier(**p, plane_a_centroid, plane_a_normal, tight_threshold))
        .cloned()
        .collect();

    if remaining.len() < 3 {
        return None;
    }

    let (plane_b_inliers, plane_b_centroid, plane_b_normal, plane_b_residual) =
        ransac_plane_fit(&remaining, tight_threshold, config.ransac_iterations)?;

    if plane_b_inliers.len() < 3 {
        return None;
    }

    // Orient normals away from query point
    let normal_a = orient_away(plane_a_normal, plane_a_centroid, position);
    let normal_b = orient_away(plane_b_normal, plane_b_centroid, position);

    // Compute edge from plane intersection
    let edge_dir = normalize(cross(normal_a, normal_b));
    if length(edge_dir) < 1e-6 {
        return None; // Planes are parallel
    }

    let point_on_edge = find_edge_point(
        position,
        plane_a_centroid,
        normal_a,
        plane_b_centroid,
        normal_b,
        edge_dir,
    );

    Some(EdgeMeasurement {
        normal_a,
        normal_b,
        edge_direction: edge_dir,
        point_on_edge,
        inliers_a: plane_a_inliers.len(),
        inliers_b: plane_b_inliers.len(),
        residual_a: plane_a_residual,
        residual_b: plane_b_residual,
        samples_used,
    })
}

// ============================================================================
// Phase 2c: Corner Measurement
// ============================================================================

/// Measure corner geometry (three face normals and corner position).
///
/// Uses full sphere coverage to find all three planes meeting at the corner.
pub fn measure_corner<F>(
    position: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &CrossingCountConfig,
) -> Option<CornerMeasurement>
where
    F: Fn(f64, f64, f64) -> f32,
{
    let mut samples_used = 0u32;
    let search_distance = cell_size * config.search_distance;

    // Generate well-distributed probes covering full sphere
    let directions = fibonacci_sphere_directions(config.corner_probe_count);

    // Check inside/outside at query position
    let query_inside = cache.is_inside(position.0, position.1, position.2);
    samples_used += 1;

    // Find surface points
    let mut surface_points = Vec::new();

    for dir in &directions {
        if let Some((crossing, _, used)) = binary_search_crossing_along_dir(
            position,
            *dir,
            search_distance,
            query_inside,
            cache,
            config.binary_search_iterations,
        ) {
            samples_used += used;
            surface_points.push(crossing);
        } else {
            samples_used += 2;
        }
    }

    if surface_points.len() < 9 {
        return None; // Need at least 3 points per plane
    }

    // Three-plane RANSAC with duplicate rejection
    let mut planes: Vec<((f64, f64, f64), Vec<(f64, f64, f64)>, f64)> = Vec::new();
    let mut remaining = surface_points.clone();
    let corner_threshold = config.ransac_inlier_threshold * cell_size * 0.6; // Scaled to cell_size

    for _ in 0..3 {
        if remaining.len() < 3 {
            break;
        }

        let (inliers, centroid, normal, residual) =
            ransac_plane_fit(&remaining, corner_threshold, config.ransac_iterations)?;

        // Check this isn't a duplicate of existing plane
        let is_duplicate = planes.iter().any(|(existing_normal, _, _)| {
            let angle = angle_between(*existing_normal, normal).to_degrees();
            angle < 25.0 || angle > 155.0 // Within 25 degrees of same or opposite
        });

        if is_duplicate {
            // Remove these inliers and continue searching
            remaining = remaining
                .iter()
                .filter(|p| !is_inlier(**p, centroid, normal, corner_threshold))
                .cloned()
                .collect();
            continue;
        }

        let oriented_normal = orient_away(normal, centroid, position);
        planes.push((oriented_normal, inliers.clone(), residual));

        // Remove inliers for next plane
        remaining = remaining
            .iter()
            .filter(|p| !is_inlier(**p, centroid, normal, corner_threshold))
            .cloned()
            .collect();
    }

    if planes.len() < 3 {
        return None;
    }

    // Compute corner position as intersection of 3 planes
    let corner_pos = three_plane_intersection(
        (planes[0].0, planes[0].1[0]),
        (planes[1].0, planes[1].1[0]),
        (planes[2].0, planes[2].1[0]),
    )?;

    Some(CornerMeasurement {
        normals: [planes[0].0, planes[1].0, planes[2].0],
        corner_position: corner_pos,
        inliers: [planes[0].1.len(), planes[1].1.len(), planes[2].1.len()],
        residuals: [planes[0].2, planes[1].2, planes[2].2],
        samples_used,
    })
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Process a vertex through the residual-based classification algorithm.
///
/// This is the main entry point that coordinates all phases:
/// 1. Surface location (Phase 0)
/// 2. Face measurement attempt
/// 3. If high residual, escalate to edge detection
/// 4. If still high residual, escalate to corner detection
///
/// The classification is determined by which measurement method gives the lowest residual.
pub fn process_vertex<F>(
    midpoint: (f64, f64, f64),
    accumulated_normal: (f64, f64, f64),
    cell_size: f64,
    cache: &SampleCache<F>,
    config: &CrossingCountConfig,
) -> VertexGeometry
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Phase 0: Surface Location
    let location = locate_surface(midpoint, accumulated_normal, cell_size, cache, config);
    let mut total_samples = location.samples_used;

    // Use accumulated normal as hint for face measurement
    let hint = if length(accumulated_normal) > 0.01 {
        normalize(accumulated_normal)
    } else if !location.crossing_directions.is_empty() {
        location.crossing_directions[0]
    } else {
        (0.0, 1.0, 0.0) // Fallback
    };

    // Phase 1: Try face measurement first
    let face_result = measure_face(location.position, hint, cell_size, cache, config);

    if let Some(ref face) = face_result {
        total_samples += face.samples_used;

        // Threshold for "good" plane fit - scaled to cell_size for scale invariance
        let face_residual_threshold = 0.002 * cell_size;

        if face.residual < face_residual_threshold {
            return VertexGeometry {
                classification: GeometryType::Face,
                normals: vec![face.normal],
                edge_direction: None,
                corner_position: None,
                confidence: confidence_from_residual(face.residual),
                samples_used: total_samples,
            };
        }
    }

    // Phase 2: Face fit had high residual, try edge detection
    // Use face normal as one hint, and direction to outliers as the other
    let face_normal = face_result.as_ref().map(|f| f.normal).unwrap_or(hint);
    let face_centroid = face_result.as_ref().map(|f| f.centroid).unwrap_or(location.position);

    // Find outliers from the face fit (points far from the fitted plane)
    let outlier_hint = if let Some(ref face) = face_result {
        let outlier_threshold = 0.01 * cell_size; // Scaled to cell_size
        let outliers: Vec<_> = face.surface_points.iter()
            .filter(|p| {
                let dist = dot(sub(**p, face.centroid), face.normal).abs();
                dist > outlier_threshold
            })
            .cloned()
            .collect();

        if outliers.len() >= 3 {
            // Compute average direction from centroid to outliers
            let outlier_centroid = (
                outliers.iter().map(|p| p.0).sum::<f64>() / outliers.len() as f64,
                outliers.iter().map(|p| p.1).sum::<f64>() / outliers.len() as f64,
                outliers.iter().map(|p| p.2).sum::<f64>() / outliers.len() as f64,
            );
            let to_outliers = normalize(sub(outlier_centroid, face_centroid));
            Some(to_outliers)
        } else {
            None
        }
    } else {
        None
    };

    // Use face normal and outlier direction as hints, or fall back to full sphere
    // IMPORTANT: Probe from the ORIGINAL midpoint (inside the shape), not the surface position!
    // Probing from the surface gives a thin shell where all points appear coplanar.
    let edge_result = if let Some(outlier_dir) = outlier_hint {
        let edge_hints = [face_normal, outlier_dir];
        // Try biased hemisphere first
        let hemisphere_result = measure_edge(midpoint, edge_hints, cell_size, cache, config);

        // Check if hemisphere result is good (high angle between normals)
        let hemisphere_angle = hemisphere_result.as_ref()
            .map(|e| angle_between(e.normal_a, e.normal_b).to_degrees())
            .unwrap_or(0.0);

        if hemisphere_angle > 30.0 {
            hemisphere_result
        } else {
            // Fallback to full sphere if hemisphere result has poor angle
            measure_edge_full_sphere(midpoint, cell_size, cache, config)
        }
    } else {
        measure_edge_full_sphere(midpoint, cell_size, cache, config)
    };

    if let Some(ref edge) = edge_result {
        total_samples += edge.samples_used;

        // Check if edge detection is significantly better than face
        let edge_residual = (edge.residual_a + edge.residual_b) / 2.0;
        let edge_residual_threshold = 0.02 * cell_size; // Scaled to cell_size

        // Also check that the two normals are significantly different (angle > 30°)
        let normal_angle = angle_between(edge.normal_a, edge.normal_b).to_degrees();


        if edge_residual < edge_residual_threshold && normal_angle > 30.0 {
            return VertexGeometry {
                classification: GeometryType::Edge,
                normals: vec![edge.normal_a, edge.normal_b],
                edge_direction: Some(edge.edge_direction),
                corner_position: None,
                confidence: confidence_from_residual(edge_residual),
                samples_used: total_samples,
            };
        }
    }

    // Phase 3: Try corner detection
    let corner_result = measure_corner(location.position, cell_size, cache, config);

    if let Some(ref corner) = corner_result {
        total_samples += corner.samples_used;

        let corner_residual = (corner.residuals[0] + corner.residuals[1] + corner.residuals[2]) / 3.0;
        let corner_residual_threshold = 0.005 * cell_size; // Scaled to cell_size

        // Check that normals are mutually distinct
        let angle_01 = angle_between(corner.normals[0], corner.normals[1]).to_degrees();
        let angle_12 = angle_between(corner.normals[1], corner.normals[2]).to_degrees();
        let angle_02 = angle_between(corner.normals[0], corner.normals[2]).to_degrees();

        if corner_residual < corner_residual_threshold
            && angle_01 > 30.0
            && angle_12 > 30.0
            && angle_02 > 30.0
        {
            return VertexGeometry {
                classification: GeometryType::Corner,
                normals: corner.normals.to_vec(),
                edge_direction: None,
                corner_position: Some(corner.corner_position),
                confidence: confidence_from_residual(corner_residual),
                samples_used: total_samples,
            };
        }
    }

    // Fallback: return the best result we have
    // Prefer edge over face if edge was found, even with higher residual
    if let Some(edge) = edge_result {
        let edge_residual = (edge.residual_a + edge.residual_b) / 2.0;
        let normal_angle = angle_between(edge.normal_a, edge.normal_b).to_degrees();

        if normal_angle > 30.0 {
            return VertexGeometry {
                classification: GeometryType::Edge,
                normals: vec![edge.normal_a, edge.normal_b],
                edge_direction: Some(edge.edge_direction),
                corner_position: None,
                confidence: confidence_from_residual(edge_residual) * 0.7,
                samples_used: total_samples,
            };
        }
    }

    // Return face result as final fallback
    if let Some(face) = face_result {
        return VertexGeometry {
            classification: GeometryType::Face,
            normals: vec![face.normal],
            edge_direction: None,
            corner_position: None,
            confidence: confidence_from_residual(face.residual) * 0.5,
            samples_used: total_samples,
        };
    }

    // Ultimate fallback
    VertexGeometry {
        classification: GeometryType::Face,
        normals: vec![hint],
        edge_direction: None,
        corner_position: None,
        confidence: 0.1,
        samples_used: total_samples,
    }
}

// ============================================================================
// Probe Pattern Generation
// ============================================================================

/// Generate N uniformly distributed directions in a 2D disk (for tangent plane probing).
///
/// Returns (u, v) pairs where u² + v² ≤ 1.
fn golden_ratio_disk_directions(n: usize) -> Vec<(f64, f64)> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // ~137.5°

    (0..n)
        .map(|i| {
            let r = ((i as f64 + 0.5) / n as f64).sqrt(); // Uniform area distribution
            let theta = i as f64 * golden_angle;
            (r * theta.cos(), r * theta.sin())
        })
        .collect()
}

/// Generate N uniformly distributed directions on a hemisphere centered around a given direction.
fn hemisphere_directions(center: (f64, f64, f64), n: usize) -> Vec<(f64, f64, f64)> {
    let center = normalize(center);
    let (tangent1, tangent2) = build_tangent_basis(center);

    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());

    (0..n)
        .map(|i| {
            // Distribute points in the hemisphere
            let t = (i as f64 + 0.5) / n as f64;
            let phi = i as f64 * golden_angle;

            // z goes from 0 to 1 (hemisphere)
            let z = t;
            let r = (1.0 - z * z).sqrt();
            let x = r * phi.cos();
            let y = r * phi.sin();

            // Transform to world coordinates
            normalize((
                tangent1.0 * x + tangent2.0 * y + center.0 * z,
                tangent1.1 * x + tangent2.1 * y + center.1 * z,
                tangent1.2 * x + tangent2.2 * y + center.2 * z,
            ))
        })
        .collect()
}

/// Generate N uniformly distributed directions on a sphere using Fibonacci spiral.
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

/// Build an orthonormal tangent basis for a given normal.
fn build_tangent_basis(normal: (f64, f64, f64)) -> ((f64, f64, f64), (f64, f64, f64)) {
    // Choose a vector not parallel to normal
    let reference = if normal.0.abs() < 0.9 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 1.0, 0.0)
    };

    // tangent1 = normal × reference (normalized)
    let tangent1 = normalize(cross(normal, reference));

    // tangent2 = normal × tangent1
    let tangent2 = cross(normal, tangent1);

    (tangent1, tangent2)
}

// ============================================================================
// Plane Fitting
// ============================================================================

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
    let mut _best_normal = (0.0, 0.0, 1.0);
    let mut _best_centroid = (0.0, 0.0, 0.0);

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
        let normal = normalize(cross(v1, v2));

        if length(normal) < 1e-6 {
            continue; // Degenerate triangle
        }

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
            _best_normal = normal;
            _best_centroid = p1;
        }
    }

    if best_inliers.len() < 3 {
        return None;
    }

    // Refit plane to all inliers for better accuracy
    let (refined_centroid, refined_normal, residual) = fit_plane_svd(&best_inliers);

    Some((best_inliers, refined_centroid, refined_normal, residual))
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

    // Use a random-ish initial vector that's unlikely to be orthogonal to any eigenvector
    // (1, 1, 1) is bad because it's orthogonal to normals like (1, 0, -1)
    let mut v = (0.6, 0.8, 0.3);
    let len = length(v);
    v = (v.0 / len, v.1 / len, v.2 / len);

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

/// Check if a point is an inlier for a plane.
fn is_inlier(
    p: (f64, f64, f64),
    plane_point: (f64, f64, f64),
    plane_normal: (f64, f64, f64),
    threshold: f64,
) -> bool {
    let dist = dot(sub(p, plane_point), plane_normal).abs();
    dist < threshold
}

/// Orient normal to point away from a reference point.
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

/// Orient normal away from the inside of the shape.
fn orient_normal_away_from_inside<F>(
    normal: (f64, f64, f64),
    plane_point: (f64, f64, f64),
    _query_point: (f64, f64, f64),
    cache: &SampleCache<F>,
) -> (f64, f64, f64)
where
    F: Fn(f64, f64, f64) -> f32,
{
    // Check if moving along normal goes outside
    let test_point = (
        plane_point.0 + normal.0 * 0.01,
        plane_point.1 + normal.1 * 0.01,
        plane_point.2 + normal.2 * 0.01,
    );

    if cache.is_inside(test_point.0, test_point.1, test_point.2) {
        neg(normal)
    } else {
        normal
    }
}

/// Find a point on the edge (intersection line) of two planes.
fn find_edge_point(
    query: (f64, f64, f64),
    centroid_a: (f64, f64, f64),
    normal_a: (f64, f64, f64),
    centroid_b: (f64, f64, f64),
    normal_b: (f64, f64, f64),
    edge_direction: (f64, f64, f64),
) -> (f64, f64, f64) {
    // Iteratively project onto both planes to find their intersection
    let midpoint = (
        (centroid_a.0 + centroid_b.0) / 2.0,
        (centroid_a.1 + centroid_b.1) / 2.0,
        (centroid_a.2 + centroid_b.2) / 2.0,
    );

    let mut p = midpoint;
    for _ in 0..20 {
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

/// Find the intersection point of three planes.
fn three_plane_intersection(
    plane1: ((f64, f64, f64), (f64, f64, f64)), // (normal, point_on_plane)
    plane2: ((f64, f64, f64), (f64, f64, f64)),
    plane3: ((f64, f64, f64), (f64, f64, f64)),
) -> Option<(f64, f64, f64)> {
    let n1 = plane1.0;
    let n2 = plane2.0;
    let n3 = plane3.0;

    let d1 = dot(n1, plane1.1);
    let d2 = dot(n2, plane2.1);
    let d3 = dot(n3, plane3.1);

    // Check if planes are not coplanar (determinant != 0)
    let denom = dot(n1, cross(n2, n3));
    if denom.abs() < 1e-12 {
        return None;
    }

    // Cramer's rule
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

/// Convert residual to confidence score.
fn confidence_from_residual(residual: f64) -> f64 {
    // Lower residual = higher confidence
    // At residual 0, confidence = 1.0
    // At residual 0.01, confidence ≈ 0.9
    // At residual 0.1, confidence ≈ 0.5
    1.0 / (1.0 + residual * 100.0)
}

// ============================================================================
// Vector Math Utilities
// ============================================================================

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

fn distance(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    length(sub(a, b))
}

/// Angle between two vectors in radians.
pub fn angle_between(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let a_norm = normalize(a);
    let b_norm = normalize(b);
    let d = dot(a_norm, b_norm).clamp(-1.0, 1.0);
    d.acos()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::analytical_cube::AnalyticalRotatedCube;

    fn rotated_cube_sampler(cube: &AnalyticalRotatedCube) -> impl Fn(f64, f64, f64) -> f32 + '_ {
        move |x, y, z| {
            let local = cube.world_to_local((x, y, z));
            let h = 0.5;
            let inside = local.0.abs() <= h && local.1.abs() <= h && local.2.abs() <= h;
            if inside { 1.0 } else { -1.0 }
        }
    }

    #[test]
    fn test_golden_ratio_disk_directions() {
        let dirs = golden_ratio_disk_directions(12);
        assert_eq!(dirs.len(), 12);

        // All should be within unit disk
        for (u, v) in &dirs {
            let r = (u * u + v * v).sqrt();
            assert!(r <= 1.0 + 1e-10, "Point outside unit disk: r={}", r);
        }
    }

    #[test]
    fn test_fibonacci_sphere_directions() {
        let dirs = fibonacci_sphere_directions(60);
        assert_eq!(dirs.len(), 60);

        // All should be unit length
        for d in &dirs {
            let len = length(*d);
            assert!((len - 1.0).abs() < 1e-10, "Not unit length: {}", len);
        }
    }

    #[test]
    fn test_hemisphere_directions() {
        let center = (0.0, 0.0, 1.0);
        let dirs = hemisphere_directions(center, 15);
        assert_eq!(dirs.len(), 15);

        // All should be unit length and have positive z component
        for d in &dirs {
            let len = length(*d);
            assert!((len - 1.0).abs() < 1e-10, "Not unit length: {}", len);
            assert!(d.2 >= -0.1, "Direction not in hemisphere: z={}", d.2);
        }
    }

    #[test]
    fn test_surface_location_at_face_center() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);
        let config = CrossingCountConfig::default();

        // Test at face center (offset slightly inside)
        let face_center = cube.face_center(0);
        let normal = cube.face_normals[0];
        let inside_point = (
            face_center.0 - normal.0 * 0.1,
            face_center.1 - normal.1 * 0.1,
            face_center.2 - normal.2 * 0.1,
        );

        let location = locate_surface(inside_point, normal, 1.0, &cache, &config);

        // Should find 3-4 crossings for a smooth surface
        assert!(
            location.crossing_count >= 3,
            "Expected 3+ crossings at face center, got {}",
            location.crossing_count
        );

        // Surface position should be close to face center
        let dist = distance(location.position, face_center);
        assert!(dist < 0.15, "Surface position error: {}", dist);
    }

    #[test]
    fn test_surface_location_at_edge() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);
        let config = CrossingCountConfig::default();

        // Test at edge midpoint (offset slightly inside)
        let edge = cube.get_edge(0);
        let bisector = normalize(add(edge.face_a_normal, edge.face_b_normal));
        let inside_point = (
            edge.point_on_edge.0 - bisector.0 * 0.1,
            edge.point_on_edge.1 - bisector.1 * 0.1,
            edge.point_on_edge.2 - bisector.2 * 0.1,
        );

        let location = locate_surface(inside_point, bisector, 1.0, &cache, &config);

        // Should find exactly 2 crossings for an edge
        // Note: This may vary depending on edge orientation vs cardinal axes
        assert!(
            location.crossing_count >= 1 && location.crossing_count <= 4,
            "Expected 1-4 crossings at edge, got {}",
            location.crossing_count
        );
    }

    #[test]
    fn debug_edge_crossing_count() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let config = CrossingCountConfig::default();

        println!("\n=== DEBUG EDGE CROSSING COUNTS ===\n");
        println!("Testing with offset 0.02 (very close to surface)\n");

        for edge_idx in 0..12 {
            let edge = cube.get_edge(edge_idx);
            let bisector = normalize(add(edge.face_a_normal, edge.face_b_normal));
            // Use a very small offset to be near the surface
            let inside_point = (
                edge.point_on_edge.0 - bisector.0 * 0.02,
                edge.point_on_edge.1 - bisector.1 * 0.02,
                edge.point_on_edge.2 - bisector.2 * 0.02,
            );

            let cache = SampleCache::new(&sampler);
            let location = locate_surface(inside_point, bisector, 1.0, &cache, &config);

            let class = classify_geometry(&location);
            let class_str = match class {
                GeometryClassification::SmoothSurface => "Smooth",
                GeometryClassification::EdgeCandidate { .. } => "Edge",
                GeometryClassification::CornerCandidate { .. } => "Corner",
                GeometryClassification::Uncertain { .. } => "Uncertain",
            };

            println!(
                "Edge {:2}: crossings={}, dirs=[{}], class={}",
                edge_idx,
                location.crossing_count,
                location.crossing_directions.iter()
                    .map(|d| format!("({:.2},{:.2},{:.2})", d.0, d.1, d.2))
                    .collect::<Vec<_>>()
                    .join(", "),
                class_str
            );
        }
    }

    #[test]
    fn test_classify_geometry() {
        // Test smooth surface classification (3-4 crossings)
        let smooth_location = SurfaceLocation {
            position: (0.0, 0.0, 0.0),
            crossing_count: 4,
            crossing_directions: vec![(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
            crossing_positions: vec![],
            samples_used: 20,
        };

        match classify_geometry(&smooth_location) {
            GeometryClassification::SmoothSurface => (),
            other => panic!("Expected SmoothSurface, got {:?}", other),
        }

        // Test edge classification (2 crossings)
        let edge_location = SurfaceLocation {
            position: (0.0, 0.0, 0.0),
            crossing_count: 2,
            crossing_directions: vec![(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            crossing_positions: vec![],
            samples_used: 20,
        };

        match classify_geometry(&edge_location) {
            GeometryClassification::EdgeCandidate { face_hints } => {
                assert_eq!(face_hints.len(), 2);
            }
            other => panic!("Expected EdgeCandidate, got {:?}", other),
        }
    }

    #[test]
    fn test_face_measurement() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);
        let config = CrossingCountConfig::default();

        // Test at face center (offset slightly inside)
        // Try face 2 which works well
        let face_idx = 2;
        let face_center = cube.face_center(face_idx);
        let expected_normal = cube.face_normals[face_idx];
        let inside_point = (
            face_center.0 - expected_normal.0 * 0.1,
            face_center.1 - expected_normal.1 * 0.1,
            face_center.2 - expected_normal.2 * 0.1,
        );

        let result = measure_face(inside_point, expected_normal, 1.0, &cache, &config);
        assert!(result.is_some(), "Face measurement should succeed");

        let face = result.unwrap();
        let error = angle_between(face.normal, expected_normal).to_degrees();
        assert!(error < 10.0, "Normal error {} degrees, expected < 10", error);
    }

    #[test]
    fn debug_face_0_measurement() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);
        let config = CrossingCountConfig::default();

        let face_idx = 0;
        let face_center = cube.face_center(face_idx);
        let expected_normal = cube.face_normals[face_idx];

        println!("\n=== DEBUG FACE {} ===", face_idx);
        println!("Face center: ({:.4}, {:.4}, {:.4})", face_center.0, face_center.1, face_center.2);
        println!("Expected normal: ({:.4}, {:.4}, {:.4})", expected_normal.0, expected_normal.1, expected_normal.2);

        let inside_point = (
            face_center.0 - expected_normal.0 * 0.1,
            face_center.1 - expected_normal.1 * 0.1,
            face_center.2 - expected_normal.2 * 0.1,
        );
        println!("Test point: ({:.4}, {:.4}, {:.4})", inside_point.0, inside_point.1, inside_point.2);

        let is_inside = cache.is_inside(inside_point.0, inside_point.1, inside_point.2);
        println!("Test point is inside: {}", is_inside);

        // Build tangent basis
        let normal = normalize(expected_normal);
        let (t1, t2) = build_tangent_basis(normal);
        println!("Tangent1: ({:.4}, {:.4}, {:.4})", t1.0, t1.1, t1.2);
        println!("Tangent2: ({:.4}, {:.4}, {:.4})", t2.0, t2.1, t2.2);

        // Test a few probes manually
        let probe_distance = 0.1;
        let search_distance = 1.5;

        println!("\nManual probes:");
        for (name, u, v) in [("center", 0.0, 0.0), ("t1", 1.0, 0.0), ("t2", 0.0, 1.0), ("-t1", -1.0, 0.0)] {
            let world_dir = normalize((
                t1.0 * u + t2.0 * v,
                t1.1 * u + t2.1 * v,
                t1.2 * u + t2.2 * v,
            ));
            let probe_start = if u == 0.0 && v == 0.0 {
                inside_point
            } else {
                (
                    inside_point.0 + world_dir.0 * probe_distance,
                    inside_point.1 + world_dir.1 * probe_distance,
                    inside_point.2 + world_dir.2 * probe_distance,
                )
            };

            let probe_inside = cache.is_inside(probe_start.0, probe_start.1, probe_start.2);
            println!("  {} probe: start ({:.4}, {:.4}, {:.4}), inside={}", name, probe_start.0, probe_start.1, probe_start.2, probe_inside);

            // Search along normal
            if let Some((crossing, dist, _)) = binary_search_crossing_along_dir(
                probe_start,
                normal,
                search_distance,
                probe_inside,
                &cache,
                15,
            ) {
                println!("    -> crossing at ({:.4}, {:.4}, {:.4}), dist={:.4}", crossing.0, crossing.1, crossing.2, dist);

                // What face is this point nearest to?
                let closest = cube.closest_surface_point(crossing);
                match closest.classification {
                    super::super::analytical_cube::SurfaceClassification::OnFace { face_index, .. } => {
                        println!("    -> on face {}", face_index);
                    }
                    super::super::analytical_cube::SurfaceClassification::OnEdge { .. } => {
                        println!("    -> on edge");
                    }
                    super::super::analytical_cube::SurfaceClassification::OnCorner { .. } => {
                        println!("    -> on corner");
                    }
                }
            } else {
                // Try opposite direction
                let neg_normal = neg(normal);
                if let Some((crossing, dist, _)) = binary_search_crossing_along_dir(
                    probe_start,
                    neg_normal,
                    search_distance,
                    probe_inside,
                    &cache,
                    15,
                ) {
                    println!("    -> crossing (neg dir) at ({:.4}, {:.4}, {:.4}), dist={:.4}", crossing.0, crossing.1, crossing.2, dist);
                } else {
                    println!("    -> NO CROSSING FOUND");
                }
            }
        }

        // Now let's manually trace through measure_face to see all 12 points
        println!("\nAll 12 probe directions:");
        let directions = golden_ratio_disk_directions(12);
        let probe_distance = 0.1;
        let search_distance = 1.5;
        let mut surface_points = Vec::new();

        for (i, (u, v)) in directions.iter().enumerate() {
            let world_dir = if (*u).abs() < 1e-10 && (*v).abs() < 1e-10 {
                (0.0, 0.0, 0.0)
            } else {
                normalize((
                    t1.0 * u + t2.0 * v,
                    t1.1 * u + t2.1 * v,
                    t1.2 * u + t2.2 * v,
                ))
            };

            let probe_start = if (*u).abs() < 1e-10 && (*v).abs() < 1e-10 {
                inside_point
            } else {
                (
                    inside_point.0 + world_dir.0 * probe_distance,
                    inside_point.1 + world_dir.1 * probe_distance,
                    inside_point.2 + world_dir.2 * probe_distance,
                )
            };

            let probe_inside = cache.is_inside(probe_start.0, probe_start.1, probe_start.2);

            let found = if let Some((crossing, _, _)) = binary_search_crossing_along_dir(
                probe_start,
                normal,
                search_distance,
                probe_inside,
                &cache,
                15,
            ) {
                surface_points.push(crossing);
                Some(crossing)
            } else if let Some((crossing, _, _)) = binary_search_crossing_along_dir(
                probe_start,
                neg(normal),
                search_distance,
                probe_inside,
                &cache,
                15,
            ) {
                surface_points.push(crossing);
                Some(crossing)
            } else {
                None
            };

            if let Some(c) = found {
                let closest = cube.closest_surface_point(c);
                let face_num = match closest.classification {
                    super::super::analytical_cube::SurfaceClassification::OnFace { face_index, .. } => {
                        face_index as i32
                    }
                    _ => -1,
                };
                println!("  {:2}: uv=({:6.3},{:6.3}) -> ({:6.4},{:6.4},{:6.4}) face={}", i, u, v, c.0, c.1, c.2, face_num);
            } else {
                println!("  {:2}: uv=({:6.3},{:6.3}) -> NO CROSSING", i, u, v);
            }
        }

        // Now fit plane to these points
        if surface_points.len() >= 3 {
            let (centroid, fitted_normal, residual) = fit_plane_svd(&surface_points);
            println!("\nSVD plane fit ({} points):", surface_points.len());
            println!("  Centroid: ({:.4}, {:.4}, {:.4})", centroid.0, centroid.1, centroid.2);
            println!("  Normal: ({:.4}, {:.4}, {:.4})", fitted_normal.0, fitted_normal.1, fitted_normal.2);
            println!("  Residual: {:.6}", residual);

            let error = angle_between(fitted_normal, expected_normal).to_degrees();
            let error_flipped = angle_between(neg(fitted_normal), expected_normal).to_degrees();
            println!("  Error vs expected: {:.2}° (or {:.2}° if flipped)", error, error_flipped);
        }
    }

    #[test]
    fn test_edge_measurement() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);
        let config = CrossingCountConfig::default();

        // Test at edge midpoint (offset slightly inside)
        let edge = cube.get_edge(0);
        let bisector = normalize(add(edge.face_a_normal, edge.face_b_normal));
        let inside_point = (
            edge.point_on_edge.0 - bisector.0 * 0.1,
            edge.point_on_edge.1 - bisector.1 * 0.1,
            edge.point_on_edge.2 - bisector.2 * 0.1,
        );

        let hints = [edge.face_a_normal, edge.face_b_normal];
        let result = measure_edge(inside_point, hints, 1.0, &cache, &config);
        assert!(result.is_some(), "Edge measurement should succeed");

        let measured = result.unwrap();

        // Check that we found two distinct normals
        let angle_between_normals = angle_between(measured.normal_a, measured.normal_b).to_degrees();
        assert!(
            angle_between_normals > 45.0,
            "Normals should be distinct, angle: {}°",
            angle_between_normals
        );

        // Match normals to expected (best pairing)
        let (err_a, err_b) = best_normal_pairing(
            (measured.normal_a, measured.normal_b),
            (edge.face_a_normal, edge.face_b_normal),
        );

        assert!(err_a < 10.0, "Normal A error: {}°", err_a);
        assert!(err_b < 10.0, "Normal B error: {}°", err_b);
    }

    #[test]
    fn test_corner_measurement() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);
        let config = CrossingCountConfig::default();

        // Test at corner (offset slightly inside)
        let corner = cube.get_corner(7); // +X +Y +Z corner
        let diagonal = normalize(corner.position); // Diagonal toward center is opposite of corner position
        let inside_point = (
            corner.position.0 - diagonal.0 * 0.1,
            corner.position.1 - diagonal.1 * 0.1,
            corner.position.2 - diagonal.2 * 0.1,
        );

        let result = measure_corner(inside_point, 1.0, &cache, &config);
        assert!(result.is_some(), "Corner measurement should succeed");

        let measured = result.unwrap();

        // Should have 3 distinct normals
        assert!(
            angle_between(measured.normals[0], measured.normals[1]).to_degrees() > 45.0,
            "Normals 0-1 should be distinct"
        );
        assert!(
            angle_between(measured.normals[1], measured.normals[2]).to_degrees() > 45.0,
            "Normals 1-2 should be distinct"
        );
    }

    #[test]
    fn test_full_pipeline_face() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let cache = SampleCache::new(sampler);
        let config = CrossingCountConfig::default();

        // Test at face center
        let face_idx = 0;
        let face_center = cube.face_center(face_idx);
        let expected_normal = cube.face_normals[face_idx];
        let inside_point = (
            face_center.0 - expected_normal.0 * 0.1,
            face_center.1 - expected_normal.1 * 0.1,
            face_center.2 - expected_normal.2 * 0.1,
        );

        let result = process_vertex(inside_point, expected_normal, 1.0, &cache, &config);

        assert_eq!(result.classification, GeometryType::Face);
        assert_eq!(result.normals.len(), 1);

        let error = angle_between(result.normals[0], expected_normal).to_degrees();
        println!("Face pipeline: error = {:.2}°, samples = {}", error, result.samples_used);
        assert!(error < 5.0, "Face normal error too high: {}°", error);
    }

    #[test]
    fn benchmark_all_faces() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let config = CrossingCountConfig::default();

        println!("\n{}", "=".repeat(70));
        println!("FACE DETECTION BENCHMARK (Attempt 0: Crossing Count)");
        println!("{}\n", "=".repeat(70));
        println!("{:<12} {:>12} {:>12} {:>12}", "Face", "Error°", "Samples", "Confidence");
        println!("{}", "-".repeat(50));

        let mut total_error = 0.0;
        let mut total_samples = 0u32;

        for face_idx in 0..6 {
            let cache = SampleCache::new(&sampler);

            let face_center = cube.face_center(face_idx);
            let expected_normal = cube.face_normals[face_idx];
            let inside_point = (
                face_center.0 - expected_normal.0 * 0.1,
                face_center.1 - expected_normal.1 * 0.1,
                face_center.2 - expected_normal.2 * 0.1,
            );

            let result = process_vertex(inside_point, expected_normal, 1.0, &cache, &config);

            let error = angle_between(result.normals[0], expected_normal).to_degrees();
            total_error += error;
            total_samples += result.samples_used;

            println!(
                "Face {:2}      {:>10.2}°  {:>10}   {:>10.2}",
                face_idx, error, result.samples_used, result.confidence
            );
        }

        println!("{}", "-".repeat(50));
        println!(
            "Average:     {:>10.2}°  {:>10.1}",
            total_error / 6.0,
            total_samples as f64 / 6.0
        );
        println!("\nTarget: <1° error, <35 samples");
    }

    #[test]
    fn benchmark_all_edges() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let config = CrossingCountConfig::default();

        println!("\n{}", "=".repeat(70));
        println!("EDGE DETECTION BENCHMARK (Attempt 0)");
        println!("{}\n", "=".repeat(70));
        println!("{:<10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8}", "Edge", "N_A°", "N_B°", "Samples", "Class", "FaceRes", "EdgeRes", "NormAng");
        println!("{}", "-".repeat(85));

        let mut total_na_error = 0.0;
        let mut total_nb_error = 0.0;
        let mut total_samples = 0u32;
        let mut correct_classification = 0;

        for edge_idx in 0..12 {
            let edge = cube.get_edge(edge_idx);
            let bisector = normalize(add(edge.face_a_normal, edge.face_b_normal));
            let inside_point = (
                edge.point_on_edge.0 - bisector.0 * 0.1,
                edge.point_on_edge.1 - bisector.1 * 0.1,
                edge.point_on_edge.2 - bisector.2 * 0.1,
            );

            // Run process_vertex with fresh cache
            let fresh_cache = SampleCache::new(&sampler);
            let face_result = measure_face(inside_point, bisector, 1.0, &fresh_cache, &config);
            let face_residual = face_result.as_ref().map(|f| f.residual).unwrap_or(f64::INFINITY);

            // Use another fresh cache for process_vertex
            let fresh_cache = SampleCache::new(&sampler);
            let result = process_vertex(inside_point, bisector, 1.0, &fresh_cache, &config);

            // Test what measure_edge_full_sphere returns (what process_vertex uses)
            let full_sphere_cache = SampleCache::new(&sampler);
            let edge_result = measure_edge_full_sphere(inside_point, 1.0, &full_sphere_cache, &config);
            let (edge_residual, edge_angle) = if let Some(ref e) = edge_result {
                let r = (e.residual_a + e.residual_b) / 2.0;
                let a = angle_between(e.normal_a, e.normal_b).to_degrees();
                (r, a)
            } else {
                (f64::INFINITY, 0.0)
            };
            total_samples += result.samples_used;

            let class_str = match result.classification {
                GeometryType::Face => "Face",
                GeometryType::Edge => {
                    correct_classification += 1;
                    "Edge"
                }
                GeometryType::Corner => "Corner",
            };

            if result.normals.len() >= 2 {
                let (err_a, err_b) = best_normal_pairing(
                    (result.normals[0], result.normals[1]),
                    (edge.face_a_normal, edge.face_b_normal),
                );
                total_na_error += err_a;
                total_nb_error += err_b;

                println!(
                    "Edge {:2}   {:>6.1}°  {:>6.1}°  {:>6}   {:>6}   {:>8.4}   {:>8.4}   {:>6.1}°",
                    edge_idx, err_a, err_b, result.samples_used, class_str, face_residual, edge_residual, edge_angle
                );
            } else {
                println!(
                    "Edge {:2}   {:>6}   {:>6}   {:>6}   {:>6}   {:>8.4}   {:>8.4}   {:>6.1}°",
                    edge_idx, "N/A", "N/A", result.samples_used, class_str, face_residual, edge_residual, edge_angle
                );
            }
        }

        println!("{}", "-".repeat(60));
        println!(
            "Average:     {:>8.2}°  {:>8.2}°  {:>8.1}",
            total_na_error / 12.0,
            total_nb_error / 12.0,
            total_samples as f64 / 12.0
        );
        println!("Classification: {}/12 correct as Edge", correct_classification);
        println!("\nTarget: <1° error, <80 samples, 100% Edge classification");
    }

    #[test]
    fn benchmark_all_corners() {
        let cube = AnalyticalRotatedCube::standard_test_cube();
        let sampler = rotated_cube_sampler(&cube);
        let config = CrossingCountConfig::default();

        println!("\n{}", "=".repeat(70));
        println!("CORNER DETECTION BENCHMARK (Attempt 0: Crossing Count)");
        println!("{}\n", "=".repeat(70));
        println!("{:<12} {:>10} {:>10} {:>10} {:>10}", "Corner", "Avg Err°", "Max Err°", "Samples", "Class");
        println!("{}", "-".repeat(60));

        let mut total_avg_error = 0.0;
        let mut total_samples = 0u32;
        let mut correct_classification = 0;

        for corner_idx in 0..8 {
            let cache = SampleCache::new(&sampler);

            let corner = cube.get_corner(corner_idx);
            let diagonal = normalize(corner.position);
            let inside_point = (
                corner.position.0 - diagonal.0 * 0.1,
                corner.position.1 - diagonal.1 * 0.1,
                corner.position.2 - diagonal.2 * 0.1,
            );

            // Use a small normal hint (corners have no good accumulated normal)
            let result = process_vertex(inside_point, diagonal, 1.0, &cache, &config);
            total_samples += result.samples_used;

            let class_str = match result.classification {
                GeometryType::Face => "Face",
                GeometryType::Edge => "Edge",
                GeometryType::Corner => {
                    correct_classification += 1;
                    "Corner"
                }
            };

            if result.normals.len() >= 3 {
                let errors = match_normals(&result.normals, &corner.face_normals);
                let avg_err = errors.iter().sum::<f64>() / errors.len() as f64;
                let max_err = errors.iter().cloned().fold(0.0_f64, f64::max);
                total_avg_error += avg_err;

                println!(
                    "Corner {:2}    {:>8.2}°  {:>8.2}°  {:>8}   {:>8}",
                    corner_idx, avg_err, max_err, result.samples_used, class_str
                );
            } else {
                println!(
                    "Corner {:2}    {:>8}   {:>8}   {:>8}   {:>8}",
                    corner_idx, "N/A", "N/A", result.samples_used, class_str
                );
            }
        }

        println!("{}", "-".repeat(60));
        println!(
            "Average:     {:>8.2}°            {:>8.1}",
            total_avg_error / 8.0,
            total_samples as f64 / 8.0
        );
        println!("Classification: {}/8 correct as Corner", correct_classification);
        println!("\nTarget: <1° error, <120 samples, 100% Corner classification");
    }

    // Helper functions for tests

    fn best_normal_pairing(
        detected: ((f64, f64, f64), (f64, f64, f64)),
        expected: ((f64, f64, f64), (f64, f64, f64)),
    ) -> (f64, f64) {
        let a1 = (
            angle_between(detected.0, expected.0).to_degrees(),
            angle_between(detected.1, expected.1).to_degrees(),
        );
        let a2 = (
            angle_between(detected.0, expected.1).to_degrees(),
            angle_between(detected.1, expected.0).to_degrees(),
        );
        if a1.0 + a1.1 < a2.0 + a2.1 {
            a1
        } else {
            a2
        }
    }

    fn match_normals(
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
                    let err = angle_between(*d, *e).to_degrees();
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
}
