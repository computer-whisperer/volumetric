//! Fit a local affine [`Subspace`] to a model feature using operator-time
//! occupancy probes — no generated SDF or persistent sampled field.
//!
//! Inputs:
//! - 0: a 3D `ModelWASM`.
//! - 1: `Seed` (`VecF64(3)`), a point on or near the desired feature.
//! - 2: config `{ kind, tolerance, search_radius, max_iterations,
//!   snap_divisions }`.
//!
//! `kind` chooses the feature dimension: a plane is one local surface
//! constraint, a line is the intersection of two detected surface sheets,
//! and a point is the intersection of three. Each iteration samples a 6×6×6
//! neighborhood, bisects only grid edges whose occupancy changes, fits the
//! required independent planes, intersects them nearest the seed, recenters,
//! and shrinks the neighborhood until successive subspaces agree within the
//! requested world-space tolerance.
//!
//! With `snap_divisions > 0`, a result already very close to an axis-aligned
//! subspace may be canonicalized: constrained coordinates within `tolerance`
//! of `min + i / snap_divisions * (max - min)` snap to that exact fraction.
//! Oblique results are never rotated merely because snapping is enabled.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use volumetric_abi::host::{
    cancelled, input_model_bounds, input_model_dimensions, input_model_sample, post_output,
    read_input, report_error,
};
use volumetric_abi::subspace::{Subspace, encode_subspace};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

type Vec3 = [f64; 3];

// Six samples per axis ensure that, after an outside seed recenters onto an
// edge or corner, every incident sheet contributes a two-dimensional patch.
// Smaller stencils can make the union of two one-dimensional sample rows look
// exactly like a spurious diagonal plane.
const GRID_SIDE: usize = 6;
const MAX_BISECTION_STEPS: usize = 28;
const EXHAUSTIVE_CANDIDATE_LIMIT: usize = 48;
const LOCAL_CANDIDATE_NEIGHBORS: usize = 8;
const INDEPENDENCE_EPSILON: f64 = 1e-6;
const AXIS_SNAP_PROJECTOR_TOLERANCE: f64 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum FeatureKind {
    Point,
    Line,
    Plane,
}

impl FeatureKind {
    fn rank(self) -> usize {
        match self {
            FeatureKind::Point => 0,
            FeatureKind::Line => 1,
            FeatureKind::Plane => 2,
        }
    }

    fn codimension(self) -> usize {
        3 - self.rank()
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct FitConfig {
    kind: FeatureKind,
    tolerance: f64,
    /// Zero chooses five percent of the model bounds diagonal.
    search_radius: f64,
    max_iterations: u32,
    /// Zero disables metric snapping; otherwise this is the common grid
    /// denominator along each model-bounds axis.
    snap_divisions: u32,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            kind: FeatureKind::Plane,
            tolerance: 1e-5,
            search_radius: 0.0,
            max_iterations: 20,
            snap_divisions: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Plane {
    normal: Vec3,
    offset: f64,
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(v: Vec3, scalar: f64) -> Vec3 {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: Vec3) -> f64 {
    dot(v, v).sqrt()
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(v: Vec3) -> Option<Vec3> {
    let length = norm(v);
    (length.is_finite() && length > 1e-12).then(|| mul(v, 1.0 / length))
}

fn distance(a: Vec3, b: Vec3) -> f64 {
    norm(sub(a, b))
}

fn canonical_plane(mut normal: Vec3, mut offset: f64) -> Plane {
    if normal
        .iter()
        .copied()
        .find(|component| component.abs() > 1e-12)
        .is_some_and(|component| component < 0.0)
    {
        normal = mul(normal, -1.0);
        offset = -offset;
    }
    Plane { normal, offset }
}

fn decode_vec3(bytes: &[u8]) -> Result<Vec3, String> {
    if bytes.len() != 24 {
        return Err(format!(
            "seed expects exactly 24 bytes of VecF64(3) data, got {}",
            bytes.len()
        ));
    }
    let point = std::array::from_fn(|axis| {
        f64::from_le_bytes(bytes[axis * 8..(axis + 1) * 8].try_into().unwrap())
    });
    if point.iter().any(|value| !value.is_finite()) {
        return Err(format!("seed coordinates must be finite, got {point:?}"));
    }
    Ok(point)
}

fn validate_bounds(bounds: &[f64]) -> Result<(), String> {
    if bounds.len() != 6 {
        return Err(format!(
            "expected six interleaved 3D bounds, got {bounds:?}"
        ));
    }
    for axis in 0..3 {
        let (lo, hi) = (bounds[2 * axis], bounds[2 * axis + 1]);
        if !(lo.is_finite() && hi.is_finite() && hi > lo) {
            return Err(format!("axis {axis} has invalid bounds [{lo}, {hi}]"));
        }
    }
    Ok(())
}

fn validate_config(config: &FitConfig, bounds: &[f64]) -> Result<f64, String> {
    validate_bounds(bounds)?;
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err("tolerance must be finite and greater than zero".to_string());
    }
    if !(config.search_radius.is_finite() && config.search_radius >= 0.0) {
        return Err("search_radius must be finite and non-negative".to_string());
    }
    if !(2..=32).contains(&config.max_iterations) {
        return Err("max_iterations must be between 2 and 32".to_string());
    }
    if config.snap_divisions > 1_000_000 {
        return Err("snap_divisions must not exceed 1000000".to_string());
    }
    let diagonal = (0..3)
        .map(|axis| {
            let extent = bounds[2 * axis + 1] - bounds[2 * axis];
            extent * extent
        })
        .sum::<f64>()
        .sqrt();
    if config.tolerance >= diagonal {
        return Err(format!(
            "tolerance {} must be smaller than the model bounds diagonal {diagonal}",
            config.tolerance
        ));
    }
    let radius = if config.search_radius == 0.0 {
        diagonal * 0.05
    } else {
        config.search_radius
    };
    if radius < config.tolerance {
        return Err(format!(
            "search radius {radius} must be at least the tolerance {}",
            config.tolerance
        ));
    }
    Ok(radius)
}

fn grid_index(x: usize, y: usize, z: usize) -> usize {
    (z * GRID_SIDE + y) * GRID_SIDE + x
}

/// Find local boundary points by sampling one 6×6×6 grid and bisecting only
/// grid edges whose endpoint occupancy differs. Every bisection round is one
/// batched model-probe call, independent of the number of active edges.
fn boundary_points<F>(
    center: Vec3,
    radius: f64,
    boundary_tolerance: f64,
    sample: &mut F,
) -> Result<Vec<Vec3>, String>
where
    F: FnMut(&[Vec3]) -> Result<Vec<bool>, String>,
{
    let mut nodes = Vec::with_capacity(GRID_SIDE.pow(3));
    for z in 0..GRID_SIDE {
        for y in 0..GRID_SIDE {
            for x in 0..GRID_SIDE {
                let grid_coordinate =
                    |index: usize| -1.0 + 2.0 * index as f64 / (GRID_SIDE.saturating_sub(1)) as f64;
                nodes.push([
                    center[0] + grid_coordinate(x) * radius,
                    center[1] + grid_coordinate(y) * radius,
                    center[2] + grid_coordinate(z) * radius,
                ]);
            }
        }
    }
    let occupied = sample(&nodes)?;
    if occupied.len() != nodes.len() {
        return Err("model sampler returned the wrong number of values".to_string());
    }

    let mut brackets: Vec<(Vec3, bool, Vec3, bool)> = Vec::new();
    for axis in 0..3 {
        for z in 0..GRID_SIDE {
            for y in 0..GRID_SIDE {
                for x in 0..GRID_SIDE {
                    let mut next = [x, y, z];
                    if next[axis] + 1 >= GRID_SIDE {
                        continue;
                    }
                    next[axis] += 1;
                    let a = grid_index(x, y, z);
                    let b = grid_index(next[0], next[1], next[2]);
                    if occupied[a] != occupied[b] {
                        brackets.push((nodes[a], occupied[a], nodes[b], occupied[b]));
                    }
                }
            }
        }
    }
    if brackets.is_empty() {
        return Err(format!(
            "no occupancy boundary crosses the probe neighborhood centered at {center:?} with radius {radius}"
        ));
    }

    let bisection_steps =
        ((radius / boundary_tolerance).log2().ceil() as usize).clamp(1, MAX_BISECTION_STEPS);
    for _ in 0..bisection_steps {
        let midpoints: Vec<Vec3> = brackets
            .iter()
            .map(|(a, _, b, _)| mul(add(*a, *b), 0.5))
            .collect();
        let mid_occupied = sample(&midpoints)?;
        if mid_occupied.len() != brackets.len() {
            return Err("model sampler returned the wrong number of bisection values".to_string());
        }
        for (bracket, mid_state) in brackets.iter_mut().zip(mid_occupied) {
            let midpoint = mul(add(bracket.0, bracket.2), 0.5);
            if mid_state == bracket.1 {
                bracket.0 = midpoint;
                bracket.1 = mid_state;
            } else {
                bracket.2 = midpoint;
                bracket.3 = mid_state;
            }
        }
    }

    let mut points = Vec::with_capacity(brackets.len());
    let dedup_tolerance = boundary_tolerance * 2.0;
    for (a, _, b, _) in brackets {
        let point = mul(add(a, b), 0.5);
        if !points
            .iter()
            .any(|existing| distance(*existing, point) <= dedup_tolerance)
        {
            points.push(point);
        }
    }
    Ok(points)
}

/// Jacobi diagonalization of a real symmetric 3×3 matrix. Returns diagonal
/// eigenvalues and eigenvectors as columns.
fn symmetric_eigen(mut matrix: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let mut pair = (0, 1);
        let mut largest = matrix[0][1].abs();
        for (p, q) in [(0, 2), (1, 2)] {
            if matrix[p][q].abs() > largest {
                pair = (p, q);
                largest = matrix[p][q].abs();
            }
        }
        if largest < 1e-15 {
            break;
        }
        let (p, q) = pair;
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sin, cos) = angle.sin_cos();

        let app =
            cos * cos * matrix[p][p] - 2.0 * sin * cos * matrix[p][q] + sin * sin * matrix[q][q];
        let aqq =
            sin * sin * matrix[p][p] + 2.0 * sin * cos * matrix[p][q] + cos * cos * matrix[q][q];
        for r in [0, 1, 2] {
            if r == p || r == q {
                continue;
            }
            let arp = cos * matrix[r][p] - sin * matrix[r][q];
            let arq = sin * matrix[r][p] + cos * matrix[r][q];
            matrix[r][p] = arp;
            matrix[p][r] = arp;
            matrix[r][q] = arq;
            matrix[q][r] = arq;
        }
        matrix[p][p] = app;
        matrix[q][q] = aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for row in &mut vectors {
            let vrp = cos * row[p] - sin * row[q];
            let vrq = sin * row[p] + cos * row[q];
            row[p] = vrp;
            row[q] = vrq;
        }
    }
    ([matrix[0][0], matrix[1][1], matrix[2][2]], vectors)
}

fn least_squares_plane(points: &[Vec3]) -> Option<Plane> {
    if points.len() < 3 {
        return None;
    }
    let centroid = mul(
        points.iter().copied().fold([0.0; 3], add),
        1.0 / points.len() as f64,
    );
    let mut covariance = [[0.0; 3]; 3];
    for &point in points {
        let relative = sub(point, centroid);
        for row in 0..3 {
            for column in 0..3 {
                covariance[row][column] += relative[row] * relative[column];
            }
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(covariance);
    let smallest = (0..3).min_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(Ordering::Equal)
    })?;
    let normal = normalized([
        eigenvectors[0][smallest],
        eigenvectors[1][smallest],
        eigenvectors[2][smallest],
    ])?;
    Some(canonical_plane(normal, dot(normal, centroid)))
}

fn independent_of(normal: Vec3, selected: &[Plane]) -> bool {
    let mut orthonormal: Vec<Vec3> = Vec::with_capacity(selected.len());
    for plane in selected {
        let mut direction = plane.normal;
        for &existing in &orthonormal {
            direction = sub(direction, mul(existing, dot(direction, existing)));
        }
        if let Some(direction) = normalized(direction) {
            orthonormal.push(direction);
        }
    }
    let mut residual = normal;
    for direction in orthonormal {
        residual = sub(residual, mul(direction, dot(residual, direction)));
    }
    norm(residual) > INDEPENDENCE_EPSILON
}

/// Plane candidates stay bounded even when a highly detailed model crosses
/// every stencil edge. Small point sets are exhaustive. Larger sets use each
/// point with pairs among its nearest neighbors, which favors actual local
/// surface patches and caps the 6×6×6 stencil at 15,120 triples rather than
/// the roughly 26 million possible triples.
fn candidate_triples(points: &[Vec3]) -> Vec<[usize; 3]> {
    if points.len() <= EXHAUSTIVE_CANDIDATE_LIMIT {
        let mut triples = Vec::new();
        for i in 0..points.len().saturating_sub(2) {
            for j in i + 1..points.len().saturating_sub(1) {
                for k in j + 1..points.len() {
                    triples.push([i, j, k]);
                }
            }
        }
        return triples;
    }

    let mut triples = BTreeSet::new();
    for (anchor, &point) in points.iter().enumerate() {
        let mut neighbors: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != anchor)
            .map(|(index, &other)| (index, dot(sub(other, point), sub(other, point))))
            .collect();
        neighbors.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        neighbors.truncate(LOCAL_CANDIDATE_NEIGHBORS);
        for left in 0..neighbors.len().saturating_sub(1) {
            for right in left + 1..neighbors.len() {
                let mut triple = [anchor, neighbors[left].0, neighbors[right].0];
                triple.sort_unstable();
                triples.insert(triple);
            }
        }
    }
    triples.into_iter().collect()
}

/// Deterministic RANSAC-style planar segmentation. Candidate planes come from
/// the bounded triple set above. Each selected plane maximizes newly covered
/// inliers, then total inliers, then minimizes squared residual.
fn fit_constraint_planes(
    points: &[Vec3],
    count: usize,
    threshold: f64,
) -> Result<Vec<Plane>, String> {
    if points.len() < 3 * count {
        return Err(format!(
            "only {} distinct boundary points were found; fitting {} surface constraints needs at least {}",
            points.len(),
            count,
            3 * count
        ));
    }
    let mut selected: Vec<Plane> = Vec::with_capacity(count);
    let mut covered = vec![false; points.len()];
    let candidates = candidate_triples(points);

    for constraint in 0..count {
        let mut best: Option<(usize, usize, f64, Plane, Vec<usize>)> = None;
        for &[i, j, k] in &candidates {
            let Some(normal) =
                normalized(cross(sub(points[j], points[i]), sub(points[k], points[i])))
            else {
                continue;
            };
            if !independent_of(normal, &selected) {
                continue;
            }
            let candidate = canonical_plane(normal, dot(normal, points[i]));
            let mut inliers = Vec::new();
            let mut squared_error = 0.0;
            for (index, &point) in points.iter().enumerate() {
                let residual = (dot(candidate.normal, point) - candidate.offset).abs();
                if residual <= threshold {
                    inliers.push(index);
                    squared_error += residual * residual;
                }
            }
            if inliers.len() < 3 {
                continue;
            }
            let newly_covered = inliers.iter().filter(|&&index| !covered[index]).count();
            let score = (newly_covered, inliers.len(), squared_error);
            let replace = best.as_ref().is_none_or(|current| {
                score.0 > current.0
                    || (score.0 == current.0 && score.1 > current.1)
                    || (score.0 == current.0 && score.1 == current.1 && score.2 < current.2)
            });
            if replace {
                best = Some((score.0, score.1, score.2, candidate, inliers));
            }
        }
        let Some((_, _, _, candidate, mut inliers)) = best else {
            return Err(format!(
                "could not find independent local surface constraint {} of {count}",
                constraint + 1
            ));
        };
        let mut plane = least_squares_plane(
            &inliers
                .iter()
                .map(|&index| points[index])
                .collect::<Vec<_>>(),
        )
        .unwrap_or(candidate);
        if !independent_of(plane.normal, &selected) {
            plane = candidate;
        }
        inliers = points
            .iter()
            .enumerate()
            .filter_map(|(index, &point)| {
                ((dot(plane.normal, point) - plane.offset).abs() <= threshold).then_some(index)
            })
            .collect();
        if let Some(refined) = least_squares_plane(
            &inliers
                .iter()
                .map(|&index| points[index])
                .collect::<Vec<_>>(),
        ) && independent_of(refined.normal, &selected)
        {
            plane = refined;
        }
        for index in inliers {
            covered[index] = true;
        }
        selected.push(plane);
    }
    Ok(selected)
}

fn planes_to_subspace(planes: &[Plane], seed: Vec3, rank: usize) -> Result<Subspace, String> {
    let mut constraints: Vec<Vec3> = Vec::with_capacity(planes.len());
    let mut targets: Vec<f64> = Vec::with_capacity(planes.len());
    for plane in planes {
        let mut residual = plane.normal;
        let mut implied = 0.0;
        for (&constraint, &target) in constraints.iter().zip(&targets) {
            let coefficient = dot(plane.normal, constraint);
            residual = sub(residual, mul(constraint, coefficient));
            implied += coefficient * target;
        }
        let length = norm(residual);
        if length <= INDEPENDENCE_EPSILON {
            return Err("fitted surface constraints are not independent".to_string());
        }
        constraints.push(mul(residual, 1.0 / length));
        targets.push((plane.offset - implied) / length);
    }

    let mut origin = seed;
    for (&constraint, &target) in constraints.iter().zip(&targets) {
        origin = sub(origin, mul(constraint, dot(constraint, origin) - target));
    }

    let mut basis: Vec<Vec3> = Vec::with_capacity(rank);
    for axis in 0..3 {
        let mut candidate = [0.0; 3];
        candidate[axis] = 1.0;
        for &constraint in &constraints {
            candidate = sub(candidate, mul(constraint, dot(candidate, constraint)));
        }
        for &existing in &basis {
            candidate = sub(candidate, mul(existing, dot(candidate, existing)));
        }
        if let Some(unit) = normalized(candidate) {
            basis.push(unit);
            if basis.len() == rank {
                break;
            }
        }
    }
    if basis.len() != rank {
        return Err(format!(
            "fitted constraints left {} tangent directions, expected {rank}",
            basis.len()
        ));
    }
    let subspace = Subspace {
        dimensions: 3,
        origin: origin.to_vec(),
        basis: basis.into_iter().flatten().collect(),
    };
    subspace.validate()?;
    Ok(subspace)
}

fn projector(subspace: &Subspace) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for basis in (0..subspace.rank()).map(|index| subspace.basis_vector(index)) {
        for row in 0..3 {
            for column in 0..3 {
                result[row][column] += basis[row] * basis[column];
            }
        }
    }
    result
}

/// Positional disagreement of two affine subspaces over a neighborhood of
/// `radius`: offset mismatch plus tangent-projector mismatch scaled to world
/// distance. It is invariant to basis sign and order.
fn subspace_delta(a: &Subspace, b: &Subspace, radius: f64) -> f64 {
    let a_origin: Vec3 = a.origin.as_slice().try_into().unwrap();
    let b_origin: Vec3 = b.origin.as_slice().try_into().unwrap();
    let offset = a
        .project(&b.origin)
        .1
        .max(b.project(&a.origin).1)
        .max(distance(a_origin, b_origin).min(radius));
    let pa = projector(a);
    let pb = projector(b);
    let projector_error = (0..3)
        .flat_map(|row| (0..3).map(move |column| (row, column)))
        .map(|(row, column)| {
            let difference = pa[row][column] - pb[row][column];
            difference * difference
        })
        .sum::<f64>()
        .sqrt();
    offset.max(radius * projector_error)
}

fn axis_combinations(rank: usize) -> &'static [&'static [usize]] {
    match rank {
        0 => &[&[]],
        1 => &[&[0], &[1], &[2]],
        2 => &[&[0, 1], &[0, 2], &[1, 2]],
        _ => &[],
    }
}

fn metric_snap(subspace: &Subspace, bounds: &[f64], divisions: u32, tolerance: f64) -> Subspace {
    if divisions == 0 {
        return subspace.clone();
    }
    let fitted_projector = projector(subspace);
    let Some((axes, projector_error)) = axis_combinations(subspace.rank())
        .iter()
        .map(|axes| {
            let mut candidate = [[0.0; 3]; 3];
            for &axis in *axes {
                candidate[axis][axis] = 1.0;
            }
            let error = (0..3)
                .flat_map(|row| (0..3).map(move |column| (row, column)))
                .map(|(row, column)| {
                    let difference = fitted_projector[row][column] - candidate[row][column];
                    difference * difference
                })
                .sum::<f64>()
                .sqrt();
            (*axes, error)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
    else {
        return subspace.clone();
    };
    if projector_error > AXIS_SNAP_PROJECTOR_TOLERANCE {
        return subspace.clone();
    }

    let mut origin = [0.0; 3];
    for axis in 0..3 {
        let (lo, hi) = (bounds[2 * axis], bounds[2 * axis + 1]);
        if axes.contains(&axis) {
            origin[axis] = (lo + hi) * 0.5;
            continue;
        }
        let extent = hi - lo;
        let fraction = (subspace.origin[axis] - lo) / extent;
        let numerator = (fraction * divisions as f64).round();
        let snapped = lo + extent * (numerator / divisions as f64);
        if (snapped - subspace.origin[axis]).abs() > tolerance {
            return subspace.clone();
        }
        origin[axis] = snapped;
    }
    Subspace::axis_aligned(origin.to_vec(), axes).unwrap_or_else(|_| subspace.clone())
}

fn fit_subspace<F>(
    seed: Vec3,
    bounds: &[f64],
    config: &FitConfig,
    mut sample: F,
) -> Result<Subspace, String>
where
    F: FnMut(&[Vec3]) -> Result<Vec<bool>, String>,
{
    let initial_radius = validate_config(config, bounds)?;
    let mut radius = initial_radius;
    let mut center = seed;
    let mut previous: Option<Subspace> = None;
    let mut last_delta = f64::INFINITY;

    for _iteration in 0..config.max_iterations {
        // Early, large neighborhoods only need relative precision. The
        // absolute target takes over as the stencil shrinks toward the final
        // tolerance, avoiding dozens of needless model probes per iteration.
        let boundary_tolerance = (config.tolerance * 0.25).max(radius * 0.01);
        let points = boundary_points(center, radius, boundary_tolerance, &mut sample)?;
        let plane_threshold = (radius * 0.05).max(config.tolerance * 2.0);
        let planes = fit_constraint_planes(&points, config.kind.codimension(), plane_threshold)?;
        let fitted = planes_to_subspace(&planes, center, config.kind.rank())?;

        if let Some(previous) = &previous {
            last_delta = subspace_delta(previous, &fitted, radius);
            if last_delta <= config.tolerance {
                return Ok(metric_snap(
                    &fitted,
                    bounds,
                    config.snap_divisions,
                    config.tolerance,
                ));
            }
        }
        center = fitted.origin.as_slice().try_into().unwrap();
        previous = Some(fitted);
        radius *= 0.5;
        if radius < config.tolerance {
            radius = config.tolerance;
        }
    }
    Err(format!(
        "fit did not converge within {} iterations (last subspace delta {last_delta}, tolerance {})",
        config.max_iterations, config.tolerance
    ))
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let dimensions = match input_model_dimensions(0) {
        Some(3) => 3,
        Some(other) => {
            report_error(&format!(
                "subspace fit currently requires a 3D model, input has {other} dimensions"
            ));
            return;
        }
        None => {
            report_error("input 0 is not a usable model");
            return;
        }
    };
    let bounds = match input_model_bounds(0, dimensions) {
        Some(bounds) => bounds,
        None => {
            report_error("failed to read model bounds");
            return;
        }
    };
    let seed = match decode_vec3(&read_input(1)) {
        Ok(seed) => seed,
        Err(error) => {
            report_error(&format!("invalid seed: {error}"));
            return;
        }
    };
    let config: FitConfig = {
        let bytes = read_input(2);
        if bytes.is_empty() {
            FitConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(bytes)) {
                Ok(config) => config,
                Err(error) => {
                    report_error(&format!("invalid configuration: {error}"));
                    return;
                }
            }
        }
    };

    let result = fit_subspace(seed, &bounds, &config, |points| {
        if cancelled() {
            return Err("fit cancelled".to_string());
        }
        let positions: Vec<f64> = points.iter().flatten().copied().collect();
        input_model_sample(0, &positions, 3)
            .ok_or_else(|| "model sampling failed".to_string())
            .map(|values| values.into_iter().map(is_occupied).collect())
    });
    match result {
        Ok(subspace) => post_output(0, &encode_subspace(&subspace)),
        Err(error) => report_error(&format!("subspace fit failed: {error}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ kind: "point" / "line" / "plane" .default "plane", tolerance: float .default 1e-5, search_radius: float .default 0.0, max_iterations: int .ge 2 .le 32 .default 20, snap_divisions: int .ge 0 .le 1000000 .default 0 }"#.to_string();
        OperatorMetadata {
            name: "subspace_fit_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Fit Subspace".to_string(),
            description: "Fit a local point, edge line, or tangent plane from a seed using bounded model probes.".to_string(),
            category: "Construction".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 17 9 7h12l-6 10Z"/>"##,
                r##"<circle cx="12" cy="12" r="2"/>"##,
                r##"<path d="M12 3v5M9.5 5.5 12 3l2.5 2.5"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::VecF64(3),
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec![
                "Model".to_string(),
                "Seed".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::Subspace],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDS: [f64; 6] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

    fn box_sampler(points: &[Vec3]) -> Result<Vec<bool>, String> {
        Ok(points
            .iter()
            .map(|point| point.iter().all(|coordinate| coordinate.abs() <= 1.0))
            .collect())
    }

    fn config(kind: FeatureKind) -> FitConfig {
        FitConfig {
            kind,
            tolerance: 1e-6,
            search_radius: 0.2,
            max_iterations: 12,
            snap_divisions: 0,
        }
    }

    fn projector_close(subspace: &Subspace, diagonal: [f64; 3], tolerance: f64) -> bool {
        let p = projector(subspace);
        (0..3).all(|row| {
            (0..3).all(|column| {
                let expected = if row == column { diagonal[row] } else { 0.0 };
                (p[row][column] - expected).abs() <= tolerance
            })
        })
    }

    #[test]
    fn fits_box_face_edge_and_corner_from_nearby_seeds() {
        let plane = fit_subspace(
            [1.002, 0.2, -0.1],
            &BOUNDS,
            &config(FeatureKind::Plane),
            box_sampler,
        )
        .unwrap();
        assert_eq!(plane.rank(), 2);
        assert!((plane.origin[0] - 1.0).abs() < 2e-6);
        assert!(projector_close(&plane, [0.0, 1.0, 1.0], 1e-5));

        let line = fit_subspace(
            [1.002, 1.001, 0.1],
            &BOUNDS,
            &config(FeatureKind::Line),
            box_sampler,
        )
        .unwrap();
        assert_eq!(line.rank(), 1);
        assert!((line.origin[0] - 1.0).abs() < 2e-6);
        assert!((line.origin[1] - 1.0).abs() < 2e-6);
        assert!(projector_close(&line, [0.0, 0.0, 1.0], 1e-5));

        let point = fit_subspace(
            [1.002, 1.001, 1.003],
            &BOUNDS,
            &config(FeatureKind::Point),
            box_sampler,
        )
        .unwrap();
        assert_eq!(point.rank(), 0);
        for coordinate in &point.origin {
            assert!((*coordinate - 1.0).abs() < 2e-6, "{point:?}");
        }
    }

    #[test]
    fn fits_an_oblique_plane_without_axis_snapping() {
        let normal = normalized([1.0, 2.0, -0.75]).unwrap();
        let offset = 0.2;
        let sampler = |points: &[Vec3]| {
            Ok(points
                .iter()
                .map(|&point| dot(normal, point) <= offset)
                .collect())
        };
        let mut cfg = config(FeatureKind::Plane);
        cfg.snap_divisions = 4;
        let plane = fit_subspace(mul(normal, offset + 0.001), &BOUNDS, &cfg, sampler).unwrap();
        let fitted_normal: Vec3 = plane.normal().unwrap().try_into().unwrap();
        let orientation = dot(fitted_normal, normal);
        assert!(orientation.abs() > 0.9999, "{plane:?}");
        assert!(
            (dot(
                mul(fitted_normal, orientation.signum()),
                plane.origin.as_slice().try_into().unwrap()
            ) - offset)
                .abs()
                < 2e-6,
            "{plane:?}"
        );
        assert!(projector(&plane)[0][1].abs() > 0.01);
    }

    #[test]
    fn shrinking_neighborhood_converges_to_curved_surface_tangent() {
        let sphere_sampler = |points: &[Vec3]| {
            Ok(points
                .iter()
                .map(|&point| dot(point, point) <= 1.0)
                .collect())
        };
        let mut cfg = config(FeatureKind::Plane);
        cfg.tolerance = 1e-5;
        cfg.max_iterations = 16;
        let tangent = fit_subspace([1.002, 0.0, 0.0], &BOUNDS, &cfg, sphere_sampler).unwrap();

        assert!((tangent.origin[0] - 1.0).abs() < 2e-5, "{tangent:?}");
        assert!(projector_close(&tangent, [0.0, 1.0, 1.0], 2e-4));
    }

    #[test]
    fn metric_snap_uses_exact_bounds_fractions() {
        let surface = 0.5000004;
        let bounds = [0.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let sampler =
            |points: &[Vec3]| Ok(points.iter().map(|point| point[0] <= surface).collect());
        let cfg = FitConfig {
            kind: FeatureKind::Plane,
            tolerance: 1e-5,
            search_radius: 0.1,
            max_iterations: 12,
            snap_divisions: 2,
        };
        let plane = fit_subspace([surface + 0.001, 0.2, -0.3], &bounds, &cfg, sampler).unwrap();
        assert_eq!(plane.origin, vec![0.5, 0.0, 0.0]);
        assert_eq!(plane.basis, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn reports_when_the_probe_neighborhood_has_no_boundary() {
        let error =
            fit_subspace([0.0; 3], &BOUNDS, &config(FeatureKind::Plane), box_sampler).unwrap_err();
        assert!(error.contains("no occupancy boundary"), "{error}");
    }
}
