//! Cloud Fit Operator.
//!
//! Fits a geometric feature to a point cloud — the nodes of a FeaMesh, so
//! a Point1 cloud from Point Cloud Import or any other mesh — and hands it
//! on as a [`Subspace`]: a point, a line, a plane, a sphere's centre or a
//! cylinder's axis, ready for Slice, Extrude, Revolve, Span and Intersect.
//! RANSAC finds the feature most points agree with (within `tolerance`),
//! least squares refines it on those inliers, and an F64Map reports the
//! numbers the Subspace cannot carry: radius, extents, inlier count, rms.
//! See README.md (the operator's docs) for the conventions.
//!
//! Inputs:
//! - Input 0: FeaMesh — the cloud (node positions; a `normal` node field
//!   lets a cylinder be fitted without a seed line).
//! - Input 1: Subspace, optional — a seed: a point narrows the search to
//!   its neighbourhood; a line or plane is the estimate to refine.
//! - Input 2: CBOR configuration, see [`CloudFitConfig`].
//!
//! Outputs: 0 Subspace (the feature), 1 F64Map (the fit).

use cloud_core::{
    Rng, Vec3, add, bounds, canonical, centroid, cross, dot, eigen_symmetric, mul, norm,
    normalized, perpendicular, scatter, solve, sub,
};
use volumetric_abi::f64_map::F64Map;
#[cfg(target_arch = "wasm32")]
use volumetric_abi::f64_map::encode as encode_f64_map;
#[cfg(target_arch = "wasm32")]
use volumetric_abi::fea::decode_fea_mesh;
use volumetric_abi::fea::{FeaMesh, NORMAL_FIELD_NAME};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
use volumetric_abi::subspace::Subspace;
#[cfg(target_arch = "wasm32")]
use volumetric_abi::subspace::{decode_subspace, encode_subspace};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Points scored per RANSAC hypothesis; larger clouds are strided down
/// to this many for the search, then every point counts for the refine.
const SCORE_LIMIT: usize = 20_000;
/// Chance RANSAC is allowed to miss the best hypothesis.
const MISS_PROBABILITY: f64 = 1e-3;
const REFINE_ROUNDS: usize = 3;
const LM_ITERATIONS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Point,
    Line,
    Plane,
    Sphere,
    Cylinder,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Point => "point",
            Kind::Line => "line",
            Kind::Plane => "plane",
            Kind::Sphere => "sphere",
            Kind::Cylinder => "cylinder",
        }
    }

    /// Points a RANSAC hypothesis needs.
    fn sample_size(self) -> usize {
        match self {
            Kind::Point => 1,
            Kind::Line => 2,
            Kind::Plane => 3,
            Kind::Sphere => 4,
            Kind::Cylinder => 3,
        }
    }
}

/// Which way a fitted plane faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Facing {
    /// Away from the whole cloud's centroid: outward for a scanned
    /// object, away from the object for the table it stands on.
    Outward,
    /// Towards the cloud's centroid.
    Inward,
    /// Towards +y.
    Up,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct CloudFitConfig {
    pub kind: Kind,
    /// Which way a plane's normal points.
    pub normal: Facing,
    /// Distance within which a point counts as on the feature; 0 takes
    /// 0.5% of the considered points' bounding diagonal.
    pub tolerance: f64,
    /// With a seed, only points within this distance of it are
    /// considered; 0 means 10% of the cloud's bounding diagonal for a
    /// point seed and no limit for a line or plane seed.
    pub search_radius: f64,
    /// Upper bound on RANSAC hypotheses.
    pub trials: u32,
}

impl Default for CloudFitConfig {
    fn default() -> Self {
        Self {
            kind: Kind::Plane,
            normal: Facing::Outward,
            tolerance: 0.0,
            search_radius: 0.0,
            trials: 500,
        }
    }
}

/// Levenberg-Marquardt over `N` parameters with one residual per point,
/// numeric partials accumulated point by point (no Jacobian is stored).
fn levenberg_marquardt<const N: usize>(
    params: &mut [f64; N],
    points: &[Vec3],
    residual: impl Fn(&[f64; N], Vec3) -> f64,
) {
    let cost = |p: &[f64; N]| points.iter().map(|&x| residual(p, x).powi(2)).sum::<f64>();
    let mut lambda = 1e-3;
    let mut current = cost(params);
    for _ in 0..LM_ITERATIONS {
        let mut jtj = vec![vec![0.0; N]; N];
        let mut jtr = vec![0.0; N];
        let steps: [f64; N] = std::array::from_fn(|i| 1e-7 * params[i].abs().max(1e-3));
        for &x in points {
            let r = residual(params, x);
            let mut partials = [0.0; N];
            for i in 0..N {
                let mut plus = *params;
                plus[i] += steps[i];
                let mut minus = *params;
                minus[i] -= steps[i];
                partials[i] = (residual(&plus, x) - residual(&minus, x)) / (2.0 * steps[i]);
            }
            for i in 0..N {
                jtr[i] += partials[i] * r;
                for j in 0..N {
                    jtj[i][j] += partials[i] * partials[j];
                }
            }
        }
        let mut improved = false;
        for _ in 0..6 {
            let mut damped = jtj.clone();
            for (i, row) in damped.iter_mut().enumerate() {
                row[i] += lambda * row[i].max(1e-12);
            }
            let Some(delta) = solve(damped, jtr.iter().map(|v| -v).collect()) else {
                lambda *= 10.0;
                continue;
            };
            let mut candidate = *params;
            for i in 0..N {
                candidate[i] += delta[i];
            }
            let next = cost(&candidate);
            if next < current {
                *params = candidate;
                current = next;
                lambda = (lambda / 3.0).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
}

/// A fitted feature; `distance` is how far a point is off it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Feature {
    Point {
        center: Vec3,
    },
    Line {
        origin: Vec3,
        direction: Vec3,
    },
    Plane {
        origin: Vec3,
        normal: Vec3,
    },
    Sphere {
        center: Vec3,
        radius: f64,
    },
    Cylinder {
        origin: Vec3,
        direction: Vec3,
        radius: f64,
    },
}

impl Feature {
    pub fn distance(&self, p: Vec3) -> f64 {
        match *self {
            Feature::Point { center } => norm(sub(p, center)),
            Feature::Line { origin, direction } => norm(cross(sub(p, origin), direction)),
            Feature::Plane { origin, normal } => dot(sub(p, origin), normal).abs(),
            Feature::Sphere { center, radius } => (norm(sub(p, center)) - radius).abs(),
            Feature::Cylinder {
                origin,
                direction,
                radius,
            } => (norm(cross(sub(p, origin), direction)) - radius).abs(),
        }
    }

    fn radius(&self) -> Option<f64> {
        match *self {
            Feature::Sphere { radius, .. } | Feature::Cylinder { radius, .. } => Some(radius),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Hypotheses from minimal samples
// ---------------------------------------------------------------------------

fn plane_through(p: &[Vec3]) -> Option<Feature> {
    let normal = normalized(cross(sub(p[1], p[0]), sub(p[2], p[0])))?;
    Some(Feature::Plane {
        origin: p[0],
        normal,
    })
}

fn line_through(p: &[Vec3]) -> Option<Feature> {
    let direction = normalized(sub(p[1], p[0]))?;
    Some(Feature::Line {
        origin: p[0],
        direction,
    })
}

fn sphere_through(p: &[Vec3]) -> Option<Feature> {
    let rows: Vec<Vec<f64>> = (1..4).map(|i| mul(sub(p[i], p[0]), 2.0).to_vec()).collect();
    let rhs: Vec<f64> = (1..4).map(|i| dot(p[i], p[i]) - dot(p[0], p[0])).collect();
    let c = solve(rows, rhs)?;
    let center = [c[0], c[1], c[2]];
    let radius = norm(sub(p[0], center));
    (radius.is_finite() && radius > 0.0).then_some(Feature::Sphere { center, radius })
}

/// A cylinder from two points with normals: the axis runs along
/// `n1 x n2`, through the meeting point of the two normal lines seen
/// along it.
fn cylinder_through_normals(p: &[Vec3], n: &[Vec3]) -> Option<Feature> {
    let raw_axis = cross(n[0], n[1]);
    if norm(raw_axis) < 0.05 {
        return None;
    }
    let direction = normalized(raw_axis)?;
    let flatten = |v: Vec3| sub(v, mul(direction, dot(v, direction)));
    let (q1, q2) = (flatten(p[0]), flatten(p[1]));
    let m1 = normalized(flatten(n[0]))?;
    let m2 = normalized(flatten(n[1]))?;
    let denominator = dot(cross(m1, m2), direction);
    if denominator.abs() < 1e-6 {
        return None;
    }
    let s = dot(cross(sub(q2, q1), m2), direction) / denominator;
    let origin = add(q1, mul(m1, s));
    let radius = (norm(sub(q1, origin)) + norm(sub(q2, origin))) / 2.0;
    (radius.is_finite() && radius > 0.0).then_some(Feature::Cylinder {
        origin,
        direction,
        radius,
    })
}

/// A cylinder of known axis direction from three points: the circle
/// through them seen along the axis.
fn cylinder_through_along(p: &[Vec3], direction: Vec3) -> Option<Feature> {
    let e1 = perpendicular(direction);
    let e2 = cross(direction, e1);
    let flat: Vec<[f64; 2]> = p.iter().map(|&q| [dot(q, e1), dot(q, e2)]).collect();
    let (a, b, c) = (flat[0], flat[1], flat[2]);
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-12 {
        return None;
    }
    let sq = |v: [f64; 2]| v[0] * v[0] + v[1] * v[1];
    let ux = (sq(a) * (b[1] - c[1]) + sq(b) * (c[1] - a[1]) + sq(c) * (a[1] - b[1])) / d;
    let uy = (sq(a) * (c[0] - b[0]) + sq(b) * (a[0] - c[0]) + sq(c) * (b[0] - a[0])) / d;
    let radius = ((a[0] - ux).powi(2) + (a[1] - uy).powi(2)).sqrt();
    (radius.is_finite() && radius > 0.0).then_some(Feature::Cylinder {
        origin: add(mul(e1, ux), mul(e2, uy)),
        direction,
        radius,
    })
}

// ---------------------------------------------------------------------------
// Least-squares refinement on inliers
// ---------------------------------------------------------------------------

fn refine(feature: Feature, inliers: &[Vec3]) -> Feature {
    match feature {
        Feature::Point { .. } => Feature::Point {
            center: centroid(inliers),
        },
        Feature::Line { direction, .. } => {
            let (origin, m) = scatter(inliers);
            let (_, vectors) = eigen_symmetric(m);
            let best = vectors[2];
            // Keep the orientation the hypothesis had.
            let direction = if dot(best, direction) < 0.0 {
                mul(best, -1.0)
            } else {
                best
            };
            Feature::Line { origin, direction }
        }
        Feature::Plane { normal, .. } => {
            let (origin, m) = scatter(inliers);
            let (_, vectors) = eigen_symmetric(m);
            let best = vectors[0];
            let normal = if dot(best, normal) < 0.0 {
                mul(best, -1.0)
            } else {
                best
            };
            Feature::Plane { origin, normal }
        }
        Feature::Sphere { center, radius } => {
            let mut params = [center[0], center[1], center[2], radius];
            levenberg_marquardt(&mut params, inliers, |q, p| {
                norm(sub(p, [q[0], q[1], q[2]])) - q[3]
            });
            Feature::Sphere {
                center: [params[0], params[1], params[2]],
                radius: params[3].abs(),
            }
        }
        Feature::Cylinder {
            origin,
            direction,
            radius,
        } => {
            // Parametrise around the current axis: the origin moves in
            // the plane across it, the direction tilts by two angles.
            let e1 = perpendicular(direction);
            let e2 = cross(direction, e1);
            let mut params = [0.0, 0.0, 0.0, 0.0, radius];
            levenberg_marquardt(&mut params, inliers, |q, p| {
                let c = add(origin, add(mul(e1, q[0]), mul(e2, q[1])));
                let d = normalized(add(direction, add(mul(e1, q[2]), mul(e2, q[3]))))
                    .unwrap_or(direction);
                norm(cross(sub(p, c), d)) - q[4]
            });
            let origin = add(origin, add(mul(e1, params[0]), mul(e2, params[1])));
            let direction = normalized(add(direction, add(mul(e1, params[2]), mul(e2, params[3]))))
                .unwrap_or(direction);
            Feature::Cylinder {
                origin,
                direction,
                radius: params[4].abs(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The fit
// ---------------------------------------------------------------------------

/// The result of a fit: the feature (oriented and positioned as the
/// subspace is), its inliers' statistics, and the extents of the inliers
/// along the feature's own axes.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    pub feature: Feature,
    pub points: usize,
    pub inliers: usize,
    pub rms: f64,
    pub max: f64,
    pub tolerance: f64,
    /// Inlier extent along each basis vector of the output subspace.
    pub extents: Vec<f64>,
    /// The output subspace for the feature.
    pub subspace: Subspace,
}

/// Cloud data: positions and, when the mesh carries them, unit normals.
pub struct Cloud {
    pub points: Vec<Vec3>,
    pub normals: Option<Vec<Vec3>>,
}

impl Cloud {
    pub fn from_mesh(mesh: &FeaMesh) -> Self {
        let points = (0..mesh.node_count())
            .map(|i| mesh.node_position(i))
            .collect();
        let normals = mesh
            .node_fields
            .iter()
            .find(|f| f.name == NORMAL_FIELD_NAME && f.components == 3)
            .map(|f| {
                f.data
                    .chunks_exact(3)
                    .map(|n| normalized([n[0], n[1], n[2]]).unwrap_or([0.0; 3]))
                    .collect()
            });
        Self { points, normals }
    }
}

fn bounding_diagonal(points: &[Vec3]) -> f64 {
    let (lo, hi) = bounds(points);
    norm(sub(hi, lo))
}

/// What the seed asks for.
enum SeedRole {
    /// Consider only points near the seed; the feature is found by RANSAC.
    Near,
    /// Refine this feature (a line or plane the user already knows).
    Estimate(Feature),
    /// A cylinder along this direction (the axis's radius and position
    /// are found by RANSAC).
    Along(Vec3),
}

fn seed_role(seed: &Subspace, kind: Kind) -> Result<SeedRole, String> {
    if seed.ambient() != 3 {
        return Err(format!(
            "the seed lives in {}-space; the cloud is 3D",
            seed.ambient()
        ));
    }
    let origin = [seed.origin[0], seed.origin[1], seed.origin[2]];
    let axis = |i: usize| {
        let b = seed.basis_vector(i);
        [b[0], b[1], b[2]]
    };
    match (kind, seed.rank()) {
        (_, 0) => Ok(SeedRole::Near),
        (Kind::Line, 1) => Ok(SeedRole::Estimate(Feature::Line {
            origin,
            direction: axis(0),
        })),
        (Kind::Cylinder, 1) => Ok(SeedRole::Along(axis(0))),
        (Kind::Plane, 2) => Ok(SeedRole::Estimate(Feature::Plane {
            origin,
            normal: seed
                .normal()
                .map(|n| [n[0], n[1], n[2]])
                .expect("rank 2 in 3-space"),
        })),
        (kind, rank) => Err(format!(
            "a rank-{rank} seed does not fit a {}: give a point to search near, {}",
            kind.name(),
            match kind {
                Kind::Line => "or a line to refine",
                Kind::Plane => "or a plane to refine",
                Kind::Cylinder => "or a line for the axis direction",
                Kind::Point | Kind::Sphere => "which is the only seed a point or sphere takes",
            }
        )),
    }
}

fn ransac(
    kind: Kind,
    cloud_points: &[Vec3],
    cloud_normals: Option<&[Vec3]>,
    along: Option<Vec3>,
    tolerance: f64,
    trials: u32,
) -> Result<Feature, String> {
    let sample_size = if along.is_some() {
        3
    } else if kind == Kind::Cylinder {
        2
    } else {
        kind.sample_size()
    };
    let stride = cloud_points.len().div_ceil(SCORE_LIMIT).max(1);
    let scored: Vec<usize> = (0..cloud_points.len()).step_by(stride).collect();
    if scored.len() < sample_size {
        return Err(format!(
            "a {} needs at least {sample_size} points; the cloud has {}",
            kind.name(),
            cloud_points.len()
        ));
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (kind as u64 + 1));
    let mut best: Option<(Feature, usize)> = None;
    let mut needed = trials.max(1) as usize;
    let mut trial = 0usize;
    let mut sample = Vec::with_capacity(sample_size);
    while trial < needed {
        trial += 1;
        sample.clear();
        while sample.len() < sample_size {
            let candidate = scored[rng.below(scored.len())];
            if !sample.contains(&candidate) {
                sample.push(candidate);
            }
        }
        let p: Vec<Vec3> = sample.iter().map(|&i| cloud_points[i]).collect();
        let hypothesis = match (kind, along) {
            (Kind::Cylinder, Some(direction)) => cylinder_through_along(&p, direction),
            (Kind::Cylinder, None) => {
                let normals = cloud_normals.expect("checked before ransac");
                let n: Vec<Vec3> = sample.iter().map(|&i| normals[i]).collect();
                cylinder_through_normals(&p, &n)
            }
            (Kind::Plane, _) => plane_through(&p),
            (Kind::Line, _) => line_through(&p),
            (Kind::Sphere, _) => sphere_through(&p),
            (Kind::Point, _) => Some(Feature::Point { center: p[0] }),
        };
        let Some(hypothesis) = hypothesis else {
            continue;
        };
        let inliers = scored
            .iter()
            .filter(|&&i| hypothesis.distance(cloud_points[i]) <= tolerance)
            .count();
        if best.is_none_or(|(_, count)| inliers > count) {
            best = Some((hypothesis, inliers));
            let ratio = inliers as f64 / scored.len() as f64;
            let miss = 1.0 - ratio.powi(sample_size as i32);
            if miss > 0.0 && miss < 1.0 {
                let estimate = (MISS_PROBABILITY.ln() / miss.ln()).ceil();
                needed = needed.min(estimate.max(1.0) as usize).max(trial);
            }
        }
    }
    best.filter(|&(_, count)| count >= sample_size)
        .map(|(feature, _)| feature)
        .ok_or_else(|| {
            format!(
                "no {} fits {sample_size} or more points within {tolerance}",
                kind.name()
            )
        })
}

/// Fit `config.kind` to the cloud. See the module docs.
pub fn fit(cloud: &Cloud, seed: Option<&Subspace>, config: &CloudFitConfig) -> Result<Fit, String> {
    if !(config.tolerance.is_finite() && config.tolerance >= 0.0) {
        return Err(format!(
            "tolerance must be finite and non-negative, got {}",
            config.tolerance
        ));
    }
    if !(config.search_radius.is_finite() && config.search_radius >= 0.0) {
        return Err(format!(
            "search_radius must be finite and non-negative, got {}",
            config.search_radius
        ));
    }
    if cloud.points.is_empty() {
        return Err("the cloud has no points".to_string());
    }
    let role = seed.map(|s| seed_role(s, config.kind)).transpose()?;
    let diagonal = bounding_diagonal(&cloud.points);

    // The points considered: near the seed, or all of them.
    let considered: Vec<usize> = match (&role, seed) {
        (Some(SeedRole::Near), Some(seed)) if config.search_radius == 0.0 || true => {
            let radius = if config.search_radius > 0.0 {
                config.search_radius
            } else if matches!(role, Some(SeedRole::Near)) {
                0.1 * diagonal
            } else {
                f64::INFINITY
            };
            (0..cloud.points.len())
                .filter(|&i| seed.project(&cloud.points[i]).1 <= radius)
                .collect()
        }
        (Some(_), Some(seed)) if config.search_radius > 0.0 => (0..cloud.points.len())
            .filter(|&i| seed.project(&cloud.points[i]).1 <= config.search_radius)
            .collect(),
        _ => (0..cloud.points.len()).collect(),
    };
    let points: Vec<Vec3> = considered.iter().map(|&i| cloud.points[i]).collect();
    let normals: Option<Vec<Vec3>> = cloud
        .normals
        .as_ref()
        .map(|n| considered.iter().map(|&i| n[i]).collect());
    let minimum = config.kind.sample_size();
    if points.len() < minimum {
        return Err(format!(
            "a {} needs at least {minimum} points; {} lie within the search radius",
            config.kind.name(),
            points.len()
        ));
    }
    let tolerance = if config.tolerance > 0.0 {
        config.tolerance
    } else {
        0.005 * bounding_diagonal(&points)
    };
    if tolerance.is_nan() || tolerance <= 0.0 {
        return Err("the considered points coincide; nothing to fit".to_string());
    }

    let mut feature = match role {
        Some(SeedRole::Estimate(feature)) => feature,
        Some(SeedRole::Along(direction)) => ransac(
            config.kind,
            &points,
            normals.as_deref(),
            Some(direction),
            tolerance,
            config.trials,
        )?,
        _ if config.kind == Kind::Point => Feature::Point {
            center: centroid(&points),
        },
        _ => {
            if config.kind == Kind::Cylinder && normals.is_none() {
                return Err(
                    "a cylinder needs point normals (a `normal` node field, e.g. from Cloud \
                     Normals) or a seed line giving its axis direction"
                        .to_string(),
                );
            }
            ransac(
                config.kind,
                &points,
                normals.as_deref(),
                None,
                tolerance,
                config.trials,
            )?
        }
    };

    // Least squares on the inliers, re-selecting them as the feature moves.
    // A point is the centroid of everything considered: no inlier test.
    let within = |feature: &Feature| -> Vec<Vec3> {
        if config.kind == Kind::Point {
            points.clone()
        } else {
            points
                .iter()
                .copied()
                .filter(|&p| feature.distance(p) <= tolerance)
                .collect()
        }
    };
    for _ in 0..REFINE_ROUNDS {
        let inliers = within(&feature);
        if inliers.len() < minimum {
            return Err(format!(
                "only {} point(s) lie within {tolerance} of the {}; loosen `tolerance`",
                inliers.len(),
                config.kind.name()
            ));
        }
        feature = refine(feature, &inliers);
    }
    let inliers = within(&feature);
    if inliers.is_empty() {
        return Err("the refined feature lost every inlier".to_string());
    }
    let residuals: Vec<f64> = inliers.iter().map(|&p| feature.distance(p)).collect();
    let rms = (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
    let max = residuals.iter().copied().fold(0.0, f64::max);

    // The subspace: origin at the inliers' centroid projected onto the
    // feature; planes face as configured, lines take one canonical sign.
    let inlier_centroid = centroid(&inliers);
    let cloud_centroid = centroid(&cloud.points);
    let (feature, subspace, extents) = match feature {
        Feature::Point { .. } | Feature::Sphere { .. } => {
            let center = match feature {
                Feature::Point { center } | Feature::Sphere { center, .. } => center,
                _ => unreachable!(),
            };
            (feature, Subspace::point(center.to_vec()), vec![])
        }
        Feature::Line { origin, direction }
        | Feature::Cylinder {
            origin, direction, ..
        } => {
            let direction = canonical(direction);
            let t = dot(sub(inlier_centroid, origin), direction);
            let origin = add(origin, mul(direction, t));
            let feature = match feature {
                Feature::Cylinder { radius, .. } => Feature::Cylinder {
                    origin,
                    direction,
                    radius,
                },
                _ => Feature::Line { origin, direction },
            };
            let along: Vec<f64> = inliers
                .iter()
                .map(|&p| dot(sub(p, origin), direction))
                .collect();
            let extent = along.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - along.iter().copied().fold(f64::INFINITY, f64::min);
            (
                feature,
                Subspace {
                    dimensions: 3,
                    origin: origin.to_vec(),
                    basis: direction.to_vec(),
                },
                vec![extent],
            )
        }
        Feature::Plane { origin, normal } => {
            let towards = match config.normal {
                Facing::Outward => -dot(sub(cloud_centroid, origin), normal),
                Facing::Inward => dot(sub(cloud_centroid, origin), normal),
                Facing::Up => normal[1],
            };
            let normal = if towards > 1e-9 * (1.0 + diagonal) {
                normal
            } else if towards < -1e-9 * (1.0 + diagonal) {
                mul(normal, -1.0)
            } else {
                canonical(normal)
            };
            let origin = sub(
                inlier_centroid,
                mul(normal, dot(sub(inlier_centroid, origin), normal)),
            );
            let e1 = perpendicular(normal);
            let e2 = cross(normal, e1);
            let extent = |e: Vec3| {
                let along: Vec<f64> = inliers.iter().map(|&p| dot(sub(p, origin), e)).collect();
                along.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - along.iter().copied().fold(f64::INFINITY, f64::min)
            };
            (
                Feature::Plane { origin, normal },
                Subspace {
                    dimensions: 3,
                    origin: origin.to_vec(),
                    basis: [e1, e2].concat(),
                },
                vec![extent(e1), extent(e2)],
            )
        }
    };
    subspace.validate()?;
    Ok(Fit {
        feature,
        points: points.len(),
        inliers: inliers.len(),
        rms,
        max,
        tolerance,
        extents,
        subspace,
    })
}

/// The fit's numbers as an F64Map.
pub fn fit_map(fit: &Fit) -> F64Map {
    let mut map = F64Map::new();
    map.insert("points".to_string(), fit.points as f64);
    map.insert("inliers".to_string(), fit.inliers as f64);
    map.insert("rms".to_string(), fit.rms);
    map.insert("max".to_string(), fit.max);
    map.insert("tolerance".to_string(), fit.tolerance);
    if let Some(radius) = fit.feature.radius() {
        map.insert("radius".to_string(), radius);
    }
    for (i, extent) in fit.extents.iter().enumerate() {
        map.insert(format!("extent_{i}"), *extent);
    }
    map
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let mesh = match decode_fea_mesh(&read_input(0)) {
        Ok(mesh) => mesh,
        Err(e) => {
            report_error(&format!("input 0 is not a usable mesh: {e}"));
            return;
        }
    };
    let seed_bytes = read_input(1);
    let seed = if seed_bytes.is_empty() {
        None
    } else {
        match decode_subspace(&seed_bytes) {
            Ok(seed) => Some(seed),
            Err(e) => {
                report_error(&format!("input 1 is not a usable subspace: {e}"));
                return;
            }
        }
    };
    let config = {
        let cfg = read_input(2);
        if cfg.is_empty() {
            CloudFitConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&cfg)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    let cloud = Cloud::from_mesh(&mesh);
    match fit(&cloud, seed.as_ref(), &config) {
        Ok(fit) => {
            if fit.inliers * 10 < fit.points {
                post_warning(&format!(
                    "only {} of {} points fit the {} within {:.3e}",
                    fit.inliers,
                    fit.points,
                    config.kind.name(),
                    fit.tolerance
                ));
            }
            post_output(0, &encode_subspace(&fit.subspace));
            match encode_f64_map(&fit_map(&fit)) {
                Ok(bytes) => post_output(1, &bytes),
                Err(e) => report_error(&format!("fit statistics failed to encode: {e}")),
            }
        }
        Err(e) => report_error(&format!("cloud fit failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ kind: "plane" / "line" / "point" / "sphere" / "cylinder" .default "plane", normal: "outward" / "inward" / "up" .default "outward", tolerance: float .ge 0.0 .default 0.0, search_radius: float .ge 0.0 .default 0.0, trials: int .ge 1 .default 500 }"#
            .to_string();
        OperatorMetadata {
            name: "cloud_fit_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Cloud Fit".to_string(),
            description: "Fit a plane, line, point, sphere or cylinder to a point cloud as a Subspace, with the fit's radius and residuals."
                .to_string(),
            category: "Construction".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 17 9 7h12l-6 10Z"/>"##,
                r##"<circle cx="7" cy="9" r="1"/>"##,
                r##"<circle cx="15" cy="10" r="1"/>"##,
                r##"<circle cx="11" cy="15" r="1"/>"##,
                r##"<circle cx="18" cy="15" r="1"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::Subspace,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec![
                "Cloud".to_string(),
                "Seed (optional)".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![
                OperatorMetadataOutput::Subspace,
                OperatorMetadataOutput::F64Map,
            ],
            output_names: vec!["Feature".to_string(), "Fit".to_string()],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gaussian-ish noise from the deterministic generator.
    fn noise(rng: &mut Rng, sigma: f64) -> f64 {
        let sum: f64 = (0..12)
            .map(|_| rng.next_u64() as f64 / u64::MAX as f64)
            .sum();
        (sum - 6.0) * sigma
    }

    fn uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (rng.next_u64() as f64 / u64::MAX as f64)
    }

    fn cloud(points: Vec<Vec3>, normals: Option<Vec<Vec3>>) -> Cloud {
        Cloud { points, normals }
    }

    /// Outliers filling the box [-1, 1]^3.
    fn outliers(rng: &mut Rng, count: usize) -> Vec<Vec3> {
        (0..count)
            .map(|_| {
                [
                    uniform(rng, -1.0, 1.0),
                    uniform(rng, -1.0, 1.0),
                    uniform(rng, -1.0, 1.0),
                ]
            })
            .collect()
    }

    fn angle_degrees(a: Vec3, b: Vec3) -> f64 {
        dot(a, b).abs().min(1.0).acos().to_degrees()
    }

    fn config(kind: Kind, tolerance: f64) -> CloudFitConfig {
        CloudFitConfig {
            kind,
            tolerance,
            ..CloudFitConfig::default()
        }
    }

    #[test]
    fn plane_survives_noise_and_outliers() {
        let mut rng = Rng(7);
        let normal = normalized([1.0, 2.0, 3.0]).unwrap();
        let (e1, e2) = {
            let e1 = perpendicular(normal);
            (e1, cross(normal, e1))
        };
        let mut points: Vec<Vec3> = (0..2000)
            .map(|_| {
                let u = uniform(&mut rng, -0.5, 0.5);
                let v = uniform(&mut rng, -0.5, 0.5);
                let off = 0.2 + noise(&mut rng, 1e-3);
                add(add(mul(e1, u), mul(e2, v)), mul(normal, off))
            })
            .collect();
        points.extend(outliers(&mut rng, 800));
        let fit = super::fit(&cloud(points, None), None, &config(Kind::Plane, 5e-3)).unwrap();
        let Feature::Plane {
            origin,
            normal: fitted,
        } = fit.feature
        else {
            panic!("{:?}", fit.feature)
        };
        assert!(angle_degrees(fitted, normal) < 0.1);
        assert!((dot(origin, normal) - 0.2).abs() < 5e-4);
        assert!(fit.inliers >= 1950 && fit.inliers < 2100, "{}", fit.inliers);
        assert!(fit.rms < 1.5e-3, "{}", fit.rms);
        assert_eq!(fit.subspace.rank(), 2);
        // The plane faces away from the cloud's centroid (the outliers'
        // box centre at the origin, below the plane).
        let n = fit.subspace.normal().unwrap();
        assert!(dot([n[0], n[1], n[2]], normal) > 0.99);
        // The square is 1 x 1; the odd outlier inside the slab can widen it.
        assert!(
            fit.extents.iter().all(|e| (0.99..2.0).contains(e)),
            "{:?}",
            fit.extents
        );
        let map = fit_map(&fit);
        assert!(map.contains_key("rms") && !map.contains_key("radius"));
    }

    #[test]
    fn a_point_seed_picks_the_nearby_plane() {
        let mut rng = Rng(11);
        let mut points: Vec<Vec3> = Vec::new();
        // A big plane at z = 0 and a small one at z = 0.5 above x, y in [0.3, 0.5].
        for _ in 0..3000 {
            points.push([
                uniform(&mut rng, -1.0, 1.0),
                uniform(&mut rng, -1.0, 1.0),
                noise(&mut rng, 1e-4),
            ]);
        }
        for _ in 0..300 {
            points.push([
                uniform(&mut rng, 0.3, 0.5),
                uniform(&mut rng, 0.3, 0.5),
                0.5 + noise(&mut rng, 1e-4),
            ]);
        }
        let unseeded = super::fit(
            &cloud(points.clone(), None),
            None,
            &config(Kind::Plane, 1e-3),
        )
        .unwrap();
        assert!(unseeded.subspace.origin[2].abs() < 1e-3);

        let seed = Subspace::point(vec![0.4, 0.4, 0.45]);
        let near = CloudFitConfig {
            search_radius: 0.15,
            ..config(Kind::Plane, 1e-3)
        };
        let seeded = super::fit(&cloud(points.clone(), None), Some(&seed), &near).unwrap();
        assert!((seeded.subspace.origin[2] - 0.5).abs() < 1e-3);
        assert_eq!(seeded.inliers, 300);

        // A plane seed refines from where it is: the big plane, even
        // though the seed is a little off.
        let tilted = Subspace {
            dimensions: 3,
            origin: vec![0.0, 0.0, 0.0003],
            basis: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };
        let refined = super::fit(
            &cloud(points, None),
            Some(&tilted),
            &config(Kind::Plane, 1e-3),
        )
        .unwrap();
        assert!(refined.subspace.origin[2].abs() < 1e-4);
        assert_eq!(refined.inliers, 3000);
    }

    #[test]
    fn line_and_point_fits() {
        let mut rng = Rng(3);
        let direction = normalized([0.0, 1.0, 1.0]).unwrap();
        let mut points: Vec<Vec3> = (0..500)
            .map(|_| {
                let t = uniform(&mut rng, -0.7, 0.7);
                add(
                    add([0.1, 0.2, 0.3], mul(direction, t)),
                    [
                        noise(&mut rng, 1e-3),
                        noise(&mut rng, 1e-3),
                        noise(&mut rng, 1e-3),
                    ],
                )
            })
            .collect();
        points.extend(outliers(&mut rng, 300));
        let fit = super::fit(
            &cloud(points.clone(), None),
            None,
            &config(Kind::Line, 5e-3),
        )
        .unwrap();
        assert_eq!(fit.subspace.rank(), 1);
        let d = fit.subspace.basis_vector(0);
        assert!(angle_degrees([d[0], d[1], d[2]], direction) < 0.2);
        assert!(fit.subspace.project(&[0.1, 0.2, 0.3]).1 < 1e-3);
        assert!((1.39..2.0).contains(&fit.extents[0]), "{}", fit.extents[0]);
        assert!(d[1] > 0.0, "canonical sign");

        let center = super::fit(&cloud(points, None), None, &config(Kind::Point, 0.0)).unwrap();
        assert_eq!(center.subspace.rank(), 0);
        assert_eq!(center.inliers, 800);
    }

    #[test]
    fn sphere_centre_and_radius() {
        let mut rng = Rng(5);
        let center = [0.2, -0.1, 0.3];
        let mut points: Vec<Vec3> = (0..1500)
            .map(|_| {
                let d = normalized([
                    noise(&mut rng, 1.0),
                    noise(&mut rng, 1.0),
                    noise(&mut rng, 1.0),
                ])
                .unwrap();
                add(center, mul(d, 0.4 + noise(&mut rng, 1e-3)))
            })
            .collect();
        points.extend(outliers(&mut rng, 500));
        let fit = super::fit(&cloud(points, None), None, &config(Kind::Sphere, 5e-3)).unwrap();
        let Feature::Sphere { center: c, radius } = fit.feature else {
            panic!("{:?}", fit.feature)
        };
        assert!(norm(sub(c, center)) < 3e-4, "{c:?}");
        assert!((radius - 0.4).abs() < 3e-4, "{radius}");
        assert_eq!(fit.subspace.rank(), 0);
        assert_eq!(fit_map(&fit)["radius"], radius);
    }

    /// Points on a cylinder of radius 0.3 about the axis through `origin`
    /// along `direction`, with exact normals when asked.
    fn cylinder_cloud(rng: &mut Rng, origin: Vec3, direction: Vec3, normals: bool) -> Cloud {
        let e1 = perpendicular(direction);
        let e2 = cross(direction, e1);
        let mut points = Vec::new();
        let mut ns = Vec::new();
        for _ in 0..2000 {
            let angle = uniform(rng, 0.0, std::f64::consts::TAU);
            let n = add(mul(e1, angle.cos()), mul(e2, angle.sin()));
            let t = uniform(rng, -0.5, 0.5);
            points.push(add(
                add(origin, mul(direction, t)),
                mul(n, 0.3 + noise(rng, 1e-3)),
            ));
            ns.push(n);
        }
        points.extend(outliers(rng, 600));
        ns.extend((0..600).map(|_| [1.0, 0.0, 0.0]));
        cloud(points, normals.then_some(ns))
    }

    #[test]
    fn cylinder_from_normals_and_from_a_seed_axis() {
        let mut rng = Rng(13);
        let direction = normalized([1.0, 0.2, -0.3]).unwrap();
        let origin = [0.05, 0.1, -0.05];
        let with_normals = cylinder_cloud(&mut rng, origin, direction, true);
        let fit = super::fit(&with_normals, None, &config(Kind::Cylinder, 5e-3)).unwrap();
        let Feature::Cylinder {
            direction: d,
            radius,
            ..
        } = fit.feature
        else {
            panic!("{:?}", fit.feature)
        };
        assert!(angle_degrees(d, direction) < 0.2, "{d:?}");
        assert!((radius - 0.3).abs() < 5e-4, "{radius}");
        assert!(fit.subspace.project(&origin).1 < 1e-3);
        assert!((0.99..2.6).contains(&fit.extents[0]), "{}", fit.extents[0]);

        let bare = cylinder_cloud(&mut rng, origin, direction, false);
        let err = super::fit(&bare, None, &config(Kind::Cylinder, 5e-3)).unwrap_err();
        assert!(err.contains("Cloud Normals"), "{err}");

        // A seed line a few degrees off still gives the axis.
        let rough = normalized(add(direction, [0.0, 0.05, 0.03])).unwrap();
        let seed = Subspace {
            dimensions: 3,
            origin: vec![0.0; 3],
            basis: rough.to_vec(),
        };
        let seeded = super::fit(&bare, Some(&seed), &config(Kind::Cylinder, 5e-3)).unwrap();
        let Feature::Cylinder {
            direction: d,
            radius,
            ..
        } = seeded.feature
        else {
            panic!("{:?}", seeded.feature)
        };
        assert!(angle_degrees(d, direction) < 0.2, "{d:?}");
        assert!((radius - 0.3).abs() < 5e-4, "{radius}");
    }

    #[test]
    fn bad_inputs_are_named() {
        let empty = cloud(vec![], None);
        assert!(
            super::fit(&empty, None, &CloudFitConfig::default())
                .unwrap_err()
                .contains("no points")
        );
        let few = cloud(vec![[0.0; 3], [1.0, 0.0, 0.0]], None);
        let err = super::fit(&few, None, &config(Kind::Plane, 1e-3)).unwrap_err();
        assert!(err.contains("at least 3"), "{err}");
        let err = super::fit(&few, None, &config(Kind::Line, -1.0)).unwrap_err();
        assert!(err.contains("tolerance"), "{err}");
        let seed = Subspace::axis_aligned(vec![0.0; 3], &[0, 1]).unwrap();
        let err = super::fit(&few, Some(&seed), &config(Kind::Sphere, 1e-3)).unwrap_err();
        assert!(err.contains("rank-2 seed"), "{err}");
    }
}
