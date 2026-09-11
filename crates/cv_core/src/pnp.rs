//! Pose of a picture from the markers it shows and the map's marker
//! corners: a planar start from one marker's homography, then a robust
//! Gauss–Newton over every corner, with the focal (and a first radial
//! term) optionally among the unknowns.
//!
//! Poses are camera-to-world in the view set's convention (OpenCV axes).
//! Residuals go through `CameraModel::project`, so a camera that carries
//! distortion is honoured in the fit.

use volumetric_abi::viewset::{CameraModel, Distortion, Marker};

use crate::detect::{Detection, homography};
use crate::linalg::{
    Mat3, add, cross, dot, inverse, mat3_mul, mat3_transpose, mat3_vec, nearest_rotation, norm,
    normalized, rotation_from_vector, scale, solve, sub,
};

/// A world point seen at a pixel, tagged by the marker it belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correspondence {
    pub world: [f64; 3],
    pub pixel: [f64; 2],
    pub marker: u32,
}

/// Solver choices.
#[derive(Clone, Debug)]
pub struct SolveOptions {
    /// Estimate a single focal length (fx = fy) alongside the pose.
    pub solve_focal: bool,
    /// Estimate the first radial distortion term alongside the pose.
    pub solve_distortion: bool,
    /// Huber transition, pixels.
    pub huber_px: f64,
    /// Corners farther than this from their prediction after the first
    /// fit are dropped and the fit repeated.
    pub max_residual_px: f64,
    pub iterations: usize,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            solve_focal: false,
            solve_distortion: false,
            huber_px: 2.0,
            max_residual_px: 12.0,
            iterations: 40,
        }
    }
}

/// One marker's part in the fit.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkerFit {
    pub id: u32,
    pub rms_px: f64,
    /// Corners of it kept after the outlier cut.
    pub corners_used: usize,
}

/// A fitted parameter with its standard error from the normal matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub value: f64,
    pub std: f64,
}

/// The result of a pose solve.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseSolve {
    pub camera_to_world: [f64; 12],
    /// The camera the residuals were measured with: the input, or with the
    /// solved focal and distortion.
    pub camera: CameraModel,
    pub rms_px: f64,
    pub markers: Vec<MarkerFit>,
    pub corners_used: usize,
    pub focal: Option<Estimate>,
    pub k1: Option<Estimate>,
}

impl PoseSolve {
    pub fn position(&self) -> [f64; 3] {
        let m = &self.camera_to_world;
        [m[3], m[7], m[11]]
    }
}

/// Pairs each detection's corners with the map's corners of the same
/// marker; detections of markers not in the map are ignored.
pub fn correspondences(markers: &[Marker], detections: &[Detection]) -> Vec<Correspondence> {
    let mut out = Vec::new();
    for d in detections {
        if let Some(m) = markers.iter().find(|m| m.id == d.id) {
            for j in 0..4 {
                out.push(Correspondence {
                    world: m.corners[j],
                    pixel: d.corners[j],
                    marker: d.id,
                });
            }
        }
    }
    out
}

/// Solves the view's pose from its detections against the map.
pub fn solve_view(
    camera: &CameraModel,
    markers: &[Marker],
    detections: &[Detection],
    options: &SolveOptions,
) -> Result<PoseSolve, String> {
    solve_pose(camera, &correspondences(markers, detections), options)
}

/// World-to-camera rigid pose.
#[derive(Clone, Copy, Debug)]
struct Rigid {
    r: Mat3,
    t: [f64; 3],
}

impl Rigid {
    fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        add(mat3_vec(&self.r, p), self.t)
    }

    fn camera_to_world(&self) -> [f64; 12] {
        let rt = mat3_transpose(&self.r);
        let c = scale(mat3_vec(&rt, self.t), -1.0);
        [
            rt[0][0], rt[0][1], rt[0][2], c[0], //
            rt[1][0], rt[1][1], rt[1][2], c[1], //
            rt[2][0], rt[2][1], rt[2][2], c[2], //
        ]
    }
}

/// The pose of a planar marker from the homography between its plane
/// coordinates and the (undistorted, normalised) pixels of its corners.
fn planar_pose(camera: &CameraModel, corners: &[Correspondence]) -> Option<Rigid> {
    if corners.len() < 4 {
        return None;
    }
    // A frame on the points' plane: origin at corner 0, u towards the
    // farthest point, v towards the point farthest off that line, w the
    // normal. (A marker's corners 0, 1, 3 would do; the card's first
    // three corners are collinear.)
    let origin = corners[0].world;
    let far = corners
        .iter()
        .max_by(|a, b| norm(sub(a.world, origin)).total_cmp(&norm(sub(b.world, origin))))?
        .world;
    if norm(sub(far, origin)) < 1e-12 {
        return None;
    }
    let u = normalized(sub(far, origin));
    let off_line = |p: [f64; 3]| {
        let d = sub(p, origin);
        norm(sub(d, scale(u, dot(d, u))))
    };
    let side = corners
        .iter()
        .max_by(|a, b| off_line(a.world).total_cmp(&off_line(b.world)))?
        .world;
    if off_line(side) < 1e-12 {
        return None;
    }
    let v0 = sub(side, origin);
    let v = normalized(sub(v0, scale(u, dot(v0, u))));
    let w = cross(u, v);
    let plane: Vec<[f64; 2]> = corners
        .iter()
        .map(|c| {
            let d = sub(c.world, origin);
            [dot(d, u), dot(d, v)]
        })
        .collect();
    let normalised: Vec<[f64; 2]> = corners
        .iter()
        .map(|c| {
            let ray = camera.ray(c.pixel);
            [ray[0] / ray[2], ray[1] / ray[2]]
        })
        .collect();
    let h = homography(&plane, &normalised)?;
    let h1 = [h[0][0], h[1][0], h[2][0]];
    let h2 = [h[0][1], h[1][1], h[2][1]];
    let h3 = [h[0][2], h[1][2], h[2][2]];
    let lambda = 2.0 / (norm(h1) + norm(h2));
    let (mut r1, mut r2, mut t) = (scale(h1, lambda), scale(h2, lambda), scale(h3, lambda));
    // The marker must be in front of the camera.
    if t[2] < 0.0 {
        r1 = scale(r1, -1.0);
        r2 = scale(r2, -1.0);
        t = scale(t, -1.0);
    }
    let r3 = cross(r1, r2);
    let r_plane = nearest_rotation(&[
        [r1[0], r2[0], r3[0]],
        [r1[1], r2[1], r3[1]],
        [r1[2], r2[2], r3[2]],
    ]);
    // World -> plane frame -> camera.
    let basis = [[u[0], v[0], w[0]], [u[1], v[1], w[1]], [u[2], v[2], w[2]]];
    let r = mat3_mul(&r_plane, &mat3_transpose(&basis));
    let t = sub(t, mat3_vec(&r, origin));
    Some(Rigid { r, t })
}

/// The camera with a focal and first radial term substituted.
fn camera_with(camera: &CameraModel, focal: Option<f64>, k1: Option<f64>) -> CameraModel {
    let mut cam = camera.clone();
    if let Some(f) = focal {
        cam.fx = f;
        cam.fy = f;
    }
    if let Some(k1) = k1 {
        cam.distortion = match &camera.distortion {
            Distortion::Radial { k, p } => {
                let mut k = k.clone();
                if k.is_empty() {
                    k.push(k1);
                } else {
                    k[0] = k1;
                }
                Distortion::Radial { k, p: *p }
            }
            _ => Distortion::Radial {
                k: vec![k1],
                p: [0.0, 0.0],
            },
        };
    }
    cam
}

fn first_k1(camera: &CameraModel) -> f64 {
    match &camera.distortion {
        Distortion::Radial { k, .. } => k.first().copied().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Residuals (predicted minus observed, two per point) of a parameter
/// vector: rotation update, translation, and optionally focal and k1.
fn residuals(
    camera: &CameraModel,
    base: &Rigid,
    params: &[f64],
    layout: &Layout,
    points: &[Correspondence],
) -> Vec<f64> {
    let pose = Rigid {
        r: mat3_mul(
            &rotation_from_vector([params[0], params[1], params[2]]),
            &base.r,
        ),
        t: [params[3], params[4], params[5]],
    };
    let cam = camera_with(
        camera,
        layout.focal.map(|i| params[i]),
        layout.k1.map(|i| params[i]),
    );
    let mut out = Vec::with_capacity(points.len() * 2);
    for p in points {
        match cam.project(pose.apply(p.world)) {
            Some(px) => {
                out.push(px[0] - p.pixel[0]);
                out.push(px[1] - p.pixel[1]);
            }
            None => {
                out.push(1e6);
                out.push(1e6);
            }
        }
    }
    out
}

/// Where each unknown sits in the parameter vector.
#[derive(Clone, Copy)]
struct Layout {
    count: usize,
    focal: Option<usize>,
    k1: Option<usize>,
}

/// Robust Gauss–Newton (Levenberg damped) from `start`; returns the pose,
/// the fitted camera, the final parameter vector and the residuals.
fn refine(
    camera: &CameraModel,
    start: Rigid,
    points: &[Correspondence],
    options: &SolveOptions,
) -> Result<(Rigid, Vec<f64>, Layout, Vec<f64>), String> {
    let mut layout = Layout {
        count: 6,
        focal: None,
        k1: None,
    };
    if options.solve_focal {
        layout.focal = Some(layout.count);
        layout.count += 1;
    }
    if options.solve_distortion {
        layout.k1 = Some(layout.count);
        layout.count += 1;
    }
    if points.len() * 2 < layout.count + 2 {
        return Err(format!(
            "{} corners cannot fix {} unknowns",
            points.len(),
            layout.count
        ));
    }
    let mut base = start;
    let mut params = vec![0.0; layout.count];
    params[3..6].copy_from_slice(&base.t);
    if let Some(i) = layout.focal {
        params[i] = (camera.fx + camera.fy) * 0.5;
    }
    if let Some(i) = layout.k1 {
        params[i] = first_k1(camera);
    }
    let mut lambda = 1e-3;
    let mut current = residuals(camera, &base, &params, &layout, points);
    let mut cost = huber_cost(&current, options.huber_px);
    for _ in 0..options.iterations {
        let jac = jacobian(camera, &base, &params, &layout, points, &current);
        let weights: Vec<f64> = current
            .chunks(2)
            .flat_map(|r| {
                let m = (r[0] * r[0] + r[1] * r[1]).sqrt();
                let w = if m <= options.huber_px {
                    1.0
                } else {
                    options.huber_px / m
                };
                [w, w]
            })
            .collect();
        let n = layout.count;
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtr = vec![0.0; n];
        for (row, (res, w)) in jac.iter().zip(current.iter().zip(&weights)) {
            for i in 0..n {
                jtr[i] += w * row[i] * res;
                for j in 0..n {
                    jtj[i][j] += w * row[i] * row[j];
                }
            }
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut damped = jtj.clone();
            for i in 0..n {
                damped[i][i] += lambda * (1.0 + jtj[i][i]);
            }
            let rhs: Vec<f64> = jtr.iter().map(|v| -v).collect();
            let Some(delta) = solve(&damped, &rhs) else {
                lambda *= 10.0;
                continue;
            };
            let trial: Vec<f64> = params.iter().zip(&delta).map(|(p, d)| p + d).collect();
            let trial_res = residuals(camera, &base, &trial, &layout, points);
            let trial_cost = huber_cost(&trial_res, options.huber_px);
            if trial_cost < cost {
                // Fold the rotation update into the base so the rotation
                // parameters stay small.
                base.r = mat3_mul(
                    &rotation_from_vector([trial[0], trial[1], trial[2]]),
                    &base.r,
                );
                base.r = nearest_rotation(&base.r);
                params = trial;
                params[0] = 0.0;
                params[1] = 0.0;
                params[2] = 0.0;
                current = residuals(camera, &base, &params, &layout, points);
                let step = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
                cost = huber_cost(&current, options.huber_px);
                lambda = (lambda * 0.3).max(1e-9);
                improved = true;
                if step < 1e-10 {
                    break;
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    base.t = [params[3], params[4], params[5]];
    Ok((base, params, layout, current))
}

fn huber_cost(residuals: &[f64], huber: f64) -> f64 {
    residuals
        .chunks(2)
        .map(|r| {
            let m = (r[0] * r[0] + r[1] * r[1]).sqrt();
            if m <= huber {
                0.5 * m * m
            } else {
                huber * (m - 0.5 * huber)
            }
        })
        .sum()
}

/// Numeric Jacobian of the residuals (rows) with respect to the
/// parameters (columns).
fn jacobian(
    camera: &CameraModel,
    base: &Rigid,
    params: &[f64],
    layout: &Layout,
    points: &[Correspondence],
    current: &[f64],
) -> Vec<Vec<f64>> {
    let n = layout.count;
    let mut columns = Vec::with_capacity(n);
    for i in 0..n {
        let eps = match i {
            0..=2 => 1e-6,
            3..=5 => 1e-5,
            _ if layout.focal == Some(i) => 1e-2,
            _ => 1e-5,
        };
        let mut p = params.to_vec();
        p[i] += eps;
        let plus = residuals(camera, base, &p, layout, points);
        columns.push(
            plus.iter()
                .zip(current)
                .map(|(a, b)| (a - b) / eps)
                .collect::<Vec<f64>>(),
        );
    }
    (0..current.len())
        .map(|row| columns.iter().map(|c| c[row]).collect())
        .collect()
}

/// Solves the pose from correspondences.
pub fn solve_pose(
    camera: &CameraModel,
    points: &[Correspondence],
    options: &SolveOptions,
) -> Result<PoseSolve, String> {
    if points.len() < 4 {
        return Err(format!(
            "{} corners matched the map; at least one whole marker is needed",
            points.len()
        ));
    }
    // Initial pose: the single marker whose planar pose predicts every
    // corner best.
    let mut ids: Vec<u32> = points.iter().map(|p| p.marker).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut best: Option<(f64, Rigid)> = None;
    for id in &ids {
        let own: Vec<Correspondence> = points.iter().copied().filter(|p| p.marker == *id).collect();
        let Some(pose) = planar_pose(camera, &own) else {
            continue;
        };
        let err: f64 = points
            .iter()
            .map(|p| match camera.project(pose.apply(p.world)) {
                Some(px) => ((px[0] - p.pixel[0]).powi(2) + (px[1] - p.pixel[1]).powi(2)).sqrt(),
                None => 1e6,
            })
            .sum();
        if best.is_none_or(|(e, _)| err < e) {
            best = Some((err, pose));
        }
    }
    let (_, start) = best.ok_or("no marker gave a starting pose")?;

    let (pose, params, layout, res) = refine(camera, start, points, options)?;
    // Outlier cut, then one more fit on what remains.
    let kept: Vec<Correspondence> = points
        .iter()
        .zip(res.chunks(2))
        .filter(|(_, r)| (r[0] * r[0] + r[1] * r[1]).sqrt() <= options.max_residual_px)
        .map(|(p, _)| *p)
        .collect();
    let (pose, params, layout, res) = if kept.len() < points.len() && kept.len() >= 4 {
        let camera_now = camera_with(
            camera,
            layout.focal.map(|i| params[i]),
            layout.k1.map(|i| params[i]),
        );
        let (pose, params, layout, res) = refine(&camera_now, pose, &kept, options)?;
        (pose, params, layout, res)
    } else {
        (pose, params, layout, res)
    };
    let used: &[Correspondence] = if kept.len() < points.len() && kept.len() >= 4 {
        &kept
    } else {
        points
    };
    let fitted = camera_with(
        camera,
        layout.focal.map(|i| params[i]),
        layout.k1.map(|i| params[i]),
    );

    let rms_px = (res.iter().map(|r| r * r).sum::<f64>() / (res.len() as f64 / 2.0)).sqrt();
    let mut markers = Vec::new();
    for id in &ids {
        let mut sq = 0.0;
        let mut count = 0usize;
        for (p, r) in used.iter().zip(res.chunks(2)) {
            if p.marker == *id {
                sq += r[0] * r[0] + r[1] * r[1];
                count += 1;
            }
        }
        markers.push(MarkerFit {
            id: *id,
            rms_px: if count > 0 {
                (sq / count as f64).sqrt()
            } else {
                f64::NAN
            },
            corners_used: count,
        });
    }

    // Standard errors from the normal matrix at the solution.
    let (focal, k1) = if layout.focal.is_some() || layout.k1.is_some() {
        let jac = jacobian(&fitted, &pose, &params, &layout, used, &res);
        let n = layout.count;
        let mut jtj = vec![vec![0.0; n]; n];
        for row in &jac {
            for i in 0..n {
                for j in 0..n {
                    jtj[i][j] += row[i] * row[j];
                }
            }
        }
        let dof = (res.len() as f64 - n as f64).max(1.0);
        let sigma_sq = res.iter().map(|r| r * r).sum::<f64>() / dof;
        let cov = inverse(&jtj);
        let estimate = |i: Option<usize>| {
            i.map(|i| Estimate {
                value: params[i],
                std: cov
                    .as_ref()
                    .map_or(f64::INFINITY, |c| (c[i][i].max(0.0) * sigma_sq).sqrt()),
            })
        };
        (estimate(layout.focal), estimate(layout.k1))
    } else {
        (None, None)
    };

    Ok(PoseSolve {
        camera_to_world: pose.camera_to_world(),
        camera: fitted,
        rms_px,
        markers,
        corners_used: used.len(),
        focal,
        k1,
    })
}

/// A camera-to-world pose's rotation as a matrix.
pub fn rotation_of(camera_to_world: &[f64; 12]) -> Mat3 {
    let m = camera_to_world;
    [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]]
}

/// The angle between two poses' rotations, radians, and the distance
/// between their positions.
pub fn pose_difference(a: &[f64; 12], b: &[f64; 12]) -> (f64, f64) {
    let ra = rotation_of(a);
    let rb = rotation_of(b);
    let rel = mat3_mul(&ra, &mat3_transpose(&rb));
    let angle = crate::linalg::rotation_angle(&rel);
    let da = [a[3] - b[3], a[7] - b[7], a[11] - b[11]];
    (angle, norm(da))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::square_marker;
    use volumetric_abi::viewset::View;

    fn floor_markers() -> Vec<Marker> {
        let right = [1.0, 0.0, 0.0];
        let down = [0.0, -1.0, 0.0];
        vec![
            square_marker(0, [-0.45, 0.85, 0.0], 0.12, right, down),
            square_marker(1, [0.3, 0.9, 0.0], 0.12, right, down),
            square_marker(2, [-0.35, 0.45, 0.0], 0.12, right, down),
            square_marker(5, [0.25, 0.4, 0.0], 0.12, right, down),
            square_marker(49, [-0.05, 0.65, 0.0], 0.16, right, down),
        ]
    }

    fn truth_view(pitch_deg: f64) -> View {
        let pitch = pitch_deg.to_radians();
        let (s, c) = pitch.sin_cos();
        let forward = [0.0, c, -s];
        let down = [0.0, -s, -c];
        View::posed(
            "cam",
            0,
            [
                1.0, down[0], forward[0], 0.1, //
                0.0, down[1], forward[1], -0.4, //
                0.0, down[2], forward[2], 1.2, //
            ],
        )
    }

    fn detections(
        camera: &CameraModel,
        view: &View,
        markers: &[Marker],
        noise: f64,
    ) -> Vec<Detection> {
        let mut seed = 12345u64;
        let mut jitter = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0 * noise
        };
        markers
            .iter()
            .map(|m| Detection {
                id: m.id,
                family: "5x5_100",
                corners: m.corners.map(|c| {
                    let p = view.project(camera, c).unwrap();
                    [p[0] + jitter(), p[1] + jitter()]
                }),
                rotation: 0,
                distance: 0,
                fit_px: 0.0,
            })
            .collect()
    }

    #[test]
    fn pose_is_recovered_from_marker_corners() {
        let camera = CameraModel::pinhole(1280, 960, 1000.0, 1000.0, 640.0, 480.0);
        let view = truth_view(50.0);
        let markers = floor_markers();
        let found = detections(&camera, &view, &markers, 0.0);
        let solve = solve_view(&camera, &markers, &found, &SolveOptions::default()).unwrap();
        let (angle, dist) = pose_difference(&solve.camera_to_world, view.pose().unwrap());
        assert!(
            angle < 1e-6 && dist < 1e-6,
            "{angle} rad, {dist} m, rms {}",
            solve.rms_px
        );
        assert!(solve.rms_px < 1e-6);
        assert_eq!(solve.corners_used, 20);

        // With 0.3 px of corner noise the pose holds to millimetres.
        let noisy = detections(&camera, &view, &markers, 0.3);
        let solve = solve_view(&camera, &markers, &noisy, &SolveOptions::default()).unwrap();
        let (angle, dist) = pose_difference(&solve.camera_to_world, view.pose().unwrap());
        assert!(angle < 0.002 && dist < 0.003, "{angle} rad, {dist} m");
        assert!(solve.rms_px < 0.5, "{}", solve.rms_px);
        assert_eq!(solve.markers.len(), 5);

        // A single marker still solves, from its own homography.
        let one = solve_view(
            &camera,
            &markers[..1],
            &found[..1],
            &SolveOptions::default(),
        )
        .unwrap();
        let (angle, dist) = pose_difference(&one.camera_to_world, view.pose().unwrap());
        assert!(angle < 1e-3 && dist < 2e-3, "{angle} rad, {dist} m");

        // A wild corner is cut and the pose survives.
        let mut spoiled = found.clone();
        spoiled[2].corners[1][0] += 40.0;
        let solve = solve_view(&camera, &markers, &spoiled, &SolveOptions::default()).unwrap();
        let (angle, dist) = pose_difference(&solve.camera_to_world, view.pose().unwrap());
        assert!(angle < 1e-4 && dist < 1e-4, "{angle} rad, {dist} m");
        assert_eq!(solve.corners_used, 19);
        assert_eq!(
            solve
                .markers
                .iter()
                .find(|m| m.id == 2)
                .unwrap()
                .corners_used,
            3
        );

        assert!(solve_view(&camera, &markers, &[], &SolveOptions::default()).is_err());
    }

    #[test]
    fn focal_is_solved_when_the_view_constrains_it() {
        let truth = CameraModel::pinhole(1280, 960, 1000.0, 1000.0, 640.0, 480.0);
        let markers = floor_markers();
        let options = SolveOptions {
            solve_focal: true,
            ..SolveOptions::default()
        };
        // Seeded 20% off, an oblique view of the floor fixes the focal.
        let view = truth_view(50.0);
        let found = detections(&truth, &view, &markers, 0.2);
        let seed = CameraModel::pinhole(1280, 960, 1200.0, 1200.0, 640.0, 480.0);
        let solve = solve_view(&seed, &markers, &found, &options).unwrap();
        let focal = solve.focal.unwrap();
        assert!((focal.value - 1000.0).abs() < 8.0, "{focal:?}");
        assert!(focal.std < 10.0, "{focal:?}");
        assert!((solve.camera.fx - focal.value).abs() < 1e-9);
        let (angle, dist) = pose_difference(&solve.camera_to_world, view.pose().unwrap());
        assert!(angle < 0.01 && dist < 0.02, "{angle} rad, {dist} m");

        // Square-on to the floor the focal trades against the height: the
        // standard error says so.
        let flat = View::posed(
            "top",
            0,
            [
                1.0, 0.0, 0.0, -0.05, //
                0.0, -1.0, 0.0, 0.65, //
                0.0, 0.0, -1.0, 1.2, //
            ],
        );
        let found = detections(&truth, &flat, &markers, 0.2);
        let solve = solve_view(&seed, &markers, &found, &options).unwrap();
        let weak = solve.focal.unwrap();
        assert!(
            weak.std > 5.0 * focal.std,
            "oblique {focal:?} vs flat {weak:?}"
        );
    }

    #[test]
    fn distortion_is_honoured_and_solvable() {
        let mut truth = CameraModel::pinhole(1280, 960, 1000.0, 1000.0, 640.0, 480.0);
        truth.distortion = Distortion::Radial {
            k: vec![0.08],
            p: [0.0, 0.0],
        };
        let markers = floor_markers();
        let view = truth_view(50.0);
        let found = detections(&truth, &view, &markers, 0.0);
        // Solving with the true camera is exact.
        let solve = solve_view(&truth, &markers, &found, &SolveOptions::default()).unwrap();
        assert!(solve.rms_px < 1e-6, "{}", solve.rms_px);
        // Ignoring the distortion leaves a residual; solving k1 removes it.
        let plain = CameraModel::pinhole(1280, 960, 1000.0, 1000.0, 640.0, 480.0);
        let ignored = solve_view(&plain, &markers, &found, &SolveOptions::default()).unwrap();
        assert!(ignored.rms_px > 0.5, "{}", ignored.rms_px);
        let solved = solve_view(
            &plain,
            &markers,
            &found,
            &SolveOptions {
                solve_distortion: true,
                ..SolveOptions::default()
            },
        )
        .unwrap();
        let k1 = solved.k1.unwrap();
        assert!(
            (k1.value - 0.08).abs() < 0.01,
            "{k1:?} rms {}",
            solved.rms_px
        );
        assert!(solved.rms_px < 0.05, "{}", solved.rms_px);
    }
}
