//! The survey bundle: from the card corners and swatch corners detected
//! in a session's stills, one camera model per focus setting, a pose per
//! frame, and every card and swatch corner as a free point in the world.
//! The card's caliper pitch sets the scale and the card's plane and axes
//! define the world frame.
//!
//! The unknowns are, per camera, `f, cx, cy, k1, k2`; per frame a
//! rotation and a translation (world to camera); per point three
//! coordinates. Each observation gives two reprojection residuals under a
//! soft-L1 loss; one more residual holds the mean of the card's solved
//! column spans to the caliper span. Levenberg–Marquardt on the dense
//! normal equations with an analytic Jacobian; one card frame's pose is
//! held during the solve and the world re-based on the nominal card after.
//! Uncertainties come from the pseudo-inverse of the normal matrix with
//! the gauge freed.

use std::collections::BTreeMap;

use volumetric_abi::viewset::{Board, BoardCorner, CameraModel, Distortion, Marker, View, ViewSet};

use crate::linalg::{
    Mat3, add, cross, mat3_mul, mat3_transpose, mat3_vec, nearest_rotation, norm,
    rotation_from_vector, scale, smallest_eigenvector, sub,
};
use crate::pnp::{Correspondence, SolveOptions, solve_pose};

/// Solver choices.
#[derive(Clone, Debug)]
pub struct SurveyOptions {
    /// The family the swatches are printed in; other marker observations
    /// (the card's tags) are not points.
    pub swatch_family: String,
    /// Swatch ids at or above this are false decodes.
    pub max_swatch_id: u32,
    /// Card corners a frame needs to be posed on the card at the start.
    pub min_card_corners: usize,
    /// Solved points a frame needs to be posed on them.
    pub min_points: usize,
    /// A camera seen in fewer frames cannot self-calibrate: its frames are
    /// left out.
    pub min_frames_per_camera: usize,
    /// Card frames with at least this many corners calibrate their camera
    /// before the bundle; with fewer than `min_calibration_frames` of them
    /// the seed is used as is.
    pub calibration_corners: usize,
    pub min_calibration_frames: usize,
    /// The soft-L1 scale, pixels.
    pub soft_l1_px: f64,
    /// Frames the bundle leaves above this rms are dropped and the rest
    /// solved again.
    pub reject_px: f64,
    /// Single observations the bundle leaves beyond this are dropped
    /// first (a corner the detector put on the wrong edge), so one slip
    /// does not cost a frame.
    pub outlier_px: f64,
    /// A single-frame pose start over this rms is not used.
    pub pose_rms_px: f64,
    pub rounds: usize,
    /// Levenberg–Marquardt iterations per bundle.
    pub max_iterations: usize,
    /// Weight of the scale residual, per millimetre of span error.
    pub scale_weight: f64,
}

impl Default for SurveyOptions {
    fn default() -> Self {
        Self {
            swatch_family: "5x5_100".to_string(),
            max_swatch_id: 60,
            min_card_corners: 8,
            min_points: 8,
            min_frames_per_camera: 3,
            calibration_corners: 40,
            min_calibration_frames: 4,
            soft_l1_px: 1.5,
            reject_px: 3.0,
            outlier_px: 10.0,
            pose_rms_px: 3.0,
            rounds: 4,
            max_iterations: 100,
            scale_weight: 100.0,
        }
    }
}

/// One camera's solved model, for the report.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CameraReport {
    pub index: usize,
    pub label: String,
    pub frames: usize,
    pub f: f64,
    pub f_std: f64,
    pub cx: f64,
    pub cx_std: f64,
    pub cy: f64,
    pub cy_std: f64,
    pub k1: f64,
    pub k2: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct SwatchReport {
    pub id: u32,
    pub side_mm: f64,
    pub sigma_mm: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FrameReport {
    pub id: String,
    pub rms_px: f64,
    pub card_corners: usize,
    pub swatches: usize,
    /// Uncertainty of the camera's position, millimetres, relative to
    /// the card's frame.
    pub position_sigma_mm: f64,
    /// Uncertainty of the camera's orientation, degrees.
    pub angle_sigma_deg: f64,
}

/// What the survey did and how well it fits.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct SurveyReport {
    pub observations: usize,
    pub parameters: usize,
    pub rms_px: f64,
    pub median_px: f64,
    pub inlier_rms_px: f64,
    pub inlier_fraction: f64,
    /// The residual standard deviation the uncertainties are scaled by.
    pub sigma_px: f64,
    pub iterations: usize,
    pub card_corners_solved: usize,
    pub card_planarity_mm: f64,
    pub span_across_mm: f64,
    pub span_across_nominal_mm: f64,
    pub span_along_mm: f64,
    pub span_along_nominal_mm: f64,
    pub cameras: Vec<CameraReport>,
    pub swatches: Vec<SwatchReport>,
    pub frames: Vec<FrameReport>,
    /// Observations dropped as outliers before the last bundle.
    pub outliers_dropped: usize,
    /// Frames the bundle rejected, by id.
    pub rejected: Vec<String>,
    /// Frames never posed (too few points, or rejected), by id.
    pub unposed: Vec<String>,
    /// Frames left out because their camera has too few frames.
    pub left_out: Vec<String>,
    pub log: Vec<String>,
}

impl SurveyReport {
    /// The numbers as an F64Map: the fit, then `camera.<index>.<f|f_std|cx|cy|k1|k2>`,
    /// `swatch.<id>.<side_mm|sigma_mm>` and `frame.<id>.rms_px`.
    pub fn to_f64_map(&self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        m.insert("observations".to_string(), self.observations as f64);
        m.insert("parameters".to_string(), self.parameters as f64);
        m.insert("rms_px".to_string(), self.rms_px);
        m.insert("median_px".to_string(), self.median_px);
        m.insert("inlier_rms_px".to_string(), self.inlier_rms_px);
        m.insert("inlier_fraction".to_string(), self.inlier_fraction);
        m.insert("sigma_px".to_string(), self.sigma_px);
        m.insert("iterations".to_string(), self.iterations as f64);
        m.insert("frames_posed".to_string(), self.frames.len() as f64);
        m.insert("frames_rejected".to_string(), self.rejected.len() as f64);
        m.insert("frames_unposed".to_string(), self.unposed.len() as f64);
        m.insert("outliers_dropped".to_string(), self.outliers_dropped as f64);
        m.insert(
            "card_corners_solved".to_string(),
            self.card_corners_solved as f64,
        );
        m.insert("card_planarity_mm".to_string(), self.card_planarity_mm);
        m.insert("span_across_mm".to_string(), self.span_across_mm);
        m.insert(
            "span_across_nominal_mm".to_string(),
            self.span_across_nominal_mm,
        );
        m.insert("span_along_mm".to_string(), self.span_along_mm);
        m.insert(
            "span_along_nominal_mm".to_string(),
            self.span_along_nominal_mm,
        );
        for c in &self.cameras {
            let p = format!("camera.{}.", c.index);
            m.insert(format!("{p}frames"), c.frames as f64);
            m.insert(format!("{p}f"), c.f);
            m.insert(format!("{p}f_std"), c.f_std);
            m.insert(format!("{p}cx"), c.cx);
            m.insert(format!("{p}cx_std"), c.cx_std);
            m.insert(format!("{p}cy"), c.cy);
            m.insert(format!("{p}cy_std"), c.cy_std);
            m.insert(format!("{p}k1"), c.k1);
            m.insert(format!("{p}k2"), c.k2);
        }
        for s in &self.swatches {
            m.insert(format!("swatch.{}.side_mm", s.id), s.side_mm);
            m.insert(format!("swatch.{}.sigma_mm", s.id), s.sigma_mm);
        }
        for f in &self.frames {
            m.insert(format!("frame.{}.rms_px", f.id), f.rms_px);
            m.insert(
                format!("frame.{}.position_sigma_mm", f.id),
                f.position_sigma_mm,
            );
            m.insert(format!("frame.{}.angle_sigma_deg", f.id), f.angle_sigma_deg);
        }
        m
    }
}

/// A frame's observations, indexed for the solve.
struct Frame {
    view: usize,
    camera: usize,
    /// (corner id, pixel)
    card: Vec<(u32, [f64; 2])>,
    /// (swatch id, corners)
    swatches: Vec<(u32, [[f64; 2]; 4])>,
}

/// One observation in the bundle.
#[derive(Clone, Copy)]
struct Obs {
    /// Slot in the posed-frame list.
    slot: usize,
    camera: usize,
    point: usize,
    uv: [f64; 2],
}

/// A world-to-camera rigid pose.
#[derive(Clone, Copy, Debug)]
struct Rigid {
    r: Mat3,
    t: [f64; 3],
}

impl Rigid {
    fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        add(mat3_vec(&self.r, p), self.t)
    }

    fn from_camera_to_world(m: &[f64; 12]) -> Self {
        let rt = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
        let r = mat3_transpose(&rt);
        let c = [m[3], m[7], m[11]];
        Rigid {
            r,
            t: scale(mat3_vec(&r, c), -1.0),
        }
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

/// The pixel a camera-space point lands on under `f, cx, cy, k1, k2`,
/// with the derivatives of the pixel with respect to the camera-space
/// point (2x3) and the five intrinsics (2x5). `None` behind the camera.
fn project(intr: &[f64; 5], xc: [f64; 3]) -> Option<([f64; 2], [[f64; 3]; 2], [[f64; 5]; 2])> {
    let [x, y, z] = xc;
    if z.is_nan() || z <= 1e-9 {
        return None;
    }
    let [f, cx, cy, k1, k2] = *intr;
    let (xn, yn) = (x / z, y / z);
    let r2 = xn * xn + yn * yn;
    let d = 1.0 + k1 * r2 + k2 * r2 * r2;
    let uv = [f * xn * d + cx, f * yn * d + cy];
    // d(uv)/d(xn, yn)
    let g = 2.0 * (k1 + 2.0 * k2 * r2);
    let duv_dn = [
        [f * (d + xn * g * xn), f * (xn * g * yn)],
        [f * (yn * g * xn), f * (d + yn * g * yn)],
    ];
    // d(xn, yn)/d(xc)
    let dn_dx = [[1.0 / z, 0.0, -x / (z * z)], [0.0, 1.0 / z, -y / (z * z)]];
    let mut duv_dx = [[0.0; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            duv_dx[i][j] = duv_dn[i][0] * dn_dx[0][j] + duv_dn[i][1] * dn_dx[1][j];
        }
    }
    let duv_dintr = [
        [xn * d, 1.0, 0.0, f * xn * r2, f * xn * r2 * r2],
        [yn * d, 0.0, 1.0, f * yn * r2, f * yn * r2 * r2],
    ];
    Some((uv, duv_dx, duv_dintr))
}

/// Residual value the solver uses for a point behind its camera: large,
/// so the fit is pushed off it, with no gradient (the step comes from
/// the other observations).
const BEHIND_PX: f64 = 1e4;

/// The bundle's state: cameras, posed frames, points, and which of them
/// are free.
struct Model {
    cams: Vec<[f64; 5]>,
    poses: Vec<Rigid>,
    /// NaN for a point not yet known.
    points: Vec<[f64; 3]>,
    obs: Vec<Obs>,
    /// (top, bottom) point indices of the card columns in the scale
    /// residual.
    spans: Vec<(usize, usize)>,
    span_nominal: f64,
    /// Scale residual weight per metre.
    scale_weight: f64,
    soft_l1: f64,
    free: Vec<bool>,
}

impl Model {
    fn n_params(&self) -> usize {
        5 * self.cams.len() + 6 * self.poses.len() + 3 * self.points.len()
    }

    fn cam_at(&self, c: usize) -> usize {
        5 * c
    }

    fn pose_at(&self, s: usize) -> usize {
        5 * self.cams.len() + 6 * s
    }

    fn point_at(&self, p: usize) -> usize {
        5 * self.cams.len() + 6 * self.poses.len() + 3 * p
    }

    /// Columns of the reduced system: parameter index to column, or
    /// `usize::MAX` when held.
    fn columns(&self) -> (Vec<usize>, usize) {
        let mut col = vec![usize::MAX; self.n_params()];
        let mut n = 0;
        for (i, free) in self.free.iter().enumerate() {
            if *free {
                col[i] = n;
                n += 1;
            }
        }
        (col, n)
    }

    /// Reprojection residuals (predicted minus observed), two per
    /// observation, and the scale residual.
    fn residuals(&self) -> (Vec<[f64; 2]>, f64) {
        let mut out = Vec::with_capacity(self.obs.len());
        for o in &self.obs {
            let xc = self.poses[o.slot].apply(self.points[o.point]);
            out.push(match project(&self.cams[o.camera], xc) {
                Some((uv, _, _)) => [uv[0] - o.uv[0], uv[1] - o.uv[1]],
                None => [BEHIND_PX, BEHIND_PX],
            });
        }
        (out, self.scale_residual())
    }

    fn span_mean(&self) -> f64 {
        if self.spans.is_empty() {
            return self.span_nominal;
        }
        self.spans
            .iter()
            .map(|&(top, bottom)| norm(sub(self.points[bottom], self.points[top])))
            .sum::<f64>()
            / self.spans.len() as f64
    }

    fn scale_residual(&self) -> f64 {
        self.scale_weight * (self.span_mean() - self.span_nominal)
    }

    /// The soft-L1 cost of the residuals plus the squared scale residual.
    fn cost(&self, res: &[[f64; 2]], span: f64) -> f64 {
        let s2 = self.soft_l1 * self.soft_l1;
        res.iter()
            .map(|r| {
                let z = (r[0] * r[0] + r[1] * r[1]) / s2;
                2.0 * s2 * ((1.0 + z).sqrt() - 1.0)
            })
            .sum::<f64>()
            + span * span
    }

    /// The robust weight of an observation: the soft-L1 loss's derivative
    /// with respect to the squared residual.
    fn weight(&self, r: &[f64; 2]) -> f64 {
        let z = (r[0] * r[0] + r[1] * r[1]) / (self.soft_l1 * self.soft_l1);
        1.0 / (1.0 + z).sqrt()
    }

    /// The Jacobian rows of one observation over its camera, pose and
    /// point parameters: (pixel, d/dcam 2x5, d/dpose 2x6, d/dpoint 2x3).
    /// The pose derivative is for a left-multiplied rotation update and
    /// a translation step. `None` behind the camera.
    fn obs_jacobian(
        &self,
        o: &Obs,
    ) -> Option<([f64; 2], [[f64; 5]; 2], [[f64; 6]; 2], [[f64; 3]; 2])> {
        let pose = &self.poses[o.slot];
        let x = self.points[o.point];
        let rx = mat3_vec(&pose.r, x);
        let xc = add(rx, pose.t);
        let (uv, duv_dx, duv_dintr) = project(&self.cams[o.camera], xc)?;
        // d xc / d theta = -[rx]x
        let skew = [
            [0.0, rx[2], -rx[1]],
            [-rx[2], 0.0, rx[0]],
            [rx[1], -rx[0], 0.0],
        ];
        let mut dpose = [[0.0; 6]; 2];
        let mut dpoint = [[0.0; 3]; 2];
        for i in 0..2 {
            for j in 0..3 {
                dpose[i][j] = (0..3).map(|k| duv_dx[i][k] * skew[k][j]).sum();
                dpose[i][3 + j] = duv_dx[i][j];
                dpoint[i][j] = (0..3).map(|k| duv_dx[i][k] * pose.r[k][j]).sum();
            }
        }
        Some((uv, duv_dintr, dpose, dpoint))
    }

    /// The weighted normal matrix (flat, n x n) and gradient of the
    /// reduced system, with the residuals they were built from.
    fn normal_equations(
        &self,
        col: &[usize],
        n: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<[f64; 2]>, f64) {
        let mut a = vec![0.0; n * n];
        let mut g = vec![0.0; n];
        let mut res = Vec::with_capacity(self.obs.len());
        let mut rows: Vec<(usize, [f64; 2])> = Vec::with_capacity(14);
        for o in &self.obs {
            let Some((uv, dcam, dpose, dpoint)) = self.obs_jacobian(o) else {
                res.push([BEHIND_PX, BEHIND_PX]);
                continue;
            };
            let r = [uv[0] - o.uv[0], uv[1] - o.uv[1]];
            res.push(r);
            let w = self.weight(&r);
            rows.clear();
            let base = self.cam_at(o.camera);
            for j in 0..5 {
                if col[base + j] != usize::MAX {
                    rows.push((col[base + j], [dcam[0][j], dcam[1][j]]));
                }
            }
            let base = self.pose_at(o.slot);
            for j in 0..6 {
                if col[base + j] != usize::MAX {
                    rows.push((col[base + j], [dpose[0][j], dpose[1][j]]));
                }
            }
            let base = self.point_at(o.point);
            for j in 0..3 {
                if col[base + j] != usize::MAX {
                    rows.push((col[base + j], [dpoint[0][j], dpoint[1][j]]));
                }
            }
            for &(ci, di) in &rows {
                g[ci] += w * (di[0] * r[0] + di[1] * r[1]);
                for &(cj, dj) in &rows {
                    a[ci * n + cj] += w * (di[0] * dj[0] + di[1] * dj[1]);
                }
            }
        }
        // The scale residual: d span / d point = unit vector / count.
        let span = self.scale_residual();
        if !self.spans.is_empty() {
            let mut entries: Vec<(usize, f64)> = Vec::with_capacity(6 * self.spans.len());
            let count = self.spans.len() as f64;
            for &(top, bottom) in &self.spans {
                let d = sub(self.points[bottom], self.points[top]);
                let len = norm(d).max(1e-12);
                for j in 0..3 {
                    let v = self.scale_weight * d[j] / (len * count);
                    let pb = self.point_at(bottom) + j;
                    let pt = self.point_at(top) + j;
                    if col[pb] != usize::MAX {
                        entries.push((col[pb], v));
                    }
                    if col[pt] != usize::MAX {
                        entries.push((col[pt], -v));
                    }
                }
            }
            for &(ci, di) in &entries {
                g[ci] += di * span;
                for &(cj, dj) in &entries {
                    a[ci * n + cj] += di * dj;
                }
            }
        }
        (a, g, res, span)
    }

    /// Applies a step of the reduced system.
    fn step(&mut self, col: &[usize], delta: &[f64]) {
        let at = |i: usize| {
            if col[i] == usize::MAX {
                0.0
            } else {
                delta[col[i]]
            }
        };
        for c in 0..self.cams.len() {
            let base = self.cam_at(c);
            for j in 0..5 {
                self.cams[c][j] += at(base + j);
            }
        }
        for s in 0..self.poses.len() {
            let base = self.pose_at(s);
            let theta = [at(base), at(base + 1), at(base + 2)];
            let pose = &mut self.poses[s];
            pose.r = nearest_rotation(&mat3_mul(&rotation_from_vector(theta), &pose.r));
            pose.t = add(pose.t, [at(base + 3), at(base + 4), at(base + 5)]);
        }
        for p in 0..self.points.len() {
            let base = self.point_at(p);
            for j in 0..3 {
                self.points[p][j] += at(base + j);
            }
        }
    }

    /// Levenberg–Marquardt to convergence; returns the iterations run.
    fn solve(&mut self, max_iterations: usize) -> usize {
        let (col, n) = self.columns();
        if n == 0 {
            return 0;
        }
        let (res, span) = self.residuals();
        let mut cost = self.cost(&res, span);
        let mut lambda = 1e-3;
        let mut iterations = 0;
        for _ in 0..max_iterations {
            iterations += 1;
            let (a, g, _, _) = self.normal_equations(&col, n);
            let mut improved = false;
            for _ in 0..12 {
                let mut damped = a.clone();
                for i in 0..n {
                    damped[i * n + i] += lambda * a[i * n + i].max(1e-9);
                }
                let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
                let Some(delta) = cholesky_solve(damped, n, &rhs) else {
                    lambda *= 10.0;
                    continue;
                };
                let saved = (self.cams.clone(), self.poses.clone(), self.points.clone());
                self.step(&col, &delta);
                let (res, span) = self.residuals();
                let trial = self.cost(&res, span);
                if trial < cost {
                    let gain = (cost - trial) / cost.max(1e-300);
                    cost = trial;
                    lambda = (lambda / 3.0).max(1e-12);
                    improved = true;
                    if gain < 1e-9 {
                        return iterations;
                    }
                    break;
                }
                (self.cams, self.poses, self.points) = saved;
                lambda *= 4.0;
            }
            if !improved {
                break;
            }
        }
        iterations
    }
}

/// Solves `a x = b` for a symmetric positive-definite `a` (flat, n x n,
/// consumed) by Cholesky; `None` when `a` is not positive-definite.
fn cholesky_solve(mut a: Vec<f64>, n: usize, b: &[f64]) -> Option<Vec<f64>> {
    cholesky(&mut a, n)?;
    let mut x = b.to_vec();
    // L y = b
    for i in 0..n {
        let mut s = x[i];
        for k in 0..i {
            s -= a[i * n + k] * x[k];
        }
        x[i] = s / a[i * n + i];
    }
    // L^T x = y
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in i + 1..n {
            s -= a[k * n + i] * x[k];
        }
        x[i] = s / a[i * n + i];
    }
    Some(x)
}

/// In-place lower Cholesky factor of a symmetric positive-definite flat
/// matrix.
fn cholesky(a: &mut [f64], n: usize) -> Option<()> {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d.is_nan() || d <= 0.0 || !d.is_finite() {
            return None;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    Some(())
}

/// The inverse of a symmetric positive-definite flat matrix, or `None`.
fn cholesky_inverse(mut a: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    cholesky(&mut a, n)?;
    // Invert L in place (lower triangular), then inv = L^-T L^-1.
    let mut linv = vec![0.0; n * n];
    for i in 0..n {
        linv[i * n + i] = 1.0 / a[i * n + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= a[i * n + k] * linv[k * n + j];
            }
            linv[i * n + j] = s / a[i * n + i];
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in i..n {
                s += linv[k * n + i] * linv[k * n + j];
            }
            inv[i * n + j] = s;
            inv[j * n + i] = s;
        }
    }
    Some(inv)
}

/// Intrinsics `f, cx, cy, k1, k2` of a camera model (fx and fy averaged,
/// radial terms beyond k2 and tangential terms dropped).
fn intrinsics_of(camera: &CameraModel) -> [f64; 5] {
    let (k1, k2) = match &camera.distortion {
        Distortion::Radial { k, .. } => (
            k.first().copied().unwrap_or(0.0),
            k.get(1).copied().unwrap_or(0.0),
        ),
        _ => (0.0, 0.0),
    };
    [(camera.fx + camera.fy) * 0.5, camera.cx, camera.cy, k1, k2]
}

fn camera_with(camera: &CameraModel, intr: &[f64; 5]) -> CameraModel {
    let mut out = camera.clone();
    out.fx = intr[0];
    out.fy = intr[0];
    out.cx = intr[1];
    out.cy = intr[2];
    out.distortion = if intr[3] == 0.0 && intr[4] == 0.0 {
        Distortion::None
    } else {
        Distortion::Radial {
            k: vec![intr[3], intr[4]],
            p: [0.0, 0.0],
        }
    };
    out
}

/// Pose of a frame from world points seen at pixels, grouped for the
/// planar start (the card as one group, each swatch its own): the planar
/// solver on every group, the best refined robustly, the worst points
/// dropped. `None` when too few points fit within `rms_limit`.
fn pose_on_points(
    camera: &CameraModel,
    points: &[Correspondence],
    rms_limit: f64,
) -> Option<(Rigid, f64)> {
    if points.len() < 6 {
        return None;
    }
    let options = SolveOptions {
        solve_focal: false,
        solve_distortion: false,
        huber_px: 2.0,
        max_residual_px: 4.0,
        iterations: 40,
    };
    let solve = solve_pose(camera, points, &options).ok()?;
    let needed = 6.max((0.6 * points.len() as f64).ceil() as usize);
    if solve.corners_used < needed || solve.rms_px.is_nan() || solve.rms_px >= rms_limit {
        return None;
    }
    Some((
        Rigid::from_camera_to_world(&solve.camera_to_world),
        solve.rms_px,
    ))
}

/// Linear multi-view triangulation of a point from (pose, camera, pixel)
/// observations; `None` when it lands behind any camera.
fn triangulate(obs: &[(&Rigid, &CameraModel, [f64; 2])]) -> Option<[f64; 3]> {
    let mut ata = vec![vec![0.0; 4]; 4];
    for (pose, camera, uv) in obs {
        let ray = camera.ray(*uv);
        let (xn, yn) = (ray[0] / ray[2], ray[1] / ray[2]);
        let p = |i: usize| -> [f64; 4] { [pose.r[i][0], pose.r[i][1], pose.r[i][2], pose.t[i]] };
        let (p0, p1, p2) = (p(0), p(1), p(2));
        for row in [
            std::array::from_fn::<f64, 4, _>(|k| xn * p2[k] - p0[k]),
            std::array::from_fn::<f64, 4, _>(|k| yn * p2[k] - p1[k]),
        ] {
            for i in 0..4 {
                for j in 0..4 {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
    }
    let v = smallest_eigenvector(&ata);
    if v[3].abs() < 1e-12 {
        return None;
    }
    let x = [v[0] / v[3], v[1] / v[3], v[2] / v[3]];
    for (pose, _, _) in obs {
        if pose.apply(x)[2] <= 0.0 {
            return None;
        }
    }
    Some(x)
}

/// Rotation and translation with `dst ~ R src + t` (Horn's closed form:
/// the unit quaternion is the top eigenvector of a 4x4 built from the
/// cross-covariance, which is sound for planar point sets where a polar
/// factor of the singular cross-covariance is not).
fn rigid_fit(src: &[[f64; 3]], dst: &[[f64; 3]]) -> (Mat3, [f64; 3]) {
    let n = src.len() as f64;
    let mean = |pts: &[[f64; 3]]| -> [f64; 3] {
        let mut m = [0.0; 3];
        for p in pts {
            m = add(m, *p);
        }
        scale(m, 1.0 / n)
    };
    let (cs, cd) = (mean(src), mean(dst));
    // S[a][b] = sum of src_a * dst_b about the centroids.
    let mut sm = [[0.0; 3]; 3];
    for (s, d) in src.iter().zip(dst) {
        let (s, d) = (sub(*s, cs), sub(*d, cd));
        for a in 0..3 {
            for b in 0..3 {
                sm[a][b] += s[a] * d[b];
            }
        }
    }
    let (sxx, sxy, sxz) = (sm[0][0], sm[0][1], sm[0][2]);
    let (syx, syy, syz) = (sm[1][0], sm[1][1], sm[1][2]);
    let (szx, szy, szz) = (sm[2][0], sm[2][1], sm[2][2]);
    let nm = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    // The top eigenvector of N is the bottom one of -N.
    let neg: Vec<Vec<f64>> = nm.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
    let q = smallest_eigenvector(&neg);
    let len = q.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
    let (w, x, y, z) = (q[0] / len, q[1] / len, q[2] / len, q[3] / len);
    let r = [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    (r, sub(cd, mat3_vec(&r, cs)))
}

/// The survey of a detected set: poses its views, solves its cameras,
/// and replaces its markers and board with the solved map. Views the
/// survey could not pose are left unposed. The report says how it went.
pub fn survey(set: &mut ViewSet, options: &SurveyOptions) -> Result<SurveyReport, String> {
    let spec = set
        .board
        .as_ref()
        .map(|b| b.spec.clone())
        .ok_or("the set has no board: run view-detect with the card first")?;
    spec.validate()?;
    let mut report = SurveyReport::default();
    let mut log = |line: String| report.log.push(line);

    // Frames: views with observations, their card corners and swatches.
    let mut frames: Vec<Frame> = Vec::new();
    for (i, view) in set.views.iter().enumerate() {
        let Some(obs) = &view.observations else {
            continue;
        };
        let card: Vec<(u32, [f64; 2])> = obs.board.iter().map(|c| (c.id, c.pixel)).collect();
        let mut swatches: Vec<(u32, [[f64; 2]; 4])> = obs
            .markers
            .iter()
            .filter(|m| m.family == options.swatch_family && m.id < options.max_swatch_id)
            .map(|m| (m.id, m.corners))
            .collect();
        swatches.sort_by_key(|s| s.0);
        swatches.dedup_by_key(|s| s.0);
        frames.push(Frame {
            view: i,
            camera: view.camera as usize,
            card,
            swatches,
        });
    }
    if frames.is_empty() {
        return Err("no view carries observations: run view-detect first".to_string());
    }
    // Cameras with too few frames cannot self-calibrate.
    let mut frames_per_camera = vec![0usize; set.cameras.len()];
    for f in &frames {
        frames_per_camera[f.camera] += 1;
    }
    let few: Vec<usize> = (0..set.cameras.len())
        .filter(|&c| {
            frames_per_camera[c] > 0 && frames_per_camera[c] < options.min_frames_per_camera
        })
        .collect();
    if !few.is_empty() {
        for f in &frames {
            if few.contains(&f.camera) {
                report.left_out.push(set.views[f.view].id.clone());
            }
        }
        log(format!(
            "{} frames left out: their cameras ({}) have under {} frames",
            report.left_out.len(),
            few.iter()
                .map(|&c| format!("{c} '{}'", set.cameras[c].label))
                .collect::<Vec<_>>()
                .join(", "),
            options.min_frames_per_camera
        ));
        frames.retain(|f| !few.contains(&f.camera));
    }
    if frames.is_empty() {
        return Err("no camera has enough frames to survey".to_string());
    }

    // Points: card corners first, then swatch corners in id order.
    let n_card = spec.n_corners() as usize;
    let mut swatch_ids: Vec<u32> = frames
        .iter()
        .flat_map(|f| f.swatches.iter().map(|s| s.0))
        .collect();
    swatch_ids.sort_unstable();
    swatch_ids.dedup();
    let swatch_base = |id: u32| n_card + 4 * swatch_ids.binary_search(&id).expect("known swatch");
    let n_points = n_card + 4 * swatch_ids.len();
    let nominal: Vec<[f64; 3]> = (0..n_card as u32)
        .map(|id| {
            let [x, y] = spec.corner(id).expect("corner in range");
            [x, y, 0.0]
        })
        .collect();

    // Cameras: the seed intrinsics.
    let mut cams: Vec<[f64; 5]> = set.cameras.iter().map(intrinsics_of).collect();
    let camera_models = |cams: &[[f64; 5]], set: &ViewSet| -> Vec<CameraModel> {
        set.cameras
            .iter()
            .zip(cams)
            .map(|(c, intr)| camera_with(c, intr))
            .collect()
    };

    // Calibration: each camera with enough full card frames refines its
    // intrinsics against the nominal card before anything else.
    let card_group = u32::MAX;
    for c in 0..set.cameras.len() {
        let good: Vec<usize> = (0..frames.len())
            .filter(|&k| {
                frames[k].camera == c && frames[k].card.len() >= options.calibration_corners
            })
            .collect();
        if frames_per_camera[c] == 0 {
            continue;
        }
        if good.len() < options.min_calibration_frames {
            log(format!(
                "camera {c} '{}': {} frames with {}+ card corners, starting from the seed f {:.0} px",
                set.cameras[c].label,
                good.len(),
                options.calibration_corners,
                cams[c][0]
            ));
            continue;
        }
        let models = camera_models(&cams, set);
        let mut poses = Vec::new();
        let mut obs = Vec::new();
        for &k in &good {
            let pts: Vec<Correspondence> = frames[k]
                .card
                .iter()
                .map(|&(id, px)| Correspondence {
                    world: nominal[id as usize],
                    pixel: px,
                    marker: card_group,
                })
                .collect();
            let Some((pose, _)) = pose_on_points(&models[c], &pts, options.pose_rms_px * 4.0)
            else {
                continue;
            };
            let slot = poses.len();
            poses.push(pose);
            for &(id, px) in &frames[k].card {
                obs.push(Obs {
                    slot,
                    camera: 0,
                    point: id as usize,
                    uv: px,
                });
            }
        }
        if poses.len() < options.min_calibration_frames {
            log(format!(
                "camera {c}: only {} card frames posed, starting from the seed",
                poses.len()
            ));
            continue;
        }
        let mut model = Model {
            cams: vec![cams[c]],
            poses,
            points: nominal.clone(),
            obs,
            spans: Vec::new(),
            span_nominal: 0.0,
            scale_weight: 0.0,
            soft_l1: options.soft_l1_px,
            free: Vec::new(),
        };
        let n = model.n_params();
        model.free = (0..n).map(|i| i < 5 + 6 * model.poses.len()).collect();
        let iterations = model.solve(options.max_iterations);
        let (res, _) = model.residuals();
        let rms = rms_of(&res);
        cams[c] = model.cams[0];
        log(format!(
            "camera {c} '{}': calibrated on {} card frames in {iterations} iterations, rms {rms:.2} px, f {:.0} px, k1 {:+.4} k2 {:+.3}",
            set.cameras[c].label,
            model.poses.len(),
            cams[c][0],
            cams[c][3],
            cams[c][4]
        ));
    }

    // Initial poses on the card, then grow: triangulate swatch corners
    // from posed frames, pose frames on known points, repeat.
    let mut points: Vec<[f64; 3]> = vec![[f64::NAN; 3]; n_points];
    points[..n_card].copy_from_slice(&nominal);
    let mut posed: BTreeMap<usize, Rigid> = BTreeMap::new();
    let mut rejected: Vec<usize> = Vec::new();
    // (frame, point) pairs dropped as outliers.
    let mut excluded: std::collections::BTreeSet<(usize, usize)> =
        std::collections::BTreeSet::new();
    {
        let models = camera_models(&cams, set);
        for (k, f) in frames.iter().enumerate() {
            if f.card.len() < options.min_card_corners {
                continue;
            }
            let pts: Vec<Correspondence> = f
                .card
                .iter()
                .map(|&(id, px)| Correspondence {
                    world: nominal[id as usize],
                    pixel: px,
                    marker: card_group,
                })
                .collect();
            if let Some((pose, _)) = pose_on_points(&models[f.camera], &pts, options.pose_rms_px) {
                posed.insert(k, pose);
            }
        }
    }
    log(format!("init: {} frames posed on the card", posed.len()));
    let extend = |cams: &[[f64; 5]],
                  posed: &mut BTreeMap<usize, Rigid>,
                  points: &mut Vec<[f64; 3]>,
                  rejected: &[usize],
                  label: &str,
                  log: &mut dyn FnMut(String)|
     -> bool {
        let models = camera_models(cams, set);
        let mut grew = false;
        for pass in 1..=6 {
            let mut new_points = 0;
            for (si, &id) in swatch_ids.iter().enumerate() {
                for j in 0..4 {
                    let p = n_card + 4 * si + j;
                    if !points[p][0].is_nan() {
                        continue;
                    }
                    let obs: Vec<(&Rigid, &CameraModel, [f64; 2])> = posed
                        .iter()
                        .filter_map(|(&k, pose)| {
                            let f = &frames[k];
                            f.swatches
                                .iter()
                                .find(|s| s.0 == id)
                                .map(|s| (pose, &models[f.camera], s.1[j]))
                        })
                        .collect();
                    if obs.len() >= 2
                        && let Some(x) = triangulate(&obs)
                    {
                        points[p] = x;
                        new_points += 1;
                    }
                }
            }
            let mut new_frames = 0;
            for (k, f) in frames.iter().enumerate() {
                if posed.contains_key(&k) || rejected.contains(&k) {
                    continue;
                }
                let mut pts: Vec<Correspondence> = f
                    .card
                    .iter()
                    .filter(|(id, _)| !points[*id as usize][0].is_nan())
                    .map(|&(id, px)| Correspondence {
                        world: points[id as usize],
                        pixel: px,
                        marker: card_group,
                    })
                    .collect();
                for (id, corners) in &f.swatches {
                    let base = swatch_base(*id);
                    if (0..4).all(|j| !points[base + j][0].is_nan()) {
                        for j in 0..4 {
                            pts.push(Correspondence {
                                world: points[base + j],
                                pixel: corners[j],
                                marker: *id,
                            });
                        }
                    }
                }
                if pts.len() >= options.min_points
                    && let Some((pose, _)) =
                        pose_on_points(&models[f.camera], &pts, options.pose_rms_px)
                {
                    posed.insert(k, pose);
                    new_frames += 1;
                }
            }
            log(format!(
                "{label} pass {pass}: {new_points} swatch corners triangulated, {new_frames} frames posed"
            ));
            grew |= new_points > 0 || new_frames > 0;
            if new_points == 0 && new_frames == 0 {
                break;
            }
        }
        grew
    };
    extend(&cams, &mut posed, &mut points, &rejected, "init", &mut log);

    // The card's columns for the scale.
    let cols = (spec.squares_x - 1) as usize;
    let rows = (spec.squares_y - 1) as usize;
    let span_nominal = (rows as f64 - 1.0) * spec.pitch_y_m;

    // Bundle rounds.
    let mut fit: Option<(Model, Vec<usize>, Vec<[f64; 2]>, usize, usize)> = None;
    for round in 1..=options.rounds {
        let slots: Vec<usize> = posed.keys().copied().collect();
        if slots.is_empty() {
            return Err("no frame could be posed".to_string());
        }
        // A point seen once is only known along a ray: it stays out.
        let mut seen = vec![0usize; n_points];
        for &k in &slots {
            for (id, _) in &frames[k].card {
                if !excluded.contains(&(k, *id as usize)) {
                    seen[*id as usize] += 1;
                }
            }
            for (id, _) in &frames[k].swatches {
                let base = swatch_base(*id);
                for j in 0..4 {
                    if !excluded.contains(&(k, base + j)) {
                        seen[base + j] += 1;
                    }
                }
            }
        }
        let known: Vec<bool> = (0..n_points)
            .map(|p| !points[p][0].is_nan() && seen[p] >= 2)
            .collect();
        let mut obs = Vec::new();
        for (slot, &k) in slots.iter().enumerate() {
            let f = &frames[k];
            for &(id, px) in &f.card {
                if known[id as usize] && !excluded.contains(&(k, id as usize)) {
                    obs.push(Obs {
                        slot,
                        camera: f.camera,
                        point: id as usize,
                        uv: px,
                    });
                }
            }
            for (id, corners) in &f.swatches {
                let base = swatch_base(*id);
                for j in 0..4 {
                    if known[base + j] && !excluded.contains(&(k, base + j)) {
                        obs.push(Obs {
                            slot,
                            camera: f.camera,
                            point: base + j,
                            uv: corners[j],
                        });
                    }
                }
            }
        }
        let spans: Vec<(usize, usize)> = (0..cols)
            .map(|c| (c, (rows - 1) * cols + c))
            .filter(|&(top, bottom)| known[top] && known[bottom])
            .collect();
        if spans.is_empty() {
            return Err("no card column has both end corners solved: no scale".to_string());
        }
        let anchor = slots
            .iter()
            .position(|&k| frames[k].card.len() >= options.min_card_corners)
            .unwrap_or(0);
        let mut model = Model {
            cams: cams.clone(),
            poses: slots.iter().map(|k| posed[k]).collect(),
            points: points
                .iter()
                .map(|p| if p[0].is_nan() { [0.0; 3] } else { *p })
                .collect(),
            obs,
            spans,
            span_nominal,
            scale_weight: options.scale_weight * 1e3,
            soft_l1: options.soft_l1_px,
            free: Vec::new(),
        };
        let n = model.n_params();
        let mut free = vec![true; n];
        for c in 0..cams.len() {
            if frames_per_camera[c] == 0 || few.contains(&c) {
                for j in 0..5 {
                    free[model.cam_at(c) + j] = false;
                }
            }
        }
        for j in 0..6 {
            free[model.pose_at(anchor) + j] = false;
        }
        for p in 0..n_points {
            if !known[p] {
                for j in 0..3 {
                    free[model.point_at(p) + j] = false;
                }
            }
        }
        model.free = free;
        let (_, n_free) = model.columns();
        log(format!(
            "bundle {round}: {} observations, {} cameras, {} frames, {} points, {n_free} parameters",
            model.obs.len(),
            cams.iter()
                .enumerate()
                .filter(|(c, _)| frames_per_camera[*c] > 0 && !few.contains(c))
                .count(),
            model.poses.len(),
            known.iter().filter(|k| **k).count()
        ));
        let iterations = model.solve(options.max_iterations);
        let (res, _) = model.residuals();
        let err: Vec<f64> = res
            .iter()
            .map(|r| (r[0] * r[0] + r[1] * r[1]).sqrt())
            .collect();
        let inliers: Vec<f64> = err
            .iter()
            .copied()
            .filter(|e| *e < 3.0 * options.soft_l1_px)
            .collect();
        log(format!(
            "bundle {round} done in {iterations} iterations: rms {:.3} px, median {:.3}, inliers {:.1} % at {:.3} px",
            rms_of(&res),
            median(err.clone()),
            100.0 * inliers.len() as f64 / err.len().max(1) as f64,
            (inliers.iter().map(|e| e * e).sum::<f64>() / inliers.len().max(1) as f64).sqrt()
        ));
        // Carry the solution back.
        cams = model.cams.clone();
        for (slot, &k) in slots.iter().enumerate() {
            posed.insert(k, model.poses[slot]);
        }
        for p in 0..n_points {
            if known[p] {
                points[p] = model.points[p];
            }
        }
        // Single outliers are dropped first; then frames over the limit
        // (rms over what remains); the rest grows again.
        let last = round == options.rounds;
        let mut outliers = 0;
        if !last {
            for (o, e) in model.obs.iter().zip(&err) {
                if *e > options.outlier_px && excluded.insert((slots[o.slot], o.point)) {
                    outliers += 1;
                }
            }
        }
        let mut bad = Vec::new();
        for (slot, &k) in slots.iter().enumerate() {
            let sel: Vec<f64> = model
                .obs
                .iter()
                .zip(&err)
                .filter(|(o, e)| o.slot == slot && (last || **e <= options.outlier_px))
                .map(|(_, e)| *e)
                .collect();
            let rms = (sel.iter().map(|e| e * e).sum::<f64>() / sel.len().max(1) as f64).sqrt();
            // Too few observations left (its corners were the outliers)
            // or too poor a fit: the frame goes.
            if sel.len() < options.min_points || rms > options.reject_px {
                bad.push(k);
            }
        }
        fit = Some((model, slots, res, iterations, anchor));
        if last {
            break;
        }
        if outliers > 0 {
            report.outliers_dropped += outliers;
            log(format!(
                "round {}: dropped {outliers} observations over {} px",
                round + 1,
                options.outlier_px
            ));
        }
        for &k in &bad {
            posed.remove(&k);
            rejected.push(k);
        }
        if !bad.is_empty() {
            log(format!(
                "round {}: dropped {} frames over {} px rms: {}",
                round + 1,
                bad.len(),
                options.reject_px,
                bad.iter()
                    .map(|&k| set.views[frames[k].view].id.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        }
        let grew = extend(
            &cams,
            &mut posed,
            &mut points,
            &rejected,
            &format!("round {}", round + 1),
            &mut log,
        );
        if !grew && bad.is_empty() && outliers == 0 {
            break;
        }
    }
    let (model, slots, res, iterations, anchor) = fit.expect("at least one round ran");
    let known: Vec<bool> = (0..n_points)
        .map(|p| model.free[model.point_at(p)])
        .collect();

    // Uncertainties: the normal matrix with the anchor freed, its
    // pseudo-inverse through the analytic gauge (the six rigid motions of
    // the world), scaled by the residual variance.
    let mut freed = model.free.clone();
    for j in 0..6 {
        freed[model.pose_at(anchor) + j] = true;
    }
    let free_model = Model {
        free: freed,
        ..model
    };
    let (col, n) = free_model.columns();
    let (a, _, _, _) = free_model.normal_equations_plain(&col, n);
    let m = 2 * free_model.obs.len() + 1;
    let dof = (m as f64 - n as f64 + 6.0).max(1.0);
    let plain: f64 = res.iter().map(|r| r[0] * r[0] + r[1] * r[1]).sum::<f64>();
    let sigma2 = plain / dof;
    let card_known: Vec<usize> = (0..n_card).filter(|&p| known[p]).collect();
    let covariance = datum_covariance(&free_model, &col, n, a, &card_known);
    let variances = covariance
        .as_ref()
        .map(|(v, _)| v.iter().map(|x| x * sigma2).collect::<Vec<f64>>());
    let pose_blocks = covariance.as_ref().map(|(_, blocks)| blocks.clone());
    let var_at = |i: usize| -> f64 {
        match (&variances, col[i]) {
            (Some(v), c) if c != usize::MAX => v[c].max(0.0),
            _ => f64::NAN,
        }
    };
    let model = free_model;

    // World frame: the nominal card's frame placed by a rigid fit onto
    // the solved corners, y and z flipped so z points at the cameras.
    let card_ok: Vec<usize> = (0..n_card).filter(|&p| known[p]).collect();
    if card_ok.len() < 4 {
        return Err(format!(
            "only {} card corners solved: no world frame",
            card_ok.len()
        ));
    }
    let src: Vec<[f64; 3]> = card_ok.iter().map(|&p| nominal[p]).collect();
    let dst: Vec<[f64; 3]> = card_ok.iter().map(|&p| model.points[p]).collect();
    let (r_fit, t_fit) = rigid_fit(&src, &dst);
    let flip = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    let to_world = |x: [f64; 3]| -> [f64; 3] {
        mat3_vec(&flip, mat3_vec(&mat3_transpose(&r_fit), sub(x, t_fit)))
    };
    // old = R_fit S world + t_fit, so a world-to-camera pose composes as
    // R' = R R_fit S, t' = R t_fit + t.
    let a_rot = mat3_mul(&r_fit, &flip);
    let world_points: Vec<[f64; 3]> = model.points.iter().map(|p| to_world(*p)).collect();

    // The set: cameras, poses, markers, board.
    for (c, intr) in model.cams.iter().enumerate() {
        if frames_per_camera[c] > 0 && !few.contains(&c) {
            set.cameras[c] = camera_with(&set.cameras[c], intr);
        }
    }
    let err: Vec<f64> = res
        .iter()
        .map(|r| (r[0] * r[0] + r[1] * r[1]).sqrt())
        .collect();
    let posed_views: Vec<usize> = slots.iter().map(|&k| frames[k].view).collect();
    for view in &mut set.views {
        view.tags
            .retain(|t| !(t.starts_with("solved:") || t.starts_with("rms:")));
    }
    for (slot, &k) in slots.iter().enumerate() {
        let pose = &model.poses[slot];
        let world_pose = Rigid {
            r: mat3_mul(&pose.r, &a_rot),
            t: add(mat3_vec(&pose.r, t_fit), pose.t),
        };
        let sel: Vec<f64> = model
            .obs
            .iter()
            .zip(&err)
            .filter(|(o, _)| o.slot == slot)
            .map(|(_, e)| *e)
            .collect();
        let rms = (sel.iter().map(|e| e * e).sum::<f64>() / sel.len().max(1) as f64).sqrt();
        // The eye c = -Rᵀ t moves by Rᵀ(δθ × t − δt) for a pose step:
        // its variance from the pose's 6x6 block.
        let (position_sigma, angle_sigma) = match &pose_blocks {
            Some(blocks) => {
                let cov = &blocks[slot];
                let t = pose.t;
                let jac: [[f64; 6]; 3] = [
                    [0.0, -t[2], t[1], -1.0, 0.0, 0.0],
                    [t[2], 0.0, -t[0], 0.0, -1.0, 0.0],
                    [-t[1], t[0], 0.0, 0.0, 0.0, -1.0],
                ];
                let mut var_c = 0.0;
                for row in &jac {
                    for i in 0..6 {
                        for j in 0..6 {
                            var_c += row[i] * cov[i][j] * row[j];
                        }
                    }
                }
                let var_theta: f64 = (0..3).map(|i| cov[i][i]).sum();
                (
                    (var_c.max(0.0) * sigma2).sqrt(),
                    (var_theta.max(0.0) * sigma2).sqrt(),
                )
            }
            None => (f64::NAN, f64::NAN),
        };
        let view: &mut View = &mut set.views[frames[k].view];
        view.camera_to_world = Some(world_pose.camera_to_world());
        view.tags.push("solved:survey".to_string());
        view.tags.push(format!("rms:{rms:.2}px"));
        report.frames.push(FrameReport {
            id: view.id.clone(),
            rms_px: rms,
            card_corners: frames[k].card.len(),
            swatches: frames[k].swatches.len(),
            position_sigma_mm: position_sigma * 1e3,
            angle_sigma_deg: angle_sigma.to_degrees(),
        });
    }
    for (i, view) in set.views.iter_mut().enumerate() {
        if !posed_views.contains(&i) && view.observations.is_some() {
            view.camera_to_world = None;
            report.unposed.push(view.id.clone());
        }
    }
    report.rejected = rejected
        .iter()
        .map(|&k| set.views[frames[k].view].id.clone())
        .collect();
    let mut markers = Vec::new();
    for (si, &id) in swatch_ids.iter().enumerate() {
        let base = n_card + 4 * si;
        if !(0..4).all(|j| known[base + j]) {
            continue;
        }
        let corners: [[f64; 3]; 4] = std::array::from_fn(|j| world_points[base + j]);
        let side = (0..4)
            .map(|j| norm(sub(corners[(j + 1) % 4], corners[j])))
            .sum::<f64>()
            / 4.0;
        let var: f64 = (0..4)
            .map(|j| {
                (0..3)
                    .map(|d| var_at(model.point_at(base + j) + d))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / 4.0;
        markers.push(Marker {
            id,
            size_m: side,
            corners,
        });
        report.swatches.push(SwatchReport {
            id,
            side_mm: side * 1e3,
            sigma_mm: var.sqrt() * 1e3,
        });
    }
    set.markers = markers;
    let mut board_corners = Vec::new();
    for &p in &card_ok {
        let var: f64 = (0..3).map(|d| var_at(model.point_at(p) + d)).sum();
        board_corners.push(BoardCorner {
            id: p as u32,
            position: world_points[p],
            sigma_m: var.sqrt(),
        });
    }
    set.board = Some(Board {
        spec: spec.clone(),
        corners: board_corners,
    });
    // The world is the card's frame with z towards the cameras: for a card
    // lying flat under the subject, the way a session is shot, that is up.
    set.world.up = [0.0, 0.0, 1.0];
    set.provenance.tools.push(format!(
        "volumetric view-survey ({} frames, rms {:.2} px)",
        slots.len(),
        rms_of(&res)
    ));

    // The report.
    let inliers: Vec<f64> = err
        .iter()
        .copied()
        .filter(|e| *e < 3.0 * options.soft_l1_px)
        .collect();
    report.observations = model.obs.len();
    report.parameters = n - 6;
    report.rms_px = rms_of(&res);
    report.median_px = median(err.clone());
    report.inlier_rms_px =
        (inliers.iter().map(|e| e * e).sum::<f64>() / inliers.len().max(1) as f64).sqrt();
    report.inlier_fraction = inliers.len() as f64 / err.len().max(1) as f64;
    report.sigma_px = sigma2.sqrt();
    report.iterations = iterations;
    report.card_corners_solved = card_ok.len();
    report.card_planarity_mm = (card_ok
        .iter()
        .map(|&p| world_points[p][2].powi(2))
        .sum::<f64>()
        / card_ok.len() as f64)
        .sqrt()
        * 1e3;
    report.span_across_mm = model.span_mean() * 1e3;
    report.span_across_nominal_mm = span_nominal * 1e3;
    let along: Vec<f64> = (0..rows)
        .filter(|&r| known[r * cols] && known[r * cols + cols - 1])
        .map(|r| {
            norm(sub(
                world_points[r * cols + cols - 1],
                world_points[r * cols],
            ))
        })
        .collect();
    report.span_along_mm = if along.is_empty() {
        f64::NAN
    } else {
        along.iter().sum::<f64>() / along.len() as f64 * 1e3
    };
    report.span_along_nominal_mm = (cols as f64 - 1.0) * spec.pitch_x_m * 1e3;
    for (c, intr) in model.cams.iter().enumerate() {
        if frames_per_camera[c] == 0 || few.contains(&c) {
            continue;
        }
        report.cameras.push(CameraReport {
            index: c,
            label: set.cameras[c].label.clone(),
            frames: slots.iter().filter(|&&k| frames[k].camera == c).count(),
            f: intr[0],
            f_std: var_at(model.cam_at(c)).sqrt(),
            cx: intr[1],
            cx_std: var_at(model.cam_at(c) + 1).sqrt(),
            cy: intr[2],
            cy_std: var_at(model.cam_at(c) + 2).sqrt(),
            k1: intr[3],
            k2: intr[4],
        });
    }
    log(format!(
        "survey done: {} frames, rms {:.3} px, {}/{n_card} card corners solved, planar to {:.3} mm rms",
        slots.len(),
        report.rms_px,
        card_ok.len(),
        report.card_planarity_mm
    ));
    Ok(report)
}

impl Model {
    /// The unweighted normal matrix (every observation at weight one), for
    /// the covariance.
    fn normal_equations_plain(
        &self,
        col: &[usize],
        n: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<[f64; 2]>, f64) {
        let plain = Model {
            soft_l1: f64::INFINITY,
            cams: self.cams.clone(),
            poses: self.poses.clone(),
            points: self.points.clone(),
            obs: self.obs.clone(),
            spans: self.spans.clone(),
            span_nominal: self.span_nominal,
            scale_weight: self.scale_weight,
            free: self.free.clone(),
        };
        plain.normal_equations(col, n)
    }
}

/// The variances of the free parameters in the datum the output is
/// expressed in: the world frame is the rigid fit of the nominal card
/// onto the solved corners, so a point's uncertainty is its uncertainty
/// relative to that fit. Computed from the pseudo-inverse of the
/// gauge-free normal matrix, `(N + E Eᵀ)⁻¹ − E Eᵀ` with `E` an orthonormal
/// basis of the six rigid motions of the world (exactly the null space of
/// `N`), carried into the card datum by the S-transformation
/// `T = I − E (Dᵀ E)⁻¹ Dᵀ` where `D` holds the same motions restricted to
/// the card's points.
fn datum_covariance(
    model: &Model,
    col: &[usize],
    n: usize,
    mut a: Vec<f64>,
    datum_points: &[usize],
) -> Option<(Vec<f64>, Vec<[[f64; 6]; 6]>)> {
    // The six rigid motions over a set of points: translations move the
    // points, rotations move the points about the origin; poses follow
    // (t by -R e, the rotation update by -R e) when `with_poses`.
    let generators = |points: &[usize], with_poses: bool| -> Vec<Vec<f64>> {
        let mut g: Vec<Vec<f64>> = Vec::with_capacity(6);
        for axis in 0..3 {
            let mut v = vec![0.0; n];
            let mut e = [0.0; 3];
            e[axis] = 1.0;
            for &p in points {
                let i = model.point_at(p) + axis;
                if col[i] != usize::MAX {
                    v[col[i]] = 1.0;
                }
            }
            if with_poses {
                for (s, pose) in model.poses.iter().enumerate() {
                    let dt = scale(mat3_vec(&pose.r, e), -1.0);
                    for j in 0..3 {
                        let i = model.pose_at(s) + 3 + j;
                        if col[i] != usize::MAX {
                            v[col[i]] = dt[j];
                        }
                    }
                }
            }
            g.push(v);
        }
        for axis in 0..3 {
            let mut v = vec![0.0; n];
            let mut e = [0.0; 3];
            e[axis] = 1.0;
            for &p in points {
                let dx = cross(e, model.points[p]);
                for j in 0..3 {
                    let i = model.point_at(p) + j;
                    if col[i] != usize::MAX {
                        v[col[i]] = dx[j];
                    }
                }
            }
            if with_poses {
                for (s, pose) in model.poses.iter().enumerate() {
                    let dtheta = scale(mat3_vec(&pose.r, e), -1.0);
                    for j in 0..3 {
                        let i = model.pose_at(s) + j;
                        if col[i] != usize::MAX {
                            v[col[i]] = dtheta[j];
                        }
                    }
                }
            }
            g.push(v);
        }
        g
    };
    let all_points: Vec<usize> = (0..model.points.len()).collect();
    // E: the null space, orthonormalised.
    let mut e_basis: Vec<Vec<f64>> = Vec::with_capacity(6);
    for mut v in generators(&all_points, true) {
        for b in &e_basis {
            let d: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            for (x, y) in v.iter_mut().zip(b) {
                *x -= d * y;
            }
        }
        let len = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if len > 1e-12 {
            for x in &mut v {
                *x /= len;
            }
            e_basis.push(v);
        }
    }
    if e_basis.len() != 6 {
        return None;
    }
    // D: the datum constraints, the motions of the card's points alone.
    let d_cols = generators(datum_points, false);
    // (N + c E Eᵀ)⁻¹ with c at the mean diagonal for conditioning; then
    // N⁺ = that − E Eᵀ / c.
    let c = (0..n).map(|i| a[i * n + i]).sum::<f64>() / n.max(1) as f64;
    for b in &e_basis {
        for i in 0..n {
            if b[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                a[i * n + j] += c * b[i] * b[j];
            }
        }
    }
    let inv = cholesky_inverse(a, n)?;
    let nplus = |i: usize, j: usize| -> f64 {
        inv[i * n + j] - e_basis.iter().map(|b| b[i] * b[j]).sum::<f64>() / c
    };
    // U = N⁺ D (n x 6), W = Dᵀ U (6 x 6), M = (Dᵀ E)⁻¹ (6 x 6).
    let u: Vec<Vec<f64>> = d_cols
        .iter()
        .map(|d| {
            let ev: Vec<f64> = e_basis
                .iter()
                .map(|b| b.iter().zip(d).map(|(x, y)| x * y).sum())
                .collect();
            (0..n)
                .map(|i| {
                    let inv_d: f64 = (0..n)
                        .filter(|&j| d[j] != 0.0)
                        .map(|j| inv[i * n + j] * d[j])
                        .sum();
                    inv_d - e_basis.iter().zip(&ev).map(|(b, v)| b[i] * v).sum::<f64>() / c
                })
                .collect()
        })
        .collect();
    let w: Vec<Vec<f64>> = (0..6)
        .map(|k| {
            (0..6)
                .map(|l| d_cols[k].iter().zip(&u[l]).map(|(x, y)| x * y).sum())
                .collect()
        })
        .collect();
    let dte: Vec<Vec<f64>> = (0..6)
        .map(|k| {
            (0..6)
                .map(|l| d_cols[k].iter().zip(&e_basis[l]).map(|(x, y)| x * y).sum())
                .collect()
        })
        .collect();
    let m = crate::linalg::inverse(&dte)?;
    // EM (n x 6), then B' = (U − EM W) Mᵀ (n x 6).
    let em: Vec<Vec<f64>> = (0..6)
        .map(|k| {
            (0..n)
                .map(|i| (0..6).map(|l| e_basis[l][i] * m[l][k]).sum())
                .collect()
        })
        .collect();
    let b_cols: Vec<Vec<f64>> = (0..6)
        .map(|k| {
            (0..n)
                .map(|i| u[k][i] - (0..6).map(|l| em[l][i] * w[l][k]).sum::<f64>())
                .collect()
        })
        .collect();
    let bm: Vec<Vec<f64>> = (0..6)
        .map(|k| {
            (0..n)
                .map(|i| (0..6).map(|l| b_cols[l][i] * m[k][l]).sum())
                .collect()
        })
        .collect();
    // Q = T N⁺ Tᵀ, element by element: A = N⁺ − EM (Dᵀ N⁺), Q = A − B' Eᵀ.
    let q = |i: usize, j: usize| -> f64 {
        let a_ij = nplus(i, j) - (0..6).map(|k| em[k][i] * u[k][j]).sum::<f64>();
        a_ij - (0..6).map(|k| bm[k][i] * e_basis[k][j]).sum::<f64>()
    };
    let diagonal: Vec<f64> = (0..n).map(|i| q(i, i)).collect();
    let poses: Vec<[[f64; 6]; 6]> = (0..model.poses.len())
        .map(|s| {
            let mut block = [[0.0; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    let (ci, cj) = (col[model.pose_at(s) + i], col[model.pose_at(s) + j]);
                    if ci != usize::MAX && cj != usize::MAX {
                        block[i][j] = q(ci, cj);
                    }
                }
            }
            block
        })
        .collect();
    Some((diagonal, poses))
}

fn rms_of(res: &[[f64; 2]]) -> f64 {
    if res.is_empty() {
        return f64::NAN;
    }
    (res.iter().map(|r| r[0] * r[0] + r[1] * r[1]).sum::<f64>() / res.len() as f64).sqrt()
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// The corners of a square of side `side` centred on `centre` in the
/// plane of `u` and `v` (unit, perpendicular), wound u then v.
pub fn square_corners(centre: [f64; 3], side: f64, u: [f64; 3], v: [f64; 3]) -> [[f64; 3]; 4] {
    let h = side * 0.5;
    [
        add(centre, add(scale(u, -h), scale(v, -h))),
        add(centre, add(scale(u, h), scale(v, -h))),
        add(centre, add(scale(u, h), scale(v, h))),
        add(centre, add(scale(u, -h), scale(v, h))),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::viewset::{BoardSpec, CornerObs, MarkerObs, Observations};

    /// A small deterministic generator (xorshift) for the synthetic field.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn uniform(&mut self, a: f64, b: f64) -> f64 {
            a + (b - a) * self.next()
        }

        /// Approximately normal: the sum of twelve uniforms.
        fn normal(&mut self, sigma: f64) -> f64 {
            ((0..12).map(|_| self.next()).sum::<f64>() - 6.0) * sigma
        }
    }

    fn look_at(centre: [f64; 3], target: [f64; 3], up: [f64; 3]) -> Rigid {
        let z = crate::linalg::normalized(sub(target, centre));
        let x = crate::linalg::normalized(cross(z, up));
        let y = cross(z, x);
        let r = [x, y, z];
        Rigid {
            r,
            t: scale(mat3_vec(&r, centre), -1.0),
        }
    }

    struct Field {
        card: Vec<[f64; 3]>,
        swatches: Vec<(u32, [[f64; 3]; 4])>,
        intr: [f64; 5],
        width: u32,
        height: u32,
        poses: Vec<Rigid>,
    }

    /// The shim's synthetic field: the card at the origin of a plane, eight
    /// swatches around it, forty frames from 1.2–2.5 m at 35–85 degrees,
    /// one camera with distortion, half a pixel of noise.
    fn field(seed: u64) -> (Field, ViewSet) {
        field_with_noise(seed, seed + 1000)
    }

    /// The field of `seed` with the pixel noise drawn from `noise_seed`.
    fn field_with_noise(seed: u64, noise_seed: u64) -> (Field, ViewSet) {
        let mut rng = Rng(seed);
        let mut noise = Rng(noise_seed);
        let spec = BoardSpec::survey_card();
        let n_card = spec.n_corners();
        let card: Vec<[f64; 3]> = (0..n_card)
            .map(|id| {
                let [x, y] = spec.corner(id).unwrap();
                [x + rng.normal(0.05e-3), y + rng.normal(0.05e-3), 0.0]
            })
            .collect();
        let places = [
            (1, -0.4, 0.2),
            (2, 0.5, -0.3),
            (5, 0.6, 0.5),
            (39, -0.5, -0.5),
            (41, 0.1, 0.7),
            (44, -0.2, 0.8),
            (49, 0.9, 0.1),
            (50, -0.8, 0.4),
        ];
        let swatches: Vec<(u32, [[f64; 3]; 4])> = places
            .iter()
            .map(|&(id, x, y)| {
                let s = if id >= 48 { 0.09 } else { 0.06 };
                let a = rng.uniform(0.0, std::f64::consts::PI);
                let u = [a.cos(), a.sin(), 0.0];
                let v = [-a.sin(), a.cos(), 0.0];
                (id, square_corners([x, y, 0.0], s, u, v))
            })
            .collect();
        let intr = [14200.0, 3080.0, 2090.0, -0.04, 0.6];
        let (width, height) = (6192u32, 4128u32);
        let seed_camera = CameraModel::pinhole(width, height, 13800.0, 13800.0, 3096.0, 2064.0);
        let mut set = ViewSet {
            cameras: vec![seed_camera],
            board: Some(Board {
                spec: spec.clone(),
                corners: Vec::new(),
            }),
            ..ViewSet::default()
        };
        let mut poses = Vec::new();
        for k in 0..90 {
            let target = [rng.uniform(-0.3, 0.9), rng.uniform(-0.3, 0.7), 0.0];
            let d = rng.uniform(1.2, 2.5);
            let el = rng.uniform(35.0, 85.0).to_radians();
            let az = rng.uniform(0.0, 2.0 * std::f64::consts::PI);
            let centre = add(
                target,
                scale([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()], d),
            );
            let roll = if k % 4 == 0 {
                rng.uniform(0.0, 2.0 * std::f64::consts::PI)
            } else {
                rng.uniform(-0.3, 0.3)
            };
            let pose = look_at(centre, target, [roll.sin(), roll.cos(), 0.0]);
            let inside = |uv: [f64; 2]| {
                uv[0] > 20.0
                    && uv[0] < f64::from(width) - 20.0
                    && uv[1] > 20.0
                    && uv[1] < f64::from(height) - 20.0
            };
            let mut obs = Observations {
                markers: Vec::new(),
                board: Vec::new(),
                blur_px: None,
            };
            for (id, x) in card.iter().enumerate() {
                if let Some((uv, _, _)) = project(&intr, pose.apply(*x))
                    && inside(uv)
                {
                    obs.board.push(CornerObs {
                        id: id as u32,
                        pixel: [uv[0] + noise.normal(0.5), uv[1] + noise.normal(0.5)],
                        fit_px: 0.1,
                    });
                }
            }
            for (id, corners) in &swatches {
                let uvs: Vec<[f64; 2]> = corners
                    .iter()
                    .filter_map(|c| project(&intr, pose.apply(*c)).map(|p| p.0))
                    .collect();
                if uvs.len() == 4 && uvs.iter().all(|uv| inside(*uv)) {
                    obs.markers.push(MarkerObs {
                        id: *id,
                        family: "5x5_100".to_string(),
                        corners: std::array::from_fn(|j| {
                            [uvs[j][0] + noise.normal(0.5), uvs[j][1] + noise.normal(0.5)]
                        }),
                        fit_px: 0.1,
                    });
                }
            }
            if obs.board.len() + 4 * obs.markers.len() >= 8 {
                let mut view = View::unposed(format!("{k:03}"), 0);
                view.observations = Some(obs);
                set.views.push(view);
                poses.push(pose);
            }
        }
        (
            Field {
                card,
                swatches,
                intr,
                width,
                height,
                poses,
            },
            set,
        )
    }

    /// The truth in the survey's world frame: the nominal card's frame with
    /// y and z flipped.
    fn truth_world(x: [f64; 3]) -> [f64; 3] {
        [x[0], -x[1], -x[2]]
    }

    #[test]
    fn the_analytic_jacobian_matches_finite_differences() {
        let (field, _) = field(11);
        let pose = field.poses[0];
        let mut model = Model {
            cams: vec![field.intr],
            poses: vec![pose],
            points: vec![field.swatches[0].1[0], field.card[5]],
            obs: vec![
                Obs {
                    slot: 0,
                    camera: 0,
                    point: 0,
                    uv: [100.0, 200.0],
                },
                Obs {
                    slot: 0,
                    camera: 0,
                    point: 1,
                    uv: [3000.0, 2000.0],
                },
            ],
            spans: vec![(0, 1)],
            span_nominal: 0.1,
            scale_weight: 1e5,
            soft_l1: f64::INFINITY,
            free: vec![true; 5 + 6 + 6],
        };
        let (col, n) = model.columns();
        let (a, g, res, span) = model.normal_equations(&col, n);
        // With unit weights N = JᵀJ and g = Jᵀr: compare against a
        // numerical J.
        let mut jac = vec![vec![0.0; n]; 5];
        let eps = [
            1e-3, 1e-4, 1e-4, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7,
            1e-7, 1e-7, 1e-7,
        ];
        // Central differences: the scale residual's second-order term
        // under its large weight would otherwise show up as a derivative.
        for i in 0..n {
            let saved = (
                model.cams.clone(),
                model.poses.clone(),
                model.points.clone(),
            );
            let mut delta = vec![0.0; n];
            delta[i] = eps[i];
            model.step(&col, &delta);
            let (res_p, span_p) = model.residuals();
            (model.cams, model.poses, model.points) = saved.clone();
            delta[i] = -eps[i];
            model.step(&col, &delta);
            let (res_m, span_m) = model.residuals();
            (model.cams, model.poses, model.points) = saved;
            for o in 0..2 {
                jac[2 * o][i] = (res_p[o][0] - res_m[o][0]) / (2.0 * eps[i]);
                jac[2 * o + 1][i] = (res_p[o][1] - res_m[o][1]) / (2.0 * eps[i]);
            }
            jac[4][i] = (span_p - span_m) / (2.0 * eps[i]);
        }
        let r_all = [res[0][0], res[0][1], res[1][0], res[1][1], span];
        for i in 0..n {
            let g_num: f64 = (0..5).map(|k| jac[k][i] * r_all[k]).sum();
            let scale = g[i].abs().max(g_num.abs()).max(1.0);
            assert!(
                (g[i] - g_num).abs() / scale < 1e-3,
                "gradient {i}: {} vs {g_num}",
                g[i]
            );
            for j in 0..n {
                let a_num: f64 = (0..5).map(|k| jac[k][i] * jac[k][j]).sum();
                let scale = a[i * n + j].abs().max(a_num.abs()).max(1.0);
                assert!(
                    (a[i * n + j] - a_num).abs() / scale < 1e-3,
                    "normal {i},{j}: {} vs {a_num}; numeric column {i}: {:?}, column {j}: {:?}",
                    a[i * n + j],
                    (0..5).map(|k| jac[k][i]).collect::<Vec<_>>(),
                    (0..5).map(|k| jac[k][j]).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn a_rigid_fit_recovers_a_rotation_of_planar_points() {
        let r = rotation_from_vector([0.3, -0.5, 0.8]);
        let t = [1.0, -2.0, 0.5];
        let src: Vec<[f64; 3]> = (0..12)
            .map(|i| [(i % 4) as f64 * 0.1, (i / 4) as f64 * 0.15, 0.0])
            .collect();
        let dst: Vec<[f64; 3]> = src.iter().map(|s| add(mat3_vec(&r, *s), t)).collect();
        let (rf, tf) = rigid_fit(&src, &dst);
        for i in 0..3 {
            for j in 0..3 {
                assert!((rf[i][j] - r[i][j]).abs() < 1e-9, "{rf:?} vs {r:?}");
            }
            assert!((tf[i] - t[i]).abs() < 1e-9, "{tf:?}");
        }
    }

    #[test]
    fn cholesky_solves_and_inverts() {
        let a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let x = cholesky_solve(a.clone(), 3, &[1.0, 2.0, 3.0]).unwrap();
        let back: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[i * 3 + j] * x[j]).sum())
            .collect();
        assert!(
            (back[0] - 1.0).abs() < 1e-12
                && (back[1] - 2.0).abs() < 1e-12
                && (back[2] - 3.0).abs() < 1e-12
        );
        let inv = cholesky_inverse(a.clone(), 3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| a[i * 3 + k] * inv[k * 3 + j]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((v - expect).abs() < 1e-12, "{i},{j}: {v}");
            }
        }
        assert!(cholesky_solve(vec![1.0, 2.0, 2.0, 1.0], 2, &[1.0, 1.0]).is_none());
    }

    #[test]
    fn a_synthetic_field_surveys_to_the_truth() {
        let (field, mut set) = field(3);
        let n_frames = set.views.len();
        assert!(n_frames >= 20, "{n_frames} frames");
        let card_frames = set
            .views
            .iter()
            .filter(|v| v.observations.as_ref().unwrap().board.len() >= 40)
            .count();
        assert!(card_frames >= 4, "{card_frames} full card frames");
        let report = survey(&mut set, &SurveyOptions::default()).unwrap();
        assert_eq!(set.world.up, [0.0, 0.0, 1.0], "the card normal is up");
        for line in &report.log {
            eprintln!("{line}");
        }
        assert!(report.rms_px < 0.8, "rms {}", report.rms_px);
        assert!(report.rejected.is_empty(), "{:?}", report.rejected);
        let unposed_points: Vec<(String, usize, usize)> = report
            .unposed
            .iter()
            .map(|id| {
                let obs = set
                    .views
                    .iter()
                    .find(|v| &v.id == id)
                    .unwrap()
                    .observations
                    .as_ref()
                    .unwrap();
                (id.clone(), obs.board.len(), obs.markers.len())
            })
            .collect();
        // A frame showing two swatches and nothing else may stay unposed;
        // every frame with more is posed.
        assert!(
            unposed_points
                .iter()
                .all(|(_, card, swatches)| *card == 0 && *swatches <= 2),
            "unposed (id, card corners, swatches) {unposed_points:?}"
        );
        assert!(
            report.frames.len() + unposed_points.len() == n_frames
                && report.frames.len() >= n_frames - 4
        );
        // The camera.
        let cam = &set.cameras[0];
        assert!(
            (cam.fx - field.intr[0]).abs() < 0.003 * field.intr[0],
            "f {}",
            cam.fx
        );
        // The principal point of a planar field is the weakly known
        // parameter: a shift trades against every pose.
        assert!(
            (cam.cx - field.intr[1]).abs() < 20.0 && (cam.cy - field.intr[2]).abs() < 20.0,
            "pp {} {}",
            cam.cx,
            cam.cy
        );
        assert!(
            report.cameras[0].cx_std > 1.0 && report.cameras[0].cx_std < 30.0,
            "cx std {}",
            report.cameras[0].cx_std
        );
        if let Distortion::Radial { k, .. } = &cam.distortion {
            assert!((k[0] - field.intr[3]).abs() < 0.01, "k1 {}", k[0]);
            // k2 acts at r^4 and the observations sit well inside the
            // frame: a tenth of it is under a pixel at the frame corner.
            assert!((k[1] - field.intr[4]).abs() < 0.2, "k2 {}", k[1]);
        } else {
            panic!("no distortion solved");
        }
        assert!(
            report.cameras[0].f_std > 0.0 && report.cameras[0].f_std < 20.0,
            "f std {}",
            report.cameras[0].f_std
        );
        // The card: every corner solved, planar, spans right, sigmas small.
        let board = set.board.as_ref().unwrap();
        assert_eq!(board.corners.len(), field.card.len());
        assert!(
            report.card_planarity_mm < 0.05,
            "planarity {}",
            report.card_planarity_mm
        );
        assert!((report.span_across_mm - report.span_across_nominal_mm).abs() < 0.02);
        assert!(
            (report.span_along_mm - report.span_along_nominal_mm).abs() < 0.1,
            "{} vs {}",
            report.span_along_mm,
            report.span_along_nominal_mm
        );
        let mut sigmas = Vec::new();
        for c in &board.corners {
            let truth = truth_world(field.card[c.id as usize]);
            let d = norm(sub(c.position, truth));
            assert!(d < 0.15e-3, "card corner {}: {d} m off", c.id);
            assert!(c.sigma_m > 0.0, "card corner {} sigma {}", c.id, c.sigma_m);
            sigmas.push(c.sigma_m * 1e3);
        }
        sigmas.sort_by(f64::total_cmp);
        assert!(
            sigmas[sigmas.len() / 2] < 0.06 && sigmas[sigmas.len() - 1] < 0.12,
            "card sigmas median {} max {}",
            sigmas[sigmas.len() / 2],
            sigmas[sigmas.len() - 1]
        );
        // The swatches: all eight, corners within 0.3 mm, sides right.
        let missing: Vec<(u32, usize, usize)> = field
            .swatches
            .iter()
            .filter(|(id, _)| !set.markers.iter().any(|m| m.id == *id))
            .map(|(id, _)| {
                let seen: Vec<&View> = set
                    .views
                    .iter()
                    .filter(|v| {
                        v.observations
                            .as_ref()
                            .unwrap()
                            .markers
                            .iter()
                            .any(|m| m.id == *id)
                    })
                    .collect();
                (
                    *id,
                    seen.len(),
                    seen.iter().filter(|v| v.pose().is_some()).count(),
                )
            })
            .collect();
        // A swatch seen from one frame is known only along a ray and
        // stays out; every other one is solved.
        assert!(
            missing.iter().all(|(_, _, posed)| *posed < 2),
            "missing (id, seen by, posed of those) {missing:?}"
        );
        assert!(set.markers.len() >= 7, "{} swatches", set.markers.len());
        for m in &set.markers {
            let (_, truth) = field.swatches.iter().find(|s| s.0 == m.id).unwrap();
            let sigma = report
                .swatches
                .iter()
                .find(|s| s.id == m.id)
                .unwrap()
                .sigma_mm
                * 1e-3;
            for j in 0..4 {
                let d = norm(sub(m.corners[j], truth_world(truth[j])));
                assert!(
                    d < (3.0 * sigma).max(0.3e-3),
                    "swatch {} corner {j}: {d} m off, sigma {sigma}",
                    m.id
                );
            }
            let side = if m.id >= 48 { 0.09 } else { 0.06 };
            assert!(
                (m.size_m - side).abs() < 0.2e-3,
                "swatch {} side {}",
                m.id,
                m.size_m
            );
        }
        for s in &report.swatches {
            assert!(
                s.sigma_mm > 0.0 && s.sigma_mm < 1.0,
                "swatch {} sigma {}",
                s.id,
                s.sigma_mm
            );
        }
        // The poses: within 2 mm and 0.02 degrees of the truth. The
        // position tolerance is the principal point's: a shift of it tilts
        // every camera a little and moves its eye to keep the planar field
        // where it was.
        for (view, truth) in set.views.iter().zip(&field.poses) {
            if view.pose().is_none() {
                continue;
            }
            let truth_c2w = Rigid {
                r: mat3_mul(
                    &truth.r,
                    &[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                ),
                t: truth.t,
            }
            .camera_to_world();
            let (angle, dist) = crate::pnp::pose_difference(view.pose().unwrap(), &truth_c2w);
            let angle = angle.to_degrees();
            // Each pose is judged against its own reported uncertainty: a
            // frame on two swatches (eight points over six centimetres at
            // two metres) is posed but hardly constrained, and says so.
            let f = report.frames.iter().find(|f| f.id == view.id).unwrap();
            assert!(
                dist < 3.0 * f.position_sigma_mm * 1e-3 + 0.3e-3
                    && angle < 3.0 * f.angle_sigma_deg + 0.01,
                "view {}: {dist} m (sigma {} mm), {angle} deg (sigma {} deg)",
                view.id,
                f.position_sigma_mm,
                f.angle_sigma_deg
            );
            if f.card_corners >= 40 {
                assert!(
                    f.position_sigma_mm < 10.0 && f.angle_sigma_deg < 0.3,
                    "card frame {}: sigma {} mm, {} deg",
                    view.id,
                    f.position_sigma_mm,
                    f.angle_sigma_deg
                );
            }
            assert!(view.tags.iter().any(|t| t == "solved:survey"));
        }
        assert!(
            set.views
                .iter()
                .filter(|v| v.pose().is_none())
                .all(|v| !v.tags.iter().any(|t| t == "solved:survey"))
        );
        // The report's map carries the numbers.
        let map = report.to_f64_map();
        assert_eq!(map["frames_posed"], report.frames.len() as f64);
        assert!(map.contains_key("camera.0.f") && map.contains_key("swatch.49.side_mm"));
        assert!((field.width, field.height) == (6192, 4128));
        // Surveying again from the solved state converges at once.
        let again = survey(&mut set, &SurveyOptions::default()).unwrap();
        assert!((again.rms_px - report.rms_px).abs() < 0.05);
    }

    /// The reported sigmas are checked against the scatter of the solved
    /// positions over six draws of the pixel noise on one geometry: they
    /// are uncertainties relative to the card's frame, the frame the set
    /// is expressed in.
    #[test]
    fn reported_uncertainties_match_the_scatter_over_noise() {
        let mut card_runs: Vec<Vec<[f64; 3]>> = Vec::new();
        let mut swatch_runs: Vec<Vec<[f64; 3]>> = Vec::new();
        let mut f_runs = Vec::new();
        let mut sig_card = 0.0;
        let mut sig_sw = 0.0;
        let mut f_std = 0.0;
        for noise in 0..6 {
            let (_, mut set) = field_with_noise(3, 500 + noise);
            let report = survey(&mut set, &SurveyOptions::default()).unwrap();
            let board = set.board.as_ref().unwrap();
            card_runs.push(board.corners.iter().map(|c| c.position).collect());
            swatch_runs.push(set.markers.iter().flat_map(|m| m.corners).collect());
            f_runs.push(set.cameras[0].fx);
            sig_card =
                board.corners.iter().map(|c| c.sigma_m).sum::<f64>() / board.corners.len() as f64;
            sig_sw = report.swatches.iter().map(|s| s.sigma_mm).sum::<f64>()
                / report.swatches.len() as f64
                * 1e-3;
            f_std = report.cameras[0].f_std;
        }
        let scatter = |runs: &Vec<Vec<[f64; 3]>>| -> f64 {
            let n = runs[0].len();
            let mut total = 0.0;
            for p in 0..n {
                let mean: [f64; 3] = std::array::from_fn(|d| {
                    runs.iter().map(|r| r[p][d]).sum::<f64>() / runs.len() as f64
                });
                total += runs
                    .iter()
                    .map(|r| norm(sub(r[p], mean)).powi(2))
                    .sum::<f64>()
                    / (runs.len() - 1) as f64;
            }
            (total / n as f64).sqrt()
        };
        let f_mean = f_runs.iter().sum::<f64>() / f_runs.len() as f64;
        let f_scatter = (f_runs.iter().map(|f| (f - f_mean).powi(2)).sum::<f64>()
            / (f_runs.len() - 1) as f64)
            .sqrt();
        let (card, swatches) = (scatter(&card_runs), scatter(&swatch_runs));
        eprintln!(
            "card: scatter {:.4} mm vs sigma {:.4} mm; swatches: scatter {:.4} mm vs sigma {:.4} mm; f: scatter {:.2} vs std {:.2}",
            card * 1e3,
            sig_card * 1e3,
            swatches * 1e3,
            sig_sw * 1e3,
            f_scatter,
            f_std
        );
        // Six draws estimate a scatter to about 30 %.
        assert!(
            card / sig_card > 0.6 && card / sig_card < 1.6,
            "card {card} vs {sig_card}"
        );
        assert!(
            swatches / sig_sw > 0.6 && swatches / sig_sw < 1.6,
            "swatches {swatches} vs {sig_sw}"
        );
        assert!(
            f_scatter / f_std > 0.4 && f_scatter / f_std < 2.5,
            "f {f_scatter} vs {f_std}"
        );
    }

    #[test]
    fn a_frame_with_wrong_observations_is_rejected() {
        let (_, mut set) = field(5);
        // Scramble one frame: swap its swatch corners around.
        let victim = set
            .views
            .iter()
            .position(|v| v.observations.as_ref().unwrap().markers.len() >= 2)
            .unwrap();
        let obs = set.views[victim].observations.as_mut().unwrap();
        for m in &mut obs.markers {
            m.corners.rotate_left(1);
            for c in &mut m.corners {
                c[0] += 40.0;
            }
        }
        // Card corners pushed apart by parity: no pose fits them.
        for c in &mut obs.board {
            c.pixel[1] += if c.id % 2 == 0 { 25.0 } else { -25.0 };
            c.pixel[0] += if (c.id / 11) % 2 == 0 { 20.0 } else { -20.0 };
        }
        let id = set.views[victim].id.clone();
        let report = survey(&mut set, &SurveyOptions::default()).unwrap();
        assert!(
            report.rejected.contains(&id) || report.unposed.contains(&id),
            "{id} survived: rejected {:?}, unposed {:?}",
            report.rejected,
            report.unposed
        );
        assert!(set.views[victim].pose().is_none());
        assert!(report.rms_px < 0.8, "rms {}", report.rms_px);
    }

    #[test]
    fn a_set_without_detections_is_refused() {
        let mut set = ViewSet {
            cameras: vec![CameraModel::pinhole(100, 100, 100.0, 100.0, 50.0, 50.0)],
            views: vec![View::unposed("a", 0)],
            ..ViewSet::default()
        };
        assert!(
            survey(&mut set, &SurveyOptions::default())
                .unwrap_err()
                .contains("board")
        );
        set.board = Some(Board {
            spec: BoardSpec::survey_card(),
            corners: Vec::new(),
        });
        assert!(
            survey(&mut set, &SurveyOptions::default())
                .unwrap_err()
                .contains("view-detect")
        );
    }
}
