//! Comparing a model against a view's depth map.
//!
//! For every measured pixel, the scan says where the surface is along the
//! pixel's ray. The model is asked the same question: its occupancy is
//! marched along the ray within a band around the scan depth and the
//! nearest inside/outside transition is bisected. The signed difference,
//! model depth minus scan depth, is the residual; a ray with no transition
//! in the band is a miss (the model has no surface where the scan has one).

use rayon::prelude::*;
use volumetric_abi::viewset::{CameraModel, View};

use crate::image::{Depth, Mask, Rgb};

/// How the model surface is searched around each scan depth.
#[derive(Clone, Copy, Debug)]
pub struct Search {
    /// Half-width of the depth band searched, in metres.
    pub band: f64,
    /// Marching step within the band, in metres.
    pub step: f64,
    /// Pixel lattice: every `stride`th pixel in x and y.
    pub stride: u32,
}

impl Default for Search {
    fn default() -> Self {
        Self {
            band: 0.05,
            step: 0.002,
            stride: 2,
        }
    }
}

/// Summary of one view's residuals.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResidualStats {
    /// Lattice pixels with a scan depth (and inside the mask, when given).
    pub pixels: usize,
    /// Of those, rays where the model has a surface within the band.
    pub hits: usize,
    /// `hits / pixels`.
    pub coverage: f64,
    pub median_abs: f64,
    pub p90_abs: f64,
    /// Signed mean: positive when the model surface sits behind the scan's.
    pub mean: f64,
    pub rms: f64,
}

impl ResidualStats {
    /// Summarises signed residuals of the hits among `pixels` rays.
    pub fn from_residuals(pixels: usize, residuals: &[f32]) -> Self {
        let hits = residuals.len();
        if hits == 0 {
            return Self {
                pixels,
                ..Self::default()
            };
        }
        let mut abs: Vec<f32> = residuals.iter().map(|r| r.abs()).collect();
        abs.sort_by(f32::total_cmp);
        let quantile = |q: f64| f64::from(abs[((abs.len() - 1) as f64 * q).round() as usize]);
        let mean = residuals.iter().map(|r| f64::from(*r)).sum::<f64>() / hits as f64;
        let rms = (residuals
            .iter()
            .map(|r| f64::from(*r) * f64::from(*r))
            .sum::<f64>()
            / hits as f64)
            .sqrt();
        Self {
            pixels,
            hits,
            coverage: hits as f64 / pixels.max(1) as f64,
            median_abs: quantile(0.5),
            p90_abs: quantile(0.9),
            mean,
            rms,
        }
    }
}

/// One view's residual field and its summary.
#[derive(Clone, Debug)]
pub struct Residual {
    pub view_id: String,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    /// Per pixel at the lattice points: the signed residual in metres,
    /// `INFINITY` for a miss, `NAN` where there is no scan depth or the
    /// pixel is off the lattice.
    pub values: Vec<f32>,
    pub stats: ResidualStats,
}

/// Compares `inside` (the model's occupancy at a world point) against the
/// view's depth map.
pub fn depth_residual(
    view: &View,
    camera: &CameraModel,
    depth: &Depth,
    mask: Option<&Mask>,
    search: &Search,
    inside: &(dyn Fn([f64; 3]) -> bool + Sync),
) -> Residual {
    let (width, height) = (depth.width, depth.height);
    let stride = search.stride.max(1);
    let origin = view.position();
    let forward = view.forward();

    let rows: Vec<Vec<(u32, f32)>> = (0..height)
        .step_by(stride as usize)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::new();
            for x in (0..width).step_by(stride as usize) {
                let scan = f64::from(depth.get(x, y));
                if !scan.is_finite() || scan <= 0.0 {
                    continue;
                }
                if let Some(mask) = mask
                    && !mask.get(x, y)
                {
                    continue;
                }
                let pixel = [f64::from(x) + 0.5, f64::from(y) + 0.5];
                let dir = view.ray(camera, pixel);
                let cos = dir[0] * forward[0] + dir[1] * forward[1] + dir[2] * forward[2];
                if cos <= 1e-6 {
                    continue;
                }
                let at = |z: f64| -> [f64; 3] {
                    let t = z / cos;
                    [
                        origin[0] + dir[0] * t,
                        origin[1] + dir[1] * t,
                        origin[2] + dir[2] * t,
                    ]
                };
                let residual = surface_near(scan, search, &|z| inside(at(z)))
                    .map_or(f32::INFINITY, |z| (z - scan) as f32);
                row.push((x, residual));
            }
            row
        })
        .collect();

    let mut values = vec![f32::NAN; (width * height) as usize];
    let mut pixels = 0usize;
    let mut hits = Vec::new();
    for (row, y) in rows.iter().zip((0..height).step_by(stride as usize)) {
        for &(x, residual) in row {
            values[(y * width + x) as usize] = residual;
            pixels += 1;
            if residual.is_finite() {
                hits.push(residual);
            }
        }
    }
    Residual {
        view_id: view.id.clone(),
        width,
        height,
        stride,
        values,
        stats: ResidualStats::from_residuals(pixels, &hits),
    }
}

/// The model's surface depth nearest to `scan` within the band: marched at
/// `search.step`, then bisected.
fn surface_near(scan: f64, search: &Search, inside: &dyn Fn(f64) -> bool) -> Option<f64> {
    let step = search.step.max(1e-6);
    let start = (scan - search.band).max(step);
    let end = scan + search.band;
    let mut best: Option<(f64, f64)> = None;
    let mut z = start;
    let mut was_inside = inside(z);
    while z < end {
        let next = (z + step).min(end);
        let is_inside = inside(next);
        if is_inside != was_inside {
            let mid = (z + next) * 0.5;
            if best.is_none_or(|(a, b)| (mid - scan).abs() < ((a + b) * 0.5 - scan).abs()) {
                best = Some((z, next));
            }
        }
        was_inside = is_inside;
        z = next;
    }
    let (mut lo, mut hi) = best?;
    let lo_inside = inside(lo);
    for _ in 0..8 {
        let mid = (lo + hi) * 0.5;
        if inside(mid) == lo_inside {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) * 0.5)
}

/// The residual field as a picture: blue where the model surface is in
/// front of the scan's, red where behind, saturating at `scale_m`; grey
/// for a miss, black where there is no depth.
pub fn residual_image(residual: &Residual, scale_m: f32) -> Rgb {
    let mut rgb = Rgb::new(residual.width, residual.height);
    let stride = residual.stride;
    for y in (0..residual.height).step_by(stride as usize) {
        for x in (0..residual.width).step_by(stride as usize) {
            let value = residual.values[(y * residual.width + x) as usize];
            let colour = if value.is_nan() {
                [0, 0, 0]
            } else if value.is_infinite() {
                [90, 90, 90]
            } else {
                let t = (value / scale_m).clamp(-1.0, 1.0);
                let lerp = |a: f32, b: f32| (a + (b - a) * t.abs()) as u8;
                if t < 0.0 {
                    [lerp(235.0, 40.0), lerp(235.0, 90.0), lerp(235.0, 255.0)]
                } else {
                    [lerp(235.0, 255.0), lerp(235.0, 60.0), lerp(235.0, 40.0)]
                }
            };
            for dy in 0..stride {
                for dx in 0..stride {
                    let (px, py) = (x + dx, y + dy);
                    if px < residual.width && py < residual.height {
                        rgb.set(px, py, colour);
                    }
                }
            }
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at the origin looking along +z at a plane z = 2 (the model)
    /// while the scan says z = 2.01: every ray reports the model 1 cm in
    /// front, except masked and unmeasured pixels.
    #[test]
    fn plane_residuals_are_the_depth_offset() {
        let camera = CameraModel::pinhole(8, 6, 10.0, 10.0, 4.0, 3.0);
        let view = View::posed(
            "v",
            0,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let mut metres = vec![2.01f32; 48];
        metres[0] = f32::NAN;
        let depth = Depth {
            width: 8,
            height: 6,
            metres,
        };
        let mut inside_mask = vec![true; 48];
        inside_mask[1] = false;
        let mask = Mask {
            width: 8,
            height: 6,
            inside: inside_mask,
        };
        let search = Search {
            band: 0.05,
            step: 0.005,
            stride: 1,
        };
        let residual = depth_residual(&view, &camera, &depth, Some(&mask), &search, &|p| {
            p[2] > 2.0
        });
        assert_eq!(residual.stats.pixels, 46);
        assert_eq!(residual.stats.hits, 46);
        assert!((residual.stats.coverage - 1.0).abs() < 1e-12);
        assert!(
            (residual.stats.mean + 0.01).abs() < 1e-4,
            "{:?}",
            residual.stats
        );
        assert!(residual.stats.median_abs > 0.0099 && residual.stats.p90_abs < 0.0101);
        assert!(residual.values[0].is_nan() && residual.values[1].is_nan());

        // A model far from the scan is a miss on every ray.
        let miss = depth_residual(&view, &camera, &depth, None, &search, &|p| p[2] > 5.0);
        assert_eq!(miss.stats.hits, 0);
        assert_eq!(miss.stats.pixels, 47);
        assert!(miss.values[2].is_infinite());

        let picture = residual_image(&residual, 0.02);
        assert_eq!(picture.get(0, 0), [0, 0, 0]);
        let front = picture.get(4, 3);
        assert!(front[2] > front[0], "in front should read blue: {front:?}");
        let miss_picture = residual_image(&miss, 0.02);
        assert_eq!(miss_picture.get(4, 3), [90, 90, 90]);
    }

    #[test]
    fn stats_quantiles_and_signs() {
        let stats = ResidualStats::from_residuals(10, &[0.01, -0.02, 0.03, 0.04, -0.05]);
        assert_eq!(stats.hits, 5);
        assert!((stats.coverage - 0.5).abs() < 1e-12);
        assert!((stats.median_abs - 0.03).abs() < 1e-6);
        assert!((stats.p90_abs - 0.05).abs() < 1e-6);
        assert!((stats.mean - 0.002).abs() < 1e-6);
        assert_eq!(ResidualStats::from_residuals(3, &[]).coverage, 0.0);
    }
}
