//! Edge blur from the markers a picture shows: the width of the step
//! across each marker edge, which is motion blur and defocus measured on
//! the frame itself.
//!
//! Along every detected marker side long enough, intensity profiles
//! across the edge are fitted with an error-function step; the step's
//! sigma in pixels is the blur. Sides are binned by their direction on
//! the picture, since motion smears one axis more than the other.

use crate::detect::Detection;
use crate::gray::Gray;

/// Blur across the picture's horizontal and vertical marker edges,
/// pixels, as the medians over all profiles that fitted; `None` where no
/// edge of that direction was measurable.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EdgeBlur {
    /// Across edges that run left to right (blur along the picture's y).
    pub horizontal: Option<f64>,
    /// Across edges that run top to bottom (blur along x).
    pub vertical: Option<f64>,
    pub profiles: usize,
}

impl EdgeBlur {
    /// The worse of the two axes.
    pub fn worst(&self) -> Option<f64> {
        match (self.horizontal, self.vertical) {
            (Some(h), Some(v)) => Some(h.max(v)),
            (h, v) => h.or(v),
        }
    }
}

/// Measures the blur across the detections' edges.
pub fn edge_blur(gray: &Gray, detections: &[Detection]) -> EdgeBlur {
    let (w, h) = (f64::from(gray.width), f64::from(gray.height));
    let mut along_x = Vec::new();
    let mut along_y = Vec::new();
    const STEP: f64 = 0.5;
    for d in detections {
        for j in 0..4 {
            let (a, b) = (d.corners[j], d.corners[(j + 1) % 4]);
            let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            if len < 24.0 {
                continue;
            }
            let t = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
            let n = [-t[1], t[0]];
            // The profile must stay within the marker's border cell on
            // the inside and its quiet zone on the outside.
            let cell = len / f64::from(cells_of(d.family));
            if cell < 6.0 {
                continue;
            }
            let reach = (0.4 * cell).min(30.0);
            let steps = (reach / STEP) as i64;
            for frac in [0.3, 0.5, 0.7] {
                let p = [a[0] + t[0] * frac * len, a[1] + t[1] * frac * len];
                let ends = [
                    [p[0] - n[0] * reach, p[1] - n[1] * reach],
                    [p[0] + n[0] * reach, p[1] + n[1] * reach],
                ];
                if ends
                    .iter()
                    .any(|e| e[0] < 2.0 || e[1] < 2.0 || e[0] > w - 2.0 || e[1] > h - 2.0)
                {
                    continue;
                }
                let profile: Vec<f64> = (-steps..=steps)
                    .map(|s| {
                        let dd = s as f64 * STEP;
                        gray.sample(p[0] + n[0] * dd, p[1] + n[1] * dd)
                    })
                    .collect();
                if let Some(sigma) = step_sigma(&profile, STEP) {
                    if t[0].abs() > t[1].abs() {
                        along_x.push(sigma);
                    } else {
                        along_y.push(sigma);
                    }
                }
            }
        }
    }
    let profiles = along_x.len() + along_y.len();
    EdgeBlur {
        horizontal: median(&mut along_x),
        vertical: median(&mut along_y),
        profiles,
    }
}

/// Cells across a marker of the family, border included.
fn cells_of(family: &str) -> u32 {
    crate::dict::Dictionary::by_name(family).map_or(7, |d| d.size + 2)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    })
}

/// Fits `a + b·Φ((x − x0) / s)` to a profile sampled every `step`
/// pixels: `x0` from the gradient's centroid, `s` by a golden-section
/// search with `a`, `b` solved linearly at each `s`. `None` without a
/// clear step (contrast under 40 levels).
pub fn step_sigma(profile: &[f64], step: f64) -> Option<f64> {
    let n = profile.len();
    if n < 8 {
        return None;
    }
    let grad: Vec<f64> = (0..n)
        .map(|j| {
            if j == 0 || j + 1 == n {
                0.0
            } else {
                (profile[j + 1] - profile[j - 1]).abs()
            }
        })
        .collect();
    let gmax = grad.iter().cloned().fold(0.0, f64::max);
    if gmax < 4.0 {
        return None;
    }
    let (mut wsum, mut xsum) = (0.0, 0.0);
    for (j, g) in grad.iter().enumerate() {
        let w = (g - 0.2 * gmax).max(0.0);
        wsum += w;
        xsum += w * j as f64;
    }
    if wsum <= 0.0 {
        return None;
    }
    let x0 = xsum / wsum;
    let xs: Vec<f64> = (0..n).map(|j| (j as f64 - x0) * step).collect();
    // Residual of the best linear (a, b) for a given sigma.
    let residual = |s: f64| -> (f64, f64) {
        let phi: Vec<f64> = xs
            .iter()
            .map(|&x| 0.5 * (1.0 + erf(x / (s * std::f64::consts::SQRT_2))))
            .collect();
        let m = n as f64;
        let (sp, sy, spp, spy) = phi
            .iter()
            .zip(profile)
            .fold((0.0, 0.0, 0.0, 0.0), |acc, (p, y)| {
                (acc.0 + p, acc.1 + y, acc.2 + p * p, acc.3 + p * y)
            });
        let det = m * spp - sp * sp;
        if det.abs() < 1e-12 {
            return (f64::INFINITY, 0.0);
        }
        let b = (m * spy - sp * sy) / det;
        let a = (sy - b * sp) / m;
        let r: f64 = phi
            .iter()
            .zip(profile)
            .map(|(p, y)| (a + b * p - y).powi(2))
            .sum();
        (r, b)
    };
    let (mut lo, mut hi) = (0.2, (n as f64 * step) * 0.25);
    let g = (5f64.sqrt() - 1.0) * 0.5;
    let (mut x1, mut x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
    let (mut f1, mut f2) = (residual(x1).0, residual(x2).0);
    for _ in 0..40 {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - g * (hi - lo);
            f1 = residual(x1).0;
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + g * (hi - lo);
            f2 = residual(x2).0;
        }
        if hi - lo < 0.01 {
            break;
        }
    }
    let s = (lo + hi) * 0.5;
    let (_, b) = residual(s);
    if b.abs() < 40.0 {
        return None;
    }
    Some(s)
}

/// The error function (Abramowitz and Stegun 7.1.26, within 1.5e-7).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Render, render, square_marker};
    use crate::detect::{DetectParams, detect};
    use crate::dict::Dictionary;
    use volumetric_abi::viewset::{CameraModel, View};

    #[test]
    fn a_synthetic_step_gives_back_its_sigma() {
        assert!((erf(0.5) - 0.520499878).abs() < 2e-7);
        assert!((erf(-1.5) + 0.966105146).abs() < 2e-7);
        for sigma in [0.6, 1.5, 3.0] {
            let profile: Vec<f64> = (0..81)
                .map(|j| {
                    let x = (j as f64 - 40.3) * 0.5;
                    30.0 + 150.0 * 0.5 * (1.0 + erf(x / (sigma * std::f64::consts::SQRT_2)))
                })
                .collect();
            let got = step_sigma(&profile, 0.5).unwrap();
            assert!((got - sigma).abs() < 0.05, "{sigma}: {got}");
        }
        assert!(step_sigma(&vec![100.0; 40], 0.5).is_none());
        assert!(
            step_sigma(
                &(0..40).map(|j| 100.0 + f64::from(j)).collect::<Vec<_>>(),
                0.5
            )
            .is_none()
        );
    }

    #[test]
    fn blur_is_measured_from_rendered_marker_edges() {
        let camera = CameraModel::pinhole(800, 600, 700.0, 700.0, 400.0, 300.0);
        let view = View::posed(
            "top",
            0,
            [
                1.0, 0.0, 0.0, 0.0, //
                0.0, -1.0, 0.0, 0.0, //
                0.0, 0.0, -1.0, 1.0, //
            ],
        );
        let markers = vec![
            square_marker(3, [-0.35, 0.3, 0.0], 0.2, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
            square_marker(7, [0.05, 0.25, 0.0], 0.2, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ];
        let dict = Dictionary::aruco_5x5_100();
        // Rendered edges are anti-aliased over a pixel (about 0.3 px of
        // sigma); blur adds in quadrature.
        for sigma in [0.0, 1.5, 3.0] {
            let picture = render(
                &camera,
                &view,
                &markers,
                &dict,
                &Render {
                    blur_sigma: sigma,
                    ..Render::default()
                },
            );
            let found = detect(&picture, &[&dict], &DetectParams::default());
            assert_eq!(found.len(), 2);
            let blur = edge_blur(&picture, &found);
            assert!(blur.profiles >= 20, "{blur:?}");
            let expected = (sigma * sigma + 0.09f64).sqrt();
            for got in [
                blur.horizontal.unwrap(),
                blur.vertical.unwrap(),
                blur.worst().unwrap(),
            ] {
                assert!((got - expected).abs() < 0.25, "sigma {sigma}: {blur:?}");
            }
        }
        assert_eq!(edge_blur(&Gray::new(8, 8), &[]).worst(), None);
    }
}
