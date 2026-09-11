//! 8-bit luma pictures: sampling, an integral image, and the local-mean
//! threshold the detector binarises with.
//!
//! Coordinates are continuous with the picture spanning `0..width` and
//! `0..height`: pixel `(i, j)` covers `[i, i+1) x [j, j+1)` and its centre
//! is `(i + 0.5, j + 0.5)`. That is the view set's pixel convention (the
//! principal point of a `width` picture centred exactly is `width / 2`);
//! OpenCV's integer-centred corners are half a pixel less on both axes.

#[derive(Clone, Debug, PartialEq)]
pub struct Gray {
    pub width: u32,
    pub height: u32,
    /// Row-major luma, `width * height` bytes.
    pub pixels: Vec<u8>,
}

impl Gray {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width as usize) * (height as usize)],
        }
    }

    /// Rec. 601 luma of packed RGB8 pixels.
    pub fn from_rgb8(width: u32, height: u32, rgb: &[u8]) -> Self {
        assert_eq!(rgb.len(), (width as usize) * (height as usize) * 3);
        let pixels = rgb
            .chunks_exact(3)
            .map(|p| {
                (0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2]))
                    .round()
                    .clamp(0.0, 255.0) as u8
            })
            .collect();
        Self {
            width,
            height,
            pixels,
        }
    }

    pub fn get(&self, x: u32, y: u32) -> u8 {
        self.pixels[(y as usize) * (self.width as usize) + x as usize]
    }

    pub fn set(&mut self, x: u32, y: u32, value: u8) {
        self.pixels[(y as usize) * (self.width as usize) + x as usize] = value;
    }

    /// The pixel at integer coordinates, clamped to the picture.
    pub fn at(&self, x: i64, y: i64) -> f64 {
        let x = x.clamp(0, i64::from(self.width) - 1) as u32;
        let y = y.clamp(0, i64::from(self.height) - 1) as u32;
        f64::from(self.get(x, y))
    }

    /// Bilinear sample at a continuous position (pixel centres at `+0.5`).
    pub fn sample(&self, x: f64, y: f64) -> f64 {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self.at(x0, y0);
        let b = self.at(x0 + 1, y0);
        let c = self.at(x0, y0 + 1);
        let d = self.at(x0 + 1, y0 + 1);
        (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty
    }

    /// The picture reduced by an integer factor, each pixel the mean of
    /// a `factor x factor` block (a trailing partial block is dropped).
    /// Continuous coordinates scale exactly: `x` here is `x * factor` in
    /// the original.
    pub fn downsampled(&self, factor: u32) -> Gray {
        let factor = factor.max(1);
        let (w, h) = (self.width / factor, self.height / factor);
        let mut out = Gray::new(w, h);
        let area = f64::from(factor * factor);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0u32;
                for dy in 0..factor {
                    let row = ((y * factor + dy) * self.width + x * factor) as usize;
                    sum += self.pixels[row..row + factor as usize]
                        .iter()
                        .map(|&v| u32::from(v))
                        .sum::<u32>();
                }
                out.pixels[(y * w + x) as usize] = (f64::from(sum) / area).round() as u8;
            }
        }
        out
    }

    pub fn integral(&self) -> Integral {
        let w = self.width as usize + 1;
        let h = self.height as usize + 1;
        let mut sums = vec![0u64; w * h];
        for y in 1..h {
            let mut row = 0u64;
            for x in 1..w {
                row += u64::from(self.pixels[(y - 1) * (w - 1) + (x - 1)]);
                sums[y * w + x] = sums[(y - 1) * w + x] + row;
            }
        }
        Integral {
            width: self.width,
            height: self.height,
            sums,
        }
    }

    /// Marks the pixels darker than their `window`-wide local mean by more
    /// than `constant` (OpenCV's mean-C adaptive threshold, inverted so the
    /// dark ink of a marker is the foreground).
    pub fn threshold_local(&self, window: u32, constant: i32) -> Binary {
        let integral = self.integral();
        let half = i64::from(window.max(1) / 2);
        let mut bits = vec![false; self.pixels.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                let x0 = (i64::from(x) - half).max(0);
                let y0 = (i64::from(y) - half).max(0);
                let x1 = (i64::from(x) + half + 1).min(i64::from(self.width));
                let y1 = (i64::from(y) + half + 1).min(i64::from(self.height));
                let area = ((x1 - x0) * (y1 - y0)) as f64;
                let mean = integral.sum(x0 as u32, y0 as u32, x1 as u32, y1 as u32) as f64 / area;
                bits[(y as usize) * (self.width as usize) + x as usize] =
                    f64::from(self.get(x, y)) <= mean - f64::from(constant);
            }
        }
        Binary {
            width: self.width,
            height: self.height,
            bits,
        }
    }

    /// Separable Gaussian blur with standard deviation `sigma` pixels.
    pub fn blurred(&self, sigma: f64) -> Gray {
        if sigma <= 0.0 {
            return self.clone();
        }
        let radius = (sigma * 3.0).ceil() as i64;
        let kernel: Vec<f64> = (-radius..=radius)
            .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp())
            .collect();
        let norm: f64 = kernel.iter().sum();
        let (w, h) = (self.width as i64, self.height as i64);
        let mut tmp = vec![0f64; self.pixels.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    acc += weight * self.at(x + k as i64 - radius, y);
                }
                tmp[(y * w + x) as usize] = acc / norm;
            }
        }
        let mut out = Gray::new(self.width, self.height);
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    let yy = (y + k as i64 - radius).clamp(0, h - 1);
                    acc += weight * tmp[(yy * w + x) as usize];
                }
                out.pixels[(y * w + x) as usize] = (acc / norm).round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }
}

/// Summed-area table: `sum(x0, y0, x1, y1)` is the total over the
/// half-open pixel rectangle.
pub struct Integral {
    pub width: u32,
    pub height: u32,
    sums: Vec<u64>,
}

impl Integral {
    pub fn sum(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> u64 {
        let w = self.width as usize + 1;
        let at = |x: u32, y: u32| self.sums[(y as usize) * w + x as usize];
        at(x1, y1) + at(x0, y0) - at(x1, y0) - at(x0, y1)
    }
}

/// A binarised picture: `true` where the foreground (dark ink) is.
#[derive(Clone, Debug)]
pub struct Binary {
    pub width: u32,
    pub height: u32,
    pub bits: Vec<bool>,
}

impl Binary {
    pub fn get(&self, x: u32, y: u32) -> bool {
        self.bits[(y as usize) * (self.width as usize) + x as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_integral_and_threshold() {
        let mut g = Gray::new(4, 3);
        for y in 0..3 {
            for x in 0..4 {
                g.set(x, y, (x * 10 + y * 100) as u8);
            }
        }
        // Pixel centres return the pixel; halfway between two, their mean.
        assert_eq!(g.sample(1.5, 0.5), 10.0);
        assert_eq!(g.sample(2.0, 0.5), 15.0);
        assert_eq!(g.sample(0.5, 1.0), 50.0);
        // Outside the picture clamps to the edge pixel.
        assert_eq!(g.sample(-3.0, -3.0), 0.0);
        assert_eq!(g.sample(9.0, 9.0), 230.0);

        let integral = g.integral();
        assert_eq!(
            integral.sum(0, 0, 4, 3),
            g.pixels.iter().map(|p| u64::from(*p)).sum()
        );
        assert_eq!(integral.sum(1, 1, 3, 2), 110 + 120);

        // A dark pixel among light ones is foreground; the light ones are not.
        let mut flat = Gray::new(5, 5);
        flat.pixels.fill(200);
        flat.set(2, 2, 40);
        let binary = flat.threshold_local(3, 7);
        assert!(binary.get(2, 2));
        assert!(!binary.get(0, 0) && !binary.get(1, 2));

        let luma = Gray::from_rgb8(1, 1, &[255, 0, 0]);
        assert_eq!(luma.get(0, 0), 76);
        let blurred = flat.blurred(1.0);
        assert!(blurred.get(2, 2) > 40 && blurred.get(2, 2) < 200);
    }
}
