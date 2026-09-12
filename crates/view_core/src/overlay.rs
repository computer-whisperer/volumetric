//! Compositing a render over the photograph it was taken through, so a
//! reader can judge where the model and the picture agree. The render
//! is drawn through the photograph's lens (the renderer's warp), so the
//! photograph is used as shot.

use anyhow::{Result, bail};

use crate::image::Rgb;

/// How the render and the photograph are combined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Overlay {
    /// The render mixed over the photograph where it has geometry.
    Blend { alpha: f32 },
    /// The render's silhouette and interior edges drawn in orange over the
    /// photograph.
    Edge,
    /// Photograph and render side by side, the render over a dark ground.
    Side,
    /// Alternating square tiles of photograph and render; where a render
    /// tile has no geometry the photograph shows through.
    Checker { tile: u32 },
}

impl Overlay {
    /// Parses `blend`, `edge`, `side`, `checker`.
    pub fn parse(name: &str, alpha: f32, tile: u32) -> Option<Self> {
        match name {
            "blend" => Some(Self::Blend { alpha }),
            "edge" => Some(Self::Edge),
            "side" => Some(Self::Side),
            "checker" => Some(Self::Checker { tile }),
            _ => None,
        }
    }
}

/// Composes `render` over `photo`, which must be the same size.
/// `coverage` is how much of each render pixel is geometry, in `[0, 1]`:
/// 1 on a surface, a fraction on a splat's fading edge, 0 where the
/// background shows. Blend weights by it; the other modes count a pixel
/// as geometry above one half.
pub fn compose(photo: &Rgb, render: &Rgb, coverage: &[f32], overlay: Overlay) -> Result<Rgb> {
    if photo.width != render.width || photo.height != render.height {
        bail!(
            "photograph is {}x{} but the render is {}x{}",
            photo.width,
            photo.height,
            render.width,
            render.height
        );
    }
    if coverage.len() != (photo.width * photo.height) as usize {
        bail!("coverage mask does not match the image");
    }
    let (w, h) = (photo.width, photo.height);
    let at = |x: u32, y: u32| coverage[(y * w + x) as usize] > 0.5;
    Ok(match overlay {
        Overlay::Blend { alpha } => {
            let alpha = alpha.clamp(0.0, 1.0);
            let mut out = photo.clone();
            for y in 0..h {
                for x in 0..w {
                    let weight = alpha * coverage[(y * w + x) as usize].clamp(0.0, 1.0);
                    if weight > 0.0 {
                        let p = photo.get(x, y);
                        let r = render.get(x, y);
                        let mix = |i: usize| {
                            (f32::from(p[i]) * (1.0 - weight) + f32::from(r[i]) * weight).round()
                                as u8
                        };
                        out.set(x, y, [mix(0), mix(1), mix(2)]);
                    }
                }
            }
            out
        }
        Overlay::Edge => {
            let luma = |x: u32, y: u32| {
                let [r, g, b] = render.get(x, y);
                0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b)
            };
            let mut out = photo.clone();
            for y in 0..h {
                for x in 0..w {
                    if !at(x, y) {
                        continue;
                    }
                    let silhouette = x == 0
                        || y == 0
                        || x + 1 == w
                        || y + 1 == h
                        || !at(x - 1, y)
                        || !at(x + 1, y)
                        || !at(x, y - 1)
                        || !at(x, y + 1);
                    let interior = !silhouette && {
                        let gx = luma(x + 1, y) - luma(x - 1, y);
                        let gy = luma(x, y + 1) - luma(x, y - 1);
                        (gx * gx + gy * gy).sqrt() > 60.0
                    };
                    if silhouette || interior {
                        out.set(x, y, [255, 96, 0]);
                    }
                }
            }
            out
        }
        Overlay::Side => {
            const GROUND: [u8; 3] = [45, 45, 45];
            let mut out = Rgb::new(w * 2, h);
            for y in 0..h {
                for x in 0..w {
                    out.set(x, y, photo.get(x, y));
                    out.set(x + w, y, if at(x, y) { render.get(x, y) } else { GROUND });
                }
            }
            out
        }
        Overlay::Checker { tile } => {
            let tile = tile.max(1);
            let mut out = photo.clone();
            for y in 0..h {
                for x in 0..w {
                    if (x / tile + y / tile) % 2 == 1 && at(x, y) {
                        out.set(x, y, render.get(x, y));
                    }
                }
            }
            out
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn images() -> (Rgb, Rgb, Vec<f32>) {
        let mut photo = Rgb::new(4, 4);
        let mut render = Rgb::new(4, 4);
        let mut covered = vec![0.0; 16];
        for y in 0..4 {
            for x in 0..4 {
                photo.set(x, y, [100, 100, 100]);
                let inside = (1..3).contains(&x) && (1..3).contains(&y);
                render.set(x, y, if inside { [200, 0, 0] } else { [10, 10, 10] });
                covered[(y * 4 + x) as usize] = if inside { 1.0 } else { 0.0 };
            }
        }
        (photo, render, covered)
    }

    #[test]
    fn overlays_compose_as_described() {
        let (photo, render, covered) = images();
        let blend = compose(&photo, &render, &covered, Overlay::Blend { alpha: 1.0 }).unwrap();
        assert_eq!(blend.get(1, 1), [200, 0, 0]);
        assert_eq!(blend.get(0, 0), [100, 100, 100]);
        let half = compose(&photo, &render, &covered, Overlay::Blend { alpha: 0.5 }).unwrap();
        assert_eq!(half.get(1, 1), [150, 50, 50]);
        // A fading edge (a splat's) blends by its coverage.
        let mut fading = covered.clone();
        fading[5] = 0.5;
        let faded = compose(&photo, &render, &fading, Overlay::Blend { alpha: 1.0 }).unwrap();
        assert_eq!(faded.get(1, 1), [150, 50, 50]);

        let edge = compose(&photo, &render, &covered, Overlay::Edge).unwrap();
        assert_eq!(edge.get(1, 1), [255, 96, 0]);
        assert_eq!(edge.get(0, 0), [100, 100, 100]);

        let side = compose(&photo, &render, &covered, Overlay::Side).unwrap();
        assert_eq!((side.width, side.height), (8, 4));
        assert_eq!(side.get(5, 1), [200, 0, 0]);
        assert_eq!(side.get(4, 0), [45, 45, 45]);

        let checker = compose(&photo, &render, &covered, Overlay::Checker { tile: 1 }).unwrap();
        assert_eq!(checker.get(0, 0), [100, 100, 100]);
        assert_eq!(
            checker.get(1, 0),
            [100, 100, 100],
            "no geometry: photo shows"
        );
        assert_eq!(checker.get(2, 1), [200, 0, 0]);

        assert!(compose(&Rgb::new(2, 2), &render, &covered, Overlay::Edge).is_err());
        assert_eq!(
            Overlay::parse("blend", 0.3, 32),
            Some(Overlay::Blend { alpha: 0.3 })
        );
        assert_eq!(Overlay::parse("nope", 0.3, 32), None);
    }
}
