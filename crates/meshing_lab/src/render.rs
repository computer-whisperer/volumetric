//! Minimal software rasterizer for research renders.
//!
//! Deliberately not the production renderer: no GPU, no windowing, just a
//! z-buffered flat-shaded projection with caller-supplied per-triangle colors,
//! so segmentation labels and feature zones can be inspected visually. Good
//! enough to make meshing defects obvious; not meant to be pretty.

use glam::DVec3;
use std::path::Path;

pub struct RenderConfig {
    pub width: usize,
    pub height: usize,
    pub camera_pos: DVec3,
    pub target: DVec3,
    pub fov_deg: f64,
    pub background: [u8; 3],
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            camera_pos: DVec3::new(2.0, 1.4, 2.0),
            target: DVec3::ZERO,
            fov_deg: 40.0,
            background: [24, 26, 30],
        }
    }
}

/// Render triangles with flat shading; `tri_color` supplies the base color
/// (linear 0..1 RGB) per triangle index. Writes a PNG.
pub fn render_png(
    positions: &[DVec3],
    indices: &[u32],
    tri_color: &dyn Fn(usize) -> [f64; 3],
    config: &RenderConfig,
    path: &Path,
) -> Result<(), image::ImageError> {
    let (w, h) = (config.width, config.height);
    let mut color = vec![config.background; w * h];
    // Inverse view-space depth; larger is closer.
    let mut inv_depth = vec![0.0f64; w * h];

    // Look-at basis (right-handed, camera looks down -Z in view space).
    let forward = (config.target - config.camera_pos).normalize();
    let up_hint = if forward.dot(DVec3::Y).abs() > 0.99 {
        DVec3::Z
    } else {
        DVec3::Y
    };
    let right = forward.cross(up_hint).normalize();
    let up = right.cross(forward);

    let focal = (h as f64 / 2.0) / (config.fov_deg.to_radians() / 2.0).tan();
    let near = 1e-3;
    let light = DVec3::new(0.45, 0.8, 0.4).normalize();

    let project = |p: DVec3| -> (DVec3, f64) {
        let rel = p - config.camera_pos;
        let view = DVec3::new(rel.dot(right), rel.dot(up), -rel.dot(forward));
        let depth = -view.z; // positive in front of the camera
        let sx = w as f64 / 2.0 + focal * view.x / depth;
        let sy = h as f64 / 2.0 - focal * view.y / depth;
        (DVec3::new(sx, sy, 0.0), depth)
    };

    for (tri_idx, tri) in indices.chunks_exact(3).enumerate() {
        let p0 = positions[tri[0] as usize];
        let p1 = positions[tri[1] as usize];
        let p2 = positions[tri[2] as usize];

        let (s0, d0) = project(p0);
        let (s1, d1) = project(p1);
        let (s2, d2) = project(p2);
        if d0 < near || d1 < near || d2 < near {
            continue; // behind the camera; whole-triangle reject is fine here
        }

        // Flat shading from the face normal, flipped toward the camera so
        // both sides are lit.
        let mut n = (p1 - p0).cross(p2 - p0);
        if n.length_squared() < 1e-30 {
            continue;
        }
        n = n.normalize();
        if n.dot(config.camera_pos - p0) < 0.0 {
            n = -n;
        }
        let intensity = 0.3 + 0.7 * n.dot(light).max(0.0);
        let base = tri_color(tri_idx);
        let shaded = [
            (base[0] * intensity * 255.0).clamp(0.0, 255.0) as u8,
            (base[1] * intensity * 255.0).clamp(0.0, 255.0) as u8,
            (base[2] * intensity * 255.0).clamp(0.0, 255.0) as u8,
        ];

        // Screen-space bounding box.
        let min_x = s0.x.min(s1.x).min(s2.x).floor().max(0.0) as usize;
        let max_x = (s0.x.max(s1.x).max(s2.x).ceil() as usize).min(w - 1);
        let min_y = s0.y.min(s1.y).min(s2.y).floor().max(0.0) as usize;
        let max_y = (s0.y.max(s1.y).max(s2.y).ceil() as usize).min(h - 1);
        if min_x > max_x || min_y > max_y {
            continue;
        }

        let edge = |a: DVec3, b: DVec3, x: f64, y: f64| -> f64 {
            (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x)
        };
        let area = edge(s0, s1, s2.x, s2.y);
        if area.abs() < 1e-12 {
            continue;
        }

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let w0 = edge(s1, s2, px, py) / area;
                let w1 = edge(s2, s0, px, py) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                // Interpolating inverse depth linearly in screen space is
                // perspective-correct for occlusion.
                let inv_z = w0 / d0 + w1 / d1 + w2 / d2;
                let px_idx = y * w + x;
                if inv_z > inv_depth[px_idx] {
                    inv_depth[px_idx] = inv_z;
                    color[px_idx] = shaded;
                }
            }
        }
    }

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Rgb(color[y as usize * w + x as usize]);
    }
    img.save(path)
}

/// Distinct color for a region id: golden-ratio hue cycling over a range that
/// avoids red, which is reserved for feature-zone highlighting.
pub fn region_color(region: u32) -> [f64; 3] {
    // Hues 0.08..0.92 span orange through violet, skipping the red wraparound.
    let hue = 0.08 + (region as f64 * 0.618_033_988_749_895).fract() * 0.84;
    hsv_to_rgb(hue, 0.55, 0.85)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f64; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i64).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}
