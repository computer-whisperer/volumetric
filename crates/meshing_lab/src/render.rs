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

/// Render with smooth (Gouraud-style per-pixel) shading from per-vertex
/// normals, like a production viewport. This is where crease shading quality
/// is visible: flat shading hides blended vertex normals entirely.
pub fn render_smooth_png(
    positions: &[DVec3],
    normals: &[DVec3],
    indices: &[u32],
    config: &RenderConfig,
    path: &Path,
) -> Result<(), image::ImageError> {
    let (w, h) = (config.width, config.height);
    let mut color = vec![config.background; w * h];
    let mut inv_depth = vec![0.0f64; w * h];

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
    let base = [0.62, 0.66, 0.72];

    let project = |p: DVec3| -> (DVec3, f64) {
        let rel = p - config.camera_pos;
        let view = DVec3::new(rel.dot(right), rel.dot(up), -rel.dot(forward));
        let depth = -view.z;
        let sx = w as f64 / 2.0 + focal * view.x / depth;
        let sy = h as f64 / 2.0 - focal * view.y / depth;
        (DVec3::new(sx, sy, 0.0), depth)
    };

    for tri in indices.chunks_exact(3) {
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (p0, p1, p2) = (positions[ia], positions[ib], positions[ic]);
        let (s0, d0) = project(p0);
        let (s1, d1) = project(p1);
        let (s2, d2) = project(p2);
        if d0 < near || d1 < near || d2 < near {
            continue;
        }
        let n0 = positions_normal(normals[ia]);
        let n1 = positions_normal(normals[ib]);
        let n2 = positions_normal(normals[ic]);

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
                let inv_z = w0 / d0 + w1 / d1 + w2 / d2;
                let px_idx = y * w + x;
                if inv_z <= inv_depth[px_idx] {
                    continue;
                }
                inv_depth[px_idx] = inv_z;
                let mut n = (n0 * w0 + n1 * w1 + n2 * w2).normalize_or_zero();
                if n == DVec3::ZERO {
                    n = DVec3::Y;
                }
                // Light both sides so inward-blended normals stay visible
                // rather than going black.
                let intensity = 0.25 + 0.75 * n.dot(light).abs();
                color[px_idx] = [
                    (base[0] * intensity * 255.0).clamp(0.0, 255.0) as u8,
                    (base[1] * intensity * 255.0).clamp(0.0, 255.0) as u8,
                    (base[2] * intensity * 255.0).clamp(0.0, 255.0) as u8,
                ];
            }
        }
    }

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Rgb(color[y as usize * w + x as usize]);
    }
    img.save(path)
}

fn positions_normal(n: DVec3) -> DVec3 {
    n.try_normalize().unwrap_or(DVec3::Y)
}

/// Render replicating the production viewport's mesh pipeline semantics:
/// back-face culling, raw (unnormalized) vertex normals interpolated then
/// normalized per pixel, and one-sided diffuse lighting. Defects the
/// two-sided [`render_smooth_png`] hides — inverted windings, wrong-side
/// normals, magnitude imbalance across a triangle — are glaring here.
pub fn render_gui_png(
    positions: &[DVec3],
    normals: &[DVec3],
    indices: &[u32],
    config: &RenderConfig,
    path: &Path,
) -> Result<(), image::ImageError> {
    let (w, h) = (config.width, config.height);
    let mut color = vec![config.background; w * h];
    let mut inv_depth = vec![0.0f64; w * h];

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
    // The production renderer's light and lighting model (mesh_gbuffer.wgsl).
    let light = DVec3::new(0.4, 0.7, 0.2).normalize();
    let base = [0.78, 0.80, 0.83];

    let project = |p: DVec3| -> (DVec3, f64) {
        let rel = p - config.camera_pos;
        let view = DVec3::new(rel.dot(right), rel.dot(up), -rel.dot(forward));
        let depth = -view.z;
        let sx = w as f64 / 2.0 + focal * view.x / depth;
        let sy = h as f64 / 2.0 - focal * view.y / depth;
        (DVec3::new(sx, sy, 0.0), depth)
    };

    for tri in indices.chunks_exact(3) {
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (p0, p1, p2) = (positions[ia], positions[ib], positions[ic]);
        let (s0, d0) = project(p0);
        let (s1, d1) = project(p1);
        let (s2, d2) = project(p2);
        if d0 < near || d1 < near || d2 < near {
            continue;
        }
        // Raw normals, as the GPU pipeline interpolates them.
        let (n0, n1, n2) = (normals[ia], normals[ib], normals[ic]);

        let edge = |a: DVec3, b: DVec3, x: f64, y: f64| -> f64 {
            (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x)
        };
        let area = edge(s0, s1, s2.x, s2.y);
        // Back-face culling: CCW-in-NDC front faces are CW in this Y-down
        // screen space, i.e. negative signed area.
        if area >= 0.0 {
            continue;
        }

        let min_x = s0.x.min(s1.x).min(s2.x).floor().max(0.0) as usize;
        let max_x = (s0.x.max(s1.x).max(s2.x).ceil() as usize).min(w - 1);
        let min_y = s0.y.min(s1.y).min(s2.y).floor().max(0.0) as usize;
        let max_y = (s0.y.max(s1.y).max(s2.y).ceil() as usize).min(h - 1);
        if min_x > max_x || min_y > max_y {
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
                let inv_z = w0 / d0 + w1 / d1 + w2 / d2;
                let px_idx = y * w + x;
                if inv_z <= inv_depth[px_idx] {
                    continue;
                }
                inv_depth[px_idx] = inv_z;
                let n = (n0 * w0 + n1 * w1 + n2 * w2).normalize_or_zero();
                let intensity = 0.22 + 0.78 * n.dot(light).max(0.0);
                color[px_idx] = [
                    (base[0] * intensity * 255.0).clamp(0.0, 255.0) as u8,
                    (base[1] * intensity * 255.0).clamp(0.0, 255.0) as u8,
                    (base[2] * intensity * 255.0).clamp(0.0, 255.0) as u8,
                ];
            }
        }
    }

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Rgb(color[y as usize * w + x as usize]);
    }
    img.save(path)
}

/// Camera framing a bounding box from the standard three-quarter view.
pub fn frame_bounds(lo: DVec3, hi: DVec3) -> RenderConfig {
    let center = (lo + hi) / 2.0;
    let radius = (hi - lo).length() / 2.0;
    RenderConfig {
        camera_pos: center + DVec3::new(1.0, 0.62, 1.05).normalize() * radius * 2.6,
        target: center,
        ..RenderConfig::default()
    }
}

/// Render with per-region colors; triangles touching any unclaimed vertex are
/// bright red so the feature zone stands out.
pub fn render_segments(
    positions: &[DVec3],
    indices: &[u32],
    labels: &[Option<u32>],
    bounds: (DVec3, DVec3),
    path: &Path,
) {
    let config = frame_bounds(bounds.0, bounds.1);
    let tri_color = |tri: usize| -> [f64; 3] {
        let vs = &indices[tri * 3..tri * 3 + 3];
        match vs.iter().map(|&v| labels[v as usize]).min().flatten() {
            // All three vertices claimed: color by one of their regions.
            Some(region) if vs.iter().all(|&v| labels[v as usize].is_some()) => {
                region_color(region)
            }
            _ => [0.95, 0.15, 0.12], // feature zone
        }
    };
    if let Err(err) = render_png(positions, indices, &tri_color, &config, path) {
        eprintln!("render failed: {err}");
    }
}

/// Render in a single neutral material color, like a production viewport.
pub fn render_plain(positions: &[DVec3], indices: &[u32], bounds: (DVec3, DVec3), path: &Path) {
    let config = frame_bounds(bounds.0, bounds.1);
    let tri_color = |_: usize| -> [f64; 3] { [0.62, 0.66, 0.72] };
    if let Err(err) = render_png(positions, indices, &tri_color, &config, path) {
        eprintln!("render failed: {err}");
    }
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
