//! Low-cost direct model thumbnails without constructing a mesh.
//!
//! Three-dimensional models are sampled along a fixed orthographic ray
//! bundle with early exit. The resulting first-hit depth field is shaded in
//! screen space. Two-dimensional models reuse the sketch rasterizer. This is
//! deliberately a transient preview path: it never feeds export geometry.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;

use crate::wasm::ParallelModelSampler as _;

#[derive(Clone, Debug)]
pub struct DirectPreviewRaster {
    pub width: u32,
    pub height: u32,
    /// Straight-alpha RGBA8, row zero at the top.
    pub rgba: Vec<u8>,
    pub samples: u64,
}

/// Renders a model directly at `width × height`. `Ok(None)` means the cancel
/// flag was observed. `ray_steps` only applies to 3D models.
#[cfg(any(feature = "native", feature = "web"))]
pub fn render_model_thumbnail(
    model_wasm: &[u8],
    width: u32,
    height: u32,
    ray_steps: usize,
    cancel: &AtomicBool,
) -> anyhow::Result<Option<DirectPreviewRaster>> {
    anyhow::ensure!(
        width > 0 && height > 0,
        "thumbnail dimensions must be positive"
    );
    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }
    let dimensions = crate::model_dimensions_static(model_wasm)
        .map(Ok)
        .unwrap_or_else(|| crate::model_dimensions_from_bytes(model_wasm))?;
    match dimensions {
        2 => render_sketch_thumbnail(model_wasm, width, height, cancel),
        3 => render_volume_thumbnail(model_wasm, width, height, ray_steps.max(1), cancel),
        dimensions => anyhow::bail!("direct thumbnails support 2D/3D models, got {dimensions}D"),
    }
}

#[cfg(any(feature = "native", feature = "web"))]
fn render_sketch_thumbnail(
    model_wasm: &[u8],
    width: u32,
    height: u32,
    cancel: &AtomicBool,
) -> anyhow::Result<Option<DirectPreviewRaster>> {
    let resolution = width.max(height) as usize;
    let raster = crate::rasterize_sketch_from_bytes(model_wasm, resolution)?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }

    let binary = raster.is_binary();
    let range = (raster.value_max - raster.value_min).max(f32::EPSILON);
    let mut rgba = vec![0; width as usize * height as usize * 4];
    for y in 0..height as usize {
        let source_y =
            ((height as usize - 1 - y) * raster.height / height as usize).min(raster.height - 1);
        for x in 0..width as usize {
            let source_x = (x * raster.width / width as usize).min(raster.width - 1);
            let value = raster.value(source_x, source_y);
            let pixel = (y * width as usize + x) * 4;
            if !value.is_finite() || (binary && !volumetric_abi::is_occupied(value)) {
                continue;
            }
            let color = if binary {
                [0.32, 0.68, 0.88]
            } else {
                crate::viridis((value - raster.value_min) / range)
            };
            rgba[pixel..pixel + 3]
                .copy_from_slice(&color.map(|channel| (channel * 255.0).round() as u8));
            rgba[pixel + 3] = 255;
        }
    }
    Ok(Some(DirectPreviewRaster {
        width,
        height,
        rgba,
        samples: (raster.width * raster.height) as u64,
    }))
}

#[cfg(any(feature = "native", feature = "web"))]
fn render_volume_thumbnail(
    model_wasm: &[u8],
    width: u32,
    height: u32,
    ray_steps: usize,
    cancel: &AtomicBool,
) -> anyhow::Result<Option<DirectPreviewRaster>> {
    let sampler = crate::wasm::create_parallel_sampler(model_wasm)
        .context("creating thumbnail model sampler")?;
    let bounds = sampler.get_bounds()?;
    let min = [bounds.min.0, bounds.min.1, bounds.min.2];
    let max = [bounds.max.0, bounds.max.1, bounds.max.2];
    anyhow::ensure!(
        (0..3).all(|axis| min[axis].is_finite() && max[axis] > min[axis]),
        "model reported invalid bounds"
    );

    let center = std::array::from_fn(|axis| (min[axis] + max[axis]) * 0.5);
    let half = std::array::from_fn(|axis| (max[axis] - min[axis]) * 0.5);
    let forward = normalize([1.3, 1.6, -1.0]);
    let right = normalize(cross(forward, [0.0, 0.0, 1.0]));
    let up = normalize(cross(right, forward));
    let projected_width = projected_extent(half, right);
    let projected_height = projected_extent(half, up);
    let aspect = width as f64 / height as f64;
    let screen_half_width = projected_width.max(projected_height * aspect) * 1.12;
    let screen_half_height = screen_half_width / aspect;
    let diagonal = length(half);
    let plane_center = sub(center, scale(forward, diagonal * 2.5));

    let rows: Vec<(Vec<f32>, u64)> = crate::parallel_iter::map_range(0..height as usize, |y| {
        let mut depths = vec![f32::NAN; width as usize];
        let mut samples = 0u64;
        if cancel.load(Ordering::Relaxed) {
            return (depths, samples);
        }
        let v = 1.0 - 2.0 * (y as f64 + 0.5) / height as f64;
        for (x, depth) in depths.iter_mut().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let u = 2.0 * (x as f64 + 0.5) / width as f64 - 1.0;
            let origin = add(
                add(plane_center, scale(right, u * screen_half_width)),
                scale(up, v * screen_half_height),
            );
            let Some((near, far)) = ray_box_intersection(origin, forward, min, max) else {
                continue;
            };
            for step in 0..ray_steps {
                let fraction = (step as f64 + 0.5) / ray_steps as f64;
                let point = add(origin, scale(forward, near + (far - near) * fraction));
                samples += 1;
                if volumetric_abi::is_occupied(sampler.sample(point[0], point[1], point[2])) {
                    *depth = fraction as f32;
                    break;
                }
            }
        }
        (depths, samples)
    });
    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }
    let samples = rows.iter().map(|(_, samples)| *samples).sum();
    let depths: Vec<f32> = rows.into_iter().flat_map(|(row, _)| row).collect();
    let rgba = shade_depths(&depths, width as usize, height as usize, ray_steps);
    Ok(Some(DirectPreviewRaster {
        width,
        height,
        rgba,
        samples,
    }))
}

fn shade_depths(depths: &[f32], width: usize, height: usize, ray_steps: usize) -> Vec<u8> {
    let mut rgba = vec![0; width * height * 4];
    let light = normalize([-0.45, -0.55, 1.0]);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let center = depths[i];
            if !center.is_finite() {
                continue;
            }
            let sample = |x: usize, y: usize| {
                let value = depths[y * width + x];
                if value.is_finite() { value } else { center }
            };
            let left = sample(x.saturating_sub(1), y);
            let right = sample((x + 1).min(width - 1), y);
            let top = sample(x, y.saturating_sub(1));
            let bottom = sample(x, (y + 1).min(height - 1));
            let slope = ray_steps as f64 * 0.55;
            let normal = normalize([
                f64::from(left - right) * slope,
                f64::from(top - bottom) * slope,
                1.0,
            ]);
            let diffuse = dot(normal, light).max(0.0);
            let intensity = 0.30 + 0.70 * diffuse;
            let depth_fade = 1.0 - f64::from(center) * 0.18;
            let base = [0.32, 0.68, 0.88];
            let pixel = i * 4;
            for channel in 0..3 {
                rgba[pixel + channel] =
                    (base[channel] * intensity * depth_fade * 255.0).round() as u8;
            }
            rgba[pixel + 3] = 255;
        }
    }
    rgba
}

fn ray_box_intersection(
    origin: [f64; 3],
    direction: [f64; 3],
    min: [f64; 3],
    max: [f64; 3],
) -> Option<(f64, f64)> {
    let mut near = f64::NEG_INFINITY;
    let mut far = f64::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() < 1.0e-12 {
            if origin[axis] < min[axis] || origin[axis] > max[axis] {
                return None;
            }
            continue;
        }
        let a = (min[axis] - origin[axis]) / direction[axis];
        let b = (max[axis] - origin[axis]) / direction[axis];
        near = near.max(a.min(b));
        far = far.min(a.max(b));
        if far < near {
            return None;
        }
    }
    (far >= 0.0).then_some((near.max(0.0), far))
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| a[axis] + b[axis])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| a[axis] - b[axis])
}

fn scale(v: [f64; 3], scale: f64) -> [f64; 3] {
    v.map(|component| component * scale)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|axis| a[axis] * b[axis]).sum()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / length(v).max(f64::EPSILON))
}

fn projected_extent(half: [f64; 3], axis: [f64; 3]) -> f64 {
    (0..3).map(|i| half[i] * axis[i].abs()).sum()
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    fn sphere() -> Vec<u8> {
        wat::parse_str(
            r#"(module
                (memory (export "memory") 1)
                (func (export "get_dimensions") (result i32) (i32.const 3))
                (func (export "get_io_ptr") (result i32) (i32.const 1024))
                (func (export "get_bounds") (param $p i32)
                    (f64.store (local.get $p) (f64.const -1))
                    (f64.store offset=8 (local.get $p) (f64.const 1))
                    (f64.store offset=16 (local.get $p) (f64.const -1))
                    (f64.store offset=24 (local.get $p) (f64.const 1))
                    (f64.store offset=32 (local.get $p) (f64.const -1))
                    (f64.store offset=40 (local.get $p) (f64.const 1)))
                (func (export "sample") (param $p i32) (result f32)
                    (if (result f32)
                        (f64.le
                            (f64.add
                                (f64.mul (f64.load (local.get $p)) (f64.load (local.get $p)))
                                (f64.add
                                    (f64.mul (f64.load offset=8 (local.get $p)) (f64.load offset=8 (local.get $p)))
                                    (f64.mul (f64.load offset=16 (local.get $p)) (f64.load offset=16 (local.get $p)))))
                            (f64.const 0.64))
                        (then (f32.const 1))
                        (else (f32.const 0)))))"#,
        )
        .unwrap()
    }

    #[test]
    fn volume_thumbnail_has_a_shaded_silhouette() {
        let raster = render_model_thumbnail(&sphere(), 48, 48, 32, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        let opaque = raster
            .rgba
            .chunks_exact(4)
            .filter(|pixel| pixel[3] != 0)
            .count();
        assert!(opaque > 200, "sphere silhouette was empty: {opaque}");
        assert!(opaque < 48 * 48, "background was not transparent");
        assert!(raster.samples < 48 * 48 * 32);
    }

    #[test]
    fn pre_cancelled_thumbnail_does_no_work() {
        let cancel = AtomicBool::new(true);
        assert!(
            render_model_thumbnail(&sphere(), 32, 32, 16, &cancel)
                .unwrap()
                .is_none()
        );
    }
}
