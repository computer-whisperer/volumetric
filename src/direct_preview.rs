//! Low-cost direct model thumbnails without constructing a mesh.
//!
//! Three-dimensional models are cast directly ([`crate::direct_cast`]) from
//! a fixed orthographic view and shaded from the normals found.
//! Two-dimensional models reuse the sketch rasterizer. This is deliberately
//! a transient preview path: it never feeds export geometry.

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
/// flag was observed.
#[cfg(any(feature = "native", feature = "web"))]
pub fn render_model_thumbnail(
    model_wasm: &[u8],
    width: u32,
    height: u32,
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
        3 => render_volume_thumbnail(model_wasm, width, height, cancel),
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
    cancel: &AtomicBool,
) -> anyhow::Result<Option<DirectPreviewRaster>> {
    use crate::direct_cast::{CastOptions, CastProjection, CastView, DirectCast};
    use glam::DVec3;

    let sampler = crate::wasm::create_parallel_sampler(model_wasm)
        .context("creating thumbnail model sampler")?;
    let bounds = sampler.get_bounds()?;
    let (min, max) = (DVec3::from(bounds.min), DVec3::from(bounds.max));
    let mut cast = DirectCast::new(min, max)?;

    // A fixed three-quarter view from above, framed on the bounds.
    let center = (min + max) * 0.5;
    let half = (max - min) * 0.5;
    let diagonal = half.length();
    let forward = DVec3::new(1.3, 1.6, -1.0).normalize();
    let eye = center - forward * (diagonal * 2.5);
    let mut view = CastView::look_at(
        eye,
        center,
        DVec3::Z,
        CastProjection::Orthographic { half_height: 1.0 },
        width,
        height,
    );
    let aspect = width as f64 / height as f64;
    let half_width = half.dot(view.right.abs());
    let half_height = half.dot(view.up.abs());
    view.projection = CastProjection::Orthographic {
        half_height: (half_width / aspect).max(half_height) * 1.12,
    };

    // An icon does not need every pixel-wide detail searched for.
    let options = CastOptions {
        spacing: 1.0,
        coarsest: 4.0,
        search: 2.0,
    };
    let Some((image, stats)) = cast.cast(&sampler, &view, &options, cancel) else {
        return Ok(None);
    };

    let light = DVec3::new(-0.45, 0.55, 1.0).normalize();
    let base = [0.32, 0.68, 0.88];
    let mut rgba = vec![0; width as usize * height as usize * 4];
    for (hit, pixel) in image.hits.iter().zip(rgba.chunks_exact_mut(4)) {
        let Some(hit) = hit else { continue };
        // The normal in the view's frame: x right, y up, z to the viewer.
        let normal = hit.normal.as_dvec3();
        let normal = DVec3::new(
            normal.dot(view.right),
            normal.dot(view.up),
            -normal.dot(view.forward),
        );
        let intensity = 0.30 + 0.70 * normal.dot(light).max(0.0);
        // Fade a little toward the back of the bounds.
        let depth = ((hit.t - diagonal * 1.5) / (diagonal * 2.0)).clamp(0.0, 1.0);
        let fade = 1.0 - depth * 0.18;
        for channel in 0..3 {
            pixel[channel] = (base[channel] * intensity * fade * 255.0).round() as u8;
        }
        pixel[3] = 255;
    }
    Ok(Some(DirectPreviewRaster {
        width,
        height,
        rgba,
        samples: stats.samples,
    }))
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
        let raster = render_model_thumbnail(&sphere(), 48, 48, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        let opaque = raster
            .rgba
            .chunks_exact(4)
            .filter(|pixel| pixel[3] != 0)
            .count();
        assert!(opaque > 200, "sphere silhouette was empty: {opaque}");
        assert!(opaque < 48 * 48, "background was not transparent");
        assert!(raster.samples > 0);
    }

    #[test]
    fn pre_cancelled_thumbnail_does_no_work() {
        let cancel = AtomicBool::new(true);
        assert!(
            render_model_thumbnail(&sphere(), 32, 32, &cancel)
                .unwrap()
                .is_none()
        );
    }
}
