//! Window-free rendering to CPU pixels, for the CLI's `render` and for
//! headless checks: one adapter and device, a renderer sized to the image,
//! and a readback of the frame as RGBA8 rows.

use std::sync::mpsc;

use crate::{CameraView, RenderSettings, Renderer};

/// A GPU device with no surface, and the readback path that turns a frame
/// into bytes.
pub struct Offscreen {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Offscreen {
    /// The frame's texture format: sRGB-encoded 8-bit RGBA, so the bytes
    /// read back are what a PNG stores.
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    /// Opens the highest-performance adapter available with no window.
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|err| format!("no GPU adapter for offscreen rendering: {err}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("volumetric offscreen"),
            ..Default::default()
        }))
        .map_err(|err| format!("GPU device: {err}"))?;
        Ok(Self {
            adapter,
            device,
            queue,
        })
    }

    pub fn adapter_name(&self) -> String {
        self.adapter.get_info().name
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// A renderer initialised on this device and sized to `width` x
    /// `height` pixels.
    pub fn renderer(&self, width: u32, height: u32) -> Renderer {
        let mut renderer = Renderer::new(Self::FORMAT);
        renderer.set_viewport_size(&self.device, width, height);
        renderer.initialize(&self.device, &self.queue, Some(&self.adapter));
        renderer
    }

    /// Draws the geometry submitted to `renderer` with `view` and returns
    /// the frame as RGBA8 rows, top row first, `width * height * 4` bytes.
    /// Ends the renderer's frame; its overflow report stays readable.
    pub fn render_rgba(
        &self,
        renderer: &mut Renderer,
        view: &CameraView,
        settings: &RenderSettings,
    ) -> Result<Vec<u8>, String> {
        let (width, height) = renderer.viewport_size();
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());

        let mut encoder = self.device.create_command_encoder(&Default::default());
        renderer.render_view(
            &self.device,
            &self.queue,
            &mut encoder,
            view,
            settings,
            &target_view,
        );
        renderer.end_frame();

        // Rows in the copy buffer are padded to the copy alignment.
        let unpadded = width * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            extent,
        );
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (mapped_tx, mapped_rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = mapped_tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|err| format!("GPU poll: {err}"))?;
        mapped_rx
            .recv()
            .map_err(|_| "readback mapping never completed".to_string())?
            .map_err(|err| format!("map readback buffer: {err}"))?;
        let data = slice
            .get_mapped_range()
            .map_err(|err| format!("read mapped buffer: {err}"))?;
        let mut rgba = Vec::with_capacity((unpadded * height) as usize);
        for row in data.chunks_exact(padded as usize) {
            rgba.extend_from_slice(&row[..unpadded as usize]);
        }
        Ok(rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SceneData, SplatData, SplatStyle};
    use glam::{Mat4, Vec3};
    use std::sync::Arc;

    /// One opaque red Gaussian, 20 cm across, at the origin, seen from
    /// 1.5 m: the frame's centre is red, its corner is the background.
    /// Skipped where no GPU adapter is available.
    #[test]
    fn a_gaussian_renders_as_a_red_disc() {
        let Ok(offscreen) = Offscreen::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let (w, h) = (128u32, 96u32);
        let mut renderer = offscreen.renderer(w, h);
        let data = SplatData {
            surfels: false,
            positions: vec![[0.0, 0.0, 0.0]],
            axes: vec![[0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]],
            opacities: vec![1.0],
            sh_degree: 0,
            sh: vec![1.0 / 0.282_094_79, -0.5 / 0.282_094_79, -0.5 / 0.282_094_79],
        };
        let mut scene = SceneData::new();
        scene.add_splat(Arc::new(data), Mat4::IDENTITY, SplatStyle::default());
        let resident = renderer.create_retained_scene(offscreen.device(), &scene);
        assert_eq!(resident.splats.len(), 1);
        let view = CameraView::look_at(
            Vec3::new(0.0, 0.0, 1.5),
            Vec3::ZERO,
            Vec3::Y,
            0.8,
            w as f32 / h as f32,
            0.1,
            10.0,
        );
        let settings = RenderSettings {
            background_color: [0.0, 0.0, 1.0, 1.0],
            ssao_enabled: false,
            show_axis_indicator: false,
            ..RenderSettings::default()
        };
        for _ in 0..2 {
            for splat in &resident.splats {
                renderer.submit_retained_splat(splat);
            }
            let rgba = offscreen
                .render_rgba(&mut renderer, &view, &settings)
                .unwrap();
            let px = |x: u32, y: u32| {
                let i = ((y * w + x) * 4) as usize;
                [rgba[i], rgba[i + 1], rgba[i + 2]]
            };
            let centre = px(w / 2, h / 2);
            let corner = px(2, 2);
            assert!(centre[0] > 200 && centre[2] < 60, "centre {centre:?}");
            assert!(corner[2] > 200 && corner[0] < 30, "corner {corner:?}");
            // The footprint fades with distance from the centre.
            let edge = px(w / 2 + 12, h / 2);
            assert!(edge[0] < centre[0] && edge[2] > centre[2], "edge {edge:?}");
        }
    }
}
