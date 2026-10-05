//! Picking: what is under a pixel of the last rendered frame.
//!
//! One texel of the retained G-buffer's surface target (object id and
//! depth) is copied into a 1x1 integer target and read back
//! asynchronously: an integer colour read-back is the one every backend
//! guarantees.

use std::sync::mpsc;

use glam::{Mat4, Vec3};

use crate::ObjectId;
use crate::pipelines::{FullscreenPass, PickUniforms, pick_pass};

pub(crate) const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

/// What the last frame drew at a pixel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Pick {
    /// The pixel asked about, in the frame's target (origin top-left).
    pub pixel: (u32, u32),
    /// The mesh draw there; [`ObjectId::NONE`] over background, lines and
    /// points.
    pub object: ObjectId,
    /// The world point on that surface; `None` where no geometry source
    /// drew (background, or only lines and points).
    pub world: Option<Vec3>,
}

/// The view a G-buffer was filled with: what turns its depth back into
/// world points.
#[derive(Copy, Clone)]
pub(crate) struct FrameRecord {
    pub inv_view_proj: Mat4,
    /// The G-buffer's size in pixels.
    pub size: (u32, u32),
}

impl FrameRecord {
    /// The world point at `pixel` and NDC depth `depth`.
    fn unproject(&self, pixel: (u32, u32), depth: f32) -> Option<Vec3> {
        if !(0.0..1.0).contains(&depth) {
            return None;
        }
        let x = (pixel.0 as f32 + 0.5) / self.size.0 as f32 * 2.0 - 1.0;
        let y = 1.0 - (pixel.1 as f32 + 0.5) / self.size.1 as f32 * 2.0;
        let world = self.inv_view_proj.project_point3(Vec3::new(x, y, depth));
        world.is_finite().then_some(world)
    }
}

struct InFlight {
    /// The pixel as the host asked for it.
    asked: (u32, u32),
    /// The G-buffer pixel read.
    pixel: (u32, u32),
    frame: FrameRecord,
    mapped: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

/// The pick pass, its 1x1 target and read-back buffer, and the one
/// request that may be in flight.
pub(crate) struct Picker {
    pass: FullscreenPass,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    in_flight: Option<InFlight>,
    latest: Option<Pick>,
}

impl Picker {
    pub fn new(device: &wgpu::Device) -> Self {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_target"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            pass: pick_pass(device),
            target,
            target_view,
            readback,
            in_flight: None,
            latest: None,
        }
    }

    /// The pick pass's bind group over a G-buffer's surface target.
    pub fn bind(&self, device: &wgpu::Device, surface_view: &wgpu::TextureView) -> wgpu::BindGroup {
        self.pass.bind(device, &[surface_view])
    }

    pub fn busy(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Reads G-buffer `pixel` (the host's `asked` pixel) of the frame
    /// `frame` describes. The caller checks [`busy`](Self::busy) first.
    pub fn request(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group: &wgpu::BindGroup,
        frame: FrameRecord,
        asked: (u32, u32),
        pixel: (u32, u32),
    ) {
        self.pass.write_uniforms(
            queue,
            &PickUniforms {
                pixel: [pixel.0 as i32, pixel.1 as i32],
                _pad0: [0; 2],
            },
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pick"),
        });
        self.pass.run(
            &mut encoder,
            bind_group,
            &self.target_view,
            wgpu::Color::TRANSPARENT,
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let (mapped_tx, mapped) = mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = mapped_tx.send(result);
            });
        self.in_flight = Some(InFlight {
            asked,
            pixel,
            frame,
            mapped,
        });
    }

    /// Lands a finished read-back, if there is one, without blocking.
    pub fn poll(&mut self, device: &wgpu::Device) {
        let Some(in_flight) = &self.in_flight else {
            return;
        };
        // Native backends run map callbacks from poll; on the web the
        // browser's event loop does and this is a no-op.
        let _ = device.poll(wgpu::PollType::Poll);
        match in_flight.mapped.try_recv() {
            Err(mpsc::TryRecvError::Empty) => return,
            Ok(Ok(())) => {
                if let Ok(data) = self.readback.slice(..).get_mapped_range() {
                    let word =
                        |i: usize| u32::from_le_bytes(data[4 * i..4 * i + 4].try_into().unwrap());
                    self.latest = Some(Pick {
                        pixel: in_flight.asked,
                        object: ObjectId(word(0)),
                        world: in_flight
                            .frame
                            .unproject(in_flight.pixel, f32::from_bits(word(1))),
                    });
                }
                self.readback.unmap();
            }
            // A failed mapping leaves the previous answer standing.
            Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => {}
        }
        self.in_flight = None;
    }

    pub fn latest(&self) -> Option<Pick> {
        self.latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pixel centres unproject through the frame's own matrices; depth 1
    /// (nothing drawn) is no point at all.
    #[test]
    fn unproject_inverts_the_frames_projection() {
        let view = crate::CameraView::look_at(
            Vec3::new(0.0, -3.0, 0.5),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::Z,
            0.8,
            1.5,
            0.1,
            10.0,
        );
        let frame = FrameRecord {
            inv_view_proj: view.view_projection().inverse(),
            size: (300, 200),
        };
        let world = Vec3::new(0.3, 0.2, 0.7);
        let at = view.project(world, 300, 200).unwrap();
        let clip = view.view_projection() * world.extend(1.0);
        let pixel = (at.x as u32, at.y as u32);
        let back = frame.unproject(pixel, clip.z / clip.w).unwrap();
        // Within the half pixel the pixel centre is off by.
        assert!((back - world).length() < 0.02, "{back}");
        assert_eq!(frame.unproject(pixel, 1.0), None);
    }
}
