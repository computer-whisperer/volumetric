//! The G-buffer: what every geometry source writes and every later pass
//! reads. No lighting happens while it is filled.
//!
//! | Target | Format | Contents |
//! |---|---|---|
//! | albedo | `Rgba8Unorm` | rgb: base colour, square-root encoded; a: material index / 255 |
//! | normal | `Rgb10a2Unorm` | rgb: world normal mapped to 0..1; a: 1 where a normal was supplied |
//! | surface | `Rg32Uint` | r: [`crate::ObjectId`] of the draw, 0 is background; g: the bits of the fragment's depth as `f32`, 1.0 is background |
//! | depth | `Depth24Plus` | hardware depth, for depth testing only |
//!
//! Depth is carried twice. Later passes read it from the surface target,
//! not from the depth attachment: reading a depth texture as a plain
//! value has no GLSL translation in wgpu (the WebGL2 fallback), while an
//! integer colour target is renderable and loadable on every backend and
//! holds the value exactly.
//!
//! The textures outlive the frame that filled them: picking reads the last
//! frame's surface target.

/// The G-buffer targets and the two layers derived from them, all at the
/// frame's internal size.
pub struct GBuffer {
    pub albedo_view: wgpu::TextureView,
    pub normal_view: wgpu::TextureView,
    pub surface_view: wgpu::TextureView,
    /// Depth attachment of every depth-tested pass.
    pub depth_view: wgpu::TextureView,
    /// Ambient occlusion, 1 = unoccluded.
    pub ao_view: wgpu::TextureView,
    /// The splat layer: Gaussians blended among themselves in their
    /// trainer's value space before being laid over the scene.
    pub splat_view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl GBuffer {
    pub const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm;
    pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
    pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
    /// Half floats so hundreds of blended layers do not quantise.
    pub const SPLAT_LAYER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    /// The surface target where nothing was drawn: no object, depth 1.0.
    pub const SURFACE_CLEAR: wgpu::Color = wgpu::Color {
        r: 0.0,
        g: 1.0f32.to_bits() as f64,
        b: 0.0,
        a: 0.0,
    };

    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = (width.max(1), height.max(1));
        let target = |label: &'static str, format: wgpu::TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        Self {
            albedo_view: target("gbuffer_albedo", Self::ALBEDO_FORMAT),
            normal_view: target("gbuffer_normal", Self::NORMAL_FORMAT),
            surface_view: target("gbuffer_surface", Self::SURFACE_FORMAT),
            depth_view: target("gbuffer_depth", Self::DEPTH_FORMAT),
            ao_view: target("gbuffer_ao", Self::AO_FORMAT),
            splat_view: target("gbuffer_splat_layer", Self::SPLAT_LAYER_FORMAT),
            size,
        }
    }

    /// Recreates the targets at a new size. Returns whether anything
    /// changed (bind groups over the old views are then stale).
    pub fn resize_if_needed(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let new_size = (width.max(1), height.max(1));
        if self.size == new_size {
            return false;
        }
        *self = Self::new(device, new_size.0, new_size.1);
        true
    }
}
