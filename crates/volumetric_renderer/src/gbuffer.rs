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
//! Beside them is the lit scene: the resolve pass's output in the frame's
//! own format, which the anti-aliasing pass reads.
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
    /// Ambient occlusion as sampled, before its blur.
    pub ao_raw_view: wgpu::TextureView,
    /// Ambient occlusion, 1 = unoccluded.
    pub ao_view: wgpu::TextureView,
    /// The lit scene before anti-aliasing, in `lit_format`.
    pub lit_view: wgpu::TextureView,
    lit_format: wgpu::TextureFormat,
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

    /// `lit_format` is the format of the frame's target.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        lit_format: wgpu::TextureFormat,
    ) -> Self {
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
            ao_raw_view: target("gbuffer_ao_raw", Self::AO_FORMAT),
            ao_view: target("gbuffer_ao", Self::AO_FORMAT),
            lit_view: target("lit_scene", lit_format),
            lit_format,
            splat_view: target("gbuffer_splat_layer", Self::SPLAT_LAYER_FORMAT),
            size,
        }
    }

    /// The format of the lit scene: the frame's target's.
    pub fn lit_format(&self) -> wgpu::TextureFormat {
        self.lit_format
    }

    /// Recreates the targets at a new size. Returns whether anything
    /// changed (bind groups over the old views are then stale).
    pub fn resize_if_needed(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let new_size = (width.max(1), height.max(1));
        if self.size == new_size {
            return false;
        }
        *self = Self::new(device, new_size.0, new_size.1, self.lit_format);
        true
    }
}

/// What the transparent meshes are drawn through, all at the frame's
/// internal size. They are peeled a layer at a time, nearest first: a
/// layer is filled like the G-buffer (with how opaque each surface is
/// beside it), lit by the resolve pass, and laid under the layers already
/// gathered. Made by the first frame that draws a transparent mesh.
///
/// | Target | Format | Contents |
/// |---|---|---|
/// | albedo, normal | as the G-buffer's | the layer being drawn |
/// | opacity | `R8Unorm` | the square root of how opaque the layer's surface is |
/// | depth | `Depth24Plus` | picks the layer's nearest fragments |
/// | surface, one a layer | `Rg32Uint` | as the G-buffer's |
/// | gathered | the frame's | the layers lit so far, premultiplied |
///
/// A layer's surface target is its own: the next layer is peeled against
/// it, every other layer's edge lines look for their surfaces in it, and
/// the nearest layer's is what a pick reads. The other targets are
/// shared, each layer filling them just before it is lit.
pub struct LayerBuffers {
    pub albedo_view: wgpu::TextureView,
    pub normal_view: wgpu::TextureView,
    pub opacity_view: wgpu::TextureView,
    pub depth_view: wgpu::TextureView,
    /// One per layer, nearest first.
    pub surface_views: Vec<wgpu::TextureView>,
    /// The layers lit so far, nearest first: colour premultiplied by
    /// coverage, and the coverage.
    pub gathered_view: wgpu::TextureView,
}

impl LayerBuffers {
    pub const OPACITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

    /// The most layers a frame peels, whatever its settings ask for: a
    /// layer's resolve reads every other layer's surface target, and
    /// WebGL2 gives a shader sixteen textures.
    pub const MAX_LAYERS: u32 = 8;

    /// Targets for `layers` layers. `lit_format` is the format of the
    /// frame's target.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        lit_format: wgpu::TextureFormat,
        layers: u32,
    ) -> Self {
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
            albedo_view: target("layer_albedo", GBuffer::ALBEDO_FORMAT),
            normal_view: target("layer_normal", GBuffer::NORMAL_FORMAT),
            opacity_view: target("layer_opacity", Self::OPACITY_FORMAT),
            depth_view: target("layer_depth", GBuffer::DEPTH_FORMAT),
            surface_views: (0..layers.clamp(1, Self::MAX_LAYERS))
                .map(|_| target("layer_surface", GBuffer::SURFACE_FORMAT))
                .collect(),
            gathered_view: target("layers_gathered", lit_format),
        }
    }
}
