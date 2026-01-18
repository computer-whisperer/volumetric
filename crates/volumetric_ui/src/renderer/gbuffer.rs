//! G-Buffer management for deferred rendering.
//!
//! The G-buffer stores intermediate rendering data:
//! - Color: Lit diffuse color from mesh rendering
//! - Normal: World-space normals encoded to [0,1]
//! - Depth: Linear depth for SSAO sampling

/// G-buffer textures for deferred rendering.
pub struct GBuffer {
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub normal_texture: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// Hardware depth-stencil for depth testing
    pub depth_stencil_texture: wgpu::Texture,
    pub depth_stencil_view: wgpu::TextureView,
    /// Current size
    pub size: (u32, u32),
    /// Format for the color attachment (matches surface format)
    pub color_format: wgpu::TextureFormat,
}

impl GBuffer {
    /// Create a new G-buffer with the given size and color format.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let size = (width.max(1), height.max(1));
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };

        // Color texture (lit diffuse)
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Normal texture (encoded world normals)
        let normal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_normal"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth texture (linear depth for SSAO)
        // On web, R32Float as render target may not be supported, so use Rgba16Float
        // which has better compatibility while still providing good precision.
        #[cfg(target_arch = "wasm32")]
        let depth_format = wgpu::TextureFormat::Rgba16Float;
        #[cfg(not(target_arch = "wasm32"))]
        let depth_format = wgpu::TextureFormat::R32Float;

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Hardware depth-stencil for actual depth testing
        let depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer_depth_stencil"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_stencil_view =
            depth_stencil_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            color_texture,
            color_view,
            normal_texture,
            normal_view,
            depth_texture,
            depth_view,
            depth_stencil_texture,
            depth_stencil_view,
            size,
            color_format,
        }
    }

    /// Resize the G-buffer if the size has changed.
    /// Returns true if the buffer was resized.
    pub fn resize_if_needed(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> bool {
        let new_size = (width.max(1), height.max(1));
        if self.size == new_size {
            return false;
        }

        // Recreate all textures with new size
        *self = Self::new(device, new_size.0, new_size.1, self.color_format);
        true
    }

    /// Get texture extent.
    pub fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.size.0,
            height: self.size.1,
            depth_or_array_layers: 1,
        }
    }
}

/// AO (ambient occlusion) texture for SSAO output.
pub struct AoTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// Dummy white texture for when SSAO is disabled
    pub dummy_texture: wgpu::Texture,
    pub dummy_view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl AoTexture {
    /// Create a new AO texture with the given size.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = (width.max(1), height.max(1));
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };

        // AO texture (single channel)
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ao_texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Dummy 1x1 white texture for when SSAO is disabled
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ao_dummy_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            dummy_texture,
            dummy_view,
            size,
        }
    }

    /// Resize if needed. Returns true if resized.
    pub fn resize_if_needed(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let new_size = (width.max(1), height.max(1));
        if self.size == new_size {
            return false;
        }

        *self = Self::new(device, new_size.0, new_size.1);
        true
    }
}
