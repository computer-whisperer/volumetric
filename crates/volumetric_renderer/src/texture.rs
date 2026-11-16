//! Mesh textures: the pixels a scene hands over, and the copy the device
//! keeps.
//!
//! [`TextureData`] is an image with its mip chain as given; the renderer
//! filters nothing itself. [`SceneTexture`] is what a mesh in a scene
//! holds: those pixels until the first upload, the device's copy after.

use std::sync::Mutex;

/// An RGBA8 image, sRGB-encoded, top row first, with mip levels:
/// `levels[0]` is `width × height` and each later level halves both sides
/// (rounding down, never below 1). The chain may stop anywhere; sampling
/// below its last level reads that level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}

impl TextureData {
    /// An image from its levels, largest first.
    ///
    /// Panics when a side is 0, there is no level, a level is not the
    /// `4 × width × height` bytes its place in the chain gives it, or a
    /// level follows the 1×1 one.
    pub fn from_levels(width: u32, height: u32, levels: Vec<Vec<u8>>) -> TextureData {
        assert!(width > 0 && height > 0, "empty texture {width}x{height}");
        assert!(!levels.is_empty(), "texture {width}x{height} has no level");
        let (mut w, mut h) = (width, height);
        for (level, bytes) in levels.iter().enumerate() {
            assert_eq!(
                bytes.len(),
                w as usize * h as usize * 4,
                "rgba bytes for level {level} ({w}x{h}) of {width}x{height}"
            );
            assert!(
                (w, h) != (1, 1) || level + 1 == levels.len(),
                "texture {width}x{height} has a level after its 1x1 level {level}"
            );
            (w, h) = (mip_side(w), mip_side(h));
        }
        TextureData {
            width,
            height,
            levels,
        }
    }

    /// A single texel of one colour.
    pub fn solid(rgba: [u8; 4]) -> TextureData {
        TextureData::from_levels(1, 1, vec![rgba.to_vec()])
    }

    /// The size of mip `level`.
    pub fn level_size(&self, level: usize) -> (u32, u32) {
        let (mut w, mut h) = (self.width, self.height);
        for _ in 0..level {
            (w, h) = (mip_side(w), mip_side(h));
        }
        (w, h)
    }
}

/// The side of the next mip level: half, rounding down, never below 1.
pub fn mip_side(side: u32) -> u32 {
    (side / 2).max(1)
}

/// Makes a texture's pixels again, or `None` when it cannot.
type Reload = Box<dyn Fn() -> Option<TextureData> + Send + Sync>;

/// A mesh's texture as a scene holds it.
///
/// A scanner's atlas is hundreds of megabytes decoded, and a scene stays
/// cached long after its first frame, so the pixels are not kept twice.
/// The first upload hands them to the device and the texture keeps that
/// copy, which every later upload of the mesh shares (hiding an output
/// and showing it again sends nothing). The decoded pixels are then
/// dropped when the texture knows how to make them again
/// ([`reloadable`](Self::reloadable)) and kept when it does not
/// ([`new`](Self::new)). They are needed again only if a different
/// renderer draws the scene.
pub struct SceneTexture {
    width: u32,
    height: u32,
    reload: Option<Reload>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    pixels: Option<TextureData>,
    /// The copy on a device, and the mesh pipeline it was made for.
    resident: Option<(PipelineId, ResidentTexture)>,
}

/// Which mesh pipeline a resident texture belongs to. Devices cannot be
/// told apart by comparing them (two instances number theirs alike), so
/// each pipeline, which lives on one device, takes a number of its own.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct PipelineId(u64);

impl PipelineId {
    /// A number no other pipeline in this process has.
    pub fn next() -> PipelineId {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        PipelineId(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

/// A texture on a device, bound for the mesh pass.
#[derive(Clone)]
pub(crate) struct ResidentTexture {
    /// Group 1 of the mesh pass; it keeps the texture alive.
    pub bind_group: std::sync::Arc<wgpu::BindGroup>,
    /// The size it was uploaded at when its full size exceeded the
    /// device's largest texture side; `None` when it fit.
    pub downscaled: Option<(u32, u32)>,
}

impl SceneTexture {
    /// A texture that keeps `pixels` for as long as it lives.
    pub fn new(pixels: TextureData) -> SceneTexture {
        SceneTexture {
            width: pixels.width,
            height: pixels.height,
            reload: None,
            state: Mutex::new(State {
                pixels: Some(pixels),
                resident: None,
            }),
        }
    }

    /// A texture that drops `pixels` once a device has them and calls
    /// `reload` if they are wanted again. `reload` must return the same
    /// image; when it returns `None` the mesh is drawn untextured.
    pub fn reloadable(
        pixels: TextureData,
        reload: impl Fn() -> Option<TextureData> + Send + Sync + 'static,
    ) -> SceneTexture {
        SceneTexture {
            reload: Some(Box::new(reload)),
            ..SceneTexture::new(pixels)
        }
    }

    /// The image's size, `(width, height)`.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Whether the decoded pixels are in memory now.
    pub fn holds_pixels(&self) -> bool {
        self.state().pixels.is_some()
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        // Nothing is taken out of the state before what replaces it
        // exists, so a panic while it was held leaves it whole.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The texture as `pipeline` binds it: the copy already made for it,
    /// or the one `make` builds from the pixels now (which replaces a
    /// copy made for any other pipeline). `None` when the pixels are gone
    /// and cannot be made again.
    pub(crate) fn resident(
        &self,
        pipeline: PipelineId,
        make: impl FnOnce(&TextureData) -> ResidentTexture,
    ) -> Option<ResidentTexture> {
        let mut state = self.state();
        if let Some((made_for, resident)) = &state.resident
            && *made_for == pipeline
        {
            return Some(resident.clone());
        }
        // The pixels are given up only once a copy has been made of
        // them, so a `make` that panics leaves the texture as it was.
        let resident = match (&state.pixels, &self.reload) {
            (Some(pixels), _) => make(pixels),
            (None, Some(reload)) => make(&reload()?),
            (None, None) => return None,
        };
        if self.reload.is_some() {
            state.pixels = None;
        }
        state.resident = Some((pipeline, resident.clone()));
        Some(resident)
    }
}

impl std::fmt::Debug for SceneTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state();
        f.debug_struct("SceneTexture")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("reloadable", &self.reload.is_some())
            .field("holds_pixels", &state.pixels.is_some())
            .field("resident", &state.resident.is_some())
            .finish()
    }
}

/// Uploads `pixels` as an sRGB RGBA8 texture with its mip levels, so
/// sampling it returns linear light. When level 0 exceeds the device's
/// largest texture side the top levels are dropped until it fits (or
/// there is no smaller level to fall back to), and the size uploaded is
/// returned alongside.
pub(crate) fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &TextureData,
) -> (wgpu::Texture, Option<(u32, u32)>) {
    let max_side = device.limits().max_texture_dimension_2d;
    let (mut width, mut height) = (pixels.width, pixels.height);
    let mut first = 0;
    while (width > max_side || height > max_side) && first + 1 < pixels.levels.len() {
        (width, height) = (mip_side(width), mip_side(height));
        first += 1;
    }
    let downscaled = (first > 0).then_some((width, height));
    let levels = &pixels.levels[first..];

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("retained_mesh_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let (mut w, mut h) = (width, height);
    for (level, bytes) in levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        (w, h) = (mip_side(w), mip_side(h));
    }
    (texture, downscaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Levels halve rounding down to 1×1, and a chain may stop early.
    #[test]
    fn levels_are_checked_against_their_place_in_the_chain() {
        let tex =
            TextureData::from_levels(5, 3, vec![vec![0; 5 * 3 * 4], vec![0; 2 * 4], vec![0; 4]]);
        assert_eq!(tex.level_size(0), (5, 3));
        assert_eq!(tex.level_size(1), (2, 1));
        assert_eq!(tex.level_size(2), (1, 1));

        let short = TextureData::from_levels(1, 9, vec![vec![0; 9 * 4], vec![0; 4 * 4]]);
        assert_eq!(short.levels.len(), 2);
        assert_eq!(short.level_size(1), (1, 4));

        assert_eq!(
            TextureData::solid([9, 8, 7, 6]).levels,
            vec![vec![9, 8, 7, 6]]
        );
    }

    #[test]
    #[should_panic(expected = "level 1 (2x1)")]
    fn a_level_of_the_wrong_size_is_refused() {
        TextureData::from_levels(5, 3, vec![vec![0; 5 * 3 * 4], vec![0; 4]]);
    }

    #[test]
    #[should_panic(expected = "after its 1x1 level 1")]
    fn a_level_past_one_texel_is_refused() {
        TextureData::from_levels(2, 1, vec![vec![0; 8], vec![0; 4], vec![0; 4]]);
    }
}
