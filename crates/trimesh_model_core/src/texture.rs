//! Decoded texture images as the payload builder consumes them, and the
//! on-payload texel encodings.

use volumetric_abi::srgb::{half_side, halve};

/// How a texture's texels are stored in the payload's colour section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TextureFormat {
    /// Three bytes per texel, top row first.
    Rgb8 = 0,
    /// BC1 (DXT1) 4×4 blocks, 8 bytes each — 6× smaller than `Rgb8`, lossy.
    #[default]
    Bc1 = 1,
}

impl TextureFormat {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(TextureFormat::Rgb8),
            1 => Some(TextureFormat::Bc1),
            _ => None,
        }
    }
}

/// A decoded RGB8 image, top row first (`rgb.len() == width * height * 3`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl TextureImage {
    /// Structural check: non-empty and the right number of bytes.
    pub fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 {
            return Err(format!("texture is {}x{}", self.width, self.height));
        }
        // Checked: where `usize` is 32 bits (the operator's own target) a
        // large enough image's byte count wraps.
        let expected = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(3));
        if expected != Some(self.rgb.len()) {
            return Err(format!(
                "texture {}x{} has {} bytes, expected {}",
                self.width,
                self.height,
                self.rgb.len(),
                self.width as u64 * self.height as u64 * 3
            ));
        }
        Ok(())
    }

    /// The image at half size in each direction, averaged in linear light
    /// by the filter the viewport's mip chain uses
    /// ([`volumetric_abi::srgb::halve`]): an odd extent rounds down and
    /// its last texels join the last block.
    pub fn halved(&self) -> TextureImage {
        TextureImage {
            width: half_side(self.width),
            height: half_side(self.height),
            rgb: halve(&self.rgb, 3, self.width, self.height),
        }
    }

    /// Halve until both extents are at most `max`; `max == 0` keeps the
    /// image as is.
    pub fn shrunk_to(mut self, max: u32) -> TextureImage {
        if max == 0 {
            return self;
        }
        while self.width > max || self.height > max {
            self = self.halved();
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_averages_in_linear_light_and_rounds_odd_extents_down() {
        let img = TextureImage {
            width: 3,
            height: 2,
            rgb: vec![
                0, 0, 0, 255, 255, 255, 0, 0, 0, //
                255, 255, 255, 0, 0, 0, 0, 0, 0,
            ],
        };
        // One block of all six texels, a third of them white: 1/3 in
        // linear light is sRGB 156 (the byte mean would be 85).
        let half = img.halved();
        assert_eq!((half.width, half.height), (1, 1));
        assert_eq!(half.rgb, vec![156, 156, 156]);
        half.validate().unwrap();

        let wide = TextureImage {
            width: 8,
            height: 2,
            rgb: vec![90; 8 * 2 * 3],
        };
        let shrunk = wide.clone().shrunk_to(3);
        assert_eq!((shrunk.width, shrunk.height), (2, 1));
        assert_eq!(shrunk.rgb, vec![90; 2 * 3]);
        assert_eq!(wide.clone().shrunk_to(0), wide);
        assert_eq!(wide.clone().shrunk_to(8), wide);
    }
}
