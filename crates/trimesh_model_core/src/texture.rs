//! Decoded texture images as the payload builder consumes them, and the
//! on-payload texel encodings.

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
        let expected = self.width as usize * self.height as usize * 3;
        if self.rgb.len() != expected {
            return Err(format!(
                "texture {}x{} has {} bytes, expected {expected}",
                self.width,
                self.height,
                self.rgb.len()
            ));
        }
        Ok(())
    }

    /// Box-filter the image to half size in each direction (odd extents
    /// round up; the last row/column averages the texels that exist).
    pub fn halved(&self) -> TextureImage {
        let (w, h) = (self.width as usize, self.height as usize);
        let nw = w.div_ceil(2);
        let nh = h.div_ceil(2);
        let mut rgb = Vec::with_capacity(nw * nh * 3);
        for y in 0..nh {
            for x in 0..nw {
                let mut sum = [0u32; 3];
                let mut n = 0u32;
                for sy in (2 * y)..(2 * y + 2).min(h) {
                    for sx in (2 * x)..(2 * x + 2).min(w) {
                        let o = (sy * w + sx) * 3;
                        for k in 0..3 {
                            sum[k] += self.rgb[o + k] as u32;
                        }
                        n += 1;
                    }
                }
                rgb.extend(sum.map(|s| ((s + n / 2) / n) as u8));
            }
        }
        TextureImage {
            width: nw as u32,
            height: nh as u32,
            rgb,
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
    fn halving_box_filters_and_rounds_odd_extents_up() {
        let img = TextureImage {
            width: 3,
            height: 2,
            rgb: vec![
                0, 0, 0, 100, 100, 100, 200, 0, 0, //
                200, 200, 200, 100, 100, 100, 0, 0, 200,
            ],
        };
        let half = img.halved();
        assert_eq!((half.width, half.height), (2, 1));
        assert_eq!(half.rgb, vec![100, 100, 100, 100, 0, 100]);
        half.validate().unwrap();

        let shrunk = img.clone().shrunk_to(1);
        assert_eq!((shrunk.width, shrunk.height), (1, 1));
        let same = img.clone().shrunk_to(0);
        assert_eq!(same, img);
        let same = img.clone().shrunk_to(3);
        assert_eq!(same, img);
    }
}
