//! Decoding what a view carries: the photograph, the 16-bit depth map and
//! the subject mask, and PNG encoding for the images this crate produces.

use anyhow::{Context, Result, anyhow};

/// An 8-bit RGB image, rows top to bottom, pixels interleaved.
#[derive(Clone, Debug, PartialEq)]
pub struct Rgb {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Rgb {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; (width * height * 3) as usize],
        }
    }

    pub fn get(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.width + x) * 3) as usize;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }

    pub fn set(&mut self, x: u32, y: u32, rgb: [u8; 3]) {
        let i = ((y * self.width + x) * 3) as usize;
        self.pixels[i..i + 3].copy_from_slice(&rgb);
    }

    /// The image resampled to `width` x `height`.
    pub fn resized(&self, width: u32, height: u32) -> Result<Rgb> {
        if (width, height) == (self.width, self.height) {
            return Ok(self.clone());
        }
        let buffer = image::RgbImage::from_raw(self.width, self.height, self.pixels.clone())
            .ok_or_else(|| anyhow!("pixel buffer does not match the image size"))?;
        let resized = image::imageops::resize(
            &buffer,
            width,
            height,
            image::imageops::FilterType::Triangle,
        );
        Ok(Rgb {
            width,
            height,
            pixels: resized.into_raw(),
        })
    }

    /// The image as JPEG bytes at `quality` (1 to 100).
    pub fn to_jpeg(&self, quality: u8) -> Result<Vec<u8>> {
        let buffer = image::RgbImage::from_raw(self.width, self.height, self.pixels.clone())
            .ok_or_else(|| anyhow!("pixel buffer does not match the image size"))?;
        let mut out = std::io::Cursor::new(Vec::new());
        let encoder =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100));
        buffer.write_with_encoder(encoder).context("encode JPEG")?;
        Ok(out.into_inner())
    }

    /// The image as PNG bytes.
    pub fn to_png(&self) -> Result<Vec<u8>> {
        let buffer = image::RgbImage::from_raw(self.width, self.height, self.pixels.clone())
            .ok_or_else(|| anyhow!("pixel buffer does not match the image size"))?;
        let mut out = std::io::Cursor::new(Vec::new());
        buffer
            .write_to(&mut out, image::ImageFormat::Png)
            .context("encode PNG")?;
        Ok(out.into_inner())
    }
}

/// The width and height of an encoded picture, read from its header.
pub fn dimensions_of(bytes: &[u8]) -> Result<(u32, u32)> {
    image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("read picture header")?
        .into_dimensions()
        .context("read picture size")
}

/// Z-depth in metres per pixel; NaN where the map has no measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Depth {
    pub width: u32,
    pub height: u32,
    pub metres: Vec<f32>,
}

impl Depth {
    pub fn get(&self, x: u32, y: u32) -> f32 {
        self.metres[(y * self.width + x) as usize]
    }

    /// How many pixels carry a measurement.
    pub fn measured(&self) -> usize {
        self.metres.iter().filter(|d| d.is_finite()).count()
    }
}

/// Where the subject is.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub inside: Vec<bool>,
}

impl Mask {
    pub fn get(&self, x: u32, y: u32) -> bool {
        self.inside[(y * self.width + x) as usize]
    }
}

/// Decodes a PNG or JPEG photograph to RGB.
pub fn decode_rgb(bytes: &[u8]) -> Result<Rgb> {
    let decoded = image::load_from_memory(bytes).context("decode image")?;
    let rgb = decoded.to_rgb8();
    Ok(Rgb {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}

/// Decodes a 16-bit grayscale PNG depth map; `unit_m` is the length of one
/// count. Zero counts mean no measurement.
pub fn decode_depth(bytes: &[u8], unit_m: f64) -> Result<Depth> {
    let decoded = image::load_from_memory(bytes).context("decode depth map")?;
    let (width, height) = (decoded.width(), decoded.height());
    let counts: Vec<u16> = match decoded {
        image::DynamicImage::ImageLuma16(buffer) => buffer.into_raw(),
        image::DynamicImage::ImageLuma8(buffer) => {
            buffer.into_raw().into_iter().map(u16::from).collect()
        }
        other => {
            return Err(anyhow!(
                "depth map must be a 16-bit grayscale PNG, not {:?}",
                other.color()
            ));
        }
    };
    let metres = counts
        .into_iter()
        .map(|c| {
            if c == 0 {
                f32::NAN
            } else {
                (f64::from(c) * unit_m) as f32
            }
        })
        .collect();
    Ok(Depth {
        width,
        height,
        metres,
    })
}

/// Decodes a mask image: any nonzero luminance is inside.
pub fn decode_mask(bytes: &[u8]) -> Result<Mask> {
    let decoded = image::load_from_memory(bytes).context("decode mask")?;
    let (width, height) = (decoded.width(), decoded.height());
    let luma = decoded.to_luma8();
    Ok(Mask {
        width,
        height,
        inside: luma.into_raw().into_iter().map(|v| v != 0).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_decodes_counts_to_metres_with_nan_holes() {
        let mut buffer = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(3, 1);
        buffer.put_pixel(0, 0, image::Luma([0]));
        buffer.put_pixel(1, 0, image::Luma([12_000]));
        buffer.put_pixel(2, 0, image::Luma([65_535]));
        let mut png = std::io::Cursor::new(Vec::new());
        buffer.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let depth = decode_depth(&png.into_inner(), 1e-4).unwrap();
        assert!(depth.get(0, 0).is_nan());
        assert!((depth.get(1, 0) - 1.2).abs() < 1e-6);
        assert!((depth.get(2, 0) - 6.5535).abs() < 1e-5);
        assert_eq!(depth.measured(), 2);
    }

    #[test]
    fn rgb_round_trips_through_png() {
        let mut rgb = Rgb::new(2, 2);
        rgb.set(1, 0, [10, 200, 30]);
        let png = rgb.to_png().unwrap();
        assert_eq!(decode_rgb(&png).unwrap(), rgb);
        let mask = decode_mask(&png).unwrap();
        assert_eq!(mask.inside, vec![false, true, false, false]);
    }
}
