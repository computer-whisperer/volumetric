//! CPU-side texture data for retained meshes: an sRGB RGBA8 image with its
//! full mip chain, ready to upload level by level.

/// An RGBA8 image, sRGB-encoded, top row first, with its mip chain:
/// `levels[0]` is `width × height`, each later level halves both sides
/// (rounding down, never below 1) down to 1×1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}

impl TextureData {
    /// Builds the mip chain of a `width × height` sRGB RGBA8 image (`rgba`
    /// has `width * height * 4` bytes, top row first). Each level averages
    /// 2×2 blocks of the one above in linear light; an odd side's last
    /// row or column is folded into the last block.
    ///
    /// Panics when `rgba` is not `width * height * 4` bytes or a side is 0.
    pub fn from_rgba8(width: u32, height: u32, rgba: Vec<u8>) -> TextureData {
        assert!(width > 0 && height > 0, "empty texture {width}x{height}");
        assert_eq!(
            rgba.len(),
            width as usize * height as usize * 4,
            "rgba bytes for {width}x{height}"
        );
        let mut levels = vec![rgba];
        let (mut w, mut h) = (width, height);
        while w > 1 || h > 1 {
            let (nw, nh) = (mip_side(w), mip_side(h));
            let next = downsample(levels.last().unwrap(), w, h, nw, nh);
            levels.push(next);
            (w, h) = (nw, nh);
        }
        TextureData {
            width,
            height,
            levels,
        }
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

/// Decodes an sRGB byte to linear light.
fn srgb_to_linear_lut() -> [f32; 256] {
    std::array::from_fn(|i| {
        let c = i as f32 / 255.0;
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    })
}

/// Encodes linear light as an sRGB byte.
fn linear_to_srgb_byte(linear: f32) -> u8 {
    let c = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// One mip step: `src` is `w × h`, the result `nw × nh`. Each destination
/// texel averages, in linear light, the source block `[2x, 2x+2) × [2y,
/// 2y+2)`, widened to the source's edge for the last column and row of an
/// odd side so every source texel contributes to exactly one block.
/// Alpha averages linearly.
fn downsample(src: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    let lut = srgb_to_linear_lut();
    let (w, h, nw, nh) = (w as usize, h as usize, nw as usize, nh as usize);
    let mut out = Vec::with_capacity(nw * nh * 4);
    for y in 0..nh {
        let y0 = y * 2;
        let y1 = if y + 1 == nh { h } else { y0 + 2 };
        for x in 0..nw {
            let x0 = x * 2;
            let x1 = if x + 1 == nw { w } else { x0 + 2 };
            let mut sum = [0.0f32; 4];
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = (sy * w + sx) * 4;
                    sum[0] += lut[src[i] as usize];
                    sum[1] += lut[src[i + 1] as usize];
                    sum[2] += lut[src[i + 2] as usize];
                    sum[3] += src[i + 3] as f32 / 255.0;
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f32;
            out.push(linear_to_srgb_byte(sum[0] / n));
            out.push(linear_to_srgb_byte(sum[1] / n));
            out.push(linear_to_srgb_byte(sum[2] / n));
            out.push((sum[3] / n * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Odd sides halve rounding down and the chain ends at 1×1.
    #[test]
    fn mip_chain_sizes_halve_down_to_one() {
        let tex = TextureData::from_rgba8(5, 3, vec![0; 5 * 3 * 4]);
        assert_eq!(tex.levels.len(), 3);
        assert_eq!(tex.level_size(0), (5, 3));
        assert_eq!(tex.level_size(1), (2, 1));
        assert_eq!(tex.level_size(2), (1, 1));
        assert_eq!(tex.levels[1].len(), 2 * 4);
        assert_eq!(tex.levels[2].len(), 4);

        let tall = TextureData::from_rgba8(1, 9, vec![0; 9 * 4]);
        let sizes: Vec<_> = (0..tall.levels.len()).map(|l| tall.level_size(l)).collect();
        assert_eq!(sizes, vec![(1, 9), (1, 4), (1, 2), (1, 1)]);
        let one = TextureData::from_rgba8(1, 1, vec![9, 8, 7, 6]);
        assert_eq!(one.levels, vec![vec![9, 8, 7, 6]]);
    }

    /// A constant image is the same constant at every level: the filter
    /// round-trips sRGB exactly for a uniform block.
    #[test]
    fn a_constant_image_stays_constant_through_the_chain() {
        let texel = [200u8, 37, 101, 255];
        let rgba: Vec<u8> = texel.iter().copied().cycle().take(7 * 5 * 4).collect();
        let tex = TextureData::from_rgba8(7, 5, rgba);
        for (level, bytes) in tex.levels.iter().enumerate() {
            let (w, h) = tex.level_size(level);
            assert_eq!(bytes.len(), (w * h * 4) as usize);
            for px in bytes.chunks_exact(4) {
                assert_eq!(px, texel, "level {level}");
            }
        }
    }

    /// Averaging happens in linear light: half black, half white is 50%
    /// linear, which encodes to sRGB 188, not the byte midpoint 128.
    #[test]
    fn mips_average_in_linear_light() {
        let rgba = vec![
            0, 0, 0, 255, 255, 255, 255, 255, //
            255, 255, 255, 255, 0, 0, 0, 255,
        ];
        let tex = TextureData::from_rgba8(2, 2, rgba);
        assert_eq!(tex.levels.len(), 2);
        assert_eq!(tex.levels[1], vec![188, 188, 188, 255]);
    }
}
