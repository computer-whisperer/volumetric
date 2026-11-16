//! sRGB image filtering: the one box filter every texture path shares.
//!
//! Colour travels sRGB-encoded (mesh textures, the sRGB sample channels),
//! but averaging texels is only right in linear light: the mean of black
//! and white is a 50 % grey, which encodes to 188, not 128. The viewport's
//! mip chain and the texture a model payload stores are both made here, so
//! a scan looks the same shrunk on the GPU as shrunk in a payload.

/// The side of the next smaller level: half, rounding down, never below 1.
pub fn half_side(side: u32) -> u32 {
    (side / 2).max(1)
}

/// The linear-light value of every sRGB byte.
fn decode_table() -> [f32; 256] {
    std::array::from_fn(|i| {
        let c = i as f32 / 255.0;
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    })
}

/// The linear-light value halfway (in sRGB) between byte `i` and `i + 1`:
/// a linear value encodes to the number of these it reaches. Counting
/// them is rounding to the nearest sRGB byte without a `powf` per texel,
/// which matters for a 64-megatexel atlas halved in single-threaded wasm.
fn encode_thresholds() -> [f32; 255] {
    std::array::from_fn(|i| {
        let c = (i as f32 + 0.5) / 255.0;
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    })
}

/// The sRGB byte nearest to `linear`.
fn encode(thresholds: &[f32; 255], linear: f32) -> u8 {
    thresholds.partition_point(|&t| t <= linear) as u8
}

/// Halves an sRGB image: `src` is `width × height` texels of `channels`
/// bytes each, top row first, and the result is `half_side(width) ×
/// half_side(height)`. Each result texel is the mean, in linear light, of
/// the source block `[2x, 2x + 2) × [2y, 2y + 2)`; for an odd side the
/// last block reaches the source's edge, so every source texel counts in
/// exactly one block. With four channels the fourth is alpha and averages
/// as it is; with three there is none.
///
/// Panics when `channels` is not 3 or 4, a side is 0, or `src` is not
/// `width * height * channels` bytes.
pub fn halve(src: &[u8], channels: usize, width: u32, height: u32) -> Vec<u8> {
    assert!(channels == 3 || channels == 4, "{channels} channels");
    assert!(width > 0 && height > 0, "empty image {width}x{height}");
    let (w, h) = (width as usize, height as usize);
    // Checked: where `usize` is 32 bits a large enough image's byte count
    // wraps, and an empty slice would pass for it.
    let expected = w.checked_mul(h).and_then(|n| n.checked_mul(channels));
    assert_eq!(Some(src.len()), expected, "bytes for {width}x{height}");
    let (nw, nh) = (half_side(width) as usize, half_side(height) as usize);
    let decode = decode_table();
    let thresholds = encode_thresholds();

    let mut out = Vec::with_capacity(nw * nh * channels);
    for y in 0..nh {
        let y0 = y * 2;
        let y1 = if y + 1 == nh { h } else { y0 + 2 };
        for x in 0..nw {
            let x0 = x * 2;
            let x1 = if x + 1 == nw { w } else { x0 + 2 };
            let mut colour = [0.0f32; 3];
            let mut alpha = 0u32;
            for sy in y0..y1 {
                let row = &src[(sy * w + x0) * channels..(sy * w + x1) * channels];
                for texel in row.chunks_exact(channels) {
                    colour[0] += decode[texel[0] as usize];
                    colour[1] += decode[texel[1] as usize];
                    colour[2] += decode[texel[2] as usize];
                    if channels == 4 {
                        alpha += texel[3] as u32;
                    }
                }
            }
            let n = (y1 - y0) * (x1 - x0);
            out.extend(colour.map(|sum| encode(&thresholds, sum / n as f32)));
            if channels == 4 {
                out.push(((alpha + n as u32 / 2) / n as u32) as u8);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte survives a decode and an encode, so a flat image stays
    /// exactly flat however often it is halved.
    #[test]
    fn every_byte_round_trips() {
        let decode = decode_table();
        let thresholds = encode_thresholds();
        for byte in 0..=255u8 {
            assert_eq!(encode(&thresholds, decode[byte as usize]), byte);
        }
        assert_eq!(encode(&thresholds, -1.0), 0);
        assert_eq!(encode(&thresholds, 2.0), 255);
    }

    /// Half black, half white is 50 % in linear light, which is sRGB 188,
    /// not the byte midpoint 128. Alpha averages as it is.
    #[test]
    fn averages_in_linear_light() {
        let rgba = [
            0, 0, 0, 255, 255, 255, 255, 255, //
            255, 255, 255, 0, 0, 0, 0, 0,
        ];
        assert_eq!(halve(&rgba, 4, 2, 2), vec![188, 188, 188, 128]);
        let rgb = [0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0];
        assert_eq!(halve(&rgb, 3, 2, 2), vec![188, 188, 188]);
    }

    /// An odd side rounds down and its last block takes the extra row or
    /// column, so no texel is left out.
    #[test]
    fn an_odd_side_folds_its_last_texels_into_the_last_block() {
        // 5 × 3: one row of two blocks, the second three texels wide and
        // all three rows tall.
        let mut rgb = vec![0u8; 5 * 3 * 3];
        // Light only the last column, which a plain 2×2 walk never reads.
        for y in 0..3 {
            let i = (y * 5 + 4) * 3;
            rgb[i..i + 3].copy_from_slice(&[255, 255, 255]);
        }
        let half = halve(&rgb, 3, 5, 3);
        assert_eq!(half.len(), 2 * 3);
        assert_eq!(&half[0..3], &[0, 0, 0]);
        // A third of the block is white: 1/3 linear is sRGB 156.
        assert_eq!(&half[3..6], &[156, 156, 156]);

        assert_eq!(half_side(1), 1);
        assert_eq!(halve(&[9, 8, 7, 6], 4, 1, 1), vec![9, 8, 7, 6]);
    }
}
