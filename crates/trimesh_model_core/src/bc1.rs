//! A small BC1 (DXT1) codec, pure Rust, no dependencies: the texture
//! compression the colour section of the payload uses by default.
//!
//! Only the opaque four-colour mode is emitted (endpoint order keeps
//! `c0 > c1`), so every block decodes to four interpolants of two RGB565
//! endpoints and the punch-through alpha mode never appears. The decoder
//! handles both modes anyway, per the spec, so a foreign BC1 stream would
//! read correctly too.
//!
//! The encoder is deliberately cheap — a PCA axis fit followed by a few
//! least-squares endpoint refinements, no exhaustive index search — since a
//! scanner's 8192² atlas (4M blocks) has to compress inside an operator
//! call on single-threaded wasm.

/// Bytes per 4×4 block.
pub const BLOCK_LEN: usize = 8;

/// Number of 4×4 blocks across (or down) an image `extent` texels wide.
pub fn blocks_along(extent: u32) -> usize {
    (extent as usize).div_ceil(4)
}

/// Encoded byte length of a `width` × `height` image.
pub fn encoded_len(width: u32, height: u32) -> usize {
    blocks_along(width) * blocks_along(height) * BLOCK_LEN
}

/// Encode a top-row-first RGB8 image (`rgb.len() == width * height * 3`).
/// Blocks are laid out row-major; images whose dimensions are not
/// multiples of four get edge blocks padded by replicating the last
/// in-image texel (those padding texels are encoded but never read back).
pub fn encode(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), width as usize * height as usize * 3);
    let (w, h) = (width as usize, height as usize);
    let (bw, bh) = (blocks_along(width), blocks_along(height));
    let mut out = Vec::with_capacity(bw * bh * BLOCK_LEN);
    let mut block = [[0.0f32; 3]; 16];
    for by in 0..bh {
        for bx in 0..bw {
            for (i, px) in block.iter_mut().enumerate() {
                let x = (bx * 4 + i % 4).min(w - 1);
                let y = (by * 4 + i / 4).min(h - 1);
                let o = (y * w + x) * 3;
                *px = [rgb[o] as f32, rgb[o + 1] as f32, rgb[o + 2] as f32];
            }
            out.extend(encode_block(&block));
        }
    }
    out
}

/// Decode a whole image back to top-row-first RGB8.
pub fn decode(width: u32, height: u32, data: &[u8]) -> Vec<u8> {
    assert_eq!(data.len(), encoded_len(width, height));
    let (w, h) = (width as usize, height as usize);
    let bw = blocks_along(width);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let b = (y / 4) * bw + x / 4;
            let texels = decode_block(data[b * BLOCK_LEN..(b + 1) * BLOCK_LEN].try_into().unwrap());
            let c = texels[(y % 4) * 4 + x % 4];
            out[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
        }
    }
    out
}

/// Decode one block to its 16 texels, row-major.
pub fn decode_block(block: &[u8; 8]) -> [[u8; 3]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let palette = palette(c0, c1);
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    std::array::from_fn(|i| palette[((bits >> (2 * i)) & 3) as usize])
}

/// The four palette entries of a block with endpoints `c0`, `c1` (RGB565):
/// four-colour mode when `c0 > c1`, three-colour-plus-black otherwise.
fn palette(c0: u16, c1: u16) -> [[u8; 3]; 4] {
    let p0 = expand565(c0);
    let p1 = expand565(c1);
    if c0 > c1 {
        [
            p0,
            p1,
            std::array::from_fn(|k| ((2 * p0[k] as u16 + p1[k] as u16 + 1) / 3) as u8),
            std::array::from_fn(|k| ((p0[k] as u16 + 2 * p1[k] as u16 + 1) / 3) as u8),
        ]
    } else {
        [
            p0,
            p1,
            std::array::from_fn(|k| ((p0[k] as u16 + p1[k] as u16) / 2) as u8),
            [0, 0, 0],
        ]
    }
}

fn expand565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 31) as u8;
    let g = ((c >> 5) & 63) as u8;
    let b = (c & 31) as u8;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

fn quantize565(c: [f32; 3]) -> u16 {
    let r = (c[0] / 255.0 * 31.0).round().clamp(0.0, 31.0) as u16;
    let g = (c[1] / 255.0 * 63.0).round().clamp(0.0, 63.0) as u16;
    let b = (c[2] / 255.0 * 31.0).round().clamp(0.0, 31.0) as u16;
    (r << 11) | (g << 5) | b
}

fn dist2(a: [f32; 3], b: [u8; 3]) -> f32 {
    let d: [f32; 3] = std::array::from_fn(|k| a[k] - b[k] as f32);
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Nearest palette entry for every texel.
fn assign(px: &[[f32; 3]; 16], palette: &[[u8; 3]; 4]) -> [u8; 16] {
    std::array::from_fn(|i| {
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for (k, &c) in palette.iter().enumerate() {
            let d = dist2(px[i], c);
            if d < best_d {
                best_d = d;
                best = k as u8;
            }
        }
        best
    })
}

/// Blend weights `(a, b)` of palette entry `k` in `a * c0 + b * c1`
/// (four-colour mode).
const WEIGHTS: [(f32, f32); 4] = [
    (1.0, 0.0),
    (0.0, 1.0),
    (2.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0),
];

/// Encode one block of 16 RGB texels (row-major, 0..255 floats).
pub fn encode_block(px: &[[f32; 3]; 16]) -> [u8; 8] {
    // Mean and covariance.
    let mut mean = [0.0f32; 3];
    for p in px {
        for k in 0..3 {
            mean[k] += p[k];
        }
    }
    for m in &mut mean {
        *m /= 16.0;
    }
    let mut cov = [0.0f32; 6]; // xx xy xz yy yz zz
    for p in px {
        let d: [f32; 3] = std::array::from_fn(|k| p[k] - mean[k]);
        cov[0] += d[0] * d[0];
        cov[1] += d[0] * d[1];
        cov[2] += d[0] * d[2];
        cov[3] += d[1] * d[1];
        cov[4] += d[1] * d[2];
        cov[5] += d[2] * d[2];
    }
    if cov[0] + cov[3] + cov[5] < 1e-3 {
        // Constant block: both endpoints the colour, every index 0. With
        // equal endpoints the decoder's entry 0 is still c0, whatever mode.
        let q = quantize565(mean);
        return pack(q, q, &[0; 16]);
    }

    // Principal axis by power iteration, started from the covariance
    // column of the highest-variance channel: a fixed start such as
    // (1, 1, 1) is exactly orthogonal to a red<->blue axis and would
    // collapse such a block to its mean.
    let diag = [cov[0], cov[3], cov[5]];
    let k = (0..3).max_by(|&a, &b| diag[a].total_cmp(&diag[b])).unwrap();
    let mut axis = match k {
        0 => [cov[0], cov[1], cov[2]],
        1 => [cov[1], cov[3], cov[4]],
        _ => [cov[2], cov[4], cov[5]],
    };
    for _ in 0..8 {
        let next = [
            cov[0] * axis[0] + cov[1] * axis[1] + cov[2] * axis[2],
            cov[1] * axis[0] + cov[3] * axis[1] + cov[4] * axis[2],
            cov[2] * axis[0] + cov[4] * axis[1] + cov[5] * axis[2],
        ];
        let n = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
        if n < 1e-12 {
            break;
        }
        axis = [next[0] / n, next[1] / n, next[2] / n];
    }

    // Initial endpoints: the extreme projections onto the axis.
    let mut t_min = f32::INFINITY;
    let mut t_max = f32::NEG_INFINITY;
    for p in px {
        let t = (0..3).map(|k| (p[k] - mean[k]) * axis[k]).sum::<f32>();
        t_min = t_min.min(t);
        t_max = t_max.max(t);
    }
    if t_max - t_min < 1e-3 {
        // The iteration found no spread (a degenerate start); fall back to
        // the highest-variance channel's own axis.
        axis = [0.0; 3];
        axis[k] = 1.0;
        t_min = f32::INFINITY;
        t_max = f32::NEG_INFINITY;
        for p in px {
            let t = p[k] - mean[k];
            t_min = t_min.min(t);
            t_max = t_max.max(t);
        }
    }
    let mut c0: [f32; 3] = std::array::from_fn(|k| (mean[k] + axis[k] * t_max).clamp(0.0, 255.0));
    let mut c1: [f32; 3] = std::array::from_fn(|k| (mean[k] + axis[k] * t_min).clamp(0.0, 255.0));

    // Least-squares refinement: assign indices against the quantised
    // palette, then refit the endpoints that best reproduce the texels
    // through those indices.
    for _ in 0..3 {
        let (q0, q1) = ordered(quantize565(c0), quantize565(c1));
        if q0 == q1 {
            break;
        }
        let indices = assign(px, &palette(q0, q1));
        let mut saa = 0.0f32;
        let mut sab = 0.0f32;
        let mut sbb = 0.0f32;
        let mut sap = [0.0f32; 3];
        let mut sbp = [0.0f32; 3];
        for (p, &i) in px.iter().zip(&indices) {
            let (a, b) = WEIGHTS[i as usize];
            saa += a * a;
            sab += a * b;
            sbb += b * b;
            for k in 0..3 {
                sap[k] += a * p[k];
                sbp[k] += b * p[k];
            }
        }
        let det = saa * sbb - sab * sab;
        if det.abs() < 1e-6 {
            break; // every texel on one endpoint: nothing to refit
        }
        for k in 0..3 {
            c0[k] = ((sbb * sap[k] - sab * sbp[k]) / det).clamp(0.0, 255.0);
            c1[k] = ((saa * sbp[k] - sab * sap[k]) / det).clamp(0.0, 255.0);
        }
    }

    let (q0, q1) = ordered(quantize565(c0), quantize565(c1));
    if q0 == q1 {
        return pack(q0, q1, &[0; 16]);
    }
    let indices = assign(px, &palette(q0, q1));
    pack(q0, q1, &indices)
}

/// Endpoints ordered so the four-colour mode is selected (`c0 > c1`).
fn ordered(a: u16, b: u16) -> (u16, u16) {
    if a >= b { (a, b) } else { (b, a) }
}

fn pack(c0: u16, c1: u16, indices: &[u8; 16]) -> [u8; 8] {
    let mut bits = 0u32;
    for (i, &k) in indices.iter().enumerate() {
        bits |= (k as u32 & 3) << (2 * i);
    }
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Peak signal-to-noise ratio between two RGB8 images, in dB.
    pub(crate) fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mse = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        }
    }

    fn lcg(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 56) as u8
    }

    #[test]
    fn constant_blocks_round_trip_within_565() {
        for &c in &[
            [0u8, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [17, 200, 90],
            [128, 128, 128],
        ] {
            let rgb: Vec<u8> = c.iter().copied().cycle().take(16 * 3).collect();
            let enc = encode(4, 4, &rgb);
            let dec = decode(4, 4, &enc);
            for texel in dec.chunks(3) {
                assert!(
                    (texel[0] as i32 - c[0] as i32).abs() <= 4
                        && (texel[1] as i32 - c[1] as i32).abs() <= 2
                        && (texel[2] as i32 - c[2] as i32).abs() <= 4,
                    "{c:?} decoded as {texel:?}"
                );
            }
        }
        // Pure red and blue are exact in 565: the test textures depend on it.
        let enc = encode(4, 4, &[255, 0, 0].repeat(16));
        assert_eq!(decode(4, 4, &enc), [255, 0, 0].repeat(16));
    }

    #[test]
    fn two_colour_checker_is_exact() {
        let (w, h) = (2u32, 2u32);
        let red = [255u8, 0, 0];
        let blue = [0u8, 0, 255];
        let rgb: Vec<u8> = [red, blue, blue, red].concat();
        let dec = decode(w, h, &encode(w, h, &rgb));
        assert_eq!(dec, rgb);
    }

    #[test]
    fn smooth_gradient_reaches_30_db() {
        let (w, h) = (64u32, 64u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.extend([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8]);
            }
        }
        let dec = decode(w, h, &encode(w, h, &rgb));
        let db = psnr(&rgb, &dec);
        eprintln!("BC1 gradient PSNR: {db:.2} dB");
        assert!(db > 30.0, "gradient PSNR {db:.2} dB");
    }

    #[test]
    fn random_images_do_not_panic_and_edge_blocks_are_handled() {
        let mut state = 0x5eed_u64;
        for &(w, h) in &[
            (1u32, 1u32),
            (3, 5),
            (5, 3),
            (7, 7),
            (9, 2),
            (16, 16),
            (13, 17),
        ] {
            let rgb: Vec<u8> = (0..w * h * 3).map(|_| lcg(&mut state)).collect();
            let enc = encode(w, h, &rgb);
            assert_eq!(enc.len(), encoded_len(w, h));
            let dec = decode(w, h, &enc);
            assert_eq!(dec.len(), rgb.len());
            // Random noise compresses badly, but never beyond the palette's
            // reach: every texel lands within the block's colour range.
            let db = psnr(&rgb, &dec);
            assert!(db > 10.0, "{w}x{h} random PSNR {db:.2} dB");
        }
        // A non-multiple-of-4 constant image stays constant (within 565):
        // padding texels must not leak into the real ones.
        let rgb = [40u8, 90, 200].repeat(5 * 3);
        let dec = decode(5, 3, &encode(5, 3, &rgb));
        for texel in dec.chunks(3) {
            assert_eq!(texel, &dec[0..3]);
            assert!((texel[0] as i32 - 40).abs() <= 4, "{texel:?}");
            assert!((texel[1] as i32 - 90).abs() <= 2, "{texel:?}");
            assert!((texel[2] as i32 - 200).abs() <= 4, "{texel:?}");
        }
    }

    #[test]
    fn three_colour_mode_decodes_per_spec() {
        // c0 <= c1 selects the three-colour mode: entry 2 is the average,
        // entry 3 black.
        let c0 = quantize565([0.0, 0.0, 0.0]);
        let c1 = quantize565([255.0, 255.0, 255.0]);
        let mut block = [0u8; 8];
        block[0..2].copy_from_slice(&c0.to_le_bytes());
        block[2..4].copy_from_slice(&c1.to_le_bytes());
        block[4] = 0b11_10_01_00;
        let texels = decode_block(&block);
        assert_eq!(texels[0], [0, 0, 0]);
        assert_eq!(texels[1], [255, 255, 255]);
        assert_eq!(texels[2], [127, 127, 127]);
        assert_eq!(texels[3], [0, 0, 0]);
    }

    /// Encoder throughput on a synthetic 1024² image; run with
    /// `cargo test --release -p trimesh_model_core bc1_timing -- --ignored --nocapture`
    /// and scale by 64 for an 8192² atlas.
    #[test]
    #[ignore]
    fn bc1_timing() {
        let (w, h) = (1024u32, 1024u32);
        let mut state = 7u64;
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let n = lcg(&mut state) / 8;
                rgb.extend([
                    ((x / 4) as u8).wrapping_add(n),
                    ((y / 4) as u8).wrapping_add(n),
                    (((x + y) / 8) as u8).wrapping_add(n),
                ]);
            }
        }
        let start = std::time::Instant::now();
        let enc = encode(w, h, &rgb);
        let elapsed = start.elapsed();
        let db = psnr(&rgb, &decode(w, h, &enc));
        eprintln!("BC1 1024^2: {elapsed:?} ({db:.2} dB)");
    }
}
