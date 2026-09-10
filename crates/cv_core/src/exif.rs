//! The little of EXIF a focal seed needs: the maker, model, focal length
//! and its 35 mm equivalent from a JPEG's APP1 segment.

/// What a picture says about the camera that took it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exif {
    pub make: Option<String>,
    pub model: Option<String>,
    /// Physical focal length, millimetres.
    pub focal_mm: Option<f64>,
    /// Focal length as on a 36 x 24 mm frame, millimetres.
    pub focal_35mm: Option<f64>,
}

impl Exif {
    /// The focal in pixels the 35 mm equivalent implies for a picture of
    /// this size: the long side spans the 36 mm frame width.
    pub fn focal_px(&self, width: u32, height: u32) -> Option<f64> {
        let f35 = self.focal_35mm.filter(|f| *f > 0.0)?;
        Some(f35 / 36.0 * f64::from(width.max(height)))
    }
}

/// The focal in pixels of a horizontal field of view over `width`.
pub fn focal_px_from_fov(fov_deg: f64, width: u32) -> f64 {
    f64::from(width) * 0.5 / (fov_deg.to_radians() * 0.5).tan()
}

/// Reads the EXIF tags from a JPEG; `None` when it carries none.
pub fn read_exif(jpeg: &[u8]) -> Option<Exif> {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return None;
    }
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            return None;
        }
        let marker = jpeg[i + 1];
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 || marker == 0xFF {
            i += if marker == 0xFF { 1 } else { 2 };
            continue;
        }
        let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
        if marker == 0xE1
            && len >= 8
            && jpeg.len() >= i + 2 + len
            && &jpeg[i + 4..i + 10] == b"Exif\0\0"
        {
            return parse_tiff(&jpeg[i + 10..i + 2 + len]);
        }
        if marker == 0xDA {
            break; // start of scan: no more headers
        }
        i += 2 + len;
    }
    None
}

fn parse_tiff(tiff: &[u8]) -> Option<Exif> {
    if tiff.len() < 8 {
        return None;
    }
    let little = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |at: usize| -> Option<u16> {
        let b = tiff.get(at..at + 2)?;
        Some(if little {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let u32_at = |at: usize| -> Option<u32> {
        let b = tiff.get(at..at + 4)?;
        Some(if little {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    };
    if u16_at(2)? != 42 {
        return None;
    }
    let mut exif = Exif::default();
    let ifd0 = u32_at(4)? as usize;
    let mut exif_ifd = None;
    let read_ifd = |at: usize, exif: &mut Exif, exif_ifd: &mut Option<usize>| -> Option<()> {
        let count = usize::from(u16_at(at)?);
        for e in 0..count {
            let entry = at + 2 + e * 12;
            let tag = u16_at(entry)?;
            let kind = u16_at(entry + 2)?;
            let n = u32_at(entry + 4)? as usize;
            let value_at = entry + 8;
            let size = match kind {
                1 | 2 | 7 => 1,
                3 => 2,
                4 | 9 => 4,
                5 | 10 => 8,
                _ => 1,
            } * n;
            let data_at = if size > 4 {
                u32_at(value_at)? as usize
            } else {
                value_at
            };
            match tag {
                0x010F | 0x0110 if kind == 2 => {
                    let bytes = tiff.get(data_at..data_at + n)?;
                    let text = String::from_utf8_lossy(bytes)
                        .trim_end_matches('\0')
                        .trim()
                        .to_string();
                    if tag == 0x010F {
                        exif.make = Some(text);
                    } else {
                        exif.model = Some(text);
                    }
                }
                0x8769 if kind == 4 => *exif_ifd = Some(u32_at(value_at)? as usize),
                0x920A if kind == 5 => {
                    let num = f64::from(u32_at(data_at)?);
                    let den = f64::from(u32_at(data_at + 4)?);
                    if den > 0.0 {
                        exif.focal_mm = Some(num / den);
                    }
                }
                0xA405 if kind == 3 => {
                    let v = u16_at(value_at)?;
                    if v > 0 {
                        exif.focal_35mm = Some(f64::from(v));
                    }
                }
                _ => {}
            }
        }
        Some(())
    };
    read_ifd(ifd0, &mut exif, &mut exif_ifd)?;
    if let Some(at) = exif_ifd {
        let mut none = None;
        read_ifd(at, &mut exif, &mut none)?;
    }
    Some(exif)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A JPEG header with one APP1 EXIF segment: IFD0 holding Make and
    /// the Exif pointer, the Exif IFD holding FocalLength 6.59 mm and a
    /// 26 mm equivalent.
    fn jpeg(little: bool) -> Vec<u8> {
        let u16b = |v: u16| {
            if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        let u32b = |v: u32| {
            if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        let mut t = Vec::new();
        t.extend_from_slice(if little { b"II" } else { b"MM" });
        t.extend_from_slice(&u16b(42));
        t.extend_from_slice(&u32b(8));
        // IFD0 at 8: two entries.
        t.extend_from_slice(&u16b(2));
        let make_at = 8 + 2 + 2 * 12 + 4; // after IFD0 and its next pointer
        let make = b"OnePlus\0";
        t.extend_from_slice(&u16b(0x010F));
        t.extend_from_slice(&u16b(2));
        t.extend_from_slice(&u32b(make.len() as u32));
        t.extend_from_slice(&u32b(make_at as u32));
        let exif_at = make_at + make.len();
        t.extend_from_slice(&u16b(0x8769));
        t.extend_from_slice(&u16b(4));
        t.extend_from_slice(&u32b(1));
        t.extend_from_slice(&u32b(exif_at as u32));
        t.extend_from_slice(&u32b(0));
        t.extend_from_slice(make);
        // Exif IFD: FocalLength (rational, out of line) and 35 mm (short, inline).
        assert_eq!(t.len(), exif_at);
        t.extend_from_slice(&u16b(2));
        let rational_at = exif_at + 2 + 2 * 12 + 4;
        t.extend_from_slice(&u16b(0x920A));
        t.extend_from_slice(&u16b(5));
        t.extend_from_slice(&u32b(1));
        t.extend_from_slice(&u32b(rational_at as u32));
        t.extend_from_slice(&u16b(0xA405));
        t.extend_from_slice(&u16b(3));
        t.extend_from_slice(&u32b(1));
        let mut inline = [0u8; 4];
        inline[..2].copy_from_slice(&u16b(26));
        t.extend_from_slice(&inline);
        t.extend_from_slice(&u32b(0));
        t.extend_from_slice(&u32b(659));
        t.extend_from_slice(&u32b(100));

        let mut out = vec![0xFF, 0xD8];
        let payload_len = 2 + 6 + t.len();
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&(payload_len as u16).to_be_bytes());
        out.extend_from_slice(b"Exif\0\0");
        out.extend_from_slice(&t);
        out.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x02]);
        out
    }

    #[test]
    fn exif_focal_reads_in_both_byte_orders() {
        for little in [true, false] {
            let exif = read_exif(&jpeg(little)).unwrap();
            assert_eq!(exif.make.as_deref(), Some("OnePlus"));
            assert_eq!(exif.model, None);
            assert!((exif.focal_mm.unwrap() - 6.59).abs() < 1e-9);
            assert_eq!(exif.focal_35mm, Some(26.0));
            // 26 mm on a 36 mm frame over 4000 px.
            assert!((exif.focal_px(4000, 3000).unwrap() - 2888.888).abs() < 0.01);
        }
        assert_eq!(read_exif(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2]), None);
        assert_eq!(read_exif(b"not a jpeg"), None);
        assert!((focal_px_from_fov(90.0, 1000) - 500.0).abs() < 1e-9);
        assert_eq!(Exif::default().focal_px(10, 10), None);
    }
}
