//! What a JPEG's EXIF says about the exposure: maker, model, lens, focal
//! length and its 35 mm equivalent, aperture, shutter, ISO, orientation,
//! and from a Sony maker note the focus mode, the lens focus position and
//! whether stabilisation was on. The focus position is what keys a
//! still's intrinsics: a manually focused lens left alone gives the same
//! camera model frame after frame.

use volumetric_abi::viewset::Shot;

/// What a picture says about the camera that took it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exif {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    /// Physical focal length, millimetres.
    pub focal_mm: Option<f64>,
    /// Focal length as on a 36 x 24 mm frame, millimetres.
    pub focal_35mm: Option<f64>,
    pub f_number: Option<f64>,
    pub exposure_s: Option<f64>,
    pub iso: Option<u32>,
    /// The EXIF orientation value, 1 for upright.
    pub orientation: Option<u32>,
    /// `manual`, `dmf`, `af-s`, `af-c`, `af-a`, or `mode N` for a value
    /// not in the table (Sony maker note tag 0x201b).
    pub focus_mode: Option<String>,
    /// The lens focus encoder, 80 to 255 with 255 at infinity (Sony maker
    /// note block 0x9402, deciphered, byte 0x2d).
    pub focus_position: Option<u32>,
    /// SteadyShot on (Sony maker note tag 0xb026).
    pub stabilisation: Option<bool>,
}

impl Exif {
    /// The focal in pixels the 35 mm equivalent implies for a picture of
    /// this size: the long side spans the 36 mm frame width.
    pub fn focal_px(&self, width: u32, height: u32) -> Option<f64> {
        let f35 = self.focal_35mm.filter(|f| *f > 0.0)?;
        Some(f35 / 36.0 * f64::from(width.max(height)))
    }

    /// The focal in pixels from the physical focal length and the sensor
    /// width, when the picture carries the focal length.
    pub fn focal_px_on_sensor(&self, sensor_mm: f64, width: u32, height: u32) -> Option<f64> {
        let f = self.focal_mm.filter(|f| *f > 0.0)?;
        if sensor_mm.is_nan() || sensor_mm <= 0.0 {
            return None;
        }
        Some(f / sensor_mm * f64::from(width.max(height)))
    }

    /// The shot state as the view set stores it.
    pub fn to_shot(&self) -> Shot {
        Shot {
            make: self.make.clone().unwrap_or_default(),
            model: self.model.clone().unwrap_or_default(),
            lens: self.lens.clone().unwrap_or_default(),
            focal_mm: self.focal_mm.unwrap_or(0.0),
            f_number: self.f_number.unwrap_or(0.0),
            exposure_s: self.exposure_s.unwrap_or(0.0),
            iso: self.iso.unwrap_or(0),
            focus_mode: self.focus_mode.clone().unwrap_or_default(),
            focus_position: self.focus_position,
            stabilisation: self.stabilisation,
            orientation: self.orientation.unwrap_or(0),
        }
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

/// A TIFF structure: the bytes and their byte order.
struct Tiff<'a> {
    bytes: &'a [u8],
    little: bool,
}

/// One IFD entry: its tag, type, count and where its value bytes start.
struct Entry {
    tag: u16,
    kind: u16,
    count: usize,
    /// Offset of the value bytes (inline in the entry when they fit).
    data_at: usize,
    /// Offset of the entry's value field, for a pointer read.
    value_at: usize,
}

impl Tiff<'_> {
    fn u16_at(&self, at: usize) -> Option<u16> {
        let b = self.bytes.get(at..at + 2)?;
        Some(if self.little {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    }

    fn u32_at(&self, at: usize) -> Option<u32> {
        let b = self.bytes.get(at..at + 4)?;
        Some(if self.little {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    }

    fn rational_at(&self, at: usize) -> Option<f64> {
        let num = f64::from(self.u32_at(at)?);
        let den = f64::from(self.u32_at(at + 4)?);
        (den > 0.0).then_some(num / den)
    }

    /// The entries of the IFD at `at`.
    fn ifd(&self, at: usize) -> Option<Vec<Entry>> {
        let count = usize::from(self.u16_at(at)?);
        let mut out = Vec::with_capacity(count);
        for e in 0..count {
            let entry = at + 2 + e * 12;
            let tag = self.u16_at(entry)?;
            let kind = self.u16_at(entry + 2)?;
            let count = self.u32_at(entry + 4)? as usize;
            let value_at = entry + 8;
            let size = match kind {
                1 | 2 | 7 => 1,
                3 => 2,
                4 | 9 => 4,
                5 | 10 => 8,
                _ => 1,
            } * count;
            let data_at = if size > 4 {
                self.u32_at(value_at)? as usize
            } else {
                value_at
            };
            out.push(Entry {
                tag,
                kind,
                count,
                data_at,
                value_at,
            });
        }
        Some(out)
    }

    fn text(&self, e: &Entry) -> Option<String> {
        let bytes = self.bytes.get(e.data_at..e.data_at + e.count)?;
        Some(
            String::from_utf8_lossy(bytes)
                .trim_end_matches('\0')
                .trim()
                .to_string(),
        )
    }

    fn short(&self, e: &Entry) -> Option<u32> {
        match e.kind {
            3 => self.u16_at(e.value_at).map(u32::from),
            4 => self.u32_at(e.value_at),
            _ => None,
        }
    }
}

fn parse_tiff(bytes: &[u8]) -> Option<Exif> {
    if bytes.len() < 8 {
        return None;
    }
    let little = match &bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let tiff = Tiff { bytes, little };
    if tiff.u16_at(2)? != 42 {
        return None;
    }
    let mut exif = Exif::default();
    let mut exif_ifd = None;
    for e in tiff.ifd(tiff.u32_at(4)? as usize)? {
        match e.tag {
            0x010F if e.kind == 2 => exif.make = tiff.text(&e),
            0x0110 if e.kind == 2 => exif.model = tiff.text(&e),
            0x0112 => exif.orientation = tiff.short(&e),
            0x8769 if e.kind == 4 => exif_ifd = Some(tiff.u32_at(e.value_at)? as usize),
            _ => {}
        }
    }
    let Some(at) = exif_ifd else {
        return Some(exif);
    };
    let mut maker_note = None;
    for e in tiff.ifd(at)? {
        match e.tag {
            0x829A if e.kind == 5 => exif.exposure_s = tiff.rational_at(e.data_at),
            0x829D if e.kind == 5 => exif.f_number = tiff.rational_at(e.data_at),
            0x8827 => exif.iso = tiff.short(&e),
            0x920A if e.kind == 5 => exif.focal_mm = tiff.rational_at(e.data_at),
            0xA405 => exif.focal_35mm = tiff.short(&e).filter(|v| *v > 0).map(f64::from),
            0xA434 if e.kind == 2 => exif.lens = tiff.text(&e),
            0x927C if e.kind == 7 => maker_note = Some((e.data_at, e.count)),
            _ => {}
        }
    }
    if let Some((at, len)) = maker_note {
        read_sony_maker_note(&tiff, at, len, &mut exif);
    }
    Some(exif)
}

/// Sony's maker note: a 12-byte `SONY DSC ` header, then an IFD whose
/// offsets are relative to the TIFF header. Focus mode is tag 0x201b,
/// SteadyShot 0xb026, and the focus position byte 0x2d of the enciphered
/// block 0x9402 (located on the ILCE-6700 by matching exiftool's
/// FocusPosition2 over 75 frames).
fn read_sony_maker_note(tiff: &Tiff, at: usize, len: usize, exif: &mut Exif) -> Option<()> {
    let head = tiff.bytes.get(at..at + len.min(12))?;
    if !head.starts_with(b"SONY") {
        return None;
    }
    for e in tiff.ifd(at + 12)? {
        match e.tag {
            0x201B => {
                let mode = *tiff.bytes.get(e.data_at)?;
                exif.focus_mode = Some(match mode {
                    0 => "manual".to_string(),
                    2 => "af-s".to_string(),
                    3 => "af-c".to_string(),
                    4 => "af-a".to_string(),
                    6 => "dmf".to_string(),
                    other => format!("mode {other}"),
                });
            }
            0x9402 if e.kind == 7 && e.count > 0x2D => {
                let c = *tiff.bytes.get(e.data_at + 0x2D)?;
                exif.focus_position = Some(u32::from(decipher(c)));
            }
            0xB026 => exif.stabilisation = tiff.short(&e).map(|v| v == 1),
            _ => {}
        }
    }
    Some(())
}

/// Sony's cipher: a plain byte `i` below 249 is stored as `i³ mod 249`;
/// bytes 249 and up are stored as they are.
pub fn decipher(c: u8) -> u8 {
    if c >= 249 {
        return c;
    }
    (0u8..249).find(|&i| encipher(i) == c).unwrap_or(c)
}

/// The inverse of [`decipher`].
pub fn encipher(i: u8) -> u8 {
    if i >= 249 {
        return i;
    }
    let i = u32::from(i);
    ((i * i % 249) * i % 249) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A JPEG header with one APP1 EXIF segment: IFD0 holding Make, the
    /// orientation and the Exif pointer; the Exif IFD holding FocalLength
    /// 6.59 mm, a 26 mm equivalent, f/8, 1/250 s, ISO 5000 and a Sony
    /// maker note (manual focus at position 170, SteadyShot off).
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
        let entry = |t: &mut Vec<u8>, tag: u16, kind: u16, count: u32, value: [u8; 4]| {
            t.extend_from_slice(&u16b(tag));
            t.extend_from_slice(&u16b(kind));
            t.extend_from_slice(&u32b(count));
            t.extend_from_slice(&value);
        };
        let inline16 = |v: u16| {
            let mut b = [0u8; 4];
            b[..2].copy_from_slice(&u16b(v));
            b
        };
        let mut t = Vec::new();
        t.extend_from_slice(if little { b"II" } else { b"MM" });
        t.extend_from_slice(&u16b(42));
        t.extend_from_slice(&u32b(8));
        // IFD0 at 8: three entries.
        t.extend_from_slice(&u16b(3));
        let make_at = 8 + 2 + 3 * 12 + 4;
        let make = b"OnePlus\0";
        entry(&mut t, 0x010F, 2, make.len() as u32, u32b(make_at as u32));
        entry(&mut t, 0x0112, 3, 1, inline16(1));
        let exif_at = make_at + make.len();
        entry(&mut t, 0x8769, 4, 1, u32b(exif_at as u32));
        t.extend_from_slice(&u32b(0));
        t.extend_from_slice(make);
        assert_eq!(t.len(), exif_at);
        // Exif IFD: six entries; the rationals and the maker note follow.
        let n = 6;
        t.extend_from_slice(&u16b(n));
        let after = exif_at + 2 + usize::from(n) * 12 + 4;
        let focal_at = after;
        let fnum_at = after + 8;
        let exposure_at = after + 16;
        let note_at = after + 24;
        entry(&mut t, 0x829A, 5, 1, u32b(exposure_at as u32));
        entry(&mut t, 0x829D, 5, 1, u32b(fnum_at as u32));
        entry(&mut t, 0x8827, 3, 1, inline16(5000));
        entry(&mut t, 0x920A, 5, 1, u32b(focal_at as u32));
        // Maker note: 12-byte header, a 3-entry IFD, then the block.
        let block_at = note_at + 12 + 2 + 3 * 12 + 4;
        let note_len = block_at + 0x40 - note_at;
        entry(&mut t, 0x927C, 7, note_len as u32, u32b(note_at as u32));
        entry(&mut t, 0xA405, 3, 1, inline16(26));
        t.extend_from_slice(&u32b(0));
        assert_eq!(t.len(), focal_at);
        t.extend_from_slice(&u32b(659));
        t.extend_from_slice(&u32b(100));
        t.extend_from_slice(&u32b(8));
        t.extend_from_slice(&u32b(1));
        t.extend_from_slice(&u32b(1));
        t.extend_from_slice(&u32b(250));
        assert_eq!(t.len(), note_at);
        t.extend_from_slice(b"SONY DSC \0\0\0");
        t.extend_from_slice(&u16b(3));
        entry(&mut t, 0x201B, 1, 1, [0, 0, 0, 0]);
        entry(&mut t, 0x9402, 7, 0x40, u32b(block_at as u32));
        entry(&mut t, 0xB026, 4, 1, u32b(0));
        t.extend_from_slice(&u32b(0));
        assert_eq!(t.len(), block_at);
        let mut block = [0u8; 0x40];
        block[0x2D] = encipher(170);
        t.extend_from_slice(&block);

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
    fn exif_reads_in_both_byte_orders() {
        for little in [true, false] {
            let exif = read_exif(&jpeg(little)).unwrap();
            assert_eq!(exif.make.as_deref(), Some("OnePlus"));
            assert_eq!(exif.model, None);
            assert_eq!(exif.orientation, Some(1));
            assert!((exif.focal_mm.unwrap() - 6.59).abs() < 1e-9);
            assert_eq!(exif.focal_35mm, Some(26.0));
            assert_eq!(exif.f_number, Some(8.0));
            assert_eq!(exif.exposure_s, Some(0.004));
            assert_eq!(exif.iso, Some(5000));
            assert_eq!(exif.focus_mode.as_deref(), Some("manual"));
            assert_eq!(exif.focus_position, Some(170));
            assert_eq!(exif.stabilisation, Some(false));
            // 26 mm on a 36 mm frame over 4000 px.
            assert!((exif.focal_px(4000, 3000).unwrap() - 2888.888).abs() < 0.01);
            // 6.59 mm on a 6.4 mm sensor over 4000 px.
            assert!((exif.focal_px_on_sensor(6.4, 4000, 3000).unwrap() - 4118.75).abs() < 0.01);
            let shot = exif.to_shot();
            assert_eq!(shot.camera_key("x"), "OnePlus:::manual:170");
        }
        assert_eq!(read_exif(&[0xFF, 0xD8, 0xFF, 0xDA, 0, 2]), None);
        assert_eq!(read_exif(b"not a jpeg"), None);
        assert!((focal_px_from_fov(90.0, 1000) - 500.0).abs() < 1e-9);
        assert_eq!(Exif::default().focal_px(10, 10), None);
    }

    #[test]
    fn the_sony_cipher_inverts() {
        for i in 0..=255u8 {
            assert_eq!(decipher(encipher(i)), i, "{i}");
        }
        // exiftool's table: 0 -> 0, 1 -> 1, 2 -> 8, 3 -> 27.
        assert_eq!([encipher(2), encipher(3)], [8, 27]);
    }
}
