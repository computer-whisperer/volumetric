//! A minimal ZIP container: the subset of PKWARE's APPNOTE that 3MF
//! packages use. Reading walks the central directory and inflates stored
//! or deflated entries; writing emits deflated (or, when smaller, stored)
//! entries with sizes in the local headers. Zip64, encryption, and
//! multi-disk archives are rejected with an error rather than misread.

use miniz_oxide::deflate::compress_to_vec;
use miniz_oxide::inflate::decompress_to_vec_with_limit;

const LOCAL_HEADER_SIG: u32 = 0x0403_4b50;
const CENTRAL_HEADER_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD_LEN: usize = 22;
const MAX_COMMENT_LEN: usize = 0xFFFF;
const METHOD_STORE: u16 = 0;
const METHOD_DEFLATE: u16 = 8;
const FLAG_ENCRYPTED: u16 = 1;
/// Largest single entry we'll inflate: a hard cap against zip bombs, far
/// above any plausible model part.
const MAX_ENTRY_BYTES: usize = 1 << 30;
/// Deflate level for written entries (6 is zlib's default trade-off).
const DEFLATE_LEVEL: u8 = 6;

/// Reflected CRC-32 (IEEE 802.3), the checksum ZIP stores per entry.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                0xEDB8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = CRC_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn u16_at(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([data[at], data[at + 1]])
}

fn u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

struct Entry {
    name: String,
    method: u16,
    flags: u16,
    crc: u32,
    compressed_size: usize,
    uncompressed_size: usize,
    local_offset: usize,
}

/// A parsed archive: the central directory over a borrowed byte slice.
pub struct Archive<'a> {
    data: &'a [u8],
    entries: Vec<Entry>,
}

impl<'a> Archive<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, String> {
        if data.len() < EOCD_LEN {
            return Err("not a ZIP archive (too short)".to_string());
        }
        // The end-of-central-directory record sits last, followed only by
        // its own (up to 64 KiB) comment: scan backwards for its signature.
        let scan_floor = data.len().saturating_sub(EOCD_LEN + MAX_COMMENT_LEN);
        let eocd = (scan_floor..=data.len() - EOCD_LEN)
            .rev()
            .find(|&at| u32_at(data, at) == EOCD_SIG)
            .ok_or_else(|| "not a ZIP archive (no end-of-central-directory record)".to_string())?;
        let disk = u16_at(data, eocd + 4);
        let entry_count = u16_at(data, eocd + 10) as usize;
        let central_size = u32_at(data, eocd + 12) as usize;
        let central_offset = u32_at(data, eocd + 16) as usize;
        if disk != 0 {
            return Err("multi-disk ZIP archives are not supported".to_string());
        }
        if entry_count == 0xFFFF || central_offset == 0xFFFF_FFFF || central_size == 0xFFFF_FFFF {
            return Err("zip64 archives are not supported".to_string());
        }
        if central_offset + central_size > eocd {
            return Err("ZIP central directory lies outside the archive".to_string());
        }

        let mut entries = Vec::with_capacity(entry_count);
        let mut at = central_offset;
        for _ in 0..entry_count {
            if at + 46 > eocd || u32_at(data, at) != CENTRAL_HEADER_SIG {
                return Err("malformed ZIP central directory".to_string());
            }
            let flags = u16_at(data, at + 8);
            let method = u16_at(data, at + 10);
            let crc = u32_at(data, at + 16);
            let compressed_size = u32_at(data, at + 20);
            let uncompressed_size = u32_at(data, at + 24);
            let name_len = u16_at(data, at + 28) as usize;
            let extra_len = u16_at(data, at + 30) as usize;
            let comment_len = u16_at(data, at + 32) as usize;
            let local_offset = u32_at(data, at + 42);
            if compressed_size == 0xFFFF_FFFF
                || uncompressed_size == 0xFFFF_FFFF
                || local_offset == 0xFFFF_FFFF
            {
                return Err("zip64 entries are not supported".to_string());
            }
            let name_end = at + 46 + name_len;
            if name_end > eocd {
                return Err("malformed ZIP central directory (entry name)".to_string());
            }
            let name = String::from_utf8_lossy(&data[at + 46..name_end]).into_owned();
            entries.push(Entry {
                name,
                method,
                flags,
                crc,
                compressed_size: compressed_size as usize,
                uncompressed_size: uncompressed_size as usize,
                local_offset: local_offset as usize,
            });
            at = name_end + extra_len + comment_len;
        }
        Ok(Self { data, entries })
    }

    /// Entry names in central-directory order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|entry| entry.name.as_str())
    }

    /// OPC part names are case-insensitive; exact matches win, then the
    /// first case-insensitive one.
    fn find(&self, name: &str) -> Option<&Entry> {
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .or_else(|| {
                self.entries
                    .iter()
                    .find(|entry| entry.name.eq_ignore_ascii_case(name))
            })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    /// The decompressed bytes of an entry, CRC-verified.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        let entry = self
            .find(name)
            .ok_or_else(|| format!("archive has no entry {name:?}"))?;
        if entry.flags & FLAG_ENCRYPTED != 0 {
            return Err(format!("entry {name:?} is encrypted"));
        }
        let data = self.data;
        let at = entry.local_offset;
        if at + 30 > data.len() || u32_at(data, at) != LOCAL_HEADER_SIG {
            return Err(format!("malformed local header for {name:?}"));
        }
        let name_len = u16_at(data, at + 26) as usize;
        let extra_len = u16_at(data, at + 28) as usize;
        let start = at + 30 + name_len + extra_len;
        let end = start + entry.compressed_size;
        if end > data.len() {
            return Err(format!("entry {name:?} is truncated"));
        }
        if entry.uncompressed_size > MAX_ENTRY_BYTES {
            return Err(format!(
                "entry {name:?} is too large ({} bytes)",
                entry.uncompressed_size
            ));
        }
        let compressed = &data[start..end];
        let bytes = match entry.method {
            METHOD_STORE => compressed.to_vec(),
            METHOD_DEFLATE => decompress_to_vec_with_limit(compressed, entry.uncompressed_size)
                .map_err(|e| format!("entry {name:?} failed to inflate: {e}"))?,
            other => {
                return Err(format!(
                    "entry {name:?} uses unsupported compression method {other}"
                ));
            }
        };
        if bytes.len() != entry.uncompressed_size {
            return Err(format!(
                "entry {name:?} inflated to {} bytes, expected {}",
                bytes.len(),
                entry.uncompressed_size
            ));
        }
        if crc32(&bytes) != entry.crc {
            return Err(format!("entry {name:?} failed its CRC check"));
        }
        Ok(bytes)
    }
}

/// Writes an archive entry by entry. Timestamps are fixed (the DOS epoch)
/// so identical input bytes produce identical archives.
pub struct Writer {
    out: Vec<u8>,
    central: Vec<u8>,
    count: u16,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            central: Vec::new(),
            count: 0,
        }
    }

    pub fn add(&mut self, name: &str, bytes: &[u8]) {
        assert!(self.count < u16::MAX, "too many entries");
        let crc = crc32(bytes);
        let deflated = compress_to_vec(bytes, DEFLATE_LEVEL);
        let (method, payload): (u16, &[u8]) = if deflated.len() < bytes.len() {
            (METHOD_DEFLATE, &deflated)
        } else {
            (METHOD_STORE, bytes)
        };
        let local_offset = self.out.len() as u32;
        let name_bytes = name.as_bytes();
        // DOS date 1980-01-01, time 00:00:00.
        let (dos_time, dos_date) = (0u16, 0x0021u16);

        let out = &mut self.out;
        out.extend_from_slice(&LOCAL_HEADER_SIG.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&dos_time.to_le_bytes());
        out.extend_from_slice(&dos_date.to_le_bytes());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra length
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(payload);

        let central = &mut self.central;
        central.extend_from_slice(&CENTRAL_HEADER_SIG.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central.extend_from_slice(&20u16.to_le_bytes()); // version needed
        central.extend_from_slice(&0u16.to_le_bytes()); // flags
        central.extend_from_slice(&method.to_le_bytes());
        central.extend_from_slice(&dos_time.to_le_bytes());
        central.extend_from_slice(&dos_date.to_le_bytes());
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        central.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes()); // extra length
        central.extend_from_slice(&0u16.to_le_bytes()); // comment length
        central.extend_from_slice(&0u16.to_le_bytes()); // disk number
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        central.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        central.extend_from_slice(&local_offset.to_le_bytes());
        central.extend_from_slice(name_bytes);
        self.count += 1;
    }

    pub fn finish(mut self) -> Vec<u8> {
        let central_offset = self.out.len() as u32;
        let central_size = self.central.len() as u32;
        self.out.extend_from_slice(&self.central);
        let out = &mut self.out;
        out.extend_from_slice(&EOCD_SIG.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // this disk
        out.extend_from_slice(&0u16.to_le_bytes()); // central directory disk
        out.extend_from_slice(&self.count.to_le_bytes()); // entries on this disk
        out.extend_from_slice(&self.count.to_le_bytes()); // entries total
        out.extend_from_slice(&central_size.to_le_bytes());
        out.extend_from_slice(&central_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment length
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_reference_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn round_trips_deflated_and_stored_entries() {
        let mut writer = Writer::new();
        let text = "hello hello hello hello hello hello".repeat(20);
        writer.add("a/text.xml", text.as_bytes());
        writer.add("tiny", b"x"); // incompressible: stored
        let bytes = writer.finish();

        let archive = Archive::parse(&bytes).unwrap();
        assert_eq!(archive.names().collect::<Vec<_>>(), ["a/text.xml", "tiny"]);
        assert_eq!(archive.read("a/text.xml").unwrap(), text.as_bytes());
        assert_eq!(archive.read("tiny").unwrap(), b"x");
        assert_eq!(archive.read("A/TEXT.XML").unwrap(), text.as_bytes());
        assert!(archive.read("missing").is_err());
        assert!(bytes.len() < text.len(), "text entry deflated");
    }

    #[test]
    fn rejects_garbage_and_corruption() {
        assert!(Archive::parse(b"not a zip").is_err());
        let mut writer = Writer::new();
        let payload: Vec<u8> = (0..=255u8).collect(); // incompressible: stored
        writer.add("f", &payload);
        let mut bytes = writer.finish();
        // Flip a payload byte (local header is 30 bytes + the 1-byte name):
        // the CRC catches it.
        bytes[31 + 10] ^= 0xFF;
        let archive = Archive::parse(&bytes).unwrap();
        assert!(archive.read("f").unwrap_err().contains("CRC"));
    }
}
