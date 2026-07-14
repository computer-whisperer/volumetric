//! Dual-blob packaging of threaded operator variants.
//!
//! A threaded operator ships as one artifact: the plain
//! wasm32-unknown-unknown module — the universal baseline every host can
//! run — with the wasm32-wasip1-threads build embedded as a custom
//! section. Custom sections are ignored by every wasm consumer by
//! construction, so the packed blob remains a valid plain module for the
//! web host while native executors extract and prefer the threaded bytes.
//! The packed blob is the artifact that travels through project files,
//! asset libraries, and content hashes.
//!
//! Packing happens in the `wasm_dist` tool; extraction in the native
//! operator executor.

/// Name of the custom section holding a complete threaded module.
pub const THREADED_SECTION: &str = "volumetric:threaded-v1";

const WASM_HEADER_LEN: usize = 8; // \0asm + 4-byte version

fn read_leb_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let mut value = 0u32;
    let mut shift = 0u32;
    loop {
        let byte = *bytes.get(*pos)?;
        *pos += 1;
        if shift >= 32 {
            return None;
        }
        value |= u32::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
        shift += 7;
    }
}

fn write_leb_u32(out: &mut Vec<u8>, mut value: u32) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// One top-level section: its id, the byte range of the whole section
/// (header included), and the byte range of its payload.
struct Section {
    id: u8,
    whole: std::ops::Range<usize>,
    payload: std::ops::Range<usize>,
}

/// Walk the top-level sections of a binary module. `None` when the bytes
/// aren't a well-formed wasm binary (e.g. WAT text, which the operator
/// pipeline also accepts — such modules simply have no embedded variant).
fn sections(bytes: &[u8]) -> Option<Vec<Section>> {
    if bytes.len() < WASM_HEADER_LEN || &bytes[0..4] != b"\0asm" {
        return None;
    }
    let mut out = Vec::new();
    let mut pos = WASM_HEADER_LEN;
    while pos < bytes.len() {
        let start = pos;
        let id = bytes[pos];
        pos += 1;
        let size = read_leb_u32(bytes, &mut pos)? as usize;
        let payload = pos..pos.checked_add(size)?;
        if payload.end > bytes.len() {
            return None;
        }
        pos = payload.end;
        out.push(Section {
            id,
            whole: start..pos,
            payload,
        });
    }
    Some(out)
}

/// The name of a custom section, with the payload data range that follows it.
fn custom_section_name(
    bytes: &[u8],
    section: &Section,
) -> Option<(String, std::ops::Range<usize>)> {
    let mut pos = section.payload.start;
    let name_len = read_leb_u32(bytes, &mut pos)? as usize;
    let name_end = pos.checked_add(name_len)?;
    if name_end > section.payload.end {
        return None;
    }
    let name = std::str::from_utf8(&bytes[pos..name_end]).ok()?;
    Some((name.to_string(), name_end..section.payload.end))
}

/// Extract the embedded threaded variant, if any. Lenient: any parse
/// irregularity reads as "no variant" and callers run the baseline.
pub fn threaded_variant(bytes: &[u8]) -> Option<&[u8]> {
    for section in sections(bytes)?.iter().filter(|s| s.id == 0) {
        if let Some((name, data)) = custom_section_name(bytes, section)
            && name == THREADED_SECTION
        {
            return Some(&bytes[data]);
        }
    }
    None
}

/// Embed `variant` into `baseline` as the threaded-variant custom section,
/// replacing any existing one — idempotent, so re-packing an already
/// packed blob is safe. Errors when `baseline` is not a binary module or
/// `variant` fails basic sanity (packing garbage would poison the
/// artifact for every native host).
pub fn embed_threaded_variant(baseline: &[u8], variant: &[u8]) -> Result<Vec<u8>, String> {
    let baseline_sections =
        sections(baseline).ok_or_else(|| "baseline is not a wasm binary module".to_string())?;
    if sections(variant).is_none() {
        return Err("threaded variant is not a wasm binary module".to_string());
    }

    let mut out = Vec::with_capacity(baseline.len() + variant.len() + 64);
    out.extend_from_slice(&baseline[..WASM_HEADER_LEN]);
    for section in &baseline_sections {
        let is_ours = section.id == 0
            && custom_section_name(baseline, section)
                .is_some_and(|(name, _)| name == THREADED_SECTION);
        if !is_ours {
            out.extend_from_slice(&baseline[section.whole.clone()]);
        }
    }

    let name = THREADED_SECTION.as_bytes();
    let payload_len = u32::try_from(1 + name.len() + variant.len())
        .map_err(|_| "threaded variant too large for a wasm section".to_string())?;
    debug_assert!(name.len() < 0x80, "name length must encode as one LEB byte");
    out.push(0); // custom section id
    write_leb_u32(&mut out, payload_len);
    write_leb_u32(&mut out, name.len() as u32);
    out.extend_from_slice(name);
    out.extend_from_slice(variant);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest valid binary module, with a distinguishing custom
    /// section so different fixtures have different bytes.
    fn module(tag: &[u8]) -> Vec<u8> {
        let mut bytes = b"\0asm\x01\0\0\0".to_vec();
        bytes.push(0);
        write_leb_u32(&mut bytes, 1 + tag.len() as u32);
        write_leb_u32(&mut bytes, tag.len() as u32);
        bytes.extend_from_slice(tag);
        bytes
    }

    #[test]
    fn embed_extract_roundtrip() {
        let baseline = module(b"base");
        let variant = module(b"fast");
        let packed = embed_threaded_variant(&baseline, &variant).unwrap();
        assert_eq!(threaded_variant(&packed), Some(variant.as_slice()));
        // The baseline's own sections are untouched.
        assert!(packed.starts_with(&baseline));
    }

    #[test]
    fn repacking_replaces_the_old_variant() {
        let baseline = module(b"base");
        let packed = embed_threaded_variant(&baseline, &module(b"old")).unwrap();
        let repacked = embed_threaded_variant(&packed, &module(b"new")).unwrap();
        assert_eq!(threaded_variant(&repacked), Some(module(b"new").as_slice()));
        assert_eq!(
            repacked.len(),
            embed_threaded_variant(&baseline, &module(b"new"))
                .unwrap()
                .len(),
            "repacking must not accumulate sections"
        );
    }

    #[test]
    fn unpacked_and_malformed_bytes_have_no_variant() {
        assert_eq!(threaded_variant(&module(b"plain")), None);
        assert_eq!(threaded_variant(b"(module)"), None);
        assert_eq!(threaded_variant(b""), None);
        // Truncated section header.
        let mut truncated = module(b"x");
        truncated.truncate(truncated.len() - 1);
        assert_eq!(threaded_variant(&truncated), None);
    }

    #[test]
    fn embedding_rejects_non_binary_inputs() {
        assert!(embed_threaded_variant(b"(module)", &module(b"v")).is_err());
        assert!(embed_threaded_variant(&module(b"b"), b"(module)").is_err());
    }
}
