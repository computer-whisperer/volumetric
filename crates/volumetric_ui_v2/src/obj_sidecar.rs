//! Finding an OBJ's texture the way OBJ viewers do: the user picks the
//! `.obj`, and the program follows `mtllib` to the material library and
//! `map_Kd` to the image. The operator has no filesystem, so this runs in
//! the host (native: next to the picked file; web: among the files picked
//! together) and the image bytes travel to the operator's texture input.
//!
//! Pure over a `read` callback so both hosts share it and it tests
//! without a disk.

/// What was found beside an OBJ.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ObjSidecars {
    /// The material library that named the texture, when one was read.
    pub mtl: Option<String>,
    /// The base-colour image: its file name (final path component) and
    /// bytes.
    pub texture: Option<(String, Vec<u8>)>,
    /// How the search went, for the status line: what was tried and what
    /// was missing. Empty when nothing was looked for.
    pub notes: Vec<String>,
}

/// Resolves the sidecars of `obj_text`. `obj_stem` is the picked file's
/// name without extension (the fallback texture name scanners use);
/// `read` returns the bytes of a path relative to the OBJ, or `None` when
/// there is no such file.
///
/// Order: every `mtllib` in the file, first `map_Kd` found (then, failing
/// that, `map_Ka`), the path as written, then its bare file name next to
/// the OBJ; without a material library or a readable map, an image named
/// after the OBJ itself.
pub fn resolve(
    obj_text: &str,
    obj_stem: &str,
    read: &mut dyn FnMut(&str) -> Option<Vec<u8>>,
) -> ObjSidecars {
    let mut out = ObjSidecars::default();

    let libraries = mtllib_names(obj_text);
    for library in &libraries {
        let Some(mtl_bytes) = read_either(library, read) else {
            out.notes
                .push(format!("material library {library} not found"));
            continue;
        };
        let mtl_text = String::from_utf8_lossy(&mtl_bytes);
        let Some(map) = diffuse_map(&mtl_text) else {
            out.notes.push(format!("{library} names no texture"));
            continue;
        };
        match read_either(&map, read) {
            Some(bytes) => {
                out.mtl = Some(library.clone());
                out.texture = Some((file_name(&map).to_string(), bytes));
                return out;
            }
            None => out
                .notes
                .push(format!("{library} names {map}, which was not found")),
        }
    }

    for extension in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"] {
        let candidate = format!("{obj_stem}.{extension}");
        if let Some(bytes) = read(&candidate) {
            out.notes.push(format!("using {candidate} by name"));
            out.texture = Some((candidate, bytes));
            return out;
        }
    }
    if libraries.is_empty() {
        out.notes.push("no mtllib line".to_string());
    }
    out
}

/// A path as written, then its bare file name (exporters often write a
/// path from the machine that made the file; the sidecar is next to the
/// OBJ anyway).
fn read_either(path: &str, read: &mut dyn FnMut(&str) -> Option<Vec<u8>>) -> Option<Vec<u8>> {
    let normalised = path.replace('\\', "/");
    if let Some(bytes) = read(&normalised) {
        return Some(bytes);
    }
    let name = file_name(&normalised);
    if name != normalised {
        return read(name);
    }
    None
}

/// The final component of a path written with either separator.
pub fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Every `mtllib` operand in the file, in order. A line may name several
/// libraries separated by spaces, but file names with spaces are far more
/// common than multiple libraries, so the whole operand is one name and
/// the split is tried only by the caller's read failing (see tests).
fn mtllib_names(obj_text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in obj_text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("mtllib") else {
            continue;
        };
        if !rest.starts_with(char::is_whitespace) {
            continue;
        }
        let rest = rest.trim();
        if !rest.is_empty() && !names.iter().any(|n| n == rest) {
            names.push(rest.to_string());
        }
    }
    names
}

/// The first `map_Kd` file name in a material library, or the first
/// `map_Ka` when there is no diffuse map. Texture-map options
/// (`-s u v w`, `-o u v w`, `-mm base gain`, `-blendu on`, ...) precede
/// the file name and are skipped; the file name is the remainder, so names
/// with spaces survive.
fn diffuse_map(mtl_text: &str) -> Option<String> {
    let find = |keyword: &str| -> Option<String> {
        mtl_text.lines().find_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix(keyword)?;
            if !rest.starts_with(char::is_whitespace) {
                return None;
            }
            let name = strip_map_options(rest.trim());
            (!name.is_empty()).then(|| name.to_string())
        })
    };
    find("map_Kd").or_else(|| find("map_Ka"))
}

/// Skips the leading `-option args` groups of a texture-map statement.
fn strip_map_options(mut rest: &str) -> &str {
    loop {
        if !rest.starts_with('-') {
            return rest;
        }
        let mut tokens = rest.split_whitespace();
        let option = tokens.next().unwrap_or("");
        // Options with a fixed argument count; `-o`, `-s` and `-t` take
        // one to three numbers, consumed while they parse.
        let fixed = match option {
            "-blendu" | "-blendv" | "-cc" | "-clamp" | "-imfchan" | "-bm" | "-boost"
            | "-texres" => Some(1),
            "-mm" => Some(2),
            "-o" | "-s" | "-t" => None,
            _ => Some(0),
        };
        let consumed = match fixed {
            Some(n) => n,
            None => tokens
                .clone()
                .take(3)
                .take_while(|t| t.parse::<f64>().is_ok())
                .count(),
        };
        // Advance past the option and its arguments by token boundaries.
        let mut after = rest;
        for _ in 0..=consumed {
            after = after.trim_start();
            let end = after.find(char::is_whitespace).unwrap_or(after.len());
            after = &after[end..];
        }
        rest = after.trim_start();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn disk(files: &[(&str, &[u8])]) -> HashMap<String, Vec<u8>> {
        files
            .iter()
            .map(|(name, bytes)| (name.to_string(), bytes.to_vec()))
            .collect()
    }

    fn resolve_on(obj: &str, stem: &str, files: &HashMap<String, Vec<u8>>) -> ObjSidecars {
        resolve(obj, stem, &mut |path| files.get(path).cloned())
    }

    /// The scanner layout: one OBJ, one MTL, one JPEG, all named alike.
    #[test]
    fn follows_mtllib_and_map_kd() {
        let files = disk(&[
            (
                "7319_scan.mtl",
                b"newmtl scan\nKd 1 1 1\nmap_Kd 7319_scan.jpg\n",
            ),
            ("7319_scan.jpg", b"\xFF\xD8jpeg"),
        ]);
        let found = resolve_on("mtllib 7319_scan.mtl\nv 0 0 0\n", "7319_scan", &files);
        assert_eq!(found.mtl.as_deref(), Some("7319_scan.mtl"));
        assert_eq!(
            found.texture,
            Some(("7319_scan.jpg".to_string(), b"\xFF\xD8jpeg".to_vec()))
        );
        assert!(found.notes.is_empty());
    }

    /// A map written with a foreign path falls back to its bare name next
    /// to the OBJ, and options before the name are skipped.
    #[test]
    fn strips_paths_and_options() {
        let files = disk(&[
            (
                "m.mtl",
                b"newmtl a\nmap_Kd -s 1 1 1 -clamp on C:\\Users\\x\\tex atlas.png\n",
            ),
            ("tex atlas.png", b"\x89PNGpng"),
        ]);
        let found = resolve_on("mtllib m.mtl\n", "m", &files);
        assert_eq!(
            found.texture,
            Some(("tex atlas.png".to_string(), b"\x89PNGpng".to_vec()))
        );
    }

    #[test]
    fn map_options_parse() {
        assert_eq!(strip_map_options("-s 1 1 1 -o 0.5 0.5 a.png"), "a.png");
        assert_eq!(strip_map_options("-mm 0 1 -blendu off b c.jpg"), "b c.jpg");
        assert_eq!(strip_map_options("-o 1 x.png"), "x.png");
        assert_eq!(strip_map_options("plain.png"), "plain.png");
        assert_eq!(strip_map_options("-bogus plain.png"), "plain.png");
    }

    /// Without a usable library, an image named after the OBJ is taken.
    #[test]
    fn falls_back_to_the_obj_name() {
        let files = disk(&[("scan.jpg", b"\xFF\xD8")]);
        let found = resolve_on("v 0 0 0\n", "scan", &files);
        assert_eq!(
            found.texture.as_ref().map(|(n, _)| n.as_str()),
            Some("scan.jpg")
        );
        assert_eq!(found.mtl, None);
        assert_eq!(found.notes, vec!["using scan.jpg by name".to_string()]);

        let files = disk(&[("other.mtl", b"newmtl a\nmap_Kd gone.png\n")]);
        let found = resolve_on("mtllib other.mtl\n", "scan", &files);
        assert_eq!(found.texture, None);
        assert_eq!(
            found.notes,
            vec!["other.mtl names gone.png, which was not found".to_string()]
        );
    }

    #[test]
    fn missing_library_is_noted() {
        let files = disk(&[]);
        let found = resolve_on("mtllib none.mtl\n", "x", &files);
        assert_eq!(found.texture, None);
        assert_eq!(
            found.notes,
            vec!["material library none.mtl not found".to_string()]
        );
        let found = resolve_on("v 0 0 0\n", "x", &files);
        assert_eq!(found.notes, vec!["no mtllib line".to_string()]);
    }

    #[test]
    fn mtllib_operand_is_one_name() {
        assert_eq!(
            mtllib_names("mtllib my scan.mtl\nmtllibx\nmtllib my scan.mtl\n"),
            vec!["my scan.mtl".to_string()]
        );
        assert_eq!(file_name("a/b\\c.png"), "c.png");
    }
}
