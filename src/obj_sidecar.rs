//! Finding an OBJ's texture the way OBJ viewers do: the user picks the
//! `.obj`, and the program follows `mtllib` to the material library and
//! `map_Kd` to the image. The operator has no filesystem, so this runs in
//! the host (native: next to the picked file; web: among the files picked
//! together) and the image bytes travel to the operator's texture input.
//!
//! An OBJ and its material library are untrusted text that names files.
//! Nothing they say can make a host read outside the OBJ's own folder: a
//! name is followed only as a relative path of plain components (no root,
//! no drive, no `..`), then as its bare file name; every read carries a
//! size limit; and bytes that are not a JPEG or PNG image are not taken
//! as one.
//!
//! [`resolve`] is pure over a `read` callback, so every host shares it
//! and it tests without a disk. [`resolve_beside`] is the native reader:
//! the GUI, the CLI and the Python bindings can all hand it a path.

use volumetric_abi::trimesh::MeshTexture;

/// The largest material library read, in bytes. They are a few lines of
/// text; anything larger is not one.
pub const MAX_MTL_BYTES: u64 = 4 << 20;

/// The largest texture file read, in bytes. A 16k atlas as PNG is under
/// this; a path that names something unbounded is not read to the end.
pub const MAX_TEXTURE_BYTES: u64 = 512 << 20;

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

/// An OBJ and what was found beside it, as `obj_import_operator` takes
/// them.
#[derive(Debug, PartialEq, Eq)]
pub struct ObjImport {
    /// The operator's Blob inputs in order: the OBJ, then the texture
    /// when there is one.
    pub blobs: Vec<Vec<u8>>,
    /// Config text fields to set: the texture's file name.
    pub config: Vec<(String, String)>,
    /// One line for the status bar: the texture and how it was found, or
    /// why there is none.
    pub note: String,
}

impl ObjSidecars {
    /// The import these sidecars make of `obj_bytes`.
    pub fn into_import(self, obj_bytes: Vec<u8>) -> ObjImport {
        match self.texture {
            Some((name, bytes)) => {
                let note = match &self.mtl {
                    Some(mtl) => format!("texture {name} via {mtl}"),
                    None => format!("texture {name}"),
                };
                ObjImport {
                    blobs: vec![obj_bytes, bytes],
                    config: vec![("texture_name".to_string(), name)],
                    note,
                }
            }
            None => ObjImport {
                blobs: vec![obj_bytes],
                config: Vec::new(),
                note: if self.notes.is_empty() {
                    "no texture".to_string()
                } else {
                    format!("no texture: {}", self.notes.join("; "))
                },
            },
        }
    }
}

/// Resolves the sidecars of `obj_text`. `obj_stem` is the picked file's
/// name without extension (the fallback texture name scanners use).
///
/// `read(path, limit)` returns the bytes of a file of at most `limit`
/// bytes at `path` relative to the OBJ's folder, or `None` when there is
/// no such file or it is larger. It is only ever asked for a relative
/// path of plain components separated by `/`: never a rooted one, never
/// one with `..`.
///
/// Order: every `mtllib` in the file, first `map_Kd` found (then, failing
/// that, `map_Ka`), the path as written when it stays inside the folder,
/// then its bare file name next to the OBJ; without a material library or
/// a readable map, an image named after the OBJ itself. A file that is
/// not a JPEG or PNG image is passed over and noted.
pub fn resolve(
    obj_text: &str,
    obj_stem: &str,
    read: &mut dyn FnMut(&str, u64) -> Option<Vec<u8>>,
) -> ObjSidecars {
    let mut out = ObjSidecars::default();

    let libraries = mtllib_names(obj_text);
    for library in &libraries {
        let Some(mtl_bytes) = read_either(library, MAX_MTL_BYTES, read) else {
            out.notes
                .push(format!("material library {library} not found"));
            continue;
        };
        let mtl_text = String::from_utf8_lossy(&mtl_bytes);
        let Some(map) = diffuse_map(&mtl_text) else {
            out.notes.push(format!("{library} names no texture"));
            continue;
        };
        match read_either(&map, MAX_TEXTURE_BYTES, read) {
            Some(bytes) if is_image(&bytes) => {
                out.mtl = Some(library.clone());
                out.texture = Some((file_name(&map).to_string(), bytes));
                return out;
            }
            Some(_) => out
                .notes
                .push(format!("{library} names {map}, which is not an image")),
            None => out
                .notes
                .push(format!("{library} names {map}, which was not found")),
        }
    }

    for extension in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"] {
        let candidate = format!("{obj_stem}.{extension}");
        if relative_inside(&candidate).is_none() {
            continue;
        }
        match read(&candidate, MAX_TEXTURE_BYTES) {
            Some(bytes) if is_image(&bytes) => {
                out.notes.push(format!("using {candidate} by name"));
                out.texture = Some((candidate, bytes));
                return out;
            }
            Some(_) => out.notes.push(format!("{candidate} is not an image")),
            None => {}
        }
    }
    if libraries.is_empty() {
        out.notes.push("no mtllib line".to_string());
    }
    out
}

/// Whether `bytes` begin as a JPEG or a PNG does: the two formats the
/// preview and Mesh to Model decode, and the two whose first bytes are
/// distinctive enough that other files do not pass for them.
fn is_image(bytes: &[u8]) -> bool {
    matches!(
        MeshTexture::sniff_media_type(bytes),
        Some("image/jpeg" | "image/png")
    )
}

/// `path` with `/` separators when it is a relative path of plain
/// components, which cannot leave the folder it is joined to: no leading
/// separator, no drive or scheme (`:`), no `.` or `..`, no empty
/// component. `None` for anything else.
fn relative_inside(path: &str) -> Option<String> {
    let normalised = path.replace('\\', "/");
    let plain = |part: &str| !part.is_empty() && part != "." && part != ".." && !part.contains(':');
    normalised.split('/').all(plain).then_some(normalised)
}

/// A path as written when it stays inside the OBJ's folder, then its bare
/// file name (exporters often write a path from the machine that made the
/// file; the sidecar is next to the OBJ anyway).
fn read_either(
    path: &str,
    limit: u64,
    read: &mut dyn FnMut(&str, u64) -> Option<Vec<u8>>,
) -> Option<Vec<u8>> {
    let inside = relative_inside(path);
    if let Some(inside) = &inside
        && let Some(bytes) = read(inside, limit)
    {
        return Some(bytes);
    }
    let name = relative_inside(file_name(path))?;
    if inside.as_deref() == Some(name.as_str()) {
        return None;
    }
    read(&name, limit)
}

/// The native reader: resolves the sidecars of the OBJ at `obj_path`
/// (whose bytes are `obj_bytes`) among the regular files of its folder.
/// A name that resolves outside the folder (through a symbolic link), to
/// something that is not a regular file, or to a file over the limit is
/// treated as missing.
///
/// What this guards against is the content of the files. A program that
/// can write to the folder while the import runs can still swap a file
/// for a link between the check and the read; it gains a regular file
/// under the size limit that begins as a JPEG or PNG does.
#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_beside(obj_path: &std::path::Path, obj_bytes: &[u8]) -> ObjSidecars {
    use std::io::Read;

    let dir = match obj_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => std::path::PathBuf::from("."),
    };
    let stem = obj_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let root = dir.canonicalize().ok();
    let mut read = |relative: &str, limit: u64| -> Option<Vec<u8>> {
        let root = root.as_ref()?;
        let path = root.join(relative).canonicalize().ok()?;
        if !path.starts_with(root) {
            return None;
        }
        // Looked at before it is opened: opening a named pipe waits for
        // a writer that may never come.
        if !std::fs::metadata(&path).ok()?.is_file() {
            return None;
        }
        let file = std::fs::File::open(&path).ok()?;
        let metadata = file.metadata().ok()?;
        if !metadata.is_file() || metadata.len() > limit {
            return None;
        }
        // The length was checked on the open file; the cap on the read
        // itself covers a file that grows meanwhile.
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(limit + 1).read_to_end(&mut bytes).ok()?;
        (bytes.len() as u64 <= limit).then_some(bytes)
    };
    resolve(&String::from_utf8_lossy(obj_bytes), stem, &mut read)
}

/// The final component of a path written with either separator.
pub fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Every `mtllib` operand in the file, in order. A line may name several
/// libraries separated by spaces, but file names with spaces are far more
/// common than multiple libraries, so the whole operand is one name.
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

    /// The first bytes of a JPEG and of a PNG, which is all the resolver
    /// looks at.
    const JPEG: &[u8] = b"\xFF\xD8\xFFjpeg";
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\npng";

    fn disk(files: &[(&str, &[u8])]) -> HashMap<String, Vec<u8>> {
        files
            .iter()
            .map(|(name, bytes)| (name.to_string(), bytes.to_vec()))
            .collect()
    }

    fn resolve_on(obj: &str, stem: &str, files: &HashMap<String, Vec<u8>>) -> ObjSidecars {
        resolve(obj, stem, &mut |path, _| files.get(path).cloned())
    }

    /// The scanner layout: one OBJ, one MTL, one JPEG, all named alike.
    #[test]
    fn follows_mtllib_and_map_kd() {
        let files = disk(&[
            ("scan.mtl", b"newmtl scan\nKd 1 1 1\nmap_Kd scan.jpg\n"),
            ("scan.jpg", JPEG),
        ]);
        let found = resolve_on("mtllib scan.mtl\nv 0 0 0\n", "scan", &files);
        assert_eq!(found.mtl.as_deref(), Some("scan.mtl"));
        assert_eq!(found.texture, Some(("scan.jpg".to_string(), JPEG.to_vec())));
        assert!(found.notes.is_empty());

        let import = found.into_import(b"obj".to_vec());
        assert_eq!(import.blobs, vec![b"obj".to_vec(), JPEG.to_vec()]);
        assert_eq!(
            import.config,
            vec![("texture_name".to_string(), "scan.jpg".to_string())]
        );
        assert_eq!(import.note, "texture scan.jpg via scan.mtl");
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
            ("tex atlas.png", PNG),
        ]);
        let found = resolve_on("mtllib m.mtl\n", "m", &files);
        assert_eq!(
            found.texture,
            Some(("tex atlas.png".to_string(), PNG.to_vec()))
        );
    }

    /// A map in a subfolder of the OBJ's folder is followed as written.
    #[test]
    fn follows_a_path_inside_the_folder() {
        let files = disk(&[
            ("m.mtl", b"map_Kd textures\\atlas.png\n"),
            ("textures/atlas.png", PNG),
        ]);
        let found = resolve_on("mtllib m.mtl\n", "m", &files);
        assert_eq!(found.texture, Some(("atlas.png".to_string(), PNG.to_vec())));
    }

    /// Whatever an OBJ or a material library names, the reader is asked
    /// only for plain relative paths: a rooted path, a drive, and a climb
    /// out of the folder each come down to the bare file name.
    #[test]
    fn never_asks_for_a_path_outside_the_folder() {
        for (library, map) in [
            ("/etc/passwd", "/home/me/.ssh/id_ed25519"),
            ("../../secret.mtl", "../outside.png"),
            ("C:\\Windows\\win.ini", "D:tex.png"),
            ("a/../../b.mtl", "sub/../../c.png"),
            ("..", "."),
            ("./m.mtl", "//server/share/t.png"),
        ] {
            let mut asked: Vec<String> = Vec::new();
            let mtl = format!("map_Kd {map}\n");
            let found = resolve(&format!("mtllib {library}\n"), "..", &mut |path, _| {
                asked.push(path.to_string());
                // Every read succeeds with the library's text, so the map
                // is looked for too (and, being text, is not an image).
                Some(mtl.clone().into_bytes())
            });
            assert_eq!(found.texture, None, "{library} {map}");
            assert!(!asked.is_empty(), "{library} {map}");
            for path in &asked {
                assert!(
                    relative_inside(path).as_deref() == Some(path.as_str()),
                    "asked for {path:?} ({library}, {map})"
                );
            }
        }
        assert_eq!(relative_inside("a/b c.png").as_deref(), Some("a/b c.png"));
        assert_eq!(relative_inside("a\\b.png").as_deref(), Some("a/b.png"));
        for outside in ["/a", "a//b", "../a", "a/..", ".", "", "C:a", "a/./b"] {
            assert_eq!(relative_inside(outside), None, "{outside:?}");
        }
    }

    /// A file that is not an image is not taken as the texture, and the
    /// search goes on to the image named after the OBJ.
    #[test]
    fn passes_over_a_file_that_is_not_an_image() {
        let files = disk(&[
            ("m.mtl", b"map_Kd notes.txt\n"),
            ("notes.txt", b"not an image"),
            ("m.png", PNG),
        ]);
        let found = resolve_on("mtllib m.mtl\n", "m", &files);
        assert_eq!(found.texture, Some(("m.png".to_string(), PNG.to_vec())));
        assert_eq!(
            found.notes,
            vec![
                "m.mtl names notes.txt, which is not an image".to_string(),
                "using m.png by name".to_string()
            ]
        );

        // Two letters are not enough to pass for an image (a BMP begins
        // "BM", and so does plenty of text).
        let files = disk(&[("m.jpg", b"BMW service notes")]);
        let found = resolve_on("v 0 0 0\n", "m", &files);
        assert_eq!(found.texture, None);
        assert_eq!(
            found.notes,
            vec![
                "m.jpg is not an image".to_string(),
                "no mtllib line".to_string()
            ]
        );
    }

    /// Libraries and textures are read under their own size limits, and a
    /// file the reader refuses (over the limit) counts as missing.
    #[test]
    fn reads_carry_the_size_limits() {
        let mut asked: Vec<(String, u64)> = Vec::new();
        let found = resolve("mtllib m.mtl\n", "m", &mut |path, limit| {
            asked.push((path.to_string(), limit));
            (path == "m.mtl").then(|| b"map_Kd huge.png\n".to_vec())
        });
        assert_eq!(found.texture, None);
        assert_eq!(asked[0], ("m.mtl".to_string(), MAX_MTL_BYTES));
        assert_eq!(asked[1], ("huge.png".to_string(), MAX_TEXTURE_BYTES));
        assert!(
            asked[2..]
                .iter()
                .all(|(_, limit)| *limit == MAX_TEXTURE_BYTES)
        );
        assert_eq!(
            found.notes,
            vec!["m.mtl names huge.png, which was not found".to_string()]
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
        let files = disk(&[("scan.jpg", JPEG)]);
        let found = resolve_on("v 0 0 0\n", "scan", &files);
        assert_eq!(
            found.texture.as_ref().map(|(n, _)| n.as_str()),
            Some("scan.jpg")
        );
        assert_eq!(found.mtl, None);
        assert_eq!(found.notes, vec!["using scan.jpg by name".to_string()]);
        assert_eq!(found.into_import(Vec::new()).note, "texture scan.jpg");

        let files = disk(&[("other.mtl", b"newmtl a\nmap_Kd gone.png\n")]);
        let found = resolve_on("mtllib other.mtl\n", "scan", &files);
        assert_eq!(found.texture, None);
        assert_eq!(
            found.notes,
            vec!["other.mtl names gone.png, which was not found".to_string()]
        );
        let import = found.into_import(b"obj".to_vec());
        assert_eq!(import.blobs, vec![b"obj".to_vec()]);
        assert!(import.config.is_empty());
        assert_eq!(
            import.note,
            "no texture: other.mtl names gone.png, which was not found"
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
        assert_eq!(
            found.into_import(Vec::new()).note,
            "no texture: no mtllib line"
        );
    }

    #[test]
    fn mtllib_operand_is_one_name() {
        assert_eq!(
            mtllib_names("mtllib my scan.mtl\nmtllibx\nmtllib my scan.mtl\n"),
            vec!["my scan.mtl".to_string()]
        );
        assert_eq!(file_name("a/b\\c.png"), "c.png");
    }

    /// A folder for one test, removed when it goes out of scope.
    #[cfg(not(target_arch = "wasm32"))]
    struct Folder(std::path::PathBuf);

    #[cfg(not(target_arch = "wasm32"))]
    impl Folder {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "volumetric_obj_sidecar_{}_{name}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(path.join("scan")).unwrap();
            Folder(path)
        }
        fn write(&self, relative: &str, bytes: &[u8]) {
            std::fs::write(self.0.join(relative), bytes).unwrap();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl Drop for Folder {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// On a real disk: the sidecars beside the OBJ are found, and an image
    /// one folder up, which the material library names, is not read.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_native_reader_stays_in_the_objs_folder() {
        let folder = Folder::new("stays");
        let obj = b"mtllib scan.mtl\nv 0 0 0\n";
        folder.write("scan/scan.obj", obj);
        folder.write("scan/scan.mtl", b"map_Kd scan.jpg\n");
        folder.write("scan/scan.jpg", JPEG);
        let found = resolve_beside(&folder.0.join("scan/scan.obj"), obj);
        assert_eq!(found.texture, Some(("scan.jpg".to_string(), JPEG.to_vec())));
        assert_eq!(found.mtl.as_deref(), Some("scan.mtl"));

        // The same library pointing up and out: the image there is real
        // and readable, and is still not taken.
        folder.write("outside.png", PNG);
        let outside = folder.0.join("outside.png");
        for map in ["../outside.png".to_string(), outside.display().to_string()] {
            folder.write("scan/scan.mtl", format!("map_Kd {map}\n").as_bytes());
            std::fs::remove_file(folder.0.join("scan/scan.jpg")).ok();
            let found = resolve_beside(&folder.0.join("scan/scan.obj"), obj);
            assert_eq!(found.texture, None, "{map}");
            assert_eq!(
                found.notes,
                vec![format!("scan.mtl names {map}, which was not found")]
            );
        }

        // A folder with the texture's name is not a file to read.
        folder.write("scan/scan.mtl", b"map_Kd tex.png\n");
        std::fs::create_dir(folder.0.join("scan/tex.png")).unwrap();
        let found = resolve_beside(&folder.0.join("scan/scan.obj"), obj);
        assert_eq!(found.texture, None);
    }

    /// A symbolic link in the OBJ's folder that points out of it is not
    /// followed, whether it stands for the library or for the image.
    #[cfg(unix)]
    #[test]
    fn the_native_reader_does_not_follow_a_link_out_of_the_folder() {
        let folder = Folder::new("link");
        let obj = b"mtllib scan.mtl\nv 0 0 0\n";
        folder.write("scan/scan.obj", obj);
        folder.write("scan/scan.mtl", b"map_Kd scan.png\n");
        folder.write("outside.png", PNG);
        std::os::unix::fs::symlink(folder.0.join("outside.png"), folder.0.join("scan/scan.png"))
            .unwrap();
        let found = resolve_beside(&folder.0.join("scan/scan.obj"), obj);
        assert_eq!(found.texture, None);

        // The same link to a file inside the folder is an ordinary file.
        folder.write("scan/real.png", PNG);
        std::fs::remove_file(folder.0.join("scan/scan.png")).unwrap();
        std::os::unix::fs::symlink(
            folder.0.join("scan/real.png"),
            folder.0.join("scan/scan.png"),
        )
        .unwrap();
        let found = resolve_beside(&folder.0.join("scan/scan.obj"), obj);
        assert_eq!(found.texture, Some(("scan.png".to_string(), PNG.to_vec())));
    }

    /// A named pipe standing where the material library or the texture
    /// should be is passed over; opening one would wait for a writer.
    #[cfg(unix)]
    #[test]
    fn the_native_reader_does_not_wait_on_a_named_pipe() {
        let folder = Folder::new("pipe");
        let obj = b"mtllib scan.mtl\nv 0 0 0\n";
        folder.write("scan/scan.obj", obj);
        for pipe in ["scan/scan.mtl", "scan/scan.jpg"] {
            let made = std::process::Command::new("mkfifo")
                .arg(folder.0.join(pipe))
                .status();
            if !made.is_ok_and(|status| status.success()) {
                eprintln!("skipped: no mkfifo");
                return;
            }
        }
        let obj_path = folder.0.join("scan/scan.obj");
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = done.send(resolve_beside(&obj_path, obj));
        });
        let found = finished
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the reader returned instead of waiting on the pipe");
        assert_eq!(found.texture, None);
        assert_eq!(
            found.notes,
            vec!["material library scan.mtl not found".to_string()]
        );
    }
}
