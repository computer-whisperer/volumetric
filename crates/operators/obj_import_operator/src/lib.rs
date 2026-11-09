//! OBJ Import Operator.
//!
//! Reads a Wavefront OBJ file and emits a [`TriMesh`] value — an explicit
//! triangle mesh, *not* a sampleable model; converting a (watertight) mesh
//! into an implicit solid is `mesh_to_model_operator`'s job. See README.md
//! (the operator's docs) for what the reader honours.
//!
//! The OBJ convention is that the user picks the `.obj` and the program
//! finds the sidecars (`mtllib` → `map_Kd` → image). That resolution is
//! the host's: this operator has no filesystem, so the host wires the
//! image bytes into input 1 and the operator attaches them to the mesh as
//! its base-colour texture, still file-encoded.
//!
//! Inputs:
//! - Input 0: Blob — OBJ file bytes
//! - Input 1: Blob — the base-colour texture image (JPEG/PNG/...), or empty
//! - Input 2: CBOR configuration: `scale` (float, default 1.0), `center`
//!   (bool, default false), `texture_name` (text, default ""). Placement
//!   is center → scale.
//!
//! Output 0: TriMesh.

use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::trimesh::encode_tri_mesh;
use volumetric_abi::trimesh::{
    COLOR_FIELD_NAME, MeshField, MeshTexture, NORMAL_FIELD_NAME, TriMesh, UV_FIELD_NAME,
};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct ObjImportConfig {
    pub scale: f64,
    pub center: bool,
    pub texture_name: String,
}

impl Default for ObjImportConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            center: false,
            texture_name: String::new(),
        }
    }
}

/// What the reader had to leave out or guess, for the host's warning line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportReport {
    /// Faces with fewer than three corners, dropped.
    pub dropped_faces: usize,
    /// Corners without a `vt` in a file that has texture coordinates;
    /// they read `(0, 0)`.
    pub corners_without_uv: usize,
    /// Corners without a `vn` in a file that has normals; the `normal`
    /// field is then omitted entirely.
    pub corners_without_normal: usize,
    /// Distinct `usemtl` names seen; more than one means the file has
    /// materials this import cannot tell apart.
    pub materials: Vec<String>,
    /// A texture was supplied but the file has no texture coordinates, so
    /// it was dropped.
    pub texture_without_uv: bool,
    /// A texture was attached whose media type could not be sniffed.
    pub texture_type_unknown: bool,
}

impl ImportReport {
    /// One human-readable warning per thing worth saying; empty when the
    /// import was clean.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.dropped_faces > 0 {
            out.push(format!(
                "dropped {} face(s) with fewer than three corners",
                self.dropped_faces
            ));
        }
        if self.corners_without_uv > 0 {
            out.push(format!(
                "{} corner(s) have no texture coordinate and read (0, 0)",
                self.corners_without_uv
            ));
        }
        if self.corners_without_normal > 0 {
            out.push(format!(
                "{} corner(s) have no normal, so the normal field is omitted",
                self.corners_without_normal
            ));
        }
        if self.materials.len() > 1 {
            out.push(format!(
                "the file uses {} materials ({}); only one texture is carried",
                self.materials.len(),
                self.materials.join(", ")
            ));
        }
        if self.texture_without_uv {
            out.push(
                "a texture was supplied but the file has no texture coordinates; it was dropped"
                    .to_string(),
            );
        }
        if self.texture_type_unknown {
            out.push(
                "the texture's image format was not recognised (expected JPEG or PNG)".to_string(),
            );
        }
        out
    }
}

/// One `f` corner as the file indexes it: resolved, 0-based indices into
/// the `v`, `vt` and `vn` tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Corner {
    v: u32,
    vt: Option<u32>,
    vn: Option<u32>,
}

/// The tables and faces of an OBJ file, as parsed.
#[derive(Debug, Default)]
struct ObjFile {
    positions: Vec<[f64; 3]>,
    /// Per-`v` colour when the line carried one (MeshLab style).
    colors: Vec<Option<[f64; 3]>>,
    texcoords: Vec<[f64; 2]>,
    normals: Vec<[f64; 3]>,
    /// Each face's corners, already fan-triangulated.
    triangles: Vec<[Corner; 3]>,
    dropped_faces: usize,
    materials: Vec<String>,
}

/// Resolves an OBJ index (1-based, negative = relative to the table's
/// current length) against a table of `len` entries.
fn resolve_index(raw: i64, len: usize, what: &str, line_no: usize) -> Result<u32, String> {
    let idx = if raw > 0 {
        raw - 1
    } else if raw < 0 {
        len as i64 + raw
    } else {
        return Err(format!(
            "line {line_no}: {what} index 0 is not valid in OBJ"
        ));
    };
    if idx < 0 || idx as usize >= len {
        return Err(format!(
            "line {line_no}: {what} index {raw} is out of range (table has {len} entries)"
        ));
    }
    Ok(idx as u32)
}

fn parse_floats<const N: usize>(
    parts: &[&str],
    what: &str,
    line_no: usize,
) -> Result<[f64; N], String> {
    if parts.len() < N {
        return Err(format!(
            "line {line_no}: {what} needs {N} values, found {}",
            parts.len()
        ));
    }
    let mut out = [0.0; N];
    for (slot, text) in out.iter_mut().zip(parts) {
        *slot = text
            .parse::<f64>()
            .map_err(|_| format!("line {line_no}: {what} value {text:?} is not a number"))?;
        if !slot.is_finite() {
            return Err(format!(
                "line {line_no}: {what} value {text:?} is not finite"
            ));
        }
    }
    Ok(out)
}

/// Joins `\`-continued lines, strips comments, and yields `(line_no,
/// text)` for every non-blank logical line. `line_no` is the physical
/// line the logical line started on.
fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut pending: Option<(usize, String)> = None;
    for (i, raw) in text.lines().enumerate() {
        let line_no = i + 1;
        let (mut body, continued) = match raw.trim_end().strip_suffix('\\') {
            Some(head) => (head.to_string(), true),
            None => (raw.to_string(), false),
        };
        if let Some(hash) = body.find('#') {
            body.truncate(hash);
        }
        match pending.take() {
            Some((start, mut acc)) => {
                acc.push(' ');
                acc.push_str(&body);
                if continued {
                    pending = Some((start, acc));
                } else if !acc.trim().is_empty() {
                    out.push((start, acc));
                }
            }
            None => {
                if continued {
                    pending = Some((line_no, body));
                } else if !body.trim().is_empty() {
                    out.push((line_no, body));
                }
            }
        }
    }
    if let Some((start, acc)) = pending
        && !acc.trim().is_empty()
    {
        out.push((start, acc));
    }
    out
}

fn parse_obj(text: &str) -> Result<ObjFile, String> {
    let mut file = ObjFile::default();
    for (line_no, line) in logical_lines(text) {
        let mut parts = line.split_whitespace();
        let Some(keyword) = parts.next() else {
            continue;
        };
        let rest: Vec<&str> = parts.collect();
        match keyword {
            "v" => {
                let p: [f64; 3] = parse_floats(&rest, "vertex", line_no)?;
                file.positions.push(p);
                // `v x y z w` is a homogeneous coordinate (rare); `v x y z
                // r g b` is the MeshLab colour extension. Six or more
                // values read as position + colour.
                let color = if rest.len() >= 6 {
                    Some(parse_floats::<3>(&rest[3..6], "vertex colour", line_no)?)
                } else {
                    None
                };
                file.colors.push(color);
            }
            "vt" => {
                // `vt u` alone is legal (v defaults to 0); `vt u v w` too.
                let u: [f64; 1] = parse_floats(&rest, "texture coordinate", line_no)?;
                let v = if rest.len() >= 2 {
                    parse_floats::<1>(&rest[1..2], "texture coordinate", line_no)?[0]
                } else {
                    0.0
                };
                file.texcoords.push([u[0], v]);
            }
            "vn" => {
                let n: [f64; 3] = parse_floats(&rest, "normal", line_no)?;
                file.normals.push(n);
            }
            "f" => {
                let mut corners: Vec<Corner> = Vec::with_capacity(rest.len());
                for token in &rest {
                    let mut fields = token.split('/');
                    let v_text = fields.next().unwrap_or("");
                    let vt_text = fields.next().unwrap_or("");
                    let vn_text = fields.next().unwrap_or("");
                    let parse = |text: &str, what: &str| -> Result<i64, String> {
                        text.parse::<i64>().map_err(|_| {
                            format!("line {line_no}: {what} index {text:?} is not an integer")
                        })
                    };
                    let v = resolve_index(
                        parse(v_text, "vertex")?,
                        file.positions.len(),
                        "vertex",
                        line_no,
                    )?;
                    let vt = if vt_text.is_empty() {
                        None
                    } else {
                        Some(resolve_index(
                            parse(vt_text, "texture coordinate")?,
                            file.texcoords.len(),
                            "texture coordinate",
                            line_no,
                        )?)
                    };
                    let vn = if vn_text.is_empty() {
                        None
                    } else {
                        Some(resolve_index(
                            parse(vn_text, "normal")?,
                            file.normals.len(),
                            "normal",
                            line_no,
                        )?)
                    };
                    corners.push(Corner { v, vt, vn });
                }
                if corners.len() < 3 {
                    file.dropped_faces += 1;
                    continue;
                }
                for i in 1..corners.len() - 1 {
                    file.triangles
                        .push([corners[0], corners[i], corners[i + 1]]);
                }
            }
            "usemtl" => {
                let name = rest.join(" ");
                if !file.materials.contains(&name) {
                    file.materials.push(name);
                }
            }
            // Grouping, smoothing, material library references,
            // free-form geometry and anything else: no effect on the
            // triangles.
            _ => {}
        }
    }
    Ok(file)
}

/// Reads the file as a mesh, attaching `texture` (the base-colour image's
/// bytes, still encoded) when the file has texture coordinates to address
/// it with. Returns the mesh with what had to be left out or guessed.
pub fn import(
    obj_bytes: &[u8],
    texture: Option<&[u8]>,
    config: &ObjImportConfig,
) -> Result<(TriMesh, ImportReport), String> {
    let text = String::from_utf8_lossy(obj_bytes);
    let file = parse_obj(&text)?;
    if file.triangles.is_empty() {
        return Err(if file.positions.is_empty() {
            "the file has no vertices".to_string()
        } else {
            "the file has no faces, so it is a point cloud; OBJ import needs a mesh".to_string()
        });
    }

    let mut report = ImportReport {
        dropped_faces: file.dropped_faces,
        materials: file.materials.clone(),
        ..Default::default()
    };

    // A corner is a vertex: every distinct (v, vt, vn) tuple becomes one
    // output vertex, so a position shared by corners with different
    // texture coordinates is split along the seam. Corners that are
    // identical tuples (the common case: a seamless vertex) weld back
    // into one.
    let has_uv = file.triangles.iter().flatten().any(|c| c.vt.is_some());
    let all_normals = file.triangles.iter().flatten().all(|c| c.vn.is_some());
    let any_normals = file.triangles.iter().flatten().any(|c| c.vn.is_some());
    let all_colors = !file.colors.is_empty() && file.colors.iter().all(Option::is_some);

    let mut vertex_of: HashMap<Corner, u32> = HashMap::new();
    let mut positions: Vec<f64> = Vec::new();
    let mut uvs: Vec<f64> = Vec::new();
    let mut normals: Vec<f64> = Vec::new();
    let mut colors: Vec<f64> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(file.triangles.len() * 3);
    for tri in &file.triangles {
        for corner in tri {
            // Normals only matter as an identity when they are kept.
            let key = Corner {
                v: corner.v,
                vt: corner.vt,
                vn: if all_normals { corner.vn } else { None },
            };
            let next = positions.len() / 3;
            let id = *vertex_of.entry(key).or_insert_with(|| {
                positions.extend(file.positions[corner.v as usize]);
                if has_uv {
                    match corner.vt {
                        Some(vt) => uvs.extend(file.texcoords[vt as usize]),
                        None => uvs.extend([0.0, 0.0]),
                    }
                }
                if all_normals {
                    normals.extend(
                        file.normals[corner.vn.expect("all corners have normals") as usize],
                    );
                }
                if all_colors {
                    colors.extend(file.colors[corner.v as usize].expect("all vertices coloured"));
                }
                next as u32
            });
            indices.push(id);
            if has_uv && corner.vt.is_none() {
                report.corners_without_uv += 1;
            }
            if any_normals && corner.vn.is_none() {
                report.corners_without_normal += 1;
            }
        }
    }

    ply_core::convert::place(&mut positions, config.center, config.scale)?;

    let mut vertex_fields = Vec::new();
    let vertex_count = positions.len() / 3;
    if has_uv {
        vertex_fields.push(MeshField {
            name: UV_FIELD_NAME.to_string(),
            components: 2,
            data: uvs,
        });
    }
    if all_normals {
        vertex_fields.push(MeshField {
            name: NORMAL_FIELD_NAME.to_string(),
            components: 3,
            data: normals,
        });
    }
    if all_colors {
        vertex_fields.push(MeshField {
            name: COLOR_FIELD_NAME.to_string(),
            components: 3,
            data: colors,
        });
    }
    debug_assert!(
        vertex_fields
            .iter()
            .all(|f| f.data.len() == vertex_count * f.components)
    );

    let mut textures = Vec::new();
    if let Some(bytes) = texture.filter(|b| !b.is_empty()) {
        if has_uv {
            let media_type = MeshTexture::sniff_media_type(bytes);
            report.texture_type_unknown = media_type.is_none();
            let name = if config.texture_name.trim().is_empty() {
                "texture".to_string()
            } else {
                config.texture_name.trim().to_string()
            };
            textures.push(MeshTexture {
                name,
                media_type: media_type.unwrap_or("").to_string(),
                bytes: bytes.to_vec(),
            });
        } else {
            report.texture_without_uv = true;
        }
    }

    let mesh = TriMesh {
        positions,
        indices,
        vertex_fields,
        face_fields: vec![],
        textures,
    };
    mesh.validate()?;
    Ok((mesh, report))
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let bytes = read_input(0);
    let texture = read_input(1);
    let config = {
        let cfg = read_input(2);
        if cfg.is_empty() {
            ObjImportConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&cfg)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    match import(&bytes, Some(&texture), &config) {
        Ok((mesh, report)) => {
            for warning in report.warnings() {
                post_warning(&warning);
            }
            post_output(0, &encode_tri_mesh(&mesh));
        }
        Err(e) => report_error(&format!("OBJ import failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            "{ scale: float .default 1.0, center: bool .default false, texture_name: tstr .default \"\" }"
                .to_string();
        OperatorMetadata {
            name: "obj_import_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "OBJ Import".to_string(),
            description:
                "Read a Wavefront OBJ mesh with its texture atlas, as a scanner exports it."
                    .to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>"##,
                r##"<path d="M14 2v4a2 2 0 0 0 2 2h4"/>"##,
                r##"<path d="M8 13h8v6H8z"/>"##,
                r##"<path d="m8 17 2.5-2.5L13 17l1.5-1.5L16 17"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec![
                "OBJ file".to_string(),
                "Texture".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::TriMesh],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured unit quad: four positions, four texture coordinates, two
    /// triangles written as one quad face, plus a comment, a continuation
    /// line and a material reference.
    const QUAD: &str = "# a quad\n\
        mtllib quad.mtl\n\
        o quad\n\
        v 0 0 0\n\
        v 1 0 0\n\
        v 1 1 0\n\
        v 0 \\\n  1 0\n\
        vt 0 0\n\
        vt 1 0\n\
        vt 1 1\n\
        vt 0 1\n\
        vn 0 0 1\n\
        usemtl scan\n\
        g scan\n\
        f 1/1/1 2/2/1 3/3/1 4/4/1\n";

    const JPEG_HEAD: &[u8] = &[0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10, b'J', b'F', b'I', b'F'];

    #[test]
    fn quad_imports_with_uvs_normals_and_texture() {
        let config = ObjImportConfig {
            texture_name: "atlas.jpg".to_string(),
            ..Default::default()
        };
        let (mesh, report) = import(QUAD.as_bytes(), Some(JPEG_HEAD), &config).unwrap();
        assert_eq!(mesh.vertex_count(), 4, "seamless corners weld");
        assert_eq!(mesh.triangle_count(), 2, "the quad fans into two triangles");
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(
            mesh.position(3),
            [0.0, 1.0, 0.0],
            "continuation line joined"
        );
        let uv = mesh.uv_field().expect("uv field");
        assert_eq!(uv.data, vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let normal = mesh
            .vertex_fields
            .iter()
            .find(|f| f.name == NORMAL_FIELD_NAME)
            .expect("normal field");
        assert_eq!(normal.components, 3);
        assert_eq!(&normal.data[..3], &[0.0, 0.0, 1.0]);
        let texture = mesh.base_color_texture().expect("texture attached");
        assert_eq!(texture.name, "atlas.jpg");
        assert_eq!(texture.media_type, "image/jpeg");
        assert_eq!(texture.bytes, JPEG_HEAD);
        assert_eq!(
            report,
            ImportReport {
                materials: vec!["scan".to_string()],
                ..Default::default()
            }
        );
        assert!(report.warnings().is_empty());
    }

    #[test]
    fn seams_split_vertices_by_texture_coordinate() {
        // Two triangles sharing the edge 1-2 but addressing it with
        // different texture coordinates: the shared positions split.
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
                   vt 0 0\nvt 1 0\nvt 1 1\nvt 0.5 0.5\nvt 0.5 0\nvt 0 1\n\
                   f 1/1 2/2 3/3\nf 1/4 3/5 4/6\n";
        let (mesh, _) = import(obj.as_bytes(), None, &ObjImportConfig::default()).unwrap();
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.triangle_count(), 2);
        assert!(mesh.textures.is_empty());
    }

    #[test]
    fn negative_indices_and_bare_corners() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n";
        let (mesh, report) = import(obj.as_bytes(), None, &ObjImportConfig::default()).unwrap();
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert!(mesh.uv_field().is_none());
        assert!(report.warnings().is_empty());
    }

    #[test]
    fn placement_centres_then_scales() {
        let obj = "v 0 0 0\nv 10 0 0\nv 10 10 0\nf 1 2 3\n";
        let config = ObjImportConfig {
            scale: 0.1,
            center: true,
            ..Default::default()
        };
        let (mesh, _) = import(obj.as_bytes(), None, &config).unwrap();
        assert_eq!(mesh.bounds(), Some([-0.5, 0.5, -0.5, 0.5, 0.0, 0.0]));
    }

    #[test]
    fn partial_uvs_and_normals_are_reported() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\n\
                   f 1/1/1 2/2/1 3/3/1\nf 2 4 3\n";
        let (mesh, report) =
            import(obj.as_bytes(), Some(JPEG_HEAD), &ObjImportConfig::default()).unwrap();
        assert_eq!(report.corners_without_uv, 3);
        assert_eq!(report.corners_without_normal, 3);
        assert!(mesh.uv_field().is_some());
        assert!(
            mesh.vertex_fields
                .iter()
                .all(|f| f.name != NORMAL_FIELD_NAME)
        );
        // The uv-less corners read (0, 0) and are distinct vertices from
        // the textured ones at the same positions.
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(report.warnings().len(), 2);
    }

    #[test]
    fn texture_without_texture_coordinates_is_dropped_with_a_warning() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let (mesh, report) =
            import(obj.as_bytes(), Some(JPEG_HEAD), &ObjImportConfig::default()).unwrap();
        assert!(mesh.textures.is_empty());
        assert!(report.texture_without_uv);
        assert_eq!(report.warnings().len(), 1);
    }

    #[test]
    fn unknown_texture_bytes_attach_with_a_warning() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";
        let (mesh, report) = import(
            obj.as_bytes(),
            Some(b"not an image"),
            &ObjImportConfig::default(),
        )
        .unwrap();
        assert_eq!(mesh.textures.len(), 1);
        assert_eq!(mesh.textures[0].media_type, "");
        assert_eq!(mesh.textures[0].name, "texture");
        assert!(report.texture_type_unknown);
    }

    #[test]
    fn vertex_colours_become_the_color_field() {
        let obj = "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3\n";
        let (mesh, _) = import(obj.as_bytes(), None, &ObjImportConfig::default()).unwrap();
        let color = mesh
            .vertex_fields
            .iter()
            .find(|f| f.name == COLOR_FIELD_NAME)
            .expect("color field");
        assert_eq!(
            color.data,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn polygons_fan_and_short_faces_drop() {
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 2 0\nf 1 2 3 4 5\nf 1 2\n";
        let (mesh, report) = import(obj.as_bytes(), None, &ObjImportConfig::default()).unwrap();
        assert_eq!(mesh.triangle_count(), 3);
        assert_eq!(report.dropped_faces, 1);
    }

    #[test]
    fn multiple_materials_warn() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl a\nf 1 2 3\nusemtl b\nf 3 2 1\n";
        let (_, report) = import(obj.as_bytes(), None, &ObjImportConfig::default()).unwrap();
        assert_eq!(report.materials, vec!["a".to_string(), "b".to_string()]);
        assert!(report.warnings()[0].contains("2 materials"));
    }

    #[test]
    fn bad_files_are_rejected_with_line_numbers() {
        let err = import(
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 7\n",
            None,
            &ObjImportConfig::default(),
        )
        .unwrap_err();
        assert!(err.contains("line 4"), "{err}");
        let err = import(b"v 0 0 x\n", None, &ObjImportConfig::default()).unwrap_err();
        assert!(err.contains("line 1"), "{err}");
        let err = import(
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\n",
            None,
            &ObjImportConfig::default(),
        )
        .unwrap_err();
        assert!(err.contains("no faces"), "{err}");
        let err = import(b"", None, &ObjImportConfig::default()).unwrap_err();
        assert!(err.contains("no vertices"), "{err}");
        let err = import(
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n",
            None,
            &ObjImportConfig::default(),
        )
        .unwrap_err();
        assert!(err.contains("index 0"), "{err}");
    }

    #[test]
    fn logical_lines_join_continuations_and_strip_comments() {
        let lines = logical_lines("a \\\n b # c\n\n# only comment\nd\\\n");
        assert_eq!(lines, vec![(1, "a   b ".to_string()), (5, "d".to_string())]);
    }
}
