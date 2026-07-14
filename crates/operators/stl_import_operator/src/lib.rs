//! STL Import Operator.
//!
//! Reads STL data (binary or ASCII) and emits a [`TriMesh`] value — an
//! explicit triangle mesh, *not* a sampleable model. Non-manifold and open
//! meshes import as-is; converting a (watertight) mesh into an implicit
//! solid is `mesh_to_model_operator`'s job.
//!
//! Vertices are welded by exact coordinate equality: STL stores each
//! triangle's corners independently as f32, so corners that coincide in
//! the file coincide bit-for-bit and share one vertex in the mesh.
//!
//! Inputs:
//! - Input 0: Blob — STL file bytes
//! - Input 1: CBOR configuration: `scale` (float, default 1.0),
//!   `translate` (3 floats, default 0), `center` (bool, default false).
//!   Applied as center → scale → translate.
//!
//! Output 0: TriMesh.

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::trimesh::{TriMesh, encode_tri_mesh};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
struct StlImportConfig {
    #[serde(default = "default_scale")]
    scale: f64,
    #[serde(default)]
    translate: [f64; 3],
    #[serde(default)]
    center: bool,
}

fn default_scale() -> f64 {
    1.0
}

impl Default for StlImportConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            translate: [0.0, 0.0, 0.0],
            center: false,
        }
    }
}

// ============================================================================
// STL parsing (triangle soup: 9 f64s per triangle)
// ============================================================================

fn is_ascii_stl(data: &[u8]) -> bool {
    if data.len() < 6 {
        return false;
    }
    // ASCII STL starts with "solid " (with space), but a binary STL's
    // 80-byte header can too — require a "facet" keyword near the start.
    if !data.starts_with(b"solid ") {
        return false;
    }
    let check_len = data.len().min(1024);
    data[..check_len].windows(5).any(|w| w == b"facet")
}

fn parse_binary_stl(data: &[u8]) -> Result<Vec<[f64; 9]>, String> {
    if data.len() < 84 {
        return Err("binary STL too short".to_string());
    }
    let triangle_count = u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize;
    let expected = 84 + triangle_count * 50;
    if data.len() < expected {
        return Err(format!(
            "binary STL truncated: {} bytes, expected {expected} for {triangle_count} triangles",
            data.len()
        ));
    }

    let mut triangles = Vec::with_capacity(triangle_count);
    for t in 0..triangle_count {
        let base = 84 + t * 50 + 12; // skip the normal
        let mut tri = [0.0f64; 9];
        for (i, value) in tri.iter_mut().enumerate() {
            let off = base + i * 4;
            *value = f32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as f64;
        }
        triangles.push(tri);
    }
    Ok(triangles)
}

fn parse_ascii_stl(data: &[u8]) -> Result<Vec<[f64; 9]>, String> {
    let text = core::str::from_utf8(data).map_err(|_| "invalid UTF-8 in ASCII STL".to_string())?;
    let mut triangles = Vec::new();
    let mut corner = 0usize;
    let mut current = [0.0f64; 9];

    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("vertex ") {
            let mut parts = rest.split_whitespace();
            for i in 0..3 {
                current[corner * 3 + i] = parts
                    .next()
                    .and_then(|p| p.parse::<f64>().ok())
                    .ok_or_else(|| format!("malformed vertex line: {line:?}"))?;
            }
            corner += 1;
            if corner == 3 {
                triangles.push(current);
                corner = 0;
            }
        }
    }
    if corner != 0 {
        return Err("ASCII STL ends mid-facet".to_string());
    }
    Ok(triangles)
}

fn parse_stl(data: &[u8]) -> Result<Vec<[f64; 9]>, String> {
    if is_ascii_stl(data) {
        parse_ascii_stl(data)
    } else {
        parse_binary_stl(data)
    }
}

// ============================================================================
// Soup -> TriMesh
// ============================================================================

fn build_mesh(mut soup: Vec<[f64; 9]>, config: &StlImportConfig) -> Result<TriMesh, String> {
    if !(config.scale.is_finite() && config.scale != 0.0) {
        return Err(format!(
            "scale must be finite and nonzero, got {}",
            config.scale
        ));
    }

    // center -> scale -> translate
    let center_offset = if config.center && !soup.is_empty() {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for tri in &soup {
            for corner in 0..3 {
                for axis in 0..3 {
                    let v = tri[corner * 3 + axis];
                    min[axis] = min[axis].min(v);
                    max[axis] = max[axis].max(v);
                }
            }
        }
        [
            -(min[0] + max[0]) / 2.0,
            -(min[1] + max[1]) / 2.0,
            -(min[2] + max[2]) / 2.0,
        ]
    } else {
        [0.0; 3]
    };
    for tri in soup.iter_mut() {
        for corner in 0..3 {
            for axis in 0..3 {
                tri[corner * 3 + axis] = (tri[corner * 3 + axis] + center_offset[axis])
                    * config.scale
                    + config.translate[axis];
            }
        }
    }

    // Weld exactly-equal corners into shared vertices.
    let mut vertex_ids: std::collections::HashMap<[u64; 3], u32> = std::collections::HashMap::new();
    let mut positions: Vec<f64> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(soup.len() * 3);
    for tri in &soup {
        for corner in 0..3 {
            let p = [tri[corner * 3], tri[corner * 3 + 1], tri[corner * 3 + 2]];
            let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
            let id = *vertex_ids.entry(key).or_insert_with(|| {
                positions.extend(p);
                (positions.len() / 3 - 1) as u32
            });
            indices.push(id);
        }
    }

    let mesh = TriMesh {
        positions,
        indices,
        vertex_fields: vec![],
        face_fields: vec![],
    };
    mesh.validate()?;
    Ok(mesh)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let stl_buf = read_input(0);
    let config = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            StlImportConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&cfg_buf)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    match parse_stl(&stl_buf).and_then(|soup| build_mesh(soup, &config)) {
        Ok(mesh) => post_output(0, &encode_tri_mesh(&mesh)),
        Err(e) => report_error(&format!("STL import failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            "{ scale: float .default 1.0, translate: [float, float, float] .default [0,0,0], center: bool .default false }"
                .to_string();
        OperatorMetadata {
            name: "stl_import_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "STL Import".to_string(),
            description: "Read an STL file (binary or ASCII) as an explicit triangle mesh."
                .to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>"##,
                r##"<path d="M14 2v4a2 2 0 0 0 2 2h4"/>"##,
                r##"<path d="m12 11-3 5h6Z"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["STL file".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::TriMesh],
        }
    })
}
