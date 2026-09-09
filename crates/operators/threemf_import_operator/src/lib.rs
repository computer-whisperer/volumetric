//! 3MF Import Operator.
//!
//! Reads a 3MF package and emits a [`TriMesh`] value in metres — an
//! explicit triangle mesh, *not* a sampleable model; converting a
//! (watertight) mesh into an implicit solid is `mesh_to_model_operator`'s
//! job. Bit-identical corners are welded ([`TriMesh::from_soup`]), as in
//! STL Import. See README.md (the operator's docs) for what the reader
//! honours.
//!
//! Inputs:
//! - Input 0: Blob — 3MF package bytes
//! - Input 1: CBOR configuration: `scale` (float, default 1.0), `center`
//!   (bool, default false), `item` (int, default 0 = all build items
//!   merged; n = the nth item alone, 1-based). The file's unit converts
//!   to metres first, then center → scale.
//!
//! Output 0: TriMesh.

use threemf_core::read_3mf;
#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::trimesh::TriMesh;
#[cfg(target_arch = "wasm32")]
use volumetric_abi::trimesh::encode_tri_mesh;
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ThreeMfImportConfig {
    #[serde(default = "default_scale")]
    pub scale: f64,
    #[serde(default)]
    pub center: bool,
    #[serde(default)]
    pub item: u32,
}

fn default_scale() -> f64 {
    1.0
}

impl Default for ThreeMfImportConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            center: false,
            item: 0,
        }
    }
}

/// Reads the package, picks the configured build item(s), and lands the
/// result in metres with the config transform applied.
pub fn import(bytes: &[u8], config: &ThreeMfImportConfig) -> Result<TriMesh, String> {
    if !(config.scale.is_finite() && config.scale != 0.0) {
        return Err(format!(
            "scale must be finite and nonzero, got {}",
            config.scale
        ));
    }
    let file = read_3mf(bytes)?;
    if file.items.is_empty() {
        return Err("the package places no objects (empty <build>)".to_string());
    }
    let selected: Vec<&TriMesh> = if config.item == 0 {
        file.items.iter().collect()
    } else {
        let index = config.item as usize - 1;
        match file.items.get(index) {
            Some(item) => vec![item],
            None => {
                return Err(format!(
                    "item {} requested but the package has {} build item(s)",
                    config.item,
                    file.items.len()
                ));
            }
        }
    };

    // Merge the items and weld bit-identical corners (as STL Import does):
    // CAD exporters routinely write seam vertices twice, which leaves a
    // geometrically closed surface topologically open.
    let mut mesh = TriMesh::from_soup(selected.iter().flat_map(|item| {
        (0..item.triangle_count()).map(|t| item.triangle(t).map(|v| item.position(v as usize)))
    }));

    // unit → metres, then center → scale.
    let unit = file.unit.metres();
    let center_offset = match (config.center, mesh.bounds()) {
        (true, Some(b)) => [
            -(b[0] + b[1]) / 2.0 * unit,
            -(b[2] + b[3]) / 2.0 * unit,
            -(b[4] + b[5]) / 2.0 * unit,
        ],
        _ => [0.0; 3],
    };
    for (i, value) in mesh.positions.iter_mut().enumerate() {
        *value = (*value * unit + center_offset[i % 3]) * config.scale;
    }
    mesh.validate()?;
    Ok(mesh)
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let package = read_input(0);
    let config = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            ThreeMfImportConfig::default()
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

    match import(&package, &config) {
        Ok(mesh) => post_output(0, &encode_tri_mesh(&mesh)),
        Err(e) => report_error(&format!("3MF import failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            "{ scale: float .default 1.0, center: bool .default false, item: int .ge 0 .default 0 }"
                .to_string();
        OperatorMetadata {
            name: "threemf_import_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "3MF Import".to_string(),
            description: "Read a 3MF package as an explicit triangle mesh, in metres.".to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>"##,
                r##"<path d="M14 2v4a2 2 0 0 0 2 2h4"/>"##,
                r##"<path d="m12 10-3.5 2v4l3.5 2 3.5-2v-4Z"/>"##,
                r##"<path d="M8.5 12 12 14l3.5-2M12 14v4"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["3MF file".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::TriMesh],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use threemf_core::{Unit, write_3mf};

    /// A unit triangle at the origin, written in `unit`.
    fn package(unit: Unit) -> Vec<u8> {
        let mesh = TriMesh {
            positions: vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            indices: vec![0, 1, 2],
            vertex_fields: vec![],
            face_fields: vec![],
        };
        write_3mf(&mesh, unit, "tri").unwrap()
    }

    #[test]
    fn honours_the_unit_then_centers_then_scales() {
        let config = ThreeMfImportConfig::default();
        let mesh = import(&package(Unit::Millimeter), &config).unwrap();
        assert_eq!(mesh.bounds(), Some([0.0, 0.01, 0.0, 0.01, 0.0, 0.0]));
        let mesh = import(&package(Unit::Inch), &config).unwrap();
        assert!((mesh.bounds().unwrap()[1] - 0.254).abs() < 1e-12);

        let config = ThreeMfImportConfig {
            scale: 2.0,
            center: true,
            item: 0,
        };
        let mesh = import(&package(Unit::Meter), &config).unwrap();
        assert_eq!(mesh.bounds(), Some([-10.0, 10.0, -10.0, 10.0, 0.0, 0.0]));
    }

    #[test]
    fn item_selection_and_errors() {
        let bytes = package(Unit::Millimeter);
        let one = ThreeMfImportConfig {
            item: 1,
            ..Default::default()
        };
        assert_eq!(import(&bytes, &one).unwrap().triangle_count(), 1);
        let two = ThreeMfImportConfig {
            item: 2,
            ..Default::default()
        };
        assert!(import(&bytes, &two).unwrap_err().contains("1 build item"));
        let zero_scale = ThreeMfImportConfig {
            scale: 0.0,
            ..Default::default()
        };
        assert!(import(&bytes, &zero_scale).is_err());
        assert!(import(b"not a package", &ThreeMfImportConfig::default()).is_err());
    }
}
