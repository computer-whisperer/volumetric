//! Mesh Merge Operator.
//!
//! Concatenates up to four same-kind meshes ([`FeaMesh`]: point clouds,
//! strut networks, volume meshes) into one, in slot order — union for
//! explicit meshes. Unwired slots are skipped; merging is associative,
//! so chain nodes for more parts.
//!
//! Only fields present in *every* wired part (same name, component
//! count, and container) survive the merge; the rest are dropped rather
//! than padded with invented values.
//!
//! With `weld_tolerance > 0`, nodes within the tolerance weld into their
//! first occurrence afterward (see
//! `mesh_edit_core::weld_coincident_nodes`): overlapping point clouds
//! dedupe, and strut networks meeting at shared boundary joints stitch
//! into one connected network (duplicate struts collapse). 0 keeps
//! everything as-is.
//!
//! Inputs:
//! - Input 0: CBOR configuration:
//!   `{ weld_tolerance: float .default 0.0 }`
//! - Inputs 1-4: FeaMesh (optional slots) — the meshes to merge
//!
//! Output 0: the merged CBOR-encoded `FeaMesh`.

use volumetric_abi::fea::{FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Number of mesh input slots. Unwired ones are ignored; merging is
/// associative, so more parts than this can be chained across nodes.
const MESH_SLOTS: usize = 4;

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct MergeConfig {
    /// Nodes closer than this weld into one after concatenation;
    /// 0 disables.
    weld_tolerance: f64,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            weld_tolerance: 0.0,
        }
    }
}

fn build_merged(config: &MergeConfig) -> Result<FeaMesh, String> {
    if !(config.weld_tolerance.is_finite() && config.weld_tolerance >= 0.0) {
        return Err(format!(
            "weld_tolerance must be non-negative, got {}",
            config.weld_tolerance
        ));
    }

    let mut parts: Vec<FeaMesh> = Vec::new();
    for idx in 1..=MESH_SLOTS {
        let bytes = read_input(idx as i32);
        if bytes.is_empty() {
            continue;
        }
        let mesh = decode_fea_mesh(&bytes)
            .map_err(|e| format!("input {idx} is not a usable mesh: {e}"))?;
        parts.push(mesh);
    }
    if parts.is_empty() {
        return Err("no meshes wired (connect at least one mesh input)".to_string());
    }

    let refs: Vec<&FeaMesh> = parts.iter().collect();
    let mut merged = mesh_edit_core::concat_meshes(&refs)?;
    if config.weld_tolerance > 0.0 {
        merged = mesh_edit_core::weld_coincident_nodes(&merged, config.weld_tolerance)?;
    }
    Ok(merged)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(0);
        if buf.is_empty() {
            MergeConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    match build_merged(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("mesh merge failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ weld_tolerance: float .default 0.0 }"#.to_string();
        let mut inputs = vec![OperatorMetadataInput::CBORConfiguration(schema)];
        let mut input_names = vec!["Config".to_string()];
        for i in 1..=MESH_SLOTS {
            inputs.push(OperatorMetadataInput::FeaMesh);
            input_names.push(format!("Mesh {i}"));
        }
        OperatorMetadata {
            name: "mesh_merge_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Mesh Merge".to_string(),
            description: "Concatenate same-kind meshes (points, struts, volumes), \
                          optionally welding coincident nodes."
                .to_string(),
            category: "Mesh".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="m8 6 4-4 4 4"/>"##,
                r##"<path d="M12 2v10.3a4 4 0 0 1-1.172 2.872L4 20"/>"##,
                r##"<path d="m20 20-5-5"/>"##,
            )
            .to_string(),
            inputs,
            input_names,
            outputs: vec![OperatorMetadataOutput::FeaMesh],
        }
    })
}
