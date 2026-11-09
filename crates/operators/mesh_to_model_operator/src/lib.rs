//! Mesh to Model Operator.
//!
//! Converts a triangle mesh into a sampleable model: point-in-mesh by ray
//! parity over a BVH. All the work happens here at conversion time —
//! `trimesh_model_core::build_payload` constructs and serializes the BVH
//! (and compresses the texture), and this operator patches the payload
//! into a prebuilt, stateless template module (`trimesh_model_template`)
//! as plain data segments. No instructions are synthesized, so there are
//! no hand-maintained code offsets to drift (the failure mode that killed
//! the old all-in-one STL importer).
//!
//! Inside/outside uses crossing parity, so a watertight mesh behaves as
//! the enclosed solid. Open meshes are not rejected — parity's literal
//! behavior applies (see `trimesh_model_core`) — but only closed surfaces
//! give meaningful solids.
//!
//! # Surface colour
//!
//! A mesh that carries colour — a base-colour texture addressed by the
//! `uv` vertex field (OBJ), or the `color` vertex field (PLY, 3MF) —
//! becomes a model with typed sample channels: occupancy plus the sRGB
//! trio (`color_r`, `color_g`, `color_b`), the colour of the nearest
//! surface point, defined everywhere in space. Every downstream operator
//! that propagates channels (boolean, pose, translate...) carries it for
//! free, and the preview's true-colour mode shows it. A mesh without
//! colour yields the plain occupancy-only model: the channel exports are
//! stripped, so `SampleFormat::color_trio()` is `None`.
//!
//! Inputs:
//! - Input 0: TriMesh (must have at least one triangle)
//! - Input 1: CBOR config (optional; a step with only the mesh input, as
//!   projects written before this slot existed have, gets the defaults):
//!   - `texture` (`"bc1"` | `"rgb8"` | `"none"`, default `"bc1"`): texel
//!     encoding of the texture stored in the model. `bc1` is 6x smaller
//!     than `rgb8` and lossy (fine for photographic atlases); `none`
//!     ignores the texture and falls back to vertex colour, if any.
//!   - `texture_max_size` (default 0 = full size): box-filter the texture
//!     down until both extents are at most this many texels.
//!
//! A texture that fails to decode is reported as a warning naming the
//! texture and the error; the conversion proceeds without it (vertex
//! colour if present, else occupancy-only).
//!
//! Output 0: ModelWASM (3D).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p trimesh_model_template
//! cp target/wasm32-unknown-unknown/release/trimesh_model_template.wasm \
//!    crates/operators/mesh_to_model_operator/template/
//! ```

use model_wrap_core::const_i32_return;
use trimesh_model_core::{BuildOptions, PayloadView, TextureFormat, TextureImage};
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
use volumetric_abi::trimesh::{MeshTexture, decode_tri_mesh};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
use walrus::{Module, ModuleConfig};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/trimesh_model_template.wasm");

/// CDDL schema of the configuration input.
const CONFIG_SCHEMA: &str =
    r#"{ texture: tstr .default "bc1", texture_max_size: uint .default 0 }"#;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct MeshToModelConfig {
    /// `"bc1"`, `"rgb8"` or `"none"`.
    pub texture: String,
    /// 0 = full size.
    pub texture_max_size: u32,
}

impl Default for MeshToModelConfig {
    fn default() -> Self {
        MeshToModelConfig {
            texture: "bc1".to_string(),
            texture_max_size: 0,
        }
    }
}

impl MeshToModelConfig {
    /// The texel encoding to store, or `None` for `"none"`.
    fn texture_format(&self) -> Result<Option<TextureFormat>, String> {
        match self.texture.as_str() {
            "bc1" => Ok(Some(TextureFormat::Bc1)),
            "rgb8" => Ok(Some(TextureFormat::Rgb8)),
            "none" => Ok(None),
            other => Err(format!(
                "texture must be \"bc1\", \"rgb8\" or \"none\", got {other:?}"
            )),
        }
    }
}

/// Decode a mesh texture's file bytes (JPEG or PNG) to RGB8 pixels.
fn decode_texture(texture: &MeshTexture) -> Result<TextureImage, String> {
    let image = image::load_from_memory(&texture.bytes)
        .map_err(|e| e.to_string())?
        .into_rgb8();
    Ok(TextureImage {
        width: image.width(),
        height: image.height(),
        rgb: image.into_raw(),
    })
}

/// Mesh bytes + config -> payload. A texture that fails to decode is a
/// warning, not a failure: the geometry is still a valid solid, so the
/// colour falls back to the vertex field or to none.
pub fn build(mesh_bytes: &[u8], cfg: &MeshToModelConfig) -> Result<Vec<u8>, String> {
    let mesh = decode_tri_mesh(mesh_bytes)?;
    let format = cfg.texture_format()?;
    let decode_or_warn = |texture: &MeshTexture| match decode_texture(texture) {
        Ok(image) => Some(image),
        Err(e) => {
            post_warning(&format!(
                "texture {:?} could not be decoded ({e}); converting without it",
                texture.name
            ));
            None
        }
    };
    let decoder: &dyn Fn(&MeshTexture) -> Option<TextureImage> = &decode_or_warn;
    let options = BuildOptions {
        texture_format: format.unwrap_or_default(),
        texture_max_size: cfg.texture_max_size,
        decode_texture: format.map(|_| decoder),
    };
    trimesh_model_core::build_payload(&mesh, &options)
}

/// Patch a payload into the embedded template. `has_color` keeps the
/// typed-channel exports (`get_sample_format`, `sample_channels` — the
/// sRGB surface-color triple); payloads without a colour section drop
/// them and stay occupancy-only.
pub fn patch_template(payload: &[u8], has_color: bool) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(TEMPLATE, &config)
        .map_err(|e| format!("failed to parse the embedded template: {e}"))?;

    if !has_color {
        let channel_exports: Vec<_> = module
            .exports
            .iter()
            .filter(|e| e.name == "get_sample_format" || e.name == "sample_channels")
            .map(|e| e.id())
            .collect();
        for id in channel_exports {
            module.exports.delete(id);
        }
    }

    let memory_id = module
        .exports
        .iter()
        .find(|e| e.name == "memory")
        .and_then(|e| match e.item {
            walrus::ExportItem::Memory(m) => Some(m),
            _ => None,
        })
        .ok_or("template missing memory export")?;

    // The patch slot's address, then drop the helper export — it is not
    // part of the Model ABI.
    let slot_export = module
        .exports
        .iter()
        .find(|e| e.name == "mesh_payload_slot")
        .map(|e| (e.id(), e.item))
        .ok_or("template missing mesh_payload_slot export")?;
    let slot_addr = match slot_export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(&module, f)
            .ok_or("template mesh_payload_slot is not a constant function")?,
        _ => return Err("template mesh_payload_slot is not a function".to_string()),
    };
    module.exports.delete(slot_export.0);

    // Payload in freshly reserved pages; base address into the slot.
    let base = {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65536;
        memory.initial += (payload.len() as u64).div_ceil(65536);
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
        base
    };
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(base as i32)),
        },
        payload.to_vec(),
    );
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(slot_addr)),
        },
        (base as u32).to_le_bytes().to_vec(),
    );

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    // A step authored before the config slot existed carries one input;
    // the host reads a missing slot as empty, which means the defaults.
    let cfg_buf = read_input(1);
    let cfg = if cfg_buf.is_empty() {
        MeshToModelConfig::default()
    } else {
        match ciborium::de::from_reader::<MeshToModelConfig, _>(std::io::Cursor::new(&cfg_buf)) {
            Ok(cfg) => cfg,
            Err(e) => {
                report_error(&format!("invalid configuration: {e}"));
                return;
            }
        }
    };

    let result = build(&read_input(0), &cfg).and_then(|payload| {
        let has_color = PayloadView::new(&payload)
            .map_err(|e| format!("built payload failed validation: {e}"))?
            .has_color();
        patch_template(&payload, has_color)
    });
    match result {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("mesh to model conversion failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "mesh_to_model_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        docs: String::new(),
        display_name: "Mesh to Model".to_string(),
        description: "Convert a triangle mesh into a sampleable model via BVH ray parity; \
                      vertex colours and textures become the sRGB surface-colour channels."
            .to_string(),
        category: "Mesh".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<path d="M12 3a9 9 0 0 1 0 18"/>"##,
            r##"<path d="M12 3 5.5 7.5v9l6.5 4.5"/>"##,
            r##"<path d="M12 3v18"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::TriMesh,
            OperatorMetadataInput::CBORConfiguration(CONFIG_SCHEMA.to_string()),
        ],
        variadic_input: None,
        input_names: vec!["Mesh".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
        output_names: vec![],
    })
}
