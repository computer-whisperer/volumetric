//! Image Model Operator.
//!
//! Loads an image (PNG/JPEG/BMP/GIF, 16-bit grayscale PNG supported) as a
//! 2D scalar-field model: `sample(x, y)` is the bilinearly interpolated
//! pixel luminance, normalized to [0, 1]. The output composes like any 2D
//! sketch — extrude it (`heightmap_extrude_operator` for value-modulated
//! height, `extrude_operator` for a constant slab), or use it as an FEA
//! target force map.
//!
//! All decoding happens here at conversion time: the pixels are baked into
//! a `gridfield_model_core` payload and patched into the prebuilt,
//! stateless `gridfield_model_template` module as data segments — the same
//! pattern as `mesh_to_model_operator`, with no hand-maintained codegen
//! offsets.
//!
//! Coordinates are z-up-sketch convention: the image spans the (x, y)
//! plane centered on the origin, +y is image-up (rows are flipped so the
//! picture isn't mirrored), and samples outside the image rectangle are
//! 0.0.
//!
//! Inputs:
//! - Input 0: Blob — the image file bytes
//! - Input 1: CBOR config `{ width, height }` — the model-space extents.
//!   `width` defaults to 1.0; `height` 0.0 (the default) means "preserve
//!   the image's aspect ratio".
//!
//! Output 0: ModelWASM (2D).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p gridfield_model_template
//! cp target/wasm32-unknown-unknown/release/gridfield_model_template.wasm \
//!    crates/operators/image_model_operator/template/
//! ```

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
use walrus::{FunctionId, Module, ModuleConfig};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/gridfield_model_template.wasm");

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct ImageConfig {
    /// Model-space width (x extent).
    width: f64,
    /// Model-space height (y extent); 0 = width * image aspect ratio.
    height: f64,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 0.0,
        }
    }
}

/// Decode the image and bake it into a grid-field payload.
fn image_payload(image_bytes: &[u8], cfg: &ImageConfig) -> Result<Vec<u8>, String> {
    if !(cfg.width > 0.0 && cfg.width.is_finite()) {
        return Err(format!("image width must be > 0, got {}", cfg.width));
    }
    if !(cfg.height >= 0.0 && cfg.height.is_finite()) {
        return Err(format!(
            "image height must be > 0 (or 0 to preserve the aspect ratio), got {}",
            cfg.height
        ));
    }

    let img = image::load_from_memory(image_bytes).map_err(|e| format!("image decode: {e}"))?;
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err("image has zero dimensions".to_string());
    }

    // 16-bit grayscale preserves full precision for 16-bit PNGs; 8-bit
    // images upconvert without loss. Rows flip so +y is image-up.
    let gray = img.to_luma16();
    let mut values = vec![0.0f32; (w as usize) * (h as usize)];
    for (x, y, p) in gray.enumerate_pixels() {
        let row = (h - 1 - y) as usize;
        values[row * w as usize + x as usize] = p.0[0] as f32 / 65535.0;
    }

    let height = if cfg.height > 0.0 {
        cfg.height
    } else {
        cfg.width * h as f64 / w as f64
    };
    let bounds = [
        -cfg.width / 2.0,
        cfg.width / 2.0,
        -height / 2.0,
        height / 2.0,
    ];
    gridfield_model_core::build_payload(w, h, bounds, &values)
}

/// Read the constant a trivial `() -> i32` function returns.
fn const_i32_return(module: &Module, func_id: FunctionId) -> Option<i32> {
    let local = match &module.funcs.get(func_id).kind {
        walrus::FunctionKind::Local(local) => local,
        _ => return None,
    };
    let block = local.block(local.entry_block());
    match block.instrs.as_slice() {
        [(walrus::ir::Instr::Const(c), _)] => match c.value {
            walrus::ir::Value::I32(v) => Some(v),
            _ => None,
        },
        _ => None,
    }
}

fn patch_template(payload: &[u8]) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(TEMPLATE, &config)
        .map_err(|e| format!("failed to parse the embedded template: {e}"))?;

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
        .find(|e| e.name == "gridfield_payload_slot")
        .map(|e| (e.id(), e.item))
        .ok_or("template missing gridfield_payload_slot export")?;
    let slot_addr = match slot_export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(&module, f)
            .ok_or("template gridfield_payload_slot is not a constant function")?,
        _ => return Err("template gridfield_payload_slot is not a function".to_string()),
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
    let cfg = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            ImageConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<ImageConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    let result = image_payload(&read_input(0), &cfg).and_then(|payload| patch_template(&payload));
    match result {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("image model conversion failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ width: float .default 1.0, height: float .default 0.0 }".to_string();
        OperatorMetadata {
            name: "image_model_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Image".to_string(),
            description: "Load an image file as a 2D scalar-field model of pixel luminance."
                .to_string(),
            category: "Import".to_string(),
            icon_svg: String::new(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Image file".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
