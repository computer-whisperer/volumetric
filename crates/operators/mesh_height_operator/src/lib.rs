//! Mesh Height Operator.
//!
//! Wraps a triangle mesh as a 2D height-query model: `sample(u, v)` is the
//! height of the mesh surface over that lateral point — the extreme
//! coordinate along the height axis where the axis-aligned line through
//! (u, v) crosses a triangle. Where the line misses the mesh entirely, the
//! sample is the configured `miss` value.
//!
//! The model's two coordinates are the non-height axes in ascending order
//! (axis z → (x, y), axis y → (x, z), axis x → (y, z)), matching the FEA
//! target-map convention, and its bounds are the mesh bounds projected
//! onto those axes. Typical uses: turn a scanned body mesh into an FEA
//! target/interface map, or chain into `heightmap_extrude_operator` to
//! rebuild the solid under the surface.
//!
//! Same construction pattern as `mesh_to_model_operator`: the BVH payload
//! is built here at conversion time (`trimesh_model_core::build_payload`)
//! and patched into the prebuilt, stateless
//! `trimesh_height_model_template` module as data segments, along with a
//! 12-byte query config (axis, surface, miss).
//!
//! Inputs:
//! - Input 0: TriMesh (must have at least one triangle)
//! - Input 1: CBOR config `{ axis, surface, miss }` — height axis
//!   ("x"/"y"/"z", default "z"), which crossing to report ("top" = max,
//!   "bottom" = min, default "top"), and the sample value for lateral
//!   points the mesh doesn't cover (default 0.0).
//!
//! Output 0: ModelWASM (2D).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p trimesh_height_model_template
//! cp target/wasm32-unknown-unknown/release/trimesh_height_model_template.wasm \
//!    crates/operators/mesh_height_operator/template/
//! ```

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::trimesh::decode_tri_mesh;
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
use walrus::{FunctionId, Module, ModuleConfig};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/trimesh_height_model_template.wasm");

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct HeightConfig {
    axis: String,
    surface: String,
    miss: f64,
}

impl Default for HeightConfig {
    fn default() -> Self {
        Self {
            axis: "z".to_string(),
            surface: "top".to_string(),
            miss: 0.0,
        }
    }
}

impl HeightConfig {
    /// The template's 12-byte config-slot encoding: axis u32, surface u32,
    /// miss f32.
    fn slot_bytes(&self) -> Result<[u8; 12], String> {
        let axis: u32 = match self.axis.as_str() {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            other => {
                return Err(format!(
                    "axis must be \"x\", \"y\" or \"z\", got \"{other}\""
                ));
            }
        };
        let surface: u32 = match self.surface.as_str() {
            "top" => 0,
            "bottom" => 1,
            other => {
                return Err(format!(
                    "surface must be \"top\" or \"bottom\", got \"{other}\""
                ));
            }
        };
        if !self.miss.is_finite() {
            return Err(format!("miss must be finite, got {}", self.miss));
        }
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&axis.to_le_bytes());
        out[4..8].copy_from_slice(&surface.to_le_bytes());
        out[8..12].copy_from_slice(&(self.miss as f32).to_le_bytes());
        Ok(out)
    }
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

fn patch_template(payload: &[u8], config_bytes: [u8; 12]) -> Result<Vec<u8>, String> {
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

    // The patch slots' addresses, then drop the helper exports — they are
    // not part of the Model ABI.
    let mut slot_addr = |name: &str| -> Result<i32, String> {
        let export = module
            .exports
            .iter()
            .find(|e| e.name == name)
            .map(|e| (e.id(), e.item))
            .ok_or(format!("template missing {name} export"))?;
        let addr = match export.1 {
            walrus::ExportItem::Function(f) => const_i32_return(&module, f)
                .ok_or(format!("template {name} is not a constant function"))?,
            _ => return Err(format!("template {name} is not a function")),
        };
        module.exports.delete(export.0);
        Ok(addr)
    };
    let payload_slot = slot_addr("mesh_payload_slot")?;
    let config_slot = slot_addr("height_config_slot")?;

    // Payload in freshly reserved pages; base address into its slot, query
    // config into its own.
    let base = {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65536;
        memory.initial += (payload.len() as u64).div_ceil(65536);
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
        base
    };
    let patch = |module: &mut Module, addr: i32, bytes: Vec<u8>| {
        module.data.add(
            walrus::DataKind::Active {
                memory: memory_id,
                offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(addr)),
            },
            bytes,
        );
    };
    patch(&mut module, base as i32, payload.to_vec());
    patch(
        &mut module,
        payload_slot,
        (base as u32).to_le_bytes().to_vec(),
    );
    patch(&mut module, config_slot, config_bytes.to_vec());

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            HeightConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<HeightConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    let result = cfg.slot_bytes().and_then(|config_bytes| {
        decode_tri_mesh(&read_input(0))
            .and_then(|mesh| trimesh_model_core::build_payload(&mesh))
            .and_then(|payload| patch_template(&payload, config_bytes))
    });
    match result {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("mesh height query conversion failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ axis: \"x\" / \"y\" / \"z\" .default \"z\", \
                      surface: \"top\" / \"bottom\" .default \"top\", \
                      miss: float .default 0.0 }"
            .to_string();
        OperatorMetadata {
            name: "mesh_height_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::TriMesh,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
