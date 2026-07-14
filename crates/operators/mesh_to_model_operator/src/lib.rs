//! Mesh to Model Operator.
//!
//! Converts a triangle mesh into a sampleable model: point-in-mesh by ray
//! parity over a BVH. All the work happens here at conversion time —
//! `trimesh_model_core::build_payload` constructs and serializes the BVH,
//! and this operator patches the payload into a prebuilt, stateless
//! template module (`trimesh_model_template`) as plain data segments. No
//! instructions are synthesized, so there are no hand-maintained code
//! offsets to drift (the failure mode that killed the old all-in-one STL
//! importer).
//!
//! Inside/outside uses crossing parity, so a watertight mesh behaves as
//! the enclosed solid. Open meshes are not rejected — parity's literal
//! behavior applies (see `trimesh_model_core`) — but only closed surfaces
//! give meaningful solids.
//!
//! Inputs:
//! - Input 0: TriMesh (must have at least one triangle)
//!
//! Output 0: ModelWASM (3D, occupancy-only).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p trimesh_model_template
//! cp target/wasm32-unknown-unknown/release/trimesh_model_template.wasm \
//!    crates/operators/mesh_to_model_operator/template/
//! ```

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::trimesh::decode_tri_mesh;
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
use walrus::{FunctionId, Module, ModuleConfig};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/trimesh_model_template.wasm");

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
    let result = decode_tri_mesh(&read_input(0))
        .and_then(|mesh| trimesh_model_core::build_payload(&mesh))
        .and_then(|payload| patch_template(&payload));
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
        display_name: "Mesh to Model".to_string(),
        description: "Convert a triangle mesh into a sampleable model via BVH ray parity."
            .to_string(),
        category: "Mesh".to_string(),
        icon_svg: String::new(),
        inputs: vec![OperatorMetadataInput::TriMesh],
        input_names: vec!["Mesh".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}
