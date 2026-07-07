//! FEA deform operator: applies a solved mesh's displacement field to a
//! 3D model as a space deformation.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Takes a 3D model and an FEA mesh carrying a per-node displacement
//! field (e.g. `fea_solve_operator` output) and produces the deformed
//! model: sampling the output at a point `y` locates the deformed element
//! containing `y`, inverts its trilinear map back to the material point
//! `X`, and samples the input model there — the pullback `f(phi^-1(y))`,
//! so occupancy carries over exactly (see `deform_model_core` for the
//! math, payload layout, and the boundary-skin behavior for points just
//! outside the mesh).
//!
//! Mechanism: `deform_model_core::build_payload` bakes the scaled
//! displacement and a BVH over the deformed elements; the payload is
//! patched into the embedded `deform_model_template` module (the
//! evaluator) as data segments, which is then merged with the input model
//! via `model_merge_core`. Emitted glue rewrites each query position to
//! its material point before handing it to the input model.
//!
//! - `scale` multiplies the displacement before it is baked in (0 gives
//!   the identity map over the mesh region; 1 the solved deformation).
//! - `field` names the 3-component node field to use.
//! - `boundary_skin` is the fallback distance for points outside every
//!   deformed element (0 picks half the mean element diagonal — about the
//!   overhang the cell-center grid mesher can leave).
//!
//! Bounds come from the deformed mesh (plus the skin), NOT the input
//! model: the deformed part only exists where the mesh moved it.
//! Dimensions, the IO buffer, and the channel layout pass through from
//! the input; `sample_channels` reads the input's channels at the
//! material point, with channel 0 forced to 0 outside the deformed mesh.

use wasm_encoder::{ExportKind, ExportSection, Function, Instruction, MemArg, ValType};

use model_merge_core::{
    MergeSections, OffsetReencoder, count_sections, find_function_export, parse_model_exports,
};
use volumetric_abi::fea::decode_fea_mesh;
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Prebuilt `deform_model_template` module (see that crate's docs for the
/// regeneration command).
static TEMPLATE: &[u8] = include_bytes!("../template/deform_model_template.wasm");

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct DeformConfig {
    /// Displacement multiplier baked into the deformation.
    scale: f64,
    /// The 3-component node field holding the displacement.
    field: String,
    /// Fallback distance for points outside every deformed element;
    /// 0 = auto (half the mean deformed-element diagonal).
    boundary_skin: f64,
}

impl Default for DeformConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            field: "displacement".to_string(),
            boundary_skin: 0.0,
        }
    }
}

/// Read the constant a trivial `() -> i32` function returns.
fn const_i32_return(module: &walrus::Module, func_id: walrus::FunctionId) -> Option<i32> {
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

/// Patch the payload into the template (fresh pages + base address into
/// the slot, as in `mesh_to_model_operator`) and read the result-buffer
/// address the glue bakes in. Returns the patched module bytes and that
/// address.
fn patch_template(payload: &[u8]) -> Result<(Vec<u8>, i32), String> {
    let mut module =
        walrus::Module::from_buffer_with_config(TEMPLATE, &walrus::ModuleConfig::new())
            .map_err(|e| format!("failed to parse the embedded template: {e}"))?;

    let memory_id = module
        .memories
        .iter()
        .next()
        .map(|m| m.id())
        .ok_or("template has no memory")?;

    let find_const_export =
        |module: &walrus::Module, name: &str| -> Result<(walrus::ExportId, i32), String> {
            let export = module
                .exports
                .iter()
                .find(|e| e.name == name)
                .map(|e| (e.id(), e.item))
                .ok_or_else(|| format!("template missing {name} export"))?;
            match export.1 {
                walrus::ExportItem::Function(f) => const_i32_return(module, f)
                    .map(|v| (export.0, v))
                    .ok_or_else(|| format!("template {name} is not a constant function")),
                _ => Err(format!("template {name} is not a function")),
            }
        };
    let (slot_export, slot_addr) = find_const_export(&module, "deform_payload_slot")?;
    let (result_export, result_ptr) = find_const_export(&module, "deform_result_ptr")?;
    module.exports.delete(slot_export);
    module.exports.delete(result_export);

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

    Ok((module.emit_wasm(), result_ptr))
}

/// Statically confirm a model declares itself 3D (constant
/// `get_dimensions`). Models with computed dimensions pass through — the
/// host rejects a dimension mismatch at run time.
fn check_three_dimensional(input: &[u8]) -> Result<(), String> {
    let module = walrus::Module::from_buffer_with_config(input, &walrus::ModuleConfig::new())
        .map_err(|e| format!("failed to parse the input model: {e}"))?;
    let Some(func) = module.exports.iter().find_map(|e| match e.item {
        walrus::ExportItem::Function(f) if e.name == "get_dimensions" => Some(f),
        _ => None,
    }) else {
        return Err("input model exports no get_dimensions".to_string());
    };
    match const_i32_return(&module, func) {
        Some(3) | None => Ok(()),
        Some(dims) => Err(format!(
            "fea_deform_operator needs a 3D input model, got {dims}D"
        )),
    }
}

struct GlueIndices {
    a_sample: u32,
    a_sample_channels: Option<u32>,
    a_memory: u32,
    t_memory: u32,
    pull_back: u32,
    deform_bounds: u32,
    /// Address of the template's 3xf64 pullback result, in its memory.
    result_ptr: i32,
}

impl GlueIndices {
    fn a_mem_f64(&self, offset: u64) -> MemArg {
        MemArg {
            offset,
            align: 3,
            memory_index: self.a_memory,
        }
    }

    fn t_mem_f64(&self, offset: u64) -> MemArg {
        MemArg {
            offset,
            align: 3,
            memory_index: self.t_memory,
        }
    }

    /// Emit `call pull_back(pos[0], pos[1], pos[2])` from A's memory.
    fn emit_pull_back_call(&self, f: &mut Function) {
        for axis in 0..3u64 {
            f.instruction(&Instruction::LocalGet(0));
            f.instruction(&Instruction::F64Load(self.a_mem_f64(axis * 8)));
        }
        f.instruction(&Instruction::Call(self.pull_back));
    }

    /// Emit `pos[k] = template_result[k]` for all three axes: the query
    /// position in A's IO buffer is overwritten with the material point
    /// (the ABI allows sample to clobber its position buffer).
    fn emit_write_material_point(&self, f: &mut Function) {
        for axis in 0..3u64 {
            f.instruction(&Instruction::LocalGet(0));
            f.instruction(&Instruction::I32Const(self.result_ptr));
            f.instruction(&Instruction::F64Load(self.t_mem_f64(axis * 8)));
            f.instruction(&Instruction::F64Store(self.a_mem_f64(axis * 8)));
        }
    }
}

/// Emit the merged `sample(pos_ptr) -> f32` glue: pull the query back to
/// its material point and sample the input model there; outside the
/// deformed mesh (and skin) the output is empty.
fn add_sample_glue(sections: &mut MergeSections, exports: &mut ExportSection, idx: &GlueIndices) {
    let ty = sections.types.len();
    sections.types.ty().function([ValType::I32], [ValType::F32]);
    sections.funcs.function(ty);

    let mut f = Function::new([]);
    idx.emit_pull_back_call(&mut f);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Result(
        ValType::F32,
    )));
    idx.emit_write_material_point(&mut f);
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::Call(idx.a_sample));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::F32Const(0.0.into()));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);
    sections.code.function(&f);

    let func_index = sections.funcs.len() - 1;
    exports.export("sample", ExportKind::Func, func_index);
}

/// Emit the merged `sample_channels(pos_ptr, out_ptr)` glue: the input's
/// channel row at the material point, with channel 0 (occupancy) forced
/// to 0 outside the deformed mesh.
fn add_sample_channels_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    idx: &GlueIndices,
    a_sample_channels: u32,
) {
    let ty = sections.types.len();
    sections
        .types
        .ty()
        .function([ValType::I32, ValType::I32], []);
    sections.funcs.function(ty);

    // params: 0 pos_ptr, 1 out_ptr; local 2: the pullback status.
    let mut f = Function::new([(1, ValType::I32)]);
    idx.emit_pull_back_call(&mut f);
    f.instruction(&Instruction::LocalSet(2));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    idx.emit_write_material_point(&mut f);
    f.instruction(&Instruction::End);
    // Fill the row from the input model (at the material point when
    // inside, at the raw query otherwise — the occupancy override below
    // makes the outside case read as empty either way).
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::Call(a_sample_channels));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::I32Eqz);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::F32Const(0.0.into()));
    f.instruction(&Instruction::F32Store(MemArg {
        offset: 0,
        align: 2,
        memory_index: idx.a_memory,
    }));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);
    sections.code.function(&f);

    let func_index = sections.funcs.len() - 1;
    exports.export("sample_channels", ExportKind::Func, func_index);
}

/// Emit the merged `get_bounds(out_ptr)` glue: the template's payload
/// bounds (deformed mesh + skin), NOT the input model's.
fn add_get_bounds_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    idx: &GlueIndices,
) {
    let ty = sections.types.len();
    sections.types.ty().function([ValType::I32], []);
    sections.funcs.function(ty);

    let mut f = Function::new([]);
    for i in 0..6 {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Const(i));
        f.instruction(&Instruction::Call(idx.deform_bounds));
        f.instruction(&Instruction::F64Store(idx.a_mem_f64(i as u64 * 8)));
    }
    f.instruction(&Instruction::End);
    sections.code.function(&f);

    let func_index = sections.funcs.len() - 1;
    exports.export("get_bounds", ExportKind::Func, func_index);
}

fn build_deformed_model(
    input: &[u8],
    mesh_bytes: &[u8],
    config: &DeformConfig,
) -> Result<Vec<u8>, String> {
    if !config.scale.is_finite() {
        return Err(format!("scale must be finite, got {}", config.scale));
    }
    if !(config.boundary_skin.is_finite() && config.boundary_skin >= 0.0) {
        return Err(format!(
            "boundary_skin must be finite and non-negative, got {}",
            config.boundary_skin
        ));
    }
    if config.field.is_empty() {
        return Err("field must not be empty".to_string());
    }
    check_three_dimensional(input)?;

    let mesh = decode_fea_mesh(mesh_bytes)?;
    let skin = (config.boundary_skin > 0.0).then_some(config.boundary_skin);
    let payload = deform_model_core::build_payload(&mesh, &config.field, config.scale, skin)?;
    let (template, result_ptr) = patch_template(&payload)?;

    let a_counts = count_sections(input)?;
    let b_counts = count_sections(&template)?;
    let a = parse_model_exports(input)?;
    let pull_back = find_function_export(&template, "deform_pull_back")? + a_counts.funcs;
    let deform_bounds = find_function_export(&template, "deform_bounds")? + a_counts.funcs;

    let mut sections = MergeSections::default();
    sections.append_module(input, &mut OffsetReencoder::identity())?;
    sections.append_module(&template, &mut OffsetReencoder::after(&a_counts))?;

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, a.memory);
    exports.export("get_dimensions", ExportKind::Func, a.get_dimensions);
    exports.export("get_io_ptr", ExportKind::Func, a.get_io_ptr);
    if let Some(get_sample_format) = a.get_sample_format {
        exports.export("get_sample_format", ExportKind::Func, get_sample_format);
    }

    let idx = GlueIndices {
        a_sample: a.sample,
        a_sample_channels: a.sample_channels,
        a_memory: a.memory,
        t_memory: a_counts.memories,
        pull_back,
        deform_bounds,
        result_ptr,
    };
    add_sample_glue(&mut sections, &mut exports, &idx);
    if let Some(a_sample_channels) = idx.a_sample_channels {
        add_sample_channels_glue(&mut sections, &mut exports, &idx, a_sample_channels);
    }
    add_get_bounds_glue(&mut sections, &mut exports, &idx);

    let data_count =
        (a_counts.has_data_count || b_counts.has_data_count).then(|| a_counts.data + b_counts.data);
    Ok(sections.finish(&exports, data_count))
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(2);
        if buf.is_empty() {
            DeformConfig::default()
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

    let input = read_input(0);
    let mesh_bytes = read_input(1);
    match build_deformed_model(&input, &mesh_bytes, &config) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ scale: float .default 1.0, field: text .default "displacement", boundary_skin: float .default 0.0 }"#.to_string();
        OperatorMetadata {
            name: "fea_deform_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec![
                "Model".to_string(),
                "Deformed mesh".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
