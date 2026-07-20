//! Coil operator: rolls a flat 3D model up into an Archimedean spiral,
//! like rolling a carpet.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! The input model is treated as a flat sheet: its x axis is rolled along
//! the spiral (x min at the inner end, wrapping counterclockwise from +x
//! toward +z around the world y axis), its z axis becomes radial depth
//! within each wrap (the z min face on the inside), and y passes through
//! as the coil axis. The radial advance per turn is the sheet's z extent
//! plus the configured `gap`, so consecutive wraps clear each other by
//! construction.
//!
//! Mechanism: the embedded `coil_model_template` module (the spiral
//! pull-back evaluator, with `inner_radius` and `gap` patched into its
//! config slot) is merged with the input model via `model_merge_core`;
//! emitted glue reads the input's own `get_bounds` per sample for the
//! sheet's x/z extents, unwinds the world point to its flat sheet point,
//! and samples the input there. Points the spiral never covers (the bore,
//! the clearance gap, past the sheet's end) pull back to a sentinel
//! outside the sheet bounds, so the input itself answers "outside" —
//! the glue never branches.
//!
//! Like the transform operators, this is pure module surgery (no host
//! sampling imports). Dimensions, IO buffer, and sample format pass
//! through from the input; `get_bounds` is wrapped to report the coil's
//! world extents (a full disc of the outermost radius — loose for sheets
//! shorter than one turn).

use wasm_encoder::{ExportKind, ExportSection, Function, Instruction, MemArg, ValType};

use model_merge_core::{
    MergeSections, OffsetReencoder, count_sections, find_function_export, find_memory_export,
    parse_model_exports,
};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Prebuilt `coil_model_template` module (see that crate's docs for the
/// regeneration command).
static TEMPLATE: &[u8] = include_bytes!("../template/coil_model_template.wasm");

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct CoilConfig {
    /// Spiral start radius (the bore), metres.
    inner_radius: f64,
    /// Radial clearance between consecutive wraps, metres.
    gap: f64,
}

impl Default for CoilConfig {
    fn default() -> Self {
        Self {
            inner_radius: 0.005,
            gap: 0.001,
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

/// Patch `inner_radius` and `gap` into the template's config slot, drop
/// the patch-helper export, and read the result address the glue bakes in.
fn patch_template(config: &CoilConfig) -> Result<(Vec<u8>, i32), String> {
    let mut module =
        walrus::Module::from_buffer_with_config(TEMPLATE, &walrus::ModuleConfig::new())
            .map_err(|e| format!("failed to parse the embedded template: {e}"))?;

    let memory_id = module
        .memories
        .iter()
        .next()
        .map(|m| m.id())
        .ok_or("template has no memory")?;

    let export = module
        .exports
        .iter()
        .find(|e| e.name == "coil_config_slot")
        .map(|e| (e.id(), e.item))
        .ok_or("template missing coil_config_slot export")?;
    let slot = match export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(&module, f)
            .ok_or("template coil_config_slot is not a constant function")?,
        _ => return Err("template coil_config_slot is not a function".to_string()),
    };
    module.exports.delete(export.0);

    let result_ptr = module
        .exports
        .iter()
        .find(|e| e.name == "coil_result_ptr")
        .and_then(|e| match e.item {
            walrus::ExportItem::Function(f) => const_i32_return(&module, f),
            _ => None,
        })
        .ok_or("template coil_result_ptr is not a constant function")?;

    let mut bytes = Vec::with_capacity(16);
    bytes.extend(config.inner_radius.to_le_bytes());
    bytes.extend(config.gap.to_le_bytes());
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(slot)),
        },
        bytes,
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
        Some(dims) => Err(format!("coil_operator needs a 3D input model, got {dims}D")),
    }
}

struct GlueIndices {
    a_sample: u32,
    a_get_io_ptr: u32,
    a_get_bounds: u32,
    a_memory: u32,
    b_pull_back: u32,
    b_bound: u32,
    b_memory: u32,
    b_result_ptr: i32,
}

impl GlueIndices {
    fn mem_a(&self, offset: u64) -> MemArg {
        MemArg {
            offset,
            align: 3,
            memory_index: self.a_memory,
        }
    }

    fn mem_b(&self, offset: u64) -> MemArg {
        MemArg {
            offset,
            align: 3,
            memory_index: self.b_memory,
        }
    }
}

/// Emit the shared front half of the `sample`/`sample_channels` glue:
/// save the position to f64 locals (`first_f64..first_f64+2`), fetch the
/// input's bounds through its IO buffer (pointer left in local `io`),
/// pull the point back through the spiral, and copy the flat point into
/// the input's IO buffer, ready for its `sample`.
///
/// The position is saved BEFORE the input's `get_bounds` runs: the host
/// may pass `pos_ptr == io_ptr`, and the bounds fetch writes 6 f64s there.
fn emit_pull_back(f: &mut Function, idx: &GlueIndices, io: u32, first_f64: u32) {
    // x, y, z
    for axis in 0..3u64 {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::F64Load(idx.mem_a(axis * 8)));
        f.instruction(&Instruction::LocalSet(first_f64 + axis as u32));
    }
    // A.get_bounds(A.get_io_ptr()) — interleaved [x0,x1, y0,y1, z0,z1]
    f.instruction(&Instruction::Call(idx.a_get_io_ptr));
    f.instruction(&Instruction::LocalTee(io));
    f.instruction(&Instruction::Call(idx.a_get_bounds));
    // x0, x1, z0, z1
    for (slot, offset) in [(3u32, 0u64), (4, 8), (5, 32), (6, 40)] {
        f.instruction(&Instruction::LocalGet(io));
        f.instruction(&Instruction::F64Load(idx.mem_a(offset)));
        f.instruction(&Instruction::LocalSet(first_f64 + slot));
    }
    // coil_pull_back(x, y, z, x0, x1, z0, z1)
    for slot in 0..7u32 {
        f.instruction(&Instruction::LocalGet(first_f64 + slot));
    }
    f.instruction(&Instruction::Call(idx.b_pull_back));
    // The flat point, template memory -> the input's IO buffer.
    for axis in 0..3u64 {
        f.instruction(&Instruction::LocalGet(io));
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::F64Load(
            idx.mem_b(idx.b_result_ptr as u64 + axis * 8),
        ));
        f.instruction(&Instruction::F64Store(idx.mem_a(axis * 8)));
    }
}

/// Emit the merged `sample(pos_ptr) -> f32` glue.
fn add_sample_glue(sections: &mut MergeSections, exports: &mut ExportSection, idx: &GlueIndices) {
    let ty = sections.types.len();
    sections.types.ty().function([ValType::I32], [ValType::F32]);
    sections.funcs.function(ty);

    // param 0: pos_ptr; local 1: the input's IO pointer; locals 2..=8:
    // x, y, z, x0, x1, z0, z1.
    let mut f = Function::new([(1, ValType::I32), (7, ValType::F64)]);
    emit_pull_back(&mut f, idx, 1, 2);
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::Call(idx.a_sample));
    f.instruction(&Instruction::End);
    sections.code.function(&f);

    let func_index = sections.funcs.len() - 1;
    exports.export("sample", ExportKind::Func, func_index);
}

/// Emit the merged `sample_channels(pos_ptr, out_ptr)` glue: the same
/// pull-back, with the input's channel row passing through.
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

    // params 0, 1: pos_ptr, out_ptr; local 2: the input's IO pointer;
    // locals 3..=9: x, y, z, x0, x1, z0, z1.
    let mut f = Function::new([(1, ValType::I32), (7, ValType::F64)]);
    emit_pull_back(&mut f, idx, 2, 3);
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::Call(a_sample_channels));
    f.instruction(&Instruction::End);
    sections.code.function(&f);

    let func_index = sections.funcs.len() - 1;
    exports.export("sample_channels", ExportKind::Func, func_index);
}

/// Emit the merged `get_bounds(out_ptr)` glue: fetch the input's bounds
/// into locals, then overwrite the buffer with the template's coil bounds.
fn add_bounds_glue(sections: &mut MergeSections, exports: &mut ExportSection, idx: &GlueIndices) {
    let ty = sections.types.len();
    sections.types.ty().function([ValType::I32], []);
    sections.funcs.function(ty);

    // param 0: out_ptr; locals 1..=6: x0, x1, y0, y1, z0, z1.
    let mut f = Function::new([(6, ValType::F64)]);
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::Call(idx.a_get_bounds));
    for slot in 0..6u32 {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::F64Load(idx.mem_a(u64::from(slot) * 8)));
        f.instruction(&Instruction::LocalSet(1 + slot));
    }
    for i in 0..6u32 {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Const(i as i32));
        // coil_bound(i, x0, x1, z0, z1, y_min, y_max)
        for slot in [1u32, 2, 5, 6, 3, 4] {
            f.instruction(&Instruction::LocalGet(slot));
        }
        f.instruction(&Instruction::Call(idx.b_bound));
        f.instruction(&Instruction::F64Store(idx.mem_a(u64::from(i) * 8)));
    }
    f.instruction(&Instruction::End);
    sections.code.function(&f);

    let func_index = sections.funcs.len() - 1;
    exports.export("get_bounds", ExportKind::Func, func_index);
}

fn build_coil_model(input: &[u8], config: &CoilConfig) -> Result<Vec<u8>, String> {
    if !(config.inner_radius.is_finite() && config.inner_radius > 0.0) {
        return Err(format!(
            "inner_radius must be positive, got {}",
            config.inner_radius
        ));
    }
    if !(config.gap.is_finite() && config.gap >= 0.0) {
        return Err(format!("gap must be non-negative, got {}", config.gap));
    }
    check_three_dimensional(input)?;

    let (template, b_result_ptr) = patch_template(config)?;

    let a_counts = count_sections(input)?;
    let b_counts = count_sections(&template)?;
    let a = parse_model_exports(input)?;
    let idx = GlueIndices {
        a_sample: a.sample,
        a_get_io_ptr: a.get_io_ptr,
        a_get_bounds: a.get_bounds,
        a_memory: a.memory,
        b_pull_back: find_function_export(&template, "coil_pull_back")? + a_counts.funcs,
        b_bound: find_function_export(&template, "coil_bound")? + a_counts.funcs,
        b_memory: find_memory_export(&template)? + a_counts.memories,
        b_result_ptr,
    };

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

    add_sample_glue(&mut sections, &mut exports, &idx);
    if let Some(a_sample_channels) = a.sample_channels {
        add_sample_channels_glue(&mut sections, &mut exports, &idx, a_sample_channels);
    }
    add_bounds_glue(&mut sections, &mut exports, &idx);

    let data_count =
        (a_counts.has_data_count || b_counts.has_data_count).then(|| a_counts.data + b_counts.data);
    Ok(sections.finish(&exports, data_count))
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(1);
        if buf.is_empty() {
            CoilConfig::default()
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
    match build_coil_model(&input, &config) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("coil failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            r#"{ inner_radius: float .default 0.005, gap: float .default 0.001 }"#.to_string();
        OperatorMetadata {
            name: "coil_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Coil".to_string(),
            description: "Roll a flat model into an Archimedean spiral around the y axis."
                .to_string(),
            category: "Construction".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M12 12c0-1.7 2.5-1.7 2.5 0 0 2.5-5 2.5-5 0 0-4.2 7.5-4.2 7.5 0 0 5.8-10 5.8-10 0 0-7.5 12.5-7.5 12.5 0"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Sheet".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
