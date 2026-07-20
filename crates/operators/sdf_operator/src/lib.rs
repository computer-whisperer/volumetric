//! Bake a truncated signed distance field (TSDF) from a model's occupancy and
//! attach it as a declared custom sample channel.
//!
//! The source model's ordinary occupancy and advertised geometry bounds remain
//! exact and unchanged. During operator execution, occupancy is sampled on a
//! dense regular lattice covering the source bounds plus `band_width`. Exact
//! separable Euclidean distance transforms produce negative distances inside
//! and positive distances outside; magnitudes are clamped to the band.
//!
//! The resulting `signed_distance` channel (`Custom("volumetric.tsdf.v1")`)
//! is defined at every position, not merely inside the model or its advertised
//! bounds. Its baked lattice includes the entire band outside the source box,
//! and positions beyond that lattice return `+band_width`: because all source
//! geometry lies within its advertised bounds, the true outside distance there
//! is necessarily at least the positive clamp value.
//!
//! This operator intentionally emits exactly two channels—occupancy and the
//! TSDF. Existing source channels are not copied. The output shell owns a
//! dimension-independent IO buffer, so adding the analysis channel does not
//! consume the source model's channel-row capacity.

use ndfield_model_core::bake::{FieldGrid, bake_tsdf, sample_occupancy};

use model_merge_core::{
    MergeSections, OffsetReencoder, count_sections, find_function_export, find_memory_export,
    parse_model_exports,
};
use volumetric_abi::host::{
    cancelled, input_model_bounds, input_model_dimensions, input_model_sample, post_output,
    read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};
use wasm_encoder::{ExportKind, ExportSection, Function, Instruction, MemArg, ValType};

static TEMPLATE: &[u8] = include_bytes!("../template/sdf_model_template.wasm");

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct SdfConfig {
    /// Lattice cells along the source bounds' longest axis.
    resolution: i64,
    /// World-space truncation distance. Zero selects four nominal cells.
    band_width: f64,
}

impl Default for SdfConfig {
    fn default() -> Self {
        Self {
            resolution: 64,
            band_width: 0.0,
        }
    }
}

fn build_field_payload<F>(
    source_bounds: &[f64],
    config: &SdfConfig,
    mut sample: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(&[f64]) -> Result<Vec<f32>, String>,
{
    let grid = FieldGrid::plan(source_bounds, config.resolution, config.band_width)?;
    let occupancy = sample_occupancy(&grid, |positions| {
        Ok(sample(positions)?.into_iter().map(is_occupied).collect())
    })?;
    let values = bake_tsdf(&grid, &occupancy)?;
    ndfield_model_core::build_payload(&grid.counts, &grid.bounds, &values, grid.band_width as f32)
}

// ---------------------------------------------------------------------------
// Generated-model template patching and merge glue
// ---------------------------------------------------------------------------

fn const_i32_return(module: &walrus::Module, function: walrus::FunctionId) -> Option<i32> {
    let local = match &module.funcs.get(function).kind {
        walrus::FunctionKind::Local(local) => local,
        _ => return None,
    };
    let block = local.block(local.entry_block());
    match block.instrs.as_slice() {
        [(walrus::ir::Instr::Const(constant), _)] => match constant.value {
            walrus::ir::Value::I32(value) => Some(value),
            _ => None,
        },
        _ => None,
    }
}

fn take_slot_export(module: &mut walrus::Module, name: &str) -> Result<i32, String> {
    let export = module
        .exports
        .iter()
        .find(|export| export.name == name)
        .map(|export| (export.id(), export.item))
        .ok_or_else(|| format!("SDF template missing {name} export"))?;
    let address = match export.1 {
        walrus::ExportItem::Function(function) => const_i32_return(module, function)
            .ok_or_else(|| format!("SDF template {name} is not a constant function"))?,
        _ => return Err(format!("SDF template {name} is not a function")),
    };
    module.exports.delete(export.0);
    Ok(address)
}

fn patch_template(payload: &[u8]) -> Result<Vec<u8>, String> {
    let mut module =
        walrus::Module::from_buffer_with_config(TEMPLATE, &walrus::ModuleConfig::new())
            .map_err(|error| format!("failed to parse embedded SDF template: {error}"))?;
    let memory_id = module
        .exports
        .iter()
        .find(|export| export.name == "memory")
        .and_then(|export| match export.item {
            walrus::ExportItem::Memory(memory) => Some(memory),
            _ => None,
        })
        .ok_or("SDF template missing memory export")?;
    let payload_slot = take_slot_export(&mut module, "sdf_payload_slot")?;
    let base = {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65_536;
        memory.initial += (payload.len() as u64).div_ceil(65_536);
        if let Some(maximum) = memory.maximum {
            memory.maximum = Some(maximum.max(memory.initial));
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
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(payload_slot)),
        },
        (base as u32).to_le_bytes().to_vec(),
    );
    Ok(module.emit_wasm())
}

fn add_bounds_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    output_memory: u32,
    bounds: &[f64],
) {
    let function_type = sections.types.len();
    sections.types.ty().function([ValType::I32], []);
    sections.funcs.function(function_type);
    let mut function = Function::new([]);
    for (index, bound) in bounds.iter().enumerate() {
        function.instruction(&Instruction::LocalGet(0));
        function.instruction(&Instruction::F64Const((*bound).into()));
        function.instruction(&Instruction::F64Store(MemArg {
            offset: (index * 8) as u64,
            align: 3,
            memory_index: output_memory,
        }));
    }
    function.instruction(&Instruction::End);
    sections.code.function(&function);
    exports.export("get_bounds", ExportKind::Func, sections.funcs.len() - 1);
}

fn emit_copy_to_source(
    function: &mut Function,
    dimensions: usize,
    source_io: u32,
    source_memory: u32,
    output_memory: u32,
) {
    function.instruction(&Instruction::Call(source_io));
    function.instruction(&Instruction::LocalSet(1));
    for axis in 0..dimensions {
        function.instruction(&Instruction::LocalGet(1));
        function.instruction(&Instruction::LocalGet(0));
        function.instruction(&Instruction::F64Load(MemArg {
            offset: (axis * 8) as u64,
            align: 3,
            memory_index: output_memory,
        }));
        function.instruction(&Instruction::F64Store(MemArg {
            offset: (axis * 8) as u64,
            align: 3,
            memory_index: source_memory,
        }));
    }
}

fn add_sample_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    dimensions: usize,
    source_io: u32,
    source_sample: u32,
    source_memory: u32,
    output_memory: u32,
) {
    let function_type = sections.types.len();
    sections.types.ty().function([ValType::I32], [ValType::F32]);
    sections.funcs.function(function_type);
    // parameter 0: output-memory position pointer; local 1: source IO pointer.
    let mut function = Function::new([(1, ValType::I32)]);
    emit_copy_to_source(
        &mut function,
        dimensions,
        source_io,
        source_memory,
        output_memory,
    );
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::Call(source_sample));
    function.instruction(&Instruction::End);
    sections.code.function(&function);
    exports.export("sample", ExportKind::Func, sections.funcs.len() - 1);
}

#[allow(clippy::too_many_arguments)]
fn add_sample_channels_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    dimensions: usize,
    source_io: u32,
    source_sample: u32,
    sdf_sample: u32,
    source_memory: u32,
    output_memory: u32,
) {
    let function_type = sections.types.len();
    sections
        .types
        .ty()
        .function([ValType::I32, ValType::I32], []);
    sections.funcs.function(function_type);
    // params 0/1: position/output pointers in output memory; local 2 (index
    // 2): source IO pointer. emit_copy_to_source expects that local at 1, so
    // this function copies it into local 2 explicitly instead.
    let mut function = Function::new([(1, ValType::I32)]);
    function.instruction(&Instruction::Call(source_io));
    function.instruction(&Instruction::LocalSet(2));
    for axis in 0..dimensions {
        function.instruction(&Instruction::LocalGet(2));
        function.instruction(&Instruction::LocalGet(0));
        function.instruction(&Instruction::F64Load(MemArg {
            offset: (axis * 8) as u64,
            align: 3,
            memory_index: output_memory,
        }));
        function.instruction(&Instruction::F64Store(MemArg {
            offset: (axis * 8) as u64,
            align: 3,
            memory_index: source_memory,
        }));
    }
    // Channel 0: exact source occupancy.
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::LocalGet(2));
    function.instruction(&Instruction::Call(source_sample));
    function.instruction(&Instruction::F32Store(MemArg {
        offset: 0,
        align: 2,
        memory_index: output_memory,
    }));
    // Channel 1: baked truncated signed distance at the original position.
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::LocalGet(0));
    function.instruction(&Instruction::Call(sdf_sample));
    function.instruction(&Instruction::F32Store(MemArg {
        offset: 4,
        align: 2,
        memory_index: output_memory,
    }));
    function.instruction(&Instruction::End);
    sections.code.function(&function);
    exports.export(
        "sample_channels",
        ExportKind::Func,
        sections.funcs.len() - 1,
    );
}

fn merge_with_source(
    source: &[u8],
    patched_template: &[u8],
    dimensions: usize,
    source_bounds: &[f64],
) -> Result<Vec<u8>, String> {
    let source_counts = count_sections(source)?;
    let template_counts = count_sections(patched_template)?;
    let source_exports = parse_model_exports(source)?;
    let output_memory = find_memory_export(patched_template)? + source_counts.memories;
    let output_io = find_function_export(patched_template, "get_io_ptr")? + source_counts.funcs;
    let output_format =
        find_function_export(patched_template, "get_sample_format")? + source_counts.funcs;
    let sdf_sample = find_function_export(patched_template, "sdf_sample")? + source_counts.funcs;

    let mut sections = MergeSections::default();
    sections.append_module(source, &mut OffsetReencoder::identity())?;
    sections.append_module(
        patched_template,
        &mut OffsetReencoder::after(&source_counts),
    )?;

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, output_memory);
    exports.export(
        "get_dimensions",
        ExportKind::Func,
        source_exports.get_dimensions,
    );
    exports.export("get_io_ptr", ExportKind::Func, output_io);
    exports.export("get_sample_format", ExportKind::Func, output_format);
    add_bounds_glue(&mut sections, &mut exports, output_memory, source_bounds);
    add_sample_glue(
        &mut sections,
        &mut exports,
        dimensions,
        source_exports.get_io_ptr,
        source_exports.sample,
        source_exports.memory,
        output_memory,
    );
    add_sample_channels_glue(
        &mut sections,
        &mut exports,
        dimensions,
        source_exports.get_io_ptr,
        source_exports.sample,
        sdf_sample,
        source_exports.memory,
        output_memory,
    );

    let data_count = (source_counts.has_data_count || template_counts.has_data_count)
        .then(|| source_counts.data + template_counts.data);
    Ok(sections.finish(&exports, data_count))
}

fn build_output(source: &[u8], config: &SdfConfig) -> Result<Vec<u8>, String> {
    let dimensions = input_model_dimensions(0)
        .ok_or_else(|| "input 0 is not a usable model".to_string())? as usize;
    let source_bounds = input_model_bounds(0, dimensions)
        .ok_or_else(|| "failed to read input model bounds".to_string())?;
    let payload = build_field_payload(&source_bounds, config, |positions| {
        if cancelled() {
            return Err("SDF generation cancelled".to_string());
        }
        input_model_sample(0, positions, dimensions)
            .ok_or_else(|| "input model sampling failed".to_string())
    })?;
    let template = patch_template(&payload)?;
    merge_with_source(source, &template, dimensions, &source_bounds)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let source = read_input(0);
    let config: SdfConfig = {
        let bytes = read_input(1);
        if bytes.is_empty() {
            SdfConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(bytes)) {
                Ok(config) => config,
                Err(error) => {
                    report_error(&format!("invalid SDF configuration: {error}"));
                    return;
                }
            }
        }
    };
    match build_output(&source, &config) {
        Ok(output) => post_output(0, &output),
        Err(error) => report_error(&format!("SDF generation failed: {error}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        OperatorMetadata {
            name: "sdf_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Generate SDF".to_string(),
            description: "Bake globally defined occupancy + truncated signed-distance channels from model occupancy (replacing other extra channels).".to_string(),
            category: "Analysis".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M4 12c2.2-5 5-7 8-7s5.8 2 8 7c-2.2 5-5 7-8 7s-5.8-2-8-7Z"/>"##,
                r##"<path d="M8 12h8M12 8v8"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(
                    r#"{ resolution: int .ge 16 .le 256 .default 64, band_width: float .ge 0.0 .default 0.0 }"#
                        .to_string(),
                ),
            ],
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_tsdf_has_sign_distance_clamp_and_global_outside_value() {
        let bounds = [-1.0, 1.0, -0.5, 0.5];
        let config = SdfConfig {
            resolution: 32,
            band_width: 0.25,
        };
        let payload = build_field_payload(&bounds, &config, |positions| {
            Ok(positions
                .chunks_exact(2)
                .map(|position| if position[0] <= 0.0 { 1.0 } else { 0.0 })
                .collect())
        })
        .unwrap();
        let field = ndfield_model_core::PayloadView::new(&payload).unwrap();

        assert!(field.sample(&[-0.1, 0.0]) < -0.07);
        assert!(field.sample(&[0.1, 0.0]) > 0.06);
        assert!(field.sample(&[0.0, 0.0]).abs() < 0.04);
        assert!((field.sample(&[0.6, 0.0]) - 0.25).abs() < 1e-6);
        assert_eq!(field.sample(&[100.0, 0.0]), 0.25);
    }

    #[test]
    fn empty_model_is_positive_band_everywhere() {
        let bounds = [-1.0, 1.0];
        let config = SdfConfig {
            resolution: 16,
            band_width: 0.5,
        };
        let payload =
            build_field_payload(&bounds, &config, |positions| Ok(vec![0.0; positions.len()]))
                .unwrap();
        let field = ndfield_model_core::PayloadView::new(&payload).unwrap();
        assert_eq!(field.sample(&[0.0]), 0.5);
        assert_eq!(field.sample(&[20.0]), 0.5);
    }
}
