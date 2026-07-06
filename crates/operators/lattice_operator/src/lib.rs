//! Lattice operator: fills a 3D model with a density-modulated lattice.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Takes a 3D model (ideally with a declared density channel, e.g.
//! `fea_density_operator` output) and produces a model whose occupancy is
//! the input's occupancy intersected with an implicit lattice whose local
//! thickness follows the input's density: density 0 is empty, density 1 is
//! solid (see `lattice_model_core` for the families and mapping contract).
//!
//! Mechanism: the embedded `lattice_model_template` module (the evaluator,
//! with the lattice kind + cell size patched into its config slot) is
//! merged with the input model via `model_merge_core`; emitted glue
//! functions sample the input's channels per position and hand the density
//! to the evaluator.
//!
//! - Inputs with `sample_channels`: the density is read from the channel
//!   index in `density_channel` (default 1 — the first channel beyond
//!   occupancy, matching `fea_density_operator` output). The output keeps
//!   the input's `get_sample_format`/channel layout, with channel 0
//!   replaced by the lattice occupancy, so downstream inspection (slice
//!   lightbox, colormaps) still sees the density field.
//! - Occupancy-only inputs: every occupied point uses `uniform_density`,
//!   so any plain model can be lattice-filled for experimentation.
//! - `density_gamma` / `density_min` / `density_max` calibrate how the
//!   density value maps to the lattice thickness parameter (curve shape,
//!   thickness floor, thickness cap) before the family mapping — see
//!   `lattice_model_core::DensityMap`.
//!
//! Bounds, dimensions, and IO buffer pass through from the input.

use wasm_encoder::{ExportKind, ExportSection, Function, Instruction, MemArg, ValType};

use model_merge_core::{
    MergeSections, OffsetReencoder, count_sections, find_function_export, parse_model_exports,
};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Prebuilt `lattice_model_template` module (see that crate's docs for the
/// regeneration command).
static TEMPLATE: &[u8] = include_bytes!("../template/lattice_model_template.wasm");

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum LatticeKindConfig {
    Gyroid,
    Schwarz,
    Struts,
    /// Vertical hexagonal cell walls (compression along z, low surface
    /// shear — the classic seat-cushion honeycomb).
    Honeycomb,
    /// Tetrahedral strut lattice: thin uniform struts on the diamond-bond
    /// tiling (four struts per node), connected down to few-percent fills.
    Tetra,
    /// Foam lattice: struts on the Voronoi edge skeleton of a jittered
    /// BCC point set (Plateau geometry — pentagon/hexagon cell faces,
    /// four struts per node). `irregularity` 0 gives regular Kelvin
    /// cells; higher values organic non-repeating foam.
    Foam,
}

impl LatticeKindConfig {
    /// The `lattice_model_core::LatticeKind` discriminant the template's
    /// config slot encodes.
    fn as_u32(self) -> u32 {
        match self {
            Self::Gyroid => 0,
            Self::Schwarz => 1,
            Self::Struts => 2,
            Self::Honeycomb => 3,
            Self::Tetra => 4,
            Self::Foam => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct LatticeConfig {
    lattice: LatticeKindConfig,
    cell_size: f64,
    density_channel: i64,
    uniform_density: f64,
    /// Density-to-thickness calibration (see `lattice_model_core::DensityMap`):
    /// `d_eff = density_min + (density_max - density_min) * d^density_gamma`.
    density_gamma: f64,
    density_min: f64,
    density_max: f64,
    /// Foam cell-shape jitter, 0 (regular Kelvin cells) ..= 1 (fully
    /// organic). Other families ignore it.
    irregularity: f64,
}

impl Default for LatticeConfig {
    fn default() -> Self {
        Self {
            lattice: LatticeKindConfig::Gyroid,
            cell_size: 0.05,
            density_channel: 1,
            uniform_density: 1.0,
            density_gamma: 1.0,
            density_min: 0.0,
            density_max: 1.0,
            irregularity: 0.3,
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

/// Patch the lattice kind, cell size, density calibration, and shape
/// parameters into the template's config slot and drop the patch-helper
/// export.
fn patch_template(
    kind: u32,
    cell_size: f32,
    density_map: [f32; 3],
    irregularity: f32,
) -> Result<Vec<u8>, String> {
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
        .find(|e| e.name == "lattice_config_slot")
        .map(|e| (e.id(), e.item))
        .ok_or("template missing lattice_config_slot export")?;
    let slot = match export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(&module, f)
            .ok_or("template lattice_config_slot is not a constant function")?,
        _ => return Err("template lattice_config_slot is not a function".to_string()),
    };
    module.exports.delete(export.0);

    let mut bytes = Vec::with_capacity(24);
    bytes.extend(kind.to_le_bytes());
    bytes.extend(cell_size.to_le_bytes());
    for value in density_map {
        bytes.extend(value.to_le_bytes());
    }
    bytes.extend(irregularity.to_le_bytes());
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(slot)),
        },
        bytes,
    );

    Ok(module.emit_wasm())
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
            "lattice_operator needs a 3D input model, got {dims}D"
        )),
    }
}

/// Where the merged `sample` glue reads the input model's channel row:
/// the output half of the input's IO buffer (positions occupy the first
/// 3 f64s; the ABI reserves the second half for channel output).
const SCRATCH_OFFSET: i32 = 24;

/// The maximum channel index that stays inside the 3D IO buffer's output
/// half (6 f32s).
const MAX_CHANNEL_INDEX: i64 = 5;

struct GlueIndices {
    a_sample: u32,
    a_sample_channels: Option<u32>,
    a_get_io_ptr: u32,
    a_memory: u32,
    lattice_sample: u32,
}

/// Emit the merged `sample(pos_ptr) -> f32` glue: input occupancy gates the
/// lattice evaluation; the density comes from the input's channel row (or a
/// constant for occupancy-only inputs).
///
/// The position is saved to locals BEFORE calling the input model: the ABI
/// allows `sample` to clobber its position buffer (scale/rotation/translate
/// do, transforming in place), and the lattice must be evaluated in the
/// merged model's own coordinate space, not the input chain's.
#[allow(clippy::too_many_arguments)]
fn add_sample_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    idx: &GlueIndices,
    density_channel: u32,
    uniform_density: f32,
) {
    let ty = sections.types.len();
    sections.types.ty().function([ValType::I32], [ValType::F32]);
    sections.funcs.function(ty);

    let mem = MemArg {
        offset: 0,
        align: 2,
        memory_index: idx.a_memory,
    };
    let mem64 = |offset: u64| MemArg {
        offset,
        align: 3,
        memory_index: idx.a_memory,
    };

    // local 0: pos_ptr (param); local 1: scratch pointer; locals 2-4: the
    // position, saved before the input model can clobber it.
    let mut f = Function::new([(1, ValType::I32), (3, ValType::F64)]);
    for axis in 0..3u64 {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::F64Load(mem64(axis * 8)));
        f.instruction(&Instruction::LocalSet(2 + axis as u32));
    }
    match idx.a_sample_channels {
        Some(a_sample_channels) => {
            // scratch = A.get_io_ptr() + 24
            f.instruction(&Instruction::Call(idx.a_get_io_ptr));
            f.instruction(&Instruction::I32Const(SCRATCH_OFFSET));
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::LocalSet(1));
            // A.sample_channels(pos_ptr, scratch)
            f.instruction(&Instruction::LocalGet(0));
            f.instruction(&Instruction::LocalGet(1));
            f.instruction(&Instruction::Call(a_sample_channels));
            // occupied = channel 0 > 0.5
            f.instruction(&Instruction::LocalGet(1));
            f.instruction(&Instruction::F32Load(mem));
        }
        None => {
            // occupied = A.sample(pos_ptr) > 0.5
            f.instruction(&Instruction::LocalGet(0));
            f.instruction(&Instruction::Call(idx.a_sample));
        }
    }
    f.instruction(&Instruction::F32Const(0.5.into()));
    f.instruction(&Instruction::F32Gt);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Result(
        ValType::F32,
    )));
    // lattice_sample(x, y, z, density) with the saved position
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::LocalGet(3));
    f.instruction(&Instruction::LocalGet(4));
    match idx.a_sample_channels {
        Some(_) => {
            f.instruction(&Instruction::LocalGet(1));
            f.instruction(&Instruction::F32Load(MemArg {
                offset: u64::from(density_channel) * 4,
                align: 2,
                memory_index: idx.a_memory,
            }));
        }
        None => {
            f.instruction(&Instruction::F32Const(uniform_density.into()));
        }
    }
    f.instruction(&Instruction::Call(idx.lattice_sample));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::F32Const(0.0.into()));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);
    sections.code.function(&f);

    let func_index = sections.funcs.len() - 1;
    exports.export("sample", ExportKind::Func, func_index);
}

/// Emit the merged `sample_channels(pos_ptr, out_ptr)` glue: the input's
/// channel row with channel 0 replaced by the lattice occupancy.
///
/// As in [`add_sample_glue`], the position is saved to locals before the
/// input model runs (its `sample_channels` may clobber the position
/// buffer in place).
fn add_sample_channels_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    idx: &GlueIndices,
    a_sample_channels: u32,
    density_channel: u32,
) {
    let ty = sections.types.len();
    sections
        .types
        .ty()
        .function([ValType::I32, ValType::I32], []);
    sections.funcs.function(ty);

    let mem = MemArg {
        offset: 0,
        align: 2,
        memory_index: idx.a_memory,
    };
    let mem64 = |offset: u64| MemArg {
        offset,
        align: 3,
        memory_index: idx.a_memory,
    };

    // params: 0 pos_ptr, 1 out_ptr; locals 2-4: the saved position.
    let mut f = Function::new([(3, ValType::F64)]);
    for axis in 0..3u64 {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::F64Load(mem64(axis * 8)));
        f.instruction(&Instruction::LocalSet(2 + axis as u32));
    }
    // A.sample_channels(pos_ptr, out_ptr) fills the full row.
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::Call(a_sample_channels));
    // out[0] = occupied ? lattice_sample(x, y, z, out[density_channel]) : 0
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::F32Load(mem));
    f.instruction(&Instruction::F32Const(0.5.into()));
    f.instruction(&Instruction::F32Gt);
    f.instruction(&Instruction::If(wasm_encoder::BlockType::Result(
        ValType::F32,
    )));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::LocalGet(3));
    f.instruction(&Instruction::LocalGet(4));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::F32Load(MemArg {
        offset: u64::from(density_channel) * 4,
        align: 2,
        memory_index: idx.a_memory,
    }));
    f.instruction(&Instruction::Call(idx.lattice_sample));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::F32Const(0.0.into()));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::F32Store(mem));
    f.instruction(&Instruction::End);
    sections.code.function(&f);

    let func_index = sections.funcs.len() - 1;
    exports.export("sample_channels", ExportKind::Func, func_index);
}

fn build_lattice_model(input: &[u8], config: &LatticeConfig) -> Result<Vec<u8>, String> {
    if !(config.cell_size.is_finite() && config.cell_size > 0.0) {
        return Err(format!(
            "cell_size must be positive, got {}",
            config.cell_size
        ));
    }
    if !(0..=MAX_CHANNEL_INDEX).contains(&config.density_channel) {
        return Err(format!(
            "density_channel must be in 0..={MAX_CHANNEL_INDEX}, got {}",
            config.density_channel
        ));
    }
    if !(config.density_gamma.is_finite() && config.density_gamma > 0.0) {
        return Err(format!(
            "density_gamma must be positive, got {}",
            config.density_gamma
        ));
    }
    let range_valid = config.density_min.is_finite()
        && config.density_max.is_finite()
        && 0.0 <= config.density_min
        && config.density_min <= config.density_max
        && config.density_max <= 1.0;
    if !range_valid {
        return Err(format!(
            "density_min/density_max must satisfy 0 <= min <= max <= 1, got {} / {}",
            config.density_min, config.density_max
        ));
    }
    if !(config.irregularity.is_finite() && (0.0..=1.0).contains(&config.irregularity)) {
        return Err(format!(
            "irregularity must be in 0..=1, got {}",
            config.irregularity
        ));
    }
    check_three_dimensional(input)?;

    let template = patch_template(
        config.lattice.as_u32(),
        config.cell_size as f32,
        [
            config.density_gamma as f32,
            config.density_min as f32,
            config.density_max as f32,
        ],
        config.irregularity as f32,
    )?;

    let a_counts = count_sections(input)?;
    let b_counts = count_sections(&template)?;
    let a = parse_model_exports(input)?;
    let lattice_sample = find_function_export(&template, "lattice_sample")? + a_counts.funcs;

    let mut sections = MergeSections::default();
    sections.append_module(input, &mut OffsetReencoder::identity())?;
    sections.append_module(&template, &mut OffsetReencoder::after(&a_counts))?;

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, a.memory);
    exports.export("get_dimensions", ExportKind::Func, a.get_dimensions);
    exports.export("get_io_ptr", ExportKind::Func, a.get_io_ptr);
    exports.export("get_bounds", ExportKind::Func, a.get_bounds);
    if let Some(get_sample_format) = a.get_sample_format {
        exports.export("get_sample_format", ExportKind::Func, get_sample_format);
    }

    let idx = GlueIndices {
        a_sample: a.sample,
        a_sample_channels: a.sample_channels,
        a_get_io_ptr: a.get_io_ptr,
        a_memory: a.memory,
        lattice_sample,
    };
    let density_channel = config.density_channel as u32;
    add_sample_glue(
        &mut sections,
        &mut exports,
        &idx,
        density_channel,
        config.uniform_density as f32,
    );
    if let Some(a_sample_channels) = a.sample_channels {
        add_sample_channels_glue(
            &mut sections,
            &mut exports,
            &idx,
            a_sample_channels,
            density_channel,
        );
    }

    let data_count =
        (a_counts.has_data_count || b_counts.has_data_count).then(|| a_counts.data + b_counts.data);
    Ok(sections.finish(&exports, data_count))
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(1);
        if buf.is_empty() {
            LatticeConfig::default()
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
    match build_lattice_model(&input, &config) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&e),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ lattice: "gyroid" / "schwarz" / "struts" / "honeycomb" / "tetra" / "foam" .default "gyroid", cell_size: float .default 0.05, density_channel: int .default 1, uniform_density: float .default 1.0, density_gamma: float .default 1.0, density_min: float .default 0.0, density_max: float .default 1.0, irregularity: float .default 0.3 }"#.to_string();
        OperatorMetadata {
            name: "lattice_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Density model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
