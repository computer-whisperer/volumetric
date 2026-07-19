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

const MIN_RESOLUTION: i64 = 16;
const MAX_RESOLUTION: i64 = 256;
const DEFAULT_BAND_CELLS: f64 = 4.0;
const MAX_GRID_POINTS: usize = 8_000_000;
const SAMPLE_BATCH_POINTS: usize = 8_192;

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

#[derive(Clone, Debug)]
struct FieldGrid {
    counts: Vec<usize>,
    bounds: Vec<f64>,
    spacing: Vec<f64>,
    band_width: f64,
    points: usize,
}

impl FieldGrid {
    fn plan(source_bounds: &[f64], config: &SdfConfig) -> Result<Self, String> {
        if source_bounds.is_empty() || !source_bounds.len().is_multiple_of(2) {
            return Err(format!("invalid source bounds {source_bounds:?}"));
        }
        let dimensions = source_bounds.len() / 2;
        if !(1..=ndfield_model_core::MAX_DIMS).contains(&dimensions) {
            return Err(format!(
                "SDF generation supports 1..={} dimensions; input has {dimensions}",
                ndfield_model_core::MAX_DIMS
            ));
        }
        if !(MIN_RESOLUTION..=MAX_RESOLUTION).contains(&config.resolution) {
            return Err(format!(
                "resolution must be in {MIN_RESOLUTION}..={MAX_RESOLUTION}, got {}",
                config.resolution
            ));
        }
        if !(config.band_width.is_finite() && config.band_width >= 0.0) {
            return Err(format!(
                "band_width must be finite and non-negative, got {}",
                config.band_width
            ));
        }

        let mut longest = 0.0f64;
        for axis in 0..dimensions {
            let (lo, hi) = (source_bounds[2 * axis], source_bounds[2 * axis + 1]);
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(format!(
                    "source axis {axis} needs finite nonempty bounds, got [{lo}, {hi}]"
                ));
            }
            longest = longest.max(hi - lo);
        }
        let nominal_cell = longest / config.resolution as f64;
        let band_width = if config.band_width == 0.0 {
            DEFAULT_BAND_CELLS * nominal_cell
        } else {
            config.band_width
        };
        if !(band_width.is_finite() && band_width > 0.0 && band_width <= f32::MAX as f64) {
            return Err(format!(
                "resolved band width must be finite, positive, and fit f32; got {band_width}"
            ));
        }

        let mut counts = Vec::with_capacity(dimensions);
        let mut bounds = Vec::with_capacity(2 * dimensions);
        let mut spacing = Vec::with_capacity(dimensions);
        let mut points = 1usize;
        for axis in 0..dimensions {
            let lo = source_bounds[2 * axis] - band_width;
            let hi = source_bounds[2 * axis + 1] + band_width;
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(format!(
                    "expanded field bounds overflow on axis {axis}: [{lo}, {hi}]"
                ));
            }
            let count = (((hi - lo) / nominal_cell).ceil() as usize)
                .checked_add(1)
                .ok_or_else(|| "field axis count overflows usize".to_string())?
                .max(2);
            points = points
                .checked_mul(count)
                .ok_or_else(|| "field lattice size overflows usize".to_string())?;
            if points > MAX_GRID_POINTS {
                return Err(format!(
                    "SDF lattice would contain {points} points (limit {MAX_GRID_POINTS}); \
                     lower resolution or band_width"
                ));
            }
            counts.push(count);
            bounds.extend([lo, hi]);
            spacing.push((hi - lo) / (count - 1) as f64);
        }
        Ok(Self {
            counts,
            bounds,
            spacing,
            band_width,
            points,
        })
    }

    fn dimensions(&self) -> usize {
        self.counts.len()
    }

    fn append_position(&self, mut index: usize, positions: &mut Vec<f64>) {
        for axis in 0..self.dimensions() {
            let coordinate = index % self.counts[axis];
            index /= self.counts[axis];
            positions.push(self.bounds[2 * axis] + coordinate as f64 * self.spacing[axis]);
        }
    }
}

/// Lower envelope of weighted parabolas:
/// `out[q] = min_p(input[p] + spacing² * (q-p)²)`.
fn distance_transform_1d(input: &[f64], spacing: f64, output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    let finite_sites: Vec<usize> = input
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect();
    if finite_sites.is_empty() {
        output.fill(f64::INFINITY);
        return;
    }

    let spacing2 = spacing * spacing;
    let mut sites = vec![0usize; finite_sites.len()];
    let mut boundaries = vec![0.0f64; finite_sites.len() + 1];
    let mut envelope = 0usize;
    sites[0] = finite_sites[0];
    boundaries[0] = f64::NEG_INFINITY;
    boundaries[1] = f64::INFINITY;

    for &candidate in &finite_sites[1..] {
        let intersection = |left: usize| {
            let candidate_x = candidate as f64;
            let left_x = left as f64;
            (input[candidate] + spacing2 * candidate_x * candidate_x
                - input[left]
                - spacing2 * left_x * left_x)
                / (2.0 * spacing2 * (candidate_x - left_x))
        };
        let mut crossing = intersection(sites[envelope]);
        while crossing <= boundaries[envelope] {
            if envelope == 0 {
                break;
            }
            envelope -= 1;
            crossing = intersection(sites[envelope]);
        }
        if envelope == 0 && crossing <= boundaries[0] {
            sites[0] = candidate;
            boundaries[1] = f64::INFINITY;
            continue;
        }
        envelope += 1;
        sites[envelope] = candidate;
        boundaries[envelope] = crossing;
        boundaries[envelope + 1] = f64::INFINITY;
    }

    let last = envelope;
    envelope = 0;
    for (query, value) in output.iter_mut().enumerate() {
        while envelope < last && boundaries[envelope + 1] < query as f64 {
            envelope += 1;
        }
        let site = sites[envelope];
        let delta = query as f64 - site as f64;
        *value = input[site] + spacing2 * delta * delta;
    }
}

/// Exact squared Euclidean distance, on the grid lattice, to every point
/// whose occupancy equals `target`.
fn squared_distance_transform(occupancy: &[bool], target: bool, grid: &FieldGrid) -> Vec<f64> {
    let mut distances: Vec<f64> = occupancy
        .iter()
        .map(|&state| if state == target { 0.0 } else { f64::INFINITY })
        .collect();

    for axis in 0..grid.dimensions() {
        let stride: usize = grid.counts[..axis].iter().product();
        let line_length = grid.counts[axis];
        let block_length = stride * line_length;
        let mut input = vec![0.0; line_length];
        let mut output = vec![0.0; line_length];
        for block in (0..grid.points).step_by(block_length) {
            for inner in 0..stride {
                for position in 0..line_length {
                    input[position] = distances[block + inner + position * stride];
                }
                distance_transform_1d(&input, grid.spacing[axis], &mut output);
                for position in 0..line_length {
                    distances[block + inner + position * stride] = output[position];
                }
            }
        }
    }
    distances
}

fn sample_occupancy<F>(grid: &FieldGrid, mut sample: F) -> Result<Vec<bool>, String>
where
    F: FnMut(&[f64]) -> Result<Vec<f32>, String>,
{
    let dimensions = grid.dimensions();
    let mut occupancy = Vec::with_capacity(grid.points);
    for start in (0..grid.points).step_by(SAMPLE_BATCH_POINTS) {
        let end = (start + SAMPLE_BATCH_POINTS).min(grid.points);
        let mut positions = Vec::with_capacity((end - start) * dimensions);
        for index in start..end {
            grid.append_position(index, &mut positions);
        }
        let samples = sample(&positions)?;
        if samples.len() != end - start {
            return Err(format!(
                "model sampler returned {} values for {} positions",
                samples.len(),
                end - start
            ));
        }
        occupancy.extend(samples.into_iter().map(is_occupied));
    }
    Ok(occupancy)
}

fn build_field_payload<F>(
    source_bounds: &[f64],
    config: &SdfConfig,
    sample: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(&[f64]) -> Result<Vec<f32>, String>,
{
    let grid = FieldGrid::plan(source_bounds, config)?;
    let occupancy = sample_occupancy(&grid, sample)?;
    if occupancy.iter().all(|state| *state) {
        return Err(
            "model remains occupied throughout its bounds plus the SDF band; advertised bounds may not enclose the geometry"
                .to_string(),
        );
    }

    // The interface lies between unlike lattice samples. Subtracting half
    // the smallest cell from the opposite-state lattice distance puts the
    // zero crossing midway across an axis-adjacent transition and bounds the
    // sub-cell surface error for oblique transitions.
    let interface_offset = 0.5 * grid.spacing.iter().copied().fold(f64::INFINITY, f64::min);
    let mut values = vec![0.0f32; grid.points];

    let to_outside = squared_distance_transform(&occupancy, false, &grid);
    for index in 0..grid.points {
        if occupancy[index] {
            let magnitude = (to_outside[index].sqrt() - interface_offset)
                .max(0.0)
                .min(grid.band_width);
            values[index] = -(magnitude as f32);
        }
    }
    drop(to_outside);

    let to_inside = squared_distance_transform(&occupancy, true, &grid);
    for index in 0..grid.points {
        if !occupancy[index] {
            let magnitude = if to_inside[index].is_finite() {
                (to_inside[index].sqrt() - interface_offset)
                    .max(0.0)
                    .min(grid.band_width)
            } else {
                grid.band_width
            };
            values[index] = magnitude as f32;
        }
    }

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
    fn weighted_distance_transform_is_exact_on_a_line() {
        let input = [f64::INFINITY, f64::INFINITY, 0.0, f64::INFINITY];
        let mut output = [0.0; 4];
        distance_transform_1d(&input, 0.25, &mut output);
        assert_eq!(output, [0.25, 0.0625, 0.0, 0.0625]);
    }

    #[test]
    fn weighted_transform_matches_brute_force_with_prior_axis_costs() {
        let input = [0.7, f64::INFINITY, 0.2, 1.3, f64::INFINITY, 0.0];
        let spacing = 0.37;
        let mut output = [0.0; 6];
        distance_transform_1d(&input, spacing, &mut output);
        for (query, &actual) in output.iter().enumerate() {
            let expected = input
                .iter()
                .enumerate()
                .map(|(site, &prior)| {
                    prior + spacing * spacing * (query as f64 - site as f64).powi(2)
                })
                .fold(f64::INFINITY, f64::min);
            assert!((actual - expected).abs() < 1e-12);
        }
    }

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

    #[test]
    fn planning_rejects_unbounded_work() {
        let bounds = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let config = SdfConfig {
            resolution: 256,
            band_width: 1.0,
        };
        let error = FieldGrid::plan(&bounds, &config).unwrap_err();
        assert!(error.contains("limit"), "{error}");
    }
}
