//! Offset Operator: dilate or erode a model by a signed distance.
//!
//! Samples the input model's occupancy on a regular lattice (the same
//! bake as `sdf_operator`, shared via `ndfield_model_core::bake`),
//! derives exact lattice distances, and emits a standalone field-backed
//! model that is inside (canonical occupancy 1.0) wherever the point
//! lies within `distance` of the source solid (dilate, `distance > 0`)
//! or deeper than `|distance|` inside it (erode, `distance < 0`): the
//! baked field is `distance - tsdf(p)` and the generated `sample`
//! classifies its zero crossing. The truncation band is chosen internally as
//! `|distance|` plus a margin of nominal cells, so the requested offset
//! always lies well inside the defined band — there is no band/offset
//! mismatch to configure.
//!
//! The result is resolution-limited: surfaces are accurate to roughly a
//! lattice cell (longest source axis / `resolution`). Unlike
//! `sdf_operator`, the source model is not merged into the output; the
//! emitted model is a pure field evaluator (occupancy only, no
//! channels).
//!
//! Inputs:
//! - Input 0: ModelWASM — sampled via host imports; the bytes are unused
//! - Input 1: CBOR config:
//!   - `distance` (metres, default 5e-4): the offset; > 0 dilates,
//!     < 0 erodes
//!   - `resolution` (default 64): lattice cells along the longest
//!     source axis
//!
//! Output 0: ModelWASM (input dimensionality).
//!
//! The embedded template binary is `sdf_model_template` (the same file
//! `sdf_operator` embeds), regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p sdf_model_template
//! cp target/wasm32-unknown-unknown/release/sdf_model_template.wasm \
//!    crates/operators/offset_operator/template/
//! ```

use ndfield_model_core::bake::{DEFAULT_BAND_CELLS, FieldGrid, bake_tsdf, sample_occupancy};
use volumetric_abi::host::{
    cancelled, input_model_bounds, input_model_dimensions, input_model_sample, post_output,
    read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/sdf_model_template.wasm");

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
pub struct OffsetConfig {
    /// Signed offset in metres; positive dilates, negative erodes.
    pub distance: f64,
    /// Lattice cells along the source bounds' longest axis.
    pub resolution: i64,
}

impl Default for OffsetConfig {
    fn default() -> Self {
        Self {
            distance: 5e-4,
            resolution: 64,
        }
    }
}

/// Bake the offset field. Returns the `ndfield` payload plus the output
/// model's advertised bounds (source bounds padded by the dilation plus
/// one nominal cell of interpolation slack).
pub fn build_offset_payload<F>(
    source_bounds: &[f64],
    config: &OffsetConfig,
    sample: F,
) -> Result<(Vec<u8>, Vec<f64>), String>
where
    F: FnMut(&[f64]) -> Result<Vec<bool>, String>,
{
    if !config.distance.is_finite() {
        return Err(format!("distance must be finite, got {}", config.distance));
    }
    let distance = config.distance;
    let cell = ndfield_model_core::bake::nominal_cell(source_bounds, config.resolution)?;
    let band = distance.abs() + DEFAULT_BAND_CELLS * cell;
    let grid = FieldGrid::plan(source_bounds, config.resolution, band)?;
    let occupancy = sample_occupancy(&grid, sample)?;
    let tsdf = bake_tsdf(&grid, &occupancy)?;
    let values: Vec<f32> = tsdf
        .iter()
        .map(|&value| (distance - value as f64) as f32)
        .collect();
    // Far from the lattice the tsdf clamps at +band, and band exceeds
    // |distance| by the cell margin, so the outside value is strictly
    // negative (empty) — the field is globally defined.
    let outside = (distance - grid.band_width) as f32;
    let payload = ndfield_model_core::build_payload(&grid.counts, &grid.bounds, &values, outside)?;

    let pad = distance.max(0.0) + cell;
    let dimensions = source_bounds.len() / 2;
    let mut out_bounds = Vec::with_capacity(2 * dimensions);
    for axis in 0..dimensions {
        out_bounds.push(source_bounds[2 * axis] - pad);
        out_bounds.push(source_bounds[2 * axis + 1] + pad);
    }
    Ok((payload, out_bounds))
}

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
        .ok_or_else(|| format!("offset template missing {name} export"))?;
    let address = match export.1 {
        walrus::ExportItem::Function(function) => const_i32_return(module, function)
            .ok_or_else(|| format!("offset template {name} is not a constant function"))?,
        _ => return Err(format!("offset template {name} is not a function")),
    };
    module.exports.delete(export.0);
    Ok(address)
}

/// Patch the payload into the template and rewire its exports into a
/// complete standalone model: `sdf_sample` becomes `sample`, constant
/// `get_dimensions`/`get_bounds` are generated, and the channel-format
/// export is dropped (the model is occupancy-only).
pub fn emit_model(
    payload: &[u8],
    dimensions: usize,
    out_bounds: &[f64],
) -> Result<Vec<u8>, String> {
    if out_bounds.len() != 2 * dimensions {
        return Err(format!(
            "expected {} bounds values, got {}",
            2 * dimensions,
            out_bounds.len()
        ));
    }
    let mut module =
        walrus::Module::from_buffer_with_config(TEMPLATE, &walrus::ModuleConfig::new())
            .map_err(|error| format!("failed to parse embedded offset template: {error}"))?;
    let memory_id = module
        .exports
        .iter()
        .find(|export| export.name == "memory")
        .and_then(|export| match export.item {
            walrus::ExportItem::Memory(memory) => Some(memory),
            _ => None,
        })
        .ok_or("offset template missing memory export")?;

    // Payload into freshly reserved pages; base address into the slot.
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

    // sdf_sample has the model `sample` signature and reads its position
    // through the pointer argument, but it returns the raw interpolated
    // field; the ABI classifies occupancy against OCCUPANCY_THRESHOLD
    // (0.5), so `sample` wraps it to the canonical 1.0/0.0 at the
    // field's zero crossing. The template's occupancy+tsdf channel
    // declaration does not describe this model, so the format export
    // goes away entirely (exportless models default to occupancy-only).
    let mut field_function = None;
    let mut exports_to_delete = Vec::new();
    for export in module.exports.iter() {
        match export.name.as_str() {
            "sdf_sample" => {
                if let walrus::ExportItem::Function(function) = export.item {
                    field_function = Some(function);
                }
                exports_to_delete.push(export.id());
            }
            "get_sample_format" => exports_to_delete.push(export.id()),
            _ => {}
        }
    }
    for id in exports_to_delete {
        module.exports.delete(id);
    }
    let field_function = field_function.ok_or("offset template missing sdf_sample export")?;

    let mut sample_builder = walrus::FunctionBuilder::new(
        &mut module.types,
        &[walrus::ValType::I32],
        &[walrus::ValType::F32],
    );
    let pos_ptr = module.locals.add(walrus::ValType::I32);
    sample_builder
        .func_body()
        .local_get(pos_ptr)
        .call(field_function)
        .f32_const(0.0)
        .binop(walrus::ir::BinaryOp::F32Gt)
        .if_else(
            walrus::ValType::F32,
            |then| {
                then.f32_const(1.0);
            },
            |otherwise| {
                otherwise.f32_const(0.0);
            },
        );
    let sample_function = sample_builder.finish(vec![pos_ptr], &mut module.funcs);
    module.exports.add("sample", sample_function);

    let mut dims_builder =
        walrus::FunctionBuilder::new(&mut module.types, &[], &[walrus::ValType::I32]);
    dims_builder.func_body().i32_const(dimensions as i32);
    let dims_function = dims_builder.finish(Vec::new(), &mut module.funcs);
    module.exports.add("get_dimensions", dims_function);

    let mut bounds_builder =
        walrus::FunctionBuilder::new(&mut module.types, &[walrus::ValType::I32], &[]);
    let out_ptr = module.locals.add(walrus::ValType::I32);
    for (index, &bound) in out_bounds.iter().enumerate() {
        bounds_builder
            .func_body()
            .local_get(out_ptr)
            .f64_const(bound)
            .store(
                memory_id,
                walrus::ir::StoreKind::F64,
                walrus::ir::MemArg {
                    align: 3,
                    offset: (index * 8) as u64,
                },
            );
    }
    let bounds_function = bounds_builder.finish(vec![out_ptr], &mut module.funcs);
    module.exports.add("get_bounds", bounds_function);

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config: OffsetConfig = {
        let bytes = read_input(1);
        if bytes.is_empty() {
            OffsetConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(bytes)) {
                Ok(config) => config,
                Err(error) => {
                    report_error(&format!("invalid offset configuration: {error}"));
                    return;
                }
            }
        }
    };
    let result = (|| {
        let dimensions = input_model_dimensions(0)
            .ok_or_else(|| "input 0 is not a usable model".to_string())?
            as usize;
        let source_bounds = input_model_bounds(0, dimensions)
            .ok_or_else(|| "failed to read input model bounds".to_string())?;
        let (payload, out_bounds) = build_offset_payload(&source_bounds, &config, |positions| {
            if cancelled() {
                return Err("offset cancelled".to_string());
            }
            input_model_sample(0, positions, dimensions)
                .map(|values| values.into_iter().map(is_occupied).collect())
                .ok_or_else(|| "input model sampling failed".to_string())
        })?;
        emit_model(&payload, dimensions, &out_bounds)
    })();
    match result {
        Ok(output) => post_output(0, &output),
        Err(error) => report_error(&format!("offset failed: {error}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "offset_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Offset".to_string(),
        description: "Dilate or erode a model by a signed distance via a baked distance field."
            .to_string(),
        category: "Transforms".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<rect x="3" y="3" width="18" height="18" rx="4"/>"##,
            r##"<rect x="8" y="8" width="8" height="8" rx="2"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::CBORConfiguration(
                r#"{ distance: float .default 5e-4, resolution: int .ge 16 .le 256 .default 64 }"#
                    .to_string(),
            ),
        ],
        input_names: vec!["Model".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndfield_model_core::PayloadView;

    /// Occupancy sampler for a sphere of radius `r` about the origin.
    fn sphere_sampler(r: f64) -> impl FnMut(&[f64]) -> Result<Vec<bool>, String> {
        move |positions: &[f64]| {
            Ok(positions
                .chunks_exact(3)
                .map(|p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= r * r)
                .collect())
        }
    }

    #[test]
    fn dilated_and_eroded_sphere_match_analytic() {
        let r = 0.010;
        let bounds = [-0.012, 0.012, -0.012, 0.012, -0.012, 0.012];
        let resolution = 64;
        let cell = 0.024 / resolution as f64;
        // Probe along an axis and a diagonal, at radii straddling the
        // offset surface by 1.5 cells.
        let directions = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.577_350, 0.577_350, 0.577_350],
        ];
        for distance in [0.003, -0.003] {
            let config = OffsetConfig {
                distance,
                resolution,
            };
            let (payload, out_bounds) =
                build_offset_payload(&bounds, &config, sphere_sampler(r)).unwrap();
            let field = PayloadView::new(&payload).unwrap();
            let surface = r + distance;
            for direction in directions {
                for (radius, expect_inside) in [
                    (surface - 1.5 * cell, true),
                    (surface + 1.5 * cell, false),
                    (0.0, true),
                ] {
                    let p = [
                        direction[0] * radius,
                        direction[1] * radius,
                        direction[2] * radius,
                    ];
                    let value = field.sample(&p);
                    assert_eq!(
                        value > 0.0,
                        expect_inside,
                        "d={distance} dir={direction:?} radius={radius}: value {value}"
                    );
                }
            }
            // The field is empty far away (globally defined).
            assert!(field.sample(&[1.0, 1.0, 1.0]) < 0.0);
            // Bounds pad: dilation plus one cell of slack.
            let pad = distance.max(0.0) + cell;
            assert!((out_bounds[0] - (bounds[0] - pad)).abs() < 1e-12);
            assert!((out_bounds[5] - (bounds[5] + pad)).abs() < 1e-12);
        }
    }

    #[test]
    fn template_surgery_emits_the_model_abi() {
        let config = OffsetConfig {
            distance: 0.002,
            resolution: 16,
        };
        let bounds = [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01];
        let (payload, out_bounds) =
            build_offset_payload(&bounds, &config, sphere_sampler(0.008)).unwrap();
        let wasm = emit_model(&payload, 3, &out_bounds).unwrap();
        let module = walrus::Module::from_buffer(&wasm).expect("emitted wasm parses");
        let names: Vec<&str> = module.exports.iter().map(|e| e.name.as_str()).collect();
        for required in [
            "sample",
            "get_bounds",
            "get_dimensions",
            "get_io_ptr",
            "memory",
        ] {
            assert!(names.contains(&required), "missing export {required}");
        }
        for dropped in ["sdf_payload_slot", "sdf_sample", "get_sample_format"] {
            assert!(!names.contains(&dropped), "stale export {dropped}");
        }
    }
}
