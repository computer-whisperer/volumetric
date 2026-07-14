//! Translate operator.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Generated Model ABI (N-dimensional):
//! - `get_dimensions() -> u32`: Passed through from input model
//! - `get_io_ptr() -> i32`: Passed through from input model
//! - `get_bounds(out_ptr: i32)`: Wrapper that adds translation to bounds
//! - `sample(pos_ptr: i32) -> f32`: Wrapper that subtracts translation from
//!   the position in place (the ABI allows clobbering the position buffer)
//! - `sample_channels(pos_ptr: i32, out_ptr: i32)`: Same wrapper as `sample`,
//!   present iff the input model has it
//! - `get_sample_format() -> i64`: Passed through from input model (a
//!   transform doesn't change what the samples mean)
//! - `memory`: Passed through from input model
//!
//! Dimension-adaptive: the wrapper reads the input's `get_dimensions`
//! constant and translates only the spatial prefix min(dims, 3) — a 2D
//! sketch uses dx/dy (dz is ignored), and higher dimensions pass through.
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1 (schema declared in metadata)
//! - Renames existing ABI functions with a hex suffix
//! - Emits new wrapper functions that apply configurable translation (dx/dy/dz)
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct TranslateConfig {
    dx: f32,
    dy: f32,
    dz: f32,
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }
}

use volumetric_abi::host::{post_output, read_input, report_error};

/// N-dimensional ABI function names. `get_dimensions`, `get_io_ptr` and
/// `get_sample_format` are passed through unchanged; `get_bounds`, `sample`
/// and `sample_channels` get wrappers.
const ABI_FUNCTIONS_ND: &[&str] = &[
    "get_dimensions",
    "get_io_ptr",
    "get_bounds",
    "sample",
    "sample_channels",
];

/// ABI functions the wrappers replace; the rest keep their exports.
const WRAPPED_FUNCTIONS: &[&str] = &["get_bounds", "sample", "sample_channels"];

/// Generate a random hex string for suffixing renamed functions
fn generate_hex_suffix() -> String {
    let mut state: u32 = 0xDEADBEEF;
    let mut result = String::with_capacity(8);
    for _ in 0..8 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let nibble = (state >> 16) & 0xF;
        result.push(char::from_digit(nibble, 16).unwrap());
    }
    result
}

/// Read the constant a trivial `() -> i32` function returns, if its body is
/// a single `i32.const`. Every model generator emits `get_dimensions` this
/// way, so this is how the operator adapts to the input's dimensionality
/// without being able to instantiate it.
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

/// Emit code that subtracts the translation from the first `spatial` dims in
/// place at `pos_ptr` (shared by the `sample` and `sample_channels`
/// wrappers). `spatial` is min(input dims, 3): the transform touches the
/// spatial prefix and passes higher dimensions through.
fn emit_position_rewrite(
    builder: &mut FunctionBuilder,
    pos_ptr: walrus::LocalId,
    memory_id: MemoryId,
    cfg: &TranslateConfig,
    spatial: usize,
) {
    // pos[i] -= d for each spatial axis
    for (offset, d) in [(0, cfg.dx), (8, cfg.dy), (16, cfg.dz)][..spatial]
        .iter()
        .copied()
    {
        let mem_arg = walrus::ir::MemArg { align: 3, offset };
        builder
            .func_body()
            .local_get(pos_ptr)
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
            .f64_const(-(d as f64))
            .binop(walrus::ir::BinaryOp::F64Add)
            .store(memory_id, walrus::ir::StoreKind::F64, mem_arg);
    }
}

/// Transform the input WASM module to apply translation by (dx, dy, dz).
fn transform_wasm(input_bytes: &[u8], cfg: TranslateConfig) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

    let suffix = generate_hex_suffix();

    // Find memory export
    let memory_id: Option<MemoryId> =
        module
            .exports
            .iter()
            .find(|e| e.name == "memory")
            .and_then(|e| {
                if let walrus::ExportItem::Memory(m) = e.item {
                    Some(m)
                } else {
                    None
                }
            });

    let memory_id = memory_id.ok_or("Input model missing memory export")?;

    // Find existing ABI functions via exports and collect their function IDs
    let mut renamed_functions: std::collections::HashMap<String, FunctionId> =
        std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();

    for export in module.exports.iter() {
        if ABI_FUNCTIONS_ND.contains(&export.name.as_str()) {
            if let walrus::ExportItem::Function(func_id) = export.item {
                renamed_functions.insert(export.name.clone(), func_id);
                exports_to_remove.push(export.id());
            }
        }
    }

    if !renamed_functions.contains_key("get_io_ptr") {
        return Err(
            "Input model missing `get_io_ptr` export; rebuild it against the \
                    current N-dimensional ABI"
                .to_string(),
        );
    }

    // Adapt to the input's dimensionality: only the spatial prefix
    // min(dims, 3) is translated. Writing all 3 axes into a 2D model would
    // corrupt memory past its 2*2-f64 IO buffer.
    let dims_func = renamed_functions
        .get("get_dimensions")
        .ok_or("Input model missing `get_dimensions` export")?;
    let dims = const_i32_return(&module, *dims_func).ok_or(
        "cannot determine input model dimensionality (get_dimensions is not a constant function)",
    )?;
    if dims < 1 {
        return Err(format!("input model reports invalid dimensionality {dims}"));
    }
    let spatial = (dims as usize).min(3);

    // Remove the exports we wrap; memory, get_dimensions and get_io_ptr pass through
    for export_id in exports_to_remove {
        let export_name = module.exports.get(export_id).name.clone();
        if WRAPPED_FUNCTIONS.contains(&export_name.as_str()) {
            module.exports.delete(export_id);
        }
    }

    // Rename the wrapped functions by adding suffix
    for (original_name, func_id) in &renamed_functions {
        if WRAPPED_FUNCTIONS.contains(&original_name.as_str()) {
            let new_name = format!("{}_{}", original_name, suffix);
            let func = module.funcs.get_mut(*func_id);
            func.name = Some(new_name);
        }
    }

    // Create wrapper for sample: subtract the translation from the first 3
    // dims in place at pos_ptr, then call the original with the same pointer.
    if let Some(&original_sample_id) = renamed_functions.get("sample") {
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);

        emit_position_rewrite(&mut builder, pos_ptr, memory_id, &cfg, spatial);
        builder
            .func_body()
            .local_get(pos_ptr)
            .call(original_sample_id);

        let wrapper_id = builder.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", wrapper_id);
    }

    // Create wrapper for sample_channels: identical position rewrite, then
    // call the original with both pointers. Only present when the input model
    // declares typed channels; get_sample_format passes through untouched.
    if let Some(&original_channels_id) = renamed_functions.get("sample_channels") {
        let mut builder =
            FunctionBuilder::new(&mut module.types, &[ValType::I32, ValType::I32], &[]);
        let pos_ptr = module.locals.add(ValType::I32);
        let out_ptr = module.locals.add(ValType::I32);

        emit_position_rewrite(&mut builder, pos_ptr, memory_id, &cfg, spatial);
        builder
            .func_body()
            .local_get(pos_ptr)
            .local_get(out_ptr)
            .call(original_channels_id);

        let wrapper_id = builder.finish(vec![pos_ptr, out_ptr], &mut module.funcs);
        module.exports.add("sample_channels", wrapper_id);
    }

    // Create wrapper for get_bounds
    if let Some(&original_bounds_id) = renamed_functions.get("get_bounds") {
        // get_bounds(out_ptr: i32)
        // Wrapper calls original, then adds translation to the returned bounds
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);

        let out_ptr = module.locals.add(ValType::I32);
        let tmp = module.locals.add(ValType::F64);

        // Call original get_bounds
        builder
            .func_body()
            .local_get(out_ptr)
            .call(original_bounds_id);

        // Add the axis delta to both interleaved min/max slots, spatial
        // prefix only — the input's bounds buffer holds exactly 2*dims f64s.
        for (axis, d) in [cfg.dx, cfg.dy, cfg.dz][..spatial].iter().enumerate() {
            for slot in 0..2 {
                let mem_arg = walrus::ir::MemArg {
                    align: 3,
                    offset: (axis * 16 + slot * 8) as u64,
                };
                builder
                    .func_body()
                    .local_get(out_ptr)
                    .load(memory_id, walrus::ir::LoadKind::F64, mem_arg)
                    .f64_const(*d as f64)
                    .binop(walrus::ir::BinaryOp::F64Add)
                    .local_set(tmp);
                builder.func_body().local_get(out_ptr).local_get(tmp).store(
                    memory_id,
                    walrus::ir::StoreKind::F64,
                    mem_arg,
                );
            }
        }

        let wrapper_id = builder.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", wrapper_id);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);

    let cfg = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            TranslateConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<TranslateConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    // Transform the WASM
    let output = match transform_wasm(&buf, cfg) {
        Ok(transformed) => transformed,
        Err(e) => {
            report_error(&format!("transform failed: {e}"));
            return;
        }
    };

    post_output(0, &output);
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ dx: float .default 0.0, dy: float .default 0.0, dz: float .default 0.0 }"
            .to_string();
        OperatorMetadata {
            name: "translate_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Translate".to_string(),
            description: "Move a model by a configurable (dx, dy, dz) offset.".to_string(),
            category: "Transforms".to_string(),
            icon_svg: String::new(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
