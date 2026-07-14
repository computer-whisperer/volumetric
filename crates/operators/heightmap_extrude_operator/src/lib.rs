//! Heightmap extrude operator: turns a 2D scalar-field model into a 3D
//! solid whose top surface follows the field.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! The input must be a 2D model (`get_dimensions() -> 2`) — an image
//! loaded by `image_model_operator`, a mesh height query, a Lua sketch —
//! whose sample value is read as a height:
//!
//! ```text
//! surface(x, y) = field(x, y) * scale + offset
//! sample(x, y, z) = (surface > clip && 0 <= z <= surface && z <= z_max)
//!                   ? 1.0 : 0.0
//! ```
//!
//! `scale` may be negative (e.g. -1 turns a downward relief — a mesh
//! height query's negative depths — into positive extrusion heights) and
//! `offset` shifts absolute-height fields into range. `clip` is in output
//! z units: surfaces at or below it produce no geometry, so the default
//! 0.0 drops the background of a normalized image and negative-scale
//! fields aren't accidentally clipped away.
//!
//! `z_max` is `max_height` when set, else `scale + offset` when that is
//! positive (the natural maximum of a normalized [0, 1] field); when
//! neither is positive the field's peak is unknowable statically and the
//! operator errors, asking for an explicit `max_height`. `z_max` is also
//! the advertised z bound, so geometry demanded above it is cut off
//! rather than left outside the bounds where meshing would miss it.
//!
//! Generated Model ABI (same wrapping scheme as `extrude_operator`):
//! - `get_dimensions() -> u32`: Returns 3
//! - `get_io_ptr() -> i32`: A fresh buffer in a newly reserved memory page
//! - `get_bounds(out_ptr: i32)`: Field x/y bounds, then z in [0, z_max]
//! - `sample(pos_ptr: i32) -> f32`: reads z, calls the field's sample for
//!   the height at (x, y), gates as above
//! - `memory`: Passed through from input model
//!
//! Typed sample channels are dropped: `get_sample_format` and
//! `sample_channels` exports are removed, so the output has the implicit
//! occupancy-only format.
//!
//! Behavior:
//! - Reads WASM model bytes from input 0
//! - Reads CBOR configuration from input 1 (scale, offset, max_height, clip)
//! - Rejects non-2D input models
//! - Outputs the modified WASM to output 0

use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct HeightmapExtrudeConfig {
    /// Multiplier from sample value to surface height; negative flips a
    /// downward relief into positive heights.
    scale: f64,
    /// Added to `field * scale`; shifts absolute-height fields into range.
    offset: f64,
    /// The advertised z bound and hard height cutoff; 0 = `scale + offset`
    /// (a normalized [0, 1] field's natural maximum) when that is positive.
    max_height: f64,
    /// Surfaces at or below this height (in output z units) produce no
    /// geometry.
    clip: f64,
}

impl Default for HeightmapExtrudeConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            offset: 0.0,
            max_height: 0.0,
            clip: 0.0,
        }
    }
}

use volumetric_abi::host::{post_output, read_input, report_error};

/// ABI exports the wrapper replaces. `sample_channels` and
/// `get_sample_format` are also removed (channels drop); `memory` passes
/// through.
const WRAPPED_FUNCTIONS: &[&str] = &["get_dimensions", "get_io_ptr", "get_bounds", "sample"];

/// Channel exports removed outright: the extruded model is occupancy-only.
const DROPPED_FUNCTIONS: &[&str] = &["get_sample_format", "sample_channels"];

/// Read the constant a trivial `() -> i32` function returns, if its body is
/// a single `i32.const`. Every model generator emits `get_dimensions` this
/// way, so this is how the operator checks the input is 2D without being
/// able to instantiate it.
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

/// Transform the input 2D field WASM into its heightmap extrusion.
fn transform_wasm(input_bytes: &[u8], cfg: HeightmapExtrudeConfig) -> Result<Vec<u8>, String> {
    if !(cfg.scale != 0.0 && cfg.scale.is_finite()) {
        return Err(format!(
            "extrude scale must be nonzero and finite, got {}",
            cfg.scale
        ));
    }
    if !cfg.offset.is_finite() {
        return Err(format!("offset must be finite, got {}", cfg.offset));
    }
    if !(cfg.max_height >= 0.0 && cfg.max_height.is_finite()) {
        return Err(format!(
            "max_height must be > 0 (or 0 for the scale + offset default), got {}",
            cfg.max_height
        ));
    }
    if !cfg.clip.is_finite() {
        return Err(format!("clip must be finite, got {}", cfg.clip));
    }
    let z_max = if cfg.max_height > 0.0 {
        cfg.max_height
    } else if cfg.scale + cfg.offset > 0.0 {
        cfg.scale + cfg.offset
    } else {
        return Err(format!(
            "the default height bound scale + offset = {} isn't positive; set max_height \
             explicitly (the field's peak surface height)",
            cfg.scale + cfg.offset
        ));
    };

    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(input_bytes, &config)
        .map_err(|e| format!("Failed to parse WASM: {}", e))?;

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

    // Collect the ABI function exports we wrap or drop
    let mut originals: std::collections::HashMap<String, FunctionId> =
        std::collections::HashMap::new();
    let mut exports_to_remove = Vec::new();
    for export in module.exports.iter() {
        let name = export.name.as_str();
        if WRAPPED_FUNCTIONS.contains(&name) || DROPPED_FUNCTIONS.contains(&name) {
            if let walrus::ExportItem::Function(func_id) = export.item {
                originals.insert(export.name.clone(), func_id);
                exports_to_remove.push(export.id());
            }
        }
    }

    for required in WRAPPED_FUNCTIONS {
        if !originals.contains_key(*required) {
            return Err(format!("Input model missing `{required}` export"));
        }
    }

    // The input's sample value is read as a height, which only makes sense
    // for a 2D field.
    match const_i32_return(&module, originals["get_dimensions"]) {
        Some(2) => {}
        Some(n) => {
            return Err(format!(
                "heightmap extrude input must be a 2D model, got a {n}-dimensional model"
            ));
        }
        None => {
            return Err(
                "cannot determine input model dimensionality (get_dimensions is not a \
                 constant function)"
                    .to_string(),
            );
        }
    }

    for export_id in exports_to_remove {
        module.exports.delete(export_id);
    }

    // The field's IO buffer holds 2*2 f64s; a 3D model must offer 2*3.
    // Reserve a fresh page and put the new buffer at its start.
    let new_io_ptr = {
        let memory = module.memories.get_mut(memory_id);
        let io_ptr = (memory.initial * 65536) as i32;
        memory.initial += 1;
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
        io_ptr
    };

    // get_dimensions() -> 3
    {
        let mut b = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        b.func_body().i32_const(3);
        let fid = b.finish(vec![], &mut module.funcs);
        module.exports.add("get_dimensions", fid);
    }

    // get_io_ptr() -> the new 2*3-f64 buffer
    {
        let mut b = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        b.func_body().i32_const(new_io_ptr);
        let fid = b.finish(vec![], &mut module.funcs);
        module.exports.add("get_io_ptr", fid);
    }

    // sample(pos_ptr): read z first (the field's sample may scribble on
    // shared memory), get the height from the field at (x, y) — the first
    // two f64s, read via the same pointer — compute the surface height,
    // then gate. All comparisons are false on a NaN surface, so garbage
    // reads as outside.
    {
        let orig_sample = originals["sample"];
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let z = module.locals.add(ValType::F64);
        let surface = module.locals.add(ValType::F64);

        use walrus::ir::BinaryOp::{F64Add, F64Ge, F64Gt, F64Le, F64Mul, I32And};
        let z_arg = walrus::ir::MemArg {
            align: 3,
            offset: 16,
        };

        b.func_body()
            .local_get(pos_ptr)
            .load(memory_id, walrus::ir::LoadKind::F64, z_arg)
            .local_set(z)
            // surface = field * scale + offset
            .local_get(pos_ptr)
            .call(orig_sample)
            .unop(walrus::ir::UnaryOp::F64PromoteF32)
            .f64_const(cfg.scale)
            .binop(F64Mul)
            .f64_const(cfg.offset)
            .binop(F64Add)
            .local_set(surface)
            // surface > clip
            .local_get(surface)
            .f64_const(cfg.clip)
            .binop(F64Gt)
            // && z >= 0
            .local_get(z)
            .f64_const(0.0)
            .binop(F64Ge)
            .binop(I32And)
            // && z <= surface
            .local_get(z)
            .local_get(surface)
            .binop(F64Le)
            .binop(I32And)
            // && z <= z_max (keep the solid inside its advertised bounds)
            .local_get(z)
            .f64_const(z_max)
            .binop(F64Le)
            .binop(I32And)
            .if_else(
                ValType::F32,
                |then| {
                    then.f32_const(1.0);
                },
                |else_| {
                    else_.f32_const(0.0);
                },
            );

        let fid = b.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", fid);
    }

    // get_bounds(out_ptr): field bounds fill offsets 0..32 (x/y), then
    // write z bounds [0, z_max] at offsets 32/40.
    {
        let orig_bounds = originals["get_bounds"];
        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);

        let min_z_arg = walrus::ir::MemArg {
            align: 3,
            offset: 32,
        };
        let max_z_arg = walrus::ir::MemArg {
            align: 3,
            offset: 40,
        };

        b.func_body()
            .local_get(out_ptr)
            .call(orig_bounds)
            .local_get(out_ptr)
            .f64_const(0.0)
            .store(memory_id, walrus::ir::StoreKind::F64, min_z_arg)
            .local_get(out_ptr)
            .f64_const(z_max)
            .store(memory_id, walrus::ir::StoreKind::F64, max_z_arg);

        let fid = b.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", fid);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);

    let cfg = {
        let cfg_buf = read_input(1);
        if cfg_buf.is_empty() {
            HeightmapExtrudeConfig::default()
        } else {
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            match ciborium::de::from_reader::<HeightmapExtrudeConfig, _>(&mut cursor) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    let output = match transform_wasm(&buf, cfg) {
        Ok(transformed) => transformed,
        Err(e) => {
            report_error(&format!("heightmap extrude failed: {e}"));
            return;
        }
    };

    post_output(0, &output);
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = "{ scale: float .default 1.0, offset: float .default 0.0, \
                      max_height: float .default 0.0, clip: float .default 0.0 }"
            .to_string();
        OperatorMetadata {
            name: "heightmap_extrude_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Heightmap Extrude".to_string(),
            description:
                "Turn a 2D scalar field into a 3D solid whose top surface follows the field."
                    .to_string(),
            category: "Construction".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 20h18"/>"##,
                r##"<path d="m4 20 5-10 3 5 3-7 5 12"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Height field (2D)".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}
