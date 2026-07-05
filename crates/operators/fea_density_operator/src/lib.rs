//! FEA Density Operator.
//!
//! Closes the FEA loop back to geometry: turns a solved mesh's per-element
//! `strain_energy_density` field into a variable-density model — the first
//! producer of the `ChannelKind::Density` sample channel.
//!
//! The output is the input *model* (unchanged geometry: `sample`,
//! `get_bounds`, `get_dimensions`, `get_io_ptr` all pass through) with the
//! density grid embedded as a WASM data segment in freshly reserved memory
//! pages, plus:
//! - `get_sample_format()` declaring `[Occupancy, Density]`
//! - `sample_channels(pos_ptr, out_ptr)` writing the original occupancy and
//!   a piecewise-constant per-cell density looked up from the grid
//!   (positions clamp to the grid, so the skin of the solid never reads a
//!   zero cell; per the ABI, density is only meaningful where occupancy
//!   says inside). Any existing channel declaration on the input model is
//!   replaced.
//!
//! Per-element energies map linearly onto `[min_density, max_density]`,
//! normalized by the peak energy (a zero-energy solve maps everything to
//! `min_density`). Grid cells not covered by any element read 0.
//!
//! Inputs:
//! - Input 0: FeaMesh — solved (must carry `strain_energy_density`)
//! - Input 1: ModelWASM — the geometry to carry the channel (must be 3D;
//!   typically the same model the mesh was gridded from)
//! - Input 2: CBOR configuration:
//!   `{ min_density: float .default 0.2, max_density: float .default 1.0 }`
//!
//! Output 0: ModelWASM with the two-channel sample format.

use volumetric_abi::fea::{FeaMesh, decode_fea_mesh};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{
    ChannelKind, OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, SampleChannel,
    SampleFormat, encode_sample_format,
};
use walrus::{FunctionBuilder, FunctionId, MemoryId, Module, ModuleConfig, ValType};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct DensityConfig {
    min_density: f64,
    max_density: f64,
}

impl Default for DensityConfig {
    fn default() -> Self {
        Self {
            min_density: 0.2,
            max_density: 1.0,
        }
    }
}

/// The density grid reconstructed from a solved mesh: cell values on the
/// mesh's uniform lattice.
struct DensityGrid {
    origin: [f64; 3],
    h: f64,
    dims: [usize; 3],
    /// f32 per cell, x-major then y then z (`(k * ny + j) * nx + i`).
    values: Vec<f32>,
}

fn build_density_grid(mesh: &FeaMesh, config: &DensityConfig) -> Result<DensityGrid, String> {
    if !(config.min_density.is_finite()
        && config.max_density.is_finite()
        && (0.0..=1.0).contains(&config.min_density)
        && (0.0..=1.0).contains(&config.max_density)
        && config.min_density <= config.max_density)
    {
        return Err(format!(
            "density range [{}, {}] must satisfy 0 <= min <= max <= 1",
            config.min_density, config.max_density
        ));
    }

    let energy = mesh
        .element_fields
        .iter()
        .find(|f| f.name == "strain_energy_density" && f.components == 1)
        .ok_or_else(|| {
            "mesh has no strain_energy_density field; run fea_solve_operator first".to_string()
        })?;

    let h = fea_core::detect_uniform_grid(mesh)?;

    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for node in 0..mesh.node_count() {
        let p = mesh.node_position(node);
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    let dims: [usize; 3] = std::array::from_fn(|a| ((hi[a] - lo[a]) / h).round().max(1.0) as usize);
    let cell_count = dims[0] * dims[1] * dims[2];
    // A guard against absurd grids (the mesher caps resolution at 128).
    if cell_count > 64 << 20 {
        return Err(format!("density grid of {cell_count} cells is too large"));
    }

    let peak = energy.data.iter().copied().fold(0.0f64, f64::max);
    let span = config.max_density - config.min_density;

    let mut values = vec![0.0f32; cell_count];
    for e in 0..mesh.element_count() {
        let base = mesh.node_position(mesh.element(e)[0] as usize);
        let cell: [usize; 3] = std::array::from_fn(|a| ((base[a] - lo[a]) / h).round() as usize);
        if cell.iter().zip(&dims).any(|(c, d)| c >= d) {
            return Err(format!("element {e} lies outside the reconstructed grid"));
        }
        let normalized = if peak > 0.0 {
            (energy.data[e] / peak).clamp(0.0, 1.0)
        } else {
            0.0
        };
        values[(cell[2] * dims[1] + cell[1]) * dims[0] + cell[0]] =
            (config.min_density + span * normalized) as f32;
    }

    Ok(DensityGrid {
        origin: lo,
        h,
        dims,
        values,
    })
}

/// Read the constant a trivial `() -> i32` function returns (how every model
/// generator emits `get_dimensions`).
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

/// Wrap the model with the density channel. See the module docs.
fn attach_density_channel(model_bytes: &[u8], grid: &DensityGrid) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(model_bytes, &config)
        .map_err(|e| format!("failed to parse model WASM: {e}"))?;

    let memory_id: MemoryId = module
        .exports
        .iter()
        .find(|e| e.name == "memory")
        .and_then(|e| match e.item {
            walrus::ExportItem::Memory(m) => Some(m),
            _ => None,
        })
        .ok_or("input model missing memory export")?;

    let mut sample_id = None;
    let mut dimensions_id = None;
    let mut replaced_exports = Vec::new();
    for export in module.exports.iter() {
        match export.name.as_str() {
            "sample" => {
                if let walrus::ExportItem::Function(f) = export.item {
                    sample_id = Some(f);
                }
            }
            "get_dimensions" => {
                if let walrus::ExportItem::Function(f) = export.item {
                    dimensions_id = Some(f);
                }
            }
            // Any prior channel declaration is replaced by ours.
            "get_sample_format" | "sample_channels" => replaced_exports.push(export.id()),
            _ => {}
        }
    }
    let sample_id = sample_id.ok_or("input model missing `sample` export")?;
    let dimensions_id = dimensions_id.ok_or("input model missing `get_dimensions` export")?;
    match const_i32_return(&module, dimensions_id) {
        Some(3) => {}
        Some(n) => return Err(format!("the model must be 3D, got {n} dimensions")),
        None => {
            return Err(
                "cannot determine model dimensionality (get_dimensions is not constant)"
                    .to_string(),
            );
        }
    }
    for export in replaced_exports {
        module.exports.delete(export);
    }

    // Reserve fresh pages past the model's memory for [format CBOR | grid].
    let format_bytes = encode_sample_format(&SampleFormat {
        channels: vec![
            SampleChannel {
                name: "occupancy".to_string(),
                kind: ChannelKind::Occupancy,
            },
            SampleChannel {
                name: "density".to_string(),
                kind: ChannelKind::Density,
            },
        ],
    });
    let grid_bytes: Vec<u8> = grid.values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let format_ptr;
    let grid_ptr;
    {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65536;
        format_ptr = base as u32;
        grid_ptr = (base as u32) + format_bytes.len().next_multiple_of(8) as u32;
        let total = grid_ptr as u64 + grid_bytes.len() as u64 - base;
        memory.initial += total.div_ceil(65536);
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
    }
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(format_ptr as i32)),
        },
        format_bytes.clone(),
    );
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(grid_ptr as i32)),
        },
        grid_bytes,
    );

    // get_sample_format() -> ptr | (len << 32)
    {
        let packed = (format_ptr as i64) | ((format_bytes.len() as i64) << 32);
        let mut b = FunctionBuilder::new(&mut module.types, &[], &[ValType::I64]);
        b.func_body().i64_const(packed);
        let fid = b.finish(vec![], &mut module.funcs);
        module.exports.add("get_sample_format", fid);
    }

    // sample_channels(pos_ptr, out_ptr): occupancy via the original sample,
    // density via a clamped nearest-cell grid lookup.
    {
        use walrus::ir::BinaryOp::{F64Div, F64Sub, I32Add, I32GtS, I32LtS, I32Mul};
        use walrus::ir::UnaryOp::I32TruncSSatF64;
        use walrus::ir::{LoadKind, MemArg, StoreKind};

        let mut b = FunctionBuilder::new(&mut module.types, &[ValType::I32, ValType::I32], &[]);
        let pos_ptr = module.locals.add(ValType::I32);
        let out_ptr = module.locals.add(ValType::I32);
        let coords = [
            module.locals.add(ValType::F64),
            module.locals.add(ValType::F64),
            module.locals.add(ValType::F64),
        ];
        let cells = [
            module.locals.add(ValType::I32),
            module.locals.add(ValType::I32),
            module.locals.add(ValType::I32),
        ];

        let mut body = b.func_body();

        // Read the position first: the model may clobber its IO region
        // during `sample`.
        for (axis, coord) in coords.iter().enumerate() {
            body.local_get(pos_ptr)
                .load(
                    memory_id,
                    LoadKind::F64,
                    MemArg {
                        align: 3,
                        offset: (axis * 8) as u64,
                    },
                )
                .local_set(*coord);
        }

        // out[0] = occupancy
        body.local_get(out_ptr)
            .local_get(pos_ptr)
            .call(sample_id)
            .store(
                memory_id,
                StoreKind::F32,
                MemArg {
                    align: 2,
                    offset: 0,
                },
            );

        // Per-axis cell index: trunc_sat((coord - origin) / h) clamped to
        // [0, dim-1]. trunc_sat turns NaN into 0 instead of trapping.
        for axis in 0..3 {
            let max_cell = grid.dims[axis] as i32 - 1;
            body.local_get(coords[axis])
                .f64_const(grid.origin[axis])
                .binop(F64Sub)
                .f64_const(grid.h)
                .binop(F64Div)
                .unop(I32TruncSSatF64)
                .local_set(cells[axis])
                // max(cell, 0)
                .i32_const(0)
                .local_get(cells[axis])
                .local_get(cells[axis])
                .i32_const(0)
                .binop(I32LtS)
                .select(Some(ValType::I32))
                .local_set(cells[axis])
                // min(cell, dim-1)
                .i32_const(max_cell)
                .local_get(cells[axis])
                .local_get(cells[axis])
                .i32_const(max_cell)
                .binop(I32GtS)
                .select(Some(ValType::I32))
                .local_set(cells[axis]);
        }

        // out[1] = grid[((k * ny + j) * nx + i) * 4 + grid_ptr]
        body.local_get(out_ptr)
            .local_get(cells[2])
            .i32_const(grid.dims[1] as i32)
            .binop(I32Mul)
            .local_get(cells[1])
            .binop(I32Add)
            .i32_const(grid.dims[0] as i32)
            .binop(I32Mul)
            .local_get(cells[0])
            .binop(I32Add)
            .i32_const(4)
            .binop(I32Mul)
            .load(
                memory_id,
                LoadKind::F32,
                MemArg {
                    align: 2,
                    offset: grid_ptr as u64,
                },
            )
            .store(
                memory_id,
                StoreKind::F32,
                MemArg {
                    align: 2,
                    offset: 4,
                },
            );

        let fid = b.finish(vec![pos_ptr, out_ptr], &mut module.funcs);
        module.exports.add("sample_channels", fid);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(2);
        if buf.is_empty() {
            DensityConfig::default()
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

    let result = decode_fea_mesh(&read_input(0))
        .and_then(|mesh| build_density_grid(&mesh, &config))
        .and_then(|grid| attach_density_channel(&read_input(1), &grid));
    match result {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("density extraction failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "fea_density_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        inputs: vec![
            OperatorMetadataInput::FeaMesh,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::CBORConfiguration(
                "{ min_density: float .default 0.2, max_density: float .default 1.0 }".to_string(),
            ),
        ],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}
