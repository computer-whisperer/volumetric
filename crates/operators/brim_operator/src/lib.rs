//! Brim operator: grows a print-bed adhesion brim from a 3D model's
//! first-layer footprint and unions it with the part.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! # Bed plane convention
//!
//! The print bed is the axis-aligned plane `z = bed_z`, normal +z —
//! `bed_z` defaults to the model's z-min bound (the part sits on the
//! bed). Parts oriented differently should be print-oriented with the
//! rotation operator first, exactly as in a slicer.
//!
//! # Mechanism
//!
//! 1. The input model is scanned through the host's `input_model_*`
//!    sampling imports: one thin slab at `z = bed_z + scan_height` on a
//!    uniform x/y grid covering the model bounds plus the brim margin.
//! 2. A Euclidean distance transform over the scanned footprint bitmap
//!    turns it into `distance_to_footprint` per grid cell. Every contact
//!    patch radiates independently — a sparse polka-dot contact pattern
//!    grows a pad around each dot, pads merging where they overlap. No
//!    hulling or contouring is involved.
//! 3. The field `brim_width - distance` is baked into a
//!    `gridfield_model_core` payload and patched into the embedded
//!    `brim_model_template` module: a standalone 3D model occupied where
//!    the bilinear field is >= 0 within `bed_z <= z <= bed_z +
//!    brim_height` (the zero crossing is the brim contour, smooth at
//!    sub-cell precision). The field is cropped to the actual brim
//!    extent, so the advertised bounds — and the downstream meshing
//!    domain — stay tight around sparse contact patterns.
//! 4. `output: "combined"` (the default) merges the brim with the input
//!    model via `model_merge_core`: emitted glue samples the input first
//!    and falls back to the brim evaluator. `output: "brim"` emits the
//!    bare brim solid for inspection or manual composition.
//!
//! Configuration:
//! - `brim_width`: how far the brim radiates from the footprint.
//! - `brim_height`: brim thickness above the bed (a few layer heights).
//! - `bed_z`: bed plane override; omitted = the model's z-min bound.
//! - `scan_height`: footprint scan offset above the bed; 0 = auto (half
//!   the brim height). Raising it also catches geometry that approaches
//!   but does not touch the bed.
//! - `gap`: also cut `distance < gap`, detaching the brim from the part
//!   (skirt-style). 0 lets the brim run under the footprint.
//! - `outside_only`: keep enclosed holes in the footprint brim-free
//!   (slicer "outer brim only"); off, holes fill like any other nearby
//!   area.
//! - `resolution`: scan grid cells along the longest model x/y axis
//!   (clamped to 16..=2048). Contact features smaller than a cell can be
//!   missed — raise this for fine polka-dot patterns.
//!
//! Typed sample channels follow the input, as with the boolean operator:
//! when the input declares a sample format, the combined output passes
//! `get_sample_format` through and keeps the input's channel row with
//! channel 0 replaced by the union occupancy. In brim-only regions the
//! other channels hold whatever the input reports at that position (e.g.
//! density 0 outside the part). The bare `output: "brim"` solid is
//! occupancy-only.

use wasm_encoder::{BlockType, ExportKind, ExportSection, Function, Instruction, MemArg, ValType};

use model_merge_core::{
    MergeSections, OffsetReencoder, count_sections, find_function_export, parse_model_exports,
};
use volumetric_abi::host::{
    input_model_bounds, input_model_dimensions, input_model_sample, post_output, read_input,
    report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// Prebuilt `brim_model_template` module (see that crate's docs for the
/// regeneration command).
static TEMPLATE: &[u8] = include_bytes!("../template/brim_model_template.wasm");

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum BrimOutput {
    Combined,
    Brim,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct BrimConfig {
    brim_width: f64,
    brim_height: f64,
    bed_z: Option<f64>,
    scan_height: f64,
    gap: f64,
    outside_only: bool,
    resolution: i64,
    output: BrimOutput,
}

impl Default for BrimConfig {
    fn default() -> Self {
        Self {
            brim_width: 5.0,
            brim_height: 0.6,
            bed_z: None,
            scan_height: 0.0,
            gap: 0.0,
            outside_only: false,
            resolution: 256,
            output: BrimOutput::Combined,
        }
    }
}

fn validate(cfg: &BrimConfig) -> Result<(), String> {
    if !(cfg.brim_width.is_finite() && cfg.brim_width > 0.0) {
        return Err(format!(
            "brim_width must be positive, got {}",
            cfg.brim_width
        ));
    }
    if !(cfg.brim_height.is_finite() && cfg.brim_height > 0.0) {
        return Err(format!(
            "brim_height must be positive, got {}",
            cfg.brim_height
        ));
    }
    if !(cfg.gap.is_finite() && cfg.gap >= 0.0) {
        return Err(format!("gap must be non-negative, got {}", cfg.gap));
    }
    if cfg.gap >= cfg.brim_width {
        return Err(format!(
            "gap {} leaves no brim inside brim_width {}",
            cfg.gap, cfg.brim_width
        ));
    }
    if !(cfg.scan_height.is_finite() && cfg.scan_height >= 0.0) {
        return Err(format!(
            "scan_height must be non-negative, got {}",
            cfg.scan_height
        ));
    }
    if let Some(bed_z) = cfg.bed_z
        && !bed_z.is_finite()
    {
        return Err(format!("bed_z must be finite, got {bed_z}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scan grid
// ---------------------------------------------------------------------------

/// The corner-aligned x/y scan lattice (matches `gridfield_model_core`
/// sampling: point (0, 0) sits exactly at (min_x, min_y)).
struct ScanGrid {
    nx: usize,
    ny: usize,
    min_x: f64,
    min_y: f64,
    cell: f64,
}

impl ScanGrid {
    /// `[min_x, max_x, min_y, max_y]` of the lattice, spacing-exact.
    fn bounds(&self) -> [f64; 4] {
        [
            self.min_x,
            self.min_x + (self.nx - 1) as f64 * self.cell,
            self.min_y,
            self.min_y + (self.ny - 1) as f64 * self.cell,
        ]
    }
}

const MAX_GRID_CELLS: u64 = 16_777_216;

fn plan_grid(model_xy: [f64; 4], brim_width: f64, resolution: i64) -> Result<ScanGrid, String> {
    let [min_x, max_x, min_y, max_y] = model_xy;
    let (ex, ey) = (max_x - min_x, max_y - min_y);
    let longest = ex.max(ey);
    if !(longest > 0.0 && longest.is_finite()) {
        return Err(format!("model x/y bounds are degenerate: {model_xy:?}"));
    }
    let cell = longest / resolution.clamp(16, 2048) as f64;
    // The brim reaches at most brim_width beyond the footprint (which lies
    // inside the model bounds); two extra cells give the contour's zero
    // crossing room on the outermost ring.
    let margin = brim_width + 2.0 * cell;
    let points = |extent: f64| ((extent + 2.0 * margin) / cell).ceil() as usize + 1;
    let (nx, ny) = (points(ex), points(ey));
    if nx as u64 * ny as u64 > MAX_GRID_CELLS {
        return Err(format!(
            "brim scan grid {nx}x{ny} exceeds {MAX_GRID_CELLS} cells; lower resolution \
             or brim_width"
        ));
    }
    Ok(ScanGrid {
        nx,
        ny,
        min_x: min_x - margin,
        min_y: min_y - margin,
        cell,
    })
}

/// Occupancy of every lattice point at `z = scan_z`, sampled a bounded
/// batch of rows per host call.
fn scan_footprint(grid: &ScanGrid, scan_z: f64) -> Result<Vec<bool>, String> {
    let ScanGrid {
        nx,
        ny,
        min_x,
        min_y,
        cell,
    } = *grid;
    let mut occupied = vec![false; nx * ny];
    let rows_per_batch = (65536 / nx).max(1);
    let mut positions = Vec::with_capacity(rows_per_batch.min(ny) * nx * 3);
    let mut row = 0usize;
    while row < ny {
        let end = (row + rows_per_batch).min(ny);
        positions.clear();
        for j in row..end {
            let y = min_y + j as f64 * cell;
            for i in 0..nx {
                positions.extend([min_x + i as f64 * cell, y, scan_z]);
            }
        }
        let samples = input_model_sample(0, &positions, 3)
            .ok_or_else(|| "sampling the input model failed".to_string())?;
        for (idx, sample) in samples.iter().enumerate() {
            occupied[row * nx + idx] = is_occupied(*sample);
        }
        row = end;
    }
    Ok(occupied)
}

// ---------------------------------------------------------------------------
// Field baking (pure; natively unit-tested below)
// ---------------------------------------------------------------------------

/// One 1D pass of the Felzenszwalb–Huttenlocher squared distance
/// transform: `out[q] = min_p (f[p] + (q - p)^2)`. `v` needs `f.len()`
/// entries and `z` one more.
fn dt_1d(f: &[f64], out: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    // Squares in f64: usize is 32-bit on wasm and nothing local bounds n.
    let sq = |i: usize| (i as f64) * (i as f64);
    let intersect =
        |q: usize, p: usize| -> f64 { ((f[q] + sq(q)) - (f[p] + sq(p))) / (2 * (q - p)) as f64 };
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = intersect(q, v[k]);
        // Finite inputs keep s finite, so this can't step below z[0].
        while s <= z[k] {
            k -= 1;
            s = intersect(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, out_q) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let d = q as f64 - v[k] as f64;
        *out_q = d * d + f[v[k]];
    }
}

/// Squared Euclidean distance (cell units) from every cell to the nearest
/// occupied cell; cells with no footprint anywhere stay at the finite
/// "far" stand-in, comfortably beyond any real distance.
fn edt_squared(occupied: &[bool], nx: usize, ny: usize) -> Vec<f64> {
    let far = ((nx as f64) * (nx as f64) + (ny as f64) * (ny as f64)) * 4.0 + 1.0;
    let mut d: Vec<f64> = occupied
        .iter()
        .map(|&o| if o { 0.0 } else { far })
        .collect();

    let n = nx.max(ny);
    let mut f = vec![0.0f64; n];
    let mut out = vec![0.0f64; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];

    for row in d.chunks_mut(nx) {
        f[..nx].copy_from_slice(row);
        dt_1d(&f[..nx], &mut out[..nx], &mut v[..nx], &mut z[..nx + 1]);
        row.copy_from_slice(&out[..nx]);
    }
    for col in 0..nx {
        for j in 0..ny {
            f[j] = d[j * nx + col];
        }
        dt_1d(&f[..ny], &mut out[..ny], &mut v[..ny], &mut z[..ny + 1]);
        for j in 0..ny {
            d[j * nx + col] = out[j];
        }
    }
    d
}

/// Cells reachable from the grid border through unoccupied cells
/// (4-connected). Unoccupied cells not in this mask are enclosed holes in
/// the footprint.
fn outside_mask(occupied: &[bool], nx: usize, ny: usize) -> Vec<bool> {
    fn visit(idx: usize, occupied: &[bool], outside: &mut [bool], stack: &mut Vec<usize>) {
        if !outside[idx] && !occupied[idx] {
            outside[idx] = true;
            stack.push(idx);
        }
    }
    let mut outside = vec![false; nx * ny];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..nx {
        visit(i, occupied, &mut outside, &mut stack);
        visit((ny - 1) * nx + i, occupied, &mut outside, &mut stack);
    }
    for j in 0..ny {
        visit(j * nx, occupied, &mut outside, &mut stack);
        visit(j * nx + nx - 1, occupied, &mut outside, &mut stack);
    }
    while let Some(idx) = stack.pop() {
        let (i, j) = (idx % nx, idx / nx);
        if i > 0 {
            visit(idx - 1, occupied, &mut outside, &mut stack);
        }
        if i + 1 < nx {
            visit(idx + 1, occupied, &mut outside, &mut stack);
        }
        if j > 0 {
            visit(idx - nx, occupied, &mut outside, &mut stack);
        }
        if j + 1 < ny {
            visit(idx + nx, occupied, &mut outside, &mut stack);
        }
    }
    outside
}

/// Bake the brim field: `brim_width - distance_to_footprint` in world
/// units, so the payload's bilinear zero crossing is the brim contour. A
/// positive `gap` also cuts `distance < gap` (detached, skirt-style), and
/// `outside_only` forces enclosed holes strongly negative so the contour
/// hugs the footprint edge instead of bleeding inward.
fn build_field(occupied: &[bool], nx: usize, ny: usize, cell: f64, cfg: &BrimConfig) -> Vec<f32> {
    let d2 = edt_squared(occupied, nx, ny);
    let outside = cfg.outside_only.then(|| outside_mask(occupied, nx, ny));
    d2.iter()
        .enumerate()
        .map(|(idx, &d2)| {
            let dist = d2.sqrt() * cell;
            let mut field = cfg.brim_width - dist;
            if cfg.gap > 0.0 {
                field = field.min(dist - cfg.gap);
            }
            if let Some(outside) = &outside
                && !outside[idx]
                && !occupied[idx]
            {
                field = -cfg.brim_width;
            }
            field as f32
        })
        .collect()
}

/// Crop the baked field to the tight bounding box of brim cells plus a
/// one-cell border (room for the zero crossing). The scan grid margins
/// the whole model on every side, but the brim only reaches `brim_width`
/// beyond the footprint — which for sparse contact patterns is far
/// smaller — and the advertised bounds set the downstream meshing
/// domain. `None` when no cell reaches the brim (a `gap` ring thinner
/// than a cell).
fn crop_field(field: Vec<f32>, grid: ScanGrid) -> Option<(Vec<f32>, ScanGrid)> {
    let (nx, ny) = (grid.nx, grid.ny);
    let (mut min_i, mut max_i, mut min_j, mut max_j) = (nx, 0usize, ny, 0usize);
    for j in 0..ny {
        for i in 0..nx {
            if field[j * nx + i] >= 0.0 {
                min_i = min_i.min(i);
                max_i = max_i.max(i);
                min_j = min_j.min(j);
                max_j = max_j.max(j);
            }
        }
    }
    if min_i > max_i {
        return None;
    }
    min_i = min_i.saturating_sub(1);
    min_j = min_j.saturating_sub(1);
    max_i = (max_i + 1).min(nx - 1);
    max_j = (max_j + 1).min(ny - 1);

    let (cx, cy) = (max_i - min_i + 1, max_j - min_j + 1);
    let mut cropped = Vec::with_capacity(cx * cy);
    for j in min_j..=max_j {
        cropped.extend_from_slice(&field[j * nx + min_i..j * nx + max_i + 1]);
    }
    Some((
        cropped,
        ScanGrid {
            nx: cx,
            ny: cy,
            min_x: grid.min_x + min_i as f64 * grid.cell,
            min_y: grid.min_y + min_j as f64 * grid.cell,
            cell: grid.cell,
        },
    ))
}

// ---------------------------------------------------------------------------
// Template patching and model merging
// ---------------------------------------------------------------------------

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

/// The patch slot's address, then drop the helper export — it is not part
/// of the Model ABI.
fn take_slot_export(module: &mut walrus::Module, name: &str) -> Result<i32, String> {
    let export = module
        .exports
        .iter()
        .find(|e| e.name == name)
        .map(|e| (e.id(), e.item))
        .ok_or_else(|| format!("template missing {name} export"))?;
    let addr = match export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(module, f)
            .ok_or_else(|| format!("template {name} is not a constant function"))?,
        _ => return Err(format!("template {name} is not a function")),
    };
    module.exports.delete(export.0);
    Ok(addr)
}

/// Patch the baked field payload (fresh pages, base address into the
/// payload slot) and the z slab (directly into the config slot).
fn patch_template(payload: &[u8], bed_z: f64, z_top: f64) -> Result<Vec<u8>, String> {
    let mut module =
        walrus::Module::from_buffer_with_config(TEMPLATE, &walrus::ModuleConfig::new())
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

    let payload_slot = take_slot_export(&mut module, "brim_payload_slot")?;
    let config_slot = take_slot_export(&mut module, "brim_config_slot")?;

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
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(payload_slot)),
        },
        (base as u32).to_le_bytes().to_vec(),
    );
    let mut config_bytes = Vec::with_capacity(16);
    config_bytes.extend(bed_z.to_le_bytes());
    config_bytes.extend(z_top.to_le_bytes());
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(config_slot)),
        },
        config_bytes,
    );

    Ok(module.emit_wasm())
}

/// `get_bounds` glue: the union of the input's and the brim's bounds is
/// known when the operator runs, so the merged model writes constants.
fn add_bounds_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    a_memory: u32,
    bounds: [f64; 6],
) {
    let ty = sections.types.len();
    sections.types.ty().function([ValType::I32], []);
    sections.funcs.function(ty);

    let mut f = Function::new([]);
    for (i, b) in bounds.iter().enumerate() {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::F64Const((*b).into()));
        f.instruction(&Instruction::F64Store(MemArg {
            offset: (i * 8) as u64,
            align: 3,
            memory_index: a_memory,
        }));
    }
    f.instruction(&Instruction::End);
    sections.code.function(&f);
    exports.export("get_bounds", ExportKind::Func, sections.funcs.len() - 1);
}

/// `sample` glue: the input's occupancy, else the brim evaluator. The
/// position is saved to locals BEFORE calling the input model: the ABI
/// allows `sample` to clobber its position buffer (scale/rotation/
/// translate do, transforming in place), and the brim must be evaluated
/// in the merged model's own coordinate space, not the input chain's.
fn add_sample_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    a_sample: u32,
    brim_sample: u32,
    a_memory: u32,
) {
    let ty = sections.types.len();
    sections.types.ty().function([ValType::I32], [ValType::F32]);
    sections.funcs.function(ty);

    // local 0: pos_ptr (param); locals 1-3: the saved position.
    let mut f = Function::new([(3, ValType::F64)]);
    for axis in 0..3u32 {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::F64Load(MemArg {
            offset: u64::from(axis) * 8,
            align: 3,
            memory_index: a_memory,
        }));
        f.instruction(&Instruction::LocalSet(1 + axis));
    }
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::Call(a_sample));
    f.instruction(&Instruction::F32Const(0.5.into()));
    f.instruction(&Instruction::F32Gt);
    f.instruction(&Instruction::If(BlockType::Result(ValType::F32)));
    f.instruction(&Instruction::F32Const(1.0.into()));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::LocalGet(3));
    f.instruction(&Instruction::Call(brim_sample));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);
    sections.code.function(&f);
    exports.export("sample", ExportKind::Func, sections.funcs.len() - 1);
}

/// `sample_channels` glue: the input's channel row with channel 0
/// replaced by the union occupancy. As in [`add_sample_glue`], the
/// position is saved to locals before the input model runs (its
/// `sample_channels` may clobber the position buffer in place).
fn add_sample_channels_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    a_sample_channels: u32,
    brim_sample: u32,
    a_memory: u32,
) {
    let ty = sections.types.len();
    sections
        .types
        .ty()
        .function([ValType::I32, ValType::I32], []);
    sections.funcs.function(ty);

    let out_mem = MemArg {
        offset: 0,
        align: 2,
        memory_index: a_memory,
    };

    // params: 0 pos_ptr, 1 out_ptr; locals 2-4: the saved position.
    let mut f = Function::new([(3, ValType::F64)]);
    for axis in 0..3u32 {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::F64Load(MemArg {
            offset: u64::from(axis) * 8,
            align: 3,
            memory_index: a_memory,
        }));
        f.instruction(&Instruction::LocalSet(2 + axis));
    }
    // A.sample_channels(pos_ptr, out_ptr) fills the full row.
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::Call(a_sample_channels));
    // out[0] = occupied ? 1.0 : brim_sample(x, y, z)
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::F32Load(out_mem));
    f.instruction(&Instruction::F32Const(0.5.into()));
    f.instruction(&Instruction::F32Gt);
    f.instruction(&Instruction::If(BlockType::Result(ValType::F32)));
    f.instruction(&Instruction::F32Const(1.0.into()));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::LocalGet(3));
    f.instruction(&Instruction::LocalGet(4));
    f.instruction(&Instruction::Call(brim_sample));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::F32Store(out_mem));
    f.instruction(&Instruction::End);
    sections.code.function(&f);
    exports.export(
        "sample_channels",
        ExportKind::Func,
        sections.funcs.len() - 1,
    );
}

/// Merge the input model with the patched brim model: dimensions, IO
/// buffer, and memory pass through from the input; `sample` is the union
/// glue and `get_bounds` the precomputed union. When the input declares
/// typed channels, its format passes through with channel 0 replaced by
/// the union occupancy.
fn merge_with_input(input: &[u8], brim: &[u8], bounds: [f64; 6]) -> Result<Vec<u8>, String> {
    let a_counts = count_sections(input)?;
    let b_counts = count_sections(brim)?;
    let a = parse_model_exports(input)?;
    let brim_sample = find_function_export(brim, "brim_sample")? + a_counts.funcs;

    let mut sections = MergeSections::default();
    sections.append_module(input, &mut OffsetReencoder::identity())?;
    sections.append_module(brim, &mut OffsetReencoder::after(&a_counts))?;

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, a.memory);
    exports.export("get_dimensions", ExportKind::Func, a.get_dimensions);
    exports.export("get_io_ptr", ExportKind::Func, a.get_io_ptr);
    add_bounds_glue(&mut sections, &mut exports, a.memory, bounds);
    add_sample_glue(&mut sections, &mut exports, a.sample, brim_sample, a.memory);
    if let (Some(get_sample_format), Some(sample_channels)) =
        (a.get_sample_format, a.sample_channels)
    {
        exports.export("get_sample_format", ExportKind::Func, get_sample_format);
        add_sample_channels_glue(
            &mut sections,
            &mut exports,
            sample_channels,
            brim_sample,
            a.memory,
        );
    }

    let data_count =
        (a_counts.has_data_count || b_counts.has_data_count).then(|| a_counts.data + b_counts.data);
    Ok(sections.finish(&exports, data_count))
}

// ---------------------------------------------------------------------------
// Operator entry points
// ---------------------------------------------------------------------------

fn build_brim(input: &[u8], cfg: &BrimConfig) -> Result<Vec<u8>, String> {
    validate(cfg)?;

    let dims =
        input_model_dimensions(0).ok_or_else(|| "input 0 is not a usable model".to_string())?;
    if dims != 3 {
        return Err(format!(
            "brim requires a 3D input model; input has {dims} dimensions"
        ));
    }
    let bounds =
        input_model_bounds(0, 3).ok_or_else(|| "failed to read model bounds".to_string())?;
    if bounds.iter().any(|b| !b.is_finite()) {
        return Err(format!("model bounds are not finite: {bounds:?}"));
    }

    let bed_z = cfg.bed_z.unwrap_or(bounds[4]);
    let z_top = bed_z + cfg.brim_height;
    let scan_offset = if cfg.scan_height > 0.0 {
        cfg.scan_height
    } else {
        cfg.brim_height * 0.5
    };
    let scan_z = bed_z + scan_offset;

    let grid = plan_grid(
        [bounds[0], bounds[1], bounds[2], bounds[3]],
        cfg.brim_width,
        cfg.resolution,
    )?;
    let occupied = scan_footprint(&grid, scan_z)?;
    if !occupied.iter().any(|&o| o) {
        return Err(format!(
            "no part geometry found at scan height z = {scan_z}; check bed_z / scan_height"
        ));
    }

    let field = build_field(&occupied, grid.nx, grid.ny, grid.cell, cfg);
    let (field, grid) = crop_field(field, grid).ok_or_else(|| {
        format!(
            "brim is empty: the gap {} ring is thinner than a scan cell; raise \
             resolution or widen brim_width - gap",
            cfg.gap
        )
    })?;
    let payload =
        gridfield_model_core::build_payload(grid.nx as u32, grid.ny as u32, grid.bounds(), &field)?;
    let brim = patch_template(&payload, bed_z, z_top)?;

    match cfg.output {
        BrimOutput::Brim => Ok(brim),
        BrimOutput::Combined => {
            let gb = grid.bounds();
            let merged_bounds = [
                bounds[0].min(gb[0]),
                bounds[1].max(gb[1]),
                bounds[2].min(gb[2]),
                bounds[3].max(gb[3]),
                bounds[4].min(bed_z),
                bounds[5].max(z_top),
            ];
            merge_with_input(input, &brim, merged_bounds)
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg = {
        let buf = read_input(1);
        if buf.is_empty() {
            BrimConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    let input = read_input(0);
    match build_brim(&input, &cfg) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("brim generation failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ brim_width: float .default 5.0, brim_height: float .default 0.6, ? bed_z: float, scan_height: float .default 0.0, gap: float .default 0.0, outside_only: bool .default false, resolution: int .default 256, output: "combined" / "brim" .default "combined" }"#.to_string();
        OperatorMetadata {
            name: "brim_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&str]) -> (Vec<bool>, usize, usize) {
        let ny = rows.len();
        let nx = rows[0].len();
        let occupied = rows
            .iter()
            .flat_map(|r| r.chars().map(|c| c == '#'))
            .collect();
        (occupied, nx, ny)
    }

    #[test]
    fn edt_measures_euclidean_distance() {
        // Single occupied cell in the middle of a 7x5 grid.
        let (occ, nx, ny) = grid(&[".......", ".......", "...#...", ".......", "......."]);
        let d2 = edt_squared(&occ, nx, ny);
        assert_eq!(d2[2 * nx + 3], 0.0);
        assert_eq!(d2[2 * nx + 5], 4.0); // two right
        assert_eq!(d2[0], 9.0 + 4.0); // corner: dx 3, dy 2
        assert_eq!(d2[4 * nx + 6], 9.0 + 4.0);
    }

    #[test]
    fn edt_radiates_from_every_island() {
        // Two dots far apart: each cell measures to its nearest dot, not
        // to any hull between them.
        let (occ, nx, ny) = grid(&["#........#"]);
        let d2 = edt_squared(&occ, nx, ny);
        assert_eq!(d2[0], 0.0);
        assert_eq!(d2[9], 0.0);
        assert_eq!(d2[2], 4.0);
        assert_eq!(d2[7], 4.0);
        assert_eq!(d2[4], 16.0); // midpoint-ish: 4 to the left dot
        assert_eq!(d2[5], 16.0); // 4 to the right dot
    }

    #[test]
    fn edt_with_no_footprint_stays_far() {
        let (occ, nx, ny) = grid(&["...", "..."]);
        let d2 = edt_squared(&occ, nx, ny);
        let far = ((nx * nx + ny * ny) as f64) * 4.0 + 1.0;
        assert!(d2.iter().all(|&d| d == far));
    }

    #[test]
    fn outside_mask_finds_holes() {
        let (occ, nx, ny) = grid(&[".....", ".###.", ".#.#.", ".###.", "....."]);
        let outside = outside_mask(&occ, nx, ny);
        assert!(outside[0]);
        assert!(outside[4 * nx + 4]);
        assert!(!outside[2 * nx + 2], "hole center must not be outside");
        assert!(!outside[nx + 1], "occupied cells are not outside");
    }

    fn test_config() -> BrimConfig {
        BrimConfig {
            brim_width: 2.0,
            brim_height: 0.5,
            ..BrimConfig::default()
        }
    }

    #[test]
    fn field_zero_crossing_sits_at_brim_width() {
        let (occ, nx, ny) = grid(&["#........."]);
        let field = build_field(&occ, nx, ny, 1.0, &test_config());
        assert_eq!(field[0], 2.0); // on the footprint: brim_width
        assert_eq!(field[2], 0.0); // exactly brim_width away
        assert!(field[3] < 0.0);
    }

    #[test]
    fn gap_detaches_the_brim() {
        let (occ, nx, ny) = grid(&["#........."]);
        let cfg = BrimConfig {
            gap: 0.5,
            ..test_config()
        };
        let field = build_field(&occ, nx, ny, 1.0, &cfg);
        assert!(field[0] < 0.0, "footprint itself is cut by the gap");
        assert!(field[1] > 0.0, "ring between gap and brim_width");
        assert!(field[3] < 0.0, "beyond brim_width");
    }

    #[test]
    fn outside_only_empties_holes() {
        let (occ, nx, ny) = grid(&[".....", ".###.", ".#.#.", ".###.", "....."]);
        let open = build_field(&occ, nx, ny, 1.0, &test_config());
        assert!(
            open[2 * nx + 2] > 0.0,
            "without outside_only the hole fills"
        );
        let cfg = BrimConfig {
            outside_only: true,
            ..test_config()
        };
        let field = build_field(&occ, nx, ny, 1.0, &cfg);
        assert!(field[2 * nx + 2] < 0.0, "hole stays brim-free");
        assert!(field[0] > 0.0, "outside corner still gets brim");
    }

    #[test]
    fn crop_hugs_the_brim() {
        // One dot in the middle of a wide grid, brim_width 2: the crop
        // keeps the dot's pad plus a one-cell border, nothing more.
        let (occ, nx, ny) = grid(&[
            "...............",
            "...............",
            ".......#.......",
            "...............",
            "...............",
        ]);
        let scan = ScanGrid {
            nx,
            ny,
            min_x: 0.0,
            min_y: 0.0,
            cell: 1.0,
        };
        let field = build_field(&occ, nx, ny, 1.0, &test_config());
        let (cropped, cropped_grid) = crop_field(field, scan).unwrap();
        assert_eq!((cropped_grid.nx, cropped_grid.ny), (7, 5));
        assert_eq!(cropped.len(), 35);
        assert_eq!(cropped_grid.min_x, 4.0); // dot at 7, pad 2, border 1
        assert_eq!(cropped_grid.min_y, 0.0); // clamped at the grid edge
        // The pad center survives the crop at its new coordinates.
        assert_eq!(cropped[2 * 7 + 3], 2.0);
    }

    #[test]
    fn crop_of_an_all_negative_field_is_none() {
        let (occ, nx, ny) = grid(&["#...."]);
        // gap ring thinner than a cell: nothing reaches field >= 0.
        let cfg = BrimConfig {
            brim_width: 0.4,
            gap: 0.35,
            ..test_config()
        };
        let field = build_field(&occ, nx, ny, 1.0, &cfg);
        let scan = ScanGrid {
            nx,
            ny,
            min_x: 0.0,
            min_y: 0.0,
            cell: 1.0,
        };
        assert!(crop_field(field, scan).is_none());
    }

    #[test]
    fn plan_grid_covers_the_brim_margin() {
        let grid = plan_grid([0.0, 10.0, 0.0, 4.0], 3.0, 100).unwrap();
        let b = grid.bounds();
        assert!(b[0] <= -3.0 && b[1] >= 13.0);
        assert!(b[2] <= -3.0 && b[3] >= 7.0);
        // Spacing-exact corner alignment.
        assert!((b[1] - b[0]) - (grid.nx - 1) as f64 * grid.cell < 1e-12);
        // Cell size follows the longest axis and the resolution.
        assert!((grid.cell - 0.1).abs() < 1e-12);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let bad_width = BrimConfig {
            brim_width: 0.0,
            ..BrimConfig::default()
        };
        assert!(validate(&bad_width).is_err());
        let bad_gap = BrimConfig {
            gap: 5.0,
            ..BrimConfig::default()
        };
        assert!(validate(&bad_gap).is_err());
        assert!(validate(&BrimConfig::default()).is_ok());
    }
}
