//! Island removal operator: ablates geometry that is not sufficiently
//! supported along a build direction — for printers with hard limits on
//! overhang geometry.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! # Mechanism
//!
//! The model is evaluated through the host's `input_model_*` imports on
//! a cubic axis-aligned fine lattice covering its bounds (d-dimensional
//! for a d-dimensional model) at a best-effort `resolution` — but never
//! swept densely. Occupancy comes from an ASN2-style sparse discovery
//! (see the `discovery` module): a coarse corner grid plus aperiodic
//! interior probes locate geometry, mixed cells subdivide toward the
//! fine pitch, and the frontier walks the surface across cell borders,
//! so samples concentrate near geometry exactly like the mesher's. The
//! support walk then streams the resolved lattice one layer at a time,
//! and only the vicinity of geometry (discovery) and of decided prunes
//! (the output mask) is ever stored.
//!
//! 1. Support then propagates layer by layer along the build `axis`,
//!    starting from the `extreme` side: the first layer containing any
//!    geometry seeds as bed-supported. In `mode: "overhang"` a point in
//!    each later layer is supported iff it is occupied and lies within
//!    `tan(overhang_angle)` cells (plus half a cell of quantization
//!    slack) of a supported point in the previous layer — a
//!    (d-1)-dimensional Euclidean distance transform per layer. In
//!    `mode: "island"` support is per connected component instead: an
//!    in-layer region that touches supported material below anywhere
//!    survives whole (in-plane cohesion), so arbitrary horizontal
//!    overhangs are fine and only genuinely detached slice regions die —
//!    the resin-printer failure mode. Everything occupied but never
//!    reached is *removed*, transitively: ablated geometry supports
//!    nothing above it.
//! 2. The removed set is dilated by one cell into *unoccupied*
//!    neighbors, so its multilinear 0.5 level set closes over the true
//!    (sub-cell) surface of an island instead of leaving a thin shell of
//!    it behind. Where a removal cut passes through solid geometry the
//!    level set stays halfway between the last kept and first removed
//!    lattice point.
//! 3. The mask — sparse tiles covering only the removed set and its
//!    dilation ring — is baked into a `voxelmask_model_core` payload,
//!    patched into the embedded `island_model_template`, and merged with
//!    the input model (`model_merge_core`): the glue `sample` cuts the
//!    input where the mask reads >= 0.5. If nothing was removed the
//!    input model passes through unchanged.
//!
//! # Configuration
//!
//! - `mode`: `"overhang"` (default) — FDM-style per-point overhang
//!   limits; `"island"` — resin-style per-component slice attachment
//!   (`overhang_angle` unused).
//! - `overhang_angle`: degrees off the build direction a layer may lean
//!   and still count as supported; 0 = only straight-up support, 45 (the
//!   default) = the common FDM guideline. Must be < 90. Note that large
//!   values make *lateral distance to any supported material* the
//!   criterion, which stops meaning "attached" — for printers that
//!   tolerate arbitrary overhangs but not detached slice regions, use
//!   `mode: "island"` instead.
//! - `axis`: the build axis, as an index, `"x"`/`"y"`/`"z"`/`"w"`, or
//!   `"auto"` (the default) — the model's last axis (z for 3D). For a
//!   tilted build direction, compose with the rotation operator.
//! - `extreme`: which end of the axis the bed is on; `"min"` (default)
//!   means support grows upward from the low end.
//! - `resolution`: fine lattice cells along the longest model axis
//!   (clamped to 16..=2048, cubic cells). Features smaller than a cell
//!   can be missed — such geometry survives untouched (the pass fails
//!   toward keeping, never toward spurious ablation). Samples scale
//!   with the geometry's *surface* at the fine pitch, not the volume.
//! - `discovery_probes`: aperiodic interior samples per corner-uniform
//!   coarse discovery cell (default 8, 0 disables), the safety net for
//!   isolated features smaller than a coarse cell (16 fine cells). The
//!   discovery contract matches the ASN2 mesher: a feature is seen if
//!   the coarse grid or a probe hits it, or if it is surface-connected
//!   to something that was — anything else the mesher would not print
//!   either.
//! - `output`: `"ablated"` (default) — the input minus the removed set;
//!   `"islands"` — the removed set itself (input ∩ mask) for inspection.
//!
//! Typed sample channels follow the input, as with the boolean and brim
//! operators: the input's format and channel row pass through with
//! channel 0 replaced by the cut occupancy.
//!
//! # Bounds
//!
//! The ablated output keeps the input's bounds: ablation only shrinks
//! geometry, and sub-cell islands the scan missed may survive anywhere
//! in the original box. The islands output advertises the removed set's
//! bounding box (padded a cell for the multilinear falloff) intersected
//! with the input bounds — outside it the mask reads 0, so those bounds
//! are exact.

mod discovery;

use wasm_encoder::{BlockType, ExportKind, ExportSection, Function, Instruction, MemArg, ValType};

use model_merge_core::{
    MergeSections, OffsetReencoder, count_sections, find_function_export, find_memory_export,
    parse_model_exports,
};
use volumetric_abi::host::{
    input_model_bounds, input_model_dimensions, input_model_sample, post_output, read_input,
    report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// Prebuilt `island_model_template` module (see that crate's docs for the
/// regeneration command).
static TEMPLATE: &[u8] = include_bytes!("../template/island_model_template.wasm");

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum IslandOutput {
    Ablated,
    Islands,
}

/// Which end of the build axis the bed is on.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Extreme {
    Min,
    Max,
}

/// What "supported" means for a layer.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    /// Per-point: within `tan(overhang_angle)` cells of supported
    /// material in the previous layer. Models FDM-style overhang limits.
    Overhang,
    /// Per connected component: an in-layer region survives whole iff it
    /// overlaps supported material below anywhere — one cured/deposited
    /// piece holds together in-plane, so arbitrary horizontal overhangs
    /// are fine and only genuinely detached regions die. Models resin
    /// printing, where a slice island cures unattached and detaches;
    /// `overhang_angle` is not used.
    Island,
}

/// The component-attach tolerance in island mode, squared: one diagonal
/// cell plus quantization slack, so a truly overlapping region never
/// reads as detached from voxel stairstepping alone.
const ISLAND_ATTACH2: f64 = 1.5 * 1.5;

/// The build axis: an index, a letter for the first four axes, or
/// `"auto"` — the model's last axis. (`"auto"` rather than an optional
/// field: the UI's schema editor has no notion of omitting a field, so
/// the default must be a value it can send.)
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(untagged)]
enum AxisSpec {
    Index(i64),
    Name(String),
}

impl AxisSpec {
    fn resolve(&self, dims: usize) -> Result<usize, String> {
        let index = match self {
            AxisSpec::Index(i) => usize::try_from(*i)
                .map_err(|_| format!("axis index must be non-negative, got {i}"))?,
            AxisSpec::Name(name) => match name.as_str() {
                "auto" => dims - 1,
                "x" => 0,
                "y" => 1,
                "z" => 2,
                "w" => 3,
                other => {
                    return Err(format!(
                        "axis must be an index or one of \"auto\"/\"x\"/\"y\"/\"z\"/\"w\", \
                         got \"{other}\""
                    ));
                }
            },
        };
        if index >= dims {
            return Err(format!(
                "axis {index} is out of range for a {dims}-dimensional model"
            ));
        }
        Ok(index)
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct IslandConfig {
    mode: Mode,
    overhang_angle: f64,
    axis: Option<AxisSpec>,
    extreme: Extreme,
    resolution: i64,
    discovery_probes: i64,
    output: IslandOutput,
}

impl Default for IslandConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Overhang,
            overhang_angle: 45.0,
            axis: None,
            extreme: Extreme::Min,
            resolution: 128,
            discovery_probes: 8,
            output: IslandOutput::Ablated,
        }
    }
}

fn validate(cfg: &IslandConfig) -> Result<(), String> {
    if !(cfg.overhang_angle.is_finite() && (0.0..90.0).contains(&cfg.overhang_angle)) {
        return Err(format!(
            "overhang_angle must be in [0, 90), got {}",
            cfg.overhang_angle
        ));
    }
    if !(0..=4096).contains(&cfg.discovery_probes) {
        return Err(format!(
            "discovery_probes must be in 0..=4096, got {}",
            cfg.discovery_probes
        ));
    }
    Ok(())
}

/// Squared support reach in cells per layer: `tan(overhang_angle)` (the
/// layer step is one cell — the lattice is cubic) plus half a cell of
/// slack so geometry exactly at the limit survives grid quantization.
fn reach_squared(overhang_angle: f64) -> f64 {
    let r = overhang_angle.to_radians().tan() + 0.5;
    r * r
}

// ---------------------------------------------------------------------------
// Scan lattice geometry
// ---------------------------------------------------------------------------

/// The corner-aligned cubic scan lattice. Layers along the build axis
/// are the walk's unit of work: a layer is a (d-1)-dimensional grid over
/// the non-axis model axes, ascending, first fastest.
#[derive(Debug)]
struct ScanGrid {
    /// Lattice point counts per model axis.
    counts: Vec<usize>,
    /// World coordinate of lattice point 0 per model axis.
    mins: Vec<f64>,
    cell: f64,
    /// The build axis.
    axis: usize,
    /// The non-axis model axes, ascending.
    inner_axes: Vec<usize>,
    layer_size: usize,
}

impl ScanGrid {
    fn plan(bounds: &[f64], axis: usize, resolution: i64) -> Result<Self, String> {
        let dims = bounds.len() / 2;
        let extent = |a: usize| bounds[2 * a + 1] - bounds[2 * a];
        let longest = (0..dims).map(extent).fold(0.0f64, f64::max);
        if !(longest > 0.0 && longest.is_finite()) {
            return Err(format!("model bounds are degenerate: {bounds:?}"));
        }
        let cell = longest / resolution.clamp(16, 2048) as f64;

        // One cell of margin on every side: geometry at the model
        // boundary keeps an unoccupied ring for the mask dilation and
        // its multilinear falloff.
        let counts: Vec<usize> = (0..dims)
            .map(|a| (extent(a) / cell).ceil() as usize + 3)
            .collect();
        let mins: Vec<f64> = (0..dims).map(|a| bounds[2 * a] - cell).collect();

        // The tiled mask addresses at most 256 tiles per axis.
        let addressable = 256 * voxelmask_model_core::default_tile_edge(dims);
        if let Some(over) = counts.iter().find(|&&n| n > addressable) {
            return Err(format!(
                "scan lattice needs {over} points on an axis; the {dims}-dimensional tiled \
                 mask addresses at most {addressable} — lower resolution"
            ));
        }
        let inner_axes: Vec<usize> = (0..dims).filter(|&a| a != axis).collect();
        let layer_size = inner_axes.iter().map(|&a| counts[a]).product();

        Ok(Self {
            counts,
            mins,
            cell,
            axis,
            inner_axes,
            layer_size,
        })
    }

    fn dims(&self) -> usize {
        self.counts.len()
    }

    /// Decompose an in-layer index into per-model-axis coordinates
    /// (the build-axis slot is left as given by `layer`).
    fn coords(&self, layer: usize, inner: usize) -> Vec<usize> {
        let mut coords = vec![0usize; self.dims()];
        coords[self.axis] = layer;
        let mut rest = inner;
        for &a in &self.inner_axes {
            coords[a] = rest % self.counts[a];
            rest /= self.counts[a];
        }
        coords
    }

    fn world(&self, coords: &[usize], out: &mut Vec<f64>) {
        for (a, &c) in coords.iter().enumerate() {
            out.push(self.mins[a] + c as f64 * self.cell);
        }
    }
}

/// Iterate every integer point of the box `[lo, hi]` (inclusive),
/// axis 0 fastest.
fn for_each_point(lo: &[usize], hi: &[usize], mut f: impl FnMut(&[usize])) {
    let d = lo.len();
    let mut coords = lo.to_vec();
    loop {
        f(&coords);
        let mut a = 0;
        loop {
            if a == d {
                return;
            }
            if coords[a] < hi[a] {
                coords[a] += 1;
                break;
            }
            coords[a] = lo[a];
            a += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Occupancy discovery
// ---------------------------------------------------------------------------

/// The host-backed sampler behind the ASN2-style discovery: fine lattice
/// coordinates to world positions, batched through `input_model_sample`.
struct HostSampler<'g> {
    grid: &'g ScanGrid,
}

impl discovery::BatchSampler for HostSampler<'_> {
    fn sample_points(&mut self, points: &[usize], dims: usize) -> Result<Vec<bool>, String> {
        const BATCH_POINTS: usize = 65536;
        let mut out = Vec::with_capacity(points.len() / dims);
        let mut positions = Vec::with_capacity(BATCH_POINTS.min(points.len() / dims) * dims);
        for chunk in points.chunks(BATCH_POINTS * dims) {
            positions.clear();
            for point in chunk.chunks_exact(dims) {
                self.grid.world(point, &mut positions);
            }
            let samples = input_model_sample(0, &positions, dims)
                .ok_or_else(|| "sampling the input model failed".to_string())?;
            out.extend(samples.iter().map(|&s| is_occupied(s)));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Support propagation (pure; natively unit-tested below)
// ---------------------------------------------------------------------------

/// One 1D pass of the Felzenszwalb–Huttenlocher squared distance
/// transform: `out[q] = min_p (f[p] + (q - p)^2)`. `v` needs `f.len()`
/// entries and `z` one more. (Same transform as the brim operator's,
/// running here over each in-layer axis of a (d-1)-dimensional layer.)
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

/// Squared Euclidean distance (cell units) from every point of a
/// (d-1)-dimensional layer to the nearest seed point, by separable 1D
/// passes over each in-layer axis; layers with no seed anywhere stay at
/// a finite "far" stand-in — returned so callers can tell it apart from
/// any real distance (which is always strictly below `far - 0.5`).
fn layer_edt_squared(seeds: &[bool], counts: &[usize], out: &mut [f64]) -> f64 {
    let far = counts.iter().map(|&n| (n as f64) * (n as f64)).sum::<f64>() * 4.0 + 1.0;
    for (o, &s) in out.iter_mut().zip(seeds) {
        *o = if s { 0.0 } else { far };
    }
    if counts.is_empty() {
        return far; // a 1-dimensional model: layers are single points
    }

    let longest = counts.iter().copied().max().unwrap();
    let mut f = vec![0.0f64; longest];
    let mut line = vec![0.0f64; longest];
    let mut v = vec![0usize; longest];
    let mut z = vec![0.0f64; longest + 1];

    let mut stride = 1usize;
    for &n in counts {
        let lines = out.len() / n;
        for l in 0..lines {
            // Lines along this axis start at indices that skip the
            // axis's own span: blocks of `stride` consecutive starts,
            // then a jump of `stride * n`.
            let start = (l / stride) * stride * n + l % stride;
            for i in 0..n {
                f[i] = out[start + i * stride];
            }
            dt_1d(&f[..n], &mut line[..n], &mut v[..n], &mut z[..n + 1]);
            for i in 0..n {
                out[start + i * stride] = line[i];
            }
        }
        stride *= n;
    }
    far
}

/// Extend support to whole in-layer connected components (Chebyshev
/// connectivity over the layer axes): a region that touches supported
/// material anywhere is one cured/deposited piece, so all of it holds.
fn spread_within_components(occ: &[bool], supported: &mut [bool], inner_counts: &[usize]) {
    let mut strides = Vec::with_capacity(inner_counts.len());
    let mut s = 1usize;
    for &n in inner_counts {
        strides.push(s);
        s *= n;
    }
    let mut stack: Vec<usize> = (0..supported.len()).filter(|&i| supported[i]).collect();
    while let Some(index) = stack.pop() {
        let mut rest = index;
        let coords: Vec<usize> = inner_counts
            .iter()
            .map(|&n| {
                let c = rest % n;
                rest /= n;
                c
            })
            .collect();
        let lo: Vec<usize> = coords.iter().map(|&c| c.saturating_sub(1)).collect();
        let hi: Vec<usize> = coords
            .iter()
            .zip(inner_counts)
            .map(|(&c, &n)| (c + 1).min(n - 1))
            .collect();
        for_each_point(&lo, &hi, |n_coords| {
            let neighbor: usize = n_coords.iter().zip(&strides).map(|(c, s)| c * s).sum();
            if occ[neighbor] && !supported[neighbor] {
                supported[neighbor] = true;
                stack.push(neighbor);
            }
        });
    }
}

/// Streaming support propagation along the build axis: layers are pushed
/// in support order (the `extreme` end first) and classified against the
/// one retained previous layer, so the full lattice never needs to be
/// stored. The first layer containing any geometry seeds as bed-supported
/// (it prints directly on the bed). `reach2` is the squared per-layer
/// support reach in cells (overhang mode; island mode uses the fixed
/// attach tolerance and spreads support through each layer's connected
/// components).
struct SupportTracker {
    mode: Mode,
    reach2: f64,
    inner_counts: Vec<usize>,
    prev_supported: Option<Vec<bool>>,
    d2: Vec<f64>,
}

impl SupportTracker {
    fn new(grid: &ScanGrid, mode: Mode, reach2: f64) -> Self {
        Self {
            mode,
            reach2,
            inner_counts: grid.inner_axes.iter().map(|&a| grid.counts[a]).collect(),
            prev_supported: None,
            d2: vec![0.0; grid.layer_size],
        }
    }

    /// Classify one layer (in support order) and return its removed
    /// flags: occupied points not reached from the layer below.
    fn push_layer(&mut self, occ: &[bool]) -> Vec<bool> {
        let supported: Vec<bool> = match &self.prev_supported {
            None => {
                if !occ.contains(&true) {
                    return vec![false; occ.len()]; // still below the seed layer
                }
                occ.to_vec()
            }
            Some(prev) => {
                let far = layer_edt_squared(prev, &self.inner_counts, &mut self.d2);
                let reach2 = match self.mode {
                    Mode::Overhang => self.reach2,
                    Mode::Island => ISLAND_ATTACH2,
                };
                // Clamp the reach below the "far" stand-in: a steeper
                // angle than the layer is wide means everything with any
                // seed below is supported, never that seedless is.
                let limit = reach2.min(far - 0.5);
                let mut supported: Vec<bool> = occ
                    .iter()
                    .zip(&self.d2)
                    .map(|(&o, &d)| o && d <= limit)
                    .collect();
                if self.mode == Mode::Island {
                    spread_within_components(occ, &mut supported, &self.inner_counts);
                }
                supported
            }
        };
        let removed = occ.iter().zip(&supported).map(|(&o, &s)| o && !s).collect();
        self.prev_supported = Some(supported);
        removed
    }
}

// ---------------------------------------------------------------------------
// Mask accumulation
// ---------------------------------------------------------------------------

/// In-plane Chebyshev dilation by one cell: the input plus its ±1
/// neighborhood along every layer axis (separable passes).
fn dilate_inplane(bits: &[bool], inner_counts: &[usize]) -> Vec<bool> {
    let mut out = bits.to_vec();
    let mut stride = 1usize;
    for &n in inner_counts {
        let src = out.clone();
        for (i, o) in out.iter_mut().enumerate() {
            let c = i / stride % n;
            *o = src[i] || (c > 0 && src[i - stride]) || (c + 1 < n && src[i + stride]);
        }
        stride *= n;
    }
    out
}

/// Accumulates removed points into the sparse tiled mask, dilating each
/// by one cell (Chebyshev) into *unoccupied* points only — so the mask's
/// 0.5 level set closes beyond an island's outermost scanned point
/// (covering the true sub-cell surface) without biting into kept solid
/// geometry. Only the vicinity of decided prunes is ever stored; layers
/// arrive in walk order and one occupancy layer is retained for the
/// downward dilation, with the upward one deferred until the next layer
/// is scanned.
struct MaskAccumulator<'g> {
    grid: &'g ScanGrid,
    mask: voxelmask_model_core::TiledMaskBuilder,
    /// The previously pushed layer's index and occupancy.
    prev: Option<(usize, Vec<bool>)>,
    /// The previous layer's dilation ball, pending into this layer.
    pending: Option<Vec<bool>>,
}

impl<'g> MaskAccumulator<'g> {
    fn new(grid: &'g ScanGrid) -> Self {
        Self {
            grid,
            mask: voxelmask_model_core::TiledMaskBuilder::new(grid.dims()),
            prev: None,
            pending: None,
        }
    }

    fn set(&mut self, layer: usize, inner: usize) {
        self.mask.set(&self.grid.coords(layer, inner));
    }

    fn push_layer(&mut self, layer: usize, occ: Vec<bool>, removed: &[bool]) {
        // The previous layer's removals dilate upward into this layer's
        // unoccupied points (occupied ones are either removed here too,
        // or kept solid the mask must not bite).
        if let Some(pending) = self.pending.take() {
            for (i, (&p, &o)) in pending.iter().zip(&occ).enumerate() {
                if p && !o {
                    self.set(layer, i);
                }
            }
        }

        if removed.contains(&true) {
            let inner_counts: Vec<usize> = self
                .grid
                .inner_axes
                .iter()
                .map(|&a| self.grid.counts[a])
                .collect();
            let ball = dilate_inplane(removed, &inner_counts);
            for i in 0..removed.len() {
                if removed[i] {
                    self.set(layer, i);
                } else if ball[i] && !occ[i] {
                    self.set(layer, i); // the in-plane dilation ring
                }
            }
            // Downward into the retained previous layer's unoccupied
            // points (removed points there are occupied, so this never
            // re-marks or bites them).
            if let Some((prev_layer, prev_occ)) = &self.prev {
                for (i, (&b, &o)) in ball.iter().zip(prev_occ.iter()).enumerate() {
                    if b && !o {
                        // Field-disjoint from the `prev` borrow.
                        self.mask.set(&self.grid.coords(*prev_layer, i));
                    }
                }
            }
            self.pending = Some(ball);
        }

        self.prev = Some((layer, occ));
    }

    fn finish(self) -> voxelmask_model_core::TiledMaskBuilder {
        // A removal in the final layer would leave its upward dilation
        // pending, but the grid's margin layer past the model bounds is
        // unoccupied and geometry never reaches it — the walk always
        // pushes it, so nothing real is dropped here.
        self.mask
    }
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

/// The patch slot's address, then drop the helper export — the merged
/// module exports only the glue and the input's ABI surface.
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

/// Patch the mask payload into the template (fresh pages, base address
/// into the payload slot) and return the patched module plus the
/// position scratch address. `None` leaves the payload slot zero — an
/// always-empty mask.
fn patch_template(payload: Option<&[u8]>) -> Result<(Vec<u8>, i32), String> {
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

    let payload_slot = take_slot_export(&mut module, "island_payload_slot")?;
    let pos_slot = take_slot_export(&mut module, "island_pos_slot")?;

    if let Some(payload) = payload {
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
    }

    Ok((module.emit_wasm(), pos_slot))
}

/// `get_bounds` glue: writes precomputed constants (the merged model's
/// box is known when the operator runs).
fn add_bounds_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    a_memory: u32,
    bounds: &[f64],
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

/// Emit the position hand-off shared by the `sample` and
/// `sample_channels` glue: copy the d input coordinates from the
/// caller's position buffer (input memory) into the template's position
/// scratch (template memory — the merged module is multi-memory), then
/// evaluate the mask and leave `mask >= 0.5` (i32) on the stack. Runs
/// BEFORE the input model is called: the ABI allows `sample` to clobber
/// its position buffer in place.
fn emit_mask_test(
    f: &mut Function,
    dims: usize,
    a_memory: u32,
    t_memory: u32,
    pos_slot: i32,
    island_sample: u32,
) {
    for i in 0..dims as u64 {
        f.instruction(&Instruction::I32Const(pos_slot));
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::F64Load(MemArg {
            offset: i * 8,
            align: 3,
            memory_index: a_memory,
        }));
        f.instruction(&Instruction::F64Store(MemArg {
            offset: i * 8,
            align: 3,
            memory_index: t_memory,
        }));
    }
    f.instruction(&Instruction::Call(island_sample));
    f.instruction(&Instruction::F32Const(0.5.into()));
    f.instruction(&Instruction::F32Ge);
}

/// `sample` glue: the input's occupancy gated by the mask —
/// `output: "ablated"` cuts where the mask reads removed,
/// `output: "islands"` keeps only there.
#[allow(clippy::too_many_arguments)]
fn add_sample_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    dims: usize,
    a_sample: u32,
    island_sample: u32,
    a_memory: u32,
    t_memory: u32,
    pos_slot: i32,
    keep_removed: bool,
) {
    let ty = sections.types.len();
    sections.types.ty().function([ValType::I32], [ValType::F32]);
    sections.funcs.function(ty);

    let mut f = Function::new([]);
    emit_mask_test(&mut f, dims, a_memory, t_memory, pos_slot, island_sample);
    if !keep_removed {
        f.instruction(&Instruction::I32Eqz);
    }
    f.instruction(&Instruction::If(BlockType::Result(ValType::F32)));
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::Call(a_sample));
    f.instruction(&Instruction::Else);
    f.instruction(&Instruction::F32Const(0.0.into()));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);
    sections.code.function(&f);
    exports.export("sample", ExportKind::Func, sections.funcs.len() - 1);
}

/// `sample_channels` glue: the input's channel row with channel 0 gated
/// by the mask, matching the `sample` glue.
#[allow(clippy::too_many_arguments)]
fn add_sample_channels_glue(
    sections: &mut MergeSections,
    exports: &mut ExportSection,
    dims: usize,
    a_sample_channels: u32,
    island_sample: u32,
    a_memory: u32,
    t_memory: u32,
    pos_slot: i32,
    keep_removed: bool,
) {
    let ty = sections.types.len();
    sections
        .types
        .ty()
        .function([ValType::I32, ValType::I32], []);
    sections.funcs.function(ty);

    // params: 0 pos_ptr, 1 out_ptr; local 2: the mask verdict.
    let mut f = Function::new([(1, ValType::I32)]);
    emit_mask_test(&mut f, dims, a_memory, t_memory, pos_slot, island_sample);
    if !keep_removed {
        f.instruction(&Instruction::I32Eqz);
    }
    f.instruction(&Instruction::LocalSet(2));
    // A.sample_channels(pos_ptr, out_ptr) fills the full row.
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::Call(a_sample_channels));
    // Cut channel 0 when the mask says so.
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::I32Eqz);
    f.instruction(&Instruction::If(BlockType::Empty));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::F32Const(0.0.into()));
    f.instruction(&Instruction::F32Store(MemArg {
        offset: 0,
        align: 2,
        memory_index: a_memory,
    }));
    f.instruction(&Instruction::End);
    f.instruction(&Instruction::End);
    sections.code.function(&f);
    exports.export(
        "sample_channels",
        ExportKind::Func,
        sections.funcs.len() - 1,
    );
}

/// Merge the input model with the patched mask template: dimensions, IO
/// buffer, memory, and sample format pass through from the input;
/// `sample` gates the input's occupancy on the mask and `get_bounds`
/// writes the precomputed box.
fn merge_with_input(
    input: &[u8],
    mask: &[u8],
    pos_slot: i32,
    dims: usize,
    bounds: &[f64],
    keep_removed: bool,
) -> Result<Vec<u8>, String> {
    let a_counts = count_sections(input)?;
    let b_counts = count_sections(mask)?;
    let a = parse_model_exports(input)?;
    let island_sample = find_function_export(mask, "island_sample")? + a_counts.funcs;
    let t_memory = find_memory_export(mask)? + a_counts.memories;

    let mut sections = MergeSections::default();
    sections.append_module(input, &mut OffsetReencoder::identity())?;
    sections.append_module(mask, &mut OffsetReencoder::after(&a_counts))?;

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, a.memory);
    exports.export("get_dimensions", ExportKind::Func, a.get_dimensions);
    exports.export("get_io_ptr", ExportKind::Func, a.get_io_ptr);
    add_bounds_glue(&mut sections, &mut exports, a.memory, bounds);
    add_sample_glue(
        &mut sections,
        &mut exports,
        dims,
        a.sample,
        island_sample,
        a.memory,
        t_memory,
        pos_slot,
        keep_removed,
    );
    if let (Some(get_sample_format), Some(sample_channels)) =
        (a.get_sample_format, a.sample_channels)
    {
        exports.export("get_sample_format", ExportKind::Func, get_sample_format);
        add_sample_channels_glue(
            &mut sections,
            &mut exports,
            dims,
            sample_channels,
            island_sample,
            a.memory,
            t_memory,
            pos_slot,
            keep_removed,
        );
    }

    let data_count =
        (a_counts.has_data_count || b_counts.has_data_count).then(|| a_counts.data + b_counts.data);
    Ok(sections.finish(&exports, data_count))
}

// ---------------------------------------------------------------------------
// Operator entry points
// ---------------------------------------------------------------------------

fn build_output(input: &[u8], cfg: &IslandConfig) -> Result<Vec<u8>, String> {
    validate(cfg)?;

    let dims = input_model_dimensions(0)
        .ok_or_else(|| "input 0 is not a usable model".to_string())? as usize;
    if !(1..=voxelmask_model_core::MAX_DIMS).contains(&dims) {
        return Err(format!(
            "island removal supports 1..={} dimensions; input has {dims}",
            voxelmask_model_core::MAX_DIMS
        ));
    }
    let axis = match &cfg.axis {
        Some(spec) => spec.resolve(dims)?,
        None => dims - 1,
    };
    let bounds =
        input_model_bounds(0, dims).ok_or_else(|| "failed to read model bounds".to_string())?;
    if bounds.iter().any(|b| !b.is_finite()) {
        return Err(format!("model bounds are not finite: {bounds:?}"));
    }

    let grid = ScanGrid::plan(&bounds, axis, cfg.resolution)?;

    // ASN2-style sparse discovery: coarse corners + probes find the
    // geometry, subdivision and frontier walking resolve its vicinity at
    // the fine pitch; empty and solid bulk cost nothing past stage 1.
    let occupancy = {
        let mut sampler = HostSampler { grid: &grid };
        discovery::Discovery::run(&grid.counts, cfg.discovery_probes as usize, &mut sampler)?
    };

    // The streaming walk: assemble one layer at a time in support order,
    // classify it against the retained previous layer, and store only
    // the vicinity of decided prunes in the sparse tiled mask.
    let mut tracker = SupportTracker::new(&grid, cfg.mode, reach_squared(cfg.overhang_angle));
    let mut acc = MaskAccumulator::new(&grid);
    let layers = grid.counts[grid.axis];
    let from_max = cfg.extreme == Extreme::Max;
    let mut occ = Vec::with_capacity(grid.layer_size);
    for step in 0..layers {
        let layer = if from_max { layers - 1 - step } else { step };
        occupancy.fill_layer(grid.axis, layer, &mut occ);
        let removed = tracker.push_layer(&occ);
        acc.push_layer(layer, std::mem::take(&mut occ), &removed);
    }
    let mask_points = acc.finish();

    let (mask, pos_slot, mask_bounds) = match mask_points.bbox() {
        None => {
            if cfg.output == IslandOutput::Ablated {
                // Nothing to remove: the input passes through unchanged.
                return Ok(input.to_vec());
            }
            // The islands output of a fully supported model is empty: an
            // unpatched mask never reads removed, and the empty-model
            // convention is an all-zero box.
            let (mask, pos_slot) = patch_template(None)?;
            (mask, pos_slot, vec![0.0; 2 * dims])
        }
        Some((lo, hi)) => {
            // The payload spans the whole lattice (absent tiles read
            // clear); the set-point bbox — padded a cell for the
            // multilinear falloff — bounds the islands output.
            let lattice_bounds: Vec<f64> = (0..dims)
                .flat_map(|a| {
                    let min = grid.mins[a];
                    [min, min + (grid.counts[a] - 1) as f64 * grid.cell]
                })
                .collect();
            let mut set_bounds = Vec::with_capacity(2 * dims);
            for a in 0..dims {
                set_bounds.push(grid.mins[a] + (lo[a] as f64 - 1.0) * grid.cell);
                set_bounds.push(grid.mins[a] + (hi[a] as f64 + 1.0) * grid.cell);
            }
            let payload = mask_points.finish(&grid.counts, &lattice_bounds)?;
            let (mask, pos_slot) = patch_template(Some(&payload))?;
            (mask, pos_slot, set_bounds)
        }
    };

    let out_bounds: Vec<f64> = match cfg.output {
        // Ablation only shrinks geometry, and sub-cell islands the scan
        // missed survive anywhere in the original box: keep the input's
        // bounds.
        IslandOutput::Ablated => bounds.clone(),
        // The mask reads 0 past its set points (padded a cell for the
        // multilinear falloff), so the islands output is exactly bounded
        // by that box ∩ input.
        IslandOutput::Islands => (0..dims)
            .flat_map(|a| {
                let lo = mask_bounds[2 * a].max(bounds[2 * a]);
                let hi = mask_bounds[2 * a + 1].min(bounds[2 * a + 1]).max(lo);
                [lo, hi]
            })
            .collect(),
    };

    merge_with_input(
        input,
        &mask,
        pos_slot,
        dims,
        &out_bounds,
        cfg.output == IslandOutput::Islands,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let cfg = {
        let buf = read_input(1);
        if buf.is_empty() {
            IslandConfig::default()
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
    match build_output(&input, &cfg) {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("island removal failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ mode: "overhang" / "island" .default "overhang", overhang_angle: float .default 45.0, axis: "auto" / "x" / "y" / "z" / "w" .default "auto", extreme: "min" / "max" .default "min", resolution: int .default 128, discovery_probes: int .default 8, output: "ablated" / "islands" .default "ablated" }"#.to_string();
        OperatorMetadata {
            name: "island_removal_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Island Removal".to_string(),
            description: "Ablate geometry unsupported along the build direction.".to_string(),
            category: "Fabrication".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 21h18"/>"##,
                r##"<path d="M6 21v-5a3 3 0 0 1 6 0v5"/>"##,
                r##"<circle cx="17.5" cy="7.5" r="2.5"/>"##,
                r##"<path d="m21 4-7 7"/>"##,
            )
            .to_string(),
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

    /// A 2D scan grid over textual layers: each string is one layer
    /// along the build axis (model axis 1), each char one point along
    /// axis 0, '#' marking occupied.
    fn grid_2d(width: usize, layers: usize) -> ScanGrid {
        ScanGrid {
            counts: vec![width, layers],
            mins: vec![0.0; 2],
            cell: 1.0,
            axis: 1,
            inner_axes: vec![0],
            layer_size: width,
        }
    }

    fn occ_row(row: &str) -> Vec<bool> {
        row.chars().map(|c| c == '#').collect()
    }

    fn flags_row(flags: &[bool]) -> String {
        flags.iter().map(|&f| if f { '#' } else { '.' }).collect()
    }

    /// Drive the streaming tracker over the layers (respecting the walk
    /// direction) and return each layer's removed flags, in layer order.
    fn walk_removed(layers: &[&str], mode: Mode, reach2: f64, from_max: bool) -> Vec<String> {
        let grid = grid_2d(layers[0].len(), layers.len());
        let mut tracker = SupportTracker::new(&grid, mode, reach2);
        let mut out = vec![String::new(); layers.len()];
        for step in 0..layers.len() {
            let layer = if from_max {
                layers.len() - 1 - step
            } else {
                step
            };
            out[layer] = flags_row(&tracker.push_layer(&occ_row(layers[layer])));
        }
        out
    }

    /// Drive the full walk (tracker + mask accumulator) and return the
    /// sparse mask builder plus the grid.
    fn walk_mask(
        layers: &[&str],
        mode: Mode,
        reach2: f64,
    ) -> (voxelmask_model_core::TiledMaskBuilder, ScanGrid) {
        let grid = grid_2d(layers[0].len(), layers.len());
        let mut tracker = SupportTracker::new(&grid, mode, reach2);
        let mut acc = MaskAccumulator::new(&grid);
        for (layer, row) in layers.iter().enumerate() {
            let occ = occ_row(row);
            let removed = tracker.push_layer(&occ);
            acc.push_layer(layer, occ, &removed);
        }
        (acc.finish(), grid)
    }

    fn mask_layers(mask: &voxelmask_model_core::TiledMaskBuilder, grid: &ScanGrid) -> Vec<String> {
        (0..grid.counts[1])
            .map(|z| {
                (0..grid.counts[0])
                    .map(|i| if mask.is_set(&[i, z]) { '#' } else { '.' })
                    .collect()
            })
            .collect()
    }

    fn none_removed(removed: &[String]) -> bool {
        removed.iter().all(|row| !row.contains('#'))
    }

    #[test]
    fn floating_island_is_removed_supported_column_stays() {
        // A column on the bed and a floating dot with nothing below.
        let removed = walk_removed(
            &[
                "..#.......", //
                "..#.......",
                "..#....#..",
                "..#....#..",
            ],
            Mode::Overhang,
            reach_squared(45.0),
            false,
        );
        assert_eq!(
            removed,
            vec!["..........", "..........", ".......#..", ".......#.."]
        );
    }

    #[test]
    fn overhang_past_the_reach_is_shaved() {
        // A one-layer cantilever jutting 4 cells off a column: at 45
        // degrees (reach 1.5 cells) only the first cell survives.
        let layers = [
            "#.....", //
            "#.....", "#####.",
        ];
        let removed = walk_removed(&layers, Mode::Overhang, reach_squared(45.0), false);
        assert_eq!(removed, vec!["......", "......", "..###."]);
        // A steeper allowance keeps the whole shelf: tan(80) ~ 5.7.
        let removed = walk_removed(&layers, Mode::Overhang, reach_squared(80.0), false);
        assert!(none_removed(&removed));
    }

    #[test]
    fn ablated_geometry_cannot_support_what_is_above_it() {
        // A column starting one layer off the bed is an island, and the
        // geometry sitting on it dies transitively — being directly above
        // occupied (but removed) points doesn't help.
        let removed = walk_removed(
            &[
                "#.....", //
                "#..#..", "#..#..",
            ],
            Mode::Overhang,
            reach_squared(0.0),
            false,
        );
        assert_eq!(removed, vec!["......", "...#..", "...#.."]);
    }

    #[test]
    fn island_mode_keeps_in_plane_connected_overhangs() {
        // A shelf jutting far past the column but connected in-plane:
        // overhang mode shaves it, island mode keeps the whole component
        // (in-plane cohesion — the resin case).
        let layers = [
            "###...", //
            ".#####",
        ];
        let removed = walk_removed(&layers, Mode::Overhang, reach_squared(45.0), false);
        assert_eq!(removed, vec!["......", "....##"]);
        let removed = walk_removed(&layers, Mode::Island, reach_squared(45.0), false);
        assert!(none_removed(&removed), "one connected slice region holds");
    }

    #[test]
    fn island_mode_removes_detached_regions_that_huge_angles_leak() {
        // A dot with nothing below it, but with a tall column elsewhere
        // in the same layers: a near-90 overhang angle reads the distant
        // column as support (lateral reach is the criterion), while
        // island mode sees a detached component and removes it.
        let layers = [
            "#.....", //
            "#..#..", "#..#..",
        ];
        let removed = walk_removed(&layers, Mode::Overhang, reach_squared(89.0), false);
        assert!(none_removed(&removed), "the documented overhang-mode leak");
        let removed = walk_removed(&layers, Mode::Island, reach_squared(89.0), false);
        assert_eq!(removed, vec!["......", "...#..", "...#.."]);
    }

    #[test]
    fn island_mode_attaches_through_stairsteps() {
        // A diagonal one-cell-per-layer staircase: consecutive layers
        // overlap only within the attach tolerance, and each layer's
        // single-cell component must still count as attached.
        let removed = walk_removed(
            &[
                "#.....", //
                ".#....", "..#...",
            ],
            Mode::Island,
            reach_squared(45.0),
            false,
        );
        assert!(none_removed(&removed));
        // A two-cell lateral jump is past the tolerance: detached.
        let removed = walk_removed(
            &[
                "#.....", //
                "...#..",
            ],
            Mode::Island,
            reach_squared(45.0),
            false,
        );
        assert_eq!(removed, vec!["......", "...#.."]);
    }

    #[test]
    fn seed_is_the_first_occupied_layer() {
        // Nothing in layer 0 (slack bounds): layer 1 seeds as the bed.
        let removed = walk_removed(
            &[
                "......", //
                ".##...", ".##...",
            ],
            Mode::Overhang,
            reach_squared(45.0),
            false,
        );
        assert!(none_removed(&removed));
    }

    #[test]
    fn extreme_max_supports_from_the_top() {
        // Hanging from the ceiling: the bottom dot is the island now.
        let removed = walk_removed(
            &[
                ".#....", //
                "......", "...##.", "...##.",
            ],
            Mode::Overhang,
            reach_squared(45.0),
            true,
        );
        assert_eq!(removed, vec![".#....", "......", "......", "......"]);
    }

    #[test]
    fn zero_angle_still_supports_straight_columns() {
        let removed = walk_removed(
            &[
                ".#....", //
                ".#....", ".#....",
            ],
            Mode::Overhang,
            reach_squared(0.0),
            false,
        );
        assert!(none_removed(&removed));
    }

    #[test]
    fn steep_angles_never_read_far_as_supported() {
        // reach^2 at 80 degrees (~38) exceeds the far stand-in of a tiny
        // 3-wide layer (37): the gap must still kill everything above it.
        let removed = walk_removed(
            &[
                "#..", //
                "...", "#..",
            ],
            Mode::Overhang,
            reach_squared(80.0),
            false,
        );
        assert_eq!(removed, vec!["...", "...", "#.."]);
    }

    #[test]
    fn layer_edt_measures_euclidean_distance() {
        let seeds = [
            false, false, false, false, false, //
            false, false, true, false, false, //
            false, false, false, false, false,
        ];
        let mut d2 = vec![0.0; 15];
        layer_edt_squared(&seeds, &[5, 3], &mut d2);
        assert_eq!(d2[5 + 2], 0.0);
        assert_eq!(d2[5 + 4], 4.0); // two right
        assert_eq!(d2[0], 4.0 + 1.0); // dx 2, dy 1
        assert_eq!(d2[10 + 4], 4.0 + 1.0);
    }

    #[test]
    fn layer_edt_with_no_seeds_stays_far() {
        let mut d2 = vec![0.0; 6];
        layer_edt_squared(&[false; 6], &[3, 2], &mut d2);
        assert!(d2.iter().all(|&d| d > 6.0 * 6.0));
    }

    #[test]
    fn zero_dimensional_layers_are_single_points() {
        // A 1-dimensional model: the layer EDT degenerates to
        // seeded-or-far.
        let mut d2 = vec![0.0; 1];
        let far = layer_edt_squared(&[true], &[], &mut d2);
        assert_eq!(d2[0], 0.0);
        let far2 = layer_edt_squared(&[false], &[], &mut d2);
        assert_eq!(d2[0], far);
        assert_eq!(far, far2);
    }

    #[test]
    fn mask_dilates_into_unoccupied_only() {
        // A wall on the left, a floating dot at (3, 1): the dot's mask
        // ball grows into empty space in all directions (including the
        // layers above and below) but never into the wall.
        let (mask, grid) = walk_mask(
            &[
                "##....", //
                "##.#..", "##....",
            ],
            Mode::Overhang,
            reach_squared(45.0),
        );
        assert_eq!(
            mask_layers(&mask, &grid),
            vec!["..###.", "..###.", "..###."],
            "the ball grows into empty space but never into the wall"
        );
    }

    #[test]
    fn mask_payload_reads_the_walked_removals() {
        // A bed column keeps the seed away from the floating dot at
        // (4, 1); the mask holds the dot plus its dilation ball, and the
        // payload's multilinear read crosses 0.5 halfway off the ball.
        let (mask, grid) = walk_mask(
            &[
                "#.......", //
                "#...#...", "#.......",
            ],
            Mode::Overhang,
            reach_squared(45.0),
        );
        assert_eq!(mask.bbox(), Some((vec![3, 0], vec![5, 2])));
        let lattice_bounds: Vec<f64> = vec![0.0, 7.0, 0.0, 2.0];
        let payload = mask.finish(&grid.counts, &lattice_bounds).unwrap();
        let view = voxelmask_model_core::PayloadView::new(&payload).unwrap();
        assert_eq!(view.sample(&[4.0, 1.0]), Some(1.0), "the removed dot");
        assert_eq!(view.sample(&[3.0, 1.0]), Some(1.0), "the dilation ring");
        assert_eq!(view.sample(&[2.0, 1.0]), Some(0.0), "clear past the ring");
        assert_eq!(view.sample(&[2.5, 1.0]), Some(0.5), "halfway crossing");
        assert_eq!(view.sample(&[0.0, 1.0]), Some(0.0), "the kept column");
    }

    #[test]
    fn plan_covers_bounds_with_margin() {
        let grid = ScanGrid::plan(&[0.0, 10.0, -2.0, 2.0, 0.0, 5.0], 2, 100).unwrap();
        assert_eq!(grid.cell, 0.1);
        assert_eq!(grid.axis, 2);
        assert_eq!(grid.inner_axes, vec![0, 1]);
        // ceil(extent/cell) + 3 points per axis, mins one cell out.
        assert_eq!(grid.counts, vec![103, 43, 53]);
        assert_eq!(grid.mins, vec![-0.1, -2.1, -0.1]);
        assert_eq!(grid.layer_size, 103 * 43);
        // Point lookups round-trip.
        let coords = grid.coords(7, 5 + 103 * 2);
        assert_eq!(coords, vec![5, 2, 7]);
    }

    #[test]
    fn plan_accepts_high_resolution_and_rejects_unaddressable() {
        // 512 on a cube — over the old dense-storage cap — now streams.
        let grid = ScanGrid::plan(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 2, 512).unwrap();
        assert_eq!(grid.counts, vec![515, 515, 515]);
        // 2048 fits the 3D tiled key space (4096 per axis).
        assert!(ScanGrid::plan(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 2, 2048).is_ok());
        // A 4D lattice at 2048 exceeds its 2048-per-axis key space.
        let err = ScanGrid::plan(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 3, 2048).unwrap_err();
        assert!(err.contains("addresses at most"), "unexpected error: {err}");
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let bad_angle = IslandConfig {
            overhang_angle: 90.0,
            ..IslandConfig::default()
        };
        assert!(validate(&bad_angle).is_err());
        assert!(validate(&IslandConfig::default()).is_ok());
        assert!(AxisSpec::Name("z".into()).resolve(2).is_err());
        assert_eq!(AxisSpec::Name("y".into()).resolve(2).unwrap(), 1);
        assert_eq!(AxisSpec::Name("auto".into()).resolve(3).unwrap(), 2);
        assert_eq!(AxisSpec::Name("auto".into()).resolve(2).unwrap(), 1);
        assert_eq!(AxisSpec::Index(0).resolve(3).unwrap(), 0);
        assert!(AxisSpec::Index(-1).resolve(3).is_err());
    }
}
