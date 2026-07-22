//! Mesh Drape Operator.
//!
//! Folds the parts of a strut network that protrude outside a model down
//! onto that model's surface: the outside segments become a
//! surface-conforming net (the "skin"), so a trimmed Voronoi lattice ends
//! in a smooth printable face instead of open cell rims or clipped stub
//! ends — without adding material the way a separate conforming surface
//! lattice would.
//!
//! Why this is not a drum skin: draping relocates the outer cells' own
//! struts instead of welding a second lattice on top, and the net it
//! forms inherits the Voronoi skeleton's degree-3 vertices — a polygonal,
//! sub-isostatic net that deforms by strut bending, far softer in-plane
//! than a triangulated skin. The `skin` element flag and
//! `skin_radius_factor` tune what stiffness remains (bending scales as
//! radius^4) or hand the skin to downstream optimization separately from
//! the bulk.
//!
//! Per Bar2 strut:
//! - both nodes inside: kept untouched (the bulk).
//! - crossing the surface: split at the crossing like
//!   `mesh_clip_operator`; the inside stub keeps its radius, and the
//!   outside half folds down — its far node lands on the surface and the
//!   piece becomes skin. A strut sticking straight out folds to (nearly)
//!   a point and welds away; a grazing strut folds flat and survives.
//! - both nodes outside: `outside: "project"` drapes the whole strut onto
//!   the surface; `"drop"` removes it — rims still fold down, but the
//!   arcs tying cell tops together vanish (the most compliant surface).
//!
//! Draped chords subdivide against the surface until they sag less than
//! `chord_tolerance` (default: the strut's own radius), so skin struts
//! follow curved surfaces as polylines. `inset_factor` sinks skin nodes
//! that many strut radii below the surface so the printed strut envelope
//! sits flush instead of half a strut proud. Nodes farther than
//! `max_distance` from the surface are dropped with their struts rather
//! than dragged across space (so box-mode Voronoi hairs vanish instead of
//! piling onto the skin).
//!
//! Point1 clouds drape too: outside points land on the surface (or
//! drop). Hex8 meshes are rejected — volume elements cannot fold.
//!
//! The model is a binary occupancy oracle (the operator ABI contract), so
//! projection estimates a surface direction from a signed stencil of
//! occupancy samples, marches along it to bracket the surface, and
//! bisects — all in lock-step batched rounds, one host call per round,
//! like the clip operator's crossing bisection.
//!
//! Inputs:
//! - Input 0: FeaMesh (Bar2 or Point1) — the mesh to drape
//! - Input 1: ModelWASM (must be 3D) — the surface to drape onto
//! - Input 2: CBOR configuration:
//!   `{ outside: "project" / "drop" .default "project",
//!   skin_radius_factor: float .default 1.0 (scales the `radius` element
//!   field on skin struts), chord_tolerance: float .default 0.0 (0 =
//!   each strut's own radius), inset_factor: float .default 0.0 (sink
//!   skin nodes this many strut radii below the surface — the largest
//!   radius at the node, bulk stubs included), max_distance:
//!   float .default 0.0 (0 = 4 x the median strut length; for Point1,
//!   an eighth of the cloud's bounding diagonal), weld_factor: float
//!   .default 1.0 (welds struts shorter than weld_factor * radius; 0
//!   disables) }`
//!
//! Output 0: CBOR-encoded `FeaMesh` with a scalar `skin` element field
//! (1.0 on draped elements, 0.0 on the bulk).

use std::collections::HashMap;

use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{
    input_model_dimensions, input_model_sample, post_output, read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// Occupancy samples per batched host call.
const SAMPLE_CHUNK: usize = 8192;

/// Directions in the occupancy stencil used to estimate where the
/// surface lies from a point (a Fibonacci sphere).
const STENCIL_DIRS: usize = 32;

/// The stencil starts at `max_distance / 2^STENCIL_DOUBLINGS` and doubles
/// until it sees the other side of the surface.
const STENCIL_DOUBLINGS: u32 = 6;

/// Bisection rounds once a surface crossing is bracketed.
const BISECT_ROUNDS: usize = 24;

/// Geometric growth of the march step while hunting the bracket.
const MARCH_GROWTH: f64 = 1.5;

/// After a first landing, re-aim along the surface normal estimated *at
/// the landing* (where the smallest stencil sees both sides) and land
/// again. Each pass shrinks the tangential error roughly quadratically —
/// without it, hairs fold to visibly off-axis stubs instead of points.
const REFINE_PASSES: usize = 2;

/// Inside stubs of crossing struts shorter than this fraction of the
/// strut are skipped; the fold then anchors at the inside node itself.
const MIN_STUB_FRACTION: f64 = 1e-3;

/// Depth cap for draped-chord subdivision (up to 2^N pieces per strut).
const MAX_DRAPE_DEPTH: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum OutsideConfig {
    /// Drape struts with both nodes outside onto the surface.
    Project,
    /// Remove them; only the outside halves of crossing struts fold.
    Drop,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct DrapeConfig {
    /// What happens to struts entirely outside the model.
    outside: OutsideConfig,
    /// Multiplies the `radius` element field on skin struts.
    skin_radius_factor: f64,
    /// Draped chords subdivide until they sag off the surface by less
    /// than this; 0 = each strut's own radius.
    chord_tolerance: f64,
    /// Sink skin nodes this many strut radii below the surface (the
    /// largest radius at the node, bulk stubs included).
    inset_factor: f64,
    /// Nodes farther than this from the surface drop with their struts;
    /// 0 = 4 x the median strut length.
    max_distance: f64,
    /// Welds struts shorter than `weld_factor * radius`; 0 disables.
    weld_factor: f64,
}

impl Default for DrapeConfig {
    fn default() -> Self {
        Self {
            outside: OutsideConfig::Project,
            skin_radius_factor: 1.0,
            chord_tolerance: 0.0,
            inset_factor: 0.0,
            max_distance: 0.0,
            weld_factor: 1.0,
        }
    }
}

/// A batched occupancy oracle: for each position, whether it lies inside
/// the model. The operator backs this with chunked host sampling; tests
/// with analytic domains.
type OccupiedBatch<'a> = dyn FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String> + 'a;

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|c| a[c] + b[c])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|c| a[c] - b[c])
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    core::array::from_fn(|c| a[c] * s)
}

fn norm(a: [f64; 3]) -> f64 {
    a.iter().map(|c| c * c).sum::<f64>().sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    core::array::from_fn(|c| a[c] + t * (b[c] - a[c]))
}

fn normalize(a: [f64; 3]) -> Option<[f64; 3]> {
    let len = norm(a);
    (len > 1e-12).then(|| scale(a, 1.0 / len))
}

/// The fixed stencil: a Fibonacci sphere of unit directions.
fn stencil_directions() -> [[f64; 3]; STENCIL_DIRS] {
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    core::array::from_fn(|k| {
        let z = 1.0 - (2.0 * k as f64 + 1.0) / STENCIL_DIRS as f64;
        let r = (1.0 - z * z).sqrt();
        let a = golden * k as f64;
        [r * a.cos(), r * a.sin(), z]
    })
}

/// Where a point met the surface.
#[derive(Clone, Copy)]
struct Landing {
    pos: [f64; 3],
    /// Unit direction pointing into the model at the landing.
    inward: [f64; 3],
}

/// March from each point along its heading until occupancy flips away
/// from `start`, then bisect the bracket. `hints` gives the first step
/// to try per point. Returns the flip parameter `t`, or `None` when no
/// flip lies within `max_distance` along that heading. Lock-step
/// batched: a handful of rounds regardless of point count.
fn march_and_bisect(
    points: &[[f64; 3]],
    headings: &[[f64; 3]],
    hints: &[f64],
    start: &[bool],
    max_distance: f64,
    occupied: &mut OccupiedBatch,
) -> Result<Vec<Option<f64>>, String> {
    let n = points.len();
    let mut lo = vec![0.0f64; n];
    let mut bracket: Vec<Option<(f64, f64)>> = vec![None; n];
    let mut t: Vec<f64> = hints
        .iter()
        .map(|&h| h.max(1e-9 * max_distance).min(max_distance))
        .collect();
    loop {
        let active: Vec<usize> = (0..n)
            .filter(|&i| bracket[i].is_none() && t[i].is_finite())
            .collect();
        if active.is_empty() {
            break;
        }
        let batch: Vec<[f64; 3]> = active
            .iter()
            .map(|&i| add(points[i], scale(headings[i], t[i])))
            .collect();
        let occ = occupied(&batch)?;
        for (&i, &o) in active.iter().zip(&occ) {
            if o != start[i] {
                bracket[i] = Some((lo[i], t[i]));
            } else if t[i] >= max_distance {
                t[i] = f64::INFINITY;
            } else {
                lo[i] = t[i];
                t[i] = (t[i] * MARCH_GROWTH).min(max_distance);
            }
        }
    }

    for _ in 0..BISECT_ROUNDS {
        let active: Vec<usize> = (0..n).filter(|&i| bracket[i].is_some()).collect();
        if active.is_empty() {
            break;
        }
        let batch: Vec<[f64; 3]> = active
            .iter()
            .map(|&i| {
                let (lo, hi) = bracket[i].unwrap();
                add(points[i], scale(headings[i], 0.5 * (lo + hi)))
            })
            .collect();
        let occ = occupied(&batch)?;
        for (&i, &o) in active.iter().zip(&occ) {
            let (lo, hi) = bracket[i].as_mut().unwrap();
            let mid = 0.5 * (*lo + *hi);
            if o != start[i] {
                *hi = mid;
            } else {
                *lo = mid;
            }
        }
    }
    Ok(bracket
        .into_iter()
        .map(|b| b.map(|(lo, hi)| 0.5 * (lo + hi)))
        .collect())
}

/// Project each point onto the model surface: estimate a direction from
/// the signed occupancy stencil, march along it to a first landing, then
/// re-aim along the normal estimated *at the landing* and land again
/// (`REFINE_PASSES` times). Works from either side of the surface.
/// Returns `None` where no surface lies within `max_distance`.
fn project_points(
    points: &[[f64; 3]],
    max_distance: f64,
    occupied: &mut OccupiedBatch,
) -> Result<Vec<Option<Landing>>, String> {
    let n = points.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let start = occupied(points)?;
    let dirs = stencil_directions();

    // Phase 1: direction. Occupancy samples on a sphere of radius `h`
    // around the point, signed (+ inside / - outside) and summed, point
    // toward the occupied side. The radius doubles until any sample lands
    // on the other side of the surface.
    let mut heading: Vec<Option<([f64; 3], f64)>> = vec![None; n];
    let mut h = max_distance / f64::from(1u32 << STENCIL_DOUBLINGS);
    while h <= max_distance * 1.001 {
        let active: Vec<usize> = (0..n).filter(|&i| heading[i].is_none()).collect();
        if active.is_empty() {
            break;
        }
        let mut batch = Vec::with_capacity(active.len() * dirs.len());
        for &i in &active {
            for u in &dirs {
                batch.push(add(points[i], scale(*u, h)));
            }
        }
        let occ = occupied(&batch)?;
        for (a, &i) in active.iter().enumerate() {
            let samples = &occ[a * dirs.len()..(a + 1) * dirs.len()];
            if !samples.iter().any(|&o| o != start[i]) {
                continue;
            }
            let mut toward_occupied = [0.0; 3];
            for (u, &o) in dirs.iter().zip(samples) {
                toward_occupied = add(toward_occupied, scale(*u, if o { 1.0 } else { -1.0 }));
            }
            let toward_surface = if start[i] {
                scale(toward_occupied, -1.0)
            } else {
                toward_occupied
            };
            let dir = normalize(toward_surface).unwrap_or_else(|| {
                // Perfectly symmetric occupancy: head for a flipped sample.
                let k = samples.iter().position(|&o| o != start[i]).unwrap();
                dirs[k]
            });
            heading[i] = Some((dir, h));
        }
        h *= 2.0;
    }

    // First landing along the coarse heading.
    let mut landing: Vec<Option<([f64; 3], [f64; 3])>> = vec![None; n]; // (pos, dir used)
    {
        let active: Vec<usize> = (0..n).filter(|&i| heading[i].is_some()).collect();
        let ts = march_and_bisect(
            &active.iter().map(|&i| points[i]).collect::<Vec<_>>(),
            &active
                .iter()
                .map(|&i| heading[i].unwrap().0)
                .collect::<Vec<_>>(),
            &active
                .iter()
                .map(|&i| 0.5 * heading[i].unwrap().1)
                .collect::<Vec<_>>(),
            &active.iter().map(|&i| start[i]).collect::<Vec<_>>(),
            max_distance,
            occupied,
        )?;
        for (&i, t) in active.iter().zip(ts) {
            if let Some(t) = t {
                let dir = heading[i].unwrap().0;
                landing[i] = Some((add(points[i], scale(dir, t)), dir));
            }
        }
    }

    // Refine: the landing sits on the surface, where the smallest
    // stencil radius sees both sides and yields a sharp normal. Re-march
    // from the original point along that normal; keep the previous
    // landing wherever the refined ray misses (grazing geometry).
    for _ in 0..REFINE_PASSES {
        let landed: Vec<usize> = (0..n).filter(|&i| landing[i].is_some()).collect();
        if landed.is_empty() {
            break;
        }
        let normals = estimate_inward(
            &landed
                .iter()
                .map(|&i| landing[i].unwrap().0)
                .collect::<Vec<_>>(),
            max_distance,
            occupied,
        )?;
        let refit: Vec<(usize, [f64; 3])> = landed
            .iter()
            .zip(normals)
            .filter_map(|(&i, nrm)| {
                // Toward the surface: into the model from outside, out
                // of it from inside.
                Some((i, if start[i] { scale(nrm?, -1.0) } else { nrm? }))
            })
            .collect();
        let ts = march_and_bisect(
            &refit.iter().map(|&(i, _)| points[i]).collect::<Vec<_>>(),
            &refit.iter().map(|&(_, d)| d).collect::<Vec<_>>(),
            &refit
                .iter()
                .map(|&(i, _)| 0.5 * dist(points[i], landing[i].unwrap().0))
                .collect::<Vec<_>>(),
            &refit.iter().map(|&(i, _)| start[i]).collect::<Vec<_>>(),
            max_distance,
            occupied,
        )?;
        for (&(i, dir), t) in refit.iter().zip(ts) {
            if let Some(t) = t {
                landing[i] = Some((add(points[i], scale(dir, t)), dir));
            }
        }
    }

    Ok((0..n)
        .map(|i| {
            let (pos, dir) = landing[i]?;
            Some(Landing {
                pos,
                inward: if start[i] { scale(dir, -1.0) } else { dir },
            })
        })
        .collect())
}

/// Estimate the into-the-model direction at points sitting on (or very
/// near) the surface: the signed occupancy stencil at the smallest
/// radius that sees both sides. `None` where every radius saw one side.
fn estimate_inward(
    points: &[[f64; 3]],
    max_distance: f64,
    occupied: &mut OccupiedBatch,
) -> Result<Vec<Option<[f64; 3]>>, String> {
    let n = points.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let dirs = stencil_directions();
    let mut inward: Vec<Option<[f64; 3]>> = vec![None; n];
    let mut h = max_distance / f64::from(1u32 << STENCIL_DOUBLINGS);
    while h <= max_distance * 1.001 {
        let active: Vec<usize> = (0..n).filter(|&i| inward[i].is_none()).collect();
        if active.is_empty() {
            break;
        }
        let mut batch = Vec::with_capacity(active.len() * dirs.len());
        for &i in &active {
            for u in &dirs {
                batch.push(add(points[i], scale(*u, h)));
            }
        }
        let occ = occupied(&batch)?;
        for (a, &i) in active.iter().enumerate() {
            let samples = &occ[a * dirs.len()..(a + 1) * dirs.len()];
            if !(samples.iter().any(|&o| o) && samples.iter().any(|&o| !o)) {
                continue;
            }
            let mut toward_occupied = [0.0; 3];
            for (u, &o) in dirs.iter().zip(samples) {
                toward_occupied = add(toward_occupied, scale(*u, if o { 1.0 } else { -1.0 }));
            }
            inward[i] = normalize(toward_occupied);
        }
        h *= 2.0;
    }
    Ok(inward)
}

/// Where an output node's positions and field values come from.
enum Source {
    /// An input node, unmoved.
    Original(u32),
    /// An input node landed on the surface: fields copy, position moves.
    Moved(u32),
    /// A fresh crossing node: fields interpolate between two input nodes.
    Cross { a: u32, b: u32, t: f64 },
    /// A drape-subdivision node: fields average two *output* nodes
    /// (always created earlier, so assembly stays a single pass).
    Mid { i: u32, j: u32 },
}

/// Accumulates output nodes; interning keeps input nodes shared.
struct Builder<'m> {
    mesh: &'m FeaMesh,
    positions: Vec<f64>,
    sources: Vec<Source>,
    /// Per output node, the into-the-model direction if it was landed.
    inward: Vec<Option<[f64; 3]>>,
    orig_map: Vec<u32>,
    moved_map: Vec<u32>,
}

impl<'m> Builder<'m> {
    fn new(mesh: &'m FeaMesh) -> Self {
        Self {
            mesh,
            positions: Vec::new(),
            sources: Vec::new(),
            inward: Vec::new(),
            orig_map: vec![u32::MAX; mesh.node_count()],
            moved_map: vec![u32::MAX; mesh.node_count()],
        }
    }

    fn push(&mut self, pos: [f64; 3], source: Source, inward: Option<[f64; 3]>) -> u32 {
        let idx = (self.positions.len() / 3) as u32;
        self.positions.extend(pos);
        self.sources.push(source);
        self.inward.push(inward);
        idx
    }

    fn original(&mut self, node: u32) -> u32 {
        if self.orig_map[node as usize] == u32::MAX {
            let pos = self.mesh.node_position(node as usize);
            self.orig_map[node as usize] = self.push(pos, Source::Original(node), None);
        }
        self.orig_map[node as usize]
    }

    fn moved(&mut self, node: u32, landing: &Landing) -> u32 {
        if self.moved_map[node as usize] == u32::MAX {
            self.moved_map[node as usize] =
                self.push(landing.pos, Source::Moved(node), Some(landing.inward));
        }
        self.moved_map[node as usize]
    }

    fn position(&self, i: u32) -> [f64; 3] {
        let base = i as usize * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }
}

/// An output strut: two output nodes, the input element it descends from
/// (for element fields and tolerances), and whether it is skin.
#[derive(Clone, Copy)]
struct Seg {
    a: u32,
    b: u32,
    origin: u32,
    skin: bool,
}

fn drape_bars(
    mesh: &FeaMesh,
    node_occ: &[bool],
    occupied: &mut OccupiedBatch,
    config: &DrapeConfig,
    max_distance: f64,
) -> Result<FeaMesh, String> {
    // Partition struts by where their nodes sit.
    struct Crossing {
        element: usize,
        in_node: u32,
        out_node: u32,
        t: f64,
    }
    let mut whole: Vec<usize> = Vec::new();
    let mut crossings: Vec<Crossing> = Vec::new();
    let mut flats: Vec<usize> = Vec::new();
    for e in 0..mesh.element_count() {
        let pair = mesh.element(e);
        match (node_occ[pair[0] as usize], node_occ[pair[1] as usize]) {
            (true, true) => whole.push(e),
            (false, false) => flats.push(e),
            (a_in, _) => {
                let (in_node, out_node) = if a_in {
                    (pair[0], pair[1])
                } else {
                    (pair[1], pair[0])
                };
                crossings.push(Crossing {
                    element: e,
                    in_node,
                    out_node,
                    t: 0.5,
                });
            }
        }
    }
    if config.outside == OutsideConfig::Drop {
        flats.clear();
    }

    // Bisect each crossing along its strut (t from the inside node).
    {
        let mut lo = vec![0.0f64; crossings.len()];
        let mut hi = vec![1.0f64; crossings.len()];
        for _ in 0..BISECT_ROUNDS {
            if crossings.is_empty() {
                break;
            }
            let mids: Vec<[f64; 3]> = crossings
                .iter()
                .zip(lo.iter().zip(&hi))
                .map(|(x, (&l, &h))| {
                    lerp3(
                        mesh.node_position(x.in_node as usize),
                        mesh.node_position(x.out_node as usize),
                        0.5 * (l + h),
                    )
                })
                .collect();
            let occ = occupied(&mids)?;
            for (k, &o) in occ.iter().enumerate() {
                let mid = 0.5 * (lo[k] + hi[k]);
                if o {
                    lo[k] = mid;
                } else {
                    hi[k] = mid;
                }
            }
        }
        for (k, x) in crossings.iter_mut().enumerate() {
            x.t = 0.5 * (lo[k] + hi[k]);
        }
    }

    // Land every outside node that still matters.
    let mut need: Vec<u32> = crossings
        .iter()
        .map(|x| x.out_node)
        .chain(flats.iter().flat_map(|&e| mesh.element(e).iter().copied()))
        .collect();
    need.sort_unstable();
    need.dedup();
    let landings = project_points(
        &need
            .iter()
            .map(|&nd| mesh.node_position(nd as usize))
            .collect::<Vec<_>>(),
        max_distance,
        occupied,
    )?;
    let landing_of: HashMap<u32, Landing> = need
        .iter()
        .zip(landings)
        .filter_map(|(&nd, l)| l.map(|l| (nd, l)))
        .collect();

    // Assemble the bulk and the initial (unsubdivided) skin.
    let mut builder = Builder::new(mesh);
    let mut segs: Vec<Seg> = Vec::new();
    let mut skin_active: Vec<Seg> = Vec::new();
    // Crossing nodes and their inside anchor, for the inset fallback.
    let mut cross_anchor: HashMap<u32, u32> = HashMap::new();
    for &e in &whole {
        let pair = mesh.element(e);
        let a = builder.original(pair[0]);
        let b = builder.original(pair[1]);
        segs.push(Seg {
            a,
            b,
            origin: e as u32,
            skin: false,
        });
    }
    for x in &crossings {
        let far = landing_of.get(&x.out_node).copied();
        if x.t >= MIN_STUB_FRACTION {
            let a = builder.original(x.in_node);
            let cross_pos = lerp3(
                mesh.node_position(x.in_node as usize),
                mesh.node_position(x.out_node as usize),
                x.t,
            );
            let c = builder.push(
                cross_pos,
                Source::Cross {
                    a: x.in_node,
                    b: x.out_node,
                    t: x.t,
                },
                None,
            );
            cross_anchor.insert(c, x.in_node);
            segs.push(Seg {
                a,
                b: c,
                origin: x.element as u32,
                skin: false,
            });
            if let Some(l) = far {
                let m = builder.moved(x.out_node, &l);
                skin_active.push(Seg {
                    a: c,
                    b: m,
                    origin: x.element as u32,
                    skin: true,
                });
            }
        } else if let Some(l) = far {
            // The crossing sits at the inside node: anchor the fold there.
            let a = builder.original(x.in_node);
            let m = builder.moved(x.out_node, &l);
            skin_active.push(Seg {
                a,
                b: m,
                origin: x.element as u32,
                skin: true,
            });
        }
    }
    for &e in &flats {
        let pair = mesh.element(e);
        let (Some(la), Some(lb)) = (landing_of.get(&pair[0]), landing_of.get(&pair[1])) else {
            continue; // out of reach: the strut drops
        };
        let (la, lb) = (*la, *lb);
        let a = builder.moved(pair[0], &la);
        let b = builder.moved(pair[1], &lb);
        skin_active.push(Seg {
            a,
            b,
            origin: e as u32,
            skin: true,
        });
    }

    // Drape subdivision: a chord whose midpoint hangs off the surface by
    // more than its tolerance splits at the midpoint's own landing.
    let radius_in = mesh
        .element_fields
        .iter()
        .find(|f| f.name == "radius" && f.components == 1);
    let tol_for = |origin: u32, chord: f64| -> f64 {
        if config.chord_tolerance > 0.0 {
            config.chord_tolerance
        } else if let Some(r) = radius_in {
            r.data[origin as usize].abs()
        } else {
            chord / 8.0
        }
    };
    let mut skin_final: Vec<Seg> = Vec::new();
    let mut active = skin_active;
    for depth in 0..=MAX_DRAPE_DEPTH {
        if active.is_empty() {
            break;
        }
        if depth == MAX_DRAPE_DEPTH {
            skin_final.append(&mut active);
            break;
        }
        let info: Vec<([f64; 3], f64)> = active
            .iter()
            .map(|s| {
                let (pa, pb) = (builder.position(s.a), builder.position(s.b));
                (lerp3(pa, pb, 0.5), dist(pa, pb))
            })
            .collect();
        let landings = project_points(
            &info.iter().map(|&(m, _)| m).collect::<Vec<_>>(),
            max_distance,
            occupied,
        )?;
        let mut next = Vec::new();
        for ((seg, &(mid, chord)), landing) in active.iter().zip(&info).zip(landings) {
            match landing {
                Some(l) if dist(l.pos, mid) > tol_for(seg.origin, chord) => {
                    let k = builder.push(
                        l.pos,
                        Source::Mid { i: seg.a, j: seg.b },
                        Some(l.inward),
                    );
                    next.push(Seg { b: k, ..*seg });
                    next.push(Seg { a: k, ..*seg });
                }
                _ => skin_final.push(*seg),
            }
        }
        active = next;
    }
    // Degenerate folds (straight-out struts landing on themselves)
    // collapse to nothing even without a radius field to weld by.
    skin_final.retain(|s| {
        s.a != s.b && dist(builder.position(s.a), builder.position(s.b)) > 1e-6 * max_distance
    });
    segs.extend(skin_final);

    if segs.is_empty() {
        return Err(format!(
            "nothing survived draping {} struts — is the mesh within max_distance \
             ({max_distance}) of the model surface?",
            mesh.element_count()
        ));
    }

    // Element fields follow each strut's input element; skin radii scale.
    let mut element_fields: Vec<FeaField> = mesh
        .element_fields
        .iter()
        .map(|f| FeaField {
            name: f.name.clone(),
            components: f.components,
            data: segs
                .iter()
                .flat_map(|s| {
                    let base = s.origin as usize * f.components;
                    f.data[base..base + f.components].iter().copied()
                })
                .collect(),
        })
        .collect();
    if let Some(rf) = element_fields
        .iter_mut()
        .find(|f| f.name == "radius" && f.components == 1)
    {
        for (k, s) in segs.iter().enumerate() {
            if s.skin {
                rf.data[k] *= config.skin_radius_factor;
            }
        }
    }
    match element_fields.iter_mut().find(|f| f.name == "skin") {
        Some(f) if f.components == 1 => {
            for (k, s) in segs.iter().enumerate() {
                if s.skin {
                    f.data[k] = 1.0;
                }
            }
        }
        Some(_) => {
            return Err("the input already has a non-scalar 'skin' element field".to_string());
        }
        None => element_fields.push(FeaField {
            name: "skin".to_string(),
            components: 1,
            data: segs.iter().map(|s| if s.skin { 1.0 } else { 0.0 }).collect(),
        }),
    }

    // Inset: sink every node touching a skin strut below the surface by
    // `inset_factor` x the largest (already scaled) radius of *any*
    // strut at it — a rim-fold anchor also carries its full-radius bulk
    // stub, whose end cap must not poke past the flush envelope.
    if config.inset_factor > 0.0 {
        let Some(rf) = element_fields
            .iter()
            .find(|f| f.name == "radius" && f.components == 1)
        else {
            return Err(
                "inset_factor needs a scalar 'radius' element field on the mesh".to_string()
            );
        };
        let node_count = builder.sources.len();
        let mut on_skin = vec![false; node_count];
        let mut max_radius = vec![0.0f64; node_count];
        for (k, s) in segs.iter().enumerate() {
            for nd in [s.a, s.b] {
                on_skin[nd as usize] |= s.skin;
                max_radius[nd as usize] = max_radius[nd as usize].max(rf.data[k]);
            }
        }
        let inset_len: Vec<f64> = on_skin
            .iter()
            .zip(&max_radius)
            .map(|(&skin, &r)| if skin { r * config.inset_factor } else { 0.0 })
            .collect();
        // Landed nodes carry their direction; crossing (and rare
        // original) anchors on skin struts estimate one from a fresh
        // stencil, falling back to along-the-strut toward the inside.
        let unknown: Vec<u32> = (0..node_count as u32)
            .filter(|&i| inset_len[i as usize] > 0.0 && builder.inward[i as usize].is_none())
            .collect();
        let estimates = estimate_inward(
            &unknown
                .iter()
                .map(|&i| builder.position(i))
                .collect::<Vec<_>>(),
            max_distance,
            occupied,
        )?;
        for (&i, est) in unknown.iter().zip(estimates) {
            builder.inward[i as usize] = est.or_else(|| {
                cross_anchor.get(&i).and_then(|&in_node| {
                    normalize(sub(
                        mesh.node_position(in_node as usize),
                        builder.position(i),
                    ))
                })
            });
        }
        for i in 0..node_count {
            if inset_len[i] > 0.0
                && let Some(inward) = builder.inward[i]
            {
                let moved = add(builder.position(i as u32), scale(inward, inset_len[i]));
                builder.positions[i * 3..i * 3 + 3].copy_from_slice(&moved);
            }
        }
    }

    // Node fields resolve per source, in output order (Mid entries
    // reference already-resolved output nodes).
    let node_fields: Vec<FeaField> = mesh
        .node_fields
        .iter()
        .map(|f| {
            let c = f.components;
            let mut data: Vec<f64> = Vec::with_capacity(builder.sources.len() * c);
            for source in &builder.sources {
                match *source {
                    Source::Original(n) | Source::Moved(n) => {
                        let base = n as usize * c;
                        data.extend_from_slice(&f.data[base..base + c]);
                    }
                    Source::Cross { a, b, t } => {
                        let (a, b) = (a as usize * c, b as usize * c);
                        for k in 0..c {
                            data.push(f.data[a + k] + t * (f.data[b + k] - f.data[a + k]));
                        }
                    }
                    Source::Mid { i, j } => {
                        let (i, j) = (i as usize * c, j as usize * c);
                        for k in 0..c {
                            let v = 0.5 * (data[i + k] + data[j + k]);
                            data.push(v);
                        }
                    }
                }
            }
            FeaField {
                name: f.name.clone(),
                components: c,
                data,
            }
        })
        .collect();

    let mut out = FeaMesh {
        element_kind: FeaElementKind::Bar2,
        node_positions: builder.positions,
        connectivity: segs.iter().flat_map(|s| [s.a, s.b]).collect(),
        node_fields,
        element_fields,
    };
    out.validate()?;
    // Degenerate-fold filtering can orphan a landed node; keeping every
    // element through filter_elements compacts the unreferenced ones
    // away (a floating node is singular for downstream FEA).
    out = mesh_edit_core::filter_elements(&out, &vec![true; out.element_count()])?;

    if config.weld_factor > 0.0
        && let Some(rf) = out
            .element_fields
            .iter()
            .find(|f| f.name == "radius" && f.components == 1)
    {
        let thresholds: Vec<f64> = rf.data.iter().map(|r| r * config.weld_factor).collect();
        out = mesh_edit_core::weld_short_bars(&out, &|e| thresholds[e])?;
        if out.element_count() == 0 {
            return Err("welding collapsed every draped strut".to_string());
        }
    }
    Ok(out)
}

fn drape_points(
    mesh: &FeaMesh,
    node_occ: &[bool],
    occupied: &mut OccupiedBatch,
    config: &DrapeConfig,
    max_distance: f64,
) -> Result<FeaMesh, String> {
    let mut working = mesh.clone();
    let mut landed = vec![false; mesh.node_count()];
    if config.outside == OutsideConfig::Project {
        let mut outs: Vec<u32> = mesh
            .connectivity
            .iter()
            .copied()
            .filter(|&n| !node_occ[n as usize])
            .collect();
        outs.sort_unstable();
        outs.dedup();
        let landings = project_points(
            &outs
                .iter()
                .map(|&n| mesh.node_position(n as usize))
                .collect::<Vec<_>>(),
            max_distance,
            occupied,
        )?;
        for (&n, l) in outs.iter().zip(&landings) {
            if let Some(l) = l {
                landed[n as usize] = true;
                working.node_positions[n as usize * 3..n as usize * 3 + 3]
                    .copy_from_slice(&l.pos);
            }
        }
    }

    // Tag before filtering so element fields follow the survivors.
    let skin: Vec<f64> = (0..mesh.element_count())
        .map(|e| {
            let n = mesh.element(e)[0] as usize;
            if !node_occ[n] && landed[n] { 1.0 } else { 0.0 }
        })
        .collect();
    match working.element_fields.iter_mut().find(|f| f.name == "skin") {
        Some(f) if f.components == 1 => {
            for (v, s) in f.data.iter_mut().zip(&skin) {
                if *s > 0.0 {
                    *v = 1.0;
                }
            }
        }
        Some(_) => {
            return Err("the input already has a non-scalar 'skin' element field".to_string());
        }
        None => working.element_fields.push(FeaField {
            name: "skin".to_string(),
            components: 1,
            data: skin,
        }),
    }

    let keep: Vec<bool> = (0..mesh.element_count())
        .map(|e| {
            let n = mesh.element(e)[0] as usize;
            node_occ[n] || landed[n]
        })
        .collect();
    let out = mesh_edit_core::filter_elements(&working, &keep)?;
    if out.element_count() == 0 {
        return Err("no points survived the drape — is the cloud within max_distance \
                    of the model?"
            .to_string());
    }
    Ok(out)
}

fn resolve_max_distance(mesh: &FeaMesh, config: &DrapeConfig) -> Result<f64, String> {
    if config.max_distance > 0.0 {
        return Ok(config.max_distance);
    }
    let auto = match mesh.element_kind {
        FeaElementKind::Bar2 => {
            let mut lens: Vec<f64> = (0..mesh.element_count())
                .map(|e| {
                    let pair = mesh.element(e);
                    dist(
                        mesh.node_position(pair[0] as usize),
                        mesh.node_position(pair[1] as usize),
                    )
                })
                .collect();
            let k = lens.len() / 2;
            lens.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
            lens[k] * 4.0
        }
        _ => {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for n in 0..mesh.node_count() {
                let p = mesh.node_position(n);
                for c in 0..3 {
                    lo[c] = lo[c].min(p[c]);
                    hi[c] = hi[c].max(p[c]);
                }
            }
            dist(lo, hi) / 8.0
        }
    };
    if auto.is_finite() && auto > 0.0 {
        Ok(auto)
    } else {
        Err("cannot infer max_distance from a degenerate mesh; set it explicitly".to_string())
    }
}

/// Shared entry for host and tests: classify nodes, resolve the reach,
/// dispatch by element kind.
fn drape(
    mesh: &FeaMesh,
    occupied: &mut OccupiedBatch,
    config: &DrapeConfig,
) -> Result<FeaMesh, String> {
    if mesh.element_count() == 0 {
        return Err("the input mesh has no elements".to_string());
    }
    let max_distance = resolve_max_distance(mesh, config)?;
    let node_points: Vec<[f64; 3]> = (0..mesh.node_count()).map(|n| mesh.node_position(n)).collect();
    let node_occ = occupied(&node_points)?;
    match mesh.element_kind {
        FeaElementKind::Point1 => drape_points(mesh, &node_occ, occupied, config, max_distance),
        FeaElementKind::Bar2 => drape_bars(mesh, &node_occ, occupied, config, max_distance),
        kind => Err(format!(
            "cannot drape {kind:?} volume elements onto a surface; clip them instead"
        )),
    }
}

fn validate_config(config: &DrapeConfig) -> Result<(), String> {
    if !(config.skin_radius_factor.is_finite() && config.skin_radius_factor > 0.0) {
        return Err(format!(
            "skin_radius_factor must be positive, got {}",
            config.skin_radius_factor
        ));
    }
    for (name, v) in [
        ("chord_tolerance", config.chord_tolerance),
        ("inset_factor", config.inset_factor),
        ("max_distance", config.max_distance),
        ("weld_factor", config.weld_factor),
    ] {
        if !(v.is_finite() && v >= 0.0) {
            return Err(format!("{name} must be non-negative, got {v}"));
        }
    }
    Ok(())
}

fn build_draped(config: &DrapeConfig) -> Result<FeaMesh, String> {
    validate_config(config)?;
    let mesh = decode_fea_mesh(&read_input(0))?;
    let dims =
        input_model_dimensions(1).ok_or_else(|| "input 1 is not a usable model".to_string())?;
    if dims != 3 {
        return Err(format!(
            "mesh drape needs a 3D model; input has {dims} dimensions"
        ));
    }
    let mut occupied = |points: &[[f64; 3]]| -> Result<Vec<bool>, String> {
        let mut out = Vec::with_capacity(points.len());
        for chunk in points.chunks(SAMPLE_CHUNK) {
            let positions: Vec<f64> = chunk.iter().flatten().copied().collect();
            let samples = input_model_sample(1, &positions, 3)
                .ok_or_else(|| "sampling the model failed".to_string())?;
            out.extend(samples.iter().map(|&s| is_occupied(s)));
        }
        Ok(out)
    };
    drape(&mesh, &mut occupied, config)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(2);
        if buf.is_empty() {
            DrapeConfig::default()
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

    match build_draped(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("mesh drape failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ outside: "project" / "drop" .default "project", skin_radius_factor: float .default 1.0, chord_tolerance: float .default 0.0, inset_factor: float .default 0.0, max_distance: float .default 0.0, weld_factor: float .default 1.0 }"#
            .to_string();
        OperatorMetadata {
            name: "mesh_drape_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Mesh Drape".to_string(),
            description: "Project the struts of a lattice that protrude outside a model \
                          onto its surface: protruding arcs fold flat into a smooth, \
                          compliance-tunable skin."
                .to_string(),
            category: "Mesh".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 18c3.5-8.5 14.5-8.5 18 0"/>"##,
                r##"<path d="M13.5 4.5 11.7 9.9"/>"##,
                r##"<path d="M11.7 9.9 15.5 11.3"/>"##,
                r##"<path d="M8 20l2.7-5.3"/>"##,
                r##"<circle cx="11.7" cy="9.9" r=".4"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec![
                "Mesh".to_string(),
                "Surface".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Occupancy oracle for a sphere of radius `r` at the origin.
    fn sphere(r: f64) -> impl FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String> {
        move |points| {
            Ok(points
                .iter()
                .map(|p| p.iter().map(|c| c * c).sum::<f64>() < r * r)
                .collect())
        }
    }

    /// Occupancy oracle for the half-space z < 0.
    fn floor() -> impl FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String> {
        move |points| Ok(points.iter().map(|p| p[2] < 0.0).collect())
    }

    fn bar_mesh(nodes: &[[f64; 3]], bars: &[[u32; 2]], radius: f64) -> FeaMesh {
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: nodes.iter().flatten().copied().collect(),
            connectivity: bars.iter().flatten().copied().collect(),
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![radius; bars.len()],
            }],
        }
    }

    fn positions(mesh: &FeaMesh) -> Vec<[f64; 3]> {
        (0..mesh.node_count()).map(|n| mesh.node_position(n)).collect()
    }

    fn field<'m>(mesh: &'m FeaMesh, name: &str) -> &'m FeaField {
        mesh.element_fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("no {name} field"))
    }

    #[test]
    fn project_points_lands_on_the_surface_from_both_sides() {
        let mut oracle = sphere(1.0);
        let points = [
            [0.0, 0.0, 1.5],   // outside, above the pole
            [0.9, 0.7, -0.4],  // outside, oblique
            [0.0, 0.0, 0.4],   // inside
            [0.0, 0.0, 5.0],   // far out of reach
        ];
        let landings = project_points(&points, 1.0, &mut oracle).unwrap();

        for (p, l) in points.iter().zip(&landings).take(3) {
            let l = l.as_ref().expect("within reach");
            assert!(
                (norm(l.pos) - 1.0).abs() < 1e-5,
                "landing {:?} of {p:?} is off the sphere",
                l.pos
            );
            // Inward points toward the origin at a sphere landing.
            let radial = normalize(l.pos).unwrap();
            let dot: f64 = (0..3).map(|c| l.inward[c] * -radial[c]).sum();
            assert!(dot > 0.9, "inward {:?} not radial (dot {dot})", l.inward);
        }
        assert!(landings[3].is_none(), "unreachable point should not land");
    }

    #[test]
    fn hairs_fold_to_nothing_and_the_bulk_passes_through() {
        // A radial chain: inside strut, then a strut crossing the r=1
        // sphere and sticking straight out. The protruding half folds
        // onto (nearly) its own crossing point and welds away.
        let mesh = bar_mesh(
            &[[0.0, 0.0, 0.5], [0.0, 0.0, 0.9], [0.0, 0.0, 1.4]],
            &[[0, 1], [1, 2]],
            0.05,
        );
        let mut oracle = sphere(1.0);
        let out = drape(&mesh, &mut oracle, &DrapeConfig::default()).unwrap();

        assert_eq!(out.element_count(), 2, "bulk strut + surface stub");
        assert!(
            field(&out, "skin").data.iter().all(|&s| s == 0.0),
            "no skin should survive a straight-out hair"
        );
        let max_r = positions(&out)
            .iter()
            .map(|p| norm(*p))
            .fold(0.0f64, f64::max);
        assert!(max_r < 1.03, "nothing may stay past the surface: {max_r}");
        // The bulk strut is untouched.
        assert!(positions(&out).iter().any(|p| dist(*p, [0.0, 0.0, 0.5]) < 1e-12));
        assert!(positions(&out).iter().any(|p| dist(*p, [0.0, 0.0, 0.9]) < 1e-12));
    }

    #[test]
    fn tangential_arcs_drape_and_subdivide_onto_the_sphere() {
        // A strut floating tangentially above the sphere: both landings
        // are on the sphere and the chord between them subdivides until
        // it hugs the surface to within the strut radius.
        let mesh = bar_mesh(&[[-0.6, 0.0, 1.1], [0.6, 0.0, 1.1]], &[[0, 1]], 0.01);
        let mut oracle = sphere(1.0);
        let config = DrapeConfig {
            skin_radius_factor: 0.5,
            ..DrapeConfig::default()
        };
        let out = drape(&mesh, &mut oracle, &config).unwrap();

        assert!(
            out.element_count() >= 4,
            "the chord must subdivide, got {} elements",
            out.element_count()
        );
        for p in positions(&out) {
            let r = norm(p);
            assert!((0.98..1.005).contains(&r), "node {p:?} off the surface");
        }
        // Every chord now sags less than ~the tolerance.
        for e in 0..out.element_count() {
            let pair = out.element(e);
            let mid = lerp3(
                out.node_position(pair[0] as usize),
                out.node_position(pair[1] as usize),
                0.5,
            );
            assert!(norm(mid) > 0.97, "segment {e} still sags: {}", norm(mid));
        }
        let skin = field(&out, "skin");
        assert!(skin.data.iter().all(|&s| s == 1.0), "all segments are skin");
        let radius = field(&out, "radius");
        assert!(
            radius.data.iter().all(|&r| (r - 0.005).abs() < 1e-12),
            "skin radius must scale by the factor: {:?}",
            radius.data
        );
    }

    #[test]
    fn outside_drop_removes_arcs_but_still_folds_rims() {
        let nodes = [
            [0.5, 0.0, 0.5],   // inside
            [1.2, 0.0, 0.5],   // outside (crossing partner)
            [0.0, 1.05, 0.3],  // outside (flat strut)
            [0.3, 1.05, 0.3],  // outside (flat strut)
        ];
        let bars = [[0, 1], [2, 3]];
        let mesh = bar_mesh(&nodes, &bars, 0.02);

        let mut oracle = sphere(1.0);
        let config = DrapeConfig {
            outside: OutsideConfig::Drop,
            ..DrapeConfig::default()
        };
        let dropped = drape(&mesh, &mut oracle, &config).unwrap();
        assert_eq!(
            dropped.element_count(),
            2,
            "stub + folded rim only: {:?}",
            field(&dropped, "skin").data
        );
        let skin = field(&dropped, "skin");
        assert_eq!(skin.data.iter().filter(|&&s| s == 1.0).count(), 1);

        let mut oracle = sphere(1.0);
        let projected = drape(&mesh, &mut oracle, &DrapeConfig::default()).unwrap();
        assert!(
            projected.element_count() > dropped.element_count(),
            "project mode keeps the draped arc"
        );
        // The flat strut's nodes ended up on the sphere.
        let on_surface = positions(&projected)
            .iter()
            .filter(|p| (norm(**p) - 1.0).abs() < 1e-4)
            .count();
        assert!(on_surface >= 3, "flat strut + crossing land on the surface");
    }

    #[test]
    fn fields_follow_the_drape() {
        // Node field: score == x + z everywhere, so any lerp of nodes
        // agrees with the geometry it lands between. Element field: a
        // distinct tag per strut, following every descendant.
        let nodes = [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.4],
            [-0.6, 0.0, 1.1],
            [0.6, 0.0, 1.1],
        ];
        let mut mesh = bar_mesh(&nodes, &[[0, 1], [2, 3]], 0.01);
        mesh.node_fields.push(FeaField {
            name: "score".to_string(),
            components: 1,
            data: nodes.iter().map(|p| p[0] + p[2]).collect(),
        });
        mesh.element_fields.push(FeaField {
            name: "tag".to_string(),
            components: 1,
            data: vec![7.0, 9.0],
        });
        let mut oracle = sphere(1.0);
        let config = DrapeConfig {
            weld_factor: 0.0, // keep every node so sources stay inspectable
            ..DrapeConfig::default()
        };
        let out = drape(&mesh, &mut oracle, &config).unwrap();

        let score = out.node_fields.iter().find(|f| f.name == "score").unwrap();
        assert_eq!(score.data.len(), out.node_count());
        // The moved far node keeps its original score even though it
        // landed elsewhere; the original node 1 score was 0 + 1.4.
        assert!(
            score.data.iter().any(|&s| (s - 1.4).abs() < 1e-9),
            "moved node keeps its field: {:?}",
            score.data
        );
        // Crossing node score interpolates to x+z of the crossing point
        // (both are linear along the strut).
        let crossing = (0..out.node_count())
            .find(|&n| {
                let p = out.node_position(n);
                (norm(p) - 1.0).abs() < 1e-6 && p[0].abs() < 1e-6 && p[1].abs() < 1e-6
            })
            .expect("crossing node on the pole");
        let p = out.node_position(crossing);
        assert!((score.data[crossing] - (p[0] + p[2])).abs() < 1e-6);

        // Tags: strut 0 descendants tagged 7, strut 1 descendants 9.
        let tag = field(&out, "tag");
        let skin = field(&out, "skin");
        for e in 0..out.element_count() {
            assert!(tag.data[e] == 7.0 || tag.data[e] == 9.0);
            if tag.data[e] == 9.0 {
                assert_eq!(skin.data[e], 1.0, "the tangential strut is all skin");
            }
        }
        assert!(tag.data.contains(&7.0) && tag.data.contains(&9.0));
    }

    #[test]
    fn inset_sinks_skin_nodes_below_the_surface() {
        // Flat strut above the floor plus a slanted crossing strut; with
        // inset_factor 1 every skin node sinks one radius below z = 0,
        // including the crossing anchor (stencil-estimated normal).
        let nodes = [
            [-0.2, 0.0, 0.25],
            [0.2, 0.0, 0.25],
            [0.5, 0.0, -0.2],
            [0.9, 0.0, 0.2],
        ];
        let mesh = bar_mesh(&nodes, &[[0, 1], [2, 3]], 0.05);
        let mut oracle = floor();
        let config = DrapeConfig {
            inset_factor: 1.0,
            ..DrapeConfig::default()
        };
        let out = drape(&mesh, &mut oracle, &config).unwrap();

        let skin = field(&out, "skin");
        let mut skin_nodes = std::collections::HashSet::new();
        for e in 0..out.element_count() {
            if skin.data[e] == 1.0 {
                skin_nodes.extend(out.element(e).iter().copied());
            }
        }
        assert!(skin_nodes.len() >= 3, "flat strut + rim fold");
        for &n in &skin_nodes {
            let z = out.node_position(n as usize)[2];
            assert!(
                (z + 0.05).abs() < 5e-3,
                "skin node {n} should sit one radius under the floor, z = {z}"
            );
        }
        // The bulk nodes are untouched.
        assert!(positions(&out).iter().any(|p| dist(*p, [0.5, 0.0, -0.2]) < 1e-12));
    }

    #[test]
    fn point_clouds_project_or_drop() {
        let cloud = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: vec![
                0.0, 0.0, 0.5, //
                0.0, 0.0, 1.3, //
                0.0, 0.0, 3.0,
            ],
            connectivity: vec![0, 1, 2],
            node_fields: vec![],
            element_fields: vec![],
        };
        let config = DrapeConfig {
            max_distance: 1.0,
            ..DrapeConfig::default()
        };
        let mut oracle = sphere(1.0);
        let out = drape(&cloud, &mut oracle, &config).unwrap();
        assert_eq!(out.element_count(), 2, "far point drops");
        assert_eq!(field(&out, "skin").data, vec![0.0, 1.0]);
        let landed = positions(&out)
            .iter()
            .any(|p| (norm(*p) - 1.0).abs() < 1e-5);
        assert!(landed, "outside point lands on the sphere");

        let mut oracle = sphere(1.0);
        let dropped = drape(
            &cloud,
            &mut oracle,
            &DrapeConfig {
                outside: OutsideConfig::Drop,
                ..config
            },
        )
        .unwrap();
        assert_eq!(dropped.element_count(), 1, "only the inside point stays");
    }

    #[test]
    fn bad_inputs_are_rejected() {
        // Volume meshes cannot drape.
        let hex = FeaMesh {
            element_kind: FeaElementKind::Hex8,
            node_positions: (0..8)
                .flat_map(|i| {
                    [
                        (i & 1) as f64,
                        ((i >> 1) & 1) as f64,
                        ((i >> 2) & 1) as f64,
                    ]
                })
                .collect(),
            connectivity: vec![0, 1, 3, 2, 4, 5, 7, 6],
            node_fields: vec![],
            element_fields: vec![],
        };
        let err = drape(&hex, &mut floor(), &DrapeConfig::default()).unwrap_err();
        assert!(err.contains("volume elements"), "unexpected error: {err}");

        // A mesh entirely out of reach fails loudly, not silently.
        let mesh = bar_mesh(&[[0.0, 0.0, 5.0], [0.4, 0.0, 5.0]], &[[0, 1]], 0.01);
        let err = drape(&mesh, &mut sphere(1.0), &DrapeConfig::default()).unwrap_err();
        assert!(err.contains("nothing survived"), "unexpected error: {err}");

        // Config validation.
        let bad = DrapeConfig {
            skin_radius_factor: 0.0,
            ..DrapeConfig::default()
        };
        assert!(validate_config(&bad).is_err());
        let bad = DrapeConfig {
            inset_factor: -1.0,
            ..DrapeConfig::default()
        };
        assert!(validate_config(&bad).is_err());
    }
}
