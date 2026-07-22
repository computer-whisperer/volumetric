//! The `surface` requirement: drape protruding struts onto the model.
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
//!   Dropped arcs whose endpoints already landed through a crossing are
//!   the reconnection pool: when the drop severs a component, the
//!   shortest such arcs re-drape (flagged `tie`) until the mesh is one
//!   piece again.
//!
//! Draped chords subdivide against the surface until they sag less than
//! `chord_tolerance` (default: the strut's own radius), so skin struts
//! follow curved surfaces as polylines. `inset_factor` sinks skin nodes
//! below the surface so the printed strut envelope sits flush instead of
//! half a strut proud. Nodes farther than `max_distance` from the
//! surface are dropped with their struts rather than dragged across
//! space (so box-mode Voronoi hairs vanish instead of piling onto the
//! skin).
//!
//! Projection runs on the binary occupancy oracle: estimate a surface
//! direction from a signed stencil of occupancy samples, march along it
//! to bracket the surface, and bisect — all in lock-step batched rounds,
//! one host call per round, like the clip operator's crossing bisection.

use std::collections::{HashMap, HashSet};

use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh};

use crate::{
    ConnectivityConfig, OccupiedBatch, OutsideConfig, SurfaceConfig, add, dist, lerp3, normalize,
    scale, sub, uf_find, uf_union,
};

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
pub(crate) struct Landing {
    pub pos: [f64; 3],
    /// Unit direction pointing into the model at the landing.
    pub inward: [f64; 3],
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
pub(crate) fn project_points(
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

    /// The landed output node for an input node, if a crossing fold
    /// already created one.
    fn try_moved(&self, node: u32) -> Option<u32> {
        let m = self.moved_map[node as usize];
        (m != u32::MAX).then_some(m)
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
/// (for element fields and tolerances), whether it is skin, and whether
/// it is a reconnection tie.
#[derive(Clone, Copy)]
struct Seg {
    a: u32,
    b: u32,
    origin: u32,
    skin: bool,
    tie: bool,
}

/// Drape subdivision: a chord whose midpoint hangs off the surface by
/// more than its tolerance splits at the midpoint's own landing.
fn subdivide_skin(
    builder: &mut Builder,
    mut active: Vec<Seg>,
    tol_for: &dyn Fn(u32, f64) -> f64,
    max_distance: f64,
    occupied: &mut OccupiedBatch,
) -> Result<Vec<Seg>, String> {
    let mut done: Vec<Seg> = Vec::new();
    for depth in 0..=MAX_DRAPE_DEPTH {
        if active.is_empty() {
            break;
        }
        if depth == MAX_DRAPE_DEPTH {
            done.append(&mut active);
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
                _ => done.push(*seg),
            }
        }
        active = next;
    }
    Ok(done)
}

fn drape_bars(
    mesh: &FeaMesh,
    node_occ: &[bool],
    occupied: &mut OccupiedBatch,
    config: &SurfaceConfig,
    reconnect: Option<&ConnectivityConfig>,
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
    // In drop mode the flat arcs vanish from the drape but stay
    // available as the reconnection pool.
    let dropped_flats: Vec<usize> = if config.outside == OutsideConfig::Drop {
        std::mem::take(&mut flats)
    } else {
        Vec::new()
    };

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
            tie: false,
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
                tie: false,
            });
            if let Some(l) = far {
                let m = builder.moved(x.out_node, &l);
                skin_active.push(Seg {
                    a: c,
                    b: m,
                    origin: x.element as u32,
                    skin: true,
                    tie: false,
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
                tie: false,
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
            tie: false,
        });
    }

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
    let mut skin_final = subdivide_skin(&mut builder, skin_active, &tol_for, max_distance, occupied)?;

    // Reconnect: if dropping the flat arcs severed the mesh, re-drape
    // the shortest dropped arcs whose endpoints already landed through
    // a crossing fold (shared rim vertices), spanning-forest style.
    let degenerate_eps = 1e-6 * max_distance;
    let mut tie_anchor_nodes: HashSet<u32> = HashSet::new();
    if reconnect.is_some_and(|r| r.fix == crate::ConnectivityFix::Reconnect)
        && !dropped_flats.is_empty()
    {
        let node_count = builder.sources.len() as u32;
        let mut parent: Vec<u32> = (0..node_count).collect();
        for s in &segs {
            uf_union(&mut parent, s.a, s.b);
        }
        // Skin segs count as connections only if they will survive the
        // degenerate filter below — a fold collapsing to a point must
        // not hide a severed component from the planner. (Anchors of
        // the ties planned here are protected from that filter, so a
        // degenerate fold a tie relies on does survive.)
        for s in &skin_final {
            if s.a != s.b && dist(builder.position(s.a), builder.position(s.b)) > degenerate_eps
            {
                uf_union(&mut parent, s.a, s.b);
            }
        }
        let mut candidates: Vec<(f64, usize, u32, u32)> = dropped_flats
            .iter()
            .filter_map(|&e| {
                let pair = mesh.element(e);
                let ia = builder.try_moved(pair[0])?;
                let ib = builder.try_moved(pair[1])?;
                (uf_find(&mut parent, ia) != uf_find(&mut parent, ib)).then(|| {
                    (
                        dist(builder.position(ia), builder.position(ib)),
                        e,
                        ia,
                        ib,
                    )
                })
            })
            .collect();
        candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut ties_active: Vec<Seg> = Vec::new();
        for &(_, e, ia, ib) in &candidates {
            if uf_union(&mut parent, ia, ib) {
                ties_active.push(Seg {
                    a: ia,
                    b: ib,
                    origin: e as u32,
                    skin: true,
                    tie: true,
                });
                tie_anchor_nodes.insert(ia);
                tie_anchor_nodes.insert(ib);
            }
        }
        let mut ties = subdivide_skin(&mut builder, ties_active, &tol_for, max_distance, occupied)?;
        skin_final.append(&mut ties);
    }

    // Degenerate folds (straight-out struts landing on themselves)
    // collapse to nothing even without a radius field to weld by —
    // except ties and the folds anchoring them, which must survive to
    // the weld so reconnection cannot be severed again.
    skin_final.retain(|s| {
        s.a != s.b
            && (s.tie
                || tie_anchor_nodes.contains(&s.a)
                || tie_anchor_nodes.contains(&s.b)
                || dist(builder.position(s.a), builder.position(s.b)) > degenerate_eps)
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
    set_flag_field(&mut element_fields, "skin", &segs, |s| s.skin)?;
    if reconnect.is_some() {
        set_flag_field(&mut element_fields, "tie", &segs, |s| s.tie)?;
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

/// Set (or create) a scalar 0/1 element field from a per-seg flag.
fn set_flag_field(
    element_fields: &mut Vec<FeaField>,
    name: &str,
    segs: &[Seg],
    flag: impl Fn(&Seg) -> bool,
) -> Result<(), String> {
    match element_fields.iter_mut().find(|f| f.name == name) {
        Some(f) if f.components == 1 => {
            for (k, s) in segs.iter().enumerate() {
                if flag(s) {
                    f.data[k] = 1.0;
                }
            }
            Ok(())
        }
        Some(_) => Err(format!(
            "the input already has a non-scalar '{name}' element field"
        )),
        None => {
            element_fields.push(FeaField {
                name: name.to_string(),
                components: 1,
                data: segs
                    .iter()
                    .map(|s| if flag(s) { 1.0 } else { 0.0 })
                    .collect(),
            });
            Ok(())
        }
    }
}

fn drape_points(
    mesh: &FeaMesh,
    node_occ: &[bool],
    occupied: &mut OccupiedBatch,
    config: &SurfaceConfig,
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

fn resolve_max_distance(mesh: &FeaMesh, config: &SurfaceConfig) -> Result<f64, String> {
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

/// The surface pass: classify nodes, resolve the reach, dispatch by
/// element kind. `reconnect` enables re-draping dropped arcs when the
/// connectivity requirement is present.
pub(crate) fn drape(
    mesh: &FeaMesh,
    occupied: &mut OccupiedBatch,
    config: &SurfaceConfig,
    reconnect: Option<&ConnectivityConfig>,
) -> Result<FeaMesh, String> {
    let max_distance = resolve_max_distance(mesh, config)?;
    let node_points: Vec<[f64; 3]> =
        (0..mesh.node_count()).map(|n| mesh.node_position(n)).collect();
    let node_occ = occupied(&node_points)?;
    match mesh.element_kind {
        FeaElementKind::Point1 => drape_points(mesh, &node_occ, occupied, config, max_distance),
        FeaElementKind::Bar2 => {
            drape_bars(mesh, &node_occ, occupied, config, reconnect, max_distance)
        }
        kind => Err(format!(
            "cannot drape {kind:?} volume elements onto a surface; clip them instead"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{bar_mesh, component_count, sphere};
    use crate::{ConnectivityFix, norm};

    /// Occupancy oracle for the half-space z < 0.
    fn floor() -> impl FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String> {
        move |points| Ok(points.iter().map(|p| p[2] < 0.0).collect())
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
            [0.0, 0.0, 1.5],  // outside, above the pole
            [0.9, 0.7, -0.4], // outside, oblique
            [0.0, 0.0, 0.4],  // inside
            [0.0, 0.0, 5.0],  // far out of reach
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
        let out = drape(&mesh, &mut oracle, &SurfaceConfig::default(), None).unwrap();

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
        let config = SurfaceConfig {
            skin_radius_factor: 0.5,
            ..SurfaceConfig::default()
        };
        let out = drape(&mesh, &mut oracle, &config, None).unwrap();

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
            [0.5, 0.0, 0.5],  // inside
            [1.2, 0.0, 0.5],  // outside (crossing partner)
            [0.0, 1.05, 0.3], // outside (flat strut)
            [0.3, 1.05, 0.3], // outside (flat strut)
        ];
        let bars = [[0, 1], [2, 3]];
        let mesh = bar_mesh(&nodes, &bars, 0.02);

        let mut oracle = sphere(1.0);
        let config = SurfaceConfig {
            outside: OutsideConfig::Drop,
            ..SurfaceConfig::default()
        };
        let dropped = drape(&mesh, &mut oracle, &config, None).unwrap();
        assert_eq!(
            dropped.element_count(),
            2,
            "stub + folded rim only: {:?}",
            field(&dropped, "skin").data
        );
        let skin = field(&dropped, "skin");
        assert_eq!(skin.data.iter().filter(|&&s| s == 1.0).count(), 1);

        let mut oracle = sphere(1.0);
        let projected = drape(&mesh, &mut oracle, &SurfaceConfig::default(), None).unwrap();
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
    fn dropped_arcs_reconnect_severed_components() {
        // Two inside clusters, each crossing out through a rim vertex,
        // tied together only by an outside arc between those vertices.
        // Drop severs them; reconnect re-drapes the dropped arc as a tie.
        let nodes = [
            [0.3, 0.0, 0.5],  // inside A
            [0.3, 0.0, 1.2],  // outside, rim vertex of A
            [-0.3, 0.0, 0.5], // inside B
            [-0.3, 0.0, 1.2], // outside, rim vertex of B
        ];
        let bars = [[0, 1], [2, 3], [1, 3]];
        let mesh = bar_mesh(&nodes, &bars, 0.02);
        let config = SurfaceConfig {
            outside: OutsideConfig::Drop,
            ..SurfaceConfig::default()
        };

        let mut oracle = sphere(1.0);
        let severed = drape(&mesh, &mut oracle, &config, None).unwrap();
        assert_eq!(component_count(&severed), 2, "drop alone severs the mesh");
        assert!(
            severed.element_fields.iter().all(|f| f.name != "tie"),
            "no tie field without the connectivity requirement"
        );

        let mut oracle = sphere(1.0);
        let reconnect = ConnectivityConfig::default();
        let out = drape(&mesh, &mut oracle, &config, Some(&reconnect)).unwrap();
        assert_eq!(component_count(&out), 1, "the tie rejoins the clusters");
        let tie = field(&out, "tie");
        let skin = field(&out, "skin");
        let ties: Vec<usize> = (0..out.element_count())
            .filter(|&e| tie.data[e] == 1.0)
            .collect();
        assert!(!ties.is_empty(), "the dropped arc came back as a tie");
        for &e in &ties {
            assert_eq!(skin.data[e], 1.0, "ties are draped skin");
        }
        // The tie hugs the sphere: every tie node sits on the surface.
        for &e in &ties {
            for &n in out.element(e) {
                let r = norm(out.node_position(n as usize));
                assert!((r - 1.0).abs() < 2e-2, "tie node off the surface: {r}");
            }
        }
        // Prune mode must NOT re-drape ties.
        let mut oracle = sphere(1.0);
        let prune = ConnectivityConfig {
            fix: ConnectivityFix::Prune,
            ..ConnectivityConfig::default()
        };
        let out = drape(&mesh, &mut oracle, &config, Some(&prune)).unwrap();
        assert_eq!(component_count(&out), 2, "prune leaves severing to connect::enforce");
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
        let config = SurfaceConfig {
            weld_factor: 0.0, // keep every node so sources stay inspectable
            ..SurfaceConfig::default()
        };
        let out = drape(&mesh, &mut oracle, &config, None).unwrap();

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
        let config = SurfaceConfig {
            inset_factor: 1.0,
            ..SurfaceConfig::default()
        };
        let out = drape(&mesh, &mut oracle, &config, None).unwrap();

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
        let config = SurfaceConfig {
            max_distance: 1.0,
            ..SurfaceConfig::default()
        };
        let mut oracle = sphere(1.0);
        let out = drape(&cloud, &mut oracle, &config, None).unwrap();
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
            &SurfaceConfig {
                outside: OutsideConfig::Drop,
                ..config
            },
            None,
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
        let err = drape(&hex, &mut floor(), &SurfaceConfig::default(), None).unwrap_err();
        assert!(err.contains("volume elements"), "unexpected error: {err}");

        // A mesh entirely out of reach fails loudly, not silently.
        let mesh = bar_mesh(&[[0.0, 0.0, 5.0], [0.4, 0.0, 5.0]], &[[0, 1]], 0.01);
        let err = drape(&mesh, &mut sphere(1.0), &SurfaceConfig::default(), None).unwrap_err();
        assert!(err.contains("nothing survived"), "unexpected error: {err}");
    }
}
