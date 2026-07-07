//! Stage 5 of the ASN2 pipeline: quadric-error-metric mesh decimation.
//!
//! Uniform-grid surface nets tile even flat or gently curved surfaces at the
//! finest cell pitch, so triangle counts grow with resolution² regardless of
//! how much geometric detail the surface actually carries — a 512³ lattice
//! mesh can reach tens of millions of triangles that no GPU buffer or STL
//! consumer wants. This pass collapses edges whose removal keeps the mesh
//! within a caller-supplied deviation budget, using Garland–Heckbert error
//! quadrics with *subset placement*: a collapse always moves one endpoint
//! onto the other, so every surviving vertex keeps the exact on-surface
//! position that stage-4 refinement gave it. Normals are re-accumulated
//! from the decimated topology at the end.
//!
//! Topology is preserved: every collapse must pass the link condition (the
//! one-ring intersection of the endpoints must be exactly the vertices
//! opposite the shared faces), which prevents thin lattice struts from
//! pinching off or fusing, plus a normal-flip guard against fold-overs that
//! is anchored to each face's orientation at decimation entry (so rotation
//! cannot accumulate across successive collapses).
//! Boundary edges (open surfaces where the model meets the sampling bounds)
//! are pinned by perpendicular constraint quadrics and may only collapse
//! along the boundary itself.
//!
//! The collapse schedule is threshold sweeps (in the spirit of Forstmann's
//! fast quadric simplification) rather than a global priority queue: passes
//! ramp the error threshold up to the budget, then repeat at the full budget
//! until no edge qualifies. Quality on near-uniform meshes matches greedy
//! ordering while staying O(faces) per pass with no heap.

use crate::adaptive_surface_nets_2::{IndexedMesh2, parallel_iter};
use std::sync::atomic::{AtomicBool, Ordering};
use web_time::Instant;

/// Configuration for the decimation post-pass.
#[derive(Clone, Debug, PartialEq)]
pub struct DecimationConfig {
    /// Maximum allowed surface deviation, as a fraction of the finest
    /// meshing cell size (so the budget self-scales with resolution).
    /// Larger values collapse more aggressively.
    pub error_tolerance_cells: f64,
    /// Number of threshold-ramp passes before the full-budget convergence
    /// passes. More passes prefer cheaper collapses first (slightly better
    /// quality, slightly slower).
    pub ramp_passes: usize,
}

impl Default for DecimationConfig {
    fn default() -> Self {
        Self {
            error_tolerance_cells: 1.0,
            ramp_passes: 6,
        }
    }
}

/// Statistics from one decimation run.
#[derive(Clone, Debug, Default)]
pub struct DecimationStats {
    pub time_secs: f64,
    pub passes_run: usize,
    pub vertices_before: usize,
    pub vertices_after: usize,
    pub triangles_before: usize,
    pub triangles_after: usize,
}

/// A plane-sum error quadric with an area-weight accumulator, so errors can
/// be read back as mean squared distance (in world units²) independent of
/// how much surface area contributed.
#[derive(Clone, Copy, Debug, Default)]
struct Quadric {
    // Symmetric 3x3 A (xx, xy, xz, yy, yz, zz), vector b, scalar c, weight.
    a: [f64; 6],
    b: [f64; 3],
    c: f64,
    w: f64,
}

impl Quadric {
    /// Accumulate the plane through `p` with unit normal `n`, weighted by `w`.
    fn add_plane(&mut self, n: [f64; 3], p: [f64; 3], w: f64) {
        let d = -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2]);
        self.a[0] += w * n[0] * n[0];
        self.a[1] += w * n[0] * n[1];
        self.a[2] += w * n[0] * n[2];
        self.a[3] += w * n[1] * n[1];
        self.a[4] += w * n[1] * n[2];
        self.a[5] += w * n[2] * n[2];
        self.b[0] += w * d * n[0];
        self.b[1] += w * d * n[1];
        self.b[2] += w * d * n[2];
        self.c += w * d * d;
        self.w += w;
    }

    fn add(&mut self, other: &Quadric) {
        for i in 0..6 {
            self.a[i] += other.a[i];
        }
        for i in 0..3 {
            self.b[i] += other.b[i];
        }
        self.c += other.c;
        self.w += other.w;
    }

    /// Mean squared plane distance of `p` (raw quadric error / total weight).
    fn mean_sq_error(&self, p: [f64; 3]) -> f64 {
        if self.w <= 0.0 {
            return 0.0;
        }
        let quad = self.a[0] * p[0] * p[0]
            + 2.0 * self.a[1] * p[0] * p[1]
            + 2.0 * self.a[2] * p[0] * p[2]
            + self.a[3] * p[1] * p[1]
            + 2.0 * self.a[4] * p[1] * p[2]
            + self.a[5] * p[2] * p[2];
        let lin = 2.0 * (self.b[0] * p[0] + self.b[1] * p[1] + self.b[2] * p[2]);
        // Clamp: the form is positive semi-definite, tiny negatives are
        // floating-point noise.
        ((quad + lin + self.c) / self.w).max(0.0)
    }
}

/// Extra weight multiplier for boundary constraint planes: deviations from an
/// open border dominate the mean error, effectively pinning the border shape.
const BOUNDARY_WEIGHT: f64 = 100.0;

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let len = dot(v, v).sqrt();
    if len <= 0.0 {
        return None;
    }
    Some([v[0] / len, v[1] / len, v[2] / len])
}

#[inline]
fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((hi as u64) << 32) | lo as u64
}

/// Compressed vertex→face adjacency, rebuilt once per pass.
struct VertexFaces {
    starts: Vec<u32>,
    faces: Vec<u32>,
}

impl VertexFaces {
    fn build(faces: &[[u32; 3]], deleted: &[bool], vertex_count: usize) -> Self {
        let mut counts = vec![0u32; vertex_count + 1];
        for (fi, face) in faces.iter().enumerate() {
            if deleted[fi] {
                continue;
            }
            for &v in face {
                counts[v as usize + 1] += 1;
            }
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        let starts = counts.clone();
        let mut cursor = starts.clone();
        let mut refs = vec![0u32; starts[vertex_count] as usize];
        for (fi, face) in faces.iter().enumerate() {
            if deleted[fi] {
                continue;
            }
            for &v in face {
                refs[cursor[v as usize] as usize] = fi as u32;
                cursor[v as usize] += 1;
            }
        }
        Self {
            starts,
            faces: refs,
        }
    }

    #[inline]
    fn of(&self, v: u32) -> &[u32] {
        &self.faces[self.starts[v as usize] as usize..self.starts[v as usize + 1] as usize]
    }
}

/// Mutable working state for collapse sweeps over one face set (the whole
/// mesh, or one spatial region of it on the parallel path).
struct SweepMesh {
    positions: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
    /// Each face's unit normal at decimation entry (zero for degenerate
    /// faces), index-aligned with `faces`. The flip guard anchors to these so
    /// rotation cannot accumulate across successive collapses.
    original_normals: Vec<[f64; 3]>,
    deleted: Vec<bool>,
    quadrics: Vec<Quadric>,
    boundary_edges: std::collections::HashSet<u64>,
    boundary_vertex: Vec<bool>,
    /// Pinned vertices take part in no collapse at all (neither endpoint).
    /// The parallel path pins region-seam vertices: with both endpoints
    /// unpinned, every face adjacent to either endpoint is inside the
    /// region, so link-condition and flip-guard reads are complete and no
    /// cross-region state is touched. Empty = nothing pinned.
    pinned: Vec<bool>,
}

impl SweepMesh {
    /// Run collapse passes: `ramp_passes` sweeps approaching the budget from
    /// below (cheap collapses first), then full-budget sweeps until yields
    /// go quiet. Returns the number of passes run. A cancellation stops
    /// between passes, leaving a valid (partially decimated) face set.
    fn run_passes(&mut self, budget: f64, ramp_passes: usize, cancel: &AtomicBool) -> usize {
        let vertex_count = self.positions.len();
        // Reused scratch buffers for the link-condition neighborhood sets.
        let mut neighbors_src: Vec<u32> = Vec::new();
        let mut neighbors_dst: Vec<u32> = Vec::new();
        let mut opposite: Vec<u32> = Vec::new();

        const MAX_CONVERGENCE_PASSES: usize = 16;
        let mut pass = 0;
        loop {
            if cancel.load(Ordering::Relaxed) {
                return pass;
            }
            let threshold = if pass < ramp_passes {
                let t = (pass + 1) as f64 / (ramp_passes + 1) as f64;
                budget * t * t
            } else {
                budget
            };

            let adjacency = VertexFaces::build(&self.faces, &self.deleted, vertex_count);
            let live_faces = adjacency.faces.len() / 3;
            let mut dirty = vec![false; vertex_count];
            let mut collapses = 0usize;

            for fi in 0..self.faces.len() {
                for k in 0..3 {
                    if self.deleted[fi] {
                        break;
                    }
                    let a = self.faces[fi][k];
                    let b = self.faces[fi][(k + 1) % 3];
                    if a == b || dirty[a as usize] || dirty[b as usize] {
                        continue;
                    }
                    if !self.pinned.is_empty()
                        && (self.pinned[a as usize] || self.pinned[b as usize])
                    {
                        continue;
                    }

                    // Boundary rule: a boundary vertex may only slide along a
                    // boundary edge (which implies the target is boundary too).
                    let edge_is_boundary = !self.boundary_edges.is_empty()
                        && self.boundary_edges.contains(&edge_key(a, b));
                    let can_move =
                        |src: u32| !self.boundary_vertex[src as usize] || edge_is_boundary;

                    // Subset placement: try the cheaper direction first.
                    let err_a_to_b = combined_error(&self.quadrics, a, b, &self.positions);
                    let err_b_to_a = combined_error(&self.quadrics, b, a, &self.positions);
                    let directions = if err_a_to_b <= err_b_to_a {
                        [(a, b, err_a_to_b), (b, a, err_b_to_a)]
                    } else {
                        [(b, a, err_b_to_a), (a, b, err_a_to_b)]
                    };
                    let chosen = directions.into_iter().find_map(|(src, dst, err)| {
                        (err <= threshold
                            && can_move(src)
                            && link_condition_holds(
                                src,
                                dst,
                                &self.faces,
                                &self.deleted,
                                &adjacency,
                                &mut neighbors_src,
                                &mut neighbors_dst,
                                &mut opposite,
                            )
                            && !collapse_flips_normals(
                                src,
                                dst,
                                &self.faces,
                                &self.original_normals,
                                &self.deleted,
                                &adjacency,
                                &self.positions,
                            ))
                        .then_some((src, dst))
                    });

                    if let Some((src, dst)) = chosen {
                        self.quadrics[dst as usize] = {
                            let mut q = self.quadrics[dst as usize];
                            q.add(&self.quadrics[src as usize]);
                            q
                        };
                        for &other_fi in adjacency.of(src) {
                            let other_fi = other_fi as usize;
                            if self.deleted[other_fi] {
                                continue;
                            }
                            if self.faces[other_fi].contains(&dst) {
                                self.deleted[other_fi] = true;
                            } else {
                                for slot in self.faces[other_fi].iter_mut() {
                                    if *slot == src {
                                        *slot = dst;
                                    }
                                }
                            }
                        }
                        dirty[src as usize] = true;
                        dirty[dst as usize] = true;
                        collapses += 1;
                    }
                }
            }

            pass += 1;
            let at_full_budget = pass > ramp_passes;
            // Stop when a full-budget pass yields almost nothing: trailing
            // convergence passes each sweep every live face for <0.2% gains.
            if (at_full_budget && collapses * 500 < live_faces)
                || pass >= ramp_passes + MAX_CONVERGENCE_PASSES
            {
                break;
            }
        }
        pass
    }
}

/// Single-threaded decimation over the whole face set.
#[allow(clippy::type_complexity)]
fn serial_decimate(
    positions: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
    face_normals: Vec<[f64; 3]>,
    quadrics: Vec<Quadric>,
    boundary_edges: std::collections::HashSet<u64>,
    boundary_vertex: Vec<bool>,
    budget: f64,
    ramp_passes: usize,
    cancel: &AtomicBool,
) -> (Vec<[u32; 3]>, Vec<bool>, usize) {
    let deleted = vec![false; faces.len()];
    let mut sweep = SweepMesh {
        positions,
        faces,
        original_normals: face_normals,
        deleted,
        quadrics,
        boundary_edges,
        boundary_vertex,
        pinned: Vec::new(),
    };
    let passes = sweep.run_passes(budget, ramp_passes, cancel);
    (sweep.faces, sweep.deleted, passes)
}

/// Parallel decimation: partition faces by centroid into a spatial grid, pin
/// every vertex whose faces span more than one region, decimate the regions
/// concurrently (pinned vertices are untouchable, so each region's reads and
/// writes stay provably inside the region), then merge and run serial
/// full-budget passes to dissolve the frozen grid-pitch seams.
#[cfg(feature = "native")]
#[allow(clippy::type_complexity)]
fn parallel_decimate(
    positions: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
    face_normals: Vec<[f64; 3]>,
    mut quadrics: Vec<Quadric>,
    boundary_edges: std::collections::HashSet<u64>,
    boundary_vertex: Vec<bool>,
    budget: f64,
    ramp_passes: usize,
    cancel: &AtomicBool,
) -> (Vec<[u32; 3]>, Vec<bool>, usize) {
    let vertex_count = positions.len();

    // ~8 buckets per worker so uneven surface density still load-balances.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let grid = (((workers * 8) as f64).cbrt().ceil() as usize).clamp(2, 8);

    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in &positions {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    let inv_extent: [f64; 3] =
        std::array::from_fn(|axis| grid as f64 / (hi[axis] - lo[axis]).max(1e-12));

    // Faces and their entry normals travel together through bucketing,
    // region sweeps, and the merge, staying index-aligned throughout.
    let mut buckets: Vec<(Vec<[u32; 3]>, Vec<[f64; 3]>)> =
        vec![(Vec::new(), Vec::new()); grid * grid * grid];
    for (face, normal) in faces.into_iter().zip(face_normals) {
        let mut cell = [0usize; 3];
        for (axis, slot) in cell.iter_mut().enumerate() {
            let centroid = (positions[face[0] as usize][axis]
                + positions[face[1] as usize][axis]
                + positions[face[2] as usize][axis])
                / 3.0;
            *slot = (((centroid - lo[axis]) * inv_extent[axis]) as usize).min(grid - 1);
        }
        let bucket = &mut buckets[cell[0] + grid * (cell[1] + grid * cell[2])];
        bucket.0.push(face);
        bucket.1.push(normal);
    }

    // A vertex whose faces span more than one bucket is a seam vertex.
    const UNSEEN: u32 = u32::MAX;
    const SHARED: u32 = u32::MAX - 1;
    let mut vertex_bucket = vec![UNSEEN; vertex_count];
    for (bi, (bucket_faces, _)) in buckets.iter().enumerate() {
        for face in bucket_faces {
            for &v in face {
                let slot = &mut vertex_bucket[v as usize];
                if *slot == UNSEEN {
                    *slot = bi as u32;
                } else if *slot != bi as u32 {
                    *slot = SHARED;
                }
            }
        }
    }

    let non_empty: Vec<(Vec<[u32; 3]>, Vec<[f64; 3]>)> = buckets
        .into_iter()
        .filter(|(faces, _)| !faces.is_empty())
        .collect();
    let results = parallel_iter::map_vec(non_empty, |(bucket_faces, bucket_normals)| {
        let mut local_to_global: Vec<u32> = bucket_faces.iter().flatten().copied().collect();
        local_to_global.sort_unstable();
        local_to_global.dedup();
        let to_local = |g: u32| {
            local_to_global
                .binary_search(&g)
                .expect("bucket vertex must be in its own vertex list") as u32
        };
        let local_faces: Vec<[u32; 3]> = bucket_faces.iter().map(|f| f.map(to_local)).collect();
        let local_boundary_edges = if boundary_edges.is_empty() {
            std::collections::HashSet::new()
        } else {
            let mut local = std::collections::HashSet::new();
            for (face, global_face) in local_faces.iter().zip(&bucket_faces) {
                for k in 0..3 {
                    if boundary_edges.contains(&edge_key(global_face[k], global_face[(k + 1) % 3]))
                    {
                        local.insert(edge_key(face[k], face[(k + 1) % 3]));
                    }
                }
            }
            local
        };
        let deleted = vec![false; local_faces.len()];
        let mut sweep = SweepMesh {
            positions: local_to_global
                .iter()
                .map(|&g| positions[g as usize])
                .collect(),
            faces: local_faces,
            original_normals: bucket_normals,
            deleted,
            quadrics: local_to_global
                .iter()
                .map(|&g| quadrics[g as usize])
                .collect(),
            boundary_edges: local_boundary_edges,
            boundary_vertex: local_to_global
                .iter()
                .map(|&g| boundary_vertex[g as usize])
                .collect(),
            pinned: local_to_global
                .iter()
                .map(|&g| vertex_bucket[g as usize] == SHARED)
                .collect(),
        };
        let passes = sweep.run_passes(budget, ramp_passes, cancel);
        let mut survivors: Vec<[u32; 3]> = Vec::new();
        let mut survivor_normals: Vec<[f64; 3]> = Vec::new();
        for (fi, face) in sweep.faces.iter().enumerate() {
            if !sweep.deleted[fi] {
                survivors.push(face.map(|l| local_to_global[l as usize]));
                survivor_normals.push(sweep.original_normals[fi]);
            }
        }
        (
            survivors,
            survivor_normals,
            local_to_global,
            sweep.quadrics,
            passes,
        )
    });

    // Merge: survivors concatenate in deterministic region order; unpinned
    // vertices are exclusive to one region, so their evolved quadrics write
    // straight back (seam vertices never collapsed, theirs are unchanged).
    let mut merged: Vec<[u32; 3]> = Vec::new();
    let mut merged_normals: Vec<[f64; 3]> = Vec::new();
    let mut region_passes = 0usize;
    for (survivors, survivor_normals, local_to_global, local_quadrics, passes) in results {
        region_passes = region_passes.max(passes);
        for (local, &global) in local_to_global.iter().enumerate() {
            if vertex_bucket[global as usize] != SHARED {
                quadrics[global as usize] = local_quadrics[local];
            }
        }
        merged.extend(survivors);
        merged_normals.extend(survivor_normals);
    }

    // Final serial passes dissolve the seams. No ramp: seam neighborhoods
    // are still at grid pitch, and everything else is already converged.
    let deleted = vec![false; merged.len()];
    let mut sweep = SweepMesh {
        positions,
        faces: merged,
        original_normals: merged_normals,
        deleted,
        quadrics,
        boundary_edges,
        boundary_vertex,
        pinned: Vec::new(),
    };
    let final_passes = sweep.run_passes(budget, 0, cancel);
    (sweep.faces, sweep.deleted, region_passes + final_passes)
}

/// Face count above which the native build partitions the mesh into spatial
/// regions and decimates them in parallel (seams frozen, then dissolved by a
/// final serial pass). Below it the partition/merge overhead isn't worth it.
const PARALLEL_FACE_THRESHOLD: usize = 200_000;

/// Decimate `mesh` in place so surviving geometry stays within roughly
/// `error_tolerance` (world units, mean plane distance) of the input surface.
/// Vertex normals are re-accumulated from the decimated topology.
pub fn decimate_mesh(
    mesh: &mut IndexedMesh2,
    error_tolerance: f64,
    ramp_passes: usize,
) -> DecimationStats {
    static NEVER: AtomicBool = AtomicBool::new(false);
    decimate_mesh_cancellable(mesh, error_tolerance, ramp_passes, &NEVER)
}

/// [`decimate_mesh`], checking `cancel` between collapse passes. On
/// cancellation the remaining passes are skipped but the mesh is still
/// compacted, so it stays valid (just less decimated than requested);
/// callers that cancel are expected to discard the result anyway.
pub fn decimate_mesh_cancellable(
    mesh: &mut IndexedMesh2,
    error_tolerance: f64,
    ramp_passes: usize,
    cancel: &AtomicBool,
) -> DecimationStats {
    decimate_mesh_impl(
        mesh,
        error_tolerance,
        ramp_passes,
        PARALLEL_FACE_THRESHOLD,
        cancel,
    )
}

fn decimate_mesh_impl(
    mesh: &mut IndexedMesh2,
    error_tolerance: f64,
    ramp_passes: usize,
    parallel_threshold: usize,
    cancel: &AtomicBool,
) -> DecimationStats {
    let start = Instant::now();
    let vertices_before = mesh.vertices.len();
    let triangles_before = mesh.indices.len() / 3;
    let mut stats = DecimationStats {
        vertices_before,
        vertices_after: vertices_before,
        triangles_before,
        triangles_after: triangles_before,
        ..Default::default()
    };
    if triangles_before == 0 || error_tolerance <= 0.0 {
        stats.time_secs = start.elapsed().as_secs_f64();
        return stats;
    }

    let positions: Vec<[f64; 3]> = mesh
        .vertices
        .iter()
        .map(|&(x, y, z)| [x as f64, y as f64, z as f64])
        .collect();
    let vertex_count = positions.len();

    let mut faces: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    // Stage 2 collects triangles in a nondeterministic parallel order; sort so
    // the collapse sequence (and thus the output) is deterministic.
    parallel_iter::sort_unstable(&mut faces);

    // Initial quadrics: every face contributes its plane, weighted by area.
    // The unit normals are also kept per face (zero when degenerate) as the
    // flip guard's anchor orientation.
    let mut quadrics = vec![Quadric::default(); vertex_count];
    let mut face_normals = vec![[0.0f64; 3]; faces.len()];
    for (fi, face) in faces.iter().enumerate() {
        let [i0, i1, i2] = *face;
        let (p0, p1, p2) = (
            positions[i0 as usize],
            positions[i1 as usize],
            positions[i2 as usize],
        );
        let n = cross(sub(p1, p0), sub(p2, p0));
        let double_area = dot(n, n).sqrt();
        if double_area <= 0.0 {
            continue;
        }
        let unit_n = [n[0] / double_area, n[1] / double_area, n[2] / double_area];
        face_normals[fi] = unit_n;
        let w = double_area * 0.5;
        quadrics[i0 as usize].add_plane(unit_n, p0, w);
        quadrics[i1 as usize].add_plane(unit_n, p0, w);
        quadrics[i2 as usize].add_plane(unit_n, p0, w);
    }

    // Boundary detection: edges with exactly one incident face. ASN2 meshes
    // are closed unless the model touches the sampling bounds, so this set is
    // usually empty.
    let boundary_edges = find_boundary_edges(&faces);
    let mut boundary_vertex = vec![false; vertex_count];
    if !boundary_edges.is_empty() {
        for &key in &boundary_edges {
            boundary_vertex[(key & 0xffff_ffff) as usize] = true;
            boundary_vertex[(key >> 32) as usize] = true;
        }
        // Pin each boundary edge with a heavily weighted plane through the
        // edge, perpendicular to its face.
        for face in &faces {
            for k in 0..3 {
                let a = face[k];
                let b = face[(k + 1) % 3];
                if !boundary_edges.contains(&edge_key(a, b)) {
                    continue;
                }
                let (pa, pb) = (positions[a as usize], positions[b as usize]);
                let pc = positions[face[(k + 2) % 3] as usize];
                let edge = sub(pb, pa);
                let face_n = cross(edge, sub(pc, pa));
                if let Some(constraint_n) = normalize(cross(edge, face_n)) {
                    let w = dot(edge, edge) * BOUNDARY_WEIGHT;
                    quadrics[a as usize].add_plane(constraint_n, pa, w);
                    quadrics[b as usize].add_plane(constraint_n, pa, w);
                }
            }
        }
    }

    let budget = error_tolerance * error_tolerance;

    #[cfg(feature = "native")]
    let (faces, deleted, passes_run) = if faces.len() >= parallel_threshold {
        parallel_decimate(
            positions,
            faces,
            face_normals,
            quadrics,
            boundary_edges,
            boundary_vertex,
            budget,
            ramp_passes,
            cancel,
        )
    } else {
        serial_decimate(
            positions,
            faces,
            face_normals,
            quadrics,
            boundary_edges,
            boundary_vertex,
            budget,
            ramp_passes,
            cancel,
        )
    };
    #[cfg(not(feature = "native"))]
    let (faces, deleted, passes_run) = {
        let _ = parallel_threshold;
        serial_decimate(
            positions,
            faces,
            face_normals,
            quadrics,
            boundary_edges,
            boundary_vertex,
            budget,
            ramp_passes,
            cancel,
        )
    };
    stats.passes_run = passes_run;

    // Compact: drop deleted faces and remap surviving vertices. Positions
    // carry through unchanged (subset placement), but normals must be
    // recomputed from the new topology: a survivor's old normal was
    // accumulated over its dense grid-pitch one-ring, and interpolating it
    // across the much larger collapsed faces shades near-flat regions as
    // torn angular sheets.
    let mut remap = vec![u32::MAX; vertex_count];
    let mut new_vertices = Vec::new();
    let mut new_indices = Vec::with_capacity(faces.len() * 3);
    for (fi, face) in faces.iter().enumerate() {
        if deleted[fi] {
            continue;
        }
        for &v in face {
            let slot = &mut remap[v as usize];
            if *slot == u32::MAX {
                *slot = new_vertices.len() as u32;
                new_vertices.push(mesh.vertices[v as usize]);
            }
            new_indices.push(*slot);
        }
    }
    let positions_f64: Vec<(f64, f64, f64)> = new_vertices
        .iter()
        .map(|&(x, y, z)| (x as f64, y as f64, z as f64))
        .collect();
    let accumulated =
        crate::adaptive_surface_nets_2::recompute_accumulated_normals(&positions_f64, &new_indices);
    mesh.normals = accumulated
        .into_iter()
        .map(crate::adaptive_surface_nets_2::normalize_or_default)
        .collect();
    mesh.vertices = new_vertices;
    mesh.indices = new_indices;

    stats.vertices_after = mesh.vertices.len();
    stats.triangles_after = mesh.indices.len() / 3;
    stats.time_secs = start.elapsed().as_secs_f64();
    stats
}

/// Error of collapsing src onto dst: the worse of the two endpoint quadrics,
/// each normalized by its own accumulated weight. Normalizing per-quadric
/// (instead of over the combined weight) is essential: a large flat region
/// has enormous accumulated area at zero error, and combined-weight
/// normalization would dilute a small feature vertex's planes to nothing —
/// letting flat plates swallow the dimples and pores embedded in them.
#[inline]
fn combined_error(quadrics: &[Quadric], src: u32, dst: u32, positions: &[[f64; 3]]) -> f64 {
    let p = positions[dst as usize];
    quadrics[src as usize]
        .mean_sq_error(p)
        .max(quadrics[dst as usize].mean_sq_error(p))
}

/// Edges incident to exactly one live face, as `edge_key`s.
fn find_boundary_edges(faces: &[[u32; 3]]) -> std::collections::HashSet<u64> {
    let mut keys: Vec<u64> = Vec::with_capacity(faces.len() * 3);
    for face in faces {
        keys.push(edge_key(face[0], face[1]));
        keys.push(edge_key(face[1], face[2]));
        keys.push(edge_key(face[2], face[0]));
    }
    parallel_iter::sort_unstable(&mut keys);
    let mut boundary = std::collections::HashSet::new();
    let mut i = 0;
    while i < keys.len() {
        let mut j = i + 1;
        while j < keys.len() && keys[j] == keys[i] {
            j += 1;
        }
        if j - i == 1 {
            boundary.insert(keys[i]);
        }
        i = j;
    }
    boundary
}

/// The link condition: collapsing (src → dst) is topology-safe iff the
/// common neighbors of src and dst are exactly the vertices opposite the
/// faces shared by the edge. Violations pinch the surface into non-manifold
/// junctions (e.g. merging the two sides of a thin strut).
#[allow(clippy::too_many_arguments)]
fn link_condition_holds(
    src: u32,
    dst: u32,
    faces: &[[u32; 3]],
    deleted: &[bool],
    adjacency: &VertexFaces,
    neighbors_src: &mut Vec<u32>,
    neighbors_dst: &mut Vec<u32>,
    opposite: &mut Vec<u32>,
) -> bool {
    collect_neighbors(src, faces, deleted, adjacency, neighbors_src);
    collect_neighbors(dst, faces, deleted, adjacency, neighbors_dst);

    opposite.clear();
    for &fi in adjacency.of(src) {
        if deleted[fi as usize] {
            continue;
        }
        let face = faces[fi as usize];
        if face.contains(&dst) {
            for &v in &face {
                if v != src && v != dst {
                    opposite.push(v);
                }
            }
        }
    }
    if opposite.is_empty() {
        return false;
    }
    opposite.sort_unstable();
    opposite.dedup();

    // shared = neighbors(src) ∩ neighbors(dst); both are sorted + deduped.
    let mut shared_count = 0usize;
    let (mut i, mut j) = (0usize, 0usize);
    while i < neighbors_src.len() && j < neighbors_dst.len() {
        match neighbors_src[i].cmp(&neighbors_dst[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                if !opposite.contains(&neighbors_src[i]) {
                    return false;
                }
                shared_count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    shared_count == opposite.len()
}

fn collect_neighbors(
    v: u32,
    faces: &[[u32; 3]],
    deleted: &[bool],
    adjacency: &VertexFaces,
    out: &mut Vec<u32>,
) {
    out.clear();
    for &fi in adjacency.of(v) {
        if deleted[fi as usize] {
            continue;
        }
        for &other in &faces[fi as usize] {
            if other != v {
                out.push(other);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
}

/// Reject collapses that fold any surviving face of `src` past perpendicular
/// (or squash it to zero area) when `src` moves onto `dst`.
///
/// Two references are checked: the face's orientation just before this
/// collapse, and its orientation at decimation entry (`original_normals`).
/// The per-collapse check alone is not enough — each collapse may rotate a
/// face by up to just-under-90°, so a face touched by several successive
/// collapses could accumulate an arbitrary total rotation and end up facing
/// back into the surface. Sub-cell foam walls made that real: decimation
/// minted thousands of near-coplanar flipped slivers on lattice meshes whose
/// input had none. Anchoring to the stage-4 orientation caps the cumulative
/// rotation at 90°, which legitimate coarsening stays well inside.
#[allow(clippy::too_many_arguments)]
fn collapse_flips_normals(
    src: u32,
    dst: u32,
    faces: &[[u32; 3]],
    original_normals: &[[f64; 3]],
    deleted: &[bool],
    adjacency: &VertexFaces,
    positions: &[[f64; 3]],
) -> bool {
    let dst_pos = positions[dst as usize];
    for &fi in adjacency.of(src) {
        if deleted[fi as usize] {
            continue;
        }
        let face = faces[fi as usize];
        if face.contains(&dst) {
            continue; // This face is removed by the collapse.
        }
        let old: [[f64; 3]; 3] = [
            positions[face[0] as usize],
            positions[face[1] as usize],
            positions[face[2] as usize],
        ];
        let mut new = old;
        for (slot, &v) in new.iter_mut().zip(&face) {
            if v == src {
                *slot = dst_pos;
            }
        }
        let n_old = cross(sub(old[1], old[0]), sub(old[2], old[0]));
        let n_new = cross(sub(new[1], new[0]), sub(new[2], new[0]));
        if dot(n_new, n_new) <= 0.0 || dot(n_old, n_new) <= 0.0 {
            return true;
        }
        // Anchored check. A zero original normal (degenerate face at entry)
        // imposes no constraint.
        let orig = original_normals[fi as usize];
        if dot(orig, n_new) <= 0.0 && dot(orig, orig) > 0.0 {
            return true;
        }
    }
    false
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    static NEVER: AtomicBool = AtomicBool::new(false);
    use crate::adaptive_surface_nets_2::{AdaptiveMeshConfig2, adaptive_surface_nets_2};

    fn mesh_sampler<F>(sampler: F, depth: usize) -> IndexedMesh2
    where
        F: Fn(f64, f64, f64) -> f32 + Send + Sync,
    {
        let config = AdaptiveMeshConfig2 {
            base_resolution: 8,
            max_depth: depth,
            vertex_refinement_iterations: 8,
            normal_sample_iterations: 0,
            edge_constrained_refinement: true,
            sharp_features: None,
            ..Default::default()
        };
        adaptive_surface_nets_2(sampler, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), &config).mesh
    }

    /// cell size for the [-1,1]³ test bounds at `depth`.
    fn cell(depth: usize) -> f64 {
        2.0 / (8 << depth) as f64
    }

    /// Map of edge → incident face count.
    fn edge_face_counts(indices: &[u32]) -> std::collections::HashMap<u64, usize> {
        let mut counts = std::collections::HashMap::new();
        for t in indices.chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *counts.entry(edge_key(a, b)).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Number of connected components of the triangle graph (via shared
    /// vertices).
    fn component_count(vertex_count: usize, indices: &[u32]) -> usize {
        let mut parent: Vec<u32> = (0..vertex_count as u32).collect();
        fn find(parent: &mut [u32], v: u32) -> u32 {
            let mut root = v;
            while parent[root as usize] != root {
                root = parent[root as usize];
            }
            let mut cur = v;
            while parent[cur as usize] != root {
                let next = parent[cur as usize];
                parent[cur as usize] = root;
                cur = next;
            }
            root
        }
        for t in indices.chunks_exact(3) {
            let r0 = find(&mut parent, t[0]);
            let r1 = find(&mut parent, t[1]);
            let r2 = find(&mut parent, t[2]);
            parent[r1 as usize] = r0;
            parent[r2 as usize] = r0;
        }
        let mut used: Vec<bool> = vec![false; vertex_count];
        for &i in indices {
            used[i as usize] = true;
        }
        let mut roots = std::collections::HashSet::new();
        for v in 0..vertex_count as u32 {
            if used[v as usize] {
                roots.insert(find(&mut parent, v));
            }
        }
        roots.len()
    }

    /// A pre-set cancel flag skips every collapse pass: the mesh survives
    /// unchanged (and valid) with zero passes run.
    #[test]
    fn pre_cancelled_decimation_is_a_noop() {
        let mut mesh = mesh_sampler(
            |x, y, z| {
                if x * x + y * y + z * z < 0.64 {
                    1.0
                } else {
                    0.0
                }
            },
            3,
        );
        let before = mesh.indices.len() / 3;
        let cancel = AtomicBool::new(true);
        let stats = decimate_mesh_cancellable(&mut mesh, cell(3), 6, &cancel);
        assert_eq!(stats.passes_run, 0);
        assert_eq!(stats.triangles_after, before, "no collapses ran");
        assert_eq!(mesh.indices.len() / 3, before);
    }

    #[test]
    fn sphere_decimation_reduces_triangles_and_stays_on_surface() {
        let radius = 0.8;
        let mut mesh = mesh_sampler(
            move |x, y, z| {
                if x * x + y * y + z * z < radius * radius {
                    1.0
                } else {
                    0.0
                }
            },
            3,
        );
        let before = mesh.indices.len() / 3;
        let stats = decimate_mesh(&mut mesh, cell(3), 6);
        let after = mesh.indices.len() / 3;
        assert_eq!(stats.triangles_before, before);
        assert_eq!(stats.triangles_after, after);
        assert!(
            after * 3 < before,
            "expected at least 3x reduction, got {before} -> {after}"
        );

        // Subset placement: every surviving vertex still sits on the sampled
        // sphere surface (to refinement precision).
        let refine_precision = cell(3) / 128.0 + 1e-6;
        for &(x, y, z) in &mesh.vertices {
            let r = ((x as f64).powi(2) + (y as f64).powi(2) + (z as f64).powi(2)).sqrt();
            assert!(
                (r - radius).abs() < refine_precision + cell(3) * 0.05,
                "vertex left the surface: r = {r}"
            );
        }

        // Triangle centroids stay within the deviation budget of the sphere.
        let tolerance = cell(3);
        for t in mesh.indices.chunks_exact(3) {
            let mut c = [0.0f64; 3];
            for &i in t {
                let (x, y, z) = mesh.vertices[i as usize];
                c[0] += x as f64 / 3.0;
                c[1] += y as f64 / 3.0;
                c[2] += z as f64 / 3.0;
            }
            let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            assert!(
                (r - radius).abs() < 2.5 * tolerance,
                "centroid deviated {} (budget {})",
                (r - radius).abs(),
                tolerance
            );
        }

        // The sphere is closed and must stay a closed 2-manifold.
        for (_, count) in edge_face_counts(&mesh.indices) {
            assert_eq!(count, 2, "decimation opened or pinched the surface");
        }
    }

    #[test]
    fn thin_strut_keeps_topology_and_boundary() {
        // A thin cylinder along z spanning the whole box: ~5 cells across
        // at depth 3, with open boundary rings at the bounds.
        let radius = 0.08;
        let mut mesh = mesh_sampler(
            move |x, y, _z| {
                if x * x + y * y < radius * radius {
                    1.0
                } else {
                    0.0
                }
            },
            3,
        );
        let before_tris = mesh.indices.len() / 3;
        let before_components = component_count(mesh.vertices.len(), &mesh.indices);
        let before_boundary = edge_face_counts(&mesh.indices)
            .values()
            .filter(|&&c| c == 1)
            .count();
        assert_eq!(before_components, 1);
        assert!(before_boundary > 0, "expected open rings at the bounds");

        let stats = decimate_mesh(&mut mesh, cell(3), 6);
        assert!(stats.triangles_after < before_tris);

        // Still one connected tube — no pinch-off, no fusing.
        assert_eq!(component_count(mesh.vertices.len(), &mesh.indices), 1);
        // Still an open tube: boundary rings survive (fewer edges is fine).
        let counts = edge_face_counts(&mesh.indices);
        let after_boundary = counts.values().filter(|&&c| c == 1).count();
        assert!(after_boundary >= 6, "boundary rings vanished");
        assert!(
            counts.values().all(|&c| c <= 2),
            "non-manifold edge created"
        );

        // Every vertex is still on the cylinder wall or its boundary rings.
        for &(x, y, z) in &mesh.vertices {
            let r = ((x as f64).powi(2) + (y as f64).powi(2)).sqrt();
            let on_wall = (r - radius).abs() < cell(3) * 0.2;
            let on_ring = (z as f64).abs() > 1.0 - 1e-6;
            assert!(on_wall || on_ring, "vertex strayed: r={r} z={z}");
        }
    }

    #[test]
    fn flat_surfaces_collapse_aggressively() {
        // An axis-aligned slab: large flat faces should collapse far past the
        // generic 3x reduction.
        let mut mesh = mesh_sampler(
            |x, y, z| {
                if x.abs() < 0.7 && y.abs() < 0.7 && z.abs() < 0.3 {
                    1.0
                } else {
                    0.0
                }
            },
            3,
        );
        let before = mesh.indices.len() / 3;
        decimate_mesh(&mut mesh, cell(3), 6);
        let after = mesh.indices.len() / 3;
        assert!(
            after * 10 < before,
            "flat slab should reduce >10x, got {before} -> {after}"
        );
        for (_, count) in edge_face_counts(&mesh.indices) {
            assert_eq!(count, 2);
        }
    }

    /// Interior edges whose two incident faces are folded back on each other
    /// (unit-normal dot < -0.9). A clean decimation of a clean mesh has none.
    fn hard_fold_edge_count(vertices: &[(f32, f32, f32)], indices: &[u32]) -> usize {
        let positions: Vec<[f64; 3]> = vertices
            .iter()
            .map(|&(x, y, z)| [x as f64, y as f64, z as f64])
            .collect();
        let normals: Vec<Option<[f64; 3]>> = indices
            .chunks_exact(3)
            .map(|t| {
                let (p0, p1, p2) = (
                    positions[t[0] as usize],
                    positions[t[1] as usize],
                    positions[t[2] as usize],
                );
                normalize(cross(sub(p1, p0), sub(p2, p0)))
            })
            .collect();
        let mut edge_faces: std::collections::HashMap<u64, Vec<usize>> =
            std::collections::HashMap::new();
        for (fi, t) in indices.chunks_exact(3).enumerate() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edge_faces.entry(edge_key(a, b)).or_default().push(fi);
            }
        }
        edge_faces
            .values()
            .filter(|faces| {
                faces.len() == 2
                    && match (normals[faces[0]], normals[faces[1]]) {
                        (Some(n0), Some(n1)) => dot(n0, n1) < -0.9,
                        _ => false,
                    }
            })
            .count()
    }

    /// Regression test for decimation-minted flipped slivers (foam lattices,
    /// 2026-07). The per-collapse flip guard lets a face rotate just under
    /// 90° per collapse, so successive collapses on sub-cell thin walls
    /// accumulated fold-overs: near-coplanar back-facing sliver patches on a
    /// mesh whose raw input had zero fold-back edges. The guard is now also
    /// anchored to each face's decimation-entry orientation, which makes
    /// accumulation impossible. Without the anchor this thin-walled gyroid
    /// yields dozens of hard fold edges; with it there must be none.
    #[test]
    fn thin_walled_gyroid_decimation_creates_no_fold_backs() {
        let mut mesh = mesh_sampler(
            |x, y, z| {
                let a = 7.0;
                let g = (a * x).sin() * (a * y).cos()
                    + (a * y).sin() * (a * z).cos()
                    + (a * z).sin() * (a * x).cos();
                if g.abs() < 0.25 { 1.0 } else { 0.0 }
            },
            3,
        );
        assert_eq!(
            hard_fold_edge_count(&mesh.vertices, &mesh.indices),
            0,
            "raw mesh must enter decimation clean for this test to mean anything"
        );
        let before = mesh.indices.len() / 3;
        decimate_mesh(&mut mesh, cell(3) * 0.5, 6);
        assert!(mesh.indices.len() / 3 < before, "decimation ran");
        assert_eq!(
            hard_fold_edge_count(&mesh.vertices, &mesh.indices),
            0,
            "decimation minted folded-back faces"
        );
    }

    #[test]
    fn parallel_region_path_matches_serial_quality() {
        let radius = 0.8;
        let sampler = move |x: f64, y: f64, z: f64| {
            if x * x + y * y + z * z < radius * radius {
                1.0
            } else {
                0.0
            }
        };
        let mut serial = mesh_sampler(sampler, 3);
        let mut parallel = serial.clone();
        // usize::MAX forces the serial path, 0 forces region partitioning.
        decimate_mesh_impl(&mut serial, cell(3), 6, usize::MAX, &NEVER);
        decimate_mesh_impl(&mut parallel, cell(3), 6, 0, &NEVER);

        assert!(parallel.indices.len() < serial.indices.len() * 3 / 2);
        // Same manifold guarantees as the serial path.
        for (_, count) in edge_face_counts(&parallel.indices) {
            assert_eq!(count, 2, "parallel path broke the closed surface");
        }
        // Subset placement holds across regions and the seam pass.
        for &(x, y, z) in &parallel.vertices {
            let r = ((x as f64).powi(2) + (y as f64).powi(2) + (z as f64).powi(2)).sqrt();
            assert!(
                (r - radius).abs() < cell(3) * 0.1,
                "vertex off surface: {r}"
            );
        }
    }

    #[test]
    fn parallel_thin_strut_keeps_topology() {
        let radius = 0.08;
        let mut mesh = mesh_sampler(
            move |x, y, _z| {
                if x * x + y * y < radius * radius {
                    1.0
                } else {
                    0.0
                }
            },
            3,
        );
        let before = mesh.indices.len() / 3;
        decimate_mesh_impl(&mut mesh, cell(3), 6, 0, &NEVER);
        assert!(mesh.indices.len() / 3 < before);
        assert_eq!(component_count(mesh.vertices.len(), &mesh.indices), 1);
        let counts = edge_face_counts(&mesh.indices);
        assert!(
            counts.values().all(|&c| c <= 2),
            "non-manifold edge created"
        );
        assert!(counts.values().filter(|&&c| c == 1).count() >= 6);
    }

    #[test]
    fn empty_and_zero_tolerance_inputs_are_noops() {
        let mut empty = IndexedMesh2 {
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        };
        let stats = decimate_mesh(&mut empty, 0.01, 6);
        assert_eq!(stats.triangles_after, 0);

        let mut mesh = mesh_sampler(
            |x, y, z| {
                if x * x + y * y + z * z < 0.64 {
                    1.0
                } else {
                    0.0
                }
            },
            2,
        );
        let before = mesh.indices.len() / 3;
        let stats = decimate_mesh(&mut mesh, 0.0, 6);
        assert_eq!(stats.triangles_after, before);
    }
}
