//! ASN2-style sparse occupancy discovery (single-threaded).
//!
//! Mirrors the mesher's sampling behavior (`adaptive_surface_nets_2`):
//! a coarse corner grid plus aperiodic interior probes locate geometry,
//! then mixed cells subdivide toward the fine lattice pitch with the
//! frontier expanding across faces where corner samples disagree — the
//! surface is *walked* from discovered seeds, so samples concentrate
//! near geometry instead of sweeping known-empty (or known-solid)
//! space. Matching the mesher's discovery contract is deliberate: a
//! feature this pass cannot find is one the mesher would not print.
//!
//! Differences from the mesher, which only needs the surface:
//! - No triangles; the product is *volumetric* occupancy of the fine
//!   lattice, resolved lazily per 16-point-edge block: sampled points
//!   are exact, and unsampled points are inferred by flood fill from
//!   the sampled shell (a region bounded by walked surface is uniformly
//!   in or out), falling back to the block's coarse classification.
//! - Blocks store three bitsets (sampled, value, visited) so memory
//!   scales with the touched vicinity, never the lattice volume.
//!
//! The walk consumes the result layer by layer through
//! [`Occupancy::fill_layer`], exactly as the dense scan did.

use std::collections::HashMap;

/// Fine lattice points per block edge (and per stage-1 coarse cell).
pub const BLOCK_EDGE: usize = 16;

/// Cap on the dense stage-1 classification array.
const MAX_COARSE_CELLS: u64 = 1 << 24;

/// Batched occupancy sampling over *fine lattice* coordinates
/// (`points` holds `dims` coordinates per point). Host-backed in
/// production, closures in tests.
pub trait BatchSampler {
    fn sample_points(&mut self, points: &[usize], dims: usize) -> Result<Vec<bool>, String>;
}

/// Splitmix64 — deterministic probe scatter (no RNG in wasm operators).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One block: 16^d fine points, three bit planes.
struct Block {
    sampled: Vec<u64>,
    value: Vec<u64>,
    /// Finest-cell visited marks for the frontier walk (keyed by the
    /// cell's min corner).
    visited: Vec<u64>,
}

impl Block {
    fn new(dims: usize) -> Self {
        let words = BLOCK_EDGE.pow(dims as u32).div_ceil(64);
        Self {
            sampled: vec![0; words],
            value: vec![0; words],
            visited: vec![0; words],
        }
    }
}

fn get_bit(plane: &[u64], bit: usize) -> bool {
    plane[bit / 64] >> (bit % 64) & 1 == 1
}

fn set_bit(plane: &mut [u64], bit: usize) {
    plane[bit / 64] |= 1 << (bit % 64);
}

/// A subdivision / frontier cell: `size` fine units per edge, a power
/// of two dividing [`BLOCK_EDGE`], never straddling a block.
#[derive(Clone)]
struct Cell {
    min: Vec<usize>,
    size: usize,
}

/// Stage-1 classification per coarse cell (the flood-fill fallback for
/// points discovery never touched).
#[derive(Clone, Copy, PartialEq)]
enum CellClass {
    UniformOut,
    UniformIn,
    Mixed,
}

pub struct Occupancy {
    dims: usize,
    counts: Vec<usize>,
    coarse_counts: Vec<usize>,
    class: Vec<CellClass>,
    blocks: HashMap<u128, Block>,
    samples_taken: u64,
}

impl Occupancy {
    fn block_key(coarse: &[usize]) -> u128 {
        coarse
            .iter()
            .enumerate()
            .map(|(a, &c)| (c as u128) << (16 * a))
            .sum()
    }

    fn point_key_bit(coords: &[usize]) -> (u128, usize) {
        let mut key = 0u128;
        let mut bit = 0usize;
        let mut stride = 1usize;
        for (a, &c) in coords.iter().enumerate() {
            key |= ((c / BLOCK_EDGE) as u128) << (16 * a);
            bit += (c % BLOCK_EDGE) * stride;
            stride *= BLOCK_EDGE;
        }
        (key, bit)
    }

    fn coarse_index(&self, coarse: &[usize]) -> usize {
        let mut index = 0usize;
        let mut stride = 1usize;
        for (a, &c) in coarse.iter().enumerate() {
            index += c * stride;
            stride *= self.coarse_counts[a];
        }
        index
    }

    fn in_lattice(&self, coords: &[usize]) -> bool {
        coords.iter().zip(&self.counts).all(|(&c, &n)| c < n)
    }

    /// The known value of a fine point: sampled/resolved bit, else
    /// `None`. Points outside the lattice read `Some(false)` — they lie
    /// beyond the model bounds plus margin, hence unoccupied.
    fn known(&self, coords: &[usize]) -> Option<bool> {
        if !self.in_lattice(coords) {
            return Some(false);
        }
        let (key, bit) = Self::point_key_bit(coords);
        let block = self.blocks.get(&key)?;
        get_bit(&block.sampled, bit).then(|| get_bit(&block.value, bit))
    }

    fn fallback(&self, coords: &[usize]) -> bool {
        let coarse: Vec<usize> = coords.iter().map(|&c| c / BLOCK_EDGE).collect();
        self.class[self.coarse_index(&coarse)] == CellClass::UniformIn
    }

    /// The resolved value of a fine point (post-discovery).
    pub fn point(&self, coords: &[usize]) -> bool {
        self.known(coords).unwrap_or_else(|| self.fallback(coords))
    }

    /// Total sampler points spent (test/diagnostic).
    #[cfg(test)]
    pub fn samples_taken(&self) -> u64 {
        self.samples_taken
    }

    /// Fill one layer of occupancy along `axis` at `layer`, in the walk's
    /// row order: the non-axis model axes ascending, first fastest.
    pub fn fill_layer(&self, axis: usize, layer: usize, out: &mut Vec<bool>) {
        out.clear();
        let inner_axes: Vec<usize> = (0..self.dims).filter(|&a| a != axis).collect();
        let layer_size: usize = inner_axes.iter().map(|&a| self.counts[a]).product();
        out.resize(layer_size.max(1), false);
        // Odometer over the inner axes, first fastest; `coords` carries
        // the full model-space coordinates.
        let mut coords = vec![0usize; self.dims];
        coords[axis] = layer;
        for slot in out.iter_mut() {
            *slot = self.point(&coords);
            for &a in &inner_axes {
                coords[a] += 1;
                if coords[a] < self.counts[a] {
                    break;
                }
                coords[a] = 0;
            }
        }
    }
}

/// The discovery driver; see the module docs for the algorithm.
pub struct Discovery<'s, S: BatchSampler> {
    occ: Occupancy,
    sampler: &'s mut S,
}

impl<'s, S: BatchSampler> Discovery<'s, S> {
    /// Run discovery over a fine lattice of `counts` points per axis.
    /// `probes` is the aperiodic interior probe count per corner-uniform
    /// coarse cell (0 disables — sub-coarse isolated features are then
    /// only found if the coarse corner grid hits them).
    pub fn run(counts: &[usize], probes: usize, sampler: &'s mut S) -> Result<Occupancy, String> {
        let dims = counts.len();
        let coarse_counts: Vec<usize> = counts.iter().map(|&n| n.div_ceil(BLOCK_EDGE)).collect();
        let coarse_total = coarse_counts.iter().map(|&n| n as u64).product::<u64>();
        if coarse_total > MAX_COARSE_CELLS {
            return Err(format!(
                "discovery grid {coarse_counts:?} exceeds {MAX_COARSE_CELLS} coarse cells; \
                 lower resolution"
            ));
        }
        let mut this = Self {
            occ: Occupancy {
                dims,
                counts: counts.to_vec(),
                coarse_counts,
                class: vec![CellClass::UniformOut; coarse_total as usize],
                blocks: HashMap::new(),
                samples_taken: 0,
            },
            sampler,
        };
        let seeds = this.stage1(probes)?;
        this.stage2(seeds)?;
        this.resolve_blocks();
        Ok(this.occ)
    }

    /// Sample every not-yet-known in-lattice point of `points` (flat,
    /// `dims` coordinates each), in bounded batches.
    fn ensure(&mut self, points: &[usize]) -> Result<(), String> {
        const BATCH: usize = 65536;
        let dims = self.occ.dims;
        let mut pending: Vec<usize> = Vec::new();
        let mut pending_slots: Vec<(u128, usize)> = Vec::new();
        for point in points.chunks_exact(dims) {
            if !self.occ.in_lattice(point) {
                continue;
            }
            let (key, bit) = Occupancy::point_key_bit(point);
            let block = self
                .occ
                .blocks
                .entry(key)
                .or_insert_with(|| Block::new(dims));
            if get_bit(&block.sampled, bit) {
                continue;
            }
            // Mark sampled now: it doubles as the in-flight dedup.
            set_bit(&mut block.sampled, bit);
            pending.extend_from_slice(point);
            pending_slots.push((key, bit));
        }
        for (chunk, slots) in pending
            .chunks(BATCH * dims)
            .zip(pending_slots.chunks(BATCH))
        {
            let values = self.sampler.sample_points(chunk, dims)?;
            self.occ.samples_taken += values.len() as u64;
            for (&(key, bit), &v) in slots.iter().zip(&values) {
                if v {
                    let block = self.occ.blocks.get_mut(&key).expect("block exists");
                    set_bit(&mut block.value, bit);
                }
            }
        }
        Ok(())
    }

    /// The 2^d corner values of a cell (out-of-lattice corners read
    /// outside); `None` if any in-lattice corner is unsampled.
    fn corners(&self, cell: &Cell) -> Option<Vec<bool>> {
        let d = self.occ.dims;
        let mut out = Vec::with_capacity(1 << d);
        let mut corner = vec![0usize; d];
        for c in 0..1usize << d {
            for (a, slot) in corner.iter_mut().enumerate() {
                *slot = cell.min[a] + if c >> a & 1 == 1 { cell.size } else { 0 };
            }
            out.push(self.occ.known(&corner)?);
        }
        Some(out)
    }

    fn corner_points(&self, cell: &Cell, out: &mut Vec<usize>) {
        let d = self.occ.dims;
        for c in 0..1usize << d {
            for a in 0..d {
                out.push(cell.min[a] + if c >> a & 1 == 1 { cell.size } else { 0 });
            }
        }
    }

    /// Stage 1: the coarse corner grid plus interior probes; classifies
    /// every coarse cell and returns frontier seeds (per coarse index)
    /// for features only a probe saw.
    fn stage1(&mut self, probes: usize) -> Result<HashMap<usize, Vec<Vec<usize>>>, String> {
        let d = self.occ.dims;
        let coarse_counts = self.occ.coarse_counts.clone();

        // All block-corner lattice points (multiples of BLOCK_EDGE).
        let mut points = Vec::new();
        let mut corner = vec![0usize; d];
        loop {
            points.extend(corner.iter().map(|&c| c * BLOCK_EDGE));
            if !advance(
                &mut corner,
                &coarse_counts.iter().map(|&n| n + 1).collect::<Vec<_>>(),
            ) {
                break;
            }
        }
        self.ensure(&points)?;

        // Classify cells; gather probe points for the uniform ones.
        let mut probe_points = Vec::new();
        let mut coarse = vec![0usize; d];
        loop {
            let cell = Cell {
                min: coarse.iter().map(|&c| c * BLOCK_EDGE).collect(),
                size: BLOCK_EDGE,
            };
            let corners = self.corners(&cell).expect("stage-1 corners sampled");
            let index = self.occ.coarse_index(&coarse);
            if corners.iter().any(|&v| v != corners[0]) {
                self.occ.class[index] = CellClass::Mixed;
            } else {
                self.occ.class[index] = if corners[0] {
                    CellClass::UniformIn
                } else {
                    CellClass::UniformOut
                };
                let key = Occupancy::block_key(&coarse) as u64;
                for i in 0..probes {
                    let mut h = splitmix64(key ^ ((i as u64) << 48));
                    for &m in cell.min.iter() {
                        probe_points.push(m + (h as usize) % BLOCK_EDGE);
                        h >>= 8;
                    }
                }
            }
            if !advance(&mut coarse, &coarse_counts) {
                break;
            }
        }
        self.ensure(&probe_points)?;

        // A probe contradicting its cell's corners reveals a feature the
        // corner grid missed. The cell keeps its uniform classification
        // (the flood-fill fallback for its untouched bulk); instead,
        // bisect from the probe toward a corner — which holds the
        // opposite value — to pin a mixed finest cell, and seed the
        // frontier walk there.
        let mut seeds: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
        for chunk_start in (0..probe_points.len()).step_by(d) {
            let point = probe_points[chunk_start..chunk_start + d].to_vec();
            if !self.occ.in_lattice(&point) {
                continue;
            }
            let coarse: Vec<usize> = point.iter().map(|&c| c / BLOCK_EDGE).collect();
            let index = self.occ.coarse_index(&coarse);
            let uniform = match self.occ.class[index] {
                CellClass::UniformIn => true,
                CellClass::UniformOut => false,
                CellClass::Mixed => continue,
            };
            if self.occ.known(&point) != Some(!uniform) {
                continue;
            }
            let corner: Vec<usize> = coarse.iter().map(|&c| c * BLOCK_EDGE).collect();
            let seed = self.bisect_to_mixed_cell(point, corner)?;
            let seed_coarse: Vec<usize> = seed.iter().map(|&c| c / BLOCK_EDGE).collect();
            seeds
                .entry(self.occ.coarse_index(&seed_coarse))
                .or_default()
                .push(seed);
        }
        Ok(seeds)
    }

    /// Bisect the segment between two sampled points of opposite value
    /// down to a Chebyshev-adjacent pair, and return the min corner of
    /// the (necessarily mixed) finest cell containing both.
    fn bisect_to_mixed_cell(
        &mut self,
        mut a: Vec<usize>,
        mut b: Vec<usize>,
    ) -> Result<Vec<usize>, String> {
        let d = self.occ.dims;
        debug_assert_ne!(self.occ.known(&a), self.occ.known(&b));
        let value_a = self.occ.known(&a).expect("bisection endpoints sampled");
        while a.iter().zip(&b).any(|(&x, &y)| x.abs_diff(y) > 1) {
            let mid: Vec<usize> = a.iter().zip(&b).map(|(&x, &y)| x.midpoint(y)).collect();
            self.ensure(&mid)?;
            if self.occ.known(&mid) == Some(value_a) {
                a = mid;
            } else {
                b = mid;
            }
        }
        // Clamp so the finest cell's far corners stay in the lattice.
        Ok((0..d)
            .map(|axis| {
                a[axis]
                    .min(b[axis])
                    .min(self.occ.counts[axis].saturating_sub(2))
            })
            .collect())
    }

    /// Stage 2: per-coarse-cell subdivision plus cross-cell frontier
    /// walking, both driven off one queue. `seeds` carries stage 1's
    /// probe-discovered finest cells.
    fn stage2(&mut self, mut seeds: HashMap<usize, Vec<Vec<usize>>>) -> Result<(), String> {
        let d = self.occ.dims;
        let coarse_counts = self.occ.coarse_counts.clone();
        // Queue of coarse cells to (re)process, with pending finest-cell
        // seeds delivered by probes and neighbors' frontier expansion.
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut queued = vec![false; self.occ.class.len()];
        let mut subdivided = vec![false; self.occ.class.len()];

        let mut coarse = vec![0usize; d];
        loop {
            let index = self.occ.coarse_index(&coarse);
            if self.occ.class[index] == CellClass::Mixed || seeds.contains_key(&index) {
                queue.push_back(index);
                queued[index] = true;
            }
            if !advance(&mut coarse, &coarse_counts) {
                break;
            }
        }

        while let Some(index) = queue.pop_front() {
            queued[index] = false;
            // Decompose the dense index back into coarse coordinates.
            let mut coarse = vec![0usize; d];
            let mut rest = index;
            for a in 0..d {
                coarse[a] = rest % coarse_counts[a];
                rest /= coarse_counts[a];
            }

            let mut wave: Vec<Cell> = Vec::new();
            if !subdivided[index] && self.occ.class[index] == CellClass::Mixed {
                subdivided[index] = true;
                wave.push(Cell {
                    min: coarse.iter().map(|&c| c * BLOCK_EDGE).collect(),
                    size: BLOCK_EDGE,
                });
            }
            for seed in seeds.remove(&index).unwrap_or_default() {
                if self.mark_visited(&seed) {
                    wave.push(Cell { min: seed, size: 1 });
                }
            }

            // Local breadth-first subdivision and frontier walk.
            let mut points = Vec::new();
            while !wave.is_empty() {
                points.clear();
                for cell in &wave {
                    self.corner_points(cell, &mut points);
                }
                self.ensure(&points)?;

                let mut next: Vec<Cell> = Vec::new();
                for cell in std::mem::take(&mut wave) {
                    let corners = self.corners(&cell).expect("wave corners sampled");
                    let mixed = corners.iter().any(|&v| v != corners[0]);
                    if !mixed {
                        continue;
                    }
                    if cell.size > 1 {
                        let half = cell.size / 2;
                        for c in 0..1usize << d {
                            let min = (0..d)
                                .map(|a| cell.min[a] + if c >> a & 1 == 1 { half } else { 0 })
                                .collect();
                            next.push(Cell { min, size: half });
                        }
                        continue;
                    }
                    // Finest mixed cell: expand across every face whose
                    // corners disagree.
                    for axis in 0..d {
                        for dir in [-1isize, 1] {
                            let face_mixed = {
                                let side = dir == 1;
                                let mut first: Option<bool> = None;
                                let mut mixed = false;
                                for (c, &v) in corners.iter().enumerate() {
                                    if (c >> axis & 1 == 1) != side {
                                        continue;
                                    }
                                    mixed |= *first.get_or_insert(v) != v;
                                }
                                mixed
                            };
                            if !face_mixed {
                                continue;
                            }
                            let mut nmin = cell.min.clone();
                            match dir {
                                1 => nmin[axis] += 1,
                                _ if nmin[axis] == 0 => continue,
                                _ => nmin[axis] -= 1,
                            }
                            // The neighbor's far corners must exist.
                            if !(0..d).all(|a| nmin[a] + 1 < self.occ.counts[a]) {
                                continue;
                            }
                            let ncoarse: Vec<usize> =
                                nmin.iter().map(|&c| c / BLOCK_EDGE).collect();
                            if ncoarse == coarse {
                                if self.mark_visited(&nmin) {
                                    next.push(Cell { min: nmin, size: 1 });
                                }
                            } else {
                                // Crossing into another coarse cell:
                                // hand the seed to the global queue.
                                let nindex = self.occ.coarse_index(&ncoarse);
                                seeds.entry(nindex).or_default().push(nmin);
                                if !queued[nindex] {
                                    queued[nindex] = true;
                                    queue.push_back(nindex);
                                }
                            }
                        }
                    }
                }
                wave = next;
            }
        }
        Ok(())
    }

    /// Mark a finest cell visited (by its min corner); false if it was.
    fn mark_visited(&mut self, min: &[usize]) -> bool {
        let dims = self.occ.dims;
        let (key, bit) = Occupancy::point_key_bit(min);
        let block = self
            .occ
            .blocks
            .entry(key)
            .or_insert_with(|| Block::new(dims));
        if get_bit(&block.visited, bit) {
            return false;
        }
        set_bit(&mut block.visited, bit);
        true
    }

    /// Resolve every block: unsampled points take the value of a sampled
    /// point their unknown-connected component touches (the walked shell
    /// bounds each region), preferring outside on conflict (fail toward
    /// keeping), and falling back to the coarse classification for
    /// components the shell never touches.
    fn resolve_blocks(&mut self) {
        let d = self.occ.dims;
        let block_points = BLOCK_EDGE.pow(d as u32);
        let keys: Vec<u128> = self.occ.blocks.keys().copied().collect();
        let mut stack: Vec<usize> = Vec::new();
        let mut component: Vec<usize> = Vec::new();
        for key in keys {
            let coarse: Vec<usize> = (0..d)
                .map(|a| (key >> (16 * a)) as usize & 0xFFFF)
                .collect();
            let base: Vec<usize> = coarse.iter().map(|&c| c * BLOCK_EDGE).collect();
            let fallback = self.occ.class[self.occ.coarse_index(&coarse)] == CellClass::UniformIn;
            let block = &self.occ.blocks[&key];

            // Component labels over unsampled in-lattice points.
            let mut seen = vec![false; block_points];
            let decode = |bit: usize| -> Vec<usize> {
                let mut rest = bit;
                (0..d)
                    .map(|a| {
                        let c = base[a] + rest % BLOCK_EDGE;
                        rest /= BLOCK_EDGE;
                        c
                    })
                    .collect()
            };
            let mut resolutions: Vec<(usize, bool)> = Vec::new();
            for start in 0..block_points {
                if seen[start] || get_bit(&block.sampled, start) {
                    continue;
                }
                if !self.occ.in_lattice(&decode(start)) {
                    seen[start] = true;
                    continue;
                }
                // Flood the unknown component (2d-adjacency, in-block),
                // collecting adjacent sampled values.
                component.clear();
                stack.push(start);
                seen[start] = true;
                let (mut saw_in, mut saw_out) = (false, false);
                while let Some(bit) = stack.pop() {
                    component.push(bit);
                    let mut stride = 1usize;
                    let mut rest = bit;
                    for _a in 0..d {
                        let c = rest % BLOCK_EDGE;
                        rest /= BLOCK_EDGE;
                        for (step, edge) in [
                            (bit.wrapping_sub(stride), c > 0),
                            (bit + stride, c + 1 < BLOCK_EDGE),
                        ] {
                            if !edge {
                                continue;
                            }
                            if get_bit(&block.sampled, step) {
                                if get_bit(&block.value, step) {
                                    saw_in = true;
                                } else {
                                    saw_out = true;
                                }
                            } else if !seen[step] && self.occ.in_lattice(&decode(step)) {
                                seen[step] = true;
                                stack.push(step);
                            }
                        }
                        stride *= BLOCK_EDGE;
                    }
                }
                let value = if saw_out {
                    false // prefer outside on conflict: fail toward keeping
                } else if saw_in {
                    true
                } else {
                    fallback
                };
                for &bit in &component {
                    resolutions.push((bit, value));
                }
            }
            let block = self.occ.blocks.get_mut(&key).expect("block exists");
            for (bit, value) in resolutions {
                set_bit(&mut block.sampled, bit);
                if value {
                    set_bit(&mut block.value, bit);
                }
            }
        }
    }
}

/// Odometer increment over `counts`, axis 0 fastest; false on wrap.
fn advance(coords: &mut [usize], counts: &[usize]) -> bool {
    for (c, &n) in coords.iter_mut().zip(counts) {
        *c += 1;
        if *c < n {
            return true;
        }
        *c = 0;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closure-backed sampler counting every point it evaluates.
    struct ShapeSampler<F: Fn(&[usize]) -> bool> {
        shape: F,
        calls: u64,
    }

    impl<F: Fn(&[usize]) -> bool> BatchSampler for ShapeSampler<F> {
        fn sample_points(&mut self, points: &[usize], dims: usize) -> Result<Vec<bool>, String> {
            self.calls += points.len() as u64 / dims as u64;
            Ok(points.chunks_exact(dims).map(|p| (self.shape)(p)).collect())
        }
    }

    /// Discovery must reproduce brute-force occupancy exactly for
    /// geometry within its contract (features connected to something the
    /// coarse grid or probes can see).
    fn assert_matches_brute_force<F: Fn(&[usize]) -> bool + Copy>(
        counts: &[usize],
        shape: F,
        probes: usize,
    ) -> u64 {
        let mut sampler = ShapeSampler { shape, calls: 0 };
        let occ = Discovery::run(counts, probes, &mut sampler).unwrap();
        let mut coords = vec![0usize; counts.len()];
        loop {
            assert_eq!(occ.point(&coords), shape(&coords), "mismatch at {coords:?}");
            if !advance(&mut coords, counts) {
                break;
            }
        }
        occ.samples_taken()
    }

    #[test]
    fn disk_matches_brute_force_with_fewer_samples() {
        // A 2D disk spanning many coarse cells.
        let counts = [96usize, 80];
        let shape = |p: &[usize]| {
            let (dx, dy) = (p[0] as f64 - 48.0, p[1] as f64 - 40.0);
            dx * dx + dy * dy < 30.0 * 30.0
        };
        let samples = assert_matches_brute_force(&counts, shape, 8);
        let dense = (96 * 80) as u64;
        assert!(
            samples < dense / 2,
            "expected sparse sampling, took {samples} of {dense}"
        );
    }

    #[test]
    fn sphere_matches_brute_force_in_3d() {
        let counts = [48usize, 48, 48];
        let shape = |p: &[usize]| {
            let d: f64 = p.iter().map(|&c| (c as f64 - 24.0).powi(2)).sum();
            d < 18.0 * 18.0
        };
        let samples = assert_matches_brute_force(&counts, shape, 8);
        let dense = 48u64.pow(3);
        assert!(
            samples < dense / 3,
            "expected sparse sampling, took {samples} of {dense}"
        );
    }

    #[test]
    fn thin_connected_strut_is_walked_through_uniform_cells() {
        // A 1-point-wide diagonal strut rising from a base slab, crossing
        // coarse cells whose corners and probes miss it entirely: the
        // frontier must follow it from the slab. Probes are disabled to
        // prove connectivity alone finds it.
        let counts = [96usize, 64];
        let shape = |p: &[usize]| p[1] < 3 || (p[0] == p[1] + 20 && p[1] < 50);
        assert_matches_brute_force(&counts, shape, 0);
    }

    #[test]
    fn internal_void_is_resolved_empty() {
        // An annulus: the void's shell is discovered from the coarse
        // grid, and the flood fill must resolve the cavity as outside
        // and the ring interior as inside.
        let counts = [96usize, 96];
        let shape = |p: &[usize]| {
            let (dx, dy) = (p[0] as f64 - 48.0, p[1] as f64 - 48.0);
            let r2 = dx * dx + dy * dy;
            r2 < 40.0 * 40.0 && r2 > 18.0 * 18.0
        };
        assert_matches_brute_force(&counts, shape, 8);
    }

    #[test]
    fn probes_find_sub_coarse_isolated_features() {
        // An isolated 6x6 blob inside one coarse cell, missing every
        // corner: probes must catch it (and without probes it is
        // invisible — the documented discovery contract).
        let counts = [64usize, 64];
        let shape = |p: &[usize]| (20..26).contains(&p[0]) && (20..26).contains(&p[1]);
        let mut sampler = ShapeSampler { shape, calls: 0 };
        let occ = Discovery::run(&counts, 32, &mut sampler).unwrap();
        assert!(occ.point(&[22, 22]), "probes must discover the blob");
        assert!(!occ.point(&[40, 40]));

        let mut sampler = ShapeSampler { shape, calls: 0 };
        let occ = Discovery::run(&counts, 0, &mut sampler).unwrap();
        assert!(
            !occ.point(&[22, 22]),
            "without probes the blob is invisible (fails toward keeping)"
        );
    }

    #[test]
    fn fill_layer_matches_pointwise_reads() {
        let counts = [40usize, 32, 24];
        let shape = |p: &[usize]| {
            let d: f64 = p.iter().map(|&c| (c as f64 - 14.0).powi(2)).sum();
            d < 10.0 * 10.0
        };
        let mut sampler = ShapeSampler { shape, calls: 0 };
        let occ = Discovery::run(&counts, 8, &mut sampler).unwrap();
        let mut row = Vec::new();
        // Build axis 1: inner axes 0 (fastest) then 2.
        occ.fill_layer(1, 14, &mut row);
        assert_eq!(row.len(), 40 * 24);
        for z in 0..24 {
            for x in 0..40 {
                assert_eq!(row[z * 40 + x], occ.point(&[x, 14, z]), "at {x},{z}");
            }
        }
    }
}
