//! The d-dimensional tiled voxel-mask payload: the data contract between
//! operators that bake a *sparse* binary mask onto a regular full-d
//! lattice (`island_removal_operator` being the first) and
//! `island_model_template`, which multilinearly samples it at model
//! sample time.
//!
//! The mask is stored as fixed-size dense tiles keyed by their packed
//! tile coordinates: only the vicinity of set points costs memory, so a
//! fine lattice (thousands of points per axis) over a large model stays
//! cheap when the set regions are sparse — the design target being
//! pruning scattered overhang defects out of foam lattices at
//! resolutions well past what a dense bitset could store.
//!
//! Same shape as the gridfield payload pattern: the operator does all
//! the work up front through [`TiledMaskBuilder`], the generated model
//! is stateless ([`PayloadView`] only reads), and both sides live in
//! this one natively unit-tested crate so the layout can't drift.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (20 + 20*d bytes):
//!    0        magic       u32    "VXT1" (0x3154_5856)
//!    4        dims        u32    d (1..=8)
//!    8        payload_len u32    total byte length, header included
//!   12        tile_edge   u32    lattice points per tile axis
//!   16        counts      d*u32  lattice point count per axis (>= 2)
//!   16+4d     bounds      2d*f64 [min_0, max_0, min_1, max_1, ...]
//!   16+20d    tile_count  u32
//! Keys (tile_count * u64 after the header): packed tile coordinates
//! (8 bits per axis, axis 0 lowest), sorted ascending.
//! Tiles (tile_count * ceil(tile_edge^d / 8) bytes): dense bitsets in
//! key order, LSB-first within each byte, axis 0 fastest within a tile.
//! ```
//!
//! # Sampling semantics
//!
//! Lattice points sit corner-aligned: point (0, ..., 0) lives exactly at
//! (min_0, ..., min_{d-1}) and the last point at the maxes, with
//! multilinear interpolation between — set bits read as 1.0, clear bits
//! (including every point of an absent tile) as 0.0, so the 0.5 level
//! set sits halfway between a set point and a clear neighbor. Outside
//! the bounds box the mask is undefined and [`PayloadView::sample`]
//! returns `None` (consumers read that as 0).

use std::collections::BTreeMap;

pub const MAGIC: u32 = 0x3154_5856; // "VXT1"
pub const MAX_DIMS: usize = 8;

/// Tile coordinates pack into 8 bits per axis.
const MAX_TILES_PER_AXIS: usize = 256;

fn header_len(dims: usize) -> usize {
    20 + 20 * dims
}

/// Points per tile: small enough that a lone set point costs little,
/// large enough that the key directory stays short. Shrinks with
/// dimensionality to keep tile_edge^d bounded.
pub fn default_tile_edge(dims: usize) -> usize {
    match dims {
        1..=3 => 16,
        4 => 8,
        _ => 4,
    }
}

fn tile_bytes(tile_edge: usize, dims: usize) -> usize {
    tile_edge.pow(dims as u32).div_ceil(8)
}

/// Accumulates set lattice points into sparse tiles, then serializes.
pub struct TiledMaskBuilder {
    dims: usize,
    tile_edge: usize,
    tiles: BTreeMap<u64, Vec<u8>>,
    /// Tight bbox of the set points, per axis, while any are set.
    lo: Vec<usize>,
    hi: Vec<usize>,
    set_points: u64,
}

impl TiledMaskBuilder {
    pub fn new(dims: usize) -> Self {
        assert!((1..=MAX_DIMS).contains(&dims));
        Self {
            dims,
            tile_edge: default_tile_edge(dims),
            tiles: BTreeMap::new(),
            lo: vec![usize::MAX; dims],
            hi: vec![0; dims],
            set_points: 0,
        }
    }

    fn locate(&self, coords: &[usize]) -> (u64, usize) {
        debug_assert_eq!(coords.len(), self.dims);
        let mut key = 0u64;
        let mut bit = 0usize;
        let mut stride = 1usize;
        for (a, &c) in coords.iter().enumerate() {
            let tc = c / self.tile_edge;
            debug_assert!(tc < MAX_TILES_PER_AXIS);
            key |= (tc as u64) << (8 * a);
            bit += (c % self.tile_edge) * stride;
            stride *= self.tile_edge;
        }
        (key, bit)
    }

    /// Set the lattice point at `coords` (idempotent).
    pub fn set(&mut self, coords: &[usize]) {
        let (key, bit) = self.locate(coords);
        let bytes = tile_bytes(self.tile_edge, self.dims);
        let tile = self.tiles.entry(key).or_insert_with(|| vec![0u8; bytes]);
        let slot = &mut tile[bit / 8];
        if *slot >> (bit % 8) & 1 == 0 {
            *slot |= 1 << (bit % 8);
            self.set_points += 1;
            for (a, &c) in coords.iter().enumerate() {
                self.lo[a] = self.lo[a].min(c);
                self.hi[a] = self.hi[a].max(c);
            }
        }
    }

    pub fn is_set(&self, coords: &[usize]) -> bool {
        let (key, bit) = self.locate(coords);
        self.tiles
            .get(&key)
            .is_some_and(|tile| tile[bit / 8] >> (bit % 8) & 1 == 1)
    }

    pub fn is_empty(&self) -> bool {
        self.set_points == 0
    }

    /// Tight per-axis `(lo, hi)` bbox of the set points; `None` if empty.
    pub fn bbox(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        (self.set_points > 0).then(|| (self.lo.clone(), self.hi.clone()))
    }

    /// Serialize over a lattice of `counts` points per axis spanning
    /// `bounds` (`[min, max]` per axis). Set points must lie within
    /// `counts`, and `counts[a] <= 256 * tile_edge` (8-bit tile keys).
    pub fn finish(self, counts: &[usize], bounds: &[f64]) -> Result<Vec<u8>, String> {
        let d = self.dims;
        if counts.len() != d || bounds.len() != 2 * d {
            return Err(format!(
                "lattice geometry must be {d}-dimensional, got {} counts and {} bounds",
                counts.len(),
                bounds.len()
            ));
        }
        for (axis, &n) in counts.iter().enumerate() {
            if n < 2 {
                return Err(format!(
                    "mask axis {axis} needs at least 2 lattice points, got {n}"
                ));
            }
            if n > MAX_TILES_PER_AXIS * self.tile_edge {
                return Err(format!(
                    "mask axis {axis} has {n} lattice points; tiled keys allow at most {}",
                    MAX_TILES_PER_AXIS * self.tile_edge
                ));
            }
            if self.set_points > 0 && self.hi[axis] >= n {
                return Err(format!(
                    "set point at {} exceeds axis {axis} count {n}",
                    self.hi[axis]
                ));
            }
            let (lo, hi) = (bounds[2 * axis], bounds[2 * axis + 1]);
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(format!(
                    "mask axis {axis} bounds must be a finite nonempty interval, got [{lo}, {hi}]"
                ));
            }
        }

        let bytes_per_tile = tile_bytes(self.tile_edge, d);
        let payload_len = header_len(d) + self.tiles.len() * (8 + bytes_per_tile);
        let mut out = Vec::with_capacity(payload_len);
        out.extend(MAGIC.to_le_bytes());
        out.extend((d as u32).to_le_bytes());
        out.extend(
            u32::try_from(payload_len)
                .map_err(|_| "mask payload exceeds 4 GiB")?
                .to_le_bytes(),
        );
        out.extend((self.tile_edge as u32).to_le_bytes());
        for &n in counts {
            out.extend((n as u32).to_le_bytes());
        }
        for &b in bounds {
            out.extend(b.to_le_bytes());
        }
        out.extend(
            u32::try_from(self.tiles.len())
                .map_err(|_| "tile count exceeds u32")?
                .to_le_bytes(),
        );
        // BTreeMap iterates in key order, so the directory is sorted.
        for key in self.tiles.keys() {
            out.extend(key.to_le_bytes());
        }
        for tile in self.tiles.values() {
            out.extend(tile);
        }
        debug_assert_eq!(out.len(), payload_len);
        Ok(out)
    }
}

/// A read-only view over a serialized payload (sample-time side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    dims: usize,
    tile_edge: usize,
    tile_count: usize,
}

impl<'a> PayloadView<'a> {
    /// Validate the header and structural sizes.
    pub fn new(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() < 20 {
            return Err("payload shorter than the fixed header");
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err("bad payload magic");
        }
        let dims = u32_at(4) as usize;
        if dims == 0 || dims > MAX_DIMS {
            return Err("mask dimension count out of range");
        }
        let tile_edge = u32_at(12) as usize;
        if tile_edge == 0 || tile_edge > 256 {
            return Err("tile edge out of range");
        }
        if bytes.len() < header_len(dims) {
            return Err("payload shorter than header");
        }
        for axis in 0..dims {
            let n = u32_at(16 + 4 * axis) as usize;
            if n < 2 || n > MAX_TILES_PER_AXIS * tile_edge {
                return Err("mask axis count out of range");
            }
        }
        let tile_count = u32_at(16 + 20 * dims) as usize;
        let expected = tile_count
            .checked_mul(8 + tile_bytes(tile_edge, dims))
            .and_then(|t| t.checked_add(header_len(dims)))
            .ok_or("tile directory overflows")?;
        if u32_at(8) as usize != expected || bytes.len() < expected {
            return Err("payload length mismatch");
        }
        Ok(Self {
            bytes,
            dims,
            tile_edge,
            tile_count,
        })
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Lattice point count along `axis`.
    pub fn count(&self, axis: usize) -> usize {
        u32::from_le_bytes(self.bytes[16 + 4 * axis..20 + 4 * axis].try_into().unwrap()) as usize
    }

    /// `[min, max]` of `axis`.
    pub fn bound(&self, axis: usize) -> [f64; 2] {
        let off = 16 + 4 * self.dims + 16 * axis;
        std::array::from_fn(|i| {
            f64::from_le_bytes(self.bytes[off + i * 8..off + i * 8 + 8].try_into().unwrap())
        })
    }

    fn key_at(&self, index: usize) -> u64 {
        let off = header_len(self.dims) + index * 8;
        u64::from_le_bytes(self.bytes[off..off + 8].try_into().unwrap())
    }

    /// The lattice point's bit: absent tiles read as clear.
    fn bit(&self, coords: &[usize; MAX_DIMS]) -> f64 {
        let mut key = 0u64;
        let mut bit = 0usize;
        let mut stride = 1usize;
        for (a, &c) in coords.iter().take(self.dims).enumerate() {
            key |= ((c / self.tile_edge) as u64) << (8 * a);
            bit += (c % self.tile_edge) * stride;
            stride *= self.tile_edge;
        }
        // Binary search the sorted key directory.
        let (mut lo, mut hi) = (0usize, self.tile_count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.key_at(mid).cmp(&key) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => {
                    let tiles_base = header_len(self.dims) + self.tile_count * 8;
                    let byte = self.bytes
                        [tiles_base + mid * tile_bytes(self.tile_edge, self.dims) + bit / 8];
                    return (byte >> (bit % 8) & 1) as f64;
                }
            }
        }
        0.0
    }

    /// Multilinear sample at `pos` (`dims` coordinates); `None` outside
    /// the bounds box.
    pub fn sample(&self, pos: &[f64]) -> Option<f32> {
        let d = self.dims;
        debug_assert_eq!(pos.len(), d);
        // Per axis: the lower lattice index and the interpolation
        // fraction toward the next point.
        let mut base = [0usize; MAX_DIMS];
        let mut frac = [0.0f64; MAX_DIMS];
        for axis in 0..d {
            let [lo, hi] = self.bound(axis);
            let p = pos[axis];
            if !(p >= lo && p <= hi) {
                return None; // also rejects NaN coordinates
            }
            let n = self.count(axis);
            let g = ((p - lo) / (hi - lo) * (n - 1) as f64).clamp(0.0, (n - 1) as f64);
            let i0 = (g.floor() as usize).min(n - 2);
            base[axis] = i0;
            frac[axis] = g - i0 as f64;
        }
        let mut value = 0.0f64;
        let mut corner_coords = [0usize; MAX_DIMS];
        for corner in 0..1usize << d {
            let mut weight = 1.0f64;
            for axis in 0..d {
                if corner >> axis & 1 == 1 {
                    weight *= frac[axis];
                    corner_coords[axis] = base[axis] + 1;
                } else {
                    weight *= 1.0 - frac[axis];
                    corner_coords[axis] = base[axis];
                }
            }
            if weight > 0.0 {
                value += weight * self.bit(&corner_coords);
            }
        }
        Some(value as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 40x24 lattice over [0,39]x[0,23] with only point (17, 5) set —
    /// straddles nothing; its tile is (1, 0).
    fn dot_payload() -> Vec<u8> {
        let mut builder = TiledMaskBuilder::new(2);
        builder.set(&[17, 5]);
        builder.finish(&[40, 24], &[0.0, 39.0, 0.0, 23.0]).unwrap()
    }

    #[test]
    fn lattice_points_and_multilinear_interior() {
        let payload = dot_payload();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.dims(), 2);
        assert_eq!(view.bound(0), [0.0, 39.0]);
        assert_eq!(view.count(1), 24);

        assert_eq!(view.sample(&[17.0, 5.0]), Some(1.0));
        assert_eq!(view.sample(&[0.0, 0.0]), Some(0.0));
        assert_eq!(view.sample(&[39.0, 23.0]), Some(0.0));
        // Halfway toward a clear neighbor the mask crosses 0.5.
        assert_eq!(view.sample(&[16.5, 5.0]), Some(0.5));
        assert_eq!(view.sample(&[17.0, 5.5]), Some(0.5));
        // Diagonal blend: (1 - 0.25) * (1 - 0.5) on the set point.
        assert!((view.sample(&[17.25, 5.5]).unwrap() - 0.375).abs() < 1e-6);
    }

    #[test]
    fn interpolation_crosses_tile_boundaries() {
        // Points at (15, 3) and (16, 3) live in adjacent tiles (edge 16);
        // the blend between them must see both.
        let mut builder = TiledMaskBuilder::new(2);
        builder.set(&[15, 3]);
        builder.set(&[16, 3]);
        let payload = builder.finish(&[40, 24], &[0.0, 39.0, 0.0, 23.0]).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.sample(&[15.5, 3.0]), Some(1.0));
        assert_eq!(view.sample(&[14.5, 3.0]), Some(0.5));
        assert_eq!(view.sample(&[16.5, 3.0]), Some(0.5));
    }

    #[test]
    fn absent_tiles_read_as_clear() {
        let payload = dot_payload();
        let view = PayloadView::new(&payload).unwrap();
        // (35, 20) lives in tile (2, 1), which was never touched.
        assert_eq!(view.sample(&[35.0, 20.0]), Some(0.0));
    }

    #[test]
    fn outside_bounds_is_none() {
        let payload = dot_payload();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.sample(&[-0.1, 5.0]), None);
        assert_eq!(view.sample(&[39.1, 5.0]), None);
        assert_eq!(view.sample(&[17.0, -0.1]), None);
        assert_eq!(view.sample(&[17.0, 23.1]), None);
        assert_eq!(view.sample(&[f64::NAN, 5.0]), None);
    }

    #[test]
    fn three_dimensions_interpolate_all_axes() {
        let mut builder = TiledMaskBuilder::new(3);
        builder.set(&[1, 1, 1]);
        let payload = builder
            .finish(&[2, 2, 2], &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            .unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.sample(&[1.0, 1.0, 1.0]), Some(1.0));
        assert_eq!(view.sample(&[0.0, 1.0, 1.0]), Some(0.0));
        assert!((view.sample(&[0.5, 0.5, 0.5]).unwrap() - 0.125).abs() < 1e-6);
        assert!((view.sample(&[0.9, 1.0, 1.0]).unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn builder_tracks_bbox_and_dedups() {
        let mut builder = TiledMaskBuilder::new(2);
        assert!(builder.is_empty());
        assert_eq!(builder.bbox(), None);
        builder.set(&[17, 5]);
        builder.set(&[17, 5]);
        builder.set(&[3, 20]);
        assert!(builder.is_set(&[17, 5]));
        assert!(!builder.is_set(&[16, 5]));
        assert_eq!(builder.set_points, 2, "set is idempotent");
        assert_eq!(builder.bbox(), Some((vec![3, 5], vec![17, 20])));
    }

    #[test]
    fn sparse_storage_stays_small() {
        // Two far-apart dots on a large fine lattice: two tiles, not a
        // dense grid.
        let mut builder = TiledMaskBuilder::new(3);
        builder.set(&[10, 10, 10]);
        builder.set(&[2000, 1500, 900]);
        let payload = builder
            .finish(&[2051, 1600, 1000], &[0.0, 2050.0, 0.0, 1599.0, 0.0, 999.0])
            .unwrap();
        // Header + 2 tiles of (8-byte key + 16^3/8 bytes).
        assert_eq!(payload.len(), 20 + 20 * 3 + 2 * (8 + 512));
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.sample(&[10.0, 10.0, 10.0]), Some(1.0));
        assert_eq!(view.sample(&[2000.0, 1500.0, 900.0]), Some(1.0));
        assert_eq!(view.sample(&[1000.0, 800.0, 500.0]), Some(0.0));
    }

    #[test]
    fn one_dimension_works() {
        let mut builder = TiledMaskBuilder::new(1);
        builder.set(&[1]);
        let payload = builder.finish(&[3], &[0.0, 2.0]).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.sample(&[1.0]), Some(1.0));
        assert_eq!(view.sample(&[1.5]), Some(0.5));
        assert_eq!(view.sample(&[2.5]), None);
    }

    #[test]
    fn invalid_payloads_are_rejected() {
        let empty = TiledMaskBuilder::new(2);
        assert!(empty.finish(&[2], &[0.0, 1.0]).is_err()); // dims mismatch
        let empty = TiledMaskBuilder::new(2);
        assert!(empty.finish(&[1, 3], &[0.0, 1.0, 0.0, 1.0]).is_err());
        let empty = TiledMaskBuilder::new(2);
        assert!(empty.finish(&[3, 3], &[1.0, 1.0, 0.0, 1.0]).is_err());
        let mut out_of_range = TiledMaskBuilder::new(2);
        out_of_range.set(&[10, 0]);
        assert!(out_of_range.finish(&[3, 3], &[0.0, 1.0, 0.0, 1.0]).is_err());
        // Axis longer than the packed tile keys can address.
        let empty = TiledMaskBuilder::new(2);
        assert!(
            empty
                .finish(&[16 * 256 + 1, 3], &[0.0, 1.0, 0.0, 1.0])
                .is_err()
        );

        assert!(PayloadView::new(&[0u8; 12]).is_err());
        let mut bad = dot_payload();
        bad[0] ^= 0xFF;
        assert!(PayloadView::new(&bad).is_err());
        bad[0] ^= 0xFF;
        bad.truncate(bad.len() - 1);
        assert!(PayloadView::new(&bad).is_err());
    }
}
