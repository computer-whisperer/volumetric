//! A signed distance lattice baked from occupancy, and its evaluation at
//! points: how a scan is measured against a model. Occupancy is sampled on
//! a regular grid, the exact Euclidean distance transform (Felzenszwalb
//! and Huttenlocher's separable parabola lower envelope) runs once over
//! the occupied cells and once over the empty ones, and the signed
//! distance to the nearest cell of the other kind is stored per cell;
//! points are read back trilinearly. Precision is the cell size: a
//! surface lies somewhere within the cell that changes state, so a
//! distance is good to about half a cell plus the interpolation.

/// A signed distance field on a regular lattice: negative inside.
#[derive(Clone, Debug)]
pub struct DistanceGrid {
    /// World position of cell (0, 0, 0)'s centre.
    pub origin: [f64; 3],
    /// Cell size, the same along every axis.
    pub spacing: f64,
    /// Cells per axis.
    pub dims: [usize; 3],
    /// Signed distance per cell, x fastest.
    pub values: Vec<f32>,
}

impl DistanceGrid {
    /// Bakes the field from `occupied` (x fastest, `dims` cells) with the
    /// cell spacing and origin given. A grid with no occupied or no empty
    /// cell gets the distance to the nearest lattice boundary instead, so
    /// the sign is still right.
    pub fn bake(origin: [f64; 3], spacing: f64, dims: [usize; 3], occupied: &[bool]) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        assert_eq!(occupied.len(), n, "occupancy must cover the lattice");
        // Squared distance from every cell to the nearest empty cell
        // (`outside`), and to the nearest occupied cell (`inside`).
        let to_empty = edt_squared(dims, |i| !occupied[i]);
        let to_full = edt_squared(dims, |i| occupied[i]);
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            // A cell's own centre sits half a cell from the boundary it
            // shares with a neighbour of the other kind.
            let d = if occupied[i] {
                -(to_empty[i].sqrt() - 0.5).max(0.0)
            } else {
                (to_full[i].sqrt() - 0.5).max(0.0)
            };
            values.push((d * spacing) as f32);
        }
        Self {
            origin,
            spacing,
            dims,
            values,
        }
    }

    /// The signed distance at `p`, trilinear between cell centres; beyond
    /// the lattice the nearest cell's value plus the distance to the
    /// lattice (which can only make a positive distance larger).
    pub fn sample(&self, p: [f64; 3]) -> f64 {
        let mut f = [0.0f64; 3];
        let mut outside = 0.0f64;
        for a in 0..3 {
            let u = (p[a] - self.origin[a]) / self.spacing;
            let max = (self.dims[a] - 1) as f64;
            if u < 0.0 {
                outside += u * u;
                f[a] = 0.0;
            } else if u > max {
                outside += (u - max) * (u - max);
                f[a] = max;
            } else {
                f[a] = u;
            }
        }
        let mut value = 0.0;
        let i0: Vec<usize> = f.iter().map(|v| v.floor() as usize).collect();
        let t: Vec<f64> = f.iter().zip(&i0).map(|(v, i)| v - *i as f64).collect();
        for corner in 0..8 {
            let mut weight = 1.0;
            let mut index = [0usize; 3];
            for a in 0..3 {
                let hi = (corner >> a) & 1 == 1;
                let last = self.dims[a] - 1;
                index[a] = if hi { (i0[a] + 1).min(last) } else { i0[a] };
                weight *= if hi { t[a] } else { 1.0 - t[a] };
            }
            if weight > 0.0 {
                value += weight * f64::from(self.at(index));
            }
        }
        value + outside.sqrt() * self.spacing
    }

    fn at(&self, i: [usize; 3]) -> f32 {
        self.values[i[0] + self.dims[0] * (i[1] + self.dims[1] * i[2])]
    }
}

/// Squared distance in cells from every cell to the nearest cell where
/// `seed` holds; a lattice with no seed cell gets the distance to its
/// nearest face plus one.
fn edt_squared(dims: [usize; 3], seed: impl Fn(usize) -> bool) -> Vec<f64> {
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    let far = (nx + ny + nz) as f64;
    let inf = far * far * 4.0;
    let mut d: Vec<f64> = (0..n).map(|i| if seed(i) { 0.0 } else { inf }).collect();
    if !(0..n).any(&seed) {
        return (0..n)
            .map(|i| {
                let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
                let edge = [x, nx - 1 - x, y, ny - 1 - y, z, nz - 1 - z]
                    .into_iter()
                    .min()
                    .unwrap_or(0) as f64
                    + 1.0;
                edge * edge
            })
            .collect();
    }
    let longest = nx.max(ny).max(nz);
    let mut line = vec![0.0; longest];
    let mut out = vec![0.0; longest];
    let mut v = vec![0usize; longest];
    let mut z = vec![0.0; longest + 1];
    let mut pass = |d: &mut Vec<f64>, len: usize, stride: usize, starts: Vec<usize>| {
        for start in starts {
            for k in 0..len {
                line[k] = d[start + k * stride];
            }
            envelope(&line[..len], &mut out[..len], &mut v, &mut z, inf);
            for k in 0..len {
                d[start + k * stride] = out[k];
            }
        }
    };
    pass(&mut d, nx, 1, (0..ny * nz).map(|yz| yz * nx).collect());
    pass(
        &mut d,
        ny,
        nx,
        (0..nx * nz)
            .map(|xz| (xz % nx) + (xz / nx) * nx * ny)
            .collect(),
    );
    pass(&mut d, nz, nx * ny, (0..nx * ny).collect());
    d
}

/// One-dimensional squared distance transform: `out[q] = min_p (q - p)^2
/// + f[p]`, the lower envelope of parabolas (Felzenszwalb & Huttenlocher).
fn envelope(f: &[f64], out: &mut [f64], v: &mut [usize], z: &mut [f64], inf: f64) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = -inf;
    z[1] = inf;
    let intersection = |q: usize, p: usize| {
        ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * (q - p) as f64)
    };
    for q in 1..n {
        let mut s = intersection(q, v[k]);
        // z[0] is -inf, so the envelope's first parabola is never popped.
        while s <= z[k] {
            k -= 1;
            s = intersection(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = inf;
    }
    let mut k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        let dq = q as f64 - p as f64;
        out[q] = dq * dq + f[p];
    }
}

/// Summary of signed distances: what a cloud-to-model audit reports.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DistanceStats {
    pub count: usize,
    pub inside: usize,
    pub mean: f64,
    pub abs_p50: f64,
    pub abs_p90: f64,
    pub abs_p99: f64,
    pub max_outside: f64,
    pub max_inside: f64,
}

impl DistanceStats {
    /// The fraction of `distances` within `band` of zero, either side.
    pub fn within(distances: &[f64], band: f64) -> f64 {
        if distances.is_empty() {
            return 0.0;
        }
        distances.iter().filter(|d| d.abs() <= band).count() as f64 / distances.len() as f64
    }

    pub fn of(distances: &[f64]) -> Self {
        if distances.is_empty() {
            return Self::default();
        }
        let mut abs: Vec<f64> = distances.iter().map(|d| d.abs()).collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |p: f64| abs[((abs.len() - 1) as f64 * p).round() as usize];
        Self {
            count: distances.len(),
            inside: distances.iter().filter(|d| **d < 0.0).count(),
            mean: distances.iter().sum::<f64>() / distances.len() as f64,
            abs_p50: q(0.5),
            abs_p90: q(0.9),
            abs_p99: q(0.99),
            max_outside: distances.iter().cloned().fold(0.0, f64::max),
            max_inside: -distances.iter().cloned().fold(0.0, f64::min),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere of radius 0.3 at the origin on a 1 m lattice of 41 cells.
    fn sphere_grid() -> DistanceGrid {
        let dims = [41, 41, 41];
        let spacing = 0.025;
        let origin = [-0.5, -0.5, -0.5];
        let mut occupied = Vec::with_capacity(41 * 41 * 41);
        for z in 0..41 {
            for y in 0..41 {
                for x in 0..41 {
                    let p = [
                        origin[0] + x as f64 * spacing,
                        origin[1] + y as f64 * spacing,
                        origin[2] + z as f64 * spacing,
                    ];
                    occupied.push((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() <= 0.3);
                }
            }
        }
        DistanceGrid::bake(origin, spacing, dims, &occupied)
    }

    #[test]
    fn envelope_matches_brute_force() {
        let f = [9.0, 0.0, 4.0, 4.0, 1.0, 16.0, 0.0, 25.0];
        let mut out = vec![0.0; f.len()];
        let (mut v, mut z) = (vec![0; f.len()], vec![0.0; f.len() + 1]);
        envelope(&f, &mut out, &mut v, &mut z, 1e9);
        for q in 0..f.len() {
            let brute = (0..f.len())
                .map(|p| ((q as f64 - p as f64).powi(2)) + f[p])
                .fold(f64::INFINITY, f64::min);
            assert!(
                (out[q] - brute).abs() < 1e-9,
                "q {q}: {} vs {brute}",
                out[q]
            );
        }
    }

    #[test]
    fn sphere_distances_are_within_a_cell() {
        let grid = sphere_grid();
        for (p, expected) in [
            ([0.0, 0.0, 0.0], -0.3),
            ([0.2, 0.0, 0.0], -0.1),
            ([0.4, 0.0, 0.0], 0.1),
            ([0.0, 0.45, 0.0], 0.15),
            ([0.3, 0.3, 0.3], 0.3 * 3f64.sqrt() - 0.3),
        ] {
            let d = grid.sample(p);
            assert!(
                (d - expected).abs() < grid.spacing,
                "{p:?}: {d} vs {expected}"
            );
        }
        // Beyond the lattice the distance keeps growing.
        let edge = grid.sample([0.5, 0.0, 0.0]);
        let beyond = grid.sample([0.8, 0.0, 0.0]);
        assert!(beyond > edge + 0.25, "{beyond} vs {edge}");
    }

    #[test]
    fn stats_summarise_signed_distances() {
        let s = DistanceStats::of(&[-0.002, 0.001, 0.010, -0.001, 0.0]);
        assert_eq!(s.count, 5);
        assert_eq!(s.inside, 2);
        assert!((s.max_outside - 0.010).abs() < 1e-12 && (s.max_inside - 0.002).abs() < 1e-12);
        assert!((s.abs_p50 - 0.001).abs() < 1e-12);
        assert_eq!(DistanceStats::of(&[]), DistanceStats::default());
        let d = [-0.002, 0.001, 0.010, -0.001, 0.0];
        assert!((DistanceStats::within(&d, 0.001) - 0.6).abs() < 1e-12);
        assert!((DistanceStats::within(&d, 0.005) - 0.8).abs() < 1e-12);
    }
}
