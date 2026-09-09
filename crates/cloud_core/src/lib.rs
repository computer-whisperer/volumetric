//! Small-vector and point-cloud kernels shared by the cloud operators
//! (`cloud_fit_operator`, `cloud_normals_operator`): 3-vector arithmetic,
//! the scatter matrix and symmetric 3x3 eigen-decomposition behind every
//! least-squares plane and line, a tiny dense solver, a deterministic
//! generator for RANSAC, and a uniform-grid nearest-neighbour index.

use std::collections::HashMap;

pub mod normals;

pub type Vec3 = [f64; 3];

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn mul(v: Vec3, s: f64) -> Vec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(v: Vec3) -> f64 {
    dot(v, v).sqrt()
}

/// `v / |v|`, or `None` for a (near-)zero or non-finite vector.
pub fn normalized(v: Vec3) -> Option<Vec3> {
    let length = norm(v);
    (length.is_finite() && length > 1e-12).then(|| mul(v, 1.0 / length))
}

/// Any unit vector perpendicular to a unit `d`.
pub fn perpendicular(d: Vec3) -> Vec3 {
    let helper = if d[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalized(cross(d, helper)).expect("helper is not parallel to d")
}

/// Flip a direction so its first non-negligible component is positive:
/// one canonical sign for a direction whose orientation is arbitrary.
pub fn canonical(d: Vec3) -> Vec3 {
    match d.iter().find(|c| c.abs() > 1e-9) {
        Some(c) if *c < 0.0 => mul(d, -1.0),
        _ => d,
    }
}

pub fn centroid(points: &[Vec3]) -> Vec3 {
    let mut sum = [0.0; 3];
    for p in points {
        sum = add(sum, *p);
    }
    mul(sum, 1.0 / points.len().max(1) as f64)
}

/// Centroid and scatter matrix (sum of outer products about the
/// centroid) of a point set.
pub fn scatter(points: &[Vec3]) -> (Vec3, [[f64; 3]; 3]) {
    let c = centroid(points);
    let mut m = [[0.0; 3]; 3];
    for p in points {
        let d = sub(*p, c);
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += d[i] * d[j];
            }
        }
    }
    (c, m)
}

/// Eigenvalues (ascending) and matching unit eigenvectors of a symmetric
/// 3x3 matrix, by Jacobi rotations.
pub fn eigen_symmetric(mut m: [[f64; 3]; 3]) -> ([f64; 3], [Vec3; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let off = m[0][1].abs() + m[0][2].abs() + m[1][2].abs();
        let scale = m[0][0].abs() + m[1][1].abs() + m[2][2].abs();
        if off <= 1e-15 * scale.max(1e-300) {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if m[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (mkp, mkq) = (m[k][p], m[k][q]);
                m[k][p] = c * mkp - s * mkq;
                m[k][q] = s * mkp + c * mkq;
            }
            for k in 0..3 {
                let (mpk, mqk) = (m[p][k], m[q][k]);
                m[p][k] = c * mpk - s * mqk;
                m[q][k] = s * mpk + c * mqk;
            }
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| m[a][a].total_cmp(&m[b][b]));
    let values = [
        m[order[0]][order[0]],
        m[order[1]][order[1]],
        m[order[2]][order[2]],
    ];
    let vectors = std::array::from_fn(|i| {
        let column = order[i];
        normalized([v[0][column], v[1][column], v[2][column]]).unwrap_or([0.0; 3])
    });
    (values, vectors)
}

/// Solve `a x = b` for a small dense system by Gaussian elimination with
/// partial pivoting; `None` when singular.
pub fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in row + 1..n {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// Axis-aligned bounds of a point set as `(min, max)`.
pub fn bounds(points: &[Vec3]) -> (Vec3, Vec3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    (lo, hi)
}

/// xorshift64*: deterministic, so a randomised fit is a pure function of
/// its inputs.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A uniform-grid index over a point set for exact k-nearest-neighbour
/// queries. Cells hold the indices of the points inside them; a query
/// gathers rings of cells around the query point until the k-th nearest
/// candidate is provably inside the searched block.
pub struct Grid<'a> {
    points: &'a [Vec3],
    origin: Vec3,
    cell: f64,
    /// Point indices sorted by cell key; `cells` maps a key to its range.
    order: Vec<u32>,
    cells: HashMap<u64, (u32, u32)>,
}

impl<'a> Grid<'a> {
    /// Index `points` with a cell size that holds about `per_cell` points
    /// if the cloud were spread over the two largest extents of its
    /// bounding box: right for the scanned surfaces this serves, and a
    /// volume-filling cloud merely makes queries expand a ring or two.
    pub fn new(points: &'a [Vec3], per_cell: usize) -> Self {
        let (lo, hi) = bounds(points);
        let mut extent = sub(hi, lo);
        extent.sort_by(|a, b| b.total_cmp(a));
        let share = per_cell.max(1) as f64 / points.len().max(1) as f64;
        let cell = if extent[1] > 1e-9 * extent[0] {
            (extent[0] * extent[1] * share).sqrt()
        } else {
            extent[0] * share
        }
        .max(1e-9);
        let mut keyed: Vec<(u64, u32)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (Self::key_at(lo, cell, *p), i as u32))
            .collect();
        keyed.sort_unstable();
        let mut cells = HashMap::new();
        let mut start = 0usize;
        while start < keyed.len() {
            let key = keyed[start].0;
            let mut end = start;
            while end < keyed.len() && keyed[end].0 == key {
                end += 1;
            }
            cells.insert(key, (start as u32, (end - start) as u32));
            start = end;
        }
        Self {
            points,
            origin: lo,
            cell,
            order: keyed.into_iter().map(|(_, i)| i).collect(),
            cells,
        }
    }

    pub fn cell_size(&self) -> f64 {
        self.cell
    }

    fn coords(origin: Vec3, cell: f64, p: Vec3) -> [i64; 3] {
        std::array::from_fn(|a| ((p[a] - origin[a]) / cell).floor() as i64)
    }

    fn key(c: [i64; 3]) -> u64 {
        // 21 bits per axis, offset so nearby negative coordinates pack too.
        let pack = |v: i64| ((v + (1 << 20)).clamp(0, (1 << 21) - 1)) as u64;
        pack(c[0]) | pack(c[1]) << 21 | pack(c[2]) << 42
    }

    fn key_at(origin: Vec3, cell: f64, p: Vec3) -> u64 {
        Self::key(Self::coords(origin, cell, p))
    }

    /// The `k` nearest points to `p` (the query point itself included
    /// when it is in the set), nearest first, as `(index, distance)`.
    /// Fewer come back only when the set has fewer than `k` points
    /// within `max_rings` cells of the query.
    pub fn nearest(&self, p: Vec3, k: usize, max_rings: i64) -> Vec<(u32, f64)> {
        let centre = Self::coords(self.origin, self.cell, p);
        let mut found: Vec<(u32, f64)> = Vec::new();
        let mut ring = 0i64;
        loop {
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    for dz in -ring..=ring {
                        if dx.abs() != ring && dy.abs() != ring && dz.abs() != ring {
                            continue;
                        }
                        let key = Self::key([centre[0] + dx, centre[1] + dy, centre[2] + dz]);
                        if let Some(&(start, len)) = self.cells.get(&key) {
                            for &i in &self.order[start as usize..(start + len) as usize] {
                                found.push((i, norm(sub(self.points[i as usize], p))));
                            }
                        }
                    }
                }
            }
            let covered = ring as f64 * self.cell;
            if found.len() >= k {
                found.select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1));
                if found[k - 1].1 <= covered || ring >= max_rings {
                    found.truncate(k);
                    found.sort_by(|a, b| a.1.total_cmp(&b.1));
                    return found;
                }
            } else if ring >= max_rings {
                found.sort_by(|a, b| a.1.total_cmp(&b.1));
                return found;
            }
            ring += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_recovers_axes_of_a_flat_scatter() {
        // Points spread 3 along x, 1 along y, none along z.
        let points: Vec<Vec3> = (0..40)
            .map(|i| {
                let t = i as f64 / 40.0 * std::f64::consts::TAU;
                [3.0 * t.cos(), t.sin(), 0.0]
            })
            .collect();
        let (_, m) = scatter(&points);
        let (values, vectors) = eigen_symmetric(m);
        assert!(values[0].abs() < 1e-9);
        assert!(values[1] < values[2]);
        assert!(vectors[0][2].abs() > 0.999, "{:?}", vectors[0]);
        assert!(vectors[2][0].abs() > 0.999, "{:?}", vectors[2]);
    }

    #[test]
    fn solve_and_vectors() {
        let x = solve(vec![vec![2.0, 1.0], vec![1.0, 3.0]], vec![3.0, 4.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
        assert!(solve(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![1.0, 2.0]).is_none());
        let d = normalized([0.0, 3.0, 4.0]).unwrap();
        assert!(dot(perpendicular(d), d).abs() < 1e-12);
        assert_eq!(canonical([-0.0, -1.0, 2.0]), [0.0, 1.0, -2.0]);
    }

    #[test]
    fn grid_finds_exact_nearest_neighbours() {
        let mut rng = Rng(42);
        let points: Vec<Vec3> = (0..3000)
            .map(|_| {
                // A surface: z is a function of x, y.
                let x = rng.unit() * 2.0 - 1.0;
                let y = rng.unit() * 2.0 - 1.0;
                [x, y, 0.3 * (3.0 * x).sin() * y]
            })
            .collect();
        let grid = Grid::new(&points, 8);
        for &query in &[points[17], points[2999], [0.1, 0.2, 5.0]] {
            let got = grid.nearest(query, 10, 64);
            let mut brute: Vec<(u32, f64)> = points
                .iter()
                .enumerate()
                .map(|(i, p)| (i as u32, norm(sub(*p, query))))
                .collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1));
            brute.truncate(10);
            assert_eq!(got.len(), 10);
            for (g, b) in got.iter().zip(&brute) {
                assert!((g.1 - b.1).abs() < 1e-12, "{g:?} vs {b:?}");
            }
        }
        let tiny = [[0.0; 3], [1.0, 0.0, 0.0]];
        let grid = Grid::new(&tiny, 8);
        assert_eq!(grid.nearest([0.0; 3], 5, 4).len(), 2);
    }
}
