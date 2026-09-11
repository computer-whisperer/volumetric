//! Marker detection: local threshold, dark components, outer borders,
//! quads, sub-pixel corners by edge fitting, perspective sampling and
//! dictionary lookup, for every family asked for in one pass.
//!
//! Large pictures are searched for quads on a reduction (a 26 MP still's
//! marker cells are tens of pixels, beyond any local-threshold window),
//! and every corner is refined and every cell read at full resolution.
//!
//! Corners come out in the marker's canonical order (top-left first,
//! clockwise as printed) in the view set's pixel convention (pixel
//! centres at `+0.5`).

use crate::dict::Dictionary;
use crate::gray::{Binary, Gray};
use crate::linalg::smallest_eigenvector;

/// Detector choices; the defaults follow OpenCV's, with thresholds scaled
/// to the picture instead of fixed.
#[derive(Clone, Debug)]
pub struct DetectParams {
    /// Local-threshold window sizes, pixels; empty picks three from the
    /// picture size (1/100, 1/50 and 1/25 of the shorter side).
    pub windows: Vec<u32>,
    /// Pixels darker than the local mean by this much are ink.
    pub threshold_constant: i32,
    /// A candidate's perimeter must be at least this fraction of the
    /// longer side.
    pub min_perimeter_rate: f64,
    /// A dark component wider than this fraction of the longer side is
    /// not a marker.
    pub max_side_rate: f64,
    /// Polygon approximation accuracy, as a fraction of the perimeter.
    pub approx_rate: f64,
    /// Corners closer than this fraction of the perimeter reject a quad.
    pub min_corner_distance_rate: f64,
    /// Border cells that may read white before a candidate is rejected,
    /// as a fraction of the border cells.
    pub border_error_rate: f64,
    /// A quad's shortest side over its longest must reach this: slivers
    /// are never markers.
    pub min_aspect: f64,
    /// Refine the corners on the grey picture by fitting the edges.
    pub refine: bool,
    /// Quads are searched on the picture reduced to about this many
    /// pixels on its longer side (0 = full resolution); corners and cells
    /// are always read at full resolution.
    pub search_px: u32,
}

impl Default for DetectParams {
    fn default() -> Self {
        Self {
            windows: Vec::new(),
            threshold_constant: 7,
            min_perimeter_rate: 0.03,
            max_side_rate: 0.5,
            approx_rate: 0.03,
            min_corner_distance_rate: 0.05,
            border_error_rate: 0.35,
            min_aspect: 0.15,
            refine: true,
            search_px: 1600,
        }
    }
}

/// One marker found in a picture.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub id: u32,
    /// The dictionary the id is from: `5x5_100`, `4x4_50` or `36h11`.
    pub family: &'static str,
    /// Canonical order: top-left, top-right, bottom-right, bottom-left.
    pub corners: [[f64; 2]; 4],
    /// Clockwise quarter turns the picture showed the marker at.
    pub rotation: u32,
    /// Bits corrected to identify it.
    pub distance: u32,
    /// RMS distance of the fitted edge points to the fitted sides,
    /// pixels (0 when not refined).
    pub fit_px: f64,
}

impl Detection {
    pub fn centre(&self) -> [f64; 2] {
        let mut c = [0.0; 2];
        for p in &self.corners {
            c[0] += p[0] * 0.25;
            c[1] += p[1] * 0.25;
        }
        c
    }

    pub fn perimeter(&self) -> f64 {
        (0..4)
            .map(|i| distance(self.corners[i], self.corners[(i + 1) % 4]))
            .sum()
    }
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Window sizes for a picture: three odd sizes from the shorter side.
pub fn auto_windows(width: u32, height: u32) -> Vec<u32> {
    let short = width.min(height);
    let mut out: Vec<u32> = [100, 50, 25]
        .into_iter()
        .map(|d| {
            let w = (short / d).max(3);
            if w % 2 == 0 { w + 1 } else { w }
        })
        .collect();
    out.dedup();
    out
}

/// The integer reduction that brings the longer side to about
/// `search_px` (1 when it already is, or when `search_px` is 0).
pub fn search_factor(width: u32, height: u32, search_px: u32) -> u32 {
    if search_px == 0 {
        return 1;
    }
    let longer = f64::from(width.max(height));
    ((longer / f64::from(search_px)).round() as u32).max(1)
}

/// Finds the dictionaries' markers in the picture: one search, each quad
/// read against every family, the nearest code wins.
pub fn detect(gray: &Gray, dicts: &[&Dictionary], params: &DetectParams) -> Vec<Detection> {
    let Some(min_cells) = dicts.iter().map(|d| d.size + 2).min() else {
        return Vec::new();
    };
    let factor = search_factor(gray.width, gray.height, params.search_px);
    let reduced;
    let search: &Gray = if factor > 1 {
        reduced = gray.downsampled(factor);
        &reduced
    } else {
        gray
    };
    let windows = if params.windows.is_empty() {
        auto_windows(search.width, search.height)
    } else {
        params.windows.clone()
    };
    let longer = f64::from(search.width.max(search.height));
    // Every cell needs a couple of pixels to read, at full resolution;
    // on the reduction a quad only has to be findable.
    let min_side_full = 2.0 * f64::from(min_cells);
    let min_side = (min_side_full / f64::from(factor)).max(6.0);
    let min_perimeter = (params.min_perimeter_rate * longer).max(4.0 * min_side);
    let max_side = params.max_side_rate * longer;
    let scale = f64::from(factor);
    let mut found = Vec::new();
    for window in windows {
        let binary = search.threshold_local(window, params.threshold_constant);
        for quad in quads(&binary, params, min_perimeter, max_side, min_side) {
            let quad = quad.map(|p| [p[0] * scale, p[1] * scale]);
            if shortest_side(&quad) < min_side_full {
                continue;
            }
            // The reduction's corners are off by up to a reduced pixel;
            // the edge search reaches that much further.
            let (quad, fit_px) = if params.refine {
                // A second pass from the settled quad reaches the whole
                // of a side the first pass only found part of.
                let (once, fit) = refine_quad(gray, &quad, scale - 1.0);
                if fit < shortest_side(&quad) && (0..4).any(|i| distance(once[i], quad[i]) > 2.0) {
                    refine_quad(gray, &once, 0.0)
                } else {
                    (once, fit)
                }
            } else {
                (quad, 0.0)
            };
            if let Some(detection) = decode(gray, &quad, fit_px, dicts, params) {
                found.push(detection);
            }
        }
    }
    dedupe(found)
}

fn shortest_side(quad: &[[f64; 2]; 4]) -> f64 {
    (0..4)
        .map(|i| distance(quad[i], quad[(i + 1) % 4]))
        .fold(f64::INFINITY, f64::min)
}

/// Candidate quads: the outer borders of dark components that
/// approximate to a convex four-sided polygon, ordered clockwise on the
/// screen.
fn quads(
    binary: &Binary,
    params: &DetectParams,
    min_perimeter: f64,
    max_side: f64,
    min_side: f64,
) -> Vec<[[f64; 2]; 4]> {
    let (w, h) = (binary.width as usize, binary.height as usize);
    let mut labels = vec![0u32; w * h];
    let mut next_label = 1u32;
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !binary.bits[start] || labels[start] != 0 {
            continue;
        }
        let label = next_label;
        next_label += 1;
        labels[start] = label;
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        let mut pixels = 0usize;
        while let Some(i) = stack.pop() {
            pixels += 1;
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if binary.bits[j] && labels[j] == 0 {
                        labels[j] = label;
                        stack.push(j);
                    }
                }
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as f64, (y1 - y0 + 1) as f64);
        // Touching the picture edge, too small to hold a marker, or far
        // too large to be one.
        if x0 == 0 || y0 == 0 || x1 + 1 == w || y1 + 1 == h {
            continue;
        }
        if 2.0 * (bw + bh) < min_perimeter || bw.max(bh) > max_side {
            continue;
        }
        let contour = trace_outer_border(&labels, w, h, start, label, pixels);
        let perimeter: f64 = contour
            .iter()
            .zip(contour.iter().cycle().skip(1))
            .map(|(a, b)| distance(*a, *b))
            .sum();
        if perimeter < min_perimeter {
            continue;
        }
        let Some(mut quad) = approximate_quad(&contour, params.approx_rate * perimeter) else {
            continue;
        };
        let sides: Vec<f64> = (0..4)
            .map(|i| distance(quad[i], quad[(i + 1) % 4]))
            .collect();
        let (shortest, longest) = sides.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), s| {
            (lo.min(*s), hi.max(*s))
        });
        if shortest < (params.min_corner_distance_rate * perimeter).max(min_side)
            || shortest < params.min_aspect * longest
        {
            continue;
        }
        if !is_convex(&quad) {
            continue;
        }
        if signed_area(&quad) < 0.0 {
            quad.swap(1, 3);
        }
        out.push(quad);
    }
    out
}

/// The outer border of a component, traced clockwise on the screen from
/// its first pixel in raster order (Moore neighbourhood), as pixel
/// centres. `pixels` is the component's size, which bounds the trace.
fn trace_outer_border(
    labels: &[u32],
    w: usize,
    h: usize,
    start: usize,
    label: u32,
    pixels: usize,
) -> Vec<[f64; 2]> {
    // Clockwise on screen (y down) starting west: W, NW, N, NE, E, SE, S, SW.
    const DIRS: [(i64, i64); 8] = [
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    ];
    let inside = |x: i64, y: i64| {
        x >= 0
            && y >= 0
            && x < w as i64
            && y < h as i64
            && labels[y as usize * w + x as usize] == label
    };
    // From `cur` with backtrack `back`, the next border pixel and the
    // backtrack to continue from (the ring position checked just before
    // it, which neighbours it).
    let step = |cur: (i64, i64), back: (i64, i64)| -> Option<((i64, i64), (i64, i64))> {
        let start_k = DIRS
            .iter()
            .position(|d| (cur.0 + d.0, cur.1 + d.1) == back)
            .expect("backtrack is a neighbour");
        for k in 1..=8 {
            let d = (start_k + k) % 8;
            let next = (cur.0 + DIRS[d].0, cur.1 + DIRS[d].1);
            if inside(next.0, next.1) {
                let new_back = (cur.0 + DIRS[(d + 7) % 8].0, cur.1 + DIRS[(d + 7) % 8].1);
                return Some((next, new_back));
            }
        }
        None
    };
    let start_xy = ((start % w) as i64, (start / w) as i64);
    let mut points = vec![[start_xy.0 as f64 + 0.5, start_xy.1 as f64 + 0.5]];
    // The backtrack starts west of the start pixel, which is outside the
    // component by raster order.
    let start_back = (start_xy.0 - 1, start_xy.1);
    let Some((second, mut back)) = step(start_xy, start_back) else {
        return points; // an isolated pixel
    };
    let mut cur = second;
    // A thin spur is walked twice, so the border is at most twice the
    // component; the bound only matters if the criterion were missed.
    let limit = 2 * pixels + 8;
    for _ in 0..limit {
        // Jacob's stopping criterion: back at the start and about to
        // leave for the same second pixel as at first.
        if cur == start_xy && step(cur, back).is_some_and(|(next, _)| next == second) {
            break;
        }
        points.push([cur.0 as f64 + 0.5, cur.1 as f64 + 0.5]);
        let Some((next, new_back)) = step(cur, back) else {
            break;
        };
        cur = next;
        back = new_back;
    }
    points
}

/// Douglas–Peucker on a closed contour, split at its first point and the
/// point farthest from it; the two split points stay vertices only where
/// the contour actually turns (the first point is wherever raster order
/// found the component, which on a nearly level top edge is tens of
/// pixels from the corner). `Some` only when exactly four vertices
/// remain.
fn approximate_quad(contour: &[[f64; 2]], epsilon: f64) -> Option<[[f64; 2]; 4]> {
    if contour.len() < 4 {
        return None;
    }
    let a = 0usize;
    let b = (1..contour.len()).max_by(|&i, &j| {
        distance(contour[a], contour[i])
            .partial_cmp(&distance(contour[a], contour[j]))
            .unwrap()
    })?;
    let mut vertices = Vec::new();
    vertices.push(a);
    douglas_peucker(contour, a, b, epsilon, &mut vertices);
    vertices.push(b);
    douglas_peucker_wrapped(contour, b, a, epsilon, &mut vertices);
    // Drop a split point that lies on the line between its neighbours.
    for split in [b, a] {
        if vertices.len() < 4 {
            break;
        }
        let k = vertices
            .iter()
            .position(|&v| v == split)
            .expect("split is a vertex");
        let n = vertices.len();
        let (prev, next) = (vertices[(k + n - 1) % n], vertices[(k + 1) % n]);
        if point_line_distance(contour[split], contour[prev], contour[next]) <= epsilon {
            vertices.remove(k);
        }
    }
    if vertices.len() != 4 {
        return None;
    }
    Some([
        contour[vertices[0]],
        contour[vertices[1]],
        contour[vertices[2]],
        contour[vertices[3]],
    ])
}

/// Vertices strictly between `from` and `to` (increasing indices) that
/// the tolerance keeps, in order.
fn douglas_peucker(
    contour: &[[f64; 2]],
    from: usize,
    to: usize,
    epsilon: f64,
    out: &mut Vec<usize>,
) {
    if to <= from + 1 {
        return;
    }
    let (mut best, mut best_d) = (from, 0.0);
    for i in (from + 1)..to {
        let d = point_line_distance(contour[i], contour[from], contour[to]);
        if d > best_d {
            best = i;
            best_d = d;
        }
    }
    if best_d > epsilon {
        douglas_peucker(contour, from, best, epsilon, out);
        out.push(best);
        douglas_peucker(contour, best, to, epsilon, out);
    }
}

/// The same across the wrap from `from` to `to` (going past the end).
fn douglas_peucker_wrapped(
    contour: &[[f64; 2]],
    from: usize,
    to: usize,
    epsilon: f64,
    out: &mut Vec<usize>,
) {
    let n = contour.len();
    let count = (to + n - from) % n;
    if count <= 1 {
        return;
    }
    let (mut best, mut best_d) = (from, 0.0);
    for k in 1..count {
        let i = (from + k) % n;
        let d = point_line_distance(contour[i], contour[from], contour[to]);
        if d > best_d {
            best = i;
            best_d = d;
        }
    }
    if best_d > epsilon {
        douglas_peucker_wrapped(contour, from, best, epsilon, out);
        out.push(best);
        douglas_peucker_wrapped(contour, best, to, epsilon, out);
    }
}

fn point_line_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        return distance(p, a);
    }
    ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len
}

fn signed_area(q: &[[f64; 2]; 4]) -> f64 {
    (0..4)
        .map(|i| {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        * 0.5
}

fn is_convex(q: &[[f64; 2]; 4]) -> bool {
    let mut sign = 0.0f64;
    for i in 0..4 {
        let (a, b, c) = (q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() < 1e-9 {
            return false;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
}

/// The homography taking `src` to `dst` (DLT with normalisation), as a
/// row-major 3x3.
pub fn homography(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<[[f64; 3]; 3]> {
    if src.len() < 4 || src.len() != dst.len() {
        return None;
    }
    let normalise = |pts: &[[f64; 2]]| {
        let n = pts.len() as f64;
        let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
        let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
        let mean_d = pts
            .iter()
            .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
            .sum::<f64>()
            / n;
        let s = if mean_d > 1e-12 {
            std::f64::consts::SQRT_2 / mean_d
        } else {
            1.0
        };
        let t = [[s, 0.0, -s * cx], [0.0, s, -s * cy], [0.0, 0.0, 1.0]];
        let mapped: Vec<[f64; 2]> = pts
            .iter()
            .map(|p| [s * (p[0] - cx), s * (p[1] - cy)])
            .collect();
        (t, mapped)
    };
    let (ts, s) = normalise(src);
    let (td, d) = normalise(dst);
    let mut ata = vec![vec![0.0; 9]; 9];
    for (p, q) in s.iter().zip(&d) {
        let rows = [
            [
                -p[0],
                -p[1],
                -1.0,
                0.0,
                0.0,
                0.0,
                q[0] * p[0],
                q[0] * p[1],
                q[0],
            ],
            [
                0.0,
                0.0,
                0.0,
                -p[0],
                -p[1],
                -1.0,
                q[1] * p[0],
                q[1] * p[1],
                q[1],
            ],
        ];
        for row in &rows {
            for i in 0..9 {
                for j in 0..9 {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
    }
    let h = smallest_eigenvector(&ata);
    let hn = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];
    // H = Td^-1 · Hn · Ts.
    let td_inv = crate::linalg::mat3_inverse(&td)?;
    let hm = crate::linalg::mat3_mul(&crate::linalg::mat3_mul(&td_inv, &hn), &ts);
    let scale = hm[2][2];
    if scale.abs() < 1e-300 {
        return None;
    }
    Some(hm.map(|row| row.map(|v| v / scale)))
}

/// Applies a homography to a point.
pub fn apply_homography(h: &[[f64; 3]; 3], p: [f64; 2]) -> [f64; 2] {
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    [
        (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w,
        (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w,
    ]
}

/// Otsu's threshold over a set of values: the split maximising the
/// between-class variance.
fn otsu(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n < 2 {
        return sorted.first().copied().unwrap_or(128.0);
    }
    let total: f64 = sorted.iter().sum();
    let (mut best, mut best_var) = (sorted[0], -1.0);
    let mut low_sum = 0.0;
    for k in 1..n {
        low_sum += sorted[k - 1];
        let (nl, nh) = (k as f64, (n - k) as f64);
        let (ml, mh) = (low_sum / nl, (total - low_sum) / nh);
        let var = nl * nh * (ml - mh).powi(2);
        if var > best_var {
            best_var = var;
            best = (sorted[k - 1] + sorted[k]) * 0.5;
        }
    }
    best
}

/// Reads a quad's cells at each family's grid, checks the border and
/// identifies it; the corners come out in the marker's canonical order.
fn decode(
    gray: &Gray,
    quad: &[[f64; 2]; 4],
    fit_px: f64,
    dicts: &[&Dictionary],
    params: &DetectParams,
) -> Option<Detection> {
    let mut best: Option<(crate::dict::Identification, &Dictionary)> = None;
    for dict in dicts {
        let Some(bits) = read_cells(gray, quad, dict.size, params) else {
            continue;
        };
        if let Some(found) = dict.identify(bits)
            && best.is_none_or(|(b, _)| found.distance < b.distance)
        {
            best = Some((found, dict));
        }
    }
    let (found, dict) = best?;
    let k = found.rotation as usize;
    let corners = [
        quad[k],
        quad[(k + 1) % 4],
        quad[(k + 2) % 4],
        quad[(k + 3) % 4],
    ];
    Some(Detection {
        id: found.id,
        family: dict.name,
        corners,
        rotation: found.rotation,
        distance: found.distance,
        fit_px,
    })
}

/// The inner `n x n` bits of a quad read as an `(n + 2)`-cell grid with a
/// dark border, or `None` when the border is not dark enough.
fn read_cells(gray: &Gray, quad: &[[f64; 2]; 4], n: u32, params: &DetectParams) -> Option<u64> {
    let cells = n + 2;
    let side = f64::from(cells);
    let canonical = [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]];
    let h = homography(&canonical, quad)?;
    const SAMPLES: u32 = 4;
    const MARGIN: f64 = 0.13;
    let mut means = vec![0.0; (cells * cells) as usize];
    for r in 0..cells {
        for c in 0..cells {
            let mut acc = 0.0;
            for sy in 0..SAMPLES {
                for sx in 0..SAMPLES {
                    let fx = f64::from(c)
                        + MARGIN
                        + (1.0 - 2.0 * MARGIN) * (f64::from(sx) + 0.5) / f64::from(SAMPLES);
                    let fy = f64::from(r)
                        + MARGIN
                        + (1.0 - 2.0 * MARGIN) * (f64::from(sy) + 0.5) / f64::from(SAMPLES);
                    let p = apply_homography(&h, [fx, fy]);
                    acc += gray.sample(p[0], p[1]);
                }
            }
            means[(r * cells + c) as usize] = acc / f64::from(SAMPLES * SAMPLES);
        }
    }
    let threshold = otsu(&means);
    let white = |r: u32, c: u32| means[(r * cells + c) as usize] > threshold;
    let mut border_errors = 0u32;
    for i in 0..cells {
        for (r, c) in [(0, i), (cells - 1, i), (i, 0), (i, cells - 1)] {
            if white(r, c) {
                border_errors += 1;
            }
        }
    }
    // Corners were counted twice; the border has 4·cells − 4 cells.
    let border_cells = 4 * cells - 4;
    if f64::from(border_errors) > params.border_error_rate * f64::from(border_cells) * 2.0 {
        return None;
    }
    let mut bits = 0u64;
    for r in 0..n {
        for c in 0..n {
            if white(r + 1, c + 1) {
                bits |= 1 << (n * n - 1 - (r * n + c));
            }
        }
    }
    Some(bits)
}

/// Sub-pixel corners: along each side, the paper-to-ink crossing on the
/// inward normal at many points, a robust quadratic through them (a
/// curled card bows an edge by pixels, the lens by one), and each corner
/// at the intersection of straight lines through the edge points of the
/// half-sides meeting there; the rms of the edge points to their curves
/// comes back as the fit. The edge search reaches `slack` pixels further
/// for a quad found on a reduced picture.
fn refine_quad(gray: &Gray, quad: &[[f64; 2]; 4], slack: f64) -> ([[f64; 2]; 4], f64) {
    let mut sides: Vec<Curve> = Vec::with_capacity(4);
    let mut residual_sq = 0.0;
    let mut residual_n = 0usize;
    // A refinement that cannot fit reports a fit as bad as the quad is
    // big, so a candidate it failed on never outranks one it settled.
    let unfit = shortest_side(quad).max(1.0);
    let centre = [
        quad.iter().map(|p| p[0]).sum::<f64>() * 0.25,
        quad.iter().map(|p| p[1]).sum::<f64>() * 0.25,
    ];
    for i in 0..4 {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        let len = distance(a, b);
        if len < 4.0 {
            return (*quad, unfit);
        }
        let dir = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
        // The normal pointing into the marker, so the crossing looked for
        // is the border's outer edge, paper to ink, and not a bit's.
        let mut normal = [-dir[1], dir[0]];
        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        if (centre[0] - mid[0]) * normal[0] + (centre[1] - mid[1]) * normal[1] < 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        let count = ((len / 4.0) as usize).clamp(8, 60);
        // The search stays inside the border cell on the inside and the
        // quiet zone outside; the smallest cell any family has is an
        // eighth of the side.
        let cell = len / 8.0;
        let reach = ((len * 0.1).clamp(2.0, 12.0) + slack)
            .min(1.2 * cell)
            .max(1.5);
        const STEP: f64 = 0.5;
        let mut points: Vec<[f64; 2]> = Vec::with_capacity(count);
        // Clear of the corners, where blur rounds the edge: six pixels
        // or four percent of the side, whichever is more.
        let t_min = (6.0 / len).max(0.04);
        for k in 0..count {
            let t = t_min + (1.0 - 2.0 * t_min) * k as f64 / (count - 1) as f64;
            let p = [a[0] + dir[0] * t * len, a[1] + dir[1] * t * len];
            let steps = (reach / STEP) as i64;
            let profile: Vec<f64> = (-steps..=steps)
                .map(|s| {
                    let d = s as f64 * STEP;
                    gray.sample(p[0] + normal[0] * d, p[1] + normal[1] * d)
                })
                .collect();
            // The darkening gradient inward, by central differences.
            let grad: Vec<f64> = (0..profile.len())
                .map(|j| {
                    if j == 0 || j + 1 == profile.len() {
                        0.0
                    } else {
                        (profile[j - 1] - profile[j + 1]).max(0.0)
                    }
                })
                .collect();
            let (jmax, gmax) = grad
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .map(|(j, g)| (j, *g))
                .unwrap();
            if gmax < 4.0 || jmax == 0 || jmax + 1 == grad.len() {
                continue;
            }
            // The edge sits at the centroid of the gradient around its
            // peak: unbiased for the symmetric profiles blur and
            // anti-aliasing produce, where a parabola through three
            // samples of a plateau is not.
            let lo = jmax.saturating_sub(3).max(1);
            let hi = (jmax + 3).min(grad.len() - 2);
            let (mut wsum, mut ssum) = (0.0, 0.0);
            for (j, g) in grad.iter().enumerate().take(hi + 1).skip(lo) {
                let w = (g - 0.3 * gmax).max(0.0);
                wsum += w;
                ssum += w * (j as f64 - steps as f64) * STEP;
            }
            if wsum <= 0.0 {
                continue;
            }
            let s = ssum / wsum;
            points.push([p[0] + normal[0] * s, p[1] + normal[1] * s]);
        }
        if points.len() < 4 {
            return (*quad, unfit);
        }
        let Some((curve, rms, used)) = fit_curve(a, dir, normal, len, &points) else {
            return (*quad, unfit);
        };
        residual_sq += rms * rms * used as f64;
        residual_n += used;
        sides.push(curve);
    }
    let mut corners = *quad;
    for i in 0..4 {
        // Side i - 1 ends at corner i, side i starts there. The corner is
        // where the edges arrive, so each is a straight line through the
        // edge points of the half of the side nearest the corner (a curled
        // card bows an edge by several pixels, and lifts most at a corner;
        // the lens bends it by one), falling back to the whole side's
        // tangent when the half has too few points.
        let (prev, this) = (&sides[(i + 3) % 4], &sides[i]);
        let end = prev
            .local_line(false)
            .unwrap_or_else(|| prev.tangent_at_end());
        let start = this
            .local_line(true)
            .unwrap_or_else(|| this.tangent_at_start());
        corners[i] = intersect(&end, &start)
            .or_else(|| intersect(&prev.straight(), &this.straight()))
            .unwrap_or(quad[i]);
    }
    let fit = if residual_n > 0 {
        (residual_sq / residual_n as f64).sqrt()
    } else {
        0.0
    };
    // A side whose points mostly went missing, that bends more than a
    // card can curl, or corners that left the quad altogether are no
    // refinement: keep the quad and report a fit as bad as it is. A poor
    // fit alone keeps the corners; the fit says what they are worth.
    let shortest = shortest_side(quad);
    let moved = (0..4)
        .map(|i| distance(corners[i], quad[i]))
        .fold(0.0, f64::max);
    let thin = sides.iter().any(|s| s.points.len() * 3 < s.looked * 2);
    let curled = sides.iter().any(|s| s.bend() > 0.05 * s.len);
    if thin || curled || moved > 0.2 * shortest {
        return (*quad, unfit.max(fit));
    }
    (corners, fit)
}

/// One side's edge as a quadratic in the side's frame: origin at the
/// quad's corner, `t` along the side, `n` along the normal; the edge
/// sits at `n = c0 + c1 t + c2 t²`.
struct Curve {
    origin: [f64; 2],
    dir: [f64; 2],
    normal: [f64; 2],
    len: f64,
    c: [f64; 3],
    /// The straight line through the same points, `n = l0 + l1 t`.
    l: [f64; 2],
    /// The edge points kept, in the side's frame.
    points: Vec<(f64, f64)>,
    looked: usize,
}

impl Curve {
    fn point(&self, t: f64) -> [f64; 2] {
        let n = self.c[0] + self.c[1] * t + self.c[2] * t * t;
        [
            self.origin[0] + self.dir[0] * t + self.normal[0] * n,
            self.origin[1] + self.dir[1] * t + self.normal[1] * n,
        ]
    }

    fn tangent(&self, t: f64) -> Line {
        let slope = self.c[1] + 2.0 * self.c[2] * t;
        let d = [
            self.dir[0] + self.normal[0] * slope,
            self.dir[1] + self.normal[1] * slope,
        ];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
        (self.point(t), [d[0] / l, d[1] / l])
    }

    fn tangent_at_start(&self) -> Line {
        self.tangent(0.0)
    }

    /// A straight line through the kept points in the first (or last)
    /// half of the side, when there are six or more of them.
    fn local_line(&self, at_start: bool) -> Option<Line> {
        let (lo, hi) = if at_start {
            (0.0, self.len * 0.5)
        } else {
            (self.len * 0.5, self.len)
        };
        let pts: Vec<(f64, f64)> = self
            .points
            .iter()
            .copied()
            .filter(|(t, _)| *t >= lo && *t <= hi)
            .collect();
        if pts.len() < 6 {
            return None;
        }
        let n = pts.len() as f64;
        let (st, sn) = pts.iter().fold((0.0, 0.0), |a, (t, v)| (a.0 + t, a.1 + v));
        let (mt, mn) = (st / n, sn / n);
        let (mut stt, mut stn) = (0.0, 0.0);
        for (t, v) in &pts {
            stt += (t - mt) * (t - mt);
            stn += (t - mt) * (v - mn);
        }
        if stt < 1e-9 {
            return None;
        }
        let slope = stn / stt;
        let d = [
            self.dir[0] + self.normal[0] * slope,
            self.dir[1] + self.normal[1] * slope,
        ];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-12);
        Some((
            [
                self.origin[0] + self.dir[0] * mt + self.normal[0] * mn,
                self.origin[1] + self.dir[1] * mt + self.normal[1] * mn,
            ],
            [d[0] / l, d[1] / l],
        ))
    }

    /// The straight-line fit through the same edge points.
    fn straight(&self) -> Line {
        let d = [
            self.dir[0] + self.normal[0] * self.l[1],
            self.dir[1] + self.normal[1] * self.l[1],
        ];
        let n = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-12);
        (
            [
                self.origin[0] + self.normal[0] * self.l[0],
                self.origin[1] + self.normal[1] * self.l[0],
            ],
            [d[0] / n, d[1] / n],
        )
    }

    /// How far the quadratic bends from its straight fit over the side.
    fn bend(&self) -> f64 {
        let mid = self.len * 0.5;
        ((self.c[0] + self.c[1] * mid + self.c[2] * mid * mid) - (self.l[0] + self.l[1] * mid))
            .abs()
    }

    fn tangent_at_end(&self) -> Line {
        self.tangent(self.len)
    }
}

/// Least-squares quadratic through the edge points in the side's frame,
/// refit once without the outliers; the curve, its rms and the points
/// kept. Fewer than six points get a straight line.
fn fit_curve(
    origin: [f64; 2],
    dir: [f64; 2],
    normal: [f64; 2],
    len: f64,
    points: &[[f64; 2]],
) -> Option<(Curve, f64, usize)> {
    let local: Vec<(f64, f64)> = points
        .iter()
        .map(|p| {
            let d = [p[0] - origin[0], p[1] - origin[1]];
            (
                d[0] * dir[0] + d[1] * dir[1],
                d[0] * normal[0] + d[1] * normal[1],
            )
        })
        .collect();
    let fit = |pts: &[(f64, f64)], quadratic: bool| -> Option<([f64; 3], Vec<f64>)> {
        let quadratic = quadratic && pts.len() >= 6;
        // Normal equations in t scaled to the side, for conditioning.
        let s = 1.0 / len.max(1.0);
        let k = if quadratic { 3 } else { 2 };
        let mut ata = [[0.0; 3]; 3];
        let mut atb = [0.0; 3];
        for &(t, n) in pts {
            let row = [1.0, t * s, (t * s) * (t * s)];
            for i in 0..k {
                for j in 0..k {
                    ata[i][j] += row[i] * row[j];
                }
                atb[i] += row[i] * n;
            }
        }
        let c = if quadratic {
            let inv = crate::linalg::mat3_inverse(&ata)?;
            let mut c = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    c[i] += inv[i][j] * atb[j];
                }
            }
            [c[0], c[1] * s, c[2] * s * s]
        } else {
            let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0];
            if det.abs() < 1e-12 {
                return None;
            }
            let c0 = (atb[0] * ata[1][1] - ata[0][1] * atb[1]) / det;
            let c1 = (ata[0][0] * atb[1] - ata[1][0] * atb[0]) / det;
            [c0, c1 * s, 0.0]
        };
        let residuals = pts
            .iter()
            .map(|&(t, n)| n - (c[0] + c[1] * t + c[2] * t * t))
            .collect();
        Some((c, residuals))
    };
    // Two passes of rejection on a robust scale (the median absolute
    // deviation), so a run of points on something touching the marker
    // (a black chair leg against a black border) is dropped even when
    // it is a third of the side.
    let mut kept = local.clone();
    for _ in 0..2 {
        let (_, residuals) = fit(&kept, true)?;
        let mut abs: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = abs[abs.len() / 2];
        let cut = (3.0 * 1.4826 * mad).max(0.3);
        let next: Vec<(f64, f64)> = kept
            .iter()
            .zip(&residuals)
            .filter(|(_, r)| r.abs() <= cut)
            .map(|(p, _)| *p)
            .collect();
        if next.len() == kept.len() {
            break;
        }
        kept = next;
    }
    if kept.len() < 3 || kept.len() * 2 < local.len() {
        return None;
    }
    let (c, residuals) = fit(&kept, true)?;
    let (l, _) = fit(&kept, false)?;
    let rms = (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
    let used = kept.len();
    Some((
        Curve {
            origin,
            dir,
            normal,
            len,
            c,
            l: [l[0], l[1]],
            looked: local.len(),
            points: kept,
        },
        rms,
        used,
    ))
}

/// The corner near `start` where every gradient in the
/// `(2 half + 1)`-pixel window is orthogonal to the offset from it (the
/// `cornerSubPix` iteration, the window moving with the estimate): the
/// intersection of the edges meeting there, or a chessboard's saddle
/// point. `None` when the window is flat or the iteration wanders off
/// by more than the window.
pub fn refine_saddle(
    gray: &Gray,
    start: [f64; 2],
    half: u32,
    iterations: u32,
    epsilon: f64,
) -> Option<[f64; 2]> {
    let half_i = half as i64;
    let sigma2 = (f64::from(half) / std::f64::consts::SQRT_2).powi(2);
    let mut q = start;
    for _ in 0..iterations {
        let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
        let (mut bx, mut by) = (0.0, 0.0);
        for j in -half_i..=half_i {
            for i in -half_i..=half_i {
                let (x, y) = (q[0] + i as f64, q[1] + j as f64);
                let gx = (gray.sample(x + 1.0, y) - gray.sample(x - 1.0, y)) * 0.5;
                let gy = (gray.sample(x, y + 1.0) - gray.sample(x, y - 1.0)) * 0.5;
                let w = (-((i * i + j * j) as f64) / sigma2).exp();
                let (gxx, gxy, gyy) = (w * gx * gx, w * gx * gy, w * gy * gy);
                a += gxx;
                b += gxy;
                c += gyy;
                bx += gxx * x + gxy * y;
                by += gxy * x + gyy * y;
            }
        }
        let det = a * c - b * b;
        if det.abs() < 1e-9 || !det.is_finite() {
            return None;
        }
        let next = [(c * bx - b * by) / det, (a * by - b * bx) / det];
        let step = distance(next, q);
        if !step.is_finite() || distance(next, start) > f64::from(half) {
            return None;
        }
        q = next;
        if step < epsilon {
            break;
        }
    }
    Some(q)
}

/// A line as a point and a unit direction.
type Line = ([f64; 2], [f64; 2]);

fn intersect(a: &Line, b: &Line) -> Option<[f64; 2]> {
    let (p, d) = a;
    let (q, e) = b;
    let denom = d[0] * e[1] - d[1] * e[0];
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / denom;
    Some([p[0] + d[0] * t, p[1] + d[1] * t])
}

/// Collapses detections of the same quad from different windows (and
/// nested false readings) to the best fit.
fn dedupe(mut found: Vec<Detection>) -> Vec<Detection> {
    found.sort_by(|a, b| {
        (a.distance, a.fit_px)
            .partial_cmp(&(b.distance, b.fit_px))
            .unwrap()
    });
    let mut kept: Vec<Detection> = Vec::new();
    for d in found {
        let centre = d.centre();
        let size = d.perimeter() * 0.25;
        let duplicate = kept.iter().any(|k| {
            let kc = k.centre();
            distance(centre, kc) < 0.5 * size.min(k.perimeter() * 0.25)
        });
        if !duplicate {
            kept.push(d);
        }
    }
    kept.sort_by_key(|d| (d.id, d.family));
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Render, render, square_marker};
    use volumetric_abi::viewset::{CameraModel, View};

    /// A tilted camera over a floor with five markers, one of them turned
    /// a quarter turn; the picture is 1280 x 960 at f = 1000.
    fn scene() -> (CameraModel, View, Vec<volumetric_abi::viewset::Marker>) {
        let camera = CameraModel::pinhole(1280, 960, 1000.0, 1000.0, 640.0, 480.0);
        // Camera 1.2 m up, pitched 50 degrees down toward +y along the floor.
        let pitch: f64 = 50f64.to_radians();
        let (s, c) = pitch.sin_cos();
        // Camera axes in world: x right = world x; z forward = pitched
        // from +y (floor, forward) down toward -z; y down = the rest.
        let forward = [0.0, c, -s];
        let down = [0.0, -s, -c];
        let view = View::posed(
            "cam",
            0,
            [
                1.0, down[0], forward[0], 0.1, //
                0.0, down[1], forward[1], -0.4, //
                0.0, down[2], forward[2], 1.2, //
            ],
        );
        let up_floor = [0.0, 1.0, 0.0];
        let right = [1.0, 0.0, 0.0];
        let markers = vec![
            square_marker(0, [-0.45, 0.85, 0.0], 0.12, right, [0.0, -1.0, 0.0]),
            square_marker(1, [0.3, 0.9, 0.0], 0.12, right, [0.0, -1.0, 0.0]),
            square_marker(2, [-0.35, 0.45, 0.0], 0.12, right, [0.0, -1.0, 0.0]),
            // Turned a quarter turn: its top edge runs along +y.
            square_marker(5, [0.25, 0.4, 0.0], 0.12, up_floor, right),
            square_marker(49, [-0.05, 0.65, 0.0], 0.16, right, [0.0, -1.0, 0.0]),
        ];
        (camera, view, markers)
    }

    fn check(
        picture: &Gray,
        camera: &CameraModel,
        view: &View,
        markers: &[volumetric_abi::viewset::Marker],
        tolerance: f64,
    ) {
        let dict = Dictionary::aruco_5x5_100();
        let found = detect(picture, &[&dict], &DetectParams::default());
        let ids: Vec<u32> = found.iter().map(|d| d.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 5, 49], "{found:?}");
        assert!(found.iter().all(|d| d.family == "5x5_100"));
        let mut errors = Vec::new();
        for (d, m) in found.iter().zip(markers) {
            for j in 0..4 {
                let truth = view.project(camera, m.corners[j]).unwrap();
                errors.push((d.id, j, distance(d.corners[j], truth)));
            }
            assert!(d.fit_px < 0.5, "{d:?}");
        }
        let worst = errors
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();
        let mean = errors.iter().map(|e| e.2).sum::<f64>() / errors.len() as f64;
        assert!(
            worst.2 < tolerance,
            "worst corner error {:.3} px (marker {} corner {}), mean {mean:.3} px: {errors:?}",
            worst.2,
            worst.0,
            worst.1
        );
    }

    #[test]
    fn markers_are_found_with_sub_pixel_corners_in_canonical_order() {
        let (camera, view, markers) = scene();
        let dict = Dictionary::aruco_5x5_100();
        // Six samples per pixel side keep the picture's own edge
        // quantisation (a sixth of a pixel) below the accuracy checked.
        let clean = render(
            &camera,
            &view,
            &markers,
            &dict,
            &Render {
                supersample: 6,
                ..Render::default()
            },
        );
        // A clean render quantises its edges to a sixth of a pixel.
        check(&clean, &camera, &view, &markers, 0.3);
        let blurred = render(
            &camera,
            &view,
            &markers,
            &dict,
            &Render {
                blur_sigma: 1.2,
                ..Render::default()
            },
        );
        check(&blurred, &camera, &view, &markers, 0.35);
        // The turned marker's top edge runs up the picture, a quarter
        // turn counter-clockwise, which is three clockwise; the others
        // are upright.
        let found = detect(&clean, &[&dict], &DetectParams::default());
        assert_eq!(found.iter().find(|d| d.id == 5).unwrap().rotation, 3);
        assert_eq!(found.iter().find(|d| d.id == 0).unwrap().rotation, 0);
        // The wrong dictionary finds nothing; asking for both families
        // finds the same five as swatches.
        let board = Dictionary::aruco_4x4_50();
        assert!(detect(&clean, &[&board], &DetectParams::default()).is_empty());
        let both = detect(&clean, &[&board, &dict], &DetectParams::default());
        assert_eq!(both.len(), 5);
        assert!(both.iter().all(|d| d.family == "5x5_100"));
    }

    #[test]
    fn a_large_picture_is_searched_reduced_and_read_in_full() {
        // The same scene at 3200 x 2400 (f = 2500): the search runs at
        // half size and the corners still come out at full accuracy.
        let (_, view, markers) = scene();
        let camera = CameraModel::pinhole(3200, 2400, 2500.0, 2500.0, 1600.0, 1200.0);
        let dict = Dictionary::aruco_5x5_100();
        let picture = render(
            &camera,
            &view,
            &markers,
            &dict,
            &Render {
                blur_sigma: 1.0,
                ..Render::default()
            },
        );
        assert_eq!(search_factor(3200, 2400, 1600), 2);
        assert_eq!(search_factor(6192, 4128, 1600), 4);
        assert_eq!(search_factor(1280, 960, 1600), 1);
        assert_eq!(search_factor(6192, 4128, 0), 1);
        check(&picture, &camera, &view, &markers, 0.35);
        // Full-resolution search finds the same markers.
        let full = detect(
            &picture,
            &[&dict],
            &DetectParams {
                search_px: 0,
                ..DetectParams::default()
            },
        );
        assert_eq!(full.len(), 5);
    }

    #[test]
    fn homography_round_trips_and_otsu_splits() {
        let src = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let dst = [[10.0, 12.0], [52.0, 15.0], [55.0, 60.0], [8.0, 58.0]];
        let h = homography(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let p = apply_homography(&h, *s);
            assert!(distance(p, *d) < 1e-9, "{p:?} vs {d:?}");
        }
        assert!((otsu(&[10.0, 12.0, 11.0, 200.0, 210.0]) - 106.0).abs() < 1e-9);
        assert!(approximate_quad(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], 0.1).is_none());
    }
}
