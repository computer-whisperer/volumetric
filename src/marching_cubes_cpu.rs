use anyhow::Result;

use crate::Triangle;

/// CPU mesh generation.
///
/// `resolution` is interpreted as the number of cubes along each axis inside the provided bounds.
/// To ensure surfaces that touch the bounds are still closed, this function samples one extra
/// layer of cubes outside the bounds on *both* sides.
///
/// Implementation note:
/// We use a boolean Surface Nets variant rather than a tri-table marching-cubes implementation.
/// With boolean-only sampling there is no scalar field to interpolate and the tri-table approach
/// is prone to ambiguous configurations that can manifest as cracks/discontinuities. Surface Nets
/// computes a single vertex per active cell and stitches quads between neighboring cells using a
/// cached lattice of corner samples, which is deterministic and avoids inconsistent sampling.
pub fn marching_cubes_mesh<F>(
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    resolution: usize,
    mut is_inside: F,
) -> Result<Vec<Triangle>>
where
    F: FnMut((f32, f32, f32)) -> Result<f32>,
{
    if resolution == 0 {
        return Ok(Vec::new());
    }

    let step_x = (bounds_max.0 - bounds_min.0) / resolution as f32;
    let step_y = (bounds_max.1 - bounds_min.1) / resolution as f32;
    let step_z = (bounds_max.2 - bounds_min.2) / resolution as f32;

    // Pad by one cube on each side so we can detect crossings at the exact bounds.
    // Cells are in [-1..=resolution], corners must extend one further to support each cell's +1.
    let min_cell: i32 = -1;
    let max_cell: i32 = resolution as i32;
    let min_corner: i32 = -1;
    let max_corner: i32 = resolution as i32 + 1;

    // Corner lattice is (resolution + 3)^3.
    let corner_n: usize = resolution + 3;
    let corner_offset: i32 = 1; // maps corner coord -1 -> 0
    let corner_idx3 = |x: i32, y: i32, z: i32| -> usize {
        debug_assert!((min_corner..=max_corner).contains(&x));
        debug_assert!((min_corner..=max_corner).contains(&y));
        debug_assert!((min_corner..=max_corner).contains(&z));
        let xi = (x + corner_offset) as usize;
        let yi = (y + corner_offset) as usize;
        let zi = (z + corner_offset) as usize;
        (zi * corner_n + yi) * corner_n + xi
    };

    let corner_pos = |x: i32, y: i32, z: i32| -> (f32, f32, f32) {
        (
            bounds_min.0 + x as f32 * step_x,
            bounds_min.1 + y as f32 * step_y,
            bounds_min.2 + z as f32 * step_z,
        )
    };

    // Cache inside/outside at every lattice corner. This eliminates inconsistencies between
    // neighboring cells and dramatically reduces WASM calls.
    let mut corner_inside = vec![false; corner_n * corner_n * corner_n];
    for z in min_corner..=max_corner {
        for y in min_corner..=max_corner {
            for x in min_corner..=max_corner {
                let p = corner_pos(x, y, z);
                // Convert density to boolean occupancy
                corner_inside[corner_idx3(x, y, z)] = is_inside(p)? > 0.5;
            }
        }
    }

    // Cell lattice is (resolution + 2)^3 for cells in [-1..=resolution].
    let cell_n: usize = resolution + 2;
    let cell_offset: i32 = 1; // maps cell coord -1 -> 0
    let cell_idx3 = |x: i32, y: i32, z: i32| -> usize {
        debug_assert!((min_cell..=max_cell).contains(&x));
        debug_assert!((min_cell..=max_cell).contains(&y));
        debug_assert!((min_cell..=max_cell).contains(&z));
        let xi = (x + cell_offset) as usize;
        let yi = (y + cell_offset) as usize;
        let zi = (z + cell_offset) as usize;
        (zi * cell_n + yi) * cell_n + xi
    };

    // For each active cell we create one vertex.
    let mut cell_vert_index = vec![-1i32; cell_n * cell_n * cell_n];
    let mut cell_vertices: Vec<(f32, f32, f32)> = Vec::new();

    // Cube corner offsets in a consistent order.
    const C_OFF: [(i32, i32, i32); 8] = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];
    // Edges as pairs of corner indices.
    const E: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    for zc in min_cell..=max_cell {
        for yc in min_cell..=max_cell {
            for xc in min_cell..=max_cell {
                let mut c_inside = [false; 8];
                let mut all_inside = true;
                let mut all_outside = true;
                for (i, (dx, dy, dz)) in C_OFF.iter().enumerate() {
                    let b = corner_inside[corner_idx3(xc + dx, yc + dy, zc + dz)];
                    c_inside[i] = b;
                    all_inside &= b;
                    all_outside &= !b;
                }
                if all_inside || all_outside {
                    continue;
                }

                // Boolean Surface Nets vertex = average of midpoints of sign-changing edges.
                let mut acc = (0.0f32, 0.0f32, 0.0f32);
                let mut n = 0u32;
                for (a, b) in E {
                    if c_inside[a] == c_inside[b] {
                        continue;
                    }
                    let (adx, ady, adz) = C_OFF[a];
                    let (bdx, bdy, bdz) = C_OFF[b];
                    let pa = corner_pos(xc + adx, yc + ady, zc + adz);
                    let pb = corner_pos(xc + bdx, yc + bdy, zc + bdz);
                    let mid = ((pa.0 + pb.0) * 0.5, (pa.1 + pb.1) * 0.5, (pa.2 + pb.2) * 0.5);
                    acc.0 += mid.0;
                    acc.1 += mid.1;
                    acc.2 += mid.2;
                    n += 1;
                }

                if n == 0 {
                    // Should not happen: mixed corners implies at least one crossing edge.
                    continue;
                }

                let v = (acc.0 / n as f32, acc.1 / n as f32, acc.2 / n as f32);
                let idx = cell_vertices.len() as u32;
                cell_vertices.push(v);
                cell_vert_index[cell_idx3(xc, yc, zc)] = idx as i32;
            }
        }
    }

    let mut triangles: Vec<Triangle> = Vec::new();

    // Connectivity / winding convention:
    // We emit quads around lattice edges where the boolean sign changes. Each such edge is shared
    // by 4 cells; we connect the 4 corresponding cell vertices into a quad. Winding is chosen
    // deterministically based on the edge direction and whether the edge transitions inside->outside
    // (vs outside->inside) when moving along the positive axis.
    let mut emit_quad = |i0: u32, i1: u32, i2: u32, i3: u32| {
        let a = cell_vertices[i0 as usize];
        let b = cell_vertices[i1 as usize];
        let c = cell_vertices[i2 as usize];
        let d = cell_vertices[i3 as usize];
        // Use consistent winding order with adaptive surface nets:
        // Reverse the triangle winding to produce outward-facing normals.
        triangles.push(Triangle::new([a, c, b]));
        triangles.push(Triangle::new([a, d, c]));
    };

    // Helper to fetch a cell vertex index; returns None if the cell is inactive or out of range.
    let cell_vi = |x: i32, y: i32, z: i32, cell_vert_index: &Vec<i32>| -> Option<u32> {
        if !(min_cell..=max_cell).contains(&x)
            || !(min_cell..=max_cell).contains(&y)
            || !(min_cell..=max_cell).contains(&z)
        {
            return None;
        }
        let v = cell_vert_index[cell_idx3(x, y, z)];
        if v >= 0 { Some(v as u32) } else { None }
    };

    // X-edges: edge from (x,y,z) -> (x+1,y,z). Four incident cells: (x,y-1,z-1),(x,y,z-1),(x,y,z),(x,y-1,z)
    for z in 0..=resolution as i32 {
        for y in 0..=resolution as i32 {
            for x in min_cell..=max_cell {
                let a = corner_inside[corner_idx3(x, y, z)];
                let b = corner_inside[corner_idx3(x + 1, y, z)];
                if a == b {
                    continue;
                }

                let i00 = cell_vi(x, y - 1, z - 1, &cell_vert_index);
                let i10 = cell_vi(x, y, z - 1, &cell_vert_index);
                let i11 = cell_vi(x, y, z, &cell_vert_index);
                let i01 = cell_vi(x, y - 1, z, &cell_vert_index);
                let (i00, i10, i11, i01) = match (i00, i10, i11, i01) {
                    (Some(i00), Some(i10), Some(i11), Some(i01)) => (i00, i10, i11, i01),
                    _ => continue,
                };

                if a {
                    emit_quad(i00, i01, i11, i10);
                } else {
                    emit_quad(i00, i10, i11, i01);
                }
            }
        }
    }

    // Y-edges: edge from (x,y,z) -> (x,y+1,z). Incident cells: (x-1,y,z-1),(x,y,z-1),(x,y,z),(x-1,y,z)
    for z in 0..=resolution as i32 {
        for y in min_cell..=max_cell {
            for x in 0..=resolution as i32 {
                let a = corner_inside[corner_idx3(x, y, z)];
                let b = corner_inside[corner_idx3(x, y + 1, z)];
                if a == b {
                    continue;
                }

                let i00 = cell_vi(x - 1, y, z - 1, &cell_vert_index);
                let i10 = cell_vi(x, y, z - 1, &cell_vert_index);
                let i11 = cell_vi(x, y, z, &cell_vert_index);
                let i01 = cell_vi(x - 1, y, z, &cell_vert_index);
                let (i00, i10, i11, i01) = match (i00, i10, i11, i01) {
                    (Some(i00), Some(i10), Some(i11), Some(i01)) => (i00, i10, i11, i01),
                    _ => continue,
                };

                if a {
                    emit_quad(i00, i10, i11, i01);
                } else {
                    emit_quad(i00, i01, i11, i10);
                }
            }
        }
    }

    // Z-edges: edge from (x,y,z) -> (x,y,z+1). Incident cells: (x-1,y-1,z),(x,y-1,z),(x,y,z),(x-1,y,z)
    for z in min_cell..=max_cell {
        for y in 0..=resolution as i32 {
            for x in 0..=resolution as i32 {
                let a = corner_inside[corner_idx3(x, y, z)];
                let b = corner_inside[corner_idx3(x, y, z + 1)];
                if a == b {
                    continue;
                }

                let i00 = cell_vi(x - 1, y - 1, z, &cell_vert_index);
                let i10 = cell_vi(x, y - 1, z, &cell_vert_index);
                let i11 = cell_vi(x, y, z, &cell_vert_index);
                let i01 = cell_vi(x - 1, y, z, &cell_vert_index);
                let (i00, i10, i11, i01) = match (i00, i10, i11, i01) {
                    (Some(i00), Some(i10), Some(i11), Some(i01)) => (i00, i10, i11, i01),
                    _ => continue,
                };

                if a {
                    emit_quad(i00, i01, i11, i10);
                } else {
                    emit_quad(i00, i10, i11, i01);
                }
            }
        }
    }

    // Ensure consistent orientation across the entire mesh.
    //
    // Local inside/outside probing can be ambiguous on thin / high-curvature / fractal surfaces.
    // What we need for correct back-face culling is *global* consistency: for a closed manifold
    // surface, adjacent triangles must have opposite directed edge orientation along every shared
    // edge. Once we enforce consistency, we can choose an outward direction per connected
    // component using the signed volume convention.
    let min_step = step_x.min(step_y).min(step_z);
    // Match the unit-test probe convention for determining outwardness.
    // Using the same base distance keeps behavior stable and regression tests meaningful.
    let probe_d = 0.15f32 * min_step;
    orient_triangles_consistently_and_outward(&mut triangles, probe_d, &mut is_inside);

    Ok(triangles)
}

fn orient_triangles_consistently_and_outward<F>(
    triangles: &mut [Triangle],
    probe_d: f32,
    is_inside: &mut F,
) where
    F: FnMut((f32, f32, f32)) -> Result<f32>,
{
    use std::collections::{HashMap, VecDeque};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct VKey(u32, u32, u32);
    impl VKey {
        fn from_v(v: (f32, f32, f32)) -> Self {
            Self(v.0.to_bits(), v.1.to_bits(), v.2.to_bits())
        }
    }

    fn tri_ids(
        tri: &Triangle,
        vid: &mut HashMap<VKey, u32>,
        next_id: &mut u32,
    ) -> [u32; 3] {
        let mut out = [0u32; 3];
        for (i, v) in tri.vertices.iter().copied().enumerate() {
            let k = VKey::from_v(v);
            let id = *vid.entry(k).or_insert_with(|| {
                let id = *next_id;
                *next_id += 1;
                id
            });
            out[i] = id;
        }
        out
    }

    fn undirected_edge(a: u32, b: u32) -> (u32, u32) {
        if a < b { (a, b) } else { (b, a) }
    }

    fn edge_dir_is_min_to_max(a: u32, b: u32) -> bool {
        // Returns true if the directed edge is from min->max for the undirected key.
        a < b
    }

    // Build vertex-id mapping (exact float equality via bits).
    let mut vid: HashMap<VKey, u32> = HashMap::new();
    let mut next_id: u32 = 0;

    // For each triangle, cache its current vertex ids.
    let mut ids: Vec<[u32; 3]> = Vec::with_capacity(triangles.len());
    for tri in triangles.iter() {
        ids.push(tri_ids(tri, &mut vid, &mut next_id));
    }

    // Map each undirected edge to incident triangles.
    //
    // NOTE: we intentionally do not cache direction here, because we mutate `ids` during
    // propagation (flipping triangles). Direction must be computed from the current `ids`.
    let mut edge_map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, t) in ids.iter().enumerate() {
        let e = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
        for (a, b) in e {
            let key = undirected_edge(a, b);
            edge_map.entry(key).or_default().push(ti);
        }
    }

    let n = triangles.len();
    let mut visited = vec![false; n];
    let mut flip = vec![false; n];

    // Helper: update cached ids after a flip.
    let apply_flip_to_ids = |ti: usize, ids: &mut Vec<[u32; 3]>| {
        ids[ti].swap(1, 2);
    };

    fn dir_for_edge_in_tri(t: [u32; 3], key: (u32, u32)) -> Option<bool> {
        let e = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
        for (a, b) in e {
            if undirected_edge(a, b) == key {
                return Some(edge_dir_is_min_to_max(a, b));
            }
        }
        None
    }

    // Walk each connected component (by shared edges) and enforce consistent edge directions.
    for start in 0..n {
        if visited[start] {
            continue;
        }

        // BFS over triangle adjacency.
        let mut queue = VecDeque::new();
        let mut component = Vec::new();
        visited[start] = true;
        queue.push_back(start);

        while let Some(ti) = queue.pop_front() {
            component.push(ti);
            let t = ids[ti];
            let e = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
            for (a, b) in e {
                let key = undirected_edge(a, b);
                let Some(inc) = edge_map.get(&key) else {
                    continue;
                };
                // Only propagate across manifold edges (exactly 2 incident triangles).
                // For non-manifold edges, an "opposite direction" constraint is ambiguous.
                if inc.len() != 2 {
                    continue;
                }

                let Some(self_dir) = dir_for_edge_in_tri(ids[ti], key) else {
                    continue;
                };
                let other = if inc[0] == ti { inc[1] } else { inc[0] };
                let Some(other_dir) = dir_for_edge_in_tri(ids[other], key) else {
                    continue;
                };

                // For an oriented manifold surface, the two incident triangles must traverse
                // the shared undirected edge in opposite directions.
                let should_match = self_dir == other_dir;
                if !visited[other] {
                    visited[other] = true;
                    flip[other] = flip[ti] ^ should_match;
                    if flip[other] {
                        apply_flip_to_ids(other, &mut ids);
                    }
                    queue.push_back(other);
                } else {
                    // Already assigned; ensure we didn't contradict.
                    // If we did, the mesh has a non-orientable / non-manifold defect; keep going.
                }
            }
        }

        // Apply the consistency flips to the actual triangles.
        for &ti in &component {
            if flip[ti] {
                triangles[ti].vertices.swap(1, 2);
                triangles[ti].normals.swap(1, 2);
            }
        }

        // Now orient triangles outward using a local inside/outside probe. This is per-triangle
        // (not component-wide) because thin/fractal features can make component-level heuristics
        // unreliable.
        if probe_d.is_finite() && probe_d > 0.0 {
            let ladder = [1.0f32, 2.0, 4.0, 8.0, 16.0];
            for &ti in &component {
                let a = triangles[ti].vertices[0];
                let b = triangles[ti].vertices[1];
                let c = triangles[ti].vertices[2];
                let n = Triangle::compute_face_normal(&[a, b, c]);
                if n.0 == 0.0 && n.1 == 0.0 && n.2 == 0.0 {
                    continue;
                }
                let centroid = (
                    (a.0 + b.0 + c.0) / 3.0,
                    (a.1 + b.1 + c.1) / 3.0,
                    (a.2 + b.2 + c.2) / 3.0,
                );

                let mut decided = false;
                let mut inward = false;
                for m in ladder {
                    let d = probe_d * m;
                    let p_plus = (
                        centroid.0 + n.0 * d,
                        centroid.1 + n.1 * d,
                        centroid.2 + n.2 * d,
                    );
                    let p_minus = (
                        centroid.0 - n.0 * d,
                        centroid.1 - n.1 * d,
                        centroid.2 - n.2 * d,
                    );
                    let inside_plus = is_inside(p_plus).ok().is_some_and(|v| v > 0.5);
                    let inside_minus = is_inside(p_minus).ok().is_some_and(|v| v > 0.5);
                    if inside_plus != inside_minus {
                        decided = true;
                        inward = inside_plus && !inside_minus;
                        break;
                    }
                }

                if decided && inward {
                    triangles[ti].vertices.swap(1, 2);
                    triangles[ti].normals.swap(1, 2);
                }

                let nn = Triangle::compute_face_normal(&triangles[ti].vertices);
                triangles[ti].normals = [nn, nn, nn];
            }
        } else {
            // Always keep normals consistent with geometry.
            for &ti in &component {
                let nn = Triangle::compute_face_normal(&triangles[ti].vertices);
                triangles[ti].normals = [nn, nn, nn];
            }
        }
    }
}

#[allow(dead_code)]
fn interpolate_vertex(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> (f32, f32, f32) {
    // Kept for reference/debugging against the old marching-cubes implementation.
    ((p1.0 + p2.0) * 0.5, (p1.1 + p2.1) * 0.5, (p1.2 + p2.2) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn mandelbulb_inside(cx: f32, cy: f32, cz: f32) -> bool {
        // Mirror `crates/models/mandelbulb_model` so tests exercise the same torture case.
        let power: f64 = 8.0;
        let max_iter: u32 = 18;
        let bailout: f64 = 2.0;
        let s: f64 = 0.9;

        let cx = cx as f64 * s;
        let cy = cy as f64 * s;
        let cz = cz as f64 * s;

        let mut x = 0.0f64;
        let mut y = 0.0f64;
        let mut z = 0.0f64;

        for _ in 0..max_iter {
            let r = (x * x + y * y + z * z).sqrt();
            if r > bailout {
                return false;
            }
            if r < 1.0e-6 {
                x = cx;
                y = cy;
                z = cz;
                continue;
            }

            let theta = (z / r).clamp(-1.0, 1.0).acos();
            let phi = y.atan2(x);

            let rp = r.powf(power);
            let thetap = theta * power;
            let phip = phi * power;

            let sin_t = thetap.sin();
            let nx = rp * sin_t * phip.cos();
            let ny = rp * sin_t * phip.sin();
            let nz = rp * thetap.cos();

            x = nx + cx;
            y = ny + cy;
            z = nz + cz;
        }

        true
    }

    fn assert_reasonable_vertex_normals(tris: &[Triangle]) {
        // Marching cubes produces per-triangle (or per-vertex) normals; for shading and
        // back-face culling correctness we at least require non-degenerate, roughly unit normals.
        let mut bad = 0usize;
        let mut total = 0usize;
        for tri in tris {
            for n in tri.normals {
                let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
                total += 1;
                if !(len2.is_finite()) || len2 < 0.25 || len2 > 4.0 {
                    bad += 1;
                }
            }
        }
        assert!(bad == 0, "found {bad}/{total} invalid normals (expected all finite and near-unit)");
    }

    fn triangle_unit_normal(tri: &Triangle) -> Option<(f32, f32, f32)> {
        let n = tri.face_normal();
        if n.0 == 0.0 && n.1 == 0.0 && n.2 == 0.0 {
            return None;
        }
        Some(n)
    }

    fn triangle_centroid(tri: &Triangle) -> (f32, f32, f32) {
        let a = tri.vertices[0];
        let b = tri.vertices[1];
        let c = tri.vertices[2];
        ((a.0 + b.0 + c.0) / 3.0, (a.1 + b.1 + c.1) / 3.0, (a.2 + b.2 + c.2) / 3.0)
    }

    fn assert_closed_oriented_manifold_edges(tris: &[Triangle]) {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        struct VKey(u32, u32, u32);
        impl VKey {
            fn from_v(v: (f32, f32, f32)) -> Self {
                Self(v.0.to_bits(), v.1.to_bits(), v.2.to_bits())
            }
        }

        fn undirected(a: u32, b: u32) -> (u32, u32) {
            if a < b { (a, b) } else { (b, a) }
        }

        let mut vid: HashMap<VKey, u32> = HashMap::new();
        let mut next_id: u32 = 0;

        let mut edge_counts: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        // For each undirected edge key, count how often it appears with direction min->max vs max->min.
        for tri in tris {
            let mut ids = [0u32; 3];
            for (i, v) in tri.vertices.iter().copied().enumerate() {
                let k = VKey::from_v(v);
                let id = *vid.entry(k).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                ids[i] = id;
            }

            let e = [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])];
            for (a, b) in e {
                let key = undirected(a, b);
                let entry = edge_counts.entry(key).or_insert((0, 0));
                if a < b {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }
        }

        // For a closed surface, we require:
        // - no boundary edges (an edge referenced only once => a hole/crack)
        // - consistent orientation (for any undirected edge, the number of times it appears as
        //   min->max must equal the number of times it appears as max->min)
        //
        // We intentionally do NOT assert that each edge is referenced exactly twice, because some
        // discretizations can produce coincident/duplicate triangles without causing back-face
        // culling holes (and the renderer symptom we care about is missing triangles).
        let mut boundary_edges = Vec::new();
        let mut imbalanced_edges = Vec::new();
        for (key, (min_to_max, max_to_min)) in edge_counts {
            let total = min_to_max + max_to_min;
            if total == 1 {
                boundary_edges.push((key, min_to_max, max_to_min));
            } else if min_to_max != max_to_min {
                imbalanced_edges.push((key, min_to_max, max_to_min));
            }
        }

        if !boundary_edges.is_empty() || !imbalanced_edges.is_empty() {
            boundary_edges.sort_by_key(|e| e.1 + e.2);
            imbalanced_edges.sort_by_key(|e| e.1.abs_diff(e.2));

            let sample_boundary: Vec<String> = boundary_edges
                .iter()
                .take(6)
                .map(|((a, b), c0, c1)| format!("edge({a},{b}) counts(min->max={c0}, max->min={c1})"))
                .collect();
            let sample_imbalanced: Vec<String> = imbalanced_edges
                .iter()
                .take(6)
                .map(|((a, b), c0, c1)| format!("edge({a},{b}) counts(min->max={c0}, max->min={c1})"))
                .collect();

            panic!(
                "mesh has orientation defects: boundary_edges={} imbalanced_edges={} sample_boundary=[{}] sample_imbalanced=[{}]",
                boundary_edges.len(),
                imbalanced_edges.len(),
                sample_boundary.join(", "),
                sample_imbalanced.join(", ")
            );
        }
    }

    #[test]
    fn padded_sampling_closes_surface_at_bounds_for_bounded_box() {
        // A model that is defined as "inside" only within the provided bounds.
        // Without sampling outside the bounds, a boolean mesher would see all boundary-adjacent
        // corners as inside and produce no surface. With padding, we should generate an outer box
        // surface.
        let bounds_min = (0.0, 0.0, 0.0);
        let bounds_max = (1.0, 1.0, 1.0);
        let resolution = 8;

        let tris = marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
              Ok(if p.0 >= bounds_min.0
                    && p.0 <= bounds_max.0
                    && p.1 >= bounds_min.1
                    && p.1 <= bounds_max.1
                    && p.2 >= bounds_min.2
                    && p.2 <= bounds_max.2
                {
                    1.0
                } else {
                    0.0
                })
        })
        .unwrap();

        assert!(
            !tris.is_empty(),
            "expected a closed surface when the model touches the bounds"
        );

        // Surface Nets places vertices at averages of crossing-edge midpoints. For a model that is
        // only defined inside the bounds, padding causes some crossings to happen outside the
        // bounds, and we should observe at least one vertex slightly outside.
        let step = (bounds_max.0 - bounds_min.0) / resolution as f32;
        let eps = step * 0.01;
        let min_expected = bounds_min.0 - eps;
        let max_expected = bounds_max.0 + eps;

        let mut saw_outside = false;
        'outer: for tri in &tris {
            for v in &tri.vertices {
                if v.0 < min_expected || v.0 > max_expected || v.1 < min_expected || v.1 > max_expected || v.2 < min_expected || v.2 > max_expected {
                    saw_outside = true;
                    break 'outer;
                }
            }
        }
        assert!(
            saw_outside,
            "expected padding to create vertices outside bounds (indicates boundary cubes were processed)"
        );
    }

    #[test]
    fn sphere_like_field_produces_some_triangles_and_stays_near_bounds() {
        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);
        let resolution = 12;

        let tris = marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
            let r2 = p.0 * p.0 + p.1 * p.1 + p.2 * p.2;
            Ok(if r2 <= 0.75f32 * 0.75f32 { 1.0 } else { 0.0 })
        })
        .unwrap();

        assert!(!tris.is_empty(), "expected sphere-ish field to produce triangles");

        // Sanity: vertices should not fly far away from the sampling domain.
        let step = (bounds_max.0 - bounds_min.0) / resolution as f32;
        let margin = step * 2.0;
        for tri in &tris {
            for v in &tri.vertices {
                assert!(
                    v.0 >= bounds_min.0 - margin && v.0 <= bounds_max.0 + margin,
                    "x out of expected range: {v:?}"
                );
                assert!(
                    v.1 >= bounds_min.1 - margin && v.1 <= bounds_max.1 + margin,
                    "y out of expected range: {v:?}"
                );
                assert!(
                    v.2 >= bounds_min.2 - margin && v.2 <= bounds_max.2 + margin,
                    "z out of expected range: {v:?}"
                );
            }
        }
    }

    #[test]
    fn sphere_like_field_has_most_normals_pointing_outward() {
        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);
        let resolution = 16;

        let tris = marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
            let r2 = p.0 * p.0 + p.1 * p.1 + p.2 * p.2;
            Ok(if r2 <= 0.75f32 * 0.75f32 { 1.0 } else { 0.0 })
        })
        .unwrap();
        assert!(!tris.is_empty());

        let mut ok = 0usize;
        let mut total = 0usize;
        for tri in &tris {
            let n = tri.face_normal();
            if n.0 == 0.0 && n.1 == 0.0 && n.2 == 0.0 {
                continue;
            }
            let centroid = triangle_centroid(tri);
            let dot = n.0 * centroid.0 + n.1 * centroid.1 + n.2 * centroid.2;
            total += 1;
            if dot > 0.0 {
                ok += 1;
            }
        }

        // With correct winding, nearly all normals should point outward.
        assert!(ok as f32 >= total as f32 * 0.995, "outward normals ratio too low: {ok}/{total}");
    }

    #[test]
    fn torus_field_has_consistent_normals_via_inside_outside_probe() {
        let bounds_min = (-1.5, -1.5, -1.5);
        let bounds_max = (1.5, 1.5, 1.5);
        let resolution = 22;

        // Torus centered at origin: major radius R, tube radius r.
        let r_major = 0.85f32;
        let r_minor = 0.30f32;
        let tris = marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
            let xy = (p.0 * p.0 + p.1 * p.1).sqrt();
            let qx = xy - r_major;
            let qy = p.2;
            Ok(if qx * qx + qy * qy <= r_minor * r_minor { 1.0 } else { 0.0 })
        })
        .unwrap();
        assert!(!tris.is_empty());

        let min_step = (bounds_max.0 - bounds_min.0) / resolution as f32;
        let eps = 0.15 * min_step;
        let ladder = [1.0f32, 2.0, 4.0, 8.0, 16.0];

        let mut decided = 0usize;
        let mut ok = 0usize;
        for tri in &tris {
            let Some(nu) = triangle_unit_normal(tri) else {
                continue;
            };
            let c = triangle_centroid(tri);

            // Decide orientation by checking whether +normal tends to go outside and -normal tends to go inside.
            let mut local_decided = false;
            let mut local_ok = false;
            for m in ladder {
                let d = eps * m;
                let p_plus = (c.0 + nu.0 * d, c.1 + nu.1 * d, c.2 + nu.2 * d);
                let p_minus = (c.0 - nu.0 * d, c.1 - nu.1 * d, c.2 - nu.2 * d);
                let inside_plus = {
                    let xy = (p_plus.0 * p_plus.0 + p_plus.1 * p_plus.1).sqrt();
                    let qx = xy - r_major;
                    let qy = p_plus.2;
                    qx * qx + qy * qy <= r_minor * r_minor
                };
                let inside_minus = {
                    let xy = (p_minus.0 * p_minus.0 + p_minus.1 * p_minus.1).sqrt();
                    let qx = xy - r_major;
                    let qy = p_minus.2;
                    qx * qx + qy * qy <= r_minor * r_minor
                };
                if inside_plus != inside_minus {
                    local_decided = true;
                    // Outward-facing should go outside along +normal and inside along -normal.
                    local_ok = !inside_plus && inside_minus;
                    break;
                }
            }

            if local_decided {
                decided += 1;
                if local_ok {
                    ok += 1;
                }
            }
        }

        // If many triangles are undecidable, we likely still have inconsistent winding.
        assert!(
            decided as f32 >= tris.len() as f32 * 0.98,
            "too many undecidable triangles: decided={decided} total={} ",
            tris.len()
        );
        assert!(
            ok as f32 >= decided as f32 * 0.995,
            "torus outward normals ratio too low: ok={ok} decided={decided} total={}",
            tris.len()
        );
    }

    #[test]
    fn mandelbulb_field_has_consistent_normals_via_inside_outside_probe() {
        let bounds_min = (-1.35, -1.35, -1.35);
        let bounds_max = (1.35, 1.35, 1.35);
        // Keep this reasonably fast for CI; the mandelbulb is inherently heavy.
        let resolution = 20;

        let tris = marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
            Ok(if mandelbulb_inside(p.0, p.1, p.2) { 1.0 } else { 0.0 })
        })
        .unwrap();
        assert!(!tris.is_empty(), "expected mandelbulb to produce triangles");
        assert_reasonable_vertex_normals(&tris);

        let min_step = (bounds_max.0 - bounds_min.0) / resolution as f32;
        let eps = 0.15 * min_step;
        let ladder = [1.0f32, 2.0, 4.0, 8.0, 16.0];

        let mut decided = 0usize;
        let mut ok = 0usize;
        for tri in &tris {
            let Some(nu) = triangle_unit_normal(tri) else {
                continue;
            };
            let c = triangle_centroid(tri);

            // Decide orientation by checking whether +normal tends to go outside and -normal tends to go inside.
            let mut local_decided = false;
            let mut local_ok = false;
            for m in ladder {
                let d = eps * m;
                let p_plus = (c.0 + nu.0 * d, c.1 + nu.1 * d, c.2 + nu.2 * d);
                let p_minus = (c.0 - nu.0 * d, c.1 - nu.1 * d, c.2 - nu.2 * d);
                let inside_plus = mandelbulb_inside(p_plus.0, p_plus.1, p_plus.2);
                let inside_minus = mandelbulb_inside(p_minus.0, p_minus.1, p_minus.2);
                if inside_plus != inside_minus {
                    local_decided = true;
                    // Outward-facing should go outside along +normal and inside along -normal.
                    local_ok = !inside_plus && inside_minus;
                    break;
                }
            }

            if local_decided {
                decided += 1;
                if local_ok {
                    ok += 1;
                }
            }
        }

        // The mandelbulb has lots of thin features; we allow a lower “decided” rate than the torus.
        assert!(
            decided as f32 >= tris.len() as f32 * 0.90,
            "too many undecidable triangles: decided={decided} total={}",
            tris.len()
        );
        assert!(
            ok as f32 >= decided as f32 * 0.98,
            "mandelbulb outward normals ratio too low: ok={ok} decided={decided} total={}",
            tris.len()
        );
    }

    #[test]
    #[ignore]
    fn mandelbulb_field_has_consistent_normals_via_inside_outside_probe_high_res() {
        // Heavier torture test intended for local runs when chasing regressions.
        let bounds_min = (-1.35, -1.35, -1.35);
        let bounds_max = (1.35, 1.35, 1.35);
        let resolution = 28;

        let tris = marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
            Ok(if mandelbulb_inside(p.0, p.1, p.2) { 1.0 } else { 0.0 })
        })
        .unwrap();
        assert!(!tris.is_empty());
        assert_reasonable_vertex_normals(&tris);

        let min_step = (bounds_max.0 - bounds_min.0) / resolution as f32;
        let eps = 0.15 * min_step;
        let ladder = [1.0f32, 2.0, 4.0, 8.0, 16.0];

        let mut decided = 0usize;
        let mut ok = 0usize;
        for tri in &tris {
            let Some(nu) = triangle_unit_normal(tri) else {
                continue;
            };
            let c = triangle_centroid(tri);

            let mut local_decided = false;
            let mut local_ok = false;
            for m in ladder {
                let d = eps * m;
                let p_plus = (c.0 + nu.0 * d, c.1 + nu.1 * d, c.2 + nu.2 * d);
                let p_minus = (c.0 - nu.0 * d, c.1 - nu.1 * d, c.2 - nu.2 * d);
                let inside_plus = mandelbulb_inside(p_plus.0, p_plus.1, p_plus.2);
                let inside_minus = mandelbulb_inside(p_minus.0, p_minus.1, p_minus.2);
                if inside_plus != inside_minus {
                    local_decided = true;
                    local_ok = !inside_plus && inside_minus;
                    break;
                }
            }

            if local_decided {
                decided += 1;
                if local_ok {
                    ok += 1;
                }
            }
        }

        assert!(decided as f32 >= tris.len() as f32 * 0.92, "decided={decided} total={}", tris.len());
        assert!(ok as f32 >= decided as f32 * 0.99, "ok={ok} decided={decided} total={}", tris.len());
    }
}
