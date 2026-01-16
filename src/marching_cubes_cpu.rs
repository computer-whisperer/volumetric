use anyhow::Result;

// Keep the old marching-cubes tables around for now (they are still useful for comparison,
// and may be re-used later). Surface Nets is used for robustness on boolean fields.
#[allow(unused_imports)]
use crate::marching_cubes_tables::{EDGE_TABLE, TRI_TABLE};
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
    // Returns density in [0,1]; current callers typically return 0.0/1.0
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
        triangles.push([a, c, b]);
        triangles.push([a, d, c]);
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

    // Ensure a consistent outward-facing winding.
    //
    // Surface Nets connectivity above is deterministic, but per-axis quad winding can still be
    // tricky to reason about. For STL export (and for downstream mesh processing), we want a
    // consistent rule: triangle normals should point from inside -> outside.
    //
    // We can enforce this by probing points offset from the triangle centroid along +/- its
    // geometric normal. If we can find a distance where one side is inside and the other is
    // outside, we can orient the triangle normal to point from inside -> outside.
    //
    // Some fields (thin features / high curvature / discretization) can yield cases where both
    // probes land on the same side for several distances. In those cases we fall back to a small
    // multi-sample vote along +/- normal to find which direction tends to go "into" the solid.
    let min_step = step_x.min(step_y).min(step_z);
    let eps = 0.1f32 * min_step;
    let bounds_diag = (
        (bounds_max.0 - bounds_min.0).powi(2)
            + (bounds_max.1 - bounds_min.1).powi(2)
            + (bounds_max.2 - bounds_min.2).powi(2),
    )
    .0
    .sqrt();
    if eps > 0.0 && bounds_diag > 0.0 {
        // Try small (local) distances first, then up to a conservative fraction of the bounds.
        // (Too large can cross unrelated parts of the surface in self-intersecting shapes.)
        let max_d = (8.0 * min_step).max(0.05 * bounds_diag);
        let ladder = [
            1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
        ];
        for tri in &mut triangles {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];

            let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
            let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
            let n = (
                ab.1 * ac.2 - ab.2 * ac.1,
                ab.2 * ac.0 - ab.0 * ac.2,
                ab.0 * ac.1 - ab.1 * ac.0,
            );
            let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
            if len2 <= f32::EPSILON {
                continue;
            }
            let inv_len = 1.0 / len2.sqrt();
            let nu = (n.0 * inv_len, n.1 * inv_len, n.2 * inv_len);

            let centroid = (
                (a.0 + b.0 + c.0) / 3.0,
                (a.1 + b.1 + c.1) / 3.0,
                (a.2 + b.2 + c.2) / 3.0,
            );

            let mut decided = false;
            let mut flip = false;

            // Primary: find a distance where the two sides differ.
            for m in ladder {
                let d = (eps * m).min(max_d);
                if d <= 0.0 {
                    continue;
                }

                let probe_plus = (
                    centroid.0 + nu.0 * d,
                    centroid.1 + nu.1 * d,
                    centroid.2 + nu.2 * d,
                );
                let probe_minus = (
                    centroid.0 - nu.0 * d,
                    centroid.1 - nu.1 * d,
                    centroid.2 - nu.2 * d,
                );

                let inside_plus = is_inside(probe_plus)? > 0.5;
                let inside_minus = is_inside(probe_minus)? > 0.5;
                if inside_plus != inside_minus {
                    decided = true;
                    // If moving along the current normal goes inside and the opposite direction
                    // goes outside, the normal is pointing inward and we must flip.
                    flip = inside_plus && !inside_minus;
                    break;
                }
            }

            // Fallback: vote along +/- normal to infer which direction tends to go inside.
            if !decided {
                let mut plus_inside_hits = 0u32;
                let mut minus_inside_hits = 0u32;
                // Sample a short segment (up to max_d) to avoid reaching unrelated parts.
                let steps = 8u32;
                for i in 1..=steps {
                    let t = i as f32 / steps as f32;
                    let d = (eps + t * (max_d - eps)).max(eps);
                    let probe_plus = (
                        centroid.0 + nu.0 * d,
                        centroid.1 + nu.1 * d,
                        centroid.2 + nu.2 * d,
                    );
                    let probe_minus = (
                        centroid.0 - nu.0 * d,
                        centroid.1 - nu.1 * d,
                        centroid.2 - nu.2 * d,
                    );
                    if is_inside(probe_plus)? > 0.5 {
                        plus_inside_hits += 1;
                    }
                    if is_inside(probe_minus)? > 0.5 {
                        minus_inside_hits += 1;
                    }
                }

                if plus_inside_hits != minus_inside_hits {
                    decided = true;
                    // More inside hits along +normal means +normal points inward.
                    flip = plus_inside_hits > minus_inside_hits;
                }
            }

            if decided && flip {
                tri.swap(1, 2);
            }
        }
    }

    Ok(triangles)
}

#[allow(dead_code)]
fn interpolate_vertex(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> (f32, f32, f32) {
    // Kept for reference/debugging against the old marching-cubes implementation.
    ((p1.0 + p2.0) * 0.5, (p1.1 + p2.1) * 0.5, (p1.2 + p2.2) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_unit_normal(tri: &Triangle) -> Option<(f32, f32, f32)> {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
        let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
        let n = (
            ab.1 * ac.2 - ab.2 * ac.1,
            ab.2 * ac.0 - ab.0 * ac.2,
            ab.0 * ac.1 - ab.1 * ac.0,
        );
        let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
        if len2 <= 1e-12 {
            return None;
        }
        let inv_len = 1.0 / len2.sqrt();
        Some((n.0 * inv_len, n.1 * inv_len, n.2 * inv_len))
    }

    fn triangle_centroid(tri: &Triangle) -> (f32, f32, f32) {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        ((a.0 + b.0 + c.0) / 3.0, (a.1 + b.1 + c.1) / 3.0, (a.2 + b.2 + c.2) / 3.0)
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
            Ok(
                p.0 >= bounds_min.0
                    && p.0 <= bounds_max.0
                    && p.1 >= bounds_min.1
                    && p.1 <= bounds_max.1
                    && p.2 >= bounds_min.2
                    && p.2 <= bounds_max.2,
            )
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
            for v in tri {
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
            Ok(r2 <= 0.75f32 * 0.75f32)
        })
        .unwrap();

        assert!(!tris.is_empty(), "expected sphere-ish field to produce triangles");

        // Sanity: vertices should not fly far away from the sampling domain.
        let step = (bounds_max.0 - bounds_min.0) / resolution as f32;
        let margin = step * 2.0;
        for tri in &tris {
            for v in tri {
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
            Ok(r2 <= 0.75f32 * 0.75f32)
        })
        .unwrap();
        assert!(!tris.is_empty());

        let mut ok = 0usize;
        let mut total = 0usize;
        for tri in &tris {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
            let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
            let n = (
                ab.1 * ac.2 - ab.2 * ac.1,
                ab.2 * ac.0 - ab.0 * ac.2,
                ab.0 * ac.1 - ab.1 * ac.0,
            );
            let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
            if len2 <= 1e-12 {
                continue;
            }
            let centroid = (
                (a.0 + b.0 + c.0) / 3.0,
                (a.1 + b.1 + c.1) / 3.0,
                (a.2 + b.2 + c.2) / 3.0,
            );
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
            Ok(qx * qx + qy * qy <= r_minor * r_minor)
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
}
