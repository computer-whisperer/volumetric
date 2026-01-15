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
    F: FnMut((f32, f32, f32)) -> Result<bool>,
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
                corner_inside[corner_idx3(x, y, z)] = is_inside(p)?;
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
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
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

                // If a is inside and b is outside, emit one winding; otherwise flip.
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
}
