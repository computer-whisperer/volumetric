use anyhow::Result;

use crate::marching_cubes_tables::{EDGE_TABLE, TRI_TABLE};
use crate::Triangle;

/// CPU marching cubes mesh generation.
///
/// `resolution` is interpreted as the number of cubes along each axis inside the provided bounds.
/// To ensure surfaces that touch the bounds are still closed, this function samples one extra
/// layer of cubes outside the bounds on *both* sides.
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

    let mut triangles = Vec::new();

    // Pad by one cube on each side so we can detect crossings at the exact bounds.
    let min_i: i32 = -1;
    let max_i: i32 = resolution as i32;

    for z_idx in min_i..=max_i {
        for y_idx in min_i..=max_i {
            for x_idx in min_i..=max_i {
                let x = bounds_min.0 + x_idx as f32 * step_x;
                let y = bounds_min.1 + y_idx as f32 * step_y;
                let z = bounds_min.2 + z_idx as f32 * step_z;

                let corners = [
                    (x, y, z),
                    (x + step_x, y, z),
                    (x + step_x, y + step_y, z),
                    (x, y + step_y, z),
                    (x, y, z + step_z),
                    (x + step_x, y, z + step_z),
                    (x + step_x, y + step_y, z + step_z),
                    (x, y + step_y, z + step_z),
                ];

                let mut cube_index = 0u8;
                for (i, &corner) in corners.iter().enumerate() {
                    if is_inside(corner)? {
                        cube_index |= 1 << i;
                    }
                }

                let edges = EDGE_TABLE[cube_index as usize];
                if edges == 0 {
                    continue;
                }

                let mut vert_list = [(0.0f32, 0.0f32, 0.0f32); 12];

                if edges & 1 != 0 {
                    vert_list[0] = interpolate_vertex(corners[0], corners[1]);
                }
                if edges & 2 != 0 {
                    vert_list[1] = interpolate_vertex(corners[1], corners[2]);
                }
                if edges & 4 != 0 {
                    vert_list[2] = interpolate_vertex(corners[2], corners[3]);
                }
                if edges & 8 != 0 {
                    vert_list[3] = interpolate_vertex(corners[3], corners[0]);
                }
                if edges & 16 != 0 {
                    vert_list[4] = interpolate_vertex(corners[4], corners[5]);
                }
                if edges & 32 != 0 {
                    vert_list[5] = interpolate_vertex(corners[5], corners[6]);
                }
                if edges & 64 != 0 {
                    vert_list[6] = interpolate_vertex(corners[6], corners[7]);
                }
                if edges & 128 != 0 {
                    vert_list[7] = interpolate_vertex(corners[7], corners[4]);
                }
                if edges & 256 != 0 {
                    vert_list[8] = interpolate_vertex(corners[0], corners[4]);
                }
                if edges & 512 != 0 {
                    vert_list[9] = interpolate_vertex(corners[1], corners[5]);
                }
                if edges & 1024 != 0 {
                    vert_list[10] = interpolate_vertex(corners[2], corners[6]);
                }
                if edges & 2048 != 0 {
                    vert_list[11] = interpolate_vertex(corners[3], corners[7]);
                }

                let tri_row = &TRI_TABLE[cube_index as usize];
                let mut i = 0;
                while i < 16 && tri_row[i] != -1 {
                    let v0 = vert_list[tri_row[i] as usize];
                    let v1 = vert_list[tri_row[i + 1] as usize];
                    let v2 = vert_list[tri_row[i + 2] as usize];
                    triangles.push([v0, v1, v2]);
                    i += 3;
                }
            }
        }
    }

    Ok(triangles)
}

fn interpolate_vertex(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> (f32, f32, f32) {
    ((p1.0 + p2.0) * 0.5, (p1.1 + p2.1) * 0.5, (p1.2 + p2.2) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_sampling_closes_surface_at_bounds_for_bounded_box() {
        // A model that is defined as "inside" only within the provided bounds.
        // Without sampling outside the bounds, marching cubes would see all corners as inside
        // and produce no surface. With padding, we should generate an outer box surface.
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

        // With midpoint interpolation, boundary-crossing vertices should be ~0.5*step outside
        // the bounds. We just want to detect *any* padding-related vertex outside by a small
        // epsilon.
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
}
