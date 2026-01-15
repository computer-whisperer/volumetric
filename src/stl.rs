use anyhow::Result;
use std::fs;
use std::path::Path;

use crate::Triangle;

fn triangle_normal(tri: &Triangle) -> (f32, f32, f32) {
    let (ax, ay, az) = tri[0];
    let (bx, by, bz) = tri[1];
    let (cx, cy, cz) = tri[2];

    let ab = (bx - ax, by - ay, bz - az);
    let ac = (cx - ax, cy - ay, cz - az);
    let n = (
        ab.1 * ac.2 - ab.2 * ac.1,
        ab.2 * ac.0 - ab.0 * ac.2,
        ab.0 * ac.1 - ab.1 * ac.0,
    );
    let len2 = n.0 * n.0 + n.1 * n.1 + n.2 * n.2;
    if len2 <= f32::EPSILON {
        return (0.0, 0.0, 0.0);
    }
    let inv_len = 1.0 / len2.sqrt();
    (n.0 * inv_len, n.1 * inv_len, n.2 * inv_len)
}

pub fn triangles_to_binary_stl_bytes(triangles: &[Triangle], header_name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + triangles.len() * 50);

    let mut header = [0u8; 80];
    let name_bytes = header_name.as_bytes();
    let copy_n = name_bytes.len().min(header.len());
    header[..copy_n].copy_from_slice(&name_bytes[..copy_n]);
    out.extend_from_slice(&header);

    out.extend_from_slice(&(triangles.len() as u32).to_le_bytes());

    for tri in triangles {
        let (nx, ny, nz) = triangle_normal(tri);

        out.extend_from_slice(&nx.to_le_bytes());
        out.extend_from_slice(&ny.to_le_bytes());
        out.extend_from_slice(&nz.to_le_bytes());

        for (x, y, z) in tri {
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            out.extend_from_slice(&z.to_le_bytes());
        }

        out.extend_from_slice(&0u16.to_le_bytes());
    }

    out
}

pub fn write_binary_stl(path: &Path, triangles: &[Triangle], header_name: &str) -> Result<()> {
    let bytes = triangles_to_binary_stl_bytes(triangles, header_name);
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_stl_has_expected_size_and_triangle_count() {
        let tris: Vec<Triangle> = vec![
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)],
        ];
        let bytes = triangles_to_binary_stl_bytes(&tris, "test");

        assert_eq!(bytes.len(), 84 + 2 * 50);
        let tri_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap());
        assert_eq!(tri_count, 2);
    }

    #[test]
    fn normal_is_right_hand_rule() {
        let tri: Triangle = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let n = triangle_normal(&tri);
        assert!((n.0 - 0.0).abs() < 1e-6);
        assert!((n.1 - 0.0).abs() < 1e-6);
        assert!((n.2 - 1.0).abs() < 1e-6);
    }
}
