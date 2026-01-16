use anyhow::Result;
use std::fs;
use std::path::Path;

use crate::Triangle;

pub fn triangles_to_binary_stl_bytes(triangles: &[Triangle], header_name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + triangles.len() * 50);

    let mut header = [0u8; 80];
    let name_bytes = header_name.as_bytes();
    let copy_n = name_bytes.len().min(header.len());
    header[..copy_n].copy_from_slice(&name_bytes[..copy_n]);
    out.extend_from_slice(&header);

    out.extend_from_slice(&(triangles.len() as u32).to_le_bytes());

    for tri in triangles {
        // Use the normal stored in the triangle
        let (nx, ny, nz) = tri.normal;

        out.extend_from_slice(&nx.to_le_bytes());
        out.extend_from_slice(&ny.to_le_bytes());
        out.extend_from_slice(&nz.to_le_bytes());

        for (x, y, z) in &tri.vertices {
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
            Triangle::new([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
            Triangle::new([(0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)]),
        ];
        let bytes = triangles_to_binary_stl_bytes(&tris, "test");

        assert_eq!(bytes.len(), 84 + 2 * 50);
        let tri_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap());
        assert_eq!(tri_count, 2);
    }

    #[test]
    fn normal_is_right_hand_rule() {
        let tri = Triangle::new([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]);
        let n = tri.normal;
        assert!((n.0 - 0.0).abs() < 1e-6);
        assert!((n.1 - 0.0).abs() < 1e-6);
        assert!((n.2 - 1.0).abs() < 1e-6);
    }
}
