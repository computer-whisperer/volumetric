//! 3MF export of host-side triangle soups — the counterpart of [`crate::stl`].
//! The package reader and writer themselves live in `threemf_core`
//! (re-exported here), shared with the import operator.

use anyhow::Result;
use std::path::Path;

pub use threemf_core::*;

use crate::Triangle;

/// Encodes a triangle soup as a 3MF package labelled in `unit`. Bit-identical
/// corners weld into shared vertices ([`TriMesh::from_soup`]); the soup's
/// coordinates are written as they are, so scale them to `unit` first.
pub fn triangles_to_3mf_bytes(
    triangles: &[Triangle],
    unit: Unit,
    title: &str,
) -> Result<Vec<u8>, String> {
    let mesh = TriMesh::from_soup(triangles.iter().map(|tri| {
        tri.vertices
            .map(|(x, y, z)| [f64::from(x), f64::from(y), f64::from(z)])
    }));
    write_3mf(&mesh, unit, title)
}

pub fn write_3mf_file(path: &Path, triangles: &[Triangle], unit: Unit, title: &str) -> Result<()> {
    let bytes = triangles_to_3mf_bytes(triangles, unit, title).map_err(anyhow::Error::msg)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soup_welds_and_round_trips() {
        let tris = [
            Triangle::new([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
            Triangle::new([(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
        ];
        let bytes = triangles_to_3mf_bytes(&tris, Unit::Millimeter, "corner").unwrap();
        let file = read_3mf(&bytes).unwrap();
        assert_eq!(file.unit, Unit::Millimeter);
        assert_eq!(file.items[0].vertex_count(), 4);
        assert_eq!(file.items[0].triangle_count(), 2);
        assert!(triangles_to_3mf_bytes(&[], Unit::Meter, "empty").is_err());
    }
}
