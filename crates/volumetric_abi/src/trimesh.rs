//! The general-purpose triangle mesh value type (declared as
//! `OperatorMetadataInput::TriMesh` / `OperatorMetadataOutput::TriMesh`).
//!
//! Like [`crate::fea::FeaMesh`], a `TriMesh` is explicit CBOR data, not a
//! sampler. It is deliberately just triangles: there is no manifold or
//! watertightness requirement, so an open scan, a soup with holes, or a
//! single free triangle are all valid values that render and process as the
//! mesh they are. Consumers that need a solid (e.g. conversion to a
//! sampleable model) define their own semantics for open input.
//!
//! Triangle winding is counter-clockwise viewed from the front/outside,
//! matching STL convention; consumers that only care about geometry may
//! ignore winding.

pub use crate::fea::FeaField as MeshField;

/// An explicit triangle mesh.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TriMesh {
    /// Vertex positions, xyz interleaved (`len == 3 * vertex_count`).
    pub positions: Vec<f64>,
    /// Vertex indices, 3 per triangle.
    pub indices: Vec<u32>,
    /// Named per-vertex data.
    pub vertex_fields: Vec<MeshField>,
    /// Named per-triangle data.
    pub face_fields: Vec<MeshField>,
}

impl TriMesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The vertex indices of triangle `idx`.
    pub fn triangle(&self, idx: usize) -> [u32; 3] {
        [
            self.indices[idx * 3],
            self.indices[idx * 3 + 1],
            self.indices[idx * 3 + 2],
        ]
    }

    /// The position of vertex `idx`.
    pub fn position(&self, idx: usize) -> [f64; 3] {
        [
            self.positions[idx * 3],
            self.positions[idx * 3 + 1],
            self.positions[idx * 3 + 2],
        ]
    }

    /// Axis-aligned bounds as `[min_x, max_x, min_y, max_y, min_z, max_z]`,
    /// or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<[f64; 6]> {
        if self.positions.is_empty() {
            return None;
        }
        let mut bounds = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for v in 0..self.vertex_count() {
            let p = self.position(v);
            for axis in 0..3 {
                bounds[axis * 2] = bounds[axis * 2].min(p[axis]);
                bounds[axis * 2 + 1] = bounds[axis * 2 + 1].max(p[axis]);
            }
        }
        Some(bounds)
    }

    /// Check the structural rules: positions come in xyz triples, indices in
    /// whole triangles referencing valid vertices, all coordinates finite,
    /// and every field has a non-empty unique name and exactly `components`
    /// values per vertex/triangle.
    pub fn validate(&self) -> Result<(), String> {
        if !self.positions.len().is_multiple_of(3) {
            return Err(format!(
                "positions length {} is not a multiple of 3",
                self.positions.len()
            ));
        }
        if !self.indices.len().is_multiple_of(3) {
            return Err(format!(
                "indices length {} is not a multiple of 3",
                self.indices.len()
            ));
        }
        if let Some(bad) = self.positions.iter().find(|v| !v.is_finite()) {
            return Err(format!("non-finite vertex coordinate {bad}"));
        }
        let vertex_count = self.vertex_count();
        if let Some(bad) = self
            .indices
            .iter()
            .find(|&&idx| idx as usize >= vertex_count)
        {
            return Err(format!(
                "triangle references vertex {bad} but the mesh has {vertex_count} vertices"
            ));
        }
        validate_fields("vertex", &self.vertex_fields, vertex_count)?;
        validate_fields("face", &self.face_fields, self.triangle_count())?;
        Ok(())
    }
}

fn validate_fields(kind: &str, fields: &[MeshField], entry_count: usize) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for field in fields {
        if field.name.is_empty() {
            return Err(format!("{kind} field with an empty name"));
        }
        if !seen.insert(field.name.as_str()) {
            return Err(format!("duplicate {kind} field name {:?}", field.name));
        }
        if field.components == 0 {
            return Err(format!("{kind} field {:?} has zero components", field.name));
        }
        let expected = entry_count * field.components;
        if field.data.len() != expected {
            return Err(format!(
                "{kind} field {:?} has {} values, expected {expected} ({entry_count} entries x {} components)",
                field.name,
                field.data.len(),
                field.components
            ));
        }
    }
    Ok(())
}

/// CBOR-encode a triangle mesh (the payload of a TriMesh operator output).
pub fn encode_tri_mesh(mesh: &TriMesh) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(mesh, &mut out)
        .expect("triangle mesh CBOR serialization should not fail");
    out
}

/// Decode and structurally validate a TriMesh payload.
pub fn decode_tri_mesh(bytes: &[u8]) -> Result<TriMesh, String> {
    let mesh: TriMesh = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode triangle mesh CBOR: {e}"))?;
    mesh.validate()?;
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_triangles() -> TriMesh {
        // An open corner: two triangles sharing an edge, NOT a closed
        // surface — deliberately valid.
        TriMesh {
            positions: vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            indices: vec![0, 1, 2, 0, 1, 3],
            vertex_fields: vec![],
            face_fields: vec![],
        }
    }

    #[test]
    fn round_trips_and_reports_shape() {
        let mut mesh = two_triangles();
        mesh.face_fields.push(MeshField {
            name: "tag".to_string(),
            components: 1,
            data: vec![1.0, 2.0],
        });
        let decoded = decode_tri_mesh(&encode_tri_mesh(&mesh)).unwrap();
        assert_eq!(decoded, mesh);
        assert_eq!(decoded.vertex_count(), 4);
        assert_eq!(decoded.triangle_count(), 2);
        assert_eq!(decoded.bounds(), Some([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]));
    }

    #[test]
    fn validation_rejects_malformed_meshes() {
        let mut mesh = two_triangles();
        mesh.indices[1] = 9;
        assert!(mesh.validate().is_err());

        let mut mesh = two_triangles();
        mesh.positions.pop();
        assert!(mesh.validate().is_err());

        let mut mesh = two_triangles();
        mesh.indices.pop();
        assert!(mesh.validate().is_err());

        let mut mesh = two_triangles();
        mesh.positions[0] = f64::NAN;
        assert!(mesh.validate().is_err());

        let mut mesh = two_triangles();
        mesh.vertex_fields.push(MeshField {
            name: "uv".to_string(),
            components: 2,
            data: vec![0.0; 7], // 4 vertices x 2 = 8 expected
        });
        assert!(mesh.validate().is_err());
    }

    #[test]
    fn empty_mesh_is_valid() {
        let mesh = TriMesh {
            positions: vec![],
            indices: vec![],
            vertex_fields: vec![],
            face_fields: vec![],
        };
        assert!(mesh.validate().is_ok());
        assert_eq!(mesh.bounds(), None);
    }
}
