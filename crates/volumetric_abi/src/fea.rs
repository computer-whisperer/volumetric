//! The FEA mesh value type: explicit finite-element data flowing between
//! operators (declared as `OperatorMetadataInput::FeaMesh` /
//! `OperatorMetadataOutput::FeaMesh`).
//!
//! Unlike a `ModelWASM` value, an FEA mesh is *data*, not a sampler: node
//! positions, element connectivity, and named per-node / per-element
//! attribute arrays, CBOR-encoded with [`encode_fea_mesh`]. Hosts never hand
//! it to the model executor.
//!
//! # Hex8 node ordering
//!
//! [`FeaElementKind::Hex8`] uses the VTK hexahedron convention. For an
//! axis-aligned unit cell the eight nodes sit at:
//!
//! ```text
//! 0: (0,0,0)  1: (1,0,0)  2: (1,1,0)  3: (0,1,0)   (z = 0 face)
//! 4: (0,0,1)  5: (1,0,1)  6: (1,1,1)  7: (0,1,1)   (z = 1 face)
//! ```
//!
//! [`HEX8_FACES`] lists the six faces as quads wound counter-clockwise when
//! viewed from outside the element, so face normals computed from the
//! winding point outward.

/// The element type of every element in a mesh (meshes are homogeneous).
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum FeaElementKind {
    /// 8-node linear hexahedron (VTK ordering, see module docs).
    Hex8,
}

impl FeaElementKind {
    /// Nodes per element of this kind.
    pub fn node_count(self) -> usize {
        match self {
            FeaElementKind::Hex8 => 8,
        }
    }
}

/// The six faces of a Hex8, as node offsets into an element's connectivity,
/// wound counter-clockwise viewed from outside (normals point outward).
/// Order: -z, +z, -y, +y, -x, +x.
pub const HEX8_FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [3, 7, 6, 2],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
];

/// A named data array attached to nodes or elements.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FeaField {
    pub name: String,
    /// Values per entry: 1 for a scalar field, 3 for a vector field, etc.
    pub components: usize,
    /// `components` values per node/element, entry-major
    /// (`data[entry * components + c]`).
    pub data: Vec<f64>,
}

/// An explicit finite-element mesh.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FeaMesh {
    pub element_kind: FeaElementKind,
    /// Node positions, xyz interleaved (`len == 3 * node_count`).
    pub node_positions: Vec<f64>,
    /// Node indices, `element_kind.node_count()` per element.
    pub connectivity: Vec<u32>,
    /// Named per-node data (e.g. a displacement vector field).
    pub node_fields: Vec<FeaField>,
    /// Named per-element data (e.g. assigned density).
    pub element_fields: Vec<FeaField>,
}

impl FeaMesh {
    pub fn node_count(&self) -> usize {
        self.node_positions.len() / 3
    }

    pub fn element_count(&self) -> usize {
        self.connectivity.len() / self.element_kind.node_count()
    }

    /// The node indices of element `idx`.
    pub fn element(&self, idx: usize) -> &[u32] {
        let n = self.element_kind.node_count();
        &self.connectivity[idx * n..(idx + 1) * n]
    }

    /// The position of node `idx`.
    pub fn node_position(&self, idx: usize) -> [f64; 3] {
        [
            self.node_positions[idx * 3],
            self.node_positions[idx * 3 + 1],
            self.node_positions[idx * 3 + 2],
        ]
    }

    /// Check the structural rules: positions come in xyz triples,
    /// connectivity is whole elements referencing valid nodes, and every
    /// field has a non-empty unique name and exactly `components` values per
    /// node/element.
    pub fn validate(&self) -> Result<(), String> {
        if !self.node_positions.len().is_multiple_of(3) {
            return Err(format!(
                "node_positions length {} is not a multiple of 3",
                self.node_positions.len()
            ));
        }
        let per_element = self.element_kind.node_count();
        if !self.connectivity.len().is_multiple_of(per_element) {
            return Err(format!(
                "connectivity length {} is not a multiple of {per_element} ({:?})",
                self.connectivity.len(),
                self.element_kind
            ));
        }
        let node_count = self.node_count();
        if let Some(bad) = self
            .connectivity
            .iter()
            .find(|&&idx| idx as usize >= node_count)
        {
            return Err(format!(
                "connectivity references node {bad} but the mesh has {node_count} nodes"
            ));
        }

        validate_fields("node", &self.node_fields, node_count)?;
        validate_fields("element", &self.element_fields, self.element_count())?;
        Ok(())
    }

    /// The mesh's boundary faces: quads belonging to exactly one element,
    /// in outward-wound node indices ([`HEX8_FACES`] winding), paired with
    /// the owning element's index (for element-field display).
    ///
    /// Interior faces (shared by two elements) cancel; a valid conforming
    /// mesh never has a face used more than twice, but if one appears (a
    /// malformed mesh) it is treated as interior and dropped.
    pub fn boundary_faces(&self) -> Vec<(u32, [u32; 4])> {
        let mut faces: std::collections::HashMap<[u32; 4], (u32, u32, [u32; 4])> =
            std::collections::HashMap::new();
        for e in 0..self.element_count() {
            let element = self.element(e);
            for face in &HEX8_FACES {
                let quad = [
                    element[face[0]],
                    element[face[1]],
                    element[face[2]],
                    element[face[3]],
                ];
                let mut key = quad;
                key.sort_unstable();
                let entry = faces.entry(key).or_insert((0, e as u32, quad));
                entry.0 += 1;
            }
        }
        let mut out: Vec<(u32, [u32; 4])> = faces
            .into_values()
            .filter_map(|(count, element, quad)| (count == 1).then_some((element, quad)))
            .collect();
        // HashMap iteration order is nondeterministic; keep output stable.
        out.sort_unstable();
        out
    }

    /// [`Self::boundary_faces`] without the element association.
    pub fn boundary_quads(&self) -> Vec<[u32; 4]> {
        self.boundary_faces()
            .into_iter()
            .map(|(_, quad)| quad)
            .collect()
    }
}

fn validate_fields(kind: &str, fields: &[FeaField], entry_count: usize) -> Result<(), String> {
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

/// CBOR-encode an FEA mesh (the payload of a FeaMesh operator output).
pub fn encode_fea_mesh(mesh: &FeaMesh) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(mesh, &mut out)
        .expect("FEA mesh CBOR serialization should not fail");
    out
}

/// Decode and structurally validate a FeaMesh payload.
pub fn decode_fea_mesh(bytes: &[u8]) -> Result<FeaMesh, String> {
    let mesh: FeaMesh = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode FEA mesh CBOR: {e}"))?;
    mesh.validate()?;
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two unit hexes sharing the x = 1 face, spanning [0,2]x[0,1]x[0,1].
    fn two_hex_mesh() -> FeaMesh {
        // 12 nodes on a 3x2x2 lattice, index = x * 4 + y * 2 + z.
        let mut node_positions = Vec::new();
        for x in 0..3 {
            for y in 0..2 {
                for z in 0..2 {
                    node_positions.extend([x as f64, y as f64, z as f64]);
                }
            }
        }
        let hex_at = |x0: u32| {
            let n = |dx: u32, dy: u32, dz: u32| (x0 + dx) * 4 + dy * 2 + dz;
            [
                n(0, 0, 0),
                n(1, 0, 0),
                n(1, 1, 0),
                n(0, 1, 0),
                n(0, 0, 1),
                n(1, 0, 1),
                n(1, 1, 1),
                n(0, 1, 1),
            ]
        };
        FeaMesh {
            element_kind: FeaElementKind::Hex8,
            node_positions,
            connectivity: [hex_at(0), hex_at(1)].concat(),
            node_fields: vec![],
            element_fields: vec![],
        }
    }

    #[test]
    fn round_trips_and_validates() {
        let mut mesh = two_hex_mesh();
        mesh.node_fields.push(FeaField {
            name: "displacement".to_string(),
            components: 3,
            data: vec![0.0; 12 * 3],
        });
        mesh.element_fields.push(FeaField {
            name: "density".to_string(),
            components: 1,
            data: vec![1.0, 0.5],
        });
        let decoded = decode_fea_mesh(&encode_fea_mesh(&mesh)).unwrap();
        assert_eq!(decoded, mesh);
        assert_eq!(decoded.node_count(), 12);
        assert_eq!(decoded.element_count(), 2);
    }

    #[test]
    fn validation_rejects_malformed_meshes() {
        let mut mesh = two_hex_mesh();
        mesh.connectivity[3] = 99; // out-of-range node
        assert!(mesh.validate().is_err());

        let mut mesh = two_hex_mesh();
        mesh.node_positions.pop(); // not a multiple of 3
        assert!(mesh.validate().is_err());

        let mut mesh = two_hex_mesh();
        mesh.connectivity.pop(); // partial element
        assert!(mesh.validate().is_err());

        let mut mesh = two_hex_mesh();
        mesh.element_fields.push(FeaField {
            name: "density".to_string(),
            components: 1,
            data: vec![1.0], // 2 elements, 1 value
        });
        assert!(mesh.validate().is_err());

        let mut mesh = two_hex_mesh();
        mesh.node_fields.push(FeaField {
            name: "a".to_string(),
            components: 1,
            data: vec![0.0; 12],
        });
        mesh.node_fields.push(FeaField {
            name: "a".to_string(), // duplicate name
            components: 1,
            data: vec![0.0; 12],
        });
        assert!(mesh.validate().is_err());
    }

    #[test]
    fn hex8_face_windings_point_outward() {
        // Single unit hex at the origin, nodes in VTK ordering.
        let positions: [[f64; 3]; 8] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let center = [0.5, 0.5, 0.5];
        for face in &HEX8_FACES {
            let [a, b, c] = [positions[face[0]], positions[face[1]], positions[face[2]]];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
            let normal = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let outward = [a[0] - center[0], a[1] - center[1], a[2] - center[2]];
            let dot: f64 = (0..3).map(|i| normal[i] * outward[i]).sum();
            assert!(dot > 0.0, "face {face:?} winds inward");
        }
    }

    #[test]
    fn boundary_quads_drop_the_shared_face() {
        let mesh = two_hex_mesh();
        let quads = mesh.boundary_quads();
        // Two hexes = 12 faces, 2 coincide on the shared x=1 plane.
        assert_eq!(quads.len(), 10);
        // The shared face's nodes (x index 1) are 4..8; no boundary quad
        // consists solely of them.
        for quad in &quads {
            assert!(
                !quad.iter().all(|&n| (4..8).contains(&n)),
                "shared interior face {quad:?} leaked into the boundary"
            );
        }
    }

    #[test]
    fn boundary_faces_know_their_element() {
        let mesh = two_hex_mesh();
        let faces = mesh.boundary_faces();
        assert_eq!(faces.len(), 10);
        // Element 0 owns nodes 0..8, element 1 owns 4..12; every face's
        // nodes must belong to its owning element.
        for (element, quad) in &faces {
            let owner = mesh.element(*element as usize);
            for node in quad {
                assert!(
                    owner.contains(node),
                    "face {quad:?} attributed to element {element} which lacks node {node}"
                );
            }
        }
        // Both elements contribute five exposed faces each.
        assert_eq!(faces.iter().filter(|(e, _)| *e == 0).count(), 5);
        assert_eq!(faces.iter().filter(|(e, _)| *e == 1).count(), 5);
    }
}
