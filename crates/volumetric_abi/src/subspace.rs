//! The affine-subspace value type (declared as
//! `OperatorMetadataInput::Subspace` / `OperatorMetadataOutput::Subspace`).
//!
//! A [`Subspace`] is a k-dimensional affine subspace of n-space — k = 0 a
//! point, 1 a line, 2 a plane, n-1 a hyperplane, n a full rigid frame —
//! carried together with an *orthonormal chart*: an origin point and k
//! orthonormal basis vectors. Like [`crate::trimesh::TriMesh`], it is
//! explicit CBOR data, not a sampler; hosts must never feed it to the
//! model executor.
//!
//! The chart is part of the value: two subspaces describing the same
//! point set with different origins or rotated bases are *different*
//! values. Consumers rely on the parameterization (e.g. laying a scan
//! lattice over a plane), and orthonormality makes it an isometry — chart
//! distances are world distances.
//!
//! Orientation is carried by basis order. For a hyperplane (k = n-1) the
//! oriented normal is the unique unit vector completing the basis to a
//! positively-oriented frame (see [`Subspace::normal`]); flipping a
//! plane means negating (or swapping) basis vectors.

/// Tolerance on basis dot products: rows must be unit length and pairwise
/// orthogonal within this bound.
pub const ORTHONORMALITY_TOLERANCE: f64 = 1e-9;

/// A k-dimensional affine subspace of n-space with an orthonormal chart.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Subspace {
    /// Ambient dimension n (>= 1).
    pub dimensions: u32,
    /// A point on the subspace, n values — the chart origin.
    pub origin: Vec<f64>,
    /// k orthonormal basis vectors, row-major k * n (k <= n; k = 0 for a
    /// point).
    pub basis: Vec<f64>,
}

/// Per-axis selector for [`Subspace::from_bounds`]: how the subspace
/// relates to a bounding box along one axis. Serialized lowercase so it
/// can double as an operator-config enum (`"min"` / `"max"` / `"center"`
/// / `"span"`).
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BoundSelector {
    /// Fix the axis at the box's minimum bound.
    Min,
    /// Fix the axis at the box's maximum bound.
    Max,
    /// Fix the axis at the box's center.
    Center,
    /// The subspace extends along this axis (contributes a basis vector).
    Span,
}

impl Subspace {
    /// Ambient dimension as a usize.
    pub fn ambient(&self) -> usize {
        self.dimensions as usize
    }

    /// Subspace dimension k (0 = point, 1 = line, 2 = plane, ...).
    pub fn rank(&self) -> usize {
        if self.dimensions == 0 {
            return 0;
        }
        self.basis.len() / self.ambient()
    }

    /// Basis vector `i` as a slice of n values.
    pub fn basis_vector(&self, i: usize) -> &[f64] {
        let n = self.ambient();
        &self.basis[i * n..(i + 1) * n]
    }

    /// A 0-dimensional subspace: just a point.
    pub fn point(origin: Vec<f64>) -> Self {
        Self {
            dimensions: origin.len() as u32,
            origin,
            basis: Vec::new(),
        }
    }

    /// An axis-aligned subspace: `origin` plus one unit basis vector per
    /// entry of `axes`, in the given order.
    pub fn axis_aligned(origin: Vec<f64>, axes: &[usize]) -> Result<Self, String> {
        let n = origin.len();
        let mut basis = vec![0.0; axes.len() * n];
        for (row, &axis) in axes.iter().enumerate() {
            if axis >= n {
                return Err(format!("axis {axis} out of range for {n}-space"));
            }
            basis[row * n + axis] = 1.0;
        }
        let subspace = Self {
            dimensions: n as u32,
            origin,
            basis,
        };
        subspace.validate()?;
        Ok(subspace)
    }

    /// The subspace touching a feature of an axis-aligned bounding box:
    /// one selector per axis, `bounds` interleaved `[min_0, max_0, ...]`.
    /// `Span` axes become basis vectors (ascending axis order) with the
    /// origin centered along them; the others fix the origin coordinate
    /// at the chosen bound. All `Span` = the box's centered full frame;
    /// none = a corner/center point.
    ///
    /// E.g. in 3-space, `[Span, Span, Min]` is the bottom face's plane
    /// and `[Min, Max, Span]` the vertical edge line at x-min/y-max.
    pub fn from_bounds(bounds: &[f64], selectors: &[BoundSelector]) -> Result<Self, String> {
        let n = selectors.len();
        if bounds.len() != 2 * n {
            return Err(format!(
                "bounds hold {} values, expected {} for {n} selectors",
                bounds.len(),
                2 * n
            ));
        }
        let origin = selectors
            .iter()
            .enumerate()
            .map(|(axis, sel)| {
                let (lo, hi) = (bounds[2 * axis], bounds[2 * axis + 1]);
                match sel {
                    BoundSelector::Min => lo,
                    BoundSelector::Max => hi,
                    BoundSelector::Center | BoundSelector::Span => (lo + hi) * 0.5,
                }
            })
            .collect();
        let axes: Vec<usize> = selectors
            .iter()
            .enumerate()
            .filter(|(_, sel)| **sel == BoundSelector::Span)
            .map(|(axis, _)| axis)
            .collect();
        Self::axis_aligned(origin, &axes)
    }

    /// Check the structural rules: sane dimensions, matching lengths, all
    /// values finite, and basis rows orthonormal within
    /// [`ORTHONORMALITY_TOLERANCE`].
    pub fn validate(&self) -> Result<(), String> {
        if self.dimensions == 0 {
            return Err("subspace ambient dimension must be >= 1".to_string());
        }
        let n = self.ambient();
        if self.origin.len() != n {
            return Err(format!(
                "origin holds {} values, expected {n}",
                self.origin.len()
            ));
        }
        if !self.basis.len().is_multiple_of(n) {
            return Err(format!(
                "basis length {} is not a multiple of the ambient dimension {n}",
                self.basis.len()
            ));
        }
        let k = self.rank();
        if k > n {
            return Err(format!("rank {k} exceeds ambient dimension {n}"));
        }
        if let Some(bad) = self
            .origin
            .iter()
            .chain(self.basis.iter())
            .find(|v| !v.is_finite())
        {
            return Err(format!("non-finite subspace value {bad}"));
        }
        for i in 0..k {
            for j in i..k {
                let dot: f64 = self
                    .basis_vector(i)
                    .iter()
                    .zip(self.basis_vector(j))
                    .map(|(a, b)| a * b)
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > ORTHONORMALITY_TOLERANCE {
                    return Err(format!(
                        "basis is not orthonormal: <b{i}, b{j}> = {dot}, expected {expected}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Map chart coordinates (k values) to a world point (n values):
    /// `origin + sum(chart[i] * basis[i])`.
    pub fn embed(&self, chart: &[f64]) -> Vec<f64> {
        debug_assert_eq!(chart.len(), self.rank());
        let mut point = self.origin.clone();
        for (i, &c) in chart.iter().enumerate() {
            for (p, b) in point.iter_mut().zip(self.basis_vector(i)) {
                *p += c * b;
            }
        }
        point
    }

    /// Orthogonally project a world point (n values) onto the subspace:
    /// its chart coordinates (k values) and its distance off the
    /// subspace.
    pub fn project(&self, point: &[f64]) -> (Vec<f64>, f64) {
        debug_assert_eq!(point.len(), self.ambient());
        let rel: Vec<f64> = point.iter().zip(&self.origin).map(|(p, o)| p - o).collect();
        let chart: Vec<f64> = (0..self.rank())
            .map(|i| {
                self.basis_vector(i)
                    .iter()
                    .zip(&rel)
                    .map(|(b, r)| b * r)
                    .sum()
            })
            .collect();
        let distance_sq: f64 = rel
            .iter()
            .enumerate()
            .map(|(axis, r)| {
                let in_plane: f64 = chart
                    .iter()
                    .enumerate()
                    .map(|(i, c)| c * self.basis_vector(i)[axis])
                    .sum();
                let out = r - in_plane;
                out * out
            })
            .sum();
        (chart, distance_sq.sqrt())
    }

    /// The oriented unit normal of a hyperplane (k = n-1): the unique
    /// unit vector completing the basis to a positively-oriented frame.
    /// `None` for any other rank.
    pub fn normal(&self) -> Option<Vec<f64>> {
        let n = self.ambient();
        let k = self.rank();
        if n == 0 || k + 1 != n {
            return None;
        }
        // Cofactor expansion along the missing last row of [basis; x]:
        // normal_j is the last row's cofactor for column j, which makes
        // det([basis; normal]) = |normal|^2 > 0 — positively oriented by
        // construction.
        let normal: Vec<f64> = (0..n)
            .map(|j| {
                let minor: Vec<f64> = (0..k)
                    .flat_map(|row| {
                        (0..n)
                            .filter(|&col| col != j)
                            .map(move |col| self.basis[row * n + col])
                    })
                    .collect();
                let sign = if (n - 1 + j) % 2 == 0 { 1.0 } else { -1.0 };
                sign * determinant(minor, k)
            })
            .collect();
        // Orthonormal bases give |normal| = 1 up to rounding; renormalize
        // to hand consumers an exact unit vector.
        let len = normal.iter().map(|v| v * v).sum::<f64>().sqrt();
        Some(normal.iter().map(|v| v / len).collect())
    }
}

/// Determinant of a k x k matrix (row-major, consumed) by Gaussian
/// elimination with partial pivoting. k = 0 gives 1.0 (the empty
/// product), so a line in 2-space gets its normal from bare cofactors.
fn determinant(mut m: Vec<f64>, k: usize) -> f64 {
    let mut det = 1.0;
    for col in 0..k {
        let pivot = (col..k)
            .max_by(|&a, &b| m[a * k + col].abs().total_cmp(&m[b * k + col].abs()))
            .unwrap();
        if m[pivot * k + col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            for j in 0..k {
                m.swap(pivot * k + j, col * k + j);
            }
            det = -det;
        }
        det *= m[col * k + col];
        for row in col + 1..k {
            let factor = m[row * k + col] / m[col * k + col];
            for j in col..k {
                m[row * k + j] -= factor * m[col * k + j];
            }
        }
    }
    det
}

/// CBOR-encode a subspace (the payload of a Subspace operator output).
pub fn encode_subspace(subspace: &Subspace) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(subspace, &mut out)
        .expect("subspace CBOR serialization should not fail");
    out
}

/// Decode and structurally validate a Subspace payload.
pub fn decode_subspace(bytes: &[u8]) -> Result<Subspace, String> {
    let subspace: Subspace = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode subspace CBOR: {e}"))?;
    subspace.validate()?;
    Ok(subspace)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_xy_at(z: f64) -> Subspace {
        Subspace::axis_aligned(vec![0.0, 0.0, z], &[0, 1]).unwrap()
    }

    #[test]
    fn validate_accepts_points_lines_planes_and_frames() {
        assert!(Subspace::point(vec![1.0, 2.0]).validate().is_ok());
        assert!(Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[2]).is_ok());
        assert!(plane_xy_at(1.0).validate().is_ok());
        assert!(Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1, 2]).is_ok());
        // A rotated but orthonormal plane basis.
        let s = 0.5f64.sqrt();
        let tilted = Subspace {
            dimensions: 3,
            origin: vec![0.0; 3],
            basis: vec![s, s, 0.0, -s, s, 0.0],
        };
        assert!(tilted.validate().is_ok());
    }

    #[test]
    fn validate_rejects_malformed_values() {
        let not_unit = Subspace {
            dimensions: 3,
            origin: vec![0.0; 3],
            basis: vec![2.0, 0.0, 0.0],
        };
        assert!(not_unit.validate().is_err());
        let not_orthogonal = Subspace {
            dimensions: 3,
            origin: vec![0.0; 3],
            basis: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };
        assert!(not_orthogonal.validate().is_err());
        let wrong_origin = Subspace {
            dimensions: 3,
            origin: vec![0.0; 2],
            basis: Vec::new(),
        };
        assert!(wrong_origin.validate().is_err());
        let ragged_basis = Subspace {
            dimensions: 3,
            origin: vec![0.0; 3],
            basis: vec![1.0, 0.0],
        };
        assert!(ragged_basis.validate().is_err());
        let overfull = Subspace {
            dimensions: 2,
            origin: vec![0.0; 2],
            basis: vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        };
        assert!(overfull.validate().is_err());
        let non_finite = Subspace::point(vec![f64::NAN]);
        assert!(non_finite.validate().is_err());
        assert!(
            Subspace {
                dimensions: 0,
                origin: vec![],
                basis: vec![]
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn embed_and_project_are_inverse_on_the_chart() {
        let s = 0.5f64.sqrt();
        let tilted = Subspace {
            dimensions: 3,
            origin: vec![1.0, 2.0, 3.0],
            basis: vec![s, s, 0.0, 0.0, 0.0, 1.0],
        };
        let world = tilted.embed(&[2.0, -1.0]);
        let (chart, dist) = tilted.project(&world);
        assert!((chart[0] - 2.0).abs() < 1e-12);
        assert!((chart[1] - -1.0).abs() < 1e-12);
        assert!(dist < 1e-12);

        // A point off the plane keeps its chart footprint and reports the
        // out-of-plane distance.
        let off = [world[0] - s * 0.5, world[1] + s * 0.5, world[2]];
        let (chart, dist) = tilted.project(&off);
        assert!((chart[0] - 2.0).abs() < 1e-12);
        assert!((dist - 0.5).abs() < 1e-12);
    }

    #[test]
    fn normal_completes_a_positively_oriented_frame() {
        // x/y plane: normal +z.
        let plane = plane_xy_at(0.0);
        let normal = plane.normal().unwrap();
        assert!((normal[2] - 1.0).abs() < 1e-12);
        // Swapped basis flips the normal.
        let flipped = Subspace {
            dimensions: 3,
            origin: vec![0.0; 3],
            basis: vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        };
        assert!((flipped.normal().unwrap()[2] - -1.0).abs() < 1e-12);
        // A line in 2-space: direction +x gives normal... rotate +90
        // degrees, i.e. det([[1, 0], [0, 1]]) > 0 places it at +y.
        let line = Subspace::axis_aligned(vec![0.0, 0.0], &[0]).unwrap();
        let normal = line.normal().unwrap();
        assert!((normal[0]).abs() < 1e-12 && (normal[1] - 1.0).abs() < 1e-12);
        // A point in 1-space: the empty basis orients the normal at +x.
        let point = Subspace::point(vec![5.0]);
        assert_eq!(point.normal().unwrap(), vec![1.0]);
        // Non-hyperplane ranks have no single normal.
        assert!(Subspace::point(vec![0.0, 0.0, 0.0]).normal().is_none());
        assert!(
            Subspace::axis_aligned(vec![0.0; 3], &[0, 1, 2])
                .unwrap()
                .normal()
                .is_none()
        );
    }

    #[test]
    fn from_bounds_selects_box_features() {
        let bounds = [0.0, 4.0, -2.0, 2.0, 10.0, 30.0];
        use BoundSelector::*;

        // Bottom face: spans x/y, sits at z-min, centered in-plane.
        let face = Subspace::from_bounds(&bounds, &[Span, Span, Min]).unwrap();
        assert_eq!(face.rank(), 2);
        assert_eq!(face.origin, vec![2.0, 0.0, 10.0]);
        assert!((face.normal().unwrap()[2] - 1.0).abs() < 1e-12);

        // Vertical edge at x-min / y-max.
        let edge = Subspace::from_bounds(&bounds, &[Min, Max, Span]).unwrap();
        assert_eq!(edge.rank(), 1);
        assert_eq!(edge.origin, vec![0.0, 2.0, 20.0]);
        assert_eq!(edge.basis, vec![0.0, 0.0, 1.0]);

        // A corner point and the box center.
        let corner = Subspace::from_bounds(&bounds, &[Max, Max, Max]).unwrap();
        assert_eq!(corner.rank(), 0);
        assert_eq!(corner.origin, vec![4.0, 2.0, 30.0]);
        let center = Subspace::from_bounds(&bounds, &[Center, Center, Center]).unwrap();
        assert_eq!(center.origin, vec![2.0, 0.0, 20.0]);

        let wrong = Subspace::from_bounds(&bounds[..4], &[Span, Span, Min]);
        assert!(wrong.is_err());
    }

    #[test]
    fn cbor_roundtrip_validates() {
        let plane = plane_xy_at(2.5);
        let bytes = encode_subspace(&plane);
        assert_eq!(decode_subspace(&bytes).unwrap(), plane);
        let garbage = decode_subspace(&encode_subspace(&Subspace {
            dimensions: 3,
            origin: vec![0.0; 3],
            basis: vec![2.0, 0.0, 0.0],
        }));
        assert!(garbage.is_err());
    }
}
