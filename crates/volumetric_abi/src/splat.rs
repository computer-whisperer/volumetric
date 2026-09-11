//! Gaussian splats: the `Splat` value.
//!
//! A splat is the trained product of the scan pipeline: a set of oriented
//! Gaussians (or flat surfels) in the world frame of the view set it was
//! trained from, each with a position, three log-scales, a rotation
//! quaternion, a logit opacity and spherical-harmonic colour. It is what
//! the trainer returns and what the viewport renders over the photographs
//! to show where the geometry can be trusted.
//!
//! The columns are stored packed (little-endian `f32` byte strings) in the
//! layouts the 3DGS PLY export uses, so a set of six hundred thousand
//! Gaussians at SH degree 2 is about 90 MB and decodes in milliseconds.
//! The parameters are stored as the trainer keeps them (log-scales, logit
//! opacities, SH coefficients, unnormalised quaternions); the accessors
//! here apply the activations.

use serde::{Deserialize, Serialize};

pub use crate::viewset::{Provenance, WorldFrame};

/// The schema this module writes.
pub const SPLAT_SCHEMA: u32 = 1;

/// The zeroth spherical-harmonic basis constant: colour = 0.5 + C0 · sh0.
pub const SH_C0: f32 = 0.282_094_79;

/// What one primitive is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplatKind {
    /// A 3D Gaussian with three scales (3DGS).
    Gaussian3d,
    /// A flat surfel: the third scale is near zero and the third axis is
    /// the surface normal (2DGS).
    Surfel2d,
}

impl SplatKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gaussian3d => "gaussian",
            Self::Surfel2d => "surfel",
        }
    }
}

/// A trained set of Gaussians or surfels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Splat {
    pub schema: u32,
    pub world: WorldFrame,
    /// The capture the splat was trained from, so combinators can refuse to
    /// mix evidence from different setups.
    pub provenance: Provenance,
    /// Content hash (hex) of the view set the splat was trained from; empty
    /// when unknown.
    #[serde(default)]
    pub views_hash: String,
    /// The trainer and its arguments, as text; empty when unknown.
    #[serde(default)]
    pub training: String,
    pub kind: SplatKind,
    /// Spherical-harmonic degree of the colour, 0 to 3.
    pub sh_degree: u32,
    /// Number of primitives, `N`.
    pub count: u32,
    /// Centres, xyz interleaved (3N), metres.
    #[serde(with = "f32_bytes")]
    pub means: Vec<f32>,
    /// Log-scales along the local axes (3N); a surfel's third is near
    /// zero.
    #[serde(with = "f32_bytes")]
    pub scales: Vec<f32>,
    /// Rotation quaternions, `w x y z` interleaved (4N), not necessarily
    /// unit length.
    #[serde(with = "f32_bytes")]
    pub quats: Vec<f32>,
    /// Logit opacities (N).
    #[serde(with = "f32_bytes")]
    pub opacities: Vec<f32>,
    /// Degree-0 SH coefficients, rgb interleaved (3N).
    #[serde(with = "f32_bytes")]
    pub sh0: Vec<f32>,
    /// Higher-degree SH coefficients, `3 · ((d + 1)² − 1)` per primitive,
    /// channel-major within a primitive as the 3DGS PLY stores them (all
    /// of red's coefficients, then green's, then blue's).
    #[serde(with = "f32_bytes")]
    pub sh_rest: Vec<f32>,
    /// Unit normals, xyz interleaved (3N), or empty when the trainer gave
    /// none; [`Splat::normal`] derives one from the orientation then.
    #[serde(default, with = "f32_bytes")]
    pub normals: Vec<f32>,
}

/// Serde adapter storing a `Vec<f32>` as a little-endian byte string.
mod f32_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(values: &[f32], serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        serde_bytes::Bytes::new(&bytes).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f32>, D::Error> {
        let bytes = serde_bytes::ByteBuf::deserialize(deserializer)?;
        if bytes.len() % 4 != 0 {
            return Err(serde::de::Error::custom(format!(
                "a float column of {} bytes is not a whole number of f32s",
                bytes.len()
            )));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// Coefficients per primitive beyond degree 0 for an SH degree.
pub fn sh_rest_per_point(sh_degree: u32) -> usize {
    3 * (((sh_degree + 1) * (sh_degree + 1)) as usize - 1)
}

impl Splat {
    /// An empty splat of a kind, to fill.
    pub fn empty(kind: SplatKind, sh_degree: u32) -> Self {
        Self {
            schema: SPLAT_SCHEMA,
            world: WorldFrame::default(),
            provenance: Provenance::default(),
            views_hash: String::new(),
            training: String::new(),
            kind,
            sh_degree,
            count: 0,
            means: Vec::new(),
            scales: Vec::new(),
            quats: Vec::new(),
            opacities: Vec::new(),
            sh0: Vec::new(),
            sh_rest: Vec::new(),
            normals: Vec::new(),
        }
    }

    /// Structural checks: schema, degree, column lengths against `count`,
    /// finite values, a unit world up, and no zero quaternion.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SPLAT_SCHEMA {
            return Err(format!(
                "splat schema {} is not the supported {SPLAT_SCHEMA}",
                self.schema
            ));
        }
        if self.sh_degree > 3 {
            return Err(format!("SH degree {} is above 3", self.sh_degree));
        }
        let up = self.world.up;
        let up_len = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
        if !(up_len.is_finite() && (up_len - 1.0).abs() < 1e-6) {
            return Err("world up axis must be a unit vector".to_string());
        }
        let n = self.count as usize;
        let columns: [(&str, &[f32], usize); 7] = [
            ("means", &self.means, 3),
            ("scales", &self.scales, 3),
            ("quats", &self.quats, 4),
            ("opacities", &self.opacities, 1),
            ("sh0", &self.sh0, 3),
            ("sh_rest", &self.sh_rest, sh_rest_per_point(self.sh_degree)),
            ("normals", &self.normals, 3),
        ];
        for (name, column, per_point) in columns {
            let expected = n * per_point;
            if column.len() != expected && !(name == "normals" && column.is_empty()) {
                return Err(format!(
                    "{name} holds {} values for {n} primitives; expected {expected}",
                    column.len()
                ));
            }
            if let Some(pos) = column.iter().position(|v| !v.is_finite()) {
                return Err(format!(
                    "{name}[{pos}] is not finite (primitive {})",
                    pos / per_point.max(1)
                ));
            }
        }
        if let Some(i) = self
            .quats
            .chunks_exact(4)
            .position(|q| q.iter().all(|v| *v == 0.0))
        {
            return Err(format!("primitive {i} has a zero quaternion"));
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The centre of primitive `i`.
    pub fn mean(&self, i: usize) -> [f32; 3] {
        [
            self.means[3 * i],
            self.means[3 * i + 1],
            self.means[3 * i + 2],
        ]
    }

    /// The standard deviations along the local axes of primitive `i`,
    /// metres (the exponential of the stored log-scales).
    pub fn scale(&self, i: usize) -> [f32; 3] {
        [
            self.scales[3 * i].exp(),
            self.scales[3 * i + 1].exp(),
            self.scales[3 * i + 2].exp(),
        ]
    }

    /// The unit quaternion `w x y z` of primitive `i`.
    pub fn quat(&self, i: usize) -> [f32; 4] {
        let q = &self.quats[4 * i..4 * i + 4];
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }

    /// The rotation of primitive `i` as rows: `world = R · local`, so the
    /// columns are the local axes in world coordinates.
    pub fn rotation(&self, i: usize) -> [[f32; 3]; 3] {
        let [w, x, y, z] = self.quat(i);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    /// Local axis `k` of primitive `i` in world coordinates, unit length.
    pub fn axis(&self, i: usize, k: usize) -> [f32; 3] {
        let r = self.rotation(i);
        [r[0][k], r[1][k], r[2][k]]
    }

    /// The opacity of primitive `i` in `[0, 1]` (the sigmoid of the stored
    /// logit).
    pub fn opacity(&self, i: usize) -> f32 {
        1.0 / (1.0 + (-self.opacities[i]).exp())
    }

    /// The view-independent colour of primitive `i` in `[0, 1]`, from the
    /// degree-0 coefficients.
    pub fn color(&self, i: usize) -> [f32; 3] {
        let c = |v: f32| (0.5 + SH_C0 * v).clamp(0.0, 1.0);
        [
            c(self.sh0[3 * i]),
            c(self.sh0[3 * i + 1]),
            c(self.sh0[3 * i + 2]),
        ]
    }

    /// The surface normal of primitive `i`: the stored normal when the
    /// trainer gave one, else a surfel's third axis or a Gaussian's
    /// smallest-scale axis. The sign is the trainer's; nothing here orients
    /// it.
    pub fn normal(&self, i: usize) -> [f32; 3] {
        if !self.normals.is_empty() {
            let n = &self.normals[3 * i..3 * i + 3];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 0.0 {
                return [n[0] / len, n[1] / len, n[2] / len];
            }
        }
        let k = match self.kind {
            SplatKind::Surfel2d => 2,
            SplatKind::Gaussian3d => {
                let s = &self.scales[3 * i..3 * i + 3];
                (0..3).min_by(|a, b| s[*a].total_cmp(&s[*b])).unwrap_or(2)
            }
        };
        self.axis(i, k)
    }

    /// The largest standard deviation of primitive `i`, metres.
    pub fn extent(&self, i: usize) -> f32 {
        let s = &self.scales[3 * i..3 * i + 3];
        s[0].max(s[1]).max(s[2]).exp()
    }

    /// Per-axis bounds of the centres at the `lo` and `hi` quantiles (in
    /// `[0, 1]`), which ignore the stray primitives a trainer leaves far
    /// out; `None` when empty.
    pub fn bounds_quantile(&self, lo: f64, hi: f64) -> Option<([f64; 3], [f64; 3])> {
        if self.is_empty() {
            return None;
        }
        let mut min = [0.0; 3];
        let mut max = [0.0; 3];
        for axis in 0..3 {
            let mut values: Vec<f32> = self.means.iter().skip(axis).step_by(3).copied().collect();
            values.sort_by(f32::total_cmp);
            min[axis] = quantile_sorted(&values, lo) as f64;
            max[axis] = quantile_sorted(&values, hi) as f64;
        }
        Some((min, max))
    }
}

/// The `q` quantile of an ascending slice, by nearest rank.
pub fn quantile_sorted(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let rank = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Encodes a splat as CBOR; the columns travel as byte strings.
pub fn encode_splat(splat: &Splat) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(splat, &mut out).expect("splat CBOR serialization should not fail");
    out
}

/// Decodes and validates a splat.
pub fn decode_splat(bytes: &[u8]) -> Result<Splat, String> {
    let splat: Splat = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode splat CBOR: {e}"))?;
    splat.validate()?;
    Ok(splat)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two primitives: one axis-aligned at the origin, one rotated 90°
    /// about z at (1, 2, 3) with its smallest scale along x.
    fn sample() -> Splat {
        let mut s = Splat::empty(SplatKind::Gaussian3d, 1);
        s.count = 2;
        s.means = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        s.scales = vec![
            0.0,
            (0.5f32).ln(),
            (0.25f32).ln(),
            (0.01f32).ln(),
            (0.1f32).ln(),
            (0.1f32).ln(),
        ];
        let h = std::f32::consts::FRAC_1_SQRT_2;
        s.quats = vec![1.0, 0.0, 0.0, 0.0, 2.0 * h, 0.0, 0.0, 2.0 * h];
        s.opacities = vec![0.0, 20.0];
        s.sh0 = vec![0.0, 0.0, 0.0, 1.0 / SH_C0, -10.0, 0.0];
        s.sh_rest = vec![0.5; 2 * sh_rest_per_point(1)];
        s
    }

    #[test]
    fn round_trips_through_cbor_as_packed_columns() {
        let s = sample();
        let bytes = encode_splat(&s);
        let back = decode_splat(&bytes).unwrap();
        assert_eq!(back, s);
        // Every f32 costs four bytes plus the fixed fields: far under the
        // five bytes an array element would take.
        let floats = 6 + 6 + 8 + 2 + 6 + s.sh_rest.len();
        assert!(bytes.len() < floats * 4 + 400, "{} bytes", bytes.len());
    }

    #[test]
    fn activations_and_axes() {
        let s = sample();
        assert_eq!(s.mean(1), [1.0, 2.0, 3.0]);
        let sc = s.scale(0);
        assert!((sc[0] - 1.0).abs() < 1e-6 && (sc[1] - 0.5).abs() < 1e-6);
        assert!((s.opacity(0) - 0.5).abs() < 1e-6);
        assert!(s.opacity(1) > 0.999);
        assert_eq!(s.color(0), [0.5, 0.5, 0.5]);
        assert_eq!(s.color(1), [1.0, 0.0, 0.5]);
        // The unnormalised quaternion (2h, 0, 0, 2h) is a 90° turn about
        // z: local x lands on world y.
        let q = s.quat(1);
        assert!((q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] - 1.0).abs() < 1e-6);
        let x = s.axis(1, 0);
        assert!((x[0]).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6 && x[2].abs() < 1e-6);
        // The Gaussian's normal is its smallest-scale axis (local x).
        assert_eq!(s.normal(1), x);
        assert_eq!(s.normal(0), [0.0, 0.0, 1.0]);
        assert!((s.extent(1) - 0.1).abs() < 1e-6);
        // A surfel's normal is its third axis whatever the scales.
        let mut surfel = s.clone();
        surfel.kind = SplatKind::Surfel2d;
        let z = surfel.axis(1, 2);
        assert_eq!(surfel.normal(1), z);
        // Stored normals win.
        surfel.normals = vec![0.0, 2.0, 0.0, 0.0, 0.0, -3.0];
        assert_eq!(surfel.normal(1), [0.0, 0.0, -1.0]);
        assert!(surfel.validate().is_ok());
        let (lo, hi) = s.bounds_quantile(0.0, 1.0).unwrap();
        assert_eq!(lo, [0.0, 0.0, 0.0]);
        assert_eq!(hi, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn validation_catches_lengths_zero_quaternions_and_nans() {
        let mut s = sample();
        s.sh_degree = 2;
        assert!(s.validate().unwrap_err().contains("sh_rest"));
        let mut s = sample();
        s.quats[4..8].fill(0.0);
        assert!(s.validate().unwrap_err().contains("zero quaternion"));
        let mut s = sample();
        s.means[4] = f32::NAN;
        let err = s.validate().unwrap_err();
        assert!(
            err.contains("means[4]") && err.contains("primitive 1"),
            "{err}"
        );
        let mut s = sample();
        s.normals = vec![0.0; 3];
        assert!(s.validate().unwrap_err().contains("normals"));
        let mut s = sample();
        s.schema = 7;
        assert!(decode_splat(&encode_splat(&s)).is_err());
    }

    #[test]
    fn sh_layout_and_quantiles() {
        assert_eq!(sh_rest_per_point(0), 0);
        assert_eq!(sh_rest_per_point(1), 9);
        assert_eq!(sh_rest_per_point(2), 24);
        assert_eq!(sh_rest_per_point(3), 45);
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_sorted(&v, 0.0), 1.0);
        assert_eq!(quantile_sorted(&v, 0.5), 3.0);
        assert_eq!(quantile_sorted(&v, 1.0), 5.0);
        assert!(quantile_sorted(&[], 0.5).is_nan());
    }
}
