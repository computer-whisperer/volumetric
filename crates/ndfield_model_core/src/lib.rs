//! Dense N-dimensional scalar-field payload shared by operators that bake a
//! regular field and generated WASM evaluators that interpolate it.
//!
//! This is deliberately separate from `gridfield_model_core`'s historical
//! 2D image payload. It is dimension-generic, requires nondegenerate axes,
//! and gives every field an explicit value outside its sampled box. The last
//! property is important for truncated distance fields: a field sampled over
//! the part bounds plus its band can return the positive clamp value globally.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (16 + 20*d bytes):
//!    0          magic          u32    "NDF1"
//!    4          dimensions     u32    d (1..=8)
//!    8          payload_len    u32    header plus values
//!   12          outside_value  f32
//!   16          counts         d*u32  lattice points per axis (>= 2)
//!   16+4d       bounds         2d*f64 [min_0, max_0, ...]
//! Values (product(counts) * f32): axis 0 fastest.
//! ```
//!
//! Lattice points are corner-aligned and samples inside the box use
//! multilinear interpolation. Samples outside it—including NaN positions—
//! return `outside_value`.

pub mod bake;
#[cfg(feature = "emit")]
pub mod emit;

pub const MAGIC: u32 = 0x3146_444e; // "NDF1"
pub const MAX_DIMS: usize = 8;

fn header_len(dimensions: usize) -> usize {
    16 + 20 * dimensions
}

fn validate_geometry(counts: &[usize], bounds: &[f64]) -> Result<usize, String> {
    let dimensions = counts.len();
    if !(1..=MAX_DIMS).contains(&dimensions) {
        return Err(format!(
            "field dimensions must be in 1..={MAX_DIMS}, got {dimensions}"
        ));
    }
    if bounds.len() != 2 * dimensions {
        return Err(format!(
            "field has {dimensions} dimensions but {} bound values",
            bounds.len()
        ));
    }
    let mut value_count = 1usize;
    for axis in 0..dimensions {
        if counts[axis] < 2 {
            return Err(format!(
                "field axis {axis} needs at least two lattice points, got {}",
                counts[axis]
            ));
        }
        let (lo, hi) = (bounds[2 * axis], bounds[2 * axis + 1]);
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(format!(
                "field axis {axis} needs finite nonempty bounds, got [{lo}, {hi}]"
            ));
        }
        value_count = value_count
            .checked_mul(counts[axis])
            .ok_or_else(|| "field lattice size overflows usize".to_string())?;
    }
    Ok(value_count)
}

/// Serialize a dense scalar field. Values are axis-0-fastest and must all be
/// finite, as must the explicit out-of-domain value.
pub fn build_payload(
    counts: &[usize],
    bounds: &[f64],
    values: &[f32],
    outside_value: f32,
) -> Result<Vec<u8>, String> {
    let value_count = validate_geometry(counts, bounds)?;
    if values.len() != value_count {
        return Err(format!(
            "field has {} values, expected {value_count} for counts {counts:?}",
            values.len()
        ));
    }
    if !outside_value.is_finite() {
        return Err(format!(
            "field outside value must be finite, got {outside_value}"
        ));
    }
    if let Some(value) = values.iter().find(|value| !value.is_finite()) {
        return Err(format!("field values must be finite, got {value}"));
    }

    let payload_len = header_len(counts.len())
        .checked_add(value_count * 4)
        .ok_or_else(|| "field payload length overflows usize".to_string())?;
    let payload_len_u32 = u32::try_from(payload_len)
        .map_err(|_| "field payload exceeds the 4 GiB format limit".to_string())?;
    let mut bytes = Vec::with_capacity(payload_len);
    bytes.extend(MAGIC.to_le_bytes());
    bytes.extend((counts.len() as u32).to_le_bytes());
    bytes.extend(payload_len_u32.to_le_bytes());
    bytes.extend(outside_value.to_le_bytes());
    for &count in counts {
        bytes.extend(
            u32::try_from(count)
                .map_err(|_| format!("field axis count {count} exceeds u32"))?
                .to_le_bytes(),
        );
    }
    for &bound in bounds {
        bytes.extend(bound.to_le_bytes());
    }
    for &value in values {
        bytes.extend(value.to_le_bytes());
    }
    debug_assert_eq!(bytes.len(), payload_len);
    Ok(bytes)
}

/// Validated read-only view over an N-dimensional scalar-field payload.
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    dimensions: usize,
    value_count: usize,
}

impl<'a> PayloadView<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() < 36 {
            return Err("payload shorter than the minimum header");
        }
        let u32_at =
            |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err("bad payload magic");
        }
        let dimensions = u32_at(4) as usize;
        if !(1..=MAX_DIMS).contains(&dimensions) || bytes.len() < header_len(dimensions) {
            return Err("field dimension count out of range");
        }
        let mut value_count = 1usize;
        for axis in 0..dimensions {
            let count = u32_at(16 + 4 * axis) as usize;
            if count < 2 {
                return Err("field axis count below two");
            }
            value_count = value_count
                .checked_mul(count)
                .ok_or("field lattice size overflows")?;
            let offset = 16 + 4 * dimensions + 16 * axis;
            let lo = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            let hi = f64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err("field bounds are invalid");
            }
        }
        let expected = header_len(dimensions)
            .checked_add(value_count.checked_mul(4).ok_or("field values overflow")?)
            .ok_or("field payload length overflows")?;
        if u32_at(8) as usize != expected || bytes.len() < expected {
            return Err("field payload length mismatch");
        }
        let outside = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if !outside.is_finite() {
            return Err("field outside value is not finite");
        }
        Ok(Self {
            bytes,
            dimensions,
            value_count,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn count(&self, axis: usize) -> usize {
        let offset = 16 + 4 * axis;
        u32::from_le_bytes(self.bytes[offset..offset + 4].try_into().unwrap()) as usize
    }

    pub fn bound(&self, axis: usize) -> [f64; 2] {
        let offset = 16 + 4 * self.dimensions + 16 * axis;
        std::array::from_fn(|side| {
            f64::from_le_bytes(
                self.bytes[offset + side * 8..offset + (side + 1) * 8]
                    .try_into()
                    .unwrap(),
            )
        })
    }

    pub fn outside_value(&self) -> f32 {
        f32::from_le_bytes(self.bytes[12..16].try_into().unwrap())
    }

    pub fn value_count(&self) -> usize {
        self.value_count
    }

    fn value(&self, index: usize) -> f64 {
        let offset = header_len(self.dimensions) + index * 4;
        f32::from_le_bytes(self.bytes[offset..offset + 4].try_into().unwrap()) as f64
    }

    /// Multilinearly sample the field. Out-of-domain and NaN positions return
    /// the payload's explicit outside value.
    pub fn sample(&self, position: &[f64]) -> f32 {
        if position.len() != self.dimensions {
            return self.outside_value();
        }
        let mut base = [0usize; MAX_DIMS];
        let mut fraction = [0.0f64; MAX_DIMS];
        for axis in 0..self.dimensions {
            let [lo, hi] = self.bound(axis);
            let coordinate = position[axis];
            if !(coordinate >= lo && coordinate <= hi) {
                return self.outside_value();
            }
            let count = self.count(axis);
            let grid =
                ((coordinate - lo) / (hi - lo) * (count - 1) as f64).clamp(0.0, (count - 1) as f64);
            base[axis] = (grid.floor() as usize).min(count - 2);
            fraction[axis] = grid - base[axis] as f64;
        }

        let mut result = 0.0f64;
        for corner in 0..1usize << self.dimensions {
            let mut weight = 1.0f64;
            let mut index = 0usize;
            let mut stride = 1usize;
            for axis in 0..self.dimensions {
                let upper = corner >> axis & 1 == 1;
                weight *= if upper {
                    fraction[axis]
                } else {
                    1.0 - fraction[axis]
                };
                index += (base[axis] + usize::from(upper)) * stride;
                stride *= self.count(axis);
            }
            if weight > 0.0 {
                result += weight * self.value(index);
            }
        }
        result as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_payload() -> Vec<u8> {
        // value = x + 10y + 100z, axis 0 fastest.
        let mut values = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..3 {
                    values.push((x + 10 * y + 100 * z) as f32);
                }
            }
        }
        build_payload(&[3, 2, 2], &[0.0, 2.0, 0.0, 1.0, 0.0, 1.0], &values, -7.0).unwrap()
    }

    #[test]
    fn interpolates_an_nd_affine_field() {
        let payload = plane_payload();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.dimensions(), 3);
        assert_eq!(view.value_count(), 12);
        assert_eq!(view.sample(&[2.0, 1.0, 1.0]), 112.0);
        assert!((view.sample(&[0.5, 0.25, 0.75]) - 78.0).abs() < 1e-5);
    }

    #[test]
    fn outside_value_is_explicit_and_global() {
        let payload = plane_payload();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.outside_value(), -7.0);
        assert_eq!(view.sample(&[-0.001, 0.5, 0.5]), -7.0);
        assert_eq!(view.sample(&[2.001, 0.5, 0.5]), -7.0);
        assert_eq!(view.sample(&[f64::NAN, 0.5, 0.5]), -7.0);
        assert_eq!(view.sample(&[0.0, 0.0]), -7.0);
    }

    #[test]
    fn invalid_payloads_are_rejected() {
        assert!(build_payload(&[], &[], &[], 0.0).is_err());
        assert!(build_payload(&[1], &[0.0, 1.0], &[0.0], 0.0).is_err());
        assert!(build_payload(&[2], &[1.0, 1.0], &[0.0; 2], 0.0).is_err());
        assert!(build_payload(&[2], &[0.0, 1.0], &[0.0], 0.0).is_err());
        assert!(build_payload(&[2], &[0.0, 1.0], &[f32::NAN; 2], 0.0).is_err());
        assert!(PayloadView::new(&[0u8; 16]).is_err());
        let mut bad = plane_payload();
        bad[0] ^= 0xff;
        assert!(PayloadView::new(&bad).is_err());
        bad[0] ^= 0xff;
        bad.truncate(bad.len() - 1);
        assert!(PayloadView::new(&bad).is_err());
    }
}
