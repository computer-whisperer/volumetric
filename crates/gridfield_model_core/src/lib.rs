//! The 2D grid-field model payload: the data contract between operators
//! that bake a scalar field onto a regular grid (`image_model_operator`
//! being the first) and `gridfield_model_template`, which bilinearly
//! samples it at model sample time.
//!
//! Same shape as the trimesh payload pattern: the operator does all the
//! work up front in [`build_payload`], the generated model is stateless
//! ([`PayloadView`] only reads), and both sides live in this one natively
//! unit-tested crate so the layout can't drift.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (48 bytes):
//!    0  magic       u32   "GRD1" (0x3144_5247)
//!    4  width       u32   grid columns (>= 1)
//!    8  height      u32   grid rows (>= 1)
//!   12  payload_len u32   total byte length, header included
//!   16  bounds      4xf64 [min_x, max_x, min_y, max_y]
//! Values (width * height x f32 at offset 48): row-major, row 0 at min_y
//! (y increases with row index).
//! ```
//!
//! # Sampling semantics
//!
//! Grid values sit corner-aligned: value (0, 0) lives exactly at
//! (min_x, min_y) and value (width-1, height-1) at (max_x, max_y), with
//! bilinear interpolation between. Outside the bounds rectangle the field
//! is undefined and [`PayloadView::sample`] returns `None`; degenerate
//! axes (width or height of 1, or zero-extent bounds) clamp to the single
//! row/column.

pub const MAGIC: u32 = 0x3144_5247; // "GRD1"
const HEADER_LEN: usize = 48;

/// Serialize a grid field. `values` is row-major, row 0 at `min_y`, and
/// must hold exactly `width * height` entries.
pub fn build_payload(
    width: u32,
    height: u32,
    bounds: [f64; 4],
    values: &[f32],
) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("grid field must have at least one column and row".to_string());
    }
    if values.len() != (width as usize) * (height as usize) {
        return Err(format!(
            "grid field value count {} doesn't match {width}x{height}",
            values.len()
        ));
    }
    let [min_x, max_x, min_y, max_y] = bounds;
    if !(min_x < max_x && min_y < max_y) {
        return Err(format!(
            "grid field bounds must be a nonempty rectangle, got x [{min_x}, {max_x}] \
             y [{min_y}, {max_y}]"
        ));
    }

    let payload_len = HEADER_LEN + values.len() * 4;
    let mut out = Vec::with_capacity(payload_len);
    out.extend(MAGIC.to_le_bytes());
    out.extend(width.to_le_bytes());
    out.extend(height.to_le_bytes());
    out.extend((payload_len as u32).to_le_bytes());
    for v in bounds {
        out.extend(v.to_le_bytes());
    }
    for v in values {
        out.extend(v.to_le_bytes());
    }
    debug_assert_eq!(out.len(), payload_len);
    Ok(out)
}

/// A read-only view over a serialized payload (sample-time side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    width: usize,
    height: usize,
}

impl<'a> PayloadView<'a> {
    /// Validate the header and structural sizes.
    pub fn new(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_LEN {
            return Err("payload shorter than header");
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err("bad payload magic");
        }
        let width = u32_at(4) as usize;
        let height = u32_at(8) as usize;
        let payload_len = u32_at(12) as usize;
        let expected = HEADER_LEN
            + width
                .checked_mul(height)
                .and_then(|n| n.checked_mul(4))
                .ok_or("grid dimensions overflow")?;
        if width == 0 || height == 0 || payload_len != expected || bytes.len() < expected {
            return Err("payload length mismatch");
        }
        Ok(Self {
            bytes,
            width,
            height,
        })
    }

    /// `[min_x, max_x, min_y, max_y]`.
    pub fn bounds(&self) -> [f64; 4] {
        std::array::from_fn(|i| {
            f64::from_le_bytes(self.bytes[16 + i * 8..24 + i * 8].try_into().unwrap())
        })
    }

    fn value(&self, col: usize, row: usize) -> f64 {
        let off = HEADER_LEN + (row * self.width + col) * 4;
        f32::from_le_bytes(self.bytes[off..off + 4].try_into().unwrap()) as f64
    }

    /// Bilinear sample at `(x, y)`; `None` outside the bounds rectangle.
    pub fn sample(&self, x: f64, y: f64) -> Option<f32> {
        let [min_x, max_x, min_y, max_y] = self.bounds();
        if !(x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
            return None; // also rejects NaN coordinates
        }
        // Corner-aligned: map the bounds span onto [0, n-1]. A single
        // row/column has nothing to interpolate along that axis.
        let grid_coord = |p: f64, lo: f64, hi: f64, n: usize| -> (usize, usize, f64) {
            if n == 1 || hi <= lo {
                return (0, 0, 0.0);
            }
            let g = ((p - lo) / (hi - lo) * (n - 1) as f64).clamp(0.0, (n - 1) as f64);
            let i0 = (g.floor() as usize).min(n - 2);
            (i0, i0 + 1, g - i0 as f64)
        };
        let (c0, c1, fx) = grid_coord(x, min_x, max_x, self.width);
        let (r0, r1, fy) = grid_coord(y, min_y, max_y, self.height);

        let v = (1.0 - fx) * (1.0 - fy) * self.value(c0, r0)
            + fx * (1.0 - fy) * self.value(c1, r0)
            + (1.0 - fx) * fy * self.value(c0, r1)
            + fx * fy * self.value(c1, r1);
        Some(v as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_payload() -> Vec<u8> {
        // 3x2 grid over [0,2]x[0,1]: value = x + 10*y at the grid corners.
        let values = [0.0f32, 1.0, 2.0, 10.0, 11.0, 12.0];
        build_payload(3, 2, [0.0, 2.0, 0.0, 1.0], &values).unwrap()
    }

    #[test]
    fn corners_and_bilinear_interior() {
        let payload = ramp_payload();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.bounds(), [0.0, 2.0, 0.0, 1.0]);

        // Exact at the corner-aligned grid points.
        assert_eq!(view.sample(0.0, 0.0), Some(0.0));
        assert_eq!(view.sample(2.0, 0.0), Some(2.0));
        assert_eq!(view.sample(0.0, 1.0), Some(10.0));
        assert_eq!(view.sample(2.0, 1.0), Some(12.0));
        assert_eq!(view.sample(1.0, 0.0), Some(1.0));

        // A bilinear ramp reproduces the plane everywhere.
        assert!((view.sample(0.5, 0.25).unwrap() - 3.0).abs() < 1e-6);
        assert!((view.sample(1.7, 0.9).unwrap() - 10.7).abs() < 1e-5);
    }

    #[test]
    fn outside_bounds_is_none() {
        let payload = ramp_payload();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.sample(-0.1, 0.5), None);
        assert_eq!(view.sample(2.1, 0.5), None);
        assert_eq!(view.sample(1.0, -0.1), None);
        assert_eq!(view.sample(1.0, 1.1), None);
        assert_eq!(view.sample(f64::NAN, 0.5), None);
    }

    #[test]
    fn single_row_and_column_clamp() {
        let payload = build_payload(2, 1, [0.0, 1.0, 0.0, 1.0], &[3.0, 5.0]).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.sample(0.0, 0.0), Some(3.0));
        assert_eq!(view.sample(1.0, 1.0), Some(5.0));
        assert_eq!(view.sample(0.5, 0.7), Some(4.0));
    }

    #[test]
    fn invalid_payloads_are_rejected() {
        assert!(build_payload(0, 2, [0.0, 1.0, 0.0, 1.0], &[]).is_err());
        assert!(build_payload(2, 2, [0.0, 1.0, 0.0, 1.0], &[1.0; 3]).is_err());
        assert!(build_payload(2, 2, [1.0, 1.0, 0.0, 1.0], &[1.0; 4]).is_err());
        assert!(PayloadView::new(&[0u8; 16]).is_err());
        let mut bad = ramp_payload();
        bad[0] ^= 0xFF;
        assert!(PayloadView::new(&bad).is_err());
        bad[0] ^= 0xFF;
        bad.truncate(50);
        assert!(PayloadView::new(&bad).is_err());
    }
}
