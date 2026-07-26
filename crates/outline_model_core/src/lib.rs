//! The 2D filled-outline model payload: the data contract between
//! operators that bake closed contours (`text_model_operator` being the
//! first) and `outline_model_template`, which classifies sample points
//! against them at model sample time.
//!
//! Same shape as the grid-field payload pattern: the operator does all the
//! work up front in [`build_payload`] (Bézier flattening, layout — anything
//! curve-shaped arrives here already as polylines), the generated model is
//! stateless ([`PayloadView`] only reads), and both sides live in this one
//! natively unit-tested crate so the layout can't drift.
//!
//! # Geometry semantics
//!
//! A payload is a set of closed contours flattened to line segments. A
//! point is inside iff its **nonzero winding number** is nonzero — the
//! TrueType fill rule, so glyph outlines (holes wound opposite to their
//! outer contour) classify correctly, as do self-overlapping unions.
//! Points exactly on a segment classify by the half-open crossing rule;
//! callers must not rely on boundary-exact samples.
//!
//! # Payload layout (little-endian)
//!
//! ```text
//! Header (48 bytes):
//!    0  magic         u32   "OTL1" (0x314C_544F)
//!    4  segment_count u32
//!    8  band_count    u32   y-band acceleration index (>= 1)
//!   12  payload_len   u32   total byte length, header included
//!   16  bounds        4xf64 [min_x, max_x, min_y, max_y] (tight)
//! Band table (band_count x 8 at 48): (start u32, len u32) into the ref
//!   array. Band i covers y in [min_y + i*h, min_y + (i+1)*h) where
//!   h = (max_y - min_y) / band_count; a segment is referenced from every
//!   band its y-range overlaps.
//! Refs (ref_count x u32): segment indices, grouped by band.
//! Segments (segment_count x 32): ax, ay, bx, by as f64.
//! ```
//!
//! [`PayloadView::new`] validates the header and region sizes in O(1) (it
//! runs on every `sample` call in the template); ref/segment reads are
//! bounds-guarded at query time, and anything malformed reads as outside —
//! the ABI's errors-read-as-outside convention.

pub const MAGIC: u32 = 0x314C_544F; // "OTL1"
const HEADER_LEN: usize = 48;
const BAND_ENTRY_LEN: usize = 8;
const SEGMENT_LEN: usize = 32;

/// Serialize a set of closed contours. Each contour is a polyline of `[x,
/// y]` points implicitly closed from its last point back to its first;
/// contours with fewer than 3 distinct points are rejected. Bounds are the
/// tight bounding box of all points.
pub fn build_payload(contours: &[Vec<[f64; 2]>]) -> Result<Vec<u8>, String> {
    let mut segments: Vec<[f64; 4]> = Vec::new();
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);

    for contour in contours {
        let mut count = 0usize;
        for i in 0..contour.len() {
            let a = contour[i];
            let b = contour[(i + 1) % contour.len()];
            if !(a[0].is_finite() && a[1].is_finite()) {
                return Err("contour contains a non-finite point".to_string());
            }
            min_x = min_x.min(a[0]);
            max_x = max_x.max(a[0]);
            min_y = min_y.min(a[1]);
            max_y = max_y.max(a[1]);
            if a != b {
                segments.push([a[0], a[1], b[0], b[1]]);
                count += 1;
            }
        }
        if count < 3 {
            return Err(format!(
                "contour with {count} distinct segments cannot enclose area (need >= 3)"
            ));
        }
    }
    if segments.is_empty() {
        return Err("outline payload needs at least one contour".to_string());
    }
    if !(min_x < max_x && min_y < max_y) {
        return Err(format!(
            "outline bounds must be a nonempty rectangle, got x [{min_x}, {max_x}] \
             y [{min_y}, {max_y}]"
        ));
    }

    // Y-band index: enough bands that a query touches a small slice of the
    // segment set, capped so tiny payloads don't carry a huge empty table.
    let band_count = (segments.len() / 8).clamp(1, 256);
    let band_h = (max_y - min_y) / band_count as f64;
    let band_range = |a_y: f64, b_y: f64| -> (usize, usize) {
        let (lo, hi) = (a_y.min(b_y), a_y.max(b_y));
        let first = ((lo - min_y) / band_h).floor().max(0.0) as usize;
        let last = (((hi - min_y) / band_h).floor() as usize).min(band_count - 1);
        (first.min(band_count - 1), last)
    };
    let mut bands: Vec<Vec<u32>> = vec![Vec::new(); band_count];
    for (idx, seg) in segments.iter().enumerate() {
        let (first, last) = band_range(seg[1], seg[3]);
        for band in &mut bands[first..=last] {
            band.push(idx as u32);
        }
    }

    let ref_count: usize = bands.iter().map(Vec::len).sum();
    let payload_len =
        HEADER_LEN + band_count * BAND_ENTRY_LEN + ref_count * 4 + segments.len() * SEGMENT_LEN;
    if payload_len > u32::MAX as usize {
        return Err("outline payload exceeds 4 GiB".to_string());
    }

    let mut out = Vec::with_capacity(payload_len);
    out.extend(MAGIC.to_le_bytes());
    out.extend((segments.len() as u32).to_le_bytes());
    out.extend((band_count as u32).to_le_bytes());
    out.extend((payload_len as u32).to_le_bytes());
    for v in [min_x, max_x, min_y, max_y] {
        out.extend(v.to_le_bytes());
    }
    let mut start = 0u32;
    for band in &bands {
        out.extend(start.to_le_bytes());
        out.extend((band.len() as u32).to_le_bytes());
        start += band.len() as u32;
    }
    for band in &bands {
        for r in band {
            out.extend(r.to_le_bytes());
        }
    }
    for seg in &segments {
        for v in seg {
            out.extend(v.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), payload_len);
    Ok(out)
}

/// A read-only view over a serialized payload (sample-time side).
pub struct PayloadView<'a> {
    bytes: &'a [u8],
    segment_count: usize,
    band_count: usize,
    ref_count: usize,
}

impl<'a> PayloadView<'a> {
    /// Validate the header and region sizes (O(1) — this runs per sample).
    pub fn new(bytes: &'a [u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_LEN {
            return Err("payload shorter than header");
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err("bad payload magic");
        }
        let segment_count = u32_at(4) as usize;
        let band_count = u32_at(8) as usize;
        let payload_len = u32_at(12) as usize;
        if segment_count == 0 || band_count == 0 {
            return Err("empty outline payload");
        }
        let fixed = HEADER_LEN
            .checked_add(
                band_count
                    .checked_mul(BAND_ENTRY_LEN)
                    .ok_or("band overflow")?,
            )
            .and_then(|n| n.checked_add(segment_count.checked_mul(SEGMENT_LEN)?))
            .ok_or("payload size overflow")?;
        if payload_len < fixed || (payload_len - fixed) % 4 != 0 || bytes.len() < payload_len {
            return Err("payload length mismatch");
        }
        Ok(Self {
            bytes,
            segment_count,
            band_count,
            ref_count: (payload_len - fixed) / 4,
        })
    }

    /// `[min_x, max_x, min_y, max_y]`.
    pub fn bounds(&self) -> [f64; 4] {
        std::array::from_fn(|i| {
            f64::from_le_bytes(self.bytes[16 + i * 8..24 + i * 8].try_into().unwrap())
        })
    }

    fn segment(&self, idx: usize) -> [f64; 4] {
        let off =
            HEADER_LEN + self.band_count * BAND_ENTRY_LEN + self.ref_count * 4 + idx * SEGMENT_LEN;
        std::array::from_fn(|i| {
            f64::from_le_bytes(
                self.bytes[off + i * 8..off + (i + 1) * 8]
                    .try_into()
                    .unwrap(),
            )
        })
    }

    /// Nonzero-winding containment test; `false` outside the bounds
    /// rectangle, for NaN coordinates, and for malformed band/ref data.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let [min_x, max_x, min_y, max_y] = self.bounds();
        if !(x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
            return false; // also rejects NaN coordinates
        }
        // Bit-identical to the build side's band assignment: recompute
        // band_h with the same expression and divide by it, so a query at
        // a boundary-exact y lands within the band range its crossing
        // segments were referenced from. (0/0 -> NaN casts to 0; x/0 ->
        // inf saturates and clamps — both match the build-side clamping.)
        let band_h = (max_y - min_y) / self.band_count as f64;
        let band = (((y - min_y) / band_h) as usize).min(self.band_count - 1);
        let entry = HEADER_LEN + band * BAND_ENTRY_LEN;
        let u32_at = |off: usize| -> Option<u32> {
            Some(u32::from_le_bytes(
                self.bytes.get(off..off + 4)?.try_into().unwrap(),
            ))
        };
        let Some((start, len)) = u32_at(entry).zip(u32_at(entry + 4)) else {
            return false;
        };
        let (start, len) = (start as usize, len as usize);
        if start
            .checked_add(len)
            .is_none_or(|end| end > self.ref_count)
        {
            return false;
        }

        // Winding number via signed crossings of the +x ray from (x, y).
        // The half-open `>` test gives every vertex to exactly one of its
        // segments, so crossings at shared endpoints count once.
        let refs_base = HEADER_LEN + self.band_count * BAND_ENTRY_LEN;
        let mut winding = 0i32;
        for i in start..start + len {
            let Some(seg_idx) = u32_at(refs_base + i * 4) else {
                return false;
            };
            let seg_idx = seg_idx as usize;
            if seg_idx >= self.segment_count {
                return false;
            }
            let [ax, ay, bx, by] = self.segment(seg_idx);
            if (ay > y) != (by > y) {
                let t = (y - ay) / (by - ay);
                if ax + t * (bx - ax) > x {
                    winding += if by > ay { 1 } else { -1 };
                }
            }
        }
        winding != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(cx: f64, cy: f64, half: f64, ccw: bool) -> Vec<[f64; 2]> {
        let mut pts = vec![
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ];
        if !ccw {
            pts.reverse();
        }
        pts
    }

    #[test]
    fn square_with_hole_uses_nonzero_winding() {
        // Outer CCW square with an opposite-wound inner square: an annulus.
        let payload =
            build_payload(&[square(0.0, 0.0, 2.0, true), square(0.0, 0.0, 1.0, false)]).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert_eq!(view.bounds(), [-2.0, 2.0, -2.0, 2.0]);

        assert!(!view.contains(0.0, 0.0)); // in the hole
        assert!(view.contains(1.5, 0.0)); // in the ring
        assert!(view.contains(0.0, -1.5));
        assert!(view.contains(-1.5, 1.5)); // ring corner region
        assert!(!view.contains(2.5, 0.0)); // outside bounds
        assert!(!view.contains(0.0, 2.5));
        assert!(!view.contains(f64::NAN, 0.0));
        assert!(!view.contains(0.0, f64::NAN));
    }

    #[test]
    fn overlapping_same_direction_contours_union() {
        // Nonzero winding: two same-wound overlapping squares fill solid.
        let payload =
            build_payload(&[square(0.0, 0.0, 1.0, true), square(0.5, 0.0, 1.0, true)]).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert!(view.contains(0.25, 0.0)); // overlap region still inside
        assert!(view.contains(-0.75, 0.0));
        assert!(view.contains(1.25, 0.0));
        assert!(!view.contains(1.75, 0.0));
    }

    #[test]
    fn winding_direction_does_not_matter_for_a_single_contour() {
        for ccw in [true, false] {
            let payload = build_payload(&[square(0.0, 0.0, 1.0, ccw)]).unwrap();
            let view = PayloadView::new(&payload).unwrap();
            assert!(view.contains(0.0, 0.0));
            assert!(!view.contains(1.5, 1.5));
        }
    }

    #[test]
    fn open_contours_are_implicitly_closed() {
        // A triangle given without repeating the first point.
        let payload = build_payload(&[vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]]).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        assert!(view.contains(1.0, 1.0));
        assert!(!view.contains(3.0, 3.0)); // beyond the hypotenuse
    }

    #[test]
    fn banding_covers_many_segments() {
        // Enough contours to force multiple bands; every square must still
        // classify correctly near its own y-range.
        let contours: Vec<_> = (0..40)
            .map(|i| square(0.0, i as f64 * 3.0, 1.0, true))
            .collect();
        let payload = build_payload(&contours).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        for i in 0..40 {
            let cy = i as f64 * 3.0;
            assert!(view.contains(0.0, cy), "center of square {i}");
            assert!(!view.contains(0.0, cy + 1.5), "gap above square {i}");
        }
    }

    #[test]
    fn band_boundary_exact_queries_read_inside() {
        // Regression: the query-side band index must be computed with the
        // exact FP expression the build side used, or a query at a
        // bit-exact band boundary lands one band low and misses the
        // crossing segments — a strictly interior point reading as
        // outside. These constants are a found divergence of the old
        // two-expression form: 30 rects -> 120 segments -> 15 bands over
        // y [0, 7502.8]; at k in {3, 6, 12, 13} the old query computed
        // band k-1 while the build assigned the bottom-edge segments
        // starting at band k.
        let rect =
            |x0: f64, y0: f64, y1: f64| vec![[x0, y0], [x0 + 10.0, y0], [x0 + 10.0, y1], [x0, y1]];
        let mut contours = vec![rect(0.0, 0.0, 100.0), rect(0.0, 7402.8, 7502.8)];
        for i in 0..24 {
            let y0 = 150.0 + i as f64 * 280.0;
            contours.push(rect(0.0, y0, y0 + 100.0));
        }
        // Matches build_payload's band choice for the final segment count;
        // if that heuristic changes, these ys stop being boundary-exact and
        // the test degrades to a plain interior check.
        let band_h = 7502.8 / ((30 * 4) / 8) as f64;
        let boundary_ys: Vec<f64> = [3.0, 6.0, 12.0, 13.0].iter().map(|k| k * band_h).collect();
        for &y in &boundary_ys {
            contours.push(rect(20.0, y, y + 200.0));
        }
        let payload = build_payload(&contours).unwrap();
        let view = PayloadView::new(&payload).unwrap();
        for &y in &boundary_ys {
            assert!(view.contains(25.0, y), "bottom edge on band boundary y={y}");
            assert!(
                view.contains(25.0, y + 100.0),
                "interior above boundary y={y}"
            );
        }
    }

    #[test]
    fn degenerate_inputs_are_rejected() {
        assert!(build_payload(&[]).is_err());
        assert!(build_payload(&[vec![]]).is_err());
        assert!(build_payload(&[vec![[0.0, 0.0], [1.0, 0.0]]]).is_err());
        // Collinear points enclose no area (zero-height bounds).
        assert!(build_payload(&[vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]).is_err());
        assert!(build_payload(&[vec![[0.0, 0.0], [f64::NAN, 1.0], [1.0, 1.0]]]).is_err());
        // Duplicate consecutive points collapse; a triangle must survive.
        assert!(build_payload(&[vec![[0.0, 0.0], [0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]]).is_ok());
    }

    #[test]
    fn corrupt_payloads_are_rejected_or_read_outside() {
        let payload = build_payload(&[square(0.0, 0.0, 1.0, true)]).unwrap();
        assert!(PayloadView::new(&payload[..20]).is_err());
        let mut bad = payload.clone();
        bad[0] ^= 0xFF;
        assert!(PayloadView::new(&bad).is_err());

        // Truncated to a valid header but missing tail bytes.
        let mut truncated = payload.clone();
        truncated.truncate(payload.len() - 8);
        assert!(PayloadView::new(&truncated).is_err());

        // A ref pointing past segment_count reads as outside, not a panic.
        let refs_base = HEADER_LEN + BAND_ENTRY_LEN; // 1 band for 4 segments
        let mut bad_ref = payload.clone();
        bad_ref[refs_base..refs_base + 4].copy_from_slice(&999u32.to_le_bytes());
        let view = PayloadView::new(&bad_ref).unwrap();
        assert!(!view.contains(0.0, 0.0));
    }
}
