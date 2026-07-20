//! Shared lattice-bake machinery for operators that sample a model's
//! occupancy onto a regular grid and derive a truncated signed distance
//! field from it (`sdf_operator`, `offset_operator`). The payload write
//! side stays in the crate root ([`crate::build_payload`]); this module
//! owns grid planning, batched occupancy sampling, and the exact
//! separable Euclidean distance transform.

pub const MIN_RESOLUTION: i64 = 16;
pub const MAX_RESOLUTION: i64 = 256;
pub const DEFAULT_BAND_CELLS: f64 = 4.0;
const MAX_GRID_POINTS: usize = 8_000_000;
const SAMPLE_BATCH_POINTS: usize = 8_192;

/// The nominal cell edge a lattice at `resolution` would use: the
/// source's longest axis divided by the resolution. Callers that derive
/// a band width from cell counts need it before planning the grid.
pub fn nominal_cell(source_bounds: &[f64], resolution: i64) -> Result<f64, String> {
    if source_bounds.is_empty() || !source_bounds.len().is_multiple_of(2) {
        return Err(format!("invalid source bounds {source_bounds:?}"));
    }
    if !(MIN_RESOLUTION..=MAX_RESOLUTION).contains(&resolution) {
        return Err(format!(
            "resolution must be in {MIN_RESOLUTION}..={MAX_RESOLUTION}, got {resolution}"
        ));
    }
    let mut longest = 0.0f64;
    for axis in 0..source_bounds.len() / 2 {
        let (lo, hi) = (source_bounds[2 * axis], source_bounds[2 * axis + 1]);
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(format!(
                "source axis {axis} needs finite nonempty bounds, got [{lo}, {hi}]"
            ));
        }
        longest = longest.max(hi - lo);
    }
    Ok(longest / resolution as f64)
}

#[derive(Clone, Debug)]
pub struct FieldGrid {
    pub counts: Vec<usize>,
    pub bounds: Vec<f64>,
    pub spacing: Vec<f64>,
    pub band_width: f64,
    pub points: usize,
}

impl FieldGrid {
    /// Plan a lattice covering `source_bounds` plus `band_width` on every
    /// side at roughly `resolution` cells along the longest source axis.
    /// A zero `band_width` selects [`DEFAULT_BAND_CELLS`] nominal cells.
    pub fn plan(source_bounds: &[f64], resolution: i64, band_width: f64) -> Result<Self, String> {
        let dimensions = source_bounds.len() / 2;
        if !(1..=crate::MAX_DIMS).contains(&dimensions) {
            return Err(format!(
                "field baking supports 1..={} dimensions; input has {dimensions}",
                crate::MAX_DIMS
            ));
        }
        if !(band_width.is_finite() && band_width >= 0.0) {
            return Err(format!(
                "band_width must be finite and non-negative, got {band_width}"
            ));
        }
        let nominal_cell = nominal_cell(source_bounds, resolution)?;
        let band_width = if band_width == 0.0 {
            DEFAULT_BAND_CELLS * nominal_cell
        } else {
            band_width
        };
        if !(band_width.is_finite() && band_width > 0.0 && band_width <= f32::MAX as f64) {
            return Err(format!(
                "resolved band width must be finite, positive, and fit f32; got {band_width}"
            ));
        }

        let mut counts = Vec::with_capacity(dimensions);
        let mut bounds = Vec::with_capacity(2 * dimensions);
        let mut spacing = Vec::with_capacity(dimensions);
        let mut points = 1usize;
        for axis in 0..dimensions {
            let lo = source_bounds[2 * axis] - band_width;
            let hi = source_bounds[2 * axis + 1] + band_width;
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(format!(
                    "expanded field bounds overflow on axis {axis}: [{lo}, {hi}]"
                ));
            }
            let count = (((hi - lo) / nominal_cell).ceil() as usize)
                .checked_add(1)
                .ok_or_else(|| "field axis count overflows usize".to_string())?
                .max(2);
            points = points
                .checked_mul(count)
                .ok_or_else(|| "field lattice size overflows usize".to_string())?;
            if points > MAX_GRID_POINTS {
                return Err(format!(
                    "field lattice would contain {points} points (limit {MAX_GRID_POINTS}); \
                     lower resolution or band_width"
                ));
            }
            counts.push(count);
            bounds.extend([lo, hi]);
            spacing.push((hi - lo) / (count - 1) as f64);
        }
        Ok(Self {
            counts,
            bounds,
            spacing,
            band_width,
            points,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.counts.len()
    }

    fn append_position(&self, mut index: usize, positions: &mut Vec<f64>) {
        for axis in 0..self.dimensions() {
            let coordinate = index % self.counts[axis];
            index /= self.counts[axis];
            positions.push(self.bounds[2 * axis] + coordinate as f64 * self.spacing[axis]);
        }
    }
}

/// Lower envelope of weighted parabolas:
/// `out[q] = min_p(input[p] + spacing² * (q-p)²)`.
fn distance_transform_1d(input: &[f64], spacing: f64, output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    let finite_sites: Vec<usize> = input
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect();
    if finite_sites.is_empty() {
        output.fill(f64::INFINITY);
        return;
    }

    let spacing2 = spacing * spacing;
    let mut sites = vec![0usize; finite_sites.len()];
    let mut boundaries = vec![0.0f64; finite_sites.len() + 1];
    let mut envelope = 0usize;
    sites[0] = finite_sites[0];
    boundaries[0] = f64::NEG_INFINITY;
    boundaries[1] = f64::INFINITY;

    for &candidate in &finite_sites[1..] {
        let intersection = |left: usize| {
            let candidate_x = candidate as f64;
            let left_x = left as f64;
            (input[candidate] + spacing2 * candidate_x * candidate_x
                - input[left]
                - spacing2 * left_x * left_x)
                / (2.0 * spacing2 * (candidate_x - left_x))
        };
        let mut crossing = intersection(sites[envelope]);
        while crossing <= boundaries[envelope] {
            if envelope == 0 {
                break;
            }
            envelope -= 1;
            crossing = intersection(sites[envelope]);
        }
        if envelope == 0 && crossing <= boundaries[0] {
            sites[0] = candidate;
            boundaries[1] = f64::INFINITY;
            continue;
        }
        envelope += 1;
        sites[envelope] = candidate;
        boundaries[envelope] = crossing;
        boundaries[envelope + 1] = f64::INFINITY;
    }

    let last = envelope;
    envelope = 0;
    for (query, value) in output.iter_mut().enumerate() {
        while envelope < last && boundaries[envelope + 1] < query as f64 {
            envelope += 1;
        }
        let site = sites[envelope];
        let delta = query as f64 - site as f64;
        *value = input[site] + spacing2 * delta * delta;
    }
}

/// Exact squared Euclidean distance, on the grid lattice, to every point
/// whose occupancy equals `target`.
fn squared_distance_transform(occupancy: &[bool], target: bool, grid: &FieldGrid) -> Vec<f64> {
    let mut distances: Vec<f64> = occupancy
        .iter()
        .map(|&state| if state == target { 0.0 } else { f64::INFINITY })
        .collect();

    for axis in 0..grid.dimensions() {
        let stride: usize = grid.counts[..axis].iter().product();
        let line_length = grid.counts[axis];
        let block_length = stride * line_length;
        let mut input = vec![0.0; line_length];
        let mut output = vec![0.0; line_length];
        for block in (0..grid.points).step_by(block_length) {
            for inner in 0..stride {
                for position in 0..line_length {
                    input[position] = distances[block + inner + position * stride];
                }
                distance_transform_1d(&input, grid.spacing[axis], &mut output);
                for position in 0..line_length {
                    distances[block + inner + position * stride] = output[position];
                }
            }
        }
    }
    distances
}

/// Classify every lattice point through `sample`, batched. The callback
/// receives concatenated positions (`dimensions` f64s per point) and
/// returns one occupancy verdict per point.
pub fn sample_occupancy<F>(grid: &FieldGrid, mut sample: F) -> Result<Vec<bool>, String>
where
    F: FnMut(&[f64]) -> Result<Vec<bool>, String>,
{
    let dimensions = grid.dimensions();
    let mut occupancy = Vec::with_capacity(grid.points);
    for start in (0..grid.points).step_by(SAMPLE_BATCH_POINTS) {
        let end = (start + SAMPLE_BATCH_POINTS).min(grid.points);
        let mut positions = Vec::with_capacity((end - start) * dimensions);
        for index in start..end {
            grid.append_position(index, &mut positions);
        }
        let samples = sample(&positions)?;
        if samples.len() != end - start {
            return Err(format!(
                "model sampler returned {} values for {} positions",
                samples.len(),
                end - start
            ));
        }
        occupancy.extend(samples);
    }
    Ok(occupancy)
}

/// Truncated signed distances (negative inside, positive outside, clamped
/// to the grid's band) for a lattice occupancy classification.
pub fn bake_tsdf(grid: &FieldGrid, occupancy: &[bool]) -> Result<Vec<f32>, String> {
    if occupancy.len() != grid.points {
        return Err(format!(
            "occupancy has {} entries for a {}-point grid",
            occupancy.len(),
            grid.points
        ));
    }
    if occupancy.iter().all(|state| *state) {
        return Err(
            "model remains occupied throughout its bounds plus the distance band; advertised bounds may not enclose the geometry"
                .to_string(),
        );
    }

    // The interface lies between unlike lattice samples. Subtracting half
    // the smallest cell from the opposite-state lattice distance puts the
    // zero crossing midway across an axis-adjacent transition and bounds the
    // sub-cell surface error for oblique transitions.
    let interface_offset = 0.5 * grid.spacing.iter().copied().fold(f64::INFINITY, f64::min);
    let mut values = vec![0.0f32; grid.points];

    let to_outside = squared_distance_transform(occupancy, false, grid);
    for index in 0..grid.points {
        if occupancy[index] {
            let magnitude = (to_outside[index].sqrt() - interface_offset)
                .max(0.0)
                .min(grid.band_width);
            values[index] = -(magnitude as f32);
        }
    }
    drop(to_outside);

    let to_inside = squared_distance_transform(occupancy, true, grid);
    for index in 0..grid.points {
        if !occupancy[index] {
            let magnitude = if to_inside[index].is_finite() {
                (to_inside[index].sqrt() - interface_offset)
                    .max(0.0)
                    .min(grid.band_width)
            } else {
                grid.band_width
            };
            values[index] = magnitude as f32;
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_distance_transform_is_exact_on_a_line() {
        let input = [f64::INFINITY, f64::INFINITY, 0.0, f64::INFINITY];
        let mut output = [0.0; 4];
        distance_transform_1d(&input, 0.25, &mut output);
        assert_eq!(output, [0.25, 0.0625, 0.0, 0.0625]);
    }

    #[test]
    fn weighted_transform_matches_brute_force_with_prior_axis_costs() {
        let input = [0.7, f64::INFINITY, 0.2, 1.3, f64::INFINITY, 0.0];
        let spacing = 0.37;
        let mut output = [0.0; 6];
        distance_transform_1d(&input, spacing, &mut output);
        for (query, &actual) in output.iter().enumerate() {
            let expected = input
                .iter()
                .enumerate()
                .map(|(site, &prior)| {
                    prior + spacing * spacing * (query as f64 - site as f64).powi(2)
                })
                .fold(f64::INFINITY, f64::min);
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn planning_rejects_unbounded_work() {
        let bounds = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let error = FieldGrid::plan(&bounds, 256, 1.0).unwrap_err();
        assert!(error.contains("limit"), "{error}");
    }
}
