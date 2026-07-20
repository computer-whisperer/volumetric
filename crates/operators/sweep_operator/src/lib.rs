//! Sweep Operator: extrude a model's occupancy along an axis.
//!
//! Every point within `distance` downstream of solid material (along
//! the signed sweep direction) becomes solid — the union of the source
//! with its translates along the axis, i.e. a directional sweep or
//! shadow. With `until` set, added material stops at that plane
//! coordinate (source material past the plane is preserved, never
//! removed). The canonical use is mold-half or clamshell cavities:
//! sweeping a clearance-dilated part toward the parting side makes the
//! cavity monotone along the demolding axis, so enclosed voids (socket
//! interiors, gaps between components) open up instead of filling with
//! housing material.
//!
//! Implementation mirrors `offset_operator`: the source occupancy is
//! sampled onto a lattice (`ndfield_model_core::bake`), a running-OR
//! scan applies the sweep per axis column, and the exact distance
//! transform of the swept occupancy is baked so the emitted standalone
//! field model gets sub-cell surface placement. Accuracy is roughly a
//! lattice cell (longest swept-domain axis / `resolution`).
//!
//! Inputs:
//! - Input 0: ModelWASM — sampled via host imports; the bytes are unused
//! - Input 1: CBOR config:
//!   - `axis` ("x"/"y"/"z", default "z"): sweep axis
//!   - `distance` (metres, default 0.01): sweep length; the sign is the
//!     sweep direction
//!   - `until` (optional, metres): plane coordinate clamping added
//!     material along the sweep direction
//!   - `resolution` (default 64): lattice cells along the swept
//!     domain's longest axis
//!
//! Output 0: ModelWASM (input dimensionality).
//!
//! The embedded template binary is `sdf_model_template` (the same file
//! `sdf_operator` and `offset_operator` embed), regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p sdf_model_template
//! cp target/wasm32-unknown-unknown/release/sdf_model_template.wasm \
//!    crates/operators/sweep_operator/template/
//! ```

use ndfield_model_core::bake::{FieldGrid, bake_tsdf, sample_occupancy};
use volumetric_abi::host::{
    cancelled, input_model_bounds, input_model_dimensions, input_model_sample, post_output,
    read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/sdf_model_template.wasm");

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
pub struct SweepConfig {
    /// Sweep axis.
    pub axis: Axis,
    /// Sweep length in metres; the sign is the sweep direction.
    pub distance: f64,
    /// Optional plane coordinate clamping added material.
    pub until: Option<f64>,
    /// Lattice cells along the swept domain's longest axis.
    pub resolution: i64,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            axis: Axis::Z,
            distance: 0.01,
            until: None,
            resolution: 64,
        }
    }
}

/// The running-OR sweep pass: mark every cell within `window` cells
/// downstream of an occupied cell, unless its coordinate lies past the
/// clamp plane. Source occupancy is always preserved.
fn sweep_columns(
    occupancy: &mut [bool],
    grid: &FieldGrid,
    axis: usize,
    positive: bool,
    window: usize,
    until: Option<f64>,
) {
    let stride: usize = grid.counts[..axis].iter().product();
    let line_length = grid.counts[axis];
    let block_length = stride * line_length;
    let origin = grid.bounds[2 * axis];
    let spacing = grid.spacing[axis];
    let mut column = vec![false; line_length];
    for block in (0..grid.points).step_by(block_length) {
        for inner in 0..stride {
            for position in 0..line_length {
                column[position] = occupancy[block + inner + position * stride];
            }
            // Walk downstream; `logical` counts cells in sweep order.
            let mut last_solid: Option<usize> = None;
            for logical in 0..line_length {
                let position = if positive {
                    logical
                } else {
                    line_length - 1 - logical
                };
                if column[position] {
                    last_solid = Some(logical);
                    continue;
                }
                let Some(last) = last_solid else { continue };
                if logical - last > window {
                    continue;
                }
                let coordinate = origin + position as f64 * spacing;
                let clamped = match until {
                    Some(plane) if positive => coordinate > plane,
                    Some(plane) => coordinate < plane,
                    None => false,
                };
                if !clamped {
                    occupancy[block + inner + position * stride] = true;
                }
            }
        }
    }
}

/// Bake the swept field. Returns the `ndfield` payload plus the output
/// model's advertised bounds (the swept domain padded by one nominal
/// cell of interpolation slack).
pub fn build_sweep_payload<F>(
    source_bounds: &[f64],
    config: &SweepConfig,
    sample: F,
) -> Result<(Vec<u8>, Vec<f64>), String>
where
    F: FnMut(&[f64]) -> Result<Vec<bool>, String>,
{
    if !(config.distance.is_finite() && config.distance != 0.0) {
        return Err(format!(
            "distance must be finite and nonzero, got {}",
            config.distance
        ));
    }
    if let Some(until) = config.until
        && !until.is_finite()
    {
        return Err(format!("until must be finite, got {until}"));
    }
    let dimensions = source_bounds.len() / 2;
    let axis = config.axis.index();
    if axis >= dimensions {
        return Err(format!(
            "sweep axis {:?} needs at least {} dimensions; input has {dimensions}",
            config.axis,
            axis + 1
        ));
    }

    // Extend the domain along the sweep direction by the reach of added
    // material: the sweep length, clamped by `until` when it cuts in
    // earlier. The domain never shrinks below the source bounds.
    let positive = config.distance > 0.0;
    let reach = config.distance.abs();
    let mut domain = source_bounds.to_vec();
    if positive {
        let mut hi = source_bounds[2 * axis + 1] + reach;
        if let Some(plane) = config.until {
            hi = hi.min(plane);
        }
        domain[2 * axis + 1] = domain[2 * axis + 1].max(hi);
    } else {
        let mut lo = source_bounds[2 * axis] - reach;
        if let Some(plane) = config.until {
            lo = lo.max(plane);
        }
        domain[2 * axis] = domain[2 * axis].min(lo);
    }

    let grid = FieldGrid::plan(&domain, config.resolution, 0.0)?;
    let mut occupancy = sample_occupancy(&grid, sample)?;
    let window = (reach / grid.spacing[axis]).floor() as usize;
    sweep_columns(&mut occupancy, &grid, axis, positive, window, config.until);
    let tsdf = bake_tsdf(&grid, &occupancy)?;
    let values: Vec<f32> = tsdf.iter().map(|&value| -value).collect();
    let outside = -(grid.band_width as f32);
    let payload = ndfield_model_core::build_payload(&grid.counts, &grid.bounds, &values, outside)?;

    let cell = ndfield_model_core::bake::nominal_cell(&domain, config.resolution)?;
    let mut out_bounds = Vec::with_capacity(2 * dimensions);
    for axis in 0..dimensions {
        out_bounds.push(domain[2 * axis] - cell);
        out_bounds.push(domain[2 * axis + 1] + cell);
    }
    Ok((payload, out_bounds))
}

/// Patch the payload into the template as a standalone model (shared
/// emission path; see `ndfield_model_core::emit`).
pub fn emit_model(
    payload: &[u8],
    dimensions: usize,
    out_bounds: &[f64],
) -> Result<Vec<u8>, String> {
    ndfield_model_core::emit::emit_field_model(TEMPLATE, payload, dimensions, out_bounds)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config: SweepConfig = {
        let bytes = read_input(1);
        if bytes.is_empty() {
            SweepConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(bytes)) {
                Ok(config) => config,
                Err(error) => {
                    report_error(&format!("invalid sweep configuration: {error}"));
                    return;
                }
            }
        }
    };
    let result = (|| {
        let dimensions = input_model_dimensions(0)
            .ok_or_else(|| "input 0 is not a usable model".to_string())?
            as usize;
        let source_bounds = input_model_bounds(0, dimensions)
            .ok_or_else(|| "failed to read input model bounds".to_string())?;
        let (payload, out_bounds) = build_sweep_payload(&source_bounds, &config, |positions| {
            if cancelled() {
                return Err("sweep cancelled".to_string());
            }
            input_model_sample(0, positions, dimensions)
                .map(|values| values.into_iter().map(is_occupied).collect())
                .ok_or_else(|| "input model sampling failed".to_string())
        })?;
        emit_model(&payload, dimensions, &out_bounds)
    })();
    match result {
        Ok(output) => post_output(0, &output),
        Err(error) => report_error(&format!("sweep failed: {error}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        OperatorMetadata {
        name: "sweep_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Sweep".to_string(),
        description:
            "Extrude occupancy along an axis (directional sweep/shadow), optionally clamped at a plane."
                .to_string(),
        category: "Transforms".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<rect x="4" y="14" width="16" height="6" rx="1"/>"##,
            r##"<path d="M12 14V4"/>"##,
            r##"<path d="M8 8l4-4 4 4"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::CBORConfiguration(
                r#"{ axis: "x" / "y" / "z" .default "z", distance: float .default 0.01, ? until: float, resolution: int .ge 16 .le 256 .default 64 }"#
                    .to_string(),
            ),
        ],
        input_names: vec!["Model".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndfield_model_core::PayloadView;

    /// Occupancy sampler for the box [-2,2] x [-2,2] x [0,1].
    fn box_sampler() -> impl FnMut(&[f64]) -> Result<Vec<bool>, String> {
        move |positions: &[f64]| {
            Ok(positions
                .chunks_exact(3)
                .map(|p| p[0].abs() <= 2.0 && p[1].abs() <= 2.0 && (0.0..=1.0).contains(&p[2]))
                .collect())
        }
    }

    const BOUNDS: [f64; 6] = [-3.0, 3.0, -3.0, 3.0, -1.0, 2.0];

    fn field(config: &SweepConfig) -> (Vec<u8>, Vec<f64>) {
        build_sweep_payload(&BOUNDS, config, box_sampler()).unwrap()
    }

    #[test]
    fn clamped_upward_sweep_fills_to_the_plane() {
        let config = SweepConfig {
            axis: Axis::Z,
            distance: 100.0,
            until: Some(4.5),
            resolution: 64,
        };
        let (payload, out_bounds) = field(&config);
        let view = PayloadView::new(&payload).unwrap();
        // Domain extends to the clamp plane, padded by a cell.
        assert!(out_bounds[5] > 4.5 && out_bounds[5] < 4.8, "{out_bounds:?}");
        for (p, inside) in [
            ([0.0, 0.0, 0.5], true),   // original box
            ([0.0, 0.0, 2.0], true),   // swept, above the box
            ([0.0, 0.0, 4.3], true),   // swept, just under the plane
            ([0.0, 0.0, 4.8], false),  // past the clamp plane
            ([2.5, 0.0, 0.5], false),  // beside the footprint
            ([0.0, 0.0, -0.5], false), // upstream of the box
        ] {
            assert_eq!(view.sample(&p) > 0.0, inside, "{p:?}");
        }
    }

    #[test]
    fn bounded_downward_sweep_respects_the_window() {
        let config = SweepConfig {
            axis: Axis::Z,
            distance: -1.0,
            until: None,
            resolution: 64,
        };
        let (payload, _) = field(&config);
        let view = PayloadView::new(&payload).unwrap();
        for (p, inside) in [
            ([0.0, 0.0, 0.5], true),   // original box
            ([0.0, 0.0, -0.5], true),  // swept one unit down
            ([0.0, 0.0, -1.8], false), // beyond the sweep length
            ([0.0, 0.0, 1.5], false),  // downstream side untouched
        ] {
            assert_eq!(view.sample(&p) > 0.0, inside, "{p:?}");
        }
    }

    #[test]
    fn template_surgery_emits_the_model_abi() {
        let config = SweepConfig {
            axis: Axis::Z,
            distance: 1.0,
            until: None,
            resolution: 16,
        };
        let (payload, out_bounds) = field(&config);
        let wasm = emit_model(&payload, 3, &out_bounds).unwrap();
        let module = walrus::Module::from_buffer(&wasm).expect("emitted wasm parses");
        let names: Vec<&str> = module.exports.iter().map(|e| e.name.as_str()).collect();
        for required in [
            "sample",
            "get_bounds",
            "get_dimensions",
            "get_io_ptr",
            "memory",
        ] {
            assert!(names.contains(&required), "missing export {required}");
        }
        for dropped in ["sdf_payload_slot", "sdf_sample", "get_sample_format"] {
            assert!(!names.contains(&dropped), "stale export {dropped}");
        }
    }
}
