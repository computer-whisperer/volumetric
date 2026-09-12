//! Cloud Distance Operator.
//!
//! The signed distance from every node of a point cloud to a model's
//! surface, as a node field plus a summary: the audit of a model against
//! the scan it was built from. See README.md (the operator's docs).
//!
//! Inputs:
//! - Input 0: FeaMesh — the cloud (any element kind; the nodes are used).
//! - Input 1: ModelWASM — the model to measure against (host-sampled).
//! - Input 2: CBOR configuration, see [`CloudDistanceConfig`].
//!
//! Output 0: FeaMesh — the cloud with the distance node field.
//! Output 1: F64Map — count, inside, percentiles of the absolute distance,
//! extremes, the lattice cell.

use cloud_core::distance::{DistanceGrid, DistanceStats};
use volumetric_abi::f64_map::F64Map;
use volumetric_abi::fea::{FeaField, FeaMesh};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::fea::{decode_fea_mesh, encode_fea_mesh};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{
    input_model_bounds, input_model_dimensions, input_model_sample, post_output, read_input,
    report_error,
};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

pub const DEFAULT_FIELD: &str = "distance";

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudDistanceConfig {
    /// Lattice cells along the longest side of the measured box.
    #[serde(default = "default_resolution")]
    pub resolution: u32,
    /// The node field to write.
    #[serde(default = "default_field")]
    pub field: String,
    /// The summary's tolerance: the fractions within it, twice it and
    /// four times it are reported.
    #[serde(default = "default_band")]
    pub band: f64,
}

fn default_band() -> f64 {
    0.005
}

fn default_resolution() -> u32 {
    128
}

fn default_field() -> String {
    DEFAULT_FIELD.to_string()
}

impl Default for CloudDistanceConfig {
    fn default() -> Self {
        Self {
            resolution: default_resolution(),
            field: default_field(),
            band: default_band(),
        }
    }
}

/// The lattice to bake: origin (first cell centre), spacing and dims that
/// cover `bounds` (min xyz, max xyz) with `resolution` cells along the
/// longest side and a two-cell pad.
pub fn lattice(bounds: ([f64; 3], [f64; 3]), resolution: u32) -> ([f64; 3], f64, [usize; 3]) {
    let (min, max) = bounds;
    let longest = (0..3).map(|a| max[a] - min[a]).fold(0.0, f64::max);
    let spacing = (longest / f64::from(resolution.max(1))).max(1e-9);
    let mut origin = [0.0; 3];
    let mut dims = [0usize; 3];
    for a in 0..3 {
        origin[a] = min[a] - 1.5 * spacing;
        dims[a] = (((max[a] - min[a]) / spacing).ceil() as usize + 4).max(2);
    }
    (origin, spacing, dims)
}

/// The union of the cloud's node bounds and `model` bounds (interleaved
/// min/max pairs), or the cloud's alone.
pub fn measured_bounds(
    mesh: &FeaMesh,
    model: Option<&[f64]>,
) -> Result<([f64; 3], [f64; 3]), String> {
    if mesh.node_count() == 0 {
        return Err("the cloud has no nodes".to_string());
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for i in 0..mesh.node_count() {
        let p = mesh.node_position(i);
        for a in 0..3 {
            min[a] = min[a].min(p[a]);
            max[a] = max[a].max(p[a]);
        }
    }
    if let Some(b) = model {
        for a in 0..3 {
            if b[2 * a] <= b[2 * a + 1] {
                min[a] = min[a].min(b[2 * a]);
                max[a] = max[a].max(b[2 * a + 1]);
            }
        }
    }
    Ok((min, max))
}

/// Writes the distances as the node field and summarises them.
pub fn apply(mesh: &mut FeaMesh, grid: &DistanceGrid, field: &str) -> (Vec<f64>, DistanceStats) {
    let distances: Vec<f64> = (0..mesh.node_count())
        .map(|i| grid.sample(mesh.node_position(i)))
        .collect();
    mesh.node_fields.retain(|f| f.name != field);
    mesh.node_fields.push(FeaField {
        name: field.to_string(),
        components: 1,
        data: distances.clone(),
    });
    let stats = DistanceStats::of(&distances);
    (distances, stats)
}

/// The summary as the F64Map output.
pub fn summary(stats: &DistanceStats, distances: &[f64], band: f64, cell: f64) -> F64Map {
    let mut map = F64Map::new();
    map.insert("band".to_string(), band);
    map.insert(
        "within_band".to_string(),
        DistanceStats::within(distances, band),
    );
    map.insert(
        "within_2band".to_string(),
        DistanceStats::within(distances, 2.0 * band),
    );
    map.insert(
        "within_4band".to_string(),
        DistanceStats::within(distances, 4.0 * band),
    );
    map.insert("count".to_string(), stats.count as f64);
    map.insert("inside".to_string(), stats.inside as f64);
    map.insert(
        "inside_fraction".to_string(),
        if stats.count > 0 {
            stats.inside as f64 / stats.count as f64
        } else {
            0.0
        },
    );
    map.insert("mean".to_string(), stats.mean);
    map.insert("abs_p50".to_string(), stats.abs_p50);
    map.insert("abs_p90".to_string(), stats.abs_p90);
    map.insert("abs_p99".to_string(), stats.abs_p99);
    map.insert("max_outside".to_string(), stats.max_outside);
    map.insert("max_inside".to_string(), stats.max_inside);
    map.insert("cell".to_string(), cell);
    map
}

/// Occupancy samples per batched host call.
#[cfg(target_arch = "wasm32")]
const BATCH: usize = 65_536;

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let mut mesh = match decode_fea_mesh(&read_input(0)) {
        Ok(mesh) => mesh,
        Err(e) => {
            report_error(&format!("input 0 is not a usable mesh: {e}"));
            return;
        }
    };
    let config: CloudDistanceConfig = {
        let cfg = read_input(2);
        if cfg.is_empty() {
            CloudDistanceConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&cfg)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    if !(16..=256).contains(&config.resolution) {
        report_error(&format!(
            "resolution must be 16..=256, got {}",
            config.resolution
        ));
        return;
    }
    match input_model_dimensions(1) {
        Some(3) => {}
        Some(n) => {
            report_error(&format!("the model must be 3D, got {n} dimensions"));
            return;
        }
        None => {
            report_error("input 1 is not a model");
            return;
        }
    }
    let model_bounds = input_model_bounds(1, 3);
    let bounds = match measured_bounds(&mesh, model_bounds.as_deref()) {
        Ok(b) => b,
        Err(e) => {
            report_error(&e);
            return;
        }
    };
    let (origin, spacing, dims) = lattice(bounds, config.resolution);
    let n = dims[0] * dims[1] * dims[2];
    let mut occupied = Vec::with_capacity(n);
    let mut positions = Vec::with_capacity(3 * BATCH);
    let flush = |positions: &mut Vec<f64>, occupied: &mut Vec<bool>| -> Result<(), String> {
        if positions.is_empty() {
            return Ok(());
        }
        let samples = input_model_sample(1, positions, 3)
            .ok_or_else(|| "sampling the model failed".to_string())?;
        occupied.extend(samples.iter().map(|&s| volumetric_abi::is_occupied(s)));
        positions.clear();
        Ok(())
    };
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                positions.extend_from_slice(&[
                    origin[0] + x as f64 * spacing,
                    origin[1] + y as f64 * spacing,
                    origin[2] + z as f64 * spacing,
                ]);
                if positions.len() >= 3 * BATCH
                    && let Err(e) = flush(&mut positions, &mut occupied)
                {
                    report_error(&e);
                    return;
                }
            }
        }
    }
    if let Err(e) = flush(&mut positions, &mut occupied) {
        report_error(&e);
        return;
    }
    let grid = DistanceGrid::bake(origin, spacing, dims, &occupied);
    if config.band <= 0.0 || !config.band.is_finite() {
        report_error("band must be positive");
        return;
    }
    let (distances, stats) = apply(&mut mesh, &grid, &config.field);
    post_output(0, &encode_fea_mesh(&mesh));
    match volumetric_abi::f64_map::encode(&summary(&stats, &distances, config.band, spacing)) {
        Ok(bytes) => post_output(1, &bytes),
        Err(e) => report_error(&format!("summary failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            r#"{ resolution: int .ge 16 .le 256 .default 128, field: tstr .default "distance", band: float .gt 0.0 .default 0.005 }"#
                .to_string();
        OperatorMetadata {
            name: "cloud_distance_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Cloud Distance".to_string(),
            description: "Measure a point cloud against a model: the signed distance of every node to the surface as a field, with a summary."
                .to_string(),
            category: "Analysis".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 17c4-6 10-8 18-6"/>"##,
                r##"<circle cx="7" cy="8" r="1"/>"##,
                r##"<circle cx="13" cy="6" r="1"/>"##,
                r##"<circle cx="18" cy="13" r="1"/>"##,
                r##"<path d="M7 9v4M13 7v3M18 12v-1"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["Cloud".to_string(), "Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::FeaMesh, OperatorMetadataOutput::F64Map],
            output_names: vec!["Measured".to_string(), "Summary".to_string()],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::fea::FeaElementKind;

    #[test]
    fn lattice_covers_the_box_with_a_pad() {
        let (origin, spacing, dims) = lattice(([0.0, 0.0, 0.0], [1.0, 0.5, 0.25]), 100);
        assert!((spacing - 0.01).abs() < 1e-12);
        assert_eq!(dims, [104, 54, 29]);
        assert!(origin[0] < 0.0 && origin[0] + (dims[0] - 1) as f64 * spacing > 1.0);
    }

    #[test]
    fn nodes_read_the_field_and_the_summary_counts_them() {
        // A unit cube [0,1]^3 baked at 20 cells; nodes inside and outside.
        let (origin, spacing, dims) = lattice(([0.0; 3], [1.0; 3]), 20);
        let occupied: Vec<bool> = (0..dims[0] * dims[1] * dims[2])
            .map(|i| {
                let (x, y, z) = (
                    i % dims[0],
                    (i / dims[0]) % dims[1],
                    i / (dims[0] * dims[1]),
                );
                [x, y, z].iter().zip(&origin).all(|(&c, &o)| {
                    let w = o + c as f64 * spacing;
                    (0.0..=1.0).contains(&w)
                })
            })
            .collect();
        let grid = DistanceGrid::bake(origin, spacing, dims, &occupied);
        let mut mesh = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: vec![0.5, 0.5, 0.5, 1.2, 0.5, 0.5, 0.5, 0.5, 0.9],
            connectivity: vec![0, 1, 2],
            node_fields: vec![],
            element_fields: vec![],
        };
        let (distances, stats) = apply(&mut mesh, &grid, "distance");
        assert!(distances[0] < -0.4, "centre {}", distances[0]);
        assert!(
            (distances[1] - 0.2).abs() < spacing,
            "outside {}",
            distances[1]
        );
        assert!(
            (distances[2] + 0.1).abs() < spacing,
            "near the face {}",
            distances[2]
        );
        assert_eq!(stats.count, 3);
        assert_eq!(stats.inside, 2);
        assert_eq!(mesh.node_fields[0].name, "distance");
        let map = summary(&stats, &distances, 0.15, spacing);
        assert_eq!(map["count"], 3.0);
        assert!((map["within_band"] - 1.0 / 3.0).abs() < 1e-12);
        assert!((map["within_2band"] - 2.0 / 3.0).abs() < 1e-12);
        assert!((map["inside_fraction"] - 2.0 / 3.0).abs() < 1e-12);
    }
}
