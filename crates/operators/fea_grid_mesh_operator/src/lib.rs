//! FEA Grid Mesh Operator.
//!
//! Meshes a 3D model into a regular hex8 grid for finite-element work: a
//! uniform lattice is laid over the model's bounds and one hex element is
//! emitted per grid cell whose center samples inside the model. The output
//! is an explicit [`FeaMesh`] (CBOR), not a model WASM.
//!
//! This is the first operator that *evaluates* its model input instead of
//! rewriting it — occupancy comes through the host's `input_model_*`
//! sampling imports (see the `volumetric_abi` crate docs).
//!
//! Inputs:
//! - Input 0: ModelWASM (must be 3D)
//! - Input 1: CBOR configuration `{ resolution: int .default 16 }` — grid
//!   cells along the longest bounds axis (clamped to 2..=128; the other
//!   axes get the same cell size)
//!
//! Output 0: CBOR-encoded `FeaMesh` (hex8 elements, no fields).

use volumetric_abi::fea::{FeaElementKind, FeaMesh, encode_fea_mesh};
use volumetric_abi::host::{
    input_model_bounds, input_model_dimensions, input_model_sample, post_output, read_input,
    report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct GridMeshConfig {
    resolution: i64,
}

impl Default for GridMeshConfig {
    fn default() -> Self {
        Self { resolution: 16 }
    }
}

/// Corner offsets of a grid cell in Hex8 (VTK) node order.
const CELL_CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

fn build_mesh(config: &GridMeshConfig) -> Result<FeaMesh, String> {
    let dims =
        input_model_dimensions(0).ok_or_else(|| "input 0 is not a usable model".to_string())?;
    if dims != 3 {
        return Err(format!(
            "grid meshing requires a 3D model; input has {dims} dimensions"
        ));
    }
    let bounds =
        input_model_bounds(0, 3).ok_or_else(|| "failed to read model bounds".to_string())?;
    let origin = [bounds[0], bounds[2], bounds[4]];
    let extent = [
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ];
    let longest = extent.iter().fold(0.0f64, |a, &b| a.max(b));
    if !(longest > 0.0 && longest.is_finite()) {
        return Err(format!("model bounds are degenerate: {bounds:?}"));
    }

    let resolution = config.resolution.clamp(2, 128);
    let cell = longest / resolution as f64;
    // Cells per axis: cover the full extent (partial cells still sample
    // their centers; centers outside the solid just come back unoccupied).
    let cells: [usize; 3] = std::array::from_fn(|a| ((extent[a] / cell).ceil() as usize).max(1));
    let [nx, ny, nz] = cells;

    // Occupancy of every cell center, sampled one z-slab per host call to
    // bound the position buffer.
    let mut occupied = vec![false; nx * ny * nz];
    let mut positions = Vec::with_capacity(nx * ny * 3);
    for k in 0..nz {
        positions.clear();
        let z = origin[2] + (k as f64 + 0.5) * cell;
        for j in 0..ny {
            let y = origin[1] + (j as f64 + 0.5) * cell;
            for i in 0..nx {
                positions.extend([origin[0] + (i as f64 + 0.5) * cell, y, z]);
            }
        }
        let samples =
            input_model_sample(0, &positions, 3).ok_or_else(|| "sampling failed".to_string())?;
        for (idx, sample) in samples.iter().enumerate() {
            occupied[k * nx * ny + idx] = is_occupied(*sample);
        }
    }

    // Emit one hex per occupied cell; lattice nodes are created on first
    // use (dense id table indexed by lattice coordinate, u32::MAX = unused).
    let (mx, my, mz) = (nx + 1, ny + 1, nz + 1);
    let mut node_ids = vec![u32::MAX; mx * my * mz];
    let mut node_positions: Vec<f64> = Vec::new();
    let mut connectivity: Vec<u32> = Vec::new();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if !occupied[k * nx * ny + j * nx + i] {
                    continue;
                }
                for corner in &CELL_CORNERS {
                    let (ci, cj, ck) = (i + corner[0], j + corner[1], k + corner[2]);
                    let lattice_idx = ck * mx * my + cj * mx + ci;
                    if node_ids[lattice_idx] == u32::MAX {
                        node_ids[lattice_idx] = (node_positions.len() / 3) as u32;
                        node_positions.extend([
                            origin[0] + ci as f64 * cell,
                            origin[1] + cj as f64 * cell,
                            origin[2] + ck as f64 * cell,
                        ]);
                    }
                    connectivity.push(node_ids[lattice_idx]);
                }
            }
        }
    }

    let mesh = FeaMesh {
        element_kind: FeaElementKind::Hex8,
        node_positions,
        connectivity,
        node_fields: vec![],
        element_fields: vec![],
    };
    mesh.validate()?;
    Ok(mesh)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(1);
        if buf.is_empty() {
            GridMeshConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    match build_mesh(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("grid meshing failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "fea_grid_mesh_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "FEA Grid Mesh".to_string(),
        description: "Mesh a 3D model into a regular hex8 grid for finite-element work."
            .to_string(),
        category: "FEA".to_string(),
        icon_svg: String::new(),
        inputs: vec![
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::CBORConfiguration("{ resolution: int .default 16 }".to_string()),
        ],
        input_names: vec!["Domain model".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::FeaMesh],
    })
}
