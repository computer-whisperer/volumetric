use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const SAMPLE_CLOUD_VERSION: u32 = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleCloudDump {
    pub version: u32,
    pub sets: Vec<SampleCloudSet>,
}

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl BBox {
    /// Create a bounding box centered on a point with given half-extents.
    pub fn centered(center: [f32; 3], half_extent: f32) -> Self {
        Self {
            min: [
                center[0] - half_extent,
                center[1] - half_extent,
                center[2] - half_extent,
            ],
            max: [
                center[0] + half_extent,
                center[1] + half_extent,
                center[2] + half_extent,
            ],
        }
    }
}

impl SampleCloudDump {
    pub fn new() -> Self {
        Self {
            version: SAMPLE_CLOUD_VERSION,
            sets: Vec::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path).with_context(|| format!("Failed to read sample cloud: {}", path.display()))?;
        let dump: SampleCloudDump =
            serde_cbor::from_slice(&data).context("Failed to decode sample cloud CBOR")?;
        Ok(dump)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_cbor::to_vec(self).context("Failed to encode sample cloud CBOR")?;
        fs::write(path, data).with_context(|| format!("Failed to write sample cloud: {}", path.display()))
    }

    pub fn add_set(&mut self, set: SampleCloudSet) {
        self.sets.push(set);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleCloudSet {
    pub id: u64,
    pub label: Option<String>,
    pub vertex: [f32; 3],
    /// Bounding box of the sampling cell (centered on vertex).
    #[serde(default)]
    pub cell_bounds: Option<BBox>,
    pub points: Vec<SamplePoint>,
    pub meta: SampleCloudMeta,
    /// Named vectors for visualization (normals, directions, etc.)
    #[serde(default)]
    pub vectors: Vec<NamedVector>,
    /// Colored triangles for rendering fitted geometry.
    #[serde(default)]
    pub triangles: Vec<ColoredTriangle>,
}

impl SampleCloudSet {
    pub fn new(id: u64, vertex: [f32; 3]) -> Self {
        Self {
            id,
            label: None,
            vertex,
            cell_bounds: None,
            points: Vec::new(),
            meta: SampleCloudMeta::default(),
            vectors: Vec::new(),
            triangles: Vec::new(),
        }
    }

    /// Create a new sample cloud set with cell bounds.
    pub fn with_cell_size(id: u64, vertex: [f32; 3], cell_size: f32) -> Self {
        Self {
            id,
            label: None,
            vertex,
            cell_bounds: Some(BBox::centered(vertex, cell_size * 0.5)),
            points: Vec::new(),
            meta: SampleCloudMeta::default(),
            vectors: Vec::new(),
            triangles: Vec::new(),
        }
    }

    /// Add a named vector for visualization.
    pub fn add_vector(&mut self, name: impl Into<String>, origin: [f32; 3], direction: [f32; 3], color: Option<[f32; 4]>) {
        self.vectors.push(NamedVector {
            name: name.into(),
            origin,
            direction,
            color,
        });
    }

    /// Add a colored triangle for fitted geometry visualization.
    pub fn add_triangle(&mut self, vertices: [[f32; 3]; 3], color: [f32; 4], name: Option<String>) {
        self.triangles.push(ColoredTriangle {
            vertices,
            color,
            name,
        });
    }
}

/// A named vector for visualization (e.g., fitted normals, edge directions).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedVector {
    pub name: String,
    /// Origin position in sample cloud space.
    pub origin: [f32; 3],
    /// Vector direction and magnitude.
    pub direction: [f32; 3],
    /// Optional RGBA color (UI will auto-assign if None).
    pub color: Option<[f32; 4]>,
}

/// A colored triangle for rendering fitted geometry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColoredTriangle {
    pub vertices: [[f32; 3]; 3],
    pub color: [f32; 4],
    /// Optional name for grouping (e.g., "face_0", "edge_fit").
    pub name: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SampleCloudMeta {
    pub samples_used: Option<u32>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplePoint {
    pub position: [f32; 3],
    pub kind: SamplePointKind,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SamplePointKind {
    Unknown,
    Probe,
    Crossing,
    Inside,
    Outside,
}
