use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const SAMPLE_CLOUD_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleCloudDump {
    pub version: u32,
    pub sets: Vec<SampleCloudSet>,
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
    pub hint_normal: [f32; 3],
    pub points: Vec<SamplePoint>,
    pub meta: SampleCloudMeta,
}

impl SampleCloudSet {
    pub fn new(id: u64, vertex: [f32; 3], hint_normal: [f32; 3]) -> Self {
        Self {
            id,
            label: None,
            vertex,
            hint_normal,
            points: Vec::new(),
            meta: SampleCloudMeta::default(),
        }
    }
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
