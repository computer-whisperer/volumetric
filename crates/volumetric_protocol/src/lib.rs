//! Wire protocol for the volumetric processing daemon.
//!
//! A `volumetric_daemon` accepts build work (project execution, model
//! meshing) over HTTP so clients — the native UI, the CLI, eventually the
//! web UI — can offload it to another process or another machine. This crate
//! defines the messages both sides exchange and (behind the `client`
//! feature) a blocking HTTP client.
//!
//! # Transport
//!
//! Plain HTTP/1.1; every request and response body is CBOR
//! (`Content-Type: application/cbor`). Errors come back as plain-text bodies
//! with a 4xx/5xx status. Endpoints:
//!
//! | Method | Path                        | Body → Response                  |
//! |--------|-----------------------------|----------------------------------|
//! | GET    | `/v1/info`                  | — → [`DaemonInfo`]               |
//! | POST   | `/v1/jobs`                  | [`JobRequest`] → [`JobTicket`]   |
//! | GET    | `/v1/jobs/{id}?wait_ms=N`   | — → [`JobStatus`] (long-poll)    |
//! | POST   | `/v1/jobs/{id}/cancel`      | — → empty                        |
//!
//! `wait_ms` blocks the status request until the job finishes or the wait
//! elapses, so a client polls with long waits instead of hammering. Job ids
//! are meaningful only to the daemon instance that issued them (they do not
//! survive a daemon restart; a client that sees 404 for a job it submitted
//! should treat the job as lost and resubmit).
//!
//! Cancellation is cooperative and asynchronous: the cancel endpoint only
//! flips a flag, and the job later finishes with [`JobOutcome::Cancelled`]
//! (or races to success/failure if it was already past the last checkpoint).
//!
//! # Payload conventions
//!
//! Projects travel as their in-memory [`volumetric::Project`] type — the
//! request envelope's CBOR encoding of that field is exactly the `.vproj`
//! format. Bulk mesh arrays travel as packed little-endian byte strings
//! ([`MeshPayload`]) rather than CBOR element arrays: meshes run to hundreds
//! of MB and per-element encoding would roughly double them.

use serde::{Deserialize, Serialize};
use volumetric::adaptive_surface_nets_2::{AdaptiveMeshConfig2, MeshingStats2};
use volumetric::{AssetTypeHint, LoadedAsset, Project};

#[cfg(feature = "client")]
mod client;
#[cfg(feature = "client")]
pub use client::{ClientError, DaemonClient};

/// Bumped whenever a message shape changes incompatibly. Clients compare
/// against [`DaemonInfo::protocol_version`] before submitting work.
pub const PROTOCOL_VERSION: u32 = 1;

/// The port daemons bind by default (`--bind` overrides).
pub const DEFAULT_PORT: u16 = 7373;

/// Response of `GET /v1/info`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DaemonInfo {
    /// Human-readable daemon name.
    pub name: String,
    /// The daemon crate version.
    pub version: String,
    /// The [`PROTOCOL_VERSION`] the daemon speaks.
    pub protocol_version: u32,
}

/// A unit of build work, submitted via `POST /v1/jobs`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JobRequest {
    /// Execute a project's timeline and return its exports
    /// (the remote form of [`volumetric::Project::run`]).
    RunProject { project: Project },
    /// Mesh a model with Adaptive Surface Nets v2 (the remote form of
    /// `volumetric::generate_adaptive_mesh_v2_from_bytes`).
    MeshModel {
        /// The model wasm to sample.
        #[serde(with = "serde_bytes")]
        model_wasm: Vec<u8>,
        config: AdaptiveMeshConfig2,
    },
}

/// Response of `POST /v1/jobs`: the id to poll and cancel with.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct JobTicket {
    pub job_id: u64,
}

/// Where a job currently sits in the daemon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobState {
    /// Accepted, waiting for an execution slot.
    Queued,
    /// Executing.
    Running,
    /// Finished; [`JobStatus::outcome`] is populated.
    Finished,
}

/// Response of `GET /v1/jobs/{id}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobStatus {
    pub state: JobState,
    /// Present exactly when `state` is [`JobState::Finished`].
    pub outcome: Option<JobOutcome>,
}

/// How a finished job ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JobOutcome {
    Success {
        output: JobOutput,
        /// Wall-clock execution time on the daemon (excludes queueing).
        elapsed_ms: u64,
    },
    Failed {
        error: String,
    },
    Cancelled,
}

/// The result payload of a successful job, one variant per [`JobRequest`]
/// variant.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JobOutput {
    RunProject { exports: Vec<ExportedAsset> },
    MeshModel { mesh: MeshPayload, stats: MeshingStats2 },
}

/// A [`LoadedAsset`] in wire form.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportedAsset {
    pub id: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    pub type_hint: Option<AssetTypeHint>,
    pub precursor_ids: Vec<String>,
}

impl ExportedAsset {
    pub fn from_loaded(asset: &LoadedAsset) -> Self {
        Self {
            id: asset.id().to_string(),
            data: asset.data().to_vec(),
            type_hint: asset.type_hint(),
            precursor_ids: asset.precursor_ids().to_vec(),
        }
    }

    pub fn into_loaded(self) -> LoadedAsset {
        LoadedAsset::from_parts(self.id, self.data, self.type_hint, self.precursor_ids)
    }
}

/// An indexed triangle mesh as packed little-endian byte strings:
/// `positions`/`normals` are xyz `f32` triples (one triple per vertex, the
/// two arrays parallel), `indices` are `u32`s in groups of three.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshPayload {
    #[serde(with = "serde_bytes")]
    pub positions: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub normals: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub indices: Vec<u8>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

/// A [`MeshPayload`] whose array lengths don't describe a valid mesh.
#[derive(Clone, Debug, thiserror::Error)]
#[error("malformed mesh payload: {0}")]
pub struct MalformedMesh(String);

impl MeshPayload {
    pub fn pack(
        vertices: &[(f32, f32, f32)],
        normals: &[(f32, f32, f32)],
        indices: &[u32],
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    ) -> Self {
        Self {
            positions: pack_f32_triples(vertices),
            normals: pack_f32_triples(normals),
            indices: indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
            bounds_min,
            bounds_max,
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 12
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 12
    }

    pub fn unpack_positions(&self) -> Result<Vec<(f32, f32, f32)>, MalformedMesh> {
        unpack_f32_triples(&self.positions, "positions")
    }

    pub fn unpack_normals(&self) -> Result<Vec<(f32, f32, f32)>, MalformedMesh> {
        let normals = unpack_f32_triples(&self.normals, "normals")?;
        if self.normals.len() != self.positions.len() {
            return Err(MalformedMesh(format!(
                "{} normal bytes for {} position bytes",
                self.normals.len(),
                self.positions.len()
            )));
        }
        Ok(normals)
    }

    pub fn unpack_indices(&self) -> Result<Vec<u32>, MalformedMesh> {
        if self.indices.len() % 12 != 0 {
            return Err(MalformedMesh(format!(
                "index bytes ({}) are not whole u32 triangles",
                self.indices.len()
            )));
        }
        let vertex_count = self.vertex_count() as u32;
        let indices: Vec<u32> = self
            .indices
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
            .collect();
        if let Some(&bad) = indices.iter().find(|&&i| i >= vertex_count) {
            return Err(MalformedMesh(format!(
                "index {bad} out of range for {vertex_count} vertices"
            )));
        }
        Ok(indices)
    }
}

fn pack_f32_triples(triples: &[(f32, f32, f32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(triples.len() * 12);
    for &(x, y, z) in triples {
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes.extend_from_slice(&z.to_le_bytes());
    }
    bytes
}

fn unpack_f32_triples(bytes: &[u8], what: &str) -> Result<Vec<(f32, f32, f32)>, MalformedMesh> {
    if bytes.len() % 12 != 0 {
        return Err(MalformedMesh(format!(
            "{what} bytes ({}) are not whole f32 triples",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(12)
        .map(|c| {
            let f = |i: usize| f32::from_le_bytes(c[i..i + 4].try_into().expect("chunk of 12"));
            (f(0), f(4), f(8))
        })
        .collect())
}

/// Encodes any protocol message as CBOR.
pub fn to_cbor<T: Serialize>(value: &T) -> Vec<u8> {
    let mut bytes = Vec::new();
    ciborium::into_writer(value, &mut bytes).expect("CBOR encoding to memory cannot fail");
    bytes
}

/// Decodes a protocol message from CBOR.
pub fn from_cbor<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    ciborium::from_reader(bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_payload_round_trips() {
        let vertices = vec![(0.0, 1.0, 2.0), (3.5, -4.25, 5.0), (-1.0, 0.0, 9.75)];
        let normals = vec![(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)];
        let indices = vec![0u32, 1, 2];
        let payload = MeshPayload::pack(
            &vertices,
            &normals,
            &indices,
            [-1.0, -4.25, 0.0],
            [3.5, 1.0, 9.75],
        );

        assert_eq!(payload.vertex_count(), 3);
        assert_eq!(payload.triangle_count(), 1);

        let decoded: MeshPayload = from_cbor(&to_cbor(&payload)).unwrap();
        assert_eq!(decoded.unpack_positions().unwrap(), vertices);
        assert_eq!(decoded.unpack_normals().unwrap(), normals);
        assert_eq!(decoded.unpack_indices().unwrap(), indices);
    }

    #[test]
    fn mesh_payload_rejects_out_of_range_index() {
        let payload = MeshPayload::pack(
            &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            &[(0.0, 0.0, 1.0); 3],
            &[0, 1, 3],
            [0.0; 3],
            [1.0; 3],
        );
        assert!(payload.unpack_indices().is_err());
    }

    #[test]
    fn job_request_round_trips_with_typed_project() {
        let project = Project {
            version: 2,
            imports: vec![volumetric::ImportedAsset::operator(
                "op".to_string(),
                vec![1, 2, 3],
            )],
            timeline: vec![],
            exports: vec![],
        };
        let request = JobRequest::RunProject { project };
        let decoded: JobRequest = from_cbor(&to_cbor(&request)).unwrap();
        let JobRequest::RunProject { project } = decoded else {
            panic!("wrong variant");
        };
        assert_eq!(project.imports.len(), 1);
        assert_eq!(project.imports[0].data, vec![1, 2, 3]);
    }

    #[test]
    fn exported_asset_round_trips_through_loaded_asset() {
        let wire = ExportedAsset {
            id: "out".to_string(),
            data: vec![9, 8, 7],
            type_hint: Some(AssetTypeHint::FeaMesh),
            precursor_ids: vec!["a".to_string()],
        };
        let loaded = wire.clone().into_loaded();
        let back = ExportedAsset::from_loaded(&loaded);
        assert_eq!(back.id, wire.id);
        assert_eq!(back.data, wire.data);
        assert_eq!(back.type_hint, wire.type_hint);
        assert_eq!(back.precursor_ids, wire.precursor_ids);
    }
}
