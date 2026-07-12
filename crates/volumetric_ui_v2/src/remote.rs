//! Remote execution backend: forwards project runs and ASN2 meshing to a
//! `volumetric_daemon` over the `volumetric_protocol` HTTP client.
//!
//! Everything here runs on the shell's background worker thread — the
//! blocking client sits exactly where a local `Project::run_cancellable`
//! call sits in [`crate::session::LocalBackend`]. Client-side cancel flags
//! never cross the wire: the daemon assigns job ids, and the client's
//! long-poll loop forwards a raised flag as a cancel request, then waits for
//! the daemon's acknowledging outcome.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(target_arch = "wasm32"))]
use volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2;
#[cfg(not(target_arch = "wasm32"))]
use volumetric_protocol::{DaemonClient, JobRequest};
use volumetric_protocol::{JobOutcome, JobOutput};

#[cfg(not(target_arch = "wasm32"))]
use crate::session::ExecutionBackend;

// ---------------------------------------------------------------------------
// Outcome mapping, shared by the native blocking backend below and the web
// host's async fetch path (which awaits the same daemon protocol through
// `volumetric_protocol::WebDaemonClient`). Keeping these together is what
// keeps the two transports semantically identical.
// ---------------------------------------------------------------------------

/// Success output, `Ok(None)` for a cancelled job, or the daemon's failure
/// message.
pub fn output_from_outcome(outcome: JobOutcome) -> Result<Option<JobOutput>, String> {
    match outcome {
        JobOutcome::Success { output, .. } => Ok(Some(output)),
        JobOutcome::Cancelled => Ok(None),
        JobOutcome::Failed { error } => Err(error),
    }
}

/// Maps a project-run job's output to the run result.
pub fn project_exports_from_output(
    output: Option<JobOutput>,
) -> Result<Vec<volumetric::LoadedAsset>, String> {
    match output {
        Some(JobOutput::RunProject { exports }) => Ok(exports
            .into_iter()
            .map(volumetric_protocol::ExportedAsset::into_loaded)
            .collect()),
        Some(_) => Err("daemon returned the wrong output kind for a project run".to_string()),
        // Matches LocalBackend, where a cancelled run surfaces as the
        // ExecutionError::Cancelled message; the session discards the
        // result by generation either way.
        None => Err("Execution cancelled".to_string()),
    }
}

/// Maps a mesh job's output to the meshing result (`Ok(None)`: cancelled).
pub fn mesh_result_from_output(
    output: Option<JobOutput>,
) -> Result<Option<volumetric::AdaptiveMeshV2Result>, String> {
    match output {
        Some(JobOutput::MeshModel { mesh, stats }) => {
            let vertices = mesh.unpack_positions().map_err(|e| e.to_string())?;
            let normals = mesh.unpack_normals().map_err(|e| e.to_string())?;
            let indices = mesh.unpack_indices().map_err(|e| e.to_string())?;
            Ok(Some(volumetric::AdaptiveMeshV2Result {
                vertices,
                normals,
                indices,
                bounds_min: mesh.bounds_min.into(),
                bounds_max: mesh.bounds_max.into(),
                stats,
            }))
        }
        Some(_) => Err("daemon returned the wrong output kind for a mesh job".to_string()),
        None => Ok(None),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct RemoteBackend {
    client: DaemonClient,
    address: String,
    /// One-time reachability + protocol-version probe, run lazily on the
    /// worker thread by the first job (constructing the backend must not
    /// block the UI thread on a connect timeout).
    probe: OnceLock<Result<(), String>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl RemoteBackend {
    /// `address` like `http://buildbox:7373`.
    pub fn new(address: &str) -> Self {
        Self {
            client: DaemonClient::new(address),
            address: address.to_string(),
            probe: OnceLock::new(),
        }
    }

    fn ensure_ready(&self) -> Result<(), String> {
        self.probe
            .get_or_init(|| match self.client.info() {
                Ok(_) => Ok(()),
                Err(err) => Err(format!("remote build at {}: {err}", self.address)),
            })
            .clone()
    }

    /// Submits a job and long-polls it to completion, forwarding a raised
    /// cancel flag and the daemon's progress snapshots. Returns the success
    /// output, `Ok(None)` for a cancelled job, or the daemon's failure
    /// message.
    fn run_job(
        &self,
        request: &JobRequest,
        cancel: &AtomicBool,
        progress: &dyn Fn(volumetric::BuildProgress),
    ) -> Result<Option<JobOutput>, String> {
        self.ensure_ready()?;
        let outcome = self
            .client
            .run(request, &|| cancel.load(Ordering::Relaxed), progress)
            .map_err(|err| format!("remote build at {}: {err}", self.address))?;
        output_from_outcome(outcome)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ExecutionBackend for RemoteBackend {
    fn run_project(
        &self,
        project: &volumetric::Project,
        cancel: &AtomicBool,
        progress: &dyn Fn(volumetric::BuildProgress),
    ) -> Result<Vec<volumetric::LoadedAsset>, String> {
        let request = JobRequest::RunProject {
            project: project.clone(),
        };
        project_exports_from_output(self.run_job(&request, cancel, progress)?)
    }

    fn mesh_model(
        &self,
        model_wasm: &[u8],
        config: &AdaptiveMeshConfig2,
        cancel: &AtomicBool,
        progress: &dyn Fn(volumetric::BuildProgress),
    ) -> Result<Option<volumetric::AdaptiveMeshV2Result>, String> {
        let request = JobRequest::MeshModel {
            model_wasm: model_wasm.to_vec(),
            config: config.clone(),
        };
        mesh_result_from_output(self.run_job(&request, cancel, progress)?)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::session::{BackgroundJob, BackgroundResult, PreviewBuildJob, execute_job_with};
    use crate::{Asn2Settings, PreviewMeshPlan, PreviewPlan, PreviewRequest, VolumetricUiV2};

    fn daemon_backend() -> (volumetric_daemon::DaemonHandle, RemoteBackend) {
        let handle = volumetric_daemon::start(volumetric_daemon::DaemonConfig {
            bind: std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
            ..Default::default()
        })
        .expect("daemon starts on an ephemeral port");
        let backend = RemoteBackend::new(&format!("http://{}", handle.addr()));
        (handle, backend)
    }

    /// The same RunProject job the shell dispatches, executed through a real
    /// daemon: the default project's exports come back as loaded assets.
    #[test]
    fn remote_backend_runs_the_default_project() {
        let (handle, backend) = daemon_backend();

        let project = VolumetricUiV2::default().project().clone();
        let result = execute_job_with(
            BackgroundJob::RunProject {
                generation: 3,
                project,
                cancel: Arc::new(AtomicBool::new(false)),
            },
            &backend,
        );

        let BackgroundResult::ProjectComplete {
            generation, result, ..
        } = result
        else {
            panic!("wrong result kind");
        };
        assert_eq!(generation, 3);
        let assets = result.expect("default project runs on the daemon");
        assert_eq!(assets.len(), 1);
        assert!(!assets[0].data().is_empty());

        handle.shutdown();
    }

    /// A preview build with the ASN2 plan meshes on the daemon and still
    /// assembles a full scene locally (colormapping needs the local model
    /// executor, so this covers the split).
    #[test]
    fn remote_backend_builds_an_asn2_preview() {
        let (handle, backend) = daemon_backend();

        let sphere = volumetric_assets::get_model("simple_sphere_model")
            .expect("bundled sphere model")
            .bytes
            .to_vec();
        let request = PreviewRequest {
            asset_id: "sphere".to_string(),
            data: Arc::new(sphere),
            type_hint: Some(volumetric::AssetTypeHint::Model),
            precursor_ids: vec![],
            plan: PreviewPlan::Model3d {
                mesh: PreviewMeshPlan::AdaptiveSurfaceNets2 {
                    target_resolution: 32,
                    base_resolution: 8,
                    max_depth: 2,
                    settings: Asn2Settings::default(),
                },
                color_channel: None,
            },
            wireframe: false,
            show_grid: false,
            show_bounds: false,
            ssao: false,
            ssao_radius: 0.5,
            ssao_bias: 0.025,
            ssao_strength: 1.0,
            stale: false,
        };
        let job = PreviewBuildJob {
            key: (&request).into(),
            request,
            cancel: Arc::new(AtomicBool::new(false)),
        };

        let result = execute_job_with(BackgroundJob::BuildPreview(job), &backend);
        let BackgroundResult::PreviewComplete(preview) = result else {
            panic!("wrong result kind");
        };
        let entity = preview
            .result
            .as_ref()
            .map_err(|_| "preview build failed")
            .expect("sphere meshes through the daemon");
        assert!(entity.stats.triangles > 0);

        handle.shutdown();
    }
}
