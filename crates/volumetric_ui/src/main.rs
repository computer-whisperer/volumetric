use anyhow::{Context, Result};
use ciborium::value::Value as CborValue;
use eframe::egui;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use volumetric::sample_cloud::{SampleCloudDump, SampleCloudSet, SamplePointKind};

#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

// Web-specific imports
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use poll_promise::Promise;

mod platform;
mod renderer;

// =============================================================================
// Background Task System
// =============================================================================

/// A cancellation token that can be shared between the UI and background worker.
#[derive(Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// A task to be executed in the background.
enum BackgroundTask {
    /// Execute a project pipeline
    RunProject {
        project: Project,
    },
    /// Resample a single asset
    ResampleAsset {
        asset_id: String,
        wasm_bytes: Vec<u8>,
        mode: ExportRenderMode,
        config: ResampleConfig,
    },
}

/// Configuration for a resample operation (captures all the parameters needed)
#[derive(Clone)]
struct ResampleConfig {
    resolution: usize,
    // ASN v2
    asn2_base_cell_size: f64,
    asn2_max_depth: usize,
    asn2_vertex_refinement_iterations: usize,
    asn2_normal_sample_iterations: usize,
    asn2_normal_epsilon_frac: f32,
    asn2_sharp_edge_config: Option<adaptive_surface_nets_2::SharpEdgeConfig>,
}

/// Result from a background task.
enum BackgroundTaskResult {
    /// Project execution completed
    ProjectComplete {
        assets: Result<Vec<LoadedAsset>, String>,
        elapsed_secs: f32,
    },
    /// Asset resampling completed
    ResampleComplete {
        asset_id: String,
        result: ResampleResult,
    },
    /// Task was cancelled
    Cancelled {
        description: String,
    },
}

/// Result of a resample operation
struct ResampleResult {
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    points: Arc<Vec<(f32, f32, f32)>>,
    triangles: Vec<Triangle>,
    mesh_vertices: Arc<Vec<renderer::MeshVertex>>,
    mesh_indices: Option<Arc<Vec<u32>>>,
    sample_time: f32,
    sample_count: usize,
    error: Option<String>,
    asn2_stats: Option<adaptive_surface_nets_2::MeshingStats2>,
}

// =============================================================================
// Native Background Worker (Thread-based)
// =============================================================================

#[cfg(not(target_arch = "wasm32"))]
struct BackgroundWorker {
    /// Channel to send tasks to the worker
    task_sender: Sender<(BackgroundTask, CancellationToken)>,
    /// Channel to receive results from the worker
    result_receiver: Receiver<BackgroundTaskResult>,
    /// Handle to the worker thread
    _thread_handle: JoinHandle<()>,
}

#[cfg(not(target_arch = "wasm32"))]
impl BackgroundWorker {
    fn new() -> Self {
        let (task_sender, task_receiver) = mpsc::channel::<(BackgroundTask, CancellationToken)>();
        let (result_sender, result_receiver) = mpsc::channel::<BackgroundTaskResult>();

        let thread_handle = thread::spawn(move || {
            Self::worker_loop(task_receiver, result_sender);
        });

        Self {
            task_sender,
            result_receiver,
            _thread_handle: thread_handle,
        }
    }

    fn send_task(&self, task: BackgroundTask, cancel_token: CancellationToken) {
        // Ignore send errors - if the worker is dead, we'll notice when we try to receive
        let _ = self.task_sender.send((task, cancel_token));
    }

    fn try_recv_result(&self) -> Option<BackgroundTaskResult> {
        self.result_receiver.try_recv().ok()
    }

    fn worker_loop(
        task_receiver: Receiver<(BackgroundTask, CancellationToken)>,
        result_sender: Sender<BackgroundTaskResult>,
    ) {
        while let Ok((task, cancel_token)) = task_receiver.recv() {
            if cancel_token.is_cancelled() {
                let _ = result_sender.send(BackgroundTaskResult::Cancelled {
                    description: "Task cancelled before starting".to_string(),
                });
                continue;
            }

            let result = match task {
                BackgroundTask::RunProject { project } => {
                    execute_project(project, &cancel_token)
                }
                BackgroundTask::ResampleAsset { asset_id, wasm_bytes, mode, config } => {
                    execute_resample(asset_id, wasm_bytes, mode, config, &cancel_token)
                }
            };

            let _ = result_sender.send(result);
        }
    }
}

// =============================================================================
// Web Background Worker (Synchronous execution in main thread)
// =============================================================================

#[cfg(target_arch = "wasm32")]
struct BackgroundWorker {
    /// Queue of pending tasks to execute
    pending_tasks: std::collections::VecDeque<(BackgroundTask, CancellationToken)>,
}

#[cfg(target_arch = "wasm32")]
impl BackgroundWorker {
    fn new() -> Self {
        Self { pending_tasks: std::collections::VecDeque::new() }
    }

    fn send_task(&mut self, task: BackgroundTask, cancel_token: CancellationToken) {
        self.pending_tasks.push_back((task, cancel_token));
    }

    /// On web, execute the next pending task synchronously when polled.
    /// This may cause brief UI freezes for long operations.
    fn try_recv_result(&mut self) -> Option<BackgroundTaskResult> {
        let (task, cancel_token) = self.pending_tasks.pop_front()?;

        if cancel_token.is_cancelled() {
            return Some(BackgroundTaskResult::Cancelled {
                description: "Task cancelled before starting".to_string(),
            });
        }

        let result = match task {
            BackgroundTask::RunProject { project } => execute_project(project, &cancel_token),
            BackgroundTask::ResampleAsset {
                asset_id,
                wasm_bytes,
                mode,
                config,
            } => execute_resample(asset_id, wasm_bytes, mode, config, &cancel_token),
        };

        Some(result)
    }
}

// =============================================================================
// Task Execution Functions (shared between native and web)
// =============================================================================

fn execute_project(project: Project, cancel_token: &CancellationToken) -> BackgroundTaskResult {
    let start_time = web_time::Instant::now();
    let mut env = Environment::new();

    // Check cancellation before starting
    if cancel_token.is_cancelled() {
        return BackgroundTaskResult::Cancelled {
            description: "Project execution cancelled".to_string(),
        };
    }

    match project.run(&mut env) {
        Ok(assets) => BackgroundTaskResult::ProjectComplete {
            assets: Ok(assets),
            elapsed_secs: start_time.elapsed().as_secs_f32(),
        },
        Err(e) => BackgroundTaskResult::ProjectComplete {
            assets: Err(format!("Failed to run project: {}", e)),
            elapsed_secs: start_time.elapsed().as_secs_f32(),
        },
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_resample(
    asset_id: String,
    wasm_bytes: Vec<u8>,
    mode: ExportRenderMode,
    config: ResampleConfig,
    cancel_token: &CancellationToken,
) -> BackgroundTaskResult {
    if cancel_token.is_cancelled() {
        return BackgroundTaskResult::Cancelled {
            description: format!("Resample of {} cancelled", asset_id),
        };
    }

    let start_time = web_time::Instant::now();
    let mut result = ResampleResult {
        bounds_min: (0.0, 0.0, 0.0),
        bounds_max: (0.0, 0.0, 0.0),
        points: Arc::new(Vec::new()),
        triangles: Vec::new(),
        mesh_vertices: Arc::new(Vec::new()),
        mesh_indices: None,
        sample_time: 0.0,
        sample_count: 0,
        error: None,
        asn2_stats: None,
    };

    match mode {
        ExportRenderMode::None => {
            // Nothing to do
        }
        ExportRenderMode::PointCloud => {
            match sample_model_from_bytes(&wasm_bytes, config.resolution) {
                Ok((pts, bounds_min, bounds_max)) => {
                    result.bounds_min = bounds_min;
                    result.bounds_max = bounds_max;
                    result.sample_count = pts.len();
                    result.points = Arc::new(pts);
                }
                Err(e) => {
                    result.error = Some(format_anyhow_error_chain(&e));
                }
            }
        }
        ExportRenderMode::MarchingCubes => {
            match generate_marching_cubes_mesh_from_bytes(&wasm_bytes, config.resolution) {
                Ok((tris, bounds_min, bounds_max)) => {
                    result.bounds_min = bounds_min;
                    result.bounds_max = bounds_max;
                    result.sample_count = tris.len();
                    result.mesh_vertices = Arc::new(
                        tris.iter()
                            .flat_map(|t| {
                                [
                                    renderer::MeshVertex {
                                        position: t.vertices[0].into(),
                                        _pad0: 0.0,
                                        normal: t.normals[0].into(),
                                        _pad1: 0.0,
                                    },
                                    renderer::MeshVertex {
                                        position: t.vertices[1].into(),
                                        _pad0: 0.0,
                                        normal: t.normals[1].into(),
                                        _pad1: 0.0,
                                    },
                                    renderer::MeshVertex {
                                        position: t.vertices[2].into(),
                                        _pad0: 0.0,
                                        normal: t.normals[2].into(),
                                        _pad1: 0.0,
                                    },
                                ]
                            })
                            .collect(),
                    );
                    result.triangles = tris;
                }
                Err(e) => {
                    result.error = Some(format_anyhow_error_chain(&e));
                }
            }
        }
        ExportRenderMode::AdaptiveSurfaceNets2 => {
            let asn2_config = adaptive_surface_nets_2::AdaptiveMeshConfig2 {
                base_cell_size: config.asn2_base_cell_size,
                max_depth: config.asn2_max_depth,
                vertex_refinement_iterations: config.asn2_vertex_refinement_iterations,
                normal_sample_iterations: config.asn2_normal_sample_iterations,
                normal_epsilon_frac: config.asn2_normal_epsilon_frac,
                num_threads: 0,
                sharp_edge_config: config.asn2_sharp_edge_config.clone(),
            };
            match generate_adaptive_mesh_v2_from_bytes(&wasm_bytes, &asn2_config) {
                Ok(meshing_result) => {
                    result.bounds_min = meshing_result.bounds_min;
                    result.bounds_max = meshing_result.bounds_max;
                    result.sample_count = meshing_result.stats.total_samples as usize;
                    result.asn2_stats = Some(meshing_result.stats);

                    // Build indexed mesh data
                    let vertices: Vec<renderer::MeshVertex> = meshing_result
                        .vertices
                        .iter()
                        .zip(meshing_result.normals.iter())
                        .map(|(pos, norm)| renderer::MeshVertex {
                            position: (*pos).into(),
                            _pad0: 0.0,
                            normal: (*norm).into(),
                            _pad1: 0.0,
                        })
                        .collect();

                    result.mesh_vertices = Arc::new(vertices);
                    result.mesh_indices = Some(Arc::new(meshing_result.indices));
                }
                Err(e) => {
                    result.error = Some(format_anyhow_error_chain(&e));
                }
            }
        }
    }

    result.sample_time = start_time.elapsed().as_secs_f32();

    BackgroundTaskResult::ResampleComplete { asset_id, result }
}

#[cfg(target_arch = "wasm32")]
fn execute_resample(
    asset_id: String,
    wasm_bytes: Vec<u8>,
    mode: ExportRenderMode,
    config: ResampleConfig,
    cancel_token: &CancellationToken,
) -> BackgroundTaskResult {
    if cancel_token.is_cancelled() {
        return BackgroundTaskResult::Cancelled {
            description: format!("Resample of {} cancelled", asset_id),
        };
    }

    let start_time = web_time::Instant::now();
    let mut result = ResampleResult {
        bounds_min: (0.0, 0.0, 0.0),
        bounds_max: (0.0, 0.0, 0.0),
        points: Arc::new(Vec::new()),
        triangles: Vec::new(),
        mesh_vertices: Arc::new(Vec::new()),
        mesh_indices: None,
        sample_time: 0.0,
        sample_count: 0,
        error: None,
        asn2_stats: None,
    };

    match mode {
        ExportRenderMode::None => {
            // Nothing to do
        }
        ExportRenderMode::PointCloud => {
            match sample_model_from_bytes(&wasm_bytes, config.resolution) {
                Ok((pts, bounds_min, bounds_max)) => {
                    result.bounds_min = bounds_min;
                    result.bounds_max = bounds_max;
                    result.sample_count = pts.len();
                    result.points = Arc::new(pts);
                }
                Err(e) => {
                    result.error = Some(format!("{}", e));
                }
            }
        }
        ExportRenderMode::MarchingCubes => {
            match generate_marching_cubes_mesh_from_bytes(&wasm_bytes, config.resolution) {
                Ok((tris, bounds_min, bounds_max)) => {
                    result.bounds_min = bounds_min;
                    result.bounds_max = bounds_max;
                    result.sample_count = tris.len();
                    result.mesh_vertices = Arc::new(
                        tris.iter()
                            .flat_map(|t| {
                                [
                                    renderer::MeshVertex {
                                        position: t.vertices[0].into(),
                                        _pad0: 0.0,
                                        normal: t.normals[0].into(),
                                        _pad1: 0.0,
                                    },
                                    renderer::MeshVertex {
                                        position: t.vertices[1].into(),
                                        _pad0: 0.0,
                                        normal: t.normals[1].into(),
                                        _pad1: 0.0,
                                    },
                                    renderer::MeshVertex {
                                        position: t.vertices[2].into(),
                                        _pad0: 0.0,
                                        normal: t.normals[2].into(),
                                        _pad1: 0.0,
                                    },
                                ]
                            })
                            .collect(),
                    );
                    result.triangles = tris;
                }
                Err(e) => {
                    result.error = Some(format!("{}", e));
                }
            }
        }
        ExportRenderMode::AdaptiveSurfaceNets2 => {
            let asn2_config = adaptive_surface_nets_2::AdaptiveMeshConfig2 {
                base_cell_size: config.asn2_base_cell_size,
                max_depth: config.asn2_max_depth,
                vertex_refinement_iterations: config.asn2_vertex_refinement_iterations,
                normal_sample_iterations: config.asn2_normal_sample_iterations,
                normal_epsilon_frac: config.asn2_normal_epsilon_frac,
                num_threads: 0, // Web is single-threaded
                sharp_edge_config: config.asn2_sharp_edge_config.clone(),
            };
            match generate_adaptive_mesh_v2_from_bytes(&wasm_bytes, &asn2_config) {
                Ok(meshing_result) => {
                    result.bounds_min = meshing_result.bounds_min;
                    result.bounds_max = meshing_result.bounds_max;
                    result.sample_count = meshing_result.stats.total_samples as usize;
                    result.asn2_stats = Some(meshing_result.stats);

                    // Build indexed mesh data
                    let vertices: Vec<renderer::MeshVertex> = meshing_result
                        .vertices
                        .iter()
                        .zip(meshing_result.normals.iter())
                        .map(|(pos, norm)| renderer::MeshVertex {
                            position: (*pos).into(),
                            _pad0: 0.0,
                            normal: (*norm).into(),
                            _pad1: 0.0,
                        })
                        .collect();

                    result.mesh_vertices = Arc::new(vertices);
                    result.mesh_indices = Some(Arc::new(meshing_result.indices));
                }
                Err(e) => {
                    result.error = Some(format!("{}", e));
                }
            }
        }
    }

    result.sample_time = start_time.elapsed().as_secs_f32();

    BackgroundTaskResult::ResampleComplete { asset_id, result }
}

/// Tracks the state of an in-progress background operation
struct InProgressOperation {
    /// Description of what's being done
    description: String,
    /// When the operation started
    start_time: web_time::Instant,
    /// Cancellation token for this operation
    cancel_token: CancellationToken,
}

// Project system - common types
use volumetric::{
    adaptive_surface_nets_2, AssetTypeHint, Environment, ExecutionInput, LoadedAsset,
    OperatorMetadata, OperatorMetadataInput, Project, Triangle,
};

// WASM execution functions available on both native and web
use volumetric::{generate_marching_cubes_mesh_from_bytes, generate_adaptive_mesh_v2_from_bytes, sample_model_from_bytes, operator_metadata_from_wasm_bytes};

// Native-only: Advanced WASM execution functions (require wasmtime directly)
#[cfg(not(target_arch = "wasm32"))]
use volumetric::stl;

enum ConfigFieldType {
    Bool,
    Int,
    Float,
    Text,
    Enum(Vec<String>),
}

#[derive(Clone, Debug)]
enum ConfigValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

fn parse_cddl_record_schema(cddl: &str) -> Result<Vec<(String, ConfigFieldType)>> {
    // v0 subset: a single CDDL record/map like: `{ dx: float, dy: float, dz: float }`
    // Supported leaf types: `bool`, `int`, `float`, `tstr`
    // Plus a small enum subset: `"a" / "b" / "c"` (tstr unions).
    let mut s = cddl.trim();
    if s.starts_with('{') {
        s = s.strip_prefix('{').unwrap().trim();
    }
    if s.ends_with('}') {
        s = s.strip_suffix('}').unwrap().trim();
    }

    if s.is_empty() {
        return Ok(vec![]);
    }

    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, ty) = part
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("Invalid CDDL field (expected `name: type`): `{part}`"))?;
        let name = name.trim();
        let ty = ty.trim();

        // Strip CDDL control operators like ".default 1.0" from the type
        // Per RFC 8610, .default is the proper syntax for default values
        let ty = if let Some(dot_idx) = ty.find(".default") {
            ty[..dot_idx].trim()
        } else if let Some(paren_idx) = ty.find('(') {
            // Also support legacy parenthetical annotations "(default 1.0)" for compatibility
            if !ty[..paren_idx].contains('"') {
                ty[..paren_idx].trim()
            } else {
                ty
            }
        } else {
            ty
        };

        let field_ty = match ty {
            "bool" => ConfigFieldType::Bool,
            "int" => ConfigFieldType::Int,
            "float" => ConfigFieldType::Float,
            "tstr" => ConfigFieldType::Text,
            other if other.contains('"') && other.contains('/') => {
                // Very small subset for string enums:
                //   op: "union" / "subtract" / "intersect"
                // Also tolerates surrounding parentheses.
                let trimmed = other.trim().trim_start_matches('(').trim_end_matches(')').trim();
                let mut options = Vec::new();
                for opt in trimmed.split('/') {
                    let opt = opt.trim();
                    if let Some(stripped) = opt.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                        if !stripped.is_empty() {
                            options.push(stripped.to_string());
                        }
                    }
                }
                if options.is_empty() {
                    return Err(anyhow::anyhow!(
                        "Unsupported enum CDDL type `{other}` (expected something like `\"a\" / \"b\"`)"
                    ));
                }
                ConfigFieldType::Enum(options)
            }
            other => {
                return Err(anyhow::anyhow!(
                    "Unsupported CDDL type `{other}` (supported: bool, int, float, tstr, and string enums like `\"a\" / \"b\"`)"
                ));
            }
        };

        if name.is_empty() {
            return Err(anyhow::anyhow!("Empty field name in CDDL: `{part}`"));
        }

        out.push((name.to_string(), field_ty));
    }

    Ok(out)
}

fn encode_config_map_to_cbor(fields: &[(String, ConfigFieldType)], values: &HashMap<String, ConfigValue>) -> Result<Vec<u8>> {
    let mut map_entries: Vec<(CborValue, CborValue)> = Vec::with_capacity(fields.len());
    for (name, ty) in fields {
        let value = values.get(name);
        let cbor_value = match (ty, value) {
            (ConfigFieldType::Bool, Some(ConfigValue::Bool(b))) => CborValue::Bool(*b),
            (ConfigFieldType::Int, Some(ConfigValue::Int(i))) => CborValue::Integer((*i).into()),
            (ConfigFieldType::Float, Some(ConfigValue::Float(f))) => CborValue::Float(*f),
            (ConfigFieldType::Text, Some(ConfigValue::Text(t))) => CborValue::Text(t.clone()),
            (ConfigFieldType::Enum(_), Some(ConfigValue::Text(t))) => CborValue::Text(t.clone()),
            // If unset or mismatched, use a sensible default for now.
            (ConfigFieldType::Bool, _) => CborValue::Bool(false),
            (ConfigFieldType::Int, _) => CborValue::Integer(0.into()),
            (ConfigFieldType::Float, _) => CborValue::Float(0.0),
            (ConfigFieldType::Text, _) => CborValue::Text(String::new()),
            (ConfigFieldType::Enum(options), _) => {
                CborValue::Text(options.first().cloned().unwrap_or_default())
            }
        };

        map_entries.push((CborValue::Text(name.clone()), cbor_value));
    }

    let value = CborValue::Map(map_entries);
    let mut out = Vec::new();
    ciborium::ser::into_writer(&value, &mut out)
        .context("Failed to encode configuration CBOR")?;
    Ok(out)
}

#[cfg(test)]
mod cddl_config_tests {
    use super::*;

    #[derive(Debug, serde::Deserialize, PartialEq)]
    struct TranslateConfig {
        dx: f32,
        dy: f32,
        dz: f32,
    }

    #[test]
    fn parse_and_encode_config_produces_cbor_map_decodable_as_struct() {
        let fields = parse_cddl_record_schema("{ dx: float, dy: float, dz: float }").unwrap();

        let mut values = HashMap::new();
        values.insert("dx".to_string(), ConfigValue::Float(2.5));
        values.insert("dy".to_string(), ConfigValue::Float(-1.0));
        values.insert("dz".to_string(), ConfigValue::Float(0.125));

        let bytes = encode_config_map_to_cbor(&fields, &values).unwrap();

        let mut cursor = std::io::Cursor::new(&bytes);
        let decoded: TranslateConfig = ciborium::de::from_reader(&mut cursor).unwrap();

        assert_eq!(
            decoded,
            TranslateConfig {
                dx: 2.5,
                dy: -1.0,
                dz: 0.125
            }
        );
    }

    #[test]
    fn parse_cddl_with_rfc8610_default_syntax() {
        // RFC 8610 specifies .default as the control operator for default values
        let fields = parse_cddl_record_schema(
            "{ width: float .default 1.0, depth: float .default 1.0, height: float .default 1.0 }"
        ).unwrap();

        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].0, "width");
        assert!(matches!(fields[0].1, ConfigFieldType::Float));
        assert_eq!(fields[1].0, "depth");
        assert!(matches!(fields[1].1, ConfigFieldType::Float));
        assert_eq!(fields[2].0, "height");
        assert!(matches!(fields[2].1, ConfigFieldType::Float));
    }

    #[test]
    fn parse_cddl_with_bool_default() {
        // RFC 8610 .default syntax
        let fields = parse_cddl_record_schema(
            "{ center: bool .default false }"
        ).unwrap();

        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, "center");
        assert!(matches!(fields[0].1, ConfigFieldType::Bool));
    }

    #[test]
    fn parse_cddl_with_legacy_paren_annotations() {
        // Legacy (default X) annotations for backward compatibility
        let fields = parse_cddl_record_schema(
            "{ scale: float (default 1.0) }"
        ).unwrap();

        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, "scale");
        assert!(matches!(fields[0].1, ConfigFieldType::Float));
    }
}

#[cfg(test)]
mod wasm_error_reporting_tests {
    use super::*;

    #[test]
    fn invalid_wasm_bytes_produce_informative_error_chain() {
        let wasm_bytes = vec![0x01, 0x02, 0x03, 0x04];
        let err = sample_model_from_bytes(&wasm_bytes, 4).unwrap_err();
        let msg = format_anyhow_error_chain(&err);
        assert!(msg.contains("Failed to load WASM module from bytes"), "{msg}");
    }
}

/// Rendering mode selection for an exported asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportRenderMode {
    None,
    PointCloud,
    MarchingCubes,
    AdaptiveSurfaceNets2,
}

impl ExportRenderMode {
    fn label(self) -> &'static str {
        match self {
            ExportRenderMode::None => "None",
            ExportRenderMode::PointCloud => "Point Cloud",
            ExportRenderMode::MarchingCubes => "Marching Cubes",
            ExportRenderMode::AdaptiveSurfaceNets2 => "ASN v2 (Indexed)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleCloudViewMode {
    Overlay,
    Split,
    CloudOnly,
}

impl SampleCloudViewMode {
    const ALL: [SampleCloudViewMode; 3] = [
        SampleCloudViewMode::Overlay,
        SampleCloudViewMode::Split,
        SampleCloudViewMode::CloudOnly,
    ];

    fn label(self) -> &'static str {
        match self {
            SampleCloudViewMode::Overlay => "Overlay",
            SampleCloudViewMode::Split => "Split",
            SampleCloudViewMode::CloudOnly => "Cloud Only",
        }
    }
}

/// A triangle in 3D space

/// Per-asset render data for multi-entity rendering support.
/// Each exported asset can have its own render mode and cached geometry.
struct AssetRenderData {
    /// The render mode for this asset
    mode: ExportRenderMode,
    /// Cached WASM bytes for this asset
    wasm_bytes: Vec<u8>,
    /// Bounding box minimum
    bounds_min: (f32, f32, f32),
    /// Bounding box maximum
    bounds_max: (f32, f32, f32),
    /// Point cloud data (for PointCloud mode)
    points: Arc<Vec<(f32, f32, f32)>>,
    /// Triangle data (for MarchingCubes mode)
    triangles: Vec<Triangle>,
    /// Mesh vertices (for non-indexed mesh modes)
    mesh_vertices: Arc<Vec<renderer::MeshVertex>>,
    /// Index buffer (for indexed mesh modes like ASN2)
    mesh_indices: Option<Arc<Vec<u32>>>,
    /// Whether this asset needs resampling
    needs_resample: bool,

    /// Time taken for the last sampling/meshing (in seconds)
    last_sample_time: Option<f32>,
    /// Number of samples taken during the last sampling/meshing
    last_sample_count: Option<usize>,

    /// Last sampling/meshing error for this asset (shown in the GUI)
    last_error: Option<String>,

    /// Resolution for PointCloud and MarchingCubes modes (samples per axis)
    resolution: usize,
    /// Whether to automatically resample when parameters change
    auto_resample: bool,

    /// Adaptive Surface Nets v2: base cell size (world units, cubic cells)
    asn2_base_cell_size: f64,
    /// Adaptive Surface Nets v2: maximum refinement depth
    asn2_max_depth: usize,
    /// Adaptive Surface Nets v2: binary search iterations for vertex position refinement
    asn2_vertex_refinement_iterations: usize,
    /// Adaptive Surface Nets v2: iterations for normal re-estimation at refined positions (0 = use mesh normals)
    asn2_normal_sample_iterations: usize,
    /// Adaptive Surface Nets v2: epsilon fraction for normal estimation
    asn2_normal_epsilon_frac: f32,
    /// Adaptive Surface Nets v2: sharp edge detection config (None = disabled)
    asn2_sharp_edge_config: Option<adaptive_surface_nets_2::SharpEdgeConfig>,

    /// Detailed profiling stats from the last ASN2 meshing operation
    asn2_stats: Option<adaptive_surface_nets_2::MeshingStats2>,
}

impl AssetRenderData {
    fn new(wasm_bytes: Vec<u8>, mode: ExportRenderMode) -> Self {
        Self {
            mode,
            wasm_bytes,
            bounds_min: (0.0, 0.0, 0.0),
            bounds_max: (0.0, 0.0, 0.0),
            points: Arc::new(Vec::new()),
            triangles: Vec::new(),
            mesh_vertices: Arc::new(Vec::new()),
            mesh_indices: None,
            needs_resample: true,
            last_sample_time: None,
            last_sample_count: None,
            last_error: None,
            resolution: 20,
            auto_resample: true,
            // ASN v2 config
            asn2_base_cell_size: 0.25,
            asn2_max_depth: 4,
            asn2_vertex_refinement_iterations: 12,
            // Normal refinement via tangent probing - works well with binary samplers
            asn2_normal_sample_iterations: 12,
            asn2_normal_epsilon_frac: 0.1,
            asn2_sharp_edge_config: None, // Sharp edge detection disabled by default
            asn2_stats: None,
        }
    }
}

fn format_anyhow_error_chain(e: &anyhow::Error) -> String {
    // `anyhow::Error` implements `Error` but its `Display` often only shows the top message.
    // For GUI reporting, include the causal chain so users can diagnose invalid WASM, missing
    // exports, etc.
    let mut parts = Vec::new();
    for (idx, cause) in e.chain().enumerate() {
        if idx == 0 {
            parts.push(cause.to_string());
        } else {
            parts.push(format!("Caused by: {cause}"));
        }
    }
    parts.join("\n")
}

fn build_sample_cloud_points(set: &SampleCloudSet) -> renderer::PointData {
    let mut points = Vec::with_capacity(set.points.len());
    for point in &set.points {
        let color = match point.kind {
            SamplePointKind::Crossing => [0.2, 1.0, 0.4, 0.95],
            SamplePointKind::Inside => [1.0, 0.8, 0.2, 0.95],
            SamplePointKind::Outside => [1.0, 0.4, 0.2, 0.95],
            SamplePointKind::Probe => [0.2, 0.8, 1.0, 0.85],
            SamplePointKind::Unknown => [0.7, 0.7, 0.7, 0.85],
        };
        points.push(renderer::PointInstance {
            position: point.position,
            color,
        });
    }
    renderer::PointData { points }
}

fn build_sample_cloud_root(set: &SampleCloudSet) -> renderer::PointData {
    renderer::PointData {
        points: vec![renderer::PointInstance {
            position: set.vertex,
            color: [1.0, 0.2, 0.4, 1.0],
        }],
    }
}

/// Application state for the volumetric renderer
pub struct VolumetricApp {
    /// The current project (contains the model pipeline)
    project: Option<Project>,
    /// Path to the current project file (for save operations) - native only
    #[cfg(not(target_arch = "wasm32"))]
    project_path: Option<PathBuf>,
    /// Exported assets from the last project run (used for UX/testing)
    exported_assets: Vec<LoadedAsset>,
    /// Per-asset render data (keyed by asset id) - supports multiple entities rendering together
    asset_render_data: HashMap<String, AssetRenderData>,
    operation_input_asset_id: Option<String>,
    operation_input_asset_id_b: Option<String>,
    operation_config_values: HashMap<String, ConfigValue>,
    /// Lua script source text for LuaSource inputs
    operation_lua_script: String,
    /// Operator metadata cache (for bundled assets on all platforms)
    operator_metadata_cache: HashMap<String, CachedOperatorMetadata>,
    wgpu_target_format: wgpu::TextureFormat,
    // Camera state
    camera_theta: f32,
    camera_phi: f32,
    camera_radius: f32,
    camera_target: glam::Vec3,
    camera_control_scheme: renderer::CameraControlScheme,
    last_mouse_pos: Option<egui::Pos2>,
    /// Last project evaluation time (in seconds)
    last_evaluation_time: Option<f32>,
    // Error message
    error_message: Option<String>,
    /// Index of the project entry currently being edited (None if edit panel is closed)
    editing_entry_index: Option<usize>,
    /// Configuration values for the entry being edited
    edit_config_values: HashMap<String, ConfigValue>,
    /// Lua script for the entry being edited
    edit_lua_script: String,
    /// Input asset IDs for the entry being edited
    edit_input_asset_ids: Vec<Option<String>>,
    /// Output asset ID for the entry being edited
    edit_output_asset_id: String,
    /// VecF64 input values for the entry being edited.
    /// Key is input index, value is (use_literal, literal_values, asset_id).
    edit_vec_inputs: HashMap<usize, (bool, Vec<f64>, Option<String>)>,
    /// Whether to automatically rebuild the project when entries change
    auto_rebuild: bool,

    /// Loaded sample cloud dump (optional)
    sample_cloud: Option<SampleCloudDump>,
    /// Selected sample cloud set index
    sample_cloud_set_index: usize,
    /// Sample cloud visualization mode
    sample_cloud_mode: SampleCloudViewMode,
    /// Sample cloud point size in pixels
    sample_cloud_point_size: f32,
    /// Whether to render the sample cloud
    sample_cloud_enabled: bool,
    /// Path to sample cloud file (native only)
    #[cfg(not(target_arch = "wasm32"))]
    sample_cloud_path: Option<PathBuf>,

    // Shading
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_bias: f32,
    ssao_strength: f32,

    // Background processing
    /// The background worker for long-running operations
    background_worker: BackgroundWorker,
    /// Currently in-progress operation (if any)
    in_progress_operation: Option<InProgressOperation>,
    /// Assets currently being resampled in the background
    pending_resample_assets: std::collections::HashSet<String>,

    // Web-specific state
    /// Whether the loading spinner has been hidden (web only)
    #[cfg(target_arch = "wasm32")]
    loading_hidden: bool,
    /// Pending WASM file import (web only)
    #[cfg(target_arch = "wasm32")]
    pending_wasm_import: Option<Promise<Option<(String, Vec<u8>)>>>,
    /// Pending project load (web only)
    #[cfg(target_arch = "wasm32")]
    pending_project_load: Option<Promise<Option<Vec<u8>>>>,
    /// Pending STL file import (web only)
    #[cfg(target_arch = "wasm32")]
    pending_stl_import: Option<Promise<Option<Vec<u8>>>>,
    /// Pending STL file data for operator input (native only, on web it's handled via Promise)
    #[cfg(not(target_arch = "wasm32"))]
    pending_stl_data: Option<Vec<u8>>,
    /// Pending heightmap image import (web only)
    #[cfg(target_arch = "wasm32")]
    pending_heightmap_import: Option<Promise<Option<Vec<u8>>>>,
    /// Pending heightmap image data for operator input (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pending_heightmap_data: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
struct CachedOperatorMetadata {
    wasm_len: u64,
    #[cfg(not(target_arch = "wasm32"))]
    wasm_modified: Option<SystemTime>,
    metadata: Option<OperatorMetadata>,
}


/// Get bundled model WASM bytes by crate name.
/// Uses the asset registry which embeds WASM at compile time.
fn get_bundled_model(crate_name: &str) -> Option<&'static [u8]> {
    volumetric_assets::get_model(crate_name).map(|asset| asset.bytes)
}

/// Get bundled operator WASM bytes by crate name.
/// Uses the asset registry which embeds WASM at compile time.
fn get_bundled_operator(crate_name: &str) -> Option<&'static [u8]> {
    volumetric_assets::get_operator(crate_name).map(|asset| asset.bytes)
}

#[cfg(not(target_arch = "wasm32"))]
fn demo_wasm_path(crate_name: &str) -> Option<PathBuf> {
    // In debug builds, allow filesystem fallback for development convenience
    #[cfg(debug_assertions)]
    {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // We are in crates/volumetric_ui, so the workspace target directory is two levels up.
        path.pop();
        path.pop();

        let release = path
            .join("target")
            .join("wasm32-unknown-unknown")
            .join("release")
            .join(format!("{crate_name}.wasm"));
        if fs::metadata(&release).is_ok() {
            return Some(release);
        }

        let debug = path
            .join("target")
            .join("wasm32-unknown-unknown")
            .join("debug")
            .join(format!("{crate_name}.wasm"));
        if fs::metadata(&debug).is_ok() {
            return Some(debug);
        }
    }

    None
}

#[cfg(not(target_arch = "wasm32"))]
fn operation_wasm_path(crate_name: &str) -> Option<PathBuf> {
    // For now operations are built the same way as demo models: a cdylib .wasm in target/{debug|release}.
    demo_wasm_path(crate_name)
}

impl VolumetricApp {
    pub fn new(cc: &eframe::CreationContext<'_>, initial_project: Option<Project>) -> Self {
        let wgpu_target_format = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| rs.target_format)
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

        // If we have an initial project, run it to extract exports
        let (exported_assets, asset_render_data) = initial_project.as_ref().map_or((vec![], HashMap::new()), |proj| {
            let mut env = Environment::new();
            let assets = proj.run(&mut env).unwrap_or_default();

            // Create render data for the first model asset with PointCloud mode
            let mut render_data = HashMap::new();
            if let Some(first_model) = assets.iter().find(|a| a.as_model().is_some()) {
                let id = first_model.id().to_string();
                let wasm_bytes = first_model.as_model().unwrap().to_vec();
                render_data.insert(id, AssetRenderData::new(wasm_bytes, ExportRenderMode::AdaptiveSurfaceNets2));
            }
            (assets, render_data)
        });

        Self {
            project: initial_project,
            #[cfg(not(target_arch = "wasm32"))]
            project_path: None,
            exported_assets,
            asset_render_data,
            operation_input_asset_id: None,
            operation_input_asset_id_b: None,
            operation_config_values: HashMap::new(),
            operation_lua_script: String::new(),
            operator_metadata_cache: HashMap::new(),
            wgpu_target_format,
            camera_theta: std::f32::consts::FRAC_PI_4,
            camera_phi: std::f32::consts::FRAC_PI_4,
            camera_radius: 4.0,
            camera_target: glam::Vec3::ZERO,
            camera_control_scheme: renderer::CameraControlScheme::default(),
            last_mouse_pos: None,
            last_evaluation_time: None,
            error_message: None,
            editing_entry_index: None,
            edit_config_values: HashMap::new(),
            edit_lua_script: String::new(),
            edit_input_asset_ids: Vec::new(),
            edit_output_asset_id: String::new(),
            edit_vec_inputs: HashMap::new(),
            auto_rebuild: true,

            sample_cloud: None,
            sample_cloud_set_index: 0,
            sample_cloud_mode: SampleCloudViewMode::Overlay,
            sample_cloud_point_size: 5.0,
            sample_cloud_enabled: true,
            #[cfg(not(target_arch = "wasm32"))]
            sample_cloud_path: None,

            ssao_enabled: true,
            ssao_radius: 0.08,
            ssao_bias: 0.002,
            ssao_strength: 1.6,

            background_worker: BackgroundWorker::new(),
            in_progress_operation: None,
            pending_resample_assets: std::collections::HashSet::new(),

            #[cfg(target_arch = "wasm32")]
            loading_hidden: false,
            #[cfg(target_arch = "wasm32")]
            pending_wasm_import: None,
            #[cfg(target_arch = "wasm32")]
            pending_project_load: None,
            #[cfg(target_arch = "wasm32")]
            pending_stl_import: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_stl_data: None,
            #[cfg(target_arch = "wasm32")]
            pending_heightmap_import: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_heightmap_data: None,
        }
    }

    /// Poll for completed background tasks and apply their results
    fn poll_background_tasks(&mut self) {
        while let Some(result) = self.background_worker.try_recv_result() {
            match result {
                BackgroundTaskResult::ProjectComplete { assets, elapsed_secs } => {
                    self.in_progress_operation = None;
                    match assets {
                        Ok(assets) => {
                            self.last_evaluation_time = Some(elapsed_secs);
                            self.error_message = None;
                            self.exported_assets = assets.clone();

                            // Prune render data for exports that no longer exist
                            let exported_ids: std::collections::HashSet<String> = self
                                .exported_assets
                                .iter()
                                .map(|a| a.id().to_string())
                                .collect();
                            self.asset_render_data
                                .retain(|id, _| exported_ids.contains(id));

                            // Update WASM bytes for existing render data entries, and add new exports
                            for asset in &self.exported_assets {
                                if let Some(wasm_bytes) = asset.as_model() {
                                    let asset_id = asset.id();
                                    if let Some(render_data) = self.asset_render_data.get_mut(asset_id) {
                                        // Update existing entry
                                        render_data.wasm_bytes = wasm_bytes.to_vec();
                                        render_data.needs_resample = true;
                                        render_data.last_error = None;
                                    } else {
                                        // New export - initialize with ASNv2 render mode
                                        self.asset_render_data.insert(
                                            asset_id.to_string(),
                                            AssetRenderData::new(wasm_bytes.to_vec(), ExportRenderMode::AdaptiveSurfaceNets2),
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            self.last_evaluation_time = None;
                            self.error_message = Some(e);
                        }
                    }
                }
                BackgroundTaskResult::ResampleComplete { asset_id, result } => {
                    self.pending_resample_assets.remove(&asset_id);
                    if self.pending_resample_assets.is_empty() {
                        self.in_progress_operation = None;
                    }

                    if let Some(render_data) = self.asset_render_data.get_mut(&asset_id) {
                        render_data.bounds_min = result.bounds_min;
                        render_data.bounds_max = result.bounds_max;
                        render_data.points = result.points;
                        render_data.triangles = result.triangles;
                        render_data.mesh_vertices = result.mesh_vertices;
                        render_data.mesh_indices = result.mesh_indices;
                        render_data.last_sample_time = Some(result.sample_time);
                        render_data.last_sample_count = Some(result.sample_count);
                        render_data.last_error = result.error;
                        render_data.asn2_stats = result.asn2_stats;
                    }
                }
                BackgroundTaskResult::Cancelled { description } => {
                    log::info!("Background task cancelled: {}", description);
                    // Clear relevant state
                    self.pending_resample_assets.clear();
                    self.in_progress_operation = None;
                }
            }
        }
    }

    /// Cancel any in-progress background operation
    fn cancel_background_operation(&mut self) {
        if let Some(ref op) = self.in_progress_operation {
            op.cancel_token.cancel();
        }
        self.in_progress_operation = None;
        self.pending_resample_assets.clear();
    }

    /// Check if a background operation is in progress
    fn is_busy(&self) -> bool {
        self.in_progress_operation.is_some()
    }

    fn operator_metadata_cached(&mut self, crate_name: &str) -> Option<OperatorMetadata> {
        // First check bundled operators (works on both native and web)
        if let Some(wasm_bytes) = get_bundled_operator(crate_name) {
            let cache_key = format!("bundled:{}", crate_name);
            let wasm_len = wasm_bytes.len() as u64;

            let is_fresh = self
                .operator_metadata_cache
                .get(&cache_key)
                .is_some_and(|c| c.wasm_len == wasm_len);

            if !is_fresh {
                let metadata = operator_metadata_from_wasm_bytes(wasm_bytes).ok();

                self.operator_metadata_cache.insert(
                    cache_key.clone(),
                    CachedOperatorMetadata {
                        wasm_len,
                        #[cfg(not(target_arch = "wasm32"))]
                        wasm_modified: None,
                        metadata,
                    },
                );
            }

            return self.operator_metadata_cache
                .get(&cache_key)
                .and_then(|c| c.metadata.clone());
        }

        // Filesystem fallback for native debug builds
        #[cfg(all(not(target_arch = "wasm32"), debug_assertions))]
        {
            if let Some(path) = operation_wasm_path(crate_name) {
                let path_str = path.to_string_lossy().to_string();

                let (wasm_len, wasm_modified) = match fs::metadata(&path) {
                    Ok(m) => (m.len(), m.modified().ok()),
                    Err(_) => {
                        self.operator_metadata_cache.remove(&path_str);
                        return None;
                    }
                };

                let is_fresh = self
                    .operator_metadata_cache
                    .get(&path_str)
                    .is_some_and(|c| c.wasm_len == wasm_len && c.wasm_modified == wasm_modified);

                if !is_fresh {
                    let metadata = fs::read(&path)
                        .ok()
                        .and_then(|wasm_bytes| operator_metadata_from_wasm_bytes(&wasm_bytes).ok());
                    self.operator_metadata_cache.insert(
                        path_str.clone(),
                        CachedOperatorMetadata {
                            wasm_len,
                            wasm_modified,
                            metadata,
                        },
                    );
                }

                return self.operator_metadata_cache
                    .get(&path_str)
                    .and_then(|c| c.metadata.clone());
            }
        }

        None
    }

    /// Sets the render mode for a specific asset. If mode is None, removes the asset from rendering.
    fn set_asset_render_mode(&mut self, asset_id: &str, mode: ExportRenderMode) {
        if mode == ExportRenderMode::None {
            // Remove from render data
            self.asset_render_data.remove(asset_id);
            return;
        }

        // Check if asset exists and is renderable
        let wasm_bytes = self
            .exported_assets
            .iter()
            .find(|a| a.id() == asset_id)
            .and_then(|a| a.as_model())
            .map(|b| b.to_vec());

        match wasm_bytes {
            Some(bytes) => {
                // Update existing or create new render data
                if let Some(data) = self.asset_render_data.get_mut(asset_id) {
                    if data.mode != mode {
                        data.mode = mode;
                        data.needs_resample = true;
                        data.last_error = None;
                    }
                } else {
                    self.asset_render_data.insert(
                        asset_id.to_string(),
                        AssetRenderData::new(bytes, mode),
                    );
                }
                self.error_message = None;
            }
            None => {
                self.error_message = Some(format!(
                    "Export '{asset_id}' is not a renderable ModelWASM asset"
                ));
            }
        }
    }
    
    /// Gets the render mode for a specific asset
    fn get_asset_render_mode(&self, asset_id: &str) -> ExportRenderMode {
        self.asset_render_data
            .get(asset_id)
            .map(|d| d.mode)
            .unwrap_or(ExportRenderMode::None)
    }

    /// Runs the current project in the background
    fn run_project(&mut self) {
        if self.is_busy() {
            // Cancel existing operation first
            self.cancel_background_operation();
        }

        if let Some(ref project) = self.project {
            self.exported_assets.clear();

            let cancel_token = CancellationToken::new();
            self.in_progress_operation = Some(InProgressOperation {
                description: "Running project...".to_string(),
                start_time: web_time::Instant::now(),
                cancel_token: cancel_token.clone(),
            });

            self.background_worker.send_task(
                BackgroundTask::RunProject {
                    project: project.clone(),
                },
                cancel_token,
            );
        } else {
            self.last_evaluation_time = None;
        }
    }

    /// Processes an STL file import by running the stl_import_operator
    fn process_stl_import(&mut self, stl_bytes: Vec<u8>) {
        // Get the stl_import_operator WASM
        let op_wasm = if let Some(bytes) = get_bundled_operator("stl_import_operator") {
            bytes.to_vec()
        } else {
            // Filesystem fallback for native debug builds only
            #[cfg(all(not(target_arch = "wasm32"), debug_assertions))]
            {
                if let Some(path) = operation_wasm_path("stl_import_operator") {
                    match fs::read(&path) {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            self.error_message = Some(format!("Failed to read STL import operator: {}", e));
                            return;
                        }
                    }
                } else {
                    self.error_message = Some(
                        "STL Import operator not found. Build it first with: cargo build --release --target wasm32-unknown-unknown -p stl_import_operator".to_string()
                    );
                    return;
                }
            }

            #[cfg(any(target_arch = "wasm32", not(debug_assertions)))]
            {
                self.error_message = Some("STL Import operator not bundled. Rebuild with WASM assets available.".to_string());
                return;
            }
        };

        // Create project if needed
        if self.project.is_none() {
            self.project = Some(Project::new());
        }

        // Generate output name for the imported model
        let output_id = self.project.as_ref()
            .map(|p| p.default_output_name("stl_import", None))
            .unwrap_or_else(|| "stl_model".to_string());

        // Build inputs: [Blob (STL data), CBORConfiguration (default config)]
        let default_config = {
            // Encode default config: { scale: 1.0, translate: [0,0,0], center: false }
            let mut cbor_bytes = Vec::new();
            let config_map = ciborium::value::Value::Map(vec![
                (ciborium::value::Value::Text("scale".to_string()), ciborium::value::Value::Float(1.0)),
                (ciborium::value::Value::Text("translate".to_string()), ciborium::value::Value::Array(vec![
                    ciborium::value::Value::Float(0.0),
                    ciborium::value::Value::Float(0.0),
                    ciborium::value::Value::Float(0.0),
                ])),
                (ciborium::value::Value::Text("center".to_string()), ciborium::value::Value::Bool(false)),
            ]);
            ciborium::into_writer(&config_map, &mut cbor_bytes).ok();
            cbor_bytes
        };

        let inputs = vec![
            ExecutionInput::Inline(stl_bytes),      // Blob: STL data
            ExecutionInput::Inline(default_config), // CBORConfiguration
        ];

        let output_ids = vec![output_id.clone()];

        if let Some(ref mut project) = self.project {
            project.insert_operation(
                "op_stl_import_operator",
                op_wasm,
                inputs,
                output_ids,
                output_id,
            );
        }

        // Clear the project path since we modified the project
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.project_path = None;
        }

        self.run_project();
    }

    /// Processes a heightmap image import by running the heightmap_extrude_operator
    fn process_heightmap_import(&mut self, image_bytes: Vec<u8>) {
        // Get the heightmap_extrude_operator WASM
        let op_wasm = if let Some(bytes) = get_bundled_operator("heightmap_extrude_operator") {
            bytes.to_vec()
        } else {
            // Filesystem fallback for native debug builds only
            #[cfg(all(not(target_arch = "wasm32"), debug_assertions))]
            {
                if let Some(path) = operation_wasm_path("heightmap_extrude_operator") {
                    match fs::read(&path) {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            self.error_message = Some(format!("Failed to read heightmap extrude operator: {}", e));
                            return;
                        }
                    }
                } else {
                    self.error_message = Some(
                        "Heightmap Extrude operator not found. Build it first with: cargo build --release --target wasm32-unknown-unknown -p heightmap_extrude_operator".to_string()
                    );
                    return;
                }
            }

            #[cfg(any(target_arch = "wasm32", not(debug_assertions)))]
            {
                self.error_message = Some("Heightmap Extrude operator not bundled. Rebuild with WASM assets available.".to_string());
                return;
            }
        };

        // Create project if needed
        if self.project.is_none() {
            self.project = Some(Project::new());
        }

        // Generate output name for the imported model
        let output_id = self.project.as_ref()
            .map(|p| p.default_output_name("heightmap", None))
            .unwrap_or_else(|| "heightmap_model".to_string());

        // Build inputs: [CBORConfiguration (default config), Blob (image data)]
        let default_config = {
            // Encode default config: { width: 1.0, depth: 1.0, height: 1.0, clip: 0.0 }
            let mut cbor_bytes = Vec::new();
            let config_map = ciborium::value::Value::Map(vec![
                (ciborium::value::Value::Text("width".to_string()), ciborium::value::Value::Float(1.0)),
                (ciborium::value::Value::Text("depth".to_string()), ciborium::value::Value::Float(1.0)),
                (ciborium::value::Value::Text("height".to_string()), ciborium::value::Value::Float(1.0)),
                (ciborium::value::Value::Text("clip".to_string()), ciborium::value::Value::Float(0.0)),
            ]);
            ciborium::into_writer(&config_map, &mut cbor_bytes).ok();
            cbor_bytes
        };

        let inputs = vec![
            ExecutionInput::Inline(default_config), // CBORConfiguration
            ExecutionInput::Inline(image_bytes),    // Blob: image data
        ];

        let output_ids = vec![output_id.clone()];

        if let Some(ref mut project) = self.project {
            project.insert_operation(
                "op_heightmap_extrude_operator",
                op_wasm,
                inputs,
                output_ids,
                output_id,
            );
        }

        // Clear the project path since we modified the project
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.project_path = None;
        }

        self.run_project();
    }

    /// Resamples all assets that need resampling (in background)
    fn resample_all_assets(&mut self) {
        // Collect assets that need resampling
        let mut assets_to_resample: Vec<(String, Vec<u8>, ExportRenderMode, ResampleConfig)> = Vec::new();

        for (asset_id, render_data) in self.asset_render_data.iter_mut() {
            if !render_data.needs_resample {
                continue;
            }
            // Don't start new resample if this asset is already being resampled
            if self.pending_resample_assets.contains(asset_id) {
                continue;
            }
            render_data.needs_resample = false;

            if render_data.mode == ExportRenderMode::None {
                // Handle None mode synchronously - it's trivial
                render_data.bounds_min = (0.0, 0.0, 0.0);
                render_data.bounds_max = (0.0, 0.0, 0.0);
                render_data.points = Arc::new(Vec::new());
                render_data.triangles.clear();
                render_data.mesh_vertices = Arc::new(Vec::new());
                continue;
            }

            let config = ResampleConfig {
                resolution: render_data.resolution,
                asn2_base_cell_size: render_data.asn2_base_cell_size,
                asn2_max_depth: render_data.asn2_max_depth,
                asn2_vertex_refinement_iterations: render_data.asn2_vertex_refinement_iterations,
                asn2_normal_sample_iterations: render_data.asn2_normal_sample_iterations,
                asn2_normal_epsilon_frac: render_data.asn2_normal_epsilon_frac,
                asn2_sharp_edge_config: render_data.asn2_sharp_edge_config.clone(),
            };

            assets_to_resample.push((
                asset_id.clone(),
                render_data.wasm_bytes.clone(),
                render_data.mode,
                config,
            ));
        }

        if assets_to_resample.is_empty() {
            return;
        }

        // Cancel any existing operation if we're starting new resamples
        if self.is_busy() && !self.pending_resample_assets.is_empty() {
            // Already resampling, just add to the queue
        } else if self.is_busy() {
            self.cancel_background_operation();
        }

        // Create cancellation token for this batch
        let cancel_token = CancellationToken::new();

        // Update in-progress operation
        let asset_count = assets_to_resample.len();
        self.in_progress_operation = Some(InProgressOperation {
            description: if asset_count == 1 {
                format!("Meshing {}...", assets_to_resample[0].0)
            } else {
                format!("Meshing {} assets...", asset_count)
            },
            start_time: web_time::Instant::now(),
            cancel_token: cancel_token.clone(),
        });

        // Queue all resample tasks
        for (asset_id, wasm_bytes, mode, config) in assets_to_resample {
            self.pending_resample_assets.insert(asset_id.clone());
            self.background_worker.send_task(
                BackgroundTask::ResampleAsset {
                    asset_id,
                    wasm_bytes,
                    mode,
                    config,
                },
                cancel_token.clone(),
            );
        }
    }

    /// Check if any asset needs resampling
    fn any_needs_resample(&self) -> bool {
        self.asset_render_data.values().any(|d| d.needs_resample)
    }

    #[allow(dead_code)]
    fn camera_position(&self) -> (f32, f32, f32) {
        let x = self.camera_radius * self.camera_phi.sin() * self.camera_theta.cos();
        let y = self.camera_radius * self.camera_phi.cos();
        let z = self.camera_radius * self.camera_phi.sin() * self.camera_theta.sin();
        (x, y, z)
    }

    fn selected_sample_cloud_set(&self) -> Option<&SampleCloudSet> {
        let dump = self.sample_cloud.as_ref()?;
        dump.sets.get(self.sample_cloud_set_index)
    }

    #[allow(dead_code)]
    fn project_point(&self, point: (f32, f32, f32), rect: &egui::Rect) -> Option<egui::Pos2> {
        let (cx, cy, cz) = self.camera_position();
        
        // Simple perspective projection
        let dx = point.0 - cx;
        let dy = point.1 - cy;
        let dz = point.2 - cz;
        
        // Camera forward direction (towards origin)
        let forward = (-cx, -cy, -cz);
        let forward_len = (forward.0 * forward.0 + forward.1 * forward.1 + forward.2 * forward.2).sqrt();
        let forward = (forward.0 / forward_len, forward.1 / forward_len, forward.2 / forward_len);
        
        // Camera right direction
        let up = (0.0, 1.0, 0.0);
        let right = (
            forward.1 * up.2 - forward.2 * up.1,
            forward.2 * up.0 - forward.0 * up.2,
            forward.0 * up.1 - forward.1 * up.0,
        );
        let right_len = (right.0 * right.0 + right.1 * right.1 + right.2 * right.2).sqrt();
        let right = if right_len > 0.001 {
            (right.0 / right_len, right.1 / right_len, right.2 / right_len)
        } else {
            (1.0, 0.0, 0.0)
        };
        
        // Camera up direction
        let cam_up = (
            right.1 * forward.2 - right.2 * forward.1,
            right.2 * forward.0 - right.0 * forward.2,
            right.0 * forward.1 - right.1 * forward.0,
        );
        
        // Project point onto camera plane
        let depth = dx * forward.0 + dy * forward.1 + dz * forward.2;
        
        if depth < 0.1 {
            return None; // Behind camera
        }
        
        let proj_x = (dx * right.0 + dy * right.1 + dz * right.2) / depth;
        let proj_y = (dx * cam_up.0 + dy * cam_up.1 + dz * cam_up.2) / depth;
        
        let scale = rect.width().min(rect.height()) * 0.8;
        let screen_x = rect.center().x + proj_x * scale;
        let screen_y = rect.center().y - proj_y * scale;
        
        if rect.contains(egui::pos2(screen_x, screen_y)) {
            Some(egui::pos2(screen_x, screen_y))
        } else {
            None
        }
    }
    
    /// Start editing a timeline step at the given index
    fn start_editing_entry(&mut self, idx: usize) {
        // First, extract all needed data from the project to avoid borrow conflicts
        let step_data = self.project.as_ref().and_then(|project| {
            project.timeline().get(idx).map(|step| {
                (
                    step.operator_id.clone(),
                    step.inputs.clone(),
                    step.outputs.clone(),
                )
            })
        });

        let Some((operator_id, inputs, outputs)) = step_data else { return };

        self.editing_entry_index = Some(idx);
        self.edit_config_values.clear();
        self.edit_lua_script.clear();
        self.edit_input_asset_ids.clear();

        // Populate output asset ID from the first output (primary output)
        self.edit_output_asset_id = outputs.first()
            .cloned()
            .unwrap_or_default();

        // Get operator metadata to understand input types
        let crate_name = operator_id.strip_prefix("op_").unwrap_or(&operator_id);
        let operator_metadata = self.operator_metadata_cached(crate_name);

        if let Some(ref metadata) = operator_metadata {
            // Use metadata to properly decode each input
            for (input_idx, input_meta) in metadata.inputs.iter().enumerate() {
                let input = inputs.get(input_idx);
                match input_meta {
                    OperatorMetadataInput::ModelWASM => {
                        if let Some(ExecutionInput::AssetRef(id)) = input {
                            self.edit_input_asset_ids.push(Some(id.clone()));
                        } else {
                            self.edit_input_asset_ids.push(None);
                        }
                    }
                    OperatorMetadataInput::CBORConfiguration(cddl) => {
                        self.edit_input_asset_ids.push(None); // Placeholder for data inputs
                        // Decode CBOR data to populate config values
                        if let Some(ExecutionInput::Inline(data)) = input {
                            if let Ok(fields) = parse_cddl_record_schema(cddl.as_str()) {
                                if let Ok(cbor_value) = ciborium::from_reader::<CborValue, _>(data.as_slice()) {
                                    if let CborValue::Map(map) = cbor_value {
                                        for (field_name, field_ty) in &fields {
                                            // Find the value in the CBOR map
                                            for (key, value) in &map {
                                                if let CborValue::Text(key_str) = key {
                                                    if key_str == field_name {
                                                        let config_value = match (field_ty, value) {
                                                            (ConfigFieldType::Bool, CborValue::Bool(b)) => {
                                                                Some(ConfigValue::Bool(*b))
                                                            }
                                                            (ConfigFieldType::Int, CborValue::Integer(i)) => {
                                                                Some(ConfigValue::Int(i128::from(*i) as i64))
                                                            }
                                                            (ConfigFieldType::Float, CborValue::Float(f)) => {
                                                                Some(ConfigValue::Float(*f))
                                                            }
                                                            (ConfigFieldType::Float, CborValue::Integer(i)) => {
                                                                Some(ConfigValue::Float(i128::from(*i) as f64))
                                                            }
                                                            (ConfigFieldType::Text, CborValue::Text(t)) => {
                                                                Some(ConfigValue::Text(t.clone()))
                                                            }
                                                            (ConfigFieldType::Enum(_), CborValue::Text(t)) => {
                                                                Some(ConfigValue::Text(t.clone()))
                                                            }
                                                            _ => None,
                                                        };
                                                        if let Some(cv) = config_value {
                                                            self.edit_config_values.insert(field_name.clone(), cv);
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    OperatorMetadataInput::LuaSource(_) => {
                        self.edit_input_asset_ids.push(None); // Placeholder for data inputs
                        // Extract Lua script from data
                        if let Some(ExecutionInput::Inline(data)) = input {
                            if let Ok(script) = std::str::from_utf8(data) {
                                if !script.is_empty() {
                                    self.edit_lua_script = script.to_string();
                                }
                            }
                        }
                    }
                    OperatorMetadataInput::Blob => {
                        // Blob inputs are not editable in the current UI
                        self.edit_input_asset_ids.push(None);
                    }
                    OperatorMetadataInput::VecF64(dim) => {
                        self.edit_input_asset_ids.push(None); // Placeholder for data inputs
                        // Decode VecF64 input - raw bytes (8 bytes per f64, little-endian) or asset reference
                        match input {
                            Some(ExecutionInput::Inline(data)) => {
                                // Decode raw bytes: each f64 is 8 little-endian bytes
                                let values: Vec<f64> = data.chunks_exact(8)
                                    .map(|chunk| {
                                        let arr: [u8; 8] = chunk.try_into().unwrap();
                                        f64::from_le_bytes(arr)
                                    })
                                    .collect();
                                self.edit_vec_inputs.insert(input_idx, (true, values, None));
                            }
                            Some(ExecutionInput::AssetRef(id)) => {
                                // Using asset reference
                                self.edit_vec_inputs.insert(input_idx, (false, vec![0.0; *dim], Some(id.clone())));
                            }
                            None => {
                                // No input, use default literal
                                self.edit_vec_inputs.insert(input_idx, (true, vec![0.0; *dim], None));
                            }
                        }
                    }
                }
            }
        } else {
            // Fallback: no metadata available, use simple extraction
            for input in &inputs {
                match input {
                    ExecutionInput::AssetRef(id) => {
                        self.edit_input_asset_ids.push(Some(id.clone()));
                    }
                    ExecutionInput::Inline(data) => {
                        // Try to interpret as UTF-8 string (Lua script)
                        if let Ok(script) = std::str::from_utf8(data) {
                            if !script.is_empty() && self.edit_lua_script.is_empty() {
                                self.edit_lua_script = script.to_string();
                            }
                        }
                        self.edit_input_asset_ids.push(None);
                    }
                }
            }
        }
    }
    
    /// Close the edit panel
    fn close_edit_panel(&mut self) {
        self.editing_entry_index = None;
        self.edit_config_values.clear();
        self.edit_lua_script.clear();
        self.edit_input_asset_ids.clear();
        self.edit_output_asset_id.clear();
        self.edit_vec_inputs.clear();
    }
    
    /// Show the edit panel UI for the currently selected timeline step
    fn show_edit_panel(&mut self, ui: &mut egui::Ui) {
        let Some(idx) = self.editing_entry_index else { return };

        ui.heading("Edit Step");
        ui.separator();

        // Get step info (we need to be careful about borrowing)
        let step_info = self.project.as_ref().and_then(|p| {
            p.timeline().get(idx).map(|step| {
                (step.operator_id.clone(), step.inputs.clone())
            })
        });

        let Some((operator_id, inputs)) = step_info else {
            ui.label("Step not found or not editable");
            if ui.button("Close").clicked() {
                self.close_edit_panel();
            }
            return;
        };

        ui.label(format!("Editing: {}", operator_id));
        ui.add_space(8.0);

        // Get available input assets for dropdowns (only models)
        let input_asset_ids: Vec<String> = self
            .project
            .as_ref()
            .map(|p| {
                p.declared_assets()
                    .into_iter()
                    .filter_map(|(id, hint)| {
                        if hint == Some(AssetTypeHint::Model) || hint.is_none() {
                            Some(id)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        // Try to get operator metadata for this operation
        // Extract crate name from operator_id (e.g., "op_translate_operator" -> "translate_operator")
        let crate_name = operator_id.strip_prefix("op_").unwrap_or(&operator_id);
        let operator_metadata = self.operator_metadata_cached(crate_name);

        ui.label("Inputs:");
        ui.indent("edit_inputs", |ui| {
            // Show input editors based on operator metadata or existing inputs
            if let Some(ref metadata) = operator_metadata {
                let mut model_input_idx = 0;
                for (input_idx, input_meta) in metadata.inputs.iter().enumerate() {
                    match input_meta {
                        OperatorMetadataInput::ModelWASM => {
                            ui.horizontal(|ui| {
                                ui.label(format!("Model {}:", model_input_idx + 1));

                                // Ensure we have enough slots
                                while self.edit_input_asset_ids.len() <= input_idx {
                                    self.edit_input_asset_ids.push(None);
                                }

                                // Initialize from existing input if not set
                                if self.edit_input_asset_ids[input_idx].is_none() {
                                    if let Some(ExecutionInput::AssetRef(id)) = inputs.get(input_idx) {
                                        self.edit_input_asset_ids[input_idx] = Some(id.clone());
                                    }
                                }

                                let selected = self.edit_input_asset_ids[input_idx]
                                    .as_deref()
                                    .unwrap_or("(none)");
                                egui::ComboBox::from_id_salt(format!("edit_input_{input_idx}"))
                                    .selected_text(selected)
                                    .show_ui(ui, |ui| {
                                        for id in &input_asset_ids {
                                            ui.selectable_value(
                                                &mut self.edit_input_asset_ids[input_idx],
                                                Some(id.clone()),
                                                id,
                                            );
                                        }
                                    });
                            });
                            model_input_idx += 1;
                        }
                        OperatorMetadataInput::CBORConfiguration(cddl) => {
                            ui.separator();
                            ui.label("Configuration:");
                            
                            match parse_cddl_record_schema(cddl.as_str()) {
                                Ok(fields) => {
                                    for (field_name, field_ty) in &fields {
                                        ui.horizontal(|ui| {
                                            ui.label(field_name);
                                            match field_ty {
                                                ConfigFieldType::Bool => {
                                                    let entry = self
                                                        .edit_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert(ConfigValue::Bool(false));
                                                    if let ConfigValue::Bool(b) = entry {
                                                        ui.checkbox(b, "");
                                                    }
                                                }
                                                ConfigFieldType::Int => {
                                                    let entry = self
                                                        .edit_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert(ConfigValue::Int(0));
                                                    if let ConfigValue::Int(i) = entry {
                                                        ui.add(egui::DragValue::new(i));
                                                    }
                                                }
                                                ConfigFieldType::Float => {
                                                    let entry = self
                                                        .edit_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert(ConfigValue::Float(0.0));
                                                    if let ConfigValue::Float(f) = entry {
                                                        ui.add(egui::DragValue::new(f));
                                                    }
                                                }
                                                ConfigFieldType::Text => {
                                                    let entry = self
                                                        .edit_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert_with(|| ConfigValue::Text(String::new()));
                                                    if let ConfigValue::Text(t) = entry {
                                                        ui.text_edit_singleline(t);
                                                    }
                                                }
                                                ConfigFieldType::Enum(options) => {
                                                    let entry = self
                                                        .edit_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert_with(|| {
                                                            ConfigValue::Text(options.first().cloned().unwrap_or_default())
                                                        });

                                                    if let ConfigValue::Text(selected) = entry {
                                                        egui::ComboBox::from_id_salt(format!("edit_cfg_enum_{field_name}"))
                                                            .selected_text(selected.as_str())
                                                            .show_ui(ui, |ui| {
                                                                for opt in options {
                                                                    ui.selectable_value(selected, opt.clone(), opt);
                                                                }
                                                            });
                                                    }
                                                }
                                            }
                                        });
                                    }
                                }
                                Err(e) => {
                                    ui.colored_label(egui::Color32::YELLOW, format!(
                                        "Unsupported configuration schema: {e}"
                                    ));
                                }
                            }
                        }
                        OperatorMetadataInput::LuaSource(template) => {
                            ui.separator();
                            ui.label("Lua Script:");
                            
                            // Initialize with existing script or template
                            if self.edit_lua_script.is_empty() {
                                // Try to get from existing input
                                for input in &inputs {
                                    if let ExecutionInput::Inline(data) = input {
                                        if let Ok(script) = std::str::from_utf8(data) {
                                            if !script.is_empty() {
                                                self.edit_lua_script = script.to_string();
                                                break;
                                            }
                                        }
                                    }
                                }
                                // Fall back to template
                                if self.edit_lua_script.is_empty() {
                                    self.edit_lua_script = template.clone();
                                }
                            }
                            
                            egui::ScrollArea::vertical()
                                .max_height(200.0)
                                .show(ui, |ui| {
                                    ui.add(
                                        egui::TextEdit::multiline(&mut self.edit_lua_script)
                                            .code_editor()
                                            .desired_width(f32::INFINITY)
                                            .desired_rows(10),
                                    );
                                });
                            
                            if ui.button("Reset to template").clicked() {
                                self.edit_lua_script = template.clone();
                            }
                        }
                        OperatorMetadataInput::Blob => {
                            ui.separator();
                            ui.label("Blob Input:");
                            ui.colored_label(egui::Color32::GRAY, "(Binary data - not editable)");
                        }
                        OperatorMetadataInput::VecF64(dim) => {
                            ui.separator();
                            let label = match dim {
                                3 => "Vector (X, Y, Z):",
                                2 => "Vector (X, Y):",
                                4 => "Vector (X, Y, Z, W):",
                                _ => "Vector:",
                            };
                            ui.label(label);

                            // Initialize if not present
                            let entry = self.edit_vec_inputs.entry(input_idx)
                                .or_insert((true, vec![0.0; *dim], None));

                            ui.horizontal(|ui| {
                                ui.radio_value(&mut entry.0, true, "Literal");
                                ui.radio_value(&mut entry.0, false, "Asset");
                            });

                            if entry.0 {
                                // Literal mode - show drag values
                                ui.horizontal(|ui| {
                                    let labels = ["X", "Y", "Z", "W"];
                                    for (i, val) in entry.1.iter_mut().enumerate() {
                                        if i < labels.len() {
                                            ui.label(format!("{}:", labels[i]));
                                        } else {
                                            ui.label(format!("[{}]:", i));
                                        }
                                        ui.add(egui::DragValue::new(val));
                                    }
                                });
                            } else {
                                // Asset mode - show asset selector
                                // Find compatible assets (VecF64 with matching dimension)
                                let vec_asset_ids: Vec<String> = self.project.as_ref()
                                    .map(|p| {
                                        p.imports().iter()
                                            .filter(|a| a.type_hint == Some(AssetTypeHint::VecF64(*dim)))
                                            .map(|a| a.id.clone())
                                            .collect()
                                    })
                                    .unwrap_or_default();

                                let selected = entry.2.as_deref().unwrap_or("(none)");
                                egui::ComboBox::from_id_salt(format!("edit_vec_input_{input_idx}"))
                                    .selected_text(selected)
                                    .show_ui(ui, |ui| {
                                        if ui.selectable_label(entry.2.is_none(), "(none)").clicked() {
                                            entry.2 = None;
                                        }
                                        for id in &vec_asset_ids {
                                            if ui.selectable_label(entry.2.as_ref() == Some(id), id).clicked() {
                                                entry.2 = Some(id.clone());
                                            }
                                        }
                                    });

                                if vec_asset_ids.is_empty() {
                                    ui.colored_label(egui::Color32::YELLOW, "(no compatible assets)");
                                }
                            }
                        }
                    }
                }
            } else {
                // No metadata available, show basic input display
                for (i, input) in inputs.iter().enumerate() {
                    ui.label(format!("{}: {}", i + 1, input.display()));
                }
            }
        });
        
        ui.add_space(8.0);
        ui.separator();
        
        // Output name editing
        ui.label("Output:");
        ui.indent("edit_output", |ui| {
            ui.horizontal(|ui| {
                ui.label("Output asset name:");
                ui.text_edit_singleline(&mut self.edit_output_asset_id);
            });
        });
        
        ui.add_space(16.0);
        ui.separator();
        
        // Action buttons
        let can_apply = !self.edit_output_asset_id.trim().is_empty();
        ui.horizontal(|ui| {
            if ui.add_enabled(can_apply, egui::Button::new("Apply Changes")).clicked() {
                self.apply_edit_changes();
            }
            if ui.button("Cancel").clicked() {
                self.close_edit_panel();
            }
        });
    }
    
    /// Apply the changes from the edit panel to the timeline step
    fn apply_edit_changes(&mut self) {
        let Some(idx) = self.editing_entry_index else { return };

        // Get operator metadata for building new inputs
        let operator_id = self.project.as_ref().and_then(|p| {
            p.timeline().get(idx).map(|step| step.operator_id.clone())
        });

        let Some(operator_id) = operator_id else { return };
        let crate_name = operator_id.strip_prefix("op_").unwrap_or(&operator_id);
        let operator_metadata = self.operator_metadata_cached(crate_name);

        // Build new inputs based on edit state
        let new_inputs: Vec<ExecutionInput> = if let Some(ref metadata) = operator_metadata {
            let mut inputs = Vec::new();
            let mut model_input_idx = 0;

            for input_meta in &metadata.inputs {
                match input_meta {
                    OperatorMetadataInput::ModelWASM => {
                        let asset_id = self.edit_input_asset_ids
                            .get(model_input_idx)
                            .and_then(|o| o.clone())
                            .unwrap_or_default();
                        inputs.push(ExecutionInput::AssetRef(asset_id));
                        model_input_idx += 1;
                    }
                    OperatorMetadataInput::CBORConfiguration(cddl) => {
                        let fields = parse_cddl_record_schema(cddl.as_str()).unwrap_or_default();
                        let bytes = encode_config_map_to_cbor(&fields, &self.edit_config_values)
                            .unwrap_or_default();
                        inputs.push(ExecutionInput::Inline(bytes));
                    }
                    OperatorMetadataInput::LuaSource(_) => {
                        let script_bytes = self.edit_lua_script.as_bytes().to_vec();
                        inputs.push(ExecutionInput::Inline(script_bytes));
                    }
                    OperatorMetadataInput::Blob => {
                        // Keep existing blob data (not editable)
                        // We need to preserve the original inline data from the step
                        if let Some(ref project) = self.project {
                            if let Some(step) = project.timeline().get(idx) {
                                if let Some(ExecutionInput::Inline(data)) = step.inputs.get(inputs.len()) {
                                    inputs.push(ExecutionInput::Inline(data.clone()));
                                } else {
                                    inputs.push(ExecutionInput::Inline(Vec::new()));
                                }
                            } else {
                                inputs.push(ExecutionInput::Inline(Vec::new()));
                            }
                        } else {
                            inputs.push(ExecutionInput::Inline(Vec::new()));
                        }
                    }
                    OperatorMetadataInput::VecF64(dim) => {
                        let input_idx = inputs.len();
                        if let Some((use_literal, values, asset_ref)) = self.edit_vec_inputs.get(&input_idx) {
                            if *use_literal {
                                // Encode as raw bytes: each f64 as 8 little-endian bytes
                                let bytes: Vec<u8> = values.iter()
                                    .flat_map(|v| v.to_le_bytes())
                                    .collect();
                                inputs.push(ExecutionInput::Inline(bytes));
                            } else if let Some(asset_id) = asset_ref {
                                inputs.push(ExecutionInput::AssetRef(asset_id.clone()));
                            } else {
                                // No asset selected, use default literal
                                let bytes: Vec<u8> = (0..*dim)
                                    .flat_map(|_| 0.0_f64.to_le_bytes())
                                    .collect();
                                inputs.push(ExecutionInput::Inline(bytes));
                            }
                        } else {
                            // No entry, use default literal
                            let bytes: Vec<u8> = (0..*dim)
                                .flat_map(|_| 0.0_f64.to_le_bytes())
                                .collect();
                            inputs.push(ExecutionInput::Inline(bytes));
                        }
                    }
                }
            }
            inputs
        } else {
            // No metadata, keep existing inputs
            return;
        };

        // Update the timeline step
        if let Some(ref mut project) = self.project {
            // Get the old output ID to update exports
            let old_output_id = project.timeline().get(idx).and_then(|step| step.outputs.first().cloned());
            let new_output_id = self.edit_output_asset_id.trim().to_string();

            if let Some(step) = project.timeline_mut().get_mut(idx) {
                // Update inputs
                step.inputs = new_inputs;

                // Update output IDs
                let new_outputs: Vec<String> = step.outputs.iter().enumerate().map(|(i, _old_out)| {
                    if i == 0 {
                        new_output_id.clone()
                    } else {
                        format!("{}_{}", new_output_id, i)
                    }
                }).collect();
                step.outputs = new_outputs;
            }

            // Update the corresponding export if the output name changed
            if let Some(old_id) = old_output_id {
                if old_id != new_output_id {
                    for export_id in project.exports_mut().iter_mut() {
                        if *export_id == old_id {
                            *export_id = new_output_id.clone();
                            break;
                        }
                    }
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.project_path = None; // Mark as modified
        }
        self.run_project();
        self.close_edit_panel();
    }
}

impl eframe::App for VolumetricApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll for completed background tasks
        self.poll_background_tasks();

        // Web-specific: hide loading spinner and poll async file operations
        #[cfg(target_arch = "wasm32")]
        {
            // Hide the loading spinner on first frame
            if !self.loading_hidden {
                self.loading_hidden = true;
                if let Some(window) = web_sys::window() {
                    if let Some(document) = window.document() {
                        if let Some(body) = document.body() {
                            let _ = body.class_list().add_1("loaded");
                        }
                    }
                }
            }

            // Poll pending WASM import
            if let Some(ref promise) = self.pending_wasm_import {
                if let Some(result) = promise.ready() {
                    let result = result.clone();
                    self.pending_wasm_import = None;
                    if let Some((name, wasm_bytes)) = result {
                        let asset_id = name.trim_end_matches(".wasm").to_string();
                        if self.project.is_none() {
                            self.project = Some(Project::new());
                        }
                        if let Some(ref mut project) = self.project {
                            project.insert_model(&asset_id, wasm_bytes);
                        }
                        self.run_project();
                    }
                }
            }

            // Poll pending project load
            if let Some(ref promise) = self.pending_project_load {
                if let Some(result) = promise.ready() {
                    let result = result.clone();
                    self.pending_project_load = None;
                    if let Some(bytes) = result {
                        match Project::from_cbor(&bytes) {
                            Ok(project) => {
                                self.project = Some(project);
                                self.run_project();
                            }
                            Err(e) => {
                                self.error_message = Some(format!("Failed to load project: {}", e));
                            }
                        }
                    }
                }
            }

            // Poll pending STL import
            if let Some(ref promise) = self.pending_stl_import {
                if let Some(result) = promise.ready() {
                    let result = result.clone();
                    self.pending_stl_import = None;
                    if let Some(stl_bytes) = result {
                        self.process_stl_import(stl_bytes);
                    }
                }
            }

            // Poll pending heightmap import
            if let Some(ref promise) = self.pending_heightmap_import {
                if let Some(result) = promise.ready() {
                    let result = result.clone();
                    self.pending_heightmap_import = None;
                    if let Some(image_bytes) = result {
                        self.process_heightmap_import(image_bytes);
                    }
                }
            }

            // Request repaint while async operations are pending
            if self.pending_wasm_import.is_some() || self.pending_project_load.is_some() || self.pending_stl_import.is_some() || self.pending_heightmap_import.is_some() {
                ctx.request_repaint();
            }
        }

        // Process pending STL import (native)
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(stl_bytes) = self.pending_stl_data.take() {
            self.process_stl_import(stl_bytes);
        }

        // Process pending heightmap import (native)
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(image_bytes) = self.pending_heightmap_data.take() {
            self.process_heightmap_import(image_bytes);
        }

        // Request continuous repaints while background tasks are running
        if self.is_busy() {
            ctx.request_repaint();
        }

        // Check if any assets need resampling
        if self.any_needs_resample() {
            self.resample_all_assets();
        }

        // Left panel with controls
        egui::SidePanel::left("controls")
            .default_width(250.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        ui.heading("Volumetric Renderer");
                        ui.separator();

                        ui.label("Project");
                        if let Some(time) = self.last_evaluation_time {
                            ui.weak(format!("Evaluated in {:.2}ms", time * 1000.0));
                        }
                        ui.horizontal(|ui| {
                            #[cfg(not(target_arch = "wasm32"))]
                            let label = match &self.project_path {
                                Some(p) => p.display().to_string(),
                                None => match &self.project {
                                    Some(_) => "(unsaved project)".to_string(),
                                    None => "(none loaded)".to_string(),
                                },
                            };
                            #[cfg(target_arch = "wasm32")]
                            let label = match &self.project {
                                Some(_) => "(in-memory project)".to_string(),
                                None => "(none loaded)".to_string(),
                            };
                            ui.label(label);
                        });


                        ui.horizontal(|ui| {
                            let can_unload = self.project.is_some();
                            if ui
                                .add_enabled(can_unload, egui::Button::new("Unload Project"))
                                .clicked()
                            {
                                self.project = None;
                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    self.project_path = None;
                                }
                                self.asset_render_data.clear();
                                self.exported_assets.clear();
                                self.error_message = None;
                            }
                        });

                        // Show progress indicator when busy
                        let mut should_cancel = false;
                        if let Some(ref op) = self.in_progress_operation {
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label(&op.description);
                            });
                            let elapsed = op.start_time.elapsed().as_secs_f32();
                            ui.weak(format!("{:.1}s elapsed", elapsed));
                            if ui.button("Cancel").clicked() {
                                should_cancel = true;
                            }
                        }
                        if should_cancel {
                            self.cancel_background_operation();
                        }

                        ui.separator();
                        ui.label("Shading");
                        ui.checkbox(&mut self.ssao_enabled, "SSAO");
                        ui.add_enabled(
                            self.ssao_enabled,
                            egui::Slider::new(&mut self.ssao_radius, 0.005..=0.5)
                                .logarithmic(true)
                                .text("SSAO radius"),
                        );
                        ui.add_enabled(
                            self.ssao_enabled,
                            egui::Slider::new(&mut self.ssao_bias, 0.0001..=0.02)
                                .logarithmic(true)
                                .text("SSAO bias"),
                        );
                        ui.add_enabled(
                            self.ssao_enabled,
                            egui::Slider::new(&mut self.ssao_strength, 0.5..=4.0)
                                .text("SSAO strength"),
                        );

                        ui.separator();
                        ui.label("Camera");
                        egui::ComboBox::from_label("Control Scheme")
                            .selected_text(self.camera_control_scheme.name())
                            .show_ui(ui, |ui| {
                                for &scheme in renderer::CameraControlScheme::ALL {
                                    ui.selectable_value(
                                        &mut self.camera_control_scheme,
                                        scheme,
                                        scheme.name(),
                                    );
                                }
                            });
                        if ui.button("Reset Camera").clicked() {
                            self.camera_theta = std::f32::consts::FRAC_PI_4;
                            self.camera_phi = std::f32::consts::FRAC_PI_4;
                            self.camera_radius = 4.0;
                            self.camera_target = glam::Vec3::ZERO;
                        }

                        ui.separator();
                        ui.label("Sample Cloud");
                        #[cfg(not(target_arch = "wasm32"))]
                        if ui.button("Load Sample Cloud…").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Sample Cloud", &["cbor"])
                                .pick_file()
                            {
                                match SampleCloudDump::load(&path) {
                                    Ok(dump) => {
                                        self.sample_cloud = Some(dump);
                                        self.sample_cloud_set_index = 0;
                                        self.sample_cloud_path = Some(path);
                                        self.error_message = None;
                                    }
                                    Err(e) => {
                                        self.error_message = Some(format!("Failed to load sample cloud: {e}"));
                                    }
                                }
                            }
                        }

                        if let Some(ref dump) = self.sample_cloud {
                            ui.checkbox(&mut self.sample_cloud_enabled, "Show sample cloud");
                            let set_count = dump.sets.len();
                            if set_count > 0 {
                                let max_index = set_count.saturating_sub(1);
                                ui.add(
                                    egui::Slider::new(&mut self.sample_cloud_set_index, 0..=max_index)
                                        .text("Set index"),
                                );
                                if let Some(set) = dump.sets.get(self.sample_cloud_set_index) {
                                    let label = set.label.as_deref().unwrap_or("(unnamed)");
                                    ui.weak(format!("id={} • {}", set.id, label));
                                    ui.weak(format!("points={}", set.points.len()));
                                }
                            } else {
                                ui.weak("No sample sets found");
                            }

                            egui::ComboBox::from_label("View")
                                .selected_text(self.sample_cloud_mode.label())
                                .show_ui(ui, |ui| {
                                    for mode in SampleCloudViewMode::ALL {
                                        ui.selectable_value(&mut self.sample_cloud_mode, mode, mode.label());
                                    }
                                });

                            ui.add(
                                egui::Slider::new(&mut self.sample_cloud_point_size, 1.0..=16.0)
                                    .text("Point size"),
                            );

                            // Focus camera on vertex button
                            if let Some(set) = dump.sets.get(self.sample_cloud_set_index) {
                                if ui.button("🎯 Focus on Vertex").clicked() {
                                    self.camera_target = glam::Vec3::from_array(set.vertex);
                                    // Use a reasonable distance based on cell size
                                    self.camera_radius = 0.15;
                                }
                            }
                        } else {
                            ui.weak("No sample cloud loaded");
                        }

                        // Project file operations (native only - uses filesystem)
                        #[cfg(not(target_arch = "wasm32"))]
                        ui.horizontal(|ui| {
                            if ui.button("Open Project…").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("Project", &["vproj"])
                                    .pick_file()
                                {
                                    match Project::load_from_file(&path) {
                                        Ok(project) => {
                                            self.project = Some(project);
                                            self.project_path = Some(path);
                                            self.run_project();
                                        }
                                        Err(e) => {
                                            self.error_message =
                                                Some(format!("Failed to load project: {}", e));
                                        }
                                    }
                                }
                            }

                            let can_save = self.project.is_some();
                            if ui
                                .add_enabled(can_save, egui::Button::new("Save Project…"))
                                .clicked()
                            {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("Project", &["vproj"])
                                    .save_file()
                                {
                                    if let Some(ref project) = self.project {
                                        match project.save_to_file(&path) {
                                            Ok(()) => {
                                                self.project_path = Some(path);
                                                self.error_message = None;
                                            }
                                            Err(e) => {
                                                self.error_message =
                                                    Some(format!("Failed to save project: {}", e));
                                            }
                                        }
                                    }
                                }
                            }
                        });

                        // Project file operations (web version - uses async file picker)
                        #[cfg(target_arch = "wasm32")]
                        ui.horizontal(|ui| {
                            let is_loading = self.pending_project_load.is_some();
                            if ui.add_enabled(!is_loading, egui::Button::new(if is_loading { "⏳ Loading…" } else { "Open Project…" })).clicked() {
                                self.pending_project_load = Some(Promise::spawn_local(async {
                                    let file = rfd::AsyncFileDialog::new()
                                        .add_filter("Project", &["vproj"])
                                        .pick_file()
                                        .await?;
                                    Some(file.read().await)
                                }));
                            }

                            let can_save = self.project.is_some();
                            if ui.add_enabled(can_save, egui::Button::new("Save Project…")).clicked() {
                                if let Some(ref project) = self.project {
                                    match project.to_cbor() {
                                        Ok(bytes) => {
                                            // Trigger browser download
                                            if let Some(window) = web_sys::window() {
                                                if let Some(document) = window.document() {
                                                    let uint8_array = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
                                                    uint8_array.copy_from(&bytes);
                                                    let array = js_sys::Array::new();
                                                    array.push(&uint8_array.buffer());
                                                    if let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence(&array) {
                                                        if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
                                                            if let Ok(elem) = document.create_element("a") {
                                                                if let Ok(anchor) = elem.dyn_into::<web_sys::HtmlAnchorElement>() {
                                                                    anchor.set_href(&url);
                                                                    anchor.set_download("project.vproj");
                                                                    anchor.click();
                                                                    let _ = web_sys::Url::revoke_object_url(&url);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            self.error_message = Some(format!("Failed to save project: {}", e));
                                        }
                                    }
                                }
                            }
                        });

                        ui.separator();

                        // ═══════════════════════════════════════════════════════════════
                        // TOOLBOX PANEL - Models Section
                        // ═══════════════════════════════════════════════════════════════
                        ui.collapsing("📦 Models", |ui| {
                            ui.add_space(4.0);
                            
                            // Track which model to add (if any)
                            let mut model_to_add: Option<&str> = None;
                            let btn_width = ui.available_width();
                            
                            // Primitive shapes
                            ui.label("Primitives");
                            if ui.add(egui::Button::new("● Sphere").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                                model_to_add = Some("simple_sphere_model");
                            }
                            if ui.add(egui::Button::new("◎ Torus").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                                model_to_add = Some("simple_torus_model");
                            }
                            if ui.add(egui::Button::new("▢ Rounded Box").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                                model_to_add = Some("rounded_box_model");
                            }
                            
                            ui.add_space(8.0);
                            ui.label("Complex");
                            if ui.add(egui::Button::new("✦ Gyroid Lattice").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                                model_to_add = Some("gyroid_lattice_model");
                            }
                            if ui.add(egui::Button::new("❋ Mandelbulb").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                                model_to_add = Some("mandelbulb_model");
                            }
                            
                            // Handle model addition - uses bundled assets with optional filesystem fallback in debug
                            if let Some(crate_name) = model_to_add {
                                // First try bundled assets (works on both native and web)
                                if let Some(wasm_bytes) = get_bundled_model(crate_name) {
                                    if self.project.is_none() {
                                        self.project = Some(Project::new());
                                    }
                                    if let Some(ref mut project) = self.project {
                                        project.insert_model(crate_name, wasm_bytes.to_vec());
                                    }
                                    #[cfg(not(target_arch = "wasm32"))]
                                    { self.project_path = None; }
                                    self.run_project();
                                } else {
                                    // Filesystem fallback for native debug builds only
                                    #[cfg(all(not(target_arch = "wasm32"), debug_assertions))]
                                    {
                                        match demo_wasm_path(crate_name) {
                                            Some(path) => {
                                                match fs::read(&path) {
                                                    Ok(wasm_bytes) => {
                                                        let asset_id = path
                                                            .file_stem()
                                                            .and_then(|s| s.to_str())
                                                            .unwrap_or("model")
                                                            .to_string();
                                                        if self.project.is_none() {
                                                            self.project = Some(Project::new());
                                                        }
                                                        if let Some(ref mut project) = self.project {
                                                            project.insert_model(asset_id.as_str(), wasm_bytes);
                                                        }
                                                        self.project_path = None;
                                                        self.run_project();
                                                    }
                                                    Err(e) => {
                                                        self.error_message =
                                                            Some(format!("Failed to read WASM file: {}", e));
                                                    }
                                                }
                                            }
                                            None => {
                                                self.error_message = Some(format!(
                                                    "Model WASM not found for '{crate_name}'. Build it first with: cargo build --release --target wasm32-unknown-unknown -p {crate_name}"
                                                ));
                                            }
                                        }
                                    }
                                    #[cfg(any(target_arch = "wasm32", not(debug_assertions)))]
                                    {
                                        self.error_message = Some(format!(
                                            "Model '{}' not bundled. Rebuild with WASM assets available.",
                                            crate_name
                                        ));
                                    }
                                }
                            }
                            
                            ui.add_space(8.0);
                            ui.separator();

                            // Import custom WASM (native only for now)
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui
                                .add(egui::Button::new("📁 Import WASM…").min_size(egui::vec2(btn_width, 28.0)))
                                .clicked()
                            {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("WASM", &["wasm"])
                                    .pick_file()
                                {
                                    match fs::read(&path) {
                                        Ok(wasm_bytes) => {
                                            let asset_id = path
                                                .file_stem()
                                                .and_then(|s| s.to_str())
                                                .unwrap_or("model")
                                                .to_string();
                                            if self.project.is_none() {
                                                self.project = Some(Project::new());
                                            }
                                            if let Some(ref mut project) = self.project {
                                                project.insert_model(asset_id.as_str(), wasm_bytes);
                                            }
                                            self.project_path = None;
                                            self.run_project();
                                        }
                                        Err(e) => {
                                            self.error_message =
                                                Some(format!("Failed to read WASM file: {}", e));
                                        }
                                    }
                                }
                            }

                            // Import custom WASM (web version using async file picker)
                            #[cfg(target_arch = "wasm32")]
                            {
                                let is_picking = self.pending_wasm_import.is_some();
                                if ui
                                    .add_enabled(!is_picking, egui::Button::new(if is_picking { "⏳ Picking…" } else { "📁 Import WASM…" }).min_size(egui::vec2(btn_width, 28.0)))
                                    .clicked()
                                {
                                    self.pending_wasm_import = Some(Promise::spawn_local(async {
                                        let file = rfd::AsyncFileDialog::new()
                                            .add_filter("WASM", &["wasm"])
                                            .pick_file()
                                            .await?;
                                        let data = file.read().await;
                                        let name = file.file_name();
                                        Some((name, data))
                                    }));
                                }
                            }

                            ui.add_space(4.0);

                            // Import STL file (native version)
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui
                                .add(egui::Button::new("📐 Import STL…").min_size(egui::vec2(btn_width, 28.0)))
                                .clicked()
                            {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("STL", &["stl"])
                                    .pick_file()
                                {
                                    match fs::read(&path) {
                                        Ok(stl_bytes) => {
                                            self.pending_stl_data = Some(stl_bytes);
                                        }
                                        Err(e) => {
                                            self.error_message =
                                                Some(format!("Failed to read STL file: {}", e));
                                        }
                                    }
                                }
                            }

                            // Import STL file (web version using async file picker)
                            #[cfg(target_arch = "wasm32")]
                            {
                                let is_picking = self.pending_stl_import.is_some();
                                if ui
                                    .add_enabled(!is_picking, egui::Button::new(if is_picking { "⏳ Picking…" } else { "📐 Import STL…" }).min_size(egui::vec2(btn_width, 28.0)))
                                    .clicked()
                                {
                                    self.pending_stl_import = Some(Promise::spawn_local(async {
                                        let file = rfd::AsyncFileDialog::new()
                                            .add_filter("STL", &["stl"])
                                            .pick_file()
                                            .await?;
                                        Some(file.read().await)
                                    }));
                                }
                            }

                            // Import heightmap image (native version)
                            #[cfg(not(target_arch = "wasm32"))]
                            if ui
                                .add(egui::Button::new("🏔 Import Heightmap…").min_size(egui::vec2(btn_width, 28.0)))
                                .clicked()
                            {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "gif"])
                                    .pick_file()
                                {
                                    match fs::read(&path) {
                                        Ok(image_bytes) => {
                                            self.pending_heightmap_data = Some(image_bytes);
                                        }
                                        Err(e) => {
                                            self.error_message =
                                                Some(format!("Failed to read image file: {}", e));
                                        }
                                    }
                                }
                            }

                            // Import heightmap image (web version using async file picker)
                            #[cfg(target_arch = "wasm32")]
                            {
                                let is_picking = self.pending_heightmap_import.is_some();
                                if ui
                                    .add_enabled(!is_picking, egui::Button::new(if is_picking { "⏳ Picking…" } else { "🏔 Import Heightmap…" }).min_size(egui::vec2(btn_width, 28.0)))
                                    .clicked()
                                {
                                    self.pending_heightmap_import = Some(Promise::spawn_local(async {
                                        let file = rfd::AsyncFileDialog::new()
                                            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "gif"])
                                            .pick_file()
                                            .await?;
                                        Some(file.read().await)
                                    }));
                                }
                            }
                        });
                // ═══════════════════════════════════════════════════════════════
                // TOOLBOX PANEL - Operators Section
                // ═══════════════════════════════════════════════════════════════
                ui.collapsing("⚙ Operators", |ui| {
                    ui.add_space(4.0);
                    
                    // Get available input assets for operators (only models)
                    let input_asset_ids: Vec<String> = self
                        .project
                        .as_ref()
                        .map(|p| {
                            p.declared_assets()
                                .into_iter()
                                .filter_map(|(id, hint)| {
                                    if hint == Some(AssetTypeHint::Model) || hint.is_none() {
                                        Some(id)
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    
                    // Update default inputs
                    let needs_default_input = match self.operation_input_asset_id.as_deref() {
                        Some(id) => !input_asset_ids.iter().any(|x| x == id),
                        None => true,
                    };
                    if needs_default_input {
                        self.operation_input_asset_id = input_asset_ids.first().cloned();
                    }
                    
                    let needs_default_input_b = match self.operation_input_asset_id_b.as_deref() {
                        Some(id) => !input_asset_ids.iter().any(|x| x == id),
                        None => true,
                    };
                    if needs_default_input_b {
                        self.operation_input_asset_id_b = input_asset_ids
                            .get(1)
                            .cloned()
                            .or_else(|| self.operation_input_asset_id.clone());
                    }
                    
                    // Track which operator to add (if any)
                    let mut operator_to_add: Option<&str> = None;
                    let btn_width = ui.available_width();
                    
                    // Transform operators
                    ui.label("Transform");
                    if ui.add(egui::Button::new("↔ Translate").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                        operator_to_add = Some("translate_operator");
                    }
                    if ui.add(egui::Button::new("⟳ Rotate").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                        operator_to_add = Some("rotation_operator");
                    }
                    if ui.add(egui::Button::new("⤢ Scale").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                        operator_to_add = Some("scale_operator");
                    }
                    
                    ui.add_space(8.0);
                    ui.label("Combine");
                    if ui.add(egui::Button::new("⊕ Boolean").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                        operator_to_add = Some("boolean_operator");
                    }
                    
                    ui.add_space(8.0);
                    ui.label("Emitters");
                    if ui.add(egui::Button::new("▭ Rectangular Prism").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                        operator_to_add = Some("rectangular_prism_operator");
                    }

                    ui.add_space(8.0);
                    ui.label("Advanced");
                    if ui.add(egui::Button::new("📜 Lua Script").min_size(egui::vec2(btn_width, 28.0))).clicked() {
                        operator_to_add = Some("lua_script_operator");
                    }
                    
                    // Handle operator addition
                    if let Some(crate_name) = operator_to_add {
                        // Check if we have a project and required inputs
                        let operator_metadata = self.operator_metadata_cached(crate_name);
                        let model_input_count = operator_metadata
                            .as_ref()
                            .map(|m| {
                                m.inputs
                                    .iter()
                                    .filter(|i| matches!(i, OperatorMetadataInput::ModelWASM))
                                    .count()
                            })
                            .unwrap_or(1);
                        
                        let has_required_inputs = if model_input_count >= 2 {
                            self.operation_input_asset_id.is_some() && self.operation_input_asset_id_b.is_some()
                        } else if model_input_count == 1 {
                            self.operation_input_asset_id.is_some()
                        } else {
                            true
                        };
                        
                        if self.project.is_none() {
                            self.error_message = Some("Please add a model first before adding operators.".to_string());
                        } else if !has_required_inputs {
                            self.error_message = Some("Not enough input models available for this operator.".to_string());
                        } else {
                            // Helper closure to add operator with given WASM bytes
                            let add_operator = |app: &mut VolumetricApp, wasm_bytes: Vec<u8>| {
                                let input_id_a = app.operation_input_asset_id.clone().unwrap_or_default();
                                let input_id_b = app
                                    .operation_input_asset_id_b
                                    .clone()
                                    .unwrap_or_else(|| input_id_a.clone());
                                let output_id = app.project.as_ref()
                                    .map(|p| p.default_output_name(crate_name, Some(&input_id_a)))
                                    .unwrap_or_else(|| format!("{}_output", crate_name));
                                let op_asset_id = format!("op_{crate_name}");

                                // Build inputs based on operator metadata
                                let (inputs, output_ids): (Vec<ExecutionInput>, Vec<String>) = match app.operator_metadata_cached(crate_name) {
                                    Some(metadata) => {
                                        let mut inputs = Vec::with_capacity(metadata.inputs.len());
                                        let mut model_inputs_iter = [input_id_a.clone(), input_id_b.clone()].into_iter();
                                        for input in metadata.inputs.iter() {
                                            match input {
                                                OperatorMetadataInput::ModelWASM => {
                                                    let id = model_inputs_iter
                                                        .next()
                                                        .unwrap_or_else(|| input_id_a.clone());
                                                    inputs.push(ExecutionInput::AssetRef(id));
                                                }
                                                OperatorMetadataInput::CBORConfiguration(cddl) => {
                                                    let fields = parse_cddl_record_schema(cddl.as_str()).unwrap_or_default();
                                                    let bytes = encode_config_map_to_cbor(&fields, &app.operation_config_values)
                                                        .unwrap_or_default();
                                                    inputs.push(ExecutionInput::Inline(bytes));
                                                }
                                                OperatorMetadataInput::LuaSource(_) => {
                                                    let script_bytes = app.operation_lua_script.as_bytes().to_vec();
                                                    inputs.push(ExecutionInput::Inline(script_bytes));
                                                }
                                                OperatorMetadataInput::Blob => {
                                                    // Blob inputs should be handled specially (via file picker)
                                                    // For now, just add empty bytes - the STL import is handled
                                                    // through the dedicated "Import STL" button instead
                                                    inputs.push(ExecutionInput::Inline(Vec::new()));
                                                }
                                                OperatorMetadataInput::VecF64(dim) => {
                                                    // Default VecF64 values (zeros) as raw bytes
                                                    let bytes: Vec<u8> = (0..*dim)
                                                        .flat_map(|_| 0.0_f64.to_le_bytes())
                                                        .collect();
                                                    inputs.push(ExecutionInput::Inline(bytes));
                                                }
                                            }
                                        }

                                        // Build output IDs (just strings now, no type needed)
                                        let outputs: Vec<String> = metadata.outputs.iter().enumerate().map(|(idx, _)| {
                                            if idx == 0 {
                                                output_id.clone()
                                            } else {
                                                format!("{output_id}_{idx}")
                                            }
                                        }).collect();

                                        (inputs, outputs)
                                    }
                                    None => (
                                        vec![ExecutionInput::AssetRef(input_id_a.clone())],
                                        vec![output_id.clone()],
                                    ),
                                };

                                let mut new_step_idx: Option<usize> = None;
                                if let Some(ref mut project) = app.project {
                                    let count_before = project.timeline().len();

                                    project.insert_operation(
                                        op_asset_id.as_str(),
                                        wasm_bytes,
                                        inputs,
                                        output_ids,
                                        output_id,
                                    );

                                    // The new step is at the end of the timeline
                                    if project.timeline().len() > count_before {
                                        new_step_idx = Some(project.timeline().len() - 1);
                                    }
                                }

                                app.run_project();

                                if let Some(idx) = new_step_idx {
                                    app.start_editing_entry(idx);
                                }
                            };

                            // First try bundled operators (works on both native and web)
                            if let Some(wasm_bytes) = get_bundled_operator(crate_name) {
                                add_operator(self, wasm_bytes.to_vec());
                            } else {
                                // Filesystem fallback for native debug builds only
                                #[cfg(all(not(target_arch = "wasm32"), debug_assertions))]
                                match operation_wasm_path(crate_name) {
                                    Some(path) => match fs::read(&path) {
                                        Ok(wasm_bytes) => {
                                            add_operator(self, wasm_bytes);
                                        }
                                        Err(e) => {
                                            self.error_message =
                                                Some(format!("Failed to read operation WASM file: {e}"));
                                        }
                                    },
                                    None => {
                                        self.error_message = Some(format!(
                                            "Operator WASM not found for '{crate_name}'. Build it first with: cargo build --release --target wasm32-unknown-unknown -p {crate_name}"
                                        ));
                                    }
                                }

                                #[cfg(any(target_arch = "wasm32", not(debug_assertions)))]
                                {
                                    self.error_message = Some(format!(
                                        "Operator '{}' not bundled. Rebuild with WASM assets available.",
                                        crate_name
                                    ));
                                }
                            }
                        }
                    }
                });

                ui.separator();
                // Track actions to perform after the UI loop (to avoid borrow conflicts)
                let mut step_to_edit: Option<usize> = None;
                let mut step_to_delete: Option<usize> = None;
                let mut step_to_move_up: Option<usize> = None;
                let mut step_to_move_down: Option<usize> = None;
                let mut import_to_delete: Option<usize> = None;
                let mut export_to_delete: Option<usize> = None;
                let mut export_to_add: Option<String> = None;

                egui::CollapsingHeader::new("Project Timeline")
                    .default_open(true)
                    .show(ui, |ui| {
                        // Auto-rebuild controls
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.auto_rebuild, "Auto Rebuild");
                            if !self.auto_rebuild {
                                if ui.button("⟳ Rebuild").clicked() {
                                    self.run_project();
                                }
                            }
                        });
                        ui.add_space(4.0);

                        if let Some(ref project) = self.project {
                            // Show imports with delete buttons
                            let imports = project.imports();
                            if !imports.is_empty() {
                                ui.label("Imports:");
                                for (idx, import) in imports.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        let icon = match import.type_hint {
                                            Some(AssetTypeHint::Model) => "📦",
                                            Some(AssetTypeHint::Operator) => "⚙️",
                                            _ => "📄",
                                        };
                                        ui.weak(format!("{} {}", icon, import.id));
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            if ui.small_button("🗑").on_hover_text("Remove import").clicked() {
                                                import_to_delete = Some(idx);
                                            }
                                        });
                                    });
                                }
                                ui.add_space(4.0);
                            }

                            // Show timeline steps
                            let timeline = project.timeline();
                            if timeline.is_empty() {
                                ui.weak("No operations in timeline");
                            } else {
                                ui.label("Timeline:");
                                egui::ScrollArea::vertical()
                                    .max_height(200.0)
                                    .show(ui, |ui| {
                                        for (idx, step) in timeline.iter().enumerate() {
                                            ui.group(|ui| {
                                                ui.horizontal(|ui| {
                                                    ui.label(format!("{}.", idx + 1));
                                                    ui.vertical(|ui| {
                                                        ui.label(format!("▶ {}", step.operator_id));
                                                        if !step.inputs.is_empty() {
                                                            ui.indent("inputs", |ui| {
                                                                for input in &step.inputs {
                                                                    ui.weak(format!("← {}", input.display()));
                                                                }
                                                            });
                                                        }
                                                        if !step.outputs.is_empty() {
                                                            ui.indent("outputs", |ui| {
                                                                for output in &step.outputs {
                                                                    ui.weak(format!("→ {}", output));
                                                                }
                                                            });
                                                        }
                                                    });

                                                    // Add Edit, Delete, and Move buttons
                                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                        if ui.small_button("🗑").on_hover_text("Delete step").clicked() {
                                                            step_to_delete = Some(idx);
                                                        }
                                                        if ui.small_button("✏").on_hover_text("Edit step").clicked() {
                                                            step_to_edit = Some(idx);
                                                        }
                                                        let is_last = idx == timeline.len() - 1;
                                                        if ui.add_enabled(!is_last, egui::Button::new("▼").small())
                                                            .on_hover_text("Move down")
                                                            .clicked()
                                                        {
                                                            step_to_move_down = Some(idx);
                                                        }
                                                        if ui.add_enabled(idx > 0, egui::Button::new("▲").small())
                                                            .on_hover_text("Move up")
                                                            .clicked()
                                                        {
                                                            step_to_move_up = Some(idx);
                                                        }
                                                    });
                                                });
                                            });
                                            ui.add_space(2.0);
                                        }
                                    });
                            }

                            // Show exports with delete buttons and add functionality
                            ui.add_space(4.0);
                            ui.label("Exports:");
                            let exports = project.exports();
                            if exports.is_empty() {
                                ui.weak("  (no exports)");
                            } else {
                                for (idx, export_id) in exports.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        ui.weak(format!("📤 {}", export_id));
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            if ui.small_button("🗑").on_hover_text("Remove export").clicked() {
                                                export_to_delete = Some(idx);
                                            }
                                        });
                                    });
                                }
                            }

                            // Add export dropdown
                            let declared = project.declared_assets();
                            let current_exports: std::collections::HashSet<&str> = exports.iter().map(|s| s.as_str()).collect();
                            let available_for_export: Vec<&str> = declared
                                .iter()
                                .filter(|(id, _)| !current_exports.contains(id.as_str()))
                                .map(|(id, _)| id.as_str())
                                .collect();

                            if !available_for_export.is_empty() {
                                ui.horizontal(|ui| {
                                    ui.label("Add:");
                                    egui::ComboBox::from_id_salt("add_export_combo")
                                        .selected_text("Select asset…")
                                        .show_ui(ui, |ui| {
                                            for asset_id in &available_for_export {
                                                if ui.selectable_label(false, *asset_id).clicked() {
                                                    export_to_add = Some(asset_id.to_string());
                                                }
                                            }
                                        });
                                });
                            }
                        } else {
                            ui.weak("No project loaded");
                        }
                    });

                // Handle edit action
                if let Some(idx) = step_to_edit {
                    self.start_editing_entry(idx);
                }

                // Handle delete action
                if let Some(idx) = step_to_delete {
                    if let Some(ref mut project) = self.project {
                        // Remove the timeline step
                        if idx < project.timeline().len() {
                            // Get the output IDs before removing
                            let output_ids: Vec<String> = project.timeline().get(idx)
                                .map(|step| step.outputs.clone())
                                .unwrap_or_default();

                            // Remove the step
                            project.timeline_mut().remove(idx);

                            // Also remove corresponding exports
                            for output_id in output_ids {
                                project.exports_mut().retain(|e| e != &output_id);
                            }

                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.project_path = None; // Mark as modified
                            }
                            self.run_project();
                        }
                    }
                    // Close edit panel if we deleted the step being edited
                    if self.editing_entry_index == Some(idx) {
                        self.editing_entry_index = None;
                    } else if let Some(edit_idx) = self.editing_entry_index {
                        if idx < edit_idx {
                            self.editing_entry_index = Some(edit_idx - 1);
                        }
                    }
                }

                // Handle move up action
                if let Some(idx) = step_to_move_up {
                    if let Some(ref mut project) = self.project {
                        if idx > 0 && idx < project.timeline().len() {
                            project.timeline_mut().swap(idx, idx - 1);
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.project_path = None; // Mark as modified
                            }
                            self.run_project();
                            // Adjust edit index if needed
                            if let Some(edit_idx) = self.editing_entry_index {
                                if edit_idx == idx {
                                    self.editing_entry_index = Some(idx - 1);
                                } else if edit_idx == idx - 1 {
                                    self.editing_entry_index = Some(idx);
                                }
                            }
                        }
                    }
                }

                // Handle move down action
                if let Some(idx) = step_to_move_down {
                    if let Some(ref mut project) = self.project {
                        let len = project.timeline().len();
                        if idx < len - 1 {
                            project.timeline_mut().swap(idx, idx + 1);
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.project_path = None; // Mark as modified
                            }
                            self.run_project();
                            // Adjust edit index if needed
                            if let Some(edit_idx) = self.editing_entry_index {
                                if edit_idx == idx {
                                    self.editing_entry_index = Some(idx + 1);
                                } else if edit_idx == idx + 1 {
                                    self.editing_entry_index = Some(idx);
                                }
                            }
                        }
                    }
                }

                // Handle import deletion
                if let Some(idx) = import_to_delete {
                    if let Some(ref mut project) = self.project {
                        if idx < project.imports().len() {
                            // Get the import ID before removing
                            let import_id = project.imports()[idx].id.clone();

                            // Remove the import
                            project.imports_mut().remove(idx);

                            // Remove any timeline steps that use this import as operator
                            let steps_to_remove: Vec<usize> = project.timeline()
                                .iter()
                                .enumerate()
                                .filter(|(_, step)| step.operator_id == import_id)
                                .map(|(i, _)| i)
                                .collect();

                            // Remove in reverse order to preserve indices
                            for step_idx in steps_to_remove.into_iter().rev() {
                                // Get output IDs for export cleanup
                                let output_ids: Vec<String> = project.timeline().get(step_idx)
                                    .map(|s| s.outputs.clone())
                                    .unwrap_or_default();

                                project.timeline_mut().remove(step_idx);

                                // Remove corresponding exports
                                for output_id in output_ids {
                                    project.exports_mut().retain(|e| e != &output_id);
                                }
                            }

                            // Also remove from exports if this import was exported directly
                            project.exports_mut().retain(|e| e != &import_id);

                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.project_path = None; // Mark as modified
                            }
                            self.run_project();
                        }
                    }
                }

                // Handle export deletion
                if let Some(idx) = export_to_delete {
                    if let Some(ref mut project) = self.project {
                        if idx < project.exports().len() {
                            project.exports_mut().remove(idx);
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.project_path = None; // Mark as modified
                            }
                            self.run_project();
                        }
                    }
                }

                // Handle export addition
                if let Some(asset_id) = export_to_add {
                    if let Some(ref mut project) = self.project {
                        if !project.exports().contains(&asset_id) {
                            project.exports_mut().push(asset_id);
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.project_path = None; // Mark as modified
                            }
                            self.run_project();
                        }
                    }
                }

                ui.separator();
                egui::CollapsingHeader::new("Project Exports (last run)")
                    .default_open(true)
                    .show(ui, |ui| {
                        if self.exported_assets.is_empty() {
                            ui.weak("(no exported assets)");
                            return;
                        }

                        // Clone to avoid borrow conflicts when UI callbacks mutate `self`.
                        let exported_assets = self.exported_assets.clone();
                        for asset in exported_assets {
                            let asset_id = asset.id().to_string();
                            let is_rendering = self.asset_render_data.contains_key(&asset_id);

                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    let type_name = asset.type_hint()
                                        .map(|h| h.to_string())
                                        .unwrap_or_else(|| "Binary".to_string());
                                    ui.label(format!(
                                        "{}: {} ({} bytes)",
                                        asset.id(),
                                        type_name,
                                        asset.data().len(),
                                    ));
                                    if is_rendering {
                                        ui.weak("(rendering)");
                                    }
                                });

                                if !asset.precursor_ids().is_empty() {
                                    ui.label(format!(
                                        "precursors: {}",
                                        asset.precursor_ids().join(", ")
                                    ));
                                }

                                let is_renderable = asset.as_model().is_some();
                                let current_mode = self.get_asset_render_mode(&asset_id);

                                ui.horizontal(|ui| {
                                    ui.label("Render:");

                                    if !is_renderable {
                                        ui.weak("(not renderable)");
                                        return;
                                    }

                                    let mut mode = current_mode;
                                    egui::ComboBox::from_id_salt(format!("render_{asset_id}"))
                                        .selected_text(mode.label())
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut mode,
                                                ExportRenderMode::None,
                                                ExportRenderMode::None.label(),
                                            );
                                            ui.selectable_value(
                                                &mut mode,
                                                ExportRenderMode::PointCloud,
                                                ExportRenderMode::PointCloud.label(),
                                            );
                                            ui.selectable_value(
                                                &mut mode,
                                                ExportRenderMode::MarchingCubes,
                                                ExportRenderMode::MarchingCubes.label(),
                                            );
                                            ui.selectable_value(
                                                &mut mode,
                                                ExportRenderMode::AdaptiveSurfaceNets2,
                                                ExportRenderMode::AdaptiveSurfaceNets2.label(),
                                            );
                                        });

                                    if mode != current_mode {
                                        self.set_asset_render_mode(&asset_id, mode);
                                    }
                                });

                                // Show config options for PointCloud and MarchingCubes modes
                                if current_mode == ExportRenderMode::PointCloud || current_mode == ExportRenderMode::MarchingCubes {
                                    if let Some(render_data) = self.asset_render_data.get_mut(&asset_id) {
                                        let mut resolution = render_data.resolution;
                                        let mut auto_resample = render_data.auto_resample;
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Resolution:");
                                            if ui.add(egui::Slider::new(&mut resolution, 5..=300)).changed() {
                                                render_data.resolution = resolution;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            if ui.checkbox(&mut auto_resample, "Auto Resample").changed() {
                                                render_data.auto_resample = auto_resample;
                                            }
                                            if !auto_resample {
                                                if ui.button("⟳ Resample").clicked() {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });
                                    }
                                }

                                // Show ASN2 config options when AdaptiveSurfaceNets2 is selected
                                if current_mode == ExportRenderMode::AdaptiveSurfaceNets2 {
                                    if let Some(render_data) = self.asset_render_data.get_mut(&asset_id) {
                                        let mut base_cell_size = render_data.asn2_base_cell_size;
                                        let mut max_depth = render_data.asn2_max_depth;
                                        let mut auto_resample = render_data.auto_resample;

                                        ui.horizontal(|ui| {
                                            ui.label("Base Cell Size:");
                                            if ui
                                                .add(egui::DragValue::new(&mut base_cell_size).speed(0.01).range(0.01..=10.0))
                                                .changed()
                                            {
                                                render_data.asn2_base_cell_size = base_cell_size;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });

                                        ui.horizontal(|ui| {
                                            ui.label("Max Depth:");
                                            if ui.add(egui::DragValue::new(&mut max_depth).range(0..=6)).changed() {
                                                render_data.asn2_max_depth = max_depth;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });

                                        let finest_cell = base_cell_size / (1 << max_depth) as f64;
                                        ui.weak(format!("Finest cell size: {:.4}", finest_cell));

                                        let mut vertex_refine = render_data.asn2_vertex_refinement_iterations;
                                        let mut normal_iters = render_data.asn2_normal_sample_iterations;

                                        ui.horizontal(|ui| {
                                            ui.label("Vertex Refinement:");
                                            if ui.add(egui::DragValue::new(&mut vertex_refine).range(0..=16)).changed() {
                                                render_data.asn2_vertex_refinement_iterations = vertex_refine;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });

                                        ui.horizontal(|ui| {
                                            ui.label("Normal Refinement:");
                                            if ui.add(egui::DragValue::new(&mut normal_iters).range(0..=16)).changed() {
                                                render_data.asn2_normal_sample_iterations = normal_iters;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });
                                        if normal_iters == 0 {
                                            ui.weak("Using accumulated face normals (fast)");
                                        } else {
                                            ui.weak(format!("Probing surface with {} iterations per direction", normal_iters));
                                        }

                                        ui.separator();

                                        // Sharp Edge Detection controls
                                        let sharp_enabled = render_data.asn2_sharp_edge_config.is_some();
                                        let mut sharp_checkbox = sharp_enabled;

                                        ui.horizontal(|ui| {
                                            if ui.checkbox(&mut sharp_checkbox, "Sharp Edge Detection").changed() {
                                                if sharp_checkbox {
                                                    render_data.asn2_sharp_edge_config = Some(
                                                        adaptive_surface_nets_2::SharpEdgeConfig::default()
                                                    );
                                                } else {
                                                    render_data.asn2_sharp_edge_config = None;
                                                }
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });

                                        if let Some(ref mut sharp_config) = render_data.asn2_sharp_edge_config {
                                            let mut angle_degrees = sharp_config.angle_threshold.to_degrees();
                                            let mut residual_mult = sharp_config.residual_multiplier;

                                            ui.horizontal(|ui| {
                                                ui.label("  Angle Threshold:");
                                                if ui.add(egui::DragValue::new(&mut angle_degrees).range(10.0..=90.0).suffix("°")).changed() {
                                                    sharp_config.angle_threshold = angle_degrees.to_radians();
                                                    if auto_resample {
                                                        render_data.needs_resample = true;
                                                    }
                                                }
                                            });

                                            ui.horizontal(|ui| {
                                                ui.label("  Residual Multiplier:");
                                                if ui.add(egui::DragValue::new(&mut residual_mult).range(1.0..=20.0)).changed() {
                                                    sharp_config.residual_multiplier = residual_mult;
                                                    if auto_resample {
                                                        render_data.needs_resample = true;
                                                    }
                                                }
                                            });

                                            ui.weak("  Detects sharp edges and duplicates vertices for crisp shading");
                                        }

                                        ui.separator();

                                        ui.horizontal(|ui| {
                                            if ui.checkbox(&mut auto_resample, "Auto Resample").changed() {
                                                render_data.auto_resample = auto_resample;
                                            }
                                            if !auto_resample {
                                                if ui.button("⟳ Resample").clicked() {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });
                                    }
                                }

                                if let Some(render_data) = self.asset_render_data.get(&asset_id) {
                                    if let Some(time) = render_data.last_sample_time {
                                        ui.horizontal(|ui| {
                                            ui.weak(format!("Meshed in {:.2}ms", time * 1000.0));
                                            if let Some(count) = render_data.last_sample_count {
                                                let avg = (time * 1000_000.0) / (count as f32);
                                                ui.weak(format!("({:.2}µs/sample avg)", avg));
                                            }
                                        });
                                    }

                                    // Show detailed ASN2 profiling stats in a collapsible section
                                    if let Some(ref stats) = render_data.asn2_stats {
                                        ui.collapsing("Profiling Details", |ui| {
                                            ui.horizontal(|ui| {
                                                ui.weak(format!("Total samples: {}", stats.total_samples));
                                                ui.weak(format!("| Vertices: {}", stats.total_vertices));
                                                ui.weak(format!("| Triangles: {}", stats.total_triangles));
                                            });
                                            ui.separator();

                                            ui.horizontal(|ui| {
                                                ui.weak("Stage 1 (Discovery):");
                                                ui.weak(format!("{:.2}ms ({:.1}%)",
                                                    stats.stage1_time_secs * 1000.0,
                                                    stats.stage1_time_secs / stats.total_time_secs * 100.0));
                                            });
                                            ui.weak(format!("  {} samples, {} mixed cells",
                                                stats.stage1_samples, stats.stage1_mixed_cells));

                                            ui.horizontal(|ui| {
                                                ui.weak("Stage 2 (Subdivision):");
                                                ui.weak(format!("{:.2}ms ({:.1}%)",
                                                    stats.stage2_time_secs * 1000.0,
                                                    stats.stage2_time_secs / stats.total_time_secs * 100.0));
                                            });
                                            ui.weak(format!("  {} samples, {} cuboids, {} triangles",
                                                stats.stage2_samples, stats.stage2_cuboids_processed, stats.stage2_triangles_emitted));

                                            ui.horizontal(|ui| {
                                                ui.weak("Stage 3 (Topology):");
                                                ui.weak(format!("{:.2}ms ({:.1}%)",
                                                    stats.stage3_time_secs * 1000.0,
                                                    stats.stage3_time_secs / stats.total_time_secs * 100.0));
                                            });
                                            ui.weak(format!("  {} unique vertices", stats.stage3_unique_vertices));

                                            ui.horizontal(|ui| {
                                                ui.weak("Stage 4 (Refinement):");
                                                ui.weak(format!("{:.2}ms ({:.1}%)",
                                                    stats.stage4_time_secs * 1000.0,
                                                    stats.stage4_time_secs / stats.total_time_secs * 100.0));
                                            });
                                            ui.weak(format!("  {} samples", stats.stage4_samples));

                                            // Show Stage 4.5 (Sharp Edges) if any processing occurred
                                            if stats.sharp_vertices_case1 > 0
                                                || stats.sharp_edge_crossings > 0
                                                || stats.sharp_vertices_duplicated > 0
                                            {
                                                ui.horizontal(|ui| {
                                                    ui.weak("Stage 4.5 (Sharp Edges):");
                                                    ui.weak(format!("{:.2}ms ({:.1}%)",
                                                        stats.stage4_5_time_secs * 1000.0,
                                                        stats.stage4_5_time_secs / stats.total_time_secs * 100.0));
                                                });
                                                ui.weak(format!("  Case1: {}, Crossings: {}, Inserted: {}, Duplicated: {}",
                                                    stats.sharp_vertices_case1,
                                                    stats.sharp_edge_crossings,
                                                    stats.sharp_vertices_inserted,
                                                    stats.sharp_vertices_duplicated));
                                            }
                                        });
                                    }

                                    #[cfg(not(target_arch = "wasm32"))]
                                    if ui.button("Export STL…").clicked() {
                                        if let Some(path) = rfd::FileDialog::new()
                                            .add_filter("STL", &["stl"])
                                            .set_file_name(format!("{}.stl", asset_id))
                                            .save_file()
                                        {
                                            // Export the currently rendered mesh
                                            // Priority: indexed mesh (ASN2) > triangles > fallback to marching cubes

                                            let triangles = if let Some(ref indices) = render_data.mesh_indices {
                                                // Convert indexed mesh to triangles for STL export
                                                let vertices = &render_data.mesh_vertices;
                                                let normals_from_vertices: Vec<Triangle> = indices
                                                    .chunks(3)
                                                    .filter_map(|tri| {
                                                        if tri.len() != 3 {
                                                            return None;
                                                        }
                                                        let i0 = tri[0] as usize;
                                                        let i1 = tri[1] as usize;
                                                        let i2 = tri[2] as usize;

                                                        let v0 = vertices.get(i0)?;
                                                        let v1 = vertices.get(i1)?;
                                                        let v2 = vertices.get(i2)?;

                                                        Some(Triangle {
                                                            vertices: [
                                                                (v0.position[0], v0.position[1], v0.position[2]),
                                                                (v1.position[0], v1.position[1], v1.position[2]),
                                                                (v2.position[0], v2.position[1], v2.position[2]),
                                                            ],
                                                            normals: [
                                                                (v0.normal[0], v0.normal[1], v0.normal[2]),
                                                                (v1.normal[0], v1.normal[1], v1.normal[2]),
                                                                (v2.normal[0], v2.normal[1], v2.normal[2]),
                                                            ],
                                                        })
                                                    })
                                                    .collect();
                                                normals_from_vertices
                                            } else if !render_data.triangles.is_empty() {
                                                render_data.triangles.clone()
                                            } else {
                                                // Fallback: run marching cubes just for export if not already meshed
                                                match generate_marching_cubes_mesh_from_bytes(&render_data.wasm_bytes, render_data.resolution) {
                                                    Ok((t, _, _)) => t,
                                                    Err(e) => {
                                                        self.error_message = Some(format!("Failed to generate mesh for export: {e}"));
                                                        Vec::new()
                                                    }
                                                }
                                            };

                                            if !triangles.is_empty() {
                                                match stl::write_binary_stl(&path, &triangles, "volumetric") {
                                                    Ok(()) => self.error_message = None,
                                                    Err(e) => {
                                                        self.error_message = Some(format!("Failed to export STL: {e}"))
                                                    }
                                                }
                                            } else {
                                                self.error_message = Some("No mesh data to export".to_string());
                                            }
                                        }
                                    }
                                }

                                // Export WASM button - always available for model exports (doesn't depend on sampling)
                                #[cfg(not(target_arch = "wasm32"))]
                                if let Some(wasm_bytes) = asset.as_model() {
                                    if ui.button("Export WASM…").clicked() {
                                        if let Some(path) = rfd::FileDialog::new()
                                            .add_filter("WASM", &["wasm"])
                                            .set_file_name(format!("{}.wasm", asset_id))
                                            .save_file()
                                        {
                                            match fs::write(&path, wasm_bytes) {
                                                Ok(()) => self.error_message = None,
                                                Err(e) => {
                                                    self.error_message = Some(format!("Failed to export WASM: {e}"))
                                                }
                                            }
                                        }
                                    }
                                }

                                if let Some(err) = self
                                    .asset_render_data
                                    .get(&asset_id)
                                    .and_then(|d| d.last_error.as_ref())
                                {
                                    ui.add_space(4.0);
                                    ui.colored_label(egui::Color32::RED, err);
                                }
                            });
                        }
                    });

                if let Some(ref error) = self.error_message {
                    ui.separator();
                    ui.colored_label(egui::Color32::RED, error);
                }
                    });
            });

        // Right panel for editing project entries (shown when editing_entry_index is Some)
        if self.editing_entry_index.is_some() {
            egui::SidePanel::right("edit_panel")
                .default_width(300.0)
                .show(ctx, |ui| {
                    self.show_edit_panel(ui);
                });
        }

        // Central panel with 3D view
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::click_and_drag(),
            );
            
            let rect = response.rect;

            // Gather input state for camera control
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            let modifiers = ui.input(|i| i.modifiers);

            let input_state = renderer::CameraInputState {
                left_down: response.dragged_by(egui::PointerButton::Primary),
                middle_down: response.dragged_by(egui::PointerButton::Middle),
                right_down: response.dragged_by(egui::PointerButton::Secondary),
                shift_down: modifiers.shift,
                ctrl_down: modifiers.ctrl,
                alt_down: modifiers.alt,
                mouse_delta: if let Some(pos) = response.interact_pointer_pos() {
                    if let Some(last_pos) = self.last_mouse_pos {
                        let delta = pos - last_pos;
                        glam::Vec2::new(delta.x, delta.y)
                    } else {
                        glam::Vec2::ZERO
                    }
                } else {
                    glam::Vec2::ZERO
                },
                scroll_delta: scroll,
            };

            // Update last mouse position for drag tracking
            if response.dragged_by(egui::PointerButton::Primary)
                || response.dragged_by(egui::PointerButton::Middle)
                || response.dragged_by(egui::PointerButton::Secondary)
            {
                if let Some(pos) = response.interact_pointer_pos() {
                    self.last_mouse_pos = Some(pos);
                }
            } else {
                self.last_mouse_pos = None;
            }

            // Determine and apply camera action based on control scheme
            let action = self.camera_control_scheme.determine_action(&input_state);
            match action {
                renderer::CameraAction::Orbit => {
                    self.camera_theta -= input_state.mouse_delta.x * 0.01;
                    self.camera_phi = (self.camera_phi - input_state.mouse_delta.y * 0.01)
                        .clamp(0.1, std::f32::consts::PI - 0.1);
                }
                renderer::CameraAction::Pan => {
                    // Create temporary camera to compute pan in view plane
                    let temp_camera = renderer::Camera {
                        target: self.camera_target,
                        radius: self.camera_radius,
                        theta: self.camera_theta,
                        phi: self.camera_phi,
                        fov_y: 60.0_f32.to_radians(),
                        near: 0.1,
                        far: 100.0,
                    };
                    let right = temp_camera.right();
                    let up = temp_camera.up();
                    let scale = self.camera_radius * 0.002;
                    self.camera_target -= right * (input_state.mouse_delta.x * scale);
                    self.camera_target += up * (input_state.mouse_delta.y * scale);
                }
                renderer::CameraAction::Zoom => {
                    // For Maya alt+right drag, use mouse delta; otherwise use scroll
                    let zoom_delta = if input_state.scroll_delta != 0.0 {
                        input_state.scroll_delta * 0.01
                    } else {
                        // For drag zoom (Maya style), use horizontal mouse movement
                        input_state.mouse_delta.x * 0.02
                    };
                    self.camera_radius = (self.camera_radius - zoom_delta).clamp(0.1, 100.0);
                }
                renderer::CameraAction::None => {}
            }
            
            // Draw background
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 38));

            // Build scene from asset data
            let pixels_per_point = ui.ctx().pixels_per_point();
            let viewport_size_px = [
                (rect.width() * pixels_per_point).round().max(1.0) as u32,
                (rect.height() * pixels_per_point).round().max(1.0) as u32,
            ];

            let sample_set = if self.sample_cloud_enabled {
                self.selected_sample_cloud_set()
            } else {
                None
            };
            let split_view = self.sample_cloud_mode == SampleCloudViewMode::Split && sample_set.is_some();

            let mut scene = renderer::SceneData::new();
            let mut scene_right = if split_view {
                Some(renderer::SceneData::new())
            } else {
                None
            };

            let add_asset_points = |scene: &mut renderer::SceneData, asset_render_data: &HashMap<String, AssetRenderData>| {
                for asset_data in asset_render_data.values() {
                    if asset_data.mode == ExportRenderMode::PointCloud && !asset_data.points.is_empty() {
                        scene.add_points(
                            renderer::convert_points_to_point_data(&asset_data.points),
                            glam::Mat4::IDENTITY,
                            renderer::PointStyle {
                                size: 3.0,
                                size_mode: renderer::WidthMode::ScreenSpace,
                                shape: renderer::PointShape::Circle,
                                depth_mode: renderer::DepthMode::Normal,
                            },
                        );
                    }
                }
            };

            let add_asset_meshes = |scene: &mut renderer::SceneData, asset_render_data: &HashMap<String, AssetRenderData>| {
                for asset_data in asset_render_data.values() {
                    if matches!(asset_data.mode,
                        ExportRenderMode::MarchingCubes |
                        ExportRenderMode::AdaptiveSurfaceNets2
                    ) && !asset_data.mesh_vertices.is_empty() {
                        scene.add_mesh(
                            renderer::convert_mesh_data(
                                &asset_data.mesh_vertices,
                                asset_data.mesh_indices.as_ref().map(|v| v.as_slice()),
                            ),
                            glam::Mat4::IDENTITY,
                            renderer::MaterialId(0),
                        );
                    }
                }
            };

            let add_sample_cloud = |scene: &mut renderer::SceneData, set: &SampleCloudSet, size: f32| {
                scene.add_points(
                    build_sample_cloud_points(set),
                    glam::Mat4::IDENTITY,
                    renderer::PointStyle {
                        size,
                        size_mode: renderer::WidthMode::ScreenSpace,
                        shape: renderer::PointShape::Circle,
                        depth_mode: renderer::DepthMode::Normal,
                    },
                );
                scene.add_points(
                    build_sample_cloud_root(set),
                    glam::Mat4::IDENTITY,
                    renderer::PointStyle {
                        size: size * 1.6,
                        size_mode: renderer::WidthMode::ScreenSpace,
                        shape: renderer::PointShape::Diamond,
                        depth_mode: renderer::DepthMode::Normal,
                    },
                );
            };

            if split_view {
                add_asset_points(&mut scene, &self.asset_render_data);
                add_asset_meshes(&mut scene, &self.asset_render_data);
                if let Some(set) = sample_set {
                    if let Some(ref mut right_scene) = scene_right {
                        add_sample_cloud(right_scene, set, self.sample_cloud_point_size);
                    }
                }
            } else {
                match self.sample_cloud_mode {
                    SampleCloudViewMode::CloudOnly if sample_set.is_some() => {
                        if let Some(set) = sample_set {
                            add_sample_cloud(&mut scene, set, self.sample_cloud_point_size);
                        }
                    }
                    _ => {
                        add_asset_points(&mut scene, &self.asset_render_data);
                        add_asset_meshes(&mut scene, &self.asset_render_data);
                        if self.sample_cloud_mode == SampleCloudViewMode::Overlay {
                            if let Some(set) = sample_set {
                                add_sample_cloud(&mut scene, set, self.sample_cloud_point_size);
                            }
                        }
                    }
                }
            }

            // Create camera from existing spherical coords
            let camera = renderer::Camera {
                target: self.camera_target,
                radius: self.camera_radius,
                theta: self.camera_theta,
                phi: self.camera_phi,
                fov_y: 60.0_f32.to_radians(),
                near: 0.1,
                far: 100.0,
            };

            let settings = renderer::RenderSettings {
                ssao_enabled: self.ssao_enabled,
                ssao_samples: 16,
                ssao_radius: self.ssao_radius,
                ssao_bias: self.ssao_bias,
                ssao_strength: self.ssao_strength,
                grid: renderer::GridSettings::default(),
                axis_indicator: renderer::AxisIndicator::default(),
                show_axis_indicator: false,
                background_color: [0.098, 0.098, 0.149, 1.0], // rgb(25,25,38)
            };

            if split_view {
                let left_rect = egui::Rect::from_min_size(
                    rect.min,
                    egui::vec2(rect.width() * 0.5, rect.height()),
                );
                let right_rect = egui::Rect::from_min_size(
                    egui::pos2(left_rect.max.x, rect.min.y),
                    egui::vec2(rect.width() * 0.5, rect.height()),
                );
                let left_size = [
                    (left_rect.width() * pixels_per_point).round().max(1.0) as u32,
                    (left_rect.height() * pixels_per_point).round().max(1.0) as u32,
                ];
                let right_size = [
                    (right_rect.width() * pixels_per_point).round().max(1.0) as u32,
                    (right_rect.height() * pixels_per_point).round().max(1.0) as u32,
                ];

                let cb_left = eframe::egui_wgpu::Callback::new_paint_callback(
                    left_rect,
                    renderer::SceneCallback {
                        data: renderer::SceneDrawData {
                            scene: scene.clone(),
                            camera: camera.clone(),
                            settings: settings.clone(),
                            viewport_size: left_size,
                            target_format: self.wgpu_target_format,
                        },
                    },
                );
                painter.add(egui::Shape::Callback(cb_left));

                if let Some(right_scene) = scene_right {
                    let cb_right = eframe::egui_wgpu::Callback::new_paint_callback(
                        right_rect,
                        renderer::SceneCallback {
                            data: renderer::SceneDrawData {
                                scene: right_scene,
                                camera,
                                settings,
                                viewport_size: right_size,
                                target_format: self.wgpu_target_format,
                            },
                        },
                    );
                    painter.add(egui::Shape::Callback(cb_right));
                }
            } else {
                // Always render the scene (grid, axis indicator render even with no models)
                let cb = eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    renderer::SceneCallback {
                        data: renderer::SceneDrawData {
                            scene: scene.clone(),
                            camera,
                            settings,
                            viewport_size: viewport_size_px,
                            target_format: self.wgpu_target_format,
                        },
                    },
                );
                painter.add(egui::Shape::Callback(cb));
            }

            // Draw info text showing totals across all rendered assets
            let total_points: usize = self.asset_render_data.values()
                .filter(|d| d.mode == ExportRenderMode::PointCloud)
                .map(|d| d.points.len())
                .sum();
            let total_triangles: usize = self.asset_render_data.values()
                .filter(|d| matches!(d.mode,
                    ExportRenderMode::MarchingCubes |
                    ExportRenderMode::AdaptiveSurfaceNets2))
                .map(|d| {
                    // For ASN2, count triangles from indices
                    if let Some(ref indices) = d.mesh_indices {
                        indices.len() / 3
                    } else {
                        d.triangles.len()
                    }
                })
                .sum();
            let num_assets = self.asset_render_data.len();

            let info_text = if num_assets == 0 {
                "No models | Drag to rotate, scroll to zoom".to_string()
            } else {
                let mut parts = vec![format!("{} asset(s)", num_assets)];
                if total_points > 0 {
                    parts.push(format!("{} points", total_points));
                }
                if total_triangles > 0 {
                    parts.push(format!("{} triangles", total_triangles));
                }
                parts.push("Drag to rotate, scroll to zoom".to_string());
                parts.join(" | ")
            };
            painter.text(
                rect.left_top() + egui::vec2(10.0, 10.0),
                egui::Align2::LEFT_TOP,
                info_text,
                egui::FontId::default(),
                egui::Color32::WHITE,
            );
        });

        // Request continuous repaints for smooth interaction
        ctx.request_repaint();
    }
}

#[allow(dead_code)]
fn triangles_to_mesh_vertices(triangles: &[Triangle]) -> Vec<renderer::MeshVertex> {
    let mut out = Vec::with_capacity(triangles.len() * 3);

    for tri in triangles {
        // With back-face culling enabled, inconsistent triangle winding will show up as
        // apparent "holes". Some mesh sources can produce triangles whose vertex order is
        // flipped relative to their normals.
        //
        // Fix this robustly by ensuring the geometric face normal points in the same general
        // direction as the average of the provided vertex normals.
        let (ax, ay, az) = tri.vertices[0];
        let (bx, by, bz) = tri.vertices[1];
        let (cx, cy, cz) = tri.vertices[2];
        let ab = (bx - ax, by - ay, bz - az);
        let ac = (cx - ax, cy - ay, cz - az);
        let face_n = (
            ab.1 * ac.2 - ab.2 * ac.1,
            ab.2 * ac.0 - ab.0 * ac.2,
            ab.0 * ac.1 - ab.1 * ac.0,
        );
        let avg_n = (
            tri.normals[0].0 + tri.normals[1].0 + tri.normals[2].0,
            tri.normals[0].1 + tri.normals[1].1 + tri.normals[2].1,
            tri.normals[0].2 + tri.normals[1].2 + tri.normals[2].2,
        );
        let dot = face_n.0 * avg_n.0 + face_n.1 * avg_n.1 + face_n.2 * avg_n.2;
        let flip_winding = dot < 0.0;

        // Use per-vertex normals for smooth shading.
        // If winding is flipped, swap indices 1 and 2 (and their normals) to restore CCW.
        let idxs: [usize; 3] = if flip_winding { [0, 2, 1] } else { [0, 1, 2] };
        for i in idxs {
            let v = tri.vertices[i];
            let n = tri.normals[i];
            
            // Fallback to up vector if normal is degenerate
            let normal = if n.0 == 0.0 && n.1 == 0.0 && n.2 == 0.0 {
                [0.0, 1.0, 0.0]
            } else {
                [n.0, n.1, n.2]
            };

            out.push(renderer::MeshVertex {
                position: [v.0, v.1, v.2],
                _pad0: 0.0,
                normal,
                _pad1: 0.0,
            });
        }
    }

    out
}

// =============================================================================
// Native Entry Point
// =============================================================================

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    // Load initial project from CLI argument (can be .wasm or .vproj)
    let initial_project = if args.len() > 1 {
        let path = PathBuf::from(&args[1]);
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        match ext {
            "vproj" => {
                // Load project file
                match Project::load_from_file(&path) {
                    Ok(project) => {
                        println!("Loaded project from: {}", path.display());
                        Some(project)
                    }
                    Err(e) => {
                        eprintln!("Failed to load project: {}", e);
                        None
                    }
                }
            }
            "wasm" => {
                // Import WASM file into a new empty project
                match fs::read(&path) {
                    Ok(wasm_bytes) => {
                        let asset_id = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("model")
                            .to_string();
                        println!("Importing WASM model from: {}", path.display());
                        let mut project = Project::new();
                        project.insert_model(asset_id.as_str(), wasm_bytes);
                        Some(project)
                    }
                    Err(e) => {
                        eprintln!("Failed to read WASM file: {}", e);
                        None
                    }
                }
            }
            _ => {
                eprintln!(
                    "Unknown file type: {}. Expected .wasm or .vproj",
                    path.display()
                );
                None
            }
        }
    } else {
        Some(Project::new())
    };

    println!("Volumetric Model Renderer (eframe/egui)");
    println!("=======================================");
    // Note: the UI can still tolerate an explicit no-project state via the "Unload" button.
    println!("Tip: import a WASM file or open a project in the UI to add models and operations.");
    println!();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_title("Volumetric Renderer"),
        // Needed for correct 3D triangle rendering (occlusion) in the wgpu callbacks.
        depth_buffer: 24,
        ..Default::default()
    };

    eframe::run_native(
        "Volumetric Renderer",
        options,
        Box::new(|cc| Ok(Box::new(VolumetricApp::new(cc, initial_project)))),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run eframe: {}", e))?;

    Ok(())
}

// WASM builds need a main function (even though wasm_bindgen(start) is the real entry)
#[cfg(target_arch = "wasm32")]
fn main() {
    // The actual entry point is the `start` function below with #[wasm_bindgen(start)]
}

// =============================================================================
// Web Entry Point
// =============================================================================

/// Web entry point for WASM builds
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    use wasm_bindgen::JsCast;

    // Set up panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize logging to browser console
    console_log::init_with_level(log::Level::Debug)
        .map_err(|e| JsValue::from_str(&format!("Failed to init logger: {}", e)))?;

    log::info!("Volumetric UI starting in web mode...");

    // Get the canvas element
    let document = web_sys::window()
        .ok_or_else(|| JsValue::from_str("No window"))?
        .document()
        .ok_or_else(|| JsValue::from_str("No document"))?;

    let canvas = document
        .get_element_by_id("volumetric_canvas")
        .ok_or_else(|| JsValue::from_str("Canvas element not found"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| JsValue::from_str("Element is not a canvas"))?;

    // Force WebGL2 backend instead of WebGPU to avoid memory issues
    let web_options = eframe::WebOptions {
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(eframe::egui_wgpu::WgpuSetupCreateNew {
                instance_descriptor: wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::GL,
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    wasm_bindgen_futures::spawn_local(async move {
        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(VolumetricApp::new(cc, None)))),
            )
            .await
            .expect("Failed to start eframe");
    });

    Ok(())
}
