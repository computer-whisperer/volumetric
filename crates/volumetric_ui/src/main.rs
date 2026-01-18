use anyhow::{Context, Result};
use eframe::egui;
use ciborium::value::Value as CborValue;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;
use std::sync::Arc;

mod point_cloud_wgpu;
mod marching_cubes_wgpu;

// Project system
use volumetric::{
    AssetType,
    Environment,
    ExecuteWasmEntry,
    ExecuteWasmInput,
    ExecuteWasmOutput,
    LoadedAsset,
    Project,
    ProjectEntry,
    OperatorMetadata,
    OperatorMetadataInput,
    OperatorMetadataOutput,
    operator_metadata_from_wasm_bytes,
    Triangle,
    adaptive_surface_nets,
    adaptive_surface_nets_2,
    stl,
    sample_model_from_bytes,
    generate_marching_cubes_mesh_from_bytes,
    generate_adaptive_mesh_from_bytes,
    generate_adaptive_mesh_v2_from_bytes,
};

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
    AdaptiveSurfaceNets,
    AdaptiveSurfaceNets2,
}

impl ExportRenderMode {
    fn label(self) -> &'static str {
        match self {
            ExportRenderMode::None => "None",
            ExportRenderMode::PointCloud => "Point Cloud",
            ExportRenderMode::MarchingCubes => "Marching Cubes",
            ExportRenderMode::AdaptiveSurfaceNets => "Adaptive Surface Nets",
            ExportRenderMode::AdaptiveSurfaceNets2 => "ASN v2 (Indexed)",
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
    /// Triangle data (for MarchingCubes and AdaptiveSurfaceNets modes)
    triangles: Vec<Triangle>,
    /// Mesh vertices (for non-indexed mesh modes)
    mesh_vertices: Arc<Vec<marching_cubes_wgpu::MeshVertex>>,
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

    /// Adaptive Surface Nets v1: base resolution (cells per axis for initial discovery)
    adaptive_base_resolution: usize,
    /// Adaptive Surface Nets v1: maximum refinement depth
    adaptive_max_depth: usize,
    /// Adaptive Surface Nets v1: binary search iterations for edge crossing refinement (0 = disabled)
    adaptive_edge_refinement_iterations: usize,
    /// Adaptive Surface Nets v1: vertex relaxation iterations to project vertices onto surface (0 = disabled)
    adaptive_vertex_relaxation_iterations: usize,
    /// Adaptive Surface Nets v1: enable high-quality (bisection-based) normals.
    adaptive_hq_normals: bool,

    /// Adaptive Surface Nets v2: base resolution (cells per axis for initial discovery)
    asn2_base_resolution: usize,
    /// Adaptive Surface Nets v2: maximum refinement depth
    asn2_max_depth: usize,
    /// Adaptive Surface Nets v2: binary search iterations for vertex position refinement
    asn2_vertex_refinement_iterations: usize,
    /// Adaptive Surface Nets v2: iterations for normal re-estimation at refined positions (0 = use mesh normals)
    asn2_normal_sample_iterations: usize,
    /// Adaptive Surface Nets v2: epsilon fraction for normal estimation
    asn2_normal_epsilon_frac: f32,
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
            // ASN v1 config
            adaptive_base_resolution: 8,
            adaptive_max_depth: 4,
            adaptive_edge_refinement_iterations: 4,
            adaptive_vertex_relaxation_iterations: 2,
            adaptive_hq_normals: false,
            // ASN v2 config
            asn2_base_resolution: 8,
            asn2_max_depth: 4,
            asn2_vertex_refinement_iterations: 8,
            asn2_normal_sample_iterations: 6,
            asn2_normal_epsilon_frac: 0.1,
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

/// Application state for the volumetric renderer
struct VolumetricApp {
    /// The current project (contains the model pipeline)
    project: Option<Project>,
    /// Path to the current project file (for save operations)
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
    operator_metadata_cache: HashMap<String, CachedOperatorMetadata>,
    wgpu_target_format: wgpu::TextureFormat,
    // Camera state
    camera_theta: f32,
    camera_phi: f32,
    camera_radius: f32,
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
    /// Whether to automatically rebuild the project when entries change
    auto_rebuild: bool,

    // Shading
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_bias: f32,
    ssao_strength: f32,
}

#[derive(Clone, Debug)]
struct CachedOperatorMetadata {
    wasm_len: u64,
    wasm_modified: Option<SystemTime>,
    metadata: Option<OperatorMetadata>,
}


fn demo_wasm_path(crate_name: &str) -> Option<PathBuf> {
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

    None
}

fn operation_wasm_path(crate_name: &str) -> Option<PathBuf> {
    // For now operations are built the same way as demo models: a cdylib .wasm in target/{debug|release}.
    demo_wasm_path(crate_name)
}

impl VolumetricApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_project: Option<Project>) -> Self {
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
            if let Some(first_model) = assets.iter().find(|a| a.as_model_wasm().is_some()) {
                let id = first_model.asset_id().to_string();
                let wasm_bytes = first_model.as_model_wasm().unwrap().to_vec();
                render_data.insert(id, AssetRenderData::new(wasm_bytes, ExportRenderMode::PointCloud));
            }
            (assets, render_data)
        });

        Self {
            project: initial_project,
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
            last_mouse_pos: None,
            last_evaluation_time: None,
            error_message: None,
            editing_entry_index: None,
            edit_config_values: HashMap::new(),
            edit_lua_script: String::new(),
            edit_input_asset_ids: Vec::new(),
            edit_output_asset_id: String::new(),
            auto_rebuild: true,

            ssao_enabled: true,
            ssao_radius: 0.08,
            ssao_bias: 0.002,
            ssao_strength: 1.6,
        }
    }

    fn operator_metadata_cached(&mut self, crate_name: &str) -> Option<OperatorMetadata> {
        let path = operation_wasm_path(crate_name)?;
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

        self.operator_metadata_cache
            .get(&path_str)
            .and_then(|c| c.metadata.clone())
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
            .find(|a| a.asset_id() == asset_id)
            .and_then(|a| a.as_model_wasm())
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

    /// Runs the current project and updates exported assets
    fn run_project(&mut self) {
        self.exported_assets.clear();
        if let Some(ref project) = self.project {
            let mut env = Environment::new();
            let start_time = std::time::Instant::now();
            match project.run(&mut env) {
                Ok(assets) => {
                    self.last_evaluation_time = Some(start_time.elapsed().as_secs_f32());
                    // Clear any previous error now that the project ran successfully.
                    self.error_message = None;
                    // Retain full exports for UX/testing
                    self.exported_assets = assets.clone();

                    // Prune render data for exports that no longer exist
                    let exported_ids: std::collections::HashSet<String> = self
                        .exported_assets
                        .iter()
                        .map(|a| a.asset_id().to_string())
                        .collect();
                    self.asset_render_data
                        .retain(|id, _| exported_ids.contains(id));

                    // Update WASM bytes for existing render data entries
                    for (asset_id, render_data) in self.asset_render_data.iter_mut() {
                        if let Some(asset) = self.exported_assets.iter().find(|a| a.asset_id() == asset_id) {
                            if let Some(wasm_bytes) = asset.as_model_wasm() {
                                render_data.wasm_bytes = wasm_bytes.to_vec();
                                render_data.needs_resample = true;
                                render_data.last_error = None;
                            }
                        }
                    }

                    // If no assets are being rendered, pick the first ModelWASM export
                    if self.asset_render_data.is_empty() {
                        if let Some(first_model) = self
                            .exported_assets
                            .iter()
                            .find(|a| a.as_model_wasm().is_some())
                        {
                            let id = first_model.asset_id().to_string();
                            let wasm_bytes = first_model.as_model_wasm().unwrap().to_vec();
                            self.asset_render_data.insert(
                                id,
                                AssetRenderData::new(wasm_bytes, ExportRenderMode::PointCloud),
                            );
                        }
                    }
                }
                Err(e) => {
                    self.last_evaluation_time = None;
                    self.error_message = Some(format!("Failed to run project: {}", e));
                }
            }
        } else {
            self.last_evaluation_time = None;
        }
    }

    /// Resamples all assets that need resampling
    fn resample_all_assets(&mut self) {
        for (asset_id, render_data) in self.asset_render_data.iter_mut() {
            if !render_data.needs_resample {
                continue;
            }
            render_data.needs_resample = false;
            let resolution = render_data.resolution;
            
            match render_data.mode {
                ExportRenderMode::None => {
                    // Should not happen - None mode assets are removed from the map
                    render_data.bounds_min = (0.0, 0.0, 0.0);
                    render_data.bounds_max = (0.0, 0.0, 0.0);
                    render_data.points = Arc::new(Vec::new());
                    render_data.triangles.clear();
                    render_data.mesh_vertices = Arc::new(Vec::new());
                }
                ExportRenderMode::PointCloud => {
                    let start_time = std::time::Instant::now();
                    match sample_model_from_bytes(&render_data.wasm_bytes, resolution) {
                        Ok((points, bounds_min, bounds_max)) => {
                            render_data.last_sample_time = Some(start_time.elapsed().as_secs_f32());
                            render_data.last_sample_count = Some(resolution * resolution * resolution);
                            render_data.bounds_min = bounds_min;
                            render_data.bounds_max = bounds_max;
                            render_data.points = Arc::new(points);
                            render_data.triangles.clear();
                            render_data.mesh_vertices = Arc::new(Vec::new());
                            render_data.last_error = None;
                        }
                        Err(e) => {
                            render_data.last_sample_time = None;
                            render_data.last_sample_count = None;
                            let msg = format!(
                                "Export '{asset_id}' failed to sample ({}):\n{}",
                                render_data.mode.label(),
                                format_anyhow_error_chain(&e)
                            );
                            render_data.last_error = Some(msg.clone());
                            if self.error_message.is_none() {
                                self.error_message = Some(msg);
                            }
                            log::error!("Failed to sample model for export '{asset_id}': {e}");
                        }
                    }
                }
                ExportRenderMode::MarchingCubes => {
                    let start_time = std::time::Instant::now();
                    match generate_marching_cubes_mesh_from_bytes(&render_data.wasm_bytes, resolution) {
                        Ok((triangles, bounds_min, bounds_max)) => {
                            render_data.last_sample_time = Some(start_time.elapsed().as_secs_f32());
                            render_data.last_sample_count = Some(resolution * resolution * resolution);
                            render_data.bounds_min = bounds_min;
                            render_data.bounds_max = bounds_max;
                            render_data.triangles = triangles;
                            render_data.mesh_vertices = Arc::new(triangles_to_mesh_vertices(&render_data.triangles));
                            render_data.points = Arc::new(Vec::new());
                            render_data.last_error = None;
                        }
                        Err(e) => {
                            render_data.last_sample_time = None;
                            render_data.last_sample_count = None;
                            let msg = format!(
                                "Export '{asset_id}' failed to mesh ({}):\n{}",
                                render_data.mode.label(),
                                format_anyhow_error_chain(&e)
                            );
                            render_data.last_error = Some(msg.clone());
                            if self.error_message.is_none() {
                                self.error_message = Some(msg);
                            }
                            log::error!("Failed to generate marching cubes mesh for export '{asset_id}': {e}");
                        }
                    }
                }
                ExportRenderMode::AdaptiveSurfaceNets => {
                    let start_time = std::time::Instant::now();
                    let config = adaptive_surface_nets::AdaptiveMeshConfig {
                        base_resolution: render_data.adaptive_base_resolution,
                        max_refinement_depth: render_data.adaptive_max_depth,
                        edge_refinement_iterations: render_data.adaptive_edge_refinement_iterations,
                        vertex_relaxation_iterations: render_data.adaptive_vertex_relaxation_iterations,
                        normal_mode: if render_data.adaptive_hq_normals {
                            adaptive_surface_nets::NormalMode::HqBisection {
                                eps_frac: 0.05,
                                bracket_frac: 0.5,
                                iterations: 10,
                            }
                        } else {
                            adaptive_surface_nets::NormalMode::Mesh
                        },
                    };
                    match generate_adaptive_mesh_from_bytes(&render_data.wasm_bytes, &config) {
                        Ok((triangles, bounds_min, bounds_max)) => {
                            let effective_res = config.base_resolution * (1 << config.max_refinement_depth);
                            render_data.last_sample_time = Some(start_time.elapsed().as_secs_f32());
                            render_data.last_sample_count = Some(effective_res * effective_res * effective_res);
                            render_data.bounds_min = bounds_min;
                            render_data.bounds_max = bounds_max;
                            render_data.triangles = triangles;
                            render_data.mesh_vertices = Arc::new(triangles_to_mesh_vertices(&render_data.triangles));
                            render_data.points = Arc::new(Vec::new());
                            render_data.last_error = None;
                        }
                        Err(e) => {
                            render_data.last_sample_time = None;
                            render_data.last_sample_count = None;
                            let msg = format!(
                                "Export '{asset_id}' failed to mesh ({}):\n{}",
                                render_data.mode.label(),
                                format_anyhow_error_chain(&e)
                            );
                            render_data.last_error = Some(msg.clone());
                            if self.error_message.is_none() {
                                self.error_message = Some(msg);
                            }
                            log::error!("Failed to generate adaptive mesh for export '{asset_id}': {e}");
                        }
                    }
                }
                ExportRenderMode::AdaptiveSurfaceNets2 => {
                    let start_time = std::time::Instant::now();
                    let config = adaptive_surface_nets_2::AdaptiveMeshConfig2 {
                        base_resolution: render_data.asn2_base_resolution,
                        max_depth: render_data.asn2_max_depth,
                        vertex_refinement_iterations: render_data.asn2_vertex_refinement_iterations,
                        normal_sample_iterations: render_data.asn2_normal_sample_iterations,
                        normal_epsilon_frac: render_data.asn2_normal_epsilon_frac,
                        num_threads: 0, // Use default parallelism
                    };
                    match generate_adaptive_mesh_v2_from_bytes(&render_data.wasm_bytes, &config) {
                        Ok((vertices, normals, indices, bounds_min, bounds_max)) => {
                            let effective_res = config.base_resolution * (1 << config.max_depth);
                            render_data.last_sample_time = Some(start_time.elapsed().as_secs_f32());
                            render_data.last_sample_count = Some(effective_res * effective_res * effective_res);
                            render_data.bounds_min = bounds_min;
                            render_data.bounds_max = bounds_max;
                            render_data.triangles.clear(); // No raw triangles for indexed mesh

                            // Convert indexed mesh to MeshVertex format
                            let mesh_vertices: Vec<marching_cubes_wgpu::MeshVertex> = vertices
                                .iter()
                                .zip(normals.iter())
                                .map(|(pos, norm)| marching_cubes_wgpu::MeshVertex {
                                    position: [pos.0, pos.1, pos.2],
                                    _pad0: 0.0,
                                    normal: [norm.0, norm.1, norm.2],
                                    _pad1: 0.0,
                                })
                                .collect();

                            render_data.mesh_vertices = Arc::new(mesh_vertices);
                            render_data.mesh_indices = Some(Arc::new(indices));
                            render_data.points = Arc::new(Vec::new());
                            render_data.last_error = None;
                        }
                        Err(e) => {
                            render_data.last_sample_time = None;
                            render_data.last_sample_count = None;
                            let msg = format!(
                                "Export '{asset_id}' failed to mesh ({}):\n{}",
                                render_data.mode.label(),
                                format_anyhow_error_chain(&e)
                            );
                            render_data.last_error = Some(msg.clone());
                            if self.error_message.is_none() {
                                self.error_message = Some(msg);
                            }
                            log::error!("Failed to generate ASN2 mesh for export '{asset_id}': {e}");
                        }
                    }
                }
            }
        }
    }

    /// Check if any asset needs resampling
    fn any_needs_resample(&self) -> bool {
        self.asset_render_data.values().any(|d| d.needs_resample)
    }

    fn camera_position(&self) -> (f32, f32, f32) {
        let x = self.camera_radius * self.camera_phi.sin() * self.camera_theta.cos();
        let y = self.camera_radius * self.camera_phi.cos();
        let z = self.camera_radius * self.camera_phi.sin() * self.camera_theta.sin();
        (x, y, z)
    }

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
    
    /// Start editing a project entry at the given index
    fn start_editing_entry(&mut self, idx: usize) {
        // First, extract all needed data from the project to avoid borrow conflicts
        let entry_data = self.project.as_ref().and_then(|project| {
            project.entries().get(idx).and_then(|entry| {
                if let ProjectEntry::ExecuteWASM(exec_entry) = entry {
                    Some((
                        exec_entry.asset_id().to_string(),
                        exec_entry.inputs().to_vec(),
                        exec_entry.outputs().to_vec(),
                    ))
                } else {
                    None
                }
            })
        });
        
        let Some((asset_id, inputs, outputs)) = entry_data else { return };
        
        self.editing_entry_index = Some(idx);
        self.edit_config_values.clear();
        self.edit_lua_script.clear();
        self.edit_input_asset_ids.clear();
        
        // Populate output asset ID from the first output (primary output)
        self.edit_output_asset_id = outputs.first()
            .map(|o| o.asset_id.clone())
            .unwrap_or_default();
        
        // Get operator metadata to understand input types
        let crate_name = asset_id.strip_prefix("op_").unwrap_or(&asset_id);
        let operator_metadata = self.operator_metadata_cached(crate_name);
        
        if let Some(ref metadata) = operator_metadata {
            // Use metadata to properly decode each input
            for (input_idx, input_meta) in metadata.inputs.iter().enumerate() {
                let input = inputs.get(input_idx);
                match input_meta {
                    OperatorMetadataInput::ModelWASM => {
                        if let Some(ExecuteWasmInput::AssetByID(id)) = input {
                            self.edit_input_asset_ids.push(Some(id.clone()));
                        } else {
                            self.edit_input_asset_ids.push(None);
                        }
                    }
                    OperatorMetadataInput::CBORConfiguration(cddl) => {
                        self.edit_input_asset_ids.push(None); // Placeholder for data inputs
                        // Decode CBOR data to populate config values
                        if let Some(ExecuteWasmInput::Data(data)) = input {
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
                        if let Some(ExecuteWasmInput::Data(data)) = input {
                            if let Ok(script) = std::str::from_utf8(data) {
                                if !script.is_empty() {
                                    self.edit_lua_script = script.to_string();
                                }
                            }
                        } else if let Some(ExecuteWasmInput::String(s)) = input {
                            self.edit_lua_script = s.clone();
                        }
                    }
                }
            }
        } else {
            // Fallback: no metadata available, use simple extraction
            for input in &inputs {
                match input {
                    ExecuteWasmInput::AssetByID(id) => {
                        self.edit_input_asset_ids.push(Some(id.clone()));
                    }
                    ExecuteWasmInput::Data(data) => {
                        // Try to interpret as UTF-8 string (Lua script)
                        if let Ok(script) = std::str::from_utf8(data) {
                            if !script.is_empty() && self.edit_lua_script.is_empty() {
                                self.edit_lua_script = script.to_string();
                            }
                        }
                        self.edit_input_asset_ids.push(None);
                    }
                    ExecuteWasmInput::String(s) => {
                        self.edit_input_asset_ids.push(None);
                        if self.edit_lua_script.is_empty() {
                            self.edit_lua_script = s.clone();
                        }
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
    }
    
    /// Show the edit panel UI for the currently selected entry
    fn show_edit_panel(&mut self, ui: &mut egui::Ui) {
        let Some(idx) = self.editing_entry_index else { return };
        
        ui.heading("Edit Entry");
        ui.separator();
        
        // Get entry info (we need to be careful about borrowing)
        let entry_info = self.project.as_ref().and_then(|p| {
            p.entries().get(idx).map(|entry| {
                match entry {
                    ProjectEntry::ExecuteWASM(exec_entry) => {
                        Some((exec_entry.asset_id().to_string(), exec_entry.inputs().to_vec()))
                    }
                    _ => None
                }
            }).flatten()
        });
        
        let Some((asset_id, inputs)) = entry_info else {
            ui.label("Entry not found or not editable");
            if ui.button("Close").clicked() {
                self.close_edit_panel();
            }
            return;
        };
        
        ui.label(format!("Editing: {}", asset_id));
        ui.add_space(8.0);
        
        // Get available input assets for dropdowns
        let input_asset_ids: Vec<String> = self
            .project
            .as_ref()
            .map(|p| {
                p.declared_assets()
                    .into_iter()
                    .filter_map(|(id, ty)| {
                        if ty == AssetType::ModelWASM {
                            Some(id)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        
        // Try to get operator metadata for this operation
        // Extract crate name from asset_id (e.g., "op_translate_operator" -> "translate_operator")
        let crate_name = asset_id.strip_prefix("op_").unwrap_or(&asset_id);
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
                                    if let Some(ExecuteWasmInput::AssetByID(id)) = inputs.get(input_idx) {
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
                                    if let ExecuteWasmInput::Data(data) = input {
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
    
    /// Apply the changes from the edit panel to the project entry
    fn apply_edit_changes(&mut self) {
        let Some(idx) = self.editing_entry_index else { return };
        
        // Get operator metadata for building new inputs
        let entry_info = self.project.as_ref().and_then(|p| {
            p.entries().get(idx).and_then(|entry| {
                if let ProjectEntry::ExecuteWASM(exec_entry) = entry {
                    Some(exec_entry.asset_id().to_string())
                } else {
                    None
                }
            })
        });
        
        let Some(asset_id) = entry_info else { return };
        let crate_name = asset_id.strip_prefix("op_").unwrap_or(&asset_id);
        let operator_metadata = self.operator_metadata_cached(crate_name);
        
        // Build new inputs based on edit state
        let new_inputs: Vec<ExecuteWasmInput> = if let Some(ref metadata) = operator_metadata {
            let mut inputs = Vec::new();
            let mut model_input_idx = 0;
            
            for input_meta in &metadata.inputs {
                match input_meta {
                    OperatorMetadataInput::ModelWASM => {
                        let asset_id = self.edit_input_asset_ids
                            .get(model_input_idx)
                            .and_then(|o| o.clone())
                            .unwrap_or_default();
                        inputs.push(ExecuteWasmInput::AssetByID(asset_id));
                        model_input_idx += 1;
                    }
                    OperatorMetadataInput::CBORConfiguration(cddl) => {
                        let fields = parse_cddl_record_schema(cddl.as_str()).unwrap_or_default();
                        let bytes = encode_config_map_to_cbor(&fields, &self.edit_config_values)
                            .unwrap_or_default();
                        inputs.push(ExecuteWasmInput::Data(bytes));
                    }
                    OperatorMetadataInput::LuaSource(_) => {
                        let script_bytes = self.edit_lua_script.as_bytes().to_vec();
                        inputs.push(ExecuteWasmInput::Data(script_bytes));
                    }
                }
            }
            inputs
        } else {
            // No metadata, keep existing inputs
            return;
        };
        
        // Update the project entry
        if let Some(ref mut project) = self.project {
            // Get the old output asset ID to find and update the corresponding ExportAsset
            let old_output_id = project.entries().get(idx).and_then(|entry| {
                if let ProjectEntry::ExecuteWASM(exec_entry) = entry {
                    exec_entry.outputs().first().map(|o| o.asset_id.clone())
                } else {
                    None
                }
            });
            
            let new_output_id = self.edit_output_asset_id.trim().to_string();
            
            if let Some(ProjectEntry::ExecuteWASM(exec_entry)) = project.entries_mut().get_mut(idx) {
                // Build new outputs with the edited output name
                let new_outputs: Vec<ExecuteWasmOutput> = exec_entry.outputs().iter().enumerate().map(|(i, old_out)| {
                    if i == 0 {
                        // Primary output uses the edited name
                        ExecuteWasmOutput::new(new_output_id.clone(), old_out.asset_type)
                    } else {
                        // Secondary outputs get a suffix
                        ExecuteWasmOutput::new(format!("{}_{}", new_output_id, i), old_out.asset_type)
                    }
                }).collect();
                
                let new_entry = ExecuteWasmEntry::new(
                    exec_entry.asset_id().to_string(),
                    new_inputs,
                    new_outputs,
                );
                *exec_entry = new_entry;
            }
            
            // Update the corresponding ExportAsset entry if the output name changed
            if let Some(old_id) = old_output_id {
                if old_id != new_output_id {
                    for entry in project.entries_mut().iter_mut() {
                        if let ProjectEntry::ExportAsset(export_id) = entry {
                            if *export_id == old_id {
                                *export_id = new_output_id.clone();
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        self.project_path = None; // Mark as modified
        self.run_project();
        self.close_edit_panel();
    }
}

impl eframe::App for VolumetricApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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
                            let label = match &self.project_path {
                                Some(p) => p.display().to_string(),
                                None => match &self.project {
                                    Some(_) => "(unsaved project)".to_string(),
                                    None => "(none loaded)".to_string(),
                                },
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
                                self.project_path = None;
                                self.asset_render_data.clear();
                                self.exported_assets.clear();
                                self.error_message = None;
                            }
                        });

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

                        // Project file operations
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
                                            self.error_message = Some(format!("Failed to load project: {}", e));
                                        }
                                    }
                                }
                            }

                            let can_save = self.project.is_some();
                            if ui.add_enabled(can_save, egui::Button::new("Save Project…")).clicked() {
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
                                                self.error_message = Some(format!("Failed to save project: {}", e));
                                            }
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
                            
                            // Handle model addition
                            if let Some(crate_name) = model_to_add {
                                match demo_wasm_path(crate_name) {
                                    Some(path) => {
                                        match fs::read(&path) {
                                            Ok(wasm_bytes) => {
                                                let asset_id = path.file_stem()
                                                    .and_then(|s| s.to_str())
                                                    .unwrap_or("model")
                                                    .to_string();
                                                if self.project.is_none() {
                                                    self.project = Some(Project::new(vec![]));
                                                }
                                                if let Some(ref mut project) = self.project {
                                                    project.insert_model_wasm(asset_id.as_str(), wasm_bytes);
                                                }
                                                self.project_path = None;
                                                self.run_project();
                                            }
                                            Err(e) => {
                                                self.error_message = Some(format!("Failed to read WASM file: {}", e));
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
                            
                            ui.add_space(8.0);
                            ui.separator();
                            
                            // Import custom WASM
                            if ui.add(egui::Button::new("📁 Import WASM…").min_size(egui::vec2(btn_width, 28.0))).clicked() {
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
                                                self.project = Some(Project::new(vec![]));
                                            }
                                            if let Some(ref mut project) = self.project {
                                                project.insert_model_wasm(asset_id.as_str(), wasm_bytes);
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
                        });
                // ═══════════════════════════════════════════════════════════════
                // TOOLBOX PANEL - Operators Section
                // ═══════════════════════════════════════════════════════════════
                ui.collapsing("⚙ Operators", |ui| {
                    ui.add_space(4.0);
                    
                    // Get available input assets for operators
                    let input_asset_ids: Vec<String> = self
                        .project
                        .as_ref()
                        .map(|p| {
                            p.declared_assets()
                                .into_iter()
                                .filter_map(|(id, ty)| {
                                    if ty == AssetType::ModelWASM {
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
                            match operation_wasm_path(crate_name) {
                                Some(path) => match fs::read(&path) {
                                    Ok(wasm_bytes) => {
                                        let input_id_a = self.operation_input_asset_id.clone().unwrap_or_default();
                                        let input_id_b = self
                                            .operation_input_asset_id_b
                                            .clone()
                                            .unwrap_or_else(|| input_id_a.clone());
                                        let output_id = self.project.as_ref()
                                            .map(|p| p.default_output_name(crate_name, Some(&input_id_a)))
                                            .unwrap_or_else(|| format!("{}_output", crate_name));
                                        let op_asset_id = format!("op_{crate_name}");

                                        let (inputs, outputs) = match self.operator_metadata_cached(crate_name) {
                                            Some(metadata) => {
                                                let mut inputs = Vec::with_capacity(metadata.inputs.len());
                                                let mut model_inputs_iter = [input_id_a.clone(), input_id_b.clone()].into_iter();
                                                for (_idx, input) in metadata.inputs.iter().enumerate() {
                                                    match input {
                                                        OperatorMetadataInput::ModelWASM => {
                                                            let id = model_inputs_iter
                                                                .next()
                                                                .unwrap_or_else(|| input_id_a.clone());
                                                            inputs.push(ExecuteWasmInput::AssetByID(id));
                                                        }
                                                        OperatorMetadataInput::CBORConfiguration(cddl) => {
                                                            let fields = parse_cddl_record_schema(cddl.as_str()).unwrap_or_default();
                                                            let bytes = encode_config_map_to_cbor(&fields, &self.operation_config_values)
                                                                .unwrap_or_default();
                                                            inputs.push(ExecuteWasmInput::Data(bytes));
                                                        }
                                                        OperatorMetadataInput::LuaSource(_) => {
                                                            let script_bytes = self.operation_lua_script.as_bytes().to_vec();
                                                            inputs.push(ExecuteWasmInput::Data(script_bytes));
                                                        }
                                                    }
                                                }

                                                let mut outputs = Vec::with_capacity(metadata.outputs.len());
                                                for (idx, output) in metadata.outputs.iter().enumerate() {
                                                    let asset_type = match output {
                                                        OperatorMetadataOutput::ModelWASM => AssetType::ModelWASM,
                                                    };
                                                    let out_id = if idx == 0 {
                                                        output_id.clone()
                                                    } else {
                                                        format!("{output_id}_{idx}")
                                                    };
                                                    outputs.push(ExecuteWasmOutput::new(out_id, asset_type));
                                                }

                                                (inputs, outputs)
                                            }
                                            None => (
                                                vec![ExecuteWasmInput::AssetByID(input_id_a.clone())],
                                                vec![ExecuteWasmOutput::new(output_id.clone(), AssetType::ModelWASM)],
                                            ),
                                        };

                                        let mut new_entry_idx: Option<usize> = None;
                                        if let Some(ref mut project) = self.project {
                                            let count_before = project.entries().len();
                                            
                                            project.insert_operation(
                                                op_asset_id.as_str(),
                                                wasm_bytes,
                                                inputs,
                                                outputs,
                                                output_id,
                                            );
                                            
                                            for (idx, entry) in project.entries().iter().enumerate().rev() {
                                                if idx >= count_before.saturating_sub(1) {
                                                    if matches!(entry, ProjectEntry::ExecuteWASM(_)) {
                                                        new_entry_idx = Some(idx);
                                                        break;
                                                    }
                                                }
                                            }
                                        }

                                        self.run_project();
                                        
                                        if let Some(idx) = new_entry_idx {
                                            self.start_editing_entry(idx);
                                        }
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
                        }
                    }
                });

                ui.separator();
                // Track actions to perform after the UI loop (to avoid borrow conflicts)
                let mut entry_to_edit: Option<usize> = None;
                let mut entry_to_delete: Option<usize> = None;
                let mut entry_to_move_up: Option<usize> = None;
                let mut entry_to_move_down: Option<usize> = None;
                
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
                            let entries = project.entries();
                            if entries.is_empty() {
                                ui.weak("No entries in project");
                            } else {
                                egui::ScrollArea::vertical()
                                    .max_height(300.0)
                                    .show(ui, |ui| {
                                        for (idx, entry) in entries.iter().enumerate() {
                                            ui.group(|ui| {
                                                ui.horizontal(|ui| {
                                                    // Step number indicator
                                                    ui.label(format!("{}.", idx + 1));

                                                    match entry {
                                                        ProjectEntry::LoadAsset(load_entry) => {
                                                            let icon = match load_entry.asset_type() {
                                                                AssetType::ModelWASM => "📦",
                                                                AssetType::OperationWASM => "⚙️",
                                                            };
                                                            ui.label(format!(
                                                                "{} Load: {}",
                                                                icon,
                                                                load_entry.asset_id()
                                                            ));
                                                        }
                                                        ProjectEntry::ExecuteWASM(exec_entry) => {
                                                            ui.vertical(|ui| {
                                                                ui.label(format!(
                                                                    "▶ Execute: {}",
                                                                    exec_entry.asset_id()
                                                                ));
                                                                // Show inputs
                                                                let inputs = exec_entry.inputs();
                                                                if !inputs.is_empty() {
                                                                    ui.indent("inputs", |ui| {
                                                                        for input in inputs {
                                                                            ui.weak(format!(
                                                                                "← {}",
                                                                                input.display()
                                                                            ));
                                                                        }
                                                                    });
                                                                }
                                                                // Show output count
                                                                let output_count = exec_entry.output_count();
                                                                if output_count > 0 {
                                                                    ui.indent("outputs", |ui| {
                                                                        ui.weak(format!(
                                                                            "→ {} output(s)",
                                                                            output_count
                                                                        ));
                                                                    });
                                                                }
                                                            });
                                                        }
                                                        ProjectEntry::ExportAsset(asset_id) => {
                                                            ui.label(format!(
                                                                "📤 Export: {}",
                                                                asset_id
                                                            ));
                                                        }
                                                    }
                                                    
                                                    // Add Edit, Delete, and Move buttons
                                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                        if ui.small_button("🗑").on_hover_text("Delete entry").clicked() {
                                                            entry_to_delete = Some(idx);
                                                        }
                                                        // Only show Edit for ExecuteWASM entries
                                                        if matches!(entry, ProjectEntry::ExecuteWASM(_)) {
                                                            if ui.small_button("✏").on_hover_text("Edit entry").clicked() {
                                                                entry_to_edit = Some(idx);
                                                            }
                                                        }
                                                        // Move down button (disabled for last entry)
                                                        let is_last = idx == entries.len() - 1;
                                                        if ui.add_enabled(!is_last, egui::Button::new("▼").small())
                                                            .on_hover_text("Move down")
                                                            .clicked()
                                                        {
                                                            entry_to_move_down = Some(idx);
                                                        }
                                                        // Move up button (disabled for first entry)
                                                        if ui.add_enabled(idx > 0, egui::Button::new("▲").small())
                                                            .on_hover_text("Move up")
                                                            .clicked()
                                                        {
                                                            entry_to_move_up = Some(idx);
                                                        }
                                                    });
                                                });
                                            });
                                            ui.add_space(2.0);
                                        }
                                    });
                            }
                        } else {
                            ui.weak("No project loaded");
                        }
                    });
                
                // Handle edit action
                if let Some(idx) = entry_to_edit {
                    self.start_editing_entry(idx);
                }
                
                // Handle delete action
                if let Some(idx) = entry_to_delete {
                    if let Some(ref mut project) = self.project {
                        project.entries_mut().remove(idx);
                        self.project_path = None; // Mark as modified
                        self.run_project();
                    }
                    // Close edit panel if we deleted the entry being edited
                    if self.editing_entry_index == Some(idx) {
                        self.editing_entry_index = None;
                    } else if let Some(edit_idx) = self.editing_entry_index {
                        // Adjust edit index if we deleted an entry before it
                        if idx < edit_idx {
                            self.editing_entry_index = Some(edit_idx - 1);
                        }
                    }
                }
                
                // Handle move up action
                if let Some(idx) = entry_to_move_up {
                    if let Some(ref mut project) = self.project {
                        if project.move_entry_up(idx) {
                            self.project_path = None; // Mark as modified
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
                if let Some(idx) = entry_to_move_down {
                    if let Some(ref mut project) = self.project {
                        if project.move_entry_down(idx) {
                            self.project_path = None; // Mark as modified
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
                            let asset_id = asset.asset_id().to_string();
                            let is_rendering = self.asset_render_data.contains_key(&asset_id);

                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label(format!(
                                        "{}: {} ({} bytes)",
                                        asset.asset_id(),
                                        asset.asset().asset_type(),
                                        asset.asset().bytes().len(),
                                    ));
                                    if is_rendering {
                                        ui.weak("(rendering)");
                                    }
                                });

                                if !asset.precursor_asset_ids().is_empty() {
                                    ui.label(format!(
                                        "precursors: {}",
                                        asset.precursor_asset_ids().join(", ")
                                    ));
                                }

                                let is_renderable = asset.as_model_wasm().is_some();
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
                                                ExportRenderMode::AdaptiveSurfaceNets,
                                                ExportRenderMode::AdaptiveSurfaceNets.label(),
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

                                // Show adaptive config options when AdaptiveSurfaceNets is selected
                                if current_mode == ExportRenderMode::AdaptiveSurfaceNets {
                                    if let Some(render_data) = self.asset_render_data.get_mut(&asset_id) {
                                        let mut base_res = render_data.adaptive_base_resolution;
                                        let mut max_depth = render_data.adaptive_max_depth;
                                        let mut auto_resample = render_data.auto_resample;
                                        let mut hq_normals = render_data.adaptive_hq_normals;
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Base Resolution:");
                                            if ui.add(egui::DragValue::new(&mut base_res).range(2..=32)).changed() {
                                                render_data.adaptive_base_resolution = base_res;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Max Depth:");
                                            if ui.add(egui::DragValue::new(&mut max_depth).range(0..=6)).changed() {
                                                render_data.adaptive_max_depth = max_depth;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });
                                        
                                        let effective_res = base_res * (1 << max_depth);
                                        ui.weak(format!("Effective resolution: {}³", effective_res));
                                        
                                        let mut edge_refine = render_data.adaptive_edge_refinement_iterations;
                                        let mut vertex_relax = render_data.adaptive_vertex_relaxation_iterations;
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Edge Refinement:");
                                            if ui.add(egui::DragValue::new(&mut edge_refine).range(0..=8)).changed() {
                                                render_data.adaptive_edge_refinement_iterations = edge_refine;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Vertex Relaxation:");
                                            if ui.add(egui::DragValue::new(&mut vertex_relax).range(0..=8)).changed() {
                                                render_data.adaptive_vertex_relaxation_iterations = vertex_relax;
                                                if auto_resample {
                                                    render_data.needs_resample = true;
                                                }
                                            }
                                        });

                                        ui.horizontal(|ui| {
                                            if ui.checkbox(&mut hq_normals, "HQ Normals").changed() {
                                                render_data.adaptive_hq_normals = hq_normals;
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
                                        let mut base_res = render_data.asn2_base_resolution;
                                        let mut max_depth = render_data.asn2_max_depth;
                                        let mut auto_resample = render_data.auto_resample;

                                        ui.horizontal(|ui| {
                                            ui.label("Base Resolution:");
                                            if ui.add(egui::DragValue::new(&mut base_res).range(2..=32)).changed() {
                                                render_data.asn2_base_resolution = base_res;
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

                                        let effective_res = base_res * (1 << max_depth);
                                        ui.weak(format!("Effective resolution: {}³", effective_res));

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
                                            ui.label("Normal Sampling:");
                                            if ui.add(egui::DragValue::new(&mut normal_iters).range(0..=16)).changed() {
                                                render_data.asn2_normal_sample_iterations = normal_iters;
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

                                if let Some(render_data) = self.asset_render_data.get(&asset_id) {
                                    if let Some(time) = render_data.last_sample_time {
                                        ui.horizontal(|ui| {
                                            ui.weak(format!("Sampled in {:.2}ms", time * 1000.0));
                                            if let Some(count) = render_data.last_sample_count {
                                                let avg = (time * 1000_000.0) / (count as f32);
                                                ui.weak(format!("({:.2}µs/sample)", avg));
                                            }
                                        });
                                    }

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
                                if let Some(wasm_bytes) = asset.as_model_wasm() {
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
            
            // Handle mouse interaction
            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(pos) = response.interact_pointer_pos() {
                    if let Some(last_pos) = self.last_mouse_pos {
                        let delta = pos - last_pos;
                        self.camera_theta += delta.x * 0.01;
                        self.camera_phi = (self.camera_phi - delta.y * 0.01)
                            .clamp(0.1, std::f32::consts::PI - 0.1);
                    }
                    self.last_mouse_pos = Some(pos);
                }
            } else {
                self.last_mouse_pos = None;
            }
            
            // Handle scroll for zoom
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                self.camera_radius = (self.camera_radius - scroll * 0.01).clamp(1.0, 20.0);
            }
            
            // Draw background
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 38));
            
            let camera_pos = self.camera_position();
            
            // Render all assets in asset_render_data - supports multiple entities in the same frame
            let aspect = rect.width() / rect.height().max(1.0);
            let view_proj = point_cloud_wgpu::view_proj_from_camera(
                camera_pos,
                (0.0, 0.0, 0.0),
                60.0_f32.to_radians(),
                aspect,
                0.1,
                100.0,
            );
            
            // Collect all point cloud data from assets using PointCloud mode
            let all_points: Vec<(f32, f32, f32)> = self.asset_render_data.values()
                .filter(|d| d.mode == ExportRenderMode::PointCloud)
                .flat_map(|d| d.points.iter().cloned())
                .collect();
            
            if !all_points.is_empty() {
                let cb = eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    point_cloud_wgpu::PointCloudCallback {
                        data: point_cloud_wgpu::PointCloudDrawData {
                            points: Arc::new(all_points.clone()),
                            camera_pos,
                            view_proj,
                            point_size_px: 3.0,
                            target_format: self.wgpu_target_format,
                        },
                    },
                );
                painter.add(egui::Shape::Callback(cb));
            }
            
            // Collect all mesh data from assets using mesh modes
            // For indexed meshes (ASN2), we can use indexed rendering if there's only one such asset
            let mesh_assets: Vec<&AssetRenderData> = self.asset_render_data.values()
                .filter(|d| matches!(d.mode,
                    ExportRenderMode::MarchingCubes |
                    ExportRenderMode::AdaptiveSurfaceNets |
                    ExportRenderMode::AdaptiveSurfaceNets2))
                .collect();

            // Determine if we can use indexed rendering (single ASN2 asset)
            let single_indexed_asset = mesh_assets.len() == 1
                && mesh_assets[0].mode == ExportRenderMode::AdaptiveSurfaceNets2
                && mesh_assets[0].mesh_indices.is_some();

            let (all_vertices, all_indices): (Vec<marching_cubes_wgpu::MeshVertex>, Option<Arc<Vec<u32>>>) =
                if single_indexed_asset {
                    // Use indexed rendering for single ASN2 asset
                    let asset = mesh_assets[0];
                    (asset.mesh_vertices.iter().cloned().collect(), asset.mesh_indices.clone())
                } else {
                    // Combine all meshes as non-indexed
                    // For indexed meshes, expand them using indices
                    let mut combined_vertices = Vec::new();
                    for asset in &mesh_assets {
                        if let Some(ref indices) = asset.mesh_indices {
                            // Expand indexed mesh
                            for &idx in indices.iter() {
                                if let Some(v) = asset.mesh_vertices.get(idx as usize) {
                                    combined_vertices.push(*v);
                                }
                            }
                        } else {
                            // Non-indexed mesh - add directly
                            combined_vertices.extend(asset.mesh_vertices.iter().cloned());
                        }
                    }
                    (combined_vertices, None)
                };

            if !all_vertices.is_empty() {
                let pixels_per_point = ui.ctx().pixels_per_point();
                let viewport_size_px = [
                    (rect.width() * pixels_per_point).round().max(1.0) as u32,
                    (rect.height() * pixels_per_point).round().max(1.0) as u32,
                ];
                let cb = eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    marching_cubes_wgpu::MarchingCubesCallback {
                        data: marching_cubes_wgpu::MarchingCubesDrawData {
                            vertices: Arc::new(all_vertices.clone()),
                            indices: all_indices,
                            view_proj,
                            viewport_size_px,
                            ssao_enabled: self.ssao_enabled,
                            ssao_radius: self.ssao_radius,
                            ssao_bias: self.ssao_bias,
                            ssao_strength: self.ssao_strength,
                            target_format: self.wgpu_target_format,
                        },
                    },
                );
                painter.add(egui::Shape::Callback(cb));
            }
            
            // Draw coordinate axes as overlay (after 3D content so they appear on top)
            let origin = (0.0, 0.0, 0.0);
            let axis_len = 0.5;
            let axes = [
                ((axis_len, 0.0, 0.0), egui::Color32::RED),
                ((0.0, axis_len, 0.0), egui::Color32::GREEN),
                ((0.0, 0.0, axis_len), egui::Color32::BLUE),
            ];
            
            if let Some(origin_screen) = self.project_point(origin, &rect) {
                for (axis_end, color) in axes {
                    if let Some(end_screen) = self.project_point(axis_end, &rect) {
                        painter.line_segment(
                            [origin_screen, end_screen],
                            egui::Stroke::new(2.0, color),
                        );
                    }
                }
            }
            
            // Draw info text showing totals across all rendered assets
            let total_points = all_points.len();
            let total_triangles: usize = self.asset_render_data.values()
                .filter(|d| matches!(d.mode,
                    ExportRenderMode::MarchingCubes |
                    ExportRenderMode::AdaptiveSurfaceNets |
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

fn triangles_to_mesh_vertices(triangles: &[Triangle]) -> Vec<marching_cubes_wgpu::MeshVertex> {
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

            out.push(marching_cubes_wgpu::MeshVertex {
                position: [v.0, v.1, v.2],
                _pad0: 0.0,
                normal,
                _pad1: 0.0,
            });
        }
    }

    out
}

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
                        let asset_id = path.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("model")
                            .to_string();
                        println!("Importing WASM model from: {}", path.display());
                        let mut project = Project::new(vec![]);
                        project.insert_model_wasm(asset_id.as_str(), wasm_bytes);
                        Some(project)
                    }
                    Err(e) => {
                        eprintln!("Failed to read WASM file: {}", e);
                        None
                    }
                }
            }
            _ => {
                eprintln!("Unknown file type: {}. Expected .wasm or .vproj", path.display());
                None
            }
        }
    } else {
        Some(Project::new(vec![]))
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
    ).map_err(|e| anyhow::anyhow!("Failed to run eframe: {}", e))?;
    
    Ok(())
}
