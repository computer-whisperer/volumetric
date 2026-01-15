use anyhow::{Context, Result};
use eframe::egui;
use ciborium::value::Value as CborValue;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::sync::Arc;
use wasmtime::*;

mod point_cloud_wgpu;
mod marching_cubes_wgpu;

// Project system
mod lib;
use lib::{
    Asset,
    AssetType,
    Environment,
    ExecuteWasmEntry,
    ExecuteWasmInput,
    ExecuteWasmOutput,
    LoadAssetEntry,
    LoadedAsset,
    Project,
    ProjectEntry,
    OperatorMetadata,
    OperatorMetadataInput,
    OperatorMetadataOutput,
    operator_metadata_from_wasm_bytes,
};

#[derive(Clone, Debug)]
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
}

impl ExportRenderMode {
    fn label(self) -> &'static str {
        match self {
            ExportRenderMode::None => "None",
            ExportRenderMode::PointCloud => "Point Cloud",
            ExportRenderMode::MarchingCubes => "Marching Cubes",
        }
    }
}

/// A triangle in 3D space
type Triangle = [(f32, f32, f32); 3];

// Marching cubes lookup tables
mod marching_cubes_tables;
mod marching_cubes_cpu;
mod stl;

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
    /// Mesh vertices (for MarchingCubes mode)
    mesh_vertices: Arc<Vec<marching_cubes_wgpu::MeshVertex>>,
    /// Whether this asset needs resampling
    needs_resample: bool,

    /// Last sampling/meshing error for this asset (shown in the GUI)
    last_error: Option<String>,
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
            needs_resample: true,
            last_error: None,
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
    demo_choice: DemoChoice,
    operation_choice: OperationChoice,
    operation_input_asset_id: Option<String>,
    operation_input_asset_id_b: Option<String>,
    operation_output_asset_id: String,
    operation_config_last_cddl: Option<String>,
    operation_config_values: HashMap<String, ConfigValue>,
    operator_metadata_cache: HashMap<String, CachedOperatorMetadata>,
    resolution: usize,
    wgpu_target_format: wgpu::TextureFormat,
    // Camera state
    camera_theta: f32,
    camera_phi: f32,
    camera_radius: f32,
    last_mouse_pos: Option<egui::Pos2>,
    // Error message
    error_message: Option<String>,
}

#[derive(Clone, Debug)]
struct CachedOperatorMetadata {
    wasm_len: u64,
    wasm_modified: Option<SystemTime>,
    metadata: Option<OperatorMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationChoice {
    Identity,
    Translate,
    Boolean,
}

impl OperationChoice {
    fn label(self) -> &'static str {
        match self {
            OperationChoice::Identity => "Identity (WASM passthrough)",
            OperationChoice::Translate => "Translate (+1 X)",
            OperationChoice::Boolean => "Boolean (union/subtract/intersect)",
        }
    }

    fn crate_name(self) -> &'static str {
        match self {
            OperationChoice::Identity => "identity_operator",
            OperationChoice::Translate => "translate_operator",
            OperationChoice::Boolean => "boolean_operator",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemoChoice {
    None,
    Sphere,
    Torus,
    RoundedBox,
    GyroidLattice,
    Mandelbulb,
}

impl DemoChoice {
    fn label(self) -> &'static str {
        match self {
            DemoChoice::None => "(none)",
            DemoChoice::Sphere => "Sphere",
            DemoChoice::Torus => "Torus",
            DemoChoice::RoundedBox => "Rounded box",
            DemoChoice::GyroidLattice => "Gyroid lattice",
            DemoChoice::Mandelbulb => "Mandelbulb fractal",
        }
    }

    fn crate_name(self) -> Option<&'static str> {
        match self {
            DemoChoice::None => None,
            DemoChoice::Sphere => Some("simple_sphere_model"),
            DemoChoice::Torus => Some("simple_torus_model"),
            DemoChoice::RoundedBox => Some("rounded_box_model"),
            DemoChoice::GyroidLattice => Some("gyroid_lattice_model"),
            DemoChoice::Mandelbulb => Some("mandelbulb_model"),
        }
    }
}

fn demo_wasm_path(crate_name: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let release = root
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join(format!("{crate_name}.wasm"));
    if fs::metadata(&release).is_ok() {
        return Some(release);
    }

    let debug = root
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
            demo_choice: DemoChoice::None,
            operation_choice: OperationChoice::Identity,
            operation_input_asset_id: None,
            operation_input_asset_id_b: None,
            operation_output_asset_id: "identity_out".to_string(),
            operation_config_last_cddl: None,
            operation_config_values: HashMap::new(),
            operator_metadata_cache: HashMap::new(),
            resolution: 20,
            wgpu_target_format,
            camera_theta: std::f32::consts::FRAC_PI_4,
            camera_phi: std::f32::consts::FRAC_PI_4,
            camera_radius: 4.0,
            last_mouse_pos: None,
            error_message: None,
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
            match project.run(&mut env) {
                Ok(assets) => {
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
                    self.error_message = Some(format!("Failed to run project: {}", e));
                }
            }
        }
    }

    /// Resamples all assets that need resampling
    fn resample_all_assets(&mut self) {
        let resolution = self.resolution;
        
        for (asset_id, render_data) in self.asset_render_data.iter_mut() {
            if !render_data.needs_resample {
                continue;
            }
            render_data.needs_resample = false;
            
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
                    match sample_model_from_bytes(&render_data.wasm_bytes, resolution) {
                        Ok((points, bounds_min, bounds_max)) => {
                            render_data.bounds_min = bounds_min;
                            render_data.bounds_max = bounds_max;
                            render_data.points = Arc::new(points);
                            render_data.triangles.clear();
                            render_data.mesh_vertices = Arc::new(Vec::new());
                            render_data.last_error = None;
                        }
                        Err(e) => {
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
                    match generate_marching_cubes_mesh_from_bytes(&render_data.wasm_bytes, resolution) {
                        Ok((triangles, bounds_min, bounds_max)) => {
                            render_data.bounds_min = bounds_min;
                            render_data.bounds_max = bounds_max;
                            render_data.triangles = triangles;
                            render_data.mesh_vertices = Arc::new(triangles_to_mesh_vertices(&render_data.triangles));
                            render_data.points = Arc::new(Vec::new());
                            render_data.last_error = None;
                        }
                        Err(e) => {
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
            }
        }
    }
    
    /// Check if any asset needs resampling
    fn any_needs_resample(&self) -> bool {
        self.asset_render_data.values().any(|d| d.needs_resample)
    }
    
    /// Mark all assets as needing resample (e.g., when resolution changes)
    fn mark_all_needs_resample(&mut self) {
        for render_data in self.asset_render_data.values_mut() {
            render_data.needs_resample = true;
        }
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
                            ui.label("Demo:");
                            egui::ComboBox::from_id_salt("demo_choice")
                                .selected_text(self.demo_choice.label())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.demo_choice, DemoChoice::None, DemoChoice::None.label());
                                    ui.selectable_value(&mut self.demo_choice, DemoChoice::Sphere, DemoChoice::Sphere.label());
                                    ui.selectable_value(&mut self.demo_choice, DemoChoice::Torus, DemoChoice::Torus.label());
                                    ui.selectable_value(
                                        &mut self.demo_choice,
                                        DemoChoice::RoundedBox,
                                        DemoChoice::RoundedBox.label(),
                                    );
                                    ui.selectable_value(
                                        &mut self.demo_choice,
                                        DemoChoice::GyroidLattice,
                                        DemoChoice::GyroidLattice.label(),
                                    );
                                    ui.selectable_value(
                                        &mut self.demo_choice,
                                        DemoChoice::Mandelbulb,
                                        DemoChoice::Mandelbulb.label(),
                                    );
                                });

                            let can_load_demo = self.demo_choice.crate_name().is_some();
                            if ui
                                .add_enabled(can_load_demo, egui::Button::new("Load demo"))
                                .clicked()
                            {
                                if let Some(crate_name) = self.demo_choice.crate_name() {
                                    match demo_wasm_path(crate_name) {
                                        Some(path) => {
                                            // Load WASM file and create a project from it
                                            match fs::read(&path) {
                                                Ok(wasm_bytes) => {
                                                    let asset_id = path.file_stem()
                                                        .and_then(|s| s.to_str())
                                                        .unwrap_or("model")
                                                        .to_string();
                                                    self.project = Some(Project::from_model_wasm(asset_id, wasm_bytes));
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
                                                "Demo WASM not found for '{crate_name}'. Build it first with: cargo build --release --target wasm32-unknown-unknown -p {crate_name}"
                                            ));
                                        }
                                    }
                                }
                            }
                        });

                        ui.horizontal(|ui| {
                            // Import a raw WASM file as a new project
                            if ui.button("Import WASM…").clicked() {
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
                                            self.project = Some(Project::from_model_wasm(asset_id, wasm_bytes));
                                            self.project_path = None;
                                            self.run_project();
                                            self.demo_choice = DemoChoice::None;
                                        }
                                        Err(e) => {
                                            self.error_message =
                                                Some(format!("Failed to read WASM file: {}", e));
                                        }
                                    }
                                }
                            }

                            let can_unload = self.project.is_some();
                            if ui
                                .add_enabled(can_unload, egui::Button::new("Unload"))
                                .clicked()
                            {
                                self.project = None;
                                self.project_path = None;
                                self.asset_render_data.clear();
                                self.exported_assets.clear();
                                self.demo_choice = DemoChoice::None;
                                self.error_message = None;
                            }
                        });

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
                                    self.demo_choice = DemoChoice::None;
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
                ui.label("Operations");

                // Populate input dropdowns from all assets declared by the project (including
                // outputs of earlier operation steps), filtering by the type required for model
                // inputs.
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
                    // Default B to the second asset if possible, else fall back to A.
                    self.operation_input_asset_id_b = input_asset_ids
                        .get(1)
                        .cloned()
                        .or_else(|| self.operation_input_asset_id.clone());
                }

                ui.horizontal(|ui| {
                    ui.label("Operation:");
                    egui::ComboBox::from_id_salt("operation_choice")
                        .selected_text(self.operation_choice.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.operation_choice,
                                OperationChoice::Identity,
                                OperationChoice::Identity.label(),
                            );
                            ui.selectable_value(
                                &mut self.operation_choice,
                                OperationChoice::Translate,
                                OperationChoice::Translate.label(),
                            );
                            ui.selectable_value(
                                &mut self.operation_choice,
                                OperationChoice::Boolean,
                                OperationChoice::Boolean.label(),
                            );
                        });
                });

                // If the selected operator declares configuration inputs, render widgets for them.
                // The encoded CBOR bytes will be inserted into `ExecuteWasmInput::Data` when adding the operation.
                let crate_name = self.operation_choice.crate_name();
                let operator_metadata = self.operator_metadata_cached(crate_name);

                if let Some(ref metadata) = operator_metadata {
                    for (input_idx, input) in metadata.inputs.iter().enumerate() {
                        if let OperatorMetadataInput::CBORConfiguration(cddl) = input {
                            let cddl_trimmed = cddl.trim().to_string();
                            if self.operation_config_last_cddl.as_deref() != Some(cddl_trimmed.as_str()) {
                                self.operation_config_last_cddl = Some(cddl_trimmed.clone());
                                self.operation_config_values.clear();
                            }

                            ui.separator();
                            ui.label(format!("Configuration (input {input_idx})"));

                            match parse_cddl_record_schema(&cddl_trimmed) {
                                Ok(fields) => {
                                    for (field_name, field_ty) in &fields {
                                        ui.horizontal(|ui| {
                                            ui.label(field_name);
                                            match field_ty {
                                                ConfigFieldType::Bool => {
                                                    let entry = self
                                                        .operation_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert(ConfigValue::Bool(false));
                                                    if let ConfigValue::Bool(b) = entry {
                                                        ui.checkbox(b, "");
                                                    }
                                                }
                                                ConfigFieldType::Int => {
                                                    let entry = self
                                                        .operation_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert(ConfigValue::Int(0));
                                                    if let ConfigValue::Int(i) = entry {
                                                        ui.add(egui::DragValue::new(i));
                                                    }
                                                }
                                                ConfigFieldType::Float => {
                                                    let entry = self
                                                        .operation_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert(ConfigValue::Float(0.0));
                                                    if let ConfigValue::Float(f) = entry {
                                                        ui.add(egui::DragValue::new(f));
                                                    }
                                                }
                                                ConfigFieldType::Text => {
                                                    let entry = self
                                                        .operation_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert_with(|| ConfigValue::Text(String::new()));
                                                    if let ConfigValue::Text(t) = entry {
                                                        ui.text_edit_singleline(t);
                                                    }
                                                }
                                                ConfigFieldType::Enum(options) => {
                                                    let entry = self
                                                        .operation_config_values
                                                        .entry(field_name.clone())
                                                        .or_insert_with(|| {
                                                            ConfigValue::Text(options.first().cloned().unwrap_or_default())
                                                        });

                                                    if let ConfigValue::Text(selected) = entry {
                                                        egui::ComboBox::from_id_salt(format!("cfg_enum_{field_name}"))
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

                                    if ui
                                        .add(egui::Button::new("Reset config"))
                                        .clicked()
                                    {
                                        self.operation_config_values.clear();
                                    }
                                }
                                Err(e) => {
                                    ui.colored_label(egui::Color32::YELLOW, format!(
                                        "Unsupported configuration schema: {e}"
                                    ));
                                }
                            }
                        }
                    }
                    ui.separator();
                }

                // Render input pickers. If the operator expects 2+ model inputs, show A/B.
                let model_input_count = operator_metadata
                    .as_ref()
                    .map(|m| {
                        m.inputs
                            .iter()
                            .filter(|i| matches!(i, OperatorMetadataInput::ModelWASM))
                            .count()
                    })
                    .unwrap_or(1);

                ui.horizontal(|ui| {
                    ui.label(if model_input_count >= 2 {
                        "Input asset A:"
                    } else {
                        "Input asset:"
                    });
                    let selected = self.operation_input_asset_id.as_deref().unwrap_or("(none)");
                    egui::ComboBox::from_id_salt("operation_input_asset_a")
                        .selected_text(selected)
                        .show_ui(ui, |ui| {
                            for id in &input_asset_ids {
                                ui.selectable_value(
                                    &mut self.operation_input_asset_id,
                                    Some(id.clone()),
                                    id,
                                );
                            }
                        });
                });

                if model_input_count >= 2 {
                    ui.horizontal(|ui| {
                        ui.label("Input asset B:");
                        let selected = self.operation_input_asset_id_b.as_deref().unwrap_or("(none)");
                        egui::ComboBox::from_id_salt("operation_input_asset_b")
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                for id in &input_asset_ids {
                                    ui.selectable_value(
                                        &mut self.operation_input_asset_id_b,
                                        Some(id.clone()),
                                        id,
                                    );
                                }
                            });
                    });
                }

                ui.horizontal(|ui| {
                    ui.label("Output asset id:");
                    ui.text_edit_singleline(&mut self.operation_output_asset_id);
                });

                let has_required_inputs = if model_input_count >= 2 {
                    self.operation_input_asset_id.is_some() && self.operation_input_asset_id_b.is_some()
                } else {
                    self.operation_input_asset_id.is_some()
                };
                let can_add_op = self.project.is_some()
                    && has_required_inputs
                    && !self.operation_output_asset_id.trim().is_empty();
                if ui
                    .add_enabled(can_add_op, egui::Button::new("Add New Operation"))
                    .clicked()
                {
                    let crate_name = self.operation_choice.crate_name();
                    match operation_wasm_path(crate_name) {
                        Some(path) => match fs::read(&path) {
                            Ok(wasm_bytes) => {
                                let input_id_a = self.operation_input_asset_id.clone().unwrap();
                                let input_id_b = self
                                    .operation_input_asset_id_b
                                    .clone()
                                    .unwrap_or_else(|| input_id_a.clone());
                                let output_id = self.operation_output_asset_id.trim().to_string();
                                let op_asset_id = format!("op_{crate_name}");

                                let (inputs, outputs) = match self.operator_metadata_cached(crate_name) {
                                    Some(metadata) => {
                                        // Use operator metadata to decide how many inputs/outputs to declare.
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

                                if let Some(ref mut project) = self.project {
                                    let insert_at = project
                                        .entries()
                                        .iter()
                                        .position(|e| matches!(e, ProjectEntry::ExportAsset(_)))
                                        .unwrap_or(project.entries().len());

                                    let entries = project.entries_mut();
                                    entries.insert(
                                        insert_at,
                                        ProjectEntry::LoadAsset(LoadAssetEntry::new(
                                            op_asset_id.clone(),
                                            Asset::OperationWASM(wasm_bytes),
                                        )),
                                    );
                                    entries.insert(
                                        insert_at + 1,
                                        ProjectEntry::ExecuteWASM(ExecuteWasmEntry::new(
                                            op_asset_id,
                                            inputs,
                                            outputs,
                                        )),
                                    );
                                    entries.insert(
                                        insert_at + 2,
                                        ProjectEntry::ExportAsset(output_id),
                                    );
                                }

                                self.run_project();
                            }
                            Err(e) => {
                                self.error_message =
                                    Some(format!("Failed to read operation WASM file: {e}"));
                            }
                        },
                        None => {
                            self.error_message = Some(format!(
                                "Operation WASM not found for '{crate_name}'. Build it first with: cargo build --release --target wasm32-unknown-unknown -p {crate_name}"
                            ));
                        }
                    }
                }

                        ui.label("Model Controls");
                        ui.horizontal(|ui| {
                            ui.label("Resolution:");
                            if ui.add(egui::Slider::new(&mut self.resolution, 5..=100)).changed() {
                                self.mark_all_needs_resample();
                            }
                        });

                        if ui.button("Resample All").clicked() {
                            self.mark_all_needs_resample();
                        }

                        ui.separator();
                        ui.label("Export");
                        // Check if any asset has triangles for STL export
                        let has_triangles = self
                            .asset_render_data
                            .values()
                            .any(|d| !d.triangles.is_empty());
                        if ui
                            .add_enabled(has_triangles, egui::Button::new("Export STL…"))
                            .clicked()
                        {
                            // Collect all triangles from all assets
                            let all_triangles: Vec<Triangle> = self
                                .asset_render_data
                                .values()
                                .flat_map(|d| d.triangles.iter().cloned())
                                .collect();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("STL", &["stl"])
                                .set_file_name("mesh.stl")
                                .save_file()
                            {
                                match stl::write_binary_stl(&path, &all_triangles, "volumetric") {
                                    Ok(()) => self.error_message = None,
                                    Err(e) => {
                                        self.error_message = Some(format!("Failed to export STL: {e}"))
                                    }
                                }
                            }
                        }

                ui.separator();
                egui::CollapsingHeader::new("Project Timeline")
                    .default_open(true)
                    .show(ui, |ui| {
                        if let Some(ref project) = self.project {
                            let entries = project.entries();
                            if entries.is_empty() {
                                ui.weak("No entries in project");
                            } else {
                                egui::ScrollArea::vertical()
                                    .max_height(200.0)
                                    .show(ui, |ui| {
                                        for (idx, entry) in entries.iter().enumerate() {
                                            ui.horizontal(|ui| {
                                                // Step number indicator
                                                ui.label(format!("{}.", idx + 1));

                                                match entry {
                                                    lib::ProjectEntry::LoadAsset(load_entry) => {
                                                        let icon = match load_entry.asset_type() {
                                                            lib::AssetType::ModelWASM => "📦",
                                                            lib::AssetType::OperationWASM => "⚙️",
                                                        };
                                                        ui.label(format!(
                                                            "{} Load: {}",
                                                            icon,
                                                            load_entry.asset_id()
                                                        ));
                                                    }
                                                    lib::ProjectEntry::ExecuteWASM(exec_entry) => {
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
                                                    lib::ProjectEntry::ExportAsset(asset_id) => {
                                                        ui.label(format!(
                                                            "📤 Export: {}",
                                                            asset_id
                                                        ));
                                                    }
                                                }
                                            });
                                            ui.add_space(2.0);
                                        }
                                    });
                            }
                        } else {
                            ui.weak("No project loaded");
                        }
                    });

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
                                        });

                                    if mode != current_mode {
                                        self.set_asset_render_mode(&asset_id, mode);
                                    }
                                });

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
            
            // Collect all mesh vertices from assets using MarchingCubes mode
            let all_vertices: Vec<marching_cubes_wgpu::MeshVertex> = self.asset_render_data.values()
                .filter(|d| d.mode == ExportRenderMode::MarchingCubes)
                .flat_map(|d| d.mesh_vertices.iter().cloned())
                .collect();
            
            if !all_vertices.is_empty() {
                let cb = eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    marching_cubes_wgpu::MarchingCubesCallback {
                        data: marching_cubes_wgpu::MarchingCubesDrawData {
                            vertices: Arc::new(all_vertices.clone()),
                            view_proj,
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
                .filter(|d| d.mode == ExportRenderMode::MarchingCubes)
                .map(|d| d.triangles.len())
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

/// Sample points from the WASM volumetric model
fn sample_model(wasm_path: &Path, resolution: usize) -> Result<(Vec<(f32, f32, f32)>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm_path)
        .with_context(|| format!("Failed to load WASM module from {}", wasm_path.display()))?;

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])
        .context("Failed to instantiate WASM module")?;
    
    let is_inside = instance
        .get_typed_func::<(f32, f32, f32), i32>(&mut store, "is_inside")
        .context("Failed to get 'is_inside' function")?;
    
    let get_bounds_min_x = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_z")?;
    
    let min_x = get_bounds_min_x.call(&mut store, ())?;
    let min_y = get_bounds_min_y.call(&mut store, ())?;
    let min_z = get_bounds_min_z.call(&mut store, ())?;
    let max_x = get_bounds_max_x.call(&mut store, ())?;
    let max_y = get_bounds_max_y.call(&mut store, ())?;
    let max_z = get_bounds_max_z.call(&mut store, ())?;
    
    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);
    
    let mut points = Vec::new();
    
    for z_idx in 0..resolution {
        let z = min_z + (max_z - min_z) * (z_idx as f32 / (resolution - 1) as f32);
        for y_idx in 0..resolution {
            let y = min_y + (max_y - min_y) * (y_idx as f32 / (resolution - 1) as f32);
            for x_idx in 0..resolution {
                let x = min_x + (max_x - min_x) * (x_idx as f32 / (resolution - 1) as f32);
                let inside = is_inside.call(&mut store, (x, y, z))?;
                if inside != 0 {
                    points.push((x, y, z));
                }
            }
        }
    }
    
    Ok((points, bounds_min, bounds_max))
}

/// Generate a mesh using marching cubes algorithm from the WASM volumetric model
fn generate_marching_cubes_mesh(wasm_path: &Path, resolution: usize) -> Result<(Vec<Triangle>, (f32, f32, f32), (f32, f32, f32))> {
    
    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm_path)
        .with_context(|| format!("Failed to load WASM module from {}", wasm_path.display()))?;
    
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])
        .context("Failed to instantiate WASM module")?;
    
    let is_inside = instance
        .get_typed_func::<(f32, f32, f32), i32>(&mut store, "is_inside")
        .context("Failed to get 'is_inside' function")?;
    
    let get_bounds_min_x = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_z")?;
    
    let min_x = get_bounds_min_x.call(&mut store, ())?;
    let min_y = get_bounds_min_y.call(&mut store, ())?;
    let min_z = get_bounds_min_z.call(&mut store, ())?;
    let max_x = get_bounds_max_x.call(&mut store, ())?;
    let max_y = get_bounds_max_y.call(&mut store, ())?;
    let max_z = get_bounds_max_z.call(&mut store, ())?;
    
    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);

    let triangles = marching_cubes_cpu::marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
        Ok(is_inside.call(&mut store, p)? != 0)
    })?;

    Ok((triangles, bounds_min, bounds_max))
}

/// Sample points from WASM bytes (in-memory model)
fn sample_model_from_bytes(wasm_bytes: &[u8], resolution: usize) -> Result<(Vec<(f32, f32, f32)>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes)
        .context("Failed to load WASM module from bytes")?;

    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])
        .context("Failed to instantiate WASM module")?;
    
    let is_inside = instance
        .get_typed_func::<(f32, f32, f32), i32>(&mut store, "is_inside")
        .context("Failed to get 'is_inside' function")?;
    
    let get_bounds_min_x = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_z")?;
    
    let min_x = get_bounds_min_x.call(&mut store, ())?;
    let min_y = get_bounds_min_y.call(&mut store, ())?;
    let min_z = get_bounds_min_z.call(&mut store, ())?;
    let max_x = get_bounds_max_x.call(&mut store, ())?;
    let max_y = get_bounds_max_y.call(&mut store, ())?;
    let max_z = get_bounds_max_z.call(&mut store, ())?;
    
    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);
    
    let mut points = Vec::new();
    
    for z_idx in 0..resolution {
        let z = min_z + (max_z - min_z) * (z_idx as f32 / (resolution - 1) as f32);
        for y_idx in 0..resolution {
            let y = min_y + (max_y - min_y) * (y_idx as f32 / (resolution - 1) as f32);
            for x_idx in 0..resolution {
                let x = min_x + (max_x - min_x) * (x_idx as f32 / (resolution - 1) as f32);
                let inside = is_inside.call(&mut store, (x, y, z))?;
                if inside != 0 {
                    points.push((x, y, z));
                }
            }
        }
    }
    
    Ok((points, bounds_min, bounds_max))
}

/// Generate a mesh using marching cubes algorithm from WASM bytes (in-memory model)
fn generate_marching_cubes_mesh_from_bytes(wasm_bytes: &[u8], resolution: usize) -> Result<(Vec<Triangle>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes)
        .context("Failed to load WASM module from bytes")?;
    
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])
        .context("Failed to instantiate WASM module")?;
    
    let is_inside = instance
        .get_typed_func::<(f32, f32, f32), i32>(&mut store, "is_inside")
        .context("Failed to get 'is_inside' function")?;
    
    let get_bounds_min_x = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_x")?;
    let get_bounds_min_y = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_y")?;
    let get_bounds_min_z = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_min_z")?;
    let get_bounds_max_x = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_x")?;
    let get_bounds_max_y = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_y")?;
    let get_bounds_max_z = instance.get_typed_func::<(), f32>(&mut store, "get_bounds_max_z")?;
    
    let min_x = get_bounds_min_x.call(&mut store, ())?;
    let min_y = get_bounds_min_y.call(&mut store, ())?;
    let min_z = get_bounds_min_z.call(&mut store, ())?;
    let max_x = get_bounds_max_x.call(&mut store, ())?;
    let max_y = get_bounds_max_y.call(&mut store, ())?;
    let max_z = get_bounds_max_z.call(&mut store, ())?;
    
    let bounds_min = (min_x, min_y, min_z);
    let bounds_max = (max_x, max_y, max_z);

    let triangles = marching_cubes_cpu::marching_cubes_mesh(bounds_min, bounds_max, resolution, |p| {
        Ok(is_inside.call(&mut store, p)? != 0)
    })?;

    Ok((triangles, bounds_min, bounds_max))
}

fn triangles_to_mesh_vertices(triangles: &[Triangle]) -> Vec<marching_cubes_wgpu::MeshVertex> {
    let mut out = Vec::with_capacity(triangles.len() * 3);

    for tri in triangles {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];

        let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
        let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
        let n = (
            ab.1 * ac.2 - ab.2 * ac.1,
            ab.2 * ac.0 - ab.0 * ac.2,
            ab.0 * ac.1 - ab.1 * ac.0,
        );
        let len = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
        let n = if len > 1.0e-12 {
            (n.0 / len, n.1 / len, n.2 / len)
        } else {
            (0.0, 1.0, 0.0)
        };

        let normal = [n.0, n.1, n.2];

        out.push(marching_cubes_wgpu::MeshVertex {
            position: [a.0, a.1, a.2],
            _pad0: 0.0,
            normal,
            _pad1: 0.0,
        });
        out.push(marching_cubes_wgpu::MeshVertex {
            position: [b.0, b.1, b.2],
            _pad0: 0.0,
            normal,
            _pad1: 0.0,
        });
        out.push(marching_cubes_wgpu::MeshVertex {
            position: [c.0, c.1, c.2],
            _pad0: 0.0,
            normal,
            _pad1: 0.0,
        });
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
                // Import WASM file as a new project
                match fs::read(&path) {
                    Ok(wasm_bytes) => {
                        let asset_id = path.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("model")
                            .to_string();
                        println!("Importing WASM model from: {}", path.display());
                        Some(Project::from_model_wasm(asset_id, wasm_bytes))
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
        None
    };
    
    println!("Volumetric Model Renderer (eframe/egui)");
    println!("=======================================");
    if initial_project.is_none() {
        println!("No model provided on CLI; start by importing a WASM file or opening a project in the UI.");
    }
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
