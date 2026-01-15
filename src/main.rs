use anyhow::{Context, Result};
use eframe::egui;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wasmtime::*;

mod point_cloud_wgpu;
mod marching_cubes_wgpu;

// Project system
mod lib;
use lib::{Project, Environment};

/// Rendering mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    PointCloud,
    MarchingCubes,
}

/// A triangle in 3D space
type Triangle = [(f32, f32, f32); 3];

// Marching cubes lookup tables
mod marching_cubes_tables;
mod marching_cubes_cpu;
mod stl;

/// Application state for the volumetric renderer
struct VolumetricApp {
    /// The current project (contains the model pipeline)
    project: Option<Project>,
    /// Path to the current project file (for save operations)
    project_path: Option<PathBuf>,
    /// Cached model WASM bytes (extracted from running the project)
    model_wasm: Option<Vec<u8>>,
    demo_choice: DemoChoice,
    resolution: usize,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    points: Arc<Vec<(f32, f32, f32)>>,
    triangles: Vec<Triangle>,
    mesh_vertices: Arc<Vec<marching_cubes_wgpu::MeshVertex>>,
    needs_resample: bool,
    render_mode: RenderMode,
    wgpu_target_format: wgpu::TextureFormat,
    // Camera state
    camera_theta: f32,
    camera_phi: f32,
    camera_radius: f32,
    last_mouse_pos: Option<egui::Pos2>,
    // Error message
    error_message: Option<String>,
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

impl VolumetricApp {
    fn new(cc: &eframe::CreationContext<'_>, initial_project: Option<Project>) -> Self {
        let wgpu_target_format = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| rs.target_format)
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

        // If we have an initial project, run it to extract the model WASM
        let model_wasm = initial_project.as_ref().and_then(|proj| {
            let mut env = Environment::new();
            proj.run(&mut env).ok().and_then(|assets| {
                assets.into_iter().find_map(|asset| asset.as_model_wasm().map(|b| b.to_vec()))
            })
        });

        Self {
            project: initial_project,
            project_path: None,
            model_wasm,
            demo_choice: DemoChoice::None,
            resolution: 20,
            bounds_min: (0.0, 0.0, 0.0),
            bounds_max: (0.0, 0.0, 0.0),
            points: Arc::new(Vec::new()),
            triangles: Vec::new(),
            mesh_vertices: Arc::new(Vec::new()),
            needs_resample: true,
            render_mode: RenderMode::PointCloud,
            wgpu_target_format,
            camera_theta: std::f32::consts::FRAC_PI_4,
            camera_phi: std::f32::consts::FRAC_PI_4,
            camera_radius: 4.0,
            last_mouse_pos: None,
            error_message: None,
        }
    }

    /// Runs the current project and extracts the model WASM bytes
    fn run_project(&mut self) {
        self.model_wasm = None;
        if let Some(ref project) = self.project {
            let mut env = Environment::new();
            match project.run(&mut env) {
                Ok(assets) => {
                    // Find the first ModelWASM asset in the exports
                    self.model_wasm = assets.into_iter()
                        .find_map(|asset| asset.as_model_wasm().map(|b| b.to_vec()));
                    if self.model_wasm.is_none() {
                        self.error_message = Some("Project produced no ModelWASM output".to_string());
                    }
                }
                Err(e) => {
                    self.error_message = Some(format!("Failed to run project: {}", e));
                }
            }
        }
    }

    fn resample_model(&mut self) {
        let Some(ref wasm_bytes) = self.model_wasm else {
            self.bounds_min = (0.0, 0.0, 0.0);
            self.bounds_max = (0.0, 0.0, 0.0);
            self.points = Arc::new(Vec::new());
            self.triangles.clear();
            self.mesh_vertices = Arc::new(Vec::new());
            self.error_message = None;
            return;
        };

        match self.render_mode {
            RenderMode::PointCloud => {
                match sample_model_from_bytes(wasm_bytes, self.resolution) {
                    Ok((points, bounds_min, bounds_max)) => {
                        self.bounds_min = bounds_min;
                        self.bounds_max = bounds_max;
                        self.points = Arc::new(points);
                        self.triangles.clear();
                        self.mesh_vertices = Arc::new(Vec::new());
                        self.error_message = None;
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Failed to sample model: {}", e));
                    }
                }
            }
            RenderMode::MarchingCubes => {
                match generate_marching_cubes_mesh_from_bytes(wasm_bytes, self.resolution) {
                    Ok((triangles, bounds_min, bounds_max)) => {
                        self.bounds_min = bounds_min;
                        self.bounds_max = bounds_max;
                        self.triangles = triangles;
                        self.mesh_vertices = Arc::new(triangles_to_mesh_vertices(&self.triangles));
                        self.points = Arc::new(Vec::new());
                        self.error_message = None;
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Failed to generate marching cubes mesh: {e}"));
                    }
                }
            }
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
        // Check if we need to resample
        if self.needs_resample {
            self.needs_resample = false;
            self.resample_model();
        }

        // Left panel with controls
        egui::SidePanel::left("controls")
            .default_width(250.0)
            .show(ctx, |ui| {
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
                                            self.error_message = None;
                                            self.needs_resample = true;
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
                                    let asset_id = path.file_stem()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("model")
                                        .to_string();
                                    self.project = Some(Project::from_model_wasm(asset_id, wasm_bytes));
                                    self.project_path = None;
                                    self.run_project();
                                    self.demo_choice = DemoChoice::None;
                                    self.error_message = None;
                                    self.needs_resample = true;
                                }
                                Err(e) => {
                                    self.error_message = Some(format!("Failed to read WASM file: {}", e));
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
                        self.model_wasm = None;
                        self.demo_choice = DemoChoice::None;
                        self.error_message = None;
                        self.needs_resample = true;
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
                                    self.error_message = None;
                                    self.needs_resample = true;
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

                
                ui.label("Model Controls");
                ui.horizontal(|ui| {
                    ui.label("Resolution:");
                    if ui.add(egui::Slider::new(&mut self.resolution, 5..=100)).changed() {
                        self.needs_resample = true;
                    }
                });
                
                if ui.button("Resample Model").clicked() {
                    self.needs_resample = true;
                }
                
                ui.separator();
                ui.label("Model Info");
                if self.model_wasm.is_none() {
                    ui.weak("No model loaded. Import a WASM file or open a project.");
                }
                match self.render_mode {
                    RenderMode::PointCloud => {
                        ui.label(format!("Points: {}", self.points.len()));
                    }
                    RenderMode::MarchingCubes => {
                        ui.label(format!("Triangles: {}", self.triangles.len()));
                    }
                }
                ui.label(format!(
                    "Bounds: ({:.2}, {:.2}, {:.2})",
                    self.bounds_min.0, self.bounds_min.1, self.bounds_min.2
                ));
                ui.label(format!(
                    "     to ({:.2}, {:.2}, {:.2})",
                    self.bounds_max.0, self.bounds_max.1, self.bounds_max.2
                ));
                
                ui.separator();
                ui.label("Camera Controls");
                ui.label("• Left-drag to rotate");
                ui.label("• Scroll to zoom");
                
                ui.horizontal(|ui| {
                    ui.label("Zoom:");
                    ui.add(egui::Slider::new(&mut self.camera_radius, 1.0..=20.0));
                });
                
                ui.separator();
                ui.label("Render Mode");
                let mut changed = false;
                changed |= ui
                    .selectable_value(&mut self.render_mode, RenderMode::PointCloud, "Point Cloud (wgpu)")
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.render_mode,
                        RenderMode::MarchingCubes,
                        "Marching Cubes (wgpu mesh)",
                    )
                    .changed();
                if changed {
                    self.needs_resample = true;
                }

                ui.separator();
                ui.label("Export");
                let can_export_stl = self.render_mode == RenderMode::MarchingCubes && !self.triangles.is_empty();
                if ui
                    .add_enabled(can_export_stl, egui::Button::new("Export STL…"))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("STL", &["stl"])
                        .set_file_name("mesh.stl")
                        .save_file()
                    {
                        match stl::write_binary_stl(&path, &self.triangles, "volumetric") {
                            Ok(()) => self.error_message = None,
                            Err(e) => self.error_message = Some(format!("Failed to export STL: {e}")),
                        }
                    }
                }
                
                if let Some(ref error) = self.error_message {
                    ui.separator();
                    ui.colored_label(egui::Color32::RED, error);
                }
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
            
            // Draw coordinate axes
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
            
            let camera_pos = self.camera_position();
            
            match self.render_mode {
                RenderMode::PointCloud => {
                    let aspect = rect.width() / rect.height().max(1.0);
                    let view_proj = point_cloud_wgpu::view_proj_from_camera(
                        camera_pos,
                        (0.0, 0.0, 0.0),
                        60.0_f32.to_radians(),
                        aspect,
                        0.1,
                        100.0,
                    );

                    let cb = eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        point_cloud_wgpu::PointCloudCallback {
                            data: point_cloud_wgpu::PointCloudDrawData {
                                points: self.points.clone(),
                                camera_pos,
                                view_proj,
                                point_size_px: 3.0,
                                target_format: self.wgpu_target_format,
                            },
                        },
                    );

                    painter.add(egui::Shape::Callback(cb));
                }
                RenderMode::MarchingCubes => {
                    let aspect = rect.width() / rect.height().max(1.0);
                    let view_proj = point_cloud_wgpu::view_proj_from_camera(
                        camera_pos,
                        (0.0, 0.0, 0.0),
                        60.0_f32.to_radians(),
                        aspect,
                        0.1,
                        100.0,
                    );

                    let cb = eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        marching_cubes_wgpu::MarchingCubesCallback {
                            data: marching_cubes_wgpu::MarchingCubesDrawData {
                                vertices: self.mesh_vertices.clone(),
                                view_proj,
                                target_format: self.wgpu_target_format,
                            },
                        },
                    );

                    painter.add(egui::Shape::Callback(cb));
                }
            }
            
            // Draw info text
            let info_text = match self.render_mode {
                RenderMode::PointCloud => format!("Points: {} | Drag to rotate, scroll to zoom", self.points.len()),
                RenderMode::MarchingCubes => format!("Triangles: {} | Drag to rotate, scroll to zoom", self.triangles.len()),
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
