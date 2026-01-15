use anyhow::{Context, Result};
use eframe::egui;
use wasmtime::*;

/// Application state for the volumetric renderer
struct VolumetricApp {
    wasm_path: String,
    resolution: usize,
    bounds_min: (f32, f32, f32),
    bounds_max: (f32, f32, f32),
    points: Vec<(f32, f32, f32)>,
    needs_resample: bool,
    // Camera state
    camera_theta: f32,
    camera_phi: f32,
    camera_radius: f32,
    last_mouse_pos: Option<egui::Pos2>,
    // Error message
    error_message: Option<String>,
}

impl VolumetricApp {
    fn new(wasm_path: String) -> Self {
        Self {
            wasm_path,
            resolution: 20,
            bounds_min: (0.0, 0.0, 0.0),
            bounds_max: (0.0, 0.0, 0.0),
            points: Vec::new(),
            needs_resample: true,
            camera_theta: std::f32::consts::FRAC_PI_4,
            camera_phi: std::f32::consts::FRAC_PI_4,
            camera_radius: 4.0,
            last_mouse_pos: None,
            error_message: None,
        }
    }

    fn resample_model(&mut self) {
        match sample_model(&self.wasm_path, self.resolution) {
            Ok((points, bounds_min, bounds_max)) => {
                self.bounds_min = bounds_min;
                self.bounds_max = bounds_max;
                self.points = points;
                self.error_message = None;
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to sample model: {}", e));
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
                
                ui.label("Model Controls");
                ui.horizontal(|ui| {
                    ui.label("Resolution:");
                    if ui.add(egui::Slider::new(&mut self.resolution, 5..=50)).changed() {
                        self.needs_resample = true;
                    }
                });
                
                if ui.button("Resample Model").clicked() {
                    self.needs_resample = true;
                }
                
                ui.separator();
                ui.label("Model Info");
                ui.label(format!("Points: {}", self.points.len()));
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
                ui.collapsing("Future Options", |ui| {
                    ui.label("• Marching cubes mesh");
                    ui.label("• Ray marching");
                    ui.label("• Export to STL/OBJ");
                    ui.label("• Custom shaders");
                });
                
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
            
            // Sort points by depth for proper rendering
            let camera_pos = self.camera_position();
            let mut points_with_depth: Vec<_> = self.points.iter()
                .filter_map(|&p| {
                    let dx = p.0 - camera_pos.0;
                    let dy = p.1 - camera_pos.1;
                    let dz = p.2 - camera_pos.2;
                    let depth = dx * dx + dy * dy + dz * dz;
                    self.project_point(p, &rect).map(|screen_pos| (p, screen_pos, depth))
                })
                .collect();
            
            // Sort back to front
            points_with_depth.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            
            // Draw points
            for (point, screen_pos, depth) in points_with_depth {
                // Color based on position (simple gradient)
                let r = ((point.0 + 1.0) * 0.5 * 255.0) as u8;
                let g = ((point.1 + 1.0) * 0.5 * 255.0) as u8;
                let b = ((point.2 + 1.0) * 0.5 * 255.0) as u8;
                
                // Size based on depth
                let size = (3.0 / depth.sqrt()).clamp(1.0, 4.0);
                
                painter.circle_filled(
                    screen_pos,
                    size,
                    egui::Color32::from_rgb(r, g, b),
                );
            }
            
            // Draw info text
            painter.text(
                rect.left_top() + egui::vec2(10.0, 10.0),
                egui::Align2::LEFT_TOP,
                format!("Points: {} | Drag to rotate, scroll to zoom", self.points.len()),
                egui::FontId::default(),
                egui::Color32::WHITE,
            );
        });

        // Request continuous repaints for smooth interaction
        ctx.request_repaint();
    }
}

/// Sample points from the WASM volumetric model
fn sample_model(wasm_path: &str, resolution: usize) -> Result<(Vec<(f32, f32, f32)>, (f32, f32, f32), (f32, f32, f32))> {
    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm_path)
        .with_context(|| format!("Failed to load WASM module from {}", wasm_path))?;
    
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

fn main() -> Result<()> {
    env_logger::init();
    
    let args: Vec<String> = std::env::args().collect();
    
    let wasm_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "target/wasm32-unknown-unknown/release/test_model.wasm".to_string()
    };
    
    println!("Volumetric Model Renderer (eframe/egui)");
    println!("=======================================");
    println!("Loading WASM model from: {}", wasm_path);
    println!();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_title("Volumetric Renderer"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Volumetric Renderer",
        options,
        Box::new(|_cc| Ok(Box::new(VolumetricApp::new(wasm_path)))),
    ).map_err(|e| anyhow::anyhow!("Failed to run eframe: {}", e))?;
    
    Ok(())
}
