//! Persisted UI preferences for the native shell.
//!
//! A small JSON file under the platform config dir
//! (`~/.config/volumetric/ui-v2.json` on Linux) holding cross-project
//! preferences: the remote daemon address and toggle, viewport/render
//! options, panel and window geometry. Per-project state (pipeline,
//! pinned outputs, overrides) stays in the project file.
//!
//! The host snapshots [`UiSettings::from_app`] once per frame and rewrites
//! the file when the snapshot changes (a few hundred bytes, tmp + rename).
//! Loading tolerates hand-edits: unknown fields are ignored, missing fields
//! take defaults, out-of-range values are clamped in [`UiSettings::apply`].

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use volumetric_renderer::{CameraControlScheme, LightingPreset, OrbitMode};

use crate::{ExecutorChoice, GridDensity, PreviewRenderMode, ViewportSettings, VolumetricUiV2};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiSettings {
    /// Daemon base URL applied when remote build is toggled on.
    pub remote_address: String,
    /// Start with the remote execution backend instead of the local one.
    pub remote_build: bool,
    /// Camera scheme by [`CameraControlScheme::name`]; unknown names keep
    /// the app default.
    pub camera_control_scheme: String,
    /// Orbit drags turn freely instead of as a turntable.
    pub free_orbit: bool,
    /// The viewport camera projects orthographically.
    pub orthographic: bool,
    /// Preview mesher by its route name (`points` | `marching-cubes` |
    /// `asn2`); unknown names keep the app default.
    pub render_mode: String,
    pub preview_resolution: usize,
    pub show_grid: bool,
    /// Draw the world axis lines with the grid.
    pub show_axes: bool,
    /// Grid density by [`GridDensity::name`] (`fine` | `normal` |
    /// `coarse`); unknown names keep the default.
    pub grid_density: String,
    pub show_gizmo: bool,
    pub show_bounds: bool,
    /// Lighting by [`LightingPreset::name`] (`studio` | `flat` |
    /// `headlight`); unknown names keep the default.
    pub lighting: String,
    pub edges: bool,
    pub edge_opacity: f32,
    /// Ambient occlusion. Files written before the view settings panel
    /// called these `ssao`, `ssao_radius` (in metres, not read) and
    /// `ssao_strength`.
    #[serde(alias = "ssao")]
    pub ao: bool,
    /// A fraction of the scene's diagonal.
    pub ao_radius: f32,
    #[serde(alias = "ssao_strength")]
    pub ao_strength: f32,
    pub antialiasing: bool,
    pub auto_rebuild: bool,
    pub auto_remesh: bool,
    pub panel_width: f32,
    /// Recents-rail tiles: catalog module names, most recent first. A
    /// missing field takes the fresh-install seed; unknown names are
    /// skipped at render time.
    pub recent_adds: Vec<String>,
    /// Physical window size at last exit; 0 means "no recorded size" and
    /// leaves the shell's default alone.
    pub window_width: u32,
    pub window_height: u32,
    /// Build-cache budget preference in MiB (the live budget can sit above
    /// it while a seeded built copy is resident).
    pub cache_budget_mb: usize,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self::from_app(&VolumetricUiV2::empty(), 0, 0)
    }
}

impl UiSettings {
    /// Snapshot the persisted subset of `app`'s state.
    pub fn from_app(app: &VolumetricUiV2, window_width: u32, window_height: u32) -> Self {
        Self {
            remote_address: app.remote_address.clone(),
            remote_build: app.remote_build,
            camera_control_scheme: app.camera_control_scheme.name().to_string(),
            free_orbit: app.orbit_mode == OrbitMode::Free,
            orthographic: app.viewport.orthographic,
            render_mode: app.render_mode.route_name().to_string(),
            preview_resolution: app.preview_resolution,
            show_grid: app.viewport.show_grid,
            show_axes: app.viewport.show_axes,
            grid_density: app.viewport.grid_density.name().to_string(),
            show_gizmo: app.viewport.show_gizmo,
            show_bounds: app.show_bounds,
            lighting: app.viewport.lighting.name().to_string(),
            edges: app.viewport.edges,
            edge_opacity: app.viewport.edge_opacity,
            ao: app.viewport.ao,
            ao_radius: app.viewport.ao_radius,
            ao_strength: app.viewport.ao_strength,
            antialiasing: app.viewport.antialiasing,
            auto_rebuild: app.auto_rebuild,
            auto_remesh: app.auto_remesh,
            panel_width: app.panel_width,
            recent_adds: app.recent_adds.clone(),
            window_width,
            window_height,
            cache_budget_mb: app.cache_budget_bytes() >> 20,
        }
    }

    /// Push these settings onto `app`, clamping out-of-range values from a
    /// hand-edited file rather than rejecting them. A persisted remote_build
    /// queues the executor swap; the host applies it on the first frame.
    pub fn apply(&self, app: &mut VolumetricUiV2) {
        let defaults = VolumetricUiV2::empty();
        app.remote_address = self.remote_address.clone();
        app.remote_build = self.remote_build && !self.remote_address.trim().is_empty();
        if app.remote_build {
            // Same normalization as the settings-popover toggle path.
            app.executor_request = Some(ExecutorChoice::Remote(
                self.remote_address.trim().to_string(),
            ));
        }
        if let Some(scheme) = CameraControlScheme::ALL
            .iter()
            .find(|s| s.name() == self.camera_control_scheme)
        {
            app.camera_control_scheme = *scheme;
        }
        app.orbit_mode = if self.free_orbit {
            OrbitMode::Free
        } else {
            OrbitMode::Turntable
        };
        if let Some(mode) = PreviewRenderMode::from_route_name(&self.render_mode) {
            app.render_mode = mode;
        }
        app.preview_resolution = self.preview_resolution.clamp(8, 1024);
        app.show_bounds = self.show_bounds;
        let view = defaults.viewport;
        app.viewport = ViewportSettings {
            orthographic: self.orthographic,
            show_grid: self.show_grid,
            show_axes: self.show_axes,
            grid_density: GridDensity::from_name(&self.grid_density).unwrap_or(view.grid_density),
            show_gizmo: self.show_gizmo,
            lighting: LightingPreset::from_name(&self.lighting).unwrap_or(view.lighting),
            edges: self.edges,
            edge_opacity: finite_or(self.edge_opacity, view.edge_opacity).clamp(0.0, 1.0),
            ao: self.ao,
            ao_radius: finite_or(self.ao_radius, view.ao_radius).clamp(0.01, 0.3),
            ao_strength: finite_or(self.ao_strength, view.ao_strength).clamp(0.5, 4.0),
            antialiasing: self.antialiasing,
        };
        app.auto_rebuild = self.auto_rebuild;
        app.auto_remesh = self.auto_remesh;
        app.panel_width = finite_or(self.panel_width, defaults.panel_width)
            .clamp(super::PANEL_WIDTH_MIN, super::PANEL_WIDTH_MAX);
        app.recent_adds = self.recent_adds.clone();
        app.recent_adds.truncate(super::RECENT_ADDS_CAP);
        app.set_cache_budget(self.cache_budget_mb.clamp(64, 64 << 10) << 20);
    }

    /// `<config dir>/volumetric/ui-v2.json`; `None` when the platform has
    /// no config directory (then nothing is persisted).
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|dir| dir.join("volumetric").join("ui-v2.json"))
    }

    /// Read settings from `path`. Missing file is a silent `None` (first
    /// run); a malformed file is logged and treated as absent.
    pub fn load(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        match serde_json::from_slice(&bytes) {
            Ok(settings) => Some(settings),
            Err(err) => {
                log::warn!("ignoring malformed settings at {}: {err}", path.display());
                None
            }
        }
    }

    /// Write settings to `path` via a sibling tmp file + rename, so a crash
    /// mid-write can't truncate the previous file. Failures are logged and
    /// dropped — settings persistence must never take the UI down.
    pub fn save(&self, path: &Path) {
        let json = match serde_json::to_vec_pretty(self) {
            Ok(json) => json,
            Err(_) => return,
        };
        let result = (|| {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, &json)?;
            std::fs::rename(&tmp, path)
        })();
        if let Err(err) = result {
            log::warn!("failed to save settings to {}: {err}", path.display());
        }
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "volumetric-ui-settings-{}-{name}.json",
            std::process::id()
        ))
    }

    #[test]
    fn missing_fields_take_defaults() {
        let parsed: UiSettings =
            serde_json::from_str(r#"{"remote_address": "http://daemon:7373"}"#).unwrap();
        assert_eq!(parsed.remote_address, "http://daemon:7373");
        assert_eq!(
            UiSettings {
                remote_address: UiSettings::default().remote_address,
                ..parsed
            },
            UiSettings::default()
        );
    }

    #[test]
    fn apply_then_snapshot_round_trips() {
        let settings = UiSettings {
            remote_address: "http://daemon:7373".to_string(),
            camera_control_scheme: "Maya".to_string(),
            free_orbit: true,
            orthographic: true,
            render_mode: "points".to_string(),
            preview_resolution: 128,
            show_grid: false,
            show_axes: false,
            grid_density: "coarse".to_string(),
            show_gizmo: false,
            lighting: "flat".to_string(),
            edges: false,
            edge_opacity: 0.35,
            ao: false,
            ao_radius: 0.12,
            ao_strength: 2.0,
            antialiasing: false,
            auto_remesh: false,
            panel_width: 300.0,
            ..UiSettings::default()
        };

        let mut app = VolumetricUiV2::empty();
        settings.apply(&mut app);
        assert_eq!(UiSettings::from_app(&app, 0, 0), settings);
    }

    #[test]
    fn apply_queues_remote_swap() {
        let settings = UiSettings {
            remote_build: true,
            remote_address: "  http://daemon:7373 ".to_string(),
            ..UiSettings::default()
        };

        let mut app = VolumetricUiV2::empty();
        settings.apply(&mut app);
        assert!(app.remote_build);
        assert_eq!(
            app.take_executor_request(),
            Some(ExecutorChoice::Remote("http://daemon:7373".to_string()))
        );
    }

    #[test]
    fn remote_build_without_address_stays_local() {
        let settings = UiSettings {
            remote_build: true,
            remote_address: "   ".to_string(),
            ..UiSettings::default()
        };

        let mut app = VolumetricUiV2::empty();
        settings.apply(&mut app);
        assert!(!app.remote_build);
        assert_eq!(app.take_executor_request(), None);
    }

    #[test]
    fn hand_edited_values_are_sanitized() {
        let settings = UiSettings {
            camera_control_scheme: "Cinema4D".to_string(),
            render_mode: "raytraced".to_string(),
            preview_resolution: 100_000,
            panel_width: f32::NAN,
            ao_radius: f32::INFINITY,
            lighting: "neon".to_string(),
            ..UiSettings::default()
        };

        let mut app = VolumetricUiV2::empty();
        let defaults = VolumetricUiV2::empty();
        settings.apply(&mut app);
        assert_eq!(app.camera_control_scheme, defaults.camera_control_scheme);
        assert_eq!(app.render_mode, defaults.render_mode);
        assert_eq!(app.preview_resolution, 1024);
        assert_eq!(app.panel_width, defaults.panel_width);
        assert_eq!(app.viewport.ao_radius, defaults.viewport.ao_radius);
        assert_eq!(app.viewport.lighting, defaults.viewport.lighting);
    }

    #[test]
    fn save_load_round_trips() {
        let path = scratch_path("round-trip");
        let settings = UiSettings {
            remote_address: "http://daemon:7373".to_string(),
            window_width: 1920,
            window_height: 1080,
            ..UiSettings::default()
        };

        settings.save(&path);
        let loaded = UiSettings::load(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded, Some(settings));
    }

    #[test]
    fn malformed_file_loads_as_none() {
        let path = scratch_path("malformed");
        std::fs::write(&path, b"{ not json").unwrap();
        let loaded = UiSettings::load(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded, None);
    }
}
