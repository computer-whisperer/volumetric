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
use volumetric_renderer::CameraControlScheme;

use crate::{ExecutorChoice, PreviewRenderMode, VolumetricUiV2};

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
    /// Preview mesher by its route name (`points` | `marching-cubes` |
    /// `asn2`); unknown names keep the app default.
    pub render_mode: String,
    pub preview_resolution: usize,
    pub show_grid: bool,
    pub show_bounds: bool,
    pub ssao: bool,
    pub ssao_radius: f32,
    pub ssao_bias: f32,
    pub ssao_strength: f32,
    pub auto_rebuild: bool,
    pub auto_remesh: bool,
    pub panel_width: f32,
    /// Physical window size at last exit; 0 means "no recorded size" and
    /// leaves the shell's default alone.
    pub window_width: u32,
    pub window_height: u32,
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
            render_mode: app.render_mode.route_name().to_string(),
            preview_resolution: app.preview_resolution,
            show_grid: app.show_grid,
            show_bounds: app.show_bounds,
            ssao: app.ssao,
            ssao_radius: app.ssao_radius,
            ssao_bias: app.ssao_bias,
            ssao_strength: app.ssao_strength,
            auto_rebuild: app.auto_rebuild,
            auto_remesh: app.auto_remesh,
            panel_width: app.panel_width,
            window_width,
            window_height,
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
        if let Some(mode) = PreviewRenderMode::from_route_name(&self.render_mode) {
            app.render_mode = mode;
        }
        app.preview_resolution = self.preview_resolution.clamp(8, 1024);
        app.show_grid = self.show_grid;
        app.show_bounds = self.show_bounds;
        app.ssao = self.ssao;
        app.ssao_radius = finite_or(self.ssao_radius, defaults.ssao_radius);
        app.ssao_bias = finite_or(self.ssao_bias, defaults.ssao_bias);
        app.ssao_strength = finite_or(self.ssao_strength, defaults.ssao_strength);
        app.auto_rebuild = self.auto_rebuild;
        app.auto_remesh = self.auto_remesh;
        app.panel_width = finite_or(self.panel_width, defaults.panel_width)
            .clamp(super::PANEL_WIDTH_MIN, super::PANEL_WIDTH_MAX);
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
            render_mode: "points".to_string(),
            preview_resolution: 128,
            show_grid: false,
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
            ssao_radius: f32::INFINITY,
            ..UiSettings::default()
        };

        let mut app = VolumetricUiV2::empty();
        let defaults = VolumetricUiV2::empty();
        settings.apply(&mut app);
        assert_eq!(app.camera_control_scheme, defaults.camera_control_scheme);
        assert_eq!(app.render_mode, defaults.render_mode);
        assert_eq!(app.preview_resolution, 1024);
        assert_eq!(app.panel_width, defaults.panel_width);
        assert_eq!(app.ssao_radius, defaults.ssao_radius);
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
