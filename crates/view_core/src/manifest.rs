//! Importing a posed-image dataset from its manifest into a `ViewSet`.
//!
//! Two manifests are read: the scanner's `cameras.json` (schema 2: one
//! rectified pinhole `K`, `views` with OpenCV `camtoworld` matrices, `depth/`
//! and optional `masks/` files beside `images/`, the marker map), and
//! nerfstudio's `transforms.json` (OpenGL camera axes, global or per-frame
//! intrinsics with OpenCV or fisheye distortion, `depth_file_path`). A
//! dataset can hold thousands of views and gigabytes of images; the
//! [`Selection`] picks the subset a project embeds.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use volumetric_abi::viewset::{CameraModel, Distortion, Marker, View, ViewSet};

/// Which manifest a file is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestKind {
    /// The scanner's `cameras.json`.
    Scanner,
    /// nerfstudio's `transforms.json`.
    Nerfstudio,
}

impl ManifestKind {
    pub fn detect(json: &Value) -> Result<Self> {
        if json.get("views").is_some() && json.get("K").is_some() {
            Ok(Self::Scanner)
        } else if json.get("frames").is_some() {
            Ok(Self::Nerfstudio)
        } else {
            bail!("not a cameras.json (views + K) or transforms.json (frames) manifest")
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Scanner => "scanner cameras.json",
            Self::Nerfstudio => "nerfstudio transforms.json",
        }
    }
}

/// Which stereo eye to keep, for datasets that tag views with one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Eye {
    Left,
    Right,
    #[default]
    Both,
}

/// Which views to embed. Filters apply in order: eye and split tags, ids,
/// nearness, then every `stride`th survivor, then at most `max`.
#[derive(Clone, Debug)]
pub struct Selection {
    /// Exact view ids to keep; empty keeps every view.
    pub ids: Vec<String>,
    pub stride: usize,
    /// Keep views whose camera lies within a radius of a world point.
    pub near: Option<([f64; 3], f64)>,
    /// At most this many views; 0 for no cap.
    pub max: usize,
    pub eye: Eye,
    /// Keep views carrying this split tag (`train`, `test`).
    pub split: Option<String>,
    pub images: bool,
    pub depth: bool,
    pub masks: bool,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            ids: Vec::new(),
            stride: 1,
            near: None,
            max: 0,
            eye: Eye::Both,
            split: None,
            images: true,
            depth: true,
            masks: true,
        }
    }
}

/// Provenance the importer cannot read from the manifest.
#[derive(Clone, Debug, Default)]
pub struct Labels {
    pub session: Option<String>,
    pub rig: Option<String>,
    pub field: Option<String>,
    pub setup: Option<String>,
}

/// What an import did.
#[derive(Clone, Debug, Default)]
pub struct ImportReport {
    pub kind: &'static str,
    pub total: usize,
    pub selected: usize,
    pub with_image: usize,
    pub with_depth: usize,
    pub with_mask: usize,
    /// Embedded bytes.
    pub bytes: usize,
}

/// A view the manifest describes, before its files are read.
struct Candidate {
    view: View,
    image: Option<PathBuf>,
    depth: Option<PathBuf>,
    mask: Option<PathBuf>,
}

/// Reads a manifest, selects views, and embeds their files.
pub fn import_manifest(
    path: &Path,
    selection: &Selection,
    labels: &Labels,
) -> Result<(ViewSet, ImportReport)> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read manifest {}", path.display()))?;
    let json: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse manifest {}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let kind = ManifestKind::detect(&json)?;
    let (mut set, candidates) = match kind {
        ManifestKind::Scanner => read_scanner(&json, base)?,
        ManifestKind::Nerfstudio => read_nerfstudio(&json, base)?,
    };

    let chosen = select(&candidates, selection)?;
    let mut report = ImportReport {
        kind: kind.name(),
        total: candidates.len(),
        selected: chosen.len(),
        ..ImportReport::default()
    };
    for candidate in chosen {
        let mut view = candidate.view.clone();
        if selection.images
            && let Some(file) = &candidate.image
        {
            let bytes =
                std::fs::read(file).with_context(|| format!("read image {}", file.display()))?;
            report.bytes += bytes.len();
            report.with_image += 1;
            view.image = Some(bytes);
        }
        if selection.depth
            && let Some(file) = &candidate.depth
        {
            let bytes = std::fs::read(file)
                .with_context(|| format!("read depth map {}", file.display()))?;
            report.bytes += bytes.len();
            report.with_depth += 1;
            view.depth = Some(bytes);
        }
        if selection.masks
            && let Some(file) = &candidate.mask
            && file.is_file()
        {
            let bytes =
                std::fs::read(file).with_context(|| format!("read mask {}", file.display()))?;
            report.bytes += bytes.len();
            report.with_mask += 1;
            view.mask = Some(bytes);
        }
        set.views.push(view);
    }

    let provenance = &mut set.provenance;
    if let Some(session) = &labels.session {
        provenance.session = session.clone();
    }
    if let Some(rig) = &labels.rig {
        provenance.rig = rig.clone();
    }
    if let Some(field) = &labels.field {
        provenance.field = field.clone();
    }
    if let Some(setup) = &labels.setup {
        provenance.setup = setup.clone();
    }
    provenance.tools.push(format!(
        "volumetric view_core {}",
        env!("CARGO_PKG_VERSION")
    ));
    set.validate()
        .map_err(|err| anyhow!("imported set is invalid: {err}"))?;
    Ok((set, report))
}

fn select<'a>(candidates: &'a [Candidate], selection: &Selection) -> Result<Vec<&'a Candidate>> {
    let views: Vec<&View> = candidates.iter().map(|c| &c.view).collect();
    let chosen = select_views(&views, selection)?;
    Ok(chosen
        .into_iter()
        .map(|view| {
            candidates
                .iter()
                .find(|c| std::ptr::eq(&c.view, view))
                .expect("a chosen view is one of the candidates")
        })
        .collect())
}

/// The views of `views` the selection keeps, in the selection's order:
/// the eye and split tags, nearness to a point, then the named ids (in
/// the order given), every `stride`th survivor, and at most `max`. An
/// empty result is an error.
pub fn select_views<'a>(views: &[&'a View], selection: &Selection) -> Result<Vec<&'a View>> {
    let has_tag = |view: &View, tag: &str| view.tags.iter().any(|t| t == tag);
    let mut kept: Vec<&View> = views
        .iter()
        .copied()
        .filter(|view| match selection.eye {
            Eye::Both => true,
            Eye::Left => has_tag(view, "left") || !has_tag(view, "right"),
            Eye::Right => has_tag(view, "right") || !has_tag(view, "left"),
        })
        .filter(|view| {
            selection
                .split
                .as_deref()
                .is_none_or(|split| has_tag(view, split))
        })
        .filter(|view| {
            selection.near.is_none_or(|(point, radius)| {
                view.position().is_some_and(|p| {
                    let d = [p[0] - point[0], p[1] - point[1], p[2] - point[2]];
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() <= radius
                })
            })
        })
        .collect();
    if !selection.ids.is_empty() {
        let mut by_id = Vec::with_capacity(selection.ids.len());
        for id in &selection.ids {
            let view = kept
                .iter()
                .find(|view| view.id == *id)
                .ok_or_else(|| anyhow!("no view {id:?} passes the filters"))?;
            by_id.push(*view);
        }
        kept = by_id;
    }
    if selection.stride > 1 {
        kept = kept
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % selection.stride == 0)
            .map(|(_, c)| c)
            .collect();
    }
    if selection.max > 0 {
        kept.truncate(selection.max);
    }
    if kept.is_empty() {
        bail!("the selection matches no view");
    }
    Ok(kept)
}

fn number(value: &Value, what: &str) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| anyhow!("manifest field {what} is not a number"))
}

fn matrix_rows(value: &Value, what: &str) -> Result<Vec<Vec<f64>>> {
    let rows = value
        .as_array()
        .ok_or_else(|| anyhow!("{what} is not a matrix"))?;
    rows.iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| anyhow!("{what} row is not an array"))?
                .iter()
                .map(|v| number(v, what))
                .collect()
        })
        .collect()
}

/// The 3x4 camera-to-world rows of a 4x4 or 3x4 matrix, columns optionally
/// negated (to convert an OpenGL camera to OpenCV) and translation scaled.
fn pose_from(rows: &[Vec<f64>], negate_yz: bool, scale: f64, what: &str) -> Result<[f64; 12]> {
    if rows.len() < 3 || rows.iter().take(3).any(|r| r.len() < 4) {
        bail!("{what} must be a 3x4 or 4x4 matrix");
    }
    let mut pose = [0.0; 12];
    for r in 0..3 {
        for c in 0..4 {
            let mut v = rows[r][c];
            if negate_yz && (c == 1 || c == 2) {
                v = -v;
            }
            if c == 3 {
                v *= scale;
            }
            pose[r * 4 + c] = v;
        }
    }
    Ok(pose)
}

fn stem_of(file: &str) -> String {
    Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file)
        .to_string()
}

fn read_scanner(json: &Value, base: &Path) -> Result<(ViewSet, Vec<Candidate>)> {
    let width = number(&json["width"], "width")? as u32;
    let height = number(&json["height"], "height")? as u32;
    let k = matrix_rows(&json["K"], "K")?;
    if k.len() < 3 || k.iter().any(|r| r.len() < 3) {
        bail!("K must be a 3x3 matrix");
    }
    let camera = CameraModel::pinhole(width, height, k[0][0], k[1][1], k[0][2], k[1][2]);
    let depth_unit_m = json
        .get("depth_unit_m")
        .and_then(Value::as_f64)
        .unwrap_or(1e-4);

    let mut set = ViewSet {
        board: None,
        cameras: vec![camera],
        ..ViewSet::default()
    };
    set.provenance.session = json
        .get("session")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    set.provenance.tools.push(format!(
        "index_scanner cameras.json schema {}",
        json.get("schema").and_then(Value::as_u64).unwrap_or(0)
    ));

    if let Some(markers) = json.get("markers").and_then(Value::as_object) {
        let sizes = markers.get("marker_size_m").and_then(Value::as_object);
        if let Some(corners) = markers.get("corners").and_then(Value::as_object) {
            for (id, quad) in corners {
                let rows = matrix_rows(quad, "marker corners")?;
                if rows.len() != 4 || rows.iter().any(|r| r.len() != 3) {
                    bail!("marker {id} must have four xyz corners");
                }
                let size_m = sizes
                    .and_then(|s| s.get(id))
                    .and_then(Value::as_f64)
                    .ok_or_else(|| anyhow!("marker {id} has no size"))?;
                set.markers.push(Marker {
                    id: id.parse().with_context(|| format!("marker id {id:?}"))?,
                    size_m,
                    corners: std::array::from_fn(|c| [rows[c][0], rows[c][1], rows[c][2]]),
                });
            }
            set.markers.sort_by_key(|m| m.id);
        }
    }

    let views = json["views"]
        .as_array()
        .ok_or_else(|| anyhow!("views is not an array"))?;
    let mut candidates = Vec::with_capacity(views.len());
    for entry in views {
        let file = entry["file"]
            .as_str()
            .ok_or_else(|| anyhow!("a view has no file"))?;
        let rows = matrix_rows(&entry["camtoworld"], "camtoworld")?;
        let mut view = View::posed(
            stem_of(file),
            0,
            pose_from(&rows, false, 1.0, "camtoworld")?,
        );
        for key in ["eye", "split"] {
            if let Some(tag) = entry.get(key).and_then(Value::as_str) {
                view.tags.push(tag.to_string());
            }
        }
        if let Some(frame) = entry.get("frame").and_then(Value::as_u64) {
            view.tags.push(format!("frame:{frame}"));
        }
        view.depth_unit_m = depth_unit_m;
        let file_name = Path::new(file).file_name().map(PathBuf::from);
        candidates.push(Candidate {
            view,
            image: Some(base.join(file)),
            depth: entry
                .get("depth")
                .and_then(Value::as_str)
                .map(|d| base.join(d)),
            mask: file_name.map(|name| base.join("masks").join(name)),
        });
    }
    Ok((set, candidates))
}

fn read_nerfstudio(json: &Value, base: &Path) -> Result<(ViewSet, Vec<Candidate>)> {
    let world_unit_m = json
        .get("world_unit_m")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let depth_unit_m = json
        .get("depth_unit_m")
        .and_then(Value::as_f64)
        .unwrap_or(1e-3);
    let model = json
        .get("camera_model")
        .and_then(Value::as_str)
        .unwrap_or("OPENCV")
        .to_string();

    let camera_for = |frame: &Value| -> Result<CameraModel> {
        let field = |name: &str| -> Option<f64> {
            frame
                .get(name)
                .or_else(|| json.get(name))
                .and_then(Value::as_f64)
        };
        let need = |name: &str| field(name).ok_or_else(|| anyhow!("intrinsic {name} missing"));
        let (w, h) = (need("w")? as u32, need("h")? as u32);
        let fx = need("fl_x")?;
        let fy = field("fl_y").unwrap_or(fx);
        let cx = field("cx").unwrap_or(f64::from(w) / 2.0);
        let cy = field("cy").unwrap_or(f64::from(h) / 2.0);
        let k: Vec<f64> = ["k1", "k2", "k3", "k4"]
            .iter()
            .map(|n| field(n).unwrap_or(0.0))
            .collect();
        let p = [field("p1").unwrap_or(0.0), field("p2").unwrap_or(0.0)];
        let distortion = if model.contains("FISHEYE") {
            Distortion::KannalaBrandt {
                k: [k[0], k[1], k[2], k[3]],
            }
        } else if k[..3].iter().chain(p.iter()).all(|c| *c == 0.0) {
            Distortion::None
        } else {
            let mut radial = k[..3].to_vec();
            while radial.last() == Some(&0.0) {
                radial.pop();
            }
            Distortion::Radial { k: radial, p }
        };
        Ok(CameraModel {
            label: String::new(),
            width: w,
            height: h,
            fx,
            fy,
            cx,
            cy,
            distortion,
        })
    };

    let mut set = ViewSet::default();
    set.provenance
        .tools
        .push("nerfstudio transforms.json".to_string());
    let frames = json["frames"]
        .as_array()
        .ok_or_else(|| anyhow!("frames is not an array"))?;
    let mut candidates = Vec::with_capacity(frames.len());
    for frame in frames {
        let file = frame["file_path"]
            .as_str()
            .ok_or_else(|| anyhow!("a frame has no file_path"))?;
        let camera = camera_for(frame)?;
        let camera_index = match set.cameras.iter().position(|c| *c == camera) {
            Some(i) => i,
            None => {
                set.cameras.push(camera);
                set.cameras.len() - 1
            }
        };
        let rows = matrix_rows(&frame["transform_matrix"], "transform_matrix")?;
        let pose = pose_from(&rows, true, world_unit_m, "transform_matrix")?;
        let mut view = View::posed(stem_of(file), camera_index as u32, pose);
        view.depth_unit_m = depth_unit_m;
        candidates.push(Candidate {
            view,
            image: Some(base.join(file)),
            depth: frame
                .get("depth_file_path")
                .and_then(Value::as_str)
                .map(|d| base.join(d)),
            mask: frame
                .get("mask_path")
                .and_then(Value::as_str)
                .map(|m| base.join(m)),
        });
    }
    Ok((set, candidates))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Rgb;

    fn write_png(path: &Path, w: u32, h: u32) {
        std::fs::write(path, Rgb::new(w, h).to_png().unwrap()).unwrap();
    }

    fn write_depth(path: &Path, w: u32, h: u32) {
        let buffer = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_pixel(
            w,
            h,
            image::Luma([15_000]),
        );
        let mut png = std::io::Cursor::new(Vec::new());
        buffer.write_to(&mut png, image::ImageFormat::Png).unwrap();
        std::fs::write(path, png.into_inner()).unwrap();
    }

    fn scanner_dataset(dir: &Path, views: usize) -> PathBuf {
        std::fs::create_dir_all(dir.join("images")).unwrap();
        std::fs::create_dir_all(dir.join("depth")).unwrap();
        std::fs::create_dir_all(dir.join("masks")).unwrap();
        let mut entries = Vec::new();
        for i in 0..views {
            for eye in ["l", "r"] {
                let name = format!("{i:05}_{eye}.png");
                write_png(&dir.join("images").join(&name), 4, 4);
                let mut entry = serde_json::json!({
                    "file": format!("images/{name}"),
                    "frame": i,
                    "eye": if eye == "l" { "left" } else { "right" },
                    "camtoworld": [[1,0,0, i as f64],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                    "split": if i % 3 == 0 { "test" } else { "train" },
                });
                if eye == "l" {
                    write_depth(&dir.join("depth").join(&name), 4, 4);
                    write_png(&dir.join("masks").join(&name), 4, 4);
                    entry["depth"] = serde_json::json!(format!("depth/{name}"));
                }
                entries.push(entry);
            }
        }
        let manifest = serde_json::json!({
            "schema": 2, "session": "sessions/test", "width": 4, "height": 4,
            "K": [[400.0, 0.0, 2.0], [0.0, 410.0, 2.5], [0.0, 0.0, 1.0]],
            "baseline_m": 0.135, "views": entries, "depth_unit_m": 1e-4,
            "markers": {
                "marker_size_m": {"3": 0.06},
                "corners": {"3": [[0,0,0],[0.06,0,0],[0.06,0,0.06],[0,0,0.06]]},
                "rms_px": 0.9
            }
        });
        let path = dir.join("cameras.json");
        std::fs::write(&path, manifest.to_string()).unwrap();
        path
    }

    #[test]
    fn scanner_manifest_imports_a_selection_with_files() {
        let dir = std::env::temp_dir().join(format!("view_core_scanner_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let manifest = scanner_dataset(&dir, 6);

        let selection = Selection {
            eye: Eye::Left,
            split: Some("train".to_string()),
            stride: 2,
            ..Selection::default()
        };
        let labels = Labels {
            field: Some("field-1".to_string()),
            ..Labels::default()
        };
        let (set, report) = import_manifest(&manifest, &selection, &labels).unwrap();
        // Left eyes of train frames 1, 2, 4, 5; every second: frames 1 and 4.
        let ids: Vec<&str> = set.views.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(ids, ["00001_l", "00004_l"]);
        assert_eq!(report.total, 12);
        assert_eq!(
            (
                report.selected,
                report.with_image,
                report.with_depth,
                report.with_mask
            ),
            (2, 2, 2, 2)
        );
        assert!(report.bytes > 0);
        assert_eq!(set.cameras[0].fx, 400.0);
        assert_eq!(set.cameras[0].cy, 2.5);
        assert_eq!(set.views[0].position(), Some([1.0, 0.0, 0.0]));
        assert!(set.views[0].tags.contains(&"frame:1".to_string()));
        assert_eq!(set.views[0].depth_unit_m, 1e-4);
        assert_eq!(set.markers.len(), 1);
        assert_eq!(set.markers[0].size_m, 0.06);
        assert_eq!(set.provenance.session, "sessions/test");
        assert_eq!(set.provenance.field, "field-1");
        assert!(set.provenance.tools[0].contains("schema 2"));

        // Ids, nearness and caps.
        let by_id = Selection {
            ids: vec!["00003_r".to_string()],
            depth: false,
            ..Selection::default()
        };
        let (set, report) = import_manifest(&manifest, &by_id, &labels).unwrap();
        assert_eq!(set.views[0].id, "00003_r");
        assert!(set.views[0].depth.is_none() && report.with_depth == 0);
        let near = Selection {
            near: Some(([5.0, 0.0, 0.0], 1.5)),
            max: 3,
            ..Selection::default()
        };
        let (set, _) = import_manifest(&manifest, &near, &labels).unwrap();
        assert_eq!(set.views.len(), 3);
        assert!(set.views.iter().all(|v| v.position().unwrap()[0] >= 4.0));
        let missing = Selection {
            ids: vec!["nope".to_string()],
            ..Selection::default()
        };
        assert!(import_manifest(&manifest, &missing, &labels).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nerfstudio_manifest_converts_axes_and_intrinsics() {
        let dir = std::env::temp_dir().join(format!("view_core_ns_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("images")).unwrap();
        write_png(&dir.join("images/a.jpg"), 4, 4);
        write_png(&dir.join("images/b.jpg"), 4, 4);
        let manifest = serde_json::json!({
            "camera_model": "OPENCV",
            "fl_x": 300.0, "fl_y": 310.0, "cx": 2.0, "cy": 2.0, "w": 4, "h": 4,
            "k1": 0.01, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            "world_unit_m": 0.5,
            "frames": [
                {"file_path": "images/a.jpg",
                 "transform_matrix": [[1,0,0,2],[0,1,0,4],[0,0,1,6],[0,0,0,1]]},
                {"file_path": "images/b.jpg", "fl_x": 500.0,
                 "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
            ]
        });
        let path = dir.join("transforms.json");
        std::fs::write(&path, manifest.to_string()).unwrap();
        let (set, report) =
            import_manifest(&path, &Selection::default(), &Labels::default()).unwrap();
        assert_eq!(report.kind, "nerfstudio transforms.json");
        assert_eq!(set.cameras.len(), 2);
        assert_eq!(
            set.cameras[0].distortion,
            Distortion::Radial {
                k: vec![0.01],
                p: [0.0, 0.0]
            }
        );
        assert_eq!(set.cameras[1].fx, 500.0);
        // OpenGL's camera looks along -z; in OpenCV it looks along +z.
        assert_eq!(set.views[0].forward(), Some([0.0, 0.0, -1.0]));
        assert_eq!(set.views[0].axis(1), Some([0.0, -1.0, 0.0]));
        assert_eq!(set.views[0].position(), Some([1.0, 2.0, 3.0]));
        assert_eq!(set.views[1].camera, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
