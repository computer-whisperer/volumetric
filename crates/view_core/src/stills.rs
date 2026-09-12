//! Importing a directory of stills straight from a camera into a
//! `ViewSet`: one unposed view per picture with its shot state, one
//! camera model per focus setting, seeded from the focal length. The
//! survey poses the views and solves the cameras; this is the intake.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use cv_core::exif::{Exif, focal_px_from_fov, read_exif};
use rayon::prelude::*;
use volumetric_abi::viewset::{CameraModel, Provenance, View, ViewSet};

use crate::image::{decode_rgb, dimensions_of};
use crate::manifest::Labels;

/// How much of each picture the set carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Embed {
    /// The original file.
    Full,
    /// A reduced JPEG for look-through and thumbnails; detection and
    /// export read the original through `source`.
    Preview,
    /// Only the reference to the original.
    None,
}

#[derive(Clone, Debug)]
pub struct StillsOptions {
    /// Files with these extensions (case-insensitive) are stills.
    pub extensions: Vec<String>,
    pub embed: Embed,
    /// The preview's longer side, pixels.
    pub preview_px: u32,
    /// JPEG quality of the preview.
    pub preview_quality: u8,
    /// The sensor width, millimetres, when the body is not in the table:
    /// the focal seed is the focal length over it.
    pub sensor_mm: Option<f64>,
    /// Horizontal field of view to seed the focal when the picture gives
    /// neither a focal length nor a 35 mm equivalent, degrees.
    pub fov_deg: f64,
    pub labels: Labels,
}

impl Default for StillsOptions {
    fn default() -> Self {
        Self {
            extensions: vec!["jpg".to_string(), "jpeg".to_string()],
            embed: Embed::Preview,
            preview_px: 1600,
            preview_quality: 88,
            sensor_mm: None,
            fov_deg: 70.0,
            labels: Labels::default(),
        }
    }
}

/// What the import found, for the report.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StillsReport {
    pub total: usize,
    /// Views per camera key, in the cameras' order.
    pub cameras: Vec<(String, usize)>,
    pub warnings: Vec<String>,
    /// Bytes of pictures embedded.
    pub bytes: usize,
}

/// One still as read, before the cameras are keyed.
struct Still {
    name: String,
    width: u32,
    height: u32,
    exif: Option<Exif>,
    image: Option<Vec<u8>>,
}

/// Sensor widths of bodies the intake knows, millimetres.
fn sensor_width_mm(model: &str) -> Option<f64> {
    let m = model.to_ascii_uppercase();
    // Sony APS-C E-mount bodies (a6xxx, a5xxx, ZV-E10, FX30); the a6700's
    // sensor is 23.5 x 15.6 mm.
    if m.starts_with("ILCE-6") || m.starts_with("ILCE-5") || m == "ZV-E10" || m == "ILME-FX30" {
        return Some(23.5);
    }
    // Sony full-frame bodies.
    if m.starts_with("ILCE-7") || m.starts_with("ILCE-9") || m.starts_with("ILCE-1") {
        return Some(35.9);
    }
    None
}

/// The stills of `dir` (not recursive), sorted by name.
pub fn still_paths(dir: &Path, extensions: &[String]) -> Result<Vec<PathBuf>> {
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("Failed to read {}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| extensions.iter().any(|x| x.eq_ignore_ascii_case(e)))
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Imports every still of `dir` as an unposed view.
pub fn import_stills(dir: &Path, options: &StillsOptions) -> Result<(ViewSet, StillsReport)> {
    let paths = still_paths(dir, &options.extensions)?;
    if paths.is_empty() {
        bail!(
            "no stills ({}) in {}",
            options.extensions.join("/"),
            dir.display()
        );
    }
    let stills: Vec<Result<Still>> = paths.par_iter().map(|p| read_still(p, options)).collect();
    let mut read = Vec::with_capacity(stills.len());
    for still in stills {
        read.push(still?);
    }
    build_set(dir, read, options)
}

fn read_still(path: &Path, options: &StillsOptions) -> Result<Still> {
    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("{} has no usable file name", path.display()))?
        .to_string();
    let exif = read_exif(&bytes);
    let (width, height, image) = match options.embed {
        Embed::None => {
            let (w, h) = dimensions_of(&bytes).with_context(|| format!("{name}: not a picture"))?;
            (w, h, None)
        }
        Embed::Full => {
            let (w, h) = dimensions_of(&bytes).with_context(|| format!("{name}: not a picture"))?;
            (w, h, Some(bytes))
        }
        Embed::Preview => {
            let (w, h) = dimensions_of(&bytes).with_context(|| format!("{name}: not a picture"))?;
            let preview = preview_jpeg(bytes, options.preview_px, options.preview_quality)
                .with_context(|| format!("{name}: does not decode"))?;
            (w, h, Some(preview))
        }
    };
    Ok(Still {
        name,
        width,
        height,
        exif,
        image,
    })
}

/// The focal seed in pixels for a still, and where it came from.
fn focal_seed(
    exif: Option<&Exif>,
    width: u32,
    height: u32,
    options: &StillsOptions,
) -> (f64, &'static str) {
    if let Some(exif) = exif {
        let sensor = options
            .sensor_mm
            .or_else(|| exif.model.as_deref().and_then(sensor_width_mm));
        if let Some(f) = sensor.and_then(|s| exif.focal_px_on_sensor(s, width, height)) {
            // A lens focused at a couple of metres is a few percent longer
            // than its infinity focal length.
            return (f * 1.05, "focal length over the sensor width");
        }
        if let Some(f) = exif.focal_px(width, height) {
            return (f * 1.05, "35 mm equivalent focal length");
        }
    }
    (focal_px_from_fov(options.fov_deg, width), "field of view")
}

fn build_set(
    dir: &Path,
    stills: Vec<Still>,
    options: &StillsOptions,
) -> Result<(ViewSet, StillsReport)> {
    let mut report = StillsReport {
        total: stills.len(),
        ..StillsReport::default()
    };
    let mut set = ViewSet {
        provenance: Provenance {
            session: options.labels.session.clone().unwrap_or_else(|| {
                dir.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string()
            }),
            rig: options.labels.rig.clone().unwrap_or_default(),
            field: options.labels.field.clone().unwrap_or_default(),
            setup: options.labels.setup.clone().unwrap_or_default(),
            tools: vec![format!(
                "volumetric view-import --stills ({} pictures, {})",
                stills.len(),
                match options.embed {
                    Embed::Full => "embedded in full",
                    Embed::Preview => "previews embedded",
                    Embed::None => "nothing embedded",
                }
            )],
            captured: String::new(),
            origin: std::fs::canonicalize(dir)
                .unwrap_or_else(|_| dir.to_path_buf())
                .display()
                .to_string(),
        },
        ..ViewSet::default()
    };
    let mut keys: Vec<String> = Vec::new();
    let (mut no_exif, mut autofocus, mut stabilised, mut high_iso, mut no_focus) = (0, 0, 0, 0, 0);
    let mut orientations: Vec<u32> = Vec::new();
    let mut seed_sources: Vec<&'static str> = Vec::new();
    for still in stills {
        let id = Path::new(&still.name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&still.name)
            .to_string();
        let shot = still.exif.as_ref().map(Exif::to_shot);
        let key = match &shot {
            Some(shot) => shot.camera_key(&id),
            None => "unknown".to_string(),
        };
        match &shot {
            None => no_exif += 1,
            Some(shot) => {
                if !shot.focus_is_held() {
                    autofocus += 1;
                }
                if shot.stabilisation == Some(true) {
                    stabilised += 1;
                }
                if shot.iso > 3200 {
                    high_iso += 1;
                }
                if shot.focus_position.is_none() {
                    no_focus += 1;
                }
                if shot.orientation > 0 && !orientations.contains(&shot.orientation) {
                    orientations.push(shot.orientation);
                }
            }
        }
        let camera = match keys.iter().position(|k| *k == key) {
            Some(i) => {
                let camera = &set.cameras[i];
                if (camera.width, camera.height) != (still.width, still.height) {
                    bail!(
                        "{}: {}x{} but other frames of camera '{key}' are {}x{}",
                        still.name,
                        still.width,
                        still.height,
                        camera.width,
                        camera.height
                    );
                }
                report.cameras[i].1 += 1;
                i as u32
            }
            None => {
                let (f, source) =
                    focal_seed(still.exif.as_ref(), still.width, still.height, options);
                seed_sources.push(source);
                let mut camera = CameraModel::pinhole(
                    still.width,
                    still.height,
                    f,
                    f,
                    f64::from(still.width) * 0.5,
                    f64::from(still.height) * 0.5,
                );
                camera.label = key.clone();
                set.cameras.push(camera);
                keys.push(key.clone());
                report.cameras.push((key, 1));
                (set.cameras.len() - 1) as u32
            }
        };
        let mut view = View::unposed(id, camera);
        report.bytes += still.image.as_ref().map_or(0, Vec::len);
        view.image = still.image;
        view.source = Some(still.name);
        view.shot = shot;
        view.tags.push("still".to_string());
        set.views.push(view);
    }
    if no_exif > 0 {
        report.warnings.push(format!(
            "{no_exif} frames carry no EXIF: their camera is seeded from the field of view and keyed 'unknown'"
        ));
    }
    if no_focus > 0 && no_exif < report.total {
        report.warnings.push(format!(
            "{no_focus} frames give no focus position (not a Sony maker note?): frames key by body and lens only"
        ));
    }
    if autofocus > 0 {
        report.warnings.push(format!(
            "{autofocus} autofocus frames: each is its own camera and cannot share a solved model"
        ));
    }
    if stabilised > 0 {
        report.warnings.push(format!(
            "{stabilised} frames with stabilisation on: the principal point moves per shot"
        ));
    }
    if high_iso > 0 {
        report
            .warnings
            .push(format!("{high_iso} frames above ISO 3200"));
    }
    if orientations.len() > 1 {
        report.warnings.push(format!(
            "mixed orientations {orientations:?}: the pictures are kept as the sensor recorded them"
        ));
    }
    for (key, n) in &report.cameras {
        if *n < 3 && report.total >= 3 {
            report.warnings.push(format!(
                "camera '{key}' has {n} frame(s): too few to solve its own intrinsics"
            ));
        }
    }
    seed_sources.sort_unstable();
    seed_sources.dedup();
    report
        .warnings
        .push(format!("focal seeds from: {}", seed_sources.join(", ")));
    set.validate().map_err(anyhow::Error::msg)?;
    Ok((set, report))
}

/// A picture reduced to `preview_px` on its longer side as a JPEG of
/// `quality`; a picture already that small is returned as it is.
pub fn preview_jpeg(bytes: Vec<u8>, preview_px: u32, quality: u8) -> Result<Vec<u8>> {
    let (width, height) = dimensions_of(&bytes)?;
    let long = width.max(height);
    let scale = f64::from(preview_px.max(1)) / f64::from(long);
    if scale >= 1.0 {
        return Ok(bytes);
    }
    let photo = decode_rgb(&bytes)?;
    let w = ((f64::from(width) * scale).round() as u32).max(1);
    let h = ((f64::from(height) * scale).round() as u32).max(1);
    photo.resized(w, h)?.to_jpeg(quality)
}

/// Re-embeds every view's picture as `embed` says: the original through
/// [`full_picture`], a preview of it, or nothing but the reference. Returns
/// the bytes embedded.
pub fn embed_pictures(
    set: &mut ViewSet,
    embed: Embed,
    preview_px: u32,
    quality: u8,
) -> Result<usize> {
    let pictures: Vec<Result<Option<Vec<u8>>>> = set
        .views
        .par_iter()
        .map(|view| match embed {
            Embed::None => Ok(None),
            Embed::Full => full_picture(set, view).map(Some),
            Embed::Preview => {
                if let Some(bytes) = &view.image
                    && dimensions_of(bytes).is_ok_and(|(w, h)| w.max(h) <= preview_px)
                {
                    return Ok(Some(bytes.clone()));
                }
                let bytes = full_picture(set, view)?;
                preview_jpeg(bytes, preview_px, quality)
                    .with_context(|| format!("view '{}': the picture does not decode", view.id))
                    .map(Some)
            }
        })
        .collect();
    let mut embedded = 0;
    for (view, picture) in set.views.iter_mut().zip(pictures) {
        let picture = picture?;
        embedded += picture.as_ref().map_or(0, Vec::len);
        view.image = picture;
    }
    Ok(embedded)
}

/// The picture of a view at the camera's full resolution: the embedded
/// one when it is full size, else the original read through `source`
/// from the provenance's origin.
pub fn full_picture(set: &ViewSet, view: &View) -> Result<Vec<u8>> {
    let camera = set.camera_of(view);
    if let Some(bytes) = &view.image
        && dimensions_of(bytes).is_ok_and(|(w, h)| (w, h) == (camera.width, camera.height))
    {
        return Ok(bytes.clone());
    }
    let Some(source) = &view.source else {
        bail!(
            "view '{}' carries no full-size picture and no source path",
            view.id
        );
    };
    let path = Path::new(&set.provenance.origin).join(source);
    std::fs::read(&path).with_context(|| {
        format!(
            "view '{}': the original {} is not readable; the set's origin is '{}'",
            view.id,
            path.display(),
            set.provenance.origin
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Rgb;

    fn write_still(dir: &Path, name: &str, w: u32, h: u32) {
        let mut rgb = Rgb::new(w, h);
        for y in 0..h {
            for x in 0..w {
                rgb.set(x, y, [(x % 256) as u8, (y % 256) as u8, 128]);
            }
        }
        std::fs::write(dir.join(name), rgb.to_jpeg(90).unwrap()).unwrap();
    }

    #[test]
    fn stills_import_unposed_with_previews_and_sources() {
        let dir = std::env::temp_dir().join(format!("volumetric_stills_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_still(&dir, "b.JPG", 800, 600);
        write_still(&dir, "a.jpg", 800, 600);
        std::fs::write(dir.join("notes.txt"), "x").unwrap();
        let options = StillsOptions {
            preview_px: 200,
            ..StillsOptions::default()
        };
        let (set, report) = import_stills(&dir, &options).unwrap();
        assert_eq!(report.total, 2);
        assert_eq!(set.views.len(), 2);
        assert_eq!(set.views[0].id, "a");
        assert_eq!(set.views[1].id, "b");
        assert!(set.views.iter().all(|v| v.pose().is_none()));
        assert_eq!(set.views[0].source.as_deref(), Some("a.jpg"));
        assert_eq!(set.cameras.len(), 1);
        assert_eq!(set.cameras[0].label, "unknown");
        assert_eq!((set.cameras[0].width, set.cameras[0].height), (800, 600));
        // No EXIF: the seed is the 70 degree field of view.
        assert!((set.cameras[0].fx - focal_px_from_fov(70.0, 800)).abs() < 1e-9);
        let preview = dimensions_of(set.views[0].image.as_ref().unwrap()).unwrap();
        assert_eq!(preview, (200, 150));
        assert!(report.warnings.iter().any(|w| w.contains("no EXIF")));
        // The full picture comes back through the source.
        let full = full_picture(&set, &set.views[1]).unwrap();
        assert_eq!(dimensions_of(&full).unwrap(), (800, 600));
        let mut moved = set.clone();
        moved.provenance.origin = "/nowhere".to_string();
        assert!(full_picture(&moved, &moved.views[1]).is_err());
        // Embedded in full, the picture itself is returned.
        let (full_set, report) = import_stills(
            &dir,
            &StillsOptions {
                embed: Embed::Full,
                ..options.clone()
            },
        )
        .unwrap();
        assert_eq!(
            dimensions_of(full_set.views[0].image.as_ref().unwrap()).unwrap(),
            (800, 600)
        );
        assert!(report.bytes > 0);
        // Nothing embedded: only the reference.
        let (bare, report) = import_stills(
            &dir,
            &StillsOptions {
                embed: Embed::None,
                ..options
            },
        )
        .unwrap();
        assert!(bare.views[0].image.is_none() && report.bytes == 0);
        assert!(
            bare.provenance
                .origin
                .ends_with(dir.file_name().unwrap().to_str().unwrap())
        );
        // Re-embedding reads the originals through the sources: previews
        // of the asked size, the full files, or nothing again.
        let mut again = bare.clone();
        let bytes = embed_pictures(&mut again, Embed::Preview, 400, 80).unwrap();
        assert!(bytes > 0);
        assert_eq!(
            dimensions_of(again.views[1].image.as_ref().unwrap()).unwrap(),
            (400, 300)
        );
        // A picture already small enough is kept as it is.
        let kept = again.views[1].image.clone();
        embed_pictures(&mut again, Embed::Preview, 400, 80).unwrap();
        assert_eq!(again.views[1].image, kept);
        embed_pictures(&mut again, Embed::Full, 400, 80).unwrap();
        assert_eq!(
            dimensions_of(again.views[0].image.as_ref().unwrap()).unwrap(),
            (800, 600)
        );
        assert_eq!(embed_pictures(&mut again, Embed::None, 400, 80).unwrap(), 0);
        assert!(again.views.iter().all(|v| v.image.is_none()));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(import_stills(&dir, &StillsOptions::default()).is_err());
    }

    #[test]
    fn known_bodies_seed_from_the_sensor() {
        assert_eq!(sensor_width_mm("ILCE-6700"), Some(23.5));
        assert_eq!(sensor_width_mm("ILCE-7M4"), Some(35.9));
        assert_eq!(sensor_width_mm("Pixel 7"), None);
        let exif = Exif {
            model: Some("ILCE-6700".to_string()),
            focal_mm: Some(50.0),
            focal_35mm: Some(75.0),
            ..Exif::default()
        };
        let options = StillsOptions::default();
        let (f, source) = focal_seed(Some(&exif), 6192, 4128, &options);
        assert!((f - 50.0 / 23.5 * 6192.0 * 1.05).abs() < 1e-6);
        assert_eq!(source, "focal length over the sensor width");
        let unknown = Exif {
            model: Some("Other".to_string()),
            ..exif
        };
        let (f, source) = focal_seed(Some(&unknown), 6192, 4128, &options);
        assert!((f - 75.0 / 36.0 * 6192.0 * 1.05).abs() < 1e-6);
        assert_eq!(source, "35 mm equivalent focal length");
        let given = StillsOptions {
            sensor_mm: Some(36.0),
            ..options
        };
        let (f, _) = focal_seed(Some(&unknown), 6192, 4128, &given);
        assert!((f - 50.0 / 36.0 * 6192.0 * 1.05).abs() < 1e-6);
    }
}
