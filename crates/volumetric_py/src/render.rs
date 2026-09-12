//! The headless frame from Python: `volumetric_render` one to one.

use glam::Vec3;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use volumetric::LoadedAsset;
use volumetric_render::{
    CameraSpec, ColorRange, Overlay, Pinhole, PlanOptions, Projection, RenderOptions,
    background_from_hex, parse_views, pose_matrix, select_assets,
};

use crate::project::{Asset, Project};
use crate::{array3, invalid, pose_from_py, runtime, vec3_from_py};

/// What a render produced: `frames` (h,w,4) uint8 arrays, one per preset
/// drawn (or one), their `names` (the preset when several were drawn,
/// else ""), `image` (the first), and `report` (per-asset stats, the
/// world up and where it came from, the GPU, notes).
#[pyclass(module = "volumetric")]
pub struct Rendered {
    #[pyo3(get)]
    frames: Py<PyList>,
    #[pyo3(get)]
    names: Vec<String>,
    #[pyo3(get)]
    report: Py<PyDict>,
}

#[pymethods]
impl Rendered {
    /// The first (or only) frame, (h,w,4) uint8.
    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.frames.bind(py).get_item(0)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Rendered({} frames: {})",
            self.frames.bind(py).len(),
            self.names.join(", ")
        )
    }
}

fn vec3_opt(value: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<Option<Vec3>> {
    match value {
        Some(v) if !v.is_none() => {
            let a = vec3_from_py(v, what)?;
            Ok(Some(Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32)))
        }
        _ => Ok(None),
    }
}

/// Draw a project's exports (the project is run) or a list of assets from
/// a run. Cameras, one of: `views` (preset names, comma-separated or a
/// list; `all` for every one), `camera=(eye, target=None, up=None)`,
/// `pinhole={"fx","fy","cx","cy","camera_to_world"}` (intrinsics in
/// pixels of the frame, pose (3,4)), or `through="view"` /
/// `"asset:view"` for a posed photograph of a view set among the assets
/// (its size unless `width`/`height`), with `overlay` (`edge`, `blend`,
/// `side`, `checker`) compositing over the photograph. `assets` names
/// what to draw (default: every renderable export; imports only when
/// named). The rest are `RenderOptions` and `PlanOptions`: `background`
/// hex sRGB, `up`, `projection` (`perspective`/`ortho`), `fov`,
/// `ortho_scale`, `near`, `far`, `grid`, `ssao`, `resolution`, `sharp`,
/// `simplify`, `color_channel`, `color_field`, `color_range`, `wireframe`.
#[pyfunction]
#[pyo3(signature = (source, assets=None, views=None, camera=None, pinhole=None, through=None,
    overlay=None, overlay_alpha=0.5, overlay_tile=64, width=None, height=None,
    projection="perspective", fov=45.0, ortho_scale=0.0, near=None, far=None, up=None,
    background="2d2d2d", grid=1.0, ssao=true, resolution=128, sharp=true, simplify=true,
    color_channel=None, color_field=None, color_range=None, wireframe=false))]
#[allow(clippy::too_many_arguments)]
fn render<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    assets: Option<Vec<String>>,
    views: Option<&Bound<'py, PyAny>>,
    camera: Option<&Bound<'py, PyAny>>,
    pinhole: Option<&Bound<'py, PyDict>>,
    through: Option<&str>,
    overlay: Option<&str>,
    overlay_alpha: f32,
    overlay_tile: u32,
    width: Option<u32>,
    height: Option<u32>,
    projection: &str,
    fov: f32,
    ortho_scale: f32,
    near: Option<f32>,
    far: Option<f32>,
    up: Option<&Bound<'py, PyAny>>,
    background: &str,
    grid: f32,
    ssao: bool,
    resolution: usize,
    sharp: bool,
    simplify: bool,
    color_channel: Option<String>,
    color_field: Option<String>,
    color_range: Option<(f64, f64)>,
    wireframe: bool,
) -> PyResult<Rendered> {
    // What to draw.
    let (exports, imports): (Vec<LoadedAsset>, Vec<LoadedAsset>) =
        if let Ok(project) = source.cast::<Project>() {
            let project = project.borrow();
            let imports = volumetric::asset_query::imports_as_assets(&project.inner);
            let mut copy = project.inner.clone();
            let exports = py.detach(move || crate::project::run_exports(&mut copy, None))?;
            (exports, imports)
        } else if let Ok(list) = source.cast::<PyList>() {
            let mut assets = Vec::new();
            for item in list.iter() {
                let asset = item
                    .cast::<Asset>()
                    .map_err(|_| invalid("source is a Project or a list of Asset"))?;
                assets.push(asset.borrow().inner.clone());
            }
            (assets, Vec::new())
        } else {
            return Err(invalid("source is a Project or a list of Asset"));
        };
    let selected =
        select_assets(exports, &imports, &assets.unwrap_or_default()).map_err(invalid)?;

    // The camera.
    let given = [camera.is_some(), pinhole.is_some(), through.is_some()]
        .iter()
        .filter(|g| **g)
        .count();
    if given > 1 {
        return Err(invalid("give one of camera=, pinhole= or through="));
    }
    let spec = if let Some(through) = through {
        let (asset, view) = match through.split_once(':') {
            Some((asset, view)) => (Some(asset.to_string()), view.to_string()),
            None => (None, through.to_string()),
        };
        if view.is_empty() {
            return Err(invalid("through= needs a view id"));
        }
        CameraSpec::Through { asset, view }
    } else if let Some(camera) = camera {
        let parts: Vec<Bound<'py, PyAny>> = camera
            .extract()
            .map_err(|_| invalid("camera is (eye, target=None, up=None)"))?;
        if parts.is_empty() || parts.len() > 3 {
            return Err(invalid("camera is (eye, target=None, up=None)"));
        }
        let eye = vec3_opt(Some(&parts[0]), "the camera eye")?
            .ok_or_else(|| invalid("the camera eye is three numbers"))?;
        CameraSpec::LookAt {
            eye,
            target: vec3_opt(parts.get(1), "the camera target")?,
            up: vec3_opt(parts.get(2), "the camera up")?,
        }
    } else if let Some(pinhole) = pinhole {
        let number = |key: &str| -> PyResult<f32> {
            pinhole
                .get_item(key)?
                .ok_or_else(|| invalid(format!("pinhole needs `{key}`")))?
                .extract::<f64>()
                .map(|v| v as f32)
                .map_err(|_| invalid(format!("pinhole `{key}` is a number")))
        };
        let pose = pinhole
            .get_item("camera_to_world")?
            .ok_or_else(|| invalid("pinhole needs `camera_to_world` (3,4)"))?;
        let rows = pose_from_py(&pose)?;
        CameraSpec::Pinhole {
            pinhole: Pinhole {
                fx: number("fx")?,
                fy: number("fy")?,
                cx: number("cx")?,
                cy: number("cy")?,
                width: width.unwrap_or(1024),
                height: height.unwrap_or(1024),
            },
            camera_to_world: pose_matrix(&rows),
        }
    } else {
        let list = match views {
            None => "iso".to_string(),
            Some(views) => {
                if let Ok(text) = views.extract::<String>() {
                    text
                } else {
                    let names: Vec<String> = views.extract().map_err(|_| {
                        invalid("views is a preset name, a comma-separated list or a list")
                    })?;
                    names.join(",")
                }
            }
        };
        CameraSpec::Presets(parse_views(&list).map_err(invalid)?)
    };

    if overlay.is_some() && through.is_none() {
        return Err(invalid(
            "overlay= composites over a view's photograph; give through=",
        ));
    }
    let overlay = match overlay {
        Some(name) => Some(
            Overlay::parse(name, overlay_alpha, overlay_tile).ok_or_else(|| {
                invalid(format!(
                    "unknown overlay `{name}`; edge, blend, side or checker"
                ))
            })?,
        ),
        None => None,
    };
    let options = RenderOptions {
        width,
        height,
        projection: match projection {
            "perspective" => Projection::Perspective,
            "ortho" | "orthographic" => Projection::Orthographic,
            other => {
                return Err(invalid(format!(
                    "unknown projection `{other}`; perspective or ortho"
                )));
            }
        },
        fov_deg: fov,
        ortho_scale,
        near,
        far,
        up: vec3_opt(up, "up")?,
        background: background_from_hex(background).map_err(invalid)?,
        grid,
        ssao,
        plan: PlanOptions {
            resolution,
            sharp,
            simplify,
            color_channel,
            color_field,
            color_range: match color_range {
                Some((lo, hi)) => Some(
                    ColorRange::new(lo, hi)
                        .ok_or_else(|| invalid("color_range needs lo below hi"))?,
                ),
                None => None,
            },
            wireframe,
        },
        overlay,
    };

    let rendered = py
        .detach(move || volumetric_render::render(&selected, &imports, spec, &options))
        .map_err(runtime)?;
    let frames = PyList::empty(py);
    let mut names = Vec::new();
    for frame in rendered.frames {
        frames.append(array3(py, frame.rgba, frame.width as usize, 4)?)?;
        names.push(frame.suffix.unwrap_or("").to_string());
    }
    let report = PyDict::new(py);
    let entities: Vec<Bound<'py, PyDict>> = rendered
        .report
        .entities
        .iter()
        .map(|e| {
            let d = PyDict::new(py);
            d.set_item("id", &e.id)?;
            d.set_item("triangles", e.triangles)?;
            d.set_item("points", e.points)?;
            d.set_item("bounds", (e.bounds.min, e.bounds.max))?;
            d.set_item("mesh_ms", e.mesh_ms)?;
            d.set_item("detail", &e.detail)?;
            Ok(d)
        })
        .collect::<PyResult<_>>()?;
    report.set_item("entities", entities)?;
    report.set_item("up", rendered.report.up.to_array())?;
    report.set_item("up_source", &rendered.report.up_source)?;
    report.set_item("gpu", &rendered.report.gpu)?;
    report.set_item("notes", &rendered.report.notes)?;
    Ok(Rendered {
        frames: frames.unbind(),
        names,
        report: report.unbind(),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Rendered>()?;
    m.add_function(wrap_pyfunction!(render, m)?)?;
    Ok(())
}
