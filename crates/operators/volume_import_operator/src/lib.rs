//! Volume Import Operator.
//!
//! Reads a sampled scalar volume (NRRD) and emits a model. The field's
//! isosurface at `threshold` becomes occupancy, and the field itself,
//! re-signed so that negative is inside and clamped to `band`, becomes the
//! `signed_distance` channel: the same two channels `sdf_operator` bakes,
//! so meshing, offsetting and every other consumer see one kind of model.
//! A fused scan (TSDF), a CT threshold, or any other regular grid comes in
//! the same door. See README.md (the operator's docs) for the conventions.
//!
//! Inputs:
//! - Input 0: Blob — NRRD file bytes
//! - Input 1: CBOR configuration, see [`VolumeImportConfig`].
//!
//! Output 0: ModelWASM (occupancy + `signed_distance` channel).

use ndfield_model_core::emit::{FieldSign, emit_field_model};
use nrrd_core::{Nrrd, read_nrrd};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::host::{post_output, post_warning, read_input, report_error};
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Which side of `threshold` is solid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Inside {
    /// Samples below the threshold are inside: a signed distance field.
    Below,
    /// Samples above the threshold are inside: a density or occupancy
    /// probability.
    Above,
}

/// What an unobserved sample (NaN in the file) stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Unobserved {
    /// Solid where observed samples enclose it, empty where it reaches the
    /// volume's edge: the interior of a scanned object fills in, the space
    /// the scan never covered stays open.
    Enclosed,
    Solid,
    Empty,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct VolumeImportConfig {
    /// Metres per file unit, applied to positions and distances alike.
    pub scale: f64,
    /// Centre the solid's bounding box on the origin (before `scale`).
    pub center: bool,
    /// The isovalue: the surface is where the field equals it.
    pub threshold: f64,
    pub inside: Inside,
    /// Distance from the surface (file units) beyond which the field is
    /// clamped; 0 takes the largest magnitude present. Also the outside
    /// value everywhere beyond the volume.
    pub band: f64,
    pub unobserved: Unobserved,
    /// Keep only the solid plus one band of margin around it.
    pub crop: bool,
    /// Keep every `stride`-th sample along each axis.
    pub stride: u32,
}

impl Default for VolumeImportConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            center: false,
            threshold: 0.0,
            inside: Inside::Below,
            band: 0.0,
            unobserved: Unobserved::Enclosed,
            crop: true,
            stride: 1,
        }
    }
}

/// A volume prepared for emission: the ndfield payload plus what the model
/// advertises and what the import had to decide.
#[derive(Clone, Debug)]
pub struct Volume {
    pub payload: Vec<u8>,
    pub dimensions: usize,
    /// Model bounds in metres, `[min_0, max_0, ...]`.
    pub bounds: Vec<f64>,
    /// Lattice points per axis in the payload.
    pub counts: Vec<usize>,
    /// The truncation distance in metres.
    pub band: f64,
    /// Unobserved samples that became solid / empty.
    pub unobserved_solid: usize,
    pub unobserved_empty: usize,
    /// Axes along which the solid reaches the volume's first or last sample,
    /// where it is cut flat.
    pub edge_axes: Vec<usize>,
}

/// Row-major-from-axis-0 strides: `strides[a]` is the index step of axis `a`.
fn strides(sizes: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(sizes.len());
    let mut stride = 1;
    for &size in sizes {
        strides.push(stride);
        stride *= size;
    }
    strides
}

/// Visits every index of a grid with its coordinates, axis 0 fastest.
fn for_each_coord(sizes: &[usize], mut visit: impl FnMut(usize, &[usize])) {
    let d = sizes.len();
    let count: usize = sizes.iter().product();
    let mut coords = vec![0usize; d];
    for index in 0..count {
        visit(index, &coords);
        for axis in 0..d {
            coords[axis] += 1;
            if coords[axis] < sizes[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

/// Which NaN samples connect, through NaN samples, to the volume's edge.
fn nan_reaching_the_edge(sizes: &[usize], nan: &[bool]) -> Vec<bool> {
    let strides = strides(sizes);
    let mut reachable = vec![false; nan.len()];
    let mut queue: Vec<usize> = Vec::new();
    for_each_coord(sizes, |index, coords| {
        if nan[index]
            && coords
                .iter()
                .zip(sizes)
                .any(|(&c, &size)| c == 0 || c + 1 == size)
        {
            reachable[index] = true;
            queue.push(index);
        }
    });
    while let Some(index) = queue.pop() {
        for axis in 0..sizes.len() {
            let coord = index / strides[axis] % sizes[axis];
            if coord > 0 {
                let neighbour = index - strides[axis];
                if nan[neighbour] && !reachable[neighbour] {
                    reachable[neighbour] = true;
                    queue.push(neighbour);
                }
            }
            if coord + 1 < sizes[axis] {
                let neighbour = index + strides[axis];
                if nan[neighbour] && !reachable[neighbour] {
                    reachable[neighbour] = true;
                    queue.push(neighbour);
                }
            }
        }
    }
    reachable
}

/// Turns the file's samples into a signed distance (negative inside),
/// clamped to the band, with every unobserved sample resolved. Returns the
/// band and the unobserved (solid, empty) counts.
fn sign_clamp_and_resolve(
    volume: &mut Nrrd,
    config: &VolumeImportConfig,
) -> Result<(f64, usize, usize), String> {
    let flip = match config.inside {
        Inside::Below => 1.0f32,
        Inside::Above => -1.0f32,
    };
    let threshold = config.threshold as f32;
    let mut largest = 0.0f32;
    let mut finite = 0usize;
    for value in &mut volume.values {
        *value = (*value - threshold) * flip;
        if value.is_finite() {
            finite += 1;
            largest = largest.max(value.abs());
        }
    }
    if finite == 0 {
        return Err("the volume has no finite samples".to_string());
    }
    let band = if config.band > 0.0 {
        config.band as f32
    } else if largest > 0.0 {
        largest
    } else {
        return Err(
            "every sample equals the threshold, so the field has no surface; set `band` \
             or check `threshold`"
                .to_string(),
        );
    };
    let nan: Vec<bool> = volume.values.iter().map(|v| v.is_nan()).collect();
    let unobserved = nan.iter().filter(|&&n| n).count();
    let mut solid = 0usize;
    if unobserved > 0 {
        match config.unobserved {
            Unobserved::Enclosed => {
                let reachable = nan_reaching_the_edge(&volume.sizes, &nan);
                for (index, value) in volume.values.iter_mut().enumerate() {
                    if nan[index] {
                        if reachable[index] {
                            *value = band;
                        } else {
                            *value = -band;
                            solid += 1;
                        }
                    }
                }
            }
            Unobserved::Solid => {
                solid = unobserved;
                for value in &mut volume.values {
                    if value.is_nan() {
                        *value = -band;
                    }
                }
            }
            Unobserved::Empty => {
                for value in &mut volume.values {
                    if value.is_nan() {
                        *value = band;
                    }
                }
            }
        }
    }
    for value in &mut volume.values {
        *value = value.clamp(-band, band);
    }
    Ok((band as f64, solid, unobserved - solid))
}

/// Reads the file and prepares it for emission. See the module docs.
pub fn import(bytes: &[u8], config: &VolumeImportConfig) -> Result<Volume, String> {
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(format!(
            "scale must be finite and positive, got {}",
            config.scale
        ));
    }
    if config.stride == 0 {
        return Err("stride must be at least 1".to_string());
    }
    if !(config.band.is_finite() && config.band >= 0.0) {
        return Err(format!(
            "band must be finite and non-negative, got {}",
            config.band
        ));
    }
    let mut volume = read_nrrd(bytes)?;
    let d = volume.dimensions();
    if d > ndfield_model_core::MAX_DIMS {
        return Err(format!(
            "the volume has {d} axes; models support at most {}",
            ndfield_model_core::MAX_DIMS
        ));
    }
    let (band, unobserved_solid, unobserved_empty) = sign_clamp_and_resolve(&mut volume, config)?;

    // The solid's extent, in samples, on the full grid.
    let mut solid_lo = vec![usize::MAX; d];
    let mut solid_hi = vec![0usize; d];
    let mut any_inside = false;
    for_each_coord(&volume.sizes, |index, coords| {
        if volume.values[index] < 0.0 {
            any_inside = true;
            for axis in 0..d {
                solid_lo[axis] = solid_lo[axis].min(coords[axis]);
                solid_hi[axis] = solid_hi[axis].max(coords[axis]);
            }
        }
    });
    if !any_inside {
        return Err(format!(
            "no sample is inside (with `inside: {}`, inside means {} {}); check `threshold`",
            match config.inside {
                Inside::Below => "below",
                Inside::Above => "above",
            },
            match config.inside {
                Inside::Below => "below",
                Inside::Above => "above",
            },
            config.threshold
        ));
    }
    let edge_axes: Vec<usize> = (0..d)
        .filter(|&axis| solid_lo[axis] == 0 || solid_hi[axis] + 1 == volume.sizes[axis])
        .collect();

    // The kept window: the solid plus a band of margin (so the field is
    // its true value out to the clamp, and meets the outside value
    // continuously), or the whole grid; then every `stride`-th sample.
    let step = config.stride as usize;
    let mut lo = vec![0usize; d];
    let mut counts = vec![0usize; d];
    for axis in 0..d {
        let (window_lo, window_hi) = if config.crop {
            let pad = (band / volume.spacing[axis]).ceil() as usize + step;
            (
                solid_lo[axis].saturating_sub(pad),
                (solid_hi[axis] + pad).min(volume.sizes[axis] - 1),
            )
        } else {
            (0, volume.sizes[axis] - 1)
        };
        lo[axis] = window_lo;
        counts[axis] = (window_hi - window_lo) / step + 1;
        if counts[axis] < 2 {
            return Err(format!(
                "axis {axis} keeps a single sample (of {}); a field needs at least two per \
                 axis — lower `stride`",
                volume.sizes[axis]
            ));
        }
    }

    let shift: Vec<f64> = (0..d)
        .map(|axis| {
            if config.center {
                -(volume.position(axis, solid_lo[axis]) + volume.position(axis, solid_hi[axis]))
                    / 2.0
            } else {
                0.0
            }
        })
        .collect();
    let mut bounds = Vec::with_capacity(2 * d);
    for axis in 0..d {
        let first = volume.position(axis, lo[axis]);
        let last = volume.position(axis, lo[axis] + (counts[axis] - 1) * step);
        bounds.push((first + shift[axis]) * config.scale);
        bounds.push((last + shift[axis]) * config.scale);
    }

    let source_strides = strides(&volume.sizes);
    let scale = config.scale as f32;
    let mut values = Vec::with_capacity(counts.iter().product());
    for_each_coord(&counts, |_, coords| {
        let source: usize = (0..d)
            .map(|axis| (lo[axis] + coords[axis] * step) * source_strides[axis])
            .sum();
        values.push(volume.values[source] * scale);
    });
    let band = band * config.scale;
    let payload = ndfield_model_core::build_payload(&counts, &bounds, &values, band as f32)?;
    Ok(Volume {
        payload,
        dimensions: d,
        bounds,
        counts,
        band,
        unobserved_solid,
        unobserved_empty,
        edge_axes,
    })
}

/// Patches the prepared volume into the field template as a standalone
/// occupancy + signed-distance model.
pub fn emit(volume: &Volume) -> Result<Vec<u8>, String> {
    emit_field_model(
        &volume.payload,
        volume.dimensions,
        &volume.bounds,
        FieldSign::SignedDistance,
    )
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let bytes = read_input(0);
    let config = {
        let cfg = read_input(1);
        if cfg.is_empty() {
            VolumeImportConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&cfg)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    let volume = match import(&bytes, &config) {
        Ok(volume) => volume,
        Err(e) => {
            report_error(&format!("volume import failed: {e}"));
            return;
        }
    };
    if volume.unobserved_solid + volume.unobserved_empty > 0 {
        post_warning(&format!(
            "{} unobserved sample(s): {} taken as solid, {} as empty",
            volume.unobserved_solid + volume.unobserved_empty,
            volume.unobserved_solid,
            volume.unobserved_empty
        ));
    }
    if !volume.edge_axes.is_empty() {
        post_warning(&format!(
            "the solid reaches the volume's edge on axis {} and is cut flat there",
            volume
                .edge_axes
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    match emit(&volume) {
        Ok(model) => post_output(0, &model),
        Err(e) => report_error(&format!("volume import failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ scale: float .default 1.0, center: bool .default false, threshold: float .default 0.0, inside: "below" / "above" .default "below", band: float .ge 0.0 .default 0.0, unobserved: "enclosed" / "solid" / "empty" .default "enclosed", crop: bool .default true, stride: int .ge 1 .default 1 }"#
            .to_string();
        OperatorMetadata {
            name: "volume_import_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Volume Import".to_string(),
            description: "Read a sampled volume (NRRD) as a solid: its isosurface becomes occupancy and the field its signed-distance channel.".to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<rect x="3" y="3" width="18" height="18" rx="2"/>"##,
                r##"<path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["NRRD file".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
            output_names: vec![],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndfield_model_core::PayloadView;
    use nrrd_core::{Encoding, write_nrrd};

    /// A 33^3 grid at 1 mm from -16 holding |p - c| - 5: a 5 mm sphere.
    fn sphere(centre: [f64; 3], origin: f64) -> Vec<u8> {
        let mut values = Vec::new();
        for z in 0..33 {
            for y in 0..33 {
                for x in 0..33 {
                    let p = [x as f64 + origin, y as f64 + origin, z as f64 + origin];
                    let r = (0..3)
                        .map(|a| (p[a] - centre[a]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    values.push((r - 5.0) as f32);
                }
            }
        }
        write_nrrd(
            &Nrrd {
                sizes: vec![33; 3],
                spacing: vec![1.0; 3],
                origin: vec![origin; 3],
                values,
            },
            Encoding::Raw,
        )
        .unwrap()
    }

    fn millimetres(band: f64) -> VolumeImportConfig {
        VolumeImportConfig {
            scale: 1e-3,
            band,
            ..VolumeImportConfig::default()
        }
    }

    fn close(a: f32, b: f64) -> bool {
        (a as f64 - b).abs() < 1e-7
    }

    #[test]
    fn sphere_sdf_imports_as_a_solid_with_its_distance_channel() {
        let volume = import(&sphere([0.0; 3], -16.0), &millimetres(3.0)).unwrap();
        assert_eq!(volume.dimensions, 3);
        // Inside samples span indices 12..=20; the band pads by 3 + 1.
        assert_eq!(volume.counts, vec![17; 3]);
        assert_eq!(volume.bounds, vec![-8e-3, 8e-3, -8e-3, 8e-3, -8e-3, 8e-3]);
        assert!(close(volume.band as f32, 3e-3));
        assert_eq!(volume.unobserved_solid + volume.unobserved_empty, 0);
        assert!(volume.edge_axes.is_empty());
        let field = PayloadView::new(&volume.payload).unwrap();
        assert!(
            close(field.sample(&[0.0, 0.0, 0.0]), -3e-3),
            "clamped centre"
        );
        assert!(close(field.sample(&[4.5e-3, 0.0, 0.0]), -0.5e-3));
        assert!(close(field.sample(&[7e-3, 0.0, 0.0]), 2e-3));
        assert!(
            close(field.sample(&[9e-3, 0.0, 0.0]), 3e-3),
            "outside value"
        );
        assert!(emit(&volume).unwrap().starts_with(b"\0asm"));
    }

    #[test]
    fn center_moves_the_solid_onto_the_origin() {
        let config = VolumeImportConfig {
            center: true,
            ..millimetres(3.0)
        };
        let volume = import(&sphere([16.0; 3], 0.0), &config).unwrap();
        assert_eq!(volume.bounds, vec![-8e-3, 8e-3, -8e-3, 8e-3, -8e-3, 8e-3]);
        let field = PayloadView::new(&volume.payload).unwrap();
        assert!(close(field.sample(&[0.0, 0.0, 0.0]), -3e-3));
    }

    #[test]
    fn stride_thins_the_grid() {
        let config = VolumeImportConfig {
            stride: 2,
            crop: false,
            ..millimetres(3.0)
        };
        let volume = import(&sphere([0.0; 3], -16.0), &config).unwrap();
        assert_eq!(volume.counts, vec![17; 3]);
        assert_eq!(
            volume.bounds,
            vec![-16e-3, 16e-3, -16e-3, 16e-3, -16e-3, 16e-3]
        );
        let field = PayloadView::new(&volume.payload).unwrap();
        assert!(close(field.sample(&[0.0, 0.0, 0.0]), -3e-3));
        assert!(close(field.sample(&[6e-3, 0.0, 0.0]), 1e-3));
    }

    /// An 11 x 11 grid at 1 unit from -5: r - 3.5 in the ring 2 <= r,
    /// unobserved inside r < 2 and in the far corner x, y >= 4.
    fn ring_with_holes() -> Vec<u8> {
        let mut values = Vec::new();
        for y in -5..=5 {
            for x in -5..=5 {
                let r = ((x * x + y * y) as f64).sqrt();
                values.push(if r < 2.0 || (x >= 4 && y >= 4) {
                    f32::NAN
                } else {
                    (r - 3.5) as f32
                });
            }
        }
        write_nrrd(
            &Nrrd {
                sizes: vec![11, 11],
                spacing: vec![1.0, 1.0],
                origin: vec![-5.0, -5.0],
                values,
            },
            Encoding::Ascii,
        )
        .unwrap()
    }

    #[test]
    fn unobserved_samples_follow_the_policy() {
        let config = VolumeImportConfig {
            band: 1.5,
            ..VolumeImportConfig::default()
        };
        let volume = import(&ring_with_holes(), &config).unwrap();
        assert_eq!(volume.dimensions, 2);
        assert_eq!((volume.unobserved_solid, volume.unobserved_empty), (9, 4));
        assert!(volume.edge_axes.is_empty());
        let field = PayloadView::new(&volume.payload).unwrap();
        assert!(
            close(field.sample(&[0.0, 0.0]), -1.5),
            "enclosed pocket is solid"
        );
        assert!(
            close(field.sample(&[4.5, 4.5]), 1.5),
            "open corner is empty"
        );

        let solid = import(
            &ring_with_holes(),
            &VolumeImportConfig {
                unobserved: Unobserved::Solid,
                ..config.clone()
            },
        )
        .unwrap();
        assert_eq!((solid.unobserved_solid, solid.unobserved_empty), (13, 0));
        assert_eq!(solid.edge_axes, vec![0, 1]);
        let field = PayloadView::new(&solid.payload).unwrap();
        assert!(close(field.sample(&[4.5, 4.5]), -1.5));

        let empty = import(
            &ring_with_holes(),
            &VolumeImportConfig {
                unobserved: Unobserved::Empty,
                ..config
            },
        )
        .unwrap();
        assert_eq!((empty.unobserved_solid, empty.unobserved_empty), (0, 13));
        let field = PayloadView::new(&empty.payload).unwrap();
        assert!(close(field.sample(&[0.0, 0.0]), 1.5), "hollow");
        assert!(
            close(field.sample(&[2.5, 0.0]), -1.0),
            "the ring is still solid"
        );
    }

    #[test]
    fn threshold_and_inside_pick_a_density_isosurface() {
        let values: Vec<f32> = (-10i32..=10).map(|p| (100 - 10 * p.abs()) as f32).collect();
        let bytes = write_nrrd(
            &Nrrd {
                sizes: vec![21],
                spacing: vec![1.0],
                origin: vec![-10.0],
                values,
            },
            Encoding::Raw,
        )
        .unwrap();
        let config = VolumeImportConfig {
            threshold: 50.0,
            inside: Inside::Above,
            ..VolumeImportConfig::default()
        };
        let volume = import(&bytes, &config).unwrap();
        assert_eq!(volume.dimensions, 1);
        assert_eq!(volume.band, 50.0, "band defaults to the largest magnitude");
        assert_eq!(volume.counts, vec![21]);
        assert_eq!(volume.bounds, vec![-10.0, 10.0]);
        let field = PayloadView::new(&volume.payload).unwrap();
        assert!(close(field.sample(&[0.0]), -50.0));
        assert!(close(field.sample(&[5.0]), 0.0));
        assert!(close(field.sample(&[7.5]), 25.0));
    }

    #[test]
    fn bad_inputs_are_named() {
        let err = import(b"ply\n", &VolumeImportConfig::default()).unwrap_err();
        assert!(err.contains("NRRD"), "{err}");
        let err = import(
            &sphere([0.0; 3], -16.0),
            &VolumeImportConfig {
                stride: 0,
                ..VolumeImportConfig::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("stride"), "{err}");
        let err = import(
            &sphere([0.0; 3], -16.0),
            &VolumeImportConfig {
                threshold: -10.0,
                ..VolumeImportConfig::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("no sample is inside"), "{err}");
        let err = import(
            &sphere([0.0; 3], -16.0),
            &VolumeImportConfig {
                scale: 0.0,
                ..VolumeImportConfig::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("scale"), "{err}");
        let err = import(
            &sphere([0.0; 3], -16.0),
            &VolumeImportConfig {
                stride: 40,
                ..VolumeImportConfig::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("single sample"), "{err}");
    }
}
