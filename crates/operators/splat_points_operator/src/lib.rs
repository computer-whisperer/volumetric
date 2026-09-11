#![doc = include_str!("../README.md")]

use volumetric_abi::fea::{COLOR_FIELD_NAME, FeaElementKind, FeaField, FeaMesh, NORMAL_FIELD_NAME};
use volumetric_abi::splat::Splat;
use volumetric_abi::viewset::ViewSet;
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput,
    fea::encode_fea_mesh,
    host::{post_output, post_warning, read_input, report_error},
    splat::decode_splat,
    viewset::decode_viewset,
};

/// Name of the per-point opacity field.
pub const OPACITY_FIELD_NAME: &str = "opacity";
/// Name of the per-point largest-scale field, metres.
pub const SCALE_FIELD_NAME: &str = "scale";

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct SplatPointsConfig {
    pub min_opacity: f64,
    pub max_scale: f64,
    pub min_scale: f64,
    pub stride: u32,
    pub bounds: Option<Bounds>,
    pub near: Option<Near>,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl Bounds {
    fn corners(&self) -> ([f64; 3], [f64; 3]) {
        (
            [self.min_x, self.min_y, self.min_z],
            [self.max_x, self.max_y, self.max_z],
        )
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct Near {
    /// A solved marker id in the view set; empty to use the point.
    pub marker: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub radius: f64,
}

impl Default for Near {
    fn default() -> Self {
        Self {
            marker: String::new(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            radius: 0.5,
        }
    }
}

impl Default for SplatPointsConfig {
    fn default() -> Self {
        Self {
            min_opacity: 0.5,
            max_scale: 0.0,
            min_scale: 0.0,
            stride: 1,
            bounds: None,
            near: None,
        }
    }
}

/// What the filters kept.
#[derive(Clone, Debug, PartialEq)]
pub struct PointsReport {
    pub total: usize,
    pub kept: usize,
}

/// The centre of `near`: the named marker's centre in the view set, or
/// the point.
fn near_centre(near: &Near, views: Option<&ViewSet>) -> Result<[f64; 3], String> {
    if near.marker.is_empty() {
        return Ok([near.x, near.y, near.z]);
    }
    let id: u32 = near
        .marker
        .trim()
        .parse()
        .map_err(|_| format!("near.marker {:?} is not a marker id", near.marker))?;
    let set = views.ok_or_else(|| {
        format!("near.marker {id} needs the view set holding the solved markers on input 2")
    })?;
    let marker = set
        .markers
        .iter()
        .find(|m| m.id == id)
        .ok_or_else(|| format!("the view set has no solved marker {id}"))?;
    let mut c = [0.0; 3];
    for corner in &marker.corners {
        for k in 0..3 {
            c[k] += corner[k] / 4.0;
        }
    }
    Ok(c)
}

/// The cloud of the primitives that pass the filters.
pub fn points(
    splat: &Splat,
    views: Option<&ViewSet>,
    config: &SplatPointsConfig,
) -> Result<(FeaMesh, PointsReport), String> {
    if config.stride == 0 {
        return Err("stride must be at least 1".to_string());
    }
    if !(config.min_opacity.is_finite() && (0.0..=1.0).contains(&config.min_opacity)) {
        return Err(format!(
            "min_opacity must be within [0, 1], got {}",
            config.min_opacity
        ));
    }
    if config.max_scale < 0.0 || config.min_scale < 0.0 {
        return Err("max_scale and min_scale must not be negative".to_string());
    }
    let bounds = config.bounds.as_ref().map(Bounds::corners);
    let near = match &config.near {
        Some(near) => {
            if near.radius.is_nan() || near.radius <= 0.0 {
                return Err(format!("near.radius must be positive, got {}", near.radius));
            }
            Some((near_centre(near, views)?, near.radius * near.radius))
        }
        None => None,
    };

    let n = splat.len();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut opacities = Vec::new();
    let mut scales = Vec::new();
    let mut passed = 0usize;
    for i in 0..n {
        let opacity = f64::from(splat.opacity(i));
        if opacity < config.min_opacity {
            continue;
        }
        let extent = f64::from(splat.extent(i));
        if (config.max_scale > 0.0 && extent > config.max_scale) || extent < config.min_scale {
            continue;
        }
        let m = splat.mean(i);
        let p = [f64::from(m[0]), f64::from(m[1]), f64::from(m[2])];
        if let Some((lo, hi)) = bounds {
            if (0..3).any(|k| p[k] < lo[k] || p[k] > hi[k]) {
                continue;
            }
        }
        if let Some((c, r2)) = near {
            let d2: f64 = (0..3).map(|k| (p[k] - c[k]).powi(2)).sum();
            if d2 > r2 {
                continue;
            }
        }
        passed += 1;
        if (passed - 1) % config.stride as usize != 0 {
            continue;
        }
        positions.extend(p);
        normals.extend(splat.normal(i).map(f64::from));
        colors.extend(splat.color(i).map(f64::from));
        opacities.push(opacity);
        scales.push(extent);
    }
    let kept = opacities.len();
    if kept == 0 {
        return Err(format!(
            "no primitive of {n} passes the filters (min_opacity {}, max_scale {}, min_scale {}{}{})",
            config.min_opacity,
            config.max_scale,
            config.min_scale,
            if bounds.is_some() { ", bounds" } else { "" },
            if near.is_some() { ", near" } else { "" },
        ));
    }
    let field = |name: &str, components: usize, data: Vec<f64>| FeaField {
        name: name.to_string(),
        components,
        data,
    };
    let mesh = FeaMesh {
        element_kind: FeaElementKind::Point1,
        node_positions: positions,
        connectivity: (0..kept as u32).collect(),
        node_fields: vec![
            field(NORMAL_FIELD_NAME, 3, normals),
            field(COLOR_FIELD_NAME, 3, colors),
            field(OPACITY_FIELD_NAME, 1, opacities),
            field(SCALE_FIELD_NAME, 1, scales),
        ],
        element_fields: vec![],
    };
    mesh.validate()?;
    Ok((mesh, PointsReport { total: n, kept }))
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let splat = match decode_splat(&read_input(0)) {
        Ok(splat) => splat,
        Err(e) => {
            report_error(&format!("input 0 is not a splat: {e}"));
            return;
        }
    };
    let views_bytes = read_input(1);
    let views = if views_bytes.is_empty() {
        None
    } else {
        match decode_viewset(&views_bytes) {
            Ok(set) => Some(set),
            Err(e) => {
                report_error(&format!("input 1 is not a view set: {e}"));
                return;
            }
        }
    };
    let config = {
        let cfg = read_input(2);
        if cfg.is_empty() {
            SplatPointsConfig::default()
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
    match points(&splat, views.as_ref(), &config) {
        Ok((mesh, report)) => {
            if report.kept < report.total {
                post_warning(&format!(
                    "{} of {} primitives kept",
                    report.kept, report.total
                ));
            }
            post_output(0, &encode_fea_mesh(&mesh));
        }
        Err(e) => report_error(&format!("splat points failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ min_opacity: float .default 0.5, max_scale: float .default 0.0, min_scale: float .default 0.0, stride: int .ge 1 .default 1, ? bounds: { min_x: float .default 0.0, min_y: float .default 0.0, min_z: float .default 0.0, max_x: float .default 0.0, max_y: float .default 0.0, max_z: float .default 0.0 }, ? near: { marker: tstr .default "", x: float .default 0.0, y: float .default 0.0, z: float .default 0.0, radius: float .default 0.5 } }"#
            .to_string();
        OperatorMetadata {
            name: "splat_points_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Splat Points".to_string(),
            description:
                "The centres of a splat's opaque primitives as a cloud with normals and colours."
                    .to_string(),
            category: "Mesh".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<ellipse cx="8" cy="8" rx="4" ry="2.5" transform="rotate(-30 8 8)"/>"##,
                r##"<circle cx="8" cy="8" r="0.8"/>"##,
                r##"<circle cx="15" cy="7" r="1.2"/>"##,
                r##"<circle cx="18" cy="13" r="1.2"/>"##,
                r##"<circle cx="12" cy="16" r="1.2"/>"##,
                r##"<circle cx="6" cy="17" r="1.2"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Splat,
                OperatorMetadataInput::ViewSet,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec![
                "Splat".to_string(),
                "Views".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
            output_names: vec!["Points".to_string()],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::splat::{SplatKind, sh_rest_per_point};
    use volumetric_abi::viewset::{Marker, Provenance, WorldFrame};

    /// Four Gaussians along x at 0, 1, 2, 3 with opacities 0.1, 0.9,
    /// 0.9, 0.9 and extents 1 mm, 1 mm, 10 cm, 1 mm.
    fn splat() -> Splat {
        let mut s = Splat::empty(SplatKind::Gaussian3d, 0);
        s.count = 4;
        s.means = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let e = |m: f32| m.ln();
        s.scales = vec![
            e(1e-3),
            e(1e-3),
            e(1e-4),
            e(1e-3),
            e(1e-4),
            e(1e-3),
            e(0.1),
            e(1e-3),
            e(1e-3),
            e(1e-3),
            e(1e-3),
            e(1e-4),
        ];
        s.quats = [1.0, 0.0, 0.0, 0.0].repeat(4);
        let logit = |p: f32| (p / (1.0 - p)).ln();
        s.opacities = vec![logit(0.1), logit(0.9), logit(0.9), logit(0.9)];
        s.sh0 = vec![0.0; 12];
        s.sh_rest = vec![0.0; 4 * sh_rest_per_point(0)];
        s.validate().unwrap();
        s
    }

    #[test]
    fn filters_by_opacity_scale_bounds_and_stride() {
        let (mesh, report) = points(&splat(), None, &SplatPointsConfig::default()).unwrap();
        assert_eq!((report.total, report.kept), (4, 3));
        assert_eq!(
            mesh.node_positions,
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]
        );
        // Normals follow the smallest scale: y for the second, z for the last.
        let normal = &mesh.node_fields[0];
        assert_eq!(normal.name, NORMAL_FIELD_NAME);
        assert_eq!(&normal.data[0..3], &[0.0, 1.0, 0.0]);
        assert_eq!(&normal.data[6..9], &[0.0, 0.0, 1.0]);
        assert_eq!(mesh.node_fields[1].name, COLOR_FIELD_NAME);
        assert_eq!(&mesh.node_fields[1].data[0..3], &[0.5, 0.5, 0.5]);
        assert!((mesh.node_fields[2].data[0] - 0.9).abs() < 1e-5);
        assert!((mesh.node_fields[3].data[1] - 0.1).abs() < 1e-6);

        let config = SplatPointsConfig {
            max_scale: 0.01,
            ..Default::default()
        };
        let (mesh, _) = points(&splat(), None, &config).unwrap();
        assert_eq!(mesh.node_positions, vec![1.0, 0.0, 0.0, 3.0, 0.0, 0.0]);

        let config = SplatPointsConfig {
            min_opacity: 0.0,
            stride: 2,
            ..Default::default()
        };
        let (mesh, _) = points(&splat(), None, &config).unwrap();
        assert_eq!(mesh.node_positions, vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);

        let config = SplatPointsConfig {
            bounds: Some(Bounds {
                min_x: 1.5,
                min_y: -1.0,
                min_z: -1.0,
                max_x: 2.5,
                max_y: 1.0,
                max_z: 1.0,
            }),
            ..Default::default()
        };
        let (mesh, _) = points(&splat(), None, &config).unwrap();
        assert_eq!(mesh.node_positions, vec![2.0, 0.0, 0.0]);
    }

    #[test]
    fn near_a_point_or_a_solved_marker() {
        let config = SplatPointsConfig {
            near: Some(Near {
                x: 3.2,
                radius: 0.5,
                ..Default::default()
            }),
            ..Default::default()
        };
        let (mesh, _) = points(&splat(), None, &config).unwrap();
        assert_eq!(mesh.node_positions, vec![3.0, 0.0, 0.0]);

        let mut set = ViewSet {
            schema: volumetric_abi::viewset::VIEWSET_SCHEMA,
            world: WorldFrame::default(),
            provenance: Provenance::default(),
            cameras: vec![],
            views: vec![],
            markers: vec![],
            board: None,
        };
        set.markers.push(Marker {
            id: 7,
            size_m: 0.2,
            corners: [
                [0.9, -0.1, 0.0],
                [1.1, -0.1, 0.0],
                [1.1, 0.1, 0.0],
                [0.9, 0.1, 0.0],
            ],
        });
        let config = SplatPointsConfig {
            near: Some(Near {
                marker: "7".to_string(),
                radius: 0.3,
                ..Default::default()
            }),
            ..Default::default()
        };
        let (mesh, _) = points(&splat(), Some(&set), &config).unwrap();
        assert_eq!(mesh.node_positions, vec![1.0, 0.0, 0.0]);
        let err = points(&splat(), None, &config).unwrap_err();
        assert!(err.contains("needs the view set"), "{err}");
        let config = SplatPointsConfig {
            near: Some(Near {
                marker: "9".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(
            points(&splat(), Some(&set), &config)
                .unwrap_err()
                .contains("no solved marker 9")
        );
    }

    #[test]
    fn refuses_empty_results_and_bad_config() {
        let config = SplatPointsConfig {
            min_opacity: 0.95,
            ..Default::default()
        };
        assert!(
            points(&splat(), None, &config)
                .unwrap_err()
                .contains("no primitive")
        );
        let config = SplatPointsConfig {
            stride: 0,
            ..Default::default()
        };
        assert!(
            points(&splat(), None, &config)
                .unwrap_err()
                .contains("stride")
        );
        let config = SplatPointsConfig {
            min_opacity: 1.5,
            ..Default::default()
        };
        assert!(
            points(&splat(), None, &config)
                .unwrap_err()
                .contains("min_opacity")
        );
    }
}
